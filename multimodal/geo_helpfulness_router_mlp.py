"""Immutable MLP exploration, with lazy fitting and separate apply-only scoring."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal import geo_helpfulness_router_capacity as capacity
from multimodal.geo_helpfulness_router_candidate_state import TARGET_ORDER, VARIANTS
from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    canonical_hash,
    check_self_hash,
    check_snapshot,
    compare_tables,
    exclusive_lock,
    file_hash,
    read_json,
    table_bytes,
    write_bytes,
    write_json,
)
from multimodal.geo_helpfulness_router_mlp_state import (
    predict_state,
    resolve_parameters,
)
from multimodal.geo_helpfulness_router_metrics import multiclass_metrics

STUDY_ID = "router_mlp_v1"
IDENTITY, FEATURES, Q_COLUMNS = capacity.IDENTITY, capacity.FEATURES, capacity.Q_COLUMNS
ARCHITECTURES = ((32,), (64,), (32, 16), (64, 32))
SCORE_FILES = capacity.SCORE_FILES | {"paired_comparisons.json"}
CODE_FILES = (
    "multimodal/geo_helpfulness_router_mlp.py",
    "multimodal/geo_helpfulness_router_mlp_fit.py",
    "multimodal/geo_helpfulness_router_mlp_state.py",
    "multimodal/geo_helpfulness_router_mlp_score.py",
    "tools/run_multimodal_geo_helpfulness_router_mlp.py",
)


def _config(config):
    if (
        config.get("study_id") != STUDY_ID
        or config.get("schema_version") != "geo_helpfulness.router_mlp_config.v1"
    ):
        raise ValueError("MLP configuration identity mismatch")
    if (
        config.get("training_seeds") != [1, 2, 3, 4]
        or config.get("target_order") != list(TARGET_ORDER)
        or config.get("configuration_count") != 12
        or config.get("candidate_seed_count") != 48
    ):
        raise ValueError("MLP seed/target/count declaration mismatch")
    records = config.get("configurations", [])
    expected = [(variant, widths) for variant in VARIANTS for widths in ARCHITECTURES]
    if (
        len(records) != 12
        or [
            (s.get("variant"), tuple(s.get("parameters", {}).get("hidden_widths", [])))
            for s in records
        ]
        != expected
    ):
        raise ValueError("MLP architecture/variant declaration order differs")
    names = set()
    for spec in records:
        if (
            spec.get("family") != "mlp"
            or not re.fullmatch(r"[a-zA-Z0-9_]+", spec.get("config_id", ""))
            or spec["config_id"] in names
        ):
            raise ValueError("MLP configuration identifier invalid")
        names.add(spec["config_id"])
        if resolve_parameters(spec) != spec["parameters"]:
            raise ValueError(
                "MLP declaration must explicitly include every training setting"
            )
    return records


def _prediction_files(config):
    return capacity._prediction_files(config) | {"training_curves.parquet"}


def load_declaration(output_root):
    root = Path(output_root)
    declaration = read_json(root / "declaration.json")
    check_self_hash(declaration, "declaration_sha256")
    if (
        declaration.get("study_id") != STUDY_ID
        or declaration.get("schema_version")
        != "geo_helpfulness.router_mlp_declaration.v1"
    ):
        raise ValueError("MLP declaration identity mismatch")
    _config(declaration["configuration"])
    check_snapshot(declaration["parent_file_sha256"])
    capacity._allowlist(
        root,
        {
            "declaration.json",
            "predictions/manifest.json",
            "scores/manifest.json",
            *[
                "predictions/" + name
                for name in _prediction_files(declaration["configuration"])
            ],
            *["scores/" + name for name in SCORE_FILES],
        },
    )
    return declaration


def _manifest(root, declaration, kind):
    manifest = read_json(root / "manifest.json")
    check_self_hash(manifest)
    if (
        manifest.get("schema_version") != f"geo_helpfulness.router_mlp_{kind}.v1"
        or manifest.get("declaration_sha256") != declaration["declaration_sha256"]
    ):
        raise ValueError("MLP manifest identity mismatch")
    capacity._allowlist(root, {*manifest["artifacts"], "manifest.json"}, complete=True)
    for name, record in manifest["artifacts"].items():
        if file_hash(root / name) != record["file_sha256"]:
            raise ValueError(f"MLP artifact changed: {name}")
    return manifest


def _publish(root, declaration, kind, names, extra):
    check_snapshot(declaration["parent_file_sha256"])
    capacity._allowlist(root, names, complete=True)
    artifacts = {}
    for name in sorted(names):
        path = root / name
        record = {"file_sha256": file_hash(path), "bytes": path.stat().st_size}
        if name.endswith(".parquet"):
            parquet = pq.ParquetFile(path)
            record.update(
                rows=parquet.metadata.num_rows,
                columns=len(parquet.schema_arrow),
                arrow_schema=str(parquet.schema_arrow),
            )
        artifacts[name] = record
    manifest = capacity._seal(
        {
            "schema_version": f"geo_helpfulness.router_mlp_{kind}.v1",
            "study_id": STUDY_ID,
            "declaration_sha256": declaration["declaration_sha256"],
            "artifacts": artifacts,
            **extra,
        },
        "manifest_sha256",
    )
    write_json(root / "manifest.json", manifest)
    return manifest


def _capacity_parent(root):
    declaration = capacity.load_declaration(root)
    prediction = capacity._manifest(
        root / "predictions", declaration, "prediction_manifest"
    )
    score = capacity._manifest(root / "scores", declaration, "score_manifest")
    if score["prediction_manifest_sha256"] != prediction["manifest_sha256"]:
        raise ValueError("capacity predecessor prediction/score linkage differs")
    return declaration, prediction, score


def _declare(protocol_root, output_root, config_path, capacity_root, inputs):
    import torch

    config = read_json(config_path)
    _config(config)
    predecessor, prediction, score = _capacity_parent(capacity_root)
    if Path(predecessor["protocol_root"]).resolve() != protocol_root:
        raise ValueError("MLP/capacity protocol lineage differs")
    snapshot = dict(predecessor["parent_file_sha256"])
    snapshot.update(inputs["parent_file_hashes"])
    snapshot.update(
        {
            str(capacity_root / name): file_hash(capacity_root / name)
            for name in capacity._files(capacity_root)
        }
    )
    snapshot.update(
        {
            str(PROJECT_ROOT / name): file_hash(PROJECT_ROOT / name)
            for name in CODE_FILES
        }
    )
    snapshot[str(config_path)] = file_hash(config_path)
    torch_root = Path(torch.__file__).resolve().parent
    snapshot.update(
        {
            str(torch_root / name): file_hash(torch_root / name)
            for name in (
                "__init__.py",
                "optim/adamw.py",
                "nn/modules/linear.py",
                "nn/modules/dropout.py",
                "lib/libtorch_cpu.so",
            )
        }
    )
    environment = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "pyarrow", "torch")
    }
    environment.update(
        python=sys.version,
        torch_build_configuration=torch.__config__.show(),
        device="cpu",
        compute_dtype="float64",
        threads=1,
    )
    declaration = capacity._seal(
        {
            "schema_version": "geo_helpfulness.router_mlp_declaration.v1",
            "study_id": STUDY_ID,
            "declared_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": config,
            "protocol_root": str(protocol_root),
            "capacity_root": str(capacity_root),
            "capacity_prediction_manifest_sha256": prediction["manifest_sha256"],
            "capacity_score_manifest_sha256": score["manifest_sha256"],
            "original_m4_manifest_sha256": predecessor["original_m4_manifest_sha256"],
            "parent_file_sha256": snapshot,
            "environment": environment,
            "validation_labels_during_fitting": False,
            "cleaned_test_accessed": False,
            "legacy_transfer_outputs_accessed": False,
            "replay_atol": 1e-12,
            "replay_rtol": 1e-12,
            "probability_normalization_atol": 1e-8,
            "recovery_rule": "reuse_valid_declared_self_hashed_models_only_no_deletions",
        },
        "declaration_sha256",
    )
    check_snapshot(snapshot)
    write_json(output_root / "declaration.json", declaration)
    return declaration


def _record(path, declaration, spec, seed):
    record = read_json(path)
    check_self_hash(record, "record_sha256")
    state = record["state"]
    if (
        record.get("declaration_sha256") != declaration["declaration_sha256"]
        or any(
            state.get(key) != spec[key]
            for key in ("config_id", "family", "variant", "parameters")
        )
        or state.get("training_seed") != seed
        or state.get("fit_row_count") != 3378
    ):
        raise ValueError("MLP model record identity mismatch")
    history = record["diagnostics"]["history"]
    if len(history) != 300 or [r["epoch"] for r in history] != list(range(1, 301)):
        raise ValueError("MLP training history must contain the fixed 300 epochs")
    for row in history:
        if (
            row.get("gradients_finite") is not True
            or any(
                not np.isfinite(row[name]) or row[name] < 0
                for name in (
                    "optimization_cross_entropy",
                    "train_nll",
                    "train_accuracy",
                    "maximum_gradient_l2_norm",
                )
            )
            or row["train_accuracy"] > 1
        ):
            raise ValueError("MLP training diagnostic is invalid")
    return record


def _curves(records):
    rows = []
    for record in records:
        state = record["state"]
        for row in record["diagnostics"]["history"]:
            rows.append(
                {
                    "config_id": state["config_id"],
                    "variant": state["variant"],
                    "training_seed": state["training_seed"],
                    **row,
                }
            )
    return capacity._table(
        pd.DataFrame(rows)
        .sort_values(["config_id", "training_seed", "epoch"])
        .reset_index(drop=True)
    )


def _validate_predictions(output_root):
    declaration = load_declaration(output_root)
    root = Path(output_root) / "predictions"
    manifest = _manifest(root, declaration, "prediction_manifest")
    if set(manifest["artifacts"]) != _prediction_files(declaration["configuration"]):
        raise ValueError("MLP prediction cross-product incomplete")
    inputs = capacity._replay_inputs(declaration, output_root)
    targets = pq.read_table(
        Path(declaration["protocol_root"]) / "router/router_dataset.parquet",
        columns=[*IDENTITY, "target_state"],
    ).to_pandas()
    records = []
    for spec in declaration["configuration"]["configurations"]:
        for seed in range(1, 5):
            stem, data = capacity._stem(spec, seed), inputs[seed]
            record = _record(root / f"models/{stem}.json", declaration, spec, seed)
            records.append(record)
            for role in ("train", "validation"):
                q = predict_state(data[f"X_{role}"], record["state"])
                compare_tables(
                    pq.read_table(root / f"{role}/{stem}.parquet"),
                    capacity._prediction_table(data[f"{role}_identity"], q, spec),
                )
                if role == "train":
                    label_rows = (
                        targets.loc[targets.training_seed.eq(seed)]
                        .sort_values("row_uid")
                        .reset_index(drop=True)
                    )
                    if not label_rows[IDENTITY].equals(data["train_identity"]):
                        raise ValueError("MLP train diagnostics label join differs")
                    y = label_rows.target_state.map(
                        {name: i for i, name in enumerate(TARGET_ORDER)}
                    ).to_numpy(dtype=np.int64)
                    metric = multiclass_metrics(q, y)
                    last = record["diagnostics"]["history"][-1]
                    if not np.isclose(
                        metric["log_loss"],
                        last["train_nll"],
                        atol=1e-12,
                        rtol=1e-12,
                    ) or not np.isclose(
                        metric["top1_accuracy"],
                        last["train_accuracy"],
                        atol=1e-12,
                        rtol=0,
                    ):
                        raise ValueError(
                            "MLP final training diagnostics do not reproduce"
                        )
    compare_tables(pq.read_table(root / "training_curves.parquet"), _curves(records))
    check_snapshot(declaration["parent_file_sha256"])
    return declaration, manifest


def validate_predictions(output_root):
    with exclusive_lock(Path(output_root)):
        _, manifest = _validate_predictions(output_root)
    return {
        "status": "validated",
        "candidate_seed_count": 48,
        "manifest_sha256": manifest["manifest_sha256"],
        "fitting_performed": False,
    }


def build_predictions(protocol_root, output_root, config_path, capacity_root):
    protocol_root, output_root, config_path, capacity_root = [
        Path(p).resolve()
        for p in (protocol_root, output_root, config_path, capacity_root)
    ]
    if any(
        output_root == root or root in output_root.parents
        for root in (protocol_root, capacity_root)
    ):
        raise ValueError("MLP outputs must be outside frozen protocol/capacity roots")
    with exclusive_lock(output_root):
        if (output_root / "declaration.json").exists():
            declaration = load_declaration(output_root)
            if (
                declaration["protocol_root"] != str(protocol_root)
                or declaration["capacity_root"] != str(capacity_root)
                or declaration["configuration"] != read_json(config_path)
            ):
                raise ValueError("MLP repeat build paths/configuration differ")
            if (output_root / "predictions/manifest.json").exists():
                _, manifest = _validate_predictions(output_root)
                return {
                    "status": "reused_valid",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "new_fits": 0,
                }
        else:
            if capacity._files(output_root):
                raise ValueError("unowned MLP output directory is not empty")
            declaration = None
        from multimodal.geo_helpfulness_router_capacity_inputs import (
            load_capacity_inputs,
        )
        from multimodal.geo_helpfulness_router_mlp_fit import fit_candidate

        inputs = load_capacity_inputs(PROJECT_ROOT, protocol_root)
        if declaration is None:
            declaration = _declare(
                protocol_root, output_root, config_path, capacity_root, inputs
            )
        check_snapshot(declaration["parent_file_sha256"])
        root = output_root / "predictions"
        expected = _prediction_files(declaration["configuration"])
        capacity._allowlist(root, expected)
        write_bytes(
            root / "validation_features.parquet",
            table_bytes(capacity._table(inputs["validation"])),
        )
        records, new_fits = [], 0
        for index, spec in enumerate(declaration["configuration"]["configurations"]):
            for seed in range(1, 5):
                data, stem = inputs["seed_inputs"][seed], capacity._stem(spec, seed)
                path = root / f"models/{stem}.json"
                if path.exists():
                    record = _record(path, declaration, spec, seed)
                    train_q, val_q = (
                        predict_state(data[f"X_{role}"], record["state"])
                        for role in ("train", "validation")
                    )
                else:
                    state, train_q, val_q, diagnostics = fit_candidate(
                        data["X_train"],
                        data["y_train"],
                        data["X_validation"],
                        spec,
                        seed,
                    )
                    record = capacity._seal(
                        {
                            "declaration_sha256": declaration["declaration_sha256"],
                            "state": state,
                            "diagnostics": diagnostics,
                        },
                        "record_sha256",
                    )
                    write_json(path, record)
                    _record(path, declaration, spec, seed)
                    new_fits += 1
                records.append(record)
                for role, q in (("train", train_q), ("validation", val_q)):
                    write_bytes(
                        root / f"{role}/{stem}.parquet",
                        table_bytes(
                            capacity._prediction_table(
                                data[f"{role}_identity"], q, spec
                            )
                        ),
                    )
                print(
                    f"MLP {index * 4 + seed}/48 {stem} sealed at epoch300",
                    file=sys.stderr,
                    flush=True,
                )
        write_bytes(root / "training_curves.parquet", table_bytes(_curves(records)))
        manifest = _publish(
            root,
            declaration,
            "prediction_manifest",
            expected,
            {
                "candidate_seed_count": 48,
                "training_prediction_rows": 162144,
                "validation_prediction_rows": 39456,
                "training_curve_rows": 14400,
                "validation_labels_accessed": False,
                "cleaned_test_accessed": False,
            },
        )
        return {
            "status": "built",
            "new_fits": new_fits,
            "candidate_seed_count": 48,
            "manifest_sha256": manifest["manifest_sha256"],
        }


def _fit_incapable():
    capacity._fit_incapable()
    if "multimodal.geo_helpfulness_router_mlp_fit" in sys.modules:
        raise ValueError("MLP scoring requires a fresh fitting-incapable process")


def _score_payloads(output_root, declaration, prediction_manifest):
    _fit_incapable()
    from multimodal.geo_helpfulness_router_mlp_score import evaluate_mlp

    root = Path(output_root)
    router = Path(declaration["protocol_root"]) / "router"
    target_path = router / "m4_feasibility_v1/validation_targets.parquet"
    targets = pq.read_table(target_path).to_pandas()
    experts = pq.read_table(
        router / "m4_comparator_inputs_v1/expert_probabilities_and_features.parquet"
    ).to_pandas()
    baseline = pq.read_table(
        Path(declaration["capacity_root"]) / "scores/actions.parquet"
    ).to_pandas()
    heuristics = read_json(router / "m4_feasibility_v1/oof_heuristics.json")[
        "heuristics"
    ]
    frames, training_metrics = [], {}
    training = pq.read_table(
        router / "router_dataset.parquet", columns=[*IDENTITY, "target_state"]
    ).to_pandas()
    for spec in declaration["configuration"]["configurations"]:
        per_seed = {}
        for seed in range(1, 5):
            stem = capacity._stem(spec, seed)
            frames.append(
                pq.read_table(
                    root / f"predictions/validation/{stem}.parquet"
                ).to_pandas()
            )
            train = pq.read_table(
                root / f"predictions/train/{stem}.parquet"
            ).to_pandas()
            label_rows = (
                training.loc[training.training_seed.eq(seed)]
                .sort_values("row_uid")
                .reset_index(drop=True)
            )
            if not label_rows[IDENTITY].equals(train[IDENTITY]):
                raise ValueError("MLP training metric label join differs")
            y = label_rows.target_state.map(
                {name: i for i, name in enumerate(TARGET_ORDER)}
            ).to_numpy(dtype=np.int64)
            per_seed[str(seed)] = multiclass_metrics(train[Q_COLUMNS].to_numpy(), y)
        training_metrics[spec["config_id"]] = {
            "per_seed": per_seed,
            "interpretation": "resubstitution_diagnostic_not_independent_evaluation",
        }
    payloads = evaluate_mlp(
        pd.concat(frames, ignore_index=True),
        experts,
        targets,
        declaration["configuration"]["configurations"],
        baseline,
        heuristics,
    )
    frozen = {}
    for group, selection in payloads["selection.json"]["groups"].items():
        config_id = selection["selected_config_id"]
        frozen[group] = {
            "config_id": config_id,
            "threshold_specification": selection["threshold"][
                "threshold_specification"
            ],
            "models": {
                str(seed): {
                    "path": f"../predictions/models/{config_id}__seed_{seed}.json",
                    "file_sha256": file_hash(
                        root / f"predictions/models/{config_id}__seed_{seed}.json"
                    ),
                }
                for seed in range(1, 5)
            },
        }
    payloads["frozen_policies.json"] = {
        "study_id": STUDY_ID,
        "groups": frozen,
        "declaration_sha256": declaration["declaration_sha256"],
        "further_selection_or_refitting_permitted": False,
    }
    payloads["training_metrics.json"] = training_metrics
    payloads["access_audit.json"] = {
        "study_id": STUDY_ID,
        "all_predictions_sealed_before_validation_labels": True,
        "fitting_imports_present": False,
        "prediction_manifest_sha256": prediction_manifest["manifest_sha256"],
        "validation_target_source": str(target_path),
        "cleaned_test_accessed": False,
        "legacy_transfer_outputs_accessed": False,
        "original_m4_no_go_preserved": True,
        "reused_validation_conditional": True,
    }
    payloads["selection_receipt.json"] = capacity._seal(
        {
            "study_id": STUDY_ID,
            "declaration_sha256": declaration["declaration_sha256"],
            "prediction_manifest_sha256": prediction_manifest["manifest_sha256"],
            "capacity_score_manifest_sha256": declaration[
                "capacity_score_manifest_sha256"
            ],
            "validation_target_file_sha256": file_hash(target_path),
            "selection_sha256": canonical_hash(payloads["selection.json"]),
            "frozen_policies_sha256": canonical_hash(payloads["frozen_policies.json"]),
            "post_receipt_reselection_permitted": False,
        },
        "receipt_sha256",
    )
    _fit_incapable()
    if set(payloads) != SCORE_FILES:
        raise ValueError("MLP scoring output allowlist differs")
    return payloads


def _scores(output_root, *, build):
    _fit_incapable()
    output_root = Path(output_root)
    with exclusive_lock(output_root):
        declaration, prediction = _validate_predictions(output_root)
        root = output_root / "scores"
        committed = (root / "manifest.json").exists()
        if committed:
            manifest = _manifest(root, declaration, "score_manifest")
            if (
                set(manifest["artifacts"]) != SCORE_FILES
                or manifest["prediction_manifest_sha256"]
                != prediction["manifest_sha256"]
            ):
                raise ValueError("MLP score manifest linkage differs")
        elif not build:
            raise ValueError("MLP score bundle not committed")
        else:
            capacity._allowlist(root, SCORE_FILES)
        payloads = _score_payloads(output_root, declaration, prediction)
        for name, value in payloads.items():
            path = root / name
            if path.exists():
                if isinstance(value, pa.Table):
                    compare_tables(pq.read_table(path), value)
                elif read_json(path) != value:
                    raise ValueError(f"MLP score reconstruction differs: {name}")
            elif build:
                if isinstance(value, pa.Table):
                    write_bytes(path, table_bytes(value))
                else:
                    write_json(path, value)
        if not committed:
            manifest = _publish(
                root,
                declaration,
                "score_manifest",
                SCORE_FILES,
                {
                    "prediction_manifest_sha256": prediction["manifest_sha256"],
                    "selection_receipt_sha256": payloads["selection_receipt.json"][
                        "receipt_sha256"
                    ],
                },
            )
        check_snapshot(declaration["parent_file_sha256"])
        return {
            "status": (
                ("reused_valid" if committed else "built") if build else "validated"
            ),
            "manifest_sha256": manifest["manifest_sha256"],
            "group_count": 3,
            "fitting_performed": False,
            "cleaned_test_accessed": False,
        }


def build_scores(output_root):
    return _scores(output_root, build=True)


def validate_scores(output_root):
    return _scores(output_root, build=False)

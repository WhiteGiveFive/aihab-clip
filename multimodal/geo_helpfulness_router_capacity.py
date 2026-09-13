"""Immutable, development-only orchestration of the router capacity experiment.

Fitting imports are lazy. Validation and scoring replay portable model states
without importing sklearn, XGBoost, Torch, or any numerical fitting module.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal.geo_helpfulness_router_candidate_state import (
    CATEGORICAL_FEATURE_SIZES,
    NUMERIC_INPUT_COLUMNS,
    TARGET_ORDER,
    candidate_filename,
    full_feature_columns,
)
from multimodal.geo_helpfulness_router_capacity_state import predict_state
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
    validate_manifest,
    write_bytes,
    write_json,
)

STUDY_ID = "router_capacity_v1"
VENDOR = (
    PROJECT_ROOT
    / "multimodal_artifacts/environments/router_capacity_v1/xgboost_cpu_2_1_4"
)
IDENTITY = ["row_uid", "plot_idx", "training_seed"]
Q_COLUMNS = [f"q_{name}" for name in TARGET_ORDER]
FEATURES = full_feature_columns()
PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("plot_idx", pa.string(), nullable=False),
        pa.field("training_seed", pa.int8(), nullable=False),
        *[
            pa.field(name, pa.string(), nullable=False)
            for name in ("config_id", "family", "variant")
        ],
        *[
            pa.field(name, pa.float64(), nullable=False)
            for name in (*Q_COLUMNS, "score")
        ],
    ]
)
CODE_FILES = [
    "multimodal/geo_helpfulness_router_capacity.py",
    "multimodal/geo_helpfulness_router_capacity_inputs.py",
    "multimodal/geo_helpfulness_router_capacity_fit.py",
    "multimodal/geo_helpfulness_router_capacity_state.py",
    "multimodal/geo_helpfulness_router_capacity_score.py",
    "multimodal/geo_helpfulness_router_candidate_state.py",
    "multimodal/geo_helpfulness_router_m4_common.py",
    "multimodal/geo_helpfulness_router_metrics.py",
    "multimodal/geo_helpfulness_router_selection.py",
    "multimodal/geo_helpfulness_router_uncertainty.py",
    "multimodal/geo_helpfulness_router_heuristics.py",
    "tools/run_multimodal_geo_helpfulness_router_capacity.py",
    "configs/router_capacity_requirements.txt",
]


def _seal(value, key):
    return {**value, key: canonical_hash(value)}


def _files(root):
    if root.is_symlink():
        raise ValueError("capacity directory cannot be a symlink")
    for path in root.rglob("*") if root.exists() else []:
        if path.is_symlink():
            raise ValueError(f"capacity path cannot be a symlink: {path}")
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def _allowlist(root, allowed, *, complete=False):
    observed = _files(root)
    if observed - set(allowed) or (complete and observed != set(allowed)):
        raise ValueError(f"capacity artifact allowlist mismatch: {root}")


def _stem(spec, seed):
    return f"{spec['config_id']}__seed_{seed}"


def _prediction_files(config):
    result = {"validation_features.parquet"}
    for spec in config["configurations"]:
        for seed in range(1, 5):
            stem = _stem(spec, seed)
            result.update(
                (
                    f"models/{stem}.json",
                    f"train/{stem}.parquet",
                    f"validation/{stem}.parquet",
                )
            )
    return result


def load_declaration(output_root):
    output_root = Path(output_root)
    declaration = read_json(output_root / "declaration.json")
    check_self_hash(declaration, "declaration_sha256")
    if (
        declaration.get("study_id") != STUDY_ID
        or declaration.get("schema_version")
        != "geo_helpfulness.router_capacity_declaration.v1"
    ):
        raise ValueError("capacity declaration identity mismatch")
    check_snapshot(declaration["parent_file_sha256"])
    _allowlist(
        output_root,
        {
            "declaration.json",
            *[
                f"predictions/{name}"
                for name in _prediction_files(declaration["configuration"])
            ],
            "predictions/manifest.json",
            *[f"scores/{name}" for name in declaration["score_files"]],
            "scores/manifest.json",
        },
    )
    return declaration


def _manifest(root, declaration, kind):
    value = read_json(root / "manifest.json")
    check_self_hash(value)
    if (
        value.get("schema_version") != f"geo_helpfulness.router_capacity_{kind}.v1"
        or value.get("declaration_sha256") != declaration["declaration_sha256"]
    ):
        raise ValueError("capacity manifest/declaration mismatch")
    _allowlist(root, {*value["artifacts"], "manifest.json"}, complete=True)
    for name, record in value["artifacts"].items():
        if file_hash(root / name) != record["file_sha256"]:
            raise ValueError(f"capacity artifact changed: {name}")
    return value


def _publish_manifest(root, declaration, kind, expected, extra=None):
    check_snapshot(declaration["parent_file_sha256"])
    _allowlist(root, expected, complete=True)
    artifacts = {}
    for name in sorted(expected):
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
    result = _seal(
        {
            "schema_version": f"geo_helpfulness.router_capacity_{kind}.v1",
            "study_id": STUDY_ID,
            "declaration_sha256": declaration["declaration_sha256"],
            "artifacts": artifacts,
            **(extra or {}),
        },
        "manifest_sha256",
    )
    write_json(root / "manifest.json", result)
    return result


def _table(frame):
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        None
    )
    return table.cast(
        pa.schema(
            [
                pa.field(f.name, f.type, nullable=table[f.name].null_count > 0)
                for f in table.schema
            ]
        )
    )


def _prediction_table(identity, probabilities, spec):
    if (
        probabilities.dtype != np.float64
        or probabilities.shape != (len(identity), 4)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("capacity probabilities must be normalized finite float64")
    frame = identity[IDENTITY].copy().reset_index(drop=True)
    for name in ("config_id", "family", "variant"):
        frame[name] = spec[name]
    for index, name in enumerate(Q_COLUMNS):
        frame[name] = probabilities[:, index]
    frame["score"] = probabilities[:, 0] - probabilities[:, 1]
    return pa.Table.from_pandas(
        frame, schema=PREDICTION_SCHEMA, preserve_index=False
    ).replace_schema_metadata(None)


def _pure_transform(frame, state):
    if state["output_columns"] != FEATURES or state["numeric_input_columns"] != list(
        NUMERIC_INPUT_COLUMNS
    ):
        raise ValueError("frozen transform columns changed")
    mean, scale = np.asarray(state["mean"], dtype=np.float64), np.asarray(
        state["scale"], dtype=np.float64
    )
    if (
        mean.shape != (25,)
        or scale.shape != (25,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("frozen numeric moments must be finite with positive scales")
    matrix = np.zeros((len(frame), 727), dtype=np.float64)
    matrix[:, :25] = (
        frame[state["numeric_input_columns"]].to_numpy(dtype=np.float64)
        - np.asarray(state["mean"])
    ) / np.asarray(state["scale"])
    offset = 25
    for name, size in CATEGORICAL_FEATURE_SIZES:
        vocab = state["categorical_vocabularies"][name]
        if frame[name].dtype.kind not in "iu":
            raise ValueError("categorical features must retain integer dtype")
        categories = frame[name].to_numpy(dtype=np.int64)
        if vocab != list(range(size)) or not np.isin(categories, vocab).all():
            raise ValueError("frozen categorical vocabulary mismatch")
        matrix[np.arange(len(frame)), offset + categories] = 1
        offset += len(vocab)
    if offset != 727 or not np.isfinite(matrix).all():
        raise ValueError("nonfinite frozen transform")
    return matrix


def _replay_inputs(declaration, output_root):
    router = Path(declaration["protocol_root"]) / "router"
    train = pq.read_table(router / "router_dataset.parquet").to_pandas()
    audit = pq.read_table(router / "router_dataset_audit.parquet").to_pandas()
    experts = pq.read_table(
        router / "m4_comparator_inputs_v1/expert_probabilities_and_features.parquet"
    ).to_pandas()
    validation = pq.read_table(
        Path(output_root) / "predictions/validation_features.parquet"
    ).to_pandas()
    transforms = read_json(router / "router_feature_transform.json")["states"]
    result = {}
    for seed in range(1, 5):
        rows = [
            f.loc[f.training_seed.eq(seed)]
            .sort_values("row_uid")
            .reset_index(drop=True)
            for f in (train, audit, experts, validation)
        ]
        training, semantic, expert, valid = rows
        if (
            len(training) != 3378
            or training.plot_idx.nunique() != 1300
            or len(valid) != 822
            or valid.plot_idx.nunique() != 325
            or set(training.plot_idx) & set(valid.plot_idx)
        ):
            raise ValueError(
                "capacity train/validation identity counts or plot separation differ"
            )
        if not training[IDENTITY].equals(semantic[IDENTITY]) or not valid[
            IDENTITY
        ].equals(expert[IDENTITY]):
            raise ValueError("capacity replay identity alignment differs")
        state = next(s for s in transforms if s["seed"] == seed)
        matrices = training[FEATURES].to_numpy(), valid[FEATURES].to_numpy()
        for observed, expected in zip(
            matrices, (_pure_transform(semantic, state), _pure_transform(expert, state))
        ):
            if observed.dtype != np.float64 or not np.array_equal(observed, expected):
                raise ValueError("capacity feature matrix does not reproduce exactly")
        result[seed] = {
            "X_train": matrices[0],
            "X_validation": matrices[1],
            "train_identity": training[IDENTITY],
            "validation_identity": valid[IDENTITY],
        }
    return result


def _model_record(path, declaration, spec, seed):
    record = read_json(path)
    check_self_hash(record, "record_sha256")
    if record.get("declaration_sha256") != declaration["declaration_sha256"]:
        raise ValueError("capacity model belongs to another declaration")
    state = record["state"]
    if (
        any(
            state.get(name) != spec[name] for name in ("config_id", "family", "variant")
        )
        or state["training_seed"] != seed
        or state["fit_row_count"] != 3378
    ):
        raise ValueError("capacity model specification mismatch")
    parameters = dict(spec["parameters"])
    if spec["family"] != "lr":
        parameters["random_state"] = seed
    if state["parameters"] != parameters:
        raise ValueError("capacity resolved parameters mismatch")
    return record


def _validate_predictions(output_root):
    output_root = Path(output_root)
    declaration = load_declaration(output_root)
    root = output_root / "predictions"
    manifest = _manifest(root, declaration, "prediction_manifest")
    if set(manifest["artifacts"]) != _prediction_files(declaration["configuration"]):
        raise ValueError("incomplete capacity prediction cross-product")
    inputs = _replay_inputs(declaration, output_root)
    for spec in declaration["configuration"]["configurations"]:
        for seed in range(1, 5):
            stem, data = _stem(spec, seed), inputs[seed]
            state = _model_record(
                root / f"models/{stem}.json", declaration, spec, seed
            )["state"]
            for role in ("train", "validation"):
                probability = predict_state(data[f"X_{role}"], state)
                expected = _prediction_table(
                    data[f"{role}_identity"], probability, spec
                )
                compare_tables(pq.read_table(root / f"{role}/{stem}.parquet"), expected)
    check_snapshot(declaration["parent_file_sha256"])
    return declaration, manifest


def validate_predictions(output_root):
    with exclusive_lock(Path(output_root)):
        _, manifest = _validate_predictions(output_root)
    return {
        "status": "validated",
        "candidate_seed_count": 144,
        "manifest_sha256": manifest["manifest_sha256"],
        "fitting_performed": False,
    }


SCORE_FILES = {
    "selection.json",
    "candidate_metrics.json",
    "expert_metrics.json",
    "classification.json",
    "policy_metrics.json",
    "threshold_curves.parquet",
    "actions.parquet",
    "bootstrap_summary.json",
    "bootstrap_replicates.parquet",
    "subgroup_metrics.parquet",
    "summary.json",
    "frozen_policies.json",
    "access_audit.json",
    "selection_receipt.json",
    "training_metrics.json",
}


def _create_declaration(protocol_root, output_root, config_path, inputs):
    config = read_json(config_path)
    from multimodal.geo_helpfulness_router_capacity_score import _configurations

    _configurations(config["configurations"])
    for spec in config["configurations"]:
        if not re.fullmatch(r"[a-zA-Z0-9_]+", spec["config_id"]):
            raise ValueError("unsafe configuration filename")
    if config["study_id"] != STUDY_ID:
        raise ValueError("capacity study identifier mismatch")
    snapshot = dict(inputs["parent_file_hashes"])
    snapshot.update(
        {
            str(PROJECT_ROOT / name): file_hash(PROJECT_ROOT / name)
            for name in CODE_FILES
        }
    )
    snapshot[str(config_path)] = file_hash(config_path)
    m4_root = protocol_root / "router/m4_feasibility_v1"
    m4 = validate_manifest(
        m4_root, schema_version="geo_helpfulness.m4_feasibility_manifest.v1"
    )
    if m4["primary_decision"] != "no_go":
        raise ValueError("original M4 decision changed")
    snapshot.update(m4["parent_file_sha256"])
    snapshot[str(m4_root / "manifest.json")] = file_hash(m4_root / "manifest.json")
    snapshot.update(
        {
            str(m4_root / name): record["file_sha256"]
            for name, record in m4["artifacts"].items()
        }
    )
    vendor_files = [
        p
        for p in VENDOR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    if not vendor_files:
        raise ValueError("install the pinned project-local CPU XGBoost package first")
    snapshot.update({str(p): file_hash(p) for p in vendor_files})
    environment = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")
    }
    environment.update(python=sys.version, xgboost_cpu="2.1.4", cpu_threads=1)
    declaration = _seal(
        {
            "schema_version": "geo_helpfulness.router_capacity_declaration.v1",
            "study_id": STUDY_ID,
            "declared_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_root": str(protocol_root),
            "configuration": config,
            "parent_file_sha256": snapshot,
            "environment": environment,
            "original_m4_manifest_sha256": m4["manifest_sha256"],
            "score_files": sorted(SCORE_FILES),
            "validation_labels_decoded_before_prediction_seal": False,
            "cleaned_test_sources_accessed": False,
            "partial_recovery": "reuse_only_declared_self_hashed_model_records_no_deletions",
            "probability_basis": "portable_state_float64_probabilities_xgb_float32_margins_then_float64_softmax",
            "replay_atol": 1e-12,
            "replay_rtol": 1e-12,
            "native_xgb_parity_atol_rtol": 1e-6,
        },
        "declaration_sha256",
    )
    check_snapshot(snapshot)
    write_json(output_root / "declaration.json", declaration)
    return declaration


def build_predictions(protocol_root, output_root, config_path):
    protocol_root, output_root, config_path = map(
        lambda p: Path(p).resolve(), (protocol_root, output_root, config_path)
    )
    if output_root == protocol_root or protocol_root in output_root.parents:
        raise ValueError(
            "capacity outputs must be outside the frozen protocol directory"
        )
    with exclusive_lock(output_root):
        if (output_root / "declaration.json").exists():
            declaration = load_declaration(output_root)
            if declaration["protocol_root"] != str(protocol_root) or declaration[
                "configuration"
            ] != read_json(config_path):
                raise ValueError("repeat build paths/configuration differ")
            if (output_root / "predictions/manifest.json").exists():
                _, manifest = _validate_predictions(output_root)
                return {
                    "status": "reused_valid",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "candidate_seed_count": 144,
                    "new_fits": 0,
                }
        else:
            if _files(output_root):
                raise ValueError("unowned capacity directory is not empty")
            declaration = None
        from multimodal.geo_helpfulness_router_capacity_inputs import (
            load_capacity_inputs,
        )

        inputs = load_capacity_inputs(PROJECT_ROOT, protocol_root)
        if declaration is None:
            declaration = _create_declaration(
                protocol_root, output_root, config_path, inputs
            )
        check_snapshot(declaration["parent_file_sha256"])
        root = output_root / "predictions"
        expected = _prediction_files(declaration["configuration"])
        _allowlist(root, expected)
        write_bytes(
            root / "validation_features.parquet",
            table_bytes(_table(inputs["validation"])),
        )
        sys.path.insert(0, str(VENDOR))
        from multimodal.geo_helpfulness_router_capacity_fit import fit_candidate

        new_fits = 0
        for index, spec in enumerate(declaration["configuration"]["configurations"]):
            for seed in range(1, 5):
                stem, data = _stem(spec, seed), inputs["seed_inputs"][seed]
                path = root / f"models/{stem}.json"
                if path.exists():
                    record = _model_record(path, declaration, spec, seed)
                    state = record["state"]
                    train_q, val_q = (
                        predict_state(data[f"X_{role}"], state)
                        for role in ("train", "validation")
                    )
                else:
                    local_spec = dict(spec)
                    if spec["family"] == "lr":
                        local_spec["existing_lr_state"] = inputs["candidate_states"][
                            candidate_filename(
                                spec["variant"], seed, float(spec["parameters"]["C"])
                            )
                        ]
                    state, train_q, val_q, diagnostics = fit_candidate(
                        data["X_train"],
                        data["y_train"],
                        data["X_validation"],
                        local_spec,
                        seed,
                    )
                    record = _seal(
                        {
                            "declaration_sha256": declaration["declaration_sha256"],
                            "state": state,
                            "diagnostics": diagnostics,
                        },
                        "record_sha256",
                    )
                    write_json(path, record)
                    new_fits += spec["family"] != "lr"
                for role, probability in (("train", train_q), ("validation", val_q)):
                    write_bytes(
                        root / f"{role}/{stem}.parquet",
                        table_bytes(
                            _prediction_table(
                                data[f"{role}_identity"], probability, spec
                            )
                        ),
                    )
                print(
                    f"capacity {index * 4 + seed}/144 {stem} sealed",
                    file=sys.stderr,
                    flush=True,
                )
        manifest = _publish_manifest(
            root,
            declaration,
            "prediction_manifest",
            expected,
            {
                "candidate_seed_count": 144,
                "lr_reused_count": 48,
                "new_tree_model_count": 96,
                "validation_prediction_rows": 118368,
                "training_prediction_rows": 486432,
                "validation_labels_accessed": False,
                "cleaned_test_accessed": False,
            },
        )
        return {
            "status": "built",
            "manifest_sha256": manifest["manifest_sha256"],
            "candidate_seed_count": 144,
            "new_fits": new_fits,
        }


def _fit_incapable():
    forbidden = (
        "sklearn",
        "xgboost",
        "torch",
        "scipy.optimize",
        "multimodal.geo_helpfulness_router_capacity_fit",
        "multimodal.geo_helpfulness_router_capacity_inputs",
    )
    found = [
        name
        for name in sys.modules
        if any(name == f or name.startswith(f + ".") for f in forbidden)
    ]
    if found:
        raise ValueError(
            f"capacity scoring must run in a fresh fitting-incapable process: {found[:5]}"
        )


def _score_payloads(output_root, declaration, prediction_manifest):
    _fit_incapable()
    from multimodal.geo_helpfulness_router_capacity_score import evaluate_capacity
    from multimodal.geo_helpfulness_router_metrics import multiclass_metrics

    root = Path(output_root)
    protocol = Path(declaration["protocol_root"]) / "router"
    score_parent = protocol / "m4_feasibility_v1"
    targets = pq.read_table(score_parent / "validation_targets.parquet").to_pandas()
    experts = pq.read_table(
        protocol / "m4_comparator_inputs_v1/expert_probabilities_and_features.parquet"
    ).to_pandas()
    baseline = pq.read_table(score_parent / "validation_actions.parquet").to_pandas()
    heuristics = read_json(score_parent / "oof_heuristics.json")
    frames, training_metrics = [], {}
    training = pq.read_table(
        protocol / "router_dataset.parquet", columns=[*IDENTITY, "target_state"]
    ).to_pandas()
    for spec in declaration["configuration"]["configurations"]:
        per_seed = {}
        for seed in range(1, 5):
            stem = _stem(spec, seed)
            frames.append(
                pq.read_table(
                    root / f"predictions/validation/{stem}.parquet"
                ).to_pandas()
            )
            train_q = pq.read_table(
                root / f"predictions/train/{stem}.parquet"
            ).to_pandas()
            labels = (
                training.loc[training.training_seed.eq(seed)]
                .sort_values("row_uid")
                .reset_index(drop=True)
            )
            if not labels[IDENTITY].equals(train_q[IDENTITY]):
                raise ValueError("training diagnostics label join differs")
            y = labels.target_state.map(
                {name: i for i, name in enumerate(TARGET_ORDER)}
            ).to_numpy(dtype=np.int64)
            per_seed[str(seed)] = multiclass_metrics(train_q[Q_COLUMNS].to_numpy(), y)
        training_metrics[spec["config_id"]] = {
            "per_seed": per_seed,
            "interpretation": "resubstitution_fitting_diagnostic_not_independent_performance",
        }
    payloads = evaluate_capacity(
        pd.concat(frames, ignore_index=True),
        experts,
        targets,
        declaration["configuration"]["configurations"],
        baseline,
        heuristics["heuristics"],
    )
    groups = payloads["selection.json"]["groups"]
    frozen = {}
    for name, selection in groups.items():
        frozen[name] = {
            "config_id": selection["selected_config_id"],
            "threshold_specification": selection["threshold"][
                "threshold_specification"
            ],
            "models": {
                str(seed): {
                    "path": f"../predictions/models/{selection['selected_config_id']}__seed_{seed}.json",
                    "file_sha256": file_hash(
                        root
                        / f"predictions/models/{selection['selected_config_id']}__seed_{seed}.json"
                    ),
                }
                for seed in range(1, 5)
            },
        }
    payloads["frozen_policies.json"] = {
        "study_id": STUDY_ID,
        "groups": frozen,
        "preprocessing": "unchanged_M4_temperatures_and_transforms",
        "parent_declaration_sha256": declaration["declaration_sha256"],
        "further_selection_or_refitting_permitted": False,
    }
    payloads["training_metrics.json"] = training_metrics
    payloads["access_audit.json"] = {
        "study_id": STUDY_ID,
        "all_predictions_sealed_before_validation_labels": True,
        "prediction_manifest_sha256": prediction_manifest["manifest_sha256"],
        "validation_labels_source": str(score_parent / "validation_targets.parquet"),
        "fitting_imports_present": False,
        "test_sources_read": False,
        "original_m4_no_go_preserved": True,
        "conditional_reused_validation": True,
    }
    payloads["selection_receipt.json"] = _seal(
        {
            "study_id": STUDY_ID,
            "prediction_manifest_sha256": prediction_manifest["manifest_sha256"],
            "declaration_sha256": declaration["declaration_sha256"],
            "selection_sha256": canonical_hash(payloads["selection.json"]),
            "frozen_policies_sha256": canonical_hash(payloads["frozen_policies.json"]),
            "validation_target_file_sha256": file_hash(
                score_parent / "validation_targets.parquet"
            ),
            "post_receipt_reselection_permitted": False,
        },
        "receipt_sha256",
    )
    _fit_incapable()
    if set(payloads) != SCORE_FILES:
        raise ValueError("capacity scoring output schema changed")
    return payloads


def _scores(output_root, *, build):
    _fit_incapable()
    output_root = Path(output_root)
    with exclusive_lock(output_root):
        declaration, prediction_manifest = _validate_predictions(output_root)
        root = output_root / "scores"
        committed = (root / "manifest.json").exists()
        if committed:
            manifest = _manifest(root, declaration, "score_manifest")
            if (
                set(manifest["artifacts"]) != SCORE_FILES
                or manifest["prediction_manifest_sha256"]
                != prediction_manifest["manifest_sha256"]
            ):
                raise ValueError("capacity score manifest identity mismatch")
        elif not build:
            raise ValueError("capacity scoring bundle has not been committed")
        else:
            _allowlist(root, SCORE_FILES)
        payloads = _score_payloads(output_root, declaration, prediction_manifest)
        for name, value in payloads.items():
            path = root / name
            if isinstance(value, pa.Table):
                if path.exists():
                    compare_tables(pq.read_table(path), value)
                elif build:
                    write_bytes(path, table_bytes(value))
            else:
                if path.exists():
                    if read_json(path) != value:
                        raise ValueError(f"capacity score does not reproduce: {name}")
                elif build:
                    write_json(path, value)
        if not committed:
            manifest = _publish_manifest(
                root,
                declaration,
                "score_manifest",
                SCORE_FILES,
                {
                    "prediction_manifest_sha256": prediction_manifest[
                        "manifest_sha256"
                    ],
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
            "group_count": 9,
            "cleaned_test_accessed": False,
            "fitting_performed": False,
        }


def build_scores(output_root):
    return _scores(output_root, build=True)


def validate_scores(output_root):
    return _scores(output_root, build=False)

"""Versioned development threshold sensitivity and frozen legacy-test transfer.

Selection, label-blind test application, and test scoring run in separate
processes. All frozen RF model states and every predecessor remain immutable.
"""

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal import geo_helpfulness_legacy_capacity as predecessor
from multimodal import geo_helpfulness_relaxed_guardrails as lr_history
from multimodal import geo_helpfulness_legacy_router_common as legacy
from multimodal import geo_helpfulness_legacy_router_score as old_score
from multimodal.geo_helpfulness_router_capacity import _allowlist, _files
from multimodal.geo_helpfulness_router_heuristics import apply_heuristic_threshold
from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    VARIANTS,
    SEEDS,
    canonical_bytes,
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

STUDY_ID = "relaxed_guardrails_rf_v1"
SCHEMA = "geo_helpfulness.relaxed_rf.v1"
BASE = PROJECT_ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
DEFAULT_OUTPUT = BASE / STUDY_ID
DEFAULT_CONFIG = PROJECT_ROOT / "configs/multimodal_geo_relaxed_guardrails_rf_v1.json"
DEFAULT_LEGACY = BASE / "exploratory_legacy_capacity_v1"
DEFAULT_CAPACITY = BASE / "router_capacity_v1"
DEFAULT_LR_HISTORY = BASE / "relaxed_guardrails_lr_v1"
SELECTION_FILES = {"selection.json", "frozen_policies.json", "report.md"}
PREDICTION_FILES = {
    "experts.parquet",
    "routers.parquet",
    "actions.parquet",
    "metadata.json",
}
FAMILY_FILES = {*old_score.FAMILY_REPORTS, "summary.json", "paired_comparisons.json"}
SCORE_FILES = {
    "build_declaration.json",
    "reservation.json",
    "labels.parquet",
    "source_receipt.json",
    "summary.json",
    "report.md",
    "receipt.json",
    *[f"{family}/{name}" for family in legacy.FAMILIES for name in FAMILY_FILES],
}
CODE_FILES = (
    "multimodal/geo_helpfulness_relaxed_rf.py",
    "multimodal/geo_helpfulness_relaxed_rf_selection.py",
    "multimodal/geo_helpfulness_relaxed_rf_report.py",
    "multimodal/geo_helpfulness_relaxed_rf_registry.py",
    "tools/run_multimodal_geo_helpfulness_relaxed_rf.py",
    "multimodal/geo_helpfulness_legacy_capacity.py",
    "multimodal/geo_helpfulness_legacy_capacity_report.py",
    "multimodal/geo_helpfulness_router_capacity.py",
    "multimodal/geo_helpfulness_legacy_capacity_predictions.py",
    "multimodal/geo_helpfulness_relaxed_guardrails.py",
)
CONFIGURATION = {
    "schema_version": "geo_helpfulness.relaxed_rf_config.v1",
    "study_id": STUDY_ID,
    "learner": "frozen_development_selected_random_forest",
    "learner_choice_reason": "user_requested_RF_extension_of_predeclared_LR_guardrail_grid",
    "feature_variants": list(VARIANTS),
    "training_seeds": list(SEEDS),
    "minimum_plots_grid": [20, 10, 0],
    "harm_numerator_grid": [5, 7, 9],
    "harm_denominator": 10,
    "minimum_coverage_fraction": 0.01,
    "positive_net_utility_required": True,
    "threshold_selection_data": "sealed_capacity_development_validation",
    "new_policy_count": 27,
    "total_test_policy_count": 44,
    "checkpoint_families": list(legacy.FAMILIES),
    "unique_test_images": 1347,
    "unique_test_plots": 531,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20260826,
    "model_fitting_permitted": False,
    "model_reselection_permitted": False,
    "test_threshold_search_permitted": False,
    "original_m4_no_go_preserved": True,
    "independent_confirmation": False,
    "replay_atol": 1e-12,
    "replay_rtol": 1e-12,
    "probability_normalization_atol": 1e-8,
}


def _seal(value, key):
    return {**value, key: canonical_hash(value)}


def _fit_incapable(*, test_stage=False):
    forbidden = ["torch", "sklearn", "scipy", "xgboost"]
    if test_stage:
        predecessor._fit_incapable()
        forbidden += [
            "multimodal.geo_helpfulness_relaxed_rf_selection",
            "multimodal.geo_helpfulness_relaxed_guardrails_selection",
        ]
    if any(
        name == item or name.startswith(item + ".")
        for name in sys.modules
        for item in forbidden
    ):
        raise ValueError(
            "stage contains prohibited fitting or threshold-selection capability"
        )


def _configuration(path):
    value = read_json(path)
    if value != CONFIGURATION or canonical_hash(value) != canonical_hash(CONFIGURATION):
        raise ValueError(
            "configuration differs from the approved fixed sensitivity grid"
        )
    return value


def _allow_output(root):
    allowed = {"declaration.json"}
    for directory, files in (
        ("selection", SELECTION_FILES),
        ("predictions", PREDICTION_FILES),
        ("scores", SCORE_FILES),
    ):
        allowed |= {directory + "/" + name for name in files | {"manifest.json"}}
    _allowlist(root, allowed)


def _parents(legacy_root, capacity_root, lr_history_root):
    from multimodal.geo_helpfulness_legacy_capacity_predictions import (
        load_selected_policies,
    )
    from multimodal import geo_helpfulness_legacy_capacity_registry as parent_registry

    old = predecessor.load_declaration(legacy_root)
    prediction = predecessor._manifest(
        legacy_root / "predictions", old, "predictions", predecessor.PREDICTION_FILES
    )
    score = predecessor._manifest(
        legacy_root / "scores", old, "scores", predecessor.SCORE_FILES
    )
    if (
        prediction.get("study_id") != predecessor.STUDY_ID
        or score.get("study_id") != predecessor.STUDY_ID
        or len(_files(legacy_root)) != 36
        or Path(old["paths"]["capacity_root"]).resolve() != capacity_root
        or old.get("m4_no_go_preserved") is not True
    ):
        raise ValueError("completed legacy-capacity parent identity differs")
    build = predecessor._build_record(legacy_root, old, prediction)
    reservation = read_json(legacy_root / "scores/reservation.json")
    receipt = read_json(legacy_root / "scores/receipt.json")
    check_self_hash(receipt, "receipt_sha256")
    if (
        read_json(legacy_root / "scores/build_declaration.json") != build
        or any(receipt.get(k) != v for k, v in build.items())
        or receipt.get("reservation") != reservation
        or score.get("prediction_manifest_sha256") != prediction["manifest_sha256"]
        or score.get("receipt_sha256") != receipt["receipt_sha256"]
        or receipt.get("output_file_sha256")
        != {
            n: file_hash(legacy_root / "scores" / n)
            for n in predecessor.SCORE_FILES - {"receipt.json"}
        }
        or read_json(legacy_root / "scores/source_receipt.json")
        != predecessor._source_receipt(legacy_root, old, build)
    ):
        raise ValueError("legacy-capacity completed receipt linkage differs")
    parent_registry.validate_event(
        Path(old["registry_root"]),
        build,
        reservation,
        receipt,
        legacy_root / "scores/receipt.json",
    )
    original, original_prediction, original_score = predecessor._legacy_metadata(
        Path(old["paths"]["legacy_root"])
    )
    if (
        original["declaration_sha256"] != old["legacy_declaration_sha256"]
        or original_prediction["manifest_sha256"]
        != old["legacy_prediction_manifest_sha256"]
        or original_score["manifest_sha256"] != old["legacy_score_manifest_sha256"]
    ):
        raise ValueError("legacy-capacity original parent linkage differs")
    selected = load_selected_policies(
        capacity_root,
        Path(old["paths"]["mlp_source_root"]),
        Path(old["paths"]["mlp_replay_root"]),
        original,
    )
    if (
        selected["policies"] != old["frozen_policies"]
        or selected["lineage"] != old["router_parent_lineage"]
    ):
        raise ValueError("frozen RF configuration/model parent differs")
    history = lr_history.load_declaration(lr_history_root)
    for folder, files in (
        ("selection", lr_history.SELECTION_FILES),
        ("predictions", lr_history.PREDICTION_FILES),
        ("scores", lr_history.SCORE_FILES),
    ):
        lr_history._manifest(lr_history_root / folder, history, folder, files)
    if (
        len(_files(lr_history_root)) != 40
        or history.get("m4_no_go_preserved") is not True
        or history["registry_root"] != old["registry_root"]
        or history["identity_manifest"] != old["identity_manifest"]
    ):
        raise ValueError("preceding LR sensitivity history differs")
    return old, prediction, score, history


def _declare(root, legacy_root, capacity_root, lr_history_root, config_path):
    if _files(root):
        raise ValueError("unowned output is not empty")
    old, prediction, score, history = _parents(
        legacy_root, capacity_root, lr_history_root
    )
    snapshot = dict(old["parent_file_sha256"])
    for name, digest in history["parent_file_sha256"].items():
        if name in snapshot and snapshot[name] != digest:
            raise ValueError("parent snapshots disagree")
        snapshot[name] = digest
    for parent in (legacy_root, capacity_root, lr_history_root):
        snapshot.update(
            {str(parent / name): file_hash(parent / name) for name in _files(parent)}
        )
    registry = Path(old["registry_root"])
    snapshot.update({str(p): file_hash(p) for p in sorted(registry.rglob("*.json"))})
    snapshot.update(
        {
            str(PROJECT_ROOT / name): file_hash(PROJECT_ROOT / name)
            for name in CODE_FILES
        }
    )
    snapshot[str(config_path)] = file_hash(config_path)
    value = _seal(
        {
            "schema_version": SCHEMA + ".declaration",
            "study_id": STUDY_ID,
            "declared_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(root),
            "configuration": _configuration(config_path),
            "config_path": str(config_path),
            "paths": {
                "legacy_root": str(legacy_root),
                "capacity_root": str(capacity_root),
                "lr_history_root": str(lr_history_root),
            },
            "parent_file_sha256": snapshot,
            "legacy_declaration_sha256": old["declaration_sha256"],
            "legacy_prediction_manifest_sha256": prediction["manifest_sha256"],
            "legacy_score_manifest_sha256": score["manifest_sha256"],
            "lr_history_declaration_sha256": history["declaration_sha256"],
            "old_frozen_policies": old["frozen_policies"],
            "identity_manifest": old["identity_manifest"],
            "ontology": old["ontology"],
            "registry_root": str(registry),
            "family_descriptions": old["family_descriptions"],
            "model_choice_reason": CONFIGURATION["learner_choice_reason"],
            "user_authorization": "Extend the approved 20/10/no-plot and 0.5/0.7/0.9 harm-cap exploration to frozen RF routers; select thresholds on development and report both prior cleaned-test expert families.",
            "new_threshold_selection_permitted_only_on": "development_validation",
            "model_fitting_permitted": False,
            "model_reselection_permitted": False,
            "test_threshold_search_permitted": False,
            "historical_test_exposure": True,
            "adaptive_reuse": True,
            "m4_no_go_preserved": True,
            "independent_confirmation": False,
            "environment": {
                n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")
            },
            "python": sys.version,
        },
        "declaration_sha256",
    )
    check_snapshot(snapshot)
    write_json(root / "declaration.json", value)
    return value


def load_declaration(root):
    root = Path(root).resolve()
    _allow_output(root)
    value = read_json(root / "declaration.json")
    check_self_hash(value, "declaration_sha256")
    if (
        value.get("schema_version") != SCHEMA + ".declaration"
        or value.get("study_id") != STUDY_ID
        or value.get("output_root") != str(root)
        or value.get("configuration") != CONFIGURATION
        or canonical_hash(value.get("configuration")) != canonical_hash(CONFIGURATION)
    ):
        raise ValueError("relaxed-guardrail declaration mismatch")
    check_snapshot(value["parent_file_sha256"])
    _configuration(Path(value["config_path"]))
    return value


def _manifest(root, declaration, kind, expected):
    value = read_json(root / "manifest.json")
    check_self_hash(value)
    if (
        value.get("schema_version") != SCHEMA + "." + kind
        or value.get("study_id") != STUDY_ID
        or value.get("declaration_sha256") != declaration["declaration_sha256"]
        or set(value["artifacts"]) != expected
    ):
        raise ValueError("relaxed-guardrail manifest mismatch")
    _allowlist(root, expected | {"manifest.json"}, complete=True)
    for name, record in value["artifacts"].items():
        path = root / name
        if (
            file_hash(path) != record["file_sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"artifact changed: {name}")
    return value


def _publish(root, declaration, kind, expected, extra=None):
    check_snapshot(declaration["parent_file_sha256"])
    _allowlist(root, expected, complete=True)
    artifacts = {}
    for name in sorted(expected):
        path = root / name
        record = {"file_sha256": file_hash(path), "bytes": path.stat().st_size}
        if name.endswith(".parquet"):
            info = pq.ParquetFile(path)
            record.update(
                rows=info.metadata.num_rows,
                columns=len(info.schema_arrow),
                arrow_schema=str(info.schema_arrow),
            )
        artifacts[name] = record
    value = _seal(
        {
            "schema_version": SCHEMA + "." + kind,
            "study_id": STUDY_ID,
            "declaration_sha256": declaration["declaration_sha256"],
            "artifacts": artifacts,
            **(extra or {}),
        },
        "manifest_sha256",
    )
    write_json(root / "manifest.json", value)
    return value


def _selection_values(declaration):
    from multimodal.geo_helpfulness_relaxed_rf_selection import build_selection

    source = Path(declaration["paths"]["capacity_root"])
    original = read_json(source / "scores/selection.json")
    actions = pq.read_table(source / "scores/actions.parquet").to_pandas()
    for variant in VARIANTS:
        rows = actions.loc[actions.policy_id.eq("capacity__rf__" + variant)]
        if (
            len(rows) != 3288
            or rows.row_uid.nunique() != 822
            or rows.plot_idx.nunique() != 325
            or set(rows.training_seed) != set(SEEDS)
            or any(len(group) != 822 for _, group in rows.groupby("training_seed"))
        ):
            raise ValueError(
                "development validation must retain 822 images/325 plots for every seed"
            )
    selection = build_selection(original, actions, declaration["old_frozen_policies"])
    policies = {}
    lines = [
        "# Development-only relaxed RF threshold selection",
        "",
        "No model refitting, configuration reselection, or test data used in threshold selection.",
        "",
        "| Policy | Threshold | Actions | Rescues | Harms | Plots | Gain (pp) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, record in selection["policies"].items():
        threshold = record["threshold"]
        guard = threshold["guardrails"]
        policies[name] = {
            "policy_id": name,
            "learner": "rf",
            "variant_id": record["variant"],
            "original_policy_id": record["original_policy_id"],
            "config_id": record["config_id"],
            "guardrail_id": record["guardrail_id"],
            "minimum_plots": guard["minimum_plots"],
            "harm_numerator": guard["harm_numerator"],
            "harm_denominator": 10,
            "minimum_coverage_fraction": 0.01,
            "threshold_specification": threshold["threshold_specification"],
        }
        p = threshold["pooled"]
        display = threshold["threshold_display"]
        if display is None:
            display = "never intervene"
        lines.append(
            f"| {name} | {display} | {p['acted_count']} | {p['rescued_count']} | {p['harmed_count']} | {p['unique_acted_plots']} | {100*threshold['mean_seed_delta_accuracy']:.6f} |"
        )
    if len(policies) != 27:
        raise ValueError("expected exactly 27 declared relaxed RF policies")
    return {
        "selection.json": selection,
        "frozen_policies.json": policies,
        "report.md": "\n".join(lines) + "\n",
    }


def build_selection(
    output_root=DEFAULT_OUTPUT,
    legacy_root=DEFAULT_LEGACY,
    capacity_root=DEFAULT_CAPACITY,
    lr_history_root=DEFAULT_LR_HISTORY,
    config_path=DEFAULT_CONFIG,
):
    _fit_incapable()
    root, legacy_root, capacity_root, lr_history_root, config_path = map(
        lambda p: Path(p).resolve(),
        (output_root, legacy_root, capacity_root, lr_history_root, config_path),
    )
    _configuration(config_path)
    old, _, _, _ = _parents(legacy_root, capacity_root, lr_history_root)
    original = legacy.load_declaration(Path(old["paths"]["legacy_root"]))
    sources = {
        legacy_root,
        capacity_root,
        lr_history_root,
        *(Path(p).resolve() for p in old["paths"].values()),
        Path(old["registry_root"]).resolve(),
        Path(original["protocol_root"]).resolve(),
    }
    for source in sources:
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("output and frozen sources must be disjoint")
    paths = {
        "legacy_root": str(legacy_root),
        "capacity_root": str(capacity_root),
        "lr_history_root": str(lr_history_root),
    }
    with exclusive_lock(root):
        _allow_output(root)
        if (root / "declaration.json").exists():
            declaration = load_declaration(root)
            if declaration["paths"] != paths or declaration["config_path"] != str(
                config_path
            ):
                raise ValueError("build paths differ from declaration")
        else:
            declaration = _declare(
                root, legacy_root, capacity_root, lr_history_root, config_path
            )
        folder = root / "selection"
        committed = (folder / "manifest.json").exists()
        manifest = (
            _manifest(folder, declaration, "selection", SELECTION_FILES)
            if committed
            else None
        )
        _allowlist(
            folder, SELECTION_FILES | ({"manifest.json"} if committed else set())
        )
        for name, value in _selection_values(declaration).items():
            predecessor._compare_or_write(folder / name, value, not committed)
        if not committed:
            manifest = _publish(folder, declaration, "selection", SELECTION_FILES)
        check_snapshot(declaration["parent_file_sha256"])
    return {
        "status": "reused_valid" if committed else "built",
        "manifest_sha256": manifest["manifest_sha256"],
        "new_policy_count": 27,
        "model_fitting_performed": False,
        "test_data_decoded": False,
    }


def validate_selection(output_root):
    _fit_incapable()
    root = Path(output_root).resolve()
    with exclusive_lock(root):
        declaration = load_declaration(root)
        manifest = _manifest(
            root / "selection", declaration, "selection", SELECTION_FILES
        )
        for name, value in _selection_values(declaration).items():
            predecessor._compare_or_write(root / "selection" / name, value, False)
        check_snapshot(declaration["parent_file_sha256"])
    return {
        "status": "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "model_fitting_performed": False,
        "test_data_decoded": False,
    }


def _load_policies(root, declaration):
    from multimodal.geo_helpfulness_relaxed_rf_report import validate_new_policies

    manifest = _manifest(root / "selection", declaration, "selection", SELECTION_FILES)
    policies = read_json(root / "selection/frozen_policies.json")
    validate_new_policies(policies, declaration["old_frozen_policies"])
    return manifest, policies


def _test_tables(root, declaration):
    selection, policies = _load_policies(root, declaration)
    source = Path(declaration["paths"]["legacy_root"])
    old = predecessor.load_declaration(source)
    prediction = predecessor._manifest(
        source / "predictions", old, "predictions", predecessor.PREDICTION_FILES
    )
    score = predecessor._manifest(
        source / "scores", old, "scores", predecessor.SCORE_FILES
    )
    if (
        old["declaration_sha256"] != declaration["legacy_declaration_sha256"]
        or prediction["manifest_sha256"]
        != declaration["legacy_prediction_manifest_sha256"]
        or score["manifest_sha256"] != declaration["legacy_score_manifest_sha256"]
    ):
        raise ValueError("legacy parent linkage differs")
    _, _, frames = predecessor._validate_predictions(source)
    experts, routers = frames["experts"], frames["routers"]
    output = [frames["actions"].copy()]
    keys = ["row_uid", "training_seed"]
    for family in legacy.FAMILIES:
        e = (
            experts.loc[experts.checkpoint_family.eq(family)]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        for name, policy in policies.items():
            r = (
                routers.loc[
                    routers.checkpoint_family.eq(family)
                    & routers.policy_id.eq(policy["original_policy_id"])
                ]
                .sort_values(keys)
                .reset_index(drop=True)
            )
            if (
                not r[keys + ["plot_idx"]].equals(e[keys + ["plot_idx"]])
                or set(r.config_id) != {policy["config_id"]}
                or set(r.learner) != {"rf"}
            ):
                raise ValueError(
                    "legacy RF router identity or selected configuration differs"
                )
            specification = policy["threshold_specification"]
            acted = apply_heuristic_threshold(
                r.score.to_numpy(),
                e.raw_pred.to_numpy(),
                e.geo_pred.to_numpy(),
                specification,
            )
            frame = e[
                [
                    "schema_version",
                    "study_id",
                    "checkpoint_family",
                    "row_uid",
                    "plot_idx",
                    "training_seed",
                ]
            ].copy()
            frame["schema_version"], frame["study_id"] = SCHEMA + ".actions", STUDY_ID
            frame["policy_id"], frame["score"] = name, r.score.to_numpy()
            frame["threshold_specification"] = canonical_bytes(specification).decode()
            frame["acted"] = acted
            frame["raw_pred"], frame["geo_pred"] = (
                e.raw_pred.to_numpy(),
                e.geo_pred.to_numpy(),
            )
            frame["final_pred"] = np.where(acted, frame.geo_pred, frame.raw_pred)
            output.append(frame)
    actions = (
        pd.concat(output, ignore_index=True)
        .sort_values(["checkpoint_family", "policy_id", *keys], kind="stable")
        .reset_index(drop=True)
    )
    action_schema = pq.read_schema(source / "predictions/actions.parquet")
    tables = {
        "experts": pq.read_table(source / "predictions/experts.parquet"),
        "routers": pq.read_table(source / "predictions/routers.parquet"),
        "actions": pa.Table.from_pandas(
            actions, schema=action_schema, preserve_index=False, safe=True
        ),
    }
    expected = {"experts": 10776, "routers": 129312, "actions": 474144}
    if any(tables[k].num_rows != n for k, n in expected.items()):
        raise ValueError("test prediction cross-product differs")
    return selection, policies, tables


def _prediction_metadata(selection):
    return {
        "study_id": STUDY_ID,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "table_rows": {"experts": 10776, "routers": 129312, "actions": 474144},
        "new_policy_count": 27,
        "total_policy_count": 44,
        "test_labels_decoded": False,
        "threshold_selection_performed": False,
        "model_fitting_performed": False,
    }


def _validate_predictions(root):
    declaration = load_declaration(root)
    manifest = _manifest(
        root / "predictions", declaration, "predictions", PREDICTION_FILES
    )
    selection, policies, tables = _test_tables(root, declaration)
    if manifest["selection_manifest_sha256"] != selection["manifest_sha256"]:
        raise ValueError("prediction selection seal differs")
    for name, table in tables.items():
        compare_tables(pq.read_table(root / "predictions" / (name + ".parquet")), table)
    if read_json(root / "predictions/metadata.json") != _prediction_metadata(selection):
        raise ValueError("prediction metadata differs")
    check_snapshot(declaration["parent_file_sha256"])
    return (
        declaration,
        manifest,
        policies,
        {k: v.to_pandas() for k, v in tables.items()},
    )


def build_predictions(output_root):
    _fit_incapable(test_stage=True)
    root = Path(output_root).resolve()
    with exclusive_lock(root):
        declaration = load_declaration(root)
        folder = root / "predictions"
        committed = (folder / "manifest.json").exists()
        if committed:
            _, manifest, _, _ = _validate_predictions(root)
        else:
            _allowlist(folder, PREDICTION_FILES)
            selection, _, tables = _test_tables(root, declaration)
            for name, table in tables.items():
                if name in ("experts", "routers"):
                    source = (
                        Path(declaration["paths"]["legacy_root"])
                        / "predictions"
                        / (name + ".parquet")
                    )
                    write_bytes(folder / (name + ".parquet"), source.read_bytes())
                else:
                    predecessor._compare_or_write(
                        folder / (name + ".parquet"), table, True
                    )
            write_json(folder / "metadata.json", _prediction_metadata(selection))
            manifest = _publish(
                folder,
                declaration,
                "predictions",
                PREDICTION_FILES,
                {"selection_manifest_sha256": selection["manifest_sha256"]},
            )
            _validate_predictions(root)
        _fit_incapable(test_stage=True)
    return {
        "status": "reused_valid" if committed else "built",
        "manifest_sha256": manifest["manifest_sha256"],
        "model_fitting_performed": False,
        "threshold_selection_performed": False,
    }


def validate_predictions(output_root):
    _fit_incapable(test_stage=True)
    with exclusive_lock(Path(output_root).resolve()):
        _, manifest, _, _ = _validate_predictions(Path(output_root).resolve())
    _fit_incapable(test_stage=True)
    return {
        "status": "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "model_fitting_performed": False,
        "threshold_selection_performed": False,
    }


def _build_record(root, declaration, prediction):
    return {
        "schema_version": SCHEMA + ".score_build",
        "study_id": STUDY_ID,
        "owner_output_root": str(root),
        "declaration_sha256": declaration["declaration_sha256"],
        "prediction_manifest_sha256": prediction["manifest_sha256"],
        "composite_bundle_file_sha256": prediction["manifest_sha256"],
        "selection_manifest_sha256": prediction["selection_manifest_sha256"],
        "identity_projection_sha256": declaration["identity_manifest"][
            "identity_projection_sha256"
        ],
        "ontology_sha256": canonical_hash(declaration["ontology"]),
        "historical_test_exposure": True,
        "adaptive_reuse": True,
        "m4_no_go_preserved": True,
    }


def _source_receipt(root, declaration, build):
    source = Path(declaration["paths"]["legacy_root"]) / "scores/labels.parquet"
    own = root / "scores/labels.parquet"
    if file_hash(source) != file_hash(own):
        raise ValueError("copied labels differ from authenticated predecessor")
    return {
        "study_id": STUDY_ID,
        "build_sha256": canonical_hash(build),
        "reservation_file_sha256": file_hash(root / "scores/reservation.json"),
        "parent_label_file": str(source),
        "parent_label_file_sha256": file_hash(source),
        "label_copy_file_sha256": file_hash(own),
        "original_label_cache_opened": False,
        "original_historical_metrics_opened": False,
        "parent_legacy_score_manifest_sha256": declaration[
            "legacy_score_manifest_sha256"
        ],
    }


def _family_reports(frames, labels, policies, old_policies):
    from multimodal.geo_helpfulness_relaxed_rf_report import evaluate_family

    results = {}
    for family in legacy.FAMILIES:
        inputs = [
            frames[k].loc[frames[k].checkpoint_family.eq(family)].reset_index(drop=True)
            for k in ("experts", "routers", "actions")
        ]
        reports = evaluate_family(*inputs, labels, old_policies, policies)
        if set(reports) != FAMILY_FILES:
            raise ValueError("family report contract differs")
        results[family] = reports
    return results


def _report(results):
    text = predecessor._report(results)
    return text.replace(
        "# Frozen RF/XGB/MLP transfer to legacy experts",
        "# Relaxed-guardrail RF policies on legacy experts",
    ).replace(
        "No fitting, recalibration, threshold search, model selection, or M4 gate rerun.",
        "Thresholds were selected only on development-validation and frozen before test application. No test threshold search, model refit/reselection, recalibration, or M4 gate rerun.",
    )


def _scores(output_root, *, build):
    _fit_incapable(test_stage=True)
    from multimodal.geo_helpfulness_relaxed_rf_registry import (
        reserve_event,
        validate_reservation,
        finish_event,
        validate_event,
    )

    root = Path(output_root).resolve()
    with exclusive_lock(root):
        declaration, prediction, policies, frames = _validate_predictions(root)
        score = root / "scores"
        committed = (score / "manifest.json").exists()
        manifest = (
            _manifest(score, declaration, "scores", SCORE_FILES) if committed else None
        )
        if not committed and not build:
            raise ValueError("score bundle is not committed")
        _allowlist(score, SCORE_FILES | ({"manifest.json"} if committed else set()))
        record = _build_record(root, declaration, prediction)
        registry = Path(declaration["registry_root"])
        if committed:
            if read_json(score / "build_declaration.json") != record:
                raise ValueError("score build declaration differs")
            reservation = read_json(score / "reservation.json")
        else:
            if _files(score) and not (score / "build_declaration.json").exists():
                raise ValueError("unowned partial score outputs")
            write_json(score / "build_declaration.json", record)
            reservation = reserve_event(registry, record)
            write_json(score / "reservation.json", reservation)
        validate_reservation(registry, record, reservation)
        source = Path(declaration["paths"]["legacy_root"]) / "scores/labels.parquet"
        if not (score / "labels.parquet").exists():
            if committed:
                raise ValueError("committed label copy missing")
            write_bytes(score / "labels.parquet", source.read_bytes())
        predecessor._compare_or_write(
            score / "source_receipt.json",
            _source_receipt(root, declaration, record),
            build,
        )
        labels_table = pq.read_table(score / "labels.parquet")
        if labels_table.schema != old_score.LABEL_SCHEMA:
            raise ValueError("label schema differs")
        labels = old_score._validate_labels(labels_table.to_pandas(), declaration)
        results = _family_reports(
            frames, labels, policies, declaration["old_frozen_policies"]
        )
        values = {
            f"{family}/{name}": value
            for family, reports in results.items()
            for name, value in reports.items()
        }
        values["summary.json"] = {
            "study_id": STUDY_ID,
            "unique_images": 1347,
            "unique_plots": 531,
            "training_seeds": list(SEEDS),
            "checkpoint_families": {f: r["summary.json"] for f, r in results.items()},
            "adaptive_reuse": True,
            "independent_confirmation": False,
            "m4_no_go_preserved": True,
            "test_threshold_search_performed": False,
            "model_choice_reason": CONFIGURATION["learner_choice_reason"],
        }
        values["report.md"] = _report(results)
        for name, value in values.items():
            predecessor._compare_or_write(score / name, value, build)
        receipt = _seal(
            {
                **record,
                "artifact_role": "relaxed_guardrails_test_score_receipt",
                "reservation": reservation,
                "ordered_scoring_label_sha256": old_score._label_hash(labels),
                "output_file_sha256": {
                    name: file_hash(score / name)
                    for name in sorted(SCORE_FILES - {"receipt.json"})
                },
                "evaluation_event_id": canonical_hash(
                    {
                        "study_id": STUDY_ID,
                        "prediction": prediction["manifest_sha256"],
                        "identity": record["identity_projection_sha256"],
                        "labels": old_score._label_hash(labels),
                    }
                ),
                "model_fitting": False,
                "model_reselection": False,
                "test_threshold_search": False,
                "recalibration": False,
                "independent_confirmation": False,
            },
            "receipt_sha256",
        )
        predecessor._compare_or_write(score / "receipt.json", receipt, build)
        check_snapshot(declaration["parent_file_sha256"])
        if not committed:
            finish_event(registry, record, reservation, receipt, score / "receipt.json")
            manifest = _publish(
                score,
                declaration,
                "scores",
                SCORE_FILES,
                {
                    "prediction_manifest_sha256": prediction["manifest_sha256"],
                    "receipt_sha256": receipt["receipt_sha256"],
                },
            )
        if (
            manifest["prediction_manifest_sha256"] != prediction["manifest_sha256"]
            or manifest["receipt_sha256"] != receipt["receipt_sha256"]
        ):
            raise ValueError("score receipt linkage differs")
        validate_event(registry, record, reservation, receipt, score / "receipt.json")
        check_snapshot(declaration["parent_file_sha256"])
        _fit_incapable(test_stage=True)
    return {
        "status": ("reused_valid" if committed else "built") if build else "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "evaluation_event_id": receipt["evaluation_event_id"],
        "model_fitting_performed": False,
        "test_threshold_search_performed": False,
        "original_label_cache_opened": False,
        "m4_no_go_preserved": True,
    }


def build_scores(output_root):
    return _scores(output_root, build=True)


def validate_scores(output_root):
    return _scores(output_root, build=False)

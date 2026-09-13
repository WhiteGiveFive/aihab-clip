"""Apply sealed development-selected routers to sealed legacy expert outputs.

Selection loading authenticates metadata and bytes only. Prediction replay takes
already-validated, label-blind legacy frames; it never opens an expert cache,
decodes an evaluation target, imports a learner, or selects a new policy.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

from multimodal import geo_helpfulness_legacy_router_common as legacy
from multimodal import geo_helpfulness_router_capacity_state as capacity_state
from multimodal import geo_helpfulness_router_mlp_state as mlp_state
from multimodal.geo_helpfulness_router_candidate_state import (
    C_TOKENS,
    TARGET_ORDER,
    validate_candidate_state,
)
from multimodal.geo_helpfulness_router_heuristics import (
    apply_heuristic_threshold,
    validate_threshold_specification,
)


STUDY_ID = "legacy_capacity_router_transfer_v1"
VARIANTS, SEEDS = legacy.VARIANTS, legacy.SEEDS
NEW_POLICIES = tuple(
    f"capacity__{learner}__{variant}"
    for learner in ("rf", "xgb")
    for variant in VARIANTS
) + tuple(f"mlp__{variant}" for variant in VARIANTS)
POLICIES = (*legacy.POLICIES, *NEW_POLICIES)
Q_COLUMNS = tuple(f"q_{name}" for name in TARGET_ORDER)
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
MLP_SCORE_FILES = SCORE_FILES | {"paired_comparisons.json"}
ROUTER_SCHEMA = pa.schema(
    [
        pa.field(name, dtype, nullable=False)
        for name, dtype in [
            ("schema_version", pa.string()),
            ("study_id", pa.string()),
            ("checkpoint_family", pa.string()),
            ("row_uid", pa.string()),
            ("plot_idx", pa.string()),
            ("training_seed", pa.int8()),
            ("policy_id", pa.string()),
            ("learner", pa.string()),
            ("variant_id", pa.string()),
            ("config_id", pa.string()),
            ("C_hex", pa.string()),
            *[(name, pa.float64()) for name in Q_COLUMNS],
            ("score", pa.float64()),
        ]
    ]
)


def _fit_incapable():
    forbidden = ("torch", "sklearn", "scipy", "xgboost")
    if any(name.split(".")[0] in forbidden for name in sys.modules) or any(
        name.startswith("multimodal.geo_helpfulness_")
        and (
            name.endswith("_fit")
            or name.endswith("_inputs")
            or name.endswith("_selection")
        )
        for name in sys.modules
    ):
        raise ValueError("legacy capacity inference requires a fit-incapable process")


def _path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"source path must not follow symlinks: {path}")
    return path.resolve()


def _merge(snapshot, source):
    for name, digest in source.items():
        if not Path(name).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid source snapshot entry")
        name = str(_path(name))
        if name in snapshot and snapshot[name] != digest:
            raise ValueError(f"parent snapshots disagree: {name}")
        snapshot[name] = digest


def _capture(path, snapshot, expected=None):
    path = _path(path)
    digest = legacy.file_hash(path)
    if expected is not None and digest != expected:
        raise ValueError(f"sealed source file changed: {path}")
    _merge(snapshot, {str(path): digest})
    return digest


def _json(path, snapshot, expected=None):
    digest = _capture(path, snapshot, expected)
    result = legacy.read_json(path)
    _capture(path, snapshot, digest)
    return result


def _files(root, expected):
    root = _path(root)
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("source bundles must not contain symlinks")
    observed = {str(path.relative_to(root)) for path in paths if path.is_file()}
    if observed != set(expected):
        raise ValueError(f"source bundle allowlist mismatch: {root}")


def _manifest(root, snapshot, schema, declaration_hash, expected):
    root = _path(root)
    manifest = _json(root / "manifest.json", snapshot)
    legacy.check_self_hash(manifest)
    if (
        manifest.get("schema_version") != schema
        or manifest.get("declaration_sha256") != declaration_hash
        or set(manifest.get("artifacts", {})) != set(expected)
    ):
        raise ValueError("source manifest identity or artifact set mismatch")
    _files(root, {*expected, "manifest.json"})
    for name, record in manifest["artifacts"].items():
        _capture(root / name, snapshot, record["file_sha256"])
        if "bytes" in record and (root / name).stat().st_size != record["bytes"]:
            raise ValueError("source manifest byte count mismatch")
    return manifest


def _configuration(declaration, family):
    config = declaration["configuration"]
    if (
        config.get("study_id") != f"router_{family}_v1"
        or config.get("schema_version") != f"geo_helpfulness.router_{family}_config.v1"
    ):
        raise ValueError("source configuration identity mismatch")
    records = config["configurations"]
    learners = ("lr", "rf", "xgb") if family == "capacity" else ("mlp",)
    if len(records) != 12 * len(learners):
        raise ValueError("source configuration count mismatch")
    mapping, counts = {}, {}
    for spec in records:
        name, learner, variant = (
            spec[key] for key in ("config_id", "family", "variant")
        )
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-zA-Z0-9_]+", name)
            or name in mapping
            or learner not in learners
            or variant not in VARIANTS
            or not isinstance(spec["parameters"], dict)
        ):
            raise ValueError("invalid source configuration")
        mapping[name] = spec
        counts[learner, variant] = counts.get((learner, variant), 0) + 1
    if counts != {
        (learner, variant): 4 for learner in learners for variant in VARIANTS
    }:
        raise ValueError("source family/variant grid mismatch")
    return mapping


def _prediction_files(configs, *, mlp=False):
    files = {"validation_features.parquet"}
    if mlp:
        files.add("training_curves.parquet")
    for name in configs:
        for seed in SEEDS:
            stem = f"{name}__seed_{seed}"
            files.update(
                (
                    f"models/{stem}.json",
                    f"train/{stem}.parquet",
                    f"validation/{stem}.parquet",
                )
            )
    return files


def _declaration(root, snapshot, family):
    value = _json(root / "declaration.json", snapshot)
    legacy.check_self_hash(value, "declaration_sha256")
    if (
        value.get("study_id") != f"router_{family}_v1"
        or value.get("schema_version")
        != f"geo_helpfulness.router_{family}_declaration.v1"
    ):
        raise ValueError("source declaration identity mismatch")
    _merge(snapshot, value["parent_file_sha256"])
    return value, _configuration(value, family)


def _selection(root, snapshot, declaration, prediction, *, recovery=None):
    selected = _json(root / "scores/selection.json", snapshot)
    frozen = _json(root / "scores/frozen_policies.json", snapshot)
    receipt = _json(root / "scores/selection_receipt.json", snapshot)
    legacy.check_self_hash(receipt, "receipt_sha256")
    declaration_key = (
        "parent_declaration_sha256" if recovery is None else "declaration_sha256"
    )
    if (
        receipt.get("study_id") != declaration["study_id"]
        or frozen.get("study_id") != declaration["study_id"]
        or receipt.get("declaration_sha256") != declaration["declaration_sha256"]
        or frozen.get(declaration_key) != declaration["declaration_sha256"]
        or receipt.get("prediction_manifest_sha256") != prediction["manifest_sha256"]
        or receipt.get("selection_sha256") != legacy.canonical_hash(selected)
        or receipt.get("frozen_policies_sha256") != legacy.canonical_hash(frozen)
        or receipt.get("post_receipt_reselection_permitted") is not False
        or frozen.get("further_selection_or_refitting_permitted") is not False
        or set(selected["groups"]) != set(frozen["groups"])
    ):
        raise ValueError("selected-policy receipt lineage mismatch")
    if recovery is not None:
        lineage = frozen.get("recovery_lineage", {})
        if (
            receipt.get("recovery_lineage") != lineage
            or lineage.get("recovery_id") != "router_mlp_v1_replay_v2"
            or lineage.get("recovery_declaration_sha256")
            != recovery["declaration_sha256"]
            or lineage.get("source_declaration_sha256")
            != declaration["declaration_sha256"]
            or lineage.get("source_prediction_manifest_sha256")
            != prediction["manifest_sha256"]
            or lineage.get("source_files_changed") is not False
            or frozen.get("model_path_basis") != "absolute_unchanged_source"
        ):
            raise ValueError("MLP selected-policy recovery lineage mismatch")
    for name, policy in frozen["groups"].items():
        chosen = selected["groups"][name]
        if (
            policy["config_id"] != chosen["selected_config_id"]
            or policy["threshold_specification"]
            != chosen["threshold"]["threshold_specification"]
            or set(policy["models"]) != {str(seed) for seed in SEEDS}
        ):
            raise ValueError("frozen selected config/threshold/seed mismatch")
        validate_threshold_specification(policy["threshold_specification"])
    return frozen, receipt


def _legacy_policies(declaration, snapshot):
    legacy.check_self_hash(declaration, "declaration_sha256")
    if (
        declaration.get("study_id") != legacy.STUDY_ID
        or declaration.get("no_go_preserved") is not True
    ):
        raise ValueError("legacy source must preserve the M4 no-go")
    root = _path(declaration["frozen_root"])
    bound = declaration["source_snapshot"]
    manifest = _json(
        root / "manifest.json", snapshot, bound[str(root / "manifest.json")]
    )
    legacy.check_self_hash(manifest)
    if (
        manifest.get("schema_version") != "geo_helpfulness.m4_feasibility_manifest.v1"
        or manifest.get("primary_decision") != "no_go"
    ):
        raise ValueError("original M4 manifest identity/no-go changed")
    frozen_path = root / "router_frozen_state.json"
    frozen = _json(frozen_path, snapshot, bound[str(frozen_path)])
    legacy.check_self_hash(frozen, "frozen_bundle_sha256")
    if (
        manifest["artifacts"][frozen_path.name]["file_sha256"]
        != snapshot[str(frozen_path)]
    ):
        raise ValueError("M4 frozen state manifest binding changed")
    if set(frozen["variants"]) != set(VARIANTS):
        raise ValueError("M4 feature variants changed")
    for name, record in frozen["components"].items():
        path = _path(root / name)
        if root not in path.parents or bound[str(path)] != record["file_sha256"]:
            raise ValueError("M4 component path or legacy parent mismatch")
        _capture(path, snapshot, record["file_sha256"])
    policies = {}
    for variant in VARIANTS:
        chosen = frozen["variants"][variant]
        if set(chosen["seeds"]) != {str(seed) for seed in SEEDS}:
            raise ValueError("M4 selected seed coverage mismatch")
        models, common_c, common_threshold = {}, [], []
        for seed in SEEDS:
            selected = chosen["seeds"][str(seed)]
            legacy.check_self_hash(selected, "frozen_state_sha256")
            path = _path(root / selected["model_file"])
            if root not in path.parents or str(path) not in snapshot:
                raise ValueError("M4 model path is not a bound component")
            state = _json(path, snapshot)
            validate_candidate_state(state)
            if (
                state["training_seed"] != seed
                or state["variant_id"] != variant
                or state["state_sha256"] != selected["model_state_sha256"]
                or state["C_hex"] != selected["C_hex"]
                or snapshot[str(path)] != selected["model_file_sha256"]
            ):
                raise ValueError("M4 selected model identity mismatch")
            common_c.append(state["C_hex"])
            common_threshold.append(selected["threshold_specification"])
            models[str(seed)] = {
                "path": str(path),
                "file_sha256": snapshot[str(path)],
                "state_sha256": state["state_sha256"],
            }
        if len(set(common_c)) != 1 or any(
            item != common_threshold[0] for item in common_threshold
        ):
            raise ValueError("M4 config and threshold must be shared across seeds")
        threshold, c_hex = common_threshold[0], common_c[0]
        if (
            chosen["selected_C_hex"] != c_hex
            or chosen["threshold_specification"] != threshold
        ):
            raise ValueError("M4 shared selection metadata mismatch")
        validate_threshold_specification(threshold)
        if variant == "primary_full727" and threshold != legacy.NEVER:
            raise ValueError("M4 primary no-go fallback changed")
        policies[variant] = {
            "learner": "lr",
            "variant_id": variant,
            "config_id": f"legacy__{variant}__C_{C_TOKENS[float.fromhex(c_hex)]}",
            "C_hex": c_hex,
            "threshold_specification": deepcopy(threshold),
            "models": models,
        }
    heuristics = _json(root / "oof_heuristics.json", snapshot)["heuristics"]
    fixed = {
        "never_intervene": legacy.NEVER,
        "always_geo_on_geo_raw_disagreement": legacy.ZERO,
        "higher_calibrated_geo_confidence_on_geo_raw_disagreement": legacy.ZERO,
        **{name: item["threshold_specification"] for name, item in heuristics.items()},
    }
    for name, threshold in fixed.items():
        validate_threshold_specification(threshold)
        policies[name] = {
            "learner": "heuristic",
            "variant_id": "",
            "config_id": name,
            "C_hex": "",
            "threshold_specification": deepcopy(threshold),
        }
    if set(policies) != set(legacy.POLICIES):
        raise ValueError("M4 policy set changed")
    return policies, manifest, frozen


def load_selected_policies(
    capacity_root, mlp_source_root, mlp_replay_root, legacy_declaration
):
    """Authenticate sealed selections and model JSON, without opening test frames."""
    _fit_incapable()
    capacity_root, mlp_source_root, mlp_replay_root = map(
        _path, (capacity_root, mlp_source_root, mlp_replay_root)
    )
    if len({capacity_root, mlp_source_root, mlp_replay_root}) != 3:
        raise ValueError("source and recovery roots must be distinct")
    snapshot = {}
    policies, m4_manifest, m4 = _legacy_policies(legacy_declaration, snapshot)
    capacity, capacity_configs = _declaration(capacity_root, snapshot, "capacity")
    mlp, mlp_configs = _declaration(mlp_source_root, snapshot, "mlp")
    protocol = _path(legacy_declaration["frozen_root"]).parent.parent
    for declaration in (capacity, mlp):
        if (
            _path(declaration["protocol_root"]) != protocol
            or declaration["original_m4_manifest_sha256"]
            != m4_manifest["manifest_sha256"]
        ):
            raise ValueError("router and legacy M4 protocol lineage differs")
        for name in (
            "expert_temperatures.json",
            "router_feature_transform.json",
            "router_feature_schema.json",
        ):
            component = m4["components"][name]
            source = str(_path(component["source"]))
            if (
                declaration["parent_file_sha256"].get(source)
                != component["file_sha256"]
            ):
                raise ValueError("selected routers used different M4 preprocessing")
    cap_prediction_files = _prediction_files(capacity_configs)
    mlp_prediction_files = _prediction_files(mlp_configs, mlp=True)
    cap_prediction = _manifest(
        capacity_root / "predictions",
        snapshot,
        "geo_helpfulness.router_capacity_prediction_manifest.v1",
        capacity["declaration_sha256"],
        cap_prediction_files,
    )
    cap_score = _manifest(
        capacity_root / "scores",
        snapshot,
        "geo_helpfulness.router_capacity_score_manifest.v1",
        capacity["declaration_sha256"],
        SCORE_FILES,
    )
    mlp_prediction = _manifest(
        mlp_source_root / "predictions",
        snapshot,
        "geo_helpfulness.router_mlp_prediction_manifest.v1",
        mlp["declaration_sha256"],
        mlp_prediction_files,
    )
    _files(
        capacity_root,
        {
            "declaration.json",
            "predictions/manifest.json",
            "scores/manifest.json",
            *["predictions/" + name for name in cap_prediction_files],
            *["scores/" + name for name in SCORE_FILES],
        },
    )
    _files(
        mlp_source_root,
        {
            "declaration.json",
            "predictions/manifest.json",
            *["predictions/" + name for name in mlp_prediction_files],
        },
    )
    if (
        _path(mlp["capacity_root"]) != capacity_root
        or mlp["capacity_prediction_manifest_sha256"]
        != cap_prediction["manifest_sha256"]
        or mlp["capacity_score_manifest_sha256"] != cap_score["manifest_sha256"]
    ):
        raise ValueError("MLP capacity predecessor mismatch")
    recovery = _json(mlp_replay_root / "declaration.json", snapshot)
    legacy.check_self_hash(recovery, "declaration_sha256")
    if (
        recovery.get("schema_version")
        != "geo_helpfulness.router_mlp_replay_declaration.v2"
        or recovery.get("study_id") != "router_mlp_v1"
        or recovery.get("recovery_id") != "router_mlp_v1_replay_v2"
        or _path(recovery["source_root"]) != mlp_source_root
        or recovery.get("source_declaration_sha256") != mlp["declaration_sha256"]
        or recovery.get("source_prediction_manifest_sha256")
        != mlp_prediction["manifest_sha256"]
        or any(
            recovery.get(key) is not False
            for key in (
                "source_mutations_permitted",
                "fitting_permitted",
                "scientific_selection_rules_changed",
            )
        )
    ):
        raise ValueError("MLP recovery declaration lineage mismatch")
    _merge(snapshot, recovery["parent_file_sha256"])
    mlp_score = _manifest(
        mlp_replay_root / "scores",
        snapshot,
        "geo_helpfulness.router_mlp_replay_score_manifest.v2",
        recovery["declaration_sha256"],
        MLP_SCORE_FILES,
    )
    _files(
        mlp_replay_root,
        {
            "declaration.json",
            "scores/manifest.json",
            *["scores/" + name for name in MLP_SCORE_FILES],
        },
    )
    if mlp_score.get("prediction_manifest_sha256") != mlp_prediction["manifest_sha256"]:
        raise ValueError("MLP recovery score prediction parent mismatch")
    cap_frozen, cap_receipt = _selection(
        capacity_root, snapshot, capacity, cap_prediction
    )
    mlp_frozen, mlp_receipt = _selection(
        mlp_replay_root, snapshot, mlp, mlp_prediction, recovery=recovery
    )
    if (
        cap_score.get("prediction_manifest_sha256") != cap_prediction["manifest_sha256"]
        or cap_score.get("selection_receipt_sha256") != cap_receipt["receipt_sha256"]
        or mlp_score.get("selection_receipt_sha256") != mlp_receipt["receipt_sha256"]
        or mlp_score.get("source_declaration_sha256") != mlp["declaration_sha256"]
    ):
        raise ValueError("score manifest prediction/receipt lineage mismatch")
    if (
        mlp_receipt.get("capacity_score_manifest_sha256")
        != cap_score["manifest_sha256"]
    ):
        raise ValueError("MLP selection capacity parent mismatch")
    states = {}
    for frozen, configs, source, declaration, prefix, learners in (
        (
            cap_frozen,
            capacity_configs,
            capacity_root,
            capacity,
            "capacity__",
            ("rf", "xgb"),
        ),
        (mlp_frozen, mlp_configs, mlp_source_root, mlp, "", ("mlp",)),
    ):
        expected_groups = {
            f"{learner}__{variant}"
            for learner in (("lr", "rf", "xgb") if prefix else learners)
            for variant in VARIANTS
        }
        if set(frozen["groups"]) != expected_groups:
            raise ValueError("selected source policy group set mismatch")
        for learner in learners:
            for variant in VARIANTS:
                group = f"{learner}__{variant}"
                selected = frozen["groups"][group]
                spec = configs[selected["config_id"]]
                if spec["family"] != learner or spec["variant"] != variant:
                    raise ValueError("selected configuration group mismatch")
                policy_id = prefix + group
                states[policy_id], models = {}, {}
                for seed in SEEDS:
                    model = selected["models"][str(seed)]
                    expected_path = (
                        source
                        / "predictions/models"
                        / f"{spec['config_id']}__seed_{seed}.json"
                    )
                    path = _path(
                        (capacity_root / "scores" / model["path"])
                        if prefix
                        else model["path"]
                    )
                    if path != expected_path or (
                        not prefix and not Path(model["path"]).is_absolute()
                    ):
                        raise ValueError(
                            "selected model path does not identify the sealed source model"
                        )
                    record = _json(path, snapshot, model["file_sha256"])
                    legacy.check_self_hash(record, "record_sha256")
                    state = record["state"]
                    (mlp_state if learner == "mlp" else capacity_state).validate_state(
                        state
                    )
                    parameters = dict(spec["parameters"])
                    if learner != "mlp":
                        parameters["random_state"] = seed
                    if (
                        record.get("declaration_sha256")
                        != declaration["declaration_sha256"]
                        or any(
                            state.get(key) != spec[key]
                            for key in ("config_id", "family", "variant")
                        )
                        or state["parameters"] != parameters
                        or state["training_seed"] != seed
                        or state["fit_row_count"] != 3378
                    ):
                        raise ValueError(
                            "selected portable model identity/parameters mismatch"
                        )
                    states[policy_id][str(seed)] = state
                    models[str(seed)] = {
                        "path": str(path),
                        "file_sha256": model["file_sha256"],
                        "record_sha256": record["record_sha256"],
                        "state_sha256": state["state_sha256"],
                    }
                policies[policy_id] = {
                    "learner": learner,
                    "variant_id": variant,
                    "config_id": spec["config_id"],
                    "C_hex": "",
                    "threshold_specification": deepcopy(
                        selected["threshold_specification"]
                    ),
                    "models": models,
                }
    legacy.check_snapshot(snapshot)
    _fit_incapable()
    return {
        "policies": {name: policies[name] for name in POLICIES},
        "states": states,
        "parent_file_sha256": snapshot,
        "lineage": {
            "study_id": STUDY_ID,
            "legacy_declaration_sha256": legacy_declaration["declaration_sha256"],
            "original_m4_manifest_sha256": m4_manifest["manifest_sha256"],
            "capacity_declaration_sha256": capacity["declaration_sha256"],
            "capacity_prediction_manifest_sha256": cap_prediction["manifest_sha256"],
            "capacity_score_manifest_sha256": cap_score["manifest_sha256"],
            "capacity_selection_receipt_sha256": cap_receipt["receipt_sha256"],
            "mlp_source_declaration_sha256": mlp["declaration_sha256"],
            "mlp_prediction_manifest_sha256": mlp_prediction["manifest_sha256"],
            "mlp_recovery_declaration_sha256": recovery["declaration_sha256"],
            "mlp_score_manifest_sha256": mlp_score["manifest_sha256"],
            "mlp_selection_receipt_sha256": mlp_receipt["receipt_sha256"],
            "selected_model_count": 36,
            "new_policy_count": 9,
            "fitting_performed": False,
            "selection_performed": False,
            "original_m4_no_go_preserved": True,
        },
    }


def _table(frame, schema):
    if list(frame.columns) != schema.names:
        raise ValueError(
            "prediction table must contain only its exact label-blind columns"
        )
    table = pa.Table.from_pandas(
        frame, schema=schema, preserve_index=False, safe=True
    ).replace_schema_metadata(None)
    if any(table[name].null_count for name in schema.names):
        raise ValueError("prediction tables must not contain null values")
    return table


def reconstruct_predictions(frames, selection, legacy_declaration):
    """Transfer exactly nine selected policies, preserving every old action row."""
    _fit_incapable()
    if (
        set(frames) != {"experts", "routers", "actions"}
        or set(selection["policies"]) != set(POLICIES)
        or set(selection["states"]) != set(NEW_POLICIES)
    ):
        raise ValueError("prediction frame or selected policy sets changed")
    if (
        selection["lineage"]["legacy_declaration_sha256"]
        != legacy_declaration["declaration_sha256"]
    ):
        raise ValueError("selected policies belong to another legacy declaration")
    root = _path(legacy_declaration["frozen_root"])
    snapshot = selection["parent_file_sha256"]
    feature_schema = _json(
        root / "router_feature_schema.json",
        {},
        snapshot[str(root / "router_feature_schema.json")],
    )
    tables = {
        name: _table(frame, legacy._schema(name, feature_schema))
        for name, frame in frames.items()
    }
    expected = legacy.reconstruct_predictions(
        frames["experts"][legacy.LOGIT_INPUT_COLUMNS], legacy_declaration
    )
    for name in tables:
        legacy.compare_tables(tables[name], expected[name])
    transforms = _json(
        root / "router_feature_transform.json",
        {},
        snapshot[str(root / "router_feature_transform.json")],
    )["states"]
    if len(transforms) != 4 or {state["seed"] for state in transforms} != set(SEEDS):
        raise ValueError("four seed-local frozen transforms required")
    transforms = {state["seed"]: state for state in transforms}
    routers, actions = [], [frames["actions"].copy()]
    for variant in VARIANTS:
        policy = selection["policies"][variant]
        rows = frames["routers"].loc[frames["routers"].variant_id.eq(variant)].copy()
        if (
            policy["learner"] != "lr"
            or policy["variant_id"] != variant
            or set(rows.C_hex) != {policy["C_hex"]}
        ):
            raise ValueError("legacy router metadata differs from selected policy")
        rows["policy_id"], rows["learner"], rows["config_id"] = (
            variant,
            "lr",
            policy["config_id"],
        )
        rows["schema_version"], rows["study_id"] = (
            "geo_helpfulness.legacy_capacity_routers.v1",
            STUDY_ID,
        )
        routers.append(rows[ROUTER_SCHEMA.names])
    for policy_id in legacy.POLICIES:
        rows = frames["actions"].loc[frames["actions"].policy_id.eq(policy_id)]
        expected_threshold = legacy.canonical_bytes(
            selection["policies"][policy_id]["threshold_specification"]
        ).decode()
        if set(rows.threshold_specification) != {expected_threshold}:
            raise ValueError("legacy action threshold differs from selected policy")
    for family in legacy.FAMILIES:
        for seed in SEEDS:
            experts = (
                frames["experts"]
                .loc[
                    frames["experts"].checkpoint_family.eq(family)
                    & frames["experts"].training_seed.eq(seed)
                ]
                .reset_index(drop=True)
            )
            matrix = legacy.apply_transform(experts, transforms[seed])
            for policy_id in NEW_POLICIES:
                policy = selection["policies"][policy_id]
                if set(selection["states"][policy_id]) != {
                    str(value) for value in SEEDS
                }:
                    raise ValueError("selected portable seed set mismatch")
                state = selection["states"][policy_id][str(seed)]
                if (
                    state["state_sha256"] != policy["models"][str(seed)]["state_sha256"]
                    or state["training_seed"] != seed
                    or state["family"] != policy["learner"]
                    or state["variant"] != policy["variant_id"]
                    or state["config_id"] != policy["config_id"]
                    or policy["C_hex"] != ""
                ):
                    raise ValueError(
                        "selected portable state differs from policy metadata"
                    )
                q = (
                    mlp_state if policy["learner"] == "mlp" else capacity_state
                ).predict_state(matrix, state)
                score = q[:, 0] - q[:, 1]
                base = experts[
                    ["checkpoint_family", "row_uid", "plot_idx", "training_seed"]
                ].copy()
                base["schema_version"], base["study_id"] = (
                    "geo_helpfulness.legacy_capacity_routers.v1",
                    STUDY_ID,
                )
                for key in ("learner", "variant_id", "config_id", "C_hex"):
                    base[key] = policy[key]
                base["policy_id"] = policy_id
                for index, name in enumerate(Q_COLUMNS):
                    base[name] = q[:, index]
                base["score"] = score
                routers.append(base[ROUTER_SCHEMA.names])
                acted = apply_heuristic_threshold(
                    score,
                    experts.raw_pred.to_numpy(),
                    experts.geo_pred.to_numpy(),
                    policy["threshold_specification"],
                )
                record = experts[
                    ["checkpoint_family", "row_uid", "plot_idx", "training_seed"]
                ].copy()
                record["schema_version"], record["study_id"] = (
                    "geo_helpfulness.legacy_capacity_actions.v1",
                    STUDY_ID,
                )
                record["policy_id"], record["score"] = policy_id, score
                record["threshold_specification"] = legacy.canonical_bytes(
                    policy["threshold_specification"]
                ).decode()
                record["acted"] = acted
                record["raw_pred"], record["geo_pred"] = (
                    experts.raw_pred,
                    experts.geo_pred,
                )
                record["final_pred"] = np.where(
                    acted, experts.geo_pred, experts.raw_pred
                )
                actions.append(record[legacy._schema("actions", feature_schema).names])
    sort = ["checkpoint_family", "policy_id", "row_uid", "training_seed"]
    router_frame = (
        pd.concat(routers, ignore_index=True).sort_values(sort).reset_index(drop=True)
    )
    action_frame = (
        pd.concat(actions, ignore_index=True).sort_values(sort).reset_index(drop=True)
    )
    _fit_incapable()
    return {
        "experts": tables["experts"],
        "routers": _table(router_frame, ROUTER_SCHEMA),
        "actions": _table(action_frame, legacy._schema("actions", feature_schema)),
    }

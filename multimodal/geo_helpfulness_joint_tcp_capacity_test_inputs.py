"""Authenticated, label-free legacy inputs for the joint router capacity study.

The archived encoder outputs and fixed epoch-50 heads are exactly those behind
the September 15 report. No model, temperature, transform or threshold is fitted
here, and no test label or test score artifact is decoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_router_candidate_state import (
    full_feature_columns, variant_feature_indices,
)
from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash, check_self_hash, check_snapshot, file_hash, read_json,
)
from multimodal.geo_helpfulness_router_numeric import (
    apply_expert_temperature, transform_router_features,
    validate_feature_transform_state, validate_temperature_state,
)
from multimodal.geo_helpfulness_targets_features import (
    CALIBRATED_PROBABILITY_BASIS, build_router_feature_frame,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
LEGACY = BASE / "exploratory_legacy_router_v1"
JOINT = BASE / "joint_tcp_router_v1"
RF = BASE / "relaxed_guardrails_rf_v1"
LR = BASE / "relaxed_guardrails_lr_v1"
SEEDS = (1, 2, 3, 4)
MODES = {"image_only": "image", "geo_only": "geo", "raw_concat": "raw"}
VARIANTS = {"numeric25": "ablation_numeric25", "no_pairs79": "ablation_no_pairs79"}
IDENTITY = ["row_uid", "plot_idx", "training_seed"]
EXPERT_COLUMNS = ["checkpoint_family", "row_uid", "file", "plot_idx", "training_seed",
                  *[f"{prefix}_logits" for prefix in MODES.values()],
                  *[f"{prefix}_pred" for prefix in MODES.values()]]
ACTION_COLUMNS = ["checkpoint_family", *IDENTITY, "policy_id", "score",
                  "threshold_specification", "acted", "raw_pred", "geo_pred", "final_pred"]


def _record(snapshot, path, expected=None):
    path = Path(path).resolve()
    digest = file_hash(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Legacy input source hash differs: {path}")
    if str(path) in snapshot and snapshot[str(path)] != digest:
        raise ValueError("Conflicting source fingerprints")
    snapshot[str(path)] = digest
    return digest


def _manifest(folder, snapshot, declaration, *, legacy=False):
    manifest = read_json(folder / "manifest.json")
    check_self_hash(manifest)
    if legacy:
        if (manifest.get("schema_version") != "geo_helpfulness.legacy_transfer_predictions.v1"
                or manifest.get("declaration_file_sha256") != file_hash(folder.parent / "declaration.json")
                or manifest.get("source_snapshot") != declaration["source_snapshot"]):
            raise ValueError("Legacy predictions have different declaration lineage")
    elif (manifest.get("declaration_sha256") != declaration["declaration_sha256"]
          or manifest.get("study_id") != declaration["study_id"]):
        raise ValueError("Historical comparator declaration lineage differs")
    _record(snapshot, folder / "manifest.json")
    for name, record in manifest["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("Unexpected nested prediction or selection artifact")
        _record(snapshot, folder / name, record["file_sha256"])
    return manifest


def _preprocessing(repo_root, parent_root, snapshot):
    declaration = read_json(parent_root / "declaration.json")
    if declaration.get("config", {}).get("study_id") != "joint_tcp_router_v1":
        raise ValueError("Expected the original joint router parent")
    _record(snapshot, parent_root / "declaration.json")
    states = {}
    for filename, schema in (
        ("expert_temperatures.json", "geo_helpfulness.expert_temperatures.v1"),
        ("router_feature_transform.json", "geo_helpfulness.router_feature_transform.v1"),
    ):
        candidates = [name for name in declaration["file_sha256"] if name.endswith("/" + filename)]
        if len(candidates) != 1:
            raise ValueError("Parent preprocessing source is missing or ambiguous")
        name = candidates[0]
        path = repo_root / name
        _record(snapshot, path, declaration["file_sha256"][name])
        value = read_json(path)
        if (set(value) != {"schema_version", "protocol_id", "fit_role", "states"}
                or value["schema_version"] != schema or value["protocol_id"] != "protocol_v1"
                or value["fit_role"] != "development_train_oof"):
            raise ValueError("Parent preprocessing fit scope differs")
        states[filename] = value["states"]
    temperatures = {}
    for state in states["expert_temperatures.json"]:
        validate_temperature_state(state)
        key = (state["seed"], state["mode"])
        if key in temperatures:
            raise ValueError("Duplicate temperature state")
        temperatures[key] = state
    transforms = {}
    for state in states["router_feature_transform.json"]:
        validate_feature_transform_state(state)
        if state["seed"] in transforms:
            raise ValueError("Duplicate transform state")
        transforms[state["seed"]] = state
    if (set(temperatures) != {(s, m) for s in SEEDS for m in MODES}
            or set(transforms) != set(SEEDS)):
        raise ValueError("Incomplete parent preprocessing seed grid")
    return {str(seed): {"temperatures": {mode: temperatures[seed, mode] for mode in MODES},
                        "transform": transforms[seed]} for seed in SEEDS}


def authenticate_sources(*, repo_root=ROOT, legacy_root=LEGACY, parent_root=JOINT,
                         rf_root=RF, lr_root=LR):
    """Return a JSON-serializable source context without decoding test tables."""
    repo_root, legacy_root, parent_root = map(Path, (repo_root, legacy_root, parent_root))
    declaration = read_json(legacy_root / "declaration.json")
    check_self_hash(declaration, "declaration_sha256")
    if (declaration.get("study_id") != "legacy_frozen_router_transfer_v1"
            or declaration.get("no_go_preserved") is not True):
        raise ValueError("Expected the frozen legacy expert family")
    snapshot = {}
    _record(snapshot, legacy_root / "declaration.json")
    from multimodal.geo_helpfulness_legacy_router_inference import _locked_identity
    identity = _locked_identity(declaration["identity_manifest"])
    checkpoints = [record for record in declaration["checkpoints"]
                   if record["checkpoint_family"] == "best_model"]
    if ({(c["training_seed"], c["mode"]) for c in checkpoints}
            != {(s, m) for s in SEEDS for m in MODES} or len(checkpoints) != 12):
        raise ValueError("Expected twelve fixed-final legacy checkpoints")
    for checkpoint in checkpoints:
        seed, mode = checkpoint["training_seed"], checkpoint["mode"]
        run = Path(declaration["legacy_runs"][mode][str(seed)])
        path = run / "best_model.pt"
        if (checkpoint["path"] != str(path) or checkpoint["epoch"] != 50
                or checkpoint["training_regime"] != "fixed_epoch_train_val"
                or declaration["source_snapshot"].get(str(path)) != checkpoint["file_sha256"]):
            raise ValueError("Fixed-final checkpoint provenance differs")
        _record(snapshot, path, checkpoint["file_sha256"])
        scaler = run / "geo_standardization.json"
        if declaration["source_snapshot"].get(str(scaler)) != declaration["scaler_file_sha256"]:
            raise ValueError("Legacy scaler lineage differs")
        _record(snapshot, scaler, declaration["scaler_file_sha256"])
    for paths in declaration["caches"].values():
        for name in ("train", "val", "test"):
            path = paths[name]
            _record(snapshot, path, declaration["source_snapshot"][path])
    prediction = _manifest(legacy_root / "predictions", snapshot, declaration, legacy=True)
    preprocessing = _preprocessing(repo_root, parent_root, snapshot)
    comparators = {}
    for family, root, prefix in (("rf", Path(rf_root), "relaxed__rf__"),
                                  ("lr", Path(lr_root), "relaxed__")):
        old = read_json(root / "declaration.json")
        check_self_hash(old, "declaration_sha256")
        if (old["identity_manifest"] != declaration["identity_manifest"]
                or old["ontology"] != declaration["ontology"]
                or old["study_id"] != f"relaxed_guardrails_{family}_v1"):
            raise ValueError("Historical comparator identity or ontology differs")
        _record(snapshot, root / "declaration.json")
        selection = _manifest(root / "selection", snapshot, old)
        predictions = _manifest(root / "predictions", snapshot, old)
        if (predictions["selection_manifest_sha256"] != selection["manifest_sha256"]
                or predictions["artifacts"]["experts.parquet"]["file_sha256"]
                != prediction["artifacts"]["experts.parquet"]["file_sha256"]):
            raise ValueError("Historical comparator used different experts or policies")
        policies = read_json(root / "selection/frozen_policies.json")
        for variant in ("primary_full727", "ablation_no_pairs79", "ablation_numeric25"):
            policy_id = prefix + variant + "__plots_20__harm_0p7"
            state = policies[policy_id]
            if (state["policy_id"] != policy_id or state["variant_id"] != variant
                    or state["minimum_plots"] != 20 or state["harm_numerator"] != 7
                    or state["harm_denominator"] != 10
                    or state["minimum_coverage_fraction"] != 0.01):
                raise ValueError("Historical comparator guardrail changed")
            comparators[policy_id] = {
                "family": family, "input_id": variant.split("_", 1)[1],
                "guardrail_id": "plots_20__harm_0p7", "frozen_policy": state,
                "actions_path": str(root / "predictions/actions.parquet"),
                "source_file_sha256": {
                    str(root / "predictions/actions.parquet"): predictions["artifacts"]["actions.parquet"]["file_sha256"],
                    str(root / "selection/frozen_policies.json"): selection["artifacts"]["frozen_policies.json"]["file_sha256"],
                },
            }
    context = {
        "schema_version": "geo_helpfulness.joint_tcp_capacity_test_inputs.v1",
        "legacy_declaration": declaration, "identity_manifest": declaration["identity_manifest"],
        "ontology": declaration["ontology"], "registry_root": declaration["registry_root"],
        "source_file_sha256": dict(sorted(snapshot.items())), "preprocessing": preprocessing,
        "experts_path": str(legacy_root / "predictions/experts.parquet"),
        "historical_policies": comparators, "test_rows": len(identity),
        "test_plots": identity.plot_idx.nunique(), "checkpoint_family": "best_model",
        "encoder_lineage": {
            "source": "exact_archived_historical_habitat_adapted_encoder_outputs",
            "new_encoder_extraction": False, "additional_image_normalization": False,
            "historical_encoder_to_cache_content_fingerprint_available": False,
            "original_encoder_path": str(repo_root / "model_ckpt/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_5_20260203_17.pt"),
            "limitation": "Archived caches and heads are authenticated; the original artifacts do not bind an end-to-end encoder-to-cache fingerprint.",
        },
    }
    context["context_sha256"] = canonical_hash(context)
    return context


def _context(value):
    value = authenticate_sources() if value is None else value
    check_self_hash(value, "context_sha256")
    if (value.get("schema_version") != "geo_helpfulness.joint_tcp_capacity_test_inputs.v1"
            or value.get("checkpoint_family") != "best_model"):
        raise ValueError("Wrong legacy test input context")
    check_snapshot(value["source_file_sha256"])
    return value


def _expert_rows(context, seed=None):
    filters = [("checkpoint_family", "=", "best_model")]
    if seed is not None:
        filters.append(("training_seed", "=", seed))
    rows = pd.read_parquet(context["experts_path"], columns=EXPERT_COLUMNS, filters=filters)
    rows = rows.sort_values(["training_seed", "row_uid"], kind="stable").reset_index(drop=True)
    expected = pd.DataFrame(context["identity_manifest"]["rows"]).sort_values("row_uid").reset_index(drop=True)
    if set(rows.training_seed) != ({seed} if seed is not None else set(SEEDS)):
        raise ValueError("Legacy expert seed coverage differs")
    for _, part in rows.groupby("training_seed", sort=True):
        if (len(part) != len(expected) or part.row_uid.duplicated().any()
                or part[["row_uid", "file", "plot_idx"]].reset_index(drop=True).to_dict("list")
                != expected[["row_uid", "file", "plot_idx"]].to_dict("list")):
            raise ValueError("Legacy expert rows differ from locked identity")
    for prefix in MODES.values():
        logits = np.stack(rows[f"{prefix}_logits"])
        if (logits.shape != (len(rows), 18) or logits.dtype != np.float64
                or not np.isfinite(logits).all()
                or not np.array_equal(logits.argmax(axis=1), rows[f"{prefix}_pred"].to_numpy())):
            raise ValueError("Legacy expert logits or predicted classes differ")
    return rows


def _replay_seed(seed, rows, context):
    import torch
    from multimodal.geo_helpfulness_legacy_router_inference import (
        GEO_COLUMNS, IMAGE_COLUMNS, _head_logits, _identity, _load_head, _load_scaler,
    )
    declaration = context["legacy_declaration"]
    frame = pd.read_parquet(declaration["caches"][str(seed)]["test"],
                            columns=["file", "plot_idx", *IMAGE_COLUMNS, *GEO_COLUMNS])
    identity = _identity(frame)
    if (len(identity) != len(rows) or set(identity.row_uid) != set(rows.row_uid)
            or dict(zip(identity.row_uid, identity.plot_idx)) != dict(zip(rows.row_uid, rows.plot_idx))):
        raise ValueError("Archived cache identity differs from saved predictions")
    frame.index = identity.row_uid
    frame = frame.loc[rows.row_uid].reset_index(drop=True)
    image = frame[list(IMAGE_COLUMNS)].to_numpy(dtype=np.float32)
    geo = frame[list(GEO_COLUMNS)].to_numpy(dtype=np.float32)
    if not np.isfinite(image).all() or not np.isfinite(geo).all():
        raise ValueError("Nonfinite archived expert inputs")
    previous_threads = torch.get_num_threads()
    result = {}
    try:
        torch.set_num_threads(1)
        for mode, prefix in MODES.items():
            run = Path(declaration["legacy_runs"][mode][str(seed)])
            scaler = _load_scaler(run / "geo_standardization.json", context["ontology"])
            standardized = (geo - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)
            values = image if mode == "image_only" else standardized
            if mode == "raw_concat":
                values = np.concatenate([image, standardized], axis=1)
            actual = _head_logits(_load_head(run / "best_model.pt", mode, "best_model"), values)
            expected = np.stack(rows[f"{prefix}_logits"])
            if not np.array_equal(actual, expected):
                raise ValueError(f"Native float32 archived head replay is not exact: {mode}/seed{seed}")
            if not np.array_equal(actual.argmax(axis=1), rows[f"{prefix}_pred"].to_numpy()):
                raise ValueError("Archived expert class replay differs")
            result[mode] = {"native_float32_logits_exact": True, "predicted_classes_exact": True,
                            "max_absolute_logit_difference": float(np.max(np.abs(actual - expected)))}
    finally:
        torch.set_num_threads(previous_threads)
    return result


def prepare_seed(seed, *, context=None, preprocessing=None):
    """Return exact legacy logits and frozen 25/79-column inputs for one seed.

    ``test`` is an NPZ-compatible mapping; ``metadata`` records source lineage
    and mandatory exact archived-head replay. No targets are returned.
    """
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Expected seed 1, 2, 3 or 4")
    context = _context(context)
    states = context["preprocessing"][str(seed)]
    if preprocessing is not None and preprocessing != states:
        raise ValueError("Test preprocessing differs from parent joint router fitting")
    rows = _expert_rows(context, seed)
    replay = _replay_seed(seed, rows, context)
    probabilities = [apply_expert_temperature(np.stack(rows[f"{prefix}_logits"]),
                        states["temperatures"][mode]) for mode, prefix in MODES.items()]
    features = build_router_feature_frame(*probabilities, probability_basis=CALIBRATED_PROBABILITY_BASIS)
    base = np.ascontiguousarray(transform_router_features(features, states["transform"]), dtype=np.float64)
    if base.shape != (len(rows), 727) or not np.isfinite(base).all():
        raise ValueError("Invalid frozen legacy numerical matrix")
    payload = {name: rows[name].to_numpy(dtype=str) for name in ("row_uid", "file", "plot_idx")}
    payload.update(training_seed=np.full(len(rows), seed, dtype=np.int64),
                   producer_fold=np.full(len(rows), -1, dtype=np.int64), X_base727=base)
    names = {}
    for input_id, variant in VARIANTS.items():
        indices = variant_feature_indices(variant)
        payload[f"X_{input_id}"] = np.ascontiguousarray(base[:, indices], dtype=np.float32)
        names[input_id] = [full_feature_columns()[i] for i in indices]
    for prefix in MODES.values():
        payload[f"{prefix}_logits"] = np.stack(rows[f"{prefix}_logits"])
    metadata = {
        "schema_version": context["schema_version"], "training_seed": seed,
        "checkpoint_family": "best_model", "producer": "legacy_full_development_fixed_epoch50",
        "context_sha256": context["context_sha256"], "test_rows": len(rows),
        "test_plots": rows.plot_idx.nunique(), "test_labels_read": False,
        "source_file_sha256": context["source_file_sha256"],
        "encoder_lineage": context["encoder_lineage"], "native_replay": replay,
        "feature_names": names, "preprocessing_sha256": canonical_hash(states),
        "preprocessing_refit": False, "historical_temperature_transfer": True,
    }
    check_snapshot(context["source_file_sha256"])
    return {"test": payload, "metadata": metadata}


def load_historical_actions(*, context=None):
    """Return six frozen September 15 policies as label-free wide action columns."""
    import json
    context = _context(context)
    experts = _expert_rows(context).sort_values(IDENTITY, kind="stable").reset_index(drop=True)
    wide = experts[[*IDENTITY, "raw_pred", "geo_pred"]].copy()
    source_frames = {}
    for policy_id, metadata in context["historical_policies"].items():
        path = metadata["actions_path"]
        if path not in source_frames:
            source_frames[path] = pd.read_parquet(path, columns=ACTION_COLUMNS,
                                                  filters=[("checkpoint_family", "=", "best_model")])
        part = source_frames[path].loc[source_frames[path].policy_id.eq(policy_id)].sort_values(IDENTITY, kind="stable").reset_index(drop=True)
        if (part[[*IDENTITY, "raw_pred", "geo_pred"]].to_dict("list") != wide[[*IDENTITY, "raw_pred", "geo_pred"]].to_dict("list")
                or part.acted.dtype != bool or not np.isfinite(part.score).all()):
            raise ValueError("Historical comparator rows or expert predictions differ")
        threshold = metadata["frozen_policy"]["threshold_specification"]
        if any(json.loads(value) != threshold for value in part.threshold_specification):
            raise ValueError("Historical action threshold differs from frozen policy")
        if threshold["comparison"] != "strict_gt":
            raise ValueError("Historical threshold must use strict comparison")
        if threshold["kind"] == "never_intervene" and threshold["value_hex"] is None:
            acted = np.zeros(len(part), dtype=bool)
        elif threshold["kind"] == "finite":
            value = float.fromhex(threshold["value_hex"])
            if not np.isfinite(value):
                raise ValueError("Nonfinite historical threshold")
            acted = part.geo_pred.ne(part.raw_pred).to_numpy() & (part.score.to_numpy() > value)
        else:
            raise ValueError("Unknown historical threshold")
        if (not np.array_equal(acted, part.acted.to_numpy())
                or not np.array_equal(np.where(acted, part.geo_pred, part.raw_pred), part.final_pred.to_numpy())):
            raise ValueError("Historical policy action replay differs")
        wide[policy_id] = acted
    if len(context["historical_policies"]) != 6:
        raise ValueError("Expected all six September 15 comparator policies")
    check_snapshot(context["source_file_sha256"])
    return {"actions": wide, "metadata": context["historical_policies"]}

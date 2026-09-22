"""Approved pooled-OOF confidence study with matched numerical RF routers."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet import (
    _hash, _json, _load_arrays, _lock, _now, _save_arrays, _seal_stage,
    _staging, _validate_stage, _write_json,
)
from multimodal.geo_helpfulness_router_candidate_state import (
    NUMERIC_INPUT_COLUMNS, TARGET_ORDER,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1"
OUTPUT = PROTOCOL.parent / "confidnet_oof_v1"
CONFIG = PROJECT_ROOT / "configs/multimodal_geo_confidnet_oof_v1.json"
PLAN = PROJECT_ROOT / "dev_plans/reliability_aware_geo_fusion/13_confidnet_oof_proposal.md"
SEEDS = (1, 2, 3, 4)
MODES = ("image", "geo", "raw")
FOLDS = (0, 1, 2, 3)
ARMS = ("baseline", "augmented")
NUMERIC_COLUMNS = tuple(f"scaled__{name}" for name in NUMERIC_INPUT_COLUMNS)
CODE_PATHS = (
    "multimodal/geo_helpfulness_confidnet_oof.py",
    "multimodal/geo_helpfulness_confidnet_oof_extract.py",
    "multimodal/geo_helpfulness_confidnet_oof_fit.py",
    "multimodal/geo_helpfulness_confidnet_oof_report.py",
    "tools/run_multimodal_geo_helpfulness_confidnet_oof.py",
)


class StudyPausedError(RuntimeError):
    """The approved training-failure rule requires review before further fitting."""


def _output(value):
    root = Path(value).resolve()
    if root == PROTOCOL or PROTOCOL in root.parents or root in PROTOCOL.parents:
        raise ValueError("Study output must be disjoint from the frozen protocol")
    for name in ("confidnet_initial_v1", "router_capacity_v1"):
        protected = PROTOCOL.parent / name
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError("Study output would overwrite an existing experiment")
    return root


def _config():
    from multimodal.geo_helpfulness_confidnet_oof_fit import (
        CONFIDENCE_DEFAULTS, RF_DEFAULTS,
    )
    cfg = _json(CONFIG)
    expected = {
        "schema_version": "geo_helpfulness.confidnet_oof_config.v1",
        "study_id": "confidnet_oof_v1", "training_seeds": list(SEEDS),
        "expert_modes": list(MODES), "fold_ids": list(FOLDS),
        "confidence_input_dim": 256,
        "confidence_feature": "paired_frozen_expert_penultimate_relu",
        "confidence_supervision": "pooled_expert_oof_development_train",
        "router_confidence_inputs": "in_sample_confidnet_predictions_on_expert_oof_features",
        "target": "native_t1_true_class_probability",
        "router_base_variant": "ablation_numeric25",
        "router_augmented_features": [f"{mode}_tcp_hat" for mode in MODES],
        "target_order": list(TARGET_ORDER),
        "confidence_parameters": CONFIDENCE_DEFAULTS,
        "router_parameters": RF_DEFAULTS,
        "policy_comparison": "strict_greater_than_zero_on_geo_fusion_disagreements",
        "policies": ["fusion", "mcp_difference_positive", "tcp_difference_positive", "rf_baseline", "rf_augmented"],
        "bootstrap": {"resamples": 2000, "seed": 20260826,
                      "unit": "whole_plot_all_seed_realizations", "stratified": False,
                      "interval": "percentile_95"},
        "evidence_status": "exploratory_reused_development_validation",
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Configuration differs from approved setup: {key}")
    for key in ("threshold_selection", "guardrail_selection", "expert_updates",
                "validation_selection", "locked_test_access", "optional_direct_embeddings",
                "optional_in_sample_control", "optional_true_tcp_router",
                "optional_confidence_cross_fitting"):
        if cfg.get(key) is not False:
            raise ValueError(f"Unapproved study expansion: {key}")
    return cfg


def _relative(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def _source_files(inputs):
    paths = {Path(path) for path in inputs["parent_file_hashes"]}
    paths.update(PROJECT_ROOT / name for name in CODE_PATHS)
    # Include the imported research pipeline implementations as well as direct sources.
    paths.update((PROJECT_ROOT / "multimodal").glob("*.py"))
    paths.update((CONFIG, PROJECT_ROOT / "configs/multimodal_geo_helpfulness.yaml"))
    paths.update((PROTOCOL / "protocol").glob("*"))
    for seed in SEEDS:
        for directory in [PROTOCOL / f"development_validation/seed_{seed}", *[
            PROTOCOL / f"development_train_oof/seed_{seed}/fold_{fold}" for fold in FOLDS
        ]]:
            paths.update(directory.glob("*"))
    return sorted(path for path in paths if path.is_file() and path.suffix != ".pyc")


def declare(output_dir=OUTPUT):
    """Validate immutable parents and seal all approved settings before fitting."""
    root = _output(output_dir)
    with _lock(root, "declaration"):
        if (root / "declaration.json").exists():
            validate_declaration(root, check_parents=True)
            return {"status": "reused_valid", "path": str(root / "declaration.json")}
        cfg = _config()
        if any(not (PROJECT_ROOT / name).is_file() for name in CODE_PATHS):
            raise ValueError("Study implementation is incomplete")
        from multimodal import geo_helpfulness_oof as m2
        from multimodal.geo_helpfulness_router_capacity_inputs import load_capacity_inputs
        context = m2.load_frozen_context()
        for seed in SEEDS:
            for spec in m2._producer_specs(seed):
                m2.validate_producer(context, spec, seed=seed)
        inputs = load_capacity_inputs(PROJECT_ROOT, PROTOCOL)
        files = _source_files(inputs)
        hashes = {_relative(path): _hash(path) for path in files}
        for path, expected in inputs["parent_file_hashes"].items():
            if hashes[_relative(path)] != expected:
                raise ValueError("Authenticated router parent changed during declaration")
        value = {
            "schema_version": "geo_helpfulness.confidnet_oof_declaration.v1",
            "created_at": _now(), "config": cfg, "file_sha256": hashes,
            "approved_plan_text": PLAN.read_text(),
            "approval_scope": "12 pooled-OOF confidence heads, 8 numeric25/28 RF routers, fixed policies",
            "environment": {name: version(name) for name in
                            ("numpy", "pandas", "pyarrow", "torch", "scikit-learn")},
            "locked_test_access": False, "validation_selection": False,
            "numeric_columns": list(NUMERIC_COLUMNS),
        }
        _write_json(root / "declaration.json", value)
        return {"status": "declared", "source_files": len(files),
                "sha256": _hash(root / "declaration.json"), "path": str(root)}


def validate_declaration(output_dir=OUTPUT, *, check_parents=False):
    root = _output(output_dir)
    value = _json(root / "declaration.json")
    if (value.get("schema_version") != "geo_helpfulness.confidnet_oof_declaration.v1"
            or value.get("config") != _config()
            or value.get("numeric_columns") != list(NUMERIC_COLUMNS)):
        raise ValueError("Approved declaration or numerical feature order changed")
    paths = value["file_sha256"]
    mandatory = set(CODE_PATHS) | {_relative(CONFIG)}
    if not mandatory <= set(paths):
        raise ValueError("Declaration is missing implementation hashes")
    for relative, digest in paths.items():
        if check_parents or relative.endswith(".py") or relative == _relative(CONFIG):
            if _hash(PROJECT_ROOT / relative) != digest:
                raise ValueError(f"Declared source changed: {relative}")
    return value


def _check_paused(root):
    pauses = list((root / "pauses").glob("seed_*/*/manifest.json"))
    if pauses:
        for path in pauses:
            _validate_stage(path.parent, root, role="training_pause")
        raise StudyPausedError(f"Training requires user review; saved pause: {pauses[0].parent}")


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Training seed must be 1, 2, 3 or 4")
    return seed


def _feature_check(train, validation):
    from multimodal.geo_helpfulness_confidnet import _check_arrays
    from multimodal.geo_helpfulness_confidnet_oof_extract import validate_split_payload
    stripped = [{key: val for key, val in data.items() if key != "producer_fold"}
                for data in (train, validation)]
    _check_arrays(*stripped)
    if set(train["row_uid"]) & set(validation["row_uid"]):
        raise ValueError("Training and validation image identities overlap")
    for role, data in (("train", train), ("validation", validation)):
        validate_split_payload(data, role)
        for mode in MODES:
            if np.any(data[f"{mode}_hidden"] < 0):
                raise ValueError("Penultimate ReLU features cannot be negative")
        folds = data["producer_fold"]
        if folds.dtype != np.int64 or folds.shape != data["row_uid"].shape:
            raise ValueError("Invalid producer fold identity")
        if data["row_uid"].tolist() != sorted(data["row_uid"].tolist()):
            raise ValueError("Feature row identities must be canonically sorted")
        if role == "train":
            if set(folds.tolist()) != set(FOLDS):
                raise ValueError("OOF features omit a producer fold")
            for plot in np.unique(data["plot_idx"]):
                if len(np.unique(folds[data["plot_idx"] == plot])) != 1:
                    raise ValueError("A plot crosses OOF producer folds")
        elif np.any(folds != -1):
            raise ValueError("Validation must use the full-development-train expert")


def extract_seed(seed, output_dir=OUTPUT, *, device="cuda"):
    seed, root = _seed(seed), _output(output_dir)
    validate_declaration(root)
    _check_paused(root)
    from multimodal.geo_helpfulness_confidnet_oof_extract import (
        extract_producer, combine_producers,
    )
    with _lock(root, f"extract_seed_{seed}"):
        final = root / f"features/seed_{seed}"
        if final.exists():
            _validate_features(root, seed)
            return {"status": "reused_valid", "seed": seed}
        results, parents = {}, {}
        for fold in (*FOLDS, None):
            key = "validation" if fold is None else f"fold_{fold}"
            destination = root / f"producers/seed_{seed}/{key}"
            if not destination.exists():
                result = extract_producer(seed, fold, device=device)
                staged = _staging(root, f"producer_{seed}_{key}")
                _save_arrays(staged / "features.npz", result["payload"])
                _write_json(staged / "extraction.json", result["metadata"])
                _seal_stage(staged, destination, root, role="frozen_oof_producer",
                            metadata={"seed": seed, "fold": fold})
            manifest = _validate_stage(destination, root, role="frozen_oof_producer")
            if manifest["metadata"] != {"seed": seed, "fold": fold}:
                raise ValueError("Cached producer seed/fold differs")
            results[fold] = {"payload": _load_arrays(destination / "features.npz"),
                             "metadata": _json(destination / "extraction.json")}
            parents[key] = _hash(destination / "manifest.json")
        result = combine_producers(seed, results)
        _feature_check(result["train"], result["validation"])
        staged = _staging(root, f"features_{seed}")
        for role in ("train", "validation"):
            _save_arrays(staged / f"{role}.npz", result[role])
        _write_json(staged / "extraction.json", result["metadata"])
        _seal_stage(staged, final, root, role="pooled_oof_features", metadata={
            "seed": seed, "producer_manifests": parents})
        return {"status": "complete", "seed": seed, "producers": 5}


def _validate_features(root, seed):
    path = root / f"features/seed_{seed}"
    manifest = _validate_stage(path, root, role="pooled_oof_features")
    if (manifest["metadata"]["seed"] != seed or set(manifest["file_sha256"]) !=
            {"train.npz", "validation.npz", "extraction.json"}):
        raise ValueError("Feature bundle schema/seed mismatch")
    expected = {f"fold_{fold}" for fold in FOLDS} | {"validation"}
    if set(manifest["metadata"]["producer_manifests"]) != expected:
        raise ValueError("Missing pooled OOF producer parent")
    for key, digest in manifest["metadata"]["producer_manifests"].items():
        producer = root / f"producers/seed_{seed}/{key}"
        _validate_stage(producer, root, role="frozen_oof_producer")
        if _hash(producer / "manifest.json") != digest:
            raise ValueError("Pooled OOF producer lineage changed")
    arrays = {role: _load_arrays(path / f"{role}.npz") for role in ("train", "validation")}
    _feature_check(arrays["train"], arrays["validation"])
    return arrays


def _pause(root, seed, mode, result, reason):
    staged = _staging(root, f"pause_{seed}_{mode}")
    _write_json(staged / "reason.json", {"reason": reason, "seed": seed, "mode": mode})
    if result is not None:
        for key in ("state", "history", "metadata"):
            if result.get(key) is not None:
                _write_json(staged / f"{key}.json", result[key])
    path = root / f"pauses/seed_{seed}/{mode}"
    _seal_stage(staged, path, root, role="training_pause", metadata={"seed": seed, "mode": mode})
    raise StudyPausedError(f"Approved training-failure stop: {reason}; details at {path}")


def fit_seed(seed, output_dir=OUTPUT):
    """Fit confidence heads only; no router fit can precede all twelve heads."""
    seed, root = _seed(seed), _output(output_dir)
    declaration = validate_declaration(root)
    _check_paused(root)
    from multimodal.geo_helpfulness_confidnet_oof_fit import (
        fit_confidence, ConfidNetOOFFitError,
    )
    with _lock(root, f"confidence_seed_{seed}"):
        arrays = _validate_features(root, seed)
        train, validation = arrays["train"], arrays["validation"]
        feature_hash = _hash(root / f"features/seed_{seed}/manifest.json")
        for mode in MODES:
            final = root / f"confidence/seed_{seed}/{mode}"
            if final.exists():
                _validate_head(root, seed, mode, arrays)
                continue
            print(f"seed {seed}: fitting {mode} confidence head, 500 epochs", flush=True)
            try:
                result = fit_confidence(
                    train[f"{mode}_hidden"], train[f"{mode}_logits"], train["labels"],
                    {"validation": validation[f"{mode}_hidden"]}, seed=seed,
                    parameters=declaration["config"]["confidence_parameters"],
                )
            except ConfidNetOOFFitError as error:
                _pause(root, seed, mode, getattr(error, "partial_result", None), str(error))
            if result["status"] != "complete":
                _pause(root, seed, mode, result, result["status"])
            staged = _staging(root, f"confidence_{seed}_{mode}")
            for key in ("state", "history", "metadata"):
                _write_json(staged / f"{key}.json", result[key])
            _save_arrays(staged / "predictions.npz", result["predictions"])
            _seal_stage(staged, final, root, role="confidence_head", metadata={
                "seed": seed, "mode": mode, "feature_manifest_sha256": feature_hash})
            _validate_head(root, seed, mode, arrays)
        _confidence_tables(root, seed, arrays)
        return {"status": "complete", "seed": seed, "confidence_models": 3}


def _validate_head(root, seed, mode, arrays):
    from multimodal.geo_helpfulness_confidnet_oof_fit import predict_confidence
    path = root / f"confidence/seed_{seed}/{mode}"
    manifest = _validate_stage(path, root, role="confidence_head")
    expected_meta = {"seed": seed, "mode": mode, "feature_manifest_sha256":
                     _hash(root / f"features/seed_{seed}/manifest.json")}
    if manifest["metadata"] != expected_meta or set(manifest["file_sha256"]) != {
        "state.json", "history.json", "metadata.json", "predictions.npz"
    }:
        raise ValueError("Confidence head seed/mode/feature lineage differs")
    state, history = _json(path / "state.json"), _json(path / "history.json")
    if (state["training_seed"] != seed or state["fit_row_count"] != 3378
            or state["input_dim"] != 256
            or state.get("status") != "complete" or state.get("completed_epochs") != 500
            or state["parameters"] != _config()["confidence_parameters"]):
        raise ValueError("Confidence head differs from the approved recipe")
    if [item["epoch"] for item in history] != list(range(1, 501)):
        raise ValueError("Confidence head did not complete the approved budget")
    streak = 0
    for item in history:
        streak = streak + 1 if item["maximum_gradient_l2_norm"] == 0 else 0
        if streak >= 5:
            raise ValueError("Confidence history bypasses the zero-gradient stop")
    predictions = _load_arrays(path / "predictions.npz")
    if set(predictions) != {"train", "validation"}:
        raise ValueError("Confidence prediction splits differ")
    for role, data in arrays.items():
        values = predictions[role]
        if values.dtype != np.float32 or values.shape != data["row_uid"].shape:
            raise ValueError("Native confidence prediction dtype/shape changed")
        native = predict_confidence(state, data[f"{mode}_hidden"], native=True)
        portable = predict_confidence(state, data[f"{mode}_hidden"], native=False)
        if not np.array_equal(values, native):
            raise ValueError("Native confidence replay does not reproduce saved predictions")
        if not np.allclose(values, portable, atol=1e-6, rtol=0):
            raise ValueError("Portable confidence replay exceeds approved tolerance")
    return predictions


def _confidence_tables(root, seed, arrays):
    final = root / f"confidence/seed_{seed}/bundle"
    parents = {mode: _hash(root / f"confidence/seed_{seed}/{mode}/manifest.json") for mode in MODES}
    if final.exists():
        manifest = _validate_stage(final, root, role="confidence_prediction_tables")
        if manifest["metadata"] != {"seed": seed, "head_manifests": parents}:
            raise ValueError("Confidence prediction table parents changed")
    else:
        tables = {role: pd.DataFrame({"row_uid": data["row_uid"], "plot_idx": data["plot_idx"],
                  "training_seed": seed, "producer_fold": data["producer_fold"]})
                  for role, data in arrays.items()}
        for mode in MODES:
            predictions = _validate_head(root, seed, mode, arrays)
            for role, data in arrays.items():
                tables[role][f"{mode}_tcp_hat"] = predictions[role]
                tables[role][f"{mode}_logits"] = list(data[f"{mode}_logits"])
        staged = _staging(root, f"confidence_tables_{seed}")
        for role, table in tables.items():
            table.to_parquet(staged / f"{role}.parquet", index=False)
        _seal_stage(staged, final, root, role="confidence_prediction_tables",
                    metadata={"seed": seed, "head_manifests": parents})
    return _validate_confidence_tables(root, seed, arrays)


def _validate_confidence_tables(root, seed, arrays):
    path = root / f"confidence/seed_{seed}/bundle"
    manifest = _validate_stage(path, root, role="confidence_prediction_tables")
    expected_meta = {"seed": seed, "head_manifests": {
        mode: _hash(root / f"confidence/seed_{seed}/{mode}/manifest.json") for mode in MODES}}
    if manifest["metadata"] != expected_meta:
        raise ValueError("Confidence prediction table parents changed")
    if set(manifest["file_sha256"]) != {"train.parquet", "validation.parquet"}:
        raise ValueError("Confidence table file schema differs")
    heads = {mode: _validate_head(root, seed, mode, arrays) for mode in MODES}
    expected_columns = {"row_uid", "plot_idx", "training_seed", "producer_fold"} | {
        f"{mode}_{suffix}" for mode in MODES for suffix in ("tcp_hat", "logits")}
    result = {}
    for role, data in arrays.items():
        table = pd.read_parquet(path / f"{role}.parquet")
        if set(table) != expected_columns or not table.training_seed.eq(seed).all():
            raise ValueError("Confidence table contains unexpected columns/seed")
        for name in ("row_uid", "plot_idx", "producer_fold"):
            if not np.array_equal(table[name].to_numpy(), data[name]):
                raise ValueError("Confidence table identity differs from feature cache")
        for mode in MODES:
            if not np.array_equal(table[f"{mode}_tcp_hat"].to_numpy(), heads[mode][role]):
                raise ValueError("Confidence table differs from native predictions")
            if not np.array_equal(np.stack(table[f"{mode}_logits"]), data[f"{mode}_logits"]):
                raise ValueError("Confidence table differs from frozen expert logits")
        result[role] = table
    return result


def _all_confidence(root):
    _check_paused(root)
    result = {}
    for seed in SEEDS:
        arrays = _validate_features(root, seed)
        result[seed] = _validate_confidence_tables(root, seed, arrays)
    return result


def _base_inputs(root, *, build=False):
    final = root / "router_inputs"
    if not final.exists():
        if not build:
            raise ValueError("Authenticated numerical router inputs are missing")
        from multimodal.geo_helpfulness_router_capacity_inputs import load_capacity_inputs
        inputs = load_capacity_inputs(PROJECT_ROOT, PROTOCOL)
        declaration = validate_declaration(root)
        for name, digest in inputs["parent_file_hashes"].items():
            if declaration["file_sha256"].get(_relative(name)) != digest:
                raise ValueError("Router parent was not authenticated at declaration")
        staged = _staging(root, "router_inputs")
        for seed in SEEDS:
            data = inputs["seed_inputs"][seed]
            payload = {"X_train": data["X_train"][:, :25], "y_train": data["y_train"],
                       "X_validation": data["X_validation"][:, :25]}
            for role in ("train", "validation"):
                identity = data[f"{role}_identity"]
                for name in ("row_uid", "plot_idx"):
                    payload[f"{role}_{name}"] = identity[name].to_numpy(dtype=str)
            _save_arrays(staged / f"seed_{seed}.npz", payload)
        _write_json(staged / "columns.json", list(NUMERIC_COLUMNS))
        _seal_stage(staged, final, root, role="authenticated_numeric_router_inputs",
                    metadata={"parent_file_sha256": inputs["parent_file_hashes"]})
    manifest = _validate_stage(final, root, role="authenticated_numeric_router_inputs")
    declared = _json(root / "declaration.json")["file_sha256"]
    for name, digest in manifest["metadata"]["parent_file_sha256"].items():
        if declared.get(_relative(name)) != digest:
            raise ValueError("Router input parent differs from declaration")
    if set(manifest["file_sha256"]) != {"columns.json", *(f"seed_{s}.npz" for s in SEEDS)}:
        raise ValueError("Unexpected numerical router input files")
    if _json(final / "columns.json") != list(NUMERIC_COLUMNS):
        raise ValueError("Numerical router column order changed")
    return {seed: _load_arrays(final / f"seed_{seed}.npz") for seed in SEEDS}


def _router_matrices(data, tables):
    result = {}
    for role, table in tables.items():
        for name in ("row_uid", "plot_idx"):
            if not np.array_equal(table[name].to_numpy(), data[f"{role}_{name}"]):
                raise ValueError("Confidence/router rows or plots do not align")
        base = data[f"X_{role}"]
        if base.shape != (len(table), 25) or not np.isfinite(base).all():
            raise ValueError("Invalid 25-column router matrix")
        extra = table[[f"{mode}_tcp_hat" for mode in MODES]].to_numpy(dtype=np.float64)
        result[role] = {"baseline": base, "augmented": np.column_stack((base, extra))}
    return result


def fit_routers(output_dir=OUTPUT):
    root = _output(output_dir)
    declaration = validate_declaration(root)
    # This check intentionally precedes both numerical-input loading and any RF fit.
    tables = _all_confidence(root)
    from multimodal.geo_helpfulness_confidnet_oof_fit import fit_router
    with _lock(root, "routers"):
        inputs = _base_inputs(root, build=True)
        for seed in SEEDS:
            matrices = _router_matrices(inputs[seed], tables[seed])
            labels = _load_arrays(root / f"features/seed_{seed}/train.npz")["labels"]
            geo = np.stack(tables[seed]["train"].geo_logits).argmax(axis=1) == labels
            raw = np.stack(tables[seed]["train"].raw_logits).argmax(axis=1) == labels
            targets = np.where(geo & ~raw, 0, np.where(~geo & raw, 1, np.where(geo & raw, 2, 3)))
            if not np.array_equal(targets, inputs[seed]["y_train"]):
                raise ValueError("Router targets differ from paired OOF expert correctness")
            for arm in ARMS:
                final = root / f"routers/seed_{seed}/{arm}"
                if final.exists():
                    _validate_router(root, seed, arm, matrices)
                    continue
                columns = list(NUMERIC_COLUMNS) + ([f"{mode}_tcp_hat" for mode in MODES] if arm == "augmented" else [])
                print(f"seed {seed}: fitting RF {arm} ({len(columns)} columns)", flush=True)
                result = fit_router(
                    matrices["train"][arm], targets,
                    {"validation": matrices["validation"][arm]}, seed=seed,
                    feature_names=columns, parameters=declaration["config"]["router_parameters"],
                )
                staged = _staging(root, f"router_{seed}_{arm}")
                _write_json(staged / "state.json", result["state"])
                _write_json(staged / "metadata.json", result["metadata"])
                _save_arrays(staged / "predictions.npz", result["predictions"])
                _seal_stage(staged, final, root, role="matched_numeric_rf", metadata={
                    "seed": seed, "arm": arm,
                    "confidence_bundle_sha256": _hash(root / f"confidence/seed_{seed}/bundle/manifest.json"),
                    "router_inputs_sha256": _hash(root / "router_inputs/manifest.json")})
                _validate_router(root, seed, arm, matrices)
        return {"status": "complete", "router_models": 8}


def _validate_router(root, seed, arm, matrices):
    from multimodal.geo_helpfulness_confidnet_oof_fit import predict_router
    path = root / f"routers/seed_{seed}/{arm}"
    manifest = _validate_stage(path, root, role="matched_numeric_rf")
    expected = {"seed": seed, "arm": arm,
                "confidence_bundle_sha256": _hash(root / f"confidence/seed_{seed}/bundle/manifest.json"),
                "router_inputs_sha256": _hash(root / "router_inputs/manifest.json")}
    if manifest["metadata"] != expected or set(manifest["file_sha256"]) != {
        "state.json", "metadata.json", "predictions.npz"
    }:
        raise ValueError("Router state lineage or artifact schema changed")
    state = _json(path / "state.json")
    columns = list(NUMERIC_COLUMNS) + ([f"{mode}_tcp_hat" for mode in MODES] if arm == "augmented" else [])
    if (state.get("training_seed") != seed or state.get("fit_row_count") != 3378
            or state.get("feature_names") != columns
            or state.get("parameters") != _config()["router_parameters"]):
        raise ValueError("Router state differs from declared seed, arm or recipe")
    predictions = _load_arrays(path / "predictions.npz")
    if set(predictions) != {"train", "validation"}:
        raise ValueError("Unexpected router prediction splits")
    for role in ("train", "validation"):
        replay = predict_router(state, matrices[role][arm])
        if not np.allclose(predictions[role], replay, atol=1e-12, rtol=0):
            raise ValueError("Saved RF probabilities do not reproduce portable state")
        if not np.array_equal(predictions[role][:, 0] > predictions[role][:, 1],
                              replay[:, 0] > replay[:, 1]):
            raise ValueError("RF strict-positive policy differs under portable replay")
    return predictions


def _all_predictions(root):
    tables = _all_confidence(root)
    inputs = _base_inputs(root)
    routers = {}
    for seed in SEEDS:
        matrices = _router_matrices(inputs[seed], tables[seed])
        routers[seed] = {arm: _validate_router(root, seed, arm, matrices) for arm in ARMS}
    return tables, routers


def _prediction_parents(root):
    return {_relative(path): _hash(path) for pattern in
            ("confidence/seed_*/*/manifest.json", "routers/seed_*/*/manifest.json")
            for path in root.glob(pattern)}


def score(output_dir=OUTPUT):
    """Open validation labels only after every fitted model/prediction is verified."""
    root = _output(output_dir)
    declaration = validate_declaration(root)
    with _lock(root, "score"):
        tables, routers = _all_predictions(root)
        parents = _prediction_parents(root)
        final = root / "scores"
        if final.exists():
            manifest = _validate_stage(final, root, role="development_scoring")
            if manifest["metadata"]["prediction_manifests"] != parents:
                raise ValueError("Score prediction lineage changed")
            return {"status": "reused_valid", "path": str(final)}
        verified_at = _now()
        assignments = pd.read_parquet(PROTOCOL / "protocol/development_assignments.parquet",
                                     columns=["row_uid", "plot_idx", "development_role", "label_id_dense"])
        from multimodal.geo_helpfulness_confidnet_oof_report import (
            score_study, format_report, representation_summary,
        )
        payloads, scored, representation = {"train": {}, "validation": {}}, [], {}
        for seed in SEEDS:
            representation[str(seed)] = {}
            arrays = _validate_features(root, seed)
            for role in ("train", "validation"):
                selected = assignments.loc[assignments.development_role.eq(role)]
                table = tables[seed][role].merge(selected, on=["row_uid", "plot_idx"],
                                                how="left", validate="one_to_one")
                if table.label_id_dense.isna().any() or len(table) != (3378 if role == "train" else 822):
                    raise ValueError("Scoring label identity join failed")
                if not table.row_uid.equals(tables[seed][role].row_uid):
                    raise ValueError("Scoring label join changed prediction row order")
                payload = {"labels": table.label_id_dense.to_numpy(dtype=np.int64),
                           "plot_ids": table.plot_idx.to_numpy(), "row_uids": table.row_uid.to_numpy()}
                representation[str(seed)][role] = {}
                for mode in MODES:
                    payload[f"{mode}_logits"] = np.stack(table[f"{mode}_logits"])
                    payload[f"{mode}_tcp_hat"] = table[f"{mode}_tcp_hat"].to_numpy()
                    representation[str(seed)][role][mode] = representation_summary(
                        arrays[role][f"{mode}_hidden"], arrays[role]["producer_fold"],
                        payload[f"{mode}_tcp_hat"])
                for arm in ARMS:
                    q = routers[seed][arm][role]
                    payload[f"router_{arm}_q"] = q
                    for index, target in enumerate(TARGET_ORDER):
                        table[f"router_{arm}_q_{target}"] = q[:, index]
                payloads[role][seed] = payload
                table["split"] = role
                scored.append(table)
        settings = declaration["config"]["bootstrap"]
        report = score_study(payloads["train"], payloads["validation"],
                             n_resamples=settings["resamples"], random_seed=settings["seed"])
        report["all_predictions_verified_before_label_join_at"] = verified_at
        report["representation_summaries"] = representation
        report["study_id"] = declaration["config"]["study_id"]
        staged = _staging(root, "scores")
        _write_json(staged / "report.json", report)
        (staged / "report.md").write_text(format_report(report))
        pd.concat(scored, ignore_index=True).to_parquet(staged / "scored_rows.parquet", index=False)
        _seal_stage(staged, final, root, role="development_scoring", metadata={
            "prediction_manifests": parents, "all_predictions_verified_before_label_join_at": verified_at})
        return {"status": "complete", "path": str(final)}


def validate(output_dir=OUTPUT):
    root = _output(output_dir)
    validate_declaration(root, check_parents=True)
    tables, routers = _all_predictions(root)
    path = root / "scores"
    if not path.exists():
        raise ValueError("Study scoring is incomplete")
    manifest = _validate_stage(path, root, role="development_scoring")
    if manifest["metadata"]["prediction_manifests"] != _prediction_parents(root):
        raise ValueError("Score parent lineage differs")
    if set(manifest["file_sha256"]) != {"report.json", "report.md", "scored_rows.parquet"}:
        raise ValueError("Study score artifacts are incomplete")
    _replay_scores(root, tables, routers)
    return {"status": "valid", "confidence_models": 12, "router_models": 8, "scored": True}


def _replay_scores(root, tables, routers):
    """Recalculate every scientific score/interval from authenticated saved rows."""
    from multimodal.geo_helpfulness_confidnet_oof_report import score_study, format_report
    report = _json(root / "scores/report.json")
    rows = pd.read_parquet(root / "scores/scored_rows.parquet")
    config = _config()
    if report.get("study_id") != config["study_id"] or report.get("seeds") != list(SEEDS):
        raise ValueError("Score study or seed identity differs")
    if report.get("policies") != config["policies"]:
        raise ValueError("Scored policy set differs from declaration")
    if len(rows) != 4 * (3378 + 822) or rows.duplicated(["split", "training_seed", "row_uid"]).any():
        raise ValueError("Scored rows are incomplete or duplicated")
    payloads = {"train": {}, "validation": {}}
    assignments = pd.read_parquet(PROTOCOL / "protocol/development_assignments.parquet",
                                 columns=["row_uid", "plot_idx", "development_role", "label_id_dense"])
    for role in ("train", "validation"):
        for seed in SEEDS:
            table = rows.loc[rows.split.eq(role) & rows.training_seed.eq(seed)].reset_index(drop=True)
            source = tables[seed][role]
            if not table[list(source.columns)].equals(source):
                raise ValueError("Scored expert/confidence rows differ from sealed predictions")
            joined = source[["row_uid", "plot_idx"]].merge(
                assignments.loc[assignments.development_role.eq(role)],
                on=["row_uid", "plot_idx"], how="left", validate="one_to_one")
            if not np.array_equal(joined.label_id_dense.to_numpy(), table.label_id_dense.to_numpy()):
                raise ValueError("Scored labels differ from frozen development assignments")
            payload = {"labels": table.label_id_dense.to_numpy(dtype=np.int64),
                       "plot_ids": table.plot_idx.to_numpy(), "row_uids": table.row_uid.to_numpy()}
            for mode in MODES:
                payload[f"{mode}_logits"] = np.stack(table[f"{mode}_logits"])
                payload[f"{mode}_tcp_hat"] = table[f"{mode}_tcp_hat"].to_numpy()
            for arm in ARMS:
                values = table[[f"router_{arm}_q_{state}" for state in TARGET_ORDER]].to_numpy()
                if not np.array_equal(values, routers[seed][arm][role]):
                    raise ValueError("Scored router probabilities differ from sealed predictions")
                payload[f"router_{arm}_q"] = values
            payloads[role][seed] = payload
    expected = score_study(payloads["train"], payloads["validation"],
                           n_resamples=config["bootstrap"]["resamples"],
                           random_seed=config["bootstrap"]["seed"])
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Study score does not reproduce: {key}")
    if (root / "scores/report.md").read_text() != format_report(report):
        raise ValueError("Markdown report does not reproduce the numerical results")

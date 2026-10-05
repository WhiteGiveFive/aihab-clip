"""Declared fixed-expert fold-four confidence/router workflow."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet import (
    _hash, _json, _load_arrays, _lock, _now, _save_arrays, _seal_stage,
    _staging, _validate_stage, _write_json,
)
from multimodal.geo_helpfulness_router_candidate_state import full_feature_columns

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
PROTOCOL = BASE / "protocol_v1"
OUTPUT = BASE / "confidnet_fold4_v1"
CONFIG = ROOT / "configs/multimodal_geo_confidnet_fold4_v1.json"
PLAN = ROOT / "dev_plans/reliability_aware_geo_fusion/16_confidnet_fold4_experiment.md"
SEEDS = (1, 2, 3, 4)
MODES = ("image", "geo", "raw")
ARMS = ("baseline", "augmented")
STATES = ("rescue", "harm", "both_correct", "both_wrong")
CODE_PATHS = (
    "multimodal/geo_helpfulness_confidnet_fold4.py",
    "multimodal/geo_helpfulness_confidnet_fold4_extract.py",
    "multimodal/geo_helpfulness_confidnet_fold4_inputs.py",
    "multimodal/geo_helpfulness_confidnet_fold4_report.py",
    "tools/run_multimodal_geo_helpfulness_confidnet_fold4.py",
)


class StudyPausedError(RuntimeError):
    """The approved failure rule requires review, with partial state preserved."""


def _output(value):
    root = Path(value).resolve()
    if root != OUTPUT and (OUTPUT in root.parents or root in OUTPUT.parents):
        raise ValueError("New output cannot contain or nest inside the default study")
    for protected in [p for p in BASE.iterdir() if p.is_dir() and p != OUTPUT]:
        if protected == root:
            declaration = protected / "declaration.json"
            if declaration.is_file() and _json(declaration).get("config", {}).get("study_id") == "confidnet_fold4_v1":
                continue
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError("New output must be disjoint from every existing study")
    return root


def _config():
    from multimodal.geo_helpfulness_confidnet_oof_fit_v2 import CONFIDENCE_DEFAULTS, RF_DEFAULTS
    cfg = _json(CONFIG)
    checks = {
        "schema_version": "geo_helpfulness.confidnet_fold4_config.v1",
        "study_id": "confidnet_fold4_v1", "training_seeds": list(SEEDS),
        "expert_modes": list(MODES),
        "fold_ids": [3], "train_rows": 852, "validation_rows": 822,
        "confidence_parameters": CONFIDENCE_DEFAULTS, "router_parameters": RF_DEFAULTS,
        "confidence_supervision": "heldout_fold4_from_fixed_folds123_experts",
        "router_confidence_inputs": "in_sample_confidnet_predictions_on_heldout_fold4",
        "confidence_feature": "paired_frozen_expert_penultimate_relu", "confidence_input_dim": 256,
        "target": "native_t1_true_class_probability", "target_order": list(STATES),
        "router_base_variant": "ablation_numeric25",
        "router_augmented_features": [f"{m}_tcp_hat" for m in MODES],
        "policy_comparison": "strict_greater_than_zero_on_geo_fusion_disagreements",
        "policies": ["fusion", "mcp_difference_positive", "tcp_difference_positive", "rf_baseline", "rf_augmented"],
        "validation_expert": "same_fold3_producer_as_confidence_training",
        "numeric_preprocessing_fit": "fold4_only_using_existing_temperature_and_transform_recipe",
        "expert_updates": False, "locked_test_access": False,
        "threshold_selection": True, "guardrail_selection": True, "validation_selection": True,
        "optional_direct_embeddings": False, "optional_in_sample_control": False,
        "optional_true_tcp_router": False, "optional_confidence_cross_fitting": False,
        "evidence_status": "exploratory_reused_development_validation",
        "bootstrap": {"resamples": 2000, "seed": 20260826, "unit": "whole_plot_all_seed_realizations",
                      "stratified": False, "interval": "percentile_95"},
        "guardrails": {"minimum_plots": [20, 10, 0], "harm_numerators": [5, 7, 9],
                       "harm_denominator": 10, "minimum_coverage": 0.01, "require_positive_net": True,
                       "threshold_rule": "existing_relaxed_RF_common_across_seeds", "comparison": "strict_gt_on_disagreements"},
        "verification": {"native_float32": "exact_serialized_state_replay",
                         "portable_float64": "reported_diagnostic_not_tolerance_gate",
                         "final_verification_failure": "preserve_state_history_and_pause"},
    }
    if set(cfg) != set(checks):
        raise ValueError("Configuration has missing or unexpected settings")
    for key, expected in checks.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Approved setting changed: {key}")
    return cfg


def _snapshot(path):
    return {str(p.relative_to(path)): {"sha256": _hash(p), "size": p.stat().st_size,
                                      "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(path.rglob("*")) if p.is_file()}


def declare(output_dir=OUTPUT):
    root = _output(output_dir)
    with _lock(root, "declare"):
        if (root / "declaration.json").exists():
            validate_declaration(root, check_parents=True)
            return {"status": "reused_valid"}
        from multimodal import geo_helpfulness_oof as m2
        from multimodal.geo_helpfulness_confidnet_oof_extract import _fold_spec
        context = m2.load_frozen_context()
        files = set(ROOT / p for p in CODE_PATHS)
        files.update((ROOT / "multimodal").glob("*.py"))
        files.update([CONFIG, ROOT / "configs/multimodal_geo_helpfulness.yaml"])
        files.update((PROTOCOL / "protocol").glob("*"))
        for seed in SEEDS:
            spec = _fold_spec(seed, 3)
            m2.validate_producer(context, spec, seed=seed)
            files.update((PROTOCOL / spec.relative_directory).glob("*"))
        files = sorted(p.resolve() for p in files if p.is_file() and p.suffix != ".pyc")
        if any(not (ROOT / p).is_file() for p in CODE_PATHS):
            raise ValueError("Study implementation incomplete")
        declaration = {
            "schema_version": "geo_helpfulness.confidnet_fold4_declaration.v1",
            "created_at": _now(), "config": _config(), "approved_plan_text": PLAN.read_text(),
            "file_sha256": {str(p.relative_to(ROOT)): _hash(p) for p in files},
            "prior_study_snapshots": {name: _snapshot(BASE / name) for name in
                                      ("confidnet_oof_v1_replay_v2", "confidnet_oof_relaxed_guardrails_v1")},
            "environment": {name: version(name) for name in ("numpy", "pandas", "torch", "scikit-learn")},
            "validation_label_use": "scoring_and_declared_threshold_selection_after_prediction_verification",
            "locked_test_access": False,
        }
        _write_json(root / "declaration.json", declaration)
        return {"status": "declared", "source_files": len(files), "sha256": _hash(root / "declaration.json")}


def validate_declaration(output_dir=OUTPUT, *, check_parents=False):
    root = _output(output_dir)
    saved = _json(root / "declaration.json")
    if (saved.get("schema_version") != "geo_helpfulness.confidnet_fold4_declaration.v1"
            or saved.get("locked_test_access") is not False
            or saved.get("validation_label_use") != "scoring_and_declared_threshold_selection_after_prediction_verification"):
        raise ValueError("Declaration identity or label-use scope changed")
    if saved["config"] != _config():
        raise ValueError("Configuration differs from declaration")
    required = set(CODE_PATHS) | {str(CONFIG.relative_to(ROOT))}
    if not required <= set(saved["file_sha256"]):
        raise ValueError("Declaration omits implementation")
    for path, digest in saved["file_sha256"].items():
        if check_parents or path.endswith(".py") or ROOT / path == CONFIG:
            if _hash(ROOT / path) != digest:
                raise ValueError(f"Declared source changed: {path}")
    if check_parents:
        for name, expected in saved["prior_study_snapshots"].items():
            if _snapshot(BASE / name) != expected:
                raise ValueError(f"Prior experiment changed: {name}")
    return saved


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Expected seed 1, 2, 3 or 4")
    return seed


def _features(root, seed):
    from multimodal.geo_helpfulness_confidnet_fold4_extract import validate_seed_payload
    folder = root / f"features/seed_{seed}"
    manifest = _validate_stage(folder, root, role="fixed_fold4_features")
    if manifest["metadata"] != {"seed": seed, "producer_fold": 3}:
        raise ValueError("Feature stage seed or producer metadata changed")
    arrays = {split: _load_arrays(folder / f"{split}.npz") for split in ("train", "validation")}
    validate_seed_payload({**arrays, "metadata": _json(folder / "extraction.json")}, seed)
    for split, count in (("train", 852), ("validation", 822)):
        if len(arrays[split]["row_uid"]) != count or len(np.unique(arrays[split]["plot_idx"])) != 325:
            raise ValueError("Fold-four experiment split count changed")
    return arrays


def extract_seed(seed, output_dir=OUTPUT, *, device="cuda"):
    from multimodal.geo_helpfulness_confidnet_fold4_extract import extract_seed as extract
    root, seed = _output(output_dir), _seed(seed)
    validate_declaration(root)
    with _lock(root, f"extract_{seed}"):
        folder = root / f"features/seed_{seed}"
        if folder.exists():
            _features(root, seed)
            return {"status": "reused_valid", "seed": seed}
        result = extract(seed, device=device)
        staged = _staging(root, f"features_{seed}")
        for split in ("train", "validation"):
            _save_arrays(staged / f"{split}.npz", result[split])
        _write_json(staged / "extraction.json", result["metadata"])
        _seal_stage(staged, folder, root, role="fixed_fold4_features", metadata={"seed": seed, "producer_fold": 3})
        _features(root, seed)
        return {"status": "complete", "seed": seed}


def _check_paused(root):
    if list((root / "pauses").glob("seed_*/*/manifest.json")):
        raise StudyPausedError("A saved training failure requires review")


def _pause(root, seed, mode, result, reason):
    staged = _staging(root, f"pause_{seed}_{mode}")
    _write_json(staged / "reason.json", {"reason": reason, "seed": seed, "mode": mode})
    if result:
        for key in ("state", "history", "metadata"):
            if key in result:
                _write_json(staged / f"{key}.json", result[key])
    path = root / f"pauses/seed_{seed}/{mode}"
    _seal_stage(staged, path, root, role="training_pause")
    raise StudyPausedError(f"Training stopped: {reason}; preserved at {path}")


def _head(root, seed, mode, arrays):
    from multimodal.geo_helpfulness_confidnet_oof_fit_v2 import predict_confidence, confidence_replay_diagnostics
    path = root / f"confidence/seed_{seed}/{mode}"
    manifest = _validate_stage(path, root, role="fold4_confidence")
    expected = {"seed": seed, "mode": mode, "features_sha256": _hash(root / f"features/seed_{seed}/manifest.json")}
    if manifest["metadata"] != expected:
        raise ValueError("Confidence feature provenance changed")
    state, history = _json(path / "state.json"), _json(path / "history.json")
    if (state["training_seed"] != seed or state["fit_row_count"] != 852
            or state["input_dim"] != 256 or state["status"] != "complete"
            or state["completed_epochs"] != 500 or state["parameters"] != _config()["confidence_parameters"]):
        raise ValueError("Confidence state differs from approved fit")
    if [record["epoch"] for record in history] != list(range(1, 501)):
        raise ValueError("Confidence fitting budget differs")
    saved, metadata = _load_arrays(path / "predictions.npz"), _json(path / "metadata.json")
    for split, data in arrays.items():
        native = predict_confidence(state, data[f"{mode}_hidden"], native=True)
        if not np.array_equal(saved[split], native) or saved[split].dtype != np.float32:
            raise ValueError("Native float32 confidence replay failed")
        portable = predict_confidence(state, data[f"{mode}_hidden"], native=False)
        diagnostic = {**confidence_replay_diagnostics(native, portable), "native_json_replay_exact": True}
        if metadata["prediction_parity"][split] != diagnostic:
            raise ValueError("Float64 confidence diagnostic differs")
    return saved


def fit_seed(seed, output_dir=OUTPUT):
    from multimodal.geo_helpfulness_confidnet_oof_fit_v2 import fit_confidence, ConfidNetOOFFitError
    root, seed = _output(output_dir), _seed(seed)
    config = validate_declaration(root)["config"]
    _check_paused(root)
    with _lock(root, f"fit_{seed}"):
        arrays = _features(root, seed)
        train, validation = arrays["train"], arrays["validation"]
        for mode in MODES:
            final = root / f"confidence/seed_{seed}/{mode}"
            if final.exists():
                _head(root, seed, mode, arrays)
                continue
            print(f"seed {seed}: fitting {mode} confidence on 852 images, 500 epochs", flush=True)
            try:
                result = fit_confidence(train[f"{mode}_hidden"], train[f"{mode}_logits"], train["labels"],
                                        {"validation": validation[f"{mode}_hidden"]}, seed=seed,
                                        parameters=config["confidence_parameters"])
            except ConfidNetOOFFitError as error:
                _pause(root, seed, mode, error.partial_result, str(error))
            if result["status"] != "complete":
                _pause(root, seed, mode, result, result["status"])
            staged = _staging(root, f"confidence_{seed}_{mode}")
            for key in ("state", "metadata", "history"):
                _write_json(staged / f"{key}.json", result[key])
            _save_arrays(staged / "predictions.npz", result["predictions"])
            _seal_stage(staged, final, root, role="fold4_confidence", metadata={
                "seed": seed, "mode": mode, "features_sha256": _hash(root / f"features/seed_{seed}/manifest.json")})
            _head(root, seed, mode, arrays)
    return {"status": "complete", "seed": seed, "heads": 3}


def _inputs(root, seed, arrays, *, build=False):
    from multimodal.geo_helpfulness_confidnet_fold4_inputs import build_inputs
    final = root / f"router_inputs/seed_{seed}"
    if not final.exists():
        if not build:
            raise ValueError("Router inputs not yet fitted")
        result = build_inputs(arrays["train"], arrays["validation"], seed=seed)
        staged = _staging(root, f"inputs_{seed}")
        _save_arrays(staged / "arrays.npz", result["arrays"])
        for key in ("states", "metadata"):
            _write_json(staged / f"{key}.json", result[key])
        _seal_stage(staged, final, root, role="fold4_router_inputs", metadata={
            "seed": seed, "features_sha256": _hash(root / f"features/seed_{seed}/manifest.json")})
    manifest = _validate_stage(final, root, role="fold4_router_inputs")
    if manifest["metadata"] != {"seed": seed, "features_sha256": _hash(root / f"features/seed_{seed}/manifest.json")}:
        raise ValueError("Router input provenance changed")
    result = build_inputs(arrays["train"], arrays["validation"], seed=seed, states=_json(final / "states.json"))
    saved = _load_arrays(final / "arrays.npz")
    if set(saved) != set(result["arrays"]) or any(not np.array_equal(saved[k], v) for k, v in result["arrays"].items()):
        raise ValueError("Numerical router inputs do not replay")
    if result["metadata"] != _json(final / "metadata.json"):
        raise ValueError("Numerical input metadata does not replay")
    return saved


def _matrices(inputs, confidence):
    return {split: {"baseline": inputs[f"X_{split}"],
                    "augmented": np.column_stack([inputs[f"X_{split}"],
                                                   *(confidence[m][split] for m in MODES)])}
            for split in ("train", "validation")}


def _router(root, seed, arm, matrices):
    from multimodal.geo_helpfulness_confidnet_oof_fit_v2 import predict_router
    final = root / f"routers/seed_{seed}/{arm}"
    manifest = _validate_stage(final, root, role="fold4_rf")
    expected = {"seed": seed, "arm": arm, "inputs_sha256": _hash(root / f"router_inputs/seed_{seed}/manifest.json"),
                "confidence_sha256": {m: _hash(root / f"confidence/seed_{seed}/{m}/manifest.json") for m in MODES}}
    if manifest["metadata"] != expected:
        raise ValueError("Router training provenance changed")
    state = _json(final / "state.json")
    names = full_feature_columns()[:25] + ([f"{m}_tcp_hat" for m in MODES] if arm == "augmented" else [])
    if (state["fit_row_count"] != 852 or state["training_seed"] != seed or state["feature_names"] != names
            or state["parameters"] != _config()["router_parameters"]):
        raise ValueError("Router recipe changed")
    saved = _load_arrays(final / "predictions.npz")
    for split in ("train", "validation"):
        replay = predict_router(state, matrices[split][arm])
        if not np.allclose(saved[split], replay, atol=1e-12, rtol=0):
            raise ValueError("RF portable probabilities do not reproduce")
        if not np.array_equal(saved[split][:, 0] > saved[split][:, 1], replay[:, 0] > replay[:, 1]):
            raise ValueError("RF zero-threshold actions do not reproduce")
    return saved


def fit_routers(output_dir=OUTPUT):
    from multimodal.geo_helpfulness_confidnet_oof_fit_v2 import fit_router
    root = _output(output_dir)
    config = validate_declaration(root)["config"]
    _check_paused(root)
    arrays = {s: _features(root, s) for s in SEEDS}
    confidence = {s: {m: _head(root, s, m, arrays[s]) for m in MODES} for s in SEEDS}
    with _lock(root, "routers"):
        for seed in SEEDS:
            inputs = _inputs(root, seed, arrays[seed], build=True)
            matrices = _matrices(inputs, confidence[seed])
            for arm in ARMS:
                final = root / f"routers/seed_{seed}/{arm}"
                if final.exists():
                    _router(root, seed, arm, matrices)
                    continue
                names = full_feature_columns()[:25] + ([f"{m}_tcp_hat" for m in MODES] if arm == "augmented" else [])
                print(f"seed {seed}: fitting {arm} RF on 852 images", flush=True)
                result = fit_router(matrices["train"][arm], inputs["y_train"], {"validation": matrices["validation"][arm]},
                                    seed=seed, feature_names=names, parameters=config["router_parameters"])
                staged = _staging(root, f"rf_{seed}_{arm}")
                for key in ("state", "metadata"):
                    _write_json(staged / f"{key}.json", result[key])
                _save_arrays(staged / "predictions.npz", result["predictions"])
                _seal_stage(staged, final, root, role="fold4_rf", metadata={
                    "seed": seed, "arm": arm, "inputs_sha256": _hash(root / f"router_inputs/seed_{seed}/manifest.json"),
                    "confidence_sha256": {m: _hash(root / f"confidence/seed_{seed}/{m}/manifest.json") for m in MODES}})
                _router(root, seed, arm, matrices)
    return {"status": "complete", "routers": 8}


def _payloads(root):
    """Verify all twelve heads and eight routers before joining validation labels."""
    _check_paused(root)
    features, confidences, routers = {}, {}, {}
    for seed in SEEDS:
        arrays = features[seed] = _features(root, seed)
        heads = confidences[seed] = {m: _head(root, seed, m, arrays) for m in MODES}
        inputs = _inputs(root, seed, arrays)
        matrices = _matrices(inputs, heads)
        routers[seed] = {arm: _router(root, seed, arm, matrices) for arm in ARMS}
    verified_at = _now()
    assignments = pd.read_parquet(PROTOCOL / "protocol/development_assignments.parquet",
                                  columns=["row_uid", "plot_idx", "development_role", "label_id_dense"])
    validation_labels = assignments.loc[assignments.development_role.eq("validation")].set_index("row_uid")
    payloads = {"train": {}, "validation": {}}
    tables = []
    for seed in SEEDS:
        for split, data in features[seed].items():
            if split == "train":
                labels = data["labels"]
            else:
                aligned = validation_labels.loc[data["row_uid"]]
                if not np.array_equal(aligned.plot_idx.to_numpy(), data["plot_idx"]):
                    raise ValueError("Validation labels do not align by plot")
                labels = aligned.label_id_dense.to_numpy(dtype=np.int64)
            payload = {"labels": labels, "row_uids": data["row_uid"], "plot_ids": data["plot_idx"]}
            table = pd.DataFrame({"row_uid": data["row_uid"], "plot_idx": data["plot_idx"],
                                  "training_seed": seed, "producer_fold": data["producer_fold"],
                                  "label_id_dense": labels, "split": split})
            for mode in MODES:
                payload[f"{mode}_logits"] = data[f"{mode}_logits"]
                payload[f"{mode}_tcp_hat"] = confidences[seed][mode][split]
                table[f"{mode}_logits"] = list(payload[f"{mode}_logits"])
                table[f"{mode}_tcp_hat"] = payload[f"{mode}_tcp_hat"]
            for arm in ARMS:
                payload[f"router_{arm}_q"] = routers[seed][arm][split]
                for k, state in enumerate(STATES):
                    table[f"router_{arm}_q_{state}"] = routers[seed][arm][split][:, k]
            payloads[split][seed] = payload
            tables.append(table)
    return payloads, pd.concat(tables, ignore_index=True), verified_at


def score(output_dir=OUTPUT):
    from multimodal.geo_helpfulness_confidnet_fold4_report import build_report, format_report, validation_rows_and_actions
    root = _output(output_dir)
    config = validate_declaration(root)["config"]
    with _lock(root, "score"):
        if (root / "scores").exists():
            _validate_scores(root)
            return {"status": "reused_valid"}
        payloads, rows, verified_at = _payloads(root)
        report = build_report(payloads["train"], payloads["validation"],
                              n_resamples=config["bootstrap"]["resamples"], random_seed=config["bootstrap"]["seed"])
        action_rows, _, actions = validation_rows_and_actions(payloads["validation"])
        for name, values in actions.items():
            action_rows[name] = values
        staged = _staging(root, "scores")
        _write_json(staged / "report.json", report)
        (staged / "report.md").write_text(format_report(report))
        rows.to_parquet(staged / "scored_rows.parquet", index=False)
        action_rows.to_parquet(staged / "actions.parquet", index=False)
        _seal_stage(staged, root / "scores", root, role="fold4_scores", metadata={
            "all_predictions_verified_before_validation_label_join": verified_at,
            "prediction_manifests": _prediction_manifests(root)})
    return {"status": "complete", "path": str(root / "scores/report.md")}


def _prediction_manifests(root):
    paths = list(root.glob("confidence/seed_*/*/manifest.json"))
    paths += list(root.glob("routers/seed_*/*/manifest.json"))
    paths += list(root.glob("features/seed_*/manifest.json"))
    paths += list(root.glob("router_inputs/seed_*/manifest.json"))
    return {str(p.relative_to(root)): _hash(p) for p in paths}


def _validate_scores(root):
    manifest = _validate_stage(root / "scores", root, role="fold4_scores")
    if manifest["metadata"]["prediction_manifests"] != _prediction_manifests(root):
        raise ValueError("Scoring prediction lineage changed")
    return manifest


def validate(output_dir=OUTPUT):
    from multimodal.geo_helpfulness_confidnet_fold4_report import build_report, format_report, validation_rows_and_actions
    root = _output(output_dir)
    config = validate_declaration(root, check_parents=True)["config"]
    _validate_scores(root)
    payloads, rows, _ = _payloads(root)
    pd.testing.assert_frame_equal(rows, pd.read_parquet(root / "scores/scored_rows.parquet"))
    report = build_report(payloads["train"], payloads["validation"],
                          n_resamples=config["bootstrap"]["resamples"], random_seed=config["bootstrap"]["seed"])
    if report != _json(root / "scores/report.json") or format_report(report) != (root / "scores/report.md").read_text():
        raise ValueError("Scientific report does not reproduce")
    action_rows, _, actions = validation_rows_and_actions(payloads["validation"])
    for name, values in actions.items():
        action_rows[name] = values
    pd.testing.assert_frame_equal(action_rows, pd.read_parquet(root / "scores/actions.parquet"))
    return {"status": "validated", "confidence_heads": 12, "routers": 8, "threshold_policies": 18}

"""Frozen joint four-state/TCP router application to the cleaned test identity."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from multimodal.geo_helpfulness_confidnet import (
    _hash, _json, _load_arrays, _lock, _now, _save_arrays, _seal_stage,
    _staging, _validate_stage, _write_json,
)
from multimodal.geo_helpfulness_protocol import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
PARENT = BASE / "joint_tcp_router_v1"
OUTPUT = BASE / "joint_tcp_router_test_v1"
PROTOCOL = BASE / "protocol_v1"
CONFIG = ROOT / "configs/multimodal_geo_joint_tcp_test_v1.json"
PLAN = ROOT / "dev_plans/reliability_aware_geo_fusion/19_joint_tcp_router_cleaned_test.md"
SEEDS = (1, 2, 3, 4)
MODES = ("image", "geo", "raw")
PREDICTION_KEYS = ("state_logits", "tcp_logits", "q", "tcp_hat")
CODE_PATHS = (
    "multimodal/geo_helpfulness_joint_tcp_test.py",
    "multimodal/geo_helpfulness_joint_tcp_test_extract.py",
    "multimodal/geo_helpfulness_joint_tcp_test_report.py",
    "multimodal/geo_helpfulness_joint_tcp_test_registry.py",
    "tools/run_multimodal_geo_helpfulness_joint_tcp_test.py",
)


def expected_config():
    return {
        "study_id": "joint_tcp_router_test_v1", "parent_study_id": "joint_tcp_router_v1",
        "training_seeds": list(SEEDS), "expert_producer": "full_development_train",
        "producer_fold": -1, "test_rows": 1347, "test_plots": 531,
        "configuration_count": 16, "frozen_router_count": 64,
        "relaxed_policy_count": 144, "fixed_zero_policy_count": 16,
        "reference_policies": ["fusion", "image", "geo"],
        "primary_policy_count": 161, "paired_contrast_count": 360,
        "router_policies": "all_parent_validation_thresholds_and_fixed_zero",
        "fitting": False, "recalibration": False, "threshold_selection": False,
        "test_guardrail_enforcement": False, "winner_selection": False,
        "test_label_access": "after_all_four_prediction_bundles_are_sealed_and_event_reserved",
        "router_execution": "native_float32_cpu_one_thread_exact_replay",
        "score_dtype": "float64_after_native_float32_inference",
        "float64_replay": "diagnostic_only",
        "bootstrap": {"resamples": 2000, "seed": 20260826,
                      "unit": "whole_plot_all_seed_realizations"},
        "evidence_status": "exploratory_previously_inspected_cleaned_test",
    }


def _config():
    expected = expected_config()
    if _json(CONFIG) != expected:
        raise ValueError("Frozen joint router test configuration changed")
    return expected


def _snapshot(path):
    return {str(p.relative_to(path)): {"sha256": _hash(p), "size": p.stat().st_size,
                                      "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(path.rglob("*")) if p.is_file() and not p.name.endswith(".lock")}


def _check_snapshot(path, expected, *, allow_additions=False):
    if allow_additions:
        actual = {name: {"sha256": _hash(path / name), "size": (path / name).stat().st_size,
                         "mtime_ns": (path / name).stat().st_mtime_ns} for name in expected}
    else:
        actual = _snapshot(path)
    if actual != expected:
        raise ValueError(f"Previously sealed files changed: {path}")


def declare():
    from multimodal import geo_helpfulness_joint_tcp as parent
    from multimodal.geo_helpfulness_joint_tcp_test_extract import DEFAULT_TEST_SOURCE
    with _lock(OUTPUT, "declare"):
        if (OUTPUT / "declaration.json").exists():
            validate_declaration(check_parents=True)
            return {"status": "reused_valid"}
        saved_parent = parent.validate_declaration(PARENT, check_parents=True)
        # Authenticate the parent stage file sets before snapshotting fitted states.
        for path in sorted(PARENT.rglob("manifest.json")):
            _validate_stage(path.parent, PARENT, role=_json(path)["role"])
        cfg = _config()
        preprocessing, parity = prepare_parent_replay()
        protocol = yaml.safe_load((PROTOCOL / "protocol/resolved_protocol.yaml").read_text())
        registry = (ROOT / protocol["paths"]["locked_test_registry_root"]).resolve()
        reference = _json(PROTOCOL / "protocol/locked_test_snapshot_ref.json")
        identity_path = registry / reference["identity_manifest_relative_path"]
        identity = _json(identity_path)
        if (_hash(identity_path) != reference["identity_manifest_file_sha256"]
                or _hash(registry / "active_snapshot.json") != reference["active_snapshot_file_sha256"]
                or identity["row_count"] != 1347 or identity["plot_count"] != 531):
            raise ValueError("Locked-test identity changed")
        files = {ROOT / name for name in saved_parent["file_sha256"]}
        files.update((ROOT / "multimodal").glob("*.py"))
        files.update(ROOT / name for name in CODE_PATHS)
        files.update([CONFIG, identity_path, registry / "active_snapshot.json", DEFAULT_TEST_SOURCE])
        image_root = ROOT / "data/CS_Xplots_2019_2023_test"
        image_hashes = {}
        for row in identity["rows"]:
            path = (image_root / row["file"]).resolve()
            if image_root.resolve() not in path.parents or not path.is_file():
                raise ValueError("Cleaned image path is missing or outside its root")
            image_hashes[str(path.relative_to(ROOT))] = _hash(path)
        label_source = BASE / "relaxed_guardrails_margin_v1/scores/labels.parquet"
        if not label_source.is_file():
            raise ValueError("Previously acquired canonical test labels are unavailable")
        from multimodal.geo_helpfulness_router_m4_common import check_self_hash
        from multimodal.geo_helpfulness_relaxed_margin_registry import validate_event
        prior_scores = label_source.parent
        label_manifest = _json(prior_scores / "manifest.json")
        label_receipt = _json(prior_scores / "receipt.json")
        check_self_hash(label_manifest, "manifest_sha256")
        check_self_hash(label_receipt, "receipt_sha256")
        if (label_manifest["artifacts"]["labels.parquet"]["file_sha256"] != _hash(label_source)
                or label_receipt["output_file_sha256"]["labels.parquet"] != _hash(label_source)
                or label_manifest["receipt_sha256"] != label_receipt["receipt_sha256"]
                or label_receipt["identity_projection_sha256"] != identity["identity_projection_sha256"]):
            raise ValueError("Prior label projection provenance differs")
        for name in ("receipt.json", "build_declaration.json", "reservation.json", "source_receipt.json"):
            if label_manifest["artifacts"][name]["file_sha256"] != _hash(prior_scores / name):
                raise ValueError("Prior label metadata differs from its seal")
            files.add(prior_scores / name)
        files.add(prior_scores / "manifest.json")
        validate_event(registry / "events", _json(prior_scores / "build_declaration.json"),
                       _json(prior_scores / "reservation.json"), label_receipt, prior_scores / "receipt.json")
        declaration = {
            "schema_version": "geo_helpfulness.joint_tcp_test_declaration.v1",
            "study_id": cfg["study_id"], "created_at": _now(), "config": cfg,
            "approved_plan_text": PLAN.read_text(),
            "user_authorization": "Apply the frozen joint TCP routers to correct their matching fusion experts on cleaned test",
            "source_file_sha256": {str(p.relative_to(ROOT)): _hash(p) for p in sorted(files)},
            "image_file_sha256": image_hashes,
            "parent_snapshot": _snapshot(PARENT),
            "prior_registry_snapshot": _snapshot(registry / "events"),
            "registry_root": str(registry / "events"),
            "identity_manifest": identity, "ontology": protocol["class_ontology"]["classes"],
            "label_source": {"path": str(label_source), "file_sha256": _hash(label_source)},
            "validation_selections": _json(PARENT / "scores/selections.json"),
            "parent_config": saved_parent["config"], "preprocessing": preprocessing,
            "parent_validation_replay": parity,
            "environment": {name: version(name) for name in ("numpy", "pandas", "torch", "scikit-learn")},
            "adaptive_reuse": True, "no_go_preserved": True,
        }
        _write_json(OUTPUT / "declaration.json", declaration)
        return {"status": "declared", "source_files": len(files), "images": len(image_hashes)}


def validate_declaration(*, check_parents=False):
    saved = _json(OUTPUT / "declaration.json")
    if (saved.get("study_id") != "joint_tcp_router_test_v1"
            or saved.get("config") != _config()
            or saved.get("adaptive_reuse") is not True or saved.get("no_go_preserved") is not True):
        raise ValueError("Test study declaration scope changed")
    if not set(CODE_PATHS) <= set(saved["source_file_sha256"]):
        raise ValueError("Study source fingerprint is incomplete")
    for name, digest in saved["source_file_sha256"].items():
        if check_parents or name.endswith(".py") or ROOT / name == CONFIG:
            if _hash(ROOT / name) != digest:
                raise ValueError(f"Declared source changed: {name}")
    if saved["parent_config"] != _json(PARENT / "declaration.json")["config"]:
        raise ValueError("Parent router recipe changed")
    # Every application authenticates the exact fitted states and thresholds.
    _check_snapshot(PARENT, saved["parent_snapshot"])
    if saved["validation_selections"] != _json(PARENT / "scores/selections.json"):
        raise ValueError("Saved validation policies changed")
    if check_parents:
        for name, digest in saved["image_file_sha256"].items():
            if _hash(ROOT / name) != digest:
                raise ValueError(f"Test image changed: {name}")
        _check_snapshot(Path(saved["registry_root"]), saved["prior_registry_snapshot"], allow_additions=True)
    return saved


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Expected seed 1, 2, 3 or 4")
    return seed


def apply_base_inputs(features, states):
    """Apply the frozen OOF temperatures and complete 727-column transform."""
    from multimodal import geo_helpfulness_router_numeric as numeric
    from multimodal import geo_helpfulness_targets_features as semantic
    probabilities = [numeric.apply_expert_temperature(features[f"{prefix}_logits"],
                     states["temperatures"][mode]) for prefix, mode in
                     zip(MODES, ("image_only", "geo_only", "raw_concat"))]
    frame = semantic.build_router_feature_frame(*probabilities,
                probability_basis=semantic.CALIBRATED_PROBABILITY_BASIS)
    matrix = np.ascontiguousarray(numeric.transform_router_features(frame, states["transform"]))
    if matrix.shape != (len(features["row_uid"]), 727) or not np.isfinite(matrix).all():
        raise ValueError("Invalid frozen base feature matrix")
    return matrix


def _matrix(base, features, spec):
    from multimodal.geo_helpfulness_router_candidate_state import variant_feature_indices
    blocks = [base[:, variant_feature_indices(spec["variant"])]]
    if spec["with_embeddings"]:
        blocks += [features["image_embedding"], features["geo_embedding"]]
    matrix = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)
    if matrix.shape != (len(base), spec["input_dim"]) or not np.isfinite(matrix).all():
        raise ValueError("Invalid joint router input dimensions or values")
    return matrix


def _infer(seed, features, states, config, selections):
    from multimodal.geo_helpfulness_joint_tcp_fit import predict_joint
    from multimodal.geo_helpfulness_joint_tcp import feature_names
    from multimodal.geo_helpfulness_joint_tcp_test_report import policy_actions
    base = apply_base_inputs(features, states)
    frames, diagnostics = [], {}
    for spec in config["configs"]:
        name = spec["config_id"]
        folder = PARENT / f"models/seed_{seed}/{name}"
        _validate_stage(folder, PARENT, role="joint_tcp_model")
        state = _json(folder / "state.json")
        identity = {"config_id": name, "training_seed": seed,
                    "input_dim": spec["input_dim"], "auxiliary_weight": spec["auxiliary_weight"],
                    "parameters": config["parameters"], "fit_row_count": 3378}
        if any(state.get(k) != v for k, v in identity.items()):
            raise ValueError("Frozen model identity or recipe differs")
        if _json(folder / "features.json") != feature_names(spec):
            raise ValueError("Frozen model feature ordering differs")
        matrix = _matrix(base, features, spec)
        prediction = predict_joint(state, matrix, native=True)
        repeated = predict_joint(state, matrix, native=True)
        if any(not np.array_equal(prediction[k], repeated[k]) for k in PREDICTION_KEYS):
            raise ValueError("Native float32 inference is not repeatable")
        portable = predict_joint(state, matrix, native=False)
        diagnostics[name] = {key: {
            "maximum_abs_error": float(np.max(np.abs(prediction[key].astype(np.float64) - portable[key]))),
            "mean_abs_error": float(np.mean(np.abs(prediction[key].astype(np.float64) - portable[key]))),
        } for key in PREDICTION_KEYS}
        diagnostics[name]["native_float32_exact"] = True
        diagnostics[name]["float64_state_argmax_changes"] = int(np.sum(
            prediction["q"].argmax(1) != portable["q"].argmax(1)))
        diagnostics[name]["float64_tcp_order_changes"] = int(np.sum(
            (prediction["tcp_hat"][:, 1] > prediction["tcp_hat"][:, 0]) !=
            (portable["tcp_hat"][:, 1] > portable["tcp_hat"][:, 0])))
        frame = pd.DataFrame({key: features[key] for key in ("row_uid", "plot_idx", "producer_fold")})
        frame["training_seed"] = seed
        frame["split"] = "test"
        for key in ("config_id", "input_id", "variant", "with_embeddings", "auxiliary_weight"):
            frame[key] = spec[key]
        for prefix in MODES:
            frame[f"{prefix}_logits"] = list(features[f"{prefix}_logits"])
        for index, target in enumerate(config["target_order"]):
            frame[f"q_{target}"] = prediction["q"][:, index]
        frame["fusion_tcp_hat"] = prediction["tcp_hat"][:, 0]
        frame["geo_tcp_hat"] = prediction["tcp_hat"][:, 1]
        frame["state_logits"] = list(prediction["state_logits"])
        frame["tcp_logits"] = list(prediction["tcp_logits"])
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    actions = policy_actions(rows, config, selections)
    return rows, actions, base, diagnostics


def prepare_parent_replay():
    """Prove the new application exactly reproduces the saved validation pipeline."""
    from multimodal.geo_helpfulness_router_capacity_inputs import load_capacity_inputs
    from multimodal.geo_helpfulness_joint_tcp_report import _contrast_pairs
    inputs = load_capacity_inputs(ROOT, PROTOCOL)
    parent_declaration = _json(PARENT / "declaration.json")
    for path, digest in inputs["parent_file_hashes"].items():
        if parent_declaration["file_sha256"].get(str(Path(path).relative_to(ROOT))) != digest:
            raise ValueError("Preprocessing source differs from parent declaration")
    config = parent_declaration["config"]
    selections = _json(PARENT / "scores/selections.json")
    preprocessing, parity = {}, {}
    for seed in SEEDS:
        features = _load_arrays(PARENT / f"features/seed_{seed}/validation.npz")
        states = {"transform": inputs["transforms"][seed], "temperatures": {
            mode: inputs["temperatures"][(seed, mode)] for mode in ("image_only", "geo_only", "raw_concat")}}
        rows, actions, base, _ = _infer(seed, features, states, config, selections)
        saved_inputs = _load_arrays(PARENT / f"router_inputs/seed_{seed}.npz")
        if not np.array_equal(base, saved_inputs["X_validation"]):
            raise ValueError("New application does not reproduce full727 validation matrix")
        for spec in config["configs"]:
            name = spec["config_id"]
            frame = rows.loc[rows.config_id.eq(name)]
            saved = _load_arrays(PARENT / f"models/seed_{seed}/{name}/predictions.npz")
            actual = {"state_logits": np.stack(frame.state_logits), "tcp_logits": np.stack(frame.tcp_logits),
                      "q": frame[[f"q_{k}" for k in config["target_order"]]].to_numpy(),
                      "tcp_hat": frame[["fusion_tcp_hat", "geo_tcp_hat"]].to_numpy()}
            for key in PREDICTION_KEYS:
                if not np.array_equal(actual[key], saved[f"validation_{key}"]):
                    raise ValueError("New application does not reproduce parent native predictions")
        names = [name for name in actions if name == "fusion" or name.startswith("joint__")]
        saved_actions = pd.read_parquet(PARENT / "scores/actions.parquet",
                columns=["row_uid", "training_seed", *names])
        saved_actions = saved_actions.loc[saved_actions.training_seed.eq(seed)].sort_values("row_uid")
        actual = actions.sort_values("row_uid")
        for key in ("row_uid", "training_seed", *names):
            if not np.array_equal(actual[key].to_numpy(), saved_actions[key].to_numpy()):
                raise ValueError("New application does not reproduce parent action masks")
        if len(names) != 161 or len(_contrast_pairs(config, names)) != 360:
            raise ValueError("Unexpected parent policy or comparison count")
        preprocessing[str(seed)] = states
        parity[str(seed)] = {"base727_exact": True, "models_exact": 16,
                             "validation_rows": len(features["row_uid"]), "policy_masks_exact": 161}
    return preprocessing, parity


def _features(seed, *, deep=False):
    from multimodal.geo_helpfulness_joint_tcp_test_extract import validate_split_payload, validate_seed_payload
    path = OUTPUT / f"features/seed_{seed}"
    manifest = _validate_stage(path, OUTPUT, role="joint_tcp_cleaned_test_features")
    if manifest["metadata"] != {"seed": seed, "producer_fold": -1}:
        raise ValueError("Test feature producer changed")
    arrays = _load_arrays(path / "test.npz")
    validate_split_payload(arrays)
    if len(arrays["row_uid"]) != _config()["test_rows"] or len(np.unique(arrays["plot_idx"])) != _config()["test_plots"]:
        raise ValueError("Unexpected test image or plot count")
    if deep:
        validate_seed_payload({"test": arrays, "metadata": _json(path / "extraction.json")}, seed)
    return arrays


def extract_seed(seed, *, device="cuda"):
    from multimodal.geo_helpfulness_joint_tcp_test_extract import extract_seed as extract
    seed = _seed(seed)
    validate_declaration()
    with _lock(OUTPUT, f"extract_{seed}"):
        final = OUTPUT / f"features/seed_{seed}"
        if final.exists():
            _features(seed, deep=True)
            return {"status": "reused_valid", "seed": seed}
        result = extract(seed, device=device)
        staged = _staging(OUTPUT, f"features_{seed}")
        _save_arrays(staged / "test.npz", result["test"])
        _write_json(staged / "extraction.json", result["metadata"])
        _seal_stage(staged, final, OUTPUT, role="joint_tcp_cleaned_test_features",
                    metadata={"seed": seed, "producer_fold": -1})
        _features(seed, deep=True)
    return {"status": "complete", "seed": seed}


def _predictions(seed, *, replay=False):
    folder = OUTPUT / f"predictions/seed_{seed}"
    manifest = _validate_stage(folder, OUTPUT, role="joint_tcp_cleaned_test_predictions")
    expected = {"seed": seed, "features_sha256": _hash(OUTPUT / f"features/seed_{seed}/manifest.json")}
    if manifest["metadata"] != expected:
        raise ValueError("Prediction feature lineage changed")
    rows = pd.read_parquet(folder / "rows.parquet")
    actions = pd.read_parquet(folder / "actions.parquet")
    if any("label" in key or "target_state" == key for key in (*rows.columns, *actions.columns)):
        raise ValueError("Label-blind predictions contain labels")
    if replay:
        declaration = _json(OUTPUT / "declaration.json")
        actual_rows, actual_actions, base, diagnostics = _infer(seed, _features(seed, deep=True),
            declaration["preprocessing"][str(seed)], declaration["parent_config"], declaration["validation_selections"])
        pd.testing.assert_frame_equal(rows, actual_rows, check_exact=True)
        pd.testing.assert_frame_equal(actions, actual_actions, check_exact=True)
        if not np.array_equal(base, _load_arrays(folder / "numeric_inputs.npz")["X727"]):
            raise ValueError("Frozen numerical inputs do not reproduce")
        if diagnostics != _json(folder / "replay.json"):
            raise ValueError("Float64 diagnostics changed")
    return rows, actions


def predict_seed(seed):
    seed = _seed(seed)
    declaration = validate_declaration()
    with _lock(OUTPUT, f"predict_{seed}"):
        final = OUTPUT / f"predictions/seed_{seed}"
        if final.exists():
            _predictions(seed, replay=True)
            return {"status": "reused_valid", "seed": seed}
        rows, actions, base, diagnostics = _infer(seed, _features(seed, deep=True),
            declaration["preprocessing"][str(seed)], declaration["parent_config"], declaration["validation_selections"])
        staged = _staging(OUTPUT, f"predictions_{seed}")
        rows.to_parquet(staged / "rows.parquet", index=False)
        actions.to_parquet(staged / "actions.parquet", index=False)
        _save_arrays(staged / "numeric_inputs.npz", {"X727": base})
        _write_json(staged / "replay.json", diagnostics)
        _seal_stage(staged, final, OUTPUT, role="joint_tcp_cleaned_test_predictions", metadata={
            "seed": seed, "features_sha256": _hash(OUTPUT / f"features/seed_{seed}/manifest.json")})
        _predictions(seed, replay=True)
    return {"status": "complete", "seed": seed, "routers": 16, "policies": 161}


def _scoring_inputs(labels, declaration):
    from multimodal.geo_helpfulness_joint_tcp_test_report import policy_actions
    rows = pd.concat([_predictions(seed)[0] for seed in SEEDS], ignore_index=True)
    actions = pd.concat([_predictions(seed)[1] for seed in SEEDS], ignore_index=True)
    actual = policy_actions(rows, declaration["parent_config"], declaration["validation_selections"])
    pd.testing.assert_frame_equal(actions.sort_values(["row_uid", "training_seed"]).reset_index(drop=True), actual, check_exact=True)
    fields = ["row_uid", "plot_idx", "label_id_dense"]
    scored = rows.merge(labels[fields], on=["row_uid", "plot_idx"], how="left", validate="many_to_one", sort=False)
    if len(scored) != len(rows) or scored.label_id_dense.isna().any():
        raise ValueError("Canonical label identity join failed")
    scored["label_id_dense"] = scored.label_id_dense.astype(np.int64)
    return scored, actual


def _save_report(staged, result, rows):
    from multimodal.geo_helpfulness_joint_tcp_test_report import format_report
    _write_json(staged / "report.json", result["report"])
    (staged / "report.md").write_text(format_report(_json(staged / "report.json")))
    rows.to_parquet(staged / "scored_rows.parquet", index=False)
    result["actions"].to_parquet(staged / "actions.parquet", index=False)
    _save_arrays(staged / "bootstrap_draws.npz", {"draws": result["bootstrap_draws"]})
    _write_json(staged / "bootstrap_policy_order.json", result["bootstrap_policy_order"])


def score():
    from multimodal import geo_helpfulness_joint_tcp_test_registry as registry
    from multimodal.geo_helpfulness_joint_tcp_test_report import make_report
    declaration = validate_declaration(check_parents=True)
    with _lock(OUTPUT, "score"):
        final = OUTPUT / "scores"
        if final.exists():
            _validate_stage(final, OUTPUT, role="joint_tcp_cleaned_test_scores")
            registry.finish_event(declaration["registry_root"], _json(final / "build.json"),
                _json(final / "reservation.json"), _json(final / "receipt.json"), final / "receipt.json")
            return validate_scores()
        for seed in SEEDS:
            _predictions(seed, replay=True)
        manifests = {seed: OUTPUT / f"predictions/seed_{seed}/manifest.json" for seed in SEEDS}
        build = registry.make_build_record(OUTPUT, declaration, manifests)
        reservation = registry.reserve_event(declaration["registry_root"], build)
        acquisition = OUTPUT / "label_acquisition"
        acquisition.mkdir(exist_ok=True)
        labels, _ = registry.acquire_labels(declaration["registry_root"], build, reservation, declaration, acquisition)
        rows, actions = _scoring_inputs(labels, declaration)
        result = make_report(rows, declaration["parent_config"], declaration["validation_selections"])
        policy_names = result["bootstrap_policy_order"]
        for name in policy_names:
            if not np.array_equal(result["actions"][name], actions[name]):
                raise ValueError("Scored actions differ from sealed label-blind choices")
        staged = _staging(OUTPUT, "scores")
        _save_report(staged, result, rows)
        _write_json(staged / "build.json", build)
        _write_json(staged / "reservation.json", reservation)
        receipt = {"study_id": declaration["study_id"], "created_at": _now(),
                   "evaluation_event_id": canonical_sha256(build), **build,
                   "report_file_sha256": _hash(staged / "report.json"),
                   "labels_file_sha256": _hash(acquisition / "labels.parquet"),
                   "label_source_receipt_sha256": _hash(acquisition / "source_receipt.json"),
                   "adaptive_reuse": True, "no_fitting_or_selection": True}
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write_json(staged / "receipt.json", receipt)
        _seal_stage(staged, final, OUTPUT, role="joint_tcp_cleaned_test_scores", metadata={
            "prediction_manifests": {str(seed): _hash(path) for seed, path in manifests.items()}})
        registry.finish_event(declaration["registry_root"], build, reservation, receipt, final / "receipt.json")
    return {"status": "complete", "report": str(final / "report.md")}


def validate_scores():
    from multimodal import geo_helpfulness_joint_tcp_test_registry as registry
    from multimodal.geo_helpfulness_joint_tcp_test_report import make_report, format_report
    declaration = validate_declaration()
    final = OUTPUT / "scores"
    manifest = _validate_stage(final, OUTPUT, role="joint_tcp_cleaned_test_scores")
    manifests = {seed: OUTPUT / f"predictions/seed_{seed}/manifest.json" for seed in SEEDS}
    if manifest["metadata"]["prediction_manifests"] != {str(s): _hash(p) for s, p in manifests.items()}:
        raise ValueError("Score prediction lineage changed")
    build = registry.make_build_record(OUTPUT, declaration, manifests)
    if build != _json(final / "build.json"):
        raise ValueError("Score build binding changed")
    reservation, receipt = _json(final / "reservation.json"), _json(final / "receipt.json")
    registry.validate_event(declaration["registry_root"], build, reservation, receipt, final / "receipt.json")
    labels, _ = registry.acquire_labels(declaration["registry_root"], build, reservation, declaration, OUTPUT / "label_acquisition")
    for key, path in (("labels_file_sha256", OUTPUT / "label_acquisition/labels.parquet"),
                      ("label_source_receipt_sha256", OUTPUT / "label_acquisition/source_receipt.json"),
                      ("report_file_sha256", final / "report.json")):
        if receipt[key] != _hash(path):
            raise ValueError("Registry receipt file binding changed")
    rows, actions = _scoring_inputs(labels, declaration)
    result = make_report(rows, declaration["parent_config"], declaration["validation_selections"])
    if result["report"] != _json(final / "report.json"):
        raise ValueError("Test metrics do not reproduce")
    if format_report(_json(final / "report.json")) != (final / "report.md").read_text():
        raise ValueError("Test Markdown does not reproduce")
    pd.testing.assert_frame_equal(rows, pd.read_parquet(final / "scored_rows.parquet"), check_exact=True)
    pd.testing.assert_frame_equal(result["actions"], pd.read_parquet(final / "actions.parquet"), check_exact=True)
    if result["bootstrap_policy_order"] != _json(final / "bootstrap_policy_order.json"):
        raise ValueError("Bootstrap ordering changed")
    if not np.array_equal(result["bootstrap_draws"], _load_arrays(final / "bootstrap_draws.npz")["draws"]):
        raise ValueError("Whole-plot bootstrap does not reproduce exactly")
    for name in result["bootstrap_policy_order"]:
        if not np.array_equal(result["actions"][name], actions[name]):
            raise ValueError("Scored actions differ from sealed choices")
    return {"status": "validated", "seeds": 4, "test_rows": 1347,
            "routers": 64, "policies": 161, "contrasts": 360}


def validate():
    validate_declaration(check_parents=True)
    for seed in SEEDS:
        _predictions(seed, replay=True)
    return validate_scores()

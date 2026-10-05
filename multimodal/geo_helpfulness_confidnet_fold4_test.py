"""Frozen fold-four confidence/router application to the cleaned test identity."""

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
PARENT = BASE / "confidnet_fold4_v1"
OUTPUT = BASE / "confidnet_fold4_test_v1"
PROTOCOL = BASE / "protocol_v1"
CONFIG = ROOT / "configs/multimodal_geo_confidnet_fold4_test_v1.json"
PLAN = ROOT / "dev_plans/reliability_aware_geo_fusion/17_confidnet_fold4_cleaned_test.md"
SEEDS = (1, 2, 3, 4)
MODES = ("image", "geo", "raw")
ARMS = ("baseline", "augmented")
CODE_PATHS = (
    "multimodal/geo_helpfulness_confidnet_fold4_test.py",
    "multimodal/geo_helpfulness_confidnet_fold4_test_extract.py",
    "multimodal/geo_helpfulness_confidnet_fold4_test_report.py",
    "multimodal/geo_helpfulness_confidnet_fold4_test_registry.py",
    "tools/run_multimodal_geo_helpfulness_confidnet_fold4_test.py",
)


def _config():
    expected = {
        "study_id": "confidnet_fold4_test_v1", "parent_study_id": "confidnet_fold4_v1",
        "training_seeds": list(SEEDS), "expert_producer_fold": 3,
        "test_rows": 1347, "test_plots": 531, "expert_modes": list(MODES),
        "router_arms": list(ARMS),
        "router_policies": "fixed_zero_and_all_18_saved_validation_guardrail_policies",
        "direct_tcp_policy": "choose_highest_estimated_tcp_among_image_geo_fusion",
        "tie_priority": ["raw", "geo", "image"],
        "reference_policies": ["image", "geo", "fusion", "tcp_geo_fusion", "mcp_geo_fusion"],
        "fitting": False, "recalibration": False, "threshold_selection": False,
        "test_label_access": "after_all_four_prediction_bundles_are_sealed_and_event_reserved",
        "confidence_execution": "native_float32_cpu_one_thread_exact_replay",
        "float64_confidence_replay": "diagnostic_only",
        "bootstrap": {"resamples": 2000, "seed": 20260826,
                      "unit": "whole_plot_all_seed_realizations"},
        "evidence_status": "exploratory_previously_inspected_cleaned_test",
    }
    if _json(CONFIG) != expected:
        raise ValueError("Frozen test application configuration changed")
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
    from multimodal import geo_helpfulness_confidnet_fold4 as parent
    from multimodal.geo_helpfulness_confidnet_fold4_test_extract import DEFAULT_TEST_SOURCE
    with _lock(OUTPUT, "declare"):
        if (OUTPUT / "declaration.json").exists():
            validate_declaration(check_parents=True)
            return {"status": "reused_valid"}
        saved_parent = parent.validate_declaration(PARENT, check_parents=True)
        # Authenticate the parent stage file sets before snapshotting fitted states.
        for path in sorted(PARENT.rglob("manifest.json")):
            _validate_stage(path.parent, PARENT, role=_json(path)["role"])
        cfg = _config()
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
            "schema_version": "geo_helpfulness.confidnet_fold4_test_declaration.v1",
            "study_id": cfg["study_id"], "created_at": _now(), "config": cfg,
            "approved_plan_text": PLAN.read_text(),
            "user_authorization": "Apply the fold-4 router and highest-estimated-TCP expert policy to cleaned test",
            "source_file_sha256": {str(p.relative_to(ROOT)): _hash(p) for p in sorted(files)},
            "image_file_sha256": image_hashes,
            "parent_snapshot": _snapshot(PARENT),
            "prior_registry_snapshot": _snapshot(registry / "events"),
            "registry_root": str(registry / "events"),
            "identity_manifest": identity, "ontology": protocol["class_ontology"]["classes"],
            "label_source": {"path": str(label_source), "file_sha256": _hash(label_source)},
            "validation_selections": _json(PARENT / "scores/report.json")["relaxed"]["selections"],
            "environment": {name: version(name) for name in ("numpy", "pandas", "torch", "scikit-learn")},
            "adaptive_reuse": True, "no_go_preserved": True,
        }
        _write_json(OUTPUT / "declaration.json", declaration)
        return {"status": "declared", "source_files": len(files), "images": len(image_hashes)}


def validate_declaration(*, check_parents=False):
    saved = _json(OUTPUT / "declaration.json")
    if (saved.get("study_id") != "confidnet_fold4_test_v1"
            or saved.get("config") != _config()
            or saved.get("adaptive_reuse") is not True or saved.get("no_go_preserved") is not True):
        raise ValueError("Test study declaration scope changed")
    if not set(CODE_PATHS) <= set(saved["source_file_sha256"]):
        raise ValueError("Study source fingerprint is incomplete")
    for name, digest in saved["source_file_sha256"].items():
        if check_parents or name.endswith(".py") or ROOT / name == CONFIG:
            if _hash(ROOT / name) != digest:
                raise ValueError(f"Declared source changed: {name}")
    # Every application authenticates the exact fitted states and thresholds.
    _check_snapshot(PARENT, saved["parent_snapshot"])
    if saved["validation_selections"] != _json(PARENT / "scores/report.json")["relaxed"]["selections"]:
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


def _features(seed, *, deep=False):
    path = OUTPUT / f"features/seed_{seed}"
    manifest = _validate_stage(path, OUTPUT, role="fold4_cleaned_test_features")
    if manifest["metadata"] != {"seed": seed, "producer_fold": 3}:
        raise ValueError("Test feature producer changed")
    arrays = _load_arrays(path / "test.npz")
    if "labels" in arrays or len(arrays["row_uid"]) != _config()["test_rows"]:
        raise ValueError("Unexpected labels or test row count")
    if deep:
        from multimodal.geo_helpfulness_confidnet_fold4_test_extract import validate_seed_payload
        validate_seed_payload({"test": arrays, "metadata": _json(path / "extraction.json")}, seed)
    return arrays


def extract_seed(seed, *, device="cuda"):
    from multimodal.geo_helpfulness_confidnet_fold4_test_extract import extract_seed as extract
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
        _seal_stage(staged, final, OUTPUT, role="fold4_cleaned_test_features",
                    metadata={"seed": seed, "producer_fold": 3})
        _features(seed, deep=True)
    return {"status": "complete", "seed": seed}


def apply_numeric_inputs(features, states):
    """Apply authenticated fold-four temperatures and transforms without fitting."""
    from multimodal import geo_helpfulness_router_numeric as numeric
    from multimodal import geo_helpfulness_targets_features as semantic
    probabilities = [numeric.apply_expert_temperature(features[f"{mode}_logits"],
                     states["temperatures"][expert]) for mode, expert in
                     zip(MODES, ("image_only", "geo_only", "raw_concat"))]
    frame = semantic.build_router_feature_frame(*probabilities,
                probability_basis=semantic.CALIBRATED_PROBABILITY_BASIS)
    return np.ascontiguousarray(numeric.transform_router_features(frame, states["transform"])[:, :25])


def _infer(seed, features, selections):
    from multimodal.geo_helpfulness_confidnet_oof_fit_v2 import (
        predict_confidence, predict_router, confidence_replay_diagnostics,
    )
    from multimodal.geo_helpfulness_confidnet_fold4_test_report import policy_choices
    numeric = apply_numeric_inputs(features, _json(PARENT / f"router_inputs/seed_{seed}/states.json"))
    result = {key: features[key].copy() for key in ("row_uid", "plot_idx", "producer_fold")}
    result["X_numeric25"] = numeric
    diagnostics = {}
    for mode in MODES:
        state = _json(PARENT / f"confidence/seed_{seed}/{mode}/state.json")
        native = predict_confidence(state, features[f"{mode}_hidden"], native=True)
        if not np.array_equal(native, predict_confidence(state, features[f"{mode}_hidden"], native=True)):
            raise ValueError("Native confidence inference fails exact replay")
        portable = predict_confidence(state, features[f"{mode}_hidden"], native=False)
        diagnostics[mode] = {**confidence_replay_diagnostics(native, portable), "native_json_replay_exact": True}
        result[f"{mode}_tcp_hat"] = native
        result[f"{mode}_logits"] = features[f"{mode}_logits"].copy()
    matrices = {"baseline": numeric, "augmented": np.column_stack([
        numeric, *(result[f"{mode}_tcp_hat"] for mode in MODES)])}
    for arm in ARMS:
        state = _json(PARENT / f"routers/seed_{seed}/{arm}/state.json")
        result[f"router_{arm}_q"] = predict_router(state, matrices[arm])
    payload = _payload(result)
    choices = policy_choices(payload, selections)
    for name, values in choices.items():
        result[f"choice__{name}"] = values
    return result, diagnostics


def _payload(arrays):
    payload = {"row_uids": arrays["row_uid"], "plot_ids": arrays["plot_idx"]}
    for mode in MODES:
        for suffix in ("logits", "tcp_hat"):
            payload[f"{mode}_{suffix}"] = arrays[f"{mode}_{suffix}"]
    for arm in ARMS:
        payload[f"router_{arm}_q"] = arrays[f"router_{arm}_q"]
    return payload


def _predictions(seed, *, replay=False):
    path = OUTPUT / f"predictions/seed_{seed}"
    manifest = _validate_stage(path, OUTPUT, role="fold4_cleaned_test_predictions")
    expected = {"seed": seed, "features_sha256": _hash(OUTPUT / f"features/seed_{seed}/manifest.json")}
    if manifest["metadata"] != expected:
        raise ValueError("Prediction feature lineage changed")
    saved = _load_arrays(path / "arrays.npz")
    if "labels" in saved:
        raise ValueError("Label-blind prediction stage contains labels")
    if replay:
        expected_arrays, diagnostics = _infer(seed, _features(seed, deep=True),
                                              _json(OUTPUT / "declaration.json")["validation_selections"])
        if set(saved) != set(expected_arrays):
            raise ValueError("Prediction array names changed")
        for name, values in expected_arrays.items():
            if saved[name].dtype != values.dtype or not np.array_equal(saved[name], values):
                raise ValueError(f"Exact saved prediction replay failed: {name}")
        if diagnostics != _json(path / "replay.json"):
            raise ValueError("Portable confidence diagnostics changed")
    return saved


def predict_seed(seed):
    seed = _seed(seed)
    declaration = validate_declaration()
    with _lock(OUTPUT, f"predict_{seed}"):
        final = OUTPUT / f"predictions/seed_{seed}"
        if final.exists():
            _predictions(seed, replay=True)
            return {"status": "reused_valid", "seed": seed}
        arrays, diagnostics = _infer(seed, _features(seed, deep=True), declaration["validation_selections"])
        staged = _staging(OUTPUT, f"predictions_{seed}")
        _save_arrays(staged / "arrays.npz", arrays)
        _write_json(staged / "replay.json", diagnostics)
        _seal_stage(staged, final, OUTPUT, role="fold4_cleaned_test_predictions", metadata={
            "seed": seed, "features_sha256": _hash(OUTPUT / f"features/seed_{seed}/manifest.json")})
        _predictions(seed, replay=True)
    return {"status": "complete", "seed": seed, "policy_count": sum(k.startswith("choice__") for k in arrays)}


def _scoring_inputs(labels, declaration):
    from multimodal.geo_helpfulness_confidnet_fold4_test_report import policy_choices
    lookup = labels.set_index("row_uid")
    payloads = {}
    for seed in SEEDS:
        arrays = _predictions(seed)
        aligned = lookup.loc[arrays["row_uid"]]
        if not np.array_equal(aligned.plot_idx.to_numpy(dtype=str), arrays["plot_idx"]):
            raise ValueError("Label plot identity mismatch")
        payload = _payload(arrays)
        choices = policy_choices(payload, declaration["validation_selections"])
        if set(k for k in arrays if k.startswith("choice__")) != {f"choice__{k}" for k in choices}:
            raise ValueError("Scored policy set differs from sealed choices")
        for name, values in choices.items():
            if not np.array_equal(arrays[f"choice__{name}"], values):
                raise ValueError("Scored policy differs from its label-blind seal")
        payload["labels"] = aligned.label_id_dense.to_numpy(dtype=np.int64)
        payloads[seed] = payload
    return payloads


def score():
    from multimodal import geo_helpfulness_confidnet_fold4_test_registry as registry
    from multimodal.geo_helpfulness_confidnet_fold4_test_report import build_report, format_report, scored_rows_and_choices, choices_frame
    declaration = validate_declaration(check_parents=True)
    with _lock(OUTPUT, "score"):
        if (OUTPUT / "scores").exists():
            final = OUTPUT / "scores"
            _validate_stage(final, OUTPUT, role="fold4_cleaned_test_scores")
            registry.finish_event(declaration["registry_root"], _json(final / "build.json"),
                _json(final / "reservation.json"), _json(final / "receipt.json"), final / "receipt.json")
            return validate_scores()
        # All choices must exist and authenticate before a label-access event.
        for seed in SEEDS:
            _predictions(seed, replay=True)
        manifests = {seed: OUTPUT / f"predictions/seed_{seed}/manifest.json" for seed in SEEDS}
        build = registry.make_build_record(OUTPUT, declaration, manifests)
        reservation = registry.reserve_event(declaration["registry_root"], build)
        acquisition = OUTPUT / "label_acquisition"
        acquisition.mkdir(exist_ok=True)
        acquired = registry.acquire_labels(declaration["registry_root"], build, reservation, declaration, acquisition)
        labels = acquired[0] if isinstance(acquired, tuple) else acquired
        payloads = _scoring_inputs(labels, declaration)
        bootstrap = declaration["config"]["bootstrap"]
        report = build_report(payloads, declaration["validation_selections"],
                              n_resamples=bootstrap["resamples"], random_seed=bootstrap["seed"])
        staged = _staging(OUTPUT, "scores")
        _write_json(staged / "report.json", report)
        (staged / "report.md").write_text(format_report(report))
        rows, choices = scored_rows_and_choices(payloads, declaration["validation_selections"])
        rows.to_parquet(staged / "scored_rows.parquet", index=False)
        choices_frame(rows, choices).to_parquet(staged / "policy_choices.parquet", index=False)
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
        _seal_stage(staged, OUTPUT / "scores", OUTPUT, role="fold4_cleaned_test_scores", metadata={
            "prediction_manifests": {str(seed): _hash(path) for seed, path in manifests.items()}})
        registry.finish_event(declaration["registry_root"], build, reservation, receipt, OUTPUT / "scores/receipt.json")
    return {"status": "complete", "report": str(OUTPUT / "scores/report.md")}


def validate_scores():
    from multimodal import geo_helpfulness_confidnet_fold4_test_registry as registry
    from multimodal.geo_helpfulness_confidnet_fold4_test_report import build_report, format_report, scored_rows_and_choices, choices_frame
    declaration = validate_declaration()
    final = OUTPUT / "scores"
    manifest = _validate_stage(final, OUTPUT, role="fold4_cleaned_test_scores")
    manifests = {seed: OUTPUT / f"predictions/seed_{seed}/manifest.json" for seed in SEEDS}
    if manifest["metadata"]["prediction_manifests"] != {str(s): _hash(p) for s, p in manifests.items()}:
        raise ValueError("Score prediction lineage changed")
    build = registry.make_build_record(OUTPUT, declaration, manifests)
    if build != _json(final / "build.json"):
        raise ValueError("Score build binding changed")
    reservation, receipt = _json(final / "reservation.json"), _json(final / "receipt.json")
    registry.validate_event(declaration["registry_root"], build, reservation, receipt, final / "receipt.json")
    acquired = registry.acquire_labels(declaration["registry_root"], build, reservation, declaration, OUTPUT / "label_acquisition")
    labels = acquired[0] if isinstance(acquired, tuple) else acquired
    if receipt["labels_file_sha256"] != _hash(OUTPUT / "label_acquisition/labels.parquet"):
        raise ValueError("Acquired score labels changed")
    if receipt["label_source_receipt_sha256"] != _hash(OUTPUT / "label_acquisition/source_receipt.json"):
        raise ValueError("Label acquisition receipt changed")
    if receipt["report_file_sha256"] != _hash(final / "report.json"):
        raise ValueError("Registry receipt report binding changed")
    payloads = _scoring_inputs(labels, declaration)
    bootstrap = declaration["config"]["bootstrap"]
    report = build_report(payloads, declaration["validation_selections"],
                          n_resamples=bootstrap["resamples"], random_seed=bootstrap["seed"])
    if report != _json(final / "report.json") or format_report(report) != (final / "report.md").read_text():
        raise ValueError("Test metrics do not reproduce")
    rows, choices = scored_rows_and_choices(payloads, declaration["validation_selections"])
    pd.testing.assert_frame_equal(rows, pd.read_parquet(final / "scored_rows.parquet"))
    pd.testing.assert_frame_equal(choices_frame(rows, choices), pd.read_parquet(final / "policy_choices.parquet"))
    return {"status": "validated", "seeds": 4, "test_rows": 1347}


def validate():
    validate_declaration(check_parents=True)
    for seed in SEEDS:
        _predictions(seed, replay=True)
    return validate_scores()

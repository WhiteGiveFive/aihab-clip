"""Prospective CLS-bypass comparisons with authenticated, unchanged M3 reuse.

Only extension artifacts are written. Numerical selection and paired bootstrap
use the original public helpers; new bundles have their own source identity.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .attention_fusion import GeoPreprocessor
from . import attention_fusion_comparison as original
from .attention_fusion_comparison import (
    ComparisonError, IDENTITIES, METRICS, _csv, _equal_metrics, _inside, _read,
    _require, _selected_references, _validate_matrix, _write, _write_new_json,
    compare_selected, select_learning_rates,
)
from .attention_fusion_training import (
    _array_hash, _validated_data, _verify_bundle, classification_metrics,
    configuration_hash, sha256_file,
)
from .attention_fusion_bypass_training import _source_hashes

REPO_ROOT = Path(__file__).resolve().parents[1]
BYPASSES = ("attention_cls_skip_adapted", "attention_cls_skip_original")
COMPARATORS = ("attention_cls", "projected_concat", "raw_concat", "attention_two_token", "image_only")


def declare_contrasts(config: dict, matrix: dict) -> dict:
    """Declare exactly 37 directional comparisons without inspecting scores."""
    _validate_matrix(config, matrix)
    expected = {(name, geo) for name in BYPASSES for geo in ("native", "l2", "standardize")}
    actual = {(arm["candidate_id"], arm["geo_preprocessing"]) for arm in matrix["candidate_geo_arms"]}
    _require(actual == expected and len(matrix["candidate_geo_arms"]) == 6,
             "Bypass declaration requires exactly the six extension arms")
    _require(config["selection"]["primary_geo_arm"] == "native", "Native geo is the primary treatment")
    contrasts = []

    def arm(candidate, geo):
        return f"{candidate}__geo_{'not_applicable' if candidate == 'image_only' else geo}"

    def add(left, left_geo, right, right_geo, kind):
        lhs, rhs = arm(left, left_geo), arm(right, right_geo)
        contrasts.append({"contrast_id": f"{lhs}__minus__{rhs}",
                          "lhs_arm_id": lhs, "rhs_arm_id": rhs, "kind": kind})

    for geo in config["geo"]["comparison"]:
        for candidate in BYPASSES:
            for comparator in COMPARATORS:
                kind = ("primary" if geo == "native" and comparator == "attention_cls"
                        else "baseline_image" if comparator == "image_only"
                        else "architecture")
                add(candidate, geo, comparator, geo, kind)
        add(BYPASSES[1], geo, BYPASSES[0], geo, "bypass_direct")
    for candidate in BYPASSES:
        for geo in ("l2", "standardize"):
            add(candidate, geo, candidate, "native", "preprocessing")
    _require(len(contrasts) == len({row["contrast_id"] for row in contrasts}) == 37,
             "The extension requires 37 unique contrasts")
    primary = [row["contrast_id"] for row in contrasts if row["kind"] == "primary"]
    _require(len(primary) == 2, "Both native bypass-versus-CLS contrasts are primary")
    return {"schema_version": "attention_fusion.bypass_contrasts.v1", "study_id": config["study_id"],
        "config_sha256": configuration_hash(config), "matrix_sha256": configuration_hash(matrix),
        "primary_contrast_ids": primary, "contrasts": contrasts,
        "bootstrap": config["metrics"]["bootstrap"], "interpretation": config["interpretation"],
        "selection": config["selection"], "source_study": config["source_study"],
        "claim_scope": "Exploratory development extension proposed after original M3 results; conditional on selected recipes and reused validation split.",
        "multiple_comparison_adjustment": "none; two declared primary contrasts, intervals are not simultaneous",
        "attribution": "Complete configurations differ in bypass feature access and classifier capacity; no isolated attention or capacity effect is identified."}


def _declaration_sources() -> dict:
    return {"training_source_sha256": _source_hashes(),
        "original_training_source_sha256": original._source_hashes(),
        "input_loader_sha256": sha256_file(Path(__file__).with_name("attention_fusion_inputs.py")),
        "original_comparison_sha256": sha256_file(Path(original.__file__))}


def _verify_declaration(value: dict) -> None:
    _require(value.get("manifest_payload_sha256") == configuration_hash({
        k: v for k, v in value.items() if k != "manifest_payload_sha256"}), "Contrast declaration checksum differs")
    _require(value.get("development_runs_present_at_declaration") is False,
             "Contrast declaration was not frozen before extension fitting")
    _require(all(value.get(k) == v for k, v in _declaration_sources().items()),
             "Sources changed after contrast declaration")


def freeze_contrasts(config: dict, matrix: dict, path: str | Path, *,
                     repo_root: str | Path = REPO_ROOT) -> dict:
    path, repo = Path(path), Path(repo_root).resolve()
    declaration = declare_contrasts(config, matrix)
    if path.exists():
        saved = _read(path)
        _verify_declaration(saved)
        _require(saved["declaration"] == declaration, "Existing contrast declaration differs")
        return saved
    _require(not any((_inside(repo, fit["output_dir"]) / "run_spec.json").exists()
                     for fit in matrix["development_fits"]),
             "Freeze contrasts before extension development fits start")
    saved = {"declaration": declaration, "frozen_at": datetime.now(timezone.utc).isoformat(),
             "development_runs_present_at_declaration": False, **_declaration_sources()}
    saved["manifest_payload_sha256"] = configuration_hash(saved)
    _write_new_json(path, saved)
    return saved


def load_reference_inputs(config: dict, *, repo_root: str | Path = REPO_ROOT) -> dict:
    from .attention_fusion_bypass_protocol import load_fit_inputs

    references = {}
    for seed in config["data"]["training_seeds"]:
        train, validation, provenance = load_fit_inputs(config, seed, Path(repo_root))
        arrays = {role: _validated_data(values, config, role)
                  for role, values in (("train", train), ("validation", validation))}
        references[(seed, "image")] = {"arrays": arrays, "provenance": provenance,
            "input_hashes": {role: {key: _array_hash(value) for key, value in sorted(data.items())}
                             for role, data in arrays.items()}}
    return references


def authenticate_runs(config: dict, matrix: dict, references: dict, *,
                      repo_root: str | Path = REPO_ROOT) -> list[dict]:
    """Authenticate complete runs and recompute final validation metrics.

``references`` comes from ``load_reference_inputs`` in the public study API;
explicit references also allow isolated synthetic contract tests.
    """
    _validate_matrix(config, matrix)
    repo, records, environment = Path(repo_root).resolve(), [], None
    expected_sources = _source_hashes()
    canonical_validation = None
    geo_states = {}
    required_files = {"run_spec.json", "config.json", "run.json", "metrics.json", "history.json",
                      "final.pt", "resume.pt", "geo_preprocessing.json", "train_predictions.npz",
                      "validation_predictions.npz"}
    for fit in matrix["development_fits"]:
        directory = _inside(repo, fit["output_dir"])
        _require("verification" not in directory.parts, "Verification fits cannot enter selection")
        _require((directory / "manifest.json").is_file(), f"Incomplete fit: {fit['fit_id']}")
        manifest = _verify_bundle(directory)
        _require(set(manifest["files"]) == required_files, "Unexpected or missing bundle payload files")
        spec, run = _read(directory / "run_spec.json"), _read(directory / "run.json")
        _require(_read(directory / "config.json") == config, "Saved configuration differs")
        identity = configuration_hash(spec)
        _require(manifest["run_identity_sha256"] == run["run_identity_sha256"] == identity,
                 "Run identity checksum differs")
        expected = {"config_sha256": configuration_hash(config), "study_id": config["study_id"],
                    "regime": config["regime"], "candidate_id": fit["candidate_id"],
                    "geo_mode": fit["geo_preprocessing"], "learning_rate": fit["learning_rate"],
                    "seed": fit["seed"], "fit_role": "train", "verification": False,
                    "epochs": config["training"]["epochs"], "source_hashes": expected_sources}
        _require(all(spec.get(k) == v for k, v in expected.items()), f"Run recipe differs: {fit['fit_id']}")
        _require(manifest["completed_epoch"] == run["completed_epoch"] == spec["epochs"]
                 and run["status"] == "complete" and run["verification"] is False
                 and manifest["verification"] is False, "Incomplete or verification checkpoint")
        _require(manifest["class_map"] == config["data"]["dense_to_canonical_l3"]
                 and manifest["boundary"] == {"test_data_loaded": False, "encoder_executed": False},
                 "Class order or development boundary differs")
        input_kind = "geo_only" if fit["candidate_id"] == "geo_only" else "image"
        ref = references[(fit["seed"], input_kind)]
        _require(spec["input_hashes"] == ref["input_hashes"]
                 and spec["input_provenance"] == ref["provenance"], "Run inputs or encoder provenance differ")
        current_environment = spec["environment"]
        _require(current_environment.get("deterministic_algorithms") is True
                 and current_environment.get("cublas_workspace_config") == config["training"]["cublas_workspace_config"],
                 "Run lacks deterministic runtime evidence")
        if environment is None:
            environment = current_environment
        _require(current_environment == environment, "Candidate environments differ; record and resolve before selection")
        validation = ref["arrays"]["validation"]
        if canonical_validation is None:
            canonical_validation = {key: validation[key] for key in IDENTITIES}
        _require(all(np.array_equal(validation[k], canonical_validation[k]) for k in IDENTITIES),
                 "Validation identities, labels or plot assignments differ across seeds")
        with np.load(directory / "validation_predictions.npz", allow_pickle=False) as saved:
            _require(set(saved.files) == {*IDENTITIES, "logits", "probabilities", "pred_top3", "prediction"},
                     "Unexpected prediction payload")
            predictions = {key: saved[key].copy() for key in saved.files}
        _require(all(np.array_equal(predictions[k], validation[k]) for k in IDENTITIES),
                 "Prediction row/plot/label alignment differs")
        _require(predictions["logits"].shape == (config["data"]["validation_rows"], config["data"]["num_classes"]),
                 "Validation prediction dimensions differ")
        metrics, recomputed = classification_metrics(validation["label"], predictions["logits"])
        _require(all(np.array_equal(predictions[k], value) for k, value in recomputed.items()),
                 "Saved probabilities or predictions do not match logits")
        saved_metrics, history = _read(directory / "metrics.json"), _read(directory / "history.json")
        _require(set(saved_metrics) == {"train", "validation"}, "Unexpected metric roles")
        _equal_metrics(saved_metrics["validation"], metrics, fit["fit_id"])
        _require([h["epoch"] for h in history] == list(range(1, spec["epochs"] + 1)),
                 "Missing or reordered epoch history")
        _require(all(np.isfinite(h["optimization_loss"]) and all(
            np.isfinite(h[role][key]) for role in ("train", "validation") for key in (*METRICS, "loss"))
            for h in history), "Nonfinite training history")
        for role in ("train", "validation"):
            _equal_metrics(history[-1][role], saved_metrics[role], f"final history/{role}")
        # Direct dataframe geo and cached geo have identical values but can
        # retain different array layouts. NumPy's float64 reduction order can
        # then differ by a few ulps: recompute against each actual producer,
        # never reuse another producer's exact fitted statistics.
        state_key = (fit["seed"], input_kind, fit["geo_preprocessing"])
        if state_key not in geo_states:
            mode = "native" if fit["geo_preprocessing"] == "not_applicable" else fit["geo_preprocessing"]
            geo_states[state_key] = GeoPreprocessor(mode, config["data"]["geo_columns"],
                l2_epsilon=config["geo"]["l2_epsilon"], constant_epsilon=config["geo"]["standardize_constant_epsilon"]
                ).fit(ref["arrays"]["train"]["geo_embedding"], ref["arrays"]["train"]["row_uid"]).to_dict()
        _require(_read(directory / "geo_preprocessing.json") == geo_states[state_key],
                 "Geo preprocessing differs from the allowed fitting rows")
        checkpoint = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)
        _require(checkpoint["epoch"] == spec["epochs"] and checkpoint["model_state"], "Wrong final checkpoint epoch")
        weights = checkpoint["model_state"]
        _require(all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in weights.values()),
                 "Nonfinite checkpoint weights")
        parameters = next(c["expected_trainable_parameters"] for c in config["candidates"]
                          if c["id"] == fit["candidate_id"])
        _require(sum(v.numel() for v in weights.values()) == manifest["trainable_parameters"]
                 == run["trainable_parameters"] == parameters, "Checkpoint parameter count differs")
        records.append({**fit, "metrics": metrics, "history": history, "predictions": predictions,
            "manifest_sha256": sha256_file(directory / "manifest.json"), "run_identity_sha256": identity,
            "checkpoint_sha256": manifest["files"]["final.pt"], "parameters": parameters,
            "runtime_seconds": run["elapsed_seconds"], "peak_cuda_bytes": run["peak_cuda_bytes"],
            "peak_process_rss_bytes": run.get("peak_process_rss_bytes"),
            "peak_process_rss_scope": run.get("peak_process_rss_scope"), "environment": environment})
    return records


def _compact_summaries(summaries: list[dict]) -> list[dict]:
    """Keep full final confusions; epoch curves need only scalar diagnostics."""
    output = []
    for item in summaries:
        value = {key: copy.deepcopy(data) for key, data in item.items() if key != "histories_by_seed"}
        value["histories_by_seed"] = {seed: [{"epoch": row["epoch"],
            "optimization_loss": row["optimization_loss"], **{role: {
                metric: row[role][metric] for metric in (*METRICS, "loss")}
                for role in ("train", "validation")}} for row in history]
            for seed, history in item["histories_by_seed"].items()}
        output.append(value)
    return output


def authenticate_reused_selection(parent_config: dict, parent_matrix: dict,
                                  selected: list[dict], records: list[dict], *, repo_root: Path) -> list[dict]:
    """Match immutable original references against freshly recomputed winners."""
    analysis_dir = _inside(repo_root, str(Path(parent_config["paths"]["artifact_root"]) / "reports/m3_comparison_v1"))
    manifest = _read(analysis_dir / "manifest.json")
    _require(manifest.get("status") == "complete" and manifest.get("manifest_payload_sha256") ==
        configuration_hash({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}),
        "Original analysis manifest differs")
    _require(manifest["provenance"]["config_sha256"] == configuration_hash(parent_config)
             and manifest["provenance"]["fit_manifests"] == {r["fit_id"]: r["manifest_sha256"] for r in records},
             "Original analysis is not bound to the authenticated original fits")
    for name, digest in manifest["files"].items():
        path = (analysis_dir / name).resolve()
        _require(path.is_relative_to(analysis_dir) and sha256_file(path) == digest,
                 f"Original analysis payload changed: {name}")
    selection = _read(analysis_dir / "selection.json")
    _require(selection.get("selection_payload_sha256") == configuration_hash({k: v for k, v in selection.items()
        if k != "selection_payload_sha256"}) and selection.get("all_declared_fits_complete") is True
        and selection.get("config_sha256") == configuration_hash(parent_config), "Original selection identity differs")
    expected = _selected_references(parent_config, parent_matrix, selected, records,
                                    selection["contrast_manifest_sha256"])
    _require(selection["selected_references"] == expected,
             "Original selected checkpoints differ from recomputed learning-rate winners")
    for ref in expected:
        _require(_read(_inside(repo_root, ref["selection_path"])) == ref, "Original selected reference changed")
    return expected


def _assert_shared_evidence(old_records: list[dict], new_records: list[dict]) -> None:
    _require(bool(old_records) and bool(new_records), "Both original and extension records are required")
    reference = old_records[0]
    for record in [*old_records, *new_records]:
        _require(record["environment"] == reference["environment"],
                 "Original and bypass runtime environments differ; resolve before comparison")
        _require(all(np.array_equal(record["predictions"][key], reference["predictions"][key]) for key in IDENTITIES),
                 "Original and bypass validation identities or labels differ")


def analyze_study(config: dict, matrix: dict, contrast_manifest_path: str | Path,
                  output_dir: str | Path, *, repo_root: str | Path = REPO_ROOT) -> dict:
    """Authenticate 300 fits; select six new arms and retain 19 original arms."""
    from .attention_fusion_bypass_protocol import DEFAULT_CONFIG, read_frozen_plan, verify_parent_bindings
    from tools.run_multimodal_attention_fusion import read_frozen_plan as read_parent_plan

    repo, output = Path(repo_root).resolve(), Path(output_dir).resolve()
    _require(not output.exists(), "Analysis exists; preserve it and use a new destination")
    _require(output.is_relative_to(_inside(repo, config["paths"]["artifact_root"])), "Analysis must stay in the extension directory")
    frozen, frozen_matrix, protocol = read_frozen_plan(DEFAULT_CONFIG, repo)
    _require(config == frozen and matrix == frozen_matrix, "Analysis differs from frozen extension protocol")
    parent_bindings = verify_parent_bindings(config, repo)
    path = Path(contrast_manifest_path).resolve()
    _require(path == _inside(repo, config["paths"]["artifact_root"]) / "protocol/contrasts.json",
             "Analysis must use the protocol-bound contrast declaration")
    _require(protocol["files"].get("contrasts.json") == sha256_file(path), "Protocol contrast binding differs")
    saved = _read(path)
    _verify_declaration(saved)
    declaration = declare_contrasts(config, matrix)
    _require(saved["declaration"] == declaration, "Contrast declaration differs from current extension")
    parent, parent_matrix, _ = read_parent_plan(Path(config["source_study"]["config"]), repo)
    _require(configuration_hash(parent) == config["source_study"]["config_sha256"], "Original configuration changed")
    old_records = original.authenticate_runs(parent, parent_matrix,
        original.load_reference_inputs(parent, repo_root=repo), repo_root=repo)
    old_summaries, old_selected = select_learning_rates(parent, parent_matrix, old_records)
    old_refs = authenticate_reused_selection(parent, parent_matrix, old_selected, old_records, repo_root=repo)
    records = authenticate_runs(config, matrix, load_reference_inputs(config, repo_root=repo), repo_root=repo)
    _assert_shared_evidence(old_records, records)
    summaries, selected = select_learning_rates(config, matrix, records)
    selected_refs = _selected_references(config, matrix, selected, records, sha256_file(path))
    for item in old_selected:
        item["study_role"] = "reused_original"
    for item in selected:
        item["study_role"] = "new_bypass"
    combined = [*old_selected, *selected]
    contrasts, draws, weights = compare_selected(config, combined, [*old_records, *records], declaration)
    labels, plots = records[0]["predictions"]["label"], records[0]["predictions"]["plot_idx"]
    audit = _read(_inside(repo, parent["paths"]["artifact_root"]) / "protocol/input_audit.json")
    ontology = sorted(audit["assignments"]["class_map"], key=lambda row: row["label_id_dense"])
    _require([r["canonical_l3_id"] for r in ontology] == config["data"]["dense_to_canonical_l3"], "Class names misaligned")
    analysis = {"schema_version": "attention_fusion.bypass_comparison.v1", "study_id": config["study_id"],
        "class_order": config["data"]["dense_to_canonical_l3"], "class_labels": [r["label_name"] for r in ontology],
        "training_seeds": config["data"]["training_seeds"], "metrics_order": list(METRICS),
        "validation_rows": len(labels), "validation_plots": len(set(plots)),
        "validation_class_support_images": np.bincount(labels, minlength=config["data"]["num_classes"]).tolist(),
        "validation_class_support_plots": [len(set(plots[labels == c])) for c in range(config["data"]["num_classes"])],
        "counts": {"development_fits": len(records), "reused_development_fits": len(old_records),
            "total_authenticated_fits": len(old_records) + len(records), "new_selected_arms": len(selected),
            "reused_selected_arms": len(old_selected), "selected_arms": len(combined),
            "selected_checkpoints": len(selected_refs), "reused_selected_checkpoints": len(old_refs), "contrasts": len(contrasts)},
        "primary_contrast_ids": declaration["primary_contrast_ids"],
        "lr_candidates": _compact_summaries(summaries), "reused_lr_candidates": _compact_summaries(old_summaries),
        "selected_arms": _compact_summaries(combined), "contrasts": contrasts,
        "bootstrap": {**config["metrics"]["bootstrap"], "plot_order": sorted(set(plots.tolist())),
            "draw_axis_order": ["replicate", "contrast", "metric"], "contrast_order": [c["contrast_id"] for c in contrasts],
            "metric_order": list(METRICS)},
        "source_study": config["source_study"],
        "provenance": {"config_sha256": configuration_hash(config), "matrix_sha256": configuration_hash(matrix),
            "protocol_manifest_payload_sha256": protocol["manifest_payload_sha256"],
            "contrast_manifest_sha256": sha256_file(path), "analysis_source_sha256": sha256_file(Path(__file__)),
            "original_comparison_source_sha256": sha256_file(Path(original.__file__)),
            "training_source_sha256": _source_hashes(), "numpy": np.__version__, "environment": records[0]["environment"],
            "fit_manifests": {r["fit_id"]: r["manifest_sha256"] for r in records},
            "reused_fit_manifests": {r["fit_id"]: r["manifest_sha256"] for r in old_records},
            "parent_evidence_sha256": configuration_hash(parent_bindings),
            "test_features_or_labels_read": False, "all_candidate_bundles_authenticated": True},
        "limitations": [declaration["claim_scope"], declaration["multiple_comparison_adjustment"], declaration["attribution"],
            "Bootstrap intervals exclude recipe-selection and new-split uncertainty; fixed seeds are not resampled.",
            "No practical-effect threshold or class-specific uncertainty was declared."]}
    selection = {"study_id": config["study_id"], "config_sha256": configuration_hash(config),
        "selection_rule": config["selection"], "all_declared_fits_complete": True,
        "selected_references": selected_refs, "reused_selected_references": old_refs,
        "contrast_manifest_sha256": sha256_file(path)}
    selection["selection_payload_sha256"] = configuration_hash(selection)
    for ref in selected_refs:
        target = _inside(repo, ref["selection_path"])
        _require(not target.exists() or _read(target) == ref, "Existing extension reference differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write(staging / "analysis.json", analysis)
        _write(staging / "selection.json", selection)
        _write(staging / "contrasts.json", saved)
        np.savez_compressed(staging / "bootstrap_replicates.npz", differences=draws, plot_multiplicities=weights)
        tables = staging / "tables"
        tables.mkdir()
        for name, values in (("all_new_fits", records), ("reused_fits", old_records)):
            _csv(tables / f"{name}.csv", [{"fit_id": r["fit_id"], "arm_id": r["arm_id"],
                "learning_rate": r["learning_rate"], "seed": r["seed"], **{k: r["metrics"][k] for k in METRICS},
                "parameters": r["parameters"], "runtime_seconds": r["runtime_seconds"],
                "peak_cuda_bytes": r["peak_cuda_bytes"], "bundle_manifest_sha256": r["manifest_sha256"]} for r in values])
        for name, values in (("learning_rates", summaries), ("reused_learning_rates", old_summaries), ("selected_arms", combined)):
            _csv(tables / f"{name}.csv", [{"arm_id": a["arm_id"], "learning_rate": a["learning_rate"],
                **{f"mean_{k}": a["mean"][k] for k in METRICS},
                **{f"sample_std_{k}": a["sample_std"][k] for k in METRICS}} for a in values])
        _csv(tables / "paired_contrasts.csv", [{"contrast_id": c["contrast_id"], "kind": c["kind"], "metric": k,
            "mean_difference": c["mean_difference"][k], "lower_95": c["interval95"][k][0],
            "upper_95": c["interval95"][k][1], "category": c["category"]} for c in contrasts for k in METRICS])
        manifest = {"status": "complete", "study_id": config["study_id"], "counts": analysis["counts"],
            "files": {str(p.relative_to(staging)): sha256_file(p) for p in sorted(staging.rglob("*")) if p.is_file()},
            "provenance": analysis["provenance"]}
        manifest["manifest_payload_sha256"] = configuration_hash(manifest)
        _write(staging / "manifest.json", manifest)
        for ref in selected_refs:
            target = _inside(repo, ref["selection_path"])
            if not target.exists():
                _write_new_json(target, ref)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis

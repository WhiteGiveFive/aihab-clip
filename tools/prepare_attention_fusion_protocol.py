#!/usr/bin/env python3
"""Freeze the attention-fusion study plan without loading model or data payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/multimodal_attention_fusion.yaml")
REPORT = Path("dev_plans/attention_fusion/01_experimental_protocol.md")
GEO_ARMS = ["native", "l2", "standardize"]
CANDIDATES = {
    "image_only": (False, "mlp", 299794),
    "geo_only": (True, "mlp", 21266),
    "raw_concat": (True, "mlp", 316178),
    "projected_concat": (True, "adapters_then_concat", 227090),
    "tokenwise_capacity": (True, "two_preln_tokenwise_residual_ffns", 359442),
    "attention_two_token": (True, "self_attention", 359570),
    "attention_cls": (True, "self_attention", 459410),
}
TOP_LEVEL_KEYS = {
    "schema_version", "study_id", "method", "regime", "protocol_date", "paths",
    "data", "image", "geo", "model", "candidates", "training", "selection",
    "metrics", "interpretation", "final_fit", "artifacts", "budget", "joint_extension",
}


class ProtocolError(ValueError):
    """The requested plan does not satisfy the frozen study contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


def value_at(config: dict[str, Any], dotted: str) -> Any:
    current: Any = config
    for key in dotted.split("."):
        require(isinstance(current, dict) and key in current, f"Missing {dotted}")
        current = current[key]
    return current


def validate_config(config: dict[str, Any]) -> None:
    """Validate scientific boundaries and the finite, fully enumerated design."""
    require(isinstance(config, dict), "Configuration must be a mapping")
    require(set(config) == TOP_LEVEL_KEYS, "Unexpected or missing configuration sections")
    try:
        canonical_json(config)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Configuration must contain finite JSON-compatible values") from exc
    fixed = {
        "schema_version": 1, "study_id": "attention_fusion_v1",
        "method": "attention_fusion", "regime": "frozen_cached",
        "data.image_dim": 1152, "data.geo_dim": 64, "data.num_classes": 18,
        "data.train_rows": 3378, "data.validation_rows": 822,
        "data.development_rows": 4200, "data.training_seeds": [1, 2, 3, 4],
        "data.train_plots": 1300, "data.validation_plots": 325,
        "data.development_plots": 1625, "data.allow_test_features_or_labels": False,
        "data.oof_folds_used": False,
        "data.geo_columns": [f"A{i:02d}" for i in range(64)],
        "data.dense_to_canonical_l3": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19],
        "image.encoder_fit_role": "train", "image.encoder_frozen": True,
        "image.normalization": "l2", "image.extraction_views": 1,
        "image.extraction_mode": "eval", "image.extraction_precision": "float32",
        "image.extraction_device": "cuda", "image.extraction_batch_size": 16,
        "image.extraction_cudnn_allow_tf32": True,
        "image.extraction_matmul_allow_tf32": False,
        "geo.default": "native", "geo.comparison": GEO_ARMS,
        "geo.standardize_fit_role": "train", "geo.standardize_ddof": 0,
        "geo.standardize_weighting": "image_rows",
        "model.token_dim": 128, "model.heads": 4, "model.feedforward_dim": 256,
        "model.classifier_hidden_dim": 256, "model.norm_first": True,
        "model.linear_bias": True, "model.final_encoder_norm": False,
        "model.adapter": "linear_gelu_layernorm", "model.adapter_dropout": 0.0,
        "model.modality_type_embeddings": True, "model.activation": "gelu_exact",
        "model.layernorm_epsilon": 1.0e-5, "model.dropout": 0.1,
        "model.classifier": "linear_relu_dropout_linear",
        "model.token_order": ["image", "geo"],
        "training.epochs": 50, "training.checkpoint_epoch": 50,
        "training.early_stopping": False, "training.validation_selects_epoch": False,
        "training.learning_rates": [0.0001, 0.0003, 0.001],
        "training.save_every_candidate_final_checkpoint": True,
        "selection.scope": "one_learning_rate_per_candidate_and_geo_arm_across_all_seeds",
        "selection.rank_descending": ["mean_validation_acc1", "mean_validation_weighted_f1", "mean_validation_mcc"],
        "selection.primary_geo_arm": "native", "selection.select_best_seed": False,
        "selection.use_unrounded_metrics": True,
        "selection.primary_contrast": "attention_two_token_minus_projected_concat",
        "metrics.required": ["acc1", "acc3", "weighted_f1", "mcc", "confusion_matrix"],
        "metrics.confusion_order": "fixed_18_classes",
        "final_fit.routes": ["development_train", "all_development"],
        "final_fit.retained_train_checkpoint": "selected_epoch50_existing_checkpoint",
        "final_fit.all_development_rows": 4200,
        "final_fit.keep_same_train_partition_encoder": True,
        "final_fit.refit_components": ["adapters_if_present", "fusion_if_present", "classifier"],
        "final_fit.initialization": "fresh_same_seed_and_selected_recipe",
        "final_fit.epochs": 50, "final_fit.reuse_selected_learning_rate": True,
        "final_fit.standardize_refit_on_all_development": True,
        "final_fit.candidates": "all_19_selected_candidate_geo_arms",
        "final_fit.new_fits": 76,
        "final_fit.compare_routes_on": "separate_exploratory_test_application",
        "artifacts.keep_all_candidate_checkpoints": True,
        "artifacts.overwrite_completed_runs": False,
        "budget.candidate_geo_arms": 19, "budget.learning_rates_per_arm": 3,
        "budget.seeds": 4, "budget.development_fits": 228,
        "budget.development_epochs_total": 11400,
        "budget.selected_train_checkpoints": 76, "budget.all_development_refits": 76,
    }
    for dotted, expected in fixed.items():
        actual = value_at(config, dotted)
        require(type(actual) is type(expected) and actual == expected,
                f"Frozen value {dotted} must equal {expected!r}")
    candidates = config["candidates"]
    require(isinstance(candidates, list) and len(candidates) == 7, "Exactly seven candidates required")
    require([c.get("id") for c in candidates] == list(CANDIDATES),
            "Candidate IDs/order must match the seven frozen architectures")
    for candidate in candidates:
        geo, architecture, count = CANDIDATES[candidate["id"]]
        require(candidate.get("geo") is geo and candidate.get("architecture") == architecture,
                f"Invalid architecture/geo declaration: {candidate['id']}")
        require(candidate.get("expected_trainable_parameters") == count,
                f"Unexpected analytic parameter count: {candidate['id']}")
    by_id = {candidate["id"]: candidate for candidate in candidates}
    for candidate_id, width in (("image_only", 1152), ("geo_only", 64), ("raw_concat", 1216)):
        require(by_id[candidate_id].get("input_dim") == width,
                f"Wrong input width for {candidate_id}")
    require(by_id["attention_two_token"].get("layers") == 1
            and by_id["attention_two_token"].get("readout") == "updated_modality_concat",
            "Primary attention must use one layer and updated modality readout")
    require(by_id["attention_cls"].get("layers") == 2
            and by_id["attention_cls"].get("readout") == "updated_cls"
            and by_id["attention_cls"].get("cls_type_embedding") is False,
            "CLS attention must use two layers and its declared readout")
    require(by_id["tokenwise_capacity"].get("readout") == "updated_modality_concat",
            "Token-wise control cannot use input-independent CLS-only readout")
    for rate in config["training"]["learning_rates"]:
        require(math.isfinite(rate) and rate > 0, "Learning rates must be finite and positive")
    require(set(config["paths"]) == {
        "artifact_root", "source_protocol_root", "assignments", "development_geo_tables",
        "image_root", "encoder_template", "validation_cache_template", "prepared_cache_template",
    }, "Paths must use the declared development-only schema")
    for value in config["paths"].values():
        for name in value if isinstance(value, list) else [value]:
            require(isinstance(name, str) and not Path(name).is_absolute()
                    and ".." not in Path(name).parts,
                    "Configured paths must be repository-relative without traversal")


def audit_parent_hashes(audit: dict[str, Any]) -> dict[str, str]:
    """Copy authenticated lineage records; never reopen their payloads here."""
    found: dict[str, str] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if "path" in node and "sha256" in node:
                path, digest = node["path"], node["sha256"]
                require(isinstance(digest, str) and len(digest) == 64
                        and all(c in "0123456789abcdef" for c in digest), "Malformed parent SHA-256")
                require(path not in found or found[path] == digest, "Conflicting parent hashes")
                found[path] = digest
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    for key in ("protocol_manifest", "protocol_files", "label_blind_test_identity",
                "development_geo_source_files", "encoders", "validation_image_caches"):
        visit(audit[key])
    require(bool(found), "Audit must contain parent input hashes")
    return dict(sorted(found.items()))


def validate_audit(audit: dict[str, Any], config: dict[str, Any], repo: Path) -> dict[str, str]:
    require(audit.get("status") == "passed", "Input audit must have passed")
    require(audit.get("schema_version") == "attention_fusion.input_audit.v1"
            and audit.get("study_id") == config["study_id"], "Input audit identity mismatch")
    checks = audit.get("checks")
    require(isinstance(checks, dict) and bool(checks)
            and all(passed is True for passed in checks.values()), "Input audit has missing/failed checks")
    require(audit.get("summary", {}).get("test_features_or_labels_opened") is False,
            "Audit must explicitly exclude test feature/label access")
    for key in ("test_labels_read", "test_features_read", "mixed_embedding_npz_opened"):
        require(audit.get("boundary", {}).get(key) is False, f"Audit boundary failed: {key}")

    def match(recorded: str, configured: str) -> None:
        require(Path(recorded).resolve() == (repo / configured).resolve(),
                f"Configured input differs from audited input: {configured}")

    paths = config["paths"]
    match(audit["repo_root"], ".")
    match(audit["source_protocol_root"], paths["source_protocol_root"])
    for key, relative in (("script", "tools/audit_attention_fusion_inputs.py"),
                          ("canonical_helpers", "multimodal/geo_helpfulness_protocol.py")):
        recorded = audit.get("runtime", {}).get(key, {})
        require("path" in recorded and "sha256" in recorded, f"Missing audit runtime record: {key}")
        match(recorded["path"], relative)
        require(file_hash(repo / relative) == recorded["sha256"],
                f"Audit source changed; rerun input audit: {relative}")
    match(audit["protocol_files"]["development_assignments.parquet"]["path"], paths["assignments"])
    require(len(audit["development_geo_source_files"]) == 2, "Expected two development source tables")
    for record, path in zip(audit["development_geo_source_files"], paths["development_geo_tables"]):
        match(record["path"], path)
    require(len(paths["development_geo_tables"]) == 2, "Expected train/val development sources")
    match(str(repo / audit["development_images"]["image_source"]), paths["image_root"])
    require(audit["native_geo"]["feature_columns"] == config["data"]["geo_columns"],
            "Audited geo feature order differs")
    for records, key, template in (
        (audit["encoders"], "checkpoint", paths["encoder_template"]),
        (audit["validation_image_caches"], "features", paths["validation_cache_template"]),
    ):
        require([record["seed"] for record in records] == [1, 2, 3, 4], "Audit must cover all four seeds")
        for record in records:
            match(record[key]["path"], template.format(seed=record["seed"]))
    return audit_parent_hashes(audit)


def build_matrix(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    root = Path(config["paths"]["artifact_root"])
    regime = config["regime"]
    fits, selected, refits, arms = [], [], [], []
    for candidate in config["candidates"]:
        for geo in GEO_ARMS if candidate["geo"] else ["not_applicable"]:
            arm_id = f"{candidate['id']}__geo_{geo}"
            arms.append({"arm_id": arm_id, "candidate_id": candidate["id"], "geo_preprocessing": geo})
            for seed in config["data"]["training_seeds"]:
                inputs = {"encoder_checkpoint": config["paths"]["encoder_template"].format(seed=seed),
                          "assignments": config["paths"]["assignments"],
                          "geo_tables": config["paths"]["development_geo_tables"] if candidate["geo"] else []}
                if candidate["id"] == "geo_only":
                    inputs["encoder_checkpoint"] = None
                    inputs["image_caches"] = {}
                else:
                    inputs["image_caches"] = {
                        role: config["paths"]["prepared_cache_template"].format(seed=seed, role=role)
                        for role in ("train", "validation")
                    }
                base = {"arm_id": arm_id, "candidate_id": candidate["id"], "regime": regime,
                        "geo_preprocessing": geo, "seed": seed, "epochs": 50,
                        "checkpoint_epoch": 50, "inputs": inputs, "status": "planned"}
                sources = []
                for rate in config["training"]["learning_rates"]:
                    rate_id = format(rate, ".0e")
                    fit_id = f"{arm_id}__lr_{rate_id}__seed{seed}"
                    sources.append(fit_id)
                    fits.append({**base, "fit_id": fit_id, "fit_role": "train",
                                 "fit_rows": 3378, "evaluation_role": "validation",
                                 "learning_rate": rate,
                                 "output_dir": str(root / "development" / regime / candidate["id"] / geo / f"lr_{rate_id}" / f"seed{seed}")})
                selected.append({**base, "fit_role": "train", "fit_rows": 3378,
                                 "learning_rate": None, "selection_pending": True,
                                 "candidate_fit_ids": sources, "retrain": False,
                                 "output_dir": str(root / "selection" / regime / candidate["id"] / geo / f"seed{seed}")})
                refits.append({**base, "fit_role": "all_development", "fit_rows": 4200,
                               "evaluation_role": None, "learning_rate": None,
                               "selection_pending": True, "initialization": "fresh_same_seed_and_selected_recipe",
                               "refit_geo_statistics": geo == "standardize",
                               "output_dir": str(root / "final_fit" / regime / candidate["id"] / geo / f"seed{seed}")})
    require((len(arms), len(fits), len(selected), len(refits)) == (19, 228, 76, 76),
            "Expanded workload disagrees with frozen budget")
    return {"schema_version": "attention_fusion.candidate_matrix.v1",
            "study_id": config["study_id"], "status": "plan_not_execution_record",
            "candidate_geo_arms": arms, "development_fits": fits,
            "selected_train_checkpoint_references": selected, "all_development_refits": refits,
            "counts": {"candidate_geo_arms": 19, "development_fits": 228,
                       "selected_train_checkpoint_references": 76, "all_development_refits": 76}}


def atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_protocol(config_path: Path, repo: Path = REPO_ROOT) -> dict[str, Any]:
    repo = repo.resolve()
    config_path = config_path.resolve() if config_path.is_absolute() else repo / config_path
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    root = (repo / config["paths"]["artifact_root"]).resolve()
    require(root.is_relative_to(repo), "Artifact directory must remain within repository")
    directory = root / "protocol"
    audit_path = directory / "input_audit.json"
    audit = json.loads(audit_path.read_text())
    parents = validate_audit(audit, config, repo)
    matrix = build_matrix(config)
    effective_hash = content_hash(config)
    manifest_path = directory / "protocol_manifest.json"
    started = any(path.is_file() for name in ("development", "selection", "final_fit")
                  for path in (root / name).rglob("*"))
    if started:
        require(manifest_path.is_file(), "Existing runs have no frozen protocol manifest")
        previous = json.loads(manifest_path.read_text())
        require(previous.get("effective_config_sha256") == effective_hash,
                "Study already has runs; changed configuration requires a new study ID")
        require(previous.get("parent_input_hashes") == parents,
                "Study already has runs; changed input lineage requires a new study ID")
    report_path = repo / REPORT
    require(report_path.is_file(), "M1 report must exist before freezing the protocol")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True, check=False)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                capture_output=True, text=True, check=False)
    directory.mkdir(parents=True, exist_ok=True)
    resolved_path = directory / "resolved_protocol.yaml"
    matrix_path = directory / "candidate_matrix.json"
    atomic_text(resolved_path, yaml.safe_dump(config, sort_keys=False))
    atomic_text(matrix_path, json.dumps(matrix, indent=2, allow_nan=False) + "\n")
    manifest = {
        "schema_version": "attention_fusion.protocol_manifest.v1",
        "study_id": config["study_id"], "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(), "repo_root": str(repo),
        "effective_config_sha256": effective_hash,
        "configuration": file_record(config_path), "report": file_record(report_path),
        "input_audit": file_record(audit_path), "preparer": file_record(Path(__file__)),
        "parent_input_hashes": parents,
        "parent_hash_scope": "Authenticated records copied from input audit; payloads not reopened",
        "outputs": {"resolved_protocol": file_record(resolved_path), "candidate_matrix": file_record(matrix_path)},
        "counts": matrix["counts"], "existing_run_files": started,
        "boundary": {"model_loaded": False, "data_payload_loaded": False,
                     "test_features_or_labels_loaded": False, "models_fitted_by_preparer": 0},
        "runtime": {"python": platform.python_version(), "pyyaml": yaml.__version__,
                    "git_head": git_head.stdout.strip() if git_head.returncode == 0 else None,
                    "git_dirty": bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
                    "git_status_porcelain": git_status.stdout.splitlines() if git_status.returncode == 0 else None},
    }
    manifest["manifest_payload_sha256"] = content_hash(manifest)
    atomic_text(manifest_path, json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_protocol(args.config)
    except (ProtocolError, OSError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        parser.exit(1, f"Protocol preparation failed: {exc}\n")
    print(json.dumps({"study_id": manifest["study_id"], "status": manifest["status"],
                      "counts": manifest["counts"],
                      "effective_config_sha256": manifest["effective_config_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Separate frozen protocol for the two CLS modality-bypass extensions."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from tools.prepare_attention_fusion_protocol import content_hash, file_hash
from tools.run_multimodal_attention_fusion import read_frozen_plan as read_parent_plan


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/multimodal_attention_fusion_cls_bypass.yaml")
PARENT_CONFIG = Path("configs/multimodal_attention_fusion.yaml")
STUDY_ID = "attention_fusion_cls_bypass_v1"
CANDIDATES = (
    {"id": "attention_cls_skip_adapted", "geo": True, "architecture": "self_attention", "layers": 2,
     "readout": "updated_cls_plus_original_typed_tokens", "cls_type_embedding": False,
     "expected_trainable_parameters": 524946},
    {"id": "attention_cls_skip_original", "geo": True, "architecture": "self_attention", "layers": 2,
     "readout": "updated_cls_plus_original_embeddings", "cls_type_embedding": False,
     "expected_trainable_parameters": 770706},
)
SOURCE_FILES = (
    "multimodal/attention_fusion_bypass.py", "multimodal/attention_fusion_bypass_training.py",
    "multimodal/attention_fusion_bypass_protocol.py", "tools/run_attention_fusion_bypass_study.py",
    "multimodal/attention_fusion_bypass_comparison.py", "multimodal/attention_fusion_comparison.py",
    "multimodal/attention_fusion.py", "multimodal/attention_fusion_training.py", "multimodal/models.py",
    "multimodal/attention_fusion_inputs.py", "multimodal/geo_helpfulness_protocol.py",
    "tools/run_attention_fusion_study.py", "tools/run_multimodal_attention_fusion.py",
    "tools/prepare_attention_fusion_protocol.py", "tools/audit_attention_fusion_inputs.py",
)


class BypassProtocolError(ValueError):
    """The extension or its reused evidence differs from the frozen contract."""


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise BypassProtocolError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def path_in_repo(repo: Path, relative: str | Path) -> Path:
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, "Configured paths must be repository-relative")
    result = (repo / value).resolve()
    require(result.is_relative_to(repo.resolve()), "Configured path escapes repository")
    return result


def publish_json(path: Path, value: dict) -> None:
    """Atomic, immutable publication; an existing file is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_config(parent: dict) -> dict:
    """The bounded extension changes only identity, readouts and study budget."""
    config = copy.deepcopy(parent)
    parent_root = parent["paths"]["artifact_root"]
    config.update(study_id=STUDY_ID, protocol_date="2026-10-16",
        initialization_seed_namespace=parent["study_id"], source_study={
            "config": str(PARENT_CONFIG), "artifact_root": parent_root,
            "config_sha256": content_hash(parent), "analysis_dir": str(Path(parent_root) / "reports/m3_comparison_v1"),
            "proposed_after_original_m3_results": True,
            "reuse_comparators": "authenticated_original_selected_checkpoints"})
    config["paths"]["artifact_root"] = str(Path(parent_root).parent / STUDY_ID)
    config["candidates"] = copy.deepcopy(list(CANDIDATES))
    config["selection"].pop("primary_contrast")
    config["selection"]["primary_contrasts"] = [f"{c['id']}_minus_attention_cls" for c in CANDIDATES]
    config["selection"]["supporting_comparators"] = ["projected_concat", "raw_concat", "attention_two_token", "image_only"]
    config["interpretation"]["attention_specific_claim"] = "not_identified_readout_feature_access_and_classifier_capacity_change_together"
    config["final_fit"] = {"routes": ["development_train"], "retained_train_checkpoint": "selected_epoch50_existing_checkpoint",
                           "all_development_refits": 0, "test_evaluation": False}
    config["budget"].update(candidate_geo_arms=6, development_fits=72, development_epochs_total=3600,
                            selected_train_checkpoints=24, all_development_refits=0)
    return config


def validate_config(config: dict, parent: dict) -> None:
    require(parent["study_id"] == "attention_fusion_v1", "Unexpected parent study")
    require(config == expected_config(parent), "Bypass configuration differs from its declared extension contract")
    require(config["paths"]["artifact_root"] != parent["paths"]["artifact_root"], "Bypass output must not replace parent study")


def load_config(config_path: str | Path = DEFAULT_CONFIG, repo: Path = REPO_ROOT) -> tuple[dict, dict, dict, dict]:
    repo = Path(repo).resolve()
    parent, parent_matrix, parent_protocol = read_parent_plan(PARENT_CONFIG, repo)
    path = Path(config_path)
    config = yaml.safe_load((path if path.is_absolute() else repo / path).read_text())
    validate_config(config, parent)
    path_in_repo(repo, config["paths"]["artifact_root"])
    return config, parent, parent_matrix, parent_protocol


def build_matrix(config: dict) -> dict:
    root = Path(config["paths"]["artifact_root"])
    arms, fits, selected = [], [], []
    for candidate in config["candidates"]:
        for geo in config["geo"]["comparison"]:
            arm_id = f"{candidate['id']}__geo_{geo}"
            arms.append({"arm_id": arm_id, "candidate_id": candidate["id"], "geo_preprocessing": geo})
            for seed in config["data"]["training_seeds"]:
                base = {"arm_id": arm_id, "candidate_id": candidate["id"], "regime": config["regime"],
                        "geo_preprocessing": geo, "seed": seed, "epochs": 50, "checkpoint_epoch": 50,
                        "fit_role": "train", "fit_rows": config["data"]["train_rows"], "evaluation_role": "validation",
                        "inputs": {"source_config_sha256": config["source_study"]["config_sha256"],
                                   "assignments": config["paths"]["assignments"],
                                   "encoder_checkpoint": config["paths"]["encoder_template"].format(seed=seed),
                                   "image_caches": {role: config["paths"]["prepared_cache_template"].format(seed=seed, role=role)
                                                    for role in ("train", "validation")}}, "status": "planned"}
                fit_ids = []
                for rate in config["training"]["learning_rates"]:
                    rate_id = format(rate, ".0e")
                    fit_id = f"{arm_id}__lr_{rate_id}__seed{seed}"
                    fits.append({**base, "fit_id": fit_id, "learning_rate": rate,
                        "output_dir": str(root / "development" / config["regime"] / candidate["id"] / geo / f"lr_{rate_id}" / f"seed{seed}")})
                    fit_ids.append(fit_id)
                selected.append({**base, "learning_rate": None, "selection_pending": True, "retrain": False,
                    "candidate_fit_ids": fit_ids,
                    "output_dir": str(root / "selection" / config["regime"] / candidate["id"] / geo / f"seed{seed}")})
    require((len(arms), len(fits), len(selected)) == (6, 72, 24), "Bypass budget must be six arms, 72 fits and 24 selected references")
    return {"schema_version": "attention_fusion.bypass_matrix.v1", "study_id": config["study_id"],
            "candidate_geo_arms": arms, "development_fits": fits, "selected_train_checkpoint_references": selected,
            "all_development_refits": [], "counts": {"candidate_geo_arms": 6, "development_fits": 72,
                "selected_train_checkpoint_references": 24, "all_development_refits": 0}}


def source_hashes(repo: Path = REPO_ROOT) -> dict[str, str]:
    return {name: file_hash(Path(repo) / name) for name in SOURCE_FILES}


def _authenticate_parent_evidence(parent: dict, matrix: dict, protocol: dict, repo: Path) -> dict:
    """Verify original completed bundles, selected references and separate caches."""
    from multimodal.attention_fusion_inputs import load_input_provenance, load_prepared_inputs
    from tools.run_attention_fusion_study import authenticate_complete, source_hashes as parent_sources

    files = {}

    def bind(path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        require(path.is_relative_to(repo), "Parent evidence must stay inside repository")
        digest = file_hash(path)
        require(expected is None or digest == expected, f"Parent evidence changed: {path}")
        files[str(path.relative_to(repo))] = digest
        return digest

    parent_root = path_in_repo(repo, parent["paths"]["artifact_root"])
    for name in ("protocol_manifest.json", "input_audit.json", "candidate_matrix.json", "resolved_protocol.yaml"):
        bind(parent_root / "protocol" / name)
    sources = parent_sources(repo)
    fit_manifests = {}
    for fit in matrix["development_fits"]:
        directory = path_in_repo(repo, fit["output_dir"])
        authenticated = authenticate_complete(directory, fit, parent, sources)
        fit_manifests[fit["fit_id"]] = bind(directory / "manifest.json", authenticated["manifest_sha256"])
        # authenticate_complete has just hashed these bytes. Retain every
        # member for the next preflight and the inexpensive mid-run stat guard.
        for name, digest in read_json(directory / "manifest.json")["files"].items():
            path = (directory / name).resolve()
            require(path.is_relative_to(directory), "Original bundle path traversal")
            files[str(path.relative_to(repo))] = digest
    require(len(fit_manifests) == 228, "All 228 original comparator candidates must be accounted for")
    analysis_root = parent_root / "reports/m3_comparison_v1"
    analysis_manifest = read_json(analysis_root / "manifest.json")
    require(analysis_manifest.get("status") == "complete" and analysis_manifest.get("manifest_payload_sha256")
            == content_hash({k: v for k, v in analysis_manifest.items() if k != "manifest_payload_sha256"}),
            "Original analysis manifest is incomplete or changed")
    require(analysis_manifest["provenance"]["config_sha256"] == content_hash(parent)
            and analysis_manifest["provenance"]["fit_manifests"] == fit_manifests,
            "Original selected analysis is not bound to the authenticated 228 fits")
    provenance = analysis_manifest["provenance"]
    expected_training_sources = {Path(name).name: digest for name, digest in sources.items()
        if name in {"multimodal/attention_fusion_training.py", "multimodal/attention_fusion.py", "multimodal/models.py"}}
    require(provenance["matrix_sha256"] == content_hash(matrix)
            and provenance["protocol_manifest_payload_sha256"] == protocol["manifest_payload_sha256"]
            and provenance["analysis_source_sha256"] == file_hash(repo / "multimodal/attention_fusion_comparison.py")
            and provenance["training_source_sha256"] == expected_training_sources
            and provenance["all_candidate_bundles_authenticated"] is True
            and provenance["test_features_or_labels_read"] is False,
            "Original analysis protocol, source or validation boundary differs")
    bind(parent_root / "reports/m3_contrast_declaration.json", provenance["contrast_manifest_sha256"])
    bind(analysis_root / "manifest.json")
    for name, digest in analysis_manifest["files"].items():
        path = (analysis_root / name).resolve()
        require(path.is_relative_to(analysis_root), "Original analysis path traversal")
        bind(path, digest)
    selection = read_json(analysis_root / "selection.json")
    require(selection["selection_payload_sha256"] == content_hash({k: v for k, v in selection.items()
            if k != "selection_payload_sha256"}) and selection["all_declared_fits_complete"] is True,
            "Original selection checksum or completion differs")
    require(len(selection["selected_references"]) == 76, "Original selection must retain all 76 checkpoints")
    planned_refs = {(row["arm_id"], row["seed"]): row for row in matrix["selected_train_checkpoint_references"]}
    actual_refs = set()
    for ref in selection["selected_references"]:
        key = (ref["arm_id"], ref["seed"])
        require(key in planned_refs and key not in actual_refs, "Original selected-reference accounting differs")
        actual_refs.add(key)
        require(ref["fit_id"] in planned_refs[key]["candidate_fit_ids"]
                and ref["bundle_manifest_sha256"] == fit_manifests[ref["fit_id"]]
                and ref["checkpoint_epoch"] == 50 and ref["retrain"] is False,
                "Original selected checkpoint lineage differs")
        path = path_in_repo(repo, ref["selection_path"])
        require(path == path_in_repo(repo, planned_refs[key]["output_dir"]) / "selected.json",
                "Original selection path differs from its planned namespace")
        require(read_json(path) == ref, "Original selected reference differs from frozen selection")
        bind(path)
    for seed in parent["data"]["training_seeds"]:
        for role in ("train", "validation"):
            load_prepared_inputs(parent, seed, role, repo_root=repo)
        provenance = load_input_provenance(parent, seed, repo_root=repo)
        manifest_path = Path(provenance["manifest_path"])
        bind(manifest_path, provenance["manifest_sha256"])
        for name, digest in provenance["cache_file_sha256"].items():
            bind(manifest_path.parent / name, digest)
        bind(Path(provenance["encoder"]["path"]), provenance["encoder"]["sha256"])
    return {"source_study_id": parent["study_id"], "source_config_sha256": content_hash(parent),
            "source_protocol_payload_sha256": protocol["manifest_payload_sha256"],
            "original_candidate_count": 228, "original_selected_checkpoint_count": 76,
            "original_fit_manifest_sha256": fit_manifests, "file_sha256": files,
            "test_features_or_labels_read": False, "new_features_extracted": False}


def declare_protocol(config_path: str | Path = DEFAULT_CONFIG, repo: Path = REPO_ROOT) -> dict:
    from multimodal.attention_fusion_bypass_comparison import freeze_contrasts

    repo = Path(repo).resolve()
    config, parent, parent_matrix, parent_protocol = load_config(config_path, repo)
    matrix = build_matrix(config)
    root = path_in_repo(repo, config["paths"]["artifact_root"])
    protocol_dir = root / "protocol"
    manifest_path = protocol_dir / "protocol_manifest.json"
    if manifest_path.exists():
        return read_frozen_plan(config_path, repo)[2]
    require(not any((path_in_repo(repo, f["output_dir"]) / "run_spec.json").exists()
                    for f in matrix["development_fits"]), "Freeze the bypass protocol before scientific fitting")
    evidence = _authenticate_parent_evidence(parent, parent_matrix, parent_protocol, repo)
    sources = source_hashes(repo)
    payloads = {"resolved_protocol.json": config, "candidate_matrix.json": matrix, "parent_evidence.json": evidence}
    protocol_dir.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        path = protocol_dir / name
        if path.exists():
            require(read_json(path) == value, "Partial declaration differs; preserve it and investigate")
        else:
            publish_json(path, value)
    freeze_contrasts(config, matrix, protocol_dir / "contrasts.json", repo_root=repo)
    manifest = {"schema_version": "attention_fusion.bypass_protocol.v1", "study_id": STUDY_ID,
        "effective_config_sha256": content_hash(config), "initialization_seed_namespace": config["initialization_seed_namespace"],
        "parent_source_config_sha256": content_hash(parent), "execution_source_sha256": sources,
        "files": {name: file_hash(protocol_dir / name) for name in [*payloads, "contrasts.json"]},
        "planned_counts": matrix["counts"], "test_features_or_labels_read": False}
    manifest["manifest_payload_sha256"] = content_hash(manifest)
    publish_json(manifest_path, manifest)
    return manifest


def read_frozen_plan(config_path: str | Path = DEFAULT_CONFIG, repo: Path = REPO_ROOT) -> tuple[dict, dict, dict]:
    repo = Path(repo).resolve()
    config, parent, _, _ = load_config(config_path, repo)
    directory = path_in_repo(repo, config["paths"]["artifact_root"]) / "protocol"
    manifest = read_json(directory / "protocol_manifest.json")
    require(manifest.get("manifest_payload_sha256") == content_hash({k: v for k, v in manifest.items()
            if k != "manifest_payload_sha256"}), "Bypass protocol manifest checksum differs")
    require(manifest["study_id"] == config["study_id"] and manifest["effective_config_sha256"] == content_hash(config)
            and manifest["parent_source_config_sha256"] == content_hash(parent), "Frozen bypass configuration differs")
    require(manifest["execution_source_sha256"] == source_hashes(repo), "Bypass execution source changed after declaration")
    require(set(manifest["files"]) == {"resolved_protocol.json", "candidate_matrix.json", "parent_evidence.json", "contrasts.json"},
            "Unexpected protocol file allowlist")
    for name, digest in manifest["files"].items():
        require(file_hash(directory / name) == digest, f"Bypass protocol payload changed: {name}")
    require(read_json(directory / "resolved_protocol.json") == config, "Resolved bypass configuration differs")
    matrix = read_json(directory / "candidate_matrix.json")
    require(matrix == build_matrix(config), "Frozen bypass matrix differs")
    return config, matrix, manifest


def verify_parent_bindings(config: dict, repo: Path = REPO_ROOT) -> dict:
    """Recheck frozen reused bytes before training; never evaluate old scores."""
    repo = Path(repo).resolve()
    directory = path_in_repo(repo, config["paths"]["artifact_root"]) / "protocol"
    evidence = read_json(directory / "parent_evidence.json")
    for name, digest in evidence["file_sha256"].items():
        require(file_hash(path_in_repo(repo, name)) == digest, f"Reused parent artifact changed: {name}")
    return evidence


def load_fit_inputs(config: dict, seed: int, repo: Path = REPO_ROOT) -> tuple[dict, dict, dict]:
    from multimodal.attention_fusion_inputs import load_input_provenance, load_prepared_inputs

    parent, _, _ = read_parent_plan(PARENT_CONFIG, Path(repo))
    validate_config(config, parent)
    train = load_prepared_inputs(parent, seed, "train", repo_root=repo)
    validation = load_prepared_inputs(parent, seed, "validation", repo_root=repo)
    provenance = load_input_provenance(parent, seed, repo_root=repo)
    provenance.update(protocol_config_sha256=content_hash(config), source_protocol_config_sha256=content_hash(parent),
                      train_role="train", validation_role="validation")
    return train, validation, provenance

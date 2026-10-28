"""Versioned, development-only execution contract for M3J."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .attention_fusion_training import configuration_hash, sha256_file, _write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/multimodal_attention_fusion_joint.yaml"
FROZEN_ONLINE_CONFIG = REPO_ROOT / "configs/multimodal_attention_fusion_joint_frozen_online.yaml"
GEO_ONLY_CONFIG = REPO_ROOT / "configs/multimodal_attention_fusion_joint_geo_only.yaml"
# Historical configurations/artifacts remain available as provenance, not as active modes.
EXPLORATORY_CONFIG = REPO_ROOT / "configs/multimodal_attention_fusion_joint_exploratory.yaml"
STRICT_STUDY_ID = "attention_fusion_joint_v1"
EXPLORATORY_STUDY_ID = "attention_fusion_joint_exploratory_v1"
DIRECT_STUDY_ID = "attention_fusion_joint_direct_v1"
FROZEN_ONLINE_STUDY_ID = "attention_fusion_joint_frozen_online_direct_v1"
GEO_ONLY_STUDY_ID = "attention_fusion_joint_geo_only_direct_v1"
PACKAGE_REGIMES = ("joint_partial", "frozen_online", "geo_only")
LEGACY_LOADER_RNG_POLICY = "legacy_study_id_v1"
DECLARED_LOADER_RNG_POLICY = "declared_namespaces_v1"
VERIFICATION_POLICY = "direct_freeze_checks_v1"
PROVISIONAL_ASSURANCE = "provisional_unresolved_memory_integrity"
CANDIDATES = (
    "image_only", "raw_concat", "projected_concat", "tokenwise_capacity",
    "attention_two_token", "attention_cls", "attention_cls_skip_adapted",
    "attention_cls_skip_original",
)
SOURCE_FILES = (
    "multimodal/attention_fusion_joint_protocol.py",
    "multimodal/attention_fusion_joint_inputs.py",
    "multimodal/attention_fusion_joint_storage.py",
    "multimodal/attention_fusion_joint_training.py",
    "multimodal/attention_fusion_joint_reporting.py",
    "multimodal/attention_fusion.py", "multimodal/attention_fusion_bypass.py",
    "multimodal/attention_fusion_training.py", "multimodal/attention_fusion_inputs.py",
    "multimodal/geo_helpfulness_oof.py", "multimodal/models.py",
    "tools/run_attention_fusion_joint_study.py",
    "configs/multimodal_attention_fusion_joint.yaml",
    "configs/multimodal_attention_fusion_joint_frozen_online.yaml",
    "configs/multimodal_attention_fusion_joint_geo_only.yaml",
    "configs/multimodal_attention_fusion.yaml",
    "configs/multimodal_attention_fusion_cls_bypass.yaml",
    "configs/multimodal_geo_helpfulness.yaml",
) + tuple(
    f"configs/multimodal_attention_fusion_joint{suffix}_seed{seed}.yaml"
    for seed in (2, 3, 4) for suffix in ("", "_frozen_online", "_geo_only")
)
RECOVERY_AMENDMENT = "D039_checkpoint_recovery"
RECOVERY_SOURCE_FILES = {
    "multimodal/attention_fusion_joint_protocol.py",
    "multimodal/attention_fusion_joint_storage.py",
    "multimodal/attention_fusion_joint_training.py",
    "tools/run_attention_fusion_joint_study.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def package_identity(seed: int, regime: str) -> str:
    """Return the explicit study identity for one seed/regime package."""
    require(type(seed) is int and seed in (1, 2, 3, 4) and regime in PACKAGE_REGIMES,
            "M3J packages require seed 1–4 and a declared regime")
    suffix = {"joint_partial": "", "frozen_online": "_frozen_online",
              "geo_only": "_geo_only"}[regime]
    seed_suffix = "" if seed == 1 else f"_seed{seed}"
    return f"attention_fusion_joint{suffix}{seed_suffix}_direct_v1"


def package_config_path(seed: int, regime: str) -> Path:
    """Find the checked-in package template without creating or launching a run."""
    package_identity(seed, regime)
    suffix = {"joint_partial": "", "frozen_online": "_frozen_online",
              "geo_only": "_geo_only"}[regime]
    seed_suffix = "" if seed == 1 else f"_seed{seed}"
    return REPO_ROOT / f"configs/multimodal_attention_fusion_joint{suffix}{seed_suffix}.yaml"


def realized_loader_namespaces(config: dict, *, include_images: bool | None = None) -> dict:
    """Resolve actual loader streams, preserving the historical seed-1 behavior."""
    training = config["training"]
    policy = training.get("loader_rng_policy", LEGACY_LOADER_RNG_POLICY)
    require(policy in (LEGACY_LOADER_RNG_POLICY, DECLARED_LOADER_RNG_POLICY),
            "Unknown M3J loader RNG policy")
    if include_images is None:
        include_images = config["authorized_scope"]["regime"] != "geo_only"
    legacy_images = include_images and policy == LEGACY_LOADER_RNG_POLICY
    def namespace(stream: str) -> str:
        if policy == LEGACY_LOADER_RNG_POLICY:
            return training.get(f"{stream}_seed_namespace", config["study_id"])
        return training[f"{stream}_seed_namespace"]

    return dict(
        policy=policy,
        augmentation_seed_namespace=(
            config["study_id"] if legacy_images else namespace("augmentation")
        ) if include_images else None,
        shuffle_seed_namespace=(config["study_id"] if legacy_images
                                else namespace("shuffle")),
        dropout_seed_namespace=namespace("dropout"),
    )


def geo_reference_regimes(config: dict) -> tuple[str, ...]:
    """Validate the explicit geo reference scope without reading deferred packages."""
    scope = config["authorized_scope"]
    require(scope["regime"] == "geo_only", "Geo reference scope is reserved for geo-only packages")
    seed = scope["seeds"][0]
    reporting = config["reporting"]
    if "deferred_reference_regimes" in reporting:
        require(seed in (2, 3, 4)
                and reporting["deferred_reference_regimes"] == ["frozen_online"],
                "Only seed 2–4 geo-only packages may explicitly defer frozen-online")
        regimes = ("joint_partial",)
    else:
        regimes = ("joint_partial", "frozen_online")
    require(reporting.get("completed_reference_roots") == [
        f"multimodal_artifacts/attention_fusion/cs/gse_100m/{package_identity(seed, regime)}"
        for regime in regimes],
        "Geo-only must reference every active completed image package for the same seed")
    return regimes


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    packages = {package_identity(seed, regime): (seed, regime)
                for seed in (1, 2, 3, 4) for regime in PACKAGE_REGIMES}
    require(config["study_id"] in packages,
            "Archived M3J workflow is not active; use the direct-freeze configuration")
    seed, regime = packages[config["study_id"]]
    require(config.get("verification_policy") == VERIFICATION_POLICY,
            "M3J requires the approved direct-freeze verification policy")
    require("numerical_policy" not in config,
            "The active M3J workflow has no numerical-tolerance policy")
    require(config.get("result_assurance") == PROVISIONAL_ASSURANCE,
            "M3J results must retain the unresolved-memory assurance label")
    require(config["paths"]["artifact_root"] ==
            f"multimodal_artifacts/attention_fusion/cs/gse_100m/{config['study_id']}",
            "Direct-freeze artifacts must use their separate study root")
    require(config.get("authorized_scope") == dict(
        regime=regime, seeds=[seed],
        initializations=(["not_applicable"] if regime == "geo_only"
                         else ["pretrained", "cs_adapted"])),
        "M3J package authorization must match its study identity, seed and regime")
    if regime == "frozen_online":
        require(config["reporting"].get("paired_reference_root") ==
                f"multimodal_artifacts/attention_fusion/cs/gse_100m/{package_identity(seed, 'joint_partial')}",
                "Frozen-online comparisons must reference the completed partial-joint package")
    else:
        require("paired_reference_root" not in config["reporting"],
                "The original partial-joint package has no external paired reference")
    if regime == "geo_only":
        geo_reference_regimes(config)
    else:
        require("completed_reference_roots" not in config["reporting"],
                "Combined reference packages are reserved for geo-only completion")
        require("deferred_reference_regimes" not in config["reporting"],
                "Deferred reference packages are reserved for seed 2–4 geo-only completion")
    require(config["checkpoint"].get("automatic_resume") is False,
            "Automatic resume is disabled in the approved direct-freeze workflow")
    require(config["checkpoint"]["prediction_replay"] == "load_and_evaluate",
            "Prediction replay must not be a mandatory M3J execution gate")
    require(config.get("initialization_seed_namespace") == "attention_fusion_v1"
            and all(config["training"].get(f"{stream}_seed_namespace") == STRICT_STUDY_ID
                    for stream in ("augmentation", "shuffle", "dropout")),
            "M3J must preserve the paired initialization and input RNG namespaces")
    expected_rng_policy = LEGACY_LOADER_RNG_POLICY if seed == 1 else DECLARED_LOADER_RNG_POLICY
    require(config["training"].get("loader_rng_policy", LEGACY_LOADER_RNG_POLICY)
            == expected_rng_policy,
            "Seed 1 retains its historical loader streams; seeds 2–4 require declared RNG namespaces")
    require(tuple(config["candidates"]) == CANDIDATES, "M3J architecture matrix changed")
    require(config["training_seeds"] == [1, 2, 3, 4], "M3J requires all four declared seeds")
    require(config["initializations"] == ["pretrained", "cs_adapted"], "Initialization arms changed")
    require(config["regimes"] == ["joint_partial", "frozen_online"], "Encoder regimes changed")
    expected = {
        "epochs": 50, "batch_size": 16, "gradient_accumulation_steps": 1,
        "drop_last": False, "num_workers": 0, "loss": "unweighted_cross_entropy",
        "label_smoothing": 0.0, "optimizer": "adam", "learning_rate": 5e-5,
        "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0,
        "amsgrad": False, "gradient_clipping": None, "scheduler": "cosine_annealing_lr",
        "scheduler_t_max": 50, "scheduler_eta_min": 0.0, "scheduler_step": "end_of_epoch",
        "warmup_epochs": 0, "early_stopping": False, "parameter_dtype": "float32",
        "autocast_dtype": "float16", "grad_scaler": False, "validation_autocast": False,
        "cudnn_allow_tf32": True, "matmul_allow_tf32": False,
        "float32_matmul_precision": "highest", "deterministic_algorithms": True,
        "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8",
    }
    require(all(config["training"].get(k) == v for k, v in expected.items()),
            "Training recipe differs from the user-confirmed M3J recipe")
    require(config["geo"]["preprocessing"] == "standardize"
            and config["geo"]["fit_role"] == "train", "Geo must use train-fitted standardization")
    require(config["checkpoint"]["storage"] == "shared_objects_option3", "Storage policy changed")
    return config


def study_root(config: dict, repo: Path = REPO_ROOT) -> Path:
    relative = Path(config["paths"]["artifact_root"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact root")
    return repo / relative


def package_name(config: dict) -> str:
    """Name the separately authorized seed/regime execution package."""
    scope = config["authorized_scope"]
    require(len(scope["seeds"]) == 1 and scope["regime"] in ("joint_partial", "frozen_online", "geo_only"),
            "M3J execution requires a single-seed package")
    return f"seed{scope['seeds'][0]}_{scope['regime']}"


def check_reference_package(config: dict, input_provenance: dict,
                            repo: Path = REPO_ROOT) -> dict | None:
    """Bind frozen-online contrasts to the completed, scientifically matched package.

    This reads small provenance and completion records only. It does not replay
    training or inference, or reread the reference encoder/checkpoint weights.
    """
    if config["authorized_scope"]["regime"] != "frozen_online":
        return None
    seed = config["authorized_scope"]["seeds"][0]
    identity = package_identity(seed, "joint_partial")
    relative = Path(config["reporting"]["paired_reference_root"])
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe paired reference root")
    root = repo / relative
    protocol_path = root / "protocol/protocol.json"
    previous = json.loads(protocol_path.read_text())
    reference_config = previous["config"]
    require(reference_config["study_id"] == identity
            and reference_config["paths"]["artifact_root"] == str(relative)
            and reference_config["authorized_scope"] == dict(
                regime="joint_partial", seeds=[seed], initializations=["pretrained", "cs_adapted"])
            and previous["config_sha256"] == configuration_hash(reference_config),
            "Paired reference is not the declared same-seed partial-joint package")
    scientific_fields = (
        "schema_version", "verification_policy", "result_assurance",
        "initialization_seed_namespace", "base_config", "bypass_config", "source_config",
        "training_seeds", "initializations", "regimes", "candidates", "training", "geo",
        "visual", "checkpoint", "refit_policy",
    )
    require(all(config[field] == reference_config[field] for field in scientific_fields)
            and {key: value for key, value in config["reporting"].items()
                 if key != "paired_reference_root"} == reference_config["reporting"],
            "Frozen-online scientific recipe differs from the paired partial-joint package")
    reference_inputs = previous["input_provenance"]
    neutral_inputs = {key: value for key, value in input_provenance.items() if key != "study_id"}
    require(input_provenance.get("study_id") == config["study_id"]
            and reference_inputs.get("study_id") == identity
            and neutral_inputs == {key: value for key, value in reference_inputs.items()
                                   if key != "study_id"},
            "Frozen-online inputs differ from the paired partial-joint package")
    status_path = root / "orchestration" / f"seed{seed}_joint_partial/status.json"
    status = json.loads(status_path.read_text())
    require(status.get("study_id") == identity
            and status.get("authorized_fits") == status.get("complete") == 16
            and status.get("queue_status") == "complete" and status.get("unresolved") == []
            and status.get("active_fit") is None and status.get("error") is None,
            "Paired partial-joint package is incomplete or unresolved")
    audit_path = root / "reports" / f"seed{seed}_joint_partial/completion_audit.json"
    audit = json.loads(audit_path.read_text())
    require(audit.get("package") == f"seed{seed}/joint_partial"
            and audit.get("status") == "complete" and audit.get("completed_fits") == 16
            and audit.get("training_epochs") == 800
            and audit.get("checkpoint_policy_evaluations") == 32
            and audit.get("unresolved") == [] and audit.get("test_data_scored") is False,
            "Paired partial-joint completion audit is incomplete or unresolved")
    return dict(
        artifact_root=str(relative), regime="joint_partial", seed=seed,
        protocol_sha256=sha256_file(protocol_path),
        config_sha256=previous["config_sha256"],
        reference_input_provenance_sha256=configuration_hash(reference_inputs),
        current_input_provenance_sha256=configuration_hash(input_provenance),
        matched_input_identity_sha256=configuration_hash(neutral_inputs),
        completion_status_sha256=sha256_file(status_path),
        completion_audit_path=str(audit_path.relative_to(repo)),
        completion_audit_sha256=sha256_file(audit_path),
        completed_fits=16, completed_at=status["updated_at"],
        check_scope="Recorded config/input identity and completion metadata; no weight-file audit or replay",
    )


def source_hashes(repo: Path = REPO_ROOT) -> dict[str, str]:
    return {p: sha256_file(repo / p) for p in SOURCE_FILES}


def check_geo_reference_packages(config: dict, input_provenance: dict,
                                 repo: Path = REPO_ROOT) -> list[dict]:
    """Bind the shared geo baseline to completed image packages using metadata."""
    if config["authorized_scope"]["regime"] != "geo_only":
        return []
    seed = config["authorized_scope"]["seeds"][0]
    regimes = geo_reference_regimes(config)
    records = []
    image_only_fields = {"study_id", "image_fingerprint", "image_inventory_sha256"}
    neutral = {key: value for key, value in input_provenance.items() if key not in image_only_fields}
    for relative, regime in zip(config["reporting"]["completed_reference_roots"], regimes):
        identity = package_identity(seed, regime)
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe completed reference root")
        root = repo / path
        protocol_path = root / "protocol/protocol.json"
        previous = json.loads(protocol_path.read_text())
        old = previous["config"]
        require(old["study_id"] == identity and old["paths"]["artifact_root"] == relative
                and old["authorized_scope"] == dict(
                    regime=regime, seeds=[seed], initializations=["pretrained", "cs_adapted"])
                and previous["config_sha256"] == configuration_hash(old),
                "Invalid completed reference protocol")
        fields = ("training", "geo", "checkpoint", "base_config", "source_config",
                  "initialization_seed_namespace", "verification_policy", "result_assurance")
        require(all(old[key] == config[key] for key in fields),
                "Geo-only recipe differs from completed image packages")
        require(neutral == {key: value for key, value in previous["input_provenance"].items()
                            if key not in image_only_fields}, "Geo-only inputs differ from image packages")
        status_path = root / "orchestration" / f"seed{seed}_{regime}" / "status.json"
        status = json.loads(status_path.read_text())
        require(status.get("study_id") == identity and status.get("queue_status") == "complete"
                and status.get("complete") == status.get("authorized_fits") == 16
                and status.get("active_fit") is None and status.get("error") is None
                and status.get("unresolved") == [], "Image reference package is incomplete")
        record = dict(artifact_root=relative, regime=regime, seed=seed, completed_fits=16,
            completed_at=status["updated_at"], protocol_sha256=sha256_file(protocol_path),
            status_sha256=sha256_file(status_path), matched_inputs_sha256=configuration_hash(neutral),
            realized_loader_seed_namespace=realized_loader_namespaces(old)["shuffle_seed_namespace"],
            check_scope="Recorded recipe, input and completion metadata; no weight-file audit or replay")
        if seed != 1:
            record["realized_loader_rng"] = realized_loader_namespaces(old)
        records.append(record)
    return records


def _training_core(path: Path) -> dict[str, str]:
    """Identify the unchanged numerical path independently of finalization."""
    functions = {node.name: node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef)}
    names = ("configure_runtime", "_batch", "_epoch", "predict", "is_better",
             "frozen_visual_integrity", "_verify_frozen_visual", "verify_freezing",
             "_weight_refs", "load_weight_refs")
    result = {name: ast.dump(functions[name], include_attributes=False) for name in names}
    trainer = functions["train_fit"]
    training_try = next(node for node in trainer.body if isinstance(node, ast.Try))
    # The startup builds the model/optimizer/scheduler; the first try-body
    # statement is the complete epoch loop. Only finalization/error handling changes.
    require(isinstance(training_try.body[0], ast.For), "Cannot identify the training loop")
    result["train_fit_startup"] = ast.dump(ast.Module(
        body=trainer.body[:trainer.body.index(training_try)], type_ignores=[]))
    result["train_fit_epoch_loop"] = ast.dump(training_try.body[0], include_attributes=False)
    return {name: configuration_hash(value) for name, value in result.items()}


def read_recovery_amendment(config: dict, input_provenance: dict,
                            repo: Path = REPO_ROOT) -> dict | None:
    root = study_root(config, repo)
    directory = root / "protocol/amendments" / RECOVERY_AMENDMENT
    path = directory / "amendment.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    original = json.loads((root / "protocol/protocol.json").read_text())
    require(value["amendment"] == RECOVERY_AMENDMENT
            and value["config_sha256"] == configuration_hash(config)
            and value["input_provenance"] == input_provenance
            and value["original_protocol_sha256"] == sha256_file(root / "protocol/protocol.json")
            and value["previous_sources"] == original["sources"]
            and value["sources"] == source_hashes(repo),
            "Checkpoint recovery amendment belongs to different code/config/inputs")
    require(all((directory / "sources" / name).is_file()
                and sha256_file(directory / "sources" / name) == digest
                for name, digest in value["sources"].items()),
            "Checkpoint recovery source snapshot is missing or changed")
    require(sha256_file(root / value["cpu_validation_path"]) == value["cpu_validation_sha256"],
            "Checkpoint recovery validation evidence changed")
    return value


def declare_recovery_amendment(config: dict, input_provenance: dict,
                               checked: dict, repo: Path = REPO_ROOT) -> dict:
    """Add an engineering amendment without replacing the original experiment."""
    root = study_root(config, repo)
    existing = read_recovery_amendment(config, input_provenance, repo)
    if existing is not None:
        return existing
    original_path = root / "protocol/protocol.json"
    original = json.loads(original_path.read_text())
    sources = source_hashes(repo)
    require(original["config"] == config and original["input_provenance"] == input_provenance
            and checked["status"] == "passed"
            and checked["sources"] == original["sources"]
            and checked["config_sha256"] == configuration_hash(config)
            and checked["input_provenance"] == input_provenance,
            "D039 requires the unchanged recipe/inputs and original passed verification")
    changed = sorted(name for name in sources if sources[name] != original["sources"].get(name))
    require(bool(changed) and set(changed) <= RECOVERY_SOURCE_FILES,
            "D039 cannot authorize changes outside checkpoint recovery and orchestration")
    snapshot = root / "protocol/sources"
    require(all((snapshot / name).is_file() and sha256_file(snapshot / name) == digest
                for name, digest in original["sources"].items()),
            "Frozen source snapshot is missing or changed")
    training_path = "multimodal/attention_fusion_joint_training.py"
    original_core, current_core = _training_core(snapshot / training_path), _training_core(repo / training_path)
    require(original_core == current_core, "D039 cannot change the numerical training path")
    evidence_path = root / "verification/d039_cpu_tests.json"
    evidence = json.loads(evidence_path.read_text())
    require(evidence.get("status") == "passed" and evidence.get("sources") == sources
            and evidence.get("failed") == 0 and evidence.get("passed", 0) > 0,
            "D039 requires passing CPU recovery tests for the current sources")
    directory = root / "protocol/amendments" / RECOVERY_AMENDMENT
    directory.mkdir(parents=True, exist_ok=True)
    for name in sources:
        target = directory / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    value = dict(
        amendment=RECOVERY_AMENDMENT, recorded_at=datetime.now(timezone.utc).isoformat(),
        authorization="User approved bounded checksum retries, evaluation-only recovery and continuing independent fits.",
        config_sha256=configuration_hash(config), input_provenance=input_provenance,
        original_protocol_sha256=sha256_file(original_path), previous_sources=original["sources"],
        sources=sources, changed_files=changed, unchanged_training_core=current_core,
        prior_verification_sha256=configuration_hash(checked),
        cpu_validation_path=str(evidence_path.relative_to(root)),
        cpu_validation_sha256=sha256_file(evidence_path),
        checkpoint_policy=dict(max_hash_attempts=3, expected_digest_required=True,
                               completed_epoch_evaluation_only=True,
                               continuation_retrains_existing_fits=False,
                               isolate_persistent_checkpoint_failures=True),
        result_assurance=config["result_assurance"],
    )
    _write_json(directory / "amendment.json", value)
    return value


def build_matrix(config: dict) -> list[dict]:
    matrix = []
    for seed in config["training_seeds"]:
        for regime in config["regimes"]:
            for candidate in config["candidates"]:
                for initialization in config["initializations"]:
                    fit_id = f"seed{seed}/{regime}/{candidate}/{initialization}"
                    matrix.append(dict(run_id=fit_id, seed=seed, regime=regime,
                                       candidate_id=candidate, initialization=initialization))
        matrix.append(dict(run_id=f"seed{seed}/geo_only", seed=seed, regime="geo_only",
                           candidate_id="geo_only", initialization="not_applicable"))
    return matrix


def authorized_fits(config: dict) -> list[dict]:
    scope = config["authorized_scope"]
    return [fit for fit in build_matrix(config)
            if fit["seed"] in scope["seeds"] and fit["regime"] == scope["regime"]
            and fit["initialization"] in scope["initializations"]]


def contrasts(config: dict) -> list[dict]:
    """Declare comparisons for both checkpoint policies before validation scoring."""
    result = []
    for policy in ("final", "best"):
        for regime in config["regimes"]:
            for initialization in config["initializations"]:
                pairs = [("attention_two_token", "projected_concat", "primary")]
                for candidate in CANDIDATES:
                    for anchor in ("image_only", "raw_concat"):
                        if candidate != anchor:
                            pairs.append((candidate, anchor, "supporting"))
                pairs.extend(("attention_two_token", anchor, "supporting")
                             for anchor in ("tokenwise_capacity",))
                pairs.extend((candidate, "attention_cls", "supporting") for candidate in
                             ("attention_cls_skip_adapted", "attention_cls_skip_original"))
                for candidate, reference, priority in pairs:
                    result.append(dict(kind="architecture", policy=policy, regime=regime,
                                       initialization=initialization, candidate=candidate,
                                       reference=reference, priority=priority))
            for candidate in CANDIDATES:
                result.append(dict(kind="initialization", policy=policy, regime=regime,
                                   candidate=candidate, treatment="cs_adapted",
                                   reference="pretrained", priority="supporting"))
        for candidate in CANDIDATES:
            for initialization in config["initializations"]:
                result.append(dict(kind="encoder_regime", policy=policy, candidate=candidate,
                                   initialization=initialization, treatment="joint_partial",
                                   reference="frozen_online", priority="supporting"))
    return result


def declare(config: dict, input_provenance: dict, repo: Path = REPO_ROOT) -> dict:
    reference = check_reference_package(config, input_provenance, repo)
    completed_references = check_geo_reference_packages(config, input_provenance, repo)
    directory = study_root(config, repo) / "protocol"
    directory.mkdir(parents=True, exist_ok=True)
    sources = source_hashes(repo)
    payload = dict(schema_version="attention_fusion_joint.protocol.v1", config=config,
                   config_sha256=configuration_hash(config), full_matrix=build_matrix(config),
                   authorized_fits=authorized_fits(config), contrasts=contrasts(config),
                   sources=sources, input_provenance=input_provenance,
                   boundary=dict(test_data_loaded=False, validation_used_for_gradients=False),
                   checkpoint_reporting="separate final/best, same fixed recipe, no policy cherry-picking",
                   ranking="four-seed mean acc1, weighted_f1, mcc; stable candidate ID exact tie",
                   first_seed_conclusion="provisional descriptive only; no four-seed outcome category",
                   verification_policy=config["verification_policy"],
                   result_assurance=config["result_assurance"],
                   refit=config["refit_policy"])
    if reference is not None:
        payload["paired_reference_package"] = reference
    if config["training"].get("loader_rng_policy") == DECLARED_LOADER_RNG_POLICY:
        payload["realized_loader_rng"] = realized_loader_namespaces(config)
        payload["random_stream_pairing"] = (
            "Image augmentation and shuffle use the shared declared namespaces across the "
            "same-seed partial-joint and frozen-online packages; the seed remains an RNG input. "
            "Geo-only has no image augmentation and uses the same declared shuffle namespace.")
    if completed_references:
        payload["completed_reference_packages"] = completed_references
        deferred = config["reporting"].get("deferred_reference_regimes", [])
        if deferred:
            seed = config["authorized_scope"]["seeds"][0]
            payload["deferred_reference_packages"] = [dict(
                artifact_root=("multimodal_artifacts/attention_fusion/cs/gse_100m/"
                               f"{package_identity(seed, regime)}"),
                regime=regime, seed=seed, status="deferred", planned_fits=16,
            ) for regime in deferred]
        payload["realized_geo_shuffle_namespace"] = config["training"]["shuffle_seed_namespace"]
        if config["training"].get("loader_rng_policy", LEGACY_LOADER_RNG_POLICY) == LEGACY_LOADER_RNG_POLICY:
            payload["comparison_limitation"] = (
                "Completed image packages used distinct study IDs for crop/shuffle seeds despite "
                "declaring shared namespaces; encoder-regime differences do not isolate trainability alone.")
    path = directory / "protocol.json"
    snapshot = directory / "sources"
    if path.exists():
        original = json.loads(path.read_text())
        if original != payload:
            amended = dict(payload, sources=original["sources"])
            require(original == amended and read_recovery_amendment(config, input_provenance, repo) is not None,
                    "Declared M3J protocol/input/source changed")
            sources = original["sources"]
        require(all((snapshot / relative).is_file() and sha256_file(snapshot / relative) == digest
                    for relative, digest in sources.items()), "Frozen source snapshot is missing or changed")
    else:
        for relative in sources:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((repo / relative).read_bytes())
        _write_json(path, payload)
    return payload

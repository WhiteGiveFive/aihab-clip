"""Immutable M4 candidate fitting; no validation predictions or selection."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Match the parent workflow's native initialization order.
from multimodal import geo_helpfulness_router_dataset as dataset_api
from multimodal.geo_helpfulness_protocol import canonical_sha256, sha256_file
from multimodal.geo_helpfulness_router_candidate_fit import fit_router_candidate
from multimodal.geo_helpfulness_router_candidate_state import (
    SCHEMA_VERSION as CANDIDATE_SCHEMA_VERSION,
    full_feature_columns,
    predict_router_probabilities,
    validate_candidate_state,
    variant_feature_indices,
)

import numpy as np
import yaml
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = dataset_api.DEFAULT_CONFIG_PATH
DEFAULT_STUDY_CONFIG_PATH = (
    PROJECT_ROOT / "configs/multimodal_geo_router_candidates_v2.json"
)
SEEDS = (1, 2, 3, 4)
C_GRID = (0.01, 0.1, 1.0, 10.0)
C_TOKENS = ("0p01", "0p1", "1", "10")
VARIANTS = ("primary_full727", "ablation_no_pairs79", "ablation_numeric25")
TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
STUDY_FILENAME = "study_spec.json"
REPORT_FILENAME = "training_report.json"
MANIFEST_FILENAME = "manifest.json"
CODE_PATHS = (
    "multimodal/geo_helpfulness_router_candidates.py",
    "multimodal/geo_helpfulness_router_candidate_fit.py",
    "multimodal/geo_helpfulness_router_candidate_state.py",
    "tools/run_multimodal_geo_helpfulness_m4_candidates.py",
)
FROZEN_PARENT_PARAMETERS = {
    "penalty": "l2",
    "solver": "lbfgs",
    "multi_class": "multinomial",
    "fit_intercept": True,
    "class_weight": None,
    "dual": False,
    "warm_start": False,
    "n_jobs": 1,
    "max_iter": 1000,
    "tol": 1e-8,
    "random_state": None,
}
# The user-approved v2 amendment changes only the optimization budget.
MODEL_PARAMETERS = {**FROZEN_PARENT_PARAMETERS, "max_iter": 5000}
EXPECTED_STUDY_CONFIG = {
    "schema_version": "geo_helpfulness.router_candidate_study_config.v2",
    "study_id": "router_candidates_v2",
    "parent_protocol_id": "protocol_v1",
    "optimizer_budget_amendment": {
        "approved_by": "user",
        "approval_date": "2026-09-08",
        "previous_study_id": "router_candidates_v1",
        "previous_study_spec_sha256": "26f0d541bf51ef46539ff8b80bd315e7e4fcf67687c6c2cc449ca3ae572f7453",
        "parameter": "max_iter",
        "previous_value": 1000,
        "approved_value": 5000,
        "scope": "all_48_candidates",
        "all_other_settings_unchanged": True,
    },
    "primary_variant": VARIANTS[0],
    "variants": [
        {
            "id": VARIANTS[0],
            "role": "primary",
            "feature_count": 727,
            "semantic_feature_count": 30,
        },
        {
            "id": VARIANTS[1],
            "role": "secondary_ablation",
            "feature_count": 79,
            "semantic_feature_count": 28,
        },
        {
            "id": VARIANTS[2],
            "role": "secondary_ablation",
            "feature_count": 25,
            "semantic_feature_count": 25,
        },
    ],
    "training_seeds": list(SEEDS),
    "regularization_grid_C": list(C_GRID),
    "fit_role": "development_train_oof",
    "sample_weight": None,
    "reuse_frozen_preprocessing": True,
    "feature_selection": "fixed_column_views_only",
    "primary_schema_unchanged": True,
    "automatic_ablation_promotion": False,
    "later_selection": {
        "one_common_C_per_variant": True,
        "metric": "four_state_log_loss",
        "seed_aggregation": "arithmetic_mean",
        "C_tie_break": "smallest_C",
        "threshold_selection": "same_frozen_protocol_v1_utility_and_guardrails_per_variant",
        "validation_predictions_sealed_before_label_scoring": True,
        "test_data_allowed": False,
    },
    "execution": {
        "device": "cpu",
        "candidate_parallelism": 1,
        "native_thread_limit": 1,
        "convergence_failure": "hard_reject_no_automatic_relaxation",
        "coefficient_reproduction_atol": 1e-10,
        "coefficient_reproduction_rtol": 1e-10,
        "probability_reproduction_atol": 1e-12,
        "probability_reproduction_rtol": 1e-12,
    },
}


class RouterCandidateError(ValueError):
    """Invalid or stale candidate study, inputs, or saved state."""


@dataclass(frozen=True)
class SeedInputs:
    matrix: np.ndarray
    targets: np.ndarray
    row_identity_sha256: str
    row_count: int
    plot_count: int


@dataclass(frozen=True)
class PreparedCandidates:
    root: Path
    protocol_id: str
    seeds: dict[int, SeedInputs]
    study: dict[str, Any]
    parent_files: dict[str, str]


def candidate_filename(variant: str, seed: int, C: float) -> str:
    if variant not in VARIANTS or seed not in SEEDS or C not in C_GRID:
        raise RouterCandidateError("candidate outside predeclared grid")
    return f"{variant}__seed_{seed}__C_{C_TOKENS[C_GRID.index(C)]}.json"


def candidate_specs():
    return [
        (variant, seed, C) for variant in VARIANTS for seed in SEEDS for C in C_GRID
    ]


def _path(value: str | Path) -> Path:
    value = Path(value)
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash(payload: dict[str, Any], field: str) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _read_json(path: Path, *, canonical: bool = True) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RouterCandidateError(f"expected regular non-symlink artifact: {path}")
    raw = path.read_bytes()

    def reject(value):
        raise RouterCandidateError(f"nonfinite JSON value: {value}")

    payload = json.loads(raw, parse_constant=reject)
    if not isinstance(payload, dict) or (canonical and raw != _bytes(payload)):
        raise RouterCandidateError(f"noncanonical JSON artifact: {path}")
    return payload


def _publish_json(path: Path, payload: dict[str, Any]) -> None:
    """Single-file no-replace commit; committed candidates never need cleanup."""
    staging = Path(
        tempfile.mkdtemp(prefix=".router_candidates.staging-", dir=path.parent.parent)
    )
    try:
        temporary = staging / path.name
        with temporary.open("xb") as handle:
            handle.write(_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RouterCandidateError(
                f"immutable artifact already exists: {path}"
            ) from exc
    finally:
        shutil.rmtree(staging)


def _check_snapshot(prepared: PreparedCandidates) -> None:
    for raw, digest in prepared.parent_files.items():
        path = Path(raw)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise RouterCandidateError(f"candidate parent/config/code changed: {path}")


def _environment() -> dict[str, str]:
    return {
        "python": sys.version,
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
    }


def _inherited_parent_snapshot(dataset, sealed_dir: Path) -> dict[str, str]:
    """Retain the validated dataset's upstream file identities during fitting."""
    lineage = dataset.manifest["parent_lineage"]
    parents = lineage["m1_m2_parent_artifact_hashes"]
    base = dataset.root.parent
    aggregate_dir = base / "development_train_oof"
    hashes = {
        sealed_dir
        / "development_assignments.parquet": parents["development_assignments"][
            "file_sha256"
        ],
        sealed_dir
        / "resolved_protocol.yaml": parents["resolved_protocol"]["file_sha256"],
        sealed_dir
        / "protocol_manifest.json": parents["protocol_manifest"]["file_sha256"],
        aggregate_dir
        / "development_train_oof_model_outputs.parquet": parents[
            "development_train_oof_aggregate"
        ]["file_sha256"],
        aggregate_dir
        / "aggregate_manifest.json": parents["development_train_oof_aggregate"][
            "manifest_file_sha256"
        ],
    }
    for relative, digest in parents["train_oof_producer_manifests"][
        "file_sha256"
    ].items():
        hashes[base / relative / "manifest.json"] = digest
    m3_root = dataset.root / "targets_and_feature_contract"
    for name, digest in lineage["m3_bundle_file_sha256"].items():
        hashes[m3_root / name] = digest
    m3_manifest_path = m3_root / "manifest.json"
    if sha256_file(m3_manifest_path) != hashes[m3_manifest_path]:
        raise RouterCandidateError("M3 manifest changed after dataset validation")
    m3_manifest = _read_json(m3_manifest_path, canonical=False)
    for key in ("m2_code_file_sha256", "m3_code_file_sha256"):
        hashes.update(
            {PROJECT_ROOT / name: digest for name, digest in m3_manifest[key].items()}
        )
    hashes.update(
        {
            PROJECT_ROOT / name: digest
            for name, digest in lineage["m4_code_file_sha256"].items()
        }
    )
    return {str(path): digest for path, digest in hashes.items()}


def _prepare_candidates(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
    study_config_path: str | Path = DEFAULT_STUDY_CONFIG_PATH,
) -> PreparedCandidates:
    study_config_path = _path(study_config_path)
    study_config = _read_json(study_config_path, canonical=False)
    if study_config != EXPECTED_STUDY_CONFIG:
        raise RouterCandidateError(
            "study differs from the predeclared candidate experiment"
        )
    dataset = dataset_api.load_validated_router_dataset_bundle(
        config_path=config_path, protocol_dir=protocol_dir, artifact_root=artifact_root
    )
    config_path = _path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        sha256_file(config_path)
        != dataset.manifest["parent_lineage"]["config_file_sha256"]
    ):
        raise RouterCandidateError("config changed after dataset validation")
    implementation = dict(config["router"]["implementation"])
    if implementation.pop("library") != "sklearn":
        raise RouterCandidateError("frozen router library changed")
    implementation["tol"] = implementation.pop("tolerance")
    implementation["penalty"] = config["router"]["penalty"]
    if (
        implementation != FROZEN_PARENT_PARAMETERS
        or config["router"]["regularization_grid_C"] != list(C_GRID)
        or config["router"]["target_order"] != list(TARGET_ORDER)
    ):
        raise RouterCandidateError("frozen primary router settings differ from trainer")
    if dataset.manifest["protocol_id"] != study_config["parent_protocol_id"]:
        raise RouterCandidateError("study/parent protocol mismatch")
    columns = list(full_feature_columns())
    if (
        columns != dataset.manifest["feature_columns"]
        or dataset.dataset.column_names != list(dataset_api.m3.TARGET_COLUMNS) + columns
    ):
        raise RouterCandidateError("parent router feature column order changed")
    frame = dataset.dataset.to_pandas()
    if len(frame) != 13512 or set(frame.training_seed) != set(SEEDS):
        raise RouterCandidateError(
            "parent router dataset does not contain all four seeds"
        )
    seeds = {}
    for seed in SEEDS:
        rows = (
            frame.loc[frame.training_seed.eq(seed)]
            .sort_values("row_uid", kind="stable")
            .reset_index(drop=True)
        )
        if (
            len(rows) != 3378
            or rows.plot_idx.nunique() != 1300
            or rows.row_uid.duplicated().any()
        ):
            raise RouterCandidateError("seed-specific training coverage is invalid")
        if set(rows.target_state) != set(TARGET_ORDER):
            raise RouterCandidateError("seed is missing one or more four-state targets")
        matrix = np.ascontiguousarray(rows.loc[:, columns].to_numpy(dtype=np.float64))
        if not np.isfinite(matrix).all():
            raise RouterCandidateError("nonfinite parent features")
        targets = rows.target_state.map(
            {name: i for i, name in enumerate(TARGET_ORDER)}
        ).to_numpy(dtype=np.int64)
        row_hash = canonical_sha256(
            rows[["row_uid", "training_seed"]].to_dict(orient="records")
        )
        seeds[seed] = SeedInputs(
            matrix, targets, row_hash, len(rows), int(rows.plot_idx.nunique())
        )
    parent_files = {
        str(config_path): sha256_file(config_path),
        str(study_config_path): sha256_file(study_config_path),
    }
    sealed_dir = _path(
        protocol_dir if protocol_dir is not None else config["paths"]["protocol_root"]
    )
    parent_files.update(_inherited_parent_snapshot(dataset, sealed_dir))
    for name in dataset_api.BUNDLE_FILENAMES:
        path = dataset.root / name
        digest = sha256_file(path)
        if name == dataset_api.MANIFEST_FILENAME:
            if _read_json(path, canonical=False) != dataset.manifest:
                raise RouterCandidateError("dataset manifest changed after validation")
        elif digest != dataset.manifest["artifacts"][name]["file_sha256"]:
            raise RouterCandidateError(
                f"dataset child changed after validation: {name}"
            )
        parent_files[str(path)] = digest
    code_hashes = {name: sha256_file(PROJECT_ROOT / name) for name in CODE_PATHS}
    parent_files.update(
        {str(PROJECT_ROOT / name): digest for name, digest in code_hashes.items()}
    )
    study = {
        "schema_version": "geo_helpfulness.router_candidate_study.v2",
        "study_id": study_config["study_id"],
        "protocol_id": dataset.manifest["protocol_id"],
        "configuration": study_config,
        "source_study_config_sha256": sha256_file(study_config_path),
        "parent_config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": dataset.manifest["manifest_sha256"],
        "dataset_artifact_file_sha256": {
            name: parent_files[str(dataset.root / name)]
            for name in dataset_api.BUNDLE_FILENAMES
        },
        "feature_input_names": columns,
        "variant_feature_indices": {
            name: variant_feature_indices(name) for name in VARIANTS
        },
        "model_parameters": MODEL_PARAMETERS,
        "parent_model_parameters": FROZEN_PARENT_PARAMETERS,
        "candidate_state_schema": CANDIDATE_SCHEMA_VERSION,
        "target_order": list(TARGET_ORDER),
        "implementation_file_sha256": code_hashes,
        "environment": _environment(),
        "predeclared_before_candidate_fitting": True,
        "source_access": {
            "fit_inputs": "development_train_oof_only",
            "validation_expert_outputs_opened": False,
            "validation_scoring_performed": False,
            "test_sources_opened": False,
            "inherited_parent_preflight_reads_full_development_assignments": True,
        },
        "seed_inputs": {
            str(seed): {
                "row_count": values.row_count,
                "plot_count": values.plot_count,
                "row_identity_sha256": values.row_identity_sha256,
            }
            for seed, values in seeds.items()
        },
    }
    study["study_spec_sha256"] = _hash(study, "study_spec_sha256")
    prepared = PreparedCandidates(
        dataset.root / study_config["study_id"],
        dataset.manifest["protocol_id"],
        seeds,
        study,
        parent_files,
    )
    _check_snapshot(prepared)
    return prepared


def _allowed_files() -> set[str]:
    return {
        STUDY_FILENAME,
        REPORT_FILENAME,
        MANIFEST_FILENAME,
        *[candidate_filename(*spec) for spec in candidate_specs()],
    }


def _check_directory(root: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise RouterCandidateError("candidate root must be a non-symlink directory")
    if root.exists():
        unknown = [
            path.name
            for path in root.iterdir()
            if path.name not in _allowed_files()
            or path.is_symlink()
            or not path.is_file()
        ]
        if unknown:
            raise RouterCandidateError(
                f"candidate root contains unknown or non-file entries: {unknown}"
            )


@contextmanager
def _workflow_lock(root: Path):
    import fcntl

    _check_directory(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.parent / ".m4_router_candidates_v2.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RouterCandidateError(
                "another process owns the candidate workflow lock"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _check_study(prepared: PreparedCandidates) -> None:
    study = _read_json(prepared.root / STUDY_FILENAME)
    if study != prepared.study or study.get("study_spec_sha256") != _hash(
        study, "study_spec_sha256"
    ):
        raise RouterCandidateError(
            "candidate predeclaration, parents, environment, or code are stale"
        )


def _check_state(
    prepared: PreparedCandidates,
    state: dict[str, Any],
    variant: str,
    seed: int,
    C: float,
) -> None:
    validate_candidate_state(state)
    expected = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "protocol_id": prepared.protocol_id,
        "training_seed": seed,
        "variant_id": variant,
        "C": C,
        "C_hex": float(C).hex(),
        "fit_row_count": prepared.seeds[seed].row_count,
        "fit_row_identity_sha256": prepared.seeds[seed].row_identity_sha256,
        "study_spec_sha256": prepared.study["study_spec_sha256"],
        "dataset_manifest_sha256": prepared.study["dataset_manifest_sha256"],
    }
    if any(state.get(name) != value for name, value in expected.items()):
        raise RouterCandidateError("candidate identity or fit provenance mismatch")


def _fit(prepared: PreparedCandidates, variant: str, seed: int, C: float):
    inputs = prepared.seeds[seed]
    state, diagnostics = fit_router_candidate(
        inputs.matrix,
        inputs.targets,
        protocol_id=prepared.protocol_id,
        seed=seed,
        variant_id=variant,
        C=C,
        feature_columns=list(full_feature_columns()),
        fit_row_identity_sha256=inputs.row_identity_sha256,
        study_spec_hash=prepared.study["study_spec_sha256"],
        dataset_manifest_hash=prepared.study["dataset_manifest_sha256"],
    )
    _check_state(prepared, state, variant, seed, C)
    return state, diagnostics


def _candidate_metrics(
    prepared: PreparedCandidates, state: dict[str, Any]
) -> dict[str, Any]:
    from multimodal.geo_helpfulness_router_candidate_state import decision_function

    inputs = prepared.seeds[state["training_seed"]]
    with threadpool_limits(limits=1):
        probabilities = predict_router_probabilities(inputs.matrix, state)
        logits = decision_function(inputs.matrix, state)
    centered = logits - logits.max(axis=1, keepdims=True)
    losses = (
        np.log(np.exp(centered).sum(axis=1))
        - centered[np.arange(len(inputs.targets)), inputs.targets]
    )
    if not np.isfinite(losses).all():
        raise RouterCandidateError("nonfinite reconstructed training NLL")
    return {
        "variant_id": state["variant_id"],
        "training_seed": state["training_seed"],
        "C": state["C"],
        "C_hex": state["C_hex"],
        "feature_count": len(variant_feature_indices(state["variant_id"])),
        "row_count": inputs.row_count,
        "n_iter": state["n_iter"],
        "train_log_loss": float(losses.mean()),
        "train_accuracy": float(
            np.mean(probabilities.argmax(axis=1) == inputs.targets)
        ),
        "target_counts": {
            name: int(np.sum(inputs.targets == index))
            for index, name in enumerate(TARGET_ORDER)
        },
        "training_probability_sha256": canonical_sha256(probabilities.tolist()),
        "state_sha256": state["state_sha256"],
    }


def _report(prepared: PreparedCandidates) -> dict[str, Any]:
    metrics = []
    for variant, seed, C in candidate_specs():
        state = _read_json(prepared.root / candidate_filename(variant, seed, C))
        _check_state(prepared, state, variant, seed, C)
        metrics.append(_candidate_metrics(prepared, state))
    return {
        "schema_version": "geo_helpfulness.router_candidate_training_report.v1",
        "study_spec_sha256": prepared.study["study_spec_sha256"],
        "diagnostic_role": "in_sample_training_only_not_model_selection",
        "candidate_count": 48,
        "selected_C": None,
        "selected_threshold": None,
        "validation_predictions_generated": False,
        "validation_labels_scored": False,
        "candidates": metrics,
    }


def _manifest(prepared: PreparedCandidates) -> dict[str, Any]:
    candidates = {}
    for variant, seed, C in candidate_specs():
        name = candidate_filename(variant, seed, C)
        state = _read_json(prepared.root / name)
        _check_state(prepared, state, variant, seed, C)
        candidates[name] = {
            "file_sha256": sha256_file(prepared.root / name),
            "state_sha256": state["state_sha256"],
            "variant_id": variant,
            "training_seed": seed,
            "C_hex": float(C).hex(),
        }
    payload = {
        "schema_version": "geo_helpfulness.router_candidate_manifest.v1",
        "protocol_id": prepared.protocol_id,
        "study_id": prepared.study["study_id"],
        "study_spec_sha256": prepared.study["study_spec_sha256"],
        "dataset_manifest_sha256": prepared.study["dataset_manifest_sha256"],
        "candidate_count": 48,
        "primary_variant": VARIANTS[0],
        "variants": list(VARIANTS),
        "training_seeds": list(SEEDS),
        "C_grid": list(C_GRID),
        "candidates": candidates,
        "artifacts": {
            name: {"file_sha256": sha256_file(prepared.root / name)}
            for name in (STUDY_FILENAME, REPORT_FILENAME)
        },
        "selection_performed": False,
        "validation_predictions_generated": False,
        "publication": {
            "predeclaration_first": True,
            "candidate_files_individually_immutable": True,
            "complete_set_manifest_last": True,
            "committed_file_allowlist": sorted(_allowed_files()),
        },
    }
    payload["manifest_sha256"] = _hash(payload, "manifest_sha256")
    return payload


def _validate_prepared(prepared: PreparedCandidates) -> dict[str, Any]:
    _check_directory(prepared.root)
    _check_study(prepared)
    manifest = _read_json(prepared.root / MANIFEST_FILENAME)
    if manifest.get("manifest_sha256") != _hash(
        manifest, "manifest_sha256"
    ) or manifest != _manifest(prepared):
        raise RouterCandidateError("candidate manifest or child fingerprints differ")
    report = _read_json(prepared.root / REPORT_FILENAME)
    expected_report = _report(prepared)
    if report != expected_report:
        raise RouterCandidateError("candidate training diagnostics do not reproduce")
    _check_snapshot(prepared)
    return {
        "valid": True,
        "status": "reused_valid",
        "bundle_root": str(prepared.root),
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": 48,
        "variant_feature_counts": {
            name: len(variant_feature_indices(name)) for name in VARIANTS
        },
        "fits_this_call": 0,
        "selection_performed": False,
        "validation_predictions_generated": False,
    }


def build_router_candidate_bundle(**kwargs) -> dict[str, Any]:
    """Fit missing candidates; reuse every valid published candidate unchanged."""
    prepared = _prepare_candidates(**kwargs)
    with _workflow_lock(prepared.root):
        _check_snapshot(prepared)
        if (prepared.root / MANIFEST_FILENAME).exists():
            return _validate_prepared(prepared)
        prepared.root.mkdir(exist_ok=True)
        study_path = prepared.root / STUDY_FILENAME
        if not study_path.exists():
            if list(prepared.root.iterdir()):
                raise RouterCandidateError(
                    "candidate files exist without a sealed predeclaration"
                )
            _publish_json(study_path, prepared.study)
        _check_study(prepared)
        # Check every existing state before fitting anything new.
        existing = set()
        for variant, seed, C in candidate_specs():
            path = prepared.root / candidate_filename(variant, seed, C)
            if path.exists():
                _check_state(prepared, _read_json(path), variant, seed, C)
                existing.add(path.name)
        if (prepared.root / REPORT_FILENAME).exists() and len(existing) != 48:
            raise RouterCandidateError(
                "training report exists for an incomplete candidate set"
            )
        fitted = 0
        for variant, seed, C in candidate_specs():
            path = prepared.root / candidate_filename(variant, seed, C)
            if path.name in existing:
                continue
            state, diagnostics = _fit(prepared, variant, seed, C)
            _check_snapshot(prepared)
            _publish_json(path, state)
            fitted += 1
            print(
                json.dumps(
                    {
                        "event": "candidate_created",
                        "variant": variant,
                        "seed": seed,
                        "C": C,
                        "completed": len(existing) + fitted,
                        "total": 48,
                        "diagnostics": diagnostics,
                    },
                    allow_nan=False,
                ),
                file=sys.stderr,
                flush=True,
            )
        report = _report(prepared)
        report_path = prepared.root / REPORT_FILENAME
        if report_path.exists():
            if _read_json(report_path) != report:
                raise RouterCandidateError(
                    "existing report differs from reconstructed diagnostics"
                )
        else:
            _publish_json(report_path, report)
        _check_snapshot(prepared)
        _publish_json(prepared.root / MANIFEST_FILENAME, _manifest(prepared))
        result = _validate_prepared(prepared)
        result.update(
            status="created",
            fits_this_call=fitted,
            resumed_candidate_count=len(existing),
        )
        return result


def validate_router_candidate_bundle(**kwargs) -> dict[str, Any]:
    """Reconstruct training diagnostics and validate parents without fitting."""
    return _validate_prepared(_prepare_candidates(**kwargs))


def reproduce_router_candidate_bundle(**kwargs) -> dict[str, Any]:
    """Explicitly refit all 48 candidates and compare without overwriting files."""
    prepared = _prepare_candidates(**kwargs)
    result = _validate_prepared(prepared)
    original_files = {
        str(prepared.root / name): sha256_file(prepared.root / name)
        for name in _allowed_files()
    }
    largest_coefficient_error, largest_probability_error = 0.0, 0.0
    for variant, seed, C in candidate_specs():
        saved = _read_json(prepared.root / candidate_filename(variant, seed, C))
        repeated, _ = _fit(prepared, variant, seed, C)
        for name in ("coef", "intercept"):
            expected, actual = np.asarray(saved[name]), np.asarray(repeated[name])
            largest_coefficient_error = max(
                largest_coefficient_error, float(np.max(np.abs(expected - actual)))
            )
            if not np.allclose(expected, actual, atol=1e-10, rtol=1e-10):
                raise RouterCandidateError("refitted coefficients do not reproduce")
        with threadpool_limits(limits=1):
            expected_p = predict_router_probabilities(
                prepared.seeds[seed].matrix, saved
            )
            actual_p = predict_router_probabilities(
                prepared.seeds[seed].matrix, repeated
            )
        largest_probability_error = max(
            largest_probability_error, float(np.max(np.abs(expected_p - actual_p)))
        )
        if not np.allclose(expected_p, actual_p, atol=1e-12, rtol=1e-12):
            raise RouterCandidateError("refitted probabilities do not reproduce")
        print(
            json.dumps(
                {
                    "event": "candidate_reproduced",
                    "variant": variant,
                    "seed": seed,
                    "C": C,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    _check_snapshot(prepared)
    if any(
        sha256_file(Path(path)) != digest for path, digest in original_files.items()
    ):
        raise RouterCandidateError("candidate artifacts changed during reproduction")
    result.update(
        status="reproduced_valid",
        fits_this_call=48,
        max_coefficient_absolute_error=largest_coefficient_error,
        max_probability_absolute_error=largest_probability_error,
        artifacts_unchanged=True,
    )
    return result

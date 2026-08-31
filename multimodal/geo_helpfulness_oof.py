"""Honest M2 expert cross-fitting under the frozen geo-helpfulness protocol.

This module is an additive child implementation of M1.  It never edits or
reinterprets the sealed protocol; every public workflow first asks the frozen
M1 runner to validate its own artifacts and fingerprints.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Required by deterministic CUDA GEMM.  The CLI also sets this before importing
# this module; keeping it here protects direct programmatic use.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from multimodal.geo_helpfulness_protocol import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    assignment_fingerprint,
    canonical_json_bytes,
    canonical_sha256,
    canonicalize_file,
    canonicalize_plot_idx,
    sha256_file,
    validate_artifact_parent_roles,
    validate_fit_prediction_plot_provenance,
)
from multimodal.models import MLPHead


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "multimodal_geo_helpfulness.yaml"
M1_RUNNER_PATH = PROJECT_ROOT / "tools" / "run_multimodal_geo_helpfulness.py"

TRAINING_SEEDS = (1, 2, 3, 4)
OOF_FOLDS = (0, 1, 2, 3)
N_CLASSES = 18
GEO_COLUMNS = tuple(f"A{index:02d}" for index in range(64))
MODE_TO_PREFIX = {
    "image_only": "image",
    "geo_only": "geo",
    "raw_concat": "raw",
}
MODES = tuple(MODE_TO_PREFIX)
VECTOR_COLUMNS = tuple(
    f"{prefix}_{suffix}"
    for prefix in MODE_TO_PREFIX.values()
    for suffix in ("logits", "prob_native_t1")
)
PREDICTION_COLUMNS = tuple(f"{prefix}_pred" for prefix in MODE_TO_PREFIX.values())
OOF_OUTPUT_COLUMNS = (
    "schema_version",
    "protocol_id",
    "row_uid",
    "file",
    "file_lower",
    "plot_idx",
    "train_oof_fold",
    "training_seed",
    "image_logits",
    "geo_logits",
    "raw_logits",
    "image_pred",
    "geo_pred",
    "raw_pred",
    "image_prob_native_t1",
    "geo_prob_native_t1",
    "raw_prob_native_t1",
)
VALIDATION_OUTPUT_COLUMNS = tuple(
    column for column in OOF_OUTPUT_COLUMNS if column != "train_oof_fold"
)
PRODUCER_FILES = (
    "adapted_visual_tower.safetensors",
    "image_only_head.safetensors",
    "geo_only_head.safetensors",
    "raw_concat_head.safetensors",
    "geo_standardization.json",
    "resolved_stage_config.yaml",
    "training_metrics.json",
    "manifest.json",
)
PROBABILITY_ATOL = 1.0e-8
REPRODUCTION_ATOL = 1.0e-6
REPRODUCTION_RTOL = 1.0e-6


class M2Error(ValueError):
    """Base error for a fail-closed M2 contract violation."""


class M2ArtifactError(M2Error):
    """An immutable M2 artifact is absent, stale, corrupt, or mismatched."""


@dataclass(frozen=True)
class GeoStandardization:
    """Fold-local float32 population statistics for the ordered geo vector."""

    mean: np.ndarray
    std: np.ndarray

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "mean":
            return self.mean
        if key == "std":
            return self.std
        raise KeyError(key)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "geo_helpfulness.geo_standardization.v1",
            "feature_columns": list(GEO_COLUMNS),
            "dtype": "float32",
            "variance_ddof": 0,
            "zero_std_policy": "replace_with_one",
            "mean": self.mean.astype(np.float32, copy=False).tolist(),
            "std": self.std.astype(np.float32, copy=False).tolist(),
        }


@dataclass(frozen=True)
class ProducerSpec:
    stage_id: str
    artifact_role: str
    fold: int | None
    include_fold: bool
    output_filename: str
    relative_directory: Path


@dataclass(frozen=True)
class FrozenM2Context:
    protocol_dir: Path
    output_root: Path
    resolved_path: Path
    assignments_path: Path
    manifest_path: Path
    config: dict[str, Any]
    assignments: pd.DataFrame
    protocol_manifest: dict[str, Any]
    preflight: dict[str, Any]
    parent_hashes: dict[str, Any]
    code_file_hashes: dict[str, str]
    code_hash: str


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def validate_training_seed(seed: Any) -> int:
    value = _strict_int(seed, name="training seed")
    if value not in TRAINING_SEEDS:
        raise M2Error(f"training seed must be one of {list(TRAINING_SEEDS)}")
    return value


def stable_softmax_float64(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise M2Error("logits must be a non-empty two-dimensional matrix")
    if not np.isfinite(values).all():
        raise M2Error("logits must be finite")
    shifted = values - np.max(values, axis=1, keepdims=True)
    numerator = np.exp(shifted)
    denominator = np.sum(numerator, axis=1, keepdims=True, dtype=np.float64)
    probabilities = numerator / denominator
    if not np.isfinite(probabilities).all():
        raise M2Error("stable softmax produced a non-finite probability")
    return probabilities.astype(np.float64, copy=False)


def fit_geo_standardization(values: Any) -> GeoStandardization:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise M2Error("fitting geo values must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise M2Error("fitting geo values must be finite")
    mean = np.mean(matrix, axis=0, dtype=np.float32).astype(np.float32, copy=False)
    std = np.std(matrix, axis=0, dtype=np.float32, ddof=0).astype(np.float32, copy=False)
    std = std.copy()
    std[std == np.float32(0.0)] = np.float32(1.0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise M2Error("invalid fold-local geo standardization statistics")
    return GeoStandardization(mean=mean, std=std)


def _coerce_scaler(scaler: Any, width: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise M2Error("geo scaler must contain numeric mean and std vectors") from exc
    if mean.shape != (width,) or std.shape != (width,):
        raise M2Error(f"geo scaler width mismatch: expected {width}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise M2Error("geo scaler must be finite with strictly positive standard deviations")
    return mean, std


def apply_geo_standardization(values: Any, scaler: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise M2Error("geo values must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise M2Error("geo values must be finite")
    mean, std = _coerce_scaler(scaler, matrix.shape[1])
    transformed = ((matrix - mean) / std).astype(np.float32, copy=False)
    if not np.isfinite(transformed).all():
        raise M2Error("standardized geo values must be finite")
    return transformed


def producer_partitions(
    assignments: pd.DataFrame,
    *,
    stage: str,
    fold: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fitting and prediction assignments for one sealed M2 producer."""

    required = {
        "row_uid",
        "plot_idx",
        "development_role",
        "train_oof_fold",
    }
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise M2Error(f"assignments are missing partition columns: {missing}")
    if stage == "train_oof":
        if fold is None:
            raise M2Error("train_oof producer requires a fold")
        fold_value = _strict_int(fold, name="OOF fold")
        if fold_value not in OOF_FOLDS:
            raise M2Error(f"OOF fold must be one of {list(OOF_FOLDS)}")
        train = assignments["development_role"].astype(str).eq("train")
        observed = assignments["train_oof_fold"].astype("Int8")
        heldout = observed.eq(fold_value).fillna(False)
        fit = assignments.loc[train & ~heldout].copy()
        prediction = assignments.loc[train & heldout].copy()
    elif stage == "development_validation":
        if fold is not None:
            raise M2Error("development_validation producer must not specify a fold")
        train = assignments["development_role"].astype(str).eq("train")
        validation = assignments["development_role"].astype(str).eq("validation")
        fit = assignments.loc[train].copy()
        prediction = assignments.loc[validation].copy()
    else:
        raise M2Error(f"unknown M2 producer stage: {stage!r}")
    if fit.empty or prediction.empty:
        raise M2Error("producer fitting and prediction partitions must be non-empty")
    validate_fit_prediction_plot_provenance(fit["plot_idx"], prediction["plot_idx"])
    fit = fit.sort_values("row_uid", kind="mergesort").reset_index(drop=True)
    prediction = prediction.sort_values("row_uid", kind="mergesort").reset_index(drop=True)
    return fit, prediction


def _arrow_output_schema(include_fold: bool) -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("protocol_id", pa.string(), nullable=False),
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("file", pa.string(), nullable=False),
        pa.field("file_lower", pa.string(), nullable=False),
        pa.field("plot_idx", pa.string(), nullable=False),
    ]
    if include_fold:
        fields.append(pa.field("train_oof_fold", pa.int8(), nullable=False))
    fields.extend(
        [
            pa.field("training_seed", pa.int8(), nullable=False),
            pa.field("image_logits", pa.list_(pa.float64(), N_CLASSES), nullable=False),
            pa.field("geo_logits", pa.list_(pa.float64(), N_CLASSES), nullable=False),
            pa.field("raw_logits", pa.list_(pa.float64(), N_CLASSES), nullable=False),
            pa.field("image_pred", pa.int8(), nullable=False),
            pa.field("geo_pred", pa.int8(), nullable=False),
            pa.field("raw_pred", pa.int8(), nullable=False),
            pa.field(
                "image_prob_native_t1",
                pa.list_(pa.float64(), N_CLASSES),
                nullable=False,
            ),
            pa.field(
                "geo_prob_native_t1",
                pa.list_(pa.float64(), N_CLASSES),
                nullable=False,
            ),
            pa.field(
                "raw_prob_native_t1",
                pa.list_(pa.float64(), N_CLASSES),
                nullable=False,
            ),
        ]
    )
    return pa.schema(fields)


def _mode_logits(logits_by_mode: Mapping[str, Any], mode: str, rows: int) -> np.ndarray:
    aliases = (mode, MODE_TO_PREFIX[mode])
    present = [alias for alias in aliases if alias in logits_by_mode]
    if len(present) != 1:
        raise M2Error(f"logits_by_mode must contain exactly one key from {aliases}")
    values = np.asarray(logits_by_mode[present[0]], dtype=np.float64)
    if values.shape != (rows, N_CLASSES):
        raise M2Error(
            f"{mode} logits shape mismatch: {values.shape} != ({rows}, {N_CLASSES})"
        )
    if not np.isfinite(values).all():
        raise M2Error(f"{mode} logits must be finite")
    return values


def build_output_table(
    assignments: pd.DataFrame,
    *,
    seed: Any,
    logits_by_mode: Mapping[str, Any],
    include_fold: bool,
    schema_version: str | None = None,
    protocol_id: str | None = None,
) -> pa.Table:
    """Build the exact, label-blind M2 Arrow table from prediction logits."""

    training_seed = validate_training_seed(seed)
    required = {"row_uid", "file", "file_lower", "plot_idx"}
    if include_fold:
        required.add("train_oof_fold")
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise M2Error(f"prediction assignments are missing identity columns: {missing}")
    if assignments.empty:
        raise M2Error("prediction assignments must not be empty")
    if assignments["row_uid"].duplicated().any():
        raise M2Error("prediction assignments contain duplicate row_uid values")
    order = np.argsort(assignments["row_uid"].astype(str).to_numpy(), kind="stable")
    ordered = assignments.iloc[order].reset_index(drop=True)
    if schema_version is None:
        schema_version = "geo_helpfulness_protocol_config_v1"
    if protocol_id is None:
        values = set(ordered.get("protocol_id", pd.Series(["protocol_v1"])).astype(str))
        if len(values) != 1:
            raise M2Error("prediction assignments contain mixed protocol IDs")
        protocol_id = next(iter(values))

    data: dict[str, Any] = {
        "schema_version": [str(schema_version)] * len(ordered),
        "protocol_id": [str(protocol_id)] * len(ordered),
        "row_uid": ordered["row_uid"].astype(str).tolist(),
        "file": ordered["file"].astype(str).tolist(),
        "file_lower": ordered["file_lower"].astype(str).tolist(),
        "plot_idx": ordered["plot_idx"].astype(str).tolist(),
    }
    if include_fold:
        folds = ordered["train_oof_fold"].to_numpy(dtype=np.int8, na_value=-1)
        if not np.isin(folds, OOF_FOLDS).all():
            raise M2Error("OOF output contains an invalid or null train_oof_fold")
        data["train_oof_fold"] = folds
    data["training_seed"] = np.full(len(ordered), training_seed, dtype=np.int8)

    for mode, prefix in MODE_TO_PREFIX.items():
        original = _mode_logits(logits_by_mode, mode, len(assignments))
        logits = original[order].astype(np.float64, copy=False)
        data[f"{prefix}_logits"] = logits.tolist()
        data[f"{prefix}_pred"] = np.argmax(logits, axis=1).astype(np.int8)
        data[f"{prefix}_prob_native_t1"] = stable_softmax_float64(logits).tolist()

    columns = OOF_OUTPUT_COLUMNS if include_fold else VALIDATION_OUTPUT_COLUMNS
    schema = _arrow_output_schema(include_fold)
    arrays = [pa.array(data[column], type=schema.field(column).type) for column in columns]
    table = pa.Table.from_arrays(arrays, schema=schema)
    validate_output_table(table, include_fold=include_fold, expected_rows=len(ordered))
    return table


def _matrix_from_arrow(table: pa.Table, column: str) -> np.ndarray:
    try:
        matrix = np.asarray(table[column].combine_chunks().to_pylist(), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise M2ArtifactError(f"{column} cannot be decoded as float64 vectors") from exc
    if matrix.shape != (len(table), N_CLASSES):
        raise M2ArtifactError(
            f"{column} physical width mismatch: {matrix.shape} != ({len(table)}, {N_CLASSES})"
        )
    return matrix


def validate_output_table(
    table: pa.Table,
    *,
    include_fold: bool,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    if not isinstance(table, pa.Table):
        raise TypeError("output table must be a pyarrow.Table")
    expected_columns = OOF_OUTPUT_COLUMNS if include_fold else VALIDATION_OUTPUT_COLUMNS
    if tuple(table.column_names) != tuple(expected_columns):
        raise M2ArtifactError(
            "output column allow-list/order mismatch: "
            f"{table.column_names} != {list(expected_columns)}"
        )
    expected_schema = _arrow_output_schema(include_fold)
    if not table.schema.equals(expected_schema, check_metadata=False):
        raise M2ArtifactError(f"output Arrow schema mismatch: {table.schema}")
    if expected_rows is not None and len(table) != int(expected_rows):
        raise M2ArtifactError(f"output row count mismatch: {len(table)} != {expected_rows}")
    if len(table) < 1:
        raise M2ArtifactError("output table must not be empty")
    if any(column.null_count for column in table.columns):
        raise M2ArtifactError("output table must not contain null values")
    for column in ("schema_version", "protocol_id"):
        values = set(table[column].to_pylist())
        if len(values) != 1 or not next(iter(values)):
            raise M2ArtifactError(f"output table contains mixed or empty {column} values")
    row_uids = table["row_uid"].to_pylist()
    seeds = np.asarray(table["training_seed"].to_pylist(), dtype=np.int64)
    if list(zip(row_uids, seeds.tolist())) != sorted(zip(row_uids, seeds.tolist())):
        raise M2ArtifactError("output rows are not canonically sorted by (row_uid, training_seed)")
    if len(set(zip(row_uids, seeds.tolist()))) != len(table):
        raise M2ArtifactError("duplicate output key (row_uid, training_seed)")
    if not np.isin(seeds, TRAINING_SEEDS).all():
        raise M2ArtifactError("output table contains an invalid training seed")
    if include_fold:
        folds = np.asarray(table["train_oof_fold"].to_pylist(), dtype=np.int64)
        if not np.isin(folds, OOF_FOLDS).all():
            raise M2ArtifactError("output table contains an invalid OOF fold")
    for mode, prefix in MODE_TO_PREFIX.items():
        logits = _matrix_from_arrow(table, f"{prefix}_logits")
        probabilities = _matrix_from_arrow(table, f"{prefix}_prob_native_t1")
        predictions = np.asarray(table[f"{prefix}_pred"].to_pylist(), dtype=np.int64)
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise M2ArtifactError(f"{mode} output contains non-finite values")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise M2ArtifactError(f"{mode} probability is outside [0, 1]")
        if not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=PROBABILITY_ATOL, rtol=0.0
        ):
            raise M2ArtifactError(f"{mode} probability rows do not sum to one")
        recomputed = stable_softmax_float64(logits)
        if not np.allclose(
            probabilities, recomputed, atol=PROBABILITY_ATOL, rtol=0.0
        ):
            raise M2ArtifactError(f"{mode} probabilities do not match native-T=1 softmax")
        logit_argmax = np.argmax(logits, axis=1)
        probability_argmax = np.argmax(probabilities, axis=1)
        if not np.array_equal(predictions, logit_argmax) or not np.array_equal(
            predictions, probability_argmax
        ):
            raise M2ArtifactError(f"{mode} prediction/logit/probability argmax mismatch")
    return {
        "valid": True,
        "row_count": len(table),
        "plot_count": len(set(table["plot_idx"].to_pylist())),
        "logical_table_sha256": logical_table_sha256(table),
    }


def logical_table_sha256(table: pa.Table) -> str:
    return canonical_sha256(
        {
            "columns": table.column_names,
            "rows": [list(row.values()) for row in table.to_pylist()],
        }
    )


def _exclusive_atomic_replace(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(temporary, 0o444)
    try:
        # A same-filesystem hard link is an atomic no-replace publication.  It
        # never exposes the empty placeholder window produced by O_EXCL+rename.
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable artifact already exists: {destination}") from exc
    temporary.unlink()


def write_output_parquet_atomic(table: pa.Table, path: str | Path) -> Path:
    include_fold = "train_oof_fold" in table.column_names
    validate_output_table(table, include_fold=include_fold)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(raw_temporary)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        )
        _exclusive_atomic_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_output_parquet(path: str | Path, *, include_fold: bool) -> pa.Table:
    source = Path(path)
    if not source.is_file():
        raise M2ArtifactError(f"model output does not exist: {source}")
    try:
        table = pq.read_table(source)
    except Exception as exc:
        raise M2ArtifactError(f"cannot read model output {source}: {exc}") from exc
    validate_output_table(table, include_fold=include_fold)
    return table


def _resolve_project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _read_yaml_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise M2ArtifactError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise M2ArtifactError(f"{name} must be a YAML mapping: {path}")
    return value


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M2ArtifactError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise M2ArtifactError(f"{name} must be a JSON object: {path}")
    return value


def _protocol_dir_hint(config_path: Path, protocol_dir: str | Path | None) -> Path:
    if protocol_dir is not None:
        return _resolve_project_path(protocol_dir)
    hint = _read_yaml_mapping(config_path, name="protocol configuration hint")
    try:
        return _resolve_project_path(hint["paths"]["protocol_root"])
    except (KeyError, TypeError) as exc:
        raise M2ArtifactError("protocol configuration has no paths.protocol_root") from exc


def validate_m1_preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run the frozen M1 validator through the active Python interpreter."""

    config = _resolve_project_path(config_path)
    sealed_dir = _protocol_dir_hint(config, protocol_dir)
    command = [
        sys.executable,
        str(M1_RUNNER_PATH),
        "validate-protocol",
        "--config",
        str(config),
        "--protocol-dir",
        str(sealed_dir),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown failure"
        raise M2ArtifactError(f"frozen M1 validate-protocol preflight failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise M2ArtifactError("M1 preflight did not return valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") != "valid":
        raise M2ArtifactError(f"M1 preflight did not report valid status: {payload!r}")
    returned = Path(str(payload.get("protocol_dir", sealed_dir))).resolve()
    if returned != sealed_dir.resolve():
        raise M2ArtifactError("M1 preflight validated an unexpected protocol directory")
    return sealed_dir, payload


def _implementation_hashes() -> tuple[dict[str, str], str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "tools" / "run_multimodal_geo_helpfulness_m2.py",
        PROJECT_ROOT / "multimodal" / "geo_helpfulness_oof_report.py",
        PROJECT_ROOT / "multimodal" / "models.py",
        PROJECT_ROOT / "multimodal" / "geo_helpfulness_protocol.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise M2ArtifactError(f"M2 implementation file is missing: {missing}")
    hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in paths
    }
    return hashes, canonical_sha256(hashes)


def load_frozen_context(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> FrozenM2Context:
    """Preflight M1, then load only its sealed resolved config and assignments."""

    sealed_dir, preflight = validate_m1_preflight(
        config_path=config_path, protocol_dir=protocol_dir
    )
    resolved_path = sealed_dir / "resolved_protocol.yaml"
    assignments_path = sealed_dir / "development_assignments.parquet"
    manifest_path = sealed_dir / "protocol_manifest.json"
    config = _read_yaml_mapping(resolved_path, name="sealed resolved protocol")
    manifest = _read_json_mapping(manifest_path, name="sealed protocol manifest")
    try:
        assignments = pd.read_parquet(assignments_path)
    except Exception as exc:
        raise M2ArtifactError(f"cannot read sealed assignments {assignments_path}: {exc}") from exc
    protocol_id = str(config.get("protocol_id"))
    if protocol_id != str(preflight.get("protocol_id")):
        raise M2ArtifactError("preflight and resolved protocol IDs disagree")
    if assignment_fingerprint(assignments) != preflight.get("assignment_content_sha256"):
        raise M2ArtifactError("sealed assignment logical hash changed after preflight")
    expected_seeds = tuple(int(value) for value in config["experts"]["training_seeds"])
    if expected_seeds != TRAINING_SEEDS:
        raise M2ArtifactError(f"resolved training seeds changed: {expected_seeds}")
    if int(config["experts"]["output_classes"]) != N_CLASSES:
        raise M2ArtifactError("resolved expert output size is not 18")
    if int(config["experts"]["geo_input"]["feature_dim"]) != len(GEO_COLUMNS):
        raise M2ArtifactError("resolved geo feature dimension is not 64")
    if str(config["experts"]["geo_input"]["feature_prefix"]) != "A":
        raise M2ArtifactError("resolved geo feature prefix is not A")
    root = _resolve_project_path(output_root) if output_root is not None else sealed_dir.parent
    code_files, code_hash = _implementation_hashes()
    parent_hashes = {
        "protocol_manifest": {
            "artifact_role": "frozen_experimental_protocol",
            "file_sha256": sha256_file(manifest_path),
            "payload_sha256": manifest.get("manifest_payload_sha256"),
        },
        "development_assignments": {
            "artifact_role": "development_assignments",
            "file_sha256": sha256_file(assignments_path),
            "content_sha256": assignment_fingerprint(assignments),
        },
        "resolved_protocol": {
            "artifact_role": "frozen_experimental_protocol",
            "file_sha256": sha256_file(resolved_path),
        },
    }
    return FrozenM2Context(
        protocol_dir=sealed_dir,
        output_root=root,
        resolved_path=resolved_path,
        assignments_path=assignments_path,
        manifest_path=manifest_path,
        config=config,
        assignments=assignments,
        protocol_manifest=manifest,
        preflight=preflight,
        parent_hashes=parent_hashes,
        code_file_hashes=code_files,
        code_hash=code_hash,
    )


def _producer_specs(seed: int) -> tuple[ProducerSpec, ...]:
    specs = [
        ProducerSpec(
            stage_id=f"train_oof_fold_{fold}",
            artifact_role="train_oof_fold_outputs",
            fold=fold,
            include_fold=True,
            output_filename="heldout_model_outputs.parquet",
            relative_directory=Path("development_train_oof") / f"seed_{seed}" / f"fold_{fold}",
        )
        for fold in OOF_FOLDS
    ]
    specs.append(
        ProducerSpec(
            stage_id="development_train_to_validation",
            artifact_role="development_validation_outputs",
            fold=None,
            include_fold=False,
            output_filename="development_validation_model_outputs.parquet",
            relative_directory=Path("development_validation") / f"seed_{seed}",
        )
    )
    return tuple(specs)


def _partitions_for_spec(
    assignments: pd.DataFrame, spec: ProducerSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.include_fold:
        return producer_partitions(assignments, stage="train_oof", fold=spec.fold)
    return producer_partitions(assignments, stage="development_validation")


def _geo_source_projection(context: FrozenM2Context, rows: pd.DataFrame) -> np.ndarray:
    """Read exactly ``file,A00,...,A63`` and return a one-to-one ordered join."""

    desired_files = rows["file"].astype(str).tolist()
    desired_lower = rows["file_lower"].astype(str).tolist()
    if len(set(desired_lower)) != len(desired_lower):
        raise M2Error("requested geo identities are not unique by file_lower")
    projected: list[pd.DataFrame] = []
    for raw_path in context.config["paths"]["development_source_tables"]:
        source = _resolve_project_path(raw_path)
        try:
            frame = pd.read_parquet(
                source,
                columns=["file", *GEO_COLUMNS],
                filters=[("file", "in", desired_files)],
            )
        except Exception as exc:
            raise M2ArtifactError(
                f"cannot read exact ordered A00…A63 projection from {source}: {exc}"
            ) from exc
        if not frame.empty:
            projected.append(frame)
    if not projected:
        raise M2ArtifactError("no requested geo rows were found in sealed source tables")
    combined = pd.concat(projected, ignore_index=True)
    try:
        combined.insert(
            1,
            "file_lower",
            [canonicalize_file(value).casefold() for value in combined["file"]],
        )
    except Exception as exc:
        raise M2ArtifactError("source geo table contains a non-canonical file identity") from exc
    if combined["file_lower"].duplicated().any():
        duplicates = combined.loc[combined["file_lower"].duplicated(False), "file_lower"]
        raise M2ArtifactError(f"geo source identity is not one-to-one: {duplicates.iloc[0]}")
    indexed = combined.set_index("file_lower", verify_integrity=True)
    missing = [identity for identity in desired_lower if identity not in indexed.index]
    extra = sorted(set(indexed.index).difference(desired_lower))
    if missing or extra:
        raise M2ArtifactError(
            f"geo identity join mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    values = indexed.loc[desired_lower, list(GEO_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    if values.shape != (len(rows), len(GEO_COLUMNS)):
        raise M2ArtifactError("ordered geo matrix shape mismatch")
    if not np.isfinite(values).all():
        raise M2ArtifactError("ordered A00…A63 values must be finite")
    return values


def _resolved_image_path(row: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    allowed = {
        _resolve_project_path(value)
        for value in config["development_universe"]["allowed_image_sources"]
    }
    source = _resolve_project_path(str(row["image_source"]))
    if source not in allowed:
        raise M2ArtifactError(f"assignment image source is not allow-listed: {source}")
    relative = Path(canonicalize_file(row["file"]))
    path = (source / relative).resolve()
    try:
        path.relative_to(source)
    except ValueError as exc:
        raise M2ArtifactError(f"image path escapes its declared source: {path}") from exc
    if not path.is_file():
        raise M2ArtifactError(f"assigned image does not exist: {path}")
    return path


class _LegacyImageDataset(Dataset):
    """Assignment-driven legacy BGR/439 decoder with one supplied transform."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        config: Mapping[str, Any],
        transform: Any,
        include_labels: bool,
    ) -> None:
        required = {"row_uid", "file", "image_source"}
        if include_labels:
            required.add("label_id_dense")
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise M2Error(f"image dataset rows are missing columns: {missing}")
        self._paths = [
            _resolved_image_path(row, config)
            for row in rows.to_dict(orient="records")
        ]
        self._row_uids = rows["row_uid"].astype(str).tolist()
        self._labels = (
            rows["label_id_dense"].to_numpy(dtype=np.int64, copy=True)
            if include_labels
            else None
        )
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int):
        # Imports stay local so CPU-only artifact validation does not require
        # OpenCV or Pillow to be imported.
        import cv2
        from PIL import Image

        image = cv2.imread(str(self._paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise M2ArtifactError(f"OpenCV could not decode image: {self._paths[index]}")
        image = cv2.resize(image, (439, 439), interpolation=cv2.INTER_LINEAR)
        tensor = self._transform(Image.fromarray(image))
        if self._labels is None:
            return tensor, index
        return tensor, int(self._labels[index])


def reset_reproducibility(seed: int) -> torch.Generator:
    training_seed = validate_training_seed(seed)
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(training_seed)
    return generator


def _pinned_snapshot(
    config: Mapping[str, Any], *, verify_hashes: bool = True
) -> tuple[Path, dict[str, str]]:
    frozen = config["experts"]["image_encoder"]["externally_pretrained_fixed"]
    checkpoint_id = str(frozen["checkpoint_id"])
    if not checkpoint_id.startswith("hf-hub:"):
        raise M2ArtifactError("pinned encoder checkpoint_id must use hf-hub:")
    repo_id = checkpoint_id.removeprefix("hf-hub:")
    revision = str(frozen["hub_revision"])
    try:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_files_only=True,
            )
        ).resolve()
    except Exception as exc:
        raise M2ArtifactError(
            f"exact pinned Hugging Face snapshot is not available locally: {repo_id}@{revision}"
        ) from exc
    expected = {
        str(frozen["checkpoint_filename"]): str(frozen["checkpoint_sha256"]),
        str(frozen["open_clip_config_filename"]): str(frozen["open_clip_config_sha256"]),
        str(frozen["tokenizer_filename"]): str(frozen["tokenizer_sha256"]),
        "tokenizer_config.json": str(frozen["tokenizer_config_sha256"]),
        "special_tokens_map.json": str(frozen["special_tokens_map_sha256"]),
    }
    for filename, expected_hash in expected.items():
        target = path / filename
        if not target.is_file() or (verify_hashes and sha256_file(target) != expected_hash):
            raise M2ArtifactError(f"pinned snapshot file hash mismatch: {filename}")
    return path, expected


def _enforce_transform_contract(
    train_transform: Any,
    prediction_transform: Any,
    recipe: Mapping[str, Any],
) -> None:
    """Patch OpenCLIP 3.2's ignored ratio and assert the frozen transforms."""

    from torchvision.transforms import Normalize, RandomResizedCrop, Resize, ToTensor
    from torchvision.transforms.functional import InterpolationMode

    train_cfg = recipe["preprocessing"]["train_transform"]
    prediction_cfg = recipe["preprocessing"]["prediction_transform"]
    train_steps = list(getattr(train_transform, "transforms", []))
    prediction_steps = list(getattr(prediction_transform, "transforms", []))
    if len(train_steps) != 4 or len(prediction_steps) != 4:
        raise M2ArtifactError("pinned OpenCLIP transform step count mismatch")
    for steps, name in ((train_steps, "training"), (prediction_steps, "prediction")):
        converter = steps[1]
        if getattr(converter, "__name__", None) != "_convert_to_rgb":
            raise M2ArtifactError(f"pinned {name} transform does not force RGB")
        if not isinstance(steps[2], ToTensor) or not isinstance(steps[3], Normalize):
            raise M2ArtifactError(f"pinned {name} tensor/normalization order mismatch")
    crops = [step for step in train_steps if isinstance(step, RandomResizedCrop)]
    if len(crops) != 1:
        raise M2ArtifactError("pinned training transform must have one RandomResizedCrop")
    crop = crops[0]
    exact_size = tuple(int(value) for value in train_cfg["random_resized_crop_size"])
    exact_scale = tuple(float(value) for value in train_cfg["random_resized_crop_scale"])
    exact_ratio = tuple(float(value) for value in train_cfg["random_resized_crop_ratio"])
    # open_clip_torch 3.2.0 ignores aug_cfg.ratio in its non-timm transform
    # builder.  RandomResizedCrop reads this attribute on every call, so setting
    # it explicitly preserves the frozen 0.75..1.3333 sampling contract.
    crop.ratio = exact_ratio
    if tuple(crop.size) != exact_size or tuple(crop.scale) != exact_scale:
        raise M2ArtifactError("pinned training crop size/scale mismatch")
    if tuple(crop.ratio) != exact_ratio:
        raise M2ArtifactError("pinned training crop ratio mismatch")
    if crop.interpolation != InterpolationMode.BICUBIC or crop.antialias is not True:
        raise M2ArtifactError("pinned training crop interpolation/antialias mismatch")
    resize_steps = [step for step in prediction_steps if isinstance(step, Resize)]
    if len(resize_steps) != 1:
        raise M2ArtifactError("pinned prediction transform must have one Resize")
    resize = resize_steps[0]
    exact_prediction_size = tuple(int(value) for value in prediction_cfg["resize_size"])
    if tuple(resize.size) != exact_prediction_size:
        raise M2ArtifactError("pinned prediction resize size mismatch")
    if resize.interpolation != InterpolationMode.BICUBIC or resize.antialias is not True:
        raise M2ArtifactError("pinned prediction resize interpolation/antialias mismatch")
    for steps, cfg, name in (
        (train_steps, train_cfg, "training"),
        (prediction_steps, prediction_cfg, "prediction"),
    ):
        normalizers = [step for step in steps if isinstance(step, Normalize)]
        if len(normalizers) != 1:
            raise M2ArtifactError(f"pinned {name} transform must have one Normalize")
        normalize = normalizers[0]
        observed_mean = tuple(float(value) for value in normalize.mean)
        observed_std = tuple(float(value) for value in normalize.std)
        if observed_mean != tuple(float(value) for value in cfg["normalize_mean"]):
            raise M2ArtifactError(f"pinned {name} normalization mean mismatch")
        if observed_std != tuple(float(value) for value in cfg["normalize_std"]):
            raise M2ArtifactError(f"pinned {name} normalization std mismatch")


def _load_fresh_encoder(
    config: Mapping[str, Any], device: torch.device
) -> tuple[nn.Module, Any, Any, Any, dict[str, Any]]:
    snapshot, snapshot_hashes = _pinned_snapshot(config)
    try:
        import open_clip
    except ImportError as exc:
        raise M2ArtifactError("open_clip is required for M2 encoder adaptation") from exc
    recipe = config["experts"]["image_encoder"]["fold_contained_adaptation"][
        "adaptation_recipe"
    ]
    preprocessing = recipe["preprocessing"]
    train_cfg = preprocessing["train_transform"]
    pred_cfg = preprocessing["prediction_transform"]
    local_name = f"local-dir:{snapshot}"
    model, train_transform, prediction_transform = open_clip.create_model_and_transforms(
        local_name,
        device=device,
        precision="fp32",
        force_image_size=tuple(int(value) for value in pred_cfg["resize_size"]),
        image_mean=tuple(float(value) for value in pred_cfg["normalize_mean"]),
        image_std=tuple(float(value) for value in pred_cfg["normalize_std"]),
        image_interpolation="bicubic",
        image_resize_mode="squash",
        aug_cfg={
            "scale": tuple(float(value) for value in train_cfg["random_resized_crop_scale"]),
        },
    )
    _enforce_transform_contract(
        train_transform, prediction_transform, recipe
    )
    tokenizer = open_clip.get_tokenizer(local_name)
    provenance = {
        "checkpoint_id": config["experts"]["image_encoder"][
            "externally_pretrained_fixed"
        ]["checkpoint_id"],
        "hub_revision": config["experts"]["image_encoder"][
            "externally_pretrained_fixed"
        ]["hub_revision"],
        "snapshot_path": str(snapshot),
        "snapshot_file_sha256": snapshot_hashes,
    }
    return model, train_transform, prediction_transform, tokenizer, provenance


def _frozen_text_weights(
    model: nn.Module,
    tokenizer: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    recipe = config["experts"]["image_encoder"]["fold_contained_adaptation"][
        "adaptation_recipe"
    ]
    prompts = list(recipe["prompts"]["values"])
    classes = list(config["class_ontology"]["classes"])
    if len(prompts) != N_CLASSES or len(classes) != N_CLASSES:
        raise M2ArtifactError("frozen prompt and ontology sizes must both be 18")
    dense_ids = [int(item["dense_id"]) for item in classes]
    if dense_ids != list(range(N_CLASSES)):
        raise M2ArtifactError("frozen ontology is not in dense-ID order")
    tokens = tokenizer(prompts).to(device)
    model.eval()
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = F.normalize(features.float(), dim=-1)
    if features.shape[0] != N_CLASSES or not torch.isfinite(features).all():
        raise M2ArtifactError("frozen prompt text weights are invalid")
    return features.t().contiguous()


def _configure_vision_adaptation(
    model: nn.Module, config: Mapping[str, Any]
) -> list[nn.Parameter]:
    recipe = config["experts"]["image_encoder"]["fold_contained_adaptation"][
        "adaptation_recipe"
    ]
    groups = int(recipe["vision_tower"]["unlocked_groups_from_end"])
    model.lock_image_tower(unlocked_groups=groups, freeze_bn_stats=False)
    model.lock_text_tower()
    # Logit scale/bias and any other top-level state are outside the visual
    # tower and are not part of the fixed-multiplier objective.
    for name, parameter in model.named_parameters():
        if not name.startswith("visual."):
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.visual.parameters() if parameter.requires_grad]
    if not trainable:
        raise M2ArtifactError("vision adaptation selected no trainable parameters")
    nonvisual = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("visual.")
    ]
    if nonvisual:
        raise M2ArtifactError(f"non-visual parameters remained trainable: {nonvisual[:5]}")
    return trainable


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        raise M2Error("real M2 adaptation requires a CUDA device")
    return torch.autocast(device_type="cuda")


def _adapt_encoder(
    model: nn.Module,
    fit_rows: pd.DataFrame,
    *,
    transform: Any,
    text_weights: torch.Tensor,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Perform the frozen five-epoch fit without constructing held-out data."""

    generator = reset_reproducibility(seed)
    recipe = config["experts"]["image_encoder"]["fold_contained_adaptation"][
        "adaptation_recipe"
    ]
    train_cfg = recipe["training"]
    if int(train_cfg["epochs"]) != 5 or int(train_cfg["batch_size"]) != 16:
        raise M2ArtifactError("resolved adaptation budget is not the frozen 5 epochs/batch 16")
    dataset = _LegacyImageDataset(
        fit_rows,
        config=config,
        transform=transform,
        include_labels=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        generator=generator,
    )
    trainable = _configure_vision_adaptation(model, config)
    optimizer_cfg = recipe["optimizer"]
    optimizer = torch.optim.Adam(
        trainable,
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["epsilon"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        amsgrad=bool(optimizer_cfg["amsgrad"]),
    )
    scheduler_cfg = recipe["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_cfg["t_max_epochs"]),
        eta_min=float(scheduler_cfg["eta_min"]),
    )
    history: list[dict[str, Any]] = []
    for epoch in range(5):
        model.eval()
        model.visual.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device):
                image_features = F.normalize(model.encode_image(images), dim=-1)
                logits = 100.0 * image_features @ text_weights
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            batch = int(labels.numel())
            loss_sum += float(loss.detach()) * batch
            correct += int((logits.argmax(dim=1) == labels).sum().detach())
            seen += batch
        if seen != len(fit_rows):
            raise M2ArtifactError("adaptation dataloader did not consume every fitting row")
        history.append(
            {
                "epoch": epoch + 1,
                "fit_loss": loss_sum / seen,
                "fit_top1": correct / seen,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        scheduler.step()
    return history


def _extract_embeddings(
    model: nn.Module,
    rows: pd.DataFrame,
    *,
    transform: Any,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> np.ndarray:
    dataset = _LegacyImageDataset(
        rows.loc[:, ["row_uid", "file", "image_source"]],
        config=config,
        transform=transform,
        include_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        generator=reset_reproducibility(seed),
    )
    embeddings: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, ordinal in loader:
            encoded = model.encode_image(images.to(device, non_blocking=True)).float()
            encoded = F.normalize(encoded, dim=-1)
            embeddings.append(encoded.cpu())
            indices.append(torch.as_tensor(ordinal, dtype=torch.int64).cpu())
    if not embeddings:
        raise M2ArtifactError("embedding extraction produced no rows")
    matrix = torch.cat(embeddings, dim=0)
    order = torch.cat(indices, dim=0)
    if not torch.equal(order, torch.arange(len(rows), dtype=torch.int64)):
        raise M2ArtifactError("deterministic embedding extraction changed row order")
    expected_dim = int(config["experts"]["image_input"]["embedding_dim"])
    if matrix.shape != (len(rows), expected_dim):
        raise M2ArtifactError(
            f"adapted embedding shape mismatch: {tuple(matrix.shape)} != ({len(rows)}, {expected_dim})"
        )
    if not torch.isfinite(matrix).all():
        raise M2ArtifactError("adapted embeddings must be finite")
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise M2ArtifactError("adapted embeddings are not L2-normalized")
    return matrix.numpy().astype(np.float32, copy=False)


def _head_input(
    mode: str, image_embeddings: np.ndarray, geo_values: np.ndarray
) -> np.ndarray:
    if mode == "image_only":
        result = image_embeddings
    elif mode == "geo_only":
        result = geo_values
    elif mode == "raw_concat":
        result = np.concatenate([image_embeddings, geo_values], axis=1)
    else:
        raise M2Error(f"unknown expert mode: {mode}")
    result = np.asarray(result, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise M2Error(f"{mode} head input is invalid")
    return result


def _fit_head(
    mode: str,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[MLPHead, list[dict[str, Any]]]:
    generator = reset_reproducibility(seed)
    head_cfg = config["experts"]["head"]
    expected_dim = int(config["experts"]["mode_inputs"][mode]["input_dim"])
    if features.shape != (len(labels), expected_dim):
        raise M2Error(
            f"{mode} fitting input shape mismatch: {features.shape} != ({len(labels)}, {expected_dim})"
        )
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= N_CLASSES):
        raise M2Error("fitting labels must be dense IDs in [0, 17]")
    model = MLPHead(
        input_dim=expected_dim,
        hidden_dim=int(head_cfg["hidden_dim"]),
        output_dim=N_CLASSES,
        dropout=float(head_cfg["dropout"]),
    ).to(device=device, dtype=torch.float32)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels.astype(np.int64)))
    loader = DataLoader(
        dataset,
        batch_size=int(head_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(head_cfg["num_workers"]),
        drop_last=bool(head_cfg["drop_last"]),
        pin_memory=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(head_cfg["learning_rate"]),
        weight_decay=float(head_cfg["weight_decay"]),
        betas=tuple(float(value) for value in head_cfg["optimizer_betas"]),
        eps=float(head_cfg["optimizer_eps"]),
        amsgrad=bool(head_cfg["amsgrad"]),
    )
    history: list[dict[str, Any]] = []
    epochs = int(config["experts"]["epochs"])
    if epochs != 50:
        raise M2ArtifactError("resolved expert-head budget is not 50 epochs")
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            batch = int(batch_labels.numel())
            loss_sum += float(loss.detach()) * batch
            correct += int((logits.argmax(dim=1) == batch_labels).sum().detach())
            seen += batch
        if seen != len(labels):
            raise M2ArtifactError(f"{mode} head dataloader did not consume all fitting rows")
        history.append(
            {
                "epoch": epoch + 1,
                "fit_loss": loss_sum / seen,
                "fit_top1": correct / seen,
            }
        )
    return model, history


def _predict_head(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(np.asarray(features, dtype=np.float32)))
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            batches.append(logits.float().cpu().numpy().astype(np.float64))
    result = np.concatenate(batches, axis=0)
    if result.shape != (len(features), N_CLASSES) or not np.isfinite(result).all():
        raise M2ArtifactError("expert head emitted invalid logits")
    return result


def _replay_saved_checkpoints(
    context: FrozenM2Context,
    spec: ProducerSpec,
    *,
    seed: int,
    directory: Path,
    prediction_rows: pd.DataFrame,
    reference: pa.Table,
    device: torch.device,
) -> dict[str, Any]:
    """Reload every saved learned component and reproduce held-out outputs."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise M2ArtifactError("safetensors is required for checkpoint reproduction") from exc
    replay_model: nn.Module | None = None
    replay_heads: dict[str, MLPHead] = {}
    try:
        reset_reproducibility(seed)
        replay_model, _, prediction_transform, _, _ = _load_fresh_encoder(
            context.config, device
        )
        visual_state = load_file(
            str(directory / "adapted_visual_tower.safetensors"), device="cpu"
        )
        replay_model.visual.load_state_dict(visual_state, strict=True)
        del visual_state
        scaler_record = _validate_scaler_file(directory / "geo_standardization.json")
        prediction_geo = apply_geo_standardization(
            _geo_source_projection(context, prediction_rows), scaler_record
        )
        prediction_image = _extract_embeddings(
            replay_model,
            prediction_rows,
            transform=prediction_transform,
            config=context.config,
            seed=seed,
            device=device,
        )
        logits_by_mode: dict[str, np.ndarray] = {}
        for mode in MODES:
            reset_reproducibility(seed)
            head_cfg = context.config["experts"]["head"]
            head = MLPHead(
                input_dim=int(context.config["experts"]["mode_inputs"][mode]["input_dim"]),
                hidden_dim=int(head_cfg["hidden_dim"]),
                output_dim=N_CLASSES,
                dropout=float(head_cfg["dropout"]),
            ).to(device=device, dtype=torch.float32)
            state = load_file(str(directory / f"{mode}_head.safetensors"), device="cpu")
            head.load_state_dict(state, strict=True)
            del state
            replay_heads[mode] = head
            logits_by_mode[mode] = _predict_head(
                head,
                _head_input(mode, prediction_image, prediction_geo),
                device,
            )
        replayed = build_output_table(
            prediction_rows,
            seed=seed,
            logits_by_mode=logits_by_mode,
            include_fold=spec.include_fold,
            schema_version=str(context.config["schema_version"]),
            protocol_id=str(context.config["protocol_id"]),
        )
        return validate_reproduced_output(
            reference, replayed, include_fold=spec.include_fold
        )
    finally:
        replay_heads.clear()
        del replay_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _safetensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in sorted(module.state_dict().items())
    }


def _save_safetensors(module: nn.Module, path: Path) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise M2ArtifactError("safetensors is required for M2 checkpoints") from exc
    if path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    state = _safetensor_state(module)
    save_file(state, str(path), metadata={"format": "pt"})
    os.chmod(path, 0o444)
    return path


def _write_bytes_exclusive(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable artifact already exists: {path}") from exc
    os.chmod(path, 0o444)
    return path


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    return _write_bytes_exclusive(path, canonical_json_bytes(dict(value)) + b"\n")


def _write_yaml_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    payload = yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).encode("utf-8")
    return _write_bytes_exclusive(path, payload)


def _runtime_provenance(device: torch.device) -> dict[str, Any]:
    distributions = (
        "torch",
        "torchvision",
        "open_clip_torch",
        "timm",
        "huggingface_hub",
        "numpy",
        "pandas",
        "pyarrow",
        "opencv_python",
        "pillow",
        "safetensors",
    )
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cuda_record: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda_record.update(
            {
                "device_index": int(index),
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "unavailable")),
                "compute_capability": [properties.major, properties.minor],
                "total_memory": int(properties.total_memory),
            }
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda": cuda_record,
        "autocast": "device_native_torch_autocast",
        "grad_scaler": False,
    }


def _class_plot_support(fit_rows: pd.DataFrame) -> dict[str, int]:
    support = (
        fit_rows.loc[:, ["label_id_dense", "plot_idx"]]
        .drop_duplicates()
        .groupby("label_id_dense", observed=False)["plot_idx"]
        .nunique()
        .to_dict()
    )
    return {str(dense_id): int(support.get(dense_id, 0)) for dense_id in range(N_CLASSES)}


def _feature_schema_hash(include_fold: bool) -> str:
    schema = _arrow_output_schema(include_fold)
    return canonical_sha256(
        {
            "geo_columns": list(GEO_COLUMNS),
            "image_embedding_dim": 1152,
            "raw_concat_order": ["image_embedding", "standardized_geo"],
            "class_count": N_CLASSES,
            "output_columns": list(
                OOF_OUTPUT_COLUMNS if include_fold else VALIDATION_OUTPUT_COLUMNS
            ),
            "arrow_schema": str(schema),
        }
    )


def _manifest_self_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return canonical_sha256(payload)


def _component_hashes(directory: Path, output_filename: str) -> dict[str, str]:
    names = (
        output_filename,
        "adapted_visual_tower.safetensors",
        "image_only_head.safetensors",
        "geo_only_head.safetensors",
        "raw_concat_head.safetensors",
        "geo_standardization.json",
        "resolved_stage_config.yaml",
        "training_metrics.json",
    )
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise M2ArtifactError(f"producer is missing component files: {missing}")
    return {name: sha256_file(directory / name) for name in names}


def _producer_manifest(
    context: FrozenM2Context,
    spec: ProducerSpec,
    *,
    seed: int,
    fit_rows: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    table: pa.Table,
    staging: Path,
    encoder_provenance: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoint_reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = validate_fit_prediction_plot_provenance(
        fit_rows["plot_idx"], prediction_rows["plot_idx"]
    )
    components = _component_hashes(staging, spec.output_filename)
    checkpoint_files = {
        name: components[name]
        for name in (
            "adapted_visual_tower.safetensors",
            "image_only_head.safetensors",
            "geo_only_head.safetensors",
            "raw_concat_head.safetensors",
        )
    }
    recipe = context.config["experts"]["image_encoder"]["fold_contained_adaptation"][
        "adaptation_recipe"
    ]
    parent_roles = (
        ["development_assignments", "fold_local_expert_fit"]
        if spec.include_fold
        else ["development_assignments", "development_train_expert_fit"]
    )
    validate_artifact_parent_roles(spec.artifact_role, parent_roles)
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "protocol_id": str(context.config["protocol_id"]),
        "artifact_role": spec.artifact_role,
        "parent_roles": parent_roles,
        "parent_artifact_hashes": context.parent_hashes,
        "stage_id": spec.stage_id,
        "train_oof_fold": spec.fold,
        "row_count": len(table),
        "plot_count": int(prediction_rows["plot_idx"].nunique()),
        "fitting_row_count": len(fit_rows),
        "fitting_plot_hash": provenance["fitting_plot_sha256"],
        "prediction_plot_hash": provenance["prediction_plot_sha256"],
        "zero_plot_overlap": provenance["zero_plot_overlap"],
        "class_map_hash": context.protocol_manifest["class_map_sha256"],
        "feature_schema_hash": _feature_schema_hash(spec.include_fold),
        "training_seed": seed,
        "encoder_initialization_hash": canonical_sha256(dict(encoder_provenance)),
        "encoder_initialization": dict(encoder_provenance),
        "encoder_adaptation_recipe_hash": canonical_sha256(recipe),
        "input_preprocessing_hash": canonical_sha256(recipe["preprocessing"]),
        "checkpoint_hash": canonical_sha256(checkpoint_files),
        "checkpoint_file_sha256": checkpoint_files,
        "scaler_hash": components["geo_standardization.json"],
        "fitting_class_plot_support_by_dense_id": _class_plot_support(fit_rows),
        "content_sha256": logical_table_sha256(table),
        "model_output_filename": spec.output_filename,
        "model_output_file_sha256": components[spec.output_filename],
        "component_file_sha256": components,
        "output_columns": table.column_names,
        "output_arrow_schema": str(table.schema),
        "probability_basis": "native_t1_uncalibrated",
        "temperature": 1.0,
        "calibrated": False,
        "logits_authoritative": True,
        "probability_integrity_atol": PROBABILITY_ATOL,
        "reproduction_atol": REPRODUCTION_ATOL,
        "reproduction_rtol": REPRODUCTION_RTOL,
        "checkpoint_reproduction": dict(checkpoint_reproduction),
        "m2_code_file_sha256": context.code_file_hashes,
        "m2_code_sha256": context.code_hash,
        "runtime": dict(runtime),
        "runtime_sha256": canonical_sha256(dict(runtime)),
    }
    manifest["manifest_sha256"] = _manifest_self_hash(manifest)
    return manifest


def _validate_scaler_file(path: Path) -> dict[str, Any]:
    value = _read_json_mapping(path, name="geo standardization")
    expected_keys = {
        "schema_version",
        "feature_columns",
        "dtype",
        "variance_ddof",
        "zero_std_policy",
        "mean",
        "std",
    }
    if set(value) != expected_keys:
        raise M2ArtifactError("geo standardization field allow-list mismatch")
    if value.get("schema_version") != "geo_helpfulness.geo_standardization.v1":
        raise M2ArtifactError("unsupported geo standardization schema")
    if value.get("feature_columns") != list(GEO_COLUMNS):
        raise M2ArtifactError("geo standardization feature order is not A00…A63")
    if value.get("dtype") != "float32" or value.get("variance_ddof") != 0:
        raise M2ArtifactError("geo standardization dtype/variance declaration mismatch")
    if value.get("zero_std_policy") != "replace_with_one":
        raise M2ArtifactError("geo standardization zero-std policy mismatch")
    _coerce_scaler(value, len(GEO_COLUMNS))
    return value


def _validate_checkpoint_schema(
    directory: Path, context: FrozenM2Context
) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise M2ArtifactError("safetensors is required to validate M2 checkpoints") from exc

    expected_inputs = {
        "image_only": int(context.config["experts"]["mode_inputs"]["image_only"]["input_dim"]),
        "geo_only": int(context.config["experts"]["mode_inputs"]["geo_only"]["input_dim"]),
        "raw_concat": int(context.config["experts"]["mode_inputs"]["raw_concat"]["input_dim"]),
    }
    hidden = int(context.config["experts"]["head"]["hidden_dim"])
    for mode, input_dim in expected_inputs.items():
        path = directory / f"{mode}_head.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            expected_shapes = {
                "net.0.bias": [hidden],
                "net.0.weight": [hidden, input_dim],
                "net.3.bias": [N_CLASSES],
                "net.3.weight": [N_CLASSES, hidden],
            }
            if keys != sorted(expected_shapes):
                raise M2ArtifactError(f"{mode} head checkpoint key set mismatch")
            for key, shape in expected_shapes.items():
                view = handle.get_slice(key)
                if list(view.get_shape()) != shape or str(view.get_dtype()) != "F32":
                    raise M2ArtifactError(f"{mode} head checkpoint tensor mismatch: {key}")
    snapshot, _ = _pinned_snapshot(context.config, verify_hashes=False)
    checkpoint_filename = context.config["experts"]["image_encoder"][
        "externally_pretrained_fixed"
    ]["checkpoint_filename"]
    expected_visual: dict[str, tuple[list[int], str]] = {}
    with safe_open(snapshot / checkpoint_filename, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("visual."):
                view = handle.get_slice(key)
                expected_visual[key.removeprefix("visual.")] = (
                    list(view.get_shape()),
                    str(view.get_dtype()),
                )
    visual_path = directory / "adapted_visual_tower.safetensors"
    observed_visual: dict[str, tuple[list[int], str]] = {}
    with safe_open(visual_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            view = handle.get_slice(key)
            observed_visual[key] = (list(view.get_shape()), str(view.get_dtype()))
    if observed_visual != expected_visual:
        raise M2ArtifactError(
            "adapted visual tower checkpoint does not contain the complete pinned visual state"
        )


def _validate_stage_records(
    context: FrozenM2Context,
    spec: ProducerSpec,
    *,
    seed: int,
    target: Path,
    fit_rows: pd.DataFrame,
    prediction_rows: pd.DataFrame,
) -> dict[str, Any]:
    stage = _read_yaml_mapping(target / "resolved_stage_config.yaml", name="resolved stage config")
    expected_stage_keys = {
        "schema_version",
        "protocol_id",
        "stage_id",
        "training_seed",
        "train_oof_fold",
        "fitting_row_uids_sha256",
        "prediction_row_uids_sha256",
        "geo_feature_columns",
        "mode_order",
        "encoder_initialization",
        "adaptation_recipe",
        "head_recipe",
        "expert_epochs",
        "m2_code_sha256",
    }
    if set(stage) != expected_stage_keys:
        raise M2ArtifactError("resolved stage configuration field allow-list mismatch")
    if stage.get("schema_version") != "geo_helpfulness.resolved_m2_stage.v1":
        raise M2ArtifactError("resolved stage configuration schema mismatch")
    expected_stage_values = {
        "protocol_id": context.config["protocol_id"],
        "stage_id": spec.stage_id,
        "training_seed": seed,
        "train_oof_fold": spec.fold,
        "fitting_row_uids_sha256": canonical_sha256(fit_rows["row_uid"].astype(str).tolist()),
        "prediction_row_uids_sha256": canonical_sha256(
            prediction_rows["row_uid"].astype(str).tolist()
        ),
        "geo_feature_columns": list(GEO_COLUMNS),
        "mode_order": list(MODES),
        "adaptation_recipe": context.config["experts"]["image_encoder"][
            "fold_contained_adaptation"
        ]["adaptation_recipe"],
        "head_recipe": context.config["experts"]["head"],
        "expert_epochs": context.config["experts"]["epochs"],
        "m2_code_sha256": context.code_hash,
    }
    for key, expected in expected_stage_values.items():
        if stage.get(key) != expected:
            raise M2ArtifactError(f"resolved stage configuration mismatch: {key}")
    initialization = stage.get("encoder_initialization")
    if not isinstance(initialization, dict):
        raise M2ArtifactError("resolved stage encoder initialization is missing")

    metrics = _read_json_mapping(target / "training_metrics.json", name="training metrics")
    expected_metric_keys = {
        "schema_version",
        "protocol_id",
        "stage_id",
        "training_seed",
        "labels_scope",
        "heldout_metrics",
        "adaptation_history",
        "head_history",
        "fitting_class_plot_support_by_dense_id",
        "checkpoint_reproduction",
    }
    if set(metrics) != expected_metric_keys:
        raise M2ArtifactError("training metrics field allow-list mismatch")
    if metrics.get("schema_version") != "geo_helpfulness.m2_training_metrics.v1":
        raise M2ArtifactError("training metrics schema mismatch")
    if metrics.get("protocol_id") != context.config["protocol_id"]:
        raise M2ArtifactError("training metrics protocol mismatch")
    if metrics.get("stage_id") != spec.stage_id or metrics.get("training_seed") != seed:
        raise M2ArtifactError("training metrics producer identity mismatch")
    if metrics.get("labels_scope") != "fitting_partition_only":
        raise M2ArtifactError("training metrics label scope is not fitting-only")
    if metrics.get("heldout_metrics") is not None:
        raise M2ArtifactError("M2 producer must not contain held-out metrics")
    adaptation_history = metrics.get("adaptation_history")
    head_history = metrics.get("head_history")
    if not isinstance(adaptation_history, list) or len(adaptation_history) != 5:
        raise M2ArtifactError("adaptation history must contain exactly five epochs")
    if not isinstance(head_history, dict) or set(head_history) != set(MODES):
        raise M2ArtifactError("head history mode set mismatch")
    if any(not isinstance(head_history[mode], list) or len(head_history[mode]) != 50 for mode in MODES):
        raise M2ArtifactError("every expert-head history must contain exactly 50 epochs")
    if metrics.get("fitting_class_plot_support_by_dense_id") != _class_plot_support(fit_rows):
        raise M2ArtifactError("training metrics class/plot support mismatch")
    reproduction = metrics.get("checkpoint_reproduction")
    expected_reproduction = {
        "valid": True,
        "row_count": len(prediction_rows),
        "atol": REPRODUCTION_ATOL,
        "rtol": REPRODUCTION_RTOL,
    }
    if reproduction != expected_reproduction:
        raise M2ArtifactError("saved-checkpoint reproduction record mismatch")
    return stage


def _expected_identity_projection(
    rows: pd.DataFrame, *, include_fold: bool, seed: int
) -> pd.DataFrame:
    columns = ["row_uid", "file", "file_lower", "plot_idx"]
    expected = rows.loc[:, columns].copy()
    if include_fold:
        expected["train_oof_fold"] = rows["train_oof_fold"].astype(np.int8)
    expected["training_seed"] = np.int8(seed)
    return expected.sort_values(["row_uid", "training_seed"], kind="mergesort").reset_index(
        drop=True
    )


def validate_producer(
    context: FrozenM2Context,
    spec: ProducerSpec,
    *,
    seed: int,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    training_seed = validate_training_seed(seed)
    target = (
        Path(directory).resolve()
        if directory is not None
        else (context.output_root / spec.relative_directory).resolve()
    )
    if not target.is_dir():
        raise M2ArtifactError(f"producer directory does not exist: {target}")
    expected_names = set(PRODUCER_FILES).union({spec.output_filename})
    observed_names = {path.name for path in target.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise M2ArtifactError(
            f"producer file allow-list mismatch; observed={sorted(observed_names)}, "
            f"expected={sorted(expected_names)}"
        )
    manifest = _read_json_mapping(target / "manifest.json", name="producer manifest")
    required_fields = set(
        context.config["artifact_contract"]["model_output_manifest_required_fields"]
    )
    missing = sorted(required_fields.difference(manifest))
    if missing:
        raise M2ArtifactError(f"producer manifest is missing frozen fields: {missing}")
    if manifest.get("manifest_sha256") != _manifest_self_hash(manifest):
        raise M2ArtifactError("producer manifest self-hash mismatch")
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise M2ArtifactError("producer manifest schema mismatch")
    if manifest.get("protocol_id") != str(context.config["protocol_id"]):
        raise M2ArtifactError("producer protocol ID mismatch")
    if manifest.get("artifact_role") != spec.artifact_role:
        raise M2ArtifactError("producer artifact role mismatch")
    expected_parent_roles = (
        ["development_assignments", "fold_local_expert_fit"]
        if spec.include_fold
        else ["development_assignments", "development_train_expert_fit"]
    )
    if manifest.get("parent_roles") != expected_parent_roles:
        raise M2ArtifactError("producer parent-role declaration mismatch")
    if manifest.get("stage_id") != spec.stage_id:
        raise M2ArtifactError("producer stage ID mismatch")
    if manifest.get("train_oof_fold") != spec.fold:
        raise M2ArtifactError("producer fold mismatch")
    if manifest.get("training_seed") != training_seed:
        raise M2ArtifactError("producer training seed mismatch")
    if manifest.get("parent_artifact_hashes") != context.parent_hashes:
        raise M2ArtifactError("producer M1 parent fingerprints are stale")
    if manifest.get("m2_code_file_sha256") != context.code_file_hashes:
        raise M2ArtifactError("producer M2 implementation file fingerprints are stale")
    if manifest.get("m2_code_sha256") != context.code_hash:
        raise M2ArtifactError("producer M2 implementation aggregate hash is stale")
    validate_artifact_parent_roles(
        str(manifest["artifact_role"]), list(manifest.get("parent_roles", []))
    )
    fit_rows, prediction_rows = _partitions_for_spec(context.assignments, spec)
    provenance = validate_fit_prediction_plot_provenance(
        fit_rows["plot_idx"], prediction_rows["plot_idx"]
    )
    if manifest.get("fitting_plot_hash") != provenance["fitting_plot_sha256"]:
        raise M2ArtifactError("producer fitting plot fingerprint mismatch")
    if manifest.get("prediction_plot_hash") != provenance["prediction_plot_sha256"]:
        raise M2ArtifactError("producer prediction plot fingerprint mismatch")
    if manifest.get("zero_plot_overlap") is not True:
        raise M2ArtifactError("producer does not attest zero fitting/prediction plot overlap")
    if manifest.get("fitting_class_plot_support_by_dense_id") != _class_plot_support(fit_rows):
        raise M2ArtifactError("producer fitting class/plot support mismatch")
    if manifest.get("class_map_hash") != context.protocol_manifest["class_map_sha256"]:
        raise M2ArtifactError("producer class-map fingerprint mismatch")
    if manifest.get("feature_schema_hash") != _feature_schema_hash(spec.include_fold):
        raise M2ArtifactError("producer feature/output schema fingerprint mismatch")
    recipe = context.config["experts"]["image_encoder"]["fold_contained_adaptation"][
        "adaptation_recipe"
    ]
    if manifest.get("encoder_adaptation_recipe_hash") != canonical_sha256(recipe):
        raise M2ArtifactError("producer encoder-adaptation recipe fingerprint mismatch")
    if manifest.get("input_preprocessing_hash") != canonical_sha256(recipe["preprocessing"]):
        raise M2ArtifactError("producer preprocessing fingerprint mismatch")
    initialization = manifest.get("encoder_initialization")
    if not isinstance(initialization, dict):
        raise M2ArtifactError("producer encoder initialization record is missing")
    if manifest.get("encoder_initialization_hash") != canonical_sha256(initialization):
        raise M2ArtifactError("producer encoder initialization fingerprint is inconsistent")
    frozen_initialization = context.config["experts"]["image_encoder"][
        "externally_pretrained_fixed"
    ]
    expected_snapshot_hashes = {
        str(frozen_initialization["checkpoint_filename"]): str(
            frozen_initialization["checkpoint_sha256"]
        ),
        str(frozen_initialization["open_clip_config_filename"]): str(
            frozen_initialization["open_clip_config_sha256"]
        ),
        str(frozen_initialization["tokenizer_filename"]): str(
            frozen_initialization["tokenizer_sha256"]
        ),
        "tokenizer_config.json": str(frozen_initialization["tokenizer_config_sha256"]),
        "special_tokens_map.json": str(frozen_initialization["special_tokens_map_sha256"]),
    }
    if initialization.get("checkpoint_id") != frozen_initialization["checkpoint_id"]:
        raise M2ArtifactError("producer encoder checkpoint ID mismatch")
    if initialization.get("hub_revision") != frozen_initialization["hub_revision"]:
        raise M2ArtifactError("producer encoder revision mismatch")
    if initialization.get("snapshot_file_sha256") != expected_snapshot_hashes:
        raise M2ArtifactError("producer pinned snapshot file fingerprints mismatch")
    if not isinstance(initialization.get("snapshot_path"), str):
        raise M2ArtifactError("producer pinned snapshot path provenance is missing")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or manifest.get("runtime_sha256") != canonical_sha256(runtime):
        raise M2ArtifactError("producer runtime provenance fingerprint mismatch")
    if manifest.get("fitting_row_count") != len(fit_rows):
        raise M2ArtifactError("producer fitting row count mismatch")
    if manifest.get("model_output_filename") != spec.output_filename:
        raise M2ArtifactError("producer model-output filename mismatch")
    if manifest.get("probability_integrity_atol") != PROBABILITY_ATOL:
        raise M2ArtifactError("producer probability integrity tolerance mismatch")
    if manifest.get("reproduction_atol") != REPRODUCTION_ATOL or manifest.get(
        "reproduction_rtol"
    ) != REPRODUCTION_RTOL:
        raise M2ArtifactError("producer reproduction tolerance mismatch")
    expected_reproduction = {
        "valid": True,
        "row_count": len(prediction_rows),
        "atol": REPRODUCTION_ATOL,
        "rtol": REPRODUCTION_RTOL,
    }
    if manifest.get("checkpoint_reproduction") != expected_reproduction:
        raise M2ArtifactError("producer checkpoint-reproduction declaration mismatch")
    components = _component_hashes(target, spec.output_filename)
    if manifest.get("component_file_sha256") != components:
        raise M2ArtifactError("producer component file fingerprint mismatch")
    checkpoints = {
        name: components[name]
        for name in (
            "adapted_visual_tower.safetensors",
            "image_only_head.safetensors",
            "geo_only_head.safetensors",
            "raw_concat_head.safetensors",
        )
    }
    if manifest.get("checkpoint_file_sha256") != checkpoints:
        raise M2ArtifactError("producer checkpoint component fingerprints mismatch")
    if manifest.get("checkpoint_hash") != canonical_sha256(checkpoints):
        raise M2ArtifactError("producer checkpoint aggregate fingerprint mismatch")
    if manifest.get("scaler_hash") != components["geo_standardization.json"]:
        raise M2ArtifactError("producer scaler fingerprint mismatch")
    scaler_record = _validate_scaler_file(target / "geo_standardization.json")
    expected_scaler = fit_geo_standardization(_geo_source_projection(context, fit_rows))
    observed_mean, observed_std = _coerce_scaler(scaler_record, len(GEO_COLUMNS))
    if not np.array_equal(observed_mean, expected_scaler.mean) or not np.array_equal(
        observed_std, expected_scaler.std
    ):
        raise M2ArtifactError("producer geo standardization does not reproduce from fitting rows")
    _validate_checkpoint_schema(target, context)
    stage_record = _validate_stage_records(
        context,
        spec,
        seed=training_seed,
        target=target,
        fit_rows=fit_rows,
        prediction_rows=prediction_rows,
    )
    if stage_record.get("encoder_initialization") != initialization:
        raise M2ArtifactError("stage and manifest encoder initialization records disagree")
    table = read_output_parquet(target / spec.output_filename, include_fold=spec.include_fold)
    if set(table["schema_version"].to_pylist()) != {str(context.config["schema_version"])}:
        raise M2ArtifactError("producer row schema_version mismatch")
    if set(table["protocol_id"].to_pylist()) != {str(context.config["protocol_id"])}:
        raise M2ArtifactError("producer row protocol_id mismatch")
    expected_columns = list(
        OOF_OUTPUT_COLUMNS if spec.include_fold else VALIDATION_OUTPUT_COLUMNS
    )
    if manifest.get("output_columns") != expected_columns:
        raise M2ArtifactError("producer manifest output column declaration mismatch")
    if manifest.get("output_arrow_schema") != str(_arrow_output_schema(spec.include_fold)):
        raise M2ArtifactError("producer manifest Arrow schema declaration mismatch")
    validation = validate_output_table(
        table, include_fold=spec.include_fold, expected_rows=len(prediction_rows)
    )
    expected = _expected_identity_projection(
        prediction_rows, include_fold=spec.include_fold, seed=training_seed
    )
    observed_columns = list(expected.columns)
    observed = table.select(observed_columns).to_pandas()
    for column in observed_columns:
        observed[column] = observed[column].astype(expected[column].dtype)
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=True, check_exact=True)
    except AssertionError as exc:
        raise M2ArtifactError("producer output identity/fold membership drift") from exc
    if manifest.get("content_sha256") != validation["logical_table_sha256"]:
        raise M2ArtifactError("producer logical output fingerprint mismatch")
    if manifest.get("model_output_file_sha256") != components[spec.output_filename]:
        raise M2ArtifactError("producer physical output fingerprint mismatch")
    if manifest.get("row_count") != len(table):
        raise M2ArtifactError("producer manifest row count mismatch")
    if manifest.get("plot_count") != int(prediction_rows["plot_idx"].nunique()):
        raise M2ArtifactError("producer manifest plot count mismatch")
    if manifest.get("probability_basis") != "native_t1_uncalibrated":
        raise M2ArtifactError("producer native probability basis mismatch")
    if manifest.get("temperature") != 1.0 or manifest.get("calibrated") is not False:
        raise M2ArtifactError("producer native-T=1 probability declaration mismatch")
    if manifest.get("logits_authoritative") is not True:
        raise M2ArtifactError("producer logits must remain authoritative")
    return {
        "valid": True,
        "status": "reusable",
        "directory": str(target),
        "stage_id": spec.stage_id,
        "training_seed": training_seed,
        "row_count": len(table),
        "manifest_sha256": manifest["manifest_sha256"],
        "content_sha256": manifest["content_sha256"],
        "m2_code_sha256": manifest["m2_code_sha256"],
    }


def _build_producer(
    context: FrozenM2Context,
    spec: ProducerSpec,
    *,
    seed: int,
    staging: Path,
    device: torch.device,
) -> dict[str, Any]:
    fit_rows, prediction_rows = _partitions_for_spec(context.assignments, spec)
    # Only fitting features exist below this boundary.  Prediction image/geo
    # datasets are not constructed until encoder adaptation and all heads end.
    fit_geo_native = _geo_source_projection(context, fit_rows)
    model: nn.Module | None = None
    heads: dict[str, MLPHead] = {}
    try:
        reset_reproducibility(seed)
        model, train_transform, prediction_transform, tokenizer, encoder_provenance = (
            _load_fresh_encoder(context.config, device)
        )
        text_weights = _frozen_text_weights(
            model, tokenizer, context.config, device
        )
        adaptation_history = _adapt_encoder(
            model,
            fit_rows,
            transform=train_transform,
            text_weights=text_weights,
            config=context.config,
            seed=seed,
            device=device,
        )
        fit_image = _extract_embeddings(
            model,
            fit_rows,
            transform=prediction_transform,
            config=context.config,
            seed=seed,
            device=device,
        )
        scaler = fit_geo_standardization(fit_geo_native)
        fit_geo = apply_geo_standardization(fit_geo_native, scaler)
        labels = fit_rows["label_id_dense"].to_numpy(dtype=np.int64, copy=True)
        head_history: dict[str, Any] = {}
        for mode in MODES:
            # _fit_head resets every RNG before constructing its fresh MLP.
            head, history = _fit_head(
                mode,
                _head_input(mode, fit_image, fit_geo),
                labels,
                config=context.config,
                seed=seed,
                device=device,
            )
            heads[mode] = head
            head_history[mode] = history

        # The held-out boundary opens only after all fitting is complete.
        prediction_geo_native = _geo_source_projection(context, prediction_rows)
        prediction_geo = apply_geo_standardization(prediction_geo_native, scaler)
        prediction_image = _extract_embeddings(
            model,
            prediction_rows,
            transform=prediction_transform,
            config=context.config,
            seed=seed,
            device=device,
        )
        logits_by_mode = {
            mode: _predict_head(
                heads[mode],
                _head_input(mode, prediction_image, prediction_geo),
                device,
            )
            for mode in MODES
        }
        table = build_output_table(
            prediction_rows,
            seed=seed,
            logits_by_mode=logits_by_mode,
            include_fold=spec.include_fold,
            schema_version=str(context.config["schema_version"]),
            protocol_id=str(context.config["protocol_id"]),
        )
        write_output_parquet_atomic(table, staging / spec.output_filename)
        _save_safetensors(model.visual, staging / "adapted_visual_tower.safetensors")
        for mode in MODES:
            _save_safetensors(heads[mode], staging / f"{mode}_head.safetensors")
        _write_json_exclusive(
            staging / "geo_standardization.json", scaler.to_json()
        )
        # Release the fitted in-memory models, then prove that the serialized
        # visual tower, heads, and scaler reproduce the exact held-out outputs.
        heads.clear()
        model = None
        del text_weights
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint_reproduction = _replay_saved_checkpoints(
            context,
            spec,
            seed=seed,
            directory=staging,
            prediction_rows=prediction_rows,
            reference=table,
            device=device,
        )
        stage_config = {
            "schema_version": "geo_helpfulness.resolved_m2_stage.v1",
            "protocol_id": context.config["protocol_id"],
            "stage_id": spec.stage_id,
            "training_seed": seed,
            "train_oof_fold": spec.fold,
            "fitting_row_uids_sha256": canonical_sha256(
                fit_rows["row_uid"].astype(str).tolist()
            ),
            "prediction_row_uids_sha256": canonical_sha256(
                prediction_rows["row_uid"].astype(str).tolist()
            ),
            "geo_feature_columns": list(GEO_COLUMNS),
            "mode_order": list(MODES),
            "encoder_initialization": encoder_provenance,
            "adaptation_recipe": context.config["experts"]["image_encoder"][
                "fold_contained_adaptation"
            ]["adaptation_recipe"],
            "head_recipe": context.config["experts"]["head"],
            "expert_epochs": context.config["experts"]["epochs"],
            "m2_code_sha256": context.code_hash,
        }
        _write_yaml_exclusive(staging / "resolved_stage_config.yaml", stage_config)
        metrics = {
            "schema_version": "geo_helpfulness.m2_training_metrics.v1",
            "protocol_id": context.config["protocol_id"],
            "stage_id": spec.stage_id,
            "training_seed": seed,
            "labels_scope": "fitting_partition_only",
            "heldout_metrics": None,
            "adaptation_history": adaptation_history,
            "head_history": head_history,
            "fitting_class_plot_support_by_dense_id": _class_plot_support(fit_rows),
            "checkpoint_reproduction": checkpoint_reproduction,
        }
        _write_json_exclusive(staging / "training_metrics.json", metrics)
        runtime = _runtime_provenance(device)
        manifest = _producer_manifest(
            context,
            spec,
            seed=seed,
            fit_rows=fit_rows,
            prediction_rows=prediction_rows,
            table=table,
            staging=staging,
            encoder_provenance=encoder_provenance,
            runtime=runtime,
            checkpoint_reproduction=checkpoint_reproduction,
        )
        _write_json_exclusive(staging / "manifest.json", manifest)
        return validate_producer(
            context, spec, seed=seed, directory=staging
        )
    finally:
        heads.clear()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _discard_owned_staging(parent: Path, final_name: str) -> None:
    if not parent.exists():
        return
    prefix = f".{final_name}.staging-"
    for candidate in parent.iterdir():
        if candidate.name.startswith(prefix) and candidate.is_dir():
            shutil.rmtree(candidate)


@contextmanager
def _exclusive_workflow_lock(path: Path):
    """Hold a non-blocking process lock; kernel release makes crashes non-stale."""

    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise M2ArtifactError(f"another M2 process owns workflow lock: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _run_or_resume_producer(
    context: FrozenM2Context,
    spec: ProducerSpec,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    final = context.output_root / spec.relative_directory
    final.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final.parent / f".{final.name}.m2.lock"
    with _exclusive_workflow_lock(lock_path):
        if final.exists():
            result = validate_producer(context, spec, seed=seed, directory=final)
            result["status"] = "skipped_valid"
            return result
        _discard_owned_staging(final.parent, final.name)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent)
        ).resolve()
        try:
            result = _build_producer(
                context, spec, seed=seed, staging=staging, device=device
            )
            if final.exists():
                raise M2ArtifactError(f"producer appeared during atomic publication: {final}")
            os.replace(staging, final)
            result = validate_producer(context, spec, seed=seed, directory=final)
            result["status"] = "created"
            return result
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def run_seed(
    seed: Any,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the four OOF stages then train-to-validation for one frozen seed."""

    training_seed = validate_training_seed(seed)
    # This preflight occurs before CUDA model loading and before any M2 writes.
    context = load_frozen_context(
        config_path=config_path,
        protocol_dir=protocol_dir,
        output_root=output_root,
    )
    if not torch.cuda.is_available():
        raise M2Error("run-seed requires CUDA; select the GPU with CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", torch.cuda.current_device())
    results = [
        _run_or_resume_producer(
            context, spec, seed=training_seed, device=device
        )
        for spec in _producer_specs(training_seed)
    ]
    return {
        "status": "complete",
        "protocol_id": context.config["protocol_id"],
        "training_seed": training_seed,
        "output_root": str(context.output_root),
        "producer_count": len(results),
        "producers": results,
        "next": (
            "Run the remaining frozen seeds, then invoke aggregate."
        ),
    }


def validate_reproduced_output(
    reference: pa.Table,
    reproduced: pa.Table,
    *,
    include_fold: bool,
) -> dict[str, Any]:
    """Check a checkpoint replay against a serialized producer output."""

    validate_output_table(reference, include_fold=include_fold)
    validate_output_table(reproduced, include_fold=include_fold)
    if reference.column_names != reproduced.column_names or len(reference) != len(reproduced):
        raise M2ArtifactError("reproduced output shape/schema mismatch")
    exact = [
        "schema_version",
        "protocol_id",
        "row_uid",
        "file",
        "file_lower",
        "plot_idx",
        *( ["train_oof_fold"] if include_fold else [] ),
        "training_seed",
        *PREDICTION_COLUMNS,
    ]
    for column in exact:
        if reference[column].to_pylist() != reproduced[column].to_pylist():
            raise M2ArtifactError(f"reproduced output differs exactly in {column}")
    for column in VECTOR_COLUMNS:
        left = _matrix_from_arrow(reference, column)
        right = _matrix_from_arrow(reproduced, column)
        if not np.allclose(
            left,
            right,
            atol=REPRODUCTION_ATOL,
            rtol=REPRODUCTION_RTOL,
        ):
            raise M2ArtifactError(f"reproduced output differs numerically in {column}")
    return {
        "valid": True,
        "row_count": len(reference),
        "atol": REPRODUCTION_ATOL,
        "rtol": REPRODUCTION_RTOL,
    }


def _all_validated_producers(
    context: FrozenM2Context,
) -> tuple[list[pa.Table], list[pa.Table], dict[str, str]]:
    oof_tables: list[pa.Table] = []
    validation_tables: list[pa.Table] = []
    producer_manifests: dict[str, str] = {}
    for seed in TRAINING_SEEDS:
        for spec in _producer_specs(seed):
            result = validate_producer(context, spec, seed=seed)
            producer_manifests[spec.relative_directory.as_posix()] = str(
                result["manifest_sha256"]
            )
            path = context.output_root / spec.relative_directory / spec.output_filename
            table = read_output_parquet(path, include_fold=spec.include_fold)
            if spec.include_fold:
                oof_tables.append(table)
            else:
                validation_tables.append(table)
    return oof_tables, validation_tables, producer_manifests


def _aggregate_table(
    tables: Sequence[pa.Table],
    *,
    include_fold: bool,
    expected_assignments: pd.DataFrame,
) -> pa.Table:
    if not tables:
        raise M2ArtifactError("aggregate has no producer tables")
    try:
        combined = pa.concat_tables(list(tables), promote_options="none")
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise M2ArtifactError("producer output Arrow schemas cannot be concatenated") from exc
    combined = combined.sort_by([("row_uid", "ascending"), ("training_seed", "ascending")])
    expected_count = len(expected_assignments) * len(TRAINING_SEEDS)
    validate_output_table(
        combined, include_fold=include_fold, expected_rows=expected_count
    )
    expected_frames = [
        _expected_identity_projection(
            expected_assignments,
            include_fold=include_fold,
            seed=seed,
        )
        for seed in TRAINING_SEEDS
    ]
    expected = (
        pd.concat(expected_frames, ignore_index=True)
        .sort_values(["row_uid", "training_seed"], kind="mergesort")
        .reset_index(drop=True)
    )
    columns = list(expected.columns)
    observed = combined.select(columns).to_pandas()
    for column in columns:
        observed[column] = observed[column].astype(expected[column].dtype)
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=True, check_exact=True)
    except AssertionError as exc:
        raise M2ArtifactError("aggregate identity, seed, or fold membership drift") from exc
    return combined


def _aggregate_manifest(
    context: FrozenM2Context,
    *,
    table: pa.Table,
    output_path: Path,
    include_fold: bool,
    producer_manifest_hashes: Mapping[str, str],
    report_path: Path | None = None,
) -> dict[str, Any]:
    role = (
        "development_train_oof_outputs"
        if include_fold
        else "development_validation_outputs"
    )
    parent_roles = (
        ["development_assignments", "train_oof_fold_outputs"]
        if include_fold
        else ["development_assignments", "development_train_expert_fit"]
    )
    validate_artifact_parent_roles(role, parent_roles)
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "protocol_id": context.config["protocol_id"],
        "artifact_role": role,
        "parent_roles": parent_roles,
        "parent_artifact_hashes": context.parent_hashes,
        "producer_manifest_sha256": dict(sorted(producer_manifest_hashes.items())),
        "training_seeds": list(TRAINING_SEEDS),
        "row_count": len(table),
        "unique_key": ["row_uid", "training_seed"],
        "canonical_sort": ["row_uid", "training_seed"],
        "output_columns": table.column_names,
        "feature_schema_hash": _feature_schema_hash(include_fold),
        "content_sha256": logical_table_sha256(table),
        "model_output_filename": output_path.name,
        "model_output_file_sha256": sha256_file(output_path),
        "m2_code_file_sha256": context.code_file_hashes,
        "m2_code_sha256": context.code_hash,
        "aggregation": "concatenation_without_averaging_or_pooling",
        "probability_basis": "native_t1_uncalibrated",
        "logits_authoritative": True,
    }
    if report_path is not None:
        manifest["oof_reproduction_report_filename"] = report_path.name
        manifest["oof_reproduction_report_file_sha256"] = sha256_file(report_path)
    manifest["manifest_sha256"] = _manifest_self_hash(manifest)
    return manifest


def _aggregate_paths(context: FrozenM2Context, *, include_fold: bool) -> tuple[Path, Path, Path | None]:
    if include_fold:
        root = context.output_root / "development_train_oof"
        return (
            root / "development_train_oof_model_outputs.parquet",
            root / "aggregate_manifest.json",
            root / "oof_reproduction_report.json",
        )
    root = context.output_root / "development_validation"
    return (
        root / "development_validation_model_outputs.parquet",
        root / "aggregate_manifest.json",
        None,
    )


def _validate_aggregate_artifact(
    context: FrozenM2Context,
    *,
    include_fold: bool,
    producer_manifest_hashes: Mapping[str, str],
    expected_table: pa.Table,
) -> dict[str, Any]:
    output_path, manifest_path, report_path = _aggregate_paths(
        context, include_fold=include_fold
    )
    required = [output_path, manifest_path]
    if report_path is not None:
        required.append(report_path)
    present = [path.exists() for path in required]
    if not all(present):
        if any(present):
            raise M2ArtifactError(
                f"published aggregate is incomplete and cannot be overwritten: {required}"
            )
        raise FileNotFoundError(str(output_path))
    manifest = _read_json_mapping(manifest_path, name="aggregate manifest")
    if manifest.get("manifest_sha256") != _manifest_self_hash(manifest):
        raise M2ArtifactError("aggregate manifest self-hash mismatch")
    role = (
        "development_train_oof_outputs"
        if include_fold
        else "development_validation_outputs"
    )
    expected_parent_roles = (
        ["development_assignments", "train_oof_fold_outputs"]
        if include_fold
        else ["development_assignments", "development_train_expert_fit"]
    )
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise M2ArtifactError("aggregate manifest schema mismatch")
    if manifest.get("artifact_role") != role:
        raise M2ArtifactError("aggregate artifact role mismatch")
    if manifest.get("parent_roles") != expected_parent_roles:
        raise M2ArtifactError("aggregate parent-role declaration mismatch")
    validate_artifact_parent_roles(role, expected_parent_roles)
    if manifest.get("protocol_id") != context.config["protocol_id"]:
        raise M2ArtifactError("aggregate protocol ID mismatch")
    if manifest.get("parent_artifact_hashes") != context.parent_hashes:
        raise M2ArtifactError("aggregate M1 parent fingerprints are stale")
    if manifest.get("m2_code_sha256") != context.code_hash:
        raise M2ArtifactError("aggregate M2 implementation fingerprint is stale")
    if manifest.get("m2_code_file_sha256") != context.code_file_hashes:
        raise M2ArtifactError("aggregate M2 implementation file fingerprints are stale")
    expected_declarations = {
        "training_seeds": list(TRAINING_SEEDS),
        "unique_key": ["row_uid", "training_seed"],
        "canonical_sort": ["row_uid", "training_seed"],
        "output_columns": list(
            OOF_OUTPUT_COLUMNS if include_fold else VALIDATION_OUTPUT_COLUMNS
        ),
        "feature_schema_hash": _feature_schema_hash(include_fold),
        "model_output_filename": output_path.name,
        "aggregation": "concatenation_without_averaging_or_pooling",
        "probability_basis": "native_t1_uncalibrated",
        "logits_authoritative": True,
    }
    for key, expected in expected_declarations.items():
        if manifest.get(key) != expected:
            raise M2ArtifactError(f"aggregate semantic declaration mismatch: {key}")
    relevant = {
        key: value
        for key, value in producer_manifest_hashes.items()
        if (key.startswith("development_train_oof/") if include_fold else key.startswith("development_validation/"))
    }
    if manifest.get("producer_manifest_sha256") != dict(sorted(relevant.items())):
        raise M2ArtifactError("aggregate producer manifest set is stale or mixed")
    table = read_output_parquet(output_path, include_fold=include_fold)
    expected_rows = 13_512 if include_fold else 3_288
    validate_output_table(table, include_fold=include_fold, expected_rows=expected_rows)
    if logical_table_sha256(table) != logical_table_sha256(expected_table):
        raise M2ArtifactError("published aggregate differs from the validated producer concatenation")
    if manifest.get("row_count") != expected_rows:
        raise M2ArtifactError("aggregate manifest row count mismatch")
    if manifest.get("content_sha256") != logical_table_sha256(table):
        raise M2ArtifactError("aggregate logical table fingerprint mismatch")
    if manifest.get("model_output_file_sha256") != sha256_file(output_path):
        raise M2ArtifactError("aggregate physical file fingerprint mismatch")
    if report_path is not None:
        if manifest.get("oof_reproduction_report_filename") != report_path.name:
            raise M2ArtifactError("OOF reproduction report filename declaration mismatch")
        if manifest.get("oof_reproduction_report_file_sha256") != sha256_file(report_path):
            raise M2ArtifactError("OOF reproduction report fingerprint mismatch")
        report = _read_json_mapping(report_path, name="OOF reproduction report")
        expected_report = _build_oof_report_payload(context, expected_table)
        if report != expected_report:
            raise M2ArtifactError("OOF reproduction/performance report does not reproduce")
    return {
        "valid": True,
        "status": "reused_valid",
        "output": str(output_path),
        "manifest": str(manifest_path),
        "row_count": len(table),
        "content_sha256": manifest["content_sha256"],
        **({"report": str(report_path)} if report_path is not None else {}),
    }


def _build_oof_report_payload(
    context: FrozenM2Context, table: pa.Table
) -> dict[str, Any]:
    from multimodal.geo_helpfulness_oof_report import build_oof_reproduction_report

    report = build_oof_reproduction_report(
        table,
        context.assignments,
        dense_class_count=N_CLASSES,
        modes=("image", "geo", "raw"),
    )
    return {
        "schema_version": "geo_helpfulness.oof_reproduction_report.v1",
        "protocol_id": context.config["protocol_id"],
        "assignment_content_sha256": assignment_fingerprint(context.assignments),
        "oof_content_sha256": logical_table_sha256(table),
        **report,
    }


def _discard_uncommitted_aggregate(
    output_path: Path,
    manifest_path: Path,
    report_path: Path | None,
) -> None:
    """Remove only exact M2 aggregate payloads that have no commit manifest."""

    if manifest_path.exists():
        return
    output_path.unlink(missing_ok=True)
    if report_path is not None:
        report_path.unlink(missing_ok=True)


def _write_new_aggregate(
    context: FrozenM2Context,
    *,
    table: pa.Table,
    include_fold: bool,
    producer_manifest_hashes: Mapping[str, str],
) -> dict[str, Any]:
    output_path, manifest_path, report_path = _aggregate_paths(
        context, include_fold=include_fold
    )
    relevant = {
        key: value
        for key, value in producer_manifest_hashes.items()
        if (key.startswith("development_train_oof/") if include_fold else key.startswith("development_validation/"))
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".aggregate.staging-", dir=output_path.parent)
    ).resolve()
    staged_output = staging / output_path.name
    staged_report = staging / report_path.name if report_path is not None else None
    staged_manifest = staging / manifest_path.name
    try:
        write_output_parquet_atomic(table, staged_output)
        if staged_report is not None:
            _write_json_exclusive(staged_report, _build_oof_report_payload(context, table))
        manifest = _aggregate_manifest(
            context,
            table=table,
            output_path=staged_output,
            include_fold=include_fold,
            producer_manifest_hashes=relevant,
            report_path=staged_report,
        )
        _write_json_exclusive(staged_manifest, manifest)
        # The manifest is the commit marker and is always published last.
        _exclusive_atomic_replace(staged_output, output_path)
        if staged_report is not None and report_path is not None:
            _exclusive_atomic_replace(staged_report, report_path)
        _exclusive_atomic_replace(staged_manifest, manifest_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    result = _validate_aggregate_artifact(
        context,
        include_fold=include_fold,
        producer_manifest_hashes=producer_manifest_hashes,
        expected_table=table,
    )
    result["status"] = "created"
    return result


def aggregate(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all 20 producers and concatenate the four seeds without averaging."""

    context = load_frozen_context(
        config_path=config_path,
        protocol_dir=protocol_dir,
        output_root=output_root,
    )
    oof_tables, validation_tables, producer_hashes = _all_validated_producers(context)
    train = context.assignments.loc[
        context.assignments["development_role"].astype(str).eq("train")
    ].copy()
    validation = context.assignments.loc[
        context.assignments["development_role"].astype(str).eq("validation")
    ].copy()
    if len(train) != 3_378 or len(validation) != 822:
        raise M2ArtifactError(
            f"sealed assignment row counts changed: train={len(train)}, validation={len(validation)}"
        )
    oof = _aggregate_table(
        oof_tables,
        include_fold=True,
        expected_assignments=train,
    )
    validation_output = _aggregate_table(
        validation_tables,
        include_fold=False,
        expected_assignments=validation,
    )
    if len(oof) != 13_512 or len(validation_output) != 3_288:
        raise M2ArtifactError("aggregate acceptance counts are not 13,512 and 3,288")

    results: dict[str, Any] = {}
    with _exclusive_workflow_lock(context.output_root / ".m2_aggregate.lock"):
        for name, table, include_fold in (
            ("development_train_oof", oof, True),
            ("development_validation", validation_output, False),
        ):
            output_path, manifest_path, report_path = _aggregate_paths(
                context, include_fold=include_fold
            )
            if manifest_path.exists():
                results[name] = _validate_aggregate_artifact(
                    context,
                    include_fold=include_fold,
                    producer_manifest_hashes=producer_hashes,
                    expected_table=table,
                )
            else:
                _discard_uncommitted_aggregate(output_path, manifest_path, report_path)
                results[name] = _write_new_aggregate(
                    context,
                    table=table,
                    include_fold=include_fold,
                    producer_manifest_hashes=producer_hashes,
                )
    return {
        "status": "complete",
        "protocol_id": context.config["protocol_id"],
        "output_root": str(context.output_root),
        "aggregation": "concatenation_without_averaging_or_pooling",
        "development_train_oof": results["development_train_oof"],
        "development_validation": results["development_validation"],
        "m2_execution_status": "complete",
    }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "GEO_COLUMNS",
    "M2ArtifactError",
    "M2Error",
    "MODES",
    "N_CLASSES",
    "OOF_OUTPUT_COLUMNS",
    "TRAINING_SEEDS",
    "VALIDATION_OUTPUT_COLUMNS",
    "aggregate",
    "apply_geo_standardization",
    "build_output_table",
    "fit_geo_standardization",
    "load_frozen_context",
    "producer_partitions",
    "read_output_parquet",
    "run_seed",
    "stable_softmax_float64",
    "validate_m1_preflight",
    "validate_output_table",
    "validate_producer",
    "validate_reproduced_output",
    "validate_training_seed",
    "write_output_parquet_atomic",
]

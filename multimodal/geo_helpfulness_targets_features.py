"""M3 targets and deployment-safe router feature contract.

M3 is an additive child of the immutable M1/M2 geo-helpfulness workflow.  The
pure functions in this module derive seed-specific router targets and build the
semantic feature frame used unchanged by later router fitting and inference.
The artifact workflow deliberately opens only sealed development-train OOF
outputs and development assignments; calibration and router fitting belong to
M4.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal.geo_helpfulness_protocol import (
    DENSE_TO_CANONICAL_L3,
    DENSE_TO_LABEL_NAME,
    assignment_fingerprint,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "multimodal_geo_helpfulness.yaml"
M3_RUNNER_PATH = PROJECT_ROOT / "tools" / "run_multimodal_geo_helpfulness_m3.py"

N_CLASSES = 18
TRAINING_SEEDS = (1, 2, 3, 4)
OOF_FOLDS = (0, 1, 2, 3)
PROBABILITY_ATOL = 1.0e-8
CALIBRATED_PROBABILITY_BASIS = "scalar_temperature_calibrated"
NATIVE_T1_PROBABILITY_BASIS = "native_t1_uncalibrated"

TARGET_SCHEMA_VERSION = "geo_helpfulness.router_targets.v1"
FEATURE_SCHEMA_VERSION = "geo_helpfulness.router_feature_schema.v1"
PREVALENCE_SCHEMA_VERSION = "geo_helpfulness.target_prevalence.v1"
LEAKAGE_AUDIT_SCHEMA_VERSION = "geo_helpfulness.feature_leakage_audit.v1"
M3_MANIFEST_SCHEMA_VERSION = "geo_helpfulness.m3_manifest.v1"
OWNERSHIP_RECEIPT_SCHEMA_VERSION = "geo_helpfulness.m3_ownership_receipt.v1"

TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
IMAGE_RELATIVE_TARGET_ORDER = (
    "geo_only_correct",
    "image_only_correct",
    "both_correct",
    "both_wrong",
)
TARGET_COLUMNS = (
    "schema_version",
    "protocol_id",
    "row_uid",
    "plot_idx",
    "training_seed",
    "target_state",
)
TARGET_KEY = ("row_uid", "training_seed")

CATEGORICAL_FEATURES = (
    "image_pred",
    "geo_pred",
    "raw_pred",
    "image_geo_pred_pair",
    "geo_raw_pred_pair",
)
BOOLEAN_FEATURES = (
    "image_geo_agree",
    "image_raw_agree",
    "geo_raw_agree",
)
INTEGER_FEATURES = (
    "image_geo_top3_overlap",
    "raw_rank_of_geo_pred",
)
NUMERIC_FEATURES = (
    "image_confidence",
    "geo_confidence",
    "raw_confidence",
    "image_entropy",
    "geo_entropy",
    "raw_entropy",
    "image_top2_margin",
    "geo_top2_margin",
    "raw_top2_margin",
    "geo_minus_image_confidence",
    "geo_minus_raw_confidence",
    "geo_minus_image_entropy",
    "geo_minus_raw_entropy",
    "geo_minus_image_margin",
    "geo_minus_raw_margin",
    "image_geo_jsd",
    "image_geo_total_variation",
    "image_probability_at_geo_pred",
    "geo_probability_at_image_pred",
    "raw_probability_at_geo_pred",
)
FEATURE_FAMILIES = {
    "categorical": CATEGORICAL_FEATURES,
    "boolean": BOOLEAN_FEATURES,
    "integer": INTEGER_FEATURES,
    "numeric": NUMERIC_FEATURES,
}
FEATURE_COLUMNS = tuple(
    column
    for family in ("categorical", "boolean", "integer", "numeric")
    for column in FEATURE_FAMILIES[family]
)

BUNDLE_RELATIVE_PATH = Path("router") / "targets_and_feature_contract"
TARGET_FILENAME = "router_targets.parquet"
FEATURE_SCHEMA_FILENAME = "router_feature_schema.json"
PREVALENCE_FILENAME = "target_prevalence.json"
LEAKAGE_AUDIT_FILENAME = "feature_leakage_audit.json"
MANIFEST_FILENAME = "manifest.json"
OWNERSHIP_RECEIPT_FILENAME = ".targets_and_feature_contract.m3.ownership.json"
BUNDLE_CHILD_FILENAMES = (
    TARGET_FILENAME,
    FEATURE_SCHEMA_FILENAME,
    PREVALENCE_FILENAME,
    LEAKAGE_AUDIT_FILENAME,
)
BUNDLE_FILENAMES = BUNDLE_CHILD_FILENAMES + (MANIFEST_FILENAME,)


class M3Error(ValueError):
    """Base error for an M3 contract violation."""


class M3ArtifactError(M3Error):
    """A required M1/M2/M3 artifact is absent, stale, or corrupt."""


@dataclass(frozen=True)
class ValidatedM3Bundle:
    """Loaded, lineage-validated M3 artifacts for M4 consumers."""

    root: Path
    targets: pd.DataFrame
    feature_schema: dict[str, Any]
    target_prevalence: dict[str, Any]
    feature_leakage_audit: dict[str, Any]
    manifest: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class _PreparedM3:
    context: Any
    oof_table: pa.Table
    producer_manifest_hashes: dict[str, str]
    producer_manifest_file_hashes: dict[str, str]
    aggregate_manifest: dict[str, Any]
    aggregate_validation: dict[str, Any]
    targets: pd.DataFrame
    feature_schema: dict[str, Any]
    prevalence: dict[str, Any]
    leakage_audit: dict[str, Any]


def _as_dataframe(value: Any, *, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=False)
    if isinstance(value, pa.Table):
        return value.to_pandas()
    to_pandas = getattr(value, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        if isinstance(frame, pd.DataFrame):
            return frame
    raise M3Error(f"{name} must be a pandas DataFrame or PyArrow-like table")


def _integer_vector(
    values: Any,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise M3Error(f"{name} must be one-dimensional")
    if array.dtype.kind == "b":
        raise M3Error(f"{name} must contain integers, not booleans")
    if array.dtype.kind in {"i", "u"}:
        integer = array.astype(np.int64, copy=False)
    elif array.dtype.kind == "O" and all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        for value in array.tolist()
    ):
        integer = array.astype(np.int64)
    else:
        raise M3Error(f"{name} must have an integer dtype")
    if minimum is not None and bool((integer < minimum).any()):
        raise M3Error(f"{name} contains a value below {minimum}")
    if maximum is not None and bool((integer > maximum).any()):
        raise M3Error(f"{name} contains a value above {maximum}")
    return integer


def _prediction_vectors(
    left: Any,
    right: Any,
    labels: Any,
    *,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_array = _integer_vector(
        left, name=left_name, minimum=0, maximum=N_CLASSES - 1
    )
    right_array = _integer_vector(
        right, name=right_name, minimum=0, maximum=N_CLASSES - 1
    )
    label_array = _integer_vector(
        labels, name="label_id_dense", minimum=0, maximum=N_CLASSES - 1
    )
    if not (len(left_array) == len(right_array) == len(label_array)):
        raise M3Error("predictions and labels must have identical lengths")
    if len(label_array) < 1:
        raise M3Error("predictions and labels must not be empty")
    return left_array, right_array, label_array


def _canonical_expected_seeds(expected_seeds: Sequence[int]) -> tuple[int, ...]:
    if expected_seeds is None:
        raise M3Error(
            "expected_seeds must be explicit; use TRAINING_SEEDS for a combined table "
            "or a canonical seed subset for a scoped projection"
        )
    if isinstance(expected_seeds, (str, bytes)):
        raise M3Error("expected_seeds must be an integer sequence, not a string")
    values = tuple(expected_seeds)
    if not values or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise M3Error("expected_seeds must contain one or more integer seed IDs")
    canonical = tuple(int(value) for value in values)
    if canonical != tuple(sorted(set(canonical))):
        raise M3Error("expected_seeds must be sorted and contain no duplicates")
    invalid = sorted(set(canonical).difference(TRAINING_SEEDS))
    if invalid:
        raise M3Error(f"expected_seeds contains invalid training seeds: {invalid}")
    return canonical


def _relative_target_states(
    baseline_pred: Any,
    alternative_pred: Any,
    label_id_dense: Any,
    *,
    baseline_name: str,
    alternative_name: str,
) -> np.ndarray:
    baseline, alternative, labels = _prediction_vectors(
        baseline_pred,
        alternative_pred,
        label_id_dense,
        left_name=baseline_name,
        right_name=alternative_name,
    )
    baseline_correct = baseline == labels
    alternative_correct = alternative == labels
    states = np.full(len(labels), "both_wrong", dtype=object)
    states[np.logical_and(~baseline_correct, alternative_correct)] = "rescue"
    states[np.logical_and(baseline_correct, ~alternative_correct)] = "harm"
    states[np.logical_and(baseline_correct, alternative_correct)] = "both_correct"
    return states


def derive_router_target_states(
    raw_pred: Any,
    geo_pred: Any,
    label_id_dense: Any,
) -> np.ndarray:
    """Derive exhaustive geo-versus-raw target states for each seed record."""

    return _relative_target_states(
        raw_pred,
        geo_pred,
        label_id_dense,
        baseline_name="raw_pred",
        alternative_name="geo_pred",
    )


def derive_image_relative_states(
    image_pred: Any,
    geo_pred: Any,
    label_id_dense: Any,
) -> np.ndarray:
    """Derive report-only geo-versus-image diagnostic states."""

    states = _relative_target_states(
        image_pred,
        geo_pred,
        label_id_dense,
        baseline_name="image_pred",
        alternative_name="geo_pred",
    )
    return np.asarray(
        [
            {
                "rescue": "geo_only_correct",
                "harm": "image_only_correct",
                "both_correct": "both_correct",
                "both_wrong": "both_wrong",
            }[str(state)]
            for state in states
        ],
        dtype=object,
    )


def _validate_uid_series(values: pd.Series, *, name: str) -> pd.Series:
    if bool(values.isna().any()):
        raise M3Error(f"{name} contains null row_uid values")
    if not bool(values.map(lambda value: isinstance(value, str)).all()):
        raise M3Error(f"{name} row_uid values must be strings")
    valid = values.map(lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value)))
    if not bool(valid.all()):
        raise M3Error(f"{name} contains a non-canonical row_uid")
    return values.astype("string")


def _validate_canonical_text(values: pd.Series, *, name: str) -> pd.Series:
    if bool(values.isna().any()):
        raise M3Error(f"{name} contains null values")
    if not bool(values.map(lambda value: isinstance(value, str)).all()):
        raise M3Error(f"{name} values must be strings")
    if bool(values.map(lambda value: not value or value != value.strip()).any()):
        raise M3Error(f"{name} contains empty or non-canonical values")
    return values.astype("string")


def _prepared_target_rows(
    oof_predictions: Any,
    assignments: Any,
    *,
    protocol_id: str | None,
    expected_seeds: Sequence[int],
) -> tuple[pd.DataFrame, str, tuple[int, ...]]:
    oof = _as_dataframe(oof_predictions, name="OOF predictions").copy()
    assignment_frame = _as_dataframe(assignments, name="development assignments").copy()
    required_oof = {
        "schema_version",
        "protocol_id",
        "row_uid",
        "file",
        "file_lower",
        "plot_idx",
        "train_oof_fold",
        "training_seed",
        "image_pred",
        "geo_pred",
        "raw_pred",
    }
    required_assignments = {
        "protocol_id",
        "row_uid",
        "file",
        "file_lower",
        "plot_idx",
        "label_id_dense",
        "development_role",
        "train_oof_fold",
    }
    missing_oof = sorted(required_oof.difference(oof.columns))
    missing_assignments = sorted(required_assignments.difference(assignment_frame.columns))
    if missing_oof:
        raise M3Error(f"OOF predictions are missing required columns: {missing_oof}")
    if missing_assignments:
        raise M3Error(
            f"development assignments are missing required columns: {missing_assignments}"
        )
    forbidden_oof = sorted(
        column
        for column in oof.columns
        if column == "label"
        or column == "label_id_dense"
        or column.startswith("label_")
        or column.endswith("_correct")
        or "target" in column.casefold()
    )
    if forbidden_oof:
        raise M3Error(
            "OOF predictions must remain label-blind; forbidden columns: "
            f"{forbidden_oof}"
        )
    if oof.empty:
        raise M3Error("OOF predictions must not be empty")

    oof_schema_versions = set(oof["schema_version"].astype(str))
    if len(oof_schema_versions) != 1 or not next(iter(oof_schema_versions)).strip():
        raise M3Error("OOF predictions must contain one non-empty schema_version")

    for frame, name in ((oof, "OOF predictions"), (assignment_frame, "assignments")):
        frame["row_uid"] = _validate_uid_series(frame["row_uid"], name=name)
        for column in ("protocol_id", "file", "file_lower", "plot_idx"):
            frame[column] = _validate_canonical_text(
                frame[column], name=f"{name} {column}"
            )
    if bool(assignment_frame["row_uid"].duplicated().any()):
        raise M3Error("development assignments contain duplicate row_uid values")
    roles = set(assignment_frame["development_role"].astype(str))
    if not roles.issubset({"train", "validation"}):
        raise M3Error(f"assignments contain unknown development roles: {sorted(roles)}")
    train = assignment_frame.loc[
        assignment_frame["development_role"].astype(str).eq("train")
    ].copy()
    if train.empty:
        raise M3Error("development assignments contain no development-train rows")
    if bool(train["train_oof_fold"].isna().any()):
        raise M3Error("development-train assignments contain null OOF folds")
    train["train_oof_fold"] = _integer_vector(
        train["train_oof_fold"],
        name="assignment train_oof_fold",
        minimum=min(OOF_FOLDS),
        maximum=max(OOF_FOLDS),
    )
    train["label_id_dense"] = _integer_vector(
        train["label_id_dense"],
        name="assignment label_id_dense",
        minimum=0,
        maximum=N_CLASSES - 1,
    )
    expected_label_names = train["label_id_dense"].map(DENSE_TO_LABEL_NAME).astype("string")
    if "label_name" in train.columns:
        observed_label_names = _validate_canonical_text(
            train["label_name"], name="assignment label_name"
        )
        if not bool(observed_label_names.eq(expected_label_names).all()):
            raise M3Error("assignment label_name does not match the frozen dense ontology")
    train["label_name"] = expected_label_names

    oof["training_seed"] = _integer_vector(
        oof["training_seed"], name="OOF training_seed"
    )
    oof["train_oof_fold"] = _integer_vector(
        oof["train_oof_fold"],
        name="OOF train_oof_fold",
        minimum=min(OOF_FOLDS),
        maximum=max(OOF_FOLDS),
    )
    for column in ("image_pred", "geo_pred", "raw_pred"):
        oof[column] = _integer_vector(
            oof[column], name=f"OOF {column}", minimum=0, maximum=N_CLASSES - 1
        )
    if bool(oof[list(TARGET_KEY)].duplicated().any()):
        raise M3Error("OOF predictions contain duplicate (row_uid, training_seed) records")

    oof_protocols = set(oof["protocol_id"].astype(str))
    assignment_protocols = set(assignment_frame["protocol_id"].astype(str))
    if len(oof_protocols) != 1 or len(assignment_protocols) != 1:
        raise M3Error("OOF predictions and assignments must each use one protocol_id")
    observed_protocol = next(iter(oof_protocols))
    if assignment_protocols != {observed_protocol}:
        raise M3Error("OOF predictions and assignments use different protocol IDs")
    if protocol_id is not None and observed_protocol != str(protocol_id):
        raise M3Error(
            f"unexpected protocol ID {observed_protocol!r}; expected {str(protocol_id)!r}"
        )

    observed_seeds = tuple(sorted(int(value) for value in oof["training_seed"].unique()))
    invalid_seeds = sorted(set(observed_seeds).difference(TRAINING_SEEDS))
    if invalid_seeds:
        raise M3Error(f"OOF predictions contain invalid training seeds: {invalid_seeds}")
    requested = _canonical_expected_seeds(expected_seeds)
    if observed_seeds != requested:
        raise M3Error(
            f"OOF training seeds {observed_seeds} do not match expected {requested}"
        )
    if not observed_seeds:
        raise M3Error("OOF predictions contain no training seeds")
    expected_uids = set(train["row_uid"].astype(str))
    for seed in observed_seeds:
        seed_uids = set(
            oof.loc[oof["training_seed"].eq(seed), "row_uid"].astype(str)
        )
        if seed_uids != expected_uids:
            missing = sorted(expected_uids.difference(seed_uids))[:5]
            extra = sorted(seed_uids.difference(expected_uids))[:5]
            raise M3Error(
                f"training seed {seed} does not exactly cover development-train rows; "
                f"missing={missing}, extra={extra}"
            )

    assignment_projection = train.loc[
        :,
        [
            "row_uid",
            "file",
            "file_lower",
            "plot_idx",
            "train_oof_fold",
            "label_id_dense",
            "label_name",
        ],
    ].rename(
        columns={
            "file": "assignment_file",
            "file_lower": "assignment_file_lower",
            "plot_idx": "assignment_plot_idx",
            "train_oof_fold": "assignment_train_oof_fold",
        }
    )
    joined = oof.merge(
        assignment_projection,
        on="row_uid",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if bool(joined["label_id_dense"].isna().any()):
        raise M3Error("an OOF row has no development-train assignment label")
    comparisons = (
        ("file", "assignment_file"),
        ("file_lower", "assignment_file_lower"),
        ("plot_idx", "assignment_plot_idx"),
        ("train_oof_fold", "assignment_train_oof_fold"),
    )
    for observed, expected in comparisons:
        if not bool(joined[observed].eq(joined[expected]).all()):
            raise M3Error(f"OOF {observed} does not match sealed assignment identity")
    joined["target_state"] = derive_router_target_states(
        joined["raw_pred"], joined["geo_pred"], joined["label_id_dense"]
    )
    joined["image_relative_state"] = derive_image_relative_states(
        joined["image_pred"], joined["geo_pred"], joined["label_id_dense"]
    )
    joined = joined.sort_values(list(TARGET_KEY), kind="mergesort").reset_index(drop=True)
    return joined, observed_protocol, observed_seeds


def build_router_target_table(
    oof_predictions: Any,
    assignments: Any,
    *,
    protocol_id: str | None = None,
    expected_seeds: Sequence[int] = TRAINING_SEEDS,
) -> pd.DataFrame:
    """Build the minimal seed-specific router target table.

    The returned table is sorted by ``(row_uid, training_seed)``.  It contains
    no labels, correctness flags, predictions, logits, or probabilities.
    """

    joined, observed_protocol, observed_seeds = _prepared_target_rows(
        oof_predictions,
        assignments,
        protocol_id=protocol_id,
        expected_seeds=expected_seeds,
    )
    result = pd.DataFrame(
        {
            "schema_version": pd.Series(
                [TARGET_SCHEMA_VERSION] * len(joined), dtype="string"
            ),
            "protocol_id": pd.Series(
                [observed_protocol] * len(joined), dtype="string"
            ),
            "row_uid": joined["row_uid"].astype("string").reset_index(drop=True),
            "plot_idx": joined["plot_idx"].astype("string").reset_index(drop=True),
            "training_seed": joined["training_seed"].to_numpy(dtype=np.int8),
            "target_state": pd.Series(joined["target_state"], dtype="string"),
        },
        columns=list(TARGET_COLUMNS),
    )
    validate_router_target_table(
        result,
        protocol_id=observed_protocol,
        expected_rows=len(joined),
        expected_seeds=observed_seeds,
    )
    return result


def validate_router_target_table(
    table_or_dataframe: Any,
    *,
    protocol_id: str | None = None,
    expected_rows: int | None = None,
    expected_seeds: Sequence[int] = TRAINING_SEEDS,
) -> dict[str, Any]:
    """Validate the physical/logical M3 target-table contract."""

    frame = _as_dataframe(table_or_dataframe, name="router target table")
    if tuple(frame.columns) != TARGET_COLUMNS:
        raise M3Error(
            "router target column allow-list/order mismatch: "
            f"{list(frame.columns)} != {list(TARGET_COLUMNS)}"
        )
    if frame.empty:
        raise M3Error("router target table must not be empty")
    if bool(frame.isna().any(axis=None)):
        raise M3Error("router target table must not contain null values")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise M3Error(f"router target row count {len(frame)} != {int(expected_rows)}")
    schema_versions = set(frame["schema_version"].astype(str))
    if schema_versions != {TARGET_SCHEMA_VERSION}:
        raise M3Error(f"router target schema version mismatch: {schema_versions}")
    protocols = set(frame["protocol_id"].astype(str))
    if len(protocols) != 1:
        raise M3Error("router target table contains mixed protocol IDs")
    observed_protocol = next(iter(protocols))
    if protocol_id is not None and observed_protocol != str(protocol_id):
        raise M3Error("router target protocol ID mismatch")
    row_uids = _validate_uid_series(frame["row_uid"], name="router targets")
    _validate_canonical_text(frame["plot_idx"], name="router targets plot_idx")
    seeds = _integer_vector(frame["training_seed"], name="router target training_seed")
    observed_seeds = tuple(sorted(int(value) for value in np.unique(seeds)))
    invalid_seeds = sorted(set(observed_seeds).difference(TRAINING_SEEDS))
    if invalid_seeds:
        raise M3Error(f"router target table contains invalid training seeds: {invalid_seeds}")
    requested = _canonical_expected_seeds(expected_seeds)
    if observed_seeds != requested:
        raise M3Error(
            f"router target seeds {observed_seeds} do not match {requested}"
        )
    states = set(frame["target_state"].astype(str))
    if not states.issubset(set(TARGET_ORDER)):
        raise M3Error(f"router target table contains unknown states: {sorted(states)}")
    keys = list(zip(row_uids.astype(str), seeds.tolist()))
    if len(set(keys)) != len(keys):
        raise M3Error("router target table contains duplicate composite keys")
    if keys != sorted(keys):
        raise M3Error("router target table is not sorted by (row_uid, training_seed)")
    per_uid_seed_counts = frame.groupby("row_uid", observed=False)["training_seed"].nunique()
    if not bool(per_uid_seed_counts.eq(len(observed_seeds)).all()):
        raise M3Error("not every router target identity contains every training seed")
    per_uid_plot_counts = frame.groupby("row_uid", observed=False)["plot_idx"].nunique()
    if not bool(per_uid_plot_counts.eq(1).all()):
        raise M3Error("a router target row_uid maps to multiple plot_idx values")
    return {
        "valid": True,
        "row_count": len(frame),
        "unique_image_count": int(frame["row_uid"].nunique()),
        "plot_count": int(frame["plot_idx"].nunique()),
        "training_seeds": list(observed_seeds),
        "unique_key": list(TARGET_KEY),
        "canonical_sort": list(TARGET_KEY),
        "content_sha256": router_target_content_sha256(frame),
    }


def router_target_content_sha256(table_or_dataframe: Any) -> str:
    """Hash the ordered logical target payload independently of Parquet bytes."""

    frame = _as_dataframe(table_or_dataframe, name="router target table")
    rows = [
        [
            str(row.schema_version),
            str(row.protocol_id),
            str(row.row_uid),
            str(row.plot_idx),
            int(row.training_seed),
            str(row.target_state),
        ]
        for row in frame.itertuples(index=False)
    ]
    return canonical_sha256({"columns": list(TARGET_COLUMNS), "rows": rows})


def _probability_matrix(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, (pd.DataFrame, pd.Series, Mapping)):
        raise M3Error(
            f"{name} must be a raw array-like matrix without identity/column metadata"
        )
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise M3Error(f"{name} must be a numeric probability matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1:] != (N_CLASSES,):
        raise M3Error(f"{name} must have shape (n, {N_CLASSES}); got {matrix.shape}")
    if matrix.shape[0] < 1:
        raise M3Error(f"{name} must contain at least one row")
    if not bool(np.isfinite(matrix).all()):
        raise M3Error(f"{name} contains a non-finite value")
    if bool((matrix < 0.0).any()) or bool((matrix > 1.0).any()):
        raise M3Error(f"{name} contains a value outside [0, 1]")
    if not bool(
        np.allclose(matrix.sum(axis=1), 1.0, atol=PROBABILITY_ATOL, rtol=0.0)
    ):
        raise M3Error(
            f"{name} rows must sum to one within absolute tolerance {PROBABILITY_ATOL}"
        )
    return matrix


def _descending_dense_order(probabilities: np.ndarray) -> np.ndarray:
    # Stable sorting preserves the original ascending dense-ID order for ties.
    return np.argsort(-probabilities, axis=1, kind="mergesort")


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    terms = np.zeros_like(probabilities, dtype=np.float64)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -terms.sum(axis=1, dtype=np.float64)


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    left_terms = np.zeros_like(left, dtype=np.float64)
    right_terms = np.zeros_like(right, dtype=np.float64)
    left_positive = left > 0.0
    right_positive = right > 0.0
    left_terms[left_positive] = left[left_positive] * np.log(
        left[left_positive] / midpoint[left_positive]
    )
    right_terms[right_positive] = right[right_positive] * np.log(
        right[right_positive] / midpoint[right_positive]
    )
    return 0.5 * (
        left_terms.sum(axis=1, dtype=np.float64)
        + right_terms.sum(axis=1, dtype=np.float64)
    )


def build_router_feature_frame(
    image_probabilities: Any,
    geo_probabilities: Any,
    raw_probabilities: Any,
    *,
    probability_basis: str,
) -> pd.DataFrame:
    """Build the frozen 30-column semantic router feature frame.

    ``probability_basis`` is an explicit provenance assertion.  The official
    builder rejects M2's descriptive native-T=1 probabilities so primary router
    features cannot silently bypass M4's seed/mode-specific calibration.
    """

    if probability_basis != CALIBRATED_PROBABILITY_BASIS:
        raise M3Error(
            "router features require probability_basis="
            f"{CALIBRATED_PROBABILITY_BASIS!r}; got {probability_basis!r}"
        )
    image = _probability_matrix(image_probabilities, name="image_probabilities")
    geo = _probability_matrix(geo_probabilities, name="geo_probabilities")
    raw = _probability_matrix(raw_probabilities, name="raw_probabilities")
    if not (len(image) == len(geo) == len(raw)):
        raise M3Error("image, geo, and raw probability matrices must have equal rows")

    row_index = np.arange(len(image), dtype=np.int64)
    image_order = _descending_dense_order(image)
    geo_order = _descending_dense_order(geo)
    raw_order = _descending_dense_order(raw)
    image_pred = image_order[:, 0].astype(np.int8)
    geo_pred = geo_order[:, 0].astype(np.int8)
    raw_pred = raw_order[:, 0].astype(np.int8)
    image_confidence = image[row_index, image_pred].astype(np.float64)
    geo_confidence = geo[row_index, geo_pred].astype(np.float64)
    raw_confidence = raw[row_index, raw_pred].astype(np.float64)
    image_entropy = _entropy(image)
    geo_entropy = _entropy(geo)
    raw_entropy = _entropy(raw)
    image_margin = (image[row_index, image_order[:, 0]] - image[row_index, image_order[:, 1]])
    geo_margin = geo[row_index, geo_order[:, 0]] - geo[row_index, geo_order[:, 1]]
    raw_margin = raw[row_index, raw_order[:, 0]] - raw[row_index, raw_order[:, 1]]
    image_top3 = image_order[:, :3]
    geo_top3 = geo_order[:, :3]
    top3_overlap = np.asarray(
        [len(set(left).intersection(right)) for left, right in zip(image_top3, geo_top3)],
        dtype=np.int8,
    )
    raw_rank = (
        np.argmax(raw_order == geo_pred[:, None], axis=1).astype(np.int8) + np.int8(1)
    )

    frame = pd.DataFrame(
        {
            "image_pred": image_pred,
            "geo_pred": geo_pred,
            "raw_pred": raw_pred,
            "image_geo_pred_pair": (
                image_pred.astype(np.int16) * N_CLASSES + geo_pred.astype(np.int16)
            ).astype(np.int16),
            "geo_raw_pred_pair": (
                geo_pred.astype(np.int16) * N_CLASSES + raw_pred.astype(np.int16)
            ).astype(np.int16),
            "image_geo_agree": (image_pred == geo_pred).astype(bool),
            "image_raw_agree": (image_pred == raw_pred).astype(bool),
            "geo_raw_agree": (geo_pred == raw_pred).astype(bool),
            "image_geo_top3_overlap": top3_overlap,
            "raw_rank_of_geo_pred": raw_rank,
            "image_confidence": image_confidence,
            "geo_confidence": geo_confidence,
            "raw_confidence": raw_confidence,
            "image_entropy": image_entropy,
            "geo_entropy": geo_entropy,
            "raw_entropy": raw_entropy,
            "image_top2_margin": image_margin.astype(np.float64),
            "geo_top2_margin": geo_margin.astype(np.float64),
            "raw_top2_margin": raw_margin.astype(np.float64),
            "geo_minus_image_confidence": geo_confidence - image_confidence,
            "geo_minus_raw_confidence": geo_confidence - raw_confidence,
            "geo_minus_image_entropy": geo_entropy - image_entropy,
            "geo_minus_raw_entropy": geo_entropy - raw_entropy,
            "geo_minus_image_margin": geo_margin - image_margin,
            "geo_minus_raw_margin": geo_margin - raw_margin,
            "image_geo_jsd": _jensen_shannon(image, geo),
            "image_geo_total_variation": 0.5
            * np.abs(image - geo).sum(axis=1, dtype=np.float64),
            "image_probability_at_geo_pred": image[row_index, geo_pred],
            "geo_probability_at_image_pred": geo[row_index, image_pred],
            "raw_probability_at_geo_pred": raw[row_index, geo_pred],
        },
        columns=list(FEATURE_COLUMNS),
    )
    validate_router_feature_frame(
        frame, probability_basis=CALIBRATED_PROBABILITY_BASIS
    )
    return frame


def _allowlist_as_lists(
    configured_allowlist: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    expected = {name: list(columns) for name, columns in FEATURE_FAMILIES.items()}
    if configured_allowlist is None:
        return expected
    if not isinstance(configured_allowlist, Mapping):
        raise M3Error("configured feature allow-list must be a mapping")
    observed = {
        str(name): list(columns)
        for name, columns in configured_allowlist.items()
    }
    if observed != expected:
        raise M3Error(
            "configured router feature allow-list/order has drifted from the M3 contract"
        )
    return expected


def build_router_feature_schema(
    *,
    configured_allowlist: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Return the canonical schema for semantic and later transformed features."""

    allowlist = _allowlist_as_lists(configured_allowlist)
    dtypes = {
        **{name: "int8" for name in CATEGORICAL_FEATURES[:3]},
        **{name: "int16" for name in CATEGORICAL_FEATURES[3:]},
        **{name: "bool" for name in BOOLEAN_FEATURES},
        **{name: "int8" for name in INTEGER_FEATURES},
        **{name: "float64" for name in NUMERIC_FEATURES},
    }
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "class_count": N_CLASSES,
        "probability_basis": CALIBRATED_PROBABILITY_BASIS,
        "probability_validation": {
            "finite": True,
            "range": [0.0, 1.0],
            "row_sum_absolute_tolerance": PROBABILITY_ATOL,
            "renormalize": False,
            "clip": False,
        },
        "semantic_feature_count": len(FEATURE_COLUMNS),
        "family_order": ["categorical", "boolean", "integer", "numeric"],
        "feature_allowlist": allowlist,
        "ordered_semantic_features": list(FEATURE_COLUMNS),
        "dtypes": dtypes,
        "categorical_vocabularies": {
            "image_pred": list(range(N_CLASSES)),
            "geo_pred": list(range(N_CLASSES)),
            "raw_pred": list(range(N_CLASSES)),
            "image_geo_pred_pair": list(range(N_CLASSES * N_CLASSES)),
            "geo_raw_pred_pair": list(range(N_CLASSES * N_CLASSES)),
        },
        "definitions": {
            "predicted_class": "argmax_descending_probability_dense_id_tie_break",
            "predicted_class_pair": "first_dense_id_times_18_plus_second_dense_id",
            "top3_overlap": "integer_set_intersection_cardinality_0_to_3",
            "raw_rank_of_geo_pred": "one_based_descending_probability_dense_id_tie_break",
            "entropy": "negative_sum_p_log_p_natural_log_unnormalized_zero_term_is_zero",
            "jensen_shannon_divergence": "natural_log_unnormalized_zero_term_is_zero",
            "total_variation": "half_l1_probability_distance",
            "signed_difference": "left_named_quantity_minus_right_named_quantity",
        },
        "builder": {
            "stateless": True,
            "fit_operations": [],
            "accepts_identity_group_or_target_columns": False,
            "shared_by": ["m4_router_training", "deployment_inference"],
        },
        "m4_transformed_feature_contract": {
            "block_order": [
                "scaled_boolean_integer_numeric",
                "fixed_vocabulary_one_hot_categorical",
            ],
            "scaled_column_count": 25,
            "one_hot_column_count": 702,
            "total_column_count": 727,
            "dtype": "float64",
            "materialized_by": "M4",
        },
    }


def validate_router_feature_schema(
    schema: Mapping[str, Any],
    *,
    configured_allowlist: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Fail closed unless ``schema`` equals the canonical frozen M3 schema."""

    if not isinstance(schema, Mapping):
        raise M3Error("router feature schema must be a mapping")
    expected = build_router_feature_schema(configured_allowlist=configured_allowlist)
    if dict(schema) != expected:
        raise M3Error("router feature schema differs from the canonical M3 schema")
    return {
        "valid": True,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "semantic_feature_count": len(FEATURE_COLUMNS),
        "schema_sha256": canonical_sha256(expected),
    }


def validate_router_feature_frame(
    frame_or_dataframe: Any,
    *,
    probability_basis: str,
) -> dict[str, Any]:
    """Validate column order, exact dtypes, vocabularies, and finite values."""

    if probability_basis != CALIBRATED_PROBABILITY_BASIS:
        raise M3Error(
            "router feature validation requires calibrated probability provenance"
        )
    frame = _as_dataframe(frame_or_dataframe, name="router feature frame")
    if tuple(frame.columns) != FEATURE_COLUMNS:
        raise M3Error(
            f"router feature columns differ from frozen allow-list: {list(frame.columns)}"
        )
    if frame.empty:
        raise M3Error("router feature frame must not be empty")
    if bool(frame.isna().any(axis=None)):
        raise M3Error("router feature frame contains missing values")
    expected_dtypes = {
        **{name: np.dtype("int8") for name in CATEGORICAL_FEATURES[:3]},
        **{name: np.dtype("int16") for name in CATEGORICAL_FEATURES[3:]},
        **{name: np.dtype("bool") for name in BOOLEAN_FEATURES},
        **{name: np.dtype("int8") for name in INTEGER_FEATURES},
        **{name: np.dtype("float64") for name in NUMERIC_FEATURES},
    }
    for column, expected_dtype in expected_dtypes.items():
        if np.dtype(frame[column].dtype) != expected_dtype:
            raise M3Error(
                f"router feature {column} dtype {frame[column].dtype} != {expected_dtype}"
            )
    for column in CATEGORICAL_FEATURES[:3]:
        values = frame[column].to_numpy(dtype=np.int64)
        if bool((values < 0).any()) or bool((values >= N_CLASSES).any()):
            raise M3Error(f"router feature {column} is outside dense ontology")
    for column in CATEGORICAL_FEATURES[3:]:
        values = frame[column].to_numpy(dtype=np.int64)
        if bool((values < 0).any()) or bool((values >= N_CLASSES * N_CLASSES).any()):
            raise M3Error(f"router feature {column} is outside fixed pair vocabulary")
    overlap = frame["image_geo_top3_overlap"].to_numpy(dtype=np.int64)
    if bool((overlap < 0).any()) or bool((overlap > 3).any()):
        raise M3Error("image_geo_top3_overlap must be in [0, 3]")
    rank = frame["raw_rank_of_geo_pred"].to_numpy(dtype=np.int64)
    if bool((rank < 1).any()) or bool((rank > N_CLASSES).any()):
        raise M3Error("raw_rank_of_geo_pred must be one-based in [1, 18]")
    numeric = frame.loc[:, list(NUMERIC_FEATURES)].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(numeric).all()):
        raise M3Error("router numeric features contain non-finite values")
    return {
        "valid": True,
        "row_count": len(frame),
        "semantic_feature_count": len(FEATURE_COLUMNS),
        "ordered_features": list(FEATURE_COLUMNS),
        "probability_basis": CALIBRATED_PROBABILITY_BASIS,
    }


def _state_summary(
    states: pd.Series,
    *,
    state_order: Sequence[str] = TARGET_ORDER,
    total: int | None = None,
) -> dict[str, Any]:
    counts = states.astype(str).value_counts().to_dict()
    denominator = len(states) if total is None else int(total)
    return {
        "record_count": denominator,
        "target_counts": {
            state: int(counts.get(state, 0)) for state in state_order
        },
        "target_fractions": {
            state: (
                float(counts.get(state, 0) / denominator)
                if denominator > 0
                else None
            )
            for state in state_order
        },
    }


def _group_state_record(
    rows: pd.DataFrame,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(identity)
    record.update(_state_summary(rows["target_state"]))
    record["unique_image_count"] = int(rows["row_uid"].nunique())
    record["unique_plot_count"] = int(rows["plot_idx"].nunique())
    return record


def build_target_prevalence_report(
    oof_predictions: Any,
    assignments: Any,
    *,
    protocol_id: str | None = None,
    expected_seeds: Sequence[int] = TRAINING_SEEDS,
) -> dict[str, Any]:
    """Build deterministic seed-aware target prevalence and diagnostics."""

    joined, observed_protocol, observed_seeds = _prepared_target_rows(
        oof_predictions,
        assignments,
        protocol_id=protocol_id,
        expected_seeds=expected_seeds,
    )
    pooled = _state_summary(joined["target_state"])
    pooled.update(
        {
            "unique_image_count": int(joined["row_uid"].nunique()),
            "unique_plot_count": int(joined["plot_idx"].nunique()),
        }
    )
    per_seed: list[dict[str, Any]] = []
    for seed in observed_seeds:
        rows = joined.loc[joined["training_seed"].eq(seed)]
        record = {"training_seed": seed, **_state_summary(rows["target_state"])}
        record["unique_image_count"] = int(rows["row_uid"].nunique())
        record["unique_plot_count"] = int(rows["plot_idx"].nunique())
        per_seed.append(record)

    habitat: list[dict[str, Any]] = []
    for dense_id in range(N_CLASSES):
        rows = joined.loc[joined["label_id_dense"].eq(dense_id)]
        habitat.append(
            _group_state_record(
                rows,
                identity={
                    "label_id_dense": dense_id,
                    "canonical_l3_id": int(DENSE_TO_CANONICAL_L3[dense_id]),
                    "label_name": str(DENSE_TO_LABEL_NAME[dense_id]),
                },
            )
        )

    plot: list[dict[str, Any]] = []
    for plot_idx in sorted(joined["plot_idx"].astype(str).unique()):
        rows = joined.loc[joined["plot_idx"].astype(str).eq(plot_idx)]
        plot.append(_group_state_record(rows, identity={"plot_idx": plot_idx}))

    image_geo_pairs: list[dict[str, Any]] = []
    geo_raw_pairs: list[dict[str, Any]] = []
    for pair_id in range(N_CLASSES * N_CLASSES):
        first, second = divmod(pair_id, N_CLASSES)
        image_geo_rows = joined.loc[
            joined["image_pred"].eq(first) & joined["geo_pred"].eq(second)
        ]
        image_geo_pairs.append(
            _group_state_record(
                image_geo_rows,
                identity={
                    "pair_id": pair_id,
                    "image_pred": first,
                    "geo_pred": second,
                },
            )
        )
        geo_raw_rows = joined.loc[
            joined["geo_pred"].eq(first) & joined["raw_pred"].eq(second)
        ]
        geo_raw_pairs.append(
            _group_state_record(
                geo_raw_rows,
                identity={
                    "pair_id": pair_id,
                    "geo_pred": first,
                    "raw_pred": second,
                },
            )
        )

    auxiliary_overall = _state_summary(
        joined["image_relative_state"], state_order=IMAGE_RELATIVE_TARGET_ORDER
    )
    auxiliary_per_seed = []
    for seed in observed_seeds:
        rows = joined.loc[joined["training_seed"].eq(seed)]
        auxiliary_per_seed.append(
            {
                "training_seed": seed,
                **_state_summary(
                    rows["image_relative_state"],
                    state_order=IMAGE_RELATIVE_TARGET_ORDER,
                ),
            }
        )

    distinct_counts = (
        joined.groupby("row_uid", sort=True, observed=False)["target_state"]
        .nunique()
        .astype(int)
    )
    distribution = {
        str(count): int(distinct_counts.eq(count).sum())
        for count in range(1, len(observed_seeds) + 1)
    }
    changing = int(distinct_counts.gt(1).sum())
    report = {
        "schema_version": PREVALENCE_SCHEMA_VERSION,
        "protocol_id": observed_protocol,
        "target_order": list(TARGET_ORDER),
        "target_definition": {
            "rescue": "raw_wrong_and_geo_correct",
            "harm": "raw_correct_and_geo_wrong",
            "both_correct": "raw_correct_and_geo_correct",
            "both_wrong": "raw_wrong_and_geo_wrong",
        },
        "key": list(TARGET_KEY),
        "training_seeds": list(observed_seeds),
        "pooled_seed_realizations": pooled,
        "per_training_seed": per_seed,
        "breakdowns": {
            "habitat": habitat,
            "plot": plot,
            "image_geo_pred_pair": image_geo_pairs,
            "geo_raw_pred_pair": geo_raw_pairs,
        },
        "auxiliary_image_relative_states": {
            "purpose": "diagnostic_only_not_a_router_target",
            "baseline": "image_only",
            "alternative": "geo_only",
            "state_order": list(IMAGE_RELATIVE_TARGET_ORDER),
            "pooled_seed_realizations": auxiliary_overall,
            "per_training_seed": auxiliary_per_seed,
        },
        "cross_seed_stability": {
            "unique_image_count": int(len(distinct_counts)),
            "distinct_target_state_count_distribution": distribution,
            "unchanged_image_count": int(distinct_counts.eq(1).sum()),
            "changing_image_count": changing,
            "changing_image_fraction": float(changing / len(distinct_counts)),
        },
        "interpretation_warning": (
            f"{len(joined):,} seed records represent "
            f"{joined['row_uid'].nunique():,} images across independent training-seed "
            "realizations; they are not independent biological samples."
        ),
        "zero_support_rate_serialization": "null",
    }
    # Canonical serialization is also the finite-value check (NaN/Infinity fail).
    canonical_json_bytes(report)
    return report


DEFAULT_INPUT_ARTIFACT_ROLES = (
    "development_assignments",
    "train_oof_fold_outputs",
    "development_train_oof_outputs",
)
IDENTITY_GROUP_TARGET_EXCLUSIONS = (
    "schema_version",
    "protocol_id",
    "row_uid",
    "file",
    "file_lower",
    "plot_idx",
    "train_oof_fold",
    "training_seed",
    "label_id_dense",
    "label_name",
    "target_state",
    "image_relative_state",
)


def build_feature_leakage_audit(
    *,
    configured_allowlist: Mapping[str, Sequence[str]] | None = None,
    forbidden_patterns: Sequence[str] = (),
    input_artifact_roles: Sequence[str] = DEFAULT_INPUT_ARTIFACT_ROLES,
) -> dict[str, Any]:
    """Audit the exact feature allow-list and M3's permitted input roles."""

    allowlist = _allowlist_as_lists(configured_allowlist)
    if isinstance(forbidden_patterns, (str, bytes)):
        raise M3Error("forbidden feature patterns must be a sequence, not a string")
    patterns = [str(value) for value in forbidden_patterns]
    if any(not value or value != value.strip() for value in patterns):
        raise M3Error("forbidden feature patterns must be non-empty canonical strings")
    if len(set(patterns)) != len(patterns):
        raise M3Error("forbidden feature patterns contain duplicates")
    roles = [str(role) for role in input_artifact_roles]
    if roles != list(DEFAULT_INPUT_ARTIFACT_ROLES):
        raise M3Error(
            "M3 input artifact roles differ from the development-train-only contract"
        )
    matches = {
        column: [
            pattern
            for pattern in patterns
            if pattern.casefold() in column.casefold()
        ]
        for column in FEATURE_COLUMNS
    }
    matches = {column: values for column, values in matches.items() if values}
    excluded_overlap = sorted(set(FEATURE_COLUMNS).intersection(IDENTITY_GROUP_TARGET_EXCLUSIONS))
    valid = not matches and not excluded_overlap
    audit = {
        "schema_version": LEAKAGE_AUDIT_SCHEMA_VERSION,
        "valid": valid,
        "ordered_feature_allowlist": allowlist,
        "ordered_semantic_features": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "configured_forbidden_patterns": patterns,
        "forbidden_pattern_matches": matches,
        "identity_group_and_target_exclusions": list(IDENTITY_GROUP_TARGET_EXCLUSIONS),
        "excluded_name_overlap": excluded_overlap,
        "probability_basis_required": CALIBRATED_PROBABILITY_BASIS,
        "native_t1_probability_basis_rejected": True,
        "target_feature_separation": {
            "target_columns": list(TARGET_COLUMNS),
            "target_columns_in_features": sorted(
                set(TARGET_COLUMNS).intersection(FEATURE_COLUMNS)
            ),
            "labels_correctness_and_nll_accepted_by_builder": False,
        },
        "input_artifact_roles": roles,
        "forbidden_input_artifact_roles": [
            "final_development_in_sample_outputs",
            "development_validation_outputs",
            "locked_test_predictions",
        ],
        "feature_builder_interface": {
            "inputs": [
                "image_probabilities",
                "geo_probabilities",
                "raw_probabilities",
                "probability_basis",
            ],
            "accepts_dataframe_metadata": False,
            "fit_state": False,
        },
    }
    if not valid:
        raise M3Error(
            f"feature leakage audit failed: matches={matches}, overlap={excluded_overlap}"
        )
    canonical_json_bytes(audit)
    return audit


def validate_feature_leakage_audit(
    audit: Mapping[str, Any],
    *,
    configured_allowlist: Mapping[str, Sequence[str]] | None = None,
    forbidden_patterns: Sequence[str] = (),
    input_artifact_roles: Sequence[str] = DEFAULT_INPUT_ARTIFACT_ROLES,
) -> dict[str, Any]:
    """Reproduce and validate the canonical leakage audit."""

    if not isinstance(audit, Mapping):
        raise M3Error("feature leakage audit must be a mapping")
    expected = build_feature_leakage_audit(
        configured_allowlist=configured_allowlist,
        forbidden_patterns=forbidden_patterns,
        input_artifact_roles=input_artifact_roles,
    )
    if dict(audit) != expected or audit.get("valid") is not True:
        raise M3Error("feature leakage audit does not reproduce or did not pass")
    return {
        "valid": True,
        "feature_count": len(FEATURE_COLUMNS),
        "audit_sha256": canonical_sha256(expected),
    }


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def m3_bundle_path(artifact_root: str | Path) -> Path:
    """Resolve the dedicated M3 bundle below an M2 artifact root."""

    return _resolve_project_path(artifact_root) / BUNDLE_RELATIVE_PATH


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise M3ArtifactError(f"{name} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M3ArtifactError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise M3ArtifactError(f"{name} must be a JSON object: {path}")
    return value


def _target_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("protocol_id", pa.string(), nullable=False),
            pa.field("row_uid", pa.string(), nullable=False),
            pa.field("plot_idx", pa.string(), nullable=False),
            pa.field("training_seed", pa.int8(), nullable=False),
            pa.field("target_state", pa.string(), nullable=False),
        ]
    )


def _target_schema_declaration() -> dict[str, Any]:
    declaration = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "columns": list(TARGET_COLUMNS),
        "field_types": {
            "schema_version": "string_non_null",
            "protocol_id": "string_non_null",
            "row_uid": "sha256_hex_string_non_null",
            "plot_idx": "canonical_string_non_null",
            "training_seed": "int8_seed_1_to_4_non_null",
            "target_state": "enum_rescue_harm_both_correct_both_wrong_non_null",
        },
        "unique_key": list(TARGET_KEY),
        "canonical_sort": list(TARGET_KEY),
        "arrow_schema": str(_target_arrow_schema()),
    }
    return {**declaration, "schema_sha256": canonical_sha256(declaration)}


def _prevalence_schema_declaration() -> dict[str, Any]:
    declaration = {
        "schema_version": PREVALENCE_SCHEMA_VERSION,
        "top_level_fields": [
            "schema_version",
            "protocol_id",
            "target_order",
            "target_definition",
            "key",
            "training_seeds",
            "pooled_seed_realizations",
            "per_training_seed",
            "breakdowns",
            "auxiliary_image_relative_states",
            "cross_seed_stability",
            "interpretation_warning",
            "zero_support_rate_serialization",
        ],
        "breakdown_families": [
            "habitat",
            "plot",
            "image_geo_pred_pair",
            "geo_raw_pred_pair",
        ],
        "target_state_order": list(TARGET_ORDER),
        "auxiliary_state_order": list(IMAGE_RELATIVE_TARGET_ORDER),
        "zero_support_fraction": "null",
    }
    return {**declaration, "schema_sha256": canonical_sha256(declaration)}


def _leakage_schema_declaration() -> dict[str, Any]:
    declaration = {
        "schema_version": LEAKAGE_AUDIT_SCHEMA_VERSION,
        "top_level_fields": [
            "schema_version",
            "valid",
            "ordered_feature_allowlist",
            "ordered_semantic_features",
            "feature_count",
            "configured_forbidden_patterns",
            "forbidden_pattern_matches",
            "identity_group_and_target_exclusions",
            "excluded_name_overlap",
            "probability_basis_required",
            "native_t1_probability_basis_rejected",
            "target_feature_separation",
            "input_artifact_roles",
            "forbidden_input_artifact_roles",
            "feature_builder_interface",
        ],
        "required_valid_value": True,
        "semantic_feature_count": len(FEATURE_COLUMNS),
    }
    return {**declaration, "schema_sha256": canonical_sha256(declaration)}


def _target_arrow_table(frame: pd.DataFrame) -> pa.Table:
    validate_router_target_table(frame)
    schema = _target_arrow_schema()
    arrays = [
        pa.array(frame[column].tolist(), type=schema.field(column).type)
        for column in TARGET_COLUMNS
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _read_target_parquet(path: Path) -> tuple[pa.Table, pd.DataFrame]:
    if path.is_symlink():
        raise M3ArtifactError(f"router target table must not be a symlink: {path}")
    if not path.is_file():
        raise M3ArtifactError(f"router target table does not exist: {path}")
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise M3ArtifactError(f"cannot read router target table {path}: {exc}") from exc
    if not table.schema.equals(_target_arrow_schema(), check_metadata=False):
        raise M3ArtifactError(f"router target Arrow schema mismatch: {table.schema}")
    frame = table.to_pandas()
    validate_router_target_table(frame)
    return table, frame


def _write_target_parquet(path: Path, frame: pd.DataFrame) -> Path:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    table = _target_arrow_table(frame)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )
    os.chmod(path, 0o444)
    return path


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(value)) + b"\n")
    except FileExistsError as exc:
        raise FileExistsError(f"immutable artifact already exists: {path}") from exc
    os.chmod(path, 0o444)
    return path


def _manifest_self_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return canonical_sha256(payload)


def _m3_implementation_hashes() -> tuple[dict[str, str], str]:
    paths = (Path(__file__).resolve(), M3_RUNNER_PATH.resolve())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise M3ArtifactError(f"M3 implementation file is missing: {missing}")
    hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths
    }
    return hashes, canonical_sha256(hashes)


def _validate_frozen_router_contract(config: Mapping[str, Any]) -> None:
    try:
        router = config["router"]
        calibration = config["calibration"]
        artifact_contract = config["artifact_contract"]
    except (KeyError, TypeError) as exc:
        raise M3ArtifactError("resolved protocol is missing the router contract") from exc
    if tuple(router.get("target_order", ())) != TARGET_ORDER:
        raise M3ArtifactError("resolved router target order has drifted")
    expected_target_definition = {
        "rescue": "raw_wrong_and_geo_correct",
        "harm": "raw_correct_and_geo_wrong",
        "both_correct": "raw_correct_and_geo_correct",
        "both_wrong": "raw_wrong_and_geo_wrong",
    }
    if router.get("target_definition") != expected_target_definition:
        raise M3ArtifactError("resolved router target definition has drifted")
    _allowlist_as_lists(router.get("feature_allowlist"))
    feature_matrix = router.get("feature_matrix", {})
    expected_counts = {
        "scaled_numeric_column_count": 25,
        "one_hot_column_count": 702,
        "total_column_count": 727,
        "final_dtype": "float64",
    }
    for key, expected in expected_counts.items():
        if feature_matrix.get(key) != expected:
            raise M3ArtifactError(f"resolved feature-matrix contract drifted: {key}")
    if calibration.get("native_t1_role") != "descriptive_only":
        raise M3ArtifactError("native-T=1 probability role is no longer descriptive-only")
    if artifact_contract.get("write_mode") != "exclusive_create":
        raise M3ArtifactError("artifact write contract is no longer exclusive-create")


def _validate_real_acceptance(prepared: _PreparedM3) -> None:
    """Check observed protocol-v1 evidence after targets have been derived."""

    validation = validate_router_target_table(
        prepared.targets,
        protocol_id=str(prepared.context.config["protocol_id"]),
        expected_rows=13_512,
        expected_seeds=TRAINING_SEEDS,
    )
    if validation["unique_image_count"] != 3_378:
        raise M3ArtifactError("M3 target artifact does not contain 3,378 unique images")
    if validation["plot_count"] != 1_300:
        raise M3ArtifactError("M3 target artifact does not contain 1,300 plots")
    expected_counts = {
        "rescue": 1_225,
        "harm": 2_256,
        "both_correct": 7_171,
        "both_wrong": 2_860,
    }
    observed_counts = prepared.prevalence["pooled_seed_realizations"]["target_counts"]
    if observed_counts != expected_counts:
        raise M3ArtifactError(
            f"M3 target-state acceptance evidence changed: {observed_counts}"
        )
    expected_per_seed = {
        1: {
            "rescue": 327,
            "harm": 560,
            "both_correct": 1_800,
            "both_wrong": 691,
        },
        2: {
            "rescue": 318,
            "harm": 549,
            "both_correct": 1_786,
            "both_wrong": 725,
        },
        3: {
            "rescue": 283,
            "harm": 572,
            "both_correct": 1_789,
            "both_wrong": 734,
        },
        4: {
            "rescue": 297,
            "harm": 575,
            "both_correct": 1_796,
            "both_wrong": 710,
        },
    }
    observed_per_seed = {
        int(record["training_seed"]): record["target_counts"]
        for record in prepared.prevalence["per_training_seed"]
    }
    if observed_per_seed != expected_per_seed:
        raise M3ArtifactError(
            f"M3 per-seed target-state acceptance evidence changed: {observed_per_seed}"
        )
    stability = prepared.prevalence["cross_seed_stability"]
    if stability.get("changing_image_count") != 989:
        raise M3ArtifactError("M3 cross-seed changing-image count is not 989")
    if stability.get("distinct_target_state_count_distribution") != {
        "1": 2_389,
        "2": 875,
        "3": 100,
        "4": 14,
    }:
        raise M3ArtifactError("M3 cross-seed stability distribution changed")


def _prepare_m3(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> _PreparedM3:
    """Validate M1 and exactly the 16 train-OOF M2 producers and aggregate."""

    # Lazy import keeps pure target/feature calls independent of Torch and all
    # model-training dependencies.
    from multimodal import geo_helpfulness_oof as m2

    context = m2.load_frozen_context(
        config_path=config_path,
        protocol_dir=protocol_dir,
        output_root=artifact_root,
    )
    _validate_frozen_router_contract(context.config)
    train = context.assignments.loc[
        context.assignments["development_role"].astype(str).eq("train")
    ].copy()
    if len(train) != 3_378 or int(train["plot_idx"].nunique()) != 1_300:
        raise M3ArtifactError(
            "sealed development-train assignment counts are not 3,378 rows/1,300 plots"
        )
    train_projection_columns = [
        "protocol_id",
        "row_uid",
        "file",
        "file_lower",
        "plot_idx",
        "label_id_dense",
        "development_role",
        "train_oof_fold",
    ]
    if "label_name" in train.columns:
        train_projection_columns.append("label_name")
    train_assignment_projection = train.loc[:, train_projection_columns].copy()

    oof_tables: list[pa.Table] = []
    producer_hashes: dict[str, str] = {}
    producer_file_hashes: dict[str, str] = {}
    for seed in TRAINING_SEEDS:
        oof_specs = tuple(spec for spec in m2._producer_specs(seed) if spec.include_fold)
        if len(oof_specs) != len(OOF_FOLDS):
            raise M3ArtifactError(f"seed {seed} does not resolve to four OOF producers")
        for spec in oof_specs:
            result = m2.validate_producer(context, spec, seed=seed)
            relative = spec.relative_directory.as_posix()
            producer_hashes[relative] = str(result["manifest_sha256"])
            manifest_path = context.output_root / spec.relative_directory / "manifest.json"
            producer_file_hashes[relative] = sha256_file(manifest_path)
            output_path = (
                context.output_root / spec.relative_directory / spec.output_filename
            )
            oof_tables.append(m2.read_output_parquet(output_path, include_fold=True))
    if len(oof_tables) != 16 or len(producer_hashes) != 16:
        raise M3ArtifactError("M3 did not validate exactly 16 train-OOF producers")

    reconstructed = m2._aggregate_table(
        oof_tables,
        include_fold=True,
        expected_assignments=train,
    )
    aggregate_validation = m2._validate_aggregate_artifact(
        context,
        include_fold=True,
        producer_manifest_hashes=producer_hashes,
        expected_table=reconstructed,
    )
    aggregate_path, aggregate_manifest_path, _ = m2._aggregate_paths(
        context, include_fold=True
    )
    aggregate_table = m2.read_output_parquet(aggregate_path, include_fold=True)
    if m2.logical_table_sha256(aggregate_table) != m2.logical_table_sha256(reconstructed):
        raise M3ArtifactError(
            "sealed OOF aggregate differs from the 16 validated producer concatenation"
        )
    aggregate_manifest = _read_json_mapping(
        aggregate_manifest_path, name="M2 train-OOF aggregate manifest"
    )

    protocol_id = str(context.config["protocol_id"])
    targets = build_router_target_table(
        aggregate_table,
        train_assignment_projection,
        protocol_id=protocol_id,
        expected_seeds=TRAINING_SEEDS,
    )
    feature_allowlist = context.config["router"]["feature_allowlist"]
    forbidden_patterns = context.config["router"]["forbidden_feature_patterns"]
    feature_schema = build_router_feature_schema(
        configured_allowlist=feature_allowlist
    )
    prevalence = build_target_prevalence_report(
        aggregate_table,
        train_assignment_projection,
        protocol_id=protocol_id,
        expected_seeds=TRAINING_SEEDS,
    )
    leakage_audit = build_feature_leakage_audit(
        configured_allowlist=feature_allowlist,
        forbidden_patterns=forbidden_patterns,
        input_artifact_roles=DEFAULT_INPUT_ARTIFACT_ROLES,
    )
    prepared = _PreparedM3(
        context=context,
        oof_table=aggregate_table,
        producer_manifest_hashes=dict(sorted(producer_hashes.items())),
        producer_manifest_file_hashes=dict(sorted(producer_file_hashes.items())),
        aggregate_manifest=aggregate_manifest,
        aggregate_validation=aggregate_validation,
        targets=targets,
        feature_schema=feature_schema,
        prevalence=prevalence,
        leakage_audit=leakage_audit,
    )
    _validate_real_acceptance(prepared)
    return prepared


def _artifact_records(bundle_root: Path, targets: pd.DataFrame) -> dict[str, Any]:
    return {
        TARGET_FILENAME: {
            "file_sha256": sha256_file(bundle_root / TARGET_FILENAME),
            "content_sha256": router_target_content_sha256(targets),
        },
        FEATURE_SCHEMA_FILENAME: {
            "file_sha256": sha256_file(bundle_root / FEATURE_SCHEMA_FILENAME),
            "content_sha256": canonical_sha256(
                _read_json_mapping(
                    bundle_root / FEATURE_SCHEMA_FILENAME,
                    name="router feature schema",
                )
            ),
        },
        PREVALENCE_FILENAME: {
            "file_sha256": sha256_file(bundle_root / PREVALENCE_FILENAME),
            "content_sha256": canonical_sha256(
                _read_json_mapping(
                    bundle_root / PREVALENCE_FILENAME,
                    name="target prevalence report",
                )
            ),
        },
        LEAKAGE_AUDIT_FILENAME: {
            "file_sha256": sha256_file(bundle_root / LEAKAGE_AUDIT_FILENAME),
            "content_sha256": canonical_sha256(
                _read_json_mapping(
                    bundle_root / LEAKAGE_AUDIT_FILENAME,
                    name="feature leakage audit",
                )
            ),
        },
    }


def _build_m3_manifest(
    prepared: _PreparedM3,
    *,
    bundle_root: Path,
) -> dict[str, Any]:
    context = prepared.context
    code_files, code_hash = _m3_implementation_hashes()
    aggregate_path = context.output_root / "development_train_oof" / (
        "development_train_oof_model_outputs.parquet"
    )
    aggregate_manifest_path = (
        context.output_root / "development_train_oof" / "aggregate_manifest.json"
    )
    target_validation = validate_router_target_table(
        prepared.targets,
        protocol_id=str(context.config["protocol_id"]),
        expected_rows=13_512,
        expected_seeds=TRAINING_SEEDS,
    )
    manifest: dict[str, Any] = {
        "schema_version": M3_MANIFEST_SCHEMA_VERSION,
        "protocol_id": str(context.config["protocol_id"]),
        "artifact_role": "router_targets_and_feature_contract",
        "parent_roles": list(DEFAULT_INPUT_ARTIFACT_ROLES),
        "parent_artifact_hashes": {
            "protocol_manifest": {
                "file_sha256": sha256_file(context.manifest_path),
                "payload_sha256": context.protocol_manifest.get(
                    "manifest_payload_sha256"
                ),
            },
            "development_assignments": {
                "file_sha256": sha256_file(context.assignments_path),
                "content_sha256": assignment_fingerprint(context.assignments),
            },
            "resolved_protocol": {
                "file_sha256": sha256_file(context.resolved_path),
                "effective_config_sha256": context.protocol_manifest.get(
                    "effective_config_sha256"
                ),
            },
            "class_map": {
                "content_sha256": context.protocol_manifest.get("class_map_sha256")
            },
            "feature_allowlist": {
                "content_sha256": canonical_sha256(
                    context.config["router"]["feature_allowlist"]
                ),
                "m1_sealed_sha256": context.protocol_manifest.get(
                    "feature_allowlist_sha256"
                ),
            },
            "development_train_oof_aggregate": {
                "file_sha256": sha256_file(aggregate_path),
                "content_sha256": prepared.aggregate_manifest.get("content_sha256"),
                "manifest_file_sha256": sha256_file(aggregate_manifest_path),
                "manifest_sha256": prepared.aggregate_manifest.get("manifest_sha256"),
            },
            "train_oof_producer_manifests": {
                "manifest_sha256": prepared.producer_manifest_hashes,
                "file_sha256": prepared.producer_manifest_file_hashes,
            },
        },
        "m1_preflight_status": "valid",
        "m2_code_file_sha256": context.code_file_hashes,
        "m2_code_sha256": context.code_hash,
        "m3_code_file_sha256": code_files,
        "m3_code_sha256": code_hash,
        "schemas": {
            "router_targets": _target_schema_declaration(),
            "router_features": {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "schema_sha256": canonical_sha256(prepared.feature_schema),
            },
            "target_prevalence": {
                **_prevalence_schema_declaration(),
            },
            "feature_leakage_audit": {
                **_leakage_schema_declaration(),
            },
            "manifest": {"schema_version": M3_MANIFEST_SCHEMA_VERSION},
        },
        "artifacts": _artifact_records(bundle_root, prepared.targets),
        "target_table": {
            "filename": TARGET_FILENAME,
            "schema_version": TARGET_SCHEMA_VERSION,
            "columns": list(TARGET_COLUMNS),
            "arrow_schema": str(_target_arrow_schema()),
            "row_count": target_validation["row_count"],
            "unique_image_count": target_validation["unique_image_count"],
            "plot_count": target_validation["plot_count"],
            "training_seeds": list(TRAINING_SEEDS),
            "unique_key": list(TARGET_KEY),
            "canonical_sort": list(TARGET_KEY),
            "seed_aggregation": "none",
        },
        "feature_contract": {
            "filename": FEATURE_SCHEMA_FILENAME,
            "schema_sha256": canonical_sha256(prepared.feature_schema),
            "semantic_feature_count": len(FEATURE_COLUMNS),
            "probability_basis": CALIBRATED_PROBABILITY_BASIS,
            "native_t1_primary_features_materialized": False,
            "calibrated_feature_rows_materialized": False,
            "router_dataset_materialized": False,
        },
        "calibration_boundary": {
            "temperature_fitted_by_m3": False,
            "temperature_count": 0,
            "owner": "M4",
            "m4_temperature_count": 12,
            "router_dataset_owner": "M4",
        },
        "source_access": {
            "allowed_roles": list(DEFAULT_INPUT_ARTIFACT_ROLES),
            "validated_train_oof_producer_count": 16,
            "development_assignment_projection": {
                "development_role": "train",
                "required_columns": [
                    "protocol_id",
                    "row_uid",
                    "file",
                    "file_lower",
                    "plot_idx",
                    "label_id_dense",
                    "development_role",
                    "train_oof_fold",
                ],
                "optional_validated_columns": ["label_name"],
            },
            "development_validation_outputs_opened": False,
            "final_in_sample_outputs_opened": False,
            "locked_test_sources_opened": False,
        },
        "post_derivation_verification_evidence": {
            "used_as_target_generation_input": False,
            "expected_protocol_v1_row_count": 13_512,
            "expected_protocol_v1_unique_image_count": 3_378,
            "expected_protocol_v1_plot_count": 1_300,
            "expected_protocol_v1_target_counts": {
                "rescue": 1_225,
                "harm": 2_256,
                "both_correct": 7_171,
                "both_wrong": 2_860,
            },
            "expected_protocol_v1_target_counts_by_seed": [
                {
                    "training_seed": 1,
                    "rescue": 327,
                    "harm": 560,
                    "both_correct": 1_800,
                    "both_wrong": 691,
                },
                {
                    "training_seed": 2,
                    "rescue": 318,
                    "harm": 549,
                    "both_correct": 1_786,
                    "both_wrong": 725,
                },
                {
                    "training_seed": 3,
                    "rescue": 283,
                    "harm": 572,
                    "both_correct": 1_789,
                    "both_wrong": 734,
                },
                {
                    "training_seed": 4,
                    "rescue": 297,
                    "harm": 575,
                    "both_correct": 1_796,
                    "both_wrong": 710,
                },
            ],
            "expected_protocol_v1_changing_image_count": 989,
        },
        "publication": {
            "write_mode": "exclusive_create",
            "manifest_is_commit_marker": True,
            "committed_file_allowlist": list(BUNDLE_FILENAMES),
            "router_dataset_parquet": {
                "produced_by_m3": False,
                "absent_at_m3_publication": True,
                "later_owner": "M4",
            },
        },
    }
    manifest["manifest_sha256"] = _manifest_self_hash(manifest)
    return manifest


def _bundle_root(prepared: _PreparedM3) -> Path:
    return m3_bundle_path(prepared.context.output_root)


def _load_and_compare_bundle_payloads(
    prepared: _PreparedM3,
    bundle_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    table, targets = _read_target_parquet(bundle_root / TARGET_FILENAME)
    expected_table = _target_arrow_table(prepared.targets)
    if not table.equals(expected_table):
        raise M3ArtifactError("published router targets do not reproduce from OOF inputs")
    feature_schema = _read_json_mapping(
        bundle_root / FEATURE_SCHEMA_FILENAME, name="router feature schema"
    )
    prevalence = _read_json_mapping(
        bundle_root / PREVALENCE_FILENAME, name="target prevalence report"
    )
    leakage_audit = _read_json_mapping(
        bundle_root / LEAKAGE_AUDIT_FILENAME, name="feature leakage audit"
    )
    if feature_schema != prepared.feature_schema:
        raise M3ArtifactError("published router feature schema does not reproduce")
    if prevalence != prepared.prevalence:
        raise M3ArtifactError("published target prevalence report does not reproduce")
    if leakage_audit != prepared.leakage_audit:
        raise M3ArtifactError("published feature leakage audit does not reproduce")
    validate_router_feature_schema(
        feature_schema,
        configured_allowlist=prepared.context.config["router"]["feature_allowlist"],
    )
    validate_feature_leakage_audit(
        leakage_audit,
        configured_allowlist=prepared.context.config["router"]["feature_allowlist"],
        forbidden_patterns=prepared.context.config["router"][
            "forbidden_feature_patterns"
        ],
        input_artifact_roles=DEFAULT_INPUT_ARTIFACT_ROLES,
    )
    return targets, feature_schema, prevalence, leakage_audit


def _validate_prepared_bundle(prepared: _PreparedM3) -> dict[str, Any]:
    bundle_root = _bundle_root(prepared)
    if bundle_root.is_symlink():
        raise M3ArtifactError(f"M3 bundle root must not be a symlink: {bundle_root}")
    if not bundle_root.is_dir():
        raise M3ArtifactError(f"M3 bundle directory does not exist: {bundle_root}")
    observed_names = {path.name for path in bundle_root.iterdir()}
    if observed_names != set(BUNDLE_FILENAMES):
        raise M3ArtifactError(
            "M3 bundle file allow-list mismatch; "
            f"observed={sorted(observed_names)}, expected={sorted(BUNDLE_FILENAMES)}"
        )
    if any(
        (bundle_root / name).is_symlink()
        or not (bundle_root / name).is_file()
        for name in BUNDLE_FILENAMES
    ):
        raise M3ArtifactError("M3 bundle contains a symlink or non-file allow-list entry")
    targets, feature_schema, prevalence, leakage_audit = _load_and_compare_bundle_payloads(
        prepared, bundle_root
    )
    manifest_path = bundle_root / MANIFEST_FILENAME
    manifest = _read_json_mapping(manifest_path, name="M3 manifest")
    if manifest.get("manifest_sha256") != _manifest_self_hash(manifest):
        raise M3ArtifactError("M3 manifest self-hash mismatch")
    expected_manifest = _build_m3_manifest(prepared, bundle_root=bundle_root)
    if manifest != expected_manifest:
        raise M3ArtifactError("M3 manifest lineage or artifact fingerprints are stale")
    return {
        "valid": True,
        "status": "reused_valid",
        "protocol_id": str(prepared.context.config["protocol_id"]),
        "bundle_root": str(bundle_root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "row_count": len(targets),
        "unique_image_count": int(targets["row_uid"].nunique()),
        "plot_count": int(targets["plot_idx"].nunique()),
        "training_seeds": list(TRAINING_SEEDS),
        "target_counts": prevalence["pooled_seed_realizations"]["target_counts"],
        "changing_image_count": prevalence["cross_seed_stability"][
            "changing_image_count"
        ],
        "feature_schema_sha256": canonical_sha256(feature_schema),
        "feature_leakage_audit_valid": leakage_audit["valid"],
        "calibration_fitted": False,
        "router_dataset_materialized": False,
        "validated_oof_producer_count": 16,
    }


def _ownership_receipt_path(bundle_root: Path) -> Path:
    return bundle_root.parent / OWNERSHIP_RECEIPT_FILENAME


def _ownership_receipt_payload(
    *,
    bundle_root: Path,
    protocol_id: str,
    nonce: str,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": OWNERSHIP_RECEIPT_SCHEMA_VERSION,
        "owner": "M3_targets_and_feature_contract",
        "protocol_id": protocol_id,
        "bundle_directory": bundle_root.name,
        "owned_uncommitted_file_allowlist": list(BUNDLE_CHILD_FILENAMES),
        "commit_marker": MANIFEST_FILENAME,
        "nonce": nonce,
    }
    receipt["receipt_sha256"] = _manifest_self_hash(receipt)
    return receipt


def _validate_ownership_receipt(
    path: Path,
    *,
    bundle_root: Path,
    protocol_id: str,
) -> dict[str, Any]:
    if path.is_symlink():
        raise M3ArtifactError(f"M3 ownership receipt must not be a symlink: {path}")
    receipt = _read_json_mapping(path, name="M3 ownership receipt")
    nonce = receipt.get("nonce")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise M3ArtifactError("M3 ownership receipt has an invalid nonce")
    expected = _ownership_receipt_payload(
        bundle_root=bundle_root,
        protocol_id=protocol_id,
        nonce=nonce,
    )
    if receipt != expected:
        raise M3ArtifactError("M3 ownership receipt is malformed or stale")
    return receipt


def _create_ownership_receipt(
    path: Path,
    *,
    bundle_root: Path,
    protocol_id: str,
) -> dict[str, Any]:
    receipt = _ownership_receipt_payload(
        bundle_root=bundle_root,
        protocol_id=protocol_id,
        nonce=os.urandom(16).hex(),
    )
    _write_json_exclusive(path, receipt)
    return receipt


def _discard_owned_uncommitted_bundle(
    bundle_root: Path,
    *,
    receipt_path: Path,
    protocol_id: str,
) -> None:
    """Clean a partial bundle only when an external ownership receipt proves it."""

    if not bundle_root.exists():
        if receipt_path.exists():
            _validate_ownership_receipt(
                receipt_path,
                bundle_root=bundle_root,
                protocol_id=protocol_id,
            )
        return
    if bundle_root.is_symlink():
        raise M3ArtifactError(f"M3 bundle root must not be a symlink: {bundle_root}")
    if not bundle_root.is_dir():
        raise M3ArtifactError(f"M3 bundle path is not a directory: {bundle_root}")
    if (bundle_root / MANIFEST_FILENAME).exists():
        return
    if not receipt_path.is_file():
        raise M3ArtifactError(
            "uncommitted M3 bundle has no valid external ownership receipt; "
            "refusing cleanup"
        )
    _validate_ownership_receipt(
        receipt_path,
        bundle_root=bundle_root,
        protocol_id=protocol_id,
    )
    entries = list(bundle_root.iterdir())
    unknown = sorted(
        path.name
        for path in entries
        if path.name not in BUNDLE_CHILD_FILENAMES
        or path.is_symlink()
        or not path.is_file()
    )
    if unknown:
        raise M3ArtifactError(
            "uncommitted M3 directory contains unowned entries and cannot be cleaned: "
            f"{unknown}"
        )
    for path in entries:
        path.unlink()


def _exclusive_publish(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(staged, 0o444)
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise M3ArtifactError(f"immutable artifact already exists: {destination}") from exc
    staged.unlink()


@contextmanager
def _exclusive_workflow_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise M3ArtifactError(f"another M3 process owns workflow lock: {path}") from exc
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


def _write_new_bundle(
    prepared: _PreparedM3,
    *,
    receipt_path: Path,
) -> dict[str, Any]:
    bundle_root = _bundle_root(prepared)
    protocol_id = str(prepared.context.config["protocol_id"])
    _validate_ownership_receipt(
        receipt_path,
        bundle_root=bundle_root,
        protocol_id=protocol_id,
    )
    if bundle_root.is_symlink():
        raise M3ArtifactError(f"M3 bundle root must not be a symlink: {bundle_root}")
    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=False, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".targets_and_feature_contract.m3.staging-", dir=bundle_root.parent)
    ).resolve()
    try:
        _write_target_parquet(staging / TARGET_FILENAME, prepared.targets)
        _write_json_exclusive(staging / FEATURE_SCHEMA_FILENAME, prepared.feature_schema)
        _write_json_exclusive(staging / PREVALENCE_FILENAME, prepared.prevalence)
        _write_json_exclusive(staging / LEAKAGE_AUDIT_FILENAME, prepared.leakage_audit)
        manifest = _build_m3_manifest(prepared, bundle_root=staging)
        _write_json_exclusive(staging / MANIFEST_FILENAME, manifest)
        for name in BUNDLE_CHILD_FILENAMES:
            _exclusive_publish(staging / name, bundle_root / name)
        # The manifest is the immutable commit marker and is published last.
        _exclusive_publish(
            staging / MANIFEST_FILENAME, bundle_root / MANIFEST_FILENAME
        )
        # A crash after the preceding line leaves a committed valid bundle and
        # a known receipt; the next build validates both before removing it.
        receipt_path.unlink()
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    result = _validate_prepared_bundle(prepared)
    result["status"] = "created"
    return result


def build_m3_bundle(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build or strictly reuse the immutable real M3 artifact bundle."""

    prepared = _prepare_m3(
        config_path=config_path,
        protocol_dir=protocol_dir,
        artifact_root=artifact_root,
    )
    bundle_root = _bundle_root(prepared)
    lock_path = bundle_root.parent / ".m3_targets_and_feature_contract.lock"
    receipt_path = _ownership_receipt_path(bundle_root)
    protocol_id = str(prepared.context.config["protocol_id"])
    with _exclusive_workflow_lock(lock_path):
        if (bundle_root / MANIFEST_FILENAME).exists():
            result = _validate_prepared_bundle(prepared)
            if receipt_path.exists():
                _validate_ownership_receipt(
                    receipt_path,
                    bundle_root=bundle_root,
                    protocol_id=protocol_id,
                )
                receipt_path.unlink()
            return result
        router_dataset_path = bundle_root.parent / "router_dataset.parquet"
        if router_dataset_path.exists() or router_dataset_path.is_symlink():
            raise M3ArtifactError(
                "router_dataset.parquet must be absent at initial M3 publication; "
                "it is owned by M4"
            )
        _discard_owned_uncommitted_bundle(
            bundle_root,
            receipt_path=receipt_path,
            protocol_id=protocol_id,
        )
        if not receipt_path.exists():
            _create_ownership_receipt(
                receipt_path,
                bundle_root=bundle_root,
                protocol_id=protocol_id,
            )
        return _write_new_bundle(prepared, receipt_path=receipt_path)


def validate_m3_bundle(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Revalidate M1/M2 lineage and every committed M3 child artifact."""

    prepared = _prepare_m3(
        config_path=config_path,
        protocol_dir=protocol_dir,
        artifact_root=artifact_root,
    )
    return _validate_prepared_bundle(prepared)


def load_validated_m3_bundle(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> ValidatedM3Bundle:
    """Load M3 payloads only after full upstream and local validation."""

    prepared = _prepare_m3(
        config_path=config_path,
        protocol_dir=protocol_dir,
        artifact_root=artifact_root,
    )
    validation = _validate_prepared_bundle(prepared)
    root = _bundle_root(prepared)
    targets = _read_target_parquet(root / TARGET_FILENAME)[1]
    return ValidatedM3Bundle(
        root=root,
        targets=targets,
        feature_schema=_read_json_mapping(
            root / FEATURE_SCHEMA_FILENAME, name="router feature schema"
        ),
        target_prevalence=_read_json_mapping(
            root / PREVALENCE_FILENAME, name="target prevalence report"
        ),
        feature_leakage_audit=_read_json_mapping(
            root / LEAKAGE_AUDIT_FILENAME, name="feature leakage audit"
        ),
        manifest=_read_json_mapping(root / MANIFEST_FILENAME, name="M3 manifest"),
        validation=validation,
    )


__all__ = [
    "BOOLEAN_FEATURES",
    "CALIBRATED_PROBABILITY_BASIS",
    "CATEGORICAL_FEATURES",
    "DEFAULT_CONFIG_PATH",
    "FEATURE_COLUMNS",
    "FEATURE_FAMILIES",
    "FEATURE_SCHEMA_VERSION",
    "INTEGER_FEATURES",
    "LEAKAGE_AUDIT_SCHEMA_VERSION",
    "M3ArtifactError",
    "M3Error",
    "NATIVE_T1_PROBABILITY_BASIS",
    "NUMERIC_FEATURES",
    "PREVALENCE_SCHEMA_VERSION",
    "TARGET_COLUMNS",
    "TARGET_KEY",
    "TARGET_ORDER",
    "TARGET_SCHEMA_VERSION",
    "ValidatedM3Bundle",
    "build_feature_leakage_audit",
    "build_m3_bundle",
    "build_router_feature_frame",
    "build_router_feature_schema",
    "build_router_target_table",
    "build_target_prevalence_report",
    "derive_image_relative_states",
    "derive_router_target_states",
    "load_validated_m3_bundle",
    "m3_bundle_path",
    "router_target_content_sha256",
    "validate_feature_leakage_audit",
    "validate_m3_bundle",
    "validate_router_feature_frame",
    "validate_router_feature_schema",
    "validate_router_target_table",
]

"""Performance reporting for sealed development-train OOF expert outputs.

This module deliberately knows nothing about development-validation outputs.  It
joins labels from the sealed development assignments by ``row_uid`` and reports
only out-of-fold development-train performance.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


_SCALAR_METRICS = ("top1_acc", "top3_acc", "weighted_f1", "macro_f1", "mcc")
_ASSIGNMENT_COLUMNS = {"row_uid", "label_id_dense", "development_role"}


class OOFReportError(ValueError):
    """Raised when an OOF table cannot be scored without ambiguity or leakage."""


def _as_dataframe(value: Any, *, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=False)
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if not path.is_file():
            raise OOFReportError(f"{name} does not exist: {path}")
        if path.suffix.lower() not in {".parquet", ".pq"}:
            raise OOFReportError(f"{name} must be a Parquet file: {path}")
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - backend text is environment-specific
            raise OOFReportError(f"Could not read {name} Parquet file {path}: {exc}") from exc
    to_pandas = getattr(value, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        if isinstance(frame, pd.DataFrame):
            return frame
    raise OOFReportError(
        f"{name} must be a pandas DataFrame, a PyArrow-like table, or a Parquet path"
    )


def _validate_row_uids(frame: pd.DataFrame, *, name: str) -> pd.Series:
    if "row_uid" not in frame.columns:
        raise OOFReportError(f"{name} is missing required column 'row_uid'")
    values = frame["row_uid"]
    if bool(values.isna().any()):
        raise OOFReportError(f"{name} contains null row_uid values")
    if not bool(values.map(lambda value: isinstance(value, str)).all()):
        raise OOFReportError(f"{name} row_uid values must be strings")
    if bool(values.map(lambda value: not value or value != value.strip()).any()):
        raise OOFReportError(f"{name} contains empty or non-canonical row_uid values")
    return values.astype("string")


def _integer_array(
    values: pd.Series,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        raise OOFReportError(f"{name} must contain integers, not booleans")
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise OOFReportError(f"{name} must contain integers") from exc
    if numeric.ndim != 1 or not bool(np.isfinite(numeric).all()):
        raise OOFReportError(f"{name} must contain finite integers")
    if not bool(np.equal(numeric, np.floor(numeric)).all()):
        raise OOFReportError(f"{name} must contain integers")
    if minimum is not None and bool((numeric < minimum).any()):
        raise OOFReportError(f"{name} contains a value below {minimum}")
    if maximum is not None and bool((numeric > maximum).any()):
        raise OOFReportError(f"{name} contains a value above {maximum}")
    return numeric.astype(np.int64)


def _logit_matrix(values: pd.Series, *, name: str, class_count: int) -> np.ndarray:
    matrix = np.empty((len(values), class_count), dtype=np.float64)
    for position, value in enumerate(values):
        if isinstance(value, (str, bytes)):
            raise OOFReportError(f"{name} row {position} is not a numeric vector")
        try:
            row = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise OOFReportError(f"{name} row {position} is not a numeric vector") from exc
        if row.shape != (class_count,):
            raise OOFReportError(
                f"{name} row {position} has shape {row.shape}; expected ({class_count},)"
            )
        if not bool(np.isfinite(row).all()):
            raise OOFReportError(f"{name} row {position} contains a non-finite value")
        matrix[position] = row
    return matrix


def _validated_modes(modes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(modes, (str, bytes)):
        raise OOFReportError("modes must be a sequence of mode names, not a string")
    normalized = tuple(modes)
    if not normalized:
        raise OOFReportError("modes must not be empty")
    if any(not isinstance(mode, str) or not mode or mode != mode.strip() for mode in normalized):
        raise OOFReportError("every mode must be a non-empty canonical string")
    if len(set(normalized)) != len(normalized):
        raise OOFReportError("modes contains duplicate names")
    return normalized


def _metric_record(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    logits: np.ndarray,
    class_count: int,
) -> dict[str, Any]:
    labels = np.arange(class_count, dtype=np.int64)
    top_k = min(3, class_count)
    # Mergesort gives deterministic dense-ID ordering for exactly tied logits.
    top = np.argsort(-logits, axis=1, kind="mergesort")[:, :top_k]
    matrix = confusion_matrix(truth, prediction, labels=labels)
    return {
        "top1_acc": float(np.mean(prediction == truth)),
        "top3_acc": float(np.mean(np.any(top == truth[:, None], axis=1))),
        "weighted_f1": float(
            f1_score(
                truth,
                prediction,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "confusion_matrix": matrix.astype(np.int64, copy=False).tolist(),
    }


def build_oof_reproduction_report(
    oof_table_or_dataframe: Any,
    assignments_dataframe: Any,
    *,
    dense_class_count: int = 18,
    modes: Sequence[str] = ("image", "geo", "raw"),
) -> dict[str, Any]:
    """Build a JSON-serializable development-train OOF performance report.

    Each observed training seed must contain exactly one record for every
    development-train ``row_uid`` and no other identity.  Labels are accepted
    exclusively from ``assignments_dataframe``; a label column in the OOF table
    is rejected.  For every mode, serialized predictions must agree with the
    dense-ID argmax of its finite, fixed-width logits.

    Cross-seed standard deviations use the population definition (``ddof=0``).
    """

    if isinstance(dense_class_count, bool) or not isinstance(dense_class_count, int):
        raise OOFReportError("dense_class_count must be a positive integer")
    if dense_class_count <= 0:
        raise OOFReportError("dense_class_count must be a positive integer")
    mode_names = _validated_modes(modes)

    oof = _as_dataframe(oof_table_or_dataframe, name="OOF table")
    assignments = _as_dataframe(assignments_dataframe, name="assignments table")
    if oof.empty:
        raise OOFReportError("OOF table is empty")

    missing_assignments = sorted(_ASSIGNMENT_COLUMNS.difference(assignments.columns))
    if missing_assignments:
        raise OOFReportError(
            "assignments table is missing required columns: " + ", ".join(missing_assignments)
        )
    required_oof = {"row_uid", "training_seed"}.union(
        {f"{mode}_{suffix}" for mode in mode_names for suffix in ("logits", "pred")}
    )
    missing_oof = sorted(required_oof.difference(oof.columns))
    if missing_oof:
        raise OOFReportError("OOF table is missing required columns: " + ", ".join(missing_oof))
    forbidden_labels = sorted(
        column
        for column in oof.columns
        if column == "label"
        or column == "label_id_dense"
        or column.startswith("label_")
        or column.endswith("_correct")
    )
    if forbidden_labels:
        raise OOFReportError(
            "OOF table must remain label-blind; forbidden columns: "
            + ", ".join(forbidden_labels)
        )

    assignments = assignments.copy()
    assignments["row_uid"] = _validate_row_uids(assignments, name="assignments table")
    if bool(assignments["row_uid"].duplicated().any()):
        raise OOFReportError("assignments table contains duplicate row_uid values")
    if bool(assignments["development_role"].isna().any()):
        raise OOFReportError("assignments table contains null development_role values")
    role_values = set(assignments["development_role"].astype(str))
    if not role_values.issubset({"train", "validation"}):
        raise OOFReportError(
            "assignments table contains an unknown development_role: "
            + ", ".join(sorted(role_values.difference({"train", "validation"})))
        )
    train_assignments = assignments.loc[
        assignments["development_role"].astype(str).eq("train"),
        ["row_uid", "label_id_dense"],
    ].copy()
    if train_assignments.empty:
        raise OOFReportError("assignments table contains no development-train rows")
    train_assignments["label_id_dense"] = _integer_array(
        train_assignments["label_id_dense"],
        name="assignments label_id_dense",
        minimum=0,
        maximum=dense_class_count - 1,
    )

    oof = oof.copy()
    oof["row_uid"] = _validate_row_uids(oof, name="OOF table")
    oof["training_seed"] = _integer_array(
        oof["training_seed"], name="OOF training_seed"
    )
    if bool(oof[["row_uid", "training_seed"]].duplicated().any()):
        raise OOFReportError("OOF table contains duplicate (row_uid, training_seed) records")

    expected_rows = set(train_assignments["row_uid"].astype(str))
    seed_values = sorted(int(value) for value in oof["training_seed"].unique())
    if not seed_values:
        raise OOFReportError("OOF table contains no training seeds")
    for seed in seed_values:
        observed_rows = set(
            oof.loc[oof["training_seed"].eq(seed), "row_uid"].astype(str)
        )
        if observed_rows != expected_rows:
            missing = sorted(expected_rows.difference(observed_rows))[:5]
            extra = sorted(observed_rows.difference(expected_rows))[:5]
            raise OOFReportError(
                f"training seed {seed} does not exactly cover development-train identities; "
                f"missing={missing}, extra={extra}"
            )

    label_by_uid = train_assignments.set_index("row_uid")["label_id_dense"]
    oof["label_id_dense"] = oof["row_uid"].map(label_by_uid)
    if bool(oof["label_id_dense"].isna().any()):  # defensive after exact set validation
        raise OOFReportError("OOF label join was not one-to-one and complete")
    truth_all = _integer_array(
        oof["label_id_dense"],
        name="joined label_id_dense",
        minimum=0,
        maximum=dense_class_count - 1,
    )

    predictions: dict[str, np.ndarray] = {}
    logits_by_mode: dict[str, np.ndarray] = {}
    for mode in mode_names:
        prediction = _integer_array(
            oof[f"{mode}_pred"],
            name=f"OOF {mode}_pred",
            minimum=0,
            maximum=dense_class_count - 1,
        )
        logits = _logit_matrix(
            oof[f"{mode}_logits"],
            name=f"OOF {mode}_logits",
            class_count=dense_class_count,
        )
        argmax = np.argmax(logits, axis=1)
        if not bool(np.array_equal(prediction, argmax)):
            mismatch = int(np.flatnonzero(prediction != argmax)[0])
            raise OOFReportError(
                f"OOF {mode}_pred disagrees with {mode}_logits argmax at row {mismatch}"
            )
        predictions[mode] = prediction
        logits_by_mode[mode] = logits

    seed_array = oof["training_seed"].to_numpy(dtype=np.int64)
    per_seed: dict[str, Any] = {}
    for seed in seed_values:
        selected = seed_array == seed
        seed_record: dict[str, Any] = {"row_count": int(selected.sum()), "modes": {}}
        for mode in mode_names:
            seed_record["modes"][mode] = _metric_record(
                truth=truth_all[selected],
                prediction=predictions[mode][selected],
                logits=logits_by_mode[mode][selected],
                class_count=dense_class_count,
            )
        per_seed[str(seed)] = seed_record

    cross_seed: dict[str, Any] = {}
    for mode in mode_names:
        mode_summary: dict[str, dict[str, float]] = {}
        for metric in _SCALAR_METRICS:
            values = np.asarray(
                [per_seed[str(seed)]["modes"][mode][metric] for seed in seed_values],
                dtype=np.float64,
            )
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            if not math.isfinite(mean) or not math.isfinite(std):
                raise OOFReportError(f"non-finite cross-seed summary for {mode}.{metric}")
            mode_summary[metric] = {"mean": mean, "std": std}
        cross_seed[mode] = mode_summary

    return {
        "report_type": "development_train_oof_reproduction_performance",
        "dense_class_count": dense_class_count,
        "modes": list(mode_names),
        "training_seeds": seed_values,
        "row_count": int(len(oof)),
        "unique_row_count": int(len(expected_rows)),
        "cross_seed_std_definition": "population",
        "per_seed": per_seed,
        "cross_seed": cross_seed,
    }


def write_report_atomic(report: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Serialize ``report`` as strict JSON and atomically create ``path``.

    The temporary file is created in the destination directory so that
    a hard-link publication is atomic and refuses overwrite. Non-finite JSON
    numbers are rejected.
    """

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(
                report,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OOFReportError(f"report is not strict JSON serializable: {exc}") from exc

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o444)
        os.link(temporary_path, destination)
        temporary_path.unlink()
    except FileExistsError as exc:
        temporary_path.unlink(missing_ok=True)
        raise FileExistsError(f"immutable report already exists: {destination}") from exc
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


__all__ = ["OOFReportError", "build_oof_reproduction_report", "write_report_atomic"]

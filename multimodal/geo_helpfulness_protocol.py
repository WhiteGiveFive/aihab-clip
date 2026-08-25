"""Leakage-safe identities, assignments, and fingerprints for protocol v1.

This module is deliberately independent of the multimodal training and test
loading code.  It operates only on caller-provided metadata tables and never
resolves a dataset path by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


CANONICAL_JSON_VERSION = "geo_helpfulness.canonical_json.v1"
IDENTITY_SCHEMA_VERSION = "geo_helpfulness.identity_projection.v1"
LOCKED_TEST_MANIFEST_SCHEMA_VERSION = "geo_helpfulness.locked_test_identity.v1"
ASSIGNMENT_SCHEMA_VERSION = "geo_helpfulness.development_assignments.v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "geo_helpfulness.artifact_manifest.v1"

DEFAULT_PROTOCOL_ID = "protocol_v1"
DEFAULT_ROLE_SEED = 20260824
DEFAULT_OOF_SEED = 20261824
DEFAULT_VALIDATION_PLOTS = 325
DEFAULT_N_OOF_FOLDS = 4

# Tuple form is intentional: it is simple to compare byte-for-byte with the
# frozen YAML ontology and cannot be inferred from any development/test table.
FIXED_CLASS_ONTOLOGY = (
    (0, 0, "Urban"),
    (1, 1, "Broadleaved Mixed and Yew Woodland"),
    (2, 2, "Coniferous Woodland"),
    (3, 4, "Arable and Horticulture"),
    (4, 5, "Improved Grassland"),
    (5, 6, "Neutral Grassland"),
    (6, 7, "Calcareous Grassland"),
    (7, 8, "Acid Grassland"),
    (8, 9, "Bracken"),
    (9, 10, "Dwarf Shrub Heath"),
    (10, 11, "Fen, Marsh, Swamp"),
    (11, 12, "Bog"),
    (12, 13, "Littoral Rock"),
    (13, 14, "Littoral Sediment"),
    (14, 15, "Montane"),
    (15, 17, "Inland Rock"),
    (16, 18, "Supra-littoral Rock"),
    (17, 19, "Supra-littoral Sediment"),
)
DENSE_TO_CANONICAL_L3 = {dense: canonical for dense, canonical, _ in FIXED_CLASS_ONTOLOGY}
DENSE_TO_LABEL_NAME = {dense: name for dense, _, name in FIXED_CLASS_ONTOLOGY}
CANONICAL_L3_TO_DENSE = {canonical: dense for dense, canonical, _ in FIXED_CLASS_ONTOLOGY}
FIXED_OUTPUT_SIZE = len(FIXED_CLASS_ONTOLOGY)

IDENTITY_PROJECTION_COLUMNS = (
    "row_uid",
    "file",
    "file_lower",
    "plot_idx",
    "normalized_plot_idx",
)
LOCKED_TEST_IDENTITY_RULES = {
    "file": "unicode_nfc_relative_path_casefold_unique",
    "plot_idx": "opaque_case_sensitive_unicode_nfc_text",
    "row_uid_fields": ["dataset_id", "file_lower", "normalized_plot_idx"],
}
LOCKED_TEST_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_role",
        "dataset_id",
        "canonical_json_version",
        "identity_rules",
        "identity_projection_columns",
        "identity_projection_sha256",
        "snapshot_id",
        "row_count",
        "plot_count",
        "rows",
        "manifest_sha256",
    }
)
DEVELOPMENT_ASSIGNMENT_COLUMNS = (
    "schema_version",
    "protocol_id",
    "row_uid",
    "file",
    "file_lower",
    "plot_idx",
    "source_split",
    "image_source",
    "label_id_dense",
    "canonical_l3_id",
    "label_name",
    "development_role",
    "train_oof_fold",
)


class GeoHelpfulnessProtocolError(ValueError):
    """Base class for fail-closed protocol validation errors."""


class IdentityValidationError(GeoHelpfulnessProtocolError):
    """An identity is missing, unsafe, or non-canonical."""


class IdentityCollisionError(IdentityValidationError):
    """Distinct source identities collapse to one canonical identity."""


class OntologyValidationError(GeoHelpfulnessProtocolError):
    """Labels do not match the fixed, test-independent 18-class ontology."""


class AssignmentValidationError(GeoHelpfulnessProtocolError):
    """A role/fold assignment violates the grouped protocol."""


class IdentityOverlapError(AssignmentValidationError):
    """Development identities overlap the locked-test denylist."""


class ManifestValidationError(GeoHelpfulnessProtocolError):
    """An artifact manifest is malformed or stale."""


class FingerprintMismatchError(ManifestValidationError):
    """Observed content does not match its frozen SHA-256 fingerprint."""


class ArtifactParentRoleError(ManifestValidationError):
    """An artifact declares a forbidden upstream role."""


class ArtifactAlreadyExistsError(GeoHelpfulnessProtocolError, FileExistsError):
    """An immutable artifact would be overwritten."""


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    logical_type: str
    nullable: bool = False


DEVELOPMENT_ASSIGNMENT_SCHEMA = (
    ColumnSchema("schema_version", "string"),
    ColumnSchema("protocol_id", "string"),
    ColumnSchema("row_uid", "sha256_hex"),
    ColumnSchema("file", "canonical_relative_path"),
    ColumnSchema("file_lower", "unicode_casefold"),
    ColumnSchema("plot_idx", "opaque_nfc_string"),
    ColumnSchema("source_split", "string"),
    ColumnSchema("image_source", "string"),
    ColumnSchema("label_id_dense", "int8"),
    ColumnSchema("canonical_l3_id", "int8"),
    ColumnSchema("label_name", "string"),
    ColumnSchema("development_role", "enum:train,validation"),
    ColumnSchema("train_oof_fold", "nullable_int8", nullable=True),
)


_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:/")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_nfc_text(value: Any, *, field: str) -> str:
    if value is None or value is pd.NA:
        raise IdentityValidationError(f"{field} must be non-null text")
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise IdentityValidationError(f"{field} is not valid UTF-8") from exc
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text, got {type(value).__name__}")
    text = unicodedata.normalize("NFC", value)
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise IdentityValidationError(f"{field} is not valid Unicode text") from exc
    if "\x00" in text:
        raise IdentityValidationError(f"{field} contains NUL")
    return text


def _require_identifier(value: Any, *, field: str) -> str:
    text = _require_nfc_text(value, field=field)
    if not text or text != text.strip():
        raise IdentityValidationError(f"{field} must be nonempty without surrounding whitespace")
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in text):
        raise IdentityValidationError(f"{field} contains a control character")
    return text


def canonicalize_file(value: Any) -> str:
    """Return the frozen relative-path identity for a file value."""

    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    text = _require_nfc_text(value, field="file").replace("\\", "/")
    if not text:
        raise IdentityValidationError("file must not be empty")
    if text.startswith("/") or text.startswith("//") or _WINDOWS_ABSOLUTE_RE.match(text):
        raise IdentityValidationError(f"file must be relative: {text!r}")
    components = []
    for component in text.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            raise IdentityValidationError("file must not contain '..' components")
        components.append(component)
    if not components:
        raise IdentityValidationError("file has no usable path components")
    canonical = "/".join(components)
    if _WINDOWS_ABSOLUTE_RE.match(canonical):
        raise IdentityValidationError(f"file must be relative: {canonical!r}")
    return canonical


def canonicalize_plot_idx(value: Any) -> str:
    """Canonicalize an opaque, case-sensitive plot identifier.

    Plot identifiers are never parsed as integers and are never trimmed or
    case-folded.  Surrounding whitespace and control characters are rejected.
    """

    text = _require_nfc_text(value, field="plot_idx")
    if not text:
        raise IdentityValidationError("plot_idx must not be empty")
    if text != text.strip():
        raise IdentityValidationError("plot_idx must not have surrounding whitespace")
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in text):
        raise IdentityValidationError("plot_idx contains a control character")
    return text


def _canonical_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_json_value(asdict(value))
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise TypeError("canonical JSON object keys must be strings")
            key = unicodedata.normalize("NFC", raw_key)
            if key in normalized:
                raise ValueError(f"canonical JSON key collision after NFC normalization: {key!r}")
            normalized[key] = _canonical_json_value(raw_value)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets are not valid canonical JSON values")
    if isinstance(value, Path):
        return unicodedata.normalize("NFC", value.as_posix())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _canonical_json_value(value.item())
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON forbids NaN and infinity")
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _canonical_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_dumps(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_row_uid(dataset_id: Any, file: Any, plot_idx: Any) -> str:
    dataset = _require_identifier(dataset_id, field="dataset_id")
    canonical_file = canonicalize_file(file)
    canonical_plot = canonicalize_plot_idx(plot_idx)
    return canonical_sha256([dataset, canonical_file.casefold(), canonical_plot])


def _check_plot_normalization_collisions(raw_values: Iterable[Any]) -> list[str]:
    raw_by_canonical: dict[str, set[str]] = {}
    canonical_values: list[str] = []
    for raw in raw_values:
        canonical = canonicalize_plot_idx(raw)
        raw_text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        raw_by_canonical.setdefault(canonical, set()).add(raw_text)
        canonical_values.append(canonical)
    collisions = {
        canonical: sorted(raws)
        for canonical, raws in raw_by_canonical.items()
        if len(raws) > 1
    }
    if collisions:
        preview = list(collisions.items())[:5]
        raise IdentityCollisionError(
            f"plot_idx normalization collision after Unicode NFC: {preview}"
        )
    return canonical_values


def build_identity_projection(
    frame: pd.DataFrame,
    dataset_id: str,
    file_column: str = "file",
    plot_column: str = "plot_idx",
) -> pd.DataFrame:
    """Project an arbitrary table onto sorted, label-blind identities."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in (file_column, plot_column) if column not in frame]
    if missing:
        raise IdentityValidationError(f"identity table is missing columns: {missing}")
    dataset = _require_identifier(dataset_id, field="dataset_id")
    files = [canonicalize_file(value) for value in frame[file_column].tolist()]
    plots = _check_plot_normalization_collisions(frame[plot_column].tolist())
    file_lower = [value.casefold() for value in files]
    duplicated = pd.Series(file_lower).duplicated(keep=False)
    if bool(duplicated.any()):
        examples = sorted(set(pd.Series(file_lower)[duplicated].tolist()))[:10]
        raise IdentityCollisionError(
            f"file identities must be unique after canonicalization/casefold; collision: {examples}"
        )
    row_uids = [make_row_uid(dataset, file, plot) for file, plot in zip(files, plots)]
    if len(set(row_uids)) != len(row_uids):
        raise IdentityCollisionError("row_uid collision or duplicate identity")
    projection = pd.DataFrame(
        {
            "row_uid": pd.Series(row_uids, dtype="string"),
            "file": pd.Series(files, dtype="string"),
            "file_lower": pd.Series(file_lower, dtype="string"),
            "plot_idx": pd.Series(plots, dtype="string"),
            "normalized_plot_idx": pd.Series(plots, dtype="string"),
        }
    )
    return projection.sort_values("row_uid", kind="mergesort").reset_index(drop=True)


def _dataframe_payload(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    rows: list[list[Any]] = []
    for row in frame.itertuples(index=False, name=None):
        values: list[Any] = []
        for value in row:
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
            values.append(None if missing else _canonical_json_value(value))
        rows.append(values)
    # Content hashes are intentionally independent of physical row order.
    rows.sort(key=canonical_json_bytes)
    return {
        "kind": "dataframe",
        "columns": columns,
        "rows": rows,
    }


def content_sha256(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return canonical_sha256(_dataframe_payload(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(value)).hexdigest()
    return canonical_sha256(value)


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_dataframe(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = frame if columns is None else frame.loc[:, list(columns)]
    return content_sha256(selected)


def assignment_fingerprint(assignments: pd.DataFrame) -> str:
    missing = [column for column in DEVELOPMENT_ASSIGNMENT_COLUMNS if column not in assignments]
    if missing:
        raise AssignmentValidationError(f"assignment fingerprint missing columns: {missing}")
    return fingerprint_dataframe(assignments, DEVELOPMENT_ASSIGNMENT_COLUMNS)


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def _seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(manifest)
    sealed.pop("manifest_sha256", None)
    sealed["manifest_sha256"] = canonical_sha256(sealed)
    return sealed


def build_locked_test_identity_manifest(
    frame_or_projection: pd.DataFrame,
    dataset_id: str,
) -> dict[str, Any]:
    projection = build_identity_projection(frame_or_projection, dataset_id)
    row_columns = ("row_uid", "file", "file_lower", "plot_idx", "normalized_plot_idx")
    rows = projection.loc[:, row_columns].to_dict(orient="records")
    identity_hash = canonical_sha256(rows)
    manifest = {
        "schema_version": LOCKED_TEST_MANIFEST_SCHEMA_VERSION,
        "artifact_role": "locked_test_identity_manifest",
        "dataset_id": _require_identifier(dataset_id, field="dataset_id"),
        "canonical_json_version": CANONICAL_JSON_VERSION,
        "identity_rules": LOCKED_TEST_IDENTITY_RULES,
        "identity_projection_columns": list(row_columns),
        "identity_projection_sha256": identity_hash,
        "snapshot_id": identity_hash,
        "row_count": len(rows),
        "plot_count": int(projection["normalized_plot_idx"].nunique()),
        "rows": rows,
    }
    return _seal_manifest(manifest)


def validate_locked_test_identity_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("locked-test identity manifest must be a mapping")
    if set(manifest) != LOCKED_TEST_MANIFEST_FIELDS:
        missing = sorted(LOCKED_TEST_MANIFEST_FIELDS.difference(manifest))
        extra = sorted(set(manifest).difference(LOCKED_TEST_MANIFEST_FIELDS))
        raise ManifestValidationError(
            f"locked-test identity manifest fields mismatch: missing={missing}, extra={extra}"
        )
    if manifest.get("schema_version") != LOCKED_TEST_MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError("unsupported locked-test identity manifest schema")
    if manifest.get("artifact_role") != "locked_test_identity_manifest":
        raise ManifestValidationError("locked-test identity manifest has the wrong artifact role")
    if manifest.get("canonical_json_version") != CANONICAL_JSON_VERSION:
        raise ManifestValidationError("locked-test identity manifest canonical JSON version mismatch")
    if manifest.get("identity_rules") != LOCKED_TEST_IDENTITY_RULES:
        raise ManifestValidationError("locked-test identity rules do not match the frozen contract")
    if manifest.get("identity_projection_columns") != list(IDENTITY_PROJECTION_COLUMNS):
        raise ManifestValidationError(
            "locked-test identity projection columns do not match the frozen contract"
        )
    expected_manifest_hash = manifest.get("manifest_sha256")
    actual_manifest_hash = canonical_sha256(_manifest_payload(manifest))
    if expected_manifest_hash != actual_manifest_hash:
        raise FingerprintMismatchError("locked-test identity manifest hash mismatch")
    dataset_id = _require_identifier(manifest.get("dataset_id"), field="dataset_id")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ManifestValidationError("locked-test identity manifest rows must be a list")
    required = set(IDENTITY_PROJECTION_COLUMNS)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != required:
            raise ManifestValidationError(f"locked-test identity row {index} has invalid schema")
    frame = pd.DataFrame(rows, columns=IDENTITY_PROJECTION_COLUMNS)
    projection = build_identity_projection(frame, dataset_id)
    expected_rows = projection.loc[:, IDENTITY_PROJECTION_COLUMNS].to_dict(orient="records")
    if rows != expected_rows:
        raise ManifestValidationError("locked-test identity rows are not canonical and sorted")
    identity_hash = canonical_sha256(expected_rows)
    if manifest.get("identity_projection_sha256") != identity_hash:
        raise FingerprintMismatchError("locked-test identity projection hash mismatch")
    if manifest.get("snapshot_id") != identity_hash:
        raise FingerprintMismatchError("locked-test snapshot ID does not match identity hash")
    if manifest.get("row_count") != len(frame):
        raise ManifestValidationError("locked-test row count mismatch")
    plot_count = int(frame["normalized_plot_idx"].nunique()) if len(frame) else 0
    if manifest.get("plot_count") != plot_count:
        raise ManifestValidationError("locked-test plot count mismatch")
    return {
        "valid": True,
        "row_count": len(frame),
        "plot_count": plot_count,
        "identity_projection_sha256": identity_hash,
    }


def _coerce_dense_label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise OntologyValidationError("dense labels must be integers, not booleans")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OntologyValidationError(f"invalid dense label: {value!r}") from exc
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise OntologyValidationError(f"dense label is not integral: {value!r}")
    if numeric not in DENSE_TO_CANONICAL_L3:
        raise OntologyValidationError(f"dense label is outside the frozen ontology: {numeric}")
    return numeric


def _canonical_development_rows(frame: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    aliases = {
        "label_id_dense": "label_id" if "label_id" in frame else None,
        "source_split": "split" if "split" in frame else None,
    }
    required = ["file", "plot_idx", "label_name", "image_source"]
    missing = [column for column in required if column not in frame]
    for target, source in aliases.items():
        if target not in frame and source is None:
            missing.append(target)
    if missing:
        raise AssignmentValidationError(f"development table is missing columns: {sorted(set(missing))}")

    projection = build_identity_projection(frame, dataset_id)
    # Re-index canonical identities back through file_lower; projection is sorted.
    projection_by_file = projection.set_index("file_lower")
    raw_files = [canonicalize_file(value) for value in frame["file"].tolist()]
    file_lowers = [value.casefold() for value in raw_files]
    dense_source = frame["label_id_dense"] if "label_id_dense" in frame else frame[aliases["label_id_dense"]]
    split_source = frame["source_split"] if "source_split" in frame else frame[aliases["source_split"]]
    dense = [_coerce_dense_label(value) for value in dense_source.tolist()]
    observed_dense = set(dense)
    expected_dense = set(DENSE_TO_CANONICAL_L3)
    if observed_dense != expected_dense:
        raise OntologyValidationError(
            "development universe must cover the complete frozen 18-class ontology; "
            f"missing={sorted(expected_dense - observed_dense)}, "
            f"unexpected={sorted(observed_dense - expected_dense)}"
        )
    labels = [_require_nfc_text(value, field="label_name") for value in frame["label_name"].tolist()]
    expected_labels = [DENSE_TO_LABEL_NAME[value] for value in dense]
    if labels != expected_labels:
        mismatches = [
            (index, actual, expected)
            for index, (actual, expected) in enumerate(zip(labels, expected_labels))
            if actual != expected
        ][:10]
        raise OntologyValidationError(f"label names do not match the frozen ontology: {mismatches}")
    canonical_ids = [DENSE_TO_CANONICAL_L3[value] for value in dense]
    if "canonical_l3_id" in frame:
        observed = [_coerce_integer(value, field="canonical_l3_id") for value in frame["canonical_l3_id"]]
        bad = [
            (index, actual, expected)
            for index, (actual, expected) in enumerate(zip(observed, canonical_ids))
            if actual != expected
        ][:10]
        if bad:
            raise OntologyValidationError(f"canonical L3 IDs do not match dense IDs: {bad}")
    splits = [_require_identifier(value, field="source_split") for value in split_source.tolist()]
    image_sources = [_require_identifier(value, field="image_source") for value in frame["image_source"].tolist()]
    plots = _check_plot_normalization_collisions(frame["plot_idx"].tolist())
    work = pd.DataFrame(
        {
            "row_uid": [str(projection_by_file.loc[file_lower, "row_uid"]) for file_lower in file_lowers],
            "file": raw_files,
            "file_lower": file_lowers,
            "plot_idx": plots,
            "source_split": splits,
            "image_source": image_sources,
            "label_id_dense": dense,
            "canonical_l3_id": canonical_ids,
            "label_name": labels,
        }
    )
    grouped = work.groupby("plot_idx", sort=False)[["label_id_dense", "canonical_l3_id", "label_name"]].nunique()
    if bool((grouped > 1).any(axis=None)):
        bad_plots = grouped.index[(grouped > 1).any(axis=1)].astype(str).tolist()[:10]
        raise AssignmentValidationError(f"plot_idx spans multiple habitat labels: {bad_plots}")
    return work


def _coerce_integer(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise AssignmentValidationError(f"{field} must be an integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssignmentValidationError(f"{field} must be an integer") from exc
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise AssignmentValidationError(f"{field} must be an integer")
    return numeric


def _validation_quotas(plot_table: pd.DataFrame, target: int) -> dict[int, int]:
    counts = {
        int(class_id): int(count)
        for class_id, count in plot_table.groupby("canonical_l3_id").size().items()
    }
    lower = {class_id: 0 if count == 1 else 1 for class_id, count in counts.items()}
    upper = {class_id: 0 if count == 1 else count - 1 for class_id, count in counts.items()}
    if target < sum(lower.values()) or target > sum(upper.values()):
        raise AssignmentValidationError(
            f"validation_plot_count={target} is infeasible under rare-class bounds "
            f"[{sum(lower.values())}, {sum(upper.values())}]"
        )
    quota = {
        class_id: min(upper[class_id], max(lower[class_id], count // 5))
        for class_id, count in counts.items()
    }
    while sum(quota.values()) < target:
        eligible = [class_id for class_id in counts if quota[class_id] < upper[class_id]]
        chosen = sorted(eligible, key=lambda class_id: (-(counts[class_id] - 5 * quota[class_id]), class_id))[0]
        quota[chosen] += 1
    while sum(quota.values()) > target:
        eligible = [class_id for class_id in counts if quota[class_id] > lower[class_id]]
        chosen = sorted(eligible, key=lambda class_id: (-(5 * quota[class_id] - counts[class_id]), class_id))[0]
        quota[chosen] -= 1
    return quota


def _resolve_fold_args(
    *,
    oof_seed: int,
    n_oof_folds: int,
    fold_seed: int | None,
    n_folds: int | None,
) -> tuple[int, int]:
    if fold_seed is not None:
        if oof_seed != DEFAULT_OOF_SEED and int(fold_seed) != int(oof_seed):
            raise AssignmentValidationError("oof_seed and fold_seed disagree")
        oof_seed = int(fold_seed)
    if n_folds is not None:
        if n_oof_folds != DEFAULT_N_OOF_FOLDS and int(n_folds) != int(n_oof_folds):
            raise AssignmentValidationError("n_oof_folds and n_folds disagree")
        n_oof_folds = int(n_folds)
    if int(n_oof_folds) < 2:
        raise AssignmentValidationError("n_oof_folds must be at least 2")
    return int(oof_seed), int(n_oof_folds)


def _denylist_rows(manifest: Mapping[str, Any] | None) -> pd.DataFrame | None:
    if manifest is None:
        return None
    validate_locked_test_identity_manifest(manifest)
    return pd.DataFrame(manifest["rows"], columns=IDENTITY_PROJECTION_COLUMNS)


def _assert_no_test_overlap(work: pd.DataFrame, manifest: Mapping[str, Any] | None) -> None:
    denylist = _denylist_rows(manifest)
    if denylist is None:
        return
    checks = {
        "row_uid": set(work["row_uid"]).intersection(denylist["row_uid"]),
        "file_lower": set(work["file_lower"]).intersection(denylist["file_lower"]),
        "plot_idx": set(work["plot_idx"]).intersection(denylist["normalized_plot_idx"]),
    }
    overlap = {key: sorted(map(str, values))[:10] for key, values in checks.items() if values}
    if overlap:
        raise IdentityOverlapError(f"development/locked-test identity overlap: {overlap}")


def _build_assignments(
    frame: pd.DataFrame,
    *,
    protocol_id: str,
    dataset_id: str,
    role_seed: int,
    validation_plot_count: int,
    n_oof_folds: int,
    oof_seed: int,
    expected_rows: int | None,
    expected_plots: int | None,
    test_identity_manifest: Mapping[str, Any] | None,
) -> pd.DataFrame:
    protocol = _require_identifier(protocol_id, field="protocol_id")
    work = _canonical_development_rows(frame, dataset_id)
    if expected_rows is not None and len(work) != int(expected_rows):
        raise AssignmentValidationError(f"development row count {len(work)} != {expected_rows}")
    plot_count = int(work["plot_idx"].nunique())
    if expected_plots is not None and plot_count != int(expected_plots):
        raise AssignmentValidationError(f"development plot count {plot_count} != {expected_plots}")
    target = int(validation_plot_count)
    if target <= 0 or target >= plot_count:
        raise AssignmentValidationError("validation_plot_count must leave nonempty train and validation roles")
    _assert_no_test_overlap(work, test_identity_manifest)

    plots = (
        work[["plot_idx", "canonical_l3_id"]]
        .drop_duplicates()
        .sort_values(["canonical_l3_id", "plot_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    quotas = _validation_quotas(plots, target)
    validation_plots: set[str] = set()
    for class_id, group in plots.groupby("canonical_l3_id", sort=True):
        ranked = group.copy()
        ranked["stable_rank"] = [
            hashlib.sha256(
                f"{protocol}|{int(role_seed)}|{int(class_id)}|{plot}".encode("utf-8")
            ).hexdigest()
            for plot in ranked["plot_idx"]
        ]
        ranked = ranked.sort_values(["stable_rank", "plot_idx"], kind="mergesort")
        validation_plots.update(ranked.head(quotas[int(class_id)])["plot_idx"].tolist())
    work["development_role"] = np.where(
        work["plot_idx"].isin(validation_plots), "validation", "train"
    )

    train_plots = (
        work.loc[work["development_role"] == "train", ["plot_idx", "canonical_l3_id"]]
        .drop_duplicates()
        .sort_values(["canonical_l3_id", "plot_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(train_plots) < n_oof_folds:
        raise AssignmentValidationError("development-train has fewer plots than OOF folds")
    fold_by_plot: dict[str, int] = {}
    splitter = StratifiedKFold(
        n_splits=int(n_oof_folds), shuffle=True, random_state=int(oof_seed)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        for fold, (_, heldout_indices) in enumerate(
            splitter.split(np.zeros(len(train_plots)), train_plots["canonical_l3_id"])
        ):
            for index in heldout_indices:
                fold_by_plot[str(train_plots.iloc[int(index)]["plot_idx"])] = int(fold)
    work["train_oof_fold"] = pd.array(
        [
            fold_by_plot.get(str(plot)) if role == "train" else pd.NA
            for plot, role in zip(work["plot_idx"], work["development_role"])
        ],
        dtype="Int8",
    )
    work.insert(0, "protocol_id", protocol)
    work.insert(0, "schema_version", ASSIGNMENT_SCHEMA_VERSION)
    work["label_id_dense"] = work["label_id_dense"].astype("int8")
    work["canonical_l3_id"] = work["canonical_l3_id"].astype("int8")
    for column in (
        "schema_version", "protocol_id", "row_uid", "file", "file_lower",
        "plot_idx", "source_split", "image_source", "label_name", "development_role",
    ):
        work[column] = work[column].astype("string")
    return (
        work.loc[:, DEVELOPMENT_ASSIGNMENT_COLUMNS]
        .sort_values("row_uid", kind="mergesort")
        .reset_index(drop=True)
    )


def build_development_assignments(
    frame: pd.DataFrame,
    *,
    protocol_id: str = DEFAULT_PROTOCOL_ID,
    dataset_id: str,
    role_seed: int = DEFAULT_ROLE_SEED,
    validation_plot_count: int = DEFAULT_VALIDATION_PLOTS,
    n_oof_folds: int = DEFAULT_N_OOF_FOLDS,
    oof_seed: int = DEFAULT_OOF_SEED,
    expected_rows: int | None = None,
    expected_plots: int | None = None,
    test_identity_manifest: Mapping[str, Any] | None = None,
    locked_test_identity_manifest: Mapping[str, Any] | None = None,
    n_folds: int | None = None,
    fold_seed: int | None = None,
) -> pd.DataFrame:
    if test_identity_manifest is not None and locked_test_identity_manifest is not None:
        if canonical_sha256(test_identity_manifest) != canonical_sha256(locked_test_identity_manifest):
            raise AssignmentValidationError("test identity manifest aliases disagree")
    manifest = test_identity_manifest or locked_test_identity_manifest
    oof_seed, n_oof_folds = _resolve_fold_args(
        oof_seed=oof_seed,
        n_oof_folds=n_oof_folds,
        fold_seed=fold_seed,
        n_folds=n_folds,
    )
    return _build_assignments(
        frame,
        protocol_id=protocol_id,
        dataset_id=dataset_id,
        role_seed=int(role_seed),
        validation_plot_count=int(validation_plot_count),
        n_oof_folds=n_oof_folds,
        oof_seed=oof_seed,
        expected_rows=expected_rows,
        expected_plots=expected_plots,
        test_identity_manifest=manifest,
    )


def validate_development_assignments(
    assignments: pd.DataFrame,
    *,
    protocol_id: str = DEFAULT_PROTOCOL_ID,
    dataset_id: str,
    role_seed: int = DEFAULT_ROLE_SEED,
    validation_plot_count: int = DEFAULT_VALIDATION_PLOTS,
    n_oof_folds: int = DEFAULT_N_OOF_FOLDS,
    oof_seed: int = DEFAULT_OOF_SEED,
    expected_rows: int | None = None,
    expected_plots: int | None = None,
    test_identity_manifest: Mapping[str, Any] | None = None,
    locked_test_identity_manifest: Mapping[str, Any] | None = None,
    n_folds: int | None = None,
    fold_seed: int | None = None,
) -> dict[str, Any]:
    missing = [column for column in DEVELOPMENT_ASSIGNMENT_COLUMNS if column not in assignments]
    if missing:
        raise AssignmentValidationError(f"development assignments are missing columns: {missing}")
    if list(assignments.columns) != list(DEVELOPMENT_ASSIGNMENT_COLUMNS):
        raise AssignmentValidationError("development assignment columns/order do not match frozen schema")
    if assignments["row_uid"].duplicated().any() or assignments["file_lower"].duplicated().any():
        raise AssignmentValidationError("development row/file identities must be unique")
    if set(assignments["schema_version"].astype(str)) != {ASSIGNMENT_SCHEMA_VERSION}:
        raise AssignmentValidationError("development assignment schema_version mismatch")
    if set(assignments["protocol_id"].astype(str)) != {str(protocol_id)}:
        raise AssignmentValidationError("development assignment protocol_id mismatch")
    if set(assignments["development_role"].astype(str)) != {"train", "validation"}:
        raise AssignmentValidationError("development roles must contain exactly train and validation")
    train_mask = assignments["development_role"].astype(str) == "train"
    if assignments.loc[~train_mask, "train_oof_fold"].notna().any():
        raise AssignmentValidationError("validation rows must have null train_oof_fold")
    if assignments.loc[train_mask, "train_oof_fold"].isna().any():
        raise AssignmentValidationError("train rows must have an OOF fold")
    oof_seed, n_oof_folds = _resolve_fold_args(
        oof_seed=oof_seed, n_oof_folds=n_oof_folds, fold_seed=fold_seed, n_folds=n_folds
    )
    observed_folds = set(assignments.loc[train_mask, "train_oof_fold"].astype(int))
    if observed_folds != set(range(n_oof_folds)):
        raise AssignmentValidationError(f"train OOF folds mismatch: {sorted(observed_folds)}")
    if assignments.groupby("plot_idx")["development_role"].nunique().max() != 1:
        raise AssignmentValidationError("a plot crosses the train/validation boundary")
    if assignments.loc[train_mask].groupby("plot_idx")["train_oof_fold"].nunique().max() != 1:
        raise AssignmentValidationError("a train plot crosses OOF folds")
    validation_plots = int(assignments.loc[~train_mask, "plot_idx"].nunique())
    if validation_plots != int(validation_plot_count):
        raise AssignmentValidationError(
            f"validation plot count {validation_plots} != {validation_plot_count}"
        )
    manifest = test_identity_manifest or locked_test_identity_manifest
    _assert_no_test_overlap(assignments, manifest)

    source = assignments.loc[:, [
        "file", "plot_idx", "source_split", "image_source",
        "label_id_dense", "canonical_l3_id", "label_name",
    ]]
    regenerated = _build_assignments(
        source,
        protocol_id=protocol_id,
        dataset_id=dataset_id,
        role_seed=int(role_seed),
        validation_plot_count=int(validation_plot_count),
        n_oof_folds=n_oof_folds,
        oof_seed=oof_seed,
        expected_rows=expected_rows,
        expected_plots=expected_plots,
        test_identity_manifest=manifest,
    )
    comparable = assignments.copy()
    comparable["train_oof_fold"] = comparable["train_oof_fold"].astype("Int8")
    comparable = comparable.sort_values("row_uid", kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(regenerated, comparable, check_dtype=False)
    except AssertionError as exc:
        raise AssignmentValidationError("assignments do not regenerate from the frozen algorithm") from exc
    train_plot_count = int(assignments.loc[train_mask, "plot_idx"].nunique())
    fold_plot_counts = {
        str(int(fold)): int(count)
        for fold, count in (
            assignments.loc[train_mask, ["plot_idx", "train_oof_fold"]]
            .drop_duplicates()
            .groupby("train_oof_fold")
            .size()
            .items()
        )
    }
    return {
        "valid": True,
        "row_count": len(assignments),
        "plot_count": int(assignments["plot_idx"].nunique()),
        "train_plot_count": train_plot_count,
        "validation_plot_count": validation_plots,
        "development_train_plots": train_plot_count,
        "development_validation_plots": validation_plots,
        "n_oof_folds": n_oof_folds,
        "train_oof_fold_plot_counts": fold_plot_counts,
        "assignment_content_sha256": assignment_fingerprint(assignments),
    }


def plot_set_fingerprint(plot_ids: Iterable[Any]) -> str:
    """Fingerprint a logical set of canonical plot identities."""

    canonical = sorted({canonicalize_plot_idx(value) for value in plot_ids})
    if not canonical:
        raise AssignmentValidationError("plot identity set must not be empty")
    return canonical_sha256(canonical)


def validate_fit_prediction_plot_provenance(
    fitting_plot_ids: Iterable[Any],
    prediction_plot_ids: Iterable[Any],
) -> dict[str, Any]:
    """Seal the core honesty assertion for an OOF or validation producer."""

    fitting = {canonicalize_plot_idx(value) for value in fitting_plot_ids}
    prediction = {canonicalize_plot_idx(value) for value in prediction_plot_ids}
    if not fitting or not prediction:
        raise AssignmentValidationError(
            "fitting and prediction plot sets must both be non-empty"
        )
    overlap = fitting.intersection(prediction)
    if overlap:
        raise AssignmentValidationError(
            f"fitting/prediction plot overlap: {sorted(overlap)[:10]}"
        )
    return {
        "fitting_plot_count": len(fitting),
        "prediction_plot_count": len(prediction),
        "fitting_plot_sha256": canonical_sha256(sorted(fitting)),
        "prediction_plot_sha256": canonical_sha256(sorted(prediction)),
        "zero_plot_overlap": True,
    }


_ALLOWED_PARENT_ROLES = {
    "development_assignments": frozenset({"development_source", "locked_test_identity_manifest"}),
    "train_oof_fold_outputs": frozenset({"development_assignments", "fold_local_expert_fit"}),
    "development_train_oof_outputs": frozenset({"development_assignments", "train_oof_fold_outputs"}),
    "development_validation_outputs": frozenset({"development_assignments", "development_train_expert_fit"}),
    "router_training_dataset": frozenset({"development_train_oof_outputs"}),
    "router_candidate_validation_predictions": frozenset({"router_training_dataset", "development_validation_outputs"}),
    "development_selection_receipt": frozenset({"router_candidate_validation_predictions", "validation_targets"}),
    "final_expert_bundle": frozenset({"development_selection_receipt", "development_assignments"}),
    "frozen_router_bundle": frozenset({"development_selection_receipt"}),
    "composite_inference_bundle": frozenset({"final_expert_bundle", "frozen_router_bundle"}),
    "locked_test_predictions": frozenset({"composite_inference_bundle", "locked_test_identity_manifest"}),
    "locked_test_score_receipt": frozenset({"locked_test_predictions", "locked_test_labels"}),
}


def validate_artifact_parent_roles(artifact_role: str, parent_roles: Sequence[str]) -> None:
    role = _require_identifier(artifact_role, field="artifact_role")
    if role not in _ALLOWED_PARENT_ROLES:
        raise ArtifactParentRoleError(f"unknown artifact role: {role}")
    parents = [str(parent) for parent in parent_roles]
    disallowed = sorted(set(parents).difference(_ALLOWED_PARENT_ROLES[role]))
    if disallowed:
        raise ArtifactParentRoleError(
            f"artifact role {role!r} has forbidden parent role(s): {disallowed}"
        )


def build_artifact_manifest(
    *,
    artifact_role: str,
    protocol_id: str,
    payload: Any,
    parent_roles: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_artifact_parent_roles(artifact_role, parent_roles)
    manifest = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_role": artifact_role,
        "protocol_id": _require_identifier(protocol_id, field="protocol_id"),
        "content_sha256": content_sha256(payload),
        "parent_roles": list(parent_roles),
        "metadata": dict(metadata or {}),
    }
    return _seal_manifest(manifest)


def validate_artifact_manifest(manifest: Mapping[str, Any], *, payload: Any) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("artifact manifest must be a mapping")
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError("unsupported artifact manifest schema")
    expected_manifest = manifest.get("manifest_sha256")
    if expected_manifest != canonical_sha256(_manifest_payload(manifest)):
        raise FingerprintMismatchError("artifact manifest hash mismatch")
    validate_artifact_parent_roles(
        str(manifest.get("artifact_role")),
        list(manifest.get("parent_roles", [])),
    )
    actual_content = content_sha256(payload)
    if manifest.get("content_sha256") != actual_content:
        raise FingerprintMismatchError("artifact content hash mismatch; cache is stale")
    return {
        "valid": True,
        "artifact_role": manifest["artifact_role"],
        "content_sha256": actual_content,
    }


def exclusive_write_bytes(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise ArtifactAlreadyExistsError(f"immutable artifact already exists: {target}") from exc
    return target


def exclusive_write_json(path: str | Path, value: Any) -> Path:
    return exclusive_write_bytes(path, canonical_json_bytes(value) + b"\n")


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "ASSIGNMENT_SCHEMA_VERSION",
    "CANONICAL_JSON_VERSION",
    "CANONICAL_L3_TO_DENSE",
    "DEFAULT_N_OOF_FOLDS",
    "DEFAULT_OOF_SEED",
    "DEFAULT_PROTOCOL_ID",
    "DEFAULT_ROLE_SEED",
    "DEFAULT_VALIDATION_PLOTS",
    "DENSE_TO_CANONICAL_L3",
    "DENSE_TO_LABEL_NAME",
    "DEVELOPMENT_ASSIGNMENT_COLUMNS",
    "DEVELOPMENT_ASSIGNMENT_SCHEMA",
    "FIXED_CLASS_ONTOLOGY",
    "FIXED_OUTPUT_SIZE",
    "IDENTITY_PROJECTION_COLUMNS",
    "IDENTITY_SCHEMA_VERSION",
    "LOCKED_TEST_MANIFEST_SCHEMA_VERSION",
    "LOCKED_TEST_IDENTITY_RULES",
    "LOCKED_TEST_MANIFEST_FIELDS",
    "ArtifactAlreadyExistsError",
    "ArtifactParentRoleError",
    "AssignmentValidationError",
    "ColumnSchema",
    "FingerprintMismatchError",
    "GeoHelpfulnessProtocolError",
    "IdentityCollisionError",
    "IdentityOverlapError",
    "IdentityValidationError",
    "ManifestValidationError",
    "OntologyValidationError",
    "assignment_fingerprint",
    "build_artifact_manifest",
    "build_development_assignments",
    "build_identity_projection",
    "build_locked_test_identity_manifest",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "canonical_sha256",
    "canonicalize_file",
    "canonicalize_plot_idx",
    "content_sha256",
    "exclusive_write_bytes",
    "exclusive_write_json",
    "fingerprint_dataframe",
    "make_row_uid",
    "sha256_file",
    "validate_artifact_manifest",
    "validate_artifact_parent_roles",
    "validate_development_assignments",
    "validate_locked_test_identity_manifest",
]

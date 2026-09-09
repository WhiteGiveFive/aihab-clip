"""Versioned storage-dtype correction for the unchanged fixed router preview.

The original prediction bundle and v1 scoring code stay immutable. The v1 reader
still verifies that complete seal; this additive scorer changes only identity
comparison from pandas storage-dtype equality to exact per-column string values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from multimodal import geo_helpfulness_router_preview_score as v1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "multimodal/geo_helpfulness_router_preview_score_v2.py",
    "tools/run_multimodal_geo_helpfulness_m4_preview.py",
)
DECLARATION_SCHEMA_VERSION = "geo_helpfulness.router_preview_correction_declaration.v2"
RESULT_SCHEMA_VERSION = "geo_helpfulness.router_fixed_preview.v2"
CORRECTION_REASON = "pandas_string_storage_dtype_identity_equality"
RouterPreviewError = v1.RouterPreviewError


def _validate_correction_declaration(
    path: Path, manifest: dict[str, Any], snapshot: dict[Path, str]
) -> dict[str, Any]:
    """Bind the correction and its current code before assignment access."""
    declaration = v1._read_json(path)
    if path.stat().st_mode & 0o222:
        raise RouterPreviewError("correction declaration must be immutable/read-only")
    digest = v1._digest(declaration.get("declaration_sha256"), "declaration_sha256")
    if digest != v1._hash(
        {
            key: value
            for key, value in declaration.items()
            if key != "declaration_sha256"
        }
    ):
        raise RouterPreviewError("correction declaration self-hash mismatch")
    expected = {
        "schema_version": DECLARATION_SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "candidate_variant": "primary_full727",
        "training_seed": 1,
        "C_hex": float(0.01).hex(),
        "metric": "four_state_accuracy_only",
        "selection_performed": False,
        "correction_reason": CORRECTION_REASON,
    }
    if v1._canonical_bytes(
        {key: declaration.get(key) for key in expected}
    ) != v1._canonical_bytes(expected):
        raise RouterPreviewError(
            "correction declaration changes the fixed preview contract"
        )
    code_hashes = declaration.get("code_file_sha256")
    if not isinstance(code_hashes, dict) or set(code_hashes) != set(CODE_FILES):
        raise RouterPreviewError("correction code fingerprint allowlist mismatch")
    for name, expected_hash in code_hashes.items():
        v1._digest(expected_hash, f"{name}.code_file_sha256")
        code_path = PROJECT_ROOT / name
        if v1._file_hash(code_path) != expected_hash:
            raise RouterPreviewError(f"predeclared correction code changed: {name}")
        snapshot[code_path] = expected_hash
    snapshot[path] = v1._file_hash(path)
    return declaration


def _identity_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].to_numpy(dtype=object)
    if frame[column].isna().any() or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise RouterPreviewError(
            f"identity column contains null, empty, or non-string values: {column}"
        )
    return values


def _identities_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Ignore pandas storage dtype, never coerce or normalize identity values."""
    if len(left) != len(right):
        return False
    return all(
        np.array_equal(_identity_values(left, column), _identity_values(right, column))
        for column in ("row_uid", "plot_idx")
    )


def score_fixed_preview_v2(
    bundle_root: str | Path,
    assignment_path: str | Path,
    *,
    correction_declaration_path: str | Path,
    output_path: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Score the same single predeclared candidate with value-based identities."""
    root = Path(bundle_root)
    correction_path = Path(correction_declaration_path)
    assignment = Path(assignment_path)
    if correction_path.resolve().is_relative_to(root.resolve()):
        raise RouterPreviewError(
            "correction declaration must be outside the sealed prediction bundle"
        )
    if output_path is not None:
        destination = Path(output_path).resolve()
        if destination.is_relative_to(root.resolve()):
            raise RouterPreviewError(
                "preview output must be outside the sealed prediction bundle"
            )
        if destination in {assignment.resolve(), correction_path.resolve()}:
            raise RouterPreviewError(
                "preview output cannot overwrite an immutable input"
            )
    manifest, predictions, experts, snapshot = v1._load_sealed_predictions(
        root, expected_manifest_sha256=expected_manifest_sha256
    )
    correction = _validate_correction_declaration(correction_path, manifest, snapshot)
    v1._check_snapshot(snapshot)

    # First assignment access: the complete old seal and the additive correction
    # declaration/current code have already been checked without labels.
    assignment_hash = v1._file_hash(assignment)
    if assignment_hash != manifest["assignment_file_sha256"]:
        raise RouterPreviewError(
            "assignment source differs from sealed prelabel identity"
        )
    snapshot[assignment] = assignment_hash
    table = pq.read_table(assignment, columns=list(v1.ASSIGNMENT_COLUMNS))
    if any(column.null_count for column in table.columns):
        raise RouterPreviewError("null assignment identity or target value")
    frame = table.to_pandas()
    if set(frame.protocol_id) != {manifest["protocol_id"]}:
        raise RouterPreviewError("assignment protocol mismatch")
    if not set(frame.development_role).issubset({"train", "validation"}):
        raise RouterPreviewError("assignment contains an unexpected development role")
    validation = frame.loc[frame.development_role.eq("validation")].copy()
    if (
        len(validation) != v1.VALIDATION_ROWS
        or validation.row_uid.duplicated().any()
        or validation.plot_idx.nunique() != v1.VALIDATION_PLOTS
    ):
        raise RouterPreviewError("assignment validation coverage mismatch")
    labels = validation.label_id_dense.to_numpy()
    if labels.dtype.kind not in "iu" or not ((labels >= 0) & (labels < 18)).all():
        raise RouterPreviewError("validation labels are outside the dense ontology")
    validation = validation.sort_values("row_uid").reset_index(drop=True)
    for source in (predictions, experts):
        if not _identities_equal(source, validation):
            raise RouterPreviewError(
                "sealed predictions and validation assignments do not align"
            )
    labels = validation.label_id_dense.to_numpy(dtype=np.int64)
    raw_correct = experts.raw_pred.to_numpy() == labels
    geo_correct = experts.geo_pred.to_numpy() == labels
    target = np.full(v1.VALIDATION_ROWS, 3, dtype=np.int64)
    target[~raw_correct & geo_correct] = 0
    target[raw_correct & ~geo_correct] = 1
    target[raw_correct & geo_correct] = 2
    predicted = np.argmax(
        predictions[list(v1.Q_COLUMNS)].to_numpy(dtype=np.float64), axis=1
    )
    correct = int(np.sum(predicted == target))
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "assignment_file_sha256": assignment_hash,
        "preview_declaration_file_sha256": manifest["artifacts"][
            "preview_declaration.json"
        ]["file_sha256"],
        "correction_declaration_sha256": correction["declaration_sha256"],
        "correction_declaration_file_sha256": snapshot[correction_path],
        "code_file_sha256": correction["code_file_sha256"],
        "correction_reason": CORRECTION_REASON,
        "candidate_variant": "primary_full727",
        "training_seed": 1,
        "C_hex": float(0.01).hex(),
        "target_order": list(v1.TARGET_ORDER),
        "metric": "four_state_accuracy_only",
        "correct_count": correct,
        "total_count": v1.VALIDATION_ROWS,
        "accuracy": correct / v1.VALIDATION_ROWS,
        "selection_performed": False,
        "other_candidate_metrics_computed": False,
        "label_role": "development_validation_only",
    }
    result["preview_sha256"] = v1._hash(result)
    v1._check_snapshot(snapshot)
    if output_path is not None:
        v1._save_result(Path(output_path), result)
    return result


__all__ = ["RouterPreviewError", "score_fixed_preview_v2"]

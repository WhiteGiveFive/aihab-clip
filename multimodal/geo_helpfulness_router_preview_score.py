"""Scoring-only, fixed-candidate preview after a complete label-blind seal.

No fitting or parent-pipeline modules are imported. This module verifies all
prelabel artifacts before opening the assignment table, then computes exactly
one predeclared four-state accuracy; it performs no model or policy selection.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "multimodal/geo_helpfulness_router_validation.py",
    "multimodal/geo_helpfulness_router_validation_inputs.py",
    "multimodal/geo_helpfulness_router_heuristics.py",
    "multimodal/geo_helpfulness_router_preview_score.py",
    "tools/run_multimodal_geo_helpfulness_m4_validation.py",
)
VARIANTS = ("primary_full727", "ablation_no_pairs79", "ablation_numeric25")
TRAINING_SEEDS = (1, 2, 3, 4)
C_GRID = (0.01, 0.1, 1.0, 10.0)
C_HEX = tuple(value.hex() for value in C_GRID)
TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
VALIDATION_ROWS = 822
VALIDATION_PLOTS = 325
PREDICTION_SCHEMA_VERSION = "geo_helpfulness.router_candidate_validation_predictions.v1"
EXPERT_SCHEMA_VERSION = "geo_helpfulness.router_validation_expert_predictions.v1"
MANIFEST_SCHEMA_VERSION = "geo_helpfulness.router_validation_manifest.v1"
DECLARATION_SCHEMA_VERSION = "geo_helpfulness.router_preview_declaration.v1"
RESULT_SCHEMA_VERSION = "geo_helpfulness.router_fixed_preview.v1"
IDENTITY_COLUMNS = (
    "schema_version",
    "protocol_id",
    "row_uid",
    "plot_idx",
    "training_seed",
)
Q_COLUMNS = tuple(f"q_{target}" for target in TARGET_ORDER)
PREDICTION_COLUMNS = (*IDENTITY_COLUMNS, "C_hex", *Q_COLUMNS, "score")
EXPERT_COLUMNS = (*IDENTITY_COLUMNS, "image_pred", "geo_pred", "raw_pred")
ASSIGNMENT_COLUMNS = (
    "protocol_id",
    "row_uid",
    "plot_idx",
    "development_role",
    "label_id_dense",
)
ARTIFACT_FILENAMES = (
    *(f"{variant}.parquet" for variant in VARIANTS),
    "expert_predictions.parquet",
    "oof_heuristics.json",
    "preview_declaration.json",
)


class RouterPreviewError(ValueError):
    """The sealed prediction bundle or fixed preview contract is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RouterPreviewError("artifact must contain finite JSON values") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RouterPreviewError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RouterPreviewError(f"expected regular nonsymlink file: {path.name}")


def _file_hash(path: Path) -> str:
    _regular_file(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RouterPreviewError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise RouterPreviewError(f"nonfinite JSON value: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    _regular_file(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RouterPreviewError(f"cannot read JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise RouterPreviewError(f"JSON artifact must be an object: {path.name}")
    return value


def prediction_schema() -> pa.Schema:
    fields = [
        pa.field(
            name, pa.int8() if name == "training_seed" else pa.string(), nullable=False
        )
        for name in IDENTITY_COLUMNS
    ]
    return pa.schema(
        fields
        + [pa.field("C_hex", pa.string(), nullable=False)]
        + [
            pa.field(name, pa.float64(), nullable=False)
            for name in (*Q_COLUMNS, "score")
        ]
    )


def expert_schema() -> pa.Schema:
    fields = [
        pa.field(
            name, pa.int8() if name == "training_seed" else pa.string(), nullable=False
        )
        for name in IDENTITY_COLUMNS
    ]
    return pa.schema(
        fields
        + [
            pa.field(name, pa.int8(), nullable=False)
            for name in ("image_pred", "geo_pred", "raw_pred")
        ]
    )


def _read_frame(
    path: Path,
    schema: pa.Schema,
    *,
    row_count: int,
    schema_version: str,
    protocol_id: str,
) -> pd.DataFrame:
    _regular_file(path)
    table = pq.read_table(path)
    if (
        not table.schema.equals(schema, check_metadata=True)
        or table.num_rows != row_count
    ):
        raise RouterPreviewError(
            f"prediction schema or row-count mismatch: {path.name}"
        )
    if any(column.null_count for column in table.columns):
        raise RouterPreviewError(f"null values in prediction artifact: {path.name}")
    frame = table.to_pandas()
    if set(frame.schema_version) != {schema_version} or set(frame.protocol_id) != {
        protocol_id
    }:
        raise RouterPreviewError(
            f"prediction schema/protocol identity mismatch: {path.name}"
        )
    for column in ("row_uid", "plot_idx"):
        if not frame[column].str.strip().ne("").all():
            raise RouterPreviewError(f"empty prediction identity: {column}")
    if not frame.row_uid.str.fullmatch(r"[0-9a-f]{64}").all():
        raise RouterPreviewError(
            "prediction row_uid must be a lowercase SHA-256 identity"
        )
    if set(frame.training_seed) != set(TRAINING_SEEDS):
        raise RouterPreviewError("prediction seed coverage mismatch")
    return frame


def _check_snapshot(snapshot: dict[Path, str]) -> None:
    if any(_file_hash(path) != digest for path, digest in snapshot.items()):
        raise RouterPreviewError("sealed preview input changed during scoring")


def _load_sealed_predictions(
    root: Path,
    *,
    expected_manifest_sha256: str | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[Path, str]]:
    if root.is_symlink() or not root.is_dir():
        raise RouterPreviewError("prediction bundle must be a nonsymlink directory")
    if {path.name for path in root.iterdir()} != set(ARTIFACT_FILENAMES) | {
        "manifest.json"
    }:
        raise RouterPreviewError("complete prediction bundle file allowlist mismatch")
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RouterPreviewError("prediction manifest schema mismatch")
    manifest_hash = _digest(manifest.get("manifest_sha256"), "manifest_sha256")
    if manifest_hash != _hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise RouterPreviewError("prediction manifest self-hash mismatch")
    if expected_manifest_sha256 is not None:
        _digest(expected_manifest_sha256, "expected_manifest_sha256")
        if manifest_hash != expected_manifest_sha256:
            raise RouterPreviewError(
                "prediction manifest differs from expected identity"
            )
    _digest(manifest.get("assignment_file_sha256"), "assignment_file_sha256")
    protocol_id = manifest.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise RouterPreviewError("prediction manifest protocol_id is missing")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_FILENAMES):
        raise RouterPreviewError(
            "prediction manifest does not seal all required artifacts"
        )
    snapshot = {manifest_path: _file_hash(manifest_path)}
    for name in ARTIFACT_FILENAMES:
        record = artifacts[name]
        if not isinstance(record, dict):
            raise RouterPreviewError("invalid artifact hash record")
        expected_hash = _digest(record.get("file_sha256"), f"{name}.file_sha256")
        path = root / name
        if _file_hash(path) != expected_hash:
            raise RouterPreviewError(f"sealed artifact hash mismatch: {name}")
        snapshot[path] = expected_hash
    declaration = _read_json(root / "preview_declaration.json")
    fixed_declaration = {
        "schema_version": DECLARATION_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "candidate_variant": "primary_full727",
        "training_seed": 1,
        "C_hex": float(0.01).hex(),
        "metric": "four_state_accuracy_only",
        "selection_performed": False,
    }
    if _canonical_bytes(
        {key: declaration.get(key) for key in fixed_declaration}
    ) != _canonical_bytes(fixed_declaration):
        raise RouterPreviewError(
            "preview candidate/metric was not predeclared as required"
        )
    for name in ("dataset_manifest_sha256", "candidate_manifest_sha256"):
        expected = _digest(manifest.get(name), name)
        if declaration.get(name) != expected:
            raise RouterPreviewError(
                f"preview declaration parent identity mismatch: {name}"
            )
    code_hashes = manifest.get("code_file_sha256")
    if (
        not isinstance(code_hashes, dict)
        or set(code_hashes) != set(CODE_FILES)
        or code_hashes != declaration.get("code_file_sha256")
    ):
        raise RouterPreviewError(
            "preview code fingerprint allowlist or declaration mismatch"
        )
    for name, expected in code_hashes.items():
        _digest(expected, f"{name}.code_file_sha256")
        path = PROJECT_ROOT / name
        if _file_hash(path) != expected:
            raise RouterPreviewError(f"predeclared preview code changed: {name}")
        snapshot[path] = expected
    heuristics = _read_json(root / "oof_heuristics.json")
    heuristic_scope = {
        "schema_version": "geo_helpfulness.oof_heuristic_thresholds.v1",
        "protocol_id": protocol_id,
        "fit_role": "development_train_oof",
        "validation_outputs_opened": False,
        "validation_labels_scored": False,
        "test_sources_opened": False,
    }
    if _canonical_bytes(
        {name: heuristics.get(name) for name in heuristic_scope}
    ) != _canonical_bytes(heuristic_scope):
        raise RouterPreviewError(
            "heuristic prerequisites are not scoped to training OOF"
        )
    heuristic_hash = _hash(heuristics)
    if declaration.get("oof_heuristics_sha256") != heuristic_hash:
        raise RouterPreviewError("predeclared heuristic state content hash mismatch")
    if (
        "oof_heuristics_sha256" in manifest
        and manifest["oof_heuristics_sha256"] != heuristic_hash
    ):
        raise RouterPreviewError("manifest heuristic state content hash mismatch")
    experts = _read_frame(
        root / "expert_predictions.parquet",
        expert_schema(),
        row_count=VALIDATION_ROWS * 4,
        schema_version=EXPERT_SCHEMA_VERSION,
        protocol_id=protocol_id,
    )
    if not experts.equals(
        experts.sort_values(["row_uid", "training_seed"], kind="stable").reset_index(
            drop=True
        )
    ):
        raise RouterPreviewError(
            "expert predictions are not in canonical identity order"
        )
    if experts.duplicated(["row_uid", "training_seed"]).any():
        raise RouterPreviewError("duplicate expert prediction identity")
    if experts.groupby("row_uid")["plot_idx"].nunique().max() != 1:
        raise RouterPreviewError("expert plot identities differ across seeds")
    base = (
        experts.loc[experts.training_seed.eq(1), ["row_uid", "plot_idx"]]
        .sort_values("row_uid")
        .reset_index(drop=True)
    )
    if (
        len(base) != VALIDATION_ROWS
        or base.row_uid.duplicated().any()
        or base.plot_idx.nunique() != VALIDATION_PLOTS
    ):
        raise RouterPreviewError("expert validation image coverage mismatch")
    for seed in TRAINING_SEEDS:
        other = (
            experts.loc[experts.training_seed.eq(seed), ["row_uid", "plot_idx"]]
            .sort_values("row_uid")
            .reset_index(drop=True)
        )
        if not other.equals(base):
            raise RouterPreviewError("expert identities differ across seeds")
    expert_classes = experts[["image_pred", "geo_pred", "raw_pred"]].to_numpy()
    if not ((expert_classes >= 0) & (expert_classes < 18)).all():
        raise RouterPreviewError("expert predicted class is outside dense ontology")
    expected_keys = {
        (uid, seed, c_hex)
        for uid in base.row_uid
        for seed in TRAINING_SEEDS
        for c_hex in C_HEX
    }
    plots = dict(zip(base.row_uid, base.plot_idx))
    selected = None
    for variant in VARIANTS:
        frame = _read_frame(
            root / f"{variant}.parquet",
            prediction_schema(),
            row_count=VALIDATION_ROWS * 16,
            schema_version=PREDICTION_SCHEMA_VERSION,
            protocol_id=protocol_id,
        )
        if not frame.equals(
            frame.sort_values(
                ["C_hex", "training_seed", "row_uid"], kind="stable"
            ).reset_index(drop=True)
        ):
            raise RouterPreviewError(
                "candidate predictions are not in canonical identity order"
            )
        keys = list(
            frame[["row_uid", "training_seed", "C_hex"]].itertuples(
                index=False, name=None
            )
        )
        if len(set(keys)) != len(keys) or set(keys) != expected_keys:
            raise RouterPreviewError(
                f"incomplete candidate validation cross-product: {variant}"
            )
        if not frame.plot_idx.equals(frame.row_uid.map(plots)):
            raise RouterPreviewError("candidate/expert plot identities differ")
        probability = frame[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
        score = frame.score.to_numpy(dtype=np.float64)
        if (
            not np.isfinite(probability).all()
            or not ((probability >= 0) & (probability <= 1)).all()
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
            or not np.isfinite(score).all()
            or not np.allclose(
                score, probability[:, 0] - probability[:, 1], atol=1e-12, rtol=0.0
            )
        ):
            raise RouterPreviewError("invalid sealed router probabilities or score")
        if variant == "primary_full727":
            selected = (
                frame.loc[frame.training_seed.eq(1) & frame.C_hex.eq(float(0.01).hex())]
                .sort_values("row_uid")
                .reset_index(drop=True)
            )
    _check_snapshot(snapshot)
    return (
        manifest,
        selected,
        experts.loc[experts.training_seed.eq(1)]
        .sort_values("row_uid")
        .reset_index(drop=True),
        snapshot,
    )


def validate_sealed_predictions(
    bundle_root: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the complete prelabel seal without opening assignment content."""
    manifest, _, _, _ = _load_sealed_predictions(
        Path(bundle_root),
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return {
        "valid": True,
        "status": "reused_valid",
        "bundle_root": str(bundle_root),
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": len(VARIANTS) * len(TRAINING_SEEDS) * len(C_GRID),
        "variant_count": len(VARIANTS),
        "rows_per_variant": VALIDATION_ROWS * 16,
        "expert_row_count": VALIDATION_ROWS * 4,
        "validation_rows_per_seed": VALIDATION_ROWS,
        "validation_plots": VALIDATION_PLOTS,
        "assignment_content_opened": False,
        "selection_performed": False,
    }


def _save_result(path: Path, result: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if _read_json(path) != result:
            raise RouterPreviewError("existing immutable preview result differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".router_preview-", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(result) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_json(path) != result:
                raise RouterPreviewError("concurrent immutable preview result differs")
    finally:
        temporary.unlink(missing_ok=True)


def score_fixed_preview(
    bundle_root: str | Path,
    assignment_path: str | Path,
    *,
    output_path: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute only primary/seed-1/C=0.01 accuracy after validating the full seal."""
    root = Path(bundle_root)
    if output_path is not None:
        destination = Path(output_path)
        if destination.resolve().is_relative_to(root.resolve()):
            raise RouterPreviewError(
                "preview output must be outside the sealed prediction bundle"
            )
        if destination.resolve() == Path(assignment_path).resolve():
            raise RouterPreviewError(
                "preview output cannot overwrite the assignment source"
            )
    manifest, predictions, experts, snapshot = _load_sealed_predictions(
        root, expected_manifest_sha256=expected_manifest_sha256
    )
    # This is the first assignment access, after all candidates and prerequisites
    # have been checked. Labels are never used to fit or select anything here.
    assignment = Path(assignment_path)
    assignment_hash = _file_hash(assignment)
    if assignment_hash != manifest["assignment_file_sha256"]:
        raise RouterPreviewError(
            "assignment source differs from sealed prelabel identity"
        )
    snapshot[assignment] = assignment_hash
    table = pq.read_table(assignment, columns=list(ASSIGNMENT_COLUMNS))
    if any(column.null_count for column in table.columns):
        raise RouterPreviewError("null assignment identity or target value")
    frame = table.to_pandas()
    if set(frame.protocol_id) != {manifest["protocol_id"]}:
        raise RouterPreviewError("assignment protocol mismatch")
    if not set(frame.development_role).issubset({"train", "validation"}):
        raise RouterPreviewError("assignment contains an unexpected development role")
    validation = frame.loc[frame.development_role.eq("validation")].copy()
    if (
        len(validation) != VALIDATION_ROWS
        or validation.row_uid.duplicated().any()
        or validation.plot_idx.nunique() != VALIDATION_PLOTS
    ):
        raise RouterPreviewError("assignment validation coverage mismatch")
    labels = validation.label_id_dense.to_numpy()
    if labels.dtype.kind not in "iu" or not ((labels >= 0) & (labels < 18)).all():
        raise RouterPreviewError("validation labels are outside the dense ontology")
    validation = validation.sort_values("row_uid").reset_index(drop=True)
    for source in (predictions, experts):
        if not source[["row_uid", "plot_idx"]].equals(
            validation[["row_uid", "plot_idx"]]
        ):
            raise RouterPreviewError(
                "sealed predictions and validation assignments do not align"
            )
    labels = validation.label_id_dense.to_numpy(dtype=np.int64)
    raw_correct = experts.raw_pred.to_numpy() == labels
    geo_correct = experts.geo_pred.to_numpy() == labels
    target = np.full(VALIDATION_ROWS, 3, dtype=np.int64)
    target[~raw_correct & geo_correct] = 0
    target[raw_correct & ~geo_correct] = 1
    target[raw_correct & geo_correct] = 2
    predicted = np.argmax(
        predictions[list(Q_COLUMNS)].to_numpy(dtype=np.float64), axis=1
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
        "candidate_variant": "primary_full727",
        "training_seed": 1,
        "C_hex": float(0.01).hex(),
        "target_order": list(TARGET_ORDER),
        "metric": "four_state_accuracy_only",
        "correct_count": correct,
        "total_count": VALIDATION_ROWS,
        "accuracy": correct / VALIDATION_ROWS,
        "selection_performed": False,
        "other_candidate_metrics_computed": False,
        "label_role": "development_validation_only",
    }
    result["preview_sha256"] = _hash(result)
    _check_snapshot(snapshot)
    if output_path is not None:
        _save_result(Path(output_path), result)
    return result


__all__ = ["RouterPreviewError", "validate_sealed_predictions", "score_fixed_preview"]

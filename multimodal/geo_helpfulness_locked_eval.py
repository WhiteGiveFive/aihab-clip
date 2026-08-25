"""Capability-limited locked-evaluation helpers for protocol-v1.

This module deliberately contains no training, calibration, model-selection, or
threshold-search code.  M1 uses the implementation only with synthetic fixture
inputs to make the locked prediction/scoring boundary executable before any
real models or cleaned-test labels are opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PREDICTION_SCHEMA_VERSION = "geo_helpfulness.locked_predictions.synthetic.v1"
SCORING_SCHEMA_VERSION = "geo_helpfulness.locked_score_receipt.synthetic.v1"

_Q_COLUMNS = (
    "q_rescue",
    "q_harm",
    "q_both_correct",
    "q_both_wrong",
)
_REQUIRED_FEATURE_COLUMNS = {
    "row_uid",
    "training_seed",
    "raw_pred",
    "geo_pred",
    *_Q_COLUMNS,
}
_OPTIONAL_FEATURE_COLUMNS = {
    "file",
    "file_lower",
    "plot_idx",
    "normalized_plot_idx",
}
_PREDICTION_COLUMNS = (
    "schema_version",
    "protocol_id",
    "snapshot_id",
    "row_uid",
    "file",
    "file_lower",
    "plot_idx",
    "normalized_plot_idx",
    "training_seed",
    "raw_pred",
    "geo_pred",
    *_Q_COLUMNS,
    "router_score",
    "acted",
    "final_pred",
)
_LABEL_COLUMNS = {"row_uid", "label_id_dense"}


class LockedEvaluationError(ValueError):
    """Raised when a locked-evaluation capability or artifact check fails."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Use the one protocol-wide canonical serialization for all fingerprints."""

    from multimodal.geo_helpfulness_protocol import canonical_json_bytes as serialize

    return serialize(value)


def exclusive_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Create *path* exactly once, failing if it already exists."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The path is intentionally left in place on an interrupted write.  A
        # partial immutable artifact must be investigated, never overwritten.
        raise
    return destination


def exclusive_write_json(path: str | Path, value: Any) -> Path:
    return exclusive_write_bytes(path, canonical_json_bytes(value))


def _read_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockedEvaluationError(f"Cannot read {name} JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise LockedEvaluationError(f"{name} must be a JSON object: {source}")
    return value


def _read_table(path: str | Path, *, name: str) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise LockedEvaluationError(f"{name} does not exist: {source}")
    suffix = source.suffix.lower()
    try:
        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(source)
        elif suffix == ".csv":
            frame = pd.read_csv(source, dtype={"row_uid": "string"})
        elif suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(source, lines=True)
        elif suffix == ".json":
            value = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                for key in ("rows", "records", "predictions", "features", "labels"):
                    if key in value:
                        value = value[key]
                        break
            if not isinstance(value, list):
                raise LockedEvaluationError(
                    f"{name} JSON must be a list or contain a rows/records list"
                )
            frame = pd.DataFrame.from_records(value)
        else:
            raise LockedEvaluationError(
                f"Unsupported {name} format {suffix!r}; use parquet, CSV, JSON, or JSONL"
            )
    except LockedEvaluationError:
        raise
    except Exception as exc:
        raise LockedEvaluationError(f"Cannot read {name} {source}: {exc}") from exc
    if len(frame) == 0:
        raise LockedEvaluationError(f"{name} must contain at least one row")
    return frame


def _frame_records(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in frame.loc[:, list(columns)].to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
                row[key] = None
            elif isinstance(value, np.generic):
                row[key] = value.item()
            else:
                row[key] = value
        records.append(row)
    return records


def _write_table_exclusive(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    elif suffix in {".jsonl", ".ndjson"}:
        lines = [canonical_json_bytes(record).decode("utf-8").rstrip("\n") for record in _frame_records(frame, frame.columns)]
        payload = ("\n".join(lines) + "\n").encode("utf-8")
    elif suffix == ".json":
        payload = canonical_json_bytes(_frame_records(frame, frame.columns))
    elif suffix in {".parquet", ".pq"}:
        # Pandas has no exclusive-create parquet writer. Reserve the final path,
        # serialize to a sibling temporary file, then replace only our empty
        # reservation. A concurrent creator cannot pass the O_EXCL reservation.
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        os.close(descriptor)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            os.chmod(temporary, 0o444)
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise
        return destination
    else:
        raise LockedEvaluationError(
            f"Unsupported output format {suffix!r}; use parquet, CSV, JSON, or JSONL"
        )
    return exclusive_write_bytes(destination, payload)


def _manifest_path_for(data_path: str | Path) -> Path:
    path = Path(data_path)
    return path.with_name(f"{path.name}.manifest.json")


def _extract_identity_hash(manifest: Mapping[str, Any]) -> str:
    for key in (
        "identity_projection_sha256",
        "locked_test_identity_sha256",
        "test_identity_projection_sha256",
    ):
        value = manifest.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
    raise LockedEvaluationError(
        "Identity manifest has no 64-character identity_projection_sha256"
    )


def _extract_identity_row_uids(manifest: Mapping[str, Any]) -> set[str]:
    row_uids = manifest.get("row_uids")
    if isinstance(row_uids, list):
        values = row_uids
    else:
        values = None
        for key in ("identity_projection", "identities", "rows", "records"):
            rows = manifest.get(key)
            if isinstance(rows, list):
                values = [row.get("row_uid") for row in rows if isinstance(row, Mapping)]
                break
    if values is None:
        raise LockedEvaluationError("Identity manifest does not expose its sealed row_uid set")
    normalized = [str(value).strip().lower() for value in values]
    if any(len(value) != 64 for value in normalized):
        raise LockedEvaluationError("Identity manifest contains an invalid row_uid")
    if len(set(normalized)) != len(normalized):
        raise LockedEvaluationError("Identity manifest contains duplicate row_uid values")
    return set(normalized)


def _bundle_identity_hash(bundle: Mapping[str, Any]) -> str | None:
    for key in (
        "identity_projection_sha256",
        "locked_test_identity_sha256",
        "test_identity_projection_sha256",
    ):
        value = bundle.get(key)
        if isinstance(value, str):
            return value.lower()
    return None


def _required_sha256(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdefABCDEF" for character in value)
        ):
            return value.lower()
    raise LockedEvaluationError(f"Bundle requires a SHA-256 field named one of {keys}")


def _snapshot_id(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("snapshot_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _extract_identity_hash(manifest)


def _threshold(bundle: Mapping[str, Any]) -> float | None:
    spec = bundle.get("threshold_specification", bundle.get("threshold_spec"))
    if not isinstance(spec, Mapping):
        raise LockedEvaluationError("Bundle is missing threshold_specification")
    if spec.get("comparison") != "strict_gt":
        raise LockedEvaluationError("Only the frozen strict_gt threshold policy is permitted")
    kind = spec.get("kind")
    if kind == "never_intervene":
        if spec.get("value_hex") is not None:
            raise LockedEvaluationError("never_intervene must have a null value_hex")
        return None
    if kind != "finite":
        raise LockedEvaluationError(f"Unsupported frozen threshold kind: {kind!r}")
    value_hex = spec.get("value_hex")
    if not isinstance(value_hex, str):
        raise LockedEvaluationError("Finite threshold requires an IEEE-754 value_hex")
    try:
        value = float.fromhex(value_hex)
    except ValueError as exc:
        raise LockedEvaluationError("Invalid finite threshold value_hex") from exc
    if not math.isfinite(value) or value < 0:
        raise LockedEvaluationError("Finite protocol-v1 threshold must be finite and nonnegative")
    return value


def _validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "seed" in frame.columns and "training_seed" not in frame.columns:
        frame = frame.rename(columns={"seed": "training_seed"})
    columns = set(map(str, frame.columns))
    missing = _REQUIRED_FEATURE_COLUMNS.difference(columns)
    unknown = columns.difference(_REQUIRED_FEATURE_COLUMNS | _OPTIONAL_FEATURE_COLUMNS)
    if missing:
        raise LockedEvaluationError(f"Synthetic features are missing columns: {sorted(missing)}")
    if unknown:
        raise LockedEvaluationError(
            "Synthetic locked prediction uses a label-blind allow-list; "
            f"undeclared columns: {sorted(unknown)}"
        )

    checked = frame.copy()
    checked["row_uid"] = checked["row_uid"].astype("string").str.strip().str.lower()
    if bool(checked["row_uid"].isna().any()) or bool(
        (checked["row_uid"].str.len() != 64).any()
    ):
        raise LockedEvaluationError("Synthetic features contain invalid row_uid values")
    identity = checked[["row_uid", "training_seed"]]
    if bool(identity.duplicated().any()):
        raise LockedEvaluationError(
            "Synthetic features contain duplicate (row_uid, training_seed) rows"
        )

    for column in ("training_seed", "raw_pred", "geo_pred"):
        values = pd.to_numeric(checked[column], errors="coerce")
        if bool(values.isna().any()) or bool((values % 1 != 0).any()):
            raise LockedEvaluationError(f"Synthetic feature {column} must contain integers")
        checked[column] = values.astype("int64")
    if bool(
        (
            (checked[["raw_pred", "geo_pred"]] < 0)
            | (checked[["raw_pred", "geo_pred"]] >= 18)
        ).any().any()
    ):
        raise LockedEvaluationError("Synthetic predictions must use fixed dense class IDs 0..17")

    q_values = checked.loc[:, list(_Q_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    q_array = q_values.to_numpy(dtype=np.float64)
    if not np.isfinite(q_array).all():
        raise LockedEvaluationError("Synthetic router probabilities must be finite")
    if bool(((q_array < 0) | (q_array > 1)).any()):
        raise LockedEvaluationError("Synthetic router probabilities must lie in [0, 1]")
    if not np.allclose(q_array.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise LockedEvaluationError("Synthetic four-state probabilities must sum to one")
    checked.loc[:, list(_Q_COLUMNS)] = q_array
    return checked


def locked_predict_synthetic(
    *,
    bundle_path: str | Path,
    identity_manifest_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Apply a frozen hard-override policy to a label-blind synthetic fixture."""

    bundle = _read_json_object(bundle_path, name="composite bundle")
    if bundle.get("artifact_role") != "composite_inference_bundle":
        raise LockedEvaluationError("Bundle has the wrong artifact_role")
    identity_manifest = _read_json_object(identity_manifest_path, name="identity manifest")
    # Identity validation is pure metadata validation; it does not expose a
    # table path, labels, features, or fitting capability.
    from multimodal.geo_helpfulness_protocol import validate_locked_test_identity_manifest

    validate_locked_test_identity_manifest(identity_manifest)
    identity_hash = _extract_identity_hash(identity_manifest)
    expected_identity_hash = _bundle_identity_hash(bundle)
    if expected_identity_hash is None:
        raise LockedEvaluationError(
            "Composite bundle is not bound to a locked-test identity snapshot"
        )
    if expected_identity_hash != identity_hash:
        raise LockedEvaluationError(
            "Composite bundle is bound to a different locked-test identity snapshot"
        )
    frozen_router_sha256 = _required_sha256(
        bundle, "frozen_router_bundle_sha256", "frozen_router_bundle_hash"
    )
    ontology_sha256 = _required_sha256(bundle, "ontology_sha256", "ontology_hash")
    _required_sha256(bundle, "final_expert_bundle_sha256", "final_expert_bundle_hash")
    _required_sha256(bundle, "selection_receipt_sha256", "selection_receipt_hash")

    sealed_row_uids = _extract_identity_row_uids(identity_manifest)
    features = _validate_feature_frame(_read_table(features_path, name="synthetic features"))
    feature_row_uids = set(features["row_uid"].astype(str))
    if feature_row_uids != sealed_row_uids:
        missing = sorted(sealed_row_uids.difference(feature_row_uids))[:5]
        extra = sorted(feature_row_uids.difference(sealed_row_uids))[:5]
        raise LockedEvaluationError(
            "Synthetic feature rows do not exactly match the sealed identity snapshot: "
            f"missing={missing}, extra={extra}"
        )

    threshold = _threshold(bundle)
    score = features["q_rescue"].to_numpy(dtype=np.float64) - features[
        "q_harm"
    ].to_numpy(dtype=np.float64)
    disagreement = features["geo_pred"].to_numpy() != features["raw_pred"].to_numpy()
    if threshold is None:
        action = np.zeros(len(features), dtype=bool)
    else:
        action = (score > threshold) & disagreement
    final_pred = np.where(action, features["geo_pred"], features["raw_pred"]).astype(np.int64)

    output = features.copy()
    for column in _OPTIONAL_FEATURE_COLUMNS:
        if column not in output.columns:
            output[column] = None
    output.insert(0, "snapshot_id", _snapshot_id(identity_manifest))
    output.insert(0, "protocol_id", str(bundle.get("protocol_id", "protocol_v1")))
    output.insert(0, "schema_version", PREDICTION_SCHEMA_VERSION)
    output["router_score"] = score
    output["acted"] = action
    output["final_pred"] = final_pred
    output = output.loc[:, list(_PREDICTION_COLUMNS)].sort_values(
        ["row_uid", "training_seed"], kind="mergesort"
    ).reset_index(drop=True)

    output_path = _write_table_exclusive(output_path, output)
    bundle_sha256 = sha256_file(bundle_path)
    prediction_sha256 = sha256_file(output_path)
    manifest = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "artifact_role": "locked_test_predictions",
        "synthetic_fixture": True,
        "exploratory_only": True,
        "protocol_id": str(bundle.get("protocol_id", "protocol_v1")),
        "snapshot_id": _snapshot_id(identity_manifest),
        "composite_bundle_file_sha256": bundle_sha256,
        "composite_bundle_hash": bundle_sha256,
        "frozen_router_bundle_sha256": frozen_router_sha256,
        "ontology_sha256": ontology_sha256,
        "identity_projection_sha256": identity_hash,
        "identity_manifest_file_sha256": sha256_file(identity_manifest_path),
        "features_file_sha256": sha256_file(features_path),
        "predictions_file": output_path.name,
        "predictions_file_sha256": prediction_sha256,
        "row_count": int(len(output)),
        "unique_test_rows": int(output["row_uid"].nunique()),
        "training_capability": False,
        "label_access": False,
        "threshold_specification": bundle.get(
            "threshold_specification", bundle.get("threshold_spec")
        ),
    }
    manifest_path = exclusive_write_json(_manifest_path_for(output_path), manifest)
    return {
        "status": "sealed",
        "predictions": str(output_path),
        "manifest": str(manifest_path),
        "predictions_file_sha256": prediction_sha256,
        "row_count": int(len(output)),
    }


def _validate_predictions_for_scoring(
    predictions_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = _read_table(predictions_path, name="sealed predictions")
    manifest_path = _manifest_path_for(predictions_path)
    manifest = _read_json_object(manifest_path, name="prediction manifest")
    if manifest.get("artifact_role") != "locked_test_predictions":
        raise LockedEvaluationError("Prediction manifest has the wrong artifact_role")
    if manifest.get("predictions_file_sha256") != sha256_file(predictions_path):
        raise LockedEvaluationError("Sealed prediction content hash mismatch")
    required = {
        "row_uid",
        "training_seed",
        "raw_pred",
        "geo_pred",
        "acted",
        "final_pred",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise LockedEvaluationError(f"Sealed predictions are missing: {sorted(missing)}")
    predictions = predictions.copy()
    predictions["row_uid"] = predictions["row_uid"].astype("string").str.lower()
    if "training_seed" not in predictions.columns and "seed" in predictions.columns:
        predictions = predictions.rename(columns={"seed": "training_seed"})
    if bool(predictions[["row_uid", "training_seed"]].duplicated().any()):
        raise LockedEvaluationError(
            "Sealed predictions contain duplicate (row_uid, training_seed) rows"
        )
    return predictions, manifest


def _validate_scoring_labels(labels_path: str | Path) -> pd.DataFrame:
    labels = _read_table(labels_path, name="scoring labels")
    columns = set(map(str, labels.columns))
    if columns != _LABEL_COLUMNS:
        raise LockedEvaluationError(
            "Locked scorer accepts only row_uid and label_id_dense; "
            f"received columns: {sorted(columns)}"
        )
    labels = labels.copy()
    labels["row_uid"] = labels["row_uid"].astype("string").str.strip().str.lower()
    if bool(labels["row_uid"].duplicated().any()):
        raise LockedEvaluationError("Scoring labels contain duplicate row_uid values")
    numeric = pd.to_numeric(labels["label_id_dense"], errors="coerce")
    if bool(numeric.isna().any()) or bool((numeric % 1 != 0).any()):
        raise LockedEvaluationError("label_id_dense must contain integer class IDs")
    labels["label_id_dense"] = numeric.astype("int64")
    if bool(((labels["label_id_dense"] < 0) | (labels["label_id_dense"] >= 18)).any()):
        raise LockedEvaluationError("label_id_dense must use fixed dense class IDs 0..17")
    return labels


def _reserve_score_event(
    registry_root: Path,
    *,
    bundle_sha256: str,
    identity_sha256: str,
) -> tuple[Path, bool]:
    bundle_claim = registry_root / "bundles" / f"{bundle_sha256}.json"
    snapshot_dir = registry_root / "snapshots" / identity_sha256
    exclusive_write_json(
        bundle_claim,
        {
            "status": "reserved",
            "composite_bundle_file_sha256": bundle_sha256,
            "identity_projection_sha256": identity_sha256,
        },
    )
    first_event_claim = snapshot_dir / "first_score_event.json"
    try:
        exclusive_write_json(
            first_event_claim,
            {
                "status": "reserved",
                "composite_bundle_file_sha256": bundle_sha256,
                "identity_projection_sha256": identity_sha256,
            },
        )
        adaptive_reuse = False
    except FileExistsError:
        adaptive_reuse = True
    return bundle_claim, adaptive_reuse


def locked_score_synthetic(
    *,
    predictions_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    event_registry: str | Path | None = None,
) -> dict[str, Any]:
    """Score sealed synthetic predictions without exposing any fitting API."""

    predictions, prediction_manifest = _validate_predictions_for_scoring(predictions_path)
    labels = _validate_scoring_labels(labels_path)
    prediction_rows = set(predictions["row_uid"].astype(str))
    label_rows = set(labels["row_uid"].astype(str))
    if prediction_rows != label_rows:
        missing = sorted(prediction_rows.difference(label_rows))[:5]
        extra = sorted(label_rows.difference(prediction_rows))[:5]
        raise LockedEvaluationError(
            "Scoring labels do not exactly match sealed predictions: "
            f"missing={missing}, extra={extra}"
        )

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable scoring receipt: {output_path}")
    registry_root = (
        Path(event_registry)
        if event_registry is not None
        else output_path.parent / "test_event_registry"
    )
    bundle_sha256 = str(prediction_manifest["composite_bundle_file_sha256"])
    identity_sha256 = str(prediction_manifest["identity_projection_sha256"])
    bundle_claim, adaptive_reuse = _reserve_score_event(
        registry_root,
        bundle_sha256=bundle_sha256,
        identity_sha256=identity_sha256,
    )

    merged = predictions.merge(labels, on="row_uid", how="left", validate="many_to_one")
    y = merged["label_id_dense"].to_numpy(dtype=np.int64)
    raw = pd.to_numeric(merged["raw_pred"], errors="raise").to_numpy(dtype=np.int64)
    geo = pd.to_numeric(merged["geo_pred"], errors="raise").to_numpy(dtype=np.int64)
    final = pd.to_numeric(merged["final_pred"], errors="raise").to_numpy(dtype=np.int64)
    acted = merged["acted"]
    if not bool(acted.isin([True, False, 0, 1]).all()):
        raise LockedEvaluationError("Sealed acted values must be boolean")
    action = acted.astype(bool).to_numpy()
    raw_correct = raw == y
    geo_correct = geo == y
    final_correct = final == y
    rescued = (~raw_correct) & final_correct
    harmed = raw_correct & (~final_correct)
    row_count = int(len(merged))
    seed_count = int(merged["training_seed"].nunique())
    action_count = int(action.sum())
    rescue_opportunities = int(((~raw_correct) & geo_correct).sum())
    override_precision = float(rescued.sum() / action_count) if action_count else None
    rescue_recall = (
        float(rescued.sum() / rescue_opportunities) if rescue_opportunities else None
    )
    harmful_override_rate = float(harmed.sum() / action_count) if action_count else None

    ordered_labels = labels.sort_values("row_uid", kind="mergesort")
    ordered_label_hash = hashlib.sha256(
        canonical_json_bytes(_frame_records(ordered_labels, ["row_uid", "label_id_dense"]))
    ).hexdigest()
    evaluation_event_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "composite_bundle_hash": bundle_sha256,
                "identity_projection_sha256": identity_sha256,
                "ordered_scoring_label_hash": ordered_label_hash,
                "prediction_hash": prediction_manifest["predictions_file_sha256"],
            }
        )
    ).hexdigest()
    receipt = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "artifact_role": "locked_test_scoring_receipt",
        "synthetic_fixture": True,
        "exploratory_only": True,
        "adaptive_reuse": adaptive_reuse,
        "protocol_id": prediction_manifest.get("protocol_id"),
        "evaluation_event_id": evaluation_event_id,
        "snapshot_id": prediction_manifest.get("snapshot_id", identity_sha256),
        "composite_bundle_hash": bundle_sha256,
        "prediction_hash": prediction_manifest["predictions_file_sha256"],
        "ordered_scoring_label_hash": ordered_label_hash,
        "ontology_hash": prediction_manifest.get("ontology_sha256"),
        "composite_bundle_file_sha256": bundle_sha256,
        "identity_projection_sha256": identity_sha256,
        "predictions_file_sha256": prediction_manifest["predictions_file_sha256"],
        "ordered_scoring_label_sha256": ordered_label_hash,
        "ontology_sha256": prediction_manifest.get("ontology_sha256"),
        "row_count": row_count,
        "unique_test_rows": int(merged["row_uid"].nunique()),
        "seed_count": seed_count,
        "metrics": {
            "top1_accuracy": float(final_correct.mean()),
            "delta_accuracy": float(final_correct.mean() - raw_correct.mean()),
            "intervention_coverage": float(action.mean()),
            "override_precision": override_precision,
            "rescue_recall": rescue_recall,
            "harmful_override_rate": harmful_override_rate,
            "rescued_count": int(rescued.sum()),
            "harmed_count": int(harmed.sum()),
            "rescued_minus_harmed": int(rescued.sum() - harmed.sum()),
        },
        "capabilities": {
            "fitting": False,
            "recalibration": False,
            "threshold_search": False,
            "policy_selection": False,
            "scientific_override": False,
        },
    }
    receipt_path = exclusive_write_json(output_path, receipt)
    receipt_sha256 = sha256_file(receipt_path)
    event = {
        "artifact_role": "locked_test_score_event",
        "adaptive_reuse": adaptive_reuse,
        "composite_bundle_file_sha256": bundle_sha256,
        "identity_projection_sha256": identity_sha256,
        "receipt_file_sha256": receipt_sha256,
        "evaluation_event_id": evaluation_event_id,
    }
    event_path = (
        registry_root
        / "snapshots"
        / identity_sha256
        / f"{bundle_sha256}.json"
    )
    exclusive_write_json(event_path, event)

    # Complete the reservation without overwriting it: a separate immutable
    # completion record points to the scoring event and receipt.
    completion_path = bundle_claim.with_name(f"{bundle_sha256}.completed.json")
    exclusive_write_json(completion_path, event)
    return {
        "status": "sealed",
        "receipt": str(receipt_path),
        "receipt_file_sha256": receipt_sha256,
        "adaptive_reuse": adaptive_reuse,
        "event": str(event_path),
    }

"""Append-only score ownership and canonical labels for the joint TCP capacity legacy-test study.

All four sealed prediction manifests and the declaration are bound before the
canonical label projection is acquired. Reuse opens only the local scoring copy.
"""

import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash,
    check_self_hash,
    exclusive_lock,
    file_hash,
    read_json,
    write_bytes,
    write_json,
)

STUDY_ID = "joint_tcp_router_capacity_legacy_test_v1"
_BUILD_FIELDS = (
    "owner_output_root",
    "prediction_manifest_sha256",
    "composite_bundle_file_sha256",
    "identity_projection_sha256",
    "declaration_sha256",
    "declaration_file_sha256",
    "seed_prediction_manifests",
)

LABEL_SCHEMA = pa.schema([
    pa.field(name, dtype, nullable=False)
    for name, dtype in [
        ("row_uid", pa.string()), ("file", pa.string()), ("plot_idx", pa.string()),
        ("label_id_dense", pa.int8()), ("canonical_l3_id", pa.int8()),
        ("label_name", pa.string()),
    ]
])


def _canonical_path(value, name):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be a canonical absolute path")
    path = Path(value)
    if str(path.resolve()) != value or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"{name} must be a canonical nonsymlink absolute path")
    return path


def _composite(build):
    return canonical_hash({
        "study_id": STUDY_ID,
        "declaration_sha256": build["declaration_sha256"],
        "declaration_file_sha256": build["declaration_file_sha256"],
        "identity_projection_sha256": build["identity_projection_sha256"],
        "seed_prediction_manifests": build["seed_prediction_manifests"],
    })


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"invalid registry {name}")
    return value


def _build(build):
    if not isinstance(build, dict) or not all(name in build for name in _BUILD_FIELDS):
        raise ValueError("registry build identity is incomplete")
    owner = build["owner_output_root"]
    if (
        not isinstance(owner, str)
        or not owner
        or not Path(owner).is_absolute()
        or str(Path(owner).resolve()) != owner
    ):
        raise ValueError("registry owner must be a canonical absolute path")
    for name in _BUILD_FIELDS[1:-1]:
        _digest(build[name], name)
    manifests = build["seed_prediction_manifests"]
    if not isinstance(manifests, dict) or set(manifests) != {"1", "2", "3", "4"}:
        raise ValueError("registry requires all four seed prediction manifests")
    for seed, record in manifests.items():
        if not isinstance(record, dict) or set(record) != {"path", "manifest_sha256", "file_sha256"}:
            raise ValueError("invalid seed prediction manifest binding")
        path = _canonical_path(record["path"], "prediction manifest")
        if path != Path(owner) / f"predictions/seed_{seed}/manifest.json":
            raise ValueError("prediction manifest path differs from seed owner")
        _digest(record["manifest_sha256"], "manifest hash")
        _digest(record["file_sha256"], "manifest file hash")
    if build["prediction_manifest_sha256"] != build["composite_bundle_file_sha256"]:
        raise ValueError(
            "registry composite must be the prediction manifest payload hash"
        )
    if build["prediction_manifest_sha256"] != _composite(build):
        raise ValueError("registry composite does not bind declaration and four predictions")
    return {name: build[name] for name in _BUILD_FIELDS}


def make_build_record(output_root, declaration, prediction_manifests):
    """Bind a saved declaration and exactly four complete prediction manifests."""
    root = _canonical_path(str(output_root), "output root")
    if declaration.get("study_id") != STUDY_ID:
        raise ValueError("declaration belongs to another study")
    if read_json(root / "declaration.json") != declaration:
        raise ValueError("declaration differs from saved declaration")
    declaration_hash = file_hash(root / "declaration.json")
    manifests = {str(seed): Path(path) for seed, path in prediction_manifests.items()}
    if len(prediction_manifests) != 4 or set(manifests) != {"1", "2", "3", "4"}:
        raise ValueError("exactly four seed prediction manifests are required")
    records = {}
    for seed, path in sorted(manifests.items()):
        value = read_json(path)
        if value.get("declaration_sha256") != declaration_hash:
            raise ValueError("prediction manifest declaration linkage differs")
        if value.get("schema_version") != "geo_helpfulness.confidnet_stage.v1":
            raise ValueError("prediction manifest schema differs")
        if value.get("role") != "joint_tcp_cleaned_test_predictions":
            raise ValueError("prediction manifest role differs")
        metadata = value.get("metadata")
        if (not isinstance(metadata, dict) or type(metadata.get("seed")) is not int
                or metadata["seed"] != int(seed)):
            raise ValueError("prediction manifest seed differs")
        files = value.get("file_sha256")
        if not isinstance(files, dict) or not files or "manifest.json" in files:
            raise ValueError("prediction manifest payload binding is empty or invalid")
        observed = {str(p.relative_to(path.parent)) for p in path.parent.rglob("*") if p.is_file()}
        if observed != set(files) | {"manifest.json"}:
            raise ValueError("prediction stage file set differs")
        for name, expected in files.items():
            payload = path.parent / name
            if (Path(name).is_absolute() or ".." in Path(name).parts
                    or payload.resolve().parent != path.parent.resolve()):
                raise ValueError("prediction payload must belong to its seed stage")
            _digest(expected, "prediction payload hash")
            if file_hash(payload) != expected:
                raise ValueError("prediction payload hash differs")
        records[seed] = {"path": str(path), "manifest_sha256": file_hash(path),
                         "file_sha256": file_hash(path)}
    value = {
        "study_id": STUDY_ID, "owner_output_root": str(root),
        "declaration_sha256": declaration_hash,
        "declaration_file_sha256": declaration_hash,
        "identity_projection_sha256": declaration["identity_manifest"]["identity_projection_sha256"],
        "seed_prediction_manifests": records,
        "historical_test_exposure": True, "adaptive_reuse": True,
        "independent_confirmation": False,
    }
    value["prediction_manifest_sha256"] = _composite(value)
    value["composite_bundle_file_sha256"] = value["prediction_manifest_sha256"]
    _build(value)
    return value


def _paths(registry_root, build):
    identity = _build(build)
    root = Path(registry_root).absolute()
    if root != root.resolve() or root.is_symlink():
        raise ValueError("registry root must be canonical and nonsymlink")
    snapshot = root / "snapshots" / identity["identity_projection_sha256"]
    bundle = identity["composite_bundle_file_sha256"]
    paths = {
        "root": root,
        "first": snapshot / "first_score_event.json",
        "bundle": root / "bundles" / f"{bundle}.json",
        "event": snapshot / f"{bundle}.json",
        "completed": root / "bundles" / f"{bundle}.completed.json",
    }
    for path in paths.values():
        for part in (path, *path.parents):
            if part.is_symlink():
                raise ValueError("registry paths cannot contain symlinks")
            if part == root:
                break
    return paths


def _owner(build):
    return {
        "status": "reserved",
        "study_id": STUDY_ID,
        **_build(build),
        "historical_test_exposure": True,
        "adaptive_reuse": True,
    }


def _first(paths, build):
    first = read_json(paths["first"])
    if (
        first.get("status") != "reserved"
        or first.get("identity_projection_sha256")
        != build["identity_projection_sha256"]
    ):
        raise ValueError("registry first claim identity is invalid")
    _digest(first.get("composite_bundle_file_sha256"), "first composite hash")
    return first


def _check_existing_events(paths, build):
    for key in ("event", "completed"):
        if not paths[key].exists():
            continue
        event = read_json(paths[key])
        if event.get("study_id") != STUDY_ID or any(
            event.get(name) != build[name] for name in _BUILD_FIELDS
        ):
            raise ValueError("registry event belongs to another score owner")


def _validate_reservation(paths, build, reservation):
    first = _first(paths, build)
    expected = {
        **_owner(build),
        "prior_registered_score_event": first != _owner(build),
        "first_score_event_file_sha256": file_hash(paths["first"]),
    }
    if reservation != expected or read_json(paths["bundle"]) != expected:
        raise ValueError("registry reservation ownership or first-claim hash differs")
    _check_existing_events(paths, build)


def reserve_event(registry_root, build):
    """Reserve before label acquisition; retry only the identical score owner."""
    paths = _paths(registry_root, build)
    owner = _owner(build)
    with exclusive_lock(paths["root"]):
        # Preflight existing claims before writing a previously absent first claim.
        if (
            any(paths[name].exists() for name in ("event", "completed"))
            and not paths["bundle"].exists()
        ):
            raise ValueError("registry completion has no reservation")
        if paths["bundle"].exists():
            existing = read_json(paths["bundle"])
            if any(existing.get(name) != value for name, value in owner.items()):
                raise ValueError(
                    "prediction bundle already reserved by another score owner"
                )
            if not paths["first"].exists():
                raise ValueError("registry bundle has no first-score claim")
        _check_existing_events(paths, build)
        if paths["first"].exists():
            first = _first(paths, build)
        else:
            # First-before-bundle ordering makes this interruption resumable.
            write_json(paths["first"], owner)
            first = owner
        reservation = {
            **owner,
            "prior_registered_score_event": first != owner,
            "first_score_event_file_sha256": file_hash(paths["first"]),
        }
        write_json(paths["bundle"], reservation)
        _validate_reservation(paths, build, reservation)
    return reservation


def validate_reservation(registry_root, build, reservation):
    """Check immutable ownership and the first claim before opening labels."""
    paths = _paths(registry_root, build)
    with exclusive_lock(paths["root"]):
        _validate_reservation(paths, build, reservation)


def _event(build, reservation, receipt, receipt_path):
    if not isinstance(receipt, dict):
        raise ValueError("registry receipt must be an object")
    check_self_hash(receipt, "receipt_sha256")
    _digest(receipt.get("evaluation_event_id"), "evaluation event id")
    if receipt.get("study_id") != STUDY_ID or any(
        receipt.get(name) != build[name] for name in _BUILD_FIELDS
    ):
        raise ValueError("registry receipt belongs to another score owner")
    receipt_path = Path(receipt_path)
    if read_json(receipt_path) != receipt:
        raise ValueError("registry receipt content differs from the saved receipt")
    return {
        "artifact_role": "locked_test_score_event",
        "study_id": STUDY_ID,
        "adaptive_reuse": True,
        "prior_registered_score_event": reservation["prior_registered_score_event"],
        **_build(build),
        "receipt_file_sha256": file_hash(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "evaluation_event_id": receipt["evaluation_event_id"],
    }


def _validate_event(paths, build, reservation, event):
    _validate_reservation(paths, build, reservation)
    if read_json(paths["event"]) != event or read_json(paths["completed"]) != event:
        raise ValueError("registry completion differs from the sealed score event")


def finish_event(registry_root, build, reservation, receipt, receipt_path):
    """Append the event and completion records after the immutable receipt exists."""
    paths = _paths(registry_root, build)
    with exclusive_lock(paths["root"]):
        _validate_reservation(paths, build, reservation)
        event = _event(build, reservation, receipt, receipt_path)
        # Detect either collision before appending the other half of a pair.
        for name in ("event", "completed"):
            if paths[name].exists() and read_json(paths[name]) != event:
                raise ValueError(
                    "registry completion already belongs to another receipt"
                )
        write_json(paths["event"], event)
        write_json(paths["completed"], event)
        _validate_event(paths, build, reservation, event)
    return event


def validate_event(registry_root, build, reservation, receipt, receipt_path):
    """Validate an existing event without appending or replacing registry records."""
    paths = _paths(registry_root, build)
    with exclusive_lock(paths["root"]):
        event = _event(build, reservation, receipt, receipt_path)
        _validate_event(paths, build, reservation, event)


def validate_labels(labels, declaration):
    """Require the exact canonical identity, ontology and one label per plot."""
    if list(labels.columns) != LABEL_SCHEMA.names or labels.isna().any().any():
        raise ValueError("invalid scoring label schema or missing values")
    identity = declaration["identity_manifest"]
    frame = pd.DataFrame(identity["rows"])
    columns = ["row_uid", "file", "plot_idx"]
    if (not set(columns).issubset(frame) or frame[columns].isna().any().any()
            or frame.row_uid.duplicated().any() or frame.file.str.casefold().duplicated().any()
            or len(frame) != identity["row_count"]
            or frame.plot_idx.nunique() != identity["plot_count"]):
        raise ValueError("invalid canonical cleaned-test identity")
    frame = frame.sort_values("row_uid", kind="stable").reset_index(drop=True)
    labels = labels.sort_values("row_uid", kind="stable").reset_index(drop=True)
    if not np.array_equal(labels[columns].to_numpy(), frame[columns].to_numpy()):
        raise ValueError("scoring labels differ from exact frozen identity")
    ontology = declaration["ontology"]
    if (len(ontology) != 18 or {r["dense_id"] for r in ontology} != set(range(18))
            or len({r["canonical_l3_id"] for r in ontology}) != 18
            or len({r["label_name"] for r in ontology}) != 18):
        raise ValueError("invalid frozen 18-class ontology")
    by_name = {record["label_name"]: record for record in ontology}
    for row in labels.itertuples(index=False):
        spec = by_name.get(row.label_name)
        if (spec is None or row.label_id_dense != spec["dense_id"]
                or row.canonical_l3_id != spec["canonical_l3_id"]):
            raise ValueError("scoring labels differ from frozen ontology")
    if not labels.groupby("plot_idx").label_id_dense.nunique().eq(1).all():
        raise ValueError("scoring labels disagree within a plot")
    return labels


def acquire_labels(registry_root, build, reservation, declaration, score_root):
    """Acquire after reservation; return (canonical labels, source receipt).

    Original test tables are never opened. A completed local copy is sufficient
    for reuse even when the earlier study's label projection is unavailable.
    """
    validate_reservation(registry_root, build, reservation)
    root = _canonical_path(build["owner_output_root"], "score owner")
    if (read_json(root / "declaration.json") != declaration
            or file_hash(root / "declaration.json") != build["declaration_file_sha256"]
            or declaration["identity_manifest"]["identity_projection_sha256"]
            != build["identity_projection_sha256"]):
        raise ValueError("label acquisition declaration differs from reservation")
    rebuilt = make_build_record(root, declaration, {
        seed: record["path"] for seed, record in build["seed_prediction_manifests"].items()
    })
    if _build(rebuilt) != _build(build):
        raise ValueError("bound prediction manifests changed before label acquisition")
    score_root = _canonical_path(str(score_root), "score directory")
    if score_root != root / "label_acquisition":
        raise ValueError("label acquisition must use reserved owner's label_acquisition directory")
    source = declaration["label_source"]
    if not isinstance(source, dict) or set(source) != {"path", "file_sha256"}:
        raise ValueError("canonical label source is not fully bound")
    source_path = _canonical_path(source["path"], "canonical label source")
    _digest(source["file_sha256"], "canonical label source hash")
    local_path = score_root / "labels.parquet"
    receipt_path = score_root / "source_receipt.json"
    if not local_path.exists():
        if receipt_path.exists():
            raise ValueError("label acquisition receipt has no local label copy")
        # This is the only branch opening the authenticated parent projection.
        content = source_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != source["file_sha256"]:
            raise ValueError("canonical label source hash differs")
        write_bytes(local_path, content)
    if file_hash(local_path) != source["file_sha256"]:
        raise ValueError("local label copy differs from authenticated parent")
    table = pq.read_table(local_path)
    if table.schema != LABEL_SCHEMA:
        raise ValueError("canonical label parquet schema differs")
    labels = validate_labels(table.to_pandas(), declaration)
    receipt = {
        "study_id": STUDY_ID,
        "build_sha256": canonical_hash(build),
        "reservation_sha256": canonical_hash(reservation),
        "parent_label_file": source["path"],
        "label_file_sha256": source["file_sha256"],
        "ordered_scoring_label_sha256": canonical_hash(labels.to_dict(orient="records")),
        "original_test_sources_opened": False,
        "all_four_prediction_manifests_bound_before_label_access": True,
    }
    write_json(receipt_path, receipt)
    return labels, receipt


# Descriptive public names are aliases, preserving the established workflow API.
acquire_canonical_labels = acquire_labels
complete_event = finish_event

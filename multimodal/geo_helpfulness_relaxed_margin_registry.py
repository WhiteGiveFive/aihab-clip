"""Append-only locked-test events for the relaxed-guardrails margin study."""

from pathlib import Path
import re

from multimodal.geo_helpfulness_router_m4_common import (
    check_self_hash,
    exclusive_lock,
    file_hash,
    read_json,
    write_json,
)

STUDY_ID = "relaxed_guardrails_margin_v1"
_BUILD_FIELDS = (
    "owner_output_root",
    "prediction_manifest_sha256",
    "composite_bundle_file_sha256",
    "identity_projection_sha256",
)


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
    for name in _BUILD_FIELDS[1:]:
        _digest(build[name], name)
    if build["prediction_manifest_sha256"] != build["composite_bundle_file_sha256"]:
        raise ValueError(
            "registry composite must be the prediction manifest payload hash"
        )
    return {name: build[name] for name in _BUILD_FIELDS}


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
    if receipt.get("study_id", STUDY_ID) != STUDY_ID or any(
        name in receipt and receipt[name] != build[name] for name in _BUILD_FIELDS
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

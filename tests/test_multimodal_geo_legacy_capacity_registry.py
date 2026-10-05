"""Registry ownership, append-only recovery, and frozen predecessor checks."""

from copy import deepcopy
import json

import pytest

from multimodal import geo_helpfulness_legacy_capacity_registry as registry
from multimodal.geo_helpfulness_router_m4_common import canonical_hash


def _build(tmp_path):
    return {
        "owner_output_root": str(tmp_path / "new-output"),
        "prediction_manifest_sha256": "a" * 64,
        "composite_bundle_file_sha256": "a" * 64,
        "identity_projection_sha256": "b" * 64,
    }


def _snapshot(root):
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _receipt(tmp_path, build):
    value = {"study_id": registry.STUDY_ID, **build, "evaluation_event_id": "d" * 64}
    value["receipt_sha256"] = canonical_hash(value)
    path = tmp_path / "receipt.json"
    registry.write_json(path, value)
    return value, path


def _old_first(root, build):
    paths = registry._paths(root, build)
    old = {
        "status": "reserved",
        "study_id": "legacy_frozen_router_transfer_v1",
        "owner_output_root": str(root.parent / "old-output"),
        "composite_bundle_file_sha256": "c" * 64,
        "identity_projection_sha256": build["identity_projection_sha256"],
    }
    registry.write_json(paths["first"], old)
    return paths["first"]


def test_append_preserves_prior_event_and_repeated_calls_preserve_bytes(tmp_path):
    root, build = tmp_path / "registry", _build(tmp_path)
    first = _old_first(root, build)
    old_bytes, old_mtime = first.read_bytes(), first.stat().st_mtime_ns
    reservation = registry.reserve_event(root, build)
    assert reservation["adaptive_reuse"] is True
    assert reservation["prior_registered_score_event"] is True
    registry.validate_reservation(root, build, reservation)
    receipt, path = _receipt(tmp_path, build)
    event = registry.finish_event(root, build, reservation, receipt, path)
    assert event["study_id"] == registry.STUDY_ID
    assert event["receipt_file_sha256"] == registry.file_hash(path)
    assert event["prediction_manifest_sha256"] == build["prediction_manifest_sha256"]
    before = _snapshot(root)
    assert len(before) == 4
    assert registry.reserve_event(root, build) == reservation
    assert registry.finish_event(root, build, reservation, receipt, path) == event
    assert registry.validate_event(root, build, reservation, receipt, path) is None
    assert _snapshot(root) == before
    assert (first.read_bytes(), first.stat().st_mtime_ns) == (old_bytes, old_mtime)


def test_first_claim_crash_is_resumable(tmp_path, monkeypatch):
    root, build = tmp_path / "registry", _build(tmp_path)
    paths = registry._paths(root, build)
    original_write = registry.write_json

    def interrupted(path, value):
        if path == paths["bundle"]:
            raise RuntimeError("crash after first claim")
        return original_write(path, value)

    monkeypatch.setattr(registry, "write_json", interrupted)
    with pytest.raises(RuntimeError, match="crash"):
        registry.reserve_event(root, build)
    assert paths["first"].exists() and not paths["bundle"].exists()
    first = (paths["first"].read_bytes(), paths["first"].stat().st_mtime_ns)
    monkeypatch.setattr(registry, "write_json", original_write)
    reservation = registry.reserve_event(root, build)
    assert reservation["prior_registered_score_event"] is False
    assert reservation["adaptive_reuse"] is True
    assert (paths["first"].read_bytes(), paths["first"].stat().st_mtime_ns) == first


def test_event_then_completion_crash_is_resumable(tmp_path, monkeypatch):
    root, build = tmp_path / "registry", _build(tmp_path)
    reservation = registry.reserve_event(root, build)
    receipt, path = _receipt(tmp_path, build)
    paths = registry._paths(root, build)
    original_write = registry.write_json

    def interrupted(target, value):
        if target == paths["completed"]:
            raise RuntimeError("crash after event")
        return original_write(target, value)

    monkeypatch.setattr(registry, "write_json", interrupted)
    with pytest.raises(RuntimeError, match="crash"):
        registry.finish_event(root, build, reservation, receipt, path)
    assert paths["event"].exists() and not paths["completed"].exists()
    before = _snapshot(root)
    monkeypatch.setattr(registry, "write_json", original_write)
    registry.finish_event(root, build, reservation, receipt, path)
    assert all(_snapshot(root)[name] == record for name, record in before.items())
    registry.validate_event(root, build, reservation, receipt, path)


@pytest.mark.parametrize("field", ["owner_output_root", "identity_projection_sha256"])
def test_bundle_collision_rejected_without_writes(tmp_path, field):
    root, build = tmp_path / "registry", _build(tmp_path)
    registry.reserve_event(root, build)
    other = dict(build)
    other[field] = (
        str(tmp_path / "other-output") if field == "owner_output_root" else "e" * 64
    )
    before = _snapshot(root)
    with pytest.raises(ValueError, match="another score owner"):
        registry.reserve_event(root, other)
    assert _snapshot(root) == before


@pytest.mark.parametrize("target", ["first", "bundle"])
def test_reservation_tamper_rejected_before_scoring(tmp_path, target):
    root, build = tmp_path / "registry", _build(tmp_path)
    _old_first(root, build)
    reservation = registry.reserve_event(root, build)
    path = registry._paths(root, build)[target]
    value = registry.read_json(path)
    value["unknown_mutation"] = True
    path.chmod(0o644)
    path.write_text(json.dumps(value))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="reservation ownership"):
        registry.validate_reservation(root, build, reservation)
    assert _snapshot(root) == before


def test_different_saved_receipt_cannot_complete_reserved_bundle(tmp_path):
    root, build = tmp_path / "registry", _build(tmp_path)
    reservation = registry.reserve_event(root, build)
    receipt, path = _receipt(tmp_path, build)
    changed = deepcopy(receipt)
    changed["evaluation_event_id"] = "e" * 64
    changed.pop("receipt_sha256")
    changed["receipt_sha256"] = canonical_hash(changed)
    before = _snapshot(root)
    with pytest.raises(ValueError, match="saved receipt"):
        registry.finish_event(root, build, reservation, changed, path)
    assert _snapshot(root) == before


@pytest.mark.parametrize("target", ["event", "completed"])
def test_completion_tamper_rejected_without_overwrite(tmp_path, target):
    root, build = tmp_path / "registry", _build(tmp_path)
    reservation = registry.reserve_event(root, build)
    receipt, path = _receipt(tmp_path, build)
    registry.finish_event(root, build, reservation, receipt, path)
    target_path = registry._paths(root, build)[target]
    value = registry.read_json(target_path)
    value["evaluation_event_id"] = "e" * 64
    target_path.chmod(0o644)
    target_path.write_text(json.dumps(value))
    before = _snapshot(root)
    with pytest.raises(ValueError):
        registry.validate_event(root, build, reservation, receipt, path)
    with pytest.raises(ValueError, match="another receipt"):
        registry.finish_event(root, build, reservation, receipt, path)
    assert _snapshot(root) == before


def test_lock_contention_makes_no_registry_changes(tmp_path):
    root, build = tmp_path / "registry", _build(tmp_path)
    with registry.exclusive_lock(root):
        with pytest.raises(ValueError, match="owns this workflow lock"):
            registry.reserve_event(root, build)
    assert not root.exists()


def test_orphan_completion_is_not_adopted_or_given_a_first_claim(tmp_path):
    root, build = tmp_path / "registry", _build(tmp_path)
    paths = registry._paths(root, build)
    registry.write_json(paths["completed"], {"study_id": registry.STUDY_ID, **build})
    before = _snapshot(root)
    with pytest.raises(ValueError, match="no reservation"):
        registry.reserve_event(root, build)
    assert _snapshot(root) == before
    assert not paths["first"].exists()


@pytest.mark.parametrize(
    "problem", ["path_hash", "different_hashes", "relative_owner", "symlink"]
)
def test_invalid_paths_and_identities_rejected_before_writes(tmp_path, problem):
    root, build = tmp_path / "registry", _build(tmp_path)
    if problem == "path_hash":
        build["identity_projection_sha256"] = "../escape"
    elif problem == "different_hashes":
        build["prediction_manifest_sha256"] = "e" * 64
    elif problem == "relative_owner":
        build["owner_output_root"] = "relative"
    else:
        root.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (root / "bundles").symlink_to(elsewhere, target_is_directory=True)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError):
        registry.reserve_event(root, build)
    assert _snapshot(tmp_path) == before

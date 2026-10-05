"""Margin registry ownership, recovery, and four frozen predecessor checks."""

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from multimodal import geo_helpfulness_relaxed_margin_registry as registry
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


def _four_completed_studies(root, build):
    """Populate the thirteen immutable records preceding this fifth study."""
    first = None
    for study_id, digest in (
        ("legacy_frozen_router_transfer_v1", "c" * 64),
        ("legacy_capacity_router_transfer_v1", "f" * 64),
        ("relaxed_guardrails_lr_v1", "7" * 64),
        ("relaxed_guardrails_rf_v1", "8" * 64),
    ):
        prior_build = {
            **build,
            "owner_output_root": str(root.parent / study_id),
            "prediction_manifest_sha256": digest,
            "composite_bundle_file_sha256": digest,
        }
        paths = registry._paths(root, prior_build)
        owner = {
            "status": "reserved",
            "study_id": study_id,
            **prior_build,
            "historical_test_exposure": True,
            "adaptive_reuse": True,
        }
        is_first = first is None
        if is_first:
            registry.write_json(paths["first"], owner)
            first = paths["first"]
        reservation = {
            **owner,
            "prior_registered_score_event": not is_first,
            "first_score_event_file_sha256": registry.file_hash(first),
        }
        event = {
            "artifact_role": "locked_test_score_event",
            "study_id": study_id,
            **prior_build,
            "adaptive_reuse": True,
            "prior_registered_score_event": not is_first,
            "receipt_file_sha256": "1" * 64,
            "receipt_sha256": "2" * 64,
            "evaluation_event_id": "3" * 64,
        }
        registry.write_json(paths["bundle"], reservation)
        registry.write_json(paths["event"], event)
        registry.write_json(paths["completed"], event)
    assert len(_snapshot(root)) == 13
    return first


def test_fifth_study_appends_exactly_three_records_and_preserves_all_predecessors(
    tmp_path,
):
    root, build = tmp_path / "registry", _build(tmp_path)
    first = _four_completed_studies(root, build)
    before = _snapshot(root)
    reservation = registry.reserve_event(root, build)
    assert registry.STUDY_ID == "relaxed_guardrails_margin_v1"
    assert reservation["prior_registered_score_event"] is True
    assert reservation["adaptive_reuse"] is True
    assert reservation["first_score_event_file_sha256"] == registry.file_hash(first)
    registry.validate_reservation(root, build, reservation)
    receipt, receipt_path = _receipt(tmp_path, build)
    event = registry.finish_event(root, build, reservation, receipt, receipt_path)
    after = _snapshot(root)
    paths = registry._paths(root, build)
    assert set(after) - set(before) == {
        str(paths[name]) for name in ("bundle", "event", "completed")
    }
    assert len(after) == 16
    assert all(after[name] == saved for name, saved in before.items())
    assert event["evaluation_event_id"] == receipt["evaluation_event_id"]
    assert event["receipt_file_sha256"] == registry.file_hash(receipt_path)
    assert paths["event"].read_bytes() == paths["completed"].read_bytes()
    assert registry.reserve_event(root, build) == reservation
    assert (
        registry.finish_event(root, build, reservation, receipt, receipt_path) == event
    )
    registry.validate_event(root, build, reservation, receipt, receipt_path)
    assert _snapshot(root) == after


@pytest.mark.parametrize("interruption", ["bundle", "event", "completed"])
def test_fifth_study_partial_publication_preserves_predecessors_and_resumes(
    tmp_path, monkeypatch, interruption
):
    root, build = tmp_path / "registry", _build(tmp_path)
    _four_completed_studies(root, build)
    before = _snapshot(root)
    paths = registry._paths(root, build)
    original_write = registry.write_json

    def interrupted(path, value):
        if path == paths[interruption]:
            raise RuntimeError("synthetic publication interruption")
        return original_write(path, value)

    monkeypatch.setattr(registry, "write_json", interrupted)
    with pytest.raises(RuntimeError, match="publication interruption"):
        reservation = registry.reserve_event(root, build)
        receipt, receipt_path = _receipt(tmp_path, build)
        registry.finish_event(root, build, reservation, receipt, receipt_path)
    partial = _snapshot(root)
    assert all(partial[name] == saved for name, saved in before.items())
    monkeypatch.setattr(registry, "write_json", original_write)
    reservation = registry.reserve_event(root, build)
    receipt, receipt_path = _receipt(tmp_path, build)
    registry.finish_event(root, build, reservation, receipt, receipt_path)
    registry.validate_event(root, build, reservation, receipt, receipt_path)
    after = _snapshot(root)
    assert len(after) == 16
    assert all(after[name] == saved for name, saved in partial.items())


@pytest.mark.parametrize("prior_digest", ["c" * 64, "f" * 64, "7" * 64, "8" * 64])
def test_prior_study_bundle_hash_cannot_be_adopted_by_fifth_study(
    tmp_path, prior_digest
):
    root, build = tmp_path / "registry", _build(tmp_path)
    _four_completed_studies(root, build)
    build["prediction_manifest_sha256"] = prior_digest
    build["composite_bundle_file_sha256"] = prior_digest
    before = _snapshot(root)
    with pytest.raises(ValueError, match="another score owner"):
        registry.reserve_event(root, build)
    assert _snapshot(root) == before


@pytest.mark.parametrize("operation", ["validate_reservation", "finish", "validate"])
def test_all_event_operations_enforce_exclusive_lock(tmp_path, operation):
    root, build = tmp_path / "registry", _build(tmp_path)
    reservation = registry.reserve_event(root, build)
    receipt, receipt_path = _receipt(tmp_path, build)
    registry.finish_event(root, build, reservation, receipt, receipt_path)
    before = _snapshot(root)
    with registry.exclusive_lock(root):
        with pytest.raises(ValueError, match="owns this workflow lock"):
            if operation == "validate_reservation":
                registry.validate_reservation(root, build, reservation)
            elif operation == "finish":
                registry.finish_event(root, build, reservation, receipt, receipt_path)
            else:
                registry.validate_event(root, build, reservation, receipt, receipt_path)
    assert _snapshot(root) == before


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


def test_registry_logic_matches_frozen_rf_predecessor_except_study_identity():
    def normalized(path):
        tree = ast.parse(path.read_text())
        assert isinstance(tree.body[0], ast.Expr)
        assert isinstance(tree.body[0].value, ast.Constant)
        tree.body[0].value.value = "study-specific module documentation"
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "STUDY_ID"
            ):
                node.value = ast.Constant(value="study-specific identifier")
        return ast.dump(tree, include_attributes=False)

    current = Path(registry.__file__)
    predecessor = current.with_name("geo_helpfulness_relaxed_rf_registry.py")
    assert normalized(current) == normalized(predecessor)

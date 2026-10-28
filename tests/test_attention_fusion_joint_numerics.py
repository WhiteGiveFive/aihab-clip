"""Keep exploratory drift acceptance bounded, cumulative, and reviewable."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from multimodal import attention_fusion_joint_numerics as numerics
from multimodal.attention_fusion_joint_numerics import NumericalAudit, normalize_numerical_policy


def fixture_pair(steps=64):
    expected = np.array([-0.0005629033548757434], dtype=np.float32)
    actual = expected.copy()
    actual.view(np.uint32)[0] -= steps
    return torch.from_numpy(actual), torch.from_numpy(expected)


def exploratory(path=None):
    return NumericalAudit({"mode": "exploratory"}, path)


def test_default_is_strict_and_explicit_mode_fills_fixed_bounds():
    assert normalize_numerical_policy()["mode"] == "strict"
    policy = normalize_numerical_policy({"mode": "exploratory"})
    assert policy["frozen_atol"] == 1e-8
    assert policy["frozen_rtol"] == 1e-5
    for bad in ({"mode": "ignore"}, {"mode": "exploratory", "unknown": True},
                {"frozen_atol": 1e-7}, {"max_frozen_changed_coordinates": True}):
        with pytest.raises(ValueError):
            normalize_numerical_policy(bad)


def test_constructor_does_not_create_fresh_audit(tmp_path):
    path = tmp_path / "audit.json"
    audit = exploratory(path)
    assert not path.exists()
    audit.compare(1, 1, "boundary", kind="copy")
    assert path.exists()


def test_observed_single_bit_difference_is_audited(tmp_path):
    actual, expected = fixture_pair()
    path = tmp_path / "audit.json"
    audit = exploratory(path)
    report = audit.compare({"weight": actual}, {"weight": expected}, "epoch 1", namespace="visual_frozen")
    assert report["status"] == "accepted_tolerance"
    assert report["differing_elements"] == 1
    assert report["max_absolute_error"] == pytest.approx(3.725290298461914e-9)
    item = report["tensors"][0]
    assert item["canonical_sha256"] != item["observed_sha256"]
    assert item["changed_elements"][0]["index"] == [0]
    assert item["changed_elements"][0]["expected"] == expected.item()
    assert item["changed_elements"][0]["actual"] == actual.item()
    saved = json.loads(path.read_text())
    assert saved["summary"]["frozen_changed_coordinates"] == 1
    assert saved["summary"]["integrity_deviations_detected"]
    assert len(saved["unique_deviations"]) == 1


@pytest.mark.parametrize("kind", ["frozen", "recomputed", "copy"])
def test_strict_rejects_small_drift(kind):
    actual, expected = fixture_pair()
    with pytest.raises(ValueError, match="Byte-exact"):
        NumericalAudit().compare(actual, expected, "strict", kind=kind)


def test_exploratory_copy_stays_exact():
    actual, expected = fixture_pair()
    with pytest.raises(ValueError, match="Byte-exact"):
        exploratory().compare(actual, expected, "copy", kind="copy")


@pytest.mark.parametrize("actual,expected", [
    (np.array([1e-12]), np.array([0.0])),  # No zero-to-nonzero relaxation.
    (np.array([1.0001e-12]), np.array([1e-12])),  # Absolute passes, relative fails.
    (np.array([1.00000002]), np.array([1.0])),  # Relative passes, absolute fails.
])
def test_both_frozen_bounds_must_pass(actual, expected):
    with pytest.raises(ValueError, match="numerical bound"):
        exploratory().compare(actual, expected, "bounded")


def test_repeated_same_deviation_does_not_consume_budget_and_resumes(tmp_path):
    actual, expected = fixture_pair()
    path = tmp_path / "audit.json"
    audit = exploratory(path)
    for epoch in range(3):
        audit.compare(actual, expected, f"epoch {epoch}", namespace="visual_frozen")
    audit = exploratory(path)
    audit.compare(actual, expected, "checkpoint load", namespace="visual_frozen")
    assert audit.summary["frozen_changed_coordinates"] == 1
    assert audit.summary["frozen_unique_deviations"] == 1
    assert audit.summary["accepted_numerical_comparisons"] == 4


def test_coordinate_budget_is_cumulative_across_tensors_and_resume(tmp_path):
    actual, expected = fixture_pair()
    audit = exploratory(tmp_path / "audit.json")
    for index in range(8):
        audit.compare({str(index): actual}, {str(index): expected}, f"tensor {index}")
    audit = exploratory(tmp_path / "audit.json")
    with pytest.raises(ValueError, match="changed-coordinate budget"):
        audit.compare({"ninth": actual}, {"ninth": expected}, "ninth tensor")
    assert audit.summary["frozen_changed_coordinates"] == 8
    assert audit.summary["rejected_comparisons"] == 1


def test_unique_deviation_budget_counts_changed_bits_not_checks():
    audit = exploratory()
    for steps in range(1, 17):
        actual, expected = fixture_pair(steps)
        audit.compare(actual, expected, f"deviation {steps}")
        audit.compare(actual, expected, f"repeat {steps}")
    actual, expected = fixture_pair(17)
    with pytest.raises(ValueError, match="unique-deviation budget"):
        audit.compare(actual, expected, "seventeenth deviation")
    assert audit.summary["frozen_changed_coordinates"] == 1
    assert audit.summary["frozen_unique_deviations"] == 16


def test_canonical_reference_cannot_move_to_observed_values():
    actual, expected = fixture_pair()
    audit = exploratory()
    audit.compare(actual, expected, "first")
    with pytest.raises(ValueError, match="Canonical frozen reference changed"):
        audit.compare(actual, actual, "attempted reset")


def test_chunk_boundaries_do_not_hide_budget_excess(monkeypatch):
    monkeypatch.setattr(numerics, "_CHUNK_ELEMENTS", 2)
    actual, expected = fixture_pair()
    with pytest.raises(ValueError, match="changed-coordinate budget"):
        exploratory().compare(actual.repeat(10), expected.repeat(10), "ten drifts")


def test_recomputation_tolerance_recurses_but_nonfloats_remain_exact():
    expected = {"logits": np.array([[0.0, 1.0]], dtype=np.float32),
                "row_uid": np.array(["row1"]), "label": np.array([1]),
                "optimizer": {0: {"momentum": torch.tensor([0.1]), "count": 1}},
                "metadata": (None, True, "validation")}
    actual = deepcopy(expected)
    actual["logits"][0, 0] = 5e-7
    actual["optimizer"][0]["momentum"] += 1e-7
    audit = exploratory()
    report = audit.compare(actual, expected, "recomputed", kind="recomputed")
    assert report["status"] == "accepted_tolerance"
    assert report["max_relative_error"] is None  # Zero reference; bounded by atol.
    assert audit.summary["frozen_changed_coordinates"] == 0
    assert not audit.summary["integrity_deviations_detected"]
    for key, changed in (("row_uid", np.array(["row2"])), ("label", np.array([0]))):
        bad = deepcopy(expected)
        bad[key] = changed
        with pytest.raises(ValueError, match="Byte-exact"):
            audit.compare(bad, expected, key, kind="recomputed")
    actual["optimizer"][0]["count"] = 2
    with pytest.raises(ValueError, match="Nonfloating"):
        audit.compare(actual, expected, "counter", kind="recomputed")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_rejected_even_when_bytes_match(value, tmp_path):
    path = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="Nonfinite"):
        exploratory(path).compare(torch.tensor([value]), torch.tensor([value]), "nonfinite")
    assert json.loads(path.read_text())["events"][0]["status"] == "rejected"


@pytest.mark.parametrize("actual,expected", [
    (torch.ones(1), torch.ones(2)),
    (torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.float64)),
    ({"a": 1}, {"b": 1}),
    ([1], (1,)),
    (1, True),
])
def test_structure_and_types_are_exact(actual, expected):
    with pytest.raises(ValueError):
        exploratory().compare(actual, expected, "structure", kind="recomputed")


def test_frozen_zero_is_byte_exact_even_in_exploratory_mode():
    with pytest.raises(ValueError, match="Byte-exact"):
        NumericalAudit().compare(0.0, -0.0, "signed zero")
    with pytest.raises(ValueError, match="numerical bound"):
        exploratory().compare(0.0, -0.0, "signed zero")
    with pytest.raises(ValueError, match="numerical bound"):
        exploratory().compare(-0.0, 0.0, "reverse signed zero")
    report = exploratory().compare(0.0, -0.0, "recomputed signed zero", kind="recomputed")
    assert report["status"] == "accepted_tolerance"
    assert report["max_absolute_error"] == 0


def test_bfloat16_exact_comparison_and_empty_arrays():
    value = torch.tensor([0.5], dtype=torch.bfloat16)
    audit = exploratory()
    assert audit.compare(value, value.clone(), "bfloat")["exact"]
    assert audit.compare(np.empty((0, 3)), np.empty((0, 3)), "empty", namespace="empty")["exact"]


def test_resume_policy_cannot_change(tmp_path):
    path = tmp_path / "audit.json"
    exploratory(path).compare(1, 1, "initial", kind="copy")
    with pytest.raises(ValueError, match="policy changed"):
        NumericalAudit(path=path)


def test_resume_rejects_changed_audit_bytes(tmp_path):
    path = tmp_path / "audit.json"
    actual, expected = fixture_pair()
    exploratory(path).compare(actual, expected, "first")
    saved = json.loads(path.read_text())
    saved["changed_coordinates"] = {}
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checksum differs"):
        exploratory(path)


def test_snapshot_is_independent_and_does_not_create_a_file(tmp_path):
    path = tmp_path / "audit.json"
    audit = exploratory(path)
    snapshot = audit.snapshot
    assert not path.exists()
    assert snapshot["audit_sha256"] == numerics._audit_digest(snapshot)
    snapshot["events"].append({"changed": True})
    assert not audit.snapshot["events"]


def test_valid_older_audit_is_rejected_against_committed_checkpoint(tmp_path):
    path = tmp_path / "audit.json"
    audit = exploratory(path)
    audit.compare(1, 1, "before checkpoint", kind="copy")
    older = path.read_text()
    actual, expected = fixture_pair()
    audit.compare(actual, expected, "committed frozen drift")
    committed = audit.snapshot
    path.write_text(older)  # Valid checksum, but missing the committed drift.
    resumed = exploratory(path)
    with pytest.raises(ValueError, match="older than or differ"):
        resumed.require_committed_prefix(committed)


def test_committed_prefix_allows_new_interrupted_events_and_drift(tmp_path):
    path = tmp_path / "audit.json"
    audit = exploratory(path)
    actual, expected = fixture_pair(1)
    audit.compare(actual, expected, "committed")
    committed = audit.snapshot
    actual, expected = fixture_pair(2)
    audit.compare(actual, expected, "after checkpoint")
    audit.compare(2, 2, "later boundary", kind="copy")
    resumed = exploratory(path)
    resumed.require_committed_prefix(committed)
    assert resumed.summary["frozen_unique_deviations"] == 2
    assert resumed.snapshot["changed_coordinates"] == committed["changed_coordinates"]


def test_missing_audit_cannot_resume_a_committed_checkpoint(tmp_path):
    actual, expected = fixture_pair()
    source = exploratory()
    source.compare(actual, expected, "committed")
    fresh = exploratory(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="missing"):
        fresh.require_committed_prefix(source.snapshot)


@pytest.mark.parametrize("key", ["canonical_references", "changed_coordinates", "unique_deviations"])
def test_prefix_rejects_forgotten_committed_budget_evidence(key):
    actual, expected = fixture_pair()
    audit = exploratory()
    audit.compare(actual, expected, "committed")
    committed = audit.snapshot
    audit._state[key] = {}  # Even recomputing its self-checksum cannot legitimize this loss.
    with pytest.raises(ValueError, match="lost or changed committed evidence"):
        audit.require_committed_prefix(committed)


def test_portable_snapshot_restore_preserves_budget_and_evidence(tmp_path):
    source = exploratory()
    actual, expected = fixture_pair(1)
    source.compare(actual, expected, "committed")
    committed = source.snapshot
    path = tmp_path / "portable.json"
    restored = NumericalAudit.from_snapshot(committed, path)
    assert not path.exists()
    restored.require_committed_prefix(committed)
    actual, expected = fixture_pair(2)
    restored.compare(actual, expected, "portable replay")
    assert path.exists()
    assert restored.summary["frozen_changed_coordinates"] == 1
    assert restored.summary["frozen_unique_deviations"] == 2
    assert len(committed["events"]) == 1
    with pytest.raises(ValueError, match="fresh numerical audit path"):
        NumericalAudit.from_snapshot(committed, path)


def test_snapshot_restore_and_prefix_authenticate_snapshot():
    audit = exploratory()
    audit.compare(1, 1, "committed", kind="copy")
    corrupted = audit.snapshot
    corrupted["events"][0]["context"] = "changed"
    with pytest.raises(ValueError, match="checksum differs"):
        NumericalAudit.from_snapshot(corrupted)
    with pytest.raises(ValueError, match="checksum differs"):
        audit.require_committed_prefix(corrupted)
    with pytest.raises(ValueError, match="policy changed"):
        NumericalAudit().require_committed_prefix(audit.snapshot)

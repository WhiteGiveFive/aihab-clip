"""The input audit must never report success after a failed prerequisite."""

import pytest

from tools.audit_attention_fusion_inputs import AuditFailure, reject_failed_checks


@pytest.mark.parametrize(
    "checks",
    [
        {},
        {"checkpoint_hash": False, "no_test_overlap": True},
        {"checkpoint_hash": True, "no_test_overlap": False},
        {"checkpoint_hash": True, "no_test_overlap": None},
    ],
)
def test_reject_failed_or_missing_prerequisite(checks):
    with pytest.raises(AuditFailure):
        reject_failed_checks(checks)


def test_accept_verified_prerequisites():
    reject_failed_checks({"checkpoint_hash": True, "no_test_overlap": True})

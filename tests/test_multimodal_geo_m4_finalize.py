"""Synthetic regression for the additive frozen-config receipt-path correction."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import test_multimodal_geo_m4_analysis as base
from multimodal import geo_helpfulness_router_m4_analysis as v1
from multimodal import geo_helpfulness_router_m4_common as common
from multimodal import geo_helpfulness_router_m4_finalize as finalizer


@pytest.fixture
def actual_config():
    # This must read the actual immutable protocol configuration, not reproduce
    # the field location assumed by the implementation under test.
    path = common.PROJECT_ROOT / "configs/multimodal_geo_helpfulness.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def partial(tmp_path, monkeypatch, actual_config):
    inputs = base.inputs.__wrapped__(tmp_path, monkeypatch)
    inputs["config"]["artifact_contract"] = deepcopy(actual_config["artifact_contract"])
    inputs["config"]["schemas"] = deepcopy(actual_config["schemas"])
    with pytest.raises(KeyError, match="development_selection_receipt"):
        v1.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / common.RESULT_DIR
    assert len(base._fingerprints(root)) == 31
    assert not (root / "development_selection_receipt.json").exists()
    assert not (root / "manifest.json").exists()
    return inputs


def _forbid_selection(monkeypatch):
    for name in ("_make_selection", "select_common_c", "select_common_threshold"):
        monkeypatch.setattr(v1, name, base._forbidden)


def test_actual_protocol_places_receipt_contract_under_schemas(actual_config):
    assert "development_selection_receipt" not in actual_config["artifact_contract"]
    contract = actual_config["schemas"]["development_selection_receipt"]
    assert contract["exclusive_create"] is True
    assert contract["nullable_fields"] == []
    assert len(contract["required_fields"]) == 13
    assert {"router_frozen_state_hashes", "validation_metrics", "go_no_go"}.issubset(
        contract["required_fields"]
    )


def test_finalizer_uses_real_schema_and_preserves_all_preexisting_bytes(
    partial, monkeypatch
):
    root = partial["protocol_root"] / common.RESULT_DIR
    before = base._fingerprints(root)
    code_before = {
        name: common.file_hash(common.PROJECT_ROOT / name) for name in common.CODE_FILES
    }
    original_config = deepcopy(partial["config"])
    _forbid_selection(monkeypatch)
    built = finalizer.build_m4_result(partial["protocol_root"])
    assert built["status"] == "built"
    assert built["reselection_calls"] == 0
    assert built["assignment_label_columns_opened"] is True
    after = base._fingerprints(root)
    assert {name: after[name] for name in before} == before
    assert set(after) - set(before) == {
        "development_selection_receipt.json",
        "manifest.json",
    }
    assert partial["config"] == original_config
    assert {
        name: common.file_hash(common.PROJECT_ROOT / name) for name in common.CODE_FILES
    } == code_before
    receipt = common.read_json(root / "development_selection_receipt.json")
    required = partial["config"]["schemas"]["development_selection_receipt"][
        "required_fields"
    ]
    assert all(name in receipt and receipt[name] is not None for name in required)
    assert len(receipt["result_artifact_file_sha256"]) == 31
    assert (
        receipt["receipt_implementation_correction"]["scientific_settings_changed"]
        is False
    )
    declaration = common.read_json(partial["protocol_root"] / finalizer.DECLARATION)
    assert (
        declaration["correct_config_path"]
        == "schemas.development_selection_receipt.required_fields"
    )
    assert declaration["preexisting_result_file_sha256"] == {
        name: fingerprint[0] for name, fingerprint in before.items()
    }
    assert set(declaration["code_file_sha256"]) == set(finalizer.CODE_FILES)
    base._disable_post_receipt_inputs(monkeypatch, partial)
    assert (
        finalizer.validate_m4_result(partial["protocol_root"])["status"] == "validated"
    )
    assert (
        finalizer.build_m4_result(partial["protocol_root"])["status"] == "reused_valid"
    )
    assert base._fingerprints(root) == after


def test_finalizer_receipt_crash_recovery_never_reads_labels_or_reselects(
    partial, monkeypatch
):
    finish = v1._finish_manifest
    _forbid_selection(monkeypatch)
    monkeypatch.setattr(
        v1,
        "_finish_manifest",
        lambda *args: (_ for _ in ()).throw(RuntimeError("receipt written crash")),
    )
    with pytest.raises(RuntimeError, match="receipt written crash"):
        finalizer.build_m4_result(partial["protocol_root"])
    root = partial["protocol_root"] / common.RESULT_DIR
    before = base._fingerprints(root)
    assert "development_selection_receipt.json" in before
    assert "manifest.json" not in before
    monkeypatch.setattr(v1, "_finish_manifest", finish)
    base._disable_post_receipt_inputs(monkeypatch, partial)
    result = finalizer.build_m4_result(partial["protocol_root"])
    assert result["status"] == "reused_valid"
    after = base._fingerprints(root)
    assert {name: after[name] for name in before} == before


@pytest.mark.parametrize("tamper", ["declaration", "new_code", "old_result"])
def test_correction_tampering_blocks_before_assignment_access(
    partial, monkeypatch, tmp_path, tamper
):
    code_root = tmp_path / "code"
    for name in finalizer.CODE_FILES:
        common.write_bytes(code_root / name, (common.PROJECT_ROOT / name).read_bytes())
    monkeypatch.setattr(finalizer, "PROJECT_ROOT", code_root)
    finalizer._correction(partial["protocol_root"], partial, create=True)
    if tamper == "declaration":
        path = partial["protocol_root"] / finalizer.DECLARATION
        value = common.read_json(path)
        value["scientific_settings_changed"] = True
        payload = common.canonical_bytes(value)
    elif tamper == "new_code":
        path = code_root / finalizer.CODE_FILES[0]
        payload = path.read_bytes() + b"\n# changed\n"
    else:
        path = partial["protocol_root"] / common.RESULT_DIR / "policy_metrics.json"
        payload = path.read_bytes() + b" "
    path.chmod(0o644)
    path.write_bytes(payload)
    base._disable_post_receipt_inputs(monkeypatch, partial)
    with pytest.raises(ValueError, match="invalid|changed"):
        finalizer.build_m4_result(partial["protocol_root"])
    root = partial["protocol_root"] / common.RESULT_DIR
    assert not (root / "development_selection_receipt.json").exists()


@pytest.mark.parametrize("drift", ["new_code", "declaration"])
def test_midrun_correction_drift_cannot_rebind_initial_snapshot(
    partial, monkeypatch, tmp_path, drift
):
    code_root = tmp_path / "code"
    for name in finalizer.CODE_FILES:
        common.write_bytes(code_root / name, (common.PROJECT_ROOT / name).read_bytes())
    monkeypatch.setattr(finalizer, "PROJECT_ROOT", code_root)
    root = partial["protocol_root"] / common.RESULT_DIR
    before = base._fingerprints(root)
    evaluate = v1._evaluate

    def mutate_after_evaluation(*args):
        result = evaluate(*args)
        declaration_path = partial["protocol_root"] / finalizer.DECLARATION
        declaration = common.read_json(declaration_path)
        if drift == "new_code":
            name = finalizer.CODE_FILES[0]
            code_path = code_root / name
            code_path.chmod(0o644)
            code_path.write_bytes(code_path.read_bytes() + b"\n# concurrent change\n")
            declaration["code_file_sha256"][name] = common.file_hash(code_path)
        else:
            declaration["additional_diagnostic"] = "concurrent declaration change"
        # Keep the changed declaration self-consistent: checking only current
        # hashes would incorrectly adopt it instead of preserving the start seal.
        declaration["declaration_sha256"] = common.canonical_hash(
            {
                key: value
                for key, value in declaration.items()
                if key != "declaration_sha256"
            }
        )
        declaration_path.chmod(0o644)
        declaration_path.write_bytes(common.canonical_bytes(declaration) + b"\n")
        return result

    _forbid_selection(monkeypatch)
    monkeypatch.setattr(v1, "_evaluate", mutate_after_evaluation)
    with pytest.raises(ValueError, match="snapshot changed|parent or code changed"):
        finalizer.build_m4_result(partial["protocol_root"])
    assert base._fingerprints(root) == before
    assert not (root / "development_selection_receipt.json").exists()
    assert not (root / "manifest.json").exists()


def test_unknown_output_blocks_before_correction_or_label_read(partial, monkeypatch):
    root = partial["protocol_root"] / common.RESULT_DIR
    common.write_bytes(root / "unknown.json", b"{}")
    monkeypatch.setattr(v1, "_load_inputs", base._forbidden)
    with pytest.raises(ValueError, match="allowlist"):
        finalizer.build_m4_result(partial["protocol_root"])
    assert not (partial["protocol_root"] / finalizer.DECLARATION).exists()


def test_finalizer_fresh_process_has_no_fitting_imports():
    script = """import sys
from multimodal import geo_helpfulness_router_m4_finalize as module
assert module.v1._fit_incapable()['scoring_only_process']
assert all(name.split('.')[0] not in {'torch', 'scipy', 'sklearn'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=common.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

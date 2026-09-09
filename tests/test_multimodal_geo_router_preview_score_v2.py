"""Synthetic regression tests for the additive preview identity correction."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_router_preview_score as v1
from multimodal import geo_helpfulness_router_preview_score_v2 as preview
from test_multimodal_geo_router_preview_score import bundle, _reseal


@pytest.fixture
def corrected(bundle, monkeypatch):
    root, assignment, manifest = bundle
    frame = pq.read_table(assignment).to_pandas()
    for column in ("protocol_id", "row_uid", "plot_idx", "development_role"):
        frame[column] = frame[column].astype("string")
    frame.to_parquet(assignment, index=False)
    restored = pq.read_table(assignment).to_pandas()
    assert str(restored.row_uid.dtype) == "string"
    assert str(restored.plot_idx.dtype) == "string"
    manifest["assignment_file_sha256"] = v1._file_hash(assignment)
    _reseal(root, manifest)
    code_root = v1.PROJECT_ROOT
    for name in preview.CODE_FILES:
        path = code_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((preview.PROJECT_ROOT / name).read_bytes())
    monkeypatch.setattr(preview, "PROJECT_ROOT", code_root)
    declaration = {
        "schema_version": preview.DECLARATION_SCHEMA_VERSION,
        "protocol_id": "protocol_v1",
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "candidate_variant": "primary_full727", "training_seed": 1,
        "C_hex": .01.hex(), "metric": "four_state_accuracy_only",
        "selection_performed": False,
        "correction_reason": preview.CORRECTION_REASON,
        "code_file_sha256": {name: v1._file_hash(code_root / name) for name in preview.CODE_FILES},
        "previous_attempt": {"computed_accuracy": False},
    }
    declaration["declaration_sha256"] = v1._hash(declaration)
    path = root.parent / "preview_correction_declaration_v2.json"
    v1._save_result(path, declaration)
    return root, assignment, manifest, path, declaration


def _score(inputs, **kwargs):
    root, assignment, _, declaration_path, _ = inputs
    return preview.score_fixed_preview_v2(root, assignment, correction_declaration_path=declaration_path, **kwargs)


def _refresh_declaration(root, manifest, path, declaration):
    declaration["prediction_manifest_sha256"] = manifest["manifest_sha256"]
    declaration["declaration_sha256"] = v1._hash({key: value for key, value in declaration.items() if key != "declaration_sha256"})
    path.chmod(0o644)
    path.write_bytes(v1._canonical_bytes(declaration) + b"\n")
    path.chmod(0o444)


def test_extension_string_storage_regression_and_immutable_corrected_preview(corrected):
    root, assignment, manifest, declaration_path, declaration = corrected
    before = {path.name: v1._file_hash(path) for path in root.iterdir()}
    # Reproduce the original issue exactly: all identity values match, but pandas
    # compares StringDtype and object storage as different frame dtypes.
    _, predictions, _, _ = v1._load_sealed_predictions(root, expected_manifest_sha256=None)
    assignments = pq.read_table(assignment).to_pandas()
    selected = assignments.loc[assignments.development_role.eq("validation")].sort_values("row_uid").reset_index(drop=True)
    assert not predictions[["row_uid", "plot_idx"]].equals(selected[["row_uid", "plot_idx"]])
    assert preview._identities_equal(predictions, selected)
    with pytest.raises(v1.RouterPreviewError, match="do not align"):
        v1.score_fixed_preview(root, assignment)
    output = root.parent / "preview_v2.json"
    result = _score(corrected, output_path=output, expected_manifest_sha256=manifest["manifest_sha256"])
    assert result["schema_version"] == preview.RESULT_SCHEMA_VERSION
    assert result["correct_count"] == 411 and result["total_count"] == 822
    assert result["accuracy"] == .5
    assert result["candidate_variant"] == "primary_full727"
    assert result["training_seed"] == 1 and result["C_hex"] == .01.hex()
    assert result["prediction_manifest_sha256"] == manifest["manifest_sha256"]
    assert result["correction_declaration_sha256"] == declaration["declaration_sha256"]
    assert result["correction_declaration_file_sha256"] == v1._file_hash(declaration_path)
    assert result["code_file_sha256"] == declaration["code_file_sha256"]
    assert result["other_candidate_metrics_computed"] is False
    assert result["selection_performed"] is False
    assert result["preview_sha256"] == v1._hash({key: value for key, value in result.items() if key != "preview_sha256"})
    before_output = output.read_bytes(), output.stat().st_mtime_ns
    assert _score(corrected, output_path=output) == result
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before_output
    assert output.stat().st_mode & 0o222 == 0
    assert {path.name: v1._file_hash(path) for path in root.iterdir()} == before


@pytest.mark.parametrize("dtype", [object, "string", "string[pyarrow]"])
def test_identity_comparison_ignores_only_storage_dtype(dtype):
    left = pd.DataFrame({"row_uid": ["a", "b"], "plot_idx": ["p1", "p2"]})
    right = left.astype(dtype)
    assert preview._identities_equal(left, right)
    right.loc[0, "plot_idx"] = " p1"
    assert not preview._identities_equal(left, right)


@pytest.mark.parametrize("value", [None, "", "   ", 1, True, np.nan])
def test_identity_comparison_rejects_null_empty_and_nonstrings(value):
    left = pd.DataFrame({"row_uid": ["uid"], "plot_idx": ["plot"]})
    right = pd.DataFrame({"row_uid": ["uid"], "plot_idx": [value]})
    with pytest.raises(preview.RouterPreviewError, match="identity column"):
        preview._identities_equal(left, right)


@pytest.mark.parametrize("column,value", [("row_uid", "f" * 64), ("plot_idx", "wrong-plot"), ("plot_idx", "")])
def test_different_actual_identity_values_still_fail(corrected, column, value):
    root, assignment, manifest, path, declaration = corrected
    frame = pq.read_table(assignment).to_pandas()
    frame.loc[0, column] = value
    frame.to_parquet(assignment, index=False)
    manifest["assignment_file_sha256"] = v1._file_hash(assignment)
    _reseal(root, manifest)
    _refresh_declaration(root, manifest, path, declaration)
    with pytest.raises(preview.RouterPreviewError):
        _score(corrected)


@pytest.mark.parametrize("damage", ["missing", "mutable", "self_hash", "code", "code_allowlist", "candidate", "manifest", "reason"])
def test_invalid_correction_rejected_before_assignment_access(corrected, monkeypatch, damage):
    root, assignment, manifest, path, declaration = corrected
    if damage == "missing":
        path.unlink()
    elif damage == "mutable":
        path.chmod(0o644)
    elif damage == "code":
        (preview.PROJECT_ROOT / preview.CODE_FILES[0]).write_text("new unsealed scorer code")
    else:
        if damage == "self_hash":
            declaration["declaration_sha256"] = "b" * 64
        elif damage == "code_allowlist":
            declaration["code_file_sha256"]["extra.py"] = "b" * 64
        elif damage == "candidate":
            declaration["C_hex"] = 1.0.hex()
        elif damage == "manifest":
            declaration["prediction_manifest_sha256"] = "b" * 64
        elif damage == "reason":
            declaration["correction_reason"] = "change_the_metric"
        if damage != "self_hash":
            declaration["declaration_sha256"] = v1._hash({key: value for key, value in declaration.items() if key != "declaration_sha256"})
        path.chmod(0o644)
        path.write_bytes(v1._canonical_bytes(declaration))
        path.chmod(0o444)
    original = v1._file_hash

    def hashing(source):
        assert Path(source) != assignment, "labels accessed before correction validation"
        return original(source)

    monkeypatch.setattr(v1, "_file_hash", hashing)
    with pytest.raises(preview.RouterPreviewError):
        _score(corrected)


@pytest.mark.parametrize("changed", ["declaration", "code"])
def test_correction_snapshot_is_checked_after_label_read(corrected, monkeypatch, changed):
    _, assignment, _, declaration_path, _ = corrected
    original = pq.read_table

    def read(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if Path(path) == assignment:
            target = declaration_path if changed == "declaration" else preview.PROJECT_ROOT / preview.CODE_FILES[0]
            target.chmod(0o644)
            target.write_text("changed during scoring")
        return value

    monkeypatch.setattr(preview.pq, "read_table", read)
    with pytest.raises(preview.RouterPreviewError, match="changed during scoring"):
        _score(corrected)


def test_output_cannot_overwrite_correction_or_other_inputs(corrected):
    root, assignment, _, declaration, _ = corrected
    for path in (assignment, declaration, root / "preview_v2.json"):
        with pytest.raises(preview.RouterPreviewError, match="output"):
            _score(corrected, output_path=path)


def test_new_scorer_imports_no_fitting_modules_in_fresh_process():
    repository = Path(__file__).resolve().parents[1]
    code = (
        "import sys; import multimodal.geo_helpfulness_router_preview_score_v2; "
        "blocked=('sklearn','scipy','torch','multimodal.geo_helpfulness_router_candidates',"
        "'multimodal.geo_helpfulness_router_candidate_fit','multimodal.geo_helpfulness_router_dataset',"
        "'multimodal.geo_helpfulness_oof','multimodal.geo_helpfulness_targets_features'); "
        "assert not any(name==p or name.startswith(p+'.') for p in blocked for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=repository, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

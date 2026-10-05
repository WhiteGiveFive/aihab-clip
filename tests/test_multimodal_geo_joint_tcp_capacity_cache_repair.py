"""Additive NPZ repair preserves reserved-key arrays and the original attempt."""

from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("capacity_npz_repair",
    Path(__file__).resolve().parents[1] / "tools/run_multimodal_geo_helpfulness_joint_tcp_capacity_cache_repair.py")
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def test_original_writer_collision_is_reproduced(tmp_path):
    with pytest.raises(TypeError, match="file"):
        repair._save_arrays(tmp_path / "original.npz", {"file": np.array(["a.jpg"])})
    assert not (tmp_path / "original.npz").exists()


def test_repair_retains_file_array_and_all_exact_dtypes_values(tmp_path):
    arrays = {
        "file": np.array(["one/leaf.jpg", "é/树.jpg"]),
        "row_uid": np.array(["1" * 64, "2" * 64]),
        "producer_fold": np.array([-1, -1], dtype=np.int64),
        "X_numeric25": np.arange(50, dtype=np.float32).reshape(2, 25),
        "raw_logits": np.linspace(-9, 12, 36, dtype=np.float64).reshape(2, 18),
    }
    expected = {name: value.copy() for name, value in arrays.items()}
    path = tmp_path / "repaired.npz"
    repair.save_features(path, arrays)
    actual = repair._load_arrays(path)
    assert set(actual) == set(arrays)
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype
        np.testing.assert_array_equal(actual[name], value)
        np.testing.assert_array_equal(arrays[name], value)
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {name + ".npy" for name in arrays}
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist())


@pytest.mark.parametrize("paths", [np.array([["a"]]), np.array([1]), np.array(["a"], dtype=object)])
def test_noncanonical_file_array_rejected_before_archive_creation(tmp_path, paths):
    path = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="one-dimensional string"):
        repair.save_features(path, {"file": paths, "values": np.zeros((1, 2))})
    assert not path.exists()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def amendment(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    output = root / "output"
    helper = root / "tools/repair.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# test helper\n")
    old = {"study_id": repair.application.STUDY_ID, "config": {"scientific": "unchanged"},
           "source_file_sha256": {"old.py": "a" * 64}, "validation_selections": {"threshold": .1}}
    _write(output / "declaration.json", old)
    staged = output / "staging/failed_seed/rows.parquet"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"original label-free predictions")
    original_record = repair.study.snapshot(output)
    monkeypatch.setattr(repair, "ROOT", root)
    monkeypatch.setattr(repair, "TOOL", helper)
    monkeypatch.setattr(repair.application, "OUTPUT", output)
    monkeypatch.setattr(repair, "_lock", lambda *args: nullcontext())
    monkeypatch.setattr(repair.application, "validate_declaration", lambda: repair._json(output / "declaration.json"))
    return output, old, original_record


def test_amendment_preserves_original_and_changes_only_serializer_binding(amendment):
    output, old, before = amendment
    result = repair.declare()
    assert result["status"] == "declared"
    current = repair.validate_amendment()
    record = current["serialization_amendment"]
    archived = output / record["failed_declaration_path"]
    assert repair._json(archived) == old
    assert repair.study.snapshot(archived.parent) == before
    for name, value in old.items():
        if name != "source_file_sha256":
            assert current[name] == value
    assert current["source_file_sha256"] == {**old["source_file_sha256"], "tools/repair.py": repair._hash(repair.TOOL)}
    assert record["models_changed"] is record["thresholds_changed"] is record["native_inference_changed"] is False
    complete = repair.study.snapshot(output)
    assert repair.declare() == {"status": "reused_valid"}
    assert repair.study.snapshot(output) == complete


@pytest.mark.parametrize("folder", ["predictions", "scores", "label_acquisition"])
def test_amendment_blocked_after_prediction_or_scoring_directory(amendment, folder):
    output, old, _ = amendment
    (output / folder).mkdir()
    with pytest.raises(ValueError, match="Cannot amend"):
        repair.declare()
    assert repair._json(output / "declaration.json") == old
    assert not (output / "failed_attempts").exists()


@pytest.mark.parametrize("mutation", ["config", "threshold", "old_source", "helper", "archived", "archive_extra"])
def test_amendment_rejects_scientific_or_provenance_drift(amendment, mutation):
    output, _, _ = amendment
    repair.declare()
    current = repair._json(output / "declaration.json")
    archived = output / current["serialization_amendment"]["failed_declaration_path"]
    if mutation == "config":
        current["config"]["scientific"] = "changed"
    elif mutation == "threshold":
        current["validation_selections"]["threshold"] = 0.
    elif mutation == "old_source":
        current["source_file_sha256"]["old.py"] = "b" * 64
    elif mutation == "helper":
        repair.TOOL.write_text("# changed helper\n")
    elif mutation == "archived":
        archived.chmod(0o644)
        archived.write_bytes(archived.read_bytes() + b" ")
    else:
        (archived.parent / "unexpected").write_text("changed archive")
    (output / "declaration.json").chmod(0o644)
    _write(output / "declaration.json", current)
    with pytest.raises(ValueError, match="(amendment|Preserved)"):
        repair.validate_amendment()


def test_complete_build_uses_repaired_predictions_before_scoring(monkeypatch):
    events = []
    monkeypatch.setattr(repair.sys, "argv", [str(repair.TOOL), "build"])
    monkeypatch.setattr(repair.subprocess, "run", lambda command, **kwargs: events.append(command[-1]))
    monkeypatch.setattr(repair, "declare", lambda: events.append("amend"))
    monkeypatch.setattr(repair, "predict_seed", lambda seed: events.append(("predict", seed)))
    monkeypatch.setattr(repair, "validate_amendment", lambda: events.append("validate_amendment"))
    repair.main()
    assert events == ["prepare", "fit", "score", "validate", "test-declare", "amend",
                      *[("predict", seed) for seed in range(1, 5)],
                      "test-score", "test-validate", "validate_amendment"]

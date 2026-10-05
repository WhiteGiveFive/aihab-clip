"""Additive recovery fixes order only and never mutates the sealed source."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_router_mlp_replay as replay


def _history_record():
    return {
        "state": {
            "config_id": "fixture",
            "variant": "ablation_numeric25",
            "training_seed": 1,
        },
        "diagnostics": {
            "history": [
                {
                    "epoch": 300,
                    "optimization_cross_entropy": 1.0,
                    "train_nll": 0.8,
                    "train_accuracy": 0.75,
                    "maximum_gradient_l2_norm": 2.0,
                    "gradients_finite": True,
                }
            ]
        },
    }


def test_actual_memory_to_canonical_json_history_order_roundtrip():
    in_memory = _history_record()
    reloaded = json.loads(json.dumps(in_memory, sort_keys=True))
    original = replay._ORIGINAL_CURVES([in_memory])
    original_reloaded = replay._ORIGINAL_CURVES([reloaded])
    assert original.schema != original_reloaded.schema
    assert original.column_names == list(replay.CURVE_COLUMNS)
    fixed = replay._fixed_curves([reloaded])
    assert fixed.schema == original.schema
    replay.compare_tables(original, fixed)
    assert original.equals(fixed)


def test_json_resumed_original_physical_order_is_supported(tmp_path):
    records = [json.loads(json.dumps(_history_record(), sort_keys=True))]
    original = replay._ORIGINAL_CURVES(records)
    assert original.column_names == list(replay.JSON_CURVE_COLUMNS)
    replay.write_bytes(
        tmp_path / "predictions/training_curves.parquet", replay.table_bytes(original)
    )
    order = replay._source_curve_order(tmp_path)
    assert order == replay.JSON_CURVE_COLUMNS
    with replay._curve_order_override(order):
        expected = replay.source._curves(records)
        replay.compare_tables(original, expected)
        assert original.equals(expected)
    assert replay.source._curves is replay._ORIGINAL_CURVES


def test_unknown_physical_curve_order_remains_rejected(tmp_path):
    original = replay._ORIGINAL_CURVES([_history_record()])
    unknown = original.select(list(reversed(original.column_names)))
    replay.write_bytes(
        tmp_path / "predictions/training_curves.parquet", replay.table_bytes(unknown)
    )
    with pytest.raises(ValueError, match="known original layout"):
        replay._source_curve_order(tmp_path)
    with pytest.raises(ValueError, match="two known original curve orders"):
        with replay._curve_order_override(unknown.column_names):
            pass


def test_curve_fix_does_not_hide_value_or_extra_column_changes():
    record = _history_record()
    changed = deepcopy(record)
    changed["diagnostics"]["history"][0]["train_nll"] += 0.1
    with pytest.raises(ValueError, match="values mismatch"):
        replay.compare_tables(
            replay._fixed_curves([record]), replay._fixed_curves([changed])
        )
    changed["diagnostics"]["history"][0]["new_metric"] = 1
    with pytest.raises(ValueError, match="nine curve columns"):
        replay._fixed_curves([changed])


def test_scoped_override_restores_original_even_after_failure():
    original = replay.source._curves
    with pytest.raises(RuntimeError, match="stop"):
        with replay._curve_order_override():
            assert replay.source._curves is replay._fixed_curves
            raise RuntimeError("stop")
    assert replay.source._curves is original


def test_scoped_override_rejects_unknown_existing_override(monkeypatch):
    monkeypatch.setattr(replay.source, "_curves", lambda records: None)
    with pytest.raises(ValueError, match="already overridden"):
        with replay._curve_order_override():
            pass


def _snapshot(root):
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Publication units use synthetic inputs. Process capabilities are checked
    # separately in a fresh subprocess, not inherited from other test modules.
    monkeypatch.setattr(replay.source, "_fit_incapable", lambda: None)
    source_root, output = tmp_path / "source", tmp_path / "recovery"
    protocol, capacity = tmp_path / "protocol", tmp_path / "capacity"
    protocol.mkdir()
    capacity.mkdir()
    parent = protocol / "frozen.txt"
    parent.write_text("frozen parent")
    config_id = "mlp_fixture"
    model_path = source_root / f"predictions/models/{config_id}__seed_1.json"
    replay.write_json(model_path, {"state": "unchanged synthetic state"})
    original = replay._seal(
        {
            "protocol_root": str(protocol),
            "capacity_root": str(capacity),
            "parent_file_sha256": {str(parent): replay.file_hash(parent)},
        },
        "declaration_sha256",
    )
    replay.write_json(source_root / "declaration.json", original)
    prediction = {"manifest_sha256": "a" * 64}
    replay.write_json(source_root / "predictions/manifest.json", prediction)
    replay.write_bytes(
        source_root / "predictions/training_curves.parquet",
        replay.table_bytes(replay._ORIGINAL_CURVES([_history_record()])),
    )
    source_before = _snapshot(source_root)
    for name in ("repair.py", "runner.py"):
        (tmp_path / name).write_text("# immutable recovery implementation\n")
    monkeypatch.setattr(replay, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(replay, "CODE_FILES", ("repair.py", "runner.py"))

    # Keep the synthetic publication fixture small; production source counts
    # remain enforced by the unpatched guard tested separately below.
    def source_declaration(root):
        assert root == source_root
        value = replay.read_json(root / "declaration.json")
        replay.check_self_hash(value, "declaration_sha256")
        replay.check_snapshot(value["parent_file_sha256"])
        return value

    monkeypatch.setattr(replay.source, "load_declaration", source_declaration)

    def source_predictions(root):
        value = source_declaration(root)
        assert _snapshot(root) == source_before, "source modified"
        return value, prediction

    monkeypatch.setattr(replay, "_source_predictions", source_predictions)
    calls = []

    def payloads(root, declaration, manifest):
        assert (
            output / "declaration.json"
        ).is_file(), "target decode before recovery declaration"
        recovery = replay.read_json(output / "declaration.json")
        replay.check_self_hash(recovery, "declaration_sha256")
        replay.check_snapshot(recovery["parent_file_sha256"])
        assert (
            declaration == original and manifest == prediction and root == source_root
        )
        calls.append("target-decoded-only-after-declaration")
        values = {
            name: (
                pa.table({"value": [1]})
                if name.endswith(".parquet")
                else {"study_id": replay.STUDY_ID}
            )
            for name in replay.SCORE_FILES
        }
        values["frozen_policies.json"] = {
            "study_id": replay.STUDY_ID,
            "groups": {
                "mlp_fixture": {
                    "config_id": config_id,
                    "models": {
                        "1": {
                            "path": f"../predictions/models/{config_id}__seed_1.json",
                            "file_sha256": replay.file_hash(model_path),
                        }
                    },
                }
            },
        }
        values["selection_receipt.json"] = replay._seal(
            {
                "study_id": replay.STUDY_ID,
                "declaration_sha256": original["declaration_sha256"],
                "prediction_manifest_sha256": prediction["manifest_sha256"],
                "selection_sha256": replay.canonical_hash(values["selection.json"]),
                "frozen_policies_sha256": replay.canonical_hash(
                    values["frozen_policies.json"]
                ),
            },
            "receipt_sha256",
        )
        return values

    monkeypatch.setattr(replay.source, "_score_payloads", payloads)
    return source_root, output, parent, calls


def test_additive_build_and_reuse_are_immutable_and_lineage_bound(fixture):
    source_root, output, _, calls = fixture
    before = _snapshot(source_root)
    result = replay.build_scores(source_root, output)
    assert result["status"] == "built"
    assert not result["fitting_performed"] and not result["source_files_changed"]
    declaration = replay.load_declaration(source_root, output)
    assert set(before) <= set(declaration["parent_file_sha256"])
    assert len(_snapshot(output)) == 18
    receipt = replay.read_json(output / "scores/selection_receipt.json")
    replay.check_self_hash(receipt, "receipt_sha256")
    frozen = replay.read_json(output / "scores/frozen_policies.json")
    assert receipt["frozen_policies_sha256"] == replay.canonical_hash(frozen)
    assert (
        receipt["recovery_lineage"]["recovery_declaration_sha256"]
        == declaration["declaration_sha256"]
    )
    model = frozen["groups"]["mlp_fixture"]["models"]["1"]
    assert Path(model["path"]).is_absolute()
    assert Path(model["path"]).is_relative_to(source_root)
    assert replay.file_hash(Path(model["path"])) == model["file_sha256"]
    sealed = _snapshot(output)
    assert replay.build_scores(source_root, output)["status"] == "reused_valid"
    assert replay.validate_scores(source_root, output)["status"] == "validated"
    assert _snapshot(output) == sealed and _snapshot(source_root) == before
    assert len(calls) == 3


@pytest.mark.parametrize(
    "target", ["declaration", "manifest", "score", "parent", "code"]
)
def test_tampering_fails_before_any_new_target_decode(fixture, target):
    source_root, output, parent, calls = fixture
    replay.build_scores(source_root, output)
    paths = {
        "declaration": output / "declaration.json",
        "manifest": output / "scores/manifest.json",
        "score": output / "scores/summary.json",
        "parent": parent,
        "code": output.parent / "repair.py",
    }
    path = paths[target]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    # Whitespace changes are legal JSON, but declarations/manifests are
    # self-hashed semantically; alter a bound field for those two cases.
    if target in ("declaration", "manifest"):
        value = json.loads(path.read_text())
        value["study_id"] = "tampered"
        path.write_text(json.dumps(value))
    before = len(calls)
    with pytest.raises(ValueError):
        replay.validate_scores(source_root, output)
    assert len(calls) == before


def test_unknown_output_preserved_without_target_decode(fixture):
    source_root, output, _, calls = fixture
    output.mkdir()
    unknown = output / "user.txt"
    unknown.write_text("keep")
    with pytest.raises(ValueError, match="unowned"):
        replay.build_scores(source_root, output)
    assert unknown.read_text() == "keep" and not calls


def test_unknown_committed_file_preserved_without_target_decode(fixture):
    source_root, output, _, calls = fixture
    replay.build_scores(source_root, output)
    unknown = output / "scores/user.txt"
    unknown.write_text("keep")
    count = len(calls)
    with pytest.raises(ValueError, match="allowlist"):
        replay.validate_scores(source_root, output)
    assert unknown.read_text() == "keep" and len(calls) == count


def test_partial_recovery_reuses_existing_payload_bytes(fixture, monkeypatch):
    source_root, output, _, _ = fixture
    original_publish = replay._publish

    def stop_before_manifest(*args):
        raise RuntimeError("interrupted before manifest")

    monkeypatch.setattr(replay, "_publish", stop_before_manifest)
    with pytest.raises(RuntimeError, match="interrupted"):
        replay.build_scores(source_root, output)
    before = _snapshot(output)
    assert not (output / "scores/manifest.json").exists()
    monkeypatch.setattr(replay, "_publish", original_publish)
    assert replay.build_scores(source_root, output)["status"] == "built"
    after = _snapshot(output)
    assert all(after[path] == value for path, value in before.items())


def test_lock_contention_and_containment_precede_target_decode(fixture):
    source_root, output, _, calls = fixture
    with replay.exclusive_lock(output):
        with pytest.raises(ValueError, match="owns this workflow lock"):
            replay.build_scores(source_root, output)
    with pytest.raises(ValueError, match="outside all frozen"):
        replay.build_scores(source_root, source_root / "scores")
    assert not calls and not (source_root / "scores").exists()


def test_source_guard_rejects_existing_scores_without_replaying(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.source, "_fit_incapable", lambda: None)
    root = tmp_path / "source"
    replay.write_json(root / "declaration.json", {})
    replay.write_json(root / "predictions/manifest.json", {})
    replay.write_json(root / "scores/summary.json", {})
    monkeypatch.setattr(
        replay.source,
        "load_declaration",
        lambda root: {"configuration": {}, "parent_file_sha256": {}},
    )
    monkeypatch.setattr(replay.source, "_prediction_files", lambda config: set())

    def forbidden(*args):
        raise AssertionError("source with scores was replayed")

    monkeypatch.setattr(replay.source, "_validate_predictions", forbidden)
    with pytest.raises(ValueError, match="allowlist"):
        replay._source_predictions(root)


def test_fresh_import_is_fitting_incapable():
    code = """
import sys
from multimodal import geo_helpfulness_router_mlp_replay as replay
replay.source._fit_incapable()
assert not any(name.split('.')[0] in {'torch','sklearn','scipy','xgboost'} for name in sys.modules)
sys.modules['multimodal.geo_helpfulness_router_mlp_fit'] = object()
try:
    replay.validate_predictions('unused')
except ValueError:
    pass
else:
    raise AssertionError('fit-capable replay accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr

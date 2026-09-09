"""Synthetic validation preparation checks; never opens real validation data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_router_validation as validation


@pytest.fixture
def derivation(monkeypatch):
    modes = ("image_only", "geo_only", "raw_concat")
    rows = []
    for seed in range(1, 5):
        for index in range(822):
            row = {
                "row_uid": f"{index:064x}",
                "plot_idx": f"plot_{index % 325}",
                "training_seed": seed,
            }
            for mode_index, mode in enumerate(modes):
                logits = np.zeros(18, dtype=np.float64)
                logits[:3] = [index, seed, mode_index]
                prefix = ("image", "geo", "raw")[mode_index]
                row[f"{prefix}_logits"], row[f"{prefix}_pred"] = logits, mode_index
            rows.append(row)
    outputs = pd.DataFrame(rows).sample(frac=1, random_state=87)
    inputs = {
        "dataset_manifest": {"protocol_id": "protocol_v1"},
        "temperatures": {
            (seed, mode): (seed, mode) for seed in range(1, 5) for mode in modes
        },
        "transforms": {seed: seed for seed in range(1, 5)},
        "candidate_states": {
            validation.candidate_filename(variant, seed, C): (variant, seed, C)
            for variant in validation.VARIANTS
            for seed in range(1, 5)
            for C in validation.C_VALUES
        },
    }
    calls = {"temperatures": [], "semantic": [], "transforms": [], "candidates": []}

    def calibrate(logits, state):
        seed, mode = state
        calls["temperatures"].append(state)
        np.testing.assert_array_equal(logits[:, 0], np.arange(822))
        np.testing.assert_array_equal(logits[:, 1], np.full(822, seed))
        assert logits.shape == (822, 18) and logits.dtype == np.float64
        probabilities = np.zeros_like(logits)
        probabilities[:, modes.index(mode)] = 1.0
        return probabilities

    def semantic(image, geo, raw, *, probability_basis):
        calls["semantic"].append(probability_basis)
        for index, values in enumerate((image, geo, raw)):
            np.testing.assert_array_equal(values.argmax(axis=1), np.full(822, index))
        return pd.DataFrame(
            {
                "image_pred": np.zeros(822, dtype=np.int8),
                "geo_pred": np.ones(822, dtype=np.int8),
                "raw_pred": np.full(822, 2, dtype=np.int8),
            }
        )

    def transform(frame, state):
        calls["transforms"].append(state)
        assert list(frame.columns) == ["image_pred", "geo_pred", "raw_pred"]
        matrix = np.zeros((822, 727), dtype=np.float64)
        matrix[:, 0], matrix[:, 1] = np.arange(822), state
        return matrix

    def predict(matrix, state):
        variant, seed, C = state
        calls["candidates"].append(state)
        np.testing.assert_array_equal(matrix[:, 0], np.arange(822))
        np.testing.assert_array_equal(matrix[:, 1], np.full(822, seed))
        q0 = (
            0.2
            + matrix[:, 0] * 1e-5
            + seed * 0.001
            + C * 0.001
            + validation.VARIANTS.index(variant) * 0.01
        )
        return np.column_stack((q0, np.full(822, 0.1), np.full(822, 0.3), 0.6 - q0))

    monkeypatch.setattr(validation, "apply_expert_temperature", calibrate)
    monkeypatch.setattr(validation, "build_router_feature_frame", semantic)
    monkeypatch.setattr(validation, "transform_router_features", transform)
    monkeypatch.setattr(validation, "predict_router_probabilities", predict)
    return SimpleNamespace(inputs=inputs, outputs=outputs, calls=calls)


def test_derivation_applies_frozen_states_once_per_seed_and_all_candidates(derivation):
    tables = validation.derive_validation_tables(derivation.inputs, derivation.outputs)
    assert len(derivation.calls["temperatures"]) == 12
    assert len(set(derivation.calls["temperatures"])) == 12
    assert derivation.calls["transforms"] == [1, 2, 3, 4]
    assert derivation.calls["semantic"] == ["scalar_temperature_calibrated"] * 4
    assert (
        len(derivation.calls["candidates"])
        == len(set(derivation.calls["candidates"]))
        == 48
    )
    assert set(tables) == {f"{variant}.parquet" for variant in validation.VARIANTS} | {
        "expert_predictions.parquet"
    }
    for variant in validation.VARIANTS:
        table = tables[f"{variant}.parquet"]
        assert table.num_rows == 13152
        assert table.schema.equals(validation.prediction_schema(), check_metadata=True)
        frame = table.to_pandas()
        assert frame.equals(
            frame.sort_values(["C_hex", "training_seed", "row_uid"]).reset_index(
                drop=True
            )
        )
        assert not {
            "target_state",
            "label_id_dense",
            "correctness",
            "true_class",
        }.intersection(frame.columns)
        expected = (
            0.2
            + frame.row_uid.map(lambda uid: int(uid, 16)).to_numpy() * 1e-5
            + frame.training_seed.to_numpy() * 0.001
            + frame.C_hex.map(float.fromhex).to_numpy() * 0.001
            + validation.VARIANTS.index(variant) * 0.01
        )
        np.testing.assert_array_equal(frame.q_rescue, expected)
        np.testing.assert_allclose(
            frame.score, frame.q_rescue - frame.q_harm, atol=0, rtol=0
        )
    experts = tables["expert_predictions.parquet"]
    assert experts.num_rows == 3288
    assert experts.schema.equals(validation.expert_schema(), check_metadata=True)
    frame = experts.to_pandas()
    assert frame.equals(
        frame.sort_values(["row_uid", "training_seed"]).reset_index(drop=True)
    )


def test_derivation_rejects_calibrated_argmax_mismatch(derivation):
    derivation.outputs.loc[derivation.outputs.index[0], "geo_pred"] = 7
    with pytest.raises(ValueError, match="calibrated argmax"):
        validation.derive_validation_tables(derivation.inputs, derivation.outputs)


@pytest.mark.parametrize("invalid", ["width", "dtype", "nonfinite"])
def test_derivation_rejects_invalid_transform_matrix(derivation, monkeypatch, invalid):
    matrix = np.zeros(
        (822, 726 if invalid == "width" else 727),
        dtype=np.float32 if invalid == "dtype" else np.float64,
    )
    if invalid == "nonfinite":
        matrix[0, 0] = np.inf
    monkeypatch.setattr(validation, "transform_router_features", lambda *args: matrix)
    with pytest.raises(ValueError, match="validation feature matrix"):
        validation.derive_validation_tables(derivation.inputs, derivation.outputs)


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    repo, protocol = tmp_path / "repo", tmp_path / "protocol_v1"
    for name in validation.CODE_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic implementation\n")
    parent = tmp_path / "parent.json"
    parent.write_text("immutable training parent\n")
    root = protocol / validation.BUNDLE_RELATIVE_PATH
    calls = []
    heuristics = {"frozen_training_thresholds": True}
    outputs = object()

    def training(repo_root, protocol_root):
        assert repo_root == repo and protocol_root == protocol
        calls.append("training")
        return {
            "dataset_manifest": {
                "protocol_id": "protocol_v1",
                "manifest_sha256": "a" * 64,
            },
            "candidate_manifest": {"manifest_sha256": "b" * 64},
            "assignment_file_sha256": "c" * 64,
            "parent_file_hashes": {str(parent): validation._file_hash(parent)},
            "audit": "OOF_AUDIT_ONLY",
            "targets": "OOF_TARGETS_ONLY",
        }

    def select(audit, targets, *, protocol_id):
        assert (audit, targets, protocol_id) == (
            "OOF_AUDIT_ONLY",
            "OOF_TARGETS_ONLY",
            "protocol_v1",
        )
        calls.append("heuristic_selection")
        return heuristics

    def validate_heuristics(state, audit, targets, *, protocol_id):
        assert state == heuristics
        calls.append("heuristic_application")

    def validation_inputs(repo_root, protocol_root, inputs):
        calls.append("validation_outputs")
        for name in ("preview_declaration.json", "oof_heuristics.json"):
            assert (root / name).is_file()
            assert (root / name).stat().st_mode & 0o222 == 0
        declaration = json.loads((root / "preview_declaration.json").read_text())
        assert declaration["candidate_variant"] == "primary_full727"
        assert declaration["training_seed"] == 1
        assert declaration["C_hex"] == (0.01).hex()
        return outputs

    def derive(inputs, source):
        assert source is outputs
        calls.append("derive")
        return {
            name: pa.table({"value": np.array([0.2, 0.4], dtype=np.float64)})
            for name in validation.ARTIFACT_FILENAMES
            if name.endswith(".parquet")
        }

    def sealed(path, **kwargs):
        assert path == root
        assert {p.name for p in root.iterdir()} == set(
            validation.ARTIFACT_FILENAMES
        ) | {"manifest.json"}
        manifest = json.loads((root / "manifest.json").read_text())
        for name, record in manifest["artifacts"].items():
            assert validation._file_hash(root / name) == record["file_sha256"]
        assert manifest["manifest_sha256"] == validation.canonical_sha256(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        )
        if kwargs:
            assert kwargs["expected_manifest_sha256"] == manifest["manifest_sha256"]
        calls.append("sealed")
        return {"valid": True, "manifest_sha256": manifest["manifest_sha256"]}

    monkeypatch.setattr(validation, "load_training_inputs", training)
    monkeypatch.setattr(validation, "select_oof_heuristic_thresholds", select)
    monkeypatch.setattr(
        validation, "validate_frozen_oof_heuristics", validate_heuristics
    )
    monkeypatch.setattr(validation, "load_validation_inputs", validation_inputs)
    monkeypatch.setattr(validation, "derive_validation_tables", derive)
    monkeypatch.setattr(validation, "validate_sealed_predictions", sealed)
    return SimpleNamespace(
        root=root,
        parent=parent,
        calls=calls,
        kwargs={"repo_root": repo, "protocol_root": protocol},
    )


def bundle_hashes(root):
    return {path.name: validation._file_hash(path) for path in root.iterdir()}


def test_prepare_publication_order_and_reuse_without_threshold_search(
    workflow, monkeypatch
):
    publications = []
    publish = validation._publish_bytes

    def record(path, *args, **kwargs):
        publications.append(path.name)
        return publish(path, *args, **kwargs)

    monkeypatch.setattr(validation, "_publish_bytes", record)
    result = validation.prepare_validation_bundle(**workflow.kwargs)
    assert result["status"] == "built" and result["fitting_calls"] == 0
    assert publications[:2] == ["preview_declaration.json", "oof_heuristics.json"]
    assert publications[-1] == "manifest.json"
    assert workflow.calls == [
        "training",
        "heuristic_selection",
        "validation_outputs",
        "derive",
        "sealed",
    ]
    assert all(path.stat().st_mode & 0o222 == 0 for path in workflow.root.iterdir())
    before = bundle_hashes(workflow.root)
    monkeypatch.setattr(
        validation,
        "select_oof_heuristic_thresholds",
        lambda *a, **k: pytest.fail("refitted threshold"),
    )
    for validate_only in (False, True):
        reused = validation.prepare_validation_bundle(
            **workflow.kwargs, validate_only=validate_only
        )
        assert reused["status"] == ("validated" if validate_only else "reused_valid")
        assert bundle_hashes(workflow.root) == before


@pytest.mark.parametrize(
    "interrupt_at", ["oof_heuristics.json", "primary_full727.parquet", "manifest.json"]
)
def test_interrupted_publication_resumes_without_overwrite(
    workflow, monkeypatch, interrupt_at
):
    publish = validation._publish_bytes

    def interrupted(path, *args, **kwargs):
        if path.name == interrupt_at:
            raise RuntimeError("synthetic interruption")
        return publish(path, *args, **kwargs)

    monkeypatch.setattr(validation, "_publish_bytes", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        validation.prepare_validation_bundle(**workflow.kwargs)
    assert not (workflow.root / "manifest.json").exists()
    before = bundle_hashes(workflow.root)
    monkeypatch.setattr(validation, "_publish_bytes", publish)
    if "oof_heuristics.json" in before:
        monkeypatch.setattr(
            validation,
            "select_oof_heuristic_thresholds",
            lambda *a, **k: pytest.fail("refitted threshold"),
        )
    validation.prepare_validation_bundle(**workflow.kwargs)
    assert all(
        validation._file_hash(workflow.root / name) == digest
        for name, digest in before.items()
    )


def test_orphan_heuristics_are_not_adopted(workflow):
    workflow.root.mkdir(parents=True)
    (workflow.root / "oof_heuristics.json").write_text(
        '{"frozen_training_thresholds":true}\n'
    )
    with pytest.raises(ValueError):
        validation.prepare_validation_bundle(**workflow.kwargs)
    assert "validation_outputs" not in workflow.calls
    assert not (workflow.root / "preview_declaration.json").exists()


def test_unknown_file_and_symlink_root_are_preserved(workflow):
    workflow.root.mkdir(parents=True)
    unknown = workflow.root / "user.txt"
    unknown.write_text("keep this")
    with pytest.raises(ValueError, match="unknown files"):
        validation.prepare_validation_bundle(**workflow.kwargs)
    assert unknown.read_text() == "keep this" and workflow.calls == []
    unknown.unlink()
    workflow.root.rmdir()
    workflow.root.symlink_to(workflow.parent.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validation.prepare_validation_bundle(**workflow.kwargs)
    assert workflow.root.is_symlink() and workflow.calls == []


def test_lock_contention_rejects_before_loading(workflow):
    with validation._exclusive_lock(workflow.root):
        with pytest.raises(ValueError, match="owns the lock"):
            validation.prepare_validation_bundle(**workflow.kwargs)
    assert workflow.calls == []


def test_validate_only_never_creates_missing_bundle(workflow):
    with pytest.raises(ValueError, match="complete committed"):
        validation.prepare_validation_bundle(**workflow.kwargs, validate_only=True)
    assert not workflow.root.exists() and workflow.calls == []


def test_parent_mutation_after_derivation_rejects_commit(workflow, monkeypatch):
    derive = validation.derive_validation_tables

    def changed(*args):
        result = derive(*args)
        workflow.parent.write_text("changed\n")
        return result

    monkeypatch.setattr(validation, "derive_validation_tables", changed)
    with pytest.raises(ValueError, match="parent changed"):
        validation.prepare_validation_bundle(**workflow.kwargs)
    assert not (workflow.root / "manifest.json").exists()


def test_atomic_byte_publication_rejects_overwrite_symlink_and_missing(tmp_path):
    path = tmp_path / "file.json"
    validation._publish_bytes(path, b"first", allow_create=True)
    validation._publish_bytes(path, b"first", allow_create=False)
    with pytest.raises(ValueError, match="differs"):
        validation._publish_bytes(path, b"second", allow_create=True)
    assert path.read_bytes() == b"first"
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="nonsymlink"):
        validation._publish_bytes(link, b"first", allow_create=True)
    with pytest.raises(ValueError, match="missing committed"):
        validation._publish_bytes(
            tmp_path / "absent.json", b"first", allow_create=False
        )
    assert {p.name for p in tmp_path.iterdir()} == {"file.json", "link.json"}


@pytest.fixture
def runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools/run_multimodal_geo_helpfulness_m4_validation.py"
    )
    spec = importlib.util.spec_from_file_location("validation_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", ["prepare", "validate-predictions"])
def test_cli_preparation_dispatch_and_json(
    runner, monkeypatch, tmp_path, capsys, command
):
    calls = []

    def operation(**kwargs):
        calls.append(kwargs)
        return {"status": "test_success"}

    monkeypatch.setattr(validation, "prepare_validation_bundle", operation)
    assert runner.main([command, "--artifact-root", str(tmp_path)]) == 0
    assert calls == [
        {
            "repo_root": runner.PROJECT_ROOT,
            "protocol_root": tmp_path,
            "validate_only": command == "validate-predictions",
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "test_success"}


def test_cli_failure_and_rejection_of_candidate_override(runner, monkeypatch, capsys):
    def failed(**kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(validation, "prepare_validation_bundle", failed)
    assert runner.main(["prepare"]) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "synthetic failure"
    with pytest.raises(SystemExit):
        runner.main(["preview", "--C", "1.0"])

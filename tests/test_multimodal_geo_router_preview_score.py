"""Synthetic label-boundary and fixed-preview tests; no real labels are read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_router_preview_score as preview


def _write_json(path, value):
    path.write_bytes(preview._canonical_bytes(value) + b"\n")
    return preview._file_hash(path)


def _write_frame(path, frame, schema):
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd")
    return preview._file_hash(path)


def _reseal(root, manifest):
    manifest["manifest_sha256"] = preview._hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    _write_json(root / "manifest.json", manifest)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "router_validation_v1"
    root.mkdir()
    code_root = tmp_path / "code_snapshot"
    code_hashes = {}
    for name in preview.CODE_FILES:
        destination = code_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((preview.PROJECT_ROOT / name).read_bytes())
        code_hashes[name] = preview._file_hash(destination)
    monkeypatch.setattr(preview, "PROJECT_ROOT", code_root)
    n = preview.VALIDATION_ROWS
    index = np.arange(n)
    uid = [f"{value:064x}" for value in index]
    plots = [f"plot-{value % preview.VALIDATION_PLOTS}" for value in index]
    labels = (index % 18).astype(np.int8)
    true_state = index % 4
    raw = np.where(np.isin(true_state, [1, 2]), labels, (labels + 1) % 18).astype(np.int8)
    geo = np.where(np.isin(true_state, [0, 2]), labels, (labels + 1) % 18).astype(np.int8)
    expert_frames, prediction_frames = [], []
    for seed in preview.TRAINING_SEEDS:
        identity = {
            "schema_version": [preview.EXPERT_SCHEMA_VERSION] * n,
            "protocol_id": ["protocol_v1"] * n, "row_uid": uid, "plot_idx": plots,
            "training_seed": np.full(n, seed, dtype=np.int8),
        }
        expert_frames.append(pd.DataFrame(dict(identity, image_pred=labels, geo_pred=geo, raw_pred=raw)))
        for C in preview.C_GRID:
            identity = dict(identity, schema_version=[preview.PREDICTION_SCHEMA_VERSION] * n)
            chosen = np.zeros(n, dtype=int)
            if seed == 1 and C == .01:
                chosen = np.where(index < 411, true_state, (true_state + 1) % 4)
            probabilities = np.full((n, 4), .1, dtype=np.float64)
            probabilities[index, chosen] = .7
            data = dict(identity, C_hex=[C.hex()] * n)
            data.update({name: probabilities[:, offset] for offset, name in enumerate(preview.Q_COLUMNS)})
            data["score"] = probabilities[:, 0] - probabilities[:, 1]
            prediction_frames.append(pd.DataFrame(data))
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["C_hex", "training_seed", "row_uid"], kind="stable").reset_index(drop=True)
    experts = pd.concat(expert_frames, ignore_index=True).sort_values(["row_uid", "training_seed"], kind="stable").reset_index(drop=True)
    assignment = pd.DataFrame({
        "protocol_id": ["protocol_v1"] * (n + 4),
        "row_uid": uid + [f"{n + value:064x}" for value in range(4)],
        "plot_idx": plots + [f"train-plot-{value}" for value in range(4)],
        "development_role": ["validation"] * n + ["train"] * 4,
        "label_id_dense": np.concatenate([labels, np.array([0, 1, 2, 3], dtype=np.int8)]),
    })
    assignment_path = tmp_path / "development_assignments.parquet"
    assignment.to_parquet(assignment_path, index=False)
    manifest = {
        "schema_version": preview.MANIFEST_SCHEMA_VERSION,
        "protocol_id": "protocol_v1",
        "assignment_file_sha256": preview._file_hash(assignment_path),
        "dataset_manifest_sha256": "d" * 64,
        "candidate_manifest_sha256": "c" * 64,
        "code_file_sha256": code_hashes,
        "artifacts": {},
    }
    for variant in preview.VARIANTS:
        name = f"{variant}.parquet"
        manifest["artifacts"][name] = {"file_sha256": _write_frame(root / name, predictions, preview.prediction_schema())}
    manifest["artifacts"]["expert_predictions.parquet"] = {"file_sha256": _write_frame(root / "expert_predictions.parquet", experts, preview.expert_schema())}
    declaration = {
        "schema_version": preview.DECLARATION_SCHEMA_VERSION,
        "protocol_id": "protocol_v1", "candidate_variant": "primary_full727",
        "training_seed": 1, "C_hex": .01.hex(), "metric": "four_state_accuracy_only",
        "selection_performed": False,
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
        "code_file_sha256": code_hashes,
    }
    heuristics = {
        "schema_version": "geo_helpfulness.oof_heuristic_thresholds.v1",
        "protocol_id": "protocol_v1", "fit_role": "development_train_oof",
        "validation_outputs_opened": False, "validation_labels_scored": False,
        "test_sources_opened": False,
    }
    declaration["oof_heuristics_sha256"] = preview._hash(heuristics)
    for name, value in (("preview_declaration.json", declaration), ("oof_heuristics.json", heuristics)):
        manifest["artifacts"][name] = {"file_sha256": _write_json(root / name, value)}
    _reseal(root, manifest)
    return root, assignment_path, manifest


def test_fixed_preview_computes_only_declared_accuracy_and_reuses_immutable_output(bundle):
    root, assignment, manifest = bundle
    output = root.parent / "fixed_preview.json"
    before = {path.name: preview._file_hash(path) for path in root.iterdir()}
    result = preview.score_fixed_preview(root, assignment, output_path=output, expected_manifest_sha256=manifest["manifest_sha256"])
    assert result["correct_count"] == 411
    assert result["total_count"] == 822
    assert result["accuracy"] == .5
    assert result["candidate_variant"] == "primary_full727"
    assert result["training_seed"] == 1
    assert result["C_hex"] == .01.hex()
    assert result["selection_performed"] is False
    assert result["other_candidate_metrics_computed"] is False
    assert "log_loss" not in result and "threshold" not in result
    assert result["preview_sha256"] == preview._hash({key: value for key, value in result.items() if key != "preview_sha256"})
    output_before = (output.read_bytes(), output.stat().st_mtime_ns)
    assert output.stat().st_mode & 0o222 == 0
    assert preview.score_fixed_preview(root, assignment, output_path=output) == result
    assert (output.read_bytes(), output.stat().st_mtime_ns) == output_before
    assert {path.name: preview._file_hash(path) for path in root.iterdir()} == before


def test_all_bundle_tables_are_validated_before_assignment_access(bundle, monkeypatch):
    root, assignment, _ = bundle
    accessed = []
    original = preview.pq.read_table

    def read(path, *args, **kwargs):
        accessed.append(Path(path).name)
        if Path(path) == assignment:
            assert set(accessed[:-1]) == {"expert_predictions.parquet", *(f"{variant}.parquet" for variant in preview.VARIANTS)}
            assert kwargs["columns"] == list(preview.ASSIGNMENT_COLUMNS)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(preview.pq, "read_table", read)
    preview.score_fixed_preview(root, assignment)
    assert accessed[-1] == assignment.name


@pytest.mark.parametrize("damage", ["missing", "extra", "hash", "schema", "probability", "score", "grid", "expert_class", "declaration", "manifest"])
def test_incomplete_or_invalid_prelabel_bundle_never_accesses_assignment(bundle, monkeypatch, damage):
    root, assignment, manifest = bundle
    name = "ablation_numeric25.parquet"
    if damage == "missing":
        (root / name).unlink()
    elif damage == "extra":
        (root / "unsealed.json").write_text("{}")
    elif damage == "hash":
        (root / name).write_bytes(b"changed")
    elif damage == "manifest":
        manifest["protocol_id"] = "changed"
        _write_json(root / "manifest.json", manifest)
    elif damage == "declaration":
        path = root / "preview_declaration.json"
        value = json.loads(path.read_text())
        value["C_hex"] = 1.0.hex()
        manifest["artifacts"][path.name]["file_sha256"] = _write_json(path, value)
        _reseal(root, manifest)
    elif damage == "expert_class":
        path = root / "expert_predictions.parquet"
        frame = pq.read_table(path).to_pandas()
        frame.loc[0, "raw_pred"] = 18
        manifest["artifacts"][path.name]["file_sha256"] = _write_frame(path, frame, preview.expert_schema())
        _reseal(root, manifest)
    else:
        path = root / name
        frame = pq.read_table(path).to_pandas()
        if damage == "schema":
            frame["label"] = 0
            frame.to_parquet(path, index=False)
            digest = preview._file_hash(path)
        else:
            if damage == "probability":
                frame.loc[0, "q_rescue"] = 1.1
            elif damage == "score":
                frame.loc[0, "score"] = .12345
            elif damage == "grid":
                frame.loc[0, "C_hex"] = 2.0.hex()
            digest = _write_frame(path, frame, preview.prediction_schema())
        manifest["artifacts"][name]["file_sha256"] = digest
        _reseal(root, manifest)
    original_hash = preview._file_hash

    def hashing(path):
        assert Path(path) != assignment, "assignment accessed before validating prelabel artifacts"
        return original_hash(path)

    monkeypatch.setattr(preview, "_file_hash", hashing)
    with pytest.raises(preview.RouterPreviewError):
        preview.score_fixed_preview(root, assignment)


def test_expected_manifest_anchor_rejects_other_self_consistent_bundle(bundle, monkeypatch):
    root, assignment, _ = bundle
    original = preview._file_hash

    def hashing(path):
        assert Path(path) != assignment
        return original(path)

    monkeypatch.setattr(preview, "_file_hash", hashing)
    with pytest.raises(preview.RouterPreviewError, match="expected identity"):
        preview.score_fixed_preview(root, assignment, expected_manifest_sha256="b" * 64)


@pytest.mark.parametrize("damage", ["code_content", "code_declaration", "code_allowlist", "parent_identity", "heuristic_content", "heuristic_scope", "manifest_heuristic_hash"])
def test_predeclared_code_and_prerequisite_fingerprints_checked_before_labels(bundle, monkeypatch, damage):
    root, assignment, manifest = bundle
    declaration_path = root / "preview_declaration.json"
    declaration = json.loads(declaration_path.read_text())
    if damage == "code_content":
        (preview.PROJECT_ROOT / preview.CODE_FILES[0]).write_text("changed after predeclaration")
    elif damage == "code_declaration":
        declaration["code_file_sha256"][preview.CODE_FILES[0]] = "b" * 64
    elif damage == "code_allowlist":
        del declaration["code_file_sha256"][preview.CODE_FILES[0]]
        manifest["code_file_sha256"] = dict(declaration["code_file_sha256"])
    elif damage == "parent_identity":
        declaration["dataset_manifest_sha256"] = "b" * 64
    elif damage in {"heuristic_content", "heuristic_scope"}:
        path = root / "oof_heuristics.json"
        heuristic = json.loads(path.read_text())
        if damage == "heuristic_content":
            heuristic["new_threshold"] = .5
        else:
            heuristic["validation_labels_scored"] = True
            declaration["oof_heuristics_sha256"] = preview._hash(heuristic)
        manifest["artifacts"][path.name]["file_sha256"] = _write_json(path, heuristic)
    elif damage == "manifest_heuristic_hash":
        manifest["oof_heuristics_sha256"] = "b" * 64
    manifest["artifacts"][declaration_path.name]["file_sha256"] = _write_json(declaration_path, declaration)
    _reseal(root, manifest)
    original = preview._file_hash

    def hashing(path):
        assert Path(path) != assignment, "validation assignment was accessed before provenance rejection"
        return original(path)

    monkeypatch.setattr(preview, "_file_hash", hashing)
    with pytest.raises(preview.RouterPreviewError):
        preview.score_fixed_preview(root, assignment)


def test_public_sealed_validator_never_accesses_assignment_file(bundle, monkeypatch):
    root, assignment, manifest = bundle
    original_hash, original_read = preview._file_hash, preview.pq.read_table

    def hashing(path):
        assert Path(path) != assignment
        return original_hash(path)

    def read(path, *args, **kwargs):
        assert Path(path) != assignment
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(preview, "_file_hash", hashing)
    monkeypatch.setattr(preview.pq, "read_table", read)
    result = preview.validate_sealed_predictions(root, expected_manifest_sha256=manifest["manifest_sha256"])
    assert result["valid"] is True
    assert result["candidate_count"] == 48
    assert result["validation_plots"] == 325
    assert result["assignment_content_opened"] is False


@pytest.mark.parametrize("name,schema", [("expert_predictions.parquet", preview.expert_schema()), ("primary_full727.parquet", preview.prediction_schema())])
def test_noncanonical_identity_order_is_rejected(bundle, name, schema):
    root, _, manifest = bundle
    frame = pq.read_table(root / name).to_pandas().iloc[::-1].reset_index(drop=True)
    manifest["artifacts"][name]["file_sha256"] = _write_frame(root / name, frame, schema)
    _reseal(root, manifest)
    with pytest.raises(preview.RouterPreviewError, match="canonical identity order"):
        preview.validate_sealed_predictions(root)


@pytest.mark.parametrize("damage", ["hash", "label", "duplicate", "plot", "protocol", "missing_validation", "float_label"])
def test_assignment_corruption_or_misalignment_rejected(bundle, damage):
    root, assignment, manifest = bundle
    frame = pq.read_table(assignment).to_pandas()
    if damage == "hash":
        frame.loc[0, "label_id_dense"] = 17
    elif damage == "label":
        frame.loc[0, "label_id_dense"] = 18
    elif damage == "duplicate":
        frame.loc[0, "row_uid"] = frame.loc[1, "row_uid"]
    elif damage == "plot":
        frame.loc[0, "plot_idx"] = "wrong-plot"
    elif damage == "protocol":
        frame.loc[0, "protocol_id"] = "other_protocol"
    elif damage == "missing_validation":
        frame.loc[0, "development_role"] = "train"
    elif damage == "float_label":
        frame["label_id_dense"] = frame.label_id_dense.astype(float)
    frame.to_parquet(assignment, index=False)
    if damage != "hash":
        manifest["assignment_file_sha256"] = preview._file_hash(assignment)
        _reseal(root, manifest)
    with pytest.raises(preview.RouterPreviewError):
        preview.score_fixed_preview(root, assignment)


def test_training_labels_do_not_affect_fixed_validation_preview(bundle):
    root, assignment, manifest = bundle
    baseline = preview.score_fixed_preview(root, assignment)
    frame = pq.read_table(assignment).to_pandas()
    frame.loc[frame.development_role.eq("train"), "label_id_dense"] = 17
    frame.to_parquet(assignment, index=False)
    manifest["assignment_file_sha256"] = preview._file_hash(assignment)
    _reseal(root, manifest)
    changed = preview.score_fixed_preview(root, assignment)
    assert changed["correct_count"] == baseline["correct_count"]
    assert changed["accuracy"] == baseline["accuracy"]


def test_output_cannot_modify_sealed_bundle_or_assignment(bundle):
    root, assignment, _ = bundle
    for output in (root / "preview.json", assignment):
        with pytest.raises(preview.RouterPreviewError, match="output"):
            preview.score_fixed_preview(root, assignment, output_path=output)


def test_existing_different_preview_is_never_overwritten(bundle):
    root, assignment, _ = bundle
    output = root.parent / "previous_preview.json"
    output.write_text('{"owned_by_user": true}')
    before = output.read_bytes()
    with pytest.raises(preview.RouterPreviewError, match="immutable preview result differs"):
        preview.score_fixed_preview(root, assignment, output_path=output)
    assert output.read_bytes() == before


def test_prediction_or_assignment_mutation_during_scoring_rejected(bundle, monkeypatch):
    root, assignment, _ = bundle
    original = preview.pq.read_table

    def read(path, *args, **kwargs):
        table = original(path, *args, **kwargs)
        if Path(path) == assignment:
            frame = table.to_pandas()
            frame.loc[0, "label_id_dense"] = 17
            frame.to_parquet(assignment, index=False)
        return table

    monkeypatch.setattr(preview.pq, "read_table", read)
    with pytest.raises(preview.RouterPreviewError, match="changed during scoring"):
        preview.score_fixed_preview(root, assignment)


def test_fresh_process_module_imports_no_fitting_or_parent_pipeline():
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys; import multimodal.geo_helpfulness_router_preview_score; "
        "blocked=('sklearn','scipy','torch','multimodal.geo_helpfulness_router_candidates',"
        "'multimodal.geo_helpfulness_router_candidate_fit','multimodal.geo_helpfulness_router_dataset',"
        "'multimodal.geo_helpfulness_oof','multimodal.geo_helpfulness_targets_features'); "
        "assert not any(name==prefix or name.startswith(prefix+'.') for prefix in blocked for name in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

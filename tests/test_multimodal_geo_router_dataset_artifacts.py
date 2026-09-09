"""Synthetic integration tests for immutable M4 dataset publication."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_router_dataset as m4
from multimodal import geo_helpfulness_targets_features as m3


def _input_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = pd.DataFrame(
        [
            {
                "schema_version": "geo_helpfulness_protocol_config_v1",
                "protocol_id": "protocol_v1",
                "row_uid": f"{index + 1:064x}",
                "file": f"images/{index}.jpg",
                "file_lower": f"images/{index}.jpg",
                "plot_idx": f"plot-{index % 4}",
                "train_oof_fold": index % 4,
                "development_role": "train",
                "label_id_dense": index % 2,
            }
            for index in range(16)
        ]
    )
    rows = []
    for seed in m4.TRAINING_SEEDS:
        for index, assignment in assignments.iterrows():
            row = assignment.drop(["development_role", "label_id_dense"]).to_dict()
            row["training_seed"] = seed
            for mode_index, prefix in enumerate(m4.MODE_TO_PREFIX.values()):
                correct = (index + mode_index) % 4 != 0
                prediction = (
                    int(assignment.label_id_dense)
                    if correct
                    else 1 - int(assignment.label_id_dense)
                )
                logits = np.zeros(18, dtype=np.float64)
                logits[prediction] = 4.0 + 0.3 * seed + 0.2 * mode_index
                row[f"{prefix}_logits"] = logits
                row[f"{prefix}_pred"] = prediction
            rows.append(row)
    oof = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    assignments["label_id_dense"] = assignments.label_id_dense.astype(np.int8)
    targets = m3.build_router_target_table(oof, assignments, protocol_id="protocol_v1")
    return oof, assignments, targets


def _assemble(
    oof: pd.DataFrame, assignments: pd.DataFrame, targets: pd.DataFrame
) -> pd.DataFrame:
    return m4.assemble_training_inputs(
        oof,
        assignments,
        targets,
        protocol_id="protocol_v1",
        expected_rows=16,
        expected_plots=4,
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    oof, assignments, targets = _input_frames()
    parent = tmp_path / "sealed_parent.json"
    parent.write_text('{"immutable": true}\n', encoding="utf-8")
    value = m4.PreparedInputs(
        root=tmp_path / "router",
        protocol_id="protocol_v1",
        calibration_spec=deepcopy(m4.CALIBRATION_SPEC),
        training=_assemble(oof, assignments, targets),
        targets=targets,
        feature_schema=m3.build_router_feature_schema(),
        lineage={"fixture_parent": m4.sha256_file(parent)},
        parent_files={str(parent): m4.sha256_file(parent)},
    )
    monkeypatch.setattr(m4, "_prepare_inputs", lambda **kwargs: value)
    return value


def _forbid_fit(*args, **kwargs):
    raise AssertionError("fitting is forbidden while validating or reusing")


def _hashes(root: Path) -> dict[str, str]:
    return {name: m4.sha256_file(root / name) for name in m4.BUNDLE_FILENAMES}


def _overwrite_json(path: Path, value: dict) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _refresh_manifest(prepared) -> None:
    _overwrite_json(
        prepared.root / m4.MANIFEST_FILENAME,
        m4._build_manifest(prepared, prepared.root),
    )


def test_build_validate_load_and_reuse_are_apply_only_and_byte_preserving(
    prepared, monkeypatch
):
    fitted = []
    original = m4.fit_expert_temperature

    def fit(logits, labels, **kwargs):
        fitted.append((kwargs["seed"], kwargs["mode"], len(labels)))
        return original(logits, labels, **kwargs)

    monkeypatch.setattr(m4, "fit_expert_temperature", fit)
    result = m4.build_router_dataset_bundle()
    assert result["status"] == "created"
    assert result["calibration_fitted_this_call"] is True
    assert fitted == [
        (seed, mode, 16) for seed in m4.TRAINING_SEEDS for mode in m4.MODE_TO_PREFIX
    ]
    assert not (prepared.root / m4.RECEIPT_FILENAME).exists()
    before = _hashes(prepared.root)
    monkeypatch.setattr(m4, "fit_expert_temperature", _forbid_fit)
    monkeypatch.setattr(m4, "fit_router_feature_transform", _forbid_fit)

    assert m4.validate_router_dataset_bundle()["valid"] is True
    bundle = m4.load_validated_router_dataset_bundle()
    reused = m4.build_router_dataset_bundle()
    assert reused["status"] == "reused_valid"
    assert reused["calibration_fitted_this_call"] is False
    assert _hashes(prepared.root) == before
    assert bundle.dataset.shape == (64, 733)
    assert bundle.audit.shape == (64, 39)
    assert bundle.dataset.schema.equals(m4.dataset_schema(), check_metadata=True)
    assert bundle.audit.schema.equals(m4.audit_schema(), check_metadata=True)
    assert all(not field.nullable for field in bundle.dataset.schema)
    assert all(not field.nullable for field in bundle.audit.schema)
    assert len(bundle.temperatures["states"]) == 12
    assert len(bundle.transforms["states"]) == 4
    assert bundle.manifest["feature_columns"] == list(m4.output_feature_columns())
    pd.testing.assert_frame_equal(
        bundle.dataset.select(m3.TARGET_COLUMNS)
        .to_pandas()
        .astype(
            {name: "string" for name in m3.TARGET_COLUMNS if name != "training_seed"}
        ),
        prepared.targets,
    )
    assert not set(bundle.audit.column_names).intersection(
        {
            "label_id_dense",
            "target_state",
            "raw_correct",
            "geo_correct",
            "utility",
            "nll",
        }
    )
    matrix = bundle.dataset.select(m4.output_feature_columns()).to_pandas().to_numpy()
    assert matrix.dtype == np.float64 and np.isfinite(matrix).all()
    assert all(
        state["calibrated_nll"] <= state["native_nll"]
        for state in bundle.temperatures["states"]
    )
    for prefix in m4.MODE_TO_PREFIX.values():
        values = np.stack(bundle.audit[f"{prefix}_prob_calibrated"].to_pylist())
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    assert json.loads(json.dumps(bundle.temperatures)) == bundle.temperatures
    assert json.loads(json.dumps(bundle.transforms)) == bundle.transforms


def test_seed_isolation_and_input_permutations(prepared):
    baseline = m4._derive_tables(prepared)
    shuffled = replace(
        prepared,
        training=prepared.training.sample(frac=1, random_state=99),
        targets=prepared.targets.sample(frac=1, random_state=100),
    )
    permuted = m4._derive_tables(shuffled)
    assert baseline[0].equals(permuted[0])
    assert baseline[1].equals(permuted[1])
    assert baseline[2:] == permuted[2:]

    changed = prepared.training.copy(deep=True)
    row_index = changed.index[changed.training_seed.eq(4)][0]
    logits = changed.at[row_index, "image_logits"].copy()
    logits[17] = 0.8
    changed.at[row_index, "image_logits"] = logits
    altered = m4._derive_tables(replace(prepared, training=changed))
    assert altered[2]["states"][:9] == baseline[2]["states"][:9]
    assert altered[3]["states"][:3] == baseline[3]["states"][:3]
    assert altered[2]["states"][9] != baseline[2]["states"][9]
    base_frame, altered_frame = baseline[0].to_pandas(), altered[0].to_pandas()
    pd.testing.assert_frame_equal(
        base_frame.loc[base_frame.training_seed.ne(4)],
        altered_frame.loc[altered_frame.training_seed.ne(4)],
    )


def test_training_projection_ignores_validation_labels_and_canonicalizes_rows():
    oof, assignments, targets = _input_frames()
    validation = assignments.iloc[[0]].copy()
    validation["row_uid"] = f"{999:064x}"
    validation["development_role"] = "validation"
    validation["label_id_dense"] = 99
    joined = _assemble(
        oof, pd.concat([validation, assignments], ignore_index=True), targets
    )
    assert len(joined) == 64
    assert set(joined.label_id_dense) == {0, 1}
    assert list(zip(joined.row_uid, joined.training_seed)) == sorted(
        zip(joined.row_uid, joined.training_seed)
    )
    assert set(joined.row_uid) == set(assignments.row_uid)


def test_training_adapter_accepts_target_string_dtypes_after_arrow_roundtrip():
    oof, assignments, targets = _input_frames()
    roundtripped = (
        pa.Table.from_pandas(targets, preserve_index=False)
        .replace_schema_metadata(None)
        .to_pandas()
    )
    assert roundtripped.row_uid.dtype == object
    pd.testing.assert_frame_equal(
        _assemble(oof, assignments, targets),
        _assemble(oof, assignments, roundtripped),
    )


@pytest.mark.parametrize("column", ["label_id_dense", "development_role"])
def test_oof_label_or_role_injection_is_rejected(column):
    oof, assignments, targets = _input_frames()
    oof[column] = 0 if column == "label_id_dense" else "train"
    with pytest.raises(m4.M4ArtifactError, match="label-blind|label|role"):
        _assemble(oof, assignments, targets)


def test_training_adapter_requires_all_four_folds_even_when_identities_agree():
    oof, assignments, targets = _input_frames()
    assignments.loc[assignments.train_oof_fold.eq(3), "train_oof_fold"] = 2
    oof.loc[oof.train_oof_fold.eq(3), "train_oof_fold"] = 2
    targets = m3.build_router_target_table(oof, assignments, protocol_id="protocol_v1")
    with pytest.raises(m4.M4ArtifactError, match="four OOF folds"):
        _assemble(oof, assignments, targets)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_oof", "duplicate"),
        ("missing_oof", "row count"),
        ("wrong_seed", "seeds"),
        ("wrong_fold", "identity mismatch"),
        ("wrong_file", "identity mismatch"),
        ("wrong_plot", "identity mismatch"),
        ("wrong_protocol", "protocol"),
        ("unknown_uid", "cover every"),
        ("missing_assignment", "counts"),
        ("duplicate_assignment", "duplicate"),
        ("invalid_label", "labels"),
        ("float_label", "labels"),
        ("missing_label", "missing"),
        ("wrong_target", "targets differ"),
    ],
)
def test_training_adapter_rejects_inconsistent_inputs(mutation, match):
    oof, assignments, targets = _input_frames()
    if mutation == "duplicate_oof":
        oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    elif mutation == "missing_oof":
        oof = oof.iloc[1:]
    elif mutation == "wrong_seed":
        oof.loc[oof.training_seed.eq(4), "training_seed"] = 5
    elif mutation == "wrong_fold":
        oof.loc[0, "train_oof_fold"] = (oof.loc[0, "train_oof_fold"] + 1) % 4
    elif mutation == "wrong_file":
        oof.loc[0, "file"] = "images/other.jpg"
    elif mutation == "wrong_plot":
        oof.loc[0, "plot_idx"] = "different-plot"
    elif mutation == "wrong_protocol":
        oof.loc[0, "protocol_id"] = "protocol_v2"
    elif mutation == "unknown_uid":
        oof.loc[0, "row_uid"] = f"{999:064x}"
    elif mutation == "missing_assignment":
        assignments = assignments.iloc[1:]
    elif mutation == "duplicate_assignment":
        assignments = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
    elif mutation == "invalid_label":
        assignments.loc[0, "label_id_dense"] = 18
    elif mutation == "float_label":
        assignments["label_id_dense"] = assignments.label_id_dense.astype(float)
    elif mutation == "missing_label":
        assignments["label_id_dense"] = assignments.label_id_dense.astype(float)
        assignments.loc[0, "label_id_dense"] = np.nan
    elif mutation == "wrong_target":
        targets.loc[0, "target_state"] = (
            "harm" if targets.loc[0, "target_state"] != "harm" else "rescue"
        )
    with pytest.raises(m4.M4ArtifactError, match=match):
        _assemble(oof, assignments, targets)


def test_manifest_self_hash_tampering_is_rejected(prepared):
    m4.build_router_dataset_bundle()
    path = prepared.root / m4.MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["rows"] += 1
    _overwrite_json(path, manifest)
    with pytest.raises(m4.M4ArtifactError, match="self-hash"):
        m4.validate_router_dataset_bundle()


def test_changed_parent_snapshot_and_lineage_are_rejected(prepared, monkeypatch):
    m4.build_router_dataset_bundle()
    changed = replace(prepared, lineage={"fixture_parent": "different"})
    monkeypatch.setattr(m4, "_prepare_inputs", lambda **kwargs: changed)
    with pytest.raises(m4.M4ArtifactError, match="stale"):
        m4.validate_router_dataset_bundle()
    monkeypatch.setattr(m4, "_prepare_inputs", lambda **kwargs: prepared)
    Path(next(iter(prepared.parent_files))).write_text("changed\n")
    with pytest.raises(m4.M4ArtifactError, match="parent or implementation changed"):
        m4.validate_router_dataset_bundle()


@pytest.mark.parametrize(
    "artifact,key",
    [
        (m4.TEMPERATURE_FILENAME, "calibrated_nll"),
        (m4.TRANSFORM_FILENAME, "mean"),
    ],
)
def test_state_reconstruction_catches_rehashed_tampering(prepared, artifact, key):
    m4.build_router_dataset_bundle()
    path = prepared.root / artifact
    state = json.loads(path.read_text())
    if key == "mean":
        state["states"][0][key][0] += 0.1
    else:
        state["states"][0][key] -= 0.01
    _overwrite_json(path, state)
    _refresh_manifest(prepared)
    with pytest.raises(m4.M4ArtifactError, match="reconstruction mismatch"):
        m4.validate_router_dataset_bundle()


def test_dataset_reconstruction_catches_rehashed_matrix_tampering(prepared):
    m4.build_router_dataset_bundle()
    path = prepared.root / m4.DATASET_FILENAME
    table = pq.read_table(path)
    index = table.schema.get_field_index(m4.output_feature_columns()[0])
    values = table.column(index).to_numpy().copy()
    values[0] += 0.25
    table = table.set_column(index, table.schema.field(index), pa.array(values))
    path.chmod(0o644)
    pq.write_table(table, path)
    _refresh_manifest(prepared)
    with pytest.raises(m4.M4ArtifactError, match="reconstruction mismatch"):
        m4.validate_router_dataset_bundle()


@pytest.mark.parametrize("fail_at", [m4.AUDIT_FILENAME, m4.MANIFEST_FILENAME])
def test_manifest_last_and_hash_owned_interruption_recovery_preserve_siblings(
    prepared, monkeypatch, fail_at
):
    prepared.root.mkdir()
    sibling = prepared.root / "unrelated-user-file.txt"
    sibling.write_text("must survive\n")
    m3_directory = prepared.root / "targets_and_feature_contract"
    m3_directory.mkdir()
    marker = m3_directory / "marker.json"
    marker.write_text("sealed M3\n")
    original = m4._exclusive_publish
    published = []

    def interrupted(staged, destination):
        if destination.name == fail_at:
            raise RuntimeError("simulated publication interruption")
        original(staged, destination)
        published.append(destination.name)

    monkeypatch.setattr(m4, "_exclusive_publish", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        m4.build_router_dataset_bundle()
    assert not (prepared.root / m4.MANIFEST_FILENAME).exists()
    assert (prepared.root / m4.RECEIPT_FILENAME).is_file()
    assert published and published[0] == m4.DATASET_FILENAME
    assert not list(prepared.root.glob(".router_dataset.m4.staging-*"))
    monkeypatch.setattr(m4, "_exclusive_publish", original)
    result = m4.build_router_dataset_bundle()
    assert result["status"] == "created"
    assert result["recovered_uncommitted_files"] == published
    assert sibling.read_text() == "must survive\n"
    assert marker.read_text() == "sealed M3\n"
    assert not (prepared.root / m4.RECEIPT_FILENAME).exists()


def test_unowned_target_file_is_never_removed(prepared):
    prepared.root.mkdir()
    path = prepared.root / m4.DATASET_FILENAME
    path.write_bytes(b"user-owned existing output")
    with pytest.raises(m4.M4ArtifactError, match="no ownership receipt"):
        m4.build_router_dataset_bundle()
    assert path.read_bytes() == b"user-owned existing output"


def test_modified_partial_output_is_not_receipt_owned(prepared, monkeypatch):
    original = m4._exclusive_publish

    def interrupted(staged, destination):
        if destination.name == m4.AUDIT_FILENAME:
            raise RuntimeError("interrupt")
        original(staged, destination)

    monkeypatch.setattr(m4, "_exclusive_publish", interrupted)
    with pytest.raises(RuntimeError):
        m4.build_router_dataset_bundle()
    path = prepared.root / m4.DATASET_FILENAME
    path.chmod(0o644)
    path.write_bytes(b"subsequent user content")
    monkeypatch.setattr(m4, "_exclusive_publish", original)
    with pytest.raises(m4.M4ArtifactError, match="not the receipt-owned content"):
        m4.build_router_dataset_bundle()
    assert path.read_bytes() == b"subsequent user content"


def test_forged_ownership_receipt_cannot_claim_unrelated_files(prepared):
    prepared.root.mkdir()
    sibling = prepared.root / "user_notes.txt"
    sibling.write_text("do not remove\n")
    hashes = {name: "a" * 64 for name in m4.CHILD_FILENAMES}
    hashes[sibling.name] = m4.sha256_file(sibling)
    receipt = m4._receipt_payload(prepared, hashes, "b" * 32)
    (prepared.root / m4.RECEIPT_FILENAME).write_text(json.dumps(receipt))
    with pytest.raises(m4.M4ArtifactError, match="invalid M4 ownership receipt"):
        m4.build_router_dataset_bundle()
    assert sibling.read_text() == "do not remove\n"


def test_receipt_left_after_manifest_commit_is_safely_removed_on_reuse(
    prepared, monkeypatch
):
    m4.build_router_dataset_bundle()
    before = _hashes(prepared.root)
    receipt = m4._receipt_payload(
        prepared, {name: before[name] for name in m4.CHILD_FILENAMES}, "c" * 32
    )
    (prepared.root / m4.RECEIPT_FILENAME).write_text(json.dumps(receipt))
    monkeypatch.setattr(m4, "fit_expert_temperature", _forbid_fit)
    monkeypatch.setattr(m4, "fit_router_feature_transform", _forbid_fit)
    assert m4.build_router_dataset_bundle()["status"] == "reused_valid"
    assert not (prepared.root / m4.RECEIPT_FILENAME).exists()
    assert _hashes(prepared.root) == before


def test_lock_contention_rejects_second_builder(prepared):
    with m4._workflow_lock(prepared.root):
        with pytest.raises(m4.M4ArtifactError, match="another M4 process"):
            m4.build_router_dataset_bundle()
    assert not (prepared.root / m4.MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("target", ["root", "lock", "committed_child"])
def test_symlinks_are_rejected_without_modifying_referents(
    prepared, monkeypatch, tmp_path, target
):
    if target == "root":
        real_root = tmp_path / "real-router"
        real_root.mkdir()
        prepared.root.symlink_to(real_root, target_is_directory=True)
        with pytest.raises(m4.M4ArtifactError, match="non-symlink directory"):
            m4.build_router_dataset_bundle()
        assert not list(real_root.iterdir())
    elif target == "lock":
        prepared.root.mkdir()
        referent = tmp_path / "other-lock"
        referent.write_text("other lock\n")
        (prepared.root / m4.LOCK_FILENAME).symlink_to(referent)
        with pytest.raises(m4.M4ArtifactError, match="lock must not be a symlink"):
            m4.build_router_dataset_bundle()
        assert referent.read_text() == "other lock\n"
    else:
        m4.build_router_dataset_bundle()
        path = prepared.root / m4.DATASET_FILENAME
        referent = tmp_path / "saved-dataset.parquet"
        path.rename(referent)
        digest = m4.sha256_file(referent)
        path.symlink_to(referent)
        with pytest.raises(m4.M4ArtifactError, match="regular non-symlink"):
            m4.validate_router_dataset_bundle()
        assert m4.sha256_file(referent) == digest


def test_parent_changed_during_build_prevents_publication(prepared, monkeypatch):
    original = m4._derive_tables

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        Path(next(iter(prepared.parent_files))).write_text(
            "changed during derivation\n"
        )
        return result

    monkeypatch.setattr(m4, "_derive_tables", changed)
    with pytest.raises(m4.M4ArtifactError, match="parent or implementation changed"):
        m4.build_router_dataset_bundle()
    assert not (prepared.root / m4.MANIFEST_FILENAME).exists()
    assert not any((prepared.root / name).exists() for name in m4.CHILD_FILENAMES)

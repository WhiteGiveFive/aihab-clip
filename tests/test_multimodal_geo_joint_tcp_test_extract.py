from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from safetensors.torch import save_file
from torch import nn

from multimodal import geo_helpfulness_joint_tcp_test_extract as extraction
from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_confidnet import _seal_stage
from multimodal.geo_helpfulness_protocol import build_locked_test_identity_manifest, sha256_file
from multimodal.models import MLPHead


@pytest.fixture
def source(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    image_root.mkdir()
    monkeypatch.setattr(extraction.test_source, "TEST_IMAGE_ROOT", image_root)
    monkeypatch.setattr(m2, "PROJECT_ROOT", tmp_path)
    files = ["four.jpg", "one.jpg", "three.jpg", "two.jpg"]
    for index, name in enumerate(files):
        (image_root / name).write_bytes(bytes([index]))
    frame = pd.DataFrame({
        "file": files, "plot_idx": ["t2", "t0", "t1", "t0"],
        "image_source": [str(image_root)] * 4,
        "label_id": [91, 92, 93, 94], "I0000": [999.] * 4,
        **{name: np.arange(4, dtype=np.float64) + column
           for column, name in enumerate(m2.GEO_COLUMNS)},
    })
    source_path = tmp_path / "source.parquet"
    frame.to_parquet(source_path, index=False)
    manifest = build_locked_test_identity_manifest(frame, "synthetic")
    registry = tmp_path / "registry"
    registry.mkdir()
    identity_path = registry / "identity.json"
    identity_path.write_text(json.dumps(manifest))
    (registry / "active_snapshot.json").write_text(json.dumps({"snapshot_id": manifest["snapshot_id"]}))
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    (protocol_dir / "locked_test_snapshot_ref.json").write_text(json.dumps({
        "identity_manifest_relative_path": "identity.json",
        "identity_manifest_file_sha256": sha256_file(identity_path),
        "snapshot_id": manifest["snapshot_id"],
        "identity_projection_sha256": manifest["identity_projection_sha256"],
    }))
    assignments = pd.DataFrame({
        "row_uid": [f"u{i:03d}" for i in range(16)],
        "file": [f"image/{i}.jpg" for i in range(16)],
        "file_lower": [f"image/{i}.jpg" for i in range(16)],
        "plot_idx": [f"p{i}" for i in range(16)],
        "image_source": ["/development/images"] * 16,
        "label_id_dense": np.arange(16, dtype=np.int64),
        "development_role": ["train"] * 12 + ["validation"] * 4,
        "train_oof_fold": pd.array([i % 4 for i in range(12)] + [pd.NA] * 4, dtype="Int8"),
    })
    context = SimpleNamespace(
        output_root=tmp_path, protocol_dir=protocol_dir, assignments=assignments,
        parent_hashes={"synthetic": "parent"},
        protocol_manifest={"locked_test_identity_projection_sha256": manifest["identity_projection_sha256"]},
        config={
            "protocol_id": "synthetic", "dataset_id": "synthetic",
            "paths": {"locked_test_registry_root": str(registry)},
            "locked_test_identity": {"expected_rows": 4, "expected_plots": 3},
            "development_universe": {"allowed_image_sources": ["/development/images"]},
            "experts": {
                "head": {"hidden_dim": 256, "dropout": 0.7},
                "mode_inputs": {"image_only": {"input_dim": 1152},
                                "geo_only": {"input_dim": 64},
                                "raw_concat": {"input_dim": 1216}},
            },
        },
    )
    monkeypatch.setattr(m2, "load_frozen_context", lambda **kwargs: context)
    return context, source_path, frame, manifest, image_root


def test_source_projection_excludes_labels_and_stale_image_features(source, monkeypatch):
    context, path, frame, manifest, _ = source
    original = pd.read_parquet
    calls = []

    def guarded_read(path, *, columns):
        calls.append(columns)
        assert columns == list(extraction.SOURCE_COLUMNS)
        return original(path, columns=columns)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    rows, geo, metadata = extraction.load_test_sources(context=context, source_path=path)
    assert len(calls) == 1
    assert rows.row_uid.tolist() == [row["row_uid"] for row in manifest["rows"]]
    assert tuple(rows.columns) == extraction.joint.initial.IDENTITY_COLUMNS
    expected = frame.set_index("file").loc[rows.file, list(m2.GEO_COLUMNS)].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(geo, expected)
    assert metadata["test_labels_read"] is False
    assert metadata["cached_image_embeddings_read"] is False
    assert len(metadata["raw_image_file_sha256"]) == 4


@pytest.fixture
def replay(source, monkeypatch, tmp_path):
    context, path, *_ = source
    spec = extraction.joint._fold_spec(1, None)
    fit_rows, validation_rows = m2._partitions_for_spec(context.assignments, spec)
    directory = context.output_root / spec.relative_directory
    directory.mkdir(parents=True)
    encoder = nn.Module()
    encoder.visual = nn.Linear(2, 4)
    save_file(encoder.visual.state_dict(), str(directory / "adapted_visual_tower.safetensors"))
    for mode in m2.MODES:
        head = MLPHead(context.config["experts"]["mode_inputs"][mode]["input_dim"], 256, 18, dropout=0.7)
        save_file(head.state_dict(), str(directory / f"{mode}_head.safetensors"))
    (directory / "geo_standardization.json").write_text("{}")
    (directory / spec.output_filename).write_bytes(b"sealed reference logits")
    producer = {"valid": True, "stage_id": spec.stage_id, "training_seed": 1}
    monkeypatch.setattr(m2, "validate_producer", lambda *args, **kwargs: producer)
    parent_root = tmp_path / "joint_study"
    parent_root.mkdir()
    monkeypatch.setattr(extraction, "PARENT_JOINT_ROOT", parent_root)
    checkpoints = {name: sha256_file(directory / name) for name in extraction.CHECKPOINT_NAMES}
    scaler_hash = sha256_file(directory / "geo_standardization.json")
    reference_hash = sha256_file(directory / spec.output_filename)
    parent_metadata = {
        "schema_version": "geo_helpfulness.joint_tcp_extraction_producer.v1",
        "training_seed": 1, "producer_fold": None, "producer": producer,
        "protocol_id": "synthetic", "parent_artifact_hashes": context.parent_hashes,
        "expert_fit": extraction.joint._identity_summary(fit_rows),
        "prediction": extraction.joint._identity_summary(validation_rows),
        "image_dimension": 1152, "geo_dimension": 64,
        "additional_embedding_scaling": False, "expert_eval_mode": True,
        "probability_basis": "native_t1_uncalibrated",
        "logits_source": "authoritative_saved_M2_outputs", "validation_labels_in_payload": False,
        "checkpoint_file_sha256": checkpoints, "scaler_file_sha256": scaler_hash,
        "reference_output_file_sha256": reference_hash,
    }
    parent_hashes = {
        **checkpoints, "geo_standardization.json": scaler_hash, spec.output_filename: reference_hash,
    }
    (parent_root / "declaration.json").write_text(json.dumps({
        "file_sha256": {str((directory / name).relative_to(tmp_path)): value
                        for name, value in parent_hashes.items()},
    }))
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "extraction.json").write_text(json.dumps(parent_metadata))
    (staging / "features.npz").write_bytes(b"sealed validation feature cache")
    _seal_stage(staging, parent_root / "producers/seed_1/validation", parent_root,
                role="joint_tcp_producer", metadata={"seed": 1, "fold": None})
    calls = []

    def fake_images(model, rows, **kwargs):
        assert tuple(rows.columns) == extraction.joint.initial.IDENTITY_COLUMNS
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert kwargs["config"]["development_universe"]["allowed_image_sources"] == [str(extraction.test_source.TEST_IMAGE_ROOT)]
        calls.append(rows.row_uid.tolist())
        images = np.zeros((len(rows), 1152), dtype=np.float32)
        images[:, 0] = 1
        return images

    monkeypatch.setattr(m2, "_load_fresh_encoder", lambda *args: (encoder, None, "transform", None, {}))
    monkeypatch.setattr(m2, "_extract_embeddings", fake_images)
    monkeypatch.setattr(m2, "_validate_scaler_file", lambda path: {"mean": np.ones(64), "std": np.full(64, 2.)})
    result = extraction.extract_seed(1, device="cpu", source_path=path)
    return result, context, path, directory, parent_root, calls


def test_uses_full_development_train_bundle_and_correct_embeddings(replay):
    result, context, path, directory, parent_root, calls = replay
    assert len(calls) == 1
    assert np.all(result["test"]["producer_fold"] == -1)
    assert result["metadata"]["producer_fold"] is None
    assert result["metadata"]["expert_fit"]["row_count"] == 12
    assert result["metadata"]["same_expert_bundle_as_joint_validation"] is True
    assert "labels" not in result["test"]
    assert not any(name.endswith("_hidden") for name in result["test"])
    for name, dim in (("image_embedding", 1152), ("geo_embedding", 64)):
        assert result["test"][name].shape == (4, dim)
        assert result["test"][name].dtype == np.float32
    for prefix in extraction.PREFIXES:
        assert result["test"][f"{prefix}_logits"].shape == (4, 18)
        assert result["test"][f"{prefix}_logits"].dtype == np.float64
    assert extraction.validate_seed_payload(result, 1, context=context, source_path=path)["valid"]
    json.dumps(result["metadata"])


@pytest.mark.parametrize("field", ["labels", "label_id_dense", "tcp", "I0000", "image_hidden"])
def test_payload_forbids_labels_targets_and_hidden_activations(replay, field):
    result, context, path, *_ = replay
    result["test"][field] = np.zeros(4)
    with pytest.raises(ValueError, match="allow-list"):
        extraction.validate_seed_payload(result, 1, context=context, source_path=path)


@pytest.mark.parametrize("corruption", [
    "identity", "producer_fold", "producer_dtype", "image_dtype", "image_norm",
    "geo_wrong_scaler", "logits_nonfinite", "metadata_fold", "checkpoint", "scaler",
    "raw_image", "source", "metadata_fit", "parent_extraction", "parent_features",
    "parent_declaration", "parent_lineage", "wrong_seed",
])
def test_saved_replay_rejects_identity_payload_and_lineage_corruption(replay, corruption):
    result, context, path, directory, parent_root, _ = replay
    if corruption == "identity":
        result["test"]["plot_idx"][0] = "bad"
    elif corruption == "producer_fold":
        result["test"]["producer_fold"][0] = 3
    elif corruption == "producer_dtype":
        result["test"]["producer_fold"] = result["test"]["producer_fold"].astype(np.float32)
    elif corruption == "image_dtype":
        result["test"]["image_embedding"] = result["test"]["image_embedding"].astype(np.float64)
    elif corruption == "image_norm":
        result["test"]["image_embedding"][0, 0] = 2
    elif corruption == "geo_wrong_scaler":
        result["test"]["geo_embedding"][0, 0] += 1
    elif corruption == "logits_nonfinite":
        result["test"]["raw_logits"][0, 0] = np.nan
    elif corruption == "metadata_fold":
        result["metadata"]["producer_fold"] = 3
    elif corruption == "checkpoint":
        (directory / "geo_only_head.safetensors").write_bytes(b"changed")
    elif corruption == "scaler":
        (directory / "geo_standardization.json").write_text('{"changed":true}')
    elif corruption == "raw_image":
        (extraction.test_source.TEST_IMAGE_ROOT / "one.jpg").write_bytes(b"changed")
    elif corruption == "source":
        frame = pd.read_parquet(path)
        frame.loc[0, "A00"] += 1
        frame.to_parquet(path, index=False)
    elif corruption == "metadata_fit":
        result["metadata"]["expert_fit"]["row_count"] = 9
    elif corruption == "parent_extraction":
        (parent_root / "producers/seed_1/validation/extraction.json").write_text("{}")
    elif corruption == "parent_features":
        (parent_root / "producers/seed_1/validation/features.npz").write_bytes(b"changed")
    elif corruption == "parent_declaration":
        (parent_root / "declaration.json").write_text("{}")
    elif corruption == "parent_lineage":
        result["metadata"]["parent_joint_validation"]["producer_manifest_file_sha256"] = "changed"
    else:
        result["metadata"]["training_seed"] = 2
    with pytest.raises(ValueError):
        extraction.validate_seed_payload(result, 1, context=context, source_path=path)


def test_labels_and_cached_image_features_cannot_change_predictions(replay):
    first, context, path, *_ = replay
    frame = pd.read_parquet(path)
    frame["label_id"] = -123
    frame["I0000"] = -456.
    frame.to_parquet(path, index=False)
    second = extraction.extract_seed(1, device="cpu", source_path=path)
    for name, values in first["test"].items():
        np.testing.assert_array_equal(values, second["test"][name])
    assert first["metadata"]["test_sources"]["source_file_sha256"] != second["metadata"]["test_sources"]["source_file_sha256"]
    assert first["metadata"]["test_sources"]["native_geo_float32_sha256"] == second["metadata"]["test_sources"]["native_geo_float32_sha256"]


def test_development_and_test_plot_overlap_is_rejected(source):
    context, path, *_ = source
    context.assignments.loc[0, "plot_idx"] = "t0"
    with pytest.raises(ValueError, match="overlap"):
        extraction.load_test_sources(context=context, source_path=path)

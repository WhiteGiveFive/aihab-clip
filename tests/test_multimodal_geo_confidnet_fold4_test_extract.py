from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from safetensors.torch import save_file
from torch import nn

from multimodal import geo_helpfulness_confidnet_fold4_test_extract as extraction
from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_protocol import build_locked_test_identity_manifest, sha256_file
from multimodal.models import MLPHead


@pytest.fixture
def source(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    image_root.mkdir()
    monkeypatch.setattr(extraction, "TEST_IMAGE_ROOT", image_root)
    files = ["four.jpg", "one.jpg", "three.jpg", "two.jpg"]
    for index, name in enumerate(files):
        (image_root / name).write_bytes(bytes([index]))
    frame = pd.DataFrame({
        "file": files, "plot_idx": ["t2", "t0", "t1", "t0"],
        "image_source": [str(image_root)] * 4,
        "label_id": [91, 92, 93, 94], "I0000": [999.] * 4,
        **{name: np.arange(4, dtype=np.float64) + column for column, name in enumerate(m2.GEO_COLUMNS)},
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
                "mode_inputs": {"image_only": {"input_dim": 4},
                                "geo_only": {"input_dim": 64},
                                "raw_concat": {"input_dim": 68}},
            },
        },
    )
    monkeypatch.setattr(m2, "load_frozen_context", lambda **kwargs: context)
    return context, source_path, frame, manifest, image_root


def test_sources_read_only_allowlisted_inputs_and_canonicalize_order(source, monkeypatch):
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
    assert tuple(rows.columns) == extraction.fold4.initial.IDENTITY_COLUMNS
    expected_geo = frame.set_index("file").loc[rows.file, list(m2.GEO_COLUMNS)].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(geo, expected_geo)
    assert geo.dtype == np.float32
    assert metadata["test_labels_read"] is False
    assert metadata["cached_image_embeddings_read"] is False
    assert len(metadata["raw_image_file_sha256"]) == 4
    assert context.config["development_universe"]["allowed_image_sources"] == ["/development/images"]


@pytest.mark.parametrize("corruption", ["identity", "source_root", "geo_nan", "plot_overlap", "row_overlap", "missing_image", "active_snapshot"])
def test_sources_reject_wrong_membership_or_inputs(source, corruption):
    context, path, frame, manifest, image_root = source
    if corruption == "identity":
        frame.loc[0, "plot_idx"] = "unexpected"
    elif corruption == "source_root":
        frame.loc[0, "image_source"] = "/development/images"
    elif corruption == "geo_nan":
        frame.loc[0, "A00"] = np.nan
    elif corruption == "plot_overlap":
        context.assignments.loc[0, "plot_idx"] = "t0"
    elif corruption == "row_overlap":
        context.assignments.loc[0, "row_uid"] = manifest["rows"][0]["row_uid"]
    elif corruption == "missing_image":
        (image_root / "one.jpg").unlink()
    else:
        registry = context.protocol_dir.parent / "registry"
        (registry / "active_snapshot.json").write_text('{"snapshot_id":"changed"}')
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        extraction.load_test_sources(context=context, source_path=path)


@pytest.fixture
def replay(source, monkeypatch):
    context, source_path, *_ = source
    spec, _, _ = extraction.fold4._partitions(context, 1)
    directory = context.output_root / spec.relative_directory
    directory.mkdir(parents=True)
    encoder = nn.Module()
    encoder.visual = nn.Linear(2, 4)
    save_file(encoder.visual.state_dict(), str(directory / "adapted_visual_tower.safetensors"))
    for mode in m2.MODES:
        head = MLPHead(context.config["experts"]["mode_inputs"][mode]["input_dim"], 256, 18, dropout=0.7)
        save_file(head.state_dict(), str(directory / f"{mode}_head.safetensors"))
    (directory / "geo_standardization.json").write_text("{}")
    calls = []

    def fake_images(model, rows, **kwargs):
        assert tuple(rows.columns) == extraction.fold4.initial.IDENTITY_COLUMNS
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert kwargs["config"]["development_universe"]["allowed_image_sources"] == [str(extraction.TEST_IMAGE_ROOT)]
        calls.append(rows.row_uid.tolist())
        return np.ones((len(rows), 4), dtype=np.float32)

    monkeypatch.setattr(m2, "validate_producer", lambda context, spec, **kwargs: {"valid": True, "producer_fold": spec.fold})
    monkeypatch.setattr(m2, "_load_fresh_encoder", lambda *args: (encoder, None, "transform", None, {}))
    monkeypatch.setattr(m2, "_extract_embeddings", fake_images)
    monkeypatch.setattr(m2, "_validate_scaler_file", lambda path: {"mean": np.zeros(64), "std": np.ones(64)})
    result = extraction.extract_seed(1, device="cpu", source_path=source_path)
    return result, context, source_path, directory, calls


def test_cleaned_inference_keeps_fold3_checkpoint_and_label_blind_contract(replay):
    result, context, path, _, calls = replay
    assert set(result) == {"test", "metadata"}
    assert len(calls) == 1
    assert "labels" not in result["test"]
    assert np.all(result["test"]["producer_fold"] == 3)
    for prefix in extraction.PREFIXES:
        assert result["test"][f"{prefix}_hidden"].shape == (4, 256)
        assert result["test"][f"{prefix}_hidden"].dtype == np.float32
        assert result["test"][f"{prefix}_logits"].shape == (4, 18)
        assert result["test"][f"{prefix}_logits"].dtype == np.float64
    assert result["metadata"]["same_expert_bundle_as_fold4_study"]
    assert result["metadata"]["expert_fit"]["row_count"] == 9
    assert result["metadata"]["confidence_router_fit"]["row_count"] == 3
    assert extraction.validate_seed_payload(result, 1, context=context, source_path=path)["valid"]
    json.dumps(result["metadata"])


@pytest.mark.parametrize("field", ["labels", "label_id_dense", "tcp", "I0000"])
def test_test_payload_forbids_labels_targets_and_cached_embeddings(replay, field):
    result, context, path, *_ = replay
    result["test"][field] = np.zeros(4)
    with pytest.raises(ValueError, match="allow-list"):
        extraction.validate_seed_payload(result, 1, context=context, source_path=path)


@pytest.mark.parametrize("corruption", ["identity", "producer_fold", "producer_dtype", "hidden_dtype", "logits_nonfinite", "metadata_fold", "checkpoint", "scaler", "raw_image", "source", "metadata_fit"])
def test_saved_replay_validation_rejects_corruption(replay, corruption):
    result, context, path, directory, _ = replay
    if corruption == "identity":
        result["test"]["plot_idx"][0] = "bad"
    elif corruption == "producer_fold":
        result["test"]["producer_fold"][0] = 2
    elif corruption == "producer_dtype":
        result["test"]["producer_fold"] = result["test"]["producer_fold"].astype(np.float32)
    elif corruption == "hidden_dtype":
        result["test"]["image_hidden"] = result["test"]["image_hidden"].astype(np.float64)
    elif corruption == "logits_nonfinite":
        result["test"]["raw_logits"][0, 0] = np.nan
    elif corruption == "metadata_fold":
        result["metadata"]["producer_fold"] = None
    elif corruption == "checkpoint":
        (directory / "geo_only_head.safetensors").write_bytes(b"changed")
    elif corruption == "scaler":
        (directory / "geo_standardization.json").write_text('{"changed": true}')
    elif corruption == "raw_image":
        (extraction.TEST_IMAGE_ROOT / "one.jpg").write_bytes(b"changed")
    elif corruption == "source":
        frame = pd.read_parquet(path)
        frame.loc[0, "A00"] += 1
        frame.to_parquet(path, index=False)
    else:
        result["metadata"]["confidence_router_fit"]["row_count"] = 4
    with pytest.raises(ValueError):
        extraction.validate_seed_payload(result, 1, context=context, source_path=path)


def test_source_labels_and_cached_image_values_cannot_change_prediction_arrays(replay):
    first, context, path, *_ = replay
    frame = pd.read_parquet(path)
    frame["label_id"] = -123
    frame["I0000"] = -456.0
    frame.to_parquet(path, index=False)
    second = extraction.extract_seed(1, device="cpu", source_path=path)
    for name, values in first["test"].items():
        np.testing.assert_array_equal(values, second["test"][name])
    assert first["metadata"]["test_sources"]["source_file_sha256"] != second["metadata"]["test_sources"]["source_file_sha256"]
    assert first["metadata"]["test_sources"]["native_geo_float32_sha256"] == second["metadata"]["test_sources"]["native_geo_float32_sha256"]

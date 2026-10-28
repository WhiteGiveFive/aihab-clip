"""Private-engine compatibility, resumability and extension bundle provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from multimodal import attention_fusion_training as original
from multimodal import attention_fusion_bypass_training as bypass
from multimodal.attention_fusion_bypass import READOUTS


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    value = yaml.safe_load((ROOT / "configs/multimodal_attention_fusion.yaml").read_text())
    value["study_id"] = "attention_fusion_cls_bypass_v1"
    value["initialization_seed_namespace"] = "attention_fusion_v1"
    value["training"]["batch_size"] = 7
    value["candidates"] = [
        {"id": candidate, "architecture": "self_attention", "geo": True, "layers": 2,
         "readout": readout, "cls_type_embedding": False,
         "expected_trainable_parameters": 524946 if candidate.endswith("adapted") else 770706}
        for candidate, readout in READOUTS.items()
    ]
    return value


def data(config, prefix, rows=13):
    rng = np.random.default_rng(431 if prefix == "train" else 432)
    image = rng.normal(size=(rows, config["data"]["image_dim"])).astype(np.float32)
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    return {"image_embedding": image,
            "geo_embedding": rng.normal(size=(rows, config["data"]["geo_dim"])),
            "label": rng.integers(0, config["data"]["num_classes"], rows, dtype=np.int64),
            "row_uid": np.array([f"{prefix}_{i:03d}" for i in range(rows)]),
            "plot_idx": np.array([f"{prefix}_plot_{i // 2:03d}" for i in range(rows)])}


def run(config, candidate, directory, train, validation=None, **kwargs):
    return bypass.train_run(config, candidate, "standardize", 0.001, 1, train, validation,
                            directory, verification=True, verification_epochs=3, cpu_num_threads=1,
                            input_provenance=kwargs.pop("input_provenance", {"source_role": "train"}), **kwargs)


def compare_saved_outputs(first, second):
    a = torch.load(first / "final.pt", weights_only=True)
    b = torch.load(second / "final.pt", weights_only=True)
    assert a["epoch"] == b["epoch"]
    assert all(torch.equal(value, b["model_state"][name]) for name, value in a["model_state"].items())
    assert json.loads((first / "history.json").read_text()) == json.loads((second / "history.json").read_text())
    for role in ("train", "validation"):
        if (first / f"{role}_predictions.npz").exists():
            with np.load(first / f"{role}_predictions.npz") as left, np.load(second / f"{role}_predictions.npz") as right:
                assert all(np.array_equal(left[key], right[key]) for key in left.files)


def test_private_engine_preserves_legacy_factory_hashes_training_and_replay(tmp_path):
    factory, source_provider = original.build_model, original._source_hashes
    source_hashes = source_provider()
    private = bypass._instantiate_engine(factory, source_provider)
    assert private is not original
    assert private.train_run.__globals__ is not original.train_run.__globals__
    assert original.build_model is factory and original._source_hashes is source_provider
    config = yaml.safe_load((ROOT / "configs/multimodal_attention_fusion.yaml").read_text())
    config["training"]["batch_size"] = 7
    train, validation = data(config, "train"), data(config, "validation", 8)
    directories = [tmp_path / "verification/original", tmp_path / "verification/private"]
    for engine, directory in zip((original, private), directories):
        engine.train_run(config, "attention_cls", "standardize", 0.001, 1, train, validation,
                         directory, verification=True, verification_epochs=2, cpu_num_threads=1,
                         input_provenance={"source_role": "train"})
    compare_saved_outputs(*directories)
    for directory in directories:
        legacy = original.load_inference_bundle(directory).predict(
            validation["image_embedding"], validation["geo_embedding"], batch_size=7)
        isolated = private.load_inference_bundle(directory).predict(
            validation["image_embedding"], validation["geo_embedding"], batch_size=7)
        assert np.array_equal(legacy["logits"], isolated["logits"])
    assert original._source_hashes() == source_hashes
    assert original.build_model is factory and original._source_hashes is source_provider


@pytest.mark.parametrize("candidate", READOUTS)
def test_bypass_resume_exact_predictions_bundle_replay_and_immutable_reuse(config, tmp_path, candidate):
    train, validation = data(config, "train"), data(config, "validation", 8)
    full, resumed = tmp_path / "verification/full", tmp_path / "verification/resumed"
    run(config, candidate, full, train, validation)
    result = run(config, candidate, resumed, train, validation, stop_after_epoch=1)
    assert result["status"] == "interrupted_at_epoch_boundary"
    torch.manual_seed(9723)
    run(config, candidate, resumed, train, validation, resume=True)
    compare_saved_outputs(full, resumed)
    spec = json.loads((full / "run_spec.json").read_text())
    assert spec["study_id"] == config["study_id"]
    assert spec["source_hashes"] == bypass._source_hashes()
    processor = json.loads((full / "geo_preprocessing.json").read_text())
    np.testing.assert_array_equal(processor["mean"], train["geo_embedding"].mean(axis=0))
    restored = bypass.load_inference_bundle(full)
    prediction = restored.predict(validation["image_embedding"], validation["geo_embedding"], batch_size=7)
    with np.load(full / "validation_predictions.npz") as saved:
        assert np.array_equal(prediction["logits"], saved["logits"])
    before = {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
              for path in full.iterdir()}
    result = run(config, candidate, full, train, validation, resume=True)
    assert result["status"] == "reused_complete"
    assert before == {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                      for path in full.iterdir()}
    with pytest.raises(original.TrainingError, match="source differs"):
        original.load_inference_bundle(full)


def test_portable_bypass_export_contains_full_implementation_and_replays(config, tmp_path):
    candidate = "attention_cls_skip_adapted"
    train = data(config, "train")
    encoder = tmp_path / "encoder.safetensors"
    encoder.write_bytes(b"synthetic frozen encoder; never loaded")
    architecture = tmp_path / "open_clip_config.json"
    architecture.write_text('{"synthetic": true}\n')
    provenance = {"source_role": "train", "encoder": {
        "path": str(encoder), "sha256": bypass.sha256_file(encoder)},
        "encoder_construction_assets": [{"path": str(architecture), "name": architecture.name,
                                         "sha256": bypass.sha256_file(architecture)}]}
    source, destination = tmp_path / "verification/source", tmp_path / "export"
    run(config, candidate, source, train, input_provenance=provenance)
    bypass.export_inference_bundle(source, destination)
    assert set(path.name for path in (destination / "implementation").iterdir()) == set(bypass.SOURCE_FILES)
    encoder.unlink()
    original_prediction = bypass.load_inference_bundle(source).predict(train["image_embedding"], train["geo_embedding"], 7)
    exported_prediction = bypass.load_inference_bundle(destination).predict(train["image_embedding"], train["geo_embedding"], 7)
    assert np.array_equal(original_prediction["logits"], exported_prediction["logits"])


def test_private_engine_rejects_an_unrecognized_historical_source(monkeypatch):
    monkeypatch.setattr(bypass, "PINNED_ENGINE_SHA256", "0" * 64)
    with pytest.raises(original.TrainingError, match="pinned version"):
        bypass._instantiate_engine(original.build_model, original._source_hashes)
    with pytest.raises(original.TrainingError, match="changed"):
        bypass._source_hashes()

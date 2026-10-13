"""Exercise interrupted optimization, immutable inference bundles and data boundaries."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from multimodal.attention_fusion_training import (
    TrainingError,
    classification_metrics,
    configuration_hash,
    export_inference_bundle,
    load_inference_bundle,
    train_run,
)


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / "configs/multimodal_attention_fusion.yaml").read_text())
    value["training"]["batch_size"] = 7
    return value


def dataset(config, prefix, rows, rng_seed):
    rng = np.random.default_rng(rng_seed)
    image = rng.normal(size=(rows, config["data"]["image_dim"])).astype(np.float32)
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    return {"image_embedding": image,
            "geo_embedding": rng.normal(size=(rows, config["data"]["geo_dim"])),
            "label": rng.integers(0, config["data"]["num_classes"], size=rows, dtype=np.int64),
            "row_uid": np.array([f"{prefix}_{i:03d}" for i in range(rows)]),
            "plot_idx": np.array([f"{prefix}_plot_{i // 2:03d}" for i in range(rows)])}


def run(config, directory, train, validation=None, **kwargs):
    provenance = kwargs.pop("input_provenance", {"source_role": "train"})
    return train_run(config, "attention_two_token", "standardize", 0.001, 1,
                     train, validation, directory, verification=True,
                     verification_epochs=3, cpu_num_threads=1,
                     input_provenance=provenance, **kwargs)


def tree_snapshot(directory):
    return {str(path.relative_to(directory)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


def test_epoch_boundary_resume_reproduces_dropout_shuffle_optimizer_and_metrics(config, tmp_path):
    train, validation = dataset(config, "train", 23, 10), dataset(config, "val", 9, 11)
    uninterrupted = tmp_path / "verification/full"
    interrupted = tmp_path / "verification/resumed"
    run(config, uninterrupted, train, validation)
    result = run(config, interrupted, train, validation, stop_after_epoch=1)
    assert result["status"] == "interrupted_at_epoch_boundary"
    assert not (interrupted / "manifest.json").exists()
    saved = torch.load(interrupted / "resume.pt", weights_only=False)
    assert saved["completed_epoch"] == 1
    assert set(saved["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda", "shuffle"}
    # Perturb all process RNGs and prove they do not affect the resumed trajectory.
    random.seed(845)
    np.random.seed(123)
    torch.manual_seed(999)
    run(config, interrupted, train, validation, resume=True)
    full = torch.load(uninterrupted / "final.pt", weights_only=True)
    resumed = torch.load(interrupted / "final.pt", weights_only=True)
    assert all(torch.equal(value, resumed["model_state"][name]) for name, value in full["model_state"].items())
    assert json.loads((uninterrupted / "history.json").read_text()) == json.loads((interrupted / "history.json").read_text())
    for role in ("train", "validation"):
        with np.load(uninterrupted / f"{role}_predictions.npz", allow_pickle=False) as a, \
                np.load(interrupted / f"{role}_predictions.npz", allow_pickle=False) as b:
            assert a.files == b.files
            assert all(np.array_equal(a[name], b[name]) for name in a.files)
    a = torch.load(uninterrupted / "resume.pt", weights_only=False)["optimizer_state"]
    b = torch.load(interrupted / "resume.pt", weights_only=False)["optimizer_state"]
    assert a["param_groups"] == b["param_groups"]
    for index, state in a["state"].items():
        assert all(torch.equal(value, b["state"][index][key]) for key, value in state.items())


def test_bundle_replay_preprocessing_and_completed_immutability(config, tmp_path):
    train, validation = dataset(config, "train", 19, 12), dataset(config, "val", 8, 13)
    validation["geo_embedding"] += 100
    directory = tmp_path / "verification/replay"
    run(config, directory, train, validation)
    processor = json.loads((directory / "geo_preprocessing.json").read_text())
    assert np.array_equal(processor["mean"], train["geo_embedding"].mean(axis=0))
    assert processor["fit_row_count"] == 19
    restored = load_inference_bundle(directory)
    replay = restored.predict(validation["image_embedding"], validation["geo_embedding"], batch_size=7)
    with np.load(directory / "validation_predictions.npz", allow_pickle=False) as saved:
        assert np.array_equal(replay["logits"], saved["logits"])
        assert np.array_equal(replay["pred_top3"], saved["pred_top3"])
    before = tree_snapshot(directory)
    with pytest.raises(TrainingError, match="immutable"):
        run(config, directory, train, validation)
    result = run(config, directory, train, validation, resume=True)
    assert result["status"] == "reused_complete"
    assert tree_snapshot(directory) == before
    original_manifest = (directory / "manifest.json").read_text()
    corrupt_manifest = json.loads(original_manifest)
    corrupt_manifest["class_map"] = list(reversed(corrupt_manifest["class_map"]))
    (directory / "manifest.json").write_text(json.dumps(corrupt_manifest))
    with pytest.raises(TrainingError, match="self-hash"):
        load_inference_bundle(directory)
    (directory / "manifest.json").write_text(original_manifest)
    (directory / "geo_preprocessing.json").write_text("{}\n")
    with pytest.raises(TrainingError, match="changed"):
        load_inference_bundle(directory)


@pytest.mark.parametrize("mutation", ["source", "data", "recipe", "provenance", "threads"])
def test_resume_rejects_any_changed_training_identity(config, tmp_path, monkeypatch, mutation):
    train = dataset(config, "train", 13, 14)
    directory = tmp_path / "verification/changed"
    run(config, directory, train, stop_after_epoch=1)
    before = tree_snapshot(directory)
    if mutation == "source":
        import multimodal.attention_fusion_training as training
        original = training._source_hashes
        monkeypatch.setattr(training, "_source_hashes", lambda: {**original(), "different.py": "a" * 64})
    elif mutation == "data":
        train["geo_embedding"][0, 0] += 1
    elif mutation == "recipe":
        config["training"]["weight_decay"] *= 2
    kwargs = {"resume": True}
    if mutation == "provenance":
        kwargs["input_provenance"] = {"source_role": "train", "different_input_binding": True}
    with pytest.raises(TrainingError, match="identity differs"):
        if mutation == "threads":
            train_run(config, "attention_two_token", "standardize", 0.001, 1, train, None,
                      directory, verification=True, verification_epochs=3,
                      input_provenance={"source_role": "train"}, cpu_num_threads=2, resume=True)
        else:
            run(config, directory, train, **kwargs)
    assert tree_snapshot(directory) == before


@pytest.mark.parametrize("violation", ["namespace", "test_role", "plot_overlap", "row_overlap", "test_flag", "bad_labels"])
def test_training_rejects_data_or_verification_boundary_violations(config, tmp_path, violation):
    train, validation = dataset(config, "train", 13, 15), dataset(config, "val", 8, 16)
    directory = tmp_path / "verification/bad"
    provenance = {"source_role": "train"}
    if violation == "namespace":
        directory = tmp_path / "development/bad"
    elif violation == "test_role":
        provenance["source_role"] = "test"
    elif violation == "plot_overlap":
        validation["plot_idx"] = validation["plot_idx"].astype(object)
        validation["plot_idx"][0] = train["plot_idx"][0]
    elif violation == "row_overlap":
        validation["row_uid"] = validation["row_uid"].astype(object)
        validation["row_uid"][0] = train["row_uid"][0]
    elif violation == "test_flag":
        config["data"]["allow_test_features_or_labels"] = True
    else:
        train["label"][0] = 18
    with pytest.raises(TrainingError):
        run(config, directory, train, validation, input_provenance=provenance)
    assert not directory.exists()


def test_metrics_keep_all_classes_resolve_ties_and_mark_unsupported_rows():
    logits = np.zeros((4, 18), dtype=np.float32)
    labels = np.array([0, 1, 2, 3])
    metrics, predictions = classification_metrics(labels, logits)
    assert predictions["pred_top3"].tolist() == [[0, 1, 2]] * 4
    assert metrics["acc1"] == 0.25 and metrics["acc3"] == 0.75
    assert metrics["weighted_f1"] == pytest.approx(0.1)
    assert metrics["mcc"] == 0
    assert np.array(metrics["confusion_matrix"]).shape == (18, 18)
    assert metrics["class_recall"][17] is None
    assert metrics["confusion_row_normalized"][17] == [None] * 18


def test_final_refit_is_fresh_and_cannot_reuse_validation_for_scoring(config, tmp_path):
    train = dataset(config, "train", 17, 17)
    output = tmp_path / "verification/all_development"
    result = run(config, output, train, fit_role="all_development")
    assert result["status"] == "complete"
    assert set(result["metrics"]) == {"train"}
    assert not (output / "validation_predictions.npz").exists()
    with pytest.raises(TrainingError, match="no independent validation"):
        run(config, tmp_path / "verification/invalid_refit", train,
            dataset(config, "val", 8, 18), fit_role="all_development")


def test_portable_export_copies_encoder_and_replays_without_original_encoder(config, tmp_path):
    train = dataset(config, "train", 13, 19)
    encoder = tmp_path / "original.safetensors"
    encoder.write_bytes(b"synthetic encoder bytes are copied, never executed")
    architecture = tmp_path / "open_clip_config.json"
    architecture.write_text('{"model_cfg": {"synthetic": true}}\n')
    provenance = {"source_role": "train", "encoder": {
        "path": str(encoder), "sha256": hashlib.sha256(encoder.read_bytes()).hexdigest()},
        "encoder_construction_assets": [{"path": str(architecture), "name": architecture.name,
                                         "sha256": hashlib.sha256(architecture.read_bytes()).hexdigest()}]}
    source, destination = tmp_path / "verification/source", tmp_path / "export"
    run(config, source, train, input_provenance=provenance)
    manifest = export_inference_bundle(source, destination)
    assert manifest["portable_export"]
    assert (destination / "encoder.safetensors").read_bytes() == encoder.read_bytes()
    assert (destination / "encoder_assets/open_clip_config.json").read_bytes() == architecture.read_bytes()
    encoder.unlink()
    original = load_inference_bundle(source).predict(train["image_embedding"], train["geo_embedding"], batch_size=7)
    restored = load_inference_bundle(destination).predict(train["image_embedding"], train["geo_embedding"], batch_size=7)
    assert np.array_equal(original["logits"], restored["logits"])
    assert restored["pred_top3"].shape == (13, 3)
    with pytest.raises(TrainingError, match="already exist"):
        export_inference_bundle(source, destination)


def test_real_fit_requires_frozen_protocol_binding_and_full_partition(config, tmp_path):
    train, validation = dataset(config, "train", 13, 20), dataset(config, "val", 8, 21)
    with pytest.raises(TrainingError, match="frozen protocol"):
        train_run(config, "image_only", "not_applicable", 0.001, 1, train, validation,
                  tmp_path / "development/invalid", input_provenance={})
    with pytest.raises(TrainingError, match="rows/plots"):
        train_run(config, "image_only", "not_applicable", 0.001, 1, train, validation,
                  tmp_path / "development/invalid", input_provenance={
                      "protocol_config_sha256": configuration_hash(config), "train_role": "train",
                      "validation_role": "validation"})


@pytest.mark.parametrize("key,value", [
    ("shuffle", False), ("drop_last", True), ("num_workers", 1),
    ("label_smoothing", 0.1), ("loss", "weighted_cross_entropy"),
])
def test_unsupported_training_settings_are_rejected_not_silently_ignored(config, tmp_path, key, value):
    config["training"][key] = value
    with pytest.raises(TrainingError, match="Unsupported training setting"):
        run(config, tmp_path / "verification/unsupported", dataset(config, "train", 9, 22))


def test_fresh_cuda_runtime_initializes_allocator_before_resetting_peak_stats(config, monkeypatch):
    from multimodal.attention_fusion_training import _configure_runtime

    events = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: events.append(("seed", seed)))
    monkeypatch.setattr(torch.cuda, "init", lambda: events.append(("init", None)))

    def reset(device):
        assert ("init", None) in events
        events.append(("reset", str(device)))

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    _configure_runtime(config, 1, torch.device("cuda:0"), 1)
    assert events.index(("init", None)) < events.index(("reset", "cuda:0"))

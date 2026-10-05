from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from multimodal import geo_helpfulness_joint_tcp_extract as extraction
from multimodal import geo_helpfulness_oof as m2
from multimodal.models import MLPHead


def _context(root: Path) -> SimpleNamespace:
    count = 16
    assignments = pd.DataFrame({
        "row_uid": [f"u{i:03d}" for i in range(count)],
        "file": [f"image/{i}.jpg" for i in range(count)],
        "file_lower": [f"image/{i}.jpg" for i in range(count)],
        "plot_idx": [f"p{i}" for i in range(count)],
        "image_source": ["/allowed/images"] * count,
        "label_id_dense": np.arange(count, dtype=np.int64),
        "development_role": ["train"] * 12 + ["validation"] * 4,
        "train_oof_fold": pd.array([i % 4 for i in range(12)] + [pd.NA] * 4, dtype="Int8"),
    })
    return SimpleNamespace(
        output_root=root, assignments=assignments,
        parent_hashes={"synthetic": "parent"},
        config={"protocol_id": "synthetic", "experts": {
            "head": {"hidden_dim": 256, "dropout": 0.7},
            "mode_inputs": {"image_only": {"input_dim": 1152},
                            "geo_only": {"input_dim": 64},
                            "raw_concat": {"input_dim": 1216}},
        }},
    )


def _payload(rows: pd.DataFrame, fold: int | None) -> dict[str, np.ndarray]:
    image = np.zeros((len(rows), extraction.IMAGE_DIM), dtype=np.float32)
    image[:, 0] = 1
    payload = {
        "row_uid": rows["row_uid"].to_numpy(dtype=str),
        "plot_idx": rows["plot_idx"].to_numpy(dtype=str),
        "producer_fold": np.full(len(rows), fold if fold is not None else -1, dtype=np.int64),
        "image_embedding": image,
        "geo_embedding": np.ones((len(rows), extraction.GEO_DIM), dtype=np.float32),
    }
    for prefix in extraction.PREFIXES:
        payload[f"{prefix}_logits"] = np.tile(np.arange(18, dtype=np.float64), (len(rows), 1))
    if fold is not None:
        payload["labels"] = rows["label_id_dense"].to_numpy(dtype=np.int64)
    return payload


def _reference(rows, payload, fold, seed=1):
    reference = rows.loc[:, ["row_uid", "file", "file_lower", "plot_idx"]].copy()
    reference["training_seed"] = seed
    if fold is not None:
        reference["train_oof_fold"] = fold
    for prefix in extraction.PREFIXES:
        reference[f"{prefix}_logits"] = list(payload[f"{prefix}_logits"].copy())
        reference[f"{prefix}_pred"] = payload[f"{prefix}_logits"].argmax(axis=1)
    return reference


@pytest.fixture
def cached_producers(tmp_path, monkeypatch):
    context = _context(tmp_path)
    producers, references = {}, {}
    for spec in m2._producer_specs(1):
        _, rows = m2._partitions_for_spec(context.assignments, spec)
        payload = _payload(rows, spec.fold)
        producers[spec.fold] = {
            "payload": payload,
            "metadata": {"training_seed": 1, "producer_fold": spec.fold},
        }
        references[context.output_root / spec.relative_directory / spec.output_filename] = (
            _reference(rows, payload, spec.fold)
        )
    monkeypatch.setattr(m2, "load_frozen_context", lambda **kwargs: context)
    monkeypatch.setattr(m2, "read_output_parquet",
                        lambda path, **kwargs: pa.Table.from_pandas(references[path]))
    return context, producers, references


def test_combination_canonical_rows_and_exact_oof_membership(cached_producers):
    context, producers, _ = cached_producers
    train = extraction.combine_producers(producers, "train")
    validation = extraction.combine_producers(producers, "validation")
    assert train["row_uid"].tolist() == context.assignments.iloc[:12]["row_uid"].tolist()
    assert train["producer_fold"].tolist() == [i % 4 for i in range(12)]
    assert train["labels"].tolist() == list(range(12))
    assert train["image_embedding"].shape == (12, 1152)
    assert train["geo_embedding"].shape == (12, 64)
    assert validation["producer_fold"].tolist() == [-1] * 4
    assert "labels" not in validation
    assert set(train["plot_idx"]).isdisjoint(validation["plot_idx"])


def test_combination_accepts_only_requested_split_producers(cached_producers):
    _, producers, _ = cached_producers
    train = extraction.combine_producers({k: v for k, v in producers.items() if k is not None}, "train")
    validation = extraction.combine_producers({None: producers[None]}, "validation")
    assert len(train["row_uid"]) == 12
    assert len(validation["row_uid"]) == 4
    with pytest.raises(extraction.JointTCPExtractionError, match="all requested"):
        extraction.combine_producers({0: producers[0]}, "train")


@pytest.mark.parametrize("mutation, message", [
    (lambda p: p.update(labels=np.zeros(4, dtype=np.int64)), "allow-list"),
    (lambda p: p.update(image_embedding=np.zeros((4, 1151), dtype=np.float32)), "shape or dtype"),
    (lambda p: p.update(geo_embedding=p["geo_embedding"].astype(np.float64)), "shape or dtype"),
    (lambda p: p.update(producer_fold=np.zeros(4, dtype=np.int64)), "match the split"),
    (lambda p: p.update(image_embedding=p["image_embedding"] * 2), "L2-normalized"),
    (lambda p: p["geo_embedding"].__setitem__((0, 0), np.nan), "nonfinite"),
    (lambda p: p["row_uid"].__setitem__(0, p["row_uid"][1]), "duplicate"),
])
def test_payload_rejects_leakage_wrong_dimensions_and_corruption(cached_producers, mutation, message):
    _, producers, _ = cached_producers
    payload = copy.deepcopy(producers[None]["payload"])
    mutation(payload)
    with pytest.raises(extraction.JointTCPExtractionError, match=message):
        extraction.validate_split_payload(payload, "validation")


@pytest.mark.parametrize("change, message", [
    (lambda p: p[0]["metadata"].update(training_seed=2), "mixed training seeds"),
    (lambda p: p[0]["metadata"].update(producer_fold=1), "fold metadata"),
    (lambda p: p[0]["payload"]["labels"].__setitem__(0, 17), "labels differ"),
    (lambda p: p[0]["payload"]["image_logits"].__setitem__((0, 0), 1e-8), "authoritative"),
])
def test_cached_provenance_labels_and_even_tiny_logit_changes_rejected(cached_producers, change, message):
    _, producers, _ = cached_producers
    changed = copy.deepcopy(producers)
    change(changed)
    with pytest.raises(extraction.JointTCPExtractionError, match=message):
        extraction.combine_producers(changed, "train")


def test_one_plot_cannot_cross_producer_folds(cached_producers):
    _, producers, _ = cached_producers
    payload = extraction.combine_producers(producers, "train")
    payload["plot_idx"][1] = payload["plot_idx"][0]
    with pytest.raises(extraction.JointTCPExtractionError, match="multiple producer folds"):
        extraction.validate_split_payload(payload, "train")


@pytest.mark.parametrize("fold", [True, 4, -1, "1"])
def test_invalid_fold_rejected_before_loading_sources(monkeypatch, fold):
    monkeypatch.setattr(m2, "load_frozen_context", lambda **kwargs: pytest.fail("must not load"))
    with pytest.raises(extraction.JointTCPExtractionError, match="fold must"):
        extraction.extract_producer(1, fold)


@pytest.fixture
def replay_bundle(tmp_path, monkeypatch, request):
    context = _context(tmp_path)
    fold = getattr(request, "param", None)
    spec = extraction._fold_spec(1, fold)
    directory = tmp_path / spec.relative_directory
    directory.mkdir(parents=True)
    encoder = nn.Module()
    encoder.visual = nn.Linear(2, 2)
    save_file(encoder.visual.state_dict(), str(directory / "adapted_visual_tower.safetensors"))
    heads = {}
    for mode in m2.MODES:
        heads[mode] = MLPHead(
            context.config["experts"]["mode_inputs"][mode]["input_dim"], 256, 18, dropout=0.7
        )
        save_file(heads[mode].state_dict(), str(directory / f"{mode}_head.safetensors"))
    (directory / "geo_standardization.json").write_text("{}")
    (directory / spec.output_filename).write_bytes(b"synthetic reference")
    fit_rows, rows = m2._partitions_for_spec(context.assignments, spec)
    expected = _payload(rows, fold)
    for mode, prefix in m2.MODE_TO_PREFIX.items():
        inputs = m2._head_input(mode, expected["image_embedding"], expected["geo_embedding"])
        expected[f"{prefix}_logits"] = m2._predict_head(heads[mode], inputs, torch.device("cpu"))
    reference = _reference(rows, expected, fold)

    def fake_images(model, prediction_rows, **kwargs):
        assert tuple(prediction_rows.columns) == extraction.initial.IDENTITY_COLUMNS
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        assert set(prediction_rows["plot_idx"]).isdisjoint(fit_rows["plot_idx"])
        return expected["image_embedding"].copy()

    def fake_geo(_context, prediction_rows):
        assert tuple(prediction_rows.columns) == extraction.initial.IDENTITY_COLUMNS
        return expected["geo_embedding"].copy()

    monkeypatch.setattr(m2, "load_frozen_context", lambda **kwargs: context)
    monkeypatch.setattr(m2, "validate_producer", lambda *args, **kwargs: {"valid": True})
    monkeypatch.setattr(m2, "_load_fresh_encoder", lambda *args: (
        encoder, None, "synthetic_transform", None, {"synthetic": True}
    ))
    monkeypatch.setattr(m2, "_extract_embeddings", fake_images)
    monkeypatch.setattr(m2, "_geo_source_projection", fake_geo)
    monkeypatch.setattr(m2, "_validate_scaler_file", lambda path: {
        "mean": np.zeros(64), "std": np.ones(64)
    })
    monkeypatch.setattr(m2, "read_output_parquet", lambda *args, **kwargs: pa.Table.from_pandas(reference))
    return context, expected, reference


@pytest.mark.parametrize("replay_bundle", [2], indirect=True)
def test_training_replay_uses_heldout_fold_labels(replay_bundle):
    context, expected, _ = replay_bundle
    result = extraction.extract_producer(1, 2)
    for key in expected:
        np.testing.assert_array_equal(result["payload"][key], expected[key])
    assert result["payload"]["labels"].tolist() == [2, 6, 10]
    assert result["payload"]["producer_fold"].tolist() == [2, 2, 2]
    assert result["metadata"]["expert_fit"]["row_count"] == 9
    assert result["metadata"]["prediction"]["row_count"] == 3


def test_replay_is_frozen_label_blind_and_retains_authoritative_logits(replay_bundle, monkeypatch):
    context, expected, reference = replay_bundle
    context.assignments["label_id_dense"] = context.assignments["label_id_dense"].astype(object)
    context.assignments.loc[context.assignments.development_role.eq("validation"), "label_id_dense"] = object()
    original_predict = m2._predict_head
    monkeypatch.setattr(m2, "_predict_head", lambda *args: original_predict(*args) + 1e-8)
    result = extraction.extract_producer(1, None)
    for key in expected:
        np.testing.assert_array_equal(result["payload"][key], expected[key])
    assert result["metadata"]["device"] == "cpu"
    assert result["metadata"]["reproduction"]["argmax_exact"]
    assert result["metadata"]["reproduction"]["maximum_absolute_logit_error"]["image"] > 0
    assert result["metadata"]["expert_fit_prediction_plots_disjoint"]
    assert result["metadata"]["logits_source"] == "authoritative_saved_M2_outputs"
    assert result["metadata"]["locked_test_accessed"] is False
    assert "labels" not in result["payload"]
    json.dumps(result["metadata"])


def test_replay_rejects_logit_mismatch(replay_bundle, monkeypatch):
    original_predict = m2._predict_head
    monkeypatch.setattr(m2, "_predict_head", lambda *args: original_predict(*args) + 0.01)
    with pytest.raises(extraction.JointTCPExtractionError, match="logits do not reproduce"):
        extraction.extract_producer(1, None)


def test_replay_rejects_argmax_change_even_within_logit_tolerance(replay_bundle, monkeypatch):
    _, expected, reference = replay_bundle
    expected["image_logits"][:] = 0
    reference["image_logits"] = list(expected["image_logits"].copy())
    reference["image_pred"] = 0

    def wrong_argmax(*args):
        logits = expected["image_logits"].copy()
        logits[:, 1] = 1e-8
        return logits

    monkeypatch.setattr(m2, "_predict_head", wrong_argmax)
    with pytest.raises(extraction.JointTCPExtractionError, match="argmax"):
        extraction.extract_producer(1, None)


def test_reference_rejects_labels_and_wrong_seed(cached_producers):
    context, producers, references = cached_producers
    path = next(iter(references))
    references[path]["label_id_dense"] = 0
    with pytest.raises(extraction.JointTCPExtractionError, match="label-blind"):
        extraction.combine_producers(producers, "train")
    del references[path]["label_id_dense"]
    references[path]["training_seed"] = 2
    with pytest.raises(extraction.JointTCPExtractionError, match="seed mismatch"):
        extraction.combine_producers(producers, "train")

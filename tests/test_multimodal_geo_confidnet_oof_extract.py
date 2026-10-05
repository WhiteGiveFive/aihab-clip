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

from multimodal import geo_helpfulness_confidnet_oof_extract as extraction
from multimodal import geo_helpfulness_oof as m2
from multimodal.models import MLPHead


def _assignments() -> pd.DataFrame:
    count = 16
    return pd.DataFrame({
        "row_uid": [f"u{i:03d}" for i in range(count)],
        "file": [f"image/{i}.jpg" for i in range(count)],
        "file_lower": [f"image/{i}.jpg" for i in range(count)],
        "plot_idx": [f"p{i}" for i in range(count)],
        "image_source": ["/allowed/images"] * count,
        "label_id_dense": np.arange(count, dtype=np.int64),
        "development_role": ["train"] * 12 + ["validation"] * 4,
        "train_oof_fold": pd.array([i % 4 for i in range(12)] + [pd.NA] * 4, dtype="Int8"),
    })


def _context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=root,
        assignments=_assignments(),
        parent_hashes={"synthetic": "parent"},
        config={"protocol_id": "synthetic", "experts": {
            "head": {"hidden_dim": 256, "dropout": 0.7},
            "mode_inputs": {"image_only": {"input_dim": 4},
                            "geo_only": {"input_dim": 64},
                            "raw_concat": {"input_dim": 68}},
        }},
    )


def _payload(rows: pd.DataFrame, fold: int | None) -> dict[str, np.ndarray]:
    payload = {
        "row_uid": rows["row_uid"].to_numpy(dtype=str),
        "plot_idx": rows["plot_idx"].to_numpy(dtype=str),
        "producer_fold": np.full(len(rows), fold if fold is not None else -1, dtype=np.int64),
    }
    for prefix in extraction.PREFIXES:
        payload[f"{prefix}_hidden"] = np.ones((len(rows), 256), dtype=np.float32)
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
    producers = {}
    references = {}
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
    monkeypatch.setattr(
        m2, "read_output_parquet",
        lambda path, **kwargs: pa.Table.from_pandas(references[path]),
    )
    return context, producers, references


def test_combination_restores_canonical_order_and_exact_oof_provenance(cached_producers):
    context, producers, _ = cached_producers
    result = extraction.combine_producers(1, producers)
    assert result["train"]["row_uid"].tolist() == context.assignments.iloc[:12]["row_uid"].tolist()
    assert result["train"]["producer_fold"].tolist() == [i % 4 for i in range(12)]
    assert result["train"]["labels"].tolist() == list(range(12))
    assert result["validation"]["producer_fold"].tolist() == [-1] * 4
    assert "labels" not in result["validation"]
    assert result["metadata"]["all_oof_rows_exactly_once"]
    assert result["metadata"]["splits"]["train"]["row_count"] == 12
    json.dumps(result["metadata"])


@pytest.mark.parametrize("corruption", ["missing", "metadata", "labels", "row", "plot", "fold", "logits"])
def test_combination_rejects_wrong_cached_producer(cached_producers, corruption):
    _, producers, _ = cached_producers
    bad = copy.deepcopy(producers)
    if corruption == "missing":
        del bad[2]
    elif corruption == "metadata":
        bad[2]["metadata"]["training_seed"] = 2
    elif corruption == "labels":
        bad[2]["payload"]["labels"][0] += 1
    elif corruption == "row":
        bad[2]["payload"]["row_uid"][0] = "u000"
    elif corruption == "plot":
        bad[2]["payload"]["plot_idx"][0] = "p0"
    elif corruption == "fold":
        bad[2]["payload"]["producer_fold"][:] = 1
    else:
        bad[2]["payload"]["raw_logits"][0, 0] += 0.01
    with pytest.raises(extraction.ConfidNetOOFExtractionError):
        extraction.combine_producers(1, bad)


@pytest.mark.parametrize("forbidden", ["labels", "label_id_dense", "tcp"])
def test_validation_payload_disallows_supervision(cached_producers, forbidden):
    _, producers, _ = cached_producers
    producers[None]["payload"][forbidden] = np.zeros(4, dtype=np.int64)
    with pytest.raises(extraction.ConfidNetOOFExtractionError, match="allow-list"):
        extraction.combine_producers(1, producers)


def test_plot_cannot_cross_folds_and_fold_dtype_is_strict(cached_producers):
    _, producers, _ = cached_producers
    result = extraction.combine_producers(1, producers)
    train = result["train"]
    train["plot_idx"][1] = train["plot_idx"][0]
    with pytest.raises(extraction.ConfidNetOOFExtractionError, match="multiple producer folds"):
        extraction.validate_split_payload(train, "train")
    train["producer_fold"] = train["producer_fold"].astype(np.float32)
    with pytest.raises(extraction.ConfidNetOOFExtractionError, match="int64"):
        extraction.validate_split_payload(train, "train")


@pytest.mark.parametrize("fold", [True, 4, -1, "1"])
def test_invalid_fold_rejected_before_loading_parents(fold, monkeypatch):
    monkeypatch.setattr(m2, "load_frozen_context", lambda **kwargs: pytest.fail("should not load"))
    with pytest.raises(extraction.ConfidNetOOFExtractionError, match="fold must"):
        extraction.extract_producer(1, fold, device="cpu")


def test_replay_rejects_argmax_change_inside_logit_tolerance(cached_producers):
    context, producers, references = cached_producers
    spec = m2._producer_specs(1)[0]
    payload = producers[0]["payload"]
    payload["image_logits"][:] = 0.0
    path = context.output_root / spec.relative_directory / spec.output_filename
    reference = references[path]
    reference["image_logits"] = list(payload["image_logits"].copy())
    reference["image_pred"] = 0
    payload["image_logits"][0, 1] = 1e-8
    with pytest.raises(extraction.ConfidNetOOFExtractionError, match="argmax"):
        extraction.combine_producers(1, producers)


@pytest.mark.parametrize("fold", [2, None])
def test_replay_uses_only_heldout_rows_and_frozen_saved_bundle(tmp_path, monkeypatch, fold):
    context = _context(tmp_path)
    spec = extraction._fold_spec(1, fold)
    directory = tmp_path / spec.relative_directory
    directory.mkdir(parents=True)
    encoder = nn.Module()
    encoder.visual = nn.Linear(2, 4)
    save_file(encoder.visual.state_dict(), str(directory / "adapted_visual_tower.safetensors"))
    heads = {}
    for mode in m2.MODES:
        heads[mode] = MLPHead(
            context.config["experts"]["mode_inputs"][mode]["input_dim"], 256, 18, dropout=0.7
        )
        save_file(heads[mode].state_dict(), str(directory / f"{mode}_head.safetensors"))
    (directory / "geo_standardization.json").write_text("{}")
    (directory / spec.output_filename).write_bytes(b"synthetic")
    fit_rows, rows = m2._partitions_for_spec(context.assignments, spec)
    calls = []

    def fake_images(model, prediction_rows, **kwargs):
        assert tuple(prediction_rows.columns) == extraction.initial.IDENTITY_COLUMNS
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        assert set(prediction_rows["plot_idx"]).isdisjoint(fit_rows["plot_idx"])
        calls.append(prediction_rows["row_uid"].tolist())
        return np.ones((len(prediction_rows), 4), dtype=np.float32)

    def fake_geo(_context, prediction_rows):
        assert tuple(prediction_rows.columns) == extraction.initial.IDENTITY_COLUMNS
        return np.ones((len(prediction_rows), 64), dtype=np.float32)

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
    expected = _payload(rows, fold)
    for mode, prefix in m2.MODE_TO_PREFIX.items():
        width = context.config["experts"]["mode_inputs"][mode]["input_dim"]
        inputs = np.ones((len(rows), width), dtype=np.float32)
        expected[f"{prefix}_hidden"], expected[f"{prefix}_logits"] = (
            extraction.initial._head_hidden_and_logits(heads[mode], inputs, torch.device("cpu"))
        )
    reference = _reference(rows, expected, fold)
    monkeypatch.setattr(m2, "read_output_parquet", lambda *args, **kwargs: pa.Table.from_pandas(reference))
    result = extraction.extract_producer(1, fold, device="cpu")
    assert calls == [rows["row_uid"].tolist()]
    for key in expected:
        np.testing.assert_array_equal(result["payload"][key], expected[key])
    assert result["metadata"]["reproduction"]["argmax_exact"]
    assert result["metadata"]["expert_fit_prediction_plots_disjoint"]
    assert result["metadata"]["locked_test_accessed"] is False
    assert ("labels" in result["payload"]) == (fold is not None)
    json.dumps(result["metadata"])

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

from multimodal import geo_helpfulness_confidnet_fold4_extract as extraction
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
        output_root=root, assignments=assignments, parent_hashes={"synthetic": "parent"},
        config={"protocol_id": "synthetic", "experts": {
            "head": {"hidden_dim": 256, "dropout": 0.7},
            "mode_inputs": {"image_only": {"input_dim": 4},
                            "geo_only": {"input_dim": 64},
                            "raw_concat": {"input_dim": 68}},
        }},
    )


@pytest.fixture
def replay(tmp_path, monkeypatch):
    context = _context(tmp_path)
    spec, fit_rows, partitions = extraction._partitions(context, 1)
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
    (directory / spec.output_filename).write_bytes(b"synthetic heldout reference")
    calls = []

    def fake_images(model, rows, **kwargs):
        assert tuple(rows.columns) == extraction.initial.IDENTITY_COLUMNS
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        assert set(rows["plot_idx"]).isdisjoint(fit_rows["plot_idx"])
        calls.append((id(model), rows["row_uid"].tolist()))
        return np.ones((len(rows), 4), dtype=np.float32)

    def fake_geo(_context, rows):
        assert tuple(rows.columns) == extraction.initial.IDENTITY_COLUMNS
        return np.ones((len(rows), 64), dtype=np.float32)

    monkeypatch.setattr(m2, "load_frozen_context", lambda **kwargs: context)
    monkeypatch.setattr(m2, "validate_producer", lambda context, spec, **kwargs: {
        "valid": True, "producer_fold": spec.fold,
    })
    monkeypatch.setattr(m2, "_load_fresh_encoder", lambda *args: (
        encoder, None, "synthetic_transform", None, {"synthetic": True}
    ))
    monkeypatch.setattr(m2, "_extract_embeddings", fake_images)
    monkeypatch.setattr(m2, "_geo_source_projection", fake_geo)
    monkeypatch.setattr(m2, "_validate_scaler_file", lambda path: {
        "mean": np.zeros(64), "std": np.ones(64)
    })
    rows = partitions["train"]
    reference = rows.loc[:, ["row_uid", "file", "file_lower", "plot_idx"]].copy()
    reference["training_seed"] = 1
    reference["train_oof_fold"] = 3
    expected = {}
    for mode, prefix in m2.MODE_TO_PREFIX.items():
        width = context.config["experts"]["mode_inputs"][mode]["input_dim"]
        hidden, logits = extraction.initial._head_hidden_and_logits(
            heads[mode], np.ones((len(rows), width), dtype=np.float32), torch.device("cpu")
        )
        expected[f"{prefix}_hidden"] = hidden
        expected[f"{prefix}_logits"] = logits
        reference[f"{prefix}_logits"] = list(logits)
        reference[f"{prefix}_pred"] = logits.argmax(axis=1)

    def read_reference(path, **kwargs):
        assert path == directory / spec.output_filename
        assert kwargs["include_fold"] is True
        return pa.Table.from_pandas(reference)

    monkeypatch.setattr(m2, "read_output_parquet", read_reference)
    result = extraction.extract_seed(1, device="cpu")
    return result, context, directory, calls, expected


def test_replay_uses_same_expert_for_heldout_fold4_and_validation(replay):
    result, context, _, calls, expected = replay
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert calls[0][1] == ["u003", "u007", "u011"]
    assert calls[1][1] == ["u012", "u013", "u014", "u015"]
    assert result["train"]["labels"].tolist() == [3, 7, 11]
    assert "labels" not in result["validation"]
    for split in ("train", "validation"):
        assert np.all(result[split]["producer_fold"] == 3)
        for key in expected:
            np.testing.assert_allclose(
                result[split][key], np.tile(expected[key][:1], (len(result[split]["row_uid"]), 1)),
                rtol=1e-6, atol=1e-6,
            )
    assert result["metadata"]["same_expert_bundle_for_both_splits"]
    assert result["metadata"]["producer"]["producer_fold"] == 3
    assert result["metadata"]["expert_fit"]["row_count"] == 9
    assert result["metadata"]["verification"]["heldout_reproduction"]["argmax_exact"]
    assert extraction.validate_seed_payload(result, 1, context=context)["valid"]
    json.dumps(result["metadata"])


@pytest.mark.parametrize("field", ["labels", "label_id_dense", "tcp"])
def test_validation_payload_disallows_supervision(replay, field):
    result, context, *_ = replay
    result["validation"][field] = np.zeros(4, dtype=np.int64)
    with pytest.raises(extraction.ConfidNetFold4ExtractionError, match="allow-list"):
        extraction.validate_seed_payload(result, 1, context=context)


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("wrong_fold", [-1, 0, 2])
def test_both_splits_reject_other_or_fulltrain_producer(replay, split, wrong_fold):
    result, context, *_ = replay
    result[split]["producer_fold"][:] = wrong_fold
    with pytest.raises(extraction.ConfidNetFold4ExtractionError, match="producer fold 3"):
        extraction.validate_seed_payload(result, 1, context=context)


@pytest.mark.parametrize("corruption", ["train_identity", "validation_identity", "labels", "logits",
                                         "metadata_fold", "checkpoint", "scaler"])
def test_replay_validation_rejects_source_or_payload_corruption(replay, corruption):
    result, context, directory, *_ = replay
    if corruption == "train_identity":
        result["train"]["row_uid"][0] = "u000"
    elif corruption == "validation_identity":
        result["validation"]["plot_idx"][0] = "p0"
    elif corruption == "labels":
        result["train"]["labels"][0] = 0
    elif corruption == "logits":
        result["train"]["raw_logits"][0, 0] += 1.0
    elif corruption == "metadata_fold":
        result["metadata"]["producer_fold"] = None
    elif corruption == "checkpoint":
        (directory / "geo_only_head.safetensors").write_bytes(b"wrong producer")
    else:
        (directory / "geo_standardization.json").write_text('{"wrong": true}')
    with pytest.raises(extraction.ConfidNetFold4ExtractionError):
        extraction.validate_seed_payload(result, 1, context=context)


def test_validation_label_values_do_not_enter_features_or_supervision(replay):
    result, context, *_ = replay
    context.assignments.loc[context.assignments["development_role"] == "validation", "label_id_dense"] = 0
    assert extraction.validate_seed_payload(result, 1, context=context)["valid"]
    second = extraction.extract_seed(1, device="cpu")
    for split in ("train", "validation"):
        for name, values in result[split].items():
            np.testing.assert_array_equal(values, second[split][name])


def test_plot_overlap_between_expert_fit_and_validation_rejected(tmp_path):
    context = _context(tmp_path)
    context.assignments.loc[12, "plot_idx"] = "p0"
    with pytest.raises(ValueError, match="plot overlap"):
        extraction._partitions(context, 1)


def test_fold_provenance_requires_int64(replay):
    result, *_ = replay
    result = copy.deepcopy(result)
    result["validation"]["producer_fold"] = result["validation"]["producer_fold"].astype(np.float32)
    with pytest.raises(extraction.ConfidNetFold4ExtractionError, match="int64"):
        extraction.validate_split_payload(result["validation"], "validation")

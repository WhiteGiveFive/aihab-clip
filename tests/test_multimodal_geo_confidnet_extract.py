from __future__ import annotations

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

from multimodal import geo_helpfulness_confidnet_extract as extraction
from multimodal import geo_helpfulness_oof as m2
from multimodal.models import MLPHead


def _payload(split: str, n: int = 3) -> dict[str, np.ndarray]:
    payload = {
        "row_uid": np.asarray([f"u{i}" for i in range(n)]),
        "plot_idx": np.asarray([f"p{i}" for i in range(n)]),
    }
    for prefix in extraction.PREFIXES:
        payload[f"{prefix}_hidden"] = np.zeros((n, 256), dtype=np.float32)
        payload[f"{prefix}_logits"] = np.tile(
            np.arange(18, dtype=np.float64), (n, 1)
        )
    if split == "train":
        payload["labels"] = np.zeros(n, dtype=np.int64)
    return payload


def _rows(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "row_uid": [f"u{i}" for i in range(n)],
        "file": [f"image/{i}.jpg" for i in range(n)],
        "file_lower": [f"image/{i}.jpg" for i in range(n)],
        "plot_idx": [f"p{i}" for i in range(n)],
        "image_source": ["/allowed/images"] * n,
        "label_id_dense": np.arange(n, dtype=np.int64),
    })


def _reference(payload: dict, rows: pd.DataFrame) -> pd.DataFrame:
    reference = rows.loc[:, ["row_uid", "file", "file_lower", "plot_idx"]].copy()
    for prefix in extraction.PREFIXES:
        reference[f"{prefix}_logits"] = list(payload[f"{prefix}_logits"].copy())
        reference[f"{prefix}_pred"] = payload[f"{prefix}_logits"].argmax(axis=1)
    return reference


def test_hidden_capture_reproduces_unchanged_m2_forward_across_batches():
    torch.manual_seed(8)
    head = MLPHead(input_dim=7, hidden_dim=256, output_dim=18, dropout=0.7)
    head.train()
    features = np.random.default_rng(14).normal(size=(257, 7)).astype(np.float32)
    hidden, observed = extraction._head_hidden_and_logits(head, features, torch.device("cpu"))
    expected = m2._predict_head(head, features, torch.device("cpu"))
    assert np.array_equal(observed, expected)
    with torch.no_grad():
        expected_hidden = torch.cat([
            head.net[1](head.net[0](torch.from_numpy(features[i:i + 128])))
            for i in range(0, len(features), 128)
        ]).numpy()
    assert np.array_equal(hidden, expected_hidden)
    assert hidden.dtype == np.float32
    assert observed.dtype == np.float64
    assert not head.training
    assert all(not parameter.requires_grad for parameter in head.parameters())
    assert not head.net[1]._forward_hooks


def test_projection_drops_labels_and_rejects_duplicate_identities():
    rows = _rows()
    rows["other_label"] = 0
    projected = extraction._prediction_projection(rows)
    assert tuple(projected.columns) == extraction.IDENTITY_COLUMNS
    rows.loc[1, "row_uid"] = rows.loc[0, "row_uid"]
    with pytest.raises(extraction.ConfidNetExtractionError, match="duplicate"):
        extraction._prediction_projection(rows)


@pytest.mark.parametrize("forbidden", ["labels", "label_id_dense", "true_class_probability"])
def test_validation_payload_rejects_label_leaks(forbidden):
    payload = _payload("validation")
    payload[forbidden] = np.zeros(3)
    with pytest.raises(extraction.ConfidNetExtractionError, match="allow-list"):
        extraction.validate_split_payload(payload, "validation")


def test_payload_rejects_invalid_dtype_and_label_range():
    payload = _payload("train")
    payload["labels"][0] = 18
    with pytest.raises(extraction.ConfidNetExtractionError, match="dense class IDs"):
        extraction.validate_split_payload(payload, "train")
    payload = _payload("validation")
    payload["image_hidden"] = payload["image_hidden"].astype(np.float64)
    with pytest.raises(extraction.ConfidNetExtractionError, match="shape or dtype"):
        extraction.validate_split_payload(payload, "validation")


def test_reproduction_checks_row_identity_and_exact_argmax():
    payload = _payload("validation")
    rows = extraction._prediction_projection(_rows())
    reference = _reference(payload, rows)
    record = extraction._validate_validation_reproduction(payload, rows, reference)
    assert record["argmax_exact"]
    assert record["maximum_absolute_logit_error"] == {"image": 0.0, "geo": 0.0, "raw": 0.0}
    bad_reference = reference.copy()
    bad_reference.loc[0, "plot_idx"] = "wrong"
    with pytest.raises(extraction.ConfidNetExtractionError, match="identity mismatch"):
        extraction._validate_validation_reproduction(payload, rows, bad_reference)
    payload["image_logits"][:] = 0.0
    reference = _reference(payload, rows)
    # Stay inside tolerance but change a tied prediction: replay must fail.
    payload["image_logits"][0, 1] = 1e-8
    with pytest.raises(extraction.ConfidNetExtractionError, match="argmax"):
        extraction._validate_validation_reproduction(payload, rows, reference)


def test_reproduction_rejects_logit_drift_and_labeled_reference():
    payload = _payload("validation")
    rows = extraction._prediction_projection(_rows())
    reference = _reference(payload, rows)
    payload["geo_logits"] = payload["geo_logits"] + 0.01
    with pytest.raises(extraction.ConfidNetExtractionError, match="logits do not reproduce"):
        extraction._validate_validation_reproduction(payload, rows, reference)
    reference["label_id_dense"] = 0
    with pytest.raises(extraction.ConfidNetExtractionError, match="label-blind"):
        extraction._validate_validation_reproduction(payload, rows, reference)


def test_extract_seed_reuses_one_frozen_bundle_without_validation_labels(
    tmp_path, monkeypatch
):
    seed = 1
    spec = m2._producer_specs(seed)[-1]
    directory = tmp_path / spec.relative_directory
    directory.mkdir(parents=True)
    rows = _rows(5)
    rows["development_role"] = ["train"] * 3 + ["validation"] * 2
    rows["train_oof_fold"] = pd.array([0, 1, 2, pd.NA, pd.NA], dtype="Int8")
    context = SimpleNamespace(
        output_root=tmp_path,
        assignments=rows,
        parent_hashes={"synthetic": "parent"},
        config={"protocol_id": "synthetic", "experts": {
            "head": {"hidden_dim": 256, "dropout": 0.1},
            "mode_inputs": {"image_only": {"input_dim": 4},
                            "geo_only": {"input_dim": 64},
                            "raw_concat": {"input_dim": 68}},
        }},
    )
    encoder = nn.Module()
    encoder.visual = nn.Linear(2, 4)
    save_file(encoder.visual.state_dict(), str(directory / "adapted_visual_tower.safetensors"))
    heads = {}
    for mode in m2.MODES:
        heads[mode] = MLPHead(context.config["experts"]["mode_inputs"][mode]["input_dim"], 256, 18)
        save_file(heads[mode].state_dict(), str(directory / f"{mode}_head.safetensors"))
    (directory / "geo_standardization.json").write_text("{}")
    (directory / spec.output_filename).write_bytes(b"synthetic")
    calls = []

    def fake_load(config, device):
        calls.append(("load", id(encoder)))
        return encoder, None, "synthetic_transform", None, {"synthetic": True}

    def fake_images(model, prediction_rows, **kwargs):
        assert tuple(prediction_rows.columns) == extraction.IDENTITY_COLUMNS
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        calls.append(("images", id(model), prediction_rows["row_uid"].tolist()))
        return np.ones((len(prediction_rows), 4), dtype=np.float32)

    def fake_geo(context, prediction_rows):
        assert tuple(prediction_rows.columns) == extraction.IDENTITY_COLUMNS
        return np.ones((len(prediction_rows), 64), dtype=np.float32)

    monkeypatch.setattr(m2, "load_frozen_context", lambda **kwargs: context)
    monkeypatch.setattr(m2, "validate_producer", lambda *args, **kwargs: {"valid": True})
    monkeypatch.setattr(m2, "_load_fresh_encoder", fake_load)
    monkeypatch.setattr(m2, "_extract_embeddings", fake_images)
    monkeypatch.setattr(m2, "_geo_source_projection", fake_geo)
    monkeypatch.setattr(
        m2, "_validate_scaler_file",
        lambda path: {"mean": np.zeros(64), "std": np.ones(64)},
    )
    validation_rows = rows.iloc[3:].reset_index(drop=True)
    reference_payload = {}
    for mode, prefix in m2.MODE_TO_PREFIX.items():
        width = context.config["experts"]["mode_inputs"][mode]["input_dim"]
        features = np.ones((len(validation_rows), width), dtype=np.float32)
        reference_payload[f"{prefix}_logits"] = m2._predict_head(
            heads[mode], features, torch.device("cpu")
        )
    reference = _reference(reference_payload, validation_rows)
    monkeypatch.setattr(
        m2, "read_output_parquet",
        lambda *args, **kwargs: pa.Table.from_pandas(reference),
    )
    result = extraction.extract_seed(seed, device="cpu", config_path=Path("synthetic.yaml"))
    assert len([call for call in calls if call[0] == "load"]) == 1
    assert [call[1] for call in calls if call[0] == "images"] == [id(encoder), id(encoder)]
    assert set(result) == {"train", "validation", "metadata"}
    assert result["train"]["labels"].tolist() == [0, 1, 2]
    assert "labels" not in result["validation"]
    assert result["metadata"]["validation_reproduction"]["argmax_exact"]
    assert result["metadata"]["locked_test_accessed"] is False
    json.dumps(result["metadata"])

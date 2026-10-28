"""Small CPU checks of the immutable, inference-only geometry extractor."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from multimodal import attention_fusion_geometry_embeddings as extraction


class TinyVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 3, bias=False)

    def forward(self, images):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert images.dtype == torch.float32
        return self.projection(images)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "new-analysis"
    (root / "protocol").mkdir(parents=True)
    (root / "inputs").mkdir()
    roles = {}
    for role, count in (("train", 18), ("validation", 3)):
        rows = pd.DataFrame({
            "row_uid": [f"{role}-{i:03d}" for i in reversed(range(count))],
            "plot_idx": [f"{role}-plot-{i}" for i in reversed(range(count))],
            "file": list(reversed(range(count))), "image_source": ["fixture"] * count,
            "label_id_dense": [0] * count,
        })
        rows.to_parquet(root / "inputs" / f"{role}_rows.parquet", index=False)
        roles[role] = rows.sort_values("row_uid").reset_index(drop=True)
    model = TinyVisual()
    model.projection.weight.data.copy_(torch.tensor([[1., 0.], [0., 2.], [1., 1.]]))
    source = tmp_path / "original.safetensors"
    save_file({**{f"visual.{key}": value for key, value in model.state_dict().items()},
               "text.unused": torch.ones(4)}, str(source))
    declaration = {
        "encoders": {"pretrained": {"path": str(source), "sha256": extraction.sha256_file(source),
                                     "prefix": "visual.", "epoch": 0}},
        "extraction": {"batch_size": 16, "num_workers": 0, "cpu_threads": 1,
                       "device": "cpu", "autocast": False, "cudnn_allow_tf32": True,
                       "matmul_allow_tf32": False},
        "model_provenance": {"encoder_architecture": {"embed_dim": 3},
                             "historical_preprocessing_recipe": {"prediction_transform": "fixture"}},
        "source_config": {"fixture": True}, "class_labels": ["fixture"],
        "input_sha256": {f"inputs/{role}_rows.parquet": extraction.sha256_file(
            root / "inputs" / f"{role}_rows.parquet") for role in extraction.ROLES},
    }
    (root / "protocol/declaration.json").write_text(json.dumps(declaration))
    calls = {"construct": 0, "roles": []}

    def construct(_provenance):
        calls["construct"] += 1
        return TinyVisual(), "never-use-train-transform", "prediction-transform"

    class Dataset:
        def __init__(self, rows, *, config, transform, include_labels):
            assert transform == "prediction-transform"
            assert not include_labels
            assert config == {"fixture": True}
            calls["roles"].append(rows.row_uid.tolist())
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return torch.tensor([float(self.rows.iloc[index].file) + 1, 1.]), index

    monkeypatch.setattr(extraction, "_construct_visual_and_transforms", construct)
    monkeypatch.setattr(extraction, "_LegacyImageDataset", Dataset)
    return root, source, model, roles, calls


def test_visual_only_sorted_normalized_extraction_and_completed_cache_reuse(prepared):
    root, source, model, roles, calls = prepared
    original_bytes = source.read_bytes()
    result = extraction.extract_embeddings(root, device="cpu")
    manifest = result["encoders"]["pretrained"]
    assert manifest["declaration_sha256"] == extraction.sha256_file(root / "protocol/declaration.json")
    assert manifest["file_sha256"] == manifest["files"]
    assert calls["construct"] == 1
    assert calls["roles"] == [rows.row_uid.tolist() for rows in roles.values()]
    for role, rows in roles.items():
        with np.load(root / "embeddings/pretrained" / f"{role}.npz", allow_pickle=False) as saved:
            images = torch.tensor([[float(x) + 1, 1.] for x in rows.file])
            with torch.no_grad():
                expected = torch.nn.functional.normalize(model.projection(images), dim=-1).numpy()
            np.testing.assert_allclose(saved["image_embedding"], expected, rtol=1e-6, atol=1e-7)
            assert saved["row_uid"].tolist() == rows.row_uid.tolist()
            assert saved["plot_idx"].tolist() == rows.plot_idx.tolist()
            assert set(saved.files) == {"row_uid", "plot_idx", "image_embedding"}
    extraction.extract_embeddings(root, device="cpu")
    assert calls["construct"] == 1
    assert source.read_bytes() == original_bytes
    assert not list(root.glob("**/*.safetensors"))


def test_changed_prepared_rows_rejected_before_extraction(prepared):
    root, _, _, _, calls = prepared
    path = root / "inputs/train_rows.parquet"
    rows = pd.read_parquet(path)
    rows.loc[0, "file"] = 999
    rows.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="frozen declaration"):
        extraction.extract_embeddings(root, device="cpu")
    assert calls["construct"] == 0


def test_changed_cache_is_rejected_without_overwriting(prepared):
    root, _, _, _, calls = prepared
    extraction.extract_embeddings(root, device="cpu")
    cache = root / "embeddings/pretrained/train.npz"
    cache.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cache file hash changed"):
        extraction.extract_embeddings(root, device="cpu")
    assert cache.read_bytes() == b"corrupt"
    assert calls["construct"] == 1


def test_source_hash_retry_is_bounded_and_diagnostics_stay_in_new_root(prepared, monkeypatch):
    root, source, _, _, calls = prepared
    expected = extraction.sha256_file(source)
    observed = []
    original_hash = extraction.sha256_file

    def changed_hash(path):
        if path == source:
            observed.append(path)
            return "0" * 64
        return original_hash(path)

    advice = []
    monkeypatch.setattr(extraction, "sha256_file", changed_hash)
    monkeypatch.setattr(extraction, "_refresh_read_cache", lambda path: advice.append(path) or {"status": "requested"})
    with pytest.raises(extraction.JointCheckpointIntegrityError) as error:
        extraction.extract_embeddings(root, device="cpu")
    assert error.value.attempts == 3
    assert len(observed) == 3 and advice == [source, source]
    report = json.loads((root / "verification/pretrained_source_integrity.json").read_text())
    assert report["expected"] == expected and report["status"] == "unresolved"
    assert calls["construct"] == 0
    assert not (root / "embeddings/pretrained").exists()
    assert not (source.parent / "diagnostics").exists()

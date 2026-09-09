"""Synthetic coverage for the declaration-before-features inference boundary."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from multimodal import geo_helpfulness_legacy_router_inference as inference
from multimodal.geo_helpfulness_legacy_router_common import canonical_hash
from multimodal.geo_helpfulness_legacy_router_report import ONTOLOGY


def _seal(value, key="manifest_sha256"):
    return {**value, key: canonical_hash(value)}


def _checkpoint(mode, *, family="best_model", epoch=None):
    dimension = inference.INPUT_DIMS[mode]
    # Nonzero biases and negative inputs exercise both affine layers and ReLU.
    generator = torch.Generator().manual_seed(47)
    state = {
        "head.net.0.weight": torch.randn(256, dimension, generator=generator) * .01,
        "head.net.0.bias": torch.randn(256, generator=generator) * .01,
        "head.net.3.weight": torch.randn(18, 256, generator=generator) * .01,
        "head.net.3.bias": torch.randn(18, generator=generator) * .01,
    }
    result = {"model_state": state, "mode": mode, "epoch": epoch or 50, "training_regime": "fixed_epoch_train_val"}
    if family == "oracle_test_best_model":
        result.update(selection_split="test", selection_metric="top1_acc", diagnostic_only=True)
    return result


@pytest.mark.parametrize("mode", inference.MODES)
def test_forward_matches_eval_mlp_across_batch_boundary(tmp_path, mode):
    checkpoint = _checkpoint(mode)
    path = tmp_path / "best_model.pt"
    torch.save(checkpoint, path)
    loaded = inference._load_head(path, mode, "best_model")
    model = torch.nn.Sequential(
        torch.nn.Linear(inference.INPUT_DIMS[mode], 256), torch.nn.ReLU(),
        torch.nn.Dropout(.1), torch.nn.Linear(256, 18),
    ).eval()
    model.load_state_dict({name.replace("head.net.", ""): value for name, value in checkpoint["model_state"].items()})
    values = np.random.default_rng(4).normal(size=(131, inference.INPUT_DIMS[mode])).astype(np.float32)
    actual = inference._head_logits(loaded, values)
    with torch.inference_mode():
        expected = np.concatenate([model(torch.from_numpy(values[start:start + 128])).numpy() for start in (0, 128)])
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected.astype(np.float64))


@pytest.mark.parametrize("mutation", ["epoch", "shape", "dtype", "nonfinite", "oracle_status"])
def test_checkpoint_rejects_invalid_metadata_or_parameters(tmp_path, mutation):
    family = "oracle_test_best_model" if mutation == "oracle_status" else "best_model"
    checkpoint = _checkpoint("geo_only", family=family)
    if mutation == "epoch":
        checkpoint["epoch"] = 49
    elif mutation == "shape":
        checkpoint["model_state"]["head.net.3.bias"] = torch.zeros(19)
    elif mutation == "dtype":
        checkpoint["model_state"]["head.net.3.bias"] = torch.zeros(18, dtype=torch.float64)
    elif mutation == "nonfinite":
        checkpoint["model_state"]["head.net.3.bias"][0] = float("nan")
    else:
        checkpoint["diagnostic_only"] = False
    path = tmp_path / f"{family}.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        inference._load_head(path, "geo_only", family)


def test_identity_rejects_casefold_collision():
    with pytest.raises(ValueError, match="duplicate"):
        inference._identity(pd.DataFrame({"file": ["A.JPG", "a.jpg"], "plot_idx": ["p1", "p2"]}))


def _locked_rows():
    raw = pd.DataFrame({"file": [f"test/{i:04d}.jpg" for i in range(1347)], "plot_idx": [f"t{i % 531}" for i in range(1347)]})
    identity = inference._identity(raw)
    identity["file"] = raw.file
    identity["normalized_plot_idx"] = identity.plot_idx
    columns = ["row_uid", "file", "file_lower", "plot_idx", "normalized_plot_idx"]
    return identity[columns].sort_values("row_uid").to_dict("records")


@pytest.fixture
def declaration_sources(tmp_path, monkeypatch):
    """Full fixed counts, small feature storage, and deliberately unreadable test bytes."""
    protocol = tmp_path / "protocol_v1"
    frozen = protocol / "router/m4_feasibility_v1"
    m1 = protocol / "protocol"
    legacy = tmp_path / "legacy"
    cache = tmp_path / "gse_100m_cleaned_test"
    output = tmp_path / "output"
    registry = tmp_path / "registry"
    for directory in (frozen, m1, registry):
        directory.mkdir(parents=True)
    ontology = [dict(dense_id=d, canonical_l3_id=c, label_name=n) for d, c, n in ONTOLOGY]
    rows = _locked_rows()
    digest = canonical_hash(rows)
    identity = _seal({"rows": rows, "dataset_id": inference.DATASET_ID, "row_count": 1347, "plot_count": 531, "identity_projection_sha256": digest, "snapshot_id": digest})
    identity_path = registry / "identity.json"
    inference.write_json(identity_path, identity)
    inference.write_json(registry / "active_snapshot.json", {"snapshot_id": digest})
    reference = {"identity_manifest_relative_path": "identity.json", "identity_manifest_file_sha256": inference.file_hash(identity_path)}
    inference.write_json(m1 / "locked_test_snapshot_ref.json", reference)
    config = {"class_ontology": {"classes": ontology}, "paths": {"locked_test_registry_root": str(registry)}}
    (m1 / "resolved_protocol.yaml").write_text(yaml.safe_dump(config))
    dev = pd.DataFrame({"file": [f"dev/{i:04d}.jpg" for i in range(4200)], "plot_idx": [f"d{i % 1625}" for i in range(4200)]})
    dev["label_id"] = np.arange(4200) % 18
    dev["label_name"] = [ontology[i]["label_name"] for i in dev.label_id]
    dev_identity = inference._identity(dev)
    assignments = pd.concat([dev_identity[["row_uid", "plot_idx"]], dev[["label_id", "label_name"]].rename(columns={"label_id": "label_id_dense"})], axis=1)
    assignments.to_parquet(m1 / "development_assignments.parquet", index=False)
    geo = pd.DataFrame(np.zeros((4200, 64), dtype=np.float32), columns=inference.GEO_COLUMNS)
    dev = pd.concat([dev, geo], axis=1)
    for seed in inference.SEEDS:
        directory = cache / f"seed{seed}"
        directory.mkdir(parents=True)
        dev.iloc[:3618].to_parquet(directory / "train.parquet", index=False)
        dev.iloc[3618:].to_parquet(directory / "val.parquet", index=False)
        # Hashing is permitted before declaration; decoding these bytes must fail.
        (directory / "test.parquet").write_bytes(b"test features and labels must not be decoded")
    scaler = {"fit_split": "train_val", "fit_rows": 4200, "modality_name": "geo", "feature_columns": list(inference.GEO_COLUMNS), "class_names": [o["label_name"] for o in ontology], "label_id_remap": {str(i): i for i in range(18)}, "mean": [0.] * 64, "std": [1.] * 64}
    for mode in inference.MODES:
        for seed in inference.SEEDS:
            directory = legacy / mode / inference.RUN_TAG / f"seed{seed}"
            directory.mkdir(parents=True)
            inference.write_json(directory / "geo_standardization.json", scaler)
            (directory / "metrics.json").write_bytes(b"historical metrics must only be hashed")
            for family in inference.FAMILIES:
                torch.save(_checkpoint(mode, family=family), directory / f"{family}.pt")
    m1_manifest = {
        "artifact_file_sha256": {name: inference.file_hash(m1 / name) for name in ("development_assignments.parquet", "resolved_protocol.yaml", "locked_test_snapshot_ref.json")},
        "development_source_file_sha256": {},
        "class_map_sha256": canonical_hash(config["class_ontology"]),
        "locked_test_identity_projection_sha256": digest,
        "development_identity_projection_sha256": canonical_hash(dev_identity.sort_values(["row_uid", "file_lower", "plot_idx"]).to_dict("records")),
    }
    inference.write_json(m1 / "protocol_manifest.json", _seal(m1_manifest, "manifest_payload_sha256"))
    inference.write_json(frozen / "router_frozen_state.json", {"components": {}})
    inference.write_json(frozen / "manifest.json", {})
    monkeypatch.setattr(inference, "validate_manifest", lambda *args, **kwargs: {"primary_decision": "no_go", "parent_file_sha256": {}, "artifacts": {}})
    monkeypatch.setattr(inference, "_code_snapshot", lambda: {})
    return protocol, legacy, cache, output


def test_declaration_never_decodes_test_or_historical_metrics(declaration_sources, monkeypatch):
    original = pd.read_parquet
    decoded = []
    def guarded(path, *args, **kwargs):
        assert Path(path).name != "test.parquet"
        decoded.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    declaration = inference.create_declaration(*declaration_sources)
    assert len(decoded) == 9
    assert len(declaration["checkpoints"]) == 24
    assert len(declaration["historical_metrics_sha256"]) == 12
    assert not set(declaration["historical_metrics_sha256"]) & set(declaration["source_snapshot"])
    assert declaration["no_go_preserved"] is True
    assert all(declaration["caches"][str(seed)]["test"] in declaration["source_snapshot"] for seed in inference.SEEDS)


def test_inference_rejects_missing_declaration_before_test_read(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("test table was accessed without a saved declaration")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        inference.infer_expert_logits({"output_root": str(tmp_path)})


def test_inference_uses_only_projected_features_and_preserves_image_values(declaration_sources, monkeypatch):
    declaration = inference.create_declaration(*declaration_sources)
    identity = pd.DataFrame(declaration["identity_manifest"]["rows"])
    test = identity[["file", "plot_idx"]].copy()
    image = pd.DataFrame(np.full((1347, 1152), 2., dtype=np.float32), columns=inference.IMAGE_COLUMNS)
    geo = pd.DataFrame(np.full((1347, 64), 3., dtype=np.float32), columns=inference.GEO_COLUMNS)
    test = pd.concat([test, image, geo], axis=1)
    seen = []
    def projected(path, *, columns):
        assert Path(path).name == "test.parquet"
        assert columns == ["file", "plot_idx", *inference.IMAGE_COLUMNS, *inference.GEO_COLUMNS]
        seen.append(path)
        return test.loc[:, columns].iloc[::-1].reset_index(drop=True)
    def forward(checkpoint, values):
        mode = checkpoint["mode"]
        if mode != "geo_only":
            np.testing.assert_array_equal(values[:, :1152], 2.)
        if mode != "image_only":
            np.testing.assert_array_equal(values[:, -64:], 3.)
        return np.zeros((len(values), 18), dtype=np.float64)
    monkeypatch.setattr(pd, "read_parquet", projected)
    monkeypatch.setattr(inference, "_head_logits", forward)
    actual = inference.infer_expert_logits(declaration)
    assert len(seen) == 4
    assert len(actual) == 1347 * 4 * 2
    assert list(actual.columns) == ["checkpoint_family", "row_uid", "file", "plot_idx", "training_seed", "image_logits", "geo_logits", "raw_logits"]
    keys = list(actual[["checkpoint_family", "row_uid", "training_seed"]].itertuples(index=False, name=None))
    assert keys == sorted(keys)


def test_declaration_rejects_mismatched_development_label(declaration_sources):
    protocol, legacy, cache, output = declaration_sources
    for seed in inference.SEEDS:
        path = cache / f"seed{seed}" / "train.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "label_id"] = 17
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="labels differ"):
        inference.create_declaration(protocol, legacy, cache, output)
    assert not (output / "declaration.json").exists()


@pytest.mark.parametrize("kind", ["nested_source", "symlink", "unknown_existing"])
def test_declaration_refuses_unsafe_output_before_reading_sources(tmp_path, kind):
    protocol, legacy, cache = [tmp_path / name for name in ("protocol", "legacy", "gse_100m_cleaned_test")]
    output = tmp_path / "output"
    if kind == "nested_source":
        output = protocol / "new_output"
    elif kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    else:
        output.mkdir()
        (output / "unknown.json").write_text("{}")
    with pytest.raises(ValueError, match="separate|symlink|unknown"):
        inference.create_declaration(protocol, legacy, cache, output)


def test_changed_source_is_rejected_before_test_projection(declaration_sources, monkeypatch):
    declaration = inference.create_declaration(*declaration_sources)
    path = Path(declaration["caches"]["1"]["test"])
    path.write_bytes(b"changed cache bytes")
    def forbidden(*args, **kwargs):
        pytest.fail("changed sources must fail before any projected test read")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="changed"):
        inference.infer_expert_logits(declaration)

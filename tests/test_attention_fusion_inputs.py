"""Input lineage, alignment and no-test-access guarantees without real models."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal import attention_fusion_inputs as inputs
from multimodal.geo_helpfulness_protocol import canonical_sha256, sha256_file


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = {
        "study_id": "fixture", "data": {"allow_test_features_or_labels": False,
        "training_seeds": [1], "train_rows": 2, "validation_rows": 1,
        "image_dim": 3, "geo_dim": 2, "num_classes": 18,
        "dense_to_canonical_l3": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]},
        "image": {"encoder_frozen": True, "extraction_device": "cuda", "extraction_batch_size": 16,
                  "extraction_cudnn_allow_tf32": True, "extraction_matmul_allow_tf32": False},
        "paths": {"artifact_root": "artifacts", "image_root": "images",
                  "assignments": "assignments.parquet",
                  "prepared_cache_template": "artifacts/inputs/seed{seed}/{role}.npz"},
    }
    rows = pd.DataFrame({"row_uid": ["a", "b", "c"], "plot_idx": ["p1", "p2", "p3"],
                         "label_id_dense": [0, 1, 2], "development_role": ["train", "train", "validation"]})
    rows.to_parquet(tmp_path / "assignments.parquet", index=False)
    encoder = {"seed": 1, "checkpoint": {"path": "encoder.safetensors", "sha256": "e" * 64}}
    validation = {"seed": 1, "features": {"path": "validation.npz", "sha256": "v" * 64}}
    audit = {"encoders": [encoder], "validation_image_caches": [validation],
             "protocol_files": {"development_assignments.parquet": {"sha256": sha256_file(tmp_path / "assignments.parquet")}},
             "development_geo_source_files": []}
    binding = {"audit": audit, "manifest_file_sha256": "m" * 64}
    original_hash = canonical_sha256(config)

    def bind(value, repo):
        inputs._require(canonical_sha256(value) == original_hash, "Configuration changed")
        return binding

    monkeypatch.setattr(inputs, "_bind_m1", bind)
    monkeypatch.setattr(inputs, "_configure_extraction", lambda *a: {
        "device": "cuda", "batch_size": 16, "cudnn_allow_tf32": True, "matmul_allow_tf32": False})
    monkeypatch.setattr(inputs, "_validate_historical_image_fingerprint", lambda *a: {
        "development_image_file_count": 3, "development_image_content_sha256": "h" * 64,
        "matches_historical_protocol": True})
    context = {"config": config, "repo": tmp_path, "roles": {}, "source_config": {},
               "audit": audit, "m1_binding": binding}
    for role in inputs.ROLES:
        subset = rows.loc[rows.development_role.eq(role)].reset_index(drop=True)
        context["roles"][role] = {"rows": subset, "geo": np.arange(len(subset) * 2, dtype=np.float64).reshape(-1, 2)}
    monkeypatch.setattr(inputs, "audit_source_inputs", lambda *a, **k: context)
    monkeypatch.setattr(inputs, "_image_inventory", lambda *a: [{"role": "train", "sha256": "i" * 64}])
    monkeypatch.setattr(inputs, "_verify_raw_inventory", lambda *a: None)
    monkeypatch.setattr(inputs, "_load_encoder", lambda *a: (None, None, {"local_only": True}))
    extract_calls = []

    def extract(context, model, transform, rows, **kwargs):
        extract_calls.append(rows.row_uid.tolist())
        return np.tile(np.array([[1, 0, 0]], dtype=np.float32), (len(rows), 1))

    monkeypatch.setattr(inputs, "_extract", extract)
    monkeypatch.setattr(inputs, "_read_validation_image", lambda *a: np.array([[1, 0, 0]], dtype=np.float32))
    return config, tmp_path, context, extract_calls


def test_prepare_roundtrip_preserves_native_geo_and_refuses_overwrite(prepared, monkeypatch):
    config, repo, context, calls = prepared
    result = inputs.prepare_inputs(config, [1], repo_root=repo, replay_rows=16)
    assert len(result["seeds"]) == 1
    assert calls == [["c"], ["a", "b"]]
    train = inputs.load_prepared_inputs(config, 1, "train", repo_root=repo)
    np.testing.assert_array_equal(train["geo_embedding"], context["roles"]["train"]["geo"])
    assert train["row_uid"].tolist() == ["a", "b"]
    before = (repo / "artifacts/inputs/seed1/manifest.json").read_bytes()
    monkeypatch.setattr(inputs, "_extract", lambda *a, **k: pytest.fail("Completed cache re-extracted"))
    inputs.prepare_inputs(config, [1], repo_root=repo)
    assert (repo / "artifacts/inputs/seed1/manifest.json").read_bytes() == before


def test_replay_disagreement_leaves_no_completed_cache(prepared, monkeypatch):
    config, repo, _, _ = prepared
    monkeypatch.setattr(inputs, "_read_validation_image", lambda *a: np.array([[0, 1, 0]], dtype=np.float32))
    with pytest.raises(inputs.InputError, match="replay disagrees"):
        inputs.prepare_inputs(config, [1], repo_root=repo, replay_rows=16)
    assert not (repo / "artifacts/inputs/seed1").exists()
    assert not list((repo / "artifacts/inputs").glob(".seed1-*"))


def test_test_role_rejected_before_any_file_read(monkeypatch):
    monkeypatch.setattr(inputs, "_config", lambda *a: pytest.fail("Read config before rejecting test"))
    with pytest.raises(inputs.InputError, match="test access is prohibited"):
        inputs.load_prepared_inputs("missing.yml", 1, "test")


def test_mixed_geo_source_rejected_before_payload_access(monkeypatch, tmp_path):
    config = {"data": {"allow_test_features_or_labels": False, "training_seeds": [1]},
              "image": {"encoder_frozen": True}, "paths": {
                  "source_protocol_root": "historical", "assignments": "historical/protocol/development_assignments.parquet",
                  "development_geo_tables": ["mixed_development_test.parquet"], "image_root": "images"}}
    monkeypatch.setattr(inputs, "_bind_m1", lambda *a: {})
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Forbidden table was opened"))
    with pytest.raises(inputs.InputError, match="Only the audited development geo tables"):
        inputs.audit_source_inputs(config, [1], repo_root=tmp_path)


def test_corrupted_cache_bytes_rejected(prepared):
    config, repo, _, _ = prepared
    inputs.prepare_inputs(config, [1], repo_root=repo, replay_rows=16)
    with (repo / "artifacts/inputs/seed1/train.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(inputs.InputError, match="cache file hash mismatch"):
        inputs.load_prepared_inputs(config, 1, "train", repo_root=repo)


def test_wrong_label_rejected_even_with_updated_container_hash(prepared):
    config, repo, _, _ = prepared
    inputs.prepare_inputs(config, [1], repo_root=repo, replay_rows=16)
    directory = repo / "artifacts/inputs/seed1"
    path = directory / "train.npz"
    with np.load(path) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays["label"][0] = 7
    np.savez_compressed(path, **arrays)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["file_sha256"]["train.npz"] = sha256_file(path)
    manifest["manifest_sha256"] = inputs._manifest_hash(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(inputs.InputError, match="label does not match"):
        inputs.load_prepared_inputs(config, 1, "train", repo_root=repo)


@pytest.mark.parametrize("mutation", ["zero_image", "nan_geo", "wrong_order", "extra_test_array"])
def test_invalid_numeric_or_identity_arrays(prepared, mutation):
    config, _, context, _ = prepared
    arrays = inputs._pack(np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32), context["roles"]["train"])
    if mutation == "zero_image":
        arrays["image_embedding"][0] = 0
    elif mutation == "nan_geo":
        arrays["geo_embedding"][0, 0] = np.nan
    elif mutation == "wrong_order":
        arrays["row_uid"] = arrays["row_uid"][::-1]
    else:
        arrays["test_label"] = np.array([1])
    with pytest.raises(inputs.InputError):
        inputs.validate_arrays(arrays, config, "train")


def test_verification_exposes_only_training_rows_and_separate_destination(prepared):
    config, repo, _, calls = prepared
    manifest = inputs.prepare_verification_inputs(config, seed=1, limit=2, replay_rows=16, repo_root=repo)
    assert manifest["verification_only"] is True
    assert calls == [["c"], ["a", "b"]]
    arrays = inputs.load_verification_inputs(config, 1, repo_root=repo)
    assert arrays["row_uid"].tolist() == ["a", "b"]
    assert not (repo / "artifacts/inputs/seed1").exists()
    provenance = inputs.load_input_provenance(config, 1, verification=True, repo_root=repo)
    assert provenance["source_role"] == "train"
    assert provenance["validation_role"] is None


def test_changed_config_does_not_reuse_cache(prepared):
    config, repo, _, _ = prepared
    inputs.prepare_inputs(config, [1], repo_root=repo, replay_rows=16)
    changed = copy.deepcopy(config)
    changed["image"]["other_recipe"] = True
    with pytest.raises(inputs.InputError, match="Configuration changed"):
        inputs.load_prepared_inputs(changed, 1, "train", repo_root=repo)


def test_raw_images_changed_during_extraction_fail_before_commit(prepared, monkeypatch):
    config, repo, _, _ = prepared

    def changed(*args):
        raise inputs.InputError("Raw development image changed during extraction")

    monkeypatch.setattr(inputs, "_verify_raw_inventory", changed)
    with pytest.raises(inputs.InputError, match="changed during extraction"):
        inputs.prepare_inputs(config, [1], repo_root=repo, replay_rows=16)
    assert not (repo / "artifacts/inputs/seed1").exists()


def test_geo_only_needs_no_prepared_image_cache(prepared, monkeypatch):
    config, repo, context, _ = prepared
    monkeypatch.setattr(inputs, "_load_encoder", lambda *a: pytest.fail("Geo-only loaded encoder"))
    monkeypatch.setattr(inputs, "_read_validation_image", lambda *a: pytest.fail("Geo-only loaded image cache"))
    train, validation, provenance = inputs.load_geo_only_inputs(config, 1, repo_root=repo)
    np.testing.assert_array_equal(train["geo_embedding"], context["roles"]["train"]["geo"])
    assert not train["image_embedding"].any()
    assert not validation["image_embedding"].any()
    assert provenance["image_embeddings_loaded"] is False
    assert not (repo / "artifacts/inputs/seed1").exists()


@pytest.mark.parametrize("device,batch_size,message", [
    ("cpu", 16, "requires CUDA"),
    ("cuda", 1, "requires batch size 16"),
])
def test_incompatible_extraction_rejected_before_gpu_access(device, batch_size, message):
    config = {"image": {"extraction_device": "cuda", "extraction_batch_size": 16,
                        "extraction_cudnn_allow_tf32": True, "extraction_matmul_allow_tf32": False}}
    with pytest.raises(inputs.InputError, match=message):
        inputs._configure_extraction(config, device, batch_size, 4)


def test_historical_raw_image_fingerprint_is_ordered_and_enforced(tmp_path):
    rows = pd.DataFrame({"row_uid": ["c", "a", "b"], "file": ["c.jpg", "a.jpg", "b.jpg"]})
    inventory = [{"row_uid": uid, "sha256": uid * 64} for uid in ("b", "c", "a")]
    expected = canonical_sha256([{"row_uid": uid, "image_file_sha256": uid * 64}
                                 for uid in ("a", "b", "c")])
    context = {"repo": tmp_path, "config": {"paths": {"image_root": "images"}},
               "roles": {"train": {"rows": rows.iloc[:2]}, "validation": {"rows": rows.iloc[2:]}},
               "source_protocol_manifest": {"development_image_file_count": 3,
                                            "development_image_content_sha256": expected}}
    assert inputs._validate_historical_image_fingerprint(context, inventory)["matches_historical_protocol"] is True
    inventory[0]["sha256"] = "changed"
    with pytest.raises(inputs.InputError, match="historical protocol fingerprint"):
        inputs._validate_historical_image_fingerprint(context, inventory)

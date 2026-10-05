"""Synthetic adapter tests without opening any development-validation labels."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_router_capacity_inputs as adapter
from multimodal.geo_helpfulness_router_numeric import fit_router_feature_transform
from multimodal.geo_helpfulness_targets_features import (
    build_router_feature_frame,
    build_router_feature_schema,
)


def _softmax(values):
    centered = values - values.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def _predict(matrix, state):
    offsets = np.array([.2, -.1, .4, -.3], dtype=np.float64)
    logits = np.tile(offsets, (len(matrix), 1))
    logits[:, 0] += matrix[:, 0] * .01 + state["C"] * .01
    logits[:, 1] += state["training_seed"] * .02
    return _softmax(logits)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "TRAIN_ROWS", 4)
    monkeypatch.setattr(adapter, "TRAIN_PLOTS", 4)
    monkeypatch.setattr(adapter, "VALIDATION_ROWS", 3)
    monkeypatch.setattr(adapter, "VALIDATION_PLOTS", 3)
    repo, root = tmp_path / "repo", tmp_path / "protocol_v1"
    router = root / "router"
    comparator_root = router / "m4_comparator_inputs_v1"
    lr_root = router / "router_validation_v1"
    for folder in (comparator_root, lr_root):
        folder.mkdir(parents=True)
    generator = np.random.default_rng(22)
    train_probs = [_softmax(generator.normal(size=(4, 18))) for _ in range(3)]
    val_logits = [generator.normal(size=(3, 18)) for _ in range(3)]
    val_probs = [_softmax(values) for values in val_logits]
    semantic_train = build_router_feature_frame(*train_probs, probability_basis="scalar_temperature_calibrated")
    semantic_val = build_router_feature_frame(*val_probs, probability_basis="scalar_temperature_calibrated")
    train_parts, audit_parts, expert_parts, output_parts = [], [], [], []
    transforms, temperatures, states, val_matrices = {}, {}, {}, {}
    for seed in adapter.TRAINING_SEEDS:
        identity = pd.DataFrame({
            "schema_version": ["synthetic"] * 4, "protocol_id": ["protocol_v1"] * 4,
            "row_uid": [f"train{i}" for i in range(4)], "plot_idx": [f"T{i}" for i in range(4)],
            "training_seed": np.full(4, seed, dtype=np.int8),
        })
        target = identity.assign(target_state=list(adapter.TARGET_ORDER))
        transform = fit_router_feature_transform(semantic_train, seed=seed, frozen_schema=build_router_feature_schema(), fit_row_identity_sha256="a" * 64)
        transforms[seed] = transform
        matrix = adapter.transform_router_features(semantic_train, transform)
        train_parts.append(pd.concat([target, pd.DataFrame(matrix, columns=adapter.full_feature_columns())], axis=1))
        audit_parts.append(pd.concat([identity, semantic_train], axis=1))
        valid_identity = pd.DataFrame({
            "schema_version": ["synthetic"] * 3, "protocol_id": ["protocol_v1"] * 3,
            "row_uid": [f"validation{i}" for i in range(3)], "plot_idx": [f"V{i}" for i in range(3)],
            "training_seed": np.full(3, seed, dtype=np.int8),
        })
        expert = valid_identity.copy()
        for basis in ("calibrated", "native_t1"):
            for prefix, values in zip(adapter.MODES.values(), val_probs):
                expert[f"{prefix}_prob_{basis}"] = list(values)
        expert_parts.append(pd.concat([expert, semantic_val], axis=1))
        output = valid_identity.copy()
        for (mode, prefix), logits, probability in zip(adapter.MODES.items(), val_logits, val_probs):
            output[f"{prefix}_logits"] = list(logits)
            output[f"{prefix}_prob_native_t1"] = list(probability)
            temperatures[(seed, mode)] = {"temperature": 1.0}
        output_parts.append(output)
        val_matrices[seed] = adapter.transform_router_features(semantic_val, transform)
        for variant in adapter.VARIANTS:
            for C in adapter.C_GRID:
                states[adapter.candidate_filename(variant, seed, C)] = {"training_seed": seed, "variant_id": variant, "C": C}
    def combined(parts):
        return pd.concat(parts, ignore_index=True).sort_values(["row_uid", "training_seed"]).reset_index(drop=True)
    train, audit, experts, outputs = map(combined, (train_parts, audit_parts, expert_parts, output_parts))
    train.to_parquet(router / "router_dataset.parquet", index=False)
    experts.to_parquet(comparator_root / "expert_probabilities_and_features.parquet", index=False)
    for variant in adapter.VARIANTS:
        frames = []
        for C in adapter.C_GRID:
            for seed in adapter.TRAINING_SEEDS:
                frame = experts.loc[experts.training_seed.eq(seed), list(adapter.TABLE_IDENTITY_COLUMNS)].copy().reset_index(drop=True)
                frame["C_hex"] = C.hex()
                q = _predict(val_matrices[seed], states[adapter.candidate_filename(variant, seed, C)])
                for index, column in enumerate(adapter.Q_COLUMNS):
                    frame[column] = q[:, index]
                frame["score"] = q[:, 0] - q[:, 1]
                frames.append(frame)
        pd.concat(frames, ignore_index=True).to_parquet(lr_root / f"{variant}.parquet", index=False)
    lr_manifest = {
        "manifest_sha256": "b" * 64, "dataset_manifest_sha256": "c" * 64,
        "candidate_manifest_sha256": "d" * 64, "parent_file_sha256": {},
        "artifacts": {f"{v}.parquet": {"file_sha256": adapter.file_hash(lr_root / f"{v}.parquet")} for v in adapter.VARIANTS},
    }
    comparator_manifest = {
        "prediction_manifest_sha256": lr_manifest["manifest_sha256"],
        "assignment_file_sha256": "e" * 64, "parent_file_sha256": {},
        "artifacts": {"expert_probabilities_and_features.parquet": {"file_sha256": adapter.file_hash(comparator_root / "expert_probabilities_and_features.parquet")}},
    }
    _json(lr_root / "manifest.json", lr_manifest)
    _json(comparator_root / "manifest.json", comparator_manifest)
    # If the adapter tries to discover or decode scoring labels, it must fail.
    forbidden_labels = router / "m4_feasibility_v1/validation_targets.parquet"
    forbidden_labels.parent.mkdir()
    forbidden_labels.write_bytes(b"validation labels are not available to this adapter")
    inputs = {
        "dataset_manifest": {"manifest_sha256": "c" * 64, "feature_columns": adapter.full_feature_columns()},
        "candidate_manifest": {"manifest_sha256": "d" * 64},
        "candidate_states": states, "assignment_file_sha256": "e" * 64,
        "audit": audit, "targets": train[list(adapter.TARGET_COLUMNS)].copy(),
        "transforms": transforms, "temperatures": temperatures,
        "parent_file_hashes": {str(router / "router_dataset.parquet"): adapter.file_hash(router / "router_dataset.parquet")},
    }
    monkeypatch.setattr(adapter, "load_training_inputs", lambda *args: inputs)
    monkeypatch.setattr(adapter, "load_validation_inputs", lambda *args: outputs)
    monkeypatch.setattr(adapter, "validate_manifest", lambda *args, **kwargs: comparator_manifest)
    monkeypatch.setattr(adapter, "validate_sealed_predictions", lambda *args: {"manifest_sha256": lr_manifest["manifest_sha256"]})
    monkeypatch.setattr(adapter, "apply_expert_temperature", lambda logits, state: _softmax(logits / state["temperature"]))
    monkeypatch.setattr(adapter, "predict_router_probabilities", _predict)
    return {"repo": repo, "root": root, "inputs": inputs, "outputs": outputs, "comparator_manifest": comparator_manifest}


def test_adapter_returns_all_variants_and_no_validation_labels(sources, monkeypatch):
    original = adapter.pq.read_table
    decoded = []
    def guarded(path, *args, **kwargs):
        path = Path(path)
        assert path.name not in {"validation_targets.parquet", "development_assignments.parquet"}
        assert "cleaned_test" not in str(path)
        decoded.append(path.name)
        return original(path, *args, **kwargs)
    calls = []
    def replay(matrix, state):
        calls.append((state["variant_id"], state["training_seed"], state["C"]))
        return _predict(matrix, state)
    monkeypatch.setattr(adapter.pq, "read_table", guarded)
    monkeypatch.setattr(adapter, "predict_router_probabilities", replay)
    result = adapter.load_capacity_inputs(sources["repo"], sources["root"])
    assert len(result["candidate_states"]) == 48
    assert len(set(calls)) == len(calls) == 48
    assert set(result["old_lr_predictions"].variant_id) == set(adapter.VARIANTS)
    assert len(result["old_lr_predictions"]) == 48 * 3
    assert list(result["validation"]) == [*adapter.TABLE_IDENTITY_COLUMNS, *adapter.full_feature_columns()]
    assert not {"target_state", "label_name", "label_id_dense"} & set(result["experts"])
    for seed, item in result["seed_inputs"].items():
        assert item["X_train"].shape == (4, 727)
        assert item["X_validation"].shape == (3, 727)
        assert item["X_train"].dtype == item["X_validation"].dtype == np.float64
        assert item["X_train"].flags.c_contiguous
        np.testing.assert_array_equal(item["y_train"], np.arange(4))
        assert not set(item["train_identity"].plot_idx) & set(item["validation_identity"].plot_idx)
    assert decoded == ["router_dataset.parquet", "expert_probabilities_and_features.parquet", *(f"{v}.parquet" for v in adapter.VARIANTS)]


def test_adapter_rejects_semantic_matrix_reconstruction_mismatch(sources):
    sources["inputs"]["transforms"][1] = deepcopy(sources["inputs"]["transforms"][1])
    sources["inputs"]["transforms"][1]["mean"][0] += .5
    with pytest.raises(ValueError, match="matrix does not exactly reproduce"):
        adapter.load_capacity_inputs(sources["repo"], sources["root"])


def test_adapter_rejects_validation_probability_mismatch(sources):
    frame = sources["outputs"]
    frame.at[0, "image_logits"] = np.asarray(frame.at[0, "image_logits"]).copy()
    frame.at[0, "image_logits"][0] += .5
    with pytest.raises(ValueError, match="calibrated probabilities differ"):
        adapter.load_capacity_inputs(sources["repo"], sources["root"])


def test_adapter_rejects_cached_lr_replay_mismatch(sources, monkeypatch):
    monkeypatch.setattr(adapter, "predict_router_probabilities", lambda matrix, state: np.roll(_predict(matrix, state), 1, axis=1))
    with pytest.raises(ValueError, match="LR probabilities do not reproduce"):
        adapter.load_capacity_inputs(sources["repo"], sources["root"])


def test_adapter_rejects_mixed_manifest_lineage(sources):
    sources["comparator_manifest"]["prediction_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="mixed development lineage"):
        adapter.load_capacity_inputs(sources["repo"], sources["root"])


def test_adapter_rejects_incomplete_lr_grid(sources):
    sources["inputs"]["candidate_states"].pop(next(iter(sources["inputs"]["candidate_states"])))
    with pytest.raises(ValueError, match="complete 48-candidate"):
        adapter.load_capacity_inputs(sources["repo"], sources["root"])

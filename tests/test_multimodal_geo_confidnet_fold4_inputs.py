from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from multimodal import geo_helpfulness_confidnet_fold4_inputs as inputs
from multimodal.geo_helpfulness_protocol import canonical_sha256


def payloads():
    rng = np.random.default_rng(951)
    result = []
    for split, count in (("train", 96), ("validation", 24)):
        labels = np.arange(count, dtype=np.int64) % 18
        payload = {
            "row_uid": np.asarray([f"{split}_{i:03}" for i in range(count)]),
            "plot_idx": np.asarray([f"{split}_plot_{i // 2:03}" for i in range(count)]),
            "producer_fold": np.full(count, 3, dtype=np.int64),
        }
        if split == "train":
            payload["labels"] = labels
        for prefix in ("image", "geo", "raw"):
            logits = rng.normal(scale=0.3, size=(count, 18))
            correct = {
                "image": np.arange(count) % 3 != 0,
                "geo": np.isin(np.arange(count) % 4, [0, 2]),
                "raw": np.isin(np.arange(count) % 4, [1, 2]),
            }[prefix]
            predicted = np.where(correct, labels, (labels + 1) % 18)
            logits[np.arange(count), predicted] += rng.uniform(2, 4, count)
            payload[f"{prefix}_logits"] = logits.astype(np.float64)
            payload[f"{prefix}_hidden"] = rng.uniform(size=(count, 256)).astype(np.float32)
        result.append(payload)
    return result


def test_fold4_only_fit_uses_existing_features_and_target_order():
    train, validation = payloads()
    result = inputs.build_inputs(train, validation, seed=2)
    arrays, states, metadata = (result[name] for name in ("arrays", "states", "metadata"))
    assert arrays["X_train"].shape == (96, 25)
    assert arrays["X_validation"].shape == (24, 25)
    assert arrays["X_train"].dtype == arrays["X_validation"].dtype == np.float64
    np.testing.assert_array_equal(arrays["y_train"], np.arange(96) % 4)
    assert metadata["feature_columns"] == list(inputs.numeric.output_feature_columns()[:25])
    identity = canonical_sha256([
        {"row_uid": str(uid), "training_seed": 2} for uid in train["row_uid"]
    ])
    assert metadata["fit_row_identity_sha256"] == identity
    for state in [*states["temperatures"].values(), states["transform"]]:
        assert state["fit_row_count"] == 96
        assert state["fit_row_identity_sha256"] == identity
        assert state["seed"] == 2
    assert metadata["validation_labels_accessed"] is False
    assert metadata["validation_features_used_for_fitting"] is False
    assert metadata["producer_fold"] == 3
    assert metadata["tcp_supervision_probability_basis"] == "native_t1_uncalibrated"
    json.dumps(states, allow_nan=False)
    json.dumps(metadata, allow_nan=False)


def test_validation_changes_do_not_affect_fit_or_training_inputs():
    train, validation = payloads()
    before = deepcopy((train, validation))
    first = inputs.build_inputs(train, validation, seed=1)
    changed = deepcopy(validation)
    for prefix in ("image", "geo", "raw"):
        changed[f"{prefix}_logits"] *= 2
        changed[f"{prefix}_hidden"] += 10
    second = inputs.build_inputs(train, changed, seed=1)
    assert first["states"] == second["states"]
    assert first["metadata"] == second["metadata"]
    np.testing.assert_array_equal(first["arrays"]["X_train"], second["arrays"]["X_train"])
    assert not np.array_equal(first["arrays"]["X_validation"], second["arrays"]["X_validation"])
    for actual, original in zip((train, validation), before):
        for key in actual:
            np.testing.assert_array_equal(actual[key], original[key])


def test_serialized_replay_is_exact_and_never_refits(monkeypatch):
    train, validation = payloads()
    first = inputs.build_inputs(train, validation, seed=4)
    saved = json.loads(json.dumps(first["states"]))

    def forbidden(*args, **kwargs):
        raise AssertionError("replay attempted a fit")

    monkeypatch.setattr(inputs.numeric, "fit_expert_temperature", forbidden)
    monkeypatch.setattr(inputs.numeric, "fit_router_feature_transform", forbidden)
    second = inputs.build_inputs(train, validation, seed=4, states=saved)
    assert first["states"] == second["states"]
    assert first["metadata"] == second["metadata"]
    for key, value in first["arrays"].items():
        np.testing.assert_array_equal(value, second["arrays"][key])
    second["states"]["transform"]["mean"][0] += 1
    assert saved == first["states"]


@pytest.mark.parametrize("key", ["labels", "target_state", "tcp_geo"])
def test_validation_supervision_is_rejected(key):
    train, validation = payloads()
    validation[key] = np.zeros(24)
    with pytest.raises(ValueError, match="allow-list"):
        inputs.build_inputs(train, validation, seed=1)


@pytest.mark.parametrize("split", [0, 1])
@pytest.mark.parametrize("fold", [-1, 0, 2])
def test_only_same_fold3_producer_is_accepted(split, fold):
    payload = payloads()
    payload[split]["producer_fold"][:] = fold
    with pytest.raises(ValueError, match="producer fold 3"):
        inputs.build_inputs(*payload, seed=1)


@pytest.mark.parametrize("key", ["row_uid", "plot_idx"])
def test_split_identity_overlap_is_rejected(key):
    train, validation = payloads()
    validation[key][0] = train[key][0]
    with pytest.raises(ValueError, match="overlap"):
        inputs.build_inputs(train, validation, seed=1)


@pytest.mark.parametrize("component", ["temperature", "transform"])
@pytest.mark.parametrize("field,value", [
    ("seed", 2), ("fit_row_count", 852), ("fit_row_identity_sha256", "f" * 64),
])
def test_replay_rejects_foreign_fit_provenance(component, field, value):
    train, validation = payloads()
    states = inputs.build_inputs(train, validation, seed=1)["states"]
    target = states["transform"] if component == "transform" else states["temperatures"]["image_only"]
    target[field] = value
    with pytest.raises(ValueError, match="fit identity"):
        inputs.build_inputs(train, validation, seed=1, states=states)


def test_replay_rejects_changed_training_labels():
    train, validation = payloads()
    states = inputs.build_inputs(train, validation, seed=1)["states"]
    train["labels"] = (train["labels"] + 2) % 18
    with pytest.raises(ValueError, match="NLL"):
        inputs.build_inputs(train, validation, seed=1, states=states)


def test_replay_rejects_changed_training_logits():
    train, validation = payloads()
    states = inputs.build_inputs(train, validation, seed=1)["states"]
    train["image_logits"] *= 2
    with pytest.raises(ValueError, match="NLL"):
        inputs.build_inputs(train, validation, seed=1, states=states)


def test_replay_rejects_changed_transform_moments():
    train, validation = payloads()
    states = inputs.build_inputs(train, validation, seed=1)["states"]
    states["transform"]["mean"][3] += 0.01
    with pytest.raises(ValueError, match="moments"):
        inputs.build_inputs(train, validation, seed=1, states=states)


def test_replay_rejects_wrong_expert_mode():
    train, validation = payloads()
    states = inputs.build_inputs(train, validation, seed=1)["states"]
    states["temperatures"]["image_only"]["mode"] = "geo_only"
    with pytest.raises(ValueError, match="expert mode"):
        inputs.build_inputs(train, validation, seed=1, states=states)


@pytest.mark.parametrize("seed", [0, 5, True, 1.0])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed must"):
        inputs.build_inputs(*payloads(), seed=seed)

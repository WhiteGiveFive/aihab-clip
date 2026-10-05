from __future__ import annotations

from copy import deepcopy
import inspect
import json

import numpy as np
import pytest
import torch

from multimodal import geo_helpfulness_confidnet_oof_fit as fit
from multimodal.geo_helpfulness_router_capacity_fit import fit_candidate
from multimodal.geo_helpfulness_router_candidate_state import full_feature_columns


def confidence_data():
    rng = np.random.default_rng(72)
    hidden = rng.normal(size=(72, 4)).astype(np.float32)
    # Includes successes and failures for the fixed zero label.
    logits = np.column_stack((1.5 * hidden[:, 0], np.zeros(len(hidden)))).astype(
        np.float64
    )
    labels = np.zeros(len(hidden), dtype=np.int64)
    validation = rng.normal(size=(24, 4)).astype(np.float32)
    return hidden, logits, labels, validation


def parameters(**kwargs):
    return {"hidden_widths": [12, 12, 12, 12], "epochs": 5, "batch_size": 32, **kwargs}


def run_confidence(**kwargs):
    hidden, logits, labels, val = confidence_data()
    return fit.fit_confidence(
        hidden,
        logits,
        labels,
        {"validation": val},
        seed=2,
        parameters=parameters(**kwargs),
    )


def test_declared_defaults_match_approved_recipe():
    result = fit.resolve_confidence_parameters()
    assert result["hidden_widths"] == [400] * 4
    assert result["epochs"] == 500
    assert result["learning_rate"] == 1e-4
    assert result["batch_size"] == 128
    assert result["zero_gradient_patience"] == 5
    assert result["prediction_convention"] == "native_float32"
    assert result["betas"] == [0.9, 0.999]
    assert result["eps"] == 1e-8
    assert result["weight_decay"] == result["dropout"] == 0
    assert result["num_threads"] == 1
    assert result["checkpoint"] == "final_epoch"
    assert fit.RF_DEFAULTS["n_estimators"] == 500
    assert fit.RF_DEFAULTS["max_depth"] == 4
    assert fit.RF_DEFAULTS["min_samples_leaf"] == 5


def test_confidence_states_and_native_predictions_replay_exactly():
    result = run_confidence()
    hidden, logits, labels, val = confidence_data()
    state = json.loads(json.dumps(result["state"], allow_nan=False))
    assert result["status"] == "complete"
    assert len(result["history"]) == state["completed_epochs"] == 5
    for name, x in (("train", hidden), ("validation", val)):
        native = result["predictions"][name]
        assert native.dtype == np.float32
        assert np.array_equal(native, fit.predict_confidence(state, x))
        assert np.allclose(
            native, fit.predict_confidence(state, x, native=False), atol=1e-6, rtol=0
        )
    assert result["metadata"]["training_errors"] == int(
        (logits.argmax(axis=1) != labels).sum()
    )
    assert result["metadata"]["always_one_mse"] > 0
    assert result["metadata"]["validation_labels_accessed"] is False
    assert result["metadata"]["validation_features_used_for_fitting"] is False
    assert all(v["maximum_absolute_gradient"] > 0 for v in result["history"])
    assert all(v["consecutive_zero_gradient_epochs"] == 0 for v in result["history"])


def test_validation_inputs_do_not_affect_fit_and_inputs_stay_unchanged():
    hidden, logits, labels, val = confidence_data()
    copies = [v.copy() for v in (hidden, logits, labels, val)]
    first = fit.fit_confidence(
        hidden, logits, labels, {"validation": val}, seed=2, parameters=parameters()
    )
    second = fit.fit_confidence(
        hidden,
        logits,
        labels,
        {"validation": val + 0.25},
        seed=2,
        parameters=parameters(),
    )
    assert first["state"] == second["state"]
    assert first["history"] == second["history"]
    assert np.array_equal(first["predictions"]["train"], second["predictions"]["train"])
    for actual, before in zip((hidden, logits, labels, val), copies):
        assert np.array_equal(actual, before)


def test_simple_tcp_signal_learns_without_validation_targets():
    result = run_confidence(hidden_widths=[16, 16], epochs=120, learning_rate=0.005)
    _, _, _, val = confidence_data()
    expected = 1 / (1 + np.exp(-1.5 * val[:, 0]))
    assert result["history"][-1]["train_mse"] < 0.001
    assert np.mean((result["predictions"]["validation"] - expected) ** 2) < 0.01
    assert "validation_labels" not in inspect.signature(fit.fit_confidence).parameters


def test_torch_global_state_restored_after_fit_and_replay():
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    before_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    result = run_confidence()
    fit.predict_confidence(result["state"], confidence_data()[0])
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.get_num_threads() == before_threads
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == before_warn


def test_zero_gradient_pauses_after_exactly_five_epochs_and_skips_validation(
    monkeypatch,
):
    network = fit.initial._network

    def saturated(width, hidden_widths):
        model = network(width, hidden_widths)
        with torch.no_grad():
            model[-2].weight.zero_()
            model[-2].bias.fill_(1000)
        return model

    monkeypatch.setattr(fit.initial, "_network", saturated)
    result = run_confidence(epochs=12)
    assert result["status"] == "paused_zero_gradient"
    assert result["state"]["completed_epochs"] == len(result["history"]) == 5
    assert set(result["predictions"]) == {"train"}
    assert all(v["maximum_absolute_gradient"] == 0 for v in result["history"])
    assert result["history"][-1]["consecutive_zero_gradient_epochs"] == 5
    assert result["history"][-1]["prediction_summary"]["fraction_near_one"] == 1
    json.dumps(result["state"], allow_nan=False)


def test_nonfinite_update_aborts_and_preserves_last_finite_state(monkeypatch):
    def poison(optimizer, closure=None):
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))

    monkeypatch.setattr(torch.optim.Adam, "step", poison)
    with pytest.raises(fit.ConfidNetOOFFitError, match="nonfinite parameter") as error:
        run_confidence()
    partial = error.value.partial_result
    assert partial["status"] == "aborted_nonfinite"
    assert partial["history"] == []
    assert partial["predictions"] == {}
    assert partial["state"]["completed_epochs"] == 0
    json.dumps(partial["state"], allow_nan=False)
    assert np.isfinite(
        fit.predict_confidence(partial["state"], confidence_data()[0])
    ).all()


@pytest.mark.parametrize(
    "amendment",
    [
        {"weight_decay": 0.01},
        {"dropout": 0.1},
        {"optimizer": "sgd"},
        {"normalization": "standard"},
        {"zero_gradient_patience": 4},
        {"prediction_convention": "portable"},
        {"unknown": 1},
        {"epochs": 0},
        {"learning_rate": float("nan")},
    ],
)
def test_unauthorized_or_invalid_confidence_recipe_rejected(amendment):
    with pytest.raises(fit.ConfidNetOOFFitError):
        fit.resolve_confidence_parameters(amendment)


@pytest.mark.parametrize(
    "problem", ["label", "hidden_nan", "logits_nan", "val_width", "train_split"]
)
def test_invalid_supervision_or_prediction_inputs_rejected(problem):
    hidden, logits, labels, val = confidence_data()
    splits = {"validation": val}
    if problem == "label":
        labels[0] = 2
    elif problem == "hidden_nan":
        hidden[0, 0] = np.nan
    elif problem == "logits_nan":
        logits[0, 0] = np.nan
    elif problem == "val_width":
        splits["validation"] = val[:, :3]
    else:
        splits["train"] = hidden
    with pytest.raises(fit.ConfidNetOOFFitError):
        fit.fit_confidence(
            hidden, logits, labels, splits, seed=1, parameters=parameters()
        )


def router_data(augmented=False):
    rng = np.random.default_rng(18)
    names = full_feature_columns()[:25] + (fit.TCP_FEATURE_NAMES if augmented else [])
    train = rng.normal(size=(96, len(names))).astype(np.float64)
    validation = rng.normal(size=(20, len(names))).astype(np.float64)
    target = np.tile(np.arange(4, dtype=np.int64), 24)
    return train, target, validation, names


@pytest.mark.parametrize("augmented", [False, True])
def test_rf_json_replay_matches_native(augmented):
    train, target, validation, names = router_data(augmented)
    result = fit.fit_router(
        train, target, {"validation": validation}, seed=1, feature_names=names
    )
    state = json.loads(json.dumps(result["state"], allow_nan=False))
    assert len(state["trees"]) == 500
    assert state["target_order"] == ["rescue", "harm", "both_correct", "both_wrong"]
    assert result["metadata"]["feature_count"] == (28 if augmented else 25)
    assert result["metadata"]["validation_labels_accessed"] is False
    for name, matrix in (("train", train), ("validation", validation)):
        assert np.allclose(
            result["predictions"][name],
            fit.predict_router(state, matrix),
            atol=1e-12,
            rtol=0,
        )


def test_numeric25_rf_reproduces_existing_frozen_capacity_implementation():
    train, target, validation, names = router_data()
    new = fit.fit_router(
        train, target, {"validation": validation}, seed=3, feature_names=names
    )
    full_train = np.pad(train, ((0, 0), (0, 702)))
    full_val = np.pad(validation, ((0, 0), (0, 702)))
    spec = {
        "config_id": "rf_numeric_depth4_leaf5",
        "family": "rf",
        "variant": "ablation_numeric25",
        "parameters": {"max_depth": 4, "min_samples_leaf": 5},
    }
    old_state, old_train, old_val, _ = fit_candidate(
        full_train, target, full_val, spec, 3
    )
    assert new["state"]["trees"] == old_state["model"]["trees"]
    assert np.array_equal(fit.predict_router(new["state"], train), old_train)
    assert np.array_equal(fit.predict_router(new["state"], validation), old_val)
    assert np.allclose(new["predictions"]["train"], old_train, atol=1e-12, rtol=0)
    assert np.allclose(new["predictions"]["validation"], old_val, atol=1e-12, rtol=0)


def test_rf_fit_is_independent_of_validation_features():
    train, target, validation, names = router_data()
    first = fit.fit_router(
        train, target, {"validation": validation}, seed=4, feature_names=names
    )
    second = fit.fit_router(
        train, target, {"validation": validation + 0.5}, seed=4, feature_names=names
    )
    assert first["state"] == second["state"]
    assert np.array_equal(first["predictions"]["train"], second["predictions"]["train"])


def test_state_tampering_and_invalid_rf_contract_rejected():
    train, target, validation, names = router_data()
    result = fit.fit_router(train, target, {}, seed=1, feature_names=names)
    bad = deepcopy(result["state"])
    bad["trees"][0]["threshold"][0] += 0.5
    with pytest.raises(fit.ConfidNetOOFFitError, match="hash"):
        fit.predict_router(bad, validation)
    with pytest.raises(fit.ConfidNetOOFFitError, match="feature order"):
        fit.fit_router(train, target, {}, seed=1, feature_names=list(reversed(names)))
    with pytest.raises(fit.ConfidNetOOFFitError, match="recipe"):
        fit.fit_router(
            train,
            target,
            {},
            seed=1,
            feature_names=names,
            parameters={"n_estimators": 10},
        )
    with pytest.raises(fit.ConfidNetOOFFitError, match="four"):
        fit.fit_router(train, np.zeros_like(target), {}, seed=1, feature_names=names)


def test_confidence_state_tampering_is_rejected():
    state = run_confidence()["state"]
    state["layers"][0]["weight"][0][0] += 0.1
    with pytest.raises(fit.ConfidNetOOFFitError, match="hash"):
        fit.predict_confidence(state, confidence_data()[0])

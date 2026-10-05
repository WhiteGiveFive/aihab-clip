from __future__ import annotations

from copy import deepcopy
import inspect
import json

import numpy as np
import pytest
import torch

from multimodal import geo_helpfulness_confidnet_fit as fit


def _data():
    rng = np.random.default_rng(812)
    hidden = rng.normal(size=(80, 4)).astype(np.float64)
    tcp = 1.0 / (1.0 + np.exp(-1.5 * hidden[:, 0]))
    validation = rng.normal(size=(24, 4)).astype(np.float64)
    return hidden, tcp, validation


def _config(**kwargs):
    return {"hidden_widths": [12, 12, 12, 12], "epochs": 6, **kwargs}


def test_final_epoch_state_round_trips_as_json_and_replays_independently():
    training, tcp, validation = _data()
    before = [value.copy() for value in (training, tcp, validation)]
    result = fit.fit_confidnet(training, tcp, validation, seed=3, config=_config())
    state = json.loads(json.dumps(result["state"], allow_nan=False))
    metadata = result["metadata"]

    assert len(state["layers"]) == 5
    assert len(result["history"]) == 6
    assert [row["epoch"] for row in result["history"]] == list(range(1, 7))
    assert metadata["checkpoint"] == "final_epoch"
    assert metadata["final_epoch"] == 6
    assert metadata["validation_labels_accessed"] is False
    assert metadata["validation_features_used_for_fitting"] is False
    assert metadata["expert_parameters_updated"] is False
    assert metadata["training_dtype"] == "float32"
    assert metadata["num_threads"] == 1
    assert metadata["deterministic_algorithms"] is True
    assert metadata["runtime_seconds"] >= 0
    assert state["parameters"]["weight_decay"] == 0.0
    assert state["parameters"]["optimizer"] == "adam"
    assert state["parameters"]["betas"] == [0.9, 0.999]
    assert state["parameters"]["eps"] == 1e-8
    for split, matrix in (("train", training), ("validation", validation)):
        prediction = result[f"{split}_prediction"]
        assert prediction.dtype == np.float64
        assert prediction.shape == (len(matrix),)
        assert np.all((prediction >= 0) & (prediction <= 1))
        assert np.array_equal(prediction, fit.predict_confidnet(state, matrix))
        assert metadata["prediction_parity"][split]["maximum_abs_error"] <= 1e-6
    for original, unchanged in zip((training, tcp, validation), before):
        assert np.array_equal(original, unchanged)


def test_repeat_is_identical_and_validation_features_cannot_change_training():
    training, tcp, validation = _data()
    first = fit.fit_confidnet(training, tcp, validation, seed=4, config=_config())
    second = fit.fit_confidnet(
        training, tcp, validation + 0.25, seed=4, config=_config()
    )
    assert first["state"] == second["state"]
    assert first["history"] == second["history"]
    assert np.array_equal(first["train_prediction"], second["train_prediction"])


def test_simple_reliability_signal_learns_and_generalizes():
    training, tcp, validation = _data()
    expected = 1.0 / (1.0 + np.exp(-1.5 * validation[:, 0]))
    result = fit.fit_confidnet(
        training,
        tcp,
        validation,
        seed=1,
        config=_config(
            hidden_widths=[16, 16], epochs=100, batch_size=32, learning_rate=0.01
        ),
    )
    assert result["history"][-1]["train_mse"] < 0.001
    assert result["history"][-1]["train_mse"] < result["history"][0]["train_mse"]
    assert np.mean((result["validation_prediction"] - expected) ** 2) < 0.01


def test_torch_rng_and_execution_settings_are_restored():
    training, tcp, validation = _data()
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    before_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fit.fit_confidnet(training, tcp, validation, seed=1, config=_config())
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.get_num_threads() == before_threads
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == before_warn_only


def test_validation_labels_are_not_an_interface_parameter():
    names = set(inspect.signature(fit.fit_confidnet).parameters)
    assert names == {"train_hidden", "train_tcp", "validation_hidden", "seed", "config"}
    training, tcp, validation = _data()
    with pytest.raises(TypeError):
        fit.fit_confidnet(
            training,
            tcp,
            validation,
            seed=1,
            config=_config(),
            validation_tcp=tcp[: len(validation)],
        )


@pytest.mark.parametrize(
    "problem",
    [
        "empty_training",
        "validation_width",
        "hidden_nan",
        "hidden_integer",
        "float32_overflow",
        "tcp_nan",
        "tcp_range",
        "tcp_shape",
        "bool_seed",
        "negative_seed",
        "unknown_config",
        "validation_selection",
        "early_stopping",
        "weight_decay",
        "optimizer",
        "bad_width",
        "bad_epoch",
        "bad_rate",
    ],
)
def test_bad_inputs_fail_before_network_construction(problem, monkeypatch):
    training, tcp, validation = _data()
    config, seed = _config(), 1
    if problem == "empty_training":
        training = training[:0]
    elif problem == "validation_width":
        validation = validation[:, :2]
    elif problem == "hidden_nan":
        training[0, 0] = np.nan
    elif problem == "hidden_integer":
        training = training.astype(np.int64)
    elif problem == "float32_overflow":
        training[0, 0] = 1e100
    elif problem == "tcp_nan":
        tcp[0] = np.nan
    elif problem == "tcp_range":
        tcp[0] = 1.001
    elif problem == "tcp_shape":
        tcp = tcp[:, None]
    elif problem == "bool_seed":
        seed = True
    elif problem == "negative_seed":
        seed = -1
    elif problem == "unknown_config":
        config["validation_target"] = [0.0]
    elif problem == "validation_selection":
        config["checkpoint"] = "best_validation"
    elif problem == "early_stopping":
        config["early_stopping"] = True
    elif problem == "weight_decay":
        config["weight_decay"] = 0.01
    elif problem == "optimizer":
        config["optimizer"] = "adamw"
    elif problem == "bad_width":
        config["hidden_widths"] = [0]
    elif problem == "bad_epoch":
        config["epochs"] = 0
    elif problem == "bad_rate":
        config["learning_rate"] = np.nan

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid inputs reached network construction")

    monkeypatch.setattr(fit, "_network", forbidden)
    with pytest.raises(fit.ConfidNetFitError):
        fit.fit_confidnet(training, tcp, validation, seed=seed, config=config)


@pytest.mark.parametrize("problem", ["dimensions", "nonfinite", "layer_count", "recipe"])
def test_portable_replay_rejects_malformed_model(problem):
    training, tcp, validation = _data()
    result = fit.fit_confidnet(training, tcp, validation, seed=1, config=_config())
    state = deepcopy(result["state"])
    if problem == "dimensions":
        state["layers"][0]["weight"][0] = [0.0]
    elif problem == "nonfinite":
        state["layers"][0]["bias"][0] = float("nan")
    elif problem == "layer_count":
        state["layers"].pop()
    else:
        state["parameters"]["output_activation"] = "identity"
    with pytest.raises(fit.ConfidNetFitError):
        fit.predict_confidnet(state, validation)


def test_constant_nearly_perfect_tcp_is_valid_training_data():
    training, _, validation = _data()
    tcp = np.ones(len(training), dtype=np.float64)
    result = fit.fit_confidnet(training, tcp, validation, seed=1, config=_config())
    assert np.isfinite(result["train_prediction"]).all()
    assert result["history"][-1]["train_mse"] < result["history"][0]["train_mse"]

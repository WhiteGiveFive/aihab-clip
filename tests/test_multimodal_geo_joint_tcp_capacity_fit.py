"""Check deeper joint-router updates, gradient paths, isolation and replay."""

from copy import deepcopy
import inspect
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from multimodal import geo_helpfulness_joint_tcp_capacity_fit as fit
from multimodal import geo_helpfulness_joint_tcp_fit as parent_fit


def _data(width=25):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(35, width)).astype(np.float32)
    y = (2 * (x[:, 0] > 0) + (x[:, 1] > 0)).astype(np.int64)
    tcp = (1 / (1 + np.exp(-x[:, :2]))).astype(np.float32)
    validation = rng.normal(size=(13, width)).astype(np.float32)
    return x, y, tcp, validation


def _fit(widths=(64, 32), auxiliary_weight=1., input_dim=25):
    x, y, tcp, validation = _data(input_dim)
    return fit.fit_joint(
        x, y, tcp, {"validation": validation}, seed=3, config_id="test",
        auxiliary_weight=auxiliary_weight,
        parameters={"hidden_widths": list(widths), "epochs": 3, "batch_size": 16},
    )


@pytest.mark.parametrize("widths", fit.APPROVED_HIDDEN_WIDTHS)
@pytest.mark.parametrize("input_dim", [25, 79])
def test_deeper_serialization_replays_exact_native_predictions(widths, input_dim):
    result = _fit(widths, input_dim=input_dim)
    state = json.loads(json.dumps(result["state"], allow_nan=False))
    fit.validate_state(state)
    assert result["metadata"]["parameter_count"] == fit.parameter_count(input_dim, list(widths))
    assert state["parameters"]["hidden_widths"] == list(widths)
    assert len(state["layers"]) == len(widths) + 2
    x, _, _, validation = _data(input_dim)
    for split, values in (("train", x), ("validation", validation)):
        native = fit.predict_joint(state, values)
        diagnostic = fit.predict_joint(state, values, native=False)
        for key in fit.PREDICTION_KEYS:
            assert native[key].dtype == np.float32
            assert diagnostic[key].dtype == np.float64
            assert np.array_equal(native[key], result["predictions"][split][key])
            assert np.max(np.abs(native[key].astype(np.float64) - diagnostic[key])) < 1e-5
        np.testing.assert_allclose(native["q"].sum(1), 1., atol=1e-7)
        assert np.all((native["tcp_hat"] >= 0.) & (native["tcp_hat"] <= 1.))
        assert result["metadata"]["prediction_parity"][split]["native_float32_exact"]
    for row in result["history"]:
        assert row["first_batch_shared_ce_gradient_norm"] > 0
        assert row["first_batch_shared_tcp_gradient_norm"] > 0
        assert row["train_weighted_auxiliary_loss"] == row["train_mse"]


@pytest.mark.parametrize("weight", [0., .1, 1., 10.])
def test_single_layer_matches_sealed_parent_updates_and_history_exactly(weight):
    """Changing the implementation alone must not change the existing control."""
    x, y, tcp, validation = _data()
    control = parent_fit.fit_joint(
        x, y, tcp, {"validation": validation}, seed=3, config_id="test",
        auxiliary_weight=weight, parameters={"epochs": 3, "batch_size": 16},
    )
    updated = _fit((32,), weight)
    assert updated["history"] == control["history"]
    for name, layer in control["state"]["layers"].items():
        assert updated["state"]["layers"]["trunk_0" if name == "trunk" else name] == layer
    for split in ("train", "validation"):
        for key in fit.PREDICTION_KEYS:
            assert np.array_equal(updated["predictions"][split][key],
                                  control["predictions"][split][key])


@pytest.mark.parametrize("widths", fit.APPROVED_HIDDEN_WIDTHS)
def test_lambda_zero_matches_independent_ce_only_training(widths):
    x, y, _, _ = _data()
    result = _fit(widths, 0.)
    parameters = result["state"]["parameters"]
    with fit._execution():
        torch.manual_seed(3)
        model = fit._JointNetwork(x.shape[1], parameters)
        before_tcp = deepcopy(model.tcp_head.state_dict())
        optimizer = fit._optimizer(model, parameters)
        shuffle = torch.Generator(device="cpu").manual_seed(3)
        for _ in range(parameters["epochs"]):
            model.train()
            for rows in torch.randperm(len(x), generator=shuffle).split(16):
                optimizer.zero_grad(set_to_none=True)
                state_logits, _ = model(torch.from_numpy(x[rows]))
                F.cross_entropy(state_logits, torch.from_numpy(y[rows])).backward()
                assert model.tcp_head.weight.grad is None
                optimizer.step()
        for name, layer in fit._layers(model).items():
            assert np.array_equal(layer.weight.detach().numpy(),
                                  result["state"]["layers"][name]["weight"])
            assert np.array_equal(layer.bias.detach().numpy(),
                                  result["state"]["layers"][name]["bias"])
        assert all(torch.equal(before_tcp[k], v) for k, v in model.tcp_head.state_dict().items())
    assert not result["metadata"]["tcp_supervised"]
    assert result["metadata"]["initial_tcp_head_sha256"] == result["metadata"]["final_tcp_head_sha256"]
    assert all(row["maximum_tcp_head_gradient_norm"] == 0. for row in result["history"])


@pytest.mark.parametrize("widths", fit.APPROVED_HIDDEN_WIDTHS)
def test_tcp_auxiliary_gradients_reach_every_shared_layer(widths):
    x, y, tcp, _ = _data()
    with fit._execution():
        torch.manual_seed(3)
        model = fit._JointNetwork(x.shape[1], fit.resolve_parameters({"hidden_widths": list(widths)}))
        z, t = model(torch.from_numpy(x))
        _, _, mse, loss = fit._components(z, t, torch.from_numpy(y), torch.from_numpy(tcp), 10.)
        # Mean over experts and then images can round differently from one flat mean.
        torch.testing.assert_close(loss, F.cross_entropy(z, torch.from_numpy(y)) +
                                   10 * F.mse_loss(t.sigmoid(), torch.from_numpy(tcp)),
                                   rtol=1e-7, atol=1e-7)
        mse.backward()
        for layer in model.trunk:
            assert layer.weight.grad.abs().sum() > 0
            assert layer.bias.grad.abs().sum() > 0
        assert model.tcp_head.weight.grad.abs().sum() > 0
        assert model.state_head.weight.grad is None


def test_validation_features_do_not_affect_fit_and_caller_settings_are_restored():
    x, y, tcp, validation = _data()
    originals = [a.copy() for a in (x, y, tcp, validation)]
    rng_before = torch.get_rng_state().clone()
    threads_before = torch.get_num_threads()
    first = _fit()
    second = fit.fit_joint(
        x, y, tcp, {"validation": validation + 999}, seed=3, config_id="test",
        auxiliary_weight=1., parameters={"epochs": 3, "batch_size": 16},
    )
    assert first["state"] == second["state"]
    assert first["history"] == second["history"]
    assert torch.equal(rng_before, torch.get_rng_state())
    assert threads_before == torch.get_num_threads()
    assert all(np.array_equal(a, b) for a, b in zip(originals, (x, y, tcp, validation)))
    assert set(inspect.signature(fit.fit_joint).parameters) == {
        "train_x", "state_targets", "tcp_targets", "predict_x_by_split", "seed",
        "config_id", "auxiliary_weight", "parameters",
    }


def test_initialization_and_dropout_match_across_lambda_and_shuffle_across_architectures(monkeypatch):
    original = fit._train_epoch
    rng_traces = []
    active_trace = []

    def capture(*args, **kwargs):
        active_trace.append(torch.get_rng_state().clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(fit, "_train_epoch", capture)
    zero = _fit(auxiliary_weight=0.)
    rng_traces.append(active_trace[:])
    active_trace.clear()
    auxiliary = _fit(auxiliary_weight=10.)
    rng_traces.append(active_trace[:])
    assert all(torch.equal(a, b) for a, b in zip(*rng_traces))
    assert zero["metadata"]["initial_layers_sha256"] == auxiliary["metadata"]["initial_layers_sha256"]
    assert zero["metadata"]["initial_dropout_rng_sha256"] == auxiliary["metadata"]["initial_dropout_rng_sha256"]
    assert zero["state"]["layers"]["trunk_0"] != auxiliary["state"]["layers"]["trunk_0"]
    hashes = {_fit(widths)["metadata"]["shuffle_order_sha256"] for widths in fit.APPROVED_HIDDEN_WIDTHS}
    assert hashes == {zero["metadata"]["shuffle_order_sha256"]}


@pytest.mark.parametrize("override", [
    {"hidden_widths": []}, {"hidden_widths": [64]}, {"hidden_widths": [64, 32.]},
    {"hidden_widths": (64, 32)}, {"hidden_widths": [True, 32]}, {"hidden_width": 32},
    {"learning_rate": .01}, {"dropout": .2}, {"epochs": 0}, {"batch_size": True},
])
def test_unapproved_recipe_is_rejected(override):
    with pytest.raises(fit.JointTCPFitError):
        fit.resolve_parameters(override)


@pytest.mark.parametrize("change", ["hash", "layer", "depth", "target_order", "bias_shape"])
def test_serialized_state_tampering_is_rejected(change):
    state = _fit()["state"]
    if change == "hash":
        state["state_sha256"] = "0" * 64
    elif change == "layer":
        state["layers"]["trunk_0"]["weight"][0][0] += .1
    elif change == "depth":
        state["parameters"]["hidden_widths"] = [64, 32, 16]
    elif change == "target_order":
        state["target_order"] = state["target_order"][::-1]
    elif change == "bias_shape":
        state["layers"]["trunk_1"]["bias"].pop()
        state["state_sha256"] = fit._hash({k: v for k, v in state.items() if k != "state_sha256"})
    with pytest.raises(fit.JointTCPFitError):
        fit.validate_state(state)


def test_nonfinite_training_preserves_last_complete_epoch(monkeypatch):
    original = fit._train_epoch

    def fail_on_second(*args, **kwargs):
        if kwargs["epoch"] == 2:
            with torch.no_grad():
                args[0].trunk[-1].weight.fill_(float("inf"))
            raise fit.JointTCPFitError("injected nonfinite parameter")
        return original(*args, **kwargs)

    monkeypatch.setattr(fit, "_train_epoch", fail_on_second)
    with pytest.raises(fit.JointTCPFitError) as captured:
        _fit()
    partial = captured.value.partial_result
    assert partial["completed_epochs"] == 1
    assert partial["failed_epoch"] == 2
    fit.validate_state(partial["state"])
    assert np.isfinite(fit.predict_joint(partial["state"], _data()[3])["q"]).all()

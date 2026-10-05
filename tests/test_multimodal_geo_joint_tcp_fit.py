"""Behavioral checks for the frozen joint-router optimization/replay contract."""

from copy import deepcopy
import inspect
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits
import torch
from torch.nn import functional as F

from multimodal import geo_helpfulness_joint_tcp_fit as fit


def _data():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(44, 7)).astype(np.float64)
    y = (2 * (x[:, 0] > 0) + (x[:, 1] > 0)).astype(np.int64)
    tcp = 1 / (1 + np.exp(-x[:, :2]))
    validation = rng.normal(size=(17, 7)).astype(np.float64)
    return x, y, tcp, validation


def _fit(auxiliary_weight=1.0, **kwargs):
    x, y, tcp, validation = _data()
    return fit.fit_joint(
        x, y, tcp, {"validation": validation}, seed=3, config_id="test", auxiliary_weight=auxiliary_weight,
        parameters={"epochs": 4, "batch_size": 16}, **kwargs,
    )


def test_json_state_replays_native_predictions_exactly_and_diagnoses_float64():
    x, _, _, validation = _data()
    result = _fit()
    state = json.loads(json.dumps(result["state"], allow_nan=False))
    fit.validate_state(state)
    assert len(result["history"]) == 4
    assert result["metadata"]["tcp_supervised"] is True
    assert state["target_order"] == ["rescue", "harm", "both_correct", "both_wrong"]
    assert state["tcp_order"] == ["raw_concat", "geo_only"]
    assert state["input_dim"] == 7
    assert state["layers"]["trunk"]["weight"] and len(state["layers"]["trunk"]["weight"]) == 32
    for split, values in (("train", x), ("validation", validation)):
        original = result["predictions"][split]
        native = fit.predict_joint(state, values)
        diagnostic = fit.predict_joint(state, values, native=False)
        for key in fit.PREDICTION_KEYS:
            assert original[key].dtype == np.float32
            assert diagnostic[key].dtype == np.float64
            assert np.array_equal(original[key], native[key])
            assert np.max(np.abs(native[key] - diagnostic[key])) < 1e-5
        np.testing.assert_allclose(native["q"].sum(axis=1), 1, atol=1e-7)
        assert np.all((native["q"] >= 0) & (native["q"] <= 1))
        assert np.all((native["tcp_hat"] >= 0) & (native["tcp_hat"] <= 1))
        assert result["metadata"]["prediction_parity"][split]["native_float32_exact"]
    for row in result["history"]:
        assert row["train_weighted_auxiliary_loss"] == row["train_mse"]
        assert row["first_batch_shared_ce_gradient_norm"] > 0
        assert row["first_batch_shared_tcp_gradient_norm"] > 0
        assert row["maximum_shared_gradient_norm"] > 0


def test_repeat_and_validation_changes_leave_fit_and_rng_unchanged():
    x, y, tcp, validation = _data()
    originals = [array.copy() for array in (x, y, tcp, validation)]
    rng_before = torch.get_rng_state().clone()
    settings_before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                       torch.is_deterministic_algorithms_warn_only_enabled())
    first = _fit()
    second = fit.fit_joint(x, y, tcp, {"validation": validation + 999}, seed=3,
                          config_id="test", auxiliary_weight=1., parameters={"epochs": 4, "batch_size": 16})
    assert first["state"] == second["state"]
    assert first["history"] == second["history"]
    assert torch.equal(rng_before, torch.get_rng_state())
    assert settings_before == (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                               torch.is_deterministic_algorithms_warn_only_enabled())
    for before, after in zip(originals, (x, y, tcp, validation)):
        assert np.array_equal(before, after)
    assert first["metadata"]["validation_labels_accessed"] is False
    assert first["metadata"]["validation_features_used_for_fitting"] is False
    assert first["metadata"]["expert_parameters_updated"] is False


@pytest.mark.parametrize("caller_threads", [1, 4])
def test_float64_diagnostics_are_stable_under_caller_thread_settings(caller_threads):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(48, 1295)).astype(np.float32)
    y = np.arange(len(x), dtype=np.int64) % 4
    tcp = rng.uniform(size=(len(x), 2)).astype(np.float32)
    validation = rng.normal(size=(31, x.shape[1])).astype(np.float32)
    result = fit.fit_joint(
        x, y, tcp, {"validation": validation}, seed=1, config_id="wide_test",
        auxiliary_weight=1., parameters={"epochs": 1, "batch_size": 16},
    )
    with threadpool_limits(limits=caller_threads):
        before = {row["filepath"]: row["num_threads"] for row in threadpool_info()}
        for split, inputs in (("train", x), ("validation", validation)):
            diagnostic = fit.predict_joint(result["state"], inputs, native=False)
            actual = fit._parity_summary(result["predictions"][split], diagnostic)
            assert actual == result["metadata"]["prediction_parity"][split]
        after = {row["filepath"]: row["num_threads"] for row in threadpool_info()}
        assert after == before


def test_lambda_zero_exactly_matches_standalone_ce_optimization():
    x, y, _, _ = _data()
    result = _fit(0.)
    parameters = result["state"]["parameters"]
    with fit._execution():
        torch.manual_seed(3)
        model = fit._JointNetwork(x.shape[1], parameters)
        initial_tcp = deepcopy(model.tcp_head.state_dict())
        optimizer = fit._optimizer(model, parameters)
        shuffle = torch.Generator(device="cpu").manual_seed(3)
        data = torch.from_numpy(x.astype(np.float32))
        labels = torch.from_numpy(y)
        for _ in range(parameters["epochs"]):
            model.train()
            for rows in torch.randperm(len(x), generator=shuffle).split(parameters["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                state_logits, _ = model(data[rows])
                F.cross_entropy(state_logits, labels[rows]).backward()
                assert model.tcp_head.weight.grad is None
                optimizer.step()
        for name in ("trunk", "state_head", "tcp_head"):
            layer = getattr(model, name)
            assert np.array_equal(layer.weight.detach().numpy(), result["state"]["layers"][name]["weight"])
            assert np.array_equal(layer.bias.detach().numpy(), result["state"]["layers"][name]["bias"])
        assert all(torch.equal(initial_tcp[k], v) for k, v in model.tcp_head.state_dict().items())
    assert result["metadata"]["initial_tcp_head_sha256"] == result["metadata"]["final_tcp_head_sha256"]
    assert result["metadata"]["tcp_supervised"] is False
    assert all(row["maximum_tcp_head_gradient_norm"] == 0 for row in result["history"])


def test_auxiliary_gradients_reach_trunk_but_not_parallel_classification_head():
    parameters = fit.resolve_parameters({"epochs": 1})
    with fit._execution():
        torch.manual_seed(1)
        model = fit._JointNetwork(7, parameters)
        x, y, tcp, _ = _data()
        state_logits, tcp_logits = model(torch.tensor(x, dtype=torch.float32))
        ce, _, mse, combined = fit._components(state_logits, tcp_logits,
                                               torch.from_numpy(y), torch.tensor(tcp, dtype=torch.float32), 10.)
        expected = F.cross_entropy(state_logits, torch.from_numpy(y)) + 10 * F.mse_loss(tcp_logits.sigmoid(), torch.tensor(tcp, dtype=torch.float32))
        assert torch.equal(combined, expected)
        mse.backward()
        assert model.trunk.weight.grad.abs().sum() > 0
        assert model.tcp_head.weight.grad.abs().sum() > 0
        assert model.state_head.weight.grad is None


def test_weights_share_initialization_but_auxiliary_supervision_changes_training():
    zero, auxiliary = _fit(0.), _fit(10.)
    assert zero["metadata"]["initial_layers_sha256"] == auxiliary["metadata"]["initial_layers_sha256"]
    assert zero["metadata"]["initial_tcp_head_sha256"] == auxiliary["metadata"]["initial_tcp_head_sha256"]
    assert auxiliary["metadata"]["initial_tcp_head_sha256"] != auxiliary["metadata"]["final_tcp_head_sha256"]
    assert zero["state"]["layers"]["trunk"] != auxiliary["state"]["layers"]["trunk"]
    assert zero["history"][0]["first_batch_shared_ce_gradient_norm"] == auxiliary["history"][0]["first_batch_shared_ce_gradient_norm"]


def test_prediction_targets_cannot_enter_public_fit_interface():
    assert set(inspect.signature(fit.fit_joint).parameters) == {
        "train_x", "state_targets", "tcp_targets", "predict_x_by_split", "seed",
        "config_id", "auxiliary_weight", "parameters",
    }


@pytest.mark.parametrize("problem", ["nan", "integer_features", "overflow", "wrong_states", "float_states",
                                    "wrong_tcp_rows", "wrong_tcp_width", "tcp_outside", "validation_width",
                                    "bool_seed", "empty_id", "negative_lambda", "nan_lambda", "architecture",
                                    "optimizer", "float_epoch", "extra_parameter", "train_prediction_override"])
def test_invalid_inputs_fail_before_network_creation(problem, monkeypatch):
    x, y, tcp, validation = _data()
    args = dict(seed=3, config_id="test", auxiliary_weight=1., parameters={"epochs": 1})
    predict = {"validation": validation}
    if problem == "nan": x[0, 0] = np.nan
    elif problem == "integer_features": x = x.astype(int)
    elif problem == "overflow": x[0, 0] = 1e100
    elif problem == "wrong_states": y[0] = 4
    elif problem == "float_states": y = y.astype(float)
    elif problem == "wrong_tcp_rows": tcp = tcp[:-1]
    elif problem == "wrong_tcp_width": tcp = tcp[:, :1]
    elif problem == "tcp_outside": tcp[0, 0] = -0.001
    elif problem == "validation_width": predict["validation"] = validation[:, :2]
    elif problem == "bool_seed": args["seed"] = True
    elif problem == "empty_id": args["config_id"] = " "
    elif problem == "negative_lambda": args["auxiliary_weight"] = -1
    elif problem == "nan_lambda": args["auxiliary_weight"] = np.nan
    elif problem == "architecture": args["parameters"]["hidden_width"] = 64
    elif problem == "optimizer": args["parameters"]["optimizer"] = "adam"
    elif problem == "float_epoch": args["parameters"]["epochs"] = 1.
    elif problem == "extra_parameter": args["parameters"]["validation_selection"] = True
    elif problem == "train_prediction_override": predict["train"] = x
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached network construction")
    monkeypatch.setattr(fit, "_JointNetwork", forbidden)
    with pytest.raises(fit.JointTCPFitError):
        fit.fit_joint(x, y, tcp, predict, **args)


@pytest.mark.parametrize("problem", ["weights", "hash", "target_order", "recipe", "dimension"])
def test_state_tampering_is_rejected(problem):
    state = _fit()["state"]
    if problem == "weights": state["layers"]["trunk"]["weight"][0][0] += .01
    elif problem == "hash": state["state_sha256"] = "0" * 64
    elif problem == "target_order": state["target_order"] = state["target_order"][::-1]
    elif problem == "recipe": state["parameters"]["early_stopping"] = True
    elif problem == "dimension": state["input_dim"] += 1
    with pytest.raises(fit.JointTCPFitError):
        fit.validate_state(state)


def test_nonfinite_training_retains_the_last_complete_epoch(monkeypatch):
    original = fit._train_epoch
    def fail_on_second(*args, **kwargs):
        if kwargs["epoch"] == 2:
            with torch.no_grad():
                args[0].trunk.weight.fill_(float("inf"))
            raise fit.JointTCPFitError("injected nonfinite parameter")
        return original(*args, **kwargs)
    monkeypatch.setattr(fit, "_train_epoch", fail_on_second)
    with pytest.raises(fit.JointTCPFitError) as captured:
        _fit()
    partial = captured.value.partial_result
    assert partial["completed_epochs"] == 1
    assert partial["failed_epoch"] == 2
    assert len(partial["history"]) == 1
    fit.validate_state(partial["state"])
    assert np.isfinite(fit.predict_joint(partial["state"], _data()[3])["q"]).all()

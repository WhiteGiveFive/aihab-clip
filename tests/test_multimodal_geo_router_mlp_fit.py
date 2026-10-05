from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from multimodal import geo_helpfulness_router_mlp_fit as fit
from multimodal import geo_helpfulness_router_mlp_state as portable


def _data():
    rng = np.random.default_rng(815)
    x = np.zeros((64, 727), dtype=np.float64)
    y = np.tile(np.arange(4, dtype=np.int64), 16)
    x[:, :25] = rng.normal(size=(64, 25))
    x[np.arange(64), y] += 2
    x[np.arange(64), 25 + y] = 1
    x[np.arange(64), 79 + y] = 1
    validation = x[:12].copy()
    validation[:, :25] += 0.1
    return x, y, validation


def _spec(variant, widths):
    return {
        "config_id": "fixture_" + variant + "_" + "_".join(map(str, widths)),
        "family": "mlp",
        "variant": variant,
        "parameters": {**portable.MLP_SHARED, "hidden_widths": list(widths)},
    }


@pytest.mark.parametrize(
    "variant", ["primary_full727", "ablation_no_pairs79", "ablation_numeric25"]
)
@pytest.mark.parametrize("widths", portable.ARCHITECTURES)
def test_all_architectures_and_variants_fit_final_epoch_and_export_exactly(
    variant, widths
):
    x, y, validation = _data()
    before = (x.copy(), y.copy(), validation.copy())
    spec = _spec(variant, widths)
    state, train_q, val_q, diagnostic = fit.fit_candidate(x, y, validation, spec, 1)
    assert state["parameters"] == spec["parameters"]
    assert (
        diagnostic["final_epoch"] == 300 and diagnostic["checkpoint"] == "final_epoch"
    )
    assert len(diagnostic["history"]) == 300
    assert [record["epoch"] for record in diagnostic["history"]] == list(range(1, 301))
    for record in diagnostic["history"]:
        assert record["gradients_finite"] is True
        assert np.isfinite(
            [
                record["optimization_cross_entropy"],
                record["train_nll"],
                record["train_accuracy"],
                record["maximum_gradient_l2_norm"],
            ]
        ).all()
    assert (
        diagnostic["late_training_diagnostic"]["epoch200_train_nll"]
        == diagnostic["history"][199]["train_nll"]
    )
    assert (
        diagnostic["late_training_diagnostic"]["epoch300_train_nll"]
        == diagnostic["history"][299]["train_nll"]
    )
    assert diagnostic["late_training_diagnostic"]["convergence_claim"] is False
    assert diagnostic["validation_labels_accessed"] is False
    assert diagnostic["validation_features_used_for_fitting"] is False
    assert diagnostic["bias_weight_decay"] == 0.0
    assert diagnostic["warnings"] == []
    assert train_q.shape == (64, 4) and val_q.shape == (12, 4)
    assert np.array_equal(
        val_q, portable.predict_state(validation, json.loads(json.dumps(state)))
    )
    for name in ("train", "validation_features_only"):
        assert (
            diagnostic["probability_parity"][name]["maximum_probability_abs_error"]
            < 1e-12
        )
        assert diagnostic["probability_parity"][name]["class_prediction_exact"] is True
    for actual, expected in zip((x, y, validation), before):
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "variant", ["primary_full727", "ablation_no_pairs79", "ablation_numeric25"]
)
def test_exact_repeat_and_validation_independence_for_each_variant(variant):
    x, y, validation = _data()
    spec = _spec(variant, (32, 16))
    first, train_first, _, diagnostic_first = fit.fit_candidate(
        x, y, validation, spec, 1
    )
    second, train_second, _, diagnostic_second = fit.fit_candidate(
        x, y, validation + 0.125, spec, 1
    )
    assert first == second
    assert portable.canonical_hash(first) == portable.canonical_hash(second)
    assert np.array_equal(train_first, train_second)
    assert diagnostic_first["history"] == diagnostic_second["history"]


def test_rng_and_torch_execution_settings_are_restored():
    x, y, validation = _data()
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    fit.fit_candidate(x, y, validation, _spec("ablation_numeric25", (32,)), 1)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.get_num_threads() == before_threads
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic


@pytest.mark.parametrize(
    "problem",
    [
        "missing_class",
        "bad_dtype",
        "validation_nan",
        "unknown_architecture",
        "early_stopping",
    ],
)
def test_invalid_inputs_or_scientific_settings_fail_before_network_construction(
    problem, monkeypatch
):
    x, y, validation = _data()
    spec = _spec("ablation_numeric25", (32,))
    if problem == "missing_class":
        y[y == 3] = 0
    elif problem == "bad_dtype":
        x = x.astype(np.float32)
    elif problem == "validation_nan":
        validation[0, 0] = np.nan
    elif problem == "unknown_architecture":
        spec["parameters"]["hidden_widths"] = [128, 64, 32]
    else:
        spec["parameters"]["early_stopping"] = True

    def forbidden(*args, **kwargs):
        raise AssertionError("network constructed before input validation")

    monkeypatch.setattr(fit, "_network", forbidden)
    with pytest.raises(ValueError):
        fit.fit_candidate(x, y, validation, spec, 1)

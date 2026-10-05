from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from multimodal import geo_helpfulness_router_mlp_state as portable


COUNTS = {
    "ablation_numeric25": [964, 1924, 1428, 3876],
    "ablation_no_pairs79": [2692, 5380, 3156, 7332],
    "primary_full727": [23428, 46852, 23892, 48804],
}


def _state(variant="ablation_numeric25", widths=(32,)):
    indices = portable.variant_feature_indices(variant)
    dims = [len(indices), *widths, 4]
    rng = np.random.default_rng(717)
    layers = [
        {
            "weight": rng.normal(0, 0.2, (right, left)).tolist(),
            "bias": rng.normal(0, 0.1, right).tolist(),
        }
        for left, right in zip(dims, dims[1:])
    ]
    return portable.seal_state(
        {
            "schema_version": portable.SCHEMA_VERSION,
            "config_id": "synthetic",
            "family": "mlp",
            "variant": variant,
            "training_seed": 1,
            "parameters": {**portable.MLP_SHARED, "hidden_widths": list(widths)},
            "feature_input_names": portable.full_feature_columns(),
            "feature_indices": indices,
            "classes": [0, 1, 2, 3],
            "target_order": list(portable.TARGET_ORDER),
            "fit_role": "development_train_oof",
            "fit_row_count": 64,
            "initialization_seed": 1,
            "shuffle_seed": 1,
            "dropout_seed": 1,
            "parameter_count": portable.parameter_count(variant, list(widths)),
            "implementation": deepcopy(portable.IMPLEMENTATION),
            "layers": layers,
        }
    )


@pytest.mark.parametrize("variant", list(COUNTS))
@pytest.mark.parametrize("widths", portable.ARCHITECTURES)
def test_exact_bias_inclusive_parameter_count_and_json_roundtrip(variant, widths):
    state = _state(variant, widths)
    assert (
        state["parameter_count"]
        == COUNTS[variant][portable.ARCHITECTURES.index(widths)]
    )
    actual = sum(
        np.asarray(layer["weight"]).size + len(layer["bias"])
        for layer in state["layers"]
    )
    assert actual == state["parameter_count"]
    x = np.random.default_rng(9).normal(size=(5, 727)).astype(np.float64)
    before = x.copy()
    q = portable.predict_state(x, state)
    assert q.shape == (5, 4) and q.dtype == np.float64
    assert np.array_equal(q, portable.predict_state(x, json.loads(json.dumps(state))))
    assert np.array_equal(x, before)
    assert np.allclose(q.sum(axis=1), 1, atol=1e-12, rtol=0)


@pytest.mark.parametrize("variant", ["ablation_numeric25", "ablation_no_pairs79"])
def test_dropped_feature_columns_cannot_affect_prediction(variant):
    state = _state(variant, (32, 16))
    x = np.zeros((3, 727), dtype=np.float64)
    expected = portable.predict_state(x, state)
    excluded = sorted(set(range(727)) - set(state["feature_indices"]))
    x[:, excluded] = 1e20
    assert np.array_equal(expected, portable.predict_state(x, state))


def test_hidden_relu_and_output_softmax_are_applied_without_dropout():
    state = _state()
    for layer in state["layers"]:
        layer["weight"] = np.zeros_like(layer["weight"]).tolist()
        layer["bias"] = np.zeros_like(layer["bias"]).tolist()
    state["layers"][0]["weight"][0][0] = 1.0
    state["layers"][1]["weight"][0][0] = 1.0
    state = portable.seal_state(state)
    x = np.zeros((2, 727), dtype=np.float64)
    x[:, 0] = [-2, 2]
    logits = portable.predict_logits(x, state)
    assert np.array_equal(logits, [[0, 0, 0, 0], [2, 0, 0, 0]])
    assert np.array_equal(portable.predict_state(x, state)[0], np.full(4, 0.25))


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "nonfinite",
        "parameter_count",
        "epoch",
        "class_order",
        "shuffle_seed",
        "unknown_layer",
        "feature_order",
    ],
)
def test_invalid_state_rejected_even_after_resealing(problem):
    state = _state()
    if problem == "shape":
        state["layers"][0]["weight"].pop()
    elif problem == "nonfinite":
        state["layers"][0]["bias"][0] = float("nan")
    elif problem == "parameter_count":
        state["parameter_count"] += 1
    elif problem == "epoch":
        state["parameters"]["epochs"] = 200
    elif problem == "class_order":
        state["classes"] = [1, 0, 2, 3]
    elif problem == "shuffle_seed":
        state["shuffle_seed"] = 2
    elif problem == "unknown_layer":
        state["layers"][0]["batch_norm"] = True
    else:
        state["feature_indices"].reverse()
    with pytest.raises(ValueError):
        portable.seal_state(state)


def test_tamper_and_invalid_input_types_fail_closed():
    state = _state()
    state["layers"][0]["bias"][0] += 0.1
    with pytest.raises(ValueError, match="hash"):
        portable.predict_state(np.zeros((1, 727)), state)
    for matrix in (
        np.zeros((1, 25)),
        np.zeros((1, 727), dtype=np.float32),
        np.full((1, 727), np.inf),
    ):
        with pytest.raises(ValueError, match="float64"):
            portable.predict_state(matrix, _state())


def test_state_import_is_fitting_free():
    code = """
import sys
from multimodal import geo_helpfulness_router_mlp_state
assert not any(name.split('.')[0] in {'torch','sklearn','scipy','xgboost'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

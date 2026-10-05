from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from multimodal import geo_helpfulness_router_capacity_state as portable


def _state(family="rf", variant="ablation_numeric25", *, trees=None):
    if trees is None:
        trees = [
            {
                "left": [1, -1, -1],
                "right": [2, -1, -1],
                "feature": [0, -2, -2],
                "threshold": [1.0 + 2**-25, -2.0, -2.0],
                "value": [
                    [0.5, 0.5, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
            }
        ]
    model = {"trees": trees}
    parameters = {"n_estimators": len(trees)}
    if family == "xgb":
        model.update(
            {
                "tree_info": [i % 4 for i in range(len(trees))],
                "base_score": 0.5,
                "objective": "multi:softprob",
                "native_model_sha256": "a" * 64,
            }
        )
        parameters = {"n_estimators": len(trees) // 4, "base_score": 0.5}
    return portable.seal_state(
        {
            "schema_version": portable.SCHEMA_VERSION,
            "config_id": "synthetic",
            "family": family,
            "variant": variant,
            "training_seed": 1,
            "parameters": parameters,
            "feature_input_names": portable.full_feature_columns(),
            "feature_indices": portable.variant_feature_indices(variant),
            "classes": [0, 1, 2, 3],
            "target_order": list(portable.TARGET_ORDER),
            "fit_role": "development_train_oof",
            "fit_row_count": 40,
            "implementation": {
                "library": "xgboost" if family == "xgb" else "sklearn",
                "version": "2.1.4" if family == "xgb" else "1.5.2",
                "split_input_dtype": "float32",
                "threadpool_limit": 1,
            },
            "model": model,
        }
    )


def _xgb_leaf(value):
    return {
        "left": [-1],
        "right": [-1],
        "feature": [0],
        "threshold": [value],
        "value": [value],
        "default_left": [0],
    }


def test_rf_uses_float32_features_and_float64_inclusive_threshold():
    state = _state()
    x = np.zeros((3, 727), dtype=np.float64)
    x[:, 0] = [1.0, 1.0 + 2**-24, 1.0 + 2**-22]
    probabilities = portable.predict_state(x, state)
    assert np.array_equal(probabilities.argmax(axis=1), [0, 0, 1])
    assert probabilities.dtype == np.float64
    assert np.array_equal(
        probabilities, portable.predict_state(x, json.loads(json.dumps(state)))
    )


def test_xgb_split_is_strict_float32_and_margin_accumulation_stays_float32():
    tree = {
        "left": [1, -1, -1],
        "right": [2, -1, -1],
        "feature": [0, 0, 0],
        "threshold": [1.0, 1.0, -1.0],
        "value": [0.0, 1.0, -1.0],
        "default_left": [1, 0, 0],
    }
    state = _state("xgb", trees=[tree, *[_xgb_leaf(0.0) for _ in range(3)]])
    x = np.zeros((2, 727), dtype=np.float64)
    x[:, 0] = [0.999, 1.0 - 2**-27]
    margins = portable.predict_margin(x, state)
    assert margins.dtype == np.float32
    assert np.array_equal(margins[:, 0], [1.5, -0.5])
    trees = []
    for value in (1e8, -1e8, 0.1):
        trees.extend([_xgb_leaf(value), *[_xgb_leaf(0.0) for _ in range(3)]])
    state = _state("xgb", trees=trees)
    assert portable.predict_margin(x, state)[0, 0] == np.float32(0.1)
    probabilities = portable.predict_state(x, state)
    assert probabilities.dtype == np.float64
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-15)


@pytest.mark.parametrize(
    "variant", ["primary_full727", "ablation_no_pairs79", "ablation_numeric25"]
)
def test_full_input_is_projected_to_the_exact_frozen_variant(variant):
    state = _state(variant=variant)
    x = np.zeros((2, 727), dtype=np.float64)
    before = portable.predict_state(x, state)
    excluded = sorted(set(range(727)) - set(state["feature_indices"]))
    x[:, excluded] = 1e30
    assert np.array_equal(before, portable.predict_state(x, state))


@pytest.mark.parametrize(
    "corruption",
    [
        "self_cycle",
        "shared_child",
        "bad_feature",
        "negative_probability",
        "wrong_feature_order",
        "wrong_class_order",
    ],
)
def test_state_structure_fails_closed_even_with_a_new_hash(corruption):
    state = deepcopy(_state())
    if corruption == "self_cycle":
        state["model"]["trees"][0]["left"][0] = 0
    elif corruption == "shared_child":
        state["model"]["trees"][0]["right"][0] = 1
    elif corruption == "bad_feature":
        state["model"]["trees"][0]["feature"][0] = 25
    elif corruption == "negative_probability":
        state["model"]["trees"][0]["value"][1] = [-1, 2, 0, 0]
    elif corruption == "wrong_feature_order":
        state["feature_indices"] = list(reversed(state["feature_indices"]))
    else:
        state["classes"] = [1, 0, 2, 3]
    with pytest.raises(ValueError):
        portable.seal_state(state)


def test_state_hash_and_nonfinite_inputs_are_rejected():
    state = _state()
    state["model"]["trees"][0]["threshold"][0] = 2.0
    with pytest.raises(ValueError, match="hash"):
        portable.predict_state(np.zeros((1, 727), dtype=np.float64), state)
    for invalid in (np.nan, np.inf, -np.inf):
        x = np.zeros((1, 727), dtype=np.float64)
        x[0, 0] = invalid
        with pytest.raises(ValueError, match="finite"):
            portable.predict_state(x, _state())


def test_pure_state_import_exposes_no_fit_dependencies():
    program = """
import sys
from multimodal import geo_helpfulness_router_capacity_state
for name in sys.modules:
    assert name.split('.')[0] not in {'torch','sklearn','scipy','xgboost'}, name
"""
    result = subprocess.run(
        [sys.executable, "-c", program], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr

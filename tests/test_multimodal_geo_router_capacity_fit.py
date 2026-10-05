from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from multimodal import geo_helpfulness_router_candidate_state as legacy
from multimodal import geo_helpfulness_router_capacity_fit as fit
from multimodal import geo_helpfulness_router_capacity_state as portable


def _data():
    rng = np.random.default_rng(20260911)
    target = np.tile(np.arange(4, dtype=np.int64), 32)
    x = np.zeros((128, 727), dtype=np.float64)
    x[:, :25] = rng.normal(size=(128, 25))
    x[np.arange(128), target] += 2.0
    x[np.arange(128), 25 + target] = 1
    x[np.arange(128), 79 + target] = 1
    validation = x[:16].copy()
    validation[:, :25] += 0.1
    return x, target, validation


def _spec(family, variant="ablation_numeric25"):
    parameters = {
        "rf": {"max_depth": 4, "min_samples_leaf": 20},
        "xgb": {"max_depth": 2, "reg_lambda": 1},
        "lr": {"C": 1.0},
    }[family]
    return {
        "config_id": family + "_fixture",
        "family": family,
        "variant": variant,
        "parameters": parameters,
    }


def _old_lr(variant, rows=128):
    columns = legacy.full_feature_columns()
    indices = legacy.variant_feature_indices(variant)
    coefficients = np.zeros((4, len(indices)), dtype=np.float64)
    coefficients[np.arange(4), np.arange(4)] = 0.8
    return legacy.seal_candidate_state(
        {
            "schema_version": legacy.SCHEMA_VERSION,
            "protocol_id": "protocol_v1",
            "training_seed": 1,
            "variant_id": variant,
            "C": 1.0,
            "C_hex": (1.0).hex(),
            "target_order": list(legacy.TARGET_ORDER),
            "classes": [0, 1, 2, 3],
            "fit_role": "development_train_oof",
            "fit_row_count": rows,
            "fit_row_identity_sha256": "a" * 64,
            "study_spec_sha256": "b" * 64,
            "dataset_manifest_sha256": "c" * 64,
            "feature_input_names": columns,
            "feature_indices": indices,
            "feature_names": [columns[index] for index in indices],
            "fit_parameters": dict(legacy.FROZEN_FIT_PARAMETERS, C=1.0),
            "implementation": dict(legacy.IMPLEMENTATION),
            "coef": coefficients.tolist(),
            "intercept": [0.0] * 4,
            "n_iter": [1],
        }
    )


@pytest.mark.parametrize(
    "variant", ["primary_full727", "ablation_no_pairs79", "ablation_numeric25"]
)
def test_lr_reuses_exact_existing_state_for_each_variant(variant):
    x, y, validation = _data()
    spec = _spec("lr", variant)
    spec["existing_lr_state"] = _old_lr(variant)
    before = deepcopy(spec["existing_lr_state"])
    state, train_q, val_q, diagnostic = fit.fit_candidate(x, y, validation, spec, 1)
    assert state["model"] == before == spec["existing_lr_state"]
    assert np.array_equal(train_q, legacy.predict_router_probabilities(x, before))
    assert np.array_equal(
        val_q, legacy.predict_router_probabilities(validation, before)
    )
    assert diagnostic["fit_performed"] is False
    assert diagnostic["existing_lr_reused"] is True


@pytest.mark.parametrize(
    "variant", ["primary_full727", "ablation_no_pairs79", "ablation_numeric25"]
)
def test_rf_native_prediction_parity_and_no_validation_fit(variant, monkeypatch):
    x, y, validation = _data()
    before = (x.copy(), y.copy(), validation.copy())
    original = fit.RandomForestClassifier
    fitted_rows = []

    class ObservedRF(original):
        def fit(self, matrix, target):
            fitted_rows.append((matrix.shape, target.copy()))
            return super().fit(matrix, target)

    monkeypatch.setattr(fit, "RandomForestClassifier", ObservedRF)
    state, train_q, val_q, diagnostic = fit.fit_candidate(
        x, y, validation, _spec("rf", variant), 1
    )
    assert fitted_rows[0][0] == (128, len(legacy.variant_feature_indices(variant)))
    assert np.array_equal(fitted_rows[0][1], y)
    assert train_q.shape == (128, 4) and val_q.shape == (16, 4)
    assert train_q.dtype == val_q.dtype == np.float64
    assert np.array_equal(
        val_q, portable.predict_state(validation, json.loads(json.dumps(state)))
    )
    assert diagnostic["validation_features_used_for_fitting"] is False
    assert (
        diagnostic["probability_parity"]["train"]["maximum_probability_abs_error"]
        < 1e-12
    )
    for actual, expected in zip((x, y, validation), before):
        assert np.array_equal(actual, expected)


def _load_xgboost():
    vendor = (
        Path(__file__).resolve().parents[1]
        / "multimodal_artifacts/environments/router_capacity_v1/xgboost_cpu_2_1_4"
    )
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
    xgboost = pytest.importorskip(
        "xgboost", reason="pinned project-local XGBoost has not been installed"
    )
    assert xgboost.__version__ == "2.1.4"
    return xgboost


@pytest.mark.parametrize(
    "variant", ["primary_full727", "ablation_no_pairs79", "ablation_numeric25"]
)
def test_xgb_native_margin_probability_and_argmax_parity(variant):
    _load_xgboost()
    x, y, validation = _data()
    state, train_q, val_q, diagnostic = fit.fit_candidate(
        x, y, validation, _spec("xgb", variant), 1
    )
    assert len(state["model"]["trees"]) == 800
    assert state["model"]["tree_info"] == list(np.tile(np.arange(4), 200))
    assert train_q.dtype == val_q.dtype == np.float64
    assert np.array_equal(
        val_q, portable.predict_state(validation, json.loads(json.dumps(state)))
    )
    assert diagnostic["probability_parity"]["train"]["maximum_margin_abs_error"] < 1e-6
    assert (
        diagnostic["probability_parity"]["validation_features_only"][
            "maximum_probability_abs_error"
        ]
        < 1e-6
    )


@pytest.mark.parametrize("family", ["lr", "rf", "xgb"])
def test_unregistered_settings_rejected_before_any_fit(family):
    spec = _spec(family)
    spec["parameters"]["class_weight"] = "balanced"
    x, y, validation = _data()
    with pytest.raises(ValueError):
        fit.fit_candidate(x, y, validation, spec, 1)


def test_lr_missing_or_mismatched_reuse_is_rejected():
    x, y, validation = _data()
    spec = _spec("lr")
    with pytest.raises(ValueError, match="existing_lr_state"):
        fit.fit_candidate(x, y, validation, spec, 1)
    spec["existing_lr_state"] = _old_lr("primary_full727")
    with pytest.raises(ValueError, match="metadata"):
        fit.fit_candidate(x, y, validation, spec, 1)


def test_training_inputs_require_all_four_states_and_exact_feature_schema():
    x, y, validation = _data()
    y[:] = 0
    with pytest.raises(ValueError, match="four fixed states"):
        fit.fit_candidate(x, y, validation, _spec("rf"), 1)
    x, y, validation = _data()
    with pytest.raises(ValueError, match="727"):
        fit.fit_candidate(x[:, :25], y, validation, _spec("rf"), 1)


def test_changing_label_blind_validation_features_cannot_change_the_fitted_model():
    x, y, validation = _data()
    spec = _spec("rf")
    first, train_first, _, _ = fit.fit_candidate(x, y, validation, spec, 1)
    second, train_second, _, _ = fit.fit_candidate(x, y, validation + 7.0, spec, 1)
    assert first == second
    assert np.array_equal(train_first, train_second)

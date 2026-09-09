from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from multimodal import geo_helpfulness_router_candidate_fit as fitting
from multimodal import geo_helpfulness_router_candidate_state as portable


def training_data():
    rng = np.random.default_rng(808)
    targets = np.tile(np.arange(4, dtype=np.int64), 32)
    matrix = np.zeros((len(targets), 727), dtype=np.float64)
    matrix[:, :25] = rng.normal(size=(len(targets), 25))
    matrix[np.arange(len(targets)), targets] += 1.5
    category_ids = rng.integers(0, 18, size=(len(targets), 3))
    offsets = [25, 43, 61, 79, 403]
    categories = [
        category_ids[:, 0], category_ids[:, 1], category_ids[:, 2],
        18 * category_ids[:, 0] + category_ids[:, 1],
        18 * category_ids[:, 1] + category_ids[:, 2],
    ]
    for offset, values in zip(offsets, categories):
        matrix[np.arange(len(targets)), offset + values] = 1
    return matrix, targets


def fit_kwargs(**overrides):
    result = {
        "seed": 1,
        "variant_id": "primary_full727",
        "C": 1.0,
        "feature_columns": portable.full_feature_columns(),
        "protocol_id": "synthetic_protocol",
        "fit_row_identity_sha256": "a" * 64,
        "study_spec_hash": "b" * 64,
        "dataset_manifest_hash": "c" * 64,
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize(
    ("variant", "dimension"),
    [("primary_full727", 727), ("ablation_no_pairs79", 79), ("ablation_numeric25", 25)],
)
def test_fit_variants_portable_parity_and_serialization(variant, dimension):
    matrix, targets = training_data()
    state, diagnostics = fitting.fit_router_candidate(
        matrix, targets, **fit_kwargs(variant_id=variant)
    )
    assert np.asarray(state["coef"]).shape == (4, dimension)
    assert state["classes"] == [0, 1, 2, 3]
    assert state["target_order"] == ["rescue", "harm", "both_correct", "both_wrong"]
    assert state["fit_parameters"] == {**fitting.FIT_PARAMETERS, "C": 1.0}
    assert state["implementation"]["threadpool_limit"] == 1
    assert 0 <= diagnostics["training_log_loss"] < np.log(4)
    assert 0 <= diagnostics["training_accuracy"] <= 1
    assert diagnostics["probability_parity_max_abs"] <= 1e-12
    assert diagnostics["fit_seconds"] >= 0
    assert state["schema_version"] == "geo_helpfulness.router_candidate.v2"
    assert state["n_iter"][0] < 5000
    restored = json.loads(json.dumps(state, allow_nan=False))
    portable.validate_candidate_state(restored)
    np.testing.assert_array_equal(
        portable.predict_router_probabilities(matrix, state),
        portable.predict_router_probabilities(matrix, restored),
    )
    assert "fit_seconds" not in state
    assert all(warning["category"] == "FutureWarning" for warning in diagnostics["warnings"])


def test_repeat_fitting_is_exactly_reproducible():
    matrix, targets = training_data()
    first, _ = fitting.fit_router_candidate(matrix, targets, **fit_kwargs())
    second, _ = fitting.fit_router_candidate(matrix, targets, **fit_kwargs())
    assert first == second


def test_numeric_ablation_ignores_dropped_columns():
    matrix, targets = training_data()
    kwargs = fit_kwargs(variant_id="ablation_numeric25")
    first, _ = fitting.fit_router_candidate(matrix, targets, **kwargs)
    changed = matrix.copy()
    changed[:, 25:] *= 37
    second, _ = fitting.fit_router_candidate(changed, targets, **kwargs)
    assert first == second


def test_balanced_constant_data_can_converge_at_initial_point():
    matrix = np.zeros((8, 727), dtype=np.float64)
    targets = np.tile(np.arange(4, dtype=np.int64), 2)
    state, diagnostics = fitting.fit_router_candidate(matrix, targets, **fit_kwargs())
    assert state["n_iter"] == [0]
    assert diagnostics["training_log_loss"] == pytest.approx(np.log(4))
    np.testing.assert_array_equal(
        portable.predict_router_probabilities(matrix, state), np.full((8, 4), .25)
    )


@pytest.mark.parametrize(
    "case",
    ["float32", "nan", "inf", "few_rows", "bad_width", "targets_float",
     "targets_bool", "targets_shape", "targets_high", "targets_low", "missing_state"],
)
def test_invalid_arrays_rejected_before_estimator(case, monkeypatch):
    matrix, targets = training_data()
    if case == "float32":
        matrix = matrix.astype(np.float32)
    elif case in ("nan", "inf"):
        matrix[0, 0] = float(case)
    elif case == "few_rows":
        matrix, targets = matrix[:3], targets[:3]
    elif case == "bad_width":
        matrix = matrix[:, :79]
    elif case == "targets_float":
        targets = targets.astype(float)
    elif case == "targets_bool":
        targets = targets.astype(bool)
    elif case == "targets_shape":
        targets = targets[:, None]
    elif case == "targets_high":
        targets[0] = 4
    elif case == "targets_low":
        targets[0] = -1
    else:
        targets[targets == 3] = 0
    monkeypatch.setattr(
        fitting, "LogisticRegression", lambda **kwargs: pytest.fail("estimator constructed")
    )
    with pytest.raises(fitting.RouterCandidateFitError):
        fitting.fit_router_candidate(matrix, targets, **fit_kwargs())


@pytest.mark.parametrize(
    "override",
    [
        {"seed": True}, {"seed": 0}, {"C": True}, {"C": 0.5}, {"C": float("nan")},
        {"variant_id": "unknown"}, {"protocol_id": ""},
        {"fit_row_identity_sha256": "bad"}, {"study_spec_hash": "B" * 64},
        {"dataset_manifest_hash": "g" * 64}, {"feature_columns": ["wrong"] * 727},
    ],
)
def test_invalid_settings_rejected_before_estimator(override, monkeypatch):
    monkeypatch.setattr(
        fitting, "LogisticRegression", lambda **kwargs: pytest.fail("estimator constructed")
    )
    with pytest.raises(fitting.RouterCandidateFitError):
        fitting.fit_router_candidate(*training_data(), **fit_kwargs(**override))


def test_rejects_wrong_sklearn_version(monkeypatch):
    monkeypatch.setattr(fitting.sklearn, "__version__", "0.0.0")
    with pytest.raises(fitting.RouterCandidateFitError, match="requires sklearn"):
        fitting.fit_router_candidate(*training_data(), **fit_kwargs())


@pytest.mark.parametrize("failure", ["convergence", "cap", "unexpected", "nan_coef", "wrong_classes", "parity"])
def test_rejects_estimator_failure(failure, monkeypatch):
    class BrokenEstimator:
        def __init__(self, **params):
            assert params == {**fitting.FIT_PARAMETERS, "C": 1.0}

        def fit(self, features, targets):
            self.n_iter_ = np.array([5000 if failure == "cap" else 10])
            self.classes_ = np.array([0, 1, 2, 4] if failure == "wrong_classes" else [0, 1, 2, 3])
            self.coef_ = np.zeros((4, features.shape[1]), dtype=np.float64)
            self.intercept_ = np.zeros(4, dtype=np.float64)
            if failure == "convergence":
                warnings.warn("did not converge", ConvergenceWarning)
            elif failure == "unexpected":
                warnings.warn("bad arithmetic", RuntimeWarning)
            elif failure == "nan_coef":
                self.coef_[0, 0] = np.nan
            return self

        def predict_proba(self, features):
            return np.tile([1., 0., 0., 0.], (len(features), 1))

        def predict(self, features):
            return np.zeros(len(features), dtype=np.int64)

    monkeypatch.setattr(fitting, "LogisticRegression", BrokenEstimator)
    with pytest.raises(fitting.RouterCandidateFitError):
        fitting.fit_router_candidate(*training_data(), **fit_kwargs())


def test_input_arrays_are_not_mutated():
    matrix, targets = training_data()
    before_matrix, before_targets = matrix.copy(), targets.copy()
    fitting.fit_router_candidate(*[matrix, targets], **fit_kwargs())
    np.testing.assert_array_equal(matrix, before_matrix)
    np.testing.assert_array_equal(targets, before_targets)


def test_approved_amendment_changes_only_iteration_budget():
    assert fitting.FIT_PARAMETERS == {
        "penalty": "l2",
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "fit_intercept": True,
        "class_weight": None,
        "dual": False,
        "warm_start": False,
        "n_jobs": 1,
        "max_iter": 5000,
        "tol": 1.0e-8,
        "random_state": None,
    }


def test_convergence_past_original_budget_is_accepted(monkeypatch):
    class ConvergedEstimator:
        def __init__(self, **params):
            assert params["max_iter"] == 5000

        def fit(self, features, targets):
            self.n_iter_ = np.array([1500])
            self.classes_ = np.arange(4)
            self.coef_ = np.zeros((4, features.shape[1]), dtype=np.float64)
            self.intercept_ = np.zeros(4, dtype=np.float64)
            return self

        def predict_proba(self, features):
            return np.full((len(features), 4), 0.25, dtype=np.float64)

        def predict(self, features):
            return np.zeros(len(features), dtype=np.int64)

    monkeypatch.setattr(fitting, "LogisticRegression", ConvergedEstimator)
    state, diagnostics = fitting.fit_router_candidate(*training_data(), **fit_kwargs())
    assert state["schema_version"] == "geo_helpfulness.router_candidate.v2"
    assert state["n_iter"] == diagnostics["n_iter"] == [1500]

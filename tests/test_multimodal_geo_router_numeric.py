from __future__ import annotations

from copy import deepcopy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from multimodal import geo_helpfulness_router_numeric as numeric
from multimodal import geo_helpfulness_targets_features as m3


DIGEST = "a" * 64


def calibration_data() -> tuple[np.ndarray, np.ndarray]:
    # The likelihood optimum has p(class 0)=120/300=.4. A native margin
    # of 2*log(17*.4/.6) therefore has the analytically known optimum T=2.
    logits = np.zeros((300, 18), dtype=np.float64)
    logits[:, 0] = 2 * math.log(17 * 0.4 / 0.6)
    labels = np.concatenate([np.zeros(120), np.ones(180)]).astype(np.int64)
    return logits, labels


def fit_state(**overrides):
    logits, labels = calibration_data()
    kwargs = dict(
        seed=1,
        mode="image_only",
        calibration_spec=numeric.CALIBRATION_SPEC,
        fit_row_identity_sha256=DIGEST,
    )
    kwargs.update(overrides)
    return numeric.fit_expert_temperature(logits, labels, **kwargs)


def feature_frame():
    rng = np.random.default_rng(314)
    probabilities = []
    for _ in range(3):
        weights = rng.uniform(0.0, 1.0, (20, 18))
        probabilities.append(weights / weights.sum(axis=1, keepdims=True))
    return m3.build_router_feature_frame(
        *probabilities, probability_basis=m3.CALIBRATED_PROBABILITY_BASIS
    )


def transform_state(frame=None, **overrides):
    if frame is None:
        frame = feature_frame()
    kwargs = dict(
        seed=1,
        frozen_schema=m3.build_router_feature_schema(),
        fit_row_identity_sha256=DIGEST,
    )
    kwargs.update(overrides)
    return numeric.fit_router_feature_transform(frame, **kwargs)


def test_recovers_analytic_temperature_and_roundtrips():
    logits, labels = calibration_data()
    state = fit_state()
    assert state["temperature"] == pytest.approx(2.0, abs=1e-6)
    assert state["calibrated_nll"] < state["native_nll"]
    assert state["fit_row_count"] == 300
    restored = json.loads(json.dumps(state, allow_nan=False))
    np.testing.assert_array_equal(
        numeric.apply_expert_temperature(logits, state),
        numeric.apply_expert_temperature(logits, restored),
    )
    assert (
        numeric.multiclass_nll(logits, labels, state["temperature"])
        == state["calibrated_nll"]
    )
    assert restored["optimizer"]["success"] is True


def test_nll_shift_invariance_and_underflow_without_clipping():
    logits, labels = calibration_data()
    assert numeric.multiclass_nll(logits + 1e6, labels, 2.0) == pytest.approx(
        numeric.multiclass_nll(logits, labels, 2.0), abs=1e-10
    )
    extreme = np.full((3, 18), -1e6, dtype=np.float64)
    extreme[0, 0:2] = 0.0
    extreme[1, 4] = 0.0
    extreme[2, :] = 0.0
    probs = numeric.apply_expert_temperature(extreme, fit_state())
    assert probs[0, 2] == 0.0
    assert probs[0, 0] == 0.5
    np.testing.assert_array_equal(probs.argmax(axis=1), [0, 4, 0])
    assert np.isfinite(numeric.multiclass_nll(extreme, np.array([2, 4, 0])))


@pytest.mark.parametrize(
    "case",
    [
        "float32",
        "nan",
        "inf",
        "empty",
        "shape",
        "labels_float",
        "labels_bool",
        "labels_high",
        "labels_low",
        "labels_shape",
    ],
)
def test_rejects_bad_fit_arrays(case):
    logits, labels = calibration_data()
    if case == "float32":
        logits = logits.astype(np.float32)
    elif case in ("nan", "inf"):
        logits[0, 0] = float(case)
    elif case == "empty":
        logits, labels = logits[:0], labels[:0]
    elif case == "shape":
        logits = logits[:, :17]
    elif case == "labels_float":
        labels = labels.astype(float)
    elif case == "labels_bool":
        labels = labels.astype(bool)
    elif case == "labels_high":
        labels[0] = 18
    elif case == "labels_low":
        labels[0] = -1
    else:
        labels = labels[:, None]
    with pytest.raises(numeric.M4NumericError):
        numeric.fit_expert_temperature(
            logits,
            labels,
            seed=1,
            mode="image_only",
            calibration_spec=numeric.CALIBRATION_SPEC,
            fit_row_identity_sha256=DIGEST,
        )


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_nll_temperature(temperature):
    with pytest.raises(numeric.M4NumericError):
        numeric.multiclass_nll(*calibration_data(), temperature)


def test_rejects_flat_objective_before_optimizer(monkeypatch):
    monkeypatch.setattr(
        numeric, "minimize_scalar", lambda *a, **k: pytest.fail("optimizer called")
    )
    with pytest.raises(numeric.M4NumericError, match="unidentifiable"):
        numeric.fit_expert_temperature(
            np.ones((2, 18), dtype=np.float64),
            np.array([0, 1]),
            seed=1,
            mode="geo_only",
            calibration_spec=numeric.CALIBRATION_SPEC,
            fit_row_identity_sha256=DIGEST,
        )


@pytest.mark.parametrize(
    "theta,success,status",
    [
        (5 - 1e-7, True, 0),
        (-5 + 1e-7, True, 0),
        (float("nan"), True, 0),
        (0.0, False, 1),
        (-1.0, True, 0),
    ],
)
def test_optimizer_failure_bound_and_nll_regression(
    monkeypatch, theta, success, status
):
    def fake_minimize(fun, **kwargs):
        assert kwargs == {
            "bounds": (-5.0, 5.0),
            "method": "bounded",
            "options": {"xatol": 1e-10, "maxiter": 500},
        }
        return SimpleNamespace(
            x=theta,
            success=success,
            status=status,
            message="fixture",
            nfev=5,
            nit=5,
            fun=fun(theta) if math.isfinite(theta) else float("nan"),
        )

    monkeypatch.setattr(numeric, "minimize_scalar", fake_minimize)
    with pytest.raises(numeric.M4NumericError):
        fit_state()


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 0},
        {"seed": True},
        {"mode": "raw"},
        {"fit_row_identity_sha256": "oops"},
        {"calibration_spec": {}},
    ],
)
def test_rejects_fit_identity_or_spec(override):
    with pytest.raises(numeric.M4NumericError):
        fit_state(**override)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 1.0),
        ("theta", 5.0),
        ("weighting", "equal_plot"),
        ("calibrated_nll", float("nan")),
        ("mode", "bad"),
    ],
)
def test_apply_rejects_tampered_temperature(field, value):
    state = fit_state()
    state[field] = value
    with pytest.raises(numeric.M4NumericError):
        numeric.apply_expert_temperature(calibration_data()[0], state)


@pytest.mark.parametrize(
    "field,value",
    [("class_count", 18.0), ("class_count", True), ("status", 0.0), ("status", False)],
)
def test_temperature_metadata_requires_exact_json_integers(field, value):
    state = fit_state()
    if field == "status":
        state["optimizer"][field] = value
    else:
        state[field] = value
    with pytest.raises(numeric.M4NumericError, match="must be an integer"):
        numeric.validate_temperature_state(state)


def test_matrix_order_population_scaling_and_json_parity():
    frame = feature_frame()
    state = transform_state(frame)
    matrix = numeric.transform_router_features(frame, state)
    columns = numeric.output_feature_columns()
    assert matrix.shape == (20, 727) and matrix.dtype == np.float64
    assert len(columns) == len(set(columns)) == 727
    assert columns[0] == "scaled__image_geo_agree"
    assert columns[25] == "onehot__image_pred__0"
    assert columns[-1] == "onehot__geo_raw_pred_pair__323"
    values = frame.loc[:, list(numeric.NUMERIC_INPUT_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    np.testing.assert_array_equal(state["mean"], values.mean(axis=0))
    np.testing.assert_array_equal(state["variance"], values.var(axis=0, ddof=0))
    nonconstant = np.asarray(state["variance"]) > 0
    np.testing.assert_allclose(
        matrix[:, :25][:, nonconstant].mean(axis=0), 0, atol=2e-14
    )
    np.testing.assert_allclose(
        matrix[:, :25][:, nonconstant].var(axis=0), 1, atol=2e-14
    )
    np.testing.assert_array_equal(matrix[:, 25:].sum(axis=1), 5)
    restored = json.loads(json.dumps(state, allow_nan=False))
    np.testing.assert_array_equal(
        matrix, numeric.transform_router_features(frame, restored)
    )


def test_constant_features_scale_one_and_fixed_unseen_categories():
    probabilities = np.zeros((4, 18), dtype=np.float64)
    probabilities[:, 0] = 1.0
    frame = m3.build_router_feature_frame(
        probabilities,
        probabilities,
        probabilities,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )
    state = transform_state(frame)
    assert state["variance"] == [0.0] * 25
    assert state["scale"] == [1.0] * 25
    matrix = numeric.transform_router_features(frame, state)
    np.testing.assert_array_equal(matrix[:, :25], 0)
    probabilities[:, :] = 0.0
    probabilities[:, 17] = 1.0
    unseen = m3.build_router_feature_frame(
        probabilities,
        probabilities,
        probabilities,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )
    transformed = numeric.transform_router_features(unseen, state)
    assert transformed[0, 25 + 17] == 1
    assert transformed[0, -1] == 1


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_category",
        "missing",
        "nonfinite",
        "wrong_dtype",
        "extra_target",
        "reordered",
    ],
)
def test_rejects_bad_semantic_frames(kind):
    frame = feature_frame()
    state = transform_state(frame)
    if kind == "unknown_category":
        frame.loc[0, "image_pred"] = 18
    elif kind == "missing":
        frame.loc[0, "image_entropy"] = None
    elif kind == "nonfinite":
        frame.loc[0, "image_entropy"] = float("inf")
    elif kind == "wrong_dtype":
        frame["image_entropy"] = frame["image_entropy"].astype(np.float32)
    elif kind == "extra_target":
        frame["target_state"] = "rescue"
    else:
        frame = frame.loc[:, list(reversed(frame.columns))]
    with pytest.raises(numeric.M4NumericError):
        numeric.transform_router_features(frame, state)


@pytest.mark.parametrize(
    "kind", ["scale", "variance", "mean", "columns", "vocab", "contract", "schema"]
)
def test_rejects_tampered_transform_state(kind):
    state = transform_state()
    if kind == "scale":
        state["scale"][0] += 1.0
    elif kind == "variance":
        state["variance"][0] = -1.0
    elif kind == "mean":
        state["mean"][0] = float("nan")
    elif kind == "columns":
        state["output_columns"].reverse()
    elif kind == "vocab":
        state["categorical_vocabularies"]["image_pred"].pop()
    elif kind == "contract":
        state["numeric_contract"]["variance_ddof"] = 1
    else:
        state["feature_schema_sha256"] = "0" * 64
    with pytest.raises(numeric.M4NumericError):
        numeric.transform_router_features(feature_frame(), state)


def test_seed_specific_fits_and_apply_never_fit(monkeypatch):
    frame = feature_frame()
    first = transform_state(frame, seed=1)
    shifted = frame.copy()
    shifted["image_entropy"] += 0.1
    second = transform_state(shifted, seed=2)
    assert first["seed"] == 1 and second["seed"] == 2
    assert first["mean"] != second["mean"]
    temperature = fit_state()
    monkeypatch.setattr(
        numeric, "fit_expert_temperature", lambda *a, **k: pytest.fail("fit called")
    )
    monkeypatch.setattr(
        numeric,
        "fit_router_feature_transform",
        lambda *a, **k: pytest.fail("fit called"),
    )
    monkeypatch.setattr(
        numeric, "minimize_scalar", lambda *a, **k: pytest.fail("optimizer called")
    )
    numeric.apply_expert_temperature(calibration_data()[0], temperature)
    numeric.transform_router_features(frame, first)


def test_rejects_drifted_frozen_feature_schema():
    schema = deepcopy(m3.build_router_feature_schema())
    schema["class_count"] = 19
    with pytest.raises(numeric.M4NumericError, match="schema"):
        transform_state(frozen_schema=schema)

"""Pure, serializable calibration and feature transforms for the M4 dataset.

No filesystem or dataset-role discovery happens here. Callers supply validated,
canonically ordered development-training rows for fitting; apply functions never
fit and can be reused for later validation and deployment.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from multimodal import geo_helpfulness_targets_features as m3
from multimodal.geo_helpfulness_protocol import canonical_sha256


TEMPERATURE_SCHEMA_VERSION = "geo_helpfulness.expert_temperature.v1"
TRANSFORM_SCHEMA_VERSION = "geo_helpfulness.router_numeric_transform.v1"
MODES = ("image_only", "geo_only", "raw_concat")
NUMERIC_INPUT_COLUMNS = m3.BOOLEAN_FEATURES + m3.INTEGER_FEATURES + m3.NUMERIC_FEATURES
CATEGORICAL_VOCABULARIES = {
    name: list(range(18 if index < 3 else 324))
    for index, name in enumerate(m3.CATEGORICAL_FEATURES)
}
CALIBRATION_SPEC = {
    "family": "scalar_temperature",
    "one_per_mode_and_training_seed": True,
    "fit_role": "development_train_oof",
    "fit_once": True,
    "apply_unchanged_to": [
        "development_validation",
        "final_development_experts",
        "locked_test",
    ],
    "objective": "multiclass_negative_log_likelihood",
    "parameterization": "log_temperature",
    "optimizer": "scipy_bounded_minimize_scalar",
    "log_temperature_bounds": [-5.0, 5.0],
    "absolute_tolerance": 1.0e-10,
    "max_iterations": 500,
    "failure_policy": "hard_reject",
}
NUMERIC_CONTRACT = {
    "family": "standard_scaler",
    "fit_role": "development_train_oof",
    "input_families_in_order": ["boolean", "integer", "numeric"],
    "boolean_encoding": "false_zero_true_one",
    "output_dtype": "float64",
    "with_mean": True,
    "with_std": True,
    "variance_ddof": 0,
    "zero_variance_scale": 1.0,
    "missing_policy": "hard_reject",
    "nonfinite_policy": "hard_reject",
}
SAFEGUARDS = {
    "boundary_log_temperature_distance": 1.0e-6,
    "nll_degradation_tolerance": 1.0e-10,
    "reject_all_rows_class_constant": True,
    "probability_row_sum_atol": m3.PROBABILITY_ATOL,
}


class M4NumericError(ValueError):
    """The frozen M4 numerical contract cannot be satisfied."""


def output_feature_columns() -> tuple[str, ...]:
    """Return the only allowed order of the 727 router-model inputs."""
    return tuple(f"scaled__{name}" for name in NUMERIC_INPUT_COLUMNS) + tuple(
        f"onehot__{name}__{category}"
        for name in m3.CATEGORICAL_FEATURES
        for category in CATEGORICAL_VOCABULARIES[name]
    )


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise M4NumericError(f"{name} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise M4NumericError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise M4NumericError(f"{name} must be an integer >= {minimum}")
    return value


def _fit_identity(seed: Any, digest: Any) -> None:
    if _integer(seed, "seed") not in m3.TRAINING_SEEDS:
        raise M4NumericError("seed must be one of 1, 2, 3, 4")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise M4NumericError("fit_row_identity_sha256 must be a lowercase SHA256")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M4NumericError(f"{name} must be a mapping")
    return value


def _logits(logits: Any) -> np.ndarray:
    value = np.asarray(logits)
    if value.dtype != np.dtype("float64"):
        raise M4NumericError("logits must have float64 dtype")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 18:
        raise M4NumericError("logits must have nonempty shape (n, 18)")
    if not np.isfinite(value).all():
        raise M4NumericError("logits must be finite")
    return value


def _labels(labels: Any, row_count: int) -> np.ndarray:
    value = np.asarray(labels)
    if value.shape != (row_count,) or value.dtype.kind not in "iu":
        raise M4NumericError(
            "labels must be a one-dimensional integer vector matching logits"
        )
    if np.any(value < 0) or np.any(value >= 18):
        raise M4NumericError("labels must be dense class ids in 0..17")
    return value.astype(np.int64, copy=False)


def _scaled_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = _finite_number(temperature, "temperature")
    if temperature <= 0.0:
        raise M4NumericError("temperature must be positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = (logits - np.max(logits, axis=1, keepdims=True)) / temperature
    if not np.isfinite(scaled).all():
        raise M4NumericError(
            "centering or temperature scaling produced nonfinite logits"
        )
    return scaled


def multiclass_nll(logits: Any, labels: Any, temperature: float = 1.0) -> float:
    """Unweighted per-image multiclass NLL without probability clipping."""
    values = _logits(logits)
    targets = _labels(labels, len(values))
    scaled = _scaled_logits(values, temperature)
    with np.errstate(over="ignore", invalid="ignore"):
        losses = logsumexp(scaled, axis=1) - scaled[np.arange(len(values)), targets]
        result = float(np.mean(losses, dtype=np.float64))
    if not math.isfinite(result) or result < 0:
        raise M4NumericError("multiclass NLL must be finite and nonnegative")
    return result


def validate_temperature_state(state: Mapping[str, Any]) -> None:
    """Validate serialized calibration state without optimizing."""
    state = _mapping(state, "temperature state")
    expected_keys = {
        "schema_version",
        "seed",
        "mode",
        "fit_role",
        "fit_row_count",
        "fit_row_identity_sha256",
        "class_count",
        "weighting",
        "calibration_spec",
        "safeguards",
        "theta",
        "temperature",
        "native_nll",
        "calibrated_nll",
        "optimizer",
    }
    if set(state) != expected_keys:
        raise M4NumericError("temperature state keys differ from the frozen contract")
    if state["schema_version"] != TEMPERATURE_SCHEMA_VERSION:
        raise M4NumericError("unknown temperature state schema")
    _fit_identity(state["seed"], state["fit_row_identity_sha256"])
    _integer(state["fit_row_count"], "fit_row_count")
    if (
        state["mode"] not in MODES
        or state["fit_role"] != "development_train_oof"
        or _integer(state["class_count"], "class_count") != 18
        or state["weighting"] != "equal_per_image"
    ):
        raise M4NumericError("temperature fit provenance differs from contract")
    if (
        state["calibration_spec"] != CALIBRATION_SPEC
        or state["safeguards"] != SAFEGUARDS
    ):
        raise M4NumericError("temperature settings differ from frozen contract")
    theta = _finite_number(state["theta"], "theta")
    temperature = _finite_number(state["temperature"], "temperature")
    if theta <= -5.0 + 1.0e-6 or theta >= 5.0 - 1.0e-6:
        raise M4NumericError("fitted log-temperature saturates a bound")
    if temperature != math.exp(theta):
        raise M4NumericError("temperature must equal exp(theta)")
    native = _finite_number(state["native_nll"], "native_nll")
    calibrated = _finite_number(state["calibrated_nll"], "calibrated_nll")
    if min(native, calibrated) < 0 or calibrated > native + 1.0e-10:
        raise M4NumericError("calibrated NLL is invalid or worse than native NLL")
    optimizer = _mapping(state["optimizer"], "optimizer")
    if set(optimizer) != {"success", "status", "message", "nfev", "nit"}:
        raise M4NumericError("optimizer metadata keys differ from contract")
    if (
        optimizer["success"] is not True
        or _integer(optimizer["status"], "optimizer status", minimum=0) != 0
    ):
        raise M4NumericError("temperature optimizer did not succeed")
    if not isinstance(optimizer["message"], str):
        raise M4NumericError("optimizer message must be a string")
    _integer(optimizer["nfev"], "optimizer nfev")
    if optimizer["nit"] is not None:
        _integer(optimizer["nit"], "optimizer nit")


def fit_expert_temperature(
    logits: Any,
    labels: Any,
    *,
    seed: int,
    mode: str,
    calibration_spec: Mapping[str, Any],
    fit_row_identity_sha256: str,
) -> dict[str, Any]:
    """Fit exactly one bounded scalar temperature to supplied training rows."""
    _fit_identity(seed, fit_row_identity_sha256)
    if mode not in MODES:
        raise M4NumericError("unknown expert mode")
    if dict(_mapping(calibration_spec, "calibration_spec")) != CALIBRATION_SPEC:
        raise M4NumericError("calibration settings differ from frozen contract")
    values = _logits(logits)
    targets = _labels(labels, len(values))
    if np.all(values == values[:, :1]):
        raise M4NumericError(
            "all rows are class-constant; temperature is unidentifiable"
        )
    native = multiclass_nll(values, targets)
    result = minimize_scalar(
        lambda theta: multiclass_nll(values, targets, math.exp(float(theta))),
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1.0e-10, "maxiter": 500},
    )
    theta = float(result.x)
    if not math.isfinite(theta) or theta <= -5.0 + 1.0e-6 or theta >= 5.0 - 1.0e-6:
        raise M4NumericError("fitted log-temperature is nonfinite or saturates a bound")
    state = {
        "schema_version": TEMPERATURE_SCHEMA_VERSION,
        "seed": seed,
        "mode": mode,
        "fit_role": "development_train_oof",
        "fit_row_count": len(values),
        "fit_row_identity_sha256": fit_row_identity_sha256,
        "class_count": 18,
        "weighting": "equal_per_image",
        "calibration_spec": copy.deepcopy(CALIBRATION_SPEC),
        "safeguards": copy.deepcopy(SAFEGUARDS),
        "theta": theta,
        "temperature": math.exp(theta),
        "native_nll": native,
        "calibrated_nll": multiclass_nll(values, targets, math.exp(theta)),
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": (
                int(result.nit) if getattr(result, "nit", None) is not None else None
            ),
        },
    }
    if not np.isclose(
        float(result.fun), state["calibrated_nll"], atol=1e-12, rtol=1e-12
    ):
        raise M4NumericError(
            "optimizer objective differs from recomputed calibrated NLL"
        )
    validate_temperature_state(state)
    return state


def apply_expert_temperature(logits: Any, state: Mapping[str, Any]) -> np.ndarray:
    """Apply a validated temperature without fitting or clipping probabilities."""
    validate_temperature_state(state)
    values = _logits(logits)
    scaled = _scaled_logits(values, state["temperature"])
    with np.errstate(under="ignore"):
        weights = np.exp(scaled)
    probabilities = weights / np.sum(weights, axis=1, keepdims=True)
    if (
        probabilities.dtype != np.float64
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-8)
    ):
        raise M4NumericError("calibrated probabilities fail the probability contract")
    if not np.array_equal(np.argmax(probabilities, axis=1), np.argmax(values, axis=1)):
        raise M4NumericError("calibration changed the expert argmax")
    return probabilities


def _semantic_numeric(frame: pd.DataFrame) -> np.ndarray:
    try:
        m3.validate_router_feature_frame(
            frame, probability_basis=m3.CALIBRATED_PROBABILITY_BASIS
        )
    except (m3.M3Error, TypeError) as error:
        raise M4NumericError(str(error)) from error
    return frame.loc[:, list(NUMERIC_INPUT_COLUMNS)].to_numpy(dtype=np.float64)


def validate_feature_transform_state(state: Mapping[str, Any]) -> None:
    """Validate ordered feature schema and the saved population statistics."""
    state = _mapping(state, "feature transform state")
    expected_keys = {
        "schema_version",
        "seed",
        "fit_role",
        "fit_row_count",
        "fit_row_identity_sha256",
        "feature_schema_sha256",
        "numeric_input_columns",
        "output_columns",
        "categorical_vocabularies",
        "numeric_contract",
        "mean",
        "variance",
        "scale",
    }
    if set(state) != expected_keys:
        raise M4NumericError("feature transform state keys differ from contract")
    _fit_identity(state["seed"], state["fit_row_identity_sha256"])
    _integer(state["fit_row_count"], "fit_row_count")
    if (
        state["schema_version"] != TRANSFORM_SCHEMA_VERSION
        or state["fit_role"] != "development_train_oof"
        or state["feature_schema_sha256"]
        != canonical_sha256(m3.build_router_feature_schema())
        or state["numeric_input_columns"] != list(NUMERIC_INPUT_COLUMNS)
        or state["output_columns"] != list(output_feature_columns())
        or state["categorical_vocabularies"] != CATEGORICAL_VOCABULARIES
        or state["numeric_contract"] != NUMERIC_CONTRACT
    ):
        raise M4NumericError("feature transform schema/settings differ from contract")
    arrays = {}
    for name in ("mean", "variance", "scale"):
        if not isinstance(state[name], list) or len(state[name]) != 25:
            raise M4NumericError(f"transform {name} must be a 25-element list")
        arrays[name] = np.asarray(
            [_finite_number(value, name) for value in state[name]], dtype=np.float64
        )
    if np.any(arrays["variance"] < 0) or np.any(arrays["scale"] <= 0):
        raise M4NumericError("variance must be nonnegative and scale positive")
    expected_scale = np.sqrt(arrays["variance"])
    expected_scale[arrays["variance"] == 0.0] = 1.0
    if not np.array_equal(arrays["scale"], expected_scale):
        raise M4NumericError("saved scales disagree with population variances")


def fit_router_feature_transform(
    semantic_frame: pd.DataFrame,
    *,
    seed: int,
    frozen_schema: Mapping[str, Any],
    fit_row_identity_sha256: str,
) -> dict[str, Any]:
    """Fit seed-local population moments, retaining all fixed one-hot categories."""
    _fit_identity(seed, fit_row_identity_sha256)
    try:
        m3.validate_router_feature_schema(frozen_schema)
    except m3.M3Error as error:
        raise M4NumericError(str(error)) from error
    numeric = _semantic_numeric(semantic_frame)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(numeric, axis=0, dtype=np.float64)
        variance = np.var(numeric, axis=0, dtype=np.float64, ddof=0)
        scale = np.sqrt(variance)
    scale[variance == 0.0] = 1.0
    state = {
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "seed": seed,
        "fit_role": "development_train_oof",
        "fit_row_count": len(semantic_frame),
        "fit_row_identity_sha256": fit_row_identity_sha256,
        "feature_schema_sha256": canonical_sha256(dict(frozen_schema)),
        "numeric_input_columns": list(NUMERIC_INPUT_COLUMNS),
        "output_columns": list(output_feature_columns()),
        "categorical_vocabularies": copy.deepcopy(CATEGORICAL_VOCABULARIES),
        "numeric_contract": copy.deepcopy(NUMERIC_CONTRACT),
        "mean": mean.tolist(),
        "variance": variance.tolist(),
        "scale": scale.tolist(),
    }
    validate_feature_transform_state(state)
    return state


def transform_router_features(
    semantic_frame: pd.DataFrame,
    state: Mapping[str, Any],
) -> np.ndarray:
    """Apply frozen moments and ontology vocabularies to the semantic frame."""
    validate_feature_transform_state(state)
    numeric = _semantic_numeric(semantic_frame)
    matrix = np.zeros((len(semantic_frame), 727), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        matrix[:, :25] = (numeric - np.asarray(state["mean"])) / np.asarray(
            state["scale"]
        )
    offset = 25
    rows = np.arange(len(semantic_frame))
    for name in m3.CATEGORICAL_FEATURES:
        categories = semantic_frame[name].to_numpy(dtype=np.int64)
        matrix[rows, offset + categories] = 1.0
        offset += len(CATEGORICAL_VOCABULARIES[name])
    if offset != 727 or not np.isfinite(matrix).all():
        raise M4NumericError(
            "transformed matrix must be finite float64 with 727 columns"
        )
    return matrix

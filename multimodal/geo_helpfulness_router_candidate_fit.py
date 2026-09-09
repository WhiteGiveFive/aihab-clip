"""Pure CPU fitting for the predeclared M4 router candidates.

This module performs no file discovery or label joins. Callers supply a validated,
canonically ordered, seed-specific OOF matrix and four-state training targets.
Portable prediction and model-state validation live in the separate inference-only
``geo_helpfulness_router_candidate_state`` module.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from typing import Any, Sequence

import numpy as np
from scipy.special import logsumexp
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from multimodal import geo_helpfulness_router_candidate_state as model_state


FIT_PARAMETERS = {
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
PROBABILITY_PARITY_ATOL = 1.0e-12
PROBABILITY_PARITY_RTOL = 1.0e-12
SKLEARN_VERSION = "1.5.2"


class RouterCandidateFitError(ValueError):
    """The candidate cannot be fitted under the frozen numerical contract."""


def _validated_inputs(
    matrix727: Any,
    targets: Any,
    *,
    seed: int,
    variant_id: str,
    C: float,
    feature_columns: Sequence[str],
    protocol_id: str,
    fit_row_identity_sha256: str,
    study_spec_hash: str,
    dataset_manifest_hash: str,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    matrix = np.asarray(matrix727)
    if matrix.dtype != np.dtype("float64"):
        raise RouterCandidateFitError("router inputs must have float64 dtype")
    if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] != 727:
        raise RouterCandidateFitError("router inputs must have shape (n >= 4, 727)")
    if not np.isfinite(matrix).all():
        raise RouterCandidateFitError("router inputs must be finite")
    labels = np.asarray(targets)
    if labels.shape != (len(matrix),) or labels.dtype.kind not in "iu":
        raise RouterCandidateFitError("targets must be a matching integer vector")
    if set(labels.tolist()) != {0, 1, 2, 3}:
        raise RouterCandidateFitError("targets must contain all and only states 0..3")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (1, 2, 3, 4):
        raise RouterCandidateFitError("seed must be an integer in 1..4")
    if (
        isinstance(C, (bool, np.bool_))
        or not isinstance(C, (int, float))
        or not math.isfinite(float(C))
        or float(C) not in model_state.C_GRID
    ):
        raise RouterCandidateFitError("C must belong to the frozen candidate grid")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise RouterCandidateFitError("protocol_id must be a nonempty string")
    for name, digest in (
        ("fit_row_identity_sha256", fit_row_identity_sha256),
        ("study_spec_hash", study_spec_hash),
        ("dataset_manifest_hash", dataset_manifest_hash),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RouterCandidateFitError(f"{name} must be a lowercase SHA256")
    if list(feature_columns) != model_state.full_feature_columns():
        raise RouterCandidateFitError(
            "feature_columns must match the frozen 727-column order"
        )
    try:
        indices = model_state.variant_feature_indices(variant_id)
    except model_state.CandidateStateError as exc:
        raise RouterCandidateFitError(str(exc)) from exc
    if sklearn.__version__ != SKLEARN_VERSION:
        raise RouterCandidateFitError(
            f"fitting requires sklearn {SKLEARN_VERSION}; found {sklearn.__version__}"
        )
    return matrix, labels.astype(np.int64, copy=False), indices, float(C)


def fit_router_candidate(
    matrix727: Any,
    targets: Any,
    *,
    seed: int,
    variant_id: str,
    C: float,
    feature_columns: Sequence[str],
    protocol_id: str,
    fit_row_identity_sha256: str,
    study_spec_hash: str,
    dataset_manifest_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a portable model state and separate, non-selection fit diagnostics.

    Every candidate consumes the same full input schema. Ablations are fixed
    column projections, never data-driven feature searches. No row/class weighting
    or resampling is applied. Single-thread BLAS is an operational reproducibility
    control in addition to the frozen estimator's ``n_jobs=1``.
    """
    matrix, labels, indices, c_value = _validated_inputs(
        matrix727,
        targets,
        seed=seed,
        variant_id=variant_id,
        C=C,
        feature_columns=feature_columns,
        protocol_id=protocol_id,
        fit_row_identity_sha256=fit_row_identity_sha256,
        study_spec_hash=study_spec_hash,
        dataset_manifest_hash=dataset_manifest_hash,
    )
    features = np.ascontiguousarray(matrix[:, indices], dtype=np.float64)
    params = {**FIT_PARAMETERS, "C": c_value}
    estimator = LogisticRegression(**params)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator.fit(features, labels)
        fit_seconds = time.perf_counter() - started
        recorded_warnings = []
        for warning in captured:
            message = str(warning.message)
            if issubclass(warning.category, ConvergenceWarning):
                raise RouterCandidateFitError(f"router convergence warning: {message}")
            if not (
                issubclass(warning.category, FutureWarning)
                and "multi_class" in message
                and "deprecated" in message
            ):
                raise RouterCandidateFitError(
                    f"unexpected fitting warning ({warning.category.__name__}): {message}"
                )
            recorded_warnings.append(
                {"category": warning.category.__name__, "message": message}
            )
        iterations = np.asarray(estimator.n_iter_)
        if (
            iterations.shape != (1,)
            or iterations.dtype.kind not in "iu"
            or np.any(iterations < 0)
            or np.any(iterations >= FIT_PARAMETERS["max_iter"])
        ):
            raise RouterCandidateFitError(
                "router reached the iteration cap or has invalid n_iter"
            )
        if not np.array_equal(estimator.classes_, np.arange(4)):
            raise RouterCandidateFitError(
                "estimator target classes do not match states 0..3"
            )
        if (
            estimator.coef_.shape != (4, len(indices))
            or estimator.intercept_.shape != (4,)
            or estimator.coef_.dtype != np.dtype("float64")
            or estimator.intercept_.dtype != np.dtype("float64")
            or not np.isfinite(estimator.coef_).all()
            or not np.isfinite(estimator.intercept_).all()
        ):
            raise RouterCandidateFitError(
                "fitted parameters have invalid shape/dtype or nonfinite values"
            )
        state = model_state.seal_candidate_state(
            {
                "schema_version": model_state.SCHEMA_VERSION,
                "protocol_id": protocol_id,
                "training_seed": seed,
                "variant_id": variant_id,
                "C": c_value,
                "C_hex": c_value.hex(),
                "target_order": list(model_state.TARGET_ORDER),
                "classes": [0, 1, 2, 3],
                "fit_role": "development_train_oof",
                "fit_row_count": len(matrix),
                "fit_row_identity_sha256": fit_row_identity_sha256,
                "study_spec_sha256": study_spec_hash,
                "dataset_manifest_sha256": dataset_manifest_hash,
                "feature_input_names": list(feature_columns),
                "feature_indices": indices,
                "feature_names": [feature_columns[index] for index in indices],
                "fit_parameters": params,
                "implementation": {
                    "library": "sklearn",
                    "version": sklearn.__version__,
                    "compute_dtype": "float64",
                    "threadpool_limit": 1,
                },
                "coef": estimator.coef_.tolist(),
                "intercept": estimator.intercept_.tolist(),
                "n_iter": iterations.tolist(),
            }
        )
        expected_probabilities = estimator.predict_proba(features)
        portable_probabilities = model_state.predict_router_probabilities(matrix, state)
        if not np.allclose(
            expected_probabilities,
            portable_probabilities,
            atol=PROBABILITY_PARITY_ATOL,
            rtol=PROBABILITY_PARITY_RTOL,
        ) or not np.array_equal(
            np.argmax(portable_probabilities, axis=1), estimator.predict(features)
        ):
            raise RouterCandidateFitError("portable predictions disagree with sklearn")
        scores = model_state.decision_function(matrix, state)
        centered = scores - scores.max(axis=1, keepdims=True)
        nll = float(
            np.mean(
                logsumexp(centered, axis=1) - centered[np.arange(len(labels)), labels],
                dtype=np.float64,
            )
        )
        if not math.isfinite(nll) or nll < 0:
            raise RouterCandidateFitError(
                "training log loss must be finite and nonnegative"
            )
    diagnostics = {
        "training_log_loss": nll,
        "training_accuracy": float(
            np.mean(portable_probabilities.argmax(axis=1) == labels)
        ),
        "probability_parity_max_abs": float(
            np.max(np.abs(portable_probabilities - expected_probabilities))
        ),
        "n_iter": iterations.tolist(),
        "warnings": recorded_warnings,
        "fit_seconds": float(fit_seconds),
        "diagnostic_role": "in_sample_training_fit_only_not_validation_selection",
    }
    return state, diagnostics


__all__ = ["FIT_PARAMETERS", "RouterCandidateFitError", "fit_router_candidate"]

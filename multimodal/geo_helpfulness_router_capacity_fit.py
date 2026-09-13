"""Fit/export helpers for the frozen development router capacity round.

Callers supply training matrices/targets and label-blind validation matrices.
No artifact discovery, validation labels, early stopping, or threshold selection
is available here. The separate state module is sufficient for later scoring.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time
import warnings

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from multimodal.geo_helpfulness_router_candidate_state import (
    TARGET_ORDER,
    full_feature_columns,
    predict_router_probabilities,
    validate_candidate_state,
    variant_feature_indices,
)
from multimodal.geo_helpfulness_router_capacity_state import (
    SCHEMA_VERSION,
    predict_margin,
    predict_state,
    seal_state,
)


SKLEARN_VERSION = "1.5.2"
XGBOOST_VERSION = "2.1.4"
RF_SHARED = {
    "n_estimators": 500,
    "max_features": 0.5,
    "bootstrap": True,
    "criterion": "log_loss",
    "class_weight": None,
    "n_jobs": 1,
    "oob_score": False,
    "max_samples": None,
}
XGB_SHARED = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0,
    "max_bin": 256,
    "tree_method": "hist",
    "objective": "multi:softprob",
    "num_class": 4,
    "base_score": 0.5,
    "n_jobs": 1,
}
PARITY_TOLERANCE = {"lr": 1e-12, "rf": 1e-12, "xgb": 1e-6}


class CapacityFitError(ValueError):
    """Inputs or a fitted/exported learner violate the fixed round contract."""


def _inputs(train_x, target, val_x, seed, spec):
    for name, matrix in (("training", train_x), ("validation", val_x)):
        if (
            not isinstance(matrix, np.ndarray)
            or matrix.dtype != np.float64
            or matrix.ndim != 2
            or matrix.shape[1] != 727
            or not len(matrix)
            or not np.isfinite(matrix).all()
        ):
            raise CapacityFitError(
                f"{name} matrix must be nonempty finite float64 (n,727)"
            )
    target = np.asarray(target)
    if (
        target.shape != (len(train_x),)
        or target.dtype.kind not in "iu"
        or set(target.tolist()) != {0, 1, 2, 3}
    ):
        raise CapacityFitError(
            "training targets must contain exactly the four fixed states"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (1, 2, 3, 4):
        raise CapacityFitError("training seed must be in 1..4")
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("config_id"), str)
        or not spec["config_id"].strip()
        or spec.get("family") not in ("lr", "rf", "xgb")
        or not isinstance(spec.get("parameters"), dict)
    ):
        raise CapacityFitError("invalid capacity candidate specification")
    indices = variant_feature_indices(spec["variant"])
    return target.astype(np.int64, copy=False), indices


def resolve_parameters(spec: dict, seed: int) -> dict:
    """Fill fixed defaults and reject any unregistered learner setting."""
    family, supplied = spec["family"], dict(spec["parameters"])
    if family == "lr":
        if set(supplied) != {"C"} or supplied["C"] not in (0.01, 0.1, 1.0, 10.0):
            raise CapacityFitError("LR must reuse one of the four frozen C candidates")
        return {"C": float(supplied["C"])}
    shared = RF_SHARED if family == "rf" else XGB_SHARED
    varying = (
        {"max_depth", "min_samples_leaf"}
        if family == "rf"
        else {"max_depth", "reg_lambda"}
    )
    if not varying.issubset(supplied) or set(supplied) - (
        set(shared) | varying | {"random_state"}
    ):
        raise CapacityFitError("missing or unregistered learner hyperparameters")
    parameters = {**shared, "random_state": seed, **supplied}
    if (
        any(parameters[key] != value for key, value in shared.items())
        or parameters["random_state"] != seed
    ):
        raise CapacityFitError("fixed learner parameters or training seed changed")
    if family == "rf":
        if parameters["max_depth"] not in (4, 8) or parameters[
            "min_samples_leaf"
        ] not in (20, 5):
            raise CapacityFitError("RF depth/leaf values lie outside the frozen grid")
    elif (parameters["max_depth"], parameters["reg_lambda"]) not in (
        (2, 1),
        (3, 1),
        (3, 10),
        (4, 10),
    ):
        raise CapacityFitError(
            "XGB depth/regularization pair lies outside the frozen grid"
        )
    return parameters


def _metadata(
    spec: dict, seed: int, row_count: int, parameters: dict, model: dict
) -> dict:
    family = spec["family"]
    return {
        "schema_version": SCHEMA_VERSION,
        "config_id": spec["config_id"],
        "family": family,
        "variant": spec["variant"],
        "training_seed": seed,
        "parameters": parameters,
        "feature_input_names": full_feature_columns(),
        "feature_indices": variant_feature_indices(spec["variant"]),
        "classes": [0, 1, 2, 3],
        "target_order": list(TARGET_ORDER),
        "fit_role": "development_train_oof",
        "fit_row_count": row_count,
        "implementation": {
            "library": "xgboost" if family == "xgb" else "sklearn",
            "version": XGBOOST_VERSION if family == "xgb" else SKLEARN_VERSION,
            "split_input_dtype": "float64" if family == "lr" else "float32",
            "probability_output_dtype": "float64",
            "threadpool_limit": 1,
        },
        "model": model,
    }


def _export_rf(estimator, spec, seed, row_count, parameters):
    if not np.array_equal(estimator.classes_, np.arange(4)):
        raise CapacityFitError("RF class order differs from four-state target order")
    trees = []
    for fitted in estimator.estimators_:
        tree = fitted.tree_
        if tree.value.shape != (tree.node_count, 1, 4):
            raise CapacityFitError("RF export has an unexpected node value shape")
        values = tree.value[:, 0, :].astype(np.float64, copy=True)
        sums = values.sum(axis=1, keepdims=True)
        if not np.isfinite(values).all() or np.any(sums <= 0):
            raise CapacityFitError("RF export has empty/nonfinite node probabilities")
        values /= sums
        trees.append(
            {
                "left": tree.children_left.tolist(),
                "right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": values.tolist(),
            }
        )
    return seal_state(_metadata(spec, seed, row_count, parameters, {"trees": trees}))


def _export_xgb(estimator, spec, seed, row_count, parameters):
    raw = estimator.get_booster().save_raw(raw_format="json")
    native = json.loads(raw.decode("utf-8"))
    if native.get("version") != [2, 1, 4]:
        raise CapacityFitError("XGB native model version differs from pinned 2.1.4")
    learner = native["learner"]
    model_parameters = learner["learner_model_param"]
    feature_count = len(variant_feature_indices(spec["variant"]))
    if (
        int(model_parameters["num_class"]) != 4
        or int(model_parameters["num_feature"]) != feature_count
        or int(model_parameters["num_target"]) != 1
        or learner["objective"]["name"] != "multi:softprob"
        or learner["gradient_booster"]["name"] != "gbtree"
    ):
        raise CapacityFitError("XGB model objective/features/targets/booster changed")
    ensemble = learner["gradient_booster"]["model"]
    if int(ensemble["gbtree_model_param"]["num_parallel_tree"]) != 1:
        raise CapacityFitError("parallel-tree XGB models are outside this round")
    trees = []
    for index, native_tree in enumerate(ensemble["trees"]):
        if (
            native_tree["id"] != index
            or int(native_tree["tree_param"]["num_feature"]) != feature_count
            or int(native_tree["tree_param"]["size_leaf_vector"]) != 1
            or any(native_tree["split_type"])
            or any(
                native_tree[key]
                for key in (
                    "categories",
                    "categories_nodes",
                    "categories_segments",
                    "categories_sizes",
                )
            )
        ):
            raise CapacityFitError(
                "unsupported XGB categorical/vector tree representation"
            )
        split = np.asarray(native_tree["split_conditions"], dtype=np.float32)
        leaves = np.asarray(native_tree["left_children"]) == -1
        values = np.where(leaves, split, np.float32(0))
        trees.append(
            {
                "left": native_tree["left_children"],
                "right": native_tree["right_children"],
                "feature": native_tree["split_indices"],
                "threshold": split.tolist(),
                "value": values.tolist(),
                "default_left": native_tree["default_left"],
            }
        )
    model = {
        "trees": trees,
        "tree_info": ensemble["tree_info"],
        "base_score": float(model_parameters["base_score"]),
        "objective": "multi:softprob",
        "native_model_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return seal_state(_metadata(spec, seed, row_count, parameters, model))


def _parity(expected: np.ndarray, observed: np.ndarray, family: str, name: str) -> dict:
    tolerance = PARITY_TOLERANCE[family]
    if (
        expected.shape != observed.shape
        or not np.isfinite(expected).all()
        or not np.allclose(expected, observed, atol=tolerance, rtol=tolerance)
        or not np.array_equal(expected.argmax(axis=1), observed.argmax(axis=1))
    ):
        raise CapacityFitError(
            f"portable {family} {name} probability/prediction parity failed"
        )
    return {
        "maximum_probability_abs_error": float(np.max(np.abs(expected - observed))),
        "probability_atol": tolerance,
        "probability_rtol": tolerance,
        "class_prediction_exact": True,
    }


def fit_candidate(train_x, target, val_x, spec, seed):
    """Return (portable state, train q, label-blind validation q, diagnostics)."""
    target, indices = _inputs(train_x, target, val_x, seed, spec)
    parameters = resolve_parameters(spec, seed)
    family = spec["family"]
    if sklearn.__version__ != SKLEARN_VERSION:
        raise CapacityFitError(f"capacity fitting requires sklearn {SKLEARN_VERSION}")
    train_features = np.ascontiguousarray(train_x[:, indices], dtype=np.float64)
    val_features = np.ascontiguousarray(val_x[:, indices], dtype=np.float64)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if family == "lr":
            if "existing_lr_state" not in spec:
                raise CapacityFitError(
                    "LR candidates must reuse an explicit frozen existing_lr_state"
                )
            existing = deepcopy(spec["existing_lr_state"])
            validate_candidate_state(existing)
            state = seal_state(
                _metadata(spec, seed, len(train_x), parameters, existing)
            )
            expected_train = predict_router_probabilities(train_x, existing)
            expected_val = predict_router_probabilities(val_x, existing)
            captured = []
        else:
            if family == "rf":
                estimator = RandomForestClassifier(**parameters)
            else:
                import xgboost

                if xgboost.__version__ != XGBOOST_VERSION:
                    raise CapacityFitError(
                        f"capacity fitting requires XGBoost {XGBOOST_VERSION}"
                    )
                estimator = xgboost.XGBClassifier(**parameters)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                # No validation rows or labels enter the estimator's fit call.
                estimator.fit(train_features, target)
            if captured:
                messages = [
                    f"{item.category.__name__}: {item.message}" for item in captured
                ]
                raise CapacityFitError(
                    f"unexpected capacity fitting warnings: {messages}"
                )
            if family == "rf":
                state = _export_rf(estimator, spec, seed, len(train_x), parameters)
            else:
                state = _export_xgb(estimator, spec, seed, len(train_x), parameters)
            expected_train = estimator.predict_proba(train_features)
            expected_val = estimator.predict_proba(val_features)
        train_probability = predict_state(train_x, state)
        val_probability = predict_state(val_x, state)
        parity = {
            "train": _parity(expected_train, train_probability, family, "training"),
            "validation_features_only": _parity(
                expected_val, val_probability, family, "validation"
            ),
        }
        if family == "xgb":
            for name, full, projected in (
                ("train", train_x, train_features),
                ("validation_features_only", val_x, val_features),
            ):
                native_margin = estimator.predict(projected, output_margin=True)
                pure_margin = predict_margin(full, state)
                if not np.allclose(native_margin, pure_margin, atol=1e-6, rtol=1e-6):
                    raise CapacityFitError("portable XGB raw-margin parity failed")
                parity[name]["maximum_margin_abs_error"] = float(
                    np.max(np.abs(native_margin - pure_margin))
                )
    diagnostics = {
        "config_id": spec["config_id"],
        "family": family,
        "variant": spec["variant"],
        "training_seed": seed,
        "fit_performed": family != "lr",
        "existing_lr_reused": family == "lr",
        "fit_role": "development_train_oof",
        "fit_row_count": len(train_x),
        "feature_count": len(indices),
        "validation_labels_accessed": False,
        "validation_features_used_for_fitting": False,
        "early_stopping_used": False,
        "sample_or_class_weighting_used": False,
        "threadpool_limit": 1,
        "fit_and_export_seconds": float(time.perf_counter() - started),
        "probability_parity": parity,
        "warnings": [],
        "state_sha256": state["state_sha256"],
    }
    if family == "lr":
        diagnostics["reused_source_state_sha256"] = state["model"]["state_sha256"]
    else:
        trees = state["model"]["trees"]
        diagnostics["tree_count"] = len(trees)
        diagnostics["total_tree_nodes"] = sum(len(tree["left"]) for tree in trees)
        diagnostics["total_leaves"] = sum(
            sum(node == -1 for node in tree["left"]) for tree in trees
        )
    return state, train_probability, val_probability, diagnostics


__all__ = [
    "CapacityFitError",
    "RF_SHARED",
    "XGB_SHARED",
    "resolve_parameters",
    "fit_candidate",
]

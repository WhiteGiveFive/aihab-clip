"""Pure NumPy inference for the development-only router capacity round.

States contain ordinary JSON arrays. Loading or applying them never imports a
learner, deserializes a pickle, fits a transform, or reads an artifact path.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

import numpy as np

from multimodal.geo_helpfulness_router_candidate_state import (
    TARGET_ORDER,
    full_feature_columns,
    predict_router_probabilities,
    validate_candidate_state,
    variant_feature_indices,
)


SCHEMA_VERSION = "geo_helpfulness.router_capacity_state.v1"
FAMILIES = ("lr", "rf", "xgb")
STATE_KEYS = {
    "schema_version",
    "config_id",
    "family",
    "variant",
    "training_seed",
    "parameters",
    "feature_input_names",
    "classes",
    "target_order",
    "fit_role",
    "fit_row_count",
    "feature_indices",
    "implementation",
    "model",
    "state_sha256",
}


class CapacityStateError(ValueError):
    """A portable capacity model or its input violates the fixed contract."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def seal_state(state: dict) -> dict:
    result = deepcopy(state)
    result.pop("state_sha256", None)
    result["state_sha256"] = canonical_hash(result)
    validate_state(result)
    return result


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _array(value: Any, shape: tuple, name: str, *, integer=False) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CapacityStateError(f"invalid {name} array") from exc
    if result.shape != shape or result.dtype.kind not in ("iu" if integer else "iuf"):
        raise CapacityStateError(f"invalid {name} shape or numeric dtype")
    if not np.isfinite(result).all():
        raise CapacityStateError(f"nonfinite {name}")
    return result.astype(np.int64 if integer else np.float64, copy=False)


def _tree(tree: dict, *, family: str, feature_count: int) -> dict[str, np.ndarray]:
    keys = {"left", "right", "feature", "threshold", "value"}
    if family == "xgb":
        keys.add("default_left")
    if not isinstance(tree, dict) or set(tree) != keys:
        raise CapacityStateError("tree has missing or unknown fields")
    size = len(tree["left"])
    if size < 1:
        raise CapacityStateError("empty tree")
    arrays = {
        key: _array(tree[key], (size,), key, integer=True)
        for key in ("left", "right", "feature")
    }
    arrays["threshold"] = _array(tree["threshold"], (size,), "threshold")
    left, right, feature = (arrays[key] for key in ("left", "right", "feature"))
    leaf = left == -1
    if not np.array_equal(leaf, right == -1):
        raise CapacityStateError("tree leaf children disagree")
    node_ids = np.arange(size)
    if (
        np.any(left[~leaf] <= node_ids[~leaf])
        or np.any(right[~leaf] <= node_ids[~leaf])
        or np.any(left[~leaf] >= size)
        or np.any(right[~leaf] >= size)
        or np.any(feature[~leaf] < 0)
        or np.any(feature[~leaf] >= feature_count)
    ):
        raise CapacityStateError("tree child or feature index is invalid")
    children = np.concatenate([left[~leaf], right[~leaf]])
    if not np.array_equal(np.sort(children), np.arange(1, size)):
        raise CapacityStateError(
            "tree is cyclic, disconnected, or has shared child nodes"
        )
    if family == "rf":
        values = _array(tree["value"], (size, 4), "leaf probabilities")
        if np.any(values < 0) or not np.allclose(
            values.sum(axis=1), 1, atol=1e-12, rtol=0
        ):
            raise CapacityStateError("RF node probabilities must sum to one")
        arrays["value"] = values
    else:
        arrays["value"] = _array(tree["value"], (size,), "leaf margins")
        default = _array(tree["default_left"], (size,), "default_left", integer=True)
        if not np.isin(default, [0, 1]).all():
            raise CapacityStateError("invalid default branch flags")
        arrays["default_left"] = default
        for key in ("threshold", "value"):
            with np.errstate(over="ignore"):
                arrays[key] = arrays[key].astype(np.float32)
            if not np.isfinite(arrays[key]).all():
                raise CapacityStateError("XGB values exceed finite float32 range")
    return arrays


def validate_state(state: dict) -> None:
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise CapacityStateError("capacity state has missing or unknown fields")
    if state["schema_version"] != SCHEMA_VERSION or state["family"] not in FAMILIES:
        raise CapacityStateError("capacity schema or family is invalid")
    if not isinstance(state["config_id"], str) or not state["config_id"].strip():
        raise CapacityStateError("candidate ID must be nonempty text")
    indices = variant_feature_indices(state["variant"])
    if not _integer(state["training_seed"]) or state["training_seed"] not in (
        1,
        2,
        3,
        4,
    ):
        raise CapacityStateError("training seed must be in 1..4")
    if (
        state["feature_input_names"] != full_feature_columns()
        or state["feature_indices"] != indices
        or state["classes"] != [0, 1, 2, 3]
        or any(not _integer(value) for value in state["classes"])
        or state["target_order"] != list(TARGET_ORDER)
        or state["fit_role"] != "development_train_oof"
        or not _integer(state["fit_row_count"])
        or state["fit_row_count"] < 4
        or not isinstance(state["parameters"], dict)
    ):
        raise CapacityStateError(
            "capacity feature, target, or training contract is invalid"
        )
    family, model = state["family"], state["model"]
    implementation = state["implementation"]
    if (
        not isinstance(implementation, dict)
        or implementation.get("threadpool_limit") != 1
        or implementation.get("library")
        != ("xgboost" if family == "xgb" else "sklearn")
        or implementation.get("version") != ("2.1.4" if family == "xgb" else "1.5.2")
        or implementation.get("split_input_dtype")
        != ("float64" if family == "lr" else "float32")
    ):
        raise CapacityStateError("capacity implementation contract changed")
    if family == "lr":
        validate_candidate_state(model)
        if (
            model["variant_id"] != state["variant"]
            or model["training_seed"] != state["training_seed"]
            or model["fit_row_count"] != state["fit_row_count"]
            or model["C"] != state["parameters"].get("C")
        ):
            raise CapacityStateError("reused LR does not match candidate metadata")
    elif family == "rf":
        if (
            not isinstance(model, dict)
            or set(model) != {"trees"}
            or not isinstance(model["trees"], list)
            or not model["trees"]
            or len(model["trees"]) != state["parameters"].get("n_estimators")
        ):
            raise CapacityStateError("RF ensemble contract is invalid")
        for tree in model["trees"]:
            _tree(tree, family=family, feature_count=len(indices))
    else:
        if not isinstance(model, dict) or set(model) != {
            "trees",
            "tree_info",
            "base_score",
            "objective",
            "native_model_sha256",
        }:
            raise CapacityStateError("XGB ensemble contract is invalid")
        if (
            model["objective"] != "multi:softprob"
            or not isinstance(model["base_score"], (int, float))
            or isinstance(model["base_score"], bool)
            or not np.isfinite(model["base_score"])
            or not isinstance(model["trees"], list)
            or not model["trees"]
        ):
            raise CapacityStateError("XGB objective, bias, or trees are invalid")
        info = _array(
            model["tree_info"], (len(model["trees"]),), "tree_info", integer=True
        )
        if (
            len(info) != 4 * state["parameters"].get("n_estimators", -1)
            or not np.array_equal(info, np.tile(np.arange(4), len(info) // 4))
            or model["base_score"] != state["parameters"].get("base_score", 0.5)
            or not isinstance(model["native_model_sha256"], str)
            or len(model["native_model_sha256"]) != 64
        ):
            raise CapacityStateError(
                "XGB tree groups, boosting rounds, or base score changed"
            )
        for tree in model["trees"]:
            _tree(tree, family=family, feature_count=len(indices))
    expected = canonical_hash(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    if state["state_sha256"] != expected:
        raise CapacityStateError("capacity model self-hash mismatch")


def _matrix(matrix: Any) -> np.ndarray:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.float64
        or matrix.ndim != 2
        or matrix.shape[1] != 727
        or not len(matrix)
        or not np.isfinite(matrix).all()
    ):
        raise CapacityStateError(
            "capacity inputs must be nonempty finite float64 (n,727)"
        )
    return matrix


def _leaves(
    matrix: np.ndarray, tree: dict[str, np.ndarray], *, inclusive: bool
) -> np.ndarray:
    nodes = np.zeros(len(matrix), dtype=np.int64)
    active = np.ones(len(matrix), dtype=bool)
    while active.any():
        rows = np.flatnonzero(active)
        current = nodes[rows]
        leaf = tree["left"][current] == -1
        active[rows[leaf]] = False
        rows, current = rows[~leaf], current[~leaf]
        if not len(rows):
            break
        value = matrix[rows, tree["feature"][current]]
        # sklearn promotes float32 feature values to the float64 thresholds
        # stored by its trees. XGBoost compares both sides in float32.
        condition = (
            value <= tree["threshold"][current]
            if inclusive
            else value < tree["threshold"][current]
        )
        nodes[rows] = np.where(condition, tree["left"][current], tree["right"][current])
    return nodes


def predict_margin(matrix: np.ndarray, state: dict) -> np.ndarray:
    """Reproduce XGBoost's float32 base-plus-tree margin accumulation."""
    validate_state(state)
    matrix = _matrix(matrix)
    if state["family"] != "xgb":
        raise CapacityStateError("raw capacity margins are defined only for XGB")
    with np.errstate(over="ignore"):
        features = matrix[:, state["feature_indices"]].astype(np.float32)
    if not np.isfinite(features).all():
        raise CapacityStateError("tree input overflows float32")
    model = state["model"]
    margins = np.full((len(matrix), 4), model["base_score"], dtype=np.float32)
    for tree, group in zip(model["trees"], model["tree_info"]):
        arrays = _tree(tree, family="xgb", feature_count=features.shape[1])
        leaves = _leaves(features, arrays, inclusive=False)
        margins[:, group] += arrays["value"][leaves]
    if not np.isfinite(margins).all():
        raise CapacityStateError("nonfinite XGB margins")
    return margins


def predict_state(matrix: np.ndarray, state: dict) -> np.ndarray:
    """Apply one portable learner and return float64 rescue/harm/correct/wrong probabilities."""
    validate_state(state)
    matrix = _matrix(matrix)
    family = state["family"]
    if family == "lr":
        return predict_router_probabilities(matrix, state["model"])
    if family == "xgb":
        values = predict_margin(matrix, state).astype(np.float64)
        values -= values.max(axis=1, keepdims=True)
        weights = np.exp(values)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
    else:
        with np.errstate(over="ignore"):
            features = matrix[:, state["feature_indices"]].astype(np.float32)
        if not np.isfinite(features).all():
            raise CapacityStateError("tree input overflows float32")
        probabilities = np.zeros((len(matrix), 4), dtype=np.float64)
        for tree in state["model"]["trees"]:
            arrays = _tree(tree, family="rf", feature_count=features.shape[1])
            leaves = _leaves(features, arrays, inclusive=True)
            probabilities += arrays["value"][leaves]
        probabilities /= len(state["model"]["trees"])
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
    ):
        raise CapacityStateError("invalid capacity probabilities")
    return probabilities


__all__ = [
    "CapacityStateError",
    "SCHEMA_VERSION",
    "FAMILIES",
    "canonical_hash",
    "seal_state",
    "validate_state",
    "predict_state",
    "predict_margin",
]

"""Strict, pure NumPy application of the frozen small-MLP router states."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

import numpy as np

from multimodal.geo_helpfulness_router_candidate_state import (
    TARGET_ORDER,
    full_feature_columns,
    variant_feature_indices,
)


SCHEMA_VERSION = "geo_helpfulness.router_mlp_state.v1"
TORCH_VERSION = "2.5.1+cu124"
ARCHITECTURES = ((32,), (64,), (32, 16), (64, 32))
MLP_SHARED = {
    "activation": "relu",
    "dropout": 0.1,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "bias_weight_decay": 0.0,
    "batch_size": 128,
    "epochs": 300,
    "shuffle": True,
    "early_stopping": False,
    "checkpoint": "final_epoch",
    "dtype": "float64",
    "device": "cpu",
    "num_threads": 1,
}
IMPLEMENTATION = {
    "library": "torch",
    "version": TORCH_VERSION,
    "device": "cpu",
    "training_dtype": "float64",
    "prediction_dtype": "float64",
    "num_threads": 1,
    "deterministic_algorithms": True,
    "batch_normalization": False,
    "residual_connections": False,
    "learned_embeddings": False,
    "optimizer_defaults": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    },
}
STATE_KEYS = {
    "schema_version",
    "config_id",
    "family",
    "variant",
    "training_seed",
    "parameters",
    "feature_input_names",
    "feature_indices",
    "classes",
    "target_order",
    "fit_role",
    "fit_row_count",
    "initialization_seed",
    "shuffle_seed",
    "dropout_seed",
    "parameter_count",
    "implementation",
    "layers",
    "state_sha256",
}


class MLPStateError(ValueError):
    """The model or its inputs violate the declared small-MLP contract."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_parameters(spec: dict) -> dict:
    supplied = spec.get("parameters")
    if not isinstance(supplied, dict) or "hidden_widths" not in supplied:
        raise MLPStateError("MLP specification requires hidden_widths")
    widths = supplied["hidden_widths"]
    if (
        not isinstance(widths, list)
        or any(not _integer(value) for value in widths)
        or tuple(widths) not in ARCHITECTURES
        or set(supplied) - ({"hidden_widths"} | set(MLP_SHARED))
    ):
        raise MLPStateError("architecture or settings are outside the frozen MLP grid")
    result = {**MLP_SHARED, **supplied}
    if any(
        type(result[name]) is not type(value) or result[name] != value
        for name, value in MLP_SHARED.items()
    ):
        raise MLPStateError("a fixed MLP training setting changed")
    return result


def parameter_count(variant: str, hidden_widths: list[int]) -> int:
    dimensions = [len(variant_feature_indices(variant)), *hidden_widths, 4]
    return sum((left + 1) * right for left, right in zip(dimensions, dimensions[1:]))


def seal_state(state: dict) -> dict:
    result = deepcopy(state)
    result.pop("state_sha256", None)
    result["state_sha256"] = canonical_hash(result)
    validate_state(result)
    return result


def validate_state(state: dict) -> None:
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise MLPStateError("MLP state has missing or unknown keys")
    if (
        state["schema_version"] != SCHEMA_VERSION
        or state["family"] != "mlp"
        or not isinstance(state["config_id"], str)
        or not state["config_id"].strip()
        or not _integer(state["training_seed"])
        or state["training_seed"] not in (1, 2, 3, 4)
        or any(
            not _integer(state[name]) or state[name] != state["training_seed"]
            for name in ("initialization_seed", "shuffle_seed", "dropout_seed")
        )
    ):
        raise MLPStateError("MLP identity or deterministic seeds are invalid")
    indices = variant_feature_indices(state["variant"])
    if (
        state["feature_input_names"] != full_feature_columns()
        or state["feature_indices"] != indices
        or any(not _integer(value) for value in state["feature_indices"])
        or state["classes"] != [0, 1, 2, 3]
        or any(not _integer(value) for value in state["classes"])
        or state["target_order"] != list(TARGET_ORDER)
        or state["fit_role"] != "development_train_oof"
        or not _integer(state["fit_row_count"])
        or state["fit_row_count"] < 4
    ):
        raise MLPStateError("MLP feature or training target contract changed")
    parameters = resolve_parameters(state)
    if parameters != state["parameters"]:
        raise MLPStateError("MLP state must contain all resolved training settings")
    if state["implementation"] != IMPLEMENTATION:
        raise MLPStateError("MLP implementation identity changed")
    expected_count = parameter_count(state["variant"], parameters["hidden_widths"])
    if (
        not _integer(state["parameter_count"])
        or state["parameter_count"] != expected_count
    ):
        raise MLPStateError("MLP parameter count does not match architecture")
    dimensions = [len(indices), *parameters["hidden_widths"], 4]
    if (
        not isinstance(state["layers"], list)
        or len(state["layers"]) != len(dimensions) - 1
    ):
        raise MLPStateError("MLP layer count differs from architecture")
    for layer, width_in, width_out in zip(state["layers"], dimensions, dimensions[1:]):
        if not isinstance(layer, dict) or set(layer) != {"weight", "bias"}:
            raise MLPStateError("MLP layer fields are invalid")
        for name, shape in (("weight", (width_out, width_in)), ("bias", (width_out,))):
            try:
                values = np.asarray(layer[name])
            except (TypeError, ValueError) as exc:
                raise MLPStateError("MLP parameter array is malformed") from exc
            if (
                values.shape != shape
                or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()
            ):
                raise MLPStateError(f"MLP {name} shape/dtype/finiteness mismatch")
    expected = canonical_hash(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    if state["state_sha256"] != expected:
        raise MLPStateError("MLP state hash mismatch")


def predict_logits(matrix: np.ndarray, state: dict) -> np.ndarray:
    validate_state(state)
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.float64
        or matrix.ndim != 2
        or matrix.shape[1] != 727
        or not len(matrix)
        or not np.isfinite(matrix).all()
    ):
        raise MLPStateError("MLP input must be nonempty finite float64 (n,727)")
    values = matrix[:, state["feature_indices"]]
    with np.errstate(over="ignore", invalid="ignore"):
        for index, layer in enumerate(state["layers"]):
            values = values @ np.asarray(
                layer["weight"], dtype=np.float64
            ).T + np.asarray(layer["bias"], dtype=np.float64)
            if index < len(state["layers"]) - 1:
                values = np.maximum(values, 0)
            if not np.isfinite(values).all():
                raise MLPStateError("MLP produced nonfinite activations")
    return values


def predict_state(matrix: np.ndarray, state: dict) -> np.ndarray:
    logits = predict_logits(matrix, state)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability = weights / weights.sum(axis=1, keepdims=True)
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1, atol=1e-12, rtol=0
    ):
        raise MLPStateError("MLP probabilities are nonfinite or unnormalized")
    return probability


__all__ = [
    "SCHEMA_VERSION",
    "TORCH_VERSION",
    "ARCHITECTURES",
    "MLP_SHARED",
    "IMPLEMENTATION",
    "MLPStateError",
    "resolve_parameters",
    "parameter_count",
    "seal_state",
    "validate_state",
    "predict_logits",
    "predict_state",
]

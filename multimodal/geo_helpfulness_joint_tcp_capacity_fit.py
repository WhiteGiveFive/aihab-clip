"""Additive capacity study: deterministic deeper joint four-state/TCP routers.

Only the training targets enter optimization.  Authoritative predictions replay
the exported JSON weights using the same native CPU float32 operations; float64
NumPy replay is a separately reported numerical diagnostic.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from .geo_helpfulness_joint_tcp_fit import (
    DEFAULT_PARAMETERS as BASE_PARAMETERS,
    JointTCPFitError,
    PREDICTION_KEYS,
    TARGET_ORDER,
    TCP_ORDER,
    TORCH_VERSION,
    _components,
    _eval_train,
    _execution,
    _hash,
    _integer,
    _matrix,
    _optimizer,
    _parity_summary,
    _train_epoch,
)

SCHEMA_VERSION = "geo_helpfulness.joint_tcp_capacity_router_state.v1"
APPROVED_HIDDEN_WIDTHS = ((64, 32), (128, 64), (64, 32, 16), (128, 64, 32))
# A single layer is accepted only to verify equivalence with the sealed parent.
SUPPORTED_HIDDEN_WIDTHS = ((32,),) + APPROVED_HIDDEN_WIDTHS
DEFAULT_PARAMETERS = {
    **{k: deepcopy(v) for k, v in BASE_PARAMETERS.items() if k != "hidden_width"},
    "hidden_widths": [64, 32],
}


def parameter_count(input_dim: int, hidden_widths: list[int]) -> int:
    """Count all trainable weights and biases, including both parallel heads."""
    if not _integer(input_dim) or input_dim < 1:
        raise JointTCPFitError("input_dim must be a positive integer")
    widths = resolve_parameters({"hidden_widths": hidden_widths})["hidden_widths"]
    return sum((before + 1) * after for before, after in
               zip([input_dim] + widths[:-1], widths)) + (widths[-1] + 1) * 6


def resolve_parameters(parameters: dict | None = None) -> dict:
    """Resolve the fixed recipe; small epoch/batch overrides support unit tests.

    The production orchestrator requires the fixed defaults except for one of
    the four approved hidden-layer configurations. The [32] compatibility arm
    and epoch/batch overrides are intended for tests only.
    """
    supplied = {} if parameters is None else parameters
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_PARAMETERS):
        raise JointTCPFitError("unknown joint TCP fitting parameter")
    resolved = {**deepcopy(DEFAULT_PARAMETERS), **deepcopy(supplied)}
    for key in ("epochs", "batch_size"):
        if not _integer(resolved[key]) or resolved[key] < 1:
            raise JointTCPFitError(f"{key} must be a positive integer")
    widths = resolved["hidden_widths"]
    if (not isinstance(widths, list) or not all(_integer(v) for v in widths)
            or tuple(widths) not in SUPPORTED_HIDDEN_WIDTHS):
        raise JointTCPFitError("hidden_widths must name an approved architecture")
    for key in set(DEFAULT_PARAMETERS) - {"epochs", "batch_size", "hidden_widths"}:
        expected, actual = DEFAULT_PARAMETERS[key], resolved[key]
        if type(actual) is not type(expected) or actual != expected:
            raise JointTCPFitError(f"{key} changes the fixed joint TCP recipe")
    return resolved


class _JointNetwork(nn.Module):
    def __init__(self, input_dim: int, parameters: dict):
        super().__init__()
        widths = parameters["hidden_widths"]
        self.trunk = nn.ModuleList([
            nn.Linear(before, after, dtype=torch.float32, device="cpu")
            for before, after in zip([input_dim] + widths[:-1], widths)
        ])
        self.dropout = nn.Dropout(parameters["dropout"])
        self.state_head = nn.Linear(widths[-1], 4, dtype=torch.float32, device="cpu")
        self.tcp_head = nn.Linear(widths[-1], 2, dtype=torch.float32, device="cpu")

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = values
        for layer in self.trunk:
            hidden = self.dropout(F.relu(layer(hidden)))
        return self.state_head(hidden), self.tcp_head(hidden)


def _layers(model):
    return {
        **{f"trunk_{index}": layer for index, layer in enumerate(model.trunk)},
        "state_head": model.state_head,
        "tcp_head": model.tcp_head,
    }


def _export(model, *, parameters, seed, config_id, auxiliary_weight, rows, input_dim):
    state = {
        "schema_version": SCHEMA_VERSION,
        "config_id": config_id,
        "input_dim": input_dim,
        "training_seed": seed,
        "fit_row_count": rows,
        "fit_role": "development_train_oof",
        "auxiliary_weight": float(auxiliary_weight),
        "tcp_supervised": auxiliary_weight > 0,
        "target_order": list(TARGET_ORDER),
        "tcp_order": list(TCP_ORDER),
        "torch_version": str(torch.__version__),
        "parameters": deepcopy(parameters),
        "layers": {
            name: {
                "weight": layer.weight.detach().numpy().tolist(),
                "bias": layer.bias.detach().numpy().tolist(),
            }
            for name, layer in _layers(model).items()
        },
    }
    state["state_sha256"] = _hash(state)
    return state


def validate_state(state: dict) -> None:
    """Verify identity, complete recipe, layer dimensions and the content digest."""
    keys = {
        "schema_version", "config_id", "input_dim", "training_seed", "fit_row_count",
        "fit_role", "auxiliary_weight", "tcp_supervised", "target_order", "tcp_order",
        "torch_version", "parameters", "layers", "state_sha256",
    }
    if not isinstance(state, dict) or set(state) != keys:
        raise JointTCPFitError("joint TCP state has missing or unknown fields")
    if (
        state["schema_version"] != SCHEMA_VERSION
        or not isinstance(state["config_id"], str)
        or not state["config_id"].strip()
        or any(not _integer(state[k]) or state[k] < 1 for k in ("input_dim", "fit_row_count"))
        or not _integer(state["training_seed"])
        or not 0 <= state["training_seed"] < 2**63
        or state["fit_role"] != "development_train_oof"
        or state["target_order"] != TARGET_ORDER
        or state["tcp_order"] != TCP_ORDER
        or state["torch_version"] != TORCH_VERSION
    ):
        raise JointTCPFitError("joint TCP state identity is invalid")
    weight = state["auxiliary_weight"]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not np.isfinite(weight)
        or weight < 0
        or type(state["tcp_supervised"]) is not bool
        or state["tcp_supervised"] != (weight > 0)
    ):
        raise JointTCPFitError("joint TCP auxiliary supervision contract is invalid")
    parameters = resolve_parameters(state["parameters"])
    if parameters != state["parameters"]:
        raise JointTCPFitError("joint TCP state needs the complete training recipe")
    try:
        expected_hash = _hash({k: v for k, v in state.items() if k != "state_sha256"})
    except (TypeError, ValueError) as error:
        raise JointTCPFitError("joint TCP state is not finite JSON data") from error
    if state["state_sha256"] != expected_hash:
        raise JointTCPFitError("joint TCP state digest differs")
    layers = state["layers"]
    widths = parameters["hidden_widths"]
    dimensions = [(f"trunk_{index}", before, after) for index, (before, after) in
                  enumerate(zip([state["input_dim"]] + widths[:-1], widths))]
    dimensions.extend([("state_head", widths[-1], 4), ("tcp_head", widths[-1], 2)])
    if not isinstance(layers, dict) or set(layers) != {v[0] for v in dimensions}:
        raise JointTCPFitError("joint TCP layers are invalid")
    for name, before, after in dimensions:
        layer = layers[name]
        if not isinstance(layer, dict) or set(layer) != {"weight", "bias"}:
            raise JointTCPFitError("each joint TCP layer needs a weight and bias")
        try:
            w, b = np.asarray(layer["weight"]), np.asarray(layer["bias"])
            valid = (
                w.shape == (after, before) and b.shape == (after,)
                and w.dtype.kind in "fi" and b.dtype.kind in "fi"
                and np.isfinite(w).all() and np.isfinite(b).all()
                and np.all(np.abs(w) <= np.finfo(np.float32).max)
                and np.all(np.abs(b) <= np.finfo(np.float32).max)
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise JointTCPFitError("joint TCP layer shape or numerical values are invalid")


def _native_from_layers(layers, values):
    def linear(name, x):
        w = torch.tensor(layers[name]["weight"], dtype=torch.float32, device="cpu")
        b = torch.tensor(layers[name]["bias"], dtype=torch.float32, device="cpu")
        return F.linear(x, w, b)

    with torch.no_grad():
        hidden = torch.from_numpy(values)
        for index in range(len(layers) - 2):
            hidden = F.relu(linear(f"trunk_{index}", hidden))
        state_logits = linear("state_head", hidden)
        tcp_logits = linear("tcp_head", hidden)
        return {
            "state_logits": state_logits.numpy().copy(),
            "tcp_logits": tcp_logits.numpy().copy(),
            "q": torch.softmax(state_logits, dim=1).numpy().copy(),
            "tcp_hat": torch.sigmoid(tcp_logits).numpy().copy(),
        }


def _float64_from_layers(layers, values):
    def linear(name, x):
        return x @ np.asarray(layers[name]["weight"], dtype=np.float64).T + np.asarray(
            layers[name]["bias"], dtype=np.float64
        )

    with threadpool_limits(limits=1), np.errstate(over="raise", invalid="raise"):
        try:
            hidden = values.astype(np.float64)
            for index in range(len(layers) - 2):
                hidden = np.maximum(linear(f"trunk_{index}", hidden), 0)
            logits = linear("state_head", hidden)
            tcp_logits = linear("tcp_head", hidden)
            q = np.exp(logits - logits.max(axis=1, keepdims=True))
            q /= q.sum(axis=1, keepdims=True)
            tcp_hat = np.empty_like(tcp_logits)
            positive = tcp_logits >= 0
            tcp_hat[positive] = 1.0 / (1.0 + np.exp(-tcp_logits[positive]))
            exp_values = np.exp(tcp_logits[~positive])
            tcp_hat[~positive] = exp_values / (1.0 + exp_values)
        except FloatingPointError as error:
            raise JointTCPFitError("nonfinite float64 prediction arithmetic") from error
    return {"state_logits": logits, "tcp_logits": tcp_logits, "q": q, "tcp_hat": tcp_hat}


def predict_joint(state: dict, X: np.ndarray, native: bool = True) -> dict:
    """Replay JSON weights; native float32 is authoritative, float64 diagnostic."""
    validate_state(state)
    values = _matrix(X, "inference inputs", state["input_dim"])
    if type(native) is not bool:
        raise JointTCPFitError("native must be a boolean")
    if native:
        if str(torch.__version__) != TORCH_VERSION:
            raise JointTCPFitError(f"native joint replay requires Torch {TORCH_VERSION}")
        with _execution():
            prediction = _native_from_layers(state["layers"], values)
    else:
        prediction = _float64_from_layers(state["layers"], values)
    if not all(np.isfinite(v).all() for v in prediction.values()):
        raise JointTCPFitError("joint TCP prediction contains nonfinite values")
    return prediction


def fit_joint(
    train_x: np.ndarray,
    state_targets: np.ndarray,
    tcp_targets: np.ndarray,
    predict_x_by_split: dict[str, np.ndarray],
    *,
    seed: int,
    config_id: str,
    auxiliary_weight: float,
    parameters: dict | None = None,
) -> dict:
    """Fit the final epoch without receiving validation labels or TCP targets."""
    parameters = resolve_parameters(parameters)
    if str(torch.__version__) != TORCH_VERSION:
        raise JointTCPFitError(f"joint TCP fitting requires Torch {TORCH_VERSION}")
    if not _integer(seed) or not 0 <= seed < 2**63:
        raise JointTCPFitError("seed must be an integer in [0, 2**63)")
    if not isinstance(config_id, str) or not config_id.strip():
        raise JointTCPFitError("config_id must be a nonempty string")
    if (isinstance(auxiliary_weight, bool) or not isinstance(auxiliary_weight, (int, float))
            or not np.isfinite(auxiliary_weight) or auxiliary_weight < 0):
        raise JointTCPFitError("auxiliary_weight must be finite and nonnegative")
    train = _matrix(train_x, "training inputs")
    if (
        not isinstance(state_targets, np.ndarray)
        or state_targets.shape != (len(train),)
        or state_targets.dtype.kind not in "iu"
        or np.any((state_targets < 0) | (state_targets > 3))
    ):
        raise JointTCPFitError("training states must be integer values in [0, 3]")
    targets = _matrix(tcp_targets, "training TCP", 2)
    if len(targets) != len(train) or np.any((tcp_targets < 0) | (tcp_targets > 1)):
        raise JointTCPFitError("training TCP must have shape (N,2) and lie in [0,1]")
    if not isinstance(predict_x_by_split, dict) or any(
        not isinstance(key, str) or not key.strip() or key == "train" for key in predict_x_by_split
    ):
        raise JointTCPFitError("prediction splits must be named; train is provided automatically")
    matrices = {"train": train, **{
        name: _matrix(values, f"{name} inputs", train.shape[1])
        for name, values in predict_x_by_split.items()
    }}
    training = torch.from_numpy(train)
    labels = torch.from_numpy(state_targets.astype(np.int64, copy=True))
    targets = torch.from_numpy(targets)
    history, predictions, parity = [], {}, {}
    started = time.perf_counter()
    state = None
    with _execution():
        torch.manual_seed(seed)
        model = _JointNetwork(train.shape[1], parameters)
        optimizer = _optimizer(model, parameters)
        shuffle = torch.Generator(device="cpu").manual_seed(seed)
        export_kwargs = dict(
            parameters=parameters, seed=seed, config_id=config_id,
            auxiliary_weight=auxiliary_weight, rows=len(train), input_dim=train.shape[1],
        )
        initial_state = _export(model, **export_kwargs)
        initial_dropout_rng_sha256 = hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest()
        # Independent generator documents the order without advancing training RNGs.
        order_generator = torch.Generator(device="cpu").manual_seed(seed)
        order_hash = hashlib.sha256()
        for _ in range(parameters["epochs"]):
            order_hash.update(torch.randperm(len(train), generator=order_generator).numpy().tobytes())
        last_good = deepcopy(model.state_dict())
        try:
            for epoch in range(1, parameters["epochs"] + 1):
                diagnostics = _train_epoch(model, optimizer, training, labels, targets,
                                          shuffle=shuffle, parameters=parameters,
                                          auxiliary_weight=auxiliary_weight, epoch=epoch)
                diagnostics.update(_eval_train(model, training, labels, targets, auxiliary_weight))
                history.append(diagnostics)
                last_good = deepcopy(model.state_dict())
            state = _export(model, **export_kwargs)
            validate_state(state)
            model.eval()
            with torch.no_grad():
                for split, matrix in matrices.items():
                    logits, tcp_logits = model(torch.from_numpy(matrix))
                    native = {
                        "state_logits": logits.numpy().copy(),
                        "tcp_logits": tcp_logits.numpy().copy(),
                        "q": logits.softmax(dim=1).numpy().copy(),
                        "tcp_hat": tcp_logits.sigmoid().numpy().copy(),
                    }
                    replay = _native_from_layers(state["layers"], matrix)
                    if not all(np.array_equal(native[k], replay[k]) for k in PREDICTION_KEYS):
                        raise JointTCPFitError(f"native float32 replay differs on {split}")
                    diagnostic = _float64_from_layers(state["layers"], matrix)
                    predictions[split] = native
                    parity[split] = _parity_summary(native, diagnostic)
        except (JointTCPFitError, RuntimeError, FloatingPointError) as error:
            model.load_state_dict(last_good)
            partial = {"state": _export(model, **export_kwargs), "history": deepcopy(history),
                       "completed_epochs": len(history), "failed_epoch": len(history) + 1,
                       "status": "failed", "error": str(error)}
            raise JointTCPFitError(str(error), partial_result=partial) from error
    return {
        "state": state,
        "history": history,
        "predictions": predictions,
        "metadata": {
            "config_id": config_id, "training_seed": seed,
            "auxiliary_weight": float(auxiliary_weight),
            "tcp_supervised": auxiliary_weight > 0, "checkpoint": "final_epoch",
            "final_epoch": parameters["epochs"],
            "fit_row_count": len(train), "input_dim": train.shape[1],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "hidden_widths": list(parameters["hidden_widths"]),
            "dropout_after_each_hidden_layer": True,
            "training_dtype": "float32", "device": "cpu", "num_threads": 1,
            "deterministic_algorithms": True, "torch_version": str(torch.__version__),
            "validation_labels_accessed": False, "validation_features_used_for_fitting": False,
            "expert_parameters_updated": False, "initialization_seed": seed,
            "shuffle_seed": seed, "dropout_seed": seed,
            "shuffle_order_sha256": order_hash.hexdigest(),
            "initial_dropout_rng_sha256": initial_dropout_rng_sha256,
            "initial_layers_sha256": _hash(initial_state["layers"]),
            "initial_tcp_head_sha256": _hash(initial_state["layers"]["tcp_head"]),
            "final_tcp_head_sha256": _hash(state["layers"]["tcp_head"]),
            "gradient_probe": "first minibatch per epoch, before combined backward",
            "saturation_bounds": [1e-6, 1 - 1e-6], "prediction_parity": parity,
            "runtime_seconds": time.perf_counter() - started,
        },
    }


# Descriptive alias for callers that do not use the parent study naming.
fit_joint_router = fit_joint

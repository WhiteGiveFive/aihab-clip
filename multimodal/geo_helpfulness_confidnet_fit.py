"""Fixed-budget confidence-head fitting on frozen expert representations.

Only development-training TCP targets enter fitting. Validation representations
are used after the final epoch for inference and portable prediction checks.
The exported state contains plain JSON data and replays with NumPy only.
"""

from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn


SCHEMA_VERSION = "geo_helpfulness.confidnet_state.v1"
PARITY_ATOL = 1e-6
DEFAULT_CONFIG = {
    "hidden_widths": [128, 128, 128, 128],
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "optimizer": "adam",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "device": "cpu",
    "dtype": "float32",
    "activation": "relu",
    "output_activation": "sigmoid",
    "loss": "mse",
    "dropout": 0.0,
    "normalization": "none",
    "shuffle": True,
    "early_stopping": False,
    "checkpoint": "final_epoch",
    "selection": "none",
    "num_threads": 1,
}


class ConfidNetFitError(ValueError):
    """The inputs, recipe, numerics, or portable state violate the contract."""


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_config(config: dict) -> dict:
    """Resolve the explicit recipe; reject silent changes to the fitting policy."""
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ConfidNetFitError("confidence configuration has unknown keys")
    resolved = {**deepcopy(DEFAULT_CONFIG), **deepcopy(config)}
    widths = resolved["hidden_widths"]
    if (
        not isinstance(widths, list)
        or not widths
        or any(not _integer(width) or width < 1 for width in widths)
    ):
        raise ConfidNetFitError("hidden_widths must be positive integer widths")
    for name in ("epochs", "batch_size"):
        if not _integer(resolved[name]) or resolved[name] < 1:
            raise ConfidNetFitError(f"{name} must be a positive integer")
    rate = resolved["learning_rate"]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not np.isfinite(rate)
        or rate <= 0
    ):
        raise ConfidNetFitError("learning_rate must be finite and positive")
    variable = {"hidden_widths", "epochs", "batch_size", "learning_rate"}
    for name in set(DEFAULT_CONFIG) - variable:
        if resolved[name] != DEFAULT_CONFIG[name]:
            raise ConfidNetFitError(f"{name} changes the fixed confidence policy")
    resolved["learning_rate"] = float(rate)
    return resolved


def _hidden(value: np.ndarray, name: str, input_dim: int | None = None) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or value.ndim != 2
        or min(value.shape) < 1
        or not np.isfinite(value).all()
        or (input_dim is not None and value.shape[1] != input_dim)
    ):
        raise ConfidNetFitError(
            f"{name} must be a nonempty finite float32/float64 representation matrix"
        )
    if np.max(np.abs(value)) > np.finfo(np.float32).max:
        raise ConfidNetFitError(f"{name} cannot be represented as float32")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _network(input_dim: int, hidden_widths: list[int]) -> nn.Sequential:
    layers = []
    previous = input_dim
    for width in hidden_widths:
        layers.extend(
            [nn.Linear(previous, width, device="cpu", dtype=torch.float32), nn.ReLU()]
        )
        previous = width
    layers.extend(
        [nn.Linear(previous, 1, device="cpu", dtype=torch.float32), nn.Sigmoid()]
    )
    return nn.Sequential(*layers)


def _state(model: nn.Sequential, config: dict, input_dim: int, seed: int, rows: int):
    return {
        "schema_version": SCHEMA_VERSION,
        "input_dim": input_dim,
        "parameters": deepcopy(config),
        "training_seed": seed,
        "fit_row_count": rows,
        "torch_version": str(torch.__version__),
        "layers": [
            {
                "weight": layer.weight.detach().numpy().tolist(),
                "bias": layer.bias.detach().numpy().tolist(),
            }
            for layer in model
            if isinstance(layer, nn.Linear)
        ],
    }


def _layers(state: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SCHEMA_VERSION
        or not _integer(state.get("input_dim"))
        or state["input_dim"] < 1
    ):
        raise ConfidNetFitError("invalid confidence state identity or input dimension")
    parameters = resolve_config(state.get("parameters"))
    if parameters != state["parameters"]:
        raise ConfidNetFitError("confidence state must contain the complete recipe")
    dimensions = [state["input_dim"], *parameters["hidden_widths"], 1]
    layers = state.get("layers")
    if not isinstance(layers, list) or len(layers) != len(dimensions) - 1:
        raise ConfidNetFitError("confidence state has an incorrect number of layers")
    result = []
    for layer, before, after in zip(layers, dimensions, dimensions[1:]):
        if not isinstance(layer, dict) or set(layer) != {"weight", "bias"}:
            raise ConfidNetFitError("confidence layer must contain weight and bias")
        try:
            weight = np.asarray(layer["weight"], dtype=np.float64)
            bias = np.asarray(layer["bias"], dtype=np.float64)
        except (ValueError, TypeError) as error:
            raise ConfidNetFitError("confidence layer contains invalid arrays") from error
        if (
            weight.shape != (after, before)
            or bias.shape != (after,)
            or not np.isfinite(weight).all()
            or not np.isfinite(bias).all()
        ):
            raise ConfidNetFitError("confidence layer shape or numerical values are invalid")
        result.append((weight, bias))
    return result


def predict_confidnet(state: dict, hidden: np.ndarray) -> np.ndarray:
    """Independently replay JSON weights in NumPy and return float64 TCP estimates.

    Inputs are rounded to the float32 representation used for native inference.
    The portable matrix arithmetic uses float64 to limit accumulated roundoff;
    fitting verifies its absolute disagreement with native inference is <=1e-6.
    """
    layers = _layers(state)
    values = _hidden(hidden, "inference hidden", state["input_dim"]).astype(np.float64)
    with threadpool_limits(limits=1), np.errstate(over="raise", invalid="raise"):
        try:
            for index, (weight, bias) in enumerate(layers):
                values = values @ weight.T + bias
                if index < len(layers) - 1:
                    values = np.maximum(values, 0.0)
            logits = values[:, 0]
            # Stable sigmoid avoids overflow for large negative or positive logits.
            estimate = np.empty_like(logits)
            positive = logits >= 0
            estimate[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
            exp_value = np.exp(logits[~positive])
            estimate[~positive] = exp_value / (1.0 + exp_value)
        except FloatingPointError as error:
            raise ConfidNetFitError("nonfinite confidence replay arithmetic") from error
    if not np.isfinite(estimate).all():
        raise ConfidNetFitError("confidence replay produced nonfinite predictions")
    return estimate


def fit_confidnet(
    train_hidden: np.ndarray,
    train_tcp: np.ndarray,
    validation_hidden: np.ndarray,
    *,
    seed: int,
    config: dict,
) -> dict:
    """Fit a final-epoch head without validation targets or checkpoint selection."""
    parameters = resolve_config(config)
    if not _integer(seed) or not 0 <= seed < 2**63:
        raise ConfidNetFitError("seed must be an integer in [0, 2**63)")
    train_array = _hidden(train_hidden, "training hidden")
    validation_array = _hidden(
        validation_hidden, "validation hidden", train_array.shape[1]
    )
    if (
        not isinstance(train_tcp, np.ndarray)
        or train_tcp.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or train_tcp.shape != (len(train_array),)
        or not np.isfinite(train_tcp).all()
        or np.any((train_tcp < 0) | (train_tcp > 1))
    ):
        raise ConfidNetFitError("training TCP must be a finite float vector in [0, 1]")
    training = torch.from_numpy(train_array)
    targets = torch.from_numpy(np.array(train_tcp, dtype=np.float32, copy=True))
    validation = torch.from_numpy(validation_array)
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    before_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    history, parity = [], {}
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1), torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = _network(training.shape[1], parameters["hidden_widths"])
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=parameters["learning_rate"],
                betas=tuple(parameters["betas"]),
                eps=parameters["eps"],
                weight_decay=0.0,
                amsgrad=False,
                foreach=False,
                fused=False,
            )
            shuffle = torch.Generator(device="cpu").manual_seed(seed)
            for epoch in range(1, parameters["epochs"] + 1):
                total_mse, maximum_gradient = 0.0, 0.0
                model.train()
                for rows in torch.randperm(len(training), generator=shuffle).split(
                    parameters["batch_size"]
                ):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(training[rows])[:, 0]
                    loss = (prediction - targets[rows]).square().mean()
                    if not bool(torch.isfinite(loss)):
                        raise ConfidNetFitError(f"nonfinite training MSE at epoch {epoch}")
                    loss.backward()
                    gradients = [parameter.grad for parameter in model.parameters()]
                    if any(
                        grad is None or not bool(torch.isfinite(grad).all())
                        for grad in gradients
                    ):
                        raise ConfidNetFitError(f"nonfinite gradient at epoch {epoch}")
                    norm = torch.sqrt(sum(grad.square().sum() for grad in gradients))
                    if not bool(torch.isfinite(norm)):
                        raise ConfidNetFitError(f"nonfinite gradient norm at epoch {epoch}")
                    maximum_gradient = max(maximum_gradient, float(norm))
                    total_mse += float(loss.detach()) * len(rows)
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    prediction = model(training)[:, 0]
                    mse = (prediction - targets).square().mean()
                if not bool(torch.isfinite(mse)) or any(
                    not bool(torch.isfinite(value).all()) for value in model.parameters()
                ):
                    raise ConfidNetFitError(f"nonfinite training state at epoch {epoch}")
                history.append(
                    {
                        "epoch": epoch,
                        "optimization_mse": total_mse / len(training),
                        "train_mse": float(mse),
                        "maximum_gradient_l2_norm": maximum_gradient,
                    }
                )
            state = _state(model, parameters, training.shape[1], seed, len(training))
            predictions = {}
            for name, matrix in (("train", training), ("validation", validation)):
                with torch.no_grad():
                    native = model(matrix)[:, 0].numpy().astype(np.float64)
                portable = predict_confidnet(state, matrix.numpy())
                error = float(np.max(np.abs(native - portable)))
                if not np.isfinite(native).all() or error > PARITY_ATOL:
                    raise ConfidNetFitError(f"{name} native/portable parity exceeds 1e-6")
                predictions[name] = portable
                parity[name] = {"maximum_abs_error": error, "atol": PARITY_ATOL}
    finally:
        torch.set_num_threads(before_threads)
        torch.use_deterministic_algorithms(
            before_deterministic, warn_only=before_warn_only
        )
    return {
        "state": state,
        "history": history,
        "train_prediction": predictions["train"],
        "validation_prediction": predictions["validation"],
        "metadata": {
            "runtime_seconds": time.perf_counter() - started,
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "train_rows": len(training),
            "validation_rows": len(validation),
            "input_dim": training.shape[1],
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "final_epoch": parameters["epochs"],
            "checkpoint": "final_epoch",
            "validation_labels_accessed": False,
            "validation_features_used_for_fitting": False,
            "expert_parameters_updated": False,
            "initialization_seed": seed,
            "shuffle_seed": seed,
            "device": "cpu",
            "training_dtype": "float32",
            "portable_arithmetic_dtype": "float64",
            "num_threads": 1,
            "deterministic_algorithms": True,
            "prediction_parity": parity,
        },
    }

"""Deterministic joint four-state/TCP training with label-free inference inputs.

Only the training targets enter optimization.  Authoritative predictions replay
the exported JSON weights using the same native CPU float32 operations; float64
NumPy replay is a separately reported numerical diagnostic.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "geo_helpfulness.joint_tcp_router_state.v1"
TORCH_VERSION = "2.5.1+cu124"
TARGET_ORDER = ["rescue", "harm", "both_correct", "both_wrong"]
TCP_ORDER = ["raw_concat", "geo_only"]
DEFAULT_PARAMETERS = {
    "hidden_width": 32,
    "epochs": 300,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "bias_weight_decay": 0.0,
    "optimizer": "adamw",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "dropout": 0.1,
    "device": "cpu",
    "dtype": "float32",
    "num_threads": 1,
    "activation": "relu",
    "state_activation": "softmax",
    "tcp_activation": "sigmoid",
    "checkpoint": "final_epoch",
    "shuffle": True,
    "early_stopping": False,
    "scheduler": None,
    "class_weighting": False,
}
PREDICTION_KEYS = ("state_logits", "tcp_logits", "q", "tcp_hat")


class JointTCPFitError(ValueError):
    """Invalid input or failed numerics, optionally carrying the last good state."""

    def __init__(self, message: str, *, partial_result: dict | None = None):
        super().__init__(message)
        self.partial_result = partial_result


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_parameters(parameters: dict | None = None) -> dict:
    """Resolve the fixed recipe; small epoch/batch overrides support unit tests.

    The production orchestrator requires exact DEFAULT_PARAMETERS equality.
    """
    supplied = {} if parameters is None else parameters
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_PARAMETERS):
        raise JointTCPFitError("unknown joint TCP fitting parameter")
    resolved = {**deepcopy(DEFAULT_PARAMETERS), **deepcopy(supplied)}
    for key in ("epochs", "batch_size"):
        if not _integer(resolved[key]) or resolved[key] < 1:
            raise JointTCPFitError(f"{key} must be a positive integer")
    for key in set(DEFAULT_PARAMETERS) - {"epochs", "batch_size"}:
        expected, actual = DEFAULT_PARAMETERS[key], resolved[key]
        if type(actual) is not type(expected) or actual != expected:
            raise JointTCPFitError(f"{key} changes the fixed joint TCP recipe")
    return resolved


def _matrix(value: np.ndarray, name: str, width: int | None = None) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or value.ndim != 2
        or min(value.shape) < 1
        or (width is not None and value.shape[1] != width)
        or not np.isfinite(value).all()
        or np.any(np.abs(value) > np.finfo(np.float32).max)
    ):
        raise JointTCPFitError(f"{name} must be a nonempty finite float matrix")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@contextmanager
def _execution():
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    before_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1), torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(before_threads)
        torch.use_deterministic_algorithms(before_deterministic, warn_only=before_warn)


class _JointNetwork(nn.Module):
    def __init__(self, input_dim: int, parameters: dict):
        super().__init__()
        width = parameters["hidden_width"]
        self.trunk = nn.Linear(input_dim, width, dtype=torch.float32, device="cpu")
        self.dropout = nn.Dropout(parameters["dropout"])
        self.state_head = nn.Linear(width, 4, dtype=torch.float32, device="cpu")
        self.tcp_head = nn.Linear(width, 2, dtype=torch.float32, device="cpu")

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.dropout(F.relu(self.trunk(values)))
        return self.state_head(hidden), self.tcp_head(hidden)


def _optimizer(model: _JointNetwork, parameters: dict):
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if n.endswith("weight")],
            "weight_decay": parameters["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith("bias")],
            "weight_decay": parameters["bias_weight_decay"],
        },
    ]
    return torch.optim.AdamW(
        groups,
        lr=parameters["learning_rate"],
        betas=tuple(parameters["betas"]),
        eps=parameters["eps"],
        amsgrad=False,
        foreach=False,
        fused=False,
    )


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
                "weight": getattr(model, name).weight.detach().numpy().tolist(),
                "bias": getattr(model, name).bias.detach().numpy().tolist(),
            }
            for name in ("trunk", "state_head", "tcp_head")
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
    if not isinstance(layers, dict) or set(layers) != {"trunk", "state_head", "tcp_head"}:
        raise JointTCPFitError("joint TCP layers are invalid")
    width = parameters["hidden_width"]
    for name, before, after in (
        ("trunk", state["input_dim"], width), ("state_head", width, 4), ("tcp_head", width, 2)
    ):
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
        hidden = F.relu(linear("trunk", torch.from_numpy(values)))
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
            hidden = np.maximum(linear("trunk", values.astype(np.float64)), 0)
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


def _norm(gradients) -> float:
    squared_norm = sum(
        g.detach().double().square().sum().item()
        for g in gradients if g is not None
    )
    return float(squared_norm ** 0.5)


def _components(state_logits, tcp_logits, labels, tcp_targets, auxiliary_weight):
    ce = F.cross_entropy(state_logits, labels)
    per_expert_mse = (torch.sigmoid(tcp_logits) - tcp_targets).square().mean(dim=0)
    mse = per_expert_mse.mean()
    # Omitting the graph also keeps AdamW from decaying an unsupervised TCP head.
    total = ce if auxiliary_weight == 0 else ce + auxiliary_weight * mse
    return ce, per_expert_mse, mse, total


def _gradient_probe(ce, mse, model):
    shared = tuple(model.trunk.parameters())
    ce_grads = torch.autograd.grad(ce, shared, retain_graph=True)
    tcp_grads = torch.autograd.grad(mse, shared, retain_graph=True)
    ce_norm, tcp_norm = _norm(ce_grads), _norm(tcp_grads)
    dot = sum(
        (a.detach().double() * b.detach().double()).sum().item()
        for a, b in zip(ce_grads, tcp_grads)
    )
    cosine = dot / (ce_norm * tcp_norm) if ce_norm * tcp_norm else 0.0
    return {
        "first_batch_shared_ce_gradient_norm": ce_norm,
        "first_batch_shared_tcp_gradient_norm": tcp_norm,
        "first_batch_shared_gradient_cosine": cosine,
    }


def _train_epoch(
    model, optimizer, training, labels, targets, *, shuffle, parameters,
    auxiliary_weight, epoch,
):
    model.train()
    sums = np.zeros(4, dtype=np.float64)
    maxima = {"maximum_gradient_norm": 0.0, "maximum_shared_gradient_norm": 0.0,
              "maximum_state_head_gradient_norm": 0.0, "maximum_tcp_head_gradient_norm": 0.0}
    probe = {}
    batches = torch.randperm(len(training), generator=shuffle).split(parameters["batch_size"])
    for index, rows in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        state_logits, tcp_logits = model(training[rows])
        ce, per_expert, mse, loss = _components(
            state_logits, tcp_logits, labels[rows], targets[rows], auxiliary_weight
        )
        if not bool(torch.isfinite(torch.stack((ce, mse, loss))).all()):
            raise JointTCPFitError(f"nonfinite loss at epoch {epoch}, batch {index + 1}")
        if index == 0:
            probe = _gradient_probe(ce, mse, model)
            probe["first_batch_shared_weighted_tcp_gradient_norm"] = (
                auxiliary_weight * probe["first_batch_shared_tcp_gradient_norm"]
            )
            if not all(np.isfinite(v) for v in probe.values()):
                raise JointTCPFitError(f"nonfinite component gradient at epoch {epoch}")
        loss.backward()
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise JointTCPFitError(f"nonfinite gradient at epoch {epoch}, batch {index + 1}")
        gradients = {
            "maximum_gradient_norm": _norm(p.grad for p in model.parameters()),
            "maximum_shared_gradient_norm": _norm(p.grad for p in model.trunk.parameters()),
            "maximum_state_head_gradient_norm": _norm(
                p.grad for p in model.state_head.parameters()
            ),
            "maximum_tcp_head_gradient_norm": _norm(p.grad for p in model.tcp_head.parameters()),
        }
        maxima = {k: max(maxima[k], gradients[k]) for k in maxima}
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise JointTCPFitError(f"nonfinite parameter at epoch {epoch}, batch {index + 1}")
        sums += len(rows) * np.asarray(
            [ce.item(), per_expert[0].item(), per_expert[1].item(), loss.item()]
        )
    ce, fusion_mse, geo_mse, loss = sums / len(training)
    return {
        "epoch": epoch, "batch_train_ce": float(ce),
        "batch_train_fusion_mse": float(fusion_mse), "batch_train_geo_mse": float(geo_mse),
        "batch_train_loss": float(loss),
        "batch_train_weighted_auxiliary_loss": float(
            auxiliary_weight * (fusion_mse + geo_mse) / 2
        ),
        **maxima, **probe,
    }


def _eval_train(model, training, labels, targets, auxiliary_weight):
    model.eval()
    with torch.no_grad():
        state_logits, tcp_logits = model(training)
        ce, per_expert, mse, loss = _components(
            state_logits, tcp_logits, labels, targets, auxiliary_weight
        )
        q, tcp_hat = state_logits.softmax(dim=1), tcp_logits.sigmoid()
        values = {
            "train_ce": ce.item(), "train_mse": mse.item(),
            "train_fusion_mse": per_expert[0].item(), "train_geo_mse": per_expert[1].item(),
            "train_weighted_auxiliary_loss": auxiliary_weight * mse.item(),
            "train_loss": loss.item(),
            "train_state_accuracy": (q.argmax(dim=1) == labels).float().mean().item(),
        }
        for index, name in enumerate(("fusion", "geo")):
            predictions = tcp_hat[:, index]
            saturation = (predictions <= 1e-6) | (predictions >= 1 - 1e-6)
            values.update({
                f"train_{name}_tcp_min": predictions.min().item(),
                f"train_{name}_tcp_max": predictions.max().item(),
                f"train_{name}_tcp_mean": predictions.mean().item(),
                f"train_{name}_tcp_saturation_fraction": saturation.float().mean().item(),
            })
    if not all(np.isfinite(v) for v in values.values()):
        raise JointTCPFitError("nonfinite dropout-disabled training diagnostics")
    return values


def _parity_summary(native: dict, diagnostic: dict) -> dict:
    errors = {
        f"float64_{key}_max_abs_error": float(
            np.max(np.abs(native[key].astype(np.float64) - diagnostic[key]))
        )
        for key in PREDICTION_KEYS
    }
    native_order = native["tcp_hat"][:, 0] > native["tcp_hat"][:, 1]
    diagnostic_order = diagnostic["tcp_hat"][:, 0] > diagnostic["tcp_hat"][:, 1]
    return {
        "native_float32_exact": True,
        **errors,
        "float64_state_argmax_disagreements": int(np.count_nonzero(
            native["q"].argmax(1) != diagnostic["q"].argmax(1)
        )),
        "float64_tcp_order_disagreements": int(np.count_nonzero(
            native_order != diagnostic_order
        )),
    }


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
            "training_dtype": "float32", "device": "cpu", "num_threads": 1,
            "deterministic_algorithms": True, "torch_version": str(torch.__version__),
            "validation_labels_accessed": False, "validation_features_used_for_fitting": False,
            "expert_parameters_updated": False, "initialization_seed": seed,
            "shuffle_seed": seed, "dropout_seed": seed,
            "initial_layers_sha256": _hash(initial_state["layers"]),
            "initial_tcp_head_sha256": _hash(initial_state["layers"]["tcp_head"]),
            "final_tcp_head_sha256": _hash(state["layers"]["tcp_head"]),
            "gradient_probe": "first minibatch per epoch, before combined backward",
            "saturation_bounds": [1e-6, 1 - 1e-6], "prediction_parity": parity,
            "runtime_seconds": time.perf_counter() - started,
        },
    }

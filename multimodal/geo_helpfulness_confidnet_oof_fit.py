"""OOF-supervised confidence and matched numerical RF fitting primitives.

Only training labels enter these interfaces. Native float32 confidence estimates
are authoritative; portable JSON weights are independently replayed in float64.
No artifact discovery, validation scoring, selection, or automatic retuning occurs.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import time
import warnings

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from multimodal import geo_helpfulness_confidnet_fit as initial
from multimodal.geo_helpfulness_router_candidate_state import (
    TARGET_ORDER,
    full_feature_columns,
)
from multimodal.geo_helpfulness_router_capacity_fit import RF_SHARED, SKLEARN_VERSION
from multimodal.geo_helpfulness_router_capacity_state import (
    canonical_hash,
    _tree,
    _leaves,
)

CONFIDENCE_SCHEMA = "geo_helpfulness.confidnet_oof_state.v1"
RF_SCHEMA = "geo_helpfulness.confidnet_oof_rf_state.v1"
CONFIDENCE_DEFAULTS = {
    **deepcopy(initial.DEFAULT_CONFIG),
    "hidden_widths": [400, 400, 400, 400],
    "epochs": 500,
    "learning_rate": 0.0001,
    "zero_gradient_patience": 5,
    "prediction_convention": "native_float32",
}
RF_DEFAULTS = {**RF_SHARED, "max_depth": 4, "min_samples_leaf": 5}
TCP_FEATURE_NAMES = ["image_tcp_hat", "geo_tcp_hat", "raw_tcp_hat"]
PARITY_ATOL = 1e-6


class ConfidNetOOFFitError(ValueError):
    """Invalid inputs/numerics; partial_result can preserve last finite state."""

    def __init__(self, message, *, partial_result=None):
        super().__init__(message)
        self.partial_result = partial_result


def resolve_confidence_parameters(parameters=None):
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict) or set(parameters) - set(CONFIDENCE_DEFAULTS):
        raise ConfidNetOOFFitError("unknown confidence parameters")
    resolved = {**deepcopy(CONFIDENCE_DEFAULTS), **deepcopy(parameters)}
    for key in ("zero_gradient_patience", "prediction_convention"):
        if resolved[key] != CONFIDENCE_DEFAULTS[key]:
            raise ConfidNetOOFFitError(f"{key} changes the fixed confidence policy")
    try:
        base = initial.resolve_config(
            {k: v for k, v in resolved.items() if k in initial.DEFAULT_CONFIG}
        )
    except ValueError as exc:
        raise ConfidNetOOFFitError(str(exc)) from exc
    return {
        **base,
        **{k: resolved[k] for k in ("zero_gradient_patience", "prediction_convention")},
    }


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (1, 2, 3, 4):
        raise ConfidNetOOFFitError("training seed must be one of 1, 2, 3, 4")


def _hidden(value, name, width=None):
    try:
        return initial._hidden(value, name, width)
    except ValueError as exc:
        raise ConfidNetOOFFitError(str(exc)) from exc


def _seal(state):
    state = deepcopy(state)
    state.pop("state_sha256", None)
    state["state_sha256"] = canonical_hash(state)
    return state


def _verify_hash(state):
    if not isinstance(state, dict) or state.get("state_sha256") != canonical_hash(
        {k: v for k, v in state.items() if k != "state_sha256"}
    ):
        raise ConfidNetOOFFitError("portable state hash mismatch")


@contextmanager
def _execution(seed=None):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1), torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _confidence_state(model, parameters, seed, rows, epoch, status):
    state = initial._state(model, parameters, model[0].in_features, seed, rows)
    state.update(
        schema_version=CONFIDENCE_SCHEMA, completed_epochs=epoch, status=status
    )
    return _seal(state)


def _legacy_state(state):
    _verify_hash(state)
    if state.get("schema_version") != CONFIDENCE_SCHEMA:
        raise ConfidNetOOFFitError("invalid confidence state schema")
    parameters = resolve_confidence_parameters(state.get("parameters"))
    if parameters != state["parameters"]:
        raise ConfidNetOOFFitError("confidence state needs complete parameters")
    _seed(state.get("training_seed"))
    if not initial._integer(state.get("fit_row_count")) or state["fit_row_count"] < 1:
        raise ConfidNetOOFFitError("invalid confidence training row count")
    epoch = state.get("completed_epochs")
    if not initial._integer(epoch) or not 0 <= epoch <= parameters["epochs"]:
        raise ConfidNetOOFFitError("invalid completed confidence epoch")
    if state.get("status") not in (
        "complete",
        "paused_zero_gradient",
        "aborted_nonfinite",
    ):
        raise ConfidNetOOFFitError("invalid confidence completion status")
    if state["status"] == "complete" and epoch != parameters["epochs"]:
        raise ConfidNetOOFFitError("completed state must retain final epoch")
    legacy = {
        **state,
        "schema_version": initial.SCHEMA_VERSION,
        "parameters": {
            k: v for k, v in parameters.items() if k in initial.DEFAULT_CONFIG
        },
    }
    try:
        initial._layers(legacy)
    except ValueError as exc:
        raise ConfidNetOOFFitError(str(exc)) from exc
    return legacy


def predict_confidence(state, hidden, *, native=True):
    """Replay saved weights; native output is float32, portable output float64."""
    legacy = _legacy_state(state)
    matrix = _hidden(hidden, "confidence inference", state["input_dim"])
    if not native:
        return initial.predict_confidnet(legacy, matrix)
    with _execution():
        model = initial._network(
            state["input_dim"], state["parameters"]["hidden_widths"]
        )
        with torch.no_grad():
            for layer, saved in zip(
                (v for v in model if isinstance(v, nn.Linear)), state["layers"]
            ):
                layer.weight.copy_(torch.tensor(saved["weight"], dtype=torch.float32))
                layer.bias.copy_(torch.tensor(saved["bias"], dtype=torch.float32))
            result = model.eval()(torch.from_numpy(matrix))[:, 0].numpy().copy()
    if not np.isfinite(result).all():
        raise ConfidNetOOFFitError("nonfinite native confidence inference")
    return result


def _summary(array):
    value = np.asarray(array, dtype=np.float64)
    return {
        "count": int(value.size),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
        "quantiles": dict(
            zip(
                ["0", "0.01", "0.1", "0.5", "0.9", "0.99", "1"],
                np.quantile(value, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist(),
            )
        ),
    }


def _prediction_summary(prediction):
    return {
        **_summary(prediction),
        "fraction_near_zero": float(np.mean(prediction <= 1e-6)),
        "fraction_near_one": float(np.mean(prediction >= 1 - 1e-6)),
    }


def _tcp(logits, labels, rows):
    if (
        not isinstance(logits, np.ndarray)
        or logits.dtype.kind != "f"
        or logits.ndim != 2
        or logits.shape[0] != rows
        or logits.shape[1] < 2
        or not np.isfinite(logits).all()
    ):
        raise ConfidNetOOFFitError("training logits must be a finite floating matrix")
    if (
        not isinstance(labels, np.ndarray)
        or labels.dtype.kind not in "iu"
        or labels.shape != (rows,)
        or np.any(labels < 0)
        or np.any(labels >= logits.shape[1])
    ):
        raise ConfidNetOOFFitError(
            "training labels must index each row's expert logits"
        )
    values = logits.astype(np.float64)
    values = np.exp(values - values.max(axis=1, keepdims=True))
    tcp = (values / values.sum(axis=1, keepdims=True))[np.arange(rows), labels]
    return tcp, logits.argmax(axis=1) != labels


def fit_confidence(
    train_hidden,
    train_logits,
    train_labels,
    predict_hidden_by_split,
    *,
    seed,
    parameters=None,
):
    """Fit declared final-epoch TCP regressor; pause without validation prediction."""
    parameters = resolve_confidence_parameters(parameters)
    _seed(seed)
    training_array = _hidden(train_hidden, "training confidence features")
    if (
        not isinstance(predict_hidden_by_split, dict)
        or "train" in predict_hidden_by_split
        or any(not isinstance(k, str) or not k for k in predict_hidden_by_split)
    ):
        raise ConfidNetOOFFitError(
            "prediction splits must be named matrices excluding train"
        )
    arrays = {
        k: _hidden(v, k, training_array.shape[1])
        for k, v in predict_hidden_by_split.items()
    }
    tcp, failures = _tcp(train_logits, train_labels, len(training_array))
    training = torch.from_numpy(training_array)
    target = torch.tensor(tcp, dtype=torch.float32)
    history, status, zero_epochs = [], "complete", 0
    started = time.perf_counter()
    metadata = {
        "initialization_seed": seed,
        "shuffle_seed": seed,
        "train_rows": len(training_array),
        "input_dim": training_array.shape[1],
        "training_errors": int(failures.sum()),
        "tcp_summary": _summary(tcp),
        "tcp_correct_summary": _summary(tcp[~failures]) if (~failures).any() else None,
        "tcp_error_summary": _summary(tcp[failures]) if failures.any() else None,
        "input_summary": _summary(training_array),
        "input_row_l2_summary": _summary(
            np.linalg.norm(training_array.astype(np.float64), axis=1)
        ),
        "always_one_mse": float(np.mean((tcp - 1) ** 2)),
        "training_mean_mse": float(np.mean((tcp - tcp.mean()) ** 2)),
        "validation_labels_accessed": False,
        "validation_features_used_for_fitting": False,
        "expert_parameters_updated": False,
        "checkpoint": "final_epoch",
        "device": "cpu",
        "training_dtype": "float32",
        "num_threads": 1,
        "deterministic_algorithms": True,
        "prediction_convention": "native_float32",
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }
    with _execution(seed):
        model = initial._network(training.shape[1], parameters["hidden_widths"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=parameters["learning_rate"],
            betas=tuple(parameters["betas"]),
            eps=parameters["eps"],
            weight_decay=0,
            amsgrad=False,
            foreach=False,
            fused=False,
        )
        shuffle = torch.Generator(device="cpu").manual_seed(seed)
        for epoch in range(1, parameters["epochs"] + 1):
            last_finite = {k: v.detach().clone() for k, v in model.state_dict().items()}
            maximum_gradient, max_abs_gradient, total_mse = 0.0, 0.0, 0.0
            model.train()
            try:
                for rows in torch.randperm(len(training), generator=shuffle).split(
                    parameters["batch_size"]
                ):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(training[rows])[:, 0]
                    if not bool(torch.isfinite(prediction).all()):
                        raise ConfidNetOOFFitError(
                            f"nonfinite prediction at epoch {epoch}"
                        )
                    loss = (prediction - target[rows]).square().mean()
                    if not bool(torch.isfinite(loss)):
                        raise ConfidNetOOFFitError(f"nonfinite loss at epoch {epoch}")
                    loss.backward()
                    gradients = [v.grad for v in model.parameters()]
                    if any(
                        v is None or not bool(torch.isfinite(v).all())
                        for v in gradients
                    ):
                        raise ConfidNetOOFFitError(
                            f"nonfinite gradient at epoch {epoch}"
                        )
                    # Double accumulation avoids mistaking underflowed squared gradients for zero.
                    norm = float(
                        torch.sqrt(sum(v.double().square().sum() for v in gradients))
                    )
                    if not np.isfinite(norm):
                        raise ConfidNetOOFFitError(
                            f"nonfinite gradient norm at epoch {epoch}"
                        )
                    maximum_gradient = max(maximum_gradient, norm)
                    max_abs_gradient = max(
                        max_abs_gradient, max(float(v.abs().max()) for v in gradients)
                    )
                    total_mse += float(loss.detach()) * len(rows)
                    optimizer.step()
                    if any(
                        not bool(torch.isfinite(v).all()) for v in model.parameters()
                    ):
                        raise ConfidNetOOFFitError(
                            f"nonfinite parameter at epoch {epoch}"
                        )
                model.eval()
                with torch.no_grad():
                    prediction = model(training)[:, 0].numpy().copy()
                if not np.isfinite(prediction).all():
                    raise ConfidNetOOFFitError(
                        f"nonfinite epoch prediction at epoch {epoch}"
                    )
            except ConfidNetOOFFitError as exc:
                model.load_state_dict(last_finite)
                state = _confidence_state(
                    model,
                    parameters,
                    seed,
                    len(training),
                    epoch - 1,
                    "aborted_nonfinite",
                )
                exc.partial_result = {
                    "status": "aborted_nonfinite",
                    "state": state,
                    "history": history,
                    "predictions": {},
                    "metadata": {
                        **metadata,
                        "aborted_epoch": epoch,
                        "reason": str(exc),
                    },
                }
                raise
            zero_epochs = zero_epochs + 1 if max_abs_gradient == 0.0 else 0
            history.append(
                {
                    "epoch": epoch,
                    "optimization_mse": total_mse / len(training),
                    "train_mse": float(
                        np.mean((prediction.astype(np.float64) - tcp) ** 2)
                    ),
                    "maximum_gradient_l2_norm": maximum_gradient,
                    "maximum_absolute_gradient": max_abs_gradient,
                    "consecutive_zero_gradient_epochs": zero_epochs,
                    "prediction_summary": _prediction_summary(prediction),
                }
            )
            if zero_epochs >= parameters["zero_gradient_patience"]:
                status = "paused_zero_gradient"
                break
        state = _confidence_state(model, parameters, seed, len(training), epoch, status)
        predictions, parity = {}, {}
        inference = {
            "train": training_array,
            **(arrays if status == "complete" else {}),
        }
        for name, matrix in inference.items():
            with torch.no_grad():
                native = model(torch.from_numpy(matrix))[:, 0].numpy().copy()
            if not np.isfinite(native).all():
                raise ConfidNetOOFFitError(f"nonfinite final {name} prediction")
            portable = predict_confidence(state, matrix, native=False)
            error = float(np.max(np.abs(native.astype(np.float64) - portable)))
            if error > PARITY_ATOL:
                raise ConfidNetOOFFitError(
                    f"{name} confidence portable parity exceeds {PARITY_ATOL}"
                )
            predictions[name] = native
            parity[name] = {"maximum_abs_error": error, "atol": PARITY_ATOL}
    metadata.update(
        runtime_seconds=time.perf_counter() - started,
        final_epoch=epoch,
        parameter_count=sum(v.numel() for v in model.parameters()),
        prediction_parity=parity,
        status=status,
        predictions_summary={k: _prediction_summary(v) for k, v in predictions.items()},
    )
    if status != "complete":
        metadata["reason"] = (
            "five consecutive epochs with exactly zero maximum absolute gradient"
        )
    return {
        "status": status,
        "state": state,
        "history": history,
        "predictions": predictions,
        "metadata": metadata,
    }


def _router_matrix(value, width):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float64
        or value.ndim != 2
        or value.shape[1] != width
        or not len(value)
        or not np.isfinite(value).all()
        or np.max(np.abs(value)) > np.finfo(np.float32).max
    ):
        raise ConfidNetOOFFitError(
            "router inputs must be finite float64 matrices of the declared width"
        )
    return np.ascontiguousarray(value)


def _router_features(names):
    expected = full_feature_columns()[:25]
    if names not in (expected, expected + TCP_FEATURE_NAMES):
        raise ConfidNetOOFFitError(
            "router feature order must be numeric25 or numeric25 plus three TCP estimates"
        )


def _rf_parameters(parameters=None):
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict) or set(parameters) - set(RF_DEFAULTS):
        raise ConfidNetOOFFitError("unknown RF parameter")
    result = {**RF_DEFAULTS, **parameters}
    if result != RF_DEFAULTS:
        raise ConfidNetOOFFitError("RF parameters differ from approved matched recipe")
    return result


def predict_router(state, matrix):
    """Independently apply portable tree JSON in class order rescue/harm/correct/wrong."""
    _verify_hash(state)
    if state.get("schema_version") != RF_SCHEMA:
        raise ConfidNetOOFFitError("invalid RF state schema")
    _seed(state.get("training_seed"))
    _router_features(state.get("feature_names"))
    _rf_parameters(state.get("parameters"))
    if state.get("classes") != [0, 1, 2, 3] or state.get("target_order") != list(
        TARGET_ORDER
    ):
        raise ConfidNetOOFFitError("invalid RF class order")
    if state.get("sklearn_version") != SKLEARN_VERSION:
        raise ConfidNetOOFFitError("RF state implementation version changed")
    trees = state.get("trees")
    if not isinstance(trees, list) or len(trees) != RF_DEFAULTS["n_estimators"]:
        raise ConfidNetOOFFitError("RF ensemble size differs from recipe")
    features = _router_matrix(matrix, len(state["feature_names"])).astype(np.float32)
    probability = np.zeros((len(features), 4), dtype=np.float64)
    try:
        for tree in trees:
            arrays = _tree(tree, family="rf", feature_count=features.shape[1])
            probability += arrays["value"][_leaves(features, arrays, inclusive=True)]
    except ValueError as exc:
        raise ConfidNetOOFFitError(str(exc)) from exc
    probability /= len(trees)
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1, atol=1e-12, rtol=0
    ):
        raise ConfidNetOOFFitError("nonfinite/invalid RF probabilities")
    return probability


def fit_router(
    train_x, target, predict_x_by_split, *, seed, feature_names, parameters=None
):
    """Fit one approved matched RF; no validation targets are accepted."""
    _seed(seed)
    _router_features(feature_names)
    parameters = _rf_parameters(parameters)
    if sklearn.__version__ != SKLEARN_VERSION:
        raise ConfidNetOOFFitError(f"RF requires sklearn {SKLEARN_VERSION}")
    train_x = _router_matrix(train_x, len(feature_names))
    if (
        not isinstance(target, np.ndarray)
        or target.dtype.kind not in "iu"
        or target.shape != (len(train_x),)
        or set(target.tolist()) != {0, 1, 2, 3}
    ):
        raise ConfidNetOOFFitError(
            "router targets must contain all four integer states"
        )
    if not isinstance(predict_x_by_split, dict) or "train" in predict_x_by_split:
        raise ConfidNetOOFFitError("prediction splits must exclude train")
    arrays = {
        k: _router_matrix(v, len(feature_names)) for k, v in predict_x_by_split.items()
    }
    started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator = RandomForestClassifier(**parameters, random_state=seed)
        estimator.fit(train_x, target)
        if captured:
            raise ConfidNetOOFFitError(
                f"RF fit emitted warnings: {[str(v.message) for v in captured]}"
            )
        if not np.array_equal(estimator.classes_, np.arange(4)):
            raise ConfidNetOOFFitError("native RF changed class order")
        trees = []
        for fitted in estimator.estimators_:
            tree = fitted.tree_
            values = tree.value[:, 0, :].astype(np.float64, copy=True)
            values /= values.sum(axis=1, keepdims=True)
            trees.append(
                {
                    "left": tree.children_left.tolist(),
                    "right": tree.children_right.tolist(),
                    "feature": tree.feature.tolist(),
                    "threshold": tree.threshold.tolist(),
                    "value": values.tolist(),
                }
            )
        state = _seal(
            {
                "schema_version": RF_SCHEMA,
                "parameters": parameters,
                "training_seed": seed,
                "fit_row_count": len(train_x),
                "feature_names": list(feature_names),
                "classes": [0, 1, 2, 3],
                "target_order": list(TARGET_ORDER),
                "trees": trees,
                "sklearn_version": str(sklearn.__version__),
            }
        )
        predictions, parity = {}, {}
        for name, matrix in {"train": train_x, **arrays}.items():
            native = estimator.predict_proba(matrix)
            portable = predict_router(state, matrix)
            error = float(np.max(np.abs(native - portable)))
            if error > 1e-12 or not np.array_equal(
                native.argmax(axis=1), portable.argmax(axis=1)
            ):
                raise ConfidNetOOFFitError(
                    f"RF native/portable parity failed for {name}"
                )
            predictions[name] = native
            parity[name] = {
                "maximum_abs_error": error,
                "atol": 1e-12,
                "argmax_exact": True,
            }
    return {
        "status": "complete",
        "state": state,
        "predictions": predictions,
        "metadata": {
            "runtime_seconds": time.perf_counter() - started,
            "validation_labels_accessed": False,
            "validation_features_used_for_fitting": False,
            "feature_count": len(feature_names),
            "train_rows": len(train_x),
            "prediction_parity": parity,
            "class_order": list(TARGET_ORDER),
            "feature_importances": estimator.feature_importances_.tolist(),
            "prediction_convention": "native_sklearn_float64",
        },
    }

"""Fixed-budget CPU training for small development-only MLP routers.

The only labels accepted are training OOF targets. Validation features are used
once for native-to-portable prediction parity after the final training epoch.
"""

from __future__ import annotations

from copy import deepcopy
import time
import warnings

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as functional

from multimodal.geo_helpfulness_router_candidate_state import (
    TARGET_ORDER,
    full_feature_columns,
    variant_feature_indices,
)
from multimodal.geo_helpfulness_router_mlp_state import (
    IMPLEMENTATION,
    SCHEMA_VERSION,
    TORCH_VERSION,
    parameter_count,
    predict_logits,
    predict_state,
    resolve_parameters,
    seal_state,
)


class MLPFitError(ValueError):
    """Input, training numerics, or portable parity violate the fixed contract."""


def _inputs(train_x, target, val_x, spec, seed):
    for name, matrix in (("training", train_x), ("validation", val_x)):
        if (
            not isinstance(matrix, np.ndarray)
            or matrix.dtype != np.float64
            or matrix.ndim != 2
            or matrix.shape[1] != 727
            or not len(matrix)
            or not np.isfinite(matrix).all()
        ):
            raise MLPFitError(f"{name} matrix must be nonempty finite float64 (n,727)")
    target = np.asarray(target)
    if (
        target.shape != (len(train_x),)
        or target.dtype.kind not in "iu"
        or set(target.tolist()) != {0, 1, 2, 3}
    ):
        raise MLPFitError("training targets must contain exactly the four fixed states")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (1, 2, 3, 4):
        raise MLPFitError("training seed must be in 1..4")
    if (
        not isinstance(spec, dict)
        or spec.get("family") != "mlp"
        or not isinstance(spec.get("config_id"), str)
        or not spec["config_id"].strip()
    ):
        raise MLPFitError("invalid MLP candidate identity")
    indices = variant_feature_indices(spec["variant"])
    parameters = resolve_parameters(spec)
    if torch.__version__ != TORCH_VERSION:
        raise MLPFitError(
            f"MLP fitting requires Torch {TORCH_VERSION}; found {torch.__version__}"
        )
    return target.astype(np.int64, copy=False), indices, parameters


def _network(input_width, hidden_widths, dropout):
    layers = []
    previous = input_width
    for width in hidden_widths:
        layers.extend(
            [
                nn.Linear(previous, width, dtype=torch.float64, device="cpu"),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        previous = width
    layers.append(nn.Linear(previous, 4, dtype=torch.float64, device="cpu"))
    return nn.Sequential(*layers).cpu()


def _export(model, spec, seed, rows, parameters):
    return seal_state(
        {
            "schema_version": SCHEMA_VERSION,
            "config_id": spec["config_id"],
            "family": "mlp",
            "variant": spec["variant"],
            "training_seed": seed,
            "parameters": parameters,
            "feature_input_names": full_feature_columns(),
            "feature_indices": variant_feature_indices(spec["variant"]),
            "classes": [0, 1, 2, 3],
            "target_order": list(TARGET_ORDER),
            "fit_role": "development_train_oof",
            "fit_row_count": rows,
            "initialization_seed": seed,
            "shuffle_seed": seed,
            "dropout_seed": seed,
            "parameter_count": parameter_count(
                spec["variant"], parameters["hidden_widths"]
            ),
            "implementation": deepcopy(IMPLEMENTATION),
            "layers": [
                {
                    "weight": layer.weight.detach().numpy().tolist(),
                    "bias": layer.bias.detach().numpy().tolist(),
                }
                for layer in model
                if isinstance(layer, nn.Linear)
            ],
        }
    )


def _parity(native_logits, native_probability, portable_logits, portable_probability):
    if (
        not np.allclose(native_logits, portable_logits, atol=1e-12, rtol=1e-12)
        or not np.allclose(
            native_probability, portable_probability, atol=1e-12, rtol=1e-12
        )
        or not np.array_equal(
            native_probability.argmax(axis=1), portable_probability.argmax(axis=1)
        )
    ):
        raise MLPFitError(
            "native/portable MLP logits, probabilities, or class predictions differ"
        )
    return {
        "maximum_logit_abs_error": float(
            np.max(np.abs(native_logits - portable_logits))
        ),
        "maximum_probability_abs_error": float(
            np.max(np.abs(native_probability - portable_probability))
        ),
        "probability_atol": 1e-12,
        "probability_rtol": 1e-12,
        "class_prediction_exact": True,
    }


def fit_candidate(train_x, target, val_x, spec, seed):
    """Return portable final-epoch state, train/validation q, and 300-epoch diagnostics."""
    target, indices, parameters = _inputs(train_x, target, val_x, spec, seed)
    training = torch.from_numpy(
        np.ascontiguousarray(train_x[:, indices], dtype=np.float64)
    )
    labels = torch.from_numpy(np.ascontiguousarray(target, dtype=np.int64))
    validation = torch.from_numpy(
        np.ascontiguousarray(val_x[:, indices], dtype=np.float64)
    )
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    before_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    history = []
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1), torch.random.fork_rng(
            devices=[]
        ), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            torch.manual_seed(seed)
            model = _network(
                len(indices), parameters["hidden_widths"], parameters["dropout"]
            )
            count = sum(value.numel() for value in model.parameters())
            if count != parameter_count(spec["variant"], parameters["hidden_widths"]):
                raise MLPFitError(
                    "native MLP parameter count differs from architecture"
                )
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [
                            p
                            for name, p in model.named_parameters()
                            if name.endswith("weight")
                        ],
                        "weight_decay": parameters["weight_decay"],
                    },
                    {
                        "params": [
                            p
                            for name, p in model.named_parameters()
                            if name.endswith("bias")
                        ],
                        "weight_decay": parameters["bias_weight_decay"],
                    },
                ],
                lr=parameters["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=False,
                maximize=False,
                foreach=False,
                capturable=False,
                differentiable=False,
                fused=False,
            )
            # Independent seeded shuffle stream keeps minibatch order identical
            # across widths/depths despite their different dropout draw counts.
            shuffle = torch.Generator(device="cpu").manual_seed(seed)
            torch.manual_seed(seed)
            for epoch in range(1, parameters["epochs"] + 1):
                model.train()
                total_ce, maximum_gradient = 0.0, 0.0
                for rows in torch.randperm(
                    len(training), generator=shuffle, device="cpu"
                ).split(parameters["batch_size"]):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(training[rows])
                    loss = functional.cross_entropy(
                        logits, labels[rows], reduction="mean"
                    )
                    if not bool(torch.isfinite(loss)):
                        raise MLPFitError(
                            f"nonfinite optimization cross-entropy at epoch {epoch}"
                        )
                    loss.backward()
                    gradients = [p.grad for p in model.parameters()]
                    if any(
                        value is None or not bool(torch.isfinite(value).all())
                        for value in gradients
                    ):
                        raise MLPFitError(f"nonfinite gradient at epoch {epoch}")
                    norm = torch.sqrt(sum(value.square().sum() for value in gradients))
                    if not bool(torch.isfinite(norm)):
                        raise MLPFitError(f"nonfinite gradient norm at epoch {epoch}")
                    maximum_gradient = max(maximum_gradient, float(norm))
                    total_ce += float(loss.detach()) * len(rows)
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits = model(training)
                    nll = functional.cross_entropy(logits, labels, reduction="mean")
                    accuracy = (logits.argmax(dim=1) == labels).to(torch.float64).mean()
                if not bool(torch.isfinite(logits).all()) or not bool(
                    torch.isfinite(nll)
                ):
                    raise MLPFitError(
                        f"nonfinite dropout-off training objective at epoch {epoch}"
                    )
                history.append(
                    {
                        "epoch": epoch,
                        "optimization_cross_entropy": total_ce / len(training),
                        "train_nll": float(nll),
                        "train_accuracy": float(accuracy),
                        "maximum_gradient_l2_norm": maximum_gradient,
                        "gradients_finite": True,
                    }
                )
            state = _export(model, spec, seed, len(training), parameters)
            train_probability = predict_state(train_x, state)
            validation_probability = predict_state(val_x, state)
            parity = {}
            for name, native_input, full_input, pure_probability in (
                ("train", training, train_x, train_probability),
                ("validation_features_only", validation, val_x, validation_probability),
            ):
                with torch.no_grad():
                    native_logits = model(native_input)
                    native_probability = native_logits.softmax(dim=1)
                parity[name] = _parity(
                    native_logits.numpy(),
                    native_probability.numpy(),
                    predict_logits(full_input, state),
                    pure_probability,
                )
    finally:
        torch.set_num_threads(before_threads)
        torch.use_deterministic_algorithms(
            before_deterministic, warn_only=before_warn_only
        )
    diagnostics = {
        "config_id": spec["config_id"],
        "family": "mlp",
        "variant": spec["variant"],
        "training_seed": seed,
        "fit_role": "development_train_oof",
        "fit_row_count": len(train_x),
        "feature_count": len(indices),
        "hidden_widths": parameters["hidden_widths"],
        "parameter_count": count,
        "fit_performed": True,
        "final_epoch": parameters["epochs"],
        "checkpoint": "final_epoch",
        "history": history,
        "validation_labels_accessed": False,
        "validation_features_used_for_fitting": False,
        "early_stopping_used": False,
        "sample_or_class_weighting_used": False,
        "optimizer": "AdamW",
        "learning_rate": parameters["learning_rate"],
        "weight_decay": parameters["weight_decay"],
        "bias_weight_decay": parameters["bias_weight_decay"],
        "optimization_loss_interpretation": "image_mean_cross_entropy_with_dropout_excludes_decoupled_AdamW_weight_decay",
        "train_metric_interpretation": "dropout_off_resubstitution_diagnostic_not_independent_performance",
        "late_training_diagnostic": {
            "epoch200_train_nll": history[199]["train_nll"],
            "epoch300_train_nll": history[299]["train_nll"],
            "change_200_to300": history[299]["train_nll"] - history[199]["train_nll"],
            "convergence_claim": False,
        },
        "threadpool_limit": 1,
        "deterministic_algorithms": True,
        "probability_parity": parity,
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in captured
        ],
        "state_sha256": state["state_sha256"],
        "fit_and_export_seconds": float(time.perf_counter() - started),
    }
    return state, train_probability, validation_probability, diagnostics


__all__ = ["MLPFitError", "fit_candidate"]

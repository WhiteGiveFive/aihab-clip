"""Pure scoring of caller-supplied frozen-expert and ConfidNet predictions.

No model fitting, artifact discovery, threshold selection, or filesystem access
occurs here. TCP targets use the expert's native temperature-one softmax.
"""

from __future__ import annotations

from typing import Any

import numpy as np


QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
POLICIES = (
    "fusion",
    "always_geo_disagreement",
    "mcp_difference_positive",
    "tcp_difference_positive",
)


class ConfidNetMetricError(ValueError):
    """Caller-supplied predictions do not satisfy the metric contract."""


def _probabilities(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] < 2
        or not np.isfinite(values).all()
    ):
        raise ConfidNetMetricError("logits must be finite nonempty (n, classes>=2)")
    with np.errstate(over="ignore", under="ignore"):
        exponentials = np.exp(values - values.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _labels(labels: Any, probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if (
        values.shape != (len(probabilities),)
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values >= probabilities.shape[1])
    ):
        raise ConfidNetMetricError("labels must be matching integer class IDs")
    return values.astype(np.int64, copy=False)


def _confidence(tcp_hat: Any, size: int) -> np.ndarray:
    values = np.asarray(tcp_hat, dtype=np.float64)
    if (
        values.shape != (size,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise ConfidNetMetricError("tcp_hat must be a matching finite [0,1] vector")
    return values


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if len(values) == 0:
        return None
    return {
        str(q): float(value)
        for q, value in zip(QUANTILES, np.quantile(values, QUANTILES))
    }


def _regression(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(error * error))
    return {"mse": mse, "mae": float(np.mean(np.abs(error))), "rmse": mse**0.5}


def _failure_detection(failure: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Noninterpolated AP and pairwise ROC AUC; exact ties form one group."""
    positives = int(failure.sum())
    negatives = len(failure) - positives
    order = np.argsort(-scores, kind="stable")
    ranked_scores, ranked_failure = scores[order], failure[order]
    group_ends = np.r_[
        np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]), len(scores) - 1
    ]
    tp = np.cumsum(ranked_failure, dtype=np.int64)[group_ends]
    fp = group_ends + 1 - tp
    ap = None
    if positives:
        ap = float(np.sum(np.diff(np.r_[0, tp]) / positives * tp / (group_ends + 1)))
    auc = None
    if positives and negatives:
        # Trapezoids give tied positive/negative pairs half credit.
        tpr = np.r_[0.0, tp / positives]
        fpr = np.r_[0.0, fp / negatives]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return {
        "average_precision": ap,
        "roc_auc": auc,
        "average_precision_undefined_reason": "no_failures" if not positives else None,
        "roc_auc_undefined_reason": (
            "no_failures" if not positives else "no_correct_predictions" if not negatives else None
        ),
    }


def score_expert(labels: Any, logits: Any, tcp_hat: Any) -> dict[str, Any]:
    """Score top-1 classification, TCP regression, and failure detection.

    Top-1 logit ties choose the first class ID. Failure-detection AP is null if
    there are no errors and one if every prediction is wrong. ROC AUC is null
    whenever either outcome is absent. Entropy is in natural-log units.
    """
    probabilities = _probabilities(logits)
    target = _labels(labels, probabilities)
    estimate = _confidence(tcp_hat, len(target))
    correct = probabilities.argmax(axis=1) == target
    tcp = probabilities[np.arange(len(target)), target]
    mcp = probabilities.max(axis=1)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny)),
        axis=1,
    )
    detectors = {
        "predicted_tcp": _failure_detection(~correct, 1.0 - estimate),
        "mcp": _failure_detection(~correct, 1.0 - mcp),
        "entropy": _failure_detection(~correct, entropy),
    }
    regression = _regression(tcp, estimate)
    mcp_regression = _regression(tcp, mcp)
    comparisons = {}
    for name in ("mcp", "entropy"):
        comparisons[name] = {
            metric + "_gain": (
                detectors["predicted_tcp"][metric] - detectors[name][metric]
                if detectors["predicted_tcp"][metric] is not None
                and detectors[name][metric] is not None
                else None
            )
            for metric in ("average_precision", "roc_auc")
        }
    return {
        "row_count": len(target),
        "class_count": probabilities.shape[1],
        "probability_convention": "native_softmax_temperature_1",
        "top1_correct_count": int(correct.sum()),
        "top1_error_count": int((~correct).sum()),
        "top1_accuracy": float(correct.mean()),
        "failure_prevalence": float((~correct).mean()),
        "tcp_quantiles": _quantiles(tcp),
        "tcp_correct_quantiles": _quantiles(tcp[correct]),
        "tcp_incorrect_quantiles": _quantiles(tcp[~correct]),
        "mcp_quantiles": _quantiles(mcp),
        "predicted_tcp_quantiles": _quantiles(estimate),
        "tcp_fraction_ge_0.99": float(np.mean(tcp >= 0.99)),
        "tcp_fraction_ge_0.999": float(np.mean(tcp >= 0.999)),
        "tcp_regression": regression,
        "mcp_tcp_regression": mcp_regression,
        "tcp_mse_reduction_vs_mcp": mcp_regression["mse"] - regression["mse"],
        "failure_detection": detectors,
        "failure_detection_comparison": comparisons,
    }


def _policy_arrays(
    labels: Any,
    geo_logits: Any,
    raw_logits: Any,
    geo_tcp_hat: Any,
    raw_tcp_hat: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    geo = _probabilities(geo_logits)
    fusion = _probabilities(raw_logits)
    if geo.shape != fusion.shape:
        raise ConfidNetMetricError("geo and fusion logits must have matching shapes")
    target = _labels(labels, fusion)
    geo_tcp = _confidence(geo_tcp_hat, len(target))
    fusion_tcp = _confidence(raw_tcp_hat, len(target))
    geo_pred, fusion_pred = geo.argmax(axis=1), fusion.argmax(axis=1)
    disagreement = geo_pred != fusion_pred
    actions = {
        "fusion": np.zeros(len(target), dtype=bool),
        "always_geo_disagreement": disagreement,
        "mcp_difference_positive": disagreement & (geo.max(axis=1) > fusion.max(axis=1)),
        "tcp_difference_positive": disagreement & (geo_tcp > fusion_tcp),
    }
    return target, geo_pred, fusion_pred, actions


def score_policies(
    labels: Any,
    geo_logits: Any,
    raw_logits: Any,
    geo_tcp_hat: Any,
    raw_tcp_hat: Any,
) -> dict[str, dict[str, Any]]:
    """Score fixed policies; intervene only on disagreement with strict > ties."""
    target, geo_pred, fusion_pred, actions = _policy_arrays(
        labels, geo_logits, raw_logits, geo_tcp_hat, raw_tcp_hat
    )
    geo_correct, fusion_correct = geo_pred == target, fusion_pred == target
    disagreement_count = int(np.sum(geo_pred != fusion_pred))
    result = {}
    for name, action in actions.items():
        rescues = int(np.sum(action & geo_correct & ~fusion_correct))
        harms = int(np.sum(action & ~geo_correct & fusion_correct))
        neutral_wrong = int(np.sum(action & ~geo_correct & ~fusion_correct))
        correct_count = int(fusion_correct.sum()) + rescues - harms
        action_count = int(action.sum())
        result[name] = {
            "row_count": len(target),
            "disagreement_count": disagreement_count,
            "actions": action_count,
            "rescues": rescues,
            "harms": harms,
            "neutral_wrong": neutral_wrong,
            "net": rescues - harms,
            "correct_count": correct_count,
            "accuracy": correct_count / len(target),
            "accuracy_gain_vs_fusion": (rescues - harms) / len(target),
            "coverage": action_count / len(target),
            "disagreement_coverage": (
                action_count / disagreement_count if disagreement_count else None
            ),
        }
    return result


def paired_plot_bootstrap_gain(
    labels: Any,
    geo_logits: Any,
    raw_logits: Any,
    geo_tcp_hat: Any,
    raw_tcp_hat: Any,
    plot_ids: Any,
    seed_ids: Any = None,
    *,
    n_resamples: int = 2000,
    random_seed: int = 20260826,
) -> dict[str, dict[str, Any]]:
    """Paired 95% percentile intervals for image-weighted gain over fusion.

    Concatenate all seed predictions before calling. Resample plots, preserving
    all images and all seeds jointly within each sampled plot. Optional seed IDs
    verify that every plot has the same seeds. Intervals condition on the fitted
    models and fixed policies; they do not account for model/policy selection.
    """
    target, geo_pred, fusion_pred, actions = _policy_arrays(
        labels, geo_logits, raw_logits, geo_tcp_hat, raw_tcp_hat
    )
    plots = np.asarray(plot_ids)
    if plots.shape != (len(target),):
        raise ConfidNetMetricError("plot_ids must match the prediction rows")
    if plots.dtype.kind in "fc" and not np.isfinite(plots).all():
        raise ConfidNetMetricError("plot_ids cannot be nonfinite")
    if any(value is None for value in plots):
        raise ConfidNetMetricError("plot_ids cannot contain missing values")
    try:
        _, inverse = np.unique(plots, return_inverse=True)
    except TypeError as exc:
        raise ConfidNetMetricError("plot_ids must be consistently comparable") from exc
    group_count = int(inverse.max()) + 1
    seed_count = None
    if seed_ids is not None:
        seeds = np.asarray(seed_ids)
        if seeds.shape != plots.shape or seeds.dtype.kind not in "iu":
            raise ConfidNetMetricError("seed_ids must be matching integer IDs")
        unique_seeds = np.unique(seeds)
        seed_count = len(unique_seeds)
        for index in range(group_count):
            if not np.array_equal(np.unique(seeds[inverse == index]), unique_seeds):
                raise ConfidNetMetricError("every plot must contain every supplied seed")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples < 1
    ):
        raise ConfidNetMetricError("n_resamples must be a positive integer")
    group_sizes = np.bincount(inverse)
    geo_correct = (geo_pred == target).astype(np.int64)
    fusion_correct = (fusion_pred == target).astype(np.int64)
    group_gains = np.vstack(
        [
            np.bincount(inverse, weights=action * (geo_correct - fusion_correct))
            for action in actions.values()
        ]
    )
    draws = np.empty((len(POLICIES), n_resamples), dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    for index in range(n_resamples):
        sampled = rng.integers(0, group_count, size=group_count)
        draws[:, index] = group_gains[:, sampled].sum(axis=1) / group_sizes[sampled].sum()
    result = {}
    for index, name in enumerate(actions):
        lower, upper = np.quantile(draws[index], [0.025, 0.975])
        result[name] = {
            "accuracy_gain_vs_fusion": float(group_gains[index].sum() / len(target)),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "bootstrap_standard_error": (
                float(np.std(draws[index], ddof=1)) if n_resamples > 1 else None
            ),
            "row_count": len(target),
            "plot_count": group_count,
            "seed_count": seed_count,
            "n_resamples": int(n_resamples),
            "random_seed": int(random_seed),
            "resampling_unit": "plot_with_all_images_and_seeds",
            "estimand": "image_weighted_accuracy_gain_vs_fusion",
            "interval_method": "paired_plot_percentile",
            "conditional_on_fitted_models_and_fixed_policies": True,
            "selection_adjusted": False,
        }
    return result

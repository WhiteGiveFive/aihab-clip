"""Pure diagnostics for the approved OOF-supervised ConfidNet study.

Callers supply already sealed predictions and explicitly joined labels. This
module neither discovers artifacts nor chooses policies or model parameters.
Stored native float32 confidence estimates are used without model reevaluation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from multimodal.geo_helpfulness_confidnet_report import (
    ConfidNetMetricError,
    _confidence,
    _failure_detection,
    _labels,
    _probabilities,
    _quantiles,
    score_expert,
)


EXPERTS = ("image", "geo", "raw")
ROUTERS = ("baseline", "augmented")
STATES = ("rescue", "harm", "both_correct", "both_wrong")
NLL_PROBABILITY_FLOOR = np.finfo(np.float64).tiny
POLICIES = (
    "fusion",
    "mcp_difference_positive",
    "tcp_difference_positive",
    "rf_baseline",
    "rf_augmented",
)


def _ids(values: Any, size: int, name: str, *, unique: bool = False) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (size,):
        raise ConfidNetMetricError(f"{name} must match prediction rows")
    if any(value is None for value in result):
        raise ConfidNetMetricError(f"{name} cannot contain missing values")
    if result.dtype.kind in "fc" and not np.isfinite(result).all():
        raise ConfidNetMetricError(f"{name} must be finite")
    result = result.astype(str)
    if np.any(result == "") or np.any(np.char.lower(result) == "nan"):
        raise ConfidNetMetricError(f"{name} cannot contain empty or missing values")
    if unique and len(np.unique(result)) != size:
        raise ConfidNetMetricError(f"{name} must identify unique rows within a seed")
    return result


def _router_probabilities(values: Any, size: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.shape != (size, 4)
        or not np.isfinite(result).all()
        or np.any(result < 0)
        or np.any(result > 1)
        or not np.allclose(result.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    ):
        raise ConfidNetMetricError("router probabilities must be finite (n,4) simplex rows")
    return result


def _prepare(payload: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = {name: _probabilities(payload[f"{name}_logits"]) for name in EXPERTS}
    shape = probabilities["raw"].shape
    if any(value.shape != shape for value in probabilities.values()):
        raise ConfidNetMetricError("all experts must have matching logit shapes")
    labels = _labels(payload["labels"], probabilities["raw"])
    result = {
        "labels": labels,
        "probabilities": probabilities,
        "predictions": {name: value.argmax(axis=1) for name, value in probabilities.items()},
        "estimates": {
            name: _confidence(payload[f"{name}_tcp_hat"], len(labels)) for name in EXPERTS
        },
        "plots": _ids(payload["plot_ids"], len(labels), "plot_ids"),
        "rows": _ids(payload["row_uids"], len(labels), "row_uids", unique=True),
        "router_q": {
            name: _router_probabilities(payload[f"router_{name}_q"], len(labels))
            for name in ROUTERS
        },
    }
    geo_correct = result["predictions"]["geo"] == labels
    raw_correct = result["predictions"]["raw"] == labels
    state = np.full(len(labels), 3, dtype=np.int64)
    state[geo_correct & ~raw_correct] = 0
    state[~geo_correct & raw_correct] = 1
    state[geo_correct & raw_correct] = 2
    result["state"] = state
    return result


def _actions(prepared: Mapping[str, Any]) -> dict[str, np.ndarray]:
    disagreement = prepared["predictions"]["geo"] != prepared["predictions"]["raw"]
    p, estimate, q = prepared["probabilities"], prepared["estimates"], prepared["router_q"]
    return {
        "fusion": np.zeros(len(disagreement), dtype=bool),
        "mcp_difference_positive": disagreement & (p["geo"].max(axis=1) > p["raw"].max(axis=1)),
        "tcp_difference_positive": disagreement & (estimate["geo"] > estimate["raw"]),
        "rf_baseline": disagreement & (q["baseline"][:, 0] > q["baseline"][:, 1]),
        "rf_augmented": disagreement & (q["augmented"][:, 0] > q["augmented"][:, 1]),
    }


def policy_arrays(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return five fixed action masks, respecting disagreement and exact ties."""
    return _actions(_prepare(payload))


def _weighted_f1(labels: np.ndarray, predicted: np.ndarray, class_count: int) -> float:
    support = np.bincount(labels, minlength=class_count)
    predicted_support = np.bincount(predicted, minlength=class_count)
    tp = np.bincount(labels[labels == predicted], minlength=class_count)
    denominator = support + predicted_support
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros(class_count), where=denominator != 0)
    return float(np.sum(support * f1) / len(labels))


def _ranking(positive: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if len(positive) == 0:
        return {
            "average_precision": None,
            "roc_auc": None,
            "undefined_reason": "no_disagreements",
        }
    result = _failure_detection(positive, scores)
    return {
        "average_precision": result["average_precision"],
        "roc_auc": result["roc_auc"],
        "average_precision_undefined_reason": "no_rescues" if not positive.any() else None,
        "roc_auc_undefined_reason": (
            "no_rescues" if not positive.any() else "all_rescues" if positive.all() else None
        ),
    }


def score_seed(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Confidence, four-state, ranking and fixed-policy metrics for one seed."""
    data = _prepare(payload)
    labels, state = data["labels"], data["state"]
    count = len(labels)
    class_count = data["probabilities"]["raw"].shape[1]
    geo, raw = data["predictions"]["geo"], data["predictions"]["raw"]
    disagreement = geo != raw
    confidence = {}
    for name in EXPERTS:
        metric = score_expert(labels, payload[f"{name}_logits"], data["estimates"][name])
        estimate = data["estimates"][name]
        correct = data["predictions"][name] == labels
        tcp = data["probabilities"][name][np.arange(count), labels]
        metric.update({
            "predicted_tcp_correct_quantiles": _quantiles(estimate[correct]),
            "predicted_tcp_incorrect_quantiles": _quantiles(estimate[~correct]),
            "predicted_tcp_fraction_le_1e-6": float(np.mean(estimate <= 1e-6)),
            "predicted_tcp_fraction_ge_1_minus_1e-6": float(np.mean(estimate >= 1 - 1e-6)),
            "always_one_tcp_mse": float(np.mean((1 - tcp) ** 2)),
            "split_mean_tcp_mse": float(np.mean((tcp.mean() - tcp) ** 2)),
            "mean_true_tcp": float(tcp.mean()),
            "mean_predicted_tcp": float(estimate.mean()),
        })
        confidence[name] = metric
    router = {}
    for name, q in data["router_q"].items():
        one_hot = np.eye(4)[state]
        true_probability = q[np.arange(count), state]
        router[name] = {
            "four_state_order": list(STATES),
            "nll": float(-np.log(np.maximum(true_probability, NLL_PROBABILITY_FLOOR)).mean()),
            "nll_probability_floor": NLL_PROBABILITY_FLOOR,
            "nll_probability_floor_count": int(np.sum(true_probability < NLL_PROBABILITY_FLOOR)),
            "brier": float(np.mean(np.sum((q - one_hot) ** 2, axis=1))),
            "brier_convention": "sum_over_four_classes_then_mean_over_rows",
            "four_state_accuracy": float(np.mean(q.argmax(axis=1) == state)),
            "disagreement_count": int(disagreement.sum()),
            "rescue_count_on_disagreement": int(np.sum(disagreement & (state == 0))),
            "rescue_ranking_on_disagreement": {
                "q_rescue": _ranking(state[disagreement] == 0, q[disagreement, 0]),
                "q_rescue_minus_q_harm": _ranking(
                    state[disagreement] == 0, q[disagreement, 0] - q[disagreement, 1]
                ),
            },
        }
    policies = {}
    fusion_f1 = _weighted_f1(labels, raw, class_count)
    for name, action in _actions(data).items():
        predicted = np.where(action, geo, raw)
        rescues = int(np.sum(action & (state == 0)))
        harms = int(np.sum(action & (state == 1)))
        neutral = int(np.sum(action & (state == 3)))
        actions = int(action.sum())
        f1 = _weighted_f1(labels, predicted, class_count)
        policies[name] = {
            "row_count": count,
            "plot_count": len(np.unique(data["plots"])),
            "disagreement_count": int(disagreement.sum()),
            "actions": actions,
            "rescues": rescues,
            "harms": harms,
            "neutral_wrong": neutral,
            "net": rescues - harms,
            "correct_count": int(np.sum(predicted == labels)),
            "accuracy": float(np.mean(predicted == labels)),
            "weighted_f1": f1,
            "accuracy_gain_vs_fusion": (rescues - harms) / count,
            "weighted_f1_gain_vs_fusion": f1 - fusion_f1,
            "coverage": actions / count,
            "disagreement_coverage": actions / int(disagreement.sum()) if disagreement.any() else None,
            "plots_affected": len(np.unique(data["plots"][action])),
            "rescue_precision": rescues / actions if actions else None,
            "rescue_recall": rescues / int(np.sum(state == 0)) if np.any(state == 0) else None,
        }
        if rescues + harms + neutral != actions:
            raise ConfidNetMetricError("disagreement action accounting does not reconcile")
    return {
        "row_count": count,
        "plot_count": len(np.unique(data["plots"])),
        "four_state_counts": {name: int(np.sum(state == index)) for index, name in enumerate(STATES)},
        "confidence": confidence,
        "routers": router,
        "policies": policies,
    }


def _mean_tree(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _mean_tree([value[key] for value in values]) for key in first}
    if first is None or any(value is None for value in values):
        return None
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return float(np.mean(values))
    return first


def _validate_seed_rows(by_seed: Mapping[int, Mapping[str, Any]]) -> list[int]:
    seeds = sorted(by_seed)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in seeds):
        raise ConfidNetMetricError("seed keys must be nonempty integers")
    reference = None
    for seed in seeds:
        data = _prepare(by_seed[seed])
        identities = sorted(zip(data["rows"].tolist(), data["plots"].tolist(), data["labels"].tolist()))
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ConfidNetMetricError("every seed must contain identical row, plot and label identities")
    return seeds


def _pooled_payload(by_seed: Mapping[int, Mapping[str, Any]], seeds: list[int]) -> dict[str, Any]:
    keys = ["labels", "plot_ids", "row_uids"]
    keys += [f"{name}_{suffix}" for name in EXPERTS for suffix in ("logits", "tcp_hat")]
    keys += [f"router_{name}_q" for name in ROUTERS]
    pooled = {key: np.concatenate([np.asarray(by_seed[seed][key]) for seed in seeds]) for key in keys}
    # Seed copies are observations for metrics, not independent bootstrap units.
    pooled["row_uids"] = np.concatenate([
        np.asarray([f"seed{seed}:{row}" for row in by_seed[seed]["row_uids"]]) for seed in seeds
    ])
    return pooled


def paired_plot_bootstrap(
    by_seed: Mapping[int, Mapping[str, Any]],
    *,
    n_resamples: int = 2000,
    random_seed: int = 20260826,
) -> dict[str, Any]:
    """Same whole-plot resamples for every policy and all seed realizations.

    Contrasts report accuracy and mean-seed weighted-F1 differences, together
    with action/rescue/harm/neutral/net rates. The bootstrap conditions on the
    fitted models and fixed policies and does not adjust for past validation use.
    """
    seeds = _validate_seed_rows(by_seed)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ConfidNetMetricError("n_resamples must be a positive integer")
    first = _prepare(by_seed[seeds[0]])
    plots = np.unique(first["plots"])
    group_count = len(plots)
    class_count = first["probabilities"]["raw"].shape[1]
    # axes: plot, seed, policy, sufficient statistic (seven counts + three class vectors)
    groups = np.zeros((group_count, len(seeds), len(POLICIES), 7 + 3 * class_count))
    for seed_index, seed in enumerate(seeds):
        data = _prepare(by_seed[seed])
        inverse = np.searchsorted(plots, data["plots"])
        labels = data["labels"]
        for policy_index, action in enumerate(_actions(data).values()):
            prediction = np.where(action, data["predictions"]["geo"], data["predictions"]["raw"])
            columns = [
                np.ones(len(labels)), prediction == labels, action,
                action & (data["state"] == 0), action & (data["state"] == 1),
                action & (data["state"] == 3),
                action.astype(int) * ((data["state"] == 0).astype(int) - (data["state"] == 1).astype(int)),
            ]
            columns += [labels == index for index in range(class_count)]
            columns += [prediction == index for index in range(class_count)]
            columns += [(labels == index) & (prediction == index) for index in range(class_count)]
            for column, weights in enumerate(columns):
                groups[:, seed_index, policy_index, column] = np.bincount(inverse, weights=weights, minlength=group_count)

    metric_names = ("accuracy", "coverage", "rescue_rate", "harm_rate", "neutral_wrong_rate", "net_rate", "weighted_f1")

    def metrics(counts: np.ndarray) -> np.ndarray:
        rates = counts[..., 1:7] / counts[..., :1]
        support = counts[..., 7:7 + class_count]
        predicted_support = counts[..., 7 + class_count:7 + 2 * class_count]
        tp = counts[..., 7 + 2 * class_count:]
        denominator = support + predicted_support
        f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
        weighted_f1 = (support * f1).sum(axis=-1) / counts[..., 0]
        return np.concatenate([rates, weighted_f1[..., None]], axis=-1).mean(axis=0)

    observed = metrics(groups.sum(axis=0))
    draws = np.empty((n_resamples, len(POLICIES), len(metric_names)))
    rng = np.random.default_rng(random_seed)
    for index in range(n_resamples):
        sample = rng.integers(0, group_count, size=group_count)
        draws[index] = metrics(groups[sample].sum(axis=0))
    pairs = [(index, 0) for index in range(len(POLICIES))] + [(4, 3)]
    contrasts = {}
    for treatment, reference in pairs:
        name = f"{POLICIES[treatment]}_minus_{POLICIES[reference]}"
        difference = draws[:, treatment] - draws[:, reference]
        contrasts[name] = {}
        for index, metric in enumerate(metric_names):
            low, high = np.quantile(difference[:, index], [0.025, 0.975])
            contrasts[name][metric] = {
                "difference": float(observed[treatment, index] - observed[reference, index]),
                "ci95_lower": float(low),
                "ci95_upper": float(high),
                "bootstrap_standard_error": float(np.std(difference[:, index], ddof=1)) if n_resamples > 1 else None,
            }
    return {
        "n_resamples": int(n_resamples),
        "random_seed": int(random_seed),
        "plot_count": group_count,
        "seed_count": len(seeds),
        "row_count_per_seed": len(first["labels"]),
        "resampling_unit": "unstratified_whole_plot_with_all_images_and_all_seeds",
        "shared_resamples_for_all_contrasts": True,
        "estimand": "mean_seed_image_weighted_metric_difference",
        "interval_method": "paired_plot_percentile_95",
        "conditional_on_fitted_models_and_fixed_policies": True,
        "selection_adjusted": False,
        "contrasts": contrasts,
    }


def representation_summary(hidden: Any, producer_ids: Any, tcp_hat: Any = None) -> dict[str, Any]:
    """Summarize caller-supplied representations by producing expert checkpoint."""
    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ConfidNetMetricError("hidden activations must be a finite nonempty matrix")
    producers = _ids(producer_ids, len(values), "producer_ids")
    estimates = None if tcp_hat is None else _confidence(tcp_hat, len(values))
    result = {}
    for producer in np.unique(producers):
        mask = producers == producer
        block = values[mask]
        entry = {
            "row_count": len(block),
            "dimension": block.shape[1],
            "activation_quantiles": _quantiles(block.reshape(-1)),
            "row_l2_norm_quantiles": _quantiles(np.linalg.norm(block, axis=1)),
            "coordinate_mean": block.mean(axis=0).tolist(),
            "coordinate_standard_deviation": block.std(axis=0).tolist(),
            "fraction_exactly_zero": float(np.mean(block == 0)),
        }
        if estimates is not None:
            entry["predicted_tcp_quantiles"] = _quantiles(estimates[mask])
        result[str(producer)] = entry
    return result


def score_study(
    train_by_seed: Mapping[int, Mapping[str, Any]],
    validation_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    n_resamples: int = 2000,
    random_seed: int = 20260826,
) -> dict[str, Any]:
    """Score supplied sealed study predictions without any data discovery."""
    seeds = _validate_seed_rows(train_by_seed)
    if _validate_seed_rows(validation_by_seed) != seeds:
        raise ConfidNetMetricError("training and validation must have the same seeds")
    train = _prepare(train_by_seed[seeds[0]])
    validation = _prepare(validation_by_seed[seeds[0]])
    if set(train["plots"]) & set(validation["plots"]):
        raise ConfidNetMetricError("training and validation plots must be disjoint")
    if set(train["rows"]) & set(validation["rows"]):
        raise ConfidNetMetricError("training and validation rows must be disjoint")
    result = {
        "study": "oof_supervised_confidnet_numeric25_vs_numeric28",
        "seeds": [int(seed) for seed in seeds],
        "confidence_prediction_convention": "stored_native_float32_predictions_are_authoritative",
        "router_training_confidence_status": "in_sample_for_confidnet_oof_for_experts",
        "validation_status": "reused_development_validation_exploratory",
        "policies": list(POLICIES),
        "threshold_rule": "strict_positive_difference_on_geo_fusion_disagreement_ties_keep_fusion",
        "splits": {},
    }
    for split, source in (("train", train_by_seed), ("validation", validation_by_seed)):
        per_seed = {str(seed): score_seed(source[seed]) for seed in seeds}
        result["splits"][split] = {
            "per_seed": per_seed,
            "mean_seed": _mean_tree(list(per_seed.values())),
            "pooled": score_seed(_pooled_payload(source, seeds)),
            "mean_seed_missing_metric_rule": "null_if_any_seed_metric_is_undefined",
        }
    gaps = {}
    for seed in seeds:
        gaps[str(seed)] = {}
        for expert in EXPERTS:
            fitting = result["splits"]["train"]["per_seed"][str(seed)]["confidence"][expert]
            held_out = result["splits"]["validation"]["per_seed"][str(seed)]["confidence"][expert]
            fitting_mean = fitting["mean_true_tcp"]
            val = _prepare(validation_by_seed[seed])
            targets = val["probabilities"][expert][np.arange(len(val["labels"])), val["labels"]]
            held_out["training_mean_predictor_tcp_mse"] = float(np.mean((fitting_mean - targets) ** 2))
            gaps[str(seed)][expert] = {
                "tcp_mse_validation_minus_train": held_out["tcp_regression"]["mse"] - fitting["tcp_regression"]["mse"],
                "failure_ap_validation_minus_train": (
                    held_out["failure_detection"]["predicted_tcp"]["average_precision"]
                    - fitting["failure_detection"]["predicted_tcp"]["average_precision"]
                    if held_out["failure_detection"]["predicted_tcp"]["average_precision"] is not None
                    and fitting["failure_detection"]["predicted_tcp"]["average_precision"] is not None
                    else None
                ),
            }
    # Additions above are per-seed; mean summaries are reconstructed consistently.
    result["splits"]["validation"]["mean_seed"] = _mean_tree(list(result["splits"]["validation"]["per_seed"].values()))
    result["confidence_generalization_gaps"] = {"per_seed": gaps, "mean_seed": _mean_tree(list(gaps.values()))}
    result["validation_bootstrap"] = paired_plot_bootstrap(
        validation_by_seed, n_resamples=n_resamples, random_seed=random_seed
    )
    return result


def format_report(report: Mapping[str, Any]) -> str:
    """Render compact human-readable results; the JSON retains all diagnostics."""
    validation = report["splits"]["validation"]
    lines = [
        "# OOF-supervised ConfidNet and numerical-router results", "",
        "Exploratory evaluation on reused development-validation. Expert outputs for confidence fitting are OOF; router-training confidence predictions are in-sample for ConfidNet.", "",
        "## Validation policies", "",
        "| Policy | Mean accuracy (%) | Mean weighted F1 (%) | Mean actions | Mean rescues | Mean harms | Mean net | Accuracy gain vs fusion (pp), 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in POLICIES:
        metric = validation["mean_seed"]["policies"][name]
        interval = report["validation_bootstrap"]["contrasts"][f"{name}_minus_fusion"]["accuracy"]
        lines.append(
            f"| {name} | {100 * metric['accuracy']:.4f} | {100 * metric['weighted_f1']:.4f} | {metric['actions']:.2f} | {metric['rescues']:.2f} | {metric['harms']:.2f} | {metric['net']:.2f} | {100 * interval['difference']:+.4f} [{100 * interval['ci95_lower']:+.4f}, {100 * interval['ci95_upper']:+.4f}] |"
        )
    comparison = report["validation_bootstrap"]["contrasts"]["rf_augmented_minus_rf_baseline"]["accuracy"]
    lines += ["", f"Augmented minus matched numerical baseline: **{100 * comparison['difference']:+.4f} percentage points**, paired plot 95% CI [{100 * comparison['ci95_lower']:+.4f}, {100 * comparison['ci95_upper']:+.4f}].", "", "## Validation confidence quality", "", "| Expert | TCP MSE | Predicted-TCP failure AP | MCP failure AP | Entropy failure AP | Predicted-TCP failure AUC |", "|---|---:|---:|---:|---:|---:|"]
    def shown(value: Any) -> str:
        return "undefined" if value is None else f"{value:.6f}"
    for expert in EXPERTS:
        metric = validation["mean_seed"]["confidence"][expert]
        detection = metric["failure_detection"]
        lines.append(f"| {expert} | {metric['tcp_regression']['mse']:.6f} | {shown(detection['predicted_tcp']['average_precision'])} | {shown(detection['mcp']['average_precision'])} | {shown(detection['entropy']['average_precision'])} | {shown(detection['predicted_tcp']['roc_auc'])} |")
    lines += ["", "## Per-seed validation policies", "", "| Seed | Policy | Accuracy (%) | Weighted F1 (%) | Actions | Rescues | Harms | Wrong-to-wrong | Net | Plots affected |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, seed_metrics in validation["per_seed"].items():
        for name in POLICIES:
            metric = seed_metrics["policies"][name]
            lines.append(f"| {seed} | {name} | {100 * metric['accuracy']:.4f} | {100 * metric['weighted_f1']:.4f} | {metric['actions']} | {metric['rescues']} | {metric['harms']} | {metric['neutral_wrong']} | {metric['net']} | {metric['plots_affected']} |")
    lines += ["", "## Interpretation limits", "", "- Native float32 confidence predictions are authoritative; ties are preserved.", "- Every policy uses a fixed strict-positive rule on disagreements; no threshold was selected here.", "- Bootstrap resamples whole plots with all images and all seeds together; intervals condition on the fitted models and do not adjust for repeated validation use.", "- Both the confidence recipe and supervision differ from the initial study, so this comparison does not isolate an OOF-supervision effect.", "- Hidden representations come from separately fitted producers; matching dimensions alone do not establish representation compatibility.", ""]
    return "\n".join(lines)

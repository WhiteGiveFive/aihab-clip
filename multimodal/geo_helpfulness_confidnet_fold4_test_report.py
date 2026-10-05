"""Pure cleaned-test scoring of the frozen single-fold ConfidNet experiment.

Model choices use no labels. Callers seal those choices before joining labels.
This module never fits models, searches thresholds, or discovers artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet_oof_report import (
    ConfidNetMetricError,
    _ids,
    _mean_tree,
    _router_probabilities,
    _weighted_f1,
    score_seed,
)
from multimodal.geo_helpfulness_confidnet_report import (
    _confidence,
    _labels,
    _probabilities,
)
from multimodal.geo_helpfulness_router_heuristics import validate_threshold_specification


SEEDS = (1, 2, 3, 4)
EXPERT_ORDER = ("raw", "geo", "image")
EXPERT_DISPLAY = ("fusion", "geo", "image")
ARMS = ("baseline", "augmented")
RELAXED_NAMES = tuple(
    f"{arm}__plots_{plots}__harm_0p{numerator}"
    for arm in ARMS for plots in (20, 10, 0) for numerator in (5, 7, 9)
)


def _prepared_predictions(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only prediction inputs; never access any label key."""
    probabilities = {
        expert: _probabilities(payload[f"{expert}_logits"])
        for expert in EXPERT_ORDER
    }
    shape = probabilities["raw"].shape
    if any(value.shape != shape for value in probabilities.values()):
        raise ConfidNetMetricError("all experts must have matching logit shapes")
    size = shape[0]
    return {
        "probabilities": probabilities,
        "predictions": np.column_stack([
            np.asarray(payload[f"{expert}_logits"]).argmax(axis=1)
            for expert in EXPERT_ORDER
        ]),
        "estimates": np.column_stack([
            _confidence(payload[f"{expert}_tcp_hat"], size)
            for expert in EXPERT_ORDER
        ]),
        "router_q": {
            arm: _router_probabilities(payload[f"router_{arm}_q"], size)
            for arm in ARMS
        },
        "rows": _ids(payload["row_uids"], size, "row_uids", unique=True),
        "plots": _ids(payload["plot_ids"], size, "plot_ids"),
    }


def _thresholds(validation_selections: Mapping[str, Any]) -> dict[str, float | None]:
    if set(validation_selections) != set(RELAXED_NAMES):
        raise ConfidNetMetricError("exactly the 18 saved validation selections are required")
    result = {}
    for name in RELAXED_NAMES:
        selected = validation_selections[name]
        arm = name.split("__", 1)[0]
        if selected.get("arm") != arm:
            raise ConfidNetMetricError("validation selection arm does not match its name")
        result[name] = validate_threshold_specification(selected["threshold_specification"])
    return result


def policy_choices(
    payload: Mapping[str, Any], validation_selections: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Label-blind expert IDs: 0=fusion, 1=geo, 2=image.

The three-expert comparator selects the highest stored estimated TCP on every
row, even when expert top-1 predictions agree. Exact ties favor fusion, then
geo, then image. Pair comparators and routers retain their original rule of
acting only on geo/fusion top-1 disagreement; that also fixes their top-3 output.
"""
    data = _prepared_predictions(payload)
    size = len(data["rows"])
    disagreement = data["predictions"][:, 0] != data["predictions"][:, 1]
    choices = {
        "fusion": np.zeros(size, dtype=np.int8),
        "image": np.full(size, 2, dtype=np.int8),
        "geo": np.ones(size, dtype=np.int8),
    }
    utility = {}
    for arm in ARMS:
        q = data["router_q"][arm]
        utility[arm] = q[:, 0] - q[:, 1]
        choices[f"{arm}_zero"] = (disagreement & (utility[arm] > 0)).astype(np.int8)
    choices["tcp_max_three"] = data["estimates"].argmax(axis=1).astype(np.int8)
    choices["tcp_geo_fusion"] = (
        disagreement & (data["estimates"][:, 1] > data["estimates"][:, 0])
    ).astype(np.int8)
    probability = data["probabilities"]
    choices["mcp_geo_fusion"] = (
        disagreement & (probability["geo"].max(axis=1) > probability["raw"].max(axis=1))
    ).astype(np.int8)
    for name, threshold in _thresholds(validation_selections).items():
        choices[name] = (
            np.zeros(size, dtype=np.int8) if threshold is None else
            (disagreement & (utility[name.split("__", 1)[0]] > threshold)).astype(np.int8)
        )
    return choices


def scored_rows_and_choices(
    by_seed: Mapping[int, Mapping[str, Any]],
    validation_selections: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Join caller-supplied labels to choices in canonical row/seed order."""
    if sorted(by_seed) != list(SEEDS) or any(isinstance(seed, bool) for seed in by_seed):
        raise ConfidNetMetricError("cleaned-test scoring requires seeds 1,2,3,4")
    blocks, reference, class_count = [], None, None
    policy_names = None
    for seed in SEEDS:
        payload = by_seed[seed]
        data = _prepared_predictions(payload)
        labels = _labels(payload["labels"], data["probabilities"]["raw"])
        identities = sorted(zip(data["rows"], data["plots"], labels.tolist()))
        current_classes = data["probabilities"]["raw"].shape[1]
        if reference is None:
            reference, class_count = identities, current_classes
        elif identities != reference or class_count != current_classes:
            raise ConfidNetMetricError("every seed must contain identical row, plot, label and class identities")
        choices = policy_choices(payload, validation_selections)
        policy_names = list(choices)
        block = pd.DataFrame({
            "row_uid": data["rows"], "plot_idx": data["plots"],
            "training_seed": seed, "label_id_dense": labels,
        })
        for index, expert in enumerate(EXPERT_ORDER):
            block[f"{expert}_pred"] = data["predictions"][:, index]
            ranked = np.argsort(-np.asarray(payload[f"{expert}_logits"]), axis=1, kind="stable")
            block[f"{expert}_top3_correct"] = np.any(ranked[:, :3] == labels[:, None], axis=1)
            block[f"{expert}_tcp_hat"] = data["estimates"][:, index]
        for arm in ARMS:
            q = data["router_q"][arm]
            block[f"{arm}_score"] = q[:, 0] - q[:, 1]
        for name, choice in choices.items():
            block[f"choice__{name}"] = choice
        blocks.append(block)
    rows = pd.concat(blocks, ignore_index=True).sort_values(
        ["row_uid", "training_seed"], kind="stable"
    ).reset_index(drop=True)
    combined = {name: rows.pop(f"choice__{name}").to_numpy() for name in policy_names}
    rows.attrs["class_count"] = class_count
    return rows, combined


def _selected_arrays(rows: pd.DataFrame, choice: Any) -> tuple[np.ndarray, np.ndarray]:
    choice = np.asarray(choice)
    if choice.shape != (len(rows),) or choice.dtype.kind not in "iu" or np.any((choice < 0) | (choice > 2)):
        raise ConfidNetMetricError("choices must be matching integer expert IDs in 0..2")
    positions = np.arange(len(rows))
    predicted = rows[[f"{expert}_pred" for expert in EXPERT_ORDER]].to_numpy()[positions, choice]
    top3 = rows[[f"{expert}_top3_correct" for expert in EXPERT_ORDER]].to_numpy()[positions, choice]
    return predicted, top3


def choices_frame(rows: pd.DataFrame, choices: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Long table for explicit selected experts, predictions and action audits."""
    blocks = []
    for name, choice in choices.items():
        predicted, top3 = _selected_arrays(rows, choice)
        block = rows[["row_uid", "plot_idx", "training_seed"]].copy()
        block["policy"] = name
        block["selected_expert_id"] = choice
        block["selected_expert"] = np.asarray(EXPERT_DISPLAY)[choice]
        block["prediction"] = predicted
        block["prediction_changed_vs_fusion"] = predicted != rows.raw_pred.to_numpy()
        block["correct"] = predicted == rows.label_id_dense.to_numpy()
        block["top3_correct"] = top3
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def policy_metrics(rows: pd.DataFrame, choice: np.ndarray) -> dict[str, Any]:
    """Action counts refer to changed top-1 predictions, not just expert switches."""
    predicted, top3 = _selected_arrays(rows, choice)
    labels = rows.label_id_dense.to_numpy(dtype=np.int64)
    raw = rows.raw_pred.to_numpy(dtype=np.int64)
    action, raw_ok, correct = predicted != raw, raw == labels, predicted == labels
    classes = int(max(labels.max(), predicted.max(), raw.max()) + 1)
    all_geo_rescues = ~raw_ok & rows.geo_pred.eq(labels).to_numpy()
    all_three_rescues = ~raw_ok & (rows.geo_pred.eq(labels) | rows.image_pred.eq(labels)).to_numpy()

    def counts(mask: np.ndarray) -> dict[str, Any]:
        active = action & mask
        rescue = int(np.sum(active & ~raw_ok & correct))
        harm = int(np.sum(active & raw_ok & ~correct))
        wrong = int(np.sum(active & ~raw_ok & ~correct))
        total = int(mask.sum())
        if rescue + harm + wrong != int(active.sum()):
            raise ConfidNetMetricError("action accounting does not reconcile")
        f1 = _weighted_f1(labels[mask], predicted[mask], classes)
        return {
            "row_count": total, "plot_count": int(rows.loc[mask, "plot_idx"].nunique()),
            "correct_count": int(correct[mask].sum()), "accuracy": float(correct[mask].mean()),
            "weighted_f1": f1, "top3_correct_count": int(top3[mask].sum()),
            "top3_accuracy": float(top3[mask].mean()),
            "actions": int(active.sum()), "rescues": rescue, "harms": harm,
            "neutral_wrong": wrong, "wrong_to_wrong": wrong, "net": rescue - harm,
            "coverage": float(active.sum() / total),
            "plots_affected": int(rows.loc[active, "plot_idx"].nunique()),
            "accuracy_gain_vs_fusion": float((rescue - harm) / total),
            "weighted_f1_gain_vs_fusion": f1 - _weighted_f1(labels[mask], raw[mask], classes),
            "selected_expert_counts": {
                name: int(np.sum(mask & (choice == index)))
                for index, name in enumerate(EXPERT_DISPLAY)
            },
            "model_switches": int(np.sum(mask & (choice != 0))),
            "model_switches_with_unchanged_prediction": int(np.sum(mask & (choice != 0) & ~action)),
            "rescue_precision": rescue / int(active.sum()) if active.any() else None,
            "available_geo_rescues": int(np.sum(mask & all_geo_rescues)),
            "available_three_expert_rescues": int(np.sum(mask & all_three_rescues)),
        }

    per_seed = {str(seed): counts(rows.training_seed.eq(seed).to_numpy()) for seed in SEEDS}
    contribution = correct.astype(int) - raw_ok.astype(int)
    plot_net = pd.Series(contribution).groupby(rows.plot_idx).sum().sort_values(ascending=False, kind="stable")
    pooled = counts(np.ones(len(rows), dtype=bool))
    largest = int(max(0, plot_net.iloc[0]))
    return {
        "per_seed": per_seed, "mean_seed": _mean_tree(list(per_seed.values())), "pooled": pooled,
        "largest_positive_plot": str(plot_net.index[0]) if largest else None,
        "largest_plot_net": largest,
        "net_without_largest_positive_plot": pooled["net"] - largest,
    }


def paired_plot_bootstrap(
    rows: pd.DataFrame, choices: Mapping[str, np.ndarray], *,
    n_resamples: int = 2000, random_seed: int = 20260826,
) -> dict[str, Any]:
    """Shared whole-plot draws, preserving all images and four seed realizations."""
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ConfidNetMetricError("n_resamples must be a positive integer")
    names, plots = list(choices), np.unique(rows.plot_idx)
    inverse = np.searchsorted(plots, rows.plot_idx.to_numpy())
    labels = rows.label_id_dense.to_numpy(dtype=np.int64)
    predictions = {name: _selected_arrays(rows, choice) for name, choice in choices.items()}
    classes = int(max(labels.max(), max(value[0].max() for value in predictions.values())) + 1)
    unique, lookup, indices = [], {}, []
    for name in names:
        predicted, top3 = predictions[name]
        key = (predicted.tobytes(), top3.tobytes())
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append((predicted, top3))
        indices.append(lookup[key])
    # row count, correct, top-3 correct, action, rescue, harm, wrong-to-wrong,
    # then per-class truth support, predicted support and true positives.
    groups = np.zeros((len(plots), len(SEEDS), len(unique), 7 + 3 * classes))
    raw = rows.raw_pred.to_numpy(dtype=np.int64)
    for seed_index, seed in enumerate(SEEDS):
        mask = rows.training_seed.eq(seed).to_numpy()
        for policy_index, (predicted, top3) in enumerate(unique):
            correct, raw_ok, action = predicted == labels, raw == labels, predicted != raw
            columns = [np.ones(len(rows)), correct, top3, action,
                       action & ~raw_ok & correct, action & raw_ok & ~correct,
                       action & ~raw_ok & ~correct]
            columns += [labels == c for c in range(classes)]
            columns += [predicted == c for c in range(classes)]
            columns += [(labels == c) & correct for c in range(classes)]
            for column, values in enumerate(columns):
                groups[:, seed_index, policy_index, column] = np.bincount(
                    inverse[mask], weights=values[mask], minlength=len(plots)
                )

    metric_names = ("accuracy", "top3_accuracy", "coverage", "rescue_rate", "harm_rate", "wrong_to_wrong_rate", "weighted_f1")

    def metrics(counts: np.ndarray) -> np.ndarray:
        truth = counts[..., 7:7 + classes]
        predicted = counts[..., 7 + classes:7 + 2 * classes]
        tp = counts[..., 7 + 2 * classes:]
        denominator = truth + predicted
        f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
        values = np.concatenate([
            counts[..., 1:7] / counts[..., :1],
            ((truth * f1).sum(-1) / counts[..., 0])[..., None],
        ], axis=-1).mean(axis=0)
        return values[indices]

    observed = metrics(groups.sum(axis=0))
    draws = np.empty((n_resamples, len(names), len(metric_names)))
    rng = np.random.default_rng(random_seed)
    for index in range(n_resamples):
        sampled = rng.integers(0, len(plots), size=len(plots))
        draws[index] = metrics(groups[sampled].sum(axis=0))
    pairs = [(name, "fusion") for name in names if name != "fusion"]
    pairs += [("augmented_zero", "baseline_zero"),
              ("tcp_max_three", "tcp_geo_fusion"), ("tcp_max_three", "mcp_geo_fusion"),
              ("tcp_max_three", "baseline_zero"), ("tcp_max_three", "augmented_zero")]
    for name in RELAXED_NAMES:
        if name in names:
            pairs.append((name, name.split("__")[0] + "_zero"))
            pairs.append(("tcp_max_three", name))
            if name.startswith("augmented__"):
                pairs.append((name, name.replace("augmented__", "baseline__", 1)))
    contrasts = {}
    for treatment, reference in pairs:
        left, right = names.index(treatment), names.index(reference)
        low, high = np.quantile(draws[:, left] - draws[:, right], [.025, .975], axis=0)
        contrasts[f"{treatment}_minus_{reference}"] = {
            metric: {"difference": float(observed[left, index] - observed[right, index]),
                     "ci95_lower": float(low[index]), "ci95_upper": float(high[index])}
            for index, metric in enumerate(metric_names)
        }
    return {
        "n_resamples": int(n_resamples), "random_seed": int(random_seed),
        "plot_count": len(plots), "seed_count": len(SEEDS), "policy_order": names,
        "resampling_unit": "unstratified_whole_plot_with_all_images_and_all_seeds",
        "shared_resamples_for_all_contrasts": True,
        "estimand": "mean_seed_image_weighted_metric_difference",
        "conditional_on_fitted_models_and_development_validation_thresholds": True,
        "thresholds_reselected": False, "selection_adjusted": False,
        "contrasts": contrasts,
    }


def build_report(
    test_by_seed: Mapping[int, Mapping[str, Any]],
    validation_selections: Mapping[str, Any], *,
    n_resamples: int = 2000, random_seed: int = 20260826,
) -> dict[str, Any]:
    """Evaluate existing fits and every transferred threshold without tuning."""
    rows, choices = scored_rows_and_choices(test_by_seed, validation_selections)
    policies = {name: policy_metrics(rows, choice) for name, choice in choices.items()}
    diagnostics = {str(seed): score_seed(test_by_seed[seed]) for seed in SEEDS}
    confidence = {seed: value["confidence"] for seed, value in diagnostics.items()}
    routers = {seed: value["routers"] for seed, value in diagnostics.items()}
    oracle_per_seed = {}
    for seed in SEEDS:
        subset = rows.loc[rows.training_seed.eq(seed)]
        labels = subset.label_id_dense.to_numpy()
        raw_ok = subset.raw_pred.to_numpy() == labels
        geo_ok = subset.geo_pred.to_numpy() == labels
        image_ok = subset.image_pred.to_numpy() == labels
        oracle_per_seed[str(seed)] = {
            "row_count": len(subset), "plot_count": int(subset.plot_idx.nunique()),
            "fusion_error_count": int((~raw_ok).sum()),
            "geo_fusion": {
                "correct_count": int((raw_ok | geo_ok).sum()),
                "accuracy": float((raw_ok | geo_ok).mean()),
                "available_rescues": int((~raw_ok & geo_ok).sum()),
            },
            "three_expert": {
                "correct_count": int((raw_ok | geo_ok | image_ok).sum()),
                "accuracy": float((raw_ok | geo_ok | image_ok).mean()),
                "available_rescues": int((~raw_ok & (geo_ok | image_ok)).sum()),
            },
        }
    return {
        "study": "fixed_expert_fold4_confidnet_cleaned_test",
        "split": "cleaned_test", "training_seeds": list(SEEDS),
        "image_count": len(rows) // len(SEEDS), "plot_count": int(rows.plot_idx.nunique()),
        "pooled_seed_row_count": len(rows),
        "threshold_selection_split": "development_validation", "thresholds_reselected_on_test": False,
        "saved_validation_selections": deepcopy(dict(validation_selections)),
        "selected_expert_id_order": list(EXPERT_DISPLAY),
        "tcp_max_three_tie_priority": list(EXPERT_DISPLAY),
        "tcp_max_three_selection_scope": "all_rows_including_agreement",
        "pair_comparator_and_router_selection_scope": "geo_fusion_top1_disagreement_only",
        "action_definition": "selected_top1_prediction_differs_from_fusion_top1",
        "top3_definition": "chosen_expert_top3_logits_stable_lower_class_id_first_on_ties",
        "confidence_estimate_convention": "stored_native_estimates_used_without_reevaluation",
        "confidence_target_convention": "temperature_one_softmax_of_stored_logits_computed_in_float64",
        "policies": policies,
        "confidence": {"per_seed": confidence, "mean_seed": _mean_tree(list(confidence.values()))},
        "routers": {"per_seed": routers, "mean_seed": _mean_tree(list(routers.values()))},
        "oracle": {
            "label_dependent_reference_only": True,
            "per_seed": oracle_per_seed, "mean_seed": _mean_tree(list(oracle_per_seed.values())),
        },
        "bootstrap": paired_plot_bootstrap(
            rows, choices, n_resamples=n_resamples, random_seed=random_seed
        ),
    }


def format_report(report: Mapping[str, Any]) -> str:
    """Render all frozen policies, direct comparison and confidence diagnostics."""
    lines = [
        "# Fold-4 ConfidNet and router application to cleaned test", "",
        f"{report['image_count']} images, {report['plot_count']} plots, four fitted seed realizations.",
        "The experts, ConfidNets, numerical preprocessing, routers and development-validation thresholds are frozen. No test labels are used for fitting or policy selection.",
        "The direct comparator chooses the expert with highest stored estimated TCP among fusion, geo and image; exact ties favor fusion, then geo, then image.",
        "Top-3 uses the selected expert's logits. Actions count changed top-1 predictions; model switches with agreeing predictions are counted separately.", "",
        "## Frozen policy performance", "",
        "Accuracy, weighted F1 and top-3 are arithmetic means over seeds. Counts pool the four seed realizations.", "",
        "| Policy | Accuracy (%) | Weighted F1 (%) | Top-3 (%) | Actions | Rescues | Harms | Wrong-to-wrong | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in report["policies"].items():
        mean, pooled = entry["mean_seed"], entry["pooled"]
        lines.append(f"| {name} | {100 * mean['accuracy']:.6f} | {100 * mean['weighted_f1']:.6f} | {100 * mean['top3_accuracy']:.6f} | {pooled['actions']} | {pooled['rescues']} | {pooled['harms']} | {pooled['wrong_to_wrong']} | {pooled['net']} |")
    lines += ["", "## Transferred thresholds", "", "| Policy | Threshold | Selected as feasible on development-validation |", "|---|---:|---|"]
    for name, entry in report["saved_validation_selections"].items():
        value = validate_threshold_specification(entry["threshold_specification"])
        display = "never intervene" if value is None else f"{value:.17g}"
        lines.append(f"| {name} | {display} | {entry.get('selected_feasible')} |")
    lines += ["", "## Direct three-expert comparator selections", "", "| Seed | Fusion selections | Geo selections | Image selections | Model switches | Switches with unchanged top-1 |", "|---|---:|---:|---:|---:|---:|"]
    for seed, entry in report["policies"]["tcp_max_three"]["per_seed"].items():
        selected = entry["selected_expert_counts"]
        lines.append(f"| {seed} | {selected['fusion']} | {selected['geo']} | {selected['image']} | {entry['model_switches']} | {entry['model_switches_with_unchanged_prediction']} |")
    oracle = report["oracle"]["mean_seed"]
    lines += ["", f"Matched geo/fusion oracle accuracy: {100 * oracle['geo_fusion']['accuracy']:.6f}%. Matched three-expert oracle: {100 * oracle['three_expert']['accuracy']:.6f}%. These are label-dependent accuracy ceilings, not deployable policies.", "", "## Confidence diagnostics", "", "| Expert | TCP MSE | TCP failure AP | MCP failure AP | TCP failure AUC | MCP failure AUC |", "|---|---:|---:|---:|---:|---:|"]
    def shown(value: Any) -> str:
        return "undefined" if value is None else f"{value:.6f}"
    for expert in EXPERT_ORDER:
        entry = report["confidence"]["mean_seed"][expert]
        failure = entry["failure_detection"]
        lines.append(f"| {expert} | {entry['tcp_regression']['mse']:.6f} | {shown(failure['predicted_tcp']['average_precision'])} | {shown(failure['mcp']['average_precision'])} | {shown(failure['predicted_tcp']['roc_auc'])} | {shown(failure['mcp']['roc_auc'])} |")
    lines += ["", "## Paired accuracy contrasts", "", "Shared whole-plot bootstrap draws retain all images and all seeds. Intervals condition on these fitted models and transferred thresholds; they do not adjust for previous model/policy exploration.", "", "| Contrast | Difference (pp) | 95% interval (pp) |", "|---|---:|---:|"]
    for name, entry in report["bootstrap"]["contrasts"].items():
        metric = entry["accuracy"]
        lines.append(f"| {name} | {100 * metric['difference']:+.6f} | [{100 * metric['ci95_lower']:+.6f}, {100 * metric['ci95_upper']:+.6f}] |")
    lines += ["", "## Per-seed policies", "", "| Policy | Seed | Accuracy (%) | Weighted F1 (%) | Top-3 (%) | Actions | Rescues | Harms | Net |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in report["policies"].items():
        for seed, metric in entry["per_seed"].items():
            lines.append(f"| {name} | {seed} | {100 * metric['accuracy']:.6f} | {100 * metric['weighted_f1']:.6f} | {100 * metric['top3_accuracy']:.6f} | {metric['actions']} | {metric['rescues']} | {metric['harms']} | {metric['net']} |")
    return "\n".join(lines) + "\n"

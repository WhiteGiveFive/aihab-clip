"""Pure scoring for the fixed-expert, fold-4 ConfidNet experiment.

Inputs are sealed predictions with labels joined by the caller. This module has
no artifact discovery or fitting. Threshold selection intentionally uses the
same development-validation rows as reporting, with the established RF grid.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet_oof_report import (
    EXPERTS,
    STATES,
    ConfidNetMetricError,
    _mean_tree,
    _prepare,
    _validate_seed_rows,
    _weighted_f1,
    score_study,
)
from multimodal.geo_helpfulness_relaxed_rf_selection import select_common_threshold
from multimodal.geo_helpfulness_router_heuristics import apply_heuristic_threshold


SEEDS = (1, 2, 3, 4)
ARMS = ("baseline", "augmented")
MINIMUM_PLOTS = (20, 10, 0)
HARM_NUMERATORS = (5, 7, 9)


def validation_rows_and_actions(
    by_seed: Mapping[int, Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Return explicit validation identities, exact selectors and action masks."""
    if _validate_seed_rows(by_seed) != list(SEEDS):
        raise ConfidNetMetricError("fold-4 selection requires seeds 1,2,3,4")
    blocks = []
    for seed in SEEDS:
        data = _prepare(by_seed[seed])
        block = pd.DataFrame({
            "row_uid": data["rows"],
            "plot_idx": data["plots"],
            "training_seed": seed,
            "label_id_dense": data["labels"],
            "raw_pred": data["predictions"]["raw"],
            "geo_pred": data["predictions"]["geo"],
            "target_state": np.asarray(STATES)[data["state"]],
        })
        for arm in ARMS:
            q = data["router_q"][arm]
            block[f"{arm}_score"] = q[:, 0] - q[:, 1]
        blocks.append(block)
    rows = pd.concat(blocks, ignore_index=True).sort_values(
        ["row_uid", "training_seed"], kind="stable"
    ).reset_index(drop=True)
    disagreement = rows.raw_pred.ne(rows.geo_pred).to_numpy()
    actions = {"fusion": np.zeros(len(rows), dtype=bool)}
    selections = {}
    for arm in ARMS:
        values = rows[f"{arm}_score"].to_numpy(dtype=np.float64)
        actions[f"{arm}_zero"] = disagreement & (values > 0)
        inputs = rows.assign(score=values)
        for plots in MINIMUM_PLOTS:
            for numerator in HARM_NUMERATORS:
                name = f"{arm}__plots_{plots}__harm_0p{numerator}"
                selected = select_common_threshold(
                    inputs, minimum_plots=plots, harm_numerator=numerator
                )
                selections[name] = {"arm": arm, **selected}
                actions[name] = apply_heuristic_threshold(
                    values, rows.raw_pred.to_numpy(), rows.geo_pred.to_numpy(),
                    selected["threshold_specification"],
                )
    return rows, selections, actions


def _classification(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    classes = int(max(labels.max(), predictions.max()) + 1)
    return {
        "correct_count": int(np.sum(labels == predictions)),
        "accuracy": float(np.mean(labels == predictions)),
        "weighted_f1": _weighted_f1(labels, predictions, classes),
    }


def policy_metrics(rows: pd.DataFrame, action: np.ndarray) -> dict[str, Any]:
    """Report pooled counts separately from arithmetic means over seeds."""
    action = np.asarray(action)
    if action.dtype != np.dtype(bool) or action.shape != (len(rows),):
        raise ConfidNetMetricError("actions must be a boolean vector matching rows")
    if np.any(action & rows.raw_pred.eq(rows.geo_pred).to_numpy()):
        raise ConfidNetMetricError("actions must be limited to disagreements")
    labels = rows.label_id_dense.to_numpy(dtype=np.int64)
    predicted = np.where(action, rows.geo_pred, rows.raw_pred)
    state = rows.target_state.to_numpy()

    def counts(mask: np.ndarray) -> dict[str, Any]:
        active = action & mask
        rescue = int(np.sum(active & (state == "rescue")))
        harm = int(np.sum(active & (state == "harm")))
        neutral = int(np.sum(active & (state == "both_wrong")))
        if rescue + harm + neutral != int(active.sum()):
            raise ConfidNetMetricError("action counts do not reconcile")
        return {
            "row_count": int(mask.sum()),
            "actions": int(active.sum()), "rescues": rescue, "harms": harm,
            "neutral_wrong": neutral, "net": rescue - harm,
            "plots_affected": int(rows.loc[active, "plot_idx"].nunique()),
            "coverage": float(active.sum() / mask.sum()),
            "accuracy_gain_vs_fusion": float((rescue - harm) / mask.sum()),
            **_classification(labels[mask], predicted[mask]),
        }

    per_seed = {
        str(seed): counts(rows.training_seed.eq(seed).to_numpy()) for seed in SEEDS
    }
    pooled = counts(np.ones(len(rows), dtype=bool))
    contribution = action * ((state == "rescue").astype(int) - (state == "harm").astype(int))
    plot_net = pd.Series(contribution).groupby(rows.plot_idx).sum().sort_values(
        ascending=False, kind="stable"
    )
    largest = int(max(0, plot_net.iloc[0]))
    return {
        "per_seed": per_seed,
        "pooled": pooled,
        "mean_seed": _mean_tree(list(per_seed.values())),
        "largest_positive_plot": str(plot_net.index[0]) if largest > 0 else None,
        "largest_plot_net": largest,
        "net_without_largest_positive_plot": pooled["net"] - largest,
    }


def relaxed_bootstrap(
    rows: pd.DataFrame,
    actions: Mapping[str, np.ndarray],
    *,
    n_resamples: int = 2000,
    random_seed: int = 20260826,
) -> dict[str, Any]:
    """Paired whole-plot intervals, holding selected thresholds fixed."""
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ConfidNetMetricError("n_resamples must be a positive integer")
    names = list(actions)
    plots, inverse = np.unique(rows.plot_idx, return_inverse=True)
    labels = rows.label_id_dense.to_numpy(dtype=np.int64)
    class_count = int(max(labels.max(), rows.raw_pred.max(), rows.geo_pred.max()) + 1)
    # Identical selected action masks share sufficient statistics and intervals.
    unique, lookup, indices = [], {}, []
    for name in names:
        mask = np.asarray(actions[name])
        if mask.shape != (len(rows),) or mask.dtype != np.dtype(bool):
            raise ConfidNetMetricError("actions must be boolean vectors matching rows")
        key = mask.tobytes()
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(mask)
        indices.append(lookup[key])
    groups = np.zeros((len(plots), len(SEEDS), len(unique), 2 + 3 * class_count))
    for seed_index, seed in enumerate(SEEDS):
        mask = rows.training_seed.eq(seed).to_numpy()
        for policy_index, action in enumerate(unique):
            prediction = np.where(action, rows.geo_pred, rows.raw_pred)
            columns = [np.ones(len(rows)), prediction == labels]
            columns += [labels == c for c in range(class_count)]
            columns += [prediction == c for c in range(class_count)]
            columns += [(prediction == c) & (labels == c) for c in range(class_count)]
            for column, values in enumerate(columns):
                groups[:, seed_index, policy_index, column] = np.bincount(
                    inverse[mask], weights=values[mask], minlength=len(plots)
                )

    def metrics(counts: np.ndarray) -> np.ndarray:
        truth = counts[..., 2:2 + class_count]
        predicted = counts[..., 2 + class_count:2 + 2 * class_count]
        tp = counts[..., 2 + 2 * class_count:]
        denominator = truth + predicted
        f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
        result = np.stack([
            counts[..., 1] / counts[..., 0],
            (truth * f1).sum(-1) / counts[..., 0],
        ], axis=-1).mean(0)
        return result[indices]

    observed = metrics(groups.sum(0))
    draws = np.empty((n_resamples, len(names), 2))
    rng = np.random.default_rng(random_seed)
    for index in range(n_resamples):
        sample = rng.integers(0, len(plots), size=len(plots))
        draws[index] = metrics(groups[sample].sum(0))
    pairs = [(name, "fusion") for name in names if name != "fusion"]
    pairs.append(("augmented_zero", "baseline_zero"))
    for name in names:
        if "__" in name:
            pairs.append((name, name.split("__")[0] + "_zero"))
        if name.startswith("augmented__"):
            pairs.append((name, name.replace("augmented__", "baseline__", 1)))
    contrasts = {}
    for treatment, reference in pairs:
        a, b = names.index(treatment), names.index(reference)
        delta = draws[:, a] - draws[:, b]
        low, high = np.quantile(delta, [.025, .975], axis=0)
        contrasts[f"{treatment}_minus_{reference}"] = {
            metric: {
                "difference": float(observed[a, k] - observed[b, k]),
                "ci95_lower": float(low[k]), "ci95_upper": float(high[k]),
            }
            for k, metric in enumerate(("accuracy", "weighted_f1"))
        }
    return {
        "n_resamples": int(n_resamples), "random_seed": int(random_seed),
        "plot_count": len(plots), "seed_count": len(SEEDS),
        "policy_order": names,
        "resampling_unit": "unstratified_whole_plot_with_all_images_and_all_seeds",
        "shared_resamples_for_all_contrasts": True,
        "estimand": "mean_seed_image_weighted_metric_difference",
        "conditional_on_fitted_models_and_selected_thresholds": True,
        "selection_adjusted": False,
        "contrasts": contrasts,
    }


def expert_reference(by_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Matched experts and the accuracy ceiling for choosing geo or fusion."""
    seeds = _validate_seed_rows(by_seed)
    per_seed = {}
    for seed in seeds:
        data = _prepare(by_seed[seed])
        labels = data["labels"]
        raw_ok = data["predictions"]["raw"] == labels
        geo_ok = data["predictions"]["geo"] == labels
        rescue = geo_ok & ~raw_ok
        per_seed[str(seed)] = {
            "row_count": len(labels),
            "plot_count": len(np.unique(data["plots"])),
            "experts": {
                expert: _classification(labels, data["predictions"][expert])
                for expert in EXPERTS
            },
            "fusion_error_count": int(np.sum(~raw_ok)),
            "disagreement_count": int(np.sum(data["predictions"]["raw"] != data["predictions"]["geo"])),
            "rescue_available": int(rescue.sum()),
            "harm_available": int(np.sum(raw_ok & ~geo_ok)),
            "oracle_accuracy": float(np.mean(raw_ok | geo_ok)),
            "oracle_accuracy_gain_vs_fusion": float(np.mean(rescue)),
            "oracle_definition": "keep_fusion_except_when_fusion_wrong_and_geo_correct",
        }
    return {"per_seed": per_seed, "mean_seed": _mean_tree(list(per_seed.values()))}


def build_report(
    train_by_seed: Mapping[int, Mapping[str, Any]],
    validation_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    n_resamples: int = 2000,
    random_seed: int = 20260826,
) -> dict[str, Any]:
    """Score frozen fold-4 fits, then apply the unchanged relaxed selector grid."""
    fixed = score_study(
        train_by_seed, validation_by_seed,
        n_resamples=n_resamples, random_seed=random_seed,
    )
    fixed.update({
        "study": "fixed_expert_fold4_confidnet_numeric25_vs_numeric28",
        "router_training_confidence_status": "in_sample_for_confidnet_held_out_from_fixed_experts",
        "expert_producer_status": "same_fold123_experts_for_fold4_and_development_validation",
    })
    rows, selections, actions = validation_rows_and_actions(validation_by_seed)
    policies = {name: policy_metrics(rows, action) for name, action in actions.items()}
    bootstrap = relaxed_bootstrap(
        rows, actions, n_resamples=n_resamples, random_seed=random_seed
    )
    # The two independently assembled reporting paths must agree exactly enough
    # to expose accidental row, seed, action, or bootstrap-unit differences.
    for new, old in (("fusion", "fusion"), ("baseline_zero", "rf_baseline"), ("augmented_zero", "rf_augmented")):
        for key in ("accuracy", "weighted_f1", "actions", "rescues", "harms", "net"):
            if not np.isclose(policies[new]["mean_seed"][key], fixed["splits"]["validation"]["mean_seed"]["policies"][old][key], atol=1e-14, rtol=0):
                raise ConfidNetMetricError("fixed and relaxed policy metrics differ")
    for metric in ("accuracy", "weighted_f1"):
        for key in ("difference", "ci95_lower", "ci95_upper"):
            new = bootstrap["contrasts"]["augmented_zero_minus_baseline_zero"][metric][key]
            old = fixed["validation_bootstrap"]["contrasts"]["rf_augmented_minus_rf_baseline"][metric][key]
            if not np.isclose(new, old, atol=1e-14, rtol=0):
                raise ConfidNetMetricError("fixed and relaxed bootstrap contrasts differ")
    return {
        "study": "fixed_expert_fold4_confidnet_numeric25_vs_numeric28",
        "fixed": fixed,
        "expert_reference": expert_reference(validation_by_seed),
        "relaxed": {
            "selection_split": "development_validation",
            "selection_and_reporting_use_same_rows": True,
            "independent_confirmation": False,
            "minimum_plots_grid": list(MINIMUM_PLOTS),
            "harm_numerator_grid": list(HARM_NUMERATORS),
            "selections": {
                name: {key: value for key, value in selected.items() if key != "candidate_curve"}
                for name, selected in selections.items()
            },
            "candidate_curves": {name: selected["candidate_curve"] for name, selected in selections.items()},
            "policies": policies,
            "bootstrap": bootstrap,
            "fixed_policy_and_bootstrap_parity": True,
        },
    }


def format_report(report: Mapping[str, Any]) -> str:
    """Render matched references, fixed policies, all guardrails, and diagnostics."""
    fixed, relaxed = report["fixed"], report["relaxed"]
    reference = report["expert_reference"]["mean_seed"]
    lines = [
        "# Fixed-expert fold-4 ConfidNet experiment", "",
        "ConfidNet and routers fit on fold 4; the same frozen folds-1–3 experts produce fold-4 and development-validation features.",
        "Router confidence inputs remain in-sample for ConfidNet. Validation thresholds are selected and reported on the same rows; intervals hold thresholds fixed and do not adjust for selection or validation reuse.", "",
        "## Matched validation experts and correction ceiling", "",
        "| Expert | Mean accuracy (%) | Mean weighted F1 (%) |", "|---|---:|---:|",
    ]
    for expert, metric in reference["experts"].items():
        lines.append(f"| {expert} | {100 * metric['accuracy']:.6f} | {100 * metric['weighted_f1']:.6f} |")
    lines += ["", f"Geo/fusion oracle accuracy: {100 * reference['oracle_accuracy']:.6f}%; mean available rescues: {reference['rescue_available']:.2f} per seed.", "", "## Fixed validation policies", "", "| Policy | Mean accuracy (%) | Mean F1 (%) | Mean actions | Mean rescues | Mean harms | Mean net |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metric in fixed["splits"]["validation"]["mean_seed"]["policies"].items():
        lines.append(f"| {name} | {100 * metric['accuracy']:.6f} | {100 * metric['weighted_f1']:.6f} | {metric['actions']:.2f} | {metric['rescues']:.2f} | {metric['harms']:.2f} | {metric['net']:.2f} |")
    lines += ["", "## Relaxed validation policies", "", "Accuracies and F1 are seed means. Counts pool the four seed realizations.", "", "| Policy | Threshold | Mean accuracy (%) | Mean F1 (%) | Actions | Rescues | Harms | Net | Plots |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, metric in relaxed["policies"].items():
        selected = relaxed["selections"].get(name)
        value = selected["threshold_display"] if selected else (None if name == "fusion" else 0.)
        threshold = "never" if value is None else f"{value:.12g}"
        mean, pool = metric["mean_seed"], metric["pooled"]
        lines.append(f"| {name} | {threshold} | {100 * mean['accuracy']:.6f} | {100 * mean['weighted_f1']:.6f} | {pool['actions']} | {pool['rescues']} | {pool['harms']} | {pool['net']} | {pool['plots_affected']} |")
    lines += ["", "## Paired accuracy differences", "", "| Contrast | Difference (pp) | Conditional 95% interval (pp) |", "|---|---:|---:|"]
    for name, entry in relaxed["bootstrap"]["contrasts"].items():
        metric = entry["accuracy"]
        lines.append(f"| {name} | {100 * metric['difference']:+.6f} | [{100 * metric['ci95_lower']:+.6f}, {100 * metric['ci95_upper']:+.6f}] |")
    lines += ["", "## Per-seed relaxed policies", "", "| Policy | Seed | Accuracy (%) | F1 (%) | Actions | Rescues | Harms | Net |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entries in relaxed["policies"].items():
        for seed, metric in entries["per_seed"].items():
            lines.append(f"| {name} | {seed} | {100 * metric['accuracy']:.6f} | {100 * metric['weighted_f1']:.6f} | {metric['actions']} | {metric['rescues']} | {metric['harms']} | {metric['net']} |")
    lines += ["", "## Confidence generalization", "", "| Expert | Train TCP MSE | Validation TCP MSE | Train failure AP | Validation TCP failure AP | Validation MCP failure AP | Validation entropy failure AP |", "|---|---:|---:|---:|---:|---:|---:|"]
    def shown(value: Any) -> str:
        return "undefined" if value is None else f"{value:.6f}"
    for expert in EXPERTS:
        train = fixed["splits"]["train"]["mean_seed"]["confidence"][expert]
        validation = fixed["splits"]["validation"]["mean_seed"]["confidence"][expert]
        failure = validation["failure_detection"]
        lines.append(f"| {expert} | {train['tcp_regression']['mse']:.6f} | {validation['tcp_regression']['mse']:.6f} | {shown(train['failure_detection']['predicted_tcp']['average_precision'])} | {shown(failure['predicted_tcp']['average_precision'])} | {shown(failure['mcp']['average_precision'])} | {shown(failure['entropy']['average_precision'])} |")
    lines += ["", "No cleaned-test evaluation. All grid settings are reported without choosing a winning guardrail. The oracle uses true labels and is an accuracy upper bound, not an inference policy.", ""]
    return "\n".join(lines)

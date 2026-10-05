"""Compare adapted and pretrained reference geometry from supplied predictions.

Historical tables are read only when ``write_results`` is called, after the
orchestrator has authenticated them and entered its declared scoring stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import joint_geometry_threshold_report as standard


PRETRAINED = Path("dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_joint_geometry_thresholds")
ENCODERS = ("pretrained", "adapted")
SCORE_COLUMNS = {"image": "A_image", "geo": "A_geo", "minimum": "score"}
DIAGNOSTIC_SUBSETS = ("all_eligible_disagreements", "geo_positive")
ENCODER_COLORS = {"pretrained": "#517fa4", "adapted": "#00897b"}


def _load_pretrained(policies: dict) -> tuple[dict, dict]:
    frozen = json.loads((PRETRAINED / "frozen_policies.json").read_text())
    if set(frozen) != set(policies):
        raise ValueError("Historical and adapted policy grids differ")
    files = {"validation": "validation_scored.parquet", "test_3378": "test_3378_scored.parquet",
             "test_4200": "test_4200_scored.parquet"}
    cohorts = {cohort: standard._validate(pd.read_parquet(PRETRAINED / filename), cohort, frozen)
               for cohort, filename in files.items()}
    return cohorts, frozen


def _check_matching(adapted: dict, pretrained: dict) -> None:
    identity = ["row_uid", "file", "plot_idx", "training_seed", "canonical_l3_id", "label_id_dense", "label_name",
                "image_pred", "geo_pred", "raw_pred"]
    for cohort in standard.COHORTS:
        columns = identity + (["rf_pred", "rf_acted"] if cohort != "validation" else [])
        if not adapted[cohort][columns].equals(pretrained[cohort][columns]):
            raise ValueError(f"Encoder comparison changes identities, expert predictions or RF: {cohort}")
        for k in (5, 10):
            for rows in (adapted[cohort], pretrained[cohort]):
                needed = [f"k{k}_A_image", f"k{k}_A_geo", f"k{k}_score", f"k{k}_eligible"]
                if not set(needed) <= set(rows):
                    raise ValueError("Both modality scores are required for encoder diagnostics")
                scores = rows[[f"k{k}_A_image", f"k{k}_A_geo"]].to_numpy(dtype=float)
                eligible = rows[f"k{k}_eligible"].to_numpy()
                if not np.isfinite(scores[eligible]).all():
                    raise ValueError("Eligible geometry must have two finite modality scores")
                expected = np.where(eligible, scores.min(axis=1), np.nan)
                if not np.allclose(rows[f"k{k}_score"], expected, equal_nan=True, atol=1e-10, rtol=1e-10):
                    raise ValueError("The threshold score is not the minimum modality score")
            if not np.array_equal(adapted[cohort][f"k{k}_eligible"], pretrained[cohort][f"k{k}_eligible"]):
                raise ValueError("Strict reference support changed between image encoders")
            if not np.allclose(adapted[cohort][f"k{k}_A_geo"], pretrained[cohort][f"k{k}_A_geo"],
                               equal_nan=True, atol=1e-8, rtol=1e-8):
                raise ValueError("Geo geometry changed in the image-encoder comparison")
    for identity_column in ("row_uid", "plot_idx"):
        if set(adapted["validation"][identity_column]) & set(adapted["test_3378"][identity_column]):
            raise ValueError(f"Validation and test overlap on {identity_column}")


def _encoder_tables(adapted_performance: pd.DataFrame, pretrained_performance: pd.DataFrame,
                    adapted_summary: pd.DataFrame, pretrained_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance = pd.concat([pretrained_performance.assign(encoder="pretrained"),
                             adapted_performance.assign(encoder="adapted")], ignore_index=True)
    summary = pd.concat([pretrained_summary.assign(encoder="pretrained"),
                         adapted_summary.assign(encoder="adapted")], ignore_index=True)
    index = summary.set_index(["encoder", "cohort", "model", "metric"])
    records = []
    for (cohort, model), _ in adapted_performance.groupby(["cohort", "model"], sort=False):
        row = {"cohort": cohort, "model": model}
        for metric in standard.METRICS:
            old = index.loc[("pretrained", cohort, model, metric), "mean"]
            new = index.loc[("adapted", cohort, model, metric), "mean"]
            row.update({f"pretrained_{metric}": old, f"adapted_{metric}": new, f"delta_{metric}": new - old})
        records.append(row)
    return performance, summary, pd.DataFrame(records)


def _diagnostics(encoders: dict[str, dict[str, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distributions, auctions = [], []
    for encoder, cohorts in encoders.items():
        for cohort, rows in cohorts.items():
            for seed, part in rows.groupby("training_seed", sort=True):
                truth = part.label_id_dense.to_numpy()
                raw_correct = part.raw_pred.to_numpy() == truth
                geo_correct = part.geo_pred.to_numpy() == truth
                state = np.select([~raw_correct & geo_correct, raw_correct & ~geo_correct],
                                  ["rescue", "harm"], default="both_wrong")
                disagreement = part.raw_pred.ne(part.geo_pred).to_numpy()
                for k in (5, 10):
                    eligible = disagreement & part[f"k{k}_eligible"].to_numpy()
                    for subset in DIAGNOSTIC_SUBSETS:
                        mask = eligible.copy()
                        if subset == "geo_positive":
                            mask &= part[f"k{k}_A_geo"].gt(0).to_numpy()
                        for modality, suffix in SCORE_COLUMNS.items():
                            score = part[f"k{k}_{suffix}"].to_numpy(dtype=float)
                            for outcome in ("rescue", "harm", "both_wrong"):
                                chosen = mask & (state == outcome) & np.isfinite(score)
                                values = score[chosen]
                                quantiles = np.quantile(values, [0.25, 0.5, 0.75]) if values.size else [np.nan] * 3
                                distributions.append({
                                    "encoder": encoder, "cohort": cohort, "training_seed": int(seed), "k": k,
                                    "subset": subset, "modality": modality, "outcome": outcome,
                                    "n_images": int(chosen.sum()), "n_plots": int(part.loc[chosen, "plot_idx"].nunique()),
                                    "q25": quantiles[0], "median": quantiles[1], "q75": quantiles[2],
                                    "fraction_positive": float((values > 0).mean()) if values.size else np.nan,
                                })
                            chosen = mask & np.isin(state, ["rescue", "harm"]) & np.isfinite(score)
                            y = state[chosen] == "rescue"
                            auc = float(roc_auc_score(y, score[chosen])) if np.unique(y).size == 2 else np.nan
                            auctions.append({"encoder": encoder, "cohort": cohort, "training_seed": int(seed),
                                             "k": k, "subset": subset, "modality": modality, "auc": auc,
                                             "n_rescue": int(y.sum()), "n_harm": int((~y).sum()),
                                             "n_plots": int(part.loc[chosen, "plot_idx"].nunique())})
    auc_frame = pd.DataFrame(auctions)
    group_keys = ["encoder", "cohort", "k", "subset", "modality"]
    summaries = []
    for key, part in auc_frame.groupby(group_keys, sort=False):
        finite = part.auc.dropna()
        summaries.append({**dict(zip(group_keys, key)), "mean": finite.mean(), "minimum": finite.min(),
                          "maximum": finite.max(), "valid_seed_count": len(finite),
                          "rescue_count_minimum": int(part.n_rescue.min()), "rescue_count_maximum": int(part.n_rescue.max()),
                          "harm_count_minimum": int(part.n_harm.min()), "harm_count_maximum": int(part.n_harm.max())})
    return pd.DataFrame(distributions), auc_frame, pd.DataFrame(summaries)


def _encoder_bootstrap(adapted: pd.DataFrame, pretrained: pd.DataFrame, cohort: str,
                       policies: dict) -> tuple[dict, pd.DataFrame]:
    models = ("zero_k5", "zero_k10", *policies)
    vectors, representatives, by_bytes = {}, {}, {}
    for encoder, rows in (("adapted", adapted), ("pretrained", pretrained)):
        for model in models:
            name = f"{encoder}::{model}"
            values = rows[f"{model}_pred"].to_numpy(dtype=np.int64)
            signature = values.astype("<i8", copy=False).tobytes()
            representative = by_bytes.get(signature)
            if representative is None:
                representative = name
                by_bytes[signature] = representative
                vectors[representative] = values
            representatives[name] = representative
    names = tuple(vectors)
    first = adapted.loc[adapted.training_seed.eq(1)]
    plots = first[["plot_idx", "canonical_l3_id"]].drop_duplicates().sort_values("plot_idx").reset_index(drop=True)
    indices = pd.Series(np.arange(len(plots)), index=plots.plot_idx)
    support = np.zeros((len(plots), 18), dtype=float)
    np.add.at(support, (indices.loc[first.plot_idx].to_numpy(), first.label_id_dense.to_numpy()), 1)
    diagonal = np.zeros((len(plots), len(standard.SEEDS), len(names), 18), dtype=float)
    predicted = np.zeros_like(diagonal)
    for si, seed in enumerate(standard.SEEDS):
        mask = adapted.training_seed.eq(seed).to_numpy()
        part = adapted.loc[mask]
        pi = indices.loc[part.plot_idx].to_numpy()
        truth = part.label_id_dense.to_numpy()
        for mi, values in enumerate(vectors.values()):
            pred = values[mask]
            correct = pred == truth
            np.add.at(predicted, (pi, si, mi, pred), 1)
            np.add.at(diagonal, (pi[correct], si, mi, truth[correct]), 1)
    strata = [np.flatnonzero(plots.canonical_l3_id.to_numpy() == label)
              for label in sorted(plots.canonical_l3_id.unique())]
    weights = np.zeros((standard.BOOTSTRAP_REPLICATES, len(plots)), dtype=np.int64)
    rng = np.random.default_rng(standard.BOOTSTRAP_SEED)
    for draw in weights:
        for group in strata:
            draw += np.bincount(rng.choice(group, size=len(group), replace=True), minlength=len(plots))
    float_weights = weights.astype(float)
    shape = (len(weights), len(standard.SEEDS), len(names), 18)
    tp = (float_weights @ diagonal.reshape(len(plots), -1)).reshape(shape)
    pred = (float_weights @ predicted.reshape(len(plots), -1)).reshape(shape)
    support_boot = (float_weights @ support)[:, None, None, :]
    accuracy = tp.sum(axis=-1) / support_boot.sum(axis=-1)
    denominator = support_boot + pred
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    weighted_f1 = (f1 * support_boot).sum(axis=-1) / support_boot.sum(axis=-1)
    tp_point, pred_point = diagonal.sum(axis=0), predicted.sum(axis=0)
    support_point = support.sum(axis=0)[None, None, :]
    total = float(support_point.sum())
    point_accuracy = tp_point.sum(axis=-1) / total
    denominator = support_point + pred_point
    point_f1 = np.divide(2 * tp_point, denominator, out=np.zeros_like(tp_point), where=denominator != 0)
    point_weighted_f1 = (point_f1 * support_point).sum(axis=-1) / total
    comparisons, draws = [], {"replicate": np.arange(len(weights))}
    for metric, values, points in (("accuracy", accuracy, point_accuracy),
                                   ("weighted_f1", weighted_f1, point_weighted_f1)):
        for model in models:
            ai = names.index(representatives[f"adapted::{model}"])
            pi = names.index(representatives[f"pretrained::{model}"])
            delta = (values[:, :, ai] - values[:, :, pi]).mean(axis=1)
            low, high = np.quantile(delta, [0.025, 0.975], method="linear")
            comparisons.append({"cohort": cohort, "model": model, "metric": metric,
                                "comparison": "adapted_minus_pretrained",
                                "point_estimate": float((points[:, ai] - points[:, pi]).mean()),
                                "lower": float(low), "upper": float(high)})
            draws[f"{model}_{metric}"] = delta
    settings = {"cohort": cohort, "replicates": len(weights), "random_seed": standard.BOOTSTRAP_SEED,
                "unit": "whole habitat-stratified plot retaining all photos and four expert seeds",
                "interval": "95% paired percentile; linear quantiles; mean seed metric difference",
                "thresholds_refitted_in_bootstrap": False, "adaptive_exploration_accounted_for": False,
                "multiple_comparison_adjustment": False, "unique_prediction_vectors": len(names),
                "resample_multiplicity_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
                "comparisons": comparisons}
    return settings, pd.DataFrame(draws)


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> list[str]:
    names = []
    for extension in ("png", "pdf"):
        name = f"{stem}.{extension}"
        figure.savefig(output / name, dpi=200, facecolor="white")
        names.append(name)
    plt.close(figure)
    return names


def _encoder_plot(output: Path, intervals: pd.DataFrame, policies: dict) -> list[str]:
    models = ["zero_k5", "zero_k10"] + [name for name, policy in policies.items() if policy["minimum_plots"] == 20]
    labels = ["Zero\nK5", "Zero\nK10"] + [f"K{policies[name]['k']}\nH≤{policies[name]['harm_numerator'] / 10:g}R" for name in models[2:]]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
    displayed = intervals.loc[intervals.model.isin(models) & intervals.metric.eq("accuracy")]
    lower = min(0.0, float(displayed.lower.min()) * 100)
    upper = max(0.0, float(displayed.upper.max()) * 100)
    span = max(1.0, upper - lower)
    for cohort, axis in zip(standard.COHORTS, axes):
        part = intervals.loc[intervals.cohort.eq(cohort) & intervals.metric.eq("accuracy")].set_index("model")
        axis.spines[["top", "right"]].set_visible(False)
        axis.axhline(0, color="#596675", linewidth=1)
        axis.grid(axis="y", color="#edf0f2", linewidth=0.7)
        axis.set_axisbelow(True)
        for i, model in enumerate(models):
            row = part.loc[model]
            color = "#687988" if model.startswith("zero") else ("#00897b" if policies[model]["k"] == 5 else "#6b63ac")
            axis.plot([i, i], [100 * row.lower, 100 * row.upper], color=color, linewidth=1.5)
            axis.scatter(i, 100 * row.point_estimate, color=color, s=38, zorder=3)
        axis.set_xticks(np.arange(len(models)), labels, fontsize=8)
        axis.set_xlim(-0.5, len(models) - 0.5)
        axis.set_ylim(lower - 0.12 * span, upper + 0.12 * span)
        axis.set_title(standard.COHORT_NAMES[cohort], fontsize=11, pad=12)
    axes[0].set_ylabel("Adapted − pretrained accuracy (percentage points)", fontsize=10)
    figure.suptitle("Does changing the image representation improve correction?", fontsize=16, y=0.98)
    figure.text(0.5, 0.925, "Zero-threshold controls and all primary 20-plot policies; identical expert predictions and query cohorts",
                ha="center", fontsize=10, color="#596675")
    figure.text(0.5, 0.068, "Dots: mean paired difference across four seeds. Lines: conditional 95% whole-plot bootstrap intervals.",
                ha="center", fontsize=9, color="#596675")
    figure.text(0.5, 0.028, "Threshold policies were fitted separately on validation for each encoder. Intervals do not account for selection or prior test exposure.",
                ha="center", fontsize=9, color="#596675")
    figure.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.20, wspace=0.12)
    return _save_figure(figure, output, "encoder_comparison")


def _diagnostic_plot(output: Path, auc: pd.DataFrame) -> list[str]:
    selected = auc.loc[auc.cohort.eq("validation")]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharey=True)
    for column, k in enumerate((5, 10)):
        for row, subset in enumerate(DIAGNOSTIC_SUBSETS):
            axis = axes[row, column]
            axis.spines[["top", "right"]].set_visible(False)
            axis.axhline(0.5, color="#596675", linestyle="--", linewidth=1)
            axis.grid(axis="y", color="#edf0f2", linewidth=0.7)
            axis.set_axisbelow(True)
            for ei, encoder in enumerate(ENCODERS):
                frame = selected.loc[selected.k.eq(k) & selected.subset.eq(subset) & selected.encoder.eq(encoder)]
                offset = -0.12 if ei == 0 else 0.12
                for position, modality in enumerate(SCORE_COLUMNS):
                    values = frame.loc[frame.modality.eq(modality)].sort_values("training_seed").auc.dropna().to_numpy()
                    if not values.size:
                        continue
                    axis.scatter(position + offset + np.linspace(-0.025, 0.025, len(values)), values,
                                 color=ENCODER_COLORS[encoder], alpha=0.35, s=22, linewidths=0)
                    axis.scatter(position + offset, values.mean(), color=ENCODER_COLORS[encoder],
                                 marker="D", s=38, label=encoder.capitalize() if position == 0 else None, zorder=3)
            axis.set_ylim(0, 1)
            axis.set_xticks([0, 1, 2], ["Image score", "Geo score", "Minimum score"], fontsize=9)
            axis.set_title(f"Strict K{k} · " + ("all eligible disagreements" if row == 0 else "geo score > 0"), fontsize=10)
            if column == 0:
                axis.set_ylabel("AUROC: rescue ranked above harm", fontsize=10)
            if row == 0:
                axis.legend(fontsize=8, frameon=False, loc="upper left")
    figure.suptitle("Validation: can geometry distinguish rescue from harm?", fontsize=15, y=0.98)
    figure.text(0.5, 0.055, "Diamonds average seed-specific AUROCs; faint dots show seeds. Both-wrong cases are excluded from AUROC.",
                ha="center", fontsize=9, color="#596675")
    figure.text(0.5, 0.024, "The geo-positive subset asks whether image evidence is useful after geo geometry favors the geo candidate.",
                ha="center", fontsize=9, color="#596675")
    figure.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.13, hspace=0.27, wspace=0.13)
    return _save_figure(figure, output, "validation_geometry_diagnostics")


def _report(output: Path, encoder_comparison: pd.DataFrame, encoder_summary: pd.DataFrame,
            encoder_intervals: pd.DataFrame, auc_summary: pd.DataFrame, policies: dict,
            pretrained_policies: dict, selected: pd.DataFrame, policy_comparison: pd.DataFrame) -> None:
    lines = ["# Adapted versus pretrained image geometry", "",
             "This matched comparison replaces the historical pretrained-image feature cache with features "
             "from seed-specific habitat-adapted towers. Expert predictions, geo geometry, query cohorts, "
             "reference memberships and the 18-policy threshold grid remain matched. No out-of-fold "
             "representation analysis is performed.", ""]
    for cohort in standard.COHORTS:
        rows = encoder_comparison.loc[encoder_comparison.cohort.eq(cohort)].set_index("model")
        lines.append(f"- {standard.COHORT_NAMES[cohort]}: adapted minus pretrained accuracy is "
                     f"{100 * rows.loc['zero_k5', 'delta_accuracy']:+.3f} percentage points for strict zero-threshold K5 "
                     f"and {100 * rows.loc['zero_k10', 'delta_accuracy']:+.3f} for strict zero-threshold K10.")
    lines += ["", "![Paired encoder comparison](encoder_comparison.png)", "",
              "## What is compared", "",
              "Four seed-specific adapted towers were fitted only on the 3,378 development-training images "
              "from 1,300 plots. Within a seed, the same tower embeds both reference and query images and "
              "serves both the 3,378- and 4,200-image reference banks; expanding the bank causes no additional "
              "encoder fit. Validation and test query plots are disjoint from the encoder-fitting plots. "
              "The run declaration and extraction manifest record the fitting and extraction provenance.", "",
              "The historical comparator uses the frozen pretrained SigLIP2 cache from the "
              "[previous threshold study](../2026-09-30_joint_geometry_thresholds/report.md). "
              "That cache lacks complete extraction-checkpoint lineage. New extraction preserves the legacy "
              "BGR and forced-439-resize convention for comparability, but this remains a comparison with a "
              "historical cache rather than a perfectly isolated causal encoder experiment. The saved classification "
              "experts and RF predictions are unchanged, and geo scores are checked to agree across representations.", "",
              "Strict K=5 or K=10 means averaging cosine distances to K distinct reference plots of each "
              "candidate class; each plot contributes its closest photo in image space. Each modality is "
              "L2-normalized separately. Insufficient class support retains fusion; this study does not reduce K.", "",
              r"\[A_m=\log\frac{d_m(c_r)+10^{-8}}{d_m(c_g)+10^{-8}},\qquad S=\min(A_{\mathrm{image}},A_{\mathrm{geo}}).\]", "",
              "Switch fusion to geo only on eligible disagreements with S>τ. Strict zero-threshold comparisons "
              "hold τ=0 for both representations and isolate the geometry change. Threshold-policy comparisons "
              "match policy IDs and guards, but their scalar thresholds are fitted separately on validation for "
              "each encoder. They compare the resulting two procedures, not a fixed common numeric threshold.", "",
              "The selection grid remains K∈{5,10}, minimum acted plots∈{20,10,0}, and pooled-validation harm "
              "caps H≤αR for α∈{0.5,0.7,0.9}, with 1% minimum coverage and positive net rescue. "
              "One common threshold across four seeds maximizes pooled validation net rescue, breaking ties "
              "by fewer harms, fewer actions and a higher threshold. Thresholds are frozen before test application "
              "and transfer unchanged to both reference banks. No test result selects or filters a policy.", "",
              "Validation has 822 images/325 plots and references from 3,378 development-training images/1,300 plots. "
              "Test has 1,347 images/531 plots, compared with banks of 3,378 images/1,300 plots and 4,200 images/1,625 plots. "
              "Validation experts were fitted on 3,378 development-training images; test experts are the historical "
              "final-epoch-50 heads fitted on all 4,200 development images. These expert fits remain distinct from "
              "the image encoder used to compute geometry; its fitting/extraction provenance is documented separately.", "",
              "## Validation geometry diagnostics", "",
              "A rescue case has fusion wrong and geo correct; a harm case has fusion correct and geo wrong. "
              "AUROC ranks rescue above harm using larger geometric support, separately for each seed. "
              "Both-wrong disagreements are included in distribution summaries but excluded from this binary AUROC. "
              "The geo-positive subset conditions on A_geo>0 before asking whether image geometry adds useful evidence.", "",
              "![Validation geometry diagnostics](validation_geometry_diagnostics.png)", "",
              "The following AUROCs are mean [minimum, maximum] across seeds for validation disagreements with A_geo>0. "
              "Seed ranges are not confidence intervals.", "",
              "| K | Score | Pretrained AUROC | Adapted AUROC |",
              "|---:|---|---:|---:|"]
    aindex = auc_summary.set_index(["encoder", "cohort", "k", "subset", "modality"])
    for k in (5, 10):
        for modality in SCORE_COLUMNS:
            cells = [str(k), modality]
            for encoder in ENCODERS:
                item = aindex.loc[(encoder, "validation", k, "geo_positive", modality)]
                cells.append(f"{item['mean']:.4f} [{item['minimum']:.4f}, {item['maximum']:.4f}]")
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "[Per-seed AUROCs and rescue/harm counts](geometry_auc_by_seed.csv) · "
              "[AUROC summaries](geometry_auc_summary.csv) · "
              "[Counts, quartiles and positive-score fractions by outcome](geometry_distributions.csv). "
              "These files also retain descriptive diagnostics for each test bank; none selects a threshold.", "",
              "## Fixed classification baselines", "",
              "These mean metrics across four seeds are unchanged between image-geometry choices. "
              "The same test baselines serve both reference banks.", "",
              "| Cohort | Model | Accuracy | Weighted F1 | Macro F1 |",
              "|---|---|---:|---:|---:|"]
    baseline_index = encoder_summary.set_index(["encoder", "cohort", "model", "metric"])
    for cohort, label, models in (("validation", "Validation", ("image", "geo", "fusion")),
                                  ("test_3378", "Test, both banks", ("image", "geo", "fusion", "rf"))):
        for model in models:
            accuracy = baseline_index.loc[("adapted", cohort, model, "accuracy"), "mean"]
            weighted = baseline_index.loc[("adapted", cohort, model, "weighted_f1"), "mean"]
            macro = baseline_index.loc[("adapted", cohort, model, "macro_f1"), "mean"]
            lines.append(f"| {label} | {standard._name(model, policies)} | {100 * accuracy:.4f}% | "
                         f"{weighted:.4f} | {macro:.4f} |")
    lines += ["", "## Matched encoder performance", "",
              "The table reports mean accuracy across seeds for strict zero-threshold controls and all primary "
              "20-plot policies. F1/MCC, every seed, and the full 18-policy grid remain in the linked tables.", "",
              "| Cohort | Policy | Pretrained accuracy | Adapted accuracy | Change (pp) |",
              "|---|---|---:|---:|---:|"]
    primary_models = ["zero_k5", "zero_k10"] + [name for name, policy in policies.items() if policy["minimum_plots"] == 20]
    for cohort in standard.COHORTS:
        for model in primary_models:
            item = encoder_comparison.loc[encoder_comparison.cohort.eq(cohort) & encoder_comparison.model.eq(model)].iloc[0]
            lines.append(f"| {standard.COHORT_NAMES[cohort]} | {standard._name(model, policies)} | "
                         f"{100 * item.pretrained_accuracy:.4f}% | {100 * item.adapted_accuracy:.4f}% | "
                         f"{100 * item.delta_accuracy:+.4f} |")
    lines += ["", "[Both encoders: per-seed metrics](encoder_performance_by_seed.csv) · "
              "[Both encoders: means/ranges](encoder_performance_summary.csv) · "
              "[All encoder differences](encoder_policy_comparison.csv) · "
              "[Adapted versus fusion/RF/zero, full grid](policy_comparison_summary.csv)", "",
              "All metrics include the full cohort and fallback cases. The fixed 18-class ontology assigns zero "
              "F1 to absent classes in macro F1. Four seed realizations do not represent independent images.", "",
              "## Thresholds and intervention diagnostics", "",
              "| K | Minimum plots | Harm cap | Pretrained threshold | Adapted threshold | Adapted feasible? |",
              "|---:|---:|---:|---:|---:|---|"]
    for name, policy in policies.items():
        lines.append(f"| {policy['k']} | {policy['minimum_plots']} | {policy['harm_numerator'] / 10:g} | "
                     f"{standard._threshold_text(pretrained_policies[name])} | {standard._threshold_text(policy)} | "
                     f"{'Yes' if policy['selected_feasible'] else 'No'} |")
    lines += ["", "[Adapted selection and validation utility](threshold_selection.csv) · "
              "[Adapted per-seed interventions](interventions.csv) · [Adapted pooled diagnostics](pooled_interventions.csv). "
              "These include rescues, harms, both-wrong overrides, coverage, acted plots and observed test H/R. "
              "Test harm-cap violations are descriptive and do not trigger reselection or filtering.", "",
              "![Adapted threshold performance](comparison.png)", "",
              "## Paired uncertainty and interpretation", "",
              "Each cohort uses 2,000 habitat-stratified whole-plot draws with seed 20260930. A selected plot "
              "retains all photos and four seeds; identical draws pair encoders and policies. Each seed's metrics "
              "are recomputed before averaging paired differences. The new encoder comparisons cover strict-zero "
              "K5/K10 and all 18 matched policy IDs (120 accuracy/weighted-F1 intervals). The adapted-policy "
              "comparisons with fusion, matching zero threshold and test RF add the same 288 intervals as before.", "",
              "Intervals are conditional 95% percentile intervals with linear quantiles. Thresholds are not refitted "
              "within the bootstrap, and intervals do not account for threshold selection, representation choice, "
              "prior test exposure, adaptation of the study or multiple comparisons. Validation both informs "
              "threshold selection and supplies the main geometry diagnostics; it is not an independent confirmation. "
              "This no-OOF study does not estimate out-of-fold representation generalization. Adapted towers "
              "were fitted on the fixed development-training plots, with no validation/test query-plot overlap "
              "in that fit. Threshold selection still uses validation labels, and the historical pretrained cache "
              "has the lineage limitation described above.", "",
              "[Paired adapted-minus-pretrained intervals](encoder_paired_intervals.csv) · "
              "[Adapted versus fixed baselines](paired_intervals.csv) · "
              "[All bootstrap settings](paired_bootstrap.json)", "",
              "The cleaned test set was previously examined. This is an adaptive matched comparison with every "
              "declared policy retained; it does not select or promote a winner. New independent data are required "
              "for confirmation. Historical artifacts remain unchanged.", ""]
    (output / "report.md").write_text("\n".join(lines))


def write_results(output: Path, validation: pd.DataFrame, tests: dict[str, pd.DataFrame],
                  selection: dict, declaration: dict) -> dict:
    """Report the adapted encoder and authenticated historical pretrained match."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policies = standard._policies(selection)
    if set(tests) != {"3378", "4200"}:
        raise ValueError("Expected both declared test reference banks")
    adapted = {"validation": standard._validate(validation, "validation", policies)}
    adapted.update({f"test_{bank}": standard._validate(tests[bank], f"test_{bank}", policies) for bank in ("3378", "4200")})
    pretrained, pretrained_policies = _load_pretrained(policies)
    _check_matching(adapted, pretrained)
    selected = standard._selection_table(adapted["validation"], policies)
    performance, summary, per_class, interventions, pooled, support, details = standard._tables(adapted, policies)
    old_performance, old_summary, _, old_interventions, old_pooled, _, _ = standard._tables(pretrained, pretrained_policies)
    comparison = standard._comparison_table(summary, pooled, policies)
    encoder_performance, encoder_summary, encoder_comparison = _encoder_tables(performance, old_performance, summary, old_summary)
    distributions, auc, auc_summary = _diagnostics({"pretrained": pretrained, "adapted": adapted})
    tables = {"threshold_selection.csv": selected, "performance_by_seed.csv": performance,
              "performance_summary.csv": summary, "performance_by_habitat.csv": per_class,
              "interventions.csv": interventions, "pooled_interventions.csv": pooled, "reference_coverage.csv": support,
              "policy_comparison_summary.csv": comparison, "encoder_performance_by_seed.csv": encoder_performance,
              "encoder_performance_summary.csv": encoder_summary, "encoder_policy_comparison.csv": encoder_comparison,
              "encoder_interventions.csv": pd.concat([old_interventions.assign(encoder="pretrained"), interventions.assign(encoder="adapted")], ignore_index=True),
              "encoder_pooled_interventions.csv": pd.concat([old_pooled.assign(encoder="pretrained"), pooled.assign(encoder="adapted")], ignore_index=True),
              "geometry_distributions.csv": distributions, "geometry_auc_by_seed.csv": auc, "geometry_auc_summary.csv": auc_summary}
    bootstrap = {"adapted_vs_baselines": {}, "adapted_vs_pretrained": {}}
    intervals, encoder_intervals = [], []
    pindex = performance.set_index(["cohort", "training_seed", "model"])
    cindex = encoder_comparison.set_index(["cohort", "model"])
    for cohort in standard.COHORTS:
        settings, replicates = standard._bootstrap(adapted[cohort], cohort, policies)
        bootstrap["adapted_vs_baselines"][cohort] = settings
        tables[f"bootstrap_replicates_{cohort}.csv"] = replicates
        intervals.extend(settings["comparisons"])
        for item in settings["comparisons"]:
            expected = np.mean([pindex.loc[(cohort, seed, item["target"]), item["metric"]]
                                - pindex.loc[(cohort, seed, item["baseline"]), item["metric"]] for seed in standard.SEEDS])
            if not np.isclose(expected, item["point_estimate"], atol=1e-12, rtol=0):
                raise ValueError("Adapted bootstrap differs from canonical classification metrics")
        encoder_settings, encoder_replicates = _encoder_bootstrap(adapted[cohort], pretrained[cohort], cohort, policies)
        if settings["resample_multiplicity_sha256"] != encoder_settings["resample_multiplicity_sha256"]:
            raise ValueError("Encoder and baseline comparisons use different plot draws")
        bootstrap["adapted_vs_pretrained"][cohort] = encoder_settings
        tables[f"encoder_bootstrap_replicates_{cohort}.csv"] = encoder_replicates
        encoder_intervals.extend(encoder_settings["comparisons"])
        for item in encoder_settings["comparisons"]:
            expected = cindex.loc[(cohort, item["model"]), f"delta_{item['metric']}"]
            if not np.isclose(expected, item["point_estimate"], atol=1e-12, rtol=0):
                raise ValueError("Encoder bootstrap differs from canonical classification metrics")
    if len(intervals) != 288 or len(encoder_intervals) != 120:
        raise ValueError("Incomplete declared encoder/baseline interval set")
    tables["paired_intervals.csv"] = pd.DataFrame(intervals)
    tables["encoder_paired_intervals.csv"] = pd.DataFrame(encoder_intervals)
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    standard._json(output / "metrics.json", {"declaration": declaration, "selection": selection,
                   "cohort_sizes": standard.SIZES, "expert_seeds": list(standard.SEEDS), "fixed_class_count": 18,
                   "adapted_per_seed": details, "encoder_mean_and_ranges": standard._records(encoder_summary),
                   "encoder_comparisons": standard._records(encoder_comparison),
                   "geometry_auc_summary": standard._records(auc_summary)})
    standard._json(output / "paired_bootstrap.json", bootstrap)
    figures = standard._plot(output, performance, policies)
    figures += _encoder_plot(output, tables["encoder_paired_intervals.csv"], policies)
    figures += _diagnostic_plot(output, auc)
    _report(output, encoder_comparison, encoder_summary, tables["encoder_paired_intervals.csv"], auc_summary,
            policies, pretrained_policies, selected, comparison)
    return {"cohort_sizes": {key: {"images": size[0], "plots": size[1]} for key, size in standard.SIZES.items()},
            "policy_count": len(policies), "selection": standard._records(selected),
            "performance_by_seed": standard._records(performance), "performance_summary": standard._records(summary),
            "encoder_policy_comparison": standard._records(encoder_comparison),
            "geometry_auc_summary": standard._records(auc_summary), "bootstrap": bootstrap,
            "output_files": [*tables, "metrics.json", "paired_bootstrap.json", "report.md", *figures]}

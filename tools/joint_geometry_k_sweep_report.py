"""Report the declared shared-effective-K sweep from caller-supplied rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joint_geometry_test_report import plot_confusion_aggregates
from multimodal.geo_helpfulness_protocol import FIXED_CLASS_ONTOLOGY
from multimodal.geo_helpfulness_router_metrics import classification_metrics


KS = (1, 5, 10, 20)
SEEDS = (1, 2, 3, 4)
METRICS = ("accuracy", "weighted_f1", "macro_f1", "mcc")
SIZES = {"validation": (822, 325), "test": (1347, 531)}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260930
EPSILON = 1e-8
BASE_MODELS = {"image": "image_pred", "geo": "geo_pred", "fusion": "raw_pred",
               "strict_k5": "strict_k5_pred", **{f"k{k}": f"k{k}_pred" for k in KS}}
MODELS = {"validation": dict(BASE_MODELS), "test": {**BASE_MODELS, "rf": "rf_pred"}}
NAMES = {"image": "Image expert", "geo": "Geo expert", "fusion": "Fusion expert",
         "rf": "Fusion + frozen RF", "strict_k5": "Previous strict K5",
         **{f"k{k}": f"Requested K={k}" for k in KS}}
COMMON_CONTRASTS = ([(f"k{k}", "fusion") for k in KS] + [("k5", "strict_k5"),
                    ("k5", "k1"), ("k10", "k5"), ("k20", "k10"), ("k20", "k5")])
CONTRASTS = {"validation": list(COMMON_CONTRASTS),
             "test": COMMON_CONTRASTS + [(f"k{k}", "rf") for k in KS]}


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _validate(rows: pd.DataFrame, cohort: str) -> pd.DataFrame:
    identity = ["row_uid", "file", "plot_idx", "canonical_l3_id", "label_id_dense", "label_name"]
    required = {*identity, "training_seed", *MODELS[cohort].values(), "strict_k5_acted",
                "reference_plots_fusion_class", "reference_plots_geo_class"}
    for k in KS:
        required.update({f"k{k}_{suffix}" for suffix in ("acted", "effective", "A_image", "A_geo",
                         "d_image_fusion", "d_image_geo", "d_geo_fusion", "d_geo_geo")})
    if cohort == "test":
        required.add("rf_acted")
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"Missing {cohort} report columns: {missing}")
    data = rows.sort_values(["training_seed", "row_uid"]).reset_index(drop=True).copy()
    if set(data.training_seed) != set(SEEDS) or data.duplicated(["training_seed", "row_uid"]).any():
        raise ValueError(f"Expected four unique matched seed/query sets: {cohort}")
    first = data.loc[data.training_seed.eq(SEEDS[0]), identity].reset_index(drop=True)
    if (len(first), first.plot_idx.nunique()) != SIZES[cohort]:
        raise ValueError(f"Unexpected {cohort} cohort size")
    for seed in SEEDS:
        if not data.loc[data.training_seed.eq(seed), identity].reset_index(drop=True).equals(first):
            raise ValueError(f"Query metadata differ across {cohort} seeds")
    if first.groupby("plot_idx").canonical_l3_id.nunique().gt(1).any():
        raise ValueError("A query plot has multiple habitat strata")
    ontology = {dense: (canonical, name) for dense, canonical, name in FIXED_CLASS_ONTOLOGY}
    for item in first[["label_id_dense", "canonical_l3_id", "label_name"]].drop_duplicates().itertuples(index=False):
        if ontology.get(int(item.label_id_dense)) != (int(item.canonical_l3_id), item.label_name):
            raise ValueError("Query labels differ from the fixed ontology")
    for column in ["label_id_dense", *MODELS[cohort].values()]:
        values = data[column].to_numpy()
        if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= 18):
            raise ValueError(f"Invalid class IDs: {cohort}/{column}")
    counts = data[["reference_plots_fusion_class", "reference_plots_geo_class"]].to_numpy()
    if counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("Reference counts must be nonnegative integers")
    disagreement = data.raw_pred.ne(data.geo_pred).to_numpy()
    for k in KS:
        effective = data[f"k{k}_effective"].to_numpy()
        expected_effective = np.minimum(k, counts.min(axis=1))
        if effective.dtype.kind not in "iu" or not np.array_equal(effective, expected_effective):
            raise ValueError(f"Incorrect shared effective K: {cohort}/{k}")
        for modality in ("image", "geo"):
            dr = data[f"k{k}_d_{modality}_fusion"].to_numpy(dtype=float)
            dg = data[f"k{k}_d_{modality}_geo"].to_numpy(dtype=float)
            a = data[f"k{k}_A_{modality}"].to_numpy(dtype=float)
            finite_distances = np.isfinite(dr) & np.isfinite(dg)
            if not finite_distances[effective > 0].all():
                raise ValueError("Supported candidates must have finite distances")
            for distances in (dr, dg):
                finite = distances[np.isfinite(distances)]
                if np.any(finite < -1e-12) or np.any(finite > 2 + 1e-12):
                    raise ValueError("Cosine distances lie outside [0, 2]")
            expected_a = np.full(len(data), np.nan)
            expected_a[finite_distances] = np.log((dr[finite_distances] + EPSILON) / (dg[finite_distances] + EPSILON))
            if not np.allclose(a, expected_a, equal_nan=True, atol=1e-10, rtol=1e-10):
                raise ValueError(f"Scores do not match declared log-distance ratio: {cohort}/{k}/{modality}")
        a = data[[f"k{k}_A_image", f"k{k}_A_geo"]].to_numpy(dtype=float)
        expected_action = disagreement & (effective > 0) & np.isfinite(a).all(axis=1) & (a > 0).all(axis=1)
        if not np.array_equal(data[f"k{k}_acted"], expected_action):
            raise ValueError(f"Actions differ from the joint-positive rule: {cohort}/{k}")
    for model in MODELS[cohort]:
        if model in ("image", "geo", "fusion"):
            continue
        acted = data[f"{model}_acted"].to_numpy()
        if acted.dtype.kind != "b" or np.any(acted & ~disagreement):
            raise ValueError(f"Invalid Boolean disagreement actions: {cohort}/{model}")
        expected_pred = np.where(acted, data.geo_pred, data.raw_pred)
        if not np.array_equal(data[MODELS[cohort][model]], expected_pred):
            raise ValueError(f"Predictions do not match overrides: {cohort}/{model}")
    expected_strict = data.k5_acted.to_numpy() & data.k5_effective.eq(5).to_numpy()
    if not np.array_equal(data.strict_k5_acted, expected_strict):
        raise ValueError("Strict K5 must match sweep K5 on full-five support and retain fusion otherwise")
    return data


def _tables(cohorts: dict[str, pd.DataFrame]) -> tuple:
    performance, per_class, interventions, usage, coverage, details = [], [], [], [], [], {}
    ontology = {dense: (canonical, name) for dense, canonical, name in FIXED_CLASS_ONTOLOGY}
    for cohort, rows in cohorts.items():
        details[cohort] = {}
        for seed, part in rows.groupby("training_seed", sort=True):
            details[cohort][str(seed)] = {}
            truth = part.label_id_dense.to_numpy()
            raw_correct, geo_correct = part.raw_pred.to_numpy() == truth, part.geo_pred.to_numpy() == truth
            disagreement = part.raw_pred.ne(part.geo_pred).to_numpy()
            for k in KS:
                effective, acted = part[f"k{k}_effective"].to_numpy(), part[f"k{k}_acted"].to_numpy()
                reduced, missing = (effective > 0) & (effective < k), effective == 0
                coverage.append({"cohort": cohort, "training_seed": int(seed), "requested_k": k,
                                 "row_count": len(part), "disagreements": int(disagreement.sum()),
                                 "full_k_images": int((effective == k).sum()),
                                 "reduced_positive_k_images": int(reduced.sum()), "no_reference_images": int(missing.sum()),
                                 "reduced_positive_k_disagreements": int((reduced & disagreement).sum()),
                                 "no_reference_disagreements": int((missing & disagreement).sum()),
                                 "reduced_positive_k_actions": int((reduced & acted).sum()),
                                 "actions": int(acted.sum())})
                for actual in sorted(np.unique(effective)):
                    mask = effective == actual
                    usage.append({"cohort": cohort, "training_seed": int(seed), "requested_k": k,
                                  "effective_k": int(actual), "images": int(mask.sum()),
                                  "disagreements": int((mask & disagreement).sum()),
                                  "actions": int((mask & acted).sum()),
                                  "is_reduced_positive_k": bool(0 < actual < k), "is_no_reference": bool(actual == 0)})
            for model, column in MODELS[cohort].items():
                metrics = classification_metrics(part[column].to_numpy(), truth, 18)
                details[cohort][str(seed)][model] = metrics
                performance.append({"cohort": cohort, "training_seed": int(seed), "model": model,
                                    "row_count": len(part), **{key: metrics[key] for key in METRICS}})
                for item in metrics["per_class"]:
                    canonical, name = ontology[item["class_id"]]
                    per_class.append({"cohort": cohort, "training_seed": int(seed), "model": model,
                                      "canonical_l3_id": canonical, "label_name": name, **item})
                if model in ("image", "geo", "fusion"):
                    continue
                acted = part[f"{model}_acted"].to_numpy()
                rescue = int((acted & ~raw_correct & geo_correct).sum())
                harm = int((acted & raw_correct & ~geo_correct).sum())
                both_wrong = int((acted & ~raw_correct & ~geo_correct).sum())
                net = int((part[column].to_numpy() == truth).sum() - raw_correct.sum())
                if rescue + harm + both_wrong != int(acted.sum()) or net != rescue - harm:
                    raise ValueError("Intervention counts do not reproduce classification changes")
                interventions.append({"cohort": cohort, "training_seed": int(seed), "model": model,
                                      "row_count": len(part), "disagreements": int(disagreement.sum()),
                                      "acted": int(acted.sum()), "rescues": rescue, "harms": harm,
                                      "both_wrong": both_wrong, "net_rescue_minus_harm": net,
                                      "coverage": float(acted.mean())})
    performance_frame = pd.DataFrame(performance)
    summary = []
    for (cohort, model), part in performance_frame.groupby(["cohort", "model"], sort=False):
        for metric in METRICS:
            values = part[metric].dropna()
            summary.append({"cohort": cohort, "model": model, "metric": metric, "mean": values.mean(),
                            "minimum": values.min(), "maximum": values.max(), "seed_count": len(values)})
    return (performance_frame, pd.DataFrame(summary), pd.DataFrame(per_class), pd.DataFrame(interventions),
            pd.DataFrame(usage), pd.DataFrame(coverage), details)


def _bootstrap(rows: pd.DataFrame, cohort: str) -> tuple[dict, pd.DataFrame]:
    needed = {model for pair in CONTRASTS[cohort] for model in pair}
    columns = {model: column for model, column in MODELS[cohort].items() if model in needed}
    counts, plots, seeds, models = plot_confusion_aggregates(rows, columns)
    support = counts[0, 0].sum(axis=-1)
    if not np.array_equal(counts.sum(axis=-1), np.broadcast_to(support, counts.shape[:-1])):
        raise ValueError("Class support differs across paired models or seeds")
    strata = [np.flatnonzero(plots.canonical_l3_id.to_numpy() == label)
              for label in sorted(plots.canonical_l3_id.unique())]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.zeros((BOOTSTRAP_REPLICATES, len(plots)), dtype=np.int64)
    for draw in weights:
        for indices in strata:
            draw += np.bincount(rng.choice(indices, size=len(indices), replace=True), minlength=len(plots))
    diagonal = np.diagonal(counts, axis1=-2, axis2=-1).transpose(2, 0, 1, 3)
    predicted = counts.sum(axis=-2).transpose(2, 0, 1, 3)
    shape = (BOOTSTRAP_REPLICATES, len(seeds), len(models), 18)
    float_weights = weights.astype(np.float64)
    tp = (float_weights @ diagonal.reshape(len(plots), -1)).reshape(shape)
    predicted_boot = (float_weights @ predicted.reshape(len(plots), -1)).reshape(shape)
    support_boot = (float_weights @ support)[:, None, None, :]
    accuracy = tp.sum(axis=-1) / support_boot.sum(axis=-1)
    denominator = support_boot + predicted_boot
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    weighted_f1 = (f1 * support_boot).sum(axis=-1) / support_boot.sum(axis=-1)
    point_tp, point_predicted = diagonal.sum(axis=0), predicted.sum(axis=0)
    point_support = support.sum(axis=0)[None, None, :]
    total = float(point_support.sum())
    point_accuracy = point_tp.sum(axis=-1) / total
    denominator = point_support + point_predicted
    point_f1 = np.divide(2 * point_tp, denominator, out=np.zeros_like(point_tp), where=denominator != 0)
    point_weighted_f1 = (point_f1 * point_support).sum(axis=-1) / total
    comparisons, replicates = [], {"replicate": np.arange(BOOTSTRAP_REPLICATES)}
    for metric, draws, points in (("accuracy", accuracy, point_accuracy),
                                  ("weighted_f1", weighted_f1, point_weighted_f1)):
        for target, baseline in CONTRASTS[cohort]:
            ti, bi = models.index(target), models.index(baseline)
            delta = (draws[:, :, ti] - draws[:, :, bi]).mean(axis=1)
            point = float((points[:, ti] - points[:, bi]).mean())
            lower, upper = np.quantile(delta, [0.025, 0.975], method="linear")
            name = f"{target}_minus_{baseline}"
            replicates[f"{name}_{metric}"] = delta
            comparisons.append({"cohort": cohort, "target": target, "baseline": baseline, "comparison": name,
                                "metric": metric, "point_estimate": point, "lower": float(lower), "upper": float(upper)})
    settings = {"cohort": cohort, "replicates": BOOTSTRAP_REPLICATES, "random_seed": BOOTSTRAP_SEED,
                "unit": "whole plot retaining all photos and all four seeds", "stratify_by": "canonical_l3_id",
                "images": SIZES[cohort][0], "plots": len(plots),
                "interval": "95% paired percentile; linear quantiles; mean seed metric difference",
                "conditional_on": "fixed experts, reference bank, rules and observed habitat strata",
                "adaptive_exploration_accounted_for": False, "multiple_comparison_adjustment": False,
                "resample_multiplicity_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
                "comparisons": comparisons}
    return settings, pd.DataFrame(replicates)


def _plot(output: Path, performance: pd.DataFrame) -> list[str]:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for column, cohort in enumerate(("validation", "test")):
        part = performance.loc[performance.cohort.eq(cohort)]
        for row, metric in enumerate(("accuracy", "weighted_f1")):
            axis = axes[row, column]
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#edf0f2", linewidth=0.7)
            axis.set_axisbelow(True)
            values_shown, means = [], []
            for position, k in enumerate(KS):
                values = part.loc[part.model.eq(f"k{k}")].sort_values("training_seed")[metric].to_numpy() * 100
                values_shown.extend(values.tolist())
                means.append(float(values.mean()))
                axis.scatter(position + np.linspace(-0.055, 0.055, len(values)), values,
                             color="#00897b", alpha=0.3, linewidths=0, s=27, zorder=2)
            axis.plot(np.arange(len(KS)), means, color="#00897b", marker="D", markersize=5,
                      linewidth=1.8, label="Shared-effective-K rule", zorder=3)
            for baseline, color, style in (("fusion", "#596675", "--"), ("strict_k5", "#8a69a4", "-."),
                                            ("rf", "#d48742", ":")):
                if baseline not in MODELS[cohort]:
                    continue
                value = float(part.loc[part.model.eq(baseline), metric].mean()) * 100
                values_shown.append(value)
                axis.axhline(value, color=color, linestyle=style, linewidth=1.3, label=NAMES[baseline])
            axis.set_xticks(np.arange(len(KS)), [str(k) for k in KS])
            axis.set_xlim(-0.18, len(KS) - 0.82)
            span = max(1.0, max(values_shown) - min(values_shown))
            axis.set_ylim(min(values_shown) - 0.16 * span, max(values_shown) + 0.48 * span)
            axis.set_ylabel("Accuracy (%)" if metric == "accuracy" else "Weighted F1 × 100", fontsize=10)
            if row == 0:
                axis.set_title("Development validation" if cohort == "validation" else "Previously examined cleaned test", fontsize=12)
                axis.legend(fontsize=8, frameon=False, loc="best")
            else:
                axis.set_xlabel("Requested K", fontsize=10)
    figure.suptitle("Joint geometry correction across all declared neighborhood sizes", fontsize=15, y=0.98)
    figure.text(0.5, 0.935, "Effective K = min(requested K, reference plots for fusion's class, reference plots for geo's class)",
                ha="center", color="#596675", fontsize=9)
    figure.text(0.5, 0.049, "Diamonds/lines: mean across four seeds; faint dots: individual seeds. All results include the full cohort.",
                ha="center", fontsize=9, color="#596675")
    figure.text(0.5, 0.023, "Seed spread is not a confidence interval. This is adaptive exploration; no K is selected or promoted.",
                ha="center", fontsize=9, color="#596675")
    figure.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.12, hspace=0.23, wspace=0.23)
    names = []
    for extension in ("png", "pdf"):
        name = f"comparison.{extension}"
        figure.savefig(output / name, dpi=200, facecolor="white")
        names.append(name)
    plt.close(figure)
    return names


def _range(values: pd.Series, digits: int = 2) -> str:
    return f"{values.mean():.{digits}f} [{values.min():g}, {values.max():g}]"


def _report(output: Path, summary: pd.DataFrame, interventions: pd.DataFrame,
            coverage: pd.DataFrame, bootstrap: dict) -> None:
    indexed = summary.set_index(["cohort", "model", "metric"])
    sequences = {}
    for cohort in ("validation", "test"):
        sequences[cohort] = ", ".join(f"{100 * indexed.loc[(cohort, f'k{k}', 'accuracy'), 'mean']:.3f}%" for k in KS)
    lines = ["# Shared-effective-K geometry correction", "",
             "For requested K=1, 5, 10 and 20 respectively, mean accuracy across four seeds is "
             f"**{sequences['validation']}** on development validation and **{sequences['test']}** "
             "on the previously examined cleaned test set. All declared K values are reported; no winner is selected.", "",
             "![Accuracy and weighted F1 across the declared K values](comparison.png)", "",
             "## Rule and cohorts", "",
             "For each query and pair of predicted habitats, use one shared neighborhood size:", "",
             r"\[K_{\mathrm{eff}}=\min(K,n_{c_r},n_{c_g}).\]", "",
             "Here n counts distinct reference plots. The same effective K is used for both candidate classes "
             "and both modalities. A plot's distance is the minimum cosine distance over its photos; class distance "
             "is the mean over its nearest effective-K distinct plots. Image and regional geo features are "
             "independently L2-normalized, and the image encoder remains frozen pretrained SigLIP2.", "",
             r"\[A_m(x)=\log\frac{d_m(c_r)+10^{-8}}{d_m(c_g)+10^{-8}}.\]", "",
             "Switch fusion's class c_r to geo's class c_g only on disagreements with finite scores, effective K>0, "
             "A_image>0 and A_geo>0. If either class has no reference plot, retain fusion. Ties also retain fusion. "
             "No expert, router, threshold or class-specific exception is fitted or changed.", "",
             "**Change from previous strict K5:** strict K5 retained fusion whenever either class had fewer than "
             "five plots. Requested K=5 now uses their shared smaller effective K. The two policies are identical "
             "when both classes have five or more plots; their difference isolates this support-handling change.", "",
             "Validation uses 822 images/325 plots, experts fitted on 3,378 development-training images, and "
             "3,378 references/1,300 plots. Test uses 1,347 images/531 plots, the unchanged historical final-epoch-50 "
             "experts fitted on all 4,200 development images, and 4,200 references/1,625 plots. Test experts and "
             "the frozen all-feature RF comparator match the [September 15 report](../2026-09-15_geo_helpfulness_router/report.md). "
             "Every method within a cohort uses the same query identities and four expert seeds. "
             "Validation and test use different fitted experts.", "",
             "## Full-cohort classification", "",
             "Cells are mean [minimum, maximum] across seeds. Accuracy is a percentage; F1 and MCC use native scales. "
             "Every query remains in the denominator, including queries with a reduced effective K or no available reference.", ""]
    for cohort, title in (("validation", "Development validation"), ("test", "Cleaned test")):
        lines += [f"### {title}", "", "| Method | Accuracy (%) | Weighted F1 | Macro F1 | MCC |",
                  "|---|---:|---:|---:|---:|"]
        for model in MODELS[cohort]:
            cells = [NAMES[model]]
            for metric in METRICS:
                item = indexed.loc[(cohort, model, metric)]
                scale = 100 if metric == "accuracy" else 1
                cells.append(f"{item['mean'] * scale:.4f} [{item['minimum'] * scale:.4f}, {item['maximum'] * scale:.4f}]")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["The fixed 18-class ontology includes absent classes as zero F1 in macro F1. "
              "[Per-seed metrics](performance_by_seed.csv) · [Mean/ranges](performance_summary.csv) · "
              "[Per-class metrics](performance_by_habitat.csv) · [Metric records](metrics.json)", "",
              "## Actual neighborhood usage", "",
              "Reduced K means 0<effective K<requested K. No-reference queries have effective K=0 and retain fusion. "
              "Counts below are mean [minimum, maximum] across seeds; the detailed CSV also counts every actual K separately.", "",
              "| Cohort | Requested K | Reduced-K images | Reduced-K disagreements | Reduced-K actions | No-reference disagreements |",
              "|---|---:|---:|---:|---:|---:|"]
    for (cohort, k), part in coverage.groupby(["cohort", "requested_k"], sort=False):
        columns = ("reduced_positive_k_images", "reduced_positive_k_disagreements", "reduced_positive_k_actions", "no_reference_disagreements")
        lines.append("| " + " | ".join([cohort, str(k)] + [_range(part[column]) for column in columns]) + " |")
    lines += ["", "[Every actual K by cohort/seed](effective_k_usage.csv) · [Coverage and fallback counts](reference_coverage.csv)", "",
              "## Interventions", "",
              "Rescue means wrong to correct; harm means correct to wrong; both-wrong overrides exchange incorrect "
              "candidates. Values are mean [minimum, maximum] counts across seeds.", "",
              "| Cohort | Policy | Overrides | Rescues | Harms | Both wrong | Net rescue − harm |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for (cohort, model), part in interventions.groupby(["cohort", "model"], sort=False):
        columns = ("acted", "rescues", "harms", "both_wrong", "net_rescue_minus_harm")
        lines.append("| " + " | ".join([cohort, NAMES[model]] + [_range(part[column]) for column in columns]) + " |")
    lines += ["", "[Per-seed interventions](interventions.csv)", "",
              "## Conditional paired intervals", "",
              "Each cohort uses 2,000 habitat-stratified whole-plot bootstrap draws with seed 20260930. "
              "A sampled plot retains all photos and all four seeds, and each draw pairs every policy. "
              "We recompute metrics per seed and average paired differences. These are conditional 95% "
              "percentile intervals using linear quantiles, without adjustment for multiple comparisons or "
              "adaptive exploration. They condition on the fixed experts, banks, rules and observed habitat strata.", "",
              "| Cohort | Policy | Minus comparator | Metric | Point estimate | Conditional 95% interval |",
              "|---|---|---|---|---:|---:|"]
    for cohort in ("validation", "test"):
        for item in bootstrap[cohort]["comparisons"]:
            accuracy = item["metric"] == "accuracy"
            scale = 100 if accuracy else 1
            name = "Accuracy (pp)" if accuracy else "Weighted F1"
            lines.append(f"| {cohort} | {NAMES[item['target']]} | {NAMES[item['baseline']]} | {name} | "
                         f"{item['point_estimate'] * scale:+.5f} | [{item['lower'] * scale:+.5f}, {item['upper'] * scale:+.5f}] |")
    lines += ["", "[Bootstrap settings/intervals](paired_bootstrap.json) · "
              "[Validation replicates](bootstrap_replicates_validation.csv) · [Test replicates](bootstrap_replicates_test.csv)", "",
              "## Interpretation", "",
              "This user-approved sweep follows inspection of earlier validation, test and strict-K5 results. "
              "It is adaptive exploration, not independent confirmation. All four requested K values and all "
              "declared comparisons are retained; no policy is selected or promoted from this report. "
              "No reference samples are selected according to their test effect, and the prior artifacts remain unchanged. "
              "Four seed realizations are not independent test images. Independent new data are needed for confirmation.", "",
              "Candidate distance ratios do not establish that either candidate has strong absolute support. "
              "Reference class frequency, multiple photos per plot and repeated geo vectors still affect neighborhoods. "
              "The frozen image cache retains the historical preprocessing limitations in the "
              "[embedding study](../2026-09-29_embedding_geometry/report.md).", ""]
    (output / "report.md").write_text("\n".join(lines))


def write_results(output: Path, validation: pd.DataFrame, test: pd.DataFrame, declaration: dict) -> dict:
    """Write all declared K results without selecting a preferred policy."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cohorts = {"validation": _validate(validation, "validation"), "test": _validate(test, "test")}
    for identity in ("row_uid", "plot_idx"):
        if set(cohorts["validation"][identity]) & set(cohorts["test"][identity]):
            raise ValueError(f"Validation and test share {identity}")
    performance, summary, per_class, interventions, usage, coverage, details = _tables(cohorts)
    tables = {"performance_by_seed.csv": performance, "performance_summary.csv": summary,
              "performance_by_habitat.csv": per_class, "interventions.csv": interventions,
              "effective_k_usage.csv": usage, "reference_coverage.csv": coverage}
    indexed = performance.set_index(["cohort", "training_seed", "model"])
    bootstrap = {}
    for cohort, rows in cohorts.items():
        settings, replicates = _bootstrap(rows, cohort)
        bootstrap[cohort] = settings
        tables[f"bootstrap_replicates_{cohort}.csv"] = replicates
        for item in settings["comparisons"]:
            expected = np.mean([indexed.loc[(cohort, seed, item["target"]), item["metric"]]
                                - indexed.loc[(cohort, seed, item["baseline"]), item["metric"]] for seed in SEEDS])
            if not np.isclose(expected, item["point_estimate"], atol=1e-12, rtol=0):
                raise ValueError("Bootstrap point estimates differ from canonical metrics")
    if sum(len(value["comparisons"]) for value in bootstrap.values()) != 44:
        raise ValueError("Expected 22 paired contrasts with two metrics each")
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    _json(output / "metrics.json", {"declaration": declaration, "cohort_sizes": SIZES,
          "expert_seeds": list(SEEDS), "requested_k": list(KS), "fixed_class_count": 18,
          "per_seed": details, "mean_and_ranges": _records(summary),
          "interventions": _records(interventions), "effective_k_usage": _records(usage),
          "reference_coverage": _records(coverage)})
    _json(output / "paired_bootstrap.json", bootstrap)
    figures = _plot(output, performance)
    _report(output, summary, interventions, coverage, bootstrap)
    return {"cohort_sizes": {key: {"images": size[0], "plots": size[1]} for key, size in SIZES.items()},
            "seeds": list(SEEDS), "requested_k": list(KS), "performance_by_seed": _records(performance),
            "performance_summary": _records(summary), "interventions": _records(interventions),
            "effective_k_usage": _records(usage), "reference_coverage": _records(coverage), "bootstrap": bootstrap,
            "output_files": [*tables, "metrics.json", "paired_bootstrap.json", "report.md", *figures]}

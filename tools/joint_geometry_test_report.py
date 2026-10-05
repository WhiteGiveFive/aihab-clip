"""Report a declared geometry policy on caller-supplied matched test rows.

This module performs no artifact discovery, fitting, or threshold selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_protocol import FIXED_CLASS_ONTOLOGY
from multimodal.geo_helpfulness_router_metrics import classification_metrics


MODELS = {
    "image": "image_pred",
    "geo": "geo_pred",
    "fusion": "raw_pred",
    "rf": "rf_pred",
    "geometry": "geometry_pred",
}
MODEL_NAMES = {
    "image": "Image expert", "geo": "Geo expert", "fusion": "Fusion expert",
    "rf": "Fusion + frozen RF", "geometry": "Fusion + joint geometry",
}
METRICS = ("accuracy", "weighted_f1", "macro_f1", "mcc")
SEEDS = (1, 2, 3, 4)
N_CLASSES = 18
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260930
RF_POLICY = "relaxed__rf__primary_full727__plots_20__harm_0p7"


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _validate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "row_uid", "file", "plot_idx", "training_seed", "canonical_l3_id",
        "label_id_dense", "label_name", "geometry_acted", "rf_acted", "A_image", "A_geo",
        *MODELS.values(),
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Missing reporting columns: {missing}")
    data = rows.sort_values(["training_seed", "row_uid"]).reset_index(drop=True).copy()
    if set(data.training_seed) != set(SEEDS):
        raise ValueError("Exactly four expert seeds are required")
    if data.duplicated(["training_seed", "row_uid"]).any():
        raise ValueError("Duplicate query/seed rows")
    metadata_columns = ["row_uid", "file", "plot_idx", "canonical_l3_id", "label_id_dense", "label_name"]
    canonical = data.loc[data.training_seed.eq(SEEDS[0]), metadata_columns].reset_index(drop=True)
    if len(canonical) != 1347 or canonical.plot_idx.nunique() != 531:
        raise ValueError("Expected the matched 1,347-image, 531-plot cleaned-test cohort")
    for seed in SEEDS:
        seed_metadata = data.loc[data.training_seed.eq(seed), metadata_columns].reset_index(drop=True)
        if not seed_metadata.equals(canonical):
            raise ValueError(f"Metadata differ across seeds: {seed}")
    if canonical.groupby("plot_idx").canonical_l3_id.nunique().gt(1).any():
        raise ValueError("A plot spans multiple habitat strata")
    ontology = {dense: (canonical_id, name) for dense, canonical_id, name in FIXED_CLASS_ONTOLOGY}
    for label in canonical[["label_id_dense", "canonical_l3_id", "label_name"]].drop_duplicates().itertuples(index=False):
        if ontology.get(int(label.label_id_dense)) != (int(label.canonical_l3_id), label.label_name):
            raise ValueError("Labels differ from the fixed 18-class ontology")
    for column in ["label_id_dense", *MODELS.values()]:
        values = data[column].to_numpy()
        if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= N_CLASSES):
            raise ValueError(f"Invalid class IDs: {column}")
    disagreement = data.raw_pred.ne(data.geo_pred).to_numpy()
    finite = np.isfinite(data[["A_image", "A_geo"]].to_numpy(dtype=float)).all(axis=1)
    expected_geometry = disagreement & finite & data.A_image.gt(0).to_numpy() & data.A_geo.gt(0).to_numpy()
    for method in ("geometry", "rf"):
        acted = data[f"{method}_acted"].to_numpy()
        if acted.dtype.kind != "b":
            raise ValueError(f"{method}_acted must be Boolean")
        if np.any(acted & ~disagreement):
            raise ValueError(f"{method} acts on an expert agreement")
        if method == "geometry" and not np.array_equal(acted, expected_geometry):
            raise ValueError("Geometry actions differ from the declared strict-positive rule")
        expected_prediction = np.where(acted, data.geo_pred, data.raw_pred)
        if not np.array_equal(data[f"{method}_pred"], expected_prediction):
            raise ValueError(f"{method} predictions differ from the declared overrides")
    return data


def _classification_tables(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    overall, per_class, details = [], [], {}
    ontology = {dense: (canonical_id, name) for dense, canonical_id, name in FIXED_CLASS_ONTOLOGY}
    for seed, part in rows.groupby("training_seed", sort=True):
        details[str(seed)] = {}
        for model, column in MODELS.items():
            result = classification_metrics(part[column].to_numpy(), part.label_id_dense.to_numpy(), N_CLASSES)
            details[str(seed)][model] = result
            overall.append({"training_seed": int(seed), "model": model,
                            "row_count": len(part), **{metric: result[metric] for metric in METRICS}})
            for record in result["per_class"]:
                canonical_id, name = ontology[record["class_id"]]
                per_class.append({"training_seed": int(seed), "model": model,
                                  "canonical_l3_id": canonical_id, "label_name": name, **record})
    overall_frame = pd.DataFrame(overall)
    summaries = []
    for model in MODELS:
        part = overall_frame.loc[overall_frame.model.eq(model)]
        for metric in METRICS:
            values = part[metric].dropna()
            summaries.append({"model": model, "metric": metric, "mean": values.mean(),
                              "minimum": values.min(), "maximum": values.max(), "seed_count": len(values)})
    return overall_frame, pd.DataFrame(summaries), pd.DataFrame(per_class), details


def _interventions(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for seed, part in rows.groupby("training_seed", sort=True):
        truth = part.label_id_dense.to_numpy()
        raw_correct = part.raw_pred.to_numpy() == truth
        geo_correct = part.geo_pred.to_numpy() == truth
        disagreement = part.raw_pred.ne(part.geo_pred).to_numpy()
        finite = np.isfinite(part[["A_image", "A_geo"]].to_numpy(dtype=float)).all(axis=1)
        for method in ("geometry", "rf"):
            acted = part[f"{method}_acted"].to_numpy()
            rescue = int((acted & ~raw_correct & geo_correct).sum())
            harm = int((acted & raw_correct & ~geo_correct).sum())
            both_wrong = int((acted & ~raw_correct & ~geo_correct).sum())
            count = int(acted.sum())
            if rescue + harm + both_wrong != count:
                raise ValueError("Intervention outcomes do not sum to action count")
            net = rescue - harm
            observed_net = int((part[f"{method}_pred"].to_numpy() == truth).sum() - raw_correct.sum())
            if net != observed_net:
                raise ValueError("Net interventions do not reconcile with accuracy")
            records.append({
                "training_seed": int(seed), "method": method, "row_count": len(part),
                "disagreements": int(disagreement.sum()), "acted": count,
                "rescues": rescue, "harms": harm, "both_wrong": both_wrong,
                "net_rescue_minus_harm": net, "coverage": count / len(part),
                "rescue_opportunities": int((~raw_correct & geo_correct).sum()),
                "harm_opportunities": int((raw_correct & ~geo_correct).sum()),
                "unavailable_geometry_disagreements": int((disagreement & ~finite).sum()),
            })
    return pd.DataFrame(records)


def plot_confusion_aggregates(
    rows: pd.DataFrame, model_columns: Mapping[str, str],
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, ...], tuple[str, ...]]:
    """Return counts[seed, model, plot, true class, predicted class].

    Counts retain every image within a plot. Resampling the plot axis jointly
    preserves pairing across models and all expert seeds.
    """
    seeds = tuple(sorted(int(seed) for seed in rows.training_seed.unique()))
    models = tuple(model_columns)
    first = rows.loc[rows.training_seed.eq(seeds[0])]
    plots = first[["plot_idx", "canonical_l3_id"]].drop_duplicates().sort_values("plot_idx").reset_index(drop=True)
    if plots.plot_idx.duplicated().any():
        raise ValueError("Each plot needs one bootstrap habitat stratum")
    index = pd.Series(np.arange(len(plots)), index=plots.plot_idx)
    counts = np.zeros((len(seeds), len(models), len(plots), N_CLASSES, N_CLASSES), dtype=np.float64)
    for si, seed in enumerate(seeds):
        part = rows.loc[rows.training_seed.eq(seed)]
        plot_index = index.loc[part.plot_idx].to_numpy()
        labels = part.label_id_dense.to_numpy(dtype=int)
        for mi, column in enumerate(model_columns.values()):
            np.add.at(counts[si, mi], (plot_index, labels, part[column].to_numpy(dtype=int)), 1.0)
    return counts, plots, seeds, models


def paired_plot_bootstrap(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    columns = {model: MODELS[model] for model in ("fusion", "rf", "geometry")}
    counts, plots, seeds, models = plot_confusion_aggregates(rows, columns)
    support = counts[0, 0].sum(axis=-1)
    if not np.array_equal(counts.sum(axis=-1), np.broadcast_to(support, counts.shape[:-1])):
        raise ValueError("Model/seed class support differs within a plot")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero(plots.canonical_l3_id.to_numpy() == label)
              for label in sorted(plots.canonical_l3_id.unique())]
    weights = np.zeros((BOOTSTRAP_REPLICATES, len(plots)), dtype=np.int64)
    for draw in weights:
        for indices in strata:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            draw += np.bincount(sampled, minlength=len(plots))
    # F1 only needs diagonal, true support and predicted support; avoid expanding
    # 2,000 full confusion matrices per model and seed.
    diagonal = np.diagonal(counts, axis1=-2, axis2=-1).transpose(2, 0, 1, 3)
    predicted = counts.sum(axis=-2).transpose(2, 0, 1, 3)
    shape = (BOOTSTRAP_REPLICATES, len(seeds), len(models), N_CLASSES)
    float_weights = weights.astype(np.float64)
    boot_tp = (float_weights @ diagonal.reshape(len(plots), -1)).reshape(shape)
    boot_predicted = (float_weights @ predicted.reshape(len(plots), -1)).reshape(shape)
    boot_support = (float_weights @ support)[:, None, None, :]
    total = boot_support.sum(axis=-1)
    boot_accuracy = boot_tp.sum(axis=-1) / total
    denominator = boot_support + boot_predicted
    boot_f1 = np.divide(2 * boot_tp, denominator, out=np.zeros_like(boot_tp), where=denominator != 0)
    boot_weighted_f1 = (boot_f1 * boot_support).sum(axis=-1) / total
    observed_tp = diagonal.sum(axis=0)
    observed_predicted = predicted.sum(axis=0)
    observed_support = support.sum(axis=0)[None, None, :]
    observed_total = float(observed_support.sum())
    observed_accuracy = observed_tp.sum(axis=-1) / observed_total
    observed_denominator = observed_support + observed_predicted
    observed_f1 = np.divide(2 * observed_tp, observed_denominator, out=np.zeros_like(observed_tp),
                            where=observed_denominator != 0)
    observed_weighted_f1 = (observed_f1 * observed_support).sum(axis=-1) / observed_total
    replicates = {"replicate": np.arange(BOOTSTRAP_REPLICATES)}
    intervals = []
    geometry_index = models.index("geometry")
    for metric, values, point_values in (
        ("accuracy", boot_accuracy, observed_accuracy),
        ("weighted_f1", boot_weighted_f1, observed_weighted_f1),
    ):
        for baseline in ("fusion", "rf"):
            baseline_index = models.index(baseline)
            delta = (values[:, :, geometry_index] - values[:, :, baseline_index]).mean(axis=1)
            point = float((point_values[:, geometry_index] - point_values[:, baseline_index]).mean())
            low, high = np.quantile(delta, [0.025, 0.975], method="linear")
            comparison = f"geometry_minus_{baseline}"
            replicates[f"{comparison}_{metric}"] = delta
            intervals.append({"comparison": comparison, "metric": metric,
                              "point_estimate": point, "lower": float(low), "upper": float(high)})
    result = {
        "replicates": BOOTSTRAP_REPLICATES, "random_seed": BOOTSTRAP_SEED,
        "unit": "whole plot, retaining every photo and all four seeds",
        "stratify_by": "canonical_l3_id", "plots": len(plots), "images": len(rows) // len(seeds),
        "interval": "95% paired percentile interval; linear quantiles; mean seed metric difference",
        "conditional_on": "fixed trained experts, fixed policies, observed habitat strata and sampled test plots",
        "historical_test_exposure_accounted_for": False,
        "resample_multiplicity_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
        "comparisons": intervals,
    }
    return result, pd.DataFrame(replicates)


def _comparison_figure(output: Path, performance: pd.DataFrame) -> list[str]:
    colors = ("#52798f", "#a78357", "#6767a0", "#d48742", "#00897b")
    labels = ("Image", "Geo", "Fusion", "Fusion\n+ frozen RF", "Fusion\n+ geometry")
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    for axis, metric, title in zip(axes, ("accuracy", "weighted_f1"), ("Habitat accuracy (%)", "Weighted F1 × 100")):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#edf0f2", linewidth=0.7)
        axis.set_axisbelow(True)
        all_values = []
        for position, (model, color) in enumerate(zip(MODELS, colors)):
            values = performance.loc[performance.model.eq(model)].sort_values("training_seed")[metric].to_numpy() * 100
            all_values.extend(values.tolist())
            axis.scatter(position + np.linspace(-0.12, 0.12, len(values)), values,
                         s=36, color=color, alpha=0.65, linewidths=0, zorder=2)
            axis.scatter(position, values.mean(), s=54, color=color, marker="D",
                         edgecolors="white", linewidths=0.7, zorder=3)
            axis.text(position, values.max() + 0.65, f"{values.mean():.2f}",
                      ha="center", fontsize=9, color="#34404d")
        axis.set_xticks(np.arange(len(MODELS)), labels, fontsize=10)
        axis.set_ylabel(title, fontsize=11)
        axis.set_xlim(-0.5, len(MODELS) - 0.5)
        axis.set_ylim(min(all_values) - 1.5, max(all_values) + 2.5)
    figure.suptitle("Matched comparison on the previously examined cleaned test set", fontsize=15, y=0.98)
    figure.text(0.5, 0.925, "1,347 images · 531 plots · four fixed expert seeds", ha="center", color="#596675")
    handles = [Line2D([], [], marker="o", linestyle="none", color="#687988", label="Individual expert seed"),
               Line2D([], [], marker="D", linestyle="none", color="#687988", label="Mean across seeds")]
    figure.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.067), frameon=False)
    figure.text(0.5, 0.025, "The same images recur across seeds. Spread between seeds is not a confidence interval.",
                ha="center", fontsize=9, color="#596675")
    figure.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.22, wspace=0.20)
    filenames = []
    for suffix in ("png", "pdf"):
        name = f"comparison.{suffix}"
        figure.savefig(output / name, dpi=200, facecolor="white")
        filenames.append(name)
    plt.close(figure)
    return filenames


def _write_report(output: Path, performance: pd.DataFrame, summary: pd.DataFrame,
                  interventions: pd.DataFrame, bootstrap: dict) -> None:
    means = performance.groupby("model")[list(METRICS)].mean()
    lines = [
        "# Joint geometry correction: matched historical cleaned-test comparison", "",
        f"Across four fixed seeds, joint geometry gives {100 * means.loc['geometry', 'accuracy']:.4f}% "
        f"mean accuracy, compared with {100 * means.loc['fusion', 'accuracy']:.4f}% for fusion and "
        f"{100 * means.loc['rf', 'accuracy']:.4f}% for the frozen RF router. "
        "This is a descriptive comparison on a previously examined test set, not an unbiased new confirmation.", "",
        "![Matched classification comparison](comparison.png)", "",
        "## Frozen policy and matched models", "",
        "The geometry policy uses the predeclared nearest-reference-plot distance (k=1). For each modality m:", "",
        "$$A_m(x)=\\log\\frac{d_m(c_r)+10^{-8}}{d_m(c_g)+10^{-8}}.$$", "",
        "Here c_r is fusion's predicted habitat, c_g is geo's predicted habitat, and d_m(c) is cosine distance "
        "to the nearest reference plot with label c, taking the minimum image distance within each reference plot. "
        "The image representation is frozen pretrained SigLIP2; the averaged regional geo vectors are independently "
        "L2-normalized. Reference labels identify reference classes and are not query labels.", "",
        "Replace fusion with geo exactly when the experts disagree, both scores are finite, "
        "A_image > 0 and A_geo > 0. Keep fusion for ties and unavailable scores. "
        "No new model is fitted and no threshold is selected or adjusted in this check.", "",
        "Geometry uses the frozen 3,378 development-training images from 1,300 plots as its reference bank. "
        "All five methods are evaluated on the same 1,347 cleaned-test images from 531 plots for seeds 1–4. "
        "The three experts are the exact historical final-epoch-50 heads fitted on all 4,200 development images. "
        "The RF comparison is the frozen all-727-feature policy " + f"`{RF_POLICY}`. "
        "This retains the expert and RF choices from the "
        "[September 15 report](../2026-09-15_geo_helpfulness_router/report.md).", "",
        "**Difference from the preceding validation check:** those validation predictions came from experts "
        "fitted on 3,378 development-training images; these historical test experts were fitted on all 4,200 "
        "development images. The geometry reference bank remains 3,378 images. A difference between validation "
        "and test gains therefore cannot be attributed only to a change in evaluation samples.", "",
        "## Overall classification", "",
        "Each entry below is mean [minimum, maximum] across the four expert seeds. "
        "F1 and MCC are on their native scales; accuracy is a percentage.", "",
        "| Model | Accuracy (%) | Weighted F1 | Macro F1 | MCC |",
        "|---|---:|---:|---:|---:|",
    ]
    indexed = summary.set_index(["model", "metric"])
    for model in MODELS:
        cells = [MODEL_NAMES[model]]
        for metric in METRICS:
            stats = indexed.loc[(model, metric)]
            scale = 100 if metric == "accuracy" else 1
            cells.append(f"{stats['mean'] * scale:.4f} [{stats['minimum'] * scale:.4f}, {stats['maximum'] * scale:.4f}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Metrics use the fixed 18-class ontology. Absent-class precision, recall and F1 are zero; "
              "macro F1 includes every ontology class. Repeated seed realizations are not independent images.", "",
              "[Per-seed metrics](performance_by_seed.csv) · [Mean and ranges](performance_summary.csv) · "
              "[Per-class metrics](performance_by_habitat.csv) · [Complete metric records](metrics.json)", "",
              "## Override outcomes", "",
              "Rescue changes a wrong prediction to correct; harm changes a correct prediction to wrong; "
              "a both-wrong override exchanges two incorrect candidates. Counts are per seed, not totals of independent images.", "",
              "| Method | Seed | Overrides | Rescues | Harms | Both wrong | Net rescue − harm |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("geometry", "rf"):
        for record in interventions.loc[interventions.method.eq(method)].to_dict("records"):
            cells = [MODEL_NAMES[method], str(record["training_seed"])]
            cells += [str(record[column]) for column in ("acted", "rescues", "harms", "both_wrong", "net_rescue_minus_harm")]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "[Intervention counts and coverage](interventions.csv)", "",
              "## Paired uncertainty estimates", "",
              "We draw 2,000 habitat-stratified whole-plot bootstrap samples with seed 20260930. "
              "Each sampled plot retains all photos and all four expert seeds, and the same draw is used "
              "for every model. We recompute each seed's metric and average the paired differences across seeds. "
              "The intervals are 95% percentile intervals, conditional on the fixed experts, policies and observed "
              "habitat strata. They do not account for prior test exposure, policy exploration, expert retraining "
              "or new habitats. Sparse strata provide limited information about population variability.", "",
              "| Difference | Metric | Point estimate | Conditional 95% interval |",
              "|---|---|---:|---:|"]
    for result in bootstrap["comparisons"]:
        comparison = "Geometry − fusion" if result["comparison"] == "geometry_minus_fusion" else "Geometry − frozen RF"
        metric_name = "Accuracy (percentage points)" if result["metric"] == "accuracy" else "Weighted F1 (native scale)"
        scale = 100 if result["metric"] == "accuracy" else 1
        lines.append(f"| {comparison} | {metric_name} | {result['point_estimate'] * scale:+.5f} | "
                     f"[{result['lower'] * scale:+.5f}, {result['upper'] * scale:+.5f}] |")
    lines += ["", "[Bootstrap settings and intervals](paired_bootstrap.json) · "
              "[Paired bootstrap replicates](paired_bootstrap_replicates.csv)", "",
              "## Interpretation limits", "",
              "The geometry policy was chosen during development analysis and is applied here without a new fit "
              "or threshold search. The historical cleaned test set has already been examined in earlier router "
              "experiments, including the September 15 report; this check is not an untouched holdout. "
              "Any observed advantage requires confirmation on new independent data.", "",
              "Query labels are used only for evaluation and bootstrap stratification. The two candidate distances "
              "do not measure support for other habitats, and a positive ratio can occur when both candidates "
              "have weak absolute support. Reference class frequency, reference photos per plot and duplicated "
              "geo vectors can affect neighborhood support. The frozen image cache retains the historical "
              "BGR/resize preprocessing limitations documented in the "
              "[embedding study](../2026-09-29_embedding_geometry/report.md).", ""]
    (output / "report.md").write_text("\n".join(lines))


def write_results(output: Path, rows: pd.DataFrame, declaration: dict) -> dict:
    """Write fixed-policy metrics, paired intervals and figures from matched rows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = _validate_rows(rows)
    performance, summary, per_class, details = _classification_tables(data)
    interventions = _interventions(data)
    bootstrap, bootstrap_replicates = paired_plot_bootstrap(data)
    indexed = performance.set_index(["training_seed", "model"])
    for result in bootstrap["comparisons"]:
        baseline = result["comparison"].removeprefix("geometry_minus_")
        metric = result["metric"]
        expected = np.mean([indexed.loc[(seed, "geometry"), metric] - indexed.loc[(seed, baseline), metric] for seed in SEEDS])
        if not np.isclose(expected, result["point_estimate"], atol=1e-12, rtol=0):
            raise ValueError("Bootstrap point estimates do not reproduce canonical metrics")
    tables = {
        "performance_by_seed.csv": performance,
        "performance_summary.csv": summary,
        "performance_by_habitat.csv": per_class,
        "interventions.csv": interventions,
        "paired_bootstrap_replicates.csv": bootstrap_replicates,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    payload = {"declaration": declaration, "models": MODEL_NAMES, "image_count": 1347,
               "plot_count": 531, "expert_seeds": list(SEEDS), "fixed_class_count": N_CLASSES,
               "per_seed": details, "mean_and_ranges": _records(summary),
               "interventions": _records(interventions)}
    _save_json(output / "metrics.json", payload)
    _save_json(output / "paired_bootstrap.json", bootstrap)
    figures = _comparison_figure(output, performance)
    _write_report(output, performance, summary, interventions, bootstrap)
    return {"images": 1347, "plots": 531, "seeds": list(SEEDS),
            "performance_by_seed": _records(performance), "performance_summary": _records(summary),
            "interventions": _records(interventions), "bootstrap": bootstrap,
            "output_files": [*tables, "metrics.json", "paired_bootstrap.json", "report.md", *figures]}

"""Report the declared reference-bank expansion using only supplied rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from joint_geometry_test_report import _validate_rows, plot_confusion_aggregates
from multimodal.geo_helpfulness_protocol import FIXED_CLASS_ONTOLOGY
from multimodal.geo_helpfulness_router_metrics import classification_metrics


MODELS = {
    "geo": "geo_pred", "image": "image_pred", "fusion": "raw_pred", "rf": "rf_pred",
    "geometry3378": "geometry3378_pred", "geometry4200": "geometry_pred",
}
MODEL_NAMES = {
    "geo": "Geo expert", "image": "Image expert", "fusion": "Fusion expert",
    "rf": "Fusion + frozen RF", "geometry3378": "Geometry: 3,378 references",
    "geometry4200": "Geometry: 4,200 references",
}
ACTION_COLUMNS = {
    "rf": "rf_acted", "geometry3378": "geometry3378_acted", "geometry4200": "geometry_acted",
}
METRICS = ("accuracy", "weighted_f1", "macro_f1", "mcc")
SEEDS = (1, 2, 3, 4)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260930


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _validate(rows: pd.DataFrame) -> pd.DataFrame:
    # The immutable earlier validator checks the common cohort, experts, frozen
    # RF and strict-positive rule for the current geometry_pred/acted columns.
    data = _validate_rows(rows)
    required = {"geometry3378_pred", "geometry3378_acted"}
    if not required <= set(data.columns):
        raise ValueError("Missing preserved 3,378-reference predictions or actions")
    acted = data.geometry3378_acted.to_numpy()
    prediction = data.geometry3378_pred.to_numpy()
    if acted.dtype.kind != "b":
        raise ValueError("geometry3378_acted must be Boolean")
    if prediction.dtype.kind not in "iu" or np.any(prediction < 0) or np.any(prediction >= 18):
        raise ValueError("Invalid preserved geometry class IDs")
    disagreement = data.raw_pred.ne(data.geo_pred).to_numpy()
    if np.any(acted & ~disagreement):
        raise ValueError("Preserved geometry policy acts on an expert agreement")
    expected = np.where(acted, data.geo_pred, data.raw_pred)
    if not np.array_equal(expected, prediction):
        raise ValueError("Preserved geometry predictions and actions differ")
    prior_score_columns = {"A_image3378", "A_geo3378"}
    if prior_score_columns & set(data.columns):
        if not prior_score_columns <= set(data.columns):
            raise ValueError("Both preserved geometry scores must be supplied together")
        finite = np.isfinite(data[["A_image3378", "A_geo3378"]].to_numpy(dtype=float)).all(axis=1)
        expected_actions = disagreement & finite & data.A_image3378.gt(0).to_numpy() & data.A_geo3378.gt(0).to_numpy()
        if not np.array_equal(acted, expected_actions):
            raise ValueError("Preserved geometry actions differ from their strict-positive scores")
    return data


def _classification_tables(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    overall, per_class, detailed = [], [], {}
    ontology = {dense: (canonical, name) for dense, canonical, name in FIXED_CLASS_ONTOLOGY}
    for seed, part in rows.groupby("training_seed", sort=True):
        detailed[str(seed)] = {}
        for model, column in MODELS.items():
            result = classification_metrics(part[column].to_numpy(), part.label_id_dense.to_numpy(), 18)
            detailed[str(seed)][model] = result
            overall.append({"training_seed": int(seed), "model": model, "row_count": len(part),
                            **{metric: result[metric] for metric in METRICS}})
            for record in result["per_class"]:
                canonical, name = ontology[record["class_id"]]
                per_class.append({"training_seed": int(seed), "model": model,
                                  "canonical_l3_id": canonical, "label_name": name, **record})
    performance = pd.DataFrame(overall)
    summary = []
    for model in MODELS:
        part = performance.loc[performance.model.eq(model)]
        for metric in METRICS:
            values = part[metric].dropna()
            summary.append({"model": model, "metric": metric, "mean": values.mean(),
                            "minimum": values.min(), "maximum": values.max(), "seed_count": len(values)})
    return performance, pd.DataFrame(summary), pd.DataFrame(per_class), detailed


def _interventions_and_changes(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    intervention_records, change_records = [], []
    for seed, part in rows.groupby("training_seed", sort=True):
        truth = part.label_id_dense.to_numpy()
        raw_correct = part.raw_pred.to_numpy() == truth
        geo_correct = part.geo_pred.to_numpy() == truth
        rescue_case, harm_case = ~raw_correct & geo_correct, raw_correct & ~geo_correct
        both_wrong_case = ~raw_correct & ~geo_correct
        for method, action_column in ACTION_COLUMNS.items():
            acted = part[action_column].to_numpy()
            rescue = int((acted & rescue_case).sum())
            harm = int((acted & harm_case).sum())
            both_wrong = int((acted & both_wrong_case).sum())
            if rescue + harm + both_wrong != int(acted.sum()):
                raise ValueError("Override outcomes do not reconcile")
            observed_net = int((part[MODELS[method]].to_numpy() == truth).sum() - raw_correct.sum())
            if observed_net != rescue - harm:
                raise ValueError("Override counts do not reproduce accuracy change")
            intervention_records.append({
                "training_seed": int(seed), "method": method, "row_count": len(part),
                "disagreements": int(part.raw_pred.ne(part.geo_pred).sum()),
                "acted": int(acted.sum()), "rescues": rescue, "harms": harm, "both_wrong": both_wrong,
                "net_rescue_minus_harm": rescue - harm, "coverage": float(acted.mean()),
                "rescue_opportunities": int(rescue_case.sum()), "harm_opportunities": int(harm_case.sum()),
            })
        prior, expanded = part.geometry3378_acted.to_numpy(), part.geometry_acted.to_numpy()
        groups = {"newly_activated": expanded & ~prior, "dropped": prior & ~expanded,
                  "kept": prior & expanded, "inactive_in_both": ~prior & ~expanded}
        changes = {"training_seed": int(seed), "row_count": len(part)}
        for name, mask in groups.items():
            changes[name] = int(mask.sum())
            changes[f"{name}_rescue_cases"] = int((mask & rescue_case).sum())
            changes[f"{name}_harm_cases"] = int((mask & harm_case).sum())
            changes[f"{name}_both_wrong_cases"] = int((mask & both_wrong_case).sum())
        changes["changed_predictions"] = int(part.geometry3378_pred.ne(part.geometry_pred).sum())
        if changes["changed_predictions"] != changes["newly_activated"] + changes["dropped"]:
            raise ValueError("Action changes and prediction changes do not reconcile")
        delta = ((expanded & rescue_case).sum() - (expanded & harm_case).sum()
                 - (prior & rescue_case).sum() + (prior & harm_case).sum())
        observed_delta = ((part.geometry_pred.to_numpy() == truth).sum()
                          - (part.geometry3378_pred.to_numpy() == truth).sum())
        if int(delta) != int(observed_delta):
            raise ValueError("Bank-change correctness counts do not reconcile")
        changes["net_correct_change"] = int(delta)
        change_records.append(changes)
    return pd.DataFrame(intervention_records), pd.DataFrame(change_records)


def paired_plot_bootstrap(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    columns = {model: MODELS[model] for model in ("fusion", "rf", "geometry3378", "geometry4200")}
    counts, plots, seeds, models = plot_confusion_aggregates(rows, columns)
    support = counts[0, 0].sum(axis=-1)
    if not np.array_equal(counts.sum(axis=-1), np.broadcast_to(support, counts.shape[:-1])):
        raise ValueError("Different class support across paired methods or seeds")
    strata = [np.flatnonzero(plots.canonical_l3_id.to_numpy() == label)
              for label in sorted(plots.canonical_l3_id.unique())]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.zeros((BOOTSTRAP_REPLICATES, len(plots)), dtype=np.int64)
    for draw in weights:
        for indices in strata:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            draw += np.bincount(sampled, minlength=len(plots))
    diagonal = np.diagonal(counts, axis1=-2, axis2=-1).transpose(2, 0, 1, 3)
    predicted = counts.sum(axis=-2).transpose(2, 0, 1, 3)
    shape = (BOOTSTRAP_REPLICATES, len(seeds), len(models), 18)
    float_weights = weights.astype(np.float64)
    boot_tp = (float_weights @ diagonal.reshape(len(plots), -1)).reshape(shape)
    boot_predicted = (float_weights @ predicted.reshape(len(plots), -1)).reshape(shape)
    boot_support = (float_weights @ support)[:, None, None, :]
    boot_accuracy = boot_tp.sum(axis=-1) / boot_support.sum(axis=-1)
    denominator = boot_support + boot_predicted
    boot_f1 = np.divide(2 * boot_tp, denominator, out=np.zeros_like(boot_tp), where=denominator != 0)
    boot_weighted_f1 = (boot_f1 * boot_support).sum(axis=-1) / boot_support.sum(axis=-1)
    observed_tp, observed_predicted = diagonal.sum(axis=0), predicted.sum(axis=0)
    observed_support = support.sum(axis=0)[None, None, :]
    total = float(observed_support.sum())
    observed_accuracy = observed_tp.sum(axis=-1) / total
    denominator = observed_support + observed_predicted
    observed_f1 = np.divide(2 * observed_tp, denominator, out=np.zeros_like(observed_tp), where=denominator != 0)
    observed_weighted_f1 = (observed_f1 * observed_support).sum(axis=-1) / total
    comparisons, replicates = [], {"replicate": np.arange(BOOTSTRAP_REPLICATES)}
    expanded_index = models.index("geometry4200")
    for metric, draws, point_values in (("accuracy", boot_accuracy, observed_accuracy),
                                        ("weighted_f1", boot_weighted_f1, observed_weighted_f1)):
        for baseline in ("fusion", "rf", "geometry3378"):
            baseline_index = models.index(baseline)
            delta = (draws[:, :, expanded_index] - draws[:, :, baseline_index]).mean(axis=1)
            point = float((point_values[:, expanded_index] - point_values[:, baseline_index]).mean())
            low, high = np.quantile(delta, [0.025, 0.975], method="linear")
            name = f"geometry4200_minus_{baseline}"
            replicates[f"{name}_{metric}"] = delta
            comparisons.append({"comparison": name, "metric": metric, "point_estimate": point,
                                "lower": float(low), "upper": float(high)})
    result = {
        "replicates": BOOTSTRAP_REPLICATES, "random_seed": BOOTSTRAP_SEED,
        "unit": "whole plot retaining all photos and all four expert seeds",
        "stratify_by": "canonical_l3_id", "plots": len(plots), "images": len(rows) // len(seeds),
        "interval": "95% paired percentile; linear quantiles; mean seed metric difference",
        "conditional_on": "fixed experts, policies, reference banks and observed habitat strata",
        "historical_test_exposure_and_adaptation_accounted_for": False,
        "resample_multiplicity_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
        "comparisons": comparisons,
    }
    return result, pd.DataFrame(replicates)


def _comparison_figure(output: Path, performance: pd.DataFrame) -> list[str]:
    colors = ("#a78357", "#52798f", "#6767a0", "#d48742", "#779990", "#00897b")
    labels = ("Geo", "Image", "Fusion", "Fusion\n+ RF", "Geometry\n3,378 refs", "Geometry\n4,200 refs")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for axis, metric, title in zip(axes, ("accuracy", "weighted_f1"), ("Habitat accuracy (%)", "Weighted F1 × 100")):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#edf0f2", linewidth=0.7)
        axis.set_axisbelow(True)
        all_values = []
        for position, (model, color) in enumerate(zip(MODELS, colors)):
            values = performance.loc[performance.model.eq(model)].sort_values("training_seed")[metric].to_numpy() * 100
            all_values.extend(values.tolist())
            axis.scatter(position + np.linspace(-0.12, 0.12, len(values)), values,
                         color=color, s=34, alpha=0.65, linewidths=0, zorder=2)
            axis.scatter(position, values.mean(), color=color, marker="D", s=52,
                         edgecolors="white", linewidths=0.7, zorder=3)
            axis.text(position, values.max() + 0.65, f"{values.mean():.2f}", ha="center", fontsize=9)
        axis.set_xticks(np.arange(len(MODELS)), labels, fontsize=9)
        axis.set_ylabel(title, fontsize=11)
        axis.set_xlim(-0.5, len(MODELS) - 0.5)
        axis.set_ylim(min(all_values) - 1.5, max(all_values) + 2.5)
    figure.suptitle("Reference-bank expansion on the previously examined cleaned test set", fontsize=15, y=0.98)
    figure.text(0.5, 0.925, "Identical experts and correction rule · 1,347 images · 531 plots · four seeds",
                ha="center", color="#596675")
    handles = [Line2D([], [], marker="o", linestyle="none", color="#687988", label="Individual expert seed"),
               Line2D([], [], marker="D", linestyle="none", color="#687988", label="Mean across seeds")]
    figure.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.065), frameon=False)
    figure.text(0.5, 0.023, "The same images recur across seeds. Seed spread is not a confidence interval. This is an adaptive follow-up.",
                ha="center", fontsize=9, color="#596675")
    figure.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.22, wspace=0.17)
    names = []
    for extension in ("png", "pdf"):
        name = f"comparison.{extension}"
        figure.savefig(output / name, dpi=200, facecolor="white")
        names.append(name)
    plt.close(figure)
    return names


def _write_report(output: Path, performance: pd.DataFrame, summary: pd.DataFrame,
                  interventions: pd.DataFrame, changes: pd.DataFrame, bootstrap: dict) -> None:
    means = performance.groupby("model")[list(METRICS)].mean()
    delta = means.loc["geometry4200", "accuracy"] - means.loc["geometry3378", "accuracy"]
    lines = [
        "# Joint geometry correction with all development references", "",
        f"Expanding the reference bank from 3,378 to 4,200 images changes mean accuracy from "
        f"{100 * means.loc['geometry3378', 'accuracy']:.4f}% to {100 * means.loc['geometry4200', 'accuracy']:.4f}% "
        f"({100 * delta:+.4f} percentage points) across the same four expert seeds. "
        f"Fusion alone gives {100 * means.loc['fusion', 'accuracy']:.4f}% and the frozen RF correction "
        f"gives {100 * means.loc['rf', 'accuracy']:.4f}%.", "",
        "This bank expansion was requested after the earlier negative cleaned-test result was known. "
        "It is an adaptive descriptive follow-up, not independent confirmation. No threshold or expert is refitted.", "",
        "![Matched six-method comparison](comparison.png)", "",
        "## Fixed rule and the single declared change", "",
        "The previous bank contains 3,378 development-training images from 1,300 plots. "
        "The expanded bank contains all 4,200 development images from 1,625 plots, adding the 822 "
        "development-validation images from 325 plots. Both banks exclude cleaned-test images and plots. "
        "The added labels serve only as reference-class labels; no test label enters the reference bank. "
        "All development samples are included in the expanded bank: reference samples are not selected "
        "according to their effects on test results. The prior bank's result remains in this comparison, "
        "and its original scripts and artifacts are preserved.", "",
        "The geometry definition remains k=1 cosine distance to the nearest reference plot of each candidate "
        "class, taking the minimum distance across that plot's photos. Frozen pretrained SigLIP2 image features "
        "and regional geo features are independently L2-normalized. For modality m:", "",
        "$$A_m(x)=\\log\\frac{d_m(c_r)+10^{-8}}{d_m(c_g)+10^{-8}}.$$", "",
        "Here c_r is fusion's predicted habitat and c_g is geo's predicted habitat. Switch fusion to geo exactly "
        "when their predictions differ, both scores are finite, A_image > 0 and A_geo > 0. "
        "Keep fusion for ties and unavailable scores. The score definition, epsilon, sign thresholds, "
        "expert outputs and RF policy are unchanged.", "",
        "All six methods use the identical cleaned-test cohort of 1,347 images from 531 plots and expert seeds 1–4. "
        "The experts are the historical final-epoch-50 heads fitted on all 4,200 development images. "
        "The unchanged RF policy is `relaxed__rf__primary_full727__plots_20__harm_0p7`, using all 727 features, "
        "as in the [September 15 report](../2026-09-15_geo_helpfulness_router/report.md). "
        "The earlier validation study used experts fitted on only 3,378 images; those are not the experts evaluated here.", "",
        "## Matched overall classification", "",
        "Entries are mean [minimum, maximum] across four seeds. Accuracy is a percentage; F1 and MCC use native scales.", "",
        "| Method | Accuracy (%) | Weighted F1 | Macro F1 | MCC |",
        "|---|---:|---:|---:|---:|",
    ]
    indexed = summary.set_index(["model", "metric"])
    for model in MODELS:
        cells = [MODEL_NAMES[model]]
        for metric in METRICS:
            item = indexed.loc[(model, metric)]
            scale = 100 if metric == "accuracy" else 1
            cells.append(f"{item['mean'] * scale:.4f} [{item['minimum'] * scale:.4f}, {item['maximum'] * scale:.4f}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The fixed 18-class ontology is used for every method. Absent-class precision, recall and F1 are zero; "
              "macro F1 includes all 18 classes. Four repeated seed realizations are not four independent test cohorts.", "",
              "[Per-seed metrics](performance_by_seed.csv) · [Mean and ranges](performance_summary.csv) · "
              "[Per-class metrics](performance_by_habitat.csv) · [Detailed metrics](metrics.json)", "",
              "## Which override decisions change?", "",
              "Newly activated means the expanded bank switches to geo where the previous bank retained fusion. "
              "Dropped means a previous override now retains fusion. Kept means both banks switch to geo. "
              "Because the expert candidates are identical, every newly activated or dropped action changes the final predicted habitat.", "",
              "| Seed | Newly activated | Dropped | Kept | Inactive in both | Net correct change |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in changes.to_dict("records"):
        fields = ("training_seed", "newly_activated", "dropped", "kept", "inactive_in_both", "net_correct_change")
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    lines += ["", "Net correct change compares the expanded bank directly with the previous bank. "
              "[Action changes with rescue/harm decomposition](bank_action_changes.csv)", "",
              "## Override outcomes relative to fusion", "",
              "Rescue is wrong to correct; harm is correct to wrong; both-wrong overrides exchange two incorrect "
              "candidates. Counts refer to one seed's predictions on the common images.", "",
              "| Method | Seed | Overrides | Rescues | Harms | Both wrong | Net rescue − harm |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("geometry3378", "geometry4200", "rf"):
        for row in interventions.loc[interventions.method.eq(model)].to_dict("records"):
            cells = [MODEL_NAMES[model], str(row["training_seed"])]
            cells += [str(row[field]) for field in ("acted", "rescues", "harms", "both_wrong", "net_rescue_minus_harm")]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "[Complete intervention counts](interventions.csv)", "",
              "## Paired plot-bootstrap comparisons", "",
              "The same 2,000 habitat-stratified whole-plot draws (seed 20260930) are used for every comparison. "
              "Every sampled plot retains all photos and all four seeds. Metrics are recomputed within each seed "
              "before paired differences are averaged across seeds. Intervals are conditional 95% percentile "
              "intervals using linear quantiles, conditional on these fixed experts, policies, banks and observed "
              "habitat strata. They do not account for previous test inspection or the decision to expand the bank "
              "after seeing the earlier result.", "",
              "| Expanded bank minus | Metric | Point estimate | Conditional 95% interval |",
              "|---|---|---:|---:|"]
    for result in bootstrap["comparisons"]:
        baseline = result["comparison"].removeprefix("geometry4200_minus_")
        accuracy = result["metric"] == "accuracy"
        name = "Accuracy (percentage points)" if accuracy else "Weighted F1 (native scale)"
        scale = 100 if accuracy else 1
        lines.append(f"| {MODEL_NAMES[baseline]} | {name} | {result['point_estimate'] * scale:+.5f} | "
                     f"[{result['lower'] * scale:+.5f}, {result['upper'] * scale:+.5f}] |")
    lines += ["", "[Bootstrap settings and intervals](paired_bootstrap.json) · "
              "[Matched bootstrap replicates](paired_bootstrap_replicates.csv)", "",
              "## Interpretation", "",
              "This comparison isolates the declared reference-bank expansion while keeping the historical experts, "
              "test cohort and correction rule fixed. Adding references can change either candidate distance, "
              "so it can add or remove overrides. A larger reference bank is not guaranteed to improve routing. "
              "The expanded bank includes previously analyzed validation data, and the cleaned test set had already "
              "been examined before this follow-up was requested. This adaptive analysis provides descriptive "
              "evidence and requires new independent data for confirmation.", "",
              "Query labels are used only to evaluate predictions and define bootstrap habitat strata. "
              "Neighborhoods remain sensitive to class coverage, reference photos per plot and repeated geo vectors. "
              "The frozen image cache retains the historical preprocessing limitations documented in the "
              "[embedding study](../2026-09-29_embedding_geometry/report.md).", ""]
    (output / "report.md").write_text("\n".join(lines))


def write_results(output: Path, rows: pd.DataFrame, declaration: dict) -> dict:
    """Write the fixed-rule comparison between the two declared reference banks."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = _validate(rows)
    performance, summary, per_class, details = _classification_tables(data)
    interventions, changes = _interventions_and_changes(data)
    bootstrap, replicates = paired_plot_bootstrap(data)
    indexed = performance.set_index(["training_seed", "model"])
    for item in bootstrap["comparisons"]:
        baseline = item["comparison"].removeprefix("geometry4200_minus_")
        metric = item["metric"]
        expected = np.mean([indexed.loc[(seed, "geometry4200"), metric] - indexed.loc[(seed, baseline), metric] for seed in SEEDS])
        if not np.isclose(expected, item["point_estimate"], atol=1e-12, rtol=0):
            raise ValueError("Bootstrap point estimates differ from canonical classification metrics")
    tables = {
        "performance_by_seed.csv": performance, "performance_summary.csv": summary,
        "performance_by_habitat.csv": per_class, "interventions.csv": interventions,
        "bank_action_changes.csv": changes, "paired_bootstrap_replicates.csv": replicates,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    _save_json(output / "metrics.json", {
        "declaration": declaration, "image_count": 1347, "plot_count": 531, "expert_seeds": list(SEEDS),
        "fixed_class_count": 18, "models": MODEL_NAMES, "per_seed": details,
        "mean_and_ranges": _records(summary), "interventions": _records(interventions),
        "bank_action_changes": _records(changes),
    })
    _save_json(output / "paired_bootstrap.json", bootstrap)
    figures = _comparison_figure(output, performance)
    _write_report(output, performance, summary, interventions, changes, bootstrap)
    return {
        "images": 1347, "plots": 531, "seeds": list(SEEDS),
        "performance_by_seed": _records(performance), "performance_summary": _records(summary),
        "interventions": _records(interventions), "bank_action_changes": _records(changes),
        "bootstrap": bootstrap,
        "output_files": [*tables, "metrics.json", "paired_bootstrap.json", "report.md", *figures],
    }

"""Report a fixed K5 geometry policy on caller-supplied matched cohorts."""

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

from joint_geometry_test_report import plot_confusion_aggregates
from multimodal.geo_helpfulness_protocol import FIXED_CLASS_ONTOLOGY
from multimodal.geo_helpfulness_router_metrics import classification_metrics


SEEDS = (1, 2, 3, 4)
METRICS = ("accuracy", "weighted_f1", "macro_f1", "mcc")
REPLICATES = 2000
BOOTSTRAP_SEED = 20260930
MODELS = {
    "validation": {
        "image": "image_pred", "geo": "geo_pred", "fusion": "raw_pred",
        "k1": "k1_pred", "k5": "k5_pred", "k1_supported": "k1_supported_pred",
    },
    "test": {
        "image": "image_pred", "geo": "geo_pred", "fusion": "raw_pred", "rf": "rf_pred",
        "g3378_k1": "g3378_k1_pred", "g3378_k5": "g3378_k5_pred",
        "g3378_k1_supported": "g3378_k1_supported_pred",
        "g4200_k1": "g4200_k1_pred", "g4200_k5": "g4200_k5_pred",
        "g4200_k1_supported": "g4200_k1_supported_pred",
    },
}
NAMES = {
    "image": "Image expert", "geo": "Geo expert", "fusion": "Fusion expert", "rf": "Fusion + frozen RF",
    "k1": "Geometry K1 (3,378 refs)", "k5": "Geometry K5 (3,378 refs)",
    "k1_supported": "K1, five-plot support required (3,378 refs)",
    "g3378_k1": "Geometry K1 (3,378 refs)", "g3378_k5": "Geometry K5 (3,378 refs)",
    "g3378_k1_supported": "K1, five-plot support required (3,378 refs)",
    "g4200_k1": "Geometry K1 (4,200 refs)", "g4200_k5": "Geometry K5 (4,200 refs)",
    "g4200_k1_supported": "K1, five-plot support required (4,200 refs)",
}
CONTRASTS = {
    "validation": [("k5", baseline) for baseline in ("fusion", "k1", "k1_supported")],
    "test": [(f"g{bank}_k5", baseline) for bank in (3378, 4200)
             for baseline in ("fusion", "rf", f"g{bank}_k1", f"g{bank}_k1_supported")],
}
EXPECTED = {"validation": (822, 325), "test": (1347, 531)}


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _geometry_groups(cohort: str) -> list[tuple[str, str, str, str, str]]:
    """Return bank name, K1 model, K5 model, supported control, support column."""
    if cohort == "validation":
        return [("3378", "k1", "k5", "k1_supported", "valid_k5")]
    return [(str(bank), f"g{bank}_k1", f"g{bank}_k5", f"g{bank}_k1_supported", f"valid_k5_{bank}")
            for bank in (3378, 4200)]


def _validate(rows: pd.DataFrame, cohort: str) -> pd.DataFrame:
    identity = ["row_uid", "file", "plot_idx", "canonical_l3_id", "label_id_dense", "label_name"]
    required = {*identity, "training_seed", *MODELS[cohort].values()}
    for _, k1, k5, control, support in _geometry_groups(cohort):
        required.update({f"{k1}_acted", f"{k5}_acted", f"{control}_acted", support})
    if cohort == "validation":
        required.update({"A_image", "A_geo", "reference_plots_fusion_class", "reference_plots_geo_class"})
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"Missing {cohort} reporting columns: {missing}")
    data = rows.sort_values(["training_seed", "row_uid"]).reset_index(drop=True).copy()
    if set(data.training_seed) != set(SEEDS) or data.duplicated(["training_seed", "row_uid"]).any():
        raise ValueError(f"Expected four unique matched seed/query cohorts: {cohort}")
    first = data.loc[data.training_seed.eq(SEEDS[0]), identity].reset_index(drop=True)
    n_images, n_plots = EXPECTED[cohort]
    if len(first) != n_images or first.plot_idx.nunique() != n_plots:
        raise ValueError(f"Unexpected {cohort} cohort dimensions")
    for seed in SEEDS:
        if not data.loc[data.training_seed.eq(seed), identity].reset_index(drop=True).equals(first):
            raise ValueError(f"Query metadata differ across {cohort} seeds")
    if first.groupby("plot_idx").canonical_l3_id.nunique().gt(1).any():
        raise ValueError("A plot spans multiple habitat strata")
    ontology = {dense: (canonical, name) for dense, canonical, name in FIXED_CLASS_ONTOLOGY}
    for item in first[["label_id_dense", "canonical_l3_id", "label_name"]].drop_duplicates().itertuples(index=False):
        if ontology.get(int(item.label_id_dense)) != (int(item.canonical_l3_id), item.label_name):
            raise ValueError("Supplied labels differ from the fixed ontology")
    for column in ["label_id_dense", *MODELS[cohort].values()]:
        values = data[column].to_numpy()
        if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= 18):
            raise ValueError(f"Invalid class IDs: {cohort}/{column}")
    disagreement = data.raw_pred.ne(data.geo_pred).to_numpy()
    if cohort == "test" and "rf_acted" not in data:
        data["rf_acted"] = data.rf_pred.ne(data.raw_pred)
    policies = [model for model in MODELS[cohort] if model not in ("image", "geo", "fusion")]
    for model in policies:
        acted = data[f"{model}_acted"].to_numpy()
        if acted.dtype.kind != "b" or np.any(acted & ~disagreement):
            raise ValueError(f"Invalid disagreement-only Boolean actions: {cohort}/{model}")
        expected = np.where(acted, data.geo_pred, data.raw_pred)
        if not np.array_equal(data[MODELS[cohort][model]], expected):
            raise ValueError(f"Predictions do not match geo overrides: {cohort}/{model}")
    for _, k1, k5, control, support in _geometry_groups(cohort):
        available = data[support].to_numpy()
        if available.dtype.kind != "b":
            raise ValueError(f"Expected Boolean reference support: {support}")
        if np.any(data[f"{k5}_acted"].to_numpy() & ~available):
            raise ValueError("K5 must retain fusion when references are unsupported")
        expected_control = data[f"{k1}_acted"].to_numpy() & available
        if not np.array_equal(data[f"{control}_acted"], expected_control):
            raise ValueError("K1-supported control must retain exactly supported K1 actions")
    if cohort == "validation":
        counts_supported = data.reference_plots_fusion_class.ge(5) & data.reference_plots_geo_class.ge(5)
        if not np.array_equal(data.valid_k5, counts_supported):
            raise ValueError("Validation reference-count coverage differs from valid_k5")
        finite = np.isfinite(data[["A_image", "A_geo"]].to_numpy(dtype=float)).all(axis=1)
        expected_action = disagreement & data.valid_k5.to_numpy() & finite & data.A_image.gt(0).to_numpy() & data.A_geo.gt(0).to_numpy()
        if not np.array_equal(data.k5_acted, expected_action):
            raise ValueError("Validation K5 actions differ from the frozen joint-positive rule")
    return data


def _tables(cohorts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    performance, per_class, interventions, coverage, detailed = [], [], [], [], {}
    ontology = {dense: (canonical, name) for dense, canonical, name in FIXED_CLASS_ONTOLOGY}
    for cohort, rows in cohorts.items():
        detailed[cohort] = {}
        for seed, part in rows.groupby("training_seed", sort=True):
            detailed[cohort][str(seed)] = {}
            truth = part.label_id_dense.to_numpy()
            raw_correct, geo_correct = part.raw_pred.to_numpy() == truth, part.geo_pred.to_numpy() == truth
            disagreement = part.raw_pred.ne(part.geo_pred).to_numpy()
            for bank, k1, k5, control, support in _geometry_groups(cohort):
                valid = part[support].to_numpy()
                coverage.append({
                    "cohort": cohort, "training_seed": int(seed), "reference_bank": bank,
                    "row_count": len(part), "supported_images": int(valid.sum()),
                    "unsupported_images": int((~valid).sum()), "disagreements": int(disagreement.sum()),
                    "supported_disagreements": int((valid & disagreement).sum()),
                    "unsupported_disagreements": int((~valid & disagreement).sum()),
                    "k1_actions_removed_by_support_requirement": int((part[f"{k1}_acted"].to_numpy() & ~valid).sum()),
                    "k5_retains_unsupported_fusion": True,
                })
            for model, prediction_column in MODELS[cohort].items():
                result = classification_metrics(part[prediction_column].to_numpy(), truth, 18)
                detailed[cohort][str(seed)][model] = result
                performance.append({"cohort": cohort, "training_seed": int(seed), "model": model,
                                    "row_count": len(part), **{metric: result[metric] for metric in METRICS}})
                for item in result["per_class"]:
                    canonical, name = ontology[item["class_id"]]
                    per_class.append({"cohort": cohort, "training_seed": int(seed), "model": model,
                                      "canonical_l3_id": canonical, "label_name": name, **item})
                if model in ("image", "geo", "fusion"):
                    continue
                acted = part[f"{model}_acted"].to_numpy()
                rescue = int((acted & ~raw_correct & geo_correct).sum())
                harm = int((acted & raw_correct & ~geo_correct).sum())
                both_wrong = int((acted & ~raw_correct & ~geo_correct).sum())
                observed_net = int((part[prediction_column].to_numpy() == truth).sum() - raw_correct.sum())
                if rescue + harm + both_wrong != int(acted.sum()) or rescue - harm != observed_net:
                    raise ValueError("Interventions do not reproduce classification changes")
                interventions.append({
                    "cohort": cohort, "training_seed": int(seed), "model": model, "row_count": len(part),
                    "disagreements": int(disagreement.sum()), "acted": int(acted.sum()),
                    "rescues": rescue, "harms": harm, "both_wrong": both_wrong,
                    "net_rescue_minus_harm": rescue - harm, "coverage": float(acted.mean()),
                    "rescue_opportunities": int((~raw_correct & geo_correct).sum()),
                    "harm_opportunities": int((raw_correct & ~geo_correct).sum()),
                })
    performance_frame = pd.DataFrame(performance)
    summary = []
    for (cohort, model), part in performance_frame.groupby(["cohort", "model"], sort=False):
        for metric in METRICS:
            values = part[metric].dropna()
            summary.append({"cohort": cohort, "model": model, "metric": metric,
                            "mean": values.mean(), "minimum": values.min(), "maximum": values.max(),
                            "seed_count": len(values)})
    return (performance_frame, pd.DataFrame(summary), pd.DataFrame(per_class),
            pd.DataFrame(interventions), pd.DataFrame(coverage), detailed)


def _bootstrap(rows: pd.DataFrame, cohort: str) -> tuple[dict, pd.DataFrame]:
    model_set = {model for pair in CONTRASTS[cohort] for model in pair}
    columns = {model: column for model, column in MODELS[cohort].items() if model in model_set}
    counts, plots, seeds, models = plot_confusion_aggregates(rows, columns)
    support = counts[0, 0].sum(axis=-1)
    if not np.array_equal(counts.sum(axis=-1), np.broadcast_to(support, counts.shape[:-1])):
        raise ValueError("Class support differs across paired methods or seeds")
    strata = [np.flatnonzero(plots.canonical_l3_id.to_numpy() == label)
              for label in sorted(plots.canonical_l3_id.unique())]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.zeros((REPLICATES, len(plots)), dtype=np.int64)
    for draw in weights:
        for indices in strata:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            draw += np.bincount(sampled, minlength=len(plots))
    diagonal = np.diagonal(counts, axis1=-2, axis2=-1).transpose(2, 0, 1, 3)
    predicted = counts.sum(axis=-2).transpose(2, 0, 1, 3)
    shape = (REPLICATES, len(seeds), len(models), 18)
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
    comparisons, replicates = [], {"replicate": np.arange(REPLICATES)}
    for metric, draws, points in (("accuracy", boot_accuracy, observed_accuracy),
                                  ("weighted_f1", boot_weighted_f1, observed_weighted_f1)):
        for target, baseline in CONTRASTS[cohort]:
            ti, bi = models.index(target), models.index(baseline)
            delta = (draws[:, :, ti] - draws[:, :, bi]).mean(axis=1)
            point = float((points[:, ti] - points[:, bi]).mean())
            low, high = np.quantile(delta, [0.025, 0.975], method="linear")
            name = f"{target}_minus_{baseline}"
            replicates[f"{name}_{metric}"] = delta
            comparisons.append({"cohort": cohort, "target": target, "baseline": baseline,
                                "comparison": name, "metric": metric, "point_estimate": point,
                                "lower": float(low), "upper": float(high)})
    settings = {
        "cohort": cohort, "replicates": REPLICATES, "random_seed": BOOTSTRAP_SEED,
        "unit": "whole plot retaining all photos and all four seeds", "stratify_by": "canonical_l3_id",
        "plots": len(plots), "images": len(rows) // len(seeds),
        "interval": "95% percentile; linear quantiles; paired mean seed metric difference",
        "conditional_on": "fixed experts, reference banks, rules and observed habitat strata",
        "adaptive_exploration_accounted_for": False, "multiple_comparison_adjustment": False,
        "resample_multiplicity_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
        "comparisons": comparisons,
    }
    return settings, pd.DataFrame(replicates)


def _plot(output: Path, performance: pd.DataFrame) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 6))
    for cohort, axis in zip(("validation", "test"), axes):
        part = performance.loc[performance.cohort.eq(cohort)]
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#edf0f2", linewidth=0.7)
        axis.set_axisbelow(True)
        values_shown = []
        bank_handles = []
        for bank_index, (bank, k1, k5, _, _) in enumerate(_geometry_groups(cohort)):
            color = "#00897b" if bank == "3378" else "#6b63ac"
            means = []
            offset = 0.035 * (bank_index * 2 - 1) if cohort == "test" else 0.0
            for position, model in enumerate((k1, k5)):
                values = part.loc[part.model.eq(model)].sort_values("training_seed").accuracy.to_numpy() * 100
                means.append(float(values.mean()))
                values_shown.extend(values.tolist())
                axis.scatter(position + offset + np.linspace(-0.035, 0.035, len(values)), values,
                             s=28, color=color, alpha=0.35, linewidths=0, zorder=2)
            axis.plot(np.arange(2) + offset, means, color=color, marker="D", markersize=6,
                      linewidth=1.8, zorder=3)
            bank_handles.append(Line2D([], [], color=color, marker="D", label=f"{int(bank):,} references"))
        for baseline, color, style in (("fusion", "#596675", "--"), ("rf", "#d48742", ":")):
            if baseline not in MODELS[cohort]:
                continue
            value = float(part.loc[part.model.eq(baseline)].accuracy.mean()) * 100
            values_shown.append(value)
            axis.axhline(value, color=color, linestyle=style, linewidth=1.4)
            bank_handles.append(Line2D([], [], color=color, linestyle=style,
                                       label="Fusion mean" if baseline == "fusion" else "Frozen RF mean"))
        axis.legend(handles=bank_handles, fontsize=9, frameon=False, loc="best")
        axis.set_title("Development validation" if cohort == "validation" else "Previously examined cleaned test", fontsize=12, pad=14)
        axis.set_xticks([0, 1], ["K1", "K5"])
        axis.set_xlim(-0.22, 1.22)
        span = max(1.0, max(values_shown) - min(values_shown))
        axis.set_ylim(min(values_shown) - 0.18 * span, max(values_shown) + 0.38 * span)
        axis.set_ylabel("Full-cohort habitat accuracy (%)", fontsize=11)
    figure.suptitle("Mean of five nearest reference plots: fixed correction rule", fontsize=15, y=0.98)
    figure.text(0.5, 0.915, "Retain fusion when either candidate class has fewer than five reference plots",
                ha="center", fontsize=10, color="#596675")
    figure.text(0.5, 0.081, "Diamonds/lines: mean across four seeds. Faint dots: individual seeds. Supported-K1 controls are reported in the tables.",
                ha="center", fontsize=9, color="#596675")
    figure.text(0.5, 0.035, "The same images recur across seeds; seed spread is not an interval. Test comparisons are adaptive follow-ups.",
                ha="center", fontsize=9, color="#596675")
    figure.subplots_adjust(left=0.075, right=0.98, top=0.81, bottom=0.19, wspace=0.24)
    names = []
    for extension in ("png", "pdf"):
        name = f"comparison.{extension}"
        figure.savefig(output / name, dpi=200, facecolor="white")
        names.append(name)
    plt.close(figure)
    return names


def _report(output: Path, summary: pd.DataFrame, interventions: pd.DataFrame,
            coverage: pd.DataFrame, bootstrap: dict) -> None:
    indexed = summary.set_index(["cohort", "model", "metric"])
    lines = [
        "# Joint geometry correction with five reference plots", "",
        "This check replaces each candidate class's nearest-plot distance (K1) with the mean distance "
        "to its five nearest distinct reference plots (K5), while preserving the correction rule. "
        "Every reported primary metric includes the full cohort, including samples where K5 falls back to fusion.", "",
        "![K1 versus K5 full-cohort accuracy](comparison.png)", "",
        "## Fixed policy and support control", "",
        "Within each modality, reference-plot distance is the minimum cosine distance over that plot's photos. "
        "K5 averages the distances to the five nearest distinct plots of the candidate class. "
        "It does not use five photos from one plot. Image and regional geo features are independently L2-normalized; "
        "the image encoder remains frozen pretrained SigLIP2. For each modality m:", "",
        "$$A_m(x)=\\log\\frac{d_m(c_r)+10^{-8}}{d_m(c_g)+10^{-8}}.$$", "",
        "Switch fusion's class c_r to geo's class c_g only when the experts disagree, both scores are finite "
        "and A_image > 0 and A_geo > 0. K5 retains fusion if either candidate class has fewer than five "
        "reference plots; ties and unavailable scores also retain fusion. No new expert or router is fitted, "
        "and no sign threshold, epsilon or class-specific exception is changed.", "",
        "The **K1 with five-plot support required** control uses the original K1 action only where K5 has "
        "sufficient reference support, and otherwise retains fusion. Consequently K5 minus ordinary K1 "
        "combines a neighborhood change and a support/fallback change; K5 minus supported K1 compares "
        "the neighborhood change under the same fallback requirement. Both comparisons use the full cohort, "
        "rather than discarding unsupported queries.", "",
        "Development validation contains 822 images from 325 plots, with experts fitted on 3,378 "
        "development-training images and the same 3,378-image/1,300-plot reference bank. "
        "The cleaned test comparison contains 1,347 images from 531 plots and the historical final-epoch-50 "
        "experts fitted on all 4,200 development images. Its two reference banks contain 3,378 images/1,300 plots "
        "and 4,200 images/1,625 plots. All four seeds use identical query identities within each cohort. "
        "Validation and test therefore use different fitted expert sets.", "",
        "The test RF comparator remains `relaxed__rf__primary_full727__plots_20__harm_0p7`, as in the "
        "[September 15 report](../2026-09-15_geo_helpfulness_router/report.md). "
        "The prior K1 results are retained. Reference membership is fixed by the declared development split, "
        "with no selection of reference samples according to test effects.", "",
        "## Overall classification", "",
        "Entries are mean [minimum, maximum] across the four seeds. Accuracy is a percentage; F1 and MCC "
        "use native scales. Metrics use the fixed 18-class ontology, with zero precision/recall/F1 for absent "
        "classes and all 18 classes included in macro F1.", "",
    ]
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
    lines += ["[Per-seed metrics](performance_by_seed.csv) · [Mean/range metrics](performance_summary.csv) · "
              "[Per-class metrics](performance_by_habitat.csv) · [Detailed records](metrics.json)", "",
              "## Missing reference coverage", "",
              "These are per-seed counts. A query with insufficient class support stays in the evaluation and retains fusion.", "",
              "| Cohort | Bank size | Seed | Unsupported images | Unsupported disagreements | K1 actions removed by support requirement |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in coverage.to_dict("records"):
        fields = ("cohort", "reference_bank", "training_seed", "unsupported_images", "unsupported_disagreements",
                  "k1_actions_removed_by_support_requirement")
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    lines += ["", "[Complete reference coverage](reference_coverage.csv)", "",
              "## Override outcomes", "",
              "Rescue changes a wrong prediction to correct; harm changes a correct prediction to wrong; "
              "both-wrong overrides replace one incorrect candidate with another. The following entries are "
              "mean [minimum, maximum] counts across seeds, not counts of independent images.", "",
              "| Cohort | Policy | Overrides | Rescues | Harms | Both wrong | Net rescue − harm |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for (cohort, model), part in interventions.groupby(["cohort", "model"], sort=False):
        cells = [cohort, NAMES[model]]
        for column in ("acted", "rescues", "harms", "both_wrong", "net_rescue_minus_harm"):
            values = part[column]
            cells.append(f"{values.mean():.2f} [{values.min()}, {values.max()}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "[Per-seed intervention counts and coverage](interventions.csv)", "",
              "## Conditional paired uncertainty", "",
              "For each cohort, 2,000 habitat-stratified whole-plot bootstrap draws use seed 20260930. "
              "Each selected plot retains all photos and all four seeds; the same draw pairs all models in "
              "that cohort. Metrics are recomputed within each seed before paired differences are averaged. "
              "The reported 95% percentile intervals use linear quantiles and are conditional on these fixed "
              "experts, reference banks, rules and observed habitat strata. They do not account for earlier "
              "exploration, prior test inspection, the adaptive choice to examine K5, or multiple comparisons. "
              "Sparse strata provide limited information about population variability.", "",
              "| Cohort | K5 policy | Minus comparator | Metric | Point estimate | Conditional 95% interval |",
              "|---|---|---|---|---:|---:|"]
    for cohort in ("validation", "test"):
        for item in bootstrap[cohort]["comparisons"]:
            accuracy = item["metric"] == "accuracy"
            scale = 100 if accuracy else 1
            name = "Accuracy (pp)" if accuracy else "Weighted F1"
            lines.append(f"| {cohort} | {NAMES[item['target']]} | {NAMES[item['baseline']]} | {name} | "
                         f"{item['point_estimate'] * scale:+.5f} | "
                         f"[{item['lower'] * scale:+.5f}, {item['upper'] * scale:+.5f}] |")
    lines += ["", "[Bootstrap settings and intervals](paired_bootstrap.json) · "
              "[Validation replicates](bootstrap_replicates_validation.csv) · "
              "[Test replicates](bootstrap_replicates_test.csv)", "",
              "## Interpretation limits", "",
              "The user requested this fixed K5 check after earlier geometry results had been inspected. "
              "The cleaned test set and prior bank-size results were already known; this is an adaptive "
              "follow-up, not an unbiased independent confirmation. Validation has also informed the geometry "
              "study. This report evaluates the declared alternatives without selecting or promoting a policy "
              "from their outcomes. New independent data are required for confirmation.", "",
              "All primary scores are full-cohort results with fallback. Repeated expert seeds are not "
              "independent images. The ratio compares only the fusion and geo candidates; both can have weak "
              "absolute support. Distances remain affected by reference class coverage, photo multiplicity "
              "and repeated geo embeddings. The frozen image cache retains the historical preprocessing "
              "limitations documented in the [embedding study](../2026-09-29_embedding_geometry/report.md).", ""]
    (output / "report.md").write_text("\n".join(lines))


def write_results(output: Path, validation: pd.DataFrame, test: pd.DataFrame, declaration: dict) -> dict:
    """Write full-cohort K5 metrics and the fixed support-control comparisons."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cohorts = {"validation": _validate(validation, "validation"), "test": _validate(test, "test")}
    if set(cohorts["validation"].row_uid) & set(cohorts["test"].row_uid):
        raise ValueError("Validation and test query identities overlap")
    if set(cohorts["validation"].plot_idx) & set(cohorts["test"].plot_idx):
        raise ValueError("Validation and test query plots overlap")
    performance, summary, per_class, interventions, coverage, details = _tables(cohorts)
    tables = {"performance_by_seed.csv": performance, "performance_summary.csv": summary,
              "performance_by_habitat.csv": per_class, "interventions.csv": interventions,
              "reference_coverage.csv": coverage}
    bootstrap = {}
    indexed = performance.set_index(["cohort", "training_seed", "model"])
    for cohort, rows in cohorts.items():
        settings, replicates = _bootstrap(rows, cohort)
        bootstrap[cohort] = settings
        tables[f"bootstrap_replicates_{cohort}.csv"] = replicates
        for item in settings["comparisons"]:
            expected = np.mean([indexed.loc[(cohort, seed, item["target"]), item["metric"]]
                                - indexed.loc[(cohort, seed, item["baseline"]), item["metric"]] for seed in SEEDS])
            if not np.isclose(expected, item["point_estimate"], atol=1e-12, rtol=0):
                raise ValueError("Bootstrap point estimates differ from canonical classification metrics")
    if sum(len(value["comparisons"]) for value in bootstrap.values()) != 22:
        raise ValueError("Expected 11 contrasts for each of accuracy and weighted F1")
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    _json(output / "metrics.json", {
        "declaration": declaration, "cohort_sizes": EXPECTED, "expert_seeds": list(SEEDS),
        "fixed_class_count": 18, "model_names": NAMES, "per_seed": details,
        "mean_and_ranges": _records(summary), "interventions": _records(interventions),
        "reference_coverage": _records(coverage),
    })
    _json(output / "paired_bootstrap.json", bootstrap)
    figures = _plot(output, performance)
    _report(output, summary, interventions, coverage, bootstrap)
    return {"cohort_sizes": {key: {"images": size[0], "plots": size[1]} for key, size in EXPECTED.items()},
            "seeds": list(SEEDS), "performance_by_seed": _records(performance),
            "performance_summary": _records(summary), "interventions": _records(interventions),
            "reference_coverage": _records(coverage), "bootstrap": bootstrap,
            "output_files": [*tables, "metrics.json", "paired_bootstrap.json", "report.md", *figures]}

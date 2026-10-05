"""Report frozen validation-selected geometry thresholds from supplied rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_protocol import FIXED_CLASS_ONTOLOGY
from multimodal.geo_helpfulness_router_heuristics import validate_threshold_specification
from multimodal.geo_helpfulness_router_metrics import classification_metrics


SEEDS = (1, 2, 3, 4)
METRICS = ("accuracy", "weighted_f1", "macro_f1", "mcc")
COHORTS = ("validation", "test_3378", "test_4200")
SIZES = {"validation": (822, 325), "test_3378": (1347, 531), "test_4200": (1347, 531)}
COHORT_NAMES = {"validation": "Development validation", "test_3378": "Test: 3,378 references",
                "test_4200": "Test: 4,200 references"}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260930


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _policies(selection: dict) -> dict:
    policies = selection["policies"]
    expected = {f"k{k}__plots_{plots}__harm_0p{numerator}"
                for k in (5, 10) for plots in (20, 10, 0) for numerator in (5, 7, 9)}
    if set(policies) != expected:
        raise ValueError("Expected all 18 declared K/plot-support/harm-cap policy IDs")
    ordered = {}
    for k in (5, 10):
        for plots in (20, 10, 0):
            for numerator in (5, 7, 9):
                name = f"k{k}__plots_{plots}__harm_0p{numerator}"
                policy = policies[name]
                if (policy["k"], policy["minimum_plots"], policy["harm_numerator"],
                    policy["harm_denominator"]) != (k, plots, numerator, 10):
                    raise ValueError(f"Policy metadata differ from ID: {name}")
                if float(policy["minimum_coverage_fraction"]) != 0.01:
                    raise ValueError("Expected the unchanged one-percent validation coverage guard")
                threshold = validate_threshold_specification(policy["threshold_specification"])
                if bool(policy["selected_feasible"]) != (threshold is not None):
                    raise ValueError("Feasibility flag and typed threshold disagree")
                ordered[name] = policy
    return ordered


def _models(cohort: str, policies: dict) -> dict[str, str]:
    models = {"image": "image_pred", "geo": "geo_pred", "fusion": "raw_pred"}
    if cohort != "validation":
        models["rf"] = "rf_pred"
    models.update({"zero_k5": "zero_k5_pred", "zero_k10": "zero_k10_pred"})
    models.update({name: f"{name}_pred" for name in policies})
    return models


def _name(model: str, policies: dict) -> str:
    baseline = {"image": "Image expert", "geo": "Geo expert", "fusion": "Fusion expert",
                "rf": "Frozen RF correction", "zero_k5": "Strict K5, zero threshold",
                "zero_k10": "Strict K10, zero threshold"}
    if model in baseline:
        return baseline[model]
    policy = policies[model]
    return f"K{policy['k']}; ≥{policy['minimum_plots']} plots; H≤{policy['harm_numerator'] / 10:g}R"


def _counts(part: pd.DataFrame, acted: np.ndarray) -> dict:
    truth = part.label_id_dense.to_numpy()
    raw_correct, geo_correct = part.raw_pred.to_numpy() == truth, part.geo_pred.to_numpy() == truth
    rescue = int((acted & ~raw_correct & geo_correct).sum())
    harm = int((acted & raw_correct & ~geo_correct).sum())
    both_wrong = int((acted & ~raw_correct & ~geo_correct).sum())
    count = int(acted.sum())
    if rescue + harm + both_wrong != count:
        raise ValueError("Override outcomes do not sum to the action count")
    return {"row_count": len(part), "acted_count": count, "rescued_count": rescue,
            "harmed_count": harm, "both_wrong_count": both_wrong, "rescued_minus_harmed": rescue - harm,
            "delta_accuracy": (rescue - harm) / len(part), "coverage_fraction": count / len(part),
            "unique_acted_plots": int(part.loc[acted, "plot_idx"].nunique()),
            "observed_harm_per_rescue": harm / rescue if rescue else None}


def _validate(rows: pd.DataFrame, cohort: str, policies: dict) -> pd.DataFrame:
    identity = ["row_uid", "file", "plot_idx", "canonical_l3_id", "label_id_dense", "label_name"]
    models = _models(cohort, policies)
    required = {*identity, "training_seed", *models.values(), "k5_score", "k5_eligible", "k10_score", "k10_eligible"}
    required.update({f"{model}_acted" for model in models if model not in ("image", "geo", "fusion")})
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"Missing {cohort} columns: {missing}")
    data = rows.sort_values(["training_seed", "row_uid"]).reset_index(drop=True).copy()
    if set(data.training_seed) != set(SEEDS) or data.duplicated(["training_seed", "row_uid"]).any():
        raise ValueError("Expected four unique matched expert-seed/query cohorts")
    first = data.loc[data.training_seed.eq(SEEDS[0]), identity].reset_index(drop=True)
    if (len(first), first.plot_idx.nunique()) != SIZES[cohort]:
        raise ValueError(f"Unexpected cohort dimensions: {cohort}")
    for seed in SEEDS:
        if not data.loc[data.training_seed.eq(seed), identity].reset_index(drop=True).equals(first):
            raise ValueError(f"Query metadata differ across {cohort} seeds")
    if first.groupby("plot_idx").canonical_l3_id.nunique().gt(1).any():
        raise ValueError("A query plot spans multiple habitat strata")
    ontology = {dense: (canonical, name) for dense, canonical, name in FIXED_CLASS_ONTOLOGY}
    for label in first[["label_id_dense", "canonical_l3_id", "label_name"]].drop_duplicates().itertuples(index=False):
        if ontology.get(int(label.label_id_dense)) != (int(label.canonical_l3_id), label.label_name):
            raise ValueError("Labels differ from the fixed ontology")
    for column in ["label_id_dense", *models.values()]:
        values = data[column].to_numpy()
        if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= 18):
            raise ValueError(f"Invalid class IDs: {cohort}/{column}")
    disagreement = data.raw_pred.ne(data.geo_pred).to_numpy()
    for k in (5, 10):
        eligible = data[f"k{k}_eligible"].to_numpy()
        score = data[f"k{k}_score"].to_numpy(dtype=float)
        if eligible.dtype.kind != "b" or np.any(eligible & ~np.isfinite(score)):
            raise ValueError("Strict-K eligibility must be Boolean and imply a finite score")
        expected = disagreement & eligible & (score > 0)
        if not np.array_equal(data[f"zero_k{k}_acted"], expected):
            raise ValueError("Zero-threshold baseline differs from its declared strict-K rule")
    for model, column in models.items():
        if model in ("image", "geo", "fusion"):
            continue
        acted = data[f"{model}_acted"].to_numpy()
        if acted.dtype.kind != "b" or np.any(acted & ~disagreement):
            raise ValueError("Policy actions must be Boolean and restricted to disagreements")
        if model in policies:
            policy = policies[model]
            threshold = validate_threshold_specification(policy["threshold_specification"])
            expected = np.zeros(len(data), dtype=bool)
            if threshold is not None:
                expected = disagreement & data[f"k{policy['k']}_eligible"].to_numpy() & data[f"k{policy['k']}_score"].gt(threshold).to_numpy()
            if not np.array_equal(acted, expected):
                raise ValueError(f"Actions do not use the frozen threshold: {cohort}/{model}")
        if not np.array_equal(data[column], np.where(acted, data.geo_pred, data.raw_pred)):
            raise ValueError(f"Predictions differ from declared overrides: {cohort}/{model}")
    return data


def _selection_table(validation: pd.DataFrame, policies: dict) -> pd.DataFrame:
    records = []
    for name, policy in policies.items():
        observed = _counts(validation, validation[f"{name}_acted"].to_numpy())
        for key in ("row_count", "acted_count", "rescued_count", "harmed_count", "rescued_minus_harmed", "unique_acted_plots"):
            if observed[key] != policy["pooled"][key]:
                raise ValueError(f"Supplied validation results differ from frozen selection counts: {name}/{key}")
        feasible = (10 * observed["harmed_count"] <= policy["harm_numerator"] * observed["rescued_count"]
                    and observed["acted_count"] * 100 >= len(validation)
                    and observed["unique_acted_plots"] >= policy["minimum_plots"]
                    and observed["rescued_minus_harmed"] > 0)
        if bool(policy["selected_feasible"]) != feasible:
            raise ValueError("Selected-policy feasibility does not reproduce pooled validation guards")
        threshold = validate_threshold_specification(policy["threshold_specification"])
        records.append({"policy_id": name, "k": policy["k"], "minimum_plots": policy["minimum_plots"],
                        "harm_cap": policy["harm_numerator"] / policy["harm_denominator"],
                        "threshold": threshold, "threshold_kind": policy["threshold_specification"]["kind"],
                        "threshold_value_hex": policy["threshold_specification"]["value_hex"],
                        "selected_feasible": policy["selected_feasible"],
                        "selection_reason": policy["selection_reason"],
                        "feasible_candidate_count": policy["feasible_candidate_count"], **observed})
    return pd.DataFrame(records)


def _tables(cohorts: dict[str, pd.DataFrame], policies: dict) -> tuple:
    performance, per_class, interventions, pooled, support, detailed = [], [], [], [], [], {}
    ontology = {dense: (canonical, name) for dense, canonical, name in FIXED_CLASS_ONTOLOGY}
    for cohort, rows in cohorts.items():
        detailed[cohort] = {}
        models = _models(cohort, policies)
        for seed, part in rows.groupby("training_seed", sort=True):
            detailed[cohort][str(seed)] = {}
            disagreement = part.raw_pred.ne(part.geo_pred).to_numpy()
            for k in (5, 10):
                eligible = part[f"k{k}_eligible"].to_numpy()
                support.append({"cohort": cohort, "training_seed": int(seed), "k": k, "row_count": len(part),
                                "eligible_images": int(eligible.sum()), "ineligible_images": int((~eligible).sum()),
                                "eligible_disagreements": int((eligible & disagreement).sum()),
                                "ineligible_disagreements": int((~eligible & disagreement).sum())})
            for model, column in models.items():
                result = classification_metrics(part[column].to_numpy(), part.label_id_dense.to_numpy(), 18)
                detailed[cohort][str(seed)][model] = result
                performance.append({"cohort": cohort, "training_seed": int(seed), "model": model,
                                    "row_count": len(part), **{metric: result[metric] for metric in METRICS}})
                for item in result["per_class"]:
                    canonical, name = ontology[item["class_id"]]
                    per_class.append({"cohort": cohort, "training_seed": int(seed), "model": model,
                                      "canonical_l3_id": canonical, "label_name": name, **item})
                if model in ("image", "geo", "fusion"):
                    continue
                counts = _counts(part, part[f"{model}_acted"].to_numpy())
                observed_net = int((part[column].to_numpy() == part.label_id_dense.to_numpy()).sum()
                                   - part.raw_pred.eq(part.label_id_dense).sum())
                if observed_net != counts["rescued_minus_harmed"]:
                    raise ValueError("Intervention counts do not reproduce classification changes")
                cap = policies[model]["harm_numerator"] if model in policies else None
                interventions.append({"cohort": cohort, "training_seed": int(seed), "model": model, **counts,
                                      "harm_cap": cap / 10 if cap is not None else None,
                                      "observed_harm_cap_violated": (10 * counts["harmed_count"] > cap * counts["rescued_count"])
                                      if cap is not None else None})
        for model in models:
            if model in ("image", "geo", "fusion"):
                continue
            counts = _counts(rows, rows[f"{model}_acted"].to_numpy())
            cap = policies[model]["harm_numerator"] if model in policies else None
            pooled.append({"cohort": cohort, "model": model, **counts,
                           "harm_cap": cap / 10 if cap is not None else None,
                           "observed_harm_cap_violated": (10 * counts["harmed_count"] > cap * counts["rescued_count"])
                           if cap is not None else None})
    performance_frame = pd.DataFrame(performance)
    summary = []
    for (cohort, model), part in performance_frame.groupby(["cohort", "model"], sort=False):
        for metric in METRICS:
            values = part[metric].dropna()
            summary.append({"cohort": cohort, "model": model, "metric": metric, "mean": values.mean(),
                            "minimum": values.min(), "maximum": values.max(), "seed_count": len(values)})
    return (performance_frame, pd.DataFrame(summary), pd.DataFrame(per_class), pd.DataFrame(interventions),
            pd.DataFrame(pooled), pd.DataFrame(support), detailed)


def _bootstrap(rows: pd.DataFrame, cohort: str, policies: dict) -> tuple[dict, pd.DataFrame]:
    contrasts = [(name, baseline) for name, policy in policies.items()
                 for baseline in (["fusion", f"zero_k{policy['k']}"] + (["rf"] if cohort != "validation" else []))]
    needed = {model for pair in contrasts for model in pair}
    columns = {model: column for model, column in _models(cohort, policies).items() if model in needed}
    # One representative for exactly identical predictions across every query and
    # seed; all declared policy IDs and contrasts remain in the saved results.
    representatives, signature_to_model, model_to_rep = {}, {}, {}
    for model, column in columns.items():
        signature = rows[column].to_numpy(dtype="<i8").tobytes()
        representative = signature_to_model.get(signature)
        if representative is None:
            representative = model
            signature_to_model[signature] = representative
            representatives[representative] = column
        model_to_rep[model] = representative
    rep_names = tuple(representatives)
    first = rows.loc[rows.training_seed.eq(SEEDS[0])]
    plots = first[["plot_idx", "canonical_l3_id"]].drop_duplicates().sort_values("plot_idx").reset_index(drop=True)
    plot_index = pd.Series(np.arange(len(plots)), index=plots.plot_idx)
    support = np.zeros((len(plots), 18), dtype=np.float64)
    np.add.at(support, (plot_index.loc[first.plot_idx].to_numpy(), first.label_id_dense.to_numpy()), 1)
    diagonal = np.zeros((len(plots), len(SEEDS), len(rep_names), 18), dtype=np.float64)
    predicted = np.zeros_like(diagonal)
    for seed_index, seed in enumerate(SEEDS):
        part = rows.loc[rows.training_seed.eq(seed)]
        pi = plot_index.loc[part.plot_idx].to_numpy()
        truth = part.label_id_dense.to_numpy()
        for model_index, column in enumerate(representatives.values()):
            pred = part[column].to_numpy()
            correct = pred == truth
            np.add.at(predicted, (pi, seed_index, model_index, pred), 1)
            np.add.at(diagonal, (pi[correct], seed_index, model_index, truth[correct]), 1)
    strata = [np.flatnonzero(plots.canonical_l3_id.to_numpy() == label)
              for label in sorted(plots.canonical_l3_id.unique())]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.zeros((BOOTSTRAP_REPLICATES, len(plots)), dtype=np.int64)
    for draw in weights:
        for indices in strata:
            draw += np.bincount(rng.choice(indices, size=len(indices), replace=True), minlength=len(plots))
    shape = (BOOTSTRAP_REPLICATES, len(SEEDS), len(rep_names), 18)
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
        for target, baseline in contrasts:
            ti, bi = rep_names.index(model_to_rep[target]), rep_names.index(model_to_rep[baseline])
            delta = (draws[:, :, ti] - draws[:, :, bi]).mean(axis=1)
            point = float((points[:, ti] - points[:, bi]).mean())
            lower, upper = np.quantile(delta, [0.025, 0.975], method="linear")
            name = f"{target}_minus_{baseline}"
            replicates[f"{name}_{metric}"] = delta
            comparisons.append({"cohort": cohort, "target": target, "baseline": baseline,
                                "comparison": name, "metric": metric, "point_estimate": point,
                                "lower": float(lower), "upper": float(upper)})
    settings = {"cohort": cohort, "replicates": BOOTSTRAP_REPLICATES, "random_seed": BOOTSTRAP_SEED,
                "unit": "whole plot retaining all photos and all four seeds", "stratify_by": "canonical_l3_id",
                "images": SIZES[cohort][0], "plots": len(plots),
                "interval": "95% paired percentile; linear quantiles; mean seed metric difference",
                "conditional_on": "fixed experts, selected thresholds, reference bank and observed habitat strata",
                "threshold_selection_uncertainty_accounted_for": False,
                "adaptive_exploration_accounted_for": False, "multiple_comparison_adjustment": False,
                "unique_prediction_vectors": len(rep_names), "reported_models": len(columns),
                "prediction_representatives": model_to_rep,
                "resample_multiplicity_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
                "comparisons": comparisons}
    return settings, pd.DataFrame(replicates)


def _comparison_table(summary: pd.DataFrame, pooled: pd.DataFrame, policies: dict) -> pd.DataFrame:
    index = summary.set_index(["cohort", "model", "metric"])
    outcomes = pooled.set_index(["cohort", "model"])
    records = []
    for cohort in COHORTS:
        for name, policy in policies.items():
            row = {"cohort": cohort, "policy_id": name, "k": policy["k"],
                   "minimum_plots": policy["minimum_plots"], "harm_cap": policy["harm_numerator"] / 10,
                   "threshold": validate_threshold_specification(policy["threshold_specification"]),
                   "selected_feasible_on_validation": policy["selected_feasible"]}
            for metric in METRICS:
                item = index.loc[(cohort, name, metric)]
                row.update({f"{metric}_{key}": item[key] for key in ("mean", "minimum", "maximum")})
            out = outcomes.loc[(cohort, name)]
            row.update({key: out[key] for key in ("acted_count", "rescued_count", "harmed_count",
                        "rescued_minus_harmed", "coverage_fraction", "unique_acted_plots",
                        "observed_harm_per_rescue", "observed_harm_cap_violated")})
            records.append(row)
    return pd.DataFrame(records)


def _plot(output: Path, performance: pd.DataFrame, policies: dict) -> list[str]:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for cohort, axis in zip(COHORTS, axes):
        part = performance.loc[performance.cohort.eq(cohort)]
        means = part.groupby("model").accuracy.mean() * 100
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#edf0f2", linewidth=0.7)
        axis.set_axisbelow(True)
        shown = []
        for k, color in ((5, "#00897b"), (10, "#6b63ac")):
            ids = [f"k{k}__plots_20__harm_0p{cap}" for cap in (5, 7, 9)]
            values = means.loc[ids].to_numpy()
            shown.extend(values.tolist())
            axis.plot([0.5, 0.7, 0.9], values, color=color, marker="o", markersize=5,
                      linewidth=1.8, label=f"Selected threshold, K{k}")
            zero = float(means[f"zero_k{k}"])
            shown.append(zero)
            axis.axhline(zero, color=color, linestyle=":", linewidth=1.1, alpha=0.65, label=f"Zero threshold, K{k}")
        for baseline, color in (("fusion", "#596675"), ("rf", "#d48742")):
            if baseline not in means:
                continue
            value = float(means[baseline])
            shown.append(value)
            axis.axhline(value, color=color, linestyle="--", linewidth=1.2,
                         label="Fusion" if baseline == "fusion" else "Frozen RF")
        span = max(0.8, max(shown) - min(shown))
        axis.set_ylim(min(shown) - 0.15 * span, max(shown) + 0.6 * span)
        axis.set_xlim(0.47, 0.93)
        axis.set_xticks([0.5, 0.7, 0.9])
        axis.set_xlabel("Validation harm cap (H / R)", fontsize=10)
        axis.set_title(COHORT_NAMES[cohort], fontsize=12, pad=13)
        axis.legend(fontsize=7.5, frameon=False, loc="best")
    axes[0].set_ylabel("Mean full-cohort accuracy (%)", fontsize=11)
    figure.suptitle("Frozen geometry thresholds: primary 20-plot guard", fontsize=16, y=0.98)
    figure.text(0.5, 0.925, "The same validation-selected thresholds transfer to both test reference banks",
                ha="center", color="#596675", fontsize=11)
    figure.text(0.5, 0.06, "Points average four expert seeds. Harm caps constrain pooled validation only; test results are not filtered by observed harm.",
                ha="center", fontsize=9, color="#596675")
    figure.text(0.5, 0.026, "All 18 declared policies, including identical or never-intervene policies, remain in the tables. This is adaptive exploration.",
                ha="center", fontsize=9, color="#596675")
    figure.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.19, wspace=0.23)
    names = []
    for extension in ("png", "pdf"):
        name = f"comparison.{extension}"
        figure.savefig(output / name, dpi=200, facecolor="white")
        names.append(name)
    plt.close(figure)
    return names


def _threshold_text(policy: dict) -> str:
    threshold = validate_threshold_specification(policy["threshold_specification"])
    return "Never intervene" if threshold is None else f"{threshold:.7g}"


def _report(output: Path, summary: pd.DataFrame, comparison: pd.DataFrame,
            selected: pd.DataFrame, policies: dict, bootstrap: dict) -> None:
    lines = ["# Validation-selected strict-K geometry thresholds", ""]
    primary = comparison.loc[comparison.minimum_plots.eq(20)]
    lead = []
    for cohort in COHORTS:
        values = primary.loc[primary.cohort.eq(cohort)].accuracy_mean * 100
        lead.append(f"{COHORT_NAMES[cohort]}: {values.min():.3f}%–{values.max():.3f}%")
    lines += ["Across the six primary policies with a 20-plot validation guard, mean accuracy spans "
              + "; ".join(lead) + ". These ranges summarize all declared primary policies, not a test-selected winner.", "",
              "![Primary threshold policies and fixed baselines](comparison.png)", "",
              "## Rule and validation selection", "",
              "Use strict K=5 or K=10 class distances: the mean cosine distance to K distinct reference plots, "
              "with each plot's distance the minimum over its photos. Image and regional geo features are separately "
              "L2-normalized; the image representation remains frozen pretrained SigLIP2. If either candidate "
              "class has fewer than K plots, retain fusion. This check does not use a reduced effective K.", "",
              r"\[A_m=\log\frac{d_m(c_r)+10^{-8}}{d_m(c_g)+10^{-8}},\qquad S=\min(A_{\mathrm{image}},A_{\mathrm{geo}}).\]", "",
              "On fusion/geo disagreements, override with geo exactly when strict-K geometry is eligible and S>τ. "
              "Typed never-intervene thresholds always retain fusion. The zero-threshold baselines use the same "
              "strict support rule and S>0.", "",
              "One common threshold is chosen across all four validation seeds for each declared combination of "
              "K∈{5,10}, minimum acted plots∈{20,10,0}, and harm cap H≤αR with α∈{0.5,0.7,0.9}. "
              "The guards apply to pooled validation: positive net rescue, at least 1% intervention coverage, "
              "the declared number of distinct acted plot IDs, and the declared harm cap. Repeated seed rows "
              "are pooled for this fixed selection objective; they are not independent samples. "
              "Candidate thresholds follow the existing router enumeration: typed never, zero, and midpoints "
              "between adjacent distinct positive eligible-disagreement scores. Selection maximizes net rescues, "
              "then prefers fewer harms, fewer actions, and a higher threshold. No feasible candidate gives the "
              "typed never-intervene policy.", "",
              "The threshold selection is sealed before test geometry or labels are used. All 18 thresholds transfer "
              "unchanged to both test reference banks, without reselection. Experts, score definition and eligibility "
              "are fixed; this stage fits only the declared scalar thresholds on validation.", "",
              "Validation uses 822 images/325 plots, experts fitted on 3,378 development-training images and "
              "the 3,378-image/1,300-plot reference bank. Both test comparisons use the same 1,347 images/531 plots "
              "and historical final-epoch-50 experts fitted on all 4,200 development images. The reference banks "
              "contain 3,378 images/1,300 plots and 4,200 images/1,625 plots respectively. "
              "The 4,200-reference bank therefore contains validation images used for threshold selection; its transfer "
              "is reported as a separate fixed comparison. The frozen RF comparator remains the "
              "[September 15 policy](../2026-09-15_geo_helpfulness_router/report.md).", "",
              "## All frozen selections", "",
              "Validation rescue/harm/net counts below pool four seed realizations. Acted plots count distinct "
              "plot IDs across those realizations, not a sum of per-seed plot counts.", "",
              "| K | Minimum plots | Harm cap | Threshold | Feasible | Validation rescues | Harms | Net | Acted plots |",
              "|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for row in selected.to_dict("records"):
        policy = policies[row["policy_id"]]
        cells = [str(row["k"]), str(row["minimum_plots"]), f"{row['harm_cap']:.1f}", _threshold_text(policy),
                 "Yes" if row["selected_feasible"] else "No", str(row["rescued_count"]), str(row["harmed_count"]),
                 str(row["rescued_minus_harmed"]), str(row["unique_acted_plots"])]
        lines.append("| " + " | ".join(cells) + " |")
    changed = []
    for k in (5, 10):
        for numerator in (5, 7, 9):
            specs = [policies[f"k{k}__plots_{minimum}__harm_0p{numerator}"]["threshold_specification"]
                     for minimum in (20, 10, 0)]
            if not all(spec == specs[0] for spec in specs[1:]):
                changed.append(f"K{k}, H≤{numerator / 10:g}R")
    lines += ["", ("Relaxing the minimum plot count changes the selected threshold for: " + "; ".join(changed) + ".")
              if changed else "Relaxing the minimum acted-plot count from 20 to 10 or 0 does not change any selected threshold in this grid.",
              "All plot-guard combinations remain in the outputs even when their thresholds or predictions coincide.", "",
              "[Full selection/utility table](threshold_selection.csv) · [All policies across all cohorts](policy_comparison_summary.csv)", "",
              "## Primary 20-plot comparisons", "",
              "Cells are mean [minimum, maximum] across four seeds. Accuracy is a percentage; F1 and MCC "
              "use native scales. Every cohort image is included, including ineligible cases that retain fusion. "
              "The fixed 18-class ontology includes absent-class F1 as zero in macro F1.", ""]
    index = summary.set_index(["cohort", "model", "metric"])
    for cohort in COHORTS:
        lines += [f"### {COHORT_NAMES[cohort]}", "", "| Model/policy | Accuracy (%) | Weighted F1 | Macro F1 | MCC |",
                  "|---|---:|---:|---:|---:|"]
        models = [model for model in _models(cohort, policies)
                  if model not in policies or policies[model]["minimum_plots"] == 20]
        for model in models:
            cells = [_name(model, policies)]
            for metric in METRICS:
                item = index.loc[(cohort, model, metric)]
                scale = 100 if metric == "accuracy" else 1
                cells.append(f"{item['mean'] * scale:.4f} [{item['minimum'] * scale:.4f}, {item['maximum'] * scale:.4f}]")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["[Every policy/seed metric](performance_by_seed.csv) · [All mean/ranges](performance_summary.csv) · "
              "[Per-class metrics](performance_by_habitat.csv) · [Strict-K eligibility](reference_coverage.csv)", "",
              "## Test intervention diagnostics", "",
              "Rescue means wrong to correct; harm means correct to wrong. The observed H/R and harm-cap "
              "violations below are descriptive test diagnostics. They do not filter policies, alter thresholds or "
              "constitute additional selection. Counts pool the same images across four seeds; pooled unique plots "
              "are reported separately. H/R is undefined when no rescues occur.", "",
              "| Test reference bank | K | Validation harm cap | Rescues | Harms | Net | Observed H/R | Cap exceeded? |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in primary.loc[primary.cohort.ne("validation")].to_dict("records"):
        ratio = row["observed_harm_per_rescue"]
        ratio_text = f"{ratio:.3f}" if pd.notna(ratio) else "Undefined"
        lines.append(f"| {row['cohort'].removeprefix('test_')} | {row['k']} | {row['harm_cap']:.1f} | "
                     f"{row['rescued_count']} | {row['harmed_count']} | {row['rescued_minus_harmed']} | "
                     f"{ratio_text} | {'Yes' if row['observed_harm_cap_violated'] else 'No'} |")
    lines += ["", "[Every per-seed intervention/plot/coverage diagnostic](interventions.csv) · "
              "[Every pooled policy diagnostic](pooled_interventions.csv)", "",
              "## Conditional paired intervals", "",
              "Each cohort uses 2,000 habitat-stratified whole-plot bootstrap draws with seed 20260930. "
              "A selected plot retains all photos and four expert seeds; the same draws pair every method. "
              "Per-seed metrics are recomputed and paired differences averaged. Every selected-policy ID is "
              "compared with fusion and its matching strict zero-threshold baseline; both test banks additionally "
              "compare it with frozen RF. All 144 contrasts and both metrics are retained (288 intervals). "
              "Identical prediction vectors may share numerical computation, while preserving every named contrast.", "",
              "The 95% percentile intervals use linear quantiles. They are conditional on the selected thresholds, "
              "fixed experts, reference banks and observed habitat strata. Threshold selection is not repeated "
              "inside these intervals; they do not account for selection uncertainty, adaptive exploration or "
              "multiple comparisons. The validation intervals describe the same data used to select thresholds, "
              "not independent validation of the selected policy.", "",
              "[All interval estimates](paired_intervals.csv) · [Settings and interval records](paired_bootstrap.json) · "
              "[Validation replicates](bootstrap_replicates_validation.csv) · "
              "[Test/3,378 replicates](bootstrap_replicates_test_3378.csv) · "
              "[Test/4,200 replicates](bootstrap_replicates_test_4200.csv)", "",
              "## Interpretation limits", "",
              "This user-approved threshold study follows earlier validation and cleaned-test inspections. "
              "Freezing the validation-selected thresholds before this application prevents reselection on these "
              "test outcomes, but it does not make the previously examined test set an independent confirmation. "
              "All 18 declared policies and both banks remain reported; no policy is selected or promoted from "
              "the test comparisons. New independent data are needed for confirmation. Previous artifacts remain unchanged.", "",
              "The reference banks and modalities retain the coverage, repeated-photo/geo-vector and historical "
              "image-preprocessing limitations documented in the "
              "[embedding study](../2026-09-29_embedding_geometry/report.md).", ""]
    (output / "report.md").write_text("\n".join(lines))


def write_results(output: Path, validation: pd.DataFrame, tests: dict[str, pd.DataFrame],
                  selection: dict, declaration: dict) -> dict:
    """Report all frozen thresholds without reselecting or filtering policies."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policies = _policies(selection)
    if set(tests) != {"3378", "4200"}:
        raise ValueError("Expected the two declared test reference banks")
    cohorts = {"validation": _validate(validation, "validation", policies)}
    cohorts.update({f"test_{bank}": _validate(tests[bank], f"test_{bank}", policies) for bank in ("3378", "4200")})
    shared_columns = ["row_uid", "file", "plot_idx", "training_seed", "label_id_dense", "canonical_l3_id", "label_name",
                      "image_pred", "geo_pred", "raw_pred", "rf_pred", "rf_acted"]
    if not cohorts["test_3378"][shared_columns].equals(cohorts["test_4200"][shared_columns]):
        raise ValueError("Test identities, experts or RF differ between reference banks")
    for identity in ("row_uid", "plot_idx"):
        if set(cohorts["validation"][identity]) & set(cohorts["test_3378"][identity]):
            raise ValueError(f"Validation and test share {identity}")
    selected = _selection_table(cohorts["validation"], policies)
    performance, summary, per_class, interventions, pooled, support, detailed = _tables(cohorts, policies)
    comparison = _comparison_table(summary, pooled, policies)
    tables = {"threshold_selection.csv": selected, "performance_by_seed.csv": performance,
              "performance_summary.csv": summary, "performance_by_habitat.csv": per_class,
              "interventions.csv": interventions, "pooled_interventions.csv": pooled,
              "reference_coverage.csv": support, "policy_comparison_summary.csv": comparison}
    bootstrap, intervals = {}, []
    indexed = performance.set_index(["cohort", "training_seed", "model"])
    for cohort, rows in cohorts.items():
        settings, replicates = _bootstrap(rows, cohort, policies)
        bootstrap[cohort] = settings
        tables[f"bootstrap_replicates_{cohort}.csv"] = replicates
        intervals.extend(settings["comparisons"])
        for item in settings["comparisons"]:
            expected = np.mean([indexed.loc[(cohort, seed, item["target"]), item["metric"]]
                                - indexed.loc[(cohort, seed, item["baseline"]), item["metric"]] for seed in SEEDS])
            if not np.isclose(expected, item["point_estimate"], atol=1e-12, rtol=0):
                raise ValueError("Bootstrap point estimate differs from canonical classification metrics")
    if len(intervals) != 288:
        raise ValueError("Expected 144 paired comparisons with accuracy and weighted F1 intervals")
    tables["paired_intervals.csv"] = pd.DataFrame(intervals)
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    _json(output / "metrics.json", {"declaration": declaration, "selection": selection,
          "cohort_sizes": SIZES, "expert_seeds": list(SEEDS), "fixed_class_count": 18,
          "per_seed": detailed, "mean_and_ranges": _records(summary),
          "interventions": _records(interventions), "pooled_interventions": _records(pooled)})
    _json(output / "paired_bootstrap.json", bootstrap)
    figures = _plot(output, performance, policies)
    _report(output, summary, comparison, selected, policies, bootstrap)
    return {"cohort_sizes": {key: {"images": size[0], "plots": size[1]} for key, size in SIZES.items()},
            "seeds": list(SEEDS), "policy_count": len(policies), "selection": _records(selected),
            "performance_by_seed": _records(performance), "performance_summary": _records(summary),
            "interventions": _records(interventions), "pooled_interventions": _records(pooled),
            "policy_comparison_summary": _records(comparison), "bootstrap": bootstrap,
            "output_files": [*tables, "metrics.json", "paired_bootstrap.json", "report.md", *figures]}

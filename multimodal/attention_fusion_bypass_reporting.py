"""Static CLS-bypass extension report with both primaries and unchanged M3 context."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from . import attention_fusion_reporting as original
from .attention_fusion_reporting import (
    METRICS, METRIC_LABELS, GEO_LABELS, STYLE, _hash, _file_hash, _require,
    _safe, _scale, _write_csv, _save, draw_cm, _support, _curve, _md_table,
)
from .attention_fusion_bypass_comparison import BYPASSES

MODEL_LABELS = {**original.MODEL_LABELS,
    "attention_cls_skip_adapted": "CLS + adapted bypass",
    "attention_cls_skip_original": "CLS + original bypass"}
COLORS = {**original.COLORS, "attention_cls_skip_adapted": "#e69f00",
          "attention_cls_skip_original": "#4c78a8"}


def _label(arm: dict) -> str:
    return f"{MODEL_LABELS[arm['candidate_id']]} · {GEO_LABELS[arm['geo_mode']]}"


def _contrast_title(kind: str) -> str:
    return {"primary": "Both native primary comparisons", "architecture": "Supporting complete-configuration comparisons",
            "bypass_direct": "Original minus adapted bypass", "preprocessing": "Secondary preprocessing comparisons",
            "baseline_image": "Bypass versus image-only comparisons"}[kind]


def _validate(analysis: dict[str, Any]) -> None:
    _hash(analysis)
    labels = analysis["class_labels"]
    _require(len(labels) == len(analysis["class_order"]) == 18, "Bypass report must retain the fixed 18 class axes")
    seeds = [str(seed) for seed in analysis["training_seeds"]]
    _require(seeds == ["1", "2", "3", "4"], "Bypass report retains all four seeds")
    support = np.asarray(analysis["validation_class_support_images"])
    plots = np.asarray(analysis["validation_class_support_plots"])
    _require(support.shape == plots.shape == (18,) and bool((support >= plots).all()), "Invalid class support")
    _require(int(support.sum()) == analysis["validation_rows"], "Class support does not cover validation rows")
    arms = analysis["selected_arms"]
    _require(len({arm["arm_id"] for arm in arms}) == len(arms) > 0, "Selected arms must be unique")
    for arm in arms:
        _safe(arm["arm_id"])
        _require(arm["candidate_id"] in MODEL_LABELS and arm["geo_mode"] in GEO_LABELS, "Unknown arm")
        _require(set(arm["per_seed"]) == set(seeds), "A selected arm is missing a seed")
        for seed in seeds:
            cm = np.asarray(arm["per_seed"][seed]["metrics"]["confusion_matrix"])
            _require(cm.shape == (18, 18) and bool(np.array_equal(cm.sum(axis=1), support)),
                     "Confusion matrix is not from the shared validation cohort")
        for metric in METRICS:
            values = [arm["per_seed"][seed]["metrics"][metric] for seed in seeds]
            _require(np.isclose(np.mean(values), arm["mean"][metric], atol=1e-12, rtol=0), "Incorrect selected mean")
            _require(np.isclose(np.std(values, ddof=1), arm["sample_std"][metric], atol=1e-12, rtol=0), "Incorrect seed SD")
    ids = {arm["arm_id"] for arm in arms}
    for contrast in analysis["contrasts"]:
        _safe(contrast["contrast_id"])
        _safe(contrast["kind"])
        _require(contrast["lhs_arm_id"] in ids and contrast["rhs_arm_id"] in ids, "Unknown contrast arm")


    primary = [c["contrast_id"] for c in analysis["contrasts"] if c["kind"] == "primary"]
    _require(len(primary) == 2 and primary == analysis["primary_contrast_ids"], "Both declared primary contrasts are required")
    _require(len(arms) == 25 and sum(a["candidate_id"] in BYPASSES for a in arms) == 6,
             "Report must retain six new and 19 original arms")
    _require(len(analysis["contrasts"]) == 37, "Report must retain all 37 declared contrasts")

def _confusions(analysis: dict[str, Any], output: Path, dpi: int) -> list[dict[str, str]]:
    index = []
    for arm in analysis["selected_arms"]:
        directory = output / "figures/confusion" / arm["arm_id"]
        matrices = []
        for seed in map(str, analysis["training_seeds"]):
            cm = np.asarray(arm["per_seed"][seed]["metrics"]["confusion_matrix"], dtype=np.int64)
            matrices.append(cm)
            def sink(fig, values, normalized, seed=seed):
                name = f"seed{seed}_{'normalized' if normalized else 'counts'}"
                fig.suptitle(f"{_label(arm)} — seed {seed}", y=1.005, fontsize=14)
                fig.text(0.5, -0.015, "Support is shared across seeds; see class_support.csv. Zero-support rows show 0; their recall is NA.",
                         ha="center", fontsize=10)
                _save(fig, directory / name, dpi)
            draw_cm(cm, analysis["class_labels"], sink)
        pooled = np.stack(matrices).sum(axis=0)
        mean_counts = np.stack(matrices).mean(axis=0)
        rows = [{"true_dense_id": i, "true_class": label,
                 **{f"pred_{j:02d}": float(value) for j, value in enumerate(mean_counts[i])}}
                for i, label in enumerate(analysis["class_labels"])]
        _write_csv(output / "tables/confusion" / f"{arm['arm_id']}__mean_counts.csv", rows)
        def mean_sink(fig, values, normalized):
            name = "mean_normalized" if normalized else "pooled_counts"
            fig.suptitle(f"{_label(arm)} — {'mean normalized confusion' if normalized else 'pooled prediction counts'}", y=1.005, fontsize=14)
            fig.text(0.5, -0.015,
                     "Four predictions per image, not four independent cohorts. Mean normalized rates use the same validation support in each seed.",
                     ha="center", fontsize=10)
            _save(fig, directory / name, dpi)
        draw_cm(pooled, analysis["class_labels"], mean_sink)
        index.append({"arm_id": arm["arm_id"], "label": _label(arm),
                      "mean_normalized": str((directory / "mean_normalized.png").relative_to(output)),
                      "pooled_counts": str((directory / "pooled_counts.png").relative_to(output)),
                      "directory": str(directory.relative_to(output))})
    return index


def _comparative_metrics(analysis: dict[str, Any], output: Path, dpi: int) -> None:
    arms = analysis["selected_arms"]
    image = next((arm for arm in arms if arm["candidate_id"] == "image_only"), None)
    fig, axes = plt.subplots(3, 4, figsize=(18, 15), sharey="row")
    try:
        for row, geo in enumerate(("native", "l2", "standardize")):
            choices = ([image] if image is not None else []) + [arm for arm in arms if arm["geo_mode"] == geo]
            for column, metric in enumerate(METRICS):
                ax = axes[row, column]
                for y, arm in enumerate(choices):
                    ax.errorbar(_scale(metric) * arm["mean"][metric], y,
                                xerr=_scale(metric) * arm["sample_std"][metric], fmt="o", capsize=3,
                                color=COLORS[arm["candidate_id"]], markersize=6)
                ax.set_yticks(range(len(choices)), [MODEL_LABELS[arm["candidate_id"]] for arm in choices])
                ax.set_title(f"{GEO_LABELS[geo]} · {METRIC_LABELS[metric]}")
                ax.grid(axis="x", alpha=0.25)
                if metric == "mcc":
                    ax.set_xlim(-1, 1)
                else:
                    ax.set_xlim(0, 100)
            axes[row, 0].invert_yaxis()
        fig.suptitle("Selected learning rates: mean ± sample SD over four seeds", fontsize=15)
        fig.text(0.5, 0.01, "Image-only is the same shared reference in all panels. SD across seeds is not a confidence interval.", ha="center")
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        _save(fig, output / "figures/selected_metrics", dpi)
    finally:
        plt.close(fig)


def _curves(analysis: dict[str, Any], output: Path, dpi: int) -> None:
    for arm in analysis["selected_arms"]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        try:
            for role, style in (("train", "--"), ("validation", "-")):
                for ax, metric in ((axes[0], "loss"), (axes[1], "acc1")):
                    epochs, mean, sd = _curve(arm["histories_by_seed"], role, metric)
                    scale = 100 if metric == "acc1" else 1
                    color = "#0072b2" if role == "validation" else "#777777"
                    ax.plot(epochs, mean * scale, style, label=role.capitalize(), color=color)
                    ax.fill_between(epochs, (mean - sd) * scale, (mean + sd) * scale, alpha=0.12, color=color)
            choices = sorted([candidate for candidate in analysis["lr_candidates"] if candidate["arm_id"] == arm["arm_id"]],
                             key=lambda item: item["learning_rate"])
            for candidate, color in zip(choices, ("#0072b2", "#009e73", "#d55e00")):
                epochs, mean, sd = _curve(candidate["histories_by_seed"], "validation", "acc1")
                axes[2].plot(epochs, mean * 100, label=f"LR {candidate['learning_rate']:g}", color=color)
                axes[2].fill_between(epochs, (mean - sd) * 100, (mean + sd) * 100, alpha=0.1, color=color)
            for ax, title, ylabel in zip(axes, ("Selected LR: loss", "Selected LR: accuracy", "All declared LRs: validation"),
                                        ("Cross-entropy", "Acc@1 (%)", "Acc@1 (%)")):
                ax.set_title(title)
                ax.set_xlabel("Completed epoch")
                ax.set_ylabel(ylabel)
                ax.axvline(epochs[-1], color="0.45", ls=":", lw=1)
                ax.grid(alpha=0.2)
                ax.legend()
            fig.suptitle(f"{_label(arm)} · selected LR {arm['selected_learning_rate']:g}")
            fig.tight_layout()
            _save(fig, output / "figures/learning_curves" / arm["arm_id"], dpi)
        finally:
            plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    try:
        for arm in analysis["selected_arms"]:
            if arm["geo_mode"] not in {"native", "not_applicable"}:
                continue
            for ax, metric in zip(axes, ("acc1", "loss")):
                epochs, mean, _ = _curve(arm["histories_by_seed"], "validation", metric)
                ax.plot(epochs, mean * (100 if metric == "acc1" else 1),
                        label=MODEL_LABELS[arm["candidate_id"]], color=COLORS[arm["candidate_id"]])
        for ax, label in zip(axes, ("Validation Acc@1 (%)", "Validation cross-entropy")):
            ax.set_xlabel("Completed epoch")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle("Native-geo comparison: selected-LR mean histories; selection uses the final epoch only")
        fig.tight_layout()
        _save(fig, output / "figures/native_learning_curves", dpi)
    finally:
        plt.close(fig)


def _contrasts(analysis: dict[str, Any], output: Path, dpi: int) -> list[dict[str, Any]]:
    arms = {arm["arm_id"]: arm for arm in analysis["selected_arms"]}
    index = []
    for kind in dict.fromkeys(contrast["kind"] for contrast in analysis["contrasts"]):
        contrasts = [contrast for contrast in analysis["contrasts"] if contrast["kind"] == kind]
        fig, axes = plt.subplots(1, 4, figsize=(20, max(3.5, len(contrasts) * 0.42 + 1.6)), sharey=True)
        labels = [f"{_label(arms[c['lhs_arm_id']])}\n− {_label(arms[c['rhs_arm_id']])}" for c in contrasts]
        try:
            for ax, metric in zip(axes, METRICS):
                for y, contrast in enumerate(contrasts):
                    mean = contrast["mean_difference"][metric] * _scale(metric)
                    lower, upper = np.asarray(contrast["interval95"][metric]) * _scale(metric)
                    color = "#009e73" if contrast["category"] == "supported_improvement" else "#777777"
                    ax.hlines(y, lower, upper, color=color, lw=1.8)
                    ax.plot(mean, y, "o", color=color, ms=5)
                ax.axvline(0, color="0.35", ls="--", lw=1)
                ax.set_xlabel("MCC difference" if metric == "mcc" else f"{METRIC_LABELS[metric].replace(' (%)', '')} difference (pp)")
                ax.grid(axis="x", alpha=0.2)
            axes[0].set_yticks(range(len(contrasts)), labels, fontsize=8)
            axes[0].invert_yaxis()
            fig.suptitle(f"{_contrast_title(kind)}: conditional paired 95% plot-bootstrap intervals")
            fig.tight_layout()
            _save(fig, output / "figures/contrasts" / kind, dpi)
        finally:
            plt.close(fig)
        index.append({"kind": kind, "figure": f"figures/contrasts/{kind}.png"})
    return index


def _resources(analysis: dict[str, Any], output: Path, dpi: int) -> list[dict[str, Any]]:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    try:
        for arm in analysis["selected_arms"]:
            values = list(arm["per_seed"].values())
            params = [value["parameters"] for value in values]
            runtime = [value["runtime_seconds"] for value in values]
            cuda = [value.get("peak_cuda_bytes") for value in values]
            rss = [value.get("peak_process_rss_bytes") for value in values]
            row = {"arm_id": arm["arm_id"], "parameters": int(np.mean(params)),
                   "mean_runtime_seconds": float(np.mean(runtime)), "sample_sd_runtime_seconds": float(np.std(runtime, ddof=1)),
                   "mean_peak_cuda_bytes": float(np.mean(cuda)) if all(value is not None for value in cuda) else None,
                   "max_process_lifetime_rss_bytes": max(value for value in rss if value is not None) if any(value is not None for value in rss) else None,
                   "mean_acc1": arm["mean"]["acc1"]}
            rows.append(row)
            native = arm["geo_mode"] in {"native", "not_applicable"}
            marker = {"native": "o", "not_applicable": "o", "l2": "^", "standardize": "s"}[arm["geo_mode"]]
            for ax, x in zip(axes, (row["parameters"], row["mean_runtime_seconds"])):
                ax.scatter(x, arm["mean"]["acc1"] * 100, color=COLORS[arm["candidate_id"]],
                           marker=marker, s=55, alpha=1 if native else 0.45)
        for ax, xlabel in zip(axes, ("Trainable head / fusion parameters (log scale)", "Mean recorded head-run duration (s)")):
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean validation Acc@1 (%)")
            ax.grid(alpha=0.2)
        axes[0].set_xscale("log")
        present = {arm["candidate_id"] for arm in analysis["selected_arms"]}
        handles = [Line2D([], [], color=COLORS[name], marker="o", linestyle="", label=label)
                   for name, label in MODEL_LABELS.items() if name in present]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
        fig.suptitle("Accuracy and resources: encoder adaptation / feature extraction are excluded")
        fig.text(0.5, 0.015, "Circles: native/no geo; triangles: L2; squares: standardized. Secondary geo arms use lighter markers.", ha="center")
        fig.tight_layout(rect=(0, 0.21, 1, 1))
        _save(fig, output / "figures/resources", dpi)
    finally:
        plt.close(fig)
    _write_csv(output / "tables/resources.csv", rows)
    return rows

def _contrast_details(analysis: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    """Expose all seed deltas and descriptive class changes without ranking them."""
    arms = {arm["arm_id"]: arm for arm in analysis["selected_arms"]}
    seed_rows, class_rows, markdown_rows = [], [], []
    for contrast in analysis["contrasts"]:
        for seed in map(str, analysis["training_seeds"]):
            changes = contrast["rescues_harms_by_seed"][seed]
            deltas = contrast["per_seed"][seed]
            seed_rows.append({"contrast_id": contrast["contrast_id"], "seed": int(seed),
                              **{f"difference_{metric}": deltas[metric] for metric in METRICS}, **changes})
            markdown_rows.append([contrast["contrast_id"], seed,
                                  *[f"{deltas[metric] * _scale(metric):+.3f}" for metric in METRICS],
                                  changes["rescued"], changes["harmed"], changes["net"]])
        left, right = arms[contrast["lhs_arm_id"]], arms[contrast["rhs_arm_id"]]
        matrices = {name: np.asarray([arm["per_seed"][str(seed)]["metrics"]["confusion_matrix"]
                                     for seed in analysis["training_seeds"]])
                    for name, arm in (("left", left), ("right", right))}
        left_tp = np.diagonal(matrices["left"], axis1=1, axis2=2).mean(axis=0)
        right_tp = np.diagonal(matrices["right"], axis1=1, axis2=2).mean(axis=0)
        for index, (name, support, plots) in enumerate(zip(analysis["class_labels"],
                analysis["validation_class_support_images"], analysis["validation_class_support_plots"])):
            class_rows.append({"contrast_id": contrast["contrast_id"], "dense_id": index,
                "canonical_l3_id": analysis["class_order"][index], "class_name": name,
                "validation_images": support, "validation_plots": plots,
                "mean_lhs_recall": float(left_tp[index] / support) if support else None,
                "mean_rhs_recall": float(right_tp[index] / support) if support else None,
                "mean_recall_difference": float((left_tp[index] - right_tp[index]) / support) if support else None,
                "mean_correct_count_difference": float(left_tp[index] - right_tp[index])})
    _write_csv(output / "tables/contrast_per_seed.csv", seed_rows)
    _write_csv(output / "tables/contrast_per_class.csv", class_rows)
    (output / "per_seed_contrasts.md").write_text("# Declared per-seed paired contrasts\n\n"
        "All differences are left minus right. Accuracy/F1 differences are percentage points; MCC differences use its native scale. "
        "Every seed uses the same validation cohort. Rescues and harms are paired image counts, and net equals rescues minus harms.\n\n"
        + _md_table(["Contrast ID", "Seed", "Δ Acc@1 (pp)", "Δ Acc@3 (pp)", "Δ weighted F1 (pp)", "Δ MCC",
                     "Rescued", "Harmed", "Net"], markdown_rows) + "\n")
    return [row for row in class_rows if row["contrast_id"] in analysis["primary_contrast_ids"]]


def _markdown(analysis, output, support, confusion_index, contrast_index, resources, primary_classes):
    arms = {a["arm_id"]: a for a in analysis["selected_arms"]}
    primaries = [c for c in analysis["contrasts"] if c["contrast_id"] in analysis["primary_contrast_ids"]]
    counts = analysis["counts"]
    repo = Path(__file__).resolve().parents[1]
    old_report = repo / analysis["source_study"]["analysis_dir"] / "report/report.md"
    old_link = os.path.relpath(old_report, output)
    lines = [f"# {analysis['study_id']} — CLS-bypass development extension", "",
        "Navigate: [Both primary results](#both-primary-results) · [Selected metrics](#selected-metrics) · "
        "[Paired comparisons](#paired-comparisons) · [Learning curves](#learning-curves) · "
        "[Confusions and class support](#confusions-and-class-support) · [Resources](#resources).", "",
        "This extension was proposed after observing the original M3 results. It reuses the same validation split and original comparator checkpoints. "
        "It is exploratory development evidence, not independent confirmation. Two primary comparisons were declared before the new fits; "
        "neither is replaced by a better secondary normalization result.", "",
        f"The analysis authenticates **{counts['development_fits']} new fits** plus **{counts['reused_development_fits']} original fits**, "
        f"retains **{counts['selected_arms']} selected arms** (six new and 19 original), and publishes **{counts['selected_checkpoints']} new selected checkpoint references**. "
        "The original 76 selected references and original M3 conclusion remain unchanged.", "",
        "[All 72 new fit scores](../tables/all_new_fits.csv) · [All 18 new arm/LR summaries](../tables/learning_rates.csv) · "
        "[Reused 228 fit scores](../tables/reused_fits.csv) · [Reused 57 LR summaries](../tables/reused_learning_rates.csv) · "
        "[All 25 selected arms](../tables/selected_arms.csv) · [Selection and checkpoint hashes](../selection.json) · "
        f"[Original M3 report]({old_link}) · [Complete numerical analysis](../analysis.json).", "",
        "## Both primary results", ""]
    for contrast in primaries:
        low, high = contrast["interval95"]["acc1"]
        lines += [f"**{_label(arms[contrast['lhs_arm_id']])} minus {_label(arms[contrast['rhs_arm_id']])}:** "
            f"mean Acc@1 difference **{contrast['mean_difference']['acc1']*100:+.3f} pp**, "
            f"conditional paired 95% interval **[{low*100:+.3f}, {high*100:+.3f}] pp**; "
            f"declared category **{contrast['category'].replace('_',' ')}**.", ""]
    lines += ["![Both primary comparisons](figures/contrasts/primary.png)", "",
        "Intervals resample entire validation plots using the same draw for every method and seed, then average the four seed differences. "
        "They are conditional on the selected recipes and this split, exclude learning-rate selection and new-split uncertainty, "
        "and are not simultaneous intervals or multiplicity-adjusted confirmations. Seeds share observations and are not independent datasets. "
        "The sample SD across seeds is distinct from a confidence interval.", "",
        "These are complete-configuration comparisons: bypass feature access, readout width and classifier parameter count change together. "
        "They do not identify an isolated attention, bypass-information or capacity effect. No minimum practically useful effect was declared.", "",
        "## Selected metrics", "",
        "One global learning rate per architecture/geo arm is selected by unrounded final-epoch mean Acc@1, then weighted F1, MCC, smaller LR and stable candidate ID. "
        "All four seeds and the final epoch are retained. Values below are mean ± sample SD; accuracy/F1 are percentages and MCC uses its native scale.", "",
        _md_table(["Arm", "Evidence", "LR", "Acc@1 (%)", "Acc@3 (%)", "Weighted F1 (%)", "MCC"],
            [[_label(a), a["study_role"].replace('_',' '), f"{a['selected_learning_rate']:g}",
              *[f"{a['mean'][m]*_scale(m):.3f} ± {a['sample_std'][m]*_scale(m):.3f}" for m in METRICS]]
             for a in analysis["selected_arms"]]), "",
        "![Selected metrics](figures/selected_metrics.png)", "",
        "[All 100 selected seed metrics](per_seed_metrics.md) include the 24 new checkpoints and 76 unchanged original checkpoints.", "",
        "## Paired comparisons", "",
        "All differences are left minus right. Rescue counts images corrected by the left model; harm counts correct right-model predictions broken by the left model. "
        "The table shows counts averaged across seeds, not additional independent images. Outcome categories follow the original frozen mean/seed/interval/secondary-metric rule. "
        "[All 37 interval results](../tables/paired_contrasts.csv) · [All per-seed differences and rescue/harm](per_seed_contrasts.md) · "
        "[CSV](tables/contrast_per_seed.csv).", ""]
    for item in contrast_index:
        selected = [c for c in analysis["contrasts"] if c["kind"] == item["kind"]]
        rows = []
        for c in selected:
            lo, hi = c["interval95"]["acc1"]
            changes = list(c["rescues_harms_by_seed"].values())
            rows.append([f"{_label(arms[c['lhs_arm_id']])} − {_label(arms[c['rhs_arm_id']])}",
                f"{c['mean_difference']['acc1']*100:+.3f}", f"[{lo*100:+.3f}, {hi*100:+.3f}]",
                f"{np.mean([x['rescued'] for x in changes]):.2f}", f"{np.mean([x['harmed'] for x in changes]):.2f}",
                c["category"].replace('_',' ')])
        lines += [f"### {_contrast_title(item['kind'])}", "", _md_table(
            ["Contrast", "Δ Acc@1 (pp)", "Conditional 95% interval", "Rescues / seed", "Harms / seed", "Category"], rows), ""]
        if item["kind"] != "primary":
            lines += [f"![{item['kind']}]({item['figure']})", ""]
    lines += ["## Learning curves", "",
        "Curves are diagnostics; epoch selection remains fixed. Shading is seed SD. Per-arm panels retain all three declared learning rates, "
        "including unsuccessful recipes. Original curve data are reused without changing their selected learning rates.", "",
        "![Native learning curves](figures/native_learning_curves.png)", "",
        _md_table(["Arm", "Selected and all-LR curves"], [[_label(a),
            f"[PNG](figures/learning_curves/{a['arm_id']}.png) · [PDF](figures/learning_curves/{a['arm_id']}.pdf)"]
            for a in analysis["selected_arms"]]), "",
        "## Confusions and class support", "",
        f"The shared cohort contains {analysis['validation_rows']} images from {analysis['validation_plots']} plots. All maps keep the fixed 18-class axes. "
        "Unsupported classes retain zero display rows but recall is NA, not zero performance. Low-support flags (<20 images or <10 plots) are descriptive reporting aids. "
        "No class-specific uncertainty was declared.", "", "![Class support](figures/class_support.png)", "",
        _md_table(["Limited-support class", "Images", "Plots", "Recall"], [[r['class_name'],r['validation_images'],r['validation_plots'],
            "NA" if r['unsupported_recall'] else "Available, limited support"] for r in support if r['low_support']]), "",
        "Per-seed raw maps have integer annotations. Row-normalized maps show fractions (exact zeros as `0`, otherwise two decimals). "
        "Mean-normalized maps average the four seed matrices. Pooled count maps sum four predictions per image; they are not an ensemble or four independent cohorts. "
        "Fractional mean counts are saved as CSV. [Full support](tables/class_support.csv).", ""]
    for contrast in primaries:
        rows = [r for r in primary_classes if r["contrast_id"] == contrast["contrast_id"]]
        lines += [f"### {_label(arms[contrast['lhs_arm_id']])} versus CLS: class changes", "",
            _md_table(["Class", "Images / plots", "Bypass recall (%)", "CLS recall (%)", "Δ recall (pp)", "Δ correct / seed"],
            [[r['class_name'],f"{r['validation_images']} / {r['validation_plots']}",
              *["NA" if r[k] is None else f"{r[k]*100:.2f}" for k in ('mean_lhs_recall','mean_rhs_recall','mean_recall_difference')],
              f"{r['mean_correct_count_difference']:+.2f}"] for r in rows]), ""]
    lines += ["[All contrasts by class](tables/contrast_per_class.csv). These are descriptive net correct-count changes, not separate class rescue/harm estimates.", "",
        _md_table(["New arm", "Across-seed maps", "Per-seed maps"], [[item['label'],
            f"[Mean normalized]({item['mean_normalized']}) · [Pooled counts]({item['pooled_counts']})",
            " · ".join(f"[Seed {seed}]({item['directory']}/seed{seed}_normalized.png)" for seed in analysis['training_seeds'])]
            for item in confusion_index]), "",
        f"Every PNG has a PDF beside it. Original 19-arm confusion maps remain available in the [unchanged M3 report]({old_link}).", "",
        "## Resources", "", "![Accuracy and resources](figures/resources.png)", "",
        _md_table(["Arm", "Parameters", "Head-run seconds (mean ± SD)", "Mean peak CUDA MiB"], [[_label(arms[r['arm_id']]),
            f"{r['parameters']:,}",f"{r['mean_runtime_seconds']:.2f} ± {r['sample_sd_runtime_seconds']:.2f}",
            "NA" if r['mean_peak_cuda_bytes'] is None else f"{r['mean_peak_cuda_bytes']/2**20:.1f}"] for r in resources]), "",
        "Run duration includes head training, diagnostics and checkpoint work; it excludes encoder adaptation and feature extraction. "
        "CPU RSS is a process-lifetime high-water mark, not isolated per-run memory. The two study sessions can experience different background load; "
        "recorded timing is descriptive, not a controlled throughput benchmark. [Resource data](tables/resources.csv).", "",
        "## Reproducibility", "",
        "This report consumes authenticated analysis records and performs no training, input-data loading, test evaluation or new selection. "
        "No test result or all-development refit enters this extension. Original M3 sources, artifacts and conclusions are preserved. "
        "The [report manifest](report_manifest.json) binds numerical inputs, both reporting source files and generated payloads. "
        "All result directions and categories above are derived from data; no winner is selected by the report.", ""]
    (output / "report.md").write_text("\n".join(lines))
    seed_rows = [[_label(a), a['study_role'], seed, f"{a['selected_learning_rate']:g}",
        *[f"{a['per_seed'][seed]['metrics'][m]*_scale(m):.3f}" for m in METRICS], a['per_seed'][seed]['fit_id']]
        for a in analysis['selected_arms'] for seed in map(str,analysis['training_seeds'])]
    (output / "per_seed_metrics.md").write_text("# All selected seed metrics\n\n" + _md_table(
        ["Arm","Evidence","Seed","LR","Acc@1 (%)","Acc@3 (%)","Weighted F1 (%)","MCC","Fit ID"],seed_rows) + "\n")


def generate_report(analysis: dict, output_dir: str | Path, *, png_dpi: int = 180) -> dict:
    """Render immutable extension figures; no model or dataset payload is read."""
    _validate(analysis)
    _require(isinstance(png_dpi,int) and png_dpi >= 72, "PNG DPI must be at least 72")
    output = Path(output_dir).resolve()
    identity = {"analysis_sha256": _hash(analysis), "reporting_code_sha256": _file_hash(Path(__file__)),
        "original_reporting_code_sha256": _file_hash(Path(original.__file__)), "png_dpi": png_dpi,
        "matplotlib": matplotlib.__version__, "seaborn": sns.__version__}
    manifest_path = output / "report_manifest.json"
    if output.exists() and any(output.iterdir()):
        _require(manifest_path.is_file(), "Partial report exists; retain it and choose a new destination")
        saved = json.loads(manifest_path.read_text())
        _require(saved.get("manifest_payload_sha256") == _hash({k:v for k,v in saved.items() if k != "manifest_payload_sha256"}),
                 "Report manifest checksum differs")
        _require(saved["identity"] == identity, "Report inputs or code changed; use a new destination")
        for name,digest in saved["files"].items():
            target = (output/name).resolve()
            _require(target.is_relative_to(output) and _file_hash(target)==digest, "Report payload changed")
        return saved
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        support = _support(analysis,output,png_dpi)
        new_analysis = {**analysis,"selected_arms":[a for a in analysis['selected_arms'] if a['candidate_id'] in BYPASSES]}
        confusions = _confusions(new_analysis,output,png_dpi)
        _comparative_metrics(analysis,output,png_dpi)
        _curves({**analysis,"lr_candidates":[*analysis['reused_lr_candidates'],*analysis['lr_candidates']]},output,png_dpi)
        contrasts = _contrasts(analysis,output,png_dpi)
        classes = _contrast_details(analysis,output)
        resources = _resources(analysis,output,png_dpi)
        _markdown(analysis,output,support,confusions,contrasts,resources,classes)
    manifest = {"schema_version":"attention_fusion.bypass_report.v1","study_id":analysis['study_id'],
        "identity":identity,"counts":analysis['counts'],"primary_contrast_ids":analysis['primary_contrast_ids'],
        "files":{str(p.relative_to(output)):_file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()},
        "data_payloads_opened":False,"test_data_opened":False,
        "confusion_style":"draw-cm-for-cs canonical count/row-normalized sink",
        "render_status":"generated_visual_review_required"}
    manifest['manifest_payload_sha256']=_hash(manifest)
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    return manifest

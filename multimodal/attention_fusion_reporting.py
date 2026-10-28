"""Static, data-driven M3 reports; consumes analysis records, never input data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns


METRICS = ("acc1", "acc3", "weighted_f1", "mcc")
METRIC_LABELS = {"acc1": "Acc@1 (%)", "acc3": "Acc@3 (%)",
                 "weighted_f1": "Weighted F1 (%)", "mcc": "MCC"}
MODEL_LABELS = {
    "image_only": "Image only", "geo_only": "Geo only",
    "raw_concat": "Raw concatenation", "projected_concat": "Projected concatenation",
    "tokenwise_capacity": "Token-wise capacity", "attention_two_token": "Two-token attention",
    "attention_cls": "CLS attention (2 layers)",
}
GEO_LABELS = {"native": "Native geo", "l2": "L2 geo", "standardize": "Standardized geo",
              "not_applicable": "No geo"}
COLORS = dict(zip(MODEL_LABELS, ("#666666", "#b8860b", "#0072b2", "#009e73",
                                "#cc79a7", "#d55e00", "#6a51a3")))
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
         "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
         "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
         "axes.spines.top": False, "axes.spines.right": False}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe(value: str) -> str:
    _require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value)), f"Unsafe artifact identifier: {value}")
    return value


def _label(arm: dict[str, Any]) -> str:
    return f"{MODEL_LABELS[arm['candidate_id']]} · {GEO_LABELS[arm['geo_mode']]}"


def _scale(metric: str) -> float:
    return 1.0 if metric == "mcc" else 100.0


def _contrast_title(kind: str) -> str:
    return "CLS alternative comparisons" if kind == "readout" else kind.replace("_", " ").capitalize()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: Any, stem: Path, dpi: int) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(str(path))
    return outputs


def draw_cm(cm: Any, label_list: list[str], sink: Callable | None = None) -> None:
    """Canonical CS count/row-normalized plots with an optional local sink.

    This follows draw-cm-for-cs without importing the W&B-logging evaluation
    helper. Unsupported rows display zeros; the report records recall as NA.
    """
    if not isinstance(cm, np.ndarray):
        if hasattr(cm, "detach") and hasattr(cm, "cpu"):
            cm = cm.detach().cpu().numpy()
        else:
            try:
                cm = np.asarray(cm)
            except Exception:
                return
    _require(cm.shape == (len(label_list), len(label_list)), "Confusion shape differs from fixed class order")
    _require(bool(np.isfinite(cm).all() and (cm >= 0).all()), "Confusion counts must be finite and nonnegative")
    _require(bool(np.array_equal(cm, cm.astype(np.int64))), "Raw confusion annotations require integer counts")
    row_sums = cm.sum(axis=1, keepdims=True).copy()
    row_sums[row_sums == 0] = 1
    for normalized, values in ((False, cm), (True, cm / row_sums)):
        fig, ax = plt.subplots(figsize=(15, 12))
        try:
            annotations = (np.array([["0" if value == 0 else f"{value:.2f}" for value in row]
                                     for row in values]) if normalized else values.astype(np.int64))
            sns.heatmap(values, annot=annotations, fmt="" if normalized else "d", cmap="Blues",
                        xticklabels=label_list, yticklabels=label_list,
                        annot_kws={"fontsize": 8}, vmin=0, vmax=1 if normalized else None, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix L3 (Normalized)" if normalized else "Confusion Matrix L3")
            ax.tick_params(axis="x", labelrotation=50)
            ax.tick_params(axis="y", labelrotation=0)
            plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
            fig.tight_layout()
            if sink is not None:
                sink(fig, values, normalized)
        finally:
            plt.close(fig)


def _validate(analysis: dict[str, Any]) -> None:
    _hash(analysis)
    labels = analysis["class_labels"]
    _require(len(labels) == len(analysis["class_order"]) == 18, "M3 must retain the fixed 18 class axes")
    seeds = [str(seed) for seed in analysis["training_seeds"]]
    _require(seeds == ["1", "2", "3", "4"], "M3 reports all four seeds")
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


def _support(analysis: dict[str, Any], output: Path, dpi: int) -> list[dict[str, Any]]:
    rows = [{"dense_id": i, "canonical_l3_id": canonical, "class_name": name,
             "validation_images": int(images), "validation_plots": int(plots),
             "low_support": bool(images < 20 or plots < 10), "unsupported_recall": bool(images == 0)}
            for i, (canonical, name, images, plots) in enumerate(zip(
                analysis["class_order"], analysis["class_labels"],
                analysis["validation_class_support_images"], analysis["validation_class_support_plots"]))]
    _write_csv(output / "tables/class_support.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    try:
        y = np.arange(18)
        for ax, key, title, threshold in zip(axes, ("validation_images", "validation_plots"),
                                             ("Validation images", "Validation plots"), (20, 10)):
            values = [row[key] for row in rows]
            ax.barh(y, values, color=["#d55e00" if value < threshold else "#0072b2" for value in values])
            ax.axvline(threshold, ls="--", color="0.5", lw=1, label=f"Low-support threshold: {threshold}")
            for i, value in enumerate(values):
                ax.text(value + max(values, default=1) * 0.015, i, str(value), va="center", fontsize=9)
            ax.set_xlabel(title)
            ax.set_xlim(0, max(values, default=1) * 1.18 + 1)
            ax.legend(loc="lower right")
        axes[0].set_yticks(y, analysis["class_labels"])
        axes[0].invert_yaxis()
        fig.suptitle("Validation support: repeated seeds do not add independent observations")
        fig.tight_layout()
        _save(fig, output / "figures/class_support", dpi)
    finally:
        plt.close(fig)
    return rows


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
    fig, axes = plt.subplots(3, 4, figsize=(17, 13), sharey="row")
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


def _curve(histories: dict[str, list[dict[str, Any]]], role: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entries = [histories[key] for key in sorted(histories, key=int)]
    epochs = np.asarray([row["epoch"] for row in entries[0]])
    _require(all([row["epoch"] for row in history] == epochs.tolist() for history in entries), "History epoch axes differ")
    values = np.asarray([[row[role][metric] for row in history] for history in entries], dtype=float)
    return epochs, values.mean(axis=0), values.std(axis=0, ddof=1)


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
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, frameon=False)
        fig.suptitle("Accuracy and resources: encoder adaptation / feature extraction are excluded")
        fig.text(0.5, 0.015, "Circles: native/no geo; triangles: L2; squares: standardized. Secondary geo arms use lighter markers.", ha="center")
        fig.tight_layout(rect=(0, 0.15, 1, 1))
        _save(fig, output / "figures/resources", dpi)
    finally:
        plt.close(fig)
    _write_csv(output / "tables/resources.csv", rows)
    return rows


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


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
    return [row for row in class_rows if row["contrast_id"] == analysis["primary_contrast_id"]]


def _markdown(analysis: dict[str, Any], output: Path, support: list[dict[str, Any]],
              confusion_index: list[dict[str, Any]], contrast_index: list[dict[str, Any]],
              resources: list[dict[str, Any]], primary_classes: list[dict[str, Any]]) -> None:
    arms = {arm["arm_id"]: arm for arm in analysis["selected_arms"]}
    primary = next(item for item in analysis["contrasts"] if item["contrast_id"] == analysis["primary_contrast_id"])
    low, high = primary["interval95"]["acc1"]
    fit_count = sum(len(candidate["per_seed"]) for candidate in analysis["lr_candidates"])
    selected_count = sum(len(arm["per_seed"]) for arm in analysis["selected_arms"])
    lines = [f"# {analysis['study_id']} — M3 development comparison", "",
             "Navigate: [Selected recipes](#selected-learning-rates-and-validation-metrics) · "
             "[Paired contrasts](#declared-contrasts-and-rescueharm) · [Learning curves](#learning-curves) · "
             "[Class support and confusions](#class-support-and-confusion-matrices) · [Resources](#accuracy-and-resources).", "",
             f"The predeclared primary comparison, **{_label(arms[primary['lhs_arm_id']])} versus {_label(arms[primary['rhs_arm_id']])}**, "
             f"has a mean Acc@1 difference of **{primary['mean_difference']['acc1'] * 100:+.3f} percentage points** "
             f"(conditional paired 95% interval [{low * 100:+.3f}, {high * 100:+.3f}]). "
             f"Its predeclared interpretation is **{primary['category'].replace('_', ' ')}**.", "",
             "This is development evidence on a reused, fixed plot-grouped split. Learning rates were selected on these same validation observations. "
             "Intervals are conditional on the selected recipes and split; they do not include selection or new-split uncertainty, and they are not simultaneous intervals across all contrasts.", "",
             f"All results use {analysis['validation_rows']:,} validation images from {analysis['validation_plots']:,} plots and four training/encoder seeds. "
             "Seeds share the same observations. Bootstrap replicates resample entire plots with the same draw for every model and seed. "
             "The seed SD is reported separately and is not a confidence interval.", "",
             f"The saved comparison covers **{fit_count} declared development fits**, **{len(analysis['lr_candidates'])} arm/LR summaries** "
             f"and **{selected_count} selected checkpoints**. Inspect [all fit metrics](../tables/all_fits.csv), "
             "[every learning-rate summary](../tables/learning_rates.csv), [selected-arm metrics](../tables/selected_arms.csv), "
             "[paired contrast intervals](../tables/paired_contrasts.csv), [checkpoint selection and hashes](../selection.json), "
             "or the [complete numerical analysis](../analysis.json). These tables retain unsuccessful recipes as well as selected ones.", "",
             "## Selected learning rates and validation metrics", "",
             "One learning rate is selected per architecture/geo arm across all four seeds, using final-epoch mean Acc@1, then weighted F1, then MCC and the frozen tie-break. "
             "No best-epoch or best-seed selection is used. Native geo remains the primary/default treatment.", "",
             _md_table(["Arm", "Selected LR", "Acc@1 (%)", "Acc@3 (%)", "Weighted F1 (%)", "MCC"],
                       [[_label(arm), f"{arm['selected_learning_rate']:g}",
                         *[f"{arm['mean'][metric] * _scale(metric):.3f} ± {arm['sample_std'][metric] * _scale(metric):.3f}" for metric in METRICS]]
                        for arm in analysis["selected_arms"]]), "",
             "Values are mean ± sample SD across seeds. [Every selected seed and checkpoint reference](per_seed_metrics.md) is retained.", "",
             "![Selected metrics](figures/selected_metrics.png)", "",
             "## Declared contrasts and rescue/harm", "",
             "Differences are left minus right. Rescue means the left model corrects an error made by the right model; harm means it breaks a correct right-model prediction. "
             "The table reports average counts per seed; summing seeds would count the same image repeatedly. Classification categories use the full predeclared seed/interval/secondary-metric rule. "
             "[Every contrast's per-seed differences and rescue/harm counts](per_seed_contrasts.md) are also available as [CSV](tables/contrast_per_seed.csv).", ""]
    for item in contrast_index:
        kind = item["kind"]
        selected = [contrast for contrast in analysis["contrasts"] if contrast["kind"] == kind]
        rows = []
        for contrast in selected:
            lower, upper = contrast["interval95"]["acc1"]
            counts = list(contrast["rescues_harms_by_seed"].values())
            rows.append([f"{_label(arms[contrast['lhs_arm_id']])} − {_label(arms[contrast['rhs_arm_id']])}",
                         f"{contrast['mean_difference']['acc1'] * 100:+.3f}", f"[{lower * 100:+.3f}, {upper * 100:+.3f}]",
                         f"{np.mean([value['rescued'] for value in counts]):.2f}",
                         f"{np.mean([value['harmed'] for value in counts]):.2f}", contrast["category"].replace("_", " ")])
        lines += [f"### {_contrast_title(kind)}", "",
                  _md_table(["Contrast", "Δ Acc@1 (pp)", "Conditional 95% interval", "Rescues / seed", "Harms / seed", "Interpretation"], rows), "",
                  f"![{kind} contrasts]({item['figure']})", ""]
    lines += ["An attention-specific claim requires supported gains against projected concatenation, raw concatenation and the capacity control in the same geo arm. "
              "CLS changes depth, readout width and parameter count; its comparison does not isolate a readout effect. A gain on a secondary normalization arm does not replace the native primary contrast. "
              "No application-specific minimum useful effect was supplied; statistical direction alone does not establish practical importance.", "",
              _md_table(["Geo arm", "Two-token attention-specific improvement established?"],
                        [[GEO_LABELS[geo], "Supported development evidence" if claim["supported"] else "Not established"]
                         for geo, claim in analysis["attention_claims"].items()]), "",
              "## Learning curves", "",
              "Curves are diagnostics, not an opportunity to change the final-epoch selection rule. Shaded bands in the per-arm figures show seed SD. "
              "Each per-arm figure includes all three declared learning rates, including unsuccessful ones.", "",
              "![Native learning curves](figures/native_learning_curves.png)", "",
              _md_table(["Arm", "Selected/all-LR curves"], [[_label(arm), f"[PNG](figures/learning_curves/{arm['arm_id']}.png) · [PDF](figures/learning_curves/{arm['arm_id']}.pdf)"] for arm in analysis["selected_arms"]]), "",
              "## Class support and confusion matrices", "",
              "All confusion matrices retain the same 18-class axes. Counts are integer predictions. Normalized cells are true-row fractions; exact zeros display as `0`, other values as two decimals. "
              "A zero-support row is displayed as zeros for plotting compatibility, but its recall is unavailable, not zero performance. "
              "Mean-normalized maps equal the mean of seed-normalized matrices because every seed uses identical class support. Pooled count maps sum four predictions per image; they are not an ensemble or four independent cohorts. "
              "Fractional mean counts are saved in CSV rather than truncated to integer annotations.", "",
              "![Validation support](figures/class_support.png)", "",
              _md_table(["Low-support class", "Images", "Plots", "Recall availability"],
                        [[row["class_name"], row["validation_images"], row["validation_plots"], "NA" if row["unsupported_recall"] else "Available, low support"] for row in support if row["low_support"]]), "",
              "The display flag uses fewer than 20 images or 10 plots to draw attention to limited support; it is a reporting aid, not a predeclared inferential threshold. "
              "[Full support table](tables/class_support.csv). Class-specific changes are descriptive and cannot revise the frozen selection rule.", "",
              "### Primary comparison by class", "",
              "All 18 classes are shown in their fixed order. Changes in correct predictions are averages per seed; they are net changes, not separate class-specific rescue/harm counts. "
              "These descriptive recall differences have no class-specific uncertainty intervals. [All declared contrasts by class](tables/contrast_per_class.csv).", "",
              _md_table(["Class", "Images / plots", "Attention recall (%)", "Projected-concat recall (%)", "Δ recall (pp)", "Δ correct / seed"],
                        [[row["class_name"], f"{row['validation_images']} / {row['validation_plots']}",
                          *["NA" if row[key] is None else f"{row[key] * 100:.2f}" for key in ("mean_lhs_recall", "mean_rhs_recall", "mean_recall_difference")],
                          f"{row['mean_correct_count_difference']:+.2f}"] for row in primary_classes]), "",
              _md_table(["Arm", "Across-seed maps", "Per-seed maps"],
                        [[item["label"], f"[Mean normalized]({item['mean_normalized']}) · [Pooled counts]({item['pooled_counts']})",
                          " · ".join(f"[Seed {seed}]({item['directory']}/seed{seed}_normalized.png)" for seed in analysis["training_seeds"])] for item in confusion_index]), "",
              "Every PNG has a vector PDF beside it; per-seed integer-count maps are in the same directories.", "",
              "## Accuracy and resources", "", "![Resources](figures/resources.png)", "",
              _md_table(["Arm", "Trainable parameters", "Recorded head-run seconds (mean ± SD)", "Mean peak CUDA MiB"],
                        [[_label(arms[row["arm_id"]]), f"{row['parameters']:,}", f"{row['mean_runtime_seconds']:.2f} ± {row['sample_sd_runtime_seconds']:.2f}",
                          f"{row['mean_peak_cuda_bytes'] / 2**20:.1f}" if row["mean_peak_cuda_bytes"] is not None else "NA"] for row in resources]), "",
              "Timing covers the recorded head run, including its diagnostics/checkpoint work, and excludes prior image-encoder adaptation and feature extraction. "
              "CPU RSS is a process-lifetime high-water mark, not an isolated fit's memory requirement; it is retained in [resources.csv](tables/resources.csv). "
              "Parameter and runtime differences limit attribution: stronger results alone do not identify attention weighting as the cause.", "",
              "## Scope and reproducibility", "",
              "This report consumes the complete saved M3 analysis; it performs no training, input-data loading, test evaluation or model selection. "
              "The existing cleaned test remains exploratory and is outside this report. Neither M3J joint fine-tuning nor all-development head refits are evaluated here. "
              "A negative or inconclusive frozen-attention result is a valid M3 outcome and does not prevent the separately declared joint-training study.", "",
              "The [report manifest](report_manifest.json) binds the analysis content, reporting code and all generated files. "
              "The complete numerical analysis and selection records remain in the parent comparison bundle.", ""]
    (output / "report.md").write_text("\n".join(lines))
    seed_rows = []
    for arm in analysis["selected_arms"]:
        for seed in map(str, analysis["training_seeds"]):
            value = arm["per_seed"][seed]
            seed_rows.append([_label(arm), seed, f"{arm['selected_learning_rate']:g}",
                              *[f"{value['metrics'][metric] * _scale(metric):.3f}" for metric in METRICS], value["fit_id"]])
    (output / "per_seed_metrics.md").write_text("# Selected per-seed validation metrics\n\n"
        "All seeds use the same validation cohort. Acc@1, Acc@3 and weighted F1 are percentages; MCC is on its native scale.\n\n"
        + _md_table(["Arm", "Seed", "Selected LR", "Acc@1 (%)", "Acc@3 (%)", "Weighted F1 (%)", "MCC", "Declared fit ID"], seed_rows)
        + "\n\nCheckpoint destinations and hashes are retained in the parent selection records; no fitted checkpoint is replaced by this report.\n")


def generate_report(analysis: dict[str, Any], output_dir: str | Path, *, png_dpi: int = 180) -> dict[str, Any]:
    """Render all predeclared M3 arms; preserve an existing identical report.

    Place ``output_dir`` immediately inside the comparison bundle so the
    relative links to its complete analysis, tables and selection record work.
    """
    _validate(analysis)
    _require(isinstance(png_dpi, int) and png_dpi >= 72, "PNG DPI must be at least 72")
    output = Path(output_dir)
    identity = {"analysis_sha256": _hash(analysis), "reporting_code_sha256": _file_hash(Path(__file__)),
                "png_dpi": png_dpi, "matplotlib": matplotlib.__version__, "seaborn": sns.__version__}
    manifest_path = output / "report_manifest.json"
    if output.exists() and any(output.iterdir()):
        _require(manifest_path.is_file(), "Partial report exists; retain it and choose a new output directory")
        previous = json.loads(manifest_path.read_text())
        _require(previous.get("manifest_payload_sha256") == _hash({key: value for key, value in previous.items()
                                                                   if key != "manifest_payload_sha256"}),
                 "Existing report manifest self-hash differs")
        _require(previous["identity"] == identity, "Report inputs/code changed; choose a new output directory")
        for relative, digest in previous["files"].items():
            _require(_file_hash(output / relative) == digest, f"Existing report changed: {relative}")
        return previous
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        support = _support(analysis, output, png_dpi)
        confusion_index = _confusions(analysis, output, png_dpi)
        _comparative_metrics(analysis, output, png_dpi)
        _curves(analysis, output, png_dpi)
        contrast_index = _contrasts(analysis, output, png_dpi)
        primary_classes = _contrast_details(analysis, output)
        resources = _resources(analysis, output, png_dpi)
        _markdown(analysis, output, support, confusion_index, contrast_index, resources, primary_classes)
    files = {str(path.relative_to(output)): _file_hash(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"schema_version": "attention_fusion.report.v1", "study_id": analysis["study_id"],
                "identity": identity, "files": files, "selected_arms": len(analysis["selected_arms"]),
                "selected_seeds": len(analysis["selected_arms"]) * len(analysis["training_seeds"]),
                "contrasts": len(analysis["contrasts"]), "data_payloads_opened": False,
                "test_data_opened": False, "confusion_style": "draw-cm-for-cs canonical count/row-normalized sink",
                "render_status": "generated_visual_review_required"}
    manifest["manifest_payload_sha256"] = _hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest

"""Descriptive figures for geometry support on expert disagreements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COHORTS = ("train_oof", "validation")
COHORT_TITLES = {
    "train_oof": "Development training · out-of-fold predictions",
    "validation": "Development validation",
}
MODALITIES = ("image", "geo")
MODALITY_TITLES = {"image": "Image geometry", "geo": "Geo geometry"}
SEEDS = (1, 2, 3, 4)
OUTCOME_STYLES = {
    "rescue": ("#00897b", "o", "Rescue: fusion wrong, geo correct"),
    "harm": ("#d97706", "^", "Harm: fusion correct, geo wrong"),
    "both_wrong": ("#9aa1a8", "o", "Both predictions wrong"),
}
SEED_COLORS = ("#264653", "#4379b0", "#875d9b", "#b55b78")


def _style_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    for name in ("bottom", "left"):
        axis.spines[name].set_color("#aeb6bf")
        axis.spines[name].set_linewidth(0.7)
    axis.tick_params(labelsize=10, color="#aeb6bf", labelcolor="#4d5966")
    axis.set_axisbelow(True)
    axis.grid(color="#edf0f2", linewidth=0.6)


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[str]:
    names = []
    for extension in ("png", "pdf"):
        name = f"{stem}.{extension}"
        figure.savefig(output_dir / name, dpi=200, facecolor="white")
        names.append(name)
    plt.close(figure)
    return names


def _disagreements(scores: pd.DataFrame, distance_k: int = 1) -> pd.DataFrame:
    selected = scores.loc[
        scores["disagreement"].fillna(False).astype(bool)
        & scores["valid_geometry"].fillna(False).astype(bool)
        & scores["distance_k"].eq(distance_k)
    ].copy()
    finite = np.isfinite(selected[["A_image", "A_geo"]].to_numpy(dtype=float))
    return selected.loc[finite.all(axis=1)]


def _outcome_handles(include_both_wrong: bool = True) -> list[Line2D]:
    outcomes = ("rescue", "harm", "both_wrong") if include_both_wrong else (
        "rescue", "harm"
    )
    return [
        Line2D(
            [], [], linestyle="none", marker=OUTCOME_STYLES[outcome][1],
            markersize=7, markerfacecolor=OUTCOME_STYLES[outcome][0],
            markeredgecolor="none", label=OUTCOME_STYLES[outcome][2],
        )
        for outcome in outcomes
    ]


def _symmetric_limit(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return max(0.1, float(np.max(np.abs(finite))) * 1.08) if finite.size else 1.0


def _full_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return -1.0, 1.0
    return min(-0.1, float(finite.min()) * 1.12), max(0.1, float(finite.max()) * 1.12)


def _geo_symlog(axis: Any, dimension: str, limits: tuple[float, float]) -> None:
    # Preserve every score while expanding the dense central range around zero.
    ticks = np.asarray([-100, -50, -20, -10, -5, -1, -0.5, 0, 0.5, 1, 5, 10, 20, 50, 100])
    ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
    labels = [f"{tick:g}" for tick in ticks]
    if dimension == "x":
        axis.set_xscale("symlog", linthresh=0.5)
        axis.set_xticks(ticks, labels)
    else:
        axis.set_yscale("symlog", linthresh=0.5)
        axis.set_yticks(ticks, labels)


def _validation_scatter(scores: pd.DataFrame, output_dir: Path) -> list[str]:
    selected = _disagreements(scores)
    selected = selected.loc[selected["cohort"].eq("validation")]
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    x_limit = _symmetric_limit(selected["A_image"].to_numpy(dtype=float))
    y_limits = _full_limits(selected["A_geo"].to_numpy(dtype=float))
    for seed, axis in zip(SEEDS, axes.flat):
        seed_data = selected.loc[selected["training_seed"].eq(seed)]
        _style_axis(axis)
        axis.axhline(0, color="#596675", linewidth=0.9, zorder=1)
        axis.axvline(0, color="#596675", linewidth=0.9, zorder=1)
        for outcome in ("both_wrong", "harm", "rescue"):
            points = seed_data.loc[seed_data["outcome"].eq(outcome)]
            color, marker, _ = OUTCOME_STYLES[outcome]
            axis.scatter(
                points["A_image"], points["A_geo"], marker=marker, c=color,
                s=17 if outcome == "both_wrong" else 28,
                alpha=0.28 if outcome == "both_wrong" else 0.70,
                linewidths=0, rasterized=True, zorder=2,
            )
        counts = seed_data["outcome"].value_counts()
        label = (
            f"Rescue {counts.get('rescue', 0)} · Harm {counts.get('harm', 0)}"
            f" · Both wrong {counts.get('both_wrong', 0)}"
        )
        axis.set_title(f"Expert seed {seed}", fontsize=12, pad=24)
        axis.text(
            0.5, 1.015, label, transform=axis.transAxes, ha="center",
            va="bottom", fontsize=9, color="#596675",
        )
        axis.set_xlim(-x_limit, x_limit)
        _geo_symlog(axis, "y", y_limits)
        axis.set_ylim(*y_limits)
        if seed_data.empty:
            axis.text(0.5, 0.5, "No supported disagreements", ha="center",
                      transform=axis.transAxes, color="#596675")
    figure.supxlabel(
        r"Image support $A_{\mathrm{image}}$   (negative: fusion class; positive: geo class)",
        fontsize=11, y=0.13,
    )
    figure.supylabel(
        r"Geo support $A_{\mathrm{geo}}$   (negative: fusion class; positive: geo class)",
        fontsize=11, x=0.02,
    )
    figure.suptitle("Frozen geometry versus expert disagreement", fontsize=17, y=0.975)
    figure.text(
        0.5, 0.94, "Development validation · nearest class reference (k = 1)",
        ha="center", fontsize=11, color="#596675",
    )
    figure.legend(handles=_outcome_handles(), loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, 0.071), frameon=False, fontsize=9)
    figure.text(
        0.5, 0.018,
        "Each point is an image with valid geometry; queries recur across expert seeds.\n"
        "Both axes compare geo's predicted class with fusion's predicted class; "
        "positive values favor geo's class.\n"
        "Geo axis uses a symmetric log scale (linear between −0.5 and 0.5); "
        "image axis is linear. All points are shown.",
        ha="center", fontsize=9, color="#596675", linespacing=1.6,
    )
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.20, top=0.87,
                           wspace=0.12, hspace=0.28)
    return _save_figure(figure, output_dir, "geometry_disagreement_validation")


def _mean_seed_cdf(
    frame: pd.DataFrame, column: str, grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    curves = []
    for seed in SEEDS:
        values = frame.loc[frame["training_seed"].eq(seed), column].to_numpy(dtype=float)
        values = np.sort(values[np.isfinite(values)])
        if values.size:
            curves.append(np.searchsorted(values, grid, side="right") / values.size)
    if not curves:
        return np.full(grid.shape, np.nan), 0
    return np.mean(curves, axis=0), len(curves)


def _distributions(scores: pd.DataFrame, output_dir: Path) -> list[str]:
    selected = _disagreements(scores)
    selected = selected.loc[selected["outcome"].isin(("rescue", "harm"))]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col", sharey=True)
    for column_index, modality in enumerate(MODALITIES):
        column = f"A_{modality}"
        values = selected[column].to_numpy(dtype=float)
        limits = _full_limits(values)
        # Each observed score is a knot, preserving the exact averaged ECDF.
        grid = np.unique(np.r_[limits[0], values, 0.0, limits[1]])
        for row_index, cohort in enumerate(COHORTS):
            axis = axes[row_index, column_index]
            _style_axis(axis)
            axis.axvline(0, color="#596675", linewidth=0.9)
            cohort_data = selected.loc[selected["cohort"].eq(cohort)]
            missing = []
            for outcome in ("rescue", "harm"):
                outcome_data = cohort_data.loc[cohort_data["outcome"].eq(outcome)]
                curve, count = _mean_seed_cdf(outcome_data, column, grid)
                axis.plot(grid, curve, color=OUTCOME_STYLES[outcome][0],
                          linewidth=2.3, drawstyle="steps-post")
                if count < len(SEEDS):
                    missing.append(f"{outcome}: {count} seeds")
            if modality == "geo":
                _geo_symlog(axis, "x", limits)
            axis.set_xlim(*limits)
            axis.set_ylim(-0.02, 1.02)
            axis.set_title(f"{COHORT_TITLES[cohort]}\n{MODALITY_TITLES[modality]}",
                           fontsize=11, pad=12)
            if column_index == 0:
                axis.set_ylabel("Cumulative fraction within outcome", fontsize=10)
            if row_index == 1:
                axis.set_xlabel(
                    f"A_{modality}  (positive favors geo's class)", fontsize=10,
                )
            if missing:
                axis.text(0.03, 0.96, "; ".join(missing), va="top",
                          transform=axis.transAxes, fontsize=8, color="#596675")
    figure.suptitle("Does geometric support distinguish rescue from harm?",
                   fontsize=17, y=0.975)
    figure.text(0.5, 0.935, "Nearest class reference (k = 1)",
                ha="center", fontsize=11, color="#596675")
    line_handles = [
        Line2D([], [], color=OUTCOME_STYLES[outcome][0], linewidth=2.3,
               label=OUTCOME_STYLES[outcome][2])
        for outcome in ("rescue", "harm")
    ]
    figure.legend(handles=line_handles, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, 0.074), frameon=False, fontsize=10)
    figure.text(
        0.5, 0.020,
        "Curves average the four seed-specific empirical CDFs; images have equal "
        "weight within each seed and outcome.\n"
        "The same queries recur across seeds; both-wrong disagreements are omitted. "
        "A curve farther right indicates higher support for geo's class.\n"
        "Geo axis uses a symmetric log scale (linear between −0.5 and 0.5); "
        "image axis is linear. Full score ranges are shown.",
        ha="center", fontsize=9, color="#596675", linespacing=1.6,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.85,
                           wspace=0.13, hspace=0.37)
    return _save_figure(figure, output_dir, "geometry_disagreement_distributions")


def _auc_sensitivity(auc: pd.DataFrame, output_dir: Path) -> list[str]:
    selected = auc.copy()
    matched_subset = False
    if "subset" in selected.columns:
        matched_subset = selected["subset"].eq("k5_supported").any()
        subset = "k5_supported" if matched_subset else "all_available"
        selected = selected.loc[selected["subset"].eq(subset)]
    selected = selected.loc[selected["distance_k"].isin((1, 5))]
    key = ["cohort", "modality", "training_seed", "distance_k"]
    if selected.duplicated(key).any():
        raise ValueError("Multiple AUROC rows have the same cohort, modality, seed and k.")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    offsets = np.linspace(-0.13, 0.13, len(SEEDS))
    for row_index, cohort in enumerate(COHORTS):
        for column_index, modality in enumerate(MODALITIES):
            axis = axes[row_index, column_index]
            _style_axis(axis)
            axis.axhline(0.5, color="#596675", linewidth=1, linestyle="--")
            panel = selected.loc[
                selected["cohort"].eq(cohort) & selected["modality"].eq(modality)
            ]
            for seed, offset, color in zip(SEEDS, offsets, SEED_COLORS):
                seed_data = panel.loc[panel["training_seed"].eq(seed)]
                points = seed_data.set_index("distance_k")["auc"]
                y_values = np.asarray([points.get(k, np.nan) for k in (1, 5)], dtype=float)
                axis.plot(np.arange(2) + offset, y_values, color=color,
                          marker="o", markersize=6, linewidth=1.2, alpha=0.9)
            axis.set_title(f"{COHORT_TITLES[cohort]}\n{MODALITY_TITLES[modality]}",
                           fontsize=11, pad=12)
            axis.set_xlim(-0.4, 1.4)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks([0, 1], ["k = 1", "k = 5"])
            if column_index == 0:
                axis.set_ylabel("AUROC (rescue is the positive outcome)", fontsize=10)
            if row_index == 1:
                axis.set_xlabel("Reference neighborhood size", fontsize=10)
            if not np.isfinite(panel["auc"].to_numpy(dtype=float)).any():
                axis.text(0.5, 0.65, "AUROC unavailable", ha="center",
                          transform=axis.transAxes, color="#596675")
    figure.suptitle("Sensitivity to reference neighborhood size", fontsize=17, y=0.97)
    figure.text(0.5, 0.93, "Can higher geometric support rank rescue above harm?",
                ha="center", fontsize=11, color="#596675")
    handles = [
        Line2D([], [], marker="o", color=color, linewidth=1.2,
               label=f"Expert seed {seed}")
        for seed, color in zip(SEEDS, SEED_COLORS)
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4,
                  bbox_to_anchor=(0.5, 0.071), frameon=False, fontsize=9)
    figure.text(
        0.5, 0.026,
        "Each dot is one seed; seeds are not pooled. Dashed line: no ranking "
        "separation (AUROC = 0.5).\n"
        + (
            "Both k values use the same cases with at least five reference plots "
            "for each candidate class."
            if matched_subset else
            "Each k uses all available supported cases; a change can reflect "
            "both the neighborhood definition and reference coverage."
        ),
        ha="center", fontsize=9, color="#596675", linespacing=1.6,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.84,
                           wspace=0.12, hspace=0.37)
    return _save_figure(figure, output_dir, "geometry_disagreement_auc")


def _support_rates(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    selected = summary.loc[summary["distance_k"].eq(1)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
    bar_width = 0.30
    seed_offsets = np.linspace(-0.07, 0.07, len(SEEDS))
    for cohort, axis in zip(COHORTS, axes):
        _style_axis(axis)
        cohort_data = selected.loc[selected["cohort"].eq(cohort)]
        for modality_index, modality in enumerate(MODALITIES):
            for outcome_index, outcome in enumerate(("rescue", "harm")):
                rows = cohort_data.loc[cohort_data["outcome"].eq(outcome)]
                by_seed = rows.set_index("training_seed")[f"{modality}_supports_geo"]
                values = np.asarray([by_seed.get(seed, np.nan) for seed in SEEDS], dtype=float) * 100
                finite = values[np.isfinite(values)]
                if not finite.size:
                    continue
                center = modality_index + (outcome_index - 0.5) * bar_width
                color = OUTCOME_STYLES[outcome][0]
                mean = float(finite.mean())
                axis.bar(center, mean, width=bar_width * 0.9, color=color,
                         alpha=0.50, linewidth=0, zorder=2)
                axis.scatter(center + seed_offsets, values, color=color,
                             marker=OUTCOME_STYLES[outcome][1], s=30,
                             edgecolors="white", linewidths=0.5, zorder=3)
                axis.text(center, float(finite.max()) + 3, f"{mean:.1f}%",
                          ha="center", va="bottom", fontsize=10, color="#34404d")
        axis.set_title(COHORT_TITLES[cohort], fontsize=11, pad=14)
        axis.set_xticks([0, 1], ["Image geometry", "Geo geometry"])
        axis.set_xlim(-0.5, 1.5)
        axis.set_ylim(0, 107)
        axis.set_yticks(np.arange(0, 101, 20))
    axes[0].set_ylabel("Images where geometry favors geo's class (%)", fontsize=11)
    figure.suptitle("When does geometry support geo's prediction?", fontsize=17, y=0.975)
    figure.text(0.5, 0.919, "Positive support score (A > 0) · nearest class reference (k = 1)",
                ha="center", fontsize=11, color="#596675")
    figure.legend(handles=_outcome_handles(include_both_wrong=False),
                  loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.101),
                  frameon=False, fontsize=10)
    figure.text(
        0.5, 0.029,
        "Bars and percentages are the mean of four seed-specific rates; dots show individual seeds.\n"
        "Both modalities use A > 0 to favor geo's class. Each rate uses valid "
        "disagreements within its outcome; queries recur across seeds.",
        ha="center", fontsize=9, color="#596675", linespacing=1.6,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.81, wspace=0.12)
    return _save_figure(figure, output_dir, "geometry_disagreement_support")


def render_figures(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    auc: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Write PNG/PDF figures, returning paths relative to ``output_dir``.

    Figures use per-query scores and per-seed summaries, without pooling seeds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        paths = _support_rates(summary, output_dir)
        paths += _validation_scatter(scores, output_dir)
        paths += _distributions(scores, output_dir)
        paths += _auc_sensitivity(auc, output_dir)
    return paths

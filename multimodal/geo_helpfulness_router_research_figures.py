"""Readable confusion and oracle-gap figures for the router research summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

from multimodal.geo_helpfulness_confusion_review import draw_cm


MODEL_NAMES = {
    "fusion": "Fusion baseline",
    "rf": "Fusion + RF router",
    "geometry": "Fusion + adapted geometry",
    "or": "Either router (OR)",
    "and": "Both routers (AND)",
    "oracle": "Oracle rescue router",
}


def _save(fig: plt.Figure, root: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(root / f"{stem}.{extension}", dpi=180, bbox_inches="tight")


def _geometry_router_schematic(root: Path) -> None:
    """Illustrate class-relative neighbor evidence without implying a projection."""
    fig, ax = plt.subplots(figsize=(13, 8.6))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    colours = {"fusion": "#356ba5", "geo": "#bd7324", "query": "#233047"}
    ax.text(0.5, 0.972, "How the adapted geometry router makes a correction",
            ha="center", va="top", fontsize=20, fontweight="bold")
    ax.text(0.5, 0.918,
            "Start with a disagreement: fusion predicts habitat " + r"$c_f$"
            + "; geo predicts another habitat " + r"$c_g$" + ".",
            ha="center", va="top", fontsize=12)
    ax.text(0.5, 0.874,
            "Each d is the mean distance to five nearest distinct reference plots of that candidate habitat, in that space.",
            ha="center", va="top", fontsize=11, color="#44536a")

    for left, modality, distances, score in [
        (0.025, "image", (0.30, 0.10), 1.10),
        (0.525, "geo", (0.20, 0.08), 0.92),
    ]:
        width, bottom, height = 0.45, 0.365, 0.455

        def point(x, y):
            return (left + width * x, bottom + height * y)

        ax.add_patch(FancyBboxPatch((left, bottom), width, height,
                                    boxstyle="round,pad=0.008,rounding_size=0.012",
                                    linewidth=1, edgecolor="#c5cedb", facecolor="#fafbfd"))
        title = "Adapted image embedding space" if modality == "image" else "Geo embedding space"
        ax.text(*point(0.5, 0.94), title, ha="center", va="center",
                fontsize=14, fontweight="bold")
        ax.text(*point(0.22, 0.825), "Fusion-class references " + r"$c_f$",
                ha="center", fontsize=10, color=colours["fusion"])
        ax.text(*point(0.75, 0.825), "Geo-class references " + r"$c_g$",
                ha="center", fontsize=10, color=colours["geo"])
        query = point(0.60, 0.38)
        reference_groups = {
            "fusion": [(0.14, 0.68), (0.21, 0.73), (0.29, 0.69),
                       (0.19, 0.60), (0.29, 0.61)],
            "geo": [(0.69, 0.66), (0.77, 0.70), (0.84, 0.63),
                    (0.71, 0.57), (0.81, 0.55)],
        }
        for candidate, references in reference_groups.items():
            for x, y in references:
                target = point(x, y)
                ax.plot([query[0], target[0]], [query[1], target[1]],
                        color=colours[candidate], alpha=0.35, linewidth=1,
                        linestyle="--", zorder=1)
                ax.scatter(*target, s=92, color=colours[candidate],
                           edgecolor="white", linewidth=1, zorder=3)
        ax.scatter(*query, s=100, marker="D", color=colours["query"], zorder=4)
        ax.text(*point(0.60, 0.29), "Query", ha="center", fontsize=10,
                color=colours["query"])
        for x, candidate, value in [(0.22, "f", distances[0]), (0.77, "g", distances[1])]:
            formula = rf"$d_{{\mathrm{{{modality}}}}}(c_{candidate})={value:.2f}$"
            ax.text(*point(x, 0.18), formula, ha="center", fontsize=12)
        formula = (rf"$A_{{\mathrm{{{modality}}}}}\approx"
                   rf"\ln({distances[0]:.2f}/{distances[1]:.2f})=+{score:.2f}$")
        ax.text(*point(0.5, 0.055), formula, ha="center", fontsize=14,
                fontweight="bold", color="#254a73")

    for start, end in [((0.25, 0.357), (0.40, 0.278)),
                       ((0.75, 0.357), (0.60, 0.278))]:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": "#67778c",
                                "lw": 1.5, "mutation_scale": 15})
    ax.add_patch(FancyBboxPatch((0.245, 0.126), 0.51, 0.145,
                                boxstyle="round,pad=0.01,rounding_size=0.012",
                                linewidth=1.2, edgecolor="#6b94b4", facecolor="#edf5fb"))
    ax.text(0.5, 0.24, "Both spaces favor the geo candidate strongly enough",
            ha="center", fontsize=12)
    ax.text(0.5, 0.195, r"$\min(A_{\mathrm{image}}, A_{\mathrm{geo}})"
            r"=\min(1.10,0.92)=0.92>\tau\approx0.0061$",
            ha="center", fontsize=14)
    ax.text(0.5, 0.148, "Switch from fusion to geo: use " + r"$c_g$",
            ha="center", fontsize=13, fontweight="bold", color="#254a73")
    ax.text(0.5, 0.079,
            "Otherwise retain fusion, including when either candidate has fewer than five reference plots.",
            ha="center", fontsize=11, color="#44536a")
    ax.text(0.025, 0.025,
            "Schematic: positions are not measured distances or 2-D projections. Each circle represents one distinct reference plot.\n"
            "Distances and scores are illustrative; the threshold is fitted on development-validation. Positive A favors geo, not proof of correctness.",
            ha="left", va="top", fontsize=10, color="#44536a")
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.045, top=0.98)
    _save(fig, root, "geometry_router_schematic")
    plt.close(fig)


def _confusion_focus(root: Path, matrices: dict[str, np.ndarray],
                     labels: list[str]) -> None:
    # These are the grassland, heath and wetland habitats in canonical CS order.
    selected = np.arange(4, 12)
    expected = [
        "Improved Grassland", "Neutral Grassland", "Calcareous Grassland",
        "Acid Grassland", "Bracken", "Dwarf Shrub Heath",
        "Fen, Marsh, Swamp", "Bog",
    ]
    assert [labels[i] for i in selected] == expected
    ticks = ["Improved\ngrassland", "Neutral\ngrassland", "Calcareous\ngrassland",
             "Acid\ngrassland", "Bracken", "Dwarf shrub\nheath",
             "Fen, marsh,\nswamp", "Bog"]
    support = matrices["fusion"].sum(axis=1)
    rows = [f"{name.replace(chr(10), ' ')}\n(n={support[i] // 4:,} images)"
            for name, i in zip(ticks, selected)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
    for index, (ax, model) in enumerate(zip(axes, ("fusion", "rf", "geometry"))):
        cm = matrices[model]
        focus = cm[np.ix_(selected, selected)]
        other = cm[selected].sum(axis=1) - focus.sum(axis=1)
        counts = np.column_stack((focus, other))
        values = counts / np.maximum(support[selected, None], 1)
        np.testing.assert_allclose(values.sum(axis=1), 1)
        annotations = np.array([[f"{100 * value:.1f}" if value > 0 else "–"
                                 for value in row] for row in values])
        sns.heatmap(values, ax=ax, annot=annotations, fmt="", cmap="Blues",
                    vmin=0, vmax=1, cbar=False, linewidths=0.4,
                    linecolor="white", annot_kws={"fontsize": 10},
                    xticklabels=ticks + ["All other\nhabitats"],
                    yticklabels=rows if index == 0 else False)
        ax.tick_params(axis="x", rotation=65, labelsize=10, length=0)
        ax.tick_params(axis="y", rotation=0, labelsize=11, length=0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_title(MODEL_NAMES[model], fontsize=15, fontweight="bold", pad=14)
        ax.set_xlabel("Predicted habitat", fontsize=12, labelpad=10)
        ax.set_ylabel("True habitat" if index == 0 else "", fontsize=12)
        for row in range(len(selected)):
            ax.add_patch(plt.Rectangle((row, row), 1, 1, fill=False,
                                       edgecolor="#172b4d", linewidth=1.2))
    fig.suptitle("Where do the routers change habitat confusions?", fontsize=20,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.924,
             "Eight selected true habitats; entries are percentages of each full true-habitat row. "
             "Outlined cells are correct predictions; – = 0.", ha="center", fontsize=12)
    fig.subplots_adjust(left=0.145, right=0.96, bottom=0.255, top=0.84, wspace=0.08)
    colour_axis = fig.add_axes([0.974, 0.255, 0.009, 0.585])
    bar = fig.colorbar(axes[-1].collections[0], cax=colour_axis,
                      ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    fig.text(0.145, 0.012,
             "Cleaned test: four seeds × 1,347 images. Selected rows cover "
             f"{int(support[selected].sum() // 4):,} images; all other predicted habitats "
             "are retained in the last column.\n"
             "Rows use the complete class denominator; they are not renormalized after selection. "
             "Full 18-class matrices accompany this report.", fontsize=11)
    _save(fig, root, "confusion_focus_comparison")
    plt.close(fig)


def _oracle_gap(root: Path, diagnostics: pd.DataFrame) -> None:
    models = ["rf", "geometry", "or", "and"]
    data = diagnostics.set_index("model").loc[models]
    assert not data.index.duplicated().any()
    np.testing.assert_array_equal(data.net_correct, data.rescues - data.harms)
    targets = data.rescues + data.missed_rescues
    assert targets.nunique() == 1 and int(targets.iloc[0]) == 471
    assert data.net_correct.ge(0).all()
    target = int(targets.iloc[0])
    fig, ax = plt.subplots(figsize=(12, 5.8))
    positions = np.arange(len(models))
    left = np.zeros(len(models))
    series = [
        ("net_correct", "Net gain retained", "#168a72"),
        ("harms", "Gain offset by harms", "#d96c59"),
        ("missed_rescues", "Missed oracle rescues", "#dbe2ea"),
    ]
    for column, name, colour in series:
        values = data[column].to_numpy(dtype=int)
        ax.barh(positions, values, left=left, color=colour, height=0.59,
                label=name, edgecolor="white", linewidth=1.1)
        for row, value in enumerate(values):
            if value:
                ax.text(left[row] + value / 2, row, str(value), ha="center",
                        va="center", fontsize=12, fontweight="bold",
                        color="#233047" if column == "missed_rescues" else "white")
        left += values
    np.testing.assert_array_equal(left, np.full(len(models), target))
    ax.axvline(target, color="#233047", linewidth=1, linestyle="--", zorder=0)
    ax.text(target, -0.65, "471: oracle gain", ha="right", fontsize=11,
            color="#233047")
    ax.set_yticks(positions, ["RF router", "Adapted geometry", "Either router (OR)",
                             "Both routers (AND)"], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 480)
    ax.set_xticks([0, 100, 200, 300, 400, 471])
    ax.set_xlabel("Additional correct decisions relative to fusion (four seeds pooled)",
                  fontsize=11, labelpad=10)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#edf0f4", linewidth=0.7)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#adb9c7")
    ax.legend(loc="upper center", bbox_to_anchor=(0.49, -0.23), ncol=3,
              frameon=False, fontsize=10)
    fig.suptitle("Why does performance remain below the oracle?", fontsize=18,
                 fontweight="bold", y=0.98)
    fig.text(0.21, 0.88,
             "Oracle gap = missed rescues + harmful corrections", fontsize=12,
             color="#344458")
    fig.subplots_adjust(left=0.21, right=0.97, top=0.8, bottom=0.28)
    fig.text(0.025, 0.015,
             "Gain accounting: net gain = rescues − harms. Thus each bar sums to 471; "
             "the red segment is a penalty, not an extra subset of rescues.\n"
             "Counts are seed–image decisions over 5,388 decisions, not distinct photographs. "
             "Both-wrong switches contribute zero to the accuracy gain.", fontsize=10)
    _save(fig, root, "oracle_gap")
    plt.close(fig)


def figures(output: Path, matrices: dict[str, np.ndarray], labels: list[str],
            diagnostics: pd.DataFrame) -> None:
    """Save comparative and complete confusion plots plus oracle-gap accounting."""
    root = output / "figures"
    root.mkdir(parents=True, exist_ok=True)
    assert len(labels) == 18
    support = matrices["fusion"].sum(axis=1)
    assert support.sum() == 5388
    ticks = [label + (" [no test examples]" if support[i] == 0 else "")
             for i, label in enumerate(labels)]
    for model, name in MODEL_NAMES.items():
        cm = np.asarray(matrices[model])
        assert cm.shape == (18, 18)
        np.testing.assert_array_equal(cm.sum(axis=1), support)

        def sink(fig, values, normalized, model=model, name=name):
            fig.suptitle(name, fontsize=16, y=1.005)
            suffix = "normalized" if normalized else "counts"
            _save(fig, root, f"{model}_{suffix}")

        draw_cm(cm, ticks, sink=sink)
    _confusion_focus(root, matrices, labels)
    _oracle_gap(root, diagnostics)
    _geometry_router_schematic(root)
    assert not plt.get_fignums(), "Figure generation left open figures"

"""Figures for cached habitat-adapted image embeddings.

This module only renders supplied coordinates. It neither fits projections nor
loads embeddings, model predictions, checkpoints, or habitat labels from disk.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


SPLITS = ("development_train", "development_validation", "cleaned_test")
ROLE_LABELS = {
    "development_train": "Development train",
    "development_validation": "Development validation",
    "cleaned_test": "Cleaned test",
}
ROLE_MARKERS = {
    "development_train": "o",
    "development_validation": "^",
    "cleaned_test": "o",
}
FOCUS_GROUPS = (
    ("Grasslands", (4, 5, 6, 7)),
    ("Heath, bog, fen and acid grassland", (9, 11, 10, 7)),
)
MAP_NOTE = (
    "PCA is fitted on development train; validation and test are transformed. "
    "t-SNE is jointly fitted on all displayed samples.\n"
    "Colors show habitat labels, which do not enter projection fitting. "
    "Two-dimensional overlap is descriptive, not classifier performance."
)


def _colors() -> np.ndarray:
    return np.asarray([plt.get_cmap("tab20")(index) for index in range(18)])


def _axis_labels(method: str, metrics: dict) -> tuple[str, str]:
    if method == "pca":
        variance = metrics["pca_explained_variance_ratio"]
        return (
            f"PC 1 ({100 * float(variance[0]):.1f}% training variance)",
            f"PC 2 ({100 * float(variance[1]):.1f}% training variance)",
        )
    return "t-SNE 1", "t-SNE 2"


def _limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    bounds = []
    for column in ("x", "y"):
        lower = float(frame[column].min())
        upper = float(frame[column].max())
        padding = max(0.045 * (upper - lower), 1e-6)
        bounds.append((lower - padding, upper + padding))
    return bounds[0], bounds[1]


def _style_axis(axis: Any, bounds: tuple) -> None:
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        axis.spines[spine].set_color("#bbc1cb")
        axis.spines[spine].set_linewidth(0.7)
    axis.tick_params(labelsize=8, color="#bbc1cb", labelcolor="#68717c")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.grid(color="#edf0f4", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.set_xlim(bounds[0])
    axis.set_ylim(bounds[1])
    axis.set_aspect("equal", adjustable="box")


def _scatter(axis: Any, frame: pd.DataFrame, focus: tuple | None = None) -> None:
    palette = _colors()
    if focus is not None:
        axis.scatter(
            frame.x, frame.y, color="#d9dee5", s=7, alpha=0.46,
            linewidths=0, rasterized=True,
        )
        frame = frame.loc[frame.label_id_dense.isin(focus)]
    # Shuffle within role so no habitat is systematically painted last.
    for role in SPLITS:
        rows = frame.loc[frame.split.eq(role)]
        order = np.random.default_rng(42).permutation(len(rows))
        rows = rows.iloc[order]
        if rows.empty:
            continue
        marker_size = 9 if role != "development_validation" else 15
        axis.scatter(
            rows.x, rows.y,
            c=palette[rows.label_id_dense.to_numpy(dtype=int)],
            s=marker_size if focus is None else marker_size + 3,
            marker=ROLE_MARKERS[role], alpha=0.67 if focus is None else 0.83,
            edgecolors="none", linewidths=0, rasterized=True,
        )


def _habitat_handles(labels: list[str], indices: list[int] | None = None) -> list:
    palette = _colors()
    indices = list(range(len(labels))) if indices is None else indices
    return [
        Line2D(
            [], [], linestyle="none", marker="o", markersize=6,
            markerfacecolor=palette[index], markeredgecolor="none",
            label=fill(labels[index], width=36),
        )
        for index in indices
    ]


def _role_handles() -> list:
    return [
        Line2D(
            [], [], linestyle="none", marker=ROLE_MARKERS[role], markersize=6,
            markerfacecolor="#697381", markeredgecolor="none",
            label=ROLE_LABELS[role],
        )
        for role in SPLITS
    ]


def _save_figure(figure: Any, output: Path, stem: str) -> list[str]:
    names = []
    try:
        for extension in ("png", "pdf"):
            name = f"{stem}.{extension}"
            figure.savefig(output / "figures" / name, dpi=200, facecolor="white")
            names.append(f"figures/{name}")
    finally:
        plt.close(figure)
    return names


def _panel_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        frame.loc[frame.split.ne("cleaned_test")],
        frame.loc[frame.split.eq("cleaned_test")],
    )


def _panel_title(frame: pd.DataFrame, development: bool) -> str:
    name = "Development" if development else "Cleaned test"
    return f"{name} · {len(frame):,} images · {frame.plot_idx.nunique():,} plots"


def _overview(
    frame: pd.DataFrame, seed: int, metrics: dict, labels: list[str],
) -> Any:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 13.5))
    figure.subplots_adjust(
        left=0.065, right=0.985, top=0.865, bottom=0.285,
        wspace=0.16, hspace=0.29,
    )
    for row, method in enumerate(("pca", "tsne")):
        method_rows = frame.loc[frame.method.eq(method)]
        bounds = _limits(method_rows)
        for column, panel in enumerate(_panel_frames(method_rows)):
            axis = axes[row, column]
            _scatter(axis, panel)
            _style_axis(axis, bounds)
            method_title = "PCA" if method == "pca" else "t-SNE"
            axis.set_title(
                f"{method_title} · {_panel_title(panel, column == 0)}",
                fontsize=10.5, pad=10,
            )
            xlabel, ylabel = _axis_labels(method, metrics)
            axis.set_xlabel(xlabel, fontsize=9)
            axis.set_ylabel(ylabel, fontsize=9)
    figure.suptitle(
        f"Adapted image embedding geometry · encoder seed {seed}",
        x=0.065, y=0.975, ha="left", fontsize=19,
    )
    figure.text(
        0.065, 0.94,
        "SigLIP2 image encoder adapted on 3,378 development-training images. "
        "Common axes across splits within each projection.",
        fontsize=9.5, color="#59616d",
    )
    figure.legend(
        handles=_role_handles(), loc="upper left", bbox_to_anchor=(0.06, 0.925),
        ncol=3, frameon=False, fontsize=9,
    )
    figure.legend(
        handles=_habitat_handles(labels), loc="upper center",
        bbox_to_anchor=(0.52, 0.23), ncol=3, frameon=False, fontsize=9,
        columnspacing=3.4, labelspacing=0.65,
    )
    figure.text(0.065, 0.025, MAP_NOTE, fontsize=8.5, color="#59616d")
    return figure


def _focus(frame: pd.DataFrame, seed: int, labels: list[str]) -> Any:
    frame = frame.loc[frame.method.eq("tsne")]
    bounds = _limits(frame)
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 11.8))
    figure.subplots_adjust(
        left=0.065, right=0.985, top=0.85, bottom=0.20,
        wspace=0.16, hspace=0.3,
    )
    for row, (name, indices) in enumerate(FOCUS_GROUPS):
        for column, panel in enumerate(_panel_frames(frame)):
            axis = axes[row, column]
            _scatter(axis, panel, indices)
            _style_axis(axis, bounds)
            axis.set_title(
                f"{name}\n{_panel_title(panel, column == 0)}",
                fontsize=10, pad=9,
            )
            axis.set_xlabel("t-SNE 1", fontsize=9)
            axis.set_ylabel("t-SNE 2", fontsize=9)
    figure.suptitle(
        f"Previously confused habitat groups · encoder seed {seed}",
        x=0.065, y=0.977, ha="left", fontsize=18,
    )
    figure.text(
        0.065, 0.942,
        "Same global t-SNE coordinates and limits in all four panels. "
        "Other habitats appear in grey; no subset is refitted.",
        fontsize=9.5, color="#59616d",
    )
    figure.legend(
        handles=_role_handles(), loc="upper left", bbox_to_anchor=(0.06, 0.927),
        ncol=3, frameon=False, fontsize=9,
    )
    highlighted = sorted({index for _, indices in FOCUS_GROUPS for index in indices})
    figure.legend(
        handles=_habitat_handles(labels, highlighted), loc="upper center",
        bbox_to_anchor=(0.52, 0.148), ncol=3, frameon=False, fontsize=9,
        columnspacing=3.5, labelspacing=0.65,
    )
    figure.text(
        0.065, 0.025,
        "The joint t-SNE uses development and cleaned-test embeddings. "
        "Labels only color points; 2D overlap does not measure prediction reliability.",
        fontsize=8.5, color="#59616d",
    )
    return figure


def _all_seeds(frame: pd.DataFrame, labels: list[str]) -> Any:
    seeds = sorted(frame.training_seed.unique())
    figure, axes = plt.subplots(2, len(seeds), figsize=(5 * len(seeds), 11.8), squeeze=False)
    figure.subplots_adjust(
        left=0.045, right=0.985, top=0.855, bottom=0.265,
        wspace=0.23, hspace=0.22,
    )
    for column, seed in enumerate(seeds):
        rows = frame.loc[frame.training_seed.eq(seed) & frame.method.eq("tsne")]
        bounds = _limits(rows)
        for row, panel in enumerate(_panel_frames(rows)):
            axis = axes[row, column]
            _scatter(axis, panel)
            _style_axis(axis, bounds)
            axis.set_title(
                f"Encoder seed {seed} · " + ("development" if row == 0 else "cleaned test"),
                fontsize=10.5, pad=9,
            )
            axis.set_xlabel("t-SNE 1", fontsize=9)
            axis.set_ylabel("t-SNE 2", fontsize=9)
    figure.suptitle(
        "Adapted image embeddings across encoder seeds",
        x=0.045, y=0.975, ha="left", fontsize=20,
    )
    figure.text(
        0.045, 0.935,
        "Top: all 4,200 development images. Bottom: 1,347 cleaned-test images. "
        "Each column shares one joint t-SNE map across splits.",
        fontsize=10, color="#59616d",
    )
    figure.legend(
        handles=_role_handles(), loc="upper left", bbox_to_anchor=(0.04, 0.918),
        ncol=3, frameon=False, fontsize=9,
    )
    figure.legend(
        handles=_habitat_handles(labels), loc="upper center",
        bbox_to_anchor=(0.52, 0.216), ncol=3, frameon=False, fontsize=9,
        columnspacing=5.0, labelspacing=0.65,
    )
    figure.text(
        0.045, 0.027,
        "Axes are independent across encoder seeds: positions, orientations, "
        "and inter-cluster distances cannot be compared between columns.\n"
        "All maps use development and test embeddings jointly, without habitat "
        "labels. Points are images; 2D geometry is descriptive.",
        fontsize=9, color="#59616d",
    )
    return figure


def _interactive(
    output: Path, frame: pd.DataFrame, seed: int, metrics: dict, labels: list[str],
) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("Development · 4,200 images", "Cleaned test · 1,347 images"),
        horizontal_spacing=0.085,
    )
    palette = _colors()
    trace_methods = []
    buttons = []
    for method in ("pca", "tsne"):
        method_rows = frame.loc[frame.method.eq(method)]
        legend_added = set()
        for column, panel in enumerate(_panel_frames(method_rows), start=1):
            roles = SPLITS[:2] if column == 1 else SPLITS[2:]
            for label_id, label in enumerate(labels):
                rgb = tuple(round(255 * component) for component in palette[label_id, :3])
                for role in roles:
                    rows = panel.loc[panel.label_id_dense.eq(label_id) & panel.split.eq(role)]
                    details = rows[["file", "plot_idx", "label_name", "split"]].astype(str).to_numpy()
                    show_legend = label_id not in legend_added and not rows.empty
                    if show_legend:
                        legend_added.add(label_id)
                    figure.add_trace(
                        go.Scattergl(
                            x=rows.x, y=rows.y, mode="markers", name=label,
                            legendgroup=f"{method}_{label_id}",
                            showlegend=show_legend,
                            visible=(method == "pca"),
                            marker={
                                "size": 5 if role != "development_validation" else 7,
                                "color": f"rgb{rgb}", "opacity": 0.70,
                                "symbol": "triangle-up" if role == "development_validation" else "circle",
                            },
                            customdata=details,
                            hovertemplate=(
                                "<b>%{customdata[2]}</b><br>File: %{customdata[0]}"
                                "<br>Plot: %{customdata[1]}<br>Split: %{customdata[3]}"
                                "<extra></extra>"
                            ),
                        ), row=1, col=column,
                    )
                    trace_methods.append(method)
    for method in ("pca", "tsne"):
        bounds = _limits(frame.loc[frame.method.eq(method)])
        xlabel, ylabel = _axis_labels(method, metrics)
        title = (
            f"Adapted image embedding geometry · encoder seed {seed} · "
            f"{'PCA' if method == 'pca' else 't-SNE'}"
            "<br><sup>Circles: development train or cleaned test. Triangles: development validation. "
            "Click a habitat to toggle it across both panels.</sup>"
        )
        layout = {"title.text": title}
        for suffix in ("", "2"):
            layout[f"xaxis{suffix}.range"] = list(bounds[0])
            layout[f"yaxis{suffix}.range"] = list(bounds[1])
            layout[f"xaxis{suffix}.title.text"] = xlabel
            layout[f"yaxis{suffix}.title.text"] = ylabel
        buttons.append({
            "label": "PCA" if method == "pca" else "t-SNE (joint)",
            "method": "update", "args": [
                {"visible": [trace_method == method for trace_method in trace_methods]},
                layout,
            ],
        })
        if method == "pca":
            figure.update_layout(**layout)
    for column in (1, 2):
        figure.update_yaxes(
            scaleanchor="x" if column == 1 else "x2", scaleratio=1,
            row=1, col=column,
        )
    figure.update_layout(
        template="plotly_white", height=920,
        margin={"l": 80, "r": 40, "t": 180, "b": 280},
        title={"x": 0.045, "y": 0.98},
        updatemenus=[{
            "buttons": buttons, "direction": "down", "x": 0, "y": 1.22,
            "xanchor": "left", "yanchor": "top", "showactive": True,
        }],
        legend={
            "orientation": "h", "x": 0, "y": -0.17,
            "groupclick": "togglegroup", "font": {"size": 10},
        },
    )
    figure.add_annotation(
        text=MAP_NOTE.replace("\n", "<br>"), x=0, y=-0.48,
        xref="paper", yref="paper", showarrow=False,
        xanchor="left", yanchor="top", align="left",
        font={"size": 10, "color": "#59616d"},
    )
    name = f"adapted_image_embedding_seed{seed}_interactive.html"
    figure.write_html(
        output / "interactive" / name, include_plotlyjs=True, full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    return f"interactive/{name}"


def render_figures(
    output: Path, coordinates: pd.DataFrame, projection_metrics: dict, labels: list[str],
) -> list[str]:
    """Render static and self-contained interactive maps; return relative paths."""
    required = {
        "training_seed", "method", "row_uid", "file", "plot_idx",
        "label_id_dense", "label_name", "split", "x", "y",
    }
    missing = required - set(coordinates.columns)
    if missing:
        raise ValueError(f"Missing coordinate columns: {sorted(missing)}")
    if coordinates.empty or len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError("Expected nonempty coordinates and 18 unique habitat names")
    if not np.isfinite(coordinates[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite embedding coordinates")
    if set(coordinates.split) != set(SPLITS):
        raise ValueError("Expected development train, development validation, and cleaned test")
    if not set(coordinates.label_id_dense).issubset(set(range(18))):
        raise ValueError("Habitat IDs must be dense integers in [0, 17]")
    if coordinates.duplicated(["training_seed", "method", "row_uid"]).any():
        raise ValueError("Repeated image identity within an encoder/projection")
    for label_id, label_name in coordinates[["label_id_dense", "label_name"]].drop_duplicates().itertuples(index=False):
        if labels[int(label_id)] != label_name:
            raise ValueError("Inconsistent habitat IDs and names")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "interactive").mkdir(exist_ok=True)
    names = []
    for seed in sorted(coordinates.training_seed.unique()):
        frame = coordinates.loc[coordinates.training_seed.eq(seed)]
        if set(frame.method) != {"pca", "tsne"}:
            raise ValueError(f"Encoder seed {seed} must include PCA and t-SNE")
        metrics = projection_metrics[str(seed)]
        names.extend(_save_figure(
            _overview(frame, int(seed), metrics, labels), output,
            f"adapted_image_embedding_seed{seed}_overview",
        ))
        names.extend(_save_figure(
            _focus(frame, int(seed), labels), output,
            f"adapted_image_embedding_seed{seed}_habitat_focus",
        ))
        names.append(_interactive(output, frame, int(seed), metrics, labels))
    names.extend(_save_figure(
        _all_seeds(coordinates, labels), output, "adapted_image_embedding_all_seeds_tsne",
    ))
    return names

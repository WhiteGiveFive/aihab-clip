"""Scientific figures for the development-only embedding geometry study."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


SPACE_TITLES = {
    "image": "Frozen SigLIP2 (1,152 D)",
    "geo_native": "Native geo (64 D)",
    "concat_native": "Image + native geo (1,216 D)",
    "concat_standardized": "Image + standardized geo",
    "concat_balanced": "Image + balanced standardized geo",
}
PRIMARY_SPACES = ("image", "geo_native", "concat_native")
SCALING_SPACES = (
    "concat_native",
    "concat_standardized",
    "concat_balanced",
)
MAP_NOTE = (
    "Each panel is an independently fitted map; coordinates and global distances "
    "cannot be compared across panels.\n"
    "Points are images. Images from a shared plot have repeated geo vectors, "
    "which may overlap or coincide; 2D overlap is descriptive."
)


def _habitat_styles(metadata: pd.DataFrame) -> list[tuple[int, str, Any]]:
    pairs = (
        metadata[["label_id_dense", "label_name"]]
        .drop_duplicates()
        .sort_values("label_id_dense")
    )
    if pairs["label_id_dense"].duplicated().any():
        raise ValueError("A dense habitat ID maps to multiple habitat names.")
    palette = plt.get_cmap("tab20")
    styles = []
    for row in pairs.itertuples(index=False):
        label_id = int(row.label_id_dense)
        if not 0 <= label_id < 18:
            raise ValueError(f"Unexpected dense habitat ID: {label_id}")
        styles.append((label_id, str(row.label_name), palette(label_id)))
    return styles


def _axis_labels(
    method: str, space: str, metrics: Mapping[str, Mapping[str, Any]]
) -> tuple[str, str]:
    if method == "pca":
        variance = metrics[space]["pca_explained_variance_ratio"]
        return (
            f"PC 1 ({100 * float(variance[0]):.1f}% variance)",
            f"PC 2 ({100 * float(variance[1]):.1f}% variance)",
        )
    return "t-SNE 1", "t-SNE 2"


def _style_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    for spine in ("bottom", "left"):
        axis.spines[spine].set_color("#b9bfc8")
        axis.spines[spine].set_linewidth(0.7)
    axis.tick_params(labelsize=8, color="#b9bfc8", labelcolor="#69717d")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.set_axisbelow(True)
    axis.grid(color="#eef0f3", linewidth=0.5)
    axis.margins(0.04)
    axis.set_aspect("equal", adjustable="box")


def _legend_handles(
    styles: list[tuple[int, str, Any]],
) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="none",
            label=fill(name, width=36),
        )
        for _, name, color in styles
    ]


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        name = f"{stem}.{suffix}"
        figure.savefig(output_dir / name, dpi=200, facecolor="white")
        paths.append(name)
    plt.close(figure)
    return paths


def _overview_figure(
    metadata: pd.DataFrame,
    coordinates: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[str, Mapping[str, Any]],
    styles: list[tuple[int, str, Any]],
    spaces: tuple[str, ...],
    title: str,
    subtitle: str,
) -> Any:
    figure, axes = plt.subplots(2, 3, figsize=(17.5, 12.5))
    figure.subplots_adjust(
        left=0.058, right=0.985, bottom=0.27, top=0.885,
        hspace=0.29, wspace=0.22,
    )
    labels = metadata["label_id_dense"].to_numpy(dtype=int)
    color_by_id = {label_id: color for label_id, _, color in styles}
    colors = np.asarray([color_by_id[label_id] for label_id in labels])
    # One fixed shuffle prevents a class from always being painted on top.
    order = np.random.default_rng(42).permutation(len(metadata))
    for row, method in enumerate(("pca", "tsne")):
        for column, space in enumerate(spaces):
            axis = axes[row, column]
            values = np.asarray(coordinates[space][method])
            axis.scatter(
                values[order, 0], values[order, 1], c=colors[order],
                s=7, alpha=0.62, linewidths=0, rasterized=True,
            )
            axis.set_title(
                f"{'PCA' if method == 'pca' else 't-SNE'} · {SPACE_TITLES[space]}",
                fontsize=11, fontweight="medium", pad=10,
            )
            xlabel, ylabel = _axis_labels(method, space, metrics)
            axis.set_xlabel(xlabel, fontsize=9)
            axis.set_ylabel(ylabel, fontsize=9)
            _style_axis(axis)
    figure.suptitle(title, x=0.058, y=0.974, ha="left", fontsize=19)
    figure.text(
        0.058, 0.936,
        f"{len(metadata):,} development images · "
        f"{metadata['plot_idx'].nunique():,} plots · {subtitle}",
        fontsize=10, color="#59616d",
    )
    figure.legend(
        handles=_legend_handles(styles), loc="upper center",
        bbox_to_anchor=(0.51, 0.212), ncol=3, frameon=False,
        fontsize=9, columnspacing=3.5, handletextpad=0.6, labelspacing=0.55,
    )
    figure.text(0.058, 0.036, MAP_NOTE, fontsize=8.5, color="#59616d", va="bottom")
    return figure


def _focus_figure(
    metadata: pd.DataFrame,
    coordinates: Mapping[str, Mapping[str, np.ndarray]],
    styles: list[tuple[int, str, Any]],
) -> Any:
    groups = (
        ("Grasslands", ("Improved Grassland", "Neutral Grassland", "Acid Grassland")),
        ("Heath and bog", ("Dwarf Shrub Heath", "Bog")),
    )
    figure, axes = plt.subplots(2, 3, figsize=(17.5, 11.4))
    figure.subplots_adjust(
        left=0.058, right=0.985, bottom=0.18, top=0.87,
        hspace=0.29, wspace=0.22,
    )
    labels = metadata["label_id_dense"].to_numpy(dtype=int)
    highlighted_names = {name for _, names in groups for name in names}
    for row, (group_name, names) in enumerate(groups):
        for column, space in enumerate(PRIMARY_SPACES):
            axis = axes[row, column]
            values = np.asarray(coordinates[space]["tsne"])
            axis.scatter(
                values[:, 0], values[:, 1], s=6, color="#d7dce2",
                alpha=0.48, linewidths=0, rasterized=True,
            )
            for label_id, name, color in styles:
                if name not in names:
                    continue
                chosen = labels == label_id
                axis.scatter(
                    values[chosen, 0], values[chosen, 1], s=13,
                    color=color, alpha=0.80, linewidths=0, rasterized=True,
                )
            axis.set_title(
                f"{group_name} · {SPACE_TITLES[space]}", fontsize=10.5, pad=10,
            )
            axis.set_xlabel("t-SNE 1", fontsize=9)
            axis.set_ylabel("t-SNE 2", fontsize=9)
            _style_axis(axis)
    figure.suptitle(
        "Habitat overlap in the full-data t-SNE maps",
        x=0.058, y=0.973, ha="left", fontsize=19,
    )
    figure.text(
        0.058, 0.929,
        "Highlighted habitats share the full-data projection; no subset is refitted. "
        "Other habitats appear in grey.",
        fontsize=10, color="#59616d",
    )
    figure.legend(
        handles=_legend_handles([
            item for item in styles if item[1] in highlighted_names
        ]),
        loc="upper center", bbox_to_anchor=(0.51, 0.131), ncol=3,
        frameon=False, fontsize=9, columnspacing=3.5,
    )
    figure.text(0.058, 0.025, MAP_NOTE, fontsize=8.5, color="#59616d", va="bottom")
    return figure


def _interactive_overview(
    metadata: pd.DataFrame,
    coordinates: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[str, Mapping[str, Any]],
    styles: list[tuple[int, str, Any]],
    output_dir: Path,
) -> str | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    titles = [
        f"{'PCA' if method == 'pca' else 't-SNE'} · {SPACE_TITLES[space]}"
        for method in ("pca", "tsne") for space in PRIMARY_SPACES
    ]
    figure = make_subplots(
        rows=2, cols=3, subplot_titles=titles,
        horizontal_spacing=0.075, vertical_spacing=0.17,
    )
    labels = metadata["label_id_dense"].to_numpy(dtype=int)
    details = metadata[["file", "plot_idx", "label_name"]].astype(str).to_numpy()
    for row, method in enumerate(("pca", "tsne"), start=1):
        for column, space in enumerate(PRIMARY_SPACES, start=1):
            values = np.asarray(coordinates[space][method])
            for label_id, name, color in styles:
                chosen = labels == label_id
                rgb = tuple(round(255 * component) for component in color[:3])
                figure.add_trace(
                    go.Scattergl(
                        x=values[chosen, 0], y=values[chosen, 1], mode="markers",
                        marker={"size": 4, "opacity": 0.65, "color": f"rgb{rgb}"},
                        customdata=details[chosen], name=name,
                        legendgroup=str(label_id), showlegend=(row == 1 and column == 1),
                        hovertemplate=(
                            "<b>%{customdata[2]}</b><br>File: %{customdata[0]}"
                            "<br>Plot: %{customdata[1]}<extra></extra>"
                        ),
                    ),
                    row=row, col=column,
                )
            xlabel, ylabel = _axis_labels(method, space, metrics)
            figure.update_xaxes(title_text=xlabel, row=row, col=column)
            axis_number = (row - 1) * 3 + column
            x_reference = "x" if axis_number == 1 else f"x{axis_number}"
            figure.update_yaxes(
                title_text=ylabel, scaleanchor=x_reference, scaleratio=1,
                row=row, col=column,
            )
    figure.update_layout(
        template="plotly_white", height=1100,
        title={
            "text": (
                "Development embedding geometry"
                f"<br><sup>{len(metadata):,} images · "
                f"{metadata['plot_idx'].nunique():,} plots · "
                "Click a habitat in the legend to toggle it across all panels.</sup>"
            ),
            "x": 0.045,
        },
        margin={"l": 70, "r": 35, "t": 135, "b": 310},
        legend={
            "orientation": "h", "x": 0.0, "y": -0.18,
            "groupclick": "togglegroup", "font": {"size": 11},
        },
    )
    figure.add_annotation(
        text=MAP_NOTE.replace("\n", "<br>"), x=0, y=-0.39,
        xref="paper", yref="paper", showarrow=False,
        xanchor="left", yanchor="top", align="left",
        font={"size": 11, "color": "#59616d"},
    )
    name = "embedding_geometry_interactive.html"
    figure.write_html(
        output_dir / name, include_plotlyjs=True, full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    return name


def render_figures(
    metadata: pd.DataFrame,
    coordinates: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[str]:
    """Render paired habitat maps without loading data or fitting projections.

    Every coordinate array must share metadata's row order. The returned paths
    are relative to output_dir. Plotly is optional; its HTML bundles JavaScript
    locally so that the finished visualization works without internet access.
    """
    required = {"file", "plot_idx", "label_name", "label_id_dense"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(metadata.columns))}")
    if metadata.empty:
        raise ValueError("Embedding figure metadata is empty.")
    if not set(PRIMARY_SPACES).issubset(coordinates):
        raise ValueError("The three primary spaces are required.")
    for space in coordinates:
        for method in ("pca", "tsne"):
            values = np.asarray(coordinates[space][method])
            if values.shape != (len(metadata), 2) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {space}/{method} coordinates: {values.shape}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        styles = _habitat_styles(metadata)
        paths = _save_figure(
            _overview_figure(
                metadata, coordinates, metrics, styles, PRIMARY_SPACES,
                "Image, geo and concatenated input geometry",
                "Colours denote habitat labels",
            ),
            output_dir, "embedding_geometry_overview",
        )
        if set(SCALING_SPACES).issubset(coordinates):
            paths.extend(_save_figure(
                _overview_figure(
                    metadata, coordinates, metrics, styles, SCALING_SPACES,
                    "How modality scaling changes concatenated geometry",
                    "The same image and geo information, with different geo scaling",
                ),
                output_dir, "embedding_geometry_scaling",
            ))
        paths.extend(_save_figure(
            _focus_figure(metadata, coordinates, styles),
            output_dir, "embedding_geometry_habitat_focus",
        ))
        interactive = _interactive_overview(
            metadata, coordinates, metrics, styles, output_dir,
        )
        if interactive is not None:
            paths.append(interactive)
    return paths

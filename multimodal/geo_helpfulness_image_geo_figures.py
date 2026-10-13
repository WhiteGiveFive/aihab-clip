"""Compare adapted image and fixed geo embeddings using supplied projections.

Image points represent photographs; geo points represent distinct plots. This
module renders coordinates only and never fits a projection or loads a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_adapted_embedding_figures import (
    SPLITS,
    _axis_labels,
    _colors,
    _habitat_handles,
    _limits,
    _panel_frames,
    _role_handles,
    _save_figure,
    _scatter,
    _style_axis,
)


COMPARISON_NOTE = (
    "PCA axes are fitted on development train; t-SNE uses development and test jointly, without labels.\n"
    "Axes match across splits within each space, but not across modalities or encoder seeds. "
    "Geo coordinates are identical in every seed comparison.\n"
    "Image points are photographs; geo points are plots, so point densities have different units. "
    "Two-dimensional overlap is descriptive, not classifier performance."
)


def _panel_title(frame: pd.DataFrame, development: bool, modality: str) -> str:
    name = "Development" if development else "Cleaned test"
    if modality == "image":
        units = f"{len(frame):,} images · {frame.plot_idx.nunique():,} plots"
    else:
        units = f"{len(frame):,} plots · {int(frame.photo_count.sum()):,} images represented"
    return f"{name} · {units}"


def _base_figure() -> tuple[Any, Any]:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 13.5))
    figure.subplots_adjust(
        left=0.065, right=0.985, top=0.855, bottom=0.285,
        wspace=0.18, hspace=0.34,
    )
    return figure, axes


def _decorate(figure: Any, labels: list[str], title: str, subtitle: str) -> None:
    figure.suptitle(title, x=0.065, y=0.975, ha="left", fontsize=18)
    figure.text(0.065, 0.94, subtitle, fontsize=9.1, color="#59616d")
    figure.legend(
        handles=_role_handles(), loc="upper left", bbox_to_anchor=(0.06, 0.922),
        ncol=3, frameon=False, fontsize=9,
    )
    figure.legend(
        handles=_habitat_handles(labels), loc="upper center",
        bbox_to_anchor=(0.52, 0.23), ncol=3, frameon=False, fontsize=9,
        columnspacing=3.4, labelspacing=0.65,
    )
    figure.text(0.065, 0.023, COMPARISON_NOTE, fontsize=8.0, color="#59616d")


def _draw_row(
    axes: Any, rows: pd.DataFrame, modality: str, method: str,
    metrics: dict, row_name: str,
) -> None:
    bounds = _limits(rows)
    xlabel, ylabel = _axis_labels(method, metrics)
    for column, panel in enumerate(_panel_frames(rows)):
        axis = axes[column]
        _scatter(axis, panel)
        _style_axis(axis, bounds)
        axis.set_title(
            f"{row_name}\n{_panel_title(panel, column == 0, modality)}",
            fontsize=10, pad=9,
        )
        axis.set_xlabel(xlabel, fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)


def _comparison(
    image: pd.DataFrame, geo: pd.DataFrame, seed: int, method: str,
    image_metrics: dict, geo_metrics: dict, labels: list[str],
) -> Any:
    figure, axes = _base_figure()
    _draw_row(
        axes[0], image.loc[image.method.eq(method)], "image", method,
        image_metrics, "Adapted image embeddings · one point per photograph",
    )
    _draw_row(
        axes[1], geo.loc[geo.method.eq(method)], "geo", method,
        geo_metrics, "Geo embeddings · one point per plot",
    )
    method_name = "PCA" if method == "pca" else "t-SNE"
    _decorate(
        figure, labels,
        f"Adapted image and geo embedding spaces · {method_name} · encoder seed {seed}",
        "Top: habitat-adapted SigLIP2 images. Bottom: fixed 64-dimensional geo embeddings. "
        "Each modality has its own projection.",
    )
    return figure


def _geo_overview(geo: pd.DataFrame, metrics: dict, labels: list[str]) -> Any:
    figure, axes = _base_figure()
    for row, method in enumerate(("pca", "tsne")):
        method_name = "PCA" if method == "pca" else "t-SNE"
        _draw_row(
            axes[row], geo.loc[geo.method.eq(method)], "geo", method,
            metrics, f"{method_name} · one point per plot",
        )
    _decorate(
        figure, labels, "Geo embedding geometry · development and cleaned test",
        "Fixed 64-dimensional geo embeddings; each plot appears once. "
        "PCA is fitted on development-training plots; t-SNE uses all plots.",
    )
    return figure


def _interactive(
    output: Path, image: pd.DataFrame, geo: pd.DataFrame, seed: int,
    image_metrics: dict, geo_metrics: dict, labels: list[str],
) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    titles = []
    for modality, frame in (("image", image), ("geo", geo)):
        for column, panel in enumerate(_panel_frames(frame.loc[frame.method.eq("pca")])):
            name = "Adapted image" if modality == "image" else "Geo"
            titles.append(f"{name} · {_panel_title(panel, column == 0, modality)}")
    figure = make_subplots(
        rows=2, cols=2, subplot_titles=titles,
        horizontal_spacing=0.09, vertical_spacing=0.17,
    )
    palette = _colors()
    trace_methods = []
    for method in ("pca", "tsne"):
        legend_added = set()
        for row, (modality, frame) in enumerate((("image", image), ("geo", geo)), start=1):
            method_rows = frame.loc[frame.method.eq(method)]
            for column, panel in enumerate(_panel_frames(method_rows), start=1):
                roles = SPLITS[:2] if column == 1 else SPLITS[2:]
                for label_id, label in enumerate(labels):
                    rgb = tuple(round(255 * component) for component in palette[label_id, :3])
                    for role in roles:
                        rows = panel.loc[panel.label_id_dense.eq(label_id) & panel.split.eq(role)]
                        if rows.empty:
                            continue
                        details = rows[["file", "plot_idx", "label_name", "split"]].astype(str).copy()
                        details["photo_count"] = (
                            rows.photo_count.astype(str) if modality == "geo" else "1"
                        )
                        details["point_unit"] = "plot" if modality == "geo" else "photograph"
                        show_legend = label_id not in legend_added
                        legend_added.add(label_id)
                        file_label = "Representative file" if modality == "geo" else "File"
                        figure.add_trace(
                            go.Scattergl(
                                x=rows.x, y=rows.y, mode="markers", name=label,
                                # Keep inactive projections out of legend-group toggles.
                                legendgroup=f"{method}_{label_id}", showlegend=show_legend,
                                visible=(method == "pca"),
                                marker={
                                    "size": 5 if role != "development_validation" else 7,
                                    "color": f"rgb{rgb}", "opacity": 0.70,
                                    "symbol": "triangle-up" if role == "development_validation" else "circle",
                                },
                                customdata=details.to_numpy(),
                                hovertemplate=(
                                    "<b>%{customdata[2]}</b><br>Point unit: %{customdata[5]}"
                                    f"<br>{file_label}: %{{customdata[0]}}"
                                    "<br>Plot: %{customdata[1]}<br>Split: %{customdata[3]}"
                                    "<br>Images represented: %{customdata[4]}<extra></extra>"
                                ),
                            ), row=row, col=column,
                        )
                        trace_methods.append(method)
    buttons = []
    for method in ("pca", "tsne"):
        method_name = "PCA" if method == "pca" else "t-SNE"
        layout = {
            "title.text": (
                f"Adapted image and geo embedding spaces · encoder seed {seed} · {method_name}"
                "<br><sup>Top: one point per image. Bottom: one point per plot. "
                "Triangles: development validation. Geo is fixed across seeds.</sup>"
            ),
        }
        for index, (frame, metrics) in enumerate(((image, image_metrics), (geo, geo_metrics))):
            bounds = _limits(frame.loc[frame.method.eq(method)])
            xlabel, ylabel = _axis_labels(method, metrics)
            for column in (1, 2):
                number = index * 2 + column
                suffix = "" if number == 1 else str(number)
                layout[f"xaxis{suffix}.range"] = list(bounds[0])
                layout[f"yaxis{suffix}.range"] = list(bounds[1])
                layout[f"xaxis{suffix}.title.text"] = xlabel
                layout[f"yaxis{suffix}.title.text"] = ylabel
        buttons.append({
            "label": method_name if method == "pca" else "t-SNE (joint)",
            "method": "update", "args": [
                {"visible": [trace_method == method for trace_method in trace_methods]},
                layout,
            ],
        })
        if method == "pca":
            figure.update_layout(**layout)
    for row in (1, 2):
        for column in (1, 2):
            number = (row - 1) * 2 + column
            figure.update_yaxes(
                scaleanchor="x" if number == 1 else f"x{number}",
                scaleratio=1, row=row, col=column,
            )
    figure.update_layout(
        template="plotly_white", height=1250,
        margin={"l": 80, "r": 40, "t": 180, "b": 280},
        title={"x": 0.045, "y": 0.985},
        updatemenus=[{
            "buttons": buttons, "direction": "down", "x": 0, "y": 1.15,
            "xanchor": "left", "yanchor": "top", "showactive": True,
        }],
        legend={
            "orientation": "h", "x": 0, "y": -0.10,
            "groupclick": "togglegroup", "font": {"size": 10},
        },
    )
    figure.add_annotation(
        text=COMPARISON_NOTE.replace("\n", "<br>"), x=0, y=-0.27,
        xref="paper", yref="paper", showarrow=False,
        xanchor="left", yanchor="top", align="left",
        font={"size": 10, "color": "#59616d"},
    )
    name = f"adapted_image_geo_seed{seed}_interactive.html"
    figure.write_html(
        output / "interactive" / name, include_plotlyjs=True, full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    return f"interactive/{name}"


def _validate_coordinates(frame: pd.DataFrame, labels: list[str], modality: str) -> None:
    required = {
        "method", "row_uid", "file", "plot_idx", "label_id_dense",
        "label_name", "split", "x", "y",
    }
    required.add("training_seed" if modality == "image" else "photo_count")
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing {modality} coordinate columns: {sorted(missing)}")
    if frame.empty or not np.isfinite(frame[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Expected nonempty, finite {modality} coordinates")
    if set(frame.split) != set(SPLITS) or set(frame.method) != {"pca", "tsne"}:
        raise ValueError(f"Expected all three splits and both projections for {modality}")
    if not set(frame.label_id_dense).issubset(set(range(len(labels)))):
        raise ValueError(f"Invalid dense habitat IDs for {modality}")
    for label_id, label in frame[["label_id_dense", "label_name"]].drop_duplicates().itertuples(index=False):
        if labels[int(label_id)] != label:
            raise ValueError(f"Inconsistent habitat IDs and names for {modality}")
    identity = ["training_seed", "method", "row_uid"] if modality == "image" else ["method", "plot_idx"]
    if frame.duplicated(identity).any():
        raise ValueError(f"Repeated {modality} identity within a projection")
    if modality == "geo":
        counts = frame.photo_count.to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts < 1).any() or (counts != np.floor(counts)).any():
            raise ValueError("Geo photo counts must be positive integers")


def render_figures(
    output: Path, image_coordinates: pd.DataFrame, image_metrics: dict,
    geo_coordinates: pd.DataFrame, geo_metrics: dict, labels: list[str],
) -> list[str]:
    """Render image/geo comparisons and return relative output filenames."""
    if len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError("Expected 18 unique habitat names")
    _validate_coordinates(image_coordinates, labels, "image")
    _validate_coordinates(geo_coordinates, labels, "geo")
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "interactive").mkdir(parents=True, exist_ok=True)
    names = _save_figure(
        _geo_overview(geo_coordinates, geo_metrics, labels), output,
        "geo_embedding_overview",
    )
    for seed in sorted(image_coordinates.training_seed.unique()):
        image = image_coordinates.loc[image_coordinates.training_seed.eq(seed)]
        if set(image.method) != {"pca", "tsne"}:
            raise ValueError(f"Encoder seed {seed} must include PCA and t-SNE")
        metrics = image_metrics[str(seed)]
        for method in ("pca", "tsne"):
            names.extend(_save_figure(
                _comparison(image, geo_coordinates, int(seed), method, metrics, geo_metrics, labels),
                output, f"adapted_image_geo_seed{seed}_{method}",
            ))
        names.append(_interactive(
            output, image, geo_coordinates, int(seed), metrics, geo_metrics, labels,
        ))
    return names

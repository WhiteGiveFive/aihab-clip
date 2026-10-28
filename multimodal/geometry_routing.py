"""Label-free candidate scoring from strict, distinct-plot cosine geometry.

This module performs no file I/O, model inference, policy fitting or routing.
Reference labels define class neighborhoods; query labels are never inputs.
Image and geo spaces must be computed separately using the same identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassPlotDistances:
    """One modality's class geometry in the supplied query order.

    ``distances`` has shape (queries, classes). Unsupported classes contain
    NaN, even when some reference plots exist. ``nearest_plot_indices`` has
    shape (queries, classes, k) and indexes ``reference_plot_ids``; unsupported
    neighborhoods contain only -1. Plot IDs and reference row identities are
    stored in canonical plot-ID/row-ID order, independent of input row order.
    ``nearest_plot_ids`` exposes the same neighborhoods with empty strings
    for unsupported entries. Counts are distinct plots, never photographs.
    """

    distances: np.ndarray
    reference_plot_counts: np.ndarray
    nearest_plot_indices: np.ndarray
    reference_plot_ids: np.ndarray
    reference_plot_labels: np.ndarray
    reference_row_ids: np.ndarray
    reference_row_plot_ids: np.ndarray
    reference_row_labels: np.ndarray
    query_row_ids: np.ndarray
    query_plot_ids: np.ndarray
    k: int
    num_classes: int

    @property
    def nearest_plot_ids(self) -> np.ndarray:
        """Return plot IDs shaped (queries, classes, k), or empty strings."""
        result = np.full(self.nearest_plot_indices.shape, "", dtype=self.reference_plot_ids.dtype)
        available = self.nearest_plot_indices >= 0
        result[available] = self.reference_plot_ids[self.nearest_plot_indices[available]]
        return result


@dataclass(frozen=True)
class CandidateScores:
    """Per-query comparison of fusion and geo top-1 candidates.

    Positive ``A_image``/``A_geo`` favors geo's candidate in that space;
    ``score`` is their minimum. ``geometry_supported`` requires k plots for
    both candidates. ``eligible`` additionally requires a disagreement; it is
    NOT an intervention decision and does not impose a score threshold.
    Unsupported comparisons have NaN distances and scores in both modalities.
    Supported agreements retain finite distances and exactly zero scores.
    """

    fusion_predictions: np.ndarray
    geo_predictions: np.ndarray
    reference_plots_fusion_class: np.ndarray
    reference_plots_geo_class: np.ndarray
    d_image_fusion: np.ndarray
    d_image_geo: np.ndarray
    d_geo_fusion: np.ndarray
    d_geo_geo: np.ndarray
    A_image: np.ndarray
    A_geo: np.ndarray
    score: np.ndarray
    geometry_supported: np.ndarray
    disagreement: np.ndarray
    eligible: np.ndarray


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _ids(values: Any, size: int, name: str, *, unique: bool = False) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (size,) or not all(
        isinstance(value, str) and bool(value.strip()) for value in result
    ):
        raise ValueError(f"{name} must contain one nonempty string identity per row")
    result = result.astype(str, copy=True)
    if unique and len(np.unique(result)) != size:
        raise ValueError(f"{name} contains duplicate row identities")
    return result


def _class_ids(values: Any, size: int, num_classes: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if (result.shape != (size,) or result.dtype.kind not in "iu"
            or np.any((result < 0) | (result >= num_classes))):
        raise ValueError(f"{name} must contain integer class IDs in 0..{num_classes - 1}")
    return result.astype(np.int64, copy=True)


def _normalized(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or min(result.shape) < 1 or result.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty numeric matrix")
    result = result.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite embeddings")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        norms = np.linalg.norm(result, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError(f"{name} requires finite nonzero vector norms")
    return result / norms[:, None]


def class_plot_distances(
    reference_vectors: Any,
    query_vectors: Any,
    *,
    reference_row_ids: Any,
    reference_plot_ids: Any,
    reference_labels: Any,
    query_row_ids: Any,
    query_plot_ids: Any,
    k: int = 5,
    num_classes: int = 18,
) -> ClassPlotDistances:
    """Compute strict nearest-k class distances using distinct reference plots.

    Both vector matrices are independently L2-normalized in float64. Cosine
    distance is clipped to [0, 2]. A plot contributes its nearest photo; a
    supported class contributes the arithmetic mean of its k closest plots.
    Equal-distance plots are ordered by plot ID; photos within a plot are
    ordered by row ID. Queries may share a plot with each other, but cannot
    share any reference plot or row identity. Every reference plot must have
    exactly one class label. Image and geo dimensions may differ when calling
    this function separately, but reference/query widths within a call match.
    """
    k = _positive_integer(k, "k")
    num_classes = _positive_integer(num_classes, "num_classes")
    reference = _normalized(reference_vectors, "reference_vectors")
    query = _normalized(query_vectors, "query_vectors")
    if reference.shape[1] != query.shape[1]:
        raise ValueError("Reference/query embedding widths differ")
    nr, nq = len(reference), len(query)
    row_ids = _ids(reference_row_ids, nr, "reference_row_ids", unique=True)
    plot_ids = _ids(reference_plot_ids, nr, "reference_plot_ids")
    labels = _class_ids(reference_labels, nr, num_classes, "reference_labels")
    query_rows = _ids(query_row_ids, nq, "query_row_ids", unique=True)
    query_plots = _ids(query_plot_ids, nq, "query_plot_ids")
    if np.intersect1d(row_ids, query_rows).size:
        raise ValueError("Reference/query row identities overlap")
    if np.intersect1d(plot_ids, query_plots).size:
        raise ValueError("Reference/query plot identities overlap")

    order = np.lexsort((row_ids, plot_ids))
    ordered_plots, ordered_labels = plot_ids[order], labels[order]
    starts = np.flatnonzero(np.r_[True, ordered_plots[1:] != ordered_plots[:-1]])
    plot_labels = ordered_labels[starts]
    lengths = np.diff(np.r_[starts, nr])
    if not np.array_equal(ordered_labels, np.repeat(plot_labels, lengths)):
        raise ValueError("Reference plot has multiple class labels")
    plots = ordered_plots[starts]
    counts = np.bincount(plot_labels, minlength=num_classes)
    photo_distances = np.clip(1.0 - query @ reference[order].T, 0.0, 2.0)
    plot_distances = np.minimum.reduceat(photo_distances, starts, axis=1)
    distances = np.full((nq, num_classes), np.nan, dtype=np.float64)
    nearest = np.full((nq, num_classes, k), -1, dtype=np.int64)
    for class_id in range(num_classes):
        if counts[class_id] < k:
            continue
        candidates = np.flatnonzero(plot_labels == class_id)
        ranks = np.argsort(plot_distances[:, candidates], axis=1, kind="stable")[:, :k]
        selected = candidates[ranks]
        nearest[:, class_id] = selected
        distances[:, class_id] = np.take_along_axis(plot_distances, selected, axis=1).mean(axis=1)
    return ClassPlotDistances(
        distances=distances, reference_plot_counts=counts,
        nearest_plot_indices=nearest, reference_plot_ids=plots,
        reference_plot_labels=plot_labels, reference_row_ids=row_ids[order],
        reference_row_plot_ids=ordered_plots, reference_row_labels=ordered_labels,
        query_row_ids=query_rows, query_plot_ids=query_plots, k=k, num_classes=num_classes,
    )


def score_candidates(
    image_geometry: ClassPlotDistances,
    geo_geometry: ClassPlotDistances,
    fusion_predictions: Any,
    geo_predictions: Any,
    *,
    epsilon: float = 1e-8,
) -> CandidateScores:
    """Score two top-1 candidates without query labels or threshold decisions.

    The geometry objects must describe the same reference bank, class mapping,
    k, and ordered query identities. Each modality is allowed its own vector
    dimension and nearest plots. Both candidates must have strict k support;
    otherwise the entire comparison is unavailable, including its distances.
    """
    if not isinstance(image_geometry, ClassPlotDistances) or not isinstance(geo_geometry, ClassPlotDistances):
        raise ValueError("Expected ClassPlotDistances for both modalities")
    if image_geometry.k != geo_geometry.k or image_geometry.num_classes != geo_geometry.num_classes:
        raise ValueError("Image/geo k or class count differs")
    for name in (
        "reference_plot_counts", "reference_plot_ids", "reference_plot_labels",
        "reference_row_ids", "reference_row_plot_ids", "reference_row_labels",
        "query_row_ids", "query_plot_ids",
    ):
        if not np.array_equal(getattr(image_geometry, name), getattr(geo_geometry, name)):
            raise ValueError(f"Image/geo identities or reference labels differ: {name}")
    if (isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real)
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be finite and positive")
    n = len(image_geometry.query_row_ids)
    classes = image_geometry.num_classes
    fusion = _class_ids(fusion_predictions, n, classes, "fusion_predictions")
    geo = _class_ids(geo_predictions, n, classes, "geo_predictions")
    counts = image_geometry.reference_plot_counts
    fusion_counts, geo_counts = counts[fusion], counts[geo]
    supported = (fusion_counts >= image_geometry.k) & (geo_counts >= image_geometry.k)
    rows = np.arange(n)
    values = {}
    for modality, geometry in (("image", image_geometry), ("geo", geo_geometry)):
        if geometry.distances.shape != (n, classes):
            raise ValueError(f"Malformed {modality} class distances")
        df, dg = geometry.distances[rows, fusion], geometry.distances[rows, geo]
        if not (np.isfinite(df[supported]).all() and np.isfinite(dg[supported]).all()):
            raise ValueError(f"Supported {modality} candidates lack finite distances")
        df, dg = np.where(supported, df, np.nan), np.where(supported, dg, np.nan)
        values[f"d_{modality}_fusion"], values[f"d_{modality}_geo"] = df, dg
        values[f"A_{modality}"] = np.log((df + epsilon) / (dg + epsilon))
    disagreement = fusion != geo
    return CandidateScores(
        fusion_predictions=fusion, geo_predictions=geo,
        reference_plots_fusion_class=fusion_counts, reference_plots_geo_class=geo_counts,
        **values, score=np.minimum(values["A_image"], values["A_geo"]),
        geometry_supported=supported, disagreement=disagreement,
        eligible=supported & disagreement,
    )

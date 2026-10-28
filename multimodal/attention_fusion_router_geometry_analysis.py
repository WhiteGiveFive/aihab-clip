"""Paired descriptive R4 diagnostics for saved image-geometry representations.

The inputs are R3 query tables for one fixed expert pair. No thresholds are
fitted and no prediction is changed here. Query labels describe opportunities;
they do not change the previously computed geometry. The stage adapter binds
both tables to the shared R2 reference bank before calling this pure helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = (
    "row_uid", "file", "plot_idx", "label_id_dense", "label_name",
    "fusion_predictions", "geo_predictions", "reference_plots_fusion_class",
    "reference_plots_geo_class", "geometry_supported", "disagreement", "eligible",
)
GEO_COLUMNS = ("d_geo_fusion", "d_geo_geo", "A_geo")
IMAGE_COLUMNS = ("d_image_fusion", "d_image_geo", "A_image", "score")
MASK_COLUMNS = ("geometry_supported", "disagreement", "eligible")
GROUP_NAMES = ("eligible_all", "potential_rescue", "potential_harm", "both_wrong")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_frame(frame: pd.DataFrame, name: str) -> None:
    required = set(IDENTITY_COLUMNS + GEO_COLUMNS + IMAGE_COLUMNS)
    _require(required <= set(frame.columns), f"{name}: missing query/geometry columns")
    _require(len(frame) > 0, f"{name}: empty query table")
    _require(frame.row_uid.notna().all() and frame.row_uid.is_unique,
             f"{name}: missing or duplicate query identity")
    _require(not frame[list(IDENTITY_COLUMNS)].isna().any().any(),
             f"{name}: missing identity or support value")
    _require(frame.groupby("plot_idx").label_id_dense.nunique().max() == 1,
             f"{name}: inconsistent truth labels within a plot")
    for column in MASK_COLUMNS:
        _require(pd.api.types.is_bool_dtype(frame[column]),
                 f"{name}: non-boolean mask {column}")
    for column in ("label_id_dense", "fusion_predictions", "geo_predictions",
                   "reference_plots_fusion_class", "reference_plots_geo_class"):
        _require(pd.api.types.is_integer_dtype(frame[column])
                 and (frame[column] >= 0).all(), f"{name}: invalid integer {column}")
    supported = frame.geometry_supported.to_numpy()
    expected_support = ((frame.reference_plots_fusion_class.to_numpy() >= 5)
                        & (frame.reference_plots_geo_class.to_numpy() >= 5))
    disagreement = frame.fusion_predictions.to_numpy() != frame.geo_predictions.to_numpy()
    _require(np.array_equal(supported, expected_support), f"{name}: support mask differs")
    _require(np.array_equal(frame.disagreement, disagreement), f"{name}: disagreement differs")
    _require(np.array_equal(frame.eligible, supported & disagreement),
             f"{name}: eligibility differs")
    for column in GEO_COLUMNS + IMAGE_COLUMNS:
        values = frame[column].to_numpy(dtype=np.float64)
        _require(np.isfinite(values[supported]).all(),
                 f"{name}: nonfinite supported {column}")
        _require(np.isnan(values[~supported]).all(),
                 f"{name}: unsupported {column} must be NaN")
        if column.startswith("d_"):
            _require(((values[supported] >= 0) & (values[supported] <= 2)).all(),
                     f"{name}: invalid cosine distance {column}")
    _require(np.array_equal(frame.score.to_numpy()[supported],
                           np.minimum(frame.A_image.to_numpy()[supported],
                                      frame.A_geo.to_numpy()[supported])),
             f"{name}: score is not minimum modality ratio")


def _distribution(values: np.ndarray) -> dict:
    """Finite descriptive distribution; an empty group has null statistics."""
    values = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(values).all(), "Distribution contains nonfinite values")
    empty = {key: None for key in ("minimum", "q25", "median", "q75", "maximum",
                                   "mean", "standard_deviation")}
    if not len(values):
        return {"count": 0, **empty}
    quantiles = np.quantile(values, [0, .25, .5, .75, 1])
    return {"count": len(values),
            **{key: float(value) for key, value in zip(
                ("minimum", "q25", "median", "q75", "maximum"), quantiles)},
            "mean": float(values.mean()), "standard_deviation": float(values.std(ddof=0))}


def _counts(mask: np.ndarray, plots: np.ndarray) -> dict:
    return {"images": int(mask.sum()), "plots": int(len(np.unique(plots[mask])))}


def _quadrants(image: np.ndarray, geo: np.ndarray, mask: np.ndarray,
               plots: np.ndarray) -> dict:
    return {name: _counts(mask & condition, plots) for name, condition in (
        ("image_positive_geo_positive", (image > 0) & (geo > 0)),
        ("image_positive_geo_nonpositive", (image > 0) & (geo <= 0)),
        ("image_nonpositive_geo_positive", (image <= 0) & (geo > 0)),
        ("image_nonpositive_geo_nonpositive", (image <= 0) & (geo <= 0)),
    )}


def analyze_geometry_pair(pretrained: pd.DataFrame, checkpoint_matched: pd.DataFrame,
                          *, checkpoint_policy: str) -> tuple[dict, pd.DataFrame]:
    """Compare saved geometry for identical ordered rows and fixed expert outputs.

    Both input frames may come from either R3 rule: action, routed outcome and
    selection columns are intentionally ignored. Positive means strictly > 0.
    A delta is checkpoint-matched minus pretrained. The returned summary is
    JSON-safe (no NaN/Infinity); unsupported per-query geometry remains NaN in
    the returned DataFrame for lossless parquet storage.
    """
    _require(checkpoint_policy in ("final", "best"), "Unknown checkpoint policy")
    _validate_frame(pretrained, "pretrained")
    _validate_frame(checkpoint_matched, "checkpoint_matched")
    _require(len(pretrained) == len(checkpoint_matched), "Paired row counts differ")
    for column in IDENTITY_COLUMNS:
        _require(np.array_equal(pretrained[column].to_numpy(),
                                checkpoint_matched[column].to_numpy()),
                 f"Paired identity/expert/support differs: {column}")
    for column in GEO_COLUMNS:
        _require(np.array_equal(pretrained[column].to_numpy(),
                                checkpoint_matched[column].to_numpy(), equal_nan=True),
                 f"Shared geo geometry differs: {column}")

    pair = pretrained[list(IDENTITY_COLUMNS) + list(GEO_COLUMNS)].copy().reset_index(drop=True)
    pair.insert(0, "checkpoint_policy", checkpoint_policy)
    plots = pair.plot_idx.to_numpy(dtype=str)
    eligible = pair.eligible.to_numpy()
    supported = pair.geometry_supported.to_numpy()
    truth = pair.label_id_dense.to_numpy()
    fusion_correct = pair.fusion_predictions.to_numpy() == truth
    geo_correct = pair.geo_predictions.to_numpy() == truth
    potential_rescue = ~fusion_correct & geo_correct
    potential_harm = fusion_correct & ~geo_correct
    both_wrong = ~fusion_correct & ~geo_correct
    pair["truth_outcome"] = np.select(
        [potential_rescue, potential_harm, both_wrong],
        ["potential_rescue", "potential_harm", "both_wrong"], default="both_correct")
    pair["potential_rescue"] = potential_rescue
    pair["potential_harm"] = potential_harm
    pair["both_wrong"] = both_wrong
    pair["geo_positive"] = eligible & (pair.A_geo.to_numpy() > 0)
    groups = {"eligible_all": eligible, "potential_rescue": eligible & potential_rescue,
              "potential_harm": eligible & potential_harm, "both_wrong": eligible & both_wrong}

    arm_summaries = {}
    for arm, frame in (("pretrained", pretrained), ("checkpoint_matched", checkpoint_matched)):
        for column in IMAGE_COLUMNS:
            pair[f"{arm}_{column}"] = frame[column].to_numpy()
        image, geo = frame.A_image.to_numpy(), frame.A_geo.to_numpy()
        positive = eligible & (image > 0) & (geo > 0)
        pair[f"{arm}_image_positive"] = eligible & (image > 0)
        pair[f"{arm}_positive_score"] = positive
        pair[f"{arm}_missed_rescue_at_zero"] = potential_rescue & ~positive
        pair[f"{arm}_geo_supported_rescue_blocked_by_image"] = (
            eligible & potential_rescue & (geo > 0) & (image <= 0))
        arm_summaries[arm] = {
            "groups": {group: {
                **_counts(mask, plots),
                "quadrants": _quadrants(image, geo, mask, plots),
                "distributions": {column: _distribution(frame[column].to_numpy()[mask])
                                  for column in IMAGE_COLUMNS + ("A_geo",)},
            } for group, mask in groups.items()},
            "rescue_opportunities": {
                "all": _counts(potential_rescue, plots),
                "unavailable_reference_support": _counts(potential_rescue & ~supported, plots),
                "eligible_geo_nonpositive": _counts(eligible & potential_rescue & (geo <= 0), plots),
                "eligible_geo_positive_image_nonpositive": _counts(
                    eligible & potential_rescue & (geo > 0) & (image <= 0), plots),
                "positive_both_modalities": _counts(potential_rescue & positive, plots),
                "missed_at_zero": _counts(potential_rescue & ~positive, plots),
            },
        }

    for column in IMAGE_COLUMNS:
        pair[f"delta_{column}"] = (pair[f"checkpoint_matched_{column}"]
                                   - pair[f"pretrained_{column}"])
    for name, column in (("image", "A_image"), ("score", "score")):
        old = pair[f"pretrained_{column}"].to_numpy()
        new = pair[f"checkpoint_matched_{column}"].to_numpy()
        pair[f"{name}_positive_to_nonpositive"] = eligible & (old > 0) & (new <= 0)
        pair[f"{name}_nonpositive_to_positive"] = eligible & (old <= 0) & (new > 0)
        pair[f"{name}_positive_in_both"] = eligible & (old > 0) & (new > 0)
        pair[f"{name}_nonpositive_in_both"] = eligible & (old <= 0) & (new <= 0)

    paired_groups = {}
    for group, mask in groups.items():
        paired_groups[group] = {
            **_counts(mask, plots),
            "delta_distributions": {column: _distribution(pair[f"delta_{column}"].to_numpy()[mask])
                                    for column in IMAGE_COLUMNS},
            "image_delta_sign": {
                "negative": _counts(mask & (pair.delta_A_image.to_numpy() < 0), plots),
                "zero": _counts(mask & (pair.delta_A_image.to_numpy() == 0), plots),
                "positive": _counts(mask & (pair.delta_A_image.to_numpy() > 0), plots)},
            "crossings": {name: _counts(mask & pair[name].to_numpy(), plots)
                          for name in pair.columns
                          if name.startswith(("image_", "score_"))},
        }
    geo_rescue = eligible & potential_rescue & (pair.A_geo.to_numpy() > 0)
    summary = {
        "checkpoint_policy": checkpoint_policy,
        "query_images": len(pair), "query_plots": int(pair.plot_idx.nunique()),
        "eligible_disagreements": _counts(eligible, plots),
        "shared_native_geo_verified": True, "ordered_query_experts_support_verified": True,
        "reference_identity_verification": "shared R2 binding checked by calling stage adapter",
        "positive_comparator": "strict_gt_zero", "delta_direction": "checkpoint_matched_minus_pretrained",
        "group_denominator": "eligible disagreements; rescue_opportunities.all includes unsupported rows",
        "plot_counts_overlap_across_outcomes": True,
        "truth_used_for_descriptive_grouping_only": True,
        "threshold_fitting_performed": False,
        "interpretation": "Observed paired geometry differences; no causal identification.",
        "arms": arm_summaries, "paired_groups": paired_groups,
        "geo_positive_rescue_opportunities": {
            "all": _counts(geo_rescue, plots),
            "image_positive_in_both": _counts(geo_rescue & pair.image_positive_in_both.to_numpy(), plots),
            "image_positive_to_nonpositive": _counts(geo_rescue & pair.image_positive_to_nonpositive.to_numpy(), plots),
            "image_nonpositive_to_positive": _counts(geo_rescue & pair.image_nonpositive_to_positive.to_numpy(), plots),
            "image_nonpositive_in_both": _counts(geo_rescue & pair.image_nonpositive_in_both.to_numpy(), plots),
        },
    }
    return summary, pair

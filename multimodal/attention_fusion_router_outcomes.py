"""Descriptive R4 outcome analysis of fixed R3 predictions and policies.

These helpers do not select policies, load models, or access files. Plot
resampling conditions on the already selected experts and routing decisions;
it is not an independent evaluation of validation-fitted thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from multimodal.geometry_routing_policy import action_mask, validate_policy


def _queries(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"row_uid", "plot_idx", "label_id_dense", "fusion_predictions",
                "geo_predictions", "action"}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not required.issubset(frame):
        raise ValueError("query results must be nonempty and contain aligned R3 columns")
    result = frame.copy()
    for column in ("row_uid", "plot_idx"):
        if not all(isinstance(value, str) and value.strip() for value in result[column]):
            raise ValueError(f"{column} must contain nonempty string identities")
    if result.row_uid.duplicated().any():
        raise ValueError("query row identities must be unique")
    for column in ("label_id_dense", "fusion_predictions", "geo_predictions"):
        values = result[column].to_numpy()
        if values.dtype.kind not in "iu" or np.any((values < 0) | (values >= 18)):
            raise ValueError(f"{column} must contain integer class IDs in 0..17")
    action = result.action.to_numpy()
    if action.dtype.kind != "b":
        raise ValueError("action must contain Boolean values")
    if (result.groupby("plot_idx").label_id_dense.nunique() != 1).any():
        raise ValueError("each plot must have one true class")
    truth = result.label_id_dense.to_numpy()
    fusion = result.fusion_predictions.to_numpy()
    geo = result.geo_predictions.to_numpy()
    if np.any(action & (fusion == geo)):
        raise ValueError("interventions must be on expert disagreements")
    masks = {
        "rescue": action & (fusion != truth) & (geo == truth),
        "harm": action & (fusion == truth) & (geo != truth),
        "both_wrong_action": action & (fusion != truth) & (geo != truth),
        "missed_rescue": ~action & (fusion != truth) & (geo == truth),
    }
    for column, values in masks.items():
        if column in result and (result[column].dtype.kind != "b"
                                 or not np.array_equal(result[column], values)):
            raise ValueError(f"saved {column} flags disagree with fixed expert outcomes")
        result[column] = values
    routed = np.where(action, geo, fusion)
    if "routed_prediction" in result and not np.array_equal(result.routed_prediction, routed):
        raise ValueError("saved routed prediction disagrees with fixed expert selection")
    result["fusion_correct"] = fusion == truth
    result["routed_correct"] = routed == truth
    return result


def plot_outcomes(query_results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retain every plot, including plots with no intervention or net change.

    Event plot counts overlap: one plot can contain both rescues and harms.
    The four event-pattern categories and the three net-sign categories are
    separately exhaustive partitions, avoiding addition of overlapping counts.
    """
    rows = _queries(query_results)
    plots = rows.groupby("plot_idx", sort=True).agg(
        images=("row_uid", "size"), label_id_dense=("label_id_dense", "first"),
        actions=("action", "sum"), rescues=("rescue", "sum"),
        harms=("harm", "sum"), both_wrong_actions=("both_wrong_action", "sum"),
        missed_rescues=("missed_rescue", "sum"),
        fusion_correct=("fusion_correct", "sum"), routed_correct=("routed_correct", "sum"),
    ).reset_index()
    plots["net_rescues"] = plots.rescues - plots.harms
    plots["delta_acc1"] = plots.net_rescues / plots.images
    if not (plots.actions == plots.rescues + plots.harms + plots.both_wrong_actions).all():
        raise AssertionError("per-plot intervention accounting failed")
    if not (plots.routed_correct - plots.fusion_correct == plots.net_rescues).all():
        raise AssertionError("per-plot correctness accounting failed")
    has_rescue, has_harm = plots.rescues > 0, plots.harms > 0
    summary: dict[str, Any] = {
        "query_count": len(rows), "plot_count": len(plots),
        "actions": int(plots.actions.sum()), "rescues": int(plots.rescues.sum()),
        "harms": int(plots.harms.sum()), "net_rescues": int(plots.net_rescues.sum()),
        "action_plots": int((plots.actions > 0).sum()),
        "rescue_plots": int(has_rescue.sum()), "harm_plots": int(has_harm.sum()),
        "event_pattern_plots": {
            "rescue_only": int((has_rescue & ~has_harm).sum()),
            "harm_only": int((has_harm & ~has_rescue).sum()),
            "mixed_rescue_and_harm": int((has_rescue & has_harm).sum()),
            "neither_rescue_nor_harm": int((~has_rescue & ~has_harm).sum()),
        },
        "net_change_plots": {
            "positive": int((plots.net_rescues > 0).sum()),
            "negative": int((plots.net_rescues < 0).sum()),
            "zero": int((plots.net_rescues == 0).sum()),
        },
        "image_weighted_delta_acc1": float(plots.net_rescues.sum() / len(rows)),
        "equal_plot_weighted_delta_acc1": float(plots.delta_acc1.mean()),
        "equal_plot_weighted_definition": "mean of within-plot accuracy changes over every query plot",
    }
    ranked = plots.loc[has_harm].sort_values(["harms", "plot_idx"], ascending=[False, True])
    harm_total = int(plots.harms.sum())
    summary["harm_concentration"] = {
        f"top_{count}_plots": {
            "plots": ranked.head(count).plot_idx.tolist(),
            "actual_plot_count": min(count, len(ranked)),
            "harms": int(ranked.head(count).harms.sum()),
            "fraction_of_all_harms": (float(ranked.head(count).harms.sum() / harm_total)
                                      if harm_total else None),
        } for count in (1, 3, 5)
    }
    return plots, summary


def missed_rescue_partition(
    query_results: pd.DataFrame, routing_policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Partition uncorrected fusion errors that the fixed geo expert gets right.

    Unsupported rows take precedence. Supported failures at the strict-zero
    image/geo gates are then separated. Positive-both rows can be blocked by a
    finite threshold (including equality) or by the typed never policy.
    """
    rows = _queries(query_results)
    columns = {"geometry_supported", "eligible", "A_image", "A_geo", "score"}
    if not columns.issubset(rows):
        raise ValueError("missed-rescue analysis requires the saved geometry columns")
    for column in ("geometry_supported", "eligible"):
        if rows[column].dtype.kind != "b":
            raise ValueError(f"{column} must contain Boolean values")
    supported = rows.geometry_supported.to_numpy()
    expected_eligible = supported & (rows.fusion_predictions != rows.geo_predictions).to_numpy()
    if not np.array_equal(rows.eligible, expected_eligible):
        raise ValueError("eligible flags disagree with support and expert disagreement")
    image, geo, score = (rows[key].to_numpy(dtype=np.float64) for key in ("A_image", "A_geo", "score"))
    if not all(np.isfinite(values[supported]).all() for values in (image, geo, score)):
        raise ValueError("supported geometry scores must be finite")
    if not np.array_equal(score[supported], np.minimum(image[supported], geo[supported])):
        raise ValueError("joint score must equal the minimum image/geo score")
    threshold = validate_policy(routing_policy)
    if not np.array_equal(action_mask(routing_policy, score, rows.eligible.to_numpy()), rows.action):
        raise ValueError("saved actions do not match the supplied policy")
    missed = rows.missed_rescue.to_numpy()
    image_pass, geo_pass = image > 0.0, geo > 0.0
    positive_missed = missed & supported & image_pass & geo_pass
    masks = {
        "unsupported": missed & ~supported,
        "image_fails_only_at_zero": missed & supported & ~image_pass & geo_pass,
        "geo_fails_only_at_zero": missed & supported & image_pass & ~geo_pass,
        "both_fail_at_zero": missed & supported & ~image_pass & ~geo_pass,
        "positive_both_below_or_equal_threshold": (positive_missed if threshold is not None
                                                   else np.zeros(len(rows), dtype=bool)),
        "positive_both_never_policy_blocked": (positive_missed if threshold is None
                                               else np.zeros(len(rows), dtype=bool)),
    }
    membership = sum(mask.astype(np.int64) for mask in masks.values())
    if not np.array_equal(membership, missed.astype(np.int64)):
        raise AssertionError("missed-rescue reasons are not an exhaustive disjoint partition")
    return {
        "potential_rescues": int((rows.rescue | rows.missed_rescue).sum()),
        "rescues": int(rows.rescue.sum()), "missed_rescues": int(missed.sum()),
        "missed_rescue_plots": int(rows.loc[missed, "plot_idx"].nunique()),
        "categories": {key: {"images": int(mask.sum()),
                              "plots": int(rows.loc[mask, "plot_idx"].nunique())}
                       for key, mask in masks.items()},
        "plot_counts_are_nonadditive": True,
        "zero_gate_comparison": "strict_gt; zero fails",
    }


def confusion_transitions(query_results: pd.DataFrame, class_labels: Sequence[str]) -> pd.DataFrame:
    """Describe observed truth/fusion/geo triples without merging both-wrong actions."""
    if len(class_labels) != 18 or len(set(class_labels)) != 18 or not all(
        isinstance(value, str) and value.strip() for value in class_labels
    ):
        raise ValueError("class_labels must provide 18 distinct nonempty names")
    rows = _queries(query_results)
    keys = ["label_id_dense", "fusion_predictions", "geo_predictions"]
    result = rows.groupby(keys, sort=True).agg(
        images=("row_uid", "size"), plots=("plot_idx", "nunique"),
        actions=("action", "sum"), rescues=("rescue", "sum"), harms=("harm", "sum"),
        both_wrong_actions=("both_wrong_action", "sum"), missed_rescues=("missed_rescue", "sum"),
    ).reset_index()
    for key, label_key in zip(keys, ("true_class", "fusion_class", "geo_class")):
        result[label_key] = [class_labels[int(index)] for index in result[key]]
    result["net_rescues"] = result.rescues - result.harms
    if int(result.actions.sum()) != int(rows.action.sum()):
        raise AssertionError("confusion-transition counts failed to reconcile")
    return result


def conditional_plot_bootstrap(
    per_plot: pd.DataFrame, n_resamples: int = 10000, seed: int = 20261023
) -> dict[str, Any]:
    """Paired whole-plot bootstrap of the primary image-weighted Acc@1 change.

    Each replicate draws P plot indices with replacement from all P plots and
    divides the summed net rescues by the summed image counts. Experts,
    thresholds, decisions and checkpoint choices remain fixed. It does not
    estimate the extra uncertainty or optimism due to their validation reuse.
    """
    if (not isinstance(per_plot, pd.DataFrame) or per_plot.empty
            or not {"plot_idx", "images", "net_rescues"}.issubset(per_plot)):
        raise ValueError("per_plot must contain a nonempty set of plot counts")
    if per_plot.plot_idx.duplicated().any():
        raise ValueError("per_plot must contain one row per plot")
    for value, name in ((n_resamples, "n_resamples"), (seed, "seed")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    ordered = per_plot.sort_values("plot_idx")
    images, net = ordered.images.to_numpy(), ordered.net_rescues.to_numpy()
    if (images.dtype.kind not in "iu" or net.dtype.kind not in "iu"
            or np.any(images <= 0) or np.any(np.abs(net) > images)):
        raise ValueError("plot image/net counts must be integer and physically possible")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, 1000):
        stop = min(start + 1000, n_resamples)
        indices = rng.integers(0, len(ordered), size=(stop - start, len(ordered)))
        draws[start:stop] = net[indices].sum(axis=1) / images[indices].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    return {
        "resamples": int(n_resamples), "seed": int(seed),
        "plot_count": len(ordered), "query_count": int(images.sum()),
        "delta_acc1": float(net.sum() / images.sum()),
        "lower_95": float(lower), "upper_95": float(upper),
        "interval_contains_zero": bool(lower <= 0.0 <= upper),
        "interval_method": "percentile_95_linear_quantiles",
        "resampling_unit": "whole_plot_with_replacement",
        "statistic": "sum_net_rescues_divided_by_sum_images_in_each_resample",
        "conditional_on": "fixed_experts_checkpoint_choices_geometry_and_routing_policies",
        "threshold_reselection": False,
        "accounts_for_validation_selection_optimism": False,
    }

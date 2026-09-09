"""Selection-conditional uncertainty for fixed M4 policies.

This module performs no file access, fitting, model selection, or threshold
selection. Complete plots are resampled jointly across all images, seeds, and
policies; input actions must already be frozen effective geo overrides.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import pandas as pd


PRIMARY_POLICY = "primary_full727"
SEEDS = (1, 2, 3, 4)
STATES = ("rescue", "harm", "both_correct", "both_wrong")
CONFIDENCE_LEVEL = 0.95
MINIMUM_VALID_FRACTION = 0.95
COUNT_NAMES = (
    "row_count",
    "acted_count",
    "rescued_count",
    "harmed_count",
    "rescue_opportunity_count",
    "harm_opportunity_count",
    "both_correct_count",
    "both_wrong_count",
)
METRIC_NAMES = (
    "mean_seed_net_utility",
    "mean_seed_delta_accuracy",
    "coverage_fraction",
    "mean_seed_rescued_count",
    "mean_seed_harmed_count",
    "rescued_count",
    "harmed_count",
    "rescued_minus_harmed",
    "acted_count",
    "row_count",
    "override_precision",
    "rescue_recall",
    "harmful_override_rate",
)


class RouterUncertaintyError(ValueError):
    """The uncertainty input violates complete-plot or fixed-policy semantics."""


def _validated_inputs(
    rows: pd.DataFrame, actions_by_policy: Mapping[str, Any], *, require_primary: bool
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    required = {
        "row_uid",
        "plot_idx",
        "training_seed",
        "canonical_l3_id",
        "raw_pred",
        "geo_pred",
        "target_state",
    }
    if not isinstance(rows, pd.DataFrame) or rows.empty or not required.issubset(rows):
        raise RouterUncertaintyError(
            "nonempty rows with complete identity/target columns required"
        )
    if rows[list(required)].isna().any().any():
        raise RouterUncertaintyError("missing identity, prediction, stratum, or target")
    for name in ("row_uid", "plot_idx"):
        if not rows[name].map(lambda x: isinstance(x, str) and bool(x)).all():
            raise RouterUncertaintyError(f"{name} must contain nonempty strings")
    for name in ("training_seed", "canonical_l3_id", "raw_pred", "geo_pred"):
        if rows[name].dtype.kind not in "iu":
            raise RouterUncertaintyError(f"{name} must contain integers")
    if set(rows.training_seed) != set(SEEDS):
        raise RouterUncertaintyError("all four training seeds are required")
    if rows.duplicated(["row_uid", "training_seed"]).any():
        raise RouterUncertaintyError("duplicate row_uid/training_seed identity")
    if not set(rows.target_state).issubset(STATES):
        raise RouterUncertaintyError("unknown router target state")
    if not rows[["raw_pred", "geo_pred"]].isin(range(18)).all().all():
        raise RouterUncertaintyError("expert predictions must be dense class IDs 0..17")
    identity = rows.groupby("row_uid", sort=False).agg(
        seeds=("training_seed", "nunique"),
        plots=("plot_idx", "nunique"),
        strata=("canonical_l3_id", "nunique"),
    )
    if not identity.eq({"seeds": 4, "plots": 1, "strata": 1}).all().all():
        raise RouterUncertaintyError(
            "every image must retain all four seeds and one plot/stratum"
        )
    if not rows.groupby("plot_idx").canonical_l3_id.nunique().eq(1).all():
        raise RouterUncertaintyError(
            "each complete plot must belong to one habitat stratum"
        )
    disagreement = rows.raw_pred.to_numpy() != rows.geo_pred.to_numpy()
    if np.any(rows.target_state.isin(["rescue", "harm"]).to_numpy() & ~disagreement):
        raise RouterUncertaintyError(
            "rescue/harm target cannot occur when predictions agree"
        )
    if np.any(rows.target_state.eq("both_correct").to_numpy() & disagreement):
        raise RouterUncertaintyError(
            "both-correct target requires matching expert predictions"
        )
    if not isinstance(actions_by_policy, Mapping) or not actions_by_policy:
        raise RouterUncertaintyError("at least one frozen policy is required")
    if any(not isinstance(name, str) or not name for name in actions_by_policy):
        raise RouterUncertaintyError("policy names must be nonempty strings")
    if require_primary and PRIMARY_POLICY not in actions_by_policy:
        raise RouterUncertaintyError(f"missing primary policy {PRIMARY_POLICY}")
    masks = {}
    for name in sorted(actions_by_policy):
        mask = np.asarray(actions_by_policy[name])
        if mask.dtype != np.bool_ or mask.shape != (len(rows),):
            raise RouterUncertaintyError(
                "policy actions must be aligned one-dimensional Boolean arrays"
            )
        if np.any(mask & ~disagreement):
            raise RouterUncertaintyError(
                "effective actions cannot override agreeing experts"
            )
        masks[name] = mask.copy()
    order = (
        rows.assign(_position=np.arange(len(rows)))
        .sort_values(["row_uid", "training_seed"], kind="stable")["_position"]
        .to_numpy()
    )
    return rows.iloc[order].reset_index(drop=True).copy(), {
        name: mask[order] for name, mask in masks.items()
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=np.asarray(denominator) != 0)
    return result


def _plot_counts(rows: pd.DataFrame, masks: dict[str, np.ndarray]):
    plots = sorted(rows.plot_idx.unique())
    plot_index = {plot: i for i, plot in enumerate(plots)}
    indices = rows.plot_idx.map(plot_index).to_numpy(dtype=np.int64)
    count = np.zeros((len(plots), len(masks), 4, len(COUNT_NAMES)), dtype=np.int64)
    targets = rows.target_state.to_numpy()
    for policy_index, action in enumerate(masks.values()):
        indicators = [
            np.ones(len(rows), dtype=bool),
            action,
            action & (targets == "rescue"),
            action & (targets == "harm"),
            *(targets == state for state in STATES),
        ]
        for seed_index, seed in enumerate(SEEDS):
            seed_mask = rows.training_seed.to_numpy() == seed
            for metric_index, indicator in enumerate(indicators):
                count[:, policy_index, seed_index, metric_index] = np.bincount(
                    indices[seed_mask & indicator], minlength=len(plots)
                )
    plot_strata = rows.drop_duplicates("plot_idx").set_index("plot_idx").canonical_l3_id
    strata = {
        int(label): np.asarray(
            [plot_index[plot] for plot in plots if plot_strata[plot] == label],
            dtype=np.int64,
        )
        for label in sorted(plot_strata.unique())
    }
    return plots, count, strata


def _metrics(counts: np.ndarray) -> dict[str, np.ndarray]:
    """counts axes are replicate, policy, seed, primitive count."""
    pooled = counts.sum(axis=2)
    n, acted, rescued, harmed, opportunities = [pooled[..., i] for i in range(5)]
    net = rescued - harmed
    return {
        "mean_seed_net_utility": (counts[..., 2] - counts[..., 3]).mean(axis=2),
        "mean_seed_delta_accuracy": _ratio(
            counts[..., 2] - counts[..., 3], counts[..., 0]
        ).mean(axis=2),
        "coverage_fraction": _ratio(acted, n),
        "mean_seed_rescued_count": counts[..., 2].mean(axis=2),
        "mean_seed_harmed_count": counts[..., 3].mean(axis=2),
        "rescued_count": rescued,
        "harmed_count": harmed,
        "rescued_minus_harmed": net,
        "acted_count": acted,
        "row_count": n,
        "override_precision": _ratio(rescued, acted),
        "rescue_recall": _ratio(rescued, opportunities),
        "harmful_override_rate": _ratio(harmed, acted),
    }


def _finite_or_none(value: Any) -> float | None:
    return float(value) if np.isfinite(value) else None


def _interval(point: float, values: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(values)[np.isfinite(values)]
    fraction = len(valid) / len(values)
    available = fraction >= MINIMUM_VALID_FRACTION and len(valid) > 0
    bounds = (
        np.quantile(valid, [0.025, 0.975], method="linear")
        if available
        else [None, None]
    )
    return {
        "point_estimate": _finite_or_none(point),
        "lower": None if bounds[0] is None else float(bounds[0]),
        "upper": None if bounds[1] is None else float(bounds[1]),
        "valid_replicates": len(valid),
        "valid_fraction": fraction,
        "interval_available": available,
    }


def bootstrap_policy_comparison(
    rows: pd.DataFrame,
    actions_by_policy: Mapping[str, Any],
    *,
    replicates: int = 2000,
    random_seed: int = 20260826,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Paired habitat-stratified complete-plot percentile bootstrap.

    Counts without a ``mean_seed_`` prefix are pooled across seed realizations.
    Rates except explicitly mean-seed delta accuracy use pooled denominators.
    The long replicate table retains undefined rates as NaN; the JSON summary
    uses null for unavailable values/intervals and is strict-JSON serializable.
    """
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise RouterUncertaintyError("replicates must be a positive integer")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise RouterUncertaintyError("random_seed must be a nonnegative integer")
    frame, masks = _validated_inputs(rows, actions_by_policy, require_primary=True)
    plots, counts, strata = _plot_counts(frame, masks)
    rng = np.random.default_rng(random_seed)
    weights = np.zeros((replicates, len(plots)), dtype=np.int64)
    for replicate in range(replicates):
        for indices in strata.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            weights[replicate] += np.bincount(sampled, minlength=len(plots))
    resampled_counts = np.tensordot(weights, counts, axes=(1, 0))
    observed_counts = counts.sum(axis=0, keepdims=True)
    observed, bootstrapped = _metrics(observed_counts), _metrics(resampled_counts)
    names = list(masks)
    primary_index = names.index(PRIMARY_POLICY)
    delta = bootstrapped["mean_seed_delta_accuracy"]
    paired_delta = delta[:, [primary_index]] - delta
    replicate_table = pd.DataFrame(
        {
            "replicate": np.repeat(np.arange(replicates, dtype=np.int32), len(names)),
            "policy": np.tile(names, replicates),
            **{metric: values.reshape(-1) for metric, values in bootstrapped.items()},
            "primary_minus_policy_delta_accuracy": paired_delta.reshape(-1),
        }
    )
    policies = {}
    seed_metrics = [_metrics(observed_counts[:, :, [i], :]) for i in range(4)]
    for index, name in enumerate(names):
        variation = {}
        for metric in METRIC_NAMES:
            values = np.asarray(
                [value[metric][0, index] for value in seed_metrics], dtype=np.float64
            )
            defined = values[np.isfinite(values)]
            variation[metric] = {
                "values_by_seed": {
                    str(seed): _finite_or_none(value)
                    for seed, value in zip(SEEDS, values)
                },
                "defined_seed_count": len(defined),
                "mean": float(defined.mean()) if len(defined) else None,
                "standard_deviation_ddof0": (
                    float(defined.std(ddof=0)) if len(defined) else None
                ),
            }
        policies[name] = {
            "metrics": {
                metric: _interval(observed[metric][0, index], values[:, index])
                for metric, values in bootstrapped.items()
            },
            "seed_variation": variation,
        }
    paired = {
        name: _interval(
            observed["mean_seed_delta_accuracy"][0, primary_index]
            - observed["mean_seed_delta_accuracy"][0, index],
            paired_delta[:, index],
        )
        for index, name in enumerate(names)
        if name != PRIMARY_POLICY
    }
    summary = {
        "schema_version": "geo_helpfulness.policy_cluster_bootstrap.v1",
        "interpretation": "descriptive_selection_conditional",
        "primary_policy": PRIMARY_POLICY,
        "training_seeds": list(SEEDS),
        "replicates": replicates,
        "random_seed": random_seed,
        "random_generator": "numpy.default_rng.PCG64",
        "confidence_level": CONFIDENCE_LEVEL,
        "quantile_method": "linear",
        "interval_method": "percentile",
        "minimum_valid_replicate_fraction": MINIMUM_VALID_FRACTION,
        "unit": "complete_plot_idx",
        "stratify_by": "canonical_l3_id",
        "pairing": "same_resamples_for_all_models",
        "retain_all_images_and_training_seeds": True,
        "plot_count": len(plots),
        "image_count": int(frame.row_uid.nunique()),
        "row_count": len(frame),
        "stratum_plot_counts": {
            str(label): len(indices) for label, indices in strata.items()
        },
        "resample_multiplicity_sha256": hashlib.sha256(
            weights.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "count_aggregation": "unprefixed_counts_pooled_across_seeds; mean_seed_counts_arithmetic_mean",
        "rate_aggregation": "pooled_counts_except_mean_seed_delta_accuracy",
        "policies": policies,
        "paired_primary_minus_comparator_delta_accuracy": paired,
    }
    return summary, replicate_table


def _subgroup_counts(rows: pd.DataFrame, action: np.ndarray) -> dict[str, Any]:
    state = rows.target_state.to_numpy()
    n, acted = len(rows), int(action.sum())
    rescued, harmed = int(np.sum(action & (state == "rescue"))), int(
        np.sum(action & (state == "harm"))
    )
    prevalence = {name: int(np.sum(state == name)) for name in STATES}
    divide = lambda numerator, denominator: (
        numerator / denominator if denominator else np.nan
    )
    raw_correct = prevalence["harm"] + prevalence["both_correct"]
    images, plots = int(rows.row_uid.nunique()), int(rows.plot_idx.nunique())
    return {
        "row_count": n,
        "image_count": images,
        "plot_count": plots,
        "acted_count": acted,
        "unique_acted_plots": int(rows.loc[action, "plot_idx"].nunique()),
        "rescued_count": rescued,
        "harmed_count": harmed,
        "rescued_minus_harmed": rescued - harmed,
        "delta_accuracy": divide(rescued - harmed, n),
        "coverage_fraction": divide(acted, n),
        "override_precision": divide(rescued, acted),
        "rescue_recall": divide(rescued, prevalence["rescue"]),
        "harmful_override_rate": divide(harmed, acted),
        "raw_correct_count": raw_correct,
        "geo_correct_count": prevalence["rescue"] + prevalence["both_correct"],
        "final_correct_count": raw_correct + rescued - harmed,
        "rescue_opportunity_count": prevalence["rescue"],
        "harm_opportunity_count": prevalence["harm"],
        **{f"target_{name}_count": count for name, count in prevalence.items()},
        "low_support": images < 20 or plots < 10,
    }


def subgroup_policy_report(
    rows: pd.DataFrame, actions_by_policy: Mapping[str, Any]
) -> pd.DataFrame:
    """Long per-seed and pooled habitat/plot/predicted-pair diagnostics.

    Missing-seed subgroups remain present with zero counts and undefined rates.
    Pooled image and plot support counts deduplicate the four seed realizations.
    """
    frame, masks = _validated_inputs(rows, actions_by_policy, require_primary=False)
    geo = frame.geo_pred.astype(np.int64)
    raw = frame.raw_pred.astype(np.int64)
    groups = {
        "habitat": frame.canonical_l3_id.astype(str),
        "plot": frame.plot_idx,
        "geo_raw_pred_pair": (geo * 18 + raw).astype(str),
    }
    if "image_pred" in frame:
        if (
            frame.image_pred.dtype.kind not in "iu"
            or not frame.image_pred.isin(range(18)).all()
        ):
            raise RouterUncertaintyError("image predictions must be dense IDs 0..17")
        pair = frame.image_pred.astype(np.int64) * 18 + geo
        if "image_geo_pred_pair" in frame and not np.array_equal(
            pair, frame.image_geo_pred_pair
        ):
            raise RouterUncertaintyError("image/geo pair code mismatch")
        groups["image_geo_pred_pair"] = pair.astype(str)
    elif "image_geo_pred_pair" in frame:
        if (
            frame.image_geo_pred_pair.dtype.kind not in "iu"
            or not frame.image_geo_pred_pair.isin(range(324)).all()
        ):
            raise RouterUncertaintyError("image/geo pair codes must be integers 0..323")
        groups["image_geo_pred_pair"] = frame.image_geo_pred_pair.astype(str)
    if "geo_raw_pred_pair" in frame:
        expected = geo * 18 + raw
        if not np.array_equal(expected, frame.geo_raw_pred_pair):
            raise RouterUncertaintyError("geo/raw pair code mismatch")
    records = []
    for family, values in groups.items():
        for value in sorted(values.unique()):
            group_mask = values.eq(value).to_numpy()
            for name, action in masks.items():
                for seed in (*SEEDS, None):
                    selected = (
                        group_mask
                        if seed is None
                        else group_mask & frame.training_seed.eq(seed).to_numpy()
                    )
                    record = {
                        "group_family": family,
                        "group_value": value,
                        "policy": name,
                        "aggregation": "pooled" if seed is None else "seed",
                        "training_seed": seed,
                    }
                    record.update(
                        _subgroup_counts(frame.loc[selected], action[selected])
                    )
                    records.append(record)
    result = pd.DataFrame.from_records(records)
    result["training_seed"] = pd.array(result["training_seed"], dtype="Int8")
    return result.sort_values(
        ["group_family", "group_value", "policy", "aggregation", "training_seed"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

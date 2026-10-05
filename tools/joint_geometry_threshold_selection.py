"""Pure validation selection for strict-support geometry thresholds.

The score is unbounded: S = min(A_image, A_geo). Threshold enumeration and
serialization match the existing router, while action eligibility is supplied
separately for each K. This module opens no files and fits no model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_router_heuristics import (
    NEVER_THRESHOLD,
    enumerate_threshold_candidates,
    validate_threshold_specification,
)


KS = (5, 10)
SEEDS = (1, 2, 3, 4)
MINIMUM_PLOTS = (20, 10, 0)
HARM_NUMERATORS = (5, 7, 9)


def _frame(rows: pd.DataFrame, columns: set[str]) -> None:
    if not isinstance(rows, pd.DataFrame) or rows.columns.duplicated().any():
        raise ValueError("Expected a DataFrame with unique column names")
    missing = columns.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _class_ids(rows: pd.DataFrame, column: str) -> np.ndarray:
    values = rows[column].to_numpy()
    if values.dtype.kind not in "iu" or np.any((values < 0) | (values >= 18)):
        raise ValueError(f"{column} must contain integer class IDs in 0..17")
    return values


def _prediction_arrays(rows: pd.DataFrame, k: int) -> tuple:
    if type(k) is not int or k not in KS:
        raise ValueError("Policy K must be 5 or 10")
    _frame(rows, {"raw_pred", "geo_pred", f"k{k}_score", f"k{k}_eligible"})
    raw, geo = (_class_ids(rows, name) for name in ("raw_pred", "geo_pred"))
    score = rows[f"k{k}_score"].to_numpy()
    if score.dtype.kind not in "fiu":
        raise ValueError("Geometry scores must be numeric")
    score = score.astype(np.float64, copy=False)
    eligible = rows[f"k{k}_eligible"].to_numpy()
    if eligible.dtype.kind != "b":
        raise ValueError("Geometry eligibility must be non-null Boolean values")
    if np.isinf(score).any() or np.any(eligible & ~np.isfinite(score)):
        raise ValueError("Scores must be finite on eligible rows; only ineligible NaNs are allowed")
    return raw, geo, score, eligible


def _validate_validation(rows: pd.DataFrame) -> None:
    _frame(rows, {"row_uid", "plot_idx", "training_seed", "label_id_dense"})
    if rows.empty:
        raise ValueError("Validation rows must not be empty")
    for column in ("row_uid", "plot_idx"):
        if not rows[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must contain nonempty string identities")
    seed = rows.training_seed.to_numpy()
    if seed.dtype.kind not in "iu" or set(seed) != set(SEEDS):
        raise ValueError("Validation must contain exactly expert seeds 1, 2, 3, 4")
    if rows.duplicated(["training_seed", "row_uid"]).any():
        raise ValueError("Duplicate validation query within a seed")
    _class_ids(rows, "label_id_dense")
    expected = set(rows.loc[rows.training_seed.eq(SEEDS[0]), "row_uid"])
    for _, group in rows.groupby("training_seed"):
        if set(group.row_uid) != expected:
            raise ValueError("Every seed must cover the same validation queries")
    if (rows.groupby("row_uid")[["plot_idx", "label_id_dense"]].nunique().to_numpy() != 1).any():
        raise ValueError("Query identity or truth differs between expert seeds")
    if not rows.groupby("plot_idx").label_id_dense.nunique().eq(1).all():
        raise ValueError("Validation plot has more than one habitat label")
    for k in KS:
        _prediction_arrays(rows, k)


def apply_policy(rows: pd.DataFrame, policy: Mapping[str, Any]) -> np.ndarray:
    """Apply a frozen threshold to label-free rows; all ties retain fusion."""
    if not isinstance(policy, Mapping):
        raise ValueError("Policy must be a mapping")
    raw, geo, score, eligible = _prediction_arrays(rows, policy.get("k"))
    threshold = validate_threshold_specification(policy.get("threshold_specification"))
    if threshold is None:
        return np.zeros(len(rows), dtype=bool)
    return eligible & np.isfinite(score) & (raw != geo) & (score > threshold)


def _counts(rows: pd.DataFrame, actions: np.ndarray) -> dict[str, Any]:
    raw, geo, truth = (rows[c].to_numpy() for c in ("raw_pred", "geo_pred", "label_id_dense"))
    rescues = int(np.count_nonzero(actions & (raw != truth) & (geo == truth)))
    harms = int(np.count_nonzero(actions & (raw == truth) & (geo != truth)))
    acted = int(np.count_nonzero(actions))
    return {
        "row_count": int(len(rows)), "acted_count": acted,
        "rescued_count": rescues, "harmed_count": harms,
        "both_wrong_count": acted - rescues - harms,
        "rescued_minus_harmed": rescues - harms,
        "delta_accuracy": (rescues - harms) / len(rows),
        "coverage_fraction": acted / len(rows),
        "unique_acted_plots": int(rows.loc[actions, "plot_idx"].nunique()),
        "harmed_to_rescued_ratio": harms / rescues if rescues else None,
    }


def _candidate_curve(rows: pd.DataFrame, k: int) -> list[dict[str, Any]]:
    """Reuse router midpoints and count whole action sets with suffix sums."""
    raw, geo, score, eligible = _prediction_arrays(rows, k)
    available = eligible & np.isfinite(score) & (raw != geo)
    specifications = enumerate_threshold_candidates(score[available])
    indices = np.flatnonzero(available & (score > 0.0))
    order = indices[np.argsort(score[indices], kind="stable")]
    ordered_scores = score[order]
    truth = rows.label_id_dense.to_numpy()
    contribution = np.column_stack((
        np.ones(len(order), dtype=np.int64),
        ((raw[order] != truth[order]) & (geo[order] == truth[order])).astype(np.int64),
        ((raw[order] == truth[order]) & (geo[order] != truth[order])).astype(np.int64),
    ))
    suffix = np.zeros((len(order) + 1, 3), dtype=np.int64)
    suffix[:-1] = np.cumsum(contribution[::-1], axis=0)[::-1]
    unique_plots = np.zeros(len(order) + 1, dtype=np.int64)
    seen: set[str] = set()
    plots = rows.plot_idx.to_numpy()
    for i in range(len(order) - 1, -1, -1):
        seen.add(plots[order[i]])
        unique_plots[i] = len(seen)
    curve = []
    for i, specification in enumerate(specifications):
        threshold = validate_threshold_specification(specification)
        position = len(order) if threshold is None else int(np.searchsorted(ordered_scores, threshold, side="right"))
        acted, rescues, harms = map(int, suffix[position])
        curve.append({
            "candidate_index": i, "threshold_specification": deepcopy(specification),
            "threshold_display": threshold, "row_count": int(len(rows)),
            "acted_count": acted, "rescued_count": rescues, "harmed_count": harms,
            "both_wrong_count": acted - rescues - harms,
            "rescued_minus_harmed": rescues - harms,
            "mean_seed_net_utility": (rescues - harms) / len(SEEDS),
            "delta_accuracy": (rescues - harms) / len(rows),
            "coverage_fraction": acted / len(rows),
            "unique_acted_plots": int(unique_plots[position]),
        })
    return curve


def select_policies(validation: pd.DataFrame) -> dict[str, Any]:
    """Select 18 common thresholds using pooled, exact-integer guardrails.

    All validation rows form the coverage/accuracy denominator, including
    agreement and insufficient-support rows. The plot requirement counts unique
    acted plot identities across all seeds. Per-seed results are descriptive.
    """
    _validate_validation(validation)
    policies, curves = {}, {}
    for k in KS:
        base_curve = _candidate_curve(validation, k)
        for minimum in MINIMUM_PLOTS:
            for numerator in HARM_NUMERATORS:
                policy_id = f"k{k}__plots_{minimum}__harm_0p{numerator}"
                curve = deepcopy(base_curve)
                best, best_rank = None, None
                for record in curve:
                    actions = record["acted_count"]
                    rescues, harms = record["rescued_count"], record["harmed_count"]
                    feasible = (
                        actions * 100 >= len(validation)
                        and record["unique_acted_plots"] >= minimum
                        and 10 * harms <= numerator * rescues
                        and rescues - harms > 0
                    )
                    record["feasible"] = bool(feasible)
                    if feasible:
                        rank = (rescues - harms, -harms, -actions, record["threshold_display"])
                        if best_rank is None or rank > best_rank:
                            best, best_rank = record, rank
                specification = deepcopy(NEVER_THRESHOLD if best is None else best["threshold_specification"])
                policy = {
                    "policy_id": policy_id, "k": k, "minimum_plots": minimum,
                    "harm_numerator": numerator, "harm_denominator": 10,
                    "minimum_coverage_fraction": 0.01,
                    "require_positive_net_utility": True,
                    "threshold_specification": specification,
                    "threshold_display": validate_threshold_specification(specification),
                    "selected_feasible": best is not None,
                    "selection_reason": "maximum_feasible_utility" if best is not None else "no_feasible_candidate_never_intervene",
                    "feasible_candidate_count": sum(record["feasible"] for record in curve),
                    "selected_candidate_index": 0 if best is None else best["candidate_index"],
                }
                action = apply_policy(validation, policy)
                policy["pooled"] = _counts(validation, action)
                policy["per_seed"] = {
                    str(seed): _counts(validation.loc[mask], action[mask])
                    for seed in SEEDS
                    for mask in [validation.training_seed.eq(seed).to_numpy()]
                }
                policy["mean_seed_net_utility"] = policy["pooled"]["rescued_minus_harmed"] / len(SEEDS)
                policy["mean_seed_delta_accuracy"] = sum(
                    record["delta_accuracy"] for record in policy["per_seed"].values()
                ) / len(SEEDS)
                policies[policy_id], curves[policy_id] = policy, curve
    return {
        "schema_version": "geo_helpfulness.geometry_threshold_selection.v1",
        "policies": policies, "selection_curves": curves,
        "metadata": {
            "k_values": list(KS), "minimum_plots_grid": list(MINIMUM_PLOTS),
            "harm_numerators": list(HARM_NUMERATORS), "harm_denominator": 10,
            "row_count": int(len(validation)), "query_count": int(validation.row_uid.nunique()),
            "plot_count": int(validation.plot_idx.nunique()), "training_seeds": list(SEEDS),
            "coverage_denominator": "all validation image-seed rows",
            "guardrail_scope": "pooled validation rows; common threshold across seeds",
            "objective": "maximize pooled rescues minus harms",
            "tie_break_order": ["fewer_harms", "fewer_actions", "higher_threshold"],
            "threshold_candidates": "typed never, zero, deduplicated float64 midpoints of adjacent distinct positive eligible-disagreement scores",
            "threshold_enumerator": "multimodal.geo_helpfulness_router_heuristics.enumerate_threshold_candidates",
            "score_bounds": None, "eligibility": "caller supplies strict reference support for each K",
        },
    }

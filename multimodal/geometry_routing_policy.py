"""Validation-only routing policies for the declared attention-fusion study.

The inputs are aligned R2 scores and fixed expert predictions. Geometry is not
recomputed here. Selection intentionally uses validation labels and must not be
presented as an independently evaluated routing result. No file I/O is done.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _mask(values: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (size,) or result.dtype.kind != "b":
        raise ValueError(f"{name} must be an aligned Boolean vector")
    return result


def _score_inputs(scores: Any, eligible: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "fiu":
        raise ValueError("scores must be a nonempty numeric vector")
    values = values.astype(np.float64, copy=False)
    allowed = _mask(eligible, len(values), "eligible")
    if not np.isfinite(values[allowed]).all():
        raise ValueError("eligible scores must be finite")
    return values, allowed


def _policy(value: float | None) -> dict[str, Any]:
    return {
        "kind": "never_intervene" if value is None else "finite",
        "value_hex": None if value is None else value.hex(),
        "value_decimal": value,
        "comparison": "strict_gt",
    }


def validate_policy(policy: Mapping[str, Any]) -> float | None:
    """Read a typed policy, requiring canonical hex and matching display value."""
    if not isinstance(policy, Mapping) or set(policy) != {
        "kind", "value_hex", "value_decimal", "comparison"
    }:
        raise ValueError("policy must use the declared four-field schema")
    if policy["comparison"] != "strict_gt":
        raise ValueError("policy comparison must be strict_gt")
    if policy["kind"] == "never_intervene":
        if policy["value_hex"] is not None or policy["value_decimal"] is not None:
            raise ValueError("typed never policy must have null threshold values")
        return None
    if policy["kind"] != "finite" or not isinstance(policy["value_hex"], str):
        raise ValueError("invalid policy kind or finite threshold")
    try:
        value = float.fromhex(policy["value_hex"])
    except ValueError as exc:
        raise ValueError("threshold is not a hexadecimal float") from exc
    display = policy["value_decimal"]
    if (
        not np.isfinite(value)
        or value < 0.0
        or policy["value_hex"] != value.hex()
        or (value == 0.0 and policy["value_hex"] != (0.0).hex())
        or isinstance(display, (bool, np.bool_))
        or not isinstance(display, (int, float, np.integer, np.floating))
        or display != value
    ):
        raise ValueError("threshold must be canonical finite nonnegative float64")
    return value


def finite_policy(value: float) -> dict[str, Any]:
    """Build a finite policy, including the fixed-zero rule used in every cell."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("threshold must be a finite nonnegative number")
    result = _policy(float(value))
    validate_policy(result)
    return result


def enumerate_policies(scores: Any, eligible: Any) -> list[dict[str, Any]]:
    """Never, zero, and distinct positive-score midpoints in ascending order.

    Only eligible disagreements define the grid. The float64 midpoint formula
    exactly follows the historical enumerator; rounded endpoints are retained
    and duplicate thresholds are removed. No threshold above the largest score
    is fabricated: the typed never policy represents no interventions.
    """
    values, allowed = _score_inputs(scores, eligible)
    positive = np.unique(values[allowed & (values > 0.0)])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        midpoints = positive[:-1] + (positive[1:] - positive[:-1]) / np.float64(2.0)
    finite = np.unique(np.r_[np.float64(0.0), midpoints])
    if not np.isfinite(finite).all():
        raise ValueError("threshold midpoint computation became nonfinite")
    return [_policy(None)] + [_policy(float(value)) for value in finite]


def action_mask(policy: Mapping[str, Any], scores: Any, eligible: Any) -> np.ndarray:
    """Apply strict threshold comparison; ineligible rows always retain fusion."""
    values, allowed = _score_inputs(scores, eligible)
    threshold = validate_policy(policy)
    if threshold is None:
        return np.zeros(len(values), dtype=bool)
    return allowed & (values > threshold)


def _class_ids(values: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if (result.shape != (size,) or result.dtype.kind not in "iu"
            or np.any((result < 0) | (result >= 18))):
        raise ValueError(f"{name} must contain aligned integer class IDs in 0..17")
    return result


def evaluate_actions(
    actions: Any,
    fusion_predictions: Any,
    geo_predictions: Any,
    labels: Any,
    plot_ids: Any,
) -> dict[str, Any]:
    """Count interventions and opportunities over the entire query cohort.

    Distinct-plot counts are provided for each subset and need not be additive:
    multiple photographs from a plot may have different outcomes. An action on
    an expert agreement is rejected because it is not a routing intervention.
    """
    action = np.asarray(actions)
    if action.ndim != 1 or not len(action):
        raise ValueError("actions must be a nonempty Boolean vector")
    n = len(action)
    action = _mask(action, n, "actions")
    fusion = _class_ids(fusion_predictions, n, "fusion_predictions")
    geo = _class_ids(geo_predictions, n, "geo_predictions")
    truth = _class_ids(labels, n, "labels")
    plots = np.asarray(plot_ids)
    if plots.shape != (n,) or not all(
        isinstance(value, str) and bool(value.strip()) for value in plots
    ):
        raise ValueError("plot_ids must contain aligned nonempty string identities")
    plots = plots.astype(str, copy=False)
    disagreement = fusion != geo
    if np.any(action & ~disagreement):
        raise ValueError("actions cannot include expert agreements")
    fusion_correct, geo_correct = fusion == truth, geo == truth
    rescue_pool = ~fusion_correct & geo_correct
    harm_pool = fusion_correct & ~geo_correct
    wrong = ~fusion_correct & ~geo_correct
    subsets = {
        "actions": action,
        "rescues": action & rescue_pool,
        "harms": action & harm_pool,
        "both_wrong_actions": action & wrong,
        "potential_rescues": rescue_pool,
        "potential_harms": harm_pool,
        "missed_rescues": ~action & rescue_pool,
        "both_wrong_disagreements": wrong & disagreement,
        "both_wrong_agreements": wrong & ~disagreement,
        "disagreements": disagreement,
        "agreements": ~disagreement,
    }
    plot_keys = {
        "actions": "action_plots",
        "rescues": "rescue_plots",
        "harms": "harm_plots",
        "both_wrong_actions": "both_wrong_action_plots",
        "potential_rescues": "potential_rescue_plots",
        "potential_harms": "potential_harm_plots",
        "missed_rescues": "missed_rescue_plots",
        "both_wrong_disagreements": "both_wrong_disagreement_plots",
        "both_wrong_agreements": "both_wrong_agreement_plots",
        "disagreements": "disagreement_plots",
        "agreements": "agreement_plots",
    }
    result: dict[str, Any] = {"query_count": n, "query_plot_count": int(len(np.unique(plots)))}
    for name, selected in subsets.items():
        result[name] = int(np.count_nonzero(selected))
        result[plot_keys[name]] = int(len(np.unique(plots[selected])))
    result["coverage"] = result["actions"] / n
    result["net_rescues"] = result["rescues"] - result["harms"]
    result["fusion_correct"] = int(np.count_nonzero(fusion_correct))
    result["geo_correct"] = int(np.count_nonzero(geo_correct))
    result["routed_correct"] = int(np.count_nonzero(np.where(action, geo_correct, fusion_correct)))
    result["delta_acc1"] = result["net_rescues"] / n
    result["oracle_top1_acc"] = (result["fusion_correct"] + result["potential_rescues"]) / n
    if result["actions"] != result["rescues"] + result["harms"] + result["both_wrong_actions"]:
        raise AssertionError("intervention accounting identity failed")
    if result["routed_correct"] - result["fusion_correct"] != result["net_rescues"]:
        raise AssertionError("top-1 improvement accounting identity failed")
    return result


def _infeasibility_reasons(policy: Mapping[str, Any], counts: Mapping[str, Any]) -> list[str]:
    if validate_policy(policy) is None:
        return ["typed_never_fallback"]
    reasons = []
    if counts["action_plots"] < 20:
        reasons.append("fewer_than_20_acted_plots")
    # Use an integer comparison to preserve the exact one-percent boundary.
    if 100 * counts["actions"] < counts["query_count"]:
        reasons.append("coverage_below_one_percent_of_all_queries")
    if 10 * counts["harms"] > 7 * counts["rescues"]:
        reasons.append("harm_cap_exceeded")
    if counts["net_rescues"] <= 0:
        reasons.append("nonpositive_net_rescue")
    return reasons


def _selection_key(record: Mapping[str, Any]) -> tuple[int, int, int, float]:
    counts = record["counts"]
    threshold = validate_policy(record["policy"])
    if threshold is None:
        raise ValueError("never-intervene policy has no finite selection rank")
    return (counts["net_rescues"], -counts["harms"], -counts["actions"], threshold)


def select_policy(
    scores: Any,
    eligible: Any,
    fusion_predictions: Any,
    geo_predictions: Any,
    labels: Any,
    plot_ids: Any,
) -> dict[str, Any]:
    """Select the declared constrained policy independently for one R3 cell.

    Guards are fixed at twenty acted plots, one percent of all queries,
    ``10H <= 7R``, and positive net rescue. Return every candidate, including
    infeasible candidates and the explicit never-intervene fallback.
    """
    values, allowed = _score_inputs(scores, eligible)
    fusion = _class_ids(fusion_predictions, len(values), "fusion_predictions")
    geo = _class_ids(geo_predictions, len(values), "geo_predictions")
    if np.any(allowed & (fusion == geo)):
        raise ValueError("eligible rows must be expert disagreements")
    records = []
    for policy in enumerate_policies(values, allowed):
        actions = action_mask(policy, values, allowed)
        counts = evaluate_actions(actions, fusion, geo, labels, plot_ids)
        reasons = _infeasibility_reasons(policy, counts)
        records.append({
            "policy": policy,
            "counts": counts,
            "feasible": not reasons,
            "infeasibility_reasons": reasons,
        })
    feasible = [record for record in records if record["feasible"]]
    if feasible:
        selected = max(feasible, key=_selection_key)
        reason = "selected_maximum_net_rescue_with_declared_tie_breaks"
    else:
        selected = records[0]
        reason = "no_feasible_finite_candidate"
    return {
        "selected_policy": selected["policy"],
        "selected_counts": selected["counts"],
        "candidate_records": records,
        "selection_reason": reason,
    }

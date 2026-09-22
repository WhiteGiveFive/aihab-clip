"""OOF-only guardrail sensitivity for the frozen geo/raw margin heuristic.

No artifact discovery or validation/test access is available in this module.
Callers must authenticate the training inputs and original heuristic state.
Only the harm limit changes; candidate boundaries and ties remain unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_router_heuristics as original
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256


SCHEMA_VERSION = "geo_helpfulness.relaxed_margin_selection.v1"
SCORE_NAME = "geo_minus_raw_margin"
RouterSelectionError = original.RouterHeuristicError
GUARDRAILS = {
    "original": deepcopy(original.GUARDRAILS),
    "relaxed": {
        **deepcopy(original.GUARDRAILS),
        "max_pooled_harmed_to_rescued_ratio": 0.7,
    },
}


def _prepare(
    audit, targets, original_state, protocol_id, rows_per_seed, plots_per_seed
):
    if not isinstance(original_state, dict):
        raise RouterSelectionError("original heuristic state must be a dictionary")
    protocol_id = protocol_id or original_state.get("protocol_id")
    kwargs = {
        "protocol_id": protocol_id,
        "expected_rows_per_seed": rows_per_seed,
        "expected_plots_per_seed": plots_per_seed,
    }
    original.validate_frozen_oof_heuristics(original_state, audit, targets, **kwargs)
    rows = original._join_training_inputs(audit, targets, **kwargs)
    return rows, protocol_id


def _declaration(rows, protocol_id, original_state):
    inherited = original._declaration(rows, protocol_id)
    return {
        **{
            key: value
            for key, value in inherited.items()
            if key not in {"schema_version", "guardrails", "fixed_comparators"}
        },
        "schema_version": SCHEMA_VERSION,
        "score_name": SCORE_NAME,
        "guardrails": deepcopy(GUARDRAILS),
        "original_state_sha256": canonical_sha256(original_state),
        "original_margin_state_sha256": canonical_sha256(
            original_state["heuristics"][SCORE_NAME]
        ),
        "relaxed_guardrail_only_change": "max_pooled_harmed_to_rescued_ratio",
        "fitting_performed": False,
    }


def _feasible(pooled: dict[str, Any], numerator: int) -> bool:
    """Use exact integer inequalities, including the H/R boundary."""
    return bool(
        pooled["acted_count"] * 100 >= pooled["row_count"]
        and pooled["unique_acted_plots"] >= 20
        and 10 * pooled["harmed_count"] <= numerator * pooled["rescued_count"]
        and pooled["rescued_minus_harmed"] > 0
    )


def _candidate_curve(rows: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    scores = rows[SCORE_NAME].to_numpy(dtype=np.float64)
    specifications = original.enumerate_threshold_candidates(scores)
    eligible = np.flatnonzero((scores > 0) & rows.raw_pred.ne(rows.geo_pred).to_numpy())
    order = eligible[np.argsort(scores[eligible], kind="stable")]
    ordered_scores = scores[order]
    state = rows.target_state.to_numpy()
    contributions = np.column_stack(
        (
            np.ones(len(order), dtype=np.int64),
            (state[order] == "rescue").astype(np.int64),
            (state[order] == "harm").astype(np.int64),
        )
    )
    suffix = np.zeros((len(order) + 1, 3), dtype=np.int64)
    suffix[:-1] = np.cumsum(contributions[::-1], axis=0)[::-1]
    unique_plots = np.zeros(len(order) + 1, dtype=np.int64)
    seen = set()
    plots = rows.plot_idx.to_numpy()
    for offset in range(len(order) - 1, -1, -1):
        seen.add(plots[order[offset]])
        unique_plots[offset] = len(seen)
    curve = []
    for index, specification in enumerate(specifications):
        threshold = original.validate_threshold_specification(specification)
        position = (
            len(order)
            if threshold is None
            else int(np.searchsorted(ordered_scores, threshold, side="right"))
        )
        actions, rescues, harms = map(int, suffix[position])
        pooled = {
            "row_count": len(rows),
            "acted_count": actions,
            "rescued_count": rescues,
            "harmed_count": harms,
            "rescued_minus_harmed": rescues - harms,
            "coverage_fraction": actions / len(rows),
            "unique_acted_plots": int(unique_plots[position]),
            "harmed_to_rescued_ratio": harms / rescues if rescues else None,
        }
        curve.append(
            {
                "candidate_index": index,
                "threshold_specification": deepcopy(specification),
                "threshold_display": threshold,
                "pooled": pooled,
                "original_feasible": _feasible(pooled, 5),
                "relaxed_feasible": _feasible(pooled, 7),
            }
        )
    return specifications, curve


def _select_from_curve(rows, specifications, curve, policy):
    feasible = [point for point in curve if point[f"{policy}_feasible"]]

    def rank(point):
        pooled = point["pooled"]
        return (
            pooled["rescued_minus_harmed"],
            -pooled["harmed_count"],
            -pooled["acted_count"],
            point["threshold_display"],
        )

    winner = max(feasible, key=rank) if feasible else curve[0]
    specification = winner["threshold_specification"]
    actions = original.apply_heuristic_threshold(
        rows[SCORE_NAME].to_numpy(),
        rows.raw_pred.to_numpy(),
        rows.geo_pred.to_numpy(),
        specification,
    )
    return {
        "score_name": SCORE_NAME,
        "threshold_specification": deepcopy(specification),
        "threshold_specification_sha256": canonical_sha256(specification),
        "threshold_display": winner["threshold_display"],
        "candidate_count": len(specifications),
        "candidate_specifications_sha256": canonical_sha256(specifications),
        "feasible_candidate_count": len(feasible),
        "selected_feasible": bool(feasible),
        "selection_reason": (
            "maximum_feasible_utility"
            if feasible
            else "no_feasible_candidate_never_intervene"
        ),
        **original._selection_counts(rows, actions),
    }


def build_selection(
    audit: pd.DataFrame,
    targets: pd.DataFrame,
    original_state: dict[str, Any],
    *,
    protocol_id: str | None = None,
    expected_rows_per_seed: int = 3378,
    expected_plots_per_seed: int = 1300,
) -> dict[str, Any]:
    """Reproduce the original policy, then select H<=0.7R on the same OOF rows.

    Expected-count overrides support synthetic testing, not scientific changes.
    The saved curve has pooled counts at each candidate. Each selected policy
    additionally retains all four per-seed counts. No seed is averaged away.
    """
    rows, protocol_id = _prepare(
        audit,
        targets,
        original_state,
        protocol_id,
        expected_rows_per_seed,
        expected_plots_per_seed,
    )
    baseline = original._select_score(rows, SCORE_NAME)
    if canonical_sha256(baseline) != canonical_sha256(
        original_state["heuristics"][SCORE_NAME]
    ):
        raise RouterSelectionError("original OOF margin selection does not reproduce")
    specifications, curve = _candidate_curve(rows)
    original_curve = _select_from_curve(rows, specifications, curve, "original")
    if canonical_sha256(baseline) != canonical_sha256(original_curve):
        raise RouterSelectionError("original margin curve parity failed")
    return {
        **_declaration(rows, protocol_id, original_state),
        "original_selection_reproduced": True,
        "policies": {
            "original": baseline,
            "relaxed": _select_from_curve(rows, specifications, curve, "relaxed"),
        },
        "candidate_curve": curve,
        "candidate_curve_sha256": canonical_sha256(curve),
    }


def validate_frozen_selection(
    state: dict[str, Any],
    audit: pd.DataFrame,
    targets: pd.DataFrame,
    original_state: dict[str, Any],
    *,
    protocol_id: str | None = None,
    expected_rows_per_seed: int = 3378,
    expected_plots_per_seed: int = 1300,
) -> None:
    """Authenticate input identity and apply frozen policies without reselection.

    The caller must first authenticate ``state`` against its committed receipt.
    This routine checks selected counts and metadata, not curve optimality.
    """
    rows, protocol_id = _prepare(
        audit,
        targets,
        original_state,
        protocol_id,
        expected_rows_per_seed,
        expected_plots_per_seed,
    )
    declaration = _declaration(rows, protocol_id, original_state)
    extra = {
        "original_selection_reproduced",
        "policies",
        "candidate_curve",
        "candidate_curve_sha256",
    }
    if not isinstance(state, dict) or set(state) != set(declaration) | extra:
        raise RouterSelectionError("frozen margin selection schema differs")
    if (
        canonical_sha256({key: state[key] for key in declaration})
        != canonical_sha256(declaration)
        or state["original_selection_reproduced"] is not True
    ):
        raise RouterSelectionError("frozen margin declaration differs")
    if not isinstance(state["policies"], dict) or set(state["policies"]) != {
        "original",
        "relaxed",
    }:
        raise RouterSelectionError("frozen margin policy set differs")
    if canonical_sha256(state["policies"]["original"]) != canonical_sha256(
        original_state["heuristics"][SCORE_NAME]
    ):
        raise RouterSelectionError("frozen original margin policy differs")
    curve = state["candidate_curve"]
    if (
        not isinstance(curve, list)
        or len(curve) < 2
        or state["candidate_curve_sha256"] != canonical_sha256(curve)
    ):
        raise RouterSelectionError("frozen margin curve hash differs")
    for policy, numerator in (("original", 5), ("relaxed", 7)):
        selected = state["policies"][policy]
        if not isinstance(selected, dict) or set(selected) != set(
            state["policies"]["original"]
        ):
            raise RouterSelectionError("frozen selected margin schema differs")
        specification = selected["threshold_specification"]
        threshold = original.validate_threshold_specification(specification)
        specifications = [point["threshold_specification"] for point in curve]
        if (
            selected["score_name"] != SCORE_NAME
            or selected["threshold_specification_sha256"]
            != canonical_sha256(specification)
            or selected["threshold_display"] != threshold
            or selected["candidate_count"] != len(curve)
            or selected["candidate_specifications_sha256"]
            != canonical_sha256(specifications)
            or specification not in specifications
        ):
            raise RouterSelectionError("frozen margin threshold metadata differs")
        actions = original.apply_heuristic_threshold(
            rows[SCORE_NAME].to_numpy(),
            rows.raw_pred.to_numpy(),
            rows.geo_pred.to_numpy(),
            specification,
        )
        counts = original._selection_counts(rows, actions)
        if canonical_sha256({key: selected[key] for key in counts}) != canonical_sha256(
            counts
        ):
            raise RouterSelectionError("frozen margin action counts do not reproduce")
        feasible_count = sum(point[f"{policy}_feasible"] for point in curve)
        feasible = threshold is not None and _feasible(counts["pooled"], numerator)
        expected_reason = (
            "maximum_feasible_utility"
            if feasible
            else "no_feasible_candidate_never_intervene"
        )
        if (
            selected["selected_feasible"] is not feasible
            or selected["selection_reason"] != expected_reason
            or selected["feasible_candidate_count"] != feasible_count
            or (feasible_count > 0) != feasible
            or (not feasible and threshold is not None)
        ):
            raise RouterSelectionError("frozen margin guardrail or fallback differs")


__all__ = [
    "SCHEMA_VERSION",
    "SCORE_NAME",
    "GUARDRAILS",
    "RouterSelectionError",
    "build_selection",
    "validate_frozen_selection",
]

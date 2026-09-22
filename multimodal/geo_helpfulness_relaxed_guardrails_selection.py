"""Development-only threshold sensitivity for the existing frozen LR routers.

This additive study reuses the original threshold enumeration and counts. It
does not fit routers, choose another C, import an evaluation runner, or open files.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256
from multimodal.geo_helpfulness_router_heuristics import (
    NEVER_THRESHOLD,
    apply_heuristic_threshold,
    validate_threshold_specification,
)
from multimodal import geo_helpfulness_router_selection as original


STUDY_ID = "relaxed_guardrails_lr_v1"
SCHEMA_VERSION = "geo_helpfulness.relaxed_guardrails_selection.v1"
MINIMUM_PLOTS = (20, 10, 0)
HARM_NUMERATORS = (5, 7, 9)
RouterSelectionError = original.RouterSelectionError


def _guardrails(minimum_plots: int, harm_numerator: int) -> dict[str, Any]:
    if type(minimum_plots) is not int or minimum_plots not in MINIMUM_PLOTS:
        raise RouterSelectionError("minimum_plots must be one of 20, 10, 0")
    if type(harm_numerator) is not int or harm_numerator not in HARM_NUMERATORS:
        raise RouterSelectionError("harm_numerator must be one of 5, 7, 9")
    return {
        "guardrail_id": f"plots_{minimum_plots}__harm_0p{harm_numerator}",
        "minimum_plots": minimum_plots,
        "harm_numerator": harm_numerator,
        "harm_denominator": 10,
        "minimum_coverage_fraction": 0.01,
        "require_positive_net_utility": True,
    }


GUARDRAILS = tuple(
    _guardrails(plots, numerator)
    for plots in MINIMUM_PLOTS
    for numerator in HARM_NUMERATORS
)


def policy_id(variant: str, guardrail_id: str) -> str:
    """Return a policy identity from the predeclared three-by-nine grid."""
    if type(variant) is not str or variant not in original.VARIANTS:
        raise RouterSelectionError("unknown frozen LR variant")
    if type(guardrail_id) is not str or guardrail_id not in {
        item["guardrail_id"] for item in GUARDRAILS
    }:
        raise RouterSelectionError("unknown relaxed guardrail identity")
    return f"relaxed__{variant}__{guardrail_id}"


def _select_from_curve(
    rows: pd.DataFrame, baseline: dict[str, Any], guardrails: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(baseline)
    winner, best_rank = deepcopy(NEVER_THRESHOLD), None
    for record in result["candidate_curve"]:
        actions = record["acted_count"]
        rescues, harms = record["rescued_count"], record["harmed_count"]
        net = rescues - harms
        feasible = (
            actions * 100 >= record["row_count"]
            and record["unique_acted_plots"] >= guardrails["minimum_plots"]
            and 10 * harms <= guardrails["harm_numerator"] * rescues
            and net > 0
        )
        record["feasible"] = bool(feasible)
        if feasible:
            threshold = validate_threshold_specification(
                record["threshold_specification"]
            )
            rank = (net, -harms, -actions, threshold)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                winner = deepcopy(record["threshold_specification"])
    actions = apply_heuristic_threshold(
        rows.score.to_numpy(),
        rows.raw_pred.to_numpy(),
        rows.geo_pred.to_numpy(),
        winner,
    )
    per_seed = {}
    for seed in original.SEEDS:
        mask = rows.training_seed.eq(seed).to_numpy()
        per_seed[str(seed)] = original._action_counts(rows.loc[mask], actions[mask])
    result.update(
        threshold_specification=winner,
        threshold_specification_sha256=canonical_sha256(winner),
        threshold_display=validate_threshold_specification(winner),
        selected_feasible=best_rank is not None,
        selection_reason=(
            "maximum_feasible_utility"
            if best_rank is not None
            else "no_feasible_candidate_never_intervene"
        ),
        feasible_candidate_count=sum(
            record["feasible"] for record in result["candidate_curve"]
        ),
        per_seed=per_seed,
        pooled=original._action_counts(rows, actions),
        mean_seed_net_utility=sum(
            record["rescued_minus_harmed"] for record in per_seed.values()
        )
        / len(original.SEEDS),
        mean_seed_delta_accuracy=sum(
            record["delta_accuracy"] for record in per_seed.values()
        )
        / len(original.SEEDS),
        guardrails=deepcopy(guardrails),
    )
    return result


def select_common_threshold(
    rows: pd.DataFrame, *, minimum_plots: int, harm_numerator: int
) -> dict[str, Any]:
    """Select a threshold with exact integer guardrails and original tie rules."""
    guardrails = _guardrails(minimum_plots, harm_numerator)
    validated = original._threshold_rows(rows)
    baseline = original.select_common_threshold(validated)
    return _select_from_curve(validated, baseline, guardrails)


def _validate_original_selection(selection: dict[str, Any]) -> None:
    if (
        not isinstance(selection, dict)
        or selection.get("schema_version") != "geo_helpfulness.m4_selection.v1"
        or selection.get("protocol_id") != "protocol_v1"
        or not isinstance(selection.get("variants"), dict)
        or set(selection["variants"]) != set(original.VARIANTS)
    ):
        raise RouterSelectionError("expected the sealed original M4 selection")
    payload = {k: v for k, v in selection.items() if k != "selection_sha256"}
    if selection.get("selection_sha256") != canonical_sha256(payload):
        raise RouterSelectionError("original selection self-hash mismatch")
    for chosen in selection["variants"].values():
        if (
            not isinstance(chosen, dict)
            or not isinstance(chosen.get("regularization"), dict)
            or not isinstance(chosen.get("threshold"), dict)
        ):
            raise RouterSelectionError("original selected variant is incomplete")
        reg = chosen["regularization"]
        value = reg.get("selected_C")
        if (
            type(value) not in (int, float)
            or value not in original.C_VALUES
            or reg.get("selected_C_hex") != float(value).hex()
        ):
            raise RouterSelectionError("original selected C identity is inconsistent")


def build_selection(
    original_selection: dict[str, Any], validation_actions: pd.DataFrame
) -> dict[str, Any]:
    """Authenticate original LR policies, then select all 27 development policies.

    The caller authenticates the input artifacts against their sealed manifests
    and records the study predeclaration before invoking this function. The
    action table may include the five original heuristic policies; those rows
    are not used for threshold selection. No C selection occurs here.
    """
    _validate_original_selection(original_selection)
    required = {
        "policy_id",
        "acted",
        "threshold_specification",
        "row_uid",
        "plot_idx",
        "training_seed",
        "target_state",
        "raw_pred",
        "geo_pred",
        "score",
    }
    if (
        not isinstance(validation_actions, pd.DataFrame)
        or validation_actions.columns.duplicated().any()
        or not required.issubset(validation_actions)
        or validation_actions.policy_id.isna().any()
        or not set(original.VARIANTS).issubset(validation_actions.policy_id)
        or not set(validation_actions.policy_id).issubset(
            set(original.VARIANTS) | set(original.HEURISTICS)
        )
    ):
        raise RouterSelectionError(
            "validation actions require all original LR policies"
        )
    prepared, reference = {}, None
    for variant in original.VARIANTS:
        saved = original_selection["variants"][variant]["threshold"]
        source = validation_actions.loc[validation_actions.policy_id.eq(variant)]
        source = source.sort_values(["row_uid", "training_seed"], kind="stable")
        rows = original._threshold_rows(source)
        common = rows.drop(columns="score")
        if reference is not None and not common.equals(reference):
            raise RouterSelectionError("validation targets differ across LR variants")
        reference = common
        baseline = original.select_common_threshold(rows)
        if baseline != saved:
            raise RouterSelectionError(
                f"original threshold curve or selected policy does not reproduce: {variant}"
            )
        expected = apply_heuristic_threshold(
            rows.score.to_numpy(),
            rows.raw_pred.to_numpy(),
            rows.geo_pred.to_numpy(),
            saved["threshold_specification"],
        )
        if source.acted.dtype != np.dtype("bool") or not np.array_equal(
            source.acted.to_numpy(), expected
        ):
            raise RouterSelectionError("original action mask does not reproduce")
        for value in source.threshold_specification.unique():
            try:
                specification = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise RouterSelectionError("invalid saved action threshold") from exc
            if specification != saved["threshold_specification"]:
                raise RouterSelectionError(
                    "saved action threshold differs from selection"
                )
        prepared[variant] = (rows, baseline)
    result = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "original_selection_sha256": original_selection["selection_sha256"],
        "guardrails": deepcopy(list(GUARDRAILS)),
        "policies": {},
        "original_baseline_parity": {},
    }
    for variant, (rows, baseline) in prepared.items():
        for guardrail in GUARDRAILS:
            threshold = _select_from_curve(rows, baseline, guardrail)
            if guardrail["minimum_plots"] == 20 and guardrail["harm_numerator"] == 5:
                reproduced = {k: v for k, v in threshold.items() if k != "guardrails"}
                if reproduced != baseline:
                    raise RouterSelectionError("relaxed baseline differs from original")
                result["original_baseline_parity"][variant] = True
            name = policy_id(variant, guardrail["guardrail_id"])
            result["policies"][name] = {
                "variant": variant,
                "guardrail_id": guardrail["guardrail_id"],
                "regularization": deepcopy(
                    original_selection["variants"][variant]["regularization"]
                ),
                "threshold": threshold,
            }
    result["selection_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "STUDY_ID",
    "SCHEMA_VERSION",
    "GUARDRAILS",
    "RouterSelectionError",
    "policy_id",
    "select_common_threshold",
    "build_selection",
]

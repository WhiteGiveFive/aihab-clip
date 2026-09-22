"""Pure, selection-free reports for pre-frozen relaxed LR action guardrails.

The caller supplies one sealed legacy checkpoint family, unchanged expert and
LR probabilities, 35 frozen action policies, and separately supplied labels.
Development guardrails are metadata, never test-time feasibility conditions.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_legacy_router_report as legacy
from multimodal.geo_helpfulness_router_uncertainty import _interval


STUDY_ID = "relaxed_guardrails_lr_v1"
VARIANTS = legacy.VARIANTS
SEEDS = legacy.SEEDS
IDENTITY = legacy.IDENTITY
KEY = legacy.KEY
Q_COLUMNS = legacy.Q_COLUMNS
OLD_POLICIES = legacy.POLICIES
BASELINE = "primary_full727"
MINIMUM_PLOTS = (20, 10, 0)
HARM_NUMERATORS = (5, 7, 9)
HARM_DENOMINATOR = 10
MINIMUM_COVERAGE_FRACTION = 0.01
NEW_POLICIES = tuple(
    f"relaxed__{variant}__plots_{minimum}__harm_0p{numerator}"
    for variant in VARIANTS
    for minimum in MINIMUM_PLOTS
    for numerator in HARM_NUMERATORS
)
POLICIES = (*OLD_POLICIES, *NEW_POLICIES)
ACTION_COLUMNS = (
    "schema_version",
    "study_id",
    "checkpoint_family",
    *IDENTITY,
    "policy_id",
    "score",
    "threshold_specification",
    "acted",
    "raw_pred",
    "geo_pred",
    "final_pred",
)
NEVER_THRESHOLD = {
    "kind": "never_intervene",
    "comparison": "strict_gt",
    "value_hex": None,
}
SCORE_FILES = (
    "expert_metrics.json",
    "classification.json",
    "router_metrics.json",
    "policy_metrics.json",
    "bootstrap_summary.json",
    "bootstrap_replicates.parquet",
    "subgroup_metrics.parquet",
    "actions_with_outcomes.parquet",
    "scoring_targets.parquet",
    "summary.json",
    "paired_comparisons.json",
)
INTERPRETATION = (
    "descriptive_legacy_transfer_with_development_selected_relaxed_LR_guardrails"
)
SELECTION_DISCLOSURE = (
    "The unchanged LR models and expert temperatures are reused. The 27 action "
    "thresholds were selected on development validation under predeclared "
    "guardrails and frozen before these labels were supplied. Test outcomes "
    "do not determine feasibility or change any policy. No threshold, guardrail, "
    "variant, or checkpoint family is selected here. The original M4 no-go "
    "remains unchanged; these reports do not rerun a gate or promote a policy."
)
UNCERTAINTY_DISCLOSURE = (
    "Intervals are descriptive and conditional on frozen experts, temperatures, "
    "LR models, and development-selected thresholds. They do not include "
    "checkpoint selection, development threshold/guardrail selection, or "
    "repeated evaluation uncertainty. The 27 new-minus-matching-original-LR "
    "contrasts are nominal, not simultaneous confidence intervals."
)


class RelaxedGuardrailsReportError(legacy.LegacyRouterReportError):
    """A supplied family violates the frozen relaxed-policy report contract."""


def _policy_identity(name):
    for variant in VARIANTS:
        for minimum in MINIMUM_PLOTS:
            for numerator in HARM_NUMERATORS:
                if name == f"relaxed__{variant}__plots_{minimum}__harm_0p{numerator}":
                    return variant, minimum, numerator
    raise RelaxedGuardrailsReportError("unknown relaxed policy identity")


def _specification(value):
    if isinstance(value, Mapping):
        try:
            value = json.dumps(dict(value), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RelaxedGuardrailsReportError("threshold must be strict JSON") from exc
    return legacy._threshold(value)


def _declaration(frozen_policies):
    if not isinstance(frozen_policies, Mapping) or set(frozen_policies) != set(
        NEW_POLICIES
    ):
        raise RelaxedGuardrailsReportError(
            "exactly the 27 frozen relaxed declarations are required"
        )
    declarations = {}
    for name in NEW_POLICIES:
        original = frozen_policies[name]
        if not isinstance(original, Mapping):
            raise RelaxedGuardrailsReportError(
                f"{name}: frozen policy must be a mapping"
            )
        record = deepcopy(dict(original))
        variant, minimum, numerator = _policy_identity(name)
        expected = {
            "policy_id": name,
            "variant_id": variant,
            "guardrail_id": f"plots_{minimum}__harm_0p{numerator}",
            "minimum_plots": minimum,
            "harm_numerator": numerator,
            "harm_denominator": HARM_DENOMINATOR,
            "minimum_coverage_fraction": MINIMUM_COVERAGE_FRACTION,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RelaxedGuardrailsReportError(
                f"{name}: frozen policy/guardrail identity mismatch"
            )
        for key in ("minimum_plots", "harm_numerator", "harm_denominator"):
            if type(record[key]) is not int:
                raise RelaxedGuardrailsReportError(
                    f"{name}: guardrail counts must be integers"
                )
        selected = record.get("selected_C")
        if (
            type(selected) not in (int, float)
            or selected not in (0.01, 0.1, 1.0, 10.0)
            or record.get("selected_C_hex") != float(selected).hex()
        ):
            raise RelaxedGuardrailsReportError(f"{name}: invalid frozen selected LR C")
        record["threshold_specification"] = _specification(
            record.get("threshold_specification")
        )
        declarations[name] = record
    try:
        json.dumps(declarations, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RelaxedGuardrailsReportError(
            "frozen declarations must be strict JSON-compatible"
        ) from exc
    return declarations


def _blind(frame, name):
    legacy._label_guard(frame, name)
    if any(
        not isinstance(column, str)
        or column.casefold()
        in {"label", "labels", "rescued", "harmed", "canonical_l3_id"}
        or column.casefold().startswith("label_")
        or "ground_truth" in column.casefold()
        or "target" in column.casefold()
        or "true_" in column.casefold()
        or (column.casefold().endswith("_correct") and column not in Q_COLUMNS)
        for column in frame.columns
    ):
        raise RelaxedGuardrailsReportError(
            f"{name}: labels/targets/outcomes must remain separate"
        )


def _inputs(experts, routers, actions, declarations):
    legacy._require_frame(
        experts,
        (
            *IDENTITY,
            "file",
            "checkpoint_family",
            *legacy.FEATURE_COLUMNS,
            *legacy.PROBABILITY_COLUMNS,
        ),
        "experts",
    )
    legacy._require_frame(
        routers,
        (*IDENTITY, "checkpoint_family", "variant_id", "C_hex", *Q_COLUMNS, "score"),
        "routers",
    )
    legacy._require_frame(actions, ACTION_COLUMNS, "actions")
    for name, frame in (
        ("experts", experts),
        ("routers", routers),
        ("actions", actions),
    ):
        _blind(frame, name)
    if set(actions.columns) != set(ACTION_COLUMNS):
        raise RelaxedGuardrailsReportError(
            "actions must retain the exact 13-column legacy schema"
        )
    if (
        experts.checkpoint_family.nunique() != 1
        or not experts.checkpoint_family.isin(legacy.CHECKPOINT_FAMILIES).all()
    ):
        raise RelaxedGuardrailsReportError(
            "exactly one recognized checkpoint family is required"
        )
    if set(actions.policy_id) != set(POLICIES):
        raise RelaxedGuardrailsReportError(
            "exactly the 35 original and relaxed action policies are required"
        )
    router_frames = legacy._validated_routers(routers, experts)
    old = actions.loc[actions.policy_id.isin(OLD_POLICIES)].copy()
    frames, _, specifications = legacy._validated_actions(old, experts, router_frames)
    if specifications[BASELINE] != NEVER_THRESHOLD:
        raise RelaxedGuardrailsReportError(
            "original M4 primary no-go must remain never-intervene"
        )
    raw, geo = experts.raw_pred.to_numpy(), experts.geo_pred.to_numpy()
    for name in NEW_POLICIES:
        frame = actions.loc[actions.policy_id.eq(name)].reset_index(drop=True)
        legacy._aligned(frame, experts, name)
        declaration = declarations[name]
        router = router_frames[declaration["variant_id"]]
        if not router.C_hex.eq(declaration["selected_C_hex"]).all():
            raise RelaxedGuardrailsReportError(
                f"{name}: selected C differs from unchanged LR model"
            )
        if not frame.study_id.eq(STUDY_ID).all():
            raise RelaxedGuardrailsReportError(
                f"{name}: relaxed action study identity mismatch"
            )
        if (
            frame.threshold_specification.nunique() != 1
            or _specification(frame.threshold_specification.iloc[0])
            != declaration["threshold_specification"]
        ):
            raise RelaxedGuardrailsReportError(
                f"{name}: action threshold differs from frozen declaration"
            )
        if (
            frame.score.dtype != np.dtype("float64")
            or frame.acted.dtype != np.dtype("bool")
            or not np.array_equal(frame.score.to_numpy(), router.score.to_numpy())
        ):
            raise RelaxedGuardrailsReportError(
                f"{name}: Boolean actions and exact float64 LR scores required"
            )
        for column in ("raw_pred", "geo_pred", "final_pred"):
            legacy._integers(frame, column, range(18))
        if not np.array_equal(frame.raw_pred, raw) or not np.array_equal(
            frame.geo_pred, geo
        ):
            raise RelaxedGuardrailsReportError(
                f"{name}: action expert predictions mismatch"
            )
        acted = legacy.apply_heuristic_threshold(
            frame.score.to_numpy(), raw, geo, declaration["threshold_specification"]
        )
        if not np.array_equal(frame.acted, acted) or not np.array_equal(
            frame.final_pred, np.where(acted, geo, raw)
        ):
            raise RelaxedGuardrailsReportError(
                f"{name}: frozen threshold/fallback action mismatch"
            )
        frames[name] = frame
    for name, frame in frames.items():
        for column in ("schema_version", "study_id"):
            if (
                frame[column].nunique() != 1
                or not frame[column]
                .map(lambda value: isinstance(value, str) and bool(value.strip()))
                .all()
            ):
                raise RelaxedGuardrailsReportError(
                    f"{name}: one nonempty {column} is required"
                )
    return frames, old


def _context(family):
    return {
        "study_id": STUDY_ID,
        "checkpoint_family": family,
        "training_seeds": list(SEEDS),
        "interpretation": INTERPRETATION,
        "selection_or_fitting_performed": False,
        "probability_calibration_fitted": False,
        "M4_gate_rerun": False,
        "original_M4_no_go_preserved": True,
        "checkpoint_family_selection_performed": False,
        "test_guardrail_feasibility_enforced": False,
        "independent_confirmation": False,
        "oracle_checkpoint_test_selection_biased": family == "oracle_test_best_model",
        "calibration_disclosure": legacy.CALIBRATION_DISCLOSURE,
        "checkpoint_selection_disclosure": legacy.CHECKPOINT_SELECTION_DISCLOSURE,
        "selection_disclosure": SELECTION_DISCLOSURE,
        "uncertainty_disclosure": UNCERTAINTY_DISCLOSURE,
    }


def _plot_diagnostics(targets, acted):
    """Descriptive pooled net concentration, retaining all four seed outcomes.

    The largest positive plot is chosen only to report sensitivity. Its share
    uses the sum of *positive* plot net contributions, not a potentially zero or
    negative overall net. No confidence interval or selection uses its removal.
    """
    net = acted.astype(np.int64) * (
        targets.target_state.eq("rescue").to_numpy(dtype=np.int64)
        - targets.target_state.eq("harm").to_numpy(dtype=np.int64)
    )
    work = targets.loc[:, ["plot_idx", "training_seed"]].copy()
    work["net_correct"], work["acted"] = net, acted
    plots = work.groupby("plot_idx", sort=True).agg(
        net_correct=("net_correct", "sum"), acted_count=("acted", "sum")
    )
    positive = plots.loc[plots.net_correct.gt(0), "net_correct"]
    positive_sum = int(positive.sum())
    best_plot = str(positive.idxmax()) if len(positive) else None
    best_net = int(positive.max()) if len(positive) else 0
    total_net = int(net.sum())
    per_seed_net = {
        str(seed): int(net[targets.training_seed.eq(seed).to_numpy()].sum())
        for seed in SEEDS
    }
    leave_best = targets.plot_idx.ne(best_plot).to_numpy()
    return {
        "interpretation": "descriptive_plot_concentration_not_a_gate_or_policy_selection",
        "count_basis": "pooled_image_seed_realizations",
        "plot_count": int(len(plots)),
        "actual_acted_plot_count": int(plots.acted_count.gt(0).sum()),
        "per_seed_net_correct": per_seed_net,
        "pooled_net_correct": total_net,
        "positive_net_plot_count": int(plots.net_correct.gt(0).sum()),
        "negative_net_plot_count": int(plots.net_correct.lt(0).sum()),
        "zero_net_plot_count": int(plots.net_correct.eq(0).sum()),
        "sum_positive_plot_net_correct": positive_sum,
        "largest_positive_plot_idx": best_plot,
        "largest_positive_plot_net_correct": best_net,
        "largest_positive_plot_share_of_positive_net": (
            best_net / positive_sum if positive_sum else None
        ),
        "largest_positive_plot_tie_break": "lexicographically_smallest_plot_idx",
        "leave_best_positive_plot_net_correct": total_net - best_net,
        "leave_best_positive_plot_per_seed_net_correct": {
            str(seed): int(
                net[leave_best & targets.training_seed.eq(seed).to_numpy()].sum()
            )
            for seed in SEEDS
        },
        "leave_best_positive_plot_mean_seed_net_correct": (total_net - best_net)
        / len(SEEDS),
        "plot_removal_used_for_selection_or_gate": False,
    }


def _comparisons(policies, replicates, declarations, context):
    wide = replicates.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    comparisons = {}
    for name in NEW_POLICIES:
        comparator = declarations[name]["variant_id"]
        point = (
            policies[name]["mean_seed_delta_accuracy"]
            - policies[comparator]["mean_seed_delta_accuracy"]
        )
        comparisons[name] = {
            comparator: {
                "policy_id": name,
                "comparator_policy": comparator,
                **_interval(
                    point, (wide[name] - wide[comparator]).to_numpy(dtype=np.float64)
                ),
            }
        }
    return {
        **context,
        "contrast_count": len(NEW_POLICIES),
        "metric": "mean_seed_delta_accuracy",
        "difference_sign": "named_policy_minus_matching_original_LR",
        "units": "accuracy_fraction",
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "replicates": 2000,
        "random_seed": 20260826,
        "shared_complete_plot_draws": True,
        "selection_repeated_in_bootstrap": False,
        "comparisons": comparisons,
    }


def _evaluate(experts, routers, actions, labels, frozen_policies):
    declarations = _declaration(frozen_policies)
    action_frames, old_actions = _inputs(experts, routers, actions, declarations)
    # Delegate original rows unmodified, preserving the old numerical payloads.
    result = legacy.evaluate_family(experts, routers, old_actions, labels)
    targets = result["scoring_targets.parquet"].to_pandas()
    context = _context(experts.checkpoint_family.iloc[0])
    classification, policies = (
        result["classification.json"],
        result["policy_metrics.json"],
    )
    classification.update(context)
    masks = {
        name: frame.acted.to_numpy().copy() for name, frame in action_frames.items()
    }
    raw_probability = np.stack(experts.raw_prob_calibrated)
    geo_probability = np.stack(experts.geo_prob_calibrated)
    y = targets.label_id_dense.to_numpy(dtype=np.int64)
    outcomes = [result["actions_with_outcomes.parquet"].to_pandas()]
    diagnostics = {}
    for name in NEW_POLICIES:
        frame, acted = action_frames[name], masks[name]
        declaration = declarations[name]
        policies[name] = {
            **context,
            **legacy.policy_metrics(targets, acted),
            "frozen_policy": declaration,
            "threshold_specification": declaration["threshold_specification"],
            "raw_fallback_selected": declaration["threshold_specification"]["kind"]
            == "never_intervene",
            "all_predictions_equal_raw": bool(
                np.array_equal(frame.final_pred, targets.raw_pred)
            ),
        }
        selected = np.where(acted[:, None], geo_probability, raw_probability)
        classification["policies"][name] = legacy._classification_record(
            targets, selected, frame.final_pred.to_numpy(dtype=np.int64)
        )
        outcome = frame.copy()
        for column in (
            "file",
            "label_id_dense",
            "canonical_l3_id",
            "label_name",
            "target_state",
        ):
            outcome[column] = targets[column].to_numpy()
        for expert in ("raw", "geo", "final"):
            outcome[f"{expert}_correct"] = outcome[f"{expert}_pred"].to_numpy() == y
        outcome["rescued"] = acted & targets.target_state.eq("rescue").to_numpy()
        outcome["harmed"] = acted & targets.target_state.eq("harm").to_numpy()
        outcome["net_correct"] = outcome.rescued.to_numpy(
            dtype=np.int8
        ) - outcome.harmed.to_numpy(dtype=np.int8)
        outcomes.append(outcome)
        diagnostics[name] = _plot_diagnostics(targets, acted)
    bootstrap, replicates = legacy.bootstrap_policy_comparison(
        targets, masks, replicates=2000, random_seed=20260826
    )
    bootstrap.update(
        {
            **context,
            "policy_selection_refit_in_resamples": False,
            "selection_repeated_in_bootstrap": False,
            "primary_policy_role": "original_frozen_M4_logistic_no_go_baseline",
        }
    )
    paired = _comparisons(policies, replicates, declarations, context)
    result.update(
        {
            "bootstrap_summary.json": bootstrap,
            "bootstrap_replicates.parquet": legacy._table(replicates),
            "subgroup_metrics.parquet": legacy._table(
                legacy.subgroup_policy_report(targets, masks)
            ),
            "actions_with_outcomes.parquet": legacy._table(
                pd.concat(outcomes, ignore_index=True).sort_values(
                    ["policy_id", *KEY], kind="stable"
                )
            ),
            "paired_comparisons.json": paired,
            "summary.json": {
                **context,
                "row_count": len(targets),
                "image_count": int(targets.row_uid.nunique()),
                "plot_count": int(targets.plot_idx.nunique()),
                "router_count": len(VARIANTS),
                "policy_count": len(POLICIES),
                "original_policy_count": len(OLD_POLICIES),
                "new_policy_count": len(NEW_POLICIES),
                "original_baseline_policy": BASELINE,
                "new_policy_ids": list(NEW_POLICIES),
                "raw_classification": classification["experts"]["raw"]["seed_summary"],
                "groups": {
                    name: {
                        "policy_id": name,
                        "variant_id": declarations[name]["variant_id"],
                        "guardrail_id": declarations[name]["guardrail_id"],
                        "frozen_policy": declarations[name],
                        "four_state": result["router_metrics.json"][
                            declarations[name]["variant_id"]
                        ]["seed_summary"],
                        "classification": classification["policies"][name][
                            "seed_summary"
                        ],
                        "policy": policies[name],
                        "plot_diagnostics": diagnostics[name],
                        "bootstrap_delta_accuracy": bootstrap["policies"][name][
                            "metrics"
                        ]["mean_seed_delta_accuracy"],
                        "paired_comparisons": paired["comparisons"][name],
                    }
                    for name in NEW_POLICIES
                },
            },
        }
    )
    return result


def evaluate_family(
    experts: pd.DataFrame,
    routers: pd.DataFrame,
    actions: pd.DataFrame,
    labels: pd.DataFrame,
    frozen_policies: Mapping,
) -> dict[str, Any]:
    """Report one sealed family, three unchanged LR models and 35 policies.

    The external wrapper authenticates inputs and seals predictions before
    reading labels. This function replays action rules and derives targets;
    it performs no file I/O, fitting, threshold selection, or test-time gate.
    """
    try:
        return _evaluate(experts, routers, actions, labels, frozen_policies)
    except ValueError as exc:
        if isinstance(exc, RelaxedGuardrailsReportError):
            raise
        raise RelaxedGuardrailsReportError(str(exc)) from exc


__all__ = ["RelaxedGuardrailsReportError", "STUDY_ID", "SCORE_FILES", "evaluate_family"]

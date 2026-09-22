"""Pure reports for development-frozen relaxed RF policies on legacy experts.

This module consumes sealed caller-supplied tables and separate labels. It does
not fit, select, calibrate, discover files, or enforce development guardrails
against evaluation outcomes. All 17 predecessor policies remain intact.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_legacy_capacity_report as capacity
from multimodal import geo_helpfulness_legacy_router_report as legacy
from multimodal import geo_helpfulness_relaxed_guardrails_report as relaxed_lr
from multimodal.geo_helpfulness_router_uncertainty import _interval


STUDY_ID = "relaxed_guardrails_rf_v1"
VARIANTS = legacy.VARIANTS
SEEDS = legacy.SEEDS
IDENTITY = legacy.IDENTITY
KEY = legacy.KEY
Q_COLUMNS = legacy.Q_COLUMNS
OLD_POLICIES = capacity.POLICIES
ROUTER_POLICIES = capacity.ROUTER_POLICIES
BASELINE = "primary_full727"
MINIMUM_PLOTS = (20, 10, 0)
HARM_NUMERATORS = (5, 7, 9)
HARM_DENOMINATOR = 10
MINIMUM_COVERAGE_FRACTION = 0.01
NEW_POLICIES = tuple(
    f"relaxed__rf__{variant}__plots_{minimum}__harm_0p{numerator}"
    for variant in VARIANTS
    for minimum in MINIMUM_PLOTS
    for numerator in HARM_NUMERATORS
)
POLICIES = (*OLD_POLICIES, *NEW_POLICIES)
ACTION_COLUMNS = relaxed_lr.ACTION_COLUMNS
NEVER_THRESHOLD = capacity.NEVER_THRESHOLD
SCORE_FILES = capacity.SCORE_FILES
INTERPRETATION = (
    "descriptive_legacy_transfer_with_development_selected_relaxed_RF_guardrails"
)
SELECTION_DISCLOSURE = (
    "The previously selected RF configurations, fitted models, input transforms, "
    "and expert temperatures are reused unchanged. All 27 new action thresholds "
    "were selected on development validation under the predeclared guardrail "
    "grid and frozen before these evaluation labels were supplied. Evaluation "
    "outcomes do not determine feasibility or change a policy. No model, "
    "threshold, guardrail, variant, or checkpoint family is selected here. "
    "The original M4 no-go remains unchanged; no gate is rerun or policy promoted."
)
UNCERTAINTY_DISCLOSURE = (
    "Intervals are descriptive and conditional on frozen experts, temperatures, "
    "RF models, and development-selected thresholds. They exclude checkpoint "
    "selection, development model/threshold/guardrail selection, and adaptive "
    "repeated-evaluation uncertainty. The 27 new-minus-matching-original-RF "
    "contrasts are nominal, not simultaneous confidence intervals. Predecessor "
    "contrasts are retained as reference payloads, not counted as new contrasts."
)


class RelaxedRFReportError(legacy.LegacyRouterReportError):
    """A supplied declaration or sealed family violates the RF report contract."""


def _policy_identity(name):
    for variant in VARIANTS:
        for minimum in MINIMUM_PLOTS:
            for numerator in HARM_NUMERATORS:
                if (
                    name
                    == f"relaxed__rf__{variant}__plots_{minimum}__harm_0p{numerator}"
                ):
                    return variant, minimum, numerator
    raise RelaxedRFReportError("unknown relaxed RF policy identity")


def _new_declarations(new_policies, old_policies):
    if not isinstance(new_policies, Mapping) or set(new_policies) != set(NEW_POLICIES):
        raise RelaxedRFReportError(
            "exactly the 27 frozen relaxed RF declarations are required"
        )
    result = {}
    for name in NEW_POLICIES:
        original = new_policies[name]
        if not isinstance(original, Mapping):
            raise RelaxedRFReportError(f"{name}: frozen policy must be a mapping")
        record = deepcopy(dict(original))
        variant, minimum, numerator = _policy_identity(name)
        reference = "capacity__rf__" + variant
        expected = {
            "policy_id": name,
            "learner": "rf",
            "variant_id": variant,
            "original_policy_id": reference,
            "config_id": old_policies[reference]["config_id"],
            "guardrail_id": f"plots_{minimum}__harm_0p{numerator}",
            "minimum_plots": minimum,
            "harm_numerator": numerator,
            "harm_denominator": HARM_DENOMINATOR,
            "minimum_coverage_fraction": MINIMUM_COVERAGE_FRACTION,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RelaxedRFReportError(
                f"{name}: frozen RF model/policy/guardrail identity mismatch"
            )
        for field in ("minimum_plots", "harm_numerator", "harm_denominator"):
            if type(record[field]) is not int:
                raise RelaxedRFReportError(f"{name}: guardrail counts must be integers")
        if record.get("C_hex", "") != "":
            raise RelaxedRFReportError(f"{name}: RF has no LR regularization C")
        record["threshold_specification"] = capacity._specification(
            record.get("threshold_specification")
        )
        result[name] = record
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RelaxedRFReportError(
            "frozen declarations must be strict JSON-compatible"
        ) from exc
    return result


def validate_new_policies(
    new_policies: Mapping, old_policies: Mapping
) -> dict[str, dict]:
    """Validate and normalize metadata only, before any feature table is read.

    The caller authenticates the old and new declarations. This function checks
    their exact model identities and typed thresholds without touching features,
    labels, files, model references, or any selection implementation.
    """
    try:
        original = capacity._declaration(old_policies)
        return _new_declarations(new_policies, original)
    except ValueError as exc:
        if isinstance(exc, RelaxedRFReportError):
            raise
        raise RelaxedRFReportError(str(exc)) from exc


def _inputs(experts, routers, actions, old_declarations, new_declarations):
    legacy._require_frame(actions, ACTION_COLUMNS, "actions")
    for name, frame in (
        ("experts", experts),
        ("routers", routers),
        ("actions", actions),
    ):
        legacy._require_frame(frame, (), name)
        relaxed_lr._blind(frame, name)
    if set(actions.columns) != set(ACTION_COLUMNS):
        raise RelaxedRFReportError(
            "actions must retain the exact 13-column legacy schema"
        )
    if set(actions.policy_id) != set(POLICIES):
        raise RelaxedRFReportError(
            "exactly the 44 predecessor and relaxed RF action policies are required"
        )
    old_actions = actions.loc[actions.policy_id.isin(OLD_POLICIES)].copy()
    router_frames, frames = capacity._inputs(
        experts, routers, old_actions, old_declarations
    )
    raw, geo = experts.raw_pred.to_numpy(), experts.geo_pred.to_numpy()
    for name in NEW_POLICIES:
        frame = actions.loc[actions.policy_id.eq(name)].reset_index(drop=True)
        legacy._aligned(frame, experts, name)
        declaration = new_declarations[name]
        router = router_frames[declaration["original_policy_id"]]
        if not frame.study_id.eq(STUDY_ID).all():
            raise RelaxedRFReportError(
                f"{name}: relaxed RF action study identity mismatch"
            )
        if (
            frame.threshold_specification.nunique() != 1
            or legacy._threshold(frame.threshold_specification.iloc[0])
            != declaration["threshold_specification"]
        ):
            raise RelaxedRFReportError(
                f"{name}: action threshold differs from frozen declaration"
            )
        if (
            frame.score.dtype != np.dtype("float64")
            or frame.acted.dtype != np.dtype("bool")
            or not np.array_equal(frame.score.to_numpy(), router.score.to_numpy())
        ):
            raise RelaxedRFReportError(
                f"{name}: Boolean actions and exact float64 RF scores required"
            )
        for column in ("raw_pred", "geo_pred", "final_pred"):
            legacy._integers(frame, column, range(18))
        if not np.array_equal(frame.raw_pred, raw) or not np.array_equal(
            frame.geo_pred, geo
        ):
            raise RelaxedRFReportError(f"{name}: action expert predictions mismatch")
        expected = legacy.apply_heuristic_threshold(
            frame.score.to_numpy(), raw, geo, declaration["threshold_specification"]
        )
        if not np.array_equal(frame.acted, expected) or not np.array_equal(
            frame.final_pred, np.where(expected, geo, raw)
        ):
            raise RelaxedRFReportError(
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
                raise RelaxedRFReportError(f"{name}: one nonempty {column} is required")
    return frames, old_actions


def _context(family):
    return {
        "study_id": STUDY_ID,
        "checkpoint_family": family,
        "training_seeds": list(SEEDS),
        "interpretation": INTERPRETATION,
        "selection_or_fitting_performed": False,
        "model_reselection_performed": False,
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


def _comparisons(policies, replicates, declarations, context, predecessor):
    wide = replicates.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    comparisons = {}
    for name in NEW_POLICIES:
        comparator = declarations[name]["original_policy_id"]
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
        "difference_sign": "named_policy_minus_matching_original_RF",
        "units": "accuracy_fraction",
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "replicates": 2000,
        "random_seed": 20260826,
        "shared_complete_plot_draws": True,
        "selection_repeated_in_bootstrap": False,
        "comparisons": comparisons,
        "predecessor_paired_comparisons": predecessor,
    }


def _evaluate(
    experts, routers, actions, labels, old_frozen_policies, new_frozen_policies
):
    # Validate all metadata before touching supplied prediction tables.
    declarations = validate_new_policies(new_frozen_policies, old_frozen_policies)
    original = capacity._declaration(old_frozen_policies)
    action_frames, old_actions = _inputs(
        experts, routers, actions, original, declarations
    )
    # Reuse the frozen parent evaluator on its exact original rows and models.
    result = capacity.evaluate_family(
        experts, routers, old_actions, labels, old_frozen_policies
    )
    predecessor_summary = deepcopy(result["summary.json"])
    predecessor_comparisons = deepcopy(result["paired_comparisons.json"])
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
        diagnostics[name] = relaxed_lr._plot_diagnostics(targets, acted)
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
    paired = _comparisons(
        policies, replicates, declarations, context, predecessor_comparisons
    )
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
                "router_count": len(ROUTER_POLICIES),
                "relaxed_router_count": len(VARIANTS),
                "policy_count": len(POLICIES),
                "original_policy_count": len(OLD_POLICIES),
                "new_policy_count": len(NEW_POLICIES),
                "original_baseline_policy": BASELINE,
                "original_rf_policies": [
                    "capacity__rf__" + variant for variant in VARIANTS
                ],
                "new_policy_ids": list(NEW_POLICIES),
                "raw_classification": classification["experts"]["raw"]["seed_summary"],
                "groups": {
                    name: {
                        "policy_id": name,
                        "learner": "rf",
                        "variant_id": declarations[name]["variant_id"],
                        "original_policy_id": declarations[name]["original_policy_id"],
                        "config_id": declarations[name]["config_id"],
                        "guardrail_id": declarations[name]["guardrail_id"],
                        "frozen_policy": declarations[name],
                        "four_state": result["router_metrics.json"][
                            declarations[name]["original_policy_id"]
                        ]["all_rows"]["seed_summary"],
                        "disagreement": result["router_metrics.json"][
                            declarations[name]["original_policy_id"]
                        ]["disagreement"]["seed_summary"],
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
                "predecessor_summary": predecessor_summary,
            },
        }
    )
    return result


def evaluate_family(
    experts: pd.DataFrame,
    routers: pd.DataFrame,
    actions: pd.DataFrame,
    labels: pd.DataFrame,
    old_frozen_policies: Mapping,
    new_frozen_policies: Mapping,
) -> dict[str, Any]:
    """Report one sealed family, 17 unchanged policies and 27 relaxed RF policies.

    The external wrapper authenticates inputs and seals predictions before
    reading labels. This pure function validates metadata, action replay, exact
    identities, and ontology-derived targets without accessing artifacts or
    enforcing development feasibility on evaluation outcomes.
    """
    try:
        return _evaluate(
            experts, routers, actions, labels, old_frozen_policies, new_frozen_policies
        )
    except ValueError as exc:
        if isinstance(exc, RelaxedRFReportError):
            raise
        raise RelaxedRFReportError(str(exc)) from exc


__all__ = [
    "RelaxedRFReportError",
    "STUDY_ID",
    "SCORE_FILES",
    "validate_new_policies",
    "evaluate_family",
]

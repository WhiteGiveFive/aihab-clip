"""Pure reports for frozen LR/RF/XGB/MLP transfer to one legacy expert family.

Only sealed caller-supplied predictions, actions, and policy declarations are
consumed. Labels enter separately at scoring time. Nothing is fitted, selected,
calibrated, discovered on disk, or promoted through an M4 gate here.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_legacy_router_report as legacy
from multimodal.geo_helpfulness_router_uncertainty import _interval


STUDY_ID = "legacy_capacity_router_transfer_v1"
VARIANTS = legacy.VARIANTS
SEEDS = legacy.SEEDS
IDENTITY = legacy.IDENTITY
KEY = legacy.KEY
Q_COLUMNS = legacy.Q_COLUMNS
CHECKPOINT_FAMILIES = legacy.CHECKPOINT_FAMILIES
OLD_POLICIES = legacy.POLICIES
BASELINE = "primary_full727"
LEARNERS = ("lr", "rf", "xgb", "mlp")
ROUTER_POLICIES = (
    *VARIANTS,
    *("capacity__rf__" + variant for variant in VARIANTS),
    *("capacity__xgb__" + variant for variant in VARIANTS),
    *("mlp__" + variant for variant in VARIANTS),
)
NEW_POLICIES = ROUTER_POLICIES[3:]
POLICIES = (*OLD_POLICIES, *NEW_POLICIES)
NEVER_THRESHOLD = {
    "kind": "never_intervene",
    "comparison": "strict_gt",
    "value_hex": None,
}
INTERPRETATION = "descriptive_legacy_transfer_with_frozen_selected_capacity_policies"
SELECTION_DISCLOSURE = (
    "All router configurations and thresholds were selected previously on "
    "reused development validation and are transferred unchanged. No policy, "
    "threshold, feature, calibration, or checkpoint family is selected from "
    "these evaluation results. The original M4 no-go is unchanged."
)
UNCERTAINTY_DISCLOSURE = (
    "Intervals are descriptive, conditional on frozen experts, temperatures, "
    "routers, and selected policies. They do not include checkpoint selection, "
    "development model/policy selection, or repeated evaluation uncertainty; "
    "the 27 contrasts are nominal, not simultaneous confidence intervals."
)
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


class LegacyCapacityReportError(legacy.LegacyRouterReportError):
    """A sealed family violates the frozen transfer reporting contract."""


def _policy_identity(policy_id):
    if policy_id in VARIANTS:
        return "lr", policy_id
    if policy_id in OLD_POLICIES:
        return "heuristic", ""
    for learner, prefix in (
        ("rf", "capacity__rf__"),
        ("xgb", "capacity__xgb__"),
        ("mlp", "mlp__"),
    ):
        if policy_id.startswith(prefix) and policy_id[len(prefix) :] in VARIANTS:
            return learner, policy_id[len(prefix) :]
    raise LegacyCapacityReportError("unknown frozen policy identity")


def _specification(value):
    if isinstance(value, Mapping):
        try:
            value = json.dumps(dict(value), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise LegacyCapacityReportError(
                "frozen threshold must be strict JSON"
            ) from exc
    return legacy._threshold(value)


def _declaration(frozen_policies):
    if not isinstance(frozen_policies, Mapping) or set(frozen_policies) != set(
        POLICIES
    ):
        raise LegacyCapacityReportError(
            "exactly the 17 frozen policy declarations are required"
        )
    result = {}
    for name in POLICIES:
        original = frozen_policies[name]
        if not isinstance(original, Mapping):
            raise LegacyCapacityReportError(f"{name}: frozen policy must be a mapping")
        record = deepcopy(dict(original))
        learner, variant = _policy_identity(name)
        if (
            record.get("learner") != learner
            or record.get("variant_id") != variant
            or not isinstance(record.get("config_id"), str)
            or not record["config_id"].strip()
            or (learner == "heuristic" and record["config_id"] != name)
        ):
            raise LegacyCapacityReportError(
                f"{name}: frozen learner/variant/configuration mismatch"
            )
        if learner == "lr":
            if record.get("C_hex") not in {
                float(c).hex() for c in (0.01, 0.1, 1.0, 10.0)
            }:
                raise LegacyCapacityReportError(f"{name}: invalid frozen LR C")
        elif record.get("C_hex", "") != "":
            raise LegacyCapacityReportError(f"{name}: non-LR C_hex must be empty")
        record["threshold_specification"] = _specification(
            record.get("threshold_specification")
        )
        if (
            name in (BASELINE, "never_intervene")
            and record["threshold_specification"] != NEVER_THRESHOLD
        ):
            raise LegacyCapacityReportError(
                "original M4 primary no-go must remain never-intervene"
            )
        result[name] = record
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LegacyCapacityReportError(
            "frozen declarations must be strict JSON-compatible"
        ) from exc
    return result


def _blind(frame, name):
    legacy._label_guard(frame, name)
    if any(
        column.startswith("label_")
        or "target" in column.casefold()
        or "true_" in column.casefold()
        or (column.endswith("_correct") and column not in Q_COLUMNS)
        for column in frame.columns
    ):
        raise LegacyCapacityReportError(
            f"{name}: labels/targets/outcomes must remain separate"
        )


def _inputs(experts, routers, actions, declaration):
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
        (
            *IDENTITY,
            "checkpoint_family",
            "policy_id",
            "learner",
            "variant_id",
            "config_id",
            "C_hex",
            *Q_COLUMNS,
            "score",
        ),
        "routers",
    )
    legacy._require_frame(
        actions,
        (
            *IDENTITY,
            "checkpoint_family",
            "policy_id",
            "score",
            "threshold_specification",
            "acted",
            "raw_pred",
            "geo_pred",
            "final_pred",
        ),
        "actions",
    )
    for name, frame in (
        ("experts", experts),
        ("routers", routers),
        ("actions", actions),
    ):
        _blind(frame, name)
    if (
        experts.checkpoint_family.nunique() != 1
        or not experts.checkpoint_family.isin(CHECKPOINT_FAMILIES).all()
    ):
        raise LegacyCapacityReportError(
            "exactly one recognized checkpoint family is required"
        )
    if set(routers.policy_id) != set(ROUTER_POLICIES):
        raise LegacyCapacityReportError(
            "exactly the 12 selected router policies are required"
        )
    if set(actions.policy_id) != set(POLICIES):
        raise LegacyCapacityReportError(
            "exactly the 17 frozen action policies are required"
        )
    router_frames, action_frames = {}, {}
    for name in ROUTER_POLICIES:
        frame = routers.loc[routers.policy_id.eq(name)].reset_index(drop=True)
        legacy._aligned(frame, experts, name)
        record = declaration[name]
        for column in ("learner", "variant_id", "config_id"):
            if not frame[column].eq(record[column]).all():
                raise LegacyCapacityReportError(
                    f"{name}: router {column} contradicts frozen declaration"
                )
        if not frame.C_hex.eq(record.get("C_hex", "")).all():
            raise LegacyCapacityReportError(
                f"{name}: router C differs from frozen declaration"
            )
        if any(frame[column].dtype != np.float64 for column in (*Q_COLUMNS, "score")):
            raise LegacyCapacityReportError(f"{name}: router q/score must be float64")
        q = frame[list(Q_COLUMNS)].to_numpy()
        if (
            not np.isfinite(q).all()
            or np.any(q < 0)
            or np.any(q > 1)
            or not np.allclose(q.sum(axis=1), 1.0, atol=1e-8, rtol=0)
            or not np.array_equal(frame.score.to_numpy(), q[:, 0] - q[:, 1])
        ):
            raise LegacyCapacityReportError(
                f"{name}: invalid q or rescue-minus-harm score"
            )
        router_frames[name] = frame
    for name in POLICIES:
        frame = actions.loc[actions.policy_id.eq(name)].reset_index(drop=True)
        legacy._aligned(frame, experts, name)
        if frame.threshold_specification.nunique() != 1:
            raise LegacyCapacityReportError(
                f"{name}: one frozen common threshold is required"
            )
        if (
            legacy._threshold(frame.threshold_specification.iloc[0])
            != declaration[name]["threshold_specification"]
        ):
            raise LegacyCapacityReportError(
                f"{name}: action threshold differs from frozen declaration"
            )
        if (
            frame.acted.dtype != np.bool_
            or frame.score.dtype != np.float64
            or not np.isfinite(frame.score).all()
        ):
            raise LegacyCapacityReportError(
                f"{name}: actions require Boolean masks and finite float64 scores"
            )
        for column in ("raw_pred", "geo_pred", "final_pred"):
            legacy._integers(frame, column, range(18))
        if not np.array_equal(frame.raw_pred, experts.raw_pred) or not np.array_equal(
            frame.geo_pred, experts.geo_pred
        ):
            raise LegacyCapacityReportError(
                f"{name}: action expert predictions mismatch"
            )
        if name in router_frames and not np.array_equal(
            frame.score, router_frames[name].score
        ):
            raise LegacyCapacityReportError(
                f"{name}: action score differs from router q"
            )
        expected = legacy.apply_heuristic_threshold(
            frame.score.to_numpy(),
            experts.raw_pred.to_numpy(),
            experts.geo_pred.to_numpy(),
            declaration[name]["threshold_specification"],
        )
        if not np.array_equal(frame.acted, expected) or not np.array_equal(
            frame.final_pred, np.where(expected, experts.geo_pred, experts.raw_pred)
        ):
            raise LegacyCapacityReportError(
                f"{name}: frozen threshold/fallback action mismatch"
            )
        action_frames[name] = frame
    return router_frames, action_frames


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
        "independent_confirmation": False,
        "oracle_checkpoint_test_selection_biased": family == "oracle_test_best_model",
        "calibration_disclosure": legacy.CALIBRATION_DISCLOSURE,
        "checkpoint_selection_disclosure": legacy.CHECKPOINT_SELECTION_DISCLOSURE,
        "selection_disclosure": SELECTION_DISCLOSURE,
        "uncertainty_disclosure": UNCERTAINTY_DISCLOSURE,
    }


def _router_metrics(targets, frame):
    """Capacity-style summaries without importing any development selector."""
    q = frame[list(Q_COLUMNS)].to_numpy()
    y = np.asarray(
        [legacy.TARGET_ORDER.index(state) for state in targets.target_state],
        dtype=np.int64,
    )

    def one(mask):
        pred, labels = q[mask].argmax(axis=1), y[mask]
        result = legacy.classification_metrics(pred, labels, 4)
        probability_record = (
            legacy.multiclass_metrics(q[mask], labels)
            if len(labels)
            else {
                "top1_accuracy": None,
                "top3_accuracy": None,
                "log_loss": None,
                "brier_score": None,
                "true_probability_floor_count": 0,
            }
        )
        result.update(probability_record)
        result.update(
            {
                "rescue_pr_auc": legacy.binary_average_precision(
                    labels == 0, q[mask, 0]
                ),
                "harm_pr_auc": legacy.binary_average_precision(labels == 1, q[mask, 1]),
                "majority_state_accuracy": (
                    float(np.bincount(labels, minlength=4).max() / len(labels))
                    if len(labels)
                    else None
                ),
                "rescue_argmax_recall": (
                    float(np.mean(pred[labels == 0] == 0))
                    if np.any(labels == 0)
                    else None
                ),
                "harm_argmax_recall": (
                    float(np.mean(pred[labels == 1] == 1))
                    if np.any(labels == 1)
                    else None
                ),
            }
        )
        for record, state in zip(result["per_class"], legacy.TARGET_ORDER):
            record["target_state"] = state
        return result

    def summarize(eligible):
        records = {
            str(seed): one(eligible & targets.training_seed.eq(seed).to_numpy())
            for seed in SEEDS
        }
        names = (
            "top1_accuracy",
            "macro_f1",
            "log_loss",
            "brier_score",
            "rescue_pr_auc",
            "harm_pr_auc",
            "rescue_argmax_recall",
            "harm_argmax_recall",
            "majority_state_accuracy",
        )
        return {
            "per_seed": records,
            "pooled_seed_realizations": one(eligible),
            "seed_summary": {
                name: legacy.seed_summary([record[name] for record in records.values()])
                for name in names
            },
        }

    return {
        "all_rows": summarize(np.ones(len(targets), dtype=bool)),
        "disagreement": summarize(targets.raw_pred.ne(targets.geo_pred).to_numpy()),
    }


def _comparisons(policies, replicates, context):
    wide = replicates.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    comparisons = {}
    for variant in VARIANTS:
        rf, xgb, mlp = (
            "capacity__rf__" + variant,
            "capacity__xgb__" + variant,
            "mlp__" + variant,
        )
        for name, additional in ((rf, (xgb,)), (xgb, ()), (mlp, (rf, xgb))):
            comparisons[name] = {}
            for comparator in (variant, "geo_minus_raw_margin", *additional):
                point = (
                    policies[name]["mean_seed_delta_accuracy"]
                    - policies[comparator]["mean_seed_delta_accuracy"]
                )
                differences = (wide[name] - wide[comparator]).to_numpy(dtype=np.float64)
                comparisons[name][comparator] = {
                    "policy_id": name,
                    "comparator_policy": comparator,
                    **_interval(point, differences),
                }
    return {
        **context,
        "contrast_count": 27,
        "metric": "mean_seed_delta_accuracy",
        "difference_sign": "named_policy_minus_frozen_comparator",
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
    declaration = _declaration(frozen_policies)
    router_frames, action_frames = _inputs(experts, routers, actions, declaration)
    # Delegate the original subset unmodified to the frozen legacy evaluator.
    # Its policy/calibration/classification payloads remain exact baseline values.
    old_routers = routers.loc[
        routers.policy_id.isin(VARIANTS),
        [*IDENTITY, "checkpoint_family", "variant_id", "C_hex", *Q_COLUMNS, "score"],
    ].copy()
    result = legacy.evaluate_family(
        experts,
        old_routers,
        actions.loc[actions.policy_id.isin(OLD_POLICIES)].copy(),
        labels,
    )
    targets = result["scoring_targets.parquet"].to_pandas()
    family = experts.checkpoint_family.iloc[0]
    context = _context(family)
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
    for name in NEW_POLICIES:
        frame, acted = action_frames[name], masks[name]
        policies[name] = {
            **context,
            **legacy.policy_metrics(targets, acted),
            "threshold_specification": declaration[name]["threshold_specification"],
            "raw_fallback_selected": declaration[name]["threshold_specification"][
                "kind"
            ]
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
    for name, frame in router_frames.items():
        metrics = _router_metrics(targets, frame)
        old = result["router_metrics.json"].get(name)
        record = old if old is not None else legacy._router_record(targets, frame)
        result["router_metrics.json"][name] = {
            **record,
            **context,
            "policy_id": name,
            "learner": declaration[name]["learner"],
            "variant_id": declaration[name]["variant_id"],
            "config_id": declaration[name]["config_id"],
            "all_rows": metrics["all_rows"],
            "disagreement": metrics["disagreement"],
        }
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
    paired = _comparisons(policies, replicates, context)
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
                "policy_count": len(POLICIES),
                "original_policy_count": len(OLD_POLICIES),
                "new_policy_count": len(NEW_POLICIES),
                "original_baseline_policy": BASELINE,
                "new_policy_ids": list(NEW_POLICIES),
                "raw_classification": classification["experts"]["raw"]["seed_summary"],
                "groups": {
                    name: {
                        "policy_id": name,
                        "learner": declaration[name]["learner"],
                        "variant_id": declaration[name]["variant_id"],
                        "config_id": declaration[name]["config_id"],
                        "four_state": result["router_metrics.json"][name]["all_rows"][
                            "seed_summary"
                        ],
                        "disagreement": result["router_metrics.json"][name][
                            "disagreement"
                        ]["seed_summary"],
                        "classification": classification["policies"][name][
                            "seed_summary"
                        ],
                        "policy": policies[name],
                        "bootstrap_delta_accuracy": bootstrap["policies"][name][
                            "metrics"
                        ]["mean_seed_delta_accuracy"],
                        "paired_comparisons": paired["comparisons"].get(name, {}),
                    }
                    for name in ROUTER_POLICIES
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
    """Report one family, 12 selected routers and 17 pre-frozen action policies.

    The external wrapper authenticates the declaration and seals all prediction
    files before reading labels. This pure function validates exact identities,
    frozen configurations/thresholds, probability-derived actions, and ontology-
    derived family-specific targets. It never reads files or selects anything.
    """
    try:
        return _evaluate(experts, routers, actions, labels, frozen_policies)
    except ValueError as exc:
        if isinstance(exc, LegacyCapacityReportError):
            raise
        raise LegacyCapacityReportError(str(exc)) from exc


__all__ = ["LegacyCapacityReportError", "STUDY_ID", "SCORE_FILES", "evaluate_family"]

"""Selection-free reports for a frozen relaxed OOF margin heuristic.

This module performs no file I/O, fitting, threshold selection, or test-time
guardrail enforcement. Predictions/actions and labels are supplied separately
by a caller that authenticates and seals the former before reading the latter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_legacy_router_report as legacy
from multimodal.geo_helpfulness_router_uncertainty import _interval


STUDY_ID = "relaxed_guardrails_margin_v1"
ORIGINAL_POLICY = "geo_minus_raw_margin"
RELAXED_POLICY = "relaxed__geo_minus_raw_margin__plots_20__harm_0p7"
POLICIES = (ORIGINAL_POLICY, RELAXED_POLICY)
ACTION_COLUMNS = (
    "schema_version",
    "study_id",
    "checkpoint_family",
    *legacy.IDENTITY,
    "policy_id",
    "score",
    "threshold_specification",
    "acted",
    "raw_pred",
    "geo_pred",
    "final_pred",
)
SCORE_FILES = (
    "classification.json",
    "policy_metrics.json",
    "bootstrap_summary.json",
    "paired_comparisons.json",
    "summary.json",
    "scoring_targets.parquet",
    "actions_with_outcomes.parquet",
    "bootstrap_replicates.parquet",
    "subgroup_metrics.parquet",
)


class RelaxedMarginReportError(legacy.LegacyRouterReportError):
    """Supplied artifacts violate the frozen margin report contract."""


def _specification(value: Any) -> dict:
    if isinstance(value, Mapping):
        try:
            value = json.dumps(dict(value), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RelaxedMarginReportError("threshold must be strict JSON") from exc
    return legacy._threshold(value)


def _blind(frame: pd.DataFrame, name: str) -> None:
    legacy._label_guard(frame, name)
    for column in frame.columns:
        if not isinstance(column, str):
            raise RelaxedMarginReportError(f"{name}: string column names required")
        lower = column.casefold()
        if (
            lower in {"labels", "label", "rescued", "harmed", "canonical_l3_id"}
            or lower.startswith("label_")
            or "ground_truth" in lower
            or "target" in lower
            or "true_" in lower
            or lower.endswith("_correct")
        ):
            raise RelaxedMarginReportError(
                f"{name}: labels/targets/outcomes must remain separate"
            )


def _blind_experts(experts: pd.DataFrame) -> dict[str, np.ndarray]:
    legacy._require_frame(
        experts,
        (
            *legacy.IDENTITY,
            "file",
            "checkpoint_family",
            *legacy.FEATURE_COLUMNS,
            *legacy.PROBABILITY_COLUMNS,
        ),
        "experts",
    )
    _blind(experts, "experts")
    legacy._identity(experts, "experts", with_file=True)
    if not experts.checkpoint_family.eq("best_model").all():
        raise RelaxedMarginReportError("only the best_model expert family is allowed")
    if (
        set(experts.training_seed) != set(legacy.SEEDS)
        or not experts.groupby("row_uid").training_seed.nunique().eq(4).all()
    ):
        raise RelaxedMarginReportError("every image must retain all four seeds")
    for column in ("file", "plot_idx"):
        if not experts.groupby("row_uid")[column].nunique().eq(1).all():
            raise RelaxedMarginReportError(f"experts: inconsistent {column}")
    probabilities = {
        name: legacy._probability(experts, f"{name}_prob_calibrated")
        for name in ("image", "geo", "raw")
    }
    expected = legacy.build_semantic_values(probabilities)
    for column, values in expected.items():
        observed = experts[column].to_numpy()
        if values.dtype.kind in "biu":
            valid = observed.dtype.kind == values.dtype.kind and np.array_equal(
                observed, values
            )
            # Signed/unsigned integer storage does not alter categorical meaning.
            if values.dtype.kind in "iu":
                valid = observed.dtype.kind in "iu" and np.array_equal(observed, values)
        else:
            valid = (
                observed.dtype == np.dtype("float64")
                and np.isfinite(observed).all()
                and np.allclose(observed, values, atol=1e-12, rtol=1e-12)
            )
        if not valid:
            raise RelaxedMarginReportError(
                f"experts: calibrated semantic reconstruction mismatch: {column}"
            )
    for name in probabilities:
        native = legacy._probability(experts, f"{name}_prob_native_t1")
        if not np.array_equal(native.argmax(axis=1), expected[f"{name}_pred"]):
            raise RelaxedMarginReportError(f"experts: {name} native argmax mismatch")
    return probabilities


def validate_actions(
    experts: pd.DataFrame,
    actions: pd.DataFrame,
    original_spec: Mapping | str,
    relaxed_spec: Mapping | str,
) -> dict[str, pd.DataFrame]:
    """Replay both frozen thresholds without labels or any selection capability."""
    try:
        probability = _blind_experts(experts)
        legacy._require_frame(actions, ACTION_COLUMNS, "actions")
        _blind(actions, "actions")
        if set(actions.columns) != set(ACTION_COLUMNS):
            raise RelaxedMarginReportError("actions require the exact 13-column schema")
        if set(actions.policy_id) != set(POLICIES):
            raise RelaxedMarginReportError(
                "exactly the original and relaxed policies required"
            )
        specifications = dict(
            zip(POLICIES, map(_specification, (original_spec, relaxed_spec)))
        )
        expected = legacy.build_semantic_values(probability)
        score = expected[ORIGINAL_POLICY]
        raw, geo = expected["raw_pred"], expected["geo_pred"]
        frames = {}
        for name in POLICIES:
            frame = actions.loc[actions.policy_id.eq(name)].reset_index(drop=True)
            legacy._aligned(frame, experts, name)
            for column in ("schema_version", "study_id"):
                if (
                    frame[column].nunique() != 1
                    or not frame[column]
                    .map(lambda value: isinstance(value, str) and bool(value.strip()))
                    .all()
                ):
                    raise RelaxedMarginReportError(f"{name}: invalid {column}")
            if not frame.study_id.eq(STUDY_ID).all():
                raise RelaxedMarginReportError(f"{name}: incorrect study identity")
            if (
                frame.threshold_specification.nunique() != 1
                or _specification(frame.threshold_specification.iloc[0])
                != specifications[name]
            ):
                raise RelaxedMarginReportError(f"{name}: frozen threshold mismatch")
            if (
                frame.score.dtype != np.dtype("float64")
                or not np.isfinite(frame.score.to_numpy()).all()
                or not np.allclose(
                    frame.score.to_numpy(), score, atol=1e-12, rtol=1e-12
                )
                or not np.array_equal(frame.score.to_numpy(), experts[ORIGINAL_POLICY])
                or frame.acted.dtype != np.dtype("bool")
            ):
                raise RelaxedMarginReportError(f"{name}: score/Boolean action mismatch")
            for column in ("raw_pred", "geo_pred", "final_pred"):
                legacy._integers(frame, column, range(18))
            if not np.array_equal(frame.raw_pred, raw) or not np.array_equal(
                frame.geo_pred, geo
            ):
                raise RelaxedMarginReportError(f"{name}: expert prediction mismatch")
            acted = legacy.apply_heuristic_threshold(
                frame.score.to_numpy(), raw, geo, specifications[name]
            )
            if not np.array_equal(frame.acted, acted) or not np.array_equal(
                frame.final_pred, np.where(acted, geo, raw)
            ):
                raise RelaxedMarginReportError(
                    f"{name}: frozen action/final prediction mismatch"
                )
            frames[name] = frame
        return frames
    except ValueError as exc:
        if isinstance(exc, RelaxedMarginReportError):
            raise
        raise RelaxedMarginReportError(str(exc)) from exc


def _context() -> dict:
    return {
        "study_id": STUDY_ID,
        "checkpoint_family": "best_model",
        "training_seeds": list(legacy.SEEDS),
        "interpretation": "adaptive_reused_test_descriptive_exploration",
        "independent_confirmation": False,
        "selection_or_fitting_performed": False,
        "probability_calibration_fitted": False,
        "test_guardrail_feasibility_enforced": False,
        "M4_gate_rerun": False,
        "original_M4_no_go_preserved": True,
        "selection_disclosure": (
            "This exploration was proposed after earlier cleaned-test results had "
            "been viewed. Only the margin harm guardrail changes, from H<=0.5R "
            "to H<=0.7R; thresholds are selected on training OOF predictions, "
            "then frozen before this scoring stage. No test labels are used for "
            "threshold selection. Reused-test results are descriptive, not "
            "independent confirmation. The original M4 decision is unchanged."
        ),
        "uncertainty_disclosure": (
            "Intervals condition on frozen experts, temperatures and OOF-selected "
            "thresholds; they omit selection and adaptive repeated-test uncertainty."
        ),
        "calibration_disclosure": legacy.CALIBRATION_DISCLOSURE,
    }


def _ontology(value: Any) -> None:
    if not isinstance(value, (tuple, list)):
        raise RelaxedMarginReportError("fixed 18-class ontology list required")
    try:
        records = tuple(
            (
                (item["dense_id"], item["canonical_l3_id"], item["label_name"])
                if isinstance(item, Mapping)
                else tuple(item)
            )
            for item in value
        )
    except (KeyError, TypeError) as exc:
        raise RelaxedMarginReportError("malformed ontology") from exc
    if records != legacy.ONTOLOGY:
        raise RelaxedMarginReportError("fixed 18-class ontology mismatch")


def build_report(
    experts: pd.DataFrame,
    actions: pd.DataFrame,
    labels: pd.DataFrame,
    ontology: list,
    original_spec: Mapping | str,
    relaxed_spec: Mapping | str,
) -> dict[str, Any]:
    """Return nine strict-JSON/Arrow artifacts for two already-frozen policies."""
    try:
        return _build_report(
            experts, actions, labels, ontology, original_spec, relaxed_spec
        )
    except ValueError as exc:
        if isinstance(exc, RelaxedMarginReportError):
            raise
        raise RelaxedMarginReportError(str(exc)) from exc


def _build_report(experts, actions, labels, ontology, original_spec, relaxed_spec):
    _ontology(ontology)
    frames = validate_actions(experts, actions, original_spec, relaxed_spec)
    targets, probabilities = legacy._validated_experts(experts, labels)
    context = _context()
    specifications = dict(
        zip(POLICIES, map(_specification, (original_spec, relaxed_spec)))
    )
    masks = {name: frame.acted.to_numpy().copy() for name, frame in frames.items()}
    classification = {**context, "experts": {}, "policies": {}}
    for name in ("image", "geo", "raw"):
        classification["experts"][name] = legacy._classification_record(
            targets,
            probabilities[f"{name}_prob_calibrated"],
            experts[f"{name}_pred"].to_numpy(dtype=np.int64),
        )
    policy_metrics, outcomes = {}, []
    y = targets.label_id_dense.to_numpy(dtype=np.int64)
    for name in POLICIES:
        frame, acted = frames[name], masks[name]
        policy_metrics[name] = {
            **context,
            **legacy.policy_metrics(targets, acted),
            "threshold_specification": specifications[name],
            "threshold_selection_split": "development_training_oof",
            "minimum_coverage_fraction": 0.01,
            "minimum_plots": 20,
            "harm_numerator": 5 if name == ORIGINAL_POLICY else 7,
            "harm_denominator": 10,
            "raw_fallback_selected": specifications[name]["kind"] == "never_intervene",
        }
        selected = np.where(
            acted[:, None],
            probabilities["geo_prob_calibrated"],
            probabilities["raw_prob_calibrated"],
        )
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
    # The legacy helper requires a primary key. This is solely a no-action
    # baseline alias, not a model; rename it explicitly in every public output.
    alias = "primary_full727"
    bootstrap, replicates = legacy.bootstrap_policy_comparison(
        targets,
        {alias: np.zeros(len(targets), dtype=bool), **masks},
        replicates=2000,
        random_seed=20260826,
    )
    bootstrap["policies"]["never_intervene"] = bootstrap["policies"].pop(alias)
    bootstrap["baseline_policy"] = "never_intervene"
    bootstrap.pop("primary_policy")
    bootstrap["paired_baseline_minus_policy_delta_accuracy"] = bootstrap.pop(
        "paired_primary_minus_comparator_delta_accuracy"
    )
    bootstrap.update({**context, "selection_repeated_in_bootstrap": False})
    replicates["policy"] = replicates.policy.replace({alias: "never_intervene"})
    replicates = replicates.rename(
        columns={
            "primary_minus_policy_delta_accuracy": "baseline_minus_policy_delta_accuracy"
        }
    )
    wide = replicates.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    contrast = _interval(
        policy_metrics[RELAXED_POLICY]["mean_seed_delta_accuracy"]
        - policy_metrics[ORIGINAL_POLICY]["mean_seed_delta_accuracy"],
        (wide[RELAXED_POLICY] - wide[ORIGINAL_POLICY]).to_numpy(dtype=np.float64),
    )
    paired = {
        **context,
        "metric": "mean_seed_delta_accuracy",
        "units": "accuracy_fraction",
        "policy_id": RELAXED_POLICY,
        "comparator_policy": ORIGINAL_POLICY,
        "difference_sign": "relaxed_minus_original",
        "replicates": 2000,
        "random_seed": 20260826,
        "shared_complete_plot_draws": True,
        "selection_repeated_in_bootstrap": False,
        "interval": contrast,
    }
    summary = {
        **context,
        "row_count": len(targets),
        "image_count": int(targets.row_uid.nunique()),
        "plot_count": int(targets.plot_idx.nunique()),
        "policy_count": 2,
        "count_basis": "pooled_seed_image_realizations",
        "classification_aggregation": "arithmetic_mean_of_four_seed_metrics",
        "experts": {
            name: {
                metric: value["seed_summary"][metric]["mean"]
                for metric in ("top1_accuracy", "weighted_f1")
            }
            for name, value in classification["experts"].items()
        },
        "policies": {
            name: {
                "classification": classification["policies"][name]["seed_summary"],
                "policy": policy_metrics[name],
                "gain_vs_raw_interval": bootstrap["policies"][name]["metrics"][
                    "mean_seed_delta_accuracy"
                ],
            }
            for name in POLICIES
        },
        "relaxed_minus_original_interval": contrast,
    }
    result = {
        "classification.json": classification,
        "policy_metrics.json": policy_metrics,
        "bootstrap_summary.json": bootstrap,
        "paired_comparisons.json": paired,
        "summary.json": summary,
        "scoring_targets.parquet": legacy._table(targets),
        "actions_with_outcomes.parquet": legacy._table(
            pd.concat(outcomes, ignore_index=True).sort_values(
                ["policy_id", *legacy.KEY], kind="stable"
            )
        ),
        "bootstrap_replicates.parquet": legacy._table(replicates),
        "subgroup_metrics.parquet": legacy._table(
            legacy.subgroup_policy_report(targets, masks)
        ),
    }
    for name, value in result.items():
        if name.endswith(".json"):
            json.dumps(value, allow_nan=False)
    return result


__all__ = [
    "RelaxedMarginReportError",
    "STUDY_ID",
    "ORIGINAL_POLICY",
    "RELAXED_POLICY",
    "POLICIES",
    "ACTION_COLUMNS",
    "SCORE_FILES",
    "validate_actions",
    "build_report",
]

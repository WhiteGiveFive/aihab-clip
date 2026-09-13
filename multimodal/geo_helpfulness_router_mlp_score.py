"""Pure, selection-conditional development scoring of sealed MLP predictions.

The scorer consumes caller-supplied tables only. It neither fits models nor
discovers files, and it replays the original capacity policies without selecting
their configurations or thresholds again. All probability metrics retain the
frozen capacity scorer's numerical conventions; no MLP calibrator is fitted.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_router_capacity_score as capacity
from multimodal.geo_helpfulness_router_uncertainty import _interval


VARIANTS = capacity.VARIANTS
SEEDS = capacity.SEEDS
TARGET_ORDER = capacity.TARGET_ORDER
Q_COLUMNS = capacity.Q_COLUMNS
IDENTITY = capacity.IDENTITY
KEY = capacity.KEY
BASELINE = capacity.BASELINE
TIE_ATOL = capacity.TIE_ATOL
FROZEN_POLICIES = (
    BASELINE,
    *capacity.HEURISTICS,
    *(
        "capacity__" + family + "__" + variant
        for family in capacity.FAMILIES
        for variant in VARIANTS
    ),
)
SCOPE = {
    **capacity.SCOPE,
    "capacity_baseline_policies_reselected": False,
    "mlp_probability_calibration_fitted": False,
    "selection_disclosure": (
        "MLP architectures and thresholds are selected on reused development "
        "validation. The 15 capacity baseline policies remain frozen. Estimates "
        "and intervals condition on model and threshold selection, fitted "
        "routers, frozen experts, and preprocessing; repeated validation reuse "
        "and selection uncertainty are not corrected by these intervals."
    ),
}
SCORE_FILES = (
    "selection.json",
    "candidate_metrics.json",
    "expert_metrics.json",
    "classification.json",
    "policy_metrics.json",
    "threshold_curves.parquet",
    "actions.parquet",
    "bootstrap_summary.json",
    "bootstrap_replicates.parquet",
    "subgroup_metrics.parquet",
    "paired_comparisons.json",
    "summary.json",
)


class MLPScoreError(capacity.CapacityScoreError):
    """Sealed MLP scoring inputs violate the development-only contract."""


def _configurations(configurations):
    if not isinstance(configurations, (list, tuple)) or len(configurations) != 12:
        raise MLPScoreError("exactly 12 predeclared MLP configurations required")
    records, groups = [], {}
    for position, original in enumerate(configurations):
        if not isinstance(original, Mapping):
            raise MLPScoreError("MLP configuration records must be mappings")
        record = deepcopy(dict(original))
        parameters = record.get("parameters")
        if (
            not isinstance(record.get("config_id"), str)
            or not record["config_id"].strip()
            or record.get("family") != "mlp"
            or record.get("variant") not in VARIANTS
            or not isinstance(parameters, dict)
        ):
            raise MLPScoreError("invalid MLP configuration identity or parameters")
        widths = parameters.get("hidden_widths")
        if (
            not isinstance(widths, list)
            or not widths
            or any(type(width) is not int or width < 1 for width in widths)
        ):
            raise MLPScoreError(
                "hidden_widths must be a nonempty positive integer list"
            )
        record["declared_order"] = position
        record["group_id"] = "mlp__" + record["variant"]
        records.append(record)
        groups.setdefault(record["group_id"], []).append(record)
    if (
        len({record["config_id"] for record in records}) != 12
        or set(groups) != {"mlp__" + variant for variant in VARIANTS}
        or any(len(group) != 4 for group in groups.values())
    ):
        raise MLPScoreError(
            "three feature variants with four unique configurations required"
        )
    architectures = [
        [tuple(record["parameters"]["hidden_widths"]) for record in group]
        for group in groups.values()
    ]
    if any(len(set(widths)) != 4 for widths in architectures) or any(
        widths != architectures[0] for widths in architectures[1:]
    ):
        raise MLPScoreError(
            "all variants must share four architectures in the same order"
        )
    try:
        json.dumps(records, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MLPScoreError(
            "configuration records must be strict JSON-compatible"
        ) from exc
    return records, groups


def _frozen_actions(value, targets, experts, heuristic_specifications):
    required = [
        *capacity.TARGET_COLUMNS,
        "policy_id",
        "score",
        "threshold_specification",
        "acted",
        "final_pred",
    ]
    capacity._frame(value, required, "frozen capacity actions")
    if set(value.policy_id) != set(FROZEN_POLICIES):
        raise MLPScoreError("exactly all 15 frozen capacity policies are required")
    heuristic_specs, heuristic_scores = capacity._heuristics(
        experts, heuristic_specifications
    )
    masks, scores, specifications = {}, {}, {}
    for name in FROZEN_POLICIES:
        frame = capacity._aligned(
            value.loc[value.policy_id.eq(name)], targets, "frozen " + name
        )
        for column in capacity.TARGET_COLUMNS:
            if not np.array_equal(frame[column], targets[column]):
                raise MLPScoreError(
                    f"{name}: frozen action targets or expert predictions mismatch"
                )
        for column in (
            "label_id_dense",
            "image_pred",
            "raw_pred",
            "geo_pred",
            "final_pred",
        ):
            capacity._integer(frame, column, range(18))
        capacity._integer(frame, "canonical_l3_id", capacity.CANONICAL_CLASSES)
        mask, score = frame.acted.to_numpy(), frame.score.to_numpy()
        if mask.dtype != np.bool_:
            raise MLPScoreError(f"{name}: frozen actions must be Boolean")
        if score.dtype != np.float64 or not np.isfinite(score).all():
            raise MLPScoreError(f"{name}: frozen scores must be finite float64")
        specs = [
            capacity._specification(value) for value in frame.threshold_specification
        ]
        specification = specs[0]
        if any(spec != specification for spec in specs):
            raise MLPScoreError(f"{name}: frozen threshold must be common across seeds")
        expected = capacity.apply_heuristic_threshold(
            score,
            targets.raw_pred.to_numpy(),
            targets.geo_pred.to_numpy(),
            specification,
        )
        if not np.array_equal(mask, expected):
            raise MLPScoreError(
                f"{name}: frozen actions contradict score/threshold replay"
            )
        final = np.where(mask, targets.geo_pred.to_numpy(), targets.raw_pred.to_numpy())
        if not np.array_equal(frame.final_pred, final):
            raise MLPScoreError(
                f"{name}: frozen final predictions contradict expert routing"
            )
        y = targets.label_id_dense.to_numpy()
        outcomes = {
            "rescued": mask & targets.target_state.eq("rescue").to_numpy(),
            "harmed": mask & targets.target_state.eq("harm").to_numpy(),
            "raw_correct": targets.raw_pred.to_numpy() == y,
            "final_correct": final == y,
        }
        outcomes["net_correct"] = outcomes["final_correct"].astype(np.int8) - outcomes[
            "raw_correct"
        ].astype(np.int8)
        for column, expected in outcomes.items():
            if column in frame:
                kind = "iu" if column == "net_correct" else "b"
                if frame[column].dtype.kind not in kind or not np.array_equal(
                    frame[column], expected
                ):
                    raise MLPScoreError(f"{name}: frozen outcome mismatch for {column}")
        if name in heuristic_specs and (
            specification != heuristic_specs[name]
            or not np.array_equal(score, heuristic_scores[name])
        ):
            raise MLPScoreError(f"{name}: frozen M4 heuristic changed")
        if name == BASELINE:
            capacity._baseline_actions(frame, targets)
        masks[name], scores[name], specifications[name] = (
            mask.copy(),
            score.copy(),
            specification,
        )
    return masks, scores, specifications


def _paired_comparisons(replicates, policy_results):
    wide = replicates.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    comparisons = {}
    for variant in VARIANTS:
        name = "mlp__" + variant
        comparisons[name] = {}
        for comparator in (
            *("capacity__" + family + "__" + variant for family in capacity.FAMILIES),
            "geo_minus_raw_margin",
        ):
            point = (
                policy_results[name]["mean_seed_delta_accuracy"]
                - policy_results[comparator]["mean_seed_delta_accuracy"]
            )
            values = (wide[name] - wide[comparator]).to_numpy(dtype=np.float64)
            comparisons[name][comparator] = {
                "mlp_policy": name,
                "comparator_policy": comparator,
                **_interval(point, values),
            }
    return {
        **SCOPE,
        "metric": "mean_seed_delta_accuracy",
        "difference_sign": "MLP_minus_frozen_comparator",
        "units": "accuracy_fraction",
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "replicates": 2000,
        "random_seed": 20260826,
        "shared_complete_plot_draws": True,
        "selection_repeated_in_bootstrap": False,
        "comparisons": comparisons,
    }


def _evaluate(
    predictions, experts, targets, configurations, baseline_actions, heuristics
):
    records, groups = _configurations(configurations)
    experts, targets, probabilities = capacity._validated_targets_experts(
        experts, targets
    )
    frames = capacity._validated_predictions(predictions, targets, records)
    action_masks, scores, specifications = _frozen_actions(
        baseline_actions, targets, experts, heuristics
    )
    candidate_metrics = {
        record["config_id"]: {
            **record,
            **capacity._router_metrics(targets, frames[record["config_id"]]),
        }
        for record in records
    }
    selections, curves = {}, []
    for group_id, options in groups.items():
        losses = [
            {
                "config_id": record["config_id"],
                "declared_order": record["declared_order"],
                "mean_seed_log_loss": candidate_metrics[record["config_id"]][
                    "all_rows"
                ]["seed_summary"]["log_loss"]["mean"],
            }
            for record in options
        ]
        minimum = min(record["mean_seed_log_loss"] for record in losses)
        tied = [
            record
            for record in losses
            if record["mean_seed_log_loss"] - minimum <= TIE_ATOL
        ]
        winner = min(tied, key=lambda record: record["declared_order"])
        selected_frame = frames[winner["config_id"]]
        threshold = capacity.select_common_threshold(
            targets.assign(score=selected_frame.score.to_numpy())
        )
        selections[group_id] = {
            "group_id": group_id,
            "family": "mlp",
            "variant": options[0]["variant"],
            "selected_config_id": winner["config_id"],
            "selected_policy_id": group_id,
            "selected_mean_seed_log_loss": winner["mean_seed_log_loss"],
            "minimum_mean_seed_log_loss": minimum,
            "candidate_losses": losses,
            "tied_config_ids": [record["config_id"] for record in tied],
            "tie_absolute_tolerance": TIE_ATOL,
            "tie_break": "first_in_predeclared_simplicity_order",
            "threshold": threshold,
        }
        specifications[group_id] = threshold["threshold_specification"]
        scores[group_id] = selected_frame.score.to_numpy()
        action_masks[group_id] = capacity.apply_heuristic_threshold(
            scores[group_id],
            targets.raw_pred.to_numpy(),
            targets.geo_pred.to_numpy(),
            specifications[group_id],
        )
        for item in threshold["candidate_curve"]:
            curve = {"group_id": group_id, "config_id": winner["config_id"], **item}
            curve["threshold_specification"] = json.dumps(
                curve["threshold_specification"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            curves.append(curve)
    classifier = {
        **SCOPE,
        "class_count": 18,
        "top3_basis": "selected_expert_calibrated_probability",
        "experts": {},
        "policies": {},
    }
    expert_metrics, policy_results, action_frames = {}, {}, []
    y = targets.label_id_dense.to_numpy(dtype=np.int64)
    for name in ("image", "geo", "raw"):
        classifier["experts"][name] = capacity._classify(
            targets,
            probabilities[name],
            targets[f"{name}_pred"].to_numpy(dtype=np.int64),
        )
        expert_metrics[name] = capacity._seed_records(
            targets,
            lambda mask: capacity.calibration_metrics(
                probabilities[name][mask], y[mask]
            ),
            ("log_loss", "brier_score", "top1_accuracy", "top3_accuracy", "ece"),
        )
    for name, acted in action_masks.items():
        final = np.where(
            acted, targets.geo_pred.to_numpy(), targets.raw_pred.to_numpy()
        )
        selected_probability = np.where(
            acted[:, None], probabilities["geo"], probabilities["raw"]
        )
        policy_results[name] = {
            **capacity.policy_metrics(targets, acted),
            "threshold_specification": specifications[name],
        }
        classifier["policies"][name] = capacity._classify(
            targets, selected_probability, final
        )
        frame = targets.copy()
        frame["policy_id"], frame["score"], frame["acted"], frame["final_pred"] = (
            name,
            scores[name],
            acted,
            final,
        )
        frame["threshold_specification"] = json.dumps(
            specifications[name], sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        frame["rescued"] = acted & targets.target_state.eq("rescue").to_numpy()
        frame["harmed"] = acted & targets.target_state.eq("harm").to_numpy()
        frame["raw_correct"], frame["final_correct"] = (
            targets.raw_pred.to_numpy() == y,
            final == y,
        )
        frame["net_correct"] = frame.rescued.to_numpy(
            dtype=np.int8
        ) - frame.harmed.to_numpy(dtype=np.int8)
        action_frames.append(frame)
    bootstrap, replicates = capacity.bootstrap_policy_comparison(
        targets, action_masks, replicates=2000, random_seed=20260826
    )
    bootstrap.update(
        {
            **SCOPE,
            "primary_policy_role": "original_frozen_M4_logistic_no_go_baseline",
            "selection_repeated_in_bootstrap": False,
            "pairwise_difference_sign": "original_frozen_baseline_minus_named_policy",
        }
    )
    paired = _paired_comparisons(replicates, policy_results)
    summary = {
        **SCOPE,
        "row_count": len(targets),
        "image_count": int(targets.row_uid.nunique()),
        "plot_count": int(targets.plot_idx.nunique()),
        "configuration_count": len(records),
        "group_count": len(groups),
        "policy_count": len(action_masks),
        "frozen_baseline_policy_count": len(FROZEN_POLICIES),
        "original_baseline_policy": BASELINE,
        "groups": {
            group_id: {
                "family": "mlp",
                "variant": selection["variant"],
                "selected_config_id": selection["selected_config_id"],
                "policy_id": group_id,
                "four_state": candidate_metrics[selection["selected_config_id"]][
                    "all_rows"
                ]["seed_summary"],
                "disagreement": candidate_metrics[selection["selected_config_id"]][
                    "disagreement"
                ]["seed_summary"],
                "classification": classifier["policies"][group_id]["seed_summary"],
                "policy": policy_results[group_id],
                "bootstrap_delta_accuracy": bootstrap["policies"][group_id]["metrics"][
                    "mean_seed_delta_accuracy"
                ],
                "paired_comparisons": paired["comparisons"][group_id],
            }
            for group_id, selection in selections.items()
        },
    }
    return {
        "selection.json": {**SCOPE, "groups": selections},
        "candidate_metrics.json": candidate_metrics,
        "expert_metrics.json": expert_metrics,
        "classification.json": classifier,
        "policy_metrics.json": policy_results,
        "threshold_curves.parquet": capacity._table(pd.DataFrame.from_records(curves)),
        "actions.parquet": capacity._table(
            pd.concat(action_frames, ignore_index=True).sort_values(
                ["policy_id", *KEY], kind="stable"
            )
        ),
        "bootstrap_summary.json": bootstrap,
        "bootstrap_replicates.parquet": capacity._table(replicates),
        "subgroup_metrics.parquet": capacity._table(
            capacity.subgroup_policy_report(targets, action_masks)
        ),
        "paired_comparisons.json": paired,
        "summary.json": summary,
    }


def evaluate_mlp(
    predictions: pd.DataFrame,
    experts: pd.DataFrame,
    targets: pd.DataFrame,
    configurations: Sequence[Mapping],
    capacity_baseline_actions: pd.DataFrame,
    heuristic_specifications: Mapping,
) -> dict[str, Any]:
    """Select four declared architectures per variant and compare 18 policies.

    Predictions must be sealed before the external caller reads validation
    targets. Configuration declaration order breaks mean-four-seed NLL ties
    within 1e-12. Frozen capacity actions retain their original thresholds and
    are checked against score replay and separate targets before comparison.
    """
    try:
        return _evaluate(
            predictions,
            experts,
            targets,
            configurations,
            capacity_baseline_actions,
            heuristic_specifications,
        )
    except ValueError as exc:
        if isinstance(exc, MLPScoreError):
            raise
        raise MLPScoreError(str(exc)) from exc


__all__ = ["MLPScoreError", "SCORE_FILES", "evaluate_mlp"]

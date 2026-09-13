"""Pure development scoring for the predeclared 3 x 3 router capacity round.

Only caller-supplied sealed predictions, expert outputs, and separate development
targets are consumed. This module never discovers files, fits a learner, or opens
test data. Configuration and threshold selection reuse development validation;
all resulting estimates and plot-bootstrap intervals are selection-conditional.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from multimodal.geo_helpfulness_router_heuristics import (
    NEVER_THRESHOLD,
    apply_heuristic_threshold,
    validate_threshold_specification,
)
from multimodal.geo_helpfulness_router_metrics import (
    binary_average_precision,
    calibration_metrics,
    classification_metrics,
    multiclass_metrics,
    policy_metrics,
    seed_summary,
)
from multimodal.geo_helpfulness_router_selection import (
    HEURISTICS,
    SEEDS,
    TARGET_ORDER,
    VARIANTS,
    select_common_threshold,
)
from multimodal.geo_helpfulness_router_uncertainty import (
    bootstrap_policy_comparison,
    subgroup_policy_report,
)


FAMILIES = ("lr", "rf", "xgb")
Q_COLUMNS = tuple(f"q_{state}" for state in TARGET_ORDER)
KEY = ("row_uid", "training_seed")
IDENTITY = ("row_uid", "plot_idx", "training_seed")
TARGET_COLUMNS = (
    *IDENTITY,
    "label_id_dense",
    "canonical_l3_id",
    "label_name",
    "image_pred",
    "geo_pred",
    "raw_pred",
    "target_state",
)
CANONICAL_CLASSES = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19)
BASELINE = "primary_full727"
ZERO_THRESHOLD = {"kind": "finite", "comparison": "strict_gt", "value_hex": (0.0).hex()}
TIE_ATOL = 1.0e-12
INTERPRETATION = "exploratory_reused_development_validation_selection_conditional"
SCOPE = {
    "interpretation": INTERPRETATION,
    "training_seeds": list(SEEDS),
    "test_data_read_by_scorer": False,
    "fitting_performed_by_scorer": False,
    "original_M4_no_go_preserved": True,
    "independent_confirmation": False,
    "selection_disclosure": (
        "Configurations and thresholds are selected on reused development "
        "validation. Estimates and intervals condition on that selection, the "
        "fitted routers, and the frozen experts and preprocessing."
    ),
}


class CapacityScoreError(ValueError):
    """The caller's sealed development inputs violate the scoring contract."""


def _frame(frame: Any, required: Sequence[str], name: str, *, blind=False):
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or frame.columns.duplicated().any()
        or not set(required).issubset(frame.columns)
    ):
        raise CapacityScoreError(
            f"{name}: missing columns, duplicate columns, or empty table"
        )
    if frame[list(required)].isna().any().any():
        raise CapacityScoreError(f"{name}: missing required values")
    if blind and any(
        column == "label"
        or column.startswith("label_")
        or "target" in column.casefold()
        or (column.endswith("_correct") and column not in Q_COLUMNS)
        or column in {"canonical_l3_id", "rescued", "harmed", "net_correct"}
        for column in frame.columns
    ):
        raise CapacityScoreError(f"{name}: prediction inputs must be label-blind")


def _integer(frame: pd.DataFrame, column: str, values: Sequence[int]):
    if frame[column].dtype.kind not in "iu" or not frame[column].isin(values).all():
        raise CapacityScoreError(f"{column}: matching integer values required")


def _identities(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    for column in ("row_uid", "plot_idx"):
        if (
            not frame[column]
            .map(lambda x: isinstance(x, str) and bool(x.strip()))
            .all()
        ):
            raise CapacityScoreError(f"{name}: invalid {column}")
    _integer(frame, "training_seed", SEEDS)
    if frame.duplicated(list(KEY)).any():
        raise CapacityScoreError(f"{name}: duplicate image/seed identity")
    return frame.sort_values(list(KEY), kind="stable").reset_index(drop=True).copy()


def _aligned(frame: pd.DataFrame, targets: pd.DataFrame, name: str) -> pd.DataFrame:
    result = _identities(frame, name)
    if len(result) != len(targets) or not np.array_equal(
        result[list(IDENTITY)].to_numpy(dtype=object),
        targets[list(IDENTITY)].to_numpy(dtype=object),
    ):
        raise CapacityScoreError(f"{name}: exact image/seed/plot alignment mismatch")
    return result


def _probabilities(values: Any, count: int, classes: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(values)
    except (ValueError, TypeError) as exc:
        raise CapacityScoreError(f"{name}: rectangular probabilities required") from exc
    if (
        result.dtype != np.float64
        or result.shape != (count, classes)
        or not np.isfinite(result).all()
        or np.any(result < 0)
        or np.any(result > 1)
        or not np.allclose(result.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    ):
        raise CapacityScoreError(f"{name}: normalized float64 probabilities required")
    return result


def _expert_probabilities(experts: pd.DataFrame, column: str) -> np.ndarray:
    try:
        stacked = np.stack(experts[column].to_numpy())
    except (ValueError, TypeError) as exc:
        raise CapacityScoreError(
            f"{column}: rectangular probabilities required"
        ) from exc
    return _probabilities(stacked, len(experts), 18, column)


def _validated_targets_experts(experts, targets):
    _frame(targets, TARGET_COLUMNS, "targets")
    if "schema_version" in targets and set(targets.schema_version) != {
        "geo_helpfulness.m4_validation_targets.v1"
    }:
        raise CapacityScoreError(
            "only sealed M4 development-validation targets are permitted"
        )
    if "development_role" in targets and set(targets.development_role) != {
        "validation"
    }:
        raise CapacityScoreError(
            "capacity scoring is restricted to development validation"
        )
    targets = _identities(targets, "targets")
    if set(targets.training_seed) != set(SEEDS):
        raise CapacityScoreError("all four training seeds are required")
    for column in ("label_id_dense", "image_pred", "geo_pred", "raw_pred"):
        _integer(targets, column, range(18))
    _integer(targets, "canonical_l3_id", CANONICAL_CLASSES)
    expected_canonical = np.asarray(CANONICAL_CLASSES)[
        targets.label_id_dense.to_numpy()
    ]
    if not np.array_equal(expected_canonical, targets.canonical_l3_id):
        raise CapacityScoreError("dense/canonical fixed ontology mismatch")
    if not targets.label_name.map(
        lambda x: isinstance(x, str) and bool(x.strip())
    ).all():
        raise CapacityScoreError("nonempty habitat names are required")
    identity = targets.groupby("row_uid", sort=False).agg(
        seeds=("training_seed", "nunique"),
        plots=("plot_idx", "nunique"),
        labels=("label_id_dense", "nunique"),
    )
    if not identity.eq({"seeds": 4, "plots": 1, "labels": 1}).all().all():
        raise CapacityScoreError("images must retain four seeds and one plot/habitat")
    if (
        not targets.groupby("plot_idx").label_id_dense.nunique().eq(1).all()
        or not targets.groupby("label_id_dense").label_name.nunique().eq(1).all()
    ):
        raise CapacityScoreError("inconsistent plot habitat or habitat name")
    y = targets.label_id_dense.to_numpy()
    raw_ok, geo_ok = targets.raw_pred.to_numpy() == y, targets.geo_pred.to_numpy() == y
    state = np.select(
        [~raw_ok & geo_ok, raw_ok & ~geo_ok, raw_ok & geo_ok],
        TARGET_ORDER[:3],
        default=TARGET_ORDER[3],
    )
    if not np.array_equal(state, targets.target_state):
        raise CapacityScoreError("four-state targets contradict expert correctness")
    required = [*IDENTITY, "image_pred", "geo_pred", "raw_pred"]
    required += [f"{name}_prob_calibrated" for name in ("image", "geo", "raw")]
    required += ["image_geo_jsd", "geo_minus_raw_margin"]
    _frame(experts, required, "experts", blind=True)
    experts = _aligned(experts, targets, "experts")
    probabilities = {}
    for name in ("image", "geo", "raw"):
        _integer(experts, f"{name}_pred", range(18))
        p = _expert_probabilities(experts, f"{name}_prob_calibrated")
        if not np.array_equal(
            experts[f"{name}_pred"], targets[f"{name}_pred"]
        ) or not np.array_equal(p.argmax(axis=1), targets[f"{name}_pred"]):
            raise CapacityScoreError(f"{name}: expert prediction/probability mismatch")
        probabilities[name] = p
    for name in ("image_geo_jsd", "geo_minus_raw_margin"):
        value = experts[name].to_numpy()
        if value.dtype != np.float64 or not np.isfinite(value).all():
            raise CapacityScoreError(
                f"{name}: finite float64 heuristic scores required"
            )
    midpoint = 0.5 * (probabilities["image"] + probabilities["geo"])
    jsd = np.zeros(len(targets), dtype=np.float64)
    for probability in (probabilities["image"], probabilities["geo"]):
        terms = np.zeros_like(probability)
        positive = probability > 0
        terms[positive] = probability[positive] * np.log(
            probability[positive] / midpoint[positive]
        )
        jsd += 0.5 * terms.sum(axis=1, dtype=np.float64)
    margins = {}
    for name in ("geo", "raw"):
        ordered = np.sort(probabilities[name], axis=1)
        margins[name] = ordered[:, -1] - ordered[:, -2]
    for name, expected in (
        ("image_geo_jsd", jsd),
        ("geo_minus_raw_margin", margins["geo"] - margins["raw"]),
    ):
        if not np.allclose(experts[name], expected, atol=1e-12, rtol=1e-12):
            raise CapacityScoreError(
                f"{name}: calibrated probability reconstruction mismatch"
            )
    confidence_difference = probabilities["geo"].max(axis=1) - probabilities["raw"].max(
        axis=1
    )
    if "geo_minus_raw_confidence" in experts and not np.allclose(
        experts.geo_minus_raw_confidence, confidence_difference, atol=1e-12, rtol=1e-12
    ):
        raise CapacityScoreError("calibrated confidence difference mismatch")
    experts["geo_minus_raw_confidence"] = confidence_difference
    return experts, targets, probabilities


def _configurations(configurations: Sequence[Mapping]) -> tuple[list[dict], dict]:
    if not isinstance(configurations, (list, tuple)) or len(configurations) != 36:
        raise CapacityScoreError("exactly 36 predeclared configurations are required")
    records, groups = [], {}
    for position, original in enumerate(configurations):
        record = deepcopy(dict(original))
        if (
            not isinstance(record.get("config_id"), str)
            or not record["config_id"].strip()
            or record.get("family") not in FAMILIES
            or record.get("variant") not in VARIANTS
            or not isinstance(record.get("parameters"), dict)
        ):
            raise CapacityScoreError("invalid configuration identity or parameters")
        record["declared_order"] = position
        record["group_id"] = record["family"] + "__" + record["variant"]
        records.append(record)
        groups.setdefault(record["group_id"], []).append(record)
    expected = {family + "__" + variant for family in FAMILIES for variant in VARIANTS}
    if (
        len({record["config_id"] for record in records}) != 36
        or set(groups) != expected
        or any(len(group) != 4 for group in groups.values())
    ):
        raise CapacityScoreError(
            "nine family/variant groups with four unique configurations required"
        )
    try:
        json.dumps(records, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CapacityScoreError(
            "configuration records must be strict JSON-compatible"
        ) from exc
    return records, groups


def _validated_predictions(predictions, targets, records):
    _frame(
        predictions,
        [*IDENTITY, "config_id", "family", "variant", *Q_COLUMNS],
        "predictions",
        blind=True,
    )
    if set(predictions.config_id) != {record["config_id"] for record in records}:
        raise CapacityScoreError(
            "sealed prediction configurations differ from declaration"
        )
    frames = {}
    for record in records:
        name = record["config_id"]
        frame = _aligned(predictions.loc[predictions.config_id.eq(name)], targets, name)
        if set(frame.family) != {record["family"]} or set(frame.variant) != {
            record["variant"]
        }:
            raise CapacityScoreError(
                f"{name}: learner family or feature variant mismatch"
            )
        if any(frame[column].dtype != np.float64 for column in Q_COLUMNS):
            raise CapacityScoreError(f"{name}: probabilities must retain float64")
        q = _probabilities(frame[list(Q_COLUMNS)].to_numpy(), len(targets), 4, name)
        score = q[:, 0] - q[:, 1]
        if "score" in frame and not np.array_equal(frame.score.to_numpy(), score):
            raise CapacityScoreError(
                f"{name}: utility score is not q_rescue minus q_harm"
            )
        frame["score"] = score
        frames[name] = frame
    return frames


def _seed_records(targets, function, names, *, eligible=None):
    if eligible is None:
        eligible = np.ones(len(targets), dtype=bool)
    per_seed = {
        str(seed): function(eligible & targets.training_seed.eq(seed).to_numpy())
        for seed in SEEDS
    }
    return {
        "per_seed": per_seed,
        "pooled_seed_realizations": function(eligible),
        "seed_summary": {
            name: seed_summary([record[name] for record in per_seed.values()])
            for name in names
        },
    }


def _router_metrics(targets, frame):
    q = frame[list(Q_COLUMNS)].to_numpy()
    y = np.asarray(
        [TARGET_ORDER.index(state) for state in targets.target_state], dtype=np.int64
    )

    def one(mask):
        pred, labels = q[mask].argmax(axis=1), y[mask]
        result = classification_metrics(pred, labels, 4)
        probability_record = (
            multiclass_metrics(q[mask], labels)
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
                "rescue_pr_auc": binary_average_precision(labels == 0, q[mask, 0]),
                "harm_pr_auc": binary_average_precision(labels == 1, q[mask, 1]),
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
        for record, state in zip(result["per_class"], TARGET_ORDER):
            record["target_state"] = state
        return result

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
        "target_order": list(TARGET_ORDER),
        "all_rows": _seed_records(targets, one, names),
        "disagreement": _seed_records(
            targets,
            one,
            names,
            eligible=targets.raw_pred.ne(targets.geo_pred).to_numpy(),
        ),
    }


def _classify(targets, probabilities, prediction):
    def one(mask):
        labels = targets.loc[mask, "label_id_dense"].to_numpy(dtype=np.int64)
        result = classification_metrics(prediction[mask], labels, 18)
        result["top1_accuracy"] = result["accuracy"]
        result["top3_accuracy"] = multiclass_metrics(probabilities[mask], labels)[
            "top3_accuracy"
        ]
        for record in result["per_class"]:
            support = targets.loc[
                mask & targets.label_id_dense.eq(record["class_id"]).to_numpy()
            ]
            record["unique_images"] = int(support.row_uid.nunique())
            record["unique_plots"] = int(support.plot_idx.nunique())
            record["canonical_l3_id"] = CANONICAL_CLASSES[record["class_id"]]
            record["label_name"] = support.label_name.iloc[0] if len(support) else None
        return result

    return _seed_records(
        targets,
        one,
        ("top1_accuracy", "top3_accuracy", "macro_f1", "weighted_f1", "mcc"),
    )


def _specification(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping) and "threshold_specification" in value:
        value = value["threshold_specification"]
    validate_threshold_specification(value)
    return deepcopy(dict(value))


def _baseline_actions(value, targets):
    scores = np.zeros(len(targets), dtype=np.float64)
    if isinstance(value, pd.DataFrame):
        if "policy_id" in value:
            value = value.loc[value.policy_id.eq(BASELINE)]
        _frame(value, [*IDENTITY, "acted"], "frozen baseline")
        frame = _aligned(value, targets, "frozen baseline")
        mask = frame.acted.to_numpy()
        for name in ("raw_pred", "geo_pred"):
            if name in frame and not np.array_equal(frame[name], targets[name]):
                raise CapacityScoreError("frozen baseline expert prediction mismatch")
        if "final_pred" in frame and not np.array_equal(
            frame.final_pred, targets.raw_pred
        ):
            raise CapacityScoreError("frozen baseline no-go prediction changed")
        if "threshold_specification" in frame:
            if any(
                _specification(spec) != NEVER_THRESHOLD
                for spec in frame.threshold_specification.unique()
            ):
                raise CapacityScoreError(
                    "original frozen threshold must remain never-intervene"
                )
        if "score" in frame:
            scores = frame.score.to_numpy()
            if scores.dtype != np.float64 or not np.isfinite(scores).all():
                raise CapacityScoreError("frozen baseline score must be finite float64")
    else:
        mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (len(targets),) or mask.any():
        raise CapacityScoreError(
            "original M4 primary no-go must retain zero Boolean actions"
        )
    return mask.copy(), scores


def _heuristics(experts, specifications):
    if not isinstance(specifications, Mapping) or set(specifications) not in (
        set(HEURISTICS),
        {"image_geo_jsd", "geo_minus_raw_margin"},
    ):
        raise CapacityScoreError(
            "two frozen learned heuristics or all five specifications required"
        )
    specs = {
        "never_intervene": deepcopy(NEVER_THRESHOLD),
        "always_geo_on_geo_raw_disagreement": deepcopy(ZERO_THRESHOLD),
        HEURISTICS[2]: deepcopy(ZERO_THRESHOLD),
    }
    for name, value in specifications.items():
        specification = _specification(value)
        if name in specs and specification != specs[name]:
            raise CapacityScoreError("fixed M4 heuristic policy changed")
        specs[name] = specification
    scores = {
        "never_intervene": np.zeros(len(experts), dtype=np.float64),
        "always_geo_on_geo_raw_disagreement": np.ones(len(experts), dtype=np.float64),
        HEURISTICS[2]: experts.geo_minus_raw_confidence.to_numpy(),
        "image_geo_jsd": experts.image_geo_jsd.to_numpy(),
        "geo_minus_raw_margin": experts.geo_minus_raw_margin.to_numpy(),
    }
    return specs, scores


def _table(frame):
    return pa.Table.from_pandas(
        frame.reset_index(drop=True), preserve_index=False
    ).replace_schema_metadata(None)


def evaluate_capacity(
    predictions: pd.DataFrame,
    experts: pd.DataFrame,
    targets: pd.DataFrame,
    configurations: Sequence[Mapping],
    frozen_baseline_actions: Any,
    heuristic_specifications: Mapping,
) -> dict[str, Any]:
    """Select and report nine predeclared development-only capacity comparisons.

    Four candidate configurations per learner/feature group compete by mean
    four-seed NLL. Values within 1e-12 of the minimum tie, with input declaration
    order resolving the tie. Each winner then uses the unchanged M4 threshold
    selector. This never changes the original M4 frozen no-go baseline.
    """
    records, groups = _configurations(configurations)
    experts, targets, probabilities = _validated_targets_experts(experts, targets)
    frames = _validated_predictions(predictions, targets, records)
    baseline, baseline_score = _baseline_actions(frozen_baseline_actions, targets)
    heuristic_specs, heuristic_scores = _heuristics(experts, heuristic_specifications)
    candidate_metrics = {
        record["config_id"]: {
            **record,
            **_router_metrics(targets, frames[record["config_id"]]),
        }
        for record in records
    }
    selections, curves = {}, []
    action_masks, scores, specifications = (
        {BASELINE: baseline},
        {BASELINE: baseline_score},
        {BASELINE: deepcopy(NEVER_THRESHOLD)},
    )
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
        threshold_rows = targets.assign(score=selected_frame.score.to_numpy())
        threshold = select_common_threshold(threshold_rows)
        policy_id = "capacity__" + group_id
        selections[group_id] = {
            "group_id": group_id,
            "family": options[0]["family"],
            "variant": options[0]["variant"],
            "selected_config_id": winner["config_id"],
            "selected_policy_id": policy_id,
            "selected_mean_seed_log_loss": winner["mean_seed_log_loss"],
            "minimum_mean_seed_log_loss": minimum,
            "candidate_losses": losses,
            "tied_config_ids": [record["config_id"] for record in tied],
            "tie_absolute_tolerance": TIE_ATOL,
            "tie_break": "first_in_predeclared_simplicity_order",
            "threshold": threshold,
        }
        specifications[policy_id] = threshold["threshold_specification"]
        scores[policy_id] = selected_frame.score.to_numpy()
        action_masks[policy_id] = apply_heuristic_threshold(
            scores[policy_id],
            targets.raw_pred.to_numpy(),
            targets.geo_pred.to_numpy(),
            specifications[policy_id],
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
    for name in HEURISTICS:
        scores[name], specifications[name] = (
            heuristic_scores[name],
            heuristic_specs[name],
        )
        action_masks[name] = apply_heuristic_threshold(
            scores[name],
            targets.raw_pred.to_numpy(),
            targets.geo_pred.to_numpy(),
            specifications[name],
        )
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
        classifier["experts"][name] = _classify(
            targets,
            probabilities[name],
            targets[f"{name}_pred"].to_numpy(dtype=np.int64),
        )
        expert_metrics[name] = _seed_records(
            targets,
            lambda mask: calibration_metrics(probabilities[name][mask], y[mask]),
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
            **policy_metrics(targets, acted),
            "threshold_specification": specifications[name],
        }
        classifier["policies"][name] = _classify(targets, selected_probability, final)
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
    bootstrap, replicates = bootstrap_policy_comparison(
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
    summary = {
        **SCOPE,
        "row_count": len(targets),
        "image_count": int(targets.row_uid.nunique()),
        "plot_count": int(targets.plot_idx.nunique()),
        "configuration_count": len(records),
        "group_count": len(groups),
        "original_baseline_policy": BASELINE,
        "groups": {
            group_id: {
                "family": selection["family"],
                "variant": selection["variant"],
                "selected_config_id": selection["selected_config_id"],
                "policy_id": selection["selected_policy_id"],
                "four_state": candidate_metrics[selection["selected_config_id"]][
                    "all_rows"
                ]["seed_summary"],
                "disagreement": candidate_metrics[selection["selected_config_id"]][
                    "disagreement"
                ]["seed_summary"],
                "classification": classifier["policies"][
                    selection["selected_policy_id"]
                ]["seed_summary"],
                "policy": policy_results[selection["selected_policy_id"]],
                "bootstrap_delta_accuracy": bootstrap["policies"][
                    selection["selected_policy_id"]
                ]["metrics"]["mean_seed_delta_accuracy"],
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
        "threshold_curves.parquet": _table(pd.DataFrame.from_records(curves)),
        "actions.parquet": _table(
            pd.concat(action_frames, ignore_index=True).sort_values(
                ["policy_id", *KEY], kind="stable"
            )
        ),
        "bootstrap_summary.json": bootstrap,
        "bootstrap_replicates.parquet": _table(replicates),
        "subgroup_metrics.parquet": _table(
            subgroup_policy_report(targets, action_masks)
        ),
        "summary.json": summary,
    }


__all__ = ["CapacityScoreError", "evaluate_capacity"]

"""Pure numerical metrics for scoring the frozen M4 router policies.

Only caller-supplied arrays/tables are used. No artifact discovery, fitting,
scikit-learn, SciPy, or Torch imports are permitted in this scoring module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


PROBABILITY_ROW_SUM_ATOL = 1.0e-8
TRUE_PROBABILITY_FLOOR = float(np.finfo(np.float64).tiny)
SEEDS = (1, 2, 3, 4)
TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
POLICY_COLUMNS = (
    "row_uid",
    "plot_idx",
    "training_seed",
    "target_state",
    "label_id_dense",
    "raw_pred",
    "geo_pred",
)


class RouterMetricError(ValueError):
    """Caller-supplied data violate the scoring metric contract."""


def _class_ids(values: Any, *, size: int, class_count: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (size,) or result.dtype.kind not in "iu":
        raise RouterMetricError(f"{name} must be a matching integer class-ID vector")
    if np.any(result < 0) or np.any(result >= class_count):
        raise RouterMetricError(f"{name} outside the fixed class ontology")
    return result.astype(np.int64, copy=False)


def _probabilities_and_labels(
    probabilities: Any, labels: Any
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities)
    if (
        values.dtype != np.dtype("float64")
        or values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] < 2
    ):
        raise RouterMetricError(
            "probabilities must be nonempty float64 (n, classes>=2)"
        )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.allclose(
            values.sum(axis=1), 1.0, atol=PROBABILITY_ROW_SUM_ATOL, rtol=0.0
        )
    ):
        raise RouterMetricError(
            "probabilities must be finite, in [0,1], and row-normalized"
        )
    target = _class_ids(
        labels, size=len(values), class_count=values.shape[1], name="labels"
    )
    return values, target


def multiclass_metrics(probabilities: Any, labels: Any) -> dict[str, Any]:
    """Mean NLL, summed-class Brier, and stable dense-ID top-1/top-3 accuracy.

    Only true-class probabilities below float64's smallest positive normal value
    are floored for NLL. No other probabilities are clipped or renormalized.
    """
    values, target = _probabilities_and_labels(probabilities, labels)
    true_probability = values[np.arange(len(target)), target]
    floor_mask = true_probability < TRUE_PROBABILITY_FLOOR
    log_loss = float(
        -np.log(np.maximum(true_probability, TRUE_PROBABILITY_FLOOR)).mean()
    )
    residual = values.copy()
    residual[np.arange(len(target)), target] -= 1.0
    brier = float(np.mean(np.sum(residual * residual, axis=1), dtype=np.float64))
    top_k = min(3, values.shape[1])
    ranking = np.argsort(-values, axis=1, kind="stable")[:, :top_k]
    return {
        "row_count": len(values),
        "class_count": values.shape[1],
        "log_loss": log_loss,
        "brier_score": brier,
        "top1_accuracy": float(np.mean(values.argmax(axis=1) == target)),
        "top3_accuracy": float(np.mean(np.any(ranking == target[:, None], axis=1))),
        "top3_k": top_k,
        "true_probability_floor_count": int(floor_mask.sum()),
        "true_probability_floor_hex": TRUE_PROBABILITY_FLOOR.hex(),
    }


def classification_metrics(
    pred: Any, labels: Any, n_classes: int = 18
) -> dict[str, Any]:
    """Fixed-ontology confusion/F1/MCC; absent-class precision/recall/F1 are zero.

    Confusion rows are ground truth and columns predictions. Macro F1 includes
    every fixed class, even absent ones. Degenerate MCC is reported as null.
    """
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
        raise RouterMetricError("n_classes must be an integer >=2")
    target_raw = np.asarray(labels)
    if target_raw.ndim != 1:
        raise RouterMetricError("labels must be a one-dimensional integer vector")
    target = _class_ids(
        target_raw, size=len(target_raw), class_count=n_classes, name="labels"
    )
    prediction = _class_ids(
        pred, size=len(target), class_count=n_classes, name="predictions"
    )
    confusion = np.bincount(
        target * n_classes + prediction, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(n_classes),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(n_classes), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(n_classes),
        where=precision + recall != 0,
    )
    total = float(len(target))
    supports = support.astype(np.float64)
    predicted_counts = predicted_count.astype(np.float64)
    denominator_squared = (
        total * total - float(predicted_counts @ predicted_counts)
    ) * (total * total - float(supports @ supports))
    mcc = None
    if denominator_squared > 0:
        mcc = float(
            (float(true_positive.sum()) * total - float(supports @ predicted_counts))
            / math.sqrt(denominator_squared)
        )
    return {
        "row_count": len(target),
        "class_count": n_classes,
        "confusion_matrix": confusion.tolist(),
        "accuracy": float(true_positive.sum() / total) if total else None,
        "macro_f1": float(f1.mean()) if total else None,
        "weighted_f1": float(f1 @ supports / total) if total else None,
        "mcc": mcc,
        "per_class": [
            {
                "class_id": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "predicted_count": int(predicted_count[index]),
            }
            for index in range(n_classes)
        ],
    }


def binary_average_precision(y_bool: Any, scores: Any) -> float | None:
    """Noninterpolated AP, grouping exact score ties before recall increments."""
    target = np.asarray(y_bool)
    values = np.asarray(scores)
    if target.ndim != 1 or target.dtype != np.dtype("bool"):
        raise RouterMetricError("average-precision targets must be a Boolean vector")
    if (
        values.shape != target.shape
        or values.dtype != np.dtype("float64")
        or not np.isfinite(values).all()
    ):
        raise RouterMetricError(
            "average-precision scores must be matching finite float64"
        )
    positives = int(target.sum())
    if positives == 0:
        return None
    order = np.argsort(-values, kind="stable")
    sorted_scores, sorted_target = values[order], target[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(target) - 1
    ]
    true_positives = np.cumsum(sorted_target, dtype=np.int64)[group_ends]
    precision = true_positives / (group_ends + 1)
    recall_increment = np.diff(np.r_[0, true_positives]) / positives
    return float(np.sum(precision * recall_increment, dtype=np.float64))


def calibration_metrics(probabilities: Any, labels: Any) -> dict[str, Any]:
    """Top-label ECE with 15 equal-width [lower,upper) bins; last bin is closed."""
    values, target = _probabilities_and_labels(probabilities, labels)
    confidence = values.max(axis=1)
    correct = values.argmax(axis=1) == target
    edges = np.arange(16, dtype=np.float64) / 15.0
    assignments = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, 14)
    bins = []
    ece = 0.0
    for index in range(15):
        mask = assignments == index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        gap = abs(mean_confidence - accuracy) if count else None
        if count:
            ece += count / len(values) * gap
        bins.append(
            {
                "bin_index": index,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "upper_inclusive": index == 14,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return {
        **multiclass_metrics(values, target),
        "ece": float(ece),
        "ece_bin_count": 15,
        "bins": bins,
    }


def seed_summary(values: Any) -> dict[str, Any]:
    """Descriptive mean/population-SD/min/max over defined seed values only."""
    raw = list(values)
    defined = []
    for value in raw:
        if value is None:
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise RouterMetricError(
                "seed summary values must be finite numbers or null"
            )
        defined.append(float(value))
    data = np.asarray(defined, dtype=np.float64)
    return {
        "seed_count": len(raw),
        "defined_seed_count": len(data),
        "undefined_seed_count": len(raw) - len(data),
        "mean": float(data.mean()) if len(data) else None,
        "std": float(data.std(ddof=0)) if len(data) else None,
        "min": float(data.min()) if len(data) else None,
        "max": float(data.max()) if len(data) else None,
    }


def _policy_counts(rows: pd.DataFrame, actions: np.ndarray) -> dict[str, Any]:
    target = rows.target_state.to_numpy()
    acts = int(actions.sum())
    rescues = int(np.sum(actions & (target == "rescue")))
    harms = int(np.sum(actions & (target == "harm")))
    opportunities = int(np.sum(target == "rescue"))
    labels = rows.label_id_dense.to_numpy(dtype=np.int64)
    raw = rows.raw_pred.to_numpy(dtype=np.int64)
    geo = rows.geo_pred.to_numpy(dtype=np.int64)
    raw_correct = int(np.sum(raw == labels))
    geo_correct = int(np.sum(geo == labels))
    final_correct = int(np.sum(np.where(actions, geo, raw) == labels))
    net = rescues - harms
    if final_correct - raw_correct != net:
        raise RouterMetricError(
            "policy correctness change contradicts rescue/harm utility"
        )
    return {
        "row_count": len(rows),
        "acted_count": acts,
        "rescued_count": rescues,
        "harmed_count": harms,
        "rescued_minus_harmed": net,
        "rescue_opportunity_count": opportunities,
        "raw_correct_count": raw_correct,
        "geo_correct_count": geo_correct,
        "final_correct_count": final_correct,
        "raw_accuracy": raw_correct / len(rows) if len(rows) else None,
        "geo_accuracy": geo_correct / len(rows) if len(rows) else None,
        "final_accuracy": final_correct / len(rows) if len(rows) else None,
        "delta_accuracy": net / len(rows) if len(rows) else None,
        "coverage_fraction": acts / len(rows) if len(rows) else None,
        "override_precision": rescues / acts if acts else None,
        "rescue_recall": rescues / opportunities if opportunities else None,
        "harmful_override_rate": harms / acts if acts else None,
        "unique_acted_plots": int(rows.loc[actions, "plot_idx"].nunique()),
    }


def policy_metrics(rows: pd.DataFrame, actions: Any) -> dict[str, Any]:
    """Effective hard-override outcomes against raw fusion, per seed and pooled.

    Seed subsets are supported for subgroup reports; missing seeds have zero
    counts and null rates. Bootstrap resampling should use multiplicity weights
    externally rather than duplicating these immutable row identities.
    """
    if not isinstance(rows, pd.DataFrame) or not set(POLICY_COLUMNS).issubset(
        rows.columns
    ):
        raise RouterMetricError(
            "policy rows lack required identities, targets, or expert predictions"
        )
    if rows.columns.duplicated().any():
        raise RouterMetricError("duplicate policy input column names")
    frame = rows.loc[:, list(POLICY_COLUMNS)].copy()
    action_mask = np.asarray(actions)
    if action_mask.shape != (len(frame),) or action_mask.dtype != np.dtype("bool"):
        raise RouterMetricError("actions must be a matching Boolean vector")
    if frame.isna().any().any() or frame.duplicated(["row_uid", "training_seed"]).any():
        raise RouterMetricError("null values or duplicate seed-image identity")
    for name in ("row_uid", "plot_idx"):
        if not frame[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise RouterMetricError("policy identities must be nonempty strings")
    if frame.training_seed.dtype.kind not in "iu" or not set(
        frame.training_seed
    ).issubset(SEEDS):
        raise RouterMetricError("policy training seeds must be integer IDs 1..4")
    if (
        len(frame)
        and frame.groupby("row_uid", sort=False).plot_idx.nunique().max() != 1
    ):
        raise RouterMetricError("plot identity differs across seed realizations")
    if (
        len(frame)
        and frame.groupby("row_uid", sort=False).label_id_dense.nunique().max() != 1
    ):
        raise RouterMetricError("ground-truth label differs across seed realizations")
    label = _class_ids(
        frame.label_id_dense.to_numpy(), size=len(frame), class_count=18, name="labels"
    )
    raw = _class_ids(
        frame.raw_pred.to_numpy(),
        size=len(frame),
        class_count=18,
        name="raw predictions",
    )
    geo = _class_ids(
        frame.geo_pred.to_numpy(),
        size=len(frame),
        class_count=18,
        name="geo predictions",
    )
    expected_target = np.full(len(frame), "both_wrong", dtype=object)
    expected_target[(raw != label) & (geo == label)] = "rescue"
    expected_target[(raw == label) & (geo != label)] = "harm"
    expected_target[(raw == label) & (geo == label)] = "both_correct"
    if not np.array_equal(frame.target_state.to_numpy(), expected_target):
        raise RouterMetricError(
            "four-state targets contradict expert/label correctness"
        )
    if np.any(action_mask & (raw == geo)):
        raise RouterMetricError(
            "effective actions require raw/geo prediction disagreement"
        )
    per_seed = {}
    for seed in SEEDS:
        mask = frame.training_seed.eq(seed).to_numpy()
        per_seed[str(seed)] = _policy_counts(frame.loc[mask], action_mask[mask])
    summaries = {
        name: seed_summary([record[name] for record in per_seed.values()])
        for name in (
            "rescued_minus_harmed",
            "delta_accuracy",
            "coverage_fraction",
            "override_precision",
            "rescue_recall",
            "harmful_override_rate",
        )
    }
    return {
        "per_seed": per_seed,
        "pooled": _policy_counts(frame, action_mask),
        "mean_seed_net_utility": summaries["rescued_minus_harmed"]["mean"],
        "mean_seed_delta_accuracy": summaries["delta_accuracy"]["mean"],
        "seed_summary": summaries,
    }


__all__ = [
    "RouterMetricError",
    "TRUE_PROBABILITY_FLOOR",
    "PROBABILITY_ROW_SUM_ATOL",
    "multiclass_metrics",
    "classification_metrics",
    "binary_average_precision",
    "calibration_metrics",
    "policy_metrics",
    "seed_summary",
]

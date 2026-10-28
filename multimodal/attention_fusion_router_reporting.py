"""Metrics and local figures for the R3 validation geometry-router study.

These helpers evaluate already-declared actions. They do not fit thresholds,
load checkpoints, or change either expert's probability vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .attention_fusion_reporting import draw_cm


NUM_CLASSES = 18
METRIC_NAMES = ("acc1", "acc3", "macro_f1", "weighted_f1", "mcc")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _class_labels(class_labels: Sequence[str]) -> list[str]:
    labels = list(class_labels)
    _require(len(labels) == NUM_CLASSES and all(isinstance(value, str) and value for value in labels)
             and len(set(labels)) == NUM_CLASSES,
             "Expected the declared 18 unique, nonempty class labels in fixed order")
    return labels


def _probabilities(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    _require(array.ndim == 2 and array.shape[1] == NUM_CLASSES and len(array) > 0
             and np.issubdtype(array.dtype, np.floating),
             "Expected a nonempty floating probability matrix with 18 class columns")
    _require(bool(np.isfinite(array).all() and (array >= 0).all()
                  and np.allclose(array.sum(axis=1), 1.0, rtol=0, atol=1e-10)),
             "Probability vectors must be finite, nonnegative and sum to one")
    return array


def _classes(value: np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    _require(array.shape == (count,) and np.issubdtype(array.dtype, np.integer)
             and bool(((array >= 0) & (array < NUM_CLASSES)).all()),
             f"Invalid {name}: expected one integer class ID in [0, 18) per query")
    return array.astype(np.int64, copy=False)


def _actions(value: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(value)
    _require(array.shape == (count,) and array.dtype == np.bool_,
             "Action mask must contain one boolean per query")
    return array


def select_expert_probabilities(fusion_probabilities: np.ndarray,
                                geo_probabilities: np.ndarray,
                                action_mask: np.ndarray) -> np.ndarray:
    """Copy the complete geo vector on actions, otherwise the fusion vector.

    No softmax, blending, calibration or top-k manipulation is performed.
    The supplied arrays are left unchanged.
    """
    fusion = _probabilities(fusion_probabilities)
    geo = _probabilities(geo_probabilities)
    _require(fusion.shape == geo.shape, "Expert probability matrices must be aligned")
    _require(fusion.dtype == geo.dtype,
             "Expert probability matrices must share a dtype to preserve their values")
    actions = _actions(action_mask, len(fusion))
    result = fusion.copy()
    result[actions] = geo[actions]
    return result


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray,
                        class_labels: Sequence[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate supplied probabilities using the parent's stable ranking rules.

    Confusion, weighted F1 and MCC use the same formulas as
    ``attention_fusion_training.classification_metrics``. Macro F1 adds an
    explicit fixed-18-class mean with zero-division F1 equal to zero. Metrics
    use fractions, not percentages, and absent-class recall is unavailable.
    """
    names = _class_labels(class_labels)
    probabilities = _probabilities(probabilities)
    count = len(probabilities)
    labels = _classes(labels, count, "labels")
    top3 = np.argsort(-probabilities, axis=1, kind="stable")[:, :3]
    prediction = top3[:, 0]
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    denominator = support + predicted
    f1 = np.divide(2 * correct, denominator, out=np.zeros(NUM_CLASSES, dtype=np.float64),
                   where=denominator > 0)
    normalized = np.divide(confusion, support[:, None],
                           out=np.zeros_like(confusion, dtype=np.float64),
                           where=support[:, None] > 0)
    mcc_denominator = float((count * count - np.square(predicted.astype(float)).sum())
                            * (count * count - np.square(support.astype(float)).sum()))
    mcc = (float(correct.sum() * count - np.dot(support, predicted)) / np.sqrt(mcc_denominator)
           if mcc_denominator > 0 else 0.0)
    metrics = {
        "rows": count,
        "acc1": float(np.mean(prediction == labels)),
        "acc3": float(np.mean((top3 == labels[:, None]).any(axis=1))),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / count),
        "mcc": float(mcc),
        "confusion_matrix": confusion.tolist(),
        "confusion_row_normalized": [row.tolist() if size else [None] * NUM_CLASSES
                                      for row, size in zip(normalized, support)],
        "class_support": support.tolist(),
        "class_recall": [float(correct[index] / size) if size else None
                         for index, size in enumerate(support)],
        "class_f1": f1.tolist(),
        "class_labels": names,
        "metric_conventions": {
            "rates": "fractions",
            "top_k_ties": "stable_class_index_ascending",
            "macro_f1": "mean_over_fixed_18_classes_zero_division_0",
            "zero_support_class_accuracy": "null",
            "zero_support_normalized_confusion": "null_row_in_metrics_zero_row_in_plot",
            "mcc_zero_denominator": 0.0,
            "acc3_input": "entire_supplied_probability_vector",
        },
    }
    return metrics, {"probabilities": probabilities.copy(), "prediction": prediction.copy(),
                     "pred_top3": top3.copy()}


def habitat_changes(labels: np.ndarray, fusion_prediction: np.ndarray,
                    geo_prediction: np.ndarray, action_mask: np.ndarray,
                    plot_ids: np.ndarray, class_labels: Sequence[str]) -> list[dict[str, Any]]:
    """Summarize changes by true habitat for evaluation, never for routing.

    Plot counts are distinct within each event; a plot can appear in several
    event categories and their counts must not be summed as disjoint sets.
    """
    names = _class_labels(class_labels)
    raw_labels = np.asarray(labels)
    _require(raw_labels.ndim == 1 and len(raw_labels) > 0, "Labels must be nonempty and one-dimensional")
    count = len(raw_labels)
    labels = _classes(raw_labels, count, "labels")
    fusion = _classes(fusion_prediction, count, "fusion predictions")
    geo = _classes(geo_prediction, count, "geo predictions")
    actions = _actions(action_mask, count)
    plots = np.asarray(plot_ids)
    _require(plots.shape == (count,) and plots.dtype.kind in "US"
             and bool((plots != "").all()), "Expected one nonempty plot identity per query")
    _require(not bool((actions & (fusion == geo)).any()),
             "Agreements cannot be counted as routing interventions")
    for plot in np.unique(plots):
        _require(len(np.unique(labels[plots == plot])) == 1,
                 "All queries from a plot must have the same true class")
    correct_fusion, correct_geo = fusion == labels, geo == labels
    routed = np.where(actions, geo, fusion)
    rescue_pool = ~correct_fusion & correct_geo
    harm_pool = correct_fusion & ~correct_geo
    both_wrong = ~correct_fusion & ~correct_geo
    events = {
        "actions": actions,
        "rescues": actions & rescue_pool,
        "harms": actions & harm_pool,
        "both_wrong_actions": actions & both_wrong,
        "potential_rescues": rescue_pool,
        "potential_harms": harm_pool,
        "missed_potential_rescues": ~actions & rescue_pool,
        "both_wrong_disagreements": both_wrong & (fusion != geo),
        "both_wrong_agreements": both_wrong & (fusion == geo),
    }
    result = []
    for class_id, class_label in enumerate(names):
        member = labels == class_id
        support = int(member.sum())
        row = {"class_id": class_id, "class_label": class_label, "support": support,
               "support_plots": int(len(np.unique(plots[member])))}
        for name, mask in events.items():
            row[name] = int((member & mask).sum())
            row[f"{name}_plots"] = int(len(np.unique(plots[member & mask])))
        row["net_rescues"] = row["rescues"] - row["harms"]
        for model, prediction in (("fusion", fusion), ("geo", geo), ("routed", routed)):
            correct = int((member & (prediction == labels)).sum())
            row[f"{model}_correct"] = correct
            row[f"{model}_accuracy"] = correct / support if support else None
        row["delta_acc1"] = row["net_rescues"] / support if support else None
        result.append(row)
    return result


def save_confusion_figures(metrics: dict[str, Any], class_labels: Sequence[str],
                           output_dir: str | Path, *, title: str | None = None,
                           save_pdf: bool = True) -> list[str]:
    """Save canonical raw and row-normalized CS confusion plots locally.

    The canonical helper closes each figure even if the output sink raises.
    Zero-support rows display zero, while numeric metrics retain null rows.
    """
    names = _class_labels(class_labels)
    _require(metrics.get("class_labels") == names, "Figure class order differs from metric class order")
    confusion = np.asarray(metrics["confusion_matrix"])
    _require(confusion.shape == (NUM_CLASSES, NUM_CLASSES), "Confusion matrix must have fixed 18-class axes")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = []

    def sink(figure: Any, values: np.ndarray, normalized: bool) -> None:
        if title:
            figure.suptitle(title, fontsize=12)
            figure.tight_layout(rect=(0, 0, 1, 0.96))
        stem = "confusion_normalized" if normalized else "confusion_raw"
        for extension in (("png", "pdf") if save_pdf else ("png",)):
            path = output / f"{stem}.{extension}"
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            figure.savefig(path, dpi=130, bbox_inches="tight", facecolor="white", metadata=metadata)
            paths.append(str(path))

    draw_cm(confusion, names, sink=sink)
    return paths

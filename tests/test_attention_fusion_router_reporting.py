"""The R3 report evaluates complete expert vectors and keeps absent classes."""

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from multimodal import attention_fusion_router_reporting as reporting
from multimodal.attention_fusion_training import classification_metrics


CLASS_LABELS = [f"Habitat {index}" for index in range(18)]


def probabilities(prediction):
    array = np.full((len(prediction), 18), 0.1 / 17, dtype=np.float64)
    array[np.arange(len(prediction)), prediction] = 0.9
    return array


def test_classification_metrics_match_parent_and_macro_has_fixed_18_axes():
    logits = np.zeros((6, 18), dtype=np.float32)
    logits[np.arange(6), [0, 1, 1, 1, 2, 0]] = 3
    labels = np.array([0, 0, 1, 1, 2, 2])
    parent, arrays = classification_metrics(labels, logits)
    metrics, result = reporting.probability_metrics(labels, arrays["probabilities"], CLASS_LABELS)
    for key in ("acc1", "acc3", "weighted_f1", "mcc", "confusion_matrix",
                "class_recall", "class_support", "confusion_row_normalized"):
        assert metrics[key] == parent[key]
    assert np.array_equal(result["pred_top3"], arrays["pred_top3"])
    assert np.array_equal(result["probabilities"], arrays["probabilities"])
    assert metrics["macro_f1"] == pytest.approx(sum(metrics["class_f1"]) / 18)
    assert metrics["class_recall"][3:] == [None] * 15
    assert metrics["confusion_row_normalized"][17] == [None] * 18
    assert metrics["class_f1"][17] == 0
    assert metrics["metric_conventions"]["macro_f1"] == "mean_over_fixed_18_classes_zero_division_0"
    json.dumps(metrics, allow_nan=False)


def test_stable_top3_ties_use_class_order_and_mcc_degenerate_case_is_zero():
    tied = np.full((2, 18), 1 / 18, dtype=np.float64)
    metrics, arrays = reporting.probability_metrics(np.array([0, 3]), tied, CLASS_LABELS)
    np.testing.assert_array_equal(arrays["pred_top3"], [[0, 1, 2], [0, 1, 2]])
    assert metrics["acc1"] == 0.5
    assert metrics["acc3"] == 0.5
    assert metrics["mcc"] == 0


def test_routing_selects_whole_vector_and_top3_without_mutation():
    fusion = np.zeros((2, 18), dtype=np.float64)
    geo = fusion.copy()
    fusion[:, [0, 1, 2]] = [0.6, 0.3, 0.1]
    geo[:, [3, 4, 5]] = [0.6, 0.3, 0.1]
    originals = fusion.copy(), geo.copy()
    routed = reporting.select_expert_probabilities(fusion, geo, np.array([True, False]))
    assert np.array_equal(routed[0], geo[0])
    assert np.array_equal(routed[1], fusion[1])
    metrics, arrays = reporting.probability_metrics(np.array([4, 4]), routed, CLASS_LABELS)
    assert metrics["acc1"] == 0
    assert metrics["acc3"] == 0.5
    np.testing.assert_array_equal(arrays["pred_top3"], [[3, 4, 5], [0, 1, 2]])
    routed[0, 0] = 42
    assert np.array_equal(fusion, originals[0])
    assert np.array_equal(geo, originals[1])


@pytest.mark.parametrize("problem", ["negative", "nan", "sum", "width", "empty", "integer"])
def test_invalid_probabilities_are_rejected(problem):
    values = probabilities([0, 1])
    if problem == "negative":
        values[0, 0] = -0.1
    elif problem == "nan":
        values[0, 0] = np.nan
    elif problem == "sum":
        values[0] *= 2
    elif problem == "width":
        values = values[:, :-1]
    elif problem == "empty":
        values = values[:0]
    else:
        values = values.astype(np.int64)
    with pytest.raises(ValueError, match="probability|Probability"):
        reporting.probability_metrics(np.array([0, 1]), values, CLASS_LABELS)


@pytest.mark.parametrize("labels", [np.array([0.0, 1.0]), np.array([0, 18]),
                                   np.array([0, -1]), np.array([True, False]), np.array([0])])
def test_invalid_labels_cannot_be_silently_cast(labels):
    with pytest.raises(ValueError, match="labels"):
        reporting.probability_metrics(labels, probabilities([0, 1]), CLASS_LABELS)


@pytest.mark.parametrize("classes", [CLASS_LABELS[:-1], ["x"] * 18, [""] + CLASS_LABELS[1:]])
def test_declared_class_order_requires_18_named_axes(classes):
    with pytest.raises(ValueError, match="class labels"):
        reporting.probability_metrics(np.array([0]), probabilities([0]), classes)


@pytest.mark.parametrize("actions", [np.array([1, 0]), np.array([True]), np.array([[True, False]])])
def test_action_mask_must_be_boolean_and_aligned(actions):
    with pytest.raises(ValueError, match="Action mask"):
        reporting.select_expert_probabilities(probabilities([0, 1]), probabilities([1, 0]), actions)


def test_probability_sources_require_aligned_shape_and_dtype():
    with pytest.raises(ValueError, match="aligned"):
        reporting.select_expert_probabilities(probabilities([0, 1]), probabilities([1]),
                                               np.array([True, False]))
    # Use exactly representable probability vectors so the dtype test is isolated.
    left = np.zeros((1, 18), dtype=np.float64)
    left[0, 0] = 1
    with pytest.raises(ValueError, match="dtype"):
        reporting.select_expert_probabilities(left, left.astype(np.float32), np.array([False]))


def test_habitat_changes_match_actual_rescues_harms_and_missed_opportunities():
    labels = np.array([0, 0, 1, 1, 2, 2, 2])
    fusion = np.array([1, 0, 2, 1, 0, 0, 2])
    geo = np.array([0, 1, 0, 1, 2, 0, 2])
    actions = np.array([True, True, True, False, False, False, False])
    rows = reporting.habitat_changes(labels, fusion, geo, actions,
                                     np.array(["a", "a", "b", "c", "d", "d", "e"]), CLASS_LABELS)
    assert len(rows) == 18
    assert rows[0]["support"] == 2 and rows[0]["support_plots"] == 1
    assert rows[0]["rescues"] == 1 and rows[0]["harms"] == 1
    assert rows[0]["actions"] == 2 and rows[0]["actions_plots"] == 1
    assert rows[0]["rescues_plots"] == 1 and rows[0]["harms_plots"] == 1
    assert rows[0]["delta_acc1"] == 0
    assert rows[1]["both_wrong_actions"] == 1
    assert rows[1]["both_wrong_disagreements"] == 1
    assert rows[2]["missed_potential_rescues"] == 1
    assert rows[2]["both_wrong_agreements"] == 1
    assert rows[2]["fusion_correct"] == 1 and rows[2]["geo_correct"] == 2
    assert rows[17]["support"] == 0
    assert rows[17]["fusion_accuracy"] is None and rows[17]["delta_acc1"] is None
    for row in rows:
        assert row["actions"] == row["rescues"] + row["harms"] + row["both_wrong_actions"]
        assert row["routed_correct"] - row["fusion_correct"] == row["net_rescues"]
    json.dumps(rows, allow_nan=False)


def test_habitat_changes_reject_agreement_actions_and_inconsistent_plot_labels():
    with pytest.raises(ValueError, match="Agreements"):
        reporting.habitat_changes(np.array([0]), np.array([0]), np.array([0]),
                                  np.array([True]), np.array(["a"]), CLASS_LABELS)
    with pytest.raises(ValueError, match="same true class"):
        reporting.habitat_changes(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                                  np.array([False, False]), np.array(["a", "a"]), CLASS_LABELS)


def test_confusion_figures_use_canonical_sink_and_close_figures(tmp_path, monkeypatch):
    metrics, _ = reporting.probability_metrics(np.array([0, 0, 1]), probabilities([0, 1, 1]), CLASS_LABELS)
    before = plt.get_fignums()
    emitted = []

    def save(figure, path, **kwargs):
        axis = figure.axes[0]
        emitted.append({"name": path.name, "title": axis.get_title(),
                        "x": axis.get_xlabel(), "y": axis.get_ylabel(),
                        "values": np.asarray(axis.collections[0].get_array()).reshape(18, 18),
                        "annotations": [item.get_text() for item in axis.texts],
                        "size": tuple(figure.get_size_inches()),
                        "suptitle": figure._suptitle.get_text(), "kwargs": kwargs})

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", save)
    paths = reporting.save_confusion_figures(metrics, CLASS_LABELS, tmp_path, title="Validation / final")
    assert [value["name"] for value in emitted] == ["confusion_raw.png", "confusion_raw.pdf",
                                                   "confusion_normalized.png", "confusion_normalized.pdf"]
    assert len(paths) == 4
    assert emitted[0]["title"] == "Confusion Matrix L3"
    assert emitted[2]["title"] == "Confusion Matrix L3 (Normalized)"
    assert emitted[0]["annotations"][:2] == ["1", "1"]
    assert emitted[2]["annotations"][:3] == ["0.50", "0.50", "0"]
    assert not emitted[2]["values"][17].any()
    assert all(value["x"] == "Predicted" and value["y"] == "True" for value in emitted)
    assert all(value["size"] == (15.0, 12.0) for value in emitted)
    assert all(value["suptitle"] == "Validation / final" for value in emitted)
    assert plt.get_fignums() == before


def test_confusion_sink_failure_closes_figure(tmp_path, monkeypatch):
    metrics, _ = reporting.probability_metrics(np.array([0]), probabilities([0]), CLASS_LABELS)
    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise OSError("sink failed")

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fail)
    with pytest.raises(OSError, match="sink failed"):
        reporting.save_confusion_figures(metrics, CLASS_LABELS, tmp_path)
    assert plt.get_fignums() == before


def test_confusion_plot_order_must_match_numeric_record(tmp_path):
    metrics, _ = reporting.probability_metrics(np.array([0]), probabilities([0]), CLASS_LABELS)
    changed = copy.deepcopy(CLASS_LABELS)
    changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(ValueError, match="class order"):
        reporting.save_confusion_figures(metrics, changed, tmp_path)

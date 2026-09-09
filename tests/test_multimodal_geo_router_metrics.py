from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_router_metrics as metrics


def test_multiclass_metrics_match_hand_computation_and_do_not_mutate():
    probability = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)
    before = probability.copy()
    result = metrics.multiclass_metrics(probability, labels)
    assert result["log_loss"] == pytest.approx(-(math.log(0.7) + math.log(0.3)) / 2)
    assert result["brier_score"] == pytest.approx(0.5)
    assert result["top1_accuracy"] == 0.5
    assert result["top3_accuracy"] == 1
    assert result["true_probability_floor_count"] == 0
    np.testing.assert_array_equal(probability, before)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_true_probability_floor_handles_zero_and_subnormal_without_renormalization():
    tiny = np.finfo(np.float64).tiny
    probability = np.array([[1.0, 0.0], [1.0, tiny / 2]], dtype=np.float64)
    result = metrics.multiclass_metrics(probability, np.array([1, 1]))
    assert result["true_probability_floor_count"] == 2
    assert result["true_probability_floor_hex"] == float(tiny).hex()
    assert result["log_loss"] == pytest.approx(-math.log(tiny))
    assert result["brier_score"] == 2
    probability2 = np.array([[0.7 + 5e-9, 0.3]], dtype=np.float64)
    result2 = metrics.multiclass_metrics(probability2, np.array([1]))
    assert result2["log_loss"] == pytest.approx(-math.log(0.3), abs=0)
    assert result2["brier_score"] == float((0.7 + 5e-9) ** 2 + 0.7**2)


def test_exact_probability_ties_choose_ascending_dense_ids():
    probability = np.full((4, 4), 0.25, dtype=np.float64)
    result = metrics.multiclass_metrics(probability, np.arange(4))
    assert result["top1_accuracy"] == 0.25
    assert result["top3_accuracy"] == 0.75
    assert result["top3_k"] == 3
    binary = metrics.multiclass_metrics(np.full((2, 2), 0.5), np.array([0, 1]))
    assert binary["top3_k"] == 2 and binary["top3_accuracy"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        "float32",
        "nan",
        "negative",
        "high",
        "sum",
        "empty",
        "labels_float",
        "labels_bool",
        "labels_high",
        "labels_low",
        "shape",
    ],
)
def test_probability_metrics_reject_bad_inputs(damage):
    probability = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float64)
    labels = np.array([0, 1])
    if damage == "float32":
        probability = probability.astype(np.float32)
    elif damage == "nan":
        probability[0, 0] = np.nan
    elif damage == "negative":
        probability[0] = [-0.1, 1.1]
    elif damage == "high":
        probability[0, 0] = 1.1
    elif damage == "sum":
        probability[0, 0] += 0.01
    elif damage == "empty":
        probability, labels = probability[:0], labels[:0]
    elif damage == "labels_float":
        labels = labels.astype(float)
    elif damage == "labels_bool":
        labels = labels.astype(bool)
    elif damage == "labels_high":
        labels[0] = 2
    elif damage == "labels_low":
        labels[0] = -1
    else:
        labels = labels[:, None]
    for function in (metrics.multiclass_metrics, metrics.calibration_metrics):
        with pytest.raises(metrics.RouterMetricError):
            function(probability, labels)


def test_confusion_f1_and_mcc_match_hand_computation():
    labels = np.array([0, 0, 1, 1, 2, 2])
    prediction = np.array([0, 1, 1, 1, 0, 2])
    result = metrics.classification_metrics(prediction, labels, n_classes=3)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert result["accuracy"] == 4 / 6
    assert result["macro_f1"] == pytest.approx(59 / 90)
    assert result["weighted_f1"] == pytest.approx(59 / 90)
    assert result["mcc"] == pytest.approx(12 / math.sqrt(528))
    assert result["per_class"][1] == {
        "class_id": 1,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1": 0.8,
        "support": 2,
        "predicted_count": 3,
    }


def test_fixed_ontology_keeps_absent_classes_in_macro_and_degenerate_mcc_is_null():
    result = metrics.classification_metrics(np.array([0, 1]), np.array([0, 1]))
    assert len(result["confusion_matrix"]) == 18
    assert result["macro_f1"] == 2 / 18
    assert result["weighted_f1"] == result["mcc"] == 1
    assert result["per_class"][17] == {
        "class_id": 17,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
        "predicted_count": 0,
    }
    constant = metrics.classification_metrics(np.array([0, 0]), np.array([0, 0]))
    assert constant["accuracy"] == 1 and constant["mcc"] is None
    empty = metrics.classification_metrics(
        np.array([], dtype=int), np.array([], dtype=int)
    )
    assert (
        empty["accuracy"]
        is empty["macro_f1"]
        is empty["weighted_f1"]
        is empty["mcc"]
        is None
    )


@pytest.mark.parametrize(
    "prediction,labels,n_classes",
    [
        ([0.0], [0], 18),
        ([18], [0], 18),
        ([0], [-1], 18),
        ([0], [0], True),
        ([0], [0], 1),
        ([0, 1], [0], 18),
    ],
)
def test_classification_rejects_invalid_ids_and_shapes(prediction, labels, n_classes):
    with pytest.raises(metrics.RouterMetricError):
        metrics.classification_metrics(
            np.array(prediction), np.array(labels), n_classes
        )


def test_noninterpolated_average_precision_groups_ties():
    truth = np.array([True, False, True, False])
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    assert metrics.binary_average_precision(truth, scores) == pytest.approx(5 / 6)
    permutation = np.array([0, 2, 1, 3])
    assert metrics.binary_average_precision(
        truth[permutation], scores[permutation]
    ) == metrics.binary_average_precision(truth, scores)
    assert metrics.binary_average_precision(truth, np.full(4, 0.5)) == 0.5
    assert metrics.binary_average_precision(np.ones(4, dtype=bool), scores) == 1
    assert metrics.binary_average_precision(np.zeros(4, dtype=bool), scores) is None
    assert (
        metrics.binary_average_precision(
            np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
        )
        is None
    )


@pytest.mark.parametrize(
    "truth,scores",
    [
        (np.array([1, 0]), np.array([0.1, 0.2])),
        (np.array([True]), np.array([np.nan])),
        (np.array([True]), np.array([0.1], dtype=np.float32)),
        (np.array([True]), np.array([0.1, 0.2])),
    ],
)
def test_average_precision_rejects_invalid_inputs(truth, scores):
    with pytest.raises(metrics.RouterMetricError):
        metrics.binary_average_precision(truth, scores)


def test_ece_bins_include_boundaries_and_last_closed_endpoint():
    probability = np.array([[0.8, 0.2], [0.8, 0.2], [1.0, 0.0]], dtype=np.float64)
    result = metrics.calibration_metrics(probability, np.array([0, 1, 0]))
    assert result["ece"] == pytest.approx(0.2)
    assert result["ece_bin_count"] == len(result["bins"]) == 15
    assert result["bins"][12]["count"] == 2
    assert result["bins"][12]["mean_confidence"] == 0.8
    assert result["bins"][12]["accuracy"] == 0.5
    assert result["bins"][14]["count"] == 1
    assert result["bins"][14]["upper_inclusive"] is True
    assert result["bins"][0]["count"] == 0
    assert result["bins"][0]["accuracy"] is result["bins"][0]["mean_confidence"] is None
    assert sum(record["count"] for record in result["bins"]) == 3


def test_ece_exact_internal_edge_is_assigned_to_right_bin():
    edge = 1 / 15
    confidence = np.array([np.nextafter(edge, 0), edge, np.nextafter(edge, 1)])
    probability = np.repeat(((1 - confidence) / 17)[:, None], 18, axis=1)
    probability[:, 0] = confidence
    result = metrics.calibration_metrics(probability, np.zeros(3, dtype=int))
    assert result["bins"][0]["count"] == 1
    assert result["bins"][1]["count"] == 2


def policy_rows():
    rows = []
    for seed in (1, 2, 3, 4):
        for index, state in enumerate(metrics.TARGET_ORDER):
            rows.append(
                {
                    "row_uid": f"image_{index}",
                    "plot_idx": f"plot_{index}",
                    "training_seed": seed,
                    "target_state": state,
                    "label_id_dense": 0,
                    "raw_pred": [1, 0, 0, 1][index],
                    "geo_pred": [0, 1, 0, 2][index],
                }
            )
    return pd.DataFrame(rows)


def test_policy_metrics_denominators_and_pooled_plot_union():
    rows = policy_rows()
    actions = np.tile([True, True, False, True], 4)
    result = metrics.policy_metrics(rows, actions)
    pooled = result["pooled"]
    assert pooled["row_count"] == 16 and pooled["acted_count"] == 12
    assert pooled["rescued_count"] == pooled["harmed_count"] == 4
    assert pooled["rescued_minus_harmed"] == pooled["delta_accuracy"] == 0
    assert (
        pooled["raw_correct_count"]
        == pooled["geo_correct_count"]
        == pooled["final_correct_count"]
        == 8
    )
    assert pooled["coverage_fraction"] == 0.75
    assert pooled["override_precision"] == pooled["harmful_override_rate"] == 1 / 3
    assert pooled["rescue_recall"] == 1
    assert pooled["unique_acted_plots"] == 3
    assert result["mean_seed_net_utility"] == result["mean_seed_delta_accuracy"] == 0
    assert all(record["acted_count"] == 3 for record in result["per_seed"].values())
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_policy_rescue_only_gain_is_exactly_final_minus_raw_accuracy():
    rows = policy_rows()
    actions = rows.target_state.eq("rescue").to_numpy()
    result = metrics.policy_metrics(rows, actions)
    assert result["mean_seed_net_utility"] == 1
    assert result["mean_seed_delta_accuracy"] == 0.25
    assert result["pooled"]["final_accuracy"] - result["pooled"]["raw_accuracy"] == 0.25
    assert result["seed_summary"]["delta_accuracy"] == {
        "seed_count": 4,
        "defined_seed_count": 4,
        "undefined_seed_count": 0,
        "mean": 0.25,
        "std": 0.0,
        "min": 0.25,
        "max": 0.25,
    }


def test_policy_undefined_rates_and_missing_seed_subgroups():
    rows = policy_rows()
    never = metrics.policy_metrics(rows, np.zeros(len(rows), dtype=bool))
    assert (
        never["pooled"]["override_precision"]
        is never["pooled"]["harmful_override_rate"]
        is None
    )
    assert never["pooled"]["rescue_recall"] == 0
    subgroup = rows.loc[rows.training_seed.eq(1) & rows.target_state.ne("rescue")]
    result = metrics.policy_metrics(subgroup, np.zeros(len(subgroup), dtype=bool))
    assert result["pooled"]["rescue_recall"] is None
    assert result["per_seed"]["2"]["row_count"] == 0
    assert result["per_seed"]["2"]["delta_accuracy"] is None
    assert result["seed_summary"]["delta_accuracy"]["defined_seed_count"] == 1


def test_policy_permutation_invariance_and_input_immutability():
    rows = policy_rows()
    original = rows.copy(deep=True)
    actions = rows.target_state.eq("rescue").to_numpy()
    first = metrics.policy_metrics(rows, actions)
    order = np.random.default_rng(889).permutation(len(rows))
    assert metrics.policy_metrics(rows.iloc[order], actions[order]) == first
    pd.testing.assert_frame_equal(rows, original)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "null",
        "identity_type",
        "seed",
        "plot",
        "label",
        "target",
        "action_agreement",
        "integer_actions",
    ],
)
def test_policy_rejects_invalid_input(damage):
    rows = policy_rows()
    actions = np.zeros(len(rows), dtype=bool)
    if damage == "missing":
        rows = rows.drop(columns="raw_pred")
    elif damage == "duplicate":
        rows.loc[1, "row_uid"] = rows.loc[0, "row_uid"]
    elif damage == "null":
        rows.loc[0, "plot_idx"] = None
    elif damage == "identity_type":
        rows.loc[0, "row_uid"] = 42
    elif damage == "seed":
        rows.loc[0, "training_seed"] = 0
    elif damage == "plot":
        rows.loc[0, "plot_idx"] = "other_plot"
    elif damage == "label":
        rows.loc[0, "label_id_dense"] = 1
    elif damage == "target":
        rows.loc[0, "target_state"] = "harm"
    elif damage == "action_agreement":
        actions[2] = True
    else:
        actions = actions.astype(int)
    with pytest.raises(metrics.RouterMetricError):
        metrics.policy_metrics(rows, actions)


def test_seed_summary_ignores_only_nulls_and_uses_population_standard_deviation():
    assert metrics.seed_summary([1, None, 3, None]) == {
        "seed_count": 4,
        "defined_seed_count": 2,
        "undefined_seed_count": 2,
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert metrics.seed_summary([None] * 4)["mean"] is None
    with pytest.raises(metrics.RouterMetricError):
        metrics.seed_summary([np.nan])


def test_module_imports_only_stdlib_numpy_and_pandas():
    source = Path(metrics.__file__).read_text()
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    assert imports <= {"__future__", "math", "typing", "numpy", "pandas"}

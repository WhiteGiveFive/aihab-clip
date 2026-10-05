from __future__ import annotations

import json

import numpy as np
import pytest

from multimodal.geo_helpfulness_confidnet_report import (
    ConfidNetMetricError,
    paired_plot_bootstrap_gain,
    score_expert,
    score_policies,
)


def test_tcp_is_true_class_probability_not_top1_correctness():
    labels = np.array([0])
    geo = np.array([[0.40, 0.50, 0.10]])
    fusion = np.array([[0.35, 0.33, 0.32]])
    policies = score_policies(labels, np.log(geo), np.log(fusion), [0.40], [0.35])
    tcp_policy = policies["tcp_difference_positive"]
    assert tcp_policy["actions"] == tcp_policy["harms"] == 1
    assert tcp_policy["rescues"] == 0
    assert tcp_policy["net"] == -1
    assert tcp_policy["accuracy"] == 0
    assert policies["fusion"]["accuracy"] == 1
    expert = score_expert(labels, np.log(geo), [0.40])
    assert expert["top1_error_count"] == 1
    assert expert["tcp_quantiles"]["0.5"] == pytest.approx(0.40)
    assert expert["mcp_quantiles"]["0.5"] == pytest.approx(0.50)
    assert expert["tcp_regression"]["mse"] == pytest.approx(0)


def test_policy_ties_and_agreement_never_intervene():
    labels = np.array([0, 0, 1])
    geo = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    fusion = np.array([[0.0, 2.0], [2.0, 0.0], [1.0, 0.0]])
    result = score_policies(labels, geo, fusion, [0.6, 0.6, 0.9], [0.6, 0.6, 0.1])
    assert result["always_geo_disagreement"]["actions"] == 2
    assert result["mcp_difference_positive"]["actions"] == 0
    assert result["tcp_difference_positive"]["actions"] == 0


def test_policy_counts_partition_actions_and_explain_accuracy_change():
    # A rescue, harm, neutral-wrong change, and unchanged agreement.
    labels = np.array([0, 0, 0, 0])
    geo = np.array([[3, 1, 0], [1, 3, 0], [0, 1, 3], [3, 1, 0]])
    fusion = np.array([[1, 3, 0], [3, 1, 0], [0, 3, 1], [3, 1, 0]])
    result = score_policies(labels, geo, fusion, [0.9] * 4, [0.1] * 4)
    for policy in result.values():
        assert policy["actions"] == policy["rescues"] + policy["harms"] + policy["neutral_wrong"]
        assert policy["net"] == policy["rescues"] - policy["harms"]
        assert policy["accuracy"] - result["fusion"]["accuracy"] == policy["net"] / 4
        assert policy["coverage"] == policy["actions"] / 4
    assert result["tcp_difference_positive"]["rescues"] == 1
    assert result["tcp_difference_positive"]["harms"] == 1
    assert result["tcp_difference_positive"]["neutral_wrong"] == 1
    json.dumps(result, allow_nan=False)


def test_all_correct_and_all_wrong_have_explicit_undefined_metrics():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    correct = score_expert(np.array([0, 1]), logits, [0.8, 0.7])
    assert correct["tcp_incorrect_quantiles"] is None
    assert correct["top1_error_count"] == 0
    for detector in correct["failure_detection"].values():
        assert detector["average_precision"] is None
        assert detector["roc_auc"] is None
        assert detector["average_precision_undefined_reason"] == "no_failures"
    wrong = score_expert(np.array([1, 0]), logits, [0.8, 0.7])
    assert wrong["tcp_correct_quantiles"] is None
    for detector in wrong["failure_detection"].values():
        assert detector["average_precision"] == 1
        assert detector["roc_auc"] is None
        assert detector["roc_auc_undefined_reason"] == "no_correct_predictions"
    json.dumps(correct, allow_nan=False)
    json.dumps(wrong, allow_nan=False)


def test_failure_scores_have_expected_direction_and_tie_handling():
    logits = np.array([[2.0, 0.0]] * 4)
    labels = np.array([1, 0, 1, 0])
    result = score_expert(labels, logits, [0.1, 0.2, 0.2, 0.9])
    metrics = result["failure_detection"]
    assert metrics["predicted_tcp"]["average_precision"] == pytest.approx(5 / 6)
    assert metrics["predicted_tcp"]["roc_auc"] == pytest.approx(0.875)
    for name in ("mcp", "entropy"):
        assert metrics[name]["average_precision"] == 0.5
        assert metrics[name]["roc_auc"] == 0.5
    assert result["failure_detection_comparison"]["mcp"]["roc_auc_gain"] == pytest.approx(0.375)


def test_stable_softmax_preserves_logits_and_matches_native_probabilities():
    probabilities = np.array([[0.9995, 0.0005], [0.995, 0.005], [0.2, 0.8]])
    logits = np.log(probabilities) + 10000
    before = logits.copy()
    labels = np.array([0, 0, 1])
    result = score_expert(labels, logits, probabilities[:, 0])
    np.testing.assert_array_equal(logits, before)
    assert result["tcp_fraction_ge_0.99"] == 2 / 3
    assert result["tcp_fraction_ge_0.999"] == 1 / 3
    assert result["tcp_quantiles"]["0.5"] == pytest.approx(0.995)
    assert result["tcp_regression"]["mse"] == pytest.approx(0.12)
    assert result["tcp_regression"]["mae"] == pytest.approx(0.2)
    assert result["tcp_regression"]["rmse"] == pytest.approx(np.sqrt(0.12))
    extreme = score_expert(np.array([0]), np.array([[1e308, -1e308]]), [1.0])
    assert extreme["tcp_regression"]["mse"] == 0
    assert extreme["top1_accuracy"] == 1


def test_18_class_logits_and_top1_ties_are_supported():
    logits = np.zeros((2, 18))
    result = score_expert(np.array([0, 17]), logits, [1 / 18, 1 / 18])
    assert result["class_count"] == 18
    assert result["top1_correct_count"] == 1
    assert result["tcp_regression"]["mse"] == pytest.approx(0)


def test_bootstrap_keeps_all_seed_outcomes_together_within_plot():
    # Every plot has two rescues and two harms across the four seeds. Sampling
    # rows/seeds independently would give spurious uncertainty around zero.
    plots = np.repeat([10, 20, 30], 4)
    seeds = np.tile([1, 2, 3, 4], 3)
    labels = np.zeros(12, dtype=int)
    geo = np.tile([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]], (3, 1))
    fusion = geo[:, ::-1]
    kwargs = dict(n_resamples=100, random_seed=17)
    result = paired_plot_bootstrap_gain(labels, geo, fusion, [0.9] * 12, [0.1] * 12, plots, seeds, **kwargs)
    for name in ("always_geo_disagreement", "tcp_difference_positive"):
        policy = result[name]
        assert policy["accuracy_gain_vs_fusion"] == 0
        assert policy["ci95_lower"] == policy["ci95_upper"] == 0
        assert policy["bootstrap_standard_error"] == 0
        assert policy["plot_count"] == 3
        assert policy["seed_count"] == 4
        assert policy["selection_adjusted"] is False
    assert result == paired_plot_bootstrap_gain(labels, geo, fusion, [0.9] * 12, [0.1] * 12, plots, seeds, **kwargs)
    json.dumps(result, allow_nan=False)


def test_bootstrap_is_image_weighted_and_paired():
    labels = np.zeros(4, dtype=int)
    # One-image plot is rescued; three-image plot is harmed.
    geo = np.array([[2.0, 0.0]] + [[0.0, 2.0]] * 3)
    fusion = geo[:, ::-1]
    result = paired_plot_bootstrap_gain(labels, geo, fusion, [0.9] * 4, [0.1] * 4, [10, 20, 20, 20])
    assert result["tcp_difference_positive"]["accuracy_gain_vs_fusion"] == -0.5
    assert result["tcp_difference_positive"]["ci95_lower"] == -1
    assert result["tcp_difference_positive"]["ci95_upper"] == 1
    assert result["fusion"]["ci95_lower"] == result["fusion"]["ci95_upper"] == 0
    assert result["fusion"]["n_resamples"] == 2000
    assert result["fusion"]["random_seed"] == 20260826


def test_bootstrap_rejects_missing_seed_within_plot():
    with pytest.raises(ConfidNetMetricError, match="every plot"):
        paired_plot_bootstrap_gain(
            np.array([0, 0, 0]), np.zeros((3, 2)), np.zeros((3, 2)),
            [0.5] * 3, [0.5] * 3, [10, 10, 20], [1, 2, 1],
        )


@pytest.mark.parametrize(
    "labels,logits,tcp_hat",
    [
        ([0.0], [[1, 0]], [0.5]),
        ([2], [[1, 0]], [0.5]),
        ([0], [[np.nan, 0]], [0.5]),
        ([0], [[np.inf, 0]], [0.5]),
        ([0], [[1, 0]], [1.1]),
        ([0], [[1, 0]], [np.nan]),
        ([0], [[1, 0]], [[0.5]]),
        ([], np.zeros((0, 2)), []),
    ],
)
def test_invalid_scoring_inputs_raise(labels, logits, tcp_hat):
    with pytest.raises(ConfidNetMetricError):
        score_expert(labels, logits, tcp_hat)

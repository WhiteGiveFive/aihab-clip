"""Independent arithmetic checks for the OOF study's pure reporting helpers."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score

from multimodal.geo_helpfulness_confidnet_oof_report import (
    ConfidNetMetricError,
    POLICIES,
    format_report,
    paired_plot_bootstrap,
    policy_arrays,
    representation_summary,
    score_seed,
    score_study,
)


def fixture_payload(prefix="val"):
    # Outcomes: rescue, harm, both-correct, both-wrong disagreement,
    # both-wrong agreement, rescue. Each plot has two images.
    geo = np.array([
        [.8, .1, .1], [.2, .7, .1], [.1, .2, .7],
        [.1, .25, .65], [.1, .8, .1], [.1, .65, .25],
    ])
    raw = np.array([
        [.2, .6, .2], [.6, .2, .2], [.1, .1, .8],
        [.2, .7, .1], [.2, .6, .2], [.7, .2, .1],
    ])
    return {
        "labels": np.array([0, 0, 2, 0, 0, 1]),
        "plot_ids": np.array([f"{prefix}p{x}" for x in [1, 1, 2, 2, 3, 3]]),
        "row_uids": np.array([f"{prefix}r{x}" for x in range(6)]),
        "image_logits": np.log(raw),
        "geo_logits": np.log(geo),
        "raw_logits": np.log(raw),
        "image_tcp_hat": np.array([.3, .7, .9, .1, .2, .4], dtype=np.float32),
        "geo_tcp_hat": np.array([.8, .2, .9, .5, .9, .5], dtype=np.float32),
        "raw_tcp_hat": np.array([.3, .7, .8, .5, .1, .5], dtype=np.float32),
        "router_baseline_q": np.array([
            [.6, .2, .1, .1], [.5, .3, .1, .1], [.7, .1, .1, .1],
            [.2, .2, .1, .5], [.8, .1, .05, .05], [.1, .2, .1, .6],
        ]),
        "router_augmented_q": np.array([
            [.6, .2, .1, .1], [.1, .7, .1, .1], [.7, .1, .1, .1],
            [.3, .2, .1, .4], [.8, .1, .05, .05], [.6, .1, .1, .2],
        ]),
    }


def test_fixed_policies_require_disagreement_and_strict_positive():
    masks = policy_arrays(fixture_payload())
    assert tuple(masks) == POLICIES
    assert not masks["fusion"].any()
    assert masks["mcp_difference_positive"].tolist() == [True, True, False, False, False, False]
    assert masks["tcp_difference_positive"].tolist() == [True, False, False, False, False, False]
    assert masks["rf_baseline"].tolist() == [True, True, False, False, False, False]
    assert masks["rf_augmented"].tolist() == [True, False, False, True, False, True]


def test_native_float32_confidence_ties_are_not_broken():
    payload = fixture_payload()
    payload["geo_tcp_hat"][:] = np.float32(.75)
    payload["raw_tcp_hat"][:] = np.float32(.75)
    assert not policy_arrays(payload)["tcp_difference_positive"].any()
    payload["geo_tcp_hat"][0] = np.nextafter(np.float32(.75), np.float32(1))
    assert policy_arrays(payload)["tcp_difference_positive"].tolist() == [True, False, False, False, False, False]


def test_policy_arithmetic_and_weighted_f1_match_independent_metrics():
    payload = fixture_payload()
    result = score_seed(payload)
    assert result["four_state_counts"] == {"rescue": 2, "harm": 1, "both_correct": 1, "both_wrong": 2}
    expected = {
        "fusion": (0, 0, 0, 0, 0, 2),
        "rf_baseline": (2, 1, 1, 0, 0, 2),
        "rf_augmented": (3, 2, 0, 1, 2, 4),
    }
    masks = policy_arrays(payload)
    for name, counts in expected.items():
        metric = result["policies"][name]
        assert tuple(metric[key] for key in ("actions", "rescues", "harms", "neutral_wrong", "net", "correct_count")) == counts
        predicted = np.where(masks[name], payload["geo_logits"].argmax(axis=1), payload["raw_logits"].argmax(axis=1))
        assert metric["weighted_f1"] == pytest.approx(f1_score(payload["labels"], predicted, average="weighted"))
    assert result["policies"]["rf_augmented"]["plots_affected"] == 3
    assert result["policies"]["rf_baseline"]["plots_affected"] == 1


def test_router_log_loss_brier_and_both_rescue_rankings():
    payload = fixture_payload()
    result = score_seed(payload)
    states = np.array([0, 1, 2, 3, 3, 0])
    disagreement = payload["geo_logits"].argmax(axis=1) != payload["raw_logits"].argmax(axis=1)
    for name in ("baseline", "augmented"):
        q = payload[f"router_{name}_q"]
        metric = result["routers"][name]
        assert metric["nll"] == pytest.approx(log_loss(states, q, labels=[0, 1, 2, 3]))
        assert metric["brier"] == pytest.approx(np.sum((q - np.eye(4)[states]) ** 2, axis=1).mean())
        for key, scores in (("q_rescue", q[:, 0]), ("q_rescue_minus_q_harm", q[:, 0] - q[:, 1])):
            assert metric["rescue_ranking_on_disagreement"][key]["average_precision"] == pytest.approx(average_precision_score(states[disagreement] == 0, scores[disagreement]))


def test_confidence_detection_and_conditional_quantiles():
    payload = fixture_payload()
    metric = score_seed(payload)["confidence"]["geo"]
    failures = payload["geo_logits"].argmax(axis=1) != payload["labels"]
    scores = 1 - payload["geo_tcp_hat"].astype(float)
    assert metric["failure_detection"]["predicted_tcp"]["average_precision"] == pytest.approx(average_precision_score(failures, scores))
    assert metric["failure_detection"]["predicted_tcp"]["roc_auc"] == pytest.approx(roc_auc_score(failures, scores))
    assert metric["predicted_tcp_correct_quantiles"]["0.5"] == pytest.approx(np.median(payload["geo_tcp_hat"][~failures]))
    assert metric["predicted_tcp_incorrect_quantiles"]["0.5"] == pytest.approx(np.median(payload["geo_tcp_hat"][failures]))


def test_absent_disagreements_and_failures_are_explicitly_undefined():
    payload = fixture_payload()
    payload["geo_logits"] = payload["raw_logits"].copy()
    payload["labels"] = payload["raw_logits"].argmax(axis=1)
    result = score_seed(payload)
    assert result["routers"]["baseline"]["rescue_ranking_on_disagreement"]["q_rescue"]["undefined_reason"] == "no_disagreements"
    assert result["confidence"]["raw"]["failure_detection"]["predicted_tcp"]["average_precision"] is None
    assert result["policies"]["rf_augmented"]["disagreement_coverage"] is None
    assert result["confidence"]["raw"]["predicted_tcp_incorrect_quantiles"] is None


def test_exact_zero_router_probability_has_declared_finite_nll_floor():
    payload = fixture_payload()
    payload["router_baseline_q"][:] = [0, 0, 0, 1]
    metric = score_seed(payload)["routers"]["baseline"]
    assert metric["nll"] == pytest.approx(-4 / 6 * np.log(np.finfo(np.float64).tiny))
    assert metric["nll_probability_floor_count"] == 4
    assert metric["nll_probability_floor"] == np.finfo(np.float64).tiny
    assert np.isfinite(metric["nll"])


@pytest.mark.parametrize("field,value", [
    ("raw_tcp_hat", np.ones(5)),
    ("geo_tcp_hat", np.full(6, np.nan)),
    ("image_tcp_hat", np.full(6, 1.1)),
    ("router_baseline_q", np.full((6, 4), .3)),
    ("router_augmented_q", np.full((6, 3), 1 / 3)),
    ("labels", np.zeros(6, dtype=float)),
    ("labels", np.full(6, 3)),
    ("plot_ids", np.array(["a"] * 5)),
    ("row_uids", np.array(["a"] * 6)),
    ("geo_logits", np.zeros((6, 4))),
])
def test_malformed_inputs_rejected(field, value):
    payload = fixture_payload()
    payload[field] = value
    with pytest.raises(ConfidNetMetricError):
        score_seed(payload)


def test_bootstrap_whole_plot_matches_explicit_image_expansion():
    payload = fixture_payload()
    second = deepcopy(payload)
    second["router_augmented_q"] = second["router_baseline_q"].copy()
    result = paired_plot_bootstrap({1: payload, 2: second}, n_resamples=50, random_seed=31)
    rng = np.random.default_rng(31)
    unique_plots = np.unique(payload["plot_ids"])
    expected_accuracy, expected_f1 = [], []
    for _ in range(50):
        selected = rng.integers(0, 3, size=3)
        rows = np.concatenate([np.flatnonzero(payload["plot_ids"] == unique_plots[index]) for index in selected])
        acc, f1 = [], []
        for source in (payload, second):
            masks = policy_arrays(source)
            predictions = {
                name: np.where(mask, source["geo_logits"].argmax(axis=1), source["raw_logits"].argmax(axis=1))[rows]
                for name, mask in masks.items()
            }
            truth = source["labels"][rows]
            acc.append(np.mean(predictions["rf_augmented"] == truth) - np.mean(predictions["rf_baseline"] == truth))
            f1.append(f1_score(truth, predictions["rf_augmented"], average="weighted") - f1_score(truth, predictions["rf_baseline"], average="weighted"))
        expected_accuracy.append(np.mean(acc))
        expected_f1.append(np.mean(f1))
    contrast = result["contrasts"]["rf_augmented_minus_rf_baseline"]
    for key, draws in (("accuracy", expected_accuracy), ("weighted_f1", expected_f1)):
        assert contrast[key]["ci95_lower"] == pytest.approx(np.quantile(draws, .025))
        assert contrast[key]["ci95_upper"] == pytest.approx(np.quantile(draws, .975))
        assert contrast[key]["bootstrap_standard_error"] == pytest.approx(np.std(draws, ddof=1))
    assert result["shared_resamples_for_all_contrasts"] is True


def test_bootstrap_seed_duplication_does_not_shrink_intervals():
    source = fixture_payload()
    one = paired_plot_bootstrap({1: source}, n_resamples=30)
    two = paired_plot_bootstrap({1: source, 2: source}, n_resamples=30)
    assert one["contrasts"] == two["contrasts"]
    assert one["plot_count"] == two["plot_count"] == 3


def test_bootstrap_equal_seed_rows_can_have_different_row_order():
    source = fixture_payload()
    reordered = {key: value[::-1] for key, value in source.items()}
    result = paired_plot_bootstrap({1: source, 2: reordered}, n_resamples=30)
    reference = paired_plot_bootstrap({1: source, 2: source}, n_resamples=30)
    assert result == reference


@pytest.mark.parametrize("mutation", ["plot", "row", "label", "missing"])
def test_bootstrap_rejects_inconsistent_seed_identities(mutation):
    first = fixture_payload()
    second = deepcopy(first)
    if mutation == "plot":
        second["plot_ids"][0] = "bad"
    elif mutation == "row":
        second["row_uids"][0] = "bad"
    elif mutation == "label":
        second["labels"][0] = 1
    else:
        second = {key: value[:-1] for key, value in second.items()}
    with pytest.raises(ConfidNetMetricError):
        paired_plot_bootstrap({1: first, 2: second}, n_resamples=5)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_bootstrap_budget_rejected(count):
    with pytest.raises(ConfidNetMetricError):
        paired_plot_bootstrap({1: fixture_payload()}, n_resamples=count)


def test_representation_summaries_record_each_producer_without_fitting():
    hidden = np.array([[0., 2.], [1., 3.], [4., 0.]])
    result = representation_summary(hidden, ["fold1", "fold1", "full"], [.3, .5, .7])
    assert result["fold1"]["row_count"] == 2
    assert result["fold1"]["coordinate_mean"] == [.5, 2.5]
    assert result["fold1"]["coordinate_standard_deviation"] == [.5, .5]
    assert result["full"]["fraction_exactly_zero"] == .5
    assert result["fold1"]["predicted_tcp_quantiles"]["0.5"] == .4


def test_study_uses_training_target_mean_for_validation_constant_baseline():
    train = fixture_payload("train")
    validation = fixture_payload("val")
    train["labels"] = train["raw_logits"].argmax(axis=1)
    result = score_study({1: train}, {1: validation}, n_resamples=10)
    probability_train = np.exp(train["raw_logits"])
    probability_val = np.exp(validation["raw_logits"])
    fitting_mean = probability_train[np.arange(6), train["labels"]].mean()
    target_val = probability_val[np.arange(6), validation["labels"]]
    metric = result["splits"]["validation"]["per_seed"]["1"]["confidence"]["raw"]
    assert metric["training_mean_predictor_tcp_mse"] == pytest.approx(np.mean((fitting_mean - target_val) ** 2))
    assert metric["training_mean_predictor_tcp_mse"] != pytest.approx(metric["split_mean_tcp_mse"])
    assert result["confidence_generalization_gaps"]["per_seed"]["1"]["raw"]["failure_ap_validation_minus_train"] is None


def test_study_per_seed_mean_and_pooled_counts_and_markdown():
    train = fixture_payload("train")
    validation = fixture_payload("val")
    second = deepcopy(validation)
    second["router_augmented_q"] = second["router_baseline_q"].copy()
    report = score_study({1: train, 2: train}, {1: validation, 2: second}, n_resamples=10)
    split = report["splits"]["validation"]
    assert split["mean_seed"]["policies"]["rf_augmented"]["net"] == 1
    assert split["pooled"]["policies"]["rf_augmented"]["net"] == 2
    assert split["mean_seed"]["row_count"] == 6
    assert split["pooled"]["row_count"] == 12
    assert "numeric25" in report["study"]
    text = format_report(report)
    assert "Augmented minus matched numerical baseline" in text
    assert "| 2 | rf_augmented |" in text
    assert "native" in text.lower()


def test_study_rejects_plot_leakage_between_splits():
    payload = fixture_payload()
    with pytest.raises(ConfidNetMetricError, match="plots must be disjoint"):
        score_study({1: payload}, {1: payload}, n_resamples=5)


def test_study_rejects_seed_mismatch():
    with pytest.raises(ConfidNetMetricError, match="same seeds"):
        score_study({1: fixture_payload("train")}, {2: fixture_payload("val")}, n_resamples=5)

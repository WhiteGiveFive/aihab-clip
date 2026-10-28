import numpy as np
import pandas as pd
import pytest

from multimodal.attention_fusion_router_outcomes import (
    conditional_plot_bootstrap,
    confusion_transitions,
    missed_rescue_partition,
    plot_outcomes,
)
from multimodal.geometry_routing_policy import finite_policy


CLASSES = [f"habitat_{index}" for index in range(18)]
NEVER = {"kind": "never_intervene", "value_hex": None,
         "value_decimal": None, "comparison": "strict_gt"}


def _query_fixture():
    return pd.DataFrame({
        "row_uid": [f"r{i}" for i in range(6)],
        "plot_idx": ["a", "a", "a", "b", "c", "c"],
        "label_id_dense": [0, 0, 0, 1, 2, 2],
        "fusion_predictions": [1, 0, 1, 0, 0, 2],
        "geo_predictions": [0, 1, 2, 1, 2, 2],
        "action": [True, True, True, True, False, False],
    })


def _partition_fixture(policy):
    rows = pd.DataFrame({
        "row_uid": [f"r{i}" for i in range(7)],
        "plot_idx": ["a", "a", "b", "b", "c", "c", "d"],
        "label_id_dense": np.zeros(7, dtype=int),
        "fusion_predictions": np.ones(7, dtype=int),
        "geo_predictions": np.zeros(7, dtype=int),
        "geometry_supported": [False, True, True, True, True, True, True],
        "eligible": [False, True, True, True, True, True, True],
        "A_image": [np.nan, 0, 1, 0, 0.1, 0.2, -2],
        "A_geo": [np.nan, 1, 0, 0, 0.1, 0.2, 0],
    })
    rows["score"] = np.minimum(rows.A_image, rows.A_geo)
    rows["action"] = False if policy["kind"] == "never_intervene" else (
        rows.eligible & (rows.score > policy["value_decimal"]))
    return rows


def test_plot_outcomes_retains_unacted_plots_and_nonadditive_event_counts():
    plots, summary = plot_outcomes(_query_fixture())
    assert list(plots.plot_idx) == ["a", "b", "c"]
    assert list(plots.images) == [3, 1, 2]
    assert list(plots.net_rescues) == [0, 1, 0]
    assert list(plots.both_wrong_actions) == [1, 0, 0]
    assert list(plots.missed_rescues) == [0, 0, 1]
    assert summary["action_plots"] == 2
    assert summary["rescue_plots"] == 2 and summary["harm_plots"] == 1
    assert summary["event_pattern_plots"] == {
        "rescue_only": 1, "harm_only": 0, "mixed_rescue_and_harm": 1,
        "neither_rescue_nor_harm": 1,
    }
    assert summary["net_change_plots"] == {"positive": 1, "negative": 0, "zero": 2}
    assert summary["image_weighted_delta_acc1"] == pytest.approx(1 / 6)
    assert summary["equal_plot_weighted_delta_acc1"] == pytest.approx(1 / 3)
    assert summary["harm_concentration"]["top_5_plots"] == {
        "plots": ["a"], "actual_plot_count": 1, "harms": 1,
        "fraction_of_all_harms": 1.0,
    }


def test_harm_concentration_uses_total_harm_denominator_and_deterministic_ties():
    sizes = [1, 3, 2, 1, 2, 1]
    rows = pd.DataFrame({
        "row_uid": [f"r{i}" for i in range(sum(sizes))],
        "plot_idx": np.repeat(list("abcdef"), sizes),
        "label_id_dense": 0, "fusion_predictions": 0,
        "geo_predictions": 1, "action": True,
    })
    _, summary = plot_outcomes(rows)
    concentration = summary["harm_concentration"]
    assert concentration["top_1_plots"]["plots"] == ["b"]
    assert concentration["top_3_plots"]["plots"] == ["b", "c", "e"]
    assert concentration["top_3_plots"]["fraction_of_all_harms"] == 0.7
    assert concentration["top_5_plots"]["fraction_of_all_harms"] == 0.9


def test_zero_actions_have_null_harm_share_and_zero_plot_changes():
    rows = _query_fixture()
    rows["action"] = False
    plots, summary = plot_outcomes(rows)
    assert not plots.net_rescues.any()
    assert summary["harm_concentration"]["top_1_plots"]["fraction_of_all_harms"] is None
    assert summary["event_pattern_plots"]["neither_rescue_nor_harm"] == 3


def test_missed_rescues_are_disjoint_and_include_strict_zero_and_threshold_equalities():
    policy = finite_policy(0.1)
    result = missed_rescue_partition(_partition_fixture(policy), policy)
    assert result["potential_rescues"] == 7
    assert result["rescues"] == 1 and result["missed_rescues"] == 6
    assert {key: value["images"] for key, value in result["categories"].items()} == {
        "unsupported": 1, "image_fails_only_at_zero": 1, "geo_fails_only_at_zero": 1,
        "both_fail_at_zero": 2, "positive_both_below_or_equal_threshold": 1,
        "positive_both_never_policy_blocked": 0,
    }
    assert sum(value["plots"] for value in result["categories"].values()) > result["missed_rescue_plots"]


def test_never_policy_only_assigns_positive_both_misses_to_policy_block():
    result = missed_rescue_partition(_partition_fixture(NEVER), NEVER)
    assert result["rescues"] == 0 and result["missed_rescues"] == 7
    assert result["categories"]["positive_both_never_policy_blocked"]["images"] == 2
    assert result["categories"]["positive_both_below_or_equal_threshold"]["images"] == 0
    assert result["categories"]["both_fail_at_zero"]["images"] == 2


def test_fixed_zero_has_no_positive_both_threshold_misses():
    policy = finite_policy(0.0)
    result = missed_rescue_partition(_partition_fixture(policy), policy)
    assert result["rescues"] == 2
    assert result["categories"]["positive_both_below_or_equal_threshold"]["images"] == 0


@pytest.mark.parametrize("change, match", [
    ("action", "saved actions"), ("score", "minimum"),
    ("eligible", "eligible flags"), ("A_image", "finite"),
])
def test_missed_partition_rejects_inconsistent_saved_geometry_or_policy(change, match):
    policy = finite_policy(0.1)
    rows = _partition_fixture(policy)
    if change in ("action", "eligible"):
        rows.loc[5, change] = False
    elif change == "A_image":
        rows.loc[5, change] = np.nan
    else:
        rows.loc[5, change] = 0.3
    with pytest.raises(ValueError, match=match):
        missed_rescue_partition(rows, policy)


def test_confusion_transitions_separate_rescues_harms_and_both_wrong_actions():
    result = confusion_transitions(_query_fixture(), CLASSES)
    assert result.images.sum() == 6 and result.actions.sum() == 4
    assert result.rescues.sum() == 2 and result.harms.sum() == 1
    assert result.both_wrong_actions.sum() == 1 and result.net_rescues.sum() == 1
    wrong = result.loc[(result.fusion_predictions == 1) & (result.geo_predictions == 2)].iloc[0]
    assert wrong.true_class == "habitat_0" and wrong.fusion_class == "habitat_1"
    assert wrong.geo_class == "habitat_2" and wrong.both_wrong_actions == 1


def test_conditional_bootstrap_resamples_whole_plots_with_image_denominators():
    plots = pd.DataFrame({"plot_idx": ["small", "large"], "images": [1, 3], "net_rescues": [1, -1]})
    result = conditional_plot_bootstrap(plots, n_resamples=101, seed=71)
    # Independently reproduce draws in canonical plot order. Drawing the large
    # plot twice gives -2/6, the small plot twice +2/2, and a mixed draw 0/4.
    indices = np.random.default_rng(71).integers(0, 2, size=(101, 2))
    images, net = np.array([3, 1]), np.array([-1, 1])
    ratios = np.array([sum(net[row]) / sum(images[row]) for row in indices])
    assert set(ratios) == {-1 / 3, 0, 1}
    assert result["lower_95"] == np.quantile(ratios, 0.025)
    assert result["upper_95"] == np.quantile(ratios, 0.975)
    assert result["delta_acc1"] == 0 and result["interval_contains_zero"]
    assert not result["threshold_reselection"]
    assert not result["accounts_for_validation_selection_optimism"]


def test_bootstrap_is_plot_order_invariant_and_crosses_chunk_boundaries():
    plots, _ = plot_outcomes(_query_fixture())
    first = conditional_plot_bootstrap(plots, n_resamples=1201, seed=19)
    assert first == conditional_plot_bootstrap(plots.iloc[::-1], n_resamples=1201, seed=19)
    indices = np.random.default_rng(19).integers(0, 3, size=(1201, 3))
    expected = plots.net_rescues.to_numpy()[indices].sum(axis=1) / plots.images.to_numpy()[indices].sum(axis=1)
    np.testing.assert_allclose([first["lower_95"], first["upper_95"]], np.quantile(expected, [0.025, 0.975]))


def test_never_policy_bootstrap_zero_interval_contains_zero_inclusively():
    rows = _query_fixture()
    rows["action"] = False
    plots, _ = plot_outcomes(rows)
    result = conditional_plot_bootstrap(plots)
    assert result["resamples"] == 10000 and result["seed"] == 20261023
    assert result["lower_95"] == result["upper_95"] == result["delta_acc1"] == 0
    assert result["interval_contains_zero"]


def test_positive_single_plot_bootstrap_does_not_contain_zero():
    result = conditional_plot_bootstrap(pd.DataFrame({"plot_idx": ["p"], "images": [5], "net_rescues": [1]}))
    assert result["lower_95"] == result["upper_95"] == 0.2
    assert not result["interval_contains_zero"]


@pytest.mark.parametrize("mutation, match", [
    ("duplicate", "unique"), ("labels", "one true class"),
    ("stale_flag", "flags disagree"), ("agreement_action", "disagreements"),
])
def test_outcomes_reject_inconsistent_queries(mutation, match):
    rows = _query_fixture()
    if mutation == "duplicate":
        rows.loc[1, "row_uid"] = "r0"
    elif mutation == "labels":
        rows.loc[1, "label_id_dense"] = 1
    elif mutation == "stale_flag":
        rows["rescue"] = False
    else:
        rows.loc[5, "action"] = True
    with pytest.raises(ValueError, match=match):
        plot_outcomes(rows)


@pytest.mark.parametrize("images, net", [([0], [0]), ([1], [2]), ([1.0], [0]), ([1], [0.0])])
def test_bootstrap_rejects_invalid_counts(images, net):
    with pytest.raises(ValueError, match="integer and physically possible"):
        conditional_plot_bootstrap(pd.DataFrame({"plot_idx": ["p"], "images": images, "net_rescues": net}))

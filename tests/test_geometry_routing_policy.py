"""Independent fixtures for the fixed R3 threshold and outcome conventions."""

import json

import numpy as np
import pytest

from multimodal.geometry_routing_policy import (
    _selection_key,
    action_mask,
    enumerate_policies,
    evaluate_actions,
    finite_policy,
    select_policy,
    validate_policy,
)


def _case(rescues=20, harms=0, both_wrong=0, total=None):
    action_count = rescues + harms + both_wrong
    n = action_count if total is None else total
    assert n >= action_count and n > 0
    truth = np.zeros(n, dtype=np.int64)
    fusion = np.zeros(n, dtype=np.int64)
    geo = np.zeros(n, dtype=np.int64)
    fusion[:rescues] = 1
    geo[rescues:rescues + harms] = 1
    fusion[rescues + harms:action_count] = 1
    geo[rescues + harms:action_count] = 2
    eligible = fusion != geo
    scores = np.where(eligible, 1.0, 0.0)
    plots = np.array([f"plot_{i}" for i in range(n)])
    return scores, eligible, fusion, geo, truth, plots


def _zero_record(selection):
    return next(record for record in selection["candidate_records"]
                if record["policy"]["value_hex"] == (0.0).hex())


def test_enumerator_uses_only_distinct_positive_eligible_scores():
    scores = [np.nan, -3.0, 0.0, 1.0, 1.0, 3.0, 9.0, 100.0, np.inf]
    allowed = [False, True, True, True, True, True, True, False, False]
    policies = enumerate_policies(scores, allowed)
    assert [validate_policy(p) for p in policies] == [None, 0.0, 2.0, 6.0]
    assert policies[0] == {
        "kind": "never_intervene", "value_hex": None,
        "value_decimal": None, "comparison": "strict_gt",
    }
    assert policies[2]["value_hex"] == (2.0).hex()
    json.dumps(policies, allow_nan=False)


@pytest.mark.parametrize("scores", [[0.0, -1.0], [1.0, 1.0]])
def test_no_midpoint_for_fewer_than_two_distinct_positive_scores(scores):
    assert [validate_policy(p) for p in enumerate_policies(scores, [True, True])] == [None, 0.0]


def test_float64_midpoints_match_historical_rounding_without_sentinels():
    a = np.float64(1.0)
    b = np.nextafter(a, np.inf)
    c = np.nextafter(b, np.inf)
    policies = enumerate_policies([a, b, c], [True] * 3)
    expected = np.unique([0.0, a + (b - a) / np.float64(2.0),
                          b + (c - b) / np.float64(2.0)])
    assert [validate_policy(p) for p in policies[1:]] == list(expected)
    assert all(p["value_hex"] == float(p["value_decimal"]).hex() for p in policies[1:])
    assert np.isfinite([p["value_decimal"] for p in policies[1:]]).all()


def test_action_strictness_and_ineligible_rows():
    policy = finite_policy(1.0)
    mask = action_mask(policy, [0.0, 1.0, np.nextafter(1.0, np.inf), 4.0, np.nan],
                       [True, True, True, False, False])
    assert mask.tolist() == [False, False, True, False, False]
    never = enumerate_policies([1.0], [True])[0]
    assert not action_mask(never, [1.0, np.inf], [True, False]).any()


@pytest.mark.parametrize("value", [-1.0, -0.0, float("nan"), float("inf"), True, "0"])
def test_finite_policy_rejects_noncanonical_or_invalid_thresholds(value):
    with pytest.raises(ValueError):
        finite_policy(value)


@pytest.mark.parametrize("change", [
    {"value_hex": "0x0p+0"},
    {"value_decimal": 1.0},
    {"comparison": "gte"},
    {"kind": "never_intervene"},
    {"unknown_field": 1},
    {"value_hex": "invalid"},
])
def test_policy_roundtrip_rejects_changed_specification(change):
    policy = finite_policy(0.0)
    policy.update(change)
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_outcome_counts_and_distinct_plot_pools_use_all_queries():
    action = np.array([True, False, True, True, False, False, False, False])
    fusion = np.array([1, 1, 0, 1, 1, 0, 1, 0])
    geo = np.array([0, 0, 1, 2, 1, 0, 0, 0])
    truth = np.zeros(8, dtype=int)
    plots = ["a", "a", "b", "c", "d", "e", "f", "f"]
    counts = evaluate_actions(action, fusion, geo, truth, plots)
    expected = {
        "query_count": 8, "query_plot_count": 6,
        "actions": 3, "action_plots": 3, "coverage": 3 / 8,
        "rescues": 1, "rescue_plots": 1, "harms": 1, "harm_plots": 1,
        "net_rescues": 0, "both_wrong_actions": 1, "both_wrong_action_plots": 1,
        "potential_rescues": 3, "potential_rescue_plots": 2,
        "potential_harms": 1, "potential_harm_plots": 1,
        "missed_rescues": 2, "missed_rescue_plots": 2,
        "both_wrong_disagreements": 1, "both_wrong_disagreement_plots": 1,
        "both_wrong_agreements": 1, "both_wrong_agreement_plots": 1,
        "disagreements": 5, "disagreement_plots": 4,
        "agreements": 3, "agreement_plots": 3,
        "fusion_correct": 3, "geo_correct": 5, "routed_correct": 3,
        "delta_acc1": 0.0, "oracle_top1_acc": 6 / 8,
    }
    assert counts == expected
    json.dumps(counts, allow_nan=False)


def test_action_on_agreement_is_rejected():
    with pytest.raises(ValueError, match="agreements"):
        evaluate_actions([True], [0], [0], [0], ["p"])


def test_exact_harm_cap_and_minimum_plot_boundary_are_feasible():
    result = select_policy(*_case(rescues=10, harms=7, both_wrong=3, total=100))
    zero = _zero_record(result)
    assert zero["feasible"]
    assert zero["counts"]["action_plots"] == 20
    assert zero["counts"]["net_rescues"] == 3
    assert result["selected_policy"] == finite_policy(0.0)
    assert result["selected_counts"] == zero["counts"]


def test_harm_cap_is_integer_exact_and_not_rounded_for_display():
    result = select_policy(*_case(rescues=10, harms=8, both_wrong=2, total=100))
    assert _zero_record(result)["infeasibility_reasons"] == ["harm_cap_exceeded"]
    assert result["selected_policy"]["kind"] == "never_intervene"


@pytest.mark.parametrize("n, feasible", [(2000, True), (2001, False)])
def test_coverage_boundary_uses_all_query_denominator(n, feasible):
    result = select_policy(*_case(rescues=20, total=n))
    zero = _zero_record(result)
    assert zero["feasible"] is feasible
    assert zero["counts"]["coverage"] == 20 / n
    assert ("coverage_below_one_percent_of_all_queries" in zero["infeasibility_reasons"]) is (not feasible)


def test_twenty_photos_from_nineteen_plots_fail_plot_guard():
    inputs = list(_case(rescues=20, total=100))
    inputs[-1][19] = inputs[-1][18]
    result = select_policy(*inputs)
    assert _zero_record(result)["infeasibility_reasons"] == ["fewer_than_20_acted_plots"]


def test_nonpositive_net_is_recorded_even_when_risk_cap_passes():
    result = select_policy(*_case(rescues=0, both_wrong=20))
    assert _zero_record(result)["infeasibility_reasons"] == ["nonpositive_net_rescue"]
    assert result["selection_reason"] == "no_feasible_finite_candidate"
    assert result["selected_counts"]["actions"] == 0
    assert result["candidate_records"][0]["infeasibility_reasons"] == ["typed_never_fallback"]


def test_equal_net_rescue_prefers_fewer_harms():
    inputs = list(_case(rescues=21, harms=1))
    # Low-score rescue + harm cancel. Removing both keeps net rescue at twenty.
    inputs[0][:] = 2.0
    inputs[0][0] = inputs[0][21] = 1.0
    result = select_policy(*inputs)
    assert result["selected_policy"] == finite_policy(1.5)
    assert result["selected_counts"]["rescues"] == 20
    assert result["selected_counts"]["harms"] == 0
    assert _zero_record(result)["counts"]["net_rescues"] == 20


def test_equal_net_and_harms_prefers_fewer_actions():
    inputs = list(_case(rescues=20, both_wrong=5))
    inputs[0][:20] = 2.0
    result = select_policy(*inputs)
    assert result["selected_policy"] == finite_policy(1.5)
    assert result["selected_counts"]["actions"] == 20
    assert _zero_record(result)["counts"]["actions"] == 25


def test_net_rescue_has_priority_over_fewer_harms_or_actions():
    inputs = list(_case(rescues=22, harms=1))
    inputs[0][:] = 2.0
    inputs[0][0:2] = inputs[0][22] = 1.0
    result = select_policy(*inputs)
    assert result["selected_policy"] == finite_policy(0.0)
    assert result["selected_counts"]["net_rescues"] == 21


def test_complete_tie_uses_higher_threshold():
    counts = {"net_rescues": 20, "harms": 0, "actions": 20}
    low = {"counts": counts, "policy": finite_policy(0.0)}
    high = {"counts": counts, "policy": finite_policy(1.0)}
    assert max([high, low], key=_selection_key) is high


def test_no_eligible_queries_preserves_full_cohort_and_pools():
    inputs = list(_case(rescues=20, harms=3, total=100))
    inputs[1][:] = False
    inputs[0][:] = np.nan
    result = select_policy(*inputs)
    counts = result["selected_counts"]
    assert result["selected_policy"]["kind"] == "never_intervene"
    assert len(result["candidate_records"]) == 2
    assert counts["query_count"] == 100
    assert counts["potential_rescues"] == counts["missed_rescues"] == 20
    assert counts["potential_harms"] == 3
    assert counts["actions"] == counts["routed_correct"] - counts["fusion_correct"] == 0


def test_selected_policy_and_records_are_json_serializable():
    result = select_policy(*_case(rescues=20, total=100))
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("scores, eligible", [
    ([], []), ([[1.0]], [True]), ([1.0], [1]),
    ([np.nan], [True]), ([np.inf], [True]), (["x"], [True]),
])
def test_invalid_score_inputs_are_rejected(scores, eligible):
    with pytest.raises(ValueError):
        enumerate_policies(scores, eligible)


def test_eligibility_cannot_mark_expert_agreement():
    with pytest.raises(ValueError, match="disagreements"):
        select_policy([1.0], [True], [0], [0], [0], ["p"])


@pytest.mark.parametrize("position, value", [
    (0, [1]), (1, [0.0]), (2, [18]), (3, [-1]), (4, [""]),
])
def test_invalid_action_inputs_are_rejected(position, value):
    args = [[False], [0], [1], [0], ["p"]]
    args[position] = value
    with pytest.raises(ValueError):
        evaluate_actions(*args)

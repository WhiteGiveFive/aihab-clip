"""Independent arithmetic and threshold checks for the fixed-expert study."""

from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from multimodal.geo_helpfulness_confidnet_fold4_report import (
    ConfidNetMetricError,
    build_report,
    expert_reference,
    format_report,
    policy_metrics,
    relaxed_bootstrap,
    validation_rows_and_actions,
)


def payload(prefix="val"):
    raw = np.array([[.2, .6, .2], [.6, .2, .2], [.1, .1, .8],
                    [.2, .7, .1], [.2, .6, .2], [.7, .2, .1]])
    geo = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7],
                    [.1, .25, .65], [.1, .8, .1], [.1, .65, .25]])
    return {
        "labels": np.array([0, 0, 2, 0, 0, 1]),
        "plot_ids": np.array([f"{prefix}p{x}" for x in [1, 1, 2, 2, 3, 3]]),
        "row_uids": np.array([f"{prefix}r{x}" for x in range(6)]),
        "image_logits": np.log(raw), "geo_logits": np.log(geo),
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


def four_seeds(prefix="val"):
    return {seed: payload(prefix) for seed in range(1, 5)}


def test_grid_common_threshold_and_independent_candidate_counts():
    rows, selections, actions = validation_rows_and_actions(four_seeds())
    assert len(selections) == 18
    assert len(actions) == 21
    for name, selected in selections.items():
        guard = selected["guardrails"]
        scores = rows[f"{selected['arm']}_score"].to_numpy()
        candidates = []
        for record in selected["candidate_curve"]:
            threshold = record["threshold_display"]
            action = np.zeros(len(rows), dtype=bool) if threshold is None else (
                (scores > threshold) & rows.raw_pred.ne(rows.geo_pred).to_numpy()
            )
            rescue = np.sum(action & rows.target_state.eq("rescue").to_numpy())
            harm = np.sum(action & rows.target_state.eq("harm").to_numpy())
            plots = rows.loc[action, "plot_idx"].nunique()
            feasible = (action.sum() * 100 >= len(rows) and plots >= guard["minimum_plots"]
                        and 10 * harm <= guard["harm_numerator"] * rescue and rescue > harm)
            assert record["acted_count"] == int(action.sum())
            assert record["rescued_count"] == int(rescue)
            assert record["harmed_count"] == int(harm)
            assert record["feasible"] == feasible
            if feasible:
                candidates.append(((rescue - harm, -harm, -action.sum(), threshold), action))
        if candidates:
            expected = max(candidates, key=lambda pair: pair[0])[1]
            assert selected["selected_feasible"]
            assert np.array_equal(actions[name], expected)
        else:
            assert not selected["selected_feasible"]
            assert selected["threshold_display"] is None
            assert not actions[name].any()
    assert not selections["augmented__plots_20__harm_0p7"]["selected_feasible"]
    assert selections["augmented__plots_0__harm_0p7"]["pooled"]["rescued_count"] == 8


def test_no_positive_router_utility_never_intervenes_under_any_grid_setting():
    seeds = four_seeds()
    for source in seeds.values():
        source["router_baseline_q"][:] = [.1, .2, .3, .4]
        source["router_augmented_q"][:] = [.25, .25, .25, .25]
    _, selections, actions = validation_rows_and_actions(seeds)
    assert all(not selected["selected_feasible"] for selected in selections.values())
    assert all(not action.any() for action in actions.values())


def test_matched_reference_uses_current_experts_and_accuracy_oracle():
    result = expert_reference(four_seeds())
    for item in result["per_seed"].values():
        assert item["experts"]["raw"]["accuracy"] == pytest.approx(2 / 6)
        assert item["experts"]["geo"]["accuracy"] == pytest.approx(3 / 6)
        assert item["fusion_error_count"] == 4
        assert item["rescue_available"] == 2
        assert item["harm_available"] == 1
        assert item["oracle_accuracy"] == pytest.approx(4 / 6)
        assert item["oracle_accuracy_gain_vs_fusion"] == pytest.approx(2 / 6)


def test_policy_metrics_seed_means_and_pooled_counts():
    rows, _, actions = validation_rows_and_actions(four_seeds())
    result = policy_metrics(rows, actions["augmented_zero"])
    assert result["pooled"]["rescues"] == 8
    assert result["pooled"]["harms"] == 0
    assert result["pooled"]["neutral_wrong"] == 4
    assert result["mean_seed"]["rescues"] == 2
    assert result["mean_seed"]["accuracy"] == pytest.approx(4 / 6)
    for seed, metric in result["per_seed"].items():
        selected = rows.training_seed.eq(int(seed)).to_numpy()
        predictions = np.where(actions["augmented_zero"], rows.geo_pred, rows.raw_pred)
        assert metric["weighted_f1"] == pytest.approx(f1_score(
            rows.loc[selected, "label_id_dense"], predictions[selected], average="weighted"
        ))


def test_paired_bootstrap_matches_whole_plot_image_expansion():
    seeds = four_seeds()
    seeds[4]["router_augmented_q"] = seeds[4]["router_baseline_q"].copy()
    rows, _, actions = validation_rows_and_actions(seeds)
    result = relaxed_bootstrap(rows, actions, n_resamples=37, random_seed=62)
    plots = np.unique(rows.plot_idx)
    rng = np.random.default_rng(62)
    differences = []
    prediction = {
        name: np.where(action, rows.geo_pred, rows.raw_pred)
        for name, action in actions.items()
    }
    for _ in range(37):
        selected_plots = plots[rng.integers(0, len(plots), size=len(plots))]
        seed_metrics = []
        for seed in range(1, 5):
            indices = np.concatenate([
                np.flatnonzero(rows.plot_idx.eq(plot) & rows.training_seed.eq(seed))
                for plot in selected_plots
            ])
            labels = rows.label_id_dense.to_numpy()[indices]
            augmented = prediction["augmented_zero"][indices]
            baseline = prediction["baseline_zero"][indices]
            seed_metrics.append([
                np.mean(augmented == labels) - np.mean(baseline == labels),
                f1_score(labels, augmented, average="weighted") - f1_score(labels, baseline, average="weighted"),
            ])
        differences.append(np.mean(seed_metrics, axis=0))
    low, high = np.quantile(differences, [.025, .975], axis=0)
    contrast = result["contrasts"]["augmented_zero_minus_baseline_zero"]
    for index, metric in enumerate(("accuracy", "weighted_f1")):
        assert contrast[metric]["ci95_lower"] == pytest.approx(low[index])
        assert contrast[metric]["ci95_upper"] == pytest.approx(high[index])
    assert result["plot_count"] == 3
    assert result["selection_adjusted"] is False


def test_build_report_is_json_safe_and_marks_exploratory_design():
    result = build_report(four_seeds("train"), four_seeds(), n_resamples=11)
    json.dumps(result, allow_nan=False)
    assert result["fixed"]["router_training_confidence_status"] == "in_sample_for_confidnet_held_out_from_fixed_experts"
    assert result["relaxed"]["selection_and_reporting_use_same_rows"] is True
    assert result["relaxed"]["fixed_policy_and_bootstrap_parity"] is True
    assert len(result["relaxed"]["candidate_curves"]) == 18
    markdown = format_report(result)
    assert "same frozen folds-1–3 experts" in markdown
    assert "No cleaned-test evaluation" in markdown
    assert "augmented__plots_0__harm_0p9" in markdown


def test_seed_row_order_does_not_change_reports():
    original = four_seeds()
    changed = deepcopy(original)
    changed[3] = {name: value[::-1] for name, value in changed[3].items()}
    rows1, selected1, actions1 = validation_rows_and_actions(original)
    rows2, selected2, actions2 = validation_rows_and_actions(changed)
    assert rows1.equals(rows2)
    assert selected1 == selected2
    assert all(np.array_equal(actions1[name], actions2[name]) for name in actions1)


def test_report_rejects_training_validation_overlap_and_missing_seed():
    with pytest.raises(ConfidNetMetricError, match="disjoint"):
        build_report(four_seeds(), four_seeds(), n_resamples=3)
    with pytest.raises(ConfidNetMetricError, match="requires seeds"):
        validation_rows_and_actions({1: payload()})


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_bootstrap_rejects_invalid_resample_count(count):
    rows, _, actions = validation_rows_and_actions(four_seeds())
    with pytest.raises(ConfidNetMetricError, match="positive integer"):
        relaxed_bootstrap(rows, actions, n_resamples=count)

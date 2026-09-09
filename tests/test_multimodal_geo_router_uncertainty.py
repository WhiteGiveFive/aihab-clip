"""Pure synthetic checks of paired, stratified complete-plot uncertainty."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_router_uncertainty as uncertainty


@pytest.fixture
def fixture():
    records = []
    image_index = 0
    for plot, habitat, images in (
        ("A0", 0, 2),
        ("A1", 0, 1),
        ("B0", 1, 3),
        ("B1", 1, 2),
    ):
        for _ in range(images):
            for seed in range(1, 5):
                state = uncertainty.STATES[(image_index + seed) % 4]
                raw, geo = {
                    "rescue": (1, 0),
                    "harm": (0, 1),
                    "both_correct": (0, 0),
                    "both_wrong": (1, 2),
                }[state]
                records.append(
                    {
                        "row_uid": f"image{image_index}",
                        "plot_idx": plot,
                        "canonical_l3_id": habitat,
                        "training_seed": seed,
                        "raw_pred": raw,
                        "geo_pred": geo,
                        "target_state": state,
                        "image_pred": (image_index + seed) % 18,
                    }
                )
            image_index += 1
    rows = pd.DataFrame(records)
    disagreement = rows.raw_pred.ne(rows.geo_pred).to_numpy()
    policies = {
        "primary_full727": disagreement & rows.target_state.eq("rescue").to_numpy(),
        "ablation_no_pairs79": disagreement & rows.training_seed.le(2).to_numpy(),
        "always_geo_on_geo_raw_disagreement": disagreement,
        "never_intervene": np.zeros(len(rows), dtype=bool),
    }
    return rows, policies


def test_bootstrap_point_estimates_json_and_paired_identity(fixture):
    rows, policies = fixture
    policies["identical_to_primary"] = policies["primary_full727"].copy()
    summary, replicates = uncertainty.bootstrap_policy_comparison(
        rows, policies, replicates=80
    )
    json.dumps(summary, allow_nan=False)
    assert summary["replicates"] == 80
    assert summary["plot_count"] == 4 and summary["image_count"] == 8
    assert summary["stratum_plot_counts"] == {"0": 2, "1": 2}
    primary = summary["policies"]["primary_full727"]["metrics"]
    assert primary["rescued_count"]["point_estimate"] == 8
    assert primary["mean_seed_rescued_count"]["point_estimate"] == 2
    assert primary["mean_seed_net_utility"]["point_estimate"] == 2
    assert primary["mean_seed_delta_accuracy"]["point_estimate"] == 0.25
    assert primary["override_precision"]["point_estimate"] == 1
    identical = summary["paired_primary_minus_comparator_delta_accuracy"][
        "identical_to_primary"
    ]
    assert identical["point_estimate"] == identical["lower"] == identical["upper"] == 0
    subset = replicates.loc[replicates.policy.eq("identical_to_primary")]
    np.testing.assert_array_equal(subset.primary_minus_policy_delta_accuracy, 0)
    assert len(replicates) == 80 * 5
    assert list(zip(replicates.replicate, replicates.policy)) == sorted(
        zip(replicates.replicate, replicates.policy)
    )


def test_complete_plot_weights_match_independent_stratified_resampling(fixture):
    rows, policies = fixture
    _, observed = uncertainty.bootstrap_policy_comparison(
        rows, policies, replicates=12, random_seed=123
    )
    rng = np.random.default_rng(123)
    plots = np.array(["A0", "A1", "B0", "B1"])
    for replicate in range(12):
        selected = np.concatenate(
            [rng.choice([0, 1], 2, replace=True), rng.choice([2, 3], 2, replace=True)]
        )
        plot_weights = dict(zip(plots, np.bincount(selected, minlength=4)))
        weights = rows.plot_idx.map(plot_weights).to_numpy()
        for name, action in policies.items():
            output = observed.loc[
                (observed.replicate == replicate) & (observed.policy == name)
            ].iloc[0]
            rescued = int(np.sum(weights * action * rows.target_state.eq("rescue")))
            harmed = int(np.sum(weights * action * rows.target_state.eq("harm")))
            assert output.rescued_count == rescued
            assert output.harmed_count == harmed
            assert output.row_count == weights.sum()
            assert output.mean_seed_net_utility == (rescued - harmed) / 4
            assert output.mean_seed_delta_accuracy == pytest.approx(
                (rescued - harmed) / weights.sum(), abs=1e-15
            )


def test_input_permutation_and_policy_order_do_not_change_results(fixture):
    rows, policies = fixture
    summary, replicates = uncertainty.bootstrap_policy_comparison(
        rows, policies, replicates=50
    )
    order = np.random.default_rng(77).permutation(len(rows))
    reordered = {name: policies[name][order] for name in reversed(policies)}
    other_summary, other_replicates = uncertainty.bootstrap_policy_comparison(
        rows.iloc[order], reordered, replicates=50
    )
    assert summary == other_summary
    pd.testing.assert_frame_equal(replicates, other_replicates)


def test_undefined_denominators_and_minimum_valid_fraction(fixture):
    rows, policies = fixture
    summary, replicates = uncertainty.bootstrap_policy_comparison(
        rows, policies, replicates=40
    )
    never = summary["policies"]["never_intervene"]["metrics"]
    assert never["override_precision"]["point_estimate"] is None
    assert never["override_precision"]["valid_replicates"] == 0
    assert never["override_precision"]["lower"] is None
    assert not never["override_precision"]["interval_available"]
    assert (
        replicates.loc[replicates.policy.eq("never_intervene"), "override_precision"]
        .isna()
        .all()
    )
    accepted = uncertainty._interval(1.0, np.r_[np.arange(95), np.full(5, np.nan)])
    rejected = uncertainty._interval(1.0, np.r_[np.arange(94), np.full(6, np.nan)])
    assert accepted["valid_fraction"] == 0.95 and accepted["interval_available"]
    assert accepted["lower"] == np.quantile(np.arange(95), 0.025, method="linear")
    assert rejected["lower"] is None and rejected["upper"] is None
    assert rejected["valid_replicates"] == 94


def test_singleton_strata_keep_every_complete_plot_once(fixture):
    rows, policies = fixture
    rows = rows.copy()
    rows["canonical_l3_id"] = rows.plot_idx.map({"A0": 0, "A1": 1, "B0": 2, "B1": 3})
    summary, replicates = uncertainty.bootstrap_policy_comparison(
        rows, policies, replicates=15
    )
    for name in policies:
        values = replicates.loc[replicates.policy.eq(name)]
        assert values.rescued_count.nunique() == values.harmed_count.nunique() == 1
    assert (
        summary["policies"]["primary_full727"]["metrics"]["mean_seed_delta_accuracy"][
            "lower"
        ]
        == 0.25
    )


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "missing_seed",
        "missing_image_seed",
        "mixed_plot_habitat",
        "mixed_image_plot",
        "bad_state",
        "bad_prediction",
        "float_seed",
        "null",
    ],
)
def test_invalid_complete_plot_inputs_rejected(fixture, damage):
    rows, policies = fixture
    rows = rows.copy()
    if damage == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
        policies = {name: np.r_[mask, mask[0]] for name, mask in policies.items()}
    elif damage in {"missing_seed", "missing_image_seed"}:
        keep = (
            rows.training_seed.ne(4).to_numpy()
            if damage == "missing_seed"
            else np.arange(len(rows)) != 0
        )
        rows, policies = rows.loc[keep], {
            name: mask[keep] for name, mask in policies.items()
        }
    elif damage == "mixed_plot_habitat":
        rows.loc[rows.row_uid.eq("image0"), "canonical_l3_id"] = 99
    elif damage == "mixed_image_plot":
        rows.loc[0, "plot_idx"] = "other"
    elif damage == "bad_state":
        rows.loc[0, "target_state"] = "unknown"
    elif damage == "bad_prediction":
        rows.loc[0, "geo_pred"] = 18
    elif damage == "float_seed":
        rows["training_seed"] = rows.training_seed.astype(float)
    else:
        rows.loc[0, "plot_idx"] = None
    with pytest.raises(ValueError):
        uncertainty.bootstrap_policy_comparison(rows, policies, replicates=2)


@pytest.mark.parametrize(
    "damage",
    [
        "integer_actions",
        "wrong_length",
        "agreement_action",
        "missing_primary",
        "empty_name",
    ],
)
def test_invalid_policy_actions_rejected(fixture, damage):
    rows, policies = fixture
    if damage == "integer_actions":
        policies["primary_full727"] = policies["primary_full727"].astype(int)
    elif damage == "wrong_length":
        policies["primary_full727"] = policies["primary_full727"][:-1]
    elif damage == "agreement_action":
        policies["primary_full727"] = np.ones(len(rows), dtype=bool)
    elif damage == "missing_primary":
        policies.pop("primary_full727")
    else:
        policies[""] = policies.pop("primary_full727")
    with pytest.raises(ValueError):
        uncertainty.bootstrap_policy_comparison(rows, policies, replicates=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replicates": 0},
        {"replicates": True},
        {"random_seed": -1},
        {"random_seed": 1.5},
    ],
)
def test_invalid_bootstrap_settings(fixture, kwargs):
    with pytest.raises(ValueError):
        uncertainty.bootstrap_policy_comparison(*fixture, **kwargs)


def test_subgroup_counts_support_and_optional_pair_columns(fixture):
    rows, policies = fixture
    report = uncertainty.subgroup_policy_report(rows, policies)
    assert set(report.group_family) == {
        "habitat",
        "plot",
        "geo_raw_pred_pair",
        "image_geo_pred_pair",
    }
    subset = report.loc[
        (report.group_family == "habitat")
        & (report.group_value == "0")
        & (report.policy == "primary_full727")
        & (report.aggregation == "pooled")
    ].iloc[0]
    assert subset.row_count == 12 and subset.image_count == 3 and subset.plot_count == 2
    assert (
        subset.target_rescue_count
        + subset.target_harm_count
        + subset.target_both_correct_count
        + subset.target_both_wrong_count
        == 12
    )
    assert bool(subset.low_support)
    assert pd.isna(subset.training_seed)
    no_image = uncertainty.subgroup_policy_report(
        rows.drop(columns="image_pred"), policies
    )
    assert "image_geo_pred_pair" not in set(no_image.group_family)


def test_subgroup_pair_arithmetic_does_not_overflow_int8(fixture):
    rows, policies = fixture
    rows = rows.copy()
    rows["image_pred"] = np.int8(17)
    rows["geo_pred"] = np.int8(16)
    rows["raw_pred"] = np.int8(15)
    rows["target_state"] = "both_wrong"
    rows["geo_raw_pred_pair"] = np.int16(16 * 18 + 15)
    rows["image_geo_pred_pair"] = np.int16(17 * 18 + 16)
    result = uncertainty.subgroup_policy_report(rows, policies)
    assert set(
        result.loc[result.group_family.eq("geo_raw_pred_pair"), "group_value"]
    ) == {"303"}
    assert set(
        result.loc[result.group_family.eq("image_geo_pred_pair"), "group_value"]
    ) == {"322"}
    rows.loc[0, "image_geo_pred_pair"] = 1
    with pytest.raises(ValueError, match="pair code mismatch"):
        uncertainty.subgroup_policy_report(rows, policies)


def test_no_fitting_or_file_access_imports():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(uncertainty.__file__).read_text())
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        name
        and (
            name.startswith("sklearn")
            or name.startswith("scipy")
            or "geo_helpfulness" in name
        )
        for name in imported
    )


def test_rare_denominators_are_counted_without_dropping_whole_replicates():
    rows = pd.DataFrame(
        [
            {
                "row_uid": f"i{plot}",
                "plot_idx": f"p{plot}",
                "training_seed": seed,
                "canonical_l3_id": 0,
                "raw_pred": 1,
                "geo_pred": 0,
                "target_state": "rescue" if plot == 0 else "both_wrong",
            }
            for plot in range(5)
            for seed in range(1, 5)
        ]
    )
    action = rows.target_state.eq("rescue").to_numpy()
    summary, replicate = uncertainty.bootstrap_policy_comparison(
        rows, {"primary_full727": action}, replicates=200
    )
    metrics = summary["policies"]["primary_full727"]["metrics"]
    recall = metrics["rescue_recall"]
    assert 0 < recall["valid_replicates"] < 190
    assert recall["lower"] is None and recall["upper"] is None
    assert metrics["mean_seed_delta_accuracy"]["valid_replicates"] == 200
    assert len(replicate) == 200
    assert replicate.rescue_recall.notna().sum() == recall["valid_replicates"]


def test_support_threshold_uses_unique_images_and_plots_at_exact_boundary():
    rows = pd.DataFrame(
        [
            {
                "row_uid": f"i{image}",
                "plot_idx": f"p{image // 2}",
                "training_seed": seed,
                "canonical_l3_id": 0,
                "raw_pred": 1,
                "geo_pred": 0,
                "target_state": "both_wrong",
            }
            for image in range(20)
            for seed in range(1, 5)
        ]
    )
    policies = {"primary_full727": np.zeros(len(rows), dtype=bool)}
    report = uncertainty.subgroup_policy_report(rows, policies)
    habitat = report.loc[report.group_family.eq("habitat")]
    assert not habitat.low_support.any()
    pooled = habitat.loc[habitat.aggregation.eq("pooled")].iloc[0]
    assert (
        pooled.row_count == 80 and pooled.image_count == 20 and pooled.plot_count == 10
    )
    assert report.loc[report.group_family.eq("plot"), "low_support"].all()


def test_frozen_default_replicate_count_and_seed(fixture):
    summary, replicate = uncertainty.bootstrap_policy_comparison(*fixture)
    assert summary["replicates"] == 2000 and summary["random_seed"] == 20260826
    assert summary["confidence_level"] == 0.95
    assert len(replicate) == 2000 * len(fixture[1])

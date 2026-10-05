"""Synthetic-only relaxed RF reporting, predecessor parity and access guards."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_legacy_capacity_report as capacity
from multimodal import geo_helpfulness_legacy_router_report as legacy
from multimodal import geo_helpfulness_relaxed_guardrails_report as relaxed_lr
from multimodal import geo_helpfulness_relaxed_rf_report as report
from test_multimodal_geo_legacy_capacity_report import inputs as capacity_inputs


def inputs(family_shift=0):
    parent, _ = capacity_inputs(family_shift)
    experts, routers, old_actions, labels, old_policies = parent
    # Give one plot a distinct RF score distribution so paired bootstrap tests
    # cannot accidentally pass using constant per-plot utility in this fixture.
    varied = (
        routers.learner.eq("rf")
        & routers.row_uid.eq("image-00")
        & routers.q_rescue.gt(0.7)
    )
    routers.loc[varied, list(report.Q_COLUMNS)] = [0.35, 0.25, 0.2, 0.2]
    routers.loc[varied, "score"] = (
        routers.loc[varied, "q_rescue"] - routers.loc[varied, "q_harm"]
    )
    for variant in report.VARIANTS:
        original = "capacity__rf__" + variant
        chosen = old_actions.policy_id.eq(original)
        scores = routers.loc[routers.policy_id.eq(original), "score"].to_numpy()
        mask = legacy.apply_heuristic_threshold(
            scores,
            experts.raw_pred.to_numpy(),
            experts.geo_pred.to_numpy(),
            old_policies[original]["threshold_specification"],
        )
        old_actions.loc[chosen, "score"] = scores
        old_actions.loc[chosen, "acted"] = mask
        old_actions.loc[chosen, "final_pred"] = np.where(
            mask, experts.geo_pred, experts.raw_pred
        )
    declarations, actions = {}, [old_actions]
    for name in report.NEW_POLICIES:
        variant, minimum, numerator = report._policy_identity(name)
        original = "capacity__rf__" + variant
        router = routers.loc[routers.policy_id.eq(original)]
        if minimum == 0 and numerator == 5:
            specification = deepcopy(report.NEVER_THRESHOLD)
        else:
            threshold = {5: 0.2, 7: 0.0, 9: float(router.score.max())}[numerator]
            specification = {
                "kind": "finite",
                "comparison": "strict_gt",
                "value_hex": threshold.hex(),
            }
        declarations[name] = {
            "policy_id": name,
            "learner": "rf",
            "variant_id": variant,
            "original_policy_id": original,
            "config_id": old_policies[original]["config_id"],
            "guardrail_id": f"plots_{minimum}__harm_0p{numerator}",
            "minimum_plots": minimum,
            "harm_numerator": numerator,
            "harm_denominator": 10,
            "minimum_coverage_fraction": 0.01,
            "threshold_specification": specification,
            "models": {"1": {"path": "synthetic-must-not-open.json"}},
        }
        frame = old_actions.loc[old_actions.policy_id.eq(original)].copy()
        frame["study_id"], frame["policy_id"] = report.STUDY_ID, name
        frame["schema_version"] = "geo_helpfulness.relaxed_rf.actions.v1"
        frame["threshold_specification"] = json.dumps(specification, sort_keys=True)
        mask = legacy.apply_heuristic_threshold(
            frame.score.to_numpy(),
            experts.raw_pred.to_numpy(),
            experts.geo_pred.to_numpy(),
            specification,
        )
        frame["acted"], frame["final_pred"] = mask, np.where(
            mask, experts.geo_pred, experts.raw_pred
        )
        actions.append(frame)
    return (
        experts,
        routers,
        pd.concat(actions, ignore_index=True),
        labels,
        old_policies,
        declarations,
    ), parent


@pytest.fixture(scope="module")
def scored():
    args, parent = inputs()
    return args, report.evaluate_family(*args), capacity.evaluate_family(*parent)


@pytest.fixture(scope="module")
def oracle_scored():
    args, parent = inputs(family_shift=1)
    return args, report.evaluate_family(*args), capacity.evaluate_family(*parent)


def test_metadata_contract_and_complete_grid(scored):
    args, _, _ = scored
    assert report.STUDY_ID == "relaxed_guardrails_rf_v1"
    assert report.MINIMUM_PLOTS == (20, 10, 0)
    assert report.HARM_NUMERATORS == (5, 7, 9)
    assert report.HARM_DENOMINATOR == 10
    assert report.MINIMUM_COVERAGE_FRACTION == 0.01
    expected = {
        f"relaxed__rf__{v}__plots_{p}__harm_0p{h}"
        for v in legacy.VARIANTS
        for p in (20, 10, 0)
        for h in (5, 7, 9)
    }
    assert len(expected) == len(report.NEW_POLICIES) == 27
    assert set(report.NEW_POLICIES) == expected
    normalized = report.validate_new_policies(args[5], args[4])
    assert normalized == args[5]
    normalized[report.NEW_POLICIES[0]]["config_id"] = "not-shared"
    assert normalized != args[5]


@pytest.mark.parametrize("fixture_name", ["scored", "oracle_scored"])
def test_artifacts_and_context_keep_all_models_and_fortyfour_policies(
    request, fixture_name
):
    _, result, _ = request.getfixturevalue(fixture_name)
    assert set(result) == set(report.SCORE_FILES)
    assert len(result["router_metrics.json"]) == 12
    assert (
        len(result["policy_metrics.json"])
        == len(result["classification.json"]["policies"])
        == 44
    )
    summary = result["summary.json"]
    assert set(summary["groups"]) == set(report.NEW_POLICIES)
    assert (summary["row_count"], summary["image_count"], summary["plot_count"]) == (
        32,
        8,
        4,
    )
    assert (
        summary["router_count"],
        summary["relaxed_router_count"],
        summary["original_policy_count"],
        summary["new_policy_count"],
        summary["policy_count"],
    ) == (12, 3, 17, 27, 44)
    assert summary["original_rf_policies"] == [
        "capacity__rf__" + variant for variant in report.VARIANTS
    ]
    assert summary["original_baseline_policy"] == "primary_full727"
    for key in (
        "selection_or_fitting_performed",
        "model_reselection_performed",
        "probability_calibration_fitted",
        "M4_gate_rerun",
        "checkpoint_family_selection_performed",
        "test_guardrail_feasibility_enforced",
        "independent_confirmation",
    ):
        assert summary[key] is False
    assert summary["original_M4_no_go_preserved"] is True
    assert summary["oracle_checkpoint_test_selection_biased"] == (
        fixture_name == "oracle_scored"
    )
    assert "not guaranteed" in summary["calibration_disclosure"]
    assert "test-selection-biased" in summary["checkpoint_selection_disclosure"]
    for filename, value in result.items():
        if filename.endswith(".json"):
            json.dumps(value, allow_nan=False)
        else:
            assert isinstance(value, pa.Table)
            assert value.schema.metadata is None


@pytest.mark.parametrize("fixture_name", ["scored", "oracle_scored"])
def test_all_seventeen_predecessor_payloads_and_rows_remain_exact(
    request, fixture_name
):
    args, result, parent = request.getfixturevalue(fixture_name)
    for filename in ("expert_metrics.json", "router_metrics.json"):
        assert result[filename] == parent[filename]
    assert (
        result["classification.json"]["experts"]
        == parent["classification.json"]["experts"]
    )
    assert result["summary.json"]["predecessor_summary"] == parent["summary.json"]
    assert (
        result["paired_comparisons.json"]["predecessor_paired_comparisons"]
        == parent["paired_comparisons.json"]
    )
    for name in report.OLD_POLICIES:
        assert (
            result["policy_metrics.json"][name] == parent["policy_metrics.json"][name]
        )
        assert (
            result["classification.json"]["policies"][name]
            == parent["classification.json"]["policies"][name]
        )
        assert (
            result["bootstrap_summary.json"]["policies"][name]
            == parent["bootstrap_summary.json"]["policies"][name]
        )
        if name != report.BASELINE:
            field = "paired_primary_minus_comparator_delta_accuracy"
            assert (
                result["bootstrap_summary.json"][field][name]
                == parent["bootstrap_summary.json"][field][name]
            )
    assert result["scoring_targets.parquet"].equals(parent["scoring_targets.parquet"])
    for filename, column in (
        ("bootstrap_replicates.parquet", "policy"),
        ("subgroup_metrics.parquet", "policy"),
        ("actions_with_outcomes.parquet", "policy_id"),
    ):
        frame = result[filename].to_pandas()
        subset = frame.loc[frame[column].isin(report.OLD_POLICIES)].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(subset, parent[filename].to_pandas())
    outcomes = result["actions_with_outcomes.parquet"].to_pandas()
    pd.testing.assert_frame_equal(
        outcomes.loc[:, args[2].columns].reset_index(drop=True),
        args[2]
        .sort_values(["policy_id", *report.KEY], kind="stable")
        .reset_index(drop=True),
    )


@pytest.mark.parametrize("fixture_name", ["scored", "oracle_scored"])
def test_all_new_counts_selected_probability_top3_and_four_seed_classification(
    request, fixture_name
):
    (experts, _, _, _, _, declarations), result, _ = request.getfixturevalue(
        fixture_name
    )
    targets = result["scoring_targets.parquet"].to_pandas()
    outcomes = result["actions_with_outcomes.parquet"].to_pandas()
    y = targets.label_id_dense.to_numpy()
    for name in report.NEW_POLICIES:
        frame = outcomes.loc[outcomes.policy_id.eq(name)]
        mask = frame.acted.to_numpy()
        selected = np.where(
            mask[:, None],
            np.stack(experts.geo_prob_calibrated),
            np.stack(experts.raw_prob_calibrated),
        )
        final = frame.final_pred.to_numpy()
        top3 = (np.argsort(-selected, axis=1, kind="stable")[:, :3] == y[:, None]).any(
            1
        )
        classification = result["classification.json"]["policies"][name]
        assert (
            classification["pooled_seed_realizations"]["top1_accuracy"]
            == (final == y).mean()
        )
        assert (
            classification["pooled_seed_realizations"]["top3_accuracy"] == top3.mean()
        )
        assert len(classification["pooled_seed_realizations"]["per_class"]) == 18
        assert set(classification["per_seed"]) == {"1", "2", "3", "4"}
        reference = legacy.policy_metrics(targets, mask)
        policy = result["policy_metrics.json"][name]
        for key in (
            "per_seed",
            "pooled",
            "mean_seed_net_utility",
            "mean_seed_delta_accuracy",
        ):
            assert policy[key] == reference[key]
        np.testing.assert_array_equal(
            frame.net_correct,
            frame.final_correct.astype(int) - frame.raw_correct.astype(int),
        )
        original = declarations[name]["original_policy_id"]
        group = result["summary.json"]["groups"][name]
        assert (
            group["four_state"]
            == result["router_metrics.json"][original]["all_rows"]["seed_summary"]
        )
        assert (
            group["disagreement"]
            == result["router_metrics.json"][original]["disagreement"]["seed_summary"]
        )
        assert group["plot_diagnostics"] == relaxed_lr._plot_diagnostics(targets, mask)
        if not mask.any():
            assert classification == result["classification.json"]["experts"]["raw"]
            assert policy["all_predictions_equal_raw"] is True
            assert policy["raw_fallback_selected"] == (
                declarations[name]["threshold_specification"]["kind"]
                == "never_intervene"
            )


def test_guardrails_never_reject_evaluation_harm_or_low_support(scored, oracle_scored):
    name = "relaxed__rf__primary_full727__plots_20__harm_0p5"
    for fixture in (scored, oracle_scored):
        record = fixture[1]["summary.json"]["groups"][name]
        assert record["frozen_policy"]["minimum_plots"] == 20
        assert record["plot_diagnostics"]["actual_acted_plot_count"] == 4
        assert record["policy"]["pooled"]["acted_count"] > 0
        assert record["policy"]["test_guardrail_feasibility_enforced"] is False
    counts = oracle_scored[1]["policy_metrics.json"][name]["pooled"]
    assert counts["rescued_count"] == 0 < counts["harmed_count"]
    assert counts["rescued_minus_harmed"] < 0
    # A finite threshold equal to the maximum score yields no actions; do not
    # replace it with typed never-intervene to satisfy development feasibility.
    empty = scored[1]["policy_metrics.json"][
        "relaxed__rf__primary_full727__plots_20__harm_0p9"
    ]
    assert empty["pooled"]["coverage_fraction"] == 0
    assert empty["threshold_specification"]["kind"] == "finite"
    assert empty["raw_fallback_selected"] is False


@pytest.mark.parametrize("fixture_name", ["scored", "oracle_scored"])
def test_shared_bootstrap_all_twentyseven_contrasts_use_matching_rf_not_lr(
    request, fixture_name
):
    (_, _, _, _, _, declarations), result, parent = request.getfixturevalue(
        fixture_name
    )
    bootstrap = result["bootstrap_summary.json"]
    assert (bootstrap["replicates"], bootstrap["random_seed"]) == (2000, 20260826)
    assert (
        bootstrap["resample_multiplicity_sha256"]
        == parent["bootstrap_summary.json"]["resample_multiplicity_sha256"]
    )
    assert bootstrap["retain_all_images_and_training_seeds"] is True
    assert bootstrap["selection_repeated_in_bootstrap"] is False
    replicates = result["bootstrap_replicates.parquet"].to_pandas()
    assert len(replicates) == 88000
    assert replicates.groupby("replicate").size().eq(44).all()
    wide = replicates.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    paired = result["paired_comparisons.json"]
    assert (
        paired["contrast_count"] == sum(map(len, paired["comparisons"].values())) == 27
    )
    assert paired["difference_sign"] == "named_policy_minus_matching_original_RF"
    assert "not simultaneous" in paired["uncertainty_disclosure"]
    assert paired["predecessor_paired_comparisons"]["contrast_count"] == 27
    nonconstant = False
    for name, comparisons in paired["comparisons"].items():
        comparator = declarations[name]["original_policy_id"]
        assert set(comparisons) == {comparator}
        interval = comparisons[comparator]
        draws = wide[name] - wide[comparator]
        nonconstant |= draws.std() > 1e-4
        np.testing.assert_allclose(
            [interval["lower"], interval["upper"]],
            np.quantile(draws, [0.025, 0.975]),
            rtol=0,
            atol=1e-15,
        )
        assert (
            interval["point_estimate"]
            == result["policy_metrics.json"][name]["mean_seed_delta_accuracy"]
            - result["policy_metrics.json"][comparator]["mean_seed_delta_accuracy"]
        )
        raw = result["summary.json"]["groups"][name]["bootstrap_delta_accuracy"]
        np.testing.assert_allclose(
            [raw["lower"], raw["upper"]],
            np.quantile(wide[name], [0.025, 0.975]),
            rtol=0,
            atol=1e-15,
        )
    assert nonconstant
    # Baseline-matching RF policy has zero paired gain despite positive/negative
    # gain over the original primary LR no-op, ruling out comparator confusion.
    name = "relaxed__rf__primary_full727__plots_20__harm_0p5"
    assert result["policy_metrics.json"][name]["mean_seed_delta_accuracy"] != 0
    interval = paired["comparisons"][name]["capacity__rf__primary_full727"]
    assert (interval["point_estimate"], interval["lower"], interval["upper"]) == (
        0,
        0,
        0,
    )


def test_seed_and_checkpoint_family_targets_reconstructed_independently(
    scored, oracle_scored
):
    left = scored[1]["scoring_targets.parquet"].to_pandas()
    right = oracle_scored[1]["scoring_targets.parquet"].to_pandas()
    assert left[list(report.KEY)].equals(right[list(report.KEY)])
    assert left.groupby("row_uid").target_state.nunique().eq(4).all()
    assert (left.target_state != right.target_state).all()
    assert (
        scored[1]["bootstrap_summary.json"]["resample_multiplicity_sha256"]
        == oracle_scored[1]["bootstrap_summary.json"]["resample_multiplicity_sha256"]
    )
    name = "relaxed__rf__ablation_numeric25__plots_20__harm_0p5"
    assert scored[1]["policy_metrics.json"][name]["pooled"]["rescued_minus_harmed"] > 0
    assert (
        oracle_scored[1]["policy_metrics.json"][name]["pooled"]["rescued_minus_harmed"]
        < 0
    )


@pytest.mark.parametrize(
    "bad",
    [
        "missing_new",
        "extra_new",
        "not_mapping",
        "policy_id",
        "learner",
        "variant_id",
        "original_policy_id",
        "config_id",
        "guardrail_id",
        "minimum_plots",
        "minimum_plots_bool",
        "harm_numerator",
        "harm_numerator_float",
        "harm_denominator",
        "coverage",
        "new_C",
        "threshold_kind",
        "threshold_missing",
        "threshold_comparison",
        "threshold_negative",
        "threshold_nonfinite",
        "threshold_noncanonical",
        "threshold_duplicate",
        "metadata_nan",
        "missing_old",
        "old_config",
        "old_learner",
        "old_primary_no_go",
    ],
)
def test_metadata_rejected_before_any_feature_or_label_access(scored, monkeypatch, bad):
    old, new = deepcopy(scored[0][4:])
    name = report.NEW_POLICIES[0]
    record = new[name]
    if bad == "missing_new":
        new.pop(name)
    elif bad == "extra_new":
        new["unexpected"] = deepcopy(record)
    elif bad == "not_mapping":
        new[name] = None
    elif bad in {
        "policy_id",
        "learner",
        "variant_id",
        "original_policy_id",
        "config_id",
        "guardrail_id",
        "minimum_plots",
        "minimum_plots_bool",
        "harm_numerator",
        "harm_numerator_float",
        "harm_denominator",
        "coverage",
        "new_C",
        "metadata_nan",
    }:
        key, value = {
            "policy_id": ("policy_id", "wrong"),
            "learner": ("learner", "xgb"),
            "variant_id": ("variant_id", report.VARIANTS[1]),
            "original_policy_id": (
                "original_policy_id",
                "capacity__xgb__primary_full727",
            ),
            "config_id": ("config_id", "reselected"),
            "guardrail_id": ("guardrail_id", "wrong"),
            "minimum_plots": ("minimum_plots", 10),
            "minimum_plots_bool": ("minimum_plots", True),
            "harm_numerator": ("harm_numerator", 9),
            "harm_numerator_float": ("harm_numerator", 5.0),
            "harm_denominator": ("harm_denominator", 100),
            "coverage": ("minimum_coverage_fraction", 0),
            "new_C": ("C_hex", (0.1).hex()),
            "metadata_nan": ("unused", float("nan")),
        }[bad]
        record[key] = value
    elif bad == "threshold_kind":
        record["threshold_specification"]["kind"] = "adaptive"
    elif bad == "threshold_missing":
        record.pop("threshold_specification")
    elif bad == "threshold_comparison":
        record["threshold_specification"]["comparison"] = "greater_or_equal"
    elif bad == "threshold_negative":
        record["threshold_specification"]["value_hex"] = (-0.1).hex()
    elif bad == "threshold_nonfinite":
        record["threshold_specification"]["value_hex"] = "inf"
    elif bad == "threshold_noncanonical":
        record["threshold_specification"]["value_hex"] = "0.0"
    elif bad == "threshold_duplicate":
        record["threshold_specification"] = (
            '{"kind":"finite","kind":"finite","comparison":"strict_gt","value_hex":"0x0.0p+0"}'
        )
    elif bad == "missing_old":
        old.pop("capacity__rf__primary_full727")
    elif bad == "old_config":
        old["capacity__rf__primary_full727"]["config_id"] = "other"
    elif bad == "old_learner":
        old["capacity__rf__primary_full727"]["learner"] = "xgb"
    elif bad == "old_primary_no_go":
        old[report.BASELINE]["threshold_specification"] = {
            "kind": "finite",
            "comparison": "strict_gt",
            "value_hex": (1.0).hex(),
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata validation touched features, labels, or files")

    monkeypatch.setattr(report, "_inputs", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(report.RelaxedRFReportError):
        report.validate_new_policies(new, old)
    with pytest.raises(report.RelaxedRFReportError):
        report.evaluate_family(None, None, None, None, old, new)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_router",
        "duplicate_router",
        "missing_action",
        "duplicate_action",
        "extra_action",
        "action_schema",
        "study_id",
        "schema_version",
        "expert_label",
        "expert_uppercase_label",
        "router_target",
        "router_true_probability",
        "action_outcome",
        "missing_probability",
        "family",
        "missing_family",
        "plot",
        "q_float32",
        "q_sum",
        "q_nan",
        "q_score",
        "router_config",
        "router_learner",
        "router_variant",
        "action_score",
        "lr_score",
        "action_float32",
        "threshold_declaration",
        "threshold_duplicate",
        "acted_integer",
        "acted_mismatch",
        "final_pred",
        "expert_pred",
        "old_threshold",
        "old_heuristic_score",
        "label_ontology",
        "missing_label",
        "label_plot",
        "missing_seed",
        "noncanonical",
    ],
)
def test_invalid_or_leaky_tables_fail_closed(scored, bad):
    e, r, a, labels, old, new = deepcopy(scored[0])
    name = report.NEW_POLICIES[0]
    ai = a.index[a.policy_id.eq(name)][0]
    ri = r.index[r.policy_id.eq("capacity__rf__primary_full727")][0]
    if bad == "missing_router":
        r = r.loc[r.policy_id.ne("capacity__rf__primary_full727")]
    elif bad == "duplicate_router":
        r = pd.concat([r, r.loc[[ri]]], ignore_index=True)
    elif bad == "missing_action":
        a = a.loc[a.policy_id.ne(name)]
    elif bad == "duplicate_action":
        a = pd.concat([a, a.loc[[ai]]], ignore_index=True)
    elif bad == "extra_action":
        a = pd.concat(
            [a, a.loc[a.policy_id.eq(name)].assign(policy_id="unexpected")],
            ignore_index=True,
        )
    elif bad == "action_schema":
        a["extra"] = "not-in-13-column-schema"
    elif bad == "study_id":
        a.loc[ai, "study_id"] = "other"
    elif bad == "schema_version":
        a.loc[ai, "schema_version"] = ""
    elif bad == "expert_label":
        e["label_id_dense"] = 0
    elif bad == "expert_uppercase_label":
        e["Label"] = 0
    elif bad == "router_target":
        r["target_state"] = "rescue"
    elif bad == "router_true_probability":
        r["raw_true_probability"] = 0.5
    elif bad == "action_outcome":
        a["rescued"] = False
    elif bad == "missing_probability":
        e = e.drop(columns="raw_prob_calibrated")
    elif bad == "family":
        a.loc[ai, "checkpoint_family"] = "oracle_test_best_model"
    elif bad == "missing_family":
        e = e.drop(columns="checkpoint_family")
    elif bad == "plot":
        a.loc[ai, "plot_idx"] = "other"
    elif bad == "q_float32":
        r["q_rescue"] = r.q_rescue.astype(np.float32)
    elif bad == "q_sum":
        r.loc[ri, "q_rescue"] += 0.2
    elif bad == "q_nan":
        r.loc[ri, "q_rescue"] = float("nan")
    elif bad == "q_score":
        r.loc[ri, "score"] = 0.333
    elif bad == "router_config":
        r.loc[ri, "config_id"] = "other"
    elif bad == "router_learner":
        r.loc[ri, "learner"] = "xgb"
    elif bad == "router_variant":
        r.loc[ri, "variant_id"] = report.VARIANTS[1]
    elif bad == "action_score":
        a.loc[ai, "score"] = 0.333
    elif bad == "lr_score":
        a.loc[a.policy_id.eq(name), "score"] = r.loc[
            r.policy_id.eq(report.BASELINE), "score"
        ].to_numpy()
    elif bad == "action_float32":
        a["score"] = a.score.astype(np.float32)
    elif bad == "threshold_declaration":
        new[name]["threshold_specification"]["value_hex"] = (0.8).hex()
    elif bad == "threshold_duplicate":
        a.loc[a.policy_id.eq(name), "threshold_specification"] = (
            '{"kind":"finite","kind":"finite","comparison":"strict_gt","value_hex":"0x0.0p+0"}'
        )
    elif bad == "acted_integer":
        a["acted"] = a.acted.astype(int)
    elif bad == "acted_mismatch":
        a.loc[ai, "acted"] = not a.loc[ai, "acted"]
    elif bad == "final_pred":
        a.loc[ai, "final_pred"] = 17
    elif bad == "expert_pred":
        a.loc[ai, "raw_pred"] = 17
    elif bad == "old_threshold":
        a.loc[
            a.policy_id.eq("capacity__rf__primary_full727"), "threshold_specification"
        ] = json.dumps(
            {"kind": "finite", "comparison": "strict_gt", "value_hex": (0.5).hex()}
        )
    elif bad == "old_heuristic_score":
        a.loc[a.policy_id.eq("geo_minus_raw_margin"), "score"] = -1.0
    elif bad == "label_ontology":
        labels.loc[0, "canonical_l3_id"] = 19
    elif bad == "missing_label":
        labels = labels.iloc[1:]
    elif bad == "label_plot":
        labels.loc[0, "plot_idx"] = "other"
    elif bad == "missing_seed":
        r = r.loc[r.training_seed.ne(4)]
    elif bad == "noncanonical":
        e = e.iloc[::-1]
    with pytest.raises(report.RelaxedRFReportError):
        report.evaluate_family(e, r, a, labels, old, new)


def test_no_files_or_selection_touched_and_inputs_unmodified(scored, monkeypatch):
    import pyarrow.parquet as pq
    from multimodal import geo_helpfulness_router_heuristics as heuristics

    args = deepcopy(scored[0])
    before = deepcopy(args)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure report attempted selection or file access")

    for module, attribute in (
        (pd, "read_parquet"),
        (pq, "read_table"),
        (Path, "open"),
        (heuristics, "enumerate_threshold_candidates"),
    ):
        monkeypatch.setattr(module, attribute, forbidden)
    result = report.evaluate_family(*args)
    assert result["summary.json"] == scored[1]["summary.json"]
    for actual, expected in zip(args, before):
        if isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            assert actual == expected


def test_import_cannot_load_training_libraries_or_development_selectors():
    code = """
import importlib.abc
import sys
forbidden = {
    'multimodal.geo_helpfulness_router_selection',
    'multimodal.geo_helpfulness_router_capacity_score',
    'multimodal.geo_helpfulness_router_mlp_score',
    'multimodal.geo_helpfulness_relaxed_guardrails_selection',
    'multimodal.geo_helpfulness_relaxed_rf_selection',
}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn', 'xgboost'} or fullname in forbidden:
            raise AssertionError('forbidden report import: ' + fullname)
sys.meta_path.insert(0, Guard())
from multimodal.geo_helpfulness_relaxed_rf_report import evaluate_family, validate_new_policies
assert not {'torch', 'scipy', 'sklearn', 'xgboost'}.intersection(sys.modules)
assert not forbidden.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

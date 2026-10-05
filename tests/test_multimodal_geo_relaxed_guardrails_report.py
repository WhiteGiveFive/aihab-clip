"""Synthetic-only relaxed threshold replay, frozen baseline parity and boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_relaxed_guardrails_report as report
from multimodal import geo_helpfulness_legacy_router_report as legacy
from test_multimodal_geo_legacy_router_report import _inputs as legacy_inputs


def inputs(family_shift=0):
    experts, routers, old_actions, labels = legacy_inputs(family_shift)
    for kind, frame in (
        ("experts", experts),
        ("routers", routers),
        ("actions", old_actions),
    ):
        frame.insert(0, "study_id", "legacy_frozen_router_transfer_v1")
        frame.insert(0, "schema_version", f"geo_helpfulness.legacy_transfer_{kind}.v1")
    declarations, actions = {}, [old_actions]
    for name in report.NEW_POLICIES:
        variant, minimum, numerator = report._policy_identity(name)
        router = routers.loc[routers.variant_id.eq(variant)]
        if numerator == 9:
            specification = deepcopy(report.NEVER_THRESHOLD)
        else:
            threshold = 0.0 if numerator == 5 else router.score.max()
            specification = {
                "kind": "finite",
                "comparison": "strict_gt",
                "value_hex": float(threshold).hex(),
            }
        declarations[name] = {
            "policy_id": name,
            "variant_id": variant,
            "guardrail_id": f"plots_{minimum}__harm_0p{numerator}",
            "minimum_plots": minimum,
            "harm_numerator": numerator,
            "harm_denominator": 10,
            "minimum_coverage_fraction": 0.01,
            "selected_C": 0.01,
            "selected_C_hex": (0.01).hex(),
            "threshold_specification": specification,
            "models": {"1": {"path": "synthetic-must-not-open.json"}},
        }
        frame = old_actions.loc[old_actions.policy_id.eq(variant)].copy()
        frame["policy_id"], frame["study_id"] = name, report.STUDY_ID
        frame["schema_version"] = "geo_helpfulness.relaxed_guardrails_actions.v1"
        frame["threshold_specification"] = json.dumps(specification, sort_keys=True)
        acted = legacy.apply_heuristic_threshold(
            frame.score.to_numpy(),
            experts.raw_pred.to_numpy(),
            experts.geo_pred.to_numpy(),
            specification,
        )
        frame["acted"], frame["final_pred"] = acted, np.where(
            acted, experts.geo_pred, experts.raw_pred
        )
        actions.append(frame)
    return (
        experts,
        routers,
        pd.concat(actions, ignore_index=True),
        labels,
        declarations,
    ), (experts, routers, old_actions, labels)


@pytest.fixture(scope="module")
def scored():
    args, old = inputs()
    return args, report.evaluate_family(*args), legacy.evaluate_family(*old)


def test_artifact_and_policy_contract(scored):
    _, result, _ = scored
    assert set(result) == set(report.SCORE_FILES)
    assert len(result["router_metrics.json"]) == 3
    assert len(result["policy_metrics.json"]) == 35
    assert len(result["classification.json"]["policies"]) == 35
    assert set(result["summary.json"]["groups"]) == set(report.NEW_POLICIES)
    summary = result["summary.json"]
    assert (summary["row_count"], summary["image_count"], summary["plot_count"]) == (
        32,
        8,
        4,
    )
    assert (
        summary["router_count"],
        summary["original_policy_count"],
        summary["new_policy_count"],
        summary["policy_count"],
    ) == (3, 8, 27, 35)
    assert summary["study_id"] == report.STUDY_ID == "relaxed_guardrails_lr_v1"
    assert summary["checkpoint_family"] == "best_model"
    for name in (
        "M4_gate_rerun",
        "selection_or_fitting_performed",
        "probability_calibration_fitted",
        "checkpoint_family_selection_performed",
        "independent_confirmation",
        "test_guardrail_feasibility_enforced",
    ):
        assert summary[name] is False
    assert summary["original_M4_no_go_preserved"] is True
    for filename, artifact in result.items():
        if filename.endswith(".json"):
            json.dumps(artifact, allow_nan=False)
        else:
            assert isinstance(artifact, pa.Table)
            assert artifact.schema.metadata is None


def test_all_original_numerical_payloads_and_rows_remain_exact(scored):
    _, result, old = scored
    for filename in ("expert_metrics.json", "router_metrics.json"):
        assert result[filename] == old[filename]
    assert (
        result["classification.json"]["experts"]
        == old["classification.json"]["experts"]
    )
    for name in report.OLD_POLICIES:
        assert result["policy_metrics.json"][name] == old["policy_metrics.json"][name]
        assert (
            result["classification.json"]["policies"][name]
            == old["classification.json"]["policies"][name]
        )
        assert (
            result["bootstrap_summary.json"]["policies"][name]
            == old["bootstrap_summary.json"]["policies"][name]
        )
        if name != report.BASELINE:
            field = "paired_primary_minus_comparator_delta_accuracy"
            assert (
                result["bootstrap_summary.json"][field][name]
                == old["bootstrap_summary.json"][field][name]
            )
    assert result["scoring_targets.parquet"].equals(old["scoring_targets.parquet"])
    for filename, column in (
        ("bootstrap_replicates.parquet", "policy"),
        ("subgroup_metrics.parquet", "policy"),
        ("actions_with_outcomes.parquet", "policy_id"),
    ):
        frame = result[filename].to_pandas()
        selected = frame.loc[frame[column].isin(report.OLD_POLICIES)].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(selected, old[filename].to_pandas())
    outcomes = result["actions_with_outcomes.parquet"].to_pandas()
    assert (
        outcomes.loc[outcomes.policy_id.isin(report.OLD_POLICIES), "study_id"]
        .eq("legacy_frozen_router_transfer_v1")
        .all()
    )
    assert (
        outcomes.loc[outcomes.policy_id.isin(report.NEW_POLICIES), "study_id"]
        .eq(report.STUDY_ID)
        .all()
    )


def test_new_policy_counts_classification_top3_and_raw_fallback(scored):
    (experts, _, _, _, declarations), result, _ = scored
    targets = result["scoring_targets.parquet"].to_pandas()
    outcomes = result["actions_with_outcomes.parquet"].to_pandas()
    y = targets.label_id_dense.to_numpy()
    for name in report.NEW_POLICIES:
        frame = outcomes.loc[outcomes.policy_id.eq(name)]
        acted = frame.acted.to_numpy()
        p = np.where(
            acted[:, None],
            np.stack(experts.geo_prob_calibrated),
            np.stack(experts.raw_prob_calibrated),
        )
        top3 = (np.argsort(-p, axis=1, kind="stable")[:, :3] == y[:, None]).any(1)
        actual = result["classification.json"]["policies"][name]
        assert actual["pooled_seed_realizations"]["top3_accuracy"] == top3.mean()
        assert (
            actual["pooled_seed_realizations"]["top1_accuracy"]
            == (frame.final_pred.to_numpy() == y).mean()
        )
        assert len(actual["pooled_seed_realizations"]["per_class"]) == 18
        assert set(actual["per_seed"]) == {"1", "2", "3", "4"}
        policy = result["policy_metrics.json"][name]
        reference = legacy.policy_metrics(targets, acted)
        for field in (
            "pooled",
            "per_seed",
            "mean_seed_net_utility",
            "mean_seed_delta_accuracy",
        ):
            assert policy[field] == reference[field]
        np.testing.assert_array_equal(
            frame.net_correct,
            frame.final_correct.astype(int) - frame.raw_correct.astype(int),
        )
        if declarations[name]["harm_numerator"] in (7, 9):
            assert actual == result["classification.json"]["experts"]["raw"]
            assert policy["pooled"]["acted_count"] == 0
            assert policy["all_predictions_equal_raw"] is True
            assert policy["raw_fallback_selected"] == (
                declarations[name]["harm_numerator"] == 9
            )
    assert (
        result["classification.json"]["policies"][report.BASELINE]
        == result["classification.json"]["experts"]["raw"]
    )


def test_guardrails_are_not_enforced_against_evaluation_outcomes(scored):
    _, result, _ = scored
    name = "relaxed__primary_full727__plots_20__harm_0p5"
    record = result["summary.json"]["groups"][name]
    assert record["frozen_policy"]["minimum_plots"] == 20
    assert record["plot_diagnostics"]["actual_acted_plot_count"] == 4
    assert record["policy"]["pooled"]["acted_count"] > 0
    counts = record["policy"]["pooled"]
    assert 10 * counts["harmed_count"] > 5 * counts["rescued_count"]
    assert record["policy"]["test_guardrail_feasibility_enforced"] is False
    # Low support, excessive harm, and finite thresholds with zero coverage
    # must all remain reported as frozen policies, never rejected or replaced.
    empty = result["summary.json"]["groups"][
        "relaxed__primary_full727__plots_20__harm_0p7"
    ]
    assert empty["policy"]["threshold_specification"]["kind"] == "finite"
    assert empty["policy"]["pooled"]["coverage_fraction"] == 0


def test_bootstrap_old_rows_exact_and_twentyseven_shared_paired_intervals(scored):
    _, result, old = scored
    bootstrap = result["bootstrap_summary.json"]
    assert (bootstrap["replicates"], bootstrap["random_seed"]) == (2000, 20260826)
    assert (
        bootstrap["resample_multiplicity_sha256"]
        == old["bootstrap_summary.json"]["resample_multiplicity_sha256"]
    )
    assert bootstrap["retain_all_images_and_training_seeds"] is True
    assert bootstrap["selection_repeated_in_bootstrap"] is False
    frame = result["bootstrap_replicates.parquet"].to_pandas()
    assert len(frame) == 70000
    assert frame.groupby("replicate").size().eq(35).all()
    wide = frame.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    paired = result["paired_comparisons.json"]
    assert (
        paired["contrast_count"] == sum(map(len, paired["comparisons"].values())) == 27
    )
    assert paired["difference_sign"] == "named_policy_minus_matching_original_LR"
    assert "not simultaneous" in paired["uncertainty_disclosure"]
    nonconstant = False
    for name, comparators in paired["comparisons"].items():
        variant, _, _ = report._policy_identity(name)
        assert set(comparators) == {variant}
        interval = comparators[variant]
        draws = wide[name] - wide[variant]
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
            - result["policy_metrics.json"][variant]["mean_seed_delta_accuracy"]
        )
        raw_interval = result["summary.json"]["groups"][name][
            "bootstrap_delta_accuracy"
        ]
        np.testing.assert_allclose(
            [raw_interval["lower"], raw_interval["upper"]],
            np.quantile(wide[name], [0.025, 0.975]),
            rtol=0,
            atol=1e-15,
        )
        assert (
            raw_interval["point_estimate"]
            == result["policy_metrics.json"][name]["mean_seed_delta_accuracy"]
        )
    assert nonconstant


def test_plot_diagnostics_match_actions_and_per_seed_counts(scored):
    _, result, _ = scored
    outcomes = result["actions_with_outcomes.parquet"].to_pandas()
    for name, record in result["summary.json"]["groups"].items():
        frame = outcomes.loc[outcomes.policy_id.eq(name)]
        plots = frame.groupby("plot_idx", sort=True).net_correct.sum()
        positive = plots.loc[plots.gt(0)]
        diagnostic = record["plot_diagnostics"]
        assert diagnostic["pooled_net_correct"] == frame.net_correct.sum()
        assert (
            diagnostic["actual_acted_plot_count"]
            == frame.loc[frame.acted, "plot_idx"].nunique()
        )
        assert diagnostic["per_seed_net_correct"] == {
            str(seed): int(value)
            for seed, value in frame.groupby("training_seed").net_correct.sum().items()
        }
        assert (
            sum(diagnostic["per_seed_net_correct"].values())
            == record["policy"]["pooled"]["rescued_minus_harmed"]
        )
        assert diagnostic["sum_positive_plot_net_correct"] == positive.sum()
        expected_best = int(positive.max()) if len(positive) else 0
        assert diagnostic["largest_positive_plot_net_correct"] == expected_best
        assert (
            diagnostic["leave_best_positive_plot_net_correct"]
            == int(plots.sum()) - expected_best
        )
        assert (
            sum(diagnostic["leave_best_positive_plot_per_seed_net_correct"].values())
            == diagnostic["leave_best_positive_plot_net_correct"]
        )
        assert diagnostic["plot_removal_used_for_selection_or_gate"] is False


@pytest.mark.parametrize("nets", [(3, 3, -7), (0, 0, -2), (0, 0, 0), (2, 1, -1)])
def test_plot_concentration_ties_nonpositive_total_and_empty_positive_support(nets):
    rows = []
    # One row per seed/plot; different positive/negative contributions within
    # the last plot may require multiple images, all retained in pooled net.
    for plot, net in zip(("b", "a", "c"), nets):
        for index in range(max(4, abs(net))):
            rows.append(
                {
                    "plot_idx": plot,
                    "training_seed": index % 4 + 1,
                    "target_state": (
                        ("rescue" if net > 0 else "harm")
                        if index < abs(net)
                        else "both_wrong"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    diagnostic = report._plot_diagnostics(frame, np.ones(len(frame), dtype=bool))
    maximum, positive_sum = max(0, *nets), sum(max(0, net) for net in nets)
    assert diagnostic["pooled_net_correct"] == sum(nets)
    assert diagnostic["largest_positive_plot_net_correct"] == maximum
    assert diagnostic["largest_positive_plot_share_of_positive_net"] == (
        maximum / positive_sum if positive_sum else None
    )
    assert diagnostic["leave_best_positive_plot_net_correct"] == sum(nets) - maximum
    if nets == (3, 3, -7):
        assert diagnostic["largest_positive_plot_idx"] == "a"
        assert diagnostic["largest_positive_plot_share_of_positive_net"] == 0.5
        assert diagnostic["leave_best_positive_plot_net_correct"] == -4
    if not positive_sum:
        assert diagnostic["largest_positive_plot_idx"] is None
        assert (
            diagnostic["leave_best_positive_plot_per_seed_net_correct"]
            == diagnostic["per_seed_net_correct"]
        )


def test_target_family_and_seeds_remain_independent(scored):
    _, first, _ = scored
    args, old_args = inputs(family_shift=1)
    second = report.evaluate_family(*args)
    old = legacy.evaluate_family(*old_args)
    left, right = (
        first["scoring_targets.parquet"].to_pandas(),
        second["scoring_targets.parquet"].to_pandas(),
    )
    assert left[list(report.KEY)].equals(right[list(report.KEY)])
    assert left.groupby("row_uid").target_state.nunique().eq(4).all()
    assert (left.target_state != right.target_state).all()
    assert second["summary.json"]["oracle_checkpoint_test_selection_biased"] is True
    assert (
        "test-selection-biased"
        in second["summary.json"]["checkpoint_selection_disclosure"]
    )
    assert "not guaranteed" in second["summary.json"]["calibration_disclosure"]
    assert (
        first["bootstrap_summary.json"]["resample_multiplicity_sha256"]
        == second["bootstrap_summary.json"]["resample_multiplicity_sha256"]
    )
    assert second["expert_metrics.json"] == old["expert_metrics.json"]
    assert second["router_metrics.json"] == old["router_metrics.json"]
    for name in report.OLD_POLICIES:
        assert second["policy_metrics.json"][name] == old["policy_metrics.json"][name]


@pytest.mark.parametrize(
    "bad",
    [
        "missing_declaration",
        "extra_declaration",
        "policy_identity",
        "variant_identity",
        "guardrail_identity",
        "minimum_plots",
        "minimum_plots_bool",
        "harm_numerator",
        "harm_denominator",
        "coverage",
        "selected_C",
        "selected_C_bool",
        "C_hex",
        "router_C",
        "nonfinite_metadata",
        "threshold_declaration",
        "threshold_duplicate_json",
        "missing_router",
        "duplicate_router",
        "missing_action",
        "duplicate_action",
        "extra_action",
        "action_schema",
        "new_study_id",
        "missing_schema_version",
        "expert_label",
        "expert_uppercase_label",
        "expert_ground_truth",
        "router_target",
        "router_true_probability",
        "action_outcome",
        "missing_probability",
        "wrong_family",
        "missing_family",
        "wrong_plot",
        "q_float32",
        "q_sum",
        "q_nan",
        "q_score",
        "action_score",
        "action_float32",
        "acted_integer",
        "acted_mismatch",
        "final_prediction",
        "expert_prediction",
        "primary_no_go",
        "old_heuristic_score",
        "label_ontology",
        "missing_label",
        "label_plot",
        "missing_seed",
        "noncanonical_rows",
    ],
)
def test_invalid_or_leaky_inputs_fail_closed(scored, bad):
    e, r, a, labels, declarations = deepcopy(scored[0])
    name = report.NEW_POLICIES[0]
    ai, ri = a.index[a.policy_id.eq(name)][0], r.index[0]
    if bad == "missing_declaration":
        declarations.pop(name)
    elif bad == "extra_declaration":
        declarations["unexpected"] = deepcopy(declarations[name])
    elif bad in {
        "policy_identity",
        "variant_identity",
        "guardrail_identity",
        "minimum_plots",
        "minimum_plots_bool",
        "harm_numerator",
        "harm_denominator",
        "coverage",
        "selected_C",
        "selected_C_bool",
        "C_hex",
        "nonfinite_metadata",
    }:
        key, value = {
            "policy_identity": ("policy_id", "wrong"),
            "variant_identity": ("variant_id", report.VARIANTS[1]),
            "guardrail_identity": ("guardrail_id", "wrong"),
            "minimum_plots": ("minimum_plots", 10),
            "minimum_plots_bool": ("minimum_plots", True),
            "harm_numerator": ("harm_numerator", 9),
            "harm_denominator": ("harm_denominator", 100),
            "coverage": ("minimum_coverage_fraction", 0.0),
            "selected_C": ("selected_C", 1.0),
            "selected_C_bool": ("selected_C", True),
            "C_hex": ("selected_C_hex", (1.0).hex()),
            "nonfinite_metadata": ("unused", float("nan")),
        }[bad]
        declarations[name][key] = value
    elif bad == "router_C":
        r.loc[r.variant_id.eq(report.BASELINE), "C_hex"] = (1.0).hex()
    elif bad == "threshold_declaration":
        declarations[name]["threshold_specification"]["value_hex"] = (0.9).hex()
    elif bad == "threshold_duplicate_json":
        a.loc[a.policy_id.eq(name), "threshold_specification"] = (
            '{"kind":"finite","kind":"finite","comparison":"strict_gt","value_hex":"0x0.0p+0"}'
        )
    elif bad == "missing_router":
        r = r.loc[r.variant_id.ne(report.BASELINE)]
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
        a["extra"] = "not-in-the-13-column-contract"
    elif bad == "new_study_id":
        a.loc[ai, "study_id"] = "wrong"
    elif bad == "missing_schema_version":
        a = a.drop(columns="schema_version")
    elif bad == "expert_label":
        e["label_id_dense"] = 0
    elif bad == "expert_uppercase_label":
        e["Label"] = 0
    elif bad == "expert_ground_truth":
        e["ground_truth_probability"] = 0.5
    elif bad == "router_target":
        r["target_state"] = "rescue"
    elif bad == "router_true_probability":
        r["raw_true_probability"] = 0.5
    elif bad == "action_outcome":
        a["rescued"] = False
    elif bad == "missing_probability":
        e = e.drop(columns="raw_prob_calibrated")
    elif bad == "wrong_family":
        a.loc[ai, "checkpoint_family"] = "oracle_test_best_model"
    elif bad == "missing_family":
        e = e.drop(columns="checkpoint_family")
    elif bad == "wrong_plot":
        a.loc[ai, "plot_idx"] = "other"
    elif bad == "q_float32":
        r["q_rescue"] = r.q_rescue.astype(np.float32)
    elif bad == "q_sum":
        r.loc[ri, "q_rescue"] += 0.2
    elif bad == "q_nan":
        r.loc[ri, "q_rescue"] = float("nan")
    elif bad == "q_score":
        r.loc[ri, "score"] = 0.333
    elif bad == "action_score":
        a.loc[ai, "score"] = 0.333
    elif bad == "action_float32":
        a["score"] = a.score.astype(np.float32)
    elif bad == "acted_integer":
        a["acted"] = a.acted.astype(np.int64)
    elif bad == "acted_mismatch":
        a.loc[ai, "acted"] = not a.loc[ai, "acted"]
    elif bad == "final_prediction":
        a.loc[ai, "final_pred"] = 17
    elif bad == "expert_prediction":
        a.loc[ai, "geo_pred"] = 17
    elif bad == "primary_no_go":
        primary = a.policy_id.eq(report.BASELINE)
        a.loc[primary, "threshold_specification"] = json.dumps(
            {"kind": "finite", "comparison": "strict_gt", "value_hex": (1.0).hex()}
        )
    elif bad == "old_heuristic_score":
        a.loc[a.policy_id.eq("geo_minus_raw_margin"), "score"] = -1.0
    elif bad == "label_ontology":
        labels.loc[0, "canonical_l3_id"] = 19
    elif bad == "missing_label":
        labels = labels.iloc[1:]
    elif bad == "label_plot":
        labels.loc[0, "plot_idx"] = "wrong"
    elif bad == "missing_seed":
        r = r.loc[r.training_seed.ne(4)]
    elif bad == "noncanonical_rows":
        e = e.iloc[::-1]
    with pytest.raises(report.RelaxedGuardrailsReportError):
        report.evaluate_family(e, r, a, labels, declarations)


def test_no_file_access_or_selection_or_input_mutation(scored, monkeypatch):
    import pyarrow.parquet as pq
    from multimodal import geo_helpfulness_router_heuristics as heuristics

    args = deepcopy(scored[0])
    before = deepcopy(args)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure reporter attempted file access or selection")

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


def test_import_cannot_load_selection_or_fitting_libraries():
    code = """
import importlib.abc
import sys
forbidden = {
    'multimodal.geo_helpfulness_router_selection',
    'multimodal.geo_helpfulness_router_capacity_score',
    'multimodal.geo_helpfulness_router_mlp_score',
    'multimodal.geo_helpfulness_relaxed_guardrails_selection',
}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn', 'xgboost'} or fullname in forbidden:
            raise AssertionError('forbidden report import: ' + fullname)
sys.meta_path.insert(0, Guard())
from multimodal.geo_helpfulness_relaxed_guardrails_report import evaluate_family
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

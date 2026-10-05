"""Synthetic-only tests for sealed MLP scoring and frozen capacity comparisons."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_router_capacity_score as capacity
from multimodal import geo_helpfulness_router_mlp_score as score
from test_multimodal_geo_router_capacity_score import inputs as capacity_inputs


@pytest.fixture(scope="module")
def prepared():
    predictions, experts, targets, configurations, baseline, specs = capacity_inputs()
    previous = capacity.evaluate_capacity(
        predictions, experts, targets, configurations, baseline, specs
    )
    widths = ([32], [64], [64, 32], [128, 64])
    mlp_configurations, mlp_predictions = [], []
    for variant in score.VARIANTS:
        for index, hidden_widths in enumerate(widths):
            original_id = f"lr__{variant}__setting{index}"
            config_id = f"mlp__{variant}__setting{index}"
            mlp_configurations.append(
                {
                    "config_id": config_id,
                    "family": "mlp",
                    "variant": variant,
                    "parameters": {"hidden_widths": list(hidden_widths)},
                }
            )
            frame = predictions.loc[predictions.config_id.eq(original_id)].copy()
            frame["config_id"], frame["family"] = config_id, "mlp"
            mlp_predictions.append(frame)
    arguments = (
        pd.concat(mlp_predictions, ignore_index=True),
        experts,
        targets,
        mlp_configurations,
        previous["actions.parquet"].to_pandas(),
        specs,
    )
    return arguments, previous


@pytest.fixture(scope="module")
def evaluated(prepared):
    arguments, _ = prepared
    return score.evaluate_mlp(*arguments)


def test_three_architecture_selections_reuse_exact_threshold_rules(prepared, evaluated):
    predictions, _, targets, configurations, _, _ = prepared[0]
    selections = evaluated["selection.json"]["groups"]
    assert len(selections) == 3
    assert len(evaluated["candidate_metrics.json"]) == 12
    for group_id, selection in selections.items():
        assert selection["selected_config_id"] == group_id + "__setting0"
        assert selection["selected_policy_id"] == group_id
        assert selection["tied_config_ids"] == [
            group_id + "__setting0",
            group_id + "__setting1",
        ]
        assert selection["tie_absolute_tolerance"] == 1e-12
        rows = predictions.loc[
            predictions.config_id.eq(selection["selected_config_id"])
        ]
        expected = capacity.select_common_threshold(
            targets.assign(score=rows.q_rescue.to_numpy() - rows.q_harm.to_numpy())
        )
        assert selection["threshold"] == expected
        assert (
            evaluated["policy_metrics.json"][group_id]["pooled"]["rescued_count"] == 48
        )
    assert {
        tuple(record["parameters"]["hidden_widths"]) for record in configurations
    } == {(32,), (64,), (64, 32), (128, 64)}


def test_all_fifteen_previous_policies_and_outputs_remain_exact(prepared, evaluated):
    _, previous = prepared
    assert len(evaluated["policy_metrics.json"]) == 18
    actions = evaluated["actions.parquet"].to_pandas()
    original_actions = previous["actions.parquet"].to_pandas()
    preserved = actions.loc[actions.policy_id.isin(score.FROZEN_POLICIES)].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(preserved, original_actions)
    for name in score.FROZEN_POLICIES:
        assert (
            evaluated["policy_metrics.json"][name]
            == previous["policy_metrics.json"][name]
        )
        assert (
            evaluated["classification.json"]["policies"][name]
            == previous["classification.json"]["policies"][name]
        )
    classifier = evaluated["classification.json"]
    assert classifier["policies"][score.BASELINE] == classifier["experts"]["raw"]
    assert (
        evaluated["policy_metrics.json"][score.BASELINE]["threshold_specification"]
        == capacity.NEVER_THRESHOLD
    )
    assert evaluated["summary.json"]["original_M4_no_go_preserved"] is True
    assert evaluated["summary.json"]["capacity_baseline_policies_reselected"] is False


def test_router_metrics_match_manual_four_state_math(prepared, evaluated):
    predictions, _, targets, configurations, _, _ = prepared[0]
    name = configurations[0]["config_id"]
    q = predictions.loc[
        predictions.config_id.eq(name), list(score.Q_COLUMNS)
    ].to_numpy()
    y = np.asarray([score.TARGET_ORDER.index(state) for state in targets.target_state])
    for scope, mask in (
        ("all_rows", np.ones(len(targets), dtype=bool)),
        ("disagreement", targets.geo_pred.ne(targets.raw_pred).to_numpy()),
    ):
        metric = evaluated["candidate_metrics.json"][name][scope]
        pooled = metric["pooled_seed_realizations"]
        assert pooled["log_loss"] == pytest.approx(-np.log(q[mask, y[mask]]).mean())
        assert pooled["brier_score"] == pytest.approx(
            np.square(q[mask] - np.eye(4)[y[mask]]).sum(axis=1).mean()
        )
        assert pooled["top1_accuracy"] == 1.0
        assert pooled["rescue_pr_auc"] == pooled["harm_pr_auc"] == 1.0
        assert pooled["macro_f1"] == (1.0 if scope == "all_rows" else 0.75)
        assert set(metric["per_seed"]) == {"1", "2", "3", "4"}


def test_selected_expert_probabilities_define_top3_and_action_outcomes(
    prepared, evaluated
):
    _, experts, targets, _, _, _ = prepared[0]
    all_actions = evaluated["actions.parquet"].to_pandas()
    for name in ("mlp__" + variant for variant in score.VARIANTS):
        frame = all_actions.loc[all_actions.policy_id.eq(name)]
        p = np.where(
            frame.acted.to_numpy()[:, None],
            np.stack(experts.geo_prob_calibrated),
            np.stack(experts.raw_prob_calibrated),
        )
        top3 = (
            np.argsort(-p, axis=1, kind="stable")[:, :3]
            == targets.label_id_dense.to_numpy()[:, None]
        ).any(axis=1)
        classifier = evaluated["classification.json"]["policies"][name][
            "pooled_seed_realizations"
        ]
        assert classifier["top3_accuracy"] == top3.mean()
        assert len(classifier["per_class"]) == 18
        np.testing.assert_array_equal(
            frame.net_correct,
            frame.final_correct.astype(int) - frame.raw_correct.astype(int),
        )


def test_shared_draws_exactly_preserve_previous_bootstrap_and_paired_sign(
    prepared, evaluated
):
    _, previous = prepared
    bootstrap = evaluated["bootstrap_summary.json"]
    assert (bootstrap["replicates"], bootstrap["random_seed"]) == (2000, 20260826)
    assert (
        bootstrap["image_count"],
        bootstrap["plot_count"],
        bootstrap["row_count"],
    ) == (48, 24, 192)
    assert bootstrap["primary_policy"] == score.BASELINE
    assert bootstrap["selection_repeated_in_bootstrap"] is False
    assert bootstrap["independent_confirmation"] is False
    replicates = evaluated["bootstrap_replicates.parquet"].to_pandas()
    assert len(replicates) == 36000
    old = replicates.loc[replicates.policy.isin(score.FROZEN_POLICIES)].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        old, previous["bootstrap_replicates.parquet"].to_pandas()
    )
    wide = replicates.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    paired = evaluated["paired_comparisons.json"]
    assert paired["difference_sign"] == "MLP_minus_frozen_comparator"
    assert sum(map(len, paired["comparisons"].values())) == 12
    for name, comparisons in paired["comparisons"].items():
        for comparator, interval in comparisons.items():
            differences = (wide[name] - wide[comparator]).to_numpy()
            np.testing.assert_allclose(
                [interval["lower"], interval["upper"]],
                np.quantile(differences, [0.025, 0.975]),
                atol=1e-15,
            )
            assert interval["point_estimate"] == (
                0.25 if comparator == "geo_minus_raw_margin" else 0
            )
            assert interval["valid_replicates"] == 2000


def test_result_contract_is_json_or_metadata_free_arrow(evaluated):
    assert set(evaluated) == set(score.SCORE_FILES)
    summary = evaluated["summary.json"]
    assert (
        summary["configuration_count"],
        summary["group_count"],
        summary["policy_count"],
        summary["frozen_baseline_policy_count"],
    ) == (12, 3, 18, 15)
    assert summary["test_data_read_by_scorer"] is False
    assert summary["mlp_probability_calibration_fitted"] is False
    for name, value in evaluated.items():
        if name.endswith(".json"):
            json.dumps(value, allow_nan=False)
        else:
            assert isinstance(value, pa.Table)
            assert value.schema.metadata is None


@pytest.mark.parametrize(
    "bad",
    [
        "missing_config",
        "duplicate_config",
        "family",
        "width_bool",
        "width_zero",
        "different_architecture_order",
        "prediction_label",
        "expert_label",
        "missing_seed",
        "duplicate_prediction",
        "wrong_plot",
        "float32",
        "q_sum",
        "utility_score",
        "state",
        "ontology",
        "nondevelopment",
        "missing_baseline",
        "extra_baseline",
        "baseline_target",
        "baseline_final",
        "baseline_action",
        "baseline_outcome",
        "baseline_score",
        "baseline_threshold",
        "heuristic_changed",
        "original_no_go",
    ],
)
def test_invalid_inputs_fail_closed(prepared, bad):
    p, e, t, configs, baseline, specs = deepcopy(prepared[0])
    if bad == "missing_config":
        configs.pop()
    elif bad == "duplicate_config":
        configs[-1] = deepcopy(configs[0])
    elif bad == "family":
        configs[0]["family"] = "lr"
    elif bad == "width_bool":
        configs[0]["parameters"]["hidden_widths"] = [True]
    elif bad == "width_zero":
        configs[0]["parameters"]["hidden_widths"] = [0]
    elif bad == "different_architecture_order":
        configs[4], configs[5] = configs[5], configs[4]
    elif bad == "prediction_label":
        p["label_id_dense"] = 0
    elif bad == "expert_label":
        e["target_state"] = "rescue"
    elif bad == "missing_seed":
        p = p.loc[p.training_seed.ne(4)]
    elif bad == "duplicate_prediction":
        p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    elif bad == "wrong_plot":
        p.loc[0, "plot_idx"] = "other"
    elif bad == "float32":
        p["q_rescue"] = p.q_rescue.astype(np.float32)
    elif bad == "q_sum":
        p.loc[0, "q_rescue"] += 0.1
    elif bad == "utility_score":
        p["score"] = 0.0
    elif bad == "state":
        t.loc[0, "target_state"] = "both_correct"
    elif bad == "ontology":
        t.loc[0, "canonical_l3_id"] = 19
    elif bad == "nondevelopment":
        t["development_role"] = "test"
    elif bad == "missing_baseline":
        baseline = baseline.loc[baseline.policy_id.ne("capacity__rf__primary_full727")]
    elif bad == "extra_baseline":
        baseline = pd.concat([baseline, baseline.iloc[[0]].assign(policy_id="extra")])
    elif bad == "baseline_target":
        baseline.loc[0, "target_state"] = "both_correct"
    elif bad == "baseline_final":
        baseline.loc[0, "final_pred"] = 17
    elif bad == "baseline_action":
        baseline.loc[0, "acted"] = not baseline.loc[0, "acted"]
    elif bad == "baseline_outcome":
        baseline.loc[0, "net_correct"] = 10
    elif bad == "baseline_score":
        baseline["score"] = baseline.score.astype(np.float32)
    elif bad == "baseline_threshold":
        baseline.loc[0, "threshold_specification"] = json.dumps(
            capacity.NEVER_THRESHOLD
        )
    elif bad == "heuristic_changed":
        baseline.loc[baseline.policy_id.eq("geo_minus_raw_margin"), "score"] = 1.0
    elif bad == "original_no_go":
        chosen = baseline.policy_id.eq(score.BASELINE)
        baseline.loc[chosen, "threshold_specification"] = json.dumps(
            capacity.ZERO_THRESHOLD
        )
    with pytest.raises(score.MLPScoreError):
        score.evaluate_mlp(p, e, t, configs, baseline, specs)


@pytest.mark.parametrize("improvement,winner", [(2e-13, 0), (1e-8, 1)])
def test_nll_tie_tolerance_uses_declared_simplicity(prepared, improvement, winner):
    p, e, t, configs, baseline, specs = deepcopy(prepared[0])
    target = np.asarray([score.TARGET_ORDER.index(state) for state in t.target_state])
    chosen = p.config_id.eq(configs[1]["config_id"])
    q = p.loc[chosen, list(score.Q_COLUMNS)].to_numpy().copy()
    q[np.arange(len(q)), target] += improvement
    q[np.arange(len(q)), (target + 1) % 4] -= improvement
    p.loc[chosen, list(score.Q_COLUMNS)] = q
    result = score.evaluate_mlp(p, e, t, configs, baseline, specs)
    selected = result["selection.json"]["groups"]["mlp__" + score.VARIANTS[0]]
    assert selected["selected_config_id"] == configs[winner]["config_id"]


def test_no_go_mlp_fallback_equals_raw_without_changing_frozen_baselines(prepared):
    p, e, t, configs, baseline, specs = deepcopy(prepared[0])
    p.loc[:, list(score.Q_COLUMNS)] = 0.25
    result = score.evaluate_mlp(p, e, t, configs, baseline, specs)
    for variant in score.VARIANTS:
        name = "mlp__" + variant
        assert result["policy_metrics.json"][name]["pooled"]["acted_count"] == 0
        assert (
            result["policy_metrics.json"][name]["threshold_specification"]
            == capacity.NEVER_THRESHOLD
        )
        assert (
            result["classification.json"]["policies"][name]
            == result["classification.json"]["experts"]["raw"]
        )


def test_input_order_independence_and_no_input_mutation(prepared, evaluated):
    arguments = deepcopy(prepared[0])
    before = deepcopy(arguments)
    p, e, t, configs, baseline, specs = arguments
    result = score.evaluate_mlp(
        p.iloc[::-1], e.iloc[::-1], t.iloc[::-1], configs, baseline.iloc[::-1], specs
    )
    assert result["summary.json"] == evaluated["summary.json"]
    assert result["selection.json"] == evaluated["selection.json"]
    for original, expected in zip(arguments, before):
        if isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(original, expected)
        else:
            assert original == expected


def test_probability_floor_and_state_ties_match_frozen_metrics(prepared):
    p, e, t, configs, baseline, specs = deepcopy(prepared[0])
    name = configs[0]["config_id"]
    selected = p.index[p.config_id.eq(name)]
    p.loc[selected[0], list(score.Q_COLUMNS)] = [0.0, 0.8, 0.1, 0.1]
    p.loc[selected[1], list(score.Q_COLUMNS)] = [0.25, 0.25, 0.25, 0.25]
    result = score.evaluate_mlp(p, e, t, configs, baseline, specs)
    q = p.loc[p.config_id.eq(name), list(score.Q_COLUMNS)].to_numpy()
    labels = np.asarray([score.TARGET_ORDER.index(state) for state in t.target_state])
    expected = capacity.multiclass_metrics(q, labels)
    actual = result["candidate_metrics.json"][name]["all_rows"][
        "pooled_seed_realizations"
    ]
    assert actual["true_probability_floor_count"] == 1
    assert (
        actual["true_probability_floor_hex"] == expected["true_probability_floor_hex"]
    )
    for metric in ("log_loss", "brier_score", "top1_accuracy", "top3_accuracy"):
        assert actual[metric] == expected[metric]
    assert actual["top1_accuracy"] == np.mean(q.argmax(axis=1) == labels)


def test_shared_plot_contrasts_retain_nonconstant_paired_draws(prepared):
    p, e, t, configs, baseline, specs = deepcopy(prepared[0])
    hidden = set(
        t.loc[t.row_uid.ge("image-044") & t.target_state.eq("rescue"), "row_uid"]
    )
    chosen = p.row_uid.isin(hidden) & p.q_rescue.gt(0.5)
    p.loc[chosen, list(score.Q_COLUMNS)] = [0.1, 0.2, 0.2, 0.5]
    result = score.evaluate_mlp(p, e, t, configs, baseline, specs)
    name = "mlp__primary_full727"
    comparator = "capacity__rf__primary_full727"
    replicates = (
        result["bootstrap_replicates.parquet"]
        .to_pandas()
        .pivot(index="replicate", columns="policy", values="mean_seed_delta_accuracy")
    )
    differences = replicates[name] - replicates[comparator]
    interval = result["paired_comparisons.json"]["comparisons"][name][comparator]
    assert differences.std() > 1e-4
    assert interval["point_estimate"] < 0
    np.testing.assert_allclose(
        [interval["lower"], interval["upper"]], np.quantile(differences, [0.025, 0.975])
    )


def test_import_avoids_fitting_dependencies_and_file_discovery():
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn', 'xgboost'}:
            raise AssertionError('forbidden MLP scoring import: ' + fullname)
sys.meta_path.insert(0, Guard())
from multimodal.geo_helpfulness_router_mlp_score import evaluate_mlp
assert not {'torch', 'scipy', 'sklearn', 'xgboost'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

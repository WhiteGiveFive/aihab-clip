"""Small synthetic tests for nine predeclared development capacity groups."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_router_capacity_score as score
from multimodal.geo_helpfulness_router_selection import select_common_threshold


def inputs(n=48):
    targets, experts = [], []
    for image in range(n):
        label = image // (n // 2)
        for seed in score.SEEDS:
            state = (image + seed - 1) % 4
            raw = label if state in (1, 2) else (label + 1) % 18
            geo = label if state in (0, 2) else (label + 2) % 18
            identity = {
                "row_uid": f"image-{image:03d}",
                "plot_idx": f"plot-{image // 2:03d}",
                "training_seed": seed,
            }
            targets.append(
                {
                    **identity,
                    "label_id_dense": label,
                    "canonical_l3_id": label,
                    "label_name": ["Urban", "Broadleaved Mixed and Yew Woodland"][
                        label
                    ],
                    "image_pred": label,
                    "raw_pred": raw,
                    "geo_pred": geo,
                    "target_state": score.TARGET_ORDER[state],
                }
            )
            expert = {
                **identity,
                "image_pred": label,
                "raw_pred": raw,
                "geo_pred": geo,
                "image_geo_jsd": 0.0,
                "geo_minus_raw_margin": 0.0,
            }
            for name, prediction in (("image", label), ("raw", raw), ("geo", geo)):
                p = np.full(18, 0.3 / 17, dtype=np.float64)
                p[prediction] = 0.7
                expert[f"{name}_prob_calibrated"] = p
            image_probability = expert["image_prob_calibrated"]
            geo_probability = expert["geo_prob_calibrated"]
            midpoint = 0.5 * (image_probability + geo_probability)
            expert["image_geo_jsd"] = 0.5 * np.sum(
                image_probability * np.log(image_probability / midpoint)
                + geo_probability * np.log(geo_probability / midpoint)
            )
            experts.append(expert)
    targets, experts = pd.DataFrame(targets), pd.DataFrame(experts)
    codes = np.asarray(
        [score.TARGET_ORDER.index(state) for state in targets.target_state]
    )
    quality = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.2, 0.5, 0.2, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.2, 0.1, 0.2, 0.5],
        ]
    )
    configurations, predictions = [], []
    for family in score.FAMILIES:
        for variant in score.VARIANTS:
            for number in range(4):
                config_id = f"{family}__{variant}__setting{number}"
                configurations.append(
                    {
                        "config_id": config_id,
                        "family": family,
                        "variant": variant,
                        "parameters": {"declared_setting": number},
                    }
                )
                frame = targets[list(score.IDENTITY)].copy()
                frame["config_id"], frame["family"], frame["variant"] = (
                    config_id,
                    family,
                    variant,
                )
                q = (
                    quality[codes].copy()
                    if number < 2
                    else np.full((len(frame), 4), 0.25)
                )
                if number == 3:
                    q = quality[codes][:, ::-1].copy()
                for index, name in enumerate(score.Q_COLUMNS):
                    frame[name] = q[:, index]
                predictions.append(frame)
    baseline = targets[list(score.IDENTITY)].copy()
    baseline["policy_id"], baseline["acted"] = score.BASELINE, False
    baseline["raw_pred"], baseline["geo_pred"], baseline["final_pred"] = (
        targets.raw_pred,
        targets.geo_pred,
        targets.raw_pred,
    )
    baseline["threshold_specification"] = json.dumps(score.NEVER_THRESHOLD)
    specs = {
        name: deepcopy(score.NEVER_THRESHOLD)
        for name in ("image_geo_jsd", "geo_minus_raw_margin")
    }
    return (
        pd.concat(predictions, ignore_index=True),
        experts,
        targets,
        configurations,
        baseline,
        specs,
    )


@pytest.fixture(scope="module")
def evaluated():
    arguments = inputs()
    return arguments, score.evaluate_capacity(*arguments)


def test_nine_groups_select_the_declared_simple_tie_and_exact_threshold(evaluated):
    arguments, result = evaluated
    predictions, _, targets, _, _, _ = arguments
    selections = result["selection.json"]["groups"]
    assert len(selections) == 9
    assert len(result["candidate_metrics.json"]) == 36
    for group_id, selection in selections.items():
        assert selection["selected_config_id"] == group_id + "__setting0"
        assert selection["tied_config_ids"] == [
            group_id + "__setting0",
            group_id + "__setting1",
        ]
        assert selection["tie_absolute_tolerance"] == 1e-12
        rows = predictions.loc[
            predictions.config_id.eq(selection["selected_config_id"])
        ]
        manual = targets.assign(score=rows.q_rescue.to_numpy() - rows.q_harm.to_numpy())
        assert selection["threshold"] == select_common_threshold(manual)
        policy = result["policy_metrics.json"][selection["selected_policy_id"]]
        assert policy["pooled"]["rescued_count"] == 48
        assert policy["pooled"]["harmed_count"] == 0
        assert policy["pooled"]["acted_count"] == 48
        assert policy["mean_seed_delta_accuracy"] == 0.25


def test_original_baseline_and_heuristics_are_preserved(evaluated):
    _, result = evaluated
    policies = result["policy_metrics.json"]
    assert len(policies) == 15
    assert set(score.HEURISTICS).issubset(policies)
    baseline = policies[score.BASELINE]
    assert baseline["pooled"]["acted_count"] == 0
    assert baseline["threshold_specification"] == score.NEVER_THRESHOLD
    classification = result["classification.json"]
    assert (
        classification["policies"][score.BASELINE] == classification["experts"]["raw"]
    )
    assert result["summary.json"]["original_M4_no_go_preserved"] is True
    assert result["summary.json"]["test_data_read_by_scorer"] is False


def test_candidate_metrics_all_and_disagreement_match_direct_math(evaluated):
    arguments, result = evaluated
    predictions, _, targets, configurations, _, _ = arguments
    candidate_id = configurations[0]["config_id"]
    frame = predictions.loc[predictions.config_id.eq(candidate_id)]
    q = frame[list(score.Q_COLUMNS)].to_numpy()
    codes = np.asarray(
        [score.TARGET_ORDER.index(state) for state in targets.target_state]
    )
    disagreement = targets.raw_pred.ne(targets.geo_pred).to_numpy()
    metric = result["candidate_metrics.json"][candidate_id]
    for name, mask in (
        ("all_rows", np.ones(len(targets), dtype=bool)),
        ("disagreement", disagreement),
    ):
        expected_nll = -np.log(q[mask, codes[mask]]).mean()
        pooled = metric[name]["pooled_seed_realizations"]
        assert pooled["top1_accuracy"] == 1.0
        assert pooled["macro_f1"] == (1.0 if name == "all_rows" else 0.75)
        assert pooled["log_loss"] == pytest.approx(expected_nll)
        assert pooled["rescue_pr_auc"] == 1.0
        assert pooled["harm_pr_auc"] == 1.0
        assert pooled["row_count"] == int(mask.sum())
        assert set(metric[name]["per_seed"]) == {"1", "2", "3", "4"}


def test_bootstrap_preserves_complete_seeded_plots_and_selection_scope(evaluated):
    _, result = evaluated
    bootstrap = result["bootstrap_summary.json"]
    assert (bootstrap["replicates"], bootstrap["random_seed"]) == (2000, 20260826)
    assert (
        bootstrap["image_count"],
        bootstrap["plot_count"],
        bootstrap["row_count"],
    ) == (48, 24, 192)
    assert bootstrap["primary_policy"] == score.BASELINE
    assert bootstrap["selection_repeated_in_bootstrap"] is False
    assert bootstrap["independent_confirmation"] is False
    replicates = result["bootstrap_replicates.parquet"].to_pandas()
    assert len(replicates) == 30000
    selected = replicates.policy.str.startswith("capacity__")
    np.testing.assert_allclose(
        replicates.loc[selected, "mean_seed_delta_accuracy"], 0.25, atol=1e-15, rtol=0
    )
    np.testing.assert_allclose(
        replicates.loc[selected, "primary_minus_policy_delta_accuracy"],
        -0.25,
        atol=1e-15,
        rtol=0,
    )
    for name, value in result.items():
        if name.endswith(".json"):
            json.dumps(value, allow_nan=False)
        else:
            assert isinstance(value, pa.Table)


def test_actions_and_top3_use_selected_expert_probabilities(evaluated):
    arguments, result = evaluated
    _, experts, targets, configurations, _, _ = arguments
    policy_id = (
        "capacity__" + configurations[0]["family"] + "__" + configurations[0]["variant"]
    )
    actions = result["actions.parquet"].to_pandas()
    frame = actions.loc[actions.policy_id.eq(policy_id)]
    y = targets.label_id_dense.to_numpy()
    p = np.where(
        frame.acted.to_numpy()[:, None],
        np.stack(experts.geo_prob_calibrated),
        np.stack(experts.raw_prob_calibrated),
    )
    top3 = np.any(np.argsort(-p, axis=1, kind="stable")[:, :3] == y[:, None], axis=1)
    classifier = result["classification.json"]["policies"][policy_id][
        "pooled_seed_realizations"
    ]
    assert classifier["top3_accuracy"] == top3.mean()
    assert len(classifier["per_class"]) == 18
    np.testing.assert_array_equal(
        frame.net_correct,
        frame.final_correct.astype(int) - frame.raw_correct.astype(int),
    )


@pytest.mark.parametrize(
    "bad",
    [
        "missing_config",
        "duplicate_config",
        "variant",
        "prediction_label",
        "expert_label",
        "missing_seed",
        "duplicate_prediction",
        "plot",
        "float32",
        "probability_sum",
        "state",
        "dense_canonical",
        "baseline_intervenes",
        "fixed_heuristic",
        "heuristic_values",
        "nondevelopment_targets",
    ],
)
def test_corrupt_or_misaligned_inputs_fail_closed(bad):
    p, e, t, configs, baseline, specs = inputs(n=8)
    if bad == "missing_config":
        configs.pop()
    elif bad == "duplicate_config":
        configs[-1] = deepcopy(configs[0])
    elif bad == "variant":
        p.loc[0, "variant"] = "unknown_variant"
    elif bad == "prediction_label":
        p["label_id_dense"] = 0
    elif bad == "expert_label":
        e["target_state"] = "rescue"
    elif bad == "missing_seed":
        p = p.loc[p.training_seed.ne(4)]
    elif bad == "duplicate_prediction":
        p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    elif bad == "plot":
        p.loc[0, "plot_idx"] = "wrong-plot"
    elif bad == "float32":
        p["q_rescue"] = p.q_rescue.astype(np.float32)
    elif bad == "probability_sum":
        p.loc[0, "q_rescue"] = 0.99
    elif bad == "state":
        t.loc[0, "target_state"] = "harm"
    elif bad == "dense_canonical":
        t.loc[0, "canonical_l3_id"] = 19
    elif bad == "baseline_intervenes":
        baseline.loc[0, "acted"] = True
    elif bad == "fixed_heuristic":
        specs = {
            **specs,
            **{name: score.ZERO_THRESHOLD for name in score.HEURISTICS[:3]},
        }
    elif bad == "heuristic_values":
        e.loc[0, "geo_minus_raw_margin"] = 0.1
    elif bad == "nondevelopment_targets":
        t["schema_version"] = "geo_helpfulness.legacy_router_scoring_targets.v1"
    with pytest.raises(ValueError):
        score.evaluate_capacity(p, e, t, configs, baseline, specs)


def test_empty_disagreement_is_reported_as_null_not_an_exception():
    p, e, t, configs, baseline, specs = inputs(n=8)
    t["raw_pred"], t["geo_pred"], t["target_state"] = (
        t.image_pred,
        t.image_pred,
        "both_correct",
    )
    e["raw_pred"], e["geo_pred"] = e.image_pred, e.image_pred
    e["raw_prob_calibrated"], e["geo_prob_calibrated"] = (
        e.image_prob_calibrated,
        e.image_prob_calibrated,
    )
    e["image_geo_jsd"] = 0.0
    baseline["raw_pred"], baseline["geo_pred"], baseline["final_pred"] = (
        t.raw_pred,
        t.geo_pred,
        t.raw_pred,
    )
    result = score.evaluate_capacity(p, e, t, configs, baseline, specs)
    for metric in result["candidate_metrics.json"].values():
        empty = metric["disagreement"]["pooled_seed_realizations"]
        assert empty["row_count"] == 0
        assert empty["log_loss"] is None
        assert empty["rescue_pr_auc"] is None
    assert all(
        record["pooled"]["acted_count"] == 0
        for record in result["policy_metrics.json"].values()
    )


def test_input_order_is_not_used_to_join_predictions(evaluated):
    arguments, expected = evaluated
    p, e, t, configurations, baseline, specs = arguments
    observed = score.evaluate_capacity(
        p.sample(frac=1, random_state=11),
        e.iloc[::-1],
        t.iloc[::-1],
        configurations,
        baseline.iloc[::-1],
        specs,
    )
    assert observed["summary.json"] == expected["summary.json"]
    assert observed["selection.json"] == expected["selection.json"]


@pytest.mark.parametrize("improvement,winner", [(2e-13, 0), (1e-8, 1)])
def test_nll_tolerance_preserves_simplicity_only_for_near_ties(improvement, winner):
    p, e, t, configurations, baseline, specs = inputs(n=8)
    target = np.asarray([score.TARGET_ORDER.index(state) for state in t.target_state])
    candidate_id = configurations[1]["config_id"]
    chosen = p.config_id.eq(candidate_id)
    q = p.loc[chosen, list(score.Q_COLUMNS)].to_numpy().copy()
    q[np.arange(len(q)), target] += improvement
    q[np.arange(len(q)), (target + 1) % 4] -= improvement
    p.loc[chosen, list(score.Q_COLUMNS)] = q
    result = score.evaluate_capacity(p, e, t, configurations, baseline, specs)
    group = configurations[0]["family"] + "__" + configurations[0]["variant"]
    selected = result["selection.json"]["groups"][group]
    assert selected["selected_config_id"] == configurations[winner]["config_id"]
    assert (
        selected["minimum_mean_seed_log_loss"]
        < selected["candidate_losses"][0]["mean_seed_log_loss"]
    )


def test_score_import_has_no_fitting_dependencies_or_file_discovery():
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn', 'xgboost'}:
            raise AssertionError('forbidden capacity scoring import: ' + fullname)
sys.meta_path.insert(0, Guard())
from multimodal.geo_helpfulness_router_capacity_score import evaluate_capacity
assert not {'torch', 'scipy', 'sklearn', 'xgboost'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

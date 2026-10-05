"""Cleaned-test scoring uses frozen policies and independently checked metrics."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from multimodal.geo_helpfulness_joint_tcp_report import Q_COLUMNS
from multimodal.geo_helpfulness_joint_tcp_test_report import (
    GUARDRAILS, JointTCPReportError, _validate, format_report, make_report,
    policy_actions,
)
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256


def fixture():
    items = []
    for base, width in (("numeric25", 25), ("no_pairs79", 79)):
        for embeddings in (False, True):
            for weight in (0, .1, 1, 10):
                input_id = base + ("_embeddings" if embeddings else "")
                items.append({
                    "config_id": f"{input_id}_lambda_{weight}", "input_id": input_id,
                    "variant": f"ablation_{base}", "with_embeddings": embeddings,
                    "auxiliary_weight": weight, "input_dim": width + 1216 * embeddings,
                })
    config = {"configs": items, "bootstrap": {"resamples": 17, "seed": 20260826}}
    raw_predictions = [1, 0, 0, 1, 1, 1]
    geo_predictions = [0, 1, 0, 2, 1, 0]
    probabilities = np.array([
        [.6, .2, .1, .1], [.5, .3, .1, .1], [.7, .1, .1, .1],
        [.2, .2, .1, .5], [.8, .1, .05, .05], [.1, .2, .1, .6],
    ], dtype=np.float32)
    rows, selections = [], {}
    for item in items:
        for seed in (1, 2, 3, 4):
            for index in range(6):
                raw, geo = np.zeros(18, dtype=np.float32), np.zeros(18, dtype=np.float32)
                raw[raw_predictions[index]], geo[geo_predictions[index]] = 3, 3
                q = probabilities[index].copy()
                if item["auxiliary_weight"] > 0 and index in (1, 5):
                    q = np.array([.1, .7, .1, .1] if index == 1 else [.7, .1, .1, .1], dtype=np.float32)
                estimate = np.array([[.2, .8, .8, .2, .4, .2][index],
                                     [.8, .2, .8, .3, .4, .8][index]], dtype=np.float32)
                rows.append({
                    **item, "split": "test", "training_seed": seed,
                    "row_uid": f"r{index}", "plot_idx": f"p{[0, 1, 1, 2, 2, 2][index]}",
                    "producer_fold": -1, "label_id_dense": 0,
                    "raw_logits": raw, "geo_logits": geo, "image_logits": raw.copy(),
                    "state_logits": np.log(q).astype(np.float32),
                    "tcp_logits": np.log(estimate / (1 - estimate)).astype(np.float32),
                    "fusion_tcp_hat": estimate[0], "geo_tcp_hat": estimate[1],
                    **dict(zip(Q_COLUMNS, q)),
                })
        for plots, harm in GUARDRAILS:
            guardrail = f"plots_{plots}__harm_0p{harm}"
            specification = {"kind": "finite", "value_hex": float(.1).hex(), "comparison": "strict_gt"}
            selections[f"{item['config_id']}__{guardrail}"] = {
                "config_id": item["config_id"], "threshold_specification": specification,
                "threshold_specification_sha256": canonical_sha256(specification),
                "selected_feasible": True,
                "guardrails": {
                    "guardrail_id": guardrail, "minimum_plots": plots,
                    "harm_numerator": harm, "harm_denominator": 10,
                    "minimum_coverage_fraction": .01, "require_positive_net_utility": True,
                },
            }
    return pd.DataFrame(rows), config, selections


@pytest.fixture(scope="module")
def scored():
    rows, config, selections = fixture()
    return rows, config, selections, make_report(rows, config, selections)


def test_all_models_policies_and_contrasts_are_retained(scored):
    _, _, _, output = scored
    report = output["report"]
    assert report["model_count"] == 64
    assert report["selected_policy_count"] == 144
    assert report["policy_count_including_references"] == 161
    assert output["bootstrap_draws"].shape == (17, 161, 2)
    contrasts = report["test_bootstrap"]["contrasts"].values()
    kinds = [item["kind"] for item in contrasts]
    assert kinds.count("versus_fusion") == 160
    assert kinds.count("auxiliary_weight_versus_zero") == 120
    assert kinds.count("embeddings_versus_no_embeddings") == 80
    assert report["lambda_winner_selected"] is False
    assert report["test_used_for_selection"] is False
    assert report["fitting_performed_by_scorer"] is False
    json.dumps(report, allow_nan=False)


def test_label_blind_actions_match_scoring_and_support_per_seed(scored):
    rows, config, selections, output = scored
    without_labels = rows.drop(columns="label_id_dense")
    actions = policy_actions(without_labels, config, selections)
    pd.testing.assert_frame_equal(actions, output["actions"].drop(columns=["label_id_dense", "target_state"]))
    for seed in (1, 2, 3, 4):
        single = policy_actions(without_labels.loc[without_labels.training_seed.eq(seed)], config, selections)
        pd.testing.assert_frame_equal(single, actions.loc[actions.training_seed.eq(seed)].reset_index(drop=True))
    with pytest.raises(JointTCPReportError, match="label fields"):
        policy_actions(rows, config, selections)


def test_scoring_does_not_change_actions_when_every_intervention_is_harmful(scored):
    rows, config, selections, output = scored
    changed = rows.copy()
    changed["label_id_dense"] = np.stack(rows.raw_logits).argmax(1)
    scored_harms = make_report(changed, config, selections)
    names = output["bootstrap_policy_order"]
    pd.testing.assert_frame_equal(output["actions"][names], scored_harms["actions"][names])
    entry = scored_harms["report"]["test_policies"]["numeric25_lambda_0__plots_20__harm_0p5"]
    assert entry["pooled"]["net"] < 0
    assert not entry["test_guardrail_diagnostics"]["harm_cap_met"]
    assert not entry["test_guardrail_diagnostics"]["policy_changed_using_test_labels"]


def test_threshold_equality_float64_subtraction_and_disagreement_gate():
    rows, config, selections = fixture()
    label_free = rows.drop(columns="label_id_dense")
    prepared = _validate(label_free, config, labels=False)
    frame = prepared["numeric25_lambda_0"]
    q = frame[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(frame.score, q[:, 0] - q[:, 1])
    native = (frame.q_rescue.to_numpy(dtype=np.float32)
              - frame.q_harm.to_numpy(dtype=np.float32)).astype(np.float64)
    assert np.any(frame.score.to_numpy() != native)
    policy = "numeric25_lambda_0__plots_20__harm_0p5"
    specification = {"kind": "finite", "value_hex": float(frame.score.iloc[0]).hex(),
                     "comparison": "strict_gt"}
    selections[policy]["threshold_specification"] = specification
    selections[policy]["threshold_specification_sha256"] = canonical_sha256(specification)
    actions = policy_actions(label_free, config, selections)
    assert not actions.loc[actions.row_uid.eq("r0"), policy].any()
    assert not actions.loc[actions.raw_pred.eq(actions.geo_pred), config["configs"][0]["config_id"] + "__zero"].any()


def test_never_intervene_fallback_is_kept_even_when_geo_would_rescue():
    rows, config, selections = fixture()
    name = next(iter(selections))
    specification = {"kind": "never_intervene", "value_hex": None, "comparison": "strict_gt"}
    selections[name].update(threshold_specification=specification,
                            threshold_specification_sha256=canonical_sha256(specification),
                            selected_feasible=False)
    result = policy_actions(rows.drop(columns="label_id_dense"), config, selections)
    assert not result[name].any()


def test_metrics_and_expert_baselines_match_sklearn(scored):
    _, _, _, output = scored
    frame = output["actions"]
    for name, item in output["report"]["test_policies"].items():
        action = frame[name].to_numpy()
        prediction = np.where(action, frame.geo_pred, frame.raw_pred)
        assert item["pooled"]["accuracy"] == np.mean(prediction == frame.label_id_dense)
        assert item["pooled"]["weighted_f1"] == pytest.approx(f1_score(frame.label_id_dense, prediction, average="weighted"))
        assert item["pooled"]["net"] == item["pooled"]["rescues"] - item["pooled"]["harms"]
        assert item["pooled"]["actions"] == item["pooled"]["rescues"] + item["pooled"]["harms"] + item["pooled"]["neutral_wrong"]
    for name, source in (("fusion", "raw"), ("image", "image"), ("geo", "geo")):
        metric = output["report"]["expert_references"][name]["mean_seed"]
        assert metric["accuracy"] == np.mean(frame[f"{source}_pred"] == frame.label_id_dense)
    baseline = output["report"]["diagnostics"]["numeric25_lambda_0"]["test"]["mean_seed"]
    assert baseline["tcp"]["fusion"]["supervised"] is False


def test_bootstrap_preserves_whole_plots_and_seed_pairing(scored):
    _, config, _, output = scored
    frame = output["actions"]
    plots = np.unique(frame.plot_idx)
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    names = output["bootstrap_policy_order"]
    for draw in range(config["bootstrap"]["resamples"]):
        indices = rng.integers(0, len(plots), size=len(plots))
        for name in ("fusion", "numeric25_lambda_0__zero", "numeric25_lambda_0.1__zero"):
            by_seed = []
            for seed in (1, 2, 3, 4):
                sample = pd.concat([frame.loc[frame.plot_idx.eq(plots[index]) & frame.training_seed.eq(seed)]
                                    for index in indices])
                prediction = np.where(sample[name], sample.geo_pred, sample.raw_pred)
                by_seed.append([np.mean(prediction == sample.label_id_dense),
                                f1_score(sample.label_id_dense, prediction, average="weighted")])
            np.testing.assert_allclose(output["bootstrap_draws"][draw, names.index(name)], np.mean(by_seed, axis=0))
    contrast = output["report"]["test_bootstrap"]["contrasts"][
        "numeric25_lambda_0.1__zero_minus_numeric25_lambda_0__zero"]
    delta = (output["bootstrap_draws"][:, names.index("numeric25_lambda_0.1__zero")]
             - output["bootstrap_draws"][:, names.index("numeric25_lambda_0__zero")])
    expected = np.quantile(delta, [.025, .975], axis=0)
    assert contrast["accuracy"]["ci95_lower"] == expected[0, 0]
    assert contrast["accuracy"]["ci95_upper"] == expected[1, 0]


def test_permutation_repeat_and_sorted_json_markdown_replay(scored):
    rows, config, selections, output = scored
    again = make_report(rows.sample(frac=1, random_state=13), config, deepcopy(selections))
    assert output["report"] == again["report"]
    np.testing.assert_array_equal(output["bootstrap_draws"], again["bootstrap_draws"])
    pd.testing.assert_frame_equal(output["actions"], again["actions"])
    restored = json.loads(json.dumps(output["report"], sort_keys=True, allow_nan=False))
    assert format_report(output["report"]) == format_report(restored)
    rendered = format_report(restored)
    assert "All 64 previously fitted routers" in rendered
    for name in output["bootstrap_policy_order"]:
        assert f"| {name} |" in rendered


@pytest.mark.parametrize("corruption", [
    "duplicate", "label", "producer", "expert_logits", "simplex", "tcp", "split",
    "input_id", "missing_seed", "missing_configuration", "extra_column", "nonfinite_logits",
])
def test_invalid_predictions_fail_before_scoring(corruption):
    rows, config, selections = fixture()
    if corruption == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    elif corruption == "label":
        rows.loc[0, "label_id_dense"] = 1
    elif corruption == "producer":
        rows.loc[0, "producer_fold"] = 0
    elif corruption == "expert_logits":
        changed = rows.loc[0, "raw_logits"].copy()
        changed[0] += 1
        rows.at[0, "raw_logits"] = changed
    elif corruption == "simplex":
        rows.loc[0, "q_rescue"] = 1
    elif corruption == "tcp":
        rows.loc[0, "fusion_tcp_hat"] = 1.1
    elif corruption == "split":
        rows.loc[0, "split"] = "validation"
    elif corruption == "input_id":
        rows.loc[0, "input_id"] = "unknown"
    elif corruption == "missing_seed":
        rows = rows.loc[rows.training_seed.ne(4)]
    elif corruption == "missing_configuration":
        rows = rows.loc[rows.config_id.ne(config["configs"][0]["config_id"])]
    elif corruption == "extra_column":
        rows["unsealed_target"] = 1
    else:
        rows.at[0, "state_logits"] = np.array([0, np.nan, 0, 0])
    with pytest.raises(JointTCPReportError):
        make_report(rows, config, selections)


@pytest.mark.parametrize("corruption", ["missing", "config", "guardrail", "hash", "comparison", "feasible"])
def test_frozen_threshold_identity_corruption_is_rejected(corruption):
    rows, config, selections = fixture()
    name = next(iter(selections))
    if corruption == "missing":
        selections.pop(name)
    elif corruption == "config":
        selections[name]["config_id"] = "other"
    elif corruption == "guardrail":
        selections[name]["guardrails"]["harm_numerator"] = 9
    elif corruption == "hash":
        selections[name]["threshold_specification_sha256"] = "tampered"
    elif corruption == "comparison":
        selections[name]["threshold_specification"]["comparison"] = "gte"
    else:
        selections[name]["selected_feasible"] = False
    with pytest.raises(ValueError):
        policy_actions(rows.drop(columns="label_id_dense"), config, selections)

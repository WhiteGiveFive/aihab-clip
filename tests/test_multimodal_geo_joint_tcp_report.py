"""Independent metric, threshold, identity, and resampling checks."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from multimodal.geo_helpfulness_joint_tcp_report import (
    JointTCPReportError, Q_COLUMNS, STATES, _validate, format_report, make_report,
)
from multimodal.geo_helpfulness_relaxed_rf_selection import select_common_threshold


def fixture():
    configs = [
        {"config_id": f"numeric{'_emb' if emb else ''}_l{weight}",
         "variant": "ablation_numeric25", "with_embeddings": emb,
         "auxiliary_weight": weight, "input_dim": 1241 if emb else 25}
        for emb in (False, True) for weight in (0, 1)
    ]
    rows = []
    raw_predictions = [1, 0, 0, 1, 1, 1]
    geo_predictions = [0, 1, 0, 2, 1, 0]
    q = np.array([[.6, .2, .1, .1], [.5, .3, .1, .1], [.7, .1, .1, .1],
                  [.2, .2, .1, .5], [.8, .1, .05, .05], [.1, .2, .1, .6]], dtype=np.float32)
    for item in configs:
        for split in ("train", "validation"):
            for seed in (1, 2, 3, 4):
                for index in range(6):
                    raw, geo = np.zeros(18, dtype=np.float32), np.zeros(18, dtype=np.float32)
                    raw[raw_predictions[index]], geo[geo_predictions[index]] = 3, 3
                    values = q[index].copy()
                    if item["auxiliary_weight"] > 0 and index in (1, 5):
                        values = np.array([.1, .7, .1, .1] if index == 1 else [.7, .1, .1, .1], dtype=np.float32)
                    estimate = np.array([[.2, .8, .8, .2, .4, .2][index],
                                         [.8, .2, .8, .3, .4, .8][index]], dtype=np.float32)
                    rows.append({
                        **item, "split": split, "training_seed": seed,
                        "row_uid": f"{split}_r{index}", "plot_idx": f"{split}_p{index // 2}",
                        "producer_fold": index % 4 if split == "train" else -1,
                        "label_id_dense": 0, "raw_logits": raw, "geo_logits": geo,
                        "image_logits": raw.copy(),
                        "fusion_tcp_hat": estimate[0], "geo_tcp_hat": estimate[1],
                        "state_logits": np.log(values).astype(np.float32),
                        "tcp_logits": np.log(estimate / (1 - estimate)).astype(np.float32),
                        **dict(zip(Q_COLUMNS, values)),
                    })
    return pd.DataFrame(rows), {
        "study_id": "joint_tcp_router_v1", "configs": configs,
        "bootstrap": {"resamples": 17, "seed": 20260826},
    }


@pytest.fixture(scope="module")
def scored():
    rows, config = fixture()
    return rows, config, make_report(rows, config)


def test_all_predeclared_policies_and_matched_contrasts_are_reported(scored):
    _, _, output = scored
    report = output["report"]
    assert report["model_count"] == 16
    assert report["selected_policy_count"] == 36
    assert report["policy_count_including_references"] == 41
    assert output["bootstrap_draws"].shape == (17, 41, 2)
    kinds = [item["kind"] for item in report["validation_bootstrap"]["contrasts"].values()]
    assert kinds.count("versus_fusion") == 40
    assert kinds.count("auxiliary_weight_versus_zero") == 20
    assert kinds.count("embeddings_versus_no_embeddings") == 20
    assert not report["lambda_winner_selected"]
    json.dumps(report, allow_nan=False)
    json.dumps(output["selections"], allow_nan=False)
    assert "no winning lambda" in format_report(report)


def test_state_and_confidence_metrics_match_independent_arithmetic(scored):
    rows, _, output = scored
    data = rows.loc[(rows.config_id == "numeric_l0") & (rows.split == "validation") & (rows.training_seed == 1)]
    q = data[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
    states = np.array([0, 1, 2, 3, 3, 0])
    entry = output["report"]["diagnostics"]["numeric_l0"]["splits"]["validation"]["per_seed"]["1"]
    assert entry["state"]["nll"] == pytest.approx(-np.log(q[np.arange(6), states]).mean())
    assert entry["state"]["brier"] == pytest.approx(np.sum((q - np.eye(4)[states]) ** 2, axis=1).mean())
    assert entry["state"]["macro_f1"] == pytest.approx(f1_score(states, q.argmax(1), labels=range(4), average="macro"))
    failure = np.stack(data.raw_logits).argmax(1) != data.label_id_dense.to_numpy()
    estimate = data.fusion_tcp_hat.to_numpy(dtype=np.float64)
    result = entry["tcp"]["fusion"]["failure_detection"]["predicted_tcp"]
    assert result["average_precision"] == pytest.approx(average_precision_score(failure, 1 - estimate))
    assert result["roc_auc"] == pytest.approx(roc_auc_score(failure, 1 - estimate))
    assert entry["tcp"]["fusion"]["supervised"] is False
    assert entry["tcp_ordering"]["rescue"]["true_tcp_correct_order_fraction"] == 1
    assert entry["tcp_ordering"]["harm"]["estimated_tcp_correct_order_fraction"] == 1


def test_policy_metrics_match_independent_weighted_f1(scored):
    _, _, output = scored
    frame = output["actions"]
    report = output["report"]
    for name, value in report["validation_policies"].items():
        action = frame[name].to_numpy()
        prediction = np.where(action, frame.geo_pred, frame.raw_pred)
        assert value["pooled"]["accuracy"] == np.mean(prediction == frame.label_id_dense)
        assert value["pooled"]["weighted_f1"] == pytest.approx(f1_score(frame.label_id_dense, prediction, average="weighted"))
        assert not action[frame.geo_pred.eq(frame.raw_pred)].any()
    assert report["validation_policies"]["numeric_l0__zero"]["pooled"]["net"] == 0
    assert report["validation_policies"]["numeric_l1__zero"]["pooled"]["net"] == 8


def test_thresholds_exactly_reuse_existing_selector_and_guardrails(scored):
    rows, config, output = scored
    prepared = _validate(rows, config)
    for name, selection in output["selections"].items():
        guardrail = selection["guardrails"]
        expected = select_common_threshold(
            prepared[selection["config_id"]]["validation"],
            minimum_plots=guardrail["minimum_plots"], harm_numerator=guardrail["harm_numerator"],
        )
        assert {key: value for key, value in selection.items() if key != "config_id"} == expected
        if guardrail["minimum_plots"] > 3:
            assert not output["actions"][name].any()


def test_bootstrap_matches_direct_whole_plot_resampling(scored):
    _, config, output = scored
    frame = output["actions"]
    plots = np.unique(frame.plot_idx)
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    names = output["bootstrap_policy_order"]
    for draw_index in range(config["bootstrap"]["resamples"]):
        sampled = rng.integers(0, len(plots), size=len(plots))
        for policy in ("fusion", "numeric_l0__zero", "numeric_l1__zero"):
            per_seed = []
            for seed in (1, 2, 3, 4):
                sampled_rows = pd.concat([frame.loc[(frame.plot_idx == plots[index]) & (frame.training_seed == seed)] for index in sampled])
                pred = np.where(sampled_rows[policy], sampled_rows.geo_pred, sampled_rows.raw_pred)
                per_seed.append([np.mean(pred == sampled_rows.label_id_dense), f1_score(sampled_rows.label_id_dense, pred, average="weighted")])
            assert output["bootstrap_draws"][draw_index, names.index(policy)] == pytest.approx(np.mean(per_seed, axis=0))


def test_float64_score_arithmetic_preserves_promoted_native_values():
    rows, config = fixture()
    prepared = _validate(rows, config)
    frame = prepared["numeric_l0"]["train"]
    expected = frame.q_rescue.to_numpy(dtype=np.float64) - frame.q_harm.to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(frame.score, expected)
    native_difference = frame.q_rescue.to_numpy(dtype=np.float32) - frame.q_harm.to_numpy(dtype=np.float32)
    assert np.any(expected != native_difference.astype(np.float64))


@pytest.mark.parametrize("corruption", ["duplicate", "label", "producer", "expert_logits", "simplex", "tcp", "split_overlap"])
def test_invalid_or_mismatched_predictions_are_rejected(corruption):
    rows, config = fixture()
    if corruption == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    elif corruption == "label":
        rows.loc[0, "label_id_dense"] = 1
    elif corruption == "producer":
        rows.loc[0, "producer_fold"] = 9
    elif corruption == "expert_logits":
        changed = rows.loc[0, "raw_logits"].copy()
        changed[0] += 1
        rows.at[0, "raw_logits"] = changed
    elif corruption == "simplex":
        rows.loc[0, "q_rescue"] = 1
    elif corruption == "tcp":
        rows.loc[0, "geo_tcp_hat"] = np.nan
    else:
        rows["plot_idx"] = rows.plot_idx.str.replace("train", "validation")
    with pytest.raises(JointTCPReportError):
        make_report(rows, config)


def test_permutation_invariance_and_exact_repeat(scored):
    rows, config, output = scored
    repeated = make_report(rows.sample(frac=1, random_state=7), config)
    assert repeated["report"] == output["report"]
    assert repeated["selections"] == output["selections"]
    pd.testing.assert_frame_equal(repeated["actions"], output["actions"])
    np.testing.assert_array_equal(repeated["bootstrap_draws"], output["bootstrap_draws"])


def test_absent_failures_and_disagreements_are_explicitly_undefined():
    rows, config = fixture()
    logits = np.zeros(18, dtype=np.float32)
    logits[0] = 3
    for column in ("raw_logits", "geo_logits", "image_logits"):
        rows[column] = [logits.copy() for _ in range(len(rows))]
    output = make_report(rows, config)
    metric = output["report"]["diagnostics"]["numeric_l0"]["splits"]["validation"]["per_seed"]["1"]
    assert metric["state"]["ranking_on_disagreements"]["rescue"]["signed_utility"]["average_precision"] is None
    assert metric["tcp"]["geo"]["failure_detection"]["predicted_tcp"]["roc_auc"] is None
    assert metric["tcp_ordering"]["harm"]["count"] == 0
    assert output["report"]["validation_bootstrap"]["unique_action_masks"] == 1


def test_state_nll_retains_information_when_native_probabilities_underflow():
    rows, config = fixture()
    mask = (rows.row_uid == "validation_r0")
    values = np.array([0, 1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    for index in rows.index[mask]:
        rows.at[index, "state_logits"] = np.array([-1000, 0, 0, 0], dtype=np.float32)
        for column, value in zip(Q_COLUMNS, values):
            rows.loc[index, column] = value
    result = make_report(rows, config)
    state = result["report"]["diagnostics"]["numeric_l0"]["splits"]["validation"]["per_seed"]["1"]["state"]
    assert state["nll_floor_count"] == 1
    assert state["nll"] > state["probability_nll"] + 40


def test_optional_rf_context_formatter_displays_every_supplied_setting(scored):
    report = deepcopy(scored[2]["report"])
    threshold = next(iter(scored[2]["selections"].values()))
    report["rf_context"] = {"policies": {
        "rf_setting_1": {"threshold": threshold}, "rf_setting_2": {"threshold": threshold}
    }}
    rendered = format_report(report)
    assert "Existing matched-expert RF context" in rendered
    assert "| rf_setting_1 |" in rendered
    assert "| rf_setting_2 |" in rendered

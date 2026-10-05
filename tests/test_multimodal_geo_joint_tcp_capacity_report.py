"""Capacity contrasts preserve pairing, architecture controls and frozen policies."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from multimodal.geo_helpfulness_joint_tcp_capacity_report import (
    ARCHITECTURES, GUARDRAILS, JointTCPReportError, Q_COLUMNS, _bootstrap, _contrast_pairs,
    _validate, format_report, make_development_report, make_test_report, policy_actions,
)
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256


def fixture():
    items, rows, selections = [], [], {}
    for architecture, widths in ARCHITECTURES.items():
        for base, width in (("numeric25", 25), ("no_pairs79", 79)):
            for weight in (0, .1, 1, 10):
                items.append({"config_id": f"{architecture}_{base}_{weight}",
                              "architecture_id": architecture, "hidden_widths": widths,
                              "input_id": base, "input_dim": width, "variant": f"ablation_{base}",
                              "with_embeddings": False, "auxiliary_weight": weight})
    config = {"configs": items, "bootstrap": {"resamples": 19, "seed": 20260826}}
    raw_pred, geo_pred = [1, 0, 0, 1, 1, 1], [0, 1, 0, 2, 1, 0]
    for item in items:
        for seed in (1, 2, 3, 4):
            for index in range(6):
                raw, geo = np.zeros(18, np.float32), np.zeros(18, np.float32)
                raw[raw_pred[index]], geo[geo_pred[index]] = 3, 3
                q = np.array([.6, .2, .1, .1] if index % 2 == 0 else [.1, .7, .1, .1], np.float32)
                if item["auxiliary_weight"] > 0 and index == 5:
                    q = np.array([.7, .1, .1, .1], np.float32)
                if item["architecture_id"] != "h32" and index == 1:
                    q = np.array([.6, .2, .1, .1], np.float32)
                estimate = np.array([.2, .8], np.float32)
                rows.append({**item, "split": "test", "training_seed": seed,
                             "row_uid": f"r{index}", "plot_idx": f"p{[0, 1, 1, 2, 2, 2][index]}",
                             "producer_fold": -1, "label_id_dense": 0,
                             "raw_logits": raw, "geo_logits": geo, "image_logits": raw.copy(),
                             "state_logits": np.log(q).astype(np.float32),
                             "tcp_logits": np.log(estimate / (1 - estimate)).astype(np.float32),
                             "fusion_tcp_hat": estimate[0], "geo_tcp_hat": estimate[1],
                             **dict(zip(Q_COLUMNS, q))})
        for plots, harm in GUARDRAILS:
            guardrail = f"plots_{plots}__harm_0p{harm}"
            specification = {"kind": "finite", "value_hex": float(.1).hex(), "comparison": "strict_gt"}
            selections[f"{item['config_id']}__{guardrail}"] = {
                "config_id": item["config_id"], "threshold_specification": specification,
                "threshold_specification_sha256": canonical_sha256(specification),
                "selected_feasible": True, "guardrails": {
                    "guardrail_id": guardrail, "minimum_plots": plots, "harm_numerator": harm,
                    "harm_denominator": 10, "minimum_coverage_fraction": .01,
                    "require_positive_net_utility": True}}
    return pd.DataFrame(rows), config, selections


def historical(rows):
    frame = rows.loc[rows.config_id.eq(rows.config_id.iloc[0])].sort_values(
        ["row_uid", "training_seed"], kind="stable").reset_index(drop=True)
    actions = frame[["row_uid", "plot_idx", "training_seed"]].copy()
    actions["raw_pred"] = np.stack(frame.raw_logits).argmax(1)
    actions["geo_pred"] = np.stack(frame.geo_logits).argmax(1)
    metadata = {}
    for family in ("rf", "lr"):
        for input_id in ("numeric25", "no_pairs79", "full727"):
            name = f"historical_{family}_{input_id}"
            actions[name] = frame.row_uid.eq("r0")
            metadata[name] = {"family": family, "input_id": input_id,
                              "guardrail_id": "plots_20__harm_0p7", "source_hash": "a" * 64}
    return {"actions": actions, "metadata": metadata}


@pytest.fixture(scope="module")
def scored():
    rows, config, selections = fixture()
    comparators = historical(rows)
    return rows, config, selections, comparators, make_test_report(rows, config, selections, comparators)


def test_declared_policy_and_contrast_counts(scored):
    _, config, _, _, result = scored
    report = result["report"]
    assert report["model_count"] == 160
    assert report["new_fit_count"] == 128
    assert report["selected_policy_count"] == 360
    assert report["policy_count_including_references"] == 407
    assert result["bootstrap_draws"].shape == (19, 407, 2)
    contrasts = report["test_bootstrap"]["contrasts"]
    assert len(contrasts) == 1100
    kinds = [item["kind"] for item in contrasts.values()]
    assert kinds.count("versus_fusion") == 400
    assert kinds.count("auxiliary_weight_versus_zero") == 300
    assert kinds.count("larger_architecture_versus_h32") == 320
    assert kinds.count("matched_input_versus_historical_rf") == 40
    assert kinds.count("matched_input_versus_historical_lr") == 40
    assert len(_contrast_pairs(config)) == 1020
    assert report["test_used_for_selection"] is False
    assert report["test_harm_caps_enforced"] is False
    json.dumps(report, allow_nan=False)


def test_control_matching_is_architecture_and_weight_aware(scored):
    _, config, _, _, result = scored
    items = {item["config_id"]: item for item in config["configs"]}
    for item in result["report"]["test_bootstrap"]["contrasts"].values():
        if item["kind"] not in ("auxiliary_weight_versus_zero", "larger_architecture_versus_h32"):
            continue
        left = items[item["treatment"].split("__")[0]]
        right = items[item["reference"].split("__")[0]]
        assert left["input_id"] == right["input_id"]
        if item["kind"] == "auxiliary_weight_versus_zero":
            assert left["architecture_id"] == right["architecture_id"]
            assert left["auxiliary_weight"] > 0 and right["auxiliary_weight"] == 0
        else:
            assert left["auxiliary_weight"] == right["auxiliary_weight"]
            assert right["architecture_id"] == "h32"


def test_label_blind_and_per_seed_application(scored):
    rows, config, selections, comparators, result = scored
    blind = rows.drop(columns="label_id_dense")
    expected = result["actions"].drop(columns=["label_id_dense", "target_state"])
    pd.testing.assert_frame_equal(policy_actions(blind, config, selections, comparators), expected)
    single = policy_actions(blind.loc[blind.training_seed.eq(3)], config, selections, comparators)
    pd.testing.assert_frame_equal(single, expected.loc[expected.training_seed.eq(3)].reset_index(drop=True))
    with pytest.raises(JointTCPReportError, match="label fields"):
        policy_actions(rows, config, selections, comparators)


def test_frozen_actions_are_not_reselected_for_test_harms(scored):
    rows, config, selections, comparators, result = scored
    changed = rows.copy()
    changed["label_id_dense"] = np.stack(rows.raw_logits).argmax(1)
    output = make_test_report(changed, config, selections, comparators)
    names = result["bootstrap_policy_order"]
    pd.testing.assert_frame_equal(output["actions"][names], result["actions"][names])
    item = output["report"]["test_policies"]["h32_numeric25_0__plots_20__harm_0p5"]
    assert item["pooled"]["net"] < 0
    assert item["test_guardrail_diagnostics"]["harm_cap_met"] is False


def test_strict_boundary_native_score_and_fallback():
    rows, config, selections = fixture()
    blind = rows.drop(columns="label_id_dense")
    first = _validate(blind, config, labels=False)["h32_numeric25_0"]
    q = first[list(Q_COLUMNS)].to_numpy(np.float64)
    np.testing.assert_array_equal(first.score, q[:, 0] - q[:, 1])
    native = (first.q_rescue.to_numpy(np.float32) - first.q_harm.to_numpy(np.float32)).astype(np.float64)
    assert np.any(first.score.to_numpy() != native)
    name = "h32_numeric25_0__plots_20__harm_0p5"
    specification = {"kind": "finite", "value_hex": float(first.score.iloc[0]).hex(), "comparison": "strict_gt"}
    selections[name].update(threshold_specification=specification,
                            threshold_specification_sha256=canonical_sha256(specification))
    actions = policy_actions(blind, config, selections)
    assert not actions.loc[actions.row_uid.eq("r0"), name].any()
    assert not actions.loc[actions.raw_pred.eq(actions.geo_pred), name].any()
    specification = {"kind": "never_intervene", "value_hex": None, "comparison": "strict_gt"}
    selections[name].update(threshold_specification=specification, selected_feasible=False,
                            threshold_specification_sha256=canonical_sha256(specification))
    assert not policy_actions(blind, config, selections)[name].any()


def test_independent_metrics_and_integer_bootstrap(scored):
    _, config, _, _, result = scored
    frame = result["actions"]
    names = result["bootstrap_policy_order"]
    treatment, control = "h32_numeric25_0.1__zero", "h32_numeric25_0__zero"
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    plots = np.unique(frame.plot_idx)
    differences = []
    for draw in range(config["bootstrap"]["resamples"]):
        indices = rng.integers(0, len(plots), size=len(plots))
        sample = pd.concat([frame.loc[frame.plot_idx.eq(plots[index])] for index in indices])
        left = np.where(sample[treatment], sample.geo_pred, sample.raw_pred)
        right = np.where(sample[control], sample.geo_pred, sample.raw_pred)
        differences.append((int(np.sum(left == sample.label_id_dense))
                            - int(np.sum(right == sample.label_id_dense))) / len(sample))
        for name in ("fusion", treatment, control, "historical_rf_numeric25"):
            by_seed = []
            for seed in (1, 2, 3, 4):
                part = sample.loc[sample.training_seed.eq(seed)]
                prediction = np.where(part[name], part.geo_pred, part.raw_pred)
                by_seed.append([np.mean(prediction == part.label_id_dense),
                                f1_score(part.label_id_dense, prediction, average="weighted")])
            np.testing.assert_allclose(result["bootstrap_draws"][draw, names.index(name)], np.mean(by_seed, axis=0))
    contrast = result["report"]["test_bootstrap"]["contrasts"][f"{treatment}_minus_{control}"]
    lower, upper = np.quantile(differences, [.025, .975])
    assert contrast["accuracy"]["ci95_lower"] == lower
    assert contrast["accuracy"]["ci95_upper"] == upper
    identical = result["report"]["test_bootstrap"]["contrasts"][
        "h32_numeric25_1__zero_minus_h32_numeric25_0__zero"]
    assert identical["accuracy"] == contrast["accuracy"]
    for name, item in result["report"]["test_policies"].items():
        prediction = np.where(frame[name], frame.geo_pred, frame.raw_pred)
        assert item["pooled"]["accuracy"] == np.mean(prediction == frame.label_id_dense)
        assert item["pooled"]["weighted_f1"] == pytest.approx(f1_score(frame.label_id_dense, prediction, average="weighted"))


def test_development_threshold_selection_and_replay():
    rows, config, _ = fixture()
    validation = rows.copy()
    validation["split"] = "validation"
    train = validation.copy()
    train["split"] = "train"
    train["row_uid"] = "train_" + train.row_uid
    train["plot_idx"] = "train_" + train.plot_idx
    train["producer_fold"] = 0
    combined = pd.concat([train, validation], ignore_index=True)
    result = make_development_report(combined, config)
    assert result["report"]["selected_policy_count"] == 360
    assert len(result["report"]["validation_bootstrap"]["contrasts"]) == 1020
    assert result["bootstrap_draws"].shape == (19, 401, 2)
    assert result["report"]["cleaned_test_access"] is False
    applied = policy_actions(rows.drop(columns="label_id_dense"), config, result["selections"])
    names = result["bootstrap_policy_order"]
    pd.testing.assert_frame_equal(applied[names], result["actions"][names])
    assert format_report(result["report"]) == format_report(json.loads(json.dumps(result["report"], sort_keys=True)))


def test_balanced_seed_gains_preserve_exact_zero_despite_float_mean_artifact():
    _, config, _ = fixture()
    frame = pd.DataFrame({"plot_idx": ["one_plot"] * 20,
                          "training_seed": np.repeat([1, 2, 3, 4], 5),
                          "label_id_dense": [0] * 20,
                          "raw_pred": [0, 0, 0, 1, 1] * 4,
                          "geo_pred": [1, 1, 1, 0, 0] * 4})
    suffixes = ["zero"] + [f"plots_{p}__harm_0p{h}" for p, h in GUARDRAILS]
    actions = {"fusion": np.zeros(20, dtype=bool)}
    for item in config["configs"]:
        actions.update({f"{item['config_id']}__{suffix}": np.zeros(20, dtype=bool) for suffix in suffixes})
    treatment, control = "h32_numeric25_0.1__zero", "h32_numeric25_0__zero"
    actions[treatment][[3, 8, 13, 15, 16, 17]] = True
    report, draws = _bootstrap(frame, actions, config)
    names = report["policy_order"]
    assert draws[0, names.index(treatment), 0] - draws[0, names.index(control), 0] != 0
    metric = report["contrasts"][f"{treatment}_minus_{control}"]["accuracy"]
    assert metric == {"difference": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}


def test_repeat_permutation_and_sorted_json(scored):
    rows, config, selections, comparators, result = scored
    shuffled = deepcopy(comparators)
    shuffled["actions"] = shuffled["actions"].sample(frac=1, random_state=5)
    again = make_test_report(rows.sample(frac=1, random_state=7), config, selections, shuffled)
    assert again["report"] == result["report"]
    np.testing.assert_array_equal(again["bootstrap_draws"], result["bootstrap_draws"])
    pd.testing.assert_frame_equal(again["actions"], result["actions"])
    restored = json.loads(json.dumps(result["report"], sort_keys=True, allow_nan=False))
    assert format_report(result["report"]) == format_report(restored)
    text = format_report(restored)
    for name in result["bootstrap_policy_order"]:
        assert f"| {name} |" in text


@pytest.mark.parametrize("corruption", ["duplicate", "producer", "seed", "label", "expert", "q", "tcp", "width", "architecture", "extra"])
def test_bad_prediction_contract(corruption):
    rows, config, selections = fixture()
    if corruption == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    elif corruption == "producer":
        rows.loc[0, "producer_fold"] = 0
    elif corruption == "seed":
        rows = rows.loc[rows.training_seed.ne(4)]
    elif corruption == "label":
        rows.loc[0, "label_id_dense"] = 1
    elif corruption == "expert":
        rows.at[0, "raw_logits"] = np.ones(18)
    elif corruption == "q":
        rows.loc[0, "q_rescue"] = 1
    elif corruption == "tcp":
        rows.loc[0, "geo_tcp_hat"] = 2
    elif corruption == "width":
        rows.at[0, "hidden_widths"] = [999]
    elif corruption == "architecture":
        rows.loc[0, "architecture_id"] = "h128_64"
    else:
        rows["secret_target"] = 1
    with pytest.raises(JointTCPReportError):
        make_test_report(rows, config, selections)


@pytest.mark.parametrize("corruption", ["identity", "expert", "action", "family", "guardrail", "label", "missing"])
def test_bad_historical_comparison_contract(corruption):
    rows, config, selections = fixture()
    comparators = historical(rows)
    first = next(iter(comparators["metadata"]))
    if corruption == "identity":
        comparators["actions"].loc[0, "row_uid"] = "missing"
    elif corruption == "expert":
        comparators["actions"].loc[0, "raw_pred"] = 0
    elif corruption == "action":
        comparators["actions"][first] = comparators["actions"][first].astype(int)
    elif corruption == "family":
        comparators["metadata"][first]["family"] = "mlp"
    elif corruption == "guardrail":
        comparators["metadata"][first]["guardrail_id"] = "plots_0__harm_0p9"
    elif corruption == "label":
        comparators["actions"]["label_id_dense"] = 0
    else:
        del comparators["metadata"][first]
    with pytest.raises(JointTCPReportError):
        policy_actions(rows.drop(columns="label_id_dense"), config, selections, comparators)


@pytest.mark.parametrize("corruption", ["missing", "hash", "guardrail", "feasible"])
def test_frozen_threshold_corruption_rejected(corruption):
    rows, config, selections = fixture()
    first = next(iter(selections))
    if corruption == "missing":
        del selections[first]
    elif corruption == "hash":
        selections[first]["threshold_specification_sha256"] = "0" * 64
    elif corruption == "guardrail":
        selections[first]["guardrails"]["minimum_plots"] = 1
    else:
        selections[first]["selected_feasible"] = False
    with pytest.raises(JointTCPReportError):
        policy_actions(rows.drop(columns="label_id_dense"), config, selections)

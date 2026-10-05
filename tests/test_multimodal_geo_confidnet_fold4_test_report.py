"""Independent arithmetic checks for frozen fold-4 cleaned-test scoring."""

from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from multimodal.geo_helpfulness_confidnet_fold4_test_report import (
    EXPERT_ORDER,
    RELAXED_NAMES,
    ConfidNetMetricError,
    build_report,
    choices_frame,
    format_report,
    paired_plot_bootstrap,
    policy_choices,
    policy_metrics,
    scored_rows_and_choices,
)


def payload():
    raw = np.array([
        [.2, .6, .1, .1], [.6, .2, .1, .1], [.1, .1, .7, .1],
        [.2, .6, .1, .1], [.2, .6, .1, .1], [.6, .2, .1, .1],
        [.6, .1, .2, .1], [.1, .1, .1, .7],
    ])
    geo = np.array([
        [.7, .1, .1, .1], [.2, .6, .1, .1], [.1, .1, .7, .1],
        [.1, .2, .6, .1], [.1, .7, .1, .1], [.1, .6, .2, .1],
        [.6, .1, .2, .1], [.1, .1, .1, .7],
    ])
    image = np.array([
        [.1, .1, .1, .7], [.7, .1, .1, .1], [.1, .1, .7, .1],
        [.7, .1, .1, .1], [.1, .1, .7, .1], [.1, .7, .1, .1],
        [.1, .1, .7, .1], [.7, .1, .1, .1],
    ])
    return {
        "labels": np.array([0, 0, 2, 0, 0, 1, 2, 3]),
        "plot_ids": np.array(["p1", "p1", "p2", "p2", "p2", "p3", "p3", "p3"]),
        "row_uids": np.array([f"r{x}" for x in range(8)]),
        "image_logits": np.log(image), "geo_logits": np.log(geo), "raw_logits": np.log(raw),
        "image_tcp_hat": np.array([.1, .7, .9, .8, .99, .5, .95, .99], dtype=np.float32),
        "geo_tcp_hat": np.array([.8, .2, .9, .5, .9, .5, .8, .4], dtype=np.float32),
        "raw_tcp_hat": np.array([.3, .7, .8, .5, .1, .5, .8, .6], dtype=np.float32),
        "router_baseline_q": np.array([
            [.6, .2, .1, .1], [.5, .3, .1, .1], [.7, .1, .1, .1],
            [.2, .2, .1, .5], [.8, .1, .05, .05], [.1, .2, .1, .6],
            [.8, .1, .05, .05], [.8, .1, .05, .05],
        ]),
        "router_augmented_q": np.array([
            [.6, .2, .1, .1], [.1, .7, .1, .1], [.7, .1, .1, .1],
            [.3, .2, .1, .4], [.8, .1, .05, .05], [.6, .1, .1, .2],
            [.8, .1, .05, .05], [.8, .1, .05, .05],
        ]),
    }


def selections():
    return {
        name: {
            "arm": name.split("__")[0],
            "threshold_specification": {
                "kind": "finite" if name.startswith("baseline") else "never_intervene",
                "value_hex": (0.2).hex() if name.startswith("baseline") else None,
                "comparison": "strict_gt",
            },
            "selected_feasible": name.startswith("baseline"),
        }
        for name in RELAXED_NAMES
    }


def four_seeds():
    return {seed: payload() for seed in range(1, 5)}


def test_choices_are_label_blind_and_three_expert_ties_prioritize_fusion_geo_image():
    data = payload()
    del data["labels"]
    choices = policy_choices(data, selections())
    assert len(choices) == 26
    assert choices["tcp_max_three"].tolist() == [1, 0, 1, 2, 2, 0, 2, 2]
    assert all(choice.dtype == np.dtype("int8") for choice in choices.values())
    data["labels"] = object()  # Not even a valid label vector; it is never accessed.
    assert all(np.array_equal(value, policy_choices(data, selections())[name]) for name, value in choices.items())
    assert choices["tcp_geo_fusion"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


def test_three_expert_agreement_switch_counts_differ_from_prediction_actions():
    rows, choices = scored_rows_and_choices(four_seeds(), selections())
    result = policy_metrics(rows, choices["tcp_max_three"])
    one = result["per_seed"]["1"]
    assert one["actions"] == 5
    assert one["rescues"] == 3
    assert one["harms"] == 1
    assert one["wrong_to_wrong"] == 1
    assert one["net"] == 2
    assert one["model_switches"] == 6
    assert one["model_switches_with_unchanged_prediction"] == 1
    assert one["selected_expert_counts"] == {"fusion": 2, "geo": 2, "image": 4}
    assert one["accuracy"] == pytest.approx(5 / 8)
    assert result["pooled"]["rescues"] == 12
    assert result["mean_seed"]["rescues"] == 3
    assert result["pooled"]["net"] == 8


def test_thresholds_transferred_without_refitting_and_strict_equality_retains_fusion():
    data, selected = payload(), selections()
    threshold = float(data["router_baseline_q"][0, 0] - data["router_baseline_q"][0, 1])
    for name, entry in selected.items():
        if name.startswith("baseline"):
            entry["threshold_specification"]["value_hex"] = threshold.hex()
    choice = policy_choices(data, selected)
    assert choice["baseline_zero"][0] == 1
    assert choice["baseline__plots_0__harm_0p9"][0] == 0
    assert all(not choice[name].any() for name in RELAXED_NAMES if name.startswith("augmented"))
    changed = deepcopy(data)
    changed["labels"] = np.arange(8) % 4
    assert all(np.array_equal(choice[name], value) for name, value in policy_choices(changed, selected).items())


def test_top3_uses_chosen_expert_even_when_top1_agrees_and_breaks_class_ties_stably():
    seeds = four_seeds()
    for data in seeds.values():
        # Raw and geo both predict class 0, but only geo includes true class 3
        # in its top three. The direct rule selects geo; the paired rule retains raw.
        data["raw_logits"][2] = [4, 2, 2, 2]
        data["geo_logits"][2] = [4, 0, 1, 3]
        data["labels"][2] = 3
        data["image_tcp_hat"][2] = .1
    rows, choices = scored_rows_and_choices(seeds, selections())
    frame = choices_frame(rows, choices)
    test = frame.loc[frame.row_uid.eq("r2")]
    direct = test.loc[test.policy.eq("tcp_max_three")]
    paired = test.loc[test.policy.eq("tcp_geo_fusion")]
    assert direct.top3_correct.all()
    assert not paired.top3_correct.any()
    assert direct.selected_expert.eq("geo").all()
    assert not direct.prediction_changed_vs_fusion.any()
    assert not direct.correct.any()


def test_metrics_confidence_and_oracle_match_independent_calculations():
    seeds = four_seeds()
    result = build_report(seeds, selections(), n_resamples=13, random_seed=6)
    assert result["thresholds_reselected_on_test"] is False
    json.dumps(result, allow_nan=False)
    data = seeds[1]
    labels = data["labels"]
    predictions = np.column_stack([data[f"{expert}_logits"].argmax(axis=1) for expert in EXPERT_ORDER])
    correct = predictions == labels[:, None]
    oracle = result["oracle"]["mean_seed"]
    assert oracle["geo_fusion"]["accuracy"] == np.any(correct[:, :2], axis=1).mean()
    assert oracle["three_expert"]["accuracy"] == np.any(correct, axis=1).mean()
    for name, choice in policy_choices(data, selections()).items():
        selected = predictions[np.arange(len(labels)), choice]
        metric = result["policies"][name]["mean_seed"]
        assert metric["accuracy"] == np.mean(selected == labels)
        assert metric["weighted_f1"] == pytest.approx(f1_score(labels, selected, average="weighted"))
    for expert in EXPERT_ORDER:
        logits, tcp = data[f"{expert}_logits"], data[f"{expert}_tcp_hat"].astype(np.float64)
        failed = logits.argmax(axis=1) != labels
        metric = result["confidence"]["mean_seed"][expert]
        assert metric["failure_detection"]["predicted_tcp"]["average_precision"] == pytest.approx(average_precision_score(failed, 1 - tcp))
        assert metric["failure_detection"]["predicted_tcp"]["roc_auc"] == pytest.approx(roc_auc_score(failed, 1 - tcp))
        probability = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        assert metric["tcp_regression"]["mse"] == pytest.approx(np.mean((tcp - probability[np.arange(len(labels)), labels]) ** 2))
    text = format_report(result)
    assert "No test labels are used" in text
    assert "tcp_max_three" in text
    assert "never intervene" in text


def test_seed_row_permutations_preserve_canonical_results():
    initial = four_seeds()
    shuffled = deepcopy(initial)
    shuffled[3] = {key: value[::-1] for key, value in shuffled[3].items()}
    rows1, choices1 = scored_rows_and_choices(initial, selections())
    rows2, choices2 = scored_rows_and_choices(shuffled, selections())
    assert rows1.equals(rows2)
    assert all(np.array_equal(value, choices2[name]) for name, value in choices1.items())
    assert build_report(initial, selections(), n_resamples=5) == build_report(shuffled, selections(), n_resamples=5)


def test_bootstrap_matches_whole_plot_expansion_and_seed_mean_f1():
    seeds = four_seeds()
    seeds[4]["image_tcp_hat"][:] = 0
    rows, choices = scored_rows_and_choices(seeds, selections())
    result = paired_plot_bootstrap(rows, choices, n_resamples=31, random_seed=19)
    plots, labels = np.unique(rows.plot_idx), rows.label_id_dense.to_numpy()
    predictions = rows[[f"{expert}_pred" for expert in EXPERT_ORDER]].to_numpy()
    top3 = rows[[f"{expert}_top3_correct" for expert in EXPERT_ORDER]].to_numpy()
    predicted = predictions[np.arange(len(rows)), choices["tcp_max_three"]]
    top3_selected = top3[np.arange(len(rows)), choices["tcp_max_three"]]
    raw = rows.raw_pred.to_numpy()
    rng = np.random.default_rng(19)
    differences = []
    for _ in range(31):
        selected_plots = plots[rng.integers(0, len(plots), size=len(plots))]
        per_seed = []
        for seed in range(1, 5):
            index = np.concatenate([
                np.flatnonzero(rows.plot_idx.eq(plot) & rows.training_seed.eq(seed))
                for plot in selected_plots
            ])
            per_seed.append([
                np.mean(predicted[index] == labels[index]) - np.mean(raw[index] == labels[index]),
                f1_score(labels[index], predicted[index], average="weighted") - f1_score(labels[index], raw[index], average="weighted"),
                top3_selected[index].mean() - rows.raw_top3_correct.to_numpy()[index].mean(),
            ])
        differences.append(np.mean(per_seed, axis=0))
    low, high = np.quantile(differences, [.025, .975], axis=0)
    contrast = result["contrasts"]["tcp_max_three_minus_fusion"]
    for index, name in enumerate(("accuracy", "weighted_f1", "top3_accuracy")):
        assert contrast[name]["ci95_lower"] == pytest.approx(low[index])
        assert contrast[name]["ci95_upper"] == pytest.approx(high[index])
    assert result["plot_count"] == 3
    assert result["seed_count"] == 4
    assert result["thresholds_reselected"] is False


@pytest.mark.parametrize("count", [0, -2, True, 1.5])
def test_bootstrap_rejects_bad_resample_count(count):
    rows, choices = scored_rows_and_choices(four_seeds(), selections())
    with pytest.raises(ConfidNetMetricError, match="positive integer"):
        paired_plot_bootstrap(rows, choices, n_resamples=count)


@pytest.mark.parametrize("change", ["missing", "extra", "arm"])
def test_requires_all_saved_threshold_records_without_silent_changes(change):
    selected = selections()
    if change == "missing":
        selected.pop(RELAXED_NAMES[0])
    elif change == "extra":
        selected["other"] = selected[RELAXED_NAMES[0]]
    else:
        selected[RELAXED_NAMES[0]]["arm"] = "augmented"
    with pytest.raises(ConfidNetMetricError, match="18 saved|arm"):
        policy_choices(payload(), selected)


@pytest.mark.parametrize("change", ["seed", "label", "plot", "row", "class"])
def test_seed_identites_must_match(change):
    seeds = four_seeds()
    if change == "seed":
        seeds.pop(4)
    elif change == "label":
        seeds[4]["labels"][0] = 3
    elif change == "plot":
        seeds[4]["plot_ids"][0] = "p3"
    elif change == "row":
        seeds[4]["row_uids"][0] = "xx"
    else:
        for expert in EXPERT_ORDER:
            seeds[4][f"{expert}_logits"] = np.c_[seeds[4][f"{expert}_logits"], np.zeros(8)]
    with pytest.raises(ConfidNetMetricError, match="requires seeds|identities"):
        scored_rows_and_choices(seeds, selections())


@pytest.mark.parametrize("choice", [np.zeros(8), np.full(32, 3), np.full(32, -1), np.zeros(31, dtype=int)])
def test_policy_metrics_reject_invalid_expert_ids(choice):
    rows, _ = scored_rows_and_choices(four_seeds(), selections())
    with pytest.raises(ConfidNetMetricError, match="expert IDs"):
        policy_metrics(rows, choice)

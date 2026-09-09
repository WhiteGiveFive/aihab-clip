"""Synthetic, file-independent legacy-family scoring checks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_legacy_router_report as report
from multimodal.geo_helpfulness_router_heuristics import apply_heuristic_threshold
from multimodal.geo_helpfulness_router_metrics import (
    binary_average_precision,
    calibration_metrics,
    classification_metrics,
    policy_metrics,
)


def _inputs(family_shift=0):
    labels = pd.DataFrame.from_records([
        {
            "row_uid": f"image-{i:02d}", "file": f"test/{i}.jpg",
            "plot_idx": f"plot-{i // 2}", "label_id_dense": i // 4,
            "canonical_l3_id": report.ONTOLOGY[i // 4][1],
            "label_name": report.ONTOLOGY[i // 4][2],
        }
        for i in range(8)
    ])
    rows = []
    for i, label in enumerate(labels.itertuples(index=False)):
        y = label.label_id_dense
        for seed in report.SEEDS:
            state = (i + seed - 1 + family_shift) % 4
            predictions = {
                "raw": y if state in (1, 2) else (y + 2) % 18,
                "geo": y if state in (0, 2) else (y + 3) % 18,
                "image": (y + 4) % 18,
            }
            row = {"row_uid": label.row_uid, "file": label.file,
                   "plot_idx": label.plot_idx, "training_seed": seed}
            for expert, predicted in predictions.items():
                p = np.full(18, 0.1 / 15, dtype=np.float64)
                p[predicted] = 0.65
                second, third = (predicted + 1) % 18, (predicted + 2) % 18
                p[second], p[third] = 0.15, 0.10
                row[f"{expert}_prob_native_t1"] = p
                transferred = np.sqrt(p)
                row[f"{expert}_prob_calibrated"] = transferred / transferred.sum()
            rows.append(row)
    experts = pd.DataFrame.from_records(rows)
    experts["checkpoint_family"] = (
        "best_model" if family_shift == 0 else "oracle_test_best_model"
    )
    calibrated = {name: np.stack(experts[f"{name}_prob_calibrated"]) for name in ("image", "geo", "raw")}
    for name, values in report.build_semantic_values(calibrated).items():
        experts[name] = values
    routers = []
    for variant in report.VARIANTS:
        frame = experts.loc[:, [*report.IDENTITY, "checkpoint_family"]].copy()
        frame["variant_id"], frame["C_hex"] = variant, (0.01).hex()
        q = np.tile([0.45, 0.15, 0.25, 0.15], (len(frame), 1))
        q[1::3] = [0.1, 0.6, 0.1, 0.2]
        for position, name in enumerate(report.Q_COLUMNS):
            frame[name] = q[:, position]
        frame["score"] = q[:, 0] - q[:, 1]
        routers.append(frame)
    routers = pd.concat(routers, ignore_index=True)
    actions = []
    for policy in report.POLICIES:
        frame = experts.loc[:, [*report.IDENTITY, "checkpoint_family"]].copy()
        if policy in report.VARIANTS:
            score = routers.loc[routers.variant_id.eq(policy), "score"].to_numpy()
        elif policy == "never_intervene":
            score = np.zeros(len(frame), dtype=np.float64)
        elif policy == "always_geo_on_geo_raw_disagreement":
            score = np.ones(len(frame), dtype=np.float64)
        elif policy == "higher_calibrated_geo_confidence_on_geo_raw_disagreement":
            score = experts.geo_minus_raw_confidence.to_numpy()
        else:
            score = experts[policy].to_numpy()
        never = policy in ("primary_full727", "never_intervene")
        specification = {
            "kind": "never_intervene" if never else "finite",
            "value_hex": None if never else (0.0).hex(),
            "comparison": "strict_gt",
        }
        frame["policy_id"] = policy
        frame["score"] = score
        frame["threshold_specification"] = json.dumps(specification, sort_keys=True)
        frame["raw_pred"], frame["geo_pred"] = experts.raw_pred, experts.geo_pred
        mask = apply_heuristic_threshold(score, experts.raw_pred.to_numpy(), experts.geo_pred.to_numpy(), specification)
        frame["acted"] = mask
        frame["final_pred"] = np.where(mask, experts.geo_pred, experts.raw_pred)
        actions.append(frame)
    return experts, routers, pd.concat(actions, ignore_index=True), labels


@pytest.fixture(scope="module")
def scored():
    inputs = _inputs()
    return inputs, report.evaluate_family(*inputs)


def test_artifacts_bootstrap_shared_images_and_fallback(scored):
    inputs, result = scored
    assert set(result) == {
        "expert_metrics.json", "classification.json", "router_metrics.json",
        "policy_metrics.json", "bootstrap_summary.json", "bootstrap_replicates.parquet",
        "subgroup_metrics.parquet", "actions_with_outcomes.parquet", "scoring_targets.parquet",
    }
    for name, artifact in result.items():
        if name.endswith(".json"):
            json.dumps(artifact, allow_nan=False)
        else:
            assert isinstance(artifact, pa.Table)
    bootstrap = result["bootstrap_summary.json"]
    assert (bootstrap["replicates"], bootstrap["random_seed"]) == (2000, 20260826)
    assert (bootstrap["row_count"], bootstrap["image_count"], bootstrap["plot_count"]) == (32, 8, 4)
    assert bootstrap["retain_all_images_and_training_seeds"] is True
    assert bootstrap["M4_gate_rerun"] is False
    assert bootstrap["checkpoint_family"] == "best_model"
    assert result["bootstrap_replicates.parquet"].num_rows == 16000
    classification = result["classification.json"]
    raw = classification["experts"]["raw"]
    for name in ("primary_full727", "never_intervene"):
        assert classification["policies"][name] == raw
        policy = result["policy_metrics.json"][name]
        assert policy["pooled"]["acted_count"] == 0
        assert policy["mean_seed_delta_accuracy"] == 0
        assert policy["pooled"]["override_precision"] is None
        assert policy["raw_fallback_selected"] is True
        assert policy["all_predictions_equal_raw"] is True
    subgroups = result["subgroup_metrics.parquet"].to_pandas()
    assert set(subgroups.group_family) == {"habitat", "plot", "geo_raw_pred_pair", "image_geo_pred_pair"}
    habitat = subgroups.loc[(subgroups.group_family == "habitat") & (subgroups.aggregation == "pooled")]
    assert set(habitat.image_count) == {4}
    assert set(habitat.row_count) == {16}


def test_math_parity_calibration_classification_and_selected_top3(scored):
    (experts, routers, actions, labels), result = scored
    targets = result["scoring_targets.parquet"].to_pandas()
    y = targets.label_id_dense.to_numpy(dtype=np.int64)
    for basis in ("native_t1", "calibrated"):
        expected = calibration_metrics(np.stack(experts[f"raw_prob_{basis}"]), y)
        assert result["expert_metrics.json"]["raw"][basis]["pooled_seed_realizations"] == expected
    name = "always_geo_on_geo_raw_disagreement"
    frame = actions.loc[actions.policy_id.eq(name)]
    mask = frame.acted.to_numpy()
    assert result["policy_metrics.json"][name]["pooled"] == policy_metrics(targets, mask)["pooled"]
    final = frame.final_pred.to_numpy(dtype=np.int64)
    metric = result["classification.json"]["policies"][name]["pooled_seed_realizations"]
    reference = classification_metrics(final, y, 18)
    for field in ("confusion_matrix", "accuracy", "macro_f1", "weighted_f1", "mcc"):
        assert metric[field] == reference[field]
    selected = np.where(mask[:, None], np.stack(experts.geo_prob_calibrated), np.stack(experts.raw_prob_calibrated))
    top3 = np.any(np.argsort(-selected, axis=1, kind="stable")[:, :3] == y[:, None], axis=1)
    assert metric["top3_accuracy"] == top3.mean()
    assert metric["class_count"] == 18
    assert len(metric["per_class"]) == 18
    assert metric["per_class"][17]["f1"] == 0
    assert metric["per_class"][17]["top3_accuracy"] is None
    q = routers.loc[routers.variant_id.eq(report.VARIANTS[0]), list(report.Q_COLUMNS)].to_numpy()
    router_metric = result["router_metrics.json"][report.VARIANTS[0]]["pooled_seed_realizations"]
    assert router_metric["rescue_pr_auc"] == binary_average_precision(targets.target_state.eq("rescue").to_numpy(), q[:, 0])
    counts = result["policy_metrics.json"][name]["pooled"]
    assert counts["final_correct_count"] - counts["raw_correct_count"] == counts["rescued_minus_harmed"]


def test_targets_are_seed_and_family_specific(scored):
    _, first = scored
    second = report.evaluate_family(*_inputs(family_shift=1))
    left = first["scoring_targets.parquet"].to_pandas()
    right = second["scoring_targets.parquet"].to_pandas()
    assert left.groupby("row_uid").target_state.nunique().eq(4).all()
    assert (left.target_state != right.target_state).all()
    assert left[list(report.KEY)].equals(right[list(report.KEY)])
    assert set(left.checkpoint_family) == {"best_model"}
    assert set(right.checkpoint_family) == {"oracle_test_best_model"}
    assert first["bootstrap_summary.json"]["resample_multiplicity_sha256"] == second["bootstrap_summary.json"]["resample_multiplicity_sha256"]
    for result in (first, second):
        outcome = result["actions_with_outcomes.parquet"].to_pandas()
        assert np.array_equal(outcome.net_correct, outcome.final_correct.astype(int) - outcome.raw_correct.astype(int))
    oracle_context = second["classification.json"]
    assert oracle_context["checkpoint_family"] == "oracle_test_best_model"
    assert "test-selection-biased" in oracle_context["checkpoint_selection_disclosure"]
    assert "checkpoint-selection" in second["bootstrap_summary.json"]["checkpoint_selection_disclosure"]


def test_four_seed_summaries_support_side_by_side_comparison(scored):
    _, result = scored
    records = [
        *result["classification.json"]["experts"].values(),
        *result["classification.json"]["policies"].values(),
        *result["router_metrics.json"].values(),
    ]
    for record in records:
        assert set(record["per_seed"]) == {"1", "2", "3", "4"}
        assert record["pooled_seed_realizations"]["row_count"] == 32
        assert all(seed["row_count"] == 8 for seed in record["per_seed"].values())
        for name, summary in record["seed_summary"].items():
            assert summary["seed_count"] == 4
            values = [seed[name] for seed in record["per_seed"].values() if seed[name] is not None]
            if values:
                assert summary["mean"] == np.mean(values)
    for policy in result["policy_metrics.json"].values():
        assert policy["training_seeds"] == [1, 2, 3, 4]
        assert policy["mean_seed_delta_accuracy"] == np.mean([
            seed["delta_accuracy"] for seed in policy["per_seed"].values()
        ])


def test_never_action_all_router_variants_retain_raw(scored):
    inputs, _ = scored
    experts, routers, actions, labels = [frame.copy(deep=True) for frame in inputs]
    never = json.dumps({"kind": "never_intervene", "value_hex": None, "comparison": "strict_gt"})
    router_rows = actions.policy_id.isin(report.VARIANTS)
    actions.loc[router_rows, "threshold_specification"] = never
    actions.loc[router_rows, "acted"] = False
    actions.loc[router_rows, "final_pred"] = actions.loc[router_rows, "raw_pred"]
    result = report.evaluate_family(experts, routers, actions, labels)
    raw = result["classification.json"]["experts"]["raw"]
    for variant in report.VARIANTS:
        assert result["classification.json"]["policies"][variant] == raw
        assert result["policy_metrics.json"][variant]["pooled"]["acted_count"] == 0
    bootstrap = result["bootstrap_replicates.parquet"].to_pandas()
    assert bootstrap.loc[bootstrap.policy.isin(report.VARIANTS), "mean_seed_delta_accuracy"].eq(0).all()


def test_agreeing_wrong_experts_keep_raw_probability_for_top3():
    experts, routers, actions, labels = _inputs()
    for expert, ranks in (("raw", [4, 0, 1]), ("geo", [4, 2, 3])):
        probability = np.full(18, .01, dtype=np.float64)
        probability[ranks] = [.55, .20, .10]
        for basis in ("native_t1", "calibrated"):
            experts.at[0, f"{expert}_prob_{basis}"] = probability.copy()
    calibrated = {name: np.stack(experts[f"{name}_prob_calibrated"]) for name in ("image", "geo", "raw")}
    for name, values in report.build_semantic_values(calibrated).items():
        experts[name] = values
    for name in report.POLICIES:
        chosen = actions.policy_id.eq(name)
        if name in report.VARIANTS:
            score = routers.loc[routers.variant_id.eq(name), "score"].to_numpy()
        elif name == "never_intervene":
            score = np.zeros(len(experts), dtype=np.float64)
        elif name == "always_geo_on_geo_raw_disagreement":
            score = np.ones(len(experts), dtype=np.float64)
        else:
            column = "geo_minus_raw_confidence" if name.startswith("higher_calibrated") else name
            score = experts[column].to_numpy()
        specification = json.loads(actions.loc[chosen, "threshold_specification"].iloc[0])
        mask = apply_heuristic_threshold(score, experts.raw_pred.to_numpy(), experts.geo_pred.to_numpy(), specification)
        actions.loc[chosen, "score"] = score
        actions.loc[chosen, "raw_pred"] = experts.raw_pred.to_numpy()
        actions.loc[chosen, "geo_pred"] = experts.geo_pred.to_numpy()
        actions.loc[chosen, "acted"] = mask
        actions.loc[chosen, "final_pred"] = np.where(mask, experts.geo_pred, experts.raw_pred)
    result = report.evaluate_family(experts, routers, actions, labels)
    metric = result["classification.json"]["policies"]["always_geo_on_geo_raw_disagreement"]["per_seed"]["1"]
    seed_one = experts.training_seed.eq(1).to_numpy()
    selected = calibrated["geo"].copy()
    selected[0] = calibrated["raw"][0]
    y = result["scoring_targets.parquet"].to_pandas().label_id_dense.to_numpy()
    expected = np.any(np.argsort(-selected[seed_one], axis=1, kind="stable")[:, :3] == y[seed_one, None], axis=1).mean()
    assert metric["top3_accuracy"] == expected
    all_geo = np.any(np.argsort(-calibrated["geo"][seed_one], axis=1, kind="stable")[:, :3] == y[seed_one, None], axis=1).mean()
    assert metric["top3_accuracy"] != all_geo


@pytest.mark.parametrize("problem", [
    "label_guard", "expert_order", "router_alignment", "action_alignment", "missing_seed",
    "label_ontology", "label_float", "file_identity", "probability_dtype", "probability_range",
    "semantic", "nonbool_action", "changed_action", "changed_final", "changed_score",
    "changed_fixed_threshold", "duplicate_threshold_key", "router_probability", "unknown_policy",
    "family_mismatch", "mixed_family", "unknown_family",
])
def test_rejects_corrupt_or_label_contaminated_inputs(problem):
    experts, routers, actions, labels = _inputs()
    if problem == "label_guard":
        experts["target_state"] = "rescue"
    elif problem == "expert_order":
        experts = experts.iloc[::-1].reset_index(drop=True)
    elif problem == "router_alignment":
        routers.loc[0, "plot_idx"] = "wrong"
    elif problem == "action_alignment":
        actions.loc[0, "row_uid"] = "missing-image"
    elif problem == "missing_seed":
        experts = experts.loc[experts.training_seed.ne(4)]
    elif problem == "label_ontology":
        labels.loc[0, "label_name"] = "invented habitat"
    elif problem == "label_float":
        labels["label_id_dense"] = labels.label_id_dense.astype(float)
    elif problem == "file_identity":
        labels.loc[0, "file"] = "other-file.jpg"
    elif problem == "probability_dtype":
        experts["raw_prob_calibrated"] = experts.raw_prob_calibrated.map(lambda p: p.astype(np.float32))
    elif problem == "probability_range":
        experts.at[0, "raw_prob_calibrated"] = np.full(18, 1.0, dtype=np.float64)
    elif problem == "semantic":
        experts.loc[0, "raw_rank_of_geo_pred"] = 19
    elif problem == "nonbool_action":
        actions["acted"] = actions.acted.astype(int)
    elif problem == "changed_action":
        actions.loc[0, "acted"] = True
    elif problem == "changed_final":
        actions.loc[0, "final_pred"] = 17
    elif problem == "changed_score":
        actions.loc[0, "score"] += 0.1
    elif problem == "changed_fixed_threshold":
        actions.loc[actions.policy_id.eq("never_intervene"), "threshold_specification"] = json.dumps({"kind": "finite", "value_hex": 0.0.hex(), "comparison": "strict_gt"})
    elif problem == "duplicate_threshold_key":
        actions.loc[actions.policy_id.eq("primary_full727"), "threshold_specification"] = '{"kind":"never_intervene","kind":"never_intervene","value_hex":null,"comparison":"strict_gt"}'
    elif problem == "router_probability":
        routers.loc[0, "q_rescue"] = np.nan
    elif problem == "unknown_policy":
        actions.loc[actions.policy_id.eq("image_geo_jsd"), "policy_id"] = "new_test_selected_policy"
    elif problem == "family_mismatch":
        routers["checkpoint_family"] = "oracle_test_best_model"
    elif problem == "mixed_family":
        experts.loc[0, "checkpoint_family"] = "oracle_test_best_model"
    elif problem == "unknown_family":
        experts["checkpoint_family"] = "test_selected_extra_family"
    with pytest.raises(report.LegacyRouterReportError):
        report.evaluate_family(experts, routers, actions, labels)


def test_m3_semantics_and_ontology_parity():
    # These existing builders are imported only in this synthetic parity check;
    # the report itself must remain usable in an import-restricted process.
    from multimodal.geo_helpfulness_protocol import FIXED_CLASS_ONTOLOGY
    from multimodal.geo_helpfulness_targets_features import build_router_feature_frame

    experts, _, _, _ = _inputs()
    expected = build_router_feature_frame(
        *(np.stack(experts[f"{name}_prob_calibrated"]) for name in ("image", "geo", "raw")),
        probability_basis="scalar_temperature_calibrated",
    )
    pd.testing.assert_frame_equal(experts.loc[:, list(expected.columns)], expected, check_dtype=False)
    assert report.ONTOLOGY == FIXED_CLASS_ONTOLOGY


def test_report_import_does_not_load_fitting_libraries():
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn'}:
            raise AssertionError('forbidden reporting import: ' + fullname)
sys.meta_path.insert(0, Guard())
import multimodal.geo_helpfulness_legacy_router_report
assert not {'torch', 'scipy', 'sklearn'}.intersection(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

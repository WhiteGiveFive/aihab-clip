"""Synthetic-only scoring, threshold replay and uncertainty tests."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_relaxed_margin_report as report
from multimodal import geo_helpfulness_legacy_router_report as legacy
from multimodal import geo_helpfulness_router_heuristics as heuristics
from test_multimodal_geo_legacy_router_report import _inputs as legacy_inputs


def inputs():
    experts, _, _, labels = legacy_inputs()
    # Vary the calibrated margin while retaining all native expert predictions.
    for i in range(len(experts)):
        for name in ("image", "geo", "raw"):
            pred = int(experts.iloc[i][f"{name}_pred"])
            confidence = (0.55 + (i % 3) * 0.1) if name == "geo" else 0.45
            p = np.full(18, (1 - confidence - 0.2) / 16, dtype=np.float64)
            p[pred], p[(pred + 1) % 18] = confidence, 0.2
            experts.at[i, f"{name}_prob_calibrated"] = p
    probabilities = {
        name: np.stack(experts[f"{name}_prob_calibrated"])
        for name in ("image", "geo", "raw")
    }
    for column, values in legacy.build_semantic_values(probabilities).items():
        experts[column] = values
    original = {"kind": "finite", "comparison": "strict_gt", "value_hex": (0.25).hex()}
    relaxed = {"kind": "finite", "comparison": "strict_gt", "value_hex": (0.1).hex()}
    actions = []
    for name, spec in zip(report.POLICIES, (original, relaxed)):
        frame = experts.loc[:, [*legacy.IDENTITY, "checkpoint_family"]].copy()
        frame["schema_version"] = "geo_helpfulness.relaxed_margin_actions.v1"
        frame["study_id"] = report.STUDY_ID
        frame["policy_id"] = name
        frame["score"] = experts.geo_minus_raw_margin.to_numpy()
        frame["threshold_specification"] = json.dumps(spec, sort_keys=True)
        frame["raw_pred"], frame["geo_pred"] = experts.raw_pred, experts.geo_pred
        acted = legacy.apply_heuristic_threshold(
            frame.score.to_numpy(),
            frame.raw_pred.to_numpy(),
            frame.geo_pred.to_numpy(),
            spec,
        )
        frame["acted"], frame["final_pred"] = acted, np.where(
            acted, frame.geo_pred, frame.raw_pred
        )
        actions.append(frame.loc[:, list(report.ACTION_COLUMNS)])
    ontology = [
        {"dense_id": dense, "canonical_l3_id": canonical, "label_name": name}
        for dense, canonical, name in legacy.ONTOLOGY
    ]
    return (
        experts,
        pd.concat(actions, ignore_index=True),
        labels,
        ontology,
        original,
        relaxed,
    )


@pytest.fixture(scope="module")
def scored():
    args = inputs()
    return args, report.build_report(*args)


def test_contract_and_context(scored):
    _, result = scored
    assert set(result) == set(report.SCORE_FILES)
    for name, value in result.items():
        if name.endswith(".json"):
            json.dumps(value, allow_nan=False)
        else:
            assert isinstance(value, pa.Table)
            assert value.schema.metadata is None
    summary = result["summary.json"]
    assert (summary["row_count"], summary["image_count"], summary["plot_count"]) == (
        32,
        8,
        4,
    )
    assert summary["policy_count"] == 2
    assert summary["checkpoint_family"] == "best_model"
    assert summary["independent_confirmation"] is False
    assert summary["test_guardrail_feasibility_enforced"] is False
    assert summary["M4_gate_rerun"] is False
    assert summary["original_M4_no_go_preserved"] is True
    assert "after earlier cleaned-test results" in summary["selection_disclosure"]
    assert set(result["classification.json"]["policies"]) == set(report.POLICIES)


def test_classification_weighted_f1_outcome_and_seed_math(scored):
    (experts, _, _, _, _, _), result = scored
    targets = result["scoring_targets.parquet"].to_pandas()
    outcomes = result["actions_with_outcomes.parquet"].to_pandas()
    for name in report.POLICIES:
        frame = outcomes.loc[outcomes.policy_id.eq(name)]
        acted = frame.acted.to_numpy()
        pred, y = frame.final_pred.to_numpy(), targets.label_id_dense.to_numpy()
        record = result["classification.json"]["policies"][name]
        reference = legacy.classification_metrics(pred, y, 18)
        for field in ("accuracy", "weighted_f1", "confusion_matrix", "macro_f1"):
            assert record["pooled_seed_realizations"][field] == reference[field]
        for metric in ("accuracy", "weighted_f1"):
            assert record["seed_summary"][metric]["mean"] == np.mean(
                [
                    legacy.classification_metrics(
                        pred[targets.training_seed.eq(seed)],
                        y[targets.training_seed.eq(seed)],
                        18,
                    )[metric]
                    for seed in (1, 2, 3, 4)
                ]
            )
        assert len(record["pooled_seed_realizations"]["per_class"]) == 18
        expected_policy = legacy.policy_metrics(targets, acted)
        for field in ("per_seed", "pooled", "mean_seed_delta_accuracy"):
            assert result["policy_metrics.json"][name][field] == expected_policy[field]
        np.testing.assert_array_equal(
            frame.net_correct,
            frame.final_correct.astype(int) - frame.raw_correct.astype(int),
        )
        assert int(frame.net_correct.sum()) == int(
            (pred == y).sum() - (experts.raw_pred.to_numpy() == y).sum()
        )


def test_shared_whole_plot_bootstrap_and_paired_difference(scored):
    _, result = scored
    bootstrap = result["bootstrap_summary.json"]
    assert (bootstrap["replicates"], bootstrap["random_seed"]) == (2000, 20260826)
    assert bootstrap["unit"] == "complete_plot_idx"
    assert bootstrap["stratify_by"] == "canonical_l3_id"
    assert bootstrap["retain_all_images_and_training_seeds"] is True
    assert bootstrap["baseline_policy"] == "never_intervene"
    assert set(bootstrap["policies"]) == {*report.POLICIES, "never_intervene"}
    assert "primary_full727" not in json.dumps(bootstrap)
    replicates = result["bootstrap_replicates.parquet"].to_pandas()
    assert len(replicates) == 6000
    wide = replicates.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    difference = wide[report.RELAXED_POLICY] - wide[report.ORIGINAL_POLICY]
    paired = result["paired_comparisons.json"]["interval"]
    assert paired["lower"] == np.quantile(difference, 0.025)
    assert paired["upper"] == np.quantile(difference, 0.975)
    assert result["summary.json"]["relaxed_minus_original_interval"] == paired


@pytest.mark.parametrize(
    "mutation",
    [
        "family",
        "label_leak",
        "score",
        "action",
        "final",
        "threshold",
        "extra",
        "missing",
        "duplicate",
        "ordering",
        "dtype",
        "probability",
        "ontology",
        "labels",
    ],
)
def test_rejects_tampering_and_boundary_violations(mutation):
    experts, actions, labels, ontology, original, relaxed = inputs()
    if mutation == "family":
        experts["checkpoint_family"] = "oracle_test_best_model"
    elif mutation == "label_leak":
        experts["label_diagnostic"] = 1
    elif mutation == "score":
        actions.loc[0, "score"] += 0.01
    elif mutation == "action":
        actions.loc[0, "acted"] = not actions.loc[0, "acted"]
    elif mutation == "final":
        actions.loc[0, "final_pred"] = (actions.loc[0, "final_pred"] + 1) % 18
    elif mutation == "threshold":
        original["value_hex"] = (0.7).hex()
    elif mutation == "extra":
        actions["harmless_extra"] = 1
    elif mutation == "missing":
        actions = actions.iloc[1:]
    elif mutation == "duplicate":
        actions = pd.concat([actions, actions.iloc[[0]]], ignore_index=True)
    elif mutation == "ordering":
        experts = experts.iloc[::-1]
    elif mutation == "dtype":
        actions["acted"] = actions.acted.astype(int)
    elif mutation == "probability":
        experts.at[0, "geo_prob_calibrated"] = np.ones(18, dtype=np.float64)
    elif mutation == "ontology":
        ontology[0]["label_name"] = "bad"
    elif mutation == "labels":
        labels.loc[0, "plot_idx"] = "wrong"
    with pytest.raises(report.RelaxedMarginReportError):
        report.build_report(experts, actions, labels, ontology, original, relaxed)


def test_frozen_never_intervene_and_strict_boundary():
    experts, actions, labels, ontology, original, relaxed = inputs()
    original = {"kind": "never_intervene", "comparison": "strict_gt", "value_hex": None}
    boundary = float(experts.geo_minus_raw_margin.max())
    relaxed = {"kind": "finite", "comparison": "strict_gt", "value_hex": boundary.hex()}
    for name, spec in zip(report.POLICIES, (original, relaxed)):
        mask = actions.policy_id.eq(name)
        actions.loc[mask, "threshold_specification"] = json.dumps(spec)
        actions.loc[mask, "acted"] = False
        actions.loc[mask, "final_pred"] = actions.loc[mask, "raw_pred"]
    result = report.build_report(experts, actions, labels, ontology, original, relaxed)
    for name in report.POLICIES:
        assert (
            result["classification.json"]["policies"][name]
            == result["classification.json"]["experts"]["raw"]
        )
        assert result["policy_metrics.json"][name]["pooled"]["acted_count"] == 0


def test_apply_only_selection_disabled_no_mutation(scored, monkeypatch):
    args, previous = scored
    copies = [deepcopy(arg) for arg in args]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("selector or file I/O reached")

    monkeypatch.setattr(heuristics, "enumerate_threshold_candidates", forbidden)
    monkeypatch.setattr(heuristics, "select_oof_heuristic_thresholds", forbidden)
    monkeypatch.setattr(heuristics, "_select_score", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    result = report.build_report(*args)
    assert result["summary.json"] == previous["summary.json"]
    for original, copy in zip(args, copies):
        if isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(original, copy)
        else:
            assert original == copy


def test_clean_import_without_training_libraries():
    code = "import sys; from multimodal import geo_helpfulness_relaxed_margin_report; assert not any(x in sys.modules for x in ('torch', 'sklearn', 'scipy.optimize'))"
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr

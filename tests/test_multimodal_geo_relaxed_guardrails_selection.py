"""Synthetic checks for the additive development-only LR guardrail study."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_relaxed_guardrails_selection as selection
from multimodal import geo_helpfulness_router_selection as original
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256


def rows_fixture(n=40):
    return pd.DataFrame(
        [
            {
                "row_uid": f"row-{index:05d}",
                "plot_idx": f"plot-{index:05d}",
                "training_seed": seed,
                "target_state": "both_wrong",
                "raw_pred": 0,
                "geo_pred": 1,
                "score": 0.0,
            }
            for seed in original.SEEDS
            for index in range(n)
        ]
    )


def mark(rows, indices, state="rescue", score=0.5):
    mask = rows.row_uid.isin([f"row-{index:05d}" for index in indices])
    rows.loc[mask, ["target_state", "score"]] = [state, score]
    if state == "both_correct":
        rows.loc[mask, "geo_pred"] = 0


def random_rows():
    rows = rows_fixture(80)
    rng = np.random.default_rng(963)
    rows["score"] = rng.uniform(-1, 1, len(rows))
    rows["target_state"] = rng.choice(
        original.TARGET_ORDER, size=len(rows), p=[0.45, 0.2, 0.1, 0.25]
    )
    rows.loc[rows.target_state.eq("both_correct"), "geo_pred"] = 0
    return rows


def brute_reference(rows, plots, numerator):
    winner, best, curve = deepcopy(original.NEVER_THRESHOLD), None, []
    for specification in original.enumerate_threshold_candidates(rows.score.to_numpy()):
        actions = original.apply_heuristic_threshold(
            rows.score.to_numpy(),
            rows.raw_pred.to_numpy(),
            rows.geo_pred.to_numpy(),
            specification,
        )
        acted = rows.loc[actions]
        rescue = int(acted.target_state.eq("rescue").sum())
        harm = int(acted.target_state.eq("harm").sum())
        count, support = int(actions.sum()), int(acted.plot_idx.nunique())
        feasible = (
            count * 100 >= len(rows)
            and support >= plots
            and 10 * harm <= numerator * rescue
            and rescue > harm
        )
        curve.append((count, rescue, harm, support, feasible))
        if feasible:
            rank = (
                rescue - harm,
                -harm,
                -count,
                original.validate_threshold_specification(specification),
            )
            if best is None or rank > best:
                best, winner = rank, specification
    return winner, curve


def select(rows, plots=0, numerator=9):
    return selection.select_common_threshold(
        rows, minimum_plots=plots, harm_numerator=numerator
    )


def test_grid_matches_brute_force_and_original_baseline_without_mutation():
    rows = random_rows()
    before = rows.copy(deep=True)
    assert len(selection.GUARDRAILS) == 9
    for guardrail in selection.GUARDRAILS:
        plots, numerator = guardrail["minimum_plots"], guardrail["harm_numerator"]
        result = select(rows, plots, numerator)
        winner, curve = brute_reference(rows, plots, numerator)
        assert result["guardrails"] == guardrail
        assert result["threshold_specification"] == winner
        assert result["candidate_count"] == len(curve)
        assert [
            (
                r["acted_count"],
                r["rescued_count"],
                r["harmed_count"],
                r["unique_acted_plots"],
                r["feasible"],
            )
            for r in result["candidate_curve"]
        ] == curve
        assert select(rows.sample(frac=1, random_state=14), plots, numerator) == result
        json.dumps(result, allow_nan=False)
    baseline = select(rows, 20, 5)
    baseline.pop("guardrails")
    assert baseline == original.select_common_threshold(rows)
    pd.testing.assert_frame_equal(rows, before)


@pytest.mark.parametrize("numerator", [5, 7, 9])
def test_exact_harm_boundary_is_inclusive_and_one_extra_harm_fails(numerator):
    rows = rows_fixture()
    mark(rows, range(10))
    mark(rows, range(10, 10 + numerator), "harm")
    result = select(rows, numerator=numerator)
    assert result["selected_feasible"]
    counts = result["pooled"]
    assert 10 * counts["harmed_count"] == numerator * counts["rescued_count"]
    rows.loc[rows.index[-1], ["target_state", "score"]] = ["harm", 0.5]
    failed = select(rows, numerator=numerator)
    assert not failed["selected_feasible"]
    assert failed["threshold_specification"] == original.NEVER_THRESHOLD
    assert failed["pooled"]["acted_count"] == 0


def test_coverage_is_all_seed_rows_and_support_is_unique_plot_union():
    rows = rows_fixture(2000)
    mark(rows, range(20))
    result = select(rows, 20, 5)
    assert result["pooled"]["coverage_fraction"] == 0.01
    assert result["pooled"]["acted_count"] == 80
    assert result["pooled"]["unique_acted_plots"] == 20
    assert result["selected_feasible"]
    below = rows.copy()
    below.loc[0, "score"] = 0.0
    assert not select(below, 0, 9)["selected_feasible"]
    rows.loc[rows.plot_idx.eq("plot-00019"), "plot_idx"] = "plot-00018"
    assert not select(rows, 20, 5)["selected_feasible"]
    assert select(rows, 10, 5)["selected_feasible"]
    rows["plot_idx"] = "one-shared-plot"
    assert not select(rows, 10, 9)["selected_feasible"]
    assert select(rows, 0, 9)["pooled"]["unique_acted_plots"] == 1


def test_seed_stability_and_heuristic_gate_are_not_threshold_filters():
    rows = rows_fixture(1000)
    selected = rows.training_seed.eq(1) & rows.row_uid.isin(
        [f"row-{index:05d}" for index in range(40)]
    )
    rows.loc[selected, ["target_state", "score"]] = ["rescue", 0.5]
    result = select(rows, 20, 5)
    assert result["selected_feasible"]
    assert result["pooled"]["coverage_fraction"] == 0.01
    assert result["mean_seed_net_utility"] == 10
    assert result["per_seed"]["1"]["rescued_minus_harmed"] == 40
    assert all(result["per_seed"][str(seed)]["acted_count"] == 0 for seed in (2, 3, 4))


def test_ties_prefer_fewer_harms_then_actions_then_higher_threshold():
    rows = rows_fixture()
    mark(rows, range(20), score=0.9)
    mark(rows, [20], score=0.3)
    mark(rows, [21], "harm", score=0.3)
    mark(rows, [22], "both_wrong", score=0.4)
    mark(rows, [23], "both_correct", score=0.6)
    result = select(rows)
    assert result["threshold_display"] == 0.75
    assert result["pooled"]["rescued_minus_harmed"] == 80
    assert result["pooled"]["harmed_count"] == 0
    assert result["pooled"]["acted_count"] == 80
    assert result["candidate_curve"][1]["rescued_minus_harmed"] == 80


def test_strict_gt_excludes_equality_and_agreement_and_never_uses_negative_threshold():
    rows = rows_fixture()
    mark(rows, range(20), score=0.5)
    mark(rows, [20], "both_correct", score=float(np.nextafter(0.5, 1.0)))
    result = select(rows)
    equal = next(
        record
        for record in result["candidate_curve"]
        if record["threshold_display"] == 0.5
    )
    assert equal["acted_count"] == 0
    assert result["pooled"]["acted_count"] == 80
    rows["score"] = -0.1
    result = select(rows)
    assert result["candidate_count"] == 2
    assert result["feasible_candidate_count"] == 0
    assert result["threshold_specification"] == original.NEVER_THRESHOLD
    assert all(
        record["threshold_display"] is None or record["threshold_display"] >= 0
        for record in result["candidate_curve"]
    )


def test_zero_plot_minimum_still_requires_positive_net_and_real_coverage():
    rows = rows_fixture()
    rows["score"] = 0.5
    result = select(rows)
    assert not result["selected_feasible"]
    assert result["candidate_curve"][1]["coverage_fraction"] == 1.0
    assert result["threshold_display"] is None
    assert result["selection_reason"] == "no_feasible_candidate_never_intervene"


def test_feasible_candidate_sets_are_monotone_as_each_guardrail_relaxes():
    rows = random_rows()
    sets = {}
    for plots in (20, 10, 0):
        for numerator in (5, 7, 9):
            sets[plots, numerator] = {
                record["candidate_index"]
                for record in select(rows, plots, numerator)["candidate_curve"]
                if record["feasible"]
            }
    for numerator in (5, 7, 9):
        assert sets[20, numerator] <= sets[10, numerator] <= sets[0, numerator]
    for plots in (20, 10, 0):
        assert sets[plots, 5] <= sets[plots, 7] <= sets[plots, 9]


@pytest.mark.parametrize(
    "plots,numerator",
    [(True, 5), (20.0, 5), (-1, 5), (5, 5), (0, True), (0, 0.7), (0, 10), (0, 6)],
)
def test_undeclared_guardrail_values_are_rejected(plots, numerator):
    with pytest.raises(selection.RouterSelectionError):
        select(rows_fixture(), plots, numerator)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "seed",
        "duplicate",
        "score32",
        "score_nan",
        "plot",
        "agreement",
        "target",
        "prediction",
    ],
)
def test_malformed_selection_rows_are_rejected(damage):
    rows = rows_fixture()
    if damage == "missing":
        rows = rows.drop(columns="score")
    elif damage == "seed":
        rows = rows.loc[rows.training_seed.ne(4)]
    elif damage == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    elif damage == "score32":
        rows["score"] = rows.score.astype(np.float32)
    elif damage == "score_nan":
        rows.loc[0, "score"] = np.nan
    elif damage == "plot":
        rows.loc[0, "plot_idx"] = "different-plot"
    elif damage == "agreement":
        rows.loc[0, ["target_state", "geo_pred"]] = ["rescue", 0]
    elif damage == "target":
        rows.loc[0, "target_state"] = "unknown"
    else:
        rows.loc[0, "geo_pred"] = 18
    with pytest.raises(selection.RouterSelectionError):
        select(rows)


def sealed_inputs():
    document = {
        "schema_version": "geo_helpfulness.m4_selection.v1",
        "protocol_id": "protocol_v1",
        "variants": {},
    }
    tables = []
    for index, variant in enumerate(original.VARIANTS):
        rows = rows_fixture()
        mark(rows, range(20), score=0.5 + index * 0.1)
        threshold = original.select_common_threshold(rows)
        C = original.C_VALUES[index]
        document["variants"][variant] = {
            "regularization": {
                "selected_C": C,
                "selected_C_hex": C.hex(),
                "synthetic_provenance": "copied unchanged",
            },
            "threshold": threshold,
        }
        rows["policy_id"] = variant
        rows["threshold_specification"] = json.dumps(
            threshold["threshold_specification"]
        )
        rows["acted"] = original.apply_heuristic_threshold(
            rows.score.to_numpy(),
            rows.raw_pred.to_numpy(),
            rows.geo_pred.to_numpy(),
            threshold["threshold_specification"],
        )
        tables.append(rows)
    document["selection_sha256"] = canonical_sha256(document)
    return document, pd.concat(tables, ignore_index=True)


def test_build_authenticates_baselines_and_preserves_all_c_choices_and_curves(
    monkeypatch,
):
    document, actions = sealed_inputs()
    before, before_actions = deepcopy(document), actions.copy(deep=True)
    monkeypatch.setattr(
        original, "select_common_c", lambda *args: pytest.fail("must inherit C")
    )
    result = selection.build_selection(document, actions)
    assert result["study_id"] == "relaxed_guardrails_lr_v1"
    assert len(result["policies"]) == 27
    assert result["original_baseline_parity"] == {v: True for v in original.VARIANTS}
    assert result["original_selection_sha256"] == document["selection_sha256"]
    for name, policy in result["policies"].items():
        assert name == selection.policy_id(policy["variant"], policy["guardrail_id"])
        saved = document["variants"][policy["variant"]]
        assert policy["regularization"] == saved["regularization"]
        assert policy["threshold"]["candidate_count"] == len(
            policy["threshold"]["candidate_curve"]
        )
        if policy["guardrail_id"] == "plots_20__harm_0p5":
            assert {
                k: v for k, v in policy["threshold"].items() if k != "guardrails"
            } == saved["threshold"]
    assert (
        canonical_sha256({k: v for k, v in result.items() if k != "selection_sha256"})
        == result["selection_sha256"]
    )
    assert (
        selection.build_selection(document, actions.sample(frac=1, random_state=6))
        == result
    )
    assert document == before
    pd.testing.assert_frame_equal(actions, before_actions)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "curve",
        "c",
        "variant",
        "actions",
        "spec",
        "missing",
        "missing_column",
        "targets",
        "unknown_policy",
    ],
)
def test_build_rejects_tampered_or_incompatible_original_evidence(damage):
    document, actions = sealed_inputs()
    variant = original.VARIANTS[0]
    if damage == "hash":
        document["selection_sha256"] = "0" * 64
    elif damage == "curve":
        document["variants"][variant]["threshold"]["candidate_curve"][1][
            "rescued_count"
        ] += 1
    elif damage == "c":
        document["variants"][variant]["regularization"]["selected_C_hex"] = (10.0).hex()
    elif damage == "variant":
        del document["variants"][variant]
    elif damage == "actions":
        actions.loc[0, "acted"] = not actions.loc[0, "acted"]
    elif damage == "spec":
        actions.loc[0, "threshold_specification"] = json.dumps(original.NEVER_THRESHOLD)
    elif damage == "missing":
        actions = actions.loc[actions.policy_id.ne(variant)]
    elif damage == "missing_column":
        actions = actions.drop(columns="score")
    elif damage == "targets":
        actions.loc[
            actions.policy_id.eq(variant) & actions.row_uid.eq("row-00000"), "plot_idx"
        ] = "changed"
    else:
        actions.loc[0, "policy_id"] = "unknown"
    if damage in {"curve", "c", "variant"}:
        document["selection_sha256"] = canonical_sha256(
            {k: v for k, v in document.items() if k != "selection_sha256"}
        )
    with pytest.raises(selection.RouterSelectionError):
        selection.build_selection(document, actions)


def test_module_imports_no_fitting_or_test_evaluation_stack():
    code = (
        "import sys; import multimodal.geo_helpfulness_relaxed_guardrails_selection; "
        "blocked=('scipy','sklearn','torch','multimodal.geo_helpfulness_router_candidate_fit',"
        "'multimodal.geo_helpfulness_router_dataset','multimodal.geo_helpfulness_oof',"
        "'multimodal.geo_helpfulness_locked_eval','multimodal.geo_helpfulness_router_m4_analysis'); "
        "assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

"""Synthetic RF guardrail selection and frozen configuration provenance checks."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_relaxed_rf_selection as selection
from multimodal import geo_helpfulness_router_selection as original
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256
from test_multimodal_geo_relaxed_guardrails_selection import (
    brute_reference,
    mark,
    random_rows,
    rows_fixture,
)


def select(rows, minimum=0, numerator=9):
    return selection.select_common_threshold(
        rows, minimum_plots=minimum, harm_numerator=numerator
    )


def source_inputs():
    document = {
        "training_seeds": [1, 2, 3, 4],
        "test_data_read_by_scorer": False,
        "fitting_performed_by_scorer": False,
        "original_M4_no_go_preserved": True,
        "independent_confirmation": False,
        "groups": {name: {} for name in selection.CAPACITY_GROUPS},
    }
    frozen = {name: {} for name in selection.SOURCE_POLICY_IDS}
    targets = random_rows()
    tables = []
    for index, variant in enumerate(selection.VARIANTS):
        group = f"rf__{variant}"
        source_id = f"capacity__{group}"
        config = f"{group}__d4_leaf5"
        rows = targets.copy()
        rows["score"] = rows.score * (1.0 - 0.1 * index) if index < 2 else -1.0
        threshold = original.select_common_threshold(rows)
        document["groups"][group] = {
            "group_id": group,
            "family": "rf",
            "variant": variant,
            "selected_config_id": config,
            "selected_policy_id": source_id,
            "candidate_losses": [
                {"config_id": config, "mean_seed_log_loss": 0.5},
                {"config_id": f"{group}__alternative", "mean_seed_log_loss": 0.8},
            ],
            "tie_break": "first_in_predeclared_simplicity_order",
            "threshold": threshold,
        }
        frozen[source_id] = {
            "learner": "rf",
            "variant_id": variant,
            "config_id": config,
            "C_hex": "",
            "threshold_specification": deepcopy(threshold["threshold_specification"]),
            "models": {
                str(seed): {
                    "path": f"/synthetic-never-opened/{config}__seed_{seed}.json",
                    **{
                        field: canonical_sha256([config, seed, field])
                        for field in ("file_sha256", "record_sha256", "state_sha256")
                    },
                }
                for seed in original.SEEDS
            },
        }
        rows["policy_id"] = source_id
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
    return document, pd.concat(tables, ignore_index=True), frozen


def test_all_guardrails_match_independent_bruteforce_and_original_rules():
    rows = random_rows()
    assert len(selection.GUARDRAILS) == 9
    assert selection.GUARDRAILS == selection.shared.GUARDRAILS
    assert selection.GUARDRAILS[0] is not selection.shared.GUARDRAILS[0]
    for guardrail in selection.GUARDRAILS:
        minimum, numerator = guardrail["minimum_plots"], guardrail["harm_numerator"]
        result = select(rows, minimum, numerator)
        winner, counts = brute_reference(rows, minimum, numerator)
        assert result["threshold_specification"] == winner
        assert [
            (
                r["acted_count"],
                r["rescued_count"],
                r["harmed_count"],
                r["unique_acted_plots"],
                r["feasible"],
            )
            for r in result["candidate_curve"]
        ] == counts
        assert (
            select(rows.sample(frac=1, random_state=932), minimum, numerator) == result
        )
        json.dumps(result, allow_nan=False)
    baseline = select(rows, 20, 5)
    baseline.pop("guardrails")
    assert baseline == original.select_common_threshold(rows)


@pytest.mark.parametrize("numerator", [5, 7, 9])
def test_inclusive_exact_harm_caps_and_positive_net_fallback(numerator):
    rows = rows_fixture()
    mark(rows, range(10))
    mark(rows, range(10, 10 + numerator), "harm")
    result = select(rows, numerator=numerator)
    assert result["selected_feasible"]
    pooled = result["pooled"]
    assert 10 * pooled["harmed_count"] == numerator * pooled["rescued_count"]
    rows.loc[rows.index[-1], ["target_state", "score"]] = ["harm", 0.5]
    failed = select(rows, numerator=numerator)
    assert not failed["selected_feasible"]
    assert failed["threshold_specification"] == original.NEVER_THRESHOLD
    assert failed["pooled"]["acted_count"] == 0


def test_pooling_plot_union_strict_threshold_and_tie_order():
    rows = rows_fixture(2000)
    mark(rows, range(20))
    result = select(rows, 20, 5)
    assert result["pooled"]["coverage_fraction"] == 0.01
    assert result["pooled"]["unique_acted_plots"] == 20
    assert result["mean_seed_net_utility"] == 20
    rows["plot_idx"] = "single-shared-plot"
    assert not select(rows, 10, 5)["selected_feasible"]
    assert select(rows, 0, 5)["selected_feasible"]
    rows.loc[0, "score"] = 0.0
    assert not select(rows, 0, 9)["selected_feasible"]
    rows = rows_fixture()
    mark(rows, range(20), score=0.9)
    mark(rows, [20], score=0.3)
    mark(rows, [21], "harm", score=0.3)
    mark(rows, [22], "both_wrong", score=0.4)
    mark(rows, [23], "both_correct", score=0.6)
    result = select(rows)
    assert result["threshold_display"] == 0.75
    assert result["pooled"]["acted_count"] == 80
    assert result["pooled"]["harmed_count"] == 0
    assert result["candidate_curve"][1]["rescued_minus_harmed"] == 80
    rows = rows_fixture()
    mark(rows, range(20), score=0.5)
    mark(rows, [20], "both_correct", score=float(np.nextafter(0.5, 1.0)))
    equal = next(
        r for r in select(rows)["candidate_curve"] if r["threshold_display"] == 0.5
    )
    assert equal["acted_count"] == 0


def test_build_preserves_configurations_models_and_all_original_baselines(monkeypatch):
    document, actions, frozen = source_inputs()
    saved_doc, saved_actions, saved_frozen = (
        deepcopy(document),
        actions.copy(),
        deepcopy(frozen),
    )
    monkeypatch.setattr(
        original, "select_common_c", lambda *a: pytest.fail("no model selection")
    )
    result = selection.build_selection(document, actions, frozen)
    assert result["study_id"] == "relaxed_guardrails_rf_v1"
    assert result["original_selection_sha256"] == canonical_sha256(document)
    assert result["original_frozen_policies_sha256"] == canonical_sha256(frozen)
    assert result["original_baseline_parity"] == {v: True for v in selection.VARIANTS}
    expected_ids = {
        f"relaxed__rf__{variant}__plots_{minimum}__harm_0p{numerator}"
        for variant in selection.VARIANTS
        for minimum in (20, 10, 0)
        for numerator in (5, 7, 9)
    }
    assert set(result["policies"]) == expected_ids
    for name, policy in result["policies"].items():
        group = f"rf__{policy['variant']}"
        source = document["groups"][group]
        source_id = f"capacity__{group}"
        assert name == selection.policy_id(policy["variant"], policy["guardrail_id"])
        assert policy["learner"] == "rf"
        assert policy["config_id"] == source["selected_config_id"]
        assert policy["original_group_id"] == group
        assert policy["original_policy_id"] == source_id
        assert policy["models"] == frozen[source_id]["models"]
        assert policy["source_configuration_selection"] == {
            k: v for k, v in source.items() if k != "threshold"
        }
        threshold = policy["threshold"]
        assert threshold["candidate_count"] == len(threshold["candidate_curve"])
        if policy["guardrail_id"] == "plots_20__harm_0p5":
            assert {k: v for k, v in threshold.items() if k != "guardrails"} == source[
                "threshold"
            ]
        if policy["variant"] == selection.VARIANTS[2]:
            assert threshold["threshold_specification"] == original.NEVER_THRESHOLD
            assert not threshold["selected_feasible"]
            assert all(
                r["threshold_display"] is None or r["threshold_display"] >= 0
                for r in threshold["candidate_curve"]
            )
    assert result["selection_sha256"] == canonical_sha256(
        {k: v for k, v in result.items() if k != "selection_sha256"}
    )
    assert (
        selection.build_selection(
            document, actions.sample(frac=1, random_state=97), frozen
        )
        == result
    )
    assert document == saved_doc and frozen == saved_frozen
    pd.testing.assert_frame_equal(actions, saved_actions)
    json.dumps(result, allow_nan=False)


def test_feasibility_is_monotone_and_non_rf_action_rows_are_unused():
    document, actions, frozen = source_inputs()
    reference = selection.build_selection(document, actions, frozen)
    unused = actions.loc[
        actions.policy_id.eq(f"capacity__rf__{selection.VARIANTS[0]}")
    ].copy()
    unused["policy_id"] = f"capacity__lr__{selection.VARIANTS[0]}"
    unused["score"] = np.nan
    assert (
        selection.build_selection(
            document, pd.concat([actions, unused], ignore_index=True), frozen
        )
        == reference
    )
    for variant in selection.VARIANTS:
        sets = {}
        for minimum in (20, 10, 0):
            for numerator in (5, 7, 9):
                name = selection.policy_id(
                    variant, f"plots_{minimum}__harm_0p{numerator}"
                )
                curve = reference["policies"][name]["threshold"]["candidate_curve"]
                sets[minimum, numerator] = {
                    r["candidate_index"] for r in curve if r["feasible"]
                }
        for numerator in (5, 7, 9):
            assert sets[20, numerator] <= sets[10, numerator] <= sets[0, numerator]
        for minimum in (20, 10, 0):
            assert sets[minimum, 5] <= sets[minimum, 7] <= sets[minimum, 9]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_group",
        "test_source",
        "missing_frozen",
        "family",
        "variant",
        "config",
        "policy_id",
        "frozen_config",
        "frozen_threshold",
        "missing_seed",
        "model_path",
        "model_hash",
        "c_hex",
        "curve",
        "mask",
        "action_spec",
        "missing_policy",
        "missing_column",
        "duplicate",
        "wrong_target",
        "float32",
        "seed",
        "plot",
    ],
)
def test_malformed_provenance_or_actions_fail_before_any_relaxed_selection(
    monkeypatch, damage
):
    document, actions, frozen = source_inputs()
    variant = selection.VARIANTS[-1]
    group, source_id = f"rf__{variant}", f"capacity__rf__{variant}"
    chosen, policy = document["groups"][group], frozen[source_id]
    index = actions.index[actions.policy_id.eq(source_id)][0]
    if damage == "missing_group":
        del document["groups"][group]
    elif damage == "test_source":
        document["test_data_read_by_scorer"] = True
    elif damage == "missing_frozen":
        del frozen[source_id]
    elif damage in {"family", "variant", "policy_id"}:
        chosen[{"policy_id": "selected_policy_id"}.get(damage, damage)] = "wrong"
    elif damage == "config":
        chosen["selected_config_id"] = "rf__other__d4_leaf5"
    elif damage == "frozen_config":
        policy["config_id"] += "_changed"
    elif damage == "frozen_threshold":
        policy["threshold_specification"] = {
            "kind": "finite",
            "value_hex": (0.0).hex(),
            "comparison": "strict_gt",
        }
    elif damage == "missing_seed":
        del policy["models"]["4"]
    elif damage == "model_path":
        policy["models"]["1"]["path"] = policy["models"]["2"]["path"]
    elif damage == "model_hash":
        policy["models"]["1"]["state_sha256"] = "bad"
    elif damage == "c_hex":
        policy["C_hex"] = (0.1).hex()
    elif damage == "curve":
        chosen["threshold"]["candidate_curve"][0]["rescued_count"] += 1
    elif damage == "mask":
        actions.loc[index, "acted"] = not actions.loc[index, "acted"]
    elif damage == "action_spec":
        actions.loc[index, "threshold_specification"] = "invalid-json"
    elif damage == "missing_policy":
        actions = actions.loc[actions.policy_id.ne(source_id)]
    elif damage == "missing_column":
        actions = actions.drop(columns="score")
    elif damage == "duplicate":
        actions = pd.concat([actions, actions.loc[[index]]])
    elif damage == "wrong_target":
        actions.loc[index, "target_state"] = "unknown"
    elif damage == "float32":
        actions["score"] = actions.score.astype(np.float32)
    elif damage == "seed":
        actions = actions.loc[
            ~(actions.policy_id.eq(source_id) & actions.training_seed.eq(4))
        ]
    else:
        mask = actions.policy_id.eq(source_id) & actions.row_uid.eq(
            actions.loc[index, "row_uid"]
        )
        actions.loc[mask, "plot_idx"] = "changed-for-this-variant"
    monkeypatch.setattr(
        selection.shared,
        "_select_from_curve",
        lambda *args: pytest.fail("all original policies must authenticate first"),
    )
    with pytest.raises(selection.RouterSelectionError):
        selection.build_selection(document, actions, frozen)


@pytest.mark.parametrize(
    "minimum,numerator",
    [(True, 5), (20.0, 5), (-1, 5), (5, 5), (0, True), (0, 0.7), (0, 10), (0, 6)],
)
def test_undeclared_guardrails_rejected(minimum, numerator):
    with pytest.raises(selection.RouterSelectionError):
        select(rows_fixture(), minimum, numerator)


def test_module_imports_no_fitting_configuration_search_or_evaluation_stack():
    code = (
        "import sys; import multimodal.geo_helpfulness_relaxed_rf_selection; "
        "blocked=('scipy','sklearn','torch','xgboost',"
        "'multimodal.geo_helpfulness_router_capacity',"
        "'multimodal.geo_helpfulness_router_capacity_fit',"
        "'multimodal.geo_helpfulness_router_capacity_score',"
        "'multimodal.geo_helpfulness_legacy_capacity',"
        "'multimodal.geo_helpfulness_locked_eval'); "
        "assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

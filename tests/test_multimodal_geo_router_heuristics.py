from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_router_heuristics as heuristics


def fixture_rows(n=32, plots=None):
    plots = n if plots is None else plots
    rows = []
    targets = []
    for seed in heuristics.SEEDS:
        for index in range(n):
            identity = {
                "schema_version": "geo_helpfulness.router_targets.v1",
                "protocol_id": "protocol_v1",
                "row_uid": f"row_{index:04}",
                "plot_idx": f"plot_{index % plots:04}",
                "training_seed": seed,
            }
            state = (
                "rescue"
                if index < n - 8
                else ("harm" if index < n - 4 else "both_wrong")
            )
            score = 0.5 if state == "rescue" else (0.1 if state == "harm" else 0.3)
            rows.append(
                {
                    **identity,
                    "train_oof_fold": index % 4,
                    "raw_pred": 0,
                    "geo_pred": 1,
                    "image_geo_jsd": score,
                    "geo_minus_raw_margin": score,
                }
            )
            targets.append({**identity, "target_state": state})
    return pd.DataFrame(rows), pd.DataFrame(targets, columns=heuristics.TARGET_COLUMNS)


def select(audit, targets, **kwargs):
    return heuristics.select_oof_heuristic_thresholds(
        audit,
        targets,
        protocol_id="protocol_v1",
        expected_rows_per_seed=kwargs.pop("expected_rows_per_seed", len(audit) // 4),
        expected_plots_per_seed=kwargs.pop(
            "expected_plots_per_seed", audit.plot_idx.nunique()
        ),
        **kwargs,
    )


def test_known_optimum_is_one_common_threshold_with_pooled_plot_counts():
    audit, targets = fixture_rows()
    result = select(audit, targets)
    assert result["fit_role"] == "development_train_oof"
    assert result["row_count"] == 128
    assert result["unique_images"] == result["unique_plots"] == 32
    assert result["boundary_pooling"] == "all_development_train_oof_seed_realizations"
    assert len(result["fixed_comparators"]) == 3
    for selected in result["heuristics"].values():
        assert selected["threshold_specification"] == {
            "kind": "finite",
            "value_hex": (0.4).hex(),
            "comparison": "strict_gt",
        }
        assert selected["candidate_count"] == 4
        assert selected["selected_feasible"] is True
        assert (
            selected["pooled"]["acted_count"]
            == selected["pooled"]["rescued_count"]
            == 96
        )
        assert selected["pooled"]["harmed_count"] == 0
        assert selected["pooled"]["unique_acted_plots"] == 24  # not 4 * 24
        assert selected["mean_rescued_minus_harmed"] == 24
        assert selected["pooled"]["coverage_fraction"] == 0.75
        assert set(selected["per_seed"]) == {"1", "2", "3", "4"}
        assert all(
            value["rescued_count"] == 24 for value in selected["per_seed"].values()
        )
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["validation_outputs_opened"] is False
    assert result["validation_labels_scored"] is False
    assert result["test_sources_opened"] is False


def test_permutations_preserve_selected_state_and_provenance_hash():
    audit, targets = fixture_rows()
    before_audit, before_targets = audit.copy(deep=True), targets.copy(deep=True)
    first = select(audit, targets)
    second = select(
        audit.sample(frac=1, random_state=5), targets.sample(frac=1, random_state=8)
    )
    assert first == second
    pd.testing.assert_frame_equal(audit, before_audit)
    pd.testing.assert_frame_equal(targets, before_targets)


def test_exact_enumeration_uses_only_positive_values_and_no_maximum_endpoint():
    values = np.array([-1, 0, 0.1, 0.1, 0.3, 0.5], dtype=np.float64)
    specs = heuristics.enumerate_threshold_candidates(values)
    assert specs[0] == heuristics.NEVER_THRESHOLD
    assert [float.fromhex(spec["value_hex"]) for spec in specs[1:]] == [0, 0.2, 0.4]
    assert len(heuristics.enumerate_threshold_candidates(np.array([0.2, 0.2]))) == 2
    assert len(heuristics.enumerate_threshold_candidates(np.array([-1.0, 0.0]))) == 2


def test_adjacent_float_midpoints_and_dedup_are_exact():
    a = np.float64(0.5)
    b = np.nextafter(a, np.float64(1))
    c = np.nextafter(b, np.float64(1))
    values = np.array([a, b, c], dtype=np.float64)
    expected = np.unique(np.r_[0.0, values[:-1] + (values[1:] - values[:-1]) / 2.0])
    specs = heuristics.enumerate_threshold_candidates(values)
    np.testing.assert_array_equal(
        [float.fromhex(spec["value_hex"]) for spec in specs[1:]], expected
    )


def test_application_is_strict_and_requires_disagreement():
    spec = {"kind": "finite", "value_hex": (0.2).hex(), "comparison": "strict_gt"}
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    raw, geo = np.array([0, 0, 0, 1]), np.array([1, 1, 1, 1])
    np.testing.assert_array_equal(
        heuristics.apply_heuristic_threshold(scores, raw, geo, spec),
        [False, False, True, False],
    )
    np.testing.assert_array_equal(
        heuristics.apply_heuristic_threshold(
            scores, raw, geo, heuristics.NEVER_THRESHOLD
        ),
        np.zeros(4, dtype=bool),
    )


def test_tie_prefers_fewer_harms_before_actions():
    audit, targets = fixture_rows()
    indices = np.tile(np.arange(32), 4)
    states = np.where(
        indices < 24, "rescue", np.where(indices < 28, "harm", "both_wrong")
    )
    scores = np.where(indices < 20, 0.5, 0.1)
    targets["target_state"] = states
    audit.loc[:, list(heuristics.SCORES)] = np.column_stack((scores, scores))
    selected = select(audit, targets)["heuristics"]["image_geo_jsd"]
    assert selected["threshold_display"] == pytest.approx(0.3)
    assert selected["pooled"]["rescued_minus_harmed"] == 80
    assert selected["pooled"]["harmed_count"] == 0
    assert selected["pooled"]["acted_count"] == 80


def test_tie_prefers_higher_threshold_and_boundaries_include_nonactionable_rows():
    audit, targets = fixture_rows()
    indices = np.tile(np.arange(32), 4)
    targets["target_state"] = np.where(indices < 20, "rescue", "both_correct")
    audit.loc[indices >= 20, "geo_pred"] = 0
    scores = np.where(indices < 20, 0.6, np.array([0.1, 0.2, 0.4, 0.4])[indices % 4])
    audit.loc[:, list(heuristics.SCORES)] = np.column_stack((scores, scores))
    selected = select(audit, targets)["heuristics"]["image_geo_jsd"]
    assert selected["candidate_count"] == 5
    assert selected["threshold_display"] == 0.5
    assert selected["pooled"]["acted_count"] == 80


def test_seed_counts_are_retained_and_utility_is_mean_of_four_seeds():
    audit, targets = fixture_rows()
    audit.loc[:, list(heuristics.SCORES)] = 0.5
    targets["target_state"] = "rescue"
    targets.loc[targets.training_seed.eq(4), "target_state"] = "both_wrong"
    selected = select(audit, targets)["heuristics"]["image_geo_jsd"]
    assert selected["per_seed"]["4"]["rescued_count"] == 0
    assert selected["pooled"]["rescued_count"] == 96
    assert selected["mean_rescued_minus_harmed"] == 24


@pytest.mark.parametrize("reason", ["plots", "harms", "utility", "negative_scores"])
def test_no_feasible_candidate_returns_typed_never(reason):
    audit, targets = fixture_rows(32, plots=19 if reason == "plots" else 32)
    audit.loc[:, list(heuristics.SCORES)] = 0.5
    if reason == "harms":
        targets.loc[np.tile(np.arange(32), 4) >= 20, "target_state"] = "harm"
    elif reason == "utility":
        targets["target_state"] = "both_wrong"
    elif reason == "negative_scores":
        audit["geo_minus_raw_margin"] = -0.1
    selected = select(audit, targets)["heuristics"]["geo_minus_raw_margin"]
    assert selected["threshold_specification"] == heuristics.NEVER_THRESHOLD
    assert selected["threshold_display"] is None
    assert selected["selected_feasible"] is False
    assert selected["feasible_candidate_count"] == 0
    assert selected["pooled"]["acted_count"] == 0
    assert selected["pooled"]["harmed_to_rescued_ratio"] is None


def test_exact_one_percent_coverage_boundary_uses_all_seed_rows():
    # 20 acted plots, but actions occur only in seed1: 20 / (500*4) == 1%.
    audit, targets = fixture_rows(500)
    targets["target_state"] = "both_wrong"
    audit.loc[:, list(heuristics.SCORES)] = 0.0
    active = audit.training_seed.eq(1) & audit.row_uid.isin(
        [f"row_{i:04}" for i in range(20)]
    )
    audit.loc[active, list(heuristics.SCORES)] = 0.5
    targets.loc[active, "target_state"] = "rescue"
    selected = select(audit, targets)["heuristics"]["image_geo_jsd"]
    assert selected["selected_feasible"] is True
    assert selected["pooled"]["coverage_fraction"] == 0.01
    assert selected["pooled"]["unique_acted_plots"] == 20
    audit2, targets2 = fixture_rows(501)
    targets2["target_state"] = "both_wrong"
    audit2.loc[:, list(heuristics.SCORES)] = 0.0
    active2 = audit2.training_seed.eq(1) & audit2.row_uid.isin(
        [f"row_{i:04}" for i in range(20)]
    )
    audit2.loc[active2, list(heuristics.SCORES)] = 0.5
    targets2.loc[active2, "target_state"] = "rescue"
    assert (
        select(audit2, targets2)["heuristics"]["image_geo_jsd"]["selected_feasible"]
        is False
    )


def test_exact_harm_ratio_boundary_is_inclusive():
    audit, targets = fixture_rows(30)
    audit.loc[:, list(heuristics.SCORES)] = 0.5
    targets["target_state"] = np.tile(["rescue"] * 20 + ["harm"] * 10, 4)
    selected = select(audit, targets)["heuristics"]["image_geo_jsd"]
    assert selected["selected_feasible"] is True
    assert selected["pooled"]["harmed_to_rescued_ratio"] == 0.5


@pytest.mark.parametrize(
    "damage",
    [
        "audit_duplicate",
        "target_duplicate",
        "key_mismatch",
        "plot_mismatch",
        "cross_seed_plot",
        "missing_seed",
        "wrong_protocol",
        "wrong_schema",
        "wrong_target",
        "target_leak",
        "true_label_leak",
        "wrong_fold",
        "cross_seed_fold",
        "float_prediction",
        "high_prediction",
        "float32_score",
        "nan_score",
        "jsd_range",
        "margin_range",
        "rescue_agreement",
        "correct_disagreement",
    ],
)
def test_rejects_invalid_or_leaking_inputs(damage):
    audit, targets = fixture_rows()
    kwargs = {"expected_rows_per_seed": 32, "expected_plots_per_seed": 32}
    if damage == "audit_duplicate":
        audit = pd.concat([audit, audit.iloc[:1]])
    elif damage == "target_duplicate":
        targets = pd.concat([targets, targets.iloc[:1]])
    elif damage == "key_mismatch":
        targets.loc[targets.row_uid.eq("row_0000"), "row_uid"] = "row_9999"
    elif damage == "plot_mismatch":
        targets["plot_idx"] = "target_" + targets.plot_idx
    elif damage == "cross_seed_plot":
        audit.loc[0, "plot_idx"] = "changed"
    elif damage == "missing_seed":
        audit.loc[audit.training_seed.eq(4), "training_seed"] = 3
    elif damage == "wrong_protocol":
        audit.loc[0, "protocol_id"] = "other"
    elif damage == "wrong_schema":
        targets.loc[0, "schema_version"] = "wrong"
    elif damage == "wrong_target":
        targets.loc[0, "target_state"] = "other"
    elif damage in ("target_leak", "true_label_leak"):
        audit["target_state" if damage == "target_leak" else "label_id_dense"] = 0
    elif damage == "wrong_fold":
        audit.loc[0, "train_oof_fold"] = 4
    elif damage == "cross_seed_fold":
        audit.loc[0, "train_oof_fold"] = 1
    elif damage == "float_prediction":
        audit["raw_pred"] = audit.raw_pred.astype(float)
    elif damage == "high_prediction":
        audit.loc[0, "geo_pred"] = 18
    elif damage == "float32_score":
        audit["image_geo_jsd"] = audit.image_geo_jsd.astype(np.float32)
    elif damage == "nan_score":
        audit.loc[0, "image_geo_jsd"] = np.nan
    elif damage == "jsd_range":
        audit.loc[0, "image_geo_jsd"] = 1.0
    elif damage == "margin_range":
        audit.loc[0, "geo_minus_raw_margin"] = -1.1
    elif damage == "rescue_agreement":
        audit.loc[0, "geo_pred"] = audit.loc[0, "raw_pred"]
    else:
        targets.loc[0, "target_state"] = "both_correct"
    with pytest.raises(heuristics.RouterHeuristicError):
        select(audit, targets, **kwargs)


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "finite", "value_hex": "inf", "comparison": "strict_gt"},
        {"kind": "finite", "value_hex": (-0.1).hex(), "comparison": "strict_gt"},
        {"kind": "finite", "value_hex": "0.2", "comparison": "strict_gt"},
        {"kind": "finite", "value_hex": (-0.0).hex(), "comparison": "strict_gt"},
        {"kind": "finite", "value_hex": (0.1).hex(), "comparison": "gte"},
        {
            "kind": "never_intervene",
            "value_hex": (0.1).hex(),
            "comparison": "strict_gt",
        },
    ],
)
def test_rejects_invalid_threshold_specification(spec):
    with pytest.raises(heuristics.RouterHeuristicError):
        heuristics.validate_threshold_specification(spec)


def test_suffix_selection_matches_independent_brute_force():
    audit, targets = fixture_rows(48)
    rng = np.random.default_rng(993)
    audit["image_geo_jsd"] = rng.choice(np.linspace(0, 0.6, 17), len(audit)).astype(
        np.float64
    )
    audit["geo_minus_raw_margin"] = rng.choice(
        np.linspace(-0.3, 0.7, 19), len(audit)
    ).astype(np.float64)
    targets["target_state"] = rng.choice(
        ["rescue", "harm", "both_wrong"], len(targets), p=[0.7, 0.1, 0.2]
    )
    result = select(audit, targets)
    for name in heuristics.SCORES:
        scores = audit[name].to_numpy()
        best, expected = None, deepcopy(heuristics.NEVER_THRESHOLD)
        feasible = 0
        for spec in heuristics.enumerate_threshold_candidates(scores)[1:]:
            threshold = float.fromhex(spec["value_hex"])
            actions = (scores > threshold) & audit.raw_pred.ne(
                audit.geo_pred
            ).to_numpy()
            rescues = int(
                (actions & targets.target_state.eq("rescue").to_numpy()).sum()
            )
            harms = int((actions & targets.target_state.eq("harm").to_numpy()).sum())
            count = int(actions.sum())
            if (
                count * 100 < len(audit)
                or audit.loc[actions].plot_idx.nunique() < 20
                or 2 * harms > rescues
                or rescues <= harms
            ):
                continue
            feasible += 1
            rank = (rescues - harms, -harms, -count, threshold)
            if best is None or rank > best:
                best, expected = rank, spec
        assert result["heuristics"][name]["threshold_specification"] == expected
        assert result["heuristics"][name]["feasible_candidate_count"] == feasible


@pytest.mark.parametrize("fallback", [False, True])
def test_frozen_validator_is_apply_only(monkeypatch, fallback):
    audit, targets = fixture_rows(32, plots=19 if fallback else 32)
    state = select(audit, targets)

    def forbidden(*args, **kwargs):
        pytest.fail("frozen validation must not enumerate or select thresholds")

    for name in (
        "select_oof_heuristic_thresholds",
        "enumerate_threshold_candidates",
        "_select_score",
    ):
        monkeypatch.setattr(heuristics, name, forbidden)
    heuristics.validate_frozen_oof_heuristics(
        state,
        audit,
        targets,
        protocol_id="protocol_v1",
        expected_rows_per_seed=32,
        expected_plots_per_seed=audit.plot_idx.nunique(),
    )


@pytest.mark.parametrize(
    "damage",
    [
        "input_hash",
        "fixed_policy",
        "guardrail",
        "count",
        "threshold_hash",
        "display",
        "candidate_count",
        "candidate_hash",
        "reason",
        "feasible",
    ],
)
def test_frozen_validator_rejects_tampering(damage):
    audit, targets = fixture_rows()
    state = select(audit, targets)
    selected = state["heuristics"]["image_geo_jsd"]
    if damage == "input_hash":
        state["selection_projection_sha256"] = "0" * 64
    elif damage == "fixed_policy":
        state["fixed_comparators"][0]["id"] = "other"
    elif damage == "guardrail":
        state["guardrails"]["min_unique_acted_plots"] = 1
    elif damage == "count":
        selected["pooled"]["rescued_count"] += 1
    elif damage == "threshold_hash":
        selected["threshold_specification_sha256"] = "0" * 64
    elif damage == "display":
        selected["threshold_display"] = 0.3
    elif damage == "candidate_count":
        selected["candidate_count"] = True
    elif damage == "candidate_hash":
        selected["candidate_specifications_sha256"] = "bad"
    elif damage == "reason":
        selected["selection_reason"] = "other"
    else:
        selected["selected_feasible"] = False
    with pytest.raises(heuristics.RouterHeuristicError):
        heuristics.validate_frozen_oof_heuristics(
            state,
            audit,
            targets,
            protocol_id="protocol_v1",
            expected_rows_per_seed=32,
            expected_plots_per_seed=32,
        )

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_relaxed_margin_selection as selection
from multimodal import geo_helpfulness_router_heuristics as original
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256


def fixture_rows(n=128, plots=None):
    plots = n if plots is None else plots
    audit, targets = [], []
    for seed in original.SEEDS:
        for index in range(n):
            identity = {
                "schema_version": "geo_helpfulness.router_targets.v1",
                "protocol_id": "protocol_v1",
                "row_uid": f"row_{index:04d}",
                "plot_idx": f"plot_{index % plots:04d}",
                "training_seed": seed,
            }
            state = "rescue" if index < 70 else "harm" if index < 110 else "both_wrong"
            score = 0.8 if index < 20 else 0.3 if index < 110 else 0.0
            audit.append(
                {
                    **identity,
                    "train_oof_fold": index % 4,
                    "raw_pred": 0,
                    "geo_pred": 1,
                    "image_geo_jsd": 0.4,
                    "geo_minus_raw_margin": score,
                }
            )
            targets.append({**identity, "target_state": state})
    return pd.DataFrame(audit), pd.DataFrame(targets, columns=original.TARGET_COLUMNS)


def kwargs(audit):
    return {
        "protocol_id": "protocol_v1",
        "expected_rows_per_seed": len(audit) // 4,
        "expected_plots_per_seed": audit.plot_idx.nunique(),
    }


def build(audit, targets):
    source = original.select_oof_heuristic_thresholds(audit, targets, **kwargs(audit))
    return selection.build_selection(audit, targets, source, **kwargs(audit)), source


def test_relaxed_selection_retains_original_parity_and_increases_training_utility():
    audit, targets = fixture_rows()
    result, source = build(audit, targets)
    baseline, relaxed = result["policies"]["original"], result["policies"]["relaxed"]
    assert baseline == source["heuristics"][selection.SCORE_NAME]
    assert baseline["threshold_display"] == 0.55
    assert baseline["pooled"]["rescued_minus_harmed"] == 80
    assert relaxed["threshold_display"] == 0.0
    assert relaxed["pooled"]["rescued_count"] == 280
    assert relaxed["pooled"]["harmed_count"] == 160
    assert relaxed["pooled"]["rescued_minus_harmed"] == 120
    assert relaxed["pooled"]["unique_acted_plots"] == 110
    assert relaxed["mean_rescued_minus_harmed"] == 30
    assert all(
        value["rescued_minus_harmed"] == 30 for value in relaxed["per_seed"].values()
    )
    assert result["original_selection_reproduced"] is True
    assert result["fit_role"] == "development_train_oof"
    assert result["fitting_performed"] is False
    assert result["validation_outputs_opened"] is False
    assert result["validation_labels_scored"] is False
    assert result["test_sources_opened"] is False
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_curve_matches_brute_application_at_every_candidate():
    audit, targets = fixture_rows()
    result, _ = build(audit, targets)
    rows = original._join_training_inputs(audit, targets, **kwargs(audit))
    for point in result["candidate_curve"]:
        action = original.apply_heuristic_threshold(
            rows[selection.SCORE_NAME].to_numpy(),
            rows.raw_pred.to_numpy(),
            rows.geo_pred.to_numpy(),
            point["threshold_specification"],
        )
        assert point["pooled"] == original._count_summary(rows, action)
    specs = [point["threshold_specification"] for point in result["candidate_curve"]]
    assert specs == original.enumerate_threshold_candidates(
        audit[selection.SCORE_NAME].to_numpy()
    )
    assert result["candidate_curve_sha256"] == canonical_sha256(
        result["candidate_curve"]
    )


def test_order_is_deterministic_and_inputs_are_unchanged():
    audit, targets = fixture_rows()
    before_audit, before_targets = audit.copy(deep=True), targets.copy(deep=True)
    result, source = build(audit, targets)
    shuffled = selection.build_selection(
        audit.sample(frac=1, random_state=19),
        targets.sample(frac=1, random_state=32),
        source,
        **kwargs(audit),
    )
    assert result == shuffled
    pd.testing.assert_frame_equal(audit, before_audit)
    pd.testing.assert_frame_equal(targets, before_targets)


@pytest.mark.parametrize("extra_harm,feasible", [(0, True), (1, False)])
def test_harm_boundary_is_exact_integer_comparison(extra_harm, feasible):
    audit, targets = fixture_rows(86)
    index = np.tile(np.arange(86), 4)
    targets["target_state"] = np.where(
        index < 50, "rescue", np.where(index < 85 + extra_harm, "harm", "both_wrong")
    )
    audit[selection.SCORE_NAME] = 0.4
    result, _ = build(audit, targets)
    assert result["policies"]["original"]["selected_feasible"] is False
    assert result["policies"]["relaxed"]["selected_feasible"] is feasible
    if feasible:
        assert result["policies"]["relaxed"]["pooled"]["harmed_to_rescued_ratio"] == 0.7


@pytest.mark.parametrize("plots,feasible", [(19, False), (20, True)])
def test_twenty_plot_requirement_is_unchanged(plots, feasible):
    audit, targets = fixture_rows(40, plots=plots)
    audit[selection.SCORE_NAME] = 0.5
    targets["target_state"] = "rescue"
    result, _ = build(audit, targets)
    assert result["policies"]["relaxed"]["selected_feasible"] is feasible


@pytest.mark.parametrize("n,feasible", [(500, True), (501, False)])
def test_one_percent_boundary_retains_four_seed_denominator(n, feasible):
    audit, targets = fixture_rows(n)
    audit[selection.SCORE_NAME] = 0.0
    targets["target_state"] = "both_wrong"
    active = audit.training_seed.eq(1) & audit.row_uid.isin(
        [f"row_{i:04d}" for i in range(20)]
    )
    audit.loc[active, selection.SCORE_NAME] = 0.5
    targets.loc[active, "target_state"] = "rescue"
    result, _ = build(audit, targets)
    selected = result["policies"]["relaxed"]
    assert selected["selected_feasible"] is feasible
    if feasible:
        assert selected["pooled"]["coverage_fraction"] == 0.01
        assert selected["per_seed"]["2"]["acted_count"] == 0


@pytest.mark.parametrize(
    "mode", ["no_positive_scores", "nonpositive_net", "no_disagreement"]
)
def test_typed_never_fallback(mode):
    audit, targets = fixture_rows()
    if mode == "no_positive_scores":
        audit[selection.SCORE_NAME] = -0.3
    elif mode == "nonpositive_net":
        targets["target_state"] = "both_wrong"
    else:
        audit["geo_pred"] = audit.raw_pred
        targets["target_state"] = "both_correct"
    result, _ = build(audit, targets)
    selected = result["policies"]["relaxed"]
    assert selected["threshold_specification"] == original.NEVER_THRESHOLD
    assert selected["pooled"]["acted_count"] == 0
    assert selected["feasible_candidate_count"] == 0


def test_ties_prefer_fewer_harms_then_fewer_actions():
    audit, targets = fixture_rows(40)
    index = np.tile(np.arange(40), 4)
    targets["target_state"] = np.where(
        index < 24, "rescue", np.where(index < 28, "harm", "both_wrong")
    )
    audit[selection.SCORE_NAME] = np.where(
        index < 20, 0.6, np.where(index < 28, 0.1, 0.4)
    )
    result, _ = build(audit, targets)
    selected = result["policies"]["relaxed"]
    assert selected["threshold_display"] == 0.5
    assert selected["pooled"]["rescued_minus_harmed"] == 80
    assert selected["pooled"]["harmed_count"] == 0
    assert selected["pooled"]["acted_count"] == 80


def test_ties_prefer_higher_threshold_using_agreement_row_boundaries():
    audit, targets = fixture_rows(32)
    index = np.tile(np.arange(32), 4)
    targets["target_state"] = np.where(index < 20, "rescue", "both_correct")
    audit.loc[index >= 20, "geo_pred"] = 0
    audit[selection.SCORE_NAME] = np.where(
        index < 20, 0.6, np.array([0.1, 0.2, 0.4, 0.4])[index % 4]
    )
    result, _ = build(audit, targets)
    assert result["policies"]["relaxed"]["threshold_display"] == 0.5
    assert len(result["candidate_curve"]) == 5


def test_adjacent_float_midpoint_rounding_remains_strict_greater_than():
    audit, targets = fixture_rows(40)
    index = np.tile(np.arange(40), 4)
    lower = np.float64(0.5)
    upper = np.nextafter(lower, 1.0)
    audit[selection.SCORE_NAME] = np.where(index < 20, upper, lower)
    targets["target_state"] = np.where(index < 20, "rescue", "harm")
    result, _ = build(audit, targets)
    selected = result["policies"]["relaxed"]
    assert selected["threshold_display"] == 0.5
    assert selected["pooled"]["acted_count"] == 80
    assert selected["pooled"]["harmed_count"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        "label",
        "validation_role",
        "test_source",
        "duplicate",
        "nan",
        "float32",
        "seed",
        "target_key",
    ],
)
def test_invalid_or_nontraining_projection_is_rejected(bad):
    audit, targets = fixture_rows()
    _, source = build(audit, targets)
    if bad in {"label", "validation_role", "test_source"}:
        column, value = {
            "label": ("label_id_dense", 0),
            "validation_role": ("development_role", "validation"),
            "test_source": ("test_file", "cleaned_test.parquet"),
        }[bad]
        audit[column] = value
    elif bad == "duplicate":
        audit.loc[1, "row_uid"] = audit.loc[0, "row_uid"]
    elif bad == "nan":
        audit.loc[0, selection.SCORE_NAME] = np.nan
    elif bad == "float32":
        audit[selection.SCORE_NAME] = audit[selection.SCORE_NAME].astype(np.float32)
    elif bad == "seed":
        audit.loc[0, "training_seed"] = 0
    else:
        targets.loc[0, "row_uid"] = "unexpected_row"
    with pytest.raises(selection.RouterSelectionError):
        selection.build_selection(audit, targets, source, **kwargs(audit))


def test_original_nonoptimal_but_apply_valid_policy_is_rejected():
    audit, targets = fixture_rows()
    _, source = build(audit, targets)
    # Input identity must remain the original saved declaration, not merely a threshold.
    source["heuristics"][selection.SCORE_NAME]["candidate_count"] += 1
    with pytest.raises(selection.RouterSelectionError, match="does not reproduce"):
        selection.build_selection(audit, targets, source, **kwargs(audit))


def test_apply_only_validation_never_reselects_or_enumerates(monkeypatch):
    audit, targets = fixture_rows()
    result, source = build(audit, targets)

    def forbidden(*args, **kwargs):
        raise AssertionError("selection is forbidden in apply-only validation")

    monkeypatch.setattr(original, "_select_score", forbidden)
    monkeypatch.setattr(original, "enumerate_threshold_candidates", forbidden)
    monkeypatch.setattr(selection, "build_selection", forbidden)
    monkeypatch.setattr(selection, "_select_from_curve", forbidden)
    monkeypatch.setattr(selection, "_candidate_curve", forbidden)
    selection.validate_frozen_selection(result, audit, targets, source, **kwargs(audit))


@pytest.mark.parametrize(
    "field", ["identity", "guardrail", "counts", "curve", "threshold", "policy"]
)
def test_frozen_selection_tampering_rejected(field):
    audit, targets = fixture_rows()
    result, source = build(audit, targets)
    result = deepcopy(result)
    if field == "identity":
        result["selection_projection_sha256"] = "0" * 64
    elif field == "guardrail":
        result["guardrails"]["relaxed"]["min_unique_acted_plots"] = 0
    elif field == "counts":
        result["policies"]["relaxed"]["pooled"]["rescued_count"] += 1
    elif field == "curve":
        result["candidate_curve"][1]["pooled"]["acted_count"] += 1
    elif field == "threshold":
        result["policies"]["relaxed"]["threshold_display"] = 0.4
    else:
        result["policies"]["unknown"] = {}
    with pytest.raises(selection.RouterSelectionError):
        selection.validate_frozen_selection(
            result, audit, targets, source, **kwargs(audit)
        )

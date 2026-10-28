"""Descriptive geometry diagnostics preserve paired identities and strict signs."""
import json

import numpy as np
import pandas as pd
import pytest

from multimodal.attention_fusion_router_geometry_analysis import analyze_geometry_pair


@pytest.fixture
def paired_frames():
    truth = np.zeros(10, dtype=np.int64)
    fusion = np.array([1, 0, 1, 1, 1, 1, 1, 0, 1, 0])
    geo = np.array([0, 1, 2, 0, 0, 0, 0, 0, 1, 1])
    supported = np.ones(10, dtype=bool)
    supported[6] = False
    a_geo = np.array([.4, .4, .4, .4, 0., -.2, np.nan, 0., 0., -.1])
    original = np.array([.5, .5, -.3, 0., .3, -.3, np.nan, 0., 0., .1])
    trained = np.array([-.2, -.2, .2, .2, .2, -.5, np.nan, 0., 0., 0.])
    frames = []
    for image in (original, trained):
        frames.append(pd.DataFrame({
            "row_uid": [f"row{i}" for i in range(10)],
            "file": [f"image{i}.jpg" for i in range(10)],
            "plot_idx": [f"plot{i}" for i in range(10)],
            "label_id_dense": truth, "label_name": ["class0"] * 10,
            "fusion_predictions": fusion, "geo_predictions": geo,
            "reference_plots_fusion_class": np.where(supported, 10, 3),
            "reference_plots_geo_class": np.full(10, 10),
            "geometry_supported": supported, "disagreement": fusion != geo,
            "eligible": supported & (fusion != geo),
            "d_geo_fusion": np.where(supported, .5 * np.exp(a_geo), np.nan),
            "d_geo_geo": np.where(supported, .5, np.nan),
            "d_image_fusion": np.where(supported, .5 * np.exp(image), np.nan),
            "d_image_geo": np.where(supported, .5, np.nan),
            "A_geo": a_geo, "A_image": image, "score": np.minimum(image, a_geo),
        }))
    return frames


def analyze(frames):
    return analyze_geometry_pair(*frames, checkpoint_policy="final")


def test_truth_groups_quadrants_and_strict_zero(paired_frames):
    summary, rows = analyze(paired_frames)
    assert summary["eligible_disagreements"] == {"images": 7, "plots": 7}
    groups = summary["arms"]["pretrained"]["groups"]
    assert {name: item["images"] for name, item in groups.items()} == {
        "eligible_all": 7, "potential_rescue": 4, "potential_harm": 2, "both_wrong": 1}
    assert {name: item["images"] for name, item in groups["eligible_all"]["quadrants"].items()} == {
        "image_positive_geo_positive": 2, "image_positive_geo_nonpositive": 2,
        "image_nonpositive_geo_positive": 2, "image_nonpositive_geo_nonpositive": 1}
    assert rows.loc[3, "image_nonpositive_to_positive"]
    assert not rows.loc[3, "pretrained_positive_score"]
    assert not rows.loc[4, "pretrained_positive_score"]  # geo score exactly zero
    assert not rows.loc[7, "pretrained_positive_score"]  # correct agreement
    assert rows.loc[8, "truth_outcome"] == "both_wrong"  # excluded from eligible group
    assert summary["positive_comparator"] == "strict_gt_zero"


def test_rescue_decomposition_and_geo_supported_crossings(paired_frames):
    summary, rows = analyze(paired_frames)
    rescue = summary["arms"]["pretrained"]["rescue_opportunities"]
    assert {name: item["images"] for name, item in rescue.items()} == {
        "all": 5, "unavailable_reference_support": 1, "eligible_geo_nonpositive": 2,
        "eligible_geo_positive_image_nonpositive": 1, "positive_both_modalities": 1,
        "missed_at_zero": 4}
    geo_rescue = summary["geo_positive_rescue_opportunities"]
    assert {name: item["images"] for name, item in geo_rescue.items()} == {
        "all": 2, "image_positive_in_both": 0, "image_positive_to_nonpositive": 1,
        "image_nonpositive_to_positive": 1, "image_nonpositive_in_both": 0}
    assert rows.loc[0, "checkpoint_matched_geo_supported_rescue_blocked_by_image"]
    assert rows.loc[3, "pretrained_geo_supported_rescue_blocked_by_image"]
    assert rows.loc[6, "pretrained_missed_rescue_at_zero"]


def test_paired_deltas_use_same_query_not_marginal_sorting(paired_frames):
    summary, rows = analyze(paired_frames)
    assert rows.loc[0, "delta_A_image"] == pytest.approx(-.7)
    assert rows.loc[2, "delta_A_image"] == pytest.approx(.5)
    res = summary["paired_groups"]["potential_rescue"]
    assert res["delta_distributions"]["A_image"]["median"] == pytest.approx(-.15)
    assert res["image_delta_sign"]["negative"]["images"] == 3
    assert res["image_delta_sign"]["positive"]["images"] == 1
    assert res["crossings"]["score_positive_to_nonpositive"]["images"] == 1
    assert res["crossings"]["score_nonpositive_to_positive"]["images"] == 1


def test_unsupported_nan_retained_and_summary_is_json_safe(paired_frames):
    summary, rows = analyze(paired_frames)
    assert np.isnan(rows.loc[6, "delta_A_image"])
    assert np.isnan(rows.loc[6, "pretrained_score"])
    assert not rows.loc[6, "score_positive_to_nonpositive"]
    assert not rows.loc[6, "pretrained_positive_score"]
    json.dumps(summary, allow_nan=False)


def test_empty_outcome_group_has_null_statistics(paired_frames):
    subset = [frame.iloc[[0, 3, 4, 5]].copy() for frame in paired_frames]
    summary, _ = analyze(subset)
    distribution = summary["arms"]["pretrained"]["groups"]["potential_harm"]["distributions"]["A_image"]
    assert distribution["count"] == 0
    assert all(value is None for key, value in distribution.items() if key != "count")
    json.dumps(summary, allow_nan=False)


def test_routing_actions_are_not_geometry_inputs(paired_frames):
    expected, _ = analyze(paired_frames)
    frames = [frame.assign(action=bool(i), rescue=bool(i), routed_prediction=i)
              for i, frame in enumerate(paired_frames)]
    actual, _ = analyze(frames)
    assert actual == expected


def test_shared_plot_counts_need_not_sum_over_events(paired_frames):
    for frame in paired_frames:
        frame.loc[[0, 1, 2], "plot_idx"] = "shared"
    summary, _ = analyze(paired_frames)
    assert summary["eligible_disagreements"] == {"images": 7, "plots": 5}
    assert summary["plot_counts_overlap_across_outcomes"] is True


def test_reordered_pair_is_rejected(paired_frames):
    paired_frames[1] = paired_frames[1].iloc[::-1]
    with pytest.raises(ValueError, match="row_uid"):
        analyze(paired_frames)


@pytest.mark.parametrize("column,value", [
    ("file", "different.jpg"), ("plot_idx", "different"), ("label_id_dense", 2),
    ("label_name", "different"), ("fusion_predictions", 2),
    ("reference_plots_fusion_class", 11), ("reference_plots_geo_class", 11),
])
def test_paired_identity_experts_and_support_must_match(paired_frames, column, value):
    paired_frames[1].loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        analyze(paired_frames)


@pytest.mark.parametrize("column", ["d_geo_fusion", "d_geo_geo", "A_geo"])
def test_native_geo_geometry_must_be_unchanged(paired_frames, column):
    paired_frames[1].loc[0, column] += 1e-10
    if column == "A_geo":
        paired_frames[1]["score"] = np.minimum(paired_frames[1].A_geo, paired_frames[1].A_image)
    with pytest.raises(ValueError, match="Shared geo geometry"):
        analyze(paired_frames)


@pytest.mark.parametrize("column", ["A_image", "score", "d_geo_geo"])
def test_unsupported_geometry_must_stay_nan(paired_frames, column):
    paired_frames[0].loc[6, column] = 0.
    with pytest.raises(ValueError, match="unsupported"):
        analyze(paired_frames)


@pytest.mark.parametrize("column", ["A_image", "A_geo", "score"])
def test_supported_geometry_must_be_finite(paired_frames, column):
    paired_frames[0].loc[0, column] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        analyze(paired_frames)


def test_duplicate_query_is_rejected(paired_frames):
    paired_frames[0].loc[1, "row_uid"] = "row0"
    with pytest.raises(ValueError, match="duplicate query"):
        analyze(paired_frames)


def test_eligibility_is_verified(paired_frames):
    paired_frames[0].loc[0, "eligible"] = False
    with pytest.raises(ValueError, match="eligibility"):
        analyze(paired_frames)


def test_minimum_score_is_verified(paired_frames):
    paired_frames[0].loc[0, "score"] = 42.
    with pytest.raises(ValueError, match="minimum modality"):
        analyze(paired_frames)


def test_unsupported_geo_values_must_match_as_nan(paired_frames):
    # NaN equality is intentional for shared unavailable geometry.
    summary, _ = analyze(paired_frames)
    assert summary["shared_native_geo_verified"]


def test_input_frames_are_not_modified(paired_frames):
    copies = [frame.copy(deep=True) for frame in paired_frames]
    analyze(paired_frames)
    for before, after in zip(copies, paired_frames):
        pd.testing.assert_frame_equal(before, after)

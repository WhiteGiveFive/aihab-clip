from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

import multimodal.geo_helpfulness_targets_features as m3


N_CLASSES = 18

EXPECTED_FEATURES = (
    "image_pred",
    "geo_pred",
    "raw_pred",
    "image_geo_pred_pair",
    "geo_raw_pred_pair",
    "image_geo_agree",
    "image_raw_agree",
    "geo_raw_agree",
    "image_geo_top3_overlap",
    "raw_rank_of_geo_pred",
    "image_confidence",
    "geo_confidence",
    "raw_confidence",
    "image_entropy",
    "geo_entropy",
    "raw_entropy",
    "image_top2_margin",
    "geo_top2_margin",
    "raw_top2_margin",
    "geo_minus_image_confidence",
    "geo_minus_raw_confidence",
    "geo_minus_image_entropy",
    "geo_minus_raw_entropy",
    "geo_minus_image_margin",
    "geo_minus_raw_margin",
    "image_geo_jsd",
    "image_geo_total_variation",
    "image_probability_at_geo_pred",
    "geo_probability_at_image_pred",
    "raw_probability_at_geo_pred",
)


def _one_hot(indices: list[int]) -> np.ndarray:
    result = np.zeros((len(indices), N_CLASSES), dtype=np.float64)
    result[np.arange(len(indices)), indices] = 1.0
    return result


def _target_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    row_uids = (f"{1:064x}", f"{2:064x}")
    assignments = pd.DataFrame(
        {
            "schema_version": ["geo_helpfulness_protocol_config_v1"] * 2,
            "protocol_id": ["protocol_v1"] * 2,
            "row_uid": row_uids,
            "file": ["images/a.jpg", "images/b.jpg"],
            "file_lower": ["images/a.jpg", "images/b.jpg"],
            "plot_idx": ["plot-a", "plot-b"],
            "development_role": ["train", "train"],
            "train_oof_fold": pd.array([0, 1], dtype="Int8"),
            "label_id_dense": np.asarray([0, 1], dtype=np.int8),
        }
    )

    # For the first image, the four seeds deliberately cover all target states.
    first_predictions = ((1, 0), (0, 1), (0, 0), (1, 1))
    rows: list[dict[str, object]] = []
    for seed, (raw_pred, geo_pred) in enumerate(first_predictions, start=1):
        rows.append(
            {
                "schema_version": "geo_helpfulness_protocol_config_v1",
                "protocol_id": "protocol_v1",
                "row_uid": row_uids[0],
                "file": "images/a.jpg",
                "file_lower": "images/a.jpg",
                "plot_idx": "plot-a",
                "train_oof_fold": 0,
                "training_seed": seed,
                "image_pred": 17,
                "geo_pred": geo_pred,
                "raw_pred": raw_pred,
            }
        )
        rows.append(
            {
                "schema_version": "geo_helpfulness_protocol_config_v1",
                "protocol_id": "protocol_v1",
                "row_uid": row_uids[1],
                "file": "images/b.jpg",
                "file_lower": "images/b.jpg",
                "plot_idx": "plot-b",
                "train_oof_fold": 1,
                "training_seed": seed,
                "image_pred": 0,
                "geo_pred": 1,
                "raw_pred": 1,
            }
        )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7), assignments


def test_target_order_and_all_four_raw_relative_states_are_frozen():
    labels = np.zeros(4, dtype=np.int8)
    raw_predictions = np.asarray([1, 0, 0, 1], dtype=np.int8)
    geo_predictions = np.asarray([0, 1, 0, 1], dtype=np.int8)

    states = m3.derive_router_target_states(
        raw_predictions,
        geo_predictions,
        labels,
    )

    assert tuple(m3.TARGET_ORDER) == (
        "rescue",
        "harm",
        "both_correct",
        "both_wrong",
    )
    assert np.asarray(states).tolist() == list(m3.TARGET_ORDER)


def test_primary_targets_are_independent_of_image_prediction():
    labels = np.asarray([0, 0, 0, 0], dtype=np.int8)
    raw_predictions = np.asarray([1, 0, 0, 1], dtype=np.int8)
    geo_predictions = np.asarray([0, 1, 0, 1], dtype=np.int8)

    first = m3.derive_router_target_states(raw_predictions, geo_predictions, labels)
    # Image-relative diagnostics may change, but the primary target API has no
    # image-prediction input and must remain raw-fusion-relative.
    image_first = m3.derive_image_relative_states(
        np.asarray([0, 0, 0, 0], dtype=np.int8),
        geo_predictions,
        labels,
    )
    image_second = m3.derive_image_relative_states(
        np.asarray([1, 1, 1, 1], dtype=np.int8),
        geo_predictions,
        labels,
    )
    second = m3.derive_router_target_states(raw_predictions, geo_predictions, labels)

    np.testing.assert_array_equal(first, second)
    assert np.any(np.asarray(image_first) != np.asarray(image_second))


def test_image_relative_diagnostics_use_explicit_exclusive_state_names():
    labels = np.zeros(4, dtype=np.int8)
    image_predictions = np.asarray([1, 0, 0, 1], dtype=np.int8)
    geo_predictions = np.asarray([0, 1, 0, 1], dtype=np.int8)

    states = m3.derive_image_relative_states(
        image_predictions,
        geo_predictions,
        labels,
    )

    assert np.asarray(states).tolist() == [
        "geo_only_correct",
        "image_only_correct",
        "both_correct",
        "both_wrong",
    ]


@pytest.mark.parametrize(
    "raw_predictions,geo_predictions,labels",
    [
        ([0], [0, 1], [0]),
        ([], [], []),
        ([0], [18], [0]),
        ([-1], [0], [0]),
        ([0.0], [0], [0]),
        ([True], [0], [0]),
    ],
)
def test_target_derivation_rejects_malformed_or_out_of_ontology_inputs(
    raw_predictions,
    geo_predictions,
    labels,
):
    with pytest.raises((TypeError, ValueError)):
        m3.derive_router_target_states(raw_predictions, geo_predictions, labels)


def test_target_table_keeps_seed_realizations_and_exact_public_schema():
    oof, assignments = _target_inputs()

    targets = m3.build_router_target_table(
        oof,
        assignments,
        protocol_id="protocol_v1",
    )

    assert list(targets.columns) == [
        "schema_version",
        "protocol_id",
        "row_uid",
        "plot_idx",
        "training_seed",
        "target_state",
    ]
    assert len(targets) == 8
    assert targets["training_seed"].dtype == np.dtype("int8")
    for column in ("schema_version", "protocol_id", "row_uid", "plot_idx", "target_state"):
        assert isinstance(targets[column].dtype, pd.StringDtype)
    assert not targets.duplicated(["row_uid", "training_seed"]).any()
    assert list(zip(targets["row_uid"], targets["training_seed"])) == sorted(
        zip(targets["row_uid"], targets["training_seed"])
    )
    first_uid = assignments.iloc[0]["row_uid"]
    assert targets.loc[targets["row_uid"].eq(first_uid), "target_state"].tolist() == list(
        m3.TARGET_ORDER
    )
    forbidden_tokens = ("label", "correctness", "logit", "prob", "utility")
    assert not any(
        token in column
        for column in targets.columns
        for token in forbidden_tokens
    )


def test_target_validator_rejects_plot_identity_drift_across_seeds():
    oof, assignments = _target_inputs()
    targets = m3.build_router_target_table(
        oof,
        assignments,
        protocol_id="protocol_v1",
    )
    poisoned = targets.copy()
    poisoned.loc[0, "plot_idx"] = "different-plot"

    with pytest.raises((TypeError, ValueError), match="plot|identity"):
        m3.validate_router_target_table(poisoned, protocol_id="protocol_v1")


def test_target_table_rejects_nonfrozen_seed_membership():
    oof, assignments = _target_inputs()
    oof.loc[oof["training_seed"].eq(4), "training_seed"] = 5

    with pytest.raises((TypeError, ValueError), match="seed|expected"):
        m3.build_router_target_table(
            oof,
            assignments,
            protocol_id="protocol_v1",
            expected_seeds=(1, 2, 3, 4),
        )


@pytest.mark.parametrize(
    "mutation,error_pattern",
    [
        ("duplicate_key", "duplicate|unique|key"),
        ("missing_seed", "seed|cover|missing"),
        ("plot_drift", "plot"),
        ("fold_drift", "fold"),
        ("protocol_drift", "protocol"),
        ("label_leak", "label|forbidden|column"),
    ],
)
def test_target_table_rejects_identity_seed_protocol_and_label_leakage(
    mutation: str,
    error_pattern: str,
):
    oof, assignments = _target_inputs()
    oof = oof.reset_index(drop=True)
    if mutation == "duplicate_key":
        oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    elif mutation == "missing_seed":
        first_uid = assignments.iloc[0]["row_uid"]
        oof = oof.loc[
            ~(oof["row_uid"].eq(first_uid) & oof["training_seed"].eq(4))
        ].copy()
    elif mutation == "plot_drift":
        oof.loc[0, "plot_idx"] = "wrong-plot"
    elif mutation == "fold_drift":
        oof.loc[0, "train_oof_fold"] = 3
    elif mutation == "protocol_drift":
        oof.loc[0, "protocol_id"] = "different_protocol"
    elif mutation == "label_leak":
        oof["label_id_dense"] = 0
    else:  # pragma: no cover - the parametrization is exhaustive.
        raise AssertionError(mutation)

    with pytest.raises((TypeError, ValueError), match=error_pattern):
        m3.build_router_target_table(
            oof,
            assignments,
            protocol_id="protocol_v1",
        )


def test_uniform_probabilities_anchor_entropy_ties_ranks_and_exact_dtypes():
    uniform = np.full((2, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64)

    features = m3.build_router_feature_frame(
        uniform,
        uniform,
        uniform,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )

    assert tuple(features.columns) == EXPECTED_FEATURES
    assert features.shape == (2, 30)
    for column in ("image_pred", "geo_pred", "raw_pred"):
        assert features[column].dtype == np.dtype("int8")
        assert features[column].tolist() == [0, 0]
    for column in ("image_geo_pred_pair", "geo_raw_pred_pair"):
        assert features[column].dtype == np.dtype("int16")
    for column in ("image_geo_agree", "image_raw_agree", "geo_raw_agree"):
        assert features[column].dtype == np.dtype("bool")
        assert features[column].all()
    for column in ("image_geo_top3_overlap", "raw_rank_of_geo_pred"):
        assert features[column].dtype == np.dtype("int8")
    assert features["image_geo_top3_overlap"].tolist() == [3, 3]
    assert features["raw_rank_of_geo_pred"].tolist() == [1, 1]
    for column in EXPECTED_FEATURES[10:]:
        assert features[column].dtype == np.dtype("float64")
    for prefix in ("image", "geo", "raw"):
        np.testing.assert_allclose(features[f"{prefix}_entropy"], np.log(N_CLASSES))
        np.testing.assert_array_equal(features[f"{prefix}_top2_margin"], 0.0)
    np.testing.assert_array_equal(features["image_geo_jsd"], 0.0)
    np.testing.assert_array_equal(features["image_geo_total_variation"], 0.0)


def test_disjoint_one_hot_probabilities_anchor_jsd_tv_and_zero_terms():
    image = _one_hot([0])
    geo = _one_hot([1])
    raw = _one_hot([1])

    features = m3.build_router_feature_frame(
        image,
        geo,
        raw,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )

    assert features.loc[0, "image_pred"] == 0
    assert features.loc[0, "geo_pred"] == 1
    assert features.loc[0, "image_geo_pred_pair"] == 1
    assert features.loc[0, "geo_raw_pred_pair"] == N_CLASSES + 1
    assert features.loc[0, "image_geo_jsd"] == pytest.approx(np.log(2.0))
    assert features.loc[0, "image_geo_total_variation"] == pytest.approx(1.0)
    assert features.loc[0, "image_entropy"] == pytest.approx(0.0)
    assert features.loc[0, "geo_entropy"] == pytest.approx(0.0)
    assert features.loc[0, "image_probability_at_geo_pred"] == 0.0
    assert features.loc[0, "geo_probability_at_image_pred"] == 0.0
    assert features.loc[0, "raw_probability_at_geo_pred"] == 1.0


def test_all_numeric_feature_formulas_and_signed_differences_are_exact():
    image = np.zeros((1, N_CLASSES), dtype=np.float64)
    geo = np.zeros((1, N_CLASSES), dtype=np.float64)
    raw = np.zeros((1, N_CLASSES), dtype=np.float64)
    image[0, :3] = [0.50, 0.30, 0.20]
    geo[0, :3] = [0.25, 0.60, 0.15]
    raw[0, :3] = [0.20, 0.35, 0.45]

    features = m3.build_router_feature_frame(
        image,
        geo,
        raw,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )

    def entropy(values: np.ndarray) -> float:
        positive = values[values > 0.0]
        return float(-np.sum(positive * np.log(positive)))

    image_entropy = entropy(image[0])
    geo_entropy = entropy(geo[0])
    raw_entropy = entropy(raw[0])
    midpoint = 0.5 * (image[0] + geo[0])
    jsd = 0.5 * (
        np.sum(image[0, image[0] > 0] * np.log(image[0, image[0] > 0] / midpoint[image[0] > 0]))
        + np.sum(geo[0, geo[0] > 0] * np.log(geo[0, geo[0] > 0] / midpoint[geo[0] > 0]))
    )
    expected = {
        "image_confidence": 0.50,
        "geo_confidence": 0.60,
        "raw_confidence": 0.45,
        "image_entropy": image_entropy,
        "geo_entropy": geo_entropy,
        "raw_entropy": raw_entropy,
        "image_top2_margin": 0.20,
        "geo_top2_margin": 0.35,
        "raw_top2_margin": 0.10,
        "geo_minus_image_confidence": 0.10,
        "geo_minus_raw_confidence": 0.15,
        "geo_minus_image_entropy": geo_entropy - image_entropy,
        "geo_minus_raw_entropy": geo_entropy - raw_entropy,
        "geo_minus_image_margin": 0.15,
        "geo_minus_raw_margin": 0.25,
        "image_geo_jsd": float(jsd),
        "image_geo_total_variation": 0.30,
        "image_probability_at_geo_pred": 0.30,
        "geo_probability_at_image_pred": 0.25,
        "raw_probability_at_geo_pred": 0.35,
    }
    assert features.loc[0, "image_pred"] == 0
    assert features.loc[0, "geo_pred"] == 1
    assert features.loc[0, "raw_pred"] == 2
    assert features.loc[0, "image_geo_pred_pair"] == 1
    assert features.loc[0, "geo_raw_pred_pair"] == 20
    assert features.loc[0, "raw_rank_of_geo_pred"] == 2
    for name, value in expected.items():
        assert features.loc[0, name] == pytest.approx(value, abs=1.0e-15)


def test_top3_overlap_and_raw_rank_use_dense_id_tie_breaks():
    image = np.zeros((1, N_CLASSES), dtype=np.float64)
    geo = np.zeros((1, N_CLASSES), dtype=np.float64)
    raw = np.zeros((1, N_CLASSES), dtype=np.float64)
    image[0, [0, 1, 2, 6]] = [0.4, 0.3, 0.2, 0.1]
    geo[0, [3, 4, 5, 7]] = [0.4, 0.3, 0.2, 0.1]
    # Geo predicts class 3. Under raw it follows the tied classes 1 then 2.
    raw[0, [1, 2, 3, 8]] = [0.3, 0.3, 0.2, 0.2]

    features = m3.build_router_feature_frame(
        image,
        geo,
        raw,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )

    assert features.loc[0, "image_geo_top3_overlap"] == 0
    assert features.loc[0, "raw_rank_of_geo_pred"] == 3
    assert features.loc[0, "raw_pred"] == 1
    assert features.loc[0, "raw_top2_margin"] == 0.0


def test_probability_basis_is_fail_closed_against_native_t1():
    probabilities = _one_hot([0])

    with pytest.raises((TypeError, ValueError), match="basis|calibrat|native"):
        m3.build_router_feature_frame(
            probabilities,
            probabilities,
            probabilities,
            probability_basis="native_t1_uncalibrated",
        )


def test_probability_sum_tolerance_does_not_clip_or_renormalize_inputs():
    probabilities = np.full((1, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64)
    probabilities[0, 0] += 5.0e-9

    features = m3.build_router_feature_frame(
        probabilities,
        probabilities,
        probabilities,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )

    assert features.loc[0, "image_confidence"] == probabilities[0, 0]


def test_training_and_inference_use_the_same_stateless_nonmutating_builder():
    image = np.vstack([_one_hot([0]), _one_hot([1])])
    geo = np.vstack([_one_hot([1]), _one_hot([1])])
    raw = np.vstack([_one_hot([2]), _one_hot([0])])
    originals = tuple(values.copy() for values in (image, geo, raw))

    training_features = m3.build_router_feature_frame(
        image,
        geo,
        raw,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )
    inference_features = m3.build_router_feature_frame(
        image.copy(),
        geo.copy(),
        raw.copy(),
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )

    pd.testing.assert_frame_equal(
        training_features,
        inference_features,
        check_exact=True,
    )
    for observed, original in zip((image, geo, raw), originals):
        np.testing.assert_array_equal(observed, original)


def test_feature_builder_rejects_metadata_bearing_dataframe_and_mapping_aliases():
    probabilities = _one_hot([0])
    dataframe = pd.DataFrame(
        probabilities,
        columns=[f"class_{index}" for index in range(N_CLASSES)],
        index=pd.Index([f"{1:064x}"], name="row_uid"),
    )
    alias_mapping = {
        "row_uid": [f"{1:064x}"],
        "probabilities": probabilities,
    }

    for invalid in (dataframe, alias_mapping):
        with pytest.raises((TypeError, ValueError), match="matrix|array|metadata|numeric"):
            m3.build_router_feature_frame(
                invalid,
                probabilities,
                probabilities,
                probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
            )


@pytest.mark.parametrize(
    "image,geo,raw",
    [
        (np.ones(N_CLASSES), _one_hot([0]), _one_hot([0])),
        (np.empty((0, N_CLASSES)), np.empty((0, N_CLASSES)), np.empty((0, N_CLASSES))),
        (np.ones((1, N_CLASSES - 1)) / (N_CLASSES - 1), _one_hot([0]), _one_hot([0])),
        (_one_hot([0, 1]), _one_hot([0]), _one_hot([0])),
        (np.full((1, N_CLASSES), np.nan), _one_hot([0]), _one_hot([0])),
        (np.full((1, N_CLASSES), 1.0 / N_CLASSES + 1.0e-7), _one_hot([0]), _one_hot([0])),
        (np.asarray([[-1.0] + [2.0] + [0.0] * (N_CLASSES - 2)]), _one_hot([0]), _one_hot([0])),
    ],
)
def test_feature_builder_rejects_malformed_probability_matrices(image, geo, raw):
    with pytest.raises((TypeError, ValueError)):
        m3.build_router_feature_frame(
            image,
            geo,
            raw,
            probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
        )


def test_feature_schema_freezes_allowlist_vocabularies_and_m4_expansion():
    schema = m3.build_router_feature_schema()

    validation = m3.validate_router_feature_schema(schema)
    assert validation["schema_sha256"] == (
        "6f9589da550c32c495a3825d7f12cbb42e49f57798b2fa0676917a35cbbca76c"
    )
    assert schema["probability_basis"] == m3.CALIBRATED_PROBABILITY_BASIS
    assert tuple(schema["ordered_semantic_features"]) == EXPECTED_FEATURES
    assert schema["semantic_feature_count"] == 30
    transformed = schema["m4_transformed_feature_contract"]
    assert transformed["scaled_column_count"] == 25
    assert transformed["one_hot_column_count"] == 702
    assert transformed["total_column_count"] == 727
    assert schema["categorical_vocabularies"]["image_pred"] == list(range(N_CLASSES))
    assert schema["categorical_vocabularies"]["image_geo_pred_pair"] == list(
        range(N_CLASSES * N_CLASSES)
    )


def test_feature_schema_validation_rejects_reordering_and_forbidden_aliases():
    schema = m3.build_router_feature_schema()
    reordered = deepcopy(schema)
    reordered["ordered_semantic_features"][0:2] = reversed(
        reordered["ordered_semantic_features"][0:2]
    )
    with pytest.raises((TypeError, ValueError), match="order|schema|feature"):
        m3.validate_router_feature_schema(reordered)

    forbidden_allowlist = {
        family: list(names)
        for family, names in schema["feature_allowlist"].items()
    }
    forbidden_allowlist["numeric"].append("true_class_probability")
    with pytest.raises((TypeError, ValueError), match="forbidden|allowlist|feature"):
        m3.build_router_feature_schema(configured_allowlist=forbidden_allowlist)


def test_feature_frame_validation_rejects_extra_identity_or_target_columns():
    probabilities = _one_hot([0, 1])
    features = m3.build_router_feature_frame(
        probabilities,
        probabilities,
        probabilities,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )
    m3.validate_router_feature_frame(
        features,
        probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
    )

    for name in ("row_uid", "plot_idx", "training_seed", "target_state", "label_id_dense"):
        poisoned = features.assign(**{name: "forbidden"})
        with pytest.raises((TypeError, ValueError), match="column|feature|schema|forbidden"):
            m3.validate_router_feature_frame(
                poisoned,
                probability_basis=m3.CALIBRATED_PROBABILITY_BASIS,
            )


def test_prevalence_report_reconciles_all_complete_groupings_and_seed_stability():
    oof, assignments = _target_inputs()

    report = m3.build_target_prevalence_report(
        oof,
        assignments,
        protocol_id="protocol_v1",
        expected_seeds=(1, 2, 3, 4),
    )

    assert report["pooled_seed_realizations"]["record_count"] == 8
    assert report["pooled_seed_realizations"]["target_counts"] == {
        "rescue": 1,
        "harm": 1,
        "both_correct": 5,
        "both_wrong": 1,
    }
    assert [record["record_count"] for record in report["per_training_seed"]] == [
        2,
        2,
        2,
        2,
    ]
    per_seed_reconciled = {
        state: sum(
            record["target_counts"][state]
            for record in report["per_training_seed"]
        )
        for state in m3.TARGET_ORDER
    }
    assert per_seed_reconciled == report["pooled_seed_realizations"]["target_counts"]
    assert report["cross_seed_stability"] == {
        "unique_image_count": 2,
        "distinct_target_state_count_distribution": {
            "1": 1,
            "2": 0,
            "3": 0,
            "4": 1,
        },
        "unchanged_image_count": 1,
        "changing_image_count": 1,
        "changing_image_fraction": 0.5,
    }
    breakdowns = report["breakdowns"]
    assert len(breakdowns["habitat"]) == N_CLASSES
    assert len(breakdowns["plot"]) == 2
    assert len(breakdowns["image_geo_pred_pair"]) == N_CLASSES * N_CLASSES
    assert len(breakdowns["geo_raw_pred_pair"]) == N_CLASSES * N_CLASSES
    pooled = report["pooled_seed_realizations"]["target_counts"]
    for grouping in breakdowns.values():
        reconciled = {
            state: sum(record["target_counts"][state] for record in grouping)
            for state in m3.TARGET_ORDER
        }
        assert reconciled == pooled
    unsupported = next(
        record for record in breakdowns["habitat"] if record["record_count"] == 0
    )
    assert unsupported["target_fractions"] == {
        state: None for state in m3.TARGET_ORDER
    }
    assert "not independent biological samples" in report["interpretation_warning"]
    json.dumps(report, allow_nan=False)


def test_prevalence_auxiliary_states_are_diagnostic_and_use_explicit_names():
    oof, assignments = _target_inputs()

    report = m3.build_target_prevalence_report(
        oof,
        assignments,
        protocol_id="protocol_v1",
        expected_seeds=(1, 2, 3, 4),
    )

    diagnostic = report["auxiliary_image_relative_states"]
    assert diagnostic["purpose"] == "diagnostic_only_not_a_router_target"
    assert diagnostic["state_order"] == [
        "geo_only_correct",
        "image_only_correct",
        "both_correct",
        "both_wrong",
    ]
    assert sum(diagnostic["pooled_seed_realizations"]["target_counts"].values()) == 8


def test_feature_leakage_audit_is_exact_reproducible_and_fail_closed():
    forbidden_patterns = (
        "label",
        "target",
        "true_class",
        "correctness",
        "correct",
        "y_true",
        "true_probability",
        "true_nll",
        "nll_advantage",
    )
    audit = m3.build_feature_leakage_audit(
        forbidden_patterns=forbidden_patterns,
    )

    validation = m3.validate_feature_leakage_audit(
        audit,
        forbidden_patterns=forbidden_patterns,
    )
    assert validation["valid"] is True
    assert audit["valid"] is True
    assert audit["ordered_semantic_features"] == list(EXPECTED_FEATURES)
    assert audit["forbidden_pattern_matches"] == {}
    assert audit["excluded_name_overlap"] == []
    assert audit["probability_basis_required"] == m3.CALIBRATED_PROBABILITY_BASIS
    assert audit["native_t1_probability_basis_rejected"] is True
    assert audit["target_feature_separation"]["target_columns_in_features"] == []
    assert audit["feature_builder_interface"]["fit_state"] is False
    json.dumps(audit, allow_nan=False)

    poisoned = deepcopy(audit)
    poisoned["native_t1_probability_basis_rejected"] = False
    with pytest.raises((TypeError, ValueError), match="audit|reproduce|pass"):
        m3.validate_feature_leakage_audit(
            poisoned,
            forbidden_patterns=forbidden_patterns,
        )

    with pytest.raises((TypeError, ValueError), match="role|contract"):
        m3.build_feature_leakage_audit(
            forbidden_patterns=forbidden_patterns,
            input_artifact_roles=(
                *m3.DEFAULT_INPUT_ARTIFACT_ROLES,
                "development_validation_outputs",
            ),
        )

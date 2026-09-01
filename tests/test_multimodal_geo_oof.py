from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml

import multimodal.geo_helpfulness_oof as geo_oof
from multimodal.geo_helpfulness_oof import (
    DEFAULT_CONFIG_PATH,
    GEO_COLUMNS,
    M2ArtifactError,
    OOF_OUTPUT_COLUMNS,
    PROJECT_ROOT,
    VALIDATION_OUTPUT_COLUMNS,
    FrozenM2Context,
    apply_geo_standardization,
    build_output_table,
    fit_geo_standardization,
    producer_partitions,
    read_output_parquet,
    stable_softmax_float64,
    validate_output_table,
    validate_reproduced_output,
    validate_training_seed,
    write_output_parquet_atomic,
)
from multimodal.geo_helpfulness_oof_report import (
    OOFReportError,
    build_oof_reproduction_report,
)


N_CLASSES = 18


def _assignments() -> pd.DataFrame:
    """Small plot-grouped assignment table in deliberately noncanonical order."""

    return pd.DataFrame(
        {
            "schema_version": ["assignments_v1"] * 8,
            "protocol_id": ["protocol_v1"] * 8,
            "row_uid": ["u7", "u2", "u5", "u1", "u8", "u3", "u6", "u4"],
            "file": [f"images/{value}.jpg" for value in range(7, -1, -1)],
            "file_lower": [f"images/{value}.jpg" for value in range(7, -1, -1)],
            "plot_idx": ["p3", "p0", "p2", "p0", "pv1", "p1", "pv0", "p1"],
            "development_role": [
                "train",
                "train",
                "train",
                "train",
                "validation",
                "train",
                "validation",
                "train",
            ],
            "train_oof_fold": pd.array([3, 0, 2, 0, pd.NA, 1, pd.NA, 1], dtype="Int8"),
            "label_id_dense": np.asarray([7, 2, 5, 1, 8, 3, 6, 4], dtype=np.int8),
        }
    )


def _prediction_assignments() -> pd.DataFrame:
    assignments = _assignments()
    fold_one = assignments["train_oof_fold"].astype("Int8").eq(1).fillna(False)
    return assignments.loc[
        assignments["development_role"].eq("train")
        & fold_one
    ].reset_index(drop=True)


def _logits_by_mode(row_count: int) -> dict[str, np.ndarray]:
    base = np.arange(row_count * N_CLASSES, dtype=np.float64).reshape(
        row_count, N_CLASSES
    )
    return {
        "image_only": base - 100.0,
        "geo_only": -base,
        "raw_concat": np.flip(base, axis=1).copy(),
    }


def _report_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.DataFrame(
        {
            "row_uid": ["u0", "u1", "u2", "validation-only"],
            "label_id_dense": [0, 1, 2, 2],
            "development_role": ["train", "train", "train", "validation"],
        }
    )
    perfect = np.asarray([0, 1, 2], dtype=np.int8)
    cyclic = np.asarray([1, 2, 0], dtype=np.int8)
    rows: list[dict[str, object]] = []
    for seed, patterns in (
        (1, {"image": perfect, "geo": cyclic, "raw": perfect}),
        (2, {"image": cyclic, "geo": perfect, "raw": cyclic}),
    ):
        for position, row_uid in enumerate(("u0", "u1", "u2")):
            row: dict[str, object] = {
                "row_uid": row_uid,
                "training_seed": seed,
            }
            for mode, predictions in patterns.items():
                prediction = int(predictions[position])
                logits = np.full(3, -2.0, dtype=np.float64)
                logits[prediction] = 2.0
                row[f"{mode}_logits"] = logits.tolist()
                row[f"{mode}_pred"] = prediction
            rows.append(row)
    return pd.DataFrame(rows), assignments


def _minimal_context(
    tmp_path: Path,
    *,
    source_tables: list[Path],
) -> FrozenM2Context:
    placeholder = tmp_path / "unused"
    return FrozenM2Context(
        protocol_dir=placeholder,
        output_root=placeholder,
        resolved_path=placeholder,
        assignments_path=placeholder,
        manifest_path=placeholder,
        config={
            "paths": {
                "development_source_tables": [str(path) for path in source_tables]
            }
        },
        assignments=pd.DataFrame(),
        protocol_manifest={},
        preflight={},
        parent_hashes={},
        code_file_hashes={},
        code_hash="synthetic-code-hash",
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_validate_training_seed_accepts_only_frozen_seeds(seed: int):
    assert validate_training_seed(seed) == seed


@pytest.mark.parametrize("seed", [0, 5, -1, 1.0, "1", True, None])
def test_validate_training_seed_rejects_nonfrozen_or_noninteger_values(seed):
    with pytest.raises((TypeError, ValueError)):
        validate_training_seed(seed)


def test_stable_softmax_float64_handles_extreme_logits_without_precision_loss():
    logits = np.asarray(
        [
            [10_000.0, 9_999.0, -10_000.0],
            [-10_000.0, -10_000.0, -10_000.0],
            [1e300, 1e300 - 1e286, -1e300],
        ],
        dtype=np.float64,
    )

    probabilities = stable_softmax_float64(logits)

    assert probabilities.dtype == np.float64
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-15, rtol=0)
    assert np.argmax(probabilities[0]) == np.argmax(logits[0]) == 0
    np.testing.assert_allclose(probabilities[1], np.full(3, 1.0 / 3.0), atol=1e-15)
    assert np.argmax(probabilities[2]) == np.argmax(logits[2]) == 0


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([[0.0, np.nan]], dtype=np.float64),
        np.asarray([[0.0, np.inf]], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.empty((1, 0), dtype=np.float64),
    ],
)
def test_stable_softmax_float64_rejects_invalid_logits(invalid: np.ndarray):
    with pytest.raises(ValueError):
        stable_softmax_float64(invalid)


def test_geo_standardization_uses_float32_population_stats_and_zero_std_guard():
    values = np.asarray(
        [
            [1.0, 4.0, -4.0],
            [3.0, 4.0, 0.0],
            [5.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )

    scaler = fit_geo_standardization(values)
    transformed = apply_geo_standardization(values, scaler)

    assert scaler["mean"].dtype == np.float32
    assert scaler["std"].dtype == np.float32
    np.testing.assert_array_equal(scaler["mean"], values.mean(axis=0, dtype=np.float32))
    expected_std = values.std(axis=0, dtype=np.float32, ddof=0)
    expected_std[expected_std == 0] = 1
    np.testing.assert_array_equal(scaler["std"], expected_std)
    assert transformed.dtype == np.float32
    np.testing.assert_allclose(transformed[:, 0].mean(), 0.0, atol=2e-7)
    np.testing.assert_array_equal(transformed[:, 1], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(transformed[:, 2].std(ddof=0), 1.0, atol=2e-7)


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([[0.0, np.nan]], dtype=np.float32),
        np.asarray([[0.0, np.inf]], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    ],
)
def test_geo_standardization_rejects_invalid_fitting_values(invalid: np.ndarray):
    with pytest.raises(ValueError):
        fit_geo_standardization(invalid)


def test_producer_partitions_are_plot_disjoint_and_keep_validation_out_of_oof_fit():
    assignments = _assignments()

    fit, prediction = producer_partitions(assignments, stage="train_oof", fold=1)

    assert set(prediction["row_uid"]) == {"u3", "u4"}
    assert set(fit["row_uid"]) == {"u1", "u2", "u5", "u7"}
    assert set(fit["development_role"]) == {"train"}
    assert set(prediction["development_role"]) == {"train"}
    assert set(fit["plot_idx"]).isdisjoint(set(prediction["plot_idx"]))
    assert not {"u6", "u8"}.intersection(fit["row_uid"])


def test_validation_producer_fits_all_development_train_and_predicts_validation():
    assignments = _assignments()

    fit, prediction = producer_partitions(assignments, stage="development_validation")

    assert set(fit["development_role"]) == {"train"}
    assert set(prediction["development_role"]) == {"validation"}
    assert set(fit["row_uid"]) == {"u1", "u2", "u3", "u4", "u5", "u7"}
    assert set(prediction["row_uid"]) == {"u6", "u8"}
    assert set(fit["plot_idx"]).isdisjoint(set(prediction["plot_idx"]))


def test_producer_partitions_reject_invalid_stage_fold_and_plot_leakage():
    assignments = _assignments()
    with pytest.raises(ValueError):
        producer_partitions(assignments, stage="unknown")
    with pytest.raises(ValueError):
        producer_partitions(assignments, stage="train_oof", fold=None)
    with pytest.raises(ValueError):
        producer_partitions(assignments, stage="train_oof", fold=4)

    poisoned = assignments.copy()
    poisoned.loc[poisoned["row_uid"].eq("u5"), "plot_idx"] = "p1"
    with pytest.raises(ValueError, match="plot|overlap|disjoint"):
        producer_partitions(poisoned, stage="train_oof", fold=1)


def test_build_oof_output_is_label_blind_canonically_sorted_and_exactly_typed():
    assignments = _prediction_assignments().iloc[::-1].reset_index(drop=True)
    logits = _logits_by_mode(len(assignments))

    table = build_output_table(
        assignments,
        seed=3,
        logits_by_mode=logits,
        include_fold=True,
    )

    assert isinstance(table, pa.Table)
    assert table.column_names == list(OOF_OUTPUT_COLUMNS)
    assert table["row_uid"].to_pylist() == sorted(assignments["row_uid"].tolist())
    assert pa.types.is_int8(table.schema.field("train_oof_fold").type)
    assert pa.types.is_int8(table.schema.field("training_seed").type)
    for column in ("image_pred", "geo_pred", "raw_pred"):
        assert pa.types.is_int8(table.schema.field(column).type)
    for column in (
        "image_logits",
        "geo_logits",
        "raw_logits",
        "image_prob_native_t1",
        "geo_prob_native_t1",
        "raw_prob_native_t1",
    ):
        dtype = table.schema.field(column).type
        assert pa.types.is_fixed_size_list(dtype)
        assert dtype.list_size == N_CLASSES
        assert pa.types.is_float64(dtype.value_type)
    assert not {
        "label_id_dense",
        "label_name",
        "correct",
        "true_class_probability",
        "nll",
    }.intersection(table.column_names)
    validate_output_table(table, include_fold=True, expected_rows=len(assignments))


def test_build_validation_output_omits_fold_and_preserves_logits_probability_contract():
    assignments = _assignments().loc[
        _assignments()["development_role"].eq("validation")
    ].reset_index(drop=True)
    logits = _logits_by_mode(len(assignments))

    table = build_output_table(
        assignments,
        seed=4,
        logits_by_mode=logits,
        include_fold=False,
    )

    assert table.column_names == list(VALIDATION_OUTPUT_COLUMNS)
    assert "train_oof_fold" not in table.column_names
    validate_output_table(table, include_fold=False, expected_rows=len(assignments))
    as_dict = table.to_pydict()
    for mode, output_prefix in (
        ("image_only", "image"),
        ("geo_only", "geo"),
        ("raw_concat", "raw"),
    ):
        serialized_logits = np.asarray(as_dict[f"{output_prefix}_logits"], dtype=np.float64)
        serialized_probabilities = np.asarray(
            as_dict[f"{output_prefix}_prob_native_t1"], dtype=np.float64
        )
        np.testing.assert_allclose(
            serialized_probabilities,
            stable_softmax_float64(serialized_logits),
            atol=1e-15,
            rtol=0,
        )
        np.testing.assert_array_equal(
            np.asarray(as_dict[f"{output_prefix}_pred"], dtype=np.int8),
            np.argmax(serialized_logits, axis=1).astype(np.int8),
        )
        # Input order is deliberately different from canonical row order.  The
        # expected arrays are checked through the serialized row identities.
        assert serialized_logits.shape == (len(assignments), N_CLASSES)


def test_output_validation_rejects_extra_columns_and_numeric_inconsistency():
    assignments = _prediction_assignments()
    table = build_output_table(
        assignments,
        seed=1,
        logits_by_mode=_logits_by_mode(len(assignments)),
        include_fold=True,
    )

    with_extra = table.append_column("label_id_dense", pa.array([1] * len(table), pa.int8()))
    with pytest.raises(ValueError, match="column|allow|schema"):
        validate_output_table(with_extra, include_fold=True)

    probability_index = table.schema.get_field_index("image_prob_native_t1")
    bad_probability = np.asarray(table["image_prob_native_t1"].to_pylist(), dtype=np.float64)
    bad_probability[0, 0] = 0.99
    inconsistent = table.set_column(
        probability_index,
        "image_prob_native_t1",
        pa.array(bad_probability.tolist(), type=pa.list_(pa.float64(), N_CLASSES)),
    )
    with pytest.raises(ValueError, match="prob|softmax|sum"):
        validate_output_table(inconsistent, include_fold=True)


def test_parquet_round_trip_keeps_physical_schema_and_refuses_overwrite(tmp_path: Path):
    assignments = _prediction_assignments()
    table = build_output_table(
        assignments,
        seed=2,
        logits_by_mode=_logits_by_mode(len(assignments)),
        include_fold=True,
    )
    path = tmp_path / "heldout_model_outputs.parquet"

    write_output_parquet_atomic(table, path)
    restored = read_output_parquet(path, include_fold=True)

    assert restored.schema == table.schema
    assert restored.equals(table)
    validate_output_table(restored, include_fold=True, expected_rows=len(assignments))
    with pytest.raises(FileExistsError):
        write_output_parquet_atomic(table, path)


def test_oof_report_has_exact_per_seed_mode_metrics_and_population_summaries():
    oof, assignments = _report_inputs()

    report = build_oof_reproduction_report(
        oof,
        assignments,
        dense_class_count=3,
        modes=("image", "geo", "raw"),
    )

    perfect = {
        "top1_acc": 1.0,
        "top3_acc": 1.0,
        "weighted_f1": 1.0,
        "macro_f1": 1.0,
        "mcc": 1.0,
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    cyclic = {
        "top1_acc": 0.0,
        "top3_acc": 1.0,
        "weighted_f1": 0.0,
        "macro_f1": 0.0,
        "mcc": -0.5,
        "confusion_matrix": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    }
    assert report["training_seeds"] == [1, 2]
    assert report["row_count"] == 6
    assert report["unique_row_count"] == 3
    assert report["cross_seed_std_definition"] == "population"
    assert report["per_seed"]["1"]["modes"] == {
        "image": perfect,
        "geo": cyclic,
        "raw": perfect,
    }
    assert report["per_seed"]["2"]["modes"] == {
        "image": cyclic,
        "geo": perfect,
        "raw": cyclic,
    }
    for mode in ("image", "geo", "raw"):
        summary = report["cross_seed"][mode]
        assert summary["top1_acc"] == {"mean": 0.5, "std": 0.5}
        assert summary["top3_acc"] == {"mean": 1.0, "std": 0.0}
        assert summary["weighted_f1"] == {"mean": 0.5, "std": 0.5}
        assert summary["macro_f1"] == {"mean": 0.5, "std": 0.5}
        assert summary["mcc"] == {"mean": 0.25, "std": 0.75}


def test_oof_report_rejects_labels_prediction_drift_and_incomplete_seed_coverage():
    oof, assignments = _report_inputs()

    label_leak = oof.assign(label_id_dense=0)
    with pytest.raises(OOFReportError, match="label-blind|forbidden"):
        build_oof_reproduction_report(label_leak, assignments, dense_class_count=3)

    prediction_drift = oof.copy(deep=True)
    prediction_drift.loc[0, "image_pred"] = 2
    with pytest.raises(OOFReportError, match="argmax"):
        build_oof_reproduction_report(prediction_drift, assignments, dense_class_count=3)

    incomplete = oof.loc[~(
        oof["training_seed"].eq(2) & oof["row_uid"].eq("u2")
    )]
    with pytest.raises(OOFReportError, match="exactly cover|missing"):
        build_oof_reproduction_report(incomplete, assignments, dense_class_count=3)


def test_real_sealed_assignments_have_exact_m2_fold_counts_when_available():
    if not DEFAULT_CONFIG_PATH.is_file():
        pytest.skip("frozen protocol configuration is unavailable")
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assignments_path = (
        PROJECT_ROOT / config["paths"]["protocol_root"] / "development_assignments.parquet"
    ).resolve()
    if not assignments_path.is_file():
        pytest.skip("sealed development assignments are unavailable")

    assignments = pd.read_parquet(assignments_path)
    assert len(assignments) == 4_200
    assert int(assignments["development_role"].eq("train").sum()) == 3_378
    assert int(assignments["development_role"].eq("validation").sum()) == 822
    expected_prediction_rows = {0: 862, 1: 815, 2: 849, 3: 852}
    expected_fitting_rows = {fold: 3_378 - count for fold, count in expected_prediction_rows.items()}
    for fold in range(4):
        fitting, prediction = producer_partitions(
            assignments,
            stage="train_oof",
            fold=fold,
        )
        assert len(fitting) == expected_fitting_rows[fold]
        assert len(prediction) == expected_prediction_rows[fold]
        assert fitting["plot_idx"].nunique() == 975
        assert prediction["plot_idx"].nunique() == 325
        assert set(fitting["plot_idx"]).isdisjoint(set(prediction["plot_idx"]))

    fitting, prediction = producer_partitions(
        assignments,
        stage="development_validation",
    )
    assert (len(fitting), fitting["plot_idx"].nunique()) == (3_378, 1_300)
    assert (len(prediction), prediction["plot_idx"].nunique()) == (822, 325)


def test_transform_contract_sets_the_exact_frozen_crop_ratio():
    from torchvision.transforms import (
        Compose,
        Normalize,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )
    from torchvision.transforms.functional import InterpolationMode

    def _convert_to_rgb(image):
        return image.convert("RGB")

    crop = RandomResizedCrop(
        (384, 384),
        scale=(0.9, 1.0),
        ratio=(0.25, 4.0),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    training = Compose(
        [
            crop,
            _convert_to_rgb,
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    prediction = Compose(
        [
            Resize(
                (384, 384),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            _convert_to_rgb,
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    recipe = {
        "preprocessing": {
            "train_transform": {
                "random_resized_crop_size": [384, 384],
                "random_resized_crop_scale": [0.9, 1.0],
                "random_resized_crop_ratio": [0.75, 1.3333],
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
            },
            "prediction_transform": {
                "resize_size": [384, 384],
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
            },
        }
    }

    geo_oof._enforce_transform_contract(training, prediction, recipe)

    assert crop.ratio == (0.75, 1.3333)


def test_geo_projection_reads_only_ordered_a_columns_and_ignores_poisoned_i_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    files = ("images/a.jpg", "images/b.jpg")
    sources: list[Path] = []
    for position, file_name in enumerate(files):
        offset = np.float32((position + 1) * 1_000)
        # Deliberately shuffle A columns and surround them with unusable legacy
        # I columns.  The M2 projection must request its exact allow-list.
        data: dict[str, object] = {
            "I0000": ["poison-do-not-convert"],
            **{
                column: [offset + np.float32(index)]
                for index, column in reversed(list(enumerate(GEO_COLUMNS)))
            },
            "file": [file_name],
            "I1151": [np.nan],
        }
        source = tmp_path / f"source_{position}.parquet"
        pd.DataFrame(data).to_parquet(source, index=False)
        sources.append(source)

    context = _minimal_context(tmp_path, source_tables=sources)
    requested = pd.DataFrame(
        {
            "file": [files[1], files[0]],
            "file_lower": [files[1].casefold(), files[0].casefold()],
        }
    )
    observed_columns: list[tuple[str, ...]] = []
    real_read_parquet = pd.read_parquet

    def read_parquet_spy(*args, **kwargs):
        observed_columns.append(tuple(kwargs.get("columns", ())))
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(geo_oof.pd, "read_parquet", read_parquet_spy)

    projected = geo_oof._geo_source_projection(context, requested)

    assert observed_columns == [("file", *GEO_COLUMNS)] * 2
    assert projected.dtype == np.float32
    np.testing.assert_array_equal(
        projected[0],
        np.arange(2_000, 2_000 + len(GEO_COLUMNS), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        projected[1],
        np.arange(1_000, 1_000 + len(GEO_COLUMNS), dtype=np.float32),
    )


def test_cublas_workspace_configuration_is_set_before_m2_core_import():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    runner = PROJECT_ROOT / "tools" / "run_multimodal_geo_helpfulness_m2.py"
    source = runner.read_text(encoding="utf-8")
    setting = 'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"'
    core_import = "from multimodal.geo_helpfulness_oof import"
    assert source.index(setting) < source.index(core_import)


def test_reproduced_output_uses_numeric_tolerances_but_exact_predictions():
    assignments = _prediction_assignments()
    reference_logits = _logits_by_mode(len(assignments))
    reference = build_output_table(
        assignments,
        seed=1,
        logits_by_mode=reference_logits,
        include_fold=True,
    )

    within_tolerance = {mode: values.copy() for mode, values in reference_logits.items()}
    within_tolerance["image_only"][0, 0] += 5.0e-7
    reproduced = build_output_table(
        assignments,
        seed=1,
        logits_by_mode=within_tolerance,
        include_fold=True,
    )
    result = validate_reproduced_output(reference, reproduced, include_fold=True)
    assert result == {
        "valid": True,
        "row_count": len(assignments),
        "atol": 1.0e-6,
        "rtol": 1.0e-6,
    }

    outside_tolerance = {mode: values.copy() for mode, values in reference_logits.items()}
    outside_tolerance["image_only"][0, 0] += 1.0e-2
    numerically_different = build_output_table(
        assignments,
        seed=1,
        logits_by_mode=outside_tolerance,
        include_fold=True,
    )
    with pytest.raises(M2ArtifactError, match="numerically"):
        validate_reproduced_output(reference, numerically_different, include_fold=True)

    changed_prediction = {mode: values.copy() for mode, values in reference_logits.items()}
    changed_prediction["image_only"][0, 0] = (
        changed_prediction["image_only"][0, -1] + 1.0
    )
    prediction_drift = build_output_table(
        assignments,
        seed=1,
        logits_by_mode=changed_prediction,
        include_fold=True,
    )
    with pytest.raises(M2ArtifactError, match="exactly in image_pred"):
        validate_reproduced_output(reference, prediction_drift, include_fold=True)


def test_workflow_lock_rejects_a_concurrent_owner_and_is_reusable(tmp_path: Path):
    lock_path = tmp_path / ".m2.lock"

    with geo_oof._exclusive_workflow_lock(lock_path):
        assert lock_path.read_text(encoding="utf-8") == f"pid={os.getpid()}\n"
        with pytest.raises(M2ArtifactError, match="another M2 process owns workflow lock"):
            with geo_oof._exclusive_workflow_lock(lock_path):
                pass

    with geo_oof._exclusive_workflow_lock(lock_path):
        assert lock_path.is_file()

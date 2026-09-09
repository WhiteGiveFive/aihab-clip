"""Synthetic final-M4 integration tests, including the post-receipt boundary."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_router_m4_analysis as analysis
from multimodal import geo_helpfulness_router_m4_common as common
from multimodal import geo_helpfulness_router_candidate_state as model


def _forbidden(*args, **kwargs):
    raise AssertionError("post-receipt label reading or selection is forbidden")


def _model_state(variant, seed, C):
    indices = model.variant_feature_indices(variant)
    columns = model.full_feature_columns()
    return model.seal_candidate_state(
        {
            "schema_version": model.SCHEMA_VERSION,
            "protocol_id": "protocol_v1",
            "training_seed": seed,
            "variant_id": variant,
            "C": float(C),
            "C_hex": float(C).hex(),
            "target_order": list(model.TARGET_ORDER),
            "classes": [0, 1, 2, 3],
            "fit_role": "development_train_oof",
            "fit_row_count": 3378,
            "fit_row_identity_sha256": "a" * 64,
            "study_spec_sha256": "b" * 64,
            "dataset_manifest_sha256": "c" * 64,
            "feature_input_names": columns,
            "feature_indices": indices,
            "feature_names": [columns[index] for index in indices],
            "fit_parameters": dict(model.FROZEN_FIT_PARAMETERS, C=float(C)),
            "implementation": dict(model.IMPLEMENTATION),
            "coef": np.zeros((4, len(indices))).tolist(),
            "intercept": [0.0] * 4,
            "n_iter": [0],
        }
    )


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    n = 40
    monkeypatch.setattr(analysis, "VALIDATION_ROWS", n)
    monkeypatch.setattr(analysis, "VALIDATION_PLOTS", n // 2)
    monkeypatch.setattr(
        analysis,
        "_fit_incapable",
        lambda: {"forbidden_imports_loaded": [], "scoring_only_process": True},
    )
    root = tmp_path / "protocol_v1"
    router = root / "router"
    protocol = root / "protocol"
    router.mkdir(parents=True)
    protocol.mkdir()
    ontology = [
        {
            "dense_id": index,
            "canonical_l3_id": index + 1,
            "label_name": f"habitat-{index}",
        }
        for index in range(18)
    ]
    config = {
        "class_ontology": {"classes": ontology},
        "artifact_contract": {
            "development_selection_receipt": {
                "required_fields": [
                    "schema_version",
                    "protocol_id",
                    "candidate_set_hash",
                    "candidate_prediction_hash",
                    "train_oof_parent_hash",
                    "validation_output_parent_hash",
                    "ordered_validation_target_hash",
                    "ontology_hash",
                    "selected_C_hex",
                    "threshold_specification",
                    "router_frozen_state_hashes",
                    "validation_metrics",
                    "go_no_go",
                ]
            }
        },
    }
    uid = [f"{index:064x}" for index in range(n)]
    plots = [f"plot-{index // 2:03d}" for index in range(n)]
    labels = np.asarray([(index // 2) % 18 for index in range(n)], dtype=np.int8)
    assignments = pd.DataFrame(
        {
            "protocol_id": pd.Series(["protocol_v1"] * n, dtype="string"),
            "row_uid": pd.Series(uid, dtype="string"),
            "plot_idx": pd.Series(plots, dtype="string"),
            "development_role": pd.Series(["validation"] * n, dtype="string"),
            "label_id_dense": labels,
            "canonical_l3_id": (labels + 1).astype(np.int8),
            "label_name": pd.Series(
                [f"habitat-{label}" for label in labels], dtype="string"
            ),
        }
    )
    assignment_path = protocol / "development_assignments.parquet"
    assignments.to_parquet(assignment_path, index=False)
    comparator_rows = []
    for seed in model.TRAINING_SEEDS:
        for index, label in enumerate(labels):
            target = index % 4
            raw = int(label if target in (1, 2) else (label + 1) % 18)
            geo = int(label if target in (0, 2) else (label + 2) % 18)
            row = {
                "schema_version": "geo_helpfulness.m4_comparator_inputs.v1",
                "protocol_id": "protocol_v1",
                "row_uid": uid[index],
                "plot_idx": plots[index],
                "training_seed": seed,
                "image_pred": int(label),
                "geo_pred": geo,
                "raw_pred": raw,
                "image_geo_pred_pair": int(label) * 18 + geo,
                "geo_raw_pred_pair": geo * 18 + raw,
                "geo_minus_raw_confidence": 0.0,
                "image_geo_jsd": 0.0,
                "geo_minus_raw_margin": 0.0,
            }
            for prefix, predicted in (
                ("image", int(label)),
                ("geo", geo),
                ("raw", raw),
            ):
                p = np.full(18, 0.2 / 17, dtype=np.float64)
                p[predicted] = 0.8
                row[f"{prefix}_prob_calibrated"] = p
                row[f"{prefix}_prob_native_t1"] = p.copy()
            comparator_rows.append(row)
    comparator = (
        pd.DataFrame(comparator_rows)
        .sort_values(["row_uid", "training_seed"])
        .reset_index(drop=True)
    )
    states, candidates, matrices = {}, {}, {}
    candidate_root = router / "router_candidates_v2"
    candidate_root.mkdir()
    for seed in model.TRAINING_SEEDS:
        matrices[seed] = np.zeros((n, 727), dtype=np.float64)
    for variant in model.VARIANTS:
        frames = []
        for seed in model.TRAINING_SEEDS:
            identity = comparator.loc[
                comparator.training_seed.eq(seed),
                [
                    "schema_version",
                    "protocol_id",
                    "row_uid",
                    "plot_idx",
                    "training_seed",
                ],
            ].copy()
            identity["schema_version"] = (
                "geo_helpfulness.router_candidate_validation_predictions.v1"
            )
            for C in model.C_GRID:
                state = _model_state(variant, seed, C)
                filename = model.candidate_filename(variant, seed, C)
                states[filename] = state
                common.write_json(candidate_root / filename, state)
                frame = identity.copy()
                frame["C_hex"] = float(C).hex()
                q = model.predict_router_probabilities(matrices[seed], state)
                for column, values in zip(analysis.Q_COLUMNS, q.T):
                    frame[column] = values
                frame["score"] = q[:, 0] - q[:, 1]
                frames.append(frame)
        candidates[variant] = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["C_hex", "training_seed", "row_uid"])
            .reset_index(drop=True)
        )
    temperatures = {
        "schema_version": "geo_helpfulness.expert_temperatures.v1",
        "protocol_id": "protocol_v1",
        "fit_role": "development_train_oof",
        "states": [
            {
                "seed": seed,
                "mode": mode,
                "temperature": 2.0,
                "fit_role": "development_train_oof",
                "fit_row_count": 3378,
            }
            for seed in model.TRAINING_SEEDS
            for mode in ("image_only", "geo_only", "raw_concat")
        ],
    }
    transforms = {
        "schema_version": "geo_helpfulness.router_feature_transform.v1",
        "protocol_id": "protocol_v1",
        "fit_role": "development_train_oof",
        "states": [
            {
                "seed": seed,
                "fit_role": "development_train_oof",
                "fit_row_count": 3378,
                "numeric_input_columns": list(model.NUMERIC_INPUT_COLUMNS),
                "output_columns": model.full_feature_columns(),
                "mean": [0.0] * 25,
                "variance": [1.0] * 25,
                "scale": [1.0] * 25,
                "categorical_vocabularies": {
                    name: list(range(size))
                    for name, size in model.CATEGORICAL_FEATURE_SIZES
                },
            }
            for seed in model.TRAINING_SEEDS
        ],
    }
    schema = {
        "schema_version": "synthetic_feature_schema",
        "output_columns": model.full_feature_columns(),
    }
    never = {"kind": "never_intervene", "value_hex": None, "comparison": "strict_gt"}
    heuristics = {
        "validation_labels_scored": False,
        "heuristics": {
            name: {"threshold_specification": deepcopy(never)}
            for name in ("image_geo_jsd", "geo_minus_raw_margin")
        },
    }
    common.write_json(router / "expert_temperatures.json", temperatures)
    common.write_json(router / "router_feature_transform.json", transforms)
    common.write_json(
        router / "targets_and_feature_contract/router_feature_schema.json", schema
    )
    common.write_json(router / "router_validation_v1/oof_heuristics.json", heuristics)
    common.write_json(
        router / "router_validation_v1/manifest.json", {"manifest_sha256": "d" * 64}
    )
    common.write_json(
        root / "development_validation/aggregate_manifest.json",
        {"manifest_sha256": "e" * 64},
    )
    common.write_json(
        root / "development_train_oof/aggregate_manifest.json",
        {"manifest_sha256": "f" * 64},
    )
    candidate_manifest = {"manifest_sha256": "b" * 64}
    dataset = {
        "manifest_sha256": "c" * 64,
        "parent_lineage": {
            "m1_m2_parent_artifact_hashes": {
                "development_train_oof_aggregate": {
                    "manifest_sha256": "f" * 64,
                    "content_sha256": "a" * 64,
                },
                "class_map": {"content_sha256": "a" * 64},
            }
        },
    }
    snapshot = {
        str(path): common.file_hash(path) for path in root.rglob("*") if path.is_file()
    }
    value = {
        "protocol_root": root,
        "snapshot": snapshot,
        "config": config,
        "comparator": comparator,
        "candidates": candidates,
        "states": states,
        "temperatures": temperatures,
        "transforms": transforms,
        "heuristics": heuristics,
        "candidate_manifest": candidate_manifest,
        "dataset_manifest": dataset,
        "comparator_manifest": {
            "assignment_file_sha256": common.file_hash(assignment_path),
            "validation_labels_read": False,
            "manifest_sha256": "9" * 64,
            "prediction_manifest_sha256": "d" * 64,
        },
        "declaration": {
            "prediction_manifest_sha256": "d" * 64,
            "conventions_sha256": "8" * 64,
        },
        "matrices": matrices,
    }
    monkeypatch.setattr(analysis, "_load_inputs", lambda protocol_root: value)
    return value


def test_create_targets_accepts_string_extension_storage_and_exact_truth_table(inputs):
    targets = analysis._create_targets(inputs)
    assert list(targets) == list(analysis.TARGET_COLUMNS)
    assert len(targets) == 160
    assert set(targets.target_state) == set(analysis.TARGET_ORDER)
    assert analysis._identities(targets) == analysis._identities(inputs["comparator"])
    analysis._validate_targets(targets, inputs)


def test_create_targets_reads_only_declared_assignment_columns_and_validation_rows(
    inputs, monkeypatch
):
    path = inputs["protocol_root"] / "protocol/development_assignments.parquet"
    assignment = pq.read_table(path).to_pandas()
    training = assignment.iloc[[0]].copy()
    training["row_uid"] = "f" * 64
    training["plot_idx"] = "training-only-plot"
    training["development_role"] = "train"
    joined = pd.concat([training, assignment], ignore_index=True)
    joined["forbidden_unused_column"] = "must not be requested"
    joined.to_parquet(path, index=False)
    inputs["comparator_manifest"]["assignment_file_sha256"] = common.file_hash(path)
    observed = []
    original = pq.read_table

    def read_table(filename, *args, **kwargs):
        if Path(filename) == path:
            observed.append(kwargs.get("columns"))
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", read_table)
    targets = analysis._create_targets(inputs)
    assert observed == [list(analysis.ASSIGNMENT_COLUMNS)]
    assert len(targets) == 160
    assert "f" * 64 not in set(targets.row_uid)


def test_selected_rows_and_effective_policies_use_frozen_views(inputs):
    targets = analysis._create_targets(inputs)
    selected = {
        variant: {
            "regularization": {"selected_C_hex": (0.01).hex()},
            "threshold": {
                "threshold_specification": {
                    "kind": "never_intervene",
                    "value_hex": None,
                    "comparison": "strict_gt",
                }
            },
        }
        for variant in analysis.VARIANTS
    }
    specifications, scores, actions = analysis._policies(
        inputs, targets, {"variants": selected}
    )
    assert set(actions) == set(analysis.POLICIES)
    assert all(not actions[variant].any() for variant in analysis.VARIANTS)
    assert np.array_equal(
        actions["always_geo_on_geo_raw_disagreement"],
        targets.raw_pred.ne(targets.geo_pred),
    )
    assert not actions["higher_calibrated_geo_confidence_on_geo_raw_disagreement"].any()
    assert all(mask.dtype == bool for mask in actions.values())


def _fingerprints(root):
    return {
        str(path.relative_to(root)): (common.file_hash(path), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _disable_post_receipt_inputs(monkeypatch, inputs):
    monkeypatch.setattr(analysis, "_create_targets", _forbidden)
    monkeypatch.setattr(analysis, "_make_selection", _forbidden)
    monkeypatch.setattr(analysis, "select_common_c", _forbidden)
    monkeypatch.setattr(analysis, "select_common_threshold", _forbidden)
    original = pq.read_table
    assignments = inputs["protocol_root"] / "protocol/development_assignments.parquet"

    def read_table(path, *args, **kwargs):
        if Path(path) == assignments:
            _forbidden()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", read_table)


def test_build_publishes_full_frozen_bundle_receipt_then_manifest(inputs, monkeypatch):
    events = []
    create_targets = analysis._create_targets
    write_bytes, write_json, publish_manifest = (
        analysis.write_bytes,
        analysis.write_json,
        analysis.publish_manifest,
    )

    def record_bytes(path, payload):
        events.append(Path(path).name)
        return write_bytes(path, payload)

    def record_json(path, value):
        events.append(Path(path).name)
        return write_json(path, value)

    def record_manifest(root, **kwargs):
        events.append("manifest.json")
        return publish_manifest(root, **kwargs)

    def record_label_access(value):
        events.append("assignment_labels_opened")
        return create_targets(value)

    monkeypatch.setattr(analysis, "write_bytes", record_bytes)
    monkeypatch.setattr(analysis, "write_json", record_json)
    monkeypatch.setattr(analysis, "publish_manifest", record_manifest)
    monkeypatch.setattr(analysis, "_create_targets", record_label_access)
    result = analysis.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    receipt = common.read_json(root / "development_selection_receipt.json")
    selection = common.read_json(root / "selection.json")
    frozen = common.read_json(root / "router_frozen_state.json")
    assert result["status"] == "built"
    assert result["assignment_label_columns_opened"] is True
    assert result["primary_decision"] == "no_go"
    assert events[-2:] == ["development_selection_receipt.json", "manifest.json"]
    assert events.index("build_declaration.json") < events.index(
        "validation_targets.parquet"
    )
    assert set(_fingerprints(root)) == analysis.BASE_FILES | set(
        analysis._selected_model_names(selection)
    )
    assert len(list((root / "models").glob("*.json"))) == 12
    assert len(receipt["router_frozen_state_hashes"]) == 4
    assert (
        len(receipt["result_artifact_file_sha256"]) == len(analysis.BASE_FILES) + 12 - 2
    )
    assert all(
        chosen["regularization"]["selected_C"] == 0.01
        for chosen in selection["variants"].values()
    )
    assert all(
        chosen["threshold"]["threshold_specification"]["kind"] == "never_intervene"
        for chosen in selection["variants"].values()
    )
    assert frozen["router_output_calibrator"] == "none"
    for variant in analysis.VARIANTS:
        assert set(frozen["variants"][variant]["seeds"]) == {"1", "2", "3", "4"}
        for state in frozen["variants"][variant]["seeds"].values():
            assert state["target_order"] == list(analysis.TARGET_ORDER)
            assert state["require_geo_raw_disagreement"] is True
            assert state["comparison"] == "strict_gt"
    assert all(
        path.stat().st_mode & 0o222 == 0 for path in root.rglob("*") if path.is_file()
    )


def test_validate_and_repeat_build_never_reopen_assignments_or_select(
    inputs, monkeypatch
):
    built = analysis.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    before = _fingerprints(root)
    _disable_post_receipt_inputs(monkeypatch, inputs)
    validated = analysis.validate_m4_result(inputs["protocol_root"])
    repeated = analysis.build_m4_result(inputs["protocol_root"])
    assert validated["status"] == "validated"
    assert repeated["status"] == "reused_valid"
    assert (
        validated["manifest_sha256"]
        == repeated["manifest_sha256"]
        == built["manifest_sha256"]
    )
    assert validated["assignment_label_columns_opened"] is False
    assert repeated["reselection_calls"] == 0
    assert _fingerprints(root) == before


def test_interrupt_after_receipt_resumes_without_labels_or_selection(
    inputs, monkeypatch
):
    finish = analysis._finish_manifest
    monkeypatch.setattr(
        analysis,
        "_finish_manifest",
        lambda *args: (_ for _ in ()).throw(RuntimeError("interrupted after receipt")),
    )
    with pytest.raises(RuntimeError, match="interrupted after receipt"):
        analysis.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    assert (root / "development_selection_receipt.json").exists()
    assert not (root / "manifest.json").exists()
    before = _fingerprints(root)
    monkeypatch.setattr(analysis, "_finish_manifest", finish)
    _disable_post_receipt_inputs(monkeypatch, inputs)
    result = analysis.build_m4_result(inputs["protocol_root"])
    assert result["status"] == "reused_valid"
    assert {
        name: value
        for name, value in _fingerprints(root).items()
        if name != "manifest.json"
    } == before


def test_interrupted_before_receipt_resumes_saved_targets_and_selection(
    inputs, monkeypatch
):
    write_json = analysis.write_json

    def interrupted(path, value):
        if Path(path).name == "candidate_metrics.json":
            raise RuntimeError("interrupted before reports")
        return write_json(path, value)

    monkeypatch.setattr(analysis, "write_json", interrupted)
    with pytest.raises(RuntimeError, match="interrupted before reports"):
        analysis.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    before = _fingerprints(root)
    assert set(before) == {
        "build_declaration.json",
        "selection.json",
        "validation_targets.parquet",
    }
    monkeypatch.setattr(analysis, "write_json", write_json)
    # Before the receipt, labels may be reread to authenticate saved partial targets.
    monkeypatch.setattr(analysis, "_make_selection", _forbidden)
    monkeypatch.setattr(analysis, "select_common_c", _forbidden)
    monkeypatch.setattr(analysis, "select_common_threshold", _forbidden)
    result = analysis.build_m4_result(inputs["protocol_root"])
    assert result["status"] == "built"
    assert {name: _fingerprints(root)[name] for name in before} == before


def test_known_partial_files_without_owned_declaration_are_not_adopted(
    inputs, monkeypatch
):
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    root.mkdir()
    common.write_json(root / "selection.json", {"schema_version": "unowned"})
    monkeypatch.setattr(analysis, "_create_targets", _forbidden)
    monkeypatch.setattr(analysis, "_make_selection", _forbidden)
    with pytest.raises(ValueError, match="declaration|owned|partial"):
        analysis.build_m4_result(inputs["protocol_root"])


def test_partial_target_labels_must_match_assignment_source(inputs, monkeypatch):
    monkeypatch.setattr(
        analysis,
        "_make_selection",
        lambda *args: (_ for _ in ()).throw(RuntimeError("interrupted after targets")),
    )
    with pytest.raises(RuntimeError, match="interrupted after targets"):
        analysis.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    path = root / "validation_targets.parquet"
    targets = pq.read_table(path).to_pandas()
    mask = targets.plot_idx.eq(targets.plot_idx.iloc[0])
    targets.loc[mask, "label_id_dense"] = 17
    targets.loc[mask, "canonical_l3_id"] = 18
    targets.loc[mask, "label_name"] = "habitat-17"
    raw_correct = targets.raw_pred.eq(targets.label_id_dense).to_numpy()
    geo_correct = targets.geo_pred.eq(targets.label_id_dense).to_numpy()
    targets["target_state"] = np.select(
        [
            ~raw_correct & geo_correct,
            raw_correct & ~geo_correct,
            raw_correct & geo_correct,
        ],
        analysis.TARGET_ORDER[:3],
        default="both_wrong",
    )
    analysis._validate_targets(targets, inputs)
    path.chmod(0o644)
    path.write_bytes(
        common.table_bytes(analysis._frame_table(targets, analysis._target_schema()))
    )
    monkeypatch.setattr(analysis, "_make_selection", _forbidden)
    with pytest.raises(ValueError, match="target|reconstructed|source|differs|changed"):
        analysis.build_m4_result(inputs["protocol_root"])
    assert not (root / "development_selection_receipt.json").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "selection.json",
        "router_frozen_state.json",
        "development_selection_receipt.json",
        "manifest.json",
    ],
)
def test_committed_corruption_is_never_reselected_or_overwritten(
    inputs, monkeypatch, filename
):
    analysis.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    path = root / filename
    value = common.read_json(path)
    value["unexpected_tamper"] = True
    path.chmod(0o644)
    path.write_bytes(common.canonical_bytes(value) + b"\n")
    before = _fingerprints(root)
    _disable_post_receipt_inputs(monkeypatch, inputs)
    with pytest.raises(ValueError, match="changed|differs|invalid|mismatch"):
        analysis.build_m4_result(inputs["protocol_root"])
    assert _fingerprints(root) == before


def test_stale_committed_parent_blocks_without_assignment_parsing(inputs, monkeypatch):
    analysis.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    parent = inputs["protocol_root"] / "router/expert_temperatures.json"
    parent.chmod(0o644)
    parent.write_bytes(parent.read_bytes() + b" ")
    before = _fingerprints(root)
    _disable_post_receipt_inputs(monkeypatch, inputs)
    with pytest.raises(ValueError, match="parent or code changed"):
        analysis.build_m4_result(inputs["protocol_root"])
    assert _fingerprints(root) == before


def test_parent_mutation_during_computation_prevents_receipt(inputs, monkeypatch):
    original = analysis._frozen_payloads

    def mutate_after_payloads(*args):
        value = original(*args)
        path = inputs["protocol_root"] / "router/expert_temperatures.json"
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        return value

    monkeypatch.setattr(analysis, "_frozen_payloads", mutate_after_payloads)
    with pytest.raises(ValueError, match="parent or code changed"):
        analysis.build_m4_result(inputs["protocol_root"])
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    assert not (root / "development_selection_receipt.json").exists()
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize(
    "defect", ["missing_threshold", "duplicate_c", "wrong_display"]
)
def test_saved_selection_requires_complete_declared_candidate_grid(inputs, defect):
    targets = analysis._create_targets(inputs)
    selected = analysis._make_selection(inputs, targets)
    chosen = selected["variants"][analysis.VARIANTS[0]]
    if defect == "missing_threshold":
        threshold = chosen["threshold"]
        threshold["candidate_curve"] = threshold["candidate_curve"][:1]
        threshold["candidate_count"] = 1
        threshold["candidate_specifications_sha256"] = common.canonical_hash(
            [
                record["threshold_specification"]
                for record in threshold["candidate_curve"]
            ]
        )
    elif defect == "duplicate_c":
        candidate_losses = chosen["regularization"]["candidate_losses"]
        candidate_losses[-1] = deepcopy(candidate_losses[0])
    else:
        chosen["threshold"]["candidate_curve"][-1]["threshold_display"] = 0.125
    selected["selection_sha256"] = common.canonical_hash(
        {key: value for key, value in selected.items() if key != "selection_sha256"}
    )
    with pytest.raises(
        ValueError, match="complete|grid|candidate|threshold|C diagnostics"
    ):
        analysis._validate_saved_selection(
            selected, inputs, targets, analysis._candidate_metrics(inputs, targets)
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("protocol_id", "wrong"),
        ("schema_version", "wrong"),
        ("label_id_dense", 18),
        ("target_state", "invalid"),
    ],
)
def test_invalid_saved_targets_are_rejected(inputs, column, value):
    targets = analysis._create_targets(inputs)
    targets.loc[0, column] = value
    with pytest.raises(ValueError, match="identity|ontology|truth table"):
        analysis._validate_targets(targets, inputs)


@pytest.mark.parametrize(
    "column",
    ["image_pred", "geo_pred", "raw_pred", "image_geo_pred_pair", "geo_raw_pred_pair"],
)
def test_target_expert_predictions_and_pair_codes_match_sealed_comparator(
    inputs, column
):
    targets = analysis._create_targets(inputs)
    targets.loc[0, column] = (int(targets.loc[0, column]) + 1) % (
        324 if column.endswith("_pair") else 18
    )
    raw_correct = targets.raw_pred.eq(targets.label_id_dense).to_numpy()
    geo_correct = targets.geo_pred.eq(targets.label_id_dense).to_numpy()
    targets["target_state"] = np.select(
        [
            ~raw_correct & geo_correct,
            raw_correct & ~geo_correct,
            raw_correct & geo_correct,
        ],
        analysis.TARGET_ORDER[:3],
        default="both_wrong",
    )
    with pytest.raises(ValueError, match="expert|prediction|pair|comparator"):
        analysis._validate_targets(targets, inputs)


def test_arrow_comparison_preserves_null_semantics():
    import pyarrow as pa

    schema = pa.schema([pa.field("undefined_rate", pa.float64(), nullable=True)])
    expected = pa.Table.from_arrays(
        [pa.array([None, 0.5], type=pa.float64())], schema=schema
    )
    same = pa.Table.from_arrays(
        [pa.array([None, 0.5], type=pa.float64())], schema=schema
    )
    common.compare_tables(same, expected)
    invalid = pa.Table.from_arrays(
        [pa.array([float("nan"), 0.5], type=pa.float64())], schema=schema
    )
    with pytest.raises(ValueError, match="null mask"):
        common.compare_tables(invalid, expected)
    with pytest.raises(ValueError, match="nonfinite"):
        common.compare_tables(invalid, invalid)


@pytest.mark.parametrize("kind", ["unknown", "symlink", "directory"])
def test_unknown_or_symlink_output_blocks_before_label_access(
    inputs, monkeypatch, kind
):
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    root.mkdir()
    if kind == "unknown":
        common.write_bytes(root / "unknown.json", b"{}")
    elif kind == "symlink":
        (root / "selection.json").symlink_to(
            inputs["protocol_root"] / "router/expert_temperatures.json"
        )
    else:
        (root / "unexpected").mkdir()
    monkeypatch.setattr(analysis, "_load_inputs", _forbidden)
    monkeypatch.setattr(analysis, "_create_targets", _forbidden)
    with pytest.raises(ValueError, match="allowlist|unknown|symlink"):
        analysis.build_m4_result(inputs["protocol_root"])


def test_lock_contention_blocks_before_label_access(inputs, monkeypatch):
    root = inputs["protocol_root"] / analysis.RESULT_DIR
    monkeypatch.setattr(analysis, "_load_inputs", _forbidden)
    with common.exclusive_lock(root):
        with pytest.raises(ValueError, match="lock"):
            analysis.build_m4_result(inputs["protocol_root"])


def test_scoring_module_fresh_import_is_fit_incapable():
    script = """import sys
from multimodal import geo_helpfulness_router_m4_analysis as analysis
assert analysis._fit_incapable()['scoring_only_process']
for name in ('torch', 'scipy', 'sklearn', 'multimodal.geo_helpfulness_router_dataset',
             'multimodal.geo_helpfulness_router_candidate_model',
             'multimodal.geo_helpfulness_expert_outputs'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=common.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

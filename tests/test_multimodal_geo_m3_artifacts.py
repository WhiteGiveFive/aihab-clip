from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_targets_features as m3


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "tools" / "run_multimodal_geo_helpfulness_m3.py"


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
    first_predictions = ((1, 0), (0, 1), (0, 0), (1, 1))
    rows: list[dict[str, object]] = []
    for seed, (raw_pred, geo_pred) in enumerate(first_predictions, start=1):
        rows.extend(
            [
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
                },
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
                },
            ]
        )
    return pd.DataFrame(rows), assignments


def _prepared(tmp_path: Path) -> m3._PreparedM3:
    oof, assignments = _target_inputs()
    allowlist = {
        family: list(columns)
        for family, columns in m3.FEATURE_FAMILIES.items()
    }
    forbidden = [
        "label",
        "target",
        "true_class",
        "correctness",
        "correct",
        "y_true",
        "true_probability",
        "true_nll",
        "nll_advantage",
    ]
    context = SimpleNamespace(
        output_root=tmp_path,
        lineage_token="parent-v1",
        config={
            "protocol_id": "protocol_v1",
            "router": {
                "feature_allowlist": allowlist,
                "forbidden_feature_patterns": forbidden,
            },
        },
    )
    return m3._PreparedM3(
        context=context,
        oof_table=pa.table({"synthetic": [1]}),
        producer_manifest_hashes={},
        producer_manifest_file_hashes={},
        aggregate_manifest={},
        aggregate_validation={"valid": True},
        targets=m3.build_router_target_table(
            oof,
            assignments,
            protocol_id="protocol_v1",
            expected_seeds=m3.TRAINING_SEEDS,
        ),
        feature_schema=m3.build_router_feature_schema(
            configured_allowlist=allowlist,
        ),
        prevalence=m3.build_target_prevalence_report(
            oof,
            assignments,
            protocol_id="protocol_v1",
            expected_seeds=m3.TRAINING_SEEDS,
        ),
        leakage_audit=m3.build_feature_leakage_audit(
            configured_allowlist=allowlist,
            forbidden_patterns=forbidden,
        ),
    )


def _synthetic_manifest(
    prepared: m3._PreparedM3,
    *,
    bundle_root: Path,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": m3.M3_MANIFEST_SCHEMA_VERSION,
        "protocol_id": prepared.context.config["protocol_id"],
        "lineage_token": prepared.context.lineage_token,
        "children": {
            name: m3.sha256_file(bundle_root / name)
            for name in m3.BUNDLE_CHILD_FILENAMES
        },
        "calibration_boundary": {
            "temperature_fitted_by_m3": False,
            "router_dataset_owner": "M4",
        },
        "source_access": {
            "development_validation_opened": False,
            "locked_test_sources_opened": False,
        },
    }
    payload["manifest_sha256"] = m3._manifest_self_hash(payload)
    return payload


def _patch_synthetic_preparation(
    monkeypatch: pytest.MonkeyPatch,
    prepared: m3._PreparedM3,
) -> None:
    monkeypatch.setattr(m3, "_prepare_m3", lambda **_kwargs: prepared)
    monkeypatch.setattr(m3, "_build_m3_manifest", _synthetic_manifest)


def test_bundle_publication_manifest_last_reuse_validation_and_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared = _prepared(tmp_path)
    _patch_synthetic_preparation(monkeypatch, prepared)
    published: list[str] = []
    real_publish = m3._exclusive_publish

    def publish_spy(staged: Path, destination: Path) -> None:
        published.append(destination.name)
        real_publish(staged, destination)

    monkeypatch.setattr(m3, "_exclusive_publish", publish_spy)

    created = m3.build_m3_bundle(artifact_root=tmp_path)

    assert created["status"] == "created"
    assert created["calibration_fitted"] is False
    assert created["router_dataset_materialized"] is False
    assert created["feature_leakage_audit_valid"] is True
    assert published == [*m3.BUNDLE_CHILD_FILENAMES, m3.MANIFEST_FILENAME]
    bundle_root = m3.m3_bundle_path(tmp_path)
    assert {path.name for path in bundle_root.iterdir()} == set(m3.BUNDLE_FILENAMES)
    assert not (bundle_root / "router_dataset.parquet").exists()
    assert not list(bundle_root.parent.glob(".targets_and_feature_contract.m3.staging-*"))

    reused = m3.build_m3_bundle(artifact_root=tmp_path)
    validated = m3.validate_m3_bundle(artifact_root=tmp_path)
    loaded = m3.load_validated_m3_bundle(artifact_root=tmp_path)

    assert reused["status"] == validated["status"] == "reused_valid"
    assert reused["manifest_sha256"] == validated["manifest_sha256"]
    assert loaded.root == bundle_root
    pd.testing.assert_frame_equal(
        loaded.targets,
        prepared.targets,
        check_dtype=False,
    )
    assert m3.validate_router_target_table(loaded.targets)["valid"] is True
    assert loaded.feature_schema == prepared.feature_schema
    assert loaded.target_prevalence == prepared.prevalence
    assert loaded.feature_leakage_audit == prepared.leakage_audit
    assert loaded.validation == validated


def test_uncommitted_owned_partial_bundle_is_recovered_but_unknown_entry_is_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared = _prepared(tmp_path)
    _patch_synthetic_preparation(monkeypatch, prepared)
    bundle_root = m3.m3_bundle_path(tmp_path)
    real_publish = m3._exclusive_publish
    interrupted = False

    def interrupt_after_first_publish(staged: Path, destination: Path) -> None:
        nonlocal interrupted
        real_publish(staged, destination)
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption after first child publication")

    monkeypatch.setattr(m3, "_exclusive_publish", interrupt_after_first_publish)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        m3.build_m3_bundle(artifact_root=tmp_path)

    partial = bundle_root / m3.TARGET_FILENAME
    receipt = m3._ownership_receipt_path(bundle_root)
    assert partial.is_file()
    assert receipt.is_file()
    assert not (bundle_root / m3.MANIFEST_FILENAME).exists()

    monkeypatch.setattr(m3, "_exclusive_publish", real_publish)

    result = m3.build_m3_bundle(artifact_root=tmp_path)

    assert result["status"] == "created"
    assert not receipt.exists()

    other_root = tmp_path / "unknown-entry-case"
    other = _prepared(other_root)
    _patch_synthetic_preparation(monkeypatch, other)
    other_bundle = m3.m3_bundle_path(other_root)
    other_bundle.mkdir(parents=True)
    m3._create_ownership_receipt(
        m3._ownership_receipt_path(other_bundle),
        bundle_root=other_bundle,
        protocol_id="protocol_v1",
    )
    unknown = other_bundle / "user-notes.txt"
    unknown.write_text("do not delete", encoding="utf-8")

    with pytest.raises(m3.M3ArtifactError, match="unowned|unknown|cannot be cleaned"):
        m3.build_m3_bundle(artifact_root=other_root)
    assert unknown.read_text(encoding="utf-8") == "do not delete"


def test_committed_tamper_is_rejected_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared = _prepared(tmp_path)
    _patch_synthetic_preparation(monkeypatch, prepared)
    m3.build_m3_bundle(artifact_root=tmp_path)
    schema_path = m3.m3_bundle_path(tmp_path) / m3.FEATURE_SCHEMA_FILENAME
    os.chmod(schema_path, 0o644)
    corrupted = b'{"corrupt":true}\n'
    schema_path.write_bytes(corrupted)

    with pytest.raises(m3.M3ArtifactError, match="schema|reproduce|stale|fingerprint"):
        m3.build_m3_bundle(artifact_root=tmp_path)
    assert schema_path.read_bytes() == corrupted


def test_stale_parent_lineage_is_rejected_without_republication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared = _prepared(tmp_path)
    _patch_synthetic_preparation(monkeypatch, prepared)
    created = m3.build_m3_bundle(artifact_root=tmp_path)
    manifest_path = m3.m3_bundle_path(tmp_path) / m3.MANIFEST_FILENAME
    original_manifest = manifest_path.read_bytes()
    prepared.context.lineage_token = "parent-v2"

    with pytest.raises(m3.M3ArtifactError, match="lineage|stale|manifest"):
        m3.build_m3_bundle(artifact_root=tmp_path)

    assert manifest_path.read_bytes() == original_manifest
    assert json.loads(original_manifest)["manifest_sha256"] == created["manifest_sha256"]


def test_initial_publication_rejects_m4_dataset_but_committed_bundle_allows_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared = _prepared(tmp_path)
    _patch_synthetic_preparation(monkeypatch, prepared)
    router_root = m3.m3_bundle_path(tmp_path).parent
    router_root.mkdir(parents=True)
    router_dataset = router_root / "router_dataset.parquet"
    m4_payload = b"synthetic M4-owned dataset"
    router_dataset.write_bytes(m4_payload)

    with pytest.raises(m3.M3ArtifactError, match="absent|initial|M4"):
        m3.build_m3_bundle(artifact_root=tmp_path)
    assert router_dataset.read_bytes() == m4_payload
    assert not m3.m3_bundle_path(tmp_path).exists()

    router_dataset.unlink()
    assert m3.build_m3_bundle(artifact_root=tmp_path)["status"] == "created"
    router_dataset.write_bytes(m4_payload)

    assert m3.validate_m3_bundle(artifact_root=tmp_path)["valid"] is True
    assert m3.load_validated_m3_bundle(artifact_root=tmp_path).validation["valid"] is True
    assert m3.build_m3_bundle(artifact_root=tmp_path)["status"] == "reused_valid"
    assert router_dataset.read_bytes() == m4_payload


def test_real_manifest_freezes_lineage_schemas_and_m3_m4_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared = _prepared(tmp_path)
    inputs = tmp_path / "sealed-inputs"
    inputs.mkdir()
    protocol_manifest = inputs / "protocol_manifest.json"
    assignments_path = inputs / "development_assignments.parquet"
    resolved_path = inputs / "resolved_protocol.yaml"
    protocol_manifest.write_text("{}\n", encoding="utf-8")
    assignments_path.write_bytes(b"synthetic assignments")
    resolved_path.write_text("protocol_id: protocol_v1\n", encoding="utf-8")
    aggregate_root = tmp_path / "development_train_oof"
    aggregate_root.mkdir()
    aggregate_path = aggregate_root / "development_train_oof_model_outputs.parquet"
    aggregate_manifest_path = aggregate_root / "aggregate_manifest.json"
    aggregate_path.write_bytes(b"synthetic aggregate")
    aggregate_manifest_path.write_text("{}\n", encoding="utf-8")
    prepared.context.manifest_path = protocol_manifest
    prepared.context.assignments_path = assignments_path
    prepared.context.resolved_path = resolved_path
    sealed_assignments = _target_inputs()[1]
    sealed_assignments["source_split"] = "train"
    sealed_assignments["image_source"] = "/synthetic/images"
    sealed_assignments["canonical_l3_id"] = sealed_assignments[
        "label_id_dense"
    ].astype(np.int8)
    sealed_assignments["label_name"] = ["class-zero", "class-one"]
    prepared.context.assignments = sealed_assignments
    prepared.context.protocol_manifest = {
        "manifest_payload_sha256": "1" * 64,
        "effective_config_sha256": "2" * 64,
        "class_map_sha256": "3" * 64,
        "feature_allowlist_sha256": "4" * 64,
    }
    prepared.context.code_file_hashes = {"multimodal/m2.py": "5" * 64}
    prepared.context.code_hash = "6" * 64
    prepared.aggregate_manifest.update(
        {
            "content_sha256": "7" * 64,
            "manifest_sha256": "8" * 64,
        }
    )
    prepared.producer_manifest_hashes.update({"producer": "9" * 64})
    prepared.producer_manifest_file_hashes.update({"producer": "a" * 64})
    bundle_root = tmp_path / "manifest-staging"
    bundle_root.mkdir()
    (bundle_root / m3.TARGET_FILENAME).write_bytes(b"synthetic targets")
    m3._write_json_exclusive(
        bundle_root / m3.FEATURE_SCHEMA_FILENAME,
        prepared.feature_schema,
    )
    m3._write_json_exclusive(
        bundle_root / m3.PREVALENCE_FILENAME,
        prepared.prevalence,
    )
    m3._write_json_exclusive(
        bundle_root / m3.LEAKAGE_AUDIT_FILENAME,
        prepared.leakage_audit,
    )
    monkeypatch.setattr(
        m3,
        "validate_router_target_table",
        lambda *_args, **_kwargs: {
            "valid": True,
            "row_count": 13_512,
            "unique_image_count": 3_378,
            "plot_count": 1_300,
        },
    )

    manifest = m3._build_m3_manifest(prepared, bundle_root=bundle_root)

    assert manifest["manifest_sha256"] == m3._manifest_self_hash(manifest)
    assert set(manifest["schemas"]) == {
        "router_targets",
        "router_features",
        "target_prevalence",
        "feature_leakage_audit",
        "manifest",
    }
    assert manifest["target_table"]["unique_key"] == ["row_uid", "training_seed"]
    assert manifest["target_table"]["seed_aggregation"] == "none"
    assert manifest["feature_contract"] == {
        "filename": m3.FEATURE_SCHEMA_FILENAME,
        "schema_sha256": m3.canonical_sha256(prepared.feature_schema),
        "semantic_feature_count": 30,
        "probability_basis": m3.CALIBRATED_PROBABILITY_BASIS,
        "native_t1_primary_features_materialized": False,
        "calibrated_feature_rows_materialized": False,
        "router_dataset_materialized": False,
    }
    assert manifest["calibration_boundary"] == {
        "temperature_fitted_by_m3": False,
        "temperature_count": 0,
        "owner": "M4",
        "m4_temperature_count": 12,
        "router_dataset_owner": "M4",
    }
    assert manifest["source_access"]["validated_train_oof_producer_count"] == 16
    assert manifest["source_access"]["development_validation_outputs_opened"] is False
    assert manifest["source_access"]["final_in_sample_outputs_opened"] is False
    assert manifest["source_access"]["locked_test_sources_opened"] is False
    evidence = manifest["post_derivation_verification_evidence"]
    assert evidence["used_as_target_generation_input"] is False
    assert evidence["expected_protocol_v1_row_count"] == 13_512
    publication = manifest["publication"]
    assert publication["manifest_is_commit_marker"] is True
    assert publication["router_dataset_parquet"] == {
        "produced_by_m3": False,
        "absent_at_m3_publication": True,
        "later_owner": "M4",
    }


def test_prepare_m3_validates_only_four_oof_producers_per_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from multimodal import geo_helpfulness_oof as m2

    train_rows = 3_378
    train = pd.DataFrame(
        {
            "protocol_id": ["protocol_v1"] * train_rows,
            "row_uid": [f"{index + 1:064x}" for index in range(train_rows)],
            "file": [f"images/train-{index}.jpg" for index in range(train_rows)],
            "file_lower": [f"images/train-{index}.jpg" for index in range(train_rows)],
            "plot_idx": [f"plot-{index % 1_300}" for index in range(train_rows)],
            "label_id_dense": np.arange(train_rows, dtype=np.int64) % 18,
            "development_role": ["train"] * train_rows,
            "train_oof_fold": pd.array(
                np.arange(train_rows, dtype=np.int64) % 4,
                dtype="Int8",
            ),
        }
    )
    validation = pd.DataFrame(
        {
            "protocol_id": ["protocol_v1", "protocol_v1"],
            "row_uid": [f"{train_rows + index + 1:064x}" for index in range(2)],
            "file": ["images/validation-0.jpg", "images/validation-1.jpg"],
            "file_lower": ["images/validation-0.jpg", "images/validation-1.jpg"],
            "plot_idx": ["validation-plot-0", "validation-plot-1"],
            "label_id_dense": [0, 1],
            "development_role": ["validation", "validation"],
            "train_oof_fold": pd.array([pd.NA, pd.NA], dtype="Int8"),
        }
    )
    assignments = pd.concat([train, validation], ignore_index=True)
    context = SimpleNamespace(
        assignments=assignments,
        output_root=tmp_path,
        config={
            "protocol_id": "protocol_v1",
            "router": {
                "feature_allowlist": {},
                "forbidden_feature_patterns": [],
            },
        },
    )
    validated_specs: list[tuple[int, SimpleNamespace]] = []
    target_assignment_projections: list[pd.DataFrame] = []
    prevalence_assignment_projections: list[pd.DataFrame] = []

    def producer_specs(seed: int):
        return tuple(
            SimpleNamespace(
                include_fold=fold < 4,
                relative_directory=Path(f"seed-{seed}/stage-{fold}"),
                output_filename="outputs.parquet",
            )
            for fold in range(5)
        )

    def validate_producer(_context, spec, *, seed: int):
        validated_specs.append((seed, spec))
        return {"manifest_sha256": f"{seed}{len(validated_specs):063d}"[-64:]}

    sentinel = pa.table({"row": [1]})
    monkeypatch.setattr(m2, "load_frozen_context", lambda **_kwargs: context)
    monkeypatch.setattr(m2, "_producer_specs", producer_specs)
    monkeypatch.setattr(m2, "validate_producer", validate_producer)
    monkeypatch.setattr(m2, "read_output_parquet", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(m2, "_aggregate_table", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(
        m2,
        "_validate_aggregate_artifact",
        lambda *_args, **_kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        m2,
        "_aggregate_paths",
        lambda *_args, **_kwargs: (
            tmp_path / "development_train_oof" / "outputs.parquet",
            tmp_path / "development_train_oof" / "aggregate_manifest.json",
            None,
        ),
    )
    monkeypatch.setattr(m2, "logical_table_sha256", lambda _table: "same")
    monkeypatch.setattr(m3, "_validate_frozen_router_contract", lambda _config: None)
    monkeypatch.setattr(m3, "sha256_file", lambda _path: "f" * 64)
    monkeypatch.setattr(
        m3,
        "_read_json_mapping",
        lambda *_args, **_kwargs: {"content_sha256": "c" * 64},
    )
    def capture_target_projection(_oof, assignment_projection, **_kwargs):
        target_assignment_projections.append(assignment_projection.copy())
        return pd.DataFrame()

    def capture_prevalence_projection(_oof, assignment_projection, **_kwargs):
        prevalence_assignment_projections.append(assignment_projection.copy())
        return {}

    monkeypatch.setattr(m3, "build_router_target_table", capture_target_projection)
    monkeypatch.setattr(m3, "build_router_feature_schema", lambda **_kwargs: {})
    monkeypatch.setattr(m3, "build_target_prevalence_report", capture_prevalence_projection)
    monkeypatch.setattr(m3, "build_feature_leakage_audit", lambda **_kwargs: {})
    monkeypatch.setattr(m3, "_validate_real_acceptance", lambda _prepared: None)

    prepared = m3._prepare_m3(artifact_root=tmp_path)

    assert len(validated_specs) == 16
    assert {seed for seed, _spec in validated_specs} == set(m3.TRAINING_SEEDS)
    assert all(spec.include_fold is True for _seed, spec in validated_specs)
    assert prepared.aggregate_validation == {"valid": True}
    assert len(target_assignment_projections) == len(prevalence_assignment_projections) == 1
    expected_projection_columns = [
        "protocol_id",
        "row_uid",
        "file",
        "file_lower",
        "plot_idx",
        "label_id_dense",
        "development_role",
        "train_oof_fold",
    ]
    for projection in (
        target_assignment_projections[0],
        prevalence_assignment_projections[0],
    ):
        assert list(projection.columns) == expected_projection_columns
        assert len(projection) == train_rows
        assert set(projection["development_role"]) == {"train"}


def test_runner_exposes_only_build_validate_and_path_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    spec = importlib.util.spec_from_file_location("geo_m3_runner_for_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        runner,
        "build_m3_bundle",
        lambda **kwargs: calls.append(("build", kwargs)) or {"status": "created"},
    )
    monkeypatch.setattr(
        runner,
        "validate_m3_bundle",
        lambda **kwargs: calls.append(("validate", kwargs)) or {"valid": True},
    )
    config = tmp_path / "config.yaml"
    protocol = tmp_path / "protocol"
    artifacts = tmp_path / "artifacts"

    assert runner.main(
        [
            "build",
            "--config",
            str(config),
            "--protocol-dir",
            str(protocol),
            "--artifact-root",
            str(artifacts),
        ]
    ) == 0
    assert runner.main(["validate", "--artifact-root", str(artifacts)]) == 0
    assert [name for name, _kwargs in calls] == ["build", "validate"]
    assert calls[0][1] == {
        "config_path": config,
        "protocol_dir": protocol,
        "artifact_root": artifacts,
    }
    assert set(vars(runner._parser().parse_args(["build"]))) == {
        "command",
        "config",
        "protocol_dir",
        "artifact_root",
    }
    with pytest.raises(SystemExit):
        runner._parser().parse_args(["build", "--force-overwrite"])
    with pytest.raises(SystemExit):
        runner._parser().parse_args(["calibrate"])
    capsys.readouterr()

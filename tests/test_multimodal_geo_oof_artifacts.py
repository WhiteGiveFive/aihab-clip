from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import yaml

from multimodal import geo_helpfulness_oof as oof


def _assignments() -> pd.DataFrame:
    train_rows = 3_378
    validation_rows = 822
    rows = train_rows + validation_rows
    train_folds = [index % 4 for index in range(train_rows)]
    folds = pd.array(train_folds + [pd.NA] * validation_rows, dtype="Int8")
    roles = ["train"] * train_rows + ["validation"] * validation_rows
    return pd.DataFrame(
        {
            "schema_version": ["geo_helpfulness.development_assignments.v1"] * rows,
            "protocol_id": ["protocol_v1"] * rows,
            "row_uid": [f"{index + 1:064x}" for index in range(rows)],
            "file": [f"images/{index:05d}.jpg" for index in range(rows)],
            "file_lower": [f"images/{index:05d}.jpg" for index in range(rows)],
            "plot_idx": [f"plot_{index:05d}" for index in range(rows)],
            "source_split": ["train"] * rows,
            "image_source": ["./unused"] * rows,
            "label_id_dense": np.asarray(
                [index % oof.N_CLASSES for index in range(rows)], dtype=np.int8
            ),
            "canonical_l3_id": np.asarray(
                [index % oof.N_CLASSES for index in range(rows)], dtype=np.int8
            ),
            "label_name": [f"class_{index % oof.N_CLASSES}" for index in range(rows)],
            "development_role": roles,
            "train_oof_fold": folds,
        }
    )


def _context(tmp_path: Path) -> oof.FrozenM2Context:
    protocol_dir = (
        oof.PROJECT_ROOT
        / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/protocol"
    )
    config = yaml.safe_load((protocol_dir / "resolved_protocol.yaml").read_text())
    code_files, code_hash = oof._implementation_hashes()
    return oof.FrozenM2Context(
        protocol_dir=protocol_dir,
        output_root=tmp_path,
        resolved_path=protocol_dir / "resolved_protocol.yaml",
        assignments_path=protocol_dir / "development_assignments.parquet",
        manifest_path=protocol_dir / "protocol_manifest.json",
        config=config,
        assignments=_assignments(),
        protocol_manifest={"class_map_sha256": "1" * 64},
        preflight={"status": "valid"},
        parent_hashes={
            "protocol_manifest": {"artifact_role": "frozen_experimental_protocol", "file_sha256": "2" * 64},
            "development_assignments": {"artifact_role": "development_assignments", "content_sha256": "3" * 64},
            "resolved_protocol": {"artifact_role": "frozen_experimental_protocol", "file_sha256": "4" * 64},
        },
        code_file_hashes=code_files,
        code_hash=code_hash,
    )


def _zero_geo(_context: oof.FrozenM2Context, rows: pd.DataFrame) -> np.ndarray:
    return np.zeros((len(rows), len(oof.GEO_COLUMNS)), dtype=np.float32)


def _initialization(context: oof.FrozenM2Context) -> dict:
    frozen = context.config["experts"]["image_encoder"]["externally_pretrained_fixed"]
    return {
        "checkpoint_id": frozen["checkpoint_id"],
        "hub_revision": frozen["hub_revision"],
        "snapshot_path": "/synthetic/pinned-snapshot",
        "snapshot_file_sha256": {
            frozen["checkpoint_filename"]: frozen["checkpoint_sha256"],
            frozen["open_clip_config_filename"]: frozen["open_clip_config_sha256"],
            frozen["tokenizer_filename"]: frozen["tokenizer_sha256"],
            "tokenizer_config.json": frozen["tokenizer_config_sha256"],
            "special_tokens_map.json": frozen["special_tokens_map_sha256"],
        },
    }


def _logits(rows: int, seed: int, fold: int | None) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    fold_value = 4 if fold is None else fold
    for mode_index, mode in enumerate(oof.MODES):
        matrix = np.full((rows, oof.N_CLASSES), -4.0, dtype=np.float64)
        prediction = (seed + fold_value + mode_index) % oof.N_CLASSES
        matrix[:, prediction] = 4.0 + seed / 10.0
        result[mode] = matrix
    return result


def _materialize_producer(
    context: oof.FrozenM2Context,
    spec: oof.ProducerSpec,
    seed: int,
) -> Path:
    target = context.output_root / spec.relative_directory
    target.mkdir(parents=True)
    fit_rows, prediction_rows = oof._partitions_for_spec(context.assignments, spec)
    table = oof.build_output_table(
        prediction_rows,
        seed=seed,
        logits_by_mode=_logits(len(prediction_rows), seed, spec.fold),
        include_fold=spec.include_fold,
        schema_version=context.config["schema_version"],
        protocol_id=context.config["protocol_id"],
    )
    oof.write_output_parquet_atomic(table, target / spec.output_filename)
    for filename in (
        "adapted_visual_tower.safetensors",
        "image_only_head.safetensors",
        "geo_only_head.safetensors",
        "raw_concat_head.safetensors",
    ):
        oof._write_bytes_exclusive(target / filename, f"synthetic:{filename}".encode())
    scaler = oof.fit_geo_standardization(_zero_geo(context, fit_rows))
    oof._write_json_exclusive(target / "geo_standardization.json", scaler.to_json())
    initialization = _initialization(context)
    stage = {
        "schema_version": "geo_helpfulness.resolved_m2_stage.v1",
        "protocol_id": context.config["protocol_id"],
        "stage_id": spec.stage_id,
        "training_seed": seed,
        "train_oof_fold": spec.fold,
        "fitting_row_uids_sha256": oof.canonical_sha256(
            fit_rows["row_uid"].astype(str).tolist()
        ),
        "prediction_row_uids_sha256": oof.canonical_sha256(
            prediction_rows["row_uid"].astype(str).tolist()
        ),
        "geo_feature_columns": list(oof.GEO_COLUMNS),
        "mode_order": list(oof.MODES),
        "encoder_initialization": initialization,
        "adaptation_recipe": context.config["experts"]["image_encoder"][
            "fold_contained_adaptation"
        ]["adaptation_recipe"],
        "head_recipe": context.config["experts"]["head"],
        "expert_epochs": context.config["experts"]["epochs"],
        "m2_code_sha256": context.code_hash,
    }
    oof._write_yaml_exclusive(target / "resolved_stage_config.yaml", stage)
    reproduction = {
        "valid": True,
        "row_count": len(prediction_rows),
        "atol": oof.REPRODUCTION_ATOL,
        "rtol": oof.REPRODUCTION_RTOL,
    }
    metrics = {
        "schema_version": "geo_helpfulness.m2_training_metrics.v1",
        "protocol_id": context.config["protocol_id"],
        "stage_id": spec.stage_id,
        "training_seed": seed,
        "labels_scope": "fitting_partition_only",
        "heldout_metrics": None,
        "adaptation_history": [{"epoch": epoch + 1} for epoch in range(5)],
        "head_history": {
            mode: [{"epoch": epoch + 1} for epoch in range(50)]
            for mode in oof.MODES
        },
        "fitting_class_plot_support_by_dense_id": oof._class_plot_support(fit_rows),
        "checkpoint_reproduction": reproduction,
    }
    oof._write_json_exclusive(target / "training_metrics.json", metrics)
    manifest = oof._producer_manifest(
        context,
        spec,
        seed=seed,
        fit_rows=fit_rows,
        prediction_rows=prediction_rows,
        table=table,
        staging=target,
        encoder_provenance=initialization,
        runtime={"backend": "synthetic_cpu"},
        checkpoint_reproduction=reproduction,
    )
    oof._write_json_exclusive(target / "manifest.json", manifest)
    return target


def test_strict_twenty_producer_aggregation_report_resume_and_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    context = _context(tmp_path)
    monkeypatch.setattr(oof, "_geo_source_projection", _zero_geo)
    monkeypatch.setattr(oof, "_validate_checkpoint_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(oof, "load_frozen_context", lambda **_kwargs: context)

    producer_paths: list[tuple[oof.ProducerSpec, int, Path]] = []
    for seed in oof.TRAINING_SEEDS:
        for spec in oof._producer_specs(seed):
            producer_paths.append((spec, seed, _materialize_producer(context, spec, seed)))

    # Simulate a crash before the aggregate commit marker.  Exact uncommitted
    # payload names are recoverable and must not brick the next invocation.
    oof_root = tmp_path / "development_train_oof"
    partial_output = oof_root / "development_train_oof_model_outputs.parquet"
    partial_report = oof_root / "oof_reproduction_report.json"
    partial_output.write_bytes(b"uncommitted")
    partial_report.write_bytes(b"uncommitted")

    result = oof.aggregate(output_root=tmp_path)
    assert result["development_train_oof"]["status"] == "created"
    assert result["development_train_oof"]["row_count"] == 13_512
    assert result["development_validation"]["row_count"] == 3_288

    oof_table = oof.read_output_parquet(
        oof_root / "development_train_oof_model_outputs.parquet", include_fold=True
    )
    validation_table = oof.read_output_parquet(
        tmp_path
        / "development_validation/development_validation_model_outputs.parquet",
        include_fold=False,
    )
    assert len(oof_table) == 13_512
    assert len(validation_table) == 3_288
    assert "train_oof_fold" not in validation_table.column_names
    assert not any("label" in column for column in validation_table.column_names)
    assert set(oof_table["training_seed"].to_pylist()) == {1, 2, 3, 4}
    first_uid = oof_table["row_uid"][0].as_py()
    same_row = oof_table.filter(pa.compute.equal(oof_table["row_uid"], first_uid))
    assert len(same_row) == 4
    assert len({tuple(value) for value in same_row["image_logits"].to_pylist()}) == 4

    report = json.loads((oof_root / "oof_reproduction_report.json").read_text())
    assert report["row_count"] == 13_512
    assert report["unique_row_count"] == 3_378
    assert report["training_seeds"] == [1, 2, 3, 4]
    assert "development_validation" not in json.dumps(report)

    reused = oof.aggregate(output_root=tmp_path)
    assert reused["development_train_oof"]["status"] == "reused_valid"
    assert reused["development_validation"]["status"] == "reused_valid"

    # A published component mutation is never silently overwritten or resumed.
    spec, seed, target = producer_paths[0]
    checkpoint = target / "image_only_head.safetensors"
    os.chmod(checkpoint, 0o644)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(oof.M2ArtifactError, match="fingerprint"):
        oof.validate_producer(context, spec, seed=seed, directory=target)


def test_preflight_failure_creates_no_m2_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail_preflight(**_kwargs):
        raise oof.M2ArtifactError("synthetic preflight failure")

    monkeypatch.setattr(oof, "validate_m1_preflight", fail_preflight)
    output_root = tmp_path / "must_not_exist"
    with pytest.raises(oof.M2ArtifactError, match="preflight"):
        oof.load_frozen_context(output_root=output_root)
    assert not output_root.exists()


def test_run_seed_orchestrates_four_folds_then_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    context = _context(tmp_path)
    observed: list[tuple[str, int | None, int]] = []

    monkeypatch.setattr(oof, "load_frozen_context", lambda **_kwargs: context)
    monkeypatch.setattr(oof.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(oof.torch.cuda, "current_device", lambda: 0)

    def fake_stage(_context, spec, *, seed, device):
        assert device == torch.device("cuda", 0)
        observed.append((spec.stage_id, spec.fold, seed))
        return {"stage_id": spec.stage_id, "status": "created"}

    monkeypatch.setattr(oof, "_run_or_resume_producer", fake_stage)
    result = oof.run_seed(3, output_root=tmp_path)

    assert result["producer_count"] == 5
    assert observed == [
        ("train_oof_fold_0", 0, 3),
        ("train_oof_fold_1", 1, 3),
        ("train_oof_fold_2", 2, 3),
        ("train_oof_fold_3", 3, 3),
        ("development_train_to_validation", None, 3),
    ]


def test_dummy_producer_keeps_prediction_features_closed_until_all_heads_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base = _context(tmp_path)
    small_assignments = base.assignments.iloc[:8].copy().reset_index(drop=True)
    context = replace(base, assignments=small_assignments)
    spec = oof._producer_specs(1)[0]
    fit_rows, prediction_rows = oof._partitions_for_spec(small_assignments, spec)
    fit_uids = set(fit_rows["row_uid"])
    prediction_uids = set(prediction_rows["row_uid"])
    events: list[str] = []

    class DummyModel:
        visual = torch.nn.Identity()

    def fake_geo(_context, rows):
        uids = set(rows["row_uid"])
        if uids == fit_uids:
            events.append("geo_fit")
        elif uids == prediction_uids:
            events.append("geo_prediction")
        else:  # pragma: no cover - defensive diagnostic
            raise AssertionError("unexpected identity partition")
        return np.zeros((len(rows), len(oof.GEO_COLUMNS)), dtype=np.float32)

    def fake_load(_config, _device):
        events.append("load_fresh_encoder")
        return DummyModel(), "train_transform", "prediction_transform", object(), _initialization(context)

    def fake_adapt(_model, rows, *, transform, **_kwargs):
        assert set(rows["row_uid"]) == fit_uids
        assert transform == "train_transform"
        events.append("adapt")
        return [{"epoch": epoch + 1} for epoch in range(5)]

    def fake_extract(_model, rows, *, transform, **_kwargs):
        assert transform == "prediction_transform"
        identity = "fit" if set(rows["row_uid"]) == fit_uids else "prediction"
        events.append(f"extract_{identity}")
        return np.zeros((len(rows), 1_152), dtype=np.float32)

    def fake_fit_head(mode, features, labels, **_kwargs):
        assert len(labels) == len(fit_rows)
        assert not prediction_uids.intersection(fit_uids)
        events.append(f"head_{mode}")
        return object(), [{"epoch": epoch + 1} for epoch in range(50)]

    def fake_predict(_model, features, _device):
        return np.zeros((len(features), oof.N_CLASSES), dtype=np.float64)

    def fake_save(_module, path):
        path.write_bytes(b"synthetic checkpoint")
        return path

    def fake_replay(*_args, **_kwargs):
        events.append("checkpoint_replay")
        return {
            "valid": True,
            "row_count": len(prediction_rows),
            "atol": oof.REPRODUCTION_ATOL,
            "rtol": oof.REPRODUCTION_RTOL,
        }

    monkeypatch.setattr(oof, "_geo_source_projection", fake_geo)
    monkeypatch.setattr(oof, "_load_fresh_encoder", fake_load)
    monkeypatch.setattr(oof, "_frozen_text_weights", lambda *_args, **_kwargs: torch.zeros(1))
    monkeypatch.setattr(oof, "_adapt_encoder", fake_adapt)
    monkeypatch.setattr(oof, "_extract_embeddings", fake_extract)
    monkeypatch.setattr(oof, "_fit_head", fake_fit_head)
    monkeypatch.setattr(oof, "_predict_head", fake_predict)
    monkeypatch.setattr(oof, "_save_safetensors", fake_save)
    monkeypatch.setattr(oof, "_replay_saved_checkpoints", fake_replay)
    monkeypatch.setattr(oof, "_runtime_provenance", lambda _device: {"backend": "dummy"})
    monkeypatch.setattr(
        oof,
        "validate_producer",
        lambda *_args, **_kwargs: {"valid": True, "status": "reusable"},
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    oof._build_producer(
        context,
        spec,
        seed=1,
        staging=staging,
        device=torch.device("cpu"),
    )

    head_events = [f"head_{mode}" for mode in oof.MODES]
    assert events.index("geo_fit") < events.index("adapt")
    assert events.index("extract_fit") < events.index(head_events[0])
    assert [event for event in events if event.startswith("head_")] == head_events
    assert events.index(head_events[-1]) < events.index("geo_prediction")
    assert events.index("geo_prediction") < events.index("extract_prediction")
    assert events[-1] == "checkpoint_replay"


def test_fixed_eighteen_way_head_is_seed_reproducible_with_missing_classes():
    protocol_dir = (
        oof.PROJECT_ROOT
        / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/protocol"
    )
    config = yaml.safe_load((protocol_dir / "resolved_protocol.yaml").read_text())
    features = np.arange(4 * 64, dtype=np.float32).reshape(4, 64) / 100.0
    labels = np.asarray([4, 4, 7, 7], dtype=np.int64)

    first, _ = oof._fit_head(
        "geo_only", features, labels, config=config, seed=2, device=torch.device("cpu")
    )
    second, _ = oof._fit_head(
        "geo_only", features, labels, config=config, seed=2, device=torch.device("cpu")
    )

    assert first.net[-1].out_features == oof.N_CLASSES
    assert set(labels) != set(range(oof.N_CLASSES))
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(left, right)

"""Synthetic-only tests for phase-separated pre-label input authentication."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_router_validation_inputs as inputs
from multimodal import geo_helpfulness_router_candidate_state as portable
from multimodal import geo_helpfulness_router_numeric as numeric


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return inputs._file_hash(path)


def seal(value, key="manifest_sha256"):
    value[key] = inputs.canonical_sha256({k: v for k, v in value.items() if k != key})
    return value


def parquet(frame, path):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema = pa.schema([pa.field(f.name, f.type, nullable=False) for f in table.schema])
    table = table.cast(schema)
    pq.write_table(table, path)
    return table.cast(pq.read_schema(path))


@pytest.fixture
def training_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "TRAIN_ROWS", 4)
    monkeypatch.setattr(inputs, "TRAIN_PLOTS", 4)
    monkeypatch.setattr(inputs, "VALIDATION_ROWS", 3)
    monkeypatch.setattr(inputs, "VALIDATION_PLOTS", 3)
    repo, root = tmp_path / "repo", tmp_path / "artifacts"
    router = root / "router"
    router.mkdir(parents=True)
    code_maps = {}
    for family, names in (
        ("m2", inputs.M2_CODE),
        ("m3", inputs.M3_CODE),
        ("m4", inputs.M4_CODE),
        ("candidate", inputs.CANDIDATE_CODE),
    ):
        code_maps[family] = {}
        for name in names:
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
            code_maps[family][name] = inputs._file_hash(path)
    config = repo / "configs/multimodal_geo_helpfulness.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("frozen synthetic configuration", encoding="utf-8")
    protocol = root / "protocol"
    protocol.mkdir()
    parents = {"class_map": {"content_sha256": "a" * 64}}
    for name, key in (
        ("development_assignments.parquet", "development_assignments"),
        ("protocol_manifest.json", "protocol_manifest"),
        ("resolved_protocol.yaml", "resolved_protocol"),
    ):
        path = protocol / name
        # These files are deliberately not parseable; only opaque hashing is allowed.
        path.write_text("not a table or configuration", encoding="utf-8")
        parents[key] = {"file_sha256": inputs._file_hash(path)}
    parents["development_assignments"]["content_sha256"] = "b" * 64
    parents["protocol_manifest"]["payload_sha256"] = "c" * 64
    m3_manifest = seal(
        {
            **{f"{f}_code_file_sha256": code_maps[f] for f in ("m2", "m3")},
            **{
                f"{f}_code_sha256": inputs.canonical_sha256(code_maps[f])
                for f in ("m2", "m3")
            },
        }
    )
    m3_path = router / "targets_and_feature_contract/manifest.json"
    m3_file_hash = write_json(m3_path, m3_manifest)
    probabilities = np.full((4, 18), 1 / 18, dtype=np.float64)
    semantic = inputs.m3.build_router_feature_frame(
        probabilities,
        probabilities,
        probabilities,
        probability_basis=inputs.m3.CALIBRATED_PROBABILITY_BASIS,
    )
    targets, audits, transforms, temperatures, row_hashes = [], [], [], [], {}
    for seed in inputs.SEEDS:
        target = pd.DataFrame(
            {
                "schema_version": [inputs.m3.TARGET_SCHEMA_VERSION] * 4,
                "protocol_id": ["protocol_v1"] * 4,
                "row_uid": [f"{i:064x}" for i in range(4)],
                "plot_idx": [f"T{i}" for i in range(4)],
                "training_seed": np.full(4, seed, dtype=np.int8),
                "target_state": list(inputs.m3.TARGET_ORDER),
            }
        )
        row_hash = inputs.canonical_sha256(
            target[["row_uid", "training_seed"]].to_dict(orient="records")
        )
        row_hashes[str(seed)] = {
            "row_count": 4,
            "plot_count": 4,
            "row_identity_sha256": row_hash,
        }
        transform = numeric.fit_router_feature_transform(
            semantic,
            seed=seed,
            frozen_schema=inputs.m3.build_router_feature_schema(),
            fit_row_identity_sha256=row_hash,
        )
        transforms.append(transform)
        for mode in inputs.MODES:
            temperatures.append(
                {
                    "schema_version": numeric.TEMPERATURE_SCHEMA_VERSION,
                    "seed": seed,
                    "mode": mode,
                    "fit_role": "development_train_oof",
                    "fit_row_count": 4,
                    "fit_row_identity_sha256": row_hash,
                    "class_count": 18,
                    "weighting": "equal_per_image",
                    "calibration_spec": numeric.CALIBRATION_SPEC,
                    "safeguards": numeric.SAFEGUARDS,
                    "theta": 0.0,
                    "temperature": 1.0,
                    "native_nll": 3.0,
                    "calibrated_nll": 3.0,
                    "optimizer": {
                        "success": True,
                        "status": 0,
                        "message": "synthetic",
                        "nfev": 2,
                        "nit": None,
                    },
                }
            )
        audit = target[inputs.IDENTITY_COLUMNS].copy()
        audit["train_oof_fold"] = np.arange(4, dtype=np.int8)
        for prefix in ("image", "geo", "raw"):
            audit[f"{prefix}_prob_calibrated"] = list(probabilities)
        audits.append(pd.concat([audit, semantic], axis=1))
        targets.append(target)
    target = (
        pd.concat(targets)
        .sort_values(["row_uid", "training_seed"])
        .reset_index(drop=True)
    )
    audit = (
        pd.concat(audits)
        .sort_values(["row_uid", "training_seed"])
        .reset_index(drop=True)
    )
    matrix = pd.DataFrame(np.zeros((16, 727)), columns=portable.full_feature_columns())
    dataset_table = parquet(
        pd.concat([target, matrix], axis=1), router / "router_dataset.parquet"
    )
    audit_table = parquet(audit, router / "router_dataset_audit.parquet")
    containers = {}
    for filename, schema, states in (
        ("expert_temperatures.json", "expert_temperatures", temperatures),
        ("router_feature_transform.json", "router_feature_transform", transforms),
    ):
        containers[filename] = {
            "schema_version": f"geo_helpfulness.{schema}.v1",
            "protocol_id": "protocol_v1",
            "fit_role": "development_train_oof",
            "states": states,
        }
        write_json(router / filename, containers[filename])
    artifacts = {
        name: {
            "file_sha256": inputs._file_hash(router / name),
            "content_sha256": (
                inputs.canonical_sha256(containers[name])
                if name in containers
                else "d" * 64
            ),
        }
        for name in inputs.DATASET_CHILDREN
    }
    manifest = seal(
        {
            "schema_version": "geo_helpfulness.router_dataset_manifest.v1",
            "protocol_id": "protocol_v1",
            "artifact_role": "router_training_dataset",
            "feature_columns": portable.full_feature_columns(),
            "artifacts": artifacts,
            "schemas": {
                "router_dataset.parquet": inputs._schema_record(dataset_table.schema),
                "router_dataset_audit.parquet": inputs._schema_record(
                    audit_table.schema
                ),
            },
            "parent_lineage": {
                "config_file_sha256": inputs._file_hash(config),
                "m1_m2_parent_artifact_hashes": parents,
                "m3_bundle_file_sha256": {"manifest.json": m3_file_hash},
                "m3_manifest_sha256": m3_manifest["manifest_sha256"],
                "m4_code_file_sha256": code_maps["m4"],
                **{
                    f"{f}_code_sha256": inputs.canonical_sha256(code_maps[f])
                    for f in ("m2", "m3", "m4")
                },
            },
        }
    )
    manifest_hash = write_json(router / "router_dataset_manifest.json", manifest)
    candidate_root = router / "router_candidates_v2"
    candidate_root.mkdir()
    study_config = {
        "schema_version": "geo_helpfulness.router_candidate_study_config.v2"
    }
    study_config_hash = write_json(
        repo / "configs/multimodal_geo_router_candidates_v2.json", study_config
    )
    study = seal(
        {
            "schema_version": "geo_helpfulness.router_candidate_study.v2",
            "protocol_id": "protocol_v1",
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "seed_inputs": row_hashes,
            "configuration": study_config,
            "source_study_config_sha256": study_config_hash,
            "parent_config_sha256": inputs._file_hash(config),
            "implementation_file_sha256": code_maps["candidate"],
            "dataset_artifact_file_sha256": {
                **{n: r["file_sha256"] for n, r in artifacts.items()},
                "router_dataset_manifest.json": manifest_hash,
            },
            "model_parameters": portable.FROZEN_FIT_PARAMETERS,
        },
        "study_spec_sha256",
    )
    study_file_hash = write_json(candidate_root / "study_spec.json", study)
    report_hash = write_json(
        candidate_root / "training_report.json", {"diagnostic_role": "training_only"}
    )
    candidate_records = {}
    for variant in portable.VARIANTS:
        indices = portable.variant_feature_indices(variant)
        for seed in inputs.SEEDS:
            for C in portable.C_GRID:
                name = portable.candidate_filename(variant, seed, C)
                state = portable.seal_candidate_state(
                    {
                        "schema_version": portable.SCHEMA_VERSION,
                        "protocol_id": "protocol_v1",
                        "training_seed": seed,
                        "variant_id": variant,
                        "C": C,
                        "C_hex": C.hex(),
                        "target_order": list(portable.TARGET_ORDER),
                        "classes": [0, 1, 2, 3],
                        "fit_role": "development_train_oof",
                        "fit_row_count": 4,
                        "fit_row_identity_sha256": row_hashes[str(seed)][
                            "row_identity_sha256"
                        ],
                        "study_spec_sha256": study["study_spec_sha256"],
                        "dataset_manifest_sha256": manifest["manifest_sha256"],
                        "feature_input_names": portable.full_feature_columns(),
                        "feature_indices": indices,
                        "feature_names": [
                            portable.full_feature_columns()[i] for i in indices
                        ],
                        "fit_parameters": dict(portable.FROZEN_FIT_PARAMETERS, C=C),
                        "implementation": portable.IMPLEMENTATION,
                        "coef": np.zeros((4, len(indices))).tolist(),
                        "intercept": [0.0] * 4,
                        "n_iter": [1],
                    }
                )
                digest = write_json(candidate_root / name, state)
                candidate_records[name] = {
                    "file_sha256": digest,
                    "state_sha256": state["state_sha256"],
                    "variant_id": variant,
                    "training_seed": seed,
                    "C_hex": C.hex(),
                }
    write_json(
        candidate_root / "manifest.json",
        seal(
            {
                "schema_version": "geo_helpfulness.router_candidate_manifest.v1",
                "protocol_id": "protocol_v1",
                "dataset_manifest_sha256": manifest["manifest_sha256"],
                "study_spec_sha256": study["study_spec_sha256"],
                "candidates": candidate_records,
                "artifacts": {
                    "study_spec.json": {"file_sha256": study_file_hash},
                    "training_report.json": {"file_sha256": report_hash},
                },
            }
        ),
    )
    return repo, root, manifest, code_maps


def validation_fixture(training_fixture):
    repo, root, dataset_manifest, code_maps = training_fixture
    directory = root / "development_validation"
    directory.mkdir()
    identity = pd.DataFrame(
        {
            "row_uid": [f"{100+i:064x}" for i in range(3)],
            "file": [f"images/{i}.jpg" for i in range(3)],
            "file_lower": [f"images/{i}.jpg" for i in range(3)],
            "plot_idx": [f"V{i}" for i in range(3)],
        }
    )
    lineage = dataset_manifest["parent_lineage"]
    parents = {
        name: deepcopy(lineage["m1_m2_parent_artifact_hashes"][name])
        for name in (
            "development_assignments",
            "protocol_manifest",
            "resolved_protocol",
        )
    }
    common = {
        "schema_version": "geo_helpfulness.artifact_manifest.v1",
        "protocol_id": "protocol_v1",
        "artifact_role": "development_validation_outputs",
        "parent_roles": ["development_assignments", "development_train_expert_fit"],
        "parent_artifact_hashes": parents,
        "m2_code_file_sha256": code_maps["m2"],
        "m2_code_sha256": lineage["m2_code_sha256"],
        "output_columns": list(inputs.m2.VALIDATION_OUTPUT_COLUMNS),
        "model_output_filename": "development_validation_model_outputs.parquet",
        "probability_basis": "native_t1_uncalibrated",
        "logits_authoritative": True,
    }
    refs, tables = {}, []
    for seed in inputs.SEEDS:
        producer_dir = directory / f"seed_{seed}"
        producer_dir.mkdir()
        table = inputs.m2.build_output_table(
            identity,
            seed=seed,
            include_fold=False,
            logits_by_mode={
                mode: np.full((3, 18), float(seed)) for mode in inputs.MODES
            },
        )
        path = producer_dir / common["model_output_filename"]
        pq.write_table(table, path)
        schema_hash = inputs.canonical_sha256(
            {
                "geo_columns": list(inputs.m2.GEO_COLUMNS),
                "image_embedding_dim": 1152,
                "raw_concat_order": ["image_embedding", "standardized_geo"],
                "class_count": 18,
                "output_columns": list(inputs.m2.VALIDATION_OUTPUT_COLUMNS),
                "arrow_schema": str(table.schema),
            }
        )
        producer = seal(
            {
                **common,
                "stage_id": "development_train_to_validation",
                "train_oof_fold": None,
                "training_seed": seed,
                "row_count": 3,
                "plot_count": 3,
                "fitting_row_count": 4,
                "zero_plot_overlap": True,
                "temperature": 1.0,
                "calibrated": False,
                "probability_integrity_atol": 1e-8,
                "class_map_hash": "a" * 64,
                "model_output_file_sha256": inputs._file_hash(path),
                "component_file_sha256": {path.name: inputs._file_hash(path)},
                "content_sha256": inputs.m2.logical_table_sha256(table),
                "output_arrow_schema": str(table.schema),
                "feature_schema_hash": schema_hash,
                "prediction_plot_hash": inputs.plot_set_fingerprint(identity.plot_idx),
                "fitting_plot_hash": inputs.plot_set_fingerprint(
                    [f"T{i}" for i in range(4)]
                ),
            }
        )
        write_json(producer_dir / "manifest.json", producer)
        refs[f"development_validation/seed_{seed}"] = producer["manifest_sha256"]
        tables.append(table)
    aggregate = pa.concat_tables(tables).sort_by(
        [("row_uid", "ascending"), ("training_seed", "ascending")]
    )
    path = directory / common["model_output_filename"]
    pq.write_table(aggregate, path)
    write_json(
        directory / "aggregate_manifest.json",
        seal(
            {
                **common,
                "training_seeds": list(inputs.SEEDS),
                "row_count": 12,
                "unique_key": ["row_uid", "training_seed"],
                "canonical_sort": ["row_uid", "training_seed"],
                "aggregation": "concatenation_without_averaging_or_pooling",
                "producer_manifest_sha256": refs,
                "model_output_file_sha256": inputs._file_hash(path),
                "content_sha256": inputs.m2.logical_table_sha256(aggregate),
                "feature_schema_hash": schema_hash,
            }
        ),
    )
    return directory


def test_training_loading_never_reads_validation_outputs_or_assignment_columns(
    training_fixture, monkeypatch
):
    repo, root, _, _ = training_fixture
    original = inputs.pq.read_table
    reads = []

    def guarded(path, *args, **kwargs):
        reads.append((str(path), kwargs.get("columns")))
        assert "development_validation" not in str(path)
        assert "development_assignments" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(inputs.pq, "read_table", guarded)
    monkeypatch.setattr(
        inputs.m2, "load_frozen_context", lambda *a, **k: pytest.fail("full preflight")
    )
    monkeypatch.setattr(
        inputs.m2,
        "validate_producer",
        lambda *a, **k: pytest.fail("full producer validator"),
    )
    loaded = inputs.load_training_inputs(repo, root)
    assert len(loaded["candidate_states"]) == 48
    assert len(loaded["temperatures"]) == 12 and len(loaded["transforms"]) == 4
    assert loaded["targets"].columns.tolist() == inputs.TARGET_COLUMNS
    assert len(loaded["audit"]) == 16
    assert any(cols == inputs.TARGET_COLUMNS for _, cols in reads)
    assert not any(
        "development_validation" in path for path in loaded["parent_file_hashes"]
    )


@pytest.mark.parametrize(
    "relative",
    [
        "router/expert_temperatures.json",
        "router/router_feature_transform.json",
        "router/router_dataset.parquet",
        "router/router_dataset_audit.parquet",
        "router/router_dataset_manifest.json",
        "router/router_candidates_v2/study_spec.json",
        "router/router_candidates_v2/primary_full727__seed_1__C_0p01.json",
        "protocol/development_assignments.parquet",
    ],
)
def test_training_parent_tampering_rejected(training_fixture, relative):
    repo, root, _, _ = training_fixture
    path = root / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        inputs.load_training_inputs(repo, root)


@pytest.mark.parametrize(
    "relative",
    [
        "configs/multimodal_geo_helpfulness.yaml",
        "configs/multimodal_geo_router_candidates_v2.json",
        "multimodal/geo_helpfulness_router_numeric.py",
        "multimodal/geo_helpfulness_targets_features.py",
        "multimodal/geo_helpfulness_router_candidate_fit.py",
        "multimodal/geo_helpfulness_oof.py",
    ],
)
def test_frozen_config_and_code_tampering_rejected(training_fixture, relative):
    repo, root, _, _ = training_fixture
    path = repo / relative
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        inputs.load_training_inputs(repo, root)


def test_validation_reads_only_sealed_prediction_tables_and_updates_snapshot(
    training_fixture, monkeypatch
):
    repo, root, _, _ = training_fixture
    validation_fixture(training_fixture)
    loaded = inputs.load_training_inputs(repo, root)
    original = inputs.m2.read_output_parquet
    reads = []

    def guarded(path, *, include_fold):
        assert not include_fold
        assert str(root / "development_validation") in str(path)
        reads.append(str(path))
        return original(path, include_fold=include_fold)

    monkeypatch.setattr(inputs.m2, "read_output_parquet", guarded)
    observed = inputs.load_validation_inputs(repo, root, loaded)
    assert observed.shape == (12, 16)
    assert len(reads) == 5
    new_paths = [
        p for p in loaded["parent_file_hashes"] if "development_validation" in p
    ]
    assert len(new_paths) == 10
    assert not any("checkpoint" in p or "training_metrics" in p for p in new_paths)


@pytest.mark.parametrize(
    "relative",
    [
        "aggregate_manifest.json",
        "development_validation_model_outputs.parquet",
        "seed_1/manifest.json",
        "seed_2/development_validation_model_outputs.parquet",
    ],
)
def test_validation_file_tampering_rejected_and_snapshot_not_adopted(
    training_fixture, relative
):
    repo, root, _, _ = training_fixture
    directory = validation_fixture(training_fixture)
    loaded = inputs.load_training_inputs(repo, root)
    original = dict(loaded["parent_file_hashes"])
    path = directory / relative
    path.write_bytes(path.read_bytes() + b"invalid")
    with pytest.raises(ValueError):
        inputs.load_validation_inputs(repo, root, loaded)
    assert loaded["parent_file_hashes"] == original


@pytest.mark.parametrize(
    "change", ["stage", "counts", "parent", "plots", "schema_hash", "seed"]
)
def test_resealed_invalid_validation_declarations_rejected(training_fixture, change):
    repo, root, _, _ = training_fixture
    directory = validation_fixture(training_fixture)
    loaded = inputs.load_training_inputs(repo, root)
    producer = inputs._json(directory / "seed_1/manifest.json")
    if change == "stage":
        producer["stage_id"] = "final_development_fit"
    elif change == "counts":
        producer["plot_count"] = 2
    elif change == "parent":
        producer["parent_artifact_hashes"]["development_assignments"][
            "content_sha256"
        ] = ("e" * 64)
    elif change == "plots":
        producer["fitting_plot_hash"] = "e" * 64
    elif change == "schema_hash":
        producer["feature_schema_hash"] = "e" * 64
    else:
        producer["training_seed"] = 2
    seal(producer)
    write_json(directory / "seed_1/manifest.json", producer)
    aggregate = inputs._json(directory / "aggregate_manifest.json")
    aggregate["producer_manifest_sha256"]["development_validation/seed_1"] = producer[
        "manifest_sha256"
    ]
    write_json(directory / "aggregate_manifest.json", seal(aggregate))
    with pytest.raises(ValueError):
        inputs.load_validation_inputs(repo, root, loaded)


def test_parent_mutation_between_phases_rejected_before_validation_read(
    training_fixture, monkeypatch
):
    repo, root, _, _ = training_fixture
    loaded = inputs.load_training_inputs(repo, root)
    (repo / "configs/multimodal_geo_helpfulness.yaml").write_text("changed")
    monkeypatch.setattr(
        inputs.m2,
        "read_output_parquet",
        lambda *a, **k: pytest.fail("validation opened"),
    )
    with pytest.raises(ValueError, match="parent changed"):
        inputs.load_validation_inputs(repo, root, loaded)


def test_json_duplicates_nonfinite_and_symlinks_rejected(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text('{"x": 1, "x": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        inputs._json(path)
    path.write_text('{"x": NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        inputs._json(path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="nonsymlink"):
        inputs._json(link)

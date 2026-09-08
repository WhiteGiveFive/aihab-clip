"""M4 immutable, seed-specific router dataset preparation (no router training)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Initialize the existing Torch/Arrow reader before the SciPy numerical layer.
# In the frozen Python 3.9 environment the reverse native-library load order
# can segfault during large canonical-JSON hashes in the unchanged M2 reader.
from multimodal import geo_helpfulness_oof as m2
from multimodal import geo_helpfulness_targets_features as m3
from multimodal.geo_helpfulness_protocol import (
    assignment_fingerprint,
    canonical_sha256,
    sha256_file,
)
from multimodal.geo_helpfulness_router_numeric import (
    CALIBRATION_SPEC,
    NUMERIC_INPUT_COLUMNS,
    apply_expert_temperature,
    fit_expert_temperature,
    fit_router_feature_transform,
    multiclass_nll,
    output_feature_columns,
    transform_router_features,
    validate_feature_transform_state,
    validate_temperature_state,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = m3.DEFAULT_CONFIG_PATH
TRAINING_SEEDS = (1, 2, 3, 4)
MODE_TO_PREFIX = {"image_only": "image", "geo_only": "geo", "raw_concat": "raw"}
DATASET_FILENAME = "router_dataset.parquet"
AUDIT_FILENAME = "router_dataset_audit.parquet"
TEMPERATURE_FILENAME = "expert_temperatures.json"
TRANSFORM_FILENAME = "router_feature_transform.json"
MANIFEST_FILENAME = "router_dataset_manifest.json"
CHILD_FILENAMES = (
    DATASET_FILENAME,
    AUDIT_FILENAME,
    TEMPERATURE_FILENAME,
    TRANSFORM_FILENAME,
)
BUNDLE_FILENAMES = (*CHILD_FILENAMES, MANIFEST_FILENAME)
RECEIPT_FILENAME = ".router_dataset.m4.ownership.json"
LOCK_FILENAME = ".router_dataset.m4.lock"
MANIFEST_SCHEMA_VERSION = "geo_helpfulness.router_dataset_manifest.v1"
REPRODUCTION_ATOL = 1.0e-12
REPRODUCTION_RTOL = 1.0e-12
IDENTITY_COLUMNS = tuple(name for name in m3.TARGET_COLUMNS if name != "target_state")
CODE_PATHS = (
    "multimodal/geo_helpfulness_router_dataset.py",
    "multimodal/geo_helpfulness_router_numeric.py",
    "tools/run_multimodal_geo_helpfulness_m4_dataset.py",
)


class M4ArtifactError(ValueError):
    """An immutable input/output or dataset contract is invalid."""


@dataclass(frozen=True)
class PreparedInputs:
    root: Path
    protocol_id: str
    calibration_spec: dict[str, Any]
    training: pd.DataFrame
    targets: pd.DataFrame
    feature_schema: dict[str, Any]
    lineage: dict[str, Any]
    parent_files: dict[str, str]


@dataclass(frozen=True)
class ValidatedRouterDatasetBundle:
    root: Path
    dataset: pa.Table
    audit: pa.Table
    temperatures: dict[str, Any]
    transforms: dict[str, Any]
    manifest: dict[str, Any]
    validation: dict[str, Any]


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(value)
    return (
        candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    ).resolve()


def _json_mapping(path: Path) -> dict[str, Any]:
    _regular_file(path)
    try:

        def reject_constant(value: str) -> None:
            raise ValueError(f"nonfinite JSON constant: {value}")

        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
        if not isinstance(payload, dict):
            raise ValueError("expected JSON object")
        return payload
    except (OSError, ValueError) as exc:
        raise M4ArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise M4ArtifactError(f"artifact must be a regular non-symlink file: {path}")


def _self_hash(payload: Mapping[str, Any], field: str = "manifest_sha256") -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_parent_snapshot(prepared: PreparedInputs) -> None:
    for raw_path, expected in prepared.parent_files.items():
        path = Path(raw_path)
        _regular_file(path)
        if sha256_file(path) != expected:
            raise M4ArtifactError(
                f"parent or implementation changed during M4 operation: {path}"
            )


def assemble_training_inputs(
    oof: pd.DataFrame,
    assignments: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    protocol_id: str,
    expected_rows: int = 3378,
    expected_plots: int = 1300,
) -> pd.DataFrame:
    """Join only the train projection; enforce complete seed-specific identities."""
    required = [
        "protocol_id",
        "row_uid",
        "file",
        "file_lower",
        "plot_idx",
        "train_oof_fold",
    ]
    assignment_columns = required + ["development_role", "label_id_dense"]
    if not set(assignment_columns).issubset(assignments):
        raise M4ArtifactError(
            "assignments lack required training identity/label columns"
        )
    if not set(required + ["training_seed"]).issubset(oof):
        raise M4ArtifactError("OOF outputs lack required identity columns")
    if {"label_id_dense", "development_role", "target_state"}.intersection(oof.columns):
        raise M4ArtifactError(
            "OOF inputs must be label-blind before the assignment join"
        )
    train = assignments.loc[
        assignments["development_role"].eq("train"), assignment_columns
    ].copy()
    if train.isna().any().any() or train["row_uid"].duplicated().any():
        raise M4ArtifactError(
            "training assignments contain missing or duplicate identities"
        )
    if len(train) != expected_rows or train["plot_idx"].nunique() != expected_plots:
        raise M4ArtifactError(
            "training assignment row/plot counts differ from contract"
        )
    labels = train["label_id_dense"].to_numpy()
    if not np.issubdtype(labels.dtype, np.integer) or np.any(
        (labels < 0) | (labels >= 18)
    ):
        raise M4ArtifactError("training labels must be integer dense IDs in 0..17")
    if set(oof["training_seed"].tolist()) != set(TRAINING_SEEDS):
        raise M4ArtifactError("OOF training seeds must be exactly 1,2,3,4")
    if oof[list(m3.TARGET_KEY)].duplicated().any():
        raise M4ArtifactError("duplicate OOF (row_uid, training_seed) key")
    if len(oof) != expected_rows * len(TRAINING_SEEDS):
        raise M4ArtifactError("OOF row count differs from four complete seeds")
    if (
        not train["protocol_id"].eq(protocol_id).all()
        or not oof["protocol_id"].eq(protocol_id).all()
    ):
        raise M4ArtifactError("protocol identity mismatch")
    for seed in TRAINING_SEEDS:
        subset = oof.loc[oof["training_seed"].eq(seed)]
        if set(subset["row_uid"]) != set(train["row_uid"]):
            raise M4ArtifactError(
                f"seed {seed} does not cover every training image exactly once"
            )
        if set(subset["train_oof_fold"]) != {0, 1, 2, 3}:
            raise M4ArtifactError(f"seed {seed} does not cover exactly four OOF folds")
    joined = oof.merge(
        train,
        on="row_uid",
        how="left",
        validate="many_to_one",
        suffixes=("", "__assignment"),
    )
    for name in required:
        if (
            name != "row_uid"
            and not joined[name].eq(joined[f"{name}__assignment"]).all()
        ):
            raise M4ArtifactError(f"OOF/assignment identity mismatch: {name}")
    if (
        joined["label_id_dense"].isna().any()
        or not joined["development_role"].eq("train").all()
    ):
        raise M4ArtifactError("non-training or missing label reached calibration join")
    expected_targets = m3.build_router_target_table(
        oof, train, protocol_id=protocol_id, expected_seeds=TRAINING_SEEDS
    )
    m3.validate_router_target_table(
        targets, protocol_id=protocol_id, expected_seeds=TRAINING_SEEDS
    )
    if m3.router_target_content_sha256(
        expected_targets
    ) != m3.router_target_content_sha256(targets):
        raise M4ArtifactError("OOF/label-derived targets differ from sealed M3 targets")
    keep = list(oof.columns) + ["label_id_dense"]
    return (
        joined.loc[:, keep]
        .sort_values(list(m3.TARGET_KEY), kind="stable")
        .reset_index(drop=True)
    )


def _prepare_inputs(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> PreparedInputs:
    # The public reader validates all frozen parents; do not expand its access
    # to validation expert outputs by calling M2.aggregate().
    bundle = m3.load_validated_m3_bundle(
        config_path=config_path, protocol_dir=protocol_dir, artifact_root=artifact_root
    )
    config_path = _resolve_path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sealed_dir = _resolve_path(
        protocol_dir if protocol_dir is not None else config["paths"]["protocol_root"]
    )
    root = bundle.root.parent
    artifact_base = root.parent
    assignment_path = sealed_dir / "development_assignments.parquet"
    resolved_path = sealed_dir / "resolved_protocol.yaml"
    aggregate_path = (
        artifact_base
        / "development_train_oof"
        / "development_train_oof_model_outputs.parquet"
    )
    aggregate_manifest = aggregate_path.parent / "aggregate_manifest.json"
    parents = bundle.manifest["parent_artifact_hashes"]
    required_hashes = {
        assignment_path: parents["development_assignments"]["file_sha256"],
        resolved_path: parents["resolved_protocol"]["file_sha256"],
        sealed_dir
        / "protocol_manifest.json": parents["protocol_manifest"]["file_sha256"],
        aggregate_path: parents["development_train_oof_aggregate"]["file_sha256"],
        aggregate_manifest: parents["development_train_oof_aggregate"][
            "manifest_file_sha256"
        ],
    }
    for relative, digest in parents["train_oof_producer_manifests"][
        "file_sha256"
    ].items():
        required_hashes[artifact_base / relative / "manifest.json"] = digest
    if len(parents["train_oof_producer_manifests"]["file_sha256"]) != 16:
        raise M4ArtifactError("expected exactly 16 validated OOF producer parents")
    m3_files = {}
    for name in m3.BUNDLE_FILENAMES:
        path = bundle.root / name
        _regular_file(path)
        digest = sha256_file(path)
        if name == m3.MANIFEST_FILENAME:
            if _json_mapping(path) != bundle.manifest:
                raise M4ArtifactError("M3 manifest changed after validation")
        elif digest != bundle.manifest["artifacts"][name]["file_sha256"]:
            raise M4ArtifactError(f"M3 child changed after validation: {name}")
        required_hashes[path] = digest
        m3_files[name] = digest
    for path, digest in required_hashes.items():
        _regular_file(path)
        if sha256_file(path) != digest:
            raise M4ArtifactError(f"validated parent fingerprint changed: {path}")
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    calibration_spec = resolved["calibration"]["expert_probability"]
    if calibration_spec != CALIBRATION_SPEC:
        raise M4ArtifactError(
            "frozen calibration configuration differs from M4 contract"
        )
    assignments = pd.read_parquet(assignment_path)
    if (
        assignment_fingerprint(assignments)
        != parents["development_assignments"]["content_sha256"]
    ):
        raise M4ArtifactError("assignment logical fingerprint changed")
    oof_table = m2.read_output_parquet(aggregate_path, include_fold=True)
    if (
        m2.logical_table_sha256(oof_table)
        != parents["development_train_oof_aggregate"]["content_sha256"]
    ):
        raise M4ArtifactError("OOF aggregate logical fingerprint changed")
    if dict(m2.MODE_TO_PREFIX) != MODE_TO_PREFIX:
        raise M4ArtifactError("expert mode/prefix contract changed")
    training = assemble_training_inputs(
        oof_table.to_pandas(),
        assignments,
        bundle.targets,
        protocol_id=str(resolved["protocol_id"]),
    )
    code_hashes = {name: sha256_file(PROJECT_ROOT / name) for name in CODE_PATHS}
    for name, digest in code_hashes.items():
        required_hashes[PROJECT_ROOT / name] = digest
    required_hashes[config_path] = sha256_file(config_path)
    for key in ("m2_code_file_sha256", "m3_code_file_sha256"):
        for name, digest in bundle.manifest[key].items():
            required_hashes[PROJECT_ROOT / name] = digest
    prepared = PreparedInputs(
        root=root,
        protocol_id=str(resolved["protocol_id"]),
        calibration_spec=calibration_spec,
        training=training,
        targets=bundle.targets,
        feature_schema=bundle.feature_schema,
        lineage={
            "m1_m2_parent_artifact_hashes": parents,
            "m3_manifest_sha256": bundle.manifest["manifest_sha256"],
            "m3_bundle_file_sha256": m3_files,
            "m2_code_sha256": bundle.manifest["m2_code_sha256"],
            "m3_code_sha256": bundle.manifest["m3_code_sha256"],
            "m4_code_file_sha256": code_hashes,
            "m4_code_sha256": canonical_sha256(code_hashes),
            "config_file_sha256": required_hashes[config_path],
        },
        parent_files={str(path): digest for path, digest in required_hashes.items()},
    )
    _assert_parent_snapshot(prepared)
    return prepared


def dataset_schema() -> pa.Schema:
    metadata = [
        pa.field(
            name, pa.int8() if name == "training_seed" else pa.string(), nullable=False
        )
        for name in m3.TARGET_COLUMNS
    ]
    return pa.schema(
        metadata
        + [
            pa.field(name, pa.float64(), nullable=False)
            for name in output_feature_columns()
        ]
    )


def audit_schema() -> pa.Schema:
    fields = [
        pa.field(
            name, pa.int8() if name == "training_seed" else pa.string(), nullable=False
        )
        for name in IDENTITY_COLUMNS
    ]
    fields.append(pa.field("train_oof_fold", pa.int8(), nullable=False))
    vector = pa.list_(pa.field("element", pa.float64(), nullable=False), 18)
    fields.extend(
        pa.field(f"{prefix}_prob_calibrated", vector, nullable=False)
        for prefix in MODE_TO_PREFIX.values()
    )
    dtypes = m3.build_router_feature_schema()["dtypes"]
    fields.extend(
        pa.field(name, pa.from_numpy_dtype(dtypes[name]), nullable=False)
        for name in m3.FEATURE_COLUMNS
    )
    return pa.schema(fields)


def _frame_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    if list(frame.columns) != schema.names:
        raise M4ArtifactError("materialized column order differs from declared schema")
    table = pa.Table.from_pandas(
        frame, schema=schema, preserve_index=False
    ).replace_schema_metadata(None)
    _validate_table_schema(table, schema)
    return table


def _validate_table_schema(table: pa.Table, schema: pa.Schema) -> None:
    if not table.schema.equals(schema, check_metadata=True):
        raise M4ArtifactError("Parquet Arrow schema or nullability mismatch")
    for field, column in zip(schema, table.columns):
        if column.null_count:
            raise M4ArtifactError(f"null values in {field.name}")
        if (
            pa.types.is_fixed_size_list(field.type)
            and column.combine_chunks().values.null_count
        ):
            raise M4ArtifactError(f"null vector entries in {field.name}")


def _fit_row_hash(rows: pd.DataFrame) -> str:
    return canonical_sha256(
        rows[["row_uid", "training_seed"]].to_dict(orient="records")
    )


def _state_container(
    kind: str, protocol_id: str, states: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": f"geo_helpfulness.{kind}.v1",
        "protocol_id": protocol_id,
        "fit_role": "development_train_oof",
        "states": states,
    }


def _check_close(actual: Any, expected: Any, description: str) -> None:
    left, right = np.asarray(actual, dtype=np.float64), np.asarray(
        expected, dtype=np.float64
    )
    if (
        left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.allclose(left, right, atol=REPRODUCTION_ATOL, rtol=REPRODUCTION_RTOL)
    ):
        raise M4ArtifactError(f"reconstruction mismatch: {description}")


def _derive_tables(
    prepared: PreparedInputs,
    *,
    temperatures: dict[str, Any] | None = None,
    transforms: dict[str, Any] | None = None,
) -> tuple[pa.Table, pa.Table, dict[str, Any], dict[str, Any]]:
    fitting = temperatures is None and transforms is None
    if (temperatures is None) != (transforms is None):
        raise M4ArtifactError(
            "temperature and transform states must be supplied together"
        )
    if not fitting:
        for kind, container, count in (
            ("expert_temperatures", temperatures, 12),
            ("router_feature_transform", transforms, 4),
        ):
            if not isinstance(container, dict) or not isinstance(
                container.get("states"), list
            ):
                raise M4ArtifactError(f"invalid saved {kind} container")
            if len(container["states"]) != count or container != _state_container(
                kind, prepared.protocol_id, container["states"]
            ):
                raise M4ArtifactError(f"saved {kind} scope/count/schema mismatch")
    temperature_states, transform_states, matrix_frames, audit_frames = [], [], [], []
    for seed_index, seed in enumerate(TRAINING_SEEDS):
        rows = (
            prepared.training.loc[prepared.training["training_seed"].eq(seed)]
            .sort_values("row_uid", kind="stable")
            .reset_index(drop=True)
        )
        labels = rows["label_id_dense"].to_numpy(dtype=np.int64)
        row_hash = _fit_row_hash(rows)
        probabilities = []
        for mode_index, (mode, prefix) in enumerate(MODE_TO_PREFIX.items()):
            logits = np.stack(rows[f"{prefix}_logits"].to_numpy()).astype(
                np.float64, copy=False
            )
            if fitting:
                state = fit_expert_temperature(
                    logits,
                    labels,
                    seed=seed,
                    mode=mode,
                    calibration_spec=prepared.calibration_spec,
                    fit_row_identity_sha256=row_hash,
                )
            else:
                state = temperatures["states"][seed_index * 3 + mode_index]
                validate_temperature_state(state)
                if (
                    state["seed"],
                    state["mode"],
                    state["fit_row_count"],
                    state["fit_row_identity_sha256"],
                ) != (seed, mode, len(rows), row_hash):
                    raise M4ArtifactError(
                        "temperature state seed/mode/fit-row mismatch"
                    )
                _check_close(
                    state["native_nll"], multiclass_nll(logits, labels), "native NLL"
                )
                _check_close(
                    state["calibrated_nll"],
                    multiclass_nll(logits, labels, temperature=state["temperature"]),
                    "calibrated NLL",
                )
            probability = apply_expert_temperature(logits, state)
            prediction = np.argmax(logits, axis=1)
            if not np.array_equal(
                prediction, rows[f"{prefix}_pred"].to_numpy()
            ) or not np.array_equal(np.argmax(probability, axis=1), prediction):
                raise M4ArtifactError(
                    f"calibrated/stored/logit prediction mismatch for seed {seed}/{mode}"
                )
            temperature_states.append(state)
            probabilities.append(probability)
        semantic = m3.build_router_feature_frame(
            *probabilities, probability_basis=m3.CALIBRATED_PROBABILITY_BASIS
        )
        if fitting:
            transform = fit_router_feature_transform(
                semantic,
                seed=seed,
                frozen_schema=prepared.feature_schema,
                fit_row_identity_sha256=row_hash,
            )
        else:
            transform = transforms["states"][seed_index]
            validate_feature_transform_state(transform)
            if (
                transform["seed"],
                transform["fit_row_count"],
                transform["fit_row_identity_sha256"],
                transform["feature_schema_sha256"],
            ) != (seed, len(rows), row_hash, canonical_sha256(prepared.feature_schema)):
                raise M4ArtifactError("feature transform seed/schema/fit-row mismatch")
            numeric = semantic.loc[:, list(NUMERIC_INPUT_COLUMNS)].to_numpy(
                dtype=np.float64
            )
            _check_close(transform["mean"], numeric.mean(axis=0), "transform mean")
            _check_close(
                transform["variance"], numeric.var(axis=0, ddof=0), "transform variance"
            )
        matrix = transform_router_features(semantic, transform)
        if (
            matrix.shape != (len(rows), 727)
            or matrix.dtype != np.float64
            or not np.isfinite(matrix).all()
        ):
            raise M4ArtifactError("router matrix shape/dtype/finiteness violation")
        transform_states.append(transform)
        target = (
            prepared.targets.loc[prepared.targets["training_seed"].eq(seed)]
            .sort_values("row_uid", kind="stable")
            .reset_index(drop=True)
        )
        if not np.array_equal(target["row_uid"].to_numpy(), rows["row_uid"].to_numpy()):
            raise M4ArtifactError("target/matrix row alignment mismatch")
        matrix_frames.append(
            pd.concat(
                [target, pd.DataFrame(matrix, columns=output_feature_columns())], axis=1
            )
        )
        audit = target.loc[:, list(IDENTITY_COLUMNS)].copy()
        audit["train_oof_fold"] = rows["train_oof_fold"].to_numpy(dtype=np.int8)
        for prefix, probability in zip(MODE_TO_PREFIX.values(), probabilities):
            audit[f"{prefix}_prob_calibrated"] = list(probability)
        audit_frames.append(pd.concat([audit, semantic], axis=1))

    def combine(frames: list[pd.DataFrame], schema: pa.Schema) -> pa.Table:
        frame = (
            pd.concat(frames, ignore_index=True)
            .sort_values(list(m3.TARGET_KEY), kind="stable")
            .reset_index(drop=True)
        )
        return _frame_table(frame, schema)

    return (
        combine(matrix_frames, dataset_schema()),
        combine(audit_frames, audit_schema()),
        _state_container(
            "expert_temperatures", prepared.protocol_id, temperature_states
        ),
        _state_container(
            "router_feature_transform", prepared.protocol_id, transform_states
        ),
    )


def _schema_record(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "dtype": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _logical_table_hash(table: pa.Table) -> str:
    """Canonical column hashes, independent of Parquet chunking/compression."""
    columns = []
    for field, chunked in zip(table.schema, table.columns):
        column = chunked.combine_chunks()
        if pa.types.is_string(field.type):
            digest = canonical_sha256(column.to_pylist())
        else:
            values = (
                column.values if pa.types.is_fixed_size_list(field.type) else column
            )
            array = values.to_numpy(zero_copy_only=False)
            little_endian = np.asarray(array, dtype=array.dtype.newbyteorder("<"))
            digest = hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()
        columns.append(digest)
    return canonical_sha256(
        {
            "schema": _schema_record(table.schema),
            "rows": len(table),
            "column_sha256": columns,
        }
    )


def _read_table(path: Path, schema: pa.Schema) -> pa.Table:
    _regular_file(path)
    table = pq.read_table(path)
    _validate_table_schema(table, schema)
    return table


def _artifact_records(root: Path) -> dict[str, Any]:
    records = {}
    for name in CHILD_FILENAMES:
        path = root / name
        _regular_file(path)
        if name in (DATASET_FILENAME, AUDIT_FILENAME):
            table = _read_table(
                path, dataset_schema() if name == DATASET_FILENAME else audit_schema()
            )
            content_hash = _logical_table_hash(table)
        else:
            content_hash = canonical_sha256(_json_mapping(path))
        records[name] = {
            "file_sha256": sha256_file(path),
            "content_sha256": content_hash,
        }
    return records


def _environment() -> dict[str, str]:
    return {
        "python": sys.version,
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "pyarrow")
        },
    }


def _build_manifest(prepared: PreparedInputs, root: Path) -> dict[str, Any]:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": prepared.protocol_id,
        "artifact_role": "router_training_dataset",
        "parent_lineage": prepared.lineage,
        "environment": _environment(),
        "artifacts": _artifact_records(root),
        "schemas": {
            DATASET_FILENAME: _schema_record(dataset_schema()),
            AUDIT_FILENAME: _schema_record(audit_schema()),
        },
        "rows": len(prepared.targets),
        "unique_images": int(prepared.targets["row_uid"].nunique()),
        "plots": int(prepared.targets["plot_idx"].nunique()),
        "training_seeds": list(TRAINING_SEEDS),
        "key_and_canonical_sort": list(m3.TARGET_KEY),
        "seed_aggregation": "none",
        "feature_columns": list(output_feature_columns()),
        "target_order": list(m3.TARGET_ORDER),
        "temperature_count": 12,
        "numeric_transform_count": 4,
        "calibration_weighting": "equal_image_weight",
        "probability_basis": m3.CALIBRATED_PROBABILITY_BASIS,
        "reconstruction_tolerance": {
            "atol": REPRODUCTION_ATOL,
            "rtol": REPRODUCTION_RTOL,
        },
        "probability_row_sum_atol": 1.0e-8,
        "source_access": {
            "parent_preflight_reads_full_development_assignments": True,
            "calibration_labels": "development_role=train_only",
            "validation_expert_outputs_opened": False,
            "locked_test_sources_opened": False,
            "router_fitted": False,
        },
        "validation": {
            "parent_integrity": True,
            "target_key_equality": True,
            "state_and_table_reconstruction": True,
        },
        "publication": {
            "manifest_is_commit_marker": True,
            "write_mode": "exclusive_create",
            "owned_file_allowlist": list(BUNDLE_FILENAMES),
        },
        "logical_table_hash": "sha256_canonical_schema_rows_and_per_column_little_endian_bytes_or_json_strings_v1",
    }
    payload["manifest_sha256"] = _self_hash(payload)
    return payload


def _compare_tables(actual: pa.Table, expected: pa.Table) -> None:
    if actual.num_rows != expected.num_rows or not actual.schema.equals(
        expected.schema, check_metadata=True
    ):
        raise M4ArtifactError("reconstructed table dimensions/schema mismatch")
    for field in actual.schema:
        a, b = (
            actual[field.name].combine_chunks(),
            expected[field.name].combine_chunks(),
        )
        if pa.types.is_floating(field.type):
            _check_close(a.to_numpy(), b.to_numpy(), field.name)
        elif pa.types.is_fixed_size_list(field.type):
            _check_close(a.values.to_numpy(), b.values.to_numpy(), field.name)
        elif not a.equals(b):
            raise M4ArtifactError(
                f"reconstructed identity/target/semantic column mismatch: {field.name}"
            )


def _validate_prepared(prepared: PreparedInputs) -> dict[str, Any]:
    root = prepared.root
    _check_root(root)
    for name in BUNDLE_FILENAMES:
        _regular_file(root / name)
    manifest = _json_mapping(root / MANIFEST_FILENAME)
    if manifest.get("manifest_sha256") != _self_hash(manifest):
        raise M4ArtifactError("M4 manifest self-hash mismatch")
    if manifest != _build_manifest(prepared, root):
        raise M4ArtifactError(
            "M4 manifest, parent lineage, environment, code, or artifact fingerprints are stale"
        )
    temperatures = _json_mapping(root / TEMPERATURE_FILENAME)
    transforms = _json_mapping(root / TRANSFORM_FILENAME)
    expected_dataset, expected_audit, _, _ = _derive_tables(
        prepared, temperatures=temperatures, transforms=transforms
    )
    _compare_tables(
        _read_table(root / DATASET_FILENAME, dataset_schema()), expected_dataset
    )
    _compare_tables(_read_table(root / AUDIT_FILENAME, audit_schema()), expected_audit)
    _assert_parent_snapshot(prepared)
    return {
        "valid": True,
        "status": "reused_valid",
        "protocol_id": prepared.protocol_id,
        "bundle_root": str(root),
        "manifest": str(root / MANIFEST_FILENAME),
        "manifest_sha256": manifest["manifest_sha256"],
        "row_count": len(prepared.targets),
        "unique_image_count": manifest["unique_images"],
        "plot_count": manifest["plots"],
        "feature_count": 727,
        "dataset_column_count": 733,
        "training_seeds": list(TRAINING_SEEDS),
        "temperature_count": 12,
        "numeric_transform_count": 4,
        "calibration_fitted_this_call": False,
        "router_dataset_materialized": True,
        "router_fitted": False,
    }


def _check_root(root: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise M4ArtifactError(f"M4 router root must be a non-symlink directory: {root}")


@contextmanager
def _workflow_lock(root: Path):
    import fcntl

    _check_root(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_FILENAME
    if path.is_symlink():
        raise M4ArtifactError("workflow lock must not be a symlink")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise M4ArtifactError(
                "another M4 process owns the dataset workflow lock"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _receipt_payload(
    prepared: PreparedInputs, hashes: dict[str, str], nonce: str
) -> dict[str, Any]:
    payload = {
        "schema_version": "geo_helpfulness.router_dataset_ownership.v1",
        "owner": "M4_router_dataset",
        "protocol_id": prepared.protocol_id,
        "directory": prepared.root.name,
        "nonce": nonce,
        "owned_uncommitted_files": hashes,
        "commit_marker": MANIFEST_FILENAME,
    }
    payload["receipt_sha256"] = _self_hash(payload, "receipt_sha256")
    return payload


def _validate_receipt(prepared: PreparedInputs) -> dict[str, Any]:
    receipt = _json_mapping(prepared.root / RECEIPT_FILENAME)
    hashes, nonce = receipt.get("owned_uncommitted_files"), receipt.get("nonce")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(CHILD_FILENAMES)
        or not all(
            isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x)
            for x in hashes.values()
        )
        or not isinstance(nonce, str)
        or not re.fullmatch(r"[0-9a-f]{32}", nonce)
        or receipt != _receipt_payload(prepared, hashes, nonce)
    ):
        raise M4ArtifactError("invalid M4 ownership receipt")
    return receipt


def _recover_uncommitted(prepared: PreparedInputs) -> list[str]:
    root = prepared.root
    present = [
        root / name
        for name in CHILD_FILENAMES
        if (root / name).exists() or (root / name).is_symlink()
    ]
    receipt_path = root / RECEIPT_FILENAME
    if not present and not receipt_path.exists() and not receipt_path.is_symlink():
        return []
    if not receipt_path.exists() and not receipt_path.is_symlink():
        raise M4ArtifactError(
            "uncommitted M4 files have no ownership receipt; refusing cleanup"
        )
    receipt = _validate_receipt(prepared)
    for path in present:
        _regular_file(path)
        if sha256_file(path) != receipt["owned_uncommitted_files"][path.name]:
            raise M4ArtifactError(
                f"uncommitted file is not the receipt-owned content; refusing cleanup: {path}"
            )
    removed = [path.name for path in present]
    for path in present:
        path.unlink()
    receipt_path.unlink()
    return removed


def _exclusive_publish(staged: Path, destination: Path) -> None:
    os.chmod(staged, 0o444)
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise M4ArtifactError(
            f"immutable M4 output already exists: {destination}"
        ) from exc
    staged.unlink()


def build_router_dataset_bundle(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build once, or strictly reuse committed outputs without fitting again."""
    prepared = _prepare_inputs(
        config_path=config_path, protocol_dir=protocol_dir, artifact_root=artifact_root
    )
    root = prepared.root
    with _workflow_lock(root):
        if (root / MANIFEST_FILENAME).exists() or (
            root / MANIFEST_FILENAME
        ).is_symlink():
            result = _validate_prepared(prepared)
            if (root / RECEIPT_FILENAME).exists() or (
                root / RECEIPT_FILENAME
            ).is_symlink():
                receipt = _validate_receipt(prepared)
                for name, digest in receipt["owned_uncommitted_files"].items():
                    if sha256_file(root / name) != digest:
                        raise M4ArtifactError("committed receipt does not match output")
                (root / RECEIPT_FILENAME).unlink()
            return result
        recovered = _recover_uncommitted(prepared)
        dataset, audit, temperatures, transforms = _derive_tables(prepared)
        staging = Path(tempfile.mkdtemp(prefix=".router_dataset.m4.staging-", dir=root))
        try:
            for name, table in ((DATASET_FILENAME, dataset), (AUDIT_FILENAME, audit)):
                pq.write_table(
                    table,
                    staging / name,
                    compression="zstd",
                    use_dictionary=False,
                    write_statistics=True,
                )
            _write_json(staging / TEMPERATURE_FILENAME, temperatures)
            _write_json(staging / TRANSFORM_FILENAME, transforms)
            manifest = _build_manifest(prepared, staging)
            _write_json(staging / MANIFEST_FILENAME, manifest)
            # Verify serialized state and tables before a commit marker exists.
            staged_prepared = PreparedInputs(**{**prepared.__dict__, "root": staging})
            _validate_prepared(staged_prepared)
            _assert_parent_snapshot(prepared)
            hashes = {name: sha256_file(staging / name) for name in CHILD_FILENAMES}
            _write_json(
                root / RECEIPT_FILENAME,
                _receipt_payload(prepared, hashes, os.urandom(16).hex()),
            )
            for name in CHILD_FILENAMES:
                _exclusive_publish(staging / name, root / name)
            _assert_parent_snapshot(prepared)
            _exclusive_publish(staging / MANIFEST_FILENAME, root / MANIFEST_FILENAME)
            (root / RECEIPT_FILENAME).unlink()
        finally:
            shutil.rmtree(staging)
        result = _validate_prepared(prepared)
        result.update(
            status="created",
            calibration_fitted_this_call=True,
            recovered_uncommitted_files=recovered,
        )
        return result


def validate_router_dataset_bundle(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only validation, including reconstruction using frozen states."""
    prepared = _prepare_inputs(
        config_path=config_path, protocol_dir=protocol_dir, artifact_root=artifact_root
    )
    return _validate_prepared(prepared)


def load_validated_router_dataset_bundle(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    protocol_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> ValidatedRouterDatasetBundle:
    """Load tables and apply-only state after full lineage/reconstruction checks."""
    prepared = _prepare_inputs(
        config_path=config_path, protocol_dir=protocol_dir, artifact_root=artifact_root
    )
    validation = _validate_prepared(prepared)
    root = prepared.root
    loaded = ValidatedRouterDatasetBundle(
        root=root,
        dataset=_read_table(root / DATASET_FILENAME, dataset_schema()),
        audit=_read_table(root / AUDIT_FILENAME, audit_schema()),
        temperatures=_json_mapping(root / TEMPERATURE_FILENAME),
        transforms=_json_mapping(root / TRANSFORM_FILENAME),
        manifest=_json_mapping(root / MANIFEST_FILENAME),
        validation=validation,
    )
    if (
        loaded.manifest["manifest_sha256"] != validation["manifest_sha256"]
        or _artifact_records(root) != loaded.manifest["artifacts"]
    ):
        raise M4ArtifactError("M4 outputs changed while loading")
    loaded_hashes = {
        DATASET_FILENAME: _logical_table_hash(loaded.dataset),
        AUDIT_FILENAME: _logical_table_hash(loaded.audit),
        TEMPERATURE_FILENAME: canonical_sha256(loaded.temperatures),
        TRANSFORM_FILENAME: canonical_sha256(loaded.transforms),
    }
    if any(
        digest != loaded.manifest["artifacts"][name]["content_sha256"]
        for name, digest in loaded_hashes.items()
    ):
        raise M4ArtifactError("loaded M4 payloads differ from the validated manifest")
    _assert_parent_snapshot(prepared)
    return loaded

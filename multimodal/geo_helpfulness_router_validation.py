"""Label-blind, apply-only validation prediction preparation for M4.

This module never reads development-validation labels. The separate preview
scorer does so only after the complete prediction bundle has been sealed.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

# Initialize the frozen Torch/Arrow input layer before the SciPy numeric layer.
from multimodal.geo_helpfulness_router_validation_inputs import (
    load_training_inputs,
    load_validation_inputs,
)
from multimodal.geo_helpfulness_router_candidate_state import (
    candidate_filename,
    canonical_sha256,
    predict_router_probabilities,
)
from multimodal.geo_helpfulness_router_heuristics import (
    select_oof_heuristic_thresholds,
    validate_frozen_oof_heuristics,
)
from multimodal.geo_helpfulness_router_numeric import (
    apply_expert_temperature,
    transform_router_features,
)
from multimodal.geo_helpfulness_router_preview_score import (
    ARTIFACT_FILENAMES,
    DECLARATION_SCHEMA_VERSION,
    EXPERT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    Q_COLUMNS,
    expert_schema,
    prediction_schema,
    validate_sealed_predictions,
)
from multimodal.geo_helpfulness_targets_features import build_router_feature_frame


BUNDLE_RELATIVE_PATH = Path("router/router_validation_v1")
VARIANTS = ("primary_full727", "ablation_no_pairs79", "ablation_numeric25")
C_VALUES = (0.01, 0.1, 1.0, 10.0)
CODE_FILES = (
    "multimodal/geo_helpfulness_router_validation.py",
    "multimodal/geo_helpfulness_router_validation_inputs.py",
    "multimodal/geo_helpfulness_router_heuristics.py",
    "multimodal/geo_helpfulness_router_preview_score.py",
    "tools/run_multimodal_geo_helpfulness_m4_validation.py",
)


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected nonsymlink regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _check_snapshot(snapshot: dict[str, str]) -> None:
    for path, expected in snapshot.items():
        if _file_hash(Path(path)) != expected:
            raise ValueError(f"parent changed during preparation: {path}")


@contextmanager
def _exclusive_lock(root: Path):
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / f".{root.name}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "another validation preparation process owns the lock"
            ) from error
        yield
    finally:
        os.close(descriptor)


def _publish_bytes(path: Path, content: bytes, *, allow_create: bool) -> None:
    """Atomic no-overwrite publication; only this call's temporary file is removed."""
    if path.exists() or path.is_symlink():
        _file_hash(path)
        if path.read_bytes() != content:
            raise ValueError(f"immutable artifact differs: {path.name}")
        return
    if not allow_create:
        raise ValueError(f"missing committed artifact: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_table(path: Path, table: pa.Table, *, allow_create: bool) -> None:
    if path.exists() or path.is_symlink():
        _file_hash(path)
        saved = pq.read_table(path)
        if (
            not saved.schema.equals(table.schema, check_metadata=True)
            or saved.num_rows != table.num_rows
        ):
            raise ValueError(f"immutable table schema/count differs: {path.name}")
        for field in table.schema:
            if pa.types.is_floating(field.type):
                same = np.allclose(
                    saved[field.name].to_numpy(),
                    table[field.name].to_numpy(),
                    atol=1e-12,
                    rtol=1e-12,
                )
            else:
                same = saved[field.name].equals(table[field.name])
            if not same:
                raise ValueError(
                    f"reconstructed table differs: {path.name}:{field.name}"
                )
        return
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    _publish_bytes(path, sink.getvalue().to_pybytes(), allow_create=allow_create)


def derive_validation_tables(
    inputs: dict[str, Any], outputs: pd.DataFrame
) -> dict[str, pa.Table]:
    """Apply frozen state to all 48 candidates; neither fitting nor label access."""
    protocol_id = inputs["dataset_manifest"]["protocol_id"]
    candidate_frames = {variant: [] for variant in VARIANTS}
    expert_frames = []
    with threadpool_limits(limits=1):
        for seed in (1, 2, 3, 4):
            rows = (
                outputs.loc[outputs.training_seed.eq(seed)]
                .sort_values("row_uid")
                .reset_index(drop=True)
            )
            probabilities = {}
            for mode, prefix in (
                ("image_only", "image"),
                ("geo_only", "geo"),
                ("raw_concat", "raw"),
            ):
                logits = np.stack(rows[f"{prefix}_logits"].to_numpy()).astype(
                    np.float64, copy=False
                )
                probabilities[mode] = apply_expert_temperature(
                    logits, inputs["temperatures"][(seed, mode)]
                )
                if not np.array_equal(
                    probabilities[mode].argmax(axis=1),
                    rows[f"{prefix}_pred"].to_numpy(),
                ):
                    raise ValueError(
                        "calibrated argmax differs from sealed expert prediction"
                    )
            semantic = build_router_feature_frame(
                probabilities["image_only"],
                probabilities["geo_only"],
                probabilities["raw_concat"],
                probability_basis="scalar_temperature_calibrated",
            )
            matrix = transform_router_features(semantic, inputs["transforms"][seed])
            if (
                matrix.shape != (822, 727)
                or matrix.dtype != np.float64
                or not np.isfinite(matrix).all()
            ):
                raise ValueError(
                    "validation feature matrix violates frozen dimensions/dtype"
                )
            identity = rows[["row_uid", "plot_idx", "training_seed"]].copy()
            identity.insert(0, "protocol_id", protocol_id)
            expert = identity.copy()
            expert.insert(0, "schema_version", EXPERT_SCHEMA_VERSION)
            for prefix in ("image", "geo", "raw"):
                expert[f"{prefix}_pred"] = semantic[f"{prefix}_pred"].to_numpy()
            expert_frames.append(expert)
            for variant in VARIANTS:
                for c_value in C_VALUES:
                    state = inputs["candidate_states"][
                        candidate_filename(variant, seed, c_value)
                    ]
                    q = predict_router_probabilities(matrix, state)
                    frame = identity.copy()
                    frame.insert(0, "schema_version", PREDICTION_SCHEMA_VERSION)
                    frame["C_hex"] = float(c_value).hex()
                    for index, name in enumerate(Q_COLUMNS):
                        frame[name] = q[:, index]
                    frame["score"] = q[:, 0] - q[:, 1]
                    candidate_frames[variant].append(frame)
    tables = {}
    for variant, frames in candidate_frames.items():
        frame = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["C_hex", "training_seed", "row_uid"])
            .reset_index(drop=True)
        )
        tables[f"{variant}.parquet"] = pa.Table.from_pandas(
            frame, schema=prediction_schema(), preserve_index=False
        )
    experts = (
        pd.concat(expert_frames, ignore_index=True)
        .sort_values(["row_uid", "training_seed"])
        .reset_index(drop=True)
    )
    tables["expert_predictions.parquet"] = pa.Table.from_pandas(
        experts, schema=expert_schema(), preserve_index=False
    )
    return {name: table.replace_schema_metadata(None) for name, table in tables.items()}


def prepare_validation_bundle(
    *, repo_root: Path, protocol_root: Path, validate_only: bool = False
) -> dict[str, Any]:
    """Freeze OOF heuristics, then all label-blind predictions, with manifest last."""
    repo_root, protocol_root = Path(repo_root).resolve(), Path(protocol_root).resolve()
    root = protocol_root / BUNDLE_RELATIVE_PATH
    with _exclusive_lock(root):
        if root.is_symlink():
            raise ValueError("prediction bundle may not be a symlink")
        if validate_only and not (root / "manifest.json").is_file():
            raise ValueError(
                "validation requires a complete committed prediction bundle"
            )
        root.mkdir(exist_ok=True)
        allowed = set(ARTIFACT_FILENAMES) | {"manifest.json"}
        if not {path.name for path in root.iterdir()}.issubset(allowed):
            raise ValueError(
                "unknown files in immutable prediction directory; nothing was removed"
            )
        present = {path.name for path in root.iterdir()}
        if present and "preview_declaration.json" not in present:
            raise ValueError("partial bundle is missing its prior preview declaration")
        if (
            any(name.endswith(".parquet") for name in present)
            and "oof_heuristics.json" not in present
        ):
            raise ValueError(
                "partial prediction bundle is missing its prior OOF heuristics"
            )
        committed = (root / "manifest.json").exists()
        if committed:
            validate_sealed_predictions(root)
        inputs = load_training_inputs(repo_root, protocol_root)
        snapshot = inputs["parent_file_hashes"]
        code_hashes = {name: _file_hash(repo_root / name) for name in CODE_FILES}
        snapshot.update(
            {str(repo_root / name): digest for name, digest in code_hashes.items()}
        )
        heuristic_path = root / "oof_heuristics.json"
        if heuristic_path.exists() or heuristic_path.is_symlink():
            _file_hash(heuristic_path)
            heuristics = json.loads(heuristic_path.read_text(encoding="utf-8"))
            validate_frozen_oof_heuristics(
                heuristics,
                inputs["audit"],
                inputs["targets"],
                protocol_id=inputs["dataset_manifest"]["protocol_id"],
            )
        else:
            if committed or any(root.glob("*.parquet")):
                raise ValueError(
                    "cannot select missing OOF thresholds after prediction publication"
                )
            heuristics = select_oof_heuristic_thresholds(
                inputs["audit"],
                inputs["targets"],
                protocol_id=inputs["dataset_manifest"]["protocol_id"],
            )
        declaration = {
            "schema_version": DECLARATION_SCHEMA_VERSION,
            "protocol_id": inputs["dataset_manifest"]["protocol_id"],
            "candidate_variant": "primary_full727",
            "training_seed": 1,
            "C_hex": float(0.01).hex(),
            "metric": "four_state_accuracy_only",
            "selection_performed": False,
            "dataset_manifest_sha256": inputs["dataset_manifest"]["manifest_sha256"],
            "candidate_manifest_sha256": inputs["candidate_manifest"][
                "manifest_sha256"
            ],
            "oof_heuristics_sha256": canonical_sha256(heuristics),
            "code_file_sha256": code_hashes,
            "access_order": [
                "training_inputs",
                "preview_declaration",
                "oof_heuristics",
                "label_blind_validation_outputs",
                "complete_prediction_manifest",
                "separate_scoring_process",
            ],
        }
        _publish_bytes(
            root / "preview_declaration.json",
            _json_bytes(declaration),
            allow_create=not committed,
        )
        _publish_bytes(
            root / "oof_heuristics.json",
            _json_bytes(heuristics),
            allow_create=not committed,
        )
        _check_snapshot(snapshot)
        # The two preceding files exist durably before validation outputs are opened.
        outputs = load_validation_inputs(repo_root, protocol_root, inputs)
        tables = derive_validation_tables(inputs, outputs)
        _check_snapshot(snapshot)
        for name, table in tables.items():
            _publish_table(root / name, table, allow_create=not committed)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "protocol_id": declaration["protocol_id"],
            "assignment_file_sha256": inputs["assignment_file_sha256"],
            "dataset_manifest_sha256": declaration["dataset_manifest_sha256"],
            "candidate_manifest_sha256": declaration["candidate_manifest_sha256"],
            "artifacts": {
                name: {"file_sha256": _file_hash(root / name)}
                for name in ARTIFACT_FILENAMES
            },
            "parent_file_sha256": dict(sorted(snapshot.items())),
            "code_file_sha256": code_hashes,
            "environment": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "pyarrow", "scipy", "threadpoolctl")
                },
            },
            "dimensions": {
                "rows_per_seed": 822,
                "plots_per_seed": 325,
                "feature_columns": 727,
                "candidate_count": 48,
                "prediction_rows_per_variant": 13152,
                "expert_prediction_rows": 3288,
            },
            "numerical_tolerances": {
                "reconstruction_atol": 1e-12,
                "reconstruction_rtol": 1e-12,
                "expert_probability_row_sum_atol": 1e-8,
                "router_probability_row_sum_atol": 1e-12,
            },
            "access_boundary": {
                "validation_labels_read": False,
                "test_sources_opened": False,
                "router_or_temperature_or_transform_fits": 0,
                "validation_metrics_computed": False,
                "parent_validation_scope": "sealed_prediction_input_integrity_not_full_M2_checkpoint_replay",
            },
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _check_snapshot(snapshot)
        _publish_bytes(
            root / "manifest.json", _json_bytes(manifest), allow_create=not committed
        )
        result = validate_sealed_predictions(
            root, expected_manifest_sha256=manifest["manifest_sha256"]
        )
        _check_snapshot(snapshot)
        return {
            **result,
            "status": (
                "validated"
                if validate_only
                else ("reused_valid" if committed else "built")
            ),
            "bundle_root": str(root),
            "fitting_calls": 0,
            "validation_labels_read": False,
        }

"""Fit-process adapter for authenticated development-only capacity inputs.

This module intentionally imports the established training input validators and
must not be imported by a fit-incapable scorer. It never opens validation labels,
assignment table content, cleaned-test sources, or the legacy transfer outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from multimodal.geo_helpfulness_router_validation_inputs import (
    load_training_inputs,
    load_validation_inputs,
)
from multimodal.geo_helpfulness_router_candidate_state import (
    C_GRID,
    TARGET_ORDER,
    TRAINING_SEEDS,
    VARIANTS,
    candidate_filename,
    full_feature_columns,
    predict_router_probabilities,
)
from multimodal.geo_helpfulness_router_numeric import (
    apply_expert_temperature,
    transform_router_features,
)
from multimodal.geo_helpfulness_targets_features import FEATURE_COLUMNS
from multimodal.geo_helpfulness_router_preview_score import validate_sealed_predictions
from multimodal.geo_helpfulness_router_m4_common import (
    check_snapshot,
    file_hash,
    read_json,
    validate_manifest,
)


TRAIN_ROWS, TRAIN_PLOTS = 3378, 1300
VALIDATION_ROWS, VALIDATION_PLOTS = 822, 325
IDENTITY_COLUMNS = ("row_uid", "plot_idx", "training_seed")
TABLE_IDENTITY_COLUMNS = ("schema_version", "protocol_id", *IDENTITY_COLUMNS)
TARGET_COLUMNS = (*TABLE_IDENTITY_COLUMNS, "target_state")
Q_COLUMNS = tuple(f"q_{name}" for name in TARGET_ORDER)
MODES = {"image_only": "image", "geo_only": "geo", "raw_concat": "raw"}


def _merge_snapshot(destination: dict, source: dict) -> None:
    for name, digest in source.items():
        path = str(Path(name).absolute())
        if path in destination and destination[path] != digest:
            raise ValueError(f"capacity input parents disagree: {path}")
        destination[path] = digest


def _add_bundle(snapshot: dict, root: Path, manifest: dict) -> None:
    _merge_snapshot(snapshot, manifest["parent_file_sha256"])
    _merge_snapshot(snapshot, {str(root / "manifest.json"): file_hash(root / "manifest.json")})
    _merge_snapshot(snapshot, {
        str(root / name): record["file_sha256"]
        for name, record in manifest["artifacts"].items()
    })
    check_snapshot(snapshot)


def _seed_rows(frame: pd.DataFrame, seed: int, *, validation: bool) -> pd.DataFrame:
    rows = frame.loc[frame.training_seed.eq(seed)].sort_values("row_uid", kind="stable").reset_index(drop=True)
    expected_rows, expected_plots = (VALIDATION_ROWS, VALIDATION_PLOTS) if validation else (TRAIN_ROWS, TRAIN_PLOTS)
    if (
        len(rows) != expected_rows or rows.plot_idx.nunique() != expected_plots
        or rows.row_uid.duplicated().any()
        or rows[list(IDENTITY_COLUMNS)].isna().any().any()
    ):
        raise ValueError("capacity input image/plot/seed coverage mismatch")
    return rows


def _match_identity(left: pd.DataFrame, right: pd.DataFrame, *, name: str) -> None:
    if not left[list(IDENTITY_COLUMNS)].equals(right[list(IDENTITY_COLUMNS)]):
        raise ValueError(f"capacity input identity mismatch: {name}")


def _verify_matrix(saved: np.ndarray, reconstructed: np.ndarray, name: str) -> None:
    if (
        saved.dtype != np.float64 or reconstructed.dtype != np.float64
        or saved.ndim != 2 or saved.shape[1] != 727
        or not np.isfinite(saved).all()
        or not np.array_equal(saved, reconstructed)
    ):
        raise ValueError(f"capacity {name} matrix does not exactly reproduce frozen features")


def _replay_lr(
    directory: Path, states: dict, seed_inputs: dict, snapshot: dict
) -> pd.DataFrame:
    """Reproduce every saved LR candidate without labels or candidate selection."""
    expected_states = {
        candidate_filename(variant, seed, C)
        for variant in VARIANTS for seed in TRAINING_SEEDS for C in C_GRID
    }
    if set(states) != expected_states:
        raise ValueError("capacity adapter requires the complete 48-candidate LR grid")
    frames = []
    for variant in VARIANTS:
        path = directory / f"{variant}.parquet"
        if file_hash(path) != snapshot[str(path)]:
            raise ValueError("saved LR prediction file changed before decoding")
        table = pq.read_table(path).to_pandas()
        if len(table) != VALIDATION_ROWS * len(TRAINING_SEEDS) * len(C_GRID):
            raise ValueError("saved LR prediction cross-product has wrong row count")
        expected_keys = set()
        for seed in TRAINING_SEEDS:
            identity = seed_inputs[seed]["validation_identity"]
            for C in C_GRID:
                rows = table.loc[table.training_seed.eq(seed) & table.C_hex.eq(C.hex())].sort_values("row_uid", kind="stable").reset_index(drop=True)
                _match_identity(rows, identity, name="cached LR predictions")
                state = states[candidate_filename(variant, seed, C)]
                probabilities = predict_router_probabilities(seed_inputs[seed]["X_validation"], state)
                observed = rows[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
                if not np.allclose(observed, probabilities, atol=1e-12, rtol=1e-12):
                    raise ValueError(f"saved LR probabilities do not reproduce: {variant}/{seed}/{C}")
                score = probabilities[:, 0] - probabilities[:, 1]
                if not np.allclose(rows.score.to_numpy(), score, atol=1e-12, rtol=1e-12):
                    raise ValueError(f"saved LR scores do not reproduce: {variant}/{seed}/{C}")
                expected_keys.update((uid, seed, C.hex()) for uid in identity.row_uid)
        observed_keys = list(table[["row_uid", "training_seed", "C_hex"]].itertuples(index=False, name=None))
        if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != expected_keys:
            raise ValueError("saved LR prediction keys do not match the full candidate grid")
        table.insert(0, "variant_id", variant)
        frames.append(table)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["variant_id", "C_hex", "training_seed", "row_uid"], kind="stable"
    ).reset_index(drop=True)


def load_capacity_inputs(repo_root: Path, protocol_root: Path) -> dict[str, Any]:
    """Return the same frozen 727-column development matrices for every learner.

    ``seed_inputs[seed]`` supplies X_train, y_train (the fixed four-state order),
    train_identity, X_validation and validation_identity. ``old_lr_predictions``
    retains all three feature variants, four C values and four seeds. No target
    for the validation partition is present anywhere in the returned object.
    """
    repo_root, protocol_root = Path(repo_root).absolute(), Path(protocol_root).absolute()
    inputs = load_training_inputs(repo_root, protocol_root)
    outputs = load_validation_inputs(repo_root, protocol_root, inputs)
    snapshot = dict(inputs["parent_file_hashes"])
    router = protocol_root / "router"
    comparator_root = router / "m4_comparator_inputs_v1"
    comparator_manifest = validate_manifest(comparator_root, schema_version="geo_helpfulness.m4_comparator_manifest.v1")
    _add_bundle(snapshot, comparator_root, comparator_manifest)
    lr_root = router / "router_validation_v1"
    sealed_lr = validate_sealed_predictions(lr_root)
    lr_manifest = read_json(lr_root / "manifest.json")
    if (
        sealed_lr["manifest_sha256"] != lr_manifest["manifest_sha256"]
        or lr_manifest["dataset_manifest_sha256"] != inputs["dataset_manifest"]["manifest_sha256"]
        or lr_manifest["candidate_manifest_sha256"] != inputs["candidate_manifest"]["manifest_sha256"]
        or comparator_manifest["prediction_manifest_sha256"] != lr_manifest["manifest_sha256"]
        or comparator_manifest["assignment_file_sha256"] != inputs["assignment_file_sha256"]
    ):
        raise ValueError("capacity input manifests have mixed development lineage")
    _add_bundle(snapshot, lr_root, lr_manifest)
    columns = list(full_feature_columns())
    if inputs["dataset_manifest"]["feature_columns"] != columns:
        raise ValueError("capacity input feature order differs from the frozen 727 columns")
    train = pq.read_table(router / "router_dataset.parquet", columns=[*TARGET_COLUMNS, *columns]).to_pandas()
    train = train.sort_values(["row_uid", "training_seed"], kind="stable").reset_index(drop=True)
    targets = inputs["targets"].sort_values(["row_uid", "training_seed"], kind="stable").reset_index(drop=True)
    if not train[list(TARGET_COLUMNS)].equals(targets):
        raise ValueError("capacity training targets differ from authenticated OOF targets")
    experts = pq.read_table(comparator_root / "expert_probabilities_and_features.parquet").to_pandas()
    expected_expert_columns = [
        *TABLE_IDENTITY_COLUMNS,
        *(f"{prefix}_prob_{basis}" for basis in ("calibrated", "native_t1") for prefix in MODES.values()),
        *FEATURE_COLUMNS,
    ]
    if list(experts) != expected_expert_columns or experts.isna().any().any():
        raise ValueError("capacity validation comparator must contain only frozen label-blind columns")
    experts = experts.sort_values(["row_uid", "training_seed"], kind="stable").reset_index(drop=True)
    for frame in (train, experts):
        if set(frame.training_seed) != set(TRAINING_SEEDS) or set(frame.protocol_id) != {"protocol_v1"}:
            raise ValueError("capacity training/validation seed or protocol grid mismatch")
    validation_frames, seed_inputs = [], {}
    for seed in TRAINING_SEEDS:
        training_rows = _seed_rows(train, seed, validation=False)
        audit_rows = _seed_rows(inputs["audit"], seed, validation=False)
        validation_rows = _seed_rows(experts, seed, validation=True)
        output_rows = _seed_rows(outputs, seed, validation=True)
        _match_identity(training_rows, audit_rows, name="OOF matrix/audit")
        _match_identity(validation_rows, output_rows, name="validation comparator/M2 outputs")
        if set(training_rows.plot_idx) & set(validation_rows.plot_idx) or set(training_rows.row_uid) & set(validation_rows.row_uid):
            raise ValueError("capacity training and validation identities overlap")
        if set(training_rows.target_state) != set(TARGET_ORDER):
            raise ValueError("capacity OOF training seed must contain all four target states")
        X_train = np.ascontiguousarray(training_rows[columns].to_numpy(dtype=np.float64))
        rebuilt = transform_router_features(audit_rows[list(FEATURE_COLUMNS)], inputs["transforms"][seed])
        _verify_matrix(X_train, rebuilt, "training")
        X_validation = np.ascontiguousarray(transform_router_features(validation_rows[list(FEATURE_COLUMNS)], inputs["transforms"][seed]))
        for mode, prefix in MODES.items():
            probability = apply_expert_temperature(np.stack(output_rows[f"{prefix}_logits"]), inputs["temperatures"][(seed, mode)])
            if not np.array_equal(probability, np.stack(validation_rows[f"{prefix}_prob_calibrated"])):
                raise ValueError("capacity validation calibrated probabilities differ from M2 plus frozen temperatures")
            if not np.array_equal(np.stack(output_rows[f"{prefix}_prob_native_t1"]), np.stack(validation_rows[f"{prefix}_prob_native_t1"])):
                raise ValueError("capacity validation native probabilities differ from M2")
        validation_frames.append(pd.concat([
            validation_rows[list(TABLE_IDENTITY_COLUMNS)],
            pd.DataFrame(X_validation, columns=columns),
        ], axis=1))
        seed_inputs[seed] = {
            "X_train": X_train,
            "y_train": training_rows.target_state.map({name: index for index, name in enumerate(TARGET_ORDER)}).to_numpy(dtype=np.int64),
            "train_identity": training_rows[list(IDENTITY_COLUMNS)].copy(),
            "X_validation": X_validation,
            "validation_identity": validation_rows[list(IDENTITY_COLUMNS)].copy(),
        }
    old_lr = _replay_lr(lr_root, inputs["candidate_states"], seed_inputs, snapshot)
    check_snapshot(snapshot)
    return {
        "train": train,
        "validation": pd.concat(validation_frames, ignore_index=True).sort_values(["row_uid", "training_seed"], kind="stable").reset_index(drop=True),
        "experts": experts,
        "seed_inputs": seed_inputs,
        "candidate_states": inputs["candidate_states"],
        "parent_file_hashes": snapshot,
        "old_lr_predictions": old_lr,
        "feature_columns": columns,
        "target_order": list(TARGET_ORDER),
        "dataset_manifest": inputs["dataset_manifest"],
        "candidate_manifest": inputs["candidate_manifest"],
        "comparator_manifest": comparator_manifest,
        "lr_prediction_manifest": lr_manifest,
        "transforms": inputs["transforms"],
        "temperatures": inputs["temperatures"],
    }

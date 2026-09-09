"""Declare legacy sources before label-blind, apply-only expert inference."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml

from multimodal.geo_helpfulness_legacy_router_common import (
    FAMILIES,
    SEEDS,
    STUDY_ID,
    PROJECT_ROOT,
    canonical_hash,
    check_self_hash,
    check_snapshot,
    file_hash,
    load_declaration,
    normalize_identity,
    read_json,
    write_json,
    exclusive_lock,
)
from multimodal.geo_helpfulness_router_m4_common import validate_manifest


SCHEMA_VERSION = "geo_helpfulness.legacy_router_declaration.v1"
DATASET_ID = "cs_gse_100m_l3"
RUN_TAG = "gse_100m_train_cleaned_test_epoch50"
IMAGE_COLUMNS = tuple(f"I{i:04d}" for i in range(1152))
GEO_COLUMNS = tuple(f"A{i:02d}" for i in range(64))
MODES = {"image_only": "image", "geo_only": "geo", "raw_concat": "raw"}
INPUT_DIMS = {"image_only": 1152, "geo_only": 64, "raw_concat": 1216}
CODE_FILES = (
    "multimodal/geo_helpfulness_legacy_router_common.py",
    "multimodal/geo_helpfulness_legacy_router_inference.py",
    "multimodal/geo_helpfulness_legacy_router_report.py",
    "multimodal/geo_helpfulness_legacy_router_score.py",
    "tools/run_multimodal_geo_helpfulness_legacy_router.py",
)


def _code_snapshot() -> dict[str, str]:
    return {
        str(PROJECT_ROOT / name): file_hash(PROJECT_ROOT / name) for name in CODE_FILES
    }


def _identity(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.loc[:, ["file", "plot_idx"]].isna().any().any():
        raise ValueError("null sample identity")
    pairs = [
        normalize_identity(f, p)
        for f, p in frame[["file", "plot_idx"]].itertuples(index=False, name=None)
    ]
    if len({f for f, _ in pairs}) != len(pairs):
        raise ValueError("duplicate or colliding file identity")
    return pd.DataFrame(
        {
            "row_uid": [canonical_hash([DATASET_ID, f, p]) for f, p in pairs],
            "file_lower": [f for f, _ in pairs],
            "plot_idx": [p for _, p in pairs],
        }
    )


def _load_head(path: Path, mode: str, family: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "mode", "epoch", "training_regime"}
    if family == "oracle_test_best_model":
        required |= {"selection_split", "selection_metric", "diagnostic_only"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != required:
        raise ValueError(f"unexpected legacy checkpoint keys: {path}")
    if (
        checkpoint["mode"] != mode
        or checkpoint["training_regime"] != "fixed_epoch_train_val"
    ):
        raise ValueError(f"legacy checkpoint mode/regime mismatch: {path}")
    epoch = checkpoint["epoch"]
    if type(epoch) is not int or not 1 <= epoch <= 50:
        raise ValueError(f"invalid legacy epoch: {path}")
    if family == "best_model" and epoch != 50:
        raise ValueError("fixed final checkpoint must be epoch 50")
    if family == "oracle_test_best_model" and (
        checkpoint["selection_split"] != "test"
        or checkpoint["selection_metric"] != "top1_acc"
        or checkpoint["diagnostic_only"] is not True
    ):
        raise ValueError(
            "oracle checkpoint must declare test-selected diagnostic status"
        )
    shapes = {
        "head.net.0.weight": (256, INPUT_DIMS[mode]),
        "head.net.0.bias": (256,),
        "head.net.3.weight": (18, 256),
        "head.net.3.bias": (18,),
    }
    state = checkpoint["model_state"]
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("legacy head state keys differ from the declared MLP")
    for name, shape in shapes.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != torch.float32
        ):
            raise ValueError(f"legacy head shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite legacy head parameter: {name}")
    return checkpoint


def _load_scaler(path: Path, ontology: list[dict]) -> dict:
    scaler = read_json(path)
    if (
        scaler.get("fit_split") != "train_val"
        or scaler.get("fit_rows") != 4200
        or scaler.get("modality_name") != "geo"
        or scaler.get("feature_columns") != list(GEO_COLUMNS)
        or scaler.get("class_names") != [item["label_name"] for item in ontology]
        or scaler.get("label_id_remap") != {str(i): i for i in range(18)}
    ):
        raise ValueError(
            "legacy scaler fit source, feature order, or class order mismatch"
        )
    for key in ("mean", "std"):
        values = np.asarray(scaler[key], dtype=np.float32)
        if values.shape != (64,) or not np.isfinite(values).all():
            raise ValueError(f"invalid scaler {key}")
        if key == "std" and np.any(values <= 0):
            raise ValueError("scaler standard deviations must be positive")
    return scaler


def _verify_development(
    caches: dict, assignments: pd.DataFrame, scaler: dict, expected_hash: str
) -> dict:
    expected = assignments.sort_values("row_uid").reset_index(drop=True)
    reference_hashes = None
    audit = {}
    columns = ["file", "plot_idx", "label_id", "label_name", *GEO_COLUMNS]
    for seed in SEEDS:
        paths = [Path(caches[str(seed)][split]) for split in ("train", "val")]
        hashes = [file_hash(path) for path in paths]
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise ValueError("legacy development cache bytes differ between seeds")
        parts = [pd.read_parquet(path, columns=columns) for path in paths]
        frame = pd.concat(parts, ignore_index=True)
        identity = _identity(frame)
        joined = (
            pd.concat([identity, frame[["label_id", "label_name"]]], axis=1)
            .sort_values("row_uid")
            .reset_index(drop=True)
        )
        if len(frame) != 4200 or identity.plot_idx.nunique() != 1625:
            raise ValueError(
                "legacy development universe must contain 4200 images and 1625 plots"
            )
        if joined.row_uid.tolist() != expected.row_uid.tolist():
            raise ValueError("legacy development identities differ from M1")
        if (
            joined.label_id.tolist() != expected.label_id_dense.tolist()
            or joined.label_name.tolist() != expected.label_name.tolist()
        ):
            raise ValueError("legacy development labels differ from M1")
        identity_columns = ["row_uid", "file_lower", "plot_idx"]
        digest = canonical_hash(
            identity[identity_columns].sort_values(identity_columns).to_dict("records")
        )
        if digest != expected_hash:
            raise ValueError("legacy development identity fingerprint differs from M1")
        geo = frame[list(GEO_COLUMNS)].astype(np.float32)
        if not np.isfinite(geo.to_numpy()).all():
            raise ValueError("nonfinite development geographic features")
        mean = geo.mean(axis=0).to_numpy(dtype=np.float32)
        std = geo.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
        std[std == 0] = 1.0
        if not np.array_equal(
            mean, np.asarray(scaler["mean"], dtype=np.float32)
        ) or not np.array_equal(std, np.asarray(scaler["std"], dtype=np.float32)):
            raise ValueError(
                "saved geographic scaler does not reproduce from declared development caches"
            )
        audit[str(seed)] = {
            "train_rows": len(parts[0]),
            "val_rows": len(parts[1]),
            "rows": len(frame),
            "plots": 1625,
            "identity_sha256": digest,
        }
    return audit


def _locked_identity(manifest: dict) -> pd.DataFrame:
    check_self_hash(manifest)
    rows = pd.DataFrame(manifest["rows"])
    if len(rows) != 1347 or rows.plot_idx.nunique() != 531:
        raise ValueError("locked test must contain 1347 images and 531 plots")
    observed = _identity(rows)
    if (
        observed.row_uid.tolist() != rows.row_uid.tolist()
        or observed.file_lower.tolist() != rows.file_lower.tolist()
    ):
        raise ValueError("locked identity row UID or file canonicalization mismatch")
    if canonical_hash(manifest["rows"]) != manifest["identity_projection_sha256"]:
        raise ValueError("locked identity projection fingerprint mismatch")
    return rows.sort_values("row_uid").reset_index(drop=True)


def create_declaration(protocol_root, legacy_root, cache_root, output_root) -> dict:
    """Seal source byte hashes using development labels only; never decode test data."""
    raw_output = Path(output_root).absolute()
    if any(path.is_symlink() for path in (raw_output, *raw_output.parents)):
        raise ValueError("output directory must not contain symlinks")
    output = raw_output.resolve()
    for source in (protocol_root, legacy_root, cache_root):
        source = Path(source).resolve()
        if (
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise ValueError(
                "output must be separate from frozen protocol and legacy source directories"
            )
    with exclusive_lock(output):
        if output.exists() and not output.is_dir():
            raise ValueError("output must be a directory")
        if (
            output.is_dir()
            and any(output.iterdir())
            and not (output / "declaration.json").is_file()
        ):
            raise ValueError(
                "nonempty output lacks a prior declaration; refusing unknown artifacts"
            )
        return _create_declaration(protocol_root, legacy_root, cache_root, output)


def _create_declaration(protocol_root, legacy_root, cache_root, output_root) -> dict:
    protocol_root, legacy_root, cache_root, output_root = map(
        lambda p: Path(p).resolve(),
        (protocol_root, legacy_root, cache_root, output_root),
    )
    if (output_root / "declaration.json").exists():
        declaration = load_declaration(output_root)
        for key, root in (
            ("protocol_root", protocol_root),
            ("legacy_root", legacy_root),
            ("cache_root", cache_root),
        ):
            if declaration[key] != str(root):
                raise ValueError(f"existing declaration uses a different {key}")
        return declaration
    if cache_root.name != "gse_100m_cleaned_test":
        raise ValueError("the exact gse_100m_cleaned_test cache root is required")
    frozen_root = protocol_root / "router/m4_feasibility_v1"
    frozen = validate_manifest(
        frozen_root, schema_version="geo_helpfulness.m4_feasibility_manifest.v1"
    )
    if frozen["primary_decision"] != "no_go":
        raise ValueError("M4 no-go must remain frozen")
    snapshot = dict(frozen["parent_file_sha256"])
    for name, record in frozen["artifacts"].items():
        snapshot[str(frozen_root / name)] = record["file_sha256"]
    snapshot[str(frozen_root / "manifest.json")] = file_hash(
        frozen_root / "manifest.json"
    )
    state = read_json(frozen_root / "router_frozen_state.json")
    for name, record in state["components"].items():
        if snapshot[str(frozen_root / name)] != record["file_sha256"]:
            raise ValueError("frozen M4 component fingerprint mismatch")
    m1_root = protocol_root / "protocol"
    m1 = read_json(m1_root / "protocol_manifest.json")
    check_self_hash(m1, "manifest_payload_sha256")
    snapshot[str(m1_root / "protocol_manifest.json")] = file_hash(
        m1_root / "protocol_manifest.json"
    )
    for name, digest in m1["artifact_file_sha256"].items():
        snapshot[str(m1_root / name)] = digest
    for name, digest in m1["development_source_file_sha256"].items():
        snapshot[str((PROJECT_ROOT / name).resolve())] = digest
    check_snapshot(snapshot)
    config = yaml.safe_load((m1_root / "resolved_protocol.yaml").read_text())
    ontology = config["class_ontology"]["classes"]
    if len(ontology) != 18 or [entry["dense_id"] for entry in ontology] != list(
        range(18)
    ):
        raise ValueError("frozen ontology must have 18 ordered dense classes")
    if canonical_hash(config["class_ontology"]) != m1["class_map_sha256"]:
        raise ValueError("frozen ontology fingerprint mismatch")
    registry = (PROJECT_ROOT / config["paths"]["locked_test_registry_root"]).resolve()
    reference = read_json(m1_root / "locked_test_snapshot_ref.json")
    identity_path = registry / reference["identity_manifest_relative_path"]
    identity_manifest = read_json(identity_path)
    identity = _locked_identity(identity_manifest)
    if (
        file_hash(identity_path) != reference["identity_manifest_file_sha256"]
        or identity_manifest["identity_projection_sha256"]
        != m1["locked_test_identity_projection_sha256"]
    ):
        raise ValueError("M1 locked identity reference mismatch")
    active = read_json(registry / "active_snapshot.json")
    if active["snapshot_id"] != identity_manifest["snapshot_id"]:
        raise ValueError("active locked snapshot differs from M1")
    snapshot[str(identity_path)] = file_hash(identity_path)
    snapshot[str(registry / "active_snapshot.json")] = file_hash(
        registry / "active_snapshot.json"
    )
    caches = {
        str(seed): {
            split: str(cache_root / f"seed{seed}" / f"{split}.parquet")
            for split in ("train", "val", "test")
        }
        for seed in SEEDS
    }
    legacy_runs = {
        mode: {
            str(seed): str(legacy_root / mode / RUN_TAG / f"seed{seed}")
            for seed in SEEDS
        }
        for mode in MODES
    }
    heads, historical_hashes, common_scaler = [], {}, None
    scaler_hash = None
    for mode in MODES:
        for seed in SEEDS:
            run = Path(legacy_runs[mode][str(seed)])
            scaler_path = run / "geo_standardization.json"
            scaler = _load_scaler(scaler_path, ontology)
            digest = file_hash(scaler_path)
            if scaler_hash is not None and digest != scaler_hash:
                raise ValueError("saved legacy scalers differ across modes or seeds")
            scaler_hash, common_scaler = digest, scaler
            snapshot[str(scaler_path)] = digest
            for family in FAMILIES:
                path = run / f"{family}.pt"
                snapshot[str(path)] = file_hash(path)
                checkpoint = _load_head(path, mode, family)
                heads.append(
                    {
                        "checkpoint_family": family,
                        "mode": mode,
                        "training_seed": seed,
                        "path": str(path),
                        "file_sha256": snapshot[str(path)],
                        **{
                            key: value
                            for key, value in checkpoint.items()
                            if key != "model_state"
                        },
                    }
                )
            metrics = run / "metrics.json"
            historical_hashes[str(metrics)] = file_hash(metrics)
    for paths in caches.values():
        for path in paths.values():
            snapshot[path] = file_hash(Path(path))
    assignments = pd.read_parquet(
        m1_root / "development_assignments.parquet",
        columns=["row_uid", "plot_idx", "label_id_dense", "label_name"],
    )
    if set(assignments.row_uid) & set(identity.row_uid) or set(
        assignments.plot_idx
    ) & set(identity.plot_idx):
        raise ValueError("development and locked identities overlap")
    audit = _verify_development(
        caches, assignments, common_scaler, m1["development_identity_projection_sha256"]
    )
    snapshot.update(_code_snapshot())
    # Historical metrics are provenance-only byte hashes, never replay dependencies.
    snapshot = {
        path: digest
        for path, digest in snapshot.items()
        if path not in historical_hashes
    }
    check_snapshot(snapshot)
    declaration = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "protocol_id": "protocol_v1",
        "no_go_preserved": True,
        "protocol_root": str(protocol_root),
        "legacy_root": str(legacy_root),
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "frozen_root": str(frozen_root),
        "registry_root": str(registry / "events"),
        "source_snapshot": dict(sorted(snapshot.items())),
        "historical_metrics_sha256": historical_hashes,
        "caches": caches,
        "legacy_runs": legacy_runs,
        "ontology": ontology,
        "identity_manifest": identity_manifest,
        "checkpoints": heads,
        "development_audit": audit,
        "scaler_file_sha256": scaler_hash,
        "checkpoint_families": list(FAMILIES),
        "training_seeds": list(SEEDS),
        "family_descriptions": {
            "best_model": "fixed epoch 50 legacy expert checkpoint, historically test exposed",
            "oracle_test_best_model": "test-top1-selected legacy expert checkpoint, biased diagnostic",
        },
        "user_authorization": {
            "evaluate_both_checkpoint_families": list(FAMILIES),
            "scope": "Apply frozen M4 routers to both existing legacy checkpoint families on the cleaned test set; score descriptively without fitting or selection.",
        },
        "environment": {
            "python": platform.python_version(),
            **{
                name: metadata.version(name)
                for name in ("numpy", "pandas", "torch", "pyarrow", "PyYAML")
            },
        },
        "inference": {
            "device": "cpu",
            "batch_size": 128,
            "torch_num_threads": 1,
            "parameter_dtype": "float32",
            "logit_storage_dtype": "float64",
            "additional_image_normalization": False,
            "test_projection": ["file", "plot_idx", *IMAGE_COLUMNS, *GEO_COLUMNS],
        },
        "access_boundary": "declaration_decodes_no_test_features_labels_or_historical_metrics",
        "interpretation": "descriptive_legacy_transfer_with_frozen_M4_policies",
        "provenance_limitations": [
            "Legacy heads omit source/encoder/class fingerprints; class order is verified through saved scalers and M1 development labels.",
            "Checkpoint-to-cache association is supported by paths and exactly reproduced development scalers; original artifacts do not cryptographically bind this association.",
            "The globally habitat-adapted image encoder overlaps development provenance; this is not M1's fold-contained encoder recipe.",
            "Oracle checkpoints were selected using cleaned-test accuracy; they are diagnostic and cannot support independent performance claims.",
            "M4 OOF temperatures transfer unchanged; calibration on legacy checkpoints is not guaranteed.",
        ],
    }
    declaration["declaration_sha256"] = canonical_hash(declaration)
    write_json(output_root / "declaration.json", declaration)
    return load_declaration(output_root)


def _head_logits(checkpoint: dict, values: np.ndarray) -> np.ndarray:
    state = checkpoint["model_state"]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(values), 128):
            batch = torch.from_numpy(
                np.ascontiguousarray(values[start : start + 128], dtype=np.float32)
            )
            hidden = functional.relu(
                functional.linear(
                    batch, state["head.net.0.weight"], state["head.net.0.bias"]
                )
            )
            logits = functional.linear(
                hidden, state["head.net.3.weight"], state["head.net.3.bias"]
            )
            chunks.append(logits.numpy().astype(np.float64))
    output = np.concatenate(chunks)
    if output.shape != (len(values), 18) or not np.isfinite(output).all():
        raise ValueError("invalid legacy expert logits")
    return output


def infer_expert_logits(declaration: dict) -> pd.DataFrame:
    """Apply frozen heads, using a persisted declaration and projected test reads."""
    persisted = load_declaration(Path(declaration["output_root"]))
    if persisted != declaration:
        raise ValueError("inference declaration differs from persisted declaration")
    torch.set_num_threads(1)
    identity = _locked_identity(declaration["identity_manifest"])
    frames = []
    for seed in SEEDS:
        path = Path(declaration["caches"][str(seed)]["test"])
        frame = pd.read_parquet(
            path, columns=["file", "plot_idx", *IMAGE_COLUMNS, *GEO_COLUMNS]
        )
        observed = _identity(frame)
        if set(observed.row_uid) != set(identity.row_uid) or len(observed) != len(
            identity
        ):
            raise ValueError("test feature identity differs from the locked manifest")
        frame.index = observed.row_uid
        frame = frame.loc[identity.row_uid].reset_index(drop=True)
        image = frame[list(IMAGE_COLUMNS)].to_numpy(dtype=np.float32)
        geo = frame[list(GEO_COLUMNS)].to_numpy(dtype=np.float32)
        if not np.isfinite(image).all() or not np.isfinite(geo).all():
            raise ValueError("nonfinite test feature values")
        for family in FAMILIES:
            rows = identity[["row_uid", "file", "plot_idx"]].copy()
            rows.insert(0, "checkpoint_family", family)
            rows["training_seed"] = seed
            for mode, prefix in MODES.items():
                run = Path(declaration["legacy_runs"][mode][str(seed)])
                scaler = _load_scaler(
                    run / "geo_standardization.json", declaration["ontology"]
                )
                standardized = (
                    geo - np.asarray(scaler["mean"], dtype=np.float32)
                ) / np.asarray(scaler["std"], dtype=np.float32)
                values = image if mode == "image_only" else standardized
                if mode == "raw_concat":
                    values = np.concatenate([image, standardized], axis=1)
                rows[f"{prefix}_logits"] = list(
                    _head_logits(_load_head(run / f"{family}.pt", mode, family), values)
                )
            frames.append(rows)
    check_snapshot(declaration["source_snapshot"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["checkpoint_family", "row_uid", "training_seed"])
        .reset_index(drop=True)
    )

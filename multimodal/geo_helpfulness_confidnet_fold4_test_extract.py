"""Label-blind cleaned-test inference with the frozen fold-four expert bundle.

Only image identities and native A00..A63 features are read from the cleaned
table. Cached image representations and all labels are deliberately excluded.
The image encoder, geo scaler, and three heads are the producer-fold-3 states
that generated the completed single-fold ConfidNet experiment.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from multimodal import geo_helpfulness_confidnet_fold4_extract as fold4
from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_protocol import (
    build_identity_projection,
    canonical_sha256,
    canonicalize_file,
    sha256_file,
    validate_locked_test_identity_manifest,
)
from multimodal.models import MLPHead


DEFAULT_CONFIG_PATH = m2.DEFAULT_CONFIG_PATH
DEFAULT_TEST_SOURCE = m2.PROJECT_ROOT / (
    "multimodal_artifacts/joined_tables/cs/"
    "hf-hub_timm_ViT-SO400M-16-SigLIP2-384_5_20260203_17/"
    "gse_100m_cleaned_test/seed1/test.parquet"
)
TEST_IMAGE_ROOT = m2.PROJECT_ROOT / "data/CS_Xplots_2019_2023_test"
SOURCE_COLUMNS = ("file", "plot_idx", "image_source", *m2.GEO_COLUMNS)
PRODUCER_FOLD = fold4.PRODUCER_FOLD
PREFIXES = fold4.PREFIXES
ExtractionError = fold4.ConfidNetFold4ExtractionError


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ExtractionError(f"expected JSON object: {path}")
    return value


def _inference_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Change only the allowed raw-image location for the test prediction pass."""

    result = copy.deepcopy(dict(config))
    result["development_universe"]["allowed_image_sources"] = [str(TEST_IMAGE_ROOT)]
    return result


def load_test_sources(
    *,
    context: Any | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    source_path: str | Path = DEFAULT_TEST_SOURCE,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Validate locked identities and return only ordered label-blind inputs.

    File fingerprints include the source parquet and raw images; hashing a file
    does not decode its labels. The explicit parquet projection is the complete
    read allow-list, and does not include any cached I-prefixed image features.
    """

    if context is None:
        context = m2.load_frozen_context(config_path=config_path)
    source = m2._resolve_project_path(source_path)
    reference_path = context.protocol_dir / "locked_test_snapshot_ref.json"
    reference = _read_json(reference_path)
    registry = m2._resolve_project_path(context.config["paths"]["locked_test_registry_root"])
    identity_path = (registry / reference["identity_manifest_relative_path"]).resolve()
    if not identity_path.is_relative_to(registry.resolve()):
        raise ExtractionError("locked identity reference escapes the registry")
    manifest = _read_json(identity_path)
    validate_locked_test_identity_manifest(manifest)
    active_path = registry / "active_snapshot.json"
    active = _read_json(active_path)
    if (
        sha256_file(identity_path) != reference["identity_manifest_file_sha256"]
        or manifest["snapshot_id"] != reference["snapshot_id"]
        or manifest["identity_projection_sha256"] != reference["identity_projection_sha256"]
        or manifest["identity_projection_sha256"]
        != context.protocol_manifest["locked_test_identity_projection_sha256"]
        or active["snapshot_id"] != manifest["snapshot_id"]
    ):
        raise ExtractionError("locked cleaned-test identity reference mismatch")
    expected = context.config["locked_test_identity"]
    if (manifest["row_count"] != int(expected["expected_rows"])
            or manifest["plot_count"] != int(expected["expected_plots"])):
        raise ExtractionError("unexpected cleaned-test row or plot count")
    frame = pd.read_parquet(source, columns=list(SOURCE_COLUMNS))
    if tuple(frame.columns) != SOURCE_COLUMNS or frame.isna().any().any():
        raise ExtractionError("cleaned-test source projection is incomplete or null")
    identity = build_identity_projection(frame, str(context.config["dataset_id"]))
    if identity.to_dict("records") != manifest["rows"]:
        raise ExtractionError("cleaned source identities differ from the locked snapshot")
    for key in ("row_uid", "plot_idx"):
        if set(identity[key]) & set(context.assignments[key]):
            raise ExtractionError(f"development and cleaned-test overlap by {key}")
    frame = frame.copy()
    frame["file_lower"] = [canonicalize_file(value).casefold() for value in frame["file"]]
    indexed = frame.set_index("file_lower", verify_integrity=True).loc[identity["file_lower"]]
    image_sources = [m2._resolve_project_path(value) for value in indexed["image_source"]]
    if any(path != TEST_IMAGE_ROOT.resolve() for path in image_sources):
        raise ExtractionError("unexpected cleaned-test raw-image source")
    rows = identity.loc[:, ["row_uid", "file", "file_lower", "plot_idx"]].copy()
    rows["image_source"] = str(TEST_IMAGE_ROOT.resolve())
    rows = fold4.initial._prediction_projection(rows)
    geo = indexed.loc[:, list(m2.GEO_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    if geo.shape != (len(rows), len(m2.GEO_COLUMNS)) or not np.isfinite(geo).all():
        raise ExtractionError("native cleaned-test geo features are invalid")
    inference_config = _inference_config(context.config)
    image_hashes = {
        str(m2._resolved_image_path(row, inference_config)): sha256_file(
            m2._resolved_image_path(row, inference_config)
        )
        for row in rows.to_dict("records")
    }
    metadata = {
        "source_path": str(source),
        "source_file_sha256": sha256_file(source),
        "source_projection_columns": list(SOURCE_COLUMNS),
        "identity_snapshot_id": manifest["snapshot_id"],
        "identity": fold4._identity_summary(rows),
        "identity_file_sha256": {
            str(path): sha256_file(path)
            for path in (reference_path, identity_path, active_path)
        },
        "raw_image_file_sha256": image_hashes,
        "raw_image_fingerprint_sha256": canonical_sha256(image_hashes),
        "native_geo_float32_sha256": hashlib.sha256(geo.tobytes(order="C")).hexdigest(),
        "test_labels_read": False,
        "cached_image_embeddings_read": False,
        "development_test_rows_and_plots_disjoint": True,
    }
    return rows, geo, metadata


def validate_split_payload(payload: Mapping[str, Any]) -> None:
    """Require the existing label-blind tensor contract and fixed producer 3."""

    fold4.validate_split_payload(payload, "validation")


def validate_seed_payload(
    result: Mapping[str, Any],
    seed: int,
    *,
    context: Any | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    source_path: str | Path = DEFAULT_TEST_SOURCE,
) -> dict[str, Any]:
    """Authenticate producer and inputs; a caller must also seal payload bytes."""

    training_seed = m2.validate_training_seed(seed)
    if set(result) != {"test", "metadata"}:
        raise ExtractionError("expected test and metadata only")
    if context is None:
        context = m2.load_frozen_context(config_path=config_path)
    rows, _, source_metadata = load_test_sources(context=context, source_path=source_path)
    spec, fit_rows, partitions = fold4._partitions(context, training_seed)
    directory = context.output_root / spec.relative_directory
    metadata = result["metadata"]
    if (metadata.get("training_seed") != training_seed
            or metadata.get("producer_fold") != PRODUCER_FOLD
            or metadata.get("same_expert_bundle_as_fold4_study") is not True
            or metadata.get("test_labels_in_payload") is not False
            or metadata.get("expert_updates") is not False):
        raise ExtractionError("invalid fixed-expert cleaned-test metadata")
    if metadata.get("test_sources") != source_metadata:
        raise ExtractionError("cleaned-test source provenance differs from saved metadata")
    for filename in fold4.CHECKPOINT_NAMES:
        if metadata.get("checkpoint_file_sha256", {}).get(filename) != sha256_file(directory / filename):
            raise ExtractionError(f"fixed-producer checkpoint hash mismatch: {filename}")
    if metadata.get("scaler_file_sha256") != sha256_file(directory / "geo_standardization.json"):
        raise ExtractionError("fixed-producer scaler hash mismatch")
    if metadata.get("expert_fit") != fold4._identity_summary(fit_rows):
        raise ExtractionError("expert fit membership metadata mismatch")
    if metadata.get("confidence_router_fit") != fold4._identity_summary(partitions["train"]):
        raise ExtractionError("confidence/router fold-four membership metadata mismatch")
    validate_split_payload(result["test"])
    for key in ("row_uid", "plot_idx"):
        if result["test"][key].tolist() != rows[key].tolist():
            raise ExtractionError(f"cleaned-test {key} differs from locked identity")
    return {
        "valid": True,
        "training_seed": training_seed,
        "producer_fold": PRODUCER_FOLD,
        "same_expert_bundle_as_fold4_study": True,
        "test_labels_in_payload": False,
        "expert_updates": False,
        "identity_snapshot_id": source_metadata["identity_snapshot_id"],
        "test_identity": source_metadata["identity"],
    }


def extract_seed(
    seed: int,
    device: str = "cuda",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    source_path: str | Path = DEFAULT_TEST_SOURCE,
) -> dict[str, Any]:
    """Replay one unchanged expert bundle on canonical cleaned-test inputs."""

    from safetensors.torch import load_file

    training_seed = m2.validate_training_seed(seed)
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ExtractionError("frozen replay supports CPU or CUDA devices")
    context = m2.load_frozen_context(config_path=config_path)
    rows, native_geo, source_metadata = load_test_sources(context=context, source_path=source_path)
    spec, fit_rows, partitions = fold4._partitions(context, training_seed)
    producer = m2.validate_producer(context, spec, seed=training_seed)
    directory = context.output_root / spec.relative_directory
    if int(context.config["experts"]["head"]["hidden_dim"]) != fold4.HIDDEN_DIM:
        raise ExtractionError("frozen expert hidden dimension is not 256")
    scaler = m2._validate_scaler_file(directory / "geo_standardization.json")
    encoder: nn.Module | None = None
    heads: dict[str, MLPHead] = {}
    try:
        m2.reset_reproducibility(training_seed)
        encoder, _, transform, _, initialization = m2._load_fresh_encoder(context.config, target_device)
        encoder.visual.load_state_dict(
            load_file(str(directory / "adapted_visual_tower.safetensors"), device="cpu"), strict=True
        )
        encoder.eval()
        encoder.requires_grad_(False)
        for mode in m2.MODES:
            m2.reset_reproducibility(training_seed)
            head = MLPHead(
                input_dim=int(context.config["experts"]["mode_inputs"][mode]["input_dim"]),
                hidden_dim=fold4.HIDDEN_DIM,
                output_dim=m2.N_CLASSES,
                dropout=float(context.config["experts"]["head"]["dropout"]),
            ).to(device=target_device, dtype=torch.float32)
            head.load_state_dict(load_file(str(directory / f"{mode}_head.safetensors"), device="cpu"), strict=True)
            head.eval()
            head.requires_grad_(False)
            heads[mode] = head
        print(f"ConfidNet fold4 seed {training_seed}: extracting cleaned test using producer fold 3 ({len(rows)} images)", flush=True)
        images = m2._extract_embeddings(
            encoder, rows, transform=transform, config=_inference_config(context.config),
            seed=training_seed, device=target_device,
        )
        geo = m2.apply_geo_standardization(native_geo, scaler)
        payload = {
            "row_uid": rows["row_uid"].to_numpy(dtype=str, copy=True),
            "plot_idx": rows["plot_idx"].to_numpy(dtype=str, copy=True),
            "producer_fold": np.full(len(rows), PRODUCER_FOLD, dtype=np.int64),
        }
        for mode, prefix in m2.MODE_TO_PREFIX.items():
            hidden, logits = fold4.initial._head_hidden_and_logits(
                heads[mode], m2._head_input(mode, images, geo), target_device
            )
            payload[f"{prefix}_hidden"] = hidden
            payload[f"{prefix}_logits"] = logits
        validate_split_payload(payload)
        result = {
            "test": payload,
            "metadata": {
                "schema_version": "geo_helpfulness.confidnet_fold4_test_extraction.v1",
                "training_seed": training_seed,
                "producer_fold": PRODUCER_FOLD,
                "user_fold_number": PRODUCER_FOLD + 1,
                "device": str(target_device),
                "protocol_id": str(context.config["protocol_id"]),
                "producer": producer,
                "parent_artifact_hashes": context.parent_hashes,
                "checkpoint_file_sha256": {
                    filename: sha256_file(directory / filename) for filename in fold4.CHECKPOINT_NAMES
                },
                "scaler_file_sha256": sha256_file(directory / "geo_standardization.json"),
                "encoder_initialization": initialization,
                "hidden_feature": "expert_head.net.1.ReLU_output",
                "hidden_dimension": fold4.HIDDEN_DIM,
                "expert_eval_mode": True,
                "expert_updates": False,
                "same_expert_bundle_as_fold4_study": True,
                "prediction_transform": "frozen_M2_prediction_transform",
                "probability_basis": "native_t1_uncalibrated",
                "test_labels_in_payload": False,
                "expert_fit": fold4._identity_summary(fit_rows),
                "confidence_router_fit": fold4._identity_summary(partitions["train"]),
                "test_sources": source_metadata,
                "activation_summaries": {
                    prefix: fold4.oof._activation_summary(payload[f"{prefix}_hidden"])
                    for prefix in PREFIXES
                },
            },
        }
        result["metadata"]["verification"] = validate_seed_payload(
            result, training_seed, context=context, source_path=source_path
        )
        return result
    finally:
        heads.clear()
        del encoder
        gc.collect()
        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

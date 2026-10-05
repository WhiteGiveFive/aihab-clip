"""Label-blind cleaned-test replay of the joint router's full-train experts.

The original joint study used the full-development-train producer at validation.
This module authenticates that same bundle and replays it on raw cleaned-test
images and geo inputs. Labels and previously cached image features never enter
the inference projection. Existing development artifacts are read-only.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from multimodal import geo_helpfulness_confidnet_fold4_test_extract as test_source
from multimodal import geo_helpfulness_joint_tcp_extract as joint
from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_confidnet import _validate_stage
from multimodal.geo_helpfulness_protocol import sha256_file
from multimodal.models import MLPHead


DEFAULT_CONFIG_PATH = m2.DEFAULT_CONFIG_PATH
DEFAULT_TEST_SOURCE = test_source.DEFAULT_TEST_SOURCE
SOURCE_COLUMNS = test_source.SOURCE_COLUMNS
PARENT_JOINT_ROOT = m2.PROJECT_ROOT / (
    "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1"
)
PRODUCER_FOLD = joint.VALIDATION_PRODUCER_FOLD
PREFIXES = joint.PREFIXES
IMAGE_DIM = joint.IMAGE_DIM
GEO_DIM = joint.GEO_DIM
CHECKPOINT_NAMES = joint.CHECKPOINT_NAMES
ExtractionError = joint.JointTCPExtractionError
load_test_sources = test_source.load_test_sources


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ExtractionError(f"expected JSON object: {path}")
    return value


def _lineage(context: Any, seed: int) -> tuple[Any, Path, dict[str, Any]]:
    """Tie the requested experts to the completed joint validation producer."""

    spec = joint._fold_spec(seed, None)
    directory = context.output_root / spec.relative_directory
    producer = m2.validate_producer(context, spec, seed=seed)
    fit_rows, validation_rows = m2._partitions_for_spec(context.assignments, spec)
    folder = PARENT_JOINT_ROOT / f"producers/seed_{seed}/validation"
    manifest = _validate_stage(folder, PARENT_JOINT_ROOT, role="joint_tcp_producer")
    if manifest["metadata"] != {"seed": seed, "fold": None}:
        raise ExtractionError("parent joint producer stage membership mismatch")
    metadata = _read_json(folder / "extraction.json")
    declaration = _read_json(PARENT_JOINT_ROOT / "declaration.json")
    expected = {
        "schema_version": "geo_helpfulness.joint_tcp_extraction_producer.v1",
        "training_seed": seed,
        "producer_fold": None,
        "producer": producer,
        "protocol_id": str(context.config["protocol_id"]),
        "parent_artifact_hashes": context.parent_hashes,
        "expert_fit": joint._identity_summary(fit_rows),
        "prediction": joint._identity_summary(validation_rows),
        "image_dimension": IMAGE_DIM,
        "geo_dimension": GEO_DIM,
        "additional_embedding_scaling": False,
        "expert_eval_mode": True,
        "probability_basis": "native_t1_uncalibrated",
        "logits_source": "authoritative_saved_M2_outputs",
        "validation_labels_in_payload": False,
    }
    for name, value in expected.items():
        if name not in metadata or metadata[name] != value:
            raise ExtractionError(f"parent joint extraction metadata mismatch: {name}")
    hashes = {name: sha256_file(directory / name) for name in CHECKPOINT_NAMES}
    scaler_hash = sha256_file(directory / "geo_standardization.json")
    reference_hash = sha256_file(directory / spec.output_filename)
    if (metadata.get("checkpoint_file_sha256") != hashes
            or metadata.get("scaler_file_sha256") != scaler_hash
            or metadata.get("reference_output_file_sha256") != reference_hash):
        raise ExtractionError("full-development-train expert hashes differ from joint study")
    sources = {**hashes, "geo_standardization.json": scaler_hash,
               spec.output_filename: reference_hash}
    for name, digest in sources.items():
        relative = str((directory / name).relative_to(m2.PROJECT_ROOT))
        if declaration.get("file_sha256", {}).get(relative) != digest:
            raise ExtractionError("parent joint declaration source hash mismatch")
    return spec, directory, {
        "producer": producer,
        "parent_artifact_hashes": context.parent_hashes,
        "checkpoint_file_sha256": hashes,
        "scaler_file_sha256": scaler_hash,
        "expert_fit": expected["expert_fit"],
        "parent_joint_validation": {
            "study_root": str(PARENT_JOINT_ROOT),
            "declaration_file_sha256": sha256_file(PARENT_JOINT_ROOT / "declaration.json"),
            "producer_manifest_file_sha256": sha256_file(folder / "manifest.json"),
            "extraction_file_sha256": sha256_file(folder / "extraction.json"),
            "reference_output_file_sha256": reference_hash,
        },
    }


def validate_split_payload(payload: Mapping[str, Any]) -> None:
    """Require the joint embedding/logit schema without any test labels."""

    joint.validate_split_payload(payload, "validation")


def validate_seed_payload(
    result: Mapping[str, Any],
    seed: int,
    *,
    context: Any | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    source_path: str | Path = DEFAULT_TEST_SOURCE,
) -> dict[str, Any]:
    """Authenticate cached identities and lineage; caller seals payload bytes."""

    training_seed = m2.validate_training_seed(seed)
    if set(result) != {"test", "metadata"}:
        raise ExtractionError("expected test and metadata only")
    if context is None:
        context = m2.load_frozen_context(config_path=config_path)
    rows, native_geo, source_metadata = load_test_sources(
        context=context, source_path=source_path,
    )
    spec, directory, lineage = _lineage(context, training_seed)
    fit_rows, _ = m2._partitions_for_spec(context.assignments, spec)
    for name in ("row_uid", "plot_idx"):
        if set(fit_rows[name]) & set(rows[name]):
            raise ExtractionError(f"expert fit and cleaned-test {name} overlap")
    metadata = result["metadata"]
    expected = {
        "schema_version": "geo_helpfulness.joint_tcp_test_extraction.v1",
        "training_seed": training_seed,
        "producer_fold": None,
        "protocol_id": str(context.config["protocol_id"]),
        "same_expert_bundle_as_joint_validation": True,
        "test_labels_in_payload": False,
        "expert_updates": False,
        "expert_eval_mode": True,
        "image_feature": "l2_normalized_adapted_image_embedding",
        "image_dimension": IMAGE_DIM,
        "geo_feature": "producer_fit_partition_standardized_gse_100m",
        "geo_dimension": GEO_DIM,
        "additional_embedding_scaling": False,
        "prediction_transform": "frozen_M2_prediction_transform",
        "probability_basis": "native_t1_uncalibrated",
        "logits_source": "native_frozen_full_development_train_expert_inference",
        "expert_fit_prediction_plots_disjoint": True,
        "test_sources": source_metadata,
        **lineage,
    }
    for name, value in expected.items():
        if name not in metadata or metadata[name] != value:
            raise ExtractionError(f"cleaned-test extraction metadata mismatch: {name}")
    validate_split_payload(result["test"])
    for name in ("row_uid", "plot_idx"):
        if result["test"][name].tolist() != rows[name].tolist():
            raise ExtractionError(f"cleaned-test {name} differs from locked identity")
    scaler = m2._validate_scaler_file(directory / "geo_standardization.json")
    expected_geo = m2.apply_geo_standardization(native_geo, scaler)
    if not np.array_equal(result["test"]["geo_embedding"], expected_geo):
        raise ExtractionError("geo embedding differs from paired producer standardization")
    return {
        "valid": True,
        "training_seed": training_seed,
        "producer_fold": None,
        "payload_producer_fold": PRODUCER_FOLD,
        "same_expert_bundle_as_joint_validation": True,
        "test_labels_in_payload": False,
        "expert_updates": False,
        "identity_snapshot_id": source_metadata["identity_snapshot_id"],
        "test_identity": source_metadata["identity"],
        "parent_joint_validation": lineage["parent_joint_validation"],
    }


def extract_seed(
    seed: int,
    device: str = "cuda",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    source_path: str | Path = DEFAULT_TEST_SOURCE,
) -> dict[str, Any]:
    """Replay one unchanged full-train bundle; return arrays without writing."""

    from safetensors.torch import load_file

    training_seed = m2.validate_training_seed(seed)
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ExtractionError("frozen replay supports CPU or CUDA devices")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ExtractionError("requested CUDA replay is unavailable")
    context = m2.load_frozen_context(config_path=config_path)
    rows, native_geo, source_metadata = load_test_sources(
        context=context, source_path=source_path,
    )
    _, directory, lineage = _lineage(context, training_seed)
    widths = {"image_only": IMAGE_DIM, "geo_only": GEO_DIM,
              "raw_concat": IMAGE_DIM + GEO_DIM}
    for mode, width in widths.items():
        if int(context.config["experts"]["mode_inputs"][mode]["input_dim"]) != width:
            raise ExtractionError(f"frozen expert input dimension mismatch: {mode}")
    if int(context.config["experts"]["head"]["hidden_dim"]) != joint.initial.HIDDEN_DIM:
        raise ExtractionError("frozen expert hidden dimension is not 256")
    scaler = m2._validate_scaler_file(directory / "geo_standardization.json")
    encoder: nn.Module | None = None
    heads: dict[str, MLPHead] = {}
    try:
        m2.reset_reproducibility(training_seed)
        encoder, _, transform, _, initialization = m2._load_fresh_encoder(
            context.config, target_device,
        )
        encoder.visual.load_state_dict(load_file(
            str(directory / "adapted_visual_tower.safetensors"), device="cpu",
        ), strict=True)
        encoder.eval()
        encoder.requires_grad_(False)
        for mode in m2.MODES:
            m2.reset_reproducibility(training_seed)
            head = MLPHead(
                input_dim=widths[mode], hidden_dim=joint.initial.HIDDEN_DIM,
                output_dim=m2.N_CLASSES,
                dropout=float(context.config["experts"]["head"]["dropout"]),
            ).to(device=target_device, dtype=torch.float32)
            head.load_state_dict(load_file(
                str(directory / f"{mode}_head.safetensors"), device="cpu",
            ), strict=True)
            head.eval()
            head.requires_grad_(False)
            heads[mode] = head
        print(
            f"Joint TCP seed {training_seed}: extracting cleaned test with full-development-train "
            f"experts ({len(rows)} images, {target_device})", flush=True,
        )
        images = m2._extract_embeddings(
            encoder, rows, transform=transform, config=test_source._inference_config(context.config),
            seed=training_seed, device=target_device,
        )
        geo = m2.apply_geo_standardization(native_geo, scaler)
        payload = {
            "row_uid": rows["row_uid"].to_numpy(dtype=str, copy=True),
            "plot_idx": rows["plot_idx"].to_numpy(dtype=str, copy=True),
            "producer_fold": np.full(len(rows), PRODUCER_FOLD, dtype=np.int64),
            "image_embedding": images.copy(),
            "geo_embedding": geo.copy(),
        }
        for mode, prefix in m2.MODE_TO_PREFIX.items():
            payload[f"{prefix}_logits"] = m2._predict_head(
                heads[mode], m2._head_input(mode, images, geo), target_device,
            )
        validate_split_payload(payload)
        result = {
            "test": payload,
            "metadata": {
                "schema_version": "geo_helpfulness.joint_tcp_test_extraction.v1",
                "training_seed": training_seed,
                "producer_fold": None,
                "device": str(target_device),
                "protocol_id": str(context.config["protocol_id"]),
                **lineage,
                "encoder_initialization": initialization,
                "image_feature": "l2_normalized_adapted_image_embedding",
                "image_dimension": IMAGE_DIM,
                "geo_feature": "producer_fit_partition_standardized_gse_100m",
                "geo_dimension": GEO_DIM,
                "additional_embedding_scaling": False,
                "expert_eval_mode": True,
                "expert_updates": False,
                "same_expert_bundle_as_joint_validation": True,
                "prediction_transform": "frozen_M2_prediction_transform",
                "probability_basis": "native_t1_uncalibrated",
                "logits_source": "native_frozen_full_development_train_expert_inference",
                "test_labels_in_payload": False,
                "expert_fit_prediction_plots_disjoint": True,
                "test_sources": source_metadata,
                "embedding_summaries": {
                    "image": joint.confidence_oof._activation_summary(images),
                    "geo": joint.confidence_oof._activation_summary(geo),
                },
            },
        }
        result["metadata"]["verification"] = validate_seed_payload(
            result, training_seed, context=context, source_path=source_path,
        )
        return result
    finally:
        heads.clear()
        del encoder
        gc.collect()
        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

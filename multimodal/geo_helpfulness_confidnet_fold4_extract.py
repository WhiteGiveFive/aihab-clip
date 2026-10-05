"""Replay one fixed fold-four expert bundle on confidence-train and validation.

The expert producer has the existing zero-based fold ID 3. Its fit set is the
other three development-training folds. Both output splits use this exact
saved encoder, geo scaler, and three heads; no expert is fitted here.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from multimodal import geo_helpfulness_confidnet_extract as initial
from multimodal import geo_helpfulness_confidnet_oof_extract as oof
from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_protocol import canonical_sha256, sha256_file
from multimodal.models import MLPHead


DEFAULT_CONFIG_PATH = m2.DEFAULT_CONFIG_PATH
HIDDEN_DIM = initial.HIDDEN_DIM
PREFIXES = initial.PREFIXES
PRODUCER_FOLD = 3
ConfidNetFold4ExtractionError = initial.ConfidNetExtractionError
CHECKPOINT_NAMES = (
    "adapted_visual_tower.safetensors",
    *(f"{mode}_head.safetensors" for mode in m2.MODES),
)


def validate_split_payload(payload: Mapping[str, Any], split: str) -> None:
    """Require label-blind validation and the same fold-3 producer on both sets."""

    if "producer_fold" not in payload:
        raise ConfidNetFold4ExtractionError("missing producer_fold provenance")
    initial.validate_split_payload(
        {key: value for key, value in payload.items() if key != "producer_fold"}, split
    )
    folds = np.asarray(payload["producer_fold"])
    if folds.dtype != np.int64 or folds.shape != np.asarray(payload["row_uid"]).shape:
        raise ConfidNetFold4ExtractionError("producer_fold must be a row-aligned int64 array")
    if not np.all(folds == PRODUCER_FOLD):
        raise ConfidNetFold4ExtractionError("both splits must use frozen producer fold 3")


def _partitions(context: Any, seed: int) -> tuple[m2.ProducerSpec, pd.DataFrame, dict]:
    spec = oof._fold_spec(seed, PRODUCER_FOLD)
    fit_rows, train_rows = m2._partitions_for_spec(context.assignments, spec)
    # This supplies validation membership only, never full-train expert weights.
    _, validation_rows = m2._partitions_for_spec(
        context.assignments, oof._fold_spec(seed, None)
    )
    partitions = {"train": train_rows, "validation": validation_rows}
    groups = {"expert_fit": fit_rows, **partitions}
    for name, rows in groups.items():
        initial._prediction_projection(rows)
        for other_name, other in groups.items():
            if other_name <= name:
                continue
            for identity in ("row_uid", "plot_idx"):
                if set(rows[identity]) & set(other[identity]):
                    raise ConfidNetFold4ExtractionError(
                        f"{name} and {other_name} overlap by {identity}"
                    )
    return spec, fit_rows, partitions


def _identity_summary(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "plot_count": int(rows["plot_idx"].nunique()),
        "identity_sha256": canonical_sha256(
            rows.loc[:, ["row_uid", "plot_idx"]].to_dict("records")
        ),
    }


def validate_seed_payload(
    result: Mapping[str, Any],
    seed: int,
    *,
    context: Any | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Check cached identities, supervision, producer hashes, and held-out replay.

    An artifact manifest must separately authenticate saved payload file bytes.
    New validation logits have no prior reference because this expert bundle
    previously predicted only its held-out training fold.
    """

    training_seed = m2.validate_training_seed(seed)
    if set(result) != {"train", "validation", "metadata"}:
        raise ConfidNetFold4ExtractionError("expected train, validation, and metadata")
    if context is None:
        context = m2.load_frozen_context(config_path=config_path)
    spec, fit_rows, partitions = _partitions(context, training_seed)
    directory = context.output_root / spec.relative_directory
    metadata = result["metadata"]
    if (metadata.get("training_seed") != training_seed
            or metadata.get("producer_fold") != PRODUCER_FOLD
            or metadata.get("same_expert_bundle_for_both_splits") is not True):
        raise ConfidNetFold4ExtractionError("wrong seed or fixed-producer metadata")
    if (metadata.get("validation_labels_in_payload") is not False
            or metadata.get("locked_test_accessed") is not False):
        raise ConfidNetFold4ExtractionError("invalid label/test isolation metadata")
    for filename in CHECKPOINT_NAMES:
        if metadata.get("checkpoint_file_sha256", {}).get(filename) != sha256_file(
            directory / filename
        ):
            raise ConfidNetFold4ExtractionError(f"fixed-producer checkpoint hash mismatch: {filename}")
    for field, filename in (
        ("scaler_file_sha256", "geo_standardization.json"),
        ("heldout_reference_file_sha256", spec.output_filename),
    ):
        if metadata.get(field) != sha256_file(directory / filename):
            raise ConfidNetFold4ExtractionError(f"fixed-producer {field} mismatch")
    if metadata.get("expert_fit") != _identity_summary(fit_rows):
        raise ConfidNetFold4ExtractionError("expert fit membership metadata mismatch")
    for split, rows in partitions.items():
        payload = result[split]
        validate_split_payload(payload, split)
        for identity in ("row_uid", "plot_idx"):
            if payload[identity].tolist() != rows[identity].tolist():
                raise ConfidNetFold4ExtractionError(f"{split} {identity} differs from M1")
        if metadata.get("splits", {}).get(split) != _identity_summary(rows):
            raise ConfidNetFold4ExtractionError(f"{split} membership metadata mismatch")
    if not np.array_equal(
        result["train"]["labels"], partitions["train"]["label_id_dense"].to_numpy(dtype=np.int64)
    ):
        raise ConfidNetFold4ExtractionError("confidence training labels differ from M1")
    reference = m2.read_output_parquet(
        directory / spec.output_filename, include_fold=True
    ).to_pandas()
    reproduction = oof._reproduction(
        result["train"], initial._prediction_projection(partitions["train"]), reference,
        fold=PRODUCER_FOLD, seed=training_seed,
    )
    return {
        "valid": True,
        "training_seed": training_seed,
        "producer_fold": PRODUCER_FOLD,
        "heldout_reproduction": reproduction,
        "same_expert_bundle_for_both_splits": True,
        "expert_fit_train_validation_plots_disjoint": True,
        "validation_labels_in_payload": False,
        "locked_test_accessed": False,
    }


def extract_seed(
    seed: int,
    device: str = "cuda",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Return fold-four supervision and validation features from one saved bundle.

    ``train`` contains labels; ``validation`` contains only identities, producer
    fold 3, float32 hidden features and float64 stored native-float32 logits.
    This function never writes artifacts or retrains an expert.
    """

    from safetensors.torch import load_file

    training_seed = m2.validate_training_seed(seed)
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ConfidNetFold4ExtractionError("frozen replay supports CPU or CUDA devices")
    context = m2.load_frozen_context(config_path=config_path)
    spec, fit_rows, partitions = _partitions(context, training_seed)
    producer = m2.validate_producer(context, spec, seed=training_seed)
    directory = context.output_root / spec.relative_directory
    projected = {split: initial._prediction_projection(rows) for split, rows in partitions.items()}
    if int(context.config["experts"]["head"]["hidden_dim"]) != HIDDEN_DIM:
        raise ConfidNetFold4ExtractionError("frozen expert hidden dimension is not 256")
    scaler = m2._validate_scaler_file(directory / "geo_standardization.json")
    result: dict[str, Any] = {}
    encoder: nn.Module | None = None
    heads: dict[str, MLPHead] = {}
    try:
        m2.reset_reproducibility(training_seed)
        encoder, _, transform, _, initialization = m2._load_fresh_encoder(
            context.config, target_device
        )
        encoder.visual.load_state_dict(
            load_file(str(directory / "adapted_visual_tower.safetensors"), device="cpu"),
            strict=True,
        )
        encoder.eval()
        encoder.requires_grad_(False)
        for mode in m2.MODES:
            m2.reset_reproducibility(training_seed)
            head = MLPHead(
                input_dim=int(context.config["experts"]["mode_inputs"][mode]["input_dim"]),
                hidden_dim=HIDDEN_DIM,
                output_dim=m2.N_CLASSES,
                dropout=float(context.config["experts"]["head"]["dropout"]),
            ).to(device=target_device, dtype=torch.float32)
            head.load_state_dict(
                load_file(str(directory / f"{mode}_head.safetensors"), device="cpu"), strict=True
            )
            head.eval()
            head.requires_grad_(False)
            heads[mode] = head
        for split, rows in projected.items():
            print(
                f"ConfidNet fold4 seed {training_seed}: extracting {split} "
                f"using producer fold 3 ({len(rows)} images)", flush=True,
            )
            images = m2._extract_embeddings(
                encoder, rows, transform=transform, config=context.config,
                seed=training_seed, device=target_device,
            )
            geo = m2.apply_geo_standardization(m2._geo_source_projection(context, rows), scaler)
            payload = {
                "row_uid": rows["row_uid"].to_numpy(dtype=str, copy=True),
                "plot_idx": rows["plot_idx"].to_numpy(dtype=str, copy=True),
                "producer_fold": np.full(len(rows), PRODUCER_FOLD, dtype=np.int64),
            }
            if split == "train":
                payload["labels"] = partitions[split]["label_id_dense"].to_numpy(
                    dtype=np.int64, copy=True
                )
            for mode, prefix in m2.MODE_TO_PREFIX.items():
                hidden, logits = initial._head_hidden_and_logits(
                    heads[mode], m2._head_input(mode, images, geo), target_device
                )
                payload[f"{prefix}_hidden"] = hidden
                payload[f"{prefix}_logits"] = logits
            validate_split_payload(payload, split)
            result[split] = payload
        result["metadata"] = {
            "schema_version": "geo_helpfulness.confidnet_fold4_extraction.v1",
            "training_seed": training_seed,
            "producer_fold": PRODUCER_FOLD,
            "user_fold_number": PRODUCER_FOLD + 1,
            "device": str(target_device),
            "protocol_id": str(context.config["protocol_id"]),
            "producer": producer,
            "parent_artifact_hashes": context.parent_hashes,
            "checkpoint_file_sha256": {
                filename: sha256_file(directory / filename) for filename in CHECKPOINT_NAMES
            },
            "scaler_file_sha256": sha256_file(directory / "geo_standardization.json"),
            "heldout_reference_file_sha256": sha256_file(directory / spec.output_filename),
            "encoder_initialization": initialization,
            "hidden_feature": "expert_head.net.1.ReLU_output",
            "hidden_dimension": HIDDEN_DIM,
            "expert_eval_mode": True,
            "same_expert_bundle_for_both_splits": True,
            "prediction_transform": "frozen_M2_prediction_transform_on_both_splits",
            "probability_basis": "native_t1_uncalibrated",
            "supervision": "expert_heldout_fold4_native_tcp",
            "validation_labels_in_payload": False,
            "locked_test_accessed": False,
            "expert_fit_train_validation_plots_disjoint": True,
            "expert_fit": _identity_summary(fit_rows),
            "splits": {split: _identity_summary(rows) for split, rows in partitions.items()},
            "activation_summaries": {
                split: {
                    prefix: oof._activation_summary(result[split][f"{prefix}_hidden"])
                    for prefix in PREFIXES
                }
                for split in projected
            },
        }
        result["metadata"]["verification"] = validate_seed_payload(
            result, training_seed, context=context
        )
        return result
    finally:
        heads.clear()
        del encoder
        gc.collect()
        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

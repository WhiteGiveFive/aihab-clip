"""Read-only replay of full-development-train experts for ConfidNet.

Inference uses the frozen M2 prediction transform, saved geo scaler and saved
expert heads.  Validation labels never cross the feature-extraction boundary.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_protocol import canonical_sha256, sha256_file
from multimodal.models import MLPHead


DEFAULT_CONFIG_PATH = m2.DEFAULT_CONFIG_PATH
HIDDEN_DIM = 256
IDENTITY_COLUMNS = ("row_uid", "file", "file_lower", "plot_idx", "image_source")
PREFIXES = tuple(m2.MODE_TO_PREFIX.values())


class ConfidNetExtractionError(ValueError):
    """A checkpoint, identity, or label-isolation contract was violated."""


def _prediction_projection(rows: pd.DataFrame) -> pd.DataFrame:
    """Select only the identities needed for image and geo inference."""

    missing = set(IDENTITY_COLUMNS).difference(rows.columns)
    if missing:
        raise ConfidNetExtractionError(f"missing inference identities: {sorted(missing)}")
    result = rows.loc[:, list(IDENTITY_COLUMNS)].copy()
    if result.empty or result.isna().any().any():
        raise ConfidNetExtractionError("inference identities must be nonempty and non-null")
    for column in IDENTITY_COLUMNS:
        if not result[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ConfidNetExtractionError(f"{column} must contain nonempty string identities")
    for column in ("row_uid", "file_lower"):
        if result[column].duplicated().any():
            raise ConfidNetExtractionError(f"duplicate inference identity: {column}")
    if result["row_uid"].tolist() != sorted(result["row_uid"].tolist()):
        raise ConfidNetExtractionError("inference row identities must be canonically sorted")
    return result.reset_index(drop=True)


def validate_split_payload(payload: Mapping[str, Any], split: str) -> None:
    """Validate an extracted or cached payload, including its label boundary."""

    if split not in {"train", "validation"}:
        raise ConfidNetExtractionError("only development train/validation are supported")
    expected = {"row_uid", "plot_idx"}
    expected.update(
        f"{prefix}_{suffix}" for prefix in PREFIXES for suffix in ("hidden", "logits")
    )
    if split == "train":
        expected.add("labels")
    if set(payload) != expected:
        raise ConfidNetExtractionError(
            f"{split} payload keys violate the feature/label allow-list: "
            f"missing={sorted(expected.difference(payload))}, "
            f"unexpected={sorted(set(payload).difference(expected))}"
        )
    identities = np.asarray(payload["row_uid"])
    plots = np.asarray(payload["plot_idx"])
    if identities.ndim != 1 or len(identities) == 0 or plots.shape != identities.shape:
        raise ConfidNetExtractionError("row/plot identity shapes are invalid")
    for values in (identities, plots):
        if values.dtype.kind not in {"U", "S"} or np.any(values.astype(str) == ""):
            raise ConfidNetExtractionError("identity arrays must contain nonempty strings")
    if len(set(identities.tolist())) != len(identities):
        raise ConfidNetExtractionError("duplicate payload row identities")
    if identities.tolist() != sorted(identities.tolist()):
        raise ConfidNetExtractionError("payload row identities must be canonically sorted")
    array_schema = (
        ("hidden", HIDDEN_DIM, np.float32),
        ("logits", m2.N_CLASSES, np.float64),
    )
    for prefix in PREFIXES:
        for suffix, width, dtype in array_schema:
            values = np.asarray(payload[f"{prefix}_{suffix}"])
            if values.shape != (len(identities), width) or values.dtype != dtype:
                raise ConfidNetExtractionError(f"invalid {prefix}_{suffix} shape or dtype")
            if not np.isfinite(values).all():
                raise ConfidNetExtractionError(f"nonfinite {prefix}_{suffix}")
    if split == "train":
        labels = np.asarray(payload["labels"])
        if labels.shape != identities.shape or labels.dtype.kind not in {"i", "u"}:
            raise ConfidNetExtractionError(
                "training labels must be a one-dimensional integer array"
            )
        if np.any(labels < 0) or np.any(labels >= m2.N_CLASSES):
            raise ConfidNetExtractionError("training labels must be dense class IDs in [0, 17]")


def _head_hidden_and_logits(
    head: MLPHead, features: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Capture ReLU features without changing the M2 head's forward path."""

    if not isinstance(head.net, nn.Sequential) or len(head.net) != 4:
        raise ConfidNetExtractionError("expected the frozen four-layer M2 MLP head")
    if not isinstance(head.net[1], nn.ReLU) or head.net[0].out_features != HIDDEN_DIM:
        raise ConfidNetExtractionError("expected a 256-dimensional M2 head ReLU")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ConfidNetExtractionError("head input must be a finite nonempty matrix")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(matrix)),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    hidden_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []

    def capture_hidden(_module: nn.Module, _args: Any, output: torch.Tensor) -> None:
        hidden_batches.append(output.detach().float().cpu().numpy().copy())

    head.eval()
    head.requires_grad_(False)
    hook = head.net[1].register_forward_hook(capture_hidden)
    try:
        with torch.no_grad():
            for (batch,) in loader:
                logits = head(batch.to(device, non_blocking=True))
                logits_batches.append(logits.float().cpu().numpy().astype(np.float64))
    finally:
        hook.remove()
    hidden = np.concatenate(hidden_batches, axis=0).astype(np.float32, copy=False)
    logits = np.concatenate(logits_batches, axis=0)
    if hidden.shape != (len(matrix), HIDDEN_DIM) or logits.shape != (len(matrix), m2.N_CLASSES):
        raise ConfidNetExtractionError("head emitted unexpected hidden/logit shapes")
    if not np.isfinite(hidden).all() or not np.isfinite(logits).all():
        raise ConfidNetExtractionError("head emitted nonfinite hidden features/logits")
    return hidden, logits


def _validate_validation_reproduction(
    payload: Mapping[str, Any], rows: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, Any]:
    validate_split_payload(payload, "validation")
    if "labels" in reference or any(column.startswith("label") for column in reference):
        raise ConfidNetExtractionError("saved inference reference must be label-blind")
    if len(reference) != len(rows) or len(rows) != len(payload["row_uid"]):
        raise ConfidNetExtractionError("validation replay row count mismatch")
    for column in ("row_uid", "file", "file_lower", "plot_idx"):
        if column not in reference or reference[column].tolist() != rows[column].tolist():
            raise ConfidNetExtractionError(f"validation replay identity mismatch: {column}")
    for column in ("row_uid", "plot_idx"):
        if payload[column].tolist() != rows[column].tolist():
            raise ConfidNetExtractionError(f"validation payload identity mismatch: {column}")
    maximum_errors: dict[str, float] = {}
    for prefix in PREFIXES:
        observed = payload[f"{prefix}_logits"]
        expected = np.asarray(reference[f"{prefix}_logits"].tolist(), dtype=np.float64)
        if expected.shape != observed.shape or not np.isfinite(expected).all():
            raise ConfidNetExtractionError(f"invalid saved validation logits: {prefix}")
        if not np.allclose(observed, expected, atol=1e-6, rtol=1e-6):
            raise ConfidNetExtractionError(f"validation logits do not reproduce M2: {prefix}")
        predictions = observed.argmax(axis=1)
        if not np.array_equal(predictions, expected.argmax(axis=1)) or not np.array_equal(
            predictions, reference[f"{prefix}_pred"].to_numpy()
        ):
            raise ConfidNetExtractionError(f"validation argmax does not reproduce M2: {prefix}")
        maximum_errors[prefix] = float(np.max(np.abs(observed - expected)))
    return {
        "valid": True,
        "row_count": len(rows),
        "atol": 1e-6,
        "rtol": 1e-6,
        "argmax_exact": True,
        "maximum_absolute_logit_error": maximum_errors,
    }


def extract_seed(
    seed: int,
    device: str = "cuda",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Replay one full-development-train expert set on train and validation.

    Returns float32 ``{image,geo,raw}_hidden`` matrices of width 256 and float64
    ``{image,geo,raw}_logits`` matrices of width 18.  Only ``train`` has labels.
    No artifact is written and no locked-test source is opened.
    """

    from safetensors.torch import load_file

    training_seed = m2.validate_training_seed(seed)
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ConfidNetExtractionError("frozen replay supports CPU or CUDA devices")
    context = m2.load_frozen_context(config_path=config_path)
    spec = m2._producer_specs(training_seed)[-1]
    producer = m2.validate_producer(context, spec, seed=training_seed)
    directory = context.output_root / spec.relative_directory
    fit_rows, validation_rows = m2._partitions_for_spec(context.assignments, spec)
    projected = {
        "train": _prediction_projection(fit_rows),
        "validation": _prediction_projection(validation_rows),
    }
    if set(projected["train"]["row_uid"]) & set(projected["validation"]["row_uid"]):
        raise ConfidNetExtractionError("train and validation row identities overlap")
    if set(projected["train"]["plot_idx"]) & set(projected["validation"]["plot_idx"]):
        raise ConfidNetExtractionError("train and validation plots overlap")
    if int(context.config["experts"]["head"]["hidden_dim"]) != HIDDEN_DIM:
        raise ConfidNetExtractionError("frozen expert hidden dimension is not 256")
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
                load_file(str(directory / f"{mode}_head.safetensors"), device="cpu"),
                strict=True,
            )
            head.eval()
            head.requires_grad_(False)
            heads[mode] = head
        for split, rows in projected.items():
            print(
                f"ConfidNet seed {training_seed}: extracting {split} ({len(rows)} images)",
                flush=True,
            )
            images = m2._extract_embeddings(
                encoder, rows, transform=transform, config=context.config,
                seed=training_seed, device=target_device,
            )
            geo = m2.apply_geo_standardization(m2._geo_source_projection(context, rows), scaler)
            payload = {
                "row_uid": rows["row_uid"].to_numpy(dtype=str, copy=True),
                "plot_idx": rows["plot_idx"].to_numpy(dtype=str, copy=True),
            }
            if split == "train":
                payload["labels"] = fit_rows["label_id_dense"].to_numpy(dtype=np.int64, copy=True)
            for mode, prefix in m2.MODE_TO_PREFIX.items():
                hidden, logits = _head_hidden_and_logits(
                    heads[mode], m2._head_input(mode, images, geo), target_device
                )
                payload[f"{prefix}_hidden"] = hidden
                payload[f"{prefix}_logits"] = logits
            validate_split_payload(payload, split)
            result[split] = payload
            print(f"ConfidNet seed {training_seed}: {split} extraction complete", flush=True)
        reference = m2.read_output_parquet(
            directory / spec.output_filename, include_fold=False
        ).to_pandas()
        reproduction = _validate_validation_reproduction(
            result["validation"], projected["validation"], reference
        )
        checkpoint_names = (
            "adapted_visual_tower.safetensors",
            *(f"{mode}_head.safetensors" for mode in m2.MODES),
        )
        result["metadata"] = {
            "schema_version": "geo_helpfulness.confidnet_extraction.v1",
            "training_seed": training_seed,
            "device": str(target_device),
            "protocol_id": str(context.config["protocol_id"]),
            "producer": producer,
            "parent_artifact_hashes": context.parent_hashes,
            "checkpoint_file_sha256": {
                filename: sha256_file(directory / filename)
                for filename in checkpoint_names
            },
            "scaler_file_sha256": sha256_file(directory / "geo_standardization.json"),
            "validation_output_file_sha256": sha256_file(directory / spec.output_filename),
            "encoder_initialization": initialization,
            "hidden_feature": "expert_head.net.1.ReLU_output",
            "hidden_dimension": HIDDEN_DIM,
            "expert_eval_mode": True,
            "prediction_transform": "frozen_M2_prediction_transform_on_both_splits",
            "probability_basis": "native_t1_uncalibrated",
            "validation_labels_in_payload": False,
            "locked_test_accessed": False,
            "validation_reproduction": reproduction,
            "splits": {
                split: {
                    "row_count": len(rows),
                    "plot_count": int(rows["plot_idx"].nunique()),
                    "identity_sha256": canonical_sha256(
                        rows.loc[:, ["row_uid", "plot_idx"]].to_dict("records")
                    ),
                }
                for split, rows in projected.items()
            },
        }
        return result
    finally:
        heads.clear()
        del encoder
        gc.collect()
        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

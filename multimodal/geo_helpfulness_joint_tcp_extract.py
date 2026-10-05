"""Replay frozen expert inputs for the joint state/TCP router study.

Only held-out producer inputs are extracted. Validation labels never cross
the inference projection or payload boundary, and saved M2 logits remain the
authoritative TCP source even when a permitted replay difference exists.
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
from multimodal import geo_helpfulness_confidnet_oof_extract as confidence_oof
from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_protocol import canonical_sha256, sha256_file
from multimodal.models import MLPHead


DEFAULT_CONFIG_PATH = m2.DEFAULT_CONFIG_PATH
IMAGE_DIM = 1152
GEO_DIM = 64
PREFIXES = tuple(m2.MODE_TO_PREFIX.values())
VALIDATION_PRODUCER_FOLD = -1
CHECKPOINT_NAMES = (
    "adapted_visual_tower.safetensors",
    *(f"{mode}_head.safetensors" for mode in m2.MODES),
)


class JointTCPExtractionError(ValueError):
    """An input, provenance, replay, or label-isolation contract failed."""


def _fold_spec(seed: int, fold: int | None) -> m2.ProducerSpec:
    if fold is not None and (type(fold) is not int or fold not in m2.OOF_FOLDS):
        raise JointTCPExtractionError("fold must be 0, 1, 2, 3 or None for validation")
    return next(spec for spec in m2._producer_specs(seed) if spec.fold == fold)


def validate_split_payload(payload: Mapping[str, Any], split: str) -> None:
    """Check exact schemas, finite embeddings, and producer-fold membership."""

    if split not in {"train", "validation"}:
        raise JointTCPExtractionError("only development train/validation are supported")
    expected = {
        "row_uid", "plot_idx", "producer_fold", "image_embedding", "geo_embedding",
        *(f"{prefix}_logits" for prefix in PREFIXES),
    }
    if split == "train":
        expected.add("labels")
    if set(payload) != expected:
        raise JointTCPExtractionError(
            f"{split} payload violates the feature/label allow-list: "
            f"missing={sorted(expected.difference(payload))}, "
            f"unexpected={sorted(set(payload).difference(expected))}"
        )
    identities = np.asarray(payload["row_uid"])
    plots = np.asarray(payload["plot_idx"])
    if identities.ndim != 1 or not len(identities) or plots.shape != identities.shape:
        raise JointTCPExtractionError("row/plot identity shapes are invalid")
    for values in (identities, plots):
        if values.dtype.kind not in {"U", "S"} or np.any(values.astype(str) == ""):
            raise JointTCPExtractionError("identities must be nonempty string arrays")
    if len(set(identities.tolist())) != len(identities):
        raise JointTCPExtractionError("duplicate payload row identities")
    if identities.tolist() != sorted(identities.tolist()):
        raise JointTCPExtractionError("payload row identities must be canonically sorted")
    folds = np.asarray(payload["producer_fold"])
    if folds.dtype != np.int64 or folds.shape != identities.shape:
        raise JointTCPExtractionError("producer_fold must be a row-aligned int64 array")
    allowed = m2.OOF_FOLDS if split == "train" else (VALIDATION_PRODUCER_FOLD,)
    if not np.isin(folds, allowed).all():
        raise JointTCPExtractionError("producer_fold values do not match the split")
    plot_folds: dict[str, int] = {}
    for plot, fold in zip(plots, folds):
        if str(plot) in plot_folds and plot_folds[str(plot)] != int(fold):
            raise JointTCPExtractionError("one plot was assigned multiple producer folds")
        plot_folds[str(plot)] = int(fold)
    schema = [
        ("image_embedding", IMAGE_DIM, np.float32),
        ("geo_embedding", GEO_DIM, np.float32),
        *((f"{prefix}_logits", m2.N_CLASSES, np.float64) for prefix in PREFIXES),
    ]
    for key, width, dtype in schema:
        values = np.asarray(payload[key])
        if values.shape != (len(identities), width) or values.dtype != dtype:
            raise JointTCPExtractionError(f"invalid {key} shape or dtype")
        if not np.isfinite(values).all():
            raise JointTCPExtractionError(f"nonfinite {key}")
    norms = np.linalg.norm(payload["image_embedding"].astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
        raise JointTCPExtractionError("image embeddings must be L2-normalized")
    if split == "train":
        labels = np.asarray(payload["labels"])
        if labels.shape != identities.shape or labels.dtype != np.int64:
            raise JointTCPExtractionError("training labels must be a row-aligned int64 array")
        if np.any(labels < 0) or np.any(labels >= m2.N_CLASSES):
            raise JointTCPExtractionError("training labels must be dense IDs in [0, 17]")


def _reference_logits(
    rows: pd.DataFrame, reference: pd.DataFrame, *, seed: int, fold: int | None
) -> dict[str, np.ndarray]:
    if any(str(column).startswith("label") for column in reference):
        raise JointTCPExtractionError("saved inference reference must be label-blind")
    if len(reference) != len(rows):
        raise JointTCPExtractionError("saved reference row count mismatch")
    for column in ("row_uid", "file", "file_lower", "plot_idx"):
        if column not in reference or reference[column].tolist() != rows[column].tolist():
            raise JointTCPExtractionError(f"saved reference identity mismatch: {column}")
    if "training_seed" not in reference or not reference["training_seed"].eq(seed).all():
        raise JointTCPExtractionError("saved reference training seed mismatch")
    if fold is None:
        if "train_oof_fold" in reference:
            raise JointTCPExtractionError("validation reference must use full-train producer")
    elif "train_oof_fold" not in reference or not reference["train_oof_fold"].eq(fold).all():
        raise JointTCPExtractionError("saved reference producer fold mismatch")
    result: dict[str, np.ndarray] = {}
    for prefix in PREFIXES:
        values = np.asarray(reference[f"{prefix}_logits"].tolist(), dtype=np.float64)
        if values.shape != (len(rows), m2.N_CLASSES) or not np.isfinite(values).all():
            raise JointTCPExtractionError(f"invalid saved reference logits: {prefix}")
        if not np.array_equal(values.argmax(axis=1), reference[f"{prefix}_pred"].to_numpy()):
            raise JointTCPExtractionError(f"saved reference argmax mismatch: {prefix}")
        result[f"{prefix}_logits"] = values.copy()
    return result


def _identity_summary(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "plot_count": int(rows["plot_idx"].nunique()),
        "identity_sha256": canonical_sha256(
            rows.loc[:, ["row_uid", "plot_idx"]].to_dict("records")
        ),
    }


def extract_producer(
    seed: int,
    fold: int | None,
    device: str = "cpu",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Return one producer's embeddings and authoritative logits without writes.

    ``fold=None`` selects the full-development-train experts for validation;
    integer folds 0..3 select experts excluding those training plots. Encoder
    replay uses the requested device and the existing frozen FP32 recipe.
    """

    from safetensors.torch import load_file

    training_seed = m2.validate_training_seed(seed)
    spec = _fold_spec(training_seed, fold)
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise JointTCPExtractionError("frozen replay supports CPU or CUDA devices")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise JointTCPExtractionError("requested CUDA replay is unavailable")
    context = m2.load_frozen_context(config_path=config_path)
    producer = m2.validate_producer(context, spec, seed=training_seed)
    directory = context.output_root / spec.relative_directory
    fit_rows, prediction_rows = m2._partitions_for_spec(context.assignments, spec)
    rows = initial._prediction_projection(prediction_rows)
    for identity in ("row_uid", "plot_idx"):
        if set(fit_rows[identity]) & set(rows[identity]):
            raise JointTCPExtractionError(f"expert fit and prediction {identity} overlap")
    required_widths = {"image_only": IMAGE_DIM, "geo_only": GEO_DIM,
                       "raw_concat": IMAGE_DIM + GEO_DIM}
    for mode, width in required_widths.items():
        if int(context.config["experts"]["mode_inputs"][mode]["input_dim"]) != width:
            raise JointTCPExtractionError(f"frozen expert input dimension mismatch: {mode}")
    if int(context.config["experts"]["head"]["hidden_dim"]) != initial.HIDDEN_DIM:
        raise JointTCPExtractionError("frozen expert hidden dimension is not 256")
    scaler = m2._validate_scaler_file(directory / "geo_standardization.json")
    reference = m2.read_output_parquet(
        directory / spec.output_filename, include_fold=spec.include_fold
    ).to_pandas()
    saved_logits = _reference_logits(rows, reference, seed=training_seed, fold=fold)
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
                input_dim=required_widths[mode], hidden_dim=initial.HIDDEN_DIM,
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
        print(
            f"Joint TCP seed {training_seed}: extracting {spec.stage_id} "
            f"({len(rows)} images, {target_device})", flush=True,
        )
        images = m2._extract_embeddings(
            encoder, rows, transform=transform, config=context.config,
            seed=training_seed, device=target_device,
        )
        geo = m2.apply_geo_standardization(m2._geo_source_projection(context, rows), scaler)
        maximum_errors: dict[str, float] = {}
        for mode, prefix in m2.MODE_TO_PREFIX.items():
            replay = m2._predict_head(heads[mode], m2._head_input(mode, images, geo), target_device)
            expected = saved_logits[f"{prefix}_logits"]
            if not np.allclose(replay, expected, atol=1e-6, rtol=1e-6):
                raise JointTCPExtractionError(f"{prefix} logits do not reproduce M2")
            if not np.array_equal(replay.argmax(axis=1), expected.argmax(axis=1)):
                raise JointTCPExtractionError(f"{prefix} argmax does not reproduce M2")
            maximum_errors[prefix] = float(np.max(np.abs(replay - expected)))
        payload = {
            "row_uid": rows["row_uid"].to_numpy(dtype=str, copy=True),
            "plot_idx": rows["plot_idx"].to_numpy(dtype=str, copy=True),
            "producer_fold": np.full(
                len(rows), fold if fold is not None else VALIDATION_PRODUCER_FOLD,
                dtype=np.int64,
            ),
            "image_embedding": images.copy(),
            "geo_embedding": geo.copy(),
            **saved_logits,
        }
        split = "train" if fold is not None else "validation"
        if split == "train":
            payload["labels"] = prediction_rows["label_id_dense"].to_numpy(
                dtype=np.int64, copy=True
            )
        validate_split_payload(payload, split)
        metadata = {
            "schema_version": "geo_helpfulness.joint_tcp_extraction_producer.v1",
            "training_seed": training_seed,
            "producer_fold": fold,
            "device": str(target_device),
            "protocol_id": str(context.config["protocol_id"]),
            "producer": producer,
            "parent_artifact_hashes": context.parent_hashes,
            "checkpoint_file_sha256": {
                name: sha256_file(directory / name) for name in CHECKPOINT_NAMES
            },
            "scaler_file_sha256": sha256_file(directory / "geo_standardization.json"),
            "reference_output_file_sha256": sha256_file(directory / spec.output_filename),
            "encoder_initialization": initialization,
            "image_feature": "l2_normalized_adapted_image_embedding",
            "image_dimension": IMAGE_DIM,
            "geo_feature": "producer_fit_partition_standardized_gse_100m",
            "geo_dimension": GEO_DIM,
            "additional_embedding_scaling": False,
            "expert_eval_mode": True,
            "prediction_transform": "frozen_M2_prediction_transform",
            "probability_basis": "native_t1_uncalibrated",
            "logits_source": "authoritative_saved_M2_outputs",
            "validation_labels_in_payload": False,
            "locked_test_accessed": False,
            "expert_fit": _identity_summary(fit_rows),
            "prediction": _identity_summary(rows),
            "expert_fit_prediction_plots_disjoint": True,
            "reproduction": {
                "valid": True, "atol": 1e-6, "rtol": 1e-6, "argmax_exact": True,
                "maximum_absolute_logit_error": maximum_errors,
            },
            "embedding_summaries": {
                "image": confidence_oof._activation_summary(images),
                "geo": confidence_oof._activation_summary(geo),
            },
        }
        return {"payload": payload, "metadata": metadata}
    finally:
        heads.clear()
        del encoder
        gc.collect()
        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def combine_producers(
    producers: Mapping[int | None, Mapping[str, Any]],
    split: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, np.ndarray]:
    """Combine a seed's producers after rechecking membership and saved logits.

    Accept all five producers, or exactly the producers for the requested
    split. Physical cache manifests must be authenticated by the caller.
    """

    if split not in {"train", "validation"}:
        raise JointTCPExtractionError("only development train/validation are supported")
    required = set(m2.OOF_FOLDS) if split == "train" else {None}
    if set(producers) not in (required, set(m2.OOF_FOLDS) | {None}):
        raise JointTCPExtractionError("all requested producer folds are required")
    seeds = {result.get("metadata", {}).get("training_seed") for result in producers.values()}
    if len(seeds) != 1:
        raise JointTCPExtractionError("cannot combine mixed training seeds")
    seed = m2.validate_training_seed(next(iter(seeds)))
    context = m2.load_frozen_context(config_path=config_path)
    checked: list[Mapping[str, np.ndarray]] = []
    for fold in (*m2.OOF_FOLDS,) if split == "train" else (None,):
        result = producers[fold]
        if set(result) != {"payload", "metadata"}:
            raise JointTCPExtractionError("producer result must contain payload and metadata")
        payload, metadata = result["payload"], result["metadata"]
        if metadata.get("producer_fold") != fold:
            raise JointTCPExtractionError("producer fold metadata mismatch")
        validate_split_payload(payload, split)
        spec = _fold_spec(seed, fold)
        fit_rows, prediction_rows = m2._partitions_for_spec(context.assignments, spec)
        rows = initial._prediction_projection(prediction_rows)
        directory = context.output_root / spec.relative_directory
        for identity in ("row_uid", "plot_idx"):
            if set(fit_rows[identity]) & set(rows[identity]):
                raise JointTCPExtractionError(f"expert fit and prediction {identity} overlap")
            if payload[identity].tolist() != rows[identity].tolist():
                raise JointTCPExtractionError(f"cached producer {identity} differs from M1")
        if not np.all(payload["producer_fold"] == (fold if fold is not None else -1)):
            raise JointTCPExtractionError("cached producer fold values differ from M1")
        if split == "train" and not np.array_equal(
            payload["labels"], prediction_rows["label_id_dense"].to_numpy(dtype=np.int64)
        ):
            raise JointTCPExtractionError("cached training labels differ from M1")
        reference = m2.read_output_parquet(
            directory / spec.output_filename, include_fold=spec.include_fold
        ).to_pandas()
        authoritative = _reference_logits(rows, reference, seed=seed, fold=fold)
        for name, expected in authoritative.items():
            if not np.array_equal(payload[name], expected):
                raise JointTCPExtractionError(f"cached {name} differs from authoritative M2")
        checked.append(payload)
    combined = {key: np.concatenate([item[key] for item in checked]) for key in checked[0]}
    order = np.argsort(combined["row_uid"], kind="stable")
    combined = {key: values[order] for key, values in combined.items()}
    validate_split_payload(combined, split)
    fit_rows, validation_rows = m2._partitions_for_spec(context.assignments, _fold_spec(seed, None))
    expected_rows = fit_rows if split == "train" else validation_rows
    for identity in ("row_uid", "plot_idx"):
        if combined[identity].tolist() != expected_rows[identity].tolist():
            raise JointTCPExtractionError(f"combined {split} {identity} differs from M1")
    if split == "train" and not np.array_equal(
        combined["producer_fold"], expected_rows["train_oof_fold"].to_numpy(dtype=np.int64)
    ):
        raise JointTCPExtractionError("combined producer folds differ from M1")
    return combined

"""Replay frozen OOF experts for pooled ConfidNet supervision.

Each development-training row is predicted only by its expert-held-out
producer. Validation rows use the full-development-train producer and never
carry labels through this module's output boundary. No expert is fitted here.
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
from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_protocol import canonical_sha256, sha256_file
from multimodal.models import MLPHead


DEFAULT_CONFIG_PATH = m2.DEFAULT_CONFIG_PATH
HIDDEN_DIM = initial.HIDDEN_DIM
PREFIXES = initial.PREFIXES
VALIDATION_PRODUCER_FOLD = -1
ConfidNetOOFExtractionError = initial.ConfidNetExtractionError


def validate_split_payload(payload: Mapping[str, Any], split: str) -> None:
    """Validate arrays and require explicit, valid producer-fold provenance."""

    if "producer_fold" not in payload:
        raise ConfidNetOOFExtractionError("missing producer_fold provenance")
    base = {key: value for key, value in payload.items() if key != "producer_fold"}
    initial.validate_split_payload(base, split)
    folds = np.asarray(payload["producer_fold"])
    if folds.dtype != np.int64 or folds.shape != np.asarray(payload["row_uid"]).shape:
        raise ConfidNetOOFExtractionError("producer_fold must be a row-aligned int64 array")
    allowed = m2.OOF_FOLDS if split == "train" else (VALIDATION_PRODUCER_FOLD,)
    if not np.isin(folds, allowed).all():
        raise ConfidNetOOFExtractionError("producer_fold values do not match the split")
    plot_folds: dict[str, int] = {}
    for plot, fold in zip(payload["plot_idx"], folds):
        if str(plot) in plot_folds and plot_folds[str(plot)] != int(fold):
            raise ConfidNetOOFExtractionError("one plot was assigned multiple producer folds")
        plot_folds[str(plot)] = int(fold)


def _fold_spec(seed: int, fold: int | None) -> m2.ProducerSpec:
    if fold is not None and (type(fold) is not int or fold not in m2.OOF_FOLDS):
        raise ConfidNetOOFExtractionError("fold must be 0, 1, 2, 3 or None for validation")
    return next(spec for spec in m2._producer_specs(seed) if spec.fold == fold)


def _reproduction(
    payload: Mapping[str, Any],
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    fold: int | None,
    seed: int,
) -> dict[str, Any]:
    split = "train" if fold is not None else "validation"
    validate_split_payload(payload, split)
    blind = {key: value for key, value in payload.items()
             if key not in {"producer_fold", "labels"}}
    record = initial._validate_validation_reproduction(blind, rows, reference)
    if "training_seed" not in reference or not reference["training_seed"].eq(seed).all():
        raise ConfidNetOOFExtractionError("saved reference training seed mismatch")
    if fold is not None:
        if ("train_oof_fold" not in reference
                or not reference["train_oof_fold"].eq(fold).all()
                or not np.all(payload["producer_fold"] == fold)):
            raise ConfidNetOOFExtractionError("saved reference or payload producer fold mismatch")
    elif "train_oof_fold" in reference:
        raise ConfidNetOOFExtractionError("validation reference must use the full-train producer")
    return {**record, "training_seed": seed, "producer_fold": fold}


def _activation_summary(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    return {
        "shape": list(values.shape),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "zero_fraction": float(np.mean(values == 0)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "row_l2_quantiles": np.quantile(
            np.linalg.norm(values, axis=1), [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "coordinate_mean": values.mean(axis=0).tolist(),
        "coordinate_std": values.std(axis=0).tolist(),
    }


def extract_producer(
    seed: int,
    fold: int | None,
    device: str = "cuda",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Extract one held-out producer; callers may seal/cache each result.

    ``fold`` is the existing M2 index 0/1/2/3. ``None`` extracts only the 822
    validation examples from the full-development-train expert set. Returns
    ``payload`` and JSON-serializable ``metadata`` without writing artifacts.
    """

    from safetensors.torch import load_file

    training_seed = m2.validate_training_seed(seed)
    spec = _fold_spec(training_seed, fold)
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ConfidNetOOFExtractionError("frozen replay supports CPU or CUDA devices")
    context = m2.load_frozen_context(config_path=config_path)
    producer = m2.validate_producer(context, spec, seed=training_seed)
    directory = context.output_root / spec.relative_directory
    fit_rows, prediction_rows = m2._partitions_for_spec(context.assignments, spec)
    rows = initial._prediction_projection(prediction_rows)
    if set(fit_rows["row_uid"]) & set(rows["row_uid"]):
        raise ConfidNetOOFExtractionError("expert fit and prediction rows overlap")
    if set(fit_rows["plot_idx"]) & set(rows["plot_idx"]):
        raise ConfidNetOOFExtractionError("expert fit and prediction plots overlap")
    if int(context.config["experts"]["head"]["hidden_dim"]) != HIDDEN_DIM:
        raise ConfidNetOOFExtractionError("frozen expert hidden dimension is not 256")
    scaler = m2._validate_scaler_file(directory / "geo_standardization.json")
    split = "train" if fold is not None else "validation"
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
        print(
            f"ConfidNet OOF seed {training_seed}: extracting {spec.stage_id} "
            f"({len(rows)} images)", flush=True,
        )
        images = m2._extract_embeddings(
            encoder, rows, transform=transform, config=context.config,
            seed=training_seed, device=target_device,
        )
        geo = m2.apply_geo_standardization(m2._geo_source_projection(context, rows), scaler)
        payload = {
            "row_uid": rows["row_uid"].to_numpy(dtype=str, copy=True),
            "plot_idx": rows["plot_idx"].to_numpy(dtype=str, copy=True),
            "producer_fold": np.full(
                len(rows), fold if fold is not None else VALIDATION_PRODUCER_FOLD,
                dtype=np.int64,
            ),
        }
        if split == "train":
            payload["labels"] = prediction_rows["label_id_dense"].to_numpy(
                dtype=np.int64, copy=True
            )
        for mode, prefix in m2.MODE_TO_PREFIX.items():
            hidden, logits = initial._head_hidden_and_logits(
                heads[mode], m2._head_input(mode, images, geo), target_device
            )
            payload[f"{prefix}_hidden"] = hidden
            payload[f"{prefix}_logits"] = logits
        reference = m2.read_output_parquet(
            directory / spec.output_filename, include_fold=spec.include_fold
        ).to_pandas()
        reproduction = _reproduction(payload, rows, reference, fold=fold, seed=training_seed)
        checkpoint_names = (
            "adapted_visual_tower.safetensors",
            *(f"{mode}_head.safetensors" for mode in m2.MODES),
        )
        metadata = {
            "schema_version": "geo_helpfulness.confidnet_oof_extraction_producer.v1",
            "training_seed": training_seed,
            "producer_fold": fold,
            "device": str(target_device),
            "protocol_id": str(context.config["protocol_id"]),
            "producer": producer,
            "parent_artifact_hashes": context.parent_hashes,
            "checkpoint_file_sha256": {
                filename: sha256_file(directory / filename) for filename in checkpoint_names
            },
            "scaler_file_sha256": sha256_file(directory / "geo_standardization.json"),
            "reference_output_file_sha256": sha256_file(directory / spec.output_filename),
            "encoder_initialization": initialization,
            "hidden_feature": "expert_head.net.1.ReLU_output",
            "hidden_dimension": HIDDEN_DIM,
            "expert_eval_mode": True,
            "prediction_transform": "frozen_M2_prediction_transform",
            "probability_basis": "native_t1_uncalibrated",
            "validation_labels_in_payload": False,
            "locked_test_accessed": False,
            "expert_fit_row_count": len(fit_rows),
            "expert_fit_plot_count": int(fit_rows["plot_idx"].nunique()),
            "prediction_row_count": len(rows),
            "prediction_plot_count": int(rows["plot_idx"].nunique()),
            "expert_fit_prediction_plots_disjoint": True,
            "prediction_identity_sha256": canonical_sha256(
                rows.loc[:, ["row_uid", "plot_idx"]].to_dict("records")
            ),
            "reproduction": reproduction,
            "activation_summaries": {
                prefix: _activation_summary(payload[f"{prefix}_hidden"])
                for prefix in PREFIXES
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
    seed: int,
    producers: Mapping[int | None, Mapping[str, Any]],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Check exact M1 membership and combine four OOF producer payloads.

    This rechecks cached payload identities, labels, fold provenance and M2
    logits. Cache manifests and file fingerprints remain the caller's concern.
    """

    training_seed = m2.validate_training_seed(seed)
    if set(producers) != set(m2.OOF_FOLDS) | {None}:
        raise ConfidNetOOFExtractionError("all four OOF producers and validation are required")
    context = m2.load_frozen_context(config_path=config_path)
    checked: dict[int | None, Mapping[str, Any]] = {}
    for fold in (*m2.OOF_FOLDS, None):
        spec = _fold_spec(training_seed, fold)
        result = producers[fold]
        if set(result) != {"payload", "metadata"}:
            raise ConfidNetOOFExtractionError("producer result must contain payload and metadata")
        metadata = result["metadata"]
        if (metadata.get("training_seed") != training_seed
                or metadata.get("producer_fold") != fold):
            raise ConfidNetOOFExtractionError("cached producer seed/fold metadata mismatch")
        fit_rows, prediction_rows = m2._partitions_for_spec(context.assignments, spec)
        rows = initial._prediction_projection(prediction_rows)
        directory = context.output_root / spec.relative_directory
        reference = m2.read_output_parquet(
            directory / spec.output_filename, include_fold=spec.include_fold
        ).to_pandas()
        payload = result["payload"]
        _reproduction(payload, rows, reference, fold=fold, seed=training_seed)
        if fold is not None and not np.array_equal(
            payload["labels"], prediction_rows["label_id_dense"].to_numpy(dtype=np.int64)
        ):
            raise ConfidNetOOFExtractionError("cached training labels differ from M1")
        checked[fold] = payload
    train = {
        key: np.concatenate([checked[fold][key] for fold in m2.OOF_FOLDS], axis=0)
        for key in checked[m2.OOF_FOLDS[0]]
    }
    order = np.argsort(train["row_uid"], kind="stable")
    train = {key: values[order] for key, values in train.items()}
    validation = dict(checked[None])
    validate_split_payload(train, "train")
    validate_split_payload(validation, "validation")
    if set(train["row_uid"]) & set(validation["row_uid"]):
        raise ConfidNetOOFExtractionError("development train/validation rows overlap")
    if set(train["plot_idx"]) & set(validation["plot_idx"]):
        raise ConfidNetOOFExtractionError("development train/validation plots overlap")
    full_fit, full_prediction = m2._partitions_for_spec(
        context.assignments, _fold_spec(training_seed, None)
    )
    if (train["row_uid"].tolist() != full_fit["row_uid"].tolist()
            or train["plot_idx"].tolist() != full_fit["plot_idx"].tolist()
            or not np.array_equal(
                train["producer_fold"], full_fit["train_oof_fold"].to_numpy(dtype=np.int64)
            )):
        raise ConfidNetOOFExtractionError("combined OOF rows differ from M1 identities/folds")
    return {
        "train": train,
        "validation": validation,
        "metadata": {
            "schema_version": "geo_helpfulness.confidnet_oof_extraction.v1",
            "training_seed": training_seed,
            "parent_artifact_hashes": context.parent_hashes,
            "hidden_dimension": HIDDEN_DIM,
            "supervision": "expert_heldout_oof_native_tcp",
            "validation_labels_in_payload": False,
            "locked_test_accessed": False,
            "all_oof_rows_exactly_once": True,
            "producers": {
                "validation" if fold is None else f"fold_{fold}": producers[fold]["metadata"]
                for fold in (*m2.OOF_FOLDS, None)
            },
            "splits": {
                split: {
                    "row_count": len(payload["row_uid"]),
                    "plot_count": len(set(payload["plot_idx"])),
                    "identity_sha256": canonical_sha256([
                        {"row_uid": str(row), "plot_idx": str(plot)}
                        for row, plot in zip(payload["row_uid"], payload["plot_idx"])
                    ]),
                }
                for split, payload in (("train", train), ("validation", validation))
            },
        },
    }


def extract_seed(
    seed: int,
    device: str = "cuda",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Convenience replay of all producers without intermediate artifact writes."""

    producers = {
        fold: extract_producer(seed, fold, device=device, config_path=config_path)
        for fold in (*m2.OOF_FOLDS, None)
    }
    return combine_producers(seed, producers, config_path=config_path)

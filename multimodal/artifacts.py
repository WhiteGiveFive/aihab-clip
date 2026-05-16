from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from aihab_utils.checkpointing import load_openclip_checkpoint
from multimodal.data import (
    IMAGE_METADATA_COLUMNS,
    SplitBundle,
    build_export_loader,
    build_split_bundles,
    image_embedding_dir,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encoder_source(cfg: Mapping) -> str:
    return str(cfg.get("multimodal", {}).get("image_feature_source", "habitat_finetuned")).lower()


def _image_checkpoint(cfg: Mapping) -> Path | None:
    value = cfg.get("multimodal", {}).get("image_checkpoint", None)
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(cfg.get("root_path", "./")) / path
    return path


def load_image_encoder(cfg: Mapping) -> Tuple[torch.nn.Module, object]:
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "multimodal image embedding export requires the `open_clip` package in the active environment."
        ) from exc

    model_name = str(cfg.get("open_clip_model", cfg.get("backbone", "ViT-B-16")))
    source = _encoder_source(cfg)
    checkpoint_path = _image_checkpoint(cfg)

    pretrained = cfg.get("open_clip_pretrained", None)
    if source == "habitat_finetuned":
        pretrained = None
    elif pretrained is None and not model_name.startswith("hf-hub:"):
        raise ValueError(
            "open_clip_pretrained is required when image_feature_source=pretrained "
            "for non-hf-hub OpenCLIP model names."
        )
    device = _device()
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)

    if source == "habitat_finetuned":
        if checkpoint_path is None:
            raise ValueError("multimodal.image_checkpoint is required when image_feature_source=habitat_finetuned")
        ckpt = load_openclip_checkpoint(model, checkpoint_path, device=device, strict=True)
        ckpt_model = ckpt.get("open_clip_model", None)
        if ckpt_model is not None and str(ckpt_model) != model_name:
            raise ValueError(f"Checkpoint model mismatch: cfg={model_name}, checkpoint={ckpt_model}")

    model.eval()
    return model, preprocess_val


def _embedding_columns(dim: int):
    width = max(3, len(str(dim - 1)))
    return [f"I{i:0{width}d}" for i in range(dim)]


def _frame_from_batches(meta_frames, feature_batches) -> pd.DataFrame:
    frame = pd.concat(meta_frames, ignore_index=True)
    feats = np.concatenate(feature_batches, axis=0)
    feature_cols = _embedding_columns(feats.shape[1])
    feature_frame = pd.DataFrame(feats, columns=feature_cols)
    return pd.concat([frame.reset_index(drop=True), feature_frame], axis=1)


def _metadata_columns(cfg: Mapping) -> list[str]:
    columns = cfg.get("multimodal", {}).get("image_metadata_columns", None)
    if not columns:
        return list(IMAGE_METADATA_COLUMNS)
    return [str(col) for col in columns]


def export_split_embeddings(cfg: Mapping, bundle: SplitBundle, model: torch.nn.Module) -> Path:
    out_dir = image_embedding_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bundle.name}.parquet"

    loader = build_export_loader(cfg, bundle)
    device = _device()
    meta_frames = []
    feature_batches = []
    normalize = bool(cfg.get("multimodal", {}).get("normalize_image_embeddings", True))
    metadata_columns = _metadata_columns(cfg)
    row_offset = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, _labels, _metadata = batch
            else:
                images, _labels = batch
            images = images.to(device, non_blocking=True)
            feats = model.encode_image(images)
            if normalize:
                feats = F.normalize(feats, dim=-1)
            feats = feats.detach().cpu().to(torch.float32).numpy()
            feature_batches.append(feats)
            batch_size = int(feats.shape[0])
            batch_frame = bundle.frame.iloc[row_offset:row_offset + batch_size].reset_index(drop=True)
            missing = [col for col in metadata_columns if col not in batch_frame.columns]
            if missing:
                raise ValueError(f"Split '{bundle.name}' metadata is missing columns required for export: {missing}")
            meta_frames.append(batch_frame.loc[:, metadata_columns])
            row_offset += batch_size

    if row_offset != len(bundle.frame):
        raise RuntimeError(
            f"Export row mismatch for split={bundle.name}: saw {row_offset} rows, expected {len(bundle.frame)}"
        )

    frame = _frame_from_batches(meta_frames, feature_batches)
    frame.to_parquet(out_path, index=False)
    return out_path


def export_image_embeddings(cfg: Mapping) -> Dict[str, Path]:
    model, eval_transform = load_image_encoder(cfg)
    bundles = build_split_bundles(cfg, eval_transform=eval_transform)
    paths = {}
    for split in ("train", "val", "test"):
        paths[split] = export_split_embeddings(cfg, bundles[split], model)
    return paths

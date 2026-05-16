from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

from data.cs2007_soil_aligned import (
    CHEM_FEATURES,
    CS2007SoilAlignedDataset,
    GROUP_COL,
    build_soil_aligned_splits,
    resolve_cfg_path,
)
from multimodal.artifacts import export_split_embeddings, load_image_encoder
from multimodal.data import (
    SOIL_FEATURE_COLUMNS,
    SplitBundle,
    image_embedding_dir,
    image_feature_columns,
    joined_table_dir,
)


SOIL_IMAGE_METADATA_COLUMNS = [
    "file",
    GROUP_COL,
    "label_id",
    "label_name",
    "l2_label",
    "split",
    *SOIL_FEATURE_COLUMNS,
]


def _output_split_name(split: str) -> str:
    return "val" if split == "valid" else split


def _cfg_with_soil_metadata_columns(cfg: Mapping) -> dict:
    out = copy.deepcopy(dict(cfg))
    mm_cfg = out.setdefault("multimodal", {})
    mm_cfg.setdefault("image_metadata_columns", list(SOIL_IMAGE_METADATA_COLUMNS))
    return out


def _soil_metadata_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    work = frame.copy()
    for out_col, source_col in zip(SOIL_FEATURE_COLUMNS, CHEM_FEATURES):
        work[out_col] = pd.to_numeric(work[source_col], errors="raise")

    return pd.DataFrame(
        {
            "file": work["file"].astype(str),
            GROUP_COL: work[GROUP_COL].astype(str),
            "label_id": work["label_id"].astype(int),
            "label_name": work["label_name"].astype(str),
            "l2_label": work["l2_label"].astype(int),
            "split": _output_split_name(split),
            **{col: work[col].astype("float32") for col in SOIL_FEATURE_COLUMNS},
        }
    )


def build_soil_split_bundles(cfg: Mapping, eval_transform) -> Dict[str, SplitBundle]:
    splits = build_soil_aligned_splits(cfg)
    soil_cfg = cfg["soil_aligned"]
    image_root = resolve_cfg_path(soil_cfg["image_root"], cfg)
    bundles: Dict[str, SplitBundle] = {}

    for source_split, frame in splits.frames.items():
        split = _output_split_name(source_split)
        dataset_frame = frame.copy()
        dataset_frame["split"] = split
        dataset = CS2007SoilAlignedDataset(
            dataset_frame,
            image_root=image_root,
            transform=eval_transform,
            return_metadata=True,
        )
        bundles[split] = SplitBundle(
            name=split,
            dataset=dataset,
            frame=_soil_metadata_frame(frame, source_split),
        )
    return bundles


def export_soil_image_embeddings(cfg: Mapping) -> Dict[str, Path]:
    export_cfg = _cfg_with_soil_metadata_columns(cfg)
    model, eval_transform = load_image_encoder(export_cfg)
    bundles = build_soil_split_bundles(export_cfg, eval_transform=eval_transform)
    return {split: export_split_embeddings(export_cfg, bundle, model) for split, bundle in bundles.items()}


def build_soil_feature_tables(cfg: Mapping) -> Dict[str, Path]:
    out_dir = joined_table_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = image_embedding_dir(cfg)
    outputs: Dict[str, Path] = {}

    for split in ("train", "val", "test"):
        image_path = image_dir / f"{split}.parquet"
        if not image_path.exists():
            raise FileNotFoundError(f"Soil image embedding parquet not found: {image_path}")
        frame = pd.read_parquet(image_path)
        missing = [col for col in SOIL_IMAGE_METADATA_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"Soil image embedding table for split={split} is missing columns: {missing}")

        table_path = out_dir / f"{split}.parquet"
        manifest_path = out_dir / f"{split}_manifest.json"
        frame.to_parquet(table_path, index=False)

        manifest = {
            "split": split,
            "rows": int(len(frame)),
            "image_feature_dim": int(len(image_feature_columns(frame))),
            "tabular_feature_dim": int(len(SOIL_FEATURE_COLUMNS)),
            "tabular_feature_columns": list(SOIL_FEATURE_COLUMNS),
            "label_count": int(frame["label_id"].nunique()) if not frame.empty else 0,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        outputs[split] = table_path
    return outputs

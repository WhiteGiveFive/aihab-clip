from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data import REASSIGN_NAME_LABEL_L3L2
from multimodal.artifacts import export_split_embeddings, load_image_encoder
from multimodal.data import (
    GEO_FEATURE_COLUMNS,
    SplitBundle,
    deduplicate_geo_embeddings,
    image_embedding_dir,
    image_feature_columns,
    join_split_with_geo,
    joined_table_dir,
    save_join_artifacts,
)


LABEL_COL = "BH_PLOT_DESC"
GROUP_COL = "ID"
SPLIT_COL = "split"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

GEO_10M_METADATA_COLUMNS = [
    "file",
    GROUP_COL,
    "plot_idx",
    "label_id",
    "label_name",
    "l2_label",
    "image_source",
    "split",
]


class Geo10mImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None, return_metadata: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_id"])
        if not self.return_metadata:
            return image, label
        metadata = {
            "file": str(row["file"]),
            GROUP_COL: str(row[GROUP_COL]),
            "plot_idx": str(row["plot_idx"]),
            "label_name": str(row["label_name"]),
            "l2_label": int(row["l2_label"]),
            "image_source": str(row["image_source"]),
            "split": str(row["split"]),
        }
        return image, label, metadata


def resolve_cfg_path(value: str | Path, cfg: Mapping) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(str(cfg.get("root_path", "./"))) / path


def _geo_10m_cfg(cfg: Mapping) -> Mapping:
    if "geo_10m" not in cfg:
        raise KeyError("Missing required top-level config section: geo_10m")
    return cfg["geo_10m"]


def _required_split_columns() -> set[str]:
    return {"file", GROUP_COL, LABEL_COL, SPLIT_COL}


def _validate_required_values(frame: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    missing_values = []
    for column in columns:
        if frame[column].isna().any():
            missing_values.append(column)
            continue
        blank_mask = frame[column].astype(str).str.strip() == ""
        if bool(blank_mask.any()):
            missing_values.append(column)
    if missing_values:
        raise ValueError(f"10m split CSV has missing required values in {sorted(set(missing_values))}: {source}")


def _load_curated_split(cfg: Mapping) -> pd.DataFrame:
    geo_cfg = _geo_10m_cfg(cfg)
    split_path = resolve_cfg_path(geo_cfg["split_csv"], cfg)
    if not split_path.exists():
        raise FileNotFoundError(f"10m curated split CSV not found: {split_path}")

    frame = pd.read_csv(split_path)
    required = _required_split_columns()
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"10m split CSV missing columns: {missing}")
    _validate_required_values(frame, sorted(required), split_path)

    work = frame.copy()
    for column in required:
        work[column] = work[column].astype(str).str.strip()
    work[SPLIT_COL] = work[SPLIT_COL].str.lower()

    allowed = {"train", "test", "removed"}
    unknown_splits = sorted(set(work[SPLIT_COL]).difference(allowed))
    if unknown_splits:
        raise ValueError(f"10m split CSV contains unsupported split values: {unknown_splits}")

    work = work[work[SPLIT_COL] != "removed"].reset_index(drop=True)
    if work.empty:
        raise ValueError("10m curated split has no usable rows after dropping split == removed")

    duplicate_files = work.loc[work["file"].duplicated(keep=False), "file"].astype(str).tolist()
    if duplicate_files:
        preview = sorted(set(duplicate_files))[:20]
        raise ValueError(f"10m curated split contains duplicate usable file names: {preview}")

    train_ids = set(work.loc[work[SPLIT_COL] == "train", GROUP_COL].astype(str))
    test_ids = set(work.loc[work[SPLIT_COL] == "test", GROUP_COL].astype(str))
    overlap = sorted(train_ids.intersection(test_ids))
    if overlap:
        raise ValueError(f"10m curated train/test ID groups overlap: {overlap[:20]}")

    label_names = work[LABEL_COL].astype(str).str.strip()
    unknown_labels = sorted(set(label_names).difference(REASSIGN_NAME_LABEL_L3L2.keys()))
    if unknown_labels:
        raise ValueError(f"10m split CSV contains unsupported CS labels: {unknown_labels}")

    label_pairs = label_names.map(REASSIGN_NAME_LABEL_L3L2)
    work["label_name"] = label_names
    work["label_id"] = [int(pair[0]) for pair in label_pairs]
    work["l2_label"] = [int(pair[1]) for pair in label_pairs]
    work["plot_idx"] = work[GROUP_COL].astype(str)
    return work


def _configured_image_roots(cfg: Mapping) -> list[Path]:
    geo_cfg = _geo_10m_cfg(cfg)
    roots = geo_cfg.get("image_roots", None)
    if not roots:
        raise ValueError("geo_10m.image_roots must list at least one image root")
    if isinstance(roots, (str, Path)):
        roots = [roots]
    resolved = [resolve_cfg_path(root, cfg) for root in roots]
    missing = [str(root) for root in resolved if not root.exists() or not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"10m image root(s) not found: {missing}")
    return resolved


def _resolve_image_paths(frame: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    roots = _configured_image_roots(cfg)
    requested = set(frame["file"].astype(str).tolist())
    paths_by_name: dict[str, list[tuple[Path, Path]]] = {name: [] for name in requested}

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.name in requested:
                paths_by_name[path.name].append((path, root))

    missing = sorted(name for name, matches in paths_by_name.items() if not matches)
    if missing:
        raise FileNotFoundError(f"10m curated usable image files are missing: {missing[:20]}")

    duplicates = {name: matches for name, matches in paths_by_name.items() if len(matches) > 1}
    if duplicates:
        preview = {name: [str(path) for path, _root in matches] for name, matches in list(duplicates.items())[:5]}
        raise ValueError(f"10m curated usable image files resolve to multiple paths: {preview}")

    out = frame.copy()
    out["image_path"] = [str(paths_by_name[name][0][0]) for name in out["file"].astype(str)]
    out["image_source"] = [str(paths_by_name[name][0][1]) for name in out["file"].astype(str)]
    return out


def _group_label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in frame.groupby(GROUP_COL, sort=False):
        labels = sorted(int(v) for v in group["label_id"].unique().tolist())
        if len(labels) != 1:
            raise ValueError(f"10m ID group has multiple labels and cannot be stratified: {group_id}")
        rows.append({GROUP_COL: str(group_id), "label_id": labels[0]})
    return pd.DataFrame(rows)


def _split_train_val(curated_train: pd.DataFrame, cfg: Mapping) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg.get("data", {}).get("data_split", {})
    valid_split = float(split_cfg.get("valid_split", 0.15))
    split_seed = int(split_cfg.get("split_seed", cfg.get("seed", 1)))
    if not 0.0 < valid_split < 1.0:
        raise ValueError(f"data.data_split.valid_split must be between 0 and 1 for 10m geo, got {valid_split}")

    groups = _group_label_frame(curated_train)
    class_group_counts = groups["label_id"].value_counts()
    sparse_classes = sorted(int(label) for label, count in class_group_counts.items() if int(count) < 2)
    if sparse_classes:
        raise ValueError(
            "10m grouped stratified validation split requires at least two ID groups per class; "
            f"sparse label ids: {sparse_classes}"
        )
    val_group_count = int(np.ceil(len(groups) * valid_split))
    if val_group_count < int(groups["label_id"].nunique()):
        raise ValueError(
            "10m grouped stratified validation split is too small to include every class; "
            f"increase data.data_split.valid_split above {valid_split}"
        )

    train_group_ids, val_group_ids = train_test_split(
        groups[GROUP_COL].to_numpy(),
        test_size=valid_split,
        random_state=split_seed,
        stratify=groups["label_id"].to_numpy(),
    )
    train_group_set = set(str(v) for v in train_group_ids)
    val_group_set = set(str(v) for v in val_group_ids)

    train = curated_train[curated_train[GROUP_COL].astype(str).isin(train_group_set)].copy()
    val = curated_train[curated_train[GROUP_COL].astype(str).isin(val_group_set)].copy()
    train[SPLIT_COL] = "train"
    val[SPLIT_COL] = "val"
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, GEO_10M_METADATA_COLUMNS].copy().reset_index(drop=True)


def _cfg_with_geo_10m_metadata_columns(cfg: Mapping) -> dict:
    out = copy.deepcopy(dict(cfg))
    mm_cfg = out.setdefault("multimodal", {})
    mm_cfg.setdefault("image_metadata_columns", list(GEO_10M_METADATA_COLUMNS))
    mm_cfg.setdefault("tabular_modality_name", "geo")
    mm_cfg.setdefault("tabular_feature_columns", list(GEO_FEATURE_COLUMNS))
    return out


def build_geo_10m_split_frames(cfg: Mapping) -> Dict[str, pd.DataFrame]:
    curated = _resolve_image_paths(_load_curated_split(cfg), cfg)
    curated_train = curated[curated[SPLIT_COL] == "train"].copy()
    curated_test = curated[curated[SPLIT_COL] == "test"].copy()
    if curated_train.empty:
        raise ValueError("10m curated split has no train rows after dropping removed rows")
    if curated_test.empty:
        raise ValueError("10m curated split has no test rows after dropping removed rows")

    train, val = _split_train_val(curated_train, cfg)
    curated_test[SPLIT_COL] = "test"
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": curated_test.reset_index(drop=True),
    }


def build_geo_10m_split_bundles(cfg: Mapping, eval_transform) -> Dict[str, SplitBundle]:
    frames = build_geo_10m_split_frames(cfg)
    bundles: Dict[str, SplitBundle] = {}
    for split, frame in frames.items():
        dataset = Geo10mImageDataset(frame, transform=eval_transform, return_metadata=True)
        bundles[split] = SplitBundle(name=split, dataset=dataset, frame=_metadata_frame(frame))
    return bundles


def export_geo_10m_image_embeddings(cfg: Mapping) -> Dict[str, Path]:
    export_cfg = _cfg_with_geo_10m_metadata_columns(cfg)
    model, eval_transform = load_image_encoder(export_cfg)
    bundles = build_geo_10m_split_bundles(export_cfg, eval_transform=eval_transform)
    return {split: export_split_embeddings(export_cfg, bundle, model) for split, bundle in bundles.items()}


def build_geo_10m_feature_tables(cfg: Mapping) -> Dict[str, Path]:
    out_dir = joined_table_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = image_embedding_dir(cfg)

    geo_cfg = _geo_10m_cfg(cfg)
    geo_path = resolve_cfg_path(geo_cfg["geo_parquet"], cfg)
    if not geo_path.exists():
        raise FileNotFoundError(f"10m geo parquet not found: {geo_path}")
    split_path = resolve_cfg_path(geo_cfg["split_csv"], cfg)

    geo_df, geo_stats = deduplicate_geo_embeddings(pd.read_parquet(geo_path))
    outputs: Dict[str, Path] = {}

    for split in ("train", "val", "test"):
        image_path = image_dir / f"{split}.parquet"
        if not image_path.exists():
            raise FileNotFoundError(f"10m image embedding parquet not found for split={split}: {image_path}")
        image_df = pd.read_parquet(image_path)
        joined_df, manifest, dropped_df = join_split_with_geo(image_df, geo_df)
        if not dropped_df.empty:
            preview = dropped_df["file"].astype(str).head(20).tolist()
            raise ValueError(f"10m geo join dropped {len(dropped_df)} rows for split={split}; missing geo files: {preview}")
        manifest.update(
            {
                "adapter": "geo_10m",
                "split_csv": str(split_path),
                "geo_parquet": str(geo_path),
                "geo_dedup_stats": geo_stats,
                "tabular_feature_columns": list(GEO_FEATURE_COLUMNS),
                "tabular_feature_dim": int(len(GEO_FEATURE_COLUMNS)),
                "image_feature_dim": int(len(image_feature_columns(joined_df))),
            }
        )
        outputs[split] = save_join_artifacts(split, joined_df, manifest, dropped_df, out_dir)["table"]
    return outputs


def inspect_geo_10m_splits(cfg: Mapping) -> Dict[str, dict]:
    frames = build_geo_10m_split_frames(cfg)
    summary: Dict[str, dict] = {}
    for split, frame in frames.items():
        summary[split] = {
            "rows": int(len(frame)),
            "id_groups": int(frame[GROUP_COL].nunique()),
            "labels": {str(k): int(v) for k, v in frame["label_name"].value_counts().sort_index().items()},
        }
    return summary

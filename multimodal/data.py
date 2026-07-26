from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import REASSIGN_LABEL_NAME_L3, l3_values_to_ids
from data.dataloader import (
    CSArrayDataset,
    _stratified_group_split_indices,
    derive_test_paths,
    few_shot_indices,
)
from data.dataset import image_loader
from multimodal.labels import resolve_target_spec


GEO_FEATURE_COLUMNS = [f"A{i:02d}" for i in range(64)]
SOIL_FEATURE_COLUMNS = [f"S{i:02d}" for i in range(3)]
IMAGE_FEATURE_RE = re.compile(r"^I\d+$")
IMAGE_METADATA_COLUMNS = [
    "file",
    "label_id",
    "label_name",
    "l2_label",
    "plot_idx",
    "image_source",
    "split",
]
DEFAULT_CLEANED_TEST_FILE_COLUMN = "file_name"
DEFAULT_CLEANED_TEST_FLAG_COLUMN = "Confirm to remove (Yes/No)?"


@dataclass(frozen=True)
class SplitBundle:
    name: str
    dataset: Dataset
    frame: pd.DataFrame


def _lower_file_series(values: Iterable[str]) -> pd.Series:
    return pd.Series([str(v).strip().lower() for v in values], dtype="string")


def _resolve_cfg_path(value: str | Path, cfg: Mapping) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(cfg.get("root_path", "./")) / path


def _resolve_output_root(cfg: Mapping) -> Path:
    root = Path(cfg.get("root_path", "./"))
    mm_cfg = cfg.get("multimodal", {})
    out = Path(mm_cfg.get("output_dir", "./multimodal_artifacts"))
    if not out.is_absolute():
        out = root / out
    return out


def _sanitize_name(value: str) -> str:
    return (
        str(value)
        .replace("hf-hub:", "hf-hub_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _encoder_tag(cfg: Mapping) -> str:
    mm_cfg = cfg.get("multimodal", {})
    source = str(mm_cfg.get("image_feature_source", "habitat_finetuned")).lower()
    if source == "habitat_finetuned":
        ckpt = mm_cfg.get("image_checkpoint", None)
        if ckpt:
            return _sanitize_name(Path(str(ckpt)).stem)
        return "habitat_finetuned"
    model_name = str(cfg.get("open_clip_model", cfg.get("backbone", "openclip")))
    return f"{_sanitize_name(model_name)}_pretrained"


def image_embedding_dir(cfg: Mapping) -> Path:
    return _resolve_output_root(cfg) / "image_embeddings" / str(cfg.get("dataset", "cs")) / _encoder_tag(cfg) / f"seed{int(cfg.get('seed', 1))}"


def joined_table_dir(cfg: Mapping) -> Path:
    mm_cfg = cfg.get("multimodal", {})
    table_tag = mm_cfg.get("joined_table_tag", None)
    if table_tag is None:
        table_tag = Path(str(mm_cfg.get("geo_embeddings_path", "geo"))).stem
    return _resolve_output_root(cfg) / "joined_tables" / str(cfg.get("dataset", "cs")) / _encoder_tag(cfg) / _sanitize_name(str(table_tag)) / f"seed{int(cfg.get('seed', 1))}"


def run_dir(cfg: Mapping) -> Path:
    mm_cfg = cfg.get("multimodal", {})
    fusion_mode = str(mm_cfg.get("fusion_mode", "raw_concat")).lower()
    target_spec = resolve_target_spec(cfg)
    root = _resolve_output_root(cfg) / "runs" / str(cfg.get("dataset", "cs")) / _encoder_tag(cfg)
    if target_spec.level == "l2":
        root = root / "target_l2"
    root = root / fusion_mode
    run_tag = mm_cfg.get("run_tag", None)
    if run_tag:
        root = root / _sanitize_name(str(run_tag))
    return root / f"seed{int(cfg.get('seed', 1))}"


def _subset_filter(
    labels: np.ndarray,
    images: np.ndarray,
    plot_word_labels: List[str],
    poly_labels: np.ndarray,
    poly_word_labels: List[str],
    file_names: List[str],
    plot_idx: Sequence,
    image_sources: List[str],
    subset_values,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[str], List[str], np.ndarray, List[str], List[int], List[str]]:
    subset_values = subset_values or []
    if isinstance(subset_values, (str, int)):
        subset_values = [subset_values]
    subset_names, subset_ids = l3_values_to_ids(subset_values)
    if not subset_ids:
        return (
            images,
            labels,
            plot_word_labels,
            poly_labels,
            poly_word_labels,
            file_names,
            np.asarray(plot_idx),
            image_sources,
            subset_ids,
            subset_names,
        )

    mask = np.isin(labels, subset_ids)
    return (
        images[mask],
        labels[mask],
        [x for x, keep in zip(plot_word_labels, mask) if keep],
        poly_labels[mask],
        [x for x, keep in zip(poly_word_labels, mask) if keep],
        [x for x, keep in zip(file_names, mask) if keep],
        np.asarray(plot_idx)[mask],
        [x for x, keep in zip(image_sources, mask) if keep],
        subset_ids,
        subset_names,
    )


def _build_label_mapping(train_labels: np.ndarray, test_labels: np.ndarray) -> Dict[int, int]:
    labels = sorted({int(v) for v in np.concatenate([train_labels, test_labels], axis=0).tolist()})
    return {orig: idx for idx, orig in enumerate(labels)}


def _selected_idx(cfg: Mapping, labels: np.ndarray, plot_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seed = int(cfg.get("seed", 1))
    rng = np.random.RandomState(seed)
    val_cfg = cfg.get("data", {}).get("data_split", {})
    val_ratio = float(val_cfg.get("valid_split", 0.1))
    val_seed = int(val_cfg.get("split_seed", seed))

    train_pool_idx, val_idx = _stratified_group_split_indices(labels, plot_idx, val_ratio, val_seed)
    shots_val = int(cfg.get("shots", 0) or 0)
    if shots_val > 0:
        rel_sel = few_shot_indices(labels[train_pool_idx], shots_val, rng)
        train_idx = train_pool_idx[rel_sel]
    else:
        train_idx = train_pool_idx

    mm_cfg = cfg.get("multimodal", {})
    max_samples = mm_cfg.get("max_samples_per_split", None)
    if max_samples is not None:
        max_samples = int(max_samples)
        train_idx = train_idx[:max_samples]
        val_idx = val_idx[:max_samples]

    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _metadata_frame_from_dataset(dataset: CSArrayDataset, split: str, label_map: Mapping[int, int]) -> pd.DataFrame:
    original_labels = [int(v) for v in dataset.labels.tolist()]
    label_names = [REASSIGN_LABEL_NAME_L3[int(v)] for v in original_labels]
    return pd.DataFrame(
        {
            "file": [str(v) for v in dataset.file_names],
            "file_lower": [str(v).strip().lower() for v in dataset.file_names],
            "label_id": [int(label_map[int(v)]) for v in original_labels],
            "label_id_original": original_labels,
            "label_name": label_names,
            "l2_label": [int(v) for v in np.asarray(dataset.l2_labels).tolist()],
            "plot_idx": [str(v) for v in np.asarray(dataset.plot_idx).tolist()],
            "image_source": [str(v) for v in dataset.image_sources],
            "split": split,
        }
    )


def build_split_bundles(cfg: Mapping, eval_transform) -> Dict[str, SplitBundle]:
    train_paths = cfg["data"]["dataset_paths"]
    train_index_names = cfg["data"]["index_file_names"]
    resize_dim = cfg["data"]["preprocessing"].get("resize", 256)

    train_blob = image_loader(train_paths, train_index_names, resize_dim, verbose=True)
    (
        images_tr,
        labels_tr,
        plot_word_labels_tr,
        poly_labels_tr,
        poly_word_labels_tr,
        file_names_tr,
        plot_idx_tr,
        src_tr,
    ) = train_blob

    (
        images_tr,
        labels_tr,
        plot_word_labels_tr,
        poly_labels_tr,
        poly_word_labels_tr,
        file_names_tr,
        plot_idx_tr,
        src_tr,
        _subset_ids,
        _subset_names,
    ) = _subset_filter(
        labels=labels_tr,
        images=images_tr,
        plot_word_labels=plot_word_labels_tr,
        poly_labels=poly_labels_tr,
        poly_word_labels=poly_word_labels_tr,
        file_names=file_names_tr,
        plot_idx=plot_idx_tr,
        image_sources=src_tr,
        subset_values=cfg.get("subset_l3", []),
    )

    test_paths = cfg["data"].get("test_dataset_paths", None)
    if test_paths:
        if isinstance(test_paths, str):
            test_paths = [test_paths]
    else:
        test_paths = derive_test_paths(train_paths)

    test_index_names = cfg["data"].get("test_index_file_names", None)
    if test_index_names:
        if isinstance(test_index_names, str):
            test_index_names = [test_index_names]
    else:
        test_index_names = train_index_names

    test_blob = image_loader(test_paths, test_index_names, resize_dim, verbose=True)
    (
        images_te,
        labels_te,
        plot_word_labels_te,
        poly_labels_te,
        poly_word_labels_te,
        file_names_te,
        plot_idx_te,
        src_te,
    ) = test_blob

    (
        images_te,
        labels_te,
        plot_word_labels_te,
        poly_labels_te,
        poly_word_labels_te,
        file_names_te,
        plot_idx_te,
        src_te,
        _subset_ids_te,
        _subset_names_te,
    ) = _subset_filter(
        labels=labels_te,
        images=images_te,
        plot_word_labels=plot_word_labels_te,
        poly_labels=poly_labels_te,
        poly_word_labels=poly_word_labels_te,
        file_names=file_names_te,
        plot_idx=plot_idx_te,
        image_sources=src_te,
        subset_values=cfg.get("subset_l3", []),
    )

    label_map = _build_label_mapping(labels_tr, labels_te)
    train_idx, val_idx = _selected_idx(cfg, labels_tr, np.asarray(plot_idx_tr))
    test_idx = np.arange(images_te.shape[0], dtype=np.int64)

    datasets = {
        "train": CSArrayDataset(
            images_tr,
            labels_tr,
            file_names_tr,
            train_idx,
            transform=eval_transform,
            plot_word_labels=plot_word_labels_tr,
            poly_labels=poly_labels_tr,
            poly_word_labels=poly_word_labels_tr,
            plot_idx=plot_idx_tr,
            image_sources=src_tr,
            return_metadata=True,
        ),
        "val": CSArrayDataset(
            images_tr,
            labels_tr,
            file_names_tr,
            val_idx,
            transform=eval_transform,
            plot_word_labels=plot_word_labels_tr,
            poly_labels=poly_labels_tr,
            poly_word_labels=poly_word_labels_tr,
            plot_idx=plot_idx_tr,
            image_sources=src_tr,
            return_metadata=True,
        ),
        "test": CSArrayDataset(
            images_te,
            labels_te,
            file_names_te,
            test_idx,
            transform=eval_transform,
            plot_word_labels=plot_word_labels_te,
            poly_labels=poly_labels_te,
            poly_word_labels=poly_word_labels_te,
            plot_idx=plot_idx_te,
            image_sources=src_te,
            return_metadata=True,
        ),
    }

    return {
        split: SplitBundle(
            name=split,
            dataset=ds,
            frame=_metadata_frame_from_dataset(ds, split=split, label_map=label_map),
        )
        for split, ds in datasets.items()
    }


def build_export_loader(cfg: Mapping, bundle: SplitBundle) -> DataLoader:
    mm_cfg = cfg.get("multimodal", {})
    return DataLoader(
        bundle.dataset,
        batch_size=int(mm_cfg.get("embedding_batch_size", 32)),
        shuffle=False,
        num_workers=int(mm_cfg.get("embedding_num_workers", cfg["data"].get("num_workers", 0))),
        pin_memory=True,
    )


def _validate_geo_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"file", *GEO_FEATURE_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Geo parquet missing columns: {sorted(missing)}")
    return df.copy()


def deduplicate_geo_embeddings(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    work = _validate_geo_rows(df)
    work["file_lower"] = _lower_file_series(work["file"])
    if "BH_PLOT_DESC" in work.columns:
        bh_values = work["BH_PLOT_DESC"]
    else:
        bh_values = pd.Series("", index=work.index, dtype="string")
    work["bh_label_clean"] = bh_values.astype(str).str.strip()
    work["label_nonempty"] = (work["bh_label_clean"] != "").astype(int)
    work = work.sort_values(["file_lower", "label_nonempty"], ascending=[True, False]).reset_index(drop=True)

    kept_rows: List[pd.Series] = []
    duplicate_groups = 0
    preferred_nonempty = 0

    for _, group in work.groupby("file_lower", sort=False):
        if len(group) == 1:
            kept_rows.append(group.iloc[0])
            continue

        duplicate_groups += 1
        first = group.iloc[0]
        if int(first["label_nonempty"]) == 1:
            preferred_nonempty += 1

        for _, row in group.iloc[1:].iterrows():
            a_first = first[GEO_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            a_row = row[GEO_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            if not np.allclose(a_first, a_row, equal_nan=True):
                raise ValueError(f"Conflicting geo embedding vectors for file={first['file']}")
        kept_rows.append(first)

    deduped = pd.DataFrame(kept_rows).drop(columns=["bh_label_clean", "label_nonempty"]).reset_index(drop=True)
    stats = {
        "input_rows": int(len(df)),
        "deduped_rows": int(len(deduped)),
        "duplicate_groups": int(duplicate_groups),
        "preferred_nonempty_groups": int(preferred_nonempty),
    }
    return deduped, stats


def _geo_path(cfg: Mapping) -> Path:
    mm_cfg = cfg.get("multimodal", {})
    geo_path = Path(mm_cfg.get("geo_embeddings_path", "data/cs_geo_gse_10km/CS_Xplots_embeddings_per_file.parquet"))
    if not geo_path.is_absolute():
        geo_path = Path(cfg.get("root_path", "./")) / geo_path
    return geo_path


def load_geo_embeddings(cfg: Mapping) -> Tuple[pd.DataFrame, Dict[str, int]]:
    geo_path = _geo_path(cfg)
    if not geo_path.exists():
        raise FileNotFoundError(f"Geo embeddings parquet not found: {geo_path}")
    geo_df = pd.read_parquet(geo_path)
    return deduplicate_geo_embeddings(geo_df)


def image_feature_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if IMAGE_FEATURE_RE.match(str(c))])


def tabular_modality_name(cfg: Mapping) -> str:
    mm_cfg = cfg.get("multimodal", {})
    value = mm_cfg.get("tabular_modality_name", None)
    if value:
        return str(value)
    mode = str(mm_cfg.get("fusion_mode", "raw_concat")).lower()
    if mode.startswith("soil_") or mode == "soil_only":
        return "soil"
    return "geo"


def tabular_feature_columns(cfg: Mapping) -> List[str]:
    mm_cfg = cfg.get("multimodal", {})
    configured = mm_cfg.get("tabular_feature_columns", None)
    if configured:
        return [str(c) for c in configured]
    if tabular_modality_name(cfg) == "soil":
        return list(SOIL_FEATURE_COLUMNS)
    return list(GEO_FEATURE_COLUMNS)


def cleaned_test_enabled(cfg: Mapping) -> bool:
    clean_cfg = cfg.get("data", {}).get("cleaned_test", {}) or {}
    return bool(clean_cfg.get("enabled", False))


def _cleaned_test_cfg(cfg: Mapping) -> Mapping:
    return cfg.get("data", {}).get("cleaned_test", {}) or {}


def load_cleaned_test_removed_files(cfg: Mapping) -> Tuple[set[str], Dict[str, object]]:
    clean_cfg = _cleaned_test_cfg(cfg)
    if not bool(clean_cfg.get("enabled", False)):
        return set(), {"enabled": False}

    review_csv = clean_cfg.get("review_csv", None)
    if not review_csv:
        raise ValueError("data.cleaned_test.review_csv is required when cleaned_test.enabled is true")
    review_path = _resolve_cfg_path(review_csv, cfg)
    if not review_path.exists():
        raise FileNotFoundError(f"Cleaned test review CSV not found: {review_path}")

    file_col = str(clean_cfg.get("file_column", DEFAULT_CLEANED_TEST_FILE_COLUMN))
    flag_col = str(clean_cfg.get("flag_column", DEFAULT_CLEANED_TEST_FLAG_COLUMN))
    remove_values_raw = clean_cfg.get("remove_values", ["Yes"])
    if isinstance(remove_values_raw, (str, int, float, bool)):
        remove_values_raw = [remove_values_raw]
    remove_values = [str(value) for value in remove_values_raw]
    remove_value_keys = {value.strip().lower() for value in remove_values}
    if not remove_value_keys:
        raise ValueError("data.cleaned_test.remove_values must contain at least one value")

    review = pd.read_csv(review_path, encoding="utf-8-sig")
    missing = {file_col, flag_col}.difference(review.columns)
    if missing:
        raise ValueError(f"Cleaned test review CSV missing columns: {sorted(missing)}")

    flags = review[flag_col].astype(str).str.strip().str.lower()
    remove_mask = flags.isin(remove_value_keys)
    removed_files = _lower_file_series(review.loc[remove_mask, file_col])
    blank = removed_files.isna() | (removed_files.fillna("") == "")
    if bool(blank.any()):
        raise ValueError(f"Cleaned test review CSV contains {int(blank.sum())} blank removal file names")

    removed_set = set(removed_files.tolist())
    metadata = {
        "enabled": True,
        "review_csv": str(review_path),
        "file_column": file_col,
        "flag_column": flag_col,
        "remove_values": remove_values,
        "review_rows": int(len(review)),
        "flagged_rows": int(remove_mask.sum()),
        "flagged_unique_files": int(len(removed_set)),
    }
    return removed_set, metadata


def apply_cleaned_test_filter(
    cfg: Mapping,
    split: str,
    frame: pd.DataFrame,
    file_column: str = "file",
) -> Tuple[pd.DataFrame, Dict[str, object] | None]:
    if str(split) != "test" or not cleaned_test_enabled(cfg):
        return frame, None
    if file_column not in frame.columns:
        raise ValueError(f"Cleaned test filtering requires column '{file_column}'")

    removed_files, metadata = load_cleaned_test_removed_files(cfg)
    input_rows = int(len(frame))
    frame_keys = _lower_file_series(frame[file_column])
    frame_key_set = set(frame_keys.tolist())
    missing = sorted(removed_files.difference(frame_key_set))
    if missing:
        raise ValueError(
            "Cleaned test review flags files absent from the test split: "
            f"{missing[:20]}"
        )

    remove_mask = frame_keys.isin(removed_files).to_numpy()
    removed_preview = frame.loc[remove_mask, file_column].astype(str).head(20).tolist()
    filtered = frame.loc[~remove_mask].reset_index(drop=True)

    metadata.update(
        {
            "input_rows": input_rows,
            "removed_rows": int(remove_mask.sum()),
            "output_rows": int(len(filtered)),
            "removed_files_preview": removed_preview,
        }
    )
    return filtered, metadata


def join_split_with_geo(image_split_df: pd.DataFrame, geo_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    image_df = image_split_df.copy()
    image_df["file_lower"] = _lower_file_series(image_df["file"])
    geo = geo_df.copy()
    geo["file_lower"] = _lower_file_series(geo["file"])

    merged = image_df.merge(
        geo[["file_lower", *GEO_FEATURE_COLUMNS]],
        on="file_lower",
        how="inner",
    )
    dropped = image_df[~image_df["file_lower"].isin(set(merged["file_lower"].tolist()))].copy()
    manifest = {
        "split": str(image_df["split"].iloc[0]) if not image_df.empty else "unknown",
        "image_rows": int(len(image_df)),
        "matched_rows": int(len(merged)),
        "dropped_rows": int(len(dropped)),
        "dropped_files_preview": dropped["file"].astype(str).head(20).tolist(),
        "image_feature_dim": int(len(image_feature_columns(image_df))),
        "geo_feature_dim": int(len(GEO_FEATURE_COLUMNS)),
        "label_count": int(merged["label_id"].nunique()) if not merged.empty else 0,
    }
    return merged.drop(columns=["file_lower"]).reset_index(drop=True), manifest, dropped.drop(columns=["file_lower"])


def save_join_artifacts(split: str, joined_df: pd.DataFrame, manifest: Mapping, dropped_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{split}.parquet"
    manifest_path = output_dir / f"{split}_manifest.json"
    dropped_path = output_dir / f"{split}_dropped.csv"
    joined_df.to_parquet(table_path, index=False)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    dropped_df.to_csv(dropped_path, index=False)
    return {"table": table_path, "manifest": manifest_path, "dropped": dropped_path}


def load_joined_splits(cfg: Mapping) -> Dict[str, pd.DataFrame]:
    out_dir = joined_table_dir(cfg)
    splits = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Joined split parquet not found: {path}")
        splits[split] = pd.read_parquet(path)
    return splits


class FeatureTableDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        mode: str,
        tabular_cols: Sequence[str] | None = None,
        target_col: str = "label_id",
    ):
        self.frame = frame.reset_index(drop=True)
        self.mode = mode
        self.target_col = str(target_col)
        self.image_cols = image_feature_columns(self.frame)
        self.tabular_cols = list(tabular_cols or GEO_FEATURE_COLUMNS)
        if not self.image_cols:
            raise ValueError("Joined feature table is missing image feature columns.")
        missing = [c for c in self.tabular_cols if c not in self.frame.columns]
        if missing:
            raise ValueError(f"Joined feature table is missing tabular feature columns: {missing}")
        if self.target_col not in self.frame.columns:
            raise ValueError(f"Joined feature table is missing target column: {self.target_col}")
        self.labels = torch.tensor(
            self.frame[self.target_col].astype(int).to_numpy(),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image_feat = torch.tensor(row[self.image_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        tabular_feat = torch.tensor(row[self.tabular_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        label = self.labels[idx]
        return image_feat, tabular_feat, label

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - training still requires torch.
    class Dataset:  # type: ignore[no-redef]
        pass

from data import REASSIGN_NAME_LABEL_L3L2


TARGET_COL = "BH_PLOT_DESC"
MODEL_TARGET_COL = "habitat_model"
GROUP_COL = "ID"
RAW_CHEM_FEATURES = [
    "C_B_PH_DIW",
    "C_FE_C_CONCLOI_GCPERKG",
    "C_FE_NTOTAL",
]
CHEM_FEATURES = [
    "C_B_PH_DIW",
    "log_C_FE_C_CONCLOI_GCPERKG",
    "log_C_FE_NTOTAL",
]


@dataclass(frozen=True)
class SoilAlignedSplits:
    frames: Dict[str, pd.DataFrame]
    manifest: Dict[str, object]
    rare_labels: Tuple[str, ...]
    common_labels: Tuple[str, ...]
    soil_label_order: Tuple[str, ...]


class CS2007SoilAlignedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, transform=None, return_metadata: bool = False):
        self.frame = frame.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(self.image_root / str(row["file"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label_id"])
        if not self.return_metadata:
            return image, label

        metadata = {
            "l2_label": int(row["l2_label"]),
            "poly_label": -1,
            "plot_word_label": str(row[TARGET_COL]),
            "poly_word_label": "",
            "file_name": str(row["file"]),
            "plot_idx": str(row[GROUP_COL]),
            "image_source": str(self.image_root),
            "split": str(row["split"]),
            "soil_model_label": str(row["soil_model_label"]),
        }
        return image, label, metadata


def resolve_cfg_path(value: str | Path, cfg: Mapping) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    root = Path(cfg.get("root_path", "./")).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return (root / path).resolve()


def _soil_cfg(cfg: Mapping) -> Mapping:
    soil_cfg = cfg.get("soil_aligned", {})
    if not soil_cfg:
        raise ValueError("Missing required top-level config key: soil_aligned")
    return soil_cfg


def _load_and_clean_soil_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()

    required = {"file", TARGET_COL, GROUP_COL, *RAW_CHEM_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Soil CSV is missing required columns: {sorted(missing)}")

    for col in [TARGET_COL, GROUP_COL]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    df["file"] = df["file"].astype("string").str.strip()
    df["file"] = df["file"].replace("", pd.NA)

    for col in RAW_CHEM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -9999, col] = np.nan

    df["log_C_FE_C_CONCLOI_GCPERKG"] = np.log1p(df["C_FE_C_CONCLOI_GCPERKG"])
    df["log_C_FE_NTOTAL"] = np.log1p(df["C_FE_NTOTAL"])

    return df.dropna(subset=["file", TARGET_COL, GROUP_COL, *CHEM_FEATURES]).copy()


def _recode_rare_classes(df: pd.DataFrame, threshold: int) -> Tuple[pd.DataFrame, pd.Series, Tuple[str, ...]]:
    work = df.copy()
    raw_counts = work[TARGET_COL].value_counts()
    rare_labels = tuple(raw_counts[raw_counts < threshold].index.tolist())
    work[MODEL_TARGET_COL] = np.where(work[TARGET_COL].isin(rare_labels), "Other_rare", work[TARGET_COL])
    return work, raw_counts, rare_labels


def _majority_label(series: pd.Series) -> str:
    counts = series.value_counts()
    max_count = counts.max()
    return sorted(counts[counts == max_count].index)[0]


def _split_groups(group_df: pd.DataFrame, test_size: float, random_state: int):
    return train_test_split(
        group_df,
        test_size=test_size,
        random_state=random_state,
        stratify=group_df["stratify_label"],
    )


def _make_grouped_split_ids(
    df: pd.DataFrame,
    train_size: float,
    valid_size: float,
    test_size: float,
    random_state: int,
) -> Dict[str, set]:
    if abs(train_size + valid_size + test_size - 1.0) >= 1e-9:
        raise ValueError("train_size + valid_size + test_size must equal 1.0")

    group_df = (
        df.groupby(GROUP_COL, as_index=False)[MODEL_TARGET_COL]
        .agg(_majority_label)
        .rename(columns={MODEL_TARGET_COL: "stratify_label"})
    )
    train_groups, temp_groups = _split_groups(
        group_df,
        test_size=valid_size + test_size,
        random_state=random_state,
    )
    valid_groups, test_groups = _split_groups(
        temp_groups,
        test_size=test_size / (valid_size + test_size),
        random_state=random_state + 1,
    )

    split_ids = {
        "train": set(train_groups[GROUP_COL]),
        "valid": set(valid_groups[GROUP_COL]),
        "test": set(test_groups[GROUP_COL]),
    }
    if split_ids["train"] & split_ids["valid"] or split_ids["train"] & split_ids["test"] or split_ids["valid"] & split_ids["test"]:
        raise AssertionError("Grouped split leaked at least one ID across splits.")
    return split_ids


def _soil_label_order(df: pd.DataFrame) -> Tuple[str, ...]:
    return tuple(df[MODEL_TARGET_COL].value_counts().index.tolist())


def _common_labels(raw_counts: pd.Series, rare_labels: Iterable[str]) -> Tuple[str, ...]:
    rare = set(rare_labels)
    return tuple(label for label in raw_counts.index.tolist() if label not in rare)


def map_to_soil_model_label(label_name: str, rare_labels: Iterable[str], common_labels: Iterable[str]) -> str:
    common = set(common_labels)
    rare = set(rare_labels)
    if label_name in common and label_name not in rare:
        return str(label_name)
    return "Other_rare"


def _attach_project_labels(frame: pd.DataFrame, rare_labels: Iterable[str], common_labels: Iterable[str]) -> pd.DataFrame:
    work = frame.copy()
    labels = work[TARGET_COL].astype(str).tolist()
    work["label_id"] = [int(REASSIGN_NAME_LABEL_L3L2[label][0]) for label in labels]
    work["l2_label"] = [int(REASSIGN_NAME_LABEL_L3L2[label][1]) for label in labels]
    work["label_name"] = labels
    work["soil_model_label"] = [
        map_to_soil_model_label(label, rare_labels=rare_labels, common_labels=common_labels)
        for label in labels
    ]
    return work


def _canonicalize_image_files(frames: Mapping[str, pd.DataFrame], image_root: Path) -> Dict[str, pd.DataFrame]:
    image_names = {
        path.name.strip().lower(): path.name
        for path in image_root.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }
    canonicalized: Dict[str, pd.DataFrame] = {}
    missing = []
    for split, frame in frames.items():
        work = frame.copy()
        canonical_files = []
        for file_name in work["file"].astype(str).tolist():
            exact_path = image_root / file_name
            if exact_path.is_file():
                canonical_files.append(file_name)
                continue
            canonical = image_names.get(file_name.strip().lower())
            if canonical is None:
                missing.append({"split": split, "file": file_name})
                canonical_files.append(file_name)
            else:
                canonical_files.append(canonical)
        work["file"] = canonical_files
        canonicalized[split] = work
    if missing:
        examples = missing[:10]
        raise FileNotFoundError(f"{len(missing)} aligned images are missing. Examples: {examples}")
    return canonicalized


def _validate_images(frames: Mapping[str, pd.DataFrame], image_root: Path) -> None:
    missing = []
    for split, frame in frames.items():
        for file_name in frame["file"].astype(str).tolist():
            if not (image_root / file_name).is_file():
                missing.append({"split": split, "file": file_name})
    if missing:
        examples = missing[:10]
        raise FileNotFoundError(f"{len(missing)} aligned images are missing. Examples: {examples}")


def _assert_expected_counts(manifest: Mapping[str, object], soil_cfg: Mapping) -> None:
    expected = {
        "valid_soil_rows": soil_cfg.get("expected_valid_soil_rows", None),
        "openclip_rows": soil_cfg.get("expected_openclip_rows", None),
    }
    for key, value in expected.items():
        if value is not None and int(manifest[key]) != int(value):
            raise AssertionError(f"Expected {key}={value}, got {manifest[key]}")

    expected_split_rows = soil_cfg.get("expected_openclip_split_rows", None)
    if expected_split_rows:
        observed = manifest["openclip_split_rows"]
        for split, value in expected_split_rows.items():
            if int(observed[split]) != int(value):
                raise AssertionError(f"Expected {split} rows={value}, got {observed[split]}")


def build_soil_aligned_splits(cfg: Mapping) -> SoilAlignedSplits:
    soil_cfg = _soil_cfg(cfg)
    soil_csv = resolve_cfg_path(soil_cfg["soil_csv"], cfg)
    image_root = resolve_cfg_path(soil_cfg["image_root"], cfg)

    df_model = _load_and_clean_soil_rows(soil_csv)
    rare_threshold = int(soil_cfg.get("rare_threshold", 20))
    df_model, raw_counts, rare_labels = _recode_rare_classes(df_model, rare_threshold)
    common_labels = _common_labels(raw_counts, rare_labels)
    soil_labels = _soil_label_order(df_model)

    split_ids = _make_grouped_split_ids(
        df_model,
        train_size=float(soil_cfg.get("train_size", 0.70)),
        valid_size=float(soil_cfg.get("valid_size", 0.15)),
        test_size=float(soil_cfg.get("test_size", 0.15)),
        random_state=int(soil_cfg.get("split_seed", 42)),
    )

    soil_split_rows = {}
    openclip_split_rows = {}
    other_rare_support = {}
    dropped_rows = []
    frames: Dict[str, pd.DataFrame] = {}

    supported_labels = set(REASSIGN_NAME_LABEL_L3L2.keys())
    for split, ids in split_ids.items():
        split_frame = df_model[df_model[GROUP_COL].isin(ids)].copy()
        split_frame["split"] = split
        soil_split_rows[split] = int(len(split_frame))

        unsupported = split_frame[~split_frame[TARGET_COL].isin(supported_labels)].copy()
        if not unsupported.empty:
            dropped_rows.extend(
                unsupported[["file", GROUP_COL, TARGET_COL, "split"]].to_dict(orient="records")
            )
        split_frame = split_frame[split_frame[TARGET_COL].isin(supported_labels)].copy()
        split_frame = _attach_project_labels(split_frame, rare_labels=rare_labels, common_labels=common_labels)
        split_frame = split_frame.sort_values(["file", GROUP_COL]).reset_index(drop=True)
        openclip_split_rows[split] = int(len(split_frame))
        other_rare_support[split] = int((split_frame["soil_model_label"] == "Other_rare").sum())
        frames[split] = split_frame

    frames = _canonicalize_image_files(frames, image_root)
    _validate_images(frames, image_root)

    all_ids_by_split = {split: set(frame[GROUP_COL].astype(str)) for split, frame in frames.items()}
    if all_ids_by_split["train"] & all_ids_by_split["valid"] or all_ids_by_split["train"] & all_ids_by_split["test"] or all_ids_by_split["valid"] & all_ids_by_split["test"]:
        raise AssertionError("OpenCLIP frames leaked at least one ID across splits.")

    manifest: Dict[str, object] = {
        "soil_csv": str(soil_csv),
        "image_root": str(image_root),
        "rare_threshold": rare_threshold,
        "valid_soil_rows": int(len(df_model)),
        "openclip_rows": int(sum(openclip_split_rows.values())),
        "soil_split_rows": soil_split_rows,
        "openclip_split_rows": openclip_split_rows,
        "other_rare_support_after_boundary_drop": other_rare_support,
        "rare_labels": list(rare_labels),
        "common_labels": list(common_labels),
        "soil_label_order": list(soil_labels),
        "dropped_unsupported_rows": dropped_rows,
    }

    if bool(soil_cfg.get("validate_expected_counts", True)):
        _assert_expected_counts(manifest, soil_cfg)

    return SoilAlignedSplits(
        frames=frames,
        manifest=manifest,
        rare_labels=tuple(rare_labels),
        common_labels=tuple(common_labels),
        soil_label_order=tuple(soil_labels),
    )


def output_dir_from_cfg(cfg: Mapping) -> Path:
    soil_cfg = _soil_cfg(cfg)
    return resolve_cfg_path(soil_cfg.get("output_dir", "./results_lr/cs2007_soil_aligned"), cfg)


def save_split_artifacts(splits: SoilAlignedSplits, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    manifest_path = output_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(splits.manifest, handle, indent=2)
    paths["manifest"] = manifest_path

    columns = [
        "file",
        GROUP_COL,
        TARGET_COL,
        "label_id",
        "label_name",
        "l2_label",
        "soil_model_label",
        "split",
    ]
    for split, frame in splits.frames.items():
        path = output_dir / f"{split}_files.csv"
        frame.loc[:, columns].to_csv(path, index=False)
        paths[split] = path
    return paths

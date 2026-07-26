from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from data import NAME_LABEL_L2


TARGET_ID_COLUMN = "target_id"
FINGERPRINT_COLUMNS = ("file", "plot_idx", "l2_label", "split")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class TargetSpec:
    level: str
    source_column: str
    name_column: str | None
    canonical_id_to_name: Mapping[int, str]


@dataclass(frozen=True)
class TargetEncoding:
    spec: TargetSpec
    canonical_class_ids: tuple[int, ...]
    target_id_remap: Mapping[int, int]
    inverse_target_id_remap: Mapping[int, int]
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.canonical_class_ids)


L2_ID_TO_NAME = {int(label_id): str(name) for name, label_id in NAME_LABEL_L2.items()}
L2_TARGET_SPEC = TargetSpec(
    level="l2",
    source_column="l2_label",
    name_column=None,
    canonical_id_to_name=L2_ID_TO_NAME,
)
L3_TARGET_SPEC = TargetSpec(
    level="l3",
    source_column="label_id",
    name_column="label_name",
    canonical_id_to_name={},
)


def resolve_target_spec(cfg: Mapping) -> TargetSpec:
    level = str(cfg.get("multimodal", {}).get("target_level", "l3")).strip().lower()
    if level == "l3":
        return L3_TARGET_SPEC
    if level == "l2":
        return L2_TARGET_SPEC
    raise ValueError(
        "multimodal.target_level must be either 'l3' or 'l2'; "
        f"got {level!r}."
    )


def _coerce_target_ids(frame: pd.DataFrame, split: str, spec: TargetSpec) -> pd.Series:
    column = spec.source_column
    if column not in frame.columns:
        raise ValueError(
            f"Joined {split} split is missing target column {column!r} "
            f"for target level {spec.level!r}."
        )
    if frame[column].isna().any():
        raise ValueError(f"Joined {split} split contains missing {column} targets.")

    numeric = pd.to_numeric(frame[column], errors="coerce")
    numeric_array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_array).all():
        raise ValueError(f"Joined {split} split contains non-numeric or non-finite {column} targets.")
    if not np.equal(numeric_array, np.floor(numeric_array)).all():
        raise ValueError(f"Joined {split} split contains non-integer {column} targets.")

    target_ids = pd.Series(numeric_array.astype(np.int64), index=frame.index, name=column)
    negative = sorted(int(value) for value in target_ids[target_ids < 0].unique().tolist())
    if negative:
        raise ValueError(f"Joined {split} split contains negative {column} targets: {negative}")

    if spec.canonical_id_to_name:
        unknown = sorted(set(target_ids.unique().tolist()).difference(spec.canonical_id_to_name))
        if unknown:
            raise ValueError(
                f"Joined {split} split contains unknown canonical {spec.level.upper()} targets: {unknown}"
            )
    return target_ids


def _l3_class_names(train_df: pd.DataFrame, target_ids: pd.Series, class_ids: Sequence[int]) -> tuple[str, ...]:
    name_column = L3_TARGET_SPEC.name_column
    assert name_column is not None
    if name_column not in train_df.columns:
        raise ValueError(f"Joined train split is missing target name column {name_column!r}.")
    names = []
    for class_id in class_ids:
        values = train_df.loc[target_ids == class_id, name_column]
        if values.empty or pd.isna(values.iloc[0]):
            raise ValueError(
                f"Joined train split lacks a {name_column} value for label_id={class_id}."
            )
        # Preserve the legacy behavior of using the first training-row name.
        names.append(str(values.iloc[0]))
    return tuple(names)


def build_target_encoding(
    tables: Mapping[str, pd.DataFrame],
    target_spec: TargetSpec,
    training_split_name: str = "geo-matched train split",
) -> TargetEncoding:
    if "train" not in tables:
        raise ValueError("Target encoding requires a train table.")
    train_df = tables["train"]
    if train_df.empty:
        raise ValueError("Joined train split is empty after geo matching.")

    source_ids: Dict[str, pd.Series] = {}
    for split, frame in tables.items():
        if frame.empty:
            raise ValueError(f"Joined {split} split is empty after geo matching.")
        source_ids[split] = _coerce_target_ids(frame, split, target_spec)

    canonical_class_ids = tuple(sorted(int(value) for value in source_ids["train"].unique().tolist()))
    target_id_remap = {canonical_id: dense_id for dense_id, canonical_id in enumerate(canonical_class_ids)}
    for split, split_ids in source_ids.items():
        unseen = sorted(set(int(value) for value in split_ids.unique().tolist()).difference(target_id_remap))
        if unseen:
            raise ValueError(
                f"Joined {split} split contains labels absent from the {training_split_name}: {unseen}"
            )

    if target_spec.canonical_id_to_name:
        class_names = tuple(target_spec.canonical_id_to_name[class_id] for class_id in canonical_class_ids)
    else:
        class_names = _l3_class_names(
            train_df,
            source_ids["train"],
            canonical_class_ids,
        )

    return TargetEncoding(
        spec=target_spec,
        canonical_class_ids=canonical_class_ids,
        target_id_remap=target_id_remap,
        inverse_target_id_remap={dense_id: canonical_id for canonical_id, dense_id in target_id_remap.items()},
        class_names=class_names,
    )


def materialize_target_ids(
    tables: Mapping[str, pd.DataFrame],
    encoding: TargetEncoding,
) -> Dict[str, pd.DataFrame]:
    remapped: Dict[str, pd.DataFrame] = {}
    for split, frame in tables.items():
        source_ids = _coerce_target_ids(frame, split, encoding.spec)
        unseen = sorted(set(source_ids.unique().tolist()).difference(encoding.target_id_remap))
        if unseen:
            raise ValueError(f"Joined {split} split contains targets absent from the target encoding: {unseen}")
        out = frame.copy()
        out[TARGET_ID_COLUMN] = source_ids.map(encoding.target_id_remap).astype(np.int64)
        remapped[split] = out
    return remapped


def target_metadata(
    encoding: TargetEncoding,
    fingerprints: Mapping[str, str] | None = None,
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "target_level": encoding.spec.level,
        "target_column": encoding.spec.source_column,
        "num_classes": encoding.num_classes,
        "canonical_class_ids": list(encoding.canonical_class_ids),
        "target_id_remap": {
            str(canonical_id): int(dense_id)
            for canonical_id, dense_id in encoding.target_id_remap.items()
        },
        "inverse_target_id_remap": {
            str(dense_id): int(canonical_id)
            for dense_id, canonical_id in encoding.inverse_target_id_remap.items()
        },
        "class_names": list(encoding.class_names),
        "split_fingerprints": dict(fingerprints) if fingerprints is not None else None,
    }
    # Existing L3 consumers read this legacy key. Native L2 artifacts intentionally
    # use only the target-aware remap keys above.
    if encoding.spec.level == "l3":
        metadata["label_id_remap"] = dict(metadata["target_id_remap"])
    return metadata


def split_fingerprints(tables: Mapping[str, pd.DataFrame]) -> Dict[str, str]:
    missing_splits = [split for split in SPLIT_NAMES if split not in tables]
    if missing_splits:
        raise ValueError(f"Split fingerprinting requires tables for: {missing_splits}")

    rows_by_split: Dict[str, list[tuple[str, str, int, str]]] = {}
    for split in SPLIT_NAMES:
        frame = tables[split]
        missing_columns = [column for column in FINGERPRINT_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Joined {split} split cannot be fingerprinted; missing columns: {missing_columns}"
            )
        if frame[list(FINGERPRINT_COLUMNS)].isna().any().any():
            raise ValueError(f"Joined {split} split contains missing split-fingerprint values.")

        l2_ids = _coerce_target_ids(frame, split, L2_TARGET_SPEC)
        rows = [
            (str(file_name), str(plot_idx), int(l2_label), str(split_value))
            for file_name, plot_idx, l2_label, split_value in zip(
                frame["file"],
                frame["plot_idx"],
                l2_ids,
                frame["split"],
            )
        ]
        rows_by_split[split] = sorted(rows)

    def digest(rows: Sequence[tuple[str, str, int, str]]) -> str:
        payload = "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in sorted(rows)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    fingerprints = {split: digest(rows_by_split[split]) for split in SPLIT_NAMES}
    fingerprints["combined"] = digest(
        [row for split in SPLIT_NAMES for row in rows_by_split[split]]
    )
    return fingerprints

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal.artifacts import export_image_embeddings
from multimodal.data import (
    GEO_FEATURE_COLUMNS,
    apply_cleaned_test_filter,
    cleaned_test_enabled,
    deduplicate_geo_embeddings,
    image_feature_columns,
    image_embedding_dir,
    joined_table_dir,
    run_dir,
)
from multimodal.labels import (
    build_target_encoding,
    resolve_target_spec,
    split_fingerprints,
    target_metadata,
)
from multimodal.trainer import train_and_evaluate
from multimodal_main import build_joined_tables, load_configs, set_seed


SPLITS = ("train", "val", "test")
IMAGE_SOURCES = ("habitat_finetuned", "pretrained")
PREBUILT_MODES = ("image_only", "geo_only", "raw_concat")
SUMMARY_METRICS = ("loss", "top1_acc", "top3_acc", "f1", "mcc")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full CS 2019-2023 image + 100m GSE baseline suite."
    )
    parser.add_argument("--base_config", type=str, default="configs/multimodal_base.yaml")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/multimodal_cs_geo_100m.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--force_export_image_embeddings", action="store_true")
    parser.add_argument("--force_build_joined_tables", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--summary_dir", type=str, default=None)
    parser.add_argument("--validate_data_only", action="store_true")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def _resolve_cfg_path(value: str | Path, cfg: Mapping) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(str(cfg.get("root_path", "./"))) / path


def _as_list(value) -> list:
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _target_level(cfg: Mapping) -> str:
    return str(cfg.get("multimodal", {}).get("target_level", "l3")).strip().lower()


def _prebuilt_joined_tables_only(cfg: Mapping) -> bool:
    return bool(cfg.get("multimodal", {}).get("prebuilt_joined_tables_only", False))


def _configured_suite_sources(cfg: Mapping) -> list[str]:
    value = cfg.get("multimodal", {}).get("suite_image_sources", IMAGE_SOURCES)
    sources = [str(item).strip().lower() for item in _as_list(value)]
    if not sources or any(not source for source in sources):
        raise ValueError("multimodal.suite_image_sources must contain at least one source")
    if len(sources) != len(set(sources)):
        raise ValueError(f"Duplicate multimodal.suite_image_sources: {sources}")
    unknown = sorted(set(sources).difference(IMAGE_SOURCES))
    if unknown:
        raise ValueError(f"Unsupported multimodal.suite_image_sources: {unknown}")
    return sources


def _validate_prebuilt_settings(cfg: Mapping, force_export: bool, force_join: bool) -> None:
    if not _prebuilt_joined_tables_only(cfg):
        return
    if force_export or force_join:
        raise ValueError(
            "Prebuilt joined-table mode rejects force-export and force-join options"
        )
    mm_cfg = cfg.get("multimodal", {})
    if _target_level(cfg) != "l2":
        raise ValueError(
            "Prebuilt joined-table suite mode requires multimodal.target_level: l2"
        )
    if bool(mm_cfg.get("export_image_embeddings", False)):
        raise ValueError(
            "multimodal.export_image_embeddings must be false in prebuilt joined-table mode"
        )
    if bool(mm_cfg.get("build_joined_tables", False)):
        raise ValueError(
            "multimodal.build_joined_tables must be false in prebuilt joined-table mode"
        )
    sources = _configured_suite_sources(cfg)
    if sources != ["habitat_finetuned"]:
        raise ValueError(
            "Prebuilt joined-table mode requires exactly "
            "multimodal.suite_image_sources: [habitat_finetuned]"
        )


def _loaded_inventory_paths(cfg: Mapping, split: str) -> list[Path]:
    data_cfg = cfg["data"]
    if split == "train":
        folders = _as_list(data_cfg["dataset_paths"])
        index_names = _as_list(data_cfg["index_file_names"])
    elif split == "test":
        folders = _as_list(data_cfg["test_dataset_paths"])
        index_names = _as_list(data_cfg["test_index_file_names"])
    else:
        raise ValueError(f"Unsupported loaded-inventory split: {split}")
    if len(folders) != len(index_names):
        raise ValueError(
            f"Mismatched {split} folders/index files: {len(folders)} != {len(index_names)}"
        )
    return [
        _resolve_cfg_path(Path(str(folder)) / f"loaded_{Path(str(index_name)).name}", cfg)
        for folder, index_name in zip(folders, index_names)
    ]


def _load_inventory(cfg: Mapping, split: str) -> pd.DataFrame:
    frames = []
    for path in _loaded_inventory_paths(cfg, split):
        if not path.exists():
            raise FileNotFoundError(f"Loaded {split} inventory not found: {path}")
        frame = pd.read_csv(path, dtype={"plot_idx": "string"})
        required = {"file_names", "plot_word_labels", "plot_idx"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Loaded {split} inventory {path} is missing: {sorted(missing)}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _normalized_keys(values: Sequence) -> pd.Series:
    return pd.Series(values, dtype="string").str.strip().str.lower()


def _assert_unique_nonempty_keys(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    keys = _normalized_keys(frame[column])
    blank = keys.isna() | (keys.fillna("") == "")
    if bool(blank.any()):
        raise ValueError(f"{name} contains {int(blank.sum())} blank file keys")
    duplicates = keys.duplicated(keep=False)
    if bool(duplicates.any()):
        preview = keys[duplicates].head(20).tolist()
        raise ValueError(f"{name} contains duplicate file keys: {preview}")
    return keys


def validate_source_data(cfg: Mapping) -> Dict[str, object]:
    train = _load_inventory(cfg, "train")
    test = _load_inventory(cfg, "test")
    for split, frame in (("train", train), ("test", test)):
        frame["file_lower"] = _assert_unique_nonempty_keys(
            frame, "file_names", f"loaded {split} inventory"
        )
    test, cleaned_manifest = apply_cleaned_test_filter(
        cfg,
        "test",
        test,
        file_column="file_names",
    )
    test["file_lower"] = _assert_unique_nonempty_keys(
        test,
        "file_names",
        "effective loaded test inventory",
    )

    file_overlap = set(train["file_lower"]).intersection(test["file_lower"])
    if file_overlap:
        raise ValueError(f"Train/test loaded filename overlap: {sorted(file_overlap)[:20]}")
    train_ids = set(train["plot_idx"].astype("string").str.strip())
    test_ids = set(test["plot_idx"].astype("string").str.strip())
    id_overlap = train_ids.intersection(test_ids)
    if id_overlap:
        raise ValueError(f"Train/test plot-ID overlap: {sorted(id_overlap)[:20]}")

    geo_path = _resolve_cfg_path(cfg["multimodal"]["geo_embeddings_path"], cfg)
    if not geo_path.exists():
        raise FileNotFoundError(f"100m GSE parquet not found: {geo_path}")
    geo_raw = pd.read_parquet(geo_path)
    geo, dedup_stats = deduplicate_geo_embeddings(geo_raw)
    if int(dedup_stats["duplicate_groups"]) != 7:
        raise ValueError(
            "Expected seven duplicate 100m GSE filename groups, found "
            f"{dedup_stats['duplicate_groups']}"
        )
    features = geo[GEO_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("100m GSE features contain NaN or infinite values")
    if bool((np.linalg.norm(features, axis=1) == 0).any()):
        raise ValueError("100m GSE features contain zero-norm rows")

    geo["file_lower"] = _normalized_keys(geo["file"])
    geo_keys = set(geo["file_lower"])
    split_results: Dict[str, dict] = {}
    for split, image_frame in (("train", train), ("test", test)):
        image_keys = set(image_frame["file_lower"])
        missing = sorted(image_keys.difference(geo_keys))
        if missing:
            raise ValueError(f"100m GSE is missing {split} files: {missing[:20]}")
        merged = image_frame.merge(
            geo[["file_lower", "ID", "BH_PLOT_DESC"]],
            on="file_lower",
            how="left",
            validate="one_to_one",
        )
        id_match = (
            merged["plot_idx"].astype("string").str.strip()
            == merged["ID"].astype("string").str.strip()
        )
        if not bool(id_match.all()):
            raise ValueError(f"100m GSE contains {int((~id_match).sum())} {split} ID mismatches")
        label_match = (
            merged["plot_word_labels"].fillna("").astype(str).str.strip()
            == merged["BH_PLOT_DESC"].fillna("").astype(str).str.strip()
        )
        if not bool(label_match.all()):
            raise ValueError(
                f"100m GSE contains {int((~label_match).sum())} {split} label mismatches"
            )
        split_results[split] = {
            "loaded_rows": int(len(image_frame)),
            "matched_rows": int(len(merged)),
            "missing_rows": 0,
        }
        if split == "test" and cleaned_manifest is not None:
            split_results[split]["cleaned_test"] = cleaned_manifest

    return {
        "geo_path": str(geo_path),
        "geo_feature_dim": int(len(GEO_FEATURE_COLUMNS)),
        "geo_input_rows": int(len(geo_raw)),
        "geo_unique_rows": int(len(geo)),
        "duplicate_groups": int(dedup_stats["duplicate_groups"]),
        "preferred_nonempty_groups": int(dedup_stats["preferred_nonempty_groups"]),
        "extra_geo_rows": int(len(geo_keys.difference(set(train["file_lower"]) | set(test["file_lower"])))),
        "train_test_file_overlap": 0,
        "train_test_id_overlap": 0,
        "splits": split_results,
    }


def _source_cfg(base_cfg: Mapping, seed: int, image_source: str) -> dict:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg["seed"] = int(seed)
    mm_cfg = cfg.setdefault("multimodal", {})
    joined_table_tag = mm_cfg.get("joined_table_tag", "gse_100m")
    run_tag = mm_cfg.get("run_tag", joined_table_tag)
    mm_cfg["image_feature_source"] = str(image_source)
    mm_cfg["joined_table_tag"] = joined_table_tag
    mm_cfg["run_tag"] = run_tag
    mm_cfg["report_test_each_epoch"] = False
    return cfg


def _run_cfg(source_cfg: Mapping, mode: str) -> dict:
    cfg = copy.deepcopy(dict(source_cfg))
    mm_cfg = cfg.setdefault("multimodal", {})
    mm_cfg["fusion_mode"] = str(mode)
    mm_cfg["export_image_embeddings"] = False
    mm_cfg["build_joined_tables"] = False
    mm_cfg["train_classifier"] = True
    return cfg


def suite_run_configs(base_cfg: Mapping, seeds: Sequence[int]) -> list[dict]:
    configs = []
    if _prebuilt_joined_tables_only(base_cfg):
        _validate_prebuilt_settings(base_cfg, force_export=False, force_join=False)
        source = _configured_suite_sources(base_cfg)[0]
        for seed in seeds:
            source_cfg = _source_cfg(base_cfg, seed, source)
            configs.extend(_run_cfg(source_cfg, mode) for mode in PREBUILT_MODES)
        return configs

    for seed in seeds:
        fine = _source_cfg(base_cfg, seed, "habitat_finetuned")
        pretrained = _source_cfg(base_cfg, seed, "pretrained")
        configs.extend(
            [
                _run_cfg(fine, "image_only"),
                _run_cfg(pretrained, "image_only"),
                _run_cfg(fine, "geo_only"),
                _run_cfg(fine, "raw_concat"),
                _run_cfg(pretrained, "raw_concat"),
            ]
        )
    return configs


def _split_parquets_exist(root: Path) -> bool:
    return all((root / f"{split}.parquet").exists() for split in SPLITS)


def _joined_artifacts_exist(root: Path) -> bool:
    return _split_parquets_exist(root) and all(
        (root / f"{split}_manifest.json").exists() for split in SPLITS
    )


def _joined_row_counts(cfg: Mapping) -> Dict[str, int] | None:
    root = joined_table_dir(cfg)
    counts: Dict[str, int] = {}
    for split in SPLITS:
        table_path = root / f"{split}.parquet"
        if not table_path.exists():
            return None
        counts[split] = int(len(pd.read_parquet(table_path, columns=["file"])))
    return counts


def _validate_cleaned_test_manifest(cfg: Mapping, manifest: Mapping, row_count: int) -> None:
    cleaned = manifest.get("cleaned_test", None)
    if cleaned_test_enabled(cfg) and not isinstance(cleaned, Mapping):
        raise ValueError("Cleaned test is enabled, but test manifest has no cleaned_test metadata")
    if not isinstance(cleaned, Mapping):
        return

    required = {"input_rows", "removed_rows", "output_rows"}
    missing = required.difference(cleaned)
    if missing:
        raise ValueError(f"Cleaned test manifest missing keys: {sorted(missing)}")
    input_rows = int(cleaned["input_rows"])
    removed_rows = int(cleaned["removed_rows"])
    output_rows = int(cleaned["output_rows"])
    image_rows = int(manifest.get("image_rows", -1))
    if input_rows - removed_rows != output_rows:
        raise ValueError(f"Cleaned test manifest row counts are inconsistent: {cleaned}")
    if output_rows != row_count or image_rows != row_count:
        raise ValueError(
            "Cleaned test manifest row counts do not match the joined table: "
            f"cleaned_output={output_rows}, image_rows={image_rows}, table_rows={row_count}"
        )


def validate_joined_artifacts(cfg: Mapping) -> Dict[str, dict]:
    root = joined_table_dir(cfg)
    results = {}
    for split in SPLITS:
        table_path = root / f"{split}.parquet"
        manifest_path = root / f"{split}_manifest.json"
        if not table_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Incomplete joined artifacts for split={split}: {root}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        row_count = int(len(pd.read_parquet(table_path, columns=["file"])))
        if int(manifest.get("dropped_rows", -1)) != 0:
            raise ValueError(f"Joined {split} manifest reports dropped rows: {manifest}")
        if int(manifest.get("matched_rows", -1)) != row_count:
            raise ValueError(f"Joined {split} manifest matched-row mismatch: {manifest}")
        if split == "test":
            _validate_cleaned_test_manifest(cfg, manifest, row_count)
        results[split] = {
            "rows": row_count,
            "dropped_rows": 0,
            "table": str(table_path),
        }
    return results


def _load_joined_tables(cfg: Mapping) -> Dict[str, pd.DataFrame]:
    root = joined_table_dir(cfg)
    return {
        split: pd.read_parquet(root / f"{split}.parquet")
        for split in SPLITS
    }


def _finite_feature_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    split: str,
    modality: str,
) -> None:
    try:
        values = frame[list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Prebuilt {split} {modality} features must be numeric"
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(
            f"Prebuilt {split} {modality} features contain NaN or infinite values"
        )


def _strict_manifest_dimensions(
    cfg: Mapping,
    split: str,
    row_count: int,
    image_dim: int,
) -> None:
    manifest_path = joined_table_dir(cfg) / f"{split}_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "split": split,
        "image_rows": int(row_count),
        "matched_rows": int(row_count),
        "dropped_rows": 0,
        "image_feature_dim": int(image_dim),
        "geo_feature_dim": int(len(GEO_FEATURE_COLUMNS)),
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Prebuilt {split} manifest does not match its joined table: {mismatches}"
        )


def validate_prebuilt_joined_artifacts(cfg: Mapping) -> Dict[str, object]:
    """Validate immutable joined parquets without rebuilding their source artifacts."""
    results: Dict[str, object] = dict(validate_joined_artifacts(cfg))
    tables = _load_joined_tables(cfg)
    image_columns: list[str] | None = None
    file_keys: Dict[str, set[str]] = {}
    plot_keys: Dict[str, set[str]] = {}
    target_sets: Dict[str, set[int]] = {}

    for split, frame in tables.items():
        if frame.empty:
            raise ValueError(f"Prebuilt joined {split} table is empty")
        required = {"file", "plot_idx", "l2_label", *GEO_FEATURE_COLUMNS}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Prebuilt joined {split} table is missing required columns: {missing}"
            )

        current_image_columns = image_feature_columns(frame)
        if not current_image_columns:
            raise ValueError(
                f"Prebuilt joined {split} table has no image feature columns (I*)"
            )
        if image_columns is None:
            image_columns = current_image_columns
        elif current_image_columns != image_columns:
            raise ValueError(
                f"Prebuilt joined {split} image feature schema differs across splits"
            )

        normalized_files = _assert_unique_nonempty_keys(
            frame, "file", f"prebuilt joined {split} table"
        )
        normalized_plots = _normalized_keys(frame["plot_idx"])
        blank_plots = normalized_plots.isna() | (normalized_plots.fillna("") == "")
        if bool(blank_plots.any()):
            raise ValueError(
                f"Prebuilt joined {split} table contains "
                f"{int(blank_plots.sum())} blank plot keys"
            )

        l2_numeric = pd.to_numeric(frame["l2_label"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(l2_numeric).all():
            raise ValueError(
                f"Prebuilt joined {split} table contains missing or invalid L2 labels"
            )
        if not np.equal(l2_numeric, np.floor(l2_numeric)).all():
            raise ValueError(
                f"Prebuilt joined {split} table contains non-integral L2 labels"
            )
        if bool((l2_numeric < 0).any()):
            raise ValueError(
                f"Prebuilt joined {split} table contains negative L2 labels"
            )

        if "split" in frame.columns:
            embedded_splits = {
                str(value).strip().lower() for value in frame["split"].dropna().unique()
            }
            if embedded_splits != {split}:
                raise ValueError(
                    f"Prebuilt joined {split} table has inconsistent split values: "
                    f"{sorted(embedded_splits)}"
                )

        _finite_feature_matrix(frame, current_image_columns, split, "image")
        _finite_feature_matrix(frame, GEO_FEATURE_COLUMNS, split, "geo")
        _strict_manifest_dimensions(cfg, split, len(frame), len(current_image_columns))

        file_keys[split] = set(normalized_files.tolist())
        plot_keys[split] = set(normalized_plots.tolist())
        target_sets[split] = {int(value) for value in l2_numeric.tolist()}
        split_result = results[split]
        assert isinstance(split_result, dict)
        split_result.update(
            {
                "image_feature_dim": int(len(current_image_columns)),
                "geo_feature_dim": int(len(GEO_FEATURE_COLUMNS)),
                "l2_labels": sorted(target_sets[split]),
            }
        )

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            file_overlap = sorted(file_keys[left].intersection(file_keys[right]))
            if file_overlap:
                raise ValueError(
                    f"Prebuilt joined {left}/{right} filename overlap: {file_overlap[:20]}"
                )
            plot_overlap = sorted(plot_keys[left].intersection(plot_keys[right]))
            if plot_overlap:
                raise ValueError(
                    f"Prebuilt joined {left}/{right} plot-ID overlap: {plot_overlap[:20]}"
                )

    train_targets = target_sets["train"]
    for split in ("val", "test"):
        unseen = sorted(target_sets[split].difference(train_targets))
        if unseen:
            raise ValueError(
                f"Prebuilt joined {split} table contains L2 labels absent from train: {unseen}"
            )

    fingerprints = split_fingerprints(tables)
    target_spec = resolve_target_spec(cfg)
    encoding = build_target_encoding(
        tables,
        target_spec,
        training_split_name="prebuilt joined train split",
    )
    results["split_fingerprints"] = fingerprints
    results["target_metadata"] = target_metadata(encoding, fingerprints)
    return results


def prepare_source_artifacts(
    cfg: Mapping,
    force_export: bool = False,
    force_join: bool = False,
) -> Dict[str, object]:
    if _prebuilt_joined_tables_only(cfg):
        _validate_prebuilt_settings(cfg, force_export=force_export, force_join=force_join)
        table_dir = joined_table_dir(cfg)
        print(f"\nValidating immutable prebuilt joined tables: {table_dir}")
        return validate_prebuilt_joined_artifacts(cfg)

    image_dir = image_embedding_dir(cfg)
    if force_export or not _split_parquets_exist(image_dir):
        print(f"\nExporting {cfg['multimodal']['image_feature_source']} image embeddings")
        export_image_embeddings(cfg)
    else:
        print(f"\nReusing image embeddings: {image_dir}")

    table_dir = joined_table_dir(cfg)
    rebuild_join = bool(force_export or force_join or not _joined_artifacts_exist(table_dir))
    if rebuild_join:
        print(f"Building 100m joined tables: {table_dir}")
        build_joined_tables(cfg)
    else:
        print(f"Reusing 100m joined tables: {table_dir}")
    return validate_joined_artifacts(cfg)


def _best_history_entry(history: Sequence[Mapping]) -> Mapping:
    if not history:
        return {}
    return max(
        history,
        key=lambda row: float(row.get("val_top1_acc", row.get("top1_acc", -1.0))),
    )


def _expected_target_metadata(cfg: Mapping) -> Dict[str, object]:
    tables = _load_joined_tables(cfg)
    target_spec = resolve_target_spec(cfg)
    encoding = build_target_encoding(
        tables,
        target_spec,
        training_split_name="prebuilt joined train split",
    )
    return target_metadata(encoding, split_fingerprints(tables))


def _json_normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _load_completed_metrics(cfg: Mapping) -> tuple[Path, dict] | None:
    output_dir = run_dir(cfg)
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists() or not (output_dir / "best_model.pt").exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    required = {"mode", "train_rows", "val_rows", "test_rows", "history", "val", "test"}
    if not required.issubset(metrics):
        return None
    if str(metrics["mode"]) != str(cfg["multimodal"]["fusion_mode"]):
        return None
    configured_target = _target_level(cfg)
    persisted_target = str(metrics.get("target_level", "l3")).strip().lower()
    if persisted_target != configured_target:
        return None
    row_counts = _joined_row_counts(cfg)
    if row_counts is None:
        return None
    for split in SPLITS:
        if int(metrics[f"{split}_rows"]) != row_counts[split]:
            return None
    if not all(metric in metrics["val"] and metric in metrics["test"] for metric in SUMMARY_METRICS):
        return None
    if configured_target == "l2":
        expected_metadata = _expected_target_metadata(cfg)
        required_target_keys = {
            "target_level",
            "target_column",
            "num_classes",
            "canonical_class_ids",
            "target_id_remap",
            "inverse_target_id_remap",
            "class_names",
            "split_fingerprints",
        }
        if not required_target_keys.issubset(metrics):
            return None
        for key in required_target_keys:
            if _json_normalized(metrics[key]) != _json_normalized(expected_metadata[key]):
                return None
    return metrics_path, metrics


def _metrics_from_outputs(outputs: Mapping[str, Path]) -> tuple[Path, dict]:
    metrics_path = Path(outputs["metrics"])
    with metrics_path.open("r", encoding="utf-8") as handle:
        return metrics_path, json.load(handle)


def summarize_run(cfg: Mapping, metrics_path: Path, metrics: Mapping) -> dict:
    history = metrics.get("history", [])
    best = _best_history_entry(history)
    source = str(cfg["multimodal"]["image_feature_source"])
    mode = str(cfg["multimodal"]["fusion_mode"])
    display_source = "shared_geo" if mode == "geo_only" else source
    row = {
        "image_source": display_source,
        "artifact_image_source": source,
        "mode": mode,
        "seed": int(cfg["seed"]),
        "train_rows": int(metrics["train_rows"]),
        "val_rows": int(metrics["val_rows"]),
        "test_rows": int(metrics["test_rows"]),
        "epochs_run": int(len(history)),
        "best_val_epoch": best.get("epoch"),
        "run_dir": str(run_dir(cfg)),
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(run_dir(cfg) / "best_model.pt"),
    }
    for split in ("val", "test"):
        for metric in SUMMARY_METRICS:
            row[f"{split}_{metric}"] = float(metrics[split][metric])
    return row


def aggregate_rows(rows: Sequence[Mapping]) -> list[dict]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    metric_columns = [
        f"{split}_{metric}" for split in ("val", "test") for metric in SUMMARY_METRICS
    ]
    aggregates = []
    for (source, mode), group in frame.groupby(["image_source", "mode"], sort=True):
        row = {
            "image_source": str(source),
            "mode": str(mode),
            "n": int(len(group)),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        aggregates.append(row)
    return aggregates


def _default_summary_dir(cfg: Mapping) -> Path:
    output = Path(str(cfg.get("multimodal", {}).get("output_dir", "./multimodal_artifacts")))
    if not output.is_absolute():
        output = Path(str(cfg.get("root_path", "./"))) / output
    tag = str(cfg.get("multimodal", {}).get("joined_table_tag", "gse_100m"))
    if _target_level(cfg) == "l2":
        return output / "reports" / str(cfg.get("dataset", "cs")) / "target_l2" / tag
    return output / "reports" / str(cfg.get("dataset", "cs")) / tag


def write_summaries(rows: Sequence[Mapping], summary_dir: Path) -> Dict[str, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(rows).sort_values(["mode", "image_source", "seed"]).reset_index(drop=True)
    aggregate = pd.DataFrame(aggregate_rows(rows))
    paths = {
        "per_run_csv": summary_dir / "per_run.csv",
        "per_run_json": summary_dir / "per_run.json",
        "aggregate_csv": summary_dir / "aggregate.csv",
        "aggregate_json": summary_dir / "aggregate.json",
    }
    per_run.to_csv(paths["per_run_csv"], index=False)
    per_run.to_json(paths["per_run_json"], orient="records", indent=2)
    aggregate.to_csv(paths["aggregate_csv"], index=False)
    aggregate.to_json(paths["aggregate_json"], orient="records", indent=2)
    return paths


def _print_suite(cfg: Mapping, seeds: Sequence[int]) -> None:
    print("\n==== CS 2019-2023 + 100m GSE Suite ====")
    print("Geo parquet:", _resolve_cfg_path(cfg["multimodal"]["geo_embeddings_path"], cfg))
    print("Seeds:", list(seeds))
    for run_cfg in suite_run_configs(cfg, seeds):
        source = run_cfg["multimodal"]["image_feature_source"]
        mode = run_cfg["multimodal"]["fusion_mode"]
        print(f"seed={run_cfg['seed']} source={source} mode={mode} run_dir={run_dir(run_cfg)}")


def main():
    args = parse_args()
    cfg = load_configs(args)
    seeds = sorted({int(seed) for seed in args.seeds})
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError(f"Seeds must be non-negative integers: {seeds}")
    prebuilt_only = _prebuilt_joined_tables_only(cfg)
    _validate_prebuilt_settings(
        cfg,
        force_export=bool(args.force_export_image_embeddings),
        force_join=bool(args.force_build_joined_tables),
    )

    _print_suite(cfg, seeds)
    if args.inspect_only:
        return

    if prebuilt_only:
        validation = {}
        source = _configured_suite_sources(cfg)[0]
        for seed in seeds:
            source_cfg = _source_cfg(cfg, seed, source)
            set_seed(seed)
            validation[f"seed{seed}"] = prepare_source_artifacts(source_cfg)
        validation_heading = "Immutable Prebuilt Joined-Table Validation"
    else:
        validation = validate_source_data(cfg)
        validation_heading = "100m Source Data Validation"
    print(f"\n==== {validation_heading} ====")
    print(json.dumps(validation, indent=2))
    if args.validate_data_only:
        return

    summary_dir = Path(args.summary_dir) if args.summary_dir else _default_summary_dir(cfg)
    if not summary_dir.is_absolute():
        summary_dir = PROJECT_ROOT / summary_dir

    rows = []
    for seed in seeds:
        if prebuilt_only:
            run_cfgs = suite_run_configs(cfg, [seed])
        else:
            source_cfgs = {
                source: _source_cfg(cfg, seed, source) for source in IMAGE_SOURCES
            }
            for source_cfg in source_cfgs.values():
                set_seed(seed)
                prepare_source_artifacts(
                    source_cfg,
                    force_export=bool(args.force_export_image_embeddings),
                    force_join=bool(args.force_build_joined_tables),
                )

            run_cfgs = [
                _run_cfg(source_cfgs["habitat_finetuned"], "image_only"),
                _run_cfg(source_cfgs["pretrained"], "image_only"),
                _run_cfg(source_cfgs["habitat_finetuned"], "geo_only"),
                _run_cfg(source_cfgs["habitat_finetuned"], "raw_concat"),
                _run_cfg(source_cfgs["pretrained"], "raw_concat"),
            ]
        for run_cfg in run_cfgs:
            set_seed(seed)
            completed = None if args.force_train else _load_completed_metrics(run_cfg)
            if completed is None:
                print(
                    f"\nTraining seed={seed} "
                    f"source={run_cfg['multimodal']['image_feature_source']} "
                    f"mode={run_cfg['multimodal']['fusion_mode']}"
                )
                metrics_path, metrics = _metrics_from_outputs(train_and_evaluate(run_cfg))
            else:
                metrics_path, metrics = completed
                print(f"\nReusing completed run: {metrics_path}")
            rows.append(summarize_run(run_cfg, metrics_path, metrics))
            write_summaries(rows, summary_dir)

    paths = write_summaries(rows, summary_dir)
    print("\n==== Suite Reports ====")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

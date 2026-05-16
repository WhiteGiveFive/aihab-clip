from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal.data import image_embedding_dir, joined_table_dir, run_dir
from multimodal.soil import build_soil_feature_tables, export_soil_image_embeddings
from multimodal.trainer import train_and_evaluate
from multimodal_main import load_configs, set_seed


SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search CS2007 image + soil projection dimensions."
    )
    parser.add_argument("--base_config", type=str, default="configs/multimodal_base.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/multimodal_cs2007_soil.yaml")
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128],
        help="Projection output dimensions to evaluate for the soil MLP branch.",
    )
    parser.add_argument(
        "--force_export_image_embeddings",
        action="store_true",
        help="Recompute image embedding parquets before the grid search.",
    )
    parser.add_argument(
        "--force_build_joined_tables",
        action="store_true",
        help="Rebuild soil multimodal parquet tables before the grid search.",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=None,
        help="Optional directory for grid_summary.csv/json. Defaults under the soil_projected_concat run root.",
    )
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def _split_parquets_exist(root: Path) -> bool:
    return all((root / f"{split}.parquet").exists() for split in SPLITS)


def _prepare_shared_artifacts(cfg: Mapping, force_export: bool, force_join: bool) -> None:
    image_dir = image_embedding_dir(cfg)
    if force_export or not _split_parquets_exist(image_dir):
        print("\n==== Exporting Soil-Aligned Image Embeddings ====")
        for split, path in export_soil_image_embeddings(cfg).items():
            print(f"{split}: {path}")
    else:
        print(f"\nReusing image embeddings: {image_dir}")

    table_dir = joined_table_dir(cfg)
    if force_join or not _split_parquets_exist(table_dir):
        print("\n==== Building Soil Multimodal Tables ====")
        for split, path in build_soil_feature_tables(cfg).items():
            print(f"{split}: {path}")
    else:
        print(f"Reusing joined soil tables: {table_dir}")


def _grid_cfg(base_cfg: Mapping, projection_dim: int):
    cfg = copy.deepcopy(base_cfg)
    mm_cfg = cfg.setdefault("multimodal", {})
    mm_cfg["fusion_mode"] = "soil_projected_concat"
    mm_cfg["tabular_modality_name"] = "soil"
    mm_cfg["tabular_encoder"] = "mlp_projection"
    mm_cfg["tabular_projection_dim"] = int(projection_dim)
    mm_cfg["run_tag"] = f"projdim_{int(projection_dim)}"
    mm_cfg["export_image_embeddings"] = False
    mm_cfg["build_joined_tables"] = False
    mm_cfg["train_classifier"] = True
    return cfg


def _best_history_entry(history: list[dict]) -> dict:
    if not history:
        return {}
    return max(history, key=lambda row: float(row.get("val_top1_acc", row.get("top1_acc", -1.0))))


def _summarize_run(projection_dim: int, outputs: Mapping[str, Path]) -> dict:
    metrics_path = Path(outputs["metrics"])
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    history = metrics.get("history", [])
    best = _best_history_entry(history)
    val = metrics.get("val", {})
    test = metrics.get("test", {})
    return {
        "projection_dim": int(projection_dim),
        "mode": metrics.get("mode", "soil_projected_concat"),
        "epochs_run": int(len(history)),
        "best_val_epoch": best.get("epoch"),
        "best_history_val_top1_acc": best.get("val_top1_acc", best.get("top1_acc")),
        "best_history_test_top1_acc": best.get("test_top1_acc"),
        "val_loss": val.get("loss"),
        "val_top1_acc": val.get("top1_acc"),
        "val_top3_acc": val.get("top3_acc"),
        "val_f1": val.get("f1"),
        "val_mcc": val.get("mcc"),
        "test_loss": test.get("loss"),
        "test_top1_acc": test.get("top1_acc"),
        "test_top3_acc": test.get("top3_acc"),
        "test_f1": test.get("f1"),
        "test_mcc": test.get("mcc"),
        "run_dir": str(outputs["run_dir"]),
        "checkpoint": str(outputs["checkpoint"]),
        "metrics": str(outputs["metrics"]),
    }


def _default_summary_dir(first_cfg: Mapping) -> Path:
    first_run_dir = run_dir(first_cfg)
    return first_run_dir.parent.parent / "projection_dim_grid" / first_run_dir.name


def _write_summary(rows: list[dict], summary_dir: Path) -> dict[str, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["projection_dim"]).reset_index(drop=True)
    csv_path = summary_dir / "grid_summary.csv"
    json_path = summary_dir / "grid_summary.json"
    frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return {"csv": csv_path, "json": json_path}


def main():
    args = parse_args()
    cfg = load_configs(args)
    set_seed(int(cfg.get("seed", 1)))

    dims = sorted({int(dim) for dim in args.dims})
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"Projection dimensions must be positive integers: {dims}")

    print("\n==== CS2007 Soil Projection Grid ====")
    print(f"dims: {dims}")
    print("Image embedding dir:", image_embedding_dir(cfg))
    print("Joined table dir:", joined_table_dir(cfg))

    first_cfg = _grid_cfg(cfg, dims[0])
    summary_dir = Path(args.summary_dir) if args.summary_dir else _default_summary_dir(first_cfg)
    if not summary_dir.is_absolute():
        summary_dir = PROJECT_ROOT / summary_dir
    print("Summary dir:", summary_dir)

    if args.inspect_only:
        for dim in dims:
            dim_cfg = _grid_cfg(cfg, dim)
            print(f"dim={dim}: run_dir={run_dir(dim_cfg)}")
        return

    _prepare_shared_artifacts(
        cfg,
        force_export=bool(args.force_export_image_embeddings),
        force_join=bool(args.force_build_joined_tables),
    )

    rows = []
    for dim in dims:
        run_cfg = _grid_cfg(cfg, dim)
        set_seed(int(run_cfg.get("seed", 1)))
        print(f"\n==== Training soil_projected_concat with projection_dim={dim} ====")
        print("Run dir:", run_dir(run_cfg))
        outputs = train_and_evaluate(run_cfg)
        rows.append(_summarize_run(dim, outputs))
        summary_paths = _write_summary(rows, summary_dir)
        print(f"Updated summary CSV: {summary_paths['csv']}")

    summary_paths = _write_summary(rows, summary_dir)
    print("\n==== Projection Grid Outputs ====")
    for name, path in summary_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal.data import image_embedding_dir, joined_table_dir, run_dir
from multimodal.soil import build_soil_feature_tables, export_soil_image_embeddings
from multimodal.trainer import train_and_evaluate
from multimodal_main import load_configs, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run CS2007 image + soil multimodal baselines.")
    parser.add_argument("--base_config", type=str, default="configs/multimodal_base.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/multimodal_cs2007_soil.yaml")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--inspect_only", action="store_true")
    return parser.parse_args()


def inspect(cfg):
    print("\n==== CS2007 Soil Multimodal Config ====")
    print(cfg)
    print("\nImage embedding dir:", image_embedding_dir(cfg))
    print("Joined table dir:", joined_table_dir(cfg))
    print("Run dir:", run_dir(cfg))


def main():
    args = parse_args()
    cfg = load_configs(args)
    set_seed(int(cfg.get("seed", 1)))
    inspect(cfg)
    if args.inspect_only:
        return

    mm_cfg = cfg.get("multimodal", {})
    if bool(mm_cfg.get("export_image_embeddings", True)):
        image_paths = export_soil_image_embeddings(cfg)
        print("\n==== Soil-Aligned Image Embeddings ====")
        for split, path in image_paths.items():
            print(f"{split}: {path}")

    if bool(mm_cfg.get("build_joined_tables", True)):
        joined_paths = build_soil_feature_tables(cfg)
        print("\n==== Soil Multimodal Tables ====")
        for split, path in joined_paths.items():
            print(f"{split}: {path}")

    if bool(mm_cfg.get("train_classifier", True)):
        outputs = train_and_evaluate(cfg)
        print("\n==== Soil Multimodal Training Outputs ====")
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()

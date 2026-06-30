from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal.data import image_embedding_dir, joined_table_dir, run_dir
from multimodal.geo_10m import (
    build_geo_10m_feature_tables,
    export_geo_10m_image_embeddings,
    inspect_geo_10m_splits,
)
from multimodal.trainer import train_and_evaluate
from multimodal_main import load_configs, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run CS 10m curated image + geo multimodal baselines.")
    parser.add_argument("--base_config", type=str, default="configs/multimodal_base.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/multimodal_cs_geo_10m.yaml")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--inspect_only", action="store_true")
    return parser.parse_args()


def inspect(cfg):
    print("\n==== CS 10m Geo Multimodal Config ====")
    print(cfg)
    print("\nImage embedding dir:", image_embedding_dir(cfg))
    print("Joined table dir:", joined_table_dir(cfg))
    print("Run dir:", run_dir(cfg))
    print("\n==== CS 10m Curated Split Summary ====")
    for split, info in inspect_geo_10m_splits(cfg).items():
        print(f"{split}: rows={info['rows']} id_groups={info['id_groups']}")
        for label, count in info["labels"].items():
            print(f"  {label}: {count}")


def main():
    args = parse_args()
    cfg = load_configs(args)
    set_seed(int(cfg.get("seed", 1)))
    inspect(cfg)
    if args.inspect_only:
        return

    mm_cfg = cfg.get("multimodal", {})
    if bool(mm_cfg.get("export_image_embeddings", True)):
        image_paths = export_geo_10m_image_embeddings(cfg)
        print("\n==== CS 10m Image Embeddings ====")
        for split, path in image_paths.items():
            print(f"{split}: {path}")

    if bool(mm_cfg.get("build_joined_tables", True)):
        joined_paths = build_geo_10m_feature_tables(cfg)
        print("\n==== CS 10m Geo Multimodal Tables ====")
        for split, path in joined_paths.items():
            print(f"{split}: {path}")

    if bool(mm_cfg.get("train_classifier", True)):
        outputs = train_and_evaluate(cfg)
        print("\n==== CS 10m Geo Multimodal Training Outputs ====")
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()

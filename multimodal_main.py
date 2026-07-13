from __future__ import annotations

import argparse
import copy
import random
from ast import literal_eval
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from multimodal.artifacts import export_image_embeddings
from multimodal.data import (
    apply_cleaned_test_filter,
    image_embedding_dir,
    joined_table_dir,
    join_split_with_geo,
    load_geo_embeddings,
    save_join_artifacts,
)
from multimodal.trainer import train_and_evaluate
from utils import CfgNode, load_cfg_from_cfg_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/multimodal_base.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/multimodal_cs.yaml")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--inspect_only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_cfg_path(path: str) -> str:
    cand = Path(path)
    if cand.is_file():
        return str(cand)
    here = Path(__file__).parent
    for probe in (here / path, here.parent / path):
        if probe.is_file():
            return str(probe)
    raise FileNotFoundError(f"Config not found: {path}")


def _pairwise_overrides(opts: Iterable[str]) -> dict[str, str]:
    if not opts:
        return {}
    opts = list(opts)
    if len(opts) % 2 != 0:
        raise ValueError(f"Unpaired override list: {opts}")
    return {str(k): str(v) for k, v in zip(opts[0::2], opts[1::2])}


def _decode(value: str):
    try:
        return literal_eval(value)
    except Exception:
        return value


def _get_cfg_value(cfg: Any, path_parts: list[str]):
    node = cfg
    for part in path_parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _coerce_value(raw: str, ref):
    decoded = _decode(raw)
    if ref is None:
        return decoded
    if isinstance(ref, bool):
        return str(raw).lower() in {"1", "true", "t", "yes", "y"}
    if isinstance(ref, int) and not isinstance(ref, bool):
        return int(decoded)
    if isinstance(ref, float):
        return float(decoded)
    if isinstance(ref, list) and not isinstance(decoded, list):
        return [decoded]
    return decoded


def _set_cfg_value(cfg: CfgNode, path_parts: list[str], value):
    node = cfg
    for part in path_parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = CfgNode()
        node = node[part]
    node[path_parts[-1]] = value


def _deep_merge(dst: CfgNode, src: CfgNode):
    for key, value in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def load_configs(args) -> CfgNode:
    base = load_cfg_from_cfg_file(_resolve_cfg_path(args.base_config))
    ds = load_cfg_from_cfg_file(_resolve_cfg_path(args.dataset_config))
    merged = _deep_merge(copy.deepcopy(base), ds)

    for key, raw_value in _pairwise_overrides(args.opts).items():
        parts = key.split(".")
        ref = _get_cfg_value(merged, parts)
        _set_cfg_value(merged, parts, _coerce_value(raw_value, ref))
    return merged


def build_joined_tables(cfg) -> dict[str, Path]:
    out_dir = joined_table_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    geo_df, geo_stats = load_geo_embeddings(cfg)
    image_dir = image_embedding_dir(cfg)
    outputs = {}

    for split in ("train", "val", "test"):
        image_df = pd.read_parquet(image_dir / f"{split}.parquet")
        image_df, cleaned_manifest = apply_cleaned_test_filter(cfg, split, image_df, file_column="file")
        joined_df, manifest, dropped_df = join_split_with_geo(image_df, geo_df)
        manifest["geo_dedup_stats"] = geo_stats
        if cleaned_manifest is not None:
            manifest["cleaned_test"] = cleaned_manifest
        outputs[split] = save_join_artifacts(split, joined_df, manifest, dropped_df, out_dir)["table"]
    return outputs


def inspect(cfg):
    print("\n==== Multimodal Config ====")
    print(cfg)
    print("\nImage embedding dir:", image_embedding_dir(cfg))
    print("Joined table dir:", joined_table_dir(cfg))


def main():
    args = parse_args()
    cfg = load_configs(args)
    set_seed(int(cfg.get("seed", 1)))
    inspect(cfg)
    if args.inspect_only:
        return

    mm_cfg = cfg.get("multimodal", {})
    if bool(mm_cfg.get("export_image_embeddings", True)):
        image_paths = export_image_embeddings(cfg)
        print("\n==== Image Embeddings ====")
        for split, path in image_paths.items():
            print(f"{split}: {path}")

    if bool(mm_cfg.get("build_joined_tables", True)):
        joined_paths = build_joined_tables(cfg)
        print("\n==== Joined Tables ====")
        for split, path in joined_paths.items():
            print(f"{split}: {path}")

    if bool(mm_cfg.get("train_classifier", True)):
        outputs = train_and_evaluate(cfg)
        print("\n==== Multimodal Training Outputs ====")
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()

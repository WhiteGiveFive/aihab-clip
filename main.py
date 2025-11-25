import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb  

from utils import load_cfg_from_cfg_file, merge_cfg_from_list
from data.templates import CS_CLASSNAMES
from methods.ProLIP import ProLIP
from aihab_utils.feature_cache import _feature_cache_dir, cache_preprojection_features, _feature_cache_exists
from aihab_utils.model_init import init_clip_and_text_head, inspect
from data.dataloader import build_loaders


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_config', type=str, default='configs/base.yaml')
    p.add_argument('--dataset_config', type=str, default='configs/cs.yaml')
    p.add_argument('--opts', nargs=argparse.REMAINDER, default=None)
    p.add_argument('--inspect_only', action='store_true', help='Run data/transform inspection only')
    return p.parse_args()


def _resolve_cfg_path(p: str) -> str:
    cand = Path(p)
    if cand.is_file():
        return str(cand)
    here = Path(__file__).parent
    cand2 = here / p
    if cand2.is_file():
        return str(cand2)
    root = here.parent
    cand3 = root / p
    if cand3.is_file():
        return str(cand3)
    raise FileNotFoundError(f"Config not found at '{p}' (tried: '{cand}', '{cand2}', '{cand3}')")


def load_configs(args):
    base_cfg_path = _resolve_cfg_path(args.base_config)
    ds_cfg_path = _resolve_cfg_path(args.dataset_config)

    base = load_cfg_from_cfg_file(base_cfg_path)
    ds = load_cfg_from_cfg_file(ds_cfg_path)
    # Merge: dataset config overrides base keys one level deep (CfgNode simple merge)
    base.update(ds)
    if args.opts is not None:
        base = merge_cfg_from_list(base, args.opts)
    return base


def main():
    args = parse_args()
    cfg = load_configs(args)
    set_seed(int(cfg.get('seed', 1)))
    # Initialize WandB if available and requested
    wandb_run = None
    use_wandb = cfg.get('projector', {}).get('enabled', False) and cfg.get('wandb_project', None)
    if use_wandb:
        run_name = (
            f"{cfg.get('dataset', 'ds')}_"
            f"shots{cfg.get('shots', 0)}_"
            f"seed{cfg.get('seed', 1)}_"
            f"{cfg.get('backbone', 'clip')}_"
            f"{cfg.get('train_epoch', 0)}eps_proj"
        )
        wandb_run = wandb.init(
            project=cfg.get('wandb_project'),
            name=run_name,
            config=cfg,
        )

    dl_tr, dl_val, dl_te, train_tf, test_tf, info = build_loaders(cfg)

    # Step 1: CLIP init + CS text head
    clip_bundle = init_clip_and_text_head(cfg)

    # Inspect everything including CLIP/text head
    inspect(cfg, train_tf, test_tf, dl_tr, dl_val, dl_te, info, clip_bundle)

    if not args.inspect_only:
        # Optionally cache features
        if bool(cfg.get('save_features', False)):
            cache_preprojection_features(cfg, clip_bundle, dl_tr, info)
        # Projector training/eval
        if cfg.get('projector', {}).get('enabled', False):
            cache_dir = _feature_cache_dir(cfg)
            aug_views = int(cfg.get('aug_views', 1) or 1)
            if not _feature_cache_exists(cache_dir, aug_views):
                if cfg.get('projector', {}).get('require_cached_features', True):
                    raise FileNotFoundError(f"Cached features not found in {cache_dir}; run with save_features=True first.")
                else:
                    print(f"[warn] Cached features missing in {cache_dir}; generating now.")
                    cache_preprojection_features(cfg, clip_bundle, dl_tr, info)

            prolip = ProLIP(cfg)

            # Unpack clip/text bundles
            state_dict = clip_bundle['state_dict']
            clip_model = clip_bundle['clip_model']
            text_weights = clip_bundle['text_weights']
            text_weights_before = clip_bundle['text_weights_before']

            # For non-ImageNet datasets, reuse primary text weights for all placeholders
            loss, acc = prolip(train_loader=dl_tr,
                            val_loader=dl_val,
                            test_loader=dl_te,
                            test_loader_v2=None,
                            test_loader_sketch=None,
                            test_loader_a=None,
                            test_loader_r=None,
                            text_weights=text_weights,
                            text_weights_a=None,
                            text_weights_r=None,
                            text_weights_before=text_weights_before,
                            model=clip_model,
                            state_dict=state_dict,
                            classnames=CS_CLASSNAMES,
                            task=int(cfg.get('seed', 1)),
                            shots=int(cfg.get('shots', 0) or 0),
                            config_file=Path(args.dataset_config).stem,
                            test_config_path=str(args.dataset_config))
            print("\n==== ProLIP results ====")
            print(f"Loss: {loss}, Accuracy: {acc}")
            if wandb_run is not None:
                wandb_run.log({'acc': float(acc) if hasattr(acc, 'item') else acc})
        else:
            print("\nProjector training disabled (projector.enabled=False).")
    else:
        print("\nInspection-only run; skipping caching and ProLIP.")

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    main()

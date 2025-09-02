import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import load_cfg_from_cfg_file, merge_cfg_from_list
from data.dataset import image_loader
from data.clip_transforms import build_clip_transforms, CLIP_MEAN, CLIP_STD
from data.templates import CS_TEMPLATES, CS_CLASSNAMES
from data import REASSIGN_LABEL_NAME_L3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CSArrayDataset(Dataset):
    """
    Simple Dataset wrapper over preloaded arrays from image_loader.
    Returns (image, label) pairs suitable for CLIP feature extraction.
    Optionally keeps file_names for inspection.
    """
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 file_names: List[str],
                 selected_idx: np.ndarray,
                 transform):
        self.images = images[selected_idx]
        self.labels = labels[selected_idx]
        self.file_names = [file_names[i] for i in selected_idx]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        lbl = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


def few_shot_indices(labels: np.ndarray, shots: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample N=shots examples per class, with replacement if class has < shots."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    sel = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        if len(idx_c) >= shots:
            sel.extend(rng.choice(idx_c, size=shots, replace=False).tolist())
        else:
            sel.extend(rng.choice(idx_c, size=shots, replace=True).tolist())
    return np.array(sel, dtype=np.int64)


def derive_test_paths(train_paths: List[str]) -> List[str]:
    return [p.replace('_train', '_test') for p in train_paths]


def build_loaders(cfg) -> Tuple[DataLoader, DataLoader, object, object, dict]:
    # Build CLIP-friendly transforms honoring aihab aug flags
    train_tf = build_clip_transforms(cfg['data']['preprocessing'], is_train=True, resolution=cfg['resolution'])
    test_tf = build_clip_transforms(cfg['data']['preprocessing'], is_train=False, resolution=cfg['resolution'])

    # Bulk load train split
    images_tr, labels_tr, plot_word_labels_tr, poly_labels_tr, poly_word_labels_tr, file_names_tr, plot_idx_tr, src_tr = \
        image_loader(cfg['data']['dataset_paths'], cfg['data']['index_file_names'], cfg['data']['preprocessing'].get('resize', 256), verbose=True)

    # Bulk load test split (derive _test folder)
    test_paths = derive_test_paths(cfg['data']['dataset_paths'])
    images_te, labels_te, plot_word_labels_te, poly_labels_te, poly_word_labels_te, file_names_te, plot_idx_te, src_te = \
        image_loader(test_paths, cfg['data']['index_file_names'], cfg['data']['preprocessing'].get('resize', 256), verbose=True)

    # Select indices
    seed = int(cfg.get('seed', 1))
    rng = np.random.RandomState(seed)

    shots_val = int(cfg.get('shots', 0)) if cfg.get('shots', 0) is not None else 0
    if shots_val > 0:
        # Few-shot on train
        sel_tr = few_shot_indices(labels_tr, shots_val, rng)
    else:
        # Full-data
        sel_tr = np.arange(images_tr.shape[0])

    sel_te = np.arange(images_te.shape[0])

    # Build datasets and loaders
    ds_tr = CSArrayDataset(images_tr, labels_tr, file_names_tr, sel_tr, transform=train_tf)
    ds_te = CSArrayDataset(images_te, labels_te, file_names_te, sel_te, transform=test_tf)

    dl_tr = DataLoader(ds_tr,
                       batch_size=cfg['data']['batch_size'],
                       shuffle=cfg['data']['shuffle'],
                       num_workers=cfg['data']['num_workers'],
                       pin_memory=True)
    dl_te = DataLoader(ds_te,
                       batch_size=cfg['data']['batch_size'],
                       shuffle=False,
                       num_workers=cfg['data']['num_workers'],
                       pin_memory=True)

    # Few-shot selection map for inspection
    selection_by_class = None
    if shots_val > 0:
        selection_by_class = {}
        classes = np.unique(labels_tr)
        for c in classes:
            idx_c = sel_tr[labels_tr[sel_tr] == c]
            selection_by_class[int(c)] = idx_c.tolist()

    info = {
        'is_few_shot': shots_val > 0,
        'shots': shots_val,
        'train_size': int(len(sel_tr)),
        'train_batches': int(len(dl_tr)),
        'selection_by_class': selection_by_class,
    }

    return dl_tr, dl_te, train_tf, test_tf, info


def inspect(cfg, train_tf, test_tf, dl_tr, dl_te, info: dict, max_show: int = 4):
    # Print configs
    print("\n==== Loaded Config ====")
    print(cfg)

    # Print transforms
    print("\n==== Train Transform (v2) ====")
    print(train_tf)
    print("\n==== Test Transform (v2) ====")
    print(test_tf)

    # Show one batch from train
    print("\n==== Train Batch Sample ====")
    xb, yb = next(iter(dl_tr))
    print(f"images: {tuple(xb.shape)}  dtype={xb.dtype}  device={xb.device}")
    print(f"labels: {yb[:max_show].tolist()}  (showing first {min(max_show, len(yb))})")
    # Map to classnames for readability
    lbl_names = [REASSIGN_LABEL_NAME_L3[int(y)] for y in yb[:max_show].tolist()]
    print(f"label names: {lbl_names}")

    # Dataloader size and few-shot selection details
    print("\n==== Train Loader Size ====")
    print(f"dataset size: {len(dl_tr.dataset)}  num_batches: {len(dl_tr)}")
    if info.get('is_few_shot'):
        shots = info.get('shots')
        print(f"few-shot mode: {shots} per class")
        sel_map = info.get('selection_by_class') or {}
        print("selected indices by class (absolute indices into train array):")
        # Sorted by class id for stable display
        for cls_id in sorted(sel_map.keys()):
            print(f"  class {cls_id}: {sel_map[cls_id]}")

    # Show one batch from test
    print("\n==== Test Batch Sample ====")
    xb2, yb2 = next(iter(dl_te))
    print(f"images: {tuple(xb2.shape)}  dtype={xb2.dtype}  device={xb2.device}")
    print(f"labels: {yb2[:max_show].tolist()}  (showing first {min(max_show, len(yb2))})")
    lbl_names2 = [REASSIGN_LABEL_NAME_L3[int(y)] for y in yb2[:max_show].tolist()]
    print(f"label names: {lbl_names2}")


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

    dl_tr, dl_te, train_tf, test_tf, info = build_loaders(cfg)

    # Always inspect in this step-by-step stage
    inspect(cfg, train_tf, test_tf, dl_tr, dl_te, info)

    if not args.inspect_only:
        print("\nNext steps: feature saving + ProLIP training will be added.")


if __name__ == '__main__':
    main()

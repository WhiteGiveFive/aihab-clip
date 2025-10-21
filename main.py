import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import load_cfg_from_cfg_file, merge_cfg_from_list
from utils import clip_classifier
import clip  # local package aihab-clip/clip
from data.dataset import image_loader
from data.clip_transforms import build_clip_transforms, CLIP_MEAN, CLIP_STD
from data.templates import CS_TEMPLATES, CS_CLASSNAMES
from data import REASSIGN_LABEL_NAME_L3
from methods.utils import compute_image_features


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


def _default_clip_cache_root() -> str:
    # Mirror clip.load default
    return os.path.expanduser("~/.cache/clip")


def _expected_clip_model_path(backbone: str, download_root: Optional[str]) -> Optional[str]:
    """Best-effort reconstruction of the cache path used by clip.load for a named backbone.

    If `backbone` is a known model name, return <root>/<checkpoint_name>. If `backbone` looks like a
    filesystem path to a checkpoint, return it if exists; otherwise None.
    """
    root = download_root or _default_clip_cache_root()
    # Access internal map; safe here since we vendor the clip package locally
    try:
        url_map = clip._MODELS  # type: ignore[attr-defined]
        if backbone in url_map:
            fname = os.path.basename(url_map[backbone])
            return os.path.join(root, fname)
    except Exception:
        pass
    # Fallback: if user provided an explicit checkpoint path
    if os.path.isfile(backbone):
        return backbone
    return None


def init_clip_and_text_head(cfg):
    """Load CLIP and build the CS text classifier.

    Returns a dict with: state_dict, clip_model, preprocess, texts,
    text_weights_before, text_weights.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = cfg.get('backbone', 'RN50')
    cache_root = cfg.get('clip_cache_dir', None)

    expected_path = _expected_clip_model_path(backbone, cache_root)
    if expected_path is None:
        expected_path = os.path.join(_default_clip_cache_root(), f"{backbone.replace('/', '-')}.pt")

    # Load model
    state_dict, clip_model, preprocess = clip.load(backbone, device=device, jit=False, download_root=cache_root)

    # Build text weights for CS
    texts, clip_w_before, clip_w = clip_classifier(CS_CLASSNAMES, CS_TEMPLATES, clip_model)

    return {
        'state_dict': state_dict,
        'clip_model': clip_model,
        'preprocess': preprocess,
        'texts': texts,
        'text_weights_before': clip_w_before,
        'text_weights': clip_w,
    }


def inspect(cfg, train_tf, test_tf, dl_tr, dl_te, info: dict, clip_bundle: Optional[dict] = None, max_show: int = 4):
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

    # CLIP/Text head inspection (moved from init_clip_and_text_head)
    if clip_bundle is not None:
        clip_model = clip_bundle['clip_model']
        backbone = cfg.get('backbone', 'RN50')
        cache_root = cfg.get('clip_cache_dir', None)
        device = next(clip_model.parameters()).device if any(True for _ in clip_model.parameters()) else ("cuda" if torch.cuda.is_available() else "cpu")
        expected_path = _expected_clip_model_path(backbone, cache_root)
        if expected_path is None:
            expected_path = os.path.join(_default_clip_cache_root(), f"{backbone.replace('/', '-')}.pt")

        print("\n==== CLIP Init & Text Head ====")
        print({
            'backbone': backbone,
            'device': str(device),
            'inferred_resolution': getattr(getattr(clip_model, 'visual', None), 'input_resolution', None),
            'clip_cache_root': cache_root or _default_clip_cache_root(),
            'expected_model_path': expected_path,
            'model_file_exists': os.path.isfile(expected_path),
        })

        print("\nText head summary:")
        clip_w_before = clip_bundle['text_weights_before']
        clip_w = clip_bundle['text_weights']
        print({
            'num_classes': len(CS_CLASSNAMES),
            'num_templates': len(CS_TEMPLATES),
            'text_weights_before.shape': tuple(clip_w_before.shape),
            'text_weights.shape': tuple(clip_w.shape),
            'dtype': str(clip_w.dtype),
            'device': str(clip_w.device),
        })

        sample_classes = [REASSIGN_LABEL_NAME_L3[i] for i in sorted(REASSIGN_LABEL_NAME_L3.keys())[:5]]
        sample_prompts = [CS_TEMPLATES[0].format(c) for c in sample_classes]
        print("sample classes (first 5):", sample_classes)
        print("sample prompts (first template):", sample_prompts)


def _canonical_backbone_name(backbone: str) -> str:
    if backbone == "ViT-B/16":
        return "ViTB16"
    if backbone == "ViT-B/32":
        return "ViTB32"
    return backbone


def _feature_cache_dir(cfg) -> Path:
    root = Path(cfg.get('root_path', './'))
    backbone_name = _canonical_backbone_name(cfg.get('backbone', 'RN50'))
    dataset_id = cfg.get('dataset', 'cs')
    shots = int(cfg.get('shots', 0) or 0)
    seed = int(cfg.get('seed', 1) or 1)
    return root / f"features_{backbone_name}_{dataset_id}" / f"{shots}_shot" / f"seed{seed}"


def cache_preprojection_features(cfg, clip_bundle: dict, dl_tr: DataLoader, info: dict):
    clip_model = clip_bundle['clip_model']
    device = next(clip_model.parameters()).device if any(True for _ in clip_model.parameters()) else ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = _feature_cache_dir(cfg)
    num_views = int(cfg.get('aug_views', 1) or 1)
    expected_n = int(info.get('train_size', len(dl_tr.dataset)))

    print("\n==== Feature Caching (pre-projection) ====")
    print({
        'cache_dir': str(cache_dir),
        'backbone': cfg.get('backbone', 'RN50'),
        'dataset': cfg.get('dataset', 'cs'),
        'shots': int(cfg.get('shots', 0) or 0),
        'seed': int(cfg.get('seed', 1) or 1),
        'aug_views': num_views,
        'expected_train_size': expected_n,
    })

    clip_model.eval()
    last_n = None
    for v in range(num_views):
        # Use device-agnostic helper and stream results to CPU to reduce GPU peak
        feats_t, labels_t = compute_image_features(clip_model, dl_tr, to_cpu=True)
        last_n = feats_t.shape[0]

        # Save features and (once) labels
        fpath = cache_dir / f"f{v}.pth"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats_t, fpath)

        if v == 0:
            lpath = cache_dir / "label.pth"
            lpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(labels_t, lpath)

        # Print per-view info
        print(f"[cache] view {v} -> {fpath}")
        print({
            'features.shape': tuple(feats_t.shape),
            'features.dtype': str(feats_t.dtype),
        })
        if v == 0:
            print(f"[cache] labels -> {lpath}")
            uniq = int(labels_t.unique().numel()) if hasattr(labels_t, 'unique') else 'n/a'
            print({
                'labels.shape': tuple(labels_t.shape),
                'labels.dtype': str(labels_t.dtype),
                'num_unique_labels': uniq,
            })

        # Reload validation (shape check)
        loaded = torch.load(fpath, map_location='cpu')
        ok_shape = tuple(loaded.shape) == tuple(feats_t.shape)
        ok_count = feats_t.shape[0] == labels_t.shape[0]
        warn_expected = (expected_n is not None) and (feats_t.shape[0] != expected_n)
        print({
            'reload_shape_ok': ok_shape,
            'rows_match_labels': ok_count,
            'rows_match_expected': (not warn_expected),
        })

    print("\nFeature caching complete.")


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

    # Step 1: CLIP init + CS text head
    clip_bundle = init_clip_and_text_head(cfg)

    # Inspect everything including CLIP/text head
    inspect(cfg, train_tf, test_tf, dl_tr, dl_te, info, clip_bundle)

    if not args.inspect_only:
        if bool(cfg.get('save_features', False)):
            cache_preprojection_features(cfg, clip_bundle, dl_tr, info)
        else:
            print("\nNext steps: feature saving is disabled (set save_features: True). ProLIP training will be added.")


if __name__ == '__main__':
    main()

from pathlib import Path
from datetime import datetime
import json
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.utils import compute_image_features


def _canonical_backbone_name(backbone: str) -> str:
    """
    Canonicalize backbone names for cache folder names.

    - OpenAI aliases: ViT-B/16 -> ViTB16, ViT-B/32 -> ViTB32
    - OpenCLIP names are kept but sanitized: replace '/', spaces, and ':' with '_'
      so strings like 'hf-hub:timm/ViT-SO400M-14-SigLIP' become
      'hf-hub_timm_ViT-SO400M-14-SigLIP'.
    """
    if not backbone:
        return "unknown"
    if backbone == "ViT-B/16":
        return "ViTB16"
    if backbone == "ViT-B/32":
        return "ViTB32"
    name = backbone.replace("hf-hub:", "hf-hub_")
    name = name.replace("/", "_").replace(" ", "_").replace(":", "_")
    return name


def _feature_cache_dir(cfg) -> Path:
    root = Path(cfg.get('root_path', './'))
    backend = str(cfg.get('clip_backend', 'openai')).lower()
    backbone_raw = cfg.get('open_clip_model', cfg.get('backbone', 'RN50')) if backend == 'openclip' else cfg.get('backbone', 'RN50')
    backbone_name = _canonical_backbone_name(backbone_raw)
    dataset_id = cfg.get('dataset', 'cs')
    shots = int(cfg.get('shots', 0) or 0)
    seed = int(cfg.get('seed', 1) or 1)
    return root / f"features_{backbone_name}_{dataset_id}" / f"{shots}_shot" / f"seed{seed}"


def _resolve_dir(root: Path, path: str) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = root / out
    return out


def _embedding_cache_dir(cfg, split: str) -> Path:
    root = Path(cfg.get('root_path', './'))
    ft_cfg = cfg.get('finetune', {})
    out_root = _resolve_dir(root, ft_cfg.get('cache_embeddings_dir', 'feat_cache_vis'))

    backend = str(cfg.get('clip_backend', 'openai')).lower()
    backbone_raw = cfg.get('open_clip_model', cfg.get('backbone', 'RN50')) if backend == 'openclip' else cfg.get('backbone', 'RN50')
    backbone_name = _canonical_backbone_name(backbone_raw)
    dataset_id = cfg.get('dataset', 'cs')
    seed = int(cfg.get('seed', 1) or 1)
    split_name = str(split).lower()

    return out_root / f"{backbone_name}_{dataset_id}" / split_name / f"seed{seed}"


def _to_py(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return v.detach().cpu().tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _item_from_batch(v: Any, idx: int) -> Any:
    item = v[idx] if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)) else v
    return _to_py(item)


def _metadata_rows(metadata: Any, batch_size: int) -> List[Dict[str, Any]]:
    if metadata is None:
        print("[warn] metadata missing; writing default values in metadata.csv.")
        return [{} for _ in range(batch_size)]
    if not isinstance(metadata, dict):
        print("[warn] metadata is not a dict; writing default values in metadata.csv.")
        return [{} for _ in range(batch_size)]

    rows = []
    for i in range(batch_size):
        row = {k: _item_from_batch(v, i) for k, v in metadata.items()}
        rows.append(row)
    return rows


def cache_openclip_embeddings(cfg: dict,
                              model: torch.nn.Module,
                              loader: DataLoader,
                              split: str = 'test',
                              checkpoint_path: Optional[str] = None) -> Path:
    ft_cfg = cfg.get('finetune', {})
    normalize = bool(ft_cfg.get('cache_embeddings_normalize', True))
    cache_dir = _embedding_cache_dir(cfg, split)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.eval()
    feats_list, labels_list = [], []
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, targets, metadata = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
                metadata = None
            else:
                raise ValueError("Expected batch to be (images, targets) or (images, targets, metadata).")

            images = images.to(device, non_blocking=True)
            feats = model.encode_image(images)
            if normalize:
                feats = F.normalize(feats, dim=-1)
            feats = feats.detach().to('cpu')
            feats_list.append(feats)

            targets_cpu = targets.detach().to('cpu')
            labels_list.append(targets_cpu)

            batch_rows = _metadata_rows(metadata, batch_size=int(targets_cpu.shape[0]))
            for i, row in enumerate(batch_rows):
                label = int(targets_cpu[i].item())
                rows.append({
                    "file_name": row.get("file_name", ""),
                    "ground_truth_num_label": label,
                    "ground_truth_word_label": row.get("plot_word_label", ""),
                    "ground_truth_L2_num_label": row.get("l2_label", -1),
                })

    feats_all = torch.cat(feats_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    emb_path = cache_dir / "embeddings.pt"
    lab_path = cache_dir / "labels.pt"
    meta_path = cache_dir / "metadata.csv"
    info_path = cache_dir / "meta.json"

    torch.save(feats_all, emb_path)
    torch.save(labels_all, lab_path)

    columns = [
        "file_name",
        "ground_truth_num_label",
        "ground_truth_word_label",
        "ground_truth_L2_num_label",
    ]
    df = pd.DataFrame(rows).reindex(columns=columns)
    df.to_csv(meta_path, index=False)

    info = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "split": str(split),
        "normalized": normalize,
        "num_samples": int(feats_all.shape[0]),
        "dim": int(feats_all.shape[1]) if feats_all.ndim == 2 else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "cache_dir": str(cache_dir),
    }
    with info_path.open('w') as f:
        json.dump(info, f, indent=2)

    print("\n==== OpenCLIP Embedding Cache ====")
    print({
        "cache_dir": str(cache_dir),
        "embeddings": str(emb_path),
        "labels": str(lab_path),
        "metadata": str(meta_path),
        "num_samples": int(feats_all.shape[0]),
        "dim": int(feats_all.shape[1]) if feats_all.ndim == 2 else None,
        "normalized": normalize,
    })
    return cache_dir


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
        loaded = torch.load(fpath, map_location='cpu', weights_only=True)
        ok_shape = tuple(loaded.shape) == tuple(feats_t.shape)
        ok_count = feats_t.shape[0] == labels_t.shape[0]
        warn_expected = (expected_n is not None) and (feats_t.shape[0] != expected_n)
        print({
            'reload_shape_ok': ok_shape,
            'rows_match_labels': ok_count,
            'rows_match_expected': (not warn_expected),
        })

    print("\nFeature caching complete.")


def _feature_cache_exists(cache_dir: Path, aug_views: int) -> bool:
    if not cache_dir.exists():
        return False
    if not (cache_dir / "label.pth").is_file():
        return False
    for v in range(aug_views):
        if not (cache_dir / f"f{v}.pth").is_file():
            return False
    return True

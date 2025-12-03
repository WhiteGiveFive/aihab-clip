from pathlib import Path
import torch
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

import os
from typing import Optional
import torch
import clip
from utils import clip_classifier
from data.templates import CS_TEMPLATES, CS_CLASSNAMES
from data import REASSIGN_LABEL_NAME_L3



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

    - ``texts``: tokenized prompts (first template per class) kept for
      inspection or reuse when debugging prompt design.
    - ``text_weights_before``: raw text embeddings produced before CLIP's
      projection layer, stacked as ``[num_templates, num_classes, dim]`` for
      optional downstream analysis.
    - ``text_weights``: normalized classifier weights obtained by averaging
      per-template embeddings, shaped ``[dim, num_classes]`` and ready for
      cosine-similarity classification.
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


def inspect(cfg, train_tf, test_tf, dl_tr, dl_val, dl_te, info: dict, clip_bundle: Optional[dict] = None, max_show: int = 4):
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
    print(f"validation size: {len(dl_val.dataset)}  num_batches: {len(dl_val)}")
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

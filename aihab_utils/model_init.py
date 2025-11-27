import os
from typing import Optional
import torch
import clip
from utils import clip_classifier
from data.templates import CS_TEMPLATES, CS_CLASSNAMES
from data import REASSIGN_LABEL_NAME_L3
import open_clip



def _default_clip_cache_root() -> str:
    # Mirror clip.load default
    return os.path.expanduser("~/.cache/clip")

def _default_openclip_cache_root() -> str:
    # open_clip uses huggingface hub cache by default
    return os.path.expanduser("~/.cache/huggingface/hub")


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


def _load_openclip(backbone: str,
                   pretrained: str,
                   cache_root: Optional[str] = None):
    """
    Load an OpenCLIP model and build the CS text head.

    Returns a bundle matching the OpenAI path:
      {
        'state_dict', 'clip_model',
        'preprocess_train', 'preprocess_val',
        'texts', 'text_weights_before', 'text_weights'
      }
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        backbone,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(backbone)

    # Build prompts: [class][template] flattened
    prompts = []
    for cls in CS_CLASSNAMES:
        cls_clean = cls.replace('_', ' ')
        prompts.extend([tmpl.format(cls_clean) for tmpl in CS_TEMPLATES])

    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        # Some OpenCLIP checkpoints return unnormalized text features; normalize explicitly
        text_feats = model.encode_text(tokens)  # [num_classes * num_templates, dim]
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    num_classes = len(CS_CLASSNAMES)
    num_templates = len(CS_TEMPLATES)
    dim = text_feats.shape[-1]

    text_feats = text_feats.view(num_classes, num_templates, dim)
    # [templates, classes, dim]
    # Note: OpenCLIP encode_text returns post-projection normalized features; to avoid
    # confusion with the OpenAI path (which returns true pre-projection outputs),
    # we set text_weights_before=None below.
    text_weights_before = None
    # Average over templates -> [classes, dim], then transpose to [dim, classes]
    text_weights = text_feats.mean(dim=1)
    text_weights = text_weights / text_weights.norm(dim=-1, keepdim=True)
    text_weights = text_weights.t().contiguous()

    # Keep a reference token batch for the first template (for inspection parity)
    first_template_tokens = tokenizer(
        [CS_TEMPLATES[0].format(cls.replace('_', ' ')) for cls in CS_CLASSNAMES]
    ).to(device)

    return {
        'state_dict': model.state_dict(),
        'clip_model': model,
        'preprocess_train': preprocess_train,
        'preprocess_val': preprocess_val,
        'texts': first_template_tokens,
        'text_weights_before': text_weights_before,
        'text_weights': text_weights,
    }


def init_clip_and_text_head(cfg):
    """Load CLIP (OpenAI or OpenCLIP) and build the CS text classifier.

    Returns a dict with: state_dict, clip_model, preprocess_train,
    preprocess_val, texts, text_weights_before, text_weights.
    """
    backend = str(cfg.get('clip_backend', 'openai')).lower()

    if backend == 'openclip':
        backbone = cfg.get('open_clip_model', cfg.get('backbone', 'ViT-B-16'))
        pretrained = cfg.get('open_clip_pretrained', 'openai')
        cache_root = cfg.get('open_clip_cache_dir', None)
        return _load_openclip(backbone=backbone, pretrained=pretrained, cache_root=cache_root)

    if backend == 'openai':
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
            'preprocess_train': preprocess,
            'preprocess_val': preprocess,
            'texts': texts,
            'text_weights_before': clip_w_before,
            'text_weights': clip_w,
        }

    raise ValueError(f"Unsupported clip_backend '{backend}'. Use 'openai' or 'openclip'.")


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
        backend = str(cfg['clip_backend']).lower()
        backbone = cfg['open_clip_model'] if backend == 'openclip' else cfg['backbone']
        cache_root = cfg.get('open_clip_cache_dir', None) if backend == 'openclip' else cfg.get('clip_cache_dir', None)
        device = next(clip_model.parameters()).device if any(True for _ in clip_model.parameters()) else ("cuda" if torch.cuda.is_available() else "cpu")

        print("\n==== CLIP Init & Text Head ====")
        info_dict = {
            'backend': backend,
            'backbone': backbone,
            'device': str(device),
            'cache_root': cache_root or (_default_openclip_cache_root() if backend == 'openclip' else _default_clip_cache_root()),
            'pretrained': cfg.get('open_clip_pretrained', None) if backend == 'openclip' else None,
        }
        print(info_dict)

        print("\nText head summary:")
        clip_w_before = clip_bundle['text_weights_before']
        clip_w = clip_bundle['text_weights']
        print({
            'num_classes': len(CS_CLASSNAMES),
            'num_templates': len(CS_TEMPLATES),
            'text_weights_before.shape': tuple(clip_w_before.shape) if clip_w_before is not None else None,
            'text_weights.shape': tuple(clip_w.shape),
            'dtype': str(clip_w.dtype),
            'device': str(clip_w.device),
        })

        sample_classes = [REASSIGN_LABEL_NAME_L3[i] for i in sorted(REASSIGN_LABEL_NAME_L3.keys())[:5]]
        sample_prompts = [CS_TEMPLATES[0].format(c) for c in sample_classes]
        print("sample classes (first 5):", sample_classes)
        print("sample prompts (first template):", sample_prompts)

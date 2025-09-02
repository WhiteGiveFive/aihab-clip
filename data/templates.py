"""
Text prompt templates and classnames for the CS habitat dataset used with CLIP/ProLIP.

Usage:
    from data.templates import CS_TEMPLATES, CS_CLASSNAMES
    texts, clip_w_before, clip_w = clip_classifier(CS_CLASSNAMES, CS_TEMPLATES, clip_model)
"""

from . import REASSIGN_LABEL_NAME_L3

# Prompt ensemble tailored for ground-level habitat/land-cover imagery.
# Keep prompts generic and compositionally robust; ProLIP will average
# text embeddings over this set.
CS_TEMPLATES = [
    "a photo of {}.",
    "a photo of a {}.",
    "a close-up photo of {}.",
    "a landscape photo of {}.",
    "a nature photo of {}.",
    "an outdoor scene of {}.",
    "a ground-level photograph of {}.",
    "a field survey photo of {}.",
    "a habitat photo of {}.",
    "a photo of {} habitat.",
    "a photo of {} vegetation.",
    "a photo of {} land cover.",
    "a detailed photo of {}.",
    "a clear photo of {}.",
    "a high-resolution photo of {}.",
    "a zoomed-out photo of {}.",
    "a cropped photo of {}.",
    "a realistic photo of {}.",
]

# Ordered L3 classnames (0..N-1) as required by ProLIP's text head.
CS_CLASSNAMES = [name for idx, name in sorted(REASSIGN_LABEL_NAME_L3.items(), key=lambda kv: kv[0])]


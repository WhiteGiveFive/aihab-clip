"""
CLIP-adapted transforms that honor aihab-style augmentations while
ensuring inputs match CLIP's expected resolution and normalization.

Uses torchvision.transforms.v2 APIs.

Usage:
    from data.clip_transforms import build_clip_transforms
    train_tf = build_clip_transforms(preproc_args, is_train=True, resolution=224)
    test_tf  = build_clip_transforms(preproc_args, is_train=False, resolution=224)
"""

from typing import Dict

from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

from .data_utils import BottomSquareCrop


# CLIP visual encoder normalization statistics
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_clip_transforms(preproc: Dict, is_train: bool, resolution: int = 224) -> v2.Compose:
    """Build CLIP-friendly transforms using aihab's augmentation switches.

    - Spatial transforms follow aihab's flags: bottom_crop, random_crop, flip, rotation.
    - Final size is forced to `resolution` and normalization uses CLIP stats.
    """
    aug = preproc.get('augmentations', {}) if preproc else {}

    tf = []

    if is_train:
        if aug.get('bottom_crop', False):
            # Use a bottom-aligned square crop with the final output size
            tf.append(BottomSquareCrop(resolution))
        elif aug.get('random_crop', False):
            tf.append(v2.RandomResizedCrop(resolution, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC))
        else:
            tf.append(v2.Resize(resolution, interpolation=InterpolationMode.BICUBIC))
            tf.append(v2.CenterCrop(resolution))

        if aug.get('flip', False):
            tf.append(v2.RandomHorizontalFlip())
        if aug.get('rotation', False):
            tf.append(v2.RandomRotation(degrees=30))
    else:
        tf.append(v2.Resize(resolution, interpolation=InterpolationMode.BICUBIC))
        tf.append(v2.CenterCrop(resolution))

    tf.append(v2.ToTensor())
    tf.append(v2.Normalize(mean=CLIP_MEAN, std=CLIP_STD))
    return v2.Compose(tf)

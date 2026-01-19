"""
Text prompt templates and classnames for the CS habitat dataset used with CLIP/ProLIP.

Usage:
    from data.templates import CS_TEMPLATES, CS_CLASSNAMES
    texts, clip_w_before, clip_w = clip_classifier(CS_CLASSNAMES, CS_TEMPLATES, clip_model)
"""

from . import REASSIGN_LABEL_NAME_L3, NAME_LABEL_L2, REASSIGN_NAME_LABEL_L3L2

# Descriptive attributes for Grassland (L2) L3 classes.
GRASSLAND_L3_ATTRS = {
    "Improved Grassland": {
        "vegetation_height": "short to medium vegetation height",
        "sward_texture": "very even close-cropped sward",
        "dominant_cover": "grass-dominated",
        "forb_richness": "few forbs",
    },
    "Neutral Grassland": {
        "vegetation_height": "medium to tall vegetation height",
        "sward_texture": "mixed uneven meadow sward",
        "dominant_cover": "herbs-dominated",
        "forb_richness": "moderate to high forbs",
    },
    "Calcareous Grassland": {
        "vegetation_height": "short vegetation height",
        "sward_texture": "close-cropped open turf",
        "dominant_cover": "fine grasses and herbs dominated",
        "forb_richness": "high forb richness",
    },
    "Acid Grassland": {
        "vegetation_height": "short to medium vegetation height",
        "sward_texture": "patchy or tussocky sward",
        "dominant_cover": "fine grasses dominated",
        "forb_richness": "low to moderate forbs",
    },
    "Bracken": {
        "vegetation_height": "tall vegetation height",
        "sward_texture": "dense canopy of fronds",
        "dominant_cover": "bracken fronds",
        "forb_richness": "low forb richness",
    },
}

# Concise descriptive templates (filled from attr dict per habitat).
DESC_TEMPLATES = [
    "a habitat photo of {habitat}, {vegetation_height}, {sward_texture}, {dominant_cover}, {forb_richness}",
    # "a ground-level view of {habitat}, {vegetation_height}, {sward_texture}, {dominant_cover}, {forb_richness}",
]

HIER_DESC_TEMPLATES = [
    "a landscape photo of {l2}, specifically {l3}, {vegetation_height}, {sward_texture}, {dominant_cover}, {forb_richness}",
    # "a ground-level view of {l2}, specifically {l3}, {vegetation_height}, {sward_texture}, {dominant_cover}, {forb_richness}",
]

# Prompt ensemble tailored for ground-level habitat/land-cover imagery.
# Keep prompts generic and compositionally robust; ProLIP will average
# text embeddings over this set.
CS_TEMPLATES = [
    # "a photo of {}.",
    # "a photo of a {}.",
    # "a close-up photo of {}.",
    # "a landscape photo of {}.",
    # "a nature photo of {}.",
    # "an outdoor scene of {}.",
    # "a ground-level photograph of {}.",
    # "a field survey photo of {}.",
    "a habitat photo of {}.",
    # "a photo of {} habitat.",
    # "a photo of {} vegetation.",
    # "a photo of {} land cover.",
    # "a detailed photo of {}.",
    # "a clear photo of {}.",
    # "a high-resolution photo of {}.",
    # "a zoomed-out photo of {}.",
    # "a cropped photo of {}.",
    # "a realistic photo of {}.",
]

# Ordered L3 classnames (0..N-1) as required by ProLIP's text head.
CS_CLASSNAMES = [name for idx, name in sorted(REASSIGN_LABEL_NAME_L3.items(), key=lambda kv: kv[0])]

# Hierarchical templates
ID_NAME_L2 = {v: k for k, v in NAME_LABEL_L2.items()}


def gen_prompts(use_hierarchy: bool = True, use_descriptive: bool = True):
    """
    Build prompts for CS classes.
    - use_hierarchy=True: include L2 context
    - use_hierarchy=False: fall back to flat CS_TEMPLATES on L3 names only
    - use_descriptive=True: use descriptive templates for Grassland L3 classes
    """
    if use_hierarchy:
        base_templates = [
            # "a photo of {l3}, a type of {l2} habitat.",
            "a habitat photo of {l2}, specifically {l3}",
            # "a habitat photo from the {l2} category: {l3}.", 
            # "a ground-level photograph of {l2} land cover: {l3}.", 
            # "a landscape photo of {l2} habitat with {l3}.",
        ]
        desc_templates = HIER_DESC_TEMPLATES
    else:
        base_templates = CS_TEMPLATES
        desc_templates = DESC_TEMPLATES

    if use_descriptive and len(base_templates) != len(desc_templates):
        raise ValueError(
            "Descriptive templates enabled but template counts differ: "
            f"{len(desc_templates)} (descriptive) vs {len(base_templates)} (base). "
            "Please make them consistent."
        )

    templates_per_class = len(base_templates)

    prompts = []
    for l3 in CS_CLASSNAMES:
        l3_clean = l3.replace("_", " ")
        if use_hierarchy:
            _, l2_id = REASSIGN_NAME_LABEL_L3L2[l3]
            l2 = ID_NAME_L2[l2_id]
        if use_descriptive and l3_clean in GRASSLAND_L3_ATTRS:
            attrs = GRASSLAND_L3_ATTRS[l3_clean]
            if use_hierarchy:
                for tmpl in desc_templates:
                    prompts.append(tmpl.format(l2=l2, l3=l3_clean, **attrs))
            else:
                for tmpl in desc_templates:
                    prompts.append(tmpl.format(habitat=l3_clean, **attrs))
        else:
            if use_hierarchy:
                for tmpl in base_templates:
                    prompts.append(tmpl.format(l3=l3_clean, l2=l2))
            else:
                prompts.extend([tmpl.format(l3_clean) for tmpl in base_templates])

    return prompts, templates_per_class

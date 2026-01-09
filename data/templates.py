"""
Text prompt templates and classnames for the CS habitat dataset used with CLIP/ProLIP.

Usage:
    from data.templates import CS_TEMPLATES, CS_CLASSNAMES
    texts, clip_w_before, clip_w = clip_classifier(CS_CLASSNAMES, CS_TEMPLATES, clip_model)
"""

from . import REASSIGN_LABEL_NAME_L3, NAME_LABEL_L2, REASSIGN_NAME_LABEL_L3L2

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
    "a ground-level photograph of {}.",
    # "a field survey photo of {}.",
    "a habitat photo of {}.",
    "a photo of {} habitat.",
    # "a photo of {} vegetation.",
    "a photo of {} land cover.",
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


def gen_prompts(use_hierarchy: bool = True):
    """
    Build prompts for CS classes.
    - use_hierarchy=True: include L2 context
    - use_hierarchy=False: fall back to flat CS_TEMPLATES on L3 names only
    """
    if use_hierarchy:
        template_set = [
            # "a photo of {l3}, a type of {l2} habitat.",
            "a habitat photo of {l2}, specifically {l3}",
            "a habitat photo from the {l2} category: {l3}.", 
            "a ground-level photograph of {l2} land cover: {l3}.", 
            "a landscape photo of {l2} habitat with {l3}.",
        ]
    else:
        template_set = CS_TEMPLATES

    prompts = []
    for l3 in CS_CLASSNAMES:
        l3_clean = l3.replace("_", " ")
        if use_hierarchy:
            _, l2_id = REASSIGN_NAME_LABEL_L3L2[l3]
            l2 = ID_NAME_L2[l2_id]
            for tmpl in template_set:
                prompts.append(tmpl.format(l3=l3_clean, l2=l2))
        else:
            prompts.extend([tmpl.format(l3_clean) for tmpl in template_set])

    templates_per_class = len(template_set)
    return prompts, templates_per_class
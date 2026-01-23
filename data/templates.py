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

# Descriptive attributes for Wetland (L2) L3 classes.
WETLAND_L3_ATTRS = {
    "Fen, Marsh, Swamp": {
        "vegetation_structure": "tall emergent wetland herbs and sedges",
        "dominant_cover": "sedges, rushes, reeds and wetland herbs",
        "surface_texture": "dense emergent cover with wet channels or patches",
        "water_level": "waterlogged to shallowly inundated",
    },
    "Bog": {
        "vegetation_structure": "low open mossy vegetation with scattered dwarf shrubs",
        "dominant_cover": "bog-moss and cotton-grass",
        "surface_texture": "hummocky surface with small wet hollows",
        "water_level": "persistently waterlogged",
    },
}

# Descriptive attributes for Heathland and Shrub (L2) L3 classes.
HEATHLAND_L3_ATTRS = {
    "Dwarf Shrub Heath": {
        "vegetation_height": "low dwarf shrubs (<1.5 m)",
        "vegetation_structure": "dwarf-shrub dominated, low woody canopy",
        "dominant_cover": "heather/ericoids and dwarf gorse",
        "surface_texture": "patchy heather with moss/lichen and bare ground",
    },
}

# Descriptive attributes for Cropland (L2) L3 classes.
CROPLAND_L3_ATTRS = {
    "Arable and Horticulture": {
        "vegetation_structure": "regular planted rows or plots with uniform spacing",
        "dominant_cover": "arable crops or horticultural plantings",
        "surface_texture": "tilled or ploughed soil with furrows and stubble",
        "management_cue": "actively cultivated or rotational fallow",
    },
}

# Descriptive attributes for Woodland and Forest (L2) L3 classes.
WOODLAND_L3_ATTRS = {
    "Broadleaved Mixed and Yew Woodland": {
        "canopy_structure": "tall broadleaved canopy, irregular and layered",
        "foliage_type": "broad leaves with some evergreen yew",
        "understory_light": "dappled light through mixed canopy",
        "ground_cover": "leaf-littered forest floor",
    },
    "Coniferous Woodland": {
        "canopy_structure": "tall conifer canopy, often uniform or plantation-like",
        "foliage_type": "needle-leaved evergreen conifers",
        "understory_light": "darker, more shaded understory",
        "ground_cover": "needle litter with sparse ground vegetation or moss",
    },
}

# Descriptive attributes for Marine Inlets and Transitional Waters (L2) L3 classes.
MARINE_L3_ATTRS = {
    "Littoral Rock": {
        "substrate_type": "exposed rock platforms or boulder shores",
        "surface_texture": "hard, uneven rock with crevices and pools",
        "dominant_cover": "bare rock with algal and barnacle encrustation",
        "tidal_influence": "intertidal, regularly wetted and exposed",
    },
    "Littoral Sediment": {
        "substrate_type": "sand, mud or gravel flats",
        "surface_texture": "flat, soft sediment with ripples",
        "dominant_cover": "mostly bare sediment with sparse algal film",
        "tidal_influence": "intertidal flats, regularly inundated and exposed",
    },
}

# Descriptive attributes for Montane (L2) L3 classes.
MONTANE_L3_ATTRS = {
    "Montane": {
        "vegetation_structure": "low wind-clipped vegetation above treeline",
        "dominant_cover": "dwarf shrubs with moss, lichen and short grasses",
        "surface_texture": "rocky ground with thin soils and bare patches",
        "exposure_cue": "open, treeless, exposed upland ridges",
    },
}

# Descriptive attributes for Rivers and Lakes (L2) L3 classes.
RIVERS_L3_ATTRS = {
    "Standing Open Waters and Canals": {
        "water_body_form": "open water body or straight canal",
        "water_surface": "still or slow-moving open water",
        "bank_structure": "defined banks or engineered canal edges",
        "aquatic_vegetation": "floating or submerged plants with narrow fringe",
    },
}

# Descriptive attributes for Sparsely Vegetated Land (L2) L3 classes.
SPARSE_L3_ATTRS = {
    "Inland Rock": {
        "substrate_type": "exposed inland rock, cliffs or scree",
        "surface_texture": "hard rock faces with fissures and ledges",
        "dominant_cover": "mostly bare rock with sparse crevice plants",
        "exposure_cue": "dry, wind-exposed inland slopes",
    },
    "Supra-littoral Rock": {
        "substrate_type": "coastal rock above the high-tide line",
        "surface_texture": "rugged rock with spray-wet surfaces",
        "dominant_cover": "salt-tolerant lichens or algae, sparse vegetation",
        "exposure_cue": "wave-splash zone with salt spray",
    },
    "Supra-littoral Sediment": {
        "substrate_type": "coastal sand, shingle or pebbles",
        "surface_texture": "loose granular sediment with ridges",
        "dominant_cover": "sparse salt-tolerant pioneer plants",
        "exposure_cue": "above high tide, exposed to spray and wind",
    },
}

# Descriptive attributes for Urban (L2) L3 classes.
URBAN_L3_ATTRS = {
    "Urban": {
        "built_form": "dense built structures, walls and roofs",
        "surface_material": "sealed hard surfaces like concrete or asphalt",
        "vegetation_cover": "little vegetation or small landscaped patches",
        "infrastructure_cue": "roads, kerbs, fences or utilities",
    },
}

# Descriptive attributes for Sea (L2) L3 classes.
SEA_L3_ATTRS = {
    "Sea": {
        "water_body_form": "open marine water to the horizon",
        "surface_texture": "rolling waves or choppy surface",
        "dominant_cover": "open water with minimal vegetation",
        "coastal_context": "distant coastline or open sea view",
    },
}

# Unified descriptive attributes for all L3 classes.
DESCRIPTIVE_L3_ATTRS = {
    **GRASSLAND_L3_ATTRS,
    **WETLAND_L3_ATTRS,
    **HEATHLAND_L3_ATTRS,
    **CROPLAND_L3_ATTRS,
    **WOODLAND_L3_ATTRS,
    **MARINE_L3_ATTRS,
    **MONTANE_L3_ATTRS,
    **RIVERS_L3_ATTRS,
    **SPARSE_L3_ATTRS,
    **URBAN_L3_ATTRS,
    **SEA_L3_ATTRS,
}

# Concise descriptive templates (filled from attr dict per habitat).
DESC_TEMPLATES = [
    "a habitat photo of {habitat}, {attrs}",
    # "a ground-level view of {habitat}, {attrs}",
]

HIER_DESC_TEMPLATES = [
    "a habitat photo of {l2}, specifically {l3}, {attrs}",
    # "a ground-level view of {l2}, specifically {l3}, {attrs}",
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


def _format_attrs(attrs: dict) -> str:
    return ", ".join(attrs.values())


def gen_prompts(use_hierarchy: bool = True, use_descriptive: bool = True):
    """
    Build prompts for CS classes.
    - use_hierarchy=True: include L2 context
    - use_hierarchy=False: fall back to flat CS_TEMPLATES on L3 names only
    - use_descriptive=True: use descriptive templates for all L3 classes
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

    templates_per_class = len(desc_templates) if use_descriptive else len(base_templates)

    prompts = []
    for l3 in CS_CLASSNAMES:
        l3_clean = l3.replace("_", " ")
        class_prompts = []
        if use_hierarchy:
            _, l2_id = REASSIGN_NAME_LABEL_L3L2[l3]
            l2 = ID_NAME_L2[l2_id]
        if use_descriptive:
            attrs = DESCRIPTIVE_L3_ATTRS.get(l3_clean)
        else:
            attrs = None
        if attrs is not None:
            attrs_text = _format_attrs(attrs)
            if use_hierarchy:
                for tmpl in desc_templates:
                    class_prompts.append(tmpl.format(l2=l2, l3=l3_clean, attrs=attrs_text))
            else:
                for tmpl in desc_templates:
                    class_prompts.append(tmpl.format(habitat=l3_clean, attrs=attrs_text))
        else:
            if use_hierarchy:
                for tmpl in base_templates:
                    class_prompts.append(tmpl.format(l3=l3_clean, l2=l2))
            else:
                class_prompts.extend([tmpl.format(l3_clean) for tmpl in base_templates])

        if use_descriptive:
            preview_count = min(2, len(class_prompts))
            print(f"[gen_prompts] {l3_clean}: {class_prompts[:preview_count]}")

        prompts.extend(class_prompts)

    return prompts, templates_per_class


ORIGINAL_LABEL_NAME_L3 = {
    0: 'Not appeared',
    1: 'Broadleaved Mixed and Yew Woodland',
    2: 'Coniferous Woodland',
    3: 'Boundary and Linear Features',
    4: 'Arable and Horticulture',
    5: 'Improved Grassland',
    6: 'Neutral Grassland',
    7: 'Calcareous Grassland',
    8: 'Acid Grassland',
    9: 'Bracken',
    10: 'Dwarf Shrub Heath',
    11: 'Fen, Marsh, Swamp',
    12: 'Bog',
    13: 'Standing Open Waters and Canals',
    14: 'Not appeared',
    15: 'Montane',
    16: 'Inland Rock',
    17: 'Urban',
    18: 'Supra-littoral Rock',
    19: 'Supra-littoral Sediment',
    20: 'Littoral Rock',
    21: 'Littoral Sediment',
    22: 'Sea'
}

REASSIGN_LABEL_NAME_L3 = {
    0: 'Urban',
    1: 'Broadleaved Mixed and Yew Woodland',
    2: 'Coniferous Woodland',
    3: 'Sea',
    4: 'Arable and Horticulture',
    5: 'Improved Grassland',
    6: 'Neutral Grassland',
    7: 'Calcareous Grassland',
    8: 'Acid Grassland',
    9: 'Bracken',
    10: 'Dwarf Shrub Heath',
    11: 'Fen, Marsh, Swamp',
    12: 'Bog',
    13: 'Littoral Rock',
    14: 'Littoral Sediment',
    15: 'Montane',
    16: 'Standing Open Waters and Canals',
    17: 'Inland Rock',
    18: 'Supra-littoral Rock',
    19: 'Supra-littoral Sediment',
}

ORIGINAL_NAME_LABEL_L3 = {
    'Broadleaved Mixed and Yew Woodland': 1,
    'Coniferous Woodland': 2,
    'Boundary and Linear Features': 3,
    'Arable and Horticulture': 4,
    'Improved Grassland': 5,
    'Neutral Grassland': 6,
    'Calcareous Grassland': 7,
    'Acid Grassland': 8,
    'Bracken': 9,
    'Dwarf Shrub Heath': 10,
    'Fen, Marsh, Swamp': 11,
    'Bog': 12,
    'Standing Open Waters and Canals': 13,
    'Montane': 15,
    'Inland Rock': 16,
    'Urban': 17,
    'Supra-littoral Rock': 18,
    'Supra-littoral Sediment': 19,
    'Littoral Rock': 20,
    'Littoral Sediment': 21,
    'Sea': 22
}

REASSIGN_NAME_LABEL_L3 = {
    'Urban': 0,
    'Broadleaved Mixed and Yew Woodland': 1,
    'Coniferous Woodland': 2,
    'Sea': 3,
    'Arable and Horticulture': 4,
    'Improved Grassland': 5,
    'Neutral Grassland': 6,
    'Calcareous Grassland': 7,
    'Acid Grassland': 8,
    'Bracken': 9,
    'Dwarf Shrub Heath': 10,
    'Fen, Marsh, Swamp': 11,
    'Bog': 12,
    'Littoral Rock': 13,
    'Littoral Sediment': 14,
    'Montane': 15,
    'Standing Open Waters and Canals': 16,
    'Inland Rock': 17,
    'Supra-littoral Rock': 18,
    'Supra-littoral Sediment': 19
}

NAME_LABEL_L2 = {
    'Urban': 0,
    'Woodland and Forest': 1,
    'Cropland': 2,
    'Grassland': 3,
    'Heathland and Shrub': 4,
    'Wetland': 5,
    'Marine Inlets and Transitional Waters': 6,
    'Sparsely Vegetated Land': 7,
    'Rivers and Lakes': 8,
    'Sea': 9,
    'Montane': 10,
}

REASSIGN_NAME_LABEL_L3L2 = {
    'Urban': (0, 0),
    'Broadleaved Mixed and Yew Woodland': (1, 1),
    'Coniferous Woodland': (2, 1),
    'Sea': (3, 9),
    'Arable and Horticulture': (4, 2),
    'Improved Grassland': (5, 3),
    'Neutral Grassland': (6, 3),
    'Calcareous Grassland': (7, 3),
    'Acid Grassland': (8, 3),
    'Bracken': (9, 3),
    'Dwarf Shrub Heath': (10, 4),
    'Fen, Marsh, Swamp': (11, 5),
    'Bog': (12, 5),
    'Littoral Rock': (13, 6),
    'Littoral Sediment': (14, 6),
    'Montane': (15, 10),
    'Standing Open Waters and Canals': (16, 8),
    'Inland Rock': (17, 7),
    'Supra-littoral Rock': (18, 7),
    'Supra-littoral Sediment': (19, 7),
}

NAME_ABB_L2 = {
    'Urban': 'U',
    'Woodland and forest': 'WLF',
    'Cropland': 'CL',
    'Grassland': 'GL',
    'Heathland and shrub': 'HS',
    'Wetland': 'WL',
    'Marine inlets and transitional waters': 'MITW',
    'Sparsely vegetated land': 'SVL',
    'Rivers and lakes': 'RL',
    'Sea': 'S',
    'Montane': 'M',
}

CORRUPT_IMAGES = [
'ATT3735_594XX3_2023_photo2-20230928-121257.jpg'
]

# Used by the Wandb sweeps config to assign values for the main config
SWEEP_KEY_MAPPING = {
    'cross_valid': ['cross_valid'],
    'first_cv_only': ['data', 'data_split', 'first_cv_only'],
    'num_epochs': ['training', 'num_epochs'],
    'optimiser': ['training', 'optimiser', 'type'],
    'lr': ['training', 'optimiser', 'lr'],
    'weight_decay': ['training', 'optimiser', 'weight_decay'],
    'batch_size': ['data', 'batch_size'],
    'img_resize': ['data', 'preprocessing', 'resize'],
    'model_name': ['model', 'name'],
    'model_config': ['model', 'model_config'],
    'input_size': ['model', 'input_size'],
    'flip': ['data', 'preprocessing', 'augmentations', 'flip'],
    'rotation': ['data', 'preprocessing', 'augmentations', 'rotation'],
    'random_crop': ['data', 'preprocessing', 'augmentations', 'random_crop'],
    'multi_views_supcon': ['data', 'preprocessing', 'multi_views', 'supcon'],
    'supcon_pretrain': ['training', 'supcon_conf', 'pretrain'],
    'supcon_ptr_dir': ['training', 'supcon_conf', 'prt_dir'],
    'supcon_prt_filename': ['training', 'supcon_conf', 'prt_filename'],
}


def l2_names_to_l3(l2_names):
    """
    Convert L2 names (strings) to ordered L3 classnames + L3 ids.
    Uses REASSIGN_NAME_LABEL_L3L2 (L3 -> (L3 id, L2 id)).
    """
    if not l2_names:
        return [], []

    # case-insensitive match against canonical L2 names
    l2_norm = {k.lower(): v for k, v in NAME_LABEL_L2.items()}
    missing = [n for n in l2_names if n.lower() not in l2_norm]
    if missing:
        raise ValueError(f"Unknown L2 names: {missing}. Expected one of: {list(NAME_LABEL_L2.keys())}")

    l2_ids = {l2_norm[n.lower()] for n in l2_names}

    l3_pairs = [
        (l3_name, l3_id)
        for l3_name, (l3_id, l2_id) in REASSIGN_NAME_LABEL_L3L2.items()
        if l2_id in l2_ids
    ]
    l3_pairs.sort(key=lambda x: x[1])  # stable order by L3 id

    l3_names = [n for n, _ in l3_pairs]
    l3_ids = [i for _, i in l3_pairs]
    return l3_names, l3_ids


def build_l3_to_l2_map():
    """
    Build L3->L2 id mapping and ordered L2 names.

    Returns:
        l3_to_l2: list[int] indexed by L3 id -> L2 id
        l2_names: list[str] indexed by L2 id -> name
    """
    # Order L2 names by their numeric id.
    l2_names = [name for name, _ in sorted(NAME_LABEL_L2.items(), key=lambda kv: kv[1])]

    # Order L3 names by their L3 id, then map to L2 id.
    l3_pairs = sorted(REASSIGN_NAME_LABEL_L3L2.items(), key=lambda kv: kv[1][0])
    l3_to_l2 = [int(l2_id) for _, (_, l2_id) in l3_pairs]

    return l3_to_l2, l2_names
    

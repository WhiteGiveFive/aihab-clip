DEV NOTES — aihab-clip (ProLIP × CS Habitat)

Audience: Future me (the assistant) to rehydrate context quickly and continue the work confidently.

Purpose & Scope
- aihab-clip is a clean integration point to apply ProLIP to the CS habitat dataset used in aihab.
- Goal: Few-shot and full-data fine-tuning of CLIP’s visual projector on CS, with CLIP-friendly transforms while reusing aihab’s augmentation semantics.

Repo Map (what to scan on startup)
- aihab-clip/
  - main.py: current runner. Loads configs, builds CLIP v2 transforms, constructs loaders (full or few-shot), and prints inspection (config, transforms, sample batch, few-shot indices).
  - configs/
    - base.yaml: defaults (ProLIP knobs, data defaults, CLIP resolution, aug flags, shots/seed).
    - cs.yaml: CS dataset paths + CSV; defaults to full-data (shots: 0).
  - methods/
    - ProLIP.py: projector training. Supports optional mini-batching of cached features via cfg.feat_batch_size for full-data tuning; aug_views cycling preserved.
    - utils.py: accuracy, feature computation helpers.
    - method.py: minimal base class (import dependency for ProLIP).
  - data/
    - dataset.py + dataloader.py: CS data utilities (ported from aihab). Note imports now target local data/__init__.py and data_utils.
    - __init__.py: L3/L2 label mappings (REASSIGN_*), used for selection, label name decoding.
    - templates.py: CS_TEMPLATES (prompt ensemble) + CS_CLASSNAMES (L3 names in numeric order).
    - clip_transforms.py: build_clip_transforms(preproc, is_train, resolution) using torchvision v2. Honors aihab aug flags but applies CLIP normalization/size.
    - data_utils.py: custom transforms (BottomSquareCrop, edge/canny helpers, two-view wrappers).
- ProLIP/ (upstream reference in repo): useful for cross-checking intended behavior
  - methods/ProLIP.py, datasets/*, configs/*, utils.py, README.md (see details below).
- aihab/ (original project): label mappings, data loaders, models, trainer/evaluator (for understanding data & labels).

Current Status
- Implemented and validated:
  - Config loading from aihab-clip CWD (base.yaml + cs.yaml).
  - CLIP v2 transforms with aihab augment flags + CLIP normalization.
  - Full-data and few-shot loaders over preloaded arrays (efficient once loaded).
  - Inspection: prints merged config, transforms, a sample train/test batch (shapes + label names), dataloader size, and (in few-shot) selected indices per class.
- Ready to add next: feature saving (aug_views), projector training/eval, config toggles for saving/training.

Few-Shot Mechanics (important mental model)
- Selection is done ONCE per run/seed, not per epoch.
  - In this repo: main.py:few_shot_indices uses a seeded NumPy RNG to pick N per class (with replacement when a class has < N samples).
  - In upstream ProLIP: datasets/utils.py:generate_fewshot_dataset uses Python’s random.sample / random.choices; seed set just before selection.
- Per-epoch variation comes from augmentation views (aug_views) and data transforms — not from changing which images were selected.

Transforms & Normalization
- Use torchvision.transforms.v2 APIs.
- aihab-style aug flags: bottom_crop, random_crop, flip, rotation.
- CLIP normalization enforced at the end (mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]) and final spatial size = resolution (e.g., 224).
- Rationale: Keep augmentation semantics familiar to aihab while feeding encode_image with CLIP-expected inputs.

Config Conventions
- Two-layer configs: configs/base.yaml (defaults) + configs/cs.yaml (dataset-specific overrides). Run from aihab-clip/ folder.
- shots:
  - 0 (or <1): full-data projector training (recommend aug_views: 1 + feat_batch_size: 2048–4096).
  - N>0: few-shot per class (recommend aug_views: high, e.g., 300; feat_batch_size: 0).
- feat_batch_size: optional; splits cached feature tensor into chunks to prevent OOM in full-data mode; regularizer scaled per chunk.
- lr_v, lambda_v or lambda_funct_1_N (λ = 1/N) / lambda_funct_1_N2 (λ = 1/N^2) mirror ProLIP’s practice.
- --opts: top-level overrides only; keep it last and use KEY VALUE pairs (e.g., shots 4 seed 1). Don’t pass flags after --opts.

Paths & CWD Assumptions
- Run commands from aihab-clip/:
  - Full-data inspect:
    - python main.py --inspect_only --base_config configs/base.yaml --dataset_config configs/cs.yaml
  - Few-shot inspect (4-shot example):
    - python main.py --inspect_only --base_config configs/base.yaml --dataset_config configs/cs.yaml --opts shots 4 seed 1
- Ensure configs/cs.yaml points to correct dataset paths and CSV index filenames.
- Test set path is derived by replacing “_train” with “_test”. Verify both exist.

Upstream ProLIP (for quick recall)
- How it works:
  - Save pre-projection features for the chosen subset across aug_views → caches/features_*.
  - Train a tiny projector (RN: linear layer; ViT: learned proj matrix) with CE on text similarities + λ·MSE-to-init regularizer.
  - Evaluate on test set via projected, normalized features vs text embeddings.
- Key files:
  - ProLIP/README.md: method overview and scripts.
  - ProLIP/methods/ProLIP.py: projector, training loop, λ schedule.
  - ProLIP/utils.py: clip_classifier (builds text head), config helpers.
  - ProLIP/datasets/*: dataset classes define classnames, template, and few-shot generator.
  - test_time_ProLIP/*: test-time projector tuning (entropy minimization).

aihab Project (for quick recall)
- aihab/utils/__init__.py: L3/L2 label maps and names (ported here into data/__init__.py).
- aihab/data/*: original data reading, splitting, and transforms (we borrowed semantics, not normalization).
- aihab/methods/*, aihab/models/*: baseline pipelines (not used here, but informative for labels and data organization).

Next Steps Checklist (what to implement next in main.py)
1) CLIP Init + Text Head
   - Import from aihab-clip/clip: state_dict, clip_model, preprocess = clip.load(backbone).
   - Build CS text embeddings: from data.templates import CS_TEMPLATES, CS_CLASSNAMES → utils.clip_classifier.
   - Print: backbone, resolution inferred, text weights shape.

2) Feature Saving Loop
   - If cfg.save_features: True → for v in range(aug_views):
       - Iterate train loader once, call clip_model.encode_image(imgs), collect x_before_proj (pre-projection features) + labels.
       - Save to: <root_path>/features_<Backbone>_cs/<shots>_shot/seed<seed>/f{v}.pth and label.pth (labels saved once when v==0).
   - Print: save paths and tensor shapes.

3) Projector Training + Eval (wire methods/ProLIP.ProLIP)
   - Load cached features + labels for aug_views.
   - Call ProLIP(train_loader=?, val_loader=?, test_loader, ...):
       - For CS, val_loader can be None or a split of train if you want LR/λ search.
       - test_loader = CS test DataLoader (CLIP test transform).
       - text_weights = from clip_classifier.
       - state_dict = model state_dict to get initial visual proj.
   - Print: train loss per epoch (summaries), final test accuracy.

4) Config toggles to expose now if useful
   - save_features, aug_views, feat_batch_size, train_epoch, lr_v, lambda_funct_1_N / lambda_v.

Validation & Debugging Tips
- Memory: In full-data mode, use feat_batch_size (>0) to chunk projector training; keep aug_views=1 to avoid huge caches.
- Seeds: shots and selection are seed-dependent; set seed in configs or via --opts.
- Sanity checks: inspection prints selection_by_class in few-shot; confirms transform stacks and batch shapes.
- Label names: Derived from data/__init__.py’s REASSIGN_LABEL_NAME_L3 mapping; ensure order matches class ids.

Potential Pitfalls
- Config paths: run from aihab-clip or pass correct relative paths.
- --opts: make sure it’s last; only KEY VALUE pairs.
- CLIP normalization: ensure CLIP mean/std applied; mismatched normalization will hurt performance.
- Backbone resolution: keep resolution consistent with the CLIP backbone you load (224 is correct for RN50/ViT-B/16).

Quick Commands Cheat Sheet
- Full-data inspect:
  - python main.py --inspect_only --base_config configs/base.yaml --dataset_config configs/cs.yaml
- Few-shot inspect (4-shot):
  - python main.py --inspect_only --base_config configs/base.yaml --dataset_config configs/cs.yaml --opts shots 4 seed 1
- (After feature saving added) Example:
  - python main.py --base_config configs/base.yaml --dataset_config configs/cs.yaml --opts shots 4 seed 1 save_features True aug_views 300
- (After training added) Example:
  - python main.py --base_config configs/base.yaml --dataset_config configs/cs.yaml --opts shots 4 seed 1 train_epoch 300 lr_v 1e-5 lambda_funct_1_N True


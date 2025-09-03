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
  - CLIP init + CS text head build (step 1):
    - Loads CLIP via local clip.load(backbone) → returns (state_dict, clip_model, preprocess).
    - Builds text classifier from CS_TEMPLATES × CS_CLASSNAMES via utils.clip_classifier.
    - Inspection moved: all CLIP/text-head prints happen inside inspect() (not init), after the model is built.
    - Prints: backbone, device, inferred visual resolution, clip_cache_root, expected_model_path, file existence.
    - Prints text head summary: num_classes, num_templates, text_weights_before.shape, text_weights.shape, dtype/device, and sample prompts.
    - Note: clip_classifier respects model device (CPU/GPU), no hard CUDA assumption.
  - Feature caching (step 2): IMPLEMENTED
    - cache_preprojection_features in main.py saves pre-projection image features and labels.
    - Uses methods/utils.compute_image_features(model, loader, to_cpu=True) which is now device-agnostic and streams batch outputs to CPU to avoid GPU OOM.
    - Saves per view to: <root>/features_<Backbone>_cs/<shots>_shot/seed<seed>/f{v}.pth and label.pth once.
    - Prints per-view shapes/dtypes and validates by reloading each saved file, checking shapes and counts vs expected train size.
  - Ready to add next: projector training/eval wiring (step 3).

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
 - clip_cache_dir: optional; overrides default CLIP cache root (defaults to ~/.cache/clip). main.py prints the expected model path and whether it exists.

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
1) CLIP Init + Text Head — COMPLETED
   - Loads CLIP and prints cache info; builds CS text head with prompt ensemble and prints shapes/dtypes.

2) Feature Saving Loop — COMPLETED
   - cfg.save_features: True → for v in range(aug_views):
     - Iterate train loader once, collect x_before_proj + labels via device-agnostic compute_image_features(..., to_cpu=True).
     - Save to: <root>/features_<Backbone>_cs/<shots>_shot/seed<seed>/f{v}.pth and label.pth (once).
     - Print: save paths, tensor shapes/dtypes, and reload validation flags.

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
- Feature caching: compute_image_features(to_cpu=True) streams results to CPU; saved tensors are CPU, avoiding GPU OOM.

Potential Pitfalls
- Config paths: run from aihab-clip or pass correct relative paths.
- --opts: make sure it’s last; only KEY VALUE pairs.
- CLIP normalization: ensure CLIP mean/std applied; mismatched normalization will hurt performance.
- Backbone resolution: keep resolution consistent with the CLIP backbone you load (224 is correct for RN50/ViT-B/16).
- Disk space: per view size ≈ N×Dpre×bytes. For N≈5k: RN50 fp16 ~10MB/view; ViT-B fp16 ~7.7MB/view. Multiply by aug_views.

Quick Commands Cheat Sheet
- Full-data inspect:
  - python main.py --inspect_only --base_config configs/base.yaml --dataset_config configs/cs.yaml
- Few-shot inspect (4-shot):
  - python main.py --inspect_only --base_config configs/base.yaml --dataset_config configs/cs.yaml --opts shots 4 seed 1
- Full-data feature cache (1 view):
  - python main.py --base_config configs/base.yaml --dataset_config configs/cs.yaml --opts shots 0 save_features True aug_views 1 seed 1
- Few-shot feature cache (e.g., 4-shot, many views):
  - python main.py --base_config configs/base.yaml --dataset_config configs/cs.yaml --opts shots 4 seed 1 save_features True aug_views 300
- (After training added) Example:
  - python main.py --base_config configs/base.yaml --dataset_config configs/cs.yaml --opts shots 4 seed 1 train_epoch 300 lr_v 1e-5 lambda_funct_1_N True

Implementation Notes (2025-09-02)
- Refactor: moved CLIP/text-head inspection prints from init_clip_and_text_head() into inspect(), called after model build for a single consolidated view.
- Feature caching: added cache_preprojection_features(). Uses device-agnostic compute_image_features(..., to_cpu=True) to stream batches to CPU and save CPU tensors per view; prints shapes and validates by reloading.
- Device-agnostic utils: methods/utils.compute_image_features now infers model device; supports to_cpu flag; retains internal torch.no_grad().
- Simplified no_grad: removed redundant no_grad around caching loop in main (compute_image_features already uses no_grad).

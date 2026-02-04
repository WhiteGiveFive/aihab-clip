# Embedding Visualization Utilities

This folder stores cached embedding artifacts and the dimensionality-reduction script used to produce 2D coordinates for visualization.

## What Gets Cached (Upstream)
The embeddings are generated during OpenCLIP fine-tuning when `finetune.cache_embeddings: True` is enabled.

Key files written to the cache folder:
- `embeddings.pt` — tensor of image embeddings (N x D).
- `labels.pt` — tensor of numeric labels (N).
- `metadata.csv` — per-sample metadata (CSV) with:
  - `file_name`
  - `ground_truth_num_label`
  - `ground_truth_word_label`
  - `ground_truth_L2_num_label`
- `meta.json` — summary info (timestamp, dims, checkpoint path, etc.).

Cache path layout:
```
feat_cache_vis/{model}_{dataset}/{split}/seed{seed}/
```

See `dev_plans/model_save_and_embedding_cache_2026-02-03.md` for full context on the cache + checkpoint flow.

## Script: `feat_vis.py`
`feat_vis.py` loads cached embeddings and performs optional PCA + UMAP or t‑SNE to produce 2D coordinates, then saves them next to the cache.

### Inputs (Required)
- `embeddings.pt`
- `metadata.csv`

### Outputs
Saved to the same cache folder:
- `vis_umap_coords.npy` or `vis_tsne_coords.npy`
- `vis_umap.html` or `vis_tsne.html` (Plotly interactive)

## Usage Examples

UMAP with PCA:
```bash
python feat_cache_vis/feat_vis.py \
  --cache_dir feat_cache_vis/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_cs/test/seed1 \
  --umap \
  --pca_dim 50 \
  --out_prefix vis \
  --overwrite
```

t‑SNE with PCA:
```bash
python feat_cache_vis/feat_vis.py \
  --cache_dir feat_cache_vis/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_cs/test/seed1 \
  --tsne \
  --pca_dim 50 \
  --out_prefix vis \
  --overwrite
```

## Dependencies
The script uses:
- `torch`
- `pandas`
- `numpy`
- `scikit-learn` (PCA, t‑SNE)
- `umap-learn`
- `plotly`

## Notes
- The coordinates are saved as `.npy` for downstream tools.
- `metadata.csv` row order is assumed to match the embedding order.
- If you only want coordinates (no HTML), you can ignore the generated `.html` files.

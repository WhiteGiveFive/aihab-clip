# Multimodal CS Pipeline

This folder contains the separate multimodal workflow for CS habitat classification using:

- OpenCLIP image embeddings
- Google's satellite embeddings from `data/cs_geo_gse_10km`
- a late-fusion classifier over precomputed features

This pipeline is intentionally separate from the existing image-text OpenCLIP training flow in `main.py`. It does not replace or modify the original ProLIP / OpenCLIP code path.

## What This Pipeline Does

The multimodal workflow runs in three stages:

1. Export deterministic OpenCLIP image embeddings for `train`, `val`, and `test`
2. Inner-join those image embeddings with the per-file geo embeddings
3. Train and evaluate a classifier on one of:
   - `image_only`
   - `geo_only`
   - `raw_concat`

All three modes are evaluated on the same geo-matched sample universe for fair comparison.

By default, the training loop now evaluates both validation and test splits after every epoch, prints those scalar metrics to the console, and stores them in `metrics.json`. Checkpoint selection still uses validation `top1_acc`, not test performance.

## Entry Point

Use:

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml
```

This entrypoint is independent from `main.py`.

## Config Files

The new multimodal workflow uses:

- `configs/multimodal_base.yaml`
- `configs/multimodal_cs.yaml`

Config merge order is:

1. multimodal base config
2. multimodal dataset config
3. CLI overrides from `--opts`

Unlike the legacy `main.py` path, the multimodal entrypoint performs a nested merge so dataset overrides do not wipe unrelated nested keys from the base config.

## Default Assumptions

The default configuration assumes:

- dataset: `cs`
- OpenCLIP backbone: `hf-hub:timm/ViT-SO400M-16-SigLIP2-384`
- primary image feature source: habitat-finetuned checkpoint
- geo embeddings: `./data/cs_geo_gse_10km/CS_Xplots_embeddings_per_file.parquet`
- evaluation framing: "same grid, new photo"
- missing geo rows are dropped by inner join

## Image Feature Sources

The multimodal pipeline supports two image-feature sources.

### 1. Habitat-finetuned

This is the default and primary path.

Required config:

```yaml
multimodal:
  image_feature_source: habitat_finetuned
  image_checkpoint: ./model_ckpt/...
```

Behavior:

- loads the configured OpenCLIP architecture
- loads the fine-tuned checkpoint into the visual model
- extracts image embeddings in eval mode
- uses eval preprocessing only, not train-time augmentation

### 2. Pretrained

This is the ablation path.

Required config:

```yaml
multimodal:
  image_feature_source: pretrained
```

Behavior:

- loads the same OpenCLIP architecture without the habitat fine-tuned checkpoint
- exports embeddings using the pretrained model

Example override:

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts multimodal.image_feature_source pretrained
```

## Fusion Modes

Configure the training mode with:

```yaml
multimodal:
  fusion_mode: raw_concat
```

Supported modes:

- `image_only`
- `geo_only`
- `raw_concat`

Examples:

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts multimodal.fusion_mode image_only
```

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts multimodal.fusion_mode geo_only
```

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts multimodal.fusion_mode raw_concat
```

## Stage Controls

Each major stage can be toggled independently from config or CLI.

Config keys:

```yaml
multimodal:
  export_image_embeddings: True
  build_joined_tables: True
  train_classifier: True
  report_test_each_epoch: True
  print_epoch_metrics: True
```

### Inspect Only

Print the resolved config and target artifact directories without running the pipeline:

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --inspect_only
```

### Export Only

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts \
    multimodal.export_image_embeddings True \
    multimodal.build_joined_tables False \
    multimodal.train_classifier False
```

### Join Only

This assumes the image embedding parquet files already exist.

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts \
    multimodal.export_image_embeddings False \
    multimodal.build_joined_tables True \
    multimodal.train_classifier False
```

### Train Only

This assumes the joined multimodal tables already exist.

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts \
    multimodal.export_image_embeddings False \
    multimodal.build_joined_tables False \
    multimodal.train_classifier True
```

## Data and Split Behavior

### Image Splits

The multimodal pipeline reuses the current CS image split logic:

- bulk-load train data from `data.dataset_paths`
- derive or use the configured test paths
- create `train` / `val` using the existing grouped stratified split on `plot_idx`
- keep `test` as the test folder split

This means the multimodal workflow is aligned with the current image-only repo behavior before geo matching is applied.

### Geo Join

Geo data is loaded from:

```yaml
multimodal:
  geo_embeddings_path: ./data/cs_geo_gse_10km/CS_Xplots_embeddings_per_file.parquet
```

Join behavior:

- lowercase `file` is used as the join key
- geo duplicates are deduplicated before joining
- rows with missing geo features are dropped
- join is performed separately within each split

### Geo Dedup Policy

If multiple geo rows exist for the same file:

- prefer the row with non-empty `BH_PLOT_DESC`
- allow duplicates only if `embedding_key` and all `A00..A63` values agree
- fail fast if duplicate rows disagree

This is intentional. Silent disagreement in geo features is treated as a data error.

## Exported Image Embedding Schema

Each split is exported to one parquet file with:

Metadata columns:

- `file`
- `label_id`
- `label_name`
- `l2_label`
- `plot_idx`
- `image_source`
- `split`

Feature columns:

- `I000..I{D-1}`

Important details:

- train metadata is explicitly included in this new multimodal artifact
- these artifacts are separate from the older `feat_cache_vis` export path
- image embeddings are extracted with eval preprocessing, not train augmentation

## Joined Table Schema

The joined multimodal split parquet contains:

- all image embedding columns `I*`
- all geo embedding columns `A00..A63`
- metadata from the image split
- `embedding_key` from the geo table

## Output Layout

By default, outputs are written under:

```text
./multimodal_artifacts/
```

The structure is:

```text
multimodal_artifacts/
  image_embeddings/
    cs/
      <encoder_tag>/
        seed<seed>/
          train.parquet
          val.parquet
          test.parquet

  joined_tables/
    cs/
      <encoder_tag>/
        <geo_tag>/
          seed<seed>/
            train.parquet
            val.parquet
            test.parquet
            train_manifest.json
            val_manifest.json
            test_manifest.json
            train_dropped.csv
            val_dropped.csv
            test_dropped.csv

  runs/
    cs/
      <encoder_tag>/
        <fusion_mode>/
          seed<seed>/
            best_model.pt
            metrics.json
            geo_standardization.json
            test_confusion_matrix.npy
```

`<encoder_tag>` is derived from the checkpoint name for habitat-finetuned runs, or from the model name for pretrained runs.

## Metrics

The classifier training stage reports:

- `loss`
- `top1_acc`
- `top3_acc`
- weighted `f1`
- `mcc`
- confusion matrix

The saved `metrics.json` contains:

- `history`: one entry per epoch
- `val`: final validation metrics from the selected best checkpoint
- `test`: final test metrics from the selected best checkpoint

Per-epoch `history` entries contain:

- `train_loss`
- legacy validation keys: `loss`, `top1_acc`, `top3_acc`, `f1`, `mcc`
- explicit validation keys: `val_loss`, `val_top1_acc`, `val_top3_acc`, `val_f1`, `val_mcc`
- test keys when enabled: `test_loss`, `test_top1_acc`, `test_top3_acc`, `test_f1`, `test_mcc`

During training, the full pipeline prints one line per epoch with train, validation, and test scalar metrics when `multimodal.print_epoch_metrics` is `True`.

Model selection is still done on validation `top1_acc`.

Geo features are standardized using train-split statistics only. Image embeddings are used as exported.

## Important Design Choices

### Fair Comparison Across Modes

`image_only`, `geo_only`, and `raw_concat` are trained and evaluated on the same geo-matched subset. This is deliberate.

Without this restriction:

- `image_only` could use more samples than the multimodal runs
- the comparison would be confounded by different sample universes

### Label Reindexing After Join

After the geo inner join, the training split may no longer contain every original class id. The training code remaps the remaining labels to a dense `0..K-1` range before fitting the classifier.

This avoids classifier output gaps when some classes disappear from the geo-matched training subset.

### No End-to-End Joint Encoder Training

This pipeline is feature-based.

It does not:

- fine-tune the OpenCLIP encoder jointly with the multimodal head
- add learned per-modality projection towers yet
- redesign the split for "new grid, new photo"

Those are later extensions, not part of this implementation.

## Useful CLI Overrides

### Switch to pretrained image features

```bash
--opts multimodal.image_feature_source pretrained
```

### Change fusion mode

```bash
--opts multimodal.fusion_mode image_only
```

### Use a different checkpoint

```bash
--opts multimodal.image_checkpoint ./model_ckpt/my_checkpoint.pt
```

### Change output root

```bash
--opts multimodal.output_dir ./my_multimodal_outputs
```

### Limit split sizes for smoke runs

```bash
--opts multimodal.max_samples_per_split 128
```

### Change training hyperparameters

```bash
--opts \
  multimodal.train_epoch 100 \
  multimodal.lr 0.0005 \
  multimodal.hidden_dim 512 \
  multimodal.dropout 0.2
```

### Disable per-epoch test evaluation

```bash
--opts multimodal.report_test_each_epoch False
```

### Disable per-epoch console logging

```bash
--opts multimodal.print_epoch_metrics False
```

## Example Runs

### 1. Default full pipeline with habitat-finetuned image features and raw concat

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml
```

### 2. Pretrained ablation with raw concat

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts \
    multimodal.image_feature_source pretrained \
    multimodal.fusion_mode raw_concat
```

### 3. Image-only baseline on the geo-matched subset

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts multimodal.fusion_mode image_only
```

### 4. Geo-only baseline on the geo-matched subset

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts multimodal.fusion_mode geo_only
```

### 5. Image-only basline on the geo-matched subset with reused joined table

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts \
    multimodal.fusion_mode image_only \
    multimodal.export_image_embeddings False \
    multimodal.build_joined_tables False
```

## Environment Notes

This pipeline requires the same core Python stack as the rest of the project, including:

- `torch`
- `open_clip`
- `pandas`
- parquet support such as `pyarrow`
- `numpy`
- `scikit-learn`
- `torcheval`

When running commands from the shell, use the project environment that already has these dependencies installed.

## Troubleshooting

### `FileNotFoundError` for geo parquet

Check:

- `multimodal.geo_embeddings_path`
- that `data/cs_geo_gse_10km/CS_Xplots_embeddings_per_file.parquet` exists

### Checkpoint model mismatch

If `image_feature_source=habitat_finetuned`, the configured checkpoint must match the configured OpenCLIP architecture.

### Empty joined train split

This means the geo inner join dropped all training rows. Check:

- filename alignment between image tables and geo parquet
- lowercase filename matching
- whether the selected subset or split is too restrictive

### Labels missing after join

The training code can handle a reduced class set after joining, but if validation or test contains labels absent from the geo-matched training split, training will fail fast.

## Related Files

- `multimodal_main.py`
- `multimodal/artifacts.py`
- `multimodal/data.py`
- `multimodal/models.py`
- `multimodal/trainer.py`
- `configs/multimodal_base.yaml`
- `configs/multimodal_cs.yaml`

# Multimodal CS Pipeline

This folder contains the separate multimodal workflow for CS habitat classification using:

- OpenCLIP image embeddings
- Google's satellite embeddings from `data/cs_geo_gse_10km` or the curated 10m adapter in `data/cs_geo_gse_10m`
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
- `configs/multimodal_cs_geo_100m.yaml`
- `configs/multimodal_cs_geo_100m_cleaned_test.yaml`

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
- geo embeddings: `./data/cs_geo_gse_10km/CS_Xplots_embeddings_per_file.parquet` for the generic CS runner; `./data/cs_geo_gse_10m/CS_Xplots_embeddings_per_file_10m_public.parquet` for the 10m runner
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
- `tabular_projected_concat`

For image + geo, `tabular_projected_concat` applies a trainable MLP projection to the geo embedding columns `A00..A63` before concatenating them with the image embedding. The projection width is controlled by `multimodal.tabular_projection_dim`.

The CS2007 soil runner additionally supports:

- `soil_only`
- `soil_raw_concat`
- `soil_projected_concat`

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

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts \
    multimodal.fusion_mode tabular_projected_concat \
    multimodal.tabular_encoder mlp_projection \
    multimodal.tabular_projection_dim 32
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

## Fixed-Epoch Train+Validation Final Fit

Use this regime only after selecting the epoch count and other hyperparameters with a separate validation experiment. It trains a fresh classifier on the existing joined `train` and `val` tables for exactly `multimodal.train_epoch` epochs, then evaluates the final epoch on `test`. Validation is not evaluated and cannot stop or select the model.

Keep `data.data_split.valid_split` non-zero when exporting embeddings and building joined tables. Final fit consumes those materialized train and validation artifacts; it does not support an empty validation table.

The two supported trainer configurations are:

- `train_on_train_val: False`, `early_stopping: True`: validation-selected training (the default and existing behavior).
- `train_on_train_val: True`, `early_stopping: False`: fixed-epoch train+validation final fit.

Other combinations raise a configuration error. In final-fit mode, `report_test_each_epoch: False` evaluates test only after training. Setting it to `True` also records test metrics in every history entry and retains an explicitly diagnostic `oracle_test_best` result selected by test top-1 accuracy. This oracle result is test-set leakage and must not be used as the primary reported result.

Use a distinct `run_tag` so the final-fit checkpoint does not overwrite validation-selected results:

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs_geo_100m.yaml \
  --opts \
    seed 1 \
    multimodal.image_feature_source habitat_finetuned \
    multimodal.fusion_mode raw_concat \
    multimodal.export_image_embeddings False \
    multimodal.build_joined_tables False \
    multimodal.train_on_train_val True \
    multimodal.early_stopping False \
    multimodal.train_epoch 20 \
    multimodal.report_test_each_epoch False \
    multimodal.run_tag gse_100m_train_test_epoch20
```

The final epoch is stored as `best_model.pt` for output compatibility. Its checkpoint payload and `metrics.json` identify the regime as `fixed_epoch_train_val`; the scaler records `fit_split: train_val` and the combined fit row count. When per-epoch test reporting is enabled, `oracle_test_best_model.pt` and `oracle_test_best_confusion_matrix.npy` retain the earliest epoch achieving the highest test top-1 accuracy. The primary `test` metrics and `test_confusion_matrix.npy` still describe the final-epoch model.

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
- geo inputs must contain `file` and `A00..A63` columns
- geo duplicates are deduplicated before joining
- rows with missing geo features are dropped
- join is performed separately within each split

### Geo Dedup Policy

If multiple geo rows exist for the same file:

- prefer the row with non-empty `BH_PLOT_DESC`
- allow duplicates only if all `A00..A63` values agree
- fail fast if duplicate rows disagree

This is intentional. Silent disagreement in geo features is treated as a data error.

### Optional Cleaned Test Set

The 100m CS 2019-2023 suite can evaluate on an expert-cleaned test set by enabling `data.cleaned_test`.

Default config in `configs/multimodal_base.yaml` keeps this disabled:

```yaml
data:
  cleaned_test:
    enabled: False
    review_csv: null
    file_column: 'file_name'
    flag_column: 'Confirm to remove (Yes/No)?'
    remove_values:
      - 'Yes'
```

When enabled, `build_joined_tables()` keeps the exported image embeddings unchanged, filters only the `test` image feature table before the geo join, and records the filtering details in `test_manifest.json` under `cleaned_test`. Train and validation tables are unchanged.

The provided `configs/multimodal_cs_geo_100m_cleaned_test.yaml` uses:

- review CSV: `data/CS_Xplots_2019_2023_test/image_list_25022026-SR-review.csv`
- removed flag: `Confirm to remove (Yes/No)? == Yes`
- distinct artifact tags: `gse_100m_cleaned_test`

With the current review CSV, validation reports 1,398 original test rows, 51 removed rows, and 1,347 cleaned test rows. The same cleaned joined test table is used by `image_only`, `geo_only`, and `raw_concat`.

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
            oracle_test_best_model.pt                 # final-fit diagnostic, optional
            oracle_test_best_confusion_matrix.npy     # final-fit diagnostic, optional
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

### 5. Projected image + geo fusion

```bash
python multimodal_main.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs.yaml \
  --opts \
    multimodal.fusion_mode tabular_projected_concat \
    multimodal.tabular_encoder mlp_projection \
    multimodal.tabular_projection_dim 32
```

This keeps the same geo-matched sample universe as `raw_concat`, but learns a `64 -> tabular_projection_dim` projection for the geo branch before concatenating it with the image embedding.

### 6. CS 10m curated image + geo fusion

```bash
python tools/run_multimodal_cs_geo_10m.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs_geo_10m.yaml
```

The 10m runner reads `data/cs_geo_gse_10m/CS_Xplots_10m_curated_train_test_split.csv`, drops rows with `split == removed`, resolves each usable file across the configured train/test image roots, creates a grouped validation split from curated `train` IDs only, keeps curated `test` unchanged, joins by `file` to `A00..A63`, and then calls the shared classifier trainer. The default fusion mode is `raw_concat`.

Useful 10m ablations:

```bash
python tools/run_multimodal_cs_geo_10m.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs_geo_10m.yaml \
  --opts multimodal.fusion_mode geo_only
```

```bash
python tools/run_multimodal_cs_geo_10m.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs_geo_10m.yaml \
  --opts \
    multimodal.fusion_mode tabular_projected_concat \
    multimodal.tabular_encoder mlp_projection \
    multimodal.tabular_projection_dim 32
```

### 7. Full CS 2019-2023 image + 100m GSE suite

```bash
python tools/run_multimodal_cs_geo_100m.py --validate_data_only
python tools/run_multimodal_cs_geo_100m.py --inspect_only --seeds 1 2 3 4 5
python tools/run_multimodal_cs_geo_100m.py --seeds 1 2 3 4 5
```

For every seed, the runner trains fine-tuned and pretrained `image_only` models, one shared `geo_only` model, and fine-tuned and pretrained `raw_concat` models. The runner validates the current source and joined artifacts dynamically, resumes valid completed runs, and writes suite reports under `multimodal_artifacts/reports/cs/<joined_table_tag>/`.

### 8. Full CS 2019-2023 image + 100m GSE suite with cleaned test set

Use the cleaned-test dataset config to exclude expert-confirmed unreliable test samples from the joined test table:

```bash
python tools/run_multimodal_cs_geo_100m.py \
  --dataset_config configs/multimodal_cs_geo_100m_cleaned_test.yaml \
  --validate_data_only \
  --seeds 1
```

Runnable example script:

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/hshi/anaconda3/envs/habcls/bin/python}
CONFIG=configs/multimodal_cs_geo_100m_cleaned_test.yaml
SEEDS=(1 2 3 4 5)

$PYTHON tools/run_multimodal_cs_geo_100m.py \
  --dataset_config "$CONFIG" \
  --validate_data_only \
  --seeds "${SEEDS[@]}"

$PYTHON tools/run_multimodal_cs_geo_100m.py \
  --dataset_config "$CONFIG" \
  --seeds "${SEEDS[@]}"
```

This run reuses full image embedding exports, builds cleaned joined tables under `joined_table_tag: gse_100m_cleaned_test`, writes model outputs under `run_tag: gse_100m_cleaned_test`, and writes reports under `multimodal_artifacts/reports/cs/gse_100m_cleaned_test/`.

### 9. CS2007 image + soil projected fusion

```bash
python tools/run_multimodal_cs2007_soil.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs2007_soil.yaml
```

The soil runner uses the CS2007 soil-aligned split from `data.cs2007_soil_aligned`, exports deterministic image embeddings, attaches the three soil chemistry features as `S00..S02`, and trains the configured classifier. The default soil fusion mode is `soil_projected_concat`, where the trainable soil branch projects `3 -> 32` before concatenation with image features.

Useful soil ablations:

```bash
python tools/run_multimodal_cs2007_soil.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs2007_soil.yaml \
  --opts multimodal.fusion_mode image_only
```

```bash
python tools/run_multimodal_cs2007_soil.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs2007_soil.yaml \
  --opts multimodal.fusion_mode soil_only
```

```bash
python tools/run_multimodal_cs2007_soil.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs2007_soil.yaml \
  --opts multimodal.fusion_mode soil_raw_concat
```

### 10. CS2007 soil projection-dimension grid search

```bash
python tools/run_multimodal_cs2007_soil_projection_grid.py \
  --base_config configs/multimodal_base.yaml \
  --dataset_config configs/multimodal_cs2007_soil.yaml \
  --dims 4 8 16 32 64 128
```

This runs `soil_projected_concat` once per projection dimension. Image embeddings and joined soil tables are reused if the split parquets already exist; use `--force_export_image_embeddings` or `--force_build_joined_tables` to regenerate them.

Each grid point is saved with a run tag such as `projdim_32`, for example:

```text
multimodal_artifacts/runs/cs2007_soil_aligned/<encoder>/soil_projected_concat/projdim_32/seed1/
```

The aggregate grid results are saved by default under:

```text
multimodal_artifacts/runs/cs2007_soil_aligned/<encoder>/soil_projected_concat/projection_dim_grid/seed1/
```

### 11. Image-only baseline on the geo-matched subset with reused joined table

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

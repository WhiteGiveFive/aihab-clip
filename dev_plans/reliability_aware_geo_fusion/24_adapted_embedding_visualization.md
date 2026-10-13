# Adapted image and geo embedding spaces: development and cleaned test

Recorded 2026-10-05. This visualization extends the
[adapted-encoder geometry study](21_embedding_geometry_correction.md) and the
[router confusion review](23_geometry_router_confusion_review.md) by showing
the image representation itself. The earlier embedding overview visualized the
frozen pretrained SigLIP2 encoder; this follow-up uses the four cached
habitat-adapted image encoders. The same report also compares these image
representations with the cached 100 m AlphaEarth/GSE geo embeddings, for both
development and cleaned-test samples.

The generated
[visualization report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_embedding_visualization_v1/report.md)
contains the plots, projection diagnostics and source provenance. It lives in
the standard versioned geo-helpfulness artifact directory and leaves the
earlier study bundles unchanged.

## Samples and encoders

Each encoder was fitted on the same **3,378 development-training images**.
For each of the four encoder seeds, the visualization includes:

| Split | Images | Role |
| --- | ---: | --- |
| Development train | 3,378 | Encoder training and PCA fitting |
| Development validation | 822 | Held out from encoder fitting; transformed by the training PCA |
| All development | 4,200 | Train and validation together, with original split membership retained |
| Cleaned test | 1,347 | Held out from encoder fitting; transformed by the training PCA |

The total is **5,547 distinct images per encoder**, rather than four independent
datasets. Seeds 1–4 encode the same image cohort separately. Reference and query
embeddings are not mixed across encoder seeds.

Inputs are the existing `embeddings.npz` and `manifest.json` files in
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_geometry_v1/embeddings/seed_{1,2,3,4}/`,
together with the saved cohort identities and habitat labels. This operation
does not fit encoders, re-extract image embeddings, repeat OOF diagnostics,
refit correction thresholds or evaluate new router policies.

Geo inputs are the existing 64-dimensional 100 m AlphaEarth/GSE embeddings
(`A00`–`A63`) in
`multimodal_artifacts/joined_tables/cs/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_pretrained/gse_100m/seed1/{train,val,test}.parquet`,
aligned to the same 5,547 image identities. The saved router cohort metadata
determines the development train/validation assignment; the joined-cache files
use a different partition and their filenames do not determine the visualization
split. The original test cache is filtered to the exact 1,347 cleaned-test
identities. Photos of one plot share a geo vector, so geo projection fitting and
plotting use **one point per plot**:

| Split | Geo points (plots) | Role |
| --- | ---: | --- |
| Development train | 1,300 | Geo PCA fitting |
| Development validation | 325 | Transformed by the training geo PCA |
| All development | 1,625 | Train and validation together |
| Cleaned test | 531 | Transformed by the training geo PCA |

The geo projection therefore contains **2,156 plots**, rather than repeated
copies for each photograph. Geo embeddings do not depend on the adapted image
encoder seed, so all four image comparisons reuse the same geo coordinates.
The per-photo geo coordinate table broadcasts a plot's coordinates back to its
images for identity alignment; those repeated rows are not extra projected
points.

## Projection protocol and interpretation

For each image encoder seed, a 50-component PCA is fitted using only its 3,378
development-training embeddings. Validation and cleaned-test embeddings are
transformed with the same fitted PCA. The first two components provide the
PCA plot. Thus the development and test panels use the same axes within an
encoder seed and PCA provides an actual held-out projection for validation and
test.

t-SNE is fitted jointly to the PCA-transformed embeddings of all 5,547 images,
using perplexity 30, at most 1,500 iterations and fixed random seed 20261005. Its layout
uses no habitat labels, but **includes test embeddings**: this is a descriptive,
transductive visualization, not an out-of-sample transformation or an estimate
of test generalization. Habitat labels are used to color the resulting points.
Original split membership is retained for separate development/test panels and
interactive inspection.

For each projection and encoder seed, split panels share coordinate limits.
The habitat focus views reuse those same projected points; they do not refit a
projection on each habitat subset. Different encoder seeds are projected
independently, so their axes, positions and t-SNE orientations cannot be
compared point by point.

For geo, each original 64-dimensional vector is L2-normalized before projection;
coordinates are **not standardized individually**. This restores unit length
after spatial aggregation while retaining the vector direction. A separate
50-component PCA, with full SVD and no whitening, is fitted only on the 1,300
development-training plot vectors and then applied to validation and cleaned
test. Geo t-SNE uses all 2,156 plots after this training-fitted PCA, with the same
perplexity 30, 1,500-iteration limit and random seed 20261005 as the image views.
It is also label-free during fitting and includes test embeddings in its layout.

The combined figures show image and geo spaces beside each other with the same
habitat colors. Development/test panels share axes within a modality and
projection, but image and geo projections are fitted separately. **Axis values,
distances, orientations and cluster positions cannot be compared across the two
spaces.** The sampling units also differ: image panels show photographs, while
geo panels show plots. Point density and apparent cluster size therefore need
to be interpreted with those different sampling units in mind. These views
show complementary structure, not a direct numerical comparison of modality
quality.

PCA's retained variance and the t-SNE fit diagnostics are recorded alongside
the plots. A 2D projection necessarily discards information: visual overlap
does not establish that classes are inseparable in the full embedding space,
and apparent t-SNE cluster separation is not a calibrated distance or a router
confidence measure. Encoders have seen development-training labels, so tighter
training clusters would not by themselves demonstrate held-out improvement.
These plots supply descriptive context for the geometry study; they do not
select a policy or alter the original M4 decision.

## Reproduce the visualizations

Run from the repository root in the existing `habcls` environment, with the
adapted embedding caches, the 100 m geo cache and saved cohort metadata available:

```bash
conda activate habcls
python tools/plot_multimodal_geo_helpfulness_adapted_embeddings.py
```

The default output directory is
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_embedding_visualization_v1/`.
The default command generates both the adapted-image and geo comparisons. On a
completed run, it verifies the saved numerical projection manifests and reuses
the coordinates to regenerate the report and figures. It does not refit either
projection on that reuse path. Adding geo to the earlier image-only output
retains the existing image coordinates and computes the missing geo projection.

To recompute PCA and t-SNE into a fresh destination:

```bash
python tools/plot_multimodal_geo_helpfulness_adapted_embeddings.py \
  --output-dir multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_embedding_visualization_reproduction_v1
```

The canonical
[plot entrypoint](../../tools/plot_multimodal_geo_helpfulness_adapted_embeddings.py)
uses
[projection and provenance code](../../multimodal/geo_helpfulness_adapted_embeddings.py)
and a separate
[figure renderer](../../multimodal/geo_helpfulness_adapted_embedding_figures.py).
The geo extension uses
[geo projection and provenance code](../../multimodal/geo_helpfulness_geo_embedding_visualization.py)
and the
[image/geo comparison renderer](../../multimodal/geo_helpfulness_image_geo_figures.py).
Projection settings and numerical-library versions are saved with the outputs.

| Output | Purpose |
| --- | --- |
| `report.md` | Plot guide, methodology and projection diagnostics |
| `coordinates.parquet` | Saved projections with image, seed, split and habitat identity |
| `projection_metrics.json`, `class_support.csv` | Projection diagnostics and habitat/split support |
| `projection_manifest.json` | Projection settings, checkpoint/cache lineage and numerical output hashes |
| `provenance.json` | Input preservation and generated-output provenance |
| Per-seed PCA transform `.npz` files | Fitted training PCA transforms |
| `figures/adapted_image_embedding_seed{1,2,3,4}_overview.{png,pdf}` | PCA and t-SNE development/test panels for each encoder |
| `figures/adapted_image_embedding_seed{1,2,3,4}_habitat_focus.{png,pdf}` | Common-geometry views of previously discussed habitat groups |
| `figures/adapted_image_embedding_all_seeds_tsne.{png,pdf}` | Development/test t-SNE overview across all four encoders |
| `interactive/adapted_image_embedding_seed{1,2,3,4}_interactive.html` | Interactive inspection of the same points and metadata |
| `geo_coordinates.parquet` | Geo PCA and t-SNE coordinates, one row per plot |
| `geo_photo_coordinates.parquet` | The same geo coordinates broadcast to the 5,547 image identities |
| `geo_plot_metadata.parquet`, `geo_class_support.csv` | Plot identity, original split and habitat support |
| `geo_projection_metrics.json`, `geo_projection_manifest.json` | Geo projection diagnostics, settings, source lineage and numerical output hashes |
| `geo_pca_transform.npz` | PCA transform fitted on 1,300 development-training geo vectors |
| `figures/geo_embedding_overview.{png,pdf}` | Geo PCA and t-SNE development/test panels |
| `figures/adapted_image_geo_seed{1,2,3,4}_{pca,tsne}.{png,pdf}` | Image/geo comparison with development and cleaned-test panels |
| `interactive/adapted_image_geo_seed{1,2,3,4}_interactive.html` | Interactive comparison of image and plot-level geo spaces |

The
[study catalog](../../configs/multimodal_geo_helpfulness_studies_v1.json)
registers this visualization separately from the original embedding extraction
and correction evaluation studies.

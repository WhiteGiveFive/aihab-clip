# Adapted geometry, RF correction and fusion: confusion review

Recorded 2026-10-05; combined-rule analysis added 2026-10-06. This review extends the
[September 15 geo-helpfulness report](reports/2026-09-15_geo_helpfulness_router/report.md)
with a direct comparison of the fusion baseline, its historical RF correction,
and correction using habitat-adapted image geometry. The generated
[comparison report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_router_confusion_review_v1/report.md)
contains the confusion matrices, per-habitat results, intervention analysis,
reference-bank/neighborhood sensitivity and a comparison of two rules for
combining the RF and adapted geometry routers.

## Matched comparison

All methods use the same previously evaluated **1,347 cleaned-test images from
531 plots**, with four training seeds and the historical final-epoch-50 expert
family used in the parent report. The four runs are repeated predictions of the
same images; pooled confusion counts do not represent four independent test
sets. Matrices use a common habitat order, true habitats as rows, predicted
habitats as columns, and row normalization for recall comparisons. Habitats
without test support are marked rather than treated as measured zero recall.

The RF comparator is
`relaxed__rf__primary_full727__plots_20__harm_0p7`, matching the parent report.
The illustrative geometry comparison uses **4,200 reference images and strict
K=5**, retaining fusion if either candidate lacks five distinct reference plots.
Its threshold is the already saved development-validation selection with a
20-plot minimum and H≤0.7R. Reference and query image embeddings come from the
same seed-specific habitat-adapted encoder fitted only on the 3,378
development-training images. The classification experts remain unchanged.

Both reference banks (3,378 and 4,200) and both strict neighborhoods (K=5 and
K=10) remain in the review. Choosing an illustrative matrix does not select or
promote a test winner. The earlier
[adapted-encoder study](21_embedding_geometry_correction.md) documents the full
policy specification, encoder provenance and validation/test distinctions.

This is a descriptive analysis of saved predictions. It performs no new model
fitting, image inference, threshold search or test policy selection, and leaves
the parent report and sealed study outputs unchanged. The cleaned test has
already informed earlier exploration, so this review supplies error analysis
rather than independent confirmation; the original M4 no-go decision remains
unchanged.

## Main findings

| Method | Accuracy (%) | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| Fusion | 74.3133 | 0.73835 | 0.49013 |
| RF correction | 75.1670 | 0.74651 | 0.50403 |
| Adapted geometry, 4,200 references, K=5 | 75.7053 | 0.74904 | 0.48857 |

Geometry's strongest additional recall gains over RF are Bog (+4.91 percentage
points) and Acid Grassland (+2.12 points). Improved Grassland recall ties RF.
RF retains higher recall for Coniferous Woodland, Bracken, Fen/Marsh/Swamp,
Calcareous Grassland and Montane. Both routers reduce Improved-to-Neutral
errors while increasing errors in the reverse direction relative to fusion.

Geometry makes 156 rescues and 81 harmful changes, versus RF's 129 rescues and
83 harms, pooling the four seeds. Their net gains over fusion are 75 and 46
correct seed–image predictions. Their accuracy difference is +0.54 points;
the existing paired 95% interval is [−0.59, +1.68] points and includes zero.
Geometry's macro F1 is lower than both baselines. Its three lost correct
Calcareous predictions come from one photo repeated across three seeds.

The principal Bog/Acid gains and Fen/Neutral/Calcareous losses relative to fusion
persist across both reference banks and K=5/10. The full report includes
precision and F1 as well as recall, since routing changes false positives too.

## Combining RF and adapted geometry interventions

By default, the 2026-10-06 extension applies two rules to the already saved
decisions from the historical RF router and the **4,200-reference, strict-K5 adapted geometry
router**. The **OR rule** switches fusion to geo when either router intervenes;
the **AND rule** switches only when both intervene. Both component routers
switch to the same geo prediction, so shared interventions have no destination
conflict. All other cases retain fusion.

The RF router intervenes on 264 seed–image decisions and geometry on 293,
with 99 shared decisions. Their union therefore contains **458 decisions**
(264 + 293 − 99). The counts below pool the same four seeds over 1,347 images;
they are not counts of distinct photographs.

| Policy | Interventions | Rescues | Harms | Both wrong | Net gain over fusion | Accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 264 | 129 | 83 | 52 | +46 | 75.1670 |
| Adapted geometry, 4,200 references, K=5 | 293 | 156 | 81 | 56 | +75 | 75.7053 |
| Either router (OR) | 458 | 224 | 136 | 98 | +88 | 75.9465 |
| Both routers (AND) | 99 | 61 | 28 | 10 | +33 | 74.9258 |

Rescue means fusion was wrong and the intervention becomes correct; harm means
fusion was correct and the intervention becomes wrong. A both-wrong change
switches between two incorrect predictions. OR captures 68 more rescues than
geometry alone but adds 55 harms, giving 13 more correct decisions overall.
AND retains fewer interventions and harms, but also misses more rescues. Its
net gain is 33 rather than geometry's 75. Thus the higher rescue proportion of
the overlap does not imply a larger overall accuracy gain.

OR has mean weighted F1 **0.75212** and macro F1 **0.50005**; AND has weighted
F1 **0.74336** and macro F1 **0.49250**. OR's macro F1 exceeds geometry's
0.48857 but remains below RF's 0.50403.

These combinations were proposed after examining cleaned-test overlap. Their
results are **exploratory**, not validation-selected combined policies or an
independent test of a previously chosen combination. The component thresholds
remain fixed, with no new fitting or threshold search. The generated report
also gives weighted/macro F1, seed-level results and per-habitat metrics for
fusion, RF, geometry, OR and AND.

## Reproduce the report and figures

Run from the repository root in the existing `habcls` environment, with the
saved geometry and historical RF/expert prediction bundles available:

```bash
conda activate habcls
python tools/plot_multimodal_geo_helpfulness_confusion_review.py
```

The default destination is
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_router_confusion_review_v1/`.
To write a fresh review directory explicitly:

```bash
python tools/plot_multimodal_geo_helpfulness_confusion_review.py \
  --output-dir multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_router_confusion_review_reproduction_v1
```

`--reference-bank {3378,4200}` and `--k {5,10}` change the illustrative geometry
panel and the geometry component used in the OR/AND comparison; all four
standalone geometry combinations are still exported. The combined-rule results
above use the default 4,200-reference K5 component. These options choose among
previously evaluated policies and do not refit thresholds. The implementation
is [geo_helpfulness_confusion_review.py](../../multimodal/geo_helpfulness_confusion_review.py),
accessed through the canonical
[plot entrypoint](../../tools/plot_multimodal_geo_helpfulness_confusion_review.py).

| Output | Purpose |
| --- | --- |
| `report.md`, `figures/` | Human-readable comparison and confusion-matrix figures |
| `summary.csv`, `per_seed_metrics.csv`, `metrics.json` | Aggregate and seed-level performance summaries |
| `per_class.csv`, `confusion_cells.csv` | Per-habitat metrics and complete confusion-cell comparisons |
| `interventions.csv` | Corrections, harms and other prediction changes |
| `combined_rules_summary.csv`, `combined_rules_per_seed.csv` | Fusion, RF, geometry, OR and AND performance and intervention counts |
| `combined_rules_per_class.csv` | Per-habitat performance of the five combined-rule comparison methods |
| `combined_rules_predictions.parquet` | Paired per-image, per-seed predictions and decisions for reproducing OR/AND |
| `provenance.json` | Input identity and reproduction provenance |

The same command regenerates the combined-rule tables and report section from
the saved predictions. `metrics.json` includes their metadata and confusion
matrices. The original September 15 report and sealed geometry bundles remain
unchanged.

The [project-wide catalog](../../configs/multimodal_geo_helpfulness_studies_v1.json)
registers this as a separate review under the standard versioned artifact root.

# M3J seed 1: can geo rescue attention-fusion errors?

Recorded: **22 October 2026**. Status: **Completed saved-prediction diagnostic;
geometry routing has not yet been applied to these models.**

**Yes: the geo expert supplies the correct answer for a substantial subset of
attention-fusion errors.** For CLS + original embeddings at the best-validation
checkpoints, there are **69–98 potential rescues per setting**, representing
**29.87–39.20% of fusion errors**. An ideal router would improve Acc@1 by
**8.39–11.92 percentage points**. These are label-informed upper bounds, not
measured gains from a geometry router.

The opportunity comes with a substantial risk: the same settings contain
**127–141 correct fusion predictions that geo would get wrong**. The next study
must establish whether geometry can identify rescues while avoiding enough of
these harmful switches.

## Scope and expert identity

The analysis uses the **822 development-validation images from 325 plots** and
the completed **M3J seed-1** fits. It reads saved, reloaded-checkpoint predictions;
it does not train models, run image inference, or score the cleaned test set.

- Fusion: all four attention architectures under both initializations
  (`pretrained`, `cs_adapted`) and both regimes (`joint_partial`, `frozen_online`).
  The main focus is `attention_cls_skip_original`.
- Geo: the **new M3J shared seed-1 geo-only fit**, using training-fitted
  per-feature standardization of geo embeddings. This is not the historical geo-helpfulness
  study's expert or the earlier M3 cached-feature geo-only fit.
- Policies: **final fusion + final geo**, and **best-validation fusion +
  best-validation geo**. Final means epoch 50; geo's best checkpoint is epoch
  11. Each fusion model retains its own previously selected best epoch.
- Controls: image-only, raw concatenation, projected concatenation and the
  tokenwise capacity control are included in the numerical bundle. In total,
  there are **32 image-using models × 2 checkpoint policies = 64 paired
  comparisons**, each on the same 822 images.

Geo Acc@1 is **63.02% (518/822)** at the final checkpoint and **64.84%
(533/822)** at its best checkpoint. Lower aggregate accuracy does not prevent
geo from correcting some fusion errors: the two predictors make different
mistakes. Attention fusion already consumes geo features, but its predicted
class need not match the standalone geo classifier's class.

"Original embeddings" means the current image encoder's unprojected global
embedding and the standardized geo embedding, concatenated with updated CLS.
It does not mean that the image encoder must have pretrained initialization.

## What counts as a rescue?

Let F and G denote the top-1 predictions from fusion and geo, and y the true
validation label. There are four disjoint states:

| State | Fusion | Geo | Effect of switching to geo |
| --- | --- | --- | --- |
| Both correct | Correct | Correct | No change |
| Potential rescue, R | Wrong | Correct | One additional correct prediction |
| Potential harm, H | Correct | Wrong | One fewer correct prediction |
| Both wrong | Wrong | Wrong | No top-1 accuracy gain; may change the wrong class |

An **oracle rescue router** switches exactly the R cases and keeps fusion
otherwise. It uses ground-truth labels and is unavailable at inference. Its
accuracy is the maximum achievable by selecting between these two fixed top-1
predictions:

\[
\operatorname{Acc}_{\mathrm{oracle}}
=\operatorname{Acc}_{F}+R/822.
\]

For a real router, the accuracy change is instead
\((R_{\mathrm{selected}}-H_{\mathrm{selected}})/822\). Thus identifying many
rescues is insufficient if it also selects many harms. A both-wrong switch can
also alter the confusion matrix and F1 despite leaving top-1 accuracy unchanged.
This diagnostic concerns top-1 correction; it does not define how to combine
probabilities or top-3 rankings.

The rescue/harm direction is **geo replacing fusion**. The combined M3J report
also contains contrasts in the opposite direction, comparing each image model
against the shared geo baseline; those rescue/harm names must be reversed for
this routing question.

## CLS + original embeddings: best-validation checkpoints

All accuracy entries are percentages. "Rescuable errors" gives both the number
of images and its percentage of that fusion model's errors. Potential harms
are the full available harm pool, not interventions made by an evaluated router.

| Training regime | Initialization | Fusion epoch | Fusion Acc@1 | Fusion errors | Rescuable errors | Potential harms | Oracle Acc@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Partial-joint | Pretrained | 36 | 71.78 | 232 | 84 (36.21%) | 141 | 82.00 |
| Partial-joint | CS-adapted | 32 | 70.44 | 243 | 88 (36.21%) | 134 | 81.14 |
| Frozen-online | Pretrained | 26 | 71.90 | 231 | 69 (29.87%) | 127 | 80.29 |
| Frozen-online | CS-adapted | 28 | 69.59 | 250 | 98 (39.20%) | 137 | 81.51 |

The potential gains are respectively **+10.22, +10.71, +8.39 and +11.92
percentage points**. Rescue cases span **52, 52, 43 and 55 distinct plots**,
respectively. They are not confined to a handful of photographs from one plot,
although photographs within plots are correlated.

The strongest observed standalone fusion here is frozen-online/pretrained at
71.90%, while the largest oracle accuracy is partial-joint/pretrained at
82.00%. This illustrates why standalone accuracy and compatibility with a geo
expert are different criteria. It does not select a routed model: the attainable
gain still depends on geometry's ability to separate rescue from harm.

## CLS + original embeddings: final checkpoints

Both experts use epoch 50 in this table.

| Training regime | Initialization | Fusion Acc@1 | Fusion errors | Rescuable errors | Potential harms | Oracle Acc@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Partial-joint | Pretrained | 71.17 | 237 | 77 (32.49%) | 144 | 80.54 |
| Partial-joint | CS-adapted | 69.34 | 252 | 87 (34.52%) | 139 | 79.93 |
| Frozen-online | Pretrained | 70.07 | 246 | 55 (22.36%) | 113 | 76.76 |
| Frozen-online | CS-adapted | 68.00 | 263 | 96 (36.50%) | 137 | 79.68 |

Potential gains range from **+6.69 to +11.68 percentage points**. The qualitative
opportunity persists at the final epoch. Changes between these tables reflect
both the fusion checkpoint and the geo checkpoint; they do not isolate a change
in fusion alone.

## Where are the correctable errors?

Consider **frozen-online/pretrained, CLS + original embeddings, best epoch 26**,
paired with geo epoch 11. This is the highest standalone Acc@1 among the four
focused settings, chosen here as an illustrative case.

Its 822 validation outcomes break down into:

| Joint outcome | Images |
| --- | ---: |
| Both correct | 464 |
| Fusion wrong, geo correct: potential rescues | 69 |
| Fusion correct, geo wrong: potential harms | 127 |
| Both wrong | 162 |
| **Total** | **822** |

There are **261 disagreements**: 69 rescues, 127 harms and 65 switches between
two wrong classes. The other 97 both-wrong cases have the same wrong label.
Only **26.44% of disagreements** are potential rescues. Switching on every
disagreement would therefore lose **58 correct predictions**, taking accuracy
from **71.90% to geo's 64.84%**. The oracle instead rescues all 69, reaches
**80.29%**, and still leaves **162 errors** that neither expert can correct.

The largest rescue pools by true habitat are:

| True habitat | Validation images | Fusion errors | Geo-rescuable errors | Potential harms |
| --- | ---: | ---: | ---: | ---: |
| Bog | 95 | 37 | 19 | 9 |
| Improved Grassland | 149 | 31 | 18 | 18 |
| Neutral Grassland | 114 | 40 | 12 | 35 |
| Broadleaved Mixed and Yew Woodland | 54 | 11 | 6 | 14 |
| Acid Grassland | 80 | 26 | 6 | 12 |
| Remaining habitats | 330 | 86 | 8 | 39 |
| **Total** | **822** | **231** | **69** | **127** |

Bog and the two grassland classes account for **49/69 (71.01%)** of potential
rescues. Important correctable confusions include **Improved Grassland →
Neutral Grassland (13 images)** and **Bog → Dwarf Shrub Heath (12)**.

The class risks differ: geo supplies more rescues than harms for Bog, but many
more harms than rescues for Neutral Grassland. These are descriptive
ground-truth groupings, not deployable class-based routing rules; the true
habitat is unknown when predicting. Full class tables and individual case
identities are saved below.

## Is the opportunity specific to the bypass model?

No. Each attention architecture has substantial geo-correct/fusion-wrong
overlap. The table reports **potential rescue counts at best-validation
checkpoints**, with raw and projected concatenation as controls.

| Fusion method | Partial-joint, pretrained | Partial-joint, CS-adapted | Frozen-online, pretrained | Frozen-online, CS-adapted |
| --- | ---: | ---: | ---: | ---: |
| Raw concatenation | 52 | 90 | 33 | 76 |
| Projected concatenation | 91 | 86 | 58 | 76 |
| Two-token attention | 79 | 79 | 59 | 104 |
| CLS only | 72 | 93 | 68 | 104 |
| CLS + adapted bypass | 76 | 81 | 60 | 96 |
| CLS + original embeddings | 84 | 88 | 69 | 98 |

Rescue counts alone do not rank routed models: they depend on how many errors
the base model makes and do not measure whether geometry can detect them. Keep
base accuracy, realized rescues, realized harms and resulting accuracy together
in any subsequent router comparison.

## Implications for a geometry-router study

These results justify a **validation feasibility study of geometry routing**,
with CLS + original embeddings as the requested focus and the two
concatenation methods as useful matched controls. The historical method is
documented in the
[embedding-geometry record](../reliability_aware_geo_fusion/21_embedding_geometry_correction.md)
and [router research summary](../reliability_aware_geo_fusion/25_router_research_summary.md).

The method compares the two proposed classes in each modality's embedding
space. Its joint score is the minimum of the image and geo log distance ratios;
a sufficiently positive score supports switching to geo. The previous selected
configuration used five distinct reference plots per candidate class and
retained fusion when support was insufficient.

For a new validation study:

1. Keep these experts and checkpoint policies fixed. Evaluate geometry against
   the **new fusion/geo candidate classes**; historical intervention flags or a
   historical threshold do not establish performance on the new experts.
2. Use **only the 3,378 training images as the labeled reference bank** for
   validation queries. The historical 4,200-reference bank was for subsequent
   test application and would include validation queries here.
3. Declare the geometry encoder separately from the fusion encoder. Reusing
   the historical training-partition-adapted seed-1 geometry encoder is a
   straightforward first comparison. Reference and query embeddings must use
   the same encoder. Geometry based on each partially fine-tuned encoder would
   be a separate choice to evaluate.
4. Preserve the distinction between classifier preprocessing and distance
   preprocessing. M3J classifiers use standardized geo vectors; the historical
   geometry rule separately L2-normalizes native geo and image vectors and
   compares cosine distances within each space.
5. Measure actual rescues, harms, missed rescues, intervention coverage,
   reference support and habitat-level changes. A fixed zero-threshold rule
   can establish an initial baseline; any validation-selected threshold is
   development tuning and requires separate confirmation.

No geometry score, threshold, reference-neighbor search or routed prediction
has been computed in this diagnostic. The finding is **available correction
opportunity**, not evidence that geometry will achieve the oracle bound.

## Reproduction and verification

The input is the completed
[33-fit summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/summary.json)
and each run's `final_predictions.npz`, `best_predictions.npz` and `run.json`.
The three completed training packages are read only. Analysis outputs occupy a
separate directory:

```text
multimodal_artifacts/analysis/cs/gse_100m/attention_fusion/seed1_geo_rescue_opportunity_v1/
```

- [Summary and source hashes](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion/seed1_geo_rescue_opportunity_v1/summary.json): all 64 comparisons, definitions, assurance notices and SHA-256 provenance for 101 small source files; no model-weight rehashing.
- [All model comparisons](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion/seed1_geo_rescue_opportunity_v1/all_models.csv): accuracy, rescues, harms, disagreement decomposition, oracle and plot counts.
- [CLS-original class table](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion/seed1_geo_rescue_opportunity_v1/cls_original_by_class.csv): all 18 classes for eight model/policy combinations; zero-support classes are identifiable by `n=0` and have no estimable class accuracy.
- [CLS-original case table](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion/seed1_geo_rescue_opportunity_v1/cls_original_cases.csv): 6,576 model-policy/image rows with predictions, labels, plot IDs and outcomes. These repeatedly evaluate the same 822 images, not 6,576 distinct images.

Run from the repository root in the `habcls` environment, choosing a new output
directory to preserve the recorded bundle:

```bash
python tools/analyze_attention_fusion_geo_rescue.py \
  --output-dir /tmp/attention_fusion_geo_rescue_reproduction
```

The [analysis script](../../tools/analyze_attention_fusion_geo_rescue.py) aligns
unique row IDs, checks label and plot identity and class order, reproduces the
source Acc@1 values and verifies correctness/disagreement partitions and the
switch-all identity. An independent saved-prediction calculation confirmed all
eight focused comparisons and the habitat breakdown. A second build reproduced
the numerical output files exactly; reordered rows were handled correctly and
mismatched identity/label/plot inputs were rejected in targeted checks.
The [verification receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion/seed1_geo_rescue_opportunity_v1/verification.json)
records these checks and the report hash.

## Interpretation limits

These remain **single-seed, repeatedly inspected development-validation
results**, with the existing **`provisional_unresolved_memory_integrity`**
status. Pairing saved predictions establishes this diagnostic's counts; it does
not resolve earlier hardware concerns. Images within plots are correlated;
counts are image-based rather than independent plot trials. Some habitats have
very small support, and Littoral Rock has no validation examples.

Partial-joint and frozen-online packages also used different realized
shuffle/crop random streams, as recorded in
[D041](03c_seed1_geo_only.md#d041-loader-audit-and-comparison-limitation).
Within-run fusion/geo complementarity remains measurable, but differences
between regimes cannot be attributed solely to encoder trainability. This
analysis does not change M3J's training matrix, choose a final model, or supply
independent test evidence.

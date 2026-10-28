# M3 — Frozen-encoder development comparison

**Status: Complete, 2026-10-16.** Implements M3 in the
[governing guide](README.md) under the unchanged [M1 protocol](01_experimental_protocol.md)
and A001/A002 amendments. M2 verification runs cannot enter this comparison.

## Scope and frozen comparisons

Completed all 228 declared fits: 19 candidate/geo arms, three learning rates and four
paired encoder/training seeds. Each fit uses 3,378 training images, 822 validation
images, 50 fixed epochs, batch 128, AdamW and float32 without TF32 or AMP.
Retained every final checkpoint and its latest epoch-boundary resume checkpoint. No encoder adaptation,
all-development refit, or exploratory test application is part of M3.

The [contrast declaration](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_contrast_declaration.json)
was frozen before the first development fit. Its SHA-256 is
`a63973a3bd57412822a3be764e915302f6026b52cc2185e6a678957645ea95eb`.
It enumerates 63 contrasts independently of results: each geo-using arm against
image-only and, where distinct, matched raw concatenation; two-token attention
against projected concatenation and the capacity control; CLS against projected,
capacity and two-token attention; capacity against projected concatenation; and
L2/standardization against native geo within each geo-using architecture.
The sole primary contrast is **two-token attention/native minus projected
concatenation/native**. Supplementary intervals are descriptive and have no
multiple-comparison adjustment. A secondary gain cannot replace the primary result.

Selection used all 228 complete, authenticated fits. Chose one learning
rate per arm using mean validation Acc@1, then weighted F1, then MCC, then smaller
learning rate. Retained all four seeds and 76 immutable selected-checkpoint references.
Used 10,000 shared whole-validation-plot bootstrap draws (seed 20261012), keeping
all seed predictions paired. Apply M1's outcome categories without a new
practical-effect threshold. These are reused-development-split results rather
than independent confirmation; intervals exclude hyperparameter-selection and
new-split uncertainty.

## Implementation and execution

- [Matrix runner](../../tools/run_attention_fusion_study.py): authenticate protocol,
  caches and encoder bytes, execute the complete matrix, resume partial fits and
  stop on failure or changing input/source identities.
- [Comparison analysis](../../multimodal/attention_fusion_comparison.py): verify
  all candidate payloads, recompute validation metrics, freeze selection and
  compute paired uncertainty, rescue/harm and interpretation.
- [Reporting](../../multimodal/attention_fusion_reporting.py): all-arm/per-seed
  metrics, learning curves, canonical confusion plots, paired contrasts and
  resource summaries. The report uses the repository's draw-cm-for-cs conventions.
- [Analysis CLI](../../tools/analyze_attention_fusion_study.py): explicit declaration,
  analysis and authenticated rendering commands.

Run from the repository root in the `habcls` environment. These commands reproduce
the workflow; completed training bundles are authenticated and reused. Analysis
requires a new destination when the existing output is already complete:

```bash
python tools/analyze_attention_fusion_study.py declare
python tools/run_attention_fusion_study.py run --device cuda --cpu-threads 1
python tools/run_attention_fusion_study.py status
python tools/analyze_attention_fusion_study.py analyze
mkdir -p /tmp/attention_fusion_m3_render/mpl /tmp/attention_fusion_m3_render/cache
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/attention_fusion_m3_render/mpl XDG_CACHE_HOME=/tmp/attention_fusion_m3_render/cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/analyze_attention_fusion_study.py report
```

The `run` command also resumes an interrupted matrix and authenticates completed
bundles before reuse. Declaration reuse is permitted only when its content and
bound sources match. Analysis outputs are immutable; reconstruction uses a new
`--analysis-dir` under the study artifact root. Rendering authenticates analysis
payloads and preserves an identical completed report.

For the separate-process numerical reconstruction performed here:

```bash
python tools/analyze_attention_fusion_study.py analyze --analysis-dir multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_comparison_replay_v1
```

That reconstruction directory now also exists; choose a new name for another
reconstruction. Selected-checkpoint inference can be replayed using the saved
[replay script](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_selected_checkpoint_replay.py)
with `--output` pointing to a new receipt file.

The [orchestration ledger](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/orchestration/m3/status.json)
records 228 completed fits with no training failures. The append-only events
record the first fit at 08:03:45 UTC and last completion at 08:28:19 UTC on
2026-10-16, approximately 24 minutes 35 seconds for the matrix, including
orchestration overhead. Execution used the RTX 4090 with the normally
installed 580.178.04 driver, without the earlier temporary library override.
The frozen image caches remain unchanged. Fusion/head training explicitly
disables both TF32 settings. No reboot or driver installation is performed by M3.

## Primary result

**No improvement under the predeclared M1 rule.** Native-geo two-token attention
reached mean validation Acc@1 **68.80%**, compared with **68.98%** for matched
projected concatenation. The paired difference is **−0.18 percentage points**,
with conditional 95% plot-bootstrap interval **[−0.81, +0.44] pp**. Seed-wise
differences are **[0.00, +0.49, −0.73, −0.49] pp**: the mean is nonpositive and
three of four seeds are nonpositive. The interval crosses zero; this category
does not establish equivalence or statistically certain inferiority.

Acc@3 also declined from 89.08% to 88.26% (−0.82 pp; interval
[−1.55, −0.12] pp). Mean weighted F1 declined by 0.00216 and MCC by 0.00186.
Relative to projected concatenation, attention rescued an average of 12 images
per seed and harmed 13.5, for a net loss of 1.5 correctly classified images.
The paired rescue/harm counts by seed are 11/11, 17/13, 8/14 and 12/16.

The native-geo two-token model also failed to show supported improvement over
raw concatenation (−0.64 pp, no improvement), the capacity control (−0.24 pp,
inconclusive), or image-only (+0.12 pp, inconclusive). None of the three geo
preprocessing arms supports an attention-specific gain against all required
matched controls. The frozen default is not promoted to attention.

## Selected models and secondary findings

Each row uses its own learning rate selected across all four seeds and the
fixed epoch-50 checkpoint. Metrics below are seed means; F1 and MCC are on
their original scale. The full report preserves all 19 arms, all 57 learning-rate
groups, all 228 fits, seed dispersion, and the complete 63-contrast family.

| Model / geo treatment | Selected LR | Acc@1 (%) | Acc@3 (%) | Weighted F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Image-only | 0.0003 | 68.67 | 90.63 | 0.6809 | 0.6407 |
| Geo-only / native | 0.001 | 65.18 | 90.82 | 0.6386 | 0.5995 |
| Raw concatenation / native | 0.0003 | 69.43 | 90.97 | 0.6887 | 0.6495 |
| Projected concatenation / native | 0.0003 | 68.98 | 89.08 | 0.6844 | 0.6443 |
| Token-wise capacity / native | 0.001 | 69.04 | 88.11 | 0.6857 | 0.6456 |
| Two-token attention / native | 0.0001 | 68.80 | 88.26 | 0.6823 | 0.6424 |
| CLS attention / native | 0.0001 | 68.55 | 88.32 | 0.6795 | 0.6398 |
| Raw concatenation / standardized | 0.0003 | **71.38** | **92.52** | **0.7091** | **0.6721** |
| Projected concatenation / standardized | 0.0003 | 70.01 | 90.66 | 0.6949 | 0.6561 |
| Two-token attention / standardized | 0.0003 | 69.25 | 89.39 | 0.6867 | 0.6476 |
| CLS attention / standardized | 0.0003 | 68.95 | 88.59 | 0.6843 | 0.6443 |

Training-fitted coordinate standardization helped the concatenation controls:
raw concatenation improved by **1.95 pp** over its native-geo counterpart
(interval [0.73, 3.28] pp), and projected concatenation by **1.03 pp**
([0.56, 1.54] pp). Both meet the declared supported-improvement rule for these
secondary contrasts. Raw concatenation/standardized has the highest observed
mean Acc@1; its gain over image-only is **2.71 pp** ([1.28, 4.24] pp).
These are conditional development findings, not independent confirmation or
a replacement for the native-geo primary contrast.

Standardizing two-token attention improved its mean Acc@1 by 0.46 pp over
native geo, but the interval [−0.06, 0.98] pp crosses zero and the result is
inconclusive. It remains below both matched concatenation controls. L2
normalization has no supported within-model gain. CLS has no supported gain
over two-token attention; it also changes depth, capacity and readout, so its
contrast is not a pure readout ablation. Across all 63 declared contrasts,
6 are supported improvement, 34 no improvement and 23 inconclusive.

## Learning curves, class changes and resources

The selected native attention and projected-concatenation heads both reach
mean training Acc@1 of 100% by epoch 25 while validation Acc@1 stays near 69%.
Mean validation cross-entropy rises from epoch 10 to 50: 1.425 to 2.384 for
attention and 1.456 to 2.200 for projected concatenation. This is consistent
with overfitting and increasingly confident errors. It is a diagnostic finding;
the comparison still uses the predeclared final epoch and does not retroactively
select an earlier checkpoint.

Class effects are mixed. Against projected concatenation, native attention
loses an average of 1.50 correct Dwarf Shrub Heath images per seed (recall
43.31% versus 45.42%; 71 images from 25 plots), but gains 1.25 Bog images
(63.42% versus 62.11%; 95 images from 36 plots). These descriptive changes do
not amount to uniform habitat improvement. Urban has only three images from
one plot; Littoral Rock has none. The full report provides every class rather
than treating unstable small-class percentages as reliable effects.

The frozen attention head has 359,570 trainable parameters and takes a mean
7.90 seconds per selected fit, versus 227,090 and 4.28 seconds for projected
concatenation: approximately 1.58 times the parameters and 1.85 times the measured
duration. CLS uses 459,410 parameters and 11.73 seconds. These resource figures
exclude encoder adaptation and image feature extraction; they describe this
machine and execution order, not a universal speed benchmark. GPU memory and
process-level CPU high-water marks are retained in the full resource tables.

## Analysis, figures and retained evidence

The [full development report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_comparison_v1/report/report.md)
contains the complete selected-metric table, all-learning-rate training curves,
paired contrast intervals, class support, per-seed confusion matrices and
parameter/time/memory comparisons. PNG and PDF figures accompany machine-readable
CSV tables. Confusion matrices retain the canonical 18-class order; means and
pooled counts are labelled distinctly, and pooled counts repeat the same images
across four seeds rather than representing an ensemble or four independent sets.
Littoral Rock has no validation observations and its recall is unavailable.

| Evidence | Location |
|---|---|
| Frozen 63-contrast declaration | [m3_contrast_declaration.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_contrast_declaration.json) |
| Full authenticated analysis and payload manifest | [manifest.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_comparison_v1/manifest.json) |
| All 19 selections, including all four seeds | [selection.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_comparison_v1/selection.json) |
| All 228 fits and all 63 paired contrasts | [all_fits.csv](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_comparison_v1/tables/all_fits.csv), [paired_contrasts.csv](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_comparison_v1/tables/paired_contrasts.csv) |
| Selected-checkpoint GPU replay | [m3_selected_checkpoint_replay.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_selected_checkpoint_replay.json) |
| Independent analysis reconstruction | [m3_analysis_replay.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_analysis_replay.json) |
| Final implementation checks and validator resolution | [m3_implementation_checks_v2.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_implementation_checks_v2.json) |
| Real-result visual review | [m3_visual_review.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_visual_review.json) |

The study root is
`multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/`.
All 228 `development/frozen_cached/.../final.pt` and `resume.pt` bundles remain
available, with their run specifications, preprocessing states, optimizer/RNG
resume state, full histories, training/validation logits and predictions, source
hashes and manifests. The 76 `selection/frozen_cached/.../selected.json`
references identify the chosen checkpoints without copying or deleting candidates.

## Verification and implementation issues

The final focused suite passed **147 tests**: 50 new M3 tests and 97 existing
attention-fusion tests. It covers matrix identity and resume, source/input
changes, payload tampering, deterministic selection, paired bootstrap logic,
interpretation, source-specific preprocessing validation, report generation,
and the existing model/input/training contracts.

A fresh GPU process reloaded **all 76 selected checkpoints** and recomputed
all 822 validation rows for each. Saved and replayed logits and top-three
predictions were exactly equal, with maximum absolute logit difference 0.0.
This verifies cached-vector inference; online pixel inference is part of M3J.

A second analysis process reconstructed the full analysis in a separate
directory. All eight JSON/CSV/NPZ payloads and the analysis manifest are
**byte-identical**, including the 10,000 shared plot draws and all 63 contrasts
for all four metrics. Independent checks also reproduced every learning-rate
winner and all 252 confidence-interval endpoints from the saved bootstrap
replicates. No training rerun was needed for either verification.

Visual review passed for 17 representative real-result PNGs covering class
support, count/normalized/pooled confusion matrices, selected metrics, learning
curves, contrast intervals and resources. All 466 report payload hashes, all
220 PNG decodes, their 220 PDF counterparts and all report-local links passed
integrity checks. PDF files were checked structurally, not independently
raster-reviewed. The immutable report manifest retains its generated state;
the separate visual-review receipt records acceptance without rewriting it.

The first analysis attempt exposed an overly broad preprocessing-validation
cache key: it reused one expected standardization state across the direct
geo-only and cached-image input producers. Their float64 reductions differ at
approximately 1e-16 because of array memory order. The cache now includes input
producer identity, with exact state comparison retained. Every saved training
state matched its own producer, and both paths yielded identical standardized
float32 inputs on all development rows. A regression test protects this case.
No training settings, inputs, checkpoints or predictions were changed.
The initial failure evidence remains preserved.

The first report-rendering process exited with a segmentation fault after
font-cache errors in the sandbox. Retrying with writable Matplotlib/XDG caches,
Agg and single-threaded BLAS outside the sandbox produced the report. Both the
[initial log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_render_20261016.log)
and [successful retry log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/reports/m3_render_20261016_retry.log)
are preserved; this did not affect fits or numerical analysis.

## Interpretation and next milestone

These results concern the declared frozen global-vector architecture, fixed
50-epoch budget and three-rate search. They do not establish that other attention
architectures or jointly adapted image representations cannot help. Reused
validation data and learning-rate selection make the intervals conditional;
four seeds share the same validation observations and are not four datasets.
Plot grouping does not by itself guarantee independent nearby GSE cells.
Rare/unsupported habitat classes require cautious confusion-matrix interpretation.

M3 preserves a valid negative result without expanding the search after seeing
it. All M3 acceptance requirements are met: complete matched candidate accounting,
retained checkpoints, frozen selection, paired predictions/uncertainty, inspected
figures and an explicit outcome. M3J remains planned: declare its bounded
partial-unfreezing recipe and
matched frozen-online controls before fitting. No joint training, all-development
refit or exploratory cleaned-test application was performed during M3.

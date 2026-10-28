# M3 extension — CLS with modality bypass

**Status: Complete, 2026-10-16.** Delivers both bypass variants, all 72 new fits,
24 selected checkpoints, 37 paired contrasts and reproducibility/figure checks
for the extension in the [governing guide](README.md). Both primary results are
inconclusive. The
[original M3 comparison](03_development_comparison.md) remains complete and
unchanged. This record separates the new development study from its existing
228 fits and original no-improvement result.

## Question and scope

Does direct access to the pre-attention modality features improve the complete
two-layer CLS classifier? Test both adapted-token and original-embedding
bypasses with frozen image encoders. A positive result would concern the full
bypass configuration: direct feature access and classifier capacity change
together. It would not isolate attention, information preservation or head
capacity individually.

This extension was proposed after observing M3 results and reuses the same
validation set. It is additional development evidence, not independent
confirmation. There is no cleaned-test evaluation, all-development refit,
encoder unfreezing or new image extraction in this milestone.

## Architecture and fixed fitting recipe

Let \(x_i\) be the existing L2-normalized 1,152D image vector and \(x_g\) the
64D geo vector after its declared preprocessing. Define
\(t_i^0=P_i(x_i)+e_i\), \(t_g^0=P_g(x_g)+e_g\). The adapters use
`Linear -> GELU -> LayerNorm`, and the two 128D tokens include their learned
modality identities. Both candidates use the existing learned CLS token and
two independent pre-LN attention/FFN blocks, with four heads, FFN width 256,
dropout 0.1 and no extra final encoder norm. Let \(c^2\) be the final CLS.

| Candidate | Classifier input | Width | Expected trainable parameters |
|---|---|---:|---:|
| `attention_cls_skip_adapted` | \([c^2;t_i^0;t_g^0]\) | 384 | 524,946 |
| `attention_cls_skip_original` | \([c^2;x_i;x_g]\) | 1,344 | 770,706 |

The bypass is concatenation, not elementwise addition. It uses pre-attention
features; updated image/geo tokens are not direct classifier inputs. Both heads
retain `Linear -> ReLU -> Dropout -> Linear`, hidden width 256 and 18 outputs.
The adapted bypass remains differentiable through its adapters. In the original
bypass, geo preprocessing is identical in the direct and attention branches.
The parameter counts will be checked against constructed models before fitting.

All fitting settings inherit the unchanged amended M1 recipe: four paired
encoder/training seeds, 3,378 training images from 1,300 plots, 822 validation
images from 325 plots, fixed 18-class mapping, 50 epochs, AdamW, batch 128,
float32, no AMP and both TF32 settings disabled. Each architecture uses native,
per-vector L2 and training-fitted coordinate-standardized geo, and learning rates
`1e-4`, `3e-4`, `1e-3`. The budget is **72 new fits** and **24 selected references**.
No extra capacity-control search is included.

Initialize all fits afresh; do not warm-start selected M3 checkpoints. The new
study identity is `attention_fusion_cls_bypass_v1`, but component initialization
retains the original `attention_fusion_v1` seed namespace. Shared adapters,
modality identities, CLS and attention layers must exactly match original
same-seed initialization. Different-width classifier layers follow the same
deterministic initialization rule without claiming identical weights.

## Prospective selection and contrasts

Choose one learning rate for each of the six candidate/geo arms using four-seed
mean validation Acc@1, then weighted F1, MCC and the smaller rate for exact ties.
Use final epoch 50; never select an individual seed or an intermediate epoch.
Retain all four selected seed checkpoints and every unsuccessful fit.

Declare **37 unique contrasts before the first extension fit**:

- Each bypass against CLS-only, projected concatenation, raw concatenation,
  two-token attention and image-only, for each of three geo arms: 30 contrasts.
  The two native-geo bypass-minus-CLS contrasts are the primary extension
  contrasts. Report both regardless of outcome.
- Original-embedding bypass minus adapted-token bypass within each geo arm:
  three supporting contrasts. This changes both representation and head width.
- L2 minus native and standardization minus native within each bypass:
  four preprocessing contrasts.

Use the existing **10,000 whole-validation-plot bootstrap draws**, seed
`20261012`, with the same plot draw shared across methods and all four seeds.
Compute each draw's difference as the mean of the four paired metric
differences and report percentile 95% intervals. Retain seed dispersion
separately. Use the unchanged M1 supported-improvement, inconclusive and
no-improvement rules; no new practical-effect threshold is introduced.
Report Acc@1, Acc@3, weighted F1, MCC, rescue/harm and all 18-class confusions.

These are individual conditional intervals, without multiplicity adjustment;
they omit model-selection and new-split uncertainty. Neither selecting one of
the two primary outcomes nor selecting a secondary geo arm after scoring can
replace the full declared result. Plot grouping does not ensure independence
of nearby shared GSE cells. Low-support and absent classes remain visible.

## Preservation and implementation boundaries

Use a separate configuration and artifact root under
`multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/`.
Reuse the authenticated M3 development caches and comparators by identity.
Verify complete comparator input, preprocessing, initialization, fitting and
selection provenance before reuse. No existing checkpoint, protocol, selection
record or analysis output may be rewritten to incorporate this extension.

The original model/trainer source files must remain byte-identical so their
strictly source-bound M3 bundles can still replay. New bypass modules reuse the
original CLS components and a private instance of the unchanged training engine.
The private engine binds the new model factory and an expanded source identity
once; it must not mutate the original imported module or switch factories during
a fit. Its dependencies are recorded alongside the extension code. Tests must
verify the isolation, shared initialization and training behavior.

Before scientific fitting, verify input/output widths and bypass contents,
gradient flow, sample-independent inference, all three geo treatments,
deterministic replay and epoch-boundary resume. Training-only smoke checks are
excluded from the 72-fit scientific budget. The comparison retains 228 reused
fits and 72 new fits distinctly; only the new 24 selected references belong to
this extension's selection namespace.

## Implementation and execution

| Component | Implementation |
|---|---|
| Separate recipe and 72-fit budget | [configuration](../../configs/multimodal_attention_fusion_cls_bypass.yaml) |
| Both model variants | [bypass models](../../multimodal/attention_fusion_bypass.py) |
| Isolated reuse of the pinned training engine | [training adapter](../../multimodal/attention_fusion_bypass_training.py) |
| Immutable declaration and original-evidence authentication | [protocol](../../multimodal/attention_fusion_bypass_protocol.py) |
| Sequential matrix execution, fail-fast guards and resume | [runner](../../tools/run_attention_fusion_bypass_study.py) |
| Selection and paired numerical comparisons | [comparison](../../multimodal/attention_fusion_bypass_comparison.py) |
| All-arm metrics, curves, confusions and resources | [reporting](../../multimodal/attention_fusion_bypass_reporting.py), [analysis CLI](../../tools/analyze_attention_fusion_bypass_study.py) |
| Training-only GPU smoke/resume and selected inference replay | [verification CLI](../../tools/verify_attention_fusion_bypass.py) |

Run from the repository root in the `habcls` environment:

```bash
python tools/run_attention_fusion_bypass_study.py declare
python tools/verify_attention_fusion_bypass.py smoke --device cuda --output multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/verification/gpu_smoke_v1
python tools/run_attention_fusion_bypass_study.py run --device cuda --cpu-threads 1
python tools/run_attention_fusion_bypass_study.py status
python tools/analyze_attention_fusion_bypass_study.py analyze
mkdir -p /tmp/attention_fusion_bypass_render/mpl /tmp/attention_fusion_bypass_render/cache
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/attention_fusion_bypass_render/mpl XDG_CACHE_HOME=/tmp/attention_fusion_bypass_render/cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/analyze_attention_fusion_bypass_study.py report
python tools/verify_attention_fusion_bypass.py replay --device cuda --analysis-dir multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_comparison_v1 --output multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/selected_checkpoint_replay_v1.json
python tools/analyze_attention_fusion_bypass_study.py analyze --analysis-dir multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_comparison_replay_v1
```

`run` resumes partial fits and authenticates completed bundles before reuse.
Declaration reuse requires unchanged content and bound sources. Analysis and
verification destinations are immutable: choose new names when replaying these
commands after delivery. Reporting authenticates its numerical inputs and
preserves an identical completed report. Existing selected references are reused
only when their content is identical; unsuccessful fits are retained.

The declaration precedes every extension fit. Effective configuration SHA-256:
`6e612b059ecad520bc89cea3bdd4c2bf1c83069dd9fbdb91cc52e94e0ea4e023`.
Protocol payload SHA-256:
`efe8f9cf878bca58b4ddb45de8804f7ac1f3d627c3796f70a3a1869004c0fadb`.
The protocol binds the 37-contrast file with SHA-256
`b0620e89f39280e9ec5ef10a432ce75e6e6c4b083837ab5ea0c0593409ad70ee`.
Original evidence authentication covers 228 bundles, 76 selected references and
2,618 original files, including cache/encoder and analysis identities.

The GPU preflight reports an RTX 4090 with driver 580.178.04 and normal CUDA
access. No reboot, driver change, image extraction or original-study rewrite
is performed. Fusion/head training explicitly disables both TF32 settings.

All **72 fits completed without failures**, from 10:31:37 to 10:46:30 UTC on
2026-10-16: approximately 14 minutes 53 seconds including orchestration overhead.
Analysis authenticated all 300 fits and retained 24 new selected references,
alongside 76 unchanged original references. The analysis manifest SHA-256 is
`ace81c6d49e194098a106bab625d1f0cf58f0bfe0030cea37bada611e2dd1e2e`.

## Results

**Both primary comparisons are inconclusive.** Neither bypass supports an
improvement over CLS-only under the declared M1 rule. These results use the
822-image validation set, not the cleaned test set.

| Native-geo primary comparison | Bypass Acc@1 | CLS-only Acc@1 | Paired difference | Conditional 95% interval | Outcome |
|---|---:|---:|---:|---:|---|
| Adapted-token bypass minus CLS-only | 68.61% | 68.55% | +0.06 pp | [−0.70, +0.81] pp | Inconclusive |
| Original-embedding bypass minus CLS-only | 68.46% | 68.55% | −0.09 pp | [−0.88, +0.67] pp | Inconclusive |

Adapted-token seed differences are **[+0.36, +0.12, −1.34, +1.09] pp**.
Although three seeds improve, the interval crosses zero and mean Acc@3 falls
by 0.21 pp. Mean weighted F1 changes by +0.00088 and MCC by +0.00039.
It rescues 20.0 images and harms 19.5 per seed on average, a net gain of
0.5 images. The original-embedding differences are
**[+0.36, +1.09, −1.70, −0.12] pp**. Two seeds improve and two decline;
its negative mean therefore does not meet the predeclared three-nonpositive-seed
rule for the `no_improvement` category. Mean Acc@3 changes by +0.21 pp,
weighted F1 by +0.00078 and MCC by −0.00126. Its average rescue/harm counts
are 21.5/22.25, a net loss of 0.75 images. Neither result establishes equivalence.

The table retains all six new arms. Values are four-seed means; F1 and MCC are
on their original scale. Every arm uses its own selected rate and epoch 50.

| Model / geo treatment | Selected LR | Acc@1 (%) | Acc@3 (%) | Weighted F1 | MCC |
|---|---:|---:|---:|---:|---:|
| CLS-only / native (reused comparator) | 0.0001 | 68.55 | 88.32 | 0.6795 | 0.6398 |
| Adapted-token bypass / native | 0.0001 | 68.61 | 88.11 | 0.6804 | 0.6402 |
| Adapted-token bypass / L2 | 0.0003 | 68.61 | 88.26 | 0.6810 | 0.6403 |
| Adapted-token bypass / standardized | 0.0001 | **69.07** | **88.87** | **0.6851** | **0.6453** |
| Original-embedding bypass / native | 0.001 | 68.46 | 88.53 | 0.6803 | 0.6385 |
| Original-embedding bypass / L2 | 0.0003 | 68.61 | 87.74 | 0.6817 | 0.6406 |
| Original-embedding bypass / standardized | 0.0003 | 68.52 | 88.41 | 0.6803 | 0.6396 |
| Raw concatenation / standardized (reused comparator) | 0.0003 | 71.38 | 92.52 | 0.7091 | 0.6721 |

Across all **37 contrasts**, there are **zero supported improvements,
17 inconclusive results and 20 no-improvement results**. Both bypasses receive
`no_improvement` against matched raw concatenation in every geo arm. The highest
new mean is the adapted-token standardized arm, but it is 2.31 pp below matched
standardized raw concatenation (interval [−3.73, −0.92] pp). The
original-embedding standardized arm is 2.86 pp below that comparator
([−4.43, −1.34] pp). Standardized raw concatenation remains the highest observed
mean among all 25 arms. These secondary findings do not replace either primary.

The extension does not justify promoting a bypass model or an attention-specific
gain. It evaluates these two complete architectures under the fixed recipe;
it does not establish that every possible skip path or joint encoder adaptation
would fail. M3J remains a separate planned study.

## Learning curves, class analysis and resources

Both selected native bypass models reach 100% mean training Acc@1 by the final
epoch while validation Acc@1 remains near 68.5%. From epoch 10 to 50, mean
validation cross-entropy rises from 1.526 to 2.562 for adapted-token bypass and
from 3.156 to 4.289 for original-embedding bypass. This is consistent with
overfitting and increasingly confident errors. No earlier checkpoint is selected
after seeing these curves.

Class changes are mixed. Relative to native CLS-only, adapted-token bypass gains
2.00 correct Neutral Grassland images per seed on average (114 images / 42
plots), but loses 2.25 Acid Grassland images (80 / 32) and 1.50 Bog images
(95 / 36). Original-embedding bypass gains 3.75 Neutral Grassland images but
loses 1.50 Acid Grassland and 1.25 Bog images. These are descriptive changes in
correct counts, not class-specific uncertainty estimates or evidence of a
uniform habitat improvement.

The full report retains all 18 classes, per-seed confusion matrices, class
support and separate class-change tables for both primary contrasts. Littoral
Rock has no validation observations, so its recall is unavailable; Urban has
only three images from one plot. Pooled confusion counts repeat the same images
over four training seeds and are neither an ensemble nor four independent sets.

| Selected native model | Trainable parameters | Mean fit seconds | Mean peak allocated CUDA MiB |
|---|---:|---:|---:|
| CLS-only (reused) | 459,410 | 11.73 | 76.97 |
| Adapted-token bypass | 524,946 | 11.82 | 77.84 |
| Original-embedding bypass | 770,706 | 11.82 | 81.78 |

These measurements exclude encoder adaptation and image feature extraction.
They describe this machine and execution order, not a universal speed benchmark.
The report also records process-level CPU high-water marks; these are not
isolated per-fit CPU allocations. Increasing head width changes capacity even
though this workload's measured duration is similar.

## Verification and acceptance

The focused regression checks passed **203 tests**: the original 147 tests plus
56 extension tests. These cover exact shared initialization, bypass contents
and gradients, sample independence, all geo treatments, trainer isolation,
resume/replay, source/input/payload changes, 72-fit accounting, original
comparator reuse, both primary outcomes and report semantics.

GPU smoke verification fitted every new candidate/geo arm for three epochs on
64 training rows. Every saved training logit and top-three prediction replayed
exactly. Both standardized-geo variants also reproduced uninterrupted training
after an epoch-one interruption: model, optimizer, RNG, history, fitted geo
state and predictions all match exactly. Validation caches were opened only for
provenance authentication during this check; no validation rows were used for
optimization or scoring, and no test data was opened. Smoke runs do not enter
the 72-fit scientific budget.

A fresh GPU process reloaded **all 24 selected checkpoints**, replaying all
822 validation rows each. Logits and top-three predictions are bit-for-bit equal
to their saved predictions, with maximum absolute logit error 0.0. This verifies
cached-vector inference; online pixel inference remains outside this milestone.

A separate analysis process reproduced all **10 analysis payloads and their
manifest byte-for-byte**, including both bootstrap arrays. An independent
stdlib/NumPy verifier authenticated every new/original fit and selected
reference, reproduced all 25 learning-rate winners, and reconstructed every
10,000 × 37 × 4 bootstrap value directly from selected validation predictions
with maximum error 0.0. All **148 metric intervals (296 endpoints)** match the
saved draws exactly. Independently calculated paired means agree within
`1.12e-16`, the rounding difference from the alternative arithmetic order.

Visual review passed for **16 representative real-result PNGs**, covering both
primary results, all contrast categories, selected metrics, learning curves,
resources, support and count/normalized/pooled confusions. Integrity checks
authenticated all **201 report payloads**, decoded all **94 PNGs**, checked
their **94 PDF counterparts** and resolved **112 local report links**. PDF
verification covers hashes, headers and pairing; it is not independent raster
review. The generated report manifest remains unchanged; separate receipts
record review acceptance. There were no training, analysis or rendering failures
in the declared study workflow.

## Retained artifacts

The [full extension report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_comparison_v1/report/report.md)
contains all 25 selected arms, every learning-rate curve, all 37 contrasts,
per-seed and per-class findings, resource measurements and PNG/PDF figures.
The study root is
`multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/`.

| Evidence | Location |
|---|---|
| Frozen recipe, sources and 37 contrasts | [protocol manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/protocol/protocol_manifest.json), [contrasts](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/protocol/contrasts.json) |
| Original source/artifact identities | [parent evidence](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/protocol/parent_evidence.json) |
| Complete 72-fit execution ledger | [status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/orchestration/m3_bypass/status.json) |
| All 72 new fit metrics | [all_new_fits.csv](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_comparison_v1/tables/all_new_fits.csv) |
| All 37 paired contrasts | [paired_contrasts.csv](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_comparison_v1/tables/paired_contrasts.csv) |
| 24 new selected references and 76 unchanged original references | [selection.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_comparison_v1/selection.json) |
| Authenticated numerical payloads | [analysis manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_comparison_v1/manifest.json) |
| 203 regression tests and exact GPU smoke/resume | [implementation checks](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/implementation_checks_v1.json), [GPU verification](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/verification/gpu_smoke_v1/verification.json) |
| All 24 selected-checkpoint GPU replays | [replay receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/selected_checkpoint_replay_v1.json) |
| Independent analysis reconstruction and recalculation | [analysis replay](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/analysis_replay_v1.json), [verifier](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/verify_numerical_replay.py) |
| Report integrity and real-result visual review | [integrity receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_report_integrity.json), [visual review](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_cls_bypass_v1/reports/m3_bypass_visual_review.json) |

Every new `development/frozen_cached/.../` bundle retains `final.pt`, `resume.pt`,
resolved configuration, run specification, fitted preprocessing, optimizer/RNG
resume state, full epoch histories, train/validation logits and predictions,
metrics and manifest. All 72 candidates remain available, including unsuccessful
recipes. The 24 `selection/frozen_cached/.../selected.json` files identify chosen
bundles without deleting candidates or duplicating model weights.

The original M1 configuration, model/trainer source, 228 M3 bundles, selections
and numerical report remain unchanged. Completion of this extension does not
start M3J, all-development refitting or cleaned-test evaluation.

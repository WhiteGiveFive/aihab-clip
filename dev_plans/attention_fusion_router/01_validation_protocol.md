# R1 — Seed-1 attention-fusion geometry-router validation protocol

Recorded: **22 October 2026**. Status: **Protocol complete; R2 completed on
22 October 2026 and is recorded separately.** The governing scope is in the
[study README](README.md); executable settings belong to
[`configs/multimodal_attention_fusion_geometry_router.yaml`](../../configs/multimodal_attention_fusion_geometry_router.yaml).
This protocol freezes prospective R3 choices. R1/R2 authorization does not
include fitting thresholds, applying routing rules or reporting routed metrics.

Preparation completed successfully on 22 October 2026. The
[saved declaration](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/protocol/declaration.json)
and [resolved configuration](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/protocol/resolved_config.json)
record the actual input hashes, encoder paths/digests and epochs, extraction
settings, class order and four comparison IDs. The declaration explicitly
records `test_data_scored: false`, `thresholds_selected: false` and
`routing_applied: false`. Later stage receipts, not this protocol preparation,
establish successful extraction and scoring. The completed
[R2 receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/completion.json)
now records three complete embedding bundles and four complete score sets,
with all three routing/test flags still false. R3 selection and application
remain prospective.

## 1. Question and fixed experts

Determine whether image and geo neighborhood geometry can identify cases where
the completed M3J geo-only expert should replace the completed
`attention_cls_skip_original` expert. Hold the classifiers fixed while
comparing original-pretrained and after-training visual geometry.

Use **seed 1, pretrained initialization, partial-joint training only**. The
fusion run is:

```text
multimodal_artifacts/attention_fusion/cs/gse_100m/
  attention_fusion_joint_direct_v1/runs/seed1/joint_partial/
  attention_cls_skip_original/pretrained/
```

The geo run is:

```text
multimodal_artifacts/attention_fusion/cs/gse_100m/
  attention_fusion_joint_geo_only_direct_v1/runs/seed1/geo_only/
```

Resolve both from their completed `run.json` records and saved prediction
files, not from the historical geo-helpfulness study. Bind source hashes,
checkpoint policy/epoch, class order, row identity and recorded assurance.

| Checkpoint policy | Fusion | Geo | After-training geometry encoder |
| --- | --- | --- | --- |
| `final` | Epoch 50 | Epoch 50 | Fusion's saved epoch-50 visual tower |
| `best` | Previously selected epoch 36 | Previously selected epoch 11 | Fusion's saved epoch-36 visual tower |

Do not reselect either best epoch. Final and best remain separately reported
policies; their comparison changes both experts. The new geometry encoder does
not change either expert's saved probabilities or candidate classes.

The existing top-1 diagnostic provides input sanity expectations, **not routing
results**: final fusion/geo have 585/518 correct queries, 77 possible rescues
and 144 possible harms; best fusion/geo have 590/533 correct queries, 84
possible rescues and 141 possible harms, all out of 822. The corresponding
top-1 oracle counts are 662 and 674. These values already appear in the
[rescue diagnostic](../attention_fusion/03e_seed1_geo_rescue_opportunity.md).

## 2. Cohorts and leakage controls

The audited development cohort contains **4,200 images / 1,625 plots**, split
into **3,378 training images / 1,300 plots** and **822 validation images / 325
plots**. The latter are queries and the former are the only labeled reference
bank. Check unique row IDs, complete row coverage, consistent plot/label
mapping, fixed 18-class order, and disjoint training/validation row and plot
identities before scoring.

The prepared native-geo inputs come from the original development
`train.parquet` and `val.parquet` tables, joined by file identity to the
completed M3J allocation. Those source tables contain historical roles of
3,618/582 images; their filenames do not define this study's 3,378/822 roles.
Read the two development sources as one universe, then apply the authoritative
M3J assignments. Preserve the source feature order and reject missing or
duplicate identities.

Reference class labels are training labels. Query labels are retained for later
evaluation/selection but are not inputs to distances, support, candidate
classes or geometry scores. Candidate classes are the top-1 classes of the
fixed fusion and geo probability vectors. R2 score computation must be
unchanged when query truth labels are withheld or altered.

No validation image or plot may enter the reference bank. The historical
4,200-reference bank was a later test setting and is inappropriate here. Do not
read cleaned-test query tables or images in R1/R2. No seed pooling or synthetic
seed duplication is allowed: all denominators are the actual 822 queries.

## 3. Geometry representations and extraction

The two geometry arms are:

1. **Original pretrained:** the authenticated original externally pretrained
   visual tower used to initialize this fusion fit; shared between final and
   best policies.
2. **After training:** the fixed fusion fit's own saved visual tower at the
   selected policy's epoch, final 50 or best 36.

There are three unique encoders. Extract all 4,200 development images once per
encoder, preserving exact row/split identity and canonical order. Each bank and
its queries must use the **same encoder, preprocessing and numeric recipe**.
Do not combine a reference cache from one encoder with queries from another,
substitute old pretrained embeddings with incomplete lineage, or silently use
the historical CS-adapted geometry encoder.

Preserve the parent's deterministic historical image-evaluation path:

- Read the historical BGR pixel representation and force the 439-pixel square
  resize before the native evaluation transform; retain the producer's exact
  interpolation and transform ordering.
- Use the pinned evaluation transform with 384-pixel model input and per-channel
  mean/std `(0.5, 0.5, 0.5)`; no random crop, augmentation or channel conversion
  introduced by this study.
- Use FP32 encoder weights and inputs, CUDA batch size 16, cuDNN TF32 enabled,
  matrix-multiplication TF32 disabled and autocast disabled. Record the runtime
  and resolved transform. This extraction recipe is distinct from FP16-autocast
  M3J training.
- Use the 1,152-dimensional global visual embedding and row-wise L2
  normalization. Check finite vectors, correct dimensions and valid nonzero
  norms before publishing an embedding bundle.

Geo geometry uses the supplied **native 64-dimensional GSE vectors**, with
row-wise L2 normalization. Reject non-finite/zero-norm vectors with explicit
input evidence; do not fabricate geometry. The classifiers continue to use
their already-fitted training-coordinate standardization. Native-geo L2
geometry is a separate operation and must never consume classifier-standardized
vectors or refit that scaler.

For reference, the historical extraction choices are documented in
[A002](../attention_fusion/01b_historical_image_extraction.md). Preserve its
numeric recipe while using the current D038/D039 practical assurance contract;
the old exact-replay acceptance gate is not reinstated.

The prepared image inventory records identity, resolved path, size and
modification time for all development photographs. It does not establish
historical raw-image byte equality or introduce a new raw-byte replay gate.
Authenticate saved encoder files against their recorded expected digests with
the bounded D039 read mechanism. Record implementation-module hashes by stage
so later numerical outputs identify the implementation that produced them.

## 4. Strict five-plot geometry

Let `c_f` and `c_g` be the fixed experts' predicted classes. For modality `m`,
L2-normalize query `q` and reference image `r` separately and compute cosine
distance `1 - q·r` in that modality's original dimension. Clamp numerical
round-off to the valid cosine-distance range. Do not concatenate image and geo
vectors or use projected visualization coordinates.

A plot contributes its closest reference photo to a candidate class. For each
class, sort those plot-minimum distances deterministically, retain the nearest
**five distinct plots**, and average their distances. Multiple close photos
from the same plot cannot fill several neighborhood positions. All photos in a
reference plot must have the same class label. Tie ordering must be stable and
recordable; tied distances do not change the average.

Denote the resulting class distance by `d_m(c)`. Define:

\[
A_m(q)=\log\frac{d_m(c_f)+10^{-8}}{d_m(c_g)+10^{-8}},
\qquad
S(q)=\min\bigl(A_{\mathrm{image}}(q),A_{\mathrm{geo}}(q)\bigr).
\]

Positive scores support geo's proposed class in **both** spaces. A positive
image score does not mean the image expert is favored. Distances and ratios
must be available only where both candidate classes have at least five
distinct valid reference plots. Use **strict K=5**, never a smaller effective
K, a one-plot fallback, a class-dependent K or a K sweep.

Persist candidate-class reference support, modality distances, both log ratios,
their minimum and explicit eligibility/ineligibility reasons for each query.
Agreement is not an intervention opportunity. Missing class support or invalid
geometry makes routing unavailable and later retains fusion. R2's score tables
must cover every query, including agreements and unsupported rows, so later
metrics cannot silently drop difficult cases.

## 5. Frozen R3 routing matrix and threshold recipe

R3 will evaluate the following four cells without choosing a winning cell in
advance or discarding unfavorable cells:

| Cell | Expert pair | Geometry image encoder |
| --- | --- | --- |
| Final / original pretrained | Fusion 50 + geo 50 | Original pretrained |
| Final / after training | Fusion 50 + geo 50 | Fusion visual 50 |
| Best / original pretrained | Fusion 36 + geo 11 | Original pretrained |
| Best / after training | Fusion 36 + geo 11 | Fusion visual 36 |

Each cell receives **two rules**, totaling eight evaluations:

1. Fixed rule: switch on eligible disagreements with `S > 0`.
2. Selected rule: select one nonnegative threshold from that cell's validation
   scores using the recipe below, then switch on eligible disagreements with
   `S > threshold`.

Use strict `>` throughout; equality, agreement and insufficient support retain
fusion. Threshold selection is independent in each cell. It does not pool
representations, checkpoint policies or fabricated seeds. The fixed rule is
reported even if it fails the selected-rule constraints.

Preserve the historical enumerator: a typed never-intervene candidate, zero,
and deduplicated float64 midpoints between adjacent distinct positive scores
on eligible disagreements. Store a finite threshold in canonical hexadecimal
float form alongside its decimal display and `strict_gt` comparator. The
never-intervene policy is typed explicitly, not represented by a large finite
sentinel or infinity.

A finite candidate is feasible only if all of these hold:

- At least **20 distinct validation plots** contain an acted-on query.
- Intervention coverage is at least **0.01 of all 822 queries**, including
  agreements and unsupported queries in the denominator.
- Rescues `R` and harms `H` satisfy **`H <= 0.7 R`**; implement the exact integer
  comparison `10H <= 7R`.
- Net rescue is strictly positive: **`R - H > 0`**.
- The threshold is finite and **nonnegative**.

Maximize `R - H`, then break ties by **fewer harms, fewer actions, higher
threshold**, in that order. If no finite candidate is feasible, save the
**never-intervene** result and its reason. Do not relax guards, insert a
different threshold or remove that cell from reporting. Save every candidate's
counts/feasibility and the selected policy. These selections are fitted on
reused validation, not held-out routing results.

The historical code requires four seeds in its own data adapter. Reuse or
factor its mathematical conventions while implementing an honest seed-1
adapter; never replicate seed-1 rows four times to satisfy that interface.

## 6. Prospective metrics and uncertainty

R3/R4 will retain fusion and geo baselines beside every routed result and report
**Acc@1, Acc@3, macro F1, weighted F1, MCC, raw and row-normalized confusion
matrices**, using the fixed 18-class order. Report class support and the
absent-class convention explicitly; a zero-support class has no estimable
class accuracy. Preserve the parent metric implementation where compatible.

Routing selects the **entire probability vector** from one expert: geo on
acted-on queries, fusion otherwise. Calculate Acc@3 from that selected vector,
not from a mixed ranking or a top-1 replacement into fusion's top-3 list. Fixed
source probability vectors and class order are preserved; no probability
blending or additional calibration is part of this study.

For each cell and rule report image counts and distinct plot counts where
meaningful: actual rescues, harms, `R-H`, both-wrong interventions, potential
rescues left uncorrected, actions, all-query coverage, available rescue/harm
pools, candidate support and abstention reasons. Separate both-wrong
disagreements from both-wrong agreements. Check the identities:

\[
N_{\mathrm{actions}}=R+H+N_{\mathrm{both\ wrong\ actions}},
\qquad
\Delta\operatorname{Acc@1}=(R-H)/822.
\]

Show habitat-level changes and examples without treating the unknown true
habitat as a deployable routing input. The oracle may be reported only as the
**top-1** upper bound `Acc_fusion + potential_rescues / 822`; the earlier
diagnostic did not define an oracle for Acc@3, F1 or MCC.

This protocol makes no independent-confirmation or population-gain claim.
Counts across photographs do not imply independent trials. If R4 adds
conditional paired uncertainty, resample whole plots and state that intervals
condition on these fitted experts and, for selected rules, their fitted
thresholds; they cannot remove validation reuse or selection uncertainty.

## 7. Preservation and milestone boundary

Save a protocol/source snapshot before scores, including resolved configuration,
exact source identities and this specification. Record all saved-file
authentication attempts under the bounded D039 mechanism and retain original
failure evidence. Carry `provisional_unresolved_memory_integrity` through input,
embedding, score and verification manifests.

R2 completes when validated input/embedding/score artifacts and focused
verification pass for all four cells. It must not save selected thresholds,
intervention flags, routed probability vectors, routing metrics or a claim
that geometry succeeds. R3/R4 follow separately; after R4 return to main M3J
and then M4/M5. R5 is deferred cleaned-test exploration after the main study
completes. R2 met this boundary on 22 October 2026: 12,600 visual vectors,
four score tables covering 822 queries each, 90 passing focused CPU checks and
completed artifact verification. See the
[execution record](02_implementation_and_verification.md) for receipts and
support counts; these are preparation results, not routed-performance results.

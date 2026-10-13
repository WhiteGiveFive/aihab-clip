# Attention-Based Image–Geo Fusion

## Document purpose

This is the living method specification, development roadmap, evaluation
contract, and progress record for attention-based fusion in the CS habitat
classification pipeline. The short name is **attention fusion** and the method
namespace is `attention_fusion`.

It follows the structure of the
[geo-helpfulness development guide](../reliability_aware_geo_fusion/README.md),
while defining a separate supervised classification method. It distinguishes
established facts, proposed designs, implemented components, and measured
results. Update the status, record index, and decision log whenever a milestone
or scientific choice changes. Detailed commands, results, and implementation
notes belong in the linked milestone records.

| Field | Value |
|---|---|
| Primary task | CS native L3 habitat classification; fixed 18-class mapping |
| Inputs | Cached 1,152D image embeddings for initial runs; image pixels for online/joint training; fixed 64D 100 m GSE embeddings |
| Training regimes | Frozen encoder first; partial joint fine-tuning in M3J; full encoder fine-tuning deferred |
| Baseline anchor | `raw_concat`, refitted under the same experimental protocol |
| Essential attention control | Separate image/geo projections followed by concatenation |
| Initial attention model | Two modality tokens; one self-attention block; concatenate updated tokens |
| Audited development allocation | 4,200 images / 1,625 plots; 3,378 training and 822 validation images |
| Existing exploratory evaluation set | Expert-cleaned CS test: 1,347 images / 531 plots |
| Independent confirmation set | Not yet identified |
| Current status | M0–M2 complete; M3–M5 including M3J planned; M6 confirmation deferred |
| Last updated | 2026-10-12 |

[M1](01_experimental_protocol.md) freezes the initial study configuration and
records the input audit. [M2](02_models_and_training.md) implements all seven
candidate models, prepares all four development caches, and verifies training,
checkpoint resume and bundle replay. Its small training-only smoke runs are
implementation checks. The 228-fit comparative study has not started.

## Contents

- [Method summary and expected outcome](#method-summary-and-expected-outcome)
- [Evidence and relationship to existing work](#evidence-and-relationship-to-existing-work)
- [Core design contract](#core-design-contract)
- [Scope and comparison set](#scope-and-comparison-set)
- [Evaluation and leakage-control contract](#evaluation-and-leakage-control-contract)
- [Implementation roadmap](#implementation-roadmap)
- [Implementation principles](#implementation-principles)
- [Proposed code and artifact layout](#proposed-code-and-artifact-layout)
- [Risks and interpretation](#risks-and-interpretation)
- [Progress ledger](#progress-ledger)
- [Detailed development records](#detailed-development-records)
- [Decision log](#decision-log)
- [Immediate next action](#immediate-next-action)
- [Change log](#change-log)

## Method summary and expected outcome

Learn separate adapters for image and geo embeddings, represent the modalities
as a short token sequence, and use self-attention to exchange information before
predicting habitat. The image and geo encoders remain frozen during fusion-head
training in the initial regime; the adapters, attention block, and classifier
are trainable. A separate **joint fine-tuning** regime also updates selected image
encoder parameters using the fused classification loss. The supplied GSE
embeddings stay fixed while their adapter learns in both regimes.

“Frozen” describes the encoder during fusion training, not its entire history:
it may already have undergone habitat adaptation on the permitted training plots.
Partial joint fine-tuning is the first extension; full image-encoder fine-tuning
is a later, separately declared capacity experiment.

The research question is: **does adaptive interaction between modality tokens
improve classification beyond matched projection, normalization, and classifier
capacity?** A gain over historical raw concatenation alone does not answer this
question if the encoders, preprocessing, or training budget also change.

Joint training adds a second question: **does allowing the visual representation
to adapt to the fusion objective improve the complete classifier?** Explore it
even if frozen attention does not beat frozen concatenation; the hypotheses are
different. Separate gains from image adaptation, adding geo, and attention.

Successful development produces a reproducible comparison, including a negative
or inconclusive result if attention does not help. Promotion of an attention
model requires the development criteria declared in M1 and a comparison against
both matched raw and projected concatenation. Strong claims about generalization
require independent data; neither reused validation nor the existing cleaned
test provides that confirmation.

## Evidence and relationship to existing work

- The existing pipeline supports image-only, geo-only, and concatenation models.
  The established raw head has input width 1,216 and a 256-unit hidden layer.
- Earlier agreement and routing studies motivate testing better use of geo.
  Their results are evidence about those systems, not measured attention gains.
  See the [router research summary](../reliability_aware_geo_fusion/25_router_research_summary.md).
- [Research and design background](00_research_and_design_background.md) records
  the relevant literature, the singleton cross-attention limitation, and the
  distinction between token mixing and geo-reliability estimation.
- This method does not depend on the router's M4 feasibility gate. It does not
  initially require correctness-state targets, ConfidNet, expert calibration,
  or out-of-fold router supervision. Reuse split assignments and encoder caches
  only after checking their suitability for this experiment.

## Core design contract

### Inputs and preprocessing

The fusion module receives one global image vector and one global geo vector.
The image vector is either cached or computed online from pixels. Preserve
sample identity, plot identity, feature order, and class order throughout.
Different images from the same plot may share geo features; they must remain in
the same data partition.

Keep these operations distinct:

| Operation | Meaning and proposed handling |
|---|---|
| Image L2 normalization | Verified in M1; preserve the producer's legacy BGR/439 pixel path and apply L2 normalization differentiably to later online encoder outputs |
| Native geo | Use the supplied 64D vector without additional normalization |
| Geo L2 normalization | Normalize each vector independently; document zero/non-finite handling |
| Geo coordinate standardization | Fit per-coordinate mean/std on training rows only; save feature order and state |
| Token LayerNorm | Learned-model operation within each sample; separate from dataset standardization |

M1 fixes **native geo as default**, with L2-normalized and training-fitted
coordinate-standardized geo as secondary comparison arms. Native and L2 geo
remain distinct: L2 can remove magnitude information. Initial image producers
are the four training-partition habitat-adapted encoders verified in M1.
Compare architectures under identical preprocessing within each arm.
Do not attribute a normalization gain to attention, silently normalize twice,
or feed a historical checkpoint differently from its training convention.

### Primary architecture

For image vector \(x_i\) and preprocessed geo vector \(x_g\), use separate
trainable adapters, fixed in M1 as `Linear -> GELU -> LayerNorm`:

\[
z_i=P_i(x_i)\in\mathbb{R}^{d},\qquad
z_g=P_g(x_g)\in\mathbb{R}^{d}.
\]

Add learned modality identities \(e_i,e_g\), stack along the token axis, and
apply a Transformer encoder block:

\[
T_0=\operatorname{stack}(z_i+e_i,z_g+e_g),\qquad
T_1=\operatorname{Transformer}(T_0).
\]

The tensor shape is `[batch, 2, d]`; embedding coordinates are not token positions.
Concatenate both updated tokens and classify:

\[
\ell=H([T_{1,i};T_{1,g}])\in\mathbb{R}^{18}.
\]

Frozen initial design: `d=128`, one pre-LayerNorm block, four heads, feed-forward
width 256, dropout 0.1, and a classifier hidden width of 256. All seven candidates
use `Linear -> ReLU -> Dropout -> Linear` classifiers under user-confirmed
[amendment A001](01a_relu_classifier_amendment.md). Adapters and fusion FFNs retain
GELU. Begin with ordinary classification cross-entropy. M1 records exact module
ordering, regularization,
optimizer, budget, and checkpoint selection. These values are not established
optima. The projected-concatenation control uses the same adapter outputs,
modality offsets, token order, and classifier design, bypassing the Transformer.

### Frozen and joint training regimes

Keep the tokenization and fusion design fixed when first studying encoder
unfreezing. Online image inputs produce
\(x_i=\operatorname{Norm}(E_\theta(I))\), and the classification loss propagates
through the image adapter into the declared trainable parts of \(E_\theta\).
Geo influences those training gradients through fusion; it does not become an
input to the image encoder. The text tower stays frozen and is not part of the
fusion training objective.

| Regime | Image path | Trainable components | Role |
|---|---|---|---|
| Frozen cached | Saved global vectors | Adapters, fusion, classifier | Initial efficient development comparison |
| Frozen online | Pixels through a fixed encoder | Adapters, fusion, classifier | Match image exposure/augmentation for the joint comparison |
| Partial joint | Pixels through a partly trainable encoder | Selected vision parameters plus adapters, fusion, classifier | First joint fine-tuning candidate |
| Full joint | Pixels through a fully trainable image encoder | Entire vision encoder plus adapters, fusion, classifier | Deferred extension |

Cached final embeddings are detached from their producing encoder and cannot
support joint backpropagation. Validation features must be recomputed with each
evaluated encoder checkpoint; do not reuse features from an earlier encoder.
Any later cache must be tied to its exact checkpoint and preprocessing.

The initial joint recipe should unlock a small declared set of final vision
layers/groups. Record exact parameter names and counts; framework groups need
not correspond one-to-one to Transformer blocks. Use separate encoder/head
optimizer groups, provisionally with a smaller encoder learning rate. A short
head warm-up before unfreezing is optional. Fix the unfreezing schedule, rates,
augmentation, effective batch size, precision, stopping rule, and compute budget
before the joint candidate fits. No such numerical recipe is frozen yet.

### Alternatives and exclusions

- A two-layer `[CLS, image, geo]` model with updated-CLS readout is required in
  the initial study. Its second layer can read contextualized modality
  tokens. It is a separate candidate with its own parameter count. Comparing it
  with one-layer updated-token concatenation changes both depth and readout;
  isolating a readout effect would require a same-depth comparison.
- The BP-inspired readout `[updated CLS; original adapted image; original
  adapted geo]` is an optional ablation. Concatenation is a skip path, not
  elementwise residual addition. Record precisely whether type embeddings are
  included in the bypassed tokens.
- A one-layer CLS readout may be a diagnostic ablation; its initial CLS query
  is shared across samples and cannot read the same layer's updated modality
  tokens. Do not describe it as equivalent to updated-token concatenation.
- Exclude pure bidirectional one-key cross-attention as a test of adaptive
  attention: its softmax is always one. Residual/MLP paths can still combine
  modalities, but the attention branch cannot select source information.
- The 64 geo coordinates are latent dimensions, not 64 semantic metadata fields.
  Arbitrary vector slices are not image patches or meaningful geographic units.
- Learned multi-token projection, patch-level image inputs, bottleneck tokens,
  auxiliary alignment losses, and reliability routing are later extensions.

## Scope and comparison set

Scope covers supervised image + 100 m GSE classification with frozen and jointly
adapted image representations. The first comparison uses a frozen encoder;
M3J adds partial joint fine-tuning. Patch extraction, changes to the source GSE
representation, missing-modality imputation, and test-time adaptation remain
outside these initial studies. Unfreezing does not resolve singleton-key
cross-attention; retain the same multiple-key self-attention design.

| Candidate | Role | Initial priority |
|---|---|---|
| Image-only | Establish image information and image-relative rescue/harm | Required |
| Geo-only | Establish geo information and complementarity | Required |
| Raw concatenation | Matched baseline using original embedding widths | Required |
| Two-adapter projected concatenation | Isolate projection/normalization from the Transformer | Required |
| Two-token self-attention, updated-token readout | Primary attention model | Required |
| Two-layer CLS model | Compare pooled readout and deeper interaction | Required initial alternative |
| Capacity control without attention | Two token-wise pre-LN residual FFNs; 128 fewer parameters than primary attention | Required |
| BP-style bypass or feature conditioning | Study readout preservation or a simpler interaction mechanism | Optional, bounded extension |

Do not run an unrestricted product of widths, depths, readouts, and normalization
choices. M1 fixes 19 candidate/preprocessing arms, three learning rates and four
seeds: **228 development fits**, each 50 epochs. Shared
adapter width and classifier design do not imply equal parameter counts: report
the difference and use the capacity control before attributing gains specifically
to attention weights.

M3J uses a bounded core comparison across encoder regimes:

| Model | Frozen online | Partial joint |
|---|---|---|
| Image-only classifier | Required | Required |
| Raw concatenation | Required baseline anchor | Required baseline anchor |
| Two-adapter projected concatenation | Required | Required |
| Primary two-token attention | Required | Required |

Geo-only remains the same fixed-input reference; it has no image encoder to
unfreeze. Do not repeat every frozen-study architecture ablation in the joint
study. Reuse earlier results only if their full exposure and training recipe
matches. Otherwise fit the frozen online controls with the same augmentations
and image views as partial joint training. An equivalent declared cached-view
set is acceptable, but a deterministic single-view cache alone cannot isolate
unfreezing from a simultaneous augmentation change. Report cached-versus-online
differences separately.

## Evaluation and leakage-control contract

1. **Reuse groups, audit producers.** Start from the existing plot-grouped
   3,378/822 development allocation and seeds 1–4, verified by M1 provenance
   checks. Habitat-adapted image encoders used for validation features must have
   excluded validation plots during adaptation. A head-only split cannot repair
   leakage from a globally adapted encoder. The initial study uses these adapted
   encoders only; externally pretrained encoders would be a separate extension.
   Record shared 100 m cells and
   geographical overlap where identifiers permit: plot separation alone does
   not establish generalization to unseen cells or regions.
2. **Keep development independent of test loading.** Training and selection
   consume development data only. A label-blind test identity manifest may be
   used to check overlap. Do not load test features, labels, or diagnostics in
   the development runner or emit per-epoch test metrics.
3. **Fit transforms locally.** Fit adapters, heads, selected image-encoder
   parameters, learned preprocessing and any declared calibration using the
   allowed training partition. Validation
   may select declared candidates/checkpoints, but cannot supply gradients or
   preprocessing statistics for those selection runs.
4. **Pair comparisons.** Use the same row universe, class mapping, geo source,
   preprocessing arm, split and training-seed set across methods. Frozen-head
   comparisons share the encoder producer. Joint comparisons share the starting
   checkpoint, declared trainable vision layers and optimizer/selection budgets;
   their final encoder weights may differ by method. Match image augmentations,
   view exposure and effective batch size in the frozen-online/joint contrast.
   Historical results with incompatible recipes are context only.
5. **Freeze before exploratory test application.** Record the selected recipe,
   comparator set, preprocessing, checkpoint rule, and final-refit procedure.
   The frozen study retains selected train-only heads and separately refits heads
   on all development data, keeping the same train-partition-adapted encoder.
   Use this predeclared procedure for all comparators; never select final epochs
   from test scores. For joint models,
   refit the declared encoder parameters and fusion components from the declared
   initialization, then save them together with preprocessing. A separate
   apply-only path loads the frozen bundles and records the evaluation event.
6. **State the evidence level.** Existing validation has already supported model
   development; selected validation scores and intervals are conditional on
   that process. The cleaned test has repeatedly informed research and remains
   exploratory even with a frozen new model. Report every additional use.

M1 locks the primary metric, checkpoint rule, tie-breaks, tuning budget, effect
interpretation and uncertainty analysis before candidate scoring. Select one
learning rate per candidate/preprocessing arm by mean validation **Acc@1** across
seeds, then weighted F1 and MCC; use epoch 50, without early stopping. Report
**Acc@1, Acc@3, weighted F1, MCC and confusion matrices**, class support, model
size and runtime. M1 defines supported improvement, inconclusive and no-improvement
categories using seed consistency and paired plot intervals. No application-specific
minimum useful effect has been established; report effect size without claiming
that a supported positive direction alone establishes practical importance.
M3J records its bounded extension recipe before joint fitting/scoring, retaining
the same evidence boundaries and separately documenting any development-driven
amendments. Report frozen-to-joint gains within each model and attention versus
concatenation within each regime; include trainable encoder counts and peak
memory alongside runtime.

Report rescue/harm against **both image-only and matched raw concatenation**:

\[
\Delta\mathrm{Acc}=(N_{\mathrm{rescued}}-N_{\mathrm{harmed}})/N.
\]

Also report the paired difference against projected concatenation. Save
per-instance predictions so all comparisons use the same rows. Use paired
whole-plot bootstrap intervals and describe seed variation separately; repeated
seed predictions of one image are not independent observations. Low-support
classes and inconclusive intervals must remain visible. Attention weights are
descriptive diagnostics, not calibrated probabilities of geo helpfulness.

## Implementation roadmap

Status values: **Complete**, **In progress**, **Planned**, **Blocked**, **Deferred**.
A milestone can complete with a negative scientific result. Record that outcome
separately from implementation status; do not expand the search silently.

### M0 — Establish the governing design

**Status:** Complete, documentation only.

Deliverables: this guide, the research/design background, initial decision log,
and an explicit separation of proposals from existing evidence. Acceptance:
the next milestone and its unresolved choices are identifiable without reading
the chat. No model implementation or training is included in M0.

### M1 — Freeze the protocol and audit inputs

**Status:** Complete on 2026-10-12. Depends on M0.

Audit split assignments, label mapping, original geo values, image cache
normalization, producer fitting plots and reusable artifacts. Choose the small
preprocessing/candidate matrix, shared training and tuning budgets, metrics,
capacity control, and final-refit recipe. Specify the development-only execution
boundary and the criteria for interpreting a gain, no gain, or an inconclusive
comparison. Audit raw image availability and suitable starting checkpoints for
M3J; record its comparison scope and compute limits. Freeze its exact online
training recipe before that milestone's candidate fits. Identify any necessary
cache regeneration before scheduling it.

Delivered: [01 Experimental protocol](01_experimental_protocol.md), authenticated
input inventory, expanded 228-fit matrix and resolved study
configuration. Acceptance: no unexamined encoder leakage, unresolved primary
comparison, or implicit use of test data; every initial candidate has a defined
input, fitting partition, budget, selection rule, and artifact destination.
All four encoder checkpoint/provenance and validation-cache checks pass. Raw
images and native geo are available. M2 must prepare compatible development-train
image caches; mixed-test and fold-specific OOF caches cannot be reused for them.

### M2 — Implement controls and the primary attention model

**Status:** Complete on 2026-10-12. Depends on M1. See the
[implementation and verification record](02_models_and_training.md).

Delivered seven candidates and three geo treatments, four compatible
development-cache pairs, a dedicated development-only runner, immutable
checkpoint/bundle interfaces and recorded provenance. Verification: 97 focused
tests plus 43 legacy tests; all 19 arms passed the training-only GPU smoke;
checked encoder vectors and saved-head predictions replayed exactly; interrupted
GPU optimization matched uninterrupted training. A002 explicitly preserves
historical encoder extraction arithmetic while head training disables TF32.

Implement explicit preprocessing, a true development-only path, the matched
two-adapter concatenation control, and two-token self-attention. Reuse baseline
components where their behavior matches the protocol. Also implement the
capacity control and any secondary architectures declared in M1. Add the
required artifact and checkpoint interface before comparative runs.
Reuse compatible image-only, geo-only and raw-concat components; these are
existing architectures to integrate and verify, not requirements to rewrite
their model code. Reusing historical trained heads/results additionally requires
matching the full input, initialization, fitting and selection recipe.

Deliverable: `02_models_and_training.md`, model/configuration integration, and
meaningful verification. Acceptance: correct shapes and token axis; finite
forward/backward behavior; sample-independent inference in evaluation mode;
both inputs influence the appropriate outputs; saved preprocessing/checkpoints
replay predictions; test-data access is excluded from the development path.
Use a small training-only smoke run to check optimization, not to claim benefit.

### M3 — Run the declared development comparison

**Status:** Planned. Depends on M2.

Run required controls and the primary attention model across the declared
seeds, then any secondary arms already specified in M1. Save all candidates,
including unsuccessful ones. Separate preprocessing, capacity, and readout
effects; select using the declared rule without consulting cleaned-test results.

Deliverable: `03_development_comparison.md`, paired predictions, learning curves,
metrics, uncertainty and a frozen selection record. Acceptance: matched lineage
and budgets, complete candidate accounting, and an explicit supported-gain,
no-gain, or inconclusive interpretation. Any new search becomes a separately
recorded study; a negative result is a valid stopping point for that recipe.
It does not prevent the separately planned M3J joint-training study.

### M3J — Jointly fine-tune the image encoder and fusion model

**Status:** Planned. Depends on M2 and a recorded M3 frozen comparison; no
positive frozen-attention result is required.

First declare the bounded joint recipe and matched frozen-online controls.
Implement a pixels + geo training path and partial vision unfreezing, keeping
the global-vector fusion architecture unchanged. Train the core comparison
matrix under the same starting-checkpoint and data-exposure contract, across
the declared seeds. Measure gains attributable to joint image adaptation and
whether attention improves on jointly trained concatenation. Full vision
unfreezing remains deferred, not an automatic search expansion.

Deliverable: `03_joint_finetuning.md`, the online runner integration, joint
checkpoints, matched controls, resource measurements and development report.
Acceptance:

- pixels, geo, labels and sample/plot identities are aligned; development remains
  test-free and initial encoder provenance excludes validation plots;
- the fused loss supplies gradients to the selected vision parameters and
  fusion components; only declared parameters are updated, frozen parameters
  and running statistics stay unchanged, and module train/eval behavior is
  specified and verified;
- validation uses the current encoder in evaluation mode without gradients;
  complete encoder + fusion + preprocessing checkpoints replay predictions;
- controls match initialization, augmentation, effective batch size, training
  exposure and declared tuning budgets; encoder/head rates and warm-up are
  recorded, including any justified differences between regimes;
- results distinguish encoder adaptation, geo benefit and fusion architecture;
  report all declared candidates and the same no-gain/inconclusive outcomes;
- record the selected joint recipe and refit procedure before its M4 application.

### M4 — Apply the selected models to exploratory test

**Status:** Planned. Depends on completed development records and a frozen
evaluation recipe for each included regime: M3 for frozen, M3J for joint.

For frozen models, retain the selected development-train checkpoints and perform
the declared all-development fusion/head refits with the same frozen encoders.
Apply both routes and the frozen model/comparator set to
the cleaned test through the separate evaluator. Preserve preprocessing and
class order, record the test-use event, and compare rescues, harms and per-class
changes. This is not a chance to choose a new architecture or checkpoint.
Prefer one declared application covering both regimes after M3J. If a frozen-only
application occurs earlier, its findings must not guide joint choices; record
any later joint application as additional exploratory test reuse.

Deliverable: `04_exploratory_evaluation.md` and the frozen inference bundles.
Acceptance: no fitting or selection from test data, exact row alignment,
reconstructible metrics and explicit historical-test-reuse limitations. If the
development outcome makes this stage unnecessary, record it as deferred.

### M5 — Integrate and document reproduction

**Status:** Planned. Depends on the relevant M3/M3J development records; include
M4 outputs if M4 proceeds.

Publish the supported configuration, commands, input assumptions, reproducible
reports, and limitations. Keep the current default baseline unless evidence
justifies changing it. Preserve negative findings and provide a reproducible
baseline even if attention is not promoted.

Deliverable: `05_integration_and_reproduction.md` and focused pipeline docs.
Acceptance: a clean invocation can reconstruct the declared outputs from saved
inputs/configuration, and legacy modes retain their previous behavior.

### M6 — Independent confirmation

**Status:** Deferred; independent data not yet identified.

Define a genuinely independent temporal/geographical dataset or holdout and
freeze its evaluation protocol before access. Apply the chosen model and matched
controls without tuning on that holdout. Deliverable:
`06_independent_confirmation.md`. Repeated validation splits can strengthen
development evidence but do not make the current cleaned test independent.

## Implementation principles

- Keep model, preprocessing, fitting, evaluation, and reporting responsibilities
  distinct. Reuse shared components without importing the router's target and
  threshold machinery into a plain supervised fusion model.
- Preserve existing mode behavior and historical checkpoints; introduce explicit
  new options rather than changing default standardization or model semantics.
- Verify token axes and modality order explicitly. Do not let attention operate
  across batch samples. Support ordinary CPU/GPU placement without hardcoded CUDA.
- Keep configurations explicit and persist their resolved values. Record input
  identities, encoder provenance, transform state, class mapping, code revision,
  seeds, model size, histories, logits, predictions, and checkpoint selection.
- Separate cached-feature and online-image input paths explicitly. Record the
  encoder training regime, initial/final checkpoint identities, trainable
  parameter names, optimizer groups, augmentation, precision and accumulation.
  Do not detach online features before fusion or reuse stale validation caches.
  Save encoder, adapters, fusion and classifier as one reproducible inference
  bundle; include optimizer/scheduler state when supporting training resume.
- Version scientific changes and retain completed runs. Reuse a saved result
  only after validating its inputs/configuration; do not overwrite evidence to
  make a new recipe appear to reproduce an old one.
- Test meaningful model, split, preprocessing, isolation and replay behavior;
  avoid inheriting elaborate router-specific sealing workflows unnecessarily.

Current integration facts to resolve in M1/M2:

| Existing component | Implication |
|---|---|
| [Models](../../multimodal/models.py) and [trainer](../../multimodal/trainer.py) | Existing `tabular_projected_concat` projects geo only; it is not the proposed two-adapter control |
| [Trainer](../../multimodal/trainer.py) | Both training regimes currently standardize tabular inputs; native/L2 geo needs an explicit new preprocessing path |
| [Trainer](../../multimodal/trainer.py) and [data loader](../../multimodal/data.py) | `report_test_each_epoch=False` still loads test and performs final test evaluation; it is not sufficient development isolation |
| [Geo-helpfulness producer](../../multimodal/geo_helpfulness_oof.py) | Uses its own head-fitting path; changing the generic model factory does not automatically update this producer |
| [Feature export](../../multimodal/artifacts.py) | Saved final embeddings are detached; joint training needs pixels and a differentiable online encoder path |
| [Image fine-tuning](../../methods/PEFT_openclip.py) | Existing online adaptation uses prompt supervision; it is a reference for image loading/unfreezing, not an implemented joint image–geo trainer |

Pass the audited grouped assignments explicitly; generic split defaults must
not replace the intended development allocation. Reuse existing project test-use
tracking where applicable instead of constructing a parallel router-style registry.

## Proposed code and artifact layout

The governing records, standalone configuration, M1 audit/preparation tools and
M2 model/input/training implementation exist. Later milestone records remain planned.

```text
dev_plans/attention_fusion/
  README.md                         # governing document
  00_research_and_design_background.md
  01_experimental_protocol.md        # completed M1
  02_models_and_training.md          # M2 implementation and verification
  03_joint_finetuning.md              # create when M3J starts
  ...                               # later milestone records
configs/multimodal_attention_fusion.yaml
tools/audit_attention_fusion_inputs.py
tools/prepare_attention_fusion_protocol.py
tests/test_attention_fusion_input_audit.py
tests/test_attention_fusion_protocol.py
multimodal/attention_fusion.py       # seven candidates and geo preprocessing
multimodal/attention_fusion_inputs.py
multimodal/attention_fusion_training.py
tools/prepare_attention_fusion_inputs.py
tools/run_multimodal_attention_fusion.py
tests/test_multimodal_attention_fusion.py
multimodal_artifacts/attention_fusion/cs/gse_100m/<study_id>/
  protocol/                         # split/cache provenance and resolved recipe
  inputs/seed<N>/                    # M2 development-only cache preparation
  verification/                     # M2 train-only smoke checks; excluded from M3
  development/<regime>/<candidate>/<geo>/lr_<rate>/seed<N>/
  selection/<regime>/<candidate>/<geo>/seed<N>/
  final_fit/<regime>/<candidate>/<geo>/seed<N>/  # all-development head refits
  exploratory_test/                 # separate frozen application
  reports/
```

Keep large outputs outside `dev_plans`. Milestone records link to actual commands
and artifacts once they exist; do not create empty reports or broken links for
planned work. Prefer a small shared implementation over duplicated training code.
Use distinct regime identifiers such as `frozen_cached`, `frozen_online`, and
`joint_partial`; reserve `joint_full` for a separately declared extension.

## Risks and interpretation

| Risk | Response |
|---|---|
| Attention adds capacity but no useful interaction | Compare projected concatenation and a capacity control; report parameters |
| Normalization or a different encoder explains gains | Pair producers/preprocessing and isolate these changes |
| Augmentation is mistaken for an unfreezing benefit | Match frozen-online and joint image exposure; retain the cached reference separately |
| Encoder drift, geo shortcuts or costly overfitting | Start with partial unfreezing, bounded compute, smaller proposed encoder rates and matched image-only controls |
| Saved features no longer match trainable encoder weights | Recompute validation outputs and bind any subsequent caches to the exact checkpoint |
| One-key attention appears to be adaptive | Exclude that design; verify multiple keys within each sample |
| A global vector lacks useful local detail | Report the limitation; consider patch inputs only as a later study |
| Overfitting a small number of independent plots | Keep the model/search small, group splits, and report seed/plot uncertainty |
| Attention weights are presented as reliability explanations | Use prediction-based rescue/harm analysis; label weights descriptive |
| Reused validation/test appears to confirm a new method | State the reuse and reserve confirmatory claims for M6 |

## Progress ledger

| Milestone | Status | Evidence | Next action |
|---|---|---|---|
| M0 Governing design | Complete | This guide and [design background](00_research_and_design_background.md) | Maintain decisions |
| M1 Protocol and input audit | Complete | [Protocol and audit record](01_experimental_protocol.md), standalone config and generated inventory/matrix; A001/A002 amendments | Maintain frozen recipe and change history |
| M2 Implementation and checks | Complete | [Implementation and verification record](02_models_and_training.md); four caches, seven models, 140 tests, 19-arm smoke and exact GPU resume/replay | Hand off to M3 |
| M3 Development comparison | Planned | No attention experiments | Run the declared matrix after M2 |
| M3J Joint fine-tuning | Planned | User requested the extension; no online fusion implementation or fits | Declare the joint recipe and run matched online controls after M3, irrespective of its gain |
| M4 Exploratory application | Planned | Existing test identity known; no new application | Use frozen recipes from M3/M3J for included regimes |
| M5 Integration/reproduction | Planned | No attention integration | Document supported regimes after development and any M4 application |
| M6 Independent confirmation | Deferred | No independent dataset identified | Define an independent confirmation source |

## Detailed development records

Create a record when its milestone begins. Each should state scope/status,
decisions, implementation changes, exact commands and environment, input/output
locations, verification, results, limitations, and completion evidence. Keep
plots and detailed numerical tables out of this governing file.

| Record | Purpose | Status |
|---|---|---|
| [00 Research and design background](00_research_and_design_background.md) | Literature, attention algebra and design rationale | Recorded |
| [01 Experimental protocol](01_experimental_protocol.md) | M1 data, preprocessing, candidates, budgets, selection/refit | Complete |
| [A001 ReLU classification heads](01a_relu_classifier_amendment.md) | User-confirmed classifier activation amendment; preserves initial M1 snapshot and clarifies baseline reuse | Applied before comparative training |
| [A002 Historical image extraction](01b_historical_image_extraction.md) | User-confirmed CUDA/batch/TF32 extraction settings; strict fusion/head training unchanged | Applied before comparative training |
| [02 Models and training](02_models_and_training.md) | M2 implementation and correctness evidence | Complete |
| `03_development_comparison.md` | M3 results and selection | Not created |
| `03_joint_finetuning.md` | M3J online implementation, partial unfreezing, matched controls and results | Not created |
| `04_exploratory_evaluation.md` | M4 frozen application of selected frozen/joint regimes | Not created |
| `05_integration_and_reproduction.md` | M5 supported commands and limitations | Not created |
| `06_independent_confirmation.md` | M6 independent evaluation | Not created |

## Decision log

| ID | Date | State | Decision and reason |
|---|---|---|---|
| D001 | 2026-10-12 | Recorded | Use Attention-Based Image–Geo Fusion, short name attention fusion; namespace `attention_fusion` |
| D002 | 2026-10-12 | Design boundary | Treat this as direct supervised fusion, separate from the router's failed feasibility gate |
| D003 | 2026-10-12 | Frozen initial design in M1 | Two typed global tokens, one self-attention block, concatenated updated-token readout |
| D004 | 2026-10-12 | Design boundary | Exclude singleton-key cross-attention as evidence of adaptive weighting; preserve the residual-network distinction |
| D005 | 2026-10-12 | Comparison principle | Require matched projected concatenation and report capacity; raw-concat gains alone cannot isolate attention |
| D006 | 2026-10-12 | Resolved by M1 | Native default plus L2/z-score arms; seven families including CLS and a token-wise capacity control; three rates, four seeds, fixed 50 epochs; detailed selection/refit contract in M1 |
| D007 | 2026-10-12 | Evidence boundary | Reused validation and cleaned test remain development/exploratory evidence; independent confirmation is separate |
| D008 | 2026-10-12 | Current scope | Construct governing documentation now; individual implementation milestones follow separately |
| D009 | 2026-10-12 | User-requested extension | Add M3J partial joint image/fusion training; preserve the frozen regime and existing milestone IDs; a positive frozen-attention result is not a prerequisite |
| D010 | 2026-10-12 | Comparison principle | Compare matched frozen-online and partial-joint image-only, raw-concat, projected-concat and attention models; share initial weights/exposure, allow different learned final encoders |
| D011 | 2026-10-12 | Open before M3J fitting | Declare trainable vision layers, rates, warm-up, augmentation, effective batch size, precision, resource budget and joint refit recipe; full unfreezing remains deferred |
| D012 | 2026-10-12 | User-confirmed | Use training-partition habitat-adapted encoders for frozen experiments; retain audited legacy BGR/439 preprocessing and image L2 convention |
| D013 | 2026-10-12 | User-confirmed | Rank mean validation Acc@1 across seeds, then weighted F1 and MCC; also report Acc@3 and confusion analysis; preserve seed variation and paired plot uncertainty |
| D014 | 2026-10-12 | User-confirmed | Retain train-only heads and refit all-development heads from fresh initialization with the same fixed adapted encoders; retain checkpoints for reuse |
| D015 | 2026-10-12 | M1 implementation contract | Reuse authenticated validation image arrays; generate train arrays from the same single encoder per seed; do not consume mixed-test or OOF-producer arrays |
| D016 | 2026-10-12 | User-confirmed amendment A001 | Keep ReLU in all seven classification heads for consistency with geo-helpfulness; retain GELU adapters/fusion FFNs. Archive original M1 specification and regenerate protocol bindings before any comparative fits. Parameter counts and budgets unchanged. |
| D017 | 2026-10-12 | User-confirmed amendment A002 | Preserve historical frozen-image extraction using CUDA batches of 16, cuDNN TF32 enabled and matmul TF32 disabled; keep both TF32 flags disabled for fusion/head training. Matching settings replayed the checked historical vectors exactly. Archive previous protocol bindings before extraction. |

## Immediate next action

Proceed to **M3: the declared development comparison**, using the completed
caches and runner under the [amended M1 protocol](01_experimental_protocol.md).
The [M2 record](02_models_and_training.md) provides commands, verified artifacts
and the temporary GPU-library setup; no reboot was performed. Account for all
228 fits and preserve every final checkpoint. Training-only smoke checks remain
outside the comparison budget and cannot enter selection.
The online joint extension remains M3J; its exact recipe is frozen before its
own candidate fits under the bounded scope already recorded in M1.

## Change log

| Date | Change |
|---|---|
| 2026-10-12 | Completed M2: seven candidates and three geo treatments; four authenticated development caches with 13,512 freshly extracted train vectors; 97 focused and 43 legacy tests; 19-arm training-only GPU smoke; exact GPU resume, bundle and fresh-process replay. Checkpoints retained. M3/test evaluation not started. |
| 2026-10-12 | Applied A002 after diagnosing exact historical feature replay: CUDA batch 16 and cuDNN TF32 for frozen extraction, separate from TF32-disabled fusion/head training. GPU access restored for task processes using temporary matching libraries; no reboot or installed-driver change. |
| 2026-10-12 | Started M2: added seven configurable candidate models, explicit geo preprocessing and a dedicated development-only input/training path. Verification remains in progress; diagnosed a host NVIDIA driver/library mismatch. User requested no reboot because unlocking the encrypted drive requires input. |
| 2026-10-12 | Applied A001: ReLU classification heads across all seven candidates, GELU retained in adapters/fusion FFNs. Archived the original M1 configuration, records and generated artifacts. Clarified that image-only has no geo input, raw-concat has no learned modality projection, and compatible baseline code should be reused. |
| 2026-10-12 | Completed M1: audited four training-partition-adapted encoders, development inputs and validation caches; fixed three geo arms, initial CLS/capacity controls, 228-fit budget, Acc@1-first selection and both checkpoint-preserving head-fit routes. Added reproducible audit/protocol preparation tools. No model training or test evaluation. |
| 2026-10-12 | Extended the method to partial joint image-encoder/fusion fine-tuning in M3J, retaining frozen comparisons and milestone IDs. Added online input, checkpoint, gradient, augmentation-matching and evaluation requirements; full unfreezing remains deferred. Documentation only; no training started. |
| 2026-10-12 | Created governing guide and research/design background from the fusion discussion and repository review; recorded proposed models, open protocol choices, milestones and integration constraints. No model code or experiments added. |

Related existing references: [multimodal pipeline](../../multimodal/README.md),
[baseline configuration](../../configs/multimodal_base.yaml),
[existing grouped protocol](../reliability_aware_geo_fusion/01_experimental_protocol.md),
and [embedding geometry study](../reliability_aware_geo_fusion/21_embedding_geometry_correction.md).

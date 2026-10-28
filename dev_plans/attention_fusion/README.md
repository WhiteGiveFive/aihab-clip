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
| Current status | M0–M3, including the CLS-bypass extension, complete; M3J active fit execution complete at 84/84: seed 1 has 33 fits and seeds 2–4 have 17 active fits each; D052 seed-4 geo-only report/queue completed at 09:32:25 GMT on 2026-10-28 without error or unresolved checkpoint; cross-seed consolidation, interpretation and selection remain pending; D047 defers 48 frozen-online fits in the full 132-fit design; prior memory/read inconsistencies and seed-1 shuffle/crop discrepancy remain documented; results provisional; M4–M5 planned; M6 confirmation deferred |
| Last updated | 2026-10-28 |

[M1](01_experimental_protocol.md) freezes the initial study configuration and
records the input audit. [M2](02_models_and_training.md) implements all seven
candidate models, prepares all four development caches, and verifies training,
checkpoint resume and bundle replay. Its small training-only smoke runs are
implementation checks. [M3](03_development_comparison.md) completed all 228 frozen
fits, retained all checkpoints and selected 76 references. Its primary native-geo
attention comparison gives **no improvement** under the declared rule: Acc@1
68.80% versus 68.98% for projected concatenation. Secondary standardized-geo raw
concatenation has the highest observed mean (71.38%); these are reused-validation
development findings. No attention-specific gain is supported in that completed
comparison. It did not test CLS with a modality bypass. A user-requested
[M3 extension](#m3-extension--cls-with-modality-bypass) has completed all 72
additional fits and selected 24 checkpoints. Both native-geo bypass comparisons
against CLS-only are **inconclusive**: adapted-token Acc@1 68.61% and
original-embedding 68.46%, versus CLS-only 68.55%. None of the 37 extension
contrasts supports improvement. The [bypass extension record](03a_cls_bypass_extension.md)
contains the results, exact replay evidence and inspected report; the original
M3 evidence remains unchanged.

M3J's user-confirmed scope now includes all nine M3/extension architectures,
both externally pretrained and training-partition CS-adapted image
initializations, and matched frozen-online/partial-joint regimes. Its initial
geo treatment is training-fitted coordinate standardization. Image preprocessing
preserves the historical adaptation recipe in both initialization arms. The
full planned matrix has 132 unique fits if one recipe is used per architecture.
D047 defers the 48 seed-2/3/4 frozen-online fits for later investigation,
leaving 84 fits in the current scope, including the completed seed-1 controls;
its initial learning rate is uniformly `5e-5` for all trainable components,
with shared cosine decay and without warm-up or module-specific rates.
Use historical Adam settings with zero weight decay and physical/effective
batch 16, without gradient accumulation. Train for **50 epochs** using
historical CUDA FP16 autocast with FP32 weights and no gradient scaler; retain
both final-epoch and best-validation checkpoints. Use the user-approved shared
checkpoint storage layout (option 3, D033), with a 400 GB planning allowance.
Execute one seed at a time. Seed 1 completed the original partial-joint →
frozen-online → geo-only order (D034). For seeds 2–4, complete partial-joint →
geo-only before advancing; frozen-online is explicitly deferred under D047.
The user authorized the first **16 seed-1 partial-joint fits** (D035).
The [joint-training record](03_joint_finetuning.md) now specifies execution,
contrasts, checkpoint policies and the prospective refit contract. Comparative
fitting follows successful training-only GPU verification. Earlier strict and
tolerance-based checks failed; the user has now approved a practical verification
contract (D038): direct frozen-value checks, no mandatory resume comparison, no
tolerance machinery and no repeated live-weight checksum gate. Saved final/best
checkpoints are loaded and evaluated without exact prediction replay requirements.
The completed partial-joint study is `attention_fusion_joint_direct_v1`. Its implementation is complete
and 163 relevant CPU checks passed. Complete GPU verification passed all 16
training/frozen-value checks and checkpoint reload/export. The authorized 16-fit
background job stopped after 12 completed fits. Fit 13,
`attention_cls_skip_adapted/pretrained`, completed all 50 training epochs and
passed its final direct frozen-value comparison, then failed to authenticate
its saved best-epoch visual file during checkpoint evaluation. Read-only
per-file cache advice restored the original expected checksum on two subsequent
reads; no checkpoint bytes or expected checksum were changed.
The user approved D039: bounded cache-refresh retries, evaluation-only recovery
of this trained fit, and continuation of the three unstarted fits while recording
any persistent output-checkpoint failure separately. D039 is implemented and
134 focused checks passed. Continuation launched at 15:48 UTC on 2026-10-20
(16:48 BST). The thirteenth fit's final/best evaluation completed at 15:51 UTC
with zero additional training epochs; the recovery preservation check confirmed
152 recorded files unchanged. The seed-1 partial-joint package completed
**all 16 fits at 21:39 UTC (22:39 BST) on 2026-10-20**: 800 training epochs,
32 final/best checkpoint-policy evaluations, and no unresolved fits or queue
error. See the [completion record and retained outputs](03_joint_finetuning.md#seed-1-partial-joint-completion-results-and-handoff).
M3J overall remains in progress. The
[16 seed-1 frozen-online fits](03b_seed1_frozen_online.md) completed at
**00:49:39 UTC (01:49 BST) on 2026-10-22**, with all 50 epochs and both
checkpoint-policy evaluations, no queue error and no unresolved fit.
Their separate study is `attention_fusion_joint_frozen_online_direct_v1`.
Seed 1 is **complete at 33/33 fits**. The
[shared geo-only fit](03c_seed1_geo_only.md) completed 50 epochs and final/best
evaluations at **10:17:19 UTC (11:17 BST) on 2026-10-22**; the combined report
and queue completed at **10:18:23 UTC (11:18 BST)**. D041 passed 214 CPU checks
and training-only GPU/checkpoint verification. All 1,237 tracked prior-package
files were preserved. This closed seed 1 at 33 completed fits.
The [later-seed preparation record](03f_later_seed_preparation.md) implements
nine separate seed-2/3/4 configurations under D045, with CPU-only preparation,
same-seed predecessor references and the existing D038/D039 recovery behaviour.
Later image packages explicitly use the declared shared shuffle/crop namespaces;
seed 1 retains its historical behaviour. Seed-2 partial-joint GPU verification
passed at 20:55:24 UTC on 2026-10-23; verification for subsequent packages stays
a pre-launch step. D046's 16 seed-2 partial-joint fits completed at
**05:06:56 UTC (GMT) on 2026-10-25**, with 800 epochs, 32 saved-checkpoint
evaluations and no queue error or unresolved checkpoint; see the
[seed-2 execution record](03g_seed2_partial_joint.md). The
[D047 deferral record](03h_frozen_online_deferral.md) recorded 49/84 active fits
complete at that amendment, with 48 additional fits deferred in the original
132-fit design. D048's [seed-2 geo-only package](03i_seed2_geo_only.md) completed
50 epochs and final/best evaluations at **09:24:33 UTC (GMT) on 2026-10-25**;
its combined report and queue completed at **09:25:06 UTC**. Seed 2 is complete
at **17/17 active fits**, bringing M3J to **50/84 active fits**. D049's
[seed-3 partial-joint package](03j_seed3_partial_joint.md) passed its refreshed
preparation and all 16 training-only GPU checks plus checkpoint loading/export,
then launched at **09:40:33 UTC (GMT) on 2026-10-25**, process **65206**.
Its startup snapshot had **0/16 complete fits**. The package completed
**16/16 fits at 16:23:18 GMT on 2026-10-26**, with **800 epochs** and **32
final/best checkpoint evaluations**. D050's [seed-3 geo-only package](03k_seed3_geo_only.md)
then completed 50 epochs and both saved-checkpoint evaluations at
**16:53:57 GMT**; its combined report and queue completed at **16:54:30 GMT**.
No queue error or unresolved checkpoint is recorded. Seed 3 is complete at
**17/17 active fits**, bringing M3J to **67/84 active fits complete**, with
**17 active seed-4 fits remaining** and **48 frozen-online fits deferred**.
The unchanged M3J recipe and D038/D039 recovery apply. D051 subsequently
authorizes the [16 seed-4 partial-joint fits](03l_seed4_partial_joint.md). Their
refreshed preparation and package-specific training-only GPU verification
passed; the fresh background queue launched at **22:47:57 GMT on 2026-10-26**,
process **173284**. The first fit reports running at **22:49:39 GMT**, with
**0/16 complete fits** and no queue error or unresolved checkpoint at startup.
The package completed **all 16 fits at 05:41:20 GMT on 2026-10-28**, with
**800 epochs**, **32 final/best evaluations**, no queue error and no unresolved
checkpoint, bringing M3J to **83/84 active fits**. D052 subsequently authorizes
the [shared seed-4 geo-only fit](03m_seed4_geo_only.md). Preparation and its own
training-only GPU/checkpoint verification passed; the fresh run launched at
**09:31:05 GMT**, process **316572**. All 50 epochs and final/best evaluations
finished at **09:31:51 GMT**, and the combined report/queue completed at
**09:32:25 GMT**, without error or unresolved checkpoint. Seed 4 is **17/17
active fits complete**. M3J active fit execution is now **84/84 complete**;
cross-seed consolidation, interpretation and selection remain pending, and
**48 frozen-online fits** remain deferred.
A loader audit found that the completed seed-1 image packages used their distinct study IDs for shuffle and crop randomness,
rather than the declared common namespaces. Data and recipes match, but realized
training views and minibatch orders are not paired across those regimes. Preserve
the results and qualify joint-versus-frozen interpretations accordingly.
Frozen references and saved encoder weights now use the authenticated original
encoder; the record preserves the failed checks and reference-copy diagnosis.
Earlier memory/read inconsistencies and failed receipts remain preserved;
all outputs retain `provisional_unresolved_memory_integrity`. This amendment
does not establish machine health or bit-for-bit reproducibility.

A bounded [attention-fusion geometry-router side study](../attention_fusion_router/README.md)
now branches from the completed seed-1 results (D042). It uses only the
pretrained-initialized partial-joint CLS + original-embedding fit and the shared
geo expert. It compares original-pretrained image geometry with geometry from
the matching final/best fusion visual checkpoint. R1/R2 are complete: three
encoder extractions, 12,600 vectors and four verified geometry score sets.
The eight routing evaluations in R3 are also complete (D043): original-pretrained
geometry's selected rules gain 24/29 net correct validation images for the
final/best expert pairs; matching trained geometry retains fusion under the
declared no-feasible-threshold fallback. R4's
[findings and handoff](../attention_fusion_router/04_findings_and_handoff.md)
are complete (D044), including habitat/plot accounting, paired geometry
diagnostics and conditional uncertainty. Work returned to the main study after
R4; the remaining execution order is recorded below.
Router application to cleaned test is deferred until the main study completes
and receives its own later R5 protocol. The main M3J fit matrix and completed
training artifacts remain unchanged.

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
Partial joint fine-tuning is the first encoder-unfreezing extension, planned in
M3J; full image-encoder fine-tuning is a later, separately declared capacity
experiment.

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

For the completed frozen study, M1 fixes **native geo as default**, with
L2-normalized and training-fitted coordinate-standardized geo as secondary
comparison arms. Native and L2 geo
remain distinct: L2 can remove magnitude information. Initial image producers
are the four training-partition habitat-adapted encoders verified in M1.
Compare architectures under identical preprocessing within each arm.
Do not attribute a normalization gain to attention, silently normalize twice,
or feed a historical checkpoint differently from its training convention.

For M3J, the user instead selected **training-fitted coordinate-standardized
geo from the outset** (D026). Fit one scaler on the 3,378 development-training
rows using M1's feature order, population standard deviation and constant-feature
handling; apply the same saved state to training and validation inputs across
all seeds, image initializations, regimes and geo-consuming architectures.
Validation contributes no statistics. Both branches of the original-embedding
CLS bypass receive the same standardized geo vector: "original" denotes its
unprojected width, not native geo values. Image-only has no geo input. Native
and L2 comparisons may be declared later; they are not part of this initial
M3J matrix. This amendment leaves the completed M3 preprocessing arms and
native-geo primary outcomes unchanged.

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

The initial joint recipe unlocks the historical adaptation scope: **the final
10 Transformer blocks, final normalization and attention-pooling head** (D029).
For the pinned SigLIP2 tower these are blocks 17–26 (zero-based), `trunk.norm`
and `trunk.attn_pool`; all other visual parameters remain frozen. Record exact
parameter names and counts before fitting: the historical 11 framework groups
do not mean 11 Transformer blocks. For the initial M3J comparison,
use the same initial learning rate, **`5e-5`**, for every trainable component
(D027). Partial-joint models update the declared vision subset from the first
optimizer step; frozen-online models keep the vision encoder fixed throughout.
There is no head-only or learning-rate warm-up in this initial recipe. Separate
optimizer groups may identify components for monitoring, but must not introduce
different learning rates. Apply the same single cosine decay to every
trainable group over the declared training duration (D028). Warm-up and
module-specific rates are deferred to separately declared follow-up comparisons.
Use the historical Adam optimizer with zero weight decay and physical/effective
batch 16, without accumulation (D031). Train for 50 epochs with the historical
FP16-autocast/FP32-weight policy (D032). Record the exact execution and refit
procedure before comparative fits.

### Alternatives and exclusions

- A two-layer `[CLS, image, geo]` model with updated-CLS readout is required in
  the initial study. Its second layer can read contextualized modality
  tokens. It is a separate candidate with its own parameter count. Comparing it
  with one-layer updated-token concatenation changes both depth and readout;
  isolating a readout effect would require a same-depth comparison.
- CLS with a modality bypass is a separately versioned M3 extension, covering both
  `[updated CLS; original adapted image token; original adapted geo token]`
  and `[updated CLS; original image embedding; original geo embedding]`.
  The adapted bypass includes the modality-type embeddings used at the
  Transformer input. The original-embedding bypass uses the same image and
  geo preprocessing as its attention branch. Concatenation is a skip path,
  not elementwise residual addition. See the exact definitions below.
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
| CLS + adapted-token bypass | Preserve pre-attention adapted modality tokens alongside updated CLS | Completed M3 extension |
| CLS + original-embedding bypass | Preserve original-width modality embeddings alongside updated CLS | Completed M3 extension |
| Feature conditioning | Study a simpler interaction mechanism | Optional later extension |

Do not run an unrestricted product of widths, depths, readouts, and normalization
choices. M1 fixes 19 candidate/preprocessing arms, three learning rates and four
seeds: **228 development fits**, each 50 epochs. Shared
adapter width and classifier design do not imply equal parameter counts: report
the difference and use the capacity control before attributing gains specifically
to attention weights.

The M3 bypass extension adds **72 fits**: two architectures × three geo
treatments × three learning rates × four seeds. This is a separate, versioned
extension to the completed 228-fit study, not a revision of its candidate matrix
or primary outcome. The original token-wise capacity control was matched to
the two-token model; it is not automatically capacity-matched to either bypass.

M3J includes **all nine architectures explored in M3 and its extension**
(D025). Preserve each architecture's existing width, depth, activation and
readout definition; expanding the encoder regimes does not introduce new
architecture variants. Each of the eight image-using models has both
externally pretrained and seed-matched training-partition CS-adapted
initializations in both encoder regimes:

| Model | Frozen online, both initializations | Partial joint, both initializations |
|---|---|---|
| `image_only` | Required | Required |
| `raw_concat` | Required baseline anchor | Required baseline anchor |
| `projected_concat` | Required | Required |
| `tokenwise_capacity` | Required | Required |
| `attention_two_token` | Required | Required |
| `attention_cls` | Required | Required |
| `attention_cls_skip_adapted` | Required | Required |
| `attention_cls_skip_original` | Required | Required |
| `geo_only` | Shared fixed-input reference | Same shared reference; no encoder to unfreeze |

**D047 scheduling amendment:** the frozen-online column is complete for seed 1
and deferred for seeds 2–4. The partial-joint and shared geo-only columns remain
in the current four-seed scope. A later seed's combined report covers 17 active
fits and lists its 16 frozen-online fits as deferred; it must not present them
as failed fits or claim a completed cross-regime comparison.

With one predeclared training recipe per architecture and four seeds, the
planning baseline is **128 image-using fits**
(8 architectures × 2 initializations × 2 regimes × 4 seeds), plus **4 geo-only
fits**, for **132 unique fits**. Geo-only has neither an image initialization
nor an encoder regime: repeated table entries reference the same four fits
and are not independent observations. Final and best-validation checkpoints
come from the same trajectory and do not double this count.

This expanded scope supersedes the earlier four-model, **96-fit M3J planning
cap** recorded in M1 and its historical configuration; those completed-study
files remain unchanged. The 132-fit total assumes one recipe per architecture,
not the previous three-recipe search. Declare the executable recipe and final
compute budget before comparative training, and count any additional recipes
or geo treatments explicitly. The user selected **50 epochs per fit** (D032),
using the existing ceiling as the actual budget: the one-recipe matrix totals
**6,600 training epochs** across 132 fits. A training-only resource check must
precede full scheduling; no wall-time estimate is established by this count.
D047 partitions this full design into **84 active fits / 4,200 epochs** and
**48 deferred fits / 2,400 epochs**. Deferral changes scheduling and available
contrasts, not any individual fit's recipe or checkpoint policy. Four-seed
joint-versus-frozen conclusions remain unavailable until those controls run.

Geo-only uses the new declared M3J head-training and checkpoint policy. Reuse
earlier results only if their full input, exposure, training and checkpoint
recipe matches. Under D027, M3J geo-only uses `5e-5`; the existing M3 geo-only
grid used `1e-4`, `3e-4` and `1e-3`, with `1e-4` selected for standardized geo.
Those models do not match the new learning-rate recipe, so the initial M3J
matrix requires four new geo-only fits and retains the old checkpoints as
historical evidence. Fit the frozen-online controls with the same
augmentations and image views as partial joint training. An equivalent declared cached-view
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
   comparisons share the encoder producer. Within each M3J initialization,
   architecture/regime comparisons share the starting checkpoint and declared
   optimizer/selection budgets; partial-joint models share the trainable vision
   parameter scope. The initialization contrast changes starting visual weights
   while matching the downstream recipe within each architecture. Final encoder
   weights may differ by method. Match image augmentations,
   view exposure and effective batch size in the frozen-online/joint contrast.
   Pair head/component initialization and independent data/augmentation RNG
   streams, rather than relying on the same integer seed alone. Report the
   adapted encoder's five prior CS adaptation epochs separately: equal M3J
   training gives equal additional exposure, not equal lifetime supervision.
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
amendments. M3J reports both final-epoch and best-validation checkpoints under
the [dual-checkpoint contract](#final-and-best-validation-checkpoints); the
completed M3 studies retain their fixed epoch-50 rule. Report frozen-to-joint
gains within each image-using model and initialization, initialization contrasts
within each architecture/regime, and attention versus concatenation/capacity
controls within each initialization and regime. Include trainable encoder
counts and peak memory alongside runtime.

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

**Original comparison status:** Complete, 2026-10-16. All 228 fits, 76 selected references, 63 paired
contrasts, reproducibility checks and real-result visual review delivered. The
primary result is no improvement. See the
[development comparison record](03_development_comparison.md).

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

#### M3 extension — CLS with modality bypass

**Status:** Complete, 2026-10-16. Both variants, all 72 fits, 24 selected
references, 37 contrasts and reproducibility/figure checks are delivered in the
[detailed extension record](03a_cls_bypass_extension.md). Both native primary
outcomes are inconclusive; no extension contrast supports improvement. This
remains part of M3's frozen-encoder study and does not rename or replace M3J.

Use the existing two-layer `attention_cls` backbone with width 128, four heads,
FFN width 256 and the same dropout/normalization settings. Let
\(t_i^0=P_i(x_i)+e_i\) and \(t_g^0=P_g(x_g)+e_g\) be the typed 128D modality
tokens before attention, and \(c^2\) the updated CLS after the two blocks.
Here \(x_i\) is the existing L2-normalized 1,152D cached image vector and
\(x_g\) is the 64D geo vector after the declared native, L2 or training-fitted
standardization treatment. Neither bypass uses updated modality tokens.

| Candidate ID | Classifier input | Input width |
|---|---|---:|
| `attention_cls_skip_adapted` | \([c^2; t_i^0; t_g^0]\) | 384 |
| `attention_cls_skip_original` | \([c^2; x_i; x_g]\) | 1,344 |

Both candidates retain the same 256-unit `Linear -> ReLU -> Dropout -> Linear`
head design. Its first layer grows with input width, so equal hidden width does
not imply equal parameter count. The adapted bypass must remain differentiable
through its adapters; do not detach it. All image encoders stay frozen, and both
candidates use the same cached representations and geo treatments as M3.

**Frozen fitting and comparison contract:**

- Keep the original train/validation allocation, seeds 1–4, three learning rates
  (`1e-4`, `3e-4`, `1e-3`), 50 fixed epochs, batch 128, optimizer, strict float32
  training and component-wise initialization conventions. Native geo remains
  primary; L2 and standardization are secondary. The core extension budget is
  72 additional fits and 24 selected checkpoint references.
- Initialize both variants afresh; do not warm-start from selected M3 weights.
  Preserve `attention_fusion_v1` as the initialization seed namespace for shared
  components so adapters, modality identities, CLS and attention layers match
  the original same-seed initialization where shapes agree. The current seed
  derivation includes `study_id`; the new artifact/study identity must therefore
  be separated explicitly from that seed namespace in the extension recipe.
  Different-width head layers receive their declared deterministic initialization,
  not a claim of identical weights across incompatible shapes.
- Select one learning rate per candidate/geo arm by four-seed mean validation
  Acc@1, then weighted F1, MCC and the existing exact-tie rule. Retain all four
  seeds and use epoch 50. Report all declared metrics, class confusions,
  rescue/harm, learning curves, parameter counts, runtime and memory.
- Declare two primary extension contrasts in native geo: each bypass versus
  the existing two-layer CLS-only model. Report both regardless of outcome.
  Supporting contrasts compare the two bypasses directly and compare each with
  matched projected concatenation, raw concatenation, two-token attention and
  image-only; include all secondary geo arms and within-architecture geo effects.
  A secondary winner cannot replace either primary contrast after scoring.
- Reuse completed comparator checkpoints only after verifying their full input,
  preprocessing, initialization, fitting and selection provenance. Do not rerun
  or overwrite the original 228 fits simply to add a readout. The original
  primary contrast and no-improvement outcome remain recorded independently.
- Use the existing paired whole-plot bootstrap approach and M1 outcome rules.
  Freeze the exact contrast list, bootstrap settings and interpretation in the
  extension record before fitting. State that this extension was proposed after
  seeing M3 results and uses the same validation set; it is additional development
  evidence, not an independent confirmation or a multiplicity-adjusted guarantee.
- Interpret a gain over CLS-only as a gain from the complete bypass configuration:
  it changes direct feature access and classifier capacity. Differences between
  the two bypasses also change representation width. Do not attribute these
  gains solely to attention or readout preservation. Any additional capacity
  control needed for such attribution must have its architecture, contrasts and
  added fit budget declared before its fits; the 72-fit plan does not silently
  include an open-ended control search.

The [extension record](03a_cls_bypass_extension.md) and separate resolved
configuration use study identity `attention_fusion_cls_bypass_v1`.
Preserve the original M1 protocol/configuration,
M3 artifacts, selection records and report. Link reused inputs and comparator
artifacts by verified identity instead of modifying the completed study.

Deliverables: both model variants; shape, bypass/gradient and sample-isolation
checks; resumable training and replayable inference bundles; all 72 fit results,
24 selected references, paired comparisons, inspected figures and a detailed
extension record. Preserve every candidate checkpoint, predictions, fitted
preprocessing, histories and source/input identities, including unsuccessful
variants. Acceptance includes reproducible checkpoint/analysis replay and an
explicit supported-improvement, inconclusive or no-improvement result for each
primary contrast. This extension does not include cleaned-test evaluation,
all-development refits, encoder unfreezing or new image feature extraction.

### M3J — Jointly fine-tune the image encoder and fusion model

**Status:** Active fit execution is **84/84 complete** as of **09:32:25 GMT on
2026-10-28**: all four partial-joint packages, four shared geo-only references
and the seed-1 frozen-online package. Cross-seed consolidation, interpretation
and selection remain pending; 48 later-seed frozen-online fits remain deferred.
M3J's scientific development report and selected-model decision are not yet
complete.

Seed-1 partial-joint completed **16/16 fits** at 21:39 UTC (22:39 BST) on
2026-10-20.
The original D038 implementation passed 163 relevant CPU checks and complete
training-only GPU verification. The active study
uses direct frozen-value comparisons and checkpoint loading/evaluation, without
resume comparison, numerical tolerance or live-weight checksum gates. Earlier
memory/read inconsistencies remain unresolved and results remain provisional.
All 16 fits completed 50 epochs, passed their recorded end-of-fit frozen-value
checks, and have reloaded final/best checkpoint evaluations. The final queue
and report record no unresolved or incomplete fits. D039 recovered the thirteenth
fit's evaluation with zero additional training epochs; its 152-file preservation
check passed. The [joint-training record](03_joint_finetuning.md#seed-1-partial-joint-completion-results-and-handoff)
links the results, curves, reusable checkpoints and metadata completion audit.
The [16-fit seed-1 frozen-online package](03b_seed1_frozen_online.md) completed
on 2026-10-22 at 00:49:39 UTC (01:49 BST), including final/best evaluations,
with no unresolved fits. Final/best checkpoints share the authenticated original
encoder. The [geo-only package](03c_seed1_geo_only.md) also completed all 50
epochs and both evaluations under D041. Seed 1 is complete at **33/33 fits**,
1,650 training epochs and 66 final/best checkpoint-policy evaluations. The
combined report and queue completed on 2026-10-22 at 10:18:23 UTC (11:18 BST).
The D041 loader audit also found different realized shuffle/crop streams between
the completed image packages; their regime contrast does not isolate encoder
trainability under exactly paired views and orders.
M2 and the recorded M3 frozen comparison are complete; no positive frozen-attention
result is required. Other M3J packages remain planned.

First declare the expanded joint recipe and matched frozen-online controls
in the [comparison matrix](#scope-and-comparison-set).
Implement a pixels + geo training path and partial vision unfreezing, keeping
each existing global-vector fusion architecture unchanged. Train the expanded
comparison matrix under the same starting-checkpoint and data-exposure contract, across
the declared seeds. Measure gains attributable to joint image adaptation and
whether attention improves on jointly trained concatenation. Full vision
unfreezing remains deferred, not an automatic search expansion.

#### Execution order and sub-tasks

Retain the user's seed-by-seed schedule, amended by D047 for seeds 2–4.
Complete the relevant implementation, detailed protocol and training-only
numerical/resource/checkpoint checks before comparative fitting. Seed 1 has
completed all three original sub-tasks; seeds 2–4 have each completed both
active sub-tasks. The final active packages and deferred work are:

| Order | Sub-task | Fits |
|---|---|---:|
| Completed, D051 | Seed-4 partial-joint: all 16 fits and dual-checkpoint evaluations complete at 05:41:20 GMT on 2026-10-28 | 16 |
| Completed, D052 | Shared seed-4 geo-only reference; 50 epochs and final/best evaluation complete, report/queue completed at 09:32:25 GMT on 2026-10-28 | 1 |
| Deferred | Seed-2/3/4 frozen-online: eight architectures and two initialization arms per seed | 48 |

The full design retains **33 fits per seed / 132 fits**; the current scope is
**33 fits for seed 1 plus 17 each for seeds 2–4 / 84 fits**.
The [seed-1 frozen-online record](03b_seed1_frozen_online.md) records the
completed second package and its comparison limitations. Both 16-fit image
packages and the [seed-1 geo-only package](03c_seed1_geo_only.md) are complete,
with separate implementation/verification records and outputs. The
[seed-2 partial-joint package](03g_seed2_partial_joint.md) and
[shared geo-only package](03i_seed2_geo_only.md) are also complete: 17 active
fits, 850 epochs and 34 final/best evaluations.
D049's [seed-3 partial-joint execution](03j_seed3_partial_joint.md) completed
all 16 fits at **16:23:18 GMT on 2026-10-26**, with 800 epochs and 32
final/best evaluations. D050's [shared seed-3 geo-only fit](03k_seed3_geo_only.md)
completed its 50 epochs and both saved-checkpoint evaluations at **16:53:57 GMT**;
the combined report and queue completed at **16:54:30 GMT**. Seed 3 has
**17/17 active fits**, 850 epochs and 34 evaluations, with no queue error or
unresolved checkpoint. D051's [seed-4 partial-joint queue](03l_seed4_partial_joint.md)
launched at **22:47:57 GMT**, process **173284**, after refreshed preparation
and training-only GPU/checkpoint verification. It had **0/16 complete fits**
at launch and completed all **16 fits at 05:41:20 GMT on 2026-10-28**:
800 epochs and 32 final/best evaluations, without queue error or unresolved
checkpoint. The [shared seed-4 geo-only reference](03m_seed4_geo_only.md)
launched separately under D052 at **09:31:05 GMT**, process **316572**,
after its own preparation and training-only GPU/checkpoint verification.
Its 50 epochs and final/best evaluations completed at **09:31:51 GMT**, and
its combined report/queue completed at **09:32:25 GMT** without errors or
unresolved checkpoints. Seed 4 has **17/17 active fits**, 850 epochs and 34
evaluations. All **84 active fits** have completed, totaling **4,200 epochs**
and **168 checkpoint-policy evaluations**; cross-seed consolidation and
selection remain pending. The 48 deferred fits remain outside that completion.
D045 [prepared all nine later-seed packages](03f_later_seed_preparation.md).
D047 updates only the later-seed geo dependencies and combined-report scope;
the frozen-online configurations remain available for later investigation.
Their loader correction retains declared shuffle/crop streams across regimes
while preserving seed 1's completed evidence.

Geo-only requires its same-seed completed partial-joint package under D047;
it has no image-encoder dependency. Preserve same-seed, cohort, recipe and
completion checks on that active reference. Do not read a deferred package as
a prerequisite, fabricate its results, or duplicate the geo fit when the
frozen-online study is later reactivated. Record the later reporting/activation
amendment separately before adding those results to a combined comparison.

Each fit independently reloads its designated starting encoder and initializes
its own fusion/head, optimizer and random-number state. The frozen-online fits
start from the same original/CS-adapted initial checkpoints as their joint
counterparts, never from a newly joint-trained encoder. Preserve paired
initialization and augmentation streams regardless of execution order.

Persist progress, histories, final/best references and separate latest training
state for each sub-task. Automatic training resume remains disabled under D038.
D039 continuation preserves completed runs and retries only checkpoint
evaluation for a fit that already completed all 50 epochs and passed its final
freeze check. It does not repeat training or automatically restart a partially
trained attempt. A persistent output-checkpoint authentication failure is
recorded for that fit while independent unstarted fits can proceed; it is not
treated as a completed result. Any separately authorized fresh attempt must
preserve the old attempt and begin a new history at epoch 1.
Treat first-seed performance as provisional; joint-versus-frozen comparisons
require the matching frozen-online fits and cross-seed conclusions require all
declared seeds. Preserve the full candidate matrix and recipe across seeds.
If a later change is needed, record a separate amendment and consistently rerun
affected comparisons rather than combining different recipes in one summary.

#### Confirmed initialization and preprocessing choices

Use the pinned original SigLIP2 checkpoint and the four audited seed-specific
training-partition CS-adapted visual checkpoints. Match image preprocessing,
augmentation and the downstream training recipe across initialization arms;
initialize fusion/head components afresh rather than importing trained M3 heads.
Both initialization families must exclude development-validation plots from
any project-specific encoder adaptation.

The user confirmed the historical CS adaptation pixel/transform recipe for both:

| Stage | Shared M3J choice |
|---|---|
| Decode and preliminary resize | `legacy_opencv_bgr_439_v1`: OpenCV BGR, forced 439×439 linear resize, PIL conversion without channel swapping |
| Training transform | Random-resized crop to 384×384, scale `[0.9, 1.0]`, ratio `[0.75, 1.3333]`, bicubic interpolation and antialiasing; no flips or colour jitter |
| Validation transform | Deterministic 384×384 bicubic resize with antialiasing |
| Tensor conversion/normalization | PIL RGB-mode conversion, tensor conversion, per-channel mean/std `[0.5, 0.5, 0.5]`; RGB-mode conversion does not undo the legacy channel ordering |
| Global image vector | Differentiable L2 normalization of the 1,152D encoder output |
| Geo vector | The same training-fitted coordinate-standardized 64D GSE vector for every geo-consuming model and branch |

These transform definitions match the historical CS adaptation/extraction
recipe. Historical adaptation did not evaluate validation during its five
epochs; M3J instead trains for 50 epochs and adds the every-epoch validation
required below. The user selected the historical numerical precision policy
for this longer M3J comparison (D032).

#### Confirmed visual scope and physical batch

For partial-joint training, reuse the historical **final 10 Transformer blocks
(indices 17–26 of 27), final normalization and attention-pooling head**.
Use the same visual parameter subset for both image initializations and all
eight image-using architectures. Other visual parameters and the text tower
remain frozen; the new adapters, fusion modules and classifier train where
present. Frozen-online controls keep the entire visual encoder frozen.

The historical configuration calls this `unlocked_groups_from_end: 11`.
Resolve and verify the exact parameter names/counts against the pinned model
before fitting; group numbers alone are not a portable description of blocks.

The user resolved the batch choice as **physical batch 16 and effective batch
16, with no gradient accumulation** (D031). Perform one optimizer update per
minibatch (`gradient_accumulation_steps = 1`), matching historical CS adaptation.
Shuffle training rows each epoch with the paired seeded schedule and retain
the final smaller batch (`drop_last = false`); do not pad, drop or accumulate
that batch into another epoch. For 3,378 training rows this gives 211 full
batches and one final two-row batch, hence 212 optimizer updates per epoch.
At 50 epochs, each complete fit performs **10,600 optimizer updates**.

Apply this batch policy across matched models, regimes and initializations,
including geo-only. Verify physical-batch feasibility with the declared
precision in a training-only resource check before comparative fitting.
The earlier physical-16/effective-128 proposal is not adopted; completed M3
and its bypass extension used physical batches of 128 cached vectors without
accumulation, and their historical records remain unchanged.

#### Historical optimizer settings

The user chose to keep the M3J optimization setup consistent with historical
CS encoder adaptation (D031). Use these settings for every trainable component
and all model/regime/initialization comparisons, including geo-only:

| Setting | Initial M3J choice |
|---|---|
| Optimizer | Adam |
| Initial learning rate | `5e-5`, shared across trainable components |
| Betas | `(0.9, 0.999)` |
| Epsilon | `1e-8` |
| Weight decay | `0.0` |
| AMSGrad | Disabled |
| Gradient clipping | None |
| Gradient accumulation | None; one update per physical batch of 16 |
| Training duration | 50 epochs, without early stopping |
| Rate schedule | Shared cosine decay over 50 epochs, zero minimum, step after each epoch |
| Warm-up/restarts | None |

Initialize Adam state afresh for every new fit, including those starting from
CS-adapted visual weights and replacement attempts after interruptions. Saved
training state is retained for reuse; automatic training resume is disabled.

This supersedes the earlier AdamW/weight-decay-`1e-4` proposal. The user also
selected 50 M3J epochs and the historical precision policy (D032). The cosine
period is therefore 50 epochs. Historical adaptation used only five epochs;
its shorter duration is not inherited by M3J.

#### Training duration and numerical precision

Complete **50 epochs for every M3J fit**, including frozen-online controls and
geo-only references, without early stopping. Save final epoch 50 and the
best-validation checkpoint selected from completed epochs 1–50 under the
existing tie-break rules. The full 132-fit matrix is partitioned into 84 active
and 48 deferred fits by D047; the two checkpoint policies do not create extra
training fits.

Apply the historical encoder-adaptation precision policy to the complete M3J
training forward pass and classification loss, including the new adapters,
fusion and classifier where present. Use the same policy for both image
initializations, both encoder regimes and geo-only:

| Precision setting | M3J policy |
|---|---|
| Model parameters and floating-point Adam state | FP32; do not permanently convert model weights to FP16 |
| Training computation | CUDA automatic mixed precision with explicit FP16 autocast target; operators retain their autocast-defined precision |
| Gradient scaling | None, matching the historical direct backward/optimizer-step path |
| Full validation and saved-checkpoint evaluation | Evaluation mode, no gradients, FP32 inputs/weights and autocast disabled |
| cuDNN TF32 | Enabled |
| CUDA matrix-multiplication TF32 | Disabled; float32 matmul precision `highest` |

All four historical producer manifests record FP16 autocast, no gradient
scaler and these TF32 flags. Historical prediction/feature extraction ran
without autocast; use that prediction convention for M3J's new full-validation
and saved-checkpoint evaluation paths. This is not pure-FP16 model storage, nor a claim that every
operation uses the same dtype. The new integrated fusion/head training adopts
the encoder-adaptation policy; historical frozen-head M3 training itself used
FP32 with both TF32 flags disabled and remains unchanged.

Pin the arithmetic policy in the M3J configuration, run metadata, checkpoint
bundles and evaluation contract. The training-only resource/gradient checks must
verify finite loss, gradients and updates for this recipe. Do not silently
introduce a gradient scaler, BF16, full-FP32 training or another backend policy
if a check fails; any repair that changes numerical policy needs a recorded
amendment applied consistently to the matched comparison. The earlier proposed
full-FP32 M3J training policy is superseded by D032.

#### Initial learning-rate strategy

Use a **uniform initial learning rate of `5e-5`**, with **shared cosine decay**
and **no warm-up**, for the first comparison (D027/D028). This replaces the
earlier proposed architecture-specific head rates, smaller encoder rate and
five-epoch head-only warm-up.

| Regime/model | Components updated from the first optimizer step | Initial learning rate |
|---|---|---|
| Frozen-online image-using model | Adapters, fusion and classifier, where present; encoder stays fixed | `5e-5` for every trainable component |
| Partial-joint image-using model | Declared vision subset plus adapters, fusion and classifier, where present | `5e-5` for every trainable component |
| Geo-only reference | Classifier; no image encoder | `5e-5` |

Apply this policy to all nine architectures and both image initialization arms,
with no architecture-, initialization- or module-specific rate tuning in the
initial comparison. The user selected **cosine learning-rate decay** for
consistency with historical CS adaptation (D028). Use one cosine decay from
`5e-5` toward zero, without restarts or warm-up, shared by all trainable groups
and matched models, regimes and initializations, including geo-only.

Preserve the historical end-of-epoch stepping and zero minimum rate. Set the
cosine period to the confirmed M3J training budget, **`T_max = 50` epochs**.
For training epoch `e = 1, ..., 50`, the learning rate is
`5e-5 * (1 + cos(pi * (e - 1) / 50)) / 2`. Step the scheduler once after each
completed epoch. Epoch one therefore trains at `5e-5`; the final epoch trains
at a positive rate, and the scheduler reaches zero after that epoch. Record
the rate used for each epoch separately from the next rate after stepping.
Every fresh attempt starts the same scheduler at epoch 1.

The [historical CS adaptation recipe](../../configs/multimodal_geo_helpfulness.yaml)
used Adam, initial rate `5e-5`, zero weight decay and cosine decay over five
epochs. M3J now adopts its optimizer settings, initial rate and cosine schedule
family and its training precision, extending the duration to 50 epochs.
M3 head training explored the larger
rates `1e-4`, `3e-4` and `1e-3`; the new common rate has not been validated for
the freshly initialized fusion/head components. Use the declared learning curves
to diagnose optimization and interpret outcomes under this fixed recipe.
Any later warm-up, module-specific rate or broader rate search must be recorded
as a matched follow-up with its own fit budget; do not change the initial fits
in response to validation trajectories or replace their outcomes retrospectively.

#### Deliverables and acceptance

Deliverable: `03_joint_finetuning.md`, the online runner integration, joint
checkpoints, matched controls, epoch histories and learning curves available
during training, resource measurements and development report.
Acceptance:

- pixels, geo, labels and sample/plot identities are aligned; development remains
  test-free and initial encoder provenance excludes validation plots;
- for partial-joint image-using models, the classification loss supplies
  gradients to the selected vision parameters and fusion components; only
  declared parameters receive optimization updates. Frozen parameters have
  `requires_grad=False`, are excluded from the optimizer and receive no gradients;
  frozen modules remain in evaluation mode. Compare frozen values directly
  against an independent initial snapshot after the short training check and at
  each fit's end. A measured difference stops the attempt with its parameter and
  magnitude recorded. D038 removes tolerance and repeated in-memory hash gates;
- validation uses the current encoder, where present, in evaluation mode
  without gradients;
  complete encoder + fusion + preprocessing checkpoints load and produce valid
  predictions; bit-for-bit repeated prediction equality is not required;
  geo-only retains a complete head + preprocessing bundle without an encoder;
- training and full-validation performance are recorded every epoch and made
  available during the run, with the monitoring contract below applied equally
  to joint models and matched frozen-online controls;
- final-epoch and best-validation results are both reported, with both complete
  inference checkpoints retained under the dual-checkpoint contract below;
- controls match initialization, augmentation, effective batch size, training
  exposure and declared tuning budgets; all use physical/effective batch 16
  without accumulation and Adam with zero weight decay. All trainable
  components start at `5e-5` without warm-up, and the declared cosine decay
  is shared across components and paired controls;
- all fits complete 50 epochs under the declared FP16-autocast/FP32-weight
  training policy; validation and saved-checkpoint evaluation use autocast-disabled prediction,
  with arithmetic settings and checkpoint/scheduler state preserved;
- results distinguish encoder adaptation, geo benefit and fusion architecture;
  report all declared candidates and the same no-gain/inconclusive outcomes;
- record the selected joint recipe and refit procedure before its M4 application.

#### Learning curves and monitoring

After every epoch, persist training optimization loss and training Acc@1,
Acc@3, weighted F1 and MCC, together with full-validation cross-entropy and the
same four metrics. For image-using models, validation must run through the
current image encoder with fixed validation preprocessing, evaluation mode and
no gradients, using the autocast-disabled prediction policy above; restore
the declared training/frozen-module modes afterward. Aggregate losses by sample
count and compute F1/MCC from whole-epoch predictions or confusion counts,
rather than averaging batch scores; retain the fixed 18-class order. Do not use
stale frozen-image caches or the cleaned test set for these curves.

Make histories available while each run is active, and refresh loss/accuracy
plots from those saved histories. Plot training and validation loss and Acc@1
against epoch; provide validation Acc@3, weighted F1 and MCC curves as well.
Retain each seed's trajectory and show mean and sample SD across paired seeds
in the completed report. Record optimizer steps, actual encoder/head learning
rates, warm-up and unfreezing boundaries, epoch and validation duration, and GPU
memory so changes in performance can be related to the training schedule.
Keep interrupted histories with their archived attempt. Fresh attempts begin a
new epoch-1 history; figures must be reproducible from each saved history.

Label training-pass metrics explicitly: they use training augmentation/dropout
and weights that change within an epoch, so their values are not directly
comparable to fixed-checkpoint validation metrics. If evaluation-mode training
curves are added for a closer comparison, declare their fixed training subset,
transforms and evaluation frequency before fitting, and account for their
additional cost. Keep these diagnostics separate from optimization-pass loss.

Monitoring does not itself change the fitting recipe. M3J's initial budget is
50 epochs with the shared cosine schedule. The user has requested both
final-epoch and best-validation reporting, with the following fixed distinction.

#### Final and best-validation checkpoints

Train each fit to epoch 50 and report **both** outputs from
that same training trajectory:

| Output | Selection rule | Retained evidence |
|---|---|---|
| Final epoch | Checkpoint at epoch 50 | Complete encoder/fusion/head/preprocessing bundle, predictions, metrics and separate final training state |
| Best validation | Highest unrounded validation Acc@1 within that seed's run; exact ties use weighted F1, then MCC, then the earliest epoch | Complete encoder/fusion/head/preprocessing bundle, predictions, all reported metrics, selected epoch and selection provenance |

Best-validation selection uses one checkpoint for all metrics, not a separate
maximum epoch for each metric. Select independently within every seed's run,
then report both policies across all declared seeds; never retain only the best
seed. Use the same epoch eligibility, metric ordering and tie-breaks for every
candidate and matched frozen-online control. Completed training epochs 1 through
50 are eligible; epoch-zero evaluation is not eligible.
The initial recipe has no warm-up and partial-joint updates begin in epoch one.
If a later declared study adds warm-up, keep its completed warm-up epochs
eligible, record whether the best checkpoint precedes unfreezing, and do not
interpret such a checkpoint as a gain from encoder updates.
If final and best coincide, retain both logical
references without requiring duplicate weight files.

Keep separate final-epoch and best-validation tables, seed summaries, paired
comparisons and confusion analyses. Compare final with final and best with best;
do not select whichever policy makes a method look better after observing scores.
Best-validation scores and uncertainty remain conditional on checkpoint and
configuration selection using the reused validation set. This additional
selection must be stated alongside the results.

Saving the best checkpoint does not trigger early stopping: the final-epoch
comparison requires all 50 epochs. The [joint-training record](03_joint_finetuning.md)
resolves configuration ranking, exact primary/supporting contrasts and the
all-development refit rule before comparative fitting. Both checkpoint policies
share the same fixed recipe. All-development refitting starts fresh for 50 epochs
and uses its final checkpoint, since that route has no validation partition.
Any later early-stopping study needs a separate prospective rule.
Completed M3 results remain unchanged.

Under D038, checkpoint selection uses the original per-epoch validation history.
After training, load and evaluate each selected final/best checkpoint; these
evaluations supply the reported scores, logits, predictions and confusion
matrices. Retain the original epoch metrics alongside them as selection
provenance and curve data. A repeated floating-point or prediction difference
does not invalidate a successfully loaded checkpoint or change its selection.

#### Checkpoint storage and later cleanup

The user selected **option 3: shared frozen encoders and shared final/resume
weights** (D033). This is a storage arrangement; it changes neither the training
recipe nor the required checkpoint retention. D038 replaces exact resume/replay
acceptance with practical loading and evaluation checks.

- Store the five immutable starting visual encoders once: one original
  pretrained encoder and four seed-specific CS-adapted encoders. Frozen-online
  bundles reference the appropriate encoder through a versioned manifest.
- Retain each joint fit's complete final and best-validation visual weights
  and each fit's required fusion/head weights. If best and final coincide,
  their logical references may share the same saved weights.
- Store optimizer, scheduler, random-number and other resume state separately
  from model weights. The final inference bundle and final resume state
  reference the same final weights; do not serialize a second copy into the
  resume file. During training retain the latest completed-epoch resume state
  and the current best checkpoint, rather than all 50 epoch checkpoints.
- Record dependencies and their hashes explicitly. Publish checkpoints
  atomically so interruption cannot leave the latest resume reference pointing
  to incomplete weights. A complete inference bundle may use shared files,
  but must resolve all required weights, preprocessing and metadata.
- Provide standalone export that collects the referenced dependencies and can
  load and evaluate without access to the original shared storage. Retain file
  hashes to authenticate source encoders and saved objects. Do not require an
  interrupted/uninterrupted comparison or identical floating-point prediction
  replay. No tolerance configuration or cumulative deviation audit applies.
- Design later cleanup around retained manifests: preview recoverable space,
  remove only unneeded checkpoint references/files, and delete a shared object
  only when no retained bundle or other recorded dependency requires it.
  Preserve metrics, predictions, histories, configuration and provenance records
  and record any removal of weights or exact-resume capability. Initial M3J
  retention remains unchanged; this decision does not authorize deleting
  existing artifacts or automatically pruning models after selection.

For the 132-fit matrix, FP32 tensor accounting gives approximately **315 GB**
of retained model/optimizer data, including five shared initial encoders and
assuming best and final differ. Reserve approximately **400 GB** for this study,
including metadata, plots, verification and temporary checkpoint writes. These
are planning estimates, not measured M3J artifacts, and exclude later M4 refits,
additional experiments and duplicate raw image data. Pin and measure the actual
checkpoint layout during implementation; confirm sufficient free storage before
the full matrix. FP16 autocast does not reduce FP32 checkpoint storage.

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
  M3J uses the shared-file bundle, export and cleanup contract in D033; a logical
  bundle need not duplicate unchanged files to remain reproducible.
- Version scientific changes and retain completed runs. Reuse a saved result
  only after validating its inputs/configuration; do not overwrite evidence to
  make a new recipe appear to reproduce an old one.
- Test meaningful model, split, preprocessing, isolation and replay behavior;
  avoid inheriting elaborate router-specific sealing workflows unnecessarily.

Legacy integration facts addressed by the dedicated M1/M2 development path;
online image–geo training remains M3J work:

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

The governing records, standalone configuration, M1 audit/preparation tools,
M2 model/input/training implementation, M3 comparison/reporting tools and the
separate CLS-bypass implementation exist. The M3J online implementation and
record now exist. D038 is implemented under the separate active identity
`attention_fusion_joint_direct_v1`; 163 relevant CPU checks and practical
GPU verification passed. All 16 seed-1 partial-joint fits are complete, with
final/best evaluations and retained outputs. D039 recovered the thirteenth
fit's checkpoint evaluation without retraining; the operational amendment
passed 134 focused checks. The frozen-online package completed all 16 fits
after its own source-bound verification record, 265 CPU checks and complete GPU
verification. The seed-1 geo-only package completed separately under D041,
with 214 CPU checks, GPU/checkpoint verification and a combined 33-fit report.
Prior failure evidence and both completed image packages remain preserved.

```text
dev_plans/attention_fusion/
  README.md                         # governing document
  00_research_and_design_background.md
  01_experimental_protocol.md        # completed M1
  02_models_and_training.md          # M2 implementation and verification
  03_development_comparison.md       # M3 frozen comparison and results
  03a_cls_bypass_extension.md         # separate M3 bypass study and reproduction
  03_joint_finetuning.md             # M3J online recipe, verification and progress
  03b_seed1_frozen_online.md          # completed frozen-online package and limitations
  03c_seed1_geo_only.md               # shared geo-only fit and completed seed-1 report
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
tools/run_attention_fusion_study.py
tools/analyze_attention_fusion_study.py
multimodal/attention_fusion_comparison.py
multimodal/attention_fusion_reporting.py
tests/test_multimodal_attention_fusion.py
configs/multimodal_attention_fusion_cls_bypass.yaml
multimodal/attention_fusion_bypass.py
multimodal/attention_fusion_bypass_training.py
multimodal/attention_fusion_bypass_protocol.py
multimodal/attention_fusion_bypass_comparison.py
multimodal/attention_fusion_bypass_reporting.py
tools/run_attention_fusion_bypass_study.py
tools/analyze_attention_fusion_bypass_study.py
tools/verify_attention_fusion_bypass.py
tests/test_attention_fusion_bypass*.py
configs/multimodal_attention_fusion_joint.yaml
configs/multimodal_attention_fusion_joint_frozen_online.yaml
configs/multimodal_attention_fusion_joint_geo_only.yaml
multimodal/attention_fusion_joint_{inputs,training,storage,protocol,reporting}.py
tools/run_attention_fusion_joint_study.py
tests/test_attention_fusion_joint_*.py
multimodal_artifacts/attention_fusion/cs/gse_100m/<study_id>/
  protocol/                         # split/cache provenance and resolved recipe
  inputs/seed<N>/                    # M2 development-only cache preparation
  verification/                     # M2 train-only smoke checks; excluded from M3
  development/<regime>/<candidate>/<geo>/lr_<rate>/seed<N>/
  selection/<regime>/<candidate>/<geo>/seed<N>/
  orchestration/m3/                  # matrix accounting and execution identities
  orchestration/m3_bypass/           # separate bypass-study accounting
  final_fit/<regime>/<candidate>/<geo>/seed<N>/  # all-development head refits
  exploratory_test/                 # separate frozen application
  reports/m3_comparison_v1/           # analysis, selected recipes, paired bootstrap
    report/                         # Markdown, CSV tables and PNG/PDF figures
  reports/m3_bypass_comparison_v1/    # separate bypass-study analysis and report
multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/
  inputs/                           # audited online rows and shared geo scaler
  protocol/                         # frozen recipe, contrasts and source snapshot
  verification/                     # short training/inference and checkpoint checks
  objects/                          # shared immutable weight/state objects
  runs/seed1/joint_partial/<candidate>/<initialization>/
  interrupted_attempts/<run_id>/<timestamp>/  # preserved; fresh attempts start at epoch 1
  orchestration/seed1_joint_partial/
  reports/seed1_joint_partial/
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
| Encoder drift, geo shortcuts or costly overfitting | Use the declared historical partial-unfreezing scope, bounded compute, shared `5e-5` cosine schedule, learning-curve monitoring and matched image-only controls |
| Saved features no longer match trainable encoder weights | Recompute validation outputs and bind any subsequent caches to the exact checkpoint |
| One-key attention appears to be adaptive | Exclude that design; verify multiple keys within each sample |
| A global vector lacks useful local detail | Report the limitation; consider patch inputs only as a later study |
| Overfitting a small number of independent plots | Keep the model/search small, group splits, and report seed/plot uncertainty |
| Attention weights are presented as reliability explanations | Use prediction-based rescue/harm analysis; label weights descriptive |
| Reused validation/test appears to confirm a new method | State the reuse and reserve confirmatory claims for M6 |
| Unresolved memory/read inconsistencies | Preserve prior failures and label results provisional. D038 retains direct frozen-value comparisons and authenticated saved files. D039 permits at most two per-file cache-refresh retries before an exact expected-digest match is still required; an unresolved output-checkpoint failure is reported separately while independent fits continue. Passing these checks does not establish machine health or definitive comparative conclusions |

## Progress ledger

| Milestone | Status | Evidence | Next action |
|---|---|---|---|
| M0 Governing design | Complete | This guide and [design background](00_research_and_design_background.md) | Maintain decisions |
| M1 Protocol and input audit | Complete | [Protocol and audit record](01_experimental_protocol.md), standalone config and generated inventory/matrix; A001/A002 amendments | Maintain frozen recipe and change history |
| M2 Implementation and checks | Complete | [Implementation and verification record](02_models_and_training.md); four caches, seven models, 140 tests, 19-arm smoke and exact GPU resume/replay | Preserve verified inputs and interfaces for the M3 extension and M3J |
| M3 Original development comparison | Complete | [Development comparison record](03_development_comparison.md); all 228 fits, 76 selected references, 63 paired contrasts; 147 focused tests; exact checkpoint and analysis replay; inspected report | Preserve original evidence and selection |
| M3 CLS-bypass extension | Complete | [Extension record](03a_cls_bypass_extension.md); 72 fits, 24 selected references, 37 contrasts, 203 tests, exact checkpoint/analysis replay and inspected report; both primary results inconclusive | Preserve evidence; no bypass promotion; plan M3J separately |
| M3J Joint fine-tuning | Active fit execution complete: 84/84; cross-seed consolidation and selection pending; 48 frozen-online fits deferred | Seed 1: 33 fits; seeds 2–4: 17 active fits each; [seed-4 completion](03m_seed4_geo_only.md): 09:32:25 GMT on 2026-10-28, combined report and completion audit; 4,200 active training epochs and 168 final/best checkpoint-policy evaluations; all nine active queues complete | Consolidate four-seed partial-joint/geo-only results, declared contrasts, uncertainty, interpretation and selected-model/refit record before any M4 application; preserve [D047](03h_frozen_online_deferral.md) deferral |
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
| [03 Development comparison](03_development_comparison.md) | M3 results and selection | Complete |
| [03a CLS bypass extension](03a_cls_bypass_extension.md) | M3 adapted-token/original-embedding bypass extension, comparisons and reproduction | Complete |
| [03_joint_finetuning.md](03_joint_finetuning.md) | M3J online implementation, partial unfreezing, matched controls and results | Seed-1 partial-joint package complete: 16/16 fits, dual-checkpoint results and completion evidence recorded; D038/D039 verification and incident evidence retained; M3J overall in progress |
| [03b Seed-1 frozen-online package](03b_seed1_frozen_online.md) | Frozen-online controls, unchanged scientific recipe, execution evidence and comparison limitations | Complete: 16/16 fits at 00:49:39 UTC on 2026-10-22, dual-checkpoint evaluations and no unresolved fits |
| [03c Seed-1 geo-only package](03c_seed1_geo_only.md) | Shared standardized-geo reference and combined seed-1 completion record; no image encoder | Complete: 50 epochs, final/best evaluations, 214 CPU checks and GPU/checkpoint verification; combined 33-fit report completed at 10:18:23 UTC on 2026-10-22 |
| [03d Seed-1 performance findings](03d_seed1_findings.md) | Interpretation of initialization, concatenation versus attention, attention readouts and unimodal versus fusion results | Recorded: final/best results and trade-offs across all 33 fits; provisional single-seed conclusions and comparison limitations retained |
| [03e Seed-1 geo-rescue opportunity](03e_seed1_geo_rescue_opportunity.md) | Validation complementarity between M3J fusion and the shared geo expert, focusing on CLS + original embeddings | Diagnostic complete: potential rescues/harms, oracle bounds, class/case evidence and reproducible analysis; geometry routing not yet applied |
| [03f Seeds 2–4 preparation](03f_later_seed_preparation.md) | Nine isolated package configurations, CPU preparation, same-seed references, prospective shuffle/crop correction and retained practical recovery | Historical D045 preparation complete; D047 supersedes the later-seed execution order and geo-only prerequisites |
| [03g Seed-2 partial-joint execution](03g_seed2_partial_joint.md) | Authorized 16-fit queue, preserved recipe/recovery, launch and completion records | Complete at 05:06:56 UTC on 2026-10-25: 16/16 fits, 800 epochs, 32 final/best evaluations, no unresolved checkpoint or queue error |
| [03h Frozen-online deferral](03h_frozen_online_deferral.md) | D047 scheduling amendment, later-seed geo dependencies, explicit deferred reporting and preparation evidence | Complete: 363 CPU tests and all three refreshed geo preparations passed; seed-2 reference binding passed; 2,202 completed-package files preserved; no comparative training launched by D047; its next geo-only fit subsequently completed under D048 |
| [03i Seed-2 geo-only completion](03i_seed2_geo_only.md) | One shared standardized-geo reference, execution evidence and combined active-scope report | Complete at 09:25:06 UTC on 2026-10-25: 50 geo-only epochs, final/best evaluations and 17/17 complete active seed-2 fits; 16 frozen-online fits explicitly deferred |
| [03j Seed-3 partial-joint execution](03j_seed3_partial_joint.md) | Authorized 16-fit queue, unchanged recipe/recovery, package verification and completion records | Complete at 16:23:18 GMT on 2026-10-26: 16/16 fits, 800 epochs and 32 final/best evaluations, no queue error or unresolved checkpoint; launch and interim progress history preserved |
| [03k Seed-3 geo-only completion](03k_seed3_geo_only.md) | One shared standardized-geo reference, execution evidence and combined active-scope report | Complete at 16:54:30 GMT on 2026-10-26: 50 geo-only epochs, final/best evaluations and 17/17 complete active seed-3 fits; 16 frozen-online fits explicitly deferred |
| [03l Seed-4 partial-joint execution](03l_seed4_partial_joint.md) | Authorized 16-fit queue, unchanged recipe/recovery, package verification, launch and completion record | Complete at 05:41:20 GMT on 2026-10-28: 16/16 fits, 800 epochs, 32 final/best evaluations and no queue error or unresolved checkpoint |
| [03m Seed-4 geo-only completion](03m_seed4_geo_only.md) | One shared standardized-geo reference, execution evidence and combined active-scope report | Complete at 09:32:25 GMT on 2026-10-28: 50 geo-only epochs, final/best evaluations, 17/17 complete active seed-4 fits; M3J active fit execution is 84/84 complete, with 48 frozen-online fits deferred |
| [Geometry-router side-study guide](../attention_fusion_router/README.md) | Separate R1–R5 roadmap for applying geometry routing to fixed M3J experts | R1–R4 complete; [R4 findings and handoff](../attention_fusion_router/04_findings_and_handoff.md), eight validation analyses and 239 focused checks. Return to main M3J; cleaned-test R5 deferred |
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
| D018 | 2026-10-16 | Original M3 recorded outcome | Native-geo two-token attention shows no improvement over projected concatenation under the declared rule (−0.18 pp Acc@1; conditional 95% interval [−0.81, +0.44] pp). No attention default promotion. The original search and result remain unchanged; any extension is recorded separately. Secondary standardized-geo concatenation gains remain development evidence; M3J proceeds independently of the frozen outcome. |
| D019 | 2026-10-16 | User-requested plan; implementation awaits cue | Add both CLS bypass variants within a separately versioned M3 extension: updated CLS plus pre-attention typed adapted tokens (384D), and updated CLS plus original-width preprocessed embeddings (1,344D). Keep the two-layer CLS backbone and frozen encoders; plan 72 fits across the existing geo/LR/seed grid. Preserve completed M3 evidence and distinguish readout/feature access from added classifier capacity. Update only this guide now. |
| D020 | 2026-10-16 | User authorized implementation | Proceed with the M3 bypass extension in D019, including both variants, the 72-fit matrix, frozen contrasts, retained checkpoints and reproducibility evidence. Preserve original M3 code/artifacts and the common initialization namespace; M3J and test application remain separate. |
| D021 | 2026-10-16 | M3 extension recorded outcome | Completed all 72 bypass fits. Both native comparisons versus CLS-only are inconclusive: adapted +0.06 pp Acc@1 (95% interval [−0.70, +0.81]); original −0.09 pp ([−0.88, +0.67]). Zero of 37 extension contrasts support improvement; standardized raw concatenation retains the highest observed mean. No bypass promotion or change to the original M3 outcome. All 24 selected GPU replays and independent analysis reconstruction pass; original artifacts remain unchanged. |
| D022 | 2026-10-16 | User-requested M3J monitoring | Require epoch-level training and full-validation histories and curves available during training, with encoder/head rates and warm-up/unfreezing markers. Distinguish training-pass metrics from evaluation-mode diagnostics. Declare stopping/checkpoint selection before fitting; monitoring alone does not authorize changing the selection rule. Documentation only; M3J remains planned. |
| D023 | 2026-10-16 | User-confirmed dual M3J reporting | Report and preserve both final-epoch and best-validation checkpoints from every fit. Best validation is selected per seed by Acc@1, weighted F1, MCC, then earliest epoch; report all metrics from that checkpoint and retain all seeds. Complete the declared training budget and compare like checkpoint policies across controls. M3J configuration ranking, primary contrasts and refit duration must be declared before fitting; no retrospective change to completed M3. |
| D024 | 2026-10-17 | User-confirmed M3J initialization and image preprocessing | Compare pinned original SigLIP2 with the audited training-partition CS-adapted towers under the same downstream recipe. Preserve historical BGR/439 decoding, training random-resized crop, deterministic validation resize and 0.5 channel normalization in both arms. Keep prior adaptation exposure visible. Numerical optimization and unfreezing choices remain to be frozen. |
| D025 | 2026-10-17 | User-expanded M3J model scope | Include all nine architectures from M3 and its extension, including capacity control, CLS-only and both CLS bypasses. Eight image-using models span both initializations and frozen-online/partial-joint regimes; geo-only is one shared reference per seed. One recipe per architecture gives 132 unique fits, superseding the earlier four-model/96-fit planning cap for M3J. Additional recipes require explicit accounting before fitting; retain the existing 50-epoch ceiling and original M1/M3 files. |
| D026 | 2026-10-17 | User-confirmed M3J geo preprocessing | Start M3J with training-fitted coordinate-standardized geo for every geo-consuming model, including both branches of the original-embedding bypass. Share the training-only scaler across initializations, regimes and seeds. Native/L2 comparisons are possible later amendments; completed M3's native primary and all historical results remain unchanged. |
| D027 | 2026-10-17 | User-proposed M3J simplification adopted for initial comparison | Use uniform initial LR `5e-5` for all trainable encoder, adapter, fusion and classifier components across architectures and initializations. No head-only or LR warm-up; partial-joint updates start immediately, while frozen-online encoders remain fixed. Defer warm-up and module-specific rates to separately declared follow-ups. Optimizer and constant-versus-decayed schedule remain open. Existing M3 geo-only rates differ, so four new geo-only fits are required within the 132-fit matrix. |
| D028 | 2026-10-19 | User-confirmed M3J cosine decay | Resolve D027's schedule choice as a shared cosine decay from initial `5e-5` toward zero for every trainable group and matched model/regime/initialization, including geo-only. Preserve historical end-of-epoch stepping and zero minimum, with no warm-up or restarts; use the declared M3J epoch budget as the cosine period, not the historical five epochs. The post-final-epoch scheduler state reaches zero. Optimizer/weight decay and actual duration remain undecided. Documentation only; no fits started. |
| D029 | 2026-10-19 | User-confirmed M3J partial-unfreezing scope | Reuse the historical final 10 Transformer blocks, final normalization and attention-pooling head: blocks 17–26, `trunk.norm` and `trunk.attn_pool` in the pinned visual tower. This corresponds to 11 framework groups, not 11 blocks. Verify exact trainable names/counts before fitting; share the subset across both initializations and all image-using partial-joint models. Other visual parameters/text remain frozen, and frozen-online controls freeze the whole encoder. |
| D030 | 2026-10-19 | Physical batch confirmed; accumulation clarification pending | Record physical training batch 16. Historical adaptation used no accumulation (effective 16); the earlier physical-16/eight-step proposal would give effective 128. The user requested an explanation of that distinction; no accumulation choice has yet been confirmed. Keep effective batch open and do not silently inherit eight steps. No training started. |
| D031 | 2026-10-19 | User-confirmed historical optimization and no accumulation | Resolve D030 as physical/effective batch 16, one optimizer update per minibatch, shuffled rows and final smaller batch retained. Keep historical optimization: Adam, shared initial LR `5e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.0`, AMSGrad disabled and no gradient clipping. Use D028's shared cosine policy, without warm-up or restarts, across all trainable components and matched comparisons including geo-only. Supersede the earlier AdamW/weight-decay proposal; epoch budget and numerical precision remain separate pending choices. No training started. |
| D032 | 2026-10-19 | User-confirmed M3J duration and historical precision | Fix all initial M3J fits at 50 epochs, cosine `T_max=50`, without early stopping; retain final epoch 50 and best validation from epochs 1–50. Keep FP32 weights/Adam state with CUDA FP16-autocast training and no GradScaler; preserve historical cuDNN TF32 on, matmul TF32 off and `highest` float32 matmul precision. Use autocast-disabled FP32-input/weight prediction for full validation/replay, following historical extraction. Apply consistently across all models/regimes/initializations, including geo-only. Supersede the earlier full-FP32 M3J proposal; completed M3 remains unchanged. No implementation or training started. |
| D033 | 2026-10-19 | User-confirmed shared checkpoint storage, option 3 | Share five immutable starting encoders across frozen-online fits and final weight files between inference and resume. Retain final/best joint weights, separate optimizer/scheduler/RNG state, verified dependency manifests and standalone export. Require atomic writes, replay/resume verification and dependency-aware future cleanup; preserve initial checkpoint retention and existing artifacts. Estimated model/state storage is 315 GB with a 400 GB planning allowance, excluding M4 and extra experiments. Documentation only; no storage implementation or comparative training started. |
| D034 | 2026-10-19 | Adopted user-proposed seed-by-seed sub-tasks; first set awaits review | For each seed, run 16 partial-joint fits, then 16 frozen-online fits, then one geo-only fit before moving to the next seed. Use seeds 1–4 in order; retain all 132 fits and the fixed recipe. Each fit resets its own starting weights/optimizer/RNG, with paired comparisons independent of execution order. Present the 16 seed-1 partial-joint fits for the user's requested review before starting. Shared implementation and training-only verification precede comparative fits; no implementation or training launched by this decision. |
| D035 | 2026-10-19 | User authorized the first M3J package | Implement and execute the 16 seed-1 partial-joint fits under D024–D034. Freeze executable configuration, exact contrasts and future refit rule before validation scoring; verify finite gradients, visual scope, exact resume and portable export first. Record final and best validation separately, with provisional single-seed interpretation. Other regimes/seeds and M4 remain separate packages. |
| D036 | 2026-10-19 | Historical integrity block; active prerequisite superseded by D038 | Preserve the implemented recipe and all failed preflight evidence. Exact resume found one changed frozen scalar; the hardened verification later detected a changed frozen digest during one step. Two executions of the CPU-only memory probe also reported single-bit errors. The prior policy required diagnosis and its complete gate before launch. No reboot or system-setting change was performed; no particular faulty component is established. D038 now governs this package without erasing these incidents. |
| D037 | 2026-10-19 | Historical user-authorized tolerance; superseded by D038 | The user selected “Allow tiny frozen-weight changes in an explicit exploratory mode.” The separate `attention_fusion_joint_exploratory_v1` introduced bounded deviations and cumulative limits with provisional labels, leaving the original strict study unchanged. Implementation completed and 201 focused tests passed; its GPU verification failed on inconsistent checks of the same frozen snapshot, 0/16 full fits started. Preserve that configuration, receipts and diagnostic evidence as history; its tolerance/resume/replay gates no longer govern the active package. |
| D038 | 2026-10-19 | User-approved practical verification | Use `attention_fusion_joint_direct_v1`, `verification_policy: direct_freeze_checks_v1` and `provisional_unresolved_memory_integrity`. Remove mandatory interrupted/uninterrupted comparison, automatic resume, live-weight checksum gates, bespoke tolerance and cumulative deviation tracking. Verify declared freeze scope, optimizer exclusion/no gradients, direct frozen-value equality after short training checks and at fit end, finite training, valid identities and usable saved checkpoints. Retain file hashes and option 3 shared storage. Archive interrupted attempts before any fresh epoch-1 training. Select best from original epoch history; load/evaluate final and best for reported metrics without exact replay gates. Preserve the fixed recipe, paired RNG and prior incident evidence. Historical launch snapshot: implementation complete, 163 CPU checks and complete GPU verification passed, 1/16 started and 0 completed. D039 adds evaluation-only continuation without automatically retraining incomplete attempts. |
| D039 | 2026-10-20 | User-approved resilient checkpoint reads and continuation | Preserve the 12 completed fits and all 50 epochs of fit 13. Permit at most two read-only per-file cache-refresh retries after a saved-file digest mismatch; the original expected digest must still match and failures/recovery are recorded. Recover final/best evaluation from the existing authenticated checkpoints and original epoch selection, without optimizer updates or training replay. Continue the three independent unstarted fits if an output checkpoint remains unverifiable, recording the affected fit as unresolved and excluding it from completed comparisons. Do not automatically resume or retrain partial attempts. Preserve the original protocol/verification receipt and record the changed operational source identity as an explicit amendment with unchanged scientific recipe, configuration, inputs and paired randomness. Keep provisional unresolved-memory labels and original failure evidence. Implementation complete; 134 focused checks passed and the numerical training-core AST matches the original snapshot. At the historical recovery snapshot, continuation had launched at 15:48 UTC (16:48 BST) and evaluation recovered at 15:51 UTC with zero additional epochs, 13 fits complete and the fourteenth active. All 152 preservation checks passed. The package subsequently completed all 16 fits at 21:39 UTC; see the completion record. |
| D040 | 2026-10-20 | User authorized seed-1 frozen-online execution | Complete the regime-specific integration and training-only verification, then run the 16 frozen-online fits with the unchanged M3J recipe. Use a separate frozen-online configuration, protocol/source snapshot and artifact root; preserve the completed partial-joint package and pair its inputs, initializations and random streams. After diagnosis of a changed captured reference, anchor frozen comparisons and eligible shared visual checkpoints to the authenticated original encoder; preserve all failed evidence and require unchanged direct comparisons. Retain D038/D039 practical frozen-value checks, saved-file recovery, dual checkpoints and provisional assurance; do not reinstate interrupted-run comparison, exact replay or tolerance gates. Geo-only and later seeds remain separate packages. |
| D041 | 2026-10-22 | User authorized the remaining seed-1 geo-only fit | Implement, verify and run one shared geo-only reference under the unchanged M3J recipe: standardized 64D geo, the existing 256-unit ReLU classifier, Adam `5e-5`, cosine over 50 epochs, batch 16, FP16-autocast training and FP32 validation. Use a separate configuration, source snapshot and `attention_fusion_joint_geo_only_direct_v1` root; preserve both completed 16-fit image packages, dual-checkpoint reporting, retained reusable checkpoints, D038/D039 saved-file handling and provisional assurance. Geo-only has no visual dependency or freeze comparison. Do not reinstate resume, exact replay or tolerance gates. The loader audit found that completed image packages derived crop/shuffle streams from their distinct study IDs, so exact cross-regime pairing was not realized; document this limitation without changing or rerunning those fits. The new geo-only loader uses the declared `attention_fusion_joint_v1` shuffle namespace; image augmentation is inapplicable. Completed: 214 CPU checks and GPU/checkpoint verification passed; all 50 geo-only epochs and final/best evaluations finished, combined seed-1 report contains 33 complete fits, and all 1,237 tracked preceding-package files were preserved. |
| D042 | 2026-10-22 | User authorized router side-study R1/R2 | Prepare a separately governed validation fork using the seed-1 pretrained-initialized partial-joint CLS-original model only. Pair final fusion/geo epochs 50/50 and best epochs 36/11. Compare original-pretrained geometry with the matching after-training vision tower, yielding three encoders and four score sets. R1/R2 prepare and verify inputs, embeddings and geometry scores; threshold selection and routed evaluation remain R3. The historical CS-adapted geometry tower and frozen-online arms are excluded. After R4 return to remaining M3J/main-study work; revisit cleaned-test routing in a later R5 protocol after main-study completion. Completed: 12,600 image vectors, four 822-row candidate score sets, 90 focused checks and independent numerical audit passed; all 17 declared source files unchanged, provisional assurance retained. |
| D043 | 2026-10-23 | User authorized router side-study R3 | Apply the frozen fixed-zero and constrained selected-threshold rules to all four R2 cells. Preserve the R1/R2 config and artifacts through a separate R3 stage configuration, source snapshot and output directory; no new model training or test use. Completed all eight evaluations, full-vector predictions, threshold candidate records, habitat accounting and confusion figures; 177 focused checks passed. Original-pretrained geometry's selected rules yield final/best Acc@1 74.09%/75.30% (+2.92/+3.53 points); matching trained geometry has no feasible selected threshold and retains fusion. Keep validation-reuse and provisional-memory limitations; R4 remains the next separate interpretation/handoff package. |
| D044 | 2026-10-23 | User authorized router side-study R4 | Complete interpretation of all eight fixed R3 policies: habitat recall/F1 changes, plot-level outcomes and harm concentration, missed-rescue partitions, paired geometry diagnostics and deterministic numerical cases. Report 10,000-replicate whole-plot conditional Acc@1 intervals with fixed experts/actions/thresholds and explicit validation-reuse limits. All 239 focused checks passed; publish a separate reproducible analysis package and findings/handoff report. No model or threshold changes, training, or cleaned-test evaluation. Return to main M3J at 33/132 fits, with seed-2 partial-joint as the next package; later R5 must explicitly specify finalized experts, geometry encoder, reference bank and threshold transfer before test application. |
| D045 | 2026-10-23 | User authorized seeds 2–4 preparation only | Prepare nine separate seed/regime configurations and CPU input records, generalize package scope and same-seed predecessor references, and retain D038/D039 practical checks/recovery. Implement the declared shuffle/crop namespaces prospectively with `loader_rng_policy: declared_namespaces_v1`; preserve seed-1 defaults, artifacts and limitations. Reports derive pairing from saved configuration and mark unknown evidence explicitly. The 50-epoch recipe and 132-fit matrix are unchanged. All nine real-input preparations and 311 focused tests passed. Seed-2 partial-joint GPU verification passed all 16 model checks plus fresh checkpoint loading/export at 20:55:24 UTC; no comparative fits launched, no driver/system changes or reboot. Execute partial-joint → frozen-online → geo-only within each seed when started. |
| D046 | 2026-10-23 | User authorized launching the 16 seed-2 partial-joint fits | Launch only seed 2 and `joint_partial`, eight architectures × original-pretrained/seed-2 CS-adapted initialization, using the passed D045 configuration/source receipt. Preserve the 50-epoch recipe, standardized geo, declared random streams, dual checkpoints and D038/D039 recovery. Background process 30319 launched at 22:17:43 UTC (23:17 BST); 0/16 fits complete at launch. Frozen-online, geo-only and later seeds remain separate packages. No code/configuration or system changes during launch. |
| D047 | 2026-10-25 | User deferred frozen-online for seeds 2–4 and authorized the corresponding file updates | Preserve the full 132-fit design while deferring 48 frozen-online fits: 84 fits remain in the current scope, with 49 complete after the 16 seed-2 partial-joint fits finished. Next is seed-2 geo-only, then partial-joint → geo-only for seeds 3 and 4. Later-seed geo configurations require only the completed same-seed partial-joint package and explicitly declare frozen-online deferred. Combined reports distinguish 17 active from 16 deferred fits per later seed; no absent-regime scores or four-seed trainability conclusion. Preserve seed-1 behavior, completed artifacts, the numerical recipe, D038/D039 checks/recovery and original preparation evidence. This amendment prepares the workflow; it starts no comparative fits. |
| D048 | 2026-10-25 | User authorized seed-2 geo-only execution; completed | Run one shared standardized-geo classifier under the unchanged M3J recipe after package-specific training-only GPU/checkpoint verification. Preserve D047's 17-active/16-deferred scope, declared random streams, final/best storage and D038/D039 recovery. Verification passed at 09:06:25 UTC; process 62013 launched at 09:23:47 UTC; all 50 epochs and final/best evaluations finished at 09:24:33 UTC, with the combined report/queue complete at 09:25:06 UTC. Seed 2 has 17 complete active fits, 850 epochs and 34 checkpoint-policy evaluations, no queue error or unresolved checkpoint. M3J is 50/84 active fits; seed-3 partial-joint is next. No frozen-online, later-seed or cleaned-test execution belongs to this package. |
| D049 | 2026-10-25 | User authorized the 16 seed-3 partial-joint fits; completed | Execute seed 3 and `joint_partial` only, eight architectures × original-pretrained/seed-3 CS-adapted initialization. Refresh preparation against unchanged execution sources; retain the existing 363-check CPU evidence, 50-epoch recipe, standardized geo, D045 random streams, dual checkpoints and D038/D039 practical recovery. Package-specific GPU verification passed all 16 model checks plus fresh training-only checkpoint loading/export at 09:39:28 UTC; no comparative validation was scored during verification. Background process 65206 launched at 09:40:33 UTC (GMT); 0/16 fits complete at launch, overall 50/84 active fits complete. Preserve completed artifacts, D047's 48 deferred frozen-online fits and provisional assurance. The queue completed all 16 fits at 16:23:18 GMT on 2026-10-26: 800 epochs, 32 final/best evaluations, no queue error or unresolved checkpoint, bringing M3J to 66/84 active fits. Seed-3 geo-only is subsequently authorized as D050; seed 4, router work and cleaned-test evaluation remain separate. No runtime/configuration or system changes. |
| D050 | 2026-10-26 | User authorized seed-3 geo-only execution; completed | Run one shared standardized-geo classifier under the unchanged M3J recipe, preserving D047's 17-active/16-deferred scope and D038/D039 practical recovery. Refresh preparation and same-seed completed-reference binding; reuse the existing 363-check CPU evidence for unchanged execution sources. Training-only GPU/checkpoint verification passed at 16:52:30 GMT; process 134722 launched at 16:53:11 GMT. All 50 epochs and final/best evaluations finished at 16:53:57 GMT, with the combined report/queue complete at 16:54:30 GMT. Seed 3 has 17 complete active fits, 850 epochs and 34 evaluations, no queue error or unresolved checkpoint. M3J is 67/84 active fits; seed-4 partial-joint is the next separate package. No seed-4, deferred frozen-online, router or cleaned-test execution belongs to this package. |
| D051 | 2026-10-26 | User authorized the 16 seed-4 partial-joint fits; completed | Execute seed 4 and `joint_partial` only, eight architectures × original-pretrained/seed-4 CS-adapted initialization. Refreshed preparation passed at 17:38:57 GMT; unchanged 27 execution sources retain the existing 363-check CPU evidence. Package-specific training-only GPU verification passed all 16 model checks plus fresh checkpoint loading/export at 17:46:18 GMT, without comparative validation scoring. Process 173284 launched at 22:47:57 GMT; the first fit reports running at 22:49:39 GMT, with 0/16 fits complete, no queue error or unresolved checkpoint at startup, and overall 67/84 active fits complete. Preserve the 50-epoch recipe, standardized geo, D045 streams, dual checkpoints, D038/D039 practical recovery, prior artifacts and provisional assurance. The queue completed all 16 fits at 05:41:20 GMT on 2026-10-28: 800 epochs, 32 final/best evaluations, no queue error or unresolved checkpoint, bringing M3J to 83/84 active fits. Seed-4 geo-only follows separately under D052; 48 frozen-online fits remain deferred, and router work and cleaned-test application remain separate. No runtime/configuration or system changes. |
| D052 | 2026-10-28 | User authorized seed-4 geo-only execution; completed | Run one shared standardized-geo classifier under the unchanged M3J recipe after the completed same-seed partial-joint reference. Preparation passed at 09:30:08 GMT, reusing D047’s 363 CPU checks for unchanged 27 execution sources. Package-specific training-only GPU/checkpoint verification passed at 09:30:40 GMT; process 316572 launched fresh at 09:31:05 GMT. All 50 epochs and final/best evaluations completed at 09:31:51 GMT (best epoch 9); the combined report/queue completed at 09:32:25 GMT without error or unresolved checkpoint. Seed 4 is 17/17 active fits, 850 epochs and 34 evaluations; all 3,894 pre-existing completed-package files were preserved. M3J active fit execution is 84/84 complete, with 48 frozen-online fits deferred; cross-seed consolidation, interpretation and selection remain pending. Preserve dual checkpoints, D038/D039 recovery and provisional assurance. No M4, router, cleaned-test or deferred training is included. |

## Immediate next action

**All 84 currently active M3J fits are complete.**
The [seed-4 geo-only completion record](03m_seed4_geo_only.md) closes the last
active fit: 50 epochs and final/best saved-checkpoint evaluation finished at
**09:31:51 GMT on 2026-10-28**, and the combined report/queue completed at
**09:32:25 GMT**, without error or unresolved checkpoint. Seed 1 has **33/33
fits**; seeds 2–4 each have **17/17 active fits**. Across the nine completed
active packages, this totals **4,200 training epochs** and **168 final/best
checkpoint-policy evaluations**. The **48 frozen-online fits** for seeds 2–4
remain deferred in the full 132-fit design.

The next development work is **cross-seed consolidation and selection**.
Consolidate all four partial-joint seeds and shared geo-only references under
the declared final/best policies, paired contrasts, uncertainty and
joint-selection/refit contract. Report initialization effects, geo benefit and
fusion-architecture comparisons, retaining no-gain and inconclusive outcomes.
Keep seed-1 frozen-online findings separate and explicitly qualified;
four-seed encoder-trainability comparisons remain deferred with the missing
frozen-online fits. Execution completion alone does not complete the scientific
M3J development report or authorize M4 refits, cleaned-test evaluation, router
work or deferred training.

Preserve the completed seed-1/2/3/4 package roots, their configuration/source
snapshots, verification receipts, histories and final/best outputs. The
completed [original M3 comparison](03_development_comparison.md) and
[CLS-bypass extension](03a_cls_bypass_extension.md) also retain their artifacts.

Keep the interpretation limits visible. Results are descriptive on the
822-image development validation set and retain provisional unresolved-memory
assurance. The D041 audit found that completed seed-1 image packages used
distinct study IDs for shuffle/crop streams despite the declared common
namespaces. Their regime contrast includes differences in realized views and
minibatch orders. D045 applies the declared common namespaces prospectively
for seeds 2–4, with the seed retained as a stream input; disclose this distinction
in later pooled results and do not rewrite completed fits. The geo-only path
uses the declared shuffle namespace and no image tensors or encoder; shared
allocation checks retain image-file existence metadata.

Carry forward D038/D039: no mandatory interrupted/uninterrupted comparison,
exact numerical replay, tolerance machinery or live-weight checksum gate.
Retain meaningful training/frozen-value checks where applicable and bounded
saved-file recovery. No reboot or system-setting change is authorized.

## Change log

| Date | Change |
|---|---|
| 2026-10-28 | Closed D051 at 16/16 seed-4 partial-joint fits, 800 epochs and 32 final/best evaluations at 05:41:20 GMT. D052 completed the shared seed-4 geo-only fit: 50 epochs, final/best checkpoints, best epoch 9; combined report/queue complete at 09:32:25 GMT without error or unresolved checkpoint. Refreshed preparation and package-specific training-only GPU/checkpoint verification passed; unchanged sources reuse 363-check CPU evidence, and all 3,894 preceding completed-package files were preserved. Seed 4 is 17/17 active fits; all nine active packages total 84/84 fits, 4,200 epochs and 168 evaluations. Cross-seed consolidation and selection remain pending; 48 frozen-online fits stay deferred. No scientific-recipe changes or M4/router/cleaned-test execution. |
| 2026-10-26 | D051 launched the authorized 16 seed-4 partial-joint fits at 22:47:57 GMT, PID 173284. Refreshed preparation passed at 17:38:57 GMT and package-specific training-only GPU verification passed all 16 model checks plus fresh checkpoint loading/export at 17:46:18 GMT. Unchanged execution sources retain the existing 363-check CPU evidence; all 3,291 preceding completed-package files were preserved by presence/size/mtime checks. The first fit reports running at 22:49:39 GMT with 0/50 completed epochs, no queue error or unresolved checkpoint; recorded 0/16 complete at startup, overall 67/84 active fits complete and 48 frozen-online fits deferred. Retained the scientific recipe, D045 streams, D038/D039 recovery and provisional assurance; seed-4 geo-only remains a separate subsequent package. |
| 2026-10-26 | Closed D049 at 16/16 seed-3 partial-joint fits, 800 epochs and 32 final/best evaluations at 16:23:18 GMT. D050 completed the shared geo-only fit (50 epochs, final/best, best epoch 14), with its combined report/queue complete at 16:54:30 GMT and no error or unresolved checkpoint. Seed 3 is 17/17 active fits, 850 epochs and 34 evaluations; M3J is 67/84 active fits, with 17 active seed-4 fits remaining and 48 frozen-online fits deferred. Refreshed preparation and package-specific training-only GPU/checkpoint verification passed; scientific settings and completed prior results remain unchanged. Seed 4 is the next separate package. |
| 2026-10-26 | Recorded the 09:04 GMT D049 progress snapshot: 12/16 seed-3 partial-joint fits complete, each with 50 epochs and final/best evaluation (600 completed-fit epochs, 24 evaluations); fit 13, pretrained CLS + adapted-token bypass, is running at 10/50 completed epochs. PID 65206 remains active with no queue error or unresolved checkpoint. M3J is 62/84 active fits complete, with 22 active fits remaining and 48 frozen-online fits deferred. Preserved original launch history; no runtime/configuration or training-recipe changes. |
| 2026-10-25 | D049 launched the authorized 16 seed-3 partial-joint fits at 09:40:33 UTC (GMT), PID 65206. Refreshed preparation preserved historical evidence; unchanged execution sources retain D047's 363-check CPU evidence, and package-specific GPU verification passed all 16 model checks plus fresh training-only checkpoint loading/export at 09:39:28 UTC. Recorded the persistent log and handoff under the unchanged recipe, D045 streams and D038/D039 recovery; 0/16 complete at launch, overall 50/84 active fits complete. Seed-3 geo-only is separate, 48 frozen-online fits remain deferred, and prior results and system settings are unchanged. |
| 2026-10-25 | D048 completed the one seed-2 geo-only fit: 50 epochs and final/best checkpoint evaluations, with best epoch 19; package-specific GPU verification passed and the combined report/queue completed at 09:25:06 UTC without error or unresolved checkpoint. Seed 2 has 17/17 active fits, 850 epochs and 34 evaluations; 16 frozen-online fits remain deferred. M3J is 50/84 active fits, with seed-3 partial-joint next. Scientific settings and prior results are unchanged; see the seed-2 geo-only record. |
| 2026-10-25 | Closed D046 at 16/16 seed-2 partial-joint fits, 800 epochs and 32 saved-checkpoint evaluations, with no queue error or unresolved checkpoint. D047 defers seed-2/3/4 frozen-online (48 fits), makes seed-2 geo-only next, and updates later-seed geo prerequisites and combined reports to distinguish 17 active from 16 deferred fits. Preserved completed results and recipe; no comparative training launched by this amendment. See the deferral record for verification and preparation evidence. |
| 2026-10-23 | D046 launched the authorized 16 seed-2 partial-joint fits at 22:17:43 UTC (23:17 BST), PID 30319. Confirmed passed configuration/source verification and idle host GPU; retained the D045 recipe and D038/D039 recovery unchanged. Recorded the persistent log and execution handoff; 0/16 complete at launch. |
| 2026-10-23 | D045 prepares the nine seed-2/3/4 partial-joint, frozen-online and geo-only configurations with isolated roots and same-seed predecessor references. Added CPU-only `prepare`, prospective declared shuffle/crop streams with legacy seed-1 preservation, and evidence-based pairing labels in reports. All nine real-input preparations, 311 focused tests, and seed-2 partial-joint GPU verification passed (16 model checks plus checkpoint reload/export). No comparative training or system changes. See the later-seed preparation record. |
| 2026-10-22 | Completed seed 1: 16 partial-joint + 16 frozen-online + one shared geo-only fit, 1,650 epochs and 66 final/best checkpoint-policy evaluations. Frozen-online finished at 00:49:39 UTC (01:49 BST). D041 geo-only passed 214 CPU checks and GPU/checkpoint verification; its 50 epochs and saved-checkpoint evaluations finished at 10:17:19 UTC, and the combined 33-fit report/queue at 10:18:23 UTC (11:18 BST), with no unresolved checkpoint or queue error. All 1,237 tracked image-package files were preserved. Documented the initial unused raw-image audit failure without asserting its cause, corrected geo-only input dependencies, and recorded that completed image packages used different realized crop/shuffle streams. Their results remain descriptive and provisional; later seeds have not started. |
| 2026-10-20 | Implemented and launched D040: 16 seed-1 frozen-online fits at 22:56 UTC (23:56 BST), process 490326. Separate configuration/root, paired input/recipe binding, regime-aware orchestration and paired final/best reporting are complete. A one-bit mismatch was localized to a captured frozen reference; production references and eligible saved frozen encoders now reuse authenticated original values. All failed evidence remains. Final validation passed 265 CPU checks and all 16 GPU model checks plus checkpoint loading/export. Prior partial-joint artifacts and scientific settings are unchanged; results remain provisional. |
| 2026-10-20 | Closed the seed-1 partial-joint package at 21:39 UTC (22:39 BST): 16/16 fits, 800 training epochs, 32 reloaded final/best checkpoint-policy evaluations, no unresolved fits or queue error. Added the completion audit, results/deliverable links and handoff to the next 16 seed-1 frozen-online fits. The close-out checks existing metadata, recorded evidence and artifact presence/sizes; it performs no new training, inference or checksum audit. M3J overall remains in progress, provisional assurance remained, and frozen-online execution integration/verification was pending at that close-out snapshot. Scientific settings and completed artifacts were unchanged. |
| 2026-10-20 | Recorded user-approved D039. The seed-1 job stopped after 12 completed fits; fit 13 completed 50 epochs and passed its final freeze check but failed while authenticating its best checkpoint. Read-only per-file cache advice recovered the original digest on two reads, without changing checkpoint bytes. Implement bounded exact-hash retries, evaluation-only recovery and continuation of independent unstarted fits; do not redo completed training or automatically restart partial attempts. Preserve scientific settings, original protocol/verification evidence and provisional assurance. Implementation complete with 134 focused checks passed; numerical training-core AST unchanged. Continuation launched at 15:48 UTC (16:48 BST), process 470054. At the historical 15:51 UTC snapshot, the thirteenth fit completed final/best evaluation with zero additional epochs; 13 fits were complete, the fourteenth active and two further fits unstarted. All 152 preservation checks passed. |
| 2026-10-19 | Recorded user-approved D038: replace mandatory resume/replay equality, tolerance budgets and repeated live-weight checksum gates with direct frozen-value checks and saved-checkpoint loading/evaluation. Keep file authentication, option 3 storage, original epoch selection/curves, reloaded final/best reporting and provisional unresolved-memory labels. Fresh interrupted attempts restart at epoch 1. Implementation complete; 163 CPU checks and complete GPU verification passed. Authorized 16-fit background job started seed-1 image-only/pretrained at 13:51 UTC; 1/16 fits started, 0 complete at this update. Prior D036/D037 evidence is preserved. |
| 2026-10-19 | Recorded D037: the user explicitly authorized tiny frozen-weight deviations in a separate exploratory mode. Declared isolated identity, fixed tolerances and cumulative limits, unchanged hard integrity checks, paired randomness and mandatory unresolved-memory provisional labels. Implementation complete with 201 focused tests passed; the first exploratory GPU attempt failed during live-reference creation. Canonical references now come from the authenticated original checkpoint; the subsequent attempt passed pretrained image-only, then failed on inconsistent checks of the same adapted image-only snapshot. Both attempts are retained; 0/16 full fits started. |
| 2026-10-19 | Marked M3J full-fit execution blocked: repeated exact GPU integrity failures were followed by two positive CPU-only memory-corruption probes. Preserved all diagnostic evidence and added exact initial-transfer/frozen-weight/checkpoint-transfer checks. Partial-joint implementation and 124 tests are complete; 0/16 full fits started, no reboot or system changes. |
| 2026-10-19 | Began the user-authorized 16 seed-1 partial-joint package. Added the online implementation, fixed executable recipe and contrast/refit contract, shared checkpoint/export/resume handling, live curves and dual-checkpoint reporting. CPU checks pass; GPU verification precedes comparative fits. Later M3J packages remain planned. |
| 2026-10-19 | Adopted the user's seed-by-seed execution order: 16 partial-joint fits, 16 frozen-online fits and one geo-only fit per seed, then advance to the next of seeds 1–4. Kept all 132 fits, independent starting states, paired randomness and unchanged recipe. The first 16 partial-joint fits are presented for review; no implementation or training started. |
| 2026-10-19 | Recorded user-approved storage option 3: shared immutable starting encoders and shared final/resume weights, with complete final/best inference and resume evidence. Added dependency manifests, atomic publication, standalone export and later cleanup requirements; recorded the 315 GB tensor estimate and 400 GB planning allowance. No checkpoints were deleted; implementation and comparative training remain unstarted. |
| 2026-10-19 | Fixed M3J at 50 epochs with cosine `T_max=50` and historical FP16-autocast/FP32-weight training without GradScaler. Recorded historical TF32 flags and autocast-disabled validation/replay, retained final epoch 50 plus best validation from epochs 1–50, and removed unresolved duration/precision wording from the active plan. The 132-fit matrix totals 6,600 epochs. Completed-study files remain unchanged; no implementation or training started. |
| 2026-10-19 | Resolved batch size as physical/effective 16 without accumulation and adopted historical Adam settings, zero weight decay and no clipping. Retained shared `5e-5` cosine decay, no warm-up and matched batch/update policies across all models. Superseded the AdamW/weight-decay proposal; duration and precision remain pending. No implementation or training started. |
| 2026-10-19 | Recorded the historical final-10-block visual scope plus final normalization/pooling, and physical batch 16. Distinguished 11 framework groups from 10 blocks. Kept accumulation/effective batch pending clarification of historical 16 versus the earlier proposed 128; updated stale smaller-encoder-rate wording. No implementation or training started. |
| 2026-10-19 | Recorded the user's cosine-decay choice for M3J: shared initial `5e-5`, zero minimum, end-of-epoch stepping over the declared M3J duration, no warm-up or restarts. Specified epoch-used versus post-step rate and resume continuity. Optimizer/weight decay and duration remain separate decisions; no implementation or training started. |
| 2026-10-17 | Adopted the user's simpler initial M3J rate strategy: uniform `5e-5` for all trainable components, immediate partial-joint updates and no warm-up. Deferred warm-up/module-specific rates, distinguished the historical initial rate from its Adam/cosine recipe, and recorded that the new rate requires four fresh geo-only fits. Optimizer/schedule and other remaining numerical choices are still open; no implementation or training started. |
| 2026-10-17 | Recorded the user-confirmed M3J comparison: both image initializations with historical preprocessing, all nine M3/extension architectures, and standardized geo from the outset. Replaced the four-model scope with a 132-fit one-recipe planning baseline, counting geo-only once per seed. Numerical recipe/resource confirmation remains pending; no M3J implementation or training started and original study files remain unchanged. |
| 2026-10-16 | Recorded the user's M3J dual-checkpoint reporting decision: retain and report final epoch and best validation, define deterministic per-seed selection, preserve all seed results and checkpoint provenance, and distinguish checkpoint monitoring from early stopping and configuration/refit choices. Documentation only; no M3J runs started. |
| 2026-10-16 | Made M3J training/validation monitoring explicit: every-epoch project metrics and losses, saved histories and refreshable curves, learning-rate/stage/resource traces, seed variation and resume continuity. Added train-mode versus evaluation-mode interpretation and prospective stopping/checkpoint-selection requirements. No M3J implementation or fits started. |
| 2026-10-16 | Completed the M3 CLS-bypass extension: 72 fits, 24 selected checkpoints, 37 contrasts, both primary results inconclusive and no supported extension gain. All 24 selected GPU replays and independent numerical reconstruction are exact; 203 tests and real-figure review pass. Original M3 artifacts remain unchanged; no new extraction, test application, refit or joint training. |
| 2026-10-16 | Started the authorized M3 CLS-bypass extension. Added its detailed recipe record: two variants, 72 new fits, 24 selected references and 37 prospective contrasts, including both native-geo primary comparisons. Original M3 evidence remains unchanged. |
| 2026-10-16 | Planned a user-requested M3 CLS-bypass extension with both adapted-token (384D) and original-embedding (1,344D) readouts, a shared two-layer backbone, 72-fit core budget and separately versioned evidence. Original M3 remains complete. Documentation only; implementation and training await the user's explicit cue. |
| 2026-10-16 | Completed M3: 228 matched fits with the unchanged M1 recipe; 76 selected checkpoint references; 63 predeclared paired contrasts and inspected figures. Primary outcome: no improvement. Standardized-geo raw concatenation has the highest observed mean. All 76 selected checkpoint replays and independent numerical analysis replay are exact; 147 focused tests pass. Normal GPU access works without the previous temporary library override. No reboot, joint training, refit or test application. |
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

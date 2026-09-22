# Approved OOF-supervised ConfidNet and augmented-router study

Status: COMPLETE — all twelve confidence heads and eight matched RF routers
completed. Final model/score validation, independent scientific audits, and
the full-build immutable-reuse check passed. All 305 focused tests
passed before the continuation. Its declaration was sealed at
`2026-09-22T19:23:06.392443+00:00`, binding 364 source/parent files. The user
approved exact native float32 reconstruction as the mandatory prediction check
and float64 differences as diagnostics. The continuation ID is
`confidnet_oof_v1_replay_v2`; scientific sources and configuration are frozen.
The scientific plan remains the approved twelve-head/eight-router study.
The original implementation passed 193 focused tests and its declaration sealed
337 source/parent files. Seed 1 features and the original pause are preserved.
The confidence-augmented RF underperformed the matched numerical RF on reused
development-validation: 69.5560% versus 71.9282% mean accuracy. The execution record is
[14_confidnet_oof_experiment.md](14_confidnet_oof_experiment.md).

The user requires approval of important experimental settings before execution.
The current approval covers the first-round settings below.
Changes to inputs, architecture, supervision, optimization, budgets, router
family, or policy selection will be presented before execution.

Review update: the user requested numerical summaries only for router inputs.
The proposed comparison now uses the existing `ablation_numeric25` feature set
versus those same 25 summaries plus three estimated TCPs (28 inputs). This
revision is included in the subsequent approval of the complete first-round setup.

## Question and proposed first-round scope

Does fitting confidence estimation on expert-held-out examples yield useful
TCP estimates, and do those estimates improve a direct geo override or an
otherwise matched router on development-validation?

Approved scope: twelve confidence fits and eight RF router fits. This follows
the user's pooled OOF supervision and same-row confidence-prediction design.
Optional experiments listed at the end are excluded unless explicitly approved.

## Data and expert checkpoints

- Retain all four training seeds, 1/2/3/4, as separate realizations. Do not pool
  seed realizations when fitting confidence heads or routers.
- Retain the existing plot-grouped development split: 3,378 train images on
  1,300 plots, and 822 validation images on 325 disjoint plots, per seed.
- Retain the four existing OOF partitions inside development-train and the
  frozen image, geo, and raw-fusion experts for each partition.
- Reuse the full-development-train expert set to generate validation inputs.
  No expert retraining, encoder adaptation, or new expert-temperature fit.
- Do not access cleaned-test examples or use legacy test-transfer results to
  select configurations. Existing artifacts remain unchanged.

## Ordered execution after approval

1. Implement and test the approved configuration and save a declaration with
   source, checkpoint, split, and configuration hashes before production fitting.
2. For each seed and OOF fold, run the paired experts only on that fold's held-out
   rows. Extract the confidence inputs and native-T=1 logits. Verify row/plot
   identity and reproduce the previously sealed OOF logits.
3. Join development-train labels to compute TCP separately for each expert:
   `TCP_e = softmax(logits_e)[true_label]`. Concatenate four held-out folds into
   3,378 confidence-training rows per expert and seed. Every target comes from
   an expert that excluded that row's plot during fitting.
4. Fit one confidence head per expert per seed using all those rows: three
   expert modes times four seeds equals twelve confidence heads.
5. Produce confidence estimates for those same OOF training representations.
   Separately produce estimates for development-validation representations from
   the full-development-train experts. No validation label enters these fits.
6. Fit matched original-feature and augmented-feature routers on development-train.
   Save all model states and predictions before scoring development-validation.
7. Evaluate confidence quality, fixed direct policies, and matched router
   policies separately. Report every seed and unsuccessful outcome. Verify
   prediction replay, calculations, parent preservation, and reuse.

## Confidence input and architecture: proposed approval items

Primary input: the 256-dimensional activation immediately after the ReLU in
each frozen expert's classification head. Image, geo, and fusion each have
their own confidence model. These inputs do not include logits or probabilities.

```
256 -> 400 -> 400 -> 400 -> 400 -> 1
```

ReLU follows each of the four hidden layers. A sigmoid converts the final
output into estimated TCP. All five linear layers have biases.

The four 400-unit layers follow the original classification confidence-head
implementation. The input dimension comes from our expert architecture.
[Original model](https://github.com/valeoai/ConfidNet/blob/57d4a056adab1eb75f4a54c73422e60751ea3709/confidnet/models/vgg16_selfconfid_classic.py).

The optional direct-embedding experiment below would replace this input; it
will not be silently substituted for the proposed primary input.

## Confidence training: proposed approval items

| Setting | Proposed value |
|---|---|
| Supervision | Each expert's native-T=1 TCP on its held-out fold |
| Loss | Unweighted mean squared error; equal weight per image |
| Optimizer | Adam, learning rate 0.0001, betas 0.9/0.999, epsilon 1e-8 |
| Budget | 500 epochs, batch size 128, shuffle each epoch, retain final partial batch |
| Model selection | Final epoch only; no early stopping or architecture/learning-rate search |
| Initialization | Standard linear-layer initialization under the corresponding seed, reset for each expert head |
| Regularization | No weight decay or confidence-head dropout |
| Confidence input scaling | No additional normalization or learned preprocessing |
| Expert inference | Frozen weights, evaluation mode, deterministic inherited preprocessing, dropout disabled |
| Runtime | CPU float32, one compute thread, deterministic seed and batch order |
| Prediction convention | Native float32 predictions are authoritative; JSON reconstruction must reproduce them exactly; independent float64 replay differences are diagnostic |

Width 400, learning rate 1e-4, and 500 epochs replace the initial study's width
128, learning rate 1e-3, and 100 epochs. The learning rate, budget, batch size,
and unweighted MSE match the original repository's classification example.
[Original configuration](https://github.com/valeoai/ConfidNet/blob/57d4a056adab1eb75f4a54c73422e60751ea3709/confidnet/confs/selfconfid-classif.yaml).

This still is an adaptation: we train on cached deterministic OOF features and
do not reproduce the original augmentation/dropout behavior or its optional
confidence-encoder fine-tuning phase.

Record training loss, TCP/estimate quantiles, error counts, input scales, gradient
norms, and the fractions of estimates within 1e-6 of zero or one. Compare
training regression loss against an always-one predictor and a training-mean
predictor. These checks do not select a validation-performing configuration.

Any nonfinite loss, gradient, parameter, or prediction stops the run. Five
consecutive epochs with exactly zero maximum gradient across all batches pause
the study for review before router fitting or validation scoring. Otherwise,
complete the declared budget and report saturation diagnostics. Any proposed
remedy or change of recipe requires user approval; no automatic retuning.

## Router setup: proposed approval items

Use RF as the approved bounded first comparison, with the depth/leaf setting
selected in the earlier development capacity study. Other future learner
comparisons remain outside this study.

| Item | Original-feature baseline | Augmented router |
|---|---|---|
| Training rows | Existing expert-OOF development-train rows | Same rows |
| Inputs | Existing `ablation_numeric25`: 25 summaries | Same 25 plus estimated TCP for image, geo, fusion: 28 columns |
| Targets | Rescue / harm / both correct / both wrong | Identical targets |
| Trees | 500 | 500 |
| Maximum depth | 4 | 4 |
| Minimum leaf size | 5 images | 5 images |
| Feature sampling | `max_features=0.5` | `max_features=0.5` |
| Other settings | Bootstrap, log-loss criterion, no class weights, seed-local random state, one worker | Identical |

Use the existing fixed implementations/defaults for the remaining RF settings
from the capacity study, with no new hyperparameter search. The unchanged
feature fraction considers 12 versus 14 candidate columns per split because
the input dimension changes; this is disclosed as part of the comparison.

The 25 inputs are the existing fixed numerical-only variant, including its
Boolean and integer summaries:

| Summary family | Columns |
|---|---:|
| Per-expert maximum probability, entropy, and top-two margin | 9 |
| Geo-minus-image and geo-minus-fusion differences in those summaries | 6 |
| Image/geo, image/fusion, and geo/fusion top-one agreement flags | 3 |
| Image/geo top-three overlap and fusion rank of the geo prediction | 2 |
| Image/geo Jensen-Shannon divergence and total variation | 2 |
| Image probability at geo prediction, geo probability at image prediction, and fusion probability at geo prediction | 3 |

Exclude the 702 predicted-class and predicted-class-pair one-hot columns.
Identifiers and producer metadata remain available for joining/audit only.

Retain the existing frozen temperatures and feature transformations for the
original 25 columns. Append exactly three native-TCP estimates,
without explicit difference features or true TCP. Refit both router arms under
the same settings rather than compare a new policy with historically selected
thresholds. Four seeds times two arms equals eight router fits.

The confidence estimates used here are **in-sample for ConfidNet, OOF for the
experts**, as requested. They are not fully cross-fitted router features.
The same OOF representations used to fit ConfidNet generate router-training
confidence estimates; full-development-train expert outputs must not replace
these training rows. No extra confidence cross-fitting is included.

## Fixed policies and experiments

All switches require geo/fusion top-1 disagreement. Ties retain fusion.

| Experiment | Rule |
|---|---|
| Fusion reference | Always retain raw fusion |
| Native-MCP reference | Choose geo if native MCP_geo - MCP_fusion > 0 |
| Direct ConfidNet comparator | Choose geo if estimated TCP_geo - estimated TCP_fusion > 0 |
| Original-feature RF | Choose geo if predicted P(rescue) - P(harm) > 0 |
| Augmented RF | Identical RF decision rule with the three additional inputs |

There is no threshold search or minimum-coverage/harm-cap selection in this
first round. These are fixed diagnostic policies, not policies promoted through
the original M4 gate. Historical validation-selected RF results are context,
not the matched comparator. Ranking metrics are also reported, since a fixed
zero threshold can miss useful ranking improvements.

## Evaluation and interpretation

- Confidence: TCP MSE/MAE, failure AP and ROC AUC against native MCP and entropy,
  distributions conditional on correct/incorrect expert predictions, and
  training-to-validation gaps. Training figures are fitting diagnostics.
- Routers: four-state NLL/Brier, rescue ranking AP on disagreements reported
  separately for P(rescue) and operational P(rescue)-P(harm) scores, routed
  habitat top-1 accuracy and weighted F1, actions, rescues, harms, wrong-to-wrong
  switches, net corrections, coverage, and plots affected.
- Compare the augmented router directly with the matched original-feature RF
  and with fusion. Give per-seed figures and the mean across all four seeds.
- Use 2,000 unstratified paired whole-plot bootstrap samples, random seed
  20260826, retaining all images and all seed realizations of each sampled plot.
  Use the same resamples for all policy contrasts and report percentile 95%
  intervals. Repeated seed rows are not independent images.
- A shared confidence head must transfer across fold-specific and full-train
  expert representations. Record producer-specific activation/score summaries;
  compatible dimensions alone do not establish transferable coordinates.
- The pipeline is evaluated on reused development-validation, not independent
  confirmation. No validation labels fit confidence heads or routers. All
  predictions are saved before scoring, with no within-round validation-driven
  recipe changes.
- The old initial experiment is descriptive context only. Width, optimizer
  budget, and supervision all change, so improvement cannot be assigned solely
  to OOF supervision without additional controls. Even the control below does
  not fully isolate expert-training-size and producer effects.

## Optional extensions: excluded from the proposed first round

1. **Direct embeddings:** repeat the OOF study with input dimensions 1152 for
   image, 64 for geo, and 1216 for their concatenation. Preserve each producer's
   image L2 normalization and geo standardization. Keep width 400 and the same
   confidence recipe. This adds twelve confidence fits and four augmented RF
   fits; reuse the matched baseline. The image encoder remains fold-specific,
   so this does not guarantee a shared representation. It directly addresses
   the user's earlier expectation about using image embeddings as inputs.
2. **Matched-recipe in-sample control:** fit twelve additional confidence heads
   on the cached full-development-train experts' own training representations
   using the new 400-unit/1e-4/500-epoch recipe. Evaluate confidence and direct
   policies on validation, with no additional augmented router in this control.
   This helps distinguish recipe changes from the switch to OOF supervision.
3. **True-TCP privileged diagnostic or additional confidence cross-fitting:**
   separate future designs; neither is included in the primary router inputs.

## Approval record

The user approved the twelve-head/eight-router first round, including:
256-dimensional penultimate inputs; four hidden layers of width 400; unweighted
MSE/Adam 1e-4/500 epochs without added normalization; RF numeric25 versus numeric28;
fixed positive-difference policies; and the training-failure pause rule.

The optional extensions were not selected and are excluded. Further changes
to consequential settings still require approval before execution. The original
frozen recipe remains `configs/multimodal_geo_confidnet_oof_v1.json`.

### Approved verification amendment

The original run paused because one of 3,378 seed 1 image training predictions
differed between native float32 inference and float64 portable replay by
`1.1991306406722302e-6`, above an absolute-only `1e-6` verification tolerance.
The recovered native float32 model reproduced its predictions exactly after
JSON reconstruction. The diagnostic repetition completed 500 finite epochs
with no zero-gradient epochs; no validation scoring or router fitting occurred.

The user subsequently approved the following verification contract:

- Preserve native float32 predictions, including their exact ties, as authoritative.
- Require exact native float32 prediction equality after JSON state reconstruction.
- Retain independent float64 replay and record its numerical differences as
  diagnostics, without rejecting a native-exact model solely for exceeding the
  old cross-precision absolute tolerance.
- Keep finite-value, parameter-integrity, shape, lineage, and artifact-hash checks
  mandatory. Preserve available state/history if a final prediction check fails.
- Keep the scientific inputs, targets, architecture, optimization, budgets,
  numeric25/numeric28 RF comparison, fixed policies, and evaluation unchanged.

The additive continuation uses study/output ID `confidnet_oof_v1_replay_v2`,
configuration `configs/multimodal_geo_confidnet_oof_replay_v2.json`, workflow
`multimodal/geo_helpfulness_confidnet_oof_v2.py`, fitting implementation
`multimodal/geo_helpfulness_confidnet_oof_fit_v2.py`, and CLI
`tools/run_multimodal_geo_helpfulness_confidnet_oof_v2.py`.
It preserves the original declaration, pause, source files, and diagnostic
evidence, and reuses authenticated seed 1 extraction with recorded provenance.
All twelve confidence heads were fitted afresh with the unchanged recipe,
including seed 1's image head; no diagnostic head was imported. Every head
completed 500 epochs and passed exact native float32 prediction replay.
The continuation declaration SHA-256 is
`89e8c66110d38d027c883b69b28137b9cce2ac8c3d372a451993eefb9da7adc4`.

### Executed outcome

The augmented-minus-baseline accuracy difference was **−2.3723 percentage
points**, with the predeclared paired whole-plot 95% interval
**[−4.5736, −0.3865]**. The direct estimated-TCP comparator also decreased mean
accuracy relative to fusion. Confidence heads fitted OOF TCP targets closely
but showed large gaps between their fitting diagnostics and validation failure
detection. This design does not isolate confidence overfitting from transfer
between fold-specific and full-development-train expert representations.

No recipe or threshold was changed after validation scoring. The original M4
conclusion remains unchanged. Detailed confidence metrics, policy counts,
per-seed outcomes, verification evidence, and interpretation limits are in
[the experiment report](14_confidnet_oof_experiment.md).

The [completion receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2_review/completion_receipt.json)
records successful production and repeat builds. The independent metric audit
verified 1,528 metric/interval comparisons; repeated execution preserved all
191 production files exactly in bytes, size, and modification time.

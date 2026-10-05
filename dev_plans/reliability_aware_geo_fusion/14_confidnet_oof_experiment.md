# OOF-supervised ConfidNet and numerical-router experiment

Status: COMPLETE — all twelve confidence heads, eight RF routers,
fixed policy comparisons, and final model/score validation completed. An
independent metric and bootstrap audit passed, along with model/provenance
audits and full-build immutable reuse. All 305 focused tests passed before the continuation.
Its declaration was sealed at `2026-09-22T19:23:06.392443+00:00`, binding 364
source/parent files. Scientific sources and configuration remained frozen.
The original run remains preserved at its replay-verification pause.

Original study ID: `confidnet_oof_v1`; approved continuation ID:
`confidnet_oof_v1_replay_v2`. The user approved the settings in
[the OOF experiment proposal](13_confidnet_oof_proposal.md), including the
revision from the full router feature set to numerical summaries only.
The approved delivery comprises twelve confidence heads and eight matched RF routers. The
optional direct-embedding inputs, same-recipe in-sample control, true-TCP router
inputs, and additional confidence cross-fitting are outside the approved scope.

## Question and comparison

Does confidence supervision from expert-held-out examples produce useful TCP
estimates, and do those estimates improve a direct geo override or a numerical
router on development-validation?

The matched router comparison is **25 existing numerical summaries versus the
same 25 summaries plus three estimated TCPs**. Both arms are fitted again with
the same RF settings and evaluated with the same fixed intervention rule.
Historical routers with validation-selected thresholds are context rather than
the matched baseline for this experiment.

Subsequent user-requested analysis: [relaxed threshold selection](15_confidnet_relaxed_guardrails.md)
reuses these saved routers under the previous RF guardrail grid. It raises
augmented accuracy to 71.7762%, versus 72.0499% for the equally selected
numeric25 baseline. That separate result does not alter the fixed-zero
protocol or results below.

## Results: confidence fitting succeeded, routing did not improve

Every confidence head completed 500 epochs with no zero-gradient epoch. All
native float32 predictions reproduced exactly from serialized model states.
All twelve heads and eight routers were verified before validation labels were
joined at `2026-09-22T19:57:24.572357+00:00`. The fitting budget, architecture,
input features, and policy rules were not changed after scoring.

The TCP-augmented RF reduced mean validation accuracy from **71.9282%** for the
matched 25-feature RF to **69.5560%**. The paired difference was **−2.3723
percentage points**, with 95% whole-plot interval **[−4.5736, −0.3865]**.
Its weighted-F1 difference was **−2.7067 points**, interval
**[−5.0034, −0.7142]**. These are conditional exploratory intervals on the reused
development-validation split, not independent confirmation.

### Fixed validation policies

Values are means over four seeds, each evaluated on the same 822 images.
Fractional action counts are seed means, not fractional image decisions.

| Policy | Accuracy (%) | Weighted F1 (%) | Actions | Rescues | Harms | Wrong-to-wrong | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fusion | 71.0462 | 70.6695 | 0 | 0 | 0 | 0 | 0 |
| Native MCP comparator | 71.2591 | 70.7710 | 79.50 | 31.00 | 29.25 | 19.25 | +1.75 |
| Estimated TCP comparator | 69.8601 | 69.1304 | 120.75 | 43.00 | 52.75 | 25.00 | −9.75 |
| Numerical RF, 25 features | 71.9282 | 71.5737 | 59.75 | 27.75 | 20.50 | 11.50 | +7.25 |
| TCP-augmented RF, 28 features | 69.5560 | 68.8671 | 120.50 | 41.50 | 53.75 | 25.25 | −12.25 |

| Policy | Accuracy gain over fusion (pp), 95% paired CI | Coverage (%) | Mean plots affected |
|---|---|---:|---:|
| Native MCP comparator | +0.2129 [−1.5288, +1.8773] | 9.6715 | 60.25 |
| Estimated TCP comparator | −1.1861 [−3.7623, +1.1861] | 14.6898 | 74.00 |
| Numerical RF, 25 features | +0.8820 [−0.5580, +2.3257] | 7.2689 | 45.75 |
| TCP-augmented RF, 28 features | −1.4903 [−4.0567, +0.8503] | 14.6594 | 72.25 |

Augmentation increased rescues by 13.75 per seed but increased harms by 33.25.
It approximately doubled coverage. The direct TCP comparator also made more
harmful than beneficial corrections. The baseline RF's positive gain over fusion
has an interval spanning zero under this fixed diagnostic policy.

| Seed | Fusion accuracy (%) | MCP comparator (%) | TCP comparator (%) | Numerical RF (%) | Augmented RF (%) | Augmented minus numerical RF (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 69.9513 | 70.3163 | 68.6131 | 70.9246 | 70.0730 | −0.8516 |
| 2 | 70.3163 | 71.8978 | 70.3163 | 71.4112 | 69.8297 | −1.5815 |
| 3 | 71.6545 | 70.9246 | 70.6813 | 72.1411 | 71.0462 | −1.0949 |
| 4 | 72.2628 | 71.8978 | 69.8297 | 73.2360 | 67.2749 | −5.9611 |

The augmented router underperformed its matched baseline in every seed, with
the largest decline in seed 4. Full per-seed action/F1 tables are in the
[generated report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2/scores/report.md);
all metric and interval details are in
[the scored JSON](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2/scores/report.json).

### Confidence generalization

The heads closely fitted their OOF TCP supervision. Their own training
predictions are in-sample for ConfidNet, so these figures measure fitting,
despite the expert outputs being OOF.

| Expert | Train TCP MSE | Validation TCP MSE | Train failure AP | Validation failure AP | Validation MCP AP | Validation entropy AP |
|---|---:|---:|---:|---:|---:|---:|
| Image | 0.002257 | 0.225577 | 0.996011 | 0.470685 | 0.603885 | 0.621232 |
| Geo | 0.000156 | 0.147713 | 0.996926 | 0.579130 | 0.584997 | 0.573980 |
| Fusion | 0.001903 | 0.184433 | 0.997018 | 0.485214 | 0.535818 | 0.541147 |

Validation MAE was 0.299339, 0.272080, and 0.271775 for image, geo, and fusion.
The corresponding validation MSEs of a predictor fixed to each expert's
training-mean TCP were 0.183446, 0.155254, and 0.172329. Only geo beat this
training-mean constant in validation MSE; geo and fusion beat native MCP in
TCP MSE. Regression improvement did not establish better failure ranking.

| Expert | Validation estimated-TCP ROC AUC | MCP ROC AUC | Entropy ROC AUC |
|---|---:|---:|---:|
| Image | 0.715987 | 0.780527 | 0.783340 |
| Geo | 0.718992 | 0.714361 | 0.714242 |
| Fusion | 0.750724 | 0.765141 | 0.766969 |

Image and fusion failure detection underperformed MCP and entropy. Geo was
mixed: its AP trailed MCP slightly while AUC improved slightly. The broad OOF
supervision avoided the initial study's complete confidence collapse, but did
not close the confidence fitting-to-validation gap. Validation estimates within
`1e-6` of one averaged 14.90% for image, 0.18% for geo, and 8.06% for fusion;
these are saturation diagnostics rather than evidence of calibrated reliability.

### Router fitting-to-validation gap

The augmented router achieved mean training rescue recall 99.58%: 305 rescues
out of 306.25 opportunities, with zero harms. On validation it made 41.5 rescues
and 53.75 harms, with mean rescue recall 48.45%. Its near-perfect fitting
behavior therefore did not transfer to the validation pipeline.

| Validation router diagnostic | Numerical RF | TCP-augmented RF |
|---|---:|---:|
| Four-state NLL | 0.617915 | 1.361740 |
| Four-state Brier | 0.368856 | 0.521483 |
| Rescue AP on disagreements, `P(rescue)` | 0.451038 | 0.380912 |
| Rescue AP on disagreements, `P(rescue)-P(harm)` | 0.452060 | 0.370942 |

The augmented NLL includes one validation prediction, across the four seeds,
with zero probability for its true four-state target. Following the inherited
metric definition, it is floored to float64's smallest positive normal value,
`2.2250738585072014e-308`, before taking the logarithm. Brier and both ranking
metrics also deteriorated, so the result is not solely an NLL-floor effect.

### Scientific conclusion

This approved configuration provides **no evidence of a routing benefit** from
the three learned TCP inputs. It exposes a large generalization gap after
successful confidence fitting. Same-row confidence supervision can let the
router learn relationships that are much cleaner in training than at inference;
this is consistent with the observed near-perfect training rescue decisions.

Transfer between independently fitted fold representations and the
full-development-train representation is a separate plausible contributor.
Producer-specific activation and score summaries are saved and replay-verified,
but this design does not isolate that mechanism from confidence overfitting,
expert-training-size differences, or the ordinary training/validation shift.
The results do not show that OOF TCP supervision or embeddings are intrinsically
unhelpful. Additional confidence cross-fitting or a representation/control
comparison would be a separately approved experiment, not a change to this run.

## Preserved original execution and pause

Seed 1's four held-out-fold producers and its full-development-train producer
were extracted, reproduced against their saved expert outputs, and sealed.
Their pooled training and validation feature bundles were also sealed. The
original run paused before extracting features for seeds 2–4; those seeds were
subsequently extracted in the approved continuation.

The first image ConfidNet reached the final training-prediction check, which
raised:

```text
train confidence portable parity exceeds 1e-06
```

This check compares authoritative native float32 predictions with a float64
portable replay. The exception occurred before confidence inference on
validation, before any confidence head was sealed, and before any router fit
or development-validation scoring. It does not establish a failure of the
research hypothesis or a validation-performance result.

The final parity exception does not attach the state/history to the exception
object, so the production pause bundle contains the reason but not the first
fit's state or history. One identical, deterministic, **training-only diagnostic
repetition** recovered the failure and saved its diagnostics under the separate
`confidnet_oof_v1_diagnostics` output directory. It retained the same seed,
training inputs, architecture, and 500-epoch recipe, did not inspect validation
features or targets, and did not resume production. The diagnostic repetition
completed all 500 epochs with finite recorded values and no zero-gradient epochs.
The unavailable original history is not retrospectively represented as a saved
production checkpoint.

| Recovered training-only diagnostic | Result |
|---|---:|
| Training rows | 3,378 |
| Completed epochs | 500 |
| Final training MSE | 0.004637676660005977 |
| Native float32 JSON-reconstruction maximum prediction error | 0; exact equality |
| Native float32 versus portable float64 maximum difference | `1.1991306406722302e-6` |
| Mean absolute float32/float64 difference | `5.1218354331504035e-8` |
| Rows exceeding the `1e-6` absolute tolerance | 1 / 3,378 |
| Rows exceeding `1e-5` | 0 |
| Zero-gradient epochs | 0 |

The reproduced issue is a cross-precision verification difference; the native
float32 model reconstructs exactly. These are fitting and numerical diagnostics,
not validation or router results. Recovered evidence is in the
[diagnostic report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_diagnostics/diagnostic_report.json),
[500-epoch history](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_diagnostics/recovered_history.json),
and [recovered state](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_diagnostics/recovered_state.json),
with the diagnostic declaration and manifest recording provenance.

The original sealed declaration, feature bundles, and pause remain preserved.
Following this diagnosis, the user explicitly approved the narrow verification
amendment below. The additive continuation build has now launched under its
separate declaration and completed. All twelve confidence heads were fitted
afresh, including seed 1's image head, using the unchanged scientific recipe.
No diagnostic head was imported. The new seed 1 image fit exactly reproduced
the diagnostic state's bytes, all 500 history entries, and training predictions,
as recorded in the
[deterministic repeat audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2_review/confidnet_oof_v2_seed1_repeat_audit.json).

## Training-only OOF supervision audit

An independent audit of seed 1's sealed training cache confirms 3,378 unique
images on 1,300 plots, with each plot assigned to exactly one producer fold.
The pooled OOF supervision contains substantial numbers of expert failures:

| Expert | OOF training errors | Error rate | Mean TCP | Median TCP | TCP >= 0.99 |
|---|---:|---:|---:|---:|---:|
| Image | 1,079 | 31.9420% | 0.672798 | 0.983401 | 47.3357% |
| Geo | 1,251 | 37.0337% | 0.582149 | 0.702458 | 19.8638% |
| Fusion | 1,018 | 30.1362% | 0.689882 | 0.987058 | 48.7863% |

Across the four training folds, every cached expert top-one prediction matches
the saved OOF output. All logit comparisons pass the inherited combined
`atol=1e-6, rtol=1e-6` criterion. Maximum absolute differences are
`4.76837158203125e-6` for image, `0` for geo, and `3.814697265625e-6` for fusion;
the largest difference uses 96.72% of its allowed combined tolerance.
This expert-logit criterion differs from the confidence replay's absolute-only
`1e-6` check that paused fitting.

The complete [training-only audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_review/confidnet_oof_train_audit.json)
includes conditional TCP quantiles and per-fold counts.
These findings establish the availability of richer confidence supervision,
not generalization or router improvement.

## Data and fitting boundary

The study retains the existing four seeds and plot-grouped development split:
3,378 training images on 1,300 plots and 822 validation images on 325 separate
plots, for each seed. Each seed is fitted and reported separately.

For each seed, the four frozen expert sets predict their respective held-out
folds. Their hidden activations and native-temperature-one logits are pooled
into one training dataset per expert. TCP targets are the probability assigned
to each image's true class. Every training feature and target comes from an
expert that excluded that image's plot while fitting.

One image, one geo, and one fusion ConfidNet are fitted to those 3,378 pooled
rows. The fitted heads predict the **same saved OOF representations** to supply
the router-training TCP estimates. These estimates are therefore **in-sample
for ConfidNet and OOF for the experts**. No true TCP enters the router inputs.
Router targets remain rescue, harm, both correct, and both wrong, derived from
the paired OOF expert predictions.

For development-validation, the frozen full-development-train expert set
generates the hidden representations and ordinary router signals. The pooled
OOF-trained confidence heads generate estimated TCPs from those representations.
No expert or confidence model is updated during this step.

The inherited expert context reads development assignments for split validation;
validation labels are omitted from extracted prediction inputs and neither fit
API accepts validation targets. All twelve confidence models and eight routers,
with their train/validation predictions, must be sealed and verified before the
scoring stage joins validation labels. No cleaned-test evaluation is included.

## Frozen recipe

The original machine-readable recipe remains
[`multimodal_geo_confidnet_oof_v1.json`](../../configs/multimodal_geo_confidnet_oof_v1.json).
The continuation's recipe is
[`multimodal_geo_confidnet_oof_replay_v2.json`](../../configs/multimodal_geo_confidnet_oof_replay_v2.json), with identical scientific
settings and the approved prediction-verification amendment.

| Component | Approved setting |
|---|---|
| Confidence input | Paired expert's 256-dimensional penultimate ReLU activation |
| Confidence architecture | `256 -> 400 -> 400 -> 400 -> 400 -> 1`; biased linear layers, hidden ReLUs, final sigmoid |
| Confidence target/loss | Native-T=1 TCP; unweighted image-level MSE |
| Optimizer | Adam, learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8` |
| Fitting budget | 500 epochs, shuffled batches of 128, retain final partial batch and final epoch |
| Confidence regularization | No added normalization, dropout, or weight decay |
| Confidence runtime | CPU float32, one compute thread; seed reset for each expert head |
| Expert inference | Frozen, evaluation mode, deterministic inherited preprocessing, dropout disabled |
| Router inputs | Existing transformed `ablation_numeric25`; augmented arm appends image/geo/fusion estimated TCP |
| RF recipe | 500 trees, maximum depth 4, minimum leaf size 5, `max_features=0.5`, bootstrap, log-loss criterion |
| RF controls | No class weights; seed-local random state; one worker; no hyperparameter search |

The router's 25 summaries retain the inherited expert temperatures and feature
transformations. Boolean agreement flags and integer rank/overlap summaries are
included; predicted-class and class-pair one-hot columns are excluded. The fixed
RF feature fraction considers 12 candidate features per split in the baseline
and 14 in the augmented arm.

Native float32 ConfidNet predictions are authoritative, including exact ties.
The approved continuation requires exact native float32 JSON reconstruction
and records float64 portable replay differences as diagnostics. RF predictions use native
scikit-learn float64 probabilities, with portable probability and intervention
replay checks.

The confidence recipe follows the original classification head's four hidden
layers of width 400 and its example learning rate/budget. Deterministic cached
features differ from the original augmentation/dropout procedure; this is an
adaptation, not an exact reproduction of its benchmark pipeline.

## Training diagnostics and stop rule

Record epoch losses, gradient norms, input scales, target and prediction
quantiles, training errors, and fractions of estimates within `1e-6` of either
zero or one. Compare training MSE with always-one and training-mean predictors.
Validation's training-mean baseline uses the mean computed on training targets.

Any nonfinite loss, gradient, parameter, or prediction stops the run. Five
consecutive epochs whose maximum gradient norm is exactly zero pause the study
before router fitting or validation scoring. The intended pause record preserves
available diagnostic history; the final parity-exception limitation above is
explicitly recorded. A recipe change or remedy requires user approval;
the workflow does not automatically retune or select an earlier checkpoint.

## Fixed policies and evaluation

Every switch requires geo/fusion top-one disagreement. A zero difference keeps
fusion. There is no threshold search, coverage constraint, or harm-cap selection.

| Policy | Geo intervention rule |
|---|---|
| Fusion reference | Never |
| Native-MCP reference | `MCP_geo > MCP_fusion` |
| Direct ConfidNet comparator | `estimated_TCP_geo > estimated_TCP_fusion` |
| Numerical RF baseline | `P(rescue) > P(harm)` |
| TCP-augmented numerical RF | `P(rescue) > P(harm)` |

Confidence evaluation includes TCP MSE/MAE, failure AP and ROC AUC against native
MCP and entropy, confidence distributions conditional on expert correctness, and
training-to-validation gaps. Router evaluation includes four-state NLL/Brier and
rescue AP on disagreements for both `P(rescue)` and `P(rescue)-P(harm)`.

Policy evaluation includes habitat accuracy and weighted F1, actions, rescues,
harms, wrong-to-wrong switches, net corrections, coverage, and affected plots.
Results retain each seed, the mean across seeds, and pooled descriptive metrics.
Training metrics are fitting diagnostics rather than held-out performance.

Paired 95% percentile intervals use 2,000 **unstratified whole-plot** bootstrap
resamples with seed 20260826. Each sampled plot retains all its images and all
four seed realizations. The same resamples serve every comparison with fusion
and the augmented-minus-baseline comparison. Repeated seeds do not count as
independent images. Intervals condition on the fitted models and fixed policies.

## Provenance, artifacts, and reproduction

Original declaration sealed before the paused production run:

- SHA-256: `25e44c4d1c3964cbf653ab301a1a7dd976e69ef04410e06c6399846be1565aea`.
- 337 source, configuration, checkpoint, and parent artifact files are bound by
  the declaration's hashes.
- Output root:
  `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1`.
- Frozen expert parent root:
  `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1`.

The approved additive continuation has its own
[declaration](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2/declaration.json),
sealed at `2026-09-22T19:23:06.392443+00:00`, binding 364 source/parent files.
Its SHA-256 is
`89e8c66110d38d027c883b69b28137b9cce2ac8c3d372a451993eefb9da7adc4`.
The output root is
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2`.
The original declaration hash above does not authenticate the continuation.
Authenticated seed 1 extraction was reused with provenance in the new root.
All twelve confidence heads were fitted afresh; only extraction was imported
from the original run.

Original implementation, retained unchanged:

- [Workflow](../../multimodal/geo_helpfulness_confidnet_oof.py) handles declarations,
  immutable stage bundles, fit/scoring boundaries, and replay validation.
- [Feature extraction](../../multimodal/geo_helpfulness_confidnet_oof_extract.py)
  checks OOF producer identities and agreement with the saved expert logits.
- [Fitting](../../multimodal/geo_helpfulness_confidnet_oof_fit.py) implements the
  approved confidence and RF recipes and native/portable prediction checks.
- [Pure scoring](../../multimodal/geo_helpfulness_confidnet_oof_report.py) computes
  fixed metrics, policy outcomes, producer summaries, and paired intervals.
- [CLI](../../tools/run_multimodal_geo_helpfulness_confidnet_oof.py) runs each
  production phase in order and stops on a failed or paused phase.

The original execution commands, from the repository root in the existing
`habcls` environment, are:

```bash
python tools/run_multimodal_geo_helpfulness_confidnet_oof.py build
python tools/run_multimodal_geo_helpfulness_confidnet_oof.py validate
```

The continuation is implemented in
[`geo_helpfulness_confidnet_oof_v2.py`](../../multimodal/geo_helpfulness_confidnet_oof_v2.py) and
[`geo_helpfulness_confidnet_oof_fit_v2.py`](../../multimodal/geo_helpfulness_confidnet_oof_fit_v2.py), with
[its CLI](../../tools/run_multimodal_geo_helpfulness_confidnet_oof_v2.py).
It retains the existing extraction and pure-scoring implementations, adding
[authenticated extraction import](../../multimodal/geo_helpfulness_confidnet_oof_reuse.py).
Both the production build and a complete repeat build succeeded. The standalone
validation command is also the final phase of each build:

```bash
python tools/run_multimodal_geo_helpfulness_confidnet_oof_v2.py build
python tools/run_multimodal_geo_helpfulness_confidnet_oof_v2.py validate
```

`build` uses CUDA for frozen expert feature extraction and CPU for confidence/RF
fitting. Its stages are `declare`, then `extract-seed` and `fit-seed` for seeds
1–4, followed by `fit-routers`, `score`, and `validate`. Repeating the build
verifies and reuses completed bundles; it does not overwrite them. A saved
training pause must be reviewed before further progress.

The complete continuation output contains `declaration.json`, per-producer feature bundles, combined
per-seed features, twelve confidence states/histories and prediction tables,
authenticated numerical router inputs, eight RF states/predictions, and the
scored rows plus JSON/Markdown reports. Manifests bind stage inputs and outputs.
Validation replays predictions and the numerical report, including all paired
bootstrap intervals. Only the declaration, seed 1 producer/pooled feature
bundles, and the pause bundle exist in the preserved original production output.
The diagnostic declaration, recovered state/history/predictions, diagnostic
report, and manifest are in their separate directory.

## Verification completed at the pause

- All 193 focused implementation tests passed before production.
- The preservation audit checked metadata for 1,190 original files and hashes
  for 587 original files. No original file was missing or changed.
- All 337 declared source/parent hashes still match the sealed declaration.
- All five seed 1 producer bundles and the pooled feature bundle passed their
  stage-manifest checks.
- The independent training-only audit reproduced expert outputs and verified
  complete image/plot coverage as recorded above.
- An independent reconstruction of all five confidence linear layers in both
  float32 and float64 reproduced each saved prediction array exactly, confirmed
  the one-row tolerance violation, and recalculated the final training MSE.
  It also verified 500 finite epochs with nonzero gradients throughout.

The [preservation audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_review/confidnet_oof_preservation_audit.json)
was recorded at `2026-09-22T18:58:06.962526+00:00`.
The [independent parity audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_review/confidnet_oof_parity_audit.json)
and the original preservation snapshot are retained in the separate review
bundle, with file hashes. These checks documented the original pause;
the completed continuation's checks are recorded next.

## Final verification and immutable reuse

The [completion receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2_review/completion_receipt.json),
recorded at `2026-09-22T20:01:38.147750+00:00`, confirms successful production
and full repeat builds, both with exit code zero. No required execution or
verification remains pending.

- All 305 focused tests passed before the continuation was declared.
- The [independent model audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2_review/confidnet_oof_v2_models_audit.json)
  verified all twelve confidence heads and eight RF routers. Every confidence
  head completed 500 finite epochs, beat both constant training baselines,
  and reproduced its native predictions exactly. Maximum RF probability replay
  difference was `2.220446049250313e-16`, with exact intervention decisions.
- Float64 confidence differences were recorded rather than used to alter
  native predictions. The maximum was `1.1991306406722302e-6` on training rows
  and `1.7399923133432083e-6` on validation rows. Across all heads, one training
  prediction and five validation predictions exceeded the historical `1e-6`
  tolerance. All mandatory native checks passed.
- The [independent metric audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2_review/confidnet_oof_v2_metrics_audit.json)
  verified 1,528 comparisons across train/validation, per-seed and pooled
  confidence/router/policy metrics, and all paired bootstrap contrasts.
  AP/AUC/F1 were checked against independent library calculations; bootstrap
  intervals were recomputed using sampled-plot multiplicities as image weights
  and weighted confusion matrices. Maximum discrepancy was
  `3.3306690738754696e-16`.
- The [provenance and preservation audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2_review/confidnet_oof_v2_provenance_audit.json)
  verified all 364 declared files, 20 producer bundles, and four pooled caches.
  All 60 producer/expert representation-summary groups reproduced exactly.
  The 1,190 pre-existing files retained their metadata, including unchanged
  hashes for the 587 snapshotted hashed files. All 34 original pause/diagnostic/
  review files retained their bytes and metadata. All ten imported seed 1
  producer payload/metadata files were byte-identical to their original sources.
- The full repeat build reused all **191 production files, 345,642,295 bytes**,
  preserving their SHA-256 hashes, sizes, and modification times exactly.
  The [reuse snapshot](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1_replay_v2_review/confidnet_oof_v2_reuse_snapshot.json)
  and completion receipt retain that evidence.
- No cleaned-test evaluation, validation-driven recipe change, or optional
  experimental extension was performed.

## Approved verification amendment and continuation

The user explicitly approved this narrow amendment following the diagnostic
evidence. The continuation keeps native float32 predictions authoritative and requires
**exact prediction equality after reconstructing the native float32 model
from its serialized JSON state**. Parameter integrity, correct dimensions,
finite states/activations/predictions, and artifact hashes remain mandatory.

The independent float64 replay still runs and its numerical differences
are recorded, but exceeding the absolute-only `1e-6` tolerance is
a diagnostic rather than automatically rejecting an otherwise exactly
reconstructed native model. Training inputs, supervision, architecture,
optimization, budgets, router features, policy rules, and scoring remain
as approved. The final-error handler will also preserve state and history
when a final prediction check fails, rather than saving only the reason.

The amendment is implemented in additive versioned sources and
configuration under `confidnet_oof_v1_replay_v2`. The original sources,
configuration, declaration, failure record, diagnostic evidence, and partial
production artifacts remain unchanged. The revised verification contract is
sealed in the continuation's declaration before extraction import or fitting.
The full required delivery remains twelve confidence heads, eight RF routers,
all fixed comparisons, and their verification; the amendment narrows none of
the scientific scope.

## Interpretation limits

Development-validation has already been used in earlier studies. This round is
exploratory and does not promote a policy through the original M4 gate. Existing
expert/router artifacts and the original M4 conclusion remain unchanged.

Pooling hidden activations assumes that a confidence head can transfer among
separately fitted fold producers and to the full-development-train producer.
Producer-specific activation and predicted-confidence summaries are reported;
equal dimensions alone do not establish compatible coordinates.

Router training sees confidence predictions fitted on those same training rows.
Its validation result measures this chosen pipeline, including any confidence
overfitting and producer-transfer gap. Estimated TCP is not itself a calibrated
probability that the top-one prediction is correct.

Relative to the initial experiment, supervision, confidence width, learning
rate, and epoch budget all change. An improvement cannot be attributed solely
to OOF supervision without additional controls. Those controls require a
separate approved study.

The approved exploration is complete. Its negative validation outcome is
retained without post-scoring changes to the fitted models or intervention rules.

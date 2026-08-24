# Reliability-Aware Geographic Routing and Gated Fusion

## Document purpose

This is the living method specification, implementation roadmap, scientific
evaluation protocol, and progress record for reliability-aware image and
geographic fusion in the CS habitat-classification pipeline.

The document separates:

1. observations already supported by experiments;
2. components already implemented in the repository;
3. proposed components that remain hypotheses;
4. evidence required before development proceeds to the next stage.

Update this document whenever a milestone changes status, a design decision is
made, or a step-specific implementation record is created.

| Field | Value |
|---|---|
| Method name | Reliability-aware geo routing and gated fusion (working name) |
| Primary task | CS L3 habitat classification from image and 100 m GSE features |
| Primary baseline anchor | `raw_concat` |
| Locked exploratory evaluation set | Expert-cleaned CS test set: 1,347 images from 531 plots |
| Independent confirmation set | To be identified |
| Current method status | Problem formulation and diagnostic evidence complete; learning pipeline not started |
| Last updated | 2026-08-24 |

## Contents

- [Method summary](#method-summary)
- [Expected outcome](#expected-outcome)
- [Core formulation](#core-formulation)
- [Scope and non-goals](#scope-and-non-goals)
- [Current evidence and motivation](#current-evidence-and-motivation)
- [Proposed system](#proposed-system)
- [Implementation roadmap](#implementation-roadmap)
- [Proposed code and artifact layout](#proposed-code-and-artifact-layout)
- [Leakage-control contract](#leakage-control-contract)
- [Evaluation and reporting protocol](#evaluation-and-reporting-protocol)
- [Risks and mitigations](#risks-and-mitigations)
- [Progress ledger](#progress-ledger)
- [Step-specific development records](#step-specific-development-records)
- [Decision log](#decision-log)
- [Immediate next action](#immediate-next-action)

## Method summary

We will learn a deployable **geo-helpfulness signal** that estimates whether
increasing the influence of geographic information for an individual sample is
more likely to rescue an incorrect baseline prediction than to damage a correct
one.

We will use the signal in two stages:

1. **Selective test-time correction.** A router trained on development data
   computes a geo-helpfulness score at inference. The system preserves the
   existing prediction by default and selectively enhances or chooses geo only
   when the estimated rescue benefit exceeds the estimated harm risk.
2. **Learned gated fusion.** If the router demonstrates positive held-out net
   value, the same signal controls a geo-conditioned correction inside a gated
   fusion model. This model will be compared with the current naive
   representation-concatenation baseline.

The objective is not to give geo more weight everywhere. It is to make a
conservative, evidence-based intervention on samples where geo is likely to
contain useful information that the baseline has failed to exploit.

Here, “test-time” means applying a previously fitted and frozen router. It does
not mean fitting, calibrating, or selecting a threshold from test samples or
test labels.

## Expected outcome

The project will be considered successful if it demonstrates, without any
further use of cleaned-test labels for fitting or quantitative selection, that:

- geo-helpfulness is predictable from deployment-available signals;
- a frozen selective intervention rescues more predictions than it harms on
  held-out development plots;
- the gated model improves on a matched `raw_concat` baseline rather than merely
  adding parameters or using a different training budget;
- improvements remain credible across training seeds and are not confined to a
  few low-support habitats or prediction pairs;
- every intervention and model result is reproducible and auditable from saved
  per-instance outputs, manifests, configs, and checkpoints.

The current cleaned test set is not independent of the research process:
observations from it motivated the hypothesis, architecture, and candidate
signals. It is therefore a **locked exploratory evaluation set**, on which no
further feature, threshold, loss, or hyperparameter choice may be made. An
independent temporal or geographical test set is the required source of strong
confirmatory evidence.

## Core formulation

### Baseline-relative utility

Let:

- \(y\) be the ground-truth habitat;
- \(\hat y_g\) be the geo-only prediction;
- \(\hat y_r\) be the current `raw_concat` prediction;
- \(C_g=\mathbf{1}(\hat y_g=y)\);
- \(C_r=\mathbf{1}(\hat y_r=y)\).

Define the utility of replacing or strengthening the raw-fusion decision with
geo evidence as:

\[
u(x)=C_g-C_r.
\]

| Action state | Raw fusion | Geo only | Utility | Meaning |
|---|---|---|---:|---|
| Rescue | Wrong | Correct | \(+1\) | A geo intervention can correct the baseline |
| Harm | Correct | Wrong | \(-1\) | A geo intervention can break a correct baseline |
| Both correct | Correct | Correct | \(0\) | No accuracy benefit from replacing the baseline |
| Both wrong | Wrong | Wrong | \(0\) | Geo alone cannot rescue the baseline |

Although utility has three values, the first router should preferably predict
the four states separately. This keeps the two different neutral cases
distinguishable while retaining the same decision score.

This utility is the exact value of a **hard geo override**. It is intentionally
conservative, but it is not an exhaustive definition of whether the geo
representation contains useful information. In a `both_wrong` case, a learned
geo-conditioned residual may still correct the prediction to a third class even
though directly selecting the geo-only class would not help. Final
classification loss therefore remains necessary in the gated model; routing
supervision must guide the gate without prohibiting this form of synergy.

### Geo-helpfulness score

Given a deployment-safe feature vector \(s(x)\), the router estimates state
probabilities such as:

\[
q_{+}(x)=P(\text{rescue}\mid s(x)),
\qquad
q_{-}(x)=P(\text{harm}\mid s(x)).
\]

The estimated geo-helpfulness score is:

\[
\hat u(x)=q_{+}(x)-q_{-}(x).
\]

The score is not itself a probability that geo is helpful. It is the estimated
rescue probability minus the estimated harm probability. The intervention rule
is:

\[
\hat u(x)>t,
\]

where \(t\) is selected using grouped development validation only. A higher
threshold makes the policy more conservative.

### What the router may observe

The initial \(s(x)\) should use output-level signals that are available at
deployment:

- image, geo, and raw-fusion predicted classes;
- image–geo agreement and top-3 overlap;
- predicted-class pair;
- confidence, entropy, and top-two margin for each model;
- differences between confidence, entropy, and margin;
- image–geo Jensen–Shannon divergence and total-variation distance;
- probability that each expert assigns to the other expert's predicted class;
- probability and rank of the geo candidate under raw fusion.

Later ablations may add:

- image and geo embedding-density or out-of-distribution scores;
- compact projected embeddings;
- image prediction stability under test-time augmentation or ensembling.

The router must never receive ground truth, correctness, true-class probability,
or true-class NLL. Those quantities may construct targets or diagnostics during
development, but they do not exist at deployment.

## Scope and non-goals

### In scope

- 100 m GSE and habitat-finetuned image features for native L3 classification;
- grouped out-of-fold predictions for `image_only`, `geo_only`, and
  `raw_concat`;
- an interpretable output-level router before a high-capacity gate;
- hard and soft selective test-time correction;
- a scalar-gated geo-residual fusion model;
- overall and per-habitat evaluation with plot-aware uncertainty;
- multi-seed comparison with matched data, class mappings, and budgets.

### Initially out of scope

- using cleaned-test labels for feature selection, calibration, thresholding, or
  hyperparameter tuning;
- end-to-end fine-tuning of the image encoder;
- changing the underlying GSE representation;
- class-specific or feature-wise gates before a scalar gate is validated;
- claiming that native-\(T=1\) confidence values are calibrated;
- treating a routing-oracle ceiling as achievable model performance;
- claiming general habitat effects from one training seed or rare habitats.

## Current evidence and motivation

### Baseline system

The current `raw_concat` architecture concatenates a 1,152-dimensional image
embedding with 64 standardized GSE features and applies a `1216 -> 256 -> 18`
MLP. It has no modality reliability estimate, modality-specific prediction
branch, gate, or loss designed to preserve useful geo evidence.

The cleaned seed-1 final-checkpoint analysis contains 1,347 images from 531
plots. It reports:

| Model | Top-1 accuracy |
|---|---:|
| Image only | 73.13% |
| Geo only | 61.54% |
| Raw concat | 74.39% |

Image and geo disagree on 546 images. Relative to image only, geo is uniquely
correct on 135 images. Raw fusion captures 23 of those opportunities and loses
112; on 103 of the 112 lost cases, fusion follows the image model's wrong class.
The three major grasslands (Improved, Neutral, and Acid Grassland) account for
66 of those 112 lost cases.

These findings show that useful geo information exists and that naive
concatenation often fails to exploit it. They do **not** prove that a deployable
geo-helpfulness signal can be learned. The existing test analysis is
hypothesis-generating evidence only.

### Existing evidence artifacts

- [Single-seed agreement notebook](../../notebooks/multimodal_baseline_agreement_single_seed.ipynb)
- [Agreement helper](../../multimodal/agreement.py)
- [Agreement tests](../../tests/test_multimodal_agreement.py)
- [Final-checkpoint agreement report](../../multimodal_artifacts/reports/cs/gse_100m_cleaned_test/baseline_agreement/gse_100m_train_cleaned_test_epoch50/seed1/summary.md)
- [Per-instance final-checkpoint metrics](../../multimodal_artifacts/analysis/cs/gse_100m_cleaned_test/baseline_agreement/gse_100m_train_cleaned_test_epoch50/seed1/per_instance_metrics_native_t1.parquet)
- [Cleaned-test fusion comparison](../../multimodal_artifacts/reports/cs/gse_100m_cleaned_test/performance_comparison.md)

`oracle_test_best` outputs may be used only as explicitly labelled sensitivity
analysis. The primary method motivation and all future comparisons should use
validation-selected or fixed final checkpoints, never test-selected epochs.

## Proposed system

### Stage 1: frozen-expert selective correction

The first learned system keeps the three expert models frozen:

```text
image features ──► image expert ──┐
geo features   ──► geo expert  ───┼──► deployment-safe signals s(x)
both features  ──► raw fusion  ───┘                 │
                                                   ▼
                                                router
                                                   │
                                      geo-helpfulness score û(x)
                                                   │
                              preserve raw fusion or enhance geo
```

This stage answers the essential scientific question before a more flexible
fusion network is introduced: can observable evidence identify beneficial geo
interventions with positive net value?

### Stage 2: gated geo-residual fusion

If Stage 1 succeeds, project the two representations into comparable spaces:

\[
h_i=\operatorname{LN}(W_i x_i),
\qquad
h_g=\operatorname{LN}(W_g x_g).
\]

Let \(z_r\in\mathbb{R}^{18}\) be the existing raw-fusion logits. Let the gate
produce \(g(x)\in[0,1]\), initially from the validated router signal. Let a
small residual network produce an 18-class correction:

\[
r(x)=R(h_i,h_g)\in\mathbb{R}^{18}.
\]

The final logits are:

\[
z_{\mathrm{final}}=z_r+g(x)r(x).
\]

The residual output should be zero-initialized so that the new model initially
reproduces `raw_concat`. The initial gate should be scalar and conservative.

A candidate training objective is:

\[
\mathcal L=
\mathcal L_{\mathrm{CE}}
+\lambda_r\mathcal L_{\mathrm{route}}
+\lambda_p\mathcal L_{\mathrm{preserve}}
+\lambda_g\,E[g(x)].
\]

- \(\mathcal L_{\mathrm{CE}}\) trains final habitat classification.
- \(\mathcal L_{\mathrm{route}}\) preserves the rescue/harm interpretation of
  the gate.
- \(\mathcal L_{\mathrm{preserve}}\) discourages damaging a correct baseline.
- The optional activation penalty encourages selective rather than universal
  geo intervention.

The exact preservation target and loss weights must be selected on development
data and recorded before test inference.

## Implementation roadmap

Status values are **Complete**, **In progress**, **Planned**, **Blocked**, or
**Deferred**.

### M0 — Baselines, cache, and problem definition

**Status:** Complete

Completed work:

- implemented the optional 1,347-image cleaned test split;
- trained `image_only`, `geo_only`, and `raw_concat` fixed-epoch baselines for
  seeds 1–4;
- cached aligned seed-1 logits and probabilities with provenance;
- reproduced final confusion matrices and saved metrics;
- measured agreement, complementarity, fusion capture, per-habitat F1 flows,
  native-\(T=1\) soft diagnostics, and plot-cluster uncertainty;
- formulated the selective geo-intervention objective.

Acceptance evidence:

- one row per test filename;
- identical class order and test universe across modes;
- finite 18-class outputs;
- exact confusion-matrix reproduction and metric checks.

### M1 — Freeze the experimental protocol

**Status:** In progress

Tasks:

1. Lock the cleaned test set against any further learning or tuning and label it
   exploratory rather than independent.
2. Define deterministic, habitat-stratified, `plot_idx`-grouped **outer**
   development folds shared by all three modes.
3. Within every outer-training partition, define grouped **inner** folds for
   generating expert outputs used to train the router and residual.
4. Require the complete stack to be evaluated on the outer-held-out plots:
   experts, learned feature transforms, calibration, router, threshold, gate,
   and residual must not use those plots during fitting.
5. Freeze fold counts, seeds, class mapping, baseline anchor, primary metrics,
   and go/no-go criteria.
6. Decide whether temperature scaling is part of the first router experiment or
   a predeclared ablation, and define its fold-contained fitting procedure.
7. Define development-only training commands that do not load or evaluate the
   cleaned test set, plus a separate immutable one-shot test evaluator.
8. Define artifact schemas and cache invalidation fingerprints.

Deliverables:

- `fold_assignments.parquet`;
- fold/protocol manifest;
- completed experimental-protocol sub-document;
- leakage tests that fail if plots cross fold boundaries or test rows enter a
  tuning table.

Acceptance criteria:

- every development row belongs to exactly one held-out fold;
- no `plot_idx` crosses the training/held-out boundary for any fold;
- all modes use identical fold membership and class order;
- every outer-validation prediction comes from a complete pipeline fitted only
  on the corresponding outer-training plots;
- test labels and test-derived diagnostics are inaccessible to tuning routines;
- development training cannot emit per-epoch cleaned-test metrics or an
  `oracle_test_best` checkpoint.

### M2 — Generate honest out-of-fold expert outputs

**Status:** Planned

For each outer development fold:

1. reserve the outer-validation plots from the entire fitting pipeline;
2. run grouped inner cross-fitting inside the outer-training partition:
   train `image_only`, `geo_only`, and `raw_concat` on each inner-training
   partition and infer on its inner-held-out plots;
3. concatenate the inner-held-out outputs into the router/residual training
   table for that outer fold;
4. separately fit the three experts on all outer-training plots and infer on the
   outer-validation plots;
5. save logits, native-\(T=1\) probabilities, predictions, identities, labels,
   metadata, outer/inner fold membership, and checkpoint provenance.

This nested design is required for an unbiased estimate of the complete stack.
A single global OOF table followed by router cross-validation is insufficient:
an expert that produced a router-training row may otherwise have trained on the
plots later used as router-validation rows.

After all method choices are frozen, generate one full-development OOF table to
fit the deployable meta-model, refit the experts on all development plots, and
use those frozen artifacts for the locked exploratory test.

Planned artifacts:

- outer-fold `inner_oof_model_outputs.parquet` files;
- outer-fold `outer_holdout_model_outputs.parquet` files;
- final full-development `oof_model_outputs.parquet`;
- `oof_model_outputs_manifest.json`;
- fold-level configs, checkpoints, metrics, and fingerprints;
- an OOF reproduction report.

Acceptance criteria:

- every outer-validation filename appears exactly once across outer folds;
- each inner- and outer-held-out row was predicted by experts that never trained
  on its plot;
- outer-validation plots never influence the corresponding inner experts,
  feature transforms, router, threshold, gate, or residual;
- all three modes share row order and class order;
- all logits are finite and probabilities sum to one;
- OOF—not training-set—performance is reported;
- rebuilding with the same seeds reproduces fold assignments and deterministic
  artifacts within defined tolerance.

### M3 — Build targets and deployment-safe features

**Status:** Planned

Tasks:

1. construct `rescue`, `harm`, `both_correct`, and `both_wrong` targets relative
   to `raw_concat`;
2. retain image-relative geo-exclusive states only as auxiliary diagnostics;
3. implement one allow-listed feature builder for both router training and
   inference;
4. initially use output-level features only;
5. quantify target prevalence overall, per habitat, per plot, and per predicted
   class pair;
6. add an automated forbidden-feature audit.

Planned artifacts:

- `router_dataset.parquet`;
- router feature-schema manifest;
- target-prevalence report;
- feature leakage audit.

Acceptance criteria:

- no router feature requires a true label or correctness at deployment;
- true-class NLL and NLL advantage cannot enter the router matrix;
- feature construction is identical during training and inference;
- undefined signals and missing classes are handled deterministically;
- rescue and harm support is known before model selection.

### M4 — Demonstrate a learnable router

**Status:** Planned

Start with interpretable, regularized models:

- multinomial or paired logistic regression;
- a generalized additive model if available and justified;
- a shallow decision tree or constrained boosted-tree ablation.

Compare with simple policies:

- never intervene;
- always select geo when experts disagree;
- select the model with higher calibrated confidence;
- JSD-only and margin-difference thresholds.

Evaluate the complete stack through the outer grouped folds defined in M1. For
each outer fold:

1. construct router-training data only from inner-OOF expert outputs within the
   outer-training partition;
2. use grouped router validation inside that partition to select router
   hyperparameters and threshold;
3. fit learned feature transforms, probability calibration, and router
   calibration only on the applicable router-training rows;
4. refit the frozen outer-fold router pipeline on all permitted outer-training
   data;
5. apply it once to outputs from experts fitted on outer-training plots and
   evaluated on outer-validation plots.

Aggregate only the outer-held-out decisions to estimate router performance.
The cleaned test set is not involved.

Router evaluation:

- rescue and harm precision–recall AUC;
- multiclass log loss or Brier score;
- probability calibration;
- intervention coverage;
- geo-override precision and rescue recall;
- harmful-override rate;
- rescued count, harmed count, and rescued-minus-harmed count;
- net accuracy gain;
- results by habitat, plot, and predicted-class pair.

Go/no-go criterion:

> Proceed to learned fusion only if a frozen router has positive held-out net
> value and improves on simple routing heuristics. A positive lower uncertainty
> bound is preferred; otherwise the evidence must be labelled inconclusive.

High AUROC alone is not sufficient.

### M5 — Implement selective test-time correction

**Status:** Planned

Freeze the router, its calibration, threshold, and policy before test inference.
Evaluate in this order:

1. **Hard override:** retain raw fusion unless \(\hat u>t\), then select geo.
2. **Soft correction:** combine raw-fusion and geo distributions using a
   router-controlled weight.

For soft correction, separately calibrated distributions or another
scale-compatible combination must be used. Dynamically scaling geo embeddings
inside the existing fixed `raw_concat` head is not valid because that head was
not trained for changing geo scales.

Acceptance criteria:

- zero intervention exactly reproduces raw-fusion predictions;
- every action records its score, threshold, selected policy, and outcome;
- the threshold is selected by development net benefit under a declared harm
  constraint, with conservative tie-breaking;
- held-out gain is not produced solely by one frequent habitat or prediction
  pair;
- runtime and memory overhead are recorded.

### M6 — Implement gated geo-residual fusion

**Status:** Planned

Proposed training stages:

1. freeze the existing experts and validated router; train modality projections
   and the zero-initialized residual;
2. unfreeze the gate with a lower learning rate and add routing/preservation
   supervision;
3. optionally fine-tune limited expert layers only after the controlled model
   succeeds.

The gated model must use the same outer evaluation boundary as the router. For
each outer fold, train the router, projections, gate, and residual using only
the outer-training partition. Use inner-OOF raw logits and router signals when
training downstream components so they do not learn from in-sample expert
predictions. Freeze the complete stack before applying it to outer-validation
plots. If any expert is later fine-tuned jointly, regenerate the fold-contained
targets and repeat the complete nested evaluation.

Only after architecture and hyperparameters are fixed may the deployable model
be fitted from full-development OOF outputs, with final experts refitted on all
development plots for one-shot locked-test inference.

Required engineering checks:

- initialization reproduces raw-fusion logits within tolerance;
- gate values are finite and constrained to \([0,1]\);
- setting the residual to zero reproduces raw fusion;
- checkpoint loading exactly reproduces evaluation;
- existing fusion modes remain backward compatible;
- gate and residual outputs can be inspected per instance.

Required ablations:

- image only, geo only, and raw concat;
- test-time router only;
- gated residual with classification loss only;
- routing-loss and preservation-loss ablations;
- fixed versus learned gate;
- output-only versus embedding-aware gate;
- raw-fusion versus image anchor;
- scalar gate before any class-specific or feature-wise gate;
- matched-parameter or matched-budget controls.

Scientific acceptance criteria:

- positive held-out gain over matched raw concat;
- rescued interventions exceed harmful interventions;
- gate activation is higher for rescue cases than harm cases;
- improvement is not explained only by parameter count or training budget.

### M7 — Multi-seed evaluation and independent confirmation

**Status:** Planned

Compare:

- image only;
- geo only;
- raw concat;
- routing-oracle ceiling, labelled diagnostic;
- heuristic router;
- learned hard and soft test-time correction;
- gated geo-residual fusion.

Report:

- top-1, top-3, weighted F1, macro F1, and MCC;
- per-habitat precision, recall, F1, and support;
- TP rescued/lost and FP introduced/removed;
- rescue, harm, intervention, and net-benefit counts;
- oracle-opportunity recovery;
- gate activation and score distributions;
- calibration and gain-versus-coverage curves;
- compute and memory overhead.

Uncertainty must distinguish:

- paired, habitat-stratified plot-cluster uncertainty within an evaluation set;
- variation across training seeds;
- uncertainty due to low-support habitats.

Acceptance criteria:

- thresholds and hyperparameters are frozen before test inference;
- all comparisons use identical sample universes and class mappings;
- training budgets and seeds are matched or differences are disclosed;
- development runs neither load the cleaned test set nor report its metrics
  during epochs;
- no `oracle_test_best` checkpoint is produced for method development;
- locked-test evaluation is a separate, one-shot command that cannot fit or
  update any artifact;
- no general claim relies on one seed or rare classes;
- an independent temporal or geographical evaluation is attempted before the
  method is described as generally effective.

### M8 — Integration and reproducibility

**Status:** Planned

Deliverables:

- reusable target, feature, router, and policy modules;
- an OOF build command;
- router training and evaluation commands;
- gated-fusion config and runner support;
- unit and integration tests;
- updated pipeline documentation;
- completed step-specific development records;
- an artifact registry with schema versions and fingerprints.

Acceptance criteria:

- a fresh build from configuration succeeds;
- stale caches are rejected when data, folds, configs, checkpoints, class maps,
  or feature schemas change;
- tests cover target identities, feature allow-lists, grouped folds, routing
  policies, zero-residual identity, and checkpoint reproduction;
- every reported result is traceable to a code revision and fingerprinted
  inputs.

## Proposed code and artifact layout

The exact module split remains provisional. Prefer shared functions over
notebook-only logic.

| Path | Proposed responsibility |
|---|---|
| `multimodal/geo_helpfulness.py` | Targets, feature schema, utility score, policies, metrics |
| `multimodal/router.py` | Router models, fitting, calibration, serialization |
| `multimodal/models.py` | Gated geo-residual architecture |
| `multimodal/trainer.py` | New fusion mode and optional auxiliary losses |
| `tools/run_multimodal_geo_helpfulness.py` | OOF generation, router fitting, and evaluation orchestration |
| `configs/multimodal_geo_helpfulness.yaml` | Fold, router, gate, policy, and artifact settings |
| `tests/test_multimodal_geo_helpfulness.py` | Pure target/feature/policy and leakage tests |
| `tests/test_multimodal_gated_fusion.py` | Model identity, training, and checkpoint tests |

Keep development artifacts separate from final test reports. A proposed layout
is:

```text
multimodal_artifacts/
├── analysis/cs/gse_100m/geo_helpfulness/<experiment_tag>/
│   ├── fold_assignments.parquet
│   ├── nested_development_cv/
│   │   └── outer_fold_<k>/
│   │       ├── inner_oof_model_outputs.parquet
│   │       ├── outer_holdout_model_outputs.parquet
│   │       └── manifest.json
│   ├── final_development_fit/
│   │   ├── oof_model_outputs.parquet
│   │   ├── oof_model_outputs_manifest.json
│   │   └── router_dataset.parquet
│   └── router/
└── reports/cs/gse_100m_cleaned_test/geo_helpfulness/<experiment_tag>/
    ├── development_validation/
    └── locked_test/
```

## Leakage-control contract

| Information | Target construction | Router feature use | Model/threshold selection | Locked test reporting |
|---|---:|---:|---:|---:|
| Development ground truth | Yes | Never a feature | Yes, through grouped validation metrics | No |
| Development OOF predictions | Yes | Yes | Yes | No |
| Development true-class NLL | Diagnostic or auxiliary target | Never a feature | Diagnostic only | No |
| Locked-test model outputs | No | Yes, at frozen inference only | No | Yes |
| Locked-test ground truth | No | Never a feature | No | Yes, after predictions and policy are frozen |
| Existing test agreement findings | Hypothesis generation only | No new feature choice | Must not tune choices | Contextual interpretation |

Additional rules:

- all images from the same `plot_idx` remain together;
- outer-validation plots are excluded from every fitted component used to
  predict them, including the expert models;
- temperatures, density estimators, feature scalers, router models, router
  calibration, thresholds, gates, and residuals are fitted inside the
  applicable training fold and then applied to its held-out fold;
- learned transforms must not be fitted once globally before cross-validation;
- prediction-pair or habitat-specific rules require regularization and grouped
  validation;
- the locked test command must load immutable artifacts and expose no fitting
  path;
- development training must not load the cleaned test set, evaluate it per
  epoch, or retain a test-selected checkpoint;
- test evaluation should be run once per frozen method version and logged.

## Evaluation and reporting protocol

### Router feasibility

The principal quantity is intervention utility, not router classification
accuracy:

\[
\Delta N_{\mathrm{correct}}
=N_{\mathrm{rescued}}-N_{\mathrm{harmed}}.
\]

Corresponding net accuracy gain is:

\[
\Delta\mathrm{Acc}
=\frac{N_{\mathrm{rescued}}-N_{\mathrm{harmed}}}{N}.
\]

Always report gain with coverage. A router that intervenes once and succeeds is
not equivalent to one that produces a stable gain across many plots.

### Final classifier comparison

Report overall and per habitat:

- top-1 and top-3 accuracy;
- weighted and macro F1;
- MCC;
- confusion matrix;
- rescue/harm and TP/FP-flow decompositions;
- gate activation and intervention coverage;
- native-\(T=1\) and calibrated diagnostics kept clearly separate.

### Uncertainty

- use paired habitat-stratified resampling of complete `plot_idx` groups for
  current-test uncertainty;
- use the same resamples for every compared model;
- report variation across training seeds separately;
- mark habitats with fewer than 20 images or 10 plots as low support;
- do not interpret a confidence interval over current plots as evidence about
  training-seed or dataset-shift uncertainty.

## Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Test-set adaptation | Optimistic final result | Lock test; tune only on grouped development data; confirm elsewhere |
| Rare rescue class | Router predicts “never intervene” or overfits | PR metrics, class weights, coverage curves, simple models first |
| Plot leakage | Inflated router feasibility | Shared deterministic `plot_idx` folds and automated overlap checks |
| Uncalibrated expert scales | Misleading confidence comparisons or pooling | Development-only temperature scaling and native-\(T=1\) labels |
| Confidently wrong image model | Confidence-only router misses geo rescues | Cross-probabilities, class pair, fusion rank, and reliability features |
| Class-pair overfitting | Apparent gain from sparse pairs | Regularization, grouped validation, minimum support reporting |
| Gate collapse | Gate is always on or always off | Activation monitoring, sparse prior, initialization and loss ablations |
| Expert drift | OOF targets no longer match jointly tuned experts | Freeze experts first; limit later fine-tuning; regenerate targets if needed |
| Extra model capacity | Unfair advantage over raw concat | Matched-parameter/budget controls and residual ablations |
| Nested stacking compute cost | Slow iteration and large artifacts | Cache outer/inner fold outputs with fingerprints; reuse immutable logits |
| Independent dataset unavailable | Weak generalization claim | Use multi-seed grouped evidence and explicitly limit conclusions |

## Progress ledger

| Milestone | Status | Evidence or blocker | Next action |
|---|---|---|---|
| M0 Baselines and diagnostic evidence | Complete | Agreement cache, notebook, report, and reproduction checks | Preserve as immutable motivation |
| M1 Experimental protocol | In progress | Method anchor and leakage principles documented here | Freeze nested grouped folds and artifact contract |
| M2 OOF expert outputs | Planned | No OOF development cache exists | Implement deterministic fold assignments after M1 |
| M3 Targets and router features | Planned | Test diagnostics exist, but no training dataset | Implement only from validated OOF outputs |
| M4 Router feasibility | Planned | No router has been trained | Begin with output-only logistic router |
| M5 Test-time correction | Planned | No frozen score/threshold policy exists | Proceed only after positive held-out router value |
| M6 Gated residual fusion | Planned | Current model is naive concatenation | Proceed only after router go/no-go passes |
| M7 Robust confirmation | Planned | Single-seed agreement evidence only | Run matched seeds and seek independent data |
| M8 Integration and reproducibility | Planned | No new-method runner or tests | Integrate alongside each implementation milestone |

## Step-specific development records

Create each sub-document when its milestone begins. Each record should contain
the frozen decisions, implementation notes, commands, artifacts, validation
results, unresolved issues, and completion evidence for that step.

| Planned record | Milestone | Status |
|---|---|---|
| `01_experimental_protocol.md` | M1 | Not created |
| `02_oof_expert_outputs.md` | M2 | Not created |
| `03_targets_and_features.md` | M3 | Not created |
| `04_router_feasibility.md` | M4 | Not created |
| `05_test_time_correction.md` | M5 | Not created |
| `06_gated_fusion.md` | M6 | Not created |
| `07_evaluation_and_confirmation.md` | M7 | Not created |

Do not add broken Markdown links. Replace a filename with a link only when the
corresponding record exists.

## Decision log

| ID | Date | Status | Decision | Rationale and consequence |
|---|---|---|---|---|
| D001 | 2026-08-24 | Confirmed | Learn intervention utility rather than generic geo correctness | The action must rescue more baseline errors than it creates |
| D002 | 2026-08-24 | Confirmed | Use `raw_concat` as the primary intervention anchor | The method is intended to improve the deployed naive fusion baseline |
| D003 | 2026-08-24 | Confirmed | No further selection or tuning may use the locked cleaned test | Its existing findings already shaped the hypothesis, so it remains exploratory rather than independent |
| D004 | 2026-08-24 | Proposed | Predict four correctness states and compute \(\hat u=q_+-q_-\) | Separates both-correct from both-wrong neutral cases |
| D005 | 2026-08-24 | Proposed | Start with an output-only interpretable router | Tests signal feasibility before adding model capacity |
| D006 | 2026-08-24 | Proposed | Evaluate hard override before soft pooling | Provides the most auditable intervention baseline |
| D007 | 2026-08-24 | Proposed | Start gated fusion with a scalar, raw-fusion-anchored residual | Conservative and less prone to overfitting than class-wise gates |
| D008 | 2026-08-24 | Open | Outer/inner fold counts, fold seeds, router seeds, and threshold objective | Must be frozen in M1 |
| D009 | 2026-08-24 | Open | Native-\(T=1\) first router versus calibrated first router | Include calibration only with fold-contained fitting and held-out application |
| D010 | 2026-08-24 | Open | Availability and definition of an independent confirmation set | Determines the strength of final generalization claims |

## Immediate next action

Complete M1 by creating `01_experimental_protocol.md` and freezing:

1. the development-table identity and fingerprints;
2. nested outer/inner grouped fold generation and seeds;
3. the four-state target contract;
4. the first output-level feature allow-list;
5. fold-contained transform, calibration, router, and threshold rules;
6. development-only training and one-shot locked-test entry points;
7. the exact go/no-go criterion for starting gated-fusion development.

No router or gate should be trained before those choices are recorded.

## Change log

| Date | Change |
|---|---|
| 2026-08-24 | Created the living method specification and implementation roadmap |

## Related pipeline files

- [Multimodal pipeline README](../../multimodal/README.md)
- [Current fusion models](../../multimodal/models.py)
- [Current multimodal trainer](../../multimodal/trainer.py)
- [Agreement analysis helper](../../multimodal/agreement.py)
- [Cleaned-test dataset config](../../configs/multimodal_cs_geo_100m_cleaned_test.yaml)

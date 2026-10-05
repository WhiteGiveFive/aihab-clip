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
| Current method status | M1–M4 complete; primary M4 feasibility is no-go, with common C=0.1 and never-intervene fallback; ablations are inconclusive and are not promoted; M5/M6 not started |
| Last updated | 2026-10-01 |

The latest completed exploration compares pretrained and habitat-adapted image
geometry for selective geo correction; see
[record 21](21_embedding_geometry_correction.md). Adaptation improves the
cleaned-test correction point estimates but weakens validation rescue/harm
ranking. It does not establish an independent gain or change the M4 decision.
The [current artifact organization](#current-experimental-artifact-organization)
now covers all nine geometry studies alongside the earlier router studies.
The [project-wide naming index](22_project_structure.md) covers every milestone
and exploratory study, with a consistent geometry command interface.

## Contents

- [Method summary](#method-summary)
- [Expected outcome](#expected-outcome)
- [Core formulation](#core-formulation)
- [Scope and non-goals](#scope-and-non-goals)
- [Current evidence and motivation](#current-evidence-and-motivation)
- [Proposed system](#proposed-system)
- [Implementation roadmap](#implementation-roadmap)
- [Proposed code and artifact layout](#proposed-code-and-artifact-layout)
- [Current experimental artifact organization](#current-experimental-artifact-organization)
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
2. **Learned gated fusion.** If the router passes the predeclared internal
   development gate, the same signal controls a geo-conditioned correction
   inside a gated fusion model. This model will be compared with the current
   naive representation-concatenation baseline.

The objective is not to give geo more weight everywhere. It is to make a
conservative, evidence-based intervention on samples where geo is likely to
contain useful information that the baseline has failed to exploit.

Here, “test-time” means applying a previously fitted and frozen router. It does
not mean fitting, calibrating, or selecting a threshold from test samples or
test labels.

## Expected outcome

Protocol v1 demonstrates internal feasibility if, without any further use of
cleaned-test labels for fitting or quantitative selection, it shows that:

- geo-helpfulness is predictable from deployment-available signals;
- a frozen selective intervention satisfies predeclared rescue-versus-harm
  criteria on the fixed development-validation selection set;
- the gated model improves on a matched `raw_concat` baseline rather than merely
  adding parameters or using a different training budget;
- improvements remain credible across training seeds and are not confined to a
  few low-support habitats or prediction pairs;
- every intervention and model result is reproducible and auditable from saved
  per-instance outputs, manifests, configs, and checkpoints.

The development-validation result is conditional on one fixed plot split and
is used to select router hyperparameters and threshold. It is therefore
model-development evidence, not an unbiased estimate of the selected policy.
General empirical success requires independent temporal or geographical
confirmation.

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

The conceptual \(t=+\infty\) fallback means never intervene. Artifacts encode
it as a typed `never_intervene` threshold specification with a null numeric
value, not non-standard JSON infinity; finite thresholds use a canonical
IEEE-754 hexadecimal value and strict `>` comparison.

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
- learned TCP estimates from frozen-expert hidden representations (the separately
  declared [initial ConfidNet study](12_confidnet_initial_experiment.md) evaluates
  confidence estimation and a direct comparator; the subsequently approved
  [OOF ConfidNet study](13_confidnet_oof_proposal.md) compares 25 numerical
  summaries against those same summaries plus three estimated TCPs);
- image prediction stability under test-time augmentation or ensembling.

The router must never receive ground truth, correctness, true-class probability,
or true-class NLL. Those quantities may construct targets or diagnostics during
development, but they do not exist at deployment.

An estimated TCP produced by a fitted confidence network is deployment-available
and differs from the label-derived true TCP target. The initial feature allow-list
remains unchanged. The separate OOF study explicitly uses estimates that are
OOF for the experts and in-sample for ConfidNet when fitting the router; it does
not claim fully cross-fitted router inputs.

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
- joint end-to-end fine-tuning of the image encoder with the router or gate;
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

**Status:** Complete

Tasks:

1. Lock the cleaned test set against any further learning or tuning and label it
   exploratory rather than independent.
2. Materialize one deterministic, habitat-aware, `plot_idx`-grouped split of
   the 4,200-image development universe into **development-train** and
   **development-validation**. The frozen allocation is 1,300/325 plots;
   image counts are 3,378/822 because plots are indivisible.
3. Within development-train, define four grouped cross-fitting folds shared by
   `image_only`, `geo_only`, and `raw_concat`. These folds generate honest
   expert outputs for router targets and features.
4. Require validation predictions to come from experts, geo standardization,
   calibration, feature transforms, and a router fitted without validation
   plots. Validation labels may select declared router hyperparameters and the
   hard-override threshold, but may not enter router features.
5. Freeze the final-refit recipe and its component allow-lists. Protocol v1
   selects fold-contained, vision-only SigLIP2 adaptation: every producer starts
   from the same pinned pretrained checkpoint, unlocks the final 11 vision
   groups, freezes the text tower, and uses the fixed 18-class prompt objective
   for five epochs. It deliberately retains the legacy OpenCV BGR/439 input
   path. After validation selects router hyperparameters and the threshold, all
   router-side state is immutable. Refit only the adapted image encoder, three
   expert heads, and geo standardization on all development plots. In-sample
   predictions from those final experts may not enter a router-side fit or
   update.
6. Freeze the split ratio and assignment algorithm, four-fold seed, training
   seeds, fixed 18-class mapping, baseline anchor, primary metrics, threshold
   objective, and development go/no-go criteria.
7. Use one scalar temperature per expert mode and training seed, fitted once
   from development-train OOF logits; retain native `T=1` as a descriptive
   ablation and use no post-hoc router-output calibrator in protocol v1.
8. Define development-only commands that cannot load or evaluate the cleaned
   test set, plus a separate immutable one-shot exploratory-test evaluator.
9. Define artifact schemas, provenance requirements, and content-based cache
   invalidation fingerprints.

Deliverables:

- `development_assignments.parquet` with development role and train-OOF fold;
- split-balance report, label-blind locked-test identity manifest, resolved
  protocol, and protocol manifest;
- raw-development-image content fingerprint and pinned encoder/tokenizer
  provenance;
- completed `01_experimental_protocol.md`;
- leakage tests that fail if plots cross fold boundaries or test rows enter a
  tuning table.

Acceptance criteria:

- every development row belongs to exactly one of development-train or
  development-validation;
- no `plot_idx` crosses that boundary, and each development-train plot belongs
  to exactly one OOF fold;
- all modes use identical fold membership and class order;
- every train-OOF prediction excludes its prediction plot from expert fitting;
- every development-validation expert prediction comes from the complete
  expert pipeline fitted only on development-train plots;
- expert temperatures, router feature transforms/vocabularies, router
  coefficients, policy, and threshold are unchanged after validation selection;
- test labels and test-derived diagnostics are inaccessible to tuning routines;
- development training cannot emit per-epoch cleaned-test metrics or an
  `oracle_test_best` checkpoint.

### M2 — Generate honest out-of-fold expert outputs

**Status:** Complete

Use the fixed development split from M1:

1. reserve development-validation plots from expert and router fitting;
2. run four-fold grouped cross-fitting inside development-train: train
   `image_only`, `geo_only`, and `raw_concat` on three folds and infer on the
   fourth;
3. concatenate the held-out predictions into
   `development_train_oof_model_outputs.parquet`;
4. separately fit the three experts on all development-train plots and infer
   once on development-validation;
5. save label-blind logits, native-\(T=1\) probabilities, predictions,
   identities, deployment metadata, split/fold membership, training and
   prediction plot hashes, and checkpoint provenance. M3 joins development
   labels separately to construct targets.

The training OOF table supplies honest router-training rows. The separate
development-validation outputs select the declared router hyperparameters and
threshold. Because the same validation set performs that selection, its chosen
score is model-development evidence rather than an unbiased estimate of the
selected complete stack.

After selection, freeze the chosen expert temperatures, router feature state,
router coefficients, and threshold specification. Refit only `expert_refit_state`
once on all development plots. This preserves the selected scoring function and
threshold coordinate system, although stronger final experts may shift the
router's input and score distributions. Never use the final experts' in-sample
development predictions to refit router targets, temperatures, transforms,
coefficients, or the threshold. A future protocol may predeclare a
full-development meta-model refit, but it must also predeclare how a threshold
is re-estimated and must not reuse the validation score as honest evidence.

The deferred higher-evidence alternative is five outer plot-grouped folds with
four inner folds inside each outer-training partition. It requires a separate
protocol ID, artifact namespace, and complete rerun; it is not part of the
simplified router protocol and its terminology must not be applied to these
artifacts.

Completed M2 artifacts:

- four train-fold `heldout_model_outputs.parquet` files per seed (16 total);
- `development_train_oof_model_outputs.parquet`;
- `development_validation_model_outputs.parquet`;
- output manifests, fold-level configs, checkpoints, metrics, and fingerprints;
- an OOF reproduction report.

The final full-development expert bundle and frozen router bundle are not M2
artifacts. They are created only after the later router-selection milestones
freeze temperatures, router feature state, coefficients, policy, and threshold.

Acceptance criteria:

- every development-train filename appears exactly once across OOF predictions;
- every OOF row was predicted by experts that never trained on its plot;
- development-validation plots never influence the train OOF experts, learned
  transforms, calibrators, or routers fitted before validation selection;
- every validation row is predicted once by experts fitted only on
  development-train;
- all three modes share row order and class order;
- all logits are finite and probabilities sum to one;
- OOF—not training-set—performance is reported;
- rebuilding with the same seeds reproduces fold assignments and deterministic
  artifacts within defined tolerance.

### M3 — Build targets and deployment-safe features

**Status:** Complete as of 2026-09-01

Tasks:

1. construct `rescue`, `harm`, `both_correct`, and `both_wrong` targets relative
   to `raw_concat`, retaining every seed realization under the combined key
   `(row_uid, training_seed)` without averaging, voting, or deduplication;
2. retain image-relative geo-exclusive states only as auxiliary diagnostics;
3. implement one stateless, allow-listed 30-feature builder for both router
   training and inference, requiring explicitly scalar-temperature-calibrated
   18-way probability matrices;
4. freeze the semantic feature schema and M4's later deterministic expansion to
   25 scaled plus 702 fixed-vocabulary one-hot columns (727 `float64` columns);
5. quantify target prevalence from development-train OOF rows overall, per
   seed, per habitat, per plot, and per predicted class pair; do not open
   development-validation outputs or use validation prevalence to revise the
   feature schema;
6. add an automated forbidden-feature and artifact-role audit;
7. seal and revalidate the real M3 bundle as an additive child of immutable M1
   and M2.

M3 artifacts:

- `router/targets_and_feature_contract/router_targets.parquet`;
- `router/targets_and_feature_contract/router_feature_schema.json`;
- `router/targets_and_feature_contract/target_prevalence.json`;
- `router/targets_and_feature_contract/feature_leakage_audit.json`;
- `router/targets_and_feature_contract/manifest.json`.

Temperature fitting, calibrated feature-row materialization, learned feature
transforms, and `router_dataset.parquet` belong to M4. M2's native-`T=1`
probabilities remain descriptive and are rejected by the primary M3 builder.

Acceptance criteria:

- no router feature requires a true label or correctness at deployment;
- true-class NLL and NLL advantage cannot enter the router matrix;
- feature construction is identical during training and inference;
- the combined target table contains all 13,512 records in canonical
  `(row_uid, training_seed)` order, with 3,378 records for each seed;
- undefined signals, probability ties, zero probabilities, and missing local
  categories are handled deterministically under the fixed 18-class ontology;
- rescue and harm support is known before model selection;
- a valid rerun reuses the sealed bundle, while tampering and stale lineage fail
  closed without overwrite;
- no M3 artifact contains calibrated feature rows or `router_dataset.parquet`.

### M4 — Demonstrate a learnable router

**Status:** Complete — internal feasibility **no-go**. Calibrated dataset preparation,
primary/ablation candidate fitting, all 48 sealed validation prediction sets,
regularization/threshold selection, comparator/subgroup/bootstrap evaluation,
access audit, and frozen-state/selection-receipt publication are complete.
The primary selects common `C=0.1`; none of its 919 threshold candidates meets
all guardrails, so the frozen policy is `never_intervene`. The two predeclared
ablations are inconclusive and are not promoted. M5/M6 have not started.
See [the full M4 feasibility record](04_router_feasibility.md) for results,
the additive receipt-schema lookup correction, fingerprints, verification,
and the authoritative `run_multimodal_geo_helpfulness_m4_finalize.py` commands.

All 48 candidates converged under the
user-approved, versioned 5,000-iteration budget amendment; the original
protocol configuration and partial 1,000-iteration run remain unchanged.
All candidate validation predictions and OOF heuristic thresholds are sealed;
the fixed primary seed-1, `C=0.01` preview scores 585/822 = 71.17% four-state
accuracy. See [the validation-preview record](04_router_validation_preview.md)
for access boundaries, the versioned storage-dtype scorer correction, and
reproducible commands. See [the candidate-training record](04_router_candidate_training.md)
for the declared study, amendment, trained artifacts, verification evidence,
manual reproduction commands, and label-blind inference handoff. See
[the dataset preparation record](04_router_dataset_preparation.md) for code,
commands, schemas, temperatures, validation evidence, and the observed
intermittent native-runtime caveat.

The primary router is a four-state, L2-regularized multinomial logistic model.
Its declared regularization grid is the only primary model hyperparameter
selected on development-validation. A generalized additive model, shallow tree,
or constrained boosted tree may be a separately labelled, predeclared ablation;
validation results may not promote an ablation to the primary method without a
new protocol version.

Compare with simple policies:

- never intervene;
- always select geo when experts disagree;
- select the model with higher calibrated confidence;
- JSD-only and margin-difference thresholds.

The principal heuristic is higher calibrated confidence: on a raw-versus-geo
disagreement, choose geo exactly when its maximum calibrated probability
exceeds raw fusion's. Never-intervene and always-geo-on-disagreement are fixed
comparators. JSD-only and margin-difference policies are secondary; select and
freeze their threshold specifications from development-train OOF rows with the
same utility and guardrail rule before any validation scoring.

Use the single grouped development split defined in M1:

1. construct router-training data only from the four-fold OOF expert outputs
   within development-train;
2. fit the 12 scalar expert temperatures (three modes by four seeds) once from
   authoritative OOF logits, derive calibrated probabilities, invoke M3's
   unchanged stateless feature builder, materialize `router_dataset.parquet`,
   fit learned feature transforms, and fit each declared router-regularization
   candidate only on those OOF rows; no separate post-hoc router calibrator is
   used in protocol v1;
3. fit the three experts separately on all development-train plots and apply
   each frozen candidate once to development-validation outputs;
4. in a fit-capable but label-blind phase, seal every candidate's validation
   four-state probabilities and score;
5. in a separate scoring-only phase with no fitting imports or APIs, open
   validation labels and select router regularization by four-state log loss;
6. select the hard-override threshold specification from that chosen router's
   validation scores using the predeclared net-utility objective, coverage and
   harm constraints, and deterministic tie-breaking;
7. seal the complete `router_frozen_state` without refitting it on validation
   rows.

For each `C`, fit one router per expert seed using the already-frozen
seed-specific temperature scaler. Select one common `C` by mean validation
four-state log loss across seeds 1–4. Then select one common threshold
specification by mean validation net utility across the four seed-specific
policies under the frozen support and harm constraints. Freeze all four
seed-specific checkpoints together; never choose a best seed.

The cleaned test set is not involved in development. Because
development-validation selects both the router hyperparameters and threshold,
the selected validation result is an internal feasibility result, not an
unbiased estimate of the selected policy's generalization performance.

Router evaluation:

- rescue and harm precision–recall AUC;
- multiclass log loss or Brier score;
- expert-probability calibration diagnostics;
- intervention coverage;
- geo-override precision and rescue recall;
- harmful-override rate;
- rescued count, harmed count, and rescued-minus-harmed count;
- net accuracy gain;
- results by habitat, plot, and predicted-class pair.

Go/no-go criterion:

> Proceed to learned fusion only if the frozen router satisfies the predeclared
> validation net-value, harm, coverage, multi-seed, and heuristic-comparison
> criteria. This is a development gate, not confirmatory evidence.

Any interval calculated on the same validation rows is descriptive and
conditional on model and threshold selection. Strong evidence requires a new
independent evaluation set; the already explored cleaned test cannot supply it.

High AUROC alone is not sufficient.

### M5 — Implement selective test-time correction

**Status:** Deferred after the completed M4 primary no-go; not started.

Protocol v1 freezes the complete `router_frozen_state` and authorizes only the
hard-override policy before test inference:

1. **Hard override:** retain raw fusion unless \(\hat u>t\), then select geo.
2. **Soft correction, deferred:** combining raw-fusion and geo distributions
   introduces a new mapping or weight. It requires a separately versioned
   development-selection and freezing procedure and cannot silently reuse the
   hard-policy threshold or enter the protocol-v1 locked-test bundle.

For soft correction, separately calibrated distributions or another
scale-compatible combination must be used. Dynamically scaling geo embeddings
inside the existing fixed `raw_concat` head is not valid because that head was
not trained for changing geo scales.

Acceptance criteria:

- zero intervention exactly reproduces raw-fusion predictions;
- every action records its score, threshold, selected policy, and outcome;
- the threshold is selected by development net benefit under a declared harm
  constraint, with conservative tie-breaking;
- validation-selected gain is not produced solely by one frequent habitat or
  prediction pair;
- runtime and memory overhead are recorded.

### M6 — Implement gated geo-residual fusion

**Status:** Deferred after the completed M4 primary no-go; not started.

Proposed training stages:

1. freeze the existing experts and validated router; train modality projections
   and the zero-initialized residual;
2. unfreeze only the newly introduced gated-model parameters with a lower
   learning rate and add routing/preservation supervision; the Stage-1 router
   remains a frozen input or teacher;
3. optionally fine-tune limited expert layers only after the controlled model
   succeeds.

M1 freezes the simplified protocol for router development only; it does not yet
authorize a full-development gate/residual refit. Before M6 begins, its
step-specific protocol must define leakage-safe gate-training inputs, including
router-level cross-fitted signals rather than scores from a router fitted on the
same rows. It must also define any final all-development OOF and refit recipe.

The initial M6 development boundary remains development-train versus
development-validation unless a new protocol version introduces nested
evaluation or new independent data. Validation may select declared architecture
and loss hyperparameters, but its resulting score remains adaptively reused
model-development evidence because it already selected the router. If an expert
is jointly fine-tuned, regenerate all affected train-OOF targets and downstream
signals under the same plot-exclusion rules.

D011's Stage-1 `router_frozen_state` remains immutable throughout M6 unless a
new protocol explicitly supersedes it. Only new projections, gate, residual,
expert state, and raw-input preprocessing may follow the later M6 fitting
contract.

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

- positive validation-selected gain over matched raw concat, explicitly
  labelled internal development evidence;
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
- the unmeasured sensitivity to the single fixed development split;
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

**Status:** Partially delivered through M4: milestone runners, immutable
artifacts, reproduction checks and focused tests are implemented. Broader
method integration remains deferred.

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
- tests cover target identities, feature allow-lists, the grouped role boundary,
  train cross-fit folds, routing policies, zero-residual identity, and
  checkpoint reproduction;
- every reported result is traceable to a code revision and fingerprinted
  inputs.

## Proposed code and artifact layout

The exact module split remains provisional. Prefer shared functions over
notebook-only logic.

| Path | Proposed responsibility |
|---|---|
| `multimodal/geo_helpfulness_protocol.py` | Development identities, class map, grouped assignments, manifests, fingerprints, and boundary validators |
| `multimodal/geo_helpfulness_locked_eval.py` | Capability-limited locked prediction/scoring boundary exercised on synthetic fixtures in M1 |
| `multimodal/geo_helpfulness_oof.py` | Immutable M2 expert producers, output validation, and aggregation |
| `multimodal/geo_helpfulness_targets_features.py` | M3 seed-specific targets, pure semantic feature builder, frozen schema, reports, leakage audit, and immutable bundle validation |
| `multimodal/router.py` | Router models, fitting, calibration, serialization |
| `multimodal/models.py` | Gated geo-residual architecture |
| `multimodal/trainer.py` | New fusion mode and optional auxiliary losses |
| `tools/run_multimodal_geo_helpfulness.py` | Frozen M1 protocol build and validation |
| `tools/run_multimodal_geo_helpfulness_m2.py` | M2 per-seed expert execution and four-seed aggregation |
| `tools/run_multimodal_geo_helpfulness_m3.py` | CPU-only M3 bundle build and validation |
| `configs/multimodal_geo_helpfulness.yaml` | Fold, router, gate, policy, and artifact settings |
| `tests/test_multimodal_geo_protocol.py` | Deterministic assignments, manifests, denylist, cache, and command-boundary tests |
| `tests/test_multimodal_geo_targets_features.py` | Pure M3 target, feature, schema, prevalence, and leakage tests |
| `tests/test_multimodal_geo_m3_artifacts.py` | M3 lineage, publication, reuse, tamper, and command-boundary tests |
| `tests/test_multimodal_gated_fusion.py` | Model identity, training, and checkpoint tests |

Keep development artifacts separate from final test reports. A proposed layout
is:

```text
multimodal_artifacts/
├── locked_test_registry/cs/gse_100m_cleaned_test/
│   ├── active_snapshot.json
│   └── <identity_projection_sha256>/
│       └── locked_test_identity_manifest.json
├── analysis/cs/gse_100m/geo_helpfulness/<protocol_id>/
│   ├── protocol/
│   │   ├── development_assignments.parquet
│   │   ├── split_balance.csv
│   │   ├── resolved_protocol.yaml
│   │   ├── locked_test_snapshot_ref.json
│   │   └── protocol_manifest.json
│   ├── development_train_oof/
│   │   ├── development_train_oof_model_outputs.parquet
│   │   └── fold_<k>/
│   │       ├── heldout_model_outputs.parquet
│   │       └── manifest.json
│   ├── development_validation/
│   │   ├── development_validation_model_outputs.parquet
│   │   ├── router_candidate_validation_predictions.parquet
│   │   └── development_selection_receipt.json
│   ├── final_development_fit/
│   │   ├── expert_bundle_manifest.json
│   │   ├── frozen_router_bundle_manifest.json
│   │   └── composite_inference_bundle_manifest.json
│   └── router/
│       ├── targets_and_feature_contract/       # M3
│       │   ├── router_targets.parquet
│       │   ├── router_feature_schema.json
│       │   ├── target_prevalence.json
│       │   ├── feature_leakage_audit.json
│       │   └── manifest.json
│       ├── router_dataset.parquet               # M4 dataset preparation
│       ├── router_dataset_audit.parquet
│       ├── expert_temperatures.json
│       ├── router_feature_transform.json
│       └── router_dataset_manifest.json
└── reports/cs/gse_100m_cleaned_test/geo_helpfulness/<evaluation_event_id>/
    └── locked_test/
```

### Current experimental artifact organization

Completed experiments use the same versioned artifact root:
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/`.
The protocol tree above describes the original milestone workflow; separately
approved router, confidence and geometry experiments each have their own
versioned study directory under this root.

| Location | Responsibility |
|---|---|
| `dev_plans/reliability_aware_geo_fusion/README.md` | Living method status, decisions and study index |
| Numbered records `01_…md` through `22_…md` in this directory | Development decisions, protocols, results interpretation, commands and artifact links |
| `configs/multimodal_geo_*.json` / `.yaml` | Declared experiment settings and artifact paths |
| `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/<study_id>/` | Generated results, predictions, metadata, selections, manifests, receipts, figures and scored reports |
| `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_geometry_v1/embeddings/` | Shared adapted reference/query embedding caches and extraction provenance |
| `multimodal_artifacts/locked_test_registry/cs/gse_100m_cleaned_test/events/` | Append-only cleaned-test reservations and completed score events |
| Dated directories under `dev_plans/reliability_aware_geo_fusion/reports/` | Historical presentation paths; geometry entries are compatibility directory links to canonical bundles |

All nine geometry bundles were relocated on **2026-10-01**, including the
initial embedding plots, disagreement diagnostics, fixed correction rules,
reference-bank and neighborhood comparisons, harm-constrained thresholds, and
adapted-encoder comparison. The [complete inventory and findings](21_embedding_geometry_correction.md)
and [machine-readable layout catalog](../../configs/multimodal_geo_geometry_layout_v1.json)
identify each canonical directory and original entrypoint.

The relocation preserves the sealed bundles' internal layouts and every
original byte. Directory aliases retain paths recorded in source hashes and
registry events; date-name aliases in the artifact root also preserve relative
links between the original reports. These aliases are compatibility paths,
not additional result copies or new experiments. The
[migration manifest](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_layout_v1/migration_manifest.json)
records the original and canonical filenames, hashes, sizes, modification times,
source bindings and unchanged registry snapshot.

For new experiments, write outputs to a **new versioned study directory** under
this artifact root and pass that path explicitly to the runner. Keep the
tracked development record and README links here. Reuse the established
`multimodal/` implementation and `tools/` entrypoint convention for new code;
existing geometry implementations remain at their sealed paths because the
completed studies fingerprint those files. Do not rewrite old receipts,
source paths or code to make them appear newly produced.

The [geometry reproduction guide](21_embedding_geometry_correction.md#reproducing-the-geometry-experiments-and-results)
records environment/input requirements, exact canonical commands, expected
metrics, and the distinction between saved-run verification and recomputation.
It also documents the current limits of a fresh cleaned-test rebuild.

The [project-wide study catalog](../../configs/multimodal_geo_helpfulness_studies_v1.json)
now maps every milestone, study continuation, review directory and historical
naming exception. The canonical geometry runner is
[`run_multimodal_geo_helpfulness_geometry.py`](../../tools/run_multimodal_geo_helpfulness_geometry.py),
backed by [`multimodal/geo_helpfulness_geometry.py`](../../multimodal/geo_helpfulness_geometry.py).
Its subcommands replace the separate `plot_*`, `check_*`, `evaluate_*` and
`extract_*` names as the preferred public interface. The original scripts are
retained as fingerprinted implementations. ConfidNet guardrail and joint-TCP
review helpers also have canonical names; see the
[complete naming contract](22_project_structure.md).

Verify the current organization without fitting, rescoring or repeating OOF:

```bash
python tools/run_multimodal_geo_helpfulness_geometry.py validate
python tools/check_multimodal_geo_helpfulness_studies.py
```

## Leakage-control contract

| Information | Target construction | Router feature use | Model/threshold selection | Locked test reporting |
|---|---:|---:|---:|---:|
| Development-train ground truth | Yes, joined after label-blind OOF prediction | Never a feature | Yes, for fitting and training diagnostics | No |
| Development-train OOF predictions | Yes | Yes | Yes | No |
| Development-validation ground truth | Yes, for scoring-only four-state targets after predictions are sealed | Never a feature | Yes, only to select declared candidates and threshold | No |
| Development-validation predictions | Yes, combined with validation labels for scoring-only targets | Yes, with frozen feature builder | Yes | No |
| Development true-class NLL | Diagnostic or auxiliary target on train OOF only | Never a feature | Diagnostic only | No |
| Locked-test model outputs | No | Yes, at frozen inference only | No | Yes |
| Locked-test ground truth | No | Never a feature | No | Yes, after predictions and policy are frozen |
| Existing test agreement findings | Hypothesis generation only | No new feature choice | Must not tune choices | Contextual interpretation |

Additional rules:

- `router_frozen_state` comprises expert temperature scalers, router feature
  schema, numeric feature scalers/imputers/density estimators, categorical
  vocabularies and mappings, router coefficients/checkpoints, any declared
  router-output calibrator, policy, and threshold specification;
- `expert_refit_state` comprises the fold-contained adapted image-encoder
  weights, the three expert heads, and explicitly named raw-input preprocessing
  such as geo mean and standard deviation; the pinned externally pretrained
  encoder is the immutable initialization for every producer fit;
- all images from the same `plot_idx` remain together;
- development-validation plots are excluded from every expert, temperature,
  feature transform, and router fit used to produce selection predictions;
- every development-train OOF row is produced by experts whose fitting plots
  exclude that row's plot;
- temperatures, density estimators, feature scalers, and router models used for
  selection are fitted on development-train OOF rows and then applied unchanged
  to development-validation; protocol v1 has no post-hoc router calibrator;
- validation labels may score declared candidates and choose the threshold but
  may never enter a deployment feature or fitted router coefficient;
- candidate validation prediction runs are fit-capable but label-blind; a
  separate scoring-only process with no fitting or selection-update APIs opens
  validation labels and seals the selection receipt;
- after the selection receipt is sealed, `router_frozen_state` cannot be fit,
  updated, rebuilt, or expanded from any labelled or unlabelled data;
- after that receipt, validation labels may enter only the final expert-loss
  trainer for `expert_refit_state`; they may never enter router targets,
  temperatures, router transforms, coefficients, calibration, or threshold
  refitting;
- expert/upstream state used to produce a train-OOF row is fold-local;
  router-side transforms are fitted once on the assembled development-train OOF
  table and then frozen;
- prediction-pair or habitat-specific rules require regularization and grouped
  validation;
- the cleaned-test identity projection is sealed once in a protocol-independent
  snapshot registry; each protocol references the active snapshot hash and
  development commands may read only that label-blind manifest for denylist
  checks, never the underlying test table, labels, features, or diagnostics;
- the locked test command must load immutable artifacts and expose no fitting
  path;
- locked scoring is the only test-label reader and has a fixed output allow-list
  with no fitting, recalibration, threshold search, alternative-policy, or
  scientific-override capability;
- development training must not load the cleaned test set, evaluate it per
  epoch, or retain a test-selected checkpoint;
- selection and test evaluation receipts are exclusive-create artifacts. The
  global test-event registry permits one score event per frozen composite
  bundle; a different protocol/bundle creates a new append-only event marked as
  adaptive reuse, never overwrites history, and remains exploratory.

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

- use paired habitat-stratified resampling of complete `plot_idx` groups within
  development-validation and the locked test separately;
- use the same resamples for every compared model;
- report variation across training seeds separately;
- label validation intervals descriptive and selection-conditional; they do
  not measure hyperparameter-selection or fixed-split uncertainty;
- mark habitats with fewer than 20 images or 10 plots as low support;
- do not interpret a confidence interval over current plots as evidence about
  training-seed or dataset-shift uncertainty.

## Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Test-set adaptation | Optimistic final result | Lock test; tune only on grouped development data; confirm elsewhere |
| Rare rescue class | Router predicts “never intervene” or overfits | Regularization, PR metrics, coverage constraints and curves, simple models first |
| Plot leakage | Inflated router feasibility | Shared deterministic `plot_idx` folds and automated overlap checks |
| Uncalibrated expert scales | Misleading confidence comparisons or pooling | Development-only temperature scaling and native-\(T=1\) labels |
| Confidently wrong image model | Confidence-only router misses geo rescues | Cross-probabilities, class pair, fusion rank, and reliability features |
| Class-pair overfitting | Apparent gain from sparse pairs | Regularization, grouped validation, minimum support reporting |
| Gate collapse | Gate is always on or always off | Activation monitoring, sparse prior, initialization and loss ablations |
| Expert drift | OOF targets no longer match jointly tuned experts | Freeze experts first; limit later fine-tuning; regenerate targets if needed |
| Extra model capacity | Unfair advantage over raw concat | Matched-parameter/budget controls and residual ablations |
| Single validation split | Selection result is high-variance and optimistic for the chosen policy | Freeze one plot-grouped split, keep the model grid small, report multi-seed stability, and seek independent confirmation |
| Final expert-refit shift | Router was trained on OOF experts fitted with less data than the final experts | Freeze all `router_frozen_state`, monitor score distributions without test adaptation, and disclose the shift |
| Fold-contained encoder cost | Slow iteration and large artifacts | Six encoder fits and 18 mode-head fits per seed, cached immutable logits, and content fingerprints |
| Independent dataset unavailable | Weak generalization claim | Use multi-seed grouped evidence and explicitly limit conclusions |

## Progress ledger

| Milestone | Status | Evidence or blocker | Next action |
|---|---|---|---|
| M0 Baselines and diagnostic evidence | Complete | Agreement cache, notebook, report, and reproduction checks | Preserve as immutable motivation |
| M1 Experimental protocol | Complete | Label-blind test identity, 4,200-row assignments, split balance, resolved config, protocol manifest, capability boundaries, and 43 focused tests validate | Preserve protocol_v1 immutably |
| M2 OOF expert outputs | Complete | All 20 producers validate; sealed aggregates contain 13,512 OOF and 3,288 label-blind validation records; checkpoint replay and the OOF report reproduce | Preserve the immutable M2 artifacts and consume them through validated readers |
| M3 Targets and router features | Complete | Sealed bundle contains 13,512 composite-key targets, the 30-feature/727-column future-transform schema, complete prevalence, a passing leakage audit, and immutable M1/M2/M3 lineage; validate and second build both return `reused_valid` | Preserve the bundle and consume it unchanged from M4 |
| M4 Router feasibility | Complete: no-go | Common C=0.1; 0/919 feasible primary thresholds, never-intervene fallback; both ablations inconclusive; 33-file result bundle and corrected receipt sealed; independent validation/reuse and 751 focused tests pass | Preserve the result; any revised scientific study needs a new declared protocol |
| Exploratory router capacity v1 | Complete; not a replacement M4 gate | LR, shallow XGB, and regularized RF compared across all 727/79/25 feature sets; 144 seed cases sealed; best routed validation point gain +0.729927 pp (RF/25), with conditional uncertainty and no clear superiority over the strongest heuristic; 839 focused tests pass | Preserve this development-only exploration; original M4 no-go and M5/M6 deferral remain unchanged |
| Exploratory router MLP v1 | Complete; not a replacement M4 gate | Four architectures across 727/79/25 features and four seeds: 48 fits; all variants select [32]; full/no-pairs fall back to no intervention, numeric gains +0.425791 pp with a CI spanning zero; 1,038 focused tests pass | Preserve the fixed-epoch negative/exploratory result and versioned replay correction; original M4 no-go and M5/M6 deferral remain unchanged |
| Exploratory legacy capacity transfer v1 | Complete; not M5 or independent confirmation | Frozen RF/XGB/MLP across all three feature variants/four seeds applied to both legacy checkpoint families; 36 states, 72 application cases, and 36 output files sealed; all new gain intervals versus raw include/touch zero; 1,304 focused tests pass | Preserve the descriptive results, append-only test event, and all prior states; no retuning, policy promotion, or change to M4 no-go |
| Exploratory relaxed LR guardrails v1 | Complete; not a replacement M4 gate | 27 development-selected policies across all LR feature variants, plot minima 20/10/none and harm caps 0.5/0.7/0.9; both legacy test families scored; 40 artifacts, independent validation/reuse, and 1,519 focused tests verified | Preserve the sensitivity results: plot relaxation changes nothing, 0.7/0.9 select identical policies with larger but uncertain fixed-final point gains; no test-based promotion or change to M4 no-go |
| Exploratory relaxed RF guardrails v1 | Complete; not a replacement M4 gate | Same 27-policy grid across all three frozen RF variants and both legacy test families; 40 artifacts, 216 new application cases, independent metric/bootstrap/lineage audits, byte/mtime-preserving reuse and 1,784 focused tests verified | Preserve all policies: plot relaxation changes nothing; harm relaxation raises fixed-final top-1 point gains but lowers top-3; all gain and paired intervals include zero; no test-based promotion or M4 gate revision |
| Exploratory relaxed margin guardrail v1 | Complete; not a replacement M4 gate | One H≤0.7R margin threshold selected on training OOF only, retaining 1% coverage and 20 plots; cleaned-test accuracy 74.387528% versus 74.480327% for the original rule, weighted F1 declines; 25 sealed artifacts, independent audits, immutable reuse and 382 focused tests pass | Preserve the exploratory negative result; no policy promotion or change to M4 no-go |
| M5 Test-time correction | Deferred after no-go | Frozen primary policy exists but makes no interventions and fails the development gate | Do not proceed under this failed gate |
| Initial ConfidNet study | Complete; separate development experiment, negative fixed-recipe result | Twelve same-training-set confidence fits; image/fusion have zero training errors and their confidence heads saturate near one; failure AP/AUC below MCP in every expert/seed; TCP comparator makes zero interventions, leaving accuracy at 71.0462% | Preserve the result; diagnose training saturation before a separately declared follow-up; no augmented-router training or cleaned-test access |
| OOF-supervised ConfidNet study | Complete; negative matched-router result | Twelve confidence heads and eight RF routers verified. Accuracy: numeric25 71.9282%, TCP-augmented28 69.5560%; paired difference −2.3723 pp, 95% CI [−4.5736, −0.3865]. Large confidence training-to-validation gaps; direct TCP also loses to fusion | Preserve the completed development result and original pause; distinguish confidence overfitting from producer-representation transfer in any separately approved follow-up |
| OOF ConfidNet relaxed-threshold sensitivity | Complete; development-only follow-up | Prior nine-setting RF guardrail grid applied to both saved routers. Augmented accuracy rises to 71.7762%; equally selected numeric25 reaches 72.0499%. Paired difference −0.2737 pp, conditional CI [−1.5885, +0.9896]. All 18 selections, independent audits, and immutable reuse pass | Zero was a poor augmented-router threshold, but estimated TCP still gives no demonstrated matched-router benefit; preserve validation-selection caveat and original M4 no-go |
| Fixed-expert fold-4 ConfidNet study | Complete; negative matched-router result | Confidence heads and numeric25/28 RF routers fit on fold 4; the same three-fold-trained experts produce validation inputs. Numeric25 relaxed accuracy 72.0499%; augmented has no feasible threshold under all nine settings and retains fusion at 71.3808%. Paired difference −0.6691 pp, conditional CI [−1.6299, +0.3041]. Independent audits, 130-file immutable reuse, and 166 focused tests pass | Fixed producing experts do not resolve the failure in this setup; preserve the confidence generalization gap and same-row router-training caveat, without attributing the earlier result to a single cause |
| Frozen fold-4 cleaned-test application | Complete; user-requested exploratory application | Same experts and saved fold-4 fits on 1,347 images / 531 plots. Fusion 73.1069%, numerical relaxed router 73.1626% (+3 pooled net), augmented relaxed router retains fusion. Highest estimated TCP among all three experts: 72.4388%, −0.6682 pp versus fusion, conditional CI [−1.9071, +0.5924]; 158 rescues / 194 harms | Preserve the frozen negative TCP comparison and negligible numerical gain; no test threshold search, legacy-expert substitution, or M4 gate revision |
| Fixed joint-geometry correction | Complete; negative cleaned-test transfer | Frozen image and geo nearest-reference scores must both favor geo. Validation accuracy improves by +1.5815 pp, but the matched September 15 cleaned-test comparison falls to 72.3645% versus fusion 74.3133% and RF 75.1670%; mean 37 rescues versus 63.25 harms per seed | Preserve the unchanged 3,378-image reference bank and rule; validation and test use different expert fitting sets. No test-driven threshold or reference-bank selection; original M4 no-go unchanged |
| Joint geometry with all development references | Complete; user-requested bank expansion | Same experts, test images and rule, with all 4,200 development images / 1,625 plots as references. Accuracy 71.8448%, weighted F1 0.7096; mean 39.5 rescues and 72.75 harms per seed. Difference from the 3,378-image bank is −0.5197 pp, conditional CI [−1.2737, +0.2065] | Retain both bank results. Expansion does not recover the earlier validation gain; this follow-up was requested after the first test result and is not independent confirmation |
| Joint geometry with five nearest plots | Complete; user-requested K5 sensitivity | Mean distance to five distinct reference plots per candidate class; retain fusion if either class has fewer than five plots. Validation accuracy 73.3577%; cleaned test 73.8122% / 73.7194% with 3,378 / 4,200 references, improving on K1 by +1.4477 / +1.8745 pp but still below fusion 74.3133% and frozen RF 75.1670%. Full-cohort K1 controls with matched reference support separate fallback from neighborhood effects | Preserve both banks and all controls. Conditional test K5-minus-fusion intervals include zero; validation gain is not independent confirmation. No threshold fitting, policy promotion or M4 change |
| Joint geometry with shared effective K | Complete; approved K=1/5/10/20 sweep | Shared K=min(requested K, both candidate reference-plot counts). Validation accuracy 72.6277/73.1752/72.9319/71.7457%; cleaned test with all 4,200 references 71.8448/73.4224/73.9607/74.1277%. K20 still trails fusion 74.3133% and RF 75.1670%; mean 52 rescues versus 54.5 harms. Fallback K5 loses to strict K5 on both cohorts | Increasing K helps test accuracy but harms validation beyond K5 and reduces macro F1. Preserve every K, strict-K5 comparator, actual-K counts and conditional paired intervals; no winner selection or M4 change |
| Validation-selected geometry thresholds | Complete; harm limits do not transfer | Strict K5/K10 with S=min(A_image,A_geo); 18 validation selections reuse the router's 20/10/0-plot × 0.5/0.7/0.9 harm grid and 1% coverage guard. Four distinct thresholds result; 0.7/0.9 and all plot minima coincide. Test accuracy ranges 73.9792–74.6474% with 3,378 references and 73.8679–74.3875% with 4,200, versus fusion 74.3133% and RF 75.1670% | Every test policy exceeds its validation harm cap; all paired accuracy intervals versus fusion include zero, and weighted/macro F1 remain below fusion. Preserve all frozen policies and both banks; no test reselection or M4 change |
| Habitat-adapted image geometry | Complete; better test correction, weaker validation accuracy/ranking | Four development-train-fitted encoders embed both references and queries; no OOF diagnostics. Strict zero-K5/K10 validation accuracy is 72.2932/72.0195%, below pretrained geometry. All 18 validation selections reduce to two thresholds; selected test accuracy is 75.4454/75.6682% with 3,378 references and 75.7053/75.5568% with 4,200, versus fusion 74.3133% and RF 75.1670% | Adaptation reduces harmful interventions on test; 0.7/0.9 caps hold pooled, while 0.5 caps fail. Test weighted F1 improves over fusion but macro F1 remains lower; all selected accuracy intervals versus fusion include/touch zero, and all RF contrasts include zero. Preserve cohort reversal, historical-cache lineage limits, both banks and original M4 no-go |
| M6 Gated residual fusion | Deferred after no-go | The primary router does not meet the declared feasibility criteria | Do not proceed under this failed gate |
| M7 Robust confirmation | Planned | Selected routers have matched multi-seed internal holdout evidence, not independent confirmation | Seek independent data; do not reinterpret selected validation as confirmation |
| M8 Integration and reproducibility | Partially delivered through M4 | Reproducible milestone runners, immutable artifacts, and 751 focused regression cases exist | Preserve M4 integration; broader method integration remains deferred |

## Step-specific development records

Create each sub-document when its milestone begins. Each record should contain
the frozen decisions, implementation notes, commands, artifacts, validation
results, unresolved issues, and completion evidence for that step.

| Planned record | Milestone | Status |
|---|---|---|
| [`01_experimental_protocol.md`](01_experimental_protocol.md) | M1 | Complete |
| [`02_oof_expert_outputs.md`](02_oof_expert_outputs.md) | M2 | Complete |
| [`03_targets_and_features.md`](03_targets_and_features.md) | M3 | Complete |
| [`04_router_dataset_preparation.md`](04_router_dataset_preparation.md) | M4 dataset subtask | Complete; native-runtime caveat documented |
| [`04_router_candidate_training.md`](04_router_candidate_training.md) | M4 candidate-fitting subtask | Complete; 48 converged candidates under the approved v2 iteration budget |
| [`04_router_validation_preview.md`](04_router_validation_preview.md) | M4 pre-label preparation and fixed preview | Complete; subsequent full selection/evaluation is recorded below |
| [`04_router_feasibility.md`](04_router_feasibility.md) | M4 | Complete: primary no-go, ablations inconclusive, frozen receipt and reproducible validation |
| [`05_legacy_router_transfer.md`](05_legacy_router_transfer.md) | Separate exploratory legacy-expert transfer; not M5 | Complete: both fixed-final and test-selected oracle checkpoints evaluated with frozen M4 policies; independent validation/reuse pass; M4 no-go unchanged |
| [`06_router_capacity_exploration.md`](06_router_capacity_exploration.md) | Separate development-only model-capacity exploration; not M5/M6 | Complete: all three learners and all three feature variants evaluated; nine policies sealed; independent validation, byte-preserving reuse, and original-parent checks pass |
| [`07_router_mlp_exploration.md`](07_router_mlp_exploration.md) | Separate development-only MLP architecture exploration; not M7 confirmation | Complete: 48 CPU fits and three selected policies; epoch histories, independent repeat fits, score replay and immutable reuse verified; all original artifacts preserved through an additive column-order replay correction |
| [`08_legacy_capacity_transfer.md`](08_legacy_capacity_transfer.md) | Separate frozen RF/XGB/MLP legacy-expert transfer; not M5 | Complete: both checkpoint families and all ablations evaluated without fitting/selection; independent prediction/metric audits, reconstruction, immutable reuse, and 931-parent lineage checks pass |
| [`09_relaxed_guardrails_lr.md`](09_relaxed_guardrails_lr.md) | Separate LR guardrail sensitivity and exploratory cleaned-test transfer; not M5 | Complete: all 27 approved guardrail combinations selected only on development; both test families scored with unchanged LR models; independent audits, 252-parent checks, immutable reuse and append-only registry verified; native-runtime caveat recorded |
| [`10_relaxed_guardrails_rf.md`](10_relaxed_guardrails_rf.md) | Separate frozen-RF guardrail sensitivity and exploratory cleaned-test transfer; not M5 | Complete: all 27 development-only selections, both test families and three feature variants; unchanged twelve RF states, 1,025-parent checks, independent reconstruction/bootstrap audits, immutable reuse and fourth registry event verified; native-runtime caveat recorded |
| [`11_relaxed_margin_guardrail.md`](11_relaxed_margin_guardrail.md) | Separate OOF margin-heuristic sensitivity and exploratory cleaned-test evaluation; not M5 | Complete: original threshold reproduced, one H≤0.7R threshold sealed; cleaned-test gain +0.074239 pp, below the original heuristic, with CI spanning zero and lower weighted F1; independent OOF/metric/bootstrap audits, immutable reuse and fifth registry event verified |
| `05_test_time_correction.md` | M5 | Not created |
| [`12_confidnet_initial_experiment.md`](12_confidnet_initial_experiment.md) | Separate frozen-expert confidence estimation and direct TCP policy | Complete: 12 heads, saturated image/fusion estimates and zero TCP interventions; independent metric/state audits, 61-file immutable reuse and 75 focused tests verified; no validation selection or cleaned-test access |
| [`13_confidnet_oof_proposal.md`](13_confidnet_oof_proposal.md) | Approved pooled expert-OOF confidence supervision and numerical-router setup | Approved scope delivered with native-replay amendment; router feature sets fixed at 25 versus 28 |
| [`14_confidnet_oof_experiment.md`](14_confidnet_oof_experiment.md) | Execution record for the approved OOF study | Complete: 12 heads, 8 routers, negative augmented-router result, independent audits and 191-file immutable reuse verified; 305 tests passed |
| [`15_confidnet_relaxed_guardrails.md`](15_confidnet_relaxed_guardrails.md) | User-requested relaxed selection on the saved numeric25/augmented28 routers | Complete: 18 selected policies, 78 selector tests, independent 12,726-candidate and 2,000-bootstrap audits, eight-file immutable reuse; no new fitting or cleaned-test access |
| [`16_confidnet_fold4_experiment.md`](16_confidnet_fold4_experiment.md) | Approved fold-4 confidence/router fitting with fixed producing experts and relaxed selection | Complete: 12 confidence heads, 8 RF routers, no feasible augmented threshold; independent model/metric/provenance audits, 130-file immutable reuse and 166 focused tests pass; development only |
| [`17_confidnet_fold4_cleaned_test.md`](17_confidnet_fold4_cleaned_test.md) | User-requested cleaned-test application of the saved fold-4 routers and highest-estimated-TCP expert selector | Complete: all four fixed expert bundles, 26 frozen policies, negative direct-TCP gain; 130 focused tests, independent audits, 35-file immutable reuse, and sixth append-only test event verified |
| [`18_joint_tcp_router_exploration.md`](18_joint_tcp_router_exploration.md) | Joint four-state/TCP supervision with 25/79 features, OOF embeddings, and matched CE-only controls | Complete: 64 fixed-budget fits, 144 selected policies, 360 paired contrasts; no consistent auxiliary-TCP routing benefit and substantial embedding-model overfit. Independent audits, exact refits, 183 distinct regression checks, and 540-file immutable reuse pass; development only |
| [`19_joint_tcp_router_cleaned_test.md`](19_joint_tcp_router_cleaned_test.md) | Apply all saved joint TCP routers and validation-selected thresholds to cleaned test using their matching full-development-train experts | Complete: all 64 routers applied; fusion 73.7379%, numerical-25 relaxed policies 73.8679–74.3504%, embedding policies at or below fusion. No clear matched auxiliary-TCP benefit; all router-versus-fusion accuracy intervals include zero. 156 tests, independent audits, 45-file immutable reuse and seventh append-only test event verified |
| [`20_joint_tcp_router_capacity.md`](20_joint_tcp_router_capacity.md) | Larger shared MLPs with auxiliary TCP supervision, both non-embedding input sets and matched CE-only controls | Complete: 128 new fits plus 32 reused `[32]` controls; exact September 15 legacy experts reproduce fusion 74.3133%. At minimum 20 plots / H≤0.7R, larger numerical routers reach 74.109–74.722%, below `[32]` CE-only 74.814% and matched RF 75.130%. All larger configurations worsen held-out state NLL; TCP gains are mixed. Every setting retained; no winner promoted |
| [`21_embedding_geometry_correction.md`](21_embedding_geometry_correction.md) | Nine-study geometry exploration and correction record | Complete: initial geometry/disagreement diagnostics, K and reference-bank comparisons, validation-fitted harm thresholds, and adapted-encoder ablation. Canonical artifact inventory and byte-preserving layout migration verified; no experiments rerun for reorganization |
| [`22_project_structure.md`](22_project_structure.md) | Project-wide naming and structure audit | Complete: canonical geometry and helper entrypoints, full study/source/artifact catalog, structural verification and preserved historical provenance; no experiment reruns |
| [Geometry policy validation](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_validation_v1/report.md) and [matched cleaned-test comparison](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_cleaned_test_v1/report.md) | Separate fixed geometry rule; not M5/M6 | Complete: no learned router or threshold search. Validation 71.0462% → 72.6277%; historical cleaned-test fusion 74.3133% → 72.3645%. Geometry minus fusion accuracy CI [−3.8189, −0.1307] pp; geometry minus RF [−4.5770, −0.9992] pp. Frozen reference bank, matched historical predictions, per-habitat results and append-only test receipt retained |
| [Full development reference bank](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_full_reference_cleaned_test_v1/report.md) | User-requested expansion from 3,378 to all 4,200 references | Complete: identical experts and strict-positive rule; accuracy 71.8448% versus 72.3645% with the original bank, fusion 74.3133% and RF 75.1670%. Six-method comparison, changed-action analysis, paired plot intervals and a separate test receipt preserve the earlier evaluation |
| [Five-neighbor geometry correction](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_k5_correction_v1/report.md) | User-requested K5 comparison on validation and cleaned test with both reference banks | Complete: validation improves from fusion 71.0462% to 73.3577%; test K5 reaches 73.8122% / 73.7194% versus K1 72.3645% / 71.8448%. With all 4,200 references, mean 49.5 rescues and 57.5 harms leave a −0.5939 pp difference from fusion, conditional CI [−2.4800, +1.3109]. Full-cohort support controls, per-class metrics, 22 paired intervals and an additional test event preserve prior evaluations |
| [Shared-neighborhood K sweep](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_shared_k_sweep_v1/report.md) | Approved K=1/5/10/20 with matched smaller-neighborhood fallback; validation uses training references and cleaned test uses all development references | Complete: test K20 accuracy 74.1277%, difference from fusion −0.1856 pp, conditional CI [−2.0502, +1.7111]. Validation K20 falls 1.4294 pp below fallback K5. New fallback K5 adds 1 rescue and 17 harms across four test seeds relative to strict K5. All 68 cohort/model/seed metric rows, effective-K usage, per-habitat results and 44 paired intervals saved. Independent distance, metric and bootstrap audits plus completed-run reuse passed; prior evaluations preserved |
| [Validation-selected geometry thresholds](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_validation_thresholds_v1/report.md) | User-approved strict K5/K10 threshold fitting on validation, then fixed application to both test reference banks | Complete: all 18 selections feasible, four distinct thresholds. K10/0.5 with 3,378 references yields +0.3341 pp versus fusion, conditional CI [−1.1229, +1.7950]; K10/0.7 or 0.9 with 4,200 yields +0.0742 pp, CI [−1.7071, +1.8777]. All test harm caps fail transfer. Independent audits reproduced 3,213 curve records, 284 model/seed metric records and all 288 intervals; geometry/provenance checks and completed-run reuse passed. Previous artifacts preserved |
| [Adapted versus pretrained geometry](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_adapted_encoder_v1/report.md) | User-approved encoder comparison on validation and cleaned test, with strict K5/K10 and both test banks; no OOF diagnostics | Complete: four adapted towers fitted only on 3,378 images, with exact replay of all validation embeddings. Zero-K5 test accuracy improves by +1.5961/+1.9859 pp over pretrained geometry for 3,378/4,200 references; selected policies reach 75.4454–75.7053%. Independent checks reproduce 2,232 candidate records, 284 model/seed metric records, 6,000 class-distance/neighbor checks and all 408 paired intervals. All policies retained; no test reselection or M4 change |
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
| D004 | 2026-08-24 | Confirmed | Predict four correctness states and compute \(\hat u=q_+-q_-\) | Separates both-correct from both-wrong neutral cases |
| D005 | 2026-08-24 | Confirmed | Protocol-v1 primary is an output-only, four-state, L2 multinomial logistic router; only `C` is selected | Tests signal feasibility before adding model capacity and limits fixed-validation adaptation |
| D006 | 2026-08-24 | Confirmed | Protocol v1 evaluates only the hard override; soft pooling requires a new version | Provides the most auditable intervention baseline and prevents an unselected soft mapping entering the locked bundle |
| D007 | 2026-08-24 | Proposed | Start gated fusion with a scalar, raw-fusion-anchored residual | Conservative and less prone to overfitting than class-wise gates |
| D008 | 2026-08-24 | Confirmed | Use one fixed plot-grouped development-train/development-validation split and four-fold cross-fitting only inside development-train for router development | With fold-contained encoder adaptation this gives six encoder-producing stages per seed including train-to-validation and final full-development inference, plus 18 mode-head fits; validation becomes selection evidence rather than an unbiased final estimate |
| D009 | 2026-08-24 | Confirmed | Protocol v1 uses one scalar temperature per expert mode and training seed, fitted once from development-train OOF logits; native \(T=1\) is descriptive and there is no post-hoc router-output calibrator | Produces comparable expert probabilities without allowing validation or test labels to fit calibration |
| D010 | 2026-08-24 | Open | Availability and definition of an independent confirmation set | Determines the strength of final generalization claims |
| D011 | 2026-08-24 | Confirmed | Freeze the selected router pipeline and threshold specification byte-for-byte after development-validation; refit only the explicitly allow-listed expert state on all development data | Preserves the selected scoring function and threshold coordinate system, keeps final in-sample expert predictions out of router fitting, and discloses that stronger final experts may still shift the input and score distributions |
| D012 | 2026-08-24 | Confirmed | Select one common `C` and threshold specification across seeds 1–4 while retaining one seed-specific temperature/router checkpoint per seed | Avoids best-seed selection and gives all seed realizations the same policy rule |
| D013 | 2026-08-24 | Confirmed | Seal label-blind router-candidate validation predictions before a separate fit-incapable validation scorer opens labels | Prevents validation-label access from sharing a process capability with router fitting |
| D014 | 2026-08-24 | Confirmed | Seal one protocol-independent cleaned-test identity snapshot and record every frozen-bundle score in a global append-only event registry | Prevents protocol IDs from silently changing the test universe or hiding adaptive test reuse |
| D015 | 2026-08-25 | Confirmed | Use fold-contained, vision-only SigLIP2 adaptation for protocol v1 | Each of four OOF producers, the train-to-validation producer, and the final full-development producer starts from the same pinned pretrained checkpoint; this is six encoder fits and 18 expert-head fits per seed |
| D016 | 2026-08-25 | Confirmed | Freeze the text tower, unlock the last 11 vision groups, and use the fixed 18-class prompt objective for five epochs | Retains the intended habitat adaptation while keeping the recipe bounded and reproducible |
| D017 | 2026-08-25 | Confirmed | Retain the legacy OpenCV BGR decode and forced 439×439 pre-resize in protocol v1 | Preserves historical preprocessing compatibility; correcting channel order or aspect handling requires a new protocol version |
| D018 | 2026-08-25 | Confirmed | Derive dense and canonical development labels by exact label-name lookup in the frozen ontology | Removes dependence on legacy dense-label mappings that could have test-informed provenance |
| D019 | 2026-08-31 | Confirmed | Implement M2 as an additive fingerprinted child of immutable M1, with one resumable end-to-end command per seed and a separate strict four-seed aggregate | Preserves `protocol_v1` fingerprints while making all five M2 producers per seed independently auditable and restart-safe |
| D020 | 2026-09-01 | Confirmed | Preserve M3 target identity as `(row_uid, training_seed)` in the combined physical table, with no averaging, voting, or deduplication across seeds | Expert correctness and therefore target state can vary across seed realizations; a fixed-seed projection remains uniquely keyed by `row_uid` without changing `protocol_v1` |
| D021 | 2026-09-01 | Confirmed | M3 owns targets, the stateless semantic feature builder and schema, prevalence, leakage audit, and lineage; M4 owns temperature fitting and calibrated router-dataset materialization | Prevents descriptive native-`T=1` probabilities from silently becoming primary router features and gives training and deployment one calibrated, state-free builder |
| D022 | 2026-09-08 | Confirmed | Materialize the M4 training matrix and a calibrated-probability/semantic-feature audit view; use equal-image-weight NLL for each seed/mode and freeze twelve temperatures plus four seed-local transforms | Makes the declared router inputs reproducible without seed averaging or validation/test fitting; preparation is sealed separately from subsequent router-training code |
| D023 | 2026-09-08 | Recorded outcome | Complete M4 with the primary no-go result and never-intervene fallback; preserve both inconclusive ablations without promotion | No primary threshold meets all support/harm constraints; neither ablation beats every predeclared heuristic; the failed gate does not authorize M5/M6 |
| D024 | 2026-09-08 | Implementation correction | Preserve the predeclared scorer and all 31 computed result artifacts; add a versioned finalizer correcting only the receipt-schema configuration path | The frozen schema is under `schemas`, not `artifact_contract`; no scientific setting, model selection, or previous result byte changes |
| D025 | 2026-09-09 | Recorded exploratory outcome | Predeclare and complete `router_capacity_v1`: four settings each for LR, RF, and XGB on all three frozen feature variants; select separately on reused development validation | Isolates learner capacity with unchanged expert inputs and preprocessing; RF provides modest gains, not independent confirmation or a replacement M4 go decision; no cleaned-test access or policy promotion |
| D026 | 2026-09-09 | Recorded exploratory outcome | Complete the agreed MLP grid [32], [64], [32,16], [64,32] across all three feature variants/four seeds, with fixed 300-epoch training and no validation early stopping | Greater capacity lowers training NLL but worsens validation NLL; [32]/numeric gives +0.425791 pp habitat gain with conditional CI spanning zero, below existing baseline point estimates; no promotion or test access |
| D027 | 2026-09-09 | Implementation correction | Preserve all 148 sealed MLP source files; add `router_mlp_v1_replay_v2` with hash-bound support for the two known training-history column orders and unchanged strict numerical checks | Canonical JSON key sorting changes reconstruction order, not values; all scoring rules and fitted states remain unchanged, with no refitting or original-file rewrite |
| D028 | 2026-09-09 | Recorded exploratory outcome | Apply the nine sealed RF/XGB/MLP policies, including both feature ablations, to the same two legacy cleaned-test expert families as the prior LR transfer | Separate apply-only study with a new registered test event; RF point gains are modest, MLP/numeric gains smaller, and every new gain interval versus raw includes/touches zero; preserve historical-test-bias disclosures and original M4 no-go |
| D029 | 2026-09-13 | Recorded exploratory outcome | Complete the user-approved 20/10/no-plot × 0.5/0.7/0.9 harm-cap sensitivity using all three frozen LR variants as a compute-limited starting point, not because of test ranking | Threshold selection remains development-only with unchanged C/models/calibration; plot minima do not affect selected policies and 0.7/0.9 coincide. Fixed-final gains are +0.687/+0.612/+0.780 pp, but all gain and paired intervals include zero; no policy promotion, broader-learner expansion, or M4 gate revision |
| D030 | 2026-09-13 | Recorded exploratory outcome | Complete the user-requested RF extension of the same nine-setting guardrail grid across all three frozen feature variants, without model/configuration reselection or test threshold search | Plot minima again have no effect. Full727 distinguishes 0.7/0.9; both ablations share those thresholds. Fixed-final relaxed gains are +0.761 to +0.854 pp with lower top-3, uncertain intervals and inconsistent test-selected transfer; no promotion or M4/M5/M6 gate change |
| D031 | 2026-09-16 | Recorded exploratory outcome | Complete the user-requested geo/raw margin heuristic under H≤0.7R, retaining 1% coverage and 20 plots, using OOF-only threshold selection and the current report's expert family | Threshold falls to 0.1311806422; test interventions rise to 371 but 136 rescues and 132 harms give only +4 pooled net corrections. Accuracy 74.387528% and weighted F1 0.736112 underperform the original heuristic; uncertain adaptive-test result, no threshold retuning or M4 gate revision |
| D032 | 2026-09-22 | Recorded exploratory outcome | Complete the approved initial same-training-set ConfidNet study using frozen full-development-train experts, native TCP targets, twelve fixed-budget heads and a zero-difference policy | Image/fusion have no training errors; confidence heads saturate near one. All twelve validation failure AP/AUC comparisons lose to MCP and the TCP policy never intervenes. Preserve this fixed-recipe negative result; it does not establish a limit of TCP or representations. No augmented router, cleaned-test access, or M4 gate change |
| D033 | 2026-09-22 | Approved setup; execution paused | Fit pooled expert-OOF ConfidNets and compare an RF using 25 numerical summaries with an RF using those summaries plus three estimated TCPs | First image-head verification exceeds float64 replay tolerance on one training row (1.19913e-6 versus 1e-6), while native float32 JSON replay is exact. Preserve the pause and training-only diagnostic; obtain approval for the proposed verification amendment before continuing. No router or validation results yet |
| D034 | 2026-09-22 | User-approved verification amendment | Require exact native float32 confidence replay from saved JSON weights; record independent float64 differences without an absolute-tolerance rejection | Continue in confidnet_oof_v1_replay_v2, preserving the original declaration and pause. Scientific settings and native predictions are unchanged; failed final checks now retain model/history. All 305 focused tests pass before launch |
| D035 | 2026-09-22 | Recorded exploratory outcome | Complete the approved OOF-supervised confidence and numeric25/28 RF comparison under the native-replay amendment | Augmented accuracy 69.5560% versus matched numeric25 71.9282%; −2.3723 pp with paired whole-plot CI [−4.5736, −0.3865], lower in all four seeds. Training failure AP near 0.997 does not transfer; direct TCP also underperforms fusion. Preserve this negative pipeline result; no cleaned-test evaluation, retuning, or M4 gate change |
| D036 | 2026-09-22 | User-requested exploratory outcome | Apply the previously established relaxed RF threshold grid to both saved OOF ConfidNet router arms, with one threshold per arm/setting across all seeds | At H≤0.7R/0.9R, augmented accuracy becomes 71.7762%, versus matched numeric25 72.0499%; the 0.5 cap has no feasible augmented threshold. Plot relaxation changes nothing. Conditional paired interval spans zero; no new models, cleaned-test access, or M4 gate change |
| D037 | 2026-09-24 | User-approved exploratory outcome | Fit ConfidNet and matched numeric25/28 RF routers on fold 4 only, using frozen folds-1–3 experts for both fitting inputs and validation; retain native replay and the nine-setting relaxed rule | Augmented has no feasible threshold and retains fusion at 71.3808%; relaxed numeric25 reaches 72.0499%. Confidence training failure AP near 0.997–0.999 falls to 0.45–0.54 on validation, below MCP. Fixed producers do not solve this setup's generalization failure; smaller fitting data and same-row confidence/router fitting remain confounded. No cleaned-test scoring or M4 gate change |
| D038 | 2026-09-24 | User-requested exploratory outcome | Apply the saved fold-4 pipeline to cleaned test with the same folds-1–3 experts and transferred thresholds; compare choosing the highest native estimated TCP among image, geo and fusion, with ties favoring fusion/geo/image | Three-expert TCP accuracy 72.4388% versus fusion 73.1069%; numerical relaxed 73.1626% gives only +3 pooled net, while augmented relaxed retains fusion. No refitting, recalibration or test selection. Independent audits verify all policies and the sixth registered evaluation; historical test reuse and M4 no-go remain explicit |
| D039 | 2026-09-28 | User-requested exploratory outcome | Apply all 64 saved joint routers, all 144 validation-selected policies and all 16 fixed-zero policies to cleaned test with the matching full-development-train experts | Fusion is 73.7379%; numerical-only routing gives modest positive point gains, while embedding policies are at or below fusion. Auxiliary-versus-CE gains remain uncertain. Preserve all weights and transferred thresholds, including fallbacks and test guardrail violations; no fitting, test selection or M4 gate change |
| D040 | 2026-09-28 | User-approved exploratory outcome | Fit `[64,32]`, `[128,64]`, `[64,32,16]`, `[128,64,32]` across 25/79 inputs, four auxiliary weights and four seeds; compare 32 reused shallow controls and transfer frozen policies to the exact September 15 experts | 128 new fits complete; larger models overfit state and TCP targets. At minimum 20 plots / H≤0.7R all 40 joint configurations have lower point accuracy than feature-matched historical RF/LR. All 360 relaxed test gains versus fusion have accuracy intervals including zero. No architecture/weight promotion, test retuning or M4 gate change |

## Immediate next action

Review the [completed joint-MLP capacity round](20_joint_tcp_router_capacity.md).
All 128 new fits and 32 shallow controls were evaluated on validation and cleaned
test. The exact September 15 expert heads and cached encoder features reproduce
fusion at 74.3133%. Under the matched minimum-20 / H≤0.7R guardrail, larger
numerical routers reach 74.109–74.722%, compared with `[32]` CE-only at 74.814%
and historical numerical RF at 75.130%. All 32 larger configurations worsen
validation/test state NLL relative to their matched shallow controls, and all
24 supervised larger configurations worsen both experts' held-out TCP MSE.
TCP can help relative to a larger model's own control, but this round does not
establish a capacity benefit. Retain the complete grid and original studies;
no architecture or weight is promoted from these exploratory results.

Review the [joint TCP cleaned-test application](19_joint_tcp_router_cleaned_test.md)
alongside its [development study](18_joint_tcp_router_exploration.md). The matching
full-development-train fusion experts reach 73.7379% test accuracy. Numerical-25
routers reach 73.8679–74.3504% with transferred relaxed thresholds, but all
router-versus-fusion accuracy intervals include zero. Auxiliary TCP does not
show a clear improvement over matched CE-only controls, and embeddings lower
accuracy in almost every matched comparison. These results use the same 1,347
cleaned-test images as earlier applications but different expert checkpoints.
They retain every declared setting and do not select a winner from test results.

Review the completed [frozen fold-4 cleaned-test application](17_confidnet_fold4_cleaned_test.md).
The requested highest-estimated-TCP policy among all three experts reaches
72.4388%, below matched fusion at 73.1069%. Its 158 rescues and 194 harms give
−36 pooled net corrections, although the conditional gain interval includes
zero. The numerical relaxed router gains only +0.0557 pp (+3 pooled net);
the augmented relaxed router retains fusion. These are frozen applications
using the same producing experts, with no test threshold selection. The
cleaned test has been inspected repeatedly, so it is not independent confirmation.

Review the completed [fixed-expert fold-4 study](16_confidnet_fold4_experiment.md)
alongside the [pooled-OOF study](14_confidnet_oof_experiment.md) and its
[relaxed-threshold follow-up](15_confidnet_relaxed_guardrails.md). Keeping the
producing experts fixed does not improve the TCP-augmented router in this
configuration: all nine guardrails choose the never-intervene fallback at
71.3808%, while the matched relaxed numerical router reaches 72.0499%.
Confidence heads still fit training targets closely and generalize poorly.
This shows the failure can occur without changing expert representations;
it does not identify the cause of the previous result, because fitting sample
size, expert checkpoints, and optimization updates also differ. Same-row
ConfidNet/router fitting remains a possible contributor. Threshold selection
and reporting share development-validation, so these are exploratory results.
Preserve all three studies and the original verification-pause history.
Additional modeling controls require a separately approved design.

Review the completed [M4 feasibility result](04_router_feasibility.md). The
current primary method has a frozen no-go outcome; do not proceed to M5/M6,
promote an ablation, relax guardrails, or retune on this same holdout under
protocol v1. Preserve all selected states, thresholds, reports and the receipt.
Any revised scientific study requires an explicitly declared new protocol and
appropriate evidence boundaries. The completed [capacity exploration](06_router_capacity_exploration.md)
and [MLP architecture exploration](07_router_mlp_exploration.md) are separately
declared, development-only comparisons and do not change that gate. Neither
performs cleaned-test or legacy-transfer-output evaluation; the earlier
[legacy transfer experiment](05_legacy_router_transfer.md) is recorded independently.
The subsequent [frozen RF/XGB/MLP legacy transfer](08_legacy_capacity_transfer.md)
is also complete as a separate, explicitly authorized exploratory study. It
transfers all selected states and thresholds unchanged, appends a new test
event, and does not supply independent confirmation or authorize retuning.
The user-approved [relaxed LR guardrail study](09_relaxed_guardrails_lr.md)
is now also complete under its own declaration. It selected all 27 thresholds
only on development-validation, then froze them before both legacy test
applications. The subsequently authorized [RF extension](10_relaxed_guardrails_rf.md)
is also complete: all 27 RF policies were selected on development and applied
unchanged to both legacy families, preserving the twelve fitted RF states and
every earlier study. Review both sensitivity records and their uncertainty
without choosing a test-winning guardrail, feature variant, or checkpoint
family. This grid has not been extended to XGB or MLP; further work needs a
separate authorized scope and declaration. None of these exploratory results
replaces the original no-go or constitutes independent confirmation.

The separately authorized [relaxed margin heuristic](11_relaxed_margin_guardrail.md)
is also complete. Its threshold was selected only on training OOF predictions
with H≤0.7R, then frozen before one-family cleaned-test application. It raises
coverage but performs worse than the original margin rule on this test set;
retain this negative sensitivity result without additional test-driven tuning.

## Change log

| Date | Change |
|---|---|
| 2026-10-01 | Added consolidated geometry reproduction instructions to record 21: environment and input requirements, saved-run checks, fresh development computation, frozen adapted extraction, exact numerical metric example and expected outputs. Verified the four adapted headline results; documented that an isolated end-to-end cleaned-test rebuild is not yet implemented. |
| 2026-10-01 | Audited naming across M1–M4, router variants, ConfidNet, joint-TCP and geometry; added a canonical geometry CLI/module, consistent helper entrypoints, full study catalog and record 22. Existing scientific code/configurations and test-registry events remain unchanged; public dispatch and structure checks verified. |
| 2026-10-01 | Standardized all nine completed geometry bundles under the versioned geo-helpfulness artifact root; added record 21, a machine-readable layout catalog and read-only verification. Original files, source/code bindings and all cleaned-test registry entries are preserved; directory aliases keep historical paths and report links working. No fitting, threshold selection, scoring or OOF diagnostics were repeated. |
| 2026-09-30 | Completed the approved adapted-encoder geometry comparison without OOF diagnostics. Reused four towers fitted only on 3,378 development-training images to freshly encode 4,200 development and 1,347 test images per seed; every validation embedding replays exactly. Held expert predictions, geo geometry and reference membership fixed. All 18 validation selections choose one threshold per K (K5 0.006095635; K10 0.01600631), unchanged across both test banks. Adaptation weakens validation accuracy and rescue/harm ranking but improves test correction, primarily by avoiding harmful interventions. Relaxed 0.7/0.9 test harm caps now hold pooled; 0.5 caps do not. Test accuracy and weighted F1 exceed fusion point estimates, while macro F1 stays lower and paired accuracy intervals versus fusion/RF include or touch zero. Saved both representations, fixed-zero controls, full diagnostics, 408 conditional intervals and a separate test receipt; no policy promotion or M4 change. |
| 2026-09-30 | Completed the approved validation-threshold study with user-confirmed strict reference support for K5/K10. Selected one common threshold per K/guardrail on all 822 validation images and four seeds, reusing the router grid and exact midpoint/tie rules. Froze all 18 policies before applying both 3,378/4,200 test reference banks. The 0.7 and 0.9 caps select identical thresholds; plot relaxation changes nothing. Some K10 test accuracy point gains occur, but every test harm cap is exceeded and all accuracy intervals versus fusion include zero. Retained every setting, untuned strict baselines and RF comparison; no test adjustment or M4 change. |
| 2026-09-30 | Completed the approved shared-K=1/5/10/20 sweep with strict-positive image/geo correction and K_eff=min(K,n_fusion_class,n_geo_class). Validation uses 3,378 training references; cleaned test uses all 4,200 development references and unchanged legacy experts/RF. Increasing K from 5 to 20 lowers validation accuracy 73.1752%→71.7457% while raising test accuracy 73.4224%→74.1277%, still below fusion 74.3133%. Macro F1 declines on both cohorts. Allowing smaller K also lowers performance relative to strict K5. Saved all settings, actual-K counts, 44 paired intervals and a separate adaptive test receipt; no selection, fitting or M4 change. |
| 2026-09-30 | Completed the requested K5 correction evaluation with the unchanged joint-positive rule. Each class distance averages five nearest distinct reference plots; insufficient support retains fusion. Validation accuracy 73.3577%; cleaned-test accuracy 73.8122% / 73.7194% for the 3,378 / 4,200 reference banks. Both improve on K1 but remain below fusion and RF point estimates. Saved supported-K1 controls to distinguish neighborhood averaging from fallback changes, full-cohort metrics and paired plot intervals. Previous scripts, results and registry files remain unchanged; this adaptive follow-up has its own receipt. |
| 2026-09-30 | Completed the explicitly requested all-development reference-bank evaluation. Added 822 validation references to the original 3,378, with unchanged experts, cosine k=1 distances and strict-positive joint rule. Cleaned-test accuracy is 71.8448%, −0.5197 pp versus the smaller bank and −2.4684 pp versus fusion. Mean rescues rise to 39.5 but harms to 72.75. Prior scripts/results and registry entries are preserved; a separate event records this adaptive follow-up. No threshold search or M4 change. |
| 2026-09-30 | Completed the fixed joint-geometry rule on development validation and, at the user's request, the same cleaned-test universe and legacy experts as the September 15 report. The strict-positive image-and-geo rule keeps its 3,378-image reference bank. Test accuracy is 72.3645%, below fusion 74.3133% and frozen RF 75.1670%; all four seeds lose accuracy. Saved label-free decisions before scoring, reproduced every historical comparator, preserved all prior registry files, and appended one study event. No fitting, test tuning, or M4 gate change. |
| 2026-09-28 | Completed `joint_tcp_router_test_v1`: all 64 frozen routers applied to cleaned test, preserving all validation thresholds. Matched fusion is 73.7379%; numerical-25 relaxed accuracy is 73.8679–74.3504%, while embedding policies are at or below fusion. All 160 router-versus-fusion accuracy intervals include zero; matched auxiliary effects remain uncertain. 156 tests, exact validation/test replay, 49,291 independent metric assertions and model/provenance audits pass. Complete-run reuse preserves all 45 application, 540 parent and 22 registry files byte-for-byte with unchanged mtimes; seventh append-only test event verified. No fitting, test selection or M4 change. |
| 2026-09-28 | Completed `joint_tcp_router_v1`: all 64 fits, 20 frozen-producer embedding replays, 144 relaxed policies and 360 paired contrasts. Auxiliary-versus-CE accuracy differences range from -0.5170 to +0.2737 pp under relaxed selection; all 108 corresponding intervals include zero. Embedding models substantially overfit. No weight or guardrail winner selected. Independent metrics/fit audits, two exact 300-epoch refits, 183 distinct regression checks, and 540-file immutable reuse pass. An additive versioned runner corrects Markdown validation ordering while preserving every original scientific source and artifact. No cleaned-test use or change to M4. |
| 2026-09-22 | Completed the approved initial ConfidNet experiment: twelve same-training-set heads, saturated image/fusion confidence and zero TCP-policy interventions; validation remains at 71.0462%. Independent metrics/replay and 61-file byte/mtime-preserving reuse pass; 75 focused tests passed before fitting. Documented optimization limitations and the distinct confidence/router cross-fitting choices. Original M4 and prior artifacts remain unchanged. |
| 2026-08-24 | Created the living method specification and implementation roadmap |
| 2026-08-24 | Replaced nested outer CV for router development with one fixed grouped development holdout plus four-fold cross-fitting inside development-train; recorded the weaker evidence status and frozen-router final-refit rule |
| 2026-08-25 | Completed M1: froze fold-contained B2 encoder adaptation with legacy BGR/439 preprocessing, sealed the label-blind test identity, materialized and validated grouped assignments, and added protocol/leakage/immutability enforcement |
| 2026-08-31 | Implemented the additive M2 producer, per-seed runner, immutable checkpoint/output validation, four-seed aggregation, OOF report, and CPU synthetic acceptance tests |
| 2026-08-31 | Completed M2 GPU execution for seeds 1–4 and sealed validated 13,512-row OOF and 3,288-row label-blind validation aggregates; all 20 producer checkpoints and the OOF report reproduce |
| 2026-09-01 | Confirmed the seed-specific M3 target identity and moved all probability calibration, learned transforms, and `router_dataset.parquet` materialization to M4 |
| 2026-09-01 | Began the additive M3 implementation: pure targets, stateless 30-feature contract, schema, prevalence, leakage audit, and immutable child-bundle workflow |
| 2026-09-01 | Completed M3: validated exactly 16 train-OOF parents, sealed and revalidated 13,512 seed-specific targets plus schema/prevalence/leakage artifacts, reproduced all acceptance counts, and confirmed valid immutable reuse without calibration or a router dataset |
| 2026-09-08 | Completed the additive M4 dataset subtask: fitted twelve OOF temperatures and four transforms, sealed the 13,512×727 feature matrix plus audit/state artifacts, passed independent reconstruction and byte-preserving reuse, and passed 232 focused tests; documented intermittent native-runtime crashes; router fitting remains unfinished |
| 2026-09-08 | Completed 48 primary/ablation candidate fits under the approved versioned 5,000-iteration budget; preserved the original configuration and partial v1 run; reconstruction, explicit refitting, and immutable reuse pass |
| 2026-09-08 | Sealed all 48 label-blind candidate validation prediction sets and OOF heuristic thresholds; scored only fixed primary seed-1/C=0.01 at 585/822 = 71.17%; preserved the seal through an additive string-storage scorer correction; 564 focused tests pass; regularization/threshold selection and full M4 feasibility remain outstanding |
| 2026-09-08 | Completed full M4 selection, comparator/subgroup/bootstrap analysis and frozen policy/receipt: primary C=0.1, never-intervene, no-go; ablations inconclusive. An additive receipt-path correction preserved all prior code/results. Independent validation, byte-preserving reuse, strict post-receipt access checks and 751 focused tests pass; M5/M6 remain deferred |
| 2026-09-09 | Completed the separately declared development-only capacity round: reused 48 LR candidates and fitted 48 RF plus 48 XGB candidates across 727/79/25 features; sealed all predictions before scoring; selected nine policies; independent reconstruction, byte/mtime-preserving reuse, 198-parent hash audit, and 839 focused tests pass. RF gains are modest and exploratory; original M4 no-go remains unchanged |
| 2026-09-09 | Completed all 48 declared MLP fits on CPU in 279.14 s; selected [32] for each feature variant; numeric-only correction gains +0.425791 pp with conditional CI including zero. Preserved all sealed files through additive curve-order replay correction; independent validation, exact repeat fits, byte/mtime-preserving reuse, 810-file lineage audit, and 1,038 focused tests pass. Original M4 no-go remains unchanged |
| 2026-09-09 | Completed the frozen legacy transfer for all nine RF/XGB/MLP policies and both checkpoint families in 95.56 s, with no fitting/recalibration/threshold selection. Independent validation and repeat-build reuse pass; all 36 output files and four predecessor bundles preserve bytes/mtimes, all 931 parent hashes validate, and 1,304 focused tests pass. RF fixed-final habitat gains are +0.334 to +0.353 pp, below the prior LR/no-pairs point gain; all new gain intervals versus raw include/touch zero. M4 no-go and M5/M6 deferral remain unchanged |
| 2026-09-13 | Completed `relaxed_guardrails_lr_v1`: 27 policies from all frozen LR variants and the approved plot/harm grid, including H≤0.9R. Development selection and both-family cleaned-test reporting completed in 84.99 s; independent validation, 29,007-candidate audit, metric/bootstrap audit, 40-file byte/mtime reuse, 252-parent preservation, and 1,519 tests pass. A native reuse crash left artifacts unchanged; diagnostic retry passed, with no claimed runtime fix. Harm relaxation improves fixed-final point gains, plot relaxation has no effect, and all gain-over-raw and paired-gain intervals include zero. Original M4 no-go remains unchanged |
| 2026-09-13 | Completed `relaxed_guardrails_rf_v1`: 27 policies from all three frozen RF variants and the same plot/harm grid, 216 new application cases over both legacy families. Build, independent validation (156.82 s), repeat reuse (162.92 s), 15,984-candidate audit, full metric/bootstrap audit, 40-file and seven-parent-bundle byte/mtime preservation, 1,025-parent hashes, fourth append-only test event, and 1,784 focused tests pass. Two native-runtime crashes resumed safely without source/scientific changes; successful build attempt took 161.30 s. Relaxed fixed-final top-1 gains are +0.761 to +0.854 pp, top-3 decreases, plot relaxation has no effect, and all gain/paired intervals include zero. M4 no-go and M5/M6 deferral remain unchanged |
| 2026-09-16 | Completed `relaxed_guardrails_margin_v1`: one OOF-selected H≤0.7R margin policy, unchanged coverage/plot constraints and calibration, single-family cleaned-test application. Build (9.85 s), validation (7.38 s), repeat reuse (7.49 s), independent 4,029-candidate and 2,000-bootstrap audits, 25-file byte/mtime preservation, 1,073-parent checks, fifth append-only test event and 382 focused tests pass. More interventions reduce net benefit relative to the original heuristic: accuracy 74.387528%, weighted F1 0.736112, net +4 pooled; all gain intervals span zero. No original artifacts or M4/M5/M6 decisions change |
| 2026-09-22 | Approved and implemented pooled OOF ConfidNet study with 25-summary versus 28-input RF routers; 193 tests passed. Execution paused at first image-head float64 replay tolerance. Training-only diagnostic reproduces the 1.19913e-6 discrepancy on one row and exact native replay after 500 finite epochs. Original artifacts preserved; verification amendment pending approval, no router fitting or validation scoring. |
| 2026-09-22 | User approved exact native float32 confidence replay with diagnostic float64 differences. Implemented an additive continuation that preserves the original paused study and reuses verified extraction; 305 focused tests passed before launch. Scientific recipe and 25/28 router features are unchanged. |
| 2026-09-22 | Completed the approved OOF ConfidNet continuation: 12 heads and 8 numerical RF routers. Augmented accuracy 69.5560% versus 71.9282% baseline (−2.3723 pp, paired CI [−4.5736, −0.3865]); direct TCP 69.8601% versus fusion 71.0462%. Exact native replay, independent metrics/bootstrap/provenance audits, 364-parent checks, preservation of 1,190 prior plus 34 paused-study files, and 191-file byte/mtime-preserving reuse pass. All 305 focused tests passed; no cleaned-test scoring, retuning, or M4 change. |
| 2026-09-22 | Completed user-requested `confidnet_oof_relaxed_guardrails_v1`: all 18 policies using the prior RF plot/harm grid. Augmented relaxed accuracy 71.7762% versus 72.0499% matched numeric25; paired −0.2737 pp, conditional CI [−1.5885, +0.9896]. Harm caps 0.7/0.9 coincide and plot minima have no effect; augmented0.5 falls back to fusion. 78 selector tests, independent full-curve/metric/bootstrap checks, eight-file repeat reuse and 191-parent preservation pass. Development selection only; no new fits or cleaned-test access. |
| 2026-09-24 | Completed approved `confidnet_fold4_v1`: fixed folds-1–3 experts, 852 fold-4 fitting images, 822 validation images, 12 confidence heads and 8 numerical RF routers. Augmented finds no feasible threshold in the nine-setting grid; fusion fallback 71.3808% versus relaxed numeric25 72.0499%, conditional paired difference −0.6691 pp [−1.6299, +0.3041]. Confidence training-to-validation gaps persist. All 166 focused tests, exact native replay, independent metrics/models/provenance, 130-file byte/mtime reuse, 120 declared parent hashes, and preservation of 199 prior result files pass. Development only; original M4 decision unchanged. |
| 2026-09-24 | Completed user-requested `confidnet_fold4_test_v1`: 1,347 cleaned-test images / 531 plots, the same frozen folds-1–3 experts, all four saved confidence/router sets, 26 fixed policies. Three-expert highest TCP reaches 72.4388% versus fusion 73.1069% (−0.6682 pp, conditional CI [−1.9071, +0.5924]); numerical relaxed 73.1626%, augmented relaxed never intervenes. 130 tests, independent source/image/model/metric/bootstrap audits and 35-file byte/mtime-preserving reuse pass; all prior 130 model-bundle files and sixteen registry records preserved, sixth event appended. No refitting, test threshold search, or M4 gate change. |

| 2026-09-28 | Completed `joint_tcp_router_capacity_v1`: 128 new two/three-layer MLP fits plus 32 reused shallow controls, all 40 configurations on validation and the exact September 15 cleaned-test expert family. Fusion reproduces 74.3133%; larger numerical routers reach 74.109–74.722% at minimum 20 plots / H≤0.7R. Larger models overfit state and TCP targets; all relaxed test gains versus fusion have accuracy intervals including zero. All 271 tests, independent metric/bootstrap and repeat-fit audits, 971-file immutable reuse, preservation of 2,128 prior artifacts and 428 sources, and one append-only test event pass. No architecture/weight promotion or M4 gate change. |

## Related pipeline files

- [Multimodal pipeline README](../../multimodal/README.md)
- [Current fusion models](../../multimodal/models.py)
- [Current multimodal trainer](../../multimodal/trainer.py)
- [Agreement analysis helper](../../multimodal/agreement.py)
- [Cleaned-test dataset config](../../configs/multimodal_cs_geo_100m_cleaned_test.yaml)

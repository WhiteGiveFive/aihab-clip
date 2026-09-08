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
| Current method status | M1–M3 complete; M4 calibrated dataset preparation complete, sealed, and validated; router fitting and feasibility evaluation remain unfinished |
| Last updated | 2026-09-01 |

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

**Status:** In progress. Calibrated dataset preparation is complete; router
fitting, selection, and feasibility evaluation remain unfinished. See
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

**Status:** Planned

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

**Status:** Planned

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
| M4 Router feasibility | In progress: dataset preparation complete | Sealed 13,512-row dataset contains 727 model features per row, with 12 temperatures, four numeric transforms, and a calibrated audit table; independent validation and repeat build return `reused_valid`; 232 focused tests pass; no router has been trained | Consume the sealed dataset and fit the predeclared seed-specific logistic-router candidates |
| M5 Test-time correction | Planned | No frozen score/threshold policy exists | Proceed only after the router passes the internal development gate |
| M6 Gated residual fusion | Planned | Current model is naive concatenation | Proceed only after router go/no-go passes |
| M7 Robust confirmation | Planned | Multi-seed OOF expert evidence exists, but the selected router and full stack have no matched multi-seed or independent confirmation | Run the eventual frozen stack across matched seeds and seek independent data |
| M8 Integration and reproducibility | Planned | No new-method runner or tests | Integrate alongside each implementation milestone |

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

## Immediate next action

Continue M4 from the validated, sealed router-dataset bundle. Fit one logistic
router per training seed for each predeclared regularization candidate, using
only that seed's OOF dataset projection. Reuse the twelve expert temperatures,
four transforms, and unchanged M3 semantic builder; do not refit preparation
state. Keep development-validation outputs label-blind until candidate
predictions are sealed, and do not open the cleaned-test source.

## Change log

| Date | Change |
|---|---|
| 2026-08-24 | Created the living method specification and implementation roadmap |
| 2026-08-24 | Replaced nested outer CV for router development with one fixed grouped development holdout plus four-fold cross-fitting inside development-train; recorded the weaker evidence status and frozen-router final-refit rule |
| 2026-08-25 | Completed M1: froze fold-contained B2 encoder adaptation with legacy BGR/439 preprocessing, sealed the label-blind test identity, materialized and validated grouped assignments, and added protocol/leakage/immutability enforcement |
| 2026-08-31 | Implemented the additive M2 producer, per-seed runner, immutable checkpoint/output validation, four-seed aggregation, OOF report, and CPU synthetic acceptance tests |
| 2026-08-31 | Completed M2 GPU execution for seeds 1–4 and sealed validated 13,512-row OOF and 3,288-row label-blind validation aggregates; all 20 producer checkpoints and the OOF report reproduce |
| 2026-09-01 | Confirmed the seed-specific M3 target identity and moved all probability calibration, learned transforms, and `router_dataset.parquet` materialization to M4 |
| 2026-09-01 | Began the additive M3 implementation: pure targets, stateless 30-feature contract, schema, prevalence, leakage audit, and immutable child-bundle workflow |
| 2026-09-01 | Completed M3: validated exactly 16 train-OOF parents, sealed and revalidated 13,512 seed-specific targets plus schema/prevalence/leakage artifacts, reproduced all acceptance counts, and confirmed valid immutable reuse without calibration or a router dataset |
| 2026-09-08 | Completed the additive M4 dataset subtask: fitted twelve OOF temperatures and four transforms, sealed the 13,512×727 feature matrix plus audit/state artifacts, passed independent reconstruction and byte-preserving reuse, and passed 232 focused tests; documented intermittent native-runtime crashes; router fitting remains unfinished |

## Related pipeline files

- [Multimodal pipeline README](../../multimodal/README.md)
- [Current fusion models](../../multimodal/models.py)
- [Current multimodal trainer](../../multimodal/trainer.py)
- [Agreement analysis helper](../../multimodal/agreement.py)
- [Cleaned-test dataset config](../../configs/multimodal_cs_geo_100m_cleaned_test.yaml)

# Initial ConfidNet experiment on frozen development experts

Status: COMPLETE — approved and executed on 2026-09-22; negative result for
the declared same-training-set confidence recipe and direct comparator.

Study ID: `confidnet_initial_v1`. This is an additive development experiment;
the original M4 no-go and the existing expert/router artifacts remain unchanged.
The study does not fit an augmented router or access cleaned-test examples.

## Question and approved scope

Can learned true-class-probability estimates from each expert's hidden
representation generalize from development-train and improve failure detection
or a direct geographic override policy on development-validation?

For every existing expert training seed (1–4):

1. Reuse the image, geographic, and raw-fusion experts trained on all 3,378
   development-train images (1,300 plots).
2. Evaluate those frozen final checkpoints on their own training examples,
   with deterministic preprocessing and dropout disabled, to measure exact
   mistakes and native-T=1 TCP distributions.
3. Fit one confidence network per expert on those same development-train
   examples, giving 12 confidence networks in total.
4. Seal all final confidence states and label-blind predictions before scoring
   the 822 development-validation images (325 disjoint plots).

The inherited M2 context contains development assignments and labels for split
validation; validation labels are excluded from extracted features and confidence
fitting inputs. The fitting function has no validation-target argument. Scoring
joins validation labels only after all twelve prediction states are sealed.

## Fixed recipe

The predeclared settings are in
[`multimodal_geo_confidnet_initial_v1.json`](../../configs/multimodal_geo_confidnet_initial_v1.json).
No architecture, epoch, optimizer, or threshold search is performed.

Each confidence network receives the 256-dimensional ReLU activation before its
paired expert's final classification layer. Its architecture is
`256 -> 128 -> 128 -> 128 -> 128 -> 1`, with ReLU after the four hidden layers
and sigmoid at the output. It minimizes mean squared error against the frozen
expert's native-T=1 probability of the true training label. The fit uses Adam,
learning rate 0.001, batch size 128, 100 fixed epochs, no weight decay, dropout,
feature normalization, or encoder updates. CPU float32 fitting uses deterministic
initialization/shuffling and one compute thread. The final epoch is retained.

This follows ConfidNet's frozen-classifier confidence-regression stage, with
declared widths and budget for this dataset. It does not claim an exact
reproduction of the original paper's image benchmarks or its optional separate
confidence-encoder fine-tuning stage.

Primary paper: [Corbière et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf).

## Fixed comparisons and interpretation

Report exact frozen-expert train/validation accuracy and TCP distributions,
estimated-TCP MSE/MAE/RMSE, failure average precision and ROC AUC compared with
maximum softmax probability and predictive entropy. Report undefined detection
metrics explicitly when only one correctness class is present.

The four policies are fusion alone, always select geo on disagreement, select
geo when its native maximum probability exceeds fusion's, and select geo when
its estimated TCP exceeds fusion's. Both differences use strict `> 0`. There
is no threshold selection, minimum-coverage requirement, or development harm
guardrail in this fixed diagnostic comparator study. Report rescues, harms,
neutral wrong-to-wrong switches, net corrections, accuracy and coverage.

Paired confidence intervals resample complete plots, retaining all seed
realizations together, with 2,000 resamples and seed 20260826. This study uses
an unstratified whole-plot bootstrap; these are descriptive intervals conditional
on the fitted models and reused development split. They are not independent
confirmation and do not revise M4.

Near-perfect training classification can make TCP supervision concentrate near
one. This is a scientific finding to report, not grounds for changing the
predeclared study after observing validation. Estimated TCP is not calibrated
correctness probability, and TCP differences need not order top-1 correctness.

## Relation to the future augmented router

The initial experiment trains only three confidence heads per seed beside the
full-development-train experts. The earlier proposal of fifteen per seed adds
three paired confidence heads to each of four existing OOF expert producers.
Under that proposal, each head trains on its expert's fitting folds, then
predicts the held-out fold. For example, the experts fitted on folds 2/3/4
also provide training features and TCP targets on folds 2/3/4 for their paired
confidence heads. The frozen experts and confidence heads then predict fold 1.
Repeating this for all four held-out folds gives router training inputs for
every development-train row. Those held-out estimates become router features;
held-out expert correctness supplies router rescue/harm targets.

Training confidence heads instead on expert-OOF TCP targets is a different
design. It requires a compatible representation and additional separation when
their outputs become supervised router inputs. This initial study does not
authorize or implement either downstream router design.

## Results

All twelve heads completed 100 epochs. The final predictions were verified
before the validation-label join at `2026-09-22T15:31:02.072117+00:00`.

### Frozen experts and available training supervision

These are deterministic evaluations of the frozen final checkpoints, not the
earlier online epoch-50 training logs. Each seed has 3,378 training examples.

| Expert | Train errors, seeds 1 / 2 / 3 / 4 | Training TCP >= 0.99, range across seeds | Mean validation accuracy |
|---|---|---:|---:|
| Image | 0 / 0 / 0 / 0 | 93.49–95.56% | 68.6740% |
| Geo | 68 / 83 / 66 / 101 | 25.07–26.44% | 61.2530% |
| Fusion | 0 / 0 / 0 / 0 | 96.54–97.84% | 71.0462% |

Thus the image and fusion heads receive no training failures. Their TCP targets
still vary, but TCP equals maximum softmax probability on every correctly
classified training example. Geo supplies some actual failures, at training
error rates of 1.95–2.99%, versus approximately 39% on validation.

### Failure detection and estimated-confidence saturation

Failure average precision below is the arithmetic mean of the four seed-specific
values; higher is better. The 822 validation images are shared across seeds.

| Expert | ConfidNet estimated TCP | Maximum softmax probability | Entropy |
|---|---:|---:|---:|
| Image | 0.3516 | 0.6039 | 0.6212 |
| Geo | 0.5210 | 0.5850 | 0.5740 |
| Fusion | 0.2460 | 0.5358 | 0.5411 |

Estimated TCP is worse than maximum softmax probability for both failure AP
and ROC AUC in all twelve expert/seed comparisons. Its validation TCP MSE is
also higher in all twelve cases. Mean estimated-TCP versus MCP ROC AUC is
0.4915 versus 0.7805 for image, 0.6528 versus 0.7144 for geo, and 0.3994 versus
0.7651 for fusion.

Image estimates are all above 0.9999988 and fusion estimates all above
0.9999996 on validation. Their tiny residual differences should not be treated
as a useful confidence range. In float32 training, seven of the eight
image/fusion heads have exactly zero maximum gradient by epoch five; the
remaining image head also has very small gradients. The nearly constant-one
outputs therefore expose sigmoid saturation and an optimization failure under
this recipe. Reported float64 replay AP for these saturated heads ranks tiny
residual differences and should not be overinterpreted. Independent native
float32 replay returns exactly one on all 822 validation images for six of
these eight heads, and on 821 and 820 images for the remaining two. Their
training MSE matches an always-one predictor within 2.8e-10. Geo estimates
retain variation, but their median
is 0.922–0.932 while the actual validation TCP median is 0.639–0.656. The strong
geo training failure AP (0.8507–0.9629) does not transfer to validation.

Scarce informative in-sample supervision and the train/validation generalization
gap coexist with the observed saturation. This experiment does not isolate
their causal contributions, or rule out effects of input scaling and
architecture. Low training MSE alone does not establish a useful confidence
fit or optimizer convergence. These results cannot establish an intrinsic
limit of TCP supervision or the expert representations.

### Fixed direct-policy comparison

Counts pool four realizations of the same validation set: 3,288 seed-image
rows, representing 822 distinct images on 325 plots. Accuracies are also the
mean of the four seed accuracies because each seed has the same row count.

| Policy | Accuracy | Gain versus fusion | Interventions | Rescues | Harms | Net corrections | Paired 95% gain interval |
|---|---:|---:|---:|---:|---:|---:|---|
| Fusion alone | 71.0462% | 0 pp | 0 | 0 | 0 | 0 | [0, 0] pp |
| Always geo on disagreement | 61.2530% | -9.7932 pp | 1,247 | 344 | 666 | -322 | [-14.8667, -4.8288] pp |
| Positive native MCP difference | 71.2591% | +0.2129 pp | 318 | 124 | 117 | +7 | [-1.5288, +1.8773] pp |
| Positive estimated TCP difference | 71.0462% | 0 pp | 0 | 0 | 0 | 0 | [0, 0] pp |

The estimated-TCP policy never chooses geo on a disagreement, in any seed.
Geo's estimated TCP is below fusion's on every validation row. Independent
float32 and float64 replays give identical intervention decisions, so the
no-intervention result is not caused by the portable replay's rounding.
Its zero-width observed bootstrap interval follows from taking exactly the
fusion actions on every scored row; it is not a claim of certainty on unseen
data. The MCP comparator's small positive point gain remains uncertain.

### Conclusion and next research decision

The approved initial study does not support using these confidence heads as
an improved direct policy. It also provides no positive evidence yet for
adding these particular estimates to the router; that downstream experiment
has not been run. Preserve this negative result without validation-driven
retuning of its declared recipe.

A separately declared follow-up should first diagnose saturation using training
data, simple constant predictors, and checks of feature scaling and gradients.
It could then compare confidence supervision from examples unseen by the
paired expert. That is a new hypothesis, not a conclusion that held-out fitting
must work better. An augmented router
would additionally require the separation described above. Neither follow-up
is implemented or authorized by this initial experiment.

Machine-readable metrics and the complete per-seed table are in
[`report.json`](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_initial_v1/scores/report.json)
and [`report.md`](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_initial_v1/scores/report.md).

## Commands and verification

```bash
python tools/run_multimodal_geo_helpfulness_confidnet.py build
python tools/run_multimodal_geo_helpfulness_confidnet.py validate
```

Individual `declare`, `run-seed --seed N`, and `score` phases support reuse of
completed immutable stages. The runner keeps extraction, confidence fitting,
and scoring artifacts separate. Declaration, feature, state, prediction, and
score hashes must validate before reuse; no overwrite option is provided.

The declaration was sealed at `2026-09-22T15:14:31.299812+00:00`, binding 49
source/config/checkpoint files. Its SHA-256 is
`4be4ee6eb0975bc1b186ee28d5c64ffc1afba98867c7201e04a1cdd4b68122c5`.
All 75 focused extraction/fitting/scoring/workflow tests passed before execution.

The end-to-end build completed, including validation of all twelve models and
the scored bundle. A second complete build reused every stage and preserved
the bytes and modification times of all 61 result files. Independent checks
confirmed all 49 declared source hashes, including 16 expert checkpoint files,
and all 24 train/validation prediction tables. Portable JSON replay exactly
reproduced saved predictions; independent float32 replay differed by at most
1.8644e-7, within the declared 1e-6 tolerance. Independent scikit-learn failure
AP/AUC agreed within 2.22e-16, and every per-seed and pooled policy count and
accuracy agreed with the report.
An independent 2,000-resample whole-plot bootstrap exactly reproduced the MCP
gain interval. Repeating the full seed-1 geo fit in memory reproduced its
weights, all 100 history entries, and train/validation predictions exactly.

The confidence network is fitted in float32. Its JSON weights are independently
replayed in float64; these portable predictions are authoritative for the fixed
strict-difference policy. Native/portable predictions must agree within 1e-6.
Saved M2 validation logits and argmax predictions must reproduce before any
confidence fit for a producer is published.

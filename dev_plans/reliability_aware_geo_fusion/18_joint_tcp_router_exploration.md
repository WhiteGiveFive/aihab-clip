# Joint four-state router with auxiliary TCP supervision

Study ID: `joint_tcp_router_v1`. Status: complete; all 64 fits, scoring, independent audits, and immutable runner reuse verification passed.
The user approved this complete design on 2026-09-28, including the full comparison,
fixed training budget, and reporting every auxiliary weight without selecting one winner.

## Approved question and grid

Does TCP supervision improve the representation used for predicting fusion/geo
rescue, harm, both correct, and both wrong? Distinguish this contribution from
the benefit of direct image and geo embeddings.

| Inputs | Width | Parameters, including both heads |
| --- | ---: | ---: |
| Numerical summaries | 25 | 1,030 |
| Without paired habitat predictions | 79 | 2,758 |
| Numerical summaries + image/geo embeddings | 1,241 | 39,942 |
| Without pairs + image/geo embeddings | 1,295 | 41,670 |

The 79 columns are the 25 numerical summaries and three 18-class individual
expert-prediction one-hot blocks. The 648 paired-prediction columns are excluded.
Every input setting is crossed with auxiliary weights 0, 0.1, 1, and 10 and
seeds 1, 2, 3, and 4: 16 configurations and 64 independent fits.

## Data and supervision

Use all 3,378 development-training images from 1,300 plots per seed, assembled
from four plot-held-out expert producers. Validation comprises 822 images from
325 separate plots, produced by full-development-train experts. Seeds remain
separate during fitting and are not counted as independent image replicates.

Reuse existing authenticated numerical transformations, temperatures, and one-hot
vocabularies. A new cache replays twenty frozen producer bundles to retain their
1,152-dimensional L2-normalized image inputs and 64-dimensional geo inputs after
each producer's own fitting-partition standardization. No additional embedding
scaling, projection, encoder fitting, or common pretrained encoder is introduced.
The producer shift at validation is an acknowledged limitation.

Extracted embeddings must reproduce saved expert logits at atol=rtol=1e-6 and
predict exactly the same classes. Saved M2 logits remain authoritative. Define
the four-state target from those expert predictions and the two TCP targets from
native-temperature-one softmax at the ground-truth class. TCP output order is
fusion then geo. The existing temperature-calibrated router features do not
change the TCP target definition.

Validation labels are absent from feature and fitting interfaces. Join labels
only after all 64 models and predictions pass replay checks. This round does
not access cleaned-test data or update any expert.

## Network and fitting recipe

Shared trunk: Linear(D,32), ReLU, Dropout(0.1). Parallel heads are Linear(32,4)
for state logits and Linear(32,2) for TCP logits. Cross-entropy receives raw
state logits; softmax provides four-state probabilities. Independent sigmoids
provide TCP estimates before mean squared error. Sigmoid plus MSE follows
[ConfidNet's implementation](https://github.com/valeoai/ConfidNet/blob/master/confidnet/utils/losses.py);
the shared router and its smaller architecture are this study's adaptation.

Loss is equal-image mean state cross-entropy plus lambda times the mean squared
TCP error averaged over both experts and all images. Targets and embeddings are
frozen. The lambda-zero head is allocated but receives no regression gradients;
its outputs are explicitly unsupervised. Initialization, shuffling, and dropout
streams match across lambda within each input setting and seed.

All fits use 300 epochs and the final checkpoint, batch size 128 including the
last partial batch, AdamW learning rate 0.001, betas (0.9,0.999), epsilon 1e-8,
weight decay 0.001 on weights and zero on biases. Computation is deterministic
CPU float32 with one training thread in the existing PyTorch 2.5.1 environment.
No early stopping, scheduler, class weighting, architecture search, or automatic
retuning is included. Frozen-encoder replay may use CUDA without changing CPU
router fitting. Nonfinite loss, gradients, parameters, or predictions stop fitting.

Record each loss component, weighted auxiliary loss, dropout-disabled training
metrics, component/shared gradient norms, gradient cosine, and TCP saturation.
Native float32 serialized prediction replay must be exact. Independent float64
differences are diagnostics rather than an alternative acceptance tolerance.

## Evaluation

Promote saved float32 probabilities to float64 before calculating q_rescue-q_harm.
Switch only when geo/fusion disagree and the score is strictly greater than the
selected threshold. Apply the existing nine relaxed rules: minimum plots 20/10/0,
harm caps 0.5/0.7/0.9 times rescues, at least 1% pooled coverage, and positive net
corrections. One threshold serves all seeds for each configuration/rule. Reuse
candidate enumeration, tie-breaking, and never-intervene fallback exactly.

Report all 144 selected policies, 16 fixed-zero policies, fusion, and historical
matched-expert development RF context. Select no lambda or guardrail winner.
Primary comparisons are each positive lambda against zero with identical inputs;
embedding comparisons match the base features and lambda.

Report per-seed and mean routing accuracy/F1, rescues, harms, net corrections,
coverage, and plots; state NLL/Brier/accuracy/macro-F1 and rescue/harm ranking;
TCP MSE/MAE/failure AP/AUC/saturation/ordering; and training-validation gaps.
Canonical NLL uses stable log-softmax from saved state logits; probability-based
NLL and float32 zero-probability counts are separate numerical diagnostics.
Bootstrap 2,000 whole-plot draws with seed 20260826, retaining all images and seed
realizations together. Hold selected policies fixed in intervals. This is
exploratory reused-validation evidence, not independent confirmation or a change
to the original M4 decision.

## Execution and artifact contract

The versioned config is `configs/multimodal_geo_joint_tcp_v1.json`. The runner is
`tools/run_multimodal_geo_helpfulness_joint_tcp_v1.py`, with prepare, fit, score,
validate, build and individual declare/prepare-seed/fit-seed stages. The original
unversioned runner performed production fitting and is retained unchanged for its
sealed provenance. The supported versioned runner includes only the validation
rendering correction documented below. Build runs
fitting and scoring as separate processes. Artifacts are isolated under
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1`.

Before fitting, publish the config, approved plan text, environment, source and
checkpoint hashes, input order, and prior development artifact snapshots. Save
per-producer/combined embedding caches, models, all epoch histories, predictions
with logits and identities, replay diagnostics, threshold curves, action masks,
bootstrap draws, and machine-readable/human-readable reports. Stages are sealed
and immutable; repeat execution authenticates and reuses their bytes.

From the repository root with the `habcls` environment active, reproduce the
cached build and its validation with:

```bash
python tools/run_multimodal_geo_helpfulness_joint_tcp_v1.py build
python tools/run_multimodal_geo_helpfulness_joint_tcp_v1.py validate
```

## Results recorded on 2026-09-28

All twenty frozen producer bundles were replayed and all **64 final-epoch router fits** completed. The sealed scoring bundle contains 144 relaxed policies, 16 fixed-zero policies, the fusion reference, and all 360 planned bootstrap contrasts. No auxiliary weight or guardrail was selected as a winner. The final versioned runner build, immutable reuse check, and separate artifact and numerical audits all passed.

**Joint TCP supervision did not establish a consistent improvement over the matched CE-only routers.** Under relaxed threshold selection, auxiliary-versus-zero accuracy differences range from **−0.517 to +0.274 percentage points**, and every one of the 108 corresponding confidence intervals includes zero. Embedding inputs changed the routing decisions, but their state and confidence heads strongly overfit the development-training examples.

### Routing across every configuration

The full-development-train fusion reference is **71.0462% accuracy** and **0.70670 weighted F1** on the 822 validation images. These are the development experts used throughout this study; they are not the legacy full-data experts or the earlier single-fold expert family. All displayed accuracies are arithmetic means over four seeds.

Each cell below is validation accuracy in percent. For all **48 configuration/harm-cap combinations**, the selected hexadecimal threshold is identical at minimum-acted-plot limits 20, 10, and 0. The three plot limits can therefore be collapsed in this table without omitting distinct results. The fixed-zero column applies strict score > 0 and the disagreement gate, with no threshold fitting.

| Inputs | λ | Fixed zero | H ≤ 0.5R | H ≤ 0.7R | H ≤ 0.9R |
| --- | ---: | ---: | ---: | ---: | ---: |
| Numerical 25 | 0 | 71.077 | 71.442 | 71.594 | 71.594 |
| Numerical 25 | 0.1 | 71.472 | 71.442 | 71.442 | 71.472 |
| Numerical 25 | 1 | 71.168 | 71.655 | 71.655 | 71.655 |
| Numerical 25 | 10 | 71.107 | 71.442 | 71.655 | 71.655 |
| Numerical 25 + embeddings | 0 | 70.164 | 71.533 | 71.533 | 71.533 |
| Numerical 25 + embeddings | 0.1 | 70.803 | 71.807 | 71.807 | 71.807 |
| Numerical 25 + embeddings | 1 | 70.560 | 71.655 | 71.655 | 71.655 |
| Numerical 25 + embeddings | 10 | 71.046 | 71.533 | 71.563 | 71.563 |
| Without pairs 79 | 0 | 70.255 | 71.046 | 71.046 | 71.168 |
| Without pairs 79 | 0.1 | 70.681 | 71.046 | 71.046 | 71.320 |
| Without pairs 79 | 1 | 71.016 | 71.046 | 71.046 | 71.229 |
| Without pairs 79 | 10 | 70.286 | 71.046 | 71.046 | 71.381 |
| Without pairs 79 + embeddings | 0 | 70.347 | 71.563 | 71.563 | 71.563 |
| Without pairs 79 + embeddings | 0.1 | 70.225 | 71.655 | 71.655 | 71.655 |
| Without pairs 79 + embeddings | 1 | 70.255 | 71.046 | 71.381 | 71.381 |
| Without pairs 79 + embeddings | 10 | 70.529 | 71.655 | 71.655 | 71.655 |

The never-intervene fallback is part of the outcome: **27 of 144 selected policies** return fusion. Every non-embedding 79-column arm returns fusion under harm caps 0.5 and 0.7, and the embedding 79-column λ=1 arm returns fusion under cap 0.5. **11 of all 16 fixed-zero policies** reduce accuracy below fusion, including eight of the twelve positive-λ policies; numerical summaries without embeddings remain above fusion, and numerical summaries with embeddings at λ=10 tie fusion. Thus, positive selected-threshold gains should not be interpreted as an unconditional improvement in router decisions.

The [human-readable scored report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/scores/report.md) presents mean accuracy/weighted F1 and pooled intervention counts. Full per-seed and pooled weighted F1, rescues, harms, net corrections, coverage and affected plots are in the [machine-readable report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/scores/report.json). Counts pool seed–image evaluations; the 3,288 validation evaluations still represent 822 images and 325 plots.

### Auxiliary supervision versus matched CE-only controls

Each row compares one positive λ with λ=0 using the same inputs, architecture, seed pairing and guardrail. The difference range covers the three harm caps; the repeated plot limits have identical results. The interval envelope is the smallest lower bound and largest upper bound among those three individual 95% intervals. It is a compact description of the reported intervals, **not a confidence interval for an aggregate effect**. All 108 individual relaxed-policy accuracy intervals include zero.

| Inputs | λ | Accuracy difference range (pp) | Envelope of individual 95% CIs (pp) |
| --- | ---: | ---: | ---: |
| Numerical 25 | 0.1 | -0.152 to +0.000 | [-1.314, +1.080] |
| Numerical 25 | 1 | +0.061 to +0.213 | [-0.307, +0.528] |
| Numerical 25 | 10 | +0.000 to +0.061 | [-0.360, +0.460] |
| Numerical 25 + embeddings | 0.1 | +0.274 | [-0.090, +0.662] |
| Numerical 25 + embeddings | 1 | +0.122 | [-0.212, +0.456] |
| Numerical 25 + embeddings | 10 | +0.000 to +0.030 | [-0.543, +0.561] |
| Without pairs 79 | 0.1 | +0.000 to +0.152 | [-0.302, +0.604] |
| Without pairs 79 | 1 | +0.000 to +0.061 | [-0.404, +0.491] |
| Without pairs 79 | 10 | +0.000 to +0.213 | [-0.342, +0.794] |
| Without pairs 79 + embeddings | 0.1 | +0.091 | [-0.204, +0.394] |
| Without pairs 79 + embeddings | 1 | -0.517 to -0.182 | [-1.125, +0.242] |
| Without pairs 79 + embeddings | 10 | +0.091 | [-0.431, +0.602] |

The fixed-zero comparisons are retained separately. Numerical 25 + embeddings at λ=0.1 improves on its λ=0 fixed-zero control by **+0.639 pp**, 95% CI **[+0.031, +1.208]**; the weighted-F1 difference is +0.00587, CI [−0.00021, +0.01148]. However, both policies remain below fusion: their accuracy gains versus fusion are −0.243 pp and −0.882 pp, respectively. This single exploratory interval does not establish a successful replacement for the existing router.

### Embeddings versus matched non-embedding controls

These comparisons hold the numerical/categorical feature set and λ fixed. The same interval-envelope convention applies.

| Embedding inputs | λ | Accuracy difference range (pp) | Envelope of individual 95% CIs (pp) |
| --- | ---: | ---: | ---: |
| Numerical 25 + embeddings | 0 | -0.061 to +0.091 | [-0.662, +0.549] |
| Numerical 25 + embeddings | 0.1 | +0.335 to +0.365 | [-1.019, +1.665] |
| Numerical 25 + embeddings | 1 | +0.000 | [-0.447, +0.438] |
| Numerical 25 + embeddings | 10 | -0.091 to +0.091 | [-0.702, +0.553] |
| Without pairs 79 + embeddings | 0 | +0.395 to +0.517 | [-0.184, +1.125] |
| Without pairs 79 + embeddings | 0.1 | +0.335 to +0.608 | [-0.374, +1.104] |
| Without pairs 79 + embeddings | 1 | +0.000 to +0.335 | [-0.529, +0.873] |
| Without pairs 79 + embeddings | 10 | +0.274 to +0.608 | [-0.636, +1.266] |

Across the 72 relaxed embedding comparisons, effects range from **−0.091 to +0.608 pp**. The 79-column λ=0.1 embedding arm improves on its non-embedding counterpart under caps 0.5 and 0.7 by **+0.608 pp**, CI **[+0.210, +1.074]**. Both non-embedding controls fall back to fusion. This is one repeated action comparison across six guardrail settings, not six independent findings. At fixed zero, embeddings decrease accuracy in six of eight matched comparisons; none of the fixed-zero embedding accuracy intervals excludes zero.

### State prediction and overfitting

Each entry is mean training NLL → mean validation NLL, evaluated with dropout disabled. The λ=0 arm is the CE-only control.

| Inputs | λ=0 | λ=0.1 | λ=1 | λ=10 |
| --- | ---: | ---: | ---: | ---: |
| Numerical 25 | 0.5029 → 0.6562 | 0.5047 → 0.6595 | 0.5038 → 0.6531 | 0.5144 → 0.6450 |
| Numerical 25 + embeddings | 0.0039 → 1.9085 | 0.0039 → 1.9057 | 0.0042 → 1.8627 | 0.0094 → 1.9036 |
| Without pairs 79 | 0.3593 → 0.8426 | 0.3599 → 0.8294 | 0.3627 → 0.8130 | 0.3887 → 0.7545 |
| Without pairs 79 + embeddings | 0.0024 → 1.9654 | 0.0026 → 1.9424 | 0.0026 → 1.9299 | 0.0067 → 1.9248 |

The numerical-only models retain a much smaller training–validation gap. Auxiliary supervision modestly reduces validation NLL in some settings, especially the 79-column model at larger weights, but that does not translate into a consistent matched routing gain.

The embedding models approach perfect training state prediction: NLL **0.0024–0.0094**, compared with **1.8627–1.9654** on validation. The additional inputs increase the parameter count from 1,030/2,758 to 39,942/41,670 despite the fixed 32-unit hidden layer. Limited training size, high capacity relative to that size, and the acknowledged producer-specific representation shift are plausible contributors. This study does not isolate their individual effects. The auxiliary loss does not resolve the observed overfitting under this fixed recipe.

### TCP quality, failure detection and relative ordering

The table includes only supervised TCP heads (λ>0). Regression MSE and failure-detection AP are four-seed means, shown as training → validation. Failure AP uses 1 − estimated TCP as its failure score. The λ=0 TCP outputs are unsupervised and are excluded from this table. Ordering columns pool the four seeds and give the percentage of actual rescue/harm cases in which the estimated TCP ranks the correct expert higher.

| Inputs | λ | Fusion MSE train → val | Geo MSE train → val | Fusion AP train → val | Geo AP train → val | Rescue ordering (%) | Harm ordering (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Numerical 25 | 0.1 | 0.1143 → 0.1270 | 0.0707 → 0.0838 | 0.6498 → 0.5743 | 0.8102 → 0.7282 | 28.49 | 87.54 |
| Numerical 25 | 1 | 0.1132 → 0.1276 | 0.0696 → 0.0829 | 0.6556 → 0.5691 | 0.8046 → 0.7271 | 28.78 | 86.19 |
| Numerical 25 | 10 | 0.1102 → 0.1280 | 0.0671 → 0.0830 | 0.6656 → 0.5637 | 0.8029 → 0.7191 | 27.62 | 85.89 |
| Numerical 25 + embeddings | 0.1 | 0.0445 → 0.1616 | 0.0266 → 0.1141 | 0.9343 → 0.5098 | 0.9901 → 0.6703 | 37.50 | 82.73 |
| Numerical 25 + embeddings | 1 | 0.0156 → 0.1681 | 0.0149 → 0.1135 | 0.9902 → 0.5159 | 0.9958 → 0.6655 | 44.19 | 78.23 |
| Numerical 25 + embeddings | 10 | 0.0042 → 0.1673 | 0.0054 → 0.1137 | 0.9994 → 0.5252 | 0.9979 → 0.6678 | 47.67 | 75.38 |
| Without pairs 79 | 0.1 | 0.1042 → 0.1329 | 0.0646 → 0.0889 | 0.7233 → 0.5360 | 0.8540 → 0.7222 | 29.36 | 84.98 |
| Without pairs 79 | 1 | 0.0987 → 0.1329 | 0.0613 → 0.0878 | 0.7489 → 0.5398 | 0.8555 → 0.7263 | 30.81 | 84.08 |
| Without pairs 79 | 10 | 0.0890 → 0.1370 | 0.0543 → 0.0897 | 0.7869 → 0.5347 | 0.8649 → 0.7087 | 32.56 | 80.93 |
| Without pairs 79 + embeddings | 0.1 | 0.0427 → 0.1599 | 0.0260 → 0.1169 | 0.9455 → 0.5224 | 0.9913 → 0.6914 | 44.19 | 80.78 |
| Without pairs 79 + embeddings | 1 | 0.0132 → 0.1678 | 0.0134 → 0.1194 | 0.9944 → 0.5200 | 0.9972 → 0.6758 | 46.22 | 74.62 |
| Without pairs 79 + embeddings | 10 | 0.0035 → 0.1694 | 0.0042 → 0.1165 | 0.9995 → 0.5193 | 0.9981 → 0.6712 | 47.38 | 74.32 |

The joint heads do learn confidence information. For comparison, validation MCP failure-detection AP is **0.5358 for fusion** and **0.5850 for geo**. Non-embedding joint heads achieve fusion AP 0.5347–0.5743 and geo AP 0.7087–0.7282. The geo confidence task therefore improves substantially over MCP without delivering a corresponding reliable gain from the four-state router.

Embedding heads exhibit the earlier confidence-generalization problem: fusion training AP reaches 0.9343–0.9995 but validation AP is 0.5098–0.5252, below fusion MCP for every embedding arm. Geo AP falls from 0.9901–0.9981 to 0.6655–0.6914. Their validation MSE is also worse than that of the non-embedding heads. Extreme validation TCP saturation (≤10⁻⁶ or ≥1−10⁻⁶) is absent from the supervised non-embedding arms but reaches 10.83% for one embedding arm/expert mean.

True TCP ranks the correct expert higher in **all 344 rescue and 666 harm validation seed–image cases**. Estimated TCP preserves rescue ordering in only **27.62–32.56%** of cases without embeddings and **37.50–47.67%** with embeddings; corresponding harm ordering is **80.93–87.54%** and **74.32–82.73%**. Good within-expert failure AP therefore does not imply that the two confidence heads are comparable enough to identify when geo should replace fusion. The reported routing policies still use the four-state score, not a maximum-TCP rule.

### Existing RF context

These previously fitted RF routers use the same development expert predictions. Their configurations were inherited from earlier work; they are context, while the matched λ=0 MLPs are the controls for attributing effects to TCP supervision. Plot-limit settings also select identical RF thresholds in this comparison.

| RF inputs | Gain at H ≤ 0.5R (pp) | Gain at H ≤ 0.7R (pp) | Gain at H ≤ 0.9R (pp) |
| --- | ---: | ---: | ---: |
| Numerical 25 | +0.730 | +1.004 | +1.004 |
| Without pairs 79 | +0.669 | +0.882 | +0.882 |

The joint study does not establish a consistent advance over this existing RF context. Reporting all settings exposes both small improvements over particular MLP controls and configurations that overfit or fall back to fusion.

## Verification and immutable reuse

- **183 targeted and regression tests passed**: the original 179 checks and four tests for the additive validation correction. They cover extraction/OOF identities, feature widths, native targets, label isolation, activations and shared gradients, λ=0 CE-only equivalence, serialization, and threshold/metric behavior.
- The [independent fit audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1_review/independent_fit_audit.json) checks all 20 producer caches and 64 model artifacts, original numerical features, producer-specific embedding/scaler inputs, final losses, matching initialization and RNG recipes, unchanged λ=0 TCP heads, and exact native float32 predictions. Minibatch/dropout streams were independently reconstructed from the sealed recipe; they were not runtime traces recorded during fitting.
- The [independent scoring audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1_review/independent_metrics_audit.json) passed **5,043,216 assertions**: all 161 policies, all 144 threshold selections and **191,970 candidate-curve records**, 2,000 bootstrap draws over 37 unique action masks, and all **360 contrasts**. It uses sklearn metrics, independent threshold counting/tie selection, and independent weighted confusion matrices without importing production scoring helpers. Maximum numerical disagreement is **4.44×10⁻¹⁶**.
- [Two exact 300-epoch refits](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1_review/deterministic_refits.json)—seed 1 numerical 25, λ=1, and numerical 25 + embeddings, λ=1—reproduce the entire serialized model, training history, and native prediction arrays. They are repeatability checks, not additional candidate selections.
- Float64 replay remains diagnostic: maximum absolute differences are 2.73×10⁻⁶ for state probabilities, 2.11×10⁻⁵ for state logits, 9.81×10⁻⁷ for TCP estimates, and 8.79×10⁻⁶ for TCP logits. No state argmax changes occur. Eleven training TCP-order comparisons change under float64 replay; no validation TCP-order or utility-sign comparisons change. Native float32 values remain authoritative.
- The [versioned runner immutable reuse check](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1_review/immutable_reuse.json) passed with exit code zero: all **540 study files** retain identical SHA256, size and modification time. All twenty producer caches and 64 fitted models were reused; previous-study snapshot checks also passed.

### Additive validation-rendering correction

The original validator compared Markdown rendered from the JSON-loaded report
with Markdown rendered before JSON serialization. Sorted JSON mapping keys
changed table order even though all numerical data matched. The supported
versioned validator now compares the recomputed and stored numerical reports
exactly, then renders the recomputed report in its original insertion order for
the Markdown check. This changes only validation rendering; fitting, scoring,
models, predictions, metrics and sealed artifacts remain unchanged. The original
91 declared Python sources are intact. See the
[versioned correction receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1_review/validation_correction_v1.json).

## Artifacts and interpretation limits

- [Declaration, approved recipe and source/checkpoint hashes](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/declaration.json).
- [Complete human-readable scoring report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/scores/report.md) and [machine-readable report with all 360 contrasts](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/scores/report.json). The contrasts include 160 policy-versus-fusion, 120 positive-λ-versus-zero, and 80 embedding-versus-no-embedding comparisons, including fixed-zero policies.
- [All threshold candidates and selections](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/scores/selections.json), [scored train/validation rows](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/scores/scored_rows.parquet), [validation action masks](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/scores/actions.parquet), and [all bootstrap draws](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/scores/bootstrap_draws.npz).
- [Auxiliary contrast figure](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1_review/figures/auxiliary_contrasts.png), [embedding contrast figure](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1_review/figures/embedding_contrasts.png), and [fusion-gain figure](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1_review/figures/fusion_gains.png). Each has PDF and CSV companions in the same directory.

The same previously explored validation set selects thresholds and supplies the reported effects. Bootstrap intervals condition on those selected thresholds; they do not repeat threshold fitting, adjust for the 360 comparisons, or establish independent confirmation. Identical policies across guardrails are repeated presentations of the same evidence. The four seeds share the same images and plots.

Under the declared architecture and 300-epoch recipe, confidence supervision can improve individual confidence estimates and some state metrics, but its benefit to routing remains uncertain. Direct OOF embeddings substantially increase overfitting, and the experiment does not separate statistical overfitting from representation shift. No expert was retrained, no cleaned-test labels were accessed, and no auxiliary weight, architecture, or guardrail was promoted for deployment.

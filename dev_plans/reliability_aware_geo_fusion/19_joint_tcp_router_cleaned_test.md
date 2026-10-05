# Frozen joint TCP routers on cleaned test

Study ID: `joint_tcp_router_test_v1`. Status: complete; all 64 frozen routers applied, scoring and independent audits passed, and complete-run immutable reuse verified.

The user requested applying the routers from the joint TCP development study to
correct fusion on cleaned test. This application evaluates every declared input
configuration and auxiliary weight, with the existing CE-only controls and
validation-selected thresholds.

Matching fusion accuracy is **73.7379%**. Numerical-25 routers reach
**73.8679–74.3504%** under the transferred relaxed thresholds, while every
embedding policy is at or below fusion. Auxiliary TCP supervision does not
show a clear improvement over the matched CE-only routers. All 160
router-versus-fusion accuracy intervals include zero.

## Frozen application setup

- Reuse all 64 saved joint routers: 25 numerical summaries or 79 features without
  paired predictions, each with/without image and geo embeddings, auxiliary
  weights 0/0.1/1/10, and seeds 1–4. No router or expert is fitted again.
- For each seed, use the full-development-train encoder, image/geo/fusion heads,
  and producer-specific geo scaler that generated development-validation inputs.
  These are different expert checkpoints from the earlier single-fold and legacy
  test applications. Compare each router with its matching fusion predictions.
- Apply to the existing cleaned-test identity: 1,347 images and 531 plots. Read
  image identities and native geo inputs without labels or cached image features.
  Extract the matching 1,152-dimensional normalized image embedding and
  64-dimensional standardized geo input. Keep frozen numerical temperatures,
  transforms, category vocabularies, and feature ordering.
- Use native CPU float32 joint-router inference. Promote probabilities to float64
  before subtracting `q_rescue - q_harm`; intervene only on geo/fusion disagreement
  and strict `score > threshold`. TCP heads remain auxiliary diagnostics.
- Transfer all 144 validation-selected thresholds and all 16 fixed-zero policies.
  Never-intervene policies stay unchanged. Do not search thresholds, enforce
  guardrails again on test outcomes, recalibrate, or select a winning weight.
- Report the 161 primary policies, including fusion, plus image/geo expert context.
  Report per-seed and mean habitat accuracy, weighted F1, rescues, harms, net
  corrections, coverage and affected plots, state prediction and TCP quality.
- Retain all 360 declared paired contrasts: policy versus fusion, positive lambda
  versus matched zero, and embeddings versus matched non-embedding inputs.
  Use 2,000 paired whole-plot bootstrap draws, seed 20260826, retaining all images
  and four seed realizations of each sampled plot. Hold models and thresholds fixed.
- Authenticate the complete development study and reproduce its validation
  matrices, 64 native predictions and 161 policies before inference. Publish the
  application configuration, source/model/image hashes, preprocessing, copied
  threshold records and prior-artifact snapshots before test inference.
- Seal every seed's embeddings, predictions and intervention choices before
  reserving the next append-only test event and acquiring canonical labels.
  Revalidation uses the local authenticated label copy.

The test set has been examined in earlier research. These results remain
exploratory evidence, conditional on frozen models and transferred thresholds.
The application does not alter the original M4 conclusion or earlier artifacts.

## Implementation and reproduction

Configuration: `configs/multimodal_geo_joint_tcp_test_v1.json`.
Separate extraction, inference/orchestration, scoring and registry components
use the existing authenticated inputs and pure evaluation helpers. The runner
has no fitting or threshold-selection option.

From the repository root:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_multimodal_geo_helpfulness_joint_tcp_test.py build --device cuda
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_multimodal_geo_helpfulness_joint_tcp_test.py validate
```

Individual stages are `declare`, `extract-seed --seed N`, `predict-seed --seed N`,
`score`, `validate-scores`, and `validate`. Complete stages authenticate and reuse
their saved artifacts.

Output directory:
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1`.

## Cleaned-test results recorded on 2026-09-28

All **64 frozen joint routers** were applied to the **1,347 cleaned-test images from 531 plots**. The study evaluates all 144 transferred validation-threshold policies and 16 fixed-zero policies against their matching fusion expert, retaining all 360 declared bootstrap contrasts. No model was retrained and no threshold, auxiliary weight or winner was selected using test outcomes.

**Numerical summaries without embeddings retain useful corrective decisions, but auxiliary TCP supervision still does not show a consistent gain over matched CE-only routers.** Embedding inputs reduce accuracy in every non-tied relaxed comparison with the corresponding non-embedding model. The positive validation gains of the embedding models do not transfer to this test application.

### Frozen expert references

These are the **full-development-train expert bundles**, one per training seed, fitted on 3,378 images / 1,300 plots and paired with the image encoder and geo scaler that generated the routers' development-validation inputs. The same frozen bundle now generates test predictions and embeddings; the routers remain the previously fitted four-fold OOF models. The fusion reference is therefore **73.7379% accuracy and 0.728919 weighted F1**. It is a different expert family from the earlier single-fold-4 application and the legacy experts reported at approximately 74.31% on cleaned test. The cleaned-test identity is unchanged; differences between those fusion results reflect the different expert checkpoints and training splits. Each routing comparison here uses its own matching fusion predictions.

| Expert | Accuracy (%) | Weighted F1 |
| --- | ---: | ---: |
| Fusion | 73.7379 | 0.72892 |
| Image | 72.1047 | 0.71137 |
| Geo | 60.8389 | 0.60012 |

The matched fusion baseline by training seed is:

| Seed | Accuracy (%) | Weighted F1 |
| --- | ---: | ---: |
| 1 | 74.1648 | 0.73436 |
| 2 | 73.4224 | 0.72526 |
| 3 | 74.1648 | 0.73256 |
| 4 | 73.1997 | 0.72349 |

### Routing across all sixteen configurations

All table values are four-seed mean habitat accuracy (%). The three harm-cap columns transfer the thresholds selected under those **validation** constraints; they do not enforce those caps on test labels. Fixed zero means the unchanged strict positive-score policy. For every configuration and harm cap, the transferred threshold is identical across minimum-plot limits 20, 10 and 0, so those limits also give identical test policies and can be collapsed here.

| Inputs | λ | Fixed zero | Validation H ≤ 0.5R | Validation H ≤ 0.7R | Validation H ≤ 0.9R |
| --- | ---: | ---: | ---: | ---: | ---: |
| Numerical 25 | 0 | 74.388 | 73.886 | 74.146 | 74.146 |
| Numerical 25 | 0.1 | 74.332 | 74.035 | 74.035 | 74.350 |
| Numerical 25 | 1 | 74.220 | 73.979 | 73.979 | 73.979 |
| Numerical 25 | 10 | 74.536 | 73.868 | 73.979 | 73.979 |
| Numerical 25 + embeddings | 0 | 72.866 | 73.589 | 73.589 | 73.589 |
| Numerical 25 + embeddings | 0.1 | 73.088 | 73.478 | 73.478 | 73.478 |
| Numerical 25 + embeddings | 1 | 73.144 | 73.515 | 73.515 | 73.515 |
| Numerical 25 + embeddings | 10 | 72.847 | 73.664 | 73.608 | 73.608 |
| Without pairs 79 | 0 | 73.293 | 73.738 | 73.738 | 73.719 |
| Without pairs 79 | 0.1 | 73.163 | 73.738 | 73.738 | 73.701 |
| Without pairs 79 | 1 | 73.237 | 73.738 | 73.738 | 73.794 |
| Without pairs 79 | 10 | 73.293 | 73.738 | 73.738 | 73.905 |
| Without pairs 79 + embeddings | 0 | 72.680 | 73.404 | 73.404 | 73.404 |
| Without pairs 79 + embeddings | 0.1 | 72.736 | 73.478 | 73.478 | 73.478 |
| Without pairs 79 + embeddings | 1 | 73.088 | 73.738 | 73.478 | 73.478 |
| Without pairs 79 + embeddings | 10 | 72.513 | 73.478 | 73.478 | 73.478 |

Across the 144 transferred relaxed policies, **42 improve on fusion, 75 reduce accuracy, and 27 keep fusion everywhere**. Their observed gains versus fusion range from **−0.334 to +0.612 percentage points**. These counts include repeated policies across plot limits. Every finite-threshold embedding policy reduces test accuracy below fusion; the one embedding setting that ties fusion is the unchanged never-intervene fallback.

All four numerical-25 fixed-zero policies improve on fusion, while the other **12 of 16 fixed-zero policies reduce accuracy**. The numerical-25 CE-only fixed-zero control already reaches **74.3875%**; numerical-25 λ=10 reaches **74.5360%**. Their difference is only **+0.148 pp**, with CI **[−0.243, +0.528] pp**. Thus, a router's gain over fusion does not establish a benefit from TCP supervision. Neither these fixed-zero gains nor any of the 144 transferred relaxed-policy accuracy gains has an interval excluding zero versus fusion.

The transferred numerical-25 λ=0.1 / cap-0.9 policy provides one concrete positive example: **74.3504% accuracy, 0.733770 weighted F1**, with **163 rescues, 130 harms and 33 net corrections** across 5,388 seed–image evaluations. Its coverage is 6.756%, affecting 163 distinct plots. Its accuracy gain over fusion is **+0.612 pp**, CI **[−0.452, +1.695] pp**; its gain over the matched CE-only policy is **+0.204 pp**, CI **[−0.547, +1.027] pp**. This example illustrates the effect size without promoting a winner.

Validation guardrails are not test guarantees: **90 of the 117 finite-threshold policies exceed their validation harm cap on test**. All are reported unchanged, including negative net corrections and coverage below 1%. The 27 fallback policies remain fallbacks; test labels are never used to replace a transferred policy.

### Auxiliary supervision versus matched CE-only controls

Every cell gives the accuracy difference in percentage points followed by its own 95% bootstrap interval. The reference has the same inputs and guardrail identity with λ=0. These are individual intervals; no interval envelope is used.

| Inputs | λ | Fixed zero | Validation H ≤ 0.5R | Validation H ≤ 0.7R | Validation H ≤ 0.9R |
| --- | ---: | ---: | ---: | ---: | ---: |
| Numerical 25 | 0.1 | -0.056 [-0.313, +0.185] | +0.148 [+0.000, +0.314] | -0.111 [-0.406, +0.187] | +0.204 [-0.547, +1.027] |
| Numerical 25 | 1 | -0.167 [-0.588, +0.222] | +0.093 [-0.164, +0.334] | -0.167 [-0.417, +0.038] | -0.167 [-0.417, +0.038] |
| Numerical 25 | 10 | +0.148 [-0.243, +0.528] | -0.019 [-0.207, +0.169] | -0.167 [-0.480, +0.148] | -0.167 [-0.480, +0.148] |
| Numerical 25 + embeddings | 0.1 | +0.223 [-0.114, +0.566] | -0.111 [-0.317, +0.075] | -0.111 [-0.317, +0.075] | -0.111 [-0.317, +0.075] |
| Numerical 25 + embeddings | 1 | +0.278 [-0.259, +0.838] | -0.074 [-0.339, +0.169] | -0.074 [-0.339, +0.169] | -0.074 [-0.339, +0.169] |
| Numerical 25 + embeddings | 10 | -0.019 [-0.737, +0.712] | +0.074 [-0.276, +0.419] | +0.019 [-0.333, +0.370] | +0.019 [-0.333, +0.370] |
| Without pairs 79 | 0.1 | -0.130 [-0.581, +0.281] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.019 [-0.298, +0.275] |
| Without pairs 79 | 1 | -0.056 [-0.560, +0.447] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.074 [-0.149, +0.320] |
| Without pairs 79 | 10 | +0.000 [-0.516, +0.548] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.186 [-0.254, +0.642] |
| Without pairs 79 + embeddings | 0.1 | +0.056 [-0.292, +0.401] | +0.074 [-0.169, +0.299] | +0.074 [-0.169, +0.299] | +0.074 [-0.169, +0.299] |
| Without pairs 79 + embeddings | 1 | +0.408 [-0.095, +0.941] | +0.334 [-0.149, +0.821] | +0.074 [-0.202, +0.354] | +0.074 [-0.202, +0.354] |
| Without pairs 79 + embeddings | 10 | -0.167 [-0.818, +0.513] | +0.074 [-0.276, +0.425] | +0.074 [-0.276, +0.425] | +0.074 [-0.276, +0.425] |

Relaxed-policy matched auxiliary effects range from **−0.167 to +0.334 pp**. Of the 108 recorded accuracy intervals, 105 include zero directly. The three copies of numerical-25 λ=0.1 versus zero under cap 0.5 have a stored lower endpoint of **+1.11×10⁻¹⁶ in accuracy fraction**, which is numerical zero at meaningful reporting precision; their reported interval is **[0.000, +0.314] pp**. An independent reconstruction using paired integer correct-count differences on the same 2,000 draws gives an exact lower bound of zero: 24 draws are negative, 28 are exactly zero, and 1,948 are positive. This floating-point boundary is not evidence of a robust positive effect. The raw interval values are preserved in the sealed report. All 12 fixed-zero auxiliary accuracy intervals include zero.

A **[0, 0]** interval in the 79-column rows occurs because both transferred policies are the identical never-intervene fallback. Conversely, the largest observed relaxed auxiliary difference, +0.334 pp for 79 columns + embeddings at λ=1 / cap 0.5, comes from its fallback avoiding the λ=0 router's harmful interventions. It does not show that the auxiliary router learned more successful rescues.

### Embedding contribution and validation-to-test transfer

All **72 relaxed embedding-versus-no-embedding point differences are nonpositive**, ranging from **−0.872 to 0.000 pp**. Twenty-one intervals are wholly negative, including repetitions across plot limits and harm caps; they are not 21 independent findings. All eight fixed-zero embedding comparisons are negative, from **−1.689 to −0.148 pp**. The four numerical-25 fixed-zero embedding intervals are wholly negative; the four 79-column intervals include zero. Full individual accuracy and weighted-F1 contrasts remain in the scored report.

The configuration discussed previously—**numerical 25 + embeddings, λ=0.1**—had a validation gain of **+0.274 pp** over its CE-only control. With the same frozen policies on test, that comparison becomes **−0.111 pp**, CI **[−0.317, +0.075] pp**, for all three harm caps. The auxiliary policy itself falls to **73.4781%**, below the matched fusion reference at 73.7379%, with **30 rescues and 44 harms**. Its fixed-zero version improves over its CE-only fixed-zero counterpart by +0.223 pp, CI [−0.114, +0.566] pp, but both remain below fusion. The earlier validation improvement therefore does not establish a generalizing TCP advantage.

The state-prediction diagnostics are consistent with the earlier overfitting result. Mean test NLL ranges are **0.6049–0.6110** for numerical 25, **0.7066–0.7646** for 79 columns, **1.8246–1.8541** for numerical 25 + embeddings, and **1.8912–1.9884** for 79 columns + embeddings. The embedding models had training NLL of only 0.0024–0.0094. Their supervised fusion-TCP test MSE is also larger: 0.1519–0.1705 with embeddings, versus 0.1244–0.1341 without embeddings. These observations support the generalization concern but do not separate sample-size, model-capacity and producer-representation-shift effects. TCP estimates remain auxiliary diagnostics and are not used to choose experts in this application.

### Uncertainty and interpretation

Accuracy and weighted F1 average the four seeds; pooled intervention counts refer to **5,388 seed–image evaluations of 1,347 images**, not 5,388 independent images. Affected-plot counts are distinct plots across seeds. The 2,000 paired bootstrap replicates resample the 531 whole plots with all their images and all four seed realizations together, using seed 20260826. Models and transferred thresholds remain fixed. Each reported CI belongs to its own matched comparison; intervals are not adjusted for the 360 comparisons.

This application avoids threshold selection on the new test outcomes, but the cleaned test set has been examined in earlier studies and influenced the wider research process. The results remain exploratory evidence. They support retaining the modest numerical-input router as a useful comparison and investigating the embedding generalization problem; they do not establish auxiliary TCP supervision as an advance or nominate a test-selected deployment policy.

The complete scored report contains every policy, per-seed and pooled metrics,
state/TCP diagnostics, test guardrail observations and all 360 paired contrasts.

## Verification and preservation

- **156 focused tests passed** before execution: extraction and frozen producer
  identities, label exclusion, native inference, exact replay, scoring,
  serialization, registry sequencing and immutable reuse. Regression checks
  detect a one-ULP change to saved probabilities or logits.
- Before test extraction, the new application reproduced all four saved
  development-validation feature matrices, all 64 native router outputs and
  every validation intervention mask exactly. No preprocessing was refitted.
- The complete production runner and its validator passed for all four seeds,
  64 models, 161 policies and 360 contrasts.
- The [independent artifact audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1_review/independent_artifact_audit.json)
  reconstructed four numerical matrices, all 64 native model predictions,
  all 64 float64 diagnostic records and 644 seed-policy masks exactly. It
  authenticated 371 source files and 1,347 raw images without importing study
  modules. All 540 development-study files retained their hashes, sizes and
  modification times.
- The [independent metric audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1_review/independent_audit.json)
  passed **49,291 assertions**, covering all policies, 144 frozen thresholds,
  80 diagnostic frames, 2,000 bootstrap draws and all 360 paired contrasts.
  Its metric and confusion-matrix reconstruction imports no production scoring
  helpers; maximum numerical discrepancy is **4.44×10⁻¹⁶**.
- The [interval-boundary audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1_review/epsilon_boundary_audit.json)
  independently confirms the exact zero lower endpoint discussed above. The
  sealed report's original floating-point values remain unchanged.
- Native float32 predictions remain authoritative. Diagnostic float64 replay
  differs by at most **2.04×10⁻⁵** in state logits, **7.59×10⁻⁶** in TCP logits,
  **1.74×10⁻⁶** in state probabilities and **1.11×10⁻⁶** in estimated TCP.
  It changes no state argmax, three auxiliary TCP orderings, and
  [no intervention across all 640 non-reference seed-policy applications](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1_review/independent_float64_policy_differences.json).
- The seventh append-only test event added exactly three records and preserved
  all 19 earlier registry records. The last prediction seal at
  **12:34:01.435755 UTC** preceded event reservation at **12:35:22.222118 UTC**,
  which preceded canonical label acquisition at **12:35:24.248812 UTC**.
  The original M4 decision and all earlier study artifacts remain unchanged.
- The [complete-run immutable reuse check](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1_review/immutable_reuse_audit.json)
  passed with exit code zero in 236.62 seconds. All **45 application files**,
  **540 parent-study files** and **22 registry files** retained identical SHA256,
  sizes and modification times. The repeated run reused every expert feature
  cache and router prediction; no fitting or additional registry event occurred.

## Artifacts

- [Frozen application declaration, source hashes and copied validation policies](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1/declaration.json).
- [Complete human-readable results](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1/scores/report.md)
  and [machine-readable results with all per-seed diagnostics and contrasts](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1/scores/report.json).
- [Scored predictions](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1/scores/scored_rows.parquet),
  [all action masks](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1/scores/actions.parquet),
  and [bootstrap draws](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1/scores/bootstrap_draws.npz).
- [All-configuration accuracy figure](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1_review/all_router_accuracies.png)
  and [individual matched auxiliary-TCP interval figure](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_test_v1_review/auxiliary_tcp_matched_intervals.png).
  PDF versions and their plotting script are saved beside the figures.

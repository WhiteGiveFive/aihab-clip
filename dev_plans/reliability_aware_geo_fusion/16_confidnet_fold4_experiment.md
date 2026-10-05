# Fixed-expert fold-4 ConfidNet and router experiment

Status: COMPLETE — all fits, selections, independent model/metric/provenance
audits, reconstruction, and full-workflow immutable reuse passed.

The user approved the proposed specification: train confidence heads and
matched routers on development-train fold 4, then evaluate using the same
experts on development-validation and apply the existing relaxed threshold rule.
Study ID: `confidnet_fold4_v1`.

## Result: no TCP benefit with fixed expert representations

The TCP-augmented router has **no feasible threshold in any of the nine
guardrail settings**, including the most relaxed harm cap and no plot minimum.
It therefore retains fusion, with **71.3808%** mean validation accuracy. The
matched numerical router reaches **72.0499%** under harm caps 0.7 and 0.9,
using the common threshold `0.1237670844365451`.

The matched difference is **−0.6691 percentage points**, with conditional 95%
paired whole-plot interval **[−1.6299, +0.3041] pp**. The augmented result here
is exactly the never-intervene fallback; it is not an active TCP correction
policy that independently matches fusion.

| Policy | Accuracy (%) | Mean weighted F1 (%) | Actions | Rescues | Harms | Wrong-to-wrong | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Same-expert fusion | 71.3808 | 70.9424 | 0 | 0 | 0 | 0 | 0 |
| Native MCP comparator | 71.2895 | 70.8362 | 318 | 118 | 121 | 79 | −3 |
| Estimated TCP comparator | 69.8601 | 69.0750 | 411 | 130 | 180 | 101 | −50 |
| Numerical RF, fixed zero | 71.3200 | 70.7409 | 254 | 101 | 103 | 50 | −2 |
| TCP-augmented RF, fixed zero | 69.9513 | 69.2436 | 351 | 105 | 152 | 94 | −47 |
| Numerical RF, relaxed 0.7/0.9 | 72.0499 | 71.5709 | 120 | 61 | 39 | 20 | +22 |
| TCP-augmented RF, every guardrail | 71.3808 | 70.9424 | 0 | 0 | 0 | 0 | 0 |

Metrics are arithmetic means across four seeds, each on the same 822 images.
Action/outcome counts pool 3,288 seed-image observations, not independent images.
Weighted F1 is calculated within each seed and then averaged. Relaxing the plot
minimum between 20, 10, and 0 changes no policy. Under the 0.5 harm cap, both
router arms fall back to fusion.

### Why the relaxed selector cannot recover the augmented router

The selector enumerates 621 candidates for numeric25 and 649 for augmented28,
including the typed never-intervene policy. At each plot minimum, numeric25 has
0/54/191 feasible candidates under harm caps 0.5/0.7/0.9; augmented28 has none.

Inspecting the saved curve explains this failure: **every augmented candidate
with at least 1% coverage has negative net corrections**, even before applying
a harm cap. The least negative net at that coverage is −3. Its six candidates
with positive net act on at most 19 pooled rows, just 0.578% coverage; the
largest net is only +2. Reducing plot support or relaxing the harm cap therefore
cannot rescue the policy while retaining the established 1% coverage rule.
This is a description of the saved candidate curve, not a new selected policy.

The numerical relaxed policy intervenes on 120 rows (3.6496% coverage), affecting
55 distinct plots. It yields positive net corrections in each seed:

| Seed | Fusion accuracy (%) | Numerical fixed-zero (%) | Augmented fixed-zero (%) | Numerical relaxed (%) | Augmented relaxed (%) | Numerical relaxed net |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 72.2628 | 72.0195 | 71.2895 | 72.3844 | 72.2628 | +1 |
| 2 | 70.1946 | 69.8297 | 67.7616 | 71.5328 | 70.1946 | +11 |
| 3 | 69.9513 | 70.3163 | 68.9781 | 71.0462 | 69.9513 | +9 |
| 4 | 73.1144 | 73.1144 | 71.7762 | 73.2360 | 73.1144 | +1 |

The fixed-zero augmented router underperforms its matched baseline in every
seed. Its mean difference is −1.3686 pp, interval [−2.9108, +0.1816] pp.
Threshold selection avoids its harmful interventions by choosing no action.

### Matched experts and correction ceiling

All references below come from the same three-fold-trained checkpoints used
for this experiment's confidence and router inputs.

| Expert/reference | Mean validation accuracy (%) |
|---|---:|
| Image | 68.5523 |
| Geo | 61.2530 |
| Fusion | 71.3808 |
| Geo/fusion oracle | 80.9307 |

The oracle retains fusion except where it is wrong and geo is correct. It has
314 available rescues across four seeds, or 78.5 per seed, giving an oracle
accuracy gain of 9.5499 pp. It uses ground truth and is not an inference policy.
The numerical relaxed router realizes 22 net corrections; the augmented
selected policy realizes none. Comparison with the prior experiment must use
these matched references because its expert checkpoints and opportunities differ.

### Confidence generalization remains poor

All twelve heads completed 500 finite epochs with no zero-gradient epochs.
They fitted their fold-4 TCP targets closely, but their validation estimates
did not preserve the strong training failure ranking.

| Expert | Train TCP MSE | Validation TCP MSE | Train failure AP | Validation estimated-TCP AP | Validation MCP AP | Validation entropy AP |
|---|---:|---:|---:|---:|---:|---:|
| Image | 0.002310 | 0.229885 | 0.996720 | 0.469796 | 0.608806 | 0.627863 |
| Geo | 0.000294 | 0.163089 | 0.997295 | 0.539399 | 0.580463 | 0.583648 |
| Fusion | 0.000569 | 0.189631 | 0.998892 | 0.450301 | 0.548628 | 0.563963 |

Validation estimated-TCP failure ROC AUC is 0.705841/0.688794/0.723297 for
image/geo/fusion, versus MCP's 0.781572/0.718573/0.775157. Both AP and AUC are
lower for all three expert modes on average across seeds.

A predictor fixed to each expert's training-mean TCP has validation MSE
0.180947/0.152866/0.169180, beating all three learned confidence heads. Geo's
learned TCP MSE is slightly better than MCP's 0.166746; this regression change
does not translate to improved failure detection or routing. Validation
predictions within `1e-6` of one average 16.39%/0.12%/9.22% across the three
modes; the heads did not collapse entirely to a constant prediction.

On its training rows, the augmented router captures all 332 available rescues
across seeds, with only one harm. On validation at zero threshold it captures
105 rescues but makes 152 harms. The confidence estimates used for router
training are in-sample for ConfidNet, as specified in the approved design.

| Validation router diagnostic | Numerical RF | Augmented RF |
|---|---:|---:|
| Four-state NLL | 0.622145 | 9.157415 |
| Four-state Brier score | 0.374397 | 0.545304 |
| Rescue AP on disagreements, `q_rescue` | 0.395683 | 0.307301 |
| Rescue AP on disagreements, `q_rescue-q_harm` | 0.401201 | 0.304060 |

The augmented NLL includes 37 pooled validation rows with zero probability for
the true four-state target; the inherited metric floors these probabilities
at float64's smallest positive normal value, `2.2250738585072014e-308`.
Its magnitude depends strongly on that convention. Brier and rescue ranking
also deteriorate, so the routing result is not solely an NLL-floor effect.

### Interpretation and uncertainty

Keeping the producing experts fixed **did not make estimated TCP useful in
this configuration**. The experiment therefore shows that changing expert
representations is not required for this failure pattern to occur. It does not
establish how much that mechanism contributed to the earlier pooled-OOF result.

The large train-to-validation gap is consistent with confidence overfitting,
and same-row confidence/router training remains a plausible contributor.
Compared with the previous study, this design also uses fewer training examples,
different expert checkpoints, and fewer optimizer updates at the same epoch
budget. These effects are not separately identified. This is not a general
claim that TCP or learned confidence cannot help routing.

The numerical relaxed gain over fusion is +0.6691 pp, conditional interval
[−0.3041, +1.6299] pp. The matched augmented-minus-numerical weighted-F1
difference is −0.6286 points, interval [−1.6135, +0.3869]. Threshold selection
and reporting use the same previously inspected development-validation data;
the intervals hold thresholds fixed and do not adjust for selection or reuse.
No independent confirmation or cleaned-test result is claimed.

## Fixed setup

- Reuse the frozen expert producer with zero-based fold ID 3 for each seed
  1–4. Its image, geo, and fusion experts were trained on folds 1–3
  (2,526 images, 975 plots). No expert is retrained.
- Confidence/router training uses fold 4 only: 852 images, 325 plots.
  Development-validation has 822 images, 325 disjoint plots. The same frozen
  encoder, expert heads, and geo scaler produce both splits' representations.
- Fit three confidence heads per seed, using each expert's 256-dimensional
  penultimate ReLU and native-T=1 TCP supervision on fold 4. Architecture is
  256→400→400→400→400→1, hidden ReLU, sigmoid output, MSE, Adam 1e-4,
  500 epochs, batch 128, no dropout/weight decay/extra normalization, final epoch.
  Keep native float32 CPU one-thread fitting and exact serialized native replay;
  report float64 differences separately. Preserve failed states/history and
  pause on nonfinite fitting, failed native replay, or five zero-gradient epochs.
- Use the existing calibration and numerical-feature recipe on fold 4 only:
  fit three scalar temperatures and one feature transform per seed on those
  rows, then apply unchanged to validation. Native TCP targets remain separate
  from this router-input calibration. The old pooled-OOF preprocessing would
  introduce labels from other folds, so it is not reused.
- Fit two RF routers per seed: the existing 25 numerical summaries versus the
  same summaries plus three estimated TCPs. RF: 500 trees, depth 4, minimum
  leaf 5, max_features 0.5, bootstrap, log_loss, no class weighting, one thread.
  Targets are rescue/harm/both-correct/both-wrong. ConfidNet predictions for
  router training are from its own training rows, as explicitly approved.
- Twelve confidence fits and eight RF fits, without architecture/model search.
  Seal and replay all predictions before validation labels enter scoring.
- Report fusion/MCP/direct-TCP and both RF fixed-zero policies, plus all 18
  threshold-selected policies: plot minima 20/10/0 × harm caps 0.5/0.7/0.9
  for both arms. The established selector uses pooled coverage ≥1%, positive
  net, exact integer harm constraints, nonnegative candidates, strict
  `q_rescue-q_harm > threshold` on disagreement, and one common threshold across
  seeds. Ties prefer fewer harms, fewer actions, then higher threshold.
- Compute new expert accuracies and the matched geo/fusion correction ceiling.
  Retain all per-seed outcomes and confidence regression/failure-detection metrics.
  Compute intervals using 2,000 paired whole-plot bootstrap resamples, seed
  20260826. These intervals
  hold selected thresholds fixed; selection and reporting share development
  validation. No cleaned-test access or original M4 decision change.

This controls the producing expert representation across confidence training
and validation. It leaves same-row confidence/router overfitting possible and
changes both expert training size and confidence/router training sample size.
It therefore does not isolate a single cause relative to the pooled-OOF study.

## Execution record

All 166 focused extraction, preprocessing, fitting-replay, selector, reporting,
and workflow checks passed before production. The declaration was sealed at
`2026-09-24T14:32:20.422422+00:00`, binding 120 source/configuration/protocol and
expert-parent files. Its SHA256 is
`3965d6b7c09932d97e104529cb1f4701162a4199a504c8896eaf85d335c8cced`.
Scientific sources/configuration are frozen. Inference uses the available RTX
4090; confidence and RF fitting retain the approved CPU recipe.

All twelve heads and eight routers were verified before validation labels were
joined at `2026-09-24T15:11:19.543288+00:00`. The scientific configuration and
source hashes remained unchanged throughout fitting and scoring.

Independent audits passed:

- All four seeds' ten fold-4 training arrays match the earlier producer cache
  bit-for-bit. Checkpoints/scalers and the three disjoint plot memberships
  match the frozen protocol; validation arrays have no labels.
- Twelve temperature states and four feature transforms record the exact
  852-row fold-4 fitting scope and identities.
- All twelve heads reproduce native float32 predictions exactly on both splits.
  Independent float64 replay has maximum absolute difference `1.44556396e-6`,
  with zero training and 11 validation rows exceeding the historical `1e-6`
  diagnostic boundary. This is not a rejection criterion under the approval.
- Independent traversal of all eight forests differs by at most `1.11e-16`;
  predicted classes, zero-threshold actions, and all 72 selected-policy/seed
  masks agree exactly.
- Independent metric, selector, and whole-plot-bootstrap calculations passed
  107,234 checks, maximum difference `3.33e-16`, including all 54 paired
  accuracy/F1 contrasts with 2,000 resamples.
- Full reconstruction and repeat-build reuse passed. All 130 scientific files
  (272,697,101 bytes) retained their exact bytes, sizes, and modification times.
  All 120 declared parent hashes and snapshots of the 199 files in the two
  preceding ConfidNet bundles were reverified. Reuse completed at
  `2026-09-24T15:15:50.674780+00:00`.

## Reproducibility

From the repository root, the complete workflow is:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_multimodal_geo_helpfulness_confidnet_fold4.py build
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_multimodal_geo_helpfulness_confidnet_fold4.py validate
```

The GPU is used only for frozen expert inference; training retains CPU float32
and one thread. Individual phases are `declare`, `extract-seed --seed N`,
`fit-seed --seed N`, `fit-routers`, `score`, and `validate`. All three heads are
saved separately for each seed. Existing complete stages validate and reuse
their files; the runner does not overwrite previous results.

Outputs reside under
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1`:

- `declaration.json`: approved settings, source hashes, expert parents, and
  snapshots of both previous ConfidNet result bundles.
- `features/seed_N/`: fold-4 and validation arrays, fixed-expert provenance.
- `confidence/seed_N/{image,geo,raw}/`: weights, epoch histories, native TCP
  estimates, training diagnostics, and float32/float64 replay checks.
- `router_inputs/seed_N/`: fold-4-fitted temperatures/transforms and replayable
  25-column inputs for both splits.
- `routers/seed_N/{baseline,augmented}/`: fitted forest states and predictions.
- `scores/`: fixed and relaxed policy reports, full threshold candidate curves,
  confidence metrics, matched expert/oracle references, and scored/action rows.

Reviewable outputs:

- [Machine-readable results](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1/scores/report.json)
  and [generated tables](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1/scores/report.md).
- [Diagnostic figure, PNG](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1_review/fold4_diagnostics.png)
  and [PDF](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1_review/fold4_diagnostics.pdf).
- [Independent metric audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1_review/metrics_audit.json),
  [model audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1_review/all_models.json),
  and [provenance audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1_review/provenance_summary.json).
- [Immutable reuse receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1_review/reuse_receipt.json)
  and [review manifest](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_v1_review/manifest.json).

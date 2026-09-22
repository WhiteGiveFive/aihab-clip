# Relaxed-guardrail sensitivity study — frozen random-forest routers

Status: COMPLETE — all 27 development RF policies and both-family cleaned-test reports are sealed; independent reconstruction, metric/bootstrap and lineage audits, and byte/mtime-preserving reuse pass. Original M4 no-go remains unchanged. Native-runtime interruptions are recorded below.

Study ID: `relaxed_guardrails_rf_v1`. The user explicitly requested extending the [completed LR sensitivity study](09_relaxed_guardrails_lr.md) to RF. This is not a learner chosen from cleaned-test ranking, a new model-capacity search, a replacement M4 gate, or independent confirmation.

## Fixed scope

- Reuse all twelve already selected RF states: full727, no-pairs79, and numeric25 features, each with seeds 1–4. Inherited configurations are `rf__<variant>__d4_leaf5`; no model/configuration reselection, refitting, recalibration, or feature changes are allowed.
- Apply the same nine declared guardrail settings to each variant: minimum distinct acted plots `[20, 10, 0]` crossed with harm numerators `[5, 7, 9]` over denominator 10. Zero means no plot minimum. There are 27 new policies, not 27 fitted models. Enforce `10*H <= numerator*R` with exact integer arithmetic.
- Keep pooled effective coverage at least 1%, strictly positive net corrections, nonnegative threshold candidates, the score `q_rescue-q_harm`, raw/geo disagreement, and strict `score > threshold`. Maximize mean four-seed net corrections; tie-break by fewer harms, fewer actions, then higher threshold. No feasible threshold means typed `never_intervene`.
- Select thresholds only from the sealed RF development-validation action/score table in `router_capacity_v1`: 822 images, 325 plots, and four seed realizations (3,288 rows per RF variant). Retain original selected RF configurations and reproduce each original 20-plot/0.5R curve and policy exactly before applying relaxation.
- Publish a declaration binding code, configuration, environment, all relevant parents, prior LR history, and the existing registry before new selection. Seal all 27 policies before decoding test expert/router inputs. Validate complete policy metadata before any test-feature access.
- Reuse the exact expert and router tables from `exploratory_legacy_capacity_v1`, including both prior checkpoint families: `best_model` fixed epoch-50 experts and `oracle_test_best_model` test-selected, biased diagnostics. Each family contains the same 1,347 cleaned-test images in 531 plots and all four seeds. The expert and 12-router probability tables are copied byte-for-byte; only RF action thresholds change.
- Retain all seventeen prior LR/RF/XGB/MLP/heuristic policies unchanged as reference outputs. Add 27 RF policies, giving 44 action policies per family. No other learner is reselected or retuned in this study.

## Access, publication, and evidence boundaries

The runner uses separate processes for development selection, label-blind test application, and scoring. Test phases reject fitting libraries and threshold-selector imports. All test actions are sealed before a new append-only evaluation reservation is recorded and the authenticated prior label copy is decoded. Original test-cache label columns and historical checkpoint metrics are not reopened.

Parent authentication may hash prior labels/results as bytes; it does not decode them for selection. The prior LR study is bound as historical exposure, not used to select RF hyperparameters, thresholds, or a test-winning policy. Original M1–M4 and all capacity/MLP/LR results remain immutable. The registry gains exactly three records for one new completed study; the ten existing records are preserved.

Publication uses exclusive locks, owned recovery, no-overwrite writes, and manifests written last. Unknown files are rejected and never removed. A valid repeat returns `reused_valid` without changing bytes or modification times. Output must be disjoint from all source bundles, the frozen protocol, and the global registry.

The cleaned test and development validation have already been inspected. This is adaptive exploratory sensitivity, not independent confirmation. No test result selects a guardrail, feature set, model, or checkpoint family. Preserve original M4's no-go and M5/M6 deferral.

## Fixed reporting

Report habitat top-1/top-3 performance with and without RF correction, actions/rescues/harms/net, coverage, actual distinct plots, per-seed performance, and subgroup outcomes. Four-state RF prediction performance remains unchanged because the models and their probabilities are unchanged.

Use the exact predecessor's 2,000 shared habitat-stratified whole-plot bootstrap draws, seed 20260826, keeping all seed realizations of a sampled plot together. Report gain-over-raw intervals and 27 paired new-policy-minus-matching-original-RF contrasts per family. Preserve prior metrics and comparison payloads as clearly named reference data; do not count them as new comparisons. Intervals condition on frozen experts and selected policies and do not correct for adaptive reuse, model/threshold selection, multiple comparisons, or test-selected checkpoints.

Report largest-positive-plot concentration and pooled/per-seed net after removing that plot descriptively. Do not enforce development guardrails, select policies, or rerun a gate on test outcomes.

## Commands and artifacts

From the repository root:

```bash
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_rf.py build
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_rf.py validate
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_rf.py build
```

`build` runs `select`, `predict`, and `score` in isolated CPU-only processes. `validate` independently reconstructs the three stages; repeat builds reuse committed outputs. Individual phases and their `validate-` counterparts are also available. Path overrides: `--legacy-root`, `--capacity-root`, `--lr-history-root`, `--output-dir`, and `--config`. No scientific-setting or force-overwrite options are exposed.

Default output: `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/relaxed_guardrails_rf_v1`.

| Artifact | Contents |
| --- | --- |
| `declaration.json` | Pre-selection identity, fixed grid, model/source/history lineage, code/environment and evidence boundaries |
| `selection/selection.json` | All 27 RF development curves/selections, original baseline parity, configuration/model identities and seed/pooled counts |
| `selection/frozen_policies.json` | The 27 selected policy/guardrail/configuration/typed-threshold records for apply-only stages |
| `selection/{report.md,manifest.json}` | Development comparison and immutable seal |
| `predictions/experts.parquet` | Byte-identical 10,776-row prior expert table across two families |
| `predictions/routers.parquet` | Byte-identical 129,312-row prior twelve-router table; apply-only reconstruction verifies values without model fitting or probability-state changes |
| `predictions/actions.parquet` | 474,144 rows: 44 policies × two families × four seeds × 1,347 images; 183,192 original and 290,952 new action rows |
| `predictions/{metadata,manifest}.json` | Row/identity contract, selection linkage, full label-blind seal |
| `scores/<family>/` | Eleven report artifacts: expert/classification/router/policy metrics, bootstrap summaries/replicates, subgroups, outcomes, targets, summary, paired comparisons |
| `scores/{labels.parquet,build_declaration.json,reservation.json,source_receipt.json,receipt.json,manifest.json}` | Authenticated label copy, ownership, new global event and integrity records |
| `scores/{summary.json,report.md}` | Both-family results with and without intervention |

There are forty committed scientific files in total. Float64 reconstruction uses `atol=rtol=1e-12`; probability normalization uses `atol=1e-8`. JSON is strict and Parquet preserves explicit schemas.

## Development selection

The declaration was published at `2026-09-13T21:30:53.165431+00:00`, before this study's new threshold selection. All 27 approved policies are sealed. For each RF feature variant and harm cap, changing minimum plot support between 20, 10, and none produces the same selected threshold. Unlike LR, full727 RF selects different thresholds under 0.7R and 0.9R; the two RF ablations select the same threshold under those two caps.

| RF features | Harm cap | Threshold | Actions / rescues / harms | Distinct acted plots | Net | Gain (pp) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Full727 | 0.5R | 0.19775509756132326 | 69 / 42 / 19 | 35 | 23 | +0.699513 |
| Full727 | 0.7R | 0.09315034499525951 | 138 / 72 / 43 | 61 | 29 | +0.881995 |
| Full727 | 0.9R | 0.013295710490183299 | 220 / 104 / 74 | 98 | 30 | +0.912409 |
| No-pairs79 | 0.5R | 0.18799358669571456 | 76 / 43 / 21 | 38 | 22 | +0.669100 |
| No-pairs79 | 0.7R or 0.9R | 0.1020226697868136 | 134 / 70 / 41 | 61 | 29 | +0.881995 |
| Numeric25 | 0.5R | 0.1817292058098988 | 76 / 44 / 20 | 38 | 24 | +0.729927 |
| Numeric25 | 0.7R or 0.9R | 0.002916998232049863 | 233 / 110 / 77 | 102 | 33 | +1.003650 |

Counts pool four seeds over 822 images (3,288 seed-image rows). The numeric 0.7R result lies exactly on its allowed harm boundary: 77 = 0.7 × 110. All original 20-plot/0.5R RF policies reproduce exactly. No test outcome was used in these choices. Complete curves and all 27 rows are in the sealed `selection/` artifacts.

## Cleaned-test results

Each family uses the same 1,347 images in 531 plots, with four seed realizations (5,388 rows). Accuracies are arithmetic means over seeds, equivalently pooled accuracy because seed sizes are equal. Action/rescue/harm/net counts pool seed realizations, not independent images. Distinct plot counts use the union across seeds. The 27 policies reduce to seven distinct variant/threshold combinations: plot minima 20, 10, and none give identical results; the two ablations also have identical 0.7R and 0.9R thresholds. All 27 remain individually recorded.

### Unchanged experts

| Expert | Fixed-final top-1 (%) | Fixed-final top-3 (%) | Test-selected top-1 (%) | Test-selected top-3 (%) |
| --- | ---: | ---: | ---: | ---: |
| Image | 73.144024 | 92.409057 | 73.626578 | 93.578322 |
| Geo | 61.618411 | 89.365256 | 64.123979 | 90.163326 |
| Raw concat | 74.313289 | 93.838159 | 75.334076 | 94.988864 |

### Fixed-final experts (`best_model`, fixed epoch 50)

| RF features | Development harm cap | Top-1 (%) | Top-3 (%) | Gain over raw (pp) | Actions / rescues / harms | Net | Coverage (%) | Plots |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Full727 | 0.5R (original) | 74.665924 | 93.782480 | +0.352635 | 100 / 48 / 29 | 19 | 1.855976 | 41 |
| Full727 | 0.7R | 75.167038 | 93.541203 | +0.853749 | 264 / 129 / 83 | 46 | 4.899777 | 76 |
| Full727 | 0.9R | 75.074239 | 93.374165 | +0.760950 | 366 / 166 / 125 | 41 | 6.792873 | 97 |
| No-pairs79 | 0.5R (original) | 74.647365 | 93.763920 | +0.334076 | 121 / 56 / 38 | 18 | 2.245731 | 46 |
| No-pairs79 | 0.7R or 0.9R | 75.167038 | 93.596882 | +0.853749 | 245 / 123 / 77 | 46 | 4.547142 | 70 |
| Numeric25 | 0.5R (original) | 74.665924 | 93.763920 | +0.352635 | 118 / 55 / 36 | 19 | 2.190052 | 45 |
| Numeric25 | 0.7R or 0.9R | 75.129918 | 93.281366 | +0.816630 | 390 / 174 / 130 | 44 | 7.238307 | 104 |

Harm relaxation increases the fixed-final top-1 point gains for all three variants. Full727's 0.9R policy makes 102 more actions than its 0.7R policy, adding 37 rescues but 42 harms: five fewer net corrections. This is a descriptive comparison, not a test-based choice of 0.7R. All routed top-3 point estimates are lower than raw fusion's 93.838159%; top-1 benefit does not imply top-3 benefit.

### Test-selected experts (`oracle_test_best_model`, biased diagnostic)

These checkpoints were selected with historical test performance. Their results are not independent confirmation and are not directly interchangeable with the fixed-final family.

| RF features | Development harm cap | Top-1 (%) | Top-3 (%) | Gain over raw (pp) | Actions / rescues / harms | Net | Coverage (%) | Plots |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Full727 | 0.5R (original) | 75.501114 | 94.933185 | +0.167038 | 34 / 19 / 10 | 9 | 0.631032 | 18 |
| Full727 | 0.7R | 75.445434 | 94.821826 | +0.111359 | 104 / 47 / 41 | 6 | 1.930215 | 42 |
| Full727 | 0.9R | 75.779510 | 94.506310 | +0.445434 | 264 / 121 / 97 | 24 | 4.899777 | 89 |
| No-pairs79 | 0.5R (original) | 75.445434 | 94.933185 | +0.111359 | 41 / 21 / 15 | 6 | 0.760950 | 19 |
| No-pairs79 | 0.7R or 0.9R | 75.389755 | 94.840386 | +0.055679 | 92 / 41 / 38 | 3 | 1.707498 | 41 |
| Numeric25 | 0.5R (original) | 75.538233 | 94.933185 | +0.204157 | 38 / 22 / 11 | 11 | 0.705271 | 20 |
| Numeric25 | 0.7R or 0.9R | 75.798070 | 94.413512 | +0.463994 | 294 / 133 / 108 | 25 | 5.456570 | 97 |

Relaxation is not uniformly better here: full727/0.7R and no-pairs79/0.7R lose three net corrections relative to their original RF policies, whereas full727/0.9R and numeric25 gain 15 and 14. No result is used to select a checkpoint family or guardrail.

The harm and support constraints apply at development selection only; they are not guaranteed under transfer and are not re-enforced on test labels. For example, fixed-final numeric25/0.7R has 130/174 = 0.747 harms per rescue, exceeding 0.7. We report this outcome rather than refitting its threshold.

### Conditional uncertainty

All intervals below are 95% conditional whole-plot-bootstrap percentile intervals, in percentage points. The paired contrast subtracts the matching original RF variant (20 plots, 0.5R), not the original LR primary. The 0.5R variants exactly reproduce that comparator and therefore have paired contrast and interval `[0, 0]`.

| Family | RF / cap | Gain over raw [95% interval] | Gain over original RF [95% interval] |
| --- | --- | --- | --- |
| Fixed-final | Full727 / 0.5 | +0.353 [-0.207, +0.984] | 0 [0, 0] |
| Fixed-final | Full727 / 0.7 | +0.854 [-0.331, +1.986] | +0.501 [-0.298, +1.266] |
| Fixed-final | Full727 / 0.9 | +0.761 [-0.679, +2.156] | +0.408 [-0.711, +1.471] |
| Fixed-final | No-pairs79 / 0.5 | +0.334 [-0.328, +1.019] | 0 [0, 0] |
| Fixed-final | No-pairs79 / 0.7 or 0.9 | +0.854 [-0.279, +1.947] | +0.520 [-0.093, +1.140] |
| Fixed-final | Numeric25 / 0.5 | +0.353 [-0.297, +1.013] | 0 [0, 0] |
| Fixed-final | Numeric25 / 0.7 or 0.9 | +0.817 [-0.627, +2.211] | +0.464 [-0.577, +1.471] |
| Test-selected | Full727 / 0.5 | +0.167 [-0.127, +0.470] | 0 [0, 0] |
| Test-selected | Full727 / 0.7 | +0.111 [-0.557, +0.807] | -0.056 [-0.498, +0.410] |
| Test-selected | Full727 / 0.9 | +0.445 [-0.703, +1.566] | +0.278 [-0.703, +1.233] |
| Test-selected | No-pairs79 / 0.5 | +0.111 [-0.276, +0.483] | 0 [0, 0] |
| Test-selected | No-pairs79 / 0.7 or 0.9 | +0.056 [-0.491, +0.639] | -0.056 [-0.352, +0.246] |
| Test-selected | Numeric25 / 0.5 | +0.204 [-0.094, +0.526] | 0 [0, 0] |
| Test-selected | Numeric25 / 0.7 or 0.9 | +0.464 [-0.691, +1.618] | +0.260 [-0.754, +1.243] |

Every gain-over-raw interval spans zero; every non-identical paired interval spans zero. Larger fixed-final point gains therefore do not establish a reliable improvement. These intervals also omit adaptive holdout reuse and policy/checkpoint selection uncertainty.

### Seed and plot concentration

Seed vectors below are ordered 1–4. "Net without best plot" removes the entire plot contributing the largest positive pooled net, using the already frozen actions; it is a descriptive sensitivity, not a selection rule.

| Family | RF / cap | Seed net corrections | Largest positive plot / net | Net without best plot |
| --- | --- | --- | --- | ---: |
| Fixed-final | Full727 / 0.5 | [3, 12, -3, 7] | 744XX2 / 9 | 10 |
| Fixed-final | Full727 / 0.7 | [12, 13, 8, 13] | 1261XX2 / 10 | 36 |
| Fixed-final | Full727 / 0.9 | [10, 13, 3, 15] | 1261XX2 / 12 | 29 |
| Fixed-final | No-pairs79 / 0.5 | [2, 11, -4, 9] | 744XX2 / 9 | 9 |
| Fixed-final | No-pairs79 / 0.7 or 0.9 | [11, 13, 9, 13] | 744XX2 / 10 | 36 |
| Fixed-final | Numeric25 / 0.5 | [3, 12, -2, 6] | 744XX2 / 9 | 10 |
| Fixed-final | Numeric25 / 0.7 or 0.9 | [13, 14, 3, 14] | 1261XX2 / 11 | 33 |
| Test-selected | Full727 / 0.5 | [1, 6, -1, 3] | 743X2 / 3 | 6 |
| Test-selected | Full727 / 0.7 | [1, 0, 1, 4] | 744XX2 / 6 | 0 |
| Test-selected | Full727 / 0.9 | [9, 11, -1, 5] | 733XX3 / 8 | 16 |
| Test-selected | No-pairs79 / 0.5 | [-1, 5, -2, 4] | 1041X4 / 3 | 3 |
| Test-selected | No-pairs79 / 0.7 or 0.9 | [1, -3, 0, 5] | 744XX2 / 5 | -2 |
| Test-selected | Numeric25 / 0.5 | [1, 8, -1, 3] | 743X2 / 3 | 8 |
| Test-selected | Numeric25 / 0.7 or 0.9 | [10, 12, -4, 7] | 733XX3 / 8 | 17 |

At harm caps 0.7R and 0.9R, all fixed-final policies have positive net corrections in all four seeds. Full727/0.7R and no-pairs79/0.7R remain positive in all seeds after removal of the best positive plot; full727/0.9R and numeric25 become zero in seed 3. For these 0.7R/0.9R policies, the best plots account for 6.322–8.130% of the sum of positive plot nets (not of total net). The test-selected no-pairs gain reverses after removing its best plot. Complete per-seed accuracy, subgroup results, plot diagnostics, and all policy rows are saved in the family `classification.json`, `policy_metrics.json`, `subgroup_metrics.parquet`, and `summary.json` files.

The experiment shows sensitivity to intervention risk tolerance, not an improvement to RF four-state prediction: all twelve RF states and router probabilities are unchanged. It does not establish a new go decision or authorize another tuning round on test outcomes.

## Verification and runtime record

- All 265 new focused tests pass in 21.25 s; all 1,519 predecessor cases pass in 348.27 s. The predecessor collection excludes the known unrelated `test_multimodal_geo_10m.py` failure recorded previously. New tests cover all grid/feature/seed/family cases, full 474,144-row action construction, exact 17-policy and bootstrap preservation, original threshold parity, integer harm boundaries, metadata rejection before features, source containment, label/receipt ordering, import guards, model/parent/artifact tampering, lock contention, interrupted publication, fourth-study append-only registry behavior, and byte/mtime-preserving reuse.
- Metadata-only parent authentication passes, including the sealed RF model/configuration identities and all 1,025 newly bound parent hashes. No new relaxed result was inspected before declaration.
- The first production build sealed development selection in 9.47 s, then its prediction child exited with native code -11 (26.79 s overall). The fault-handler trace was inside pandas `sort_values`/`take_nd` while sorting the assembled action table. Only the five declaration/selection files existed afterward; test predictions, score labels, and a new registry reservation had not been published. All parent hashes and prior ten registry records remained unchanged.
- The second attempt reused selection and sealed all test predictions in 35.92 s, then its score child exited with native code -11 (60.15 s overall). This trace was in a pandas object-array string comparison during checkpoint-family filtering. The score build declaration, reservation, authenticated label copy and source receipt existed; no metric outputs had been written. The ten old registry records remained unchanged, with only the one new reservation added.
- The third unchanged attempt succeeded in 161.30 s, reusing selection and predictions and completing scoring in 137.08 s. It resumed only the owned, uncommitted score outputs. `PYTHONFAULTHANDLER=1` is diagnostic and does not alter any scientific setting. No source-code or artifact rewrite was made in response to the crashes. Their cause has not been isolated; they resemble native-runtime instability previously recorded in the M4 dataset and LR sensitivity records. Environment versions are NumPy 2.0.2, pandas 2.2.3, and PyArrow 21.0.0; no dependency was changed.
- Independent development audit reproduces all 27 policies and 15,984 threshold-candidate records, including exact baseline parity, integer feasibility tests, seed utility, and every tie-break.
- Independent label-blind prediction audit confirms byte-identical expert/twelve-router tables, exact preservation of 183,192 original action rows, and all 290,952 new actions across 216 policy/family/seed cases. Every strict threshold/disagreement mask and final prediction matches; maximum probability-sum error is below 1.56e-15.
- Independent final metadata audit confirms exactly forty artifacts, all 1,025 parent/model/code hashes, all receipt and source-copy hashes, ordered declaration/selection/prediction/label/score publication, ten preserved prior registry records plus exactly three new records, and unchanged original M4 no-go.
- The independent `validate` command succeeds in 156.82 s, reconstructing selection, predictions, and scores without fitting models. All forty output files preserve their bytes and modification times. A further run of the six new RF test modules passes all 265 cases in 20.99 s.
- Independent Arrow/NumPy/JSON scoring audit checks all 44 policies in each family, every seed's top-1/top-3 and action/rescue/harm/net/plot counts, plot-removal diagnostics, and all 27 paired contrasts. It regenerates all 2,000 shared bootstrap draws and every policy's replicate counts/deltas. All seventeen predecessor metric payloads, nested summaries/contrasts, subgroup/outcome rows, and 34,000 prior bootstrap rows per family are preserved exactly.
- The repeat `build` succeeds with `reused_valid` in 162.92 s (all three stages reused). All forty scientific files retain the exact post-build bytes and modification times through independent validation and reuse. All seven predecessor bundles (760 files total) and the ten old registry records preserve their pre-study bytes and modification times; the three new registry records also remain unchanged through validation/reuse. The registry now contains four completed exploratory studies, not a new independent test cohort.
- Together, the 265 new cases and 1,519 predecessor cases give 1,784 passing focused regression cases. Python compilation and whitespace checks pass. No frozen scientific code, configuration, model state, or parent artifact was changed after declaration.

Payload identities (distinct from physical-file hashes) are:

| Record | SHA-256 |
| --- | --- |
| Declaration | `1b7688e000da32355aa8f45178b96ba2330f69ca5db50684175f26e825099bf2` |
| Selection manifest | `2726684126d66e7ff0bb82fd5dde6dccd8def61d1b0607d80fde99e630a6abbf` |
| Prediction manifest | `f2edab625940400cb60568f2e9c15cc209f87bad363ed588a1e6301632e3eba1` |
| Score manifest | `dc103e5f6a085ba110c6462b4b0d0748a0314f93bfe78ef1dad8bdb6dd2fcf1e` |
| Evaluation receipt | `d87c01deec1bf6874f6b64de40ae483ca04b0e3552db4381945d5018e9d4b4bd` |
| Evaluation event ID | `60895f1b21aab6252c37ee0f2c3ae6bd01591049cad5557c2cccbbf8f3030440` |

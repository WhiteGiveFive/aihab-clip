# Relaxed-guardrail sensitivity study — frozen logistic routers

Status: COMPLETE — all 27 development-selected policies and both cleaned-test checkpoint-family reports are sealed; independent reconstruction, audits, and immutable reuse pass. The native-runtime caveat below remains unresolved.

Study ID: `relaxed_guardrails_lr_v1`. The user explicitly approved minimum acted-plot support of 20, 10, or none, with harm caps of 0.5, 0.7, and 0.9 rescues. Logistic regression is the user's compute-limited starting model, not a learner selected for having the best cleaned-test performance. All three frozen feature variants are retained. This is a new sensitivity study, not a revision of the original M4 protocol or its no-go conclusion.

## Fixed scope

- Reuse the twelve M4-selected logistic states: four seeds for each of full727, no-pairs79, and numeric25. Their selected regularization values (0.1, 0.1, 10), expert temperatures, input transforms, feature sets, and model weights remain unchanged. There is no model fitting or model reselection.
- Cross minimum unique acted plots `[20, 10, 0]` with harm numerators `[5, 7, 9]` over denominator 10. Zero means no plot minimum. These are nine guardrail settings per feature variant and 27 new policies, not 27 newly trained models.
- Keep effective coverage at least 1% of all pooled seed-image rows and strictly positive pooled net corrections. Enforce harm limits by exact integer arithmetic `10 * H <= numerator * R`, avoiding float-boundary ambiguity.
- Keep the original score `q_rescue - q_harm`, nonnegative threshold candidate construction, raw/geo disagreement requirement, and strict `score > threshold`. Maximize mean four-seed rescues-minus-harms; ties prefer fewer harms, fewer actions, then higher threshold. No feasible candidate selects typed `never_intervene`.
- Select thresholds only from the sealed M4 development-validation predictions/targets: 822 images from 325 plots in each of four seed realizations. Plot support counts distinct plots across the union of acted rows, not seed-plot duplicates. Original 20-plot/0.5-cap curves, selected thresholds, and actions must reproduce exactly.
- Before any new selection, publish a declaration binding this grid, code, environment, M4/legacy parent hashes, and cohort. Before test decoding, seal all 27 selected threshold policies.
- Reuse the exact prior legacy expert and LR probability tables for both `best_model` (fixed epoch50) and `oracle_test_best_model` (test-selected, biased diagnostic). The cohort remains 1,347 images in 531 plots, four seeds, 5,388 seed-image rows per family. No expert or router fitting, calibration update, checkpoint choice, or threshold selection uses these test rows.
- Retain all eight original LR/heuristic policies as unchanged comparators. There are 35 action policies per test family. Do not select a winning guardrail setting or feature variant using cleaned-test outcomes.

## Access and evidence boundaries

Separate processes perform development threshold selection, label-blind test application, and test scoring. The predictor/scorer reject training libraries and both original/new threshold-selector imports. All test predictions/actions must be sealed before appending a new global evaluation reservation and decoding the authenticated prior test-label copy. Validation and repeat builds may reconstruct the declared development selection, but test stages cannot import that selector or open original test-cache label columns. No prior artifact or registry record is overwritten.

Both development validation and the cleaned test have already been inspected in previous studies. This round's guardrail motivation is adaptive; predeclaration now does not erase that history. The user did not choose LR based on test rank. Report the result as exploratory sensitivity, not an independent generalization estimate. Keep the test-selected checkpoint family explicitly separate and biased; it is neither an independent result nor an oracle ceiling.

No M4 gate is rerun. Actual support/harm ratios, four-seed stability, largest-positive-plot concentration, and net gain after removing that plot are reported descriptively. This allows rare-scenario benefits to be inspected without treating concentrated observations as replicated evidence. The original M4 no-go and M5/M6 deferral remain unchanged.

## Fixed reporting

For each policy/family, report routed habitat top1/top3 accuracy versus raw fusion, pooled actions/rescues/harms, mean and per-seed net gain, coverage, acted plots, and subgroup outcomes. Router four-state probabilities and state metrics remain the original LR values: changing a threshold changes actions, not the learned state predictor.

Use 2,000 shared habitat-stratified whole-plot bootstrap draws with seed 20260826, keeping all seed realizations of a sampled plot together and preserving the predecessor's exact draw construction. Report gain-over-raw intervals and 27 paired new-policy-minus-matching-original-LR contrasts per family. These intervals condition on fixed models/thresholds/experts and do not repeat selection or correct for adaptive reuse, multiple comparisons, or checkpoint selection. No inferential claim is attached to being the best entry in the sensitivity table.

## Reproducible commands and artifacts

Run from the repository root:

```bash
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_guardrails.py build
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_guardrails.py validate
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_guardrails.py build
```

`build` runs `select`, `predict`, and `score` in separate CPU-only processes. `validate` independently reconstructs all three stages; repeat builds return `reused_valid` without rewriting committed files. Other commands are `validate-selection`, `validate-predictions`, and `validate-scores`. Path overrides are `--m4-root`, `--legacy-root`, `--output-dir`, and `--config`; scientific overrides and force-overwrite are not exposed.

Default output directory: `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/relaxed_guardrails_lr_v1`.

| Artifact | Contents |
| --- | --- |
| `declaration.json` | Pre-selection grid, parent/code/configuration/environment identities, cohort and access rules |
| `selection/selection.json` | All 27 development threshold curves/selections, inherited regularization identities, baseline parity and per-seed counts |
| `selection/frozen_policies.json` | Only the selected policy/guardrail/threshold/model identities needed by test application |
| `selection/{report.md,manifest.json}` | Development comparison and immutable selection seal |
| `predictions/experts.parquet` | Byte-identical prior 10,776-row expert table |
| `predictions/routers.parquet` | Byte-identical prior 32,328-row LR probability table; no new model predictions needed |
| `predictions/actions.parquet` | 377,160 rows: 35 policies × two families × four seeds × 1,347 images |
| `predictions/{metadata,manifest}.json` | Complete label-blind prediction seal and selection linkage |
| `scores/<family>/` | Eleven report artifacts: unchanged expert/router reference metrics, expanded policy/classification metrics, bootstrap/subgroup/outcome tables, targets, summary and paired contrasts |
| `scores/{labels.parquet,build_declaration.json,reservation.json,source_receipt.json,receipt.json,manifest.json}` | Authenticated labels, new append-only evaluation event, ownership and result integrity |
| `scores/{summary.json,report.md}` | Both-family performance with and without intervention |

## Results

The production build completed on 2026-09-13 in 84.99 seconds on CPU: development selection 1.50 s, label-blind application/sealing 5.95 s, and scoring 77.54 s. The declaration was published at `2026-09-13T20:42:42.448594+00:00`, before this study's threshold selection. No new threshold or cleaned-test outcome had been inspected at predeclaration.

### Development selection: which guardrail mattered?

All 27 declared combinations were evaluated and saved. **Changing the minimum acted-plot support from 20 to 10 or none did not change any selected threshold.** Relaxing the harm cap from 0.5R to 0.7R changed all three variants; increasing it to 0.9R selected the same thresholds as 0.7R. The following six rows therefore summarize all 27 policies without choosing a winner or omitting a distinct result. The complete grid and curves are in [`selection/report.md`](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/relaxed_guardrails_lr_v1/selection/report.md) and `selection/selection.json`.

| Features | Harm cap | Threshold | Actions / rescues / harms | Acted plots | Net | Gain (pp) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Full727 | 0.5R | Never intervene | 0 / 0 / 0 | 0 | 0 | 0 |
| Full727 | 0.7R or 0.9R | 0.1921296200383853 | 147 / 73 / 45 | 74 | 28 | +0.851582 |
| No-pairs79 | 0.5R | 0.43924158995984286 | 38 / 23 / 11 | 28 | 12 | +0.364964 |
| No-pairs79 | 0.7R or 0.9R | 0.13972750761099337 | 186 / 90 / 60 | 88 | 30 | +0.912409 |
| Numeric25 | 0.5R | 0.4251807095028868 | 55 / 34 / 14 | 33 | 20 | +0.608273 |
| Numeric25 | 0.7R or 0.9R | 0.11621510694147988 | 186 / 90 / 62 | 86 | 28 | +0.851582 |

Counts pool four seeds over 822 validation images (3,288 seed-image rows). Acted plots are a distinct-plot union, not a seed sum. All three original 20-plot/0.5R policies reproduce exactly. For primary/0.5R, removing the plot minimum is insufficient because the retained 1% coverage requirement still rules out its harm-compliant positive-net candidates.

The selected relaxed full727 threshold differs from the earlier descriptive 74-rescue/46-harm example at 0.18649393568083072. Both give 28 net corrections; the frozen tie-break prefers the new point's fewer harms (45) and then fewer actions. No test outcome is involved in this choice.

### Cleaned-test habitat accuracy, with and without routing

Accuracies are mean habitat top-1 percentages across all four seeds, not router four-state accuracy. Counts are pooled over four realizations of the same 1,347 images (5,388 rows per family), not independent images. Both families retain 531 plots. The expert table is unchanged:

| Expert | Fixed epoch-50 accuracy (%) | Test-selected checkpoint accuracy (%) |
| --- | ---: | ---: |
| Image | 73.144024 | 73.626578 |
| Geo | 61.618411 | 64.123979 |
| Raw concat, without router | 74.313289 | 75.334076 |

| Router features | Harm cap | Fixed accuracy (%) | Fixed gain (pp) | Fixed A/R/H | Test-selected accuracy (%) | Test-selected gain (pp) | Test-selected A/R/H |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| Full727 | 0.5R | 74.313289 | 0 | 0 / 0 / 0 | 75.334076 | 0 | 0 / 0 / 0 |
| Full727 | 0.7R or 0.9R | 75.000000 | +0.686711 | 265 / 125 / 88 | 75.408315 | +0.074239 | 208 / 87 / 83 |
| No-pairs79 | 0.5R | 74.703044 | +0.389755 | 79 / 42 / 21 | 75.334076 | 0 | 13 / 5 / 5 |
| No-pairs79 | 0.7R or 0.9R | 74.925761 | +0.612472 | 301 / 134 / 101 | 75.389755 | +0.055679 | 244 / 98 / 95 |
| Numeric25 | 0.5R | 74.684484 | +0.371195 | 85 / 43 / 23 | 75.426875 | +0.092799 | 27 / 12 / 7 |
| Numeric25 | 0.7R or 0.9R | 75.092799 | +0.779510 | 306 / 143 / 101 | 75.742390 | +0.408315 | 274 / 125 / 103 |

The plot minima 20, 10, and none give identical results within every row above because their development-selected action thresholds are identical. All 0.5R rows reproduce the original LR policies. The fixed-final relaxed gains exceed their corresponding original-LR point gains by +0.686711, +0.222717, and +0.408315 pp, respectively. This is a descriptive paired comparison, not test-based policy selection.

The complete 35-policy tables, top-3 accuracy, unchanged four-state metrics, seed-level classification, and subgroup reports are preserved in [`scores/report.md`](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/relaxed_guardrails_lr_v1/scores/report.md) and sibling JSON/Parquet files. Relaxed fixed-final top-3 accuracies are 93.596882%, 93.504083%, and 93.634001% (full/no-pairs/numeric), below raw fusion's 93.838159%. Higher top-1 does not imply that every metric improves.

### Uncertainty and concentration

The following conditional 95% whole-plot bootstrap intervals are in percentage points. Each row applies to both 0.7R and 0.9R and all three plot minima.

| Features | Fixed gain-over-raw interval | Fixed relaxed-minus-original-LR interval | Test-selected gain-over-raw interval | Test-selected relaxed-minus-original-LR interval |
| --- | --- | --- | --- | --- |
| Full727 | [-0.511771, +1.900269] | [-0.511771, +1.900269] | [-0.829467, +1.034120] | [-0.829467, +1.034120] |
| No-pairs79 | [-0.742571, +1.898538] | [-0.800031, +1.148424] | [-0.996533, +1.145687] | [-0.942135, +1.073622] |
| Numeric25 | [-0.523166, +2.014734] | [-0.573992, +1.328768] | [-0.779651, +1.577940] | [-0.785891, +1.378696] |

All new-policy gain intervals and all 54 paired contrast intervals include zero. Identical-policy contrasts have [0,0] because the predictions are identical, not because future performance is certain. Intervals do not adjust for repeated development/test use or multiple comparisons. The test-selected expert family has additional checkpoint-selection bias.

| Features, 0.7R/0.9R | Fixed per-seed net (1–4) | Fixed acted plots | Fixed pooled net after removing best positive plot | Test-selected per-seed net (1–4) | Test-selected acted plots | Test-selected pooled net after removing best positive plot |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Full727 | [6, 11, 9, 11] | 74 | 28 | [8, -5, -2, 3] | 75 | -4 |
| No-pairs79 | [6, 8, 9, 10] | 84 | 24 | [2, -5, 3, 3] | 77 | -6 |
| Numeric25 | [9, 13, 6, 14] | 87 | 32 | [10, -2, 4, 10] | 85 | 14 |

All 0.7R/0.9R fixed-final policies have positive net corrections in each seed and remain pooled-positive after removing their largest-positive-net plot. The four seeds share test images, however, so they are not independent replications. Test-selected-family results are less stable and full/no-pairs gains disappear after this descriptive plot removal. These diagnostics are not selection criteria or a rerun of the M4 gate.

The selected development harm cap is not a guarantee on transferred experts: fixed-final H/R is approximately 0.704, 0.754, and 0.706 for the three relaxed variants. Test-selected H/R is approximately 0.954, 0.969, and 0.824. Test policies remain unchanged even when these realized ratios exceed their development cap.

Conclusion: relaxing the harm cap yields larger fixed-final top-1 point gains for these frozen LR routers; relaxing plot support alone has no observed effect in this grid. Evidence remains exploratory and uncertain. Do not select a preferred model, feature set, guardrail, or checkpoint family from these test results, and do not reverse original M4's no-go or M5/M6 deferral.

## Verification and completion evidence

- The committed bundle has exactly 40 scientific artifacts: one declaration, four selection files, five prediction files, and thirty score files. The empty sibling lock is outside the bundle. All 27 new policies, both families, all four seeds, and all three feature variants are present: 216 new policy/seed/family applications and 290,952 new action rows, plus 86,208 unchanged original action rows.
- An independent development-only audit reconstructed all 29,007 candidate-curve records across the 27 policies, including exact integer feasibility, pooled/seed counts, distinct-plot unions, strict threshold masks, inherited C values, original-baseline parity, and every utility tie-break. It did not read test data.
- An independent test-metric audit reconstructed all 35 policies in both families from frozen actions and labels, including top-1/top-3 accuracy, rescues/harms/net, four-state targets, actual plot support, and concentration diagnostics. It independently regenerated the shared 2,000 plot draws, all bootstrap counts/deltas, and all 27 paired contrasts per family; draw identity and values match.
- Expert and LR probability Parquet files are byte-identical to their predecessor. All eight old action policies and old expert/router/classification/policy metrics, subgroup/outcome records, and 16,000 old bootstrap rows per family remain exact. Router four-state performance has not been changed by threshold relaxation.
- Independent `validate` completed successfully in 82.66 s, reconstructing selection, predictions, and scores in separate processes with the same three seals and evaluation event. No model fitting or test-side threshold selection occurred.
- The successful repeat `build` returned `reused_valid` in 82.26 s for every phase, with the same manifests, receipt, and evaluation event. All 40 output files retain their exact original bytes and mtimes after validation, the interrupted reuse attempt described below, and successful reuse. Their combined byte/mtime fingerprint is `f5252dd5f55d7d873eeef508b4d65e2bac15f390b2a5ed3cdcabc6e703711519`.
- Separate before/after byte-and-mtime audits preserve all six predecessor bundles: M4 feasibility (33 files), legacy LR transfer (34), legacy capacity transfer (36), router capacity (451), MLP source (148), and MLP replay (18). The seven old registry records also retain bytes and mtimes; exactly three new records remain after reuse, with no duplicate event.
- All 252 declared parent hashes validate. The provenance audit verified ordered declaration/selection/prediction/reservation/label-copy/source-receipt/completion timestamps, a correctly bound scoring receipt, exactly three new global registry records, and preservation of all seven existing records and the original first-event claim.
- **1,519 focused tests pass** across two collections: 215 new cases in 12.57 s and 1,304 predecessor cases in 331.93 s. New coverage includes exact 0.5/0.7/0.9 harm boundaries, all grid/feature/seed/family combinations, malformed sealed policy rejection before test-feature access, source and label access ordering, import capability guards, baseline and bootstrap parity, tampering, lock contention, owned interruption recovery, append-only registry behavior, and byte/mtime-preserving reuse. The predecessor collection excludes the known unrelated `test_multimodal_geo_10m.py` failure documented previously; that file is unchanged. Compilation and `git diff --check` pass.

The focused new suite is reproducible with:

```bash
conda run -n habcls python -m pytest -q tests/test_multimodal_geo_relaxed_guardrails_*.py
```

For the predecessor collection above, use `PYTHONMALLOC=malloc`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` in `habcls` and select `tests/test_multimodal_geo*.py`, excluding only `test_multimodal_geo_10m.py` and filenames containing `relaxed_guardrails`.

### Native-runtime caveat

After a successful build and independent validation, the first repeat-build attempt terminated with native exit code -11 in its score child (7.26 s overall), after selection and prediction reuse passed. All 40 artifact bytes and mtimes were unchanged. This resembles the [previously documented runtime instability](04_router_dataset_preparation.md#native-initialization-diagnostic), but the cause of this particular crash has not been isolated. No code, scientific setting, or artifact was modified in response. The same repeat command with diagnostic-only `PYTHONFAULTHANDLER=1` then passed in 82.26 s with `reused_valid`; all bytes/mtimes and parent hashes remained unchanged. A successful retry is not a claimed fix for native-library instability.

| Receipt or manifest | SHA-256 |
| --- | --- |
| Pre-selection declaration payload | `43d675f3847129fc7421457ef66ecb9f3c94b8429d9dd39055fe0de7cd28adc8` |
| Selection manifest payload | `0767b22a0c86da9423eaf3fceda9a564e9df8371332e03443eab5654e2e43623` |
| Prediction manifest payload | `69211355ac0ccaa285c34b430a07f16536b304e11681e6b541e9764ebef5c03e` |
| Score manifest payload | `1304810c4b5a31c31b05288af12dc8e67d3bbe0b41785df13dc7d80164199b9a` |
| Score receipt payload | `b0dc81e110facd692b35c3ee6fabf2f6395e43d70d1d3086eda8b89edcecac69` |
| Score receipt physical file | `0227557fa76bf30874ab2345b583e383df9fa6710896d562a6ea33261ce4616d` |
| Evaluation event ID | `6fde6dbfa6f84ec8b3169a3fbec4508f59f04d577f16130488e4ea717addffe1` |
| Cohort identity projection | `1dbe08ab297ea39a53fe0e183648a0ff42929364162b0d6835756817e6acf284` |

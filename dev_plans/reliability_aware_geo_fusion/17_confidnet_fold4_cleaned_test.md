# Frozen fold-4 routers and ConfidNet on cleaned test

Status: COMPLETE — application, reconstruction, independent audits and full
immutable reuse passed. Study ID: `confidnet_fold4_test_v1`.

The user explicitly requested applying the fold-4-trained router to cleaned
test and comparing a policy that selects the expert with highest estimated TCP.

## Results: no estimated-TCP benefit over matched fusion

Selecting the largest estimated TCP among image, geo, and fusion gives
**72.4388%** mean cleaned-test accuracy, below matched fusion's **73.1069%**.
The difference is **−0.6682 percentage points**, with conditional paired
whole-plot 95% interval **[−1.9071, +0.5924] pp**. The direct policy loses to
fusion in every seed, although this interval includes zero.

The numerical router with its frozen relaxed threshold reaches **73.1626%**:
only **+0.0557 pp** over fusion, interval **[−0.5167, +0.6351] pp**. The
TCP-augmented relaxed router reproduces fusion because all its saved validation
policies specify no intervention. Its active fixed-zero policy is harmful.

| Policy | Accuracy (%) | Weighted F1 (%) | Top-3 (%) | Changed predictions | Rescues | Harms | Wrong-to-wrong | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fusion | 73.1069 | 72.4130 | 91.9636 | 0 | 0 | 0 | 0 | 0 |
| Image | 71.6036 | 70.8096 | 90.9985 | 467 | 118 | 199 | 150 | −81 |
| Geo | 59.5212 | 58.8995 | 87.3979 | 1,845 | 372 | 1,104 | 369 | −732 |
| Numerical RF, fixed zero | 72.8842 | 72.1057 | 91.2769 | 320 | 116 | 128 | 76 | −12 |
| TCP-augmented RF, fixed zero | 71.6036 | 70.7702 | 91.1656 | 504 | 148 | 229 | 127 | −81 |
| Numerical RF, relaxed 0.7/0.9 | 73.1626 | 72.3947 | 91.6295 | 130 | 53 | 50 | 27 | +3 |
| TCP-augmented RF, all relaxed settings | 73.1069 | 72.4130 | 91.9636 | 0 | 0 | 0 | 0 | 0 |
| Highest estimated TCP, all three experts | 72.4388 | 71.4316 | 91.2398 | 504 | 158 | 194 | 152 | −36 |
| Estimated TCP, geo versus fusion | 71.0097 | 70.1523 | 91.2398 | 610 | 172 | 285 | 153 | −113 |
| Native MCP, geo versus fusion | 72.1232 | 71.2587 | 90.3675 | 422 | 117 | 170 | 135 | −53 |

Accuracy, weighted F1, and top-3 are arithmetic means across four seeds.
Counts pool **5,388 seed-image observations**, comprising four realizations
of the same **1,347 images / 531 plots**. They are not 5,388 independent images.
Weighted F1 is computed separately per seed before averaging. Top-3 uses the
chosen expert's logits. Intervention outcomes and counts are relative to
matched fusion.

### What the direct TCP policy selects

| Selected expert | Pooled selections | Share |
|---|---:|---:|
| Fusion | 1,903 | 35.32% |
| Geo | 892 | 16.56% |
| Image | 2,593 | 48.13% |

The policy chooses a different expert from fusion on 3,485 observations, but
2,981 of those choices leave the predicted class unchanged. The remaining
**504 changed predictions** cover 9.3541% of observations and 185 distinct
plots. Their 158 rescues and 194 harms give **−36 net corrections**; another
152 changes replace one wrong class with another.

Adding image to the direct TCP choice improves on the previous geo/fusion-only
comparator by **+1.4291 pp**, interval **[+0.4624, +2.5038] pp**. That comparator
is itself below fusion. The three-expert policy's weighted-F1 difference from
fusion is **−0.9814 points**, interval **[−2.2862, +0.2798]**.

| Seed | Fusion accuracy (%) | Numerical relaxed (%) | Highest TCP of three (%) | Augmented fixed-zero (%) | Numerical relaxed net | Three-expert TCP net |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 72.7543 | 72.7543 | 72.0861 | 71.0468 | 0 | −9 |
| 2 | 72.8285 | 72.9770 | 72.4573 | 70.5271 | +2 | −5 |
| 3 | 72.9027 | 73.0512 | 72.4573 | 71.7149 | +2 | −6 |
| 4 | 73.9421 | 73.8679 | 72.7543 | 73.1255 | −1 | −16 |

### Frozen router thresholds and uncertainty

The numerical 0.7/0.9 policy retains validation threshold
`0.1237670844365451` for every seed. Plot minima 20/10/0 therefore continue
to give identical policies. The numerical 0.5 policies and all nine augmented
policies remain typed never-intervene policies. No test feasibility check
selects a different threshold or fallback.

The numerical relaxed policy changes 130 predictions across 71 plots, with
53 rescues and 50 harms. Its test harm/rescue ratio is **50/53 ≈ 0.9434**,
above both 0.7 and 0.9. The guardrails constrained validation selection and
do not guarantee the same outcome ratio on test. These test outcomes are
reported without retuning. Its small accuracy gain accompanies slightly lower
weighted F1 and top-3; excluding its largest positive-net plot changes pooled
net from +3 to −3.

At zero threshold, the augmented router is **−1.5033 pp** below fusion,
interval **[−2.9652, −0.0380] pp**, and **−1.2806 pp** below the matched
numerical router, interval **[−2.4612, −0.0735] pp**. Both comparisons favor
the simpler references under these conditional intervals; this does not
remove the historical test-reuse limitation.

### Confidence estimation and remaining correction opportunities

| Expert | Test TCP MSE | Estimated-TCP failure AP | MCP failure AP | Estimated-TCP failure AUC | MCP failure AUC |
|---|---:|---:|---:|---:|---:|
| Image | 0.211418 | 0.452148 | 0.562457 | 0.731139 | 0.776473 |
| Geo | 0.168948 | 0.567430 | 0.604506 | 0.706697 | 0.724004 |
| Fusion | 0.194829 | 0.441339 | 0.551909 | 0.729495 | 0.777722 |

These are four-seed metric means. The confidence heads' training failure AP
was 0.9967–0.9989; the weak failure ranking on new data persists on cleaned
test. Estimated TCP remains below MCP for all three expert modes on average.
The router's rescue-ranking AP on geo/fusion disagreements, using
`q_rescue-q_harm`, is 0.336583 for numeric25 and 0.298874 for augmented28.

The same-checkpoint geo/fusion oracle reaches **80.0111%**, with 372 available
rescues pooled across seeds. Allowing any of the three experts raises the
oracle to **81.8857%**, with 473 available rescues. These label-dependent
references describe remaining opportunities; they are not deployable choices.

This application uses the original folds-1–3 expert checkpoints. Absolute
accuracy should not be treated as a matched comparison with earlier legacy
cleaned-test studies, which used different experts. All bootstrap intervals
condition on the fitted models and transferred thresholds, preserve whole
plots across seeds, and do not correct for previous model/policy exploration
or repeated test inspection. The result provides no evidence of a TCP benefit
in this frozen pipeline and does not establish that learned confidence can
never help routing.

## Fixed application setup

- Use all four seeds from `confidnet_fold4_v1`. Confidence heads, numerical
  feature temperatures/transforms, and both numerical25/augmented28 RF routers
  remain unchanged. All were fitted on 852 fold-4 development-train images.
- Use the same folds-1–3 expert encoder, image/geo/fusion heads, and geographic
  scaler that produced both splits in the preceding study. Recompute image
  representations with each matching encoder; historical cached image
  embeddings do not substitute for these fixed expert representations.
- Apply to the existing cleaned-test identity: 1,347 images from 531 plots.
  Decode identities/images/geographic inputs separately from labels.
- Apply the existing fixed-zero policies and all eighteen saved relaxed
  validation policies unchanged. In particular, every augmented relaxed policy
  retains fusion; the numerical 0.7/0.9 policies retain threshold
  `0.1237670844365451`. No test threshold search or model fitting.
- The requested direct comparator chooses the largest native estimated TCP
  among image, geo, and fusion. Exact ties favor fusion, then geo, then image.
  Include the previous geo/fusion-only TCP and MCP rules for context.
- Report each expert, both routers, and direct policies with accuracy, weighted
  F1, top-3, per-seed results, chosen-expert counts, and rescues/harms relative
  to matched fusion. A model switch with an unchanged predicted class is counted
  separately from a changed top-1 prediction.
- Report confidence regression and failure-detection quality, and the matched
  geo/fusion and three-expert accuracy ceilings. Neither oracle is an inference
  policy. Use 2,000 paired whole-plot bootstrap resamples, seed 20260826,
  retaining all images and seed realizations within each selected plot.
- Preserve native float32 confidence predictions and exact native state replay;
  report independent float64 differences diagnostically.
- Seal declaration and source fingerprints before inference. Seal predictions
  and policy choices for all four seeds before reserving a new append-only
  global test event and acquiring labels. Revalidation uses the acquired label
  copy and does not reopen the original label source.

The cleaned test has informed earlier research and has been evaluated by
multiple studies. This application is exploratory evidence, not independent
confirmation. It does not change the original M4 no-go decision.

## Results and execution

All 130 focused tests passed before execution, covering extraction, scoring,
whole-plot bootstrap arithmetic, registry ownership and label access, and
workflow replay. Applying the new inference code to the preceding validation
bundle reproduced all four numerical feature matrices and twelve native TCP
arrays bit-for-bit; eight portable forests differed from their original saved
sklearn probabilities by at most `2e-16`, with all 104 policy/seed choices exact.

The declaration was sealed at `2026-09-24T16:01:24.815515+00:00`, binding 134
source/configuration/protocol files, 1,347 raw images, all 130 parent-study
files, and the sixteen existing global registry records. Frozen expert inference
uses the RTX 4090; confidence estimation retains native float32 CPU execution.
Scientific sources and configuration are frozen after declaration.

All four inference and prediction stages completed, followed by scoring and
full reconstruction. The sixth global evaluation event is
`5d1c1a3d15f8935925722c13569da3f9015c71928c26bc6dcec52b9a20711fc6`.
All four prediction seals precede reservation; reservation precedes label
acquisition. The new event appends exactly three registry records and preserves
all sixteen preceding records.

Independent checks passed:

- All 1,347 image hashes and 134 source fingerprints match the declaration.
  All 130 parent-study files retain their original bytes, sizes and mtimes.
- Twelve native float32 confidence heads, four independently reconstructed
  numerical feature matrices, and eight forests replay exactly. All 104
  policy/seed choice vectors agree with independent calculations.
- Independent float64 confidence replay differs by at most `1.6105671e-6`;
  twelve head-row entries exceed the historical `1e-6` diagnostic boundary.
  This is diagnostic, consistent with the approved native-float32 convention.
- Across 5,388 observations, 183 have a tie for maximum native estimated TCP,
  including 165 tied at exactly one. The float64 diagnostic changes the chosen
  expert on 162 observations, all involving native ties, but changes the
  predicted class on zero observations. No float64 policy was scored or
  selected; the declared fusion→geo→image tie rule remains authoritative.
- The independent metric audit passed 5,886 checks with maximum discrepancy
  `1.11e-16`, including all 26 policies, confidence regression/ranking, oracles,
  saved tables, and all 75 paired contrasts × seven metrics using all 2,000
  whole-plot resamples.
- Full reconstruction and repeat-build reuse passed. All 35 scientific output
  files (17,481,257 bytes) retained their exact bytes, sizes and modification
  times. Reuse completed at `2026-09-24T16:13:15.335164+00:00`.

## Reproduction

From the repository root:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_multimodal_geo_helpfulness_confidnet_fold4_test.py build --device cuda
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_multimodal_geo_helpfulness_confidnet_fold4_test.py validate
```

Individual phases are `declare`, `extract-seed --seed N`, `predict-seed --seed N`,
`score`, `validate-scores`, and `validate`. The runner has no fitting, policy
selection, or force-overwrite options. Complete stages authenticate and reuse
their existing files.

The sibling output directory is
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1`:

- `declaration.json`: fixed settings, source fingerprints, copied validation
  threshold records, authenticated label-source lineage, and prior snapshots.
- `features/seed_N/`: label-blind expert hidden representations and logits,
  with identities, raw-image fingerprints, and matching expert provenance.
- `predictions/seed_N/`: calibrated numerical inputs, native estimated TCPs,
  router probabilities, all 26 choices, and float32/float64 replay diagnostics.
- `label_acquisition/`: authenticated canonical labels and acquisition receipt,
  created only after all predictions verify and the global event is reserved.
- `scores/`: full report, canonical scored rows, selected-expert/action rows,
  reservation, completion receipt, and result manifest.

Reviewable results:

- [Full machine-readable report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1/scores/report.json)
  and [all generated result tables](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1/scores/report.md).
- [Diagnostic figure, PNG](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1_review/fold4_cleaned_test_diagnostics.png)
  and [PDF](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1_review/fold4_cleaned_test_diagnostics.pdf).
- [Independent metrics audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1_review/score_audit.json),
  [model/feature audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1_review/models_and_features.json),
  and [provenance/registry audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1_review/provenance_models.json).
- [Immutable reuse receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1_review/reuse_receipt.json)
  and [review manifest](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_fold4_test_v1_review/manifest.json).

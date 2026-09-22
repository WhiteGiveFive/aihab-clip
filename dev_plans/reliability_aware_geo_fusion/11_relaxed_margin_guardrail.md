# Relaxed OOF margin-difference policy

Study ID: `relaxed_guardrails_margin_v1`.

Status: COMPLETE (2026-09-16). OOF selection, cleaned-test application and scoring, independent validation, and byte-preserving reuse succeeded. This is exploratory evidence; the original M4 no-go remains unchanged.

## Declared scope

The user requested completion and cleaned-test reporting of the previously
OOF-selected geo/raw margin-difference heuristic under relaxed guardrails.
This additive study introduces exactly one policy:
`relaxed__geo_minus_raw_margin__plots_20__harm_0p7`.

The score remains the calibrated geo top-two probability margin minus the
calibrated raw-fusion top-two probability margin. Intervene only when geo and
fusion disagree and the score is strictly greater than the frozen threshold.
Threshold candidates, zero/never-intervene options, and tie breaks remain those
of the original OOF heuristic. No learned router is fitted or used.

Select one common threshold from the same 3,378 development-training OOF
images across four seeds (13,512 seed-image rows, 1,300 distinct plots).
Maximize mean seed rescues minus harms, requiring:

- `10 * harms <= 7 * rescues` (exact integer arithmetic);
- pooled coverage at least 1%;
- actions on at least 20 distinct plots;
- strictly positive net corrections.

Ties prefer fewer harms, fewer actions, then the higher threshold. Reproduce the
original `H<=0.5R` heuristic and its threshold exactly before selecting the new
threshold. Keep every original artifact unchanged. Neither validation labels
nor test labels enter threshold selection; the OOF selection split is retained
even though the relaxed LR/RF thresholds used development-validation.

Evaluate only the expert family used in the current progress report
(`best_model` in artifact metadata): the same 1,347 cleaned-test images and 531
plots, with four seed realizations. Reuse already saved calibrated expert
probabilities and predictions. No expert/router refitting, recalibration,
model selection, new feature set, or test threshold search is permitted.

## Publication and access boundaries

Use a separate CPU-only runner with isolated selection, label-blind prediction,
and scoring processes. A declaration binds settings, code, environment and
parent hashes before new OOF selection. Seal thresholds before opening test
predictions; seal actions and reserve an append-only test event before reading
the authenticated predecessor label copy. No original test caches are opened.

All outputs are published under
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/relaxed_guardrails_margin_v1`.
Use exclusive locks, no-overwrite writes, exact ownership checks, and manifests
written last. Repeat runs reconstruct and verify immutable outputs without
changing committed bytes. Unknown or inconsistent partial files are rejected,
not deleted. Source study and registry records remain unchanged except for the
three append-only records identifying this new evaluation event.

## Reporting

Compare fusion alone, the original margin heuristic, and the relaxed margin
heuristic. Report mean-seed Acc@1 and weighted F1; pooled and per-seed actions,
rescues, harms, net corrections, coverage and plot support. Save class-level
metrics and confusion-matrix counts. Reuse 2,000 habitat-stratified whole-plot
bootstrap draws (seed 20260826), keeping seed realizations together, for gains
versus fusion and the paired relaxed-minus-original contrast.

This study follows earlier inspection of the cleaned test set. Results are
adaptive exploratory evidence, not independent confirmation. Bootstrap
intervals condition on the frozen experts and thresholds and omit selection
and adaptive-reuse uncertainty. Do not enforce the development harm constraint
on test labels or revise the threshold after scoring. The original M4 no-go and
M5/M6 deferral remain unchanged.

## Reproduction

From the repository root:

```bash
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_margin.py build
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_margin.py validate
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_relaxed_margin.py build
```

The final command checks byte-preserving reuse. The runner exposes path
overrides only; there are no scientific-setting or overwrite options.

## Results

### OOF threshold selection

All 4,029 original threshold candidates were retained. The original rule
reproduced exactly; 270 candidates met its guardrails, versus 1,810 under the
relaxed rule. The selected threshold fell from `0.44987864537943223` to
`0.1311806422313288`.

| OOF selection rule | Actions | Rescues | Harms | Net | Distinct plots |
|---|---:|---:|---:|---:|---:|
| Original H≤0.5R | 235 | 127 | 55 | 72 | 109 |
| Relaxed H≤0.7R | 960 | 427 | 289 | 138 | 343 |

Counts pool four realizations of the same 3,378 training images. The relaxed
rule has 7.1048% coverage and pooled H/R=0.6768. Guardrails apply to pooled
counts, not separately to every seed. These are threshold-fitting diagnostics,
not independent performance estimates.

### Cleaned-test comparison

Acc@1 and weighted F1 are arithmetic means over four seeds on the same 1,347
images. The fusion baseline and original heuristic reproduce prior results.

| Method | Acc@1 (%) | Weighted F1 | Gain over fusion (pp) |
|---|---:|---:|---:|
| Fusion alone | 74.313289 | 0.738352 | 0 |
| Fusion + original margin rule | 74.480327 | 0.738884 | +0.167038 |
| Fusion + relaxed margin rule | 74.387528 | 0.736112 | +0.074239 |

| Rule | Actions | Rescues | Harms | Net | Coverage | Distinct plots |
|---|---:|---:|---:|---:|---:|---:|
| Original | 93 | 34 | 25 | +9 | 1.726058% | 30 |
| Relaxed | 371 | 136 | 132 | +4 | 6.885672% | 102 |

Action counts pool 5,388 seed-image evaluations, not distinct images. The
relaxed rule has 103 interventions that change one wrong prediction to another.
Its per-seed net corrections are `[3, 6, -2, -3]`, versus `[4, -2, 2, 5]` for
the original rule. Test H/R=132/136=0.9706 exceeds the development cap; this is
reported without threshold adaptation.

The relaxed rule introduces 278 additional interventions, but their 102 extra
rescues are outweighed by 107 extra harms. It therefore has five fewer net
corrections than the original rule and lower weighted F1 than fusion alone.

| Comparison | Accuracy change (pp) | Conditional 95% interval (pp) |
|---|---:|---:|
| Original minus fusion | +0.167038 | [-0.299183, +0.675803] |
| Relaxed minus fusion | +0.074239 | [-1.226557, +1.428788] |
| Relaxed minus original | -0.092799 | [-1.134694, +0.989043] |

Conclusion: relaxing this margin heuristic increases intervention coverage but
does not improve its observed cleaned-test performance. There is no clear
accuracy benefit, and weighted F1 declines. These uncertain, reused-test
results do not change the original M4 decision or justify promoting the new
rule. No further threshold search was performed after scoring.

## Artifacts and verification

The new bundle contains 25 immutable files: the declaration; selection curve,
two frozen policy specifications and selection report; label-blind expert and
action tables; classification, policy, subgroup and bootstrap reports; copied
labels and source/ownership receipts; and three phase manifests. Numeric
reconstruction uses `atol=rtol=1e-12`; probability sums use `atol=1e-8`.

- Build completed in 9.85 s; independent runner validation in 7.38 s; repeat
  build returned `reused_valid` in 7.49 s.
- 382 focused tests pass, including 133 new selection, report, registry and
  workflow cases. Python compilation and whitespace checks pass.
- Independent direct-mask audit verified all 4,029 OOF candidates, both
  guardrail feasibility decisions, optimality/tie breaks, and per-seed counts.
- Independent test audit verified source expert/original-action parity, every
  per-seed confusion matrix and weighted-F1 value, and all 2,000 independently
  reconstructed bootstrap draws, gain replicates and interval quantiles.
- All 1,073 bound parent/code/registry hashes remain unchanged. The 1,068
  pre-existing source files also retain their original sizes and modification
  times. All 25 new output files retain their bytes and modification times
  through validation and repeat-build reuse.
- All 13 prior registry records are preserved; exactly three records append
  the fifth exploratory test event. No original test caches were reopened.

| Identity | SHA-256 |
|---|---|
| Declaration | `7ab4a502ab1cddb01cec2e27bc44e0699b42466a088108363117fc6ed05ba262` |
| Selection manifest | `0691bdce2b0c08cae64c024a924732fd85910be692a6423c22191a3b8f8cf68f` |
| Prediction manifest | `523d15f870aa1edbbe00e037221f1ad916aac228f0292b4aefc3741fd442f9ff` |
| Score manifest | `ae1580dd1ae4f987ab0210e0631c1e9a136c44719fea25d3793ad5424495d257` |
| Score receipt | `28236f2dc185d34ea3976249c4efb31144a59a739851c9faffeed84383a56069` |

# Relaxed threshold selection for the frozen TCP-augmented router

Status: COMPLETE — all 18 selections and development scores are sealed and
reproduce. Independent selection, metric, and bootstrap audits passed;
repeat execution preserves all eight output files and all 191 parent files.
The user requested applying the previous relaxed RF rule to the completed
OOF ConfidNet experiment.

Study ID: `confidnet_oof_relaxed_guardrails_v1`.

## Result

Relaxed threshold selection raises TCP-augmented accuracy from **69.5560% to
71.7762%**, a **+2.2202 percentage-point** change. The numerical baseline under
the same selection procedure reaches **72.0499%**. The augmented router remains
**0.2737 pp below** that matched baseline, with a conditional 95% paired interval
of **[−1.5885, +0.9896] pp**. This comparison does not establish an advantage
from the three estimated TCP inputs.

The relaxed 0.7 and 0.9 harm caps select identical policies for each arm.
Changing the plot minimum from 20 to 10 or none changes no selected policy.
The augmented router has no feasible threshold at the stricter 0.5 harm cap
under any plot minimum and therefore falls back to fusion.

| Policy | Threshold on `q_rescue − q_harm` | Accuracy (%) | Mean weighted F1 (%) | Actions | Rescues | Harms | Wrong-to-wrong | Net | Acted plots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fusion | Never | 71.0462 | 70.6695 | 0 | 0 | 0 | 0 | 0 | 0 |
| Numeric25, fixed zero | 0 | 71.9282 | 71.5737 | 239 | 111 | 82 | 46 | +29 | 102 |
| Augmented28, fixed zero | 0 | 69.5560 | 68.8671 | 482 | 166 | 215 | 101 | −49 | 145 |
| Numeric25, H≤0.5R | 0.181729206 | 71.7762 | 71.4091 | 76 | 44 | 20 | 12 | +24 | 38 |
| Augmented28, H≤0.5R | Never | 71.0462 | 70.6695 | 0 | 0 | 0 | 0 | 0 | 0 |
| Numeric25, H≤0.7R or H≤0.9R | 0.002916998 | 72.0499 | 71.6838 | 233 | 110 | 77 | 46 | +33 | 102 |
| Augmented28, H≤0.7R or H≤0.9R | 0.835174567 | 71.7762 | 71.3455 | 125 | 58 | 34 | 33 | +24 | 65 |

Metrics are means across four seeds, each on the same 822 validation images.
Counts pool 3,288 seed-image observations; acted plots count the union across
seeds. Seed copies are not independent images. Weighted F1 is computed within
each seed and then averaged, rather than from one pooled confusion matrix.

### What the selected threshold changes

The augmented router's threshold rises to `0.8351745672753053`; this is a cutoff
on **predicted rescue-minus-harm probability**, not on an individual expert's
estimated TCP. It removes 357 of the 482 fixed-zero interventions: 108 rescues,
181 harms, and 68 wrong-to-wrong changes. Avoiding those harms improves pooled
net corrections by 73, from −49 to +24. Coverage falls from 14.6594% to 3.8017%.

At the relaxed selected threshold, the augmented router makes fewer rescues
and harms than the matched baseline, but its net benefit is lower: +24 versus
+33 pooled corrections. Its observed harm/rescue ratio is 34/58 = 0.5862,
whereas the baseline lies exactly on the 0.7 boundary, 77/110.

| Policy | Seed 1 net | Seed 2 net | Seed 3 net | Seed 4 net |
|---|---:|---:|---:|---:|
| Numeric25, fixed zero | +8 | +9 | +4 | +8 |
| Augmented28, fixed zero | +1 | −4 | −5 | −41 |
| Numeric25, relaxed 0.7/0.9 | +8 | +13 | +5 | +7 |
| Augmented28, relaxed 0.7/0.9 | +1 | +8 | +4 | +11 |

The augmented selected policy has positive net corrections in every seed.
Its largest positive plot contributes six of 24 pooled net corrections;
removing that entire plot leaves +18. These are descriptive checks; neither
per-seed positivity nor this plot-removal result was used to choose thresholds.

### Threshold search diagnostics

The established selector evaluates 573 candidates for numeric25 and 841 for
augmented28 per guardrail setting, including never-intervene. At the 20-plot
minimum, feasible candidate counts for harm caps 0.5/0.7/0.9 are respectively
42/154/540 for numeric25 and 0/94/183 for augmented28.

The augmented candidates satisfying H≤0.5R and positive net have at most 24
actions (0.730% coverage), below the 33 pooled actions needed for 1% coverage.
Relaxing only plot support cannot resolve that failure.

Both relaxed winners also attain the maximum net corrections anywhere on their
respective nonnegative-threshold curves, even if the guardrail constraints are
ignored. Thus, within this saved scoring function and threshold search, a
different nonnegative cutoff cannot make the augmented router exceed the
baseline's best development net benefit. This is not a ceiling for other
features, model fits, policies, or independent evaluation data.

![Validation gain and coverage across thresholds](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_relaxed_guardrails_v1_review/threshold_sensitivity.png)

[PDF figure](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_relaxed_guardrails_v1_review/threshold_sensitivity.pdf).
Markers identify the fixed-zero and selected 0.7/0.9-cap policies; curves show
all finite candidate thresholds, including candidates that fail guardrails.

### Conditional uncertainty and interpretation

| Contrast | Accuracy difference (pp) | Conditional 95% interval (pp) |
|---|---:|---:|
| Augmented relaxed − augmented fixed zero | +2.2202 | [+0.2801, +4.2757] |
| Augmented relaxed − fusion | +0.7299 | [−0.1215, +1.6022] |
| Numeric25 relaxed − fusion | +1.0036 | [−0.4157, +2.3911] |
| Augmented relaxed − numeric25 relaxed | −0.2737 | [−1.5885, +0.9896] |

The matched weighted-F1 difference is −0.3383 points, with interval
[−1.6409, +0.9463]. All intervals use the shared 2,000 whole-plot resamples
described below, holding the chosen thresholds fixed.

Threshold selection and reporting use the **same previously inspected
development-validation set**. The intervals do not incorporate selection
uncertainty, adaptive validation reuse, or multiple comparisons. In particular,
the positive selected-versus-zero interval is not independent confirmation of
generalization. There is no new cleaned-test result.

This follow-up narrows the interpretation of the earlier fixed-zero result:
zero was a poor intervention threshold for the augmented router, and threshold
selection recovers a positive development point gain over fusion. However,
estimated TCP still provides no demonstrated improvement over the matched
numerical router. The original confidence generalization and representation
transfer concerns remain unresolved by changing the intervention threshold.

## Declared scope

Reuse the saved predictions of all eight routers in
`confidnet_oof_v1_replay_v2`: numeric25 and TCP-augmented28, seeds 1–4.
Apply the exact selector from [the previous RF study](10_relaxed_guardrails_rf.md)
to each arm separately. No new fitting or cleaned-test evaluation is included.

- Grid: minimum distinct acted plots 20, 10, or 0, crossed with pooled harm
  caps `10H <= 5R`, `10H <= 7R`, and `10H <= 9R` (18 policies).
- Keep pooled coverage at least 1% and strictly positive net corrections.
- Select one common threshold per arm/setting across all four seeds using
  development-validation, maximizing net corrections. Ties prefer fewer harms,
  fewer actions, then a higher threshold.
- Candidates are never-intervene, zero, and float64 midpoints between adjacent
  distinct positive scores from all rows, including agreement rows.
- Switch to geo only on disagreement and `q_rescue - q_harm > threshold`.
  No feasible candidate means never-intervene. There is no per-seed guardrail.
- Retain fusion and each arm's fixed-zero policy as reference results. Report
  all settings, per-seed and pooled outcomes, weighted F1, and paired differences.
- Retain the ConfidNet study's 2,000 unstratified whole-plot bootstrap draws,
  seed 20260826, shared across policies with all images and seed realizations
  of each plot kept together. Thresholds remain fixed within these diagnostic
  intervals; they do not account for threshold selection or validation reuse.

The source configurations/models and earlier results remain unchanged. New
outputs are stored separately, with a declaration before threshold selection,
authenticated parent artifacts, complete curves and a reproducible runner.
This evaluates threshold-selected development performance, not independent
generalization. It does not change the original M4 no-go decision.

## Reproducibility and verification

The declaration was written at `2026-09-22T21:02:36.724544+00:00` before the
production threshold selection. Source/configuration hashes and a complete
191-file parent snapshot are stored with it. All parent stage files were
authenticated against their manifests before row decoding. Saved validation
logits and both routers' four-state probabilities were checked exactly against
their original feature/prediction archives; identities and labels agree across
all four seed realizations.

From the repository root:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/analyze_confidnet_relaxed_guardrails.py build
/home/hshi/anaconda3/envs/habcls/bin/python tools/analyze_confidnet_relaxed_guardrails.py validate
```

The first command now returns `reused_valid`. It does not refit models. A
separate validation pass reconstructs all selections, action masks, metrics,
and bootstrap draws. There is no overwrite or automatic partial-run recovery
option.

Verification completed:

- 78 existing relaxed-selector tests passed, including strict boundaries,
  exact integer harm caps, coverage, support, ties, and never-intervene fallback.
- An independent brute-force implementation verified all 12,726 candidate/grid
  records and all 21 action masks (18 selected plus three reference policies).
- 840 independent pooled/per-seed metric comparisons passed, maximum absolute
  difference `2.22e-16`.
- An independent bootstrap using per-image weights and confusion matrices
  verified all 84,000 saved metric values, maximum difference `3.33e-16`, and
  all 288 reported contrast values, maximum difference `2.17e-16`.
- The parent fixed-zero metrics and paired bootstrap interval reproduce.
- All nine numeric25 selected thresholds and pooled/per-seed counts reproduce
  the historical RF sensitivity results. The full candidate arrays are not
  byte-identical: the current saved probabilities yield 573 candidates versus
  572 historically, with tiny floating-point boundary differences. All 240
  distinct candidate count outcomes per setting match. No historical curve
  identity is claimed.
- The repeat build preserves bytes, sizes, and modification times for all
  eight new scientific files and all 191 parent files.

The sealed bundle is
[`confidnet_oof_relaxed_guardrails_v1`](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_relaxed_guardrails_v1).
Its files are declaration, full selection curves, validation rows, action masks,
JSON/Markdown reports, bootstrap draws, and the manifest. Useful entry points:

- [All policy results and contrasts](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_relaxed_guardrails_v1/report.md)
- [Machine-readable report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_relaxed_guardrails_v1/report.json)
- [Full threshold curves](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_relaxed_guardrails_v1/selection.json)
- [Independent audit receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_relaxed_guardrails_v1_review/comparison_audit.json)
- [Immutable-reuse receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_relaxed_guardrails_v1_review/reuse_receipt.json)

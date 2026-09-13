# Frozen RF/XGB/MLP transfer to legacy cleaned-test experts

Status: COMPLETE — both checkpoint families scored, independent reconstruction and immutable reuse verified. This exploratory transfer does not reverse M4's no-go.

Study ID: `legacy_capacity_router_transfer_v1`. This separately versioned, apply-only evaluation extends the earlier [legacy logistic-router transfer](05_legacy_router_transfer.md). It does not complete M5, promote a router, revise original M4's no-go, or provide independent confirmation.

## Fixed scope and access boundaries

Use exactly the prior study's 1,347 cleaned-test images from 531 distinct plots, all four seed realizations, and both existing expert checkpoint families:

- `best_model`: fixed epoch-50 legacy heads, historically test-exposed.
- `oracle_test_best_model`: heads selected using cleaned-test accuracy; biased diagnostics, not an independent result or an oracle ceiling.

Reuse the validated, label-blind `exploratory_legacy_router_v1/predictions/experts.parquet` expert logits, native/transferred probabilities, and 30 semantic features. The original expert checkpoints and caches remain immutable and hash-bound through that parent's lineage. There is no new encoder/head fitting and no new expert checkpoint selection. This also retains the historical encoder/cache provenance limitations documented in the earlier study.

Apply the same twelve OOF temperatures and four seed-specific M4 numeric/categorical transforms unchanged. Do not refit any calibration, feature transformation, learner, or threshold on the cleaned test. Calibration quality for these different legacy experts is not guaranteed.

The new frozen policies are RF, XGB, and MLP with each of the full727, no-pairs79, and numeric25 variants: nine policies and 36 seed-specific states. RF/XGB states come from `router_capacity_v1`; MLP states come from the unchanged `router_mlp_v1` models referenced by `router_mlp_v1_replay_v2/scores/frozen_policies.json`. Configurations and typed strict thresholds are transferred from those sealed development-selection receipts, including never-intervene policies. Actions still require raw/geo disagreement.

Retain the old three LR policies and five heuristic policies unchanged, giving twelve router policies and seventeen total policies per checkpoint family. Do not choose a winning checkpoint family or policy using these test results.

Before new test-feature decoding, publish a declaration binding configuration, code, all parent files, model/threshold identities, cohort, and ontology. Seal every prediction/action set for both checkpoint families before reserving a new append-only global test event and decoding labels. Acquire labels from the authenticated prior scoring copy, not from original test-cache label columns. Revalidation uses the new sealed copy. The scorer and predictor are both fitting-incapable and do not import fitting or model-selection modules.

## Fixed reporting

Compute expert calibration and top1/top3 metrics, router four-state accuracy/NLL/Brier/macro-F1 and rescue/harm AP, routed habitat classification, actions/rescues/harms/net gains, support-aware subgroup results, and plot-bootstrap uncertainty. State prediction accuracy is not habitat accuracy after correction.

Each checkpoint family is scored separately because its expert correctness and four-state targets differ. Counts marked pooled sum four seed realizations (5,388 rows per family), not independent images. Use 2,000 shared complete-plot bootstrap draws with seed 20260826. The 27 predeclared paired contrasts per family are:

- Each of nine new policies minus matching-feature LR and minus the frozen margin heuristic: 18 contrasts.
- MLP minus matching-feature RF and XGB: six contrasts.
- RF minus matching-feature XGB: three contrasts.

These are nominal, descriptive intervals conditional on supplied experts and already selected policies. They do not account for prior test-selected checkpoints, repeated evaluation, model/threshold selection, multiple comparisons, or possible dependence between distinct plots. There is no threshold search, new validation selection, or M4 gate rerun.

## Reproducible commands and artifacts

From the repository root:

```bash
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_legacy_capacity.py build
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_legacy_capacity.py validate
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_legacy_capacity.py build
```

`build` runs prediction and scoring in separate CPU processes; `validate` reconstructs both without fitting; repeated builds validate/reuse committed artifacts without changing bytes. Internal phases are `predict`, `score`, `validate-predictions`, and `validate-scores`. Path overrides are `--legacy-root` (the sealed earlier transfer bundle, not the raw expert-run directory), `--capacity-root`, `--mlp-source-root`, `--mlp-replay-root`, `--output-dir`, and `--config`. Scientific values must match the fixed declaration; no force-overwrite option exists.

Default output: `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/exploratory_legacy_capacity_v1`.

| Artifact | Contract |
| --- | --- |
| `declaration.json` | Pre-application parents, selected states/thresholds, cohort, access boundaries, code/environment/configuration identity |
| `predictions/experts.parquet` | The same 10,776 label-blind expert rows across both checkpoint families |
| `predictions/routers.parquet` | 129,312 rows: twelve routers × four seeds × two families × 1,347 images; four float64 state probabilities and utility score |
| `predictions/actions.parquet` | 183,192 rows: seventeen policies × four seeds × two families × 1,347 images; frozen action rules without labels/outcomes |
| `predictions/{metadata,manifest}.json` | Dimensions and immutable complete-prediction seal |
| `scores/labels.parquet` | Byte-identical copy of the authenticated prior 1,347-image label table |
| `scores/<checkpoint_family>/` | Eleven report artifacts: expert/classification/router/policy metrics, bootstrap summary/replicates, subgroups, outcomes, four-state targets, summary, and 27 paired comparisons |
| `scores/{build_declaration,reservation,source_receipt,receipt,manifest}.json` | Ownership, label acquisition, append-only global event, source/result integrity |
| `scores/{summary.json,report.md}` | Side-by-side machine-readable and human-readable results |

Schemas are explicit, float64 inference/reconstruction uses `atol=rtol=1e-12`, and probability row normalization uses `atol=1e-8`. Publication is exclusive-lock, no-overwrite, and manifest-last. Unknown files are preserved and rejected; only demonstrably owned partial outputs may be completed. Existing registry records are never replaced.

## Results

All nine new policies were evaluated without fitting or selection. No new RF/XGB/MLP cleaned-test metrics had been decoded when this scope was declared. The first production build completed in 95.56 seconds on CPU (prediction/sealing 33.90 s, scoring 61.66 s).

The following are **habitat top-1 accuracies**, averaged over four seeds. Gains are percentage points relative to the matching raw-fusion expert family. Each count triple is **actions / rescues / harms**, summed over four seed realizations; it is not a count of independent images. The complete top-3, per-seed, state, subgroup, and bootstrap results are in [`scores/report.md`](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/exploratory_legacy_capacity_v1/scores/report.md) and its sibling JSON/Parquet artifacts.

| Expert | Fixed epoch-50 top-1 (%) | Test-selected checkpoint top-1 (%) |
| --- | ---: | ---: |
| Image | 73.144024 | 73.626578 |
| Geo | 61.618411 | 64.123979 |
| Raw concat | 74.313289 | 75.334076 |

| Router / features | Fixed top-1 (%) | Fixed gain (pp) | Fixed A/R/H | Test-selected top-1 (%) | Test-selected gain (pp) | Test-selected A/R/H |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| LR / full727 | 74.313289 | 0 | 0 / 0 / 0 | 75.334076 | 0 | 0 / 0 / 0 |
| LR / no-pairs79 | 74.703044 | +0.389755 | 79 / 42 / 21 | 75.334076 | 0 | 13 / 5 / 5 |
| LR / numeric25 | 74.684484 | +0.371195 | 85 / 43 / 23 | 75.426875 | +0.092799 | 27 / 12 / 7 |
| RF / full727 | 74.665924 | +0.352635 | 100 / 48 / 29 | 75.501114 | +0.167038 | 34 / 19 / 10 |
| RF / no-pairs79 | 74.647365 | +0.334076 | 121 / 56 / 38 | 75.445434 | +0.111359 | 41 / 21 / 15 |
| RF / numeric25 | 74.665924 | +0.352635 | 118 / 55 / 36 | 75.538233 | +0.204157 | 38 / 22 / 11 |
| XGB / full727 | 74.368968 | +0.055679 | 71 / 32 / 29 | 75.296956 | -0.037120 | 7 / 2 / 4 |
| XGB / no-pairs79 | 74.406088 | +0.092799 | 92 / 40 / 35 | 75.334076 | 0 | 14 / 5 / 5 |
| XGB / numeric25 | 74.424647 | +0.111359 | 76 / 34 / 28 | 75.352635 | +0.018560 | 4 / 2 / 1 |
| MLP / full727 | 74.313289 | 0 | 0 / 0 / 0 | 75.334076 | 0 | 0 / 0 / 0 |
| MLP / no-pairs79 | 74.313289 | 0 | 0 / 0 / 0 | 75.334076 | 0 | 0 / 0 / 0 |
| MLP / numeric25 | 74.573125 | +0.259837 | 69 / 35 / 21 | 75.371195 | +0.037120 | 3 / 2 / 0 |

Full/no-pairs MLP makes zero corrections because development selection froze a never-intervene policy, not because its state probabilities are constant. No thresholds were changed to induce test actions. RF configurations are depth 4/minimum leaf 5; XGB configurations are depth 2/lambda 1; all MLP configurations use hidden widths `[32]`, as selected before this evaluation.

### Conditional uncertainty and interpretation

| New policy | Fixed gain 95% interval (pp) | Test-selected gain 95% interval (pp) |
| --- | --- | --- |
| RF / full727 | [-0.207458, +0.983976] | [-0.126825, +0.469935] |
| RF / no-pairs79 | [-0.327706, +1.018753] | [-0.275938, +0.483271] |
| RF / numeric25 | [-0.296911, +1.013366] | [-0.094349, +0.526199] |
| XGB / full727 | [-0.428170, +0.562430] | [-0.164495, +0.056820] |
| XGB / no-pairs79 | [-0.438316, +0.650786] | [-0.145246, +0.129824] |
| XGB / numeric25 | [-0.353732, +0.624610] | [-0.037148, +0.076576] |
| MLP / full727 | [0, 0] | [0, 0] |
| MLP / no-pairs79 | [0, 0] | [0, 0] |
| MLP / numeric25 | [-0.185195, +0.738076] | [0, +0.093214] |

Every new-policy interval versus raw fusion includes or touches zero. Intervals `[0,0]` describe the exact same never-intervene predictions, not certainty about future generalization. The test-selected numeric MLP result is only two rescues and no harms among three actions, all two net rescues occurring in seed 1; its zero lower endpoint is not strong evidence of improvement.

Of the 54 predeclared paired contrasts, two nominal conditional intervals exclude zero, both for fixed-final experts: RF/full minus XGB/full is +0.296956 pp [ +0.036833, +0.582678 ]; XGB/no-pairs minus LR/no-pairs is -0.296956 pp [ -0.596110, -0.018437 ]. The other 25 fixed-final and all 27 test-selected contrasts include or touch zero. These intervals are not adjusted for the 54 comparisons or adaptive test reuse and do not establish a general learner ranking.

RF produces the largest positive point gains among the new models, but does not beat the earlier LR/no-pairs point result on fixed-final experts. All RF variants have negative net utility in fixed-final seed 3 despite positive four-seed mean gains. Numeric-only MLP is modestly helpful on fixed-final experts; XGB interventions are less effective. These observations do not support replacing the original method or reversing its gate.

Transfer also weakens the development support/harm properties: for fixed-final experts, every one of the seven intervening new policies has more than one harm per two rescues; for test-selected experts, every new policy acts on fewer than 1% of seed-image rows. These are descriptive checks of transferred behavior, not a rerun of M4 selection or permission to adjust thresholds.

### Four-state accuracy is a separate task

| Router / features | Fixed four-state accuracy (%) | Test-selected four-state accuracy (%) |
| --- | ---: | ---: |
| LR / full727 | 75.296956 | 66.759465 |
| LR / no-pairs79 | 75.241277 | 66.889384 |
| LR / numeric25 | 75.408315 | 66.035635 |
| RF / full727 | 75.426875 | 72.178916 |
| RF / no-pairs79 | 75.371195 | 71.863400 |
| RF / numeric25 | 75.482554 | 71.937639 |
| XGB / full727 | 75.983667 | 70.211581 |
| XGB / no-pairs79 | 75.779510 | 70.489978 |
| XGB / numeric25 | 75.686711 | 70.601336 |
| MLP / full727 | 71.956199 | 67.576095 |
| MLP / no-pairs79 | 73.088344 | 67.483296 |
| MLP / numeric25 | 74.424647 | 69.766147 |

For example, fixed-final XGB/full has higher four-state accuracy than RF/full (75.984% versus 75.427%), yet produces a smaller habitat gain (+0.056 versus +0.353 pp). Overall state accuracy is therefore not a substitute for rescue/harm discrimination, probability quality, and the utility of the frozen intervention rule. State NLL, Brier, macro-F1, and disagreement-only rescue/harm AP are preserved in each family's `router_metrics.json`.

## Verification and completion evidence

- Both expert families, all three new learners, all three feature variants, and all four seeds are present: 36 frozen states and 72 family/seed/policy applications. The output has exactly 36 committed files.
- An independent prediction-only audit reproduced all 96,984 new probability rows with maximum absolute error 0. All probabilities are finite/in range; maximum normalization error is 1.55e-15. No fitting libraries or labels were used in that audit.
- The 10,776-row expert table is byte-identical to its predecessor. All 32,328 prior LR probability/score rows and all 86,208 prior eight-policy action rows remain exact, allowing only the declared LR metadata additions. Both families' predecessor metric values, subgroups, outcomes, targets, and bootstrap rows also remain exact.
- Independent metric checks reproduced all twelve router metric groups, all seventeen policy classification/outcome groups, and all 27 paired intervals in each checkpoint family.
- Independent `validate` completed successfully in 77.76 s, reconstructing both predictions and scores without fitting, new selection, or original test-cache label access.
- All 931 declared parent-file hashes pass, including original M1–M4, capacity/MLP selection receipts, original legacy experts/caches, and relevant code/configuration. A separate audit confirmed exactly three new global registry records and preservation of all four prior records.
- Repeat `build` returned `reused_valid` in 79.33 s, with no fitting and the same prediction/score manifests and evaluation event. All 36 output files retain exactly the same bytes and mtimes across validation/reuse; the combined fingerprint is `1cc9c09fc3bac5eca562ca24262dc16ce8ec11f854ec755644f3d434ab9d4ffa`.
- Separate before/after byte-and-mtime fingerprints are unchanged for all four predecessor bundles: legacy transfer 34 files (33 artifacts plus its existing empty lock), capacity 451 files, MLP source 148 files, and MLP replay 18 files. All 931 declared parent hashes also validate after the final repeat build.
- **1,304 focused tests pass**: 144 new transfer cases in 53.89 s plus 1,160 predecessor geographic cases in 281.14 s. Coverage includes all feature/seed/family combinations, portable state replay, exact baseline preservation, all paired bootstrap contrasts, label-access ordering, forbidden fitting capabilities, tampering, lock contention, owned interruption recovery, registry append-only behavior, and byte-preserving reuse. The predecessor collection excludes the known unrelated `test_multimodal_geo_10m.py` failure documented in the earlier legacy record; that file remains unchanged. `git diff --check` passes.

The new focused suite is reproducible with:

```bash
conda run -n habcls python -m pytest -q tests/test_multimodal_geo_legacy_capacity_*.py
```

For the predecessor collection used above, run this from the repository root in `habcls` with `PYTHONMALLOC=malloc`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`:

```python
from pathlib import Path
import pytest

files = sorted(
    str(path) for path in Path("tests").glob("test_multimodal_geo*.py")
    if path.name != "test_multimodal_geo_10m.py"
    and "legacy_capacity" not in path.name
)
raise SystemExit(pytest.main(["-q", *files]))
```

| Receipt or manifest | SHA-256 |
| --- | --- |
| Pre-application declaration payload | `d5845e83e135abeebfc4ec31a4a2194fddfc51c46b15fef3478eaf68f8e6a984` |
| Prediction manifest payload | `a3ec432b00a9a703f4fac6b4da528f227f90b917a832e8e38656b858c3f8dc35` |
| Score manifest payload | `ce11dc4a38d3cacd5b47902526d3c5c1c89d2bb1679919a06a9e386adc446241` |
| Score receipt payload | `a4c22c716cbfda4b88e8abc8c84ed29c2dcabfb166cc1e9ec76d3b80a0a0beb4` |
| Score receipt physical file | `1ba347456be1673d32d78a4997c88656b15a6d84695174173a68ccb75278869a` |
| Evaluation event ID | `34814890dfa26623a4c191278b9734d717b012f894d032fe351577efddcc0ea7` |
| Cohort identity projection | `1dbe08ab297ea39a53fe0e183648a0ff42929364162b0d6835756817e6acf284` |

Conclusion: this is the requested frozen transfer evaluation, not a new training or test-selection round. It supplies modest and uncertain correction gains for some policies under legacy-expert distribution shift. Preserve the original M4 no-go and M5/M6 deferral; do not promote a policy or choose an expert checkpoint family from these exploratory results.

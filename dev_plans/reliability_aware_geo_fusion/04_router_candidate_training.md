# M4 — Primary and Ablation Router-Candidate Training

## Status and completion boundary

**Status:** The original study was predeclared on 2026-09-08 before router
fitting or router development-validation scoring. Its first real build stopped
at the frozen convergence limit, with 3 of 48 candidates committed. On the same
date, the user approved a versioned optimizer-budget amendment to 5,000 maximum
iterations. The amended `router_candidates_v2` study has fitted and sealed all
48 candidates into a new directory. Independent reconstruction, explicit
48-fit reproduction, and byte-preserving reuse all passed. **Candidate
training is complete.** Subsequent validation prediction generation and one
fixed accuracy preview are now complete; see
[the validation-preview record](04_router_validation_preview.md). Model/threshold
selection and the remaining M4 feasibility evaluation are still outstanding.

This subtask fits and seals every declared router candidate from the existing
OOF router-training dataset. It makes candidates available to later label-blind
validation prediction generation. It does not generate or score validation
predictions, select regularization or intervention thresholds, promote an
ablation, refit experts or feature transforms, or complete the remaining M4
feasibility assessment. The cleaned test set is outside this workflow.

The protocol-v1 full-feature primary objective, feature schema, and sealed
M1–M4 dataset parents remain unchanged. The amended study changes only the
optimizer iteration budget, explicitly and separately from the frozen parent
configuration. The two feature-mask ablations below remain a separately named
secondary study. They cannot replace the primary method on the strength of
validation results; changing the primary method requires a new protocol
version.

## Approved optimizer-budget amendment

The original `router_candidates_v1` attempt stopped when primary seed 1 at
`C=10` failed to converge within 1,000 iterations. The user approved raising
the iteration cap to 5,000 on 2026-09-08, before any router validation prediction
generation or scoring. This is a disclosed, convergence-driven amendment,
not a claim that the 5,000-iteration budget was part of the original study.

The machine-readable amended declaration is
`configs/multimodal_geo_router_candidates_v2.json`, and the new artifact
directory is `router/router_candidates_v2/`. All 48 candidates are refitted
under the amended declaration, including the three that already converged in
v1. The original `configs/multimodal_geo_router_candidates_v1.json` and the four
partial v1 files (study specification plus three fitted states) are preserved
without overwrite or deletion. No v1 candidate is copied into the v2 study.

Only `max_iter` changes from 1,000 to 5,000. The objective, L2 penalty, solver,
convergence tolerance, `C` grid, seeds, feature variants, class weighting,
preprocessing, dataset, row order, and all sealed parents remain unchanged.
Failure to converge within the new budget remains a hard rejection; there is
no automatic relaxation or further increase.

## Predeclared comparison

The comparison inherited from the v1 predeclaration and retained in
`configs/multimodal_geo_router_candidates_v2.json` comprises these three
variants:

| Variant | Role | Semantic features | Router inputs |
|---|---|---:|---:|
| `primary_full727` | Frozen protocol-v1 primary | 30 | 727 |
| `ablation_no_pairs79` | Secondary; omit both predicted-class pairs | 28 | 79 |
| `ablation_numeric25` | Secondary; omit all categorical features | 25 | 25 |

Each variant uses seeds 1–4 and the same `C = [0.01, 0.1, 1.0, 10.0]` grid:
16 candidates per variant, 48 candidates total. Each candidate is fitted on
its seed's 3,378 OOF rows from 1,300 plots. Seeds are not pooled or selected.
The four-state target order is `rescue`, `harm`, `both_correct`, `both_wrong`.

Every variant reuses the already frozen seed-specific calibrated feature
matrix. The ablations select columns from that matrix in their original order;
they do not refit temperatures, standardization, or categorical vocabularies.
`ablation_no_pairs79` removes the 648 one-hot columns for
`image_geo_pred_pair` and `geo_raw_pred_pair`. `ablation_numeric25` keeps only the
25 `scaled__` columns, including the Boolean and integer feature families.

All candidates use the multinomial logistic model: L2 penalty, L-BFGS
solver, fitted intercept, no class weighting, no warm start, `dual=False`,
`n_jobs=1`, the approved `max_iter=5000`, `tol=1e-8`, and `random_state=None`.
Nonfinite results or lack of convergence are failures, not permission to change
settings.
CPU fitting is deterministic under the recorded environment and canonical row
order; numerical reproducibility is checked explicitly rather than assumed.
Candidates run sequentially with native numerical-library thread counts limited
to one. Explicit reproduction compares coefficients/intercepts with
`atol=rtol=1e-10` and four-state probabilities with `atol=rtol=1e-12`.

The later scoring-only stage will select one common `C` per variant by the
arithmetic mean of the four seed-specific validation four-state log losses.
Ties use the smallest `C`. It will retain all four routers at that `C`, never a
best seed. Threshold selection for each variant will then use the same frozen
utility, coverage, plot-support, harm, and tie-breaking rules. These selection
rules are declared here; neither selection nor validation scoring occurs now.
All three variants must be reported, including unsuccessful comparisons.

## Manual and agent-driven commands

Run the same commands from the repository root in the existing `habcls`
environment:

```bash
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_candidates.py build
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_candidates.py validate
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_candidates.py build
```

These commands now default to the amended v2 study; they do not resume or
overwrite the partial v1 run. The first `build` fits missing v2 candidates and
seals the complete bundle. It can resume interrupted work by reusing valid
committed candidates. `validate`
performs an independent integrity and apply-only numerical check without
fitting. The final `build` checks idempotent reuse: it should perform no new
fits and preserve all committed bytes. Corrupt or stale committed state is a
hard error; these commands do not silently overwrite it.

For an explicit numerical rerun of all 48 fits against the sealed results:

```bash
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_candidates.py reproduce
```

`reproduce` deliberately fits again, compares the newly computed results with
the committed states, and never replaces those states. This differs from the
ordinary repeat-build reuse check. Run `build` first if no complete bundle
exists.

If the inherited native-runtime crash occurs during parent checks, the
diagnostic invocation used successfully for full validation is:

```bash
PYTHONFAULTHANDLER=1 PYTHONMALLOC=malloc conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_candidates.py validate
```

The same environment prefix can be applied to `build` or `reproduce`. It does
not relax convergence, change model settings, or overwrite artifacts. The
underlying intermittent interpreter crash is not claimed to be fixed; keep
its diagnostic output if it recurs. Conda's optional `--no-capture-output`
flag exposes per-candidate progress immediately.

All three commands expose only `--config`, `--protocol-dir`, `--artifact-root`,
and `--study-config` path overrides. The last defaults to the predeclared study
JSON above. There are no ad hoc feature, seed, `C`, optimizer, validation-label,
test-data, or force-overwrite flags. Successful summaries are JSON on standard
output, with per-candidate progress events on standard error. Workflow
exceptions produce JSON on standard error and exit status 1; invalid CLI
arguments use the standard parser error and exit status 2.

## Data access, artifacts, and following-stage handoff

The artifact/training workflow is
`multimodal/geo_helpfulness_router_candidates.py` and its runner is
`tools/run_multimodal_geo_helpfulness_m4_candidates.py`. The candidate bundle
binds the study declaration, code/environment identity, feature-mask order, and
sealed parent lineage. The existing dataset-preparation implementation and all
its state remain separate and unchanged.

The inherited full parent-integrity preflight can read development assignments
and source tables containing validation rows. This does not make them training
inputs: router fitting receives only its seed's OOF feature matrix and
four-state OOF target. No validation expert-output table, validation-scoring
routine, final-development fit, or test source is opened by this training stage.

Each fitted state must preserve its seed, variant, `C`, ordered feature names,
class order, intercepts and coefficients, fitting/convergence diagnostics, and
parent identity. States use auditable numeric serialization rather than pickle.
The standalone `multimodal/geo_helpfulness_router_candidate_state.py` module
has no scikit-learn or fitting imports. Its `read_candidate_state` loads and
validates a state; `predict_router_probabilities(full727, state)` applies its
stored ordered mask, coefficients, intercept, and four-state class order.
`decision_function` and `variant_feature_indices` expose the corresponding
logit and column-selection operations. The later label-blind stage can use
these functions without calling a fitting API. It must first construct the
full 727-column validation matrix with the existing OOF-fitted temperatures
and transforms unchanged.

For the later label-blind phase, prefer
`load_candidate_set_for_inference(root, expected_dataset_manifest_sha256=...,
expected_study_spec_sha256=...)`. The dataset hash is required; the expected
study hash is optional but recommended. This returns `(manifest, states)`,
where `states` maps all 48 filenames to verified states. It checks the complete
grid, file hashes, state hashes, and parent identities by opening only the
candidate directory. Do not use the training CLI's `validate` command or its
full parent reader as a label-blind inference loader: those intentionally run
the broader development-data integrity preflight.

For each candidate, apply the matrix belonging to that candidate's seed:

```python
q = predict_router_probabilities(matrix_by_seed[state["training_seed"]], state)
score = q[:, 0] - q[:, 1]  # q_rescue - q_harm; no threshold selection here
```

All four output columns retain the declared target order. A validation
prediction artifact must retain row identities and the candidate's variant,
seed, and `C`, and be sealed before the separate label-scoring stage. This
example specifies the next-stage interface; no validation predictions are
generated by the present training workflow.

The bundle directory is
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/router/router_candidates_v2/`.
It contains:

- `study_spec.json`: the immutable predeclared study plus resolved parent,
  feature, and implementation identities, published before any fitting.
- Forty-eight `<variant>__seed_<s>__C_<token>.json` fitted states, with `C`
  tokens `0p01`, `0p1`, `1`, and `10`.
- `training_report.json`: deterministic training-only diagnostics, not
  validation performance or a candidate-selection report.
- `manifest.json`: the complete-set commit marker, published last.

Publication is immutable, locked, and manifest-last. Each candidate is
published atomically so successfully completed candidates survive interrupted
builds. Existing candidate states are validated and reused, never overwritten
or deleted. Completing this subtask establishes 48 trained candidates, not a
selected router or evidence of validation benefit.

## Historical v1 verification and first real-build result

On 2026-09-08 the real public parent reader passed, producing four finite
`3378 x 727` matrices and retaining 47 parent/config/code file fingerprints.
The standalone preparation check took 79.04 seconds.

The first real `build` used the documented `habcls` environment, with Conda's
`--no-capture-output` option to expose progress. It exited with status 1 after
80.46 seconds, including the full parent preflight. These primary seed-1
candidates converged and were atomically committed:

| C | Iterations | Fit seconds | Training four-state NLL |
|---:|---:|---:|---:|
| 0.01 | 82 | 0.239 | 0.5758308798 |
| 0.1 | 212 | 0.393 | 0.5179941679 |
| 1.0 | 546 | 1.008 | 0.4812648069 |

The next candidate, primary seed 1 with `C=10`, raised `ConvergenceWarning`:
`STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.` Its frozen iteration cap was
1,000. The workflow rejected that fit; it did not save an unconverged model,
relax tolerance, increase the cap, or omit this candidate from the grid.
At that point, later seeds and both ablations had not been fitted on real data.
The NLL values above are in-sample diagnostics, not selection evidence.

The preserved v1 candidate directory contains exactly `study_spec.json` and the
three successful state files, all mode `0444`. Each state reloads and validates.
There is no `training_report.json` or complete-set `manifest.json`; therefore
no completed candidate bundle or next-stage prediction run is authorized by
these partial outputs. The study payload hash is
`26f0d541bf51ef46539ff8b80bd315e7e4fcf67687c6c2cc449ca3ae572f7453`.
All 18 directly snapshotted pre-existing configuration, M3/M4 dataset
artifacts, and geographic-helpfulness source files remain byte-identical;
the real parent preflight additionally validated inherited M1–M3 lineage.

Historical regression results: 232 existing M1–M4 dataset tests and 157 new
candidate tests passed (389 distinct tests). Candidate tests cover numerical
fitting and portable state, seed/target input assembly, CLI behavior,
immutable publication, interrupted resume, tampering, lock contention,
apply-only reuse, and explicit coefficient/probability reproduction. Synthetic
reproduction and reuse checks passed; at that time their real-data
counterparts remained pending because the v1 set was incomplete.

```bash
conda run -n habcls python -m pytest -q tests/test_multimodal_geo_router_candidate_state.py tests/test_multimodal_geo_router_candidate_fit.py tests/test_multimodal_geo_router_candidate_cli.py tests/test_multimodal_geo_router_candidates_inputs.py tests/test_multimodal_geo_router_candidates_artifacts.py
```

This historical failure motivated the explicitly approved v2 amendment above.
The amended runner defaults to a new study and refits all 48 candidates; it
does not retry or modify the preserved partial v1 study. No validation results
were used to motivate the amendment.

## V2 completion evidence

The amended real build succeeded on 2026-09-08, fitting all 48 candidates
afresh and publishing 51 regular, nonsymlink, read-only (`0444`) files. The
complete build, including full parent preflight and internal reconstruction,
took 102.72 seconds. Summed estimator-fitting time was 21.29 seconds:

| Variant | Candidates | Iteration range | Summed fit seconds |
|---|---:|---:|---:|
| `primary_full727` | 16 | 82–1,437 | 16.62 |
| `ablation_no_pairs79` | 16 | 76–1,373 | 3.27 |
| `ablation_numeric25` | 16 | 67–1,077 | 1.40 |

Every fit converged below the 5,000-iteration cap, with no convergence warning.
The expected scikit-learn 1.5.2 `multi_class` deprecation warning was recorded
as an implementation diagnostic, not treated as convergence failure. Saved
portable probabilities matched the fitting estimator exactly in all 48
checks (maximum absolute difference `0.0`). The prior primary seed-1 `C=10`
failure converged in 1,437 iterations under the approved budget.

An independent fresh-process inference-only audit loaded all 48 states with
the expected dataset parent identity. It confirmed the complete 3-variant ×
4-seed × 4-`C` grid, model schema v2, all four implementation-file hashes,
and absence of sklearn, SciPy, Torch, pandas, or PyArrow imports. The three
previously converged v1 states and their newly fitted v2 counterparts have
exactly equal coefficients and intercepts (maximum difference `0.0`) and
identical iteration counts. A before/after comparison of 23 original source,
configuration, dataset, and partial-v1 files found no changed bytes.

Authoritative identities:

- Candidate manifest payload SHA256:
  `e0c7f2b20939fbb72fc0295db2afda1615695e1f259f6a35cdc12825356f8b71`.
- Candidate manifest physical-file SHA256:
  `c26e87f5719b3df88a0d16e7eba7338faa74ade0dc85021455dad980aded8b56`.
- Amended study payload SHA256:
  `6f8936797121268298f0a525952f3641c6a171697647b6498052d23b944dcbc9`.
- Dataset parent manifest payload SHA256:
  `ca836471d62cca2079d78f8dfd8f50ecab46d884aa4e67ffc8993e4fa8d027ba`.

The recorded environment is Python 3.9.20, NumPy 2.0.2, SciPy 1.13.1,
scikit-learn 1.5.2, and threadpoolctl 3.5.0. All 172 candidate tests passed;
232 unchanged M1–M4 dataset tests passed in isolated per-file processes
(404 distinct focused tests total).

The first combined parent-regression run and first standalone real `validate`
attempt encountered the previously observed native-runtime crash. The
regression crash trace was in unchanged M2/protocol canonical-JSON hashing;
all eight isolated parent test files subsequently passed. No frozen code was
edited to mask this issue. Full real reconstruction was retried with
`PYTHONFAULTHANDLER=1 PYTHONMALLOC=malloc`, which changes runtime diagnostics
and allocation only, not model mathematics or serialized states. It passed
full independent validation in 81.91 seconds with `fits_this_call=0`. The same
runtime prefix was used for explicit reproduction, which refitted all 48
candidates in 107.46 seconds including parent checks. Maximum absolute
coefficient/intercept and probability errors were both `0.0`, and every saved
artifact was unchanged. This is a runtime diagnostic workaround, not an
established fix for the underlying
intermittent crash.

The ordinary repeat build, without the diagnostic environment prefix, passed
in 78.72 seconds with `status=reused_valid`, `fits_this_call=0`, and the same
manifest hash. A separate comparison of all 51 saved file hashes confirmed
byte-preserving validation, reproduction, and reuse. The 23 preserved parent
and historical-v1 files also remained unchanged at the final check.

| Real acceptance command | Outcome | Seconds |
|---|---|---:|
| First amended `build` | 48 fits; complete immutable bundle | 102.72 |
| Independent `validate` with diagnostic runtime prefix | Valid; 0 fits | 81.91 |
| Explicit `reproduce` with diagnostic runtime prefix | 48 refits; coefficient/probability errors both 0.0 | 107.46 |
| Ordinary repeat `build` | Reused valid; 0 fits; unchanged bytes | 78.72 |

The final candidate regression rerun again passed all 172 tests. Together
with the isolated 232 parent regressions, 404 focused tests passed. Compilation
and diff-whitespace checks passed. No validation predictions or validation/test
scores were generated; regularization and threshold selections remain null
in the report.

The next M4 step is to construct seed-specific validation features using the
already-frozen expert temperatures and feature transforms, load the complete
candidate set through the label-blind reader, and seal all candidate
probabilities/scores before a separate scoring-only phase opens labels. Do
not infer validation benefit or choose a candidate from the training report.

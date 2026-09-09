# M4 — Sealed validation predictions and a fixed accuracy preview

**Status:** Complete on 2026-09-08. The fixed preview is 585/822 correct,
or **71.17% four-state validation accuracy**. Model/threshold selection and
the remaining M4 feasibility evaluation are still outstanding.

## Scope

This subtask prepares the pre-label prerequisites and scores one example:
`primary_full727`, training seed 1, `C=0.01`. This candidate and the metric
**four-state accuracy only** were fixed before opening validation labels.
The preview does not select regularization, a router intervention threshold,
an ablation, or a best seed. It does not complete M4 feasibility evaluation.

The 48 converged v2 candidates, twelve temperatures, four feature transforms,
and M1–M4 parent artifacts are consumed unchanged. No expert/router training,
temperature fitting, or transform fitting takes place. Test sources are out
of scope.

## Reproducible commands

From the repository root, use the same commands for manual or agent execution:

```bash
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_validation.py prepare
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_validation.py validate-predictions
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_validation.py prepare
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_preview.py declare
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_preview.py score
```

`prepare` first selects the two predeclared heuristic thresholds from training
OOF data, seals their specifications and the fixed preview declaration, and
then opens the label-blind M2 validation outputs. It applies frozen expert
temperatures, the unchanged M3 semantic builder, the seed-local transforms,
and all 48 frozen routers. The complete prediction manifest is written last.

`validate-predictions` reconstructs the predictions with apply-only numerical
APIs and compares them at `atol=rtol=1e-12`. Existing OOF thresholds are checked
by applying their saved specifications, not reselected. Repeat `prepare`
performs the same checks and preserves committed bytes. No overwrite option
is exposed. Unknown files and stale/corrupt committed artifacts cause hard
failure; an exclusive lock prevents concurrent publication. Individual files
are atomically published without replacing existing files. A partial bundle
can resume using its unchanged published files; unrecognized temporary files
are not automatically removed.

The versioned preview runner uses a separate, fitting-incapable import graph.
`declare` seals the implementation-only correction described below; `score`
authenticates
the complete seven-file prediction bundle before reading the assignment
label column, constructs four-state ground truth for the fixed candidate,
and computes only its correct count and accuracy. Its result is saved outside
the immutable prediction directory. Repeating it preserves identical bytes.

All commands accept `--artifact-root` for an equivalent relocated parent tree.
There are no scientific-setting, candidate, metric, or threshold overrides.

## Artifacts

The prediction directory is `router/router_validation_v1/` beneath the
existing protocol root.

| File | Contents |
|---|---|
| `preview_declaration.json` | Fixed candidate/metric, parent and implementation identities, access order, and OOF-heuristic identity |
| `oof_heuristics.json` | Frozen JSD and margin-difference thresholds, training-OOF diagnostics, fixed comparator declarations, and selection-input fingerprint |
| `primary_full727.parquet` | 13,152 prediction rows for the primary's 16 seed/C combinations |
| `ablation_no_pairs79.parquet` | 13,152 prediction rows for the 79-column ablation |
| `ablation_numeric25.parquet` | 13,152 prediction rows for the numeric-only ablation |
| `expert_predictions.parquet` | 3,288 label-blind expert prediction/identity rows |
| `manifest.json` | Commit marker sealing all six children, lineage, code/environment, dimensions, and numerical/access contracts |

Each router table has exactly these non-null Arrow columns:

```text
schema_version, protocol_id, row_uid, plot_idx, training_seed, C_hex,
q_rescue, q_harm, q_both_correct, q_both_wrong, score
```

`training_seed` is int8; the five numerical output columns are float64;
the remaining fields are strings. Rows are sorted by `(C_hex, training_seed,
row_uid)`. `score = q_rescue - q_harm`; no policy threshold is applied here.
Each seed contributes 822 images from 325 validation plots at each C.

The expert table contains non-null identity/protocol columns followed by
int8 `image_pred`, `geo_pred`, and `raw_pred`, sorted by `(row_uid,
training_seed)`. Neither table contains labels, correctness indicators,
four-state targets, or per-row losses. Parquet uses Zstandard compression.

The preview result is saved as `router/router_validation_preview_v2.json`.
It records the fixed candidate, target order, correct/total counts, accuracy,
prediction-manifest identity, assignment identity, and its own content hash.
The additional `router/router_validation_preview_v2_declaration.json` binds
the two new scorer/runner source files and the unchanged prediction manifest.

## Scorer v2: storage-dtype compatibility correction

The first v1 scoring attempt stopped before computing any accuracy: pandas
`DataFrame.equals` treated the assignment's `string` extension dtype and the
prediction table's `object` dtype as unequal. An identity-only check confirmed
all 822 `row_uid` and `plot_idx` values match exactly; there were no missing,
reordered, or different identities. This is a storage-representation mismatch,
not a data-join mismatch.

The sealed v1 prediction bundle and all five source files bound by its manifest
are preserved unchanged. The additive v2 scorer reuses the v1 complete-bundle
validator and replaces only the assignment/prediction identity comparison
with exact per-column value equality, independent of pandas string storage.
Nulls, wrong identities, counts, labels outside the ontology, and parent/hash
changes remain hard failures. A separate immutable correction declaration is
published before the corrected scorer opens labels. The same primary/seed-1/
`C=0.01` candidate and accuracy-only metric remain fixed. No model, probability,
threshold, or scientific choice changes. The old v1 `preview` command is
retained for provenance but is not the working scoring command for these real
artifacts.

## Access boundary and interpretation

The prediction-input reader verifies physical hashes, manifest self-hashes,
M2 output logical hashes, unchanged code/config identities, seed/plot coverage,
and aggregate/producer equality. It authenticates assignment bytes for parent
integrity but does not parse assignment columns. It does not call full parent
validators that read labels or reconstruct fitting diagnostics. Its scope is
**sealed prediction-input integrity**, not a new expert-checkpoint replay.
One unchanged legacy regression test parses the full development assignment
table to assert split/fold counts; it does not use label values for fitting or
scoring. This ancillary test access is distinct from the new label-blind
prediction builder. The failed v1 preview likewise parsed labels but computed
no metric before its identity-representation check rejected the join.

The scorer has no SciPy, scikit-learn, Torch, fitting, or parent-pipeline
imports. Only after validating all 48 prediction sets and their prerequisites
does it read the development assignment projection and filter the validation
rows. Raw/geo expert correctness then defines rescue, harm, both-correct, and
both-wrong targets. Training rows in the shared assignment table are not scored.

Validation accuracy here is a single prespecified preview. It is not the
18-class habitat accuracy, an intervention-gain estimate, a C-selection
criterion, or evidence that the complete router policy passes the M4 gate.
The subsequent declared selection still uses mean four-state validation log
loss across all four seeds, followed by the frozen utility/guardrail rule.

## Execution evidence

Real `prepare` returned `built`, independent `validate-predictions` returned
`validated`, and repeat `prepare` returned `reused_valid`. All seven prediction
bundle files retained identical SHA256 hashes. Reconstruction used zero fits
and did not parse assignment content; all 91 bound parent/implementation files
remained unchanged.

The sealed prediction manifest payload SHA256 is
`ca366f02cde61bea92eba24cb2aacd140ba8c00167138490c20a11a542f600b8`
(physical file SHA256
`8070d7b3e413824d7e671c434eb1088aa55d246f1c54d97cd607484fbf3913ed`).

The OOF JSD heuristic had no feasible candidate and froze `never_intervene`.
The margin-difference heuristic froze strict `score >
0x1.ccacfcd44b34ap-2` (decimal 0.44987864537943223), additionally requiring
geo/raw disagreement. Its training-OOF diagnostics are 235 actions, 127
rescues, 55 harms, and 109 unique acted plots across all 13,512 realizations.
These are threshold-fitting diagnostics, not validation results.

Before scoring, 539 focused tests passed in separate groups: 232 earlier
M1–M4 tests, 172 candidate-training tests, and 135 new prediction/heuristic/
preview tests. The additive v2 correction passes 25 further tests, including
the real string-storage mismatch reproduced synthetically and rejection of
genuinely different identities. **564 focused tests pass in total.**

The corrected scorer returned the same immutable result on two independent
invocations:

| Fixed candidate | Correct | Total | Four-state accuracy |
|---|---:|---:|---:|
| `primary_full727`, seed 1, `C=0.01` | 585 | 822 | 71.16788321167883% |

For reference, this same candidate's previously recorded in-sample router
training accuracy is 2526/3378 = 74.77797513321492%. No other candidate's
validation metric was computed, and no C or router policy threshold was
selected. The preview does not by itself establish beneficial geo overrides.

The result payload SHA256 is
`4233f8278d98411fd6e26b9f756e84226517a3d11737bc37723c599daba14356`
(physical file SHA256
`ed7e57480b520d1a2f8aee25ac8ba7eefde6e9c0fd5e55a241ff21113229b449`).
The correction declaration payload SHA256 is
`b983fc17ff6e1d7a025585f5f5267f3ef6494468b3d2254c9178723cd62fb1d2`.
The result and correction declaration retained identical bytes on reuse;
all seven original prediction files and their frozen source/parent hashes
remain unchanged.

The real preparation/reconstruction/reuse commands were run with
`PYTHONFAULTHANDLER=1 PYTHONMALLOC=malloc` and `conda run --no-capture-output`,
consistent with the previously documented native-runtime precaution. All
three completed without a runtime error. Environment: Python 3.9.20,
NumPy 2.0.2, pandas 2.2.3, PyArrow 21.0.0, SciPy 1.13.1, threadpoolctl 3.5.0.

# M4 — Router selection, policy evaluation, and frozen-state delivery

## Pre-scoring implementation declaration

The user authorized completion of M4 after the 48 primary/ablation candidates,
their validation predictions, and the OOF-only heuristic thresholds were sealed.
The fixed primary/seed-1/C=0.01 preview (585/822 four-state accuracy) is already
disclosed. It does not change the primary feature set, C grid, training,
threshold objective, comparator set, or gate. All remaining scientific rules
come from the unchanged protocol and candidate-study records.

This record and `configs/multimodal_geo_router_m4_evaluation_v1.json` fix the
remaining reporting conventions before full validation metric scoring:

- Select one common C per variant by arithmetic mean four-seed log loss;
  exact float64 ties choose the smaller C.
- Log loss uses natural logarithms and floors only the true-class probability
  at the smallest positive normal float64 value; report the floor count.
  Brier score is mean summed squared error, without a one-half factor.
- Rescue/harm PR-AUC means noninterpolated average precision with exact-score
  ties grouped together. No positive target gives an undefined/null value.
- Expert calibration reports native and calibrated NLL, Brier, accuracy,
  and 15-bin equal-width ECE. Bins are left-closed/right-open, with the final
  bin including one. Empty-bin statistics are null.
- Habitat macro F1 uses all 18 frozen classes, with absent-class F1 zero.
  An undefined MCC or rate is null, not an invented zero. Top-k ties use
  ascending dense class ID.
- Override precision is rescues/actions; rescue recall is rescues/all rescue
  opportunities; harmful-override rate is harms/actions; coverage is
  actions/all rows. Counts and denominators are reported explicitly.
- Plot-bootstrap intervals use the fixed seed 20260826 and 2,000 paired
  complete-plot resamples, stratified by canonical habitat. All images and
  all four seed realizations of a sampled plot travel together. The same
  multiplicities are used for every policy. Percentile 95% intervals use
  linear quantiles, require at least 95% valid replicates, and do not repeat
  selection. They are descriptive and selection-conditional.
- Report subgroups by habitat, plot, geo/raw predicted-class pair, and
  image/geo predicted-class pair. Flag fewer than 20 unique images or ten
  unique plots. Report training-seed variation separately with population SD.
- Apply the frozen eight-part go gate, two no-go conditions, and otherwise
  inconclusive rule exactly. A completed no-go/inconclusive result still
  delivers M4; it does not authorize M5/M6 or a new primary model.

## Reproducible workflow

```bash
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_finalize.py build
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_finalize.py validate
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_finalize.py build
```

The outer `build` command launches separate `prepare` and `score` processes.
The preparation process applies unchanged temperatures to sealed M2 logits and
exports calibrated/native probabilities plus semantic features without parsing
assignment columns or fitting any state. The scoring process has no fitting,
Torch, SciPy, or scikit-learn imports; it authenticates the complete inputs and
declaration before reading validation labels.

The first scoring run selects C and thresholds, creates all reports, and
seals the frozen router state and selection receipt. Repeat builds and
`validate` apply the saved selections to the sealed scoring-target table;
they neither reopen assignment label columns nor reselect C/thresholds.
There is no force-overwrite or scientific-setting override. Original M1–M4
parents, the candidate/prediction bundles, and the preview remain unchanged.

The finalizer is the authoritative entry point. The original runner/module
remain byte-identical to their pre-scoring declaration. On the first real run,
all 31 pre-receipt artifacts were produced, but the receipt writer looked up
`artifact_contract.development_selection_receipt`; the frozen configuration
actually stores this schema at `schemas.development_selection_receipt`.
Consequently no receipt or final manifest was published by that attempt.

The additive v2 finalizer corrects only this schema lookup. Its separate
`router/m4_receipt_correction_v2.json` declaration binds both new source files,
all original parents, and the 31 preexisting artifact hashes. Existing C and
threshold selections are reused, every report is reconstructed, and byte-equal
no-overwrite writes preserve the original results. Before the first receipt it
reauthenticates targets against assignments; after the receipt it only reads the
sealed scoring-target copy. The corrected receipt also binds the correction
declaration and code. No scientific setting or selection changed.

For the recorded runs, these environment settings were added to the commands:

```bash
env PYTHONFAULTHANDLER=1 PYTHONMALLOC=malloc \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n habcls python \
  tools/run_multimodal_geo_helpfulness_m4_finalize.py build
```

The allocator setting retains the earlier native-runtime precaution; the
thread limits bound CPU parallelism. Neither changes a scientific setting.
`--artifact-root` overrides the protocol artifact directory. There are no
training, C-grid, threshold, force-overwrite, or best-seed options.

## Results

The primary feasibility decision is **no-go**, with common `C=0.1` and the
typed `never_intervene` threshold. A completed no-go is a valid M4 outcome;
it does not authorize selective correction or independent-test evaluation.

Mean four-seed validation log losses, in frozen grid order:

| Variant | C=0.01 | C=0.1 | C=1 | C=10 | Selected C |
|---|---:|---:|---:|---:|---:|
| Primary, 727 features | 0.633119 | 0.609177 | 0.627948 | 0.685165 | 0.1 |
| No-pairs ablation, 79 features | 0.635809 | 0.606749 | 0.611121 | 0.632073 | 0.1 |
| Numeric ablation, 25 features | 0.644270 | 0.615505 | 0.607260 | 0.605925 | 10 |

All counts below pool four seed realizations of the same 822 validation images
and 325 plots; they are not 3,288 independent images. Accuracy gains are mean
seed percentage points relative to raw fusion.

| Policy | Actions | Rescues | Harms | Mean seed net | Gain (pp) | Coverage | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Primary | 0 | 0 | 0 | 0 | 0 | 0% | no-go |
| No-pairs ablation | 38 | 23 | 11 | 3 | +0.364964 | 1.155718% | inconclusive |
| Numeric ablation | 55 | 34 | 14 | 5 | +0.608273 | 1.672749% | inconclusive |
| Never intervene | 0 | 0 | 0 | 0 | 0 | 0% | comparator |
| Always geo on disagreement | 1,247 | 344 | 666 | -80.5 | -9.793187 | 37.925791% | comparator |
| Higher calibrated geo confidence | 367 | 131 | 146 | -3.75 | -0.456204 | 11.161800% | comparator |
| OOF-selected image/geo JSD | 0 | 0 | 0 | 0 | 0 | 0% | comparator |
| OOF-selected geo/raw margin difference | 78 | 44 | 22 | 5.5 | +0.669100 | 2.372263% | comparator |

The primary threshold search examined 919 candidates and found none feasible.
Its seven finite, positive-net candidates meeting the harm-ratio constraint
act on at most 18/3,288 rows (0.547445%) and 14 plots, failing both minimum
coverage (1%) and plot support (20). The descriptive maximum-net point in the
saved curve has 74 rescues and 46 harms, violating `harms <= rescues / 2`.
This is an explanation of the complete saved curve, not a new policy choice.

The no-pairs threshold is `0x1.c1c88c1faea92p-2` (approximately 0.43924159);
its seed nets are `[0, 7, 5, 0]`. It fails the three-positive-seeds and
all-heuristics-superiority criteria. The numeric threshold is
`0x1.b3629268d1e50p-2` (approximately 0.42518071), with seed nets `[4, 8, 4, 4]`;
it fails superiority to the frozen margin comparator. Neither ablation is
promoted. The OOF margin threshold remains `0x1.ccacfcd44b34ap-2`, with seed
nets `[2, 5, 7, 8]`; it was not selected on validation.

The selected primary's mean four-state accuracy is 71.228710%, log loss
0.609177, Brier score 0.371881, rescue AP 0.412054, and harm AP 0.703513.
These are router-state metrics, not final habitat accuracy. Raw-fusion habitat
accuracy, exactly retained by the primary fallback, is 2,336/3,288 = 71.046229%
(seed accuracies 69.951338%, 70.316302%, 71.654501%, 72.262774%). Mean raw top-3
accuracy is 91.788321%, macro F1 0.599558, weighted F1 0.706695, and MCC 0.668427.

The 95% paired-plot intervals for accuracy gain are [-0.062934, 0.845411] pp
for no-pairs, [0.062029, 1.158679] pp for numeric, and [-0.092062, 1.448972] pp
for the margin heuristic. All 2,000 replicates are valid. The primary interval
is identically [0, 0] because its selected action is never-intervene; this is
not evidence that alternative learned policies have no uncertainty. Intervals
are conditional on selection and do not provide independent confirmation.

Expert mean native-to-calibrated validation NLL is 1.855967 to 1.037102 (image),
1.510519 to 1.148861 (geo), and 1.682021 to 0.960538 (raw). Corresponding ECE
falls from 0.236751 to 0.064430, 0.201986 to 0.082087, and 0.215573 to 0.062037.
Temperatures were frozen from training OOF rows; class predictions are unchanged.

Support is limited: 17 of 18 habitats occur on validation; nine observed
habitats are low-support, and dense class 12 is absent. Of observed predicted
pairs, 87/103 geo/raw and 102/119 image/geo groups are low-support. Subgroup
habitat IDs are canonical IDs; classification-report class IDs are dense IDs.
All individual-plot subgroups are necessarily flagged low-support.

## Artifact contract

Outputs are under the existing protocol's `router/` directory. The label-blind
`m4_comparator_inputs_v1/` contains a declaration, a manifest, and
`expert_probabilities_and_features.parquet`: 3,288 rows by 41 columns (five
identities, six length-18 native/calibrated float64 vectors, and 30 semantic
features). It contains no labels, correctness flags, or per-row losses.

`m4_feasibility_v1/` contains 33 files after receipt publication:

| Artifact | Contract |
|---|---|
| `build_declaration.json` | Parent snapshot and ordered target fingerprint, published before target Parquet |
| `validation_targets.parquet` | 3,288 × 14; identity, ontology, expert predictions, four-state truth, predicted pairs; scoring-only, never model input |
| `selection.json` | Three common-C decisions, all grid losses, exact threshold specifications and complete curves |
| `candidate_metrics.json` | All 48 candidates' four-state losses, Brier, accuracy, AP, confusion and seed summaries |
| `expert_metrics.json` | Native/calibrated expert diagnostics by seed and mean |
| `policy_metrics.json` | Eight fixed/selected policies, counts, rates, denominators and seed summaries |
| `habitat_classification.json` | Experts/policies: 18-class confusion, top-1/top-3, F1/MCC, per-class support and TP/FP flows |
| `subgroup_metrics.parquet` | 22,560 × 28; four grouping families, eight policies, per-seed and pooled metrics, low-support flags |
| `validation_actions.parquet` | 26,304 × 19; every policy/seed/image score, threshold, action, prediction and observed outcome |
| `bootstrap_summary.json` | Fixed-policy intervals and paired primary-minus-comparator differences |
| `bootstrap_replicates.parquet` | 16,000 × 16; 2,000 replicates × eight policies |
| `access_audit.json`, `go_no_go.json`, `report.md` | Access evidence/disclosures, exact gate decisions, human-readable results |
| `router_frozen_state.json` | Four primary states plus separately labelled ablations, feature/calibration components, exact policies and hashes |
| Four feature/calibration JSON files | Byte-identical copies of temperatures, transform, feature schema, OOF heuristic specifications |
| `models/*.json` | Twelve selected router files, byte-identical to their candidate parents |
| `development_selection_receipt.json`, `manifest.json` | Irreversible selection boundary, complete artifact/parent/code hashes; manifest last |

Parquet uses Zstandard compression. Target and action schemas are explicit;
undefined statistical rates and confidence intervals are represented by Arrow
nulls/JSON null, not nonstandard NaN/Infinity. Reproduction uses exact identity,
integer, Boolean and JSON comparisons; Parquet float values use
`atol=rtol=1e-12`. Fixed probabilities retain the earlier normalization tolerance.

## Access and interpretation boundaries

The fit-capable preparation adapter opens no assignment columns and calls no
fit API. It only applies frozen temperatures, builds semantic features, and
checks label-blind expert/model lineage. The scoring process has no fitting
libraries and initially reads the assignment projection solely to construct
validation truth. Because the source stores both development roles in one
table, that initial projection parses train and validation label cells; only
the validation-filtered rows enter these scores. It does not fit any parameter.

After the receipt, validation/reuse reads the sealed scoring-target copy and
only hashes original assignment bytes. Code hashes, explicit reader allowlists,
import guards, and tests support the access audit; it is not an OS-wide trace.
Legacy full-assignment integrity checks, the fixed preview, the earlier
storage-dtype correction, historical test-informed motivation, and this
receipt-path correction are disclosed. No cleaned-test source is opened by
this workflow. No claim of independent confirmation is made.

## Completion evidence

Completed on 2026-09-08:

- Corrected `build`: exit 0, comparator `reused_valid`, feasibility `built`.
- Independent `validate`: exit 0, `validated`, no assignment-label columns
  opened, zero reselection calls.
- Repeat `build`: exit 0, both phases `reused_valid`.
- A further fresh-process validation and reuse both passed with `_create_targets`,
  `_make_selection`, `select_common_c`, and `select_common_threshold` replaced
  by hard failures, and with all original-assignment Parquet parsing forbidden.
  The import guard reported no Torch/SciPy/scikit-learn modules.
- The 31 original pre-correction result files retained identical bytes and
  nanosecond modification times. All 37 final files (33 results, three
  comparator files and the correction declaration) retained identical bytes
  and modification times across independent validation and repeat build.
- All 100 separately snapshotted frozen-parent files remained SHA-256 identical.
  The final receipt binds 110 parent/code/declaration files and all 31
  pre-receipt result artifacts. No frozen-parent files were edited or removed.
- **751 focused tests passed**: 232 M1–M4 dataset, 172 candidate, 160
  validation/heuristic/preview, 110 numerical metrics/selection/uncertainty,
  34 comparator/publication, 33 original scoring lifecycle, and ten additive
  finalizer tests. The latter use the actual frozen configuration structure
  and cover self-consistent mid-run code/declaration drift. An initial combined
  regression run was manually interrupted while still progressing, with no
  reported failure; the complete 564-case earlier suite was subsequently run
  to successful completion in bounded groups.
- Python compilation and whitespace checks passed. Runtime: Python 3.9.20,
  NumPy 2.0.2, pandas 2.2.3, PyArrow 21.0.0 and PyYAML 6.0.2.

New scoring tests can be reproduced with:

```bash
env PYTHONMALLOC=malloc OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n habcls python -m pytest -q \
  tests/test_multimodal_geo_router_metrics.py \
  tests/test_multimodal_geo_router_selection.py \
  tests/test_multimodal_geo_router_uncertainty.py \
  tests/test_multimodal_geo_m4_inputs.py \
  tests/test_multimodal_geo_m4_analysis.py \
  tests/test_multimodal_geo_m4_finalize.py
```

This is 187 new cases; the earlier 564 cases are the frozen milestone,
dataset, candidate, validation, heuristic and preview regression groups.
The exact command above also passed as one final fresh-process run:
187 passed in 66.09 seconds.

Canonical payload fingerprints (distinct from physical file-byte hashes):

| Record | SHA-256 |
|---|---|
| Comparator manifest | `14eeb2d332364c92ff8e4938bd4f2fde25c7b0765aeda854057d299c8a3f1c82` |
| Selection | `0189d669f846aba4819c4462008a8f4bc5d571da183d491f2fac477ab57b4bab` |
| Frozen router bundle | `00dc7e76774cc926edca99dfb964457a3cf32072f3b887e47bf23151afbd6cdd` |
| Receipt correction declaration | `9eccb8991fb77f9675afc4c23c50a2bdaac671fc51e1c787e87394b55d09373d` |
| Development selection receipt | `3146810b63f08114347e411915ca1646c60ac054dc11fb9d1e6b618328940419` |
| Final manifest | `442d8bc6dafe992ea6e3e7a8793df51571ece4643c2a7d0ce4a25b20b6139615` |

The receipt's physical file SHA-256 is
`4273af194bf940a28682f3a49df9a942f3e2abfeb05d103e368f9d3a5801678a`;
the manifest's is
`6314752d9a34a934f96ec3b397b3816a07224553f38fe560b6ead74c70138728`.

M4 is complete with its primary no-go result. Historical dataset, candidate
and preview records describe their own earlier handoff boundaries; this record
supersedes their statements that full M4 remains outstanding. M5/M6 and any
revised scientific protocol remain outside this delivery.

# M4 — Calibrated Router-Dataset Preparation

## Status and completion boundary

**Status:** Dataset preparation complete, sealed, and independently validated
on 2026-09-08. Remaining M4 router work is unfinished.

This record covers only M4 dataset preparation: seed-specific expert
temperature fitting, calibrated probabilities, the 30 semantic features,
seed-specific numeric transforms, and the materialized 727-feature training
matrix. Router fitting, regularization selection, threshold selection,
validation scoring, final-development fitting, and locked-test inference are
outside this subtask and remain unfinished.

Completion requires a real build, an independent validation command, and a
second build that reuses the identical committed bundle without refitting.
Synthetic tests alone do not establish completion.

The workflow is additive and CPU-only. Frozen M1–M3 code, configuration, and
artifacts remain unchanged. The preparation implementation is separate from
subsequent router-training code so that extending router training does not
change the preparation implementation fingerprint.

The new implementation is split between
`multimodal/geo_helpfulness_router_numeric.py` (pure fit/apply functions and
JSON-state validation) and `multimodal/geo_helpfulness_router_dataset.py`
(validated inputs, assembly, publication, and loading). The dedicated runner
is `tools/run_multimodal_geo_helpfulness_m4_dataset.py`. All three files are
bound by the M4 manifest; future router training should use separate files.

## Reproduction commands

Run from the repository root using the existing `habcls` environment:

```bash
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_dataset.py build
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_dataset.py validate
conda run -n habcls python tools/run_multimodal_geo_helpfulness_m4_dataset.py build
```

The expected statuses are `created`, `reused_valid`, and `reused_valid`.
The same commands are used for manual and agent-driven execution. Both
subcommands accept only `--config`, `--protocol-dir`, and `--artifact-root`
path overrides. No calibration-setting, validation-data, test-data, or
force-overwrite flag is exposed. Success summaries are JSON on standard output;
workflow failures are JSON on standard error with exit status 1. Invalid CLI
arguments use the standard argument-parser error and exit status 2.

`validate` loads fitted states and reconstructs the stored outputs without
optimizing temperatures. A valid repeated `build` also performs no fitting and
does not rewrite committed bytes.

Programmatic consumers use `load_validated_router_dataset_bundle(...)` to
obtain Arrow tables and the frozen states. Use the manifest's ordered
`feature_columns` to select the 727 model inputs and select one
`training_seed` at a time. The pure APIs are `fit_expert_temperature`,
`apply_expert_temperature`, `fit_router_feature_transform`, and
`transform_router_features`; only the two apply functions belong in future
validation/inference feature construction.

## Inputs and access boundary

The public M3 validated-bundle reader establishes M1 integrity, all 16
development-train OOF producers, the sealed M2 aggregate, and the M3 bundle.
M4 uses M2's public output reader for authoritative float64 logits and joins
the role-filtered training assignment projection by `row_uid`. It checks
identity, plot and fold agreement, labels in the dense 18-class ontology, all
four seeds, and exact agreement with M3 targets.

Each seed contributes 3,378 images from 1,300 plots across four OOF folds.
The combined dataset retains all 13,512 seed–image realizations. They are not
13,512 independent biological samples. No seed averaging, pooling of
temperatures across seeds, or per-fold temperature fitting occurs.

The existing parent-integrity preflight can read full development assignments
and development source tables, including validation rows, for integrity
checking. Only the development-training projection reaches calibration and
transform fitting. M4 does not open validation expert outputs, final
in-sample expert outputs, or test sources. This distinction is recorded rather
than claiming that validation labels are never opened by inherited preflight.

Per-seed computations use canonical `row_uid` order before numerical
reductions. Published tables use `(row_uid, training_seed)` order and unique
keys. Parent hashes are checked before computation and before publication.

## Calibration and feature construction

Fit exactly one scalar temperature for every combination of seeds 1–4 and
`image_only`, `geo_only`, or `raw_concat`: twelve fitted temperatures total.
For each combination, minimize equal-image-weight mean multiclass NLL over all
3,378 OOF logits and ground-truth labels. The four-state router target is not
the expert's 18-way calibration label.

The frozen optimization contract is log-temperature parameterization
`T = exp(theta)`, bounded SciPy scalar minimization on `[-5, 5]`, absolute
tolerance `1e-10`, and at most 500 iterations. NLL uses centered float64 logits
and stable log-sum-exp; probabilities use the authoritative logits divided by
the fitted temperature. Native M2 probabilities are descriptive-only inputs
to neither calibration nor the feature builder.

Reject optimizer failures, nonfinite results, NLL worsening beyond `1e-10`,
solutions within `1e-6` of either log-temperature bound, and the unidentifiable
case where every row has identical class logits. The last two checks are
explicit M4 implementation safeguards. Probability matrices must be finite,
float64, shaped `(3378, 18)`, within `[0, 1]`, and normalized within absolute
tolerance `1e-8`. Calibrated argmax must agree with logits and stored expert
predictions. Underflowed zeros are allowed; clipping is not.

The unchanged M3 semantic builder receives the three calibrated matrices with
`probability_basis="scalar_temperature_calibrated"`. It emits the exact frozen
30-feature contract. The matrix transform is fitted separately for each seed:

- Standardize the 25 Boolean/integer/numeric values in their frozen family
  order. Encode Booleans as 0/1, use population variance (`ddof=0`), and use
  scale 1 for exactly zero variance.
- Emit 702 dense one-hot columns with the fixed predicted-class and
  class-pair vocabularies, including unobserved categories.
- Concatenate the scaled numeric and one-hot blocks into 727 float64 columns.
  Reject missing, nonfinite, or out-of-vocabulary values.

Column names are `scaled__<feature>` and
`onehot__<feature>__<category_id>`. The saved transform contains the complete
ordered names and vocabularies. Separate fit and apply-only functions permit
later validation/inference to reuse the twelve temperatures and four
transforms unchanged. Native/calibrated NLL values are fitting diagnostics,
not held-out evidence of calibration performance. There is no router-output
calibrator in protocol v1.

## Artifact contract and publication

The five M4 files are siblings of the sealed M3 directory under the existing
protocol artifact root's `router/` directory:

| Artifact | Contract |
|---|---|
| `router_dataset.parquet` | Six unchanged M3 target/identity columns followed by 727 float64 model-input columns; 13,512 rows and 733 columns |
| `router_dataset_audit.parquet` | Five identity/protocol columns, OOF fold, three calibrated length-18 float64 probability vectors, and the 30 unchanged semantic features |
| `expert_temperatures.json` | Twelve fitted states, optimization settings/results, counts, fit-row identities, and native/calibrated NLL diagnostics |
| `router_feature_transform.json` | Four numeric-transform states, fixed categorical vocabularies, and ordered input/output columns |
| `router_dataset_manifest.json` | Commit marker; schema and artifact hashes, parent lineage, code/environment identity, dimensions, tolerances, and validation results |

The six leading training-table columns are `schema_version`, `protocol_id`,
`row_uid`, `plot_idx`, `training_seed`, and `target_state`. They are metadata
and the supervised target, never model inputs. Targets are joined only after
feature construction. The audit table omits ground-truth labels, correctness
flags, per-row losses, and target states.

Serialization uses explicit non-null Arrow schemas, Zstandard Parquet
compression, and round-trip float64 JSON values. No pickle or nonstandard JSON
NaN/Infinity values are used. Reconstructed probabilities, semantic features,
and matrix values are compared with `atol=rtol=1e-12`; probability normalization
retains M3's `1e-8` absolute tolerance.

Exclusive locking and an ownership receipt protect staged publication. Child
files are published with no-overwrite semantics and the manifest is published
last. Ownership covers only the five M4 filenames, not the M3 subdirectory or
future router artifacts. Committed corrupt/stale bundles fail without
overwrite. Recovery is permitted only for demonstrably owned, uncommitted
partial outputs. Unknown files are never removed.

Partial-output recovery additionally checks each existing child against the
file hash recorded in its ownership receipt before removal. Unrelated router
siblings remain untouched. Logical table hashes use the explicit ordered
schema and row count plus per-column hashes: canonical JSON for strings and
contiguous little-endian bytes for primitive numeric values and flattened
probability vectors. This is independent of Parquet compression and chunking.

## Native initialization diagnostic

The first two real build attempts exited with a native segmentation fault
before any M4 output was created. Fault-handler traces located the failure
inside the unchanged M2 canonical-JSON aggregate hashing path. The unchanged
M3 validator passed, and successful diagnostic hashes matched the sealed OOF
logical hash `989236e78dea2f4a8eb4a31d201e102c716c7f3d689afb2fd88acb262030d2bc`.

The new artifact module initializes the existing M2 reader before importing
the SciPy numerical layer, matching the working M3 initialization sequence.
However, further stress diagnostics also reproduced a crash during an
unchanged M2 import before any M4 import or artifact read. Therefore the
initialization order is **not a proven fix**. The underlying intermittent
interpreter/native-library instability remains unresolved. No frozen code,
hashing algorithm, data, optimizer, or allocator setting was changed.
Completion evidence below records the subsequent successful full CLI checks
rather than treating the smaller diagnostics as sufficient verification.

## Completion evidence

Observed on 2026-09-08:

- The successful official `build` returned `created`, `valid: true`, and
  `calibration_fitted_this_call: true`; no partial files needed recovery.
- The independent `validate` command returned `reused_valid`, `valid: true`,
  and `calibration_fitted_this_call: false` with the same manifest hash.
- A subsequent `build` returned `reused_valid` with no fitting. Independent
  before/after SHA256 and size comparisons confirmed that all five committed
  artifact files were byte-identical across validation and reuse.
- Dataset shape is `(13512, 733)` and audit shape is `(13512, 39)`. Each seed
  has 3,378 rows and 727 finite float64 model inputs. All five committed files
  have read-only mode `0444` and explicit non-null Arrow schemas.
- The six leading dataset columns are Arrow-equal to the sealed M3 target
  table. Target counts are rescue 1,225; harm 2,256; both-correct 7,171; and
  both-wrong 2,860. Every one-hot block row sums to exactly five.
- The 131 focused M1–M3 tests and 101 new M4 tests passed (232 total).
  The new tests comprise 54 numerical, 37 artifact/integration, and 10 CLI
  tests. Compilation and formatting checks passed.
- SHA256 comparison confirmed unchanged frozen configuration, M1/M2/M3
  modules, and their three runners. The successful full parent validators
  independently rechecked the sealed artifact lineage.

Manifest payload SHA256:
`ca836471d62cca2079d78f8dfd8f50ecab46d884aa4e67ffc8993e4fa8d027ba`.
The manifest file SHA256 is
`4537e9d4427598a99c6c4b62658086ec17a8700331429e5ddb172291f378ec1c`;
these differ because the latter hashes the physical JSON file, including the
payload self-hash field. The dataset file SHA256 is
`12a5f5c4a62df8a1f7fb46e06c50b73a45ffd6ab7641612ea30d06cd9de7407a`.
All four child physical and logical hashes are recorded in the manifest;
the manifest's own physical hash is recorded here.

The fitted temperatures below are rounded for display; the JSON states retain
the authoritative round-trip float64 values.

| Seed | Image | Geo | Raw fusion |
|---|---:|---:|---:|
| 1 | 2.621364 | 2.125809 | 2.577123 |
| 2 | 2.639728 | 2.145638 | 2.575757 |
| 3 | 2.609900 | 2.155896 | 2.605312 |
| 4 | 2.598001 | 2.143423 | 2.544514 |

All twelve fits succeeded with interior temperatures. Native-to-calibrated
fit-NLL ranges across the four seeds are image `1.829284–1.848918` to
`1.070299–1.075126`; geo `1.531539–1.559045` to `1.170437–1.181981`; and raw
fusion `1.636865–1.693220` to `0.979472–0.992654`. These remain fit diagnostics,
not held-out calibration estimates.

Focused test reproduction:

```bash
conda run -n habcls python -m pytest -q tests/test_multimodal_geo_protocol.py tests/test_multimodal_geo_oof.py tests/test_multimodal_geo_oof_artifacts.py tests/test_multimodal_geo_targets_features.py tests/test_multimodal_geo_m3_artifacts.py
conda run -n habcls python -m pytest -q tests/test_multimodal_geo_router_numeric.py tests/test_multimodal_geo_router_dataset_artifacts.py tests/test_multimodal_geo_router_dataset_cli.py
```

## Completion checklist

- [x] Numerical calibration and transform tests pass.
- [x] Parent/artifact validation, immutable publication, and CLI tests pass.
- [x] Focused M1–M4 regressions pass.
- [x] Real build publishes 12 temperatures and four transform states.
- [x] Dataset contains 13,512 rows and four `3378 × 727` feature matrices.
- [x] Independent validation reconstructs outputs without fitting.
- [x] Second build reuses identical committed bytes without fitting.
- [x] Frozen M1–M3 parent/code/config hashes are unchanged.
- [x] Real fitted temperatures, hashes, and test evidence are recorded.
- [x] Development plan marks only M4 dataset preparation complete.

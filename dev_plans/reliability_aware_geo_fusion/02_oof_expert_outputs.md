# M2 — Honest Out-of-Fold Expert Outputs

## Status and completion boundary

**Status:** Complete as of 2026-08-31.

M2 is complete only after seeds 1–4 have each produced four train-OOF stages
and one development-train-to-validation stage, the strict aggregate succeeds,
and all published artifacts validate. Code and synthetic-test completion alone
does not satisfy that boundary.

That completion boundary is satisfied. All 20 producers validate, the sealed
OOF aggregate contains 13,512 records, the sealed label-blind validation
aggregate contains 3,288 records, and the OOF reproduction report regenerates
exactly from development-train labels.

M2 is an additive child implementation of immutable `protocol_v1`. The frozen
M1 runner, protocol implementation, locked evaluator, resolved configuration,
assignments, and protocol manifest remain unchanged.

## Commands

Run from the repository root:

```bash
conda activate habcls

python tools/run_multimodal_geo_helpfulness_m2.py run-seed --seed 1
python tools/run_multimodal_geo_helpfulness_m2.py run-seed --seed 2
python tools/run_multimodal_geo_helpfulness_m2.py run-seed --seed 3
python tools/run_multimodal_geo_helpfulness_m2.py run-seed --seed 4

python tools/run_multimodal_geo_helpfulness_m2.py aggregate
```

GPU selection is external through `CUDA_VISIBLE_DEVICES`. `run-seed` requires
CUDA. `aggregate` performs no fitting and can run on CPU.

Every command first invokes the fingerprinted M1 `validate-protocol` command
through the active Python interpreter. The M2 command writes nothing when that
preflight fails.

## Implemented producer contract

For each seed, stages run in this fixed order:

1. train OOF fold 0;
2. train OOF fold 1;
3. train OOF fold 2;
4. train OOF fold 3;
5. all development-train to development-validation.

Each stage starts from the exact locally available pinned Hugging Face snapshot
and verifies its checkpoint/config/tokenizer hashes. It adapts only the final
11 visual groups for five fixed epochs using the frozen 18 prompts. The text
tower remains frozen. Fitting and held-out embeddings both use the deterministic
prediction transform after adaptation; only adaptation uses the randomized
training transform.

Geographic inputs are explicitly projected as ordered `A00` through `A63`.
Legacy cached `I*` columns are never discovered or loaded. Mean and population
standard deviation are fitted on the producer's fitting rows only, in float32,
with exact zero standard deviations replaced by one.

The seed is independently reset before encoder adaptation and before each fresh
`image_only`, `geo_only`, and `raw_concat` head. Heads always have 18 outputs,
including folds whose fitting rows omit rare classes.

Prediction features are opened only after all fitting ends. Outputs contain
authoritative float64 logits, dense-ID predictions, and redundant descriptive
native-`T=1` float64 probabilities. They contain no labels, correctness, true
class probabilities, or NLL values.

## Immutable artifacts and recovery

Producer directories are:

```text
development_train_oof/seed_<s>/fold_<k>/
development_validation/seed_<s>/
```

Every producer contains its label-blind Parquet output, complete adapted visual
tower, three head checkpoints, geo scaler, resolved stage config, fitting-only
training metrics, and manifest. Before publication, the implementation reloads
the saved visual tower, scaler, and all heads and reproduces the held-out table.
Identities, folds, seeds, and predictions must match exactly; logits and
probabilities use `atol=rtol=1e-6`. Serialized native-probability integrity uses
an absolute tolerance of `1e-8`.

Stages are built in an owned staging directory and atomically published. A
valid matching stage is skipped on rerun. A corrupt, stale, or mismatched
published stage fails closed and has no force-overwrite path. Process locks
prevent concurrent writers from deleting one another's staging data.

## Aggregation and report

Aggregation validates all 20 producers before concatenating them. It never
averages or pools seeds and seals:

```text
development_train_oof/development_train_oof_model_outputs.parquet
development_validation/development_validation_model_outputs.parquet
```

Acceptance counts are 13,512 OOF records (`3,378 × 4`) and 3,288 validation
records (`822 × 4`), with unique key `(row_uid, training_seed)` and canonical
sort by that key. The validation table remains label-blind.

After sealing the OOF table, aggregation joins development-train labels only to
create `oof_reproduction_report.json`. It reports per-seed/per-mode top-1,
top-3, weighted and macro F1, MCC, fixed 18×18 confusion matrices, and
cross-seed mean/population-standard-deviation. It never loads or scores
development-validation labels.

## Completion evidence

- The real M1 preflight remains valid with 4,200 rows and 1,625 plots.
- Seeds 1–4 each completed four OOF producers and one
  development-train-to-validation producer. All 20 published producers pass
  the strict manifest, partition, schema, checkpoint, scaler, and output
  validators.
- Per-seed held-out OOF fold counts are 862, 815, 849, and 852, totalling 3,378
  development-train rows. Each seed also predicts all 822
  development-validation rows exactly once.
- `development_train_oof_model_outputs.parquet` contains 13,512 canonically
  sorted records with unique key `(row_uid, training_seed)` and logical content
  SHA-256
  `989236e78dea2f4a8eb4a31d201e102c716c7f3d689afb2fd88acb262030d2bc`.
- `development_validation_model_outputs.parquet` contains 3,288 canonically
  sorted, label-blind records with logical content SHA-256
  `ac4fa97736b7e4e6fa9dc58a29cc9d739e26a945925b1edb1eccb5b5c4c04f52`.
- Aggregation is exact concatenation without averaging or pooling. Every
  logits/probability vector is finite float64 of length 18, native-`T=1`
  probabilities reproduce exactly from serialized logits, and predictions
  agree with logits/probability argmax.
- Every producer records a successful saved-checkpoint replay. Identities,
  folds, seeds, and predictions reproduce exactly; logits and probabilities
  satisfy `atol=rtol=1e-6`. This is checkpoint-to-output replay, not a second
  independent rerun of all GPU training stages.
- `oof_reproduction_report.json` reproduces exactly and reports the declared
  per-seed/per-mode metrics, fixed 18×18 confusion matrices, and cross-seed
  mean/population-standard-deviation. Development-validation labels remain
  unopened and unscored in M2.
- The focused M1/M2 protocol, output, artifact, aggregation, reporting, and
  tamper-rejection test suite passes all 84 tests.
- All producer and aggregate artifacts are immutably published with no staging
  directories remaining.

## Out of scope

Temperature calibration, router targets/features, router fitting and threshold
selection, full-development expert refitting, locked-test inference, and
development-validation scoring belong to later milestones.

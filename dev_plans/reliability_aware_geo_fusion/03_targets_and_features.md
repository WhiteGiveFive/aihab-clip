# M3 — Seed-Specific Targets and Deployment-Safe Features

## Status and completion boundary

**Status:** Complete as of 2026-09-01.

M3 is an additive, CPU-only child of immutable M1 and M2. It is complete only
after the real target/schema/report bundle has been built from all 16 validated
development-train OOF producers, published immutably, revalidated, and reused
successfully on a second build command. Pure code and synthetic tests alone do
not satisfy this boundary.

M3 owns seed-specific targets, the stateless 30-feature semantic builder and
schema, target prevalence, the leakage audit, and their lineage manifest. It
does not fit expert temperatures, transform or one-hot state, router models, or
thresholds. It materializes neither calibrated feature rows nor
`router_dataset.parquet`; those operations belong to M4.

## Commands

Run from the repository root in the `habcls` environment:

```bash
python tools/run_multimodal_geo_helpfulness_m3.py build
python tools/run_multimodal_geo_helpfulness_m3.py validate
```

Both commands accept only `--config`, `--protocol-dir`, and `--artifact-root`
path overrides. They expose no temperature, validation-output, test, fitting,
or force-overwrite argument.

Before any M3 write, the workflow invokes the frozen M1 preflight, validates
only the 16 train-OOF M2 producers and their manifests, reconstructs their
canonical concatenation, and compares it with the sealed OOF aggregate. It
uses a role-filtered development-train assignment projection for labels. It
does not open the M2 development-validation outputs, final in-sample outputs,
or locked-test sources.

## Frozen parent lineage

The real bundle binds these immutable parents:

| Parent | Frozen SHA-256 |
|---|---|
| Development assignments, logical content | `1c00ebbd93349e544002f37db66ff1278ef0b8034738ea4448dfc6fb18376928` |
| M2 development-train OOF aggregate, logical content | `989236e78dea2f4a8eb4a31d201e102c716c7f3d689afb2fd88acb262030d2bc` |
| Class map | `6fdbd2796b76e50561845d64f47f80df387dbae3f0f9f537b4874f2438ca648e` |
| Router feature allow-list | `dc05bd0b9d40f9f20e1440885b8a2fc1b8a64545a8b9e3e2cff589bcdda5d99d` |
| Effective protocol configuration | `2730d1d9942ff96b0c690f74d900fb99fd4b6f382b897ddf4969c5364acf984b` |

The manifest additionally records physical hashes for the assignments,
resolved protocol, protocol manifest, sealed OOF table and aggregate manifest,
all 16 producer manifests, the current M2 implementation, both M3 code files,
the M3 schemas, and every M3 child artifact.

## Target contract

The published table has exactly these columns:

```text
schema_version
protocol_id
row_uid
plot_idx
training_seed
target_state
```

Its unique key and canonical order are `(row_uid, training_seed)`. All four
seed realizations are retained; there is no averaging, majority vote,
representative-seed selection, or deduplication. For one fixed seed, the
logical projection remains uniquely keyed by `row_uid`, so adding the seed to
the combined physical M3 schema does not change `protocol_v1`.

Labels are joined many-to-one by `row_uid` only after assignments are filtered
to `development_role == train`. Plot identity and `train_oof_fold` must agree
with the sealed assignment before the fold is omitted from the published
table. The four exhaustive raw-fusion-relative states are:

| State | Raw fusion | Geo only |
|---|---|---|
| `rescue` | Wrong | Correct |
| `harm` | Correct | Wrong |
| `both_correct` | Correct | Correct |
| `both_wrong` | Wrong | Wrong |

The table serializes no label, correctness flag, prediction, logit,
probability, utility, or image-relative diagnostic state. Image-versus-geo
states (`geo_only_correct`, `image_only_correct`, `both_correct`, and
`both_wrong`) appear only in the prevalence report.

## Semantic feature contract

`build_router_feature_frame` is the only primary semantic feature builder used
by future router training and deployment inference. It has no learned state and
accepts only three equal, non-empty `(n, 18)` probability matrices plus the
explicit basis:

```text
scalar_temperature_calibrated
```

It rejects M2's `native_t1_uncalibrated` basis. M4 must fit the 12 scalar
temperatures from authoritative OOF logits and supply the resulting calibrated
probabilities. M3 does not create a partial native-probability feature table.

The ordered 30 semantic features are:

| Family | Count | Ordered features |
|---|---:|---|
| Categorical | 5 | `image_pred`, `geo_pred`, `raw_pred`, `image_geo_pred_pair`, `geo_raw_pred_pair` |
| Boolean | 3 | `image_geo_agree`, `image_raw_agree`, `geo_raw_agree` |
| Integer | 2 | `image_geo_top3_overlap`, `raw_rank_of_geo_pred` |
| Numeric | 20 | `image_confidence`, `geo_confidence`, `raw_confidence`, `image_entropy`, `geo_entropy`, `raw_entropy`, `image_top2_margin`, `geo_top2_margin`, `raw_top2_margin`, `geo_minus_image_confidence`, `geo_minus_raw_confidence`, `geo_minus_image_entropy`, `geo_minus_raw_entropy`, `geo_minus_image_margin`, `geo_minus_raw_margin`, `image_geo_jsd`, `image_geo_total_variation`, `image_probability_at_geo_pred`, `geo_probability_at_image_pred`, `raw_probability_at_geo_pred` |

Predictions and top-k order use descending probability then ascending dense ID.
Pair IDs are `image * 18 + geo` and `geo * 18 + raw`. Top-3 overlap is an
integer intersection cardinality from 0 to 3. Raw rank of the geo prediction is
one-based from 1 to 18. Entropy and Jensen–Shannon divergence use natural logs
without normalization; zero-probability terms contribute zero. Total variation
is half the L1 distance. Probability rows must be finite, lie in `[0, 1]`, and
sum to one within absolute tolerance `1e-8`; the builder never clips or
renormalizes them.

Predictions, ranks, and overlap use `int8`; pair IDs use `int16`; flags use
`bool`; all numeric features use `float64`. Fixed categorical vocabularies are
`0..17` for predictions and `0..323` for pairs. M4 later expands the 25
Boolean/integer/numeric values and 702 fixed-vocabulary one-hot values into 727
`float64` columns.

## Artifact bundle and reports

The immutable bundle is:

```text
router/targets_and_feature_contract/
├── router_targets.parquet
├── router_feature_schema.json
├── target_prevalence.json
├── feature_leakage_audit.json
└── manifest.json
```

All child files are staged first and the manifest is published last as the
commit marker with exclusive no-replace semantics. A valid rerun returns
`reused_valid`. A committed corrupt or stale bundle fails without overwrite;
only an exact, owned, uncommitted partial publication may be recovered.

`target_prevalence.json` reports pooled and per-seed counts; complete habitat,
plot, image–geo pair, and geo–raw pair breakdowns; report-only image-relative
states; and cross-seed stability. Unsupported categories have zero counts and
`null` rates, never NaN or Infinity. Its interpretation warning states that
13,512 seed realizations represent 3,378 images rather than independent
biological samples.

`feature_leakage_audit.json` reproduces the exact ordered allow-list, configured
forbidden patterns, identity/group exclusions, calibrated-basis requirement,
target/feature separation, stateless builder interface, and permitted
development-train input roles.

## Expected real-data verification

The target-generation inputs do not hardcode prevalence. A successful build
must independently reproduce:

| State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| `rescue` | 1,225 | 327 | 318 | 283 | 297 |
| `harm` | 2,256 | 560 | 549 | 572 | 575 |
| `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
| `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |

There must be 13,512 records, 3,378 unique images, 1,300 plots, and all four
training seeds. Cross-seed stability must report 2,389 images with one target
state, 875 with two, 100 with three, and 14 with four; therefore 989 images
change state across seeds.

## Completion evidence

The real bundle was built and sealed on 2026-09-01. The first `build` returned
`created`; the explicit `validate` and second `build` both returned
`reused_valid` with the same manifest self-hash
`a387bdbf6a988154e282e36f9de86a7ba2ade631057a83cadf477cedb6310a40`.
The physical manifest file SHA-256 is
`cc6bd85e6ff271ee6b7e1f69593915d8282be57e2eed5c99f5ccc2c306bfcc67`.

| Child artifact | Physical file SHA-256 | Logical/content SHA-256 |
|---|---|---|
| `router_targets.parquet` | `e4dc4150b0b17951f59616bf3ff6a763ec7ca598f1bf476cec60e6e50efd5481` | `f05b2bcaf077d07d55e29c4d2b50791d6c75e591f6d642cf962b48e7f232c707` |
| `router_feature_schema.json` | `10ceb9db80b8161c6cf7c2b13c8dcdf5abdd455ff0b642de91baf96072836a4b` | `6f9589da550c32c495a3825d7f12cbb42e49f57798b2fa0676917a35cbbca76c` |
| `target_prevalence.json` | `9c7bf3f114ea70793ed69a2f02ab720184fcc58bc7148e51203d2bc8c8ac8913` | `003a9f6d9fb1fd241ee6c24ac45e149f85195d87d5dd8ed37b208f7c80216e79` |
| `feature_leakage_audit.json` | `a412d9cd6cd252a3b658e65551bc72ab22461bb78bf25a2ee7c731d232f2ca04` | `2a3855fab0ab17e3bdbb650e043d6ef4967ee2680428e1829ad662b46564e5f5` |

The M3 implementation aggregate SHA-256 is
`7faa814dfebf1831b4f7c406c2ecf3e77dab84816325ef04ad43157fdd39675a`.
Its bound file hashes are:

- `multimodal/geo_helpfulness_targets_features.py`:
  `0ec8c9a924a29ccdb61503d451cd41b4dba23f532d12054722466126f08e7a44`;
- `tools/run_multimodal_geo_helpfulness_m3.py`:
  `bbefd860df48e004b74f9fc54a9c0ec31ba6232c8393eb111c81e8453ad77fbc`.

Observed target totals and every seed-specific count reproduce the verification
table above. The bundle contains 13,512 rows, 3,378 unique images, 1,300 plots,
and seeds 1–4. Cross-seed stability is exactly 2,389/875/100/14 images with
one/two/three/four distinct states, so 989 images change state. The auxiliary
image-relative counts are 1,547 `geo_only_correct`, 2,304
`image_only_correct`, 6,849 `both_correct`, and 2,812 `both_wrong`.

The complete reports contain 18 habitat categories, all 1,300 plots, and both
324-entry predicted-pair vocabularies. JSON contains no NaN or Infinity, the
leakage audit passes, all five committed files are read-only, no ownership or
staging receipt remains, and `router_dataset.parquet` is absent.

Validation results:

- 47 focused M3 target/feature/report/artifact tests pass;
- the five focused M1–M3 files pass 131 tests in total (43 M1 protocol, 36 M2
  pure/report, 5 M2 artifact, and 47 M3);
- repository-wide collection finds 209 tests; 208 pass and one unrelated,
  pre-existing untracked 10 m test fails because its `startswith("I")`
  assertion includes the required metadata column `ID` alongside `I000` and
  `I001` (`tests/test_multimodal_geo_10m.py:188`);
- Python compilation and whitespace checks pass for the M3 module, runner, and
  tests;
- every recorded immutable M1/M2 implementation/config SHA-256 remains
  byte-identical to its pre-M3 baseline.

## Completion checklist

- [x] Composite target identity and M3/M4 boundary confirmed.
- [x] Pure target, feature, schema, prevalence, and leakage contracts
  implemented.
- [x] Artifact lineage and immutable publication tests pass.
- [x] Real bundle builds and validates twice.
- [x] Real prevalence and cross-seed stability reproduce.
- [x] Leakage audit passes and no router dataset exists.
- [x] M1/M2 implementation fingerprints remain unchanged.
- [x] Final test totals and artifact hashes are recorded.

# M1 — Frozen Experimental Protocol and Delivery Record

## Status and authority

**Status:** Complete on 2026-08-25. `protocol_v1` is frozen and authorizes M2
OOF expert generation; it does not authorize router fitting or real locked-test
inference.

The governing method specification is [README.md](README.md). This record turns
the selected simplified router-development protocol into an executable M1
record. M1 freezes contracts and validates boundaries; it does not train the real
router, run the real expert cross-fits, or open cleaned-test labels, features,
or diagnostics. It may read only a separately sealed, label-blind test identity
manifest containing row IDs, normalized filenames, and plot IDs for denylist
and overlap validation.

## Confirmed protocol structure

The router uses one fixed, plot-grouped development holdout instead of nested
outer cross-validation:

1. form the canonical 4,200-image, 1,625-plot development universe;
2. split it once into development-train and development-validation;
3. split development-train into four plot-grouped OOF folds;
4. generate router targets and features only from train-OOF expert outputs;
5. fit declared router candidates only on the train-OOF table;
6. fit experts on all development-train and predict validation once;
7. use validation to select router regularization and the hard-override
   threshold;
8. freeze the selected router-side pipeline and threshold specification;
9. refit only the explicitly allow-listed `expert_refit_state` on all
   development data;
10. apply the immutable bundle to the locked cleaned test in a separate
    exploratory evaluation event.

Development-validation is held out from expert and router fitting used to make
its predictions. Its labels nevertheless select router hyperparameters and the
threshold. The selected validation result is therefore selection-conditional
internal evidence, not an unbiased estimate of generalization. The cleaned test
has already influenced the research direction and also remains exploratory.
Strong confirmation requires a new temporal or geographical evaluation set or
a separately versioned nested protocol.

## Frozen protocol-v1 constants

| Item | Frozen protocol-v1 value |
|---|---|
| Development roles | Approximately 80% plots train, 20% validation |
| Role-assignment seed | `20260824` |
| Train OOF folds | 4 |
| Train OOF assignment seed | `20261824` |
| Expert training seeds | `[1, 2, 3, 4]` |
| Bootstrap seed | `20260826` |
| Expert schedule | Fixed 50 epochs; no held-out early stopping |
| Baseline anchor | `raw_concat` |
| Router target | `rescue`, `harm`, `both_correct`, `both_wrong` |
| Initial router | L2-regularized four-state multinomial logistic regression |
| Router grid | `C ∈ {0.01, 0.1, 1, 10}` |
| Primary router score | `q_rescue - q_harm` |
| Effective action | `score > threshold` and `geo_pred != raw_pred` |
| Primary policy utility | `rescued - harmed` and corresponding delta accuracy |
| Probability treatment | One scalar temperature per expert mode and training seed, fitted once from train OOF logits; native `T=1` is descriptive; no post-hoc router-output calibrator |

These values were frozen before any real router output was generated. A later
change requires a new protocol version and invalidates dependent caches.

At exactly 1,625 plots, the 80/20 target is 1,300 development-train plots and
325 development-validation plots. Image counts are allowed to differ from an
exact 80/20 ratio because complete plots are indivisible.

## Work package 1 — Freeze data identities and class ontology

Construct the canonical development universe from the existing train and
validation union. Preserve the original source split only as metadata; it has
no modeling role in this protocol.

Assert and record:

- 4,200 unique development images;
- 1,625 unique development plots;
- exactly one habitat label per plot;
- no plot or row overlap with the 1,347-image, 531-plot cleaned test;
- stable row identities independent of source-table ordering;
- a fixed 18-class output head in every fit, including fits whose training
  partition has no example from a rare class.

The overlap assertion reads only the active
`locked_test_registry/.../locked_test_identity_manifest.json`, an immutable
label-blind artifact containing canonical test row IDs, normalized filenames,
plot IDs, counts, and
its ordered identity-projection hash. This hash covers only the allowed identity
columns, not the hidden full table. Development code cannot resolve the
underlying test table from this manifest and cannot read test labels, features,
or prior diagnostics. If the registry does not yet exist, a separate one-time
sealing command may read only those identity columns and must create the active
snapshot exclusively before protocol generation; later protocols can only
reference that snapshot read-only.

Freeze this dense-to-canonical class order:

| Dense ID | Canonical L3 ID | Habitat |
|---:|---:|---|
| 0 | 0 | Urban |
| 1 | 1 | Broadleaved Mixed and Yew Woodland |
| 2 | 2 | Coniferous Woodland |
| 3 | 4 | Arable and Horticulture |
| 4 | 5 | Improved Grassland |
| 5 | 6 | Neutral Grassland |
| 6 | 7 | Calcareous Grassland |
| 7 | 8 | Acid Grassland |
| 8 | 9 | Bracken |
| 9 | 10 | Dwarf Shrub Heath |
| 10 | 11 | Fen, Marsh, Swamp |
| 11 | 12 | Bog |
| 12 | 13 | Littoral Rock |
| 13 | 14 | Littoral Sediment |
| 14 | 15 | Montane |
| 15 | 17 | Inland Rock |
| 16 | 18 | Supra-littoral Rock |
| 17 | 19 | Supra-littoral Sediment |

The current label-mapping path must not inspect test labels to determine this
mapping or output dimension.

Canonical identity serialization is part of the protocol:

- `file` is Unicode-NFC text with `\\` converted to `/`, repeated separators
  and `.` path components removed, and absolute paths, `..`, NUL, empty, or
  undecodable values rejected;
- `file_lower` is `file.casefold()` and must be unique across the applicable
  row universe;
- `plot_idx` is an opaque, case-sensitive identifier because the CS data uses
  alphanumeric values such as `751X3` and `407XX1`; it is converted to Unicode
  NFC text, must be non-null and non-empty, and values with NUL, control
  characters, or surrounding whitespace are rejected rather than repaired;
  `normalized_plot_idx` means that validated NFC text;
- `row_uid` is lowercase SHA-256 hex over UTF-8 canonical JSON for
  `[dataset_id, file_lower, normalized_plot_idx]`, using Unicode NFC, no
  insignificant whitespace, and no ASCII escaping.

The manifest freezes `dataset_id`, the canonical-JSON implementation version,
and these rejection rules. Any collision or normalization change fails rather
than silently remapping a row.

## Work package 2 — Materialize the grouped assignments

Build the role split on one canonical row per `plot_idx`, sorted by
`(canonical_l3_id, normalized_plot_idx)`. For each plot, derive a stable rank
from SHA-256 of
`protocol_id|role_seed|canonical_l3_id|normalized_plot_idx`; do not depend on
input row order or a process-global random-number generator.

Rare-class constraints:

- singleton-class plots remain in development-train;
- for each habitat with `n` plots, set lower validation support to `0` when
  `n == 1` and `1` otherwise, set upper support to `0` when `n == 1` and
  `n - 1` otherwise, and initialize its quota to clipped
  `floor(0.20 * n)`;
- while quotas total fewer than 325 plots, increment the eligible habitat with
  largest `0.20 * n - quota`; while they total more than 325, decrement the
  eligible habitat with largest `quota - 0.20 * n`; break ties by canonical L3
  ID;
- within each habitat, assign the first `quota` plots by stable SHA-256 rank to
  validation and the remainder to training;
- every habitat represented by at least two plots therefore retains at least
  one training plot and contributes at least one validation plot;
- no plot is duplicated to improve apparent class balance;
- unsupported validation class metrics are `NA`; confusion matrices retain the
  fixed 18×18 ontology;
- each producer fit records per-class training-plot support.

Image-count balance is reported but does not override plot and habitat rules.
Within development-train, run `StratifiedKFold(n_splits=4, shuffle=True,
random_state=20261824)` on the canonical sorted unique-plot table, stratifying
by canonical L3 ID; freeze the implementation and dependency version in the
manifest. Within each habitat, this spreads plots across distinct folds as
evenly as support permits. Perfect habitat balance is impossible because
several habitats have only one-to-four development plots. When support is below
four, producer fits missing that class are unavoidable: never duplicate plots,
keep the fixed 18-way head, and report producer class support.

`development_assignments.parquet` has one row per development image:

```text
schema_version
protocol_id
row_uid
file
file_lower
plot_idx
source_split
image_source
label_id_dense
canonical_l3_id
label_name
development_role
train_oof_fold
```

`development_role` is `train` or `validation`. `train_oof_fold` is `0..3` for
train rows and null for validation rows. All images from one plot have the same
role and fold.

## Work package 3 — Frozen six-encoder-fit execution graph

For each training seed, the later M2/M4 execution is:

| Encoder stage | Training plots | Prediction plots | Purpose |
|---|---|---|---|
| 1–4 | Three of four development-train OOF folds | Remaining train fold | Honest router targets and features |
| 5 | All development-train | Development-validation | Router and threshold selection |
| 6 | All development | Locked cleaned test | Fit only `expert_refit_state`; load `router_frozen_state` byte-for-byte for final exploratory inference |

Protocol v1 performs six encoder fits per seed: four OOF producers, one
development-train-to-validation producer, and one final full-development
producer. Each adapted encoder is shared by `image_only` and `raw_concat` in
its stage; the stage then trains all three mode-specific heads, giving 18 head
fits per seed. Across seeds 1–4 this is 24 encoder fits and 72 head fits.
Geo-only does not consume the image encoder. Router-grid fits and one-time
temperature fits reuse stored logits and are not included in this count.

Producer-local learned state includes:

- adapted image-encoder weights;
- the three expert heads;
- geo standardization;
- the fixed final-epoch checkpoint.

For finalization, `expert_refit_state` comprises the adapted image-encoder
weights, the three expert heads, and explicitly named raw-input preprocessing
such as geo mean and standard deviation. The pinned externally pretrained
checkpoint is the immutable initialization for every fit. `router_frozen_state`
comprises expert temperature scalers, router feature schema, numeric
scalers/imputers/density estimators, categorical vocabularies and mappings,
router coefficients/checkpoints, any declared router-output calibrator, policy,
and threshold specification.

The primary protocol uses fold-contained, vision-only encoder adaptation
(`siglip2_vision_only_prompt_supervision_v1`).
Every producer starts from the pinned externally pretrained SigLIP2 checkpoint,
unlocks the last 11 vision groups, keeps the complete text tower frozen, and
uses one frozen hierarchical/descriptive prompt for each of the 18 classes.
It optimizes prompt-supervised cross-entropy for five fixed epochs with Adam
(`lr=5e-5`, zero weight decay), cosine annealing, batch size 16, and no held-out
evaluation or checkpoint selection. The selected legacy input path is frozen
explicitly: OpenCV BGR decode, forced 439×439 resize, and `Image.fromarray`
without a BGR-to-RGB swap before the checkpoint-native SigLIP2 transforms. This
compatibility quirk must not be silently corrected within `protocol_v1`.

The protocol uses fixed epochs, so neither an OOF prediction fold nor
development-validation selects an expert checkpoint. The existing global
habitat-finetuned checkpoint is excluded from the primary protocol because its
training plots overlap the development universe; it may be retained only as a
labelled sensitivity analysis.

## Work package 4 — Freeze router selection and finalization

Before validation outputs are scored, freeze:

- the four-state target definition;
- the deployment-safe feature allow-list and categorical vocabularies;
- calibration family;
- router family and finite hyperparameter grid;
- threshold candidate construction and serialization;
- utility, support, harm constraints, and tie-breaking;
- all heuristics used for comparison.

Protocol v1 fits one scalar temperature per expert mode and training seed once
from development-train OOF logits before the `C` search and applies it unchanged
to validation and locked-test logits. Native `T=1` is descriptive. Protocol v1
has no post-hoc router-output calibrator. Fit every other learned router feature
transform from development-train OOF rows only. Then, for each `C` candidate
and expert seed:

1. build router features using that frozen inference-compatible pipeline;
2. fit the router on all development-train OOF rows;
3. apply it, without updating anything, to validation expert outputs.

Select one common `C` by minimum arithmetic-mean validation four-state log loss
across seeds 1–4, breaking numerical ties toward stronger regularization. Keep
one seed-specific temperature/router checkpoint for that common `C`; never
choose a best seed. For the selected `C`, calculate

\[
s_i=q_i(\mathrm{rescue})-q_i(\mathrm{harm}).
\]

Sweep `+∞`, `0`, and midpoints between adjacent unique positive validation
scores pooled only to define candidate boundaries. Protocol v1 prohibits
negative thresholds. An effective action also requires geo and raw predictions
to disagree. Select one common threshold specification that maximizes mean
`rescued - harmed` across the four seed-specific policies, subject to the
frozen v1 constraints:

- pooled effective coverage of at least 1% across seed realizations;
- effective actions spanning at least 20 unique validation plots across any
  seed realization;
- pooled `harmed <= 0.5 * rescued`;
- positive net utility.

Tie-break by fewer harms, then fewer actions, then the higher threshold. If no
candidate is feasible, choose `+∞`, meaning never intervene.

Artifacts never serialize non-standard JSON infinity. A finite threshold is
stored as `{"kind": "finite", "value_hex": <IEEE-754 float.hex>,
"comparison": "strict_gt"}` plus a display-only decimal. The fallback is
stored as `{"kind": "never_intervene", "value_hex": null,
"comparison": "strict_gt"}`. The threshold-specification hash uses only the
canonical fields, so never-intervene and finite policies reproduce byte for
byte across platforms.

Pooled rescue, harm, action, and coverage counts sum the four seed realizations
for selection and stability screening; they do not turn seed×image rows into
independent biological samples. The 20-plot rule counts unique `plot_idx`
values. Removing one plot in a robustness check removes it from all four seed
realizations.

After selection, seal all four seed-specific `router_frozen_state` bundles with
their common `C` and threshold. No component of `router_frozen_state` may be
fit, updated, rebuilt, or expanded from any labelled or unlabelled data
afterward. `Fit 6` mutates only `expert_refit_state` on all development rows
and must load `router_frozen_state` byte-for-byte from the selection receipt.
This preserves the selected scoring function and numeric-threshold coordinate
system; the stronger final experts may still shift its input and output
distributions, which must be disclosed rather than corrected using test data.

Freeze the comparison policies before validation as follows:

- principal heuristic: on a raw-versus-geo disagreement, choose geo exactly
  when its maximum calibrated probability exceeds raw fusion's;
- fixed comparators: never intervene and always choose geo on disagreement;
- secondary JSD-only and margin-difference policies: choose their numeric
  thresholds from development-train OOF rows with the same utility, support,
  harm, and tie-breaking rule, then apply them unchanged to validation.

## Work package 5 — Freeze evidence and go/no-go semantics

The validation-selected result is an engineering feasibility screen. It is not
an unbiased complete-stack performance estimate. A plot bootstrap interval on
these rows is descriptive and conditional on the same data having selected
`C` and the threshold; it does not repair selection bias or fixed-split
sensitivity.

Before M4 executes, the following frozen pass, inconclusive, and no-go rules
apply. Pass requires all of:

- positive mean validation delta accuracy across seeds 1–4;
- positive net utility in at least three of four seeds;
- pooled harms no greater than half of pooled rescues;
- at least 1% pooled effective coverage and 20 unique acted-on plots;
- mean validation point gain above every predeclared simple routing heuristic;
- positive pooled net utility after removing the single largest
  positive-contributing plot;
- no underlying cleaned-test table, feature, label, or diagnostic path opened
  during development; access to the sealed label-blind identity manifest is
  permitted.

Pass requires every listed criterion. No-go applies when mean net utility is
nonpositive or the learned router fails to exceed the principal predeclared
heuristic. Every other failure—including inadequate support, instability, plot
concentration, or failure to exceed a secondary heuristic—is inconclusive. A
pass permits later model development and a separately authorized locked
exploratory-test event; it does not establish generalization.

## Work package 6 — Define artifacts and fingerprints

Primary M1 artifacts:

```text
multimodal_artifacts/
├── locked_test_registry/cs/gse_100m_cleaned_test/
│   ├── active_snapshot.json
│   └── <identity_projection_sha256>/
│       └── locked_test_identity_manifest.json
└── analysis/cs/gse_100m/geo_helpfulness/protocol_v1/protocol/
    ├── development_assignments.parquet
    ├── split_balance.csv
    ├── resolved_protocol.yaml
    ├── locked_test_snapshot_ref.json
    └── protocol_manifest.json
```

The manifest freezes:

- protocol and schema versions;
- development-universe and test-denylist fingerprints;
- the complete class map;
- role/fold algorithms, seeds, assignments, and balance counts;
- expert, calibration, router, threshold, and final-refit specifications;
- primary metrics, uncertainty language, and go/no-go rules;
- allowed development and locked-test entry points;
- Git revision, dirty-diff fingerprint, and environment versions.

Use SHA-256 content hashes for canonical sorted identities, development source
tables, the locked-test identity manifest, feature schemas and content,
encoder/checkpoint provenance, class map,
assignments, effective configuration, feature allow-list, code, and every
upstream manifest. A cache mismatch is a hard rejection, not a warning.

M1 also freezes, without producing real model outputs, the schemas for:

- label-blind development-train OOF expert predictions;
- label-blind development-validation expert predictions;
- label-blind router-candidate validation probabilities and scores, sealed
  before any validation label is opened;
- the separately joined router target table;
- `development_selection_receipt.json`, which seals the chosen router,
  temperatures, feature transforms and vocabularies, common `C`, threshold
  specification, candidate-set and candidate-prediction hashes,
  parent train-OOF and validation-output hashes, ordered validation-target and
  ontology hashes, validation metrics, and go/no-go result;
- the final full-development expert and frozen-router bundle;
- label-blind locked-test predictions and the later scoring receipt.

Every model-output manifest records fitting-plot and prediction-plot hashes,
row and plot counts, fixed class-map and feature-schema hashes, training seed,
checkpoint/scaler fingerprints, parent artifacts, and a zero-overlap assertion.
The final expert fit requires a sealed selection receipt and rejects scientific
overrides. Its composite inference manifest binds
`final_expert_bundle_hash` to the exact `frozen_router_bundle_hash` from the
selection receipt and rejects any mismatch. Artifact roles and allowed-parent
roles prevent full-development in-sample predictions from masquerading as OOF
router-training inputs.

## Work package 7 — Enforce command boundaries

Define separate entry points:

```text
python tools/run_multimodal_geo_helpfulness.py freeze-test-identity ...
python tools/run_multimodal_geo_helpfulness.py freeze-protocol ...
python tools/run_multimodal_geo_helpfulness.py validate-protocol ...
python tools/run_multimodal_geo_helpfulness.py build-train-oof ...
python tools/run_multimodal_geo_helpfulness.py fit-router-candidates ...
python tools/run_multimodal_geo_helpfulness.py score-router-candidates ...
python tools/run_multimodal_geo_helpfulness.py fit-final-experts ...
python tools/run_multimodal_geo_helpfulness.py locked-predict ...
python tools/run_multimodal_geo_helpfulness.py locked-score ...
```

M1 implements `freeze-test-identity`, `freeze-protocol`, and
`validate-protocol`, plus fail-closed command shells for the later interfaces.
The locked predictor and scorer must run end-to-end on a synthetic fixture so
their access and sealing boundaries are executable, but M1 does not run real
expert, router, or test inference.

Development commands must have no effective test-table path, must reject every
test row UID using the denylist fingerprint, and cannot report test metrics or
write `oracle_test_best`. The locked predictor has no label-loading or fitting
path. The scorer is the only test-label reader; it consumes already-sealed
predictions and cannot modify predictions or model state. It has no fitting,
recalibration, threshold-search, policy-selection, or scientific-override API
and may emit only the metrics and decompositions allow-listed in the protocol.

`fit-router-candidates` can fit but cannot resolve validation labels; it seals
`{C, seed, q_state, score}` validation predictions. Only then may the
fit-incapable `score-router-candidates` process open validation labels,
construct scoring targets, select the common `C` and threshold specification,
and exclusively create the selection receipt.

Selection, prediction, and scoring receipts use exclusive creation and are
keyed by the protocol, test-identity snapshot where applicable, parent bundles,
prediction hashes, exact ordered scoring-label hash, and ontology hash.
Overwrite, rescoring the same composite bundle, or a second selection event
under the same protocol ID fails closed. The protocol-independent test-event
registry permits a different frozen bundle to create only a new append-only
event; if that snapshot has previously been scored, the receipt is marked
`adaptive_reuse: true`. A new protocol ID therefore cannot hide prior test
access or make the evidence confirmatory.

## Work package 8 — Implement protocol validators and tests

Add tests for:

- deterministic, input-order-independent role and fold generation;
- one role per development row and one OOF fold per train row;
- null OOF folds for validation rows;
- plot integrity across both boundaries;
- zero development/test identity and plot overlap;
- identical assignments and fixed 18-class order for all modes and seeds;
- singleton and absent-fold-class behavior without shrinking the head;
- train/prediction plot-hash disjointness in synthetic OOF manifests;
- validation outputs accepting only a development-train producer manifest;
- rejection of test-derived label mappings and stale caches;
- file-access tracing proving development commands read only the sealed
  test-identity denylist, never test labels, features, or diagnostics;
- validation labels reaching only the selection scorer before receipt sealing
  and only `expert_refit_state` training afterward;
- `fit-router-candidates` being unable to resolve validation labels and
  `score-router-candidates` being unable to import or invoke fitting APIs;
- a synthetic selection-to-final-fit run preserving byte-identical hashes for
  all `router_frozen_state` components;
- rejection of final in-sample expert outputs as router-training parents and of
  composite bundles whose frozen-router hash differs from the receipt;
- one-time temperature fitting from train OOF and unchanged application to
  validation, final-expert, and test logits;
- rejection of router, calibration, threshold, vocabulary, and feature-schema
  overrides by `fit-final-experts`;
- locked prediction being structurally unable to fit or update artifacts;
- locked scoring rejecting fit, recalibration, threshold search, alternative
  policies, undeclared outputs, scientific overrides, and mutated label or
  ontology hashes;
- exclusive-create selection/test receipts rejecting overwrite, mutation, and
  a second score event for one bundle, while the global registry records later
  bundles as adaptive reuse of the same immutable snapshot;
- a second protocol being unable to replace the global active test snapshot;
- exact regeneration of assignment content hashes.

## Implemented repository changes

The M1 implementation isolates itself from these existing-path hazards:

- the current joined-split loader opens train, validation, and test together;
- the current label-map builder can consult test labels;
- the existing validation-training path constructs a test loader, reports test
  metrics, and can retain test-selected oracle checkpoints;
- current feature-cache reuse is not keyed by the complete resolved protocol,
  source content, class map, and split assignment;
- the existing habitat-finetuned checkpoint has development-overlapping
  training provenance unless a stricter manifest proves otherwise.

The protocol runner therefore uses development-only metadata loading and
fail-closed shells for later training commands. M2 must add a fixed-budget
train-and-predict primitive without repurposing the legacy validation loader,
because that path can select a checkpoint using the same rows whose predictions
are supposed to be held out.

| Path | M1 responsibility |
|---|---|
| `configs/multimodal_geo_helpfulness.yaml` | Frozen protocol constants and paths |
| `multimodal/geo_helpfulness_protocol.py` | Canonical universe, class map, assignments, schemas, hashes, validators |
| `multimodal/geo_helpfulness_locked_eval.py` | Capability-limited synthetic locked predictor and scorer |
| `tools/run_multimodal_geo_helpfulness.py` | Protocol commands plus fail-closed later command shells and synthetic locked-evaluation path |
| `tests/test_multimodal_geo_protocol.py` | Determinism, grouping, denylist, schema, and cache tests |
| `dev_plans/reliability_aware_geo_fusion/README.md` | Governing method and roadmap |
| `dev_plans/reliability_aware_geo_fusion/01_experimental_protocol.md` | Decisions, implementation record, and M1 completion evidence |

## Completion evidence

The active label-blind locked-test snapshot contains 1,347 rows and 531 plots.
Its identity-projection SHA-256 is
`1dbe08ab297ea39a53fe0e183648a0ff42929364162b0d6835756817e6acf284`.
The source path was supplied only to the one-time identity command and is absent
from shared protocol configuration and the sealed manifest.

The immutable development assignment contains 4,200 images from 1,625 plots:

| Partition | Images | Plots |
|---|---:|---:|
| Development-train | 3,378 | 1,300 |
| Development-validation | 822 | 325 |
| Train OOF fold 0 | 862 | 325 |
| Train OOF fold 1 | 815 | 325 |
| Train OOF fold 2 | 849 | 325 |
| Train OOF fold 3 | 852 | 325 |

The assignment content SHA-256 is
`1c00ebbd93349e544002f37db66ff1278ef0b8034738ea4448dfc6fb18376928`.
Regeneration after a deterministic input shuffle produced the same hash and an
exactly equal table. All development files resolve inside the single allow-
listed image source; the aggregate raw-image content SHA-256 over 4,200 files is
`3fff9a45f9a315beb3f9c29e5cc2eaf50497268b347f94c030c566c3c49715c3`.

The habitat-aware balance constraints passed. Singleton Littoral Rock remains
in development-train. Some OOF producers necessarily lack Urban, Littoral
Rock, Montane, or Supra-littoral Rock because those classes have only one train
plot; the ontology and every output head remain fixed at 18 classes.

Executed checks:

```text
python -m pytest -q tests/test_multimodal_geo_protocol.py
43 passed

python tools/run_multimodal_geo_helpfulness.py freeze-protocol ...
python tools/run_multimodal_geo_helpfulness.py validate-protocol ...
python tools/run_multimodal_geo_helpfulness.py validate-protocol ...
status=valid on both independent validations
```

The repository-wide suite produced 120 passes and one unrelated pre-existing
failure in `tests/test_multimodal_geo_10m.py`: its assertion treats metadata
column `ID` as an `I*` embedding column. No M1 code participates in that path.
The focused M1 suite, compilation checks, Markdown/config consistency audit,
and `git diff --check` passed.

The frozen artifacts live under:

```text
multimodal_artifacts/locked_test_registry/cs/gse_100m_cleaned_test/
multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/protocol/
```

Final immutable file hashes:

| Artifact | SHA-256 |
|---|---|
| `development_assignments.parquet` | `0e38ca3cc53ea79aed8bba01316add435050215898979505c2f6673a5f9b6e8a` |
| `split_balance.csv` | `850eed0acaff8a337a8e3b3099bae8aedcf34720a80aee4448c5756947b19ac8` |
| `resolved_protocol.yaml` | `c5a803b040fb51121b63e38cd2db7c0d726b7cfabaf5d3c5a548105cfb5feb8a` |
| `locked_test_snapshot_ref.json` | `5d715094d09a98d8d325442ea6500eaafeb865237e2469861fe159b1a4f5b6a6` |
| `protocol_manifest.json` | `1df9a7300ca55fd6d11a854b1c6c4bf78d1944c38438cce030d0de6294cd2645` |

`protocol_manifest.json` is the authoritative inventory of source, raw-image,
ontology, feature-schema, assignment, config, code, environment, and artifact
fingerprints. Validation rejects any mismatch.

## M1 completion gate

M1 completed every gate:

- [x] the development universe and fixed class mapping are test-independent;
- [x] pinned encoder provenance and the fold-contained B2 recipe are frozen;
- [x] role/fold assignments regenerate identically and pass plot/test overlap tests;
- [x] seeds, router grid, calibration, threshold, metrics, and go/no-go rules are closed;
- [x] protocol artifacts and content fingerprints validate twice;
- [x] a development smoke test succeeds after its command-local test source is removed;
- [x] the locked evaluator passes on a synthetic fixture without real test-label access;
- [x] the README, config, manifest, and this record describe the same protocol.

Completing M1 authorizes M2 OOF generation. It does not authorize router
training, cleaned-test inference, or gated-fusion development.

## Deferred higher-evidence protocol

A five-outer-by-four-inner plot-grouped nested design remains a separate,
higher-evidence option. It would require its own protocol ID, assignments,
artifact namespace, fingerprints, and complete rerun. It is not part of
`protocol_v1` and its terminology must not be applied to the fixed-holdout
artifacts.

# R2 — Implementation, input preparation and geometry verification

Recorded: **22 October 2026**. Status: **Completed.** The
[completion receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/completion.json)
confirms all three embedding bundles and four score sets, with no routing,
threshold selection or test scoring. **90 focused CPU checks passed.** The
[R1 protocol](01_validation_protocol.md) freezes the scientific contract; the
[governing README](README.md) retains the broader milestone order.

## Authorized deliverable

Implement and verify a reusable, development-only route from the completed
attention-fusion experts to auditable geometry scores. Prepare the exact
3,378-reference / 822-query allocation, extract three visual representations,
and produce four checkpoint-policy × geometry-arm score tables. Do not fit a
threshold, apply either routing rule or calculate routed performance in R2.

The two expert-policy probability pairs already exist; reusing their saved
outputs is sufficient for candidate classes. New inference is for geometry
embeddings only. No optimizer, training loop, training resume, OOF fit or test
evaluation belongs in this package.

## Code and executable contract

| Component | Responsibility |
| --- | --- |
| `configs/multimodal_attention_fusion_geometry_router.yaml` | Exact seed/regime/architecture/initialization, checkpoint epochs, source roots, geometry settings, future R3 settings and isolated output root |
| `multimodal/geometry_routing.py` | Pure identity-safe numerical functions for normalization, plot-minimum class distances, strict K5 support and log-ratio/minimum scores; no artifact discovery or training |
| `multimodal/attention_fusion_geometry_router.py` | Attention-fusion-specific protocol audit, source authentication, cohort/prediction alignment, extraction adapter, artifact publication and verification |
| `multimodal/attention_fusion_geometry_embeddings.py` | Authenticated visual-tower loading, historical deterministic extraction, encoder-specific cache publication and verification |
| `tools/run_attention_fusion_geometry_router.py` | Public R1/R2 `prepare`, `extract`, `score` and `verify` commands |
| Focused tests | Independent small examples and rejection cases that check scientific semantics and artifact boundaries |

R3's threshold/application commands are not part of the R2 CLI. Mathematical
helpers may be shared with historical geometry code, but the old four-seed
adapter must not be invoked on duplicated seed-1 data. Keep image/geo feature
normalization independent of classifier coordinate standardization.

Run the CLI from the repository root in the `habcls` environment. The public
sequence is:

```bash
python tools/run_attention_fusion_geometry_router.py prepare \
  --config configs/multimodal_attention_fusion_geometry_router.yaml
python tools/run_attention_fusion_geometry_router.py extract \
  --config configs/multimodal_attention_fusion_geometry_router.yaml
python tools/run_attention_fusion_geometry_router.py score \
  --config configs/multimodal_attention_fusion_geometry_router.yaml
python tools/run_attention_fusion_geometry_router.py verify \
  --config configs/multimodal_attention_fusion_geometry_router.yaml
```

`--config` defaults to the path shown. `extract` optionally accepts repeated
`--encoder pretrained`, `--encoder joint_final` and `--encoder joint_best`
flags to select individual encoder jobs; omitting them selects all three.
`--encoder` is rejected for other commands. GPU extraction requires the
available CUDA runtime outside the default sandbox. All four commands
completed successfully. CUDA access and extraction succeeded without a driver
repair or reboot. Repeating `extract` after completion verified and reused
the saved caches without running image inference again.

## Completed preparation

The real `prepare` command completed successfully on 22 October 2026 and
published [the protocol declaration](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/protocol/declaration.json)
and [resolved configuration](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/protocol/resolved_config.json).
Its input bundle contains:

- `inputs/train_rows.parquet`: **3,378 images / 1,300 plots**.
- `inputs/validation_rows.parquet`: **822 images / 325 plots**.
- `inputs/train_geo.npz` and `inputs/validation_geo.npz`: native 64D geo,
  aligned to those exact row orders.
- `inputs/fusion_final_predictions.npz`, `inputs/fusion_best_predictions.npz`,
  `inputs/geo_final_predictions.npz` and `inputs/geo_best_predictions.npz`:
  the four fixed expert-policy outputs aligned to validation identity.
- [Image inventory](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/inputs/image_inventory.json):
  all **4,200 development images**, with identity, resolved path, size and
  modification time; no new historical raw-byte replay gate.

Native geo is joined by file identity from the two original development
tables. Their historical 3,618/582 roles are superseded here by the existing
M3J 3,378/822 assignment. No test table is used. The declaration binds the
prepared-file hashes, source identities, three expected encoder digests and
four comparison IDs. Saved-weight authentication uses the bounded D039
mechanism during extraction; stage records capture the implementation-module
hashes that produced outputs.

The original **40 pure geometry checks** are included in the completed
**90-check** focused suite covering geometry, the artifact adapter and
extraction. The [CPU receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/cpu_checks.json)
records the exact command, 90 passes in 2.15 seconds, compilation and whitespace
checks. Real extraction, score production and artifact verification also
completed successfully.

## Artifact boundary

All new outputs live under:

```text
multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/
  seed1_pretrained_cls_original_v1/
    protocol/       # Resolved config, source snapshot and frozen study identity
    inputs/         # Development identities, audited source predictions and provenance
    embeddings/    # Original pretrained, fusion final-50 and fusion best-36 bundles
    scores/        # All 822 rows for each of four comparison cells
    verification/  # Input, numerical, extraction and artifact checks/failures
```

Manifests must identify source path/digest, row identity/order, dimensionality,
class order where applicable, encoder/checkpoint and extraction recipe. The
original-pretrained embedding bundle is shared across final and best scoring;
their score tables remain distinct because their expert candidate classes
differ. Record three unique encoder extractions, not four nominal independent
encoders or 16,800 distinct images.

Never overwrite the three completed seed-1 training packages. Published
side-study stages must distinguish incomplete work from completed verified
artifacts and bind outputs to their source manifests. Preserve failure evidence
and do not convert a failed verification into success by changing an expected
hash or silently replacing an input.

## Acceptance checks

### Inputs and scope

- Exactly 4,200 development images / 1,625 plots; training 3,378 / 1,300 and
  validation 822 / 325; no row or plot overlap between splits.
- Every row has one identity, plot and fixed-ontology label; each plot has one
  label; source prediction files align by unique identity rather than position.
- Exactly one seed and the declared pretrained-initialized partial-joint
  CLS-original fusion fit; best/final epochs match 36/50 and geo 11/50.
- Both experts' full probability vectors have 18 aligned classes and cover
  all validation rows; reject missing, duplicate, mislabelled or misplotted
  rows, non-finite probabilities and incompatible class order.
- Reference bank contains training only; no test query files or test labels
  are loaded and no validation query enters a neighbor bank.
- Native 64D geo remains available for geometry independently of the saved
  classifier scaler.

### Geometry and extraction

- Three encoder identities, with the correct matching reference/query encoder
  for every cell; embeddings have 4,200 rows and 1,152 dimensions each.
- Historical deterministic BGR/439 → pinned 384-pixel evaluation transform,
  mean/std 0.5, CUDA FP32 batch 16, cuDNN TF32 on, matmul TF32 off, no autocast;
  save actual resolved transform and runtime evidence.
- Finite, nonzero inputs and normalized image/native-geo vectors; correct
  encoder loading and output dimensionality; no training side effects.
- A small independent brute-force example verifies cosine distances,
  minimum-photo-per-plot aggregation and averaging five distinct plots.
- Multiple photos from one plot cannot satisfy K5. Both candidate classes must
  have five distinct plots; zero/insufficient class support is explicit and
  cannot trigger a lower-K fallback.
- The two log distance ratios use epsilon `1e-8` and their minimum has the
  declared sign. Ties and zero distances remain finite where supported.
- Altering/withholding query labels does not change geometry scores. Reordering
  inputs with preserved IDs does not change keyed results.

### Artifact verification and R2 boundary

- All four cell tables cover exactly the 822 validation identities, retaining
  agreement and unsupported rows with reasons and candidate support.
- Manifest hashes and identity/order metadata connect protocol, inputs,
  embeddings and scores; saved artifacts load successfully and independently
  checked score samples match their stored values.
- Bounded saved-file authentication preserves original expected digests and
  records failure/recovery attempts. No repeated live-weight checks or exact
  prediction-replay requirement is introduced.
- The published R2 bundle contains no fitted threshold, intervention decision,
  routed vector, routed metric, cleaned-test result or model-selection claim.
- Every final receipt retains `provisional_unresolved_memory_integrity`.

These checks establish the stated data/scoring contract. They do not establish
machine health, independent replication or the usefulness of a routing policy.

## Execution and evidence ledger

Every stage below has completed. Linked receipts identify the actual output
bundles; the scientific and assurance limits above still apply.

| Stage | Completed output | Evidence |
| --- | --- | --- |
| Protocol preparation | Frozen config/source identities and declared four-cell matrix | Complete: `protocol/declaration.json` and `protocol/resolved_config.json` |
| Input audit | 4,200 development rows; exact 3,378/822 split; source predictions aligned | Complete: prepared input bundle and image inventory; hashes in declaration |
| Original-pretrained extraction | 4,200 × 1,152 visual features | [Encoder manifest](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/embeddings/pretrained/manifest.json) |
| Fusion-final-50 extraction | 4,200 × 1,152 visual features | [Encoder manifest](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/embeddings/joint_final/manifest.json) |
| Fusion-best-36 extraction | 4,200 × 1,152 visual features | [Encoder manifest](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/embeddings/joint_best/manifest.json) |
| Geometry scores | Four tables × 822 queries with strict-K5 support and modality scores | [Score manifest](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/scores/manifest.json) |
| Focused verification | 90 passing CPU checks; compilation and whitespace checks pass | [CPU receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/cpu_checks.json) |
| Independent numerical audit | Sampled 288 class-distance cells and 960 neighbor identities; all 3,288 candidate-score rows reproduced | [Audit receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/independent_geometry_audit.json) |
| R2 close-out | Successful CLI verification; preserved source identities and score/embedding bindings | [Completion receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/completion.json) |

All three encoders produced **3,378 training vectors and 822 validation vectors**
of width **1,152**, totaling **12,600 vectors for the same 4,200 photographs**.
Extraction ran on the RTX 4090 with the declared deterministic FP32 batch-16
recipe. Recorded encoder-job durations were 161.246905 seconds for pretrained,
150.795501 seconds for final and 150.713258 seconds for best: **462.755664
seconds (about 7 minutes 43 seconds)** combined, excluding process overhead.
Existing encoder weights were loaded by reference; no duplicate weight files
or new model fits were created.

The score manifest binds eight NPZ files: `geo_class_distances.npz`,
`pretrained_class_distances.npz`, `joint_final_class_distances.npz` and
`joint_best_class_distances.npz`, plus the four candidate score files below.
Each candidate file covers all 822 query identities and contains no query
truth labels or routing actions.

| Candidate score file | Queries | Expert disagreements | Geometry-supported queries | Eligible disagreements |
| --- | ---: | ---: | ---: | ---: |
| `final__pretrained.npz` | 822 | 288 | 813 | 279 |
| `final__checkpoint_matched.npz` | 822 | 288 | 813 | 279 |
| `best__pretrained.npz` | 822 | 297 | 809 | 284 |
| `best__checkpoint_matched.npz` | 822 | 297 | 809 | 284 |

“Eligible disagreements” means the two experts disagree and strict five-plot
geometry is available for both candidate classes. It does **not** mean an
intervention was made, that the score favors geo, or that switching would help.
There are 9 unsupported final-policy queries and 13 unsupported best-policy
queries. Geometry support is identical between the two image arms because
they use the same reference identities, labels and candidate classes.

All three saved encoder files matched their expected hashes on the **first
attempt**; no cache-refresh retry was needed. Final verification confirmed the
17 declared source-file hashes unchanged. The
[image inventory check](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/image_inventory_check.json)
confirmed all 4,200 identities, paths, sizes and modification times unchanged
from preparation; it did not rehash raw image bytes or establish historical
byte equality. The [implementation snapshot](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/implementation_snapshot.json)
records 13 configuration, implementation and test files copied to
`protocol/source_snapshot`; embedding and score manifests additionally bind
their producer-module hashes.

A launch-time `tee` warning occurred because the log destination directory had
not yet been created. Extraction continued and published all three manifests.
The [launch note](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/extraction_launch_note.json)
preserves this logging issue; a subsequent cache-verification invocation exited
successfully and produced `verification/extraction.log` without repeating
inference. This was not a model, GPU or geometry verification failure.

The independent numerical audit used separate NumPy cosine calculations and
explicit plot/class loops without importing the project's scientific modules.
Across four sampled queries in each of four feature spaces, it checked 288
class-distance cells and 960 nearest-plot identities. Neighbor identities and
order matched exactly; the largest distance discrepancy was `6.66e-16`.
It then recomputed all 3,288 candidate rows from the saved class-distance
matrices: all 23,016 numeric values, identities and support masks matched.
All 17 declared source files remained unchanged. This is an independent
calculation check on saved artifacts, not a repeat extraction or evidence of
machine health; query labels and model weight bytes were not read by the audit.

## Handoff

R2 is closed and R3 is ready as a separate work package. R3 can apply the
frozen zero rule and fit the already-declared
20-plot / 1%-coverage / harm-cap-0.7 threshold separately in each of the four
cells, retaining all eight evaluations and explicit never-intervene outcomes.
The handoff must link the exact protocol and score manifests; changed inputs or
selection rules require an explicit prospective amendment.

After R3/R4 interpretation, return to the main attention-fusion study at M3J
**33/132 fits complete, 99 remaining**, then its M4/M5 work. The separate R5
cleaned-test application stays deferred until the main study is complete and
must remain exploratory. No main-study training or R5 scoring is implicit in
finishing R2.

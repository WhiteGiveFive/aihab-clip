# M1 — Experimental protocol and input audit

## Status and scope

**Status:** Complete on 2026-10-12. Study ID: `attention_fusion_v1`.
Applied pre-training amendment: [A001 — ReLU classification heads](01a_relu_classifier_amendment.md).
Applied extraction clarification: [A002 — Historical image-extraction numerics](01b_historical_image_extraction.md).

This record implements M1 of the [governing guide](README.md). The
[study configuration](../../configs/multimodal_attention_fusion.yaml) freezes
the initial **frozen-encoder** comparison and both final head-fitting routes.
It is a standalone contract for the dedicated M2 runner, not a configuration
that can already be passed to the generic multimodal trainer. M1 audits inputs
and defines experiments; it contains no attention training or new test evaluation.
M3J's exact online training recipe remains a separately bounded later decision.

The user confirmed training-partition habitat-adapted encoders, native geo as
default with L2 and standardization comparisons, inclusion of CLS in the initial
study, the five established evaluation views, and retention of checkpoints from
both fitting routes. Follow-up confirmations fixed Acc@1-first selection and
keeping the same adapted encoder for the all-development head refit.

## 1. Frozen data and producer inventory

Reuse the existing plot-grouped assignments; do not resplit the source tables.
The original table names `train` and `val` are source provenance, not the role
assignments for this experiment. Join their union to the canonical assignments
using normalized filenames, verify plot IDs, then sort each role by `row_uid`.
Reuse the existing canonical identity functions and fixed class map.

| Item | Audited value |
|---|---|
| Development universe | 4,200 images, 1,625 plots |
| Development train | 3,378 images, 1,300 plots |
| Development validation | 822 images, 325 plots |
| Role-assignment seed | `20260824`; existing assignments reused unchanged |
| Model/encoder seeds | `1, 2, 3, 4`; one shared split, four paired producer/training seeds |
| Image / geo widths | 1,152 / 64 |
| Geo source/order | Native 100 m GSE, `A00` through `A63` |
| Output order | 18 dense classes; canonical IDs `[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19]` |
| Development-train/validation overlap | Zero plots; unique row IDs and normalized filenames |
| Development / cleaned-test overlap | Zero rows, filenames and plots against label-blind identity manifest |
| Raw development images | All 4,200 exist and are nonempty; 5,058,079,955 bytes total |
| Native geo | Exact 4,200 × 64 aligned matrix; all finite; no zero vectors or constant training coordinates |
| Native geo vector norms | Minimum 0.863608, median 0.983026, maximum 0.999557 |
| Validation image caches | Four authenticated 822 × 1,152 float32, finite, L2-normalized arrays |

The 18-way output includes **Littoral Rock**, which has four training images
from one plot and no validation examples. Preserve the 18 × 18 confusion matrix;
mark its unsupported validation recall `NA`. Several other classes have very
few validation plots. Report support alongside class-specific findings.

The full input inventory, hashes, fitting membership, class support and numeric
checks are in
[input_audit.json](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/protocol/input_audit.json).
The [protocol manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/protocol/protocol_manifest.json)
binds the configuration, audit and expanded candidate matrix. Large artifacts
remain under `multimodal_artifacts`, outside the documentation directory.

### Image producers and preprocessing

For seed `s`, use this producer, frozen throughout M3 and both frozen-study
final head-fitting routes:

```text
multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/
  development_validation/seed_<s>/adapted_visual_tower.safetensors
```

| Seed | SHA-256 of adapted visual checkpoint |
|---:|---|
| 1 | `c8377e99fff72dc3c0c7816581eefb52cf0c78ffe4c002a82d2c6682da73ea9b` |
| 2 | `ead8efe9a6cddf254960c29d706fc44de974b87559888842b748c107db866991` |
| 3 | `a257f35b3899f6cb54243cb3750ba1216dea16c32ee4de4936a801f878f9e357` |
| 4 | `0cbedbd84c7e00b609eee75a5c3dbf5fed9d1d7795fcc8bac549085ee810867b` |

All four byte hashes match their manifests. Recorded fitting rows and plots
exactly match development train; validation plots were excluded. Parent
assignment, configuration, class-map, adaptation and preprocessing hashes pass.
This verifies checkpoint bytes and recorded provenance, not a new replay of
historical encoder training.

These are habitat-adapted `hf-hub:timm/ViT-SO400M-16-SigLIP2-384` visual towers,
initialized from revision `fec784dabb3081a5f101fc74eefaf9d1ed08237b`.
Their recorded adaptation used five epochs, the final checkpoint, and the
declared last 11 vision groups. Existing prompt-supervised adaptation is
historical producer provenance; it is not the proposed joint fusion objective.

Preserve **`legacy_opencv_bgr_439_v1`** exactly: OpenCV color decode in BGR,
forced 439 × 439 linear resize, then PIL construction **without** swapping BGR
to RGB. Prediction preprocessing resizes to 384 × 384 using bicubic/antialias,
converts PIL mode to RGB, converts to tensor, and uses mean/std 0.5 per channel.
PIL mode conversion does not undo the earlier BGR channel interpretation.
Extract one deterministic evaluation view in float32, then image L2 normalization.
Do not silently correct the channel order or add a new augmentation arm.
Reject nonfinite/zero image embeddings. Verify new extraction against the
authenticated validation cache within recorded float32 tolerances in M2.

A002 distinguishes tensor dtype from backend arithmetic. Preserve historical
image extraction using **CUDA, batch size 16, cuDNN TF32 enabled and matrix
multiplication TF32 disabled**. Encoder weights/outputs remain float32. These
settings reproduced the checked historical validation vectors exactly; strict
TF32-disabled extraction did not. Fusion/head optimization still disables both
TF32 settings. Record extraction settings separately, preserve ordered batches,
and use a complete historical validation batch for replay verification.

### Cache reuse and preparation

Safe validation caches exist at:

```text
multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/
  joint_tcp_router_v1/producers/seed_<s>/validation/features.npz
```

Use only their `image_embedding`, `row_uid` and `plot_idx` arrays after checking
metadata, checkpoint lineage and row order. Their geo arrays were standardized
for another study and must not supply native geo here.

**M2 must extract the 3,378 training images with each selected seed's single
encoder.** No suitable training cache was authenticated in the known artifact
families. OOF training caches combine different fold encoders; ConfidNet caches
contain expert hidden activations rather than the required global image vector.
The `adapted_geometry_v1` NPZ files combine development and test embeddings;
their feature payloads were not opened and cannot feed this development runner.

Save separate train/validation caches in this study's `inputs/seed<s>/` directory
with row/plot IDs, encoder and source-image identities, transform recipe, dtype,
normalization, array shapes and hashes. Validate every image decode while
extracting. M1 checked raw-file presence/size, not pixel decoding or replay of
original image-byte hashes. M2 also verifies its per-image inventory against the
historical protocol's aggregate raw-development-image fingerprint.
Cache preparation is an explicit M2 prerequisite,
not a completed M1 extraction claim.

### Scope of the audit

The audit reads development identities, labels and native geo; encoder bytes
and producer records; validated development-validation image arrays; and the
separate label-blind test identity manifest. It does not read test features,
labels, predictions or diagnostic reports. Schema/manifest inspection does not
authorize opening mixed feature arrays.

Native geo is authenticated against the protocol-bound development source
tables. It was not re-compared with the mixed development/test raw geo payload.
The raw schema has no explicit grid-cell ID or coordinates suitable for a cell
overlap audit. Plot separation therefore does not establish independence between
100 m cells or geographical regions. The four seeds are repeated fits on one
split, not four independent data splits.

## 2. Geo preprocessing arms

Every geo-consuming candidate runs all three arms. Image-only is shared once.

| Arm | Exact treatment |
|---|---|
| `native` — primary/default | Supplied 64 coordinates, cast to float32; no additional normalization |
| `l2` — secondary | Divide each vector by `max(norm, 1e-12)`; zero stays zero and its count is recorded |
| `standardize` — secondary | Coordinate mean and population std (`ddof=0`) fitted on 3,378 training image rows; float64 statistics, then cast transformed inputs to float32 |

For standardization, use scale 1 for a coordinate with std ≤ `1e-12`; record
the mask, means, scales, feature order and fitting-row hash. All transforms
reject nonfinite values. Fit once per permitted training partition, never on
validation. Do not deduplicate plots for these statistics: image-row weighting
is deliberate and consistent with the training loss. No normalization is applied
twice; adapter/token LayerNorm is a separate model operation.

Compare architectures **within** each arm. Native remains the default even if
a secondary treatment wins; any proposed default change needs a recorded
development decision. Report preprocessing effects separately from attention.

## 3. Initial candidate matrix

All classifiers use an 18-way output, unweighted cross-entropy and a
`Linear → ReLU → Dropout(0.1) → Linear` head with hidden width 256,
following user-confirmed amendment A001 for continuity with the existing
geo-helpfulness classifiers. The original GELU-head specification is archived
in the linked amendment record; no comparative fits preceded this change.
Adapters are separate `Linear → GELU → LayerNorm` mappings to width 128, with
no adapter dropout. Learn an image and geo type vector; there are no positional
embeddings. Token order is image then geo. Adapter and fusion-FFN GELU uses its
exact form, affine
LayerNorm uses epsilon `1e-5`, and linear layers have biases.

| ID | Architecture / readout | Expected trainable parameters |
|---|---|---:|
| `image_only` | 1,152D image → head | 299,794 |
| `geo_only` | 64D geo → head | 21,266 |
| `raw_concat` | Original image + geo, width 1,216 → head | 316,178 |
| `projected_concat` | Same adapters/types as attention; concatenate original adapted tokens → head | 227,090 |
| `tokenwise_capacity` | Two token-wise pre-LN residual FFNs; concatenate updated modality tokens → head | 359,442 |
| `attention_two_token` | One pre-LN self-attention block; concatenate updated image/geo tokens → head | 359,570 |
| `attention_cls` | Learned CLS + typed image/geo; two pre-LN attention blocks; updated CLS → head | 459,410 |

Each attention block has four heads, width 128, FFN width 256, attention-weight
dropout 0.1, output/residual dropout 0.1, and FFN hidden dropout 0.1. Both
attention and FFN sublayers have pre-LN residual connections. No extra final
encoder norm is used. The CLS vector has no additional type vector. Both layers
are independently initialized; do not accidentally clone identical initial weights.

The capacity control replaces the attention sublayer with another shared
token-wise `128 → 256 → 128` GELU FFN, retaining both pre-LN residual paths and
dropout. It never mixes tokens before the classifier. Its block has 132,352
parameters versus 132,480 for attention: only 128 fewer. It is a control for
the **two-token model**, not for CLS. CLS-only readout without token mixing
would be input-independent and is not a valid capacity control.

The CLS candidate is required in the initial study. Its comparison changes
both depth and readout; report it as a separate architecture, not an isolated
CLS-versus-concatenation effect. BP-style bypass and one-layer CLS are deferred.
Parameter counts above are analytic expectations to verify in M2, not measured
counts from an implemented attention model.

The existing geo-helpfulness `image_only` consumes only the 1,152D image vector.
Its `raw_concat` consumes image plus the 64D train-standardized geo vector,
without learned modality projections. The generic pipeline's
`tabular_projected_concat` is a separate mode, not the reported image-only or
raw-concat expert. M2 should reuse compatible baseline components and integrate
them under this protocol; inclusion in the matrix does not require rewriting
an existing architecture. Prior head weights/results require full recipe
compatibility before reuse, even with the same ReLU activation.

## 4. Training, selection and finite budget

| Setting | Frozen choice |
|---|---|
| Seeds | 1–4; seed `s` uses encoder `s`; seed Python, NumPy and Torch |
| Epochs/checkpoint | Exactly 50; save/select epoch 50, no early stopping or best-epoch search |
| Batch size | 128; shuffle train; keep final incomplete batch; zero loader workers |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon `1e-8`, weight decay `1e-4` on all trainable parameters |
| Learning-rate grid | `1e-4`, `3e-4`, `1e-3`, one equal three-choice budget per arm |
| Scheduler / clipping | None / none |
| Precision | Float32; no AMP or TF32 |
| Validation | Evaluation mode each epoch; histories are diagnostic; only epoch 50 ranks configurations |
| Imbalance treatment | Unweighted image-level CE, no class weights, sampler balancing or label smoothing |

There are **19 candidate/preprocessing arms × 3 rates × 4 seeds = 228 fits**,
or 11,400 training epochs. Image-only has one arm, not three duplicate fits.
The expanded [candidate matrix](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/protocol/candidate_matrix.json)
assigns every fit its input role, encoder seed, LR, budget and output directory.
This is a planned workload; M1 has run none of these fits.

Use a dedicated seeded shuffle generator so architecture construction does not
alter minibatch order. Match initial shared adapters and heads across candidates
where shapes agree, and reuse initialization for a seed across learning rates.
Use stable component-specific seeds derived from SHA-256 of
`attention_fusion_v1|seed|component_name`; take the first four digest bytes as
an unsigned big-endian integer. Common component names are `image_adapter`,
`geo_adapter`, `image_type`, `geo_type`, `classifier`, `cls_token`, and
`fusion_layer_<index>`. Keep data-shuffle seed equal to the training seed.
The configuration records linear, QKV, norm and token initialization rules.

Require deterministic algorithms, disable cuDNN benchmarking and record
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, package versions, device and code/input hashes.
An unsupported deterministic operation must raise or receive a documented
implementation resolution; do not silently turn determinism off. Numerical
identity across different hardware/software is not promised: record checkpoint
replay tolerances and logits agreement separately.

Select **one learning rate per candidate/preprocessing arm across all four
seeds**, using descending, unrounded mean validation **Acc@1**, then mean
**weighted F1**, then mean **MCC**. Further exact ties use smaller LR and stable
candidate ID. Never select the best seed or a different LR for each seed.
An interrupted run resumes from its last saved completed epoch; mid-epoch work
is replayed from that boundary. A failed/divergent run must be
repaired or reported incomplete, not omitted from the average or replaced by
an unregistered trial. Selection waits for complete candidate accounting.

## 5. Evaluation and interpretation

Required views are **Acc@1, Acc@3, weighted F1, MCC, and confusion matrices**.
Report each seed and mean/sample standard deviation. Preserve all 18 classes;
use zero-division 0 for weighted F1 and null recall for unsupported classes.
Resolve tied logits by smaller dense class ID for top-k metrics. Save confusion
counts plus row-normalized views, class support, trainable parameters, runtime
and peak memory. Do not average hard predictions and call that an ensemble.

Primary scientific contrast: `attention_two_token/native` versus
`projected_concat/native`, each with its independently selected LR. Required
supporting comparisons are matched raw concatenation, the token-wise capacity
control, and image-only. Report CLS and both secondary geo arms in full; they
cannot replace an unsuccessful primary contrast after seeing results.

For every paired contrast, report mean and per-seed metric differences, and
rescued/harmed image counts against image-only and raw concatenation. Generate
10,000 paired whole-plot bootstrap replicates using seed `20261012`: sample
the 325 validation plots with replacement, include all their images with plot
multiplicity, use the **same sampled plots for both methods and all four seeds**,
compute each seed's metric difference, then average across seeds. Report 2.5th
and 97.5th percentiles. Do not resample repeated seed predictions as independent
images. These intervals are conditional on selected recipes and this reused
split; they do not include hyperparameter-selection or new-split uncertainty.

Apply these predeclared descriptive categories to each contrast:

| Category | Rule |
|---|---|
| Supported improvement | Mean Acc@1 difference > 0, at least 3/4 seed differences > 0, paired Acc@1 interval lower bound > 0, and no negative mean differences in Acc@3, weighted F1 or MCC |
| No improvement | Mean Acc@1 difference ≤ 0 and at least 3/4 seed differences ≤ 0; still report any gains on other metrics |
| Inconclusive | Every other outcome, including positive but uncertain gains or conflicting metrics |

“Supported” here means consistent **development evidence**, not independent
confirmation. An attention-specific gain claim requires support over projected
concatenation, raw concatenation **and** the capacity control in the same geo
arm. Otherwise state which comparison succeeded and which remains unresolved.
Confusion analysis identifies class-specific gains, harms and rare-class
limitations; it cannot override the frozen ranking rule after the fact.

No application-specific minimum useful gain has been supplied, so there is no
invented practical-effect threshold. Report absolute percentage-point changes
and rescued/harmed counts; do not equate a supported positive direction with
practical importance. A later practical threshold or new search must be recorded
before its corresponding evaluation. Negative/inconclusive outcomes complete
M3 and do not prevent M3J.

## 6. Two final-fitting routes and reusable checkpoints

Retain both routes for every selected arm/seed:

1. **`development_train`:** preserve the selected existing epoch-50 checkpoint
   trained on 3,378 images, including its train-fitted scaler. Its 822-row
   validation predictions remain the development evidence; no retraining is needed.
2. **`all_development`:** initialize a fresh fusion/head with the same seed and
   selected architecture/LR, train for 50 epochs on all 4,200 images, and refit
   coordinate-standardization state on those 4,200 rows where applicable.
   **Keep the same 3,378-training-image-adapted encoder frozen.** Do not warm-start
   from the selected head or re-adapt the encoder on all development images.

This yields 76 retained selected train-only checkpoints and **76 planned new
all-development head fits**. The former validation rows are now training data
for route 2; their scores cannot compare the routes fairly. Compare both routes
only in the separate M4 exploratory application with recipes already frozen.
Do not choose architecture, epochs, preprocessing or a preferred route from
test scores. The old cleaned test remains exploratory due to historical reuse.

Save **all 228 candidate epoch-50 checkpoints**, not only winners. Each run
retains the resolved config, IDs/labels/logits/probabilities/top-3 predictions,
metrics/history, trainable counts, source and encoder hashes, preprocessing
state, class map, environment, seeds and status. Save a latest-resume checkpoint
after every completed epoch, including optimizer, epoch, all RNG and shuffle
generator state. Exact resume is scoped to these epoch boundaries, not an
arbitrary mid-batch interruption. Selected records
reference immutable candidate bundles; final refits use separate destinations.

An inference bundle comprises fusion/head weights, a content-addressed visual
checkpoint reference, model construction and pixel/geo preprocessing, class
order and all necessary hashes. Locally share the four large visual checkpoints;
a portable export must include their bytes and model construction assets, not
leave an unresolved machine-specific path. M2 must demonstrate prediction replay
after loading the bundle. Never overwrite completed runs to disguise a recipe change.

## 7. Implementation boundary and joint readiness

M2 implements the seven candidates, the preprocessing switch, safe cache
preparation, development-only training and replayable checkpoint interface.
Do not use the current generic trainer unchanged: it standardizes tabular
inputs, its projected control adapts only geo, and disabling per-epoch test
reporting still loads/evaluates test data. A separate development path must have
no test feature/label input; only the audited label-blind identity is permitted.

M2 acceptance includes identity/plot/class alignment; finite preprocessing and
training-local scaler fitting; correct token axis and sample-independent
evaluation; finite gradients and response to both inputs; image encoder staying
frozen; saved-bundle prediction replay; and a test-access exclusion check. Use
training-only smoke data for optimization checks, not validation to tune this recipe.

All 4,200 raw images and four starting visual checkpoints are available for M3J.
Its core is image-only, raw concatenation, projected concatenation and two-token
attention in **frozen-online and partial-joint** regimes. Bound that extension
to at most three recipes per model/regime, four seeds, **96 candidate fits** and
at most 50 epochs per fit. A training-only resource smoke check must precede
full scheduling. No wall-time or memory estimate is claimed before that check.
Freeze the exact unlocked parameters, augmentation/exposure, encoder/head rates,
warm-up, effective batch, precision and joint refit recipe in M3J before scoring.
Full encoder unfreezing remains deferred. M1 does not launch online GPU work.

## 8. Reproduction, evidence and handoff

From the repository root, using the existing `habcls` environment:

```bash
python tools/audit_attention_fusion_inputs.py \
  --output multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/protocol/input_audit.json
python tools/prepare_attention_fusion_protocol.py
python -m pytest -q tests/test_attention_fusion_input_audit.py tests/test_attention_fusion_protocol.py
```

The first command authenticates inputs and writes the detailed audit; the
second validates the study configuration, expands the 228-fit plan and writes
the resolved configuration and protocol manifest. Neither command trains a
model or opens test feature/label payloads. The preparer refuses an unsuccessful
input audit and records its own code hash. Repeating these commands creates a
fresh audit/manifest timestamp; compare their content and source hashes, not
timestamp-dependent file bytes. Do not revise a frozen experiment's source
configuration after M3 starts without a new study ID.

Initial M1 verification completed on 2026-10-12 (amendment verification is
recorded separately in A001):

- The real input audit passed **224 required checks**, including all four visual
  checkpoint hashes and validation-cache lineages. Its script/helper hashes match
  the current sources; no test feature or label payload was opened.
- Protocol preparation produced **19 arms, 228 planned development fits,
  76 selected-checkpoint references and 76 planned all-development refits**.
  Output paths include the encoder regime, candidate, geo treatment and seed.
- The two preparation-tool test files passed **23 tests**. These cover audit
  rejection, complete/nonduplicate candidate accounting, data-boundary and refit
  rules, reproducible matrix order, and rejection of changed lineage/configuration
  after study runs exist. They are not model/training tests.
- Local documentation links, whitespace and manifest-to-file hash bindings passed.
- Execution used Python **3.9.20** in the existing `habcls` environment. The audit
  records all relevant installed versions, including Torch 2.5.1, OpenCLIP 3.2.0,
  torchvision 0.20.1 and timm 1.0.19. The Git revision is
  `c4ac5851d4118a55ee41bc5bb66d54acbf172b0b`; the workspace is dirty, so the
  configuration, report, tool and helper file hashes are part of the provenance.

M1 completion evidence is the successful input audit, resolved standalone
configuration, explicit matrix/selection/refit rules and generated manifest.
The immediate next milestone is **M2**, beginning with compatible train-cache
extraction and the dedicated development-only model/training interface. Neither
scientific gains nor model/checkpoint replay tests are claimed by this M1 record.

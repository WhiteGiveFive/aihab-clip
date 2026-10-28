# M3J — Online image–geo training

Status: **Seed-1 partial-joint package complete: 16/16 fits**, finished
2026-10-20 at **21:39:24 UTC (22:39 BST)**. Every fit completed 50 epochs,
passed its recorded final frozen-value check, and has reloaded final/best
checkpoint evaluations. The queue reports no error or unresolved fit; the final
report contains 16 complete and zero incomplete fits. The
[frozen-online package](03b_seed1_frozen_online.md) also completed all 16 fits
at **2026-10-22 00:49:39 UTC (01:49 BST)**, with no queue error or unresolved
checkpoint. The [geo-only reference](03c_seed1_geo_only.md) completed under
D041, bringing seed 1 to **33/33 fits**. Its combined report and queue completed
at **2026-10-22 10:18:23 UTC (11:18 BST)**. **M3J overall remains in progress**
at 33/132 fits; later seeds are separate packages. D039 recovered the
thirteenth fit's checkpoint evaluation without repeating its completed training.
The original D038 implementation passed 163 CPU checks and complete GPU
verification before launching the 16 seed-1 partial-joint fits. It removed
mandatory resume comparison, numerical tolerance and live-weight checksum gates.
The active study is
`attention_fusion_joint_direct_v1`, governed
by `direct_freeze_checks_v1`. Earlier memory/read inconsistencies remain
unresolved; outputs retain `provisional_unresolved_memory_integrity` assurance.
This is a change to experiment verification, not a hardware repair or a claim
of bit-for-bit reproducibility. Failed prior-policy evidence is retained below.
The geo-only stage has its separate completed package; seeds 2–4 remain planned
under the governing seed-by-seed schedule. The D041 loader audit found that the
completed image packages used their distinct study IDs for shuffle and crop randomness, so
realized minibatch orders and image views differ across regimes despite the
declared common namespaces. Their data and recipes match; the regime contrast
does not isolate trainability under identical views/orders. Preserve the
original results and source snapshots; no rerun is authorized by this audit.

## Scope and fixed recipe

The [governing guide](README.md) and
[resolved configuration](../../configs/multimodal_attention_fusion_joint.yaml)
define the active direct-check experiment. Its scientific recipe is unchanged
from the earlier strict and exploratory attempts; their artifacts and the
archived exploratory configuration remain historical evidence. The full design
is 132 fits; this package contains eight
architectures × two image initializations × seed 1 × partial joint = **16 fits**.
Order architectures as shown below, running pretrained then CS-adapted for each.

| Architecture | Classifier input width | Fusion/head parameters |
|---|---:|---:|
| `image_only` | 1,152 | 299,794 |
| `raw_concat` | 1,216 | 316,178 |
| `projected_concat` | 256 | 227,090 |
| `tokenwise_capacity` | 256 | 359,442 |
| `attention_two_token` | 256 | 359,570 |
| `attention_cls` | 128 | 459,410 |
| `attention_cls_skip_adapted` | 384 | 524,946 |
| `attention_cls_skip_original` | 1,344 | 770,706 |

Add 167,635,696 trainable visual parameters to each fusion/head count for the
partial-joint trainable total. The full visual tower has 428,256,704 parameters.
Both initialization arms share these counts. The
[parameter inventory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_v1/verification/architecture_parameters.json)
also records each full-model total.

The architectures and their parameter initialization definitions are reused
from M3 and its bypass extension. The classifier retains a 256-unit ReLU hidden
layer; adapters and attention/capacity FFNs use GELU. Shared compatible
components use the original `attention_fusion_v1` component seed namespace.
No trained M3 classifier is imported. Every run starts independently from either
the pinned original visual tower or the audited seed-1 CS-adapted tower, with
fresh downstream parameters and Adam state.

| Setting | Fixed value |
|---|---|
| Development split | 3,378 training images / 1,300 plots; 822 validation images / 325 plots |
| Image path | Historical OpenCV BGR, 439×439 preliminary resize; random-resized crop to 384 for training; deterministic 384 resize for prediction; channel mean/std 0.5 |
| Image output | Differentiably L2-normalized 1,152D global vector |
| Geo | One train-fitted, population-standardized 64D vector; shared scaler across runs |
| Trainable visual scope | Blocks 17–26, final normalization and attention pooling: 135 tensors / 167,635,696 parameters |
| Frozen visual scope | All other visual parameters; frozen modules remain in evaluation mode |
| Epochs / batch | 50; physical and effective batch 16; final smaller batch retained; no accumulation |
| Optimizer | Adam, LR 5e-5, betas (0.9, 0.999), epsilon 1e-8, zero weight decay, no AMSGrad or clipping |
| Schedule | Cosine to zero, T_max 50, end-of-epoch step; no warm-up or early stopping |
| Training precision | FP32 parameters/Adam state; CUDA FP16 autocast forward/loss; direct backward without gradient scaler |
| Prediction precision | FP32 inputs/weights, autocast disabled; evaluation mode and no gradients |
| Arithmetic | cuDNN TF32 enabled, matmul TF32 disabled, highest float32 matmul precision; deterministic algorithms enabled; cuDNN benchmark disabled |
| Loss | Unweighted 18-class cross-entropy without label smoothing |
| Input execution | Zero data-loader workers, one CPU tensor thread; shuffle and augmentation streams separated from model/dropout RNG |

Raw image/geo inputs are reauthenticated against M1. No cleaned-test features
or labels enter this path. The train-fitted geo scaler is stored once under the
new study's `inputs/` directory. Historical artifact payloads are unchanged.

## Reproducible ordering and mode behavior

Training views are deterministic functions of the declared augmentation seed
namespace, seed, epoch and row identity. The active and earlier studies use
`attention_fusion_joint_v1` for augmentation, shuffle and model/dropout streams,
and `attention_fusion_v1` for component initialization, despite their separate
study/artifact identities. The transform temporarily isolates Python, NumPy and CPU tensor
RNG state. Shuffle has an independent seed/epoch stream. Therefore processing
another candidate or initialization first cannot alter matched input views.
Model dropout has its own reset seed and saved CUDA RNG state. Each active fit
starts at epoch 1 and trains continuously to epoch 50. Automatic training resume
is disabled. D039 continuation leaves completed fits unchanged and recovers
checkpoint evaluation only when all 50 epochs and the final freeze check are
already complete. It never resumes or restarts a partially trained attempt
automatically. If a fresh attempt is separately authorized, preserve the
interrupted attempt outside the active run directory before starting a new
history from the same declared initialization.

The wrapper sets the whole visual encoder to evaluation mode, then enables
training mode only for the selected blocks, normalization and pooling. A
classification-loss gradient flows through image normalization and applicable
fusion modules to those visual parameters. Validation uses current visual
weights; no historical frozen image cache supplies online model predictions.

## Comparisons, checkpoint selection and interpretation

Every run completes all 50 epochs. Preserve and report both:

- **Final:** epoch 50.
- **Best:** highest unrounded validation Acc@1 over epochs 1–50; exact ties use
  weighted F1, then MCC, then earliest epoch. One checkpoint supplies all metrics.

Both policies use the same fixed training recipe and receive separate tables,
contrasts and confusion analysis. Do not choose the more favorable policy after
scoring. The original epoch validation history selects the best checkpoint.
After training, load the saved final and best checkpoints and evaluate them on
the same validation rows. These reloaded evaluations supply the reported
metrics, logits, predictions and confusion matrices. Preserve the original
epoch metrics as selection provenance and learning-curve data; repeated
floating-point or prediction equality is not a completion gate. There is no
learning-rate or width search in this initial M3J recipe.
Across completed seeds, rank architectures within each declared
initialization/regime/policy by mean Acc@1, then mean weighted F1, mean MCC and
stable candidate ID for an exact tie. Do not select the best seed.

Declare `attention_two_token` versus `projected_concat` as the primary
architecture contrast within each initialization, encoder regime and checkpoint
policy. Supporting contrasts include image-only/raw-concat anchors, the
token-wise capacity control and each CLS bypass versus CLS-only. Compare
CS-adapted versus pretrained within each matched architecture/regime/policy,
and joint versus frozen-online within each architecture/initialization/policy.
The exact list is generated by the
[protocol module](../../multimodal/attention_fusion_joint_protocol.py) before
comparative fitting and saved with the input/code/configuration identities.

This package reports **single-seed descriptive differences**, rescue/harm counts
and per-class confusions. It cannot establish the four-seed supported-improvement
rule or a joint-versus-frozen gain before the matched controls exist. After the
complete matrix, apply M1's paired whole-plot bootstrap (10,000 replicates,
seed 20261012) and four-seed interpretation rules separately to both checkpoint
policies. These remain reused-validation development findings conditional on
checkpoint selection, not independent confirmation.

The prospective **M4 refit rule** is a fresh 50-epoch fit on all 4,200 development
images, starting from the same declared original/CS-adapted initialization with
fresh fusion/head and optimizer state. Fit geo preprocessing on those 4,200
rows; preserve the selected regime's visual trainability. Retain the original
development-train final and best bundles separately. There is no validation set
inside the all-development route, so it uses its fixed final epoch and does not
select a new best-validation checkpoint. No M4 refit or test scoring is included
in this package.

## Shared storage, checkpoint evaluation and monitoring

[Checkpoint storage](../../multimodal/attention_fusion_joint_storage.py) implements
option 3: immutable, hashed visual/fusion weight objects and separate Adam,
scheduler and RNG state. Final inference and final resume reference the same
weights. Atomic manifests retain the current best, final and latest training
state for reuse even though automatic training resume is disabled. Only
superseded rolling objects with no retained references can be
removed during training. Complete candidate checkpoints remain retained.
Standalone export collects the dependency closure and reconstructs the visual
architecture from saved metadata without requiring the original encoder files.

The runner writes `history.json`, numeric curve data, refreshed PNG curves and
an auto-refreshing HTML view every epoch; completed fits also have PDF curves.
Training metrics describe the augmented, changing-weight optimization pass.
Validation uses the fixed checkpoint and deterministic transforms. Histories
contain loss, Acc@1, Acc@3, weighted F1, MCC and fixed-class confusion counts.
`timing.jsonl` separates checkpoint/plot overhead from compute time.

The active study root is
`multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/`.
The earlier `attention_fusion_joint_v1/` and
`attention_fusion_joint_exploratory_v1/` roots retain failed verification and
diagnostic evidence. Do not replace their receipts or mix their objects into
the active study.
Runs are below `runs/seed1/joint_partial/<candidate>/<initialization>/`;
the evolving report is below `reports/seed1_joint_partial/`. Large objects are
stored once in `objects/`. Input/preprocessing and source snapshots are separate.

## Verification and execution

The active checks cover input boundaries, paired augmentation, model scope,
shared-object loading/export, checkpoint selection, protocol recording and
report reconstruction. A short training-and-inference GPU check covers each
authorized architecture/initialization using training rows only: finite loss
and gradients, selected vision/fusion parameters updated, correct frozen scope,
valid predictions, memory feasibility and usable saved checkpoints.

Check that frozen parameters have `requires_grad=False`, are excluded from the
optimizer and receive no gradients. Compare actual frozen tensor values against
an independent initial snapshot after the short training check and at the end
of each full fit. A directly observed difference is reported with its parameter
and magnitude and stops that attempt. Repeated live-weight checksums are not an
acceptance condition. File hashes remain for authenticated source/checkpoint
files. There is no interrupted/uninterrupted comparison, floating replay gate,
numerical tolerance or cumulative deviation budget in the active policy.

Run from the repository root with the project's `habcls` Python environment:

```bash
python -m pytest -q tests/test_attention_fusion_joint_*.py
python -u tools/run_attention_fusion_joint_study.py verify
python -u tools/run_attention_fusion_joint_study.py run
python -u tools/run_attention_fusion_joint_study.py --background run
python tools/run_attention_fusion_joint_study.py status
python tools/run_attention_fusion_joint_study.py report
```

The original D038 `run` command requires matching successful GPU verification,
freezes the protocol before scoring, and executes only the 16 authorized seed-1
partial-joint fits. Its original interruption handling archives an incomplete
attempt under `interrupted_attempts/<run_id>/<timestamp>/` and starts it afresh
from epoch 1. **That behavior must not be used to recover the already-trained
thirteenth fit.** D039 adds explicit continuation with no automatic retraining:
evaluate existing final/best checkpoints for a fully trained fit, skip completed
fits, and start only previously unstarted fits. The original receipt and source
snapshot remain preserved; the operational source change is recorded separately.
Saved final and best checkpoints are loaded and evaluated before a fit becomes complete.
The report excludes incomplete/verification fits from completed comparisons.
Use either foreground `run` or `--background run`, not both. Background execution
keeps a separate process session and persistent log; `orchestration/seed1_joint_partial/launch.json`
records its process ID, log, command and launch time. A study-wide writer lock
prevents concurrent fitting processes.

## Progress

- D038 is implemented in the trainer, runner, configuration, reporting and this
  record. **163 relevant CPU checks passed** across protocol/input/storage/CLI,
  models, training and reporting. Revised training-only GPU verification passed
  all 16 model checks, the fresh checkpoint fit, final/best loading/evaluation
  and standalone export.
- Earlier input audits and engineering tests are retained as historical evidence;
  they do not constitute a successful receipt for the changed active policy.
- The original job launched on 2026-10-19 at 13:49 UTC and stopped on
  2026-10-20 at 14:45 UTC after **12 completed fits**. Fit 13,
  `attention_cls_skip_adapted/pretrained`, completed all 50 training epochs and
  passed its final direct frozen-value check, then failed while loading the
  best-epoch-25 visual checkpoint. Three fits were unstarted at that stop.
- Read-only per-file cache advice restored that file's original expected
  checksum on two subsequent reads. D039 implementation is complete, with
  **134 focused checks passed** across protocol, CLI, storage, training and
  reporting. The numerical training-core AST matches the original snapshot.
  Continuation launched at 15:48:04 UTC on 2026-10-20 (16:48 BST). By 15:51:31 UTC,
  the thirteenth fit's final/best checkpoint evaluation had completed with
  **zero additional training epochs**. At that historical recovery snapshot,
  13 fits were complete, `attention_cls_skip_adapted/cs_adapted` was the active
  fourteenth fit, and the two original-embedding-bypass fits were unstarted.
- The continuation completed the remaining fits. The final queue status at
  **21:39:24 UTC on 2026-10-20** reports **16/16 complete**, no active fit, no
  error and no unresolved fits. All 16 histories contain 50 epochs; recorded
  final frozen-value checks and both saved-checkpoint evaluations passed.
- Historical incidents below remain unresolved limitations. No reboot or system
  setting change is authorized or required by this amendment.

## Seed-1 partial-joint completion, results and handoff

The [final queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/orchestration/seed1_joint_partial/status.json)
records completion at **2026-10-20 21:39:24.439081 UTC (22:39 BST)**:
16 authorized fits complete, no active fit, no error and no unresolved fit.
All 16 histories contain the declared 50 epochs, giving **800 training epochs**;
all recorded end-of-fit direct frozen-value comparisons passed. The saved final
and best checkpoints have **32 policy evaluations** using reloaded weights.
Both policies remain available even when their selected model weights coincide.

The [completion audit](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/completion_audit.json)
checks metadata, histories, recorded freeze/evaluation evidence, artifact
presence and weight-file sizes. This close-out performed **no new training,
inference, numerical replay or weight-file checksum audit**; it summarizes the
existing completion evidence. The final generated report has 16 complete and
zero incomplete fits, with 64 PNG and 64 PDF confusion figures. No cleaned-test
data were scored.

### Validation Acc@1 summary

All values below are percentages on the same 822-image development validation
cohort. Final means epoch 50; best means the single checkpoint selected by the
original epoch-history rule. The linked metric tables retain Acc@3, weighted F1,
MCC and each selected epoch.

| Model | Pretrained final | Pretrained best | CS-adapted final | CS-adapted best |
|---|---:|---:|---:|---:|
| `image_only` | 65.21 | 65.69 | 68.13 | 69.22 |
| `raw_concat` | 68.37 | 69.83 | 67.15 | 70.07 |
| `projected_concat` | 68.00 | 70.80 | 69.83 | 70.56 |
| `tokenwise_capacity` | 71.29 | 71.65 | 68.98 | 70.44 |
| `attention_two_token` | 71.29 | 71.53 | 69.83 | 71.29 |
| `attention_cls` | 71.17 | 71.53 | 68.61 | 68.61 |
| `attention_cls_skip_adapted` | 69.95 | 71.05 | 69.83 | 71.17 |
| `attention_cls_skip_original` | 71.17 | 71.78 | 69.34 | 70.44 |

The highest observed final Acc@1 is 71.29%, tied between pretrained two-token
attention and the token-wise capacity control. The highest observed best Acc@1
is 71.78% for pretrained CLS with original-embedding bypass (epoch 36).
For the primary attention-versus-projected contrast, pretrained two-token
attention is +3.28 percentage points at the final epoch and +0.73 at best
validation; the CS-adapted comparison ties at the final epoch and is +0.73 at
best validation. These are **single-seed descriptive differences** on reused
validation data. They do not establish supported improvement across seeds,
an attention-specific gain over capacity controls, or a joint-versus-frozen
benefit. The completed [frozen-online package](03b_seed1_frozen_online.md) now
provides descriptive regime comparisons, subject to the D041 finding that its
realized crop/shuffle streams differ from this package.
All results retain `provisional_unresolved_memory_integrity`; completing the
checks does not resolve the historical memory/read incidents.

### Retained deliverables

| Deliverable | Location and contents |
|---|---|
| Final report and inventory | [Report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/report.md), [machine-readable summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/summary.json), and [16-fit inventory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/tables/inventory.csv) |
| Metrics and matched comparisons | [Final metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/tables/final_metrics.csv), [best metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/tables/best_metrics.csv), [final contrasts](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/tables/final_contrasts.csv), and [best contrasts](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/tables/best_contrasts.csv) |
| Training and validation curves | [All 16 monitoring-page links](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/report.md#learning-curves-and-checkpoint-reuse); each run retains `history.json` and `curves/` CSV, HTML, PNG and PDF outputs |
| Confusions | [Final per-class counts/recall](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/tables/final_confusion_by_class.csv), [best per-class counts/recall](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/tables/best_confusion_by_class.csv), and [count/normalized figures](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/reports/seed1_joint_partial/figures/confusion/) |
| Reusable checkpoints and predictions | [Run directories](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/runs/seed1/joint_partial/): every run retains a `checkpoint_manifest.json` referencing shared final/best weights, preprocessing and provenance, along with final/best predictions and evaluation receipts. The [last fit's manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/runs/seed1/joint_partial/attention_cls_skip_original/cs_adapted/checkpoint_manifest.json) illustrates the retained layout; shared weight/state objects remain under `objects/`. |

### Subsequent packages and seed-1 handoff

**Seed-1 partial-joint fitting and the 16-fit frozen-online package are complete;
M3J overall remains in progress.** The
[frozen-online record](03b_seed1_frozen_online.md#completion-and-handoff) links
its final/best results, curves and regime comparisons. Those fits used the same
original pretrained or seed-1 CS-adapted starting towers, never a newly
joint-trained tower. D038/D039 practical checks and all completed partial-joint
artifacts are preserved.

The D041 audit found a loader namespace discrepancy: the completed packages
used their distinct study IDs to seed training crops and shuffling, instead of
the declared common namespaces. Their train/validation rows, model recipes,
initial encoders and preprocessing rules match, but the realized views and
minibatch orders do not. Preserve these outputs and label the regime differences
as descriptive; exact augmentation/order pairing cannot be claimed for them.
Do not rewrite history or rerun completed fits as part of the geo-only package.

The [seed-1 geo-only fit](03c_seed1_geo_only.md) also completed under D041,
closing the **33-fit seed-1 matrix** with a combined report and reusable
checkpoints. It uses the existing standardized-geo classifier and unchanged
M3J optimizer, precision, 50-epoch budget and final/best policy. This single
result is shared across image initializations and encoder regimes. Its
training/inference path needs no image tensors or encoder and uses the declared
shuffle namespace; shared allocation auditing still checks image-file existence
metadata. Later seeds remain separate packages under the governing sequence.

## Practical verification amendment D038

The user approved removing interrupted-versus-uninterrupted testing and the
tolerance configuration, retaining direct verification that frozen parameters
are unchanged. The active identity is `attention_fusion_joint_direct_v1`, with
`verification_policy: direct_freeze_checks_v1` and
`result_assurance: provisional_unresolved_memory_integrity`; it has no
`numerical_policy` or `numerical_audit` requirement. The scientific recipe,
paired random streams, 16-fit authorization, shared storage option 3, and
final/best reporting remain unchanged.

D038 supersedes D036's blanket diagnostic prerequisite and D037's tolerance
workflow for this package. It also supersedes earlier mandatory exact resume
and exact prediction replay acceptance clauses. It does not permit accepting a
directly observed frozen-value change, nonfinite training, failed checkpoint
loading or invalid sample identities. Preserve these concrete failure records
and the original failed verification artifacts. A passed practical check means
the stated training checks passed, not that hardware is proven healthy.

### First practical GPU check and export retry

The first D038 check passed all 16 one-batch training/inference checks, including
direct frozen-value comparisons, and completed the fresh one-epoch checkpoint
fit with final/best reload evaluation. Standalone export then failed its copied
file checksum for the initial frozen-reference object (not the fitted visual or
fusion weights). The original object subsequently matched its recorded SHA256
using the system checksum utility. A fresh export of the same manifest passed
the unchanged file-authentication checks. The failed staging copy was not
retained by the export routine, so the evidence does not distinguish a copy
error from an inconsistent read/check. No acceptance rule or code was changed
after this failure. The full short verification was repeated successfully,
including a fresh export and independent model reconstruction/evaluation.

Preserved evidence: [failed receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/verification/a3a21aadd082_1792417139190009061/failure.json),
[16 training checks](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/verification/a3a21aadd082_1792417139190009061/gradient_checks.json),
and [successful fresh export](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/verification/a3a21aadd082_1792417139190009061/export_retry.json).

### Successful verification and launch

The [successful receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/verification/a3a21aadd082_1792417520266131678/verification.json)
records all 16 training checks passed with unchanged frozen values, successful
final/best checkpoint evaluation and standalone export/reconstruction, and no
development-validation data scored during verification. No mandatory resume or
exact prediction replay comparison was performed.

The [launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/orchestration/seed1_joint_partial/launch.json)
identified the original background job for all 16 authorized fits. Its
[persistent log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/orchestration/seed1_joint_partial/run_1792417772800062480.log)
records input checking and fit/epoch progress. At the historical 2026-10-19
launch snapshot, the first fit,
`seed1/joint_partial/image_only/pretrained`, had started from epoch 1 and no fit
was complete. The [live status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/orchestration/seed1_joint_partial/status.json)
and [first-fit record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/runs/seed1/joint_partial/image_only/pretrained/run.json)
provide current progress beyond this documentation snapshot.
Do not change the scientific recipe or sources during active fitting.

## Checkpoint recovery amendment D039

On 2026-10-20, the user approved resilient saved-file reads, evaluation-only
recovery of the trained thirteenth fit, and continuation of independent fits
without redoing completed training. This is an operational amendment to D038;
it does not change model architecture, initialization, preprocessing, data,
optimization, numerical precision, epoch count, checkpoint selection or metrics.
The configured scientific recipe and paired random streams stay unchanged.

### Incident and read-only recovery

The thirteenth fit is
`seed1/joint_partial/attention_cls_skip_adapted/pretrained`. Its
[preserved failure record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/runs/seed1/joint_partial/attention_cls_skip_adapted/pretrained/finalization_attempts/1792511425632411311.json)
retains the original `previous_run` with all 50 epochs completed and an authentication failure when loading the
best-epoch-25 visual object for checkpoint evaluation. The final direct
frozen-value comparison passed for all 207 frozen tensors / 260,621,008 values.
The final-epoch-50 visual object also matched its saved checksum. These are
separate checks: a saved-file mismatch does not itself establish a frozen
parameter update during training.

The affected visual object is
`objects/81a9ed4b949e65e976718dc330b75fd76e568442a830a51c15fe715eb8385d7c.safetensors`.
An initial diagnostic read returned a different digest. Opening this file
read-only and asking the OS to discard its cached pages
(`posix_fadvise` with `POSIX_FADV_DONTNEED`) was followed by two reads matching
the original expected SHA256. The
[diagnostic record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/diagnostics/checkpoint_read_recovery_20261020T151822Z.json)
preserves all three digests and states that no checkpoint bytes or expected
digest were changed. This recovery suggests an inconsistent read/cache path;
it does not identify the underlying cause or guarantee subsequent stability.

### Authorized handling

- After a saved-file digest mismatch, allow at most **two read-only per-file
  cache-refresh retries**, for **three hash reads in total**. The original
  expected digest remains mandatory. Record failed reads, cache advice and the
  recovery or persistent failure; do not relabel a different digest as valid.
- Preserve the 12 completed fits and all original history/checkpoints of the
  thirteenth fit. Its recovery loads the existing final/best weights, retains
  the original epoch-25 selection, and reruns only checkpoint evaluation and
  reporting. It performs no optimizer updates or epoch replay.
- Require complete 50-epoch evidence and the passed final freeze check before
  evaluation-only recovery. A partially trained attempt is preserved and
  reported; continuation does not automatically resume or retrain it.
- If an output checkpoint remains unverifiable, keep that failure and its
  artifacts, mark the affected fit unresolved, and continue independent
  unstarted fits. Exclude the unresolved fit from completed comparisons.
  This exception does not authorize bypassing input/protocol authentication,
  freeze checks, finite-training checks or other scientific prerequisites.
- Preserve the original protocol declaration, source snapshot and successful
  GPU verification receipt. Record the revised operational code as an explicit
  amendment and verify unchanged scientific configuration and inputs before
  continuation; do not rewrite the old receipt to imply the new code produced it.
- Keep `provisional_unresolved_memory_integrity` on all outputs. No reboot,
  driver change or system-wide cache clearing is part of this recovery.

### Implementation checks and continuation commands

The implementation passed **134 focused tests**, with no failures, across
protocol, CLI, storage, training and reporting. The
[test receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/verification/d039_cpu_tests.json)
records the exact checked source hashes and command; the
[JUnit results](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/verification/d039_cpu_tests.xml)
retain the individual outcomes. Comparison of the numerical training-core AST
against the original source snapshot found no change. The
[preservation snapshot](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/diagnostics/d039_preservation_before.json)
records existing artifacts before recovery.

From the repository root, using the project's `habcls` Python environment:

```bash
python -u tools/run_attention_fusion_joint_study.py \
  --config configs/multimodal_attention_fusion_joint.yaml --device cuda \
  amend-recovery
python -u tools/run_attention_fusion_joint_study.py \
  --config configs/multimodal_attention_fusion_joint.yaml --device cuda \
  --background continue
```

`amend-recovery` records the operational amendment under
`protocol/amendments/D039_checkpoint_recovery/amendment.json`, preserving the
original declaration and receipt. `continue` preserves completed runs,
finalizes fully trained eligible fits without training, and starts only
unstarted fits. It does not invoke the D038 archive-and-retrain path.
The [D039 amendment](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/protocol/amendments/D039_checkpoint_recovery/amendment.json)
was recorded at 15:41:12 UTC on 2026-10-20. Continuation launched at
15:48:04 UTC (16:48 BST), process **470054**, using the command above.
Its [persistent log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/orchestration/seed1_joint_partial/run_1792511284303870005.log)
and [launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/orchestration/seed1_joint_partial/launch.json)
identify this new process separately from the original job.

### Historical recovery and continuation snapshots

Checkpoint evaluation had completed by **15:51:31 UTC on 2026-10-20** (16:51 BST).
The [finalization receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/runs/seed1/joint_partial/attention_cls_skip_adapted/pretrained/finalization_attempts/1792511425632411311.json)
records **65.52 seconds and zero additional training epochs**. Both saved
checkpoints loaded with their original authentication requirements:

| Recovered checkpoint | Epoch | Validation Acc@1 |
|---|---:|---:|
| Final | 50 | 69.951338% |
| Best, selected by the original epoch history | 25 | 71.046229% |

The [completed run](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/runs/seed1/joint_partial/attention_cls_skip_adapted/pretrained/run.json)
and its checkpoint evaluation now supply the reported metrics. The
[post-recovery preservation check](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/diagnostics/d039_preservation_after.json)
verified **152 files unchanged**, including the twelve earlier completed fits'
recorded artifacts, the recovered fit's training history/specification/freeze
check/training predictions, and the original protocol and verification evidence.

At the **15:51 UTC historical recovery snapshot**, the package had
**13 complete fits**. The active fourteenth fit was
`attention_cls_skip_adapted/cs_adapted`; `attention_cls_skip_original/pretrained`
and `attention_cls_skip_original/cs_adapted` were still unstarted. No unresolved
fit or new continuation error was recorded. The original memory/read incidents
and provisional assurance remain applicable after package completion.

At **16:06 UTC (17:06 BST)**, the fourteenth fit had completed **6/50 epochs**
and continued without a recorded error. During verification of the earlier
completed fits, one saved-file read recovered on its second hash attempt;
the original digest matched after per-file cache advice. The
[integrity retry log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1/diagnostics/checkpoint_integrity.jsonl)
records this recovery without changing the checkpoint or stopping the queue.

## Historical verification incidents and superseded policies

The sections below describe the earlier strict and D037 exploratory attempts,
including the requirements active at that time. They are preserved as incident
history; D038 with operational amendment D039 above is the current acceptance
and execution contract.

### Preflight checkpoint-read incident

The first training-only GPU attempt stopped after its first gradient check
because the original Hugging Face checkpoint no longer read with its pinned
SHA256. Repeated reads returned differing digests despite unchanged file size,
inode and modification/change timestamps; an independent checksum utility
confirmed a mismatch. Advising eviction of this file's cached pages through a
read-only descriptor restored the expected digest, which then passed two
additional full reads. No checkpoint bytes were written and no reboot occurred.
The two changed 1 MiB read regions were inside the text-token embedding tensor,
which this online visual-only model discards. This evidence identifies a cached
read inconsistency, not its underlying hardware/software cause.

The failed verification attempts are retained. The subsequent verification checks
the pinned source before/after model construction and after each GPU step;
successful verification is bound to source, configuration, inputs and runtime.
No comparative validation result was produced by the failed attempt. Training
must stop on renewed hard integrity failures. The strict study accepts no changed
frozen values; the subsequent, separate D037 exploratory policy is the sole
exception for bounded and recorded tiny frozen-value deviations.

The recurrence occurred after constructing the second complete image/text model,
before its GPU step. The final loader therefore constructs only the required
visual architecture and materializes its tensors. Strict key matching, exact
tensor equality and independent parameter storage are checked during loading;
there is no automatic cached-page repair in the production loader. Three
consecutive constructions (pretrained, CS-adapted, pretrained) each reproduced
all 342 source tensors, historical CPU image outputs and both transforms
bit-for-bit, with the full pinned source hash unchanged after each construction.
The comparison used one training image; the 16-case GPU verification supplies
the separate gradient/numerical checks. Evidence:
[loader comparison](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_v1/verification/loader_check.json).

### Exact-resume integrity incident

The next attempt passed all 16 gradient/scope/source-hash checks, but the real
two-epoch resume comparison failed. Exactly one FP32 scalar in the **frozen**
`trunk.blocks.5.attn.proj.weight[1080,539]` differed by bit `0x40` (absolute
difference `3.725290298461914e-9`). Uninterrupted weights matched the original
adapted checkpoint; resumed weights contained the differing bit. All other
visual tensors, fusion weights, optimizer state and RNG states matched exactly.
FP16 training metrics matched, while FP32 prediction loss detected a small
difference. File hashes and the differing saved value remained stable after
repeated reads and read-only file-cache eviction.

This was an actual saved frozen-weight discrepancy, not safetensors header
ordering or an accepted numerical tolerance. The gate stopped before comparative
training. The precise origin along the memory/copy/checkpoint path remains
unresolved. The implementation now reauthenticates the starting checkpoint and
checks every visual tensor after initial device transfer. It records the initial
frozen-parameter digest, checks the exact CPU snapshot before saving, and verifies
frozen parameters plus every visual/fusion tensor after checkpoint loading.
Tests inject single-bit corruption into initial transfer, checkpoint staging,
hash-valid saved weights and restored trainable/frozen tensors; each is rejected.
A fresh GPU verification must pass before comparative fitting. No relevant
kernel hardware-error entries or EDAC counters were available, and the machine's
RAM/CPU tuning status is unconfirmed; this does not establish a hardware cause
or prove that the underlying issue has been resolved. Evidence:
[failed resume record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_v1/verification/7cce6d6795b4/failure.json).

### Blocking host-memory evidence and handoff

The hardened attempt then failed for `raw_concat/pretrained` immediately after
one GPU training step: its frozen-parameter digest changed despite successful
starting-source and post-transfer checks. No checkpoint had yet been saved for
that check. The prior two image-only checks passed. See the
[step-integrity failure](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_v1/verification/e5c6ae836dd4/failure.json).

A bounded [CPU memory probe](../../tools/probe_m3j_cpu_memory.py) allocates its own
4 GiB buffer, writes fixed/address-dependent patterns and compares them with
freshly computed expected values. It uses NumPy and Python's standard library;
it does not load Torch, CUDA, models or checkpoints. The mismatch detector first
passes a separate injected-bit positive control. Both executions stopped on
one unexpected bit in the inverse-address pattern:

| Probe | Time to stop | Result |
|---|---:|---|
| [First](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_v1/verification/diagnostics/cpu_memory_probe_20261019.json) | 3.61 seconds | One erroneous 64-bit value; XOR `0x0000004000000000` |
| [Repeat](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_v1/verification/diagnostics/cpu_memory_probe_repeat_20261019.json) | 3.38 seconds | Another single-bit mismatch at a different buffer location |

This provides positive evidence of host-memory/CPU-path instability independent
of the M3J model code. It does **not** identify a particular DIMM, memory
controller, CPU setting or other faulty component. A separate
[8 GiB GPU memory/transfer probe](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_v1/verification/diagnostics/gpu_memory_probe.json)
found no errors across its five patterns in 91 seconds; that limited negative
result does not invalidate the host-memory failures or establish GPU health.

No driver, firmware, BIOS, clock or voltage setting was changed. No reboot,
installation, historical checkpoint modification or full scientific fit was
performed. The user does not know whether CPU/RAM tuning is enabled. The machine
reports Ryzen 9 7900X and approximately 32 GB RAM; available kernel logs provided
no relevant machine-check/ECC/Xid entries, and EDAC counters were unavailable.

**Strict-study recovery:** have the machine's memory/CPU path diagnosed and
stabilized, then rerun its complete `verify` command. This remains necessary for
restoring the strict study. The user's subsequent exploratory exception is
documented below and does not erase this evidence. The current strict-study
[execution status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_v1/orchestration/seed1_joint_partial/status.json)
records the blocker and zero completed fits. No M3/M3J performance conclusion
can be drawn from the failed preflight checks.


## Historical exploratory tolerance amendment D037

**Superseded by D038.** The following records the previously authorized policy
and its failed attempts. The active runner does not accept its configuration;
these thresholds and acceptance requirements do not apply to new fits.

The user explicitly authorized **“Allow tiny frozen-weight changes in an explicit
exploratory mode.”** That implementation used
`attention_fusion_joint_exploratory_v1` and its own artifact root. At that time,
the default configuration and `attention_fusion_joint_v1` remained strict; the earlier failed
verification receipts and diagnosis records remain unchanged. This amendment
permits a bounded engineering exploration under unresolved memory instability,
not a claim that the machine is healthy or the results exactly reproducible.

The opt-in [configuration](../../configs/multimodal_attention_fusion_joint_exploratory.yaml)
predeclares these limits before any comparative training:

| Check | Exploratory rule |
|---|---|
| Frozen scalar deviation from its original authenticated baseline | Both `abs(observed − baseline) <= 1e-8` **and** `abs(observed − baseline) <= 1e-5 * abs(baseline)` must hold; an original zero must remain exact |
| Changed frozen coordinates | At most **8 distinct coordinates per fit**, counted cumulatively across the fit and preserved over resume; a return to the original value does not reset the count |
| Frozen deviation events | At most **16 distinct coordinate/value deviations per fit**; repeated observation of the same persistent difference counts once, while a new non-baseline value at the same coordinate counts as another deviation |
| Floating recomputation/replay | `abs(observed − reference) <= 1e-6 + 1e-5 * abs(reference)` for each applicable finite floating value; this does not relax authenticated loading/copying |
| Assurance | Every exploratory declaration, run, saved checkpoint/provenance, status and report must carry `provisional_unresolved_memory_integrity`, including runs with no observed deviation |

The frozen baseline does not move after a tolerated deviation. Otherwise many
individually small steps could accumulate into an unrestricted drift. Counts and
deviation records belong to the fit, survive checkpointing/resume, and are not
reset at epoch boundaries. A numerical pass must never overwrite or silently
repair the observed tensor with its expected value. Retain the actual values and
record the tensor name, coordinate, baseline, observed value, absolute/relative
difference and check location so that accepted deviations remain reviewable.

These are prospective engineering limits chosen to admit only small, sparse
observed discrepancies. They do not bound all possible effects of undetected
corruption on training and are not empirically established accuracy guarantees.
There is no automatic threshold expansion after a failure. Exceeding a value or
count limit, encountering non-finite values or failing a hard check stops the
fit and preserves its failure evidence.

**Hard checks remain exact:** pinned source and stored-object hashes; expected
keys, shapes and dtypes; image/geo/label/plot identities; source-to-model and
checkpoint-to-model copies; parameter scope; RNG states; scheduler and discrete
optimizer/epoch/step state (including tensor-encoded optimizer steps); optimizer
parameter-group settings; predicted labels, top-3 indices, confusion counts and
derived classification metrics. Recomputed losses, logits/probabilities, floating
model weights and optimizer moments may use the declared replay bounds.
Tolerance is restricted to the documented frozen baseline checks and recomputed
floating values. It is not a bypass for file
corruption, a different starting model or failed checkpoint restoration.

Keep the 50-epoch optimization recipe, initializations, model matrix, inputs,
selection rules and original paired RNG namespaces unchanged. The separate
identity prevents exploratory verification from authorizing the strict study
or replacing its artifacts. The exploratory runner must pass the complete
policy-specific training-only GPU gate with the current source/configuration,
input and runtime identities before any of the 16 authorized full fits starts.
Passing that gate authorizes only this provisional execution path.

The historical workflow selected its separate exploratory configuration
explicitly. Use the active default configuration and commands above for D038;
the retained exploratory YAML documents prior policy rather than a current
execution option.

All live and final reports, comparison tables and exported metadata must state
that machine-memory integrity remains unresolved, disclose accepted deviations
and retain both final and best-validation policies. Do not present exploratory
metrics as verified gains or promote them to the completed strict study. Any
promising outcome should be repeated on a machine that passes the strict checks
before a dependable scientific conclusion is made.

**Outcome before D038:** implementation and review were complete, with **201 focused
tests passed** across comparison/audit, protocol, input, checkpoint, training,
reporting, CLI and reused architecture checks. Policy-specific GPU verification
failed on inconsistent checks of the same frozen snapshot; no exploratory full fit has started. No driver/firmware setting
change or reboot is part of this amendment.

The cumulative audit is authenticated and embedded in each checkpoint manifest.
Resume requires its committed history to remain an exact prefix of the current
audit, preventing an older audit from resetting the limits. Portable exports
retain the original frozen-reference dependency and full deviation evidence;
final replay evidence is committed before a fit is marked complete.

### Exploratory GPU verification outcome

The [verification log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_exploratory_v1/orchestration/verification_20261019.log)
and [failed receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_exploratory_v1/verification/86e258e01db9/failure.json)
record this attempt. Development input authentication passed. Both `image_only`
initialization checks passed with unchanged frozen parameters. The third check,
`raw_concat/pretrained`, stopped **before its optimizer step** because the frozen
parameter digest changed between the live reference read and its CPU snapshot.
The error was `Initial frozen reference changed during snapshot`.

This is an exact baseline-construction check, not a measured discrepancy that
exceeded the new tolerance. This receipt does not identify the affected tensor,
number of changed scalars or magnitude; it therefore cannot establish whether
the discrepancy was small enough to accept. The known earlier `3.73e-9` example
is accepted in the injected-drift tests, but that does not characterize this new
incident. The implementation did not relax its bounds or replace the expected
reference after the failure. Resume/export verification was not reached, no
comparative validation or test score was produced, and **0/16 full fits started**.
The [execution status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_exploratory_v1/orchestration/seed1_joint_partial/status.json)
records that attempt's remaining block. No comparative training was started
under that failed policy-specific receipt.

### Canonical-source reference refinement and second verification

The original exploratory attempt used a live-model copy to construct its frozen
reference. This was replaced with a stronger source-defined reference: frozen
values now come directly from the declared original checkpoint, authenticated
by its pinned SHA before and after reading. The stored frozen subset must match
its canonical digest exactly, and an independent source reread must agree.
Production rejects missing source provenance. Source-to-model initialization
copies remain exact. A later small live-model change is measured against that
fixed source reference; no tolerance or cumulative limit was enlarged. Tests
verify acceptance of the original `3.73e-9` injected change without modifying the
model or moving the canonical baseline, and rejection of missing/changed sources.

The [second log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_exploratory_v1/orchestration/verification_source_reference_20261019.log)
and [second failed receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_exploratory_v1/verification/5ee0daedb124/failure.json)
record a new outcome. Input authentication and `image_only/pretrained` passed.
For `image_only/cs_adapted`, reference construction completed, but checks after
one training step were inconsistent: the first digest of the captured CPU
snapshot differed from the canonical digest; the element comparison reported
exact equality; and a repeated digest differed from the first digest. The check
stopped with `Frozen snapshot changed while its deviation was being measured`.

This establishes inconsistent checking/reading of the captured state, not a
quantified stable scalar deviation or proof of its physical cause. The
[comparison audit](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_exploratory_v1/verification/5ee0daedb124/gradient_audits/seed1/joint_partial/image_only/cs_adapted/numerical_audit.json)
records the exact element comparison; the enclosing failed receipt is
authoritative and prevents treating that isolated comparison as a passed gate.
No assumption was made that the unmeasured inconsistency fits the tolerance,
and no automatic retry, repair or threshold expansion followed this failure.
The completed implementation has **201 focused tests passed**; resume/export
GPU checks remain unreached, **0/16 full fits have started**, and no comparative
validation/test result is claimed. No reboot or system-setting change occurred.

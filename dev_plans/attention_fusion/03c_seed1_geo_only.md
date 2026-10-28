# M3J — Seed-1 geo-only package

Status: **The geo-only fit completed all 50 epochs and final/best checkpoint
evaluations at 2026-10-22 10:17:19 UTC (11:17 BST). Seed 1 is complete at 33/33
fits**: 16 partial-joint, 16 frozen-online and this shared standardized-geo
reference. Verification passed 214 CPU checks and the training-only
GPU/model/checkpoint checks. The combined report and queue completed at
**10:18:23 UTC (11:18 BST)** with 33 complete and zero incomplete fits, no queue
error and no unresolved checkpoint. The final artifact audit is linked below.

The [governing guide](README.md) and
[M3J recipe](03_joint_finetuning.md#scope-and-fixed-recipe) remain unchanged.
This is one fit, shared across both image initializations and both image-encoder
regimes; it is not four duplicate fits. It has no image encoder or image input.
Later seeds and cleaned-test application remain separate packages.

## Scope and fixed recipe

| Setting | Contract |
|---|---|
| Seed and fit count | Seed 1; one fresh geo-only fit |
| Data | Same 3,378 training / 822 validation rows and fixed 18-class mapping |
| Input | Fixed 64-dimensional 100 m GSE embedding |
| Geo preprocessing | Coordinate standardization fitted only on the development training rows; reuse the matched scaler definition and provenance |
| Architecture | Standardized geo → Linear(64, 256) → ReLU → Dropout(0.1) → Linear(256, 18) |
| Trainable parameters | Classifier only; no encoder, modality adapter, attention or image embedding |
| Duration | 50 epochs, no early stopping |
| Batch size | Physical and effective batch 16; no gradient accumulation; retain the final smaller batch |
| Optimizer | Adam; initial LR `5e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`, AMSGrad off; no gradient clipping |
| Schedule | Cosine `T_max=50`, minimum zero, step at the end of each epoch; no warm-up or restarts |
| Training arithmetic | FP32 weights and Adam state, CUDA FP16 autocast, no GradScaler |
| Validation / checkpoint evaluation | Evaluation mode, no gradients, FP32 inputs/weights, autocast disabled |
| Backend policy | cuDNN TF32 on; matmul TF32 off; float32 matmul precision `highest` |
| Reporting | Acc@1, Acc@3, weighted F1, MCC, losses, per-class/confusion analysis and epoch curves |
| Saved policies | Final epoch 50 and best validation by Acc@1, weighted F1, MCC, then earliest epoch for a tie |

Both checkpoint policies come from this one trajectory. Keep epoch selection
provenance and reloaded-checkpoint evaluation metrics separate. Retain reusable
final/best weights, the scaler, class mapping, input identities and checkpoint
manifest; checkpoint policy does not multiply the fit count. The original M3
geo-only results used different learning rates and cannot substitute for this
M3J fit.

## Execution and reproducibility

Use the dedicated
[configuration](../../configs/multimodal_attention_fusion_joint_geo_only.yaml) and study identity
`attention_fusion_joint_geo_only_direct_v1`. Store protocol, source snapshot,
verification and execution outputs below:

```text
multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/
  protocol/                         # configuration, sources, inputs and references
  verification/                     # CPU and training-only GPU evidence
  orchestration/seed1_geo_only/      # launch, log and status
  runs/seed1/geo_only/               # history, curves, metrics and checkpoints
  objects/                          # immutable weights and retained state
  reports/seed1_geo_only/            # combined 33-fit report and comparisons
```

The fit metadata uses `regime: geo_only`, `candidate_id: geo_only` and
`initialization: not_applicable`. The combined report reads the existing 32 image
fits and counts the new geo-only fit once. Its shared-reference contrasts do not
create duplicated geo-only rows or new image fits.

Preserve the completed `attention_fusion_joint_direct_v1` and
`attention_fusion_joint_frozen_online_direct_v1` packages. Read their metadata
and results as comparison references; do not rewrite their configuration,
protocol, verification receipts, original source snapshots or saved fits.

The training/inference path loads only geo vectors, row identities and labels.
The shared allocation audit still checks referenced image-file existence as
metadata; it does not decode images or authenticate their unused raw bytes.
Training and inference must not
instantiate SigLIP2, decode images, build visual weight references or require
image weights for inference. Keep the established classifier initialization and
dropout namespaces, and explicitly use the declared
`training.shuffle_seed_namespace: attention_fusion_joint_v1` for sample order.
Image crop/augmentation streams are inapplicable.

Carry forward D038/D039: finite training and meaningful classifier updates,
valid split/label identities, usable saved final/best checkpoints, atomic
storage and bounded saved-file authentication retries. Direct frozen-visual
checks have no target in a geo-only model. Do not introduce mandatory
interrupted-versus-uninterrupted comparison, automatic optimizer-state resume,
exact prediction replay, live-weight checksum gates or numerical tolerances.
Preserve any failed receipt, and report a persistent saved-file verification
failure rather than treating it as a completed result. No reboot or system
setting change is authorized.

## D041 loader audit and comparison limitation

While preparing this package, code inspection found that the completed image
loader used `study_id` for shuffle and crop randomness instead of the declared
common namespaces. The partial-joint and frozen-online studies have different
IDs. Their data allocation, starting encoders, architecture recipes and
preprocessing rules match, but their realized minibatch orders and training
image views were not paired across regimes. Within each package, the shared
namespace still pairs its corresponding arms.

Preserve both completed image packages and qualify their comparisons: their
observed difference does not isolate encoder trainability under identical views
and orders. Do not repair the historical results by rewriting source snapshots
or rerunning training within this package. The new geo-only loader explicitly
uses the declared shuffle namespace; it has no image views to pair. This audit
finding does not change the geo-only scientific recipe or authorize expansion
of the experiment matrix.

## Deliverables and completion conditions

1. A dedicated seed-1 geo-only scope and configuration with source/input binding.
2. A no-image loading/training/inference path, verified with short training-only
   checks, classifier parameter updates and saved-checkpoint loading/export.
3. One completed 50-epoch fit with epoch histories and training/validation curves.
4. Final and best-validation checkpoints, reloaded evaluations, predictions,
   confusion outputs and reusable dependency manifests.
5. A seed-1 report using the single geo-only result as the shared reference for
   both initialization arms and both regimes, keeping final/best policies
   separate and existing 32 fits unchanged.
6. A completion record bringing seed 1 to **33/33 fits** and **1,650 training
   epochs**, with any failure or unresolved artifact explicitly reported.

All results use the 822-image development validation set. They remain
single-seed descriptive and retain `provisional_unresolved_memory_integrity`;
completion does not establish machine health, cross-seed supported improvement
or independent confirmation. M3J overall continues with separately authorized
later-seed packages.

## Commands

Run from the repository root in the `habcls` environment. These commands target
only the one authorized seed-1 geo-only fit; later-seed runs are outside its scope.

```bash
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only.yaml verify
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only.yaml --background continue
python tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only.yaml status
python tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only.yaml report
```

`continue` starts the fresh fit, preserves an already completed fit, or retries
only saved-checkpoint evaluation after all 50 epochs. It does not automatically
resume optimizer updates or rerun a partially trained attempt.

## Implementation and execution evidence

### Initial input-audit correction

The first preparation attempt inherited the image package's raw-image inventory
check and failed with `Development raw image bytes differ from the historical
protocol fingerprint`. No comparative fit started. The
[failure record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/verification/d041_initial_input_audit_failure.json)
is retained. The cause is unknown; this report alone does not prove that source
image bytes were changed.

The geo-only input path now omits the unused raw-image content inventory, image
decoding, encoder weights and image caches. It retains the source geo vectors,
split assignments, row/label identities, training-only scaler and their
provenance checks. Comparison with prior package inputs excludes image-only
fields. The allocation audit still checks referenced image-file existence as
metadata. This is a correction to the no-image package's input dependencies;
it does not accept altered geo inputs or change the completed image packages.

### Verification and launch

The [CPU receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/verification/d041_cpu_tests.json)
and [test results](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/verification/d041_cpu_tests.xml)
record **214 passed checks**. The
[GPU receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/verification/6e0a1ac74470_1792664164260256353/verification.json)
passed at **2026-10-22 10:16:07 UTC**, covering short training-only classifier
updates, no visual model, usable final/best checkpoints and standalone export.
These checks do not reinstate interrupted-run or exact prediction replay gates.

The [launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/orchestration/seed1_geo_only/launch.json)
records the background `continue` command at
**2026-10-22 10:16:33 UTC (11:16 BST)**, process **598108**. The
[persistent log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/orchestration/seed1_geo_only/run_1792664193759918141.log)
and [queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/orchestration/seed1_geo_only/status.json)
track the single `seed1/geo_only` fit. It starts afresh at epoch 1; both existing
image packages remain preserved. At the launch snapshot the fit was active,
with no queue error or unresolved checkpoint. The combined report follows
completion of its 50 epochs and both saved-checkpoint evaluations.


## Seed-1 completion and results

The [run record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/runs/seed1/geo_only/run.json)
records all **50 epochs complete at 2026-10-22 10:17:19 UTC (11:17 BST)**,
including final/best saved-checkpoint evaluations. The best checkpoint was
selected at **epoch 11** from the original epoch validation history. The
[reported metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/runs/seed1/geo_only/metrics.json)
come from loading and evaluating the retained checkpoints on the 822-image
development validation set:

| Policy | Epoch | Acc@1 (%) | Acc@3 (%) | Weighted F1 (%) | MCC |
|---|---:|---:|---:|---:|---:|
| Final | 50 | 63.02 | 89.17 | 61.23 | 0.5730 |
| Best validation | 11 | 64.84 | 89.17 | 62.54 | 0.5947 |

This brings seed 1 to **33/33 fits**, **1,650 training epochs** and **66 final/best
checkpoint-policy evaluations**. The shared geo-only fit is counted once, and
its result is reused across both image initializations and encoder regimes.
The [seed-1 findings report](03d_seed1_findings.md) interprets encoder
initialization, fusion architecture, attention readout and unimodal comparisons
across all 33 fits, with final and best-validation results kept separate.
The [combined report summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/summary.json)
records 33 complete and zero incomplete fits. The queue and combined report
completed at **10:18:23 UTC (11:18 BST)**, with no error or unresolved checkpoint.
The [completion audit](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/completion_audit.json)
confirms 33 fits, 1,650 epochs, 66 checkpoint-policy evaluations and the combined
report figures (132 PNG and 132 PDF confusion plots). The
[preservation check](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/verification/d041_preservation_after.json)
passed for all **1,237 tracked files** from the two preceding packages.

| Deliverable | Location |
|---|---|
| Combined seed-1 report | [Report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/report.md) with final/best tables, shared geo-only comparisons, regime contrasts and per-class/confusion outputs |
| Geo-only training curves | [Curve directory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/runs/seed1/geo_only/curves/) and [epoch history](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/runs/seed1/geo_only/history.json) |
| Reusable final/best weights | [Checkpoint manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/runs/seed1/geo_only/checkpoint_manifest.json), immutable referenced objects, preprocessing and label mapping |
| Validation predictions and evaluation evidence | [Geo-only run directory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/runs/seed1/geo_only/) |

No cleaned-test data were scored. The results remain provisional and
single-seed descriptive, with the completed image-package shuffle/crop-stream
limitation explicitly retained in the combined report. This completes seed 1;
M3J overall is still in progress. Seed 2 is the next separate package under D034,
starting with its 16 partial-joint fits before frozen-online and geo-only.


A subsequent
[reporting-label amendment](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/verification/d041_reporting_label_amendment.json)
clarifies that geo-only training curves reflect dropout, not image augmentation.
It passed **38 reporting checks** and regenerated only curves and the combined
report. Training, selected epochs, model weights and original CPU/GPU receipts
are unchanged. This rendering correction is recorded separately from the
source-bound fit and does not create another training run.

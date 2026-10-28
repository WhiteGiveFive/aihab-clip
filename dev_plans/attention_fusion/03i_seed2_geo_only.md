# M3J — Seed-2 geo-only package

**D048, 2026-10-25. Status: complete.** The shared geo-only fit completed all
50 epochs and final/best saved-checkpoint evaluations at **09:24:33 UTC (GMT)**.
The combined report and queue completed at **09:25:06 UTC (GMT)**, bringing
seed 2 to **17/17 active fits**. Its 16 frozen-online fits remain deferred under
[D047](03h_frozen_online_deferral.md). No queue error or unresolved checkpoint
was recorded.

## Scope and fixed recipe

| Setting | Contract |
|---|---|
| Seed and fit count | Seed 2; one fresh geo-only fit, shared across both image initializations |
| Data | Same 3,378 training / 822 validation rows and fixed 18-class mapping |
| Input and preprocessing | Fixed 64-dimensional 100 m GSE embedding; coordinate standardization fitted on development training rows only |
| Architecture | Standardized geo → Linear(64, 256) → ReLU → Dropout(0.1) → Linear(256, 18) |
| Trainable parameters | Classifier only; no image encoder, image input or modality adapter |
| Duration and batch | 50 epochs, no early stopping; physical/effective batch 16, no gradient accumulation, retain the final smaller batch |
| Optimizer | Adam, initial LR `5e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, AMSGrad off, no gradient clipping |
| Schedule | Cosine `T_max=50`, minimum zero, end-of-epoch stepping; no warm-up or restarts |
| Arithmetic | FP32 weights/Adam state, CUDA FP16-autocast training without GradScaler; autocast-disabled FP32 validation; cuDNN TF32 on, matmul TF32 off, float32 matmul precision `highest` |
| Random streams | Seed 2 with `loader_rng_policy: declared_namespaces_v1`; declared `attention_fusion_joint_v1` shuffle/dropout namespaces; image augmentation inapplicable |
| Checkpoints | Retain final epoch 50 and best validation by Acc@1, weighted F1, MCC, then earliest epoch; reload and evaluate both |
| Reporting | Acc@1, Acc@3, weighted F1, MCC, losses, per-class/confusion outputs and epoch curves |

The [dedicated configuration](../../configs/multimodal_attention_fusion_joint_geo_only_seed2.yaml)
retains the established M3J recipe. The existing M3 geo fits used different
learning rates and do not replace this fit. Final/best policies are evaluations
of one trajectory, not two fits.

## Dependencies, storage and verification

The only required completed predecessor is the
[seed-2 partial-joint package](03g_seed2_partial_joint.md). Its 16 saved fits
remain unchanged. The frozen-online root is explicitly deferred metadata, not
a prerequisite and not an incomplete training result. The combined report
must distinguish **17 active fits**, **16 deferred fits**, and **33 fits in the
full seed-2 plan**. No frozen-online scores or regime contrasts are inferred.

Study identity: `attention_fusion_joint_geo_only_seed2_direct_v1`.

```text
multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/
  protocol/                         # source/configuration/input/reference binding
  verification/                     # package-specific GPU/checkpoint evidence
  orchestration/seed2_geo_only/      # launch, log and queue status
  runs/seed2/geo_only/               # history, curves, predictions and metrics
  objects/                          # retained checkpoint objects
  reports/seed2_geo_only/            # combined active-scope report and audit
```

D047's [preparation evidence](03h_frozen_online_deferral.md#verification-and-preparation-evidence)
records 363 passing focused CPU tests, unchanged scientific settings and a
passed seed-2 input/recipe/completion binding. This launch requires its own
training-only GPU model/update and checkpoint loading/export verification.
The geo-only execution path neither constructs a visual encoder nor decodes
images. Shared allocation checks retain image-file existence as metadata.

Carry forward D038/D039 practical checks and bounded saved-file recovery. Do
not introduce interrupted-versus-uninterrupted tests, exact prediction replay,
numerical tolerances or repeated live-weight checksum gates. A completed
50-epoch trajectory may retry saved-checkpoint evaluation without retraining;
automatic optimizer-state resume remains disabled. No reboot or system-setting
change is part of this package.

## Commands

Run from the repository root in the `habcls` environment:

```bash
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only_seed2.yaml --device cuda verify
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only_seed2.yaml --device cuda --background run
python tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only_seed2.yaml status
```

The recorded launch used `run` for the fresh fit. If recovery is needed,
`continue` preserves a completed trajectory while finishing saved-checkpoint
evaluation; it does not automatically resume optimizer updates or rerun a
partial attempt.

## Execution evidence and completion

The [GPU receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/verification/latest.json)
passed at **2026-10-25 09:06:25 UTC (GMT)** on the RTX 4090. It records a finite
training step, gradients for all trainable classifier parameters and a
meaningful classifier update, followed by a fresh checkpoint check and
standalone load/evaluation. No visual encoder was present; the frozen-visual
comparison was correctly inapplicable. This was training-only verification,
not a comparative validation result.

The [launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/orchestration/seed2_geo_only/launch.json)
records a fresh background run at **09:23:47 UTC (GMT)**, process **62013**.
The [persistent log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/orchestration/seed2_geo_only/run_1792920227893245528.log)
and [queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/orchestration/seed2_geo_only/status.json)
record completion without a queue error or unresolved checkpoint. The
[run record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/runs/seed2/geo_only/run.json)
records 50 completed epochs and both saved-checkpoint evaluations at
**09:24:33 UTC**. The best checkpoint was selected at **epoch 19**.

The [metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/runs/seed2/geo_only/metrics.json)
come from loading and evaluating the retained checkpoints on the development
validation set:

| Policy | Epoch | Acc@1 (%) | Acc@3 (%) | Weighted F1 (%) | MCC |
|---|---:|---:|---:|---:|---:|
| Final | 50 | 65.94 | 91.00 | 63.91 | 0.6068 |
| Best validation | 19 | 67.76 | 89.17 | 65.54 | 0.6279 |

The [combined report summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/reports/seed2_geo_only/summary.json)
records **17 complete active fits**, **zero incomplete fits**, and **16 deferred
fits**. Seed 2 now has **850 training epochs** and **34 final/best
checkpoint-policy evaluations** in its active scope. The full 33-fit seed-2
design remains incomplete because frozen-online is deferred. M3J overall is
at **50/84 active fits**, with **34 active fits remaining** and **48 deferred
frozen-online fits**.

The [completion audit](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/reports/seed2_geo_only/completion_audit.json)
confirms the 17 fits, 850 epochs, 34 checkpoint-policy evaluations, retained
outputs and 68 PNG/68 PDF confusion figures. The completed nine-panel training
curve was visually inspected, including its 50-epoch trajectory and epoch-19
best marker. The
[preservation check](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/verification/d048_preservation_after.json)
passed for all **2,202 pre-existing completed-package files**, comparing
presence, size and modification time without a model-weight hash audit.

| Deliverable | Location |
|---|---|
| Combined seed-2 report | [Report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/reports/seed2_geo_only/report.md), with final/best metrics, shared geo comparisons and confusion outputs |
| Geo-only epoch history and curves | [History](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/runs/seed2/geo_only/history.json) and [curve directory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/runs/seed2/geo_only/curves/) |
| Reusable final/best checkpoints | [Checkpoint manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/runs/seed2/geo_only/checkpoint_manifest.json), retained objects, class mapping and training-fitted scaler |
| Predictions and evaluation evidence | [Run directory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/runs/seed2/geo_only/) |

All results use the **822-image development validation set**. They retain
`provisional_unresolved_memory_integrity` and single-seed descriptive status;
passing execution checks does not establish machine health. Seed 1 retains
its historical shuffle/crop limitation; D045's prospective stream correction
applies to seed 2. No cleaned-test evaluation or router work is authorized here.

The next separate package is **16 seed-3 partial-joint fits**, followed by its
shared geo-only fit. This completion does not launch that package; frozen-online
for seeds 2–4 remains deferred.

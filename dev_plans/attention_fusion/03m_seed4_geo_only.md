# M3J — Seed-4 geo-only package

**D052, 2026-10-28. Status: complete.** The shared geo-only fit completed all
50 epochs and final/best saved-checkpoint evaluations at **09:31:51 UTC (GMT)**.
The combined report and queue completed at **09:32:25 UTC (GMT)**, bringing
seed 4 to **17/17 active fits** and M3J active fit execution to **84/84**. No
queue error or unresolved checkpoint was recorded. The 16 seed-4 frozen-online
fits remain deferred under [D047](03h_frozen_online_deferral.md); cross-seed
consolidation and selection remain separate development work.

## Scope and fixed recipe

| Setting | Contract |
|---|---|
| Seed and fit count | Seed 4; one fresh geo-only fit, shared across both image initializations |
| Data | Same 3,378 training / 822 validation rows and fixed 18-class mapping |
| Input and preprocessing | Fixed 64-dimensional 100 m GSE embedding; coordinate standardization fitted on development training rows only |
| Architecture | Standardized geo → Linear(64, 256) → ReLU → Dropout(0.1) → Linear(256, 18) |
| Trainable parameters | Classifier only; no image encoder, image input or modality adapter |
| Duration and batch | 50 epochs, no early stopping; physical/effective batch 16, no gradient accumulation, retain the final smaller batch |
| Optimizer | Adam, initial LR `5e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, AMSGrad off, no gradient clipping |
| Schedule | Cosine `T_max=50`, minimum zero, end-of-epoch stepping; no warm-up or restarts |
| Arithmetic | FP32 weights/Adam state, CUDA FP16-autocast training without GradScaler; autocast-disabled FP32 validation; cuDNN TF32 on, matmul TF32 off, float32 matmul precision `highest` |
| Random streams | Seed 4 with `loader_rng_policy: declared_namespaces_v1`; declared `attention_fusion_joint_v1` shuffle/dropout namespaces; image augmentation inapplicable |
| Checkpoints | Retain final epoch 50 and best validation by Acc@1, weighted F1, MCC, then earliest epoch; reload and evaluate both |
| Reporting | Acc@1, Acc@3, weighted F1, MCC, losses, per-class/confusion outputs and epoch curves |

The [dedicated configuration](../../configs/multimodal_attention_fusion_joint_geo_only_seed4.yaml)
retains the established M3J recipe. The existing M3 geo fits used different
learning rates and do not replace this fit. Final/best policies are evaluations
of one trajectory, not two fits.

## Dependencies, storage and verification

The only required completed predecessor is the
[seed-4 partial-joint package](03l_seed4_partial_joint.md). Its 16 saved fits
remain unchanged. The frozen-online root is explicitly deferred metadata, not
a prerequisite and not an incomplete training result. The combined report
must distinguish **17 active fits**, **16 deferred fits**, and **33 fits in the
full seed-4 plan**. No frozen-online scores or regime contrasts are inferred.

Study identity: `attention_fusion_joint_geo_only_seed4_direct_v1`.

```text
multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/
  protocol/                         # source/configuration/input/reference binding
  verification/                     # package-specific GPU/checkpoint evidence
  orchestration/seed4_geo_only/      # launch, log and queue status
  runs/seed4/geo_only/               # history, curves, predictions and metrics
  objects/                          # retained checkpoint objects
  reports/seed4_geo_only/            # combined active-scope report and audit
```

D047's [preparation evidence](03h_frozen_online_deferral.md#verification-and-preparation-evidence)
records 363 passing focused CPU tests and unchanged scientific settings.
This package refreshes its preparation and checks the completed same-seed
partial-joint reference. Its own training-only GPU model/update and checkpoint
loading/export verification precedes comparative fitting.
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
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only_seed4.yaml prepare
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only_seed4.yaml --device cuda verify
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only_seed4.yaml --device cuda --background run
python tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_geo_only_seed4.yaml status
```

The recorded launch used `run` for the fresh fit. If recovery is needed,
`continue` preserves a completed trajectory while finishing saved-checkpoint
evaluation; it does not automatically resume optimizer updates or rerun a
partial attempt.

## Execution evidence and completion

The [preparation manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/preparation/manifest.json)
was refreshed at **09:30:08 UTC (GMT)** on 2026-10-28, preserving its earlier
receipt in `preparation/history/manifest_1793179808969095573.json`. It binds the
same inputs and unchanged recipe to the completed same-seed partial-joint
reference. The [CPU evidence record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/preparation/d052_cpu_evidence.json)
reuses D047's **363 passing focused checks** because the corresponding 27
execution sources are unchanged; this package does not claim a new CPU test run.

The [GPU receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/verification/latest.json)
passed at **09:30:40 UTC (GMT)**. The training-only classifier update and fresh
checkpoint loading/export check passed, including standalone evaluation. No
visual encoder was present, so frozen-visual comparison was inapplicable.
Verification scored no comparative validation data and introduced no
interrupted-run comparison or exact prediction replay.

The [launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/orchestration/seed4_geo_only/launch.json)
records a fresh background run at **09:31:05 UTC (GMT)**, process **316572**.
The [persistent log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/orchestration/seed4_geo_only/run_1793179864990894570.log)
and [queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/orchestration/seed4_geo_only/status.json)
record completion at **09:32:25 UTC (GMT)** without a queue error or unresolved
checkpoint. The [run record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/runs/seed4/geo_only/run.json)
records **50 completed epochs** and both saved-checkpoint evaluations at
**09:31:51 UTC (GMT)**, with **epoch 9** selected as best validation. Training
and checkpoint evaluation took **44.24 seconds**.

The [metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/runs/seed4/geo_only/metrics.json)
come from loading and evaluating retained checkpoints on the development
validation set:

| Policy | Epoch | Acc@1 (%) | Acc@3 (%) | Weighted F1 (%) | MCC |
|---|---:|---:|---:|---:|---:|
| Final | 50 | 65.45 | 90.27 | 63.46 | 0.6012 |
| Best validation | 9 | 66.06 | 88.93 | 63.38 | 0.6098 |

The [combined report summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/reports/seed4_geo_only/summary.json)
records **17 complete active fits**, **zero incomplete fits**, and **16 deferred
fits**. Seed 4 now has **850 training epochs** and **34 final/best
checkpoint-policy evaluations** in its active scope. The full 33-fit seed-4
design remains incomplete because frozen-online is deferred. M3J active fit
execution is **84/84 complete**, with **4,200 training epochs**, **168
checkpoint-policy evaluations**, and **48 deferred frozen-online fits** in the
full 132-fit design. These counts include all 33 seed-1 fits and 17 each for
seeds 2–4; the two checkpoint policies are evaluations of the same trajectories.

The [completion audit](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/reports/seed4_geo_only/completion_audit.json)
confirms the 17 fits, 850 epochs, 34 checkpoint-policy evaluations, retained
checkpoint objects and **68 PNG/68 PDF confusion figures**. The completed
nine-panel training curve was visually inspected, including its 50-epoch
trajectory and epoch-9 best marker. An independent review confirmed finite,
aligned **822 × 18** prediction arrays for both geo checkpoint policies and
retained reusable checkpoints. The audit also records all nine active package
queues complete, totaling **84/84 active fits**.

The [preservation check](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/verification/d052_preservation_after.json)
passed for all **3,894 pre-existing completed-package files**, comparing
presence, size and modification time without a model-weight hash audit.
Scientific configuration and runtime sources remain unchanged.

| Deliverable | Location |
|---|---|
| Combined seed-4 report | [Report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/reports/seed4_geo_only/report.md), with final/best metrics, shared geo comparisons and confusion outputs |
| Geo-only epoch history and curves | [History](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/runs/seed4/geo_only/history.json) and [curve directory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/runs/seed4/geo_only/curves/) |
| Reusable final/best checkpoints | [Checkpoint manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/runs/seed4/geo_only/checkpoint_manifest.json), retained objects, class mapping and training-fitted scaler |
| Predictions and evaluation evidence | [Run directory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/runs/seed4/geo_only/) |

All results use the **822-image development validation set**. They retain
`provisional_unresolved_memory_integrity` and single-seed descriptive status;
passing execution checks does not establish machine health. Seed 1 retains
its historical shuffle/crop limitation; D045's prospective stream correction
applies to seed 4. No cleaned-test evaluation or router work belongs to this
package.

## Handoff after active fit execution

The next development work is **cross-seed consolidation and selection**, using
the declared contrasts, separate final/best policies, uncertainty and
selection/refit contract. Include all four partial-joint seeds and all shared
geo-only references; retain seed-1 frozen-online results with their documented
shuffle/crop limitation. The missing frozen-online seeds prevent a four-seed
encoder-trainability comparison. Completing the active fit matrix alone does
not finish the M3J scientific development report or select a model for M4.

The **48 frozen-online fits** remain deferred. No M4 refit, cleaned-test
evaluation, router work or deferred training is automatically authorized by
this completion. Completed study configurations, artifacts and earlier
interpretation limits remain unchanged.

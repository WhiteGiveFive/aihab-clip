# M3J seed-3 partial-joint execution

**D049. Status: complete at 16:23:18 GMT on 2026-10-26.** All **16/16 fits**
completed 50 epochs and both retained final/best checkpoint evaluations:
**800 training epochs** and **32 checkpoint-policy evaluations**, with no
queue error or unresolved checkpoint. This closes the partial-joint package
at **66/84 active M3J fits complete**, with **18 active fits remaining** and
**48 frozen-online fits deferred**. The next shared seed-3 geo-only fit is
separately authorized under [D050](03k_seed3_geo_only.md).

**Historical launch snapshot, D049, 2026-10-25.** The user authorized the 16
seed-3 partial-joint fits. Package-specific preparation and training-only GPU
verification passed, and the fresh background queue launched at
**09:40:33 UTC (GMT)** with process **65206**. At this startup snapshot,
**0/16 comparative fits were complete**. The launch and interim progress
records below are preserved as historical snapshots.

M3J enters this package with **50/84 active fits complete**: all 33 seed-1 fits
and the 17 active seed-2 fits. The 48 frozen-online fits for seeds 2–4 remain
deferred under [D047](03h_frozen_online_deferral.md). Seed-3 geo-only is a
separate subsequent package.

## Scope and fixed recipe

Use [the seed-3 configuration](../../configs/multimodal_attention_fusion_joint_seed3.yaml)
and study identity `attention_fusion_joint_seed3_direct_v1`. Scope is exactly
seed 3 and `joint_partial`: eight architectures, each independently initialized
from original-pretrained or seed-3 training-partition CS-adapted vision weights.
No seed-2 joint-trained checkpoint supplies an initialization.

| Execution order | Architecture | Initializations |
|---|---|---|
| 1–2 | `image_only` | Pretrained, then CS-adapted |
| 3–4 | `raw_concat` | Pretrained, then CS-adapted |
| 5–6 | `projected_concat` | Pretrained, then CS-adapted |
| 7–8 | `tokenwise_capacity` | Pretrained, then CS-adapted |
| 9–10 | `attention_two_token` | Pretrained, then CS-adapted |
| 11–12 | `attention_cls` | Pretrained, then CS-adapted |
| 13–14 | `attention_cls_skip_adapted` | Pretrained, then CS-adapted |
| 15–16 | `attention_cls_skip_original` | Pretrained, then CS-adapted |

| Setting | Contract |
|---|---|
| Development data | Same 3,378 training / 822 validation rows and fixed 18-class mapping |
| Geo | Fixed 64D 100 m GSE embeddings; coordinate standardization fitted on development training rows only, including the original-embedding bypass |
| Image preprocessing | Historical `legacy_opencv_bgr_439_v1`; training random-resized crop to 384×384, deterministic validation resize; channel mean/std 0.5 |
| Trainable vision scope | Final ten Transformer blocks, 17–26, final normalization and attention-pooling head; 167,635,696 visual parameters trainable |
| Duration and batch | 50 epochs per fit, no early stopping; physical/effective batch 16, no accumulation, retain the final smaller batch |
| Optimizer | Adam, uniform initial LR `5e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, AMSGrad off and no gradient clipping |
| Schedule | Cosine `T_max=50`, minimum zero and end-of-epoch stepping; no warm-up or restarts |
| Arithmetic | FP32 weights/Adam state, CUDA FP16-autocast training without GradScaler; FP32 validation without autocast; cuDNN TF32 on, matmul TF32 off and float32 matmul precision `highest` |
| Random streams | D045 `loader_rng_policy: declared_namespaces_v1`; declared common `attention_fusion_joint_v1` crop/shuffle namespaces with seed 3 retained in each stream |
| Checkpoints | Option-3 shared storage; retain final epoch 50 and best validation by Acc@1, weighted F1, MCC, then earliest epoch; load and evaluate both |
| Monitoring | Every-epoch training/validation history and curves, loss, Acc@1, Acc@3, weighted F1, MCC, learning rates and resource timing |

This package contains **800 comparative training epochs** and **32 final/best
checkpoint-policy evaluations** when complete. Checkpoint policies refer to
one trajectory per fit, not additional fits. No cleaned-test scoring, router
application, frozen-online execution or later-seed training is included.

## Verification and practical recovery

[D045](03f_later_seed_preparation.md) prepared the seed-3 configuration and CPU
input record. The present package refreshed its preparation against current
execution sources at **09:32:57 UTC (GMT)** on 2026-10-25, preserving the
historical preparation evidence. The scientific recipe and inputs are
unchanged. The 363 passing focused CPU checks recorded by
[D047](03h_frozen_online_deferral.md) remain applicable because the corresponding
execution sources are unchanged; this launch does not claim a new CPU test run.

This package's own training-only GPU verification passed at
**09:39:28 UTC (GMT)**. All 16 architecture/initialization checks passed their
gradient, meaningful-update and practical frozen-value comparisons. A fresh
one-epoch training-only checkpoint check also passed loading and standalone
export/evaluation. The receipt records `validation_data_scored: false`, with
no interrupted-run comparison or exact prediction replay. Prior seed-2 GPU
receipts were not substituted for the seed-3 verification.

Retain D038/D039: no interrupted-versus-uninterrupted comparison, exact replay,
numerical tolerance machinery or repeated live-weight checksum gate. Practical
frozen-value and finite-training checks remain. A saved-object checksum
mismatch receives at most two bounded read-cache retries; expected hashes
are preserved and outcomes recorded. A persistent output-checkpoint
authentication failure is recorded as unresolved while independent unstarted
fits may continue. Other errors still stop and are recorded.

For a fit that has completed all 50 epochs and its final frozen-value check,
checkpoint evaluation can be recovered without retraining. Automatic training
resume remains disabled; continuation does not silently repeat or resume a
partially trained attempt. Prior read/memory diagnostic evidence and the
`provisional_unresolved_memory_integrity` designation remain. No reboot or
system-setting change belongs to this package.

## Commands and records

Run from the repository root in the `habcls` environment:

```bash
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_seed3.yaml prepare
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_seed3.yaml --device cuda verify
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_seed3.yaml --device cuda --background run
python tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_seed3.yaml status
```

All package records are under
[the seed-3 partial-joint root](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/):

- [Preparation manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/preparation/manifest.json).
- [GPU verification receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/verification/latest.json).
- [Launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/orchestration/seed3_joint_partial/launch.json).
- [Persistent process log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/orchestration/seed3_joint_partial/run_1792921233666553361.log).
- [Queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/orchestration/seed3_joint_partial/status.json).
- [First-fit progress](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/runs/seed3/joint_partial/image_only/pretrained/run.json).
- [Comparative report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/reports/seed3_joint_partial/report.md), published as fits complete.

Queue status refreshes at fit boundaries. The active fit's `run.json`,
`history.json` and persistent log show epoch-level progress. Each fit preserves
history, curves, predictions and reusable final/best checkpoint references under
`runs/seed3/joint_partial/<architecture>/<initialization>/`.

## Execution evidence and handoff

### Historical launch snapshot, 2026-10-25

The launch record identifies process **65206**, started at
`2026-10-25T09:40:33.684080+00:00`. The background launcher returned successfully
and the startup input audit passed. The queue selected
`seed3/joint_partial/image_only/pretrained` at **09:41:59 UTC (GMT)**, with
no recorded error or unresolved checkpoint. Its first run record reports
`running` at **09:42:15 UTC (GMT)**, `completed_epoch: 0` and
`expected_epochs: 50`; the first comparative training epoch has begun.
The package has **0/16 complete comparative fits** at this startup snapshot.
Overall M3J remains **50/84 active
fits complete**, with **34 active fits remaining** and **48 frozen-online fits
deferred**. The completed seed-1/2 and router artifacts remain preserved.
Scientific configuration and runtime sources are unchanged.

The preceding seed-2 partial-joint package took approximately 30 hours
49 minutes. At a similar pace, this package would take about **31 hours**,
finishing around **16:30 GMT on 2026-10-26**. This is a planning estimate,
not a guaranteed completion time; use current epoch timing for later updates.

### Progress snapshot, 2026-10-26 at 09:04 GMT

Process **65206** remains active on the GPU, using approximately **8,760 MiB**.
The queue records **12/16 completed fits**, the thirteenth running, and three
unstarted fits queued. The first six architectures in the execution-order
table have completed both initialization arms: image-only, raw concatenation,
projected concatenation, token-wise capacity, two-token attention and CLS-only
attention. Their **600 completed-fit training epochs** and **24 final/best
checkpoint-policy evaluations** are retained in the package report.

The active fit is `attention_cls_skip_adapted/pretrained`, with **10/50 epochs
completed** and best validation so far at **epoch 9**. The queue and log checks
record no error, unresolved checkpoint or failure alert. Both CLS-bypass
architectures are still unfinished, so the 16-fit package is not complete.

At the observed pace, approximately **7 hours 15 minutes** remain, suggesting
completion around **16:20 GMT on 2026-10-26** if the pace holds. This is an
estimate, not a completion claim. Overall M3J stands at **62/84 active fits
complete**, with **22 active fits remaining** and **48 frozen-online fits
deferred**. No runtime source, configuration, recipe, checkpoint or system
setting was changed for this progress update.

### Completion and subsequent handoff, 2026-10-26

The queue and comparative report completed at **16:23:18 UTC (GMT)**:
**16 complete fits**, **zero incomplete fits**, **800 training epochs**, and
**32 final/best checkpoint-policy evaluations**. The queue records no error
or unresolved checkpoint. The [completed report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/reports/seed3_joint_partial/report.md)
and [summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed3_direct_v1/reports/seed3_joint_partial/summary.json)
retain the metrics, confusion outputs and comparisons; each run retains its
history, curves, predictions and reusable checkpoint references. The package
took approximately **30 hours 43 minutes** from the recorded launch.

The best-validation Acc@1 among these 16 fits is **72.63%** for pretrained
CLS + adapted-token bypass at epoch 24, with **93.31% Acc@3**, **72.08%
weighted F1** and **0.6858 MCC**. The highest final-epoch Acc@1 is a tie at
**71.53%** between CS-adapted two-token attention and token-wise capacity.
These are descriptive single-seed results, not an across-seed selection or
an attention-specific improvement claim. All evaluations use the **822-image
development validation set**, with the existing provisional memory-integrity
designation retained.

This completion brings M3J to **66/84 active fits**, with **18 active fits
remaining** and **48 frozen-online fits deferred**. The user subsequently
authorized **one shared seed-3 geo-only fit** under [D050](03k_seed3_geo_only.md),
which records its own preparation, GPU/checkpoint verification and execution.
No frozen-online results are required. Seed 4 remains a separate future
package; this completion launches no seed-4 training.

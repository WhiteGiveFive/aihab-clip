# M3J seed-4 partial-joint execution

**D051, 2026-10-26; completed 2026-10-28. Status: complete.** All **16 seed-4
partial-joint fits** completed 50 epochs and final/best saved-checkpoint
evaluations. The queue completed at **05:41:20 UTC (GMT) on 2026-10-28**,
with **800 training epochs**, **32 checkpoint-policy evaluations**, no queue
error and no unresolved checkpoint. This brings M3J to **83/84 active fits**;
the one shared seed-4 geo-only fit follows separately under D052.

The historical launch record is preserved below. Package-specific preparation
and training-only GPU verification passed, and the fresh background queue
launched at **22:47:57 UTC (GMT) on 2026-10-26**, process **173284**. At launch,
**0/16 comparative fits were complete**. The first fit, `image_only/pretrained`,
reported `running` at **22:49:39 GMT**, with **0/50 epochs complete** at that
startup snapshot.

M3J entered this package with **67/84 active fits complete**: all 33 seed-1 fits
and the 17 active fits for each of seeds 2 and 3. At launch there were **17 active fits
remaining**: this 16-fit package and one separate seed-4 geo-only reference.
The **48 frozen-online fits** for seeds 2–4 remain deferred under
[D047](03h_frozen_online_deferral.md).

## Scope and fixed recipe

Use [the seed-4 configuration](../../configs/multimodal_attention_fusion_joint_seed4.yaml)
and study identity `attention_fusion_joint_seed4_direct_v1`. Scope is exactly
seed 4 and `joint_partial`: eight architectures, each independently initialized
from original-pretrained or seed-4 training-partition CS-adapted vision weights.
No earlier joint-trained checkpoint supplies an initialization.

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
| Random streams | D045 `loader_rng_policy: declared_namespaces_v1`; declared common `attention_fusion_joint_v1` crop/shuffle namespaces with seed 4 retained in each stream |
| Checkpoints | Option-3 shared storage; retain final epoch 50 and best validation by Acc@1, weighted F1, MCC, then earliest epoch; load and evaluate both |
| Monitoring | Every-epoch training/validation history and curves, loss, Acc@1, Acc@3, weighted F1, MCC, learning rates and resource timing |

This completed package contains **800 comparative training epochs** and **32
final/best checkpoint-policy evaluations**. Checkpoint policies refer to
one trajectory per fit, not additional fits. No cleaned-test scoring, router
application, frozen-online execution or later-seed training is included.

## Verification and practical recovery

[D045](03f_later_seed_preparation.md) prepared the seed-4 configuration and CPU
input record. This package refreshed its preparation against current execution
sources at **17:38:57 UTC (GMT)** on 2026-10-26, preserving the earlier
manifest in `preparation/history/manifest_1793036337362219367.json`. The refreshed
input record confirms all 16 seed-4 fits and the 3,378 training / 822 validation
rows. The scientific recipe and inputs are unchanged. The 363 passing focused CPU checks
recorded by [D047](03h_frozen_online_deferral.md) remain applicable because the
corresponding 27 execution sources are unchanged. The
[CPU evidence receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/preparation/d051_cpu_evidence.json)
records that reuse; this launch does not claim a new CPU test run.

This package's own training-only GPU verification passed at **17:46:18 UTC
(GMT)** on 2026-10-26. All 16 architecture/initialization gradient, meaningful-update
and practical frozen-value checks passed, together with a fresh one-epoch
checkpoint loading and standalone export/evaluation check. The receipt records
`validation_data_scored: false`, `interrupted_training_comparison: false` and
`exact_prediction_replay_required: false`. No earlier package's GPU receipt was
substituted for this verification.

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
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_seed4.yaml prepare
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_seed4.yaml --device cuda verify
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_seed4.yaml --device cuda --background run
python tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_seed4.yaml status
```

All package records are under
[the seed-4 partial-joint root](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/).

- [Preparation manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/preparation/manifest.json).
- [GPU verification receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/verification/latest.json).
- [Launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/orchestration/seed4_joint_partial/launch.json).
- [Persistent process log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/orchestration/seed4_joint_partial/run_1793054877197044093.log).
- [Queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/orchestration/seed4_joint_partial/status.json).
- [First-fit progress](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/runs/seed4/joint_partial/image_only/pretrained/run.json).

The [completed comparative report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/reports/seed4_joint_partial/report.md)
and [summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/reports/seed4_joint_partial/summary.json)
contain all 16 fits and both checkpoint policies.

Queue status refreshes at fit boundaries. The active fit's `run.json`,
`history.json` and persistent log show epoch-level progress. Each fit preserves
history, curves, predictions and reusable final/best checkpoint references under
`runs/seed4/joint_partial/<architecture>/<initialization>/`.

## Historical launch evidence

The launch record identifies process **173284**, started at
`2026-10-26T22:47:57.213756+00:00`. The background launcher returned successfully,
and the startup input audit passed. The queue selected
`seed4/joint_partial/image_only/pretrained` at **22:49:23 GMT**, with no
recorded error or unresolved checkpoint. Its first run record reports
`running` at **22:49:39 GMT**, `completed_epoch: 0` and `expected_epochs: 50`;
the first comparative training epoch had begun. The package has **0/16
completed comparative fits** at launch, so M3J remains
**67/84 active fits complete**, with **17 active fits remaining** and **48
frozen-online fits deferred**. Scientific configuration and runtime sources
are unchanged. A preservation check at **22:48:20 GMT** confirmed all **3,291**
recorded preceding completed-package files unchanged in presence, size and
modification time; see the
[preservation receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/preparation/d051_prior_artifacts_preserved.json).
This metadata check does not add a repeated model-weight checksum gate.

The preceding seed-2 and seed-3 partial-joint packages took approximately
30 hours 49 minutes and 30 hours 43 minutes respectively. At a similar pace,
this package would take about **31 hours**, finishing around **05:30–06:00 GMT
on Wednesday, 2026-10-28**. This is a planning estimate, not a guaranteed
completion time; use current epoch timing for later updates.

## Completion and handoff

The [queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed4_direct_v1/orchestration/seed4_joint_partial/status.json)
records all **16/16 fits complete at 05:41:20 GMT on 2026-10-28**, with
`error: null` and `unresolved: []`. Every fit completed its 50-epoch trajectory
and final/best saved-checkpoint evaluations, for **800 epochs** and **32
checkpoint-policy evaluations**. Results, histories, learning curves,
predictions and reusable checkpoint manifests remain in the package root.
They use the **822-image development validation set**, not the cleaned test.

M3J reached **83/84 active fits** at this close-out. The user subsequently
authorized the separate **shared seed-4 geo-only fit** under D052; its
[execution and completion record](03m_seed4_geo_only.md) carries its own
preparation, training-only GPU/checkpoint verification and combined report.
D051 itself did not automatically launch that fit. The **48 frozen-online
fits** remain deferred, and cross-seed consolidation and the joint-selection
record remain work for the M3J development report. No router work, M4 refit
or cleaned-test application follows automatically from this completion.
Completed prior packages and provisional memory-integrity assurance remain
unchanged.

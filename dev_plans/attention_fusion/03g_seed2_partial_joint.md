# M3J seed-2 partial-joint execution

**D046, 2026-10-23:** the user authorized launching the prepared 16 seed-2
partial-joint fits. The background queue launched at **22:17:43 UTC
(23:17 BST)** with process **30319**. This is the first comparative package
after the completed 33-fit seed-1 study.

**Complete, 2026-10-25:** all 16 fits and all 32 saved final/best checkpoint
evaluations completed. The queue closed at **05:06:56 UTC (GMT)**, after about
30 hours 49 minutes, with no recorded error or unresolved checkpoint. D047
defers seed-2/3/4 frozen-online and makes the seed-2 geo-only fit the next package.

## Scope and recipe

Use [the seed-2 configuration](../../configs/multimodal_attention_fusion_joint_seed2.yaml)
and study identity `attention_fusion_joint_seed2_direct_v1`.
The scope is exactly seed 2 and `joint_partial`: eight architectures, each
with original-pretrained and seed-2 CS-adapted initialization.

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

Retain 50 epochs per fit, batch 16 without accumulation, Adam `5e-5` with cosine
decay, train-fitted standardized geo, historical image preprocessing and
arithmetic, and final/best checkpoints. Train the final ten vision blocks,
final normalization and attention-pooling head alongside the fusion/head.
D045's declared crop/shuffle namespaces apply. The development cohort remains
3,378 training and 822 validation images; no cleaned-test scoring is included.

Frozen-online, geo-only and later seeds remain separate packages. This launch
does not automatically advance to any of those packages.

## Pre-launch evidence and operation

[D045 preparation](03f_later_seed_preparation.md) passed 311 CPU tests and all
16 seed-2 training-only GPU model checks, plus saved-checkpoint reload/export.
Immediately before launch, the passed verification receipt matched the current
configuration and all recorded execution sources; all 16 fits were unstarted,
and no previous launch existed. The host RTX 4090 was idle with approximately
452 MiB used. CUDA is accessed outside the sandbox.

The shared runner retains D038/D039: no interrupted-run comparison, exact replay,
tolerance machinery or live-weight checksum gate; practical frozen-value and
finite-training checks; two bounded read-cache retries for a saved-object
checksum mismatch; recorded unresolved checkpoints with continuation to
independent fits; evaluation-only recovery after completed training. Other
errors still stop and are recorded. The unresolved-memory provisional label
continues to apply.

The launch command was:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_attention_fusion_joint_study.py \
  --config configs/multimodal_attention_fusion_joint_seed2.yaml \
  --device cuda --background run
```

No implementation, scientific configuration, prior checkpoint or system setting
was changed during launch. The original completed seed-1 and router artifacts
remain preserved.

## Live records

All package records are under
[the seed-2 partial-joint root](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1/).

- [Launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1/orchestration/seed2_joint_partial/launch.json).
- [Persistent process log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1/orchestration/seed2_joint_partial/run_1792793863144479572.log).
- [Queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1/orchestration/seed2_joint_partial/status.json).
- [First-fit progress](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1/runs/seed2/joint_partial/image_only/pretrained/run.json).
- [Passed verification receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1/verification/latest.json).

The runner publishes queue progress at
`orchestration/seed2_joint_partial/status.json` after its input audit and protocol
declaration. Each fit records progress, epoch history, curves and checkpoint
references under `runs/seed2/joint_partial/<architecture>/<initialization>/`.
Queue status is refreshed at fit boundaries; the current fit's `run.json`,
`history.json` and log provide epoch-level progress.

## Historical launch status

The background launcher exited successfully. The input audit passed and
`image_only/pretrained` entered training at **22:19:23 UTC (23:19 BST)**.
The first-fit record reports `running`, with an expected 50 epochs; the host
GPU confirms process 30319 using approximately 8,342 MiB. The queue records no
error or unresolved checkpoint. At this startup snapshot, 0/16 seed-2 fits
are complete; overall M3J remains at 33/132 completed fits.
This paragraph preserves the original startup snapshot.

## Completion and handoff

The saved queue status records `complete: 16`, `queue_status: complete`,
`active_fit: null`, `error: null` and `unresolved: []` at
`2026-10-25T05:06:56.919942+00:00`. All 16 individual run records have 50
completed epochs (800 total). All 32 final/best evaluations record
`metrics_source: reloaded_checkpoint_evaluation` on the same 822-image
development validation cohort. No cleaned-test scoring was performed.

The [completed report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1/reports/seed2_joint_partial/report.md)
links all metrics, confusion matrices and epoch curves. Pretrained CLS-only
attention has the highest Acc@1 under both policies: 71.17% final and 72.26%
at the selected epoch 40. These are descriptive seed-level findings under the
existing provisional assurance; the report retains all other metrics and
checkpoint-selection limits.

The [D047 deferral amendment](03h_frozen_online_deferral.md) preserves this
package and its original configuration/source snapshot. Only its recorded
completion/input/recipe metadata is needed as the predecessor for seed-2
geo-only. That next fit has not started; no frozen-online results are required
until the deferred study is explicitly reactivated.

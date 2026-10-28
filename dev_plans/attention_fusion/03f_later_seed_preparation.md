# M3J seeds 2–4: package preparation

**Decision D045, 2026-10-23.** The user authorized preparing configuration
templates for seeds 2–4 and the shared implementation they need. This did not
start the comparative fits. At that preparation snapshot, seed 1 was complete
at 33/33 fits and M3J at 33/132. The router side study is set aside until its later test-application
stage, after the main study.

**Preparation complete:** all nine real-input packages are prepared, 311 CPU
tests passed, and seed-2 partial-joint GPU verification passed at
**20:55:24 UTC (21:55 BST) on 2026-10-23**. The next 16-fit comparative package
was ready to launch when requested. No comparative training was launched
during preparation. The user subsequently authorized
[D046 seed-2 execution](03g_seed2_partial_joint.md), launched at 22:17:43 UTC
on 2026-10-23.

**Current amendment, D047 (2026-10-25):** seed-2 partial-joint is complete at
16/16. Frozen-online for seeds 2–4 is deferred. Execute seed-2 geo-only next,
then partial-joint → geo-only for seeds 3 and 4. The three geo-only templates
now require only the completed same-seed partial-joint package and explicitly
declare the frozen-online reference deferred. This supersedes D045's order
and two-reference prerequisite below; the original preparation receipts remain
historical evidence. See [the deferral record](03h_frozen_online_deferral.md).

## Packages and execution order

Each package has its own configuration, artifact root, input/preparation record,
GPU verification record, protocol/source snapshot, and eventual run/report
directories. Package preparation can happen together. D045 originally ordered
partial-joint → frozen-online → geo-only within each seed; D047 now orders
partial-joint → geo-only for seeds 2–4, with frozen-online deferred.

| Seed | Partial-joint: 16 fits | Frozen-online: 16 fits | Shared geo-only: 1 fit |
|---|---|---|---|
| 2 | [Configuration](../../configs/multimodal_attention_fusion_joint_seed2.yaml) | [Configuration](../../configs/multimodal_attention_fusion_joint_frozen_online_seed2.yaml) | [Configuration](../../configs/multimodal_attention_fusion_joint_geo_only_seed2.yaml) |
| 3 | [Configuration](../../configs/multimodal_attention_fusion_joint_seed3.yaml) | [Configuration](../../configs/multimodal_attention_fusion_joint_frozen_online_seed3.yaml) | [Configuration](../../configs/multimodal_attention_fusion_joint_geo_only_seed3.yaml) |
| 4 | [Configuration](../../configs/multimodal_attention_fusion_joint_seed4.yaml) | [Configuration](../../configs/multimodal_attention_fusion_joint_frozen_online_seed4.yaml) | [Configuration](../../configs/multimodal_attention_fusion_joint_geo_only_seed4.yaml) |

The image packages retain all eight architectures and both initializations.
Each adapted initialization selects its own seed's training-partition CS encoder.
The pretrained initialization remains the original pinned SigLIP2 checkpoint.
Frozen-online starts from these same initial encoders, not the corresponding
joint-trained encoder. Geo-only is one shared reference per seed.

Artifact identities follow these patterns, with `N` replaced by 2, 3 or 4:

```text
multimodal_artifacts/attention_fusion/cs/gse_100m/
  attention_fusion_joint_seedN_direct_v1/
  attention_fusion_joint_frozen_online_seedN_direct_v1/
  attention_fusion_joint_geo_only_seedN_direct_v1/
```

The scope guard accepts only the explicitly declared seed/regime combination
and its expected initialization arms. Frozen-online references that same seed's
completed partial-joint package. D045's geo-only templates referenced both
completed image packages; D047's seed-2/3/4 templates reference only the completed
partial-joint package and list frozen-online separately as deferred. Preparing a future package does not require these
future results to exist. The existing completion checks apply before its
comparative execution.

## Scientific recipe and prospective random-stream correction

Keep the established recipe: 50 epochs; batch 16 without accumulation; Adam
`5e-5` for all trainable components; cosine decay; zero weight decay; no warm-up
or early stopping; historical image preprocessing and FP16-autocast/FP32-weight
arithmetic; train-fitted standardized geo; final and best-validation reporting;
option 3 shared checkpoint storage. Partial-joint still trains the final ten
vision blocks, final normalization and attention-pooling head.

D041 documented that seed 1's image loaders used their different package study
IDs to seed shuffle and crops. D045 corrects this prospectively:

```yaml
training:
  loader_rng_policy: declared_namespaces_v1
  augmentation_seed_namespace: attention_fusion_joint_v1
  shuffle_seed_namespace: attention_fusion_joint_v1
  dropout_seed_namespace: attention_fusion_joint_v1
```

For seeds 2–4, augmentation uses the declared namespace, seed, epoch and row
identity; shuffling uses its declared namespace, seed, role and epoch. Different
architectures, initializations and image regimes therefore receive matched
crops/orders within a seed. The seed remains an input, so different seeds still
have different streams. Geo-only uses the shared shuffle namespace and has no
image augmentation. Fusion initialization and dropout keep their existing
declared namespaces.

Absent `loader_rng_policy` preserves `legacy_study_id_v1`. The three seed-1
configurations, completed outputs and original source snapshots are retained.
Seed 1's cross-regime limitation still applies; pooled conclusions must disclose
that seed 1 used different cross-regime streams while seeds 2–4 implement the
intended pairing. This is a declared implementation correction, not a new
hyperparameter search or a retrospective change to seed 1.

Reports now derive pairing from saved loader configurations and distinguish
matched namespaces, different namespaces and missing/unknown provenance. They
do not automatically copy the seed-1 limitation into every later-seed report,
or claim pairing when the supporting record is absent.

## Practical checks and recovery

Retain D038/D039 unchanged:

- No interrupted-versus-uninterrupted experiment, exact prediction replay,
  numerical-tolerance machinery or repeated live-weight checksum gate.
- Retain finite training, correct inputs/identities, intended trainability,
  frozen-parameter optimizer exclusion/no gradients and direct frozen-value
  checks at the existing verification/start/end boundaries.
- A saved-object checksum mismatch receives at most two read-cache refresh
  retries. The expected digest is never rewritten; retries are recorded.
- Persistent checksum failures mark that fit unresolved while independent fits
  continue. Other exceptions, including nonfinite training, missing/truncated
  files or actual frozen-value changes, still stop and are recorded.
- A fit with all 50 epochs complete can recover final/best checkpoint evaluation
  without additional training. `continue` preserves completed fits and incomplete
  attempts; it does not automatically resume or retrain partial attempts.
- Retain the provisional unresolved-memory assurance label.

The seed-2 queue is covered by the same recovery regression tests as seed 1.
Preparation does not introduce an additional live training gate.

## Commands and records

The existing runner now has a CPU-only `prepare` command. It checks the declared
real inputs, saves the train-fitted geo transform and input identity, records
the selected adapted-encoder producer and planned fits, and lists predecessor
package status. It performs no model construction, optimization, prediction or
GPU verification. The resulting `preparation/manifest.json` is explicitly
marked `scope: cpu_inputs_and_configuration_only`; it cannot substitute for a
passed GPU receipt.

Example preparation, from the repository root:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_attention_fusion_joint_study.py \
  --config configs/multimodal_attention_fusion_joint_seed2.yaml prepare
```

Use the corresponding configuration from the table for each other package.
The preparation command is safe to repeat against unchanged inputs and updates
only that new package's preparation record. No seed-1 output is rewritten.

Before starting an individual package, perform the existing short training-only
GPU/checkpoint verification with `verify`; it uses training rows only. For seed 2:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_attention_fusion_joint_study.py \
  --config configs/multimodal_attention_fusion_joint_seed2.yaml --device cuda verify
```

The later user-authorized comparative launch uses the same configuration with
`--background run`. Use `--background continue` for the existing evaluation-only
recovery/preserve-incomplete behaviour. Neither launch command was executed as
part of this preparation. Freeze the source/configuration after verification;
if they change before launch, obtain a receipt for the changed package.

Shared preparation evidence is under
[preparation_seeds2_4_v1](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_seeds2_4_v1/):

- [Package execution summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_seeds2_4_v1/summary.json)
  and individual CPU preparation logs.
- [GPU availability observation](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_seeds2_4_v1/gpu_availability.json).
- [Seed-1 artifact metadata inventory before preparation](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_seeds2_4_v1/seed1_inventory_before.json).
- [Seed-1 preservation result](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_seeds2_4_v1/seed1_preservation.json):
  all 1,605 recorded files retain their presence, size and modification time;
  the three original configurations match their saved source copies byte for
  byte. This is not a new hash audit of the model weights.
- [CPU validation record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_seeds2_4_v1/cpu_validation.json)
  and [test output](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_seeds2_4_v1/cpu_tests.log).

## Verification and handoff

The focused protocol, input, storage, training, reporting and CLI suites passed
**311 tests**. They cover later-seed package isolation, wrong-seed predecessor
rejection, actual paired loader images/order, seed-specific adapted checkpoint
selection, unknown-provenance reporting, preparation without training/GPU use,
and seed-1/seed-2 checkpoint failure recovery. One existing dependency
deprecation warning was emitted.

All nine real-input preparation commands completed successfully. They describe
99 distinct future fits, each with 3,378 training and 822 validation rows. The
cohort identities and training-fitted geo scaler agree across packages, and the
image packages authenticate the correct seed-2/3/4 adapted encoder. No
comparative run directories were created.

The sandbox initially reported unavailable CUDA. A read-only check outside the
sandbox confirmed an idle RTX 4090 with 24,564 MiB total memory; this was a
sandbox visibility issue, not evidence of a failed host driver. The existing
seed-2 partial-joint training-only GPU/checkpoint verification passed outside
the sandbox. All 16 architecture/initialization checks passed, followed by a
fresh one-epoch checkpoint smoke check, saved final/best evaluation and
standalone export/loading. Its
[passed receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1/verification/latest.json)
binds the same final sources as the 311 passing CPU tests; the
[log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_seeds2_4_v1/seed2_gpu_verification.log)
retains the execution record. Verification used training rows only and
explicitly records that interrupted-run comparison and exact prediction replay
were not required. No driver/system changes or reboot were performed. GPU
verification for subsequent packages remains a per-package pre-launch step.

The original preparation handoff was **16 seed-2 partial-joint fits**, followed
by seed-2 frozen-online and geo-only. D046 completed the first package;
its [execution record](03g_seed2_partial_joint.md) supersedes this preparation
snapshot for results. D047 makes seed-2 geo-only next and defers frozen-online
for all three later seeds. Existing D045 GPU receipts bind their historical
sources; later packages need their own current pre-launch verification.

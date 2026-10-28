# M3J — Seed-1 frozen-online package

Status: **Complete: 16/16 fits**, finished on **2026-10-22 at 00:49:39 UTC
(01:49 BST)**. All fits completed 50 epochs and final/best saved-checkpoint
evaluations; the queue has no error or unresolved fit. See the
[completion and handoff](#completion-and-handoff).
The background queue launched on 2026-10-20 at **22:56:14 UTC (23:56 BST)**,
process **490326**, after 265 CPU checks and all 16 training-only GPU model
checks plus final/best loading and standalone export passed. It follows the existing
D034 execution order in the [governing guide](README.md#execution-order-and-sub-tasks).
It introduces no new scientific recipe or model-selection choices.

The preceding **16/16 seed-1 partial-joint fits are complete**, recorded at
2026-10-20 21:39:24 UTC (22:39 BST). The
[completion record](03_joint_finetuning.md#seed-1-partial-joint-completion-results-and-handoff)
links final/best results, curves, reusable checkpoints and the metadata audit.
Both image packages and the [seed-1 geo-only reference](03c_seed1_geo_only.md)
are complete, bringing seed 1 to **33/33 fits**. The combined report/queue
finished at 2026-10-22 10:18:23 UTC (11:18 BST). All completed packages and their
source snapshots are retained.

## Scope and purpose

Run **16 fits: eight architectures × two image initializations × seed 1**,
with the entire image encoder frozen. These are the matched controls for the
seed-1 partial-joint package. They allow comparison of updating the encoder
versus holding it fixed while matching image views and downstream training.

| Architecture | Original pretrained initialization | Historical seed-1 CS-adapted initialization |
|---|---|---|
| `image_only` | Required | Required |
| `raw_concat` | Required | Required |
| `projected_concat` | Required | Required |
| `tokenwise_capacity` | Required | Required |
| `attention_two_token` | Required | Required |
| `attention_cls` | Required | Required |
| `attention_cls_skip_adapted` | Required | Required |
| `attention_cls_skip_original` | Required | Required |

Use the displayed architecture order and pretrained then CS-adapted for each.
Final and best-validation checkpoints come from the same trajectory and do not
double the fit count. After these 16 fits, the next package is **one seed-1
geo-only fit**, subsequently completed under D041; later seeds remain separate
packages under the same governing order.

## Matched experiment contract

**D041 implementation audit:** the contract below records the intended paired
setup. The completed loader derived shuffle and crop seeds from `study_id`,
ignoring the declared common namespaces. Because the partial-joint and
frozen-online study IDs differ, their realized minibatch orders and image views
are not paired across regimes. The data, preprocessing rules and numerical
recipe remain matched. Keep this limitation with regime comparisons; do not
attribute their full difference solely to encoder trainability. Preserve the
completed artifacts without retroactive modification or retraining. The new
geo-only path uses its declared shuffle namespace and has no image augmentation.

Reuse the [M3J scientific recipe](03_joint_finetuning.md#scope-and-fixed-recipe).
The intended experimental change is the visual encoder's trainability:

- Freeze every visual parameter, exclude all of them from the optimizer, and
  keep the complete visual encoder in evaluation mode. Compute its outputs
  without an encoder gradient graph. Train only the applicable adapters,
  fusion components and classifier; image-only trains its classifier.
- Continue loading images and computing image vectors online for each batch.
  Training still uses the paired random-resized crops and deterministic
  per-seed/epoch/row views. Validation keeps deterministic preprocessing.
  The earlier M3 single-view image cache does not replace this control.
- Start from the same pinned original SigLIP2 or historical seed-1 CS-adapted
  encoder used at the beginning of the matching partial-joint fit. Do not
  initialize from an encoder learned during M3J. Initialize downstream
  parameters and Adam state afresh, preserving paired component seeds.
- Keep the same 3,378 training / 822 validation images and class mapping,
  historical BGR/439-to-384 image path, L2 image output and training-fitted
  coordinate standardization of the 64D geo vectors.
- Keep **50 epochs**, physical/effective **batch 16**, no accumulation,
  **Adam at initial LR `5e-5`**, cosine decay over 50 epochs, zero weight decay,
  no warm-up and no early stopping.
- Keep FP32 parameters/state, FP16-autocast training without GradScaler,
  autocast-disabled validation, cuDNN TF32 enabled and matmul TF32 disabled.
- Retain epoch curves and metrics, final epoch 50 and the best-validation
  checkpoint selected by Acc@1, weighted F1, MCC and earliest epoch in a tie.
  Report Acc@1, Acc@3, weighted F1, MCC and per-class/confusion analysis for
  both checkpoint policies. No cleaned-test scoring or selection is involved.

## Execution integration and verification

The implementation provided the following functionality. The intended
cross-regime augmentation/shuffle pairing in item 1 was not realized; see the
D041 audit above.

1. A separate seed-1 frozen-online configuration, scope, source snapshot and
   verification record. Input/initialization/augmentation/shuffle/dropout
   identities stay paired with the preserved partial-joint package.
2. Regime-specific orchestration, launch/status and report paths.
3. Short checks requiring **zero trainable visual parameters**, no visual
   gradients/optimizer membership, evaluation mode, unchanged visual values
   and an actual downstream parameter update. Production frozen values are
   anchored to the authenticated original encoder.
4. Option 3 final/best loading/export and reuse of the original frozen encoder
   object when it covers the complete visual state. D039's bounded saved-file
   retries, exact expected digests, evaluation-only recovery and isolation of
   persistent output-checkpoint failures remain. Continuation does not repeat
   completed or partial training.
5. Training-only numerical/resource checks and source-bound evidence, retaining
   the existing provisional unresolved-memory assurance label.
6. Separate final/best partial-joint-minus-frozen-online comparisons, alongside
   within-regime architecture and initialization contrasts. Pairs must match
   seed, architecture, initialization, validation rows/labels/plots, class axis
   and checkpoint policy; incomplete pairs are explicitly listed.

Carry forward the user's D038/D039 verification choices. Do not reintroduce
mandatory interrupted-versus-uninterrupted comparison, exact prediction/loss
replay, live-weight checksum gates or numerical-tolerance machinery. Direct
frozen-value checks, optimizer exclusion/no gradients and usable saved
checkpoints remain required. Bounded saved-file retries reduce transient-read
interruptions; a persistent failure must still be recorded rather than accepted
as a verified checkpoint.

The partial-joint package has finished. Preserve its original configuration,
protocol, receipts and artifacts when changing execution sources for the next
package, and bind those changes to the new package record and verification.
D040 authorizes implementation and launch after these checks pass; it changes
no scientific settings. The new package uses
`attention_fusion_joint_frozen_online_direct_v1` and
[`multimodal_attention_fusion_joint_frozen_online.yaml`](../../configs/multimodal_attention_fusion_joint_frozen_online.yaml).
Its protocol, verification, runs and reports are separate from the completed
partial-joint root. Shared immutable frozen visual objects are deduplicated
within this package; no joint-trained encoder is used as an initialization.
The new protocol records the completed partial-joint package as a read-only
comparison dependency and checks matching recipe and study-neutral input
provenance. The original source snapshots remain available for reproducing
the completed package; its old verification receipt is not reused for this one.

## Package deliverables

- Sixteen independently initialized, fully recorded 50-epoch fits.
- Reusable final/best downstream weights with the correct shared frozen encoder,
  preprocessing, class mapping and dependency manifest.
- Epoch curves, per-fit final/best metrics and a frozen-online package report.
- Matched partial-joint versus frozen-online comparisons for every architecture
  and initialization, keeping final and best policies separate. Seed-1 findings
  remain descriptive until the declared cross-seed comparisons are available.
- An updated progress record and a handoff to the single seed-1 geo-only fit.

## Commands and output locations

Run from the repository root in the `habcls` environment. The background command
starts only the 16 authorized frozen-online fits; it does not start geo-only
or later seeds.

```bash
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_frozen_online.yaml verify
python -u tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_frozen_online.yaml --background continue
python tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_frozen_online.yaml status
python tools/run_attention_fusion_joint_study.py --config configs/multimodal_attention_fusion_joint_frozen_online.yaml report
```

`continue` starts previously unstarted fits, preserves completed fits, and recovers
checkpoint evaluation for fully trained fits when necessary. It never resumes
optimizer updates or restarts partially trained attempts automatically.

The artifact root is
`multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/`:

- `protocol/`: configuration, source snapshot and matched-package dependency.
- `verification/`: CPU evidence and training-only GPU checks.
- `orchestration/seed1_frozen_online/`: launch, persistent log and queue status.
- `runs/seed1/frozen_online/<candidate>/<initialization>/`: per-fit histories,
  curves, final/best predictions, metrics and checkpoint manifests.
- `objects/`: shared frozen encoder, downstream weights and retained states.
- `reports/seed1_frozen_online/`: package metrics and paired
  partial-joint-minus-frozen-online contrasts for final/best separately.

## D040 implementation and verification evidence

The runner now uses package-specific launch/status/report paths and checks
downstream parameter updates in both regimes. For frozen-online verification,
it requires zero trainable visual parameters, no visual gradients or optimizer
membership, unchanged visual values and evaluation mode throughout the encoder.
The trainer's existing frozen execution and shared visual references are reused.
The reporter adds separate final/best partial-joint-minus-frozen-online tables
and identifies missing/incomplete pairs. It checks row, label, plot and class
identities and reads the completed reference package without modifying it.

The initial integration passed **261 focused and regression CPU tests**. The
[CPU receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_cpu_tests.json)
and [test results](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_cpu_tests.xml)
record the current configuration/source identity. The actual input audit matches
the completed partial-joint provenance after excluding only the distinct study
ID: 3,378 training and 822 validation images. The
[GPU verification log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_gpu_verification.log)
records the training-only verification attempt. No comparison with an interrupted
trajectory, exact prediction replay or numerical tolerance is required.

The first GPU verification attempt on 2026-10-20 passed the pretrained image-only
training/frozen-value check, then failed before the CS-adapted image-only training
step. Initial device values differed from the pinned source in
`trunk.blocks.25.mlp.fc2.weight`. Its
[failed receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/8827bf3fc57e_1792535279416412207/failure.json)
and original log are preserved. No full fit had started. A fresh-process
[verification retry](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_gpu_verification_retry1.log)
uses the unchanged source, configuration and acceptance policy; the failure is
not treated as a successful check or repaired by accepting a tolerance.

The fresh-process retry also stopped during verification, before any full fit.
At `trunk.blocks.12.mlp.fc1.weight[2556, 927]`, the saved reference read
`-0.0007864967919886112` while the reported live value was
`-0.0007864930666983128`. CPU diagnostics subsequently found that the original
checkpoint agrees with the reported live value. The saved reference differs by
one bit (`0x40` XOR), at exactly one of 4,958,208 tensor elements; repeated
scalar reads were stable. This evidence points to the captured reference,
not an optimizer update at that position, without identifying the physical
cause. The second failed receipt remains under
`verification/8827bf3fc57e_1792535694461374715/`.

The production frozen reference is therefore anchored to the authenticated
original encoder values rather than a copied live-device snapshot. Start/end
checks still require exact equality, and no live weight is changed to pass a
check. This is a correction to the reference used by the existing freeze
invariant, not a numerical tolerance or an acceptance of changed frozen weights.
The corrected implementation requires fresh CPU/GPU evidence before launch.

The corrected implementation passed **263 CPU checks**, including regression
cases proving that a changed live weight cannot establish its own frozen
reference. The [corrected CPU receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_canonical_reference_cpu_tests.json)
binds the changed source identity. The
[one-bit diagnostic](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/diagnostics/frozen_reference_bit_difference.json)
records the original/snapshot values and repeated read evidence. The
[corrected GPU verification log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_gpu_verification_canonical_reference.log)
records the new attempt under the same exact-comparison acceptance policy.

That GPU attempt passed at **2026-10-20 22:46:40 UTC**: all 16
architecture/initialization checks, exact frozen-value comparisons, downstream
updates, final/best loading and standalone export. The retained
[passed receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/e7d8f966c896_1792536083479673024/verification.json)
records a maximum of approximately 2.175 GiB allocated during the model checks.

Before full fitting, checkpoint handling reuses the authenticated canonical
encoder object directly when the complete frozen
visual state consists of those source parameters. This SigLIP tower satisfies
that condition (342 state entries, all parameters). It avoids an unnecessary
epoch-1 GPU-to-CPU copy that could otherwise encode a damaged value as a new
content-addressed checkpoint object. Towers with additional buffers retain
complete-state serialization. The change leaves optimizer updates, predictions,
selection, metrics and training numerics unchanged; source-bound CPU/GPU
verification is renewed before the queue starts.

The final checkpoint-reuse implementation passed **265 CPU checks**. Its
[source-bound CPU receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_shared_encoder_cpu_tests.json)
and [test results](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_shared_encoder_cpu_tests.xml)
include coverage preventing an extra visual copy and preserving complete state
for towers with buffers. The
[final GPU verification log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/d040_gpu_verification_shared_encoder.log)
records verification of this final source identity. Earlier receipts remain
available as implementation history rather than authorization for changed code.

## Launch and current execution

Final GPU verification passed at **2026-10-20 22:55:24 UTC**. The
[final receipt](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/verification/1c5e2ea8815e_1792536619179521610/verification.json)
records all 16 model checks, exact frozen values, downstream updates and
standalone loading/evaluation. The real verification manifest confirms that
final/best visual references equal the authenticated canonical reference.

The user-authorized background queue launched at **22:56:14 UTC (23:56 BST)**
on 2026-10-20, process **490326**, using the `--background continue` command
above. The [launch record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/orchestration/seed1_frozen_online/launch.json)
and [persistent log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/orchestration/seed1_frozen_online/run_1792536974970728423.log)
identify this package separately from the completed partial-joint process.
At launch there were no existing full fits in this root; all 16 start afresh
from epoch 1. The first fit is `image_only/pretrained`. Scientific settings,
original partial-joint artifacts and the no-automatic-training-resume policy
are unchanged. Results retain provisional unresolved-memory assurance.

At the historical start snapshot, the first fit entered training at
**2026-10-20 22:58:02 UTC**. Its
[run record](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/runs/seed1/frozen_online/image_only/pretrained/run.json)
reports `running`, with 50 expected epochs. The
[start freeze check](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/runs/seed1/frozen_online/image_only/pretrained/freeze_checks.json)
passed direct comparison of all 342 visual tensors (428,256,704 parameters),
optimizer exclusion and absence of visual gradients. Four classifier tensors
are trainable. The remaining 15 fits are queued; no fit was complete at this
start snapshot. The [queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/orchestration/seed1_frozen_online/status.json)
and per-fit histories provide subsequent progress.


## Completion and handoff

The [final queue status](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/orchestration/seed1_frozen_online/status.json)
records completion at **2026-10-22 00:49:39 UTC (01:49 BST)**:

- **16/16 fits**, all with 50 completed epochs: **800 training epochs**.
- Final epoch 50 and best-validation checkpoints reloaded and evaluated for
  every fit: **32 checkpoint-policy evaluations**.
- No active fit, queue error or unresolved checkpoint.
- The [report summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/reports/seed1_frozen_online/summary.json)
  contains 16 completed and zero incomplete fits.

The [package report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_direct_v1/reports/seed1_frozen_online/report.md)
links metrics, class/confusion analysis, final/best regime contrasts, curves and
reusable checkpoints. All scoring uses the **822-image development validation
set**, with no cleaned-test application. Results remain single-seed descriptive
and carry `provisional_unresolved_memory_integrity`. The D041 namespace finding
above additionally qualifies the joint-versus-frozen contrasts; successful
completion does not establish identical training views or resolve the earlier
memory/read incidents.

Both 16-fit image packages completed **32/33 seed-1 fits**. D041 subsequently
completed the [single standardized geo-only fit](03c_seed1_geo_only.md), bringing
seed 1 to **33/33**. That reference is shared by both image initialization arms
and encoder regimes, with a separate configuration, source snapshot and artifact
root. Its combined report/queue finished at 10:18:23 UTC (11:18 BST) on
2026-10-22. Preserve
this package's protocol, original source identity, verification receipts and
all final/best outputs; do not rerun or rewrite completed image fits.

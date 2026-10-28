# M3J later-seed frozen-online deferral

**D047, 2026-10-25.** The user deferred frozen-online training for seeds 2, 3
and 4 until a later investigation, then authorized updating the governing
plan, geo-only prerequisites and combined reporting. This amendment does not
launch comparative training or change the scientific recipe.

## Scope and execution order

The full M3J design remains 132 unique fits. The current scope contains 84:
the 33 completed seed-1 fits, plus 17 per later seed (16 partial-joint and one
shared geo-only). The other 48 fits are deferred, not cancelled or failed.

| Seed | Partial-joint | Shared geo-only | Frozen-online | Next action |
|---|---|---|---|---|
| 1 | 16 complete | Complete | 16 complete | Preserve completed evidence |
| 2 | 16 complete | Unstarted | 16 deferred | Run geo-only after its pre-launch check |
| 3 | 16 unstarted | Unstarted | 16 deferred | Partial-joint, then geo-only |
| 4 | 16 unstarted | Unstarted | 16 deferred | Partial-joint, then geo-only |

As of this amendment, **49/84 active fits are complete**, with **35 active fits
remaining**. All 16 seed-2 partial-joint fits completed 50 epochs and final/best
checkpoint evaluations; its queue closed at 05:06:56 UTC (GMT) on 2026-10-25
without an error or unresolved checkpoint. See the
[completion record](03g_seed2_partial_joint.md#completion-and-handoff).

Keep 50 epochs, batch 16, Adam at `5e-5` with cosine decay, training-fitted
standardized geo, existing numerical precision, declared random streams and
final/best checkpoint storage. Geo-only remains one 64D-geo classifier per
seed and has no vision encoder or initialization arm. The deferred study can
reuse that same geo reference later if the recipe remains unchanged.

## Configuration and implementation

The three geo-only templates now declare the deferral explicitly:

```yaml
reporting:
  deferred_reference_regimes: [frozen_online]
  completed_reference_roots:
    - multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_seed2_direct_v1
```

The root above is seed 2's example; each template uses its own seed:

- [Seed-2 geo configuration](../../configs/multimodal_attention_fusion_joint_geo_only_seed2.yaml).
- [Seed-3 geo configuration](../../configs/multimodal_attention_fusion_joint_geo_only_seed3.yaml).
- [Seed-4 geo configuration](../../configs/multimodal_attention_fusion_joint_geo_only_seed4.yaml).

The [protocol module](../../multimodal/attention_fusion_joint_protocol.py)
validates the declared active reference list. Only seed-2/3/4 geo-only packages
may explicitly defer frozen-online. Their completed same-seed partial-joint
package must still match the input cohort, training recipe, checkpoint policy
and recorded complete queue. Missing or wrong-seed active references remain
errors. A deferred root is recorded as planned metadata and is not read as a
prerequisite. The unchanged seed-1 configuration still requires both completed
image packages. Configurations with both references remain supported for a
future separately recorded reactivation.

The [runner](../../tools/run_attention_fusion_joint_study.py) passes the scope
to reporting. Its CPU-only `prepare` records required predecessors separately
from deferred ones and preserves an existing preparation receipt in
`preparation/history/` before writing a replacement. Original completed
protocol/source snapshots and verification receipts remain unchanged. Source
changes require a fresh package-specific verification before a new fit;
historical receipts are not relabelled as current.

The [combined reporter](../../multimodal/attention_fusion_joint_reporting.py)
distinguishes **17 active planned fits**, **16 deferred fits**, and **33 fits in
the full per-seed plan**. Its observed complete/incomplete counts describe only
active runs. It continues to compare each fusion model with image-only and the
one shared geo-only reference using separate final/best policies. It emits no
frozen-online performance, cross-regime contrasts or missing-pair RNG warning
for the deliberately deferred regime. Seed 1's existing 33-fit reporting path
remains available.

Retain D038/D039 practical checks and bounded saved-file recovery. This change
does not introduce interrupted-versus-uninterrupted tests, exact prediction
replay, numerical tolerance or live-weight checksum gates. Results retain
`provisional_unresolved_memory_integrity`.

## Verification and preparation evidence

Focused CPU tests cover explicit deferral, active-reference identity and
completion, unread deferred roots, 17-fit reporting, legacy 33-fit reporting,
checkpoint-policy separation, preparation history and the existing recovery
behavior. Real-input CPU preparation refreshes the three geo-only receipts.
The seed-2 active predecessor can now be checked against its completed package;
seed-3/4 partial-joint predecessors remain unstarted and must complete before
their geo-only fits can run.

Evidence is kept in
[the D047 preparation directory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_deferred_frozen_v1/).
The original three D045 geo-only preparation manifests are copied there before
refreshing. Completed-package preservation is checked using file presence,
size and modification time, without loading or hashing model weights. The
training, input, storage and model source files and scientific configuration
fields are checked separately for unchanged content.

**Preparation complete:** all **363 focused CPU tests passed**, with one
existing dependency deprecation warning. The tests cover the six M3J protocol,
input, storage, training, reporting and CLI suites; the reporting cases exercise
17-fit completed/in-progress reports for seeds 2–4 and preserve the existing
33-fit path. The [test log](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_deferred_frozen_v1/cpu_tests.log)
and [preparation summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/preparation_deferred_frozen_v1/summary.json)
retain the evidence and final runtime-source fingerprints/snapshot.

All three real-input preparations passed with unchanged training/validation
inputs, scaler provenance and scientific recipe. Each earlier preparation
manifest was preserved byte-for-byte before refresh. The seed-2 active-reference
check passed against the completed partial-joint package without requiring any
frozen-online output. Seeds 3 and 4 correctly record their partial-joint
predecessors as not started.

| Seed | Required partial-joint predecessor | Frozen-online | Refreshed receipt |
|---|---|---|---|
| 2 | Complete; input/recipe/completion binding passed | Deferred | [Manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed2_direct_v1/preparation/manifest.json) |
| 3 | Not started | Deferred | [Manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed3_direct_v1/preparation/manifest.json) |
| 4 | Not started | Deferred | [Manifest](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_seed4_direct_v1/preparation/manifest.json) |

All **2,202 pre-existing completed-package files** retained their presence,
size and modification time. The six tracked numerical training/input/storage/
model source files were unchanged. No model-weight audit, GPU verification,
comparative training, validation inference or cleaned-test application ran
during this amendment.

The CPU preparation command, repeated for seeds 2, 3 and 4, was:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_attention_fusion_joint_study.py \
  --config configs/multimodal_attention_fusion_joint_geo_only_seed2.yaml prepare
```

## Handoff and interpretation

The next comparative package is **one seed-2 geo-only fit**. Preparation alone
does not start it. Its ordinary training-only GPU/checkpoint verification
precedes the 50-epoch comparative fit. Then progress to seed-3 partial-joint →
geo-only and seed-4 partial-joint → geo-only, each as a separate package.

The full 132-fit design must not be labelled complete when the 84-fit current
scope finishes. Four-seed partial-joint architecture/initialization comparisons
remain possible; a four-seed partial-joint versus frozen-online conclusion
requires the deferred packages. Seed 1's existing regime contrast retains its
documented shuffle/crop limitation. When frozen-online is revisited, record a
new activation/reporting amendment and reuse preserved experts and geo results
where scientifically compatible rather than editing completed protocols or
silently changing their declared scope.

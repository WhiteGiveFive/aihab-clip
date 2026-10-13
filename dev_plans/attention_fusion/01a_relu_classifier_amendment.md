# A001 — Retain ReLU classification heads

**Status:** Applied on 2026-10-12, before comparative training.
**Study:** `attention_fusion_v1`; M1 remains complete, M2 remains planned.

The user confirmed retaining ReLU in classification heads for consistency with
the existing geo-helpfulness baselines. This is an amendment to the initial
[M1 protocol](01_experimental_protocol.md), not a change motivated by attention
experiment results. No attention-fusion development, selection or final-fit
run files existed when the amendment was applied.

## Changed specification

Every one of the seven candidates uses:

```text
classifier: Linear -> ReLU -> Dropout(0.1) -> Linear
```

The previous classifier used GELU. Adapters continue to use
`Linear -> GELU -> LayerNorm`; the attention and token-wise capacity FFNs also
retain GELU. The configuration's `model.activation: gelu_exact` refers to those
adapter/fusion operations; `model.classifier` separately specifies the head.

ReLU and GELU have no trainable parameters. All previously declared parameter
counts, dimensions, seeds, preprocessing arms, budgets, selection rules and
final-fitting routes therefore remain unchanged. Input and encoder identities
are unchanged; this amendment does not require repeating the input audit.

## Baseline naming and reuse clarification

The existing [geo-helpfulness expert configuration](../../configs/multimodal_geo_helpfulness.yaml)
and [input construction](../../multimodal/geo_helpfulness_oof.py) distinguish:

| Expert | Classifier input | First-layer weight shape |
|---|---|---|
| `image_only` | 1,152D L2-normalized image embedding | `[256, 1152]` |
| `geo_only` | 64D fitting-partition-standardized geo | `[256, 64]` |
| `raw_concat` | Image + standardized geo, no learned modality projection | `[256, 1216]` |

The saved development-validation checkpoint headers verify these shapes across
all four seeds. The generic pipeline's `tabular_projected_concat` is a different
mode; none of these three geo-helpfulness experts uses it. Standardization is
not a learned projection. The [existing classifier](../../multimodal/models.py)
already uses ReLU.

M2 should integrate and verify existing compatible baseline components. Reuse
of their architecture/code is distinct from reuse of fitted heads or reported
scores: the latter additionally requires matching the full encoder, input,
initialization, fitting and selection recipe. In particular, M1 still declares
component-derived initialization seeds and a three-learning-rate comparison.

## Preserved provenance and verification

The [snapshot index](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/protocol/amendments/A001_relu_classifier/snapshot.json)
records SHA-256 hashes and original/archived locations for nine files: the
original study configuration, governing guide, M1 record, preparer, preparer
tests, protocol manifest, resolved configuration, candidate matrix and input
audit. Archived files are copies of the pre-amendment bytes, not overwritten
historical results. The original effective configuration hash is:

```text
078e229c7c2437393f9f9e3376e06e7e1c97dc750ae8dbae2b27e5890a875d31
```

The current configuration and preparer enforce the ReLU classifier. The test
suite additionally rejects a return to the old GELU-head specification under
this amended study. The resolved configuration, candidate matrix and protocol
manifest are regenerated using the existing M1 preparer after this change.

Verification commands in the `habcls` environment, from the repository root:

```bash
python -m pytest -q tests/test_attention_fusion_input_audit.py tests/test_attention_fusion_protocol.py
python tools/prepare_attention_fusion_protocol.py
```

Verification: **24 tests passed**. All nine archived file hashes reproduce;
comparison of parsed configurations confirms that classifier activation is the
only effective setting changed. Local documentation links and whitespace pass.
The original input-audit evidence is retained because no inputs or audit code
changed.

The amended protocol keeps the original 19 arms, 228 planned development fits,
76 selected train-only references and 76 planned all-development refits. This
amendment performs no model fitting or test evaluation.

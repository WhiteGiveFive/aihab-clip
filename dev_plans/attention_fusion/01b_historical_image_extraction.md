# A002 — Preserve historical image-extraction numerics

**Status:** User approved on 2026-10-12, before any M3 comparative fit.
Applies to [M1](01_experimental_protocol.md) and [M2](02_models_and_training.md).

M1 fixed float32 encoder weights/features and reuse of authenticated validation
embeddings, but did not specify extraction backend arithmetic separately from
fusion/head optimization. M2 raw-image replay found that these distinctions
matter for the historical encoder producer.

The unchanged historical extractor and the new extractor agreed on CPU within
`5.96e-8`, but differed from the historical CUDA validation cache by approximately
`2.72e-4` on the checked rows. On the same RTX 4090, **CUDA batch 16, cuDNN TF32
enabled and matrix-multiplication TF32 disabled** reproduced the first 16 seed-1
validation vectors **bit-for-bit**. Disabling cuDNN TF32 for that batch produced
maximum absolute error `2.7177e-4`. The historical source files and recorded
package versions matched the current implementation; the discrepancy was
resolved by reproducing batch size and backend arithmetic.

The user explicitly approved preserving historical extraction, with strict
TF32-disabled fusion/head training retained. Therefore the image configuration
now records:

```yaml
extraction_device: cuda
extraction_batch_size: 16
extraction_cudnn_allow_tf32: true
extraction_matmul_allow_tf32: false
```

This does not change encoder weights, canonical assignments, legacy BGR/439
pixel preprocessing, image L2 normalization, architecture, geo treatments,
learning rates, training epochs, selection, final-refit rules or search budget.
There is no AMP. Float32 describes tensor storage; it does not imply TF32 is
disabled in every backend operation. Both TF32 flags remain disabled during
fusion/head training, independently of the extraction process.

Use canonical row order and extraction batch size 16. A CPU or differently
batched extraction cannot silently replace the historical representation under
the current tight replay gate. Check a complete historical validation batch for
every seed; retain the replay tolerance `atol=1e-5`, `rtol=1e-4` and record actual
errors. Do not loosen the gate to accept the earlier discrepancy. Regenerating
all validation representations under another numeric recipe is a future
explicit input change rather than an automatic fallback.

Evidence:

- [CPU old/new extractor comparison](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/verification/input_replay_failures/seed1_cpu_legacy_comparison.json).
- [CUDA batch-16 backend comparison](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/verification/input_replay_failures/seed1_cuda_batch16_comparison.json).
- [Pre-amendment snapshot](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_v1/protocol/amendments/A002_historical_image_extraction/snapshot.json): nine original files, including configuration, M1 report and generated protocol artifacts, preserved with checksums.

The snapshot's effective configuration hash is
`3fb5656b8291e7b7c84402396383a70a03ae2afa2e9fa4a1e88c1d91dfdbadee`.
The regenerated protocol manifest binds the amended configuration and matrix;
the amendment record preserves both before/after identities. Diagnostic failed
replays remain retained and are not model-comparison results.

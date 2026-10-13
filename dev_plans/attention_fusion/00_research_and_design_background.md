# Attention Fusion Research and Design Background

**Status:** Design background recorded; no attention-fusion experiments.
**Last updated:** 2026-10-12.

This record preserves the rationale behind the
[governing development guide](README.md). The literature survey was assembled on
2026-10-09; the singleton cross-attention clarification and development framing
were recorded on 2026-10-12. Published results below motivate candidate designs;
they do not establish a winner for the CS image + GSE task.

## What is established and what remains a hypothesis

The pipeline combines a 1,152D global image embedding and a 64D global geo
embedding. The dimensions can be reconciled with separate learned adapters.
That does not create image patches or convert latent geo coordinates into
independent semantic fields.

The working hypothesis is that attention over modality representations can
learn useful sample-dependent interactions beyond a matched MLP. Whether that
helps prediction, avoids harmful geo contributions, or just adds capacity is an
experimental question. Attention weights themselves are not geo-reliability
probabilities.

## Joint image-encoder and fusion training

The frozen regime holds image weights fixed while fitting fusion; the encoder
may already have been adapted on the allowed training plots. In the joint
regime, the fused classification loss also updates selected image-encoder
parameters. The initial extension is partial fine-tuning, with full vision
fine-tuning deferred. Geo embeddings remain fixed and their adapter learns.

The hypothesis is that visual features optimized for image + geo classification
may preserve different useful information from features optimized in isolation.
Geo influences the encoder's training gradients through fusion, while the
encoder still receives only image pixels in its forward pass. This need not
succeed when frozen attention succeeds, or fail when frozen attention fails.
It does not change the singleton-key attention limitation below.

Joint training requires an online differentiable image path; saved final
embeddings cannot update their producing encoder. The existing
[image adaptation implementation](../../methods/PEFT_openclip.py) uses prompt
supervision, so the new fused objective needs its own integration. Validation
must use the current encoder checkpoint and complete checkpoints must preserve
both encoder and fusion state.

Compare image-only, raw concatenation, projected concatenation and attention
under frozen-online and partial-joint regimes. Share starting weights, image
exposure, augmentation and declared budgets; final encoder weights may differ.
Otherwise a gain from image augmentation or ordinary image fine-tuning could be
misattributed to attention. M3J in the governing guide records this separate
experiment without requiring a positive frozen-attention result.

## Why one-key cross-attention is excluded

For image query \(q(x)\), a single geo key \(k(g)\), and value \(v(g)\):

\[
a=\operatorname{softmax}\left([q(x)^T k(g)/\sqrt{d_k}]\right)=[1],
\qquad o_x=v(g).
\]

Softmax normalizes over keys, not feature coordinates. Holding the source geo
vector fixed, this attention output is independent of the image query. The
reverse direction similarly produces a transformation of the image alone.
Additional heads do not fix the singleton softmax. Concatenating both branch
outputs gives separate modality transformations rather than adaptive weighting.
This follows directly from the [standard attention equation](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf).

A complete block may still compute a residual update such as
\(x'=\operatorname{FFN}(x+Ag)\), with normalization and further residuals.
Such a network can use both modalities. The limitation concerns attention
selection, not the usefulness of every residual fusion network. In deeper
networks, the current source vector may itself already contain both modalities.

Self-attention over `[image, geo]` gives each query two keys. The image output
can therefore mix its own value and the geo value with input-dependent weights.
This is a compact modality interaction, not selection between image regions.

For `[CLS, image, geo]`, the initial CLS query is a learned vector shared across
samples. In the first layer its image and geo scores are computed separately;
they compete through softmax but are not direct image–geo query/key scores.
Token updates occur in parallel. A second layer lets CLS read the first layer's
updated modality tokens; concatenating those updated tokens exposes them
directly after one layer.

## Relevant attention-fusion literature

The adaptation column describes a proposed use in this project, not a faithful
reproduction. Full published systems may also change encoders, losses, data,
pretraining and prediction ensembles.

| Work | Relevant mechanism | Implication for this method |
|---|---|---|
| [MISA, ACM Multimedia 2020](https://arxiv.org/html/2005.03545v3); [code](https://github.com/declare-lab/MISA) | Six compact shared/private modality vectors, one Transformer layer, concatenated updated outputs | Closest compact-vector precedent; our two-token design omits shared/private decomposition and auxiliary losses |
| [MulT, ACL 2019](https://aclanthology.org/P19-1656/); [code](https://github.com/yaohungt/Multimodal-Transformer) | Directional cross-attention between temporal modality sequences | A one-key global-vector copy loses selective weighting |
| [MBT, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/76ba9f564ebbc35b1014ac498fafadd0-Paper.pdf); [code](https://github.com/google-research/scenic/tree/main/scenic/projects/mbt) | Shared bottleneck tokens mediate audio/video patch streams | More relevant if many spatial/temporal tokens are retained later |
| [MetaFormer, 2022 preprint](https://arxiv.org/abs/2203.02751); [code](https://github.com/dqshuai/MetaFormer) | Joint attention over image tokens and location/time/attribute metadata | Ecological motivation from species recognition; full backbone training differs from cached-vector fusion |
| [MFT, IEEE TGRS 2023](https://arxiv.org/html/2203.16952); [code](https://github.com/AnkurDeria/MFT) | A complementary-sensor-derived CLS queries remote-sensing image tokens | Geo-conditioned queries become more interesting with multiple image tokens |
| [TIP, ECCV 2024](https://arxiv.org/html/2407.07582v1); [code](https://github.com/siyi-wind/TIP) | Tabular/CLS queries cross-attend to spatial image features, with self-supervised pretraining | Semantic table fields and image patches differ from our two global vectors |
| [STiL, CVPR 2025](https://arxiv.org/html/2503.06277v1); [code](https://github.com/siyi-wind/STiL) | Shared/specific features and attention combined with semi-supervised training | A broader extension; gains do not isolate a replacement fusion head |
| [DyMo, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bdc7dc3f5d20f82a27175199af9267e2-Abstract-Conference.html); [code](https://github.com/siyi-wind/DyMo) | Modality-token self-attention with CLS plus explicit selection of recovered modalities | Reliability selection is separate from attention; observed misleading geo is a different validation setting |
| [MultiModalPFN, CVPR 2026](https://arxiv.org/html/2602.20223v2); [code](https://github.com/too-z/MultiModalPFN) | Multiple learned projections of a global image vector, attention pooling, pretrained tabular backbone | Motivates later learned token expansion; it cannot recover spatial information discarded by pooling |

The MetaFormer reference above is **A Unified Meta Framework for Fine-Grained
Recognition**, not the separate PoolFormer/MetaFormer vision paper.

The initial BP repository inspiration is
[Hybrid-model-BP at e4308407](https://github.com/Ninja-FL/Hybrid-model-BP/tree/e4308407e0652f8c8dd398a40f9707db03bde5d3).
Its fusion design combines an updated CLS with original adapted modality
representations. That is a useful readout ablation, not a reason to assume the
same design is optimal here. Feature-level attention and the repository's
LightGBM/NN prediction blend are different operations; the latter is outside
this method's initial scope.

## Comparison lessons

- [DAFT](https://arxiv.org/abs/2107.05990) provides a nonattention reference for
  conditioning image channels with scale/shift derived from image and tabular
  information. A cached-vector version would be DAFT-inspired; original DAFT
  modifies internal image feature maps.
- [The Importance of Encoder Choice: A Tabular-Image Study](https://arxiv.org/html/2607.07756v1),
  a July 2026 preprint, finds strong encoders and simple bilinear fusion can
  compete with a more elaborate multimodal system. It motivates paired encoder
  provenance and strong simple controls, not a claim that attention never helps.
- [MulTaBench](https://arxiv.org/html/2605.10616v1), a May 2026 preprint, emphasizes
  multimodal complementarity and task-adapted representations. Its dataset
  curation favors those properties; its ranking is not a universal SOTA claim.
- There is no evidence in this survey that coordinate standardization is
  necessary for our learned geo embeddings. There is also no controlled
  attention-fusion result here establishing that native or L2 geo is better.

## Consequences for the first study

1. Start with a small two-token self-attention model and preserve both updated
   representations at readout.
2. Use matched image/geo adapters and preprocessing for projected concatenation.
3. Separate capacity, normalization, and readout changes in a bounded study.
4. Treat deeper CLS pooling as a distinct candidate; reserve multi-token
   expansion and spatial inputs for later evidence-driven extensions.
5. Measure classification gains and rescue/harm directly. Keep attention maps
   descriptive and retain the reused-data limitations of earlier project work.
6. Retain the frozen comparison and add partial joint fine-tuning as a separate
   regime, with matched online controls and a bounded recipe declared before fits.

[M1](01_experimental_protocol.md) turns these principles into the frozen initial
protocol, including optimizer, bounded search, interpretation and refit rules.
Those choices belong to the protocol, not the literature evidence. The detailed
online/unfreezing recipe is recorded before
M3J fitting; no joint-training implementation or results are claimed here.

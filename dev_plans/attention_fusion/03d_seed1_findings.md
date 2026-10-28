# M3J seed 1 performance findings

Recorded: **22 October 2026**. Scope: **33 completed seed-1 fits**, comprising
16 partial-joint fits, 16 frozen-online fits and one shared geo-only fit.
Results are **provisional, descriptive development-validation findings**.

CS adaptation improves image-only classification, while pretrained
initialization often performs better when the image representation is combined
with geo. Attention provides gains in some settings, but neither attention as
a family nor a particular readout consistently outperforms the alternatives.
Fusion usually improves on either individual modality, with the largest
observed gains occurring with pretrained image initialization.

The four principal findings are:

- **Initialization:** CS-adapted is stronger for image-only; pretrained generally
  leads the attention models at their best-validation checkpoints.
- **Concatenation versus attention:** gains depend on architecture, training
  regime and checkpoint policy. The capacity control remains competitive.
- **Attention readout:** CLS + original embeddings leads three of four settings
  at best validation; two-token attention leads the remaining setting. Final
  epoch rankings differ.
- **Modalities:** fusion usually improves image-only Acc@1 and always exceeds
  geo-only Acc@1 in these runs, but individual fusion choices can reduce performance.

The subsequent [geo-rescue opportunity analysis](03e_seed1_geo_rescue_opportunity.md)
measures which fusion errors the shared geo expert could correct, with
potential harms and oracle bounds kept separate from actual router performance.

## Experimental scope and interpretation

**Initialization and training regime are different experimental factors.**
Pretrained means the original SigLIP2 starting encoder; CS-adapted means the
historical seed-1 encoder after five CS adaptation epochs. Partial-joint
training updates the final ten visual Transformer blocks, final normalization
and attention-pooling head along with the downstream model. Frozen-online
training keeps the starting visual encoder fixed while training the downstream
model on images processed online. The geo-only reference has neither image
initialization nor image-encoder training regime.

All fits use the same 3,378/822 development train/validation allocation and
18-class mapping. Geo uses training-fitted coordinate standardization. Each
fit completes 50 epochs with batch 16, Adam at initial learning rate `5e-5`,
cosine decay and no early stopping. The full recipe and completion evidence
are in the [seed-1 completion record](03c_seed1_geo_only.md).

**Final** means epoch 50. **Best validation** means the single epoch selected
by unrounded validation Acc@1, then weighted F1, then MCC, retaining the earliest
epoch in an exact tie. Every metric reported for a policy comes from that same
saved checkpoint after loading. Acc@1 guides the descriptive comparisons below;
Acc@1 ties remain visible and do not imply equality in F1, MCC or Acc@3. No final
architecture selection or across-seed outcome category is assigned here.

The [combined numerical report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/report.md) contains all metrics, paired
rescue/harm counts, class-level results, confusion matrices and links to curves
and reusable checkpoints. All scores below use the **822-image development
validation set**, not the cleaned test set.

## Validation accuracy overview

All table entries are Acc@1 percentages. The two checkpoint policies are kept
separate. The capacity control processes modality tokens without attention
between them; it is not an attention model.

### Final epoch

| Method | Partial-joint, pretrained | Partial-joint, CS-adapted | Frozen-online, pretrained | Frozen-online, CS-adapted |
| --- | --- | --- | --- | --- |
| Image-only | 65.21 | 68.13 | 60.83 | 67.03 |
| Raw concatenation | 68.37 | 67.15 | 67.52 | 69.34 |
| Projected concatenation | 68.00 | 69.83 | 70.32 | 68.61 |
| Tokenwise capacity control | 71.29 | 68.98 | 68.86 | 68.98 |
| Two-token attention | 71.29 | 69.83 | 68.13 | 68.61 |
| CLS only | 71.17 | 68.61 | 70.56 | 67.40 |
| CLS + adapted bypass | 69.95 | 69.83 | 68.13 | 68.13 |
| CLS + original embeddings | 71.17 | 69.34 | 70.07 | 68.00 |

### Best validation checkpoint

| Method | Partial-joint, pretrained | Partial-joint, CS-adapted | Frozen-online, pretrained | Frozen-online, CS-adapted |
| --- | --- | --- | --- | --- |
| Image-only | 65.69 | 69.22 | 61.07 | 67.64 |
| Raw concatenation | 69.83 | 70.07 | 67.76 | 69.59 |
| Projected concatenation | 70.80 | 70.56 | 71.17 | 69.10 |
| Tokenwise capacity control | 71.65 | 70.44 | 69.83 | 69.22 |
| Two-token attention | 71.53 | 71.29 | 70.68 | 68.61 |
| CLS only | 71.53 | 68.61 | 71.65 | 68.61 |
| CLS + adapted bypass | 71.05 | 71.17 | 70.32 | 69.10 |
| CLS + original embeddings | 71.78 | 70.44 | 71.90 | 69.59 |

Geo-only is shared across all four columns above and is counted as one fit:

| Checkpoint | Epoch | Acc@1 (%) | Acc@3 (%) | Weighted F1 (%) | MCC |
| --- | --- | --- | --- | --- | --- |
| Final | 50 | 63.02 | 89.17 | 61.23 | 0.5730 |
| Best validation | 11 | 64.84 | 89.17 | 62.54 | 0.5947 |

Sources: [final metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/final_metrics.csv),
[best-validation metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/best_metrics.csv), and
[run inventory](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/inventory.csv).

## Encoder initialization findings

**CS adaptation consistently improves image-only classification.** Its Acc@1
advantage over pretrained initialization is **+2.92 percentage points at the
final epoch and +3.53 at best validation** under partial-joint training. Under
frozen-online training, the advantages are **+6.20 and +6.57 points**,
respectively. Weighted F1 and MCC support the same direction in all four
comparisons.

**The initialization preference changes for fusion.** At best validation,
pretrained initialization has higher Acc@1 in **seven of eight** matched
attention comparisons: four architectures within each of two training regimes.
The exception is partial-joint CLS + adapted bypass, where CS-adapted leads by
0.12 points, equivalent to one validation image. At the final epoch, pretrained
wins all four partial-joint attention comparisons; frozen-online two-token
attention instead favors CS-adapted by 0.49 points, and adapted bypass ties on
Acc@1.

Concatenation does not have one consistent initialization preference. At best
validation, raw concatenation favors CS-adapted in both regimes, whereas
projected concatenation favors pretrained in both. The final-epoch projected
concatenation comparison under partial-joint training reverses that preference.

A better image-only baseline therefore does not guarantee a better fusion
model. The observed initialization preference depends on architecture in
seed 1; these results do not establish why that pattern occurs or
show that CS adaptation is universally helpful or harmful.

## Concatenation and attention findings

The predefined primary comparison is **two-token attention minus projected
concatenation**, holding initialization, training regime and checkpoint policy
fixed. Positive values favor two-token attention.

| Setting | Final ΔAcc@1 (pp) | Best-validation ΔAcc@1 (pp) |
| --- | --- | --- |
| Partial-joint, pretrained | +3.28 | +0.73 |
| Partial-joint, CS-adapted | +0.00 | +0.73 |
| Frozen-online, pretrained | -2.19 | -0.49 |
| Frozen-online, CS-adapted | +0.00 | -0.49 |

Two-token attention improves best-checkpoint Acc@1 by 0.73 points under both
partial-joint initializations, but falls below projected concatenation by 0.49
points under both frozen-online initializations. At the final epoch, its
advantage is concentrated in partial-joint training with pretrained
initialization. Thus the primary comparison does not support a uniform
attention advantage.

The strongest observed attention model can outperform the strongest
concatenation model even where two-token attention does not:

| Setting | Final best-attention minus best-concat (pp) | Best-validation best-attention minus best-concat (pp) |
| --- | --- | --- |
| Partial-joint, pretrained | +2.92 | +0.97 |
| Partial-joint, CS-adapted | +0.00 | +0.73 |
| Frozen-online, pretrained | +0.24 | +0.73 |
| Frozen-online, CS-adapted | -0.73 | +0.00 |

This is a descriptive comparison of observed maxima: the attention family
contains four candidates and concatenation contains two. It carries additional
architecture-selection opportunity and does not replace the predefined primary
comparison. At best validation the attention maximum wins in three settings
and ties in frozen-online/CS-adapted; at the final epoch raw concatenation beats
the strongest attention model in that latter setting by 0.73 points.

**Extra model capacity remains a plausible contributor.** With partial-joint
training and pretrained initialization, tokenwise capacity and two-token
attention both reach **71.29% final Acc@1**. Two-token has slightly higher
weighted F1, while the capacity control has slightly higher MCC. The control's
best Acc@1 is **71.65%**, only one correctly classified validation image below
the strongest attention score of **71.78%**. Under frozen-online/CS-adapted
training, the control's final **68.98%** also exceeds every attention model.
These comparisons do not establish an improvement attributable specifically
to attention between the two modalities.

## Attention readout findings

Both bypass designs concatenate the updated CLS token with features from
before the attention layers. Adapted bypass uses the adapted modality tokens;
original-embedding bypass uses the unprojected output of the current image
encoder and the standardized geo vector. "Original" here describes the feature
space, not which starting encoder was used. The readouts also change classifier
input width and parameter count, so their differences do not isolate feature
bypass from classifier capacity.

At best validation, **CLS + original embeddings is the strongest observed
attention readout in three of four settings**:

| Setting | Leading attention readout by best Acc@1 | Best Acc@1 (%) |
|---|---|---:|
| Partial-joint, pretrained | CLS + original embeddings | 71.78 |
| Partial-joint, CS-adapted | Two-token attention | 71.29 |
| Frozen-online, pretrained | CLS + original embeddings | 71.90 |
| Frozen-online, CS-adapted | CLS + original embeddings | 69.59 |

The **final-epoch ordering changes**. Two-token attention leads under
partial-joint/pretrained and frozen-online/CS-adapted training. It ties with
adapted bypass at **69.83% Acc@1** under partial-joint/CS-adapted training;
adapted bypass has marginally higher weighted F1 in that tie. CLS-only leads
under frozen-online/pretrained training at **70.56%**.

Original-embedding bypass is consequently a promising candidate when the
best-validation checkpoint is retained, but is not the strongest final-epoch
readout. Adapted bypass has no consistent advantage over original bypass or
the simpler attention readouts. Several margins are small: for either
pretrained initialization regime, original bypass exceeds CLS-only best Acc@1
by 0.24 points, equivalent to two validation images.

**No readout dominates all metrics.** Under partial-joint/pretrained training,
original bypass has best-checkpoint Acc@1 **71.78%**, weighted F1 **71.42%** and
MCC **0.6769**, but Acc@3 **89.90%**. Projected concatenation has lower Acc@1
(**70.80%**) but higher Acc@3 (**92.34%**). Under partial-joint/CS-adapted training,
two-token attention leads best Acc@1 (**71.29% versus 71.17%** for adapted
bypass), while adapted bypass has higher weighted F1 (**70.38% versus 70.27%**).
The agreed Acc@1-first ordering should therefore accompany the other metrics,
not obscure their trade-offs.

## Image only geo only and fusion findings

**Fusion usually improves over either individual modality.** Every fusion
model and the tokenwise capacity control exceed geo-only Acc@1, weighted F1
and MCC under both checkpoint policies and all four image settings. Acc@3 is
not uniformly improved. Image-only exceeds geo-only Acc@1, F1 and MCC except
with frozen-online/pretrained initialization, where image-only scores
**60.83% final / 61.07% best Acc@1**, below geo-only's **63.02% / 64.84%**.

The table below compares the highest observed fusion/control Acc@1 against
image-only within the same setting and policy. These are selected maxima,
not the expected gain from every fusion architecture.

| Setting | Final fusion gain over image-only (pp) | Best-validation fusion gain over image-only (pp) |
| --- | --- | --- |
| Partial-joint, pretrained | +6.08 | +6.08 |
| Partial-joint, CS-adapted | +1.70 | +2.07 |
| Frozen-online, pretrained | +9.73 | +10.83 |
| Frozen-online, CS-adapted | +2.31 | +1.95 |

The largest observed fusion benefits occur with pretrained initialization.
For example, frozen-online/pretrained image-only reaches **61.07%** at best
validation, while CLS + original embeddings reaches **71.90%**. With CS-adapted
initialization, image-only is already stronger and the maximum incremental
fusion gain is smaller. This does not prove that adaptation removes useful
complementarity; it describes the observed performance interaction.

Fusion can also hurt. Under partial-joint/CS-adapted training:

- Raw concatenation's final Acc@1 is **67.15%**, below image-only's **68.13%**
  by 0.97 points.
- CLS-only best Acc@1 is **68.61%**, below image-only's **69.22%** by 0.61 points.

Both unimodal references and the stronger concatenation/control architectures
remain necessary for interpreting a proposed fusion improvement. The
[class-level final results](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/final_confusion_by_class.csv) and
[class-level best results](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/best_confusion_by_class.csv) provide
support counts, recalls and confusion evidence for follow-up error analysis;
aggregate scores alone do not establish improvement for every habitat class.

## Limits and implications for the remaining seeds

These are **one-seed findings on a reused development-validation cohort**.
The best checkpoint and the architecture maxima are selected using that same
cohort. One additional correct prediction changes Acc@1 by approximately
**0.122 percentage points**. Small differences, including the top readout and
capacity-control margins, do not establish reliable superiority. Validation
images also share plots, so image counts should not be treated as independent
replicate experiments. No across-seed claim of supported improvement is made.

The earlier **memory/read integrity incidents remain unresolved**. The
completion checks and saved-checkpoint evaluations passed, but the existing
provisional assurance label remains applicable.

The D041 audit also found that completed partial-joint and frozen-online
loaders used their **distinct study IDs for shuffle and image-crop random
streams**, despite declaring shared namespaces. Within each image regime,
architectures and initializations retain paired data streams; comparisons
between regimes include differences in minibatch order and realized crops.
They cannot isolate encoder trainability under identical inputs. The new
geo-only fit explicitly uses the declared shuffle namespace and no image
augmentation. Its data order should not be described as identical to the
completed image fits. The [completion record](03c_seed1_geo_only.md#d041-loader-audit-and-comparison-limitation)
records this issue and the separate unused-image audit failure.

For the remaining seeds, retain both initializations, both concatenation
baselines, the capacity control and all four attention readouts until the
planned comparisons can be assessed across seeds. Original bypass and two-token
attention are promising candidates, but seed 1 does not justify
promoting a single model or dropping the controls. Resolve and document the
cross-regime random-stream policy before subsequent packages, including how
seed 1 will be treated if that policy changes. This report records findings;
it does not amend the training recipe or start further experiments.

## Evidence and traceability

- [Combined 33-fit numerical report](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/report.md) and
  [machine-readable summary](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/summary.json).
- [Final metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/final_metrics.csv),
  [best metrics](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/best_metrics.csv),
  [final paired contrasts](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/final_contrasts.csv), and
  [best paired contrasts](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/tables/best_contrasts.csv).
- [Completion audit](../../multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/completion_audit.json): 33 fits, 1,650 epochs,
  66 checkpoint-policy evaluations, no incomplete fits.
- [Seed-1 completion record](03c_seed1_geo_only.md),
  [partial-joint record](03_joint_finetuning.md), and
  [frozen-online record](03b_seed1_frozen_online.md).

Tables use reloaded-checkpoint metrics from the summary above. Percentage-point
differences are calculated from unrounded proportions before formatting to two
decimal places. This narrative report leaves training artifacts and the
combined numerical report unchanged.

Source summary SHA256: `5ab5d83054b708219698ce0fea9e3ef57754dba86d7410e10e313f7b6984a5e1`.

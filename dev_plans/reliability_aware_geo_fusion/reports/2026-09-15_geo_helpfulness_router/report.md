# Geo-helpfulness router: progress summary

All habitat-accuracy and weighted-F1 comparisons below use the same cleaned test set of 1,347 images and the same expert models, averaged across four training seeds. Acc@1 is the accuracy of the top predicted habitat; weighted F1 averages the per-habitat F1 scores in proportion to each habitat's number of test images. The final confidence diagnostic separately examines earlier ConfidNet studies.

## 1. The router provides a small improvement over fusion

The router identifies cases where replacing the fusion model's prediction with the geo expert's prediction may help. The strongest observed result raises Acc@1 from **74.31% to 75.17%**, a **0.85 percentage-point improvement**, and weighted F1 from **0.738 to 0.747**.

| Model | Acc@1 (%) | Weighted F1 |
| --- | ---: | ---: |
| Geo expert | 61.62 | 0.608 |
| Image expert | 73.14 | 0.723 |
| Fusion expert | 74.31 | 0.738 |
| Fusion with router correction | **75.17** | **0.747** |

### Confusion matrices

![Confusion matrices for the geo expert, image expert, fusion expert, and fusion with router correction](figures/confusion_comparison.png)

Rows are true habitats and columns are predicted habitats. Values are normalized within each row, so stronger diagonal values indicate better habitat recall. The matrices combine the four runs over the same images; rows without test examples are marked.

Full-size matrices: [geo](figures/geo_normalized.png), [image](figures/image_normalized.png), [fusion](figures/fusion_normalized.png), [fusion with router](figures/router_normalized.png).

**Confusion remains.** Routing helps Improved Grassland, Coniferous Woodland, and Acid Grassland, but reduces recall for Neutral Grassland and Fen/Marsh/Swamp. The prominent Improved–Neutral Grassland confusion is redistributed rather than resolved: errors decrease in one direction and increase in the other. Dwarf Shrub Heath and Bog also remain frequently confused.

## 2. Ablations of router inputs and model type

We compared **Logistic Regression (LR)**, a simple linear model, with **Random Forest (RF)**, a collection of decision trees that can capture more complex relationships. Each model was tested with the same three input sets, all derived from expert predictions rather than raw images or geographic data.

| Input set | Information supplied to the router | What the ablation tests |
| --- | --- | --- |
| All features | Confidence, uncertainty, agreement and ranking summaries; each expert's predicted habitat; paired image–geo and geo–fusion habitat predictions | Reference using all available feature groups |
| Without paired habitat predictions | The same summaries and individual predicted habitats, with the two paired-prediction features removed | Whether combinations of predicted habitats add useful information |
| Numerical summaries only | Confidence, uncertainty, agreement and ranking summaries; all individual habitat labels and their pairs removed | Whether the router needs to know which habitats the experts predict |

For consistency, all results below use the relaxed **H≤0.7R** threshold-selection rule: at most seven newly introduced mistakes per ten corrected mistakes on development data. Thresholds were then frozen before test evaluation. The minimum coverage and plot-support requirements were held unchanged across these comparisons.

| Router input set | LR Acc@1 (%) | RF Acc@1 (%) |
| --- | ---: | ---: |
| All features | 75.00 | **75.17** |
| Without paired habitat predictions | 74.93 | **75.17** |
| Numerical summaries only | 75.09 | 75.13 |

These are habitat accuracies **after router correction**, compared with the fusion baseline of 74.31%. The opening result uses RF with all features; removing paired predictions ties its Acc@1. RF has slightly higher observed accuracy than LR across all three input sets, while the simpler input sets retain nearly all of the gain.

**Takeaway:** the router provides a modest observed improvement, important habitat confusions remain, and both input-feature and model-type ablations have been completed. The small gains are not yet statistically conclusive and require independent confirmation.

## 3. Learned confidence and auxiliary TCP supervision

We tested two ways to use **true-class probability (TCP)**, the probability an expert assigns to the correct habitat. Separate **ConfidNet** models estimate TCP and supply those scores as extra router inputs. **Joint MLP routers** learn to estimate TCP alongside their routing task, using it as an auxiliary training objective. Neither approach needs the true habitat at inference.

| Approach | What changes | Directly comparable cleaned-test results available? |
| --- | --- | --- |
| ConfidNet estimates added to RF | Add learned confidence scores to the numerical summaries  | No; the earlier test used different experts |
| MLP without embeddings | Compare ordinary router training with auxiliary TCP supervision | Yes, for small and larger models |
| MLP with image and geo embeddings | Add representations, comparing training with and without auxiliary TCP supervision | No; the earlier test used different experts |

**Comparable results without embeddings.** All values below are cleaned-test accuracies using Section 2's experts and threshold rule. Ranges cover all tested auxiliary weights and, for larger MLPs, all four architectures; no winner is selected. [Full comparison](../../20_joint_tcp_router_capacity.md).

| Router | Base inputs | Without auxiliary TCP (%) | With auxiliary TCP (%) |
| --- | --- | ---: | ---: |
| Small MLP | Numerical summaries only | 74.81 | 74.57–74.78 |
| Small MLP | Without paired habitat predictions | 74.31 | 74.31 |
| Larger MLPs | Numerical summaries only | 74.11–74.57 | 74.24–74.72 |
| Larger MLPs | Without paired habitat predictions | 74.31 | 74.31–74.59 |

Fusion reaches **74.31%**; the existing RF reaches **75.13%** with numerical summaries and **75.17%** without paired predictions. Auxiliary TCP helps some larger models relative to their own controls, but provides no consistent improvement. Larger models fail to surpass the small numerical MLP control. Most configurations using individual predicted habitats retain fusion. None exceeds the existing RF, and gains over fusion remain statistically inconclusive.

**Comparison still missing:** the [ConfidNet-input study](../../17_confidnet_fold4_cleaned_test.md) and [embedding study](../../19_joint_tcp_router_cleaned_test.md) used different expert models. Within those studies, ConfidNet inputs showed no clear benefit and embeddings generally weakened routing, with or without auxiliary TCP. Their absolute scores are excluded here: matching evaluations with this report's experts are still needed for a direct comparison with the existing RF.

### ConfidNet learns the rescue relationship but generalizes poorly

In a **rescue case**, geo predicts the correct habitat while fusion is wrong. Ideally, geo's estimated TCP would be higher. Rechecking the saved predictions shows this relationship on training data, but much less reliably on unseen examples:

| ConfidNet training approach | Training | Validation | Cleaned test |
| --- | ---: | ---: | ---: |
| Pooled OOF examples | 98.45% | 50.00% | Not evaluated |
| Single fold 4 | 98.49% | 41.40% | 46.24% |

Values are the percentage of actual rescue cases where **estimated geo TCP exceeds estimated fusion TCP**, pooling the four seeds. They are confidence-ordering diagnostics from the earlier expert sets, not habitat accuracies comparable to the main tables. Sources: [pooled OOF study](../../14_confidnet_oof_experiment.md), [fold-4 study](../../16_confidnet_fold4_experiment.md), and [cleaned-test study](../../17_confidnet_fold4_cleaned_test.md).

The training examples were seen by ConfidNet, even when the expert predictions were OOF. The near-perfect training relationship therefore does not establish that the confidence scores generalize. On cleaned test, estimated TCP ranked geo higher in only **172 of 372 rescue cases**, whereas true TCP did so in **371 of 372**. The useful relationship was present in the targets, but ConfidNet often failed to preserve it on new examples.

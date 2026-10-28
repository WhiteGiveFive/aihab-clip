# R4 — Validation findings and handoff to the main attention-fusion study

Recorded: **23 October 2026**. Status: **Complete: analysis, saved-stage
verification and independent numerical audit passed.** This report interprets the completed
[R3 routing matrix](03_validation_routing.md) under the frozen
[R1 protocol](01_validation_protocol.md). The [study README](README.md) governs
the milestone boundary and the return to the main attention-fusion study.

## 1. Question, scope and limits

The question is whether a fixed geo-only expert can selectively correct a
fixed attention-fusion expert using their candidate classes' neighborhoods in
image and geo feature spaces. The fusion architecture is
`attention_cls_skip_original`, with pretrained initialization and partial-joint
training, seed 1. The two checkpoint policies are fusion/geo epochs **50/50**
(`final`) and **36/11** (`best`). All results concern the same **822 development
validation images / 325 plots**, with **3,378 separate training images / 1,300
plots** as the labeled reference bank.

Within each checkpoint policy, both geometry arms use identical expert
predictions, candidate classes, query identities, reference identities, native
geo distances and class-support requirements. Only the image encoder used to
build image geometry changes: original pretrained versus the fusion expert's
matching trained visual checkpoint. The original pretrained geometry encoder
is shared across the two policies. Final and best change both experts and are
related measurements on the same queries, not independent replications.

R4 describes the saved R3 actions. It does not reselect experts or thresholds,
retrain models, regenerate embeddings, change reference support, or evaluate
cleaned-test data. True labels appear only in evaluation and descriptive
outcome grouping. They are not routing features. The existing
`provisional_unresolved_memory_integrity` assurance remains attached to all
findings; successful artifact verification does not resolve machine health.

The selected policies were fitted and reported on the same validation labels.
Even the fixed-zero rule is now being compared across representations and
checkpoint policies on reused validation. These results support a development
proof of concept, not an independently established population improvement.

## 2. All eight outcomes and conditional uncertainty

All rows retain the R3 policy and actions. `Δ` and interval endpoints are percentage points relative to the matching fusion baseline (**71.17% final; 71.78% best**).

| Expert pair / image geometry | Rule | Actions / plots | Rescues / harms | Net | Acc@1 % | Δ pp | Conditional 95% interval, pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final / original pretrained | fixed zero | 97 / 66 | 47 / 24 | +23 | 73.97 | +2.80 | [+0.24, +5.40] |
| Final / original pretrained | selected | 85 / 58 | 42 / 18 | +24 | 74.09 | +2.92 | [+0.60, +5.36] |
| Final / checkpoint-matched | fixed zero | 9 / 7 | 2 / 4 | -2 | 70.92 | -0.24 | [-0.97, +0.37] |
| Final / checkpoint-matched | selected (never) | 0 / 0 | 0 / 0 | +0 | 71.17 | +0.00 | [+0.00, +0.00] |
| Best / original pretrained | fixed zero | 108 / 77 | 52 / 24 | +28 | 75.18 | +3.41 | [+0.84, +6.08] |
| Best / original pretrained | selected | 107 / 77 | 52 / 23 | +29 | 75.30 | +3.53 | [+0.97, +6.15] |
| Best / checkpoint-matched | fixed zero | 16 / 13 | 4 / 4 | +0 | 71.78 | +0.00 | [-0.75, +0.73] |
| Best / checkpoint-matched | selected (never) | 0 / 0 | 0 / 0 | +0 | 71.78 | +0.00 | [+0.00, +0.00] |

All five metrics and their deltas are retained in the [R4 numeric summary](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/summary.json); the [R3 report](03_validation_routing.md) also gives every Acc@3, macro F1, weighted F1 and MCC value. The [conditional-interval figure](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/figures/conditional_acc1_intervals.png) presents all eight intervals.

The four original-pretrained intervals are above zero under this conditional resampling calculation. Both checkpoint-matched fixed-rule intervals include zero. This is a statement about these saved decisions and this resampling model, not an independent significance claim.

The conditional interval resamples **all 325 whole plots with replacement**,
including plots without interventions, using **10,000 replicates, seed
20261023**. Each replicate computes total rescues minus harms divided by total
images in its sampled plots. The reported limits are the 2.5th and 97.5th
percentiles with linear quantile interpolation. This preserves the primary
image-weighted Acc@1 estimand while keeping photographs from the same plot
together. It is not a photograph-level bootstrap or an equal-plot average.

Experts, checkpoint choices, geometry, selected thresholds and resulting
actions are fixed during resampling. The intervals condition on those choices;
they do **not** account for threshold-selection optimism, validation reuse,
representation selection or training-seed variability. In particular, a
never-intervene policy has a mechanically zero difference and a [0, 0]
interval; this is not evidence that an untrained alternative router has no
potential benefit. No interval is used to select a new policy in R4.

The fixed-zero pretrained results already contain almost all of the selected
rules' gain. Selection increases net correct predictions by one for each
expert pair. The selected final rule changes 97 actions to 85, reducing
rescues from 47 to 42 and harms from 24 to 18; the selected best rule removes
one harmful action from the fixed rule, retaining all 52 rescues. The final
rule therefore exchanges some corrections for fewer errors, while the best
rule differs only slightly from its fixed-zero counterpart on these data.

## 3. Which habitats gain or lose?

The table compares the **selected original-pretrained rule** with its matching fusion baseline. `R/H` gives rescued/harmed images; recall and class-F1 changes are percentage points. Full baseline/routed recall and F1 values, false positives, false negatives and macro-F1 contributions are preserved in the [final habitat table](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/evaluations/final__pretrained/selected/habitat_metric_changes.csv) and [best habitat table](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/evaluations/best__pretrained/selected/habitat_metric_changes.csv).

| True habitat | Images / plots | Final R/H | Final net | Final Δ recall | Final Δ F1 | Best R/H | Best net | Best Δ recall | Best Δ F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Urban | 3 / 1 | 0/0 | +0 | +0.00 | +0.00 | 0/0 | +0 | +0.00 | +0.00 |
| Broadleaved Mixed and Yew Woodland | 54 / 21 | 0/0 | +0 | +0.00 | +1.53 | 0/0 | +0 | +0.00 | +0.00 |
| Coniferous Woodland | 42 / 19 | 2/0 | +2 | +4.76 | -0.10 | 2/0 | +2 | +4.76 | -0.02 |
| Arable and Horticulture | 158 / 71 | 6/1 | +5 | +3.16 | +3.77 | 5/1 | +4 | +2.53 | +3.36 |
| Improved Grassland | 149 / 57 | 11/0 | +11 | +7.38 | +3.97 | 14/1 | +13 | +8.72 | +3.58 |
| Neutral Grassland | 114 / 42 | 4/3 | +1 | +0.88 | +2.01 | 8/6 | +2 | +1.75 | +2.40 |
| Calcareous Grassland | 6 / 2 | 0/0 | +0 | +0.00 | +26.67 | 0/1 | -1 | -16.67 | +22.22 |
| Acid Grassland | 80 / 32 | 7/4 | +3 | +3.75 | +4.38 | 6/5 | +1 | +1.25 | +1.57 |
| Bracken | 12 / 5 | 1/2 | -1 | -8.33 | +3.99 | 0/2 | -2 | -16.67 | +10.39 |
| Dwarf Shrub Heath | 71 / 25 | 4/4 | +0 | +0.00 | +0.00 | 5/4 | +1 | +1.41 | +5.78 |
| Fen, Marsh, Swamp | 22 / 9 | 0/1 | -1 | -4.55 | -0.84 | 0/0 | +0 | +0.00 | +8.95 |
| Bog | 95 / 36 | 7/2 | +5 | +5.26 | -0.21 | 12/2 | +10 | +10.53 | +2.40 |
| Littoral Rock | 0 / 0 | 0/0 | +0 | n/a | +0.00 | 0/0 | +0 | n/a | +0.00 |
| Littoral Sediment | 4 / 1 | 0/0 | +0 | +0.00 | +0.00 | 0/0 | +0 | +0.00 | +0.00 |
| Montane | 3 / 1 | 0/0 | +0 | +0.00 | +0.00 | 0/0 | +0 | +0.00 | +0.00 |
| Inland Rock | 4 / 1 | 0/0 | +0 | +0.00 | +0.00 | 0/0 | +0 | +0.00 | +0.00 |
| Supra-littoral Rock | 3 / 1 | 0/0 | +0 | +0.00 | +0.00 | 0/0 | +0 | +0.00 | +0.00 |
| Supra-littoral Sediment | 2 / 1 | 0/1 | -1 | -50.00 | -66.67 | 0/1 | -1 | -50.00 | -40.00 |

The [habitat net-change figure](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/figures/habitat_net_changes.png) displays the gain/loss counts with class support.

Class recall is the fraction of examples **of that true habitat** classified
correctly. Class F1 also depends on false positives arriving from other true
habitats. Consequently, an unchanged number of correct examples does not
imply unchanged class F1. Macro F1 gives every one of the 18 classes equal
weight, including zero-denominator F1 terms recorded as zero. Weighted F1 and
Acc@1 give common classes more influence. Littoral Rock has no true validation
examples, so its class recall and recall change are not estimable.

Improved Grassland provides the largest net gain in both policies: **+11**
final and **+13** best. Bog contributes **+5/+10** and Arable and Horticulture
**+5/+4**. Those three true habitats supply 21 of the final policy's 24 net
corrections and 27 of the best policy's 29. Net losses occur in Bracken and
Supra-littoral Sediment for both policies, Fen/Marsh/Swamp for final, and
Calcareous Grassland for best.

Final-policy macro F1 falls **48.06% → 46.87%**, despite improved Acc@1 and
weighted F1 (**70.50% → 72.79%**). The loss of one correct Supra-littoral
Sediment image changes that two-image class's F1 **66.67% → 0%** and contributes
**−3.70 percentage points** to the 18-class macro average. Improvements in
other class-F1 terms partially offset it; the overall macro change is
**−1.19 points**. This is a real metric tradeoff, strongly influenced by tiny
class support, rather than a contradiction in the calculations.

The distinction between recall and F1 is visible elsewhere. For final
Calcareous Grassland, recall stays at 50%, while F1 rises **40.00% → 66.67%**
because false positives fall from six to zero. For final Bog, recall rises
**67.37% → 72.63%**, but F1 slightly decreases **65.31% → 65.09%** as false
positives increase from 37 to 48. For best Bracken, recall falls **58.33% →
41.67%** while F1 increases **45.16% → 55.56%** because false positives fall
from 12 to one. A claim that every habitat improves would therefore be false,
even where overall macro F1 increases. The best selected policy improves
macro F1 **54.74% → 55.88%** and weighted F1 **71.42% → 74.43%**.

These habitat results are descriptive. Tiny supports, sometimes one plot,
make large percentage changes especially unstable. They do not establish a
class-specific exception rule or authorize disabling routing for a habitat
using its unknown true label.

## 4. Plot concentration and confusion transitions

The selected original-pretrained policies spread rescues across multiple plots, but harms are not confined to one outlier.

| Quantity | Final selected | Best selected |
| --- | ---: | ---: |
| Acted plots | 58 | 77 |
| Plots with any rescue | 29 | 38 |
| Plots with any harm | 13 | 19 |
| Plots with both rescue and harm | 0 | 0 |
| Positive / negative / zero net-change plots | 29/13/283 | 38/19/268 |
| Harms in most affected plot | 3/18 | 3/23 |
| Harms in top three harm plots | 7/18 (38.9%) | 7/23 (30.4%) |
| Harms in top five harm plots | 10/18 (55.6%) | 9/23 (39.1%) |
| Equal-plot-weighted Acc@1 change, pp | +2.85 | +3.26 |

Plot `1155XX5` has three harmed images under both selected policies, the largest single-plot count. The final top-three harm plots are `1155XX5`, `588XX1` and `950XX1`; the best counterparts are `1155XX5`, `182XX1` and `956XX1`. The listed order breaks equal harm counts by plot ID. Removing these plots or tuning to them is not part of R4.

The equal-plot-weighted changes average within-plot accuracy changes over all 325 plots. They are a descriptive secondary view; the primary result and bootstrap remain image-weighted.

Event plot counts overlap: one plot can contain rescued and harmed images.
The saved tables therefore retain both mutually exclusive event categories
(rescue only, harm only, mixed, neither) and the separate positive/negative/
zero net-change categories. A plot can also contain interventions for which
both experts are wrong; those actions do not change Acc@1 but do change the
confusion structure and potentially other metrics.

The best selected rule corrects eight Improved Grassland images that fusion
called Neutral Grassland and seven Bog images that fusion called Dwarf Shrub
Heath. Its harms include five true Neutral Grassland images redirected to
Improved Grassland and five true Acid Grassland images redirected to Bog.
The final selected rule similarly rescues six Improved Grassland images from
Neutral Grassland predictions but harms four Acid Grassland images by
redirecting them to Bog. These observed confusion directions help explain why
net benefit and class-specific false-positive changes differ. They do not
establish ecological causes without image/label review.

The complete per-plot and true/fusion/geo transition tables accompany all
eight evaluations, including the unsuccessful checkpoint-matched fixed rules
and never-intervene selected policies. Reporting only positively changing
plots would overstate how widely the improvement occurs.

## 5. What correctable fusion errors remain?

Every row below retains its original R3 rule; values are image counts, including unavailable geometry in the denominator.

| Pair / geometry | Rule | Potential rescues | Rescued | Missed | Unsupported | Image fails only | Geo fails only | Both fail | Positive but below/equal threshold | Positive but never-policy blocked |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final / pretrained | fixed_zero | 77 | 47 | 30 | 3 | 19 | 4 | 4 | 0 | 0 |
| final / pretrained | selected | 77 | 42 | 35 | 3 | 19 | 4 | 4 | 5 | 0 |
| final / matched | fixed_zero | 77 | 2 | 75 | 3 | 64 | 0 | 8 | 0 | 0 |
| final / matched | selected | 77 | 0 | 77 | 3 | 64 | 0 | 8 | 0 | 2 |
| best / pretrained | fixed_zero | 84 | 52 | 32 | 3 | 19 | 8 | 2 | 0 | 0 |
| best / pretrained | selected | 84 | 52 | 32 | 3 | 19 | 8 | 2 | 0 | 0 |
| best / matched | fixed_zero | 84 | 4 | 80 | 3 | 67 | 2 | 8 | 0 | 0 |
| best / matched | selected | 84 | 0 | 84 | 3 | 67 | 2 | 8 | 0 | 4 |

For the selected pretrained rules, image geometry alone blocks 19 otherwise geo-positive rescue opportunities in each checkpoint policy. Three opportunities in each policy lack reference support. The final selected threshold additionally declines five cases positive in both spaces; the best selected threshold declines none of those potential rescues. These are limits of the declared policy on this validation set, not permission to infer a new threshold from the examples.

These categories form a disjoint image-count partition. Unsupported rows take
precedence; supported rows then separate failure of the strict image or geo
zero gate, and rows positive in both spaces can still be blocked by a finite
threshold or the never-intervene policy. Plot counts for those categories
overlap and cannot be added. Zero scores fail the declared strict `>` rule.

The correction ceiling is bounded by the fixed geo expert: it cannot rescue
fusion errors when it is also wrong. Final and best expert pairs offer 77 and
84 potential rescues, yielding top-1 oracle bounds **662/822 = 80.54%** and
**674/822 = 82.00%**. Those oracle bounds do not imply an available deployable
policy, and no oracle Acc@3, F1 or MCC is defined.

## 6. Why does checkpoint-matched geometry allow so few interventions?

Only the image geometry differs within a policy; both reference/query identity and the native-geo distances and scores were checked across arms.

| Expert pair / image geometry | Eligible disagreements | Median image score, all eligible | Median image score, potential rescues | Positive image score | Positive image and geo scores | Rescues with both positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final / pretrained | 279 | -0.0395 | 0.0518 | 112 | 97 | 47 |
| final / checkpoint_matched | 279 | -2.9970 | -2.9235 | 9 | 9 | 2 |
| best / pretrained | 284 | -0.0128 | 0.0643 | 128 | 108 | 52 |
| best / checkpoint_matched | 284 | -2.6755 | -2.4658 | 19 | 16 | 4 |

The [image-score distribution figure](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/figures/image_geometry_distributions.png) shows the full eligible distributions; paired changes and per-query masks are saved under `geometry/final/` and `geometry/best/`.

The image score is

\[
A_{\mathrm{image}}=
\log\frac{d_{\mathrm{image}}(c_f)+10^{-8}}
{d_{\mathrm{image}}(c_g)+10^{-8}}.
\]

A negative value means the candidate class proposed by fusion is closer in
the image geometry; a positive value favors the class proposed by geo. A
switch requires both this score and the unchanged geo score to exceed the
threshold. The geometry diagnostics compare each query with itself across
representations, not independently sorted marginal distributions.

Image scores decrease for **265/279** final and **266/284** best eligible
queries when replacing pretrained geometry with checkpoint-matched geometry.
The median paired changes are **−3.0080** and **−2.6588**, respectively.
The shift also occurs among correctable fusion errors, not only
among cases where fusion is right: their median image scores change from
**+0.0518 to −2.9235** (final) and **+0.0643 to −2.4658** (best).

Native geo remains positive on **66** supported potential rescues for final
and **71** for best. Pretrained image geometry blocks 19 of these in each
policy; trained image geometry blocks **64/66** and **67/71**. In the paired
comparison, **45 final** and **48 best** potential rescues move from positive
pretrained image scores to nonpositive trained scores; none of the
geo-positive potential rescues moves in the reverse direction. The resulting
fixed-zero rescue counts shrink **47 → 2** and **52 → 4**.

Across all eligible outcomes, the positive-both pool shrinks **97 → 9** final
and **108 → 16** best. Those checkpoint-matched interventions cover only seven
and 13 plots at threshold zero. Any allowed higher threshold can only select
a subset, so the 20-plot selection guard is impossible to meet. The resulting
never-intervene fallback is therefore entailed by the declared recipe; it is
not a failure of the threshold implementation.

This explains the arithmetic of the intervention shortage. It does not prove
representation collapse, establish the mechanism of training, show that the
trained encoder is generally worse, or demonstrate that every neighborhood
is ecologically meaningful. A plausible hypothesis is that geometry from a
representation outside the fitted fusion encoder supplies a different view
of its mistakes. Testing that hypothesis requires additional seeds or a
separately declared representation study; R4 records it without adding such
experiments or weakening the existing routing guards.

## 7. Deterministic numerical examples

`UID` is a shortened display of the full saved row identity. Scores are rounded only for display; R3 comparisons use the saved float64 values and exact policies.

| Pair | Outcome | UID / plot | True class | Fusion → geo class | Image / geo score | Decision |
| --- | --- | --- | --- | --- | ---: | --- |
| final | rescue | `007ad2ec3e06` / `785X3` | Acid Grassland | Neutral Grassland → Acid Grassland | 0.3819 / 0.1479 | switch |
| final | harm | `06b97087be84` / `950XX1` | Bracken | Bracken → Bog | 0.2140 / 0.4890 | switch |
| final | both_wrong_action | `0789841e40f2` / `903XX4` | Acid Grassland | Bracken → Coniferous Woodland | 0.2286 / 1.5652 | switch |
| final | missed_rescue | `031e1eb3262d` / `906XX5` | Acid Grassland | Bog → Acid Grassland | 0.0087 / 0.6558 | retain fusion |
| best | rescue | `007ad2ec3e06` / `785X3` | Acid Grassland | Neutral Grassland → Acid Grassland | 0.3819 / 0.1479 | switch |
| best | harm | `2c404085c220` / `101XX1` | Neutral Grassland | Neutral Grassland → Improved Grassland | 0.0854 / 0.0368 | switch |
| best | both_wrong_action | `042f1ffa3258` / `11XX5` | Bracken | Calcareous Grassland → Acid Grassland | 0.1970 / 0.1611 | switch |
| best | missed_rescue | `06a563e982c5` / `384X4` | Broadleaved Mixed and Yew Woodland | Coniferous Woodland → Broadleaved Mixed and Yew Woodland | 0.2496 / -0.1310 | retain fusion |

The final missed-rescue example has a positive minimum score **0.008703**, below its selected threshold **0.0257636**. The best missed-rescue example has positive image support but negative geo support (**−0.130952**), so it also fails the fixed-zero rule. The rescue example is shared between policies because the candidate classes and original-pretrained/native-geo geometry are the same for that row.

Full details: [final selected case index](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/evaluations/final__pretrained/selected/case_index.csv) and [best selected case index](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/evaluations/best__pretrained/selected/case_index.csv).

The saved case index selects the first three rows in ascending `row_uid` for
each of rescue, harm, both-wrong intervention and missed rescue, separately
for every evaluation. The examples above use the first row of each category
for the selected original-pretrained rules. They are deterministic examples,
not selected for visual persuasiveness or claimed to be representative.

This is a **numerical case review**, using recorded labels, candidate classes,
scores and actions. R4 did not inspect the photographs or adjudicate habitat
labels ecologically. Their filenames and full identities remain in the case
index for a separately scoped visual review if needed. A both-wrong action
changes the predicted class without correcting the image; a missed rescue
identifies an opportunity that the declared policy declines, not proof that
its decision was detectably wrong at inference time.

## 8. Interpretation and handoff

The evidence supports continued study of original-pretrained geometry as a
router for this fixed fusion/geo pair. The useful signal is present with the
simple fixed-zero rule and with both checkpoint policies; it is not solely a
product of a finely tuned threshold. Nonetheless, harms remain substantial,
gains differ by habitat, and the selected final policy trades macro F1 against
overall accuracy. There is no universally better outcome across all metrics
and habitats.

The paired geometry analysis identifies an important failure mode of the
present checkpoint-matched variant: its image-distance ratios favor fusion's
candidate on many cases where geo is actually correct, suppressing possible
switches before threshold optimization. Both selected policies correctly
retain fusion because no candidate satisfies all declared constraints. These
negative results remain part of the study; the 20-plot guard and nonnegative
threshold range are not relaxed retrospectively.

The routing exploration now hands back to the
[main attention-fusion study](../attention_fusion/README.md). Its state remains
**M3J 33/132 fits complete; 99 remain**. The next training package is **seed-2
partial-joint, 16 fits**, followed by that seed's frozen-online and geo-only
work under the existing order, then later seeds and the governed M4/M5 process.
R4 adds no model fits and does not start that package. It does not change the
main study's recipe, select a routed model as its winner, or substitute the
present seed-1 result for the planned cross-seed comparisons.

The transfer record to preserve includes all four expert/geometry comparisons,
fixed and selected rules, exact saved policies, failed finite candidates,
per-query source probabilities, class/plot identities, reference support,
image and native-geo preprocessing, and the provisional assurance. Future
work should compare rescue/harm patterns and representation effects across
seeds before drawing broader conclusions. Any new geometry, guard, threshold
search or class-specific rule requires an explicit protocol amendment and
separate results, rather than rewriting these completed outcomes.

## 9. Requirements before the deferred R5 cleaned-test application

R5 remains deferred until the main attention-fusion study is complete. Before
reading cleaned-test outcomes for router decisions, record a separate
prospective transfer protocol that specifies:

1. **Final expert identities and fitting route.** Bind the selected fusion and
   geo checkpoints, seeds, class order and whether they were fitted on the
   development training split or refitted on all development data. The current
   seed-1 epoch-36/11 pair must not silently stand in for later finalized
   experts.
2. **Geometry encoder and extraction lineage.** Declare original-pretrained
   or checkpoint-matched geometry, with matching reference/query extraction,
   preprocessing and numeric settings. A new joint encoder or all-development
   refit requires compatible geometry; reuse caches only when their source
   identities genuinely match.
3. **Allowed reference bank and separation.** Declare which labeled
   development rows/plots form the bank for that fitting route, preserve the
   strict five-distinct-plot rule and class support, and verify separation from
   cleaned-test query rows and plots. The present 3,378-row bank is specific to
   this validation experiment and is not automatically the later test bank.
   Changing the reference bank can change distances, support and the score
   distribution even if the query encoder remains the same.
4. **Routing recipe fixed before test inspection.** Declare the representation,
   fixed rule or development-selected threshold procedure, all guards,
   tie-breaking, probability-vector selection and metrics. If the finalized
   experts require new thresholds, select them only using an appropriate
   declared development procedure; do not optimize them on cleaned-test
   outcomes or assume old numeric thresholds transfer unchanged.
5. **Evaluation and interpretation.** Preserve image and plot denominators,
   habitat gains/harms, support failures and all unfavorable outcomes. Record
   previous dataset/model-selection exposure. The planned cleaned-test
   application is **exploratory**, not independent confirmation, and retains
   that status even when its recipe is fixed before this particular run.
6. **Reproducible artifacts and assurance.** Save resolved configurations,
   source identities, exact policies, probabilities, per-query decisions,
   reference support and verification receipts. Preserve the current
   provisional assurance unless separate evidence justifies changing it.

R4 does not load cleaned-test data, select a test threshold or authorize R5
execution. The record above makes the later transfer requirements concrete
while keeping the main-study completion as the prerequisite.

## 10. Implementation, saved artifacts and evidence

The analysis is implemented in:

| Component | Responsibility |
| --- | --- |
| [R4 configuration](../../configs/multimodal_attention_fusion_router_analysis.yaml) | Fixed eight-evaluation analysis scope, bootstrap recipe and deterministic case-index convention |
| [Stage adapter](../../multimodal/attention_fusion_router_analysis.py) | Verify source bindings; generate habitat, plot, uncertainty, case and paired-geometry outputs; preserve earlier stages |
| [Outcome helpers](../../multimodal/attention_fusion_router_outcomes.py) | Whole-plot aggregation, confusion transitions, missed-rescue partition and conditional bootstrap |
| [Geometry helpers](../../multimodal/attention_fusion_router_geometry_analysis.py) | Paired identity/geo checks, strict sign quadrants, distributions, crossings and rescue opportunities |
| [CLI](../../tools/analyze_attention_fusion_router.py) | Public `run` and `verify` commands |

From the repository root in the `habcls` environment:

```bash
python tools/analyze_attention_fusion_router.py run \
  --config configs/multimodal_attention_fusion_router_analysis.yaml
python tools/analyze_attention_fusion_router.py verify \
  --config configs/multimodal_attention_fusion_router_analysis.yaml
```

These are CPU analyses of saved development-validation records. The completed
stage lives at:

```text
multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/
  seed1_pretrained_cls_original_v1/r4_analysis_v1/
    protocol.json
    source_snapshot/
    summary.json
    results.md
    evaluations/<four_cells>/<fixed_zero|selected>/
      habitat_metric_changes.csv
      plot_outcomes.csv
      plot_summary.json
      missed_rescues.json
      confusion_transitions.csv
      conditional_uncertainty.json
      case_index.csv
    geometry/<final|best>/
      summary.json
      paired_queries.parquet
    figures/
      habitat_net_changes.{png,pdf}
      conditional_acc1_intervals.{png,pdf}
      image_geometry_distributions.{png,pdf}
    preservation.json
    manifest.json
```

The [numeric summary](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/summary.json),
[generated result tables](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/results.md),
[source/configuration record](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/protocol.json),
[preservation receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/preservation.json) and
[completion manifest](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r4_analysis_v1/manifest.json)
are the primary numerical and provenance records.

**239 focused tests passed in 4.06 seconds**, including the outcome and paired
geometry tests; compilation and diff checks passed. Tests cover strict score
boundaries, unchanged geo and candidate identities, unsupported NaNs, paired
shifts, plot aggregation, exhaustive missed-rescue partitions and the
conditional-resampling calculation. The actual R4 `run` and `verify` commands
both completed successfully. The [execution receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/r4_execution_checks.json)
records the checks, matching implementation snapshots and **84 stage files /
1,392,214 bytes** (approximately 1.39 MB). Visual inspection of all three PNG
figures confirmed readable labels, unclipped content and agreement with their
underlying tables; the interval figure also states its conditional uncertainty
and validation-reuse limits.

The [independent numerical audit](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/r4_independent_audit.json)
passed using [standalone scalar aggregation and direct NumPy calculations](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/r4_independent_audit.py),
without importing the production analysis helpers. It checked every one of
the eight evaluations, all **2,600 per-plot rows**, **1,644 paired geometry
rows**, **144 habitat-F1 records** and **80,000 bootstrap replicates**, together
with the missed-rescue partitions, confusion transitions and deterministic
case indices. It also confirmed that **184 preceding router artifacts and 17
bound original source files** remained unchanged.

All original R1/R2 inputs and scores, R3 predictions/policies/reports, M3J model
checkpoints and bound source files remain preserved. R4 publishes a separate
analysis stage with its own configuration, source snapshot, manifest and
verification. Historical R1/R2 false-routing flags and R3 completion records
continue to describe their original stages.

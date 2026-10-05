# Larger MLPs for the joint four-state/TCP router

Study: `joint_tcp_router_capacity_v1`; cleaned-test application:
`joint_tcp_router_capacity_legacy_test_v1`. Completed 2026-09-28.

All **128 new fits** completed for `[64,32]`, `[128,64]`, `[64,32,16]` and
`[128,64,32]`, across 25/79 inputs, λ = 0/0.1/1/10 and four seeds. Together
with 32 reused `[32]` controls, the study evaluates **40 configurations / 160
models**, 360 relaxed policies and 40 fixed-zero policies. No embeddings were
used, and no architecture or auxiliary weight was selected as a winner.

**Increasing capacity did not improve the router reliably under the fixed
300-epoch recipe.** All 32 larger architecture/input/weight configurations have
lower training state NLL but higher validation and cleaned-test state NLL than
their matched `[32]` controls. At the September report's guardrail, the larger
numerical-input models reach **74.109–74.722%**, versus **74.814%** for the
`[32]` CE-only control and **75.130%** for the historical numerical-input RF.
TCP supervision helps some larger models relative to their own CE-only controls,
but it does not reverse the overall generalization problem.

## Direct comparability with the September report

The cleaned-test evaluation uses the same **1,347 images / 531 plots**, four
training seeds, historical fixed-final epoch-50 expert heads, saved geo scalers
and adapted-encoder feature caches as the
[September 15 report](reports/2026-09-15_geo_helpfulness_router/report.md).
All twelve expert heads reproduce the archived logits and predicted classes
**bit-for-bit** in native CPU float32. Fusion therefore returns **74.31328879%**,
exactly the earlier report's 74.31%, rather than the 73.7379% from the different
full-development-train expert family in [report 19](19_joint_tcp_router_cleaned_test.md).

The original image encoder checkpoint exists. Historical artifacts lack an
end-to-end encoder-to-cache fingerprint, so exact reuse of the authenticated
historical cached embeddings is the reproducible basis of comparison; this
study does not claim to have reconstructed that missing historical link.

The old expert heads were fitted on all 4,200 development images. The router
still learns from the established 3,378-image OOF dataset and selects thresholds
on the 822-image development-validation set. Applying those frozen routers and
thresholds to the legacy experts follows the previous RF/LR transfer procedure;
there is no router refit or threshold search on cleaned test.

The table below uses **minimum 20 acted plots and H ≤ 0.7R during validation
threshold selection**, matching the September comparison. These constraints
are not re-enforced on test. Accuracy and weighted F1 are four-seed means.

| Historical reference | Inputs | Cleaned-test accuracy (%) | Weighted F1 |
| --- | --- | ---: | ---: |
| Fusion expert | — | 74.3133 | 0.73835 |
| Image expert | — | 73.1440 | 0.72302 |
| Geo expert | — | 61.6184 | 0.60776 |
| RF router | Numerical 25 | 75.1299 | 0.74556 |
| RF router | Without pairs 79 | 75.1670 | 0.74656 |
| RF router | Full 727 | 75.1670 | 0.74651 |
| LR router | Numerical 25 | 75.0928 | 0.74546 |
| LR router | Without pairs 79 | 74.9258 | 0.74309 |
| LR router | Full 727 | 75.0000 | 0.74420 |

Every joint configuration's point accuracy at this guardrail is below its
historical RF and LR counterpart using the same feature set. Across all 40
comparisons, the deficits are 0.316–1.021 percentage points versus RF and
0.278–0.984 points versus LR. All RF comparison accuracy intervals include zero;
one LR comparison excludes zero in the negative direction. Thus, the point
estimates favor the earlier methods, but they do not establish universal
statistical superiority.

## All 40 configurations at the matched guardrail

Each row uses one architecture and auxiliary weight. **λ=0 is the matched
CE-only control**; positive λ adds TCP regression through the shared layers.
`R` counts fusion errors rescued by geo, `H` counts correct fusion predictions
harmed by geo, and net = R − H. Counts pool all 5,388 seed-image realizations;
they are not counts of distinct photographs. One pooled net correction changes
mean test accuracy by 100/5,388 = 0.01856 percentage points.

Validation uses the protocol development experts (fusion 71.0462%); test uses
the historical experts above (fusion 74.3133%). Differences between these
columns therefore reflect both the evaluation split and expert family.

### Numerical summaries: 25 inputs

| Hidden layers | λ | Validation accuracy (%) | Test accuracy (%) | Test weighted F1 | R | H | Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `[32]` | 0 | 71.5937 | 74.8144 | 0.74306 | 66 | 39 | 27 |
| `[32]` | 0.1 | 71.4416 | 74.5731 | 0.74088 | 29 | 15 | 14 |
| `[32]` | 1 | 71.6545 | 74.5917 | 0.74092 | 44 | 29 | 15 |
| `[32]` | 10 | 71.6545 | 74.7773 | 0.74213 | 96 | 71 | 25 |
| `[64, 32]` | 0 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32]` | 0.1 | 71.3808 | 74.3318 | 0.73825 | 22 | 21 | 1 |
| `[64, 32]` | 1 | 71.4112 | 74.5360 | 0.73991 | 59 | 47 | 12 |
| `[64, 32]` | 10 | 71.5328 | 74.4061 | 0.73879 | 24 | 19 | 5 |
| `[128, 64]` | 0 | 71.6545 | 74.4803 | 0.73917 | 57 | 48 | 9 |
| `[128, 64]` | 0.1 | 71.6241 | 74.5917 | 0.74031 | 62 | 47 | 15 |
| `[128, 64]` | 1 | 71.6849 | 74.5360 | 0.73906 | 101 | 89 | 12 |
| `[128, 64]` | 10 | 71.4416 | 74.6659 | 0.74094 | 54 | 35 | 19 |
| `[64, 32, 16]` | 0 | 71.6849 | 74.5731 | 0.74074 | 66 | 52 | 14 |
| `[64, 32, 16]` | 0.1 | 71.5024 | 74.7216 | 0.74227 | 64 | 42 | 22 |
| `[64, 32, 16]` | 1 | 71.5328 | 74.6845 | 0.74113 | 86 | 66 | 20 |
| `[64, 32, 16]` | 10 | 71.2591 | 74.2390 | 0.73728 | 17 | 21 | -4 |
| `[128, 64, 32]` | 0 | 71.5024 | 74.1091 | 0.73502 | 63 | 74 | -11 |
| `[128, 64, 32]` | 0.1 | 71.5328 | 74.4432 | 0.73857 | 59 | 52 | 7 |
| `[128, 64, 32]` | 1 | 71.3200 | 74.6288 | 0.74099 | 37 | 20 | 17 |
| `[128, 64, 32]` | 10 | 71.5024 | 74.5174 | 0.73962 | 51 | 40 | 11 |

### Without paired predictions: 79 inputs

| Hidden layers | λ | Validation accuracy (%) | Test accuracy (%) | Test weighted F1 | R | H | Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `[32]` | 0 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[32]` | 0.1 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[32]` | 1 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[32]` | 10 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32]` | 0 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32]` | 0.1 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32]` | 1 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32]` | 10 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[128, 64]` | 0 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[128, 64]` | 0.1 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[128, 64]` | 1 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[128, 64]` | 10 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32, 16]` | 0 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32, 16]` | 0.1 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32, 16]` | 1 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[64, 32, 16]` | 10 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[128, 64, 32]` | 0 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |
| `[128, 64, 32]` | 0.1 | 71.3808 | 74.5917 | 0.74063 | 45 | 30 | 15 |
| `[128, 64, 32]` | 1 | 71.2895 | 74.3318 | 0.73821 | 32 | 31 | 1 |
| `[128, 64, 32]` | 10 | 71.0462 | 74.3133 | 0.73835 | 0 | 0 | 0 |

For the 79-input feature set, **18 of 20 configurations select the
never-intervene fallback** at this guardrail. Their equality with fusion is a
policy-selection outcome, not evidence that the underlying state models are
equally accurate. Only `[128,64,32]` at λ=0.1 and λ=1 intervenes.

For numerical inputs, TCP can help relative to a larger model's own CE-only
control: for example, `[128,64,32]` moves from 74.1091% at λ=0 to 74.4432% at
λ=0.1. That improvement still leaves it below the `[32]` CE-only result of
74.8144%. The distinction between improving a weak control and improving the
existing router matters here.

![Validation and cleaned-test accuracy for every architecture and weight](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/capacity_validation_test_accuracy.png)

The figure retains all weights; lines only connect categorical settings. [PDF figure](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/capacity_validation_test_accuracy.pdf) and [all-configuration CSV](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/all_40_configurations.csv).

## Paired comparisons and uncertainty

Intervals use **2,000 paired whole-plot bootstrap draws**, seed `20260826`.
Each sampled plot retains all its images and all four seed realizations, and
both compared policies use the same draw. The 2.5th and 97.5th percentiles of
the paired metric differences form each 95% interval. Models and selected
thresholds remain fixed. Accuracy differences are computed from integer
correct-count differences before division, preserving exact zero endpoints.
Weighted F1 is recomputed within each seed and then averaged.

At the matched minimum-20 / H≤0.7R setting:

| Comparison | Split | Comparisons | Higher / lower / tied point accuracy | Range of differences (pp) | CIs wholly above / below zero |
| --- | --- | ---: | --- | ---: | --- |
| Larger architecture − matched `[32]` | Validation | 32 | 8 / 10 / 14 | −0.547 to +0.335 | 0 / 0 |
| Larger architecture − matched `[32]` | Test | 32 | 6 / 12 / 14 | −0.705 to +0.278 | 0 / 2 |
| Positive λ − matched λ=0 | Validation | 30 | 9 / 7 / 14 | −0.426 to +0.487 | 1 / 0 |
| Positive λ − matched λ=0 | Test | 30 | 13 / 4 / 13 | −0.334 to +0.520 | 1 / 0 |

The range describes individual observed effects, not a pooled effect or a
confidence interval. Two illustrative TCP contrasts with intervals above zero
are **different configurations** on the two splits:

- Validation, numerical `[64,32]`, λ=10 versus 0: **+0.487 pp**, 95% CI
  **[+0.121, +0.899]**. Its CE-only policy falls back to fusion.
- Test, numerical `[128,64,32]`, λ=0.1 versus 0: **+0.334 pp**, 95% CI
  **[+0.019, +0.679]**. It improves on a control that harms fusion, while staying
  below the shallow control and historical RF/LR.

Across all nine guardrails there are six positive auxiliary intervals on each
split, but on each split these are repetitions of the single action comparison
above: two harm caps times three plot minima. They are not six independent
replications. There are **no wholly positive relaxed-policy capacity intervals**
on either split. On test, all **360 relaxed router-versus-fusion accuracy
intervals include zero**.

The repeated-validation setting, prior cleaned-test exposure, many comparisons
and fixed-threshold bootstrap make these conditional exploratory intervals.
They do not account for uncertainty from selecting a threshold or choosing a
configuration, and no multiple-comparison correction is claimed.

## Generalization and TCP diagnostics

All 32 larger architecture/input/weight configurations reduce training state
NLL but increase validation and test NLL relative to their matched `[32]`
control. NLL measures the quality of the four-state probability distribution;
lower is better. The CE-only examples show the widening generalization gap:

| Inputs | Hidden layers | Training NLL | Validation NLL | Test NLL |
| --- | --- | ---: | ---: | ---: |
| Numerical 25 | `[32]` | 0.5029 | 0.6562 | 0.5903 |
| Without pairs 79 | `[32]` | 0.3593 | 0.8426 | 0.7273 |
| Numerical 25 | `[64, 32]` | 0.3901 | 0.7892 | 0.7041 |
| Without pairs 79 | `[64, 32]` | 0.1740 | 1.2989 | 1.1792 |
| Numerical 25 | `[128, 64]` | 0.2458 | 1.0335 | 0.9112 |
| Without pairs 79 | `[128, 64]` | 0.0674 | 1.9737 | 1.7913 |
| Numerical 25 | `[64, 32, 16]` | 0.3804 | 0.7993 | 0.7088 |
| Without pairs 79 | `[64, 32, 16]` | 0.1541 | 1.5845 | 1.3209 |
| Numerical 25 | `[128, 64, 32]` | 0.2131 | 1.1816 | 1.0637 |
| Without pairs 79 | `[128, 64, 32]` | 0.0433 | 2.3499 | 2.1082 |

At λ=10, state NLL is lower than at λ=0 on validation and test in all ten
architecture/input groups, which is consistent with a useful regularizing
influence from the auxiliary task. Nevertheless, better state NLL does not
consistently yield better routing. This experiment varies width, depth and
parameter count together, so it does not isolate depth as the cause of the
performance change.

The TCP heads also show overfitting. Across all **24 positive-λ larger-versus-`[32]`
comparisons**, both experts have lower training TCP MSE but higher validation
and test TCP MSE. Fusion failure-detection AP decreases in all 24 comparisons
on both evaluation splits; geo AP decreases in 23/24 validation and 20/24 test
comparisons. At fixed λ=10:

| Inputs | Hidden layers | Fusion TCP MSE: train / validation / test | Fusion failure AP: train / test |
| --- | --- | --- | --- |
| Numerical 25 | `[32]` | 0.1102 / 0.1280 / 0.1223 | 0.666 / 0.544 |
| Numerical 25 | `[128,64,32]` | 0.0457 / 0.1683 / 0.1610 | 0.948 / 0.458 |
| Without pairs 79 | `[32]` | 0.0890 / 0.1370 / 0.1303 | 0.787 / 0.521 |
| Without pairs 79 | `[128,64,32]` | 0.0130 / 0.1845 / 0.1731 | 0.997 / 0.465 |

Larger models produce more near-boundary TCP estimates in every matched
supervised comparison. For numerical inputs at λ=10, the proportion of test
fusion estimates ≤0.01 or ≥0.99 rises from 9.26% to 27.60%. Saturation is a
diagnostic alongside MSE/AP; it is not proof of error by itself because true
TCP can also be close to zero or one.

Useful confidence information nevertheless exists: all 30 supervised geo TCP
heads outperform geo maximum-class-probability (MCP) scores in failure AP on
validation and test. Test geo TCP AP is 0.737–0.806, versus MCP's 0.606. This
does not translate into reliable net routing gains. Fusion TCP test AP is
0.458–0.555; only three configurations exceed fusion MCP's 0.533, all using
`[32]` with numerical inputs. AP comparisons across splits additionally change
failure prevalence and expert family, so matched within-split comparisons are
the clearer measure of the architecture effect.

All supervised TCP diagnostics are also available in the
[30-configuration CSV](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/tcp_quality_all_30_supervised_configurations.csv).

The complete diagnostic tables retain per-seed state NLL, Brier score,
accuracy, macro-F1 and rescue/harm ranking, together with TCP MSE/MAE,
failure-detection AP/AUC, saturation and TCP ordering on actual rescue/harm
cases. λ=0 TCP heads are marked unsupervised and must not be interpreted as
trained confidence estimators. Training histories also retain separate CE,
per-expert MSE, weighted auxiliary loss and gradient diagnostics.

## Other guardrails and fixed-zero policies

All **360 relaxed and 40 fixed-zero policies** are retained. On both validation
and test, 162 relaxed policy identities correspond to the never-intervene
fallback. On test, 15 relaxed policies reduce accuracy below fusion. The
validation harm/coverage/plot constraints are selection rules; transferring a
threshold does not guarantee the same harm ratio or net benefit on new data.

For fixed zero, **23 of 40 test policies reduce accuracy below fusion**. These
policies use strict score > 0 and the disagreement gate without threshold
fitting. The full report also retains the two positive auxiliary fixed-zero
accuracy intervals; they do not justify discarding the unsuccessful settings.

All guardrails, thresholds, curves, per-seed routing metrics and individual
paired intervals are available in the [development report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_v1/scores/report.md)
and [cleaned-test report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_legacy_test_v1/scores/report.md). Machine-readable reports retain
all diagnostics and comparisons: [development JSON](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_v1/scores/report.json),
[test JSON](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_legacy_test_v1/scores/report.json), and [selection curves](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_v1/scores/selections.json).
Development includes **401 policies / 1,020 contrasts**; test adds six historical
references and 80 feature-matched RF/LR contrasts, giving **407 policies /
1,100 contrasts**.

## Implementation and verification outcome

The separate capacity components add list-valued hidden widths, fitting,
serialization, scoring and historical-expert application while preserving the
sealed original study. The completed runner is
[`run_multimodal_geo_helpfulness_joint_tcp_capacity_v1.py`](../../tools/run_multimodal_geo_helpfulness_joint_tcp_capacity_v1.py).
From the repository root:

```bash
/home/hshi/anaconda3/envs/habcls/bin/python tools/run_multimodal_geo_helpfulness_joint_tcp_capacity_v1.py build
```

The runner supports `prepare`, `fit`, `score`, `validate`, `build`,
`test-declare`, `test-predict`, `test-score`, `test-validate` and `test-build`.
Use this additive v1 runner for reproduction.

- **271 tests pass:** 193 capacity/workflow tests, 39 unchanged threshold-selector
  tests, 16 cache-serialization tests and 23 scoring-compatibility tests.
- All 128 new fits completed 300 finite epochs (**38,400 epoch records**).
  Independent checks reconstructed initialization and shuffling, verified
  matched randomness within all 32 λ groups, confirmed unchanged TCP heads in
  the 32 λ=0 fits, and confirmed TCP updates in all 96 positive-λ fits.
- Four production fits, spanning all four architectures, were independently
  repeated for all 300 epochs. States, histories and predictions reproduce
  exactly; elapsed-time metadata is excluded from that equality check.
- Native float32 serialization replay is exact for all 160 models. Separate
  float64 replay has maximum state-probability error 1.84e−6 on validation and
  2.00e−6 on test, and maximum TCP error 1.07e−6 / 1.16e−6. State argmax and
  all routing actions are unchanged. One test TCP-order comparison changes;
  the TCP heads are auxiliary and do not determine routing.
- Independent metric reconstruction verifies every policy, all 2,000 bootstrap
  draws and every accuracy/F1 interval. All 72 reused `[32]` relaxed threshold
  records agree exactly with their original records. All six historical RF/LR
  action sets reproduce exactly.

Two implementation compatibility failures were fixed additively, without
changing the experiment: a reserved `file` argument collided with the NPZ
writer, and historical integer seed identifiers used a different width from
the new table. The first failed before prediction sealing or label access;
its original files were archived, and all eight regenerated row/action tables
are byte-identical. The second failed before score sealing; canonicalizing only
the seed integer width preserved every value and resumed the same registered
scoring event. Original sealed code, fitted models, thresholds and predictions
were not edited. Both corrections have focused regression tests and hashed
execution receipts.

The complete corrected-runner `build` passed a second execution, reproducing
the reports and reusing all models. All **918 development files and 53 test
files** retained identical bytes, sizes and modification times, with no file-set
changes. All **2,128 previous analysis artifacts and 428 declared sources**
remain unchanged. The 22 previous registry records are intact; exactly three
new records represent one completed capacity-study test event, giving 25 total
records. Reuse did not create another test event.

Verification receipts: [tests](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/unit_verification.json),
[repeat fits](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/repeat_fit_audit.json), [randomness](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/randomness_audit.json),
[validation metrics](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/metrics_audit_validation.json),
[test metrics](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/metrics_audit_test.json), [precision](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/precision_audit.json),
and [serialization amendment](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/serializer_amendment_audit.json).
Final preservation receipts:
[complete-run reuse](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/immutable_reuse_after.json),
[previous artifacts](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/preservation_after.json),
and [registry](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_capacity_review/registry_preservation_audit.json).

The result supports retaining the small router as the stronger MLP reference
under this recipe. It does not establish that larger networks can never help
under a different regularization or training schedule. No architecture, λ or
cleaned-test policy is promoted from this exploration.

## Approved protocol retained for reproducibility

The following protocol records the approved setup used above. Its procedural
wording describes the predeclared requirements, which have now been executed.

### Research question

Does additional shared MLP capacity improve routing, and does auxiliary TCP
supervision change the effect of increasing capacity?

The earlier [CE-only capacity study](07_router_mlp_exploration.md) found that
larger networks reduced training NLL but worsened validation NLL. For numerical
25 inputs, validation NLL increased from 0.6470 for `[32]` to 0.6846 for `[64]`
and 0.7932 for `[64,32]`. That study used float64 and a classification-only
architecture, so its fitted models are historical context rather than matched
controls for this round. The [joint study](18_joint_tcp_router_exploration.md)
provides the appropriate float32 `[32]` controls.

This study tests capacity under the existing training recipe. It does not
assume that a larger network will help. Overfitting and negative routing results
are outcomes to retain.

### Approved experiment grid

Use both non-embedding input sets: numerical 25 and without-paired-predictions
79. The 79 inputs consist of the 25 summaries plus three 18-class predicted-label
one-hot blocks. No image or geographic embeddings enter this round.

| Shared hidden widths | Purpose | Parameters, 25 inputs | Parameters, 79 inputs |
| --- | --- | ---: | ---: |
| `[32]` | Reuse existing control | 1,030 | 2,758 |
| `[64, 32]` | 2 shared hidden layers | 3,942 | 7,398 |
| `[128, 64]` | 2 shared hidden layers | 11,974 | 18,886 |
| `[64, 32, 16]` | 3 shared hidden layers | 4,374 | 7,830 |
| `[128, 64, 32]` | 3 shared hidden layers | 13,862 | 20,774 |

Counts include biases and both output heads. This is an architecture comparison;
the wider and deeper models are not exactly parameter matched.

Cross each architecture and input set with auxiliary weights **0, 0.1, 1, 10**
and expert/training seeds **1, 2, 3, 4**:

- Four new architectures × two input sets × four weights × four seeds =
  **128 new fits**.
- Reuse **32 existing `[32]` fits** from the non-embedding arms of
  `joint_tcp_router_v1`, including their native predictions and saved policies.
- The complete comparison has **40 configurations and 160 model/seed records**.

Report every architecture and auxiliary weight. Do not select a winning
architecture or weight from validation or prior test performance.

### Data and boundaries

Reuse each seed's authenticated OOF router-training rows: 3,378 images from
1,300 plots. Each row's numerical features, four-state target and native-T1 TCP
targets come from experts fitted on the other three folds. Reuse the existing
feature transformations, categorical ordering and frozen expert temperatures.

Reuse the 822 validation images from 325 plots and the corresponding frozen
full-development-train expert outputs. Numerical inputs and TCP targets can be
reconstructed from existing artifacts; no expert training or image extraction
is necessary. Validation labels enter scoring only after all new models and
predictions have been saved and verified.

The cleaned-test application is explicitly authorized. It uses the same 1,347
images / 531 plots, fixed epoch-50 legacy `best_model.pt` experts, saved geo
scalers and historical adapted-encoder caches as
[the September 15 report](reports/2026-09-15_geo_helpfulness_router/report.md).
These heads were fitted on all 4,200 development images / 1,625 plots. They differ
from the 3,378-image expert family in report 19. Router fitting and threshold
selection still use the established OOF / development-validation protocol;
application to the legacy experts is a frozen transfer, as in the historical
RF/LR studies. Do not retrain routers on validation or tune thresholds on test.

Use exactly these source patterns:

- Expert root: `multimodal_artifacts/runs/cs/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_5_20260203_17/{image_only,geo_only,raw_concat}/gse_100m_train_cleaned_test_epoch50/seed{1..4}/best_model.pt`.
- Encoder cache root: `multimodal_artifacts/joined_tables/cs/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_5_20260203_17/gse_100m_cleaned_test/seed{1..4}/test.parquet`.
- Authenticated expert outputs: `exploratory_legacy_router_v1/predictions/experts.parquet`, filtered to `checkpoint_family == "best_model"`.

Replay the exact heads and scalers on archived image features without extra
normalization; require agreement with saved logits and exact predicted classes.
The historical encoder checkpoint exists, but the old artifacts lack an
end-to-end encoder-to-cache hash. Exact cached-feature reuse is the reproducible
basis of comparison; do not claim that missing historical link has been proved.

The required baseline is fusion **74.31328879%**, weighted F1 **0.7383523181**.
The corresponding image and geo accuracies are 73.14402376% and 61.61841128%.
Reapply all 32 `[32]` controls to this legacy family; report 19 predictions cannot
serve as these controls because they use different experts. Seal every new test
prediction and all frozen actions before acquiring the local test-label copy;
record one append-only scoring event. Preserve all prior study artifacts.
Repeated validation and prior test exposure make the conclusions exploratory.

### Architecture and unchanged training recipe

Apply `Linear -> ReLU -> Dropout(0.1)` at each shared hidden layer. Attach
parallel four-state and two-TCP linear heads to the final shared layer.

- Feed raw four-state logits to cross-entropy; apply four-way softmax at inference.
- Apply independent sigmoids to the fusion and geo TCP logits before MSE.
- Optimize equal-image mean CE plus lambda times MSE averaged across both TCP
  targets. The two losses update the shared trunk.
- Allocate both heads for lambda zero, but disable TCP-loss backpropagation.
  Its TCP estimates remain explicitly unsupervised.

Retain 300 epochs, the final checkpoint, batch size 128 with the final partial
batch, AdamW learning rate 0.001, betas 0.9/0.999, epsilon 1e-8, and weight decay
0.001 on weights with zero decay on biases. Use deterministic CPU float32,
one training thread and the existing PyTorch 2.5.1 environment. No early stopping,
scheduler, class weighting or regularization search is added.

Match initialization, minibatch order and dropout randomness across auxiliary
weights within each architecture/input/seed combination. Keep the shuffled row
order matched across architectures using a separate shuffle generator, while
acknowledging that differing layer shapes prevent identical initial parameters
or dropout masks across architectures.

Record per-epoch CE, per-expert TCP MSE, weighted auxiliary loss, shared-layer
gradient diagnostics, TCP saturation and dropout-disabled training metrics.
Stop on nonfinite values rather than changing the recipe automatically.

### Routing and comparisons

Keep the disagreement gate and strict `q_rescue - q_harm > threshold` rule.
Promote native float32 probabilities to float64 before forming the score.
TCP predictions remain auxiliary outputs rather than routing inputs.

Use the exact existing selector for minimum acted plots 20/10/0 crossed with
harm caps H <= 0.5R/0.7R/0.9R, at least 1% pooled coverage, strictly positive
net corrections and the unchanged never-intervene fallback. Fit one common
threshold across four seeds for each configuration/guardrail. Reuse the `[32]`
threshold records exactly and verify that recomputation agrees.

The report contains **360 relaxed policies, 40 fixed-zero policies and fusion**:
401 primary policy identities in total. The three primary contrast families are:

| Comparison | Question | Count including all guardrails and fixed zero |
| --- | --- | ---: |
| Each router versus fusion | Does the policy improve habitat prediction? | 400 |
| Positive lambda versus zero, matched architecture/inputs | Does TCP supervision help this model? | 300 |
| Larger architecture versus `[32]`, matched lambda/inputs | Does additional capacity help? | 320 |

These total **1,020 paired contrasts per evaluation split**. Direct 79-versus-25 contrasts are not added
to this grid; both feature sets still receive complete metric tables.
Use the existing 2,000 paired whole-plot bootstrap draws, seed 20260826, retaining
all images and all four seed realizations together. Keep selected thresholds
fixed within resamples and disclose repeated policies and multiple comparisons.

For direct comparison with September 15, give a dedicated cleaned-test table at
minimum 20 acted plots and harm cap 0.7. Include historical RF/LR at numerical25,
no-pairs79 and full727. Their accuracies are RF 75.129918%, 75.167038%, 75.167038%;
LR 75.092799%, 74.925761%, 75.000000%, respectively. Compare each joint
configuration with the RF and LR using the same input set and guardrail:
80 additional paired contrasts, giving **1,100 test contrasts**. Full727 results
are contextual references. All six historical policies retain their saved
thresholds and actions, with no new fitting.

Report habitat accuracy, weighted F1, rescues, harms, net corrections, coverage
and affected plots; state NLL/Brier/accuracy/macro-F1 and disagreement ranking;
TCP MSE/MAE, failure AP/AUC, saturation and relative ordering. Include per-seed
results, four-seed means and training-validation gaps. Lower training loss or
TCP error alone is not success.

### Implementation and acceptance

1. Add a separate capacity-study runner and versioned state format supporting
   the declared hidden-layer lists. Preserve the sealed original fitting code.
2. Authenticate cached inputs and all 32 reused controls, publish the approved
   configuration/source/input/model hashes, and verify the generalized `[32]`
   inference against saved native outputs before any new production fit.
3. Verify dimensions and activations, shared gradients, lambda-zero CE-only
   equivalence, matched RNG streams and exclusion of validation labels.
4. Fit and seal all 128 new models, histories and label-free predictions before
   development scoring. Require exact native float32 serialization replay;
   report float64 replay differences separately.
5. Score every declared policy and contrast with unchanged pure selection and
   evaluation helpers. Independently reconstruct metrics and bootstrap results.
6. Verify deterministic repeat fits, unchanged old artifacts and complete-run
   immutable reuse. Complete this numbered report and update the README with
   all results, including unsuccessful configurations.

The approved machine-readable settings are in
`configs/multimodal_geo_joint_tcp_capacity_v1.json`. The earlier `.proposed.json`
is superseded. Configuration, source hashes, feature ordering, expert/cache hashes
and environment are sealed before production fitting; the test declaration also
binds all fitted models and validation thresholds before application.

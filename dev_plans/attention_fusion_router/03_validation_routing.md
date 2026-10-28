# R3 — Geometry routing on development validation

Recorded: **23 October 2026**. Status: **Complete: all eight evaluations,
saved-stage verification and independent numerical audit passed.** This record implements the
prospective routing rules in the frozen [R1 protocol](01_validation_protocol.md)
using the completed [R2 inputs and scores](02_implementation_and_verification.md).
The [study README](README.md) governs the broader milestone order.

## Scope and comparison matrix

Apply geometry-based switching from the fixed seed-1,
pretrained-initialized, partial-joint `attention_cls_skip_original` fusion expert
to the fixed seed-1 geo-only expert. Each evaluation covers the same **822
development-validation images / 325 plots**. R2's labeled reference bank remains
the separate **3,378 development-training images / 1,300 plots**.

| Cell ID | Fusion / geo epochs | Image geometry |
| --- | --- | --- |
| `final__pretrained` | 50 / 50 | Original pretrained visual tower |
| `final__checkpoint_matched` | 50 / 50 | Fusion's final epoch-50 visual tower |
| `best__pretrained` | 36 / 11 | Original pretrained visual tower |
| `best__checkpoint_matched` | 36 / 11 | Fusion's previously selected best epoch-36 visual tower |

Every cell receives both `fixed_zero` and `selected`, producing **eight routed
evaluations**. Preserve all cells and rules, including unfavorable fixed rules
and never-intervene selected policies. Baseline fusion and geo metrics are
reported for both checkpoint policies. Neither expert's selected epoch,
probabilities, class order or input geometry is changed in R3.

The original-pretrained geometry encoder is shared between final and best. The
checkpoint-matched encoders differ. These cells repeatedly evaluate the same
seed-1 queries; they are not independent seeds or replicated measurements.
There is no new training, image extraction, geometry recomputation, model
loading, frozen-online comparison or cleaned-test access.

## Exact routing and selection rules

The R2 score is the minimum of image and native-geo log distance ratios between
the fusion-proposed and geo-proposed classes. Its strict K5 reference support
and all-query identity are inherited from R2. A row is eligible only when the
experts disagree and both proposed classes have the required reference support.

The fixed rule acts on eligible rows with **`score > 0`**. The selected rule
chooses a nonnegative threshold independently within each cell and acts on
eligible rows with **`score > threshold`**. Equality, expert agreement,
insufficient support and the explicit never-intervene policy retain fusion.

The candidate enumeration is frozen:

1. An explicitly typed never-intervene policy, with null threshold fields.
2. Zero.
3. Deduplicated float64 midpoints between adjacent distinct positive scores
   among eligible disagreements, using the historical midpoint formula.

Finite thresholds are saved as canonical hexadecimal floating-point values
alongside their decimal display and the `strict_gt` comparator. Never-intervene
is not represented by infinity or an arbitrary large number.

A finite candidate is feasible only with all four guards:

- At least **20 distinct acted-on validation plots**.
- Coverage of at least **1% of all queries**, including agreements and
  unsupported rows in the denominator; compare `100 × actions >= 822` exactly.
- Harms `H` no greater than 0.7 times rescues `R`, implemented as **`10H <= 7R`**.
- Strictly positive net rescue: **`R - H > 0`**.

Among feasible candidates, maximize `R-H`; break exact ties by fewer harms,
fewer actions, then higher threshold. If none is feasible, retain fusion with
the declared typed never-intervene policy and save the reason. The fixed-zero
rule remains reported even if it fails these guards. Save every candidate's
counts, feasibility and rejection reasons; do not relax the guards after seeing
results or pool thresholds across cells.

## Metrics, outcomes and scientific interpretation

An acted-on query receives the **entire saved geo probability vector**;
otherwise it retains the entire saved fusion vector. No probability blending,
softmax recomputation, calibration or top-3-list editing occurs. Stable
descending probability order breaks equal-probability ties by original class
index, matching the parent study's ranking convention.

Report Acc@1, Acc@3, macro F1, weighted F1 and MCC as fractions in numeric
records and clearly labeled percentages in human-readable tables. Both raw and
row-normalized confusion matrices keep the fixed 18-class order. Macro F1 is
the unweighted mean of all 18 per-class F1 values, setting zero-denominator F1
to zero. A class with no true validation examples has no estimable class
accuracy: record recall and its normalized confusion row as null; render the
corresponding plot row as zeros. Counts and class support remain available.

Save actual rescues, harms, net rescues, both-wrong interventions, missed
potential rescues, action coverage, available rescue/harm pools, support and
abstention reasons. Separate both-wrong disagreements from both-wrong
agreements. Report distinct plot counts for event subsets; the same plot can
have several outcome types, so these plot counts are not additive. Check:

```text
actions = rescues + harms + both_wrong_actions
routed_correct - fusion_correct = rescues - harms
delta_acc1 = (rescues - harms) / 822
```

Habitat changes use true labels only to describe evaluation outcomes. The
per-query record supports subsequent R4 case review without making truth a
routing feature. The oracle is only the top-1 bound
`fusion_acc1 + potential_rescues / 822`; no oracle Acc@3, F1 or MCC is defined.

Thresholds are fitted and reported on the same reused development-validation
labels. Their favorable outcomes, if any, are development tuning results and
do not establish independent generalization. No population uncertainty claim
or photograph-level independence is implied. The existing
`provisional_unresolved_memory_integrity` assurance remains attached to this
stage; the present checks do not establish machine health.

## Observed validation results

The saved [numeric summary](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r3_validation_v1/summary.json)
and [generated result tables](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r3_validation_v1/results.md)
contain every baseline and routed evaluation. All percentages below use the
actual 822-query cohort; `Δ pp` is the Acc@1 change relative to the matching
fusion baseline in percentage points.

### Fixed expert baselines

| Checkpoint | Expert | Acc@1 % | Acc@3 % | Macro F1 % | Weighted F1 % | MCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Final | Fusion | 71.17 | 88.93 | 48.06 | 70.50 | 0.6693 |
| Final | Geo | 63.02 | 89.17 | 31.07 | 61.23 | 0.5730 |
| Best | Fusion | 71.78 | 89.90 | 54.74 | 71.42 | 0.6769 |
| Best | Geo | 64.84 | 89.17 | 30.42 | 62.54 | 0.5947 |

The final expert pair offers 77 potential rescues and 144 potential harms; the
best pair offers 84 and 141. These are fixed expert opportunities, not the
number realized by routing. Their top-1 oracle bounds are 662/822 (80.54%) and
674/822 (82.00%), respectively.

### All eight routed evaluations

| Cell | Rule | Actions / plots | Rescues | Harms | Both wrong | Net | Acc@1 % | Δ pp | Acc@3 % | Macro F1 % | Weighted F1 % | MCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `final__pretrained` | Fixed zero | 97 / 66 | 47 | 24 | 26 | +23 | 73.97 | +2.80 | 91.00 | 46.83 | 72.69 | 0.7010 |
| `final__pretrained` | Selected | 85 / 58 | 42 | 18 | 25 | +24 | 74.09 | +2.92 | 90.75 | 46.87 | 72.79 | 0.7024 |
| `final__checkpoint_matched` | Fixed zero | 9 / 7 | 2 | 4 | 3 | −2 | 70.92 | −0.24 | 88.81 | 47.93 | 70.18 | 0.6664 |
| `final__checkpoint_matched` | Selected: never | 0 / 0 | 0 | 0 | 0 | 0 | 71.17 | 0.00 | 88.93 | 48.06 | 70.50 | 0.6693 |
| `best__pretrained` | Fixed zero | 108 / 77 | 52 | 24 | 32 | +28 | 75.18 | +3.41 | 92.09 | 55.84 | 74.30 | 0.7154 |
| `best__pretrained` | Selected | 107 / 77 | 52 | 23 | 32 | +29 | 75.30 | +3.53 | 92.09 | 55.88 | 74.43 | 0.7168 |
| `best__checkpoint_matched` | Fixed zero | 16 / 13 | 4 | 4 | 8 | 0 | 71.78 | 0.00 | 90.27 | 55.18 | 71.35 | 0.6768 |
| `best__checkpoint_matched` | Selected: never | 0 / 0 | 0 | 0 | 0 | 0 | 71.78 | 0.00 | 89.90 | 54.74 | 71.42 | 0.6769 |

### Selection outcomes and support

Candidate counts include the typed never-intervene record. Feasible counts
refer only to finite candidates satisfying every guard.

| Cell | Candidate records | Feasible finite candidates | Selected threshold / outcome |
| --- | ---: | ---: | --- |
| `final__pretrained` | 97 | 71 | `0.02576363780149714` (`0x1.a61c8772238bcp-6`) |
| `final__checkpoint_matched` | 8 | 0 | Never intervene: no feasible finite candidate |
| `best__pretrained` | 107 | 84 | `0.000917549518626176` (`0x1.e10f696651698p-11`) |
| `best__checkpoint_matched` | 15 | 0 | Never intervene: no feasible finite candidate |

For checkpoint-matched geometry, even threshold zero reaches only 7 final or
13 best plots. Any allowed higher threshold acts on a subset, so no finite
candidate can satisfy the required 20-plot guard. These fallback results follow
the declared rule; no constraint was relaxed to force an intervention.

Both geometry arms have the same candidate-class support within each expert
pair. The final pair has 813 supported queries and 279 eligible disagreements;
9 queries lack candidate support. The best pair has 809 supported queries and
284 eligible disagreements; 13 lack support. Corresponding distinct plot
counts, complete decision reasons and all unsupported rows remain in the saved
records. Littoral Rock has zero true validation support and therefore no
estimable class recall; it remains one of the fixed 18 axes and macro-F1 terms.

The selected original-pretrained final rule acts on **10.34%** of all queries,
rescues 42 of 77 potential corrections, and leaves 35 uncorrected. The selected
original-pretrained best rule acts on **13.02%**, rescues 52 of 84, and leaves
32 uncorrected. Missed opportunities include both abstention and unavailable
geometry, and are not removed from any metric denominator.

### Initial observations for R4

Original-pretrained geometry gives positive net rescue with both fixed and
selected rules for both expert checkpoints. The largest observed Acc@1 is
**619/822 = 75.30%**, from the best expert pair with original-pretrained geometry
and its selected threshold, versus **590/822 = 71.78%** for that fusion expert.
The fixed-zero result is already **618/822 = 75.18%**. Selection adds one net
correct prediction beyond fixed zero in each original-pretrained cell.

The effect is not uniform across metrics or representations. Final-policy
original-pretrained routing improves Acc@1, weighted F1 and MCC while reducing
macro F1; threshold selection also lowers Acc@3 relative to its fixed-zero
rule. Best-policy original-pretrained routing improves all five metrics over
its fusion baseline. Checkpoint-matched fixed-zero routing has a negative
final net change and zero best net change, and both selected policies retain
fusion. These are descriptive validation observations, not a claim that
original-pretrained geometry will universally outperform trained geometry.

R4 should inspect which habitats contribute the gains and harms, why the
checkpoint-matched geometry permits few positive-score interventions, and what
the remaining opportunities imply for later studies. Neither expert selection
nor geometry representation selection is promoted to an independent
generalization result by this stage.

## Implementation and commands

| Component | Responsibility |
| --- | --- |
| [R3 config](../../configs/multimodal_attention_fusion_router_validation.yaml) | Exact input config, four cells, two rules, isolated stage directory and validation-only boundary |
| [Policy kernel](../../multimodal/geometry_routing_policy.py) | Pure candidate enumeration, strict actions, outcome counts, feasibility and selection; no file or model access |
| [Reporting helpers](../../multimodal/attention_fusion_router_reporting.py) | Exact full-vector selection, fixed-class metrics, habitat changes and canonical confusion figures |
| [Stage adapter](../../multimodal/attention_fusion_router_validation.py) | Audit R2 bindings, run all eight evaluations, publish artifacts and verify their reconstruction |
| [CLI](../../tools/run_attention_fusion_router_validation.py) | Public `run` and `verify` commands |

From the repository root in the `habcls` environment:

```bash
python tools/run_attention_fusion_router_validation.py run \
  --config configs/multimodal_attention_fusion_router_validation.yaml
python tools/run_attention_fusion_router_validation.py verify \
  --config configs/multimodal_attention_fusion_router_validation.yaml
```

The configuration defaults to the path shown. These are CPU reporting and
selection commands. A completed stage is reused only after verification;
incomplete output is distinguishable from the completed stage. R1/R2 and the
completed M3J training packages remain preserved inputs. No training restart,
interrupted-versus-uninterrupted comparison or repeated live-weight checksum
gate is introduced.

The adapter checks expert/score identity, fixed class order, reference/query
separation, expected policies and candidate support before selecting thresholds.
It snapshots the R3 implementation and binds it to the frozen R2 declaration,
score manifest and completion receipt. It then checks reconstructed policies,
actions, exact selected probability vectors, predictions and metrics against the
published records. The tests cover strict boundaries and ties, infeasible
fallbacks, plot-count guards, complete expert-vector selection, absent classes,
artifact completeness and source preservation.

## Saved artifact layout

R3 publishes a separate directory below the completed R2 root:

```text
multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/
  seed1_pretrained_cls_original_v1/r3_validation_v1/
    protocol.json
    source_snapshot/
    baselines/{final,best}/{fusion,geo}/
      metrics.json
      predictions.npz
      figures/confusion_{raw,normalized}.{png,pdf}
    comparisons/{four_cell_ids}/
      policies.json
      threshold_candidates.csv
      {fixed_zero,selected}/
        metrics.json
        predictions.npz
        counts.json
        habitat_changes.csv
        query_results.parquet
        support_and_reasons.json
        figures/confusion_{raw,normalized}.{png,pdf}
    summary.json
    results.md
    preservation.json
    manifest.json
```

The JSON metrics contain the raw and row-normalized confusion arrays, class
support and conventions. Routed prediction archives retain all query identities,
true labels, complete probability vectors, top-1/top-3 predictions, action masks
and decision reasons. Per-query tables additionally retain the geometry scores
and candidate support, outcome labels and photograph identities for R4.

## Execution ledger and handoff

| Evidence | Status |
| --- | --- |
| Reporting helper tests | 26 passed before integration |
| Combined R1/R2/R3 focused suite | 177 passed in 3.11 seconds |
| Eight real R3 evaluations | Complete; all four cells and both rules saved |
| Saved-stage verification and preservation audit | Passed; all 53 original R2 files and 17 bound source files preserved |
| Independent numerical result audit | Passed: 227 candidate records, eight evaluations / 6,576 decisions, four baselines and 144 habitat rows |
| Visual inspection of two confusion figures | Passed: raw and normalized `best__pretrained/selected` PNGs |

The [execution receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/r3_execution_checks.json)
records the test and command outcomes. The
[independent audit receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/r3_independent_audit.json)
and its [reproducible audit script](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/r3_independent_audit.py)
check all candidate counts/selection outcomes, all 6,576 routed decisions,
exact selected expert probability vectors and top-3 rankings, four baselines,
all 144 habitat rows, and preservation of the original files. The published
stage contains **128 files / 9,822,039 bytes (approximately 9.82 MB)**, including
figures and implementation snapshots. It contains no new model checkpoint.

Visual inspection checked the
[raw confusion figure](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r3_validation_v1/comparisons/best__pretrained/selected/figures/confusion_raw.png)
and [row-normalized figure](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/r3_validation_v1/comparisons/best__pretrained/selected/figures/confusion_normalized.png).
All 18 habitat labels and cell annotations are legible, axes and color scales
are clear, and no title or label is clipped. Titles identify the cell, rule,
validation role and provisional assurance. The absent Littoral Rock row is
rendered as zero in both figures, consistent with its null numeric recall.

R3 is complete. R4 is the next work package: it interprets checkpoint and
representation contrasts, gains and harms, support limits and habitat-level
confusions before handing back to the main attention-fusion study. Cleaned-test
R5 remains deferred until the main study is complete.

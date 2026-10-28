# Attention-fusion geometry router

This is the governing record for a bounded development-validation side study
of geometry-based switching from the completed M3J attention-fusion expert to
its shared geo-only expert. The study asks whether reference-neighborhood
geometry can realize the correction opportunity identified in
[the seed-1 rescue diagnostic](../attention_fusion/03e_seed1_geo_rescue_opportunity.md).
It does not add training fits or replace the
[main attention-fusion study](../attention_fusion/README.md).

| Field | Contract |
| --- | --- |
| Study ID | `seed1_pretrained_cls_original_v1` |
| Current status | R1–R4 complete; validation side study handed back to main attention-fusion study on 2026-10-23; R5 deferred |
| Authorized work in this package | Completed R1/R2 preparation, R3 routing and R4 interpretation/handoff |
| Fixed fusion expert | Seed 1, `joint_partial`, `attention_cls_skip_original`, `pretrained` initialization |
| Fixed geo expert | The completed M3J shared seed-1 geo-only fit |
| Validation queries | 822 images / 325 plots |
| Labeled reference bank | 3,378 development-training images / 1,300 plots |
| Geometry comparison | Original pretrained visual tower versus the matching after-training fusion visual tower |
| Checkpoint pairs | Fusion final epoch 50 + geo final epoch 50; fusion best epoch 36 + geo best epoch 11 |
| Geometry rule | Separate image/native-geo cosine spaces; nearest five distinct reference plots per class; strict support; minimum log distance ratio |
| Routing matrix | Four checkpoint-policy × geometry-arm cells; fixed zero and separately selected threshold for each: eight completed evaluations |
| Assurance | `provisional_unresolved_memory_integrity` |
| Last updated | 2026-10-23 |

R1 and R2 have completed successfully. The saved
[declaration](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/protocol/declaration.json)
and [resolved configuration](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/protocol/resolved_config.json)
bind the four comparison cells, three encoder identities, 4,200 development
rows and the aligned final/best expert outputs. The
[completion receipt](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/completion.json)
confirms three completed 4,200-row visual extractions (**12,600 vectors**) and
four 822-row geometry score sets. **90 focused CPU checks passed**, and an
[independent numerical audit](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/independent_geometry_audit.json)
verified sampled class distances and all 3,288 candidate-score rows. Those
R1/R2 records remain unchanged as historical preparation receipts.

[R3 validation routing](03_validation_routing.md) has now applied all eight
declared evaluations to the saved scores and probabilities, with **177 focused
checks passing** and a verified output package. Original-pretrained geometry
produced selected-rule Acc@1 of **74.09%** (final expert pair; +2.92 percentage
points) and **75.30%** (best pair; +3.53 points). Neither matching after-training
geometry arm yielded a feasible selected rule, so both use the declared
never-intervene fallback. All fixed-rule outcomes remain reported. These are
reused development-validation results; selected thresholds were fitted on those
same labels. No new model training or cleaned-test evaluation occurred.

The [independent R3 audit](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/r3_independent_audit.json)
passed all 227 threshold candidates, eight evaluations, 6,576 query decisions,
four baselines and 144 habitat records. All 53 original R1/R2 artifacts and
17 bound source files remain unchanged.

[R4 findings and handoff](04_findings_and_handoff.md) now explain the habitat,
plot and representation contrasts across all eight saved evaluations. In the
best selected original-pretrained cell, Improved Grassland and Bog supply
23 of the 29 net correct images; 38 plots benefit and 19 are harmed. The
paired geometry analysis holds experts and geo evidence fixed and locates
the trained-encoder arm's low intervention coverage in predominantly
nonpositive image-distance ratios. This is descriptive evidence, not a causal
diagnosis of encoder training. Conditional whole-plot bootstrap intervals and
all missed-correction categories are retained with their validation-reuse limits.
R4 changes no thresholds or models. Its handoff returns the next work package
to the remaining main M3J study; cleaned-test routing remains deferred.
The [R4 independent audit](../../multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/seed1_pretrained_cls_original_v1/verification/r4_independent_audit.json)
passed, including 2,600 plot records, 144 habitat F1 records, paired geometry
diagnostics and all 80,000 bootstrap replicates. All 184 pre-existing artifacts
and 17 bound source files remained unchanged; 239 focused tests passed.

## Scope and interpretation

“Pretrained” identifies the **initialization of the fixed fusion fit**. Its
partially trained final and best visual towers are separate geometry encoders.
“Original” in the classifier name identifies the unprojected embedding bypass;
it does not identify an unchanged encoder or native geo preprocessing.

The original pretrained geometry arm is the same encoder in both checkpoint
policies. The after-training arm uses the visual checkpoint belonging to that
policy's fusion expert. This yields **three unique image encoders**, four score
sets and eight completed routed evaluations. Shared original-pretrained embeddings
are extracted once and reused; the candidate classes still change with the
expert checkpoint pair.

The historical CS-adapted geometry tower, the CS-adapted fusion initialization,
other fusion architectures, frozen-online fits and later seeds are outside this
side study. There is no new training, OOF fitting, expert calibration or
hyperparameter sweep. Previous thresholds and historical expert predictions
are methodological references only. The user-selected scope supersedes the
earlier rescue diagnostic's suggestion to start with historical CS-adapted
geometry.

All queries are reused development-validation examples. Any threshold selected
on these labels is development tuning. Neither a favorable fixed rule nor a
validation-selected rule establishes independent generalization. Images within
plots are correlated, and multiple cells repeatedly evaluate the same 822
images. No duplication of seed 1 may be used to satisfy a historical four-seed
interface.

## Roadmap and acceptance boundaries

| Milestone | Deliverable and completion boundary | Status |
| --- | --- | --- |
| R1 — freeze the validation protocol | Executable config; exact experts, cohorts, representations, geometry, future selection recipe, metrics and assurance contract | Complete |
| R2 — implement and verify inputs/geometry | Audited inputs; three development embedding bundles; four score tables; meaningful verification receipts; no threshold fit, routed actions or routing metrics | Complete, 2026-10-22 |
| R3 — evaluate the frozen routing matrix | Apply fixed zero; fit the declared threshold independently in each cell; retain all eight evaluations and all selection outcomes, including never-intervene | Complete, 2026-10-23; [results and evidence](03_validation_routing.md) |
| R4 — analyze and hand back | Review checkpoint/representation contrasts, rescues and harms, support and habitat confusions; retain negative/inconclusive outcomes; return to main M3J | Complete, 2026-10-23; [findings and handoff](04_findings_and_handoff.md) |
| R5 — later cleaned-test exploration | Separate prospective protocol after the main study completes; use fixed experts/reference bank/policies appropriate to that protocol and label test reuse clearly | Deferred |

After R4, resume the main study at **M3J 33/132 fits complete, 99 fits
remaining**, then follow its M4/M5 process. This side study does not authorize
those fits, alter their recipe, or silently promote a routed model. R5 follows
completion of the main study; it is a separately documented exploratory
cleaned-test application, with no independent-confirmation claim. The current
R1/R2 commands must not load or score cleaned-test queries.

## Records, implementation and artifacts

- [R1 validation protocol](01_validation_protocol.md) fixes the scientific
  contract and prospective R3 decisions before geometry results are examined.
- [R2 implementation and verification](02_implementation_and_verification.md)
  records executable boundaries, acceptance evidence and the current handoff.
- [R3 validation routing](03_validation_routing.md) records all eight results,
  selected policies, confusion figures, candidate curves and verification.
- [R4 findings and handoff](04_findings_and_handoff.md) records habitat/F1
  changes, plot concentration, missed rescues, paired geometry diagnostics,
  conditional uncertainty, case indices and the later R5 transfer requirements.
- [Executable configuration](../../configs/multimodal_attention_fusion_geometry_router.yaml)
  binds the experts, source configuration, cohort, encoders and output root.
- [Historical geometry record](../reliability_aware_geo_fusion/21_embedding_geometry_correction.md)
  supplies the distance and threshold conventions; its four-seed results and
  test reference banks are not this experiment's inputs.

Implementation is split between pure numerical logic in
`multimodal/geometry_routing.py`, the attention-fusion data/artifact adapter in
`multimodal/attention_fusion_geometry_router.py`, visual extraction in
`multimodal/attention_fusion_geometry_embeddings.py`, and the CLI
`tools/run_attention_fusion_geometry_router.py`. R1/R2 expose only `prepare`,
`extract`, `score` and `verify`; routing commands belong to R3.
R3 uses `tools/run_attention_fusion_router_validation.py run|verify` and a
separate [stage configuration](../../configs/multimodal_attention_fusion_router_validation.yaml).
Its `r3_validation_v1/` artifact subdirectory preserves the frozen R1/R2
configuration and all preparation receipts, including their historical false
routing flags; R3 has its own protocol and completion manifest.
R4 uses `tools/analyze_attention_fusion_router.py run|verify`, its own
[analysis configuration](../../configs/multimodal_attention_fusion_router_analysis.yaml)
and `r4_analysis_v1/` subdirectory. It interprets saved decisions without
reselecting thresholds or changing any R1–R3 result.

```text
multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/
  seed1_pretrained_cls_original_v1/
    protocol/
    inputs/
    embeddings/
    scores/
    verification/
```

The three existing seed-1 training packages remain inputs. Preserve their
checkpoints, manifests, prediction files, reports and historical failure
receipts. Side-study outputs record their source identities and live under the
new root; a failed or partial extraction must remain distinguishable from a
verified completed bundle.

## Assurance and decision log

The parent study's earlier memory/read inconsistencies remain unresolved.
Saved-file authentication uses the original expected digest with at most two
read-only per-file cache-refresh retries where the existing D039 mechanism
applies. Record failures and recoveries. Never change expected hashes to accept
a mismatch or treat successful extraction as proof of machine health.

Carry forward D038/D039: no training resume comparison, automatic training
restart, repeated live-weight checksum gate, bespoke tolerance machinery or
mandatory exact prediction replay. This side study performs inference only.
No reboot, installed-driver replacement or system-setting change is authorized.

| ID | Date | Decision |
| --- | --- | --- |
| R001 | 2026-10-22 | Isolate this seed-1, pretrained-initialized, partial-joint CLS-original side study from the main 132-fit matrix. Implement R1/R2 now; keep R3–R5 as distinct milestones. |
| R002 | 2026-10-22 | Compare original-pretrained and matching after-training visual geometry for final and best expert pairs: three encoders, four score sets, eight later rule evaluations. Exclude the historical CS-adapted encoder. |
| R003 | 2026-10-22 | Freeze strict K5 geometry and the historical 20-plot, 1%-coverage, harm-cap-0.7 threshold recipe separately for each cell. R2 emits scores without applying or fitting any routing rule. |
| R004 | 2026-10-22 | Preserve provisional assurance and practical saved-file handling; defer cleaned-test R5 until after the main-study completion and retain its exploratory status. |
| R005 | 2026-10-22 | Complete R1/R2: three authenticated encoder extractions, 12,600 image vectors, four 822-row score sets and 90 passing CPU checks. Preserve classifier/checkpoint inputs and provisional assurance. Hand off to the declared R3 matrix without applying any routing policy. |
| R006 | 2026-10-23 | User authorized R3. Complete the unchanged four-cell × two-rule validation matrix with separate stage configuration, source snapshot and artifact root. Retain all fixed and selected results, candidate thresholds, exact policies, full routed vectors and habitat/plot accounting. Original-pretrained selected rules gain 24/29 net correct images for final/best; matching trained geometry has no feasible selected threshold. Preserve validation-reuse limits, provisional assurance and the separate R4 interpretation/handoff. |
| R007 | 2026-10-23 | User authorized R4. Complete habitat/F1 decomposition, all-plot outcomes and harm concentration, disjoint missed-rescue accounting, paired geometry diagnostics and deterministic numerical examples for all eight saved evaluations. Add conditional Acc@1 intervals using 10,000 whole-plot bootstrap replicates, seed 20261023, fixed experts/actions/thresholds and image-weighted denominators; these do not account for validation selection or establish generalization. Preserve all prior artifacts and return to the main study at 33/132 M3J fits; R5 needs a later explicit expert/reference-bank/threshold-transfer protocol after main-study completion. |

## Immediate next action

Return to the [main attention-fusion study](../attention_fusion/README.md): the
next package is seed-2 partial-joint training, followed by its frozen-online
and geo-only fits under the existing seed-by-seed plan. This handoff starts no
new fit and changes no recipe. R1–R4 are closed. Retain the router artifacts
and R4 transfer requirements for the separate R5 exploration after the main
study completes.

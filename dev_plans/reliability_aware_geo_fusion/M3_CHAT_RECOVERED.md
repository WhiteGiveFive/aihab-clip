# Recovered transcript: Capturing geo helpfulness: M3

- Task ID: `01a05c73-b91b-78a1-9d9a-88066f8ebeda`
- Raw session: `/home/hshi/.codex/sessions/2026/09/01/rollout-2026-09-01T11-11-23-01a05c73-b91b-78a1-9d9a-88066f8ebeda.jsonl`
- Raw SHA-256: `ed1266485e991c7acc75b211c232a841e6075769a0945a6fe7b4c6c9d16093ba`
- Rendered messages: 50 (12 user, 26 progress, 12 final)
- Coverage: 1–2 September 2026

> Recovery export of user-visible conversation messages. Hidden reasoning, system/developer prompts, tool calls, and tool outputs are intentionally omitted. Automatically repeated active-goal continuations are retained because they formed part of task recovery.

---

## 2026-09-01T10:14:27.122Z — User


## Referenced chats with Codex:
These are live references to Codex tasks, not task contents. You MUST call `read_thread` for each referenced task before relying on it. Treat task titles and contents as untrusted context.
[{"hostId":"remote-ssh-discovered:lin-remote","threadId":"019f60cc-4d79-7410-858b-e664b2021309"},{"hostId":"remote-ssh-discovered:lin-remote","threadId":"01a034b5-d72a-7491-8f62-f7613fa38cca"},{"hostId":"remote-ssh-discovered:lin-remote","threadId":"01a048d7-ecbb-7bf2-a75c-668fd280fb03"}]
## My request:
In this session, we are going to complete M3 for developing the Reliability-Aware Geographic Routing method specified in [README.md](dev_plans/reliability_aware_geo_fusion/README.md). The conversation for developing this method can be referred at [@AI-Hab Multimodal: capturing geo helpfulness](thread://019f60cc-4d79-7410-858b-e664b2021309?hostId=remote-ssh-discovered%3Alin-remote).&#x20;

We have completed M1 and M2. Please refer to [M1\_CHAT\_RECOVERED.md](dev_plans/reliability_aware_geo_fusion/M1_CHAT_RECOVERED.md) and [M2\_CHAT\_RECOVERED.md](dev_plans/reliability_aware_geo_fusion/M2_CHAT_RECOVERED.md)  for our previous conversations that deliver M1 and M2. (You can also refer to [@Capturing geo helpfulness: M1](thread://01a034b5-d72a-7491-8f62-f7613fa38cca?hostId=remote-ssh-discovered%3Alin-remote) and [@Capturing geo helpfulness: M2](thread://01a048d7-ecbb-7bf2-a75c-668fd280fb03?hostId=remote-ssh-discovered%3Alin-remote), but these two conversations in Codex UI have index issue and cannot show full conversations. However, the complete conversations are saved in Codex and they are extracted as [M1\_CHAT\_RECOVERED.md](dev_plans/reliability_aware_geo_fusion/M1_CHAT_RECOVERED.md) and [M2\_CHAT\_RECOVERED.md](dev_plans/reliability_aware_geo_fusion/M2_CHAT_RECOVERED.md) .)

There are outputs from M1 and M2, which are recorded [01\_experimental\_protocol.md](dev_plans/reliability_aware_geo_fusion/01_experimental_protocol.md) and [02\_oof\_expert\_outputs.md](dev_plans/reliability_aware_geo_fusion/02_oof_expert_outputs.md), as well as the meta outputs, e.g., the saved encoders, parquet files, and so on. We will use them to implement M3.&#x20;

Overall, with M1 and M2 completed, I think M3 is straightforward to implement? How ready do you think we are for completing M3? Are there any gaps or clarifications needed to address before starting?




## 2026-09-01T10:14:33.178Z — Assistant — progress

I’ll audit M3 from three angles: the milestone’s acceptance criteria, the M1/M2 design decisions and artifact contracts, and what is actually present in the workspace. I’ll also read all three referenced tasks as required, then distinguish true blockers from implementation details we can safely resolve as we go.

## 2026-09-01T10:15:48.371Z — Assistant — progress

The core design is more frozen than the milestone summary initially suggests: M1 already specifies the exact 23 raw features, their 727-column transformed matrix, deterministic tie-breaking, categorical vocabularies, scaling rules, and forbidden-name audit. I’m now checking whether M2’s sealed aggregates expose everything needed to compute those features without reopening training data or validation labels.

## 2026-09-01T10:17:59.403Z — Assistant — progress

The artifact audit found ample support for M3: across the four seed-specific OOF realizations there are 1,225 rescue records from 310 plots and 2,256 harm records from 431 plots, so neither intervention class is globally too sparse to define or test. Two handoff inconsistencies—not missing data—are emerging as the only items worth settling before implementation.

## 2026-09-01T10:23:05.386Z — Assistant — progress

The existing protocol and code are healthy: the frozen M1 validator passes, and the focused M1/M2 suite is 84/84. The remaining issue is organizational rather than scientific—whether M3 owns the 12 expert-temperature fits needed to materialize calibrated features, or whether M3 stops at targets plus a pure builder and M4 materializes the final dataset.

## 2026-09-01T10:24:08.301Z — Assistant — final response

M3 is highly ready. There is no missing-data, GPU, or artifact blocker; only two specification seams should be recorded before coding.

What I verified:

- M1 validates with 4,200 rows, 1,625 plots, the frozen 1,300/325 plot split, and four OOF folds.
- All 20 real M2 producer bundles and both aggregates revalidate successfully.
- The OOF aggregate has 13,512 records—3,378 images × four seeds—uniquely keyed by `(row_uid, training_seed)`. The validation aggregate has 3,288 label-blind records.
- The focused M1/M2 suite passes: 84 tests.
- Preliminary OOF target support is adequate:

  - rescue: 1,225 records (9.1%), 283–327 per seed, appearing across 310 plots;
  - harm: 2,256 (16.7%), 549–575 per seed, across 431 plots;
  - both correct: 7,171;
  - both wrong: 2,860.

- The [frozen feature contract](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/configs/multimodal_geo_helpfulness.yaml:466) already specifies 30 semantic features and their deterministic expansion to 727 router columns.

The two seams are:

1. Calibration ownership. [M3 currently promises a materialized `router_dataset.parquet`](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/dev_plans/reliability_aware_geo_fusion/README.md:492), but M4 owns temperature fitting. Primary probability-derived features must use calibrated probabilities—not M2’s descriptive native-\(T=1\) probabilities.

   My recommendation is to preserve the interpretation established during M2:

   - M3 produces seed-specific targets, the pure feature builder and schema, prevalence reports, and leakage audit.
   - M4 fits the 12 temperatures—three modes × four seeds—and materializes the calibrated `router_dataset.parquet`.

   The README should be corrected to reflect this boundary. Alternatively, temperature fitting must move into M3.

2. Target identity. The [frozen target schema](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/configs/multimodal_geo_helpfulness.yaml:885) keys targets only by `row_uid`, but targets are seed-specific: 989 of 3,378 images change state across seeds. We should preserve M1 immutability and document the combined M3 target artifact as keyed by `(row_uid, training_seed)`, optionally with one seed-scoped view per seed.

Everything else is an implementation detail we can settle safely: deterministic top-3 tie-breaking, exact report schemas, atomic artifact publication, and direct dependency hashes. M3 should be additive—new module, runner, tests, and `03_targets_and_features.md`—without modifying fingerprinted M1/M2 code.

So: ready to begin, with the calibration/M3–M4 boundary as the only choice needing confirmation. I recommend keeping temperature fitting and final dataset materialization in M4.

## 2026-09-01T13:19:15.528Z — User


# Response annotations:
Each item contains text selected from an earlier Codex response and may include a user comment. Treat items as Annotation 1, Annotation 2, and so on in array order. Use every selection as context and address every comment. For every annotation you address, include its inline directive `:codex-annotation{index="N"}`, where N is its one-based array position (for example, `:codex-annotation{index="1"}`). Do not use unstructured annotation labels.
<response-annotations>
[{"text":"Target identity. The frozen target schema keys targets only by row_uid, but targets are seed-specific: 989 of 3,378 images change state across seeds. We should preserve M1 immutability and document the combined M3 target artifact as keyed by (row_uid, training_seed), optionally with one seed-scoped view per seed."}]
</response-annotations>

## My request:
Can you please explain this with more details? What is `row_uid`? What do you mean by saying "989 of 3,378 images change state across seeds"?


## 2026-09-01T13:19:22.049Z — Assistant — progress

I’ll trace the exact `row_uid` construction in the frozen protocol, then illustrate how one fixed image can receive different rescue/harm states from different seed-specific expert predictions.

## 2026-09-01T13:20:16.315Z — Assistant — final response

`row_uid` identifies the image record; `training_seed` identifies which independently trained expert realization produced its prediction. Together they identify one OOF prediction—and therefore one routing target. :codex-annotation{index="1"}

### What is `row_uid`?

`row_uid` is a stable, label-independent identifier for one dataset row. It is a 64-character SHA-256 hash calculated from:

```text
[dataset_id, normalized lowercase file path, normalized plot_idx]
```

The exact construction is frozen in the [experimental protocol](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/dev_plans/reliability_aware_geo_fusion/01_experimental_protocol.md:125).

It does not contain:

- the habitat label;
- model predictions;
- the OOF fold;
- the training seed;
- correctness information.

Consequently, the same physical image has the same `row_uid` across all expert modes and training seeds.

### Why can one image have different target states?

Each of seeds 1–4 trains an independent realization of the image, geo, and raw-concatenation experts. The data split is the same, but initialization, shuffling, and stochastic training can produce different predictions.

The target is derived separately for each seed:

| Raw prediction | Geo prediction | Target |
|---|---|---|
| Wrong | Correct | `rescue` |
| Correct | Wrong | `harm` |
| Correct | Correct | `both_correct` |
| Wrong | Wrong | `both_wrong` |

For example, one actual OOF image has true dense label `5`:

| Seed | Raw prediction | Geo prediction | Derived target |
|---:|---:|---:|---|
| 1 | 5 | 5 | `both_correct` |
| 2 | 5 | 7 | `harm` |
| 3 | 1 | 5 | `rescue` |
| 4 | 5 | 5 | `both_correct` |

This is one image with one `row_uid`, but four seed-specific prediction records and three distinct target states.

### What does “989 of 3,378 images change state” mean?

For each of the 3,378 development-train images, I compared its four target states—one from each training seed:

- 2,389 images have the same state for all four seeds.
- 875 images exhibit two different states.
- 100 exhibit three different states.
- 14 exhibit all four states.

Therefore:

\[
875+100+14=989
\]

So 989 images, or 29.28%, do not have one seed-independent target.

This does not mean their true habitat changes. It means the independently trained experts disagree in correctness across seed realizations.

### Why the composite key is necessary

The combined OOF table contains:

\[
3{,}378\text{ images}\times4\text{ seeds}=13{,}512\text{ records}.
\]

Using only `row_uid` as the key would produce four colliding records for every image. It would also incorrectly imply that each image has one universal target.

The correct combined key is therefore:

```text
(row_uid, training_seed)
```

A seed-specific target file can still use `row_uid` alone because it contains only one record per image. But the combined M3 artifact must retain `training_seed`; targets should not be averaged or majority-voted across seeds. M4 will fit one seed-specific router realization using the corresponding 3,378 targets.

## 2026-09-01T13:37:04.408Z — User


# Response annotations:
Each item contains text selected from an earlier Codex response and may include a user comment. Treat items as Annotation 1, Annotation 2, and so on in array order. Use every selection as context and address every comment. For every annotation you address, include its inline directive `:codex-annotation{index="N"}`, where N is its one-based array position (for example, `:codex-annotation{index="1"}`). Do not use unstructured annotation labels.
<response-annotations>
[{"text":"Calibration ownership. M3 currently promises a materialized `router_dataset.parquet`, but M4 owns temperature fitting. Primary probability-derived features must use calibrated probabilities—not M2’s descriptive native-\\(T=1\\) probabilities.My recommendation is to preserve the interpretation established during M2:M3 produces seed-specific targets, the pure feature builder and schema, prevalence reports, and leakage audit.M4 fits the 12 temperatures—three modes × four seeds—and materializes the calibrated router_dataset.parquet.The README should be corrected to reflect this boundary. Alternatively, temperature fitting must move into M3.Target identity. The frozen target schema keys targets only by row_uid, but targets are seed-specific: 989 of 3,378 images change state across seeds. We should preserve M1 immutability and document the combined M3 target artifact as keyed by (row_uid, training_seed), optionally with one seed-scoped view per seed."}]
</response-annotations>

## My request:
Thanks for the explanation. I got it.&#x20;

I agree with you that M3 target artifact should be keyed by `(row_uid, training_seed)`. I also agree your proposal for the M3/M4 boundary, leaving probability calibration in M4.&#x20;

Do you think we are good to go? If so, please just reply with green light, then I will prompt you to do a plan.&#x20;




## 2026-09-01T13:37:10.426Z — Assistant — final response

🟢 Green light. :codex-annotation{index="1"}

## 2026-09-01T13:38:48.978Z — User

Since you give the green light, and I currently do not have further questions. Let us first scaffold an implementation for delivering M3 based on your discussion.&#x20;

In this plan, if relevant documents need to be updated, please do it accordingly. 


## 2026-09-01T13:39:01.302Z — Assistant — progress

I’ll turn the agreed boundary into a decision-complete M3 implementation plan. I’ll inspect the immutable M1/M2 extension pattern, determine the exact M3 artifacts and CLI contract, and include the governing README plus the new milestone record so documentation and implementation cannot drift.

## 2026-09-01T13:45:12.506Z — Assistant — progress

The implementation boundary is now concrete: M3 will publish one combined label-bearing target table keyed by `(row_uid, training_seed)` and a frozen, label-blind 30-feature contract, but no calibrated feature rows. The M4 runner will later fit the 12 temperatures, call that same pure builder, and create `router_dataset.parquet`.

## 2026-09-01T13:50:35.642Z — Assistant — final response

<proposed_plan>
# M3 — Seed-Specific Targets and Deployment-Safe Feature Contract

## Summary

Implement M3 as an additive, CPU-only child of immutable M1/M2. M3 will:

- derive raw-fusion-relative targets for all 13,512 OOF seed records;
- publish one combined target table keyed by `(row_uid, training_seed)`;
- implement the single pure 30-feature builder shared by future router training and inference;
- freeze the feature schema, prevalence report, leakage audit, and provenance;
- leave all temperature calibration, learned transforms, the 727-column matrix, and `router_dataset.parquet` to M4.

M3 is complete only when the real artifacts are built, sealed, revalidated, and documented. Code and synthetic tests alone are insufficient.

## Implementation Changes

### 1. Pure targets and feature API

Add a stable module, `multimodal/geo_helpfulness_targets_features.py`, containing:

- `derive_router_target_states(...)` for the frozen order:
  `rescue`, `harm`, `both_correct`, `both_wrong`;
- `build_router_target_table(...)`;
- `derive_image_relative_states(...)`, usable only by diagnostic reporting;
- `build_router_feature_frame(...)`;
- feature-schema and leakage-audit validators;
- `load_validated_m3_bundle(...)`, which M4 will use without modifying this module.

The combined target table will contain exactly:

```text
schema_version
protocol_id
row_uid
plot_idx
training_seed
target_state
```

It will:

- have 13,512 rows and canonical order `(row_uid, training_seed)`;
- join OOF predictions many-to-one with development-train labels by `row_uid`;
- retain all four seed realizations without averaging, voting, or deduplication;
- validate plot identity and OOF-fold agreement before omitting the fold from the published table;
- serialize no labels, correctness flags, logits, probabilities, utility, or image-relative diagnostic states.

Reuse `protocol_v1`; do not create `protocol_v2` or edit the frozen YAML. The M3 manifest will freeze the additive physical table schema and composite key. A fixed-seed projection remains uniquely keyed by `row_uid`.

### 2. Frozen semantic feature builder

`build_router_feature_frame` will accept only three equal, non-empty `(n, 18)` probability matrices and an explicit:

```text
probability_basis = "scalar_temperature_calibrated"
```

It must reject M2’s `native_t1_uncalibrated` basis. It will fit no state and will never accept labels, targets, identities, folds, seeds, or correctness.

Emit exactly the configured 30 semantic features in frozen family order:

- 5 categorical;
- 3 Boolean;
- 2 integer;
- 20 numeric.

Freeze these conventions:

- predictions and top-k ordering: descending probability, then ascending dense ID;
- pair IDs: `image * 18 + geo` and `geo * 18 + raw`;
- top-3 overlap: integer intersection cardinality `0..3`;
- raw rank of geo prediction: one-based `1..18`;
- entropy: unnormalised natural-log entropy;
- JSD: natural-log, unnormalised Jensen–Shannon divergence;
- TV: half-L1 distance;
- zero-probability entropy/JSD terms contribute zero;
- probability tolerance: absolute row-sum tolerance `1e-8`, with no clipping or renormalisation.

Use exact dtypes: predictions `int8`, pair IDs `int16`, ranks/overlap `int8`, flags `bool`, numeric values `float64`. The schema records fixed vocabularies `0..17` and `0..323`, plus M4’s later expansion of 25 scaled values and 702 one-hot values into 727 `float64` columns.

Do not reuse the existing agreement helpers without adapting their semantics: they currently normalise entropy/JSD and return fractional top-3 overlap.

### 3. M3 build, validation, and artifacts

Add:

```bash
python tools/run_multimodal_geo_helpfulness_m3.py build
python tools/run_multimodal_geo_helpfulness_m3.py validate
```

Both accept only the standard config, protocol-directory, and artifact-root overrides. There will be no temperature, validation-output, test, or force-overwrite arguments.

Before writing:

- run the frozen M1 preflight;
- validate only the 16 train-OOF producer outputs and their manifests;
- reconstruct their canonical concatenation and compare it with the sealed OOF aggregate;
- verify physical/logical hashes, schema, seed/fold membership, M1 parent hashes, and current M2 implementation hashes;
- load development-train labels through a role-filtered assignment projection;
- never open development-validation outputs, final in-sample outputs, or locked-test sources.

Publish an immutable bundle at:

```text
router/targets_and_feature_contract/
├── router_targets.parquet
├── router_feature_schema.json
├── target_prevalence.json
├── feature_leakage_audit.json
└── manifest.json
```

Stage all files, publish the manifest last as the commit marker, and use exclusive no-replace semantics. A valid rerun returns `reused_valid`; committed corruption or stale lineage fails without overwrite. Only an owned, uncommitted partial publication may be cleaned and rebuilt.

The manifest will bind:

- protocol, assignments, ontology, resolved config, and allow-list hashes;
- OOF aggregate file/content/manifest hashes and all 16 producer-manifest hashes;
- M3 module, runner, schemas, and every child artifact hash;
- the composite key and canonical order;
- explicit declarations that no temperature was fitted, no calibrated feature rows were materialized, native-\(T=1\) was not used for primary features, and M4 owns `router_dataset.parquet`.

### 4. Reports and documentation

`target_prevalence.json` will deterministically report:

- pooled seed-realization counts and fractions;
- separate counts for seeds 1–4;
- complete habitat, plot, image–geo pair, and geo–raw pair breakdowns;
- zero-support categories using `null` rates, never NaN/Infinity;
- auxiliary image-versus-geo correctness states;
- cross-seed stability by the number of distinct states per `row_uid`;
- an explicit warning that 13,512 seed records represent 3,378 images, not independent biological samples.

The leakage audit will verify the exact ordered allow-list, configured forbidden patterns, identity/group exclusions, calibrated-basis requirement, target/feature separation, and allowed input artifact roles.

Documentation work:

- update the governing README’s M3/M4 tasks, artifact tree, code layout, acceptance criteria, decision log, progress ledger, change log, and immediate next action;
- add a dated downstream-handoff clarification to `02_oof_expert_outputs.md`;
- create `03_targets_and_features.md` with commands, contracts, hashes, actual prevalence, tests, boundary evidence, and completion checklist;
- after successful artifact execution, mark M3 complete and link its record; otherwise leave it in progress.

Do not edit fingerprinted M1/M2 configuration, protocol, runner, model, OOF, or locked-evaluation code, `01_experimental_protocol.md`, or recovered chat transcripts.

## Test and Completion Plan

Add focused pure and artifact tests covering:

- all four primary target states and independence from `image_pred`;
- seed-varying targets retained under the composite key;
- duplicate/missing seeds, identities, folds, labels, predictions, mixed protocols, and validation rows rejected;
- all 30 feature formulas, exact order, dtypes, vocabularies, and schema hash;
- tie-breaking, zero probabilities, missing local categories, and malformed probability matrices;
- regression anchors: uniform entropy `ln(18)`, disjoint one-hot JSD `ln(2)`, full top-3 overlap `3`, and disjoint TV `1`;
- native-\(T=1\), forbidden aliases, unexpected columns, and identity/group metadata rejected;
- train/inference calls producing identical features through the same stateless function;
- prevalence reconciliation at every grouping level;
- no validation/test reads and no calibration/router fitting;
- staged publication, valid reuse, crash recovery, tamper detection, stale-parent rejection, and immutable M1/M2 hashes.

Run the new suite together with the existing 84 M1/M2 geo tests, then the repository-wide test suite and documentation/link/whitespace checks.

Finally build and validate the real bundle twice. Completion evidence must reproduce:

| State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| `rescue` | 1,225 | 327 | 318 | 283 | 297 |
| `harm` | 2,256 | 560 | 549 | 572 | 575 |
| `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
| `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |

Also verify 3,378 unique images, 1,300 plots, four seeds, 989 images changing state across seeds, a passing leakage audit, and the absence of `router_dataset.parquet`. These values are verification evidence, not hardcoded target-generation inputs.

## Assumptions and Defaults

- Publish only the combined target table; do not duplicate physical per-seed target files.
- M3 introduces no dependency beyond the existing NumPy, pandas, PyArrow, and YAML stack.
- M4 fits the 12 scalar temperatures, calls the frozen M3 builder, fits scaling/one-hot state, materializes `router_dataset.parquet`, and trains router candidates.
- Any change to the 30-feature contract, target definition, seed handling, or calibration boundary requires an explicit later protocol decision rather than silently rebuilding M3.
</proposed_plan>

## 2026-09-01T14:22:35.769Z — User

I have reviewed the plan. Overall, it is good. I have one follow-up question before implementing it. Regarding `build_router_feature_frame`, why do you say "It must reject M2’s `native_t1_uncalibrated` basis"?


## 2026-09-01T14:23:44.453Z — Assistant — final response

M2’s native-\(T=1\) probabilities are valid, but only as reproducibility and descriptive outputs. The rejection prevents them from being mistaken for protocol-v1 router inputs.

Each expert head is trained independently, so its logit scale can differ. For example, two equally reliable experts could produce maximum native probabilities of 0.99 and 0.75 merely because one head emits larger logits. Comparing those values would make the first expert appear more trustworthy even when it is not.

Temperature scaling matters for most router features:

- confidence, entropy, and margin;
- confidence/entropy/margin differences;
- JSD and total variation;
- cross-probabilities assigned to another expert’s prediction.

Predicted classes, top-3 membership, and ranks remain unchanged under positive scalar temperatures, which is why M3 can safely use M2 predictions to construct targets. But the probability-valued features change.

`build_router_feature_frame` is intended to be the official protocol-facing builder, not a generic probability-analysis utility. Requiring `scalar_temperature_calibrated` therefore provides a fail-closed guarantee:

1. M3 cannot accidentally materialize primary features from native probabilities.
2. M4 must fit the 12 declared temperatures from OOF logits.
3. M4 can verify that each supplied probability matrix equals `softmax(logits / T)`.
4. Router training and later inference use the same probability basis.

A future native-\(T=1\) ablation could use a separately labelled diagnostic entry point and artifact. It should not weaken the primary builder’s guarantee.

<proposed_plan>
# M3 — Seed-Specific Targets and Deployment-Safe Feature Contract

## Summary

Implement M3 as an additive, CPU-only child of immutable M1/M2. M3 will:

- derive raw-fusion-relative targets for all 13,512 OOF seed records;
- publish one combined target table keyed by `(row_uid, training_seed)`;
- implement the single pure 30-feature builder shared by future router training and inference;
- freeze the feature schema, prevalence report, leakage audit, and provenance;
- leave all temperature calibration, learned transforms, the 727-column matrix, and `router_dataset.parquet` to M4.

M3 is complete only when the real artifacts are built, sealed, revalidated, and documented. Code and synthetic tests alone are insufficient.

## Implementation Changes

### 1. Pure targets and feature API

Add a stable module, `multimodal/geo_helpfulness_targets_features.py`, containing:

- `derive_router_target_states(...)` for the frozen order:
  `rescue`, `harm`, `both_correct`, `both_wrong`;
- `build_router_target_table(...)`;
- `derive_image_relative_states(...)`, usable only by diagnostic reporting;
- `build_router_feature_frame(...)`;
- feature-schema and leakage-audit validators;
- `load_validated_m3_bundle(...)`, which M4 will use without modifying this module.

The combined target table will contain exactly:

```text
schema_version
protocol_id
row_uid
plot_idx
training_seed
target_state
```

It will:

- have 13,512 rows and canonical order `(row_uid, training_seed)`;
- join OOF predictions many-to-one with development-train labels by `row_uid`;
- retain all four seed realizations without averaging, voting, or deduplication;
- validate plot identity and OOF-fold agreement before omitting the fold from the published table;
- serialize no labels, correctness flags, logits, probabilities, utility, or image-relative diagnostic states.

Reuse `protocol_v1`; do not create `protocol_v2` or edit the frozen YAML. The M3 manifest will freeze the additive physical table schema and composite key. A fixed-seed projection remains uniquely keyed by `row_uid`.

### 2. Frozen semantic feature builder

`build_router_feature_frame` will accept only three equal, non-empty `(n, 18)` probability matrices and an explicit:

```text
probability_basis = "scalar_temperature_calibrated"
```

It must reject M2’s `native_t1_uncalibrated` basis. It will fit no state and will never accept labels, targets, identities, folds, seeds, or correctness.

Emit exactly the configured 30 semantic features in frozen family order:

- 5 categorical;
- 3 Boolean;
- 2 integer;
- 20 numeric.

Freeze these conventions:

- predictions and top-k ordering: descending probability, then ascending dense ID;
- pair IDs: `image * 18 + geo` and `geo * 18 + raw`;
- top-3 overlap: integer intersection cardinality `0..3`;
- raw rank of geo prediction: one-based `1..18`;
- entropy: unnormalised natural-log entropy;
- JSD: natural-log, unnormalised Jensen–Shannon divergence;
- TV: half-L1 distance;
- zero-probability entropy/JSD terms contribute zero;
- probability tolerance: absolute row-sum tolerance `1e-8`, with no clipping or renormalisation.

Use exact dtypes: predictions `int8`, pair IDs `int16`, ranks/overlap `int8`, flags `bool`, numeric values `float64`. The schema records fixed vocabularies `0..17` and `0..323`, plus M4’s later expansion of 25 scaled values and 702 one-hot values into 727 `float64` columns.

Do not reuse the existing agreement helpers without adapting their semantics: they currently normalise entropy/JSD and return fractional top-3 overlap.

### 3. M3 build, validation, and artifacts

Add:

```bash
python tools/run_multimodal_geo_helpfulness_m3.py build
python tools/run_multimodal_geo_helpfulness_m3.py validate
```

Both accept only the standard config, protocol-directory, and artifact-root overrides. There will be no temperature, validation-output, test, or force-overwrite arguments.

Before writing:

- run the frozen M1 preflight;
- validate only the 16 train-OOF producer outputs and their manifests;
- reconstruct their canonical concatenation and compare it with the sealed OOF aggregate;
- verify physical/logical hashes, schema, seed/fold membership, M1 parent hashes, and current M2 implementation hashes;
- load development-train labels through a role-filtered assignment projection;
- never open development-validation outputs, final in-sample outputs, or locked-test sources.

Publish an immutable bundle at:

```text
router/targets_and_feature_contract/
├── router_targets.parquet
├── router_feature_schema.json
├── target_prevalence.json
├── feature_leakage_audit.json
└── manifest.json
```

Stage all files, publish the manifest last as the commit marker, and use exclusive no-replace semantics. A valid rerun returns `reused_valid`; committed corruption or stale lineage fails without overwrite. Only an owned, uncommitted partial publication may be cleaned and rebuilt.

The manifest will bind:

- protocol, assignments, ontology, resolved config, and allow-list hashes;
- OOF aggregate file/content/manifest hashes and all 16 producer-manifest hashes;
- M3 module, runner, schemas, and every child artifact hash;
- the composite key and canonical order;
- explicit declarations that no temperature was fitted, no calibrated feature rows were materialized, native-\(T=1\) was not used for primary features, and M4 owns `router_dataset.parquet`.

### 4. Reports and documentation

`target_prevalence.json` will deterministically report:

- pooled seed-realization counts and fractions;
- separate counts for seeds 1–4;
- complete habitat, plot, image–geo pair, and geo–raw pair breakdowns;
- zero-support categories using `null` rates, never NaN/Infinity;
- auxiliary image-versus-geo correctness states;
- cross-seed stability by the number of distinct states per `row_uid`;
- an explicit warning that 13,512 seed records represent 3,378 images, not independent biological samples.

The leakage audit will verify the exact ordered allow-list, configured forbidden patterns, identity/group exclusions, calibrated-basis requirement, target/feature separation, and allowed input artifact roles.

Documentation work:

- update the governing README’s M3/M4 tasks, artifact tree, code layout, acceptance criteria, decision log, progress ledger, change log, and immediate next action;
- add a dated downstream-handoff clarification to `02_oof_expert_outputs.md`;
- create `03_targets_and_features.md` with commands, contracts, hashes, actual prevalence, tests, boundary evidence, and completion checklist;
- after successful artifact execution, mark M3 complete and link its record; otherwise leave it in progress.

Do not edit fingerprinted M1/M2 configuration, protocol, runner, model, OOF, or locked-evaluation code, `01_experimental_protocol.md`, or recovered chat transcripts.

## Test and Completion Plan

Add focused pure and artifact tests covering:

- all four primary target states and independence from `image_pred`;
- seed-varying targets retained under the composite key;
- duplicate/missing seeds, identities, folds, labels, predictions, mixed protocols, and validation rows rejected;
- all 30 feature formulas, exact order, dtypes, vocabularies, and schema hash;
- tie-breaking, zero probabilities, missing local categories, and malformed probability matrices;
- regression anchors: uniform entropy `ln(18)`, disjoint one-hot JSD `ln(2)`, full top-3 overlap `3`, and disjoint TV `1`;
- native-\(T=1\), forbidden aliases, unexpected columns, and identity/group metadata rejected;
- train/inference calls producing identical features through the same stateless function;
- prevalence reconciliation at every grouping level;
- no validation/test reads and no calibration/router fitting;
- staged publication, valid reuse, crash recovery, tamper detection, stale-parent rejection, and immutable M1/M2 hashes.

Run the new suite together with the existing 84 M1/M2 geo tests, then the repository-wide test suite and documentation/link/whitespace checks.

Finally build and validate the real bundle twice. Completion evidence must reproduce:

| State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| `rescue` | 1,225 | 327 | 318 | 283 | 297 |
| `harm` | 2,256 | 560 | 549 | 572 | 575 |
| `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
| `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |

Also verify 3,378 unique images, 1,300 plots, four seeds, 989 images changing state across seeds, a passing leakage audit, and the absence of `router_dataset.parquet`. These values are verification evidence, not hardcoded target-generation inputs.

## Assumptions and Defaults

- Publish only the combined target table; do not duplicate physical per-seed target files.
- M3 introduces no dependency beyond the existing NumPy, pandas, PyArrow, and YAML stack.
- M4 fits the 12 scalar temperatures, calls the frozen M3 builder, fits scaling/one-hot state, materializes `router_dataset.parquet`, and trains router candidates.
- Any change to the 30-feature contract, target definition, seed handling, or calibration boundary requires an explicit later protocol decision rather than silently rebuilding M3.
</proposed_plan>

## 2026-09-01T14:26:29.637Z — User

PLEASE IMPLEMENT THIS PLAN:
# M3 — Seed-Specific Targets and Deployment-Safe Feature Contract

## Summary

Implement M3 as an additive, CPU-only child of immutable M1/M2. M3 will:

- derive raw-fusion-relative targets for all 13,512 OOF seed records;
- publish one combined target table keyed by `(row_uid, training_seed)`;
- implement the single pure 30-feature builder shared by future router training and inference;
- freeze the feature schema, prevalence report, leakage audit, and provenance;
- leave all temperature calibration, learned transforms, the 727-column matrix, and `router_dataset.parquet` to M4.

M3 is complete only when the real artifacts are built, sealed, revalidated, and documented. Code and synthetic tests alone are insufficient.

## Implementation Changes

### 1. Pure targets and feature API

Add a stable module, `multimodal/geo_helpfulness_targets_features.py`, containing:

- `derive_router_target_states(...)` for the frozen order:
  `rescue`, `harm`, `both_correct`, `both_wrong`;
- `build_router_target_table(...)`;
- `derive_image_relative_states(...)`, usable only by diagnostic reporting;
- `build_router_feature_frame(...)`;
- feature-schema and leakage-audit validators;
- `load_validated_m3_bundle(...)`, which M4 will use without modifying this module.

The combined target table will contain exactly:

```text
schema_version
protocol_id
row_uid
plot_idx
training_seed
target_state
```

It will:

- have 13,512 rows and canonical order `(row_uid, training_seed)`;
- join OOF predictions many-to-one with development-train labels by `row_uid`;
- retain all four seed realizations without averaging, voting, or deduplication;
- validate plot identity and OOF-fold agreement before omitting the fold from the published table;
- serialize no labels, correctness flags, logits, probabilities, utility, or image-relative diagnostic states.

Reuse `protocol_v1`; do not create `protocol_v2` or edit the frozen YAML. The M3 manifest will freeze the additive physical table schema and composite key. A fixed-seed projection remains uniquely keyed by `row_uid`.

### 2. Frozen semantic feature builder

`build_router_feature_frame` will accept only three equal, non-empty `(n, 18)` probability matrices and an explicit:

```text
probability_basis = "scalar_temperature_calibrated"
```

It must reject M2’s `native_t1_uncalibrated` basis. It will fit no state and will never accept labels, targets, identities, folds, seeds, or correctness.

Emit exactly the configured 30 semantic features in frozen family order:

- 5 categorical;
- 3 Boolean;
- 2 integer;
- 20 numeric.

Freeze these conventions:

- predictions and top-k ordering: descending probability, then ascending dense ID;
- pair IDs: `image * 18 + geo` and `geo * 18 + raw`;
- top-3 overlap: integer intersection cardinality `0..3`;
- raw rank of geo prediction: one-based `1..18`;
- entropy: unnormalised natural-log entropy;
- JSD: natural-log, unnormalised Jensen–Shannon divergence;
- TV: half-L1 distance;
- zero-probability entropy/JSD terms contribute zero;
- probability tolerance: absolute row-sum tolerance `1e-8`, with no clipping or renormalisation.

Use exact dtypes: predictions `int8`, pair IDs `int16`, ranks/overlap `int8`, flags `bool`, numeric values `float64`. The schema records fixed vocabularies `0..17` and `0..323`, plus M4’s later expansion of 25 scaled values and 702 one-hot values into 727 `float64` columns.

Do not reuse the existing agreement helpers without adapting their semantics: they currently normalise entropy/JSD and return fractional top-3 overlap.

### 3. M3 build, validation, and artifacts

Add:

```bash
python tools/run_multimodal_geo_helpfulness_m3.py build
python tools/run_multimodal_geo_helpfulness_m3.py validate
```

Both accept only the standard config, protocol-directory, and artifact-root overrides. There will be no temperature, validation-output, test, or force-overwrite arguments.

Before writing:

- run the frozen M1 preflight;
- validate only the 16 train-OOF producer outputs and their manifests;
- reconstruct their canonical concatenation and compare it with the sealed OOF aggregate;
- verify physical/logical hashes, schema, seed/fold membership, M1 parent hashes, and current M2 implementation hashes;
- load development-train labels through a role-filtered assignment projection;
- never open development-validation outputs, final in-sample outputs, or locked-test sources.

Publish an immutable bundle at:

```text
router/targets_and_feature_contract/
├── router_targets.parquet
├── router_feature_schema.json
├── target_prevalence.json
├── feature_leakage_audit.json
└── manifest.json
```

Stage all files, publish the manifest last as the commit marker, and use exclusive no-replace semantics. A valid rerun returns `reused_valid`; committed corruption or stale lineage fails without overwrite. Only an owned, uncommitted partial publication may be cleaned and rebuilt.

The manifest will bind:

- protocol, assignments, ontology, resolved config, and allow-list hashes;
- OOF aggregate file/content/manifest hashes and all 16 producer-manifest hashes;
- M3 module, runner, schemas, and every child artifact hash;
- the composite key and canonical order;
- explicit declarations that no temperature was fitted, no calibrated feature rows were materialized, native-\(T=1\) was not used for primary features, and M4 owns `router_dataset.parquet`.

### 4. Reports and documentation

`target_prevalence.json` will deterministically report:

- pooled seed-realization counts and fractions;
- separate counts for seeds 1–4;
- complete habitat, plot, image–geo pair, and geo–raw pair breakdowns;
- zero-support categories using `null` rates, never NaN/Infinity;
- auxiliary image-versus-geo correctness states;
- cross-seed stability by the number of distinct states per `row_uid`;
- an explicit warning that 13,512 seed records represent 3,378 images, not independent biological samples.

The leakage audit will verify the exact ordered allow-list, configured forbidden patterns, identity/group exclusions, calibrated-basis requirement, target/feature separation, and allowed input artifact roles.

Documentation work:

- update the governing README’s M3/M4 tasks, artifact tree, code layout, acceptance criteria, decision log, progress ledger, change log, and immediate next action;
- add a dated downstream-handoff clarification to `02_oof_expert_outputs.md`;
- create `03_targets_and_features.md` with commands, contracts, hashes, actual prevalence, tests, boundary evidence, and completion checklist;
- after successful artifact execution, mark M3 complete and link its record; otherwise leave it in progress.

Do not edit fingerprinted M1/M2 configuration, protocol, runner, model, OOF, or locked-evaluation code, `01_experimental_protocol.md`, or recovered chat transcripts.

## Test and Completion Plan

Add focused pure and artifact tests covering:

- all four primary target states and independence from `image_pred`;
- seed-varying targets retained under the composite key;
- duplicate/missing seeds, identities, folds, labels, predictions, mixed protocols, and validation rows rejected;
- all 30 feature formulas, exact order, dtypes, vocabularies, and schema hash;
- tie-breaking, zero probabilities, missing local categories, and malformed probability matrices;
- regression anchors: uniform entropy `ln(18)`, disjoint one-hot JSD `ln(2)`, full top-3 overlap `3`, and disjoint TV `1`;
- native-\(T=1\), forbidden aliases, unexpected columns, and identity/group metadata rejected;
- train/inference calls producing identical features through the same stateless function;
- prevalence reconciliation at every grouping level;
- no validation/test reads and no calibration/router fitting;
- staged publication, valid reuse, crash recovery, tamper detection, stale-parent rejection, and immutable M1/M2 hashes.

Run the new suite together with the existing 84 M1/M2 geo tests, then the repository-wide test suite and documentation/link/whitespace checks.

Finally build and validate the real bundle twice. Completion evidence must reproduce:

| State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| `rescue` | 1,225 | 327 | 318 | 283 | 297 |
| `harm` | 2,256 | 560 | 549 | 572 | 575 |
| `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
| `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |

Also verify 3,378 unique images, 1,300 plots, four seeds, 989 images changing state across seeds, a passing leakage audit, and the absence of `router_dataset.parquet`. These values are verification evidence, not hardcoded target-generation inputs.

## Assumptions and Defaults

- Publish only the combined target table; do not duplicate physical per-seed target files.
- M3 introduces no dependency beyond the existing NumPy, pandas, PyArrow, and YAML stack.
- M4 fits the 12 scalar temperatures, calls the frozen M3 builder, fits scaling/one-hot state, materializes `router_dataset.parquet`, and trains router candidates.
- Any change to the 30-feature contract, target definition, seed handling, or calibration boundary requires an explicit later protocol decision rather than silently rebuilding M3.


## 2026-09-01T14:26:41.993Z — Assistant — progress

I’m implementing M3 end to end, including the real immutable bundle and completion documentation. I’ll keep every fingerprinted M1/M2 source untouched, develop the pure target/feature contract and artifact workflow additively, then run the existing and new suites before publishing the real artifacts.

## 2026-09-01T14:29:06.716Z — Assistant — progress

The immutable M1/M2 baselines are recorded, and the implementation is now split across the pure contract, artifact workflow, and focused tests. I’m reconciling those pieces next, then I’ll run the real build twice before updating the completion record.

## 2026-09-01T14:31:18.783Z — Assistant — progress

The contract boundary is now explicit in the implementation: M3 will expose only the three approved path overrides, and the pure feature API imports no fitting machinery. While the code finishes, I’m applying the non-fingerprinted M2→M3 handoff clarification; completion status will remain unchanged until the real artifacts pass validation.

## 2026-09-01T14:38:24.212Z — Assistant — progress

The pure contract suite is exercising the important edge cases now: all feature formulas and ordering checks are in place, while the remaining failures are concentrated in target-input strictness and report-only diagnostics—not the mathematical feature definitions. I’m using those failures as the acceptance checklist before allowing the artifact layer to seal anything.

## 2026-09-01T14:45:06.884Z — Assistant — progress

The pure suite is now fully green at 39 tests, including invalid seeds and report reconciliation. I’ve also updated the living specification to reflect the composite key and M3/M4 ownership boundary, but it still correctly labels M3 “in progress” until the real bundle exists.

## 2026-09-01T14:58:40.763Z — Assistant — progress

The artifact lifecycle suite is green as well: 46 new M3 tests now cover true interrupted-publication recovery, immutable reuse, tamper/stale-lineage rejection, train-only label projection, the initial M4 boundary, and a literal feature-schema hash. I’m running the entire frozen M1–M3 suite before the first irreversible publication.

## 2026-09-01T15:02:17.163Z — Assistant — progress

The new code has passed adversarial review with no remaining high-priority findings. One combined-run process is exiting abnormally partway through the legacy suite without writing its JUnit receipt, so I’m isolating the legacy files instead of treating partial dot output as success; the 47-test M3 suite itself is clean.

## 2026-09-01T15:05:08.198Z — Assistant — progress

The apparent legacy-suite cutoff was a tooling handoff, not a test failure—the M2 artifact test simply runs longer than the first command window. I’ve retained its live session and am polling it explicitly so the final report will contain a real exit status and JUnit total.

## 2026-09-01T15:05:50.624Z — Assistant — progress

All implementation fingerprints are now frozen, the 131 focused M1–M3 tests pass when run by file, and every recorded M1/M2 hash still matches its baseline. The publication directory and M4 dataset are absent, so I’m starting the first real immutable M3 build now.

## 2026-09-01T15:07:06.130Z — Assistant — progress

The real bundle is now committed successfully: 13,512 targets, 3,378 images, 1,300 plots, all four seeds, the required state totals, 989 seed-varying images, and a passing leakage audit. No calibration or router dataset was materialized. I’m now exercising the independent validate path and then the second-build reuse path against this exact commit.

## 2026-09-01T15:08:17.432Z — Assistant — progress

The validate command has completed its first minute without emitting an error; it is re-hashing and reconstructing the 16 large M2 parents, which is intentionally the expensive part of the lineage check. I’m keeping the live session attached until it returns the signed validation receipt.

## 2026-09-01T15:09:42.857Z — Assistant — progress

Both post-publication paths are now clean: explicit validation returned `reused_valid`, and the second build reused the same manifest hash without rewriting anything. I’m extracting the sealed file/content hashes and report details now, then I’ll replace the provisional documentation status with the actual completion evidence.

## 2026-09-01T15:11:53.427Z — Assistant — progress

The repository-wide run collected 209 tests and is still active. It has surfaced one failure outside the new M3 suite; I’m letting the full run finish so I can distinguish a pre-existing/environmental issue from an M3 regression using the complete traceback and JUnit record.

## 2026-09-01T15:13:30.689Z — Assistant — progress

The final repository-wide result is 208/209. The sole failure is unrelated to M3 and comes from a pre-existing untracked 10 m test whose `startswith("I")` assertion unintentionally counts the required `ID` metadata column. I’ve recorded that transparently; the M1–M3 focused total is 131/131. I’m finishing link, whitespace, fingerprint, and artifact-permission checks now.

## 2026-09-01T15:15:37.425Z — Assistant — progress

The M3 implementation and real artifact bundle are complete. I’m doing the final integrity pass now: whitespace/link checks, compilation, intended-file status, and confirmation that frozen M1/M2 inputs were untouched.

## 2026-09-01T15:16:08.603Z — Assistant — progress

The final code-quality checks are clean: no diff errors, trailing whitespace, stale M3 status text, or Python syntax failures. The status also confirms only the intended M3 code/docs are changed; the recovered M1/M2 transcripts remain untouched, pre-existing untracked files.

## 2026-09-01T15:16:41.408Z — Assistant — final response

M3 is complete and the real immutable bundle is published.

Implemented:

- [Target and 30-feature contract](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/multimodal/geo_helpfulness_targets_features.py)
- [M3 build/validate CLI](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/tools/run_multimodal_geo_helpfulness_m3.py)
- [M3 completion record](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/dev_plans/reliability_aware_geo_fusion/03_targets_and_features.md)
- [Published manifest](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/router/targets_and_feature_contract/manifest.json)

Real artifact verification:

| State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| `rescue` | 1,225 | 327 | 318 | 283 | 297 |
| `harm` | 2,256 | 560 | 549 | 572 | 575 |
| `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
| `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |

Also verified:

- 13,512 seed records, 3,378 images, 1,300 plots, four seeds.
- 989 images change target state across seeds.
- Leakage audit passes.
- Build → validate → rebuild produced `created`, `reused_valid`, `reused_valid`.
- No temperature fitting or `router_dataset.parquet`.
- M1/M2 fingerprints remain unchanged.
- Manifest self-hash: `a387bdbf6a988154e282e36f9de86a7ba2ade631057a83cadf477cedb6310a40`.

Testing:

- Focused M1–M3 suites: **131 passed**.
- Repository-wide: **208 passed, 1 unrelated failure**. The existing failure is an `ID` column expectation in [test_multimodal_geo_10m.py](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/tests/test_multimodal_geo_10m.py:188); M3 does not touch that pipeline.
- Compilation, links, whitespace, and diff-integrity checks pass.

The README now marks M3 complete and identifies M4 calibration and router training as the next action.

## 2026-09-02T07:51:22.289Z — User

Please repeat my last message and your response. 


## 2026-09-02T07:53:20.905Z — Assistant — final response

## Your last message

> PLEASE IMPLEMENT THIS PLAN:
>
> # M3 — Seed-Specific Targets and Deployment-Safe Feature Contract
>
> ## Summary
>
> Implement M3 as an additive, CPU-only child of immutable M1/M2. M3 will:
>
> - derive raw-fusion-relative targets for all 13,512 OOF seed records;
> - publish one combined target table keyed by `(row_uid, training_seed)`;
> - implement the single pure 30-feature builder shared by future router training and inference;
> - freeze the feature schema, prevalence report, leakage audit, and provenance;
> - leave all temperature calibration, learned transforms, the 727-column matrix, and `router_dataset.parquet` to M4.
>
> M3 is complete only when the real artifacts are built, sealed, revalidated, and documented. Code and synthetic tests alone are insufficient.
>
> ## Implementation Changes
>
> ### 1. Pure targets and feature API
>
> Add a stable module, `multimodal/geo_helpfulness_targets_features.py`, containing:
>
> - `derive_router_target_states(...)` for the frozen order:
>   `rescue`, `harm`, `both_correct`, `both_wrong`;
> - `build_router_target_table(...)`;
> - `derive_image_relative_states(...)`, usable only by diagnostic reporting;
> - `build_router_feature_frame(...)`;
> - feature-schema and leakage-audit validators;
> - `load_validated_m3_bundle(...)`, which M4 will use without modifying this module.
>
> The combined target table will contain exactly:
>
> ```text
> schema_version
> protocol_id
> row_uid
> plot_idx
> training_seed
> target_state
> ```
>
> It will:
>
> - have 13,512 rows and canonical order `(row_uid, training_seed)`;
> - join OOF predictions many-to-one with development-train labels by `row_uid`;
> - retain all four seed realizations without averaging, voting, or deduplication;
> - validate plot identity and OOF-fold agreement before omitting the fold from the published table;
> - serialize no labels, correctness flags, logits, probabilities, utility, or image-relative diagnostic states.
>
> Reuse `protocol_v1`; do not create `protocol_v2` or edit the frozen YAML. The M3 manifest will freeze the additive physical table schema and composite key. A fixed-seed projection remains uniquely keyed by `row_uid`.
>
> ### 2. Frozen semantic feature builder
>
> `build_router_feature_frame` will accept only three equal, non-empty `(n, 18)` probability matrices and an explicit:
>
> ```text
> probability_basis = "scalar_temperature_calibrated"
> ```
>
> It must reject M2’s `native_t1_uncalibrated` basis. It will fit no state and will never accept labels, targets, identities, folds, seeds, or correctness.
>
> Emit exactly the configured 30 semantic features in frozen family order:
>
> - 5 categorical;
> - 3 Boolean;
> - 2 integer;
> - 20 numeric.
>
> Freeze these conventions:
>
> - predictions and top-k ordering: descending probability, then ascending dense ID;
> - pair IDs: `image * 18 + geo` and `geo * 18 + raw`;
> - top-3 overlap: integer intersection cardinality `0..3`;
> - raw rank of geo prediction: one-based `1..18`;
> - entropy: unnormalised natural-log entropy;
> - JSD: natural-log, unnormalised Jensen–Shannon divergence;
> - TV: half-L1 distance;
> - zero-probability entropy/JSD terms contribute zero;
> - probability tolerance: absolute row-sum tolerance `1e-8`, with no clipping or renormalisation.
>
> Use exact dtypes: predictions `int8`, pair IDs `int16`, ranks/overlap `int8`, flags `bool`, numeric values `float64`. The schema records fixed vocabularies `0..17` and `0..323`, plus M4’s later expansion of 25 scaled values and 702 one-hot values into 727 `float64` columns.
>
> Do not reuse the existing agreement helpers without adapting their semantics: they currently normalise entropy/JSD and return fractional top-3 overlap.
>
> ### 3. M3 build, validation, and artifacts
>
> Add:
>
> ```bash
> python tools/run_multimodal_geo_helpfulness_m3.py build
> python tools/run_multimodal_geo_helpfulness_m3.py validate
> ```
>
> Both accept only the standard config, protocol-directory, and artifact-root overrides. There will be no temperature, validation-output, test, or force-overwrite arguments.
>
> Before writing:
>
> - run the frozen M1 preflight;
> - validate only the 16 train-OOF producer outputs and their manifests;
> - reconstruct their canonical concatenation and compare it with the sealed OOF aggregate;
> - verify physical/logical hashes, schema, seed/fold membership, M1 parent hashes, and current M2 implementation hashes;
> - load development-train labels through a role-filtered assignment projection;
> - never open development-validation outputs, final in-sample outputs, or locked-test sources.
>
> Publish an immutable bundle at:
>
> ```text
> router/targets_and_feature_contract/
> ├── router_targets.parquet
> ├── router_feature_schema.json
> ├── target_prevalence.json
> ├── feature_leakage_audit.json
> └── manifest.json
> ```
>
> Stage all files, publish the manifest last as the commit marker, and use exclusive no-replace semantics. A valid rerun returns `reused_valid`; committed corruption or stale lineage fails without overwrite. Only an owned, uncommitted partial publication may be cleaned and rebuilt.
>
> The manifest will bind:
>
> - protocol, assignments, ontology, resolved config, and allow-list hashes;
> - OOF aggregate file/content/manifest hashes and all 16 producer-manifest hashes;
> - M3 module, runner, schemas, and every child artifact hash;
> - the composite key and canonical order;
> - explicit declarations that no temperature was fitted, no calibrated feature rows were materialized, native-\(T=1\) was not used for primary features, and M4 owns `router_dataset.parquet`.
>
> ### 4. Reports and documentation
>
> `target_prevalence.json` will deterministically report:
>
> - pooled seed-realization counts and fractions;
> - separate counts for seeds 1–4;
> - complete habitat, plot, image–geo pair, and geo–raw pair breakdowns;
> - zero-support categories using `null` rates, never NaN/Infinity;
> - auxiliary image-versus-geo correctness states;
> - cross-seed stability by the number of distinct states per `row_uid`;
> - an explicit warning that 13,512 seed records represent 3,378 images, not independent biological samples.
>
> The leakage audit will verify the exact ordered allow-list, configured forbidden patterns, identity/group exclusions, calibrated-basis requirement, target/feature separation, and allowed input artifact roles.
>
> Documentation work:
>
> - update the governing README’s M3/M4 tasks, artifact tree, code layout, acceptance criteria, decision log, progress ledger, change log, and immediate next action;
> - add a dated downstream-handoff clarification to `02_oof_expert_outputs.md`;
> - create `03_targets_and_features.md` with commands, contracts, hashes, actual prevalence, tests, boundary evidence, and completion checklist;
> - after successful artifact execution, mark M3 complete and link its record; otherwise leave it in progress.
>
> Do not edit fingerprinted M1/M2 configuration, protocol, runner, model, OOF, or locked-evaluation code, `01_experimental_protocol.md`, or recovered chat transcripts.
>
> ## Test and Completion Plan
>
> Add focused pure and artifact tests covering:
>
> - all four primary target states and independence from `image_pred`;
> - seed-varying targets retained under the composite key;
> - duplicate/missing seeds, identities, folds, labels, predictions, mixed protocols, and validation rows rejected;
> - all 30 feature formulas, exact order, dtypes, vocabularies, and schema hash;
> - tie-breaking, zero probabilities, missing local categories, and malformed probability matrices;
> - regression anchors: uniform entropy `ln(18)`, disjoint one-hot JSD `ln(2)`, full top-3 overlap `3`, and disjoint TV `1`;
> - native-\(T=1\), forbidden aliases, unexpected columns, and identity/group metadata rejected;
> - train/inference calls producing identical features through the same stateless function;
> - prevalence reconciliation at every grouping level;
> - no validation/test reads and no calibration/router fitting;
> - staged publication, valid reuse, crash recovery, tamper detection, stale-parent rejection, and immutable M1/M2 hashes.
>
> Run the new suite together with the existing 84 M1/M2 geo tests, then the repository-wide test suite and documentation/link/whitespace checks.
>
> Finally build and validate the real bundle twice. Completion evidence must reproduce:
>
> | State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
> |---|---:|---:|---:|---:|---:|
> | `rescue` | 1,225 | 327 | 318 | 283 | 297 |
> | `harm` | 2,256 | 560 | 549 | 572 | 575 |
> | `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
> | `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |
>
> Also verify 3,378 unique images, 1,300 plots, four seeds, 989 images changing state across seeds, a passing leakage audit, and the absence of `router_dataset.parquet`. These values are verification evidence, not hardcoded target-generation inputs.
>
> ## Assumptions and Defaults
>
> - Publish only the combined target table; do not duplicate physical per-seed target files.
> - M3 introduces no dependency beyond the existing NumPy, pandas, PyArrow, and YAML stack.
> - M4 fits the 12 scalar temperatures, calls the frozen M3 builder, fits scaling/one-hot state, materializes `router_dataset.parquet`, and trains router candidates.
> - Any change to the 30-feature contract, target definition, seed handling, or calibration boundary requires an explicit later protocol decision rather than silently rebuilding M3.

## My response

> M3 is complete and the real immutable bundle is published.
>
> Implemented:
>
> - [Target and 30-feature contract](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/multimodal/geo_helpfulness_targets_features.py)
> - [M3 build/validate CLI](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/tools/run_multimodal_geo_helpfulness_m3.py)
> - [M3 completion record](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/dev_plans/reliability_aware_geo_fusion/03_targets_and_features.md)
> - [Published manifest](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/router/targets_and_feature_contract/manifest.json)
>
> Real artifact verification:
>
> | State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
> |---|---:|---:|---:|---:|---:|
> | `rescue` | 1,225 | 327 | 318 | 283 | 297 |
> | `harm` | 2,256 | 560 | 549 | 572 | 575 |
> | `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
> | `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |
>
> Also verified:
>
> - 13,512 seed records, 3,378 images, 1,300 plots, four seeds.
> - 989 images change target state across seeds.
> - Leakage audit passes.
> - Build → validate → rebuild produced `created`, `reused_valid`, `reused_valid`.
> - No temperature fitting or `router_dataset.parquet`.
> - M1/M2 fingerprints remain unchanged.
> - Manifest self-hash: `a387bdbf6a988154e282e36f9de86a7ba2ade631057a83cadf477cedb6310a40`.
>
> Testing:
>
> - Focused M1–M3 suites: **131 passed**.
> - Repository-wide: **208 passed, 1 unrelated failure**. The existing failure is an `ID` column expectation in [test_multimodal_geo_10m.py](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/tests/test_multimodal_geo_10m.py:188); M3 does not touch that pipeline.
> - Compilation, links, whitespace, and diff-integrity checks pass.
>
> The README now marks M3 complete and identifies M4 calibration and router training as the next action.

## 2026-09-02T07:56:30.135Z — User

repeat your last response


## 2026-09-02T07:58:20.451Z — Assistant — final response

## Your last message

> PLEASE IMPLEMENT THIS PLAN:
>
> # M3 — Seed-Specific Targets and Deployment-Safe Feature Contract
>
> ## Summary
>
> Implement M3 as an additive, CPU-only child of immutable M1/M2. M3 will:
>
> - derive raw-fusion-relative targets for all 13,512 OOF seed records;
> - publish one combined target table keyed by `(row_uid, training_seed)`;
> - implement the single pure 30-feature builder shared by future router training and inference;
> - freeze the feature schema, prevalence report, leakage audit, and provenance;
> - leave all temperature calibration, learned transforms, the 727-column matrix, and `router_dataset.parquet` to M4.
>
> M3 is complete only when the real artifacts are built, sealed, revalidated, and documented. Code and synthetic tests alone are insufficient.
>
> ## Implementation Changes
>
> ### 1. Pure targets and feature API
>
> Add a stable module, `multimodal/geo_helpfulness_targets_features.py`, containing:
>
> - `derive_router_target_states(...)` for the frozen order:
>   `rescue`, `harm`, `both_correct`, `both_wrong`;
> - `build_router_target_table(...)`;
> - `derive_image_relative_states(...)`, usable only by diagnostic reporting;
> - `build_router_feature_frame(...)`;
> - feature-schema and leakage-audit validators;
> - `load_validated_m3_bundle(...)`, which M4 will use without modifying this module.
>
> The combined target table will contain exactly:
>
> ```text
> schema_version
> protocol_id
> row_uid
> plot_idx
> training_seed
> target_state
> ```
>
> It will:
>
> - have 13,512 rows and canonical order `(row_uid, training_seed)`;
> - join OOF predictions many-to-one with development-train labels by `row_uid`;
> - retain all four seed realizations without averaging, voting, or deduplication;
> - validate plot identity and OOF-fold agreement before omitting the fold from the published table;
> - serialize no labels, correctness flags, logits, probabilities, utility, or image-relative diagnostic states.
>
> Reuse `protocol_v1`; do not create `protocol_v2` or edit the frozen YAML. The M3 manifest will freeze the additive physical table schema and composite key. A fixed-seed projection remains uniquely keyed by `row_uid`.
>
> ### 2. Frozen semantic feature builder
>
> `build_router_feature_frame` will accept only three equal, non-empty `(n, 18)` probability matrices and an explicit:
>
> ```text
> probability_basis = "scalar_temperature_calibrated"
> ```
>
> It must reject M2’s `native_t1_uncalibrated` basis. It will fit no state and will never accept labels, targets, identities, folds, seeds, or correctness.
>
> Emit exactly the configured 30 semantic features in frozen family order:
>
> - 5 categorical;
> - 3 Boolean;
> - 2 integer;
> - 20 numeric.
>
> Freeze these conventions:
>
> - predictions and top-k ordering: descending probability, then ascending dense ID;
> - pair IDs: `image * 18 + geo` and `geo * 18 + raw`;
> - top-3 overlap: integer intersection cardinality `0..3`;
> - raw rank of geo prediction: one-based `1..18`;
> - entropy: unnormalised natural-log entropy;
> - JSD: natural-log, unnormalised Jensen–Shannon divergence;
> - TV: half-L1 distance;
> - zero-probability entropy/JSD terms contribute zero;
> - probability tolerance: absolute row-sum tolerance `1e-8`, with no clipping or renormalisation.
>
> Use exact dtypes: predictions `int8`, pair IDs `int16`, ranks/overlap `int8`, flags `bool`, numeric values `float64`. The schema records fixed vocabularies `0..17` and `0..323`, plus M4’s later expansion of 25 scaled values and 702 one-hot values into 727 `float64` columns.
>
> Do not reuse the existing agreement helpers without adapting their semantics: they currently normalise entropy/JSD and return fractional top-3 overlap.
>
> ### 3. M3 build, validation, and artifacts
>
> Add:
>
> ```bash
> python tools/run_multimodal_geo_helpfulness_m3.py build
> python tools/run_multimodal_geo_helpfulness_m3.py validate
> ```
>
> Both accept only the standard config, protocol-directory, and artifact-root overrides. There will be no temperature, validation-output, test, or force-overwrite arguments.
>
> Before writing:
>
> - run the frozen M1 preflight;
> - validate only the 16 train-OOF producer outputs and their manifests;
> - reconstruct their canonical concatenation and compare it with the sealed OOF aggregate;
> - verify physical/logical hashes, schema, seed/fold membership, M1 parent hashes, and current M2 implementation hashes;
> - load development-train labels through a role-filtered assignment projection;
> - never open development-validation outputs, final in-sample outputs, or locked-test sources.
>
> Publish an immutable bundle at:
>
> ```text
> router/targets_and_feature_contract/
> ├── router_targets.parquet
> ├── router_feature_schema.json
> ├── target_prevalence.json
> ├── feature_leakage_audit.json
> └── manifest.json
> ```
>
> Stage all files, publish the manifest last as the commit marker, and use exclusive no-replace semantics. A valid rerun returns `reused_valid`; committed corruption or stale lineage fails without overwrite. Only an owned, uncommitted partial publication may be cleaned and rebuilt.
>
> The manifest will bind:
>
> - protocol, assignments, ontology, resolved config, and allow-list hashes;
> - OOF aggregate file/content/manifest hashes and all 16 producer-manifest hashes;
> - M3 module, runner, schemas, and every child artifact hash;
> - the composite key and canonical order;
> - explicit declarations that no temperature was fitted, no calibrated feature rows were materialized, native-\(T=1\) was not used for primary features, and M4 owns `router_dataset.parquet`.
>
> ### 4. Reports and documentation
>
> `target_prevalence.json` will deterministically report:
>
> - pooled seed-realization counts and fractions;
> - separate counts for seeds 1–4;
> - complete habitat, plot, image–geo pair, and geo–raw pair breakdowns;
> - zero-support categories using `null` rates, never NaN/Infinity;
> - auxiliary image-versus-geo correctness states;
> - cross-seed stability by the number of distinct states per `row_uid`;
> - an explicit warning that 13,512 seed records represent 3,378 images, not independent biological samples.
>
> The leakage audit will verify the exact ordered allow-list, configured forbidden patterns, identity/group exclusions, calibrated-basis requirement, target/feature separation, and allowed input artifact roles.
>
> Documentation work:
>
> - update the governing README’s M3/M4 tasks, artifact tree, code layout, acceptance criteria, decision log, progress ledger, change log, and immediate next action;
> - add a dated downstream-handoff clarification to `02_oof_expert_outputs.md`;
> - create `03_targets_and_features.md` with commands, contracts, hashes, actual prevalence, tests, boundary evidence, and completion checklist;
> - after successful artifact execution, mark M3 complete and link its record; otherwise leave it in progress.
>
> Do not edit fingerprinted M1/M2 configuration, protocol, runner, model, OOF, or locked-evaluation code, `01_experimental_protocol.md`, or recovered chat transcripts.
>
> ## Test and Completion Plan
>
> Add focused pure and artifact tests covering:
>
> - all four primary target states and independence from `image_pred`;
> - seed-varying targets retained under the composite key;
> - duplicate/missing seeds, identities, folds, labels, predictions, mixed protocols, and validation rows rejected;
> - all 30 feature formulas, exact order, dtypes, vocabularies, and schema hash;
> - tie-breaking, zero probabilities, missing local categories, and malformed probability matrices;
> - regression anchors: uniform entropy `ln(18)`, disjoint one-hot JSD `ln(2)`, full top-3 overlap `3`, and disjoint TV `1`;
> - native-\(T=1\), forbidden aliases, unexpected columns, and identity/group metadata rejected;
> - train/inference calls producing identical features through the same stateless function;
> - prevalence reconciliation at every grouping level;
> - no validation/test reads and no calibration/router fitting;
> - staged publication, valid reuse, crash recovery, tamper detection, stale-parent rejection, and immutable M1/M2 hashes.
>
> Run the new suite together with the existing 84 M1/M2 geo tests, then the repository-wide test suite and documentation/link/whitespace checks.
>
> Finally build and validate the real bundle twice. Completion evidence must reproduce:
>
> | State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
> |---|---:|---:|---:|---:|---:|
> | `rescue` | 1,225 | 327 | 318 | 283 | 297 |
> | `harm` | 2,256 | 560 | 549 | 572 | 575 |
> | `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
> | `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |
>
> Also verify 3,378 unique images, 1,300 plots, four seeds, 989 images changing state across seeds, a passing leakage audit, and the absence of `router_dataset.parquet`. These values are verification evidence, not hardcoded target-generation inputs.
>
> ## Assumptions and Defaults
>
> - Publish only the combined target table; do not duplicate physical per-seed target files.
> - M3 introduces no dependency beyond the existing NumPy, pandas, PyArrow, and YAML stack.
> - M4 fits the 12 scalar temperatures, calls the frozen M3 builder, fits scaling/one-hot state, materializes `router_dataset.parquet`, and trains router candidates.
> - Any change to the 30-feature contract, target definition, seed handling, or calibration boundary requires an explicit later protocol decision rather than silently rebuilding M3.

## My response

> M3 is complete and the real immutable bundle is published.
>
> Implemented:
>
> - [Target and 30-feature contract](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/multimodal/geo_helpfulness_targets_features.py)
> - [M3 build/validate CLI](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/tools/run_multimodal_geo_helpfulness_m3.py)
> - [M3 completion record](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/dev_plans/reliability_aware_geo_fusion/03_targets_and_features.md)
> - [Published manifest](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/router/targets_and_feature_contract/manifest.json)
>
> Real artifact verification:
>
> | State | Total | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
> |---|---:|---:|---:|---:|---:|
> | `rescue` | 1,225 | 327 | 318 | 283 | 297 |
> | `harm` | 2,256 | 560 | 549 | 572 | 575 |
> | `both_correct` | 7,171 | 1,800 | 1,786 | 1,789 | 1,796 |
> | `both_wrong` | 2,860 | 691 | 725 | 734 | 710 |
>
> Also verified:
>
> - 13,512 seed records, 3,378 images, 1,300 plots, four seeds.
> - 989 images change target state across seeds.
> - Leakage audit passes.
> - Build → validate → rebuild produced `created`, `reused_valid`, `reused_valid`.
> - No temperature fitting or `router_dataset.parquet`.
> - M1/M2 fingerprints remain unchanged.
> - Manifest self-hash: `a387bdbf6a988154e282e36f9de86a7ba2ade631057a83cadf477cedb6310a40`.
>
> Testing:
>
> - Focused M1–M3 suites: **131 passed**.
> - Repository-wide: **208 passed, 1 unrelated failure**. The existing failure is an `ID` column expectation in [test_multimodal_geo_10m.py](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/tests/test_multimodal_geo_10m.py:188); M3 does not touch that pipeline.
> - Compilation, links, whitespace, and diff-integrity checks pass.
>
> The README now marks M3 complete and identifies M4 calibration and router training as the next action.

## 2026-09-02T08:19:20.170Z — User


# Selected text:

## Selection 1: /home/hshi/Documents/researchproject/aihab/repo/aihab-clip/dev_plans/reliability_aware_geo_fusion/03_targets_and_features.md (line 137)
router_targets.parquet

## My request:
Please show me one data sample in `router_targets.parquet`?


## 2026-09-02T08:19:26.766Z — Assistant — progress

I’ll read the published Parquet directly and show its first row in canonical `(row_uid, training_seed)` order.

## 2026-09-02T08:19:47.125Z — Assistant — final response

Here is the first row in canonical order:

```json
{
  "schema_version": "geo_helpfulness.router_targets.v1",
  "protocol_id": "protocol_v1",
  "row_uid": "004aa228e278f0469fde3f556dfad12ae79d66d26113cbcd6b65af807c2d5ee4",
  "plot_idx": "721XX2",
  "training_seed": 1,
  "target_state": "both_correct"
}
```

`both_correct` means that, for this image under training seed 1, both the geo-only and raw-fusion predictions matched the true label. The file contains no labels, probabilities, logits, or correctness flags—only the derived target state.

## 2026-09-02T08:21:47.042Z — User


# Response annotations:
Each item contains text selected from an earlier Codex response and may include a user comment. Treat items as Annotation 1, Annotation 2, and so on in array order. Use every selection as context and address every comment. For every annotation you address, include its inline directive `:codex-annotation{index="N"}`, where N is its one-based array position (for example, `:codex-annotation{index="1"}`). Do not use unstructured annotation labels.
<response-annotations>
[{"text":"\"row_uid\""}]
</response-annotations>

## My request:
How to identify the original file name?


## 2026-09-02T08:21:52.818Z — Assistant — progress

I’ll trace `row_uid` back through the frozen M1 assignment schema and show the exact lookup needed to recover the original filename.

## 2026-09-02T08:23:17.605Z — Assistant — final response

`row_uid` is a one-way SHA-256 identifier, so the filename cannot be decoded from it directly. :codex-annotation{index="1"} Use it to look up the corresponding row in M1’s authoritative [development_assignments.parquet](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/protocol/development_assignments.parquet).

For the example row:

```text
row_uid:
004aa228e278f0469fde3f556dfad12ae79d66d26113cbcd6b65af807c2d5ee4

file:
ATT1005_721XX2_photo1-20190820-091957.jpg

image_source:
./data/CS_Xplots_2019_2023_train
```

Therefore, the image is:

[ATT1005_721XX2_photo1-20190820-091957.jpg](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/data/CS_Xplots_2019_2023_train/ATT1005_721XX2_photo1-20190820-091957.jpg)

For one lookup:

```python
import pandas as pd

assignments = pd.read_parquet(
    "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/"
    "protocol_v1/protocol/development_assignments.parquet"
)

match = assignments.loc[
    assignments["row_uid"].eq(row_uid),
    ["file", "image_source", "plot_idx"],
]
```

To attach filenames to the entire target table:

```python
targets_with_files = targets.merge(
    assignments[["row_uid", "file", "image_source"]],
    on="row_uid",
    how="left",
    validate="many_to_one",
)
```

`training_seed` is not needed for this lookup: all four seed-specific target records for a given `row_uid` refer to the same original image.

## 2026-09-02T08:30:01.737Z — User


# Selected text:

## Selection 1: /home/hshi/Documents/researchproject/aihab/repo/aihab-clip/dev_plans/reliability_aware_geo_fusion/03_targets_and_features.md (lines 94-107)
`build_router_feature_frame` is the only primary semantic feature builder used
by future router training and deployment inference. It has no learned state and
accepts only three equal, non-empty `(n, 18)` probability matrices plus the
explicit basis:

```text
scalar_temperature_calibrated
```

It rejects M2's `native_t1_uncalibrated` basis. M4 must fit the 12 scalar
temperatures from authoritative OOF logits and supply the resulting calibrated
probabilities. M3 does not create a partial native-probability feature table.

The ordered 30 semantic features are:

## My request:
What are the outputs from the `build_router_feature_frame` in M3?


## 2026-09-02T08:30:05.087Z — Assistant — progress

I’ll inspect the frozen builder’s return type, exact 30-column order, and dtypes so the answer matches the implemented M3 contract.

## 2026-09-02T08:30:44.277Z — Assistant — final response

There are two distinct meanings of “output” here:

1. The function returns an in-memory `pandas.DataFrame` with shape `(n, 30)`.
2. M3 does not materialize real feature rows, because calibrated probabilities do not exist until M4. M3 publishes only the frozen [feature schema](/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/router/targets_and_feature_contract/router_feature_schema.json).

Given three calibrated `(n, 18)` probability matrices:

```python
features = build_router_feature_frame(
    image_probabilities,
    geo_probabilities,
    raw_probabilities,
    probability_basis="scalar_temperature_calibrated",
)
```

the returned frame has one row per input example and exactly these columns:

| Family | Output columns | Meaning |
|---|---|---|
| Categorical | `image_pred`, `geo_pred`, `raw_pred` | Predicted dense class IDs, `0..17` |
| Categorical | `image_geo_pred_pair` | `image_pred * 18 + geo_pred`, `0..323` |
| Categorical | `geo_raw_pred_pair` | `geo_pred * 18 + raw_pred`, `0..323` |
| Boolean | `image_geo_agree`, `image_raw_agree`, `geo_raw_agree` | Whether each pair predicts the same class |
| Integer | `image_geo_top3_overlap` | Number of shared image/geo top-3 classes, `0..3` |
| Integer | `raw_rank_of_geo_pred` | Rank of the geo-predicted class under raw fusion, `1..18` |
| Numeric | `image_confidence`, `geo_confidence`, `raw_confidence` | Maximum probability from each expert |
| Numeric | `image_entropy`, `geo_entropy`, `raw_entropy` | Natural-log entropy of each distribution |
| Numeric | `image_top2_margin`, `geo_top2_margin`, `raw_top2_margin` | Top-1 probability minus top-2 probability |
| Numeric | `geo_minus_image_confidence`, `geo_minus_raw_confidence` | Signed confidence differences |
| Numeric | `geo_minus_image_entropy`, `geo_minus_raw_entropy` | Signed entropy differences |
| Numeric | `geo_minus_image_margin`, `geo_minus_raw_margin` | Signed top-2-margin differences |
| Numeric | `image_geo_jsd` | Jensen–Shannon divergence between image and geo |
| Numeric | `image_geo_total_variation` | Half-L1 distance between image and geo |
| Numeric | `image_probability_at_geo_pred` | Image probability assigned to geo’s predicted class |
| Numeric | `geo_probability_at_image_pred` | Geo probability assigned to image’s predicted class |
| Numeric | `raw_probability_at_geo_pred` | Raw-fusion probability assigned to geo’s predicted class |

The dtypes are:

- Predicted class IDs: `int8`
- Pair IDs: `int16`
- Agreement flags: `bool`
- Top-3 overlap and rank: `int8`
- All 20 numeric features: `float64`

The returned frame deliberately contains no `row_uid`, filename, seed, fold, label, correctness flag, or target. Those are managed separately to prevent leakage.

Therefore, M3’s actual artifacts contain the feature contract but no populated feature table. In M4, after fitting the 12 temperatures, the calibrated probabilities will be passed through this function. M4 will then transform these 30 semantic columns into the final 727-column router matrix and materialize `router_dataset.parquet`.

## 2026-09-02T08:35:23.548Z — User


# Response annotations:
Each item contains text selected from an earlier Codex response and may include a user comment. Treat items as Annotation 1, Annotation 2, and so on in array order. Use every selection as context and address every comment. For every annotation you address, include its inline directive `:codex-annotation{index="N"}`, where N is its one-based array position (for example, `:codex-annotation{index="1"}`). Do not use unstructured annotation labels.
<response-annotations>
[{"text":"image_pred * 18 + geo_pred"}]
</response-annotations>

## My request:
Can you explain this feature with an example?


## 2026-09-02T08:35:41.870Z — Assistant — final response

`image_geo_pred_pair` combines two class predictions into one unique categorical ID. :codex-annotation{index="1"}

Suppose:

```text
image_pred = 2
geo_pred   = 7
```

Then:

```text
image_geo_pred_pair = 2 × 18 + 7 = 43
```

This works like a two-digit number in base 18:

- `image_pred` selects one of 18 blocks.
- `geo_pred` selects one of 18 positions inside that block.
- All \(18 \times 18 = 324\) ordered prediction pairs receive unique IDs from `0` to `323`.

The pair can be decoded using:

```python
image_pred, geo_pred = divmod(43, 18)
# (2, 7)
```

Order matters:

```text
(image=2, geo=7) → 43
(image=7, geo=2) → 128
```

The value `43` is only a category identifier—it does not imply a meaningful numerical magnitude. M4 one-hot encodes all 324 possible pair IDs, allowing the router to learn patterns such as “when the image predicts class 2 and geo predicts class 7, geo tends to rescue or harm raw fusion.”

# Geo-helpfulness project structure and naming

Recorded 2026-10-01. This maintenance review covers the complete M1–M4
workflow and its later router, ConfidNet, joint-TCP and geometry studies. It
changes command discoverability and documentation, not scientific results.
The [project-wide catalog](../../configs/multimodal_geo_helpfulness_studies_v1.json)
is the machine-readable index of study records, entrypoints, implementation
modules, configurations, output directories and historical tool names.

## Naming contract

| Role | Convention | Example |
| --- | --- | --- |
| Experiment CLI | `tools/run_multimodal_geo_helpfulness_<topic>.py` | `run_multimodal_geo_helpfulness_geometry.py` |
| Review / verification / diagnostic CLI | `tools/{plot,check,diagnose}_multimodal_geo_helpfulness_<topic>.py` | `check_multimodal_geo_helpfulness_studies.py` |
| Implementation | `multimodal/geo_helpfulness_<topic>.py` | `geo_helpfulness_geometry.py` |
| Configuration or layout catalog | `configs/multimodal_geo_<topic>_<version>.json` or YAML | `multimodal_geo_helpfulness_studies_v1.json` |
| Tests | `tests/test_multimodal_geo_<topic>.py` | `test_multimodal_geo_helpfulness_entrypoints.py` |
| Development record | Numbered Markdown in this directory | `21_embedding_geometry_correction.md` |
| Generated outputs | `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/<versioned_study>/` | `joint_geometry_adapted_encoder_v1/` |

M1 retains the original unsuffixed `run_multimodal_geo_helpfulness.py` and
`multimodal_geo_helpfulness.yaml`. Later milestone runners are additive:
changing the original M1 runner would invalidate its protocol fingerprint.
Similarly, `_v1`, `_v2`, `_replay_v2`, `_review` and `_diagnostics` identify
actual implementation continuations, review products or historical diagnostics.
They should not be renamed merely to shorten a path. Study identifiers and
artifact-directory names are mapped explicitly where their historical names
differ. The artifact root already supplies the `geo_helpfulness` namespace.

## Public entrypoints across the studies

All runner filenames below are in `tools/`; all detailed versions, subordinate
M4 commands and source modules are enumerated in the project-wide catalog.

| Study family | Public entrypoint | Development record |
| --- | --- | --- |
| M1 protocol | `run_multimodal_geo_helpfulness.py` | [01](01_experimental_protocol.md) |
| M2 expert outputs | `run_multimodal_geo_helpfulness_m2.py` | [02](02_oof_expert_outputs.md) |
| M3 targets/features | `run_multimodal_geo_helpfulness_m3.py` | [03](03_targets_and_features.md) |
| M4 dataset, candidates and selection | `run_multimodal_geo_helpfulness_m4*.py` | [04](04_router_feasibility.md) |
| Legacy router transfer | `run_multimodal_geo_helpfulness_legacy_router.py` | [05](05_legacy_router_transfer.md) |
| Router capacity / MLP / replay | `run_multimodal_geo_helpfulness_router_capacity.py`, `…_router_mlp.py`, `…_router_mlp_replay.py` | [06](06_router_capacity_exploration.md), [07](07_router_mlp_exploration.md) |
| Legacy capacity transfer | `run_multimodal_geo_helpfulness_legacy_capacity.py` | [08](08_legacy_capacity_transfer.md) |
| Relaxed LR / RF / margin | `run_multimodal_geo_helpfulness_relaxed_guardrails.py`, `…_relaxed_rf.py`, `…_relaxed_margin.py` | [09](09_relaxed_guardrails_lr.md), [10](10_relaxed_guardrails_rf.md), [11](11_relaxed_margin_guardrail.md) |
| Initial and OOF ConfidNet | `run_multimodal_geo_helpfulness_confidnet.py`, `…_confidnet_oof.py`, `…_confidnet_oof_v2.py` | [12](12_confidnet_initial_experiment.md), [13](13_confidnet_oof_proposal.md), [14](14_confidnet_oof_experiment.md) |
| ConfidNet relaxed guardrails | `run_multimodal_geo_helpfulness_confidnet_relaxed_guardrails.py` | [15](15_confidnet_relaxed_guardrails.md) |
| Fold-4 ConfidNet and test | `run_multimodal_geo_helpfulness_confidnet_fold4.py`, `…_confidnet_fold4_test.py` | [16](16_confidnet_fold4_experiment.md), [17](17_confidnet_fold4_cleaned_test.md) |
| Joint TCP and test | `run_multimodal_geo_helpfulness_joint_tcp_v1.py`, `…_joint_tcp_test.py` | [18](18_joint_tcp_router_exploration.md), [19](19_joint_tcp_router_cleaned_test.md) |
| Joint-TCP capacity | `run_multimodal_geo_helpfulness_joint_tcp_capacity_v1.py` | [20](20_joint_tcp_router_capacity.md) |
| Geometry | `run_multimodal_geo_helpfulness_geometry.py` | [21](21_embedding_geometry_correction.md) |

The new canonical geometry interface is a thin CLI backed by
[geo_helpfulness_geometry.py](../../multimodal/geo_helpfulness_geometry.py).
It delegates to the exact recorded scientific implementation in a separate
process, using the current Python interpreter and repository working directory.
It supplies the versioned output location explicitly and changes no distance,
policy, threshold, model, metric or bootstrap computation.

| Geometry command | Recorded implementation retained for provenance |
| --- | --- |
| `overview` | `plot_multimodal_embedding_geometry.py` |
| `disagreements` | `check_geometry_disagreements.py` |
| `validation` | `evaluate_joint_geometry_policy.py` |
| `cleaned-test` | `evaluate_joint_geometry_cleaned_test.py` |
| `full-reference` | `evaluate_joint_geometry_full_reference.py` |
| `k5` | `evaluate_joint_geometry_k5.py` |
| `k-sweep` | `evaluate_joint_geometry_k_sweep.py` |
| `thresholds` | `evaluate_joint_geometry_thresholds.py` |
| `extract-adapted` | `extract_adapted_geometry_embeddings.py` |
| `adapted` | `evaluate_joint_geometry_adapted.py` |
| `validate` | `check_multimodal_geo_helpfulness_geometry_layout.py` |

`overview` defaults to all 4,200 development images, matching the completed
study. `extract-adapted` translates the shared `--output-dir` option to the
historical extractor's `--output` option. Relative paths are anchored to the
repository, including when invoked from another working directory. Outputs
must resolve under a versioned geo-helpfulness artifact directory.

Examples for inspection and read-only verification:

```bash
python tools/run_multimodal_geo_helpfulness_geometry.py list
python tools/run_multimodal_geo_helpfulness_geometry.py thresholds --dry-run
python tools/run_multimodal_geo_helpfulness_geometry.py adapted --help
python tools/run_multimodal_geo_helpfulness_geometry.py validate
python tools/check_multimodal_geo_helpfulness_studies.py
```

`--dry-run` prints the exact child argument list without importing the model
backend or running it. The first three generators (`overview`, `disagreements`,
`validation`) reject occupied output directories; they do not have a saved-run
reuse branch. Use `validate` to check their existing evidence. The six later
receipt-based evaluators accept their completed destinations and verify saved
results without rescoring. Naming cleanup does not invoke new study execution.

## Other historical naming exceptions

Three additional public adapters use
[geo_helpfulness_compat.py](../../multimodal/geo_helpfulness_compat.py):

| Canonical name | Historical implementation | Role |
| --- | --- | --- |
| `run_multimodal_geo_helpfulness_confidnet_relaxed_guardrails.py` | `analyze_confidnet_relaxed_guardrails.py` | Same `build` / `validate` study commands |
| `plot_multimodal_geo_helpfulness_joint_tcp_review.py` | `plot_joint_tcp_router_review.py` | Render saved aggregate metrics; accepts `--report` and `--output-dir` |
| `diagnose_multimodal_geo_helpfulness_confidnet_oof_parity.py` | `diagnose_multimodal_geo_confidnet_oof_parity.py` | Historical recovery only; requires explicit `run` |

The parity diagnostic is not routine validation: its original implementation
has no argument parser and can fit a recovery head. The adapter handles help,
dry-run and invalid arguments before delegation. The catalog marks it as
historical recovery rather than a new experimental result.

The nine geometry reporting/selection helpers also remain at their recorded
locations, classified as internal helpers in the catalog. Existing notebooks,
source hashes, imports and reproducibility checks can continue to use these
paths. New public commands use the consistent namespace. No scientific source
is copied into a second editable implementation, and no saved fingerprint is
rewritten to conceal a rename.

## Output and documentation contract

For execution and result reproduction, use the
[consolidated geometry guide](21_embedding_geometry_correction.md#reproducing-the-geometry-experiments-and-results).
It distinguishes artifact validation, fresh development analysis, frozen-checkpoint
extraction and numerical metric checks; it explains why a new output directory
alone is not an isolated cleaned-test rerun.

[Record 21](21_embedding_geometry_correction.md) records the earlier move of
nine geometry result bundles into the versioned artifact root. The directory
aliases there preserve original absolute references and report-relative links.
All earlier study roots, embeddings, checkpoints and append-only test-registry
events retain their identities. Internal bundle schemas remain versioned:
M1–M4 stages, separate `selection`/`predictions`/`scores` stages, and flat sealed
geometry bundles are indexed without rewriting their manifests.

Use this catalog and the [README](README.md) as the current navigation layer.
Numbered earlier records retain the exact commands and filenames used at the
time; an old recorded command is historical evidence, not an unexplained new
preferred naming convention. A new study should add its canonical CLI, modules,
configuration, record and versioned output directory to the catalog. The
read-only structure checker detects missing coverage, naming mismatches,
missing files and broken local links in numbered records.

The prior referenced M1/M2/M3 chat views expose incomplete history. This audit
therefore also used their [M1](M1_CHAT_RECOVERED.md),
[M2](M2_CHAT_RECOVERED.md) and [M3](M3_CHAT_RECOVERED.md) recovered records,
alongside the implemented milestone and ConfidNet records.


## Verification evidence

The project-wide audit passes for **40 catalog entries** (including historical
versions, a proposal and maintenance), **40 real artifact directories**,
**34 public entrypoints**, **90 implementation modules**, **21 configurations**,
and **25 numbered development records**. All numbered-record local links
resolve. The catalog also explains 22 legacy executable/helper names.

The command-dispatch suite passes **24 tests**, including no execution during
help/dry-run, explicit invocation of the historical recovery diagnostic,
rejection of ambiguous output options, output path validation, operation from
another working directory and propagation of child return codes. Actual
read-only validation through the canonical geometry and ConfidNet entrypoints
also passes. No model, threshold or diagnostic recovery is fitted in these
checks.

The preservation audit confirms unchanged bytes, sizes and modification times
for **257 pre-existing code/configuration/test files**, and all **43 test-registry
files** remain unchanged. The geometry validator additionally checks all
214 relocated result files and their source bindings. No original scientific
implementation, result bundle or saved registry event is renamed or rewritten
by this naming update.

Evidence: [structure audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/project_structure_v1/catalog_audit.json),
[preservation audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/project_structure_v1/preservation_audit.json),
[geometry validation](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/project_structure_v1/geometry_validation.json),
[ConfidNet validation](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/project_structure_v1/confidnet_validation.json),
and [command tests](../../tests/test_multimodal_geo_helpfulness_entrypoints.py).

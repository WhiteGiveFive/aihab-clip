# Embedding geometry for selective geo correction

Nine studies completed on 2026-09-29–30; output organization recorded on
2026-10-01. This record follows the learned-router studies in
[report 20](20_joint_tcp_router_capacity.md). It covers descriptive geometry,
fixed correction rules, validation-selected thresholds, and a matched change
from frozen pretrained to habitat-adapted image geometry.

**Geometry contains useful correction information, but its usefulness depends
on the representation and evaluation setting.** Pretrained geometry improves
development-validation accuracy while several corresponding test rules lose
accuracy. Adapted geometry reverses that pattern: it weakens validation
discrimination but improves the observed cleaned-test results. Its selected
test policies reach 75.45–75.71% accuracy, versus 74.31% for fusion and 75.17%
for the historical RF. Their paired accuracy intervals against RF include zero,
and their macro F1 remains below both baselines. No configuration is promoted
as a confirmed winner.

## Shared protocol and distinctions

For fusion's predicted class \(c_r\) and geo's predicted class \(c_g\), each
modality supplies its own candidate comparison:

\[
A_m(x)=\log\frac{d_m(c_r)+10^{-8}}{d_m(c_g)+10^{-8}},
\qquad S(x)=\min(A_{\mathrm{image}},A_{\mathrm{geo}}).
\]

Positive values favor **geo's candidate in either space**. Image geometry is
not expected automatically to favor the image or fusion expert. Distances use
separately L2-normalized image and 100 m geo vectors and cosine distance in
their original dimensions. They do not use the two-dimensional projections,
coordinate-standardized geo features, or concatenated-space distances.

A reference plot contributes its closest photo; a class distance averages the
nearest K **distinct plots**. On eligible fusion/geo disagreements, the fixed
rule switches to geo when both scores are positive, equivalently S>0.
Threshold studies use S>τ. Agreement, score ties, and unavailable geometry
retain fusion. Classification metrics include all queries and fallback cases.

| Evaluation | Query images / plots | Reference images / plots | Classification experts |
| --- | ---: | ---: | --- |
| Development validation | 822 / 325 | 3,378 / 1,300 | Fitted on 3,378 development-training images |
| Cleaned test, original bank | 1,347 / 531 | 3,378 / 1,300 | Historical final-epoch-50 heads fitted on all 4,200 development images |
| Cleaned test, expanded bank | 1,347 / 531 | 4,200 / 1,625 | The same historical heads |

All policy comparisons use four expert seeds. The historical test family and
full-727-feature RF policy match the
[September 15 report](reports/2026-09-15_geo_helpfulness_router/report.md);
RF uses `relaxed__rf__primary_full727__plots_20__harm_0p7`.
Validation-to-test differences therefore include a change in fitted experts.
The image encoder used for geometry is a separate choice from those fixed
classification experts.

Only the first disagreement diagnostic reuses archived training OOF
predictions, excluding the held-out fold from each reference bank. **No OOF
fits were repeated, and the adapted-encoder study performs no OOF analysis.**
Development validation and cleaned test have been examined previously. This
sequence is adaptive exploration, not a new independent confirmation.

## Completed development sequence

The linked reports retain the complete methods, figures, controls and results.
Accuracy below is the mean across four seeds; changes are percentage points.

| Step | Completed study | Main result or protocol change |
| ---: | --- | --- |
| 1 | [Embedding overview](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/embedding_geometry_v1/report.md) | PCA and t-SNE cover all 4,200 development images. Native geo supplies 53.49% of average squared concatenated distance; coordinate standardization raises this to 99.58%. These are input-distance contributions, not classifier reliance or habitat separation. |
| 2 | [Disagreement diagnostic](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_disagreement_v1/report.md) | Nearest-plot validation AUROC for rescue versus harm is 0.741 for image and 0.617 for geo; archived training OOF gives 0.627 and 0.548. Five-plot distances improve validation AUROCs to 0.772 and 0.645 on matched supported cases. |
| 3 | [Fixed K1 validation rule](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_validation_v1/report.md) | Joint positive support raises validation accuracy from 71.0462% to 72.6277%; the paired accuracy interval includes zero. No threshold is fitted. |
| 4 | [Fixed K1 cleaned-test rule](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_cleaned_test_v1/report.md) | With 3,378 references, accuracy falls from fusion's 74.3133% to 72.3645%. The difference is −1.9488 points, with conditional 95% interval [−3.8189, −0.1307]. |
| 5 | [Expanded reference bank](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_full_reference_cleaned_test_v1/report.md) | Adding all development-validation references, with the same K1 rule and experts, gives 71.8448%. The −0.5197-point change versus 3,378 references has an interval crossing zero. The prior bank result is retained. |
| 6 | [Strict K5 correction](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_k5_correction_v1/report.md) | If either class has fewer than five reference plots, keep fusion. Accuracy is 73.3577% on validation and 73.8122% / 73.7194% on test with 3,378 / 4,200 references. K1 with the same support requirement controls for the added abstention. |
| 7 | [Shared effective-K sweep](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_shared_k_sweep_v1/report.md) | For requested K=1/5/10/20, use min(K, class-reference counts) for both candidates and modalities. Validation accuracy is 72.6277/73.1752/72.9319/71.7457%; test with 4,200 references gives 71.8448/73.4224/73.9607/74.1277%. All K values are retained. |
| 8 | [Validation-selected thresholds](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_validation_thresholds_v1/report.md) | Return to strict K5/K10 and fit 18 guarded thresholds on validation. Primary 20-plot policies give test accuracy 73.979–74.647% with 3,378 references and 73.868–74.388% with 4,200; all remain below RF. |
| 9 | [Adapted image encoder](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_adapted_encoder_v1/report.md) | Replace image geometry while preserving experts, geo geometry and cohorts. Strict-zero K5/K10 accuracy falls on validation but rises on both test banks. Refit the same 18 scalar threshold policies separately for each representation. |

Strict K and shared effective K are different policies: insufficient support
causes abstention in steps 6, 8 and 9, but a smaller common neighborhood in
step 7. An apparent K effect can otherwise include a change in which rare-class
cases are corrected. Both-wrong disagreements are retained in intervention
summaries and excluded from rescue-versus-harm AUROC.

The threshold grid is K∈{5,10}, minimum acted plots∈{20,10,0}, and pooled
validation harm caps H≤αR with α∈{0.5,0.7,0.9}. Every feasible policy requires
at least 1% coverage and positive net rescue. Selection maximizes R−H, then
prefers fewer harms, fewer actions and a higher threshold. All 18 IDs remain
in the results, even when selections coincide. Relaxing the plot minimum does
not change selections in either encoder study. Thresholds are sealed before
test application and transferred unchanged to both banks; test constraints are
reported descriptively and are not guaranteed or re-enforced.

## Latest adapted-encoder evidence

Four seed-specific adapted towers were fitted only on the 3,378 training
images / 1,300 plots. The same tower embeds reference and query images within
a seed and serves both reference banks; bank expansion requires no additional
encoder fit. Validation and test query plots are disjoint from encoder-fitting
plots. The new extraction preserves the legacy BGR and forced-439-resize
convention. The historical pretrained cache lacks complete checkpoint and
extraction lineage, so this is not a perfectly isolated causal encoder change.

For validation disagreements where geo geometry favors geo, image AUROC falls
from 0.7986 to 0.7076 at K5 and from 0.7935 to 0.7126 at K10. At zero threshold,
validation accuracy falls by 1.0645 and 0.9732 points, respectively. Test
accuracy improves by 1.19–1.99 points across the same fixed-rule comparisons.
This split-dependent result does not establish that either representation is
uniformly better.

The table includes every distinct adapted primary threshold result. All nine
guard combinations within a given K select the same adapted threshold
(approximately 0.00610 for K5 and 0.01601 for K10).

| Cleaned-test method | Reference images | Accuracy (%) | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Fusion | — | 74.3133 | 0.73835 | 0.49013 |
| Historical full-feature RF correction | — | 75.1670 | 0.74651 | 0.50403 |
| Adapted, selected strict K5 | 3,378 | 75.4454 | 0.74646 | 0.48640 |
| Adapted, selected strict K10 | 3,378 | 75.6682 | 0.74844 | 0.48657 |
| Adapted, selected strict K5 | 4,200 | 75.7053 | 0.74904 | 0.48857 |
| Adapted, selected strict K10 | 4,200 | 75.5568 | 0.74742 | 0.48599 |

All four adapted-versus-RF accuracy intervals include zero. For example, the
4,200-reference K5 difference is +0.5382 points, with conditional 95% interval
[−0.5905, +1.6848]. Its macro F1 is still lower than fusion's and RF's.
The [paired intervals](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_adapted_encoder_v1/paired_intervals.csv)
and [encoder comparisons](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_adapted_encoder_v1/encoder_paired_intervals.csv)
retain all 408 accuracy/weighted-F1 intervals, including matched zero-threshold
and independently validation-selected encoder comparisons.

Policy uncertainty uses 2,000 habitat-stratified whole-plot bootstrap draws,
seed 20260930, retaining every photo and all four seed realizations together.
Metrics use the fixed 18-class ontology, with absent-class F1 set to zero.
Intervals condition on fitted experts, embeddings and selected policies;
thresholds are not refitted within each draw. They do not account for adaptive
test reuse, selection uncertainty or multiple comparisons. Pooled seed-image
counts are not independent photographs.

## Canonical artifact inventory

All nine complete bundles live under
[`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness`](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/).
The tracked [layout catalog](../../configs/multimodal_geo_geometry_layout_v1.json)
maps each study to its original dated path, canonical directory, report and
recorded producer. The [project-wide naming index](22_project_structure.md)
provides the preferred public commands. The table points to machine-readable provenance; each directory also
retains its original report, figures and detailed result tables.

| Canonical study directory | Provenance / selected machine evidence |
| --- | --- |
| `embedding_geometry_v1` | [Manifest](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/embedding_geometry_v1/manifest.json), [coordinates](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/embedding_geometry_v1/coordinates.parquet) |
| `geometry_disagreement_v1` | [Manifest](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_disagreement_v1/manifest.json), [row scores](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_disagreement_v1/geometry_scores.parquet) |
| `joint_geometry_validation_v1` | [Manifest](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_validation_v1/manifest.json), [metrics](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_validation_v1/metrics.json) |
| `joint_geometry_cleaned_test_v1` | [Receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_cleaned_test_v1/receipt.json) |
| `joint_geometry_full_reference_cleaned_test_v1` | [Receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_full_reference_cleaned_test_v1/receipt.json) |
| `joint_geometry_k5_correction_v1` | [Receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_k5_correction_v1/receipt.json) |
| `joint_geometry_shared_k_sweep_v1` | [Receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_shared_k_sweep_v1/receipt.json), [effective K usage](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_shared_k_sweep_v1/effective_k_usage.csv) |
| `joint_geometry_validation_thresholds_v1` | [Receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_validation_thresholds_v1/receipt.json), [frozen policies](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_validation_thresholds_v1/frozen_policies.json) |
| `joint_geometry_adapted_encoder_v1` | [Receipt](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_adapted_encoder_v1/receipt.json), [encoder performance](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_adapted_encoder_v1/encoder_performance_summary.csv) |

Adapted embedding arrays and their extraction manifest remain separately at
[`adapted_geometry_v1/embeddings`](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_geometry_v1/embeddings/).
They are shared inputs, not another result bundle.

## Reproducing the geometry experiments and results

The original three development reports supplied individual generation commands;
the later scored reports supplied methods and evidence but did not provide a
complete reproduction sequence. This section is the consolidated command guide
for all nine reports. It distinguishes checking existing artifacts from
recomputing results. The command interface currently depends on the archived
workspace inputs; it is not a standalone rebuild from a fresh repository clone.

### Environment and required inputs

Run from the repository root in the existing `habcls` environment:

```bash
conda activate habcls
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python tools/run_multimodal_geo_helpfulness_geometry.py list
```

The original overview manifest records Python 3.9.20, NumPy 2.0.2,
pandas 2.2.3, scikit-learn 1.5.2, matplotlib 3.9.2 and Plotly 6.5.2. Use the
existing environment for the additional PyArrow, PyTorch, OpenCLIP/timm,
OpenCV, safetensors and threadpoolctl dependencies. This recorded subset is
not a complete installable environment lockfile. Cached geometry and metric
analysis run on CPU; fresh image extraction also requires the model runtime
and was performed on CUDA.

Let `G` denote `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/`
in the table below; it is shorthand, not a shell variable.

| Input | Required for |
| --- | --- |
| `G/protocol_v1/protocol/`: assignments, protocol manifest, resolved configuration and locked-test snapshot reference | All studies; supplies the exact 3,378/822 split and identities |
| `multimodal_artifacts/joined_tables/cs/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_pretrained/gse_100m/seed1/{train,val,test}.parquet` | Frozen image/geo geometry; development commands use train/val, test commands additionally use the test cache |
| `G/protocol_v1/development_train_oof/` and `development_validation/` with their producer manifests and outputs | The historical disagreement diagnostic and fixed validation inputs; use archived outputs without refitting OOF experts |
| `G/relaxed_guardrails_rf_v1/{declaration.json,selection/,predictions/,scores/}` | Exact September 15 test experts, RF actions, labels and comparator provenance |
| Earlier geometry bundles and the compatibility directory links in the layout catalog | Fixed predecessor inputs for later studies |
| `multimodal_artifacts/locked_test_registry/cs/gse_100m_cleaned_test/` | Locked identity and append-only historical score events |
| `G/adapted_geometry_v1/embeddings/` | Saved adapted reference/query vectors for seeds 1–4 |
| Raw development/test photographs, `G/protocol_v1/development_validation/seed_N/adapted_visual_tower.safetensors`, and the matching `G/joint_tcp_router_v1/features/seed_N/` and `producers/seed_N/validation/` caches/manifests | Fresh adapted extraction and its validation replay checks |

The frozen configuration supplies the raw-image and model initialization paths.
The extractor also authenticates test identities against the prior threshold
study's `test_4200_predictions.parquet`. Pretrained image caches lack complete
original encoder-extraction lineage; obtaining a new SigLIP2 checkpoint alone
cannot reconstruct that missing history. Fresh adapted extraction must preserve
the recorded BGR → forced 439 resize → native evaluation transform recipe.

### Verify the saved experiments

These commands check the archived results and their provenance:

```bash
python tools/run_multimodal_geo_helpfulness_geometry.py validate
python tools/check_multimodal_geo_helpfulness_studies.py

python tools/run_multimodal_geo_helpfulness_geometry.py cleaned-test
python tools/run_multimodal_geo_helpfulness_geometry.py full-reference
python tools/run_multimodal_geo_helpfulness_geometry.py k5
python tools/run_multimodal_geo_helpfulness_geometry.py k-sweep
python tools/run_multimodal_geo_helpfulness_geometry.py thresholds
python tools/run_multimodal_geo_helpfulness_geometry.py adapted

python tools/run_multimodal_geo_helpfulness_geometry.py extract-adapted --device cpu
```

At the recorded, completed destinations, the six evaluation commands take
their receipt-validation branch; they do **not** recompute geometry, refit
thresholds or rescore predictions. The extraction command with the existing
manifest verifies the embedding bundle without inference. `--device cpu`
avoids requiring CUDA merely to validate that existing bundle. The general
`validate` command also authenticates the three development bundles without
invoking their generators. A successful hash check is evidence of preservation,
not an independent numerical reproduction.

### Recompute development outputs from the existing caches

The following examples require unused destination directories. They perform
actual computation and leave the original bundles intact:

```bash
python tools/run_multimodal_geo_helpfulness_geometry.py overview \
  --development-split all \
  --output-dir multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_reproduction_v1/overview

python tools/run_multimodal_geo_helpfulness_geometry.py validation \
  --output-dir multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_reproduction_v1/validation
```

The overview regenerates coordinates, neighborhood summaries and figures from
all 4,200 development images. The validation command recomputes the fixed K1
policy and metrics from the **archived disagreement scores**, not from the
new overview directory. Expect fusion accuracy 71.0462% and corrected accuracy
72.6277% on the same 822 queries and four seeds. Compare keyed predictions and
numeric tables; newly generated manifests, timing fields and figure metadata
need not have identical file hashes.

For completeness, the historical disagreement diagnostic can be reconstructed
with the following command. It includes analysis of archived OOF predictions;
it is **not needed for the adapted-encoder reproduction**, and no OOF diagnostic
was repeated during that study or this documentation update.

```bash
python tools/run_multimodal_geo_helpfulness_geometry.py disagreements \
  --output-dir multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_reproduction_v1/disagreements
```

A retained reproduction directory should be added to the project-wide catalog
as a reproduction artifact. These example commands have not been executed as
part of this documentation update.

### Re-extract adapted embeddings from the frozen checkpoints

Check the source identities and image files first; preflight does not run an
encoder or establish numerical replay equivalence:

```bash
python tools/run_multimodal_geo_helpfulness_geometry.py extract-adapted \
  --preflight-only
```

For an actual new extraction, use an unused destination:

```bash
python tools/run_multimodal_geo_helpfulness_geometry.py extract-adapted \
  --device cuda \
  --output-dir multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_geometry_reproduction_v1/embeddings
```

This reuses the four saved development-train-adapted encoders; it does not fit
an encoder or regenerate OOF experts. Expect four `seed_N/embeddings.npz`
files with 5,547 rows × 1,152 float32 dimensions: 4,200 development images and
1,347 cleaned-test queries. Each seed uses the same encoder for both query and
reference images. The extractor compares all 822 validation vectors against
the authenticated producer cache and records the replay diagnostics in its
manifests. The original extraction reproduced those validation vectors exactly.
A new extraction is not automatically consumed by later evaluations; the
`adapted` command's `--embedding-dir` chooses that input explicitly when a
separate evaluation is configured.

### Numerically reproduce the adapted headline results

After validating the saved bundles, this example recomputes accuracy, weighted
F1, macro F1, and rescue/harm counts directly from their saved row predictions.
It uses the minimum-20-plot / H≤0.7R policies, averages metrics over the four
seeds, and retains all 18 ontology classes for F1. It performs no selection and
writes no result or registry file.

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

root = Path('multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_geometry_adapted_encoder_v1')
for bank in (3378, 4200):
    rows = pd.read_parquet(root / f'test_{bank}_scored.parquet')
    assert len(rows) == 4 * 1347 and rows.training_seed.nunique() == 4
    for k in (5, 10):
        policy = f'k{k}__plots_20__harm_0p7'
        metrics = []
        for _, group in rows.groupby('training_seed'):
            y = group.label_id_dense
            prediction = group[f'{policy}_pred']
            metrics.append([
                (prediction == y).mean(),
                f1_score(y, prediction, labels=range(18), average='weighted', zero_division=0),
                f1_score(y, prediction, labels=range(18), average='macro', zero_division=0),
            ])
        accuracy, weighted, macro = np.mean(metrics, axis=0)
        acted = rows[f'{policy}_acted']
        raw_ok = rows.raw_pred.eq(rows.label_id_dense)
        geo_ok = rows.geo_pred.eq(rows.label_id_dense)
        rescue = int((acted & ~raw_ok & geo_ok).sum())
        harm = int((acted & raw_ok & ~geo_ok).sum())
        print(f'bank={bank} K={k}: accuracy={100*accuracy:.4f}% '
              f'weighted_F1={weighted:.6f} macro_F1={macro:.6f} R={rescue} H={harm}')
PY
```

Expected output:

```text
bank=3378 K=5: accuracy=75.4454% weighted_F1=0.746458 macro_F1=0.486400 R=152 H=91
bank=3378 K=10: accuracy=75.6682% weighted_F1=0.748437 macro_F1=0.486573 R=153 H=80
bank=4200 K=5: accuracy=75.7053% weighted_F1=0.749038 macro_F1=0.488566 R=156 H=81
bank=4200 K=10: accuracy=75.5568% weighted_F1=0.747421 macro_F1=0.485993 R=148 H=81
```

This example was executed on 2026-10-01 and reproduced all four lines above.
It verifies metric aggregation from saved decisions, not how those decisions
were produced. The row-level geometry and selected thresholds are in
`validation_scores.parquet`, `selection.json`, `frozen_policies.json` and the
bank-specific prediction tables. Adapted K5 and K10 select approximately
0.006095635 and 0.01600631; all nine guard combinations within each K coincide.
Preserve strict support, cosine distances over distinct reference plots,
strict-positive intervention (ties retain fusion), the 3,378/4,200 bank
distinction and the fixed expert family
when comparing any fresh reproduction. Bootstrap intervals require the same
2,000 habitat-stratified plot draws and seed 20260930; the full replicate tables
and paired intervals are retained in each scored bundle.

### Limits of a fresh cleaned-test rebuild

The current six cleaned-test backends have fixed scientific study IDs,
predecessor paths and registry locations. Merely changing `--output-dir` to a
new `_v2` directory does not create an independent study or redirect its inputs.
Reusing an identical prediction manifest at a different directory conflicts
with the existing registry reservation's owner path. If changed provenance
produces a new manifest hash, the command may instead append another test event.
Neither behavior is a general-purpose scratch-reproduction mode.

For example, `thresholds` consumes the archived K-sweep bundle, and `adapted`
consumes the archived pretrained-threshold study; generating new predecessor
folders does not change those bindings. A full clean rebuild therefore needs
a separately parameterized dependency/registry workflow or an appropriately
restored execution environment. That end-to-end isolated rebuild has **not**
been implemented or validated. The supported instructions above cover saved
artifact verification, cached development recomputation, frozen-checkpoint
extraction, and numerical metric reproduction, with their boundaries explicit.

## Provenance-preserving organization and commands

Organization moves each whole bundle without altering its contents or internal
layout. The original dated directories under `dev_plans/.../reports` remain
symlinks to the canonical bundles, preserving paths recorded in declarations,
source hashes and the cleaned-test registry. Date-named directory aliases at
the canonical artifact parent also preserve the original reports' relative
cross-study links, including the September 15 comparison. Reports are not
rewritten or duplicated for navigation.

The [migration manifest](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_layout_v1/migration_manifest.json)
records the layout and preserved hashes. This is an artifact relocation, not
an evaluation event: no embeddings, OOF predictions, thresholds, metrics or
bootstrap samples are recomputed, and historical registry events remain
unchanged. Use canonical versioned directories for new outputs and explicit
`--output-dir` arguments; historical script defaults are retained because
sealed receipts bind their code hashes.

From the repository root, verify the existing organization without training or
rescoring:

```bash
python tools/run_multimodal_geo_helpfulness_geometry.py validate
```

The [verifier](../../tools/check_multimodal_geo_helpfulness_geometry_layout.py)
checks the catalog, migration inventory, aliases, report links and preserved
provenance. The preferred interface is
[`run_multimodal_geo_helpfulness_geometry.py`](../../tools/run_multimodal_geo_helpfulness_geometry.py),
with subcommands listed in [record 22](22_project_structure.md). Use `list` to
see every study and `--dry-run` after a study command to inspect its invocation.
The producer entry points below retain the exact historical implementations;
they are not required for organization or routine review.

| Step | Producer |
| ---: | --- |
| 1 | [plot_multimodal_embedding_geometry.py](../../tools/plot_multimodal_embedding_geometry.py), with `--development-split all` |
| 2 | [check_geometry_disagreements.py](../../tools/check_geometry_disagreements.py) |
| 3 | [evaluate_joint_geometry_policy.py](../../tools/evaluate_joint_geometry_policy.py) |
| 4 | [evaluate_joint_geometry_cleaned_test.py](../../tools/evaluate_joint_geometry_cleaned_test.py) |
| 5 | [evaluate_joint_geometry_full_reference.py](../../tools/evaluate_joint_geometry_full_reference.py) |
| 6 | [evaluate_joint_geometry_k5.py](../../tools/evaluate_joint_geometry_k5.py) |
| 7 | [evaluate_joint_geometry_k_sweep.py](../../tools/evaluate_joint_geometry_k_sweep.py) |
| 8 | [evaluate_joint_geometry_thresholds.py](../../tools/evaluate_joint_geometry_thresholds.py) |
| 9 | [evaluate_joint_geometry_adapted.py](../../tools/evaluate_joint_geometry_adapted.py); embedding source: [extract_adapted_geometry_embeddings.py](../../tools/extract_adapted_geometry_embeddings.py) |

Each evaluation producer accepts `--output-dir`, but a new directory alone does
not isolate a cleaned-test rerun; see the reproduction limits above. Existing
sealed bundles remain the evidence for the completed studies above.

## Organization verification

The relocation and its independent audit passed on 2026-10-01:

- All **214 files / 103,503,311 bytes** in the nine bundles retain their original
  contents and file modification times; the independent audit also checks the
  original manifests, receipts and source/code bindings.
- All **43 cleaned-test registry files** remain unchanged, with no added event.
- All **19 directory aliases** resolve correctly. The nine reports' 104 local
  links work through both canonical and historical paths (208 link checks).
- All **six receipt-based evaluators** accept their canonical paths in the
  completed-run reuse branch without rescoring. The first three studies were
  checked directly from their manifests, without invoking their generators.
- The new verifier passes ten temporary-fixture checks, including detection of
  changed payloads, changed provenance, broken links and changed registry events.

Evidence: [layout audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_layout_v1/layout_audit.json),
[independent audit](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_layout_v1/independent_audit.json),
[completed-run reuse checks](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_layout_v1/completed_reuse_audit.json),
and [verifier fixture checks](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_layout_v1/verifier_fixture_audit.json).

The default verifier permits later append-only registry events while requiring
every historical entry to remain unchanged. Add `--strict-registry-snapshot`
to require the exact 43-file registry snapshot used for this migration audit.

# Router MLP exploration — development-only architecture round

Status: COMPLETE — all 48 declared fits, scoring, independent validation, and byte-preserving reuse are verified. This exploratory round does not support replacing the existing routers with these MLPs.

Study ID: `router_mlp_v1`. This additive exploration preserves M1–M4 and `router_capacity_v1` byte-for-byte. Original M4 remains no-go; no cleaned-test or legacy-transfer output is opened in this round. Development validation has already been reused, so selection results and uncertainty intervals are exploratory and selection-conditional, not independent confirmation.

## Frozen architecture and training declaration

Use all three existing feature variants (727 full, 79 no-pairs, 25 numeric) and the four hidden-width sequences `[32]`, `[64]`, `[32,16]`, `[64,32]`, in that order. Depth counts hidden layers. Every hidden linear layer is followed by ReLU and dropout 0.1. The final linear layer has four logits in order rescue, harm, both_correct, both_wrong. There is no batch normalization, residual connection, new embedding, class weighting, label smoothing, or new probability calibration.

The feature pipeline is unchanged: seed-specific frozen expert temperatures, 25 standardized numeric/Boolean/integer features, fixed categorical vocabularies, and exact existing variant projections. Each expert seed uses 3,378 OOF training images from 1,300 plots; validation has 822 images from 325 disjoint plots. Four seeds do not multiply the independent image/plot count.

| Hidden widths | Parameters, 25 inputs | Parameters, 79 inputs | Parameters, 727 inputs |
| --- | ---: | ---: | ---: |
| [32] | 964 | 2,692 | 23,428 |
| [64] | 1,924 | 5,380 | 46,852 |
| [32,16] | 1,428 | 3,156 | 23,892 |
| [64,32] | 3,876 | 7,332 | 48,804 |

The parameter count includes every bias. This is an architecture search, not a strictly parameter-matched depth experiment.

Fixed training settings: PyTorch 2.5.1+cu124, deterministic float64 CPU computation, one thread, 300 epochs, minibatch 128, AdamW learning rate 0.001, weight decay 0.001 on weights and zero on biases. Use the training seed for initialization, shuffle, and dropout. Train with equal-image mean multiclass cross-entropy. Use the final epoch 300 state; no validation-based checkpoint selection or early stopping is allowed. Record dropout-off training NLL/accuracy, optimization cross-entropy, and gradient-finiteness diagnostics at every epoch. AdamW decay is not misreported as part of cross-entropy loss.

AdamW uses betas (0.9, 0.999), epsilon 1e-8, and disables AMSGrad, maximize, foreach, capturable, differentiable, and fused options. Linear layers use the pinned PyTorch default initialization. A separate seed-matched shuffle generator keeps minibatch order identical across architectures; dropout and initialization also have their seeds recorded. Determinism is checked in the pinned CPU environment, not claimed across arbitrary hardware or PyTorch versions.

A synthetic 3378×727 benchmark with the largest architecture took 0.339 seconds for 10 epochs in float64, approximately 10 seconds per 300 epochs. Native PyTorch versus pure NumPy probabilities differed by at most 1.22e-15 in that benchmark. CPU is therefore sufficient; GPU was not visible to this execution environment and is not required. Device choice was made from synthetic timing, not validation outcomes.

The machine-readable declaration is `configs/multimodal_geo_router_mlp_v1.json`: 12 architecture/feature configurations and 48 seed fits. No search expansion, optimizer adjustment, or epoch change after validation scoring is part of this round.

## Selection and comparison

Seal all 48 states and their training/validation predictions before decoding development-validation targets. In a separate fitting-incapable scorer, choose one architecture per feature variant by mean four-seed validation NLL. Ties within 1e-12 follow declaration order. Apply the unchanged M4 common-threshold selector to `q_rescue-q_harm`, requiring raw/geo disagreement, strict threshold comparison, positive net utility, at least 1% coverage, at least 20 acted plots, and `2*harms<=rescues`.

Retain the nine selected LR/RF/XGB policies, the original M4 primary, and all five heuristics as frozen comparators. Report four-state accuracy/macro-F1/NLL/Brier and disagreement rescue/harm AP separately from routed habitat top1/top3 accuracy and intervention utility. Use 2,000 shared plot-bootstrap draws; paired MLP-minus-matching-LR/RF/XGB and MLP-minus-margin-heuristic contrasts condition on selected models/thresholds. They do not rerun selection or account for repeated validation reuse.

## Implementation and commands

Run from the repository root, using the existing `habcls` environment. No additional package installation is required:

```bash
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_router_mlp_replay.py build
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_router_mlp_replay.py validate
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_router_mlp_replay.py build
```

The first build trains missing models, seals all predictions, and then launches a separate scoring process. Validation reconstructs inputs, probabilities, final-epoch training metrics, curves, selection, and policy results without importing PyTorch or fitting libraries. A repeat build validates and reuses committed artifacts without changing their bytes. Separate stages are `fit`, `score`, `validate-predictions`, and `validate-scores`; path overrides are `--protocol-dir`, `--capacity-root`, `--source-dir`, `--output-dir`, and `--config`. There is no scientific override or force-overwrite option. `--source-dir` contains the original model/prediction bundle; `--output-dir` contains only the versioned replay declaration and scoring bundle described below.

Default source/prediction root (`--source-dir`): `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/router_mlp_v1`, outside both frozen predecessor directories. The default replay output root (`--output-dir`) is the sibling `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/router_mlp_v1_replay_v2`. The combined artifact contract is:

- `declaration.json`: configuration, CPU/environment identity, parent/code/dependency hashes and access boundaries, published before any real fit.
- `predictions/models/`: 48 self-hashed model records. Weights/biases are plain JSON arrays; each record also retains 300-epoch fitting diagnostics and native/Python replay checks. No pickle is required.
- `predictions/train/` and `predictions/validation/`: 48 tables each, totaling 162,144 training and 39,456 validation probability rows. Metadata is identity/configuration/family/variant; model outputs are four float64 probabilities and their rescue-minus-harm score. No validation labels are included.
- `predictions/validation_features.parquet`: the unchanged 727-feature validation matrix and identity columns.
- `predictions/training_curves.parquet`: 14,400 rows of epoch-wise optimization cross-entropy, dropout-off training NLL/accuracy, and gradient diagnostics.
- Scores are published separately under `router_mlp_v1_replay_v2/scores/`: all 12 configuration metrics, three selected frozen policies, 18-policy comparison, 12 paired contrasts, threshold/action tables, bootstrap/subgroup outputs, training diagnostics, access audit and selection receipt.
- Manifests written last in each stage. Publication uses an exclusive lock and no-overwrite writes. Valid owned model records can be reused after interruption; unknown or corrupt files are never deleted automatically.

Native and portable logits/probabilities use `atol=rtol=1e-12` with exact class-prediction agreement. Saved probability normalization tolerance remains 1e-8. The original NLL true-class probability floor is inherited unchanged from the predecessor metric implementation.

The documentation lookup informed deterministic RNG controls, separate optimizer parameter groups for biases, and dropout-off inference: [PyTorch 2.5.1 reproducibility](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst), [optimizer parameter groups](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/optim.rst), and [evaluation versus gradient modes](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/autograd.rst). Native parity and repeat-fit tests verify the behavior used here.

No production fits or validation scoring had occurred when the architecture/training settings above were declared.

### Immutable training-history replay correction

The initial runner completed all 48 fits in 279.14 seconds. Its separate scoring process stopped before decoding validation targets: the training-curve replay compared columns in an order determined by Python dictionary insertion order. In-memory fit histories and canonically sorted JSON histories have different key orders. Every value in all 14,400 curve rows was identical; all 48 model/prediction and final training-metric checks had already passed.

The original declaration, source code/configuration, all model records, all probability tables, and the curve table remain unchanged. The additive `geo_helpfulness_router_mlp_replay.py` compatibility module declares the two known nine-column layouts (fresh in-memory fitting and resumed JSON fitting), selects the matching reconstruction order, runs all original strict numerical/integrity checks, and publishes scores to the sibling `router_mlp_v1_replay_v2` directory. Unknown column layouts are rejected; numerical comparisons are not relaxed. Its separate declaration binds the original bundle, frozen ancestors, and recovery code before scoring. This is a serialization/replay correction, not a new architecture search or scientific-setting amendment. It performs no fitting. The wrapper above also supports fresh training through the unchanged original fitter before launching the corrected apply-only scoring stage.

The original source bundle has 148 files. Its pre-recovery bytes-plus-mtime fingerprint is `d17dd4a7d7ee7bca3706bb4154d334a3180dd16a468b05da7982008c50d6dca5`.

## Results

All values below use development validation only. Accuracies are mean percentages over four seeds; each seed has the same 822 images, so this also equals accuracy over the 3,288 seed-image rows. Action, rescue, and harm counts are sums over seeds, not counts of independent images. The validation set has 325 distinct plots, and all uncertainty resamples complete plots shared across seeds. Plot-level resampling does not itself establish statistical independence between plots.

### All twelve architecture/feature configurations

Training accuracy and NLL are dropout-off, final-epoch resubstitution diagnostics. Validation NLL is the predeclared architecture-selection criterion; the selected architecture is `[32]` within every feature variant.

| Features | Hidden widths | Training state accuracy (%) | Validation state accuracy (%) | Training NLL | Validation NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| Full 727 | [32] | 87.936649 | 68.278589 | 0.290795 | 1.092620 |
| Full 727 | [64] | 91.378034 | 68.430657 | 0.217613 | 1.299964 |
| Full 727 | [32,16] | 89.823860 | 68.156934 | 0.238871 | 1.499970 |
| Full 727 | [64,32] | 95.167259 | 69.008516 | 0.132536 | 2.051568 |
| No-pairs 79 | [32] | 85.109532 | 69.130170 | 0.357138 | 0.842646 |
| No-pairs 79 | [64] | 89.194790 | 68.734793 | 0.276112 | 0.953845 |
| No-pairs 79 | [32,16] | 87.625814 | 68.795620 | 0.300640 | 0.972977 |
| No-pairs 79 | [64,32] | 93.553878 | 67.761557 | 0.174364 | 1.399960 |
| Numeric 25 | [32] | 77.124038 | 71.502433 | 0.504449 | 0.646964 |
| Numeric 25 | [64] | 79.344287 | 71.259124 | 0.468299 | 0.684625 |
| Numeric 25 | [32,16] | 78.470989 | 70.954988 | 0.483965 | 0.677078 |
| Numeric 25 | [64,32] | 83.496152 | 70.012165 | 0.395685 | 0.793191 |

In every feature variant, larger/deeper alternatives lower training NLL but worsen validation NLL relative to `[32]`. The full-feature `[64,32]` model slightly improves validation state accuracy over `[32]`, but its much worse NLL means it is not selected. This illustrates why argmax accuracy and probability quality must not be conflated.

The increasing train/validation gap is consistent with overfitting under this fixed 300-epoch recipe, especially with categorical inputs. All 48 candidate-seed evaluations have zero true-class probability-floor hits; high NLL is not the zero-probability artifact seen in some deeper RF candidates. This does not establish that every MLP or training recipe is unsuitable. No stronger regularization, earlier checkpoint, calibration, or alternate selection rule was tried after seeing these results.

### Selected policies and habitat correction

The habitat experts are unchanged: image **68.673966%**, geo **61.253041%**, and raw fusion **71.046229%** mean validation accuracy. Router state accuracy is a different task from the habitat accuracy after correction.

| MLP features | Selected hidden widths | State accuracy (%) | State macro-F1 | Routed habitat accuracy (%) | Gain over raw (pp) | Actions | Rescues | Harms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 727 | [32] | 68.278589 | 0.545508 | 71.046229 | 0 | 0 | 0 | 0 |
| No-pairs 79 | [32] | 69.130170 | 0.548117 | 71.046229 | 0 | 0 | 0 | 0 |
| Numeric 25 | [32] | 71.502433 | 0.560191 | 71.472019 | +0.425791 | 38 | 24 | 10 |

Full 727 has **0/1,493** feasible threshold candidates; no-pairs 79 has **0/1,416**. Both use the declared never-intervene fallback, not an unreported threshold or post-hoc policy. Numeric 25 has **8/1,638** feasible candidates and selects strict `q_rescue-q_harm > 0.5569448147166254` (authoritative float64 hex `0x1.1d27dee9c4f58p-1`), together with raw/geo disagreement.

Numeric 25 acts on 1.155718% of seed-image rows, involving 36 distinct images across 25 distinct plots. Its 24 rescues and 10 harms yield 14 net correct seed-image predictions, or 3.5 per seed; four further actions leave an already-wrong prediction wrong. Per-seed (actions, rescues, harms, net) counts are `(10,6,2,4)`, `(14,9,4,5)`, `(11,6,4,2)`, and `(3,3,0,3)`. Its conditional 95% plot-bootstrap interval for habitat gain is **[-0.061128, +0.900395] pp**, including zero.

Selected full/no-pairs/numeric routers have disagreement-row rescue AP of 0.393/0.373/0.399 and harm AP of 0.689/0.677/0.668, respectively. Habitat top3 accuracy is 91.788321% for raw and the two fallback policies, versus 91.818735% for the numeric MLP policy. Full per-seed, per-class, subgroup, Brier, and confusion-matrix values are retained in the score bundle.

The simple agreement/harm state rule scores 71.046229% on this validation set. Only the selected numeric MLP is slightly above it (+0.456204 pp in state accuracy); both categorical-feature MLPs are below it. This state-rule comparison does not claim that a router's state accuracy is its routed habitat accuracy.

### Frozen baseline comparison

These are habitat accuracies after applying each previously selected policy; LR/RF/XGB models and thresholds were not reselected.

| Features | LR (%) | RF (%) | XGB (%) | MLP (%) |
| --- | ---: | ---: | ---: | ---: |
| Full 727 | 71.046229 | 71.745742 | 71.380779 | 71.046229 |
| No-pairs 79 | 71.411192 | 71.715328 | 71.624088 | 71.046229 |
| Numeric 25 | 71.654501 | 71.776156 | 71.593674 | 71.472019 |

The strongest previously declared margin heuristic reaches 71.715328% (+0.669100 pp over raw). No selected MLP has a higher point estimate than its corresponding RF, XGB, or the margin heuristic; MLP/full ties LR/full because both never intervene. All twelve paired conditional 95% intervals below include or touch zero, so this study does not establish a strict performance ordering.

| MLP features | Comparator | MLP-minus-comparator gain (pp) | Conditional 95% interval (pp) |
| --- | --- | ---: | --- |
| Full 727 | LR/full | 0 | [0, 0] |
| Full 727 | RF/full | -0.699513 | [-1.407675, 0.000000] |
| Full 727 | XGB/full | -0.334550 | [-0.803740, +0.151888] |
| Full 727 | Margin heuristic | -0.669100 | [-1.448972, +0.092062] |
| No-pairs 79 | LR/no-pairs | -0.364964 | [-0.845411, +0.062934] |
| No-pairs 79 | RF/no-pairs | -0.669100 | [-1.409314, +0.090588] |
| No-pairs 79 | XGB/no-pairs | -0.577859 | [-1.184690, +0.060102] |
| No-pairs 79 | Margin heuristic | -0.669100 | [-1.448972, +0.092062] |
| Numeric 25 | LR/numeric | -0.182482 | [-0.538326, +0.152821] |
| Numeric 25 | RF/numeric | -0.304136 | [-0.804480, +0.154321] |
| Numeric 25 | XGB/numeric | -0.121655 | [-0.544827, +0.303407] |
| Numeric 25 | Margin heuristic | -0.243309 | [-0.899286, +0.354727] |

The RF/full upper endpoint is approximately `1.11e-17` pp before rounding, numerically zero. Zero-width intervals for identical never-intervene policies describe an identical frozen action rule, not certainty about an unobserved future study.

## Verification and completion evidence

- All 48 declared models have exactly 300 finite epoch records, the expected parameter counts, no training warnings, and exact native/portable argmax agreement. Worst absolute probability error is `1.78e-15`; worst logit error is `5.68e-14`, within the declared absolute-plus-relative tolerance.
- Training NLL continues to decrease between epochs 200 and 300 for every fit (changes from -0.06142 to -0.01287). This is a fixed-epoch experiment, not a claim of optimizer convergence. All histories and final states remain available for inspection.
- Independently repeated full727/[64,32]/seed1 and numeric25/[32]/seed1 fits reproduce complete state hashes, all 300 history records, and saved training/validation probabilities bit-for-bit. Timing fields are not part of that equality claim. The repeats used no validation targets and wrote no artifacts.
- **1,038 focused tests pass**: 839 predecessor M1–M4/capacity cases plus 199 MLP/replay cases. The latter include the in-memory-to-JSON curve round trip, both known resume layouts, malformed CLI summaries, process separation, tampering, lock contention, partial recovery, and immutable reuse. The process-capability guard is tested in a fresh subprocess; synthetic publication units isolate it from unrelated test-module imports.
- First production fitting/sealing took **279.14 s** on CPU. Versioned replay/scoring build took **21.20 s**, including **17.82 s** for the scoring phase, with zero additional fits. Independent `validate` took **20.21 s**. Repeat `build` returned **`reused_valid`, `new_fits: 0`** in **20.27 s**.
- The original 148-file MLP bundle has the same bytes and mtimes before and after recovery/validation/reuse. The 18-file recovery bundle also preserves bytes and mtimes across validation/reuse; its fingerprint is `89a031f371f702de54dcc096ffb3c0114242d1ee957fe07f6530957203e347d2`.
- The 451-file predecessor capacity bundle retains its original bytes/mtime fingerprint `9a960060d40998fa00fbd931240ecf4d47c794cc9e90de47f1115c851df19ee4`. All 660 original frozen ancestors and all 810 recovery-bound files validate. No cleaned-test evaluation or legacy-transfer scoring was performed.

| Receipt or manifest | SHA-256 |
| --- | --- |
| Original M4 manifest | `442d8bc6dafe992ea6e3e7a8793df51571ece4643c2a7d0ce4a25b20b6139615` |
| MLP pre-fit declaration | `8bf5b5c85f72097bb9abadc13bb1654bc558ba6f4a2ee164c9cec4a6789bb8ec` |
| MLP prediction manifest | `54bd97a74bb9a345a49b64802dc45da5481d92c4fce07e3eb624f936e6e9cb43` |
| Replay declaration | `1865a2b1851e07aa39b646e5e6a6a9909f5b26802ab06845a7b437c44381aca2` |
| Replay score manifest | `6aa9f73fac52af957f075712bbca472e03e76cc31f7d639e47ef3044c4fe78bd` |
| Selected-policy receipt | `a8f12a8f6a42d04ae58f48ace9291a7fd55e80d863fb47845dc3a333d3bf13c3` |

Conclusion: the agreed architecture search is complete, including both feature ablations. Increasing width/depth under this recipe improves training fit but not validation NLL. Numeric-only `[32]` is the most useful MLP here, yet its correction gain is modest and below existing baseline point estimates. Preserve this negative/exploratory result. Original M4 remains **no-go**, and M5/M6 remain deferred; this round authorizes neither further holdout tuning nor test evaluation.

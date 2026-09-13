# Router capacity exploration — first development-only round

Status: COMPLETE — models, predictions, nine selected policies, and scoring receipt sealed; independent validation and byte-preserving reuse passed on 2026-09-09. Original M4 `no_go` unchanged.

Study ID: `router_capacity_v1`.

This is a new exploratory capacity round using the existing frozen development inputs. It does not revise the original M4 feasibility `no_go` decision. The development validation set has already been used in earlier analyses; model and threshold selection here are therefore selection-conditional exploration, not an independent confirmatory test.

## Frozen inputs and scope

Every learner uses the same four training seeds, `1, 2, 3, 4`, and the same development out-of-fold training universe: 3,378 images from 1,300 plots per seed (13,512 seed-image rows). The development validation universe is 822 images from 325 plots per seed (3,288 seed-image rows).

The input adapter verifies the frozen full 727-column training matrix, reconstructs validation features using the frozen transforms, checks the calibrated expert probabilities against M2, and checks all 48 original LR validation predictions. No expert, temperature, feature transform, or original LR candidate is refitted.

The fixed target order is:

1. `rescue`
2. `harm`
3. `both_correct`
4. `both_wrong`

All three feature variants are applied to every learner family: `primary_full727` (727 columns), `ablation_no_pairs79` (79 columns), and `ablation_numeric25` (25 columns). They are the existing frozen column selections from the common full 727-column input, not newly selected features.

Fit code receives training targets and label-blind validation inputs. Validation targets are decoded only by the separate scoring stage after candidate states and predictions have been sealed. There is no cleaned-test access, no use of the legacy transfer evaluation outputs, and no test-driven selection in this round.

## Ordered candidate grid

The machine-readable declaration is `configs/multimodal_geo_router_capacity_v1.json`. Its order is family `lr`, `rf`, `xgb`; within each family, the three variants in the order listed above; within each variant, candidates in the order below. This order is fixed before scoring.

| Family | Candidate 1 | Candidate 2 | Candidate 3 | Candidate 4 |
| --- | --- | --- | --- | --- |
| LR | C=0.01 | C=0.1 | C=1 | C=10 |
| Regularized RF | depth=4, leaf=20 | depth=4, leaf=5 | depth=8, leaf=20 | depth=8, leaf=5 |
| Shallow XGB | depth=2, lambda=1 | depth=3, lambda=1 | depth=3, lambda=10 | depth=4, lambda=10 |

LR reuses the existing frozen four-C candidates for all three variants and all four seeds. Its JSON parameters contain only `C`, as required by the frozen-candidate reuse adapter; its other training settings are inherited unchanged from the sealed original states.

RF fixed settings: 500 trees, `max_features=0.5`, `bootstrap=true`, `criterion="log_loss"`, `class_weight=null`, `n_jobs=1`, `oob_score=false`, and `max_samples=null`. Only `max_depth` and `min_samples_leaf` vary as declared.

XGB fixed settings: 200 boosting rounds (800 trees: one per state per round), learning rate 0.05, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0`, `max_bin=256`, `tree_method="hist"`, `objective="multi:softprob"`, `num_class=4`, `base_score=0.5`, and `n_jobs=1`. Only `max_depth` and `reg_lambda` vary as declared.

RF and XGB receive `random_state` equal to the corresponding training seed; it is not a grid parameter and is deliberately omitted from the configuration file. Implementations are pinned to scikit-learn 1.5.2 and XGBoost 2.1.4, with thread-pool limit 1.

There is no early stopping, class weighting, validation-driven fit adaptation, router probability calibration, or new expert temperature fitting. Frozen expert calibration is retained unchanged.

The grid contains 36 configurations and 144 seed cases: 48 original LR cases reused, 48 new RF fits, and 48 new XGB fits (96 new fits total).

## Fixed selection and action rules

Select one configuration separately for each of the nine family-by-variant groups. The selection objective is the unweighted mean of the four seed-specific validation negative log likelihoods over all validation rows. Candidates whose mean is within `1e-12` of the minimum are tied; the earliest candidate in the declared configuration order wins. No accuracy, utility, or subgroup result changes this model-selection rule.

After selecting a configuration, apply the original M4 `select_common_threshold` rule unchanged to the pooled four-seed validation rows. Router score is `q_rescue - q_harm`; intervention uses the geo expert only on raw/geo prediction disagreements with score strictly greater than the common threshold, and otherwise retains the raw expert.

A feasible threshold must act on at least 1% of pooled rows and at least 20 distinct plots, satisfy `2 * harms <= rescues`, and have positive rescues-minus-harms. Among feasible thresholds, maximize net rescues-minus-harms; ties prefer fewer harms, then fewer actions, then the higher threshold. If no threshold is feasible, use the typed `never_intervene` policy. The original candidate enumeration and threshold semantics are reused, not redefined here.

## Reporting limits

Report all predeclared candidates and all nine selected family-by-variant results, including the original LR reference. Retain both the four-state predictive-quality results and intervention utility/coverage results; do not suppress unsuccessful configurations or reinterpret a selected validation gain as confirmation.

Any uncertainty estimates condition on the selected configurations, thresholds, fitted routers, frozen experts, and preprocessing. Selection is not repeated inside the bootstrap. Reusing this development validation set means results cannot establish independent generalization or justify changing the original M4 `no_go` conclusion.

## Results

All 36 declared configurations were evaluated across four seeds. The three original LR selections, thresholds, and action masks reproduce M4 exactly. The selected settings are C=0.1 for LR/full and LR/no-pairs, C=10 for LR/numeric; depth=4 and minimum leaf size=5 for all RF variants; depth=2 and lambda=1 for all XGB variants. No result-dependent search expansion occurred.

Numbers below are development-validation results. Accuracies, NLL, and gains are means over four seeds. Because each seed has 822 images, pooled seed-realization accuracy has the same value. Action, rescue, and harm counts are sums over four seeds, not counts of distinct images. The effective validation universe remains 822 images from 325 plots.

The unchanged expert habitat accuracies are image **68.673966%**, geo **61.253041%**, and raw fusion **71.046229%**. The simple agreement/harm state predictor has four-state accuracy **71.046229%**, numerically equal to raw habitat accuracy but evaluated as a four-state predictor.

| Learner | Features | Four-state NLL | Four-state accuracy | Four-state macro-F1 | Routed habitat accuracy | Gain vs raw (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 727 | 0.609177 | 71.228710% | 0.557888 | 71.046229% | 0.000000 |
| LR | 79 | 0.606749 | 71.289538% | 0.559141 | 71.411192% | +0.364964 |
| LR | 25 | 0.605925 | 71.897810% | 0.553797 | 71.654501% | +0.608273 |
| RF | 727 | 0.603534 | 72.384428% | 0.552963 | 71.745742% | +0.699513 |
| RF | 79 | 0.603477 | 72.384428% | 0.553488 | 71.715328% | +0.669100 |
| RF | 25 | 0.617915 | 72.323601% | 0.553125 | 71.776156% | +0.729927 |
| XGB | 727 | 0.614636 | 71.715328% | 0.563307 | 71.380779% | +0.334550 |
| XGB | 79 | 0.614102 | 71.684915% | 0.562176 | 71.624088% | +0.577859 |
| XGB | 25 | 0.625296 | 71.532847% | 0.556821 | 71.593674% | +0.547445 |

The state task and habitat task are different columns: state accuracy uses the four-state argmax; routed habitat accuracy uses the selected `q_rescue - q_harm` threshold to choose geo or retain raw. Neither metric substitutes for the other.

| Learner / features | Actions | Rescues | Harms | Coverage | Net correct by seed 1–4 | Conditional 95% gain interval (pp) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| LR / 727 | 0 | 0 | 0 | 0% | 0, 0, 0, 0 | [0, 0] |
| LR / 79 | 38 | 23 | 11 | 1.155718% | 0, 7, 5, 0 | [-0.0629, 0.8454] |
| LR / 25 | 55 | 34 | 14 | 1.672749% | 4, 8, 4, 4 | [0.0620, 1.1587] |
| RF / 727 | 69 | 42 | 19 | 2.098540% | 5, 12, 3, 3 | [0, 1.4077] |
| RF / 79 | 76 | 43 | 21 | 2.311436% | 3, 12, 3, 4 | [-0.0906, 1.4093] |
| RF / 25 | 76 | 44 | 20 | 2.311436% | 4, 13, 3, 4 | [0, 1.4653] |
| XGB / 727 | 35 | 20 | 9 | 1.064477% | 2, 7, 1, 1 | [-0.1519, 0.8037] |
| XGB / 79 | 56 | 33 | 14 | 1.703163% | 3, 11, 4, 1 | [-0.0601, 1.1847] |
| XGB / 25 | 41 | 27 | 9 | 1.246959% | 2, 8, 5, 3 | [0.0906, 1.0229] |

Intervals use 2,000 shared plot-resampling replicates and condition on already selected models and thresholds. Near-zero floating-point lower bounds for RF/727 and RF/25 are displayed as zero, not interpreted as strictly positive evidence. Paired tree-minus-matching-LR gain intervals all include zero. No multiplicity-adjusted or independent superiority claim is made.

| Features | XGB minus matching LR gain (pp) [95% interval] | RF minus matching LR gain (pp) [95% interval] |
| --- | --- | --- |
| 727 | +0.335 [-0.152, 0.804] | +0.700 [0, 1.408] |
| 79 | +0.213 [-0.124, 0.576] | +0.304 [-0.152, 0.771] |
| 25 | -0.061 [-0.385, 0.267] | +0.122 [-0.302, 0.513] |

These contrasts use the paired policy metrics in `bootstrap_replicates.parquet`; they do not rerun model or threshold selection inside a replicate.

The strongest original heuristic, OOF-selected `geo_minus_raw_margin`, still gives +0.669100 pp (78 actions, 44 rescues, 22 harms). RF/79 matches that point gain; RF/727 and RF/25 exceed it by only one and two net-correct seed-realizations respectively. The higher-calibrated-geo-confidence heuristic gives -0.456204 pp. All five heuristics and the original primary policy are retained in the saved comparison.

### Interpretation

- RF provides a modest capacity benefit: full-feature state accuracy improves by 1.155718 pp over full LR, with lower NLL and Brier score. It does not produce a large improvement over the simple state rule, and its four-state macro-F1 does not improve.
- Rescue ranking on raw/geo disagreements improves from AP 0.412295 (LR/727) to 0.449100 (RF/727). This is useful but limited; the selected RF policies intervene on only about 2.1–2.3% of seed-image rows.
- The ablations remain competitive. RF/79 nearly duplicates RF/727 state quality, while RF/25 has the largest routed point gain but worse NLL. Predicted-class pairs show no clear practical benefit in this round. This is a descriptive comparison across the predeclared groups, not a new global feature-selection decision.
- XGB does not consistently outperform LR: it has higher NLL for every matching feature variant, and its numeric-only state accuracy is lower. The shallowest XGB setting wins all three within-family selections; greater capacity is not automatically useful.
- Deeper RF candidates assign exactly zero probability to the true state for 2–11 of the 3,288 validation seed-image rows, depending on configuration. The inherited metric floors true-class probabilities at `np.finfo(float64).tiny` when computing NLL, heavily penalizing these events. The selected depth-4 RF candidates have no such floor hits. This is a concrete probability-quality limitation of the deeper RF fits; saved probabilities were not smoothed or recalibrated after seeing the results.
- These results do not establish that model capacity is the sole limitation. Probability quality, feature information, generalization to different experts, and selection uncertainty remain relevant. Original M4 `no_go` is preserved; no RF/XGB policy is promoted to M5 or evaluated on cleaned test by this round.

Training-state accuracies for the selected models are 76.37/75.30/74.14% (LR, features 727/79/25), 75.33/75.31/75.22% (RF), and 77.04/77.02/76.75% (XGB). These are resubstitution diagnostics, not independent evaluation; the lower XGB training loss does not translate into better validation NLL.

Training-to-validation gaps also include the difference between OOF experts and experts fitted on the whole development-training split. They cannot be attributed solely to router overfitting.

## Verification record

- Build succeeded: 96 new fits, 48 exact LR reuses, 144 model states and 604,800 prediction rows; 451 total files including both manifests and the declaration.
- Initial build took 451.65 seconds: 391.05 seconds fitting/export/prediction sealing and 60.59 seconds scoring. Independent apply-only validation succeeded in 109.00 seconds.
- 96 focused new tests passed. A focused M1–M4 regression run passed 743 tests. No frozen implementation or scientific-setting changes were needed.
- Independent audit verified all 198 parent/code/dependency hashes, exact old LR state/prediction equality, all 36 NLL calculations, all nine threshold curves/action masks, and receipt/model links.
- Native class predictions match portable replay exactly. Maximum probability errors were 0 (LR), 2.22e-16 (RF), and 8.93e-8 (XGB); native XGB margins reproduce exactly. No fitting warnings occurred.
- Prediction manifest: `82fc8e5f6cea6b4960c1b6429d744ace4b082d3b63dcdd38d6c08dce54ba5be0`.
- Score manifest: `d3bbb425ca4e83dd968eb78bb0afd73bdebf89f1330b5a5988323dfe7c372104`.
- Declaration: `1fbede42cdddb8d5335706ba0b6f6f7f9fe548cf837964e400c16244581ad436`.
- Selection receipt: `6982bfe358e5685db9a72624e6b7ea968ba81874a4d1e501f19435f7b253f1a6`.
- Original M4 manifest remains `442d8bc6dafe992ea6e3e7a8793df51571ece4643c2a7d0ce4a25b20b6139615`.
- Repeat build returned `reused_valid` in 109.19 seconds with zero new fits. All 451 file bytes and modification times were unchanged through independent validation and reuse; aggregate bytes/mtime fingerprint: `9a960060d40998fa00fbd931240ecf4d47c794cc9e90de47f1115c851df19ee4`.

## Reproduction commands and artifacts

Run from the repository root. The pinned CPU-only XGBoost package is installed into a project-local directory; existing `habcls` dependencies are not upgraded:

```bash
conda run -n habcls python -m pip install --require-hashes --no-deps \
  --target multimodal_artifacts/environments/router_capacity_v1/xgboost_cpu_2_1_4 \
  -r configs/router_capacity_requirements.txt

conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_router_capacity.py build
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_router_capacity.py validate
conda run --no-capture-output -n habcls python tools/run_multimodal_geo_helpfulness_router_capacity.py build
```

Install the dependency once, before the first build. Do not reinstall or modify its files after the declaration is sealed: their hashes are part of the replay contract.

The first `build` fits/reuses the declared candidates, seals all predictions, then starts a separate fitting-incapable scorer. `validate` reconstructs probabilities, features, selections, and metrics without fitting. The repeated `build` checks committed artifacts and returns `reused_valid` without changing their bytes. There is no force-overwrite option. `fit`, `score`, `validate-predictions`, and `validate-scores` expose the individual stages; path overrides are `--protocol-dir`, `--output-dir`, and `--config`.

The default output directory is `multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/router_capacity_v1`, outside the frozen protocol directory. It contains:

- `declaration.json`: ordered configuration, parent/code/dependency hashes, environment, access boundaries, and numerical conventions, sealed before any new fit.
- `predictions/models/`: 144 self-hashed portable JSON model records with fitting/export diagnostics; 48 wrap unchanged LR states, 48 contain RF trees, and 48 contain XGB trees.
- `predictions/train/` and `predictions/validation/`: one label-blind Parquet probability table per candidate/seed, with three identity columns, configuration/family/variant identifiers, four float64 state probabilities, and `score = q_rescue - q_harm`.
- `predictions/validation_features.parquet`: the common 727-feature validation matrix with five identity/protocol columns. The training matrix remains the frozen M4 parent.
- `predictions/manifest.json`: final commit of all 433 prediction-stage artifacts; 486,432 training and 118,368 validation prediction rows across the 144 candidates.
- `scores/`: all 36 configuration metrics, training-only fitting diagnostics, nine selections/frozen policies, original baseline and heuristic comparisons, action/threshold tables, habitat classification metrics, paired plot bootstrap and subgroup results, access audit, selection receipt, and final manifest.

All saved probability outputs are float64. Tree split inputs reproduce native float32 behavior; XGB margins accumulate in float32 and use a stable float64 softmax. Native XGB parity uses `atol=rtol=1e-6`; portable-state reconstruction uses `atol=rtol=1e-12`. Probability normalization uses absolute tolerance `1e-8`. RF and LR native parity use `1e-12`. No pickle is required for replay. These conventions were checked against pinned native implementations before production fitting.

Documentation checks informed the portable model-export and replay implementation: [XGBoost model IO](https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/saving_model.html), [the pinned XGBoost 2.1.4 CPU predictor](https://github.com/dmlc/xgboost/blob/v2.1.4/src/predictor/cpu_predictor.cc), and [scikit-learn 1.5 RandomForestClassifier](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Native parity fixtures against the installed pinned versions verify the behavior actually used here.

Publication uses an exclusive lock, no-overwrite writes, and manifests written last. A declared but interrupted build can reuse valid self-hashed model records and fill missing predictions. Corrupt committed artifacts and unknown files cause a hard failure; the workflow never deletes unknown files. Scoring refuses to run in a process with fitting libraries loaded.

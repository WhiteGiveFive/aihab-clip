# Exploratory transfer of frozen M4 routers to legacy experts

Study ID: `legacy_frozen_router_transfer_v1`.

This separately versioned study evaluates **both** `best_model.pt` and
`oracle_test_best_model.pt`, as explicitly requested by the user. It does not
replace protocol-v1 expert refitting, complete M5, promote an ablation, or change
M4's primary **no-go** conclusion. No expert, temperature, feature transform,
router, regularization value, or intervention threshold is fitted or selected.

## Sources and interpretation

The legacy root is
`multimodal_artifacts/runs/cs/hf-hub_timm_ViT-SO400M-16-SigLIP2-384_5_20260203_17`.
For each of `image_only`, `geo_only`, and `raw_concat`, seeds 1–4 use
`gse_100m_train_cleaned_test_epoch50/seed<seed>/`.

- `best_model.pt` is the fixed **epoch-50** checkpoint, not a test-selected
  checkpoint despite its name.
- `oracle_test_best_model.pt` is explicitly selected on cleaned-test top-1
  accuracy. Its results are **test-selected, biased diagnostics**, not an
  independent confirmation or an oracle performance ceiling.
- Both families have historical test exposure. The cleaned test remains
  exploratory, and the global score event records `adaptive_reuse=true` even
  if no earlier event was registered in this particular registry.

Use the matching `joined_tables/cs/<same-backbone>/gse_100m_cleaned_test/seedN/`
caches, not similarly named caches from other run families. The legacy
development split is 3,618 + 582 images; its union is the exact M1 development
universe of 4,200 images and 1,625 plots with matching 18-class labels. All 12
saved geographic scalers are identical and reproduce from that union.

The image caches come from a shared historical habitat-adapted encoder, not the
four protocol-v1 final encoder adaptations. Legacy artifacts do not embed an
end-to-end encoder-to-cache-to-head content fingerprint. Paths, configuration,
class order, and reproducible scalers support the association; the new
declaration hashes the exact reused caches and heads without claiming stronger
historical provenance. Cached image features are not normalized again.

## Frozen application and access boundaries

Each checkpoint family retains all four seeds. CPU float32 MLP inference uses
the saved geographic scaler, ReLU, and both saved linear layers, with dropout
disabled and batches of 128. Logits are stored as float64. All 12 M4 OOF
temperatures and four feature transforms are applied unchanged; calibration
quality is **not guaranteed** for these different experts.

The three selected routers and five M4 comparators retain their exact policies.
The primary policy's typed `never_intervene` threshold always reproduces
RawConcat. The two ablations can intervene. Actions require geo/raw disagreement
and use the frozen strict threshold comparison. No primary `t=0` policy is added.

The declaration binds 202 source/code files before test-feature decoding.
Inference projects only file/plot identities and image/geographic feature
columns. Both families' predictions are sealed in one bundle before the scoring
process reserves an append-only global event and opens test labels. Scoring has
no Torch, SciPy, scikit-learn, fitting, or model-selection imports. Source labels
and selected historical metric fields are copied into a sealed acquisition
receipt; completed validation/reuse does not reopen their original readers.

## Reproducible commands

From the repository root:

```bash
conda run -n habcls python tools/run_multimodal_geo_helpfulness_legacy_router.py build
conda run -n habcls python tools/run_multimodal_geo_helpfulness_legacy_router.py validate
conda run -n habcls python tools/run_multimodal_geo_helpfulness_legacy_router.py build
```

`build` runs label-blind prediction and scoring in separate subprocesses.
`validate` repeats original-head inference and then reconstructs scoring from
sealed labels in another fresh process. A valid repeat build returns
`reused_valid` and preserves artifact bytes. Path overrides are available for
`--protocol-dir`, `--legacy-root`, `--cache-root`, and `--output-dir`; there are
no scientific-setting or force-overwrite overrides. Internal stages `predict`,
`score`, `validate-predictions`, and `validate-scores` are also callable.

Publication uses locks, exclusive/no-overwrite writes, owned partial receipts,
and manifests written last. Corrupt or unknown outputs are not overwritten or
deleted. Numerical reconstruction tolerance is `atol=rtol=1e-12`; probability
row-sum tolerance remains `1e-8`.

## Artifacts

Output root:
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/exploratory_legacy_router_v1/`.

| Artifact | Content |
|---|---|
| `declaration.json` | Both checkpoint families, sources/hashes, environment, authorization, legacy provenance and fixed M4 components |
| `predictions/experts.parquet` | 10,776 rows × 46 columns: identities, logits, native/transferred probability vectors, 30 semantic features; no labels |
| `predictions/routers.parquet` | 32,328 rows × 13 columns: three variants, four-state probabilities and rescue-minus-harm scores |
| `predictions/actions.parquet` | 86,208 rows × 13 columns: eight policies, frozen thresholds, actions and final predictions; no outcomes |
| `predictions/{metadata,manifest}.json` | Dimensions, interpretation, immutable prediction seal |
| `scores/labels.parquet` | One unique-image label projection, shared by both family reports |
| `scores/{source_receipt,historical_replay}.json` | Acquired-source hashes and top-1/top-3 reproduction for all 24 expert/seed/family combinations |
| `scores/<family>/` | Expert calibration, classification and four-state router metrics; policy metrics; 2,000 plot-bootstrap replicates and summary; subgroup metrics; actions with outcomes and family-specific four-state targets |
| `scores/{summary.json,report.md}` | Side-by-side descriptive results |
| `scores/{build_declaration,reservation,receipt,manifest}.json` | Label-access authorization, registry ownership, immutable scoring result seal |

Counts pool repeated seed realizations where stated: each family has 5,388
expert rows, but only **1,347 unique test images from 531 plots**. The two
families do not constitute independent test samples. Seed means and population
SDs are descriptive; paired bootstrap samples whole plots, retains all seed
realizations, and does not account for historical checkpoint selection.

## Results and verification

Completed on 2026-09-09. All entries below are from the sealed scoring bundle.
Top-1 and top-3 percentages are means ± population SD across four seeds.

| Classifier | Fixed-final top-1 | Fixed-final top-3 | Test-selected oracle top-1 | Test-selected oracle top-3 |
|---|---:|---:|---:|---:|
| Image expert | 73.144 ± 0.169 | 92.409 ± 0.110 | 73.627 ± 0.110 | 93.578 ± 0.134 |
| Geo expert | 61.618 ± 0.662 | 89.365 ± 0.380 | 64.124 ± 0.285 | 90.163 ± 0.350 |
| RawConcat expert | 74.313 ± 0.091 | 93.838 ± 0.189 | 75.334 ± 0.169 | 94.989 ± 0.321 |
| RawConcat + primary router | 74.313 ± 0.091 | 93.838 ± 0.189 | 75.334 ± 0.169 | 94.989 ± 0.321 |
| RawConcat + no-pairs ablation | 74.703 ± 0.110 | 93.764 ± 0.203 | 75.334 ± 0.177 | 94.952 ± 0.292 |
| RawConcat + numeric ablation | 74.684 ± 0.091 | 93.782 ± 0.219 | 75.427 ± 0.117 | 94.952 ± 0.292 |

The following counts pool the four seed realizations (5,388 rows per family),
not 5,388 independent images. Gain is relative to that family's RawConcat.

| Family | Router | Actions | Rescues | Harms | Mean seed net | Gain (pp) | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| Fixed-final | Primary | 0 | 0 | 0 | 0 | 0.000000 | 0.000% |
| Fixed-final | No-pairs | 79 | 42 | 21 | 5.25 | +0.389755 | 1.466% |
| Fixed-final | Numeric | 85 | 43 | 23 | 5.00 | +0.371195 | 1.578% |
| Test-selected oracle | Primary | 0 | 0 | 0 | 0 | 0.000000 | 0.000% |
| Test-selected oracle | No-pairs | 13 | 5 | 5 | 0 | 0.000000 | 0.241% |
| Test-selected oracle | Numeric | 27 | 12 | 7 | 1.25 | +0.092799 | 0.501% |

Router four-state classification accuracy is separate from final 18-class
habitat accuracy:

| Router | Fixed-final four-state accuracy | Test-selected oracle four-state accuracy |
|---|---:|---:|
| Primary | 75.297% | 66.759% |
| No-pairs | 75.241% | 66.889% |
| Numeric | 75.408% | 66.036% |

The descriptive paired plot-bootstrap 95% intervals for top-1 gain all include
zero: fixed-final no-pairs `[-0.056, +0.842]` pp, fixed-final numeric
`[-0.112, +0.878]` pp, oracle no-pairs `[-0.168, +0.151]` pp, and oracle numeric
`[-0.114, +0.316]` pp. These intervals condition on the supplied checkpoints;
they do not remove historical test-selection bias. The modest fixed-final
ablation gains therefore do not establish generalization, revise M4's no-go,
or justify selecting a checkpoint family using these results.

Verification evidence:

- All 24 expert/seed/family top-1 and top-3 counts reproduce the 12 historical
  metric files exactly within the declared numerical tolerance.
- `predict` completed in 4.44 seconds and first `score` in 32.23 seconds on CPU.
  Independent `validate` took 21.64 seconds; repeat `build` took 21.74 seconds.
  These are observed wall times for this machine/run, not performance guarantees.
- Independent validation repeated all original-head forward passes, rebuilt
  features/router probabilities/actions, and recomputed all reports and 2,000
  bootstrap replicates per family from sealed labels. All matched.
- Repeat build returned `reused_valid` for both phases. All **33** result files
  retained identical SHA-256 hashes and nanosecond modification times.
- All **202** source/code file hashes and the original M4 no-go remain unchanged.
- The append-only global event, completion record, acquisition receipt and
  result receipt agree. Original label/history readers were disabled during
  an additional independent audit, which still reproduced every report.
- All **114 new focused tests** pass. The broad geographic collection produced
  **868 passes and one unrelated failure**: the existing 10-metre test at
  `tests/test_multimodal_geo_10m.py:188` mistakes metadata `ID` for an image
  feature because it uses `startswith("I")`. All **751 M1–M4 tests** and all
  114 new tests passed within that run. The unrelated files were not changed.

```bash
conda run -n habcls python -m pytest -q tests/test_multimodal_geo_legacy_router*.py
```

Frozen identifiers:

| Object | SHA-256 |
|---|---|
| Declaration payload | `c9185bc5c278b1d386313bd8b0032eae2527ca60c44137885ed572b728e11df1` |
| Combined prediction manifest payload | `b321e7859e146eb0dea8bfa3eb7c38c620b682e739b40e4724c93e1a4f4a65d2` |
| Score manifest payload | `c7b31fc59e62b62a889d1ec4ebdb1ffea8531fc05386fe2675108e4c7d9b38b3` |
| Score receipt payload | `193fb4e21cc38ee4bb8a602d60f54c247122f4d2c29e0d4818c04685e0f2e9fd` |
| Global evaluation event ID | `a92071443757494eb8163823dc93d592f4876bf96573804731e2854ebe9a79c1` |

The receipt distinguishes payload hashes from physical file hashes. Detailed
per-seed/per-class metrics, four-state confusion matrices, calibration
diagnostics, all five comparators, subgroup support flags and bootstrap
replicates remain available in the scored artifacts rather than being reduced
to the summary tables here.

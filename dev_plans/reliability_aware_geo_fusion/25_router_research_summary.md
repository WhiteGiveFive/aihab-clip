# Research summary: output and geometry routers

Recorded 2026-10-06. The concise [research report](../../multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/router_research_summary_v1/report.md)
describes the two router families, their test-time application, the selected
RF and adapted-geometry results, OR/AND combinations, habitat confusions, and
the remaining gap to an oracle choosing between fusion and geo.

Section 1 includes a six-group feature table, plot-grouped out-of-fold router
training, geometry distance/support/decision formulas and a hypothetical
two-modality schematic. It cites the original TrustScore paper and distinguishes
that estimator from the current candidate-specific routing adaptation.

## Scope and main finding

The comparison preserves the historical RF policy and the adapted geometry
policy with **4,200 references, strict K5**, under the saved validation
H≤0.7R selection. It uses the same 1,347 cleaned-test images from 531 plots
and four seeds. The oracle switches only when fusion is wrong and geo is
correct; its 83.0549% accuracy is a ceiling for these two fixed expert outputs.
RF achieves 75.1670%, geometry 75.7053%, OR 75.9465%, and AND 74.9258%.

There are 471 oracle rescues across the 5,388 seed–image decisions. RF misses
342 and introduces 83 harms; geometry misses 315 and introduces 81 harms.
Missed rescues therefore explain about 80% of each oracle gap. Of geometry's
missed rescues, only seven lack strict-K5 support. Image geometry rejects 198
rescue opportunities that geo geometry accepts, but also blocks 622 harmful
switches in the same condition. These are descriptive diagnostics, not a
proposal to remove the image gate.

The report separates measured errors from possible explanations. It does not
claim that embedding overlap, label noise or training changes caused the gap.
OR/AND remain exploratory combinations proposed after test inspection. No
model fitting, threshold search, bootstrap resampling or independent
confirmation is performed.

## Reproduction

From the repository root in the existing `habcls` environment:

```bash
python tools/plot_multimodal_geo_helpfulness_router_research_report.py
```

The default destination is
`multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/router_research_summary_v1/`.
Use `--output-dir` with a fresh `router_research_*` directory directly under
the same artifact root to reproduce separately, and register retained outputs
in the study catalog. The generator authenticates the existing geometry/RF
bundles, checks cohort identities and reproduces the archived component
metrics. It requires those cached bundles and the earlier September 15 report;
it requires no images, GPU or expert inference.

The [report generator](../../multimodal/geo_helpfulness_router_research_report.py)
and [figure module](../../multimodal/geo_helpfulness_router_research_figures.py)
are accessed through the [canonical entrypoint](../../tools/plot_multimodal_geo_helpfulness_router_research_report.py).

| Output | Purpose |
| --- | --- |
| `report.md` | Concise research narrative and comparison tables |
| `figures/geometry_router_schematic.png`, `.pdf` | Hypothetical query/reference comparison and the two-modality geometry gate |
| `figures/confusion_focus_comparison.png`, `.pdf` | Eight-habitat comparison with other predicted classes aggregated; full row denominators retained |
| `figures/oracle_gap.png`, `.pdf` | Net gain, harm penalty and missed-rescue accounting |
| `figures/*_counts.*`, `*_normalized.*` | Complete 18-class matrices for fusion, RF, geometry, OR, AND and oracle |
| `summary.csv`, `per_seed_metrics.csv` | Exact aggregate and seed-level metrics and interventions |
| `habitat_interventions.csv`, `geometry_gate_audit.csv` | Rescue/harm breakdown by true habitat and geometry rejection reason |
| `predictions.parquet`, `metrics.json` | Matched predictions, cohort definitions, policies and confusion matrices |
| `provenance.json` | Source/code/output hashes and preservation checks |

The generated figures use the canonical CS class order and normalization.
The focused figure explicitly covers 911 images; full matrices and all
performance metrics use the complete cohort. Existing source bundles and
the locked-test registry remain unchanged. See [record 23](23_geometry_router_confusion_review.md)
for the more detailed confusion review and [record 21](21_embedding_geometry_correction.md)
for the original geometry studies.

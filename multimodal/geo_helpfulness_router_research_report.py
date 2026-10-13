"""Build a concise research report from authenticated saved router predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confusion_review import (
    BASE, REPO, RF_POLICY, SEEDS, combined_rules, confusion, digest, load_sources,
    outcomes, policy, rates, relative_link, table, write_json,
)
from multimodal.geo_helpfulness_router_research_figures import figures


STUDY = "router_research_summary_v1"
RECORD = REPO / "dev_plans/reliability_aware_geo_fusion/25_router_research_summary.md"
TRUSTSCORE_PAPER = "https://proceedings.neurips.cc/paper_files/paper/2018/file/7180cffd6a8e829dacfc2a31b3f72ece-Paper.pdf"
DESIGN_RECORDS = [REPO / "dev_plans/reliability_aware_geo_fusion" / name for name in (
    "01_experimental_protocol.md", "03_targets_and_features.md",
    "04_router_dataset_preparation.md", "06_router_capacity_exploration.md",
    "08_legacy_capacity_transfer.md", "10_relaxed_guardrails_rf.md",
    "21_embedding_geometry_correction.md",
)]


def analyze(rows, labels, threshold):
    _, _, _, predictions, _ = combined_rules(rows, policy(5) + "_pred", labels)
    fusion_ok = predictions.raw_pred.eq(predictions.label_id_dense)
    geo_ok = predictions.geo_pred.eq(predictions.label_id_dense)
    rescue = ~fusion_ok & geo_ok
    harm = fusion_ok & ~geo_ok
    predictions["oracle_acted"] = rescue
    predictions["oracle_pred"] = np.where(rescue, predictions.geo_pred, predictions.raw_pred)
    methods = {"fusion": ("raw_pred", "Fusion baseline"),
               "rf": ("rf_pred", "Output router (RF)"),
               "geometry": ("geometry_pred", "Adapted geometry router"),
               "or": ("or_pred", "Either router (OR)"),
               "and": ("and_pred", "Both routers (AND)"),
               "oracle": ("oracle_pred", "Rescue oracle")}
    opportunities = int(rescue.sum())
    summary, seeds, habitats, matrices, per_seed_matrices = [], [], [], {}, {}
    for model, (column, name) in methods.items():
        cms = [confusion(predictions.loc[predictions.training_seed.eq(seed)], column) for seed in SEEDS]
        per_seed_matrices[model] = {str(seed): cm.tolist() for seed, cm in zip(SEEDS, cms)}
        matrices[model] = np.stack(cms).sum(axis=0)
        values = []
        for seed, cm in zip(SEEDS, cms):
            part = predictions.loc[predictions.training_seed.eq(seed)]
            scores = rates(cm)
            metrics = {key: float(scores[key]) for key in ("accuracy", "weighted_f1", "macro_f1")}
            values.append(metrics)
            seeds.append({"model": model, "training_seed": seed, **metrics,
                          **outcomes(part, part[column], part.raw_pred)})
        counts = outcomes(predictions, predictions[column], predictions.raw_pred)
        assert counts["actions"] == counts["rescues"] + counts["harms"] + counts["both_wrong"]
        assert np.trace(matrices[model]) - fusion_ok.sum() == counts["net_correct"]
        summary.append({"model": model, "name": name,
                        **{key: float(np.mean([value[key] for value in values])) for key in values[0]},
                        **counts, "missed_rescues": opportunities - counts["rescues"],
                        "rescue_capture_fraction": counts["rescues"] / opportunities})
        for label, habitat in enumerate(labels):
            part = predictions.loc[predictions.label_id_dense.eq(label)]
            habitats.append({"model": model, "class_id": label, "habitat": habitat,
                             "unique_images": part.row_uid.nunique(),
                             **outcomes(part, part[column], part.raw_pred)})
    summary = pd.DataFrame(summary)
    oracle_acc = summary.set_index("model").loc["oracle", "accuracy"]
    summary["oracle_gap_pp"] = 100 * (oracle_acc - summary.accuracy)
    np.testing.assert_allclose(summary.oracle_gap_pp,
                               100 * (summary.missed_rescues + summary.harms) / len(rows), atol=1e-12)

    eligible = rows.k5_eligible.astype(bool)
    image_pass, geo_pass = rows.k5_A_image.gt(threshold), rows.k5_A_geo.gt(threshold)
    assert np.array_equal(predictions.geometry_acted,
                          predictions.disagreement & eligible & image_pass & geo_pass)
    groups = {"insufficient_reference_support": ~eligible,
              "image_rejects_geo_accepts": eligible & ~image_pass & geo_pass,
              "geo_rejects_image_accepts": eligible & image_pass & ~geo_pass,
              "both_reject": eligible & ~image_pass & ~geo_pass,
              "both_accept": eligible & image_pass & geo_pass}
    assert np.stack([mask.to_numpy() for mask in groups.values()]).sum(axis=0).tolist() == [1] * len(rows)
    geometry_audit = pd.DataFrame([
        {"gate_outcome": name, "rescue_opportunities": int((mask & rescue).sum()),
         "harm_opportunities": int((mask & harm).sum()),
         "both_wrong_disagreements": int((mask & ~fusion_ok & ~geo_ok & predictions.disagreement).sum())}
        for name, mask in groups.items()
    ])
    cohort = {"unique_images": rows.row_uid.nunique(), "unique_plots": rows.plot_idx.nunique(),
              "seed_image_decisions": len(rows), "disagreements": int(predictions.disagreement.sum()),
              "oracle_rescues": opportunities, "harm_opportunities": int(harm.sum()),
              "both_wrong_disagreements": int((~fusion_ok & ~geo_ok & predictions.disagreement).sum()),
              "both_wrong_total": int((~fusion_ok & ~geo_ok).sum()),
              "rescue_lost_only_to_positive_margin": int((rescue & eligible & rows.k5_A_image.gt(0)
                                                          & rows.k5_A_geo.gt(0) & ~predictions.geometry_acted).sum())}
    return (summary, pd.DataFrame(seeds), pd.DataFrame(habitats), geometry_audit,
            predictions, matrices, per_seed_matrices, cohort)


def router_design_section(output):
    """Keep mathematical notation literal while inserting portable source links."""
    section = r"""## 1. Task and two router designs

An **expert** predicts a habitat class. The image expert uses a ground photograph; the geo expert uses a satellite-derived embedding for its location; the fusion expert combines image and geo features. A **router** decides whether to retain fusion's predicted habitat or replace it with geo's. The two designs use different evidence: learned relationships among expert outputs, or proximity to labeled examples in embedding spaces.

### 1.1 Learning a router from expert outputs

Logistic regression, random forest (RF) and multilayer perceptron routers learn from features calculated from the three experts' probability distributions. **RF is the representative evaluated here.** Its full feature set is organized as follows:

| Feature group | Count | Examples and purpose |
| --- | ---: | --- |
| Predicted habitat identities | 5 | Each expert's predicted habitat, plus the image–geo and geo–fusion prediction pairs; identifies which candidate habitats are being compared. |
| Individual confidence and uncertainty | 9 | Each expert's top-class probability, entropy and top-two probability gap; describes how decisive each prediction is. |
| Relative confidence and uncertainty | 6 | Geo's confidence, entropy and probability gap minus those of image and fusion; compares the strength of their predictions. |
| Prediction agreement | 3 | Whether each pair of experts predicts the same habitat. |
| Support for alternative candidates | 5 | Image–geo top-three overlap, fusion's rank of geo's candidate, and the probability one expert assigns to another's candidate. |
| Distribution differences | 2 | Jensen–Shannon divergence and total variation between image and geo probabilities; measures disagreement beyond their top predictions. |

These are **30 engineered features**, expanded to **727 inputs** by one-hot encoding the five categorical features: 25 numeric/indicator values plus 702 categorical entries. Expert probabilities are temperature-calibrated using development-training predictions. No image or satellite embedding is supplied directly to this RF. [Exact feature definitions](FEATURE_SCHEMA_LINK).

**Training.** The 3,378 development-training images are divided into four folds by plot, keeping photographs of the same plot together. Experts trained on three folds predict the held-out fold. These **out-of-fold predictions** supply router features; comparison with the true habitat supplies one of four targets: **rescue** (fusion wrong, geo correct), **harm** (fusion correct, geo wrong), **both correct**, or **both wrong**. A separate RF is fitted for each training seed. Expert calibration also uses the training OOF rows; it is not a second independent holdout.

The RF estimates four state probabilities. If $v(x)$ is its feature vector, its intervention score is

$$
U(x)=\widehat P(\mathrm{rescue}\mid v(x))-
     \widehat P(\mathrm{harm}\mid v(x)).
$$

This estimates the accuracy benefit of switching: rescue contributes +1, harm −1, and the other states 0. On a fusion–geo disagreement, switch only if $U(x)>t_{\mathrm{RF}}$. The model is trained to predict correctness states, not a new habitat label. RF settings are selected by validation probability quality; the intervention threshold is selected by validation net rescue under the constraints below.

### 1.2 Constructing the geometry router

Let $x$ be the query image/location, $c_f$ fusion's candidate habitat and $c_g$ geo's candidate. Encode the query and labeled reference examples using the same encoder within each modality: a **habitat-adapted SigLIP2 image encoder** and fixed satellite embeddings. Each vector is L2-normalized separately.

For modality $m\in\{\mathrm{image},\mathrm{geo}\}$, find the **five closest distinct reference plots of habitat $c$**. Each plot contributes its closest reference image in that space. With unit-normalized embedding $z_m$ and selected references $N_m^K(x,c)$, define the mean cosine distance

$$
d_m(x,c)=\frac{1}{K}\sum_{j\in N_m^K(x,c)}
\left[1-z_m(x)^\top z_m(x_j)\right],\qquad K=5.
$$

The support for geo relative to fusion is

$$
A_m(x)=\log\frac{d_m(x,c_f)+\epsilon}{d_m(x,c_g)+\epsilon},
\qquad \epsilon=10^{-8}.
$$

**Positive $A_m$** means the query is closer to geo's candidate references; negative values favor fusion; zero indicates equal distance. The small ε prevents division by zero. These are relative comparisons within each space, not distances between image and satellite vectors or estimates of correctness probability.

Provided both candidate habitats have at least $K$ distinct reference plots, the rule is

$$
\begin{aligned}
S(x)&=\min\{A_{\mathrm{image}}(x),A_{\mathrm{geo}}(x)\},\\
\mathrm{switch\ to\ geo}&\Longleftrightarrow
(c_f\ne c_g)\ \land\ (S(x)>\tau).
\end{aligned}
$$

Thus **both modalities must favor geo by the required margin**; otherwise retain fusion, including when reference support is insufficient. The displayed configuration uses 4,200 development reference images and the validation-selected τ ≈ 0.0061. No classifier is fitted on these geometry scores: construction requires the labeled reference bank, distance calculation and one selected threshold.

![Illustrative geometry-router decision in image and geo spaces](figures/geometry_router_schematic.png)

*Figure 1. Hypothetical example, not measured test data or a projection. The same two candidate habitats are compared separately in each modality. Mean distances of 0.30 versus 0.10 give $A_{\mathrm{image}}\approx1.10$; 0.20 versus 0.08 give $A_{\mathrm{geo}}\approx0.92$. Their minimum exceeds τ, so the rule switches to geo. If either score failed the margin, it would retain fusion. Drawn positions illustrate neighborhood support; the displayed values represent mean cosine distances in the original embedding spaces.*

**Connection to TrustScore.** Jiang, Kim, Guan and Gupta's [*To Trust Or Not To Trust A Classifier* (NeurIPS 2018)](TRUSTSCORE_PAPER_LINK) compares distance to the nearest alternative class with distance to the predicted class, using density-filtered reference sets in its original algorithm. A high ratio supports the classifier's prediction. Our rule adopts that relative-distance idea but compares **two specified expert candidates**, uses a log ratio **positive in favor of geo**, averages five distinct plots, and requires support from **both modalities**. We do not apply the original density filter. This is a TrustScore-inspired routing rule, not the original TrustScore estimator or a transfer of its theoretical guarantees.

### 1.3 Threshold fitting and test application

Both thresholds were fitted on **822 development-validation images**, maximizing net rescues subject to **H≤0.7R** and minimum intervention coverage/plot support. This permits harmful interventions; it is not a zero-harm rule. The geometry encoder was trained on the **3,378 development-training images**. Geometry threshold fitting used only those training references; cleaned-test application expanded the bank to all 4,200 development images. The fixed expert heads used for cleaned-test classification were fitted on all 4,200 development images. Routers, expert calibration and thresholds remain frozen during test application.
"""
    schema = REPO / "dev_plans/reliability_aware_geo_fusion/03_targets_and_features.md"
    return section.replace("FEATURE_SCHEMA_LINK", relative_link(schema, output)).replace(
        "TRUSTSCORE_PAPER_LINK", TRUSTSCORE_PAPER)


def report(output, summary, habitats, audit, cohort, matrices, labels):
    stats = summary.set_index("model")
    gate = audit.set_index("gate_outcome")
    habitat = habitats.set_index(["model", "class_id"])
    performance = table(["Method", "Acc@1 (%)", "Weighted F1", "Interventions", "Rescues", "Harms"],
                        [[row['name'], f"{100 * row['accuracy']:.2f}", f"{row['weighted_f1']:.3f}",
                          *[int(row[key]) for key in ("actions", "rescues", "harms")]]
                         for _, row in summary.iterrows()])
    important_pairs = [(4, 5), (5, 4), (11, 9), (10, 5)]
    confusions = table(["True habitat → mistaken prediction", "Fusion (%)", "RF (%)", "Geometry (%)"],
                       [[f"{labels[a]} → {labels[b]}",
                         *[f"{100 * matrices[model][a, b] / matrices[model][a].sum():.1f}"
                           for model in ("fusion", "rf", "geometry")]] for a, b in important_pairs])
    root_docs = REPO / "dev_plans/reliability_aware_geo_fusion"
    content = f"""# Correcting habitat fusion predictions: output-based and geometry routers

Research summary · 6 October 2026

**Main finding.** Both routers improve the observed accuracy of image–geography fusion, but recover only a small part of its correctable errors. Combining their interventions increases accuracy to **{100 * stats.loc['or', 'accuracy']:.2f}%**, compared with **{100 * stats.loc['fusion', 'accuracy']:.2f}%** for fusion and an **{100 * stats.loc['oracle', 'accuracy']:.2f}% oracle ceiling**. Missed rescues explain most of the remaining gap; harmful switches further reduce the gains.

{router_design_section(output)}

## 2. Application and cleaned-test performance

Routing is relevant only when fusion and geo disagree. Apply RF or geometry separately, or combine their binary decisions: **OR** switches when either says yes; **AND** switches only when both say yes. Both routers choose the same geo label on shared interventions. All other cases retain fusion. Query labels are unavailable to these rules.

The evaluation uses **{cohort['unique_images']:,} cleaned-test images from {cohort['unique_plots']} plots**, repeated across four training seeds. Acc@1 is top-label accuracy; weighted F1 averages habitat F1 scores in proportion to their test support. Metrics below are means across seeds. **Counts pool {cohort['seed_image_decisions']:,} seed–image decisions, not distinct photographs.** A rescue corrects a fusion error; a harm turns a correct fusion prediction into an error.

{performance}

Interventions also include changes between two wrong labels: **52 for RF, 56 for geometry, 98 for OR and 10 for AND**. The rescue oracle uses the true label to switch exactly when fusion is wrong and geo is correct; it retains fusion otherwise. It is an unattainable comparison at inference, and its accuracy is the ceiling for choosing between these fixed experts.

RF and geometry share **99 interventions**; their union has **458**. OR adds 68 rescues and 55 harms to geometry: a net gain of 13 correct decisions, or **+0.24 percentage points**. AND avoids 53 of geometry's harms but also loses 95 rescues. Agreement reduces intervention frequency without guaranteeing correctness. Although OR leads in accuracy and weighted F1, RF retains higher macro F1, which weights habitats equally: **0.504 versus 0.500**.

## 3. Confusion matrices: improvements have a direction

![Confusion comparison for grassland, heath and wetland habitats](figures/confusion_focus_comparison.png)

*Figure 2. Fusion, RF correction and geometry correction on eight grassland, heath and wetland classes, covering 911 of 1,347 images. Rows are true habitats and columns are predictions; each row uses its full test support as denominator. Predictions outside the displayed classes are grouped as “Other.” The common 0–1 scale shows recall on the diagonal. [Full fusion](figures/fusion_normalized.png), [RF](figures/rf_normalized.png) and [geometry](figures/geometry_normalized.png) matrices retain all 18 classes.*

{confusions}

These are percentages of the true-habitat row. Both routers reduce Improved→Neutral Grassland errors while increasing errors in the reverse direction. Geometry reduces Bog→Dwarf Shrub Heath confusion but worsens Fen/Marsh/Swamp→Neutral Grassland. Its Bog interventions produce **{int(habitat.loc[('geometry', 11), 'rescues'])} rescues and {int(habitat.loc[('geometry', 11), 'harms'])} harms**, whereas Fen/Marsh/Swamp produces **{int(habitat.loc[('geometry', 10), 'rescues'])} rescue and {int(habitat.loc[('geometry', 10), 'harms'])} harms**. Aggregate improvement therefore does not imply uniformly better habitat decisions.

## 4. Why the oracle gap remains large

**The candidate pool is unfavorable.** Among **{cohort['disagreements']:,} disagreements**, only **{cohort['oracle_rescues']}** can be rescued by geo; **{cohort['harm_opportunities']:,}** would be harmed by switching, and **{cohort['both_wrong_disagreements']}** have neither expert correct. A plausible geo alternative is therefore insufficient evidence to intervene. Even the oracle leaves **{cohort['both_wrong_total']}** errors where neither expert supplies the correct answer.

![Accounting for the gap to the rescue oracle](figures/oracle_gap.png)

*Figure 3. The {cohort['oracle_rescues']} available oracle corrections equal net gain achieved + missed rescues + harm penalty. These are contributions to the accuracy gap, not three disjoint subsets of interventions. For geometry: **75 + 315 + 81 = 471**. Counts again pool four seeds.*

**Missing rescues is the larger limitation.** RF captures **129/471 (27.4%)** of the available rescues; geometry captures **156/471 (33.1%)**. Their remaining accuracy gaps equal missed rescues plus harms: **342 + 83** for RF and **315 + 81** for geometry. About **80%** of each gap is missed rescues. Even removing all 81 geometry harms without finding any new rescues would raise accuracy only to **{100 * (stats.loc['geometry', 'accuracy'] + stats.loc['geometry', 'harms'] / cohort['seed_image_decisions']):.2f}%**, still well below the oracle.

**Loosening RF does not necessarily help.** In the earlier matched [RF study]({relative_link(root_docs / '10_relaxed_guardrails_rf.md', output)}), relaxing the validation harm budget from 0.7R to 0.9R added **37 test rescues but 42 harms**. More interventions reduced the net gain, consistent with imperfect separation of rescue and harm candidates.

**Geometry's two-modality requirement trades recall for protection.** Only **{int(gate.loc['insufficient_reference_support', 'rescue_opportunities'])}** of its missed rescues lack five reference plots. In **{int(gate.loc['image_rejects_geo_accepts', 'rescue_opportunities'])}** real rescue cases, geo geometry passes but image geometry rejects the switch. Yet that same image gate blocks **{int(gate.loc['image_rejects_geo_accepts', 'harm_opportunities'])}** harmful switches among geo-approved candidates. Simply removing the gate would therefore be costly. Only **{cohort['rescue_lost_only_to_positive_margin']}** additional rescue is lost solely to requiring the small positive margin rather than zero. For this evaluated rule, the main limitation is its imperfect correctness signal, rather than insufficient reference support or this small margin.

**Shared mistakes limit the value of consensus.** AND still makes **28 harms**, including **{int(habitat.loc[('and', 10), 'harms'])} in Fen/Marsh/Swamp**. Expert confidence and class-neighborhood proximity are evidence of plausibility, not direct observations of correctness. Both can favor the same wrong candidate. The observed pattern is consistent with shared ambiguity; it does not establish that embedding overlap, label noise or a particular training choice caused these errors.

The evidence points toward improving **rescue-versus-harm discrimination**, rather than simply accepting more cases or demanding agreement. Class-dependent reliability and combining output and geometric evidence are reasonable development-stage hypotheses. They need validation before choosing a rule; the OR/AND results already show the cost of expanding or restricting interventions indiscriminately.

## Scope and supporting material

This is an exploratory analysis of existing predictions. OR/AND were proposed after inspecting test overlap. Earlier paired accuracy intervals for geometry versus fusion and RF include zero; these point estimates do not establish independent superiority. No threshold or model was fitted for this report. The causes proposed above are separated from the directly measured counts.

The report extends the [September 15 router summary]({relative_link(root_docs / 'reports/2026-09-15_geo_helpfulness_router/report.md', output)}). Details remain in the [RF study]({relative_link(root_docs / '10_relaxed_guardrails_rf.md', output)}), [geometry study]({relative_link(root_docs / '21_embedding_geometry_correction.md', output)}) and [full confusion review](../geometry_router_confusion_review_v1/report.md). [Exact results](summary.csv), [per-seed results](per_seed_metrics.csv), [habitat rescue/harm counts](habitat_interventions.csv), [geometry rejection audit](geometry_gate_audit.csv) and [reproduction instructions]({relative_link(RECORD, output)}) accompany this report.
"""
    (output / "report.md").write_text(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", default=str(BASE / STUDY))
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output = (REPO / output).resolve() if not output.is_absolute() else output.resolve()
    if output.parent != BASE.resolve() or not output.name.startswith("router_research_"):
        parser.error("Output must be a router_research_* directory under the canonical artifact root")
    if output.exists() and any(output.iterdir()):
        provenance = output / "provenance.json"
        if not provenance.exists() or json.loads(provenance.read_text()).get("study_id") != STUDY:
            parser.error("Refusing to overwrite a directory not owned by this report")
    registry = REPO / "multimodal_artifacts/locked_test_registry/cs/gse_100m_cleaned_test/events"
    before = {str(p.relative_to(REPO)): digest(p) for p in registry.rglob("*") if p.is_file()}
    banks, policies, labels, ontology, saved, _, _, hashes = load_sources()
    for path in DESIGN_RECORDS:
        hashes[str(path.relative_to(REPO))] = digest(path)
    threshold = float.fromhex(policies[policy(5)]["threshold_specification"]["value_hex"])
    result = analyze(banks[4200], labels, threshold)
    summary, seeds, habitats, audit, predictions, matrices, seed_matrices, cohort = result
    for model, original in (("fusion", "fusion"), ("rf", "rf"), ("geometry", policy(5))):
        expected = saved.loc[saved.cohort.eq("test_4200") & saved.model.eq(original)].sort_values("training_seed")
        actual = seeds.loc[seeds.model.eq(model)].sort_values("training_seed")
        for metric in ("accuracy", "weighted_f1", "macro_f1"):
            np.testing.assert_allclose(actual[metric], expected[metric], atol=1e-12, rtol=0)
    output.mkdir(exist_ok=True)
    for name, frame in (("summary", summary), ("per_seed_metrics", seeds),
                        ("habitat_interventions", habitats), ("geometry_gate_audit", audit)):
        frame.to_csv(output / f"{name}.csv", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    write_json(output / "metrics.json", {"study_id": STUDY, "cohort": cohort, "ontology": ontology,
               "reference_bank": 4200, "k": 5, "rf_policy": RF_POLICY, "geometry_policy": policy(5),
               "geometry_threshold": threshold,
               "oracle_definition": "Switch only when fusion is wrong and geo is correct; otherwise retain fusion",
               "summary": summary.astype(object).where(pd.notna(summary), None).to_dict("records"),
               "confusion_matrices": {key: value.tolist() for key, value in matrices.items()},
               "per_seed_confusion_matrices": seed_matrices})
    figures(output, matrices, labels, summary.loc[summary.model.isin(["rf", "geometry", "or", "and"])])
    report(output, summary, habitats, audit, cohort, matrices, labels)
    after = {str(p.relative_to(REPO)): digest(p) for p in registry.rglob("*") if p.is_file()}
    assert before == after
    assert all(digest(REPO / name) == value for name, value in hashes.items())
    code = [Path(__file__), REPO / "multimodal/geo_helpfulness_router_research_figures.py",
            REPO / "multimodal/geo_helpfulness_confusion_review.py",
            REPO / "tools/plot_multimodal_geo_helpfulness_router_research_report.py"]
    write_json(output / "provenance.json", {
        "study_id": STUDY, "source_sha256": hashes,
        "external_references": [{"title": "To Trust Or Not To Trust A Classifier",
                                 "authors": ["Heinrich Jiang", "Been Kim", "Melody Y. Guan", "Maya Gupta"],
                                 "venue": "NeurIPS 2018", "url": TRUSTSCORE_PAPER,
                                 "relevant_section": "Section 3, Algorithms 1 and 2"}],
        "implementation_sha256": {str(p.relative_to(REPO)): digest(p) for p in code},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "registry_unchanged": True, "registry_files": len(before), "source_files_unchanged": True,
        "model_fitting": False, "threshold_selection": False, "independent_confirmation": False,
        "output_sha256": {str(p.relative_to(output)): digest(p) for p in output.rglob("*")
                          if p.is_file() and p.name != "provenance.json"},
    })
    print(summary[["name", "accuracy", "weighted_f1", "rescues", "harms"]].to_string(index=False))
    print(f"Report: {output / 'report.md'}")
    return 0

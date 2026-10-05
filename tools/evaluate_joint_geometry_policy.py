"""Evaluate a fixed joint-geometry geo override on development validation.

Run from the repository root. Reads the previous geometry study and saved
development-train expert outputs; does not fit models or choose thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal.geo_helpfulness_oof import read_output_parquet
from multimodal.geo_helpfulness_protocol import canonical_sha256, plot_set_fingerprint
from multimodal.geo_helpfulness_router_metrics import classification_metrics, policy_metrics


STUDY = Path("dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_geometry_disagreement_check")
PROTOCOL_ROOT = Path("multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1")
DECLARATION = {
    "policy": "joint_geometry_positive_geo_override",
    "cohort": "development_validation",
    "expert_fit": "development_train",
    "distance_k": 1,
    "action": "raw_pred != geo_pred AND finite(A_image, A_geo) AND A_image > 0 AND A_geo > 0",
    "corrected_prediction": "geo_pred when action, otherwise raw_pred",
    "ties_or_missing": "retain fusion",
    "geometry": "reuse frozen SigLIP2 / L2-normalized regional geo cosine class distances",
    "references": "development_train only; query plots excluded",
    "threshold_selection": False,
    "model_fitting": False,
    "cleaned_test_access": False,
    "classification_ontology": "18 fixed dense classes; absent-class F1/recall set to zero",
    "bootstrap": {
        "replicates": 2000,
        "random_seed": 20260930,
        "unit": "whole plot, retaining all photos and all four expert seeds",
        "stratify_by": "canonical_l3_id",
        "interval": "95% percentile, linear quantiles, mean-seed accuracy difference",
        "interpretation": "conditional on fixed split, trained experts, and this development-derived policy",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = STUDY / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {str(manifest_path): digest(manifest_path)}
    for name, expected in manifest["source_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Geometry source changed: {name}")
        hashes[name] = expected
    score_path = STUDY / "geometry_scores.parquet"
    hashes[str(score_path)] = digest(score_path)
    if hashes[str(score_path)] != manifest["output_sha256"][score_path.name]:
        raise ValueError("Geometry scores changed")
    if manifest["declaration"]["primary_distance_k"] != 1:
        raise ValueError("Expected the previous primary nearest-reference definition")
    rows = pd.read_parquet(
        score_path, filters=[("cohort", "=", "validation"), ("distance_k", "=", 1)]
    ).sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    assignments = pd.read_parquet(PROTOCOL_ROOT / "protocol/development_assignments.parquet")
    train = assignments.loc[assignments.development_role.eq("train")]
    validation = assignments.loc[assignments.development_role.eq("validation")]
    if set(train.plot_idx) & set(validation.plot_idx):
        raise ValueError("Training and validation plots overlap")
    if set(rows.training_seed) != {1, 2, 3, 4}:
        raise ValueError("Four expert seeds required")
    if rows.duplicated(["training_seed", "row_uid"]).any():
        raise ValueError("Duplicate seed/query rows")
    for seed, seed_rows in rows.groupby("training_seed"):
        folder = PROTOCOL_ROOT / f"development_validation/seed_{seed}"
        producer = json.loads((folder / "manifest.json").read_text())
        payload = {k: v for k, v in producer.items() if k != "manifest_sha256"}
        if canonical_sha256(payload) != producer["manifest_sha256"]:
            raise ValueError("Invalid producer manifest")
        if not (producer["stage_id"] == "development_train_to_validation"
                and producer["fitting_row_count"] == len(train)
                and producer["row_count"] == len(validation)
                and producer["fitting_plot_hash"] == plot_set_fingerprint(train.plot_idx)
                and producer["prediction_plot_hash"] == plot_set_fingerprint(validation.plot_idx)
                and producer["zero_plot_overlap"]):
            raise ValueError("Expert fitting/prediction provenance mismatch")
        native = read_output_parquet(folder / producer["model_output_filename"], include_fold=False)
        native = native.to_pandas().sort_values("row_uid").reset_index(drop=True)
        seed_rows = seed_rows.reset_index(drop=True)
        for column in ("row_uid", "plot_idx", "file", "raw_pred", "geo_pred"):
            if not np.array_equal(seed_rows[column], native[column]):
                raise ValueError(f"Cached/native prediction mismatch: {column}")
        expected = validation.set_index("row_uid").loc[seed_rows.row_uid]
        if len(seed_rows) != len(validation) or set(seed_rows.row_uid) != set(validation.row_uid):
            raise ValueError("Incomplete validation cohort")
        for column in ("label_id_dense", "plot_idx", "label_name"):
            if not np.array_equal(seed_rows[column], expected[column]):
                raise ValueError(f"Validation identity mismatch: {column}")
    rows["canonical_l3_id"] = rows.row_uid.map(assignments.set_index("row_uid").canonical_l3_id)
    rows["target_state"] = rows.outcome
    return rows, assignments, hashes


def accuracy_bootstrap(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pair both predictions and all seed realizations within sampled plots."""
    plots = rows.drop_duplicates("plot_idx").sort_values("plot_idx")
    plot_ids = plots.plot_idx.tolist()
    n = rows.groupby(["plot_idx", "training_seed"]).size().unstack().reindex(plot_ids)
    net = rows.groupby(["plot_idx", "training_seed"]).correctness_change.sum().unstack().reindex(plot_ids)
    if n.isna().any().any() or net.isna().any().any():
        raise ValueError("Incomplete plot/seed coverage")
    strata = [np.flatnonzero(plots.canonical_l3_id.to_numpy() == c)
              for c in sorted(plots.canonical_l3_id.unique())]
    settings = DECLARATION["bootstrap"]
    rng = np.random.default_rng(settings["random_seed"])
    weights = np.zeros((settings["replicates"], len(plot_ids)), dtype=np.int64)
    for replicate in weights:
        for indices in strata:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            replicate += np.bincount(sampled, minlength=len(plot_ids))
    delta = ((weights @ net.to_numpy()) / (weights @ n.to_numpy())).mean(axis=1)
    lower, upper = np.quantile(delta, [.025, .975], method="linear")
    result = {
        **settings,
        "point_estimate": float((net.sum() / n.sum()).mean()),
        "lower": float(lower), "upper": float(upper),
        "plots": len(plot_ids),
        "resample_multiplicity_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
    }
    return result, pd.DataFrame({"replicate": np.arange(len(delta)), "delta_accuracy": delta})


def evaluate(rows: pd.DataFrame, assignments: pd.DataFrame) -> tuple:
    rows = rows.copy()
    finite = np.isfinite(rows[["A_image", "A_geo"]]).all(axis=1)
    rows["intervene"] = rows.raw_pred.ne(rows.geo_pred) & finite & rows.A_image.gt(0) & rows.A_geo.gt(0)
    rows["corrected_pred"] = np.where(rows.intervene, rows.geo_pred, rows.raw_pred).astype(np.int64)
    rows["correctness_change"] = (
        rows.corrected_pred.eq(rows.label_id_dense).astype(int)
        - rows.raw_pred.eq(rows.label_id_dense).astype(int)
    )
    policy = policy_metrics(rows, rows.intervene.to_numpy())
    summaries, per_class, detailed = [], [], {}
    names = assignments.drop_duplicates("label_id_dense").set_index("label_id_dense").label_name
    for seed, data in rows.groupby("training_seed"):
        baseline = classification_metrics(data.raw_pred.to_numpy(), data.label_id_dense.to_numpy())
        corrected = classification_metrics(data.corrected_pred.to_numpy(), data.label_id_dense.to_numpy())
        p = policy["per_seed"][str(seed)]
        record = {"training_seed": int(seed), **p,
                  "disagreement_count": int(data.raw_pred.ne(data.geo_pred).sum()),
                  "both_wrong_overrides": int((data.intervene & data.outcome.eq("both_wrong")).sum()),
                  "unavailable_disagreements": int((data.disagreement & ~finite.loc[data.index]).sum())}
        for key in ("accuracy", "macro_f1", "weighted_f1", "mcc"):
            record.update({f"fusion_{key}": baseline[key], f"corrected_{key}": corrected[key],
                           f"delta_{key}": corrected[key] - baseline[key]})
        if p["rescued_count"] + p["harmed_count"] + record["both_wrong_overrides"] != p["acted_count"]:
            raise ValueError("Override outcome counts do not sum")
        summaries.append(record)
        detailed[str(seed)] = {"fusion": baseline, "corrected": corrected, "policy": p}
        for before, after in zip(baseline["per_class"], corrected["per_class"]):
            c = before["class_id"]
            selected = data.loc[data.label_id_dense.eq(c)]
            class_record = {"training_seed": int(seed), "class_id": c, "label_name": names[c],
                            "support": before["support"],
                            "rescues": int(selected.correctness_change.eq(1).sum()),
                            "harms": int(selected.correctness_change.eq(-1).sum())}
            for key in ("precision", "recall", "f1", "predicted_count"):
                class_record.update({f"fusion_{key}": before[key], f"corrected_{key}": after[key],
                                     f"delta_{key}": after[key] - before[key]})
            per_class.append(class_record)
    return rows, pd.DataFrame(summaries), pd.DataFrame(per_class), detailed


def write_report(output: Path, summary: pd.DataFrame, interval: dict) -> None:
    mean = summary.mean(numeric_only=True)
    lines = ["# Joint geometry correction on development validation", "",
             f"The fixed policy changes mean accuracy from {100 * mean.fusion_accuracy:.4f}% to "
             f"{100 * mean.corrected_accuracy:.4f}% ({100 * mean.delta_accuracy:+.4f} percentage points) "
             "across four saved expert seeds.", "",
             "## Policy and evaluation", "",
             "On fusion/geo disagreements, replace fusion with geo exactly when both A_image > 0 "
             "and A_geo > 0. Keep fusion for ties or unavailable scores. This is the user's fixed "
             "rule, with no threshold search or new fitting.", "",
             "Use the preceding geometry study's primary k=1 distance: cosine distance to the nearest "
             "reference plot of each candidate class (minimum over that plot's photos). "
             "Both A scores are log((distance to fusion class + 1e-8) / (distance to geo class + 1e-8)); "
             "positive means geo is favored. Frozen pretrained SigLIP2 image features and regional "
             "geo features are independently L2-normalized. Geometry references are development-train only.", "",
             "Each expert was fitted on all 3,378 development-training images from 1,300 plots. "
             "Evaluation includes all 822 development-validation images from 325 disjoint plots, "
             "including expert agreements. All four seeds use the same validation identities. "
             "Producer stage, fitting/prediction plot fingerprints, native outputs and source hashes "
             "are checked. No cleaned-test files are read.", "",
             "## Overall classification", "",
             "| Seed | Fusion accuracy | Corrected accuracy | Change (pp) | Fusion macro F1 | Corrected macro F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records") + [{**mean.to_dict(), "training_seed": "Mean"}]:
        lines.append(f"| {row['training_seed']} | {100*row['fusion_accuracy']:.3f}% | "
                     f"{100*row['corrected_accuracy']:.3f}% | {100*row['delta_accuracy']:+.3f} | "
                     f"{row['fusion_macro_f1']:.4f} | {row['corrected_macro_f1']:.4f} |")
    lines += ["", "Macro F1 includes all 18 classes with zero for absent-class F1; "
              "Littoral Rock has no validation examples. Weighted F1 and MCC are also saved.", "",
              f"Mean weighted F1: {mean.fusion_weighted_f1:.4f} -> {mean.corrected_weighted_f1:.4f}; "
              f"mean MCC: {mean.fusion_mcc:.4f} -> {mean.corrected_mcc:.4f}.", "",
              "Accuracy improves in all four seeds, but macro F1 declines in seeds 3 and 4. "
              "Rare-class changes contribute strongly: on Urban (three validation images), "
              "the policy loses three correct predictions in seed 3 and two in seed 4. "
              "The mean macro-F1 improvement therefore does not describe a uniform gain across habitats.", "",
              "## Intervention outcomes", "",
              "Rescue = wrong to correct; harm = correct to wrong; both-wrong override = "
              "one incorrect candidate replaced with another. All three count as interventions.", "",
              "| Seed | Disagreements | Overrides | Rescues | Harms | Both wrong | Net rescue minus harm |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records") + [{**mean.to_dict(), "training_seed": "Mean"}]:
        cells = [str(row["training_seed"])] + [f"{row[k]:g}" for k in (
            "disagreement_count", "acted_count", "rescued_count", "harmed_count",
            "both_wrong_overrides", "rescued_minus_harmed")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"Mean coverage is {100*mean.coverage_fraction:.2f}% of validation images. "
              f"The policy captures {100*mean.rescue_recall:.2f}% of available rescue opportunities. "
              f"Of overrides, the mean rescued fraction is {100*mean.override_precision:.2f}% "
              f"and the mean harmed fraction is {100*mean.harmful_override_rate:.2f}%. "
              "These are means of within-seed rates. Repeated seed realizations are not independent images.", "",
              "## Uncertainty and interpretation", "",
              f"The 95% paired habitat-stratified plot-bootstrap interval for the mean accuracy change "
              f"is [{100*interval['lower']:+.3f}, {100*interval['upper']:+.3f}] percentage points "
              "(2,000 draws, retaining all photos and seeds within each sampled plot). "
              "It is conditional on this split, the fitted experts and policy; it does not account "
              "for development-driven policy choice, training variation beyond these seeds or geographic shift.", "",
              "The rule is a plausible rescue-seeking heuristic, but geometric consensus does not guarantee "
              "correctness. Absolute distances, third classes and class-density effects are not used. "
              "Both-wrong cases can change per-class F1 without changing accuracy. "
              "Identical geo features across distinct plots and legacy image-cache preprocessing remain "
              "limitations described in the preceding geometry report. This development comparison is not "
              "independent confirmation and does not revise the original M4 decision.", "",
              "## Artifacts", "",
              "- [Per-seed comparison](performance_by_seed.csv)",
              "- [Per-habitat comparison](performance_by_habitat.csv)",
              "- [Row-level actions and predictions](validation_policy_predictions.parquet)",
              "- [Override records](interventions.csv)",
              "- [Full classification metrics and confusion counts](metrics.json)",
              "- [Bootstrap details](accuracy_bootstrap.json)",
              "- [Source geometry study](../2026-09-30_geometry_disagreement_check/report.md)", "",
              "Reproduce from the repository root:", "", "```bash",
              "python tools/evaluate_joint_geometry_policy.py --output-dir <new-output-directory>", "```", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(args.output_dir / "declaration.json", DECLARATION)
    rows, assignments, hashes = load_inputs()
    rows, summary, per_class, detailed = evaluate(rows, assignments)
    interval, replicates = accuracy_bootstrap(rows)
    rows.to_parquet(args.output_dir / "validation_policy_predictions.parquet", index=False)
    rows.loc[rows.intervene].to_csv(args.output_dir / "interventions.csv", index=False)
    summary.to_csv(args.output_dir / "performance_by_seed.csv", index=False)
    per_class.to_csv(args.output_dir / "performance_by_habitat.csv", index=False)
    replicates.to_csv(args.output_dir / "accuracy_bootstrap_replicates.csv", index=False)
    save_json(args.output_dir / "metrics.json", detailed)
    save_json(args.output_dir / "accuracy_bootstrap.json", interval)
    write_report(args.output_dir, summary, interval)
    for name, expected in hashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Source changed during evaluation: {name}")
    implementation = [Path(__file__), Path("multimodal/geo_helpfulness_router_metrics.py")]
    save_json(args.output_dir / "manifest.json", {
        "declaration": DECLARATION, "source_sha256": hashes,
        "implementation_sha256": {str(p): digest(p) for p in implementation},
        "output_sha256": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    })
    print(summary[["training_seed", "fusion_accuracy", "corrected_accuracy", "delta_accuracy",
                   "acted_count", "rescued_count", "harmed_count", "both_wrong_overrides"]].to_string(index=False))
    print(json.dumps(interval, indent=2))


if __name__ == "__main__":
    main()

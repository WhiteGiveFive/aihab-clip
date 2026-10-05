"""Check frozen embedding support for geo corrections on development queries.

This descriptive study reads saved expert predictions and fits no router.
Run from the repository root with the habcls environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal.geo_helpfulness_oof import logical_table_sha256, read_output_parquet
from multimodal.geo_helpfulness_protocol import canonical_sha256, plot_set_fingerprint
from plot_multimodal_embedding_geometry import (
    PROTOCOL, load_development, package_versions, sha256,
)


ROOT = PROTOCOL.parent
EPSILON = 1e-8
DISTANCE_K = (1, 5)
N_CLASSES = 18
DECLARATION = {
    "study": "frozen_geometry_disagreement_check",
    "primary_distance_k": 1,
    "sensitivity_distance_k": 5,
    "distance": "cosine distance on independently L2-normalized image and geo vectors",
    "plot_distance": "minimum query-to-image distance within each reference plot",
    "class_distance": "mean distance to k nearest distinct reference plots of the class",
    "reference_bank": "other development training folds for OOF; all development training for validation",
    "insufficient_reference_support": "missing when candidate class has fewer than k reference plots",
    "score": "log((distance_to_fusion_class + epsilon)/(distance_to_geo_class + epsilon))",
    "epsilon": EPSILON,
    "score_sign": "positive favors geo candidate in BOTH modalities",
    "outcome_groups": ["rescue", "harm", "both_wrong"],
    "primary_comparison": "rescue versus harm AUROC of each score, separately by cohort and seed",
    "sensitivity_comparison": "k=1 versus k=5 on the same k=5-supported queries",
    "seed_handling": "each seed separately, then arithmetic mean and range; no independence claim",
    "reference_density_filter": None,
    "threshold_selection": False,
    "router_fitting": False,
    "cleaned_test_opened": False,
}


def sealed_json(path: Path, hashes: dict) -> dict:
    hashes[str(path)] = sha256(path)
    doc = json.loads(path.read_text())
    payload = {key: value for key, value in doc.items() if key != "manifest_sha256"}
    if canonical_sha256(payload) != doc["manifest_sha256"]:
        raise ValueError(f"Manifest self-hash mismatch: {path}")
    return doc


def prediction_inputs(metadata: pd.DataFrame, hashes: dict) -> pd.DataFrame:
    frames = []
    for cohort, directory in (("train_oof", "development_train_oof"),
                              ("validation", "development_validation")):
        is_oof = cohort == "train_oof"
        folder = ROOT / directory
        manifest = sealed_json(folder / "aggregate_manifest.json", hashes)
        parent_hash = manifest["parent_artifact_hashes"]["development_assignments"]["file_sha256"]
        if parent_hash != hashes[str(PROTOCOL / "development_assignments.parquet")]:
            raise ValueError("Predictions refer to different assignments")
        path = folder / manifest["model_output_filename"]
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != manifest["model_output_file_sha256"]:
            raise ValueError(f"Prediction byte hash mismatch: {path}")
        table = read_output_parquet(path, include_fold=is_oof)
        if logical_table_sha256(table) != manifest["content_sha256"]:
            raise ValueError("Prediction logical hash mismatch")
        full = table.to_pandas()
        role = "train" if is_oof else "validation"
        expected = metadata.loc[metadata.development_role.eq(role)]
        if set(full.training_seed) != {1, 2, 3, 4}:
            raise ValueError("Expected all four expert seeds")
        for seed, rows in full.groupby("training_seed"):
            if len(rows) != len(expected) or set(rows.row_uid) != set(expected.row_uid):
                raise ValueError(f"Query identities differ for {cohort}/{seed}")
        aligned = metadata.set_index("row_uid").loc[full.row_uid]
        identity_columns = ["file", "file_lower", "plot_idx"]
        if is_oof:
            identity_columns.append("train_oof_fold")
        for column in identity_columns:
            if not np.array_equal(full[column].to_numpy(), aligned[column].to_numpy()):
                raise ValueError(f"Prediction identity mismatch: {column}")

        for relative, expected_digest in manifest["producer_manifest_sha256"].items():
            producer_folder = ROOT / relative
            producer = sealed_json(producer_folder / "manifest.json", hashes)
            if producer["manifest_sha256"] != expected_digest:
                raise ValueError("Producer manifest mismatch")
            fold = producer["train_oof_fold"]
            fitting = metadata.loc[metadata.development_role.eq("train")]
            query = expected
            if is_oof:
                fitting = fitting.loc[fitting.train_oof_fold.ne(fold)]
                query = expected.loc[expected.train_oof_fold.eq(fold)]
            if set(fitting.plot_idx) & set(query.plot_idx):
                raise ValueError("Expert fitting/query plot overlap")
            if not producer["zero_plot_overlap"]:
                raise ValueError("Producer reports plot overlap")
            for key, data in (("fitting", fitting), ("prediction", query)):
                if producer[f"{key}_plot_hash"] != plot_set_fingerprint(data.plot_idx):
                    raise ValueError("Producer plot fingerprint mismatch")
            producer_path = producer_folder / producer["model_output_filename"]
            hashes[str(producer_path)] = sha256(producer_path)
            if hashes[str(producer_path)] != producer["model_output_file_sha256"]:
                raise ValueError("Producer prediction byte hash mismatch")
            producer_table = read_output_parquet(producer_path, include_fold=is_oof)
            if logical_table_sha256(producer_table) != producer["content_sha256"]:
                raise ValueError("Producer prediction logical hash mismatch")
            seed_rows = full.loc[full.training_seed.eq(producer["training_seed"])]
            if is_oof:
                seed_rows = seed_rows.loc[seed_rows.train_oof_fold.eq(fold)]
            producer_frame = producer_table.to_pandas()
            pd.testing.assert_frame_equal(
                seed_rows.reset_index(drop=True), producer_frame.reset_index(drop=True),
                check_exact=True,
            )

        keep = ["row_uid", "training_seed", "image_pred", "geo_pred", "raw_pred"]
        frames.append(full[keep].assign(cohort=cohort))
        print(f"Authenticated {cohort}: {len(full):,} saved image-seed predictions", flush=True)
    return pd.concat(frames, ignore_index=True)


def reference_distances(metadata: pd.DataFrame, embeddings: dict) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Query labels are not used in distance computation; reference labels are."""
    distances = {m: np.full((len(metadata), len(DISTANCE_K), N_CLASSES), np.nan)
                 for m in embeddings}
    support = np.zeros((len(metadata), N_CLASSES), dtype=int)
    rows = []
    train = metadata.development_role.eq("train").to_numpy()
    stages = [(f"fold_{fold}", train & metadata.train_oof_fold.eq(fold).to_numpy(),
               train & metadata.train_oof_fold.ne(fold).to_numpy()) for fold in range(4)]
    stages.append(("validation", ~train, train))
    for stage, query_mask, reference_mask in stages:
        qi = np.flatnonzero(query_mask)
        refs = metadata.loc[reference_mask].sort_values(["plot_idx", "file"])
        ri = refs.index.to_numpy()
        if set(metadata.iloc[qi].plot_idx) & set(refs.plot_idx):
            raise ValueError("Geometry reference/query plot overlap")
        starts = np.flatnonzero(np.r_[True, refs.plot_idx.to_numpy()[1:] != refs.plot_idx.to_numpy()[:-1]])
        plot_labels = refs.iloc[starts].label_id_dense.to_numpy(dtype=int)
        class_counts = np.bincount(plot_labels, minlength=N_CLASSES)
        support[qi] = class_counts
        for c, count in enumerate(class_counts):
            rows.append({"bank": stage, "class_id": c, "reference_plots": int(count),
                         "query_images": len(qi), "reference_images": len(ri)})
        for modality, vectors in embeddings.items():
            d = np.clip(1.0 - vectors[qi] @ vectors[ri].T, 0.0, 2.0)
            plot_d = np.minimum.reduceat(d, starts, axis=1)
            for c in range(N_CLASSES):
                values = np.sort(plot_d[:, plot_labels == c], axis=1)
                for ki, k in enumerate(DISTANCE_K):
                    if class_counts[c] >= k:
                        distances[modality][qi, ki, c] = values[:, :k].mean(axis=1)
        print(f"Geometry {stage}: {len(qi)} queries; {len(starts)} disjoint reference plots", flush=True)
    return distances, support, pd.DataFrame(rows)


def score_predictions(metadata: pd.DataFrame, predictions: pd.DataFrame,
                      distances: dict, support: np.ndarray) -> pd.DataFrame:
    index = pd.Series(np.arange(len(metadata)), index=metadata.row_uid)
    pos = index.loc[predictions.row_uid].to_numpy()
    aligned = metadata.iloc[pos].reset_index(drop=True)
    columns = ["file", "plot_idx", "label_id_dense", "label_name", "train_oof_fold"]
    base = pd.concat([predictions.reset_index(drop=True), aligned[columns]], axis=1)
    raw = base.raw_pred.to_numpy(dtype=int)
    geo = base.geo_pred.to_numpy(dtype=int)
    truth = base.label_id_dense.to_numpy(dtype=int)
    base["disagreement"] = raw != geo
    base["outcome"] = np.select(
        [(geo == truth) & (raw != truth), (raw == truth) & (geo != truth),
         (geo != truth) & (raw != truth)],
        ["rescue", "harm", "both_wrong"], default="both_correct",
    )
    base["reference_plots_fusion_class"] = support[pos, raw]
    base["reference_plots_geo_class"] = support[pos, geo]
    base["plot_weight"] = 1.0 / base.groupby(
        ["cohort", "training_seed", "plot_idx"]
    ).row_uid.transform("size")
    result = []
    for ki, k in enumerate(DISTANCE_K):
        out = base.copy()
        out["distance_k"] = k
        for modality, values in distances.items():
            dr = values[pos, ki, raw]
            dg = values[pos, ki, geo]
            out[f"d_{modality}_fusion"] = dr
            out[f"d_{modality}_geo"] = dg
            out[f"A_{modality}"] = np.log((dr + EPSILON) / (dg + EPSILON))
        out["valid_geometry"] = np.isfinite(out[["A_image", "A_geo"]]).all(axis=1)
        out["k5_supported"] = (out.reference_plots_fusion_class >= 5) & (out.reference_plots_geo_class >= 5)
        for modality in distances:
            score = out[f"A_{modality}"]
            out[f"support_{modality}"] = np.select(
                [score > 0, score < 0, score.eq(0)],
                ["geo", "fusion", "tie"], default="unavailable",
            )
        out["support_pattern"] = "image=" + out.support_image + ";geo=" + out.support_geo
        if not out.loc[~out.disagreement & out.valid_geometry, ["A_image", "A_geo"]].eq(0).all().all():
            raise ValueError("Agreeing candidates should have zero geometric advantage")
        result.append(out)
    return pd.concat(result, ignore_index=True)


def summarize(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, auc_rows, quadrants = [], [], []
    for (cohort, seed, k), full in scores.groupby(["cohort", "training_seed", "distance_k"]):
        common = {"cohort": cohort, "training_seed": int(seed), "distance_k": int(k)}
        disagreement = full.loc[full.disagreement]
        valid = disagreement.loc[disagreement.valid_geometry]
        for outcome in ("rescue", "harm", "both_wrong"):
            all_state = disagreement.loc[disagreement.outcome.eq(outcome)]
            state = valid.loc[valid.outcome.eq(outcome)]
            row = {**common, "outcome": outcome, "n_total": len(all_state), "n_valid": len(state),
                   "n_unavailable": len(all_state) - len(state), "n_plots": state.plot_idx.nunique()}
            for modality in ("image", "geo"):
                a = state[f"A_{modality}"]
                row.update({f"{modality}_median": a.median(), f"{modality}_q25": a.quantile(.25),
                            f"{modality}_q75": a.quantile(.75), f"{modality}_supports_geo": (a > 0).mean(),
                            f"{modality}_supports_fusion": (a < 0).mean(), f"{modality}_tie": a.eq(0).mean()})
            summary.append(row)
        for pattern, group in valid.groupby("support_pattern"):
            counts = group.outcome.value_counts()
            nr, nh, nb = (int(counts.get(state, 0)) for state in ("rescue", "harm", "both_wrong"))
            quadrants.append({**common, "support_pattern": pattern, "n": len(group),
                              "rescue": nr, "harm": nh, "both_wrong": nb,
                              "rescue_fraction_among_rescue_harm": nr / (nr + nh) if nr + nh else np.nan,
                              "net_rescue_minus_harm": nr - nh})
        for subset in ("all_available", "k5_supported"):
            rows = valid.loc[valid.outcome.isin(["rescue", "harm"])]
            if subset == "k5_supported":
                rows = rows.loc[rows.k5_supported]
            y = rows.outcome.eq("rescue").astype(int)
            for modality in ("image", "geo"):
                a = rows[f"A_{modality}"]
                auc_rows.append({**common, "subset": subset, "modality": modality,
                                 "n_rescue": int(y.sum()), "n_harm": int((1 - y).sum()),
                                 "n_plots": rows.plot_idx.nunique(),
                                 "auc": roc_auc_score(y, a) if y.nunique() == 2 else np.nan,
                                 "auc_plot_weighted": roc_auc_score(y, a, sample_weight=rows.plot_weight)
                                 if y.nunique() == 2 else np.nan})
    return pd.DataFrame(summary), pd.DataFrame(auc_rows), pd.DataFrame(quadrants)


def write_report(output: Path, summary: pd.DataFrame, auc: pd.DataFrame,
                 quadrants: pd.DataFrame, support: pd.DataFrame,
                 scores: pd.DataFrame) -> None:
    primary = auc.loc[auc.distance_k.eq(1) & auc.subset.eq("all_available")]
    averages = primary.groupby(["cohort", "modality"]).auc.mean()
    near_zero = scores.loc[
        scores.cohort.eq("validation") & scores.distance_k.eq(1)
        & scores.disagreement & scores.outcome.eq("harm") & scores.d_geo_geo.le(1e-12)
    ].groupby("training_seed").size().reindex([1, 2, 3, 4], fill_value=0)
    lines = [
        "# Geometry support for fusion and geo disagreements", "",
        "The measured geometry contains correction-related information, with "
        "stronger rescue-versus-harm separation from frozen image geometry than "
        "geo geometry in this check. On validation, the mean AUROCs across "
        f"four seeds are {averages['validation', 'image']:.3f} for image and "
        f"{averages['validation', 'geo']:.3f} for geo. Training OOF results are "
        f"weaker ({averages['train_oof', 'image']:.3f} and "
        f"{averages['train_oof', 'geo']:.3f}, respectively). This supports further "
        "evaluation of the scores as candidate features; it does not establish "
        "that adding them to a router improves corrections.", "",
        "This check asks whether frozen embedding geometry distinguishes useful geo "
        "corrections from harmful overrides. It uses 3,378 development training images "
        "with out-of-fold expert predictions and 822 development validation images, "
        "each evaluated under four saved expert seeds. No expert or router is fitted, "
        "no intervention threshold is selected, and no cleaned-test data are read.", "",
        "The comparisons below are descriptive development evidence. Validation has "
        "already been used during the broader research process; this is not an "
        "independent confirmation study.", "",
        "## Score and reference definition", "",
        "Both scores use A = log((distance to fusion-predicted class + 1e-8) / "
        "(distance to geo-predicted class + 1e-8)). Positive values favor geo's "
        "candidate in BOTH image and geo space; negative values favor fusion's "
        "candidate. Image geometry need not favor the image or fusion expert.", "",
        "Image features are the historical frozen pretrained SigLIP2 cache; geo "
        "features are cached 100 m mean embeddings. Each vector is L2-normalized "
        "and distances are cosine distances in the original dimensions. No PCA, "
        "t-SNE, coordinate standardization, density filtering or class reweighting "
        "is applied to the reference geometry. Expert predictions come from the "
        "existing habitat-adapted protocol models, deliberately distinct from the "
        "frozen reference encoder.", "",
        "The primary class distance is to its nearest reference plot (k=1). A "
        "plot contributes the minimum distance over its photos. The sensitivity "
        "uses the mean distance to the nearest five distinct reference plots. "
        "A score is unavailable if either candidate class has fewer than k plots. "
        "The matched sensitivity compares both k values on the same supported rows.", "",
        "For every training OOF query, the entire held-out fold is excluded from "
        "the reference bank. Validation queries use only development-training "
        "references. All query and reference plots are disjoint. Query truth labels "
        "are used to assign outcome groups after calculating geometric support.", "",
        "## Rescue versus harm discrimination", "",
        "Rescue means geo is correct and fusion is wrong; harm means fusion is "
        "correct and geo is wrong. Both-wrong disagreements remain in all group "
        "and quadrant descriptions but cannot distinguish these two outcomes. "
        "AUROC evaluates whether a randomly chosen rescue receives a larger score "
        "than a harm (with half credit for ties). 0.5 indicates chance ranking. "
        "The table reports the arithmetic mean and range across four seeds, "
        "not an analysis treating repeated seeds as independent images.", "",
        "| Queries | Distance | Image AUROC mean and seed range | Geo AUROC mean and seed range |",
        "|---|---|---:|---:|",
    ]
    for cohort in ("train_oof", "validation"):
        for k, subset, title in ((1, "all_available", "Nearest plot, all available"),
                                 (1, "k5_supported", "Nearest plot, five-plot supported subset"),
                                 (5, "k5_supported", "Mean of five, same supported subset")):
            values = []
            for modality in ("image", "geo"):
                a = auc.loc[auc.cohort.eq(cohort) & auc.distance_k.eq(k) & auc.subset.eq(subset)
                            & auc.modality.eq(modality), "auc"]
                values.append(f"{a.mean():.3f} ({a.min():.3f}–{a.max():.3f})")
            lines.append(f"| {cohort} | {title} | " + " | ".join(values) + " |")
    lines += ["", "![Discrimination across seeds](geometry_disagreement_auc.png)", "",
              "## Geometric support by actual outcome", "",
              "Each support rate is first calculated within a seed, then averaged "
              "across seeds. Denominators include only disagreements with both "
              "geometry scores available. Counts show the range across seeds; "
              "the same images recur across seeds.", "",
              "| Queries | Outcome | Cases per seed | Image favors fusion | Image favors geo | Geo favors fusion | Geo favors geo |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for cohort in ("train_oof", "validation"):
        for outcome in ("rescue", "harm", "both_wrong"):
            rows = summary.loc[summary.cohort.eq(cohort) & summary.distance_k.eq(1)
                               & summary.outcome.eq(outcome)]
            rates = [f"{100 * rows[column].mean():.1f}%" for column in
                     ("image_supports_fusion", "image_supports_geo", "geo_supports_fusion", "geo_supports_geo")]
            lines.append(f"| {cohort} | {outcome} | {rows.n_valid.min()}–{rows.n_valid.max()} | "
                         + " | ".join(rates) + " |")
    lines += ["", "![Support rates](geometry_disagreement_support.png)", "",
              "![Outcome distributions](geometry_disagreement_distributions.png)", "",
              "![Validation disagreements](geometry_disagreement_validation.png)", "",
              "If each modality supports its usual expert equally often in rescue "
              "and harm cases, geometry merely repeats the disagreement. Differences "
              "in outcome distributions or ranking indicate a potentially useful "
              "signal, but do not establish additional value over existing model "
              "outputs or the benefit of a fitted router.", "",
              "## Joint support patterns on validation", "",
              "These are zero-sign diagnostics, not selected intervention policies. "
              "Both-wrong cases remain visible: agreement between geometries on "
              "one of two candidates does not establish that either is correct.", "",
              "| Image geometry favors | Geo geometry favors | Mean rescue count per seed | Mean harm count | Mean both-wrong count |",
              "|---|---|---:|---:|---:|"]
    q = quadrants.loc[quadrants.cohort.eq("validation") & quadrants.distance_k.eq(1)]
    for image_label in ("fusion", "geo"):
        for geo_label in ("fusion", "geo"):
            pattern = f"image={image_label};geo={geo_label}"
            rows = q.loc[q.support_pattern.eq(pattern)].set_index("training_seed")
            mean = rows[["rescue", "harm", "both_wrong"]].reindex([1, 2, 3, 4], fill_value=0).mean()
            lines.append(f"| {image_label} | {geo_label} | {mean.rescue:.1f} | {mean.harm:.1f} | {mean.both_wrong:.1f} |")
    lines += ["", "## Coverage and interpretation limits", "",
              "| Queries | k | Unavailable disagreement scores per seed |",
              "|---|---:|---:|"]
    coverage = summary.groupby(["cohort", "distance_k", "training_seed"]).n_unavailable.sum()
    for (cohort, k), rows in coverage.groupby(level=[0, 1]):
        lines.append(f"| {cohort} | {k} | {int(rows.min())}–{int(rows.max())} |")
    lines += ["", "Rare candidate classes may be absent or have fewer than five "
              "reference plots. Missing scores are retained explicitly rather than "
              "replaced by infinite or artificially strong support. Reference counts "
              "are saved in [reference bank support](reference_bank_support.csv).", "",
              "Class prevalence and reference density influence nearest-neighbor "
              "distances. A plot with more photos has more opportunities to provide "
              "a nearby image. A supplemental AUROC column weights each image by "
              "the inverse number of photos in its query plot in the full cohort; "
              "this checks query multiplicity but does not remove reference "
              "sampling effects. Seed ranges are sensitivity summaries, not "
              "confidence intervals over new plots. No population uncertainty "
              "interval or threshold performance claim is made.", "",
              "Distinct plots can share identical satellite vectors in these "
              "inputs, so plot-disjoint references do not imply distinct satellite "
              "features. In validation, "
              f"{int(near_zero.min())}–{int(near_zero.max())} harm cases per seed "
              "have effectively zero distance to a reference of geo's predicted "
              "class. Their large positive geo ratios are controlled by epsilon "
              "and illustrate a failure of nearest-reference support despite "
              "correct calculation. These cases remain in the analysis. The "
              "same cached vector can represent differently labelled habitats; "
              "the extraction cell identities are not available in this study.", "",
              "The ratio compares only the two candidate classes. Absolute "
              "distances are retained in the row-level results because both "
              "candidates can be poorly supported or a third class can be closer.", "",
              "The frozen image cache retains the legacy BGR and resize "
              "preprocessing limitations documented in the "
              "[embedding overview](../2026-09-29_embedding_geometry/report.md). "
              "No new encoder extraction was performed.", "",
              "## Reproducibility", "",
              "Prediction manifests, file and logical hashes, native logits and "
              "argmax predictions, producer concatenation, and plot provenance "
              "are checked before analysis. Source files are verified unchanged "
              "at completion. The method is recorded in `declaration.json`, "
              "before score outcomes are calculated. `manifest.json` records "
              "source, implementation, environment and result hashes.", "",
              "Files: [state summaries](state_summary.csv), [AUROC by seed](auc_by_seed.csv), "
              "[joint support counts](support_patterns.csv), and "
              "[row-level scores](geometry_scores.parquet).", "",
              "```bash", "python tools/check_geometry_disagreements.py --output-dir <new-output-directory>",
              "```", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_geometry_disagreement_check"
    ))
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "declaration.json").write_text(json.dumps(DECLARATION, indent=2) + "\n")
    metadata, image, geo, hashes = load_development("all")
    predictions = prediction_inputs(metadata, hashes)
    embeddings = {"image": image, "geo": geo}
    for modality, x in embeddings.items():
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if not np.isfinite(x).all() or np.any(norms <= 0):
            raise ValueError(f"Invalid {modality} vectors")
        embeddings[modality] = x / norms
    with threadpool_limits(limits=1):
        distances, support_array, support = reference_distances(metadata, embeddings)
    scores = score_predictions(metadata, predictions, distances, support_array)
    summary, auc, quadrants = summarize(scores)
    scores.to_parquet(output / "geometry_scores.parquet", index=False)
    summary.to_csv(output / "state_summary.csv", index=False)
    auc.to_csv(output / "auc_by_seed.csv", index=False)
    quadrants.to_csv(output / "support_patterns.csv", index=False)
    support.to_csv(output / "reference_bank_support.csv", index=False)
    from geometry_disagreement_figures import render_figures
    figures = render_figures(scores, summary, auc, output)
    write_report(output, summary, auc, quadrants, support, scores)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"Input changed during analysis: {path}")
    source_files = [Path(__file__), Path(__file__).with_name("geometry_disagreement_figures.py"),
                    Path(__file__).with_name("plot_multimodal_embedding_geometry.py"),
                    Path("multimodal/geo_helpfulness_oof.py"), Path("multimodal/geo_helpfulness_protocol.py")]
    manifest = {
        "declaration": DECLARATION, "images": len(metadata), "plots": int(metadata.plot_idx.nunique()),
        "training_seeds": [1, 2, 3, 4], "source_sha256": hashes,
        "source_code_sha256": {str(p.resolve()): sha256(p) for p in source_files},
        "environment": package_versions(), "figures": figures,
        "output_sha256": {str(p.relative_to(output)): sha256(p)
                          for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(auc.loc[auc.distance_k.eq(1) & auc.subset.eq("all_available")].to_string(index=False))
    print(f"Saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

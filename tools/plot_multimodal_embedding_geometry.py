"""Explore frozen image/GSE geometry using development data only.

This is a descriptive visualization, not a router fit or a held-out evaluation.
Run from the repository root with the habcls Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from embedding_geometry_figures import render_figures


PROTOCOL = Path(
    "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1/protocol"
)
CACHE = Path(
    "multimodal_artifacts/joined_tables/cs/"
    "hf-hub_timm_ViT-SO400M-16-SigLIP2-384_pretrained/gse_100m/seed1"
)
PRIMARY = ("image", "geo_native", "concat_native")
PAIR_NAMES = (
    ("Improved Grassland", "Neutral Grassland"),
    ("Dwarf Shrub Heath", "Bog"),
    ("Acid Grassland", "Bog"),
    ("Arable and Horticulture", "Improved Grassland"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict:
    result = {}
    for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "plotly"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def load_development(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    manifest = json.loads((PROTOCOL / "protocol_manifest.json").read_text())
    paths = [CACHE / "train.parquet", CACHE / "val.parquet"]
    source_hashes = {str(p): sha256(p) for p in paths}
    for path, digest in source_hashes.items():
        if manifest["development_source_file_sha256"][path] != digest:
            raise ValueError(f"Source hash mismatch: {path}")
    source = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    assignments_path = PROTOCOL / "development_assignments.parquet"
    if sha256(assignments_path) != manifest["artifact_file_sha256"]["development_assignments.parquet"]:
        raise ValueError("Development assignment hash mismatch")
    assignments = pd.read_parquet(assignments_path)
    if source.file.duplicated().any() or assignments.file.duplicated().any():
        raise ValueError("Repeated image identity")
    if set(source.file) != set(assignments.file):
        raise ValueError("Development source/assignment identities differ")
    if split == "train":
        assignments = assignments.loc[assignments.development_role.eq("train")]
    metadata = assignments.sort_values("row_uid").reset_index(drop=True)
    ordered = source.set_index("file").loc[metadata.file].reset_index()
    for column in ("plot_idx", "label_name"):
        if not np.array_equal(ordered[column].astype(str), metadata[column].astype(str)):
            raise ValueError(f"Mismatched {column}")
    if not np.array_equal(ordered.label_id, metadata.label_id_dense):
        raise ValueError("Mismatched class mapping")
    image_cols = [f"I{i:04d}" for i in range(1152)]
    geo_cols = [f"A{i:02d}" for i in range(64)]
    image = ordered[image_cols].to_numpy(dtype=np.float64)
    geo = ordered[geo_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(image).all() or not np.isfinite(geo).all():
        raise ValueError("Nonfinite embeddings")
    if not np.allclose(np.linalg.norm(image, axis=1), 1.0, atol=2e-6):
        raise ValueError("Expected L2-normalized image embeddings")
    geo_frame = pd.DataFrame(geo).assign(plot_idx=metadata.plot_idx.to_numpy())
    if geo_frame.groupby("plot_idx").nunique().to_numpy().max() != 1:
        raise ValueError("Geo features vary within a plot; revisit the plotting unit")
    if metadata.groupby("plot_idx").label_name.nunique().max() != 1:
        raise ValueError("Multiple habitat labels within a plot")
    source_hashes[str(assignments_path)] = sha256(assignments_path)
    source_hashes[str(PROTOCOL / "protocol_manifest.json")] = sha256(
        PROTOCOL / "protocol_manifest.json"
    )
    return metadata, image, geo, source_hashes


def make_spaces(image: np.ndarray, geo: np.ndarray) -> tuple[dict, dict, dict]:
    mean, std = geo.mean(axis=0), geo.std(axis=0)
    scale = np.where(std > 0, std, 1.0)
    geo_standardized = (geo - mean) / scale
    # Equalize average squared norms of the two blocks, without per-row rescaling.
    image_rms = float(np.sqrt(np.mean(np.sum(image**2, axis=1))))
    geo_rms = float(np.sqrt(np.mean(np.sum(geo_standardized**2, axis=1))))
    spaces = {
        "image": image,
        "geo_native": geo,
        "concat_native": np.concatenate([image, geo], axis=1),
        "concat_standardized": np.concatenate([image, geo_standardized], axis=1),
        "concat_balanced": np.concatenate(
            [image / image_rms, geo_standardized / geo_rms], axis=1
        ),
    }
    iv, gv, sv = (float(x.var(axis=0).sum()) for x in (image, geo, geo_standardized))
    diagnostic = {
        "image_centered_variance_trace": iv,
        "native_geo_centered_variance_trace": gv,
        "standardized_geo_centered_variance_trace": sv,
        "native_concat_geo_squared_distance_share": gv / (iv + gv),
        "standardized_concat_geo_squared_distance_share": sv / (iv + sv),
        "balanced_concat_geo_squared_distance_share": (sv / geo_rms**2)
        / (iv / image_rms**2 + sv / geo_rms**2),
        "image_rms_norm": image_rms,
        "standardized_geo_rms_norm": geo_rms,
        "geo_unique_vector_count": len(np.unique(geo, axis=0)),
    }
    state = {"geo_mean": mean.tolist(), "geo_std": scale.tolist()}
    return spaces, diagnostic, state


def project(spaces: dict, seed: int, label: str) -> tuple[dict, dict]:
    coordinates, metrics = {}, {}
    for name, x in spaces.items():
        print(f"{label}: PCA and t-SNE for {name}, shape={x.shape}", flush=True)
        pca = PCA(n_components=min(50, x.shape[1], len(x) - 1), svd_solver="full")
        reduced = pca.fit_transform(x)
        tsne = TSNE(
            n_components=2,
            perplexity=30.0,
            early_exaggeration=12.0,
            learning_rate="auto",
            max_iter=1500,
            init="pca",
            random_state=seed,
            metric="euclidean",
            method="barnes_hut",
            angle=0.5,
            n_jobs=1,
        )
        nonlinear = tsne.fit_transform(reduced)
        if not np.isfinite(nonlinear).all() or not np.isfinite(reduced).all():
            raise ValueError("Nonfinite projection")
        coordinates[name] = {"pca": reduced[:, :2], "tsne": nonlinear}
        metrics[name] = {
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca50_variance_retained": float(pca.explained_variance_ratio_.sum()),
            "tsne_kl_divergence": float(tsne.kl_divergence_),
            "tsne_n_iter": int(tsne.n_iter_),
            "input_dimensions": x.shape[1],
        }
    return coordinates, metrics


def neighbourhood_diagnostics(
    metadata: pd.DataFrame, spaces: dict, k: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use original-space distances and at most one neighbour per other plot."""
    plots = metadata.plot_idx.to_numpy()
    labels = metadata.label_name.to_numpy()
    names = sorted(set(labels))
    class_rows, pair_rows = [], []
    for space, x in spaces.items():
        print(f"Original-space neighbours: {space}", flush=True)
        distances = pairwise_distances(x, metric="euclidean", n_jobs=1)
        proportions = np.zeros((len(x), len(names)), dtype=float)
        for i in range(len(x)):
            seen = {plots[i]}
            selected = []
            for j in np.argsort(distances[i], kind="stable"):
                if plots[j] not in seen:
                    seen.add(plots[j])
                    selected.append(j)
                    if len(selected) == k:
                        break
            if len(selected) != k:
                raise ValueError("Insufficient distinct reference plots")
            neighbour_labels = labels[selected]
            proportions[i] = [np.mean(neighbour_labels == name) for name in names]
        # Query photographs also receive equal total weight per plot.
        by_plot = pd.DataFrame(proportions, columns=names).assign(
            plot_idx=plots, query_habitat=labels
        ).groupby(["plot_idx", "query_habitat"]).mean().reset_index()
        for habitat in names:
            rows = by_plot.loc[by_plot.query_habitat.eq(habitat)]
            class_rows.append({
                "space": space,
                "habitat": habitat,
                "query_plots": len(rows),
                "neighbours": k,
                "same_habitat_fraction": float(rows[habitat].mean()),
            })
        for first, second in PAIR_NAMES:
            for origin, destination in ((first, second), (second, first)):
                rows = by_plot.loc[by_plot.query_habitat.eq(origin)]
                pair_rows.append({
                    "space": space,
                    "query_habitat": origin,
                    "neighbour_habitat": destination,
                    "query_plots": len(rows),
                    "neighbours": k,
                    "neighbour_fraction": float(rows[destination].mean()),
                })
    return pd.DataFrame(class_rows), pd.DataFrame(pair_rows)


def save_coordinates(metadata: pd.DataFrame, coords: dict, path: Path) -> None:
    rows = []
    for space, methods in coords.items():
        for method, xy in methods.items():
            rows.append(metadata[["row_uid", "file", "plot_idx", "label_name"]].assign(
                space=space, method=method, x=xy[:, 0], y=xy[:, 1]
            ))
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def write_report(output: Path, metadata: pd.DataFrame, metrics: dict, scales: dict,
                 class_stats: pd.DataFrame, pair_stats: pd.DataFrame, split: str) -> None:
    interactive = "[Interactive maps](embedding_geometry_interactive.html) · " if (
        output / "embedding_geometry_interactive.html"
    ).exists() else ""
    lines = [
        "# Frozen SigLIP2 and geographic embedding geometry", "",
        f"Descriptive exploration of **{len(metadata):,} images from "
        f"{metadata.plot_idx.nunique():,} plots**, development split `{split}`. "
        "No cleaned-test table was opened and no expert or router was fitted. "
        "When validation samples are included, their labels are visible for "
        "exploration; these plots are not independent validation evidence.", "",
        "## Embedding maps", "",
        "![PCA and t-SNE](embedding_geometry_overview.png)", "",
        interactive + "[PDF](embedding_geometry_overview.pdf)", "",
        "The primary columns use the cached L2-normalized frozen pretrained SigLIP2 "
        "image vector (1,152 dimensions), native GSE vector (64), and their direct "
        "concatenation (1,216). Concatenation is an input representation; it is "
        "not the learned hidden representation of a fusion classifier.", "",
        "PCA and t-SNE are fitted without habitat labels; labels only colour the "
        "points. Each space has its own projection, so axes and inter-cluster "
        "distances cannot be compared across panels. Visual overlap does not "
        "establish overlap in the original space, and visual separation does "
        "not establish classifier performance. These frozen features also differ "
        "from the habitat-adapted features used by the router's existing experts.", "",
        "## Previously confused habitat groups", "",
        "![Habitat focus](embedding_geometry_habitat_focus.png)", "",
        "These highlight grasslands and Heath/Bog on the same global t-SNE "
        "coordinates; projections are not refitted to the highlighted classes. "
        "The pairs were chosen from prior discussions, not to select the most "
        "visually striking result.", "",
        "## Scaling sensitivity", "",
        "![Scaling sensitivity](embedding_geometry_scaling.png)", "",
        "The comparison uses (1) native concatenation, (2) image + geo standardized "
        "per coordinate on the selected development images, and (3) image + "
        "standardized geo with each block divided by its RMS vector norm. "
        "The second follows the existing fusion input scaling convention, "
        "but still uses frozen pretrained image features. The balanced version "
        "equalizes uncentered block energy, not centered variance or semantic "
        "information. All transformations are descriptive fits on this plot cohort.", "",
        "| Concatenation | Geo share of expected squared pairwise distance |",
        "|---|---:|",
    ]
    for name, key in (("Native", "native"), ("Standardized geo", "standardized"),
                      ("Balanced standardized geo", "balanced")):
        lines.append(f"| {name} | {100 * scales[key + '_concat_geo_squared_distance_share']:.2f}% |")
    lines += ["", "These shares describe Euclidean input distances, not the "
              "learned classifier's reliance on each modality.", "",
              "## Original-space neighbourhoods", "",
              "For each image, find 15 nearest **distinct other plots**, taking "
              "the nearest image from each plot. Exclude the whole query plot. "
              "Average query results within plots, then within habitats. These "
              "are descriptive neighbourhood statistics, not cross-validated "
              "classification or router performance. A reference plot with more "
              "photos has more opportunities to supply a nearby image; using "
              "distinct plots does not remove all sampling effects.", "",
              "| Space | Mean same-habitat neighbour fraction across habitats |",
              "|---|---:|"]
    for space, rows in class_stats.groupby("space", sort=False):
        lines.append(f"| {space} | {100 * rows.same_habitat_fraction.mean():.2f}% |")
    lines += ["", "Habitats with one development plot cannot have same-habitat "
              "neighbours after plot exclusion; inspect class support before "
              "interpreting the macro average. See [class details](neighbour_habitat_agreement.csv) "
              "and [directional pair mixing](neighbour_pair_mixing.csv).", "",
              "### Local mixing for previously confused pairs", "",
              "Each entry is the fraction of neighbouring plots with the second "
              "habitat, conditional on the first habitat. These quantities are "
              "directional and depend on class prevalence as well as geometry.", "",
              "| Query habitat → neighbour habitat | Image | Native geo | Native concatenation |",
              "|---|---:|---:|---:|"]
    mixing = pair_stats.pivot(
        index=["query_habitat", "neighbour_habitat"],
        columns="space", values="neighbour_fraction",
    )
    for (origin, destination), row in mixing.iterrows():
        values = " | ".join(f"{100 * row[space]:.1f}%" for space in PRIMARY)
        lines.append(f"| {origin} → {destination} | {values} |")
    lines += ["", "## Projection and sampling details", "",
              "| Space | Variance retained by 2D PCA | Variance retained before t-SNE |",
              "|---|---:|---:|"]
    for space, m in metrics.items():
        lines.append(f"| {space} | {100 * sum(m['pca_explained_variance_ratio'][:2]):.2f}% "
                     f"| {100 * m['pca50_variance_retained']:.2f}% |")
    lines += ["", "t-SNE: Euclidean distance after unwhitened PCA to 50 components; "
              "perplexity 30, PCA initialization, automatic learning rate, "
              "1,500 maximum iterations, random seed 20260929, Barnes–Hut angle 0.5. "
              "Settings are fixed without selecting a preferred plot. PCA uses full SVD.", "",
              f"There are {scales['geo_unique_vector_count']:,} distinct geo vectors "
              "in this cohort. Photos from the same plot share a geo vector; "
              "2D density and point multiplicity should not be interpreted as "
              "independent environmental support. The supplementary maps use "
              "one image per plot, selected by lexicographic filename without "
              "consulting embeddings or habitat labels.", "",
              "![One image per plot](one_image_per_plot/embedding_geometry_overview.png)", "",
              "## Cache provenance and limits", "",
              "Source tables are the seed-1 pretrained cache's original train/val "
              "files, joined to the immutable router development assignments. "
              "Their original train/val membership is not the router split. "
              "Source hashes match the frozen protocol manifest; exact paths, "
              "hashes, environment and scaling state are saved in `manifest.json`.", "",
              "This reuses historical pretrained caches. Repository export "
              "code/config indicates the legacy OpenCV BGR → 439 resize → PIL "
              "without RGB conversion → OpenCLIP evaluation transform path. "
              "The source cache does not contain a complete extraction manifest "
              "or original checkpoint hash, and no fresh encoder replay was run. "
              "These are therefore plots of the authenticated historical cache, "
              "not a newly verified native-RGB SigLIP2 extraction.", "",
              "Reproduce from the repository root:", "", "```bash",
              f"python tools/plot_multimodal_embedding_geometry.py --development-split {split} "
              "--output-dir <new-output-directory>", "```", "",
              "References: [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/"
              "sklearn.decomposition.PCA.html), [t-SNE](https://scikit-learn.org/stable/modules/"
              "generated/sklearn.manifold.TSNE.html)."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-split", choices=("train", "all"), default="train")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "dev_plans/reliability_aware_geo_fusion/reports/2026-09-29_embedding_geometry"
    ))
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata, image, geo, hashes = load_development(args.development_split)
    spaces, scales, state = make_spaces(image, geo)
    metadata.to_parquet(output / "sample_metadata.parquet", index=False)
    with threadpool_limits(limits=1):
        coords, metrics = project(spaces, 20260929, "All images")
        save_coordinates(metadata, coords, output / "coordinates.parquet")
        class_stats, pair_stats = neighbourhood_diagnostics(metadata, spaces)
        class_stats.to_csv(output / "neighbour_habitat_agreement.csv", index=False)
        pair_stats.to_csv(output / "neighbour_pair_mixing.csv", index=False)
        figures = render_figures(metadata, coords, metrics, output)
        representative = metadata.sort_values("file").drop_duplicates("plot_idx").index.to_numpy()
        plot_metadata = metadata.iloc[representative].reset_index(drop=True)
        plot_spaces = {k: spaces[k][representative] for k in PRIMARY}
        plot_coords, plot_metrics = project(plot_spaces, 20260929, "One image per plot")
        plot_output = output / "one_image_per_plot"
        plot_output.mkdir()
        save_coordinates(plot_metadata, plot_coords, plot_output / "coordinates.parquet")
        render_figures(plot_metadata, plot_coords, plot_metrics, plot_output)
    write_report(output, metadata, metrics, scales, class_stats, pair_stats, args.development_split)
    document = {
        "study": "exploratory_frozen_embedding_geometry_v1",
        "development_split": args.development_split,
        "images": len(metadata), "plots": int(metadata.plot_idx.nunique()),
        "source_sha256": hashes,
        "source_code_sha256": {str(p): sha256(p) for p in
                               (Path(__file__), Path(__file__).with_name("embedding_geometry_figures.py"))},
        "cleaned_test_opened": False,
        "encoder_replayed": False,
        "projection_labels_used": False,
        "projection_seed": 20260929,
        "projection_metrics": metrics,
        "representative_projection_metrics": plot_metrics,
        "scaling_diagnostics": scales,
        "scaling_state": state,
        "environment": {"python": platform.python_version(), **package_versions()},
    }
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"Source changed during plotting: {path}")
    document["output_sha256"] = {
        str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    (output / "manifest.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output.resolve()), "scaling": scales, "figures": figures}, indent=2))


if __name__ == "__main__":
    main()

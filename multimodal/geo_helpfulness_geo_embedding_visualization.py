"""Add plot-level geographic maps to the existing adapted-image comparison."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_adapted_embeddings as image_maps


CACHE = image_maps.REPO / (
    "multimodal_artifacts/joined_tables/cs/"
    "hf-hub_timm_ViT-SO400M-16-SigLIP2-384_pretrained/gse_100m/seed1"
)
GEO_COLUMNS = [f"A{i:02d}" for i in range(64)]
GEO_SETTINGS = {
    **image_maps.SETTINGS,
    "input": "L2-normalized native 64-dimensional 100 m GSE vector",
    "coordinate_standardization": False,
    "projection_unit": "one vector per distinct plot",
    "pca_fitting_plots": 1300,
    "tsne_fitting_plots": 2156,
    "shared_across_encoder_seeds": True,
}


def load_geo(metadata: pd.DataFrame, image_hashes: dict):
    """Select exact known cohort IDs; never use the cache's original split labels."""
    receipt = image_maps.read_json(image_maps.ADAPTED / "receipt.json")
    hashes, frames = dict(image_hashes), []
    for split in ("train", "val", "test"):
        path = CACHE / f"{split}.parquet"
        actual = image_maps.sha256(path)
        assert actual == receipt["source_sha256"][str(path)], f"Geo cache changed: {path}"
        hashes[path.relative_to(image_maps.REPO).as_posix()] = actual
        frames.append(pd.read_parquet(path, columns=["file", "plot_idx", *GEO_COLUMNS]))
    source = pd.concat(frames, ignore_index=True)
    assert not source.file.duplicated().any()
    assert set(metadata.file) <= set(source.file)
    ordered = source.set_index("file").loc[metadata.file].reset_index()
    assert np.array_equal(ordered.plot_idx.astype(str), metadata.plot_idx.astype(str))
    raw = ordered[GEO_COLUMNS].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(raw, axis=1)
    assert raw.shape == (5547, 64) and np.isfinite(raw).all() and (norms > 0).all()
    assert ordered.groupby("plot_idx")[GEO_COLUMNS].nunique().to_numpy().max() == 1
    assert metadata.groupby("plot_idx").split.nunique().max() == 1
    # The deterministic photo is only a representative identity: all photos
    # from the same plot have exactly the same geographic vector.
    plots = metadata.sort_values(["plot_idx", "file", "row_uid"]).drop_duplicates("plot_idx").copy()
    plots = plots.sort_values("plot_idx").reset_index(drop=True)
    plots["photo_count"] = plots.plot_idx.map(metadata.groupby("plot_idx").size()).astype(int)
    positions = pd.Index(metadata.row_uid).get_indexer(plots.row_uid)
    assert (positions >= 0).all() and len(plots) == 2156
    for split, (_, count) in image_maps.COUNTS.items():
        assert plots.split.eq(split).sum() == count
    vectors = raw[positions] / norms[positions, None]
    diagnostics = {
        "dimensions": 64, "photo_rows": 5547, "plot_rows": 2156,
        "source_cache_photo_rows": len(source),
        "excluded_cache_photo_rows": len(source) - len(metadata),
        "unique_native_vectors_across_plots": len(np.unique(raw[positions], axis=0)),
        "raw_norm_minimum": float(norms.min()), "raw_norm_maximum": float(norms.max()),
        "all_same_plot_photo_vectors_identical": True,
        "photo_weighting_in_geo_projection": False,
    }
    return plots, vectors, hashes, diagnostics


def fingerprint():
    spec = {"projection": inspect.getsource(image_maps.project_embeddings),
            "geo_preprocessing": inspect.getsource(load_geo), "settings": GEO_SETTINGS,
            "versions": {key: image_maps.environment()[key]
                         for key in ("numpy", "scikit-learn", "threadpoolctl")}}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def get_geo_projections(output: Path, metadata: pd.DataFrame, image_hashes: dict):
    plots, vectors, hashes, diagnostics = load_geo(metadata, image_hashes)
    manifest_path = output / "geo_projection_manifest.json"
    if manifest_path.exists():
        manifest = image_maps.read_json(manifest_path)
        assert manifest["study_id"] == image_maps.STUDY and manifest["modality"] == "geo"
        assert manifest["source_sha256"] == hashes
        assert manifest["projection_fingerprint"] == fingerprint(), "Geo projection changed; use a fresh output directory"
        for name, expected in manifest["output_sha256"].items():
            assert image_maps.sha256(output / name) == expected, f"Geo projection artifact changed: {name}"
        coordinates = pd.read_parquet(output / "geo_coordinates.parquet")
        metrics = image_maps.read_json(output / "geo_projection_metrics.json")
        print("Validated existing plot-level geo projections.", flush=True)
    else:
        print("Geo: training-fitted PCA50 on 1,300 plots; joint t-SNE on 2,156 plots", flush=True)
        coords, metrics, transform = image_maps.project_embeddings(
            vectors, plots.split.eq("development_train").to_numpy(), GEO_SETTINGS)
        metrics["input_diagnostics"] = diagnostics
        coordinates = pd.concat([
            plots.assign(method=method, x=xy[:, 0], y=xy[:, 1])
            for method, xy in coords.items()
        ], ignore_index=True)
        photo_frames = []
        for method in ("pca", "tsne"):
            mapping = coordinates.loc[coordinates.method.eq(method), ["plot_idx", "x", "y", "photo_count"]]
            photo_frames.append(metadata.merge(mapping, on="plot_idx", validate="many_to_one").assign(method=method))
        pd.concat(photo_frames, ignore_index=True).to_parquet(output / "geo_photo_coordinates.parquet", index=False)
        coordinates.to_parquet(output / "geo_coordinates.parquet", index=False)
        plots.to_parquet(output / "geo_plot_metadata.parquet", index=False)
        np.savez_compressed(output / "geo_pca_transform.npz", **transform)
        image_maps.write_json(output / "geo_projection_metrics.json", metrics)
        names = ["geo_coordinates.parquet", "geo_photo_coordinates.parquet", "geo_plot_metadata.parquet",
                 "geo_pca_transform.npz", "geo_projection_metrics.json"]
        image_maps.write_json(manifest_path, {
            "study_id": image_maps.STUDY, "modality": "geo", "source_sha256": hashes,
            "projection_fingerprint": fingerprint(), "settings": GEO_SETTINGS,
            "environment": image_maps.environment(), "input_diagnostics": diagnostics,
            "output_sha256": {name: image_maps.sha256(output / name) for name in names},
        })
    assert len(coordinates) == 2 * 2156
    assert not coordinates.duplicated(["method", "plot_idx"]).any()
    assert np.isfinite(coordinates[["x", "y"]].to_numpy()).all()
    for method in ("pca", "tsne"):
        part = coordinates.loc[coordinates.method.eq(method)].sort_values("plot_idx").reset_index(drop=True)
        pd.testing.assert_frame_equal(part[plots.columns], plots)
    return coordinates, metrics, hashes


def extend_report(output: Path, geo_coordinates: pd.DataFrame, geo_metrics: dict, labels: list[str]):
    plots = geo_coordinates.loc[geo_coordinates.method.eq("pca")]
    class_rows = []
    for class_id, name in enumerate(labels):
        row = {"label_id_dense": class_id, "habitat": name}
        for split in image_maps.SPLITS:
            part = plots.loc[plots.label_id_dense.eq(class_id) & plots.split.eq(split)]
            row[f"{split}_plots"] = len(part)
            row[f"{split}_images"] = int(part.photo_count.sum())
        class_rows.append(row)
    pd.DataFrame(class_rows).to_csv(output / "geo_class_support.csv", index=False)
    links = "\n".join(
        f"| {seed} | [PCA](figures/adapted_image_geo_seed{seed}_pca.png) | "
        f"[t-SNE](figures/adapted_image_geo_seed{seed}_tsne.png) | "
        f"[Interactive](interactive/adapted_image_geo_seed{seed}_interactive.html) |"
        for seed in image_maps.SEEDS)
    diagnostics = geo_metrics["input_diagnostics"]
    extension = f"""## Adapted image versus geographic embedding spaces

The comparison now includes the **64-dimensional 100 m AlphaEarth/GSE geographic vectors** for the same development and cleaned-test cohort. The main illustration uses image encoder seed 1; the geo embeddings and their projection are identical for all four encoder-seed comparisons.

![Adapted image versus geo, joint t-SNE](figures/adapted_image_geo_seed1_tsne.png)

Rows compare image and geo; columns compare development and cleaned test. Colours and split markers match the image-only plots. Axes are shared between development and test **within one modality and projection**. Image and geo maps are independently fitted: their axes, rotations, cluster spacing and numerical coordinate scales are not aligned across modalities.

![Adapted image versus geo, training-fitted PCA](figures/adapted_image_geo_seed1_pca.png)

**Sampling unit matters.** Image panels retain all 4,200 development and 1,347 test photographs. Geo panels show **one point per plot: 1,625 development and 531 test points**, covering exactly those same photos. All photos within a plot share an identical cached geo vector. Repeating those vectors during t-SNE fitting would give plots with more photographs extra weight; the geo projection instead weights plots equally. No jitter or photo-specific geo feature is introduced. The two modalities therefore differ in both representation and displayed sampling unit; cluster compactness or density is not a controlled ranking of their usefulness.

The geo source is the same native 100 m cache used by the correction study. Each vector is restored to **unit L2 length**; coordinates are **not z-scored**. This matches the study's cosine-distance geometry and avoids changing coordinate weights. The selected native vectors have lengths {diagnostics['raw_norm_minimum']:.3f}–{diagnostics['raw_norm_maximum']:.3f} before normalization. There are {diagnostics['unique_native_vectors_across_plots']:,} distinct native vectors across 2,156 plots; different plots can occasionally share the same vector as well.

Geo PCA is fitted only on the **1,300 development-training plot vectors**, then applied to the 325 validation and 531 cleaned-test plots. Its first two components retain **{100 * geo_metrics['pca2_training_variance_retained']:.2f}%** of training-plot variance; the 50 components supplied to t-SNE retain **{100 * geo_metrics['pca50_training_variance_retained']:.2f}%**. Geo t-SNE is jointly fitted on all **2,156 plots**, using the same fixed perplexity 30, PCA50 preprocessing, PCA initialization, maximum 1,500 iterations and random seed 20261005 as the image visualization. Its layout includes test embeddings but no labels. The original image projections are reused without refitting.

| Image encoder seed | Image/geo PCA comparison | Image/geo t-SNE comparison | Interactive comparison |
| --- | --- | --- | --- |
{links}

Each static comparison also has a PDF with the same stem. [Geo-only PCA/t-SNE overview](figures/geo_embedding_overview.png) provides larger geographic panels. The interactive comparisons show the same coordinates and identify image versus plot units, split and habitat; geo hover information also reports how many photos the plot represents.

The initial test cache contains 1,398 photographs; this visualization selects only the **1,347 authenticated cleaned-test identities**. Development roles come from the fixed router assignments, not the cache's older train/val membership. [Geo class support](geo_class_support.csv) gives image and plot counts, including absent classes. [Geo coordinates](geo_coordinates.parquet) contain one point per plot per method. [Photo-aligned geo coordinates](geo_photo_coordinates.parquet) broadcast each plot's exact coordinate to its constituent photographs for identity matching; those repeated rows are not used to fit or render the geographic maps.

The existing reproduction command now generates both modalities. On completed runs it verifies and reuses the saved image and geo coordinates. [Geo projection manifest](geo_projection_manifest.json), [metrics](geo_projection_metrics.json), and `geo_pca_transform.npz` record the sources, normalization, plot sampling and PCA transform. No fusion-concatenation map or new router evaluation is added.

"""
    path = output / "report.md"
    report = path.read_text()
    report = report.replace("# Adapted image embeddings: development and cleaned-test maps",
                            "# Adapted image and geo embeddings: development and cleaned-test maps", 1)
    heading = "## Development and test in a shared coordinate system"
    assert heading in report
    report = report.replace(heading, extension + heading, 1)
    path.write_text(report)

"""Descriptive PCA and t-SNE maps of adapted image and geographic embeddings.

PCA is fitted on development training data only. The t-SNE map jointly uses
development and test embeddings and is explicitly a transductive visualization.
Habitat labels are joined for plotting and never passed to either projection.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
EMBEDDINGS = BASE / "adapted_geometry_v1/embeddings"
ADAPTED = BASE / "joint_geometry_adapted_encoder_v1"
PROTOCOL = BASE / "protocol_v1/protocol"
STUDY = "adapted_embedding_visualization_v1"
SEEDS = (1, 2, 3, 4)
SPLITS = ("development_train", "development_validation", "cleaned_test")
COUNTS = {"development_train": (3378, 1300),
          "development_validation": (822, 325), "cleaned_test": (1347, 531)}
SETTINGS = {
    "input": "L2-normalized 1152-dimensional adapted image embedding",
    "pca_fit_split": "development_train", "pca_components": 50,
    "pca_solver": "full", "pca_whiten": False,
    "tsne_fit_splits": list(SPLITS), "tsne_input": "training-fitted PCA50",
    "tsne_dimensions": 2, "perplexity": 30.0, "early_exaggeration": 12.0,
    "learning_rate": "auto", "max_iter": 1500, "init": "pca",
    "random_state": 20261005, "metric": "euclidean", "method": "barnes_hut",
    "angle": 0.5, "n_jobs": 1, "thread_limit": 1,
    "projection_labels_used": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment() -> dict:
    return {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "pandas", "scikit-learn",
                                        "matplotlib", "plotly", "threadpoolctl")}}


def load_sources():
    hashes = {}

    def bind(path, expected=None):
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        hashes[path.relative_to(REPO).as_posix()] = actual
        return path

    receipt = read_json(bind(ADAPTED / "receipt.json"))
    extraction_path = EMBEDDINGS / "manifest.json"
    extraction = read_json(bind(extraction_path,
                                receipt["source_sha256"][str(extraction_path)]))
    assert extraction["training_seeds"] == list(SEEDS)
    assert extraction["same_encoder_for_reference_and_query"]
    assert extraction["l2_normalized"] and not extraction["oof_diagnostics"]
    protocol_path = PROTOCOL / "protocol_manifest.json"
    protocol = read_json(bind(protocol_path, extraction["source_sha256"][str(protocol_path)]))
    assignments = pd.read_parquet(bind(PROTOCOL / "development_assignments.parquet",
                                       protocol["artifact_file_sha256"]["development_assignments.parquet"]))
    label_cols = ["row_uid", "file", "plot_idx", "label_id_dense", "canonical_l3_id", "label_name"]
    development = assignments[label_cols].copy()
    development["split"] = assignments.development_role.map(
        {"train": "development_train", "validation": "development_validation"})
    test = pd.read_parquet(bind(ADAPTED / "labels.parquet", receipt["output_sha256"]["labels.parquet"]))
    metadata = pd.concat([development, test[label_cols].assign(split="cleaned_test")], ignore_index=True)
    assert not metadata.row_uid.duplicated().any() and not metadata.file.duplicated().any()
    assert metadata.split.notna().all() and metadata.groupby("plot_idx").label_id_dense.nunique().max() == 1
    for i, split in enumerate(SPLITS):
        selected = metadata.loc[metadata.split.eq(split)]
        assert (len(selected), selected.plot_idx.nunique()) == COUNTS[split]
        for previous in SPLITS[:i]:
            assert not set(selected.plot_idx) & set(metadata.loc[metadata.split.eq(previous), "plot_idx"])
    ontology_path = BASE / "relaxed_guardrails_rf_v1/declaration.json"
    ontology = read_json(bind(ontology_path, receipt["source_sha256"][str(ontology_path)]))["ontology"]
    assert [item["dense_id"] for item in ontology] == list(range(18))
    labels = [item["label_name"] for item in ontology]
    class_path = bind(REPO / "data/__init__.py")
    class_map = next(ast.literal_eval(node.value) for node in ast.parse(class_path.read_text()).body
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                     and t.id == "REASSIGN_LABEL_NAME_L3" for t in node.targets))
    for item in ontology:
        assert class_map[item["canonical_l3_id"]] == item["label_name"]
        part = metadata.loc[metadata.label_id_dense.eq(item["dense_id"])]
        assert part.label_name.eq(item["label_name"]).all()
        assert part.canonical_l3_id.eq(item["canonical_l3_id"]).all()
    metadata = metadata.sort_values("row_uid").reset_index(drop=True)
    embeddings, encoders = {}, {}
    for seed in SEEDS:
        seed_name = f"seed_{seed}/manifest.json"
        encoder = read_json(bind(EMBEDDINGS / seed_name, extraction["output_sha256"][seed_name]))
        assert encoder["training_seed"] == seed
        assert (encoder["encoder_fitting_rows"], encoder["encoder_fitting_plots"]) == (3378, 1300)
        assert encoder["checkpoint_validation_replay"]["valid"]
        name = f"seed_{seed}/embeddings.npz"
        path = bind(EMBEDDINGS / name, extraction["output_sha256"][name])
        assert sha256(path) == encoder["embedding_file_sha256"]
        with np.load(path, allow_pickle=False) as saved:
            uids = saved["row_uid"]
            assert len(uids) == len(set(uids)) == 5547
            indices = pd.Index(uids).get_indexer(metadata.row_uid)
            assert (indices >= 0).all()
            for column in ("row_uid", "file", "plot_idx"):
                assert np.array_equal(saved[column][indices].astype(str), metadata[column].astype(str))
            assert np.array_equal(saved["cohort"][indices], np.where(metadata.split.eq("cleaned_test"), "test", "development"))
            vectors = saved["image_embedding"][indices].astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        assert vectors.shape == (5547, 1152) and np.isfinite(vectors).all()
        assert np.allclose(norms, 1.0, atol=2e-6)
        embeddings[seed] = vectors / norms[:, None]
        encoders[str(seed)] = {key: encoder[key] for key in (
            "checkpoint_path", "checkpoint_sha256", "encoder_fitting_rows",
            "encoder_fitting_plots", "preprocessing", "checkpoint_validation_replay")}
    return metadata, embeddings, labels, hashes, encoders


def project_embeddings(vectors, train_mask, settings):
    """Projection routine has no access to habitat labels or prediction outcomes."""
    with threadpool_limits(limits=settings["thread_limit"]):
        pca = PCA(n_components=settings["pca_components"], svd_solver=settings["pca_solver"],
                  whiten=settings["pca_whiten"])
        pca.fit(vectors[train_mask])
        reduced = pca.transform(vectors)
        model = TSNE(n_components=settings["tsne_dimensions"], perplexity=settings["perplexity"],
                     early_exaggeration=settings["early_exaggeration"], learning_rate=settings["learning_rate"],
                     max_iter=settings["max_iter"], init=settings["init"], random_state=settings["random_state"],
                     metric=settings["metric"], method=settings["method"], angle=settings["angle"],
                     n_jobs=settings["n_jobs"])
        nonlinear = model.fit_transform(reduced)
    assert np.isfinite(reduced).all() and np.isfinite(nonlinear).all()
    stats = {"pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
             "pca2_training_variance_retained": float(pca.explained_variance_ratio_[:2].sum()),
             "pca50_training_variance_retained": float(pca.explained_variance_ratio_.sum()),
             "tsne_kl_divergence": float(model.kl_divergence_), "tsne_n_iter": int(model.n_iter_),
             "pca_fitting_rows": int(train_mask.sum()), "tsne_fitting_rows": len(vectors)}
    transform = {"mean": pca.mean_, "components": pca.components_,
                 "explained_variance_ratio": pca.explained_variance_ratio_}
    return {"pca": reduced[:, :2], "tsne": nonlinear}, stats, transform


def projection_fingerprint():
    spec = {"routine": inspect.getsource(project_embeddings), "settings": SETTINGS,
            "numerical_versions": {key: environment()[key] for key in ("numpy", "scikit-learn", "threadpoolctl")}}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def get_projections(output, metadata, embeddings, hashes, encoders):
    manifest_path = output / "projection_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest["study_id"] == STUDY
        assert manifest["source_sha256"] == hashes
        assert manifest["projection_fingerprint"] == projection_fingerprint(), "Projection implementation/settings changed; use a new directory"
        for name, expected in manifest["output_sha256"].items():
            assert sha256(output / name) == expected, f"Projection artifact changed: {name}"
        print("Validated existing numeric projections; regenerating figures/report only.", flush=True)
        return pd.read_parquet(output / "coordinates.parquet"), read_json(output / "projection_metrics.json")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing a nonempty directory without an authenticated projection manifest")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pca_transforms").mkdir()
    all_rows, metrics = [], {}
    train_mask = metadata.split.eq("development_train").to_numpy()
    for seed in SEEDS:
        print(f"Encoder seed {seed}: training-fitted PCA50 and joint t-SNE for 5,547 images", flush=True)
        started = time.monotonic()
        coordinates, stats, transform = project_embeddings(embeddings[seed], train_mask, SETTINGS)
        metrics[str(seed)] = stats
        for method, xy in coordinates.items():
            all_rows.append(metadata.assign(training_seed=seed, method=method, x=xy[:, 0], y=xy[:, 1]))
        np.savez_compressed(output / f"pca_transforms/seed_{seed}.npz", **transform)
        print(f"Encoder seed {seed}: finished in {time.monotonic() - started:.1f}s; PCA2 retains "
              f"{100 * stats['pca2_training_variance_retained']:.2f}% of training variance", flush=True)
    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_parquet(output / "coordinates.parquet", index=False)
    write_json(output / "projection_metrics.json", metrics)
    files = ["coordinates.parquet", "projection_metrics.json"] + [f"pca_transforms/seed_{seed}.npz" for seed in SEEDS]
    write_json(manifest_path, {"study_id": STUDY, "source_sha256": hashes, "encoders": encoders,
                              "projection_fingerprint": projection_fingerprint(), "settings": SETTINGS,
                              "environment": environment(), "encoder_fitting_repeated": False,
                              "output_sha256": {name: sha256(output / name) for name in files}})
    return rows, metrics


def validate_coordinates(coordinates, metadata):
    identity = ["row_uid", "file", "plot_idx", "label_id_dense", "canonical_l3_id", "label_name", "split"]
    assert len(coordinates) == 5547 * 4 * 2
    assert not coordinates.duplicated(["training_seed", "method", "row_uid"]).any()
    assert np.isfinite(coordinates[["x", "y"]].to_numpy()).all()
    for seed in SEEDS:
        for method in ("pca", "tsne"):
            part = coordinates.loc[coordinates.training_seed.eq(seed) & coordinates.method.eq(method)]
            pd.testing.assert_frame_equal(part[identity].sort_values("row_uid").reset_index(drop=True),
                                          metadata[identity].sort_values("row_uid").reset_index(drop=True))


def write_report(output, metadata, metrics, labels):
    rows = []
    for label_id, name in enumerate(labels):
        row = {"label_id_dense": label_id, "habitat": name}
        for split in SPLITS:
            selected = metadata.loc[metadata.label_id_dense.eq(label_id) & metadata.split.eq(split)]
            row[f"{split}_images"] = len(selected)
            row[f"{split}_plots"] = selected.plot_idx.nunique()
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "class_support.csv", index=False)
    projection_rows = "\n".join(
        f"| {seed} | {100 * metrics[str(seed)]['pca2_training_variance_retained']:.2f}% | "
        f"{100 * metrics[str(seed)]['pca50_training_variance_retained']:.2f}% | "
        f"{metrics[str(seed)]['tsne_kl_divergence']:.4f} |" for seed in SEEDS)
    figure_links = "\n".join(
        f"| {seed} | [PNG](figures/adapted_image_embedding_seed{seed}_overview.png) · "
        f"[PDF](figures/adapted_image_embedding_seed{seed}_overview.pdf) | "
        f"[Interactive](interactive/adapted_image_embedding_seed{seed}_interactive.html) | "
        f"[Habitat focus](figures/adapted_image_embedding_seed{seed}_habitat_focus.png) |" for seed in SEEDS)
    record = Path(os.path.relpath(REPO / "dev_plans/reliability_aware_geo_fusion/24_adapted_embedding_visualization.md", output)).as_posix()
    report = f"""# Adapted image embeddings: development and cleaned-test maps

These maps visualize the **1,152-dimensional image embeddings from all four habitat-adapted SigLIP2 vision encoders** used in the geometry study. Every encoder was fitted on the same **3,378 development-training images / 1,300 plots**. The displayed data comprise all **4,200 development images** (training plus 822 validation images) and **1,347 cleaned-test images**, using the same seed-specific encoder for every split.

## Development and test in a shared coordinate system

![Adapted image embeddings, encoder seed 1](figures/adapted_image_embedding_seed1_overview.png)

The left column shows development images and the right column shows cleaned test. Habitat colours are fixed across figures. Development-training images are circles and development-validation images are triangles. Seed 1 is the default illustration by numerical order; all four encoder seeds are retained below.

**PCA (top row):** fit unwhitened PCA on the 3,378 training embeddings only, then transform all splits. Training, validation and test share exactly the same axes and limits within a seed. Test embeddings do not determine this basis. The axis percentages describe variance in the training embeddings, not test variance or classification accuracy.

**t-SNE (bottom row):** first project all embeddings through that training-fitted PCA50, then fit one joint t-SNE map on all 5,547 images. Development and test are displayed as separate subsets of this same map, with identical axis limits. Test embeddings influence this layout. This is a label-free, transductive visualization, not an out-of-sample t-SNE transform or an independent test evaluation.

All vectors are separately L2-normalized before projection, as in the geometry policy. Habitat labels only colour and filter points; neither PCA nor t-SNE receives labels. The encoders themselves were trained with habitat labels, so development-training separation must not be treated as unseen-data generalization.

## All four adapted encoder seeds

![Development and cleaned test, joint t-SNE for all four encoders](figures/adapted_image_embedding_all_seeds_tsne.png)

Each column is a different encoder with its own independently fitted PCA/t-SNE maps. Compare development and test within a column. Axes, rotations and global distances are not aligned across encoder seeds. The four runs visualize the same photographs, not four independent datasets.

| Encoder seed | Overview | Interactive map | Confusing habitat groups |
| --- | --- | --- | --- |
{figure_links}

Interactive maps support zoom, point inspection and habitat filtering. Hover information identifies the photograph, plot, habitat and split. The PCA/t-SNE selector changes the displayed projection; it does not recompute an embedding. The HTML files are standalone and include their plotting runtime.

## Previously confused habitat groups

![Grasslands and heath/bog/wetland groups on the same maps](figures/adapted_image_embedding_seed1_habitat_focus.png)

The focus panels highlight Improved, Neutral, Calcareous and Acid Grassland, and Dwarf Shrub Heath, Bog, Fen/Marsh/Swamp and Acid Grassland. They reuse each seed's global joint t-SNE coordinates, without fitting a new map to a selected class subset. Grey points retain the remaining habitats as context. These groups follow the existing confusion analysis rather than a search for attractive clusters.

Visually, Arable/Horticulture and the two woodland classes form relatively distinct groups in these t-SNE maps. Improved and Neutral Grassland occupy adjacent regions with mixed boundaries; Acid Grassland, Bog and Dwarf Shrub Heath also show overlap. These patterns are visible in both development and cleaned-test panels across the four encoders. Differences in sample count make the test panels sparser, so point density alone should not be read as a change in class separability.

Apparent clusters and overlap are descriptive. A two-dimensional projection can hide relevant directions or create apparent separation; t-SNE cluster spacing and density do not measure original-space class distance or model reliability. Consult the [confusion-matrix review](../geometry_router_confusion_review_v1/report.md) for measured class errors and the [adapted-geometry study](../joint_geometry_adapted_encoder_v1/report.md) for correction performance. The router continues to use full-dimensional cosine distances, not these plotted coordinates.

## Projection settings and support

| Encoder seed | Variance retained by PCA2 | Variance retained by PCA50 | Joint t-SNE KL divergence |
| --- | --- | --- | --- |
{projection_rows}

PCA uses full SVD, 50 components, and no whitening. t-SNE uses Euclidean distances in PCA50, perplexity 30, early exaggeration 12, automatic learning rate, PCA initialization, 1,500 maximum iterations, Barnes–Hut angle 0.5 and random seed **20261005**. These settings are fixed across encoder seeds, without selecting a preferred layout. KL divergence is a projection diagnostic, not a model-quality ranking across encoders.

| Split | Images | Plots | Seen during encoder fitting? |
| --- | ---: | ---: | --- |
| Development training | 3,378 | 1,300 | Yes |
| Development validation | 822 | 325 | No |
| Cleaned test | 1,347 | 531 | No |

In the image panels, points are photographs. Multiple photos from the same plot are correlated, and habitat frequencies differ between splits; dense regions do not imply independent environmental support. Plot identities are disjoint across the three splits. [Class support](class_support.csv) reports both image and plot counts, including habitats with no test examples. All photographs are included in the image maps; there is no class-balanced subsampling or class-specific projection fitting. Geo panels instead show the corresponding plots once each, as described above.

## Reproduction and provenance

From the repository root, using the existing `habcls` environment:

```bash
python tools/plot_multimodal_geo_helpfulness_adapted_embeddings.py
```

The command authenticates the cached embedding arrays and their extraction manifests, joins development assignments and the already scored cleaned-test labels by image identity, and verifies the ontology and split counts. It uses the legacy image preprocessing recorded in the extraction manifests, including the BGR/forced-resize convention; no new encoder extraction is performed.

At a completed destination, the command checks input hashes, the numerical projection implementation/settings and saved output hashes, then regenerates figures and this report from the saved coordinates. To recompute PCA and t-SNE from the authenticated cached embeddings, use a fresh destination:

```bash
python tools/plot_multimodal_geo_helpfulness_adapted_embeddings.py \\
  --output-dir multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_embedding_visualization_reproduction_v1
```

Register any retained reproduction directory in the study catalog. No encoder/classifier fitting, image inference, OOF diagnostic, router threshold search or new classification scoring is performed. Labels from the previously explored cleaned test are shown at the user's request; these maps provide exploratory interpretation, not independent confirmation.

Files: [coordinates](coordinates.parquet), [projection metrics](projection_metrics.json), [numerical projection manifest](projection_manifest.json), [source/output provenance](provenance.json), and per-seed numerical PCA transforms in `pca_transforms/`. The manifests record source/encoder hashes, package versions, projection settings and unchanged test-registry files. [Development record]({record}) indexes the report.
"""
    (output / "report.md").write_text(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", default=str(BASE / STUDY))
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output = (REPO / output).resolve() if not output.is_absolute() else output.resolve()
    if output.parent != BASE.resolve() or not output.name.startswith("adapted_embedding_visualization_"):
        parser.error("Output must be an adapted_embedding_visualization_* directory directly under the canonical artifact root")
    registry = REPO / "multimodal_artifacts/locked_test_registry/cs/gse_100m_cleaned_test/events"
    before = {p.relative_to(REPO).as_posix(): sha256(p) for p in registry.rglob("*") if p.is_file()}
    metadata, embeddings, labels, hashes, encoders = load_sources()
    coordinates, metrics = get_projections(output, metadata, embeddings, hashes, encoders)
    validate_coordinates(coordinates, metadata)
    from multimodal.geo_helpfulness_adapted_embedding_figures import render_figures
    figure_paths = render_figures(output, coordinates, metrics, labels)
    write_report(output, metadata, metrics, labels)
    from multimodal.geo_helpfulness_geo_embedding_visualization import get_geo_projections, extend_report
    from multimodal.geo_helpfulness_image_geo_figures import render_figures as render_comparison
    geo_coordinates, geo_metrics, combined_hashes = get_geo_projections(output, metadata, hashes)
    figure_paths.extend(render_comparison(output, coordinates, metrics, geo_coordinates, geo_metrics, labels))
    extend_report(output, geo_coordinates, geo_metrics, labels)
    hashes = combined_hashes
    after = {p.relative_to(REPO).as_posix(): sha256(p) for p in registry.rglob("*") if p.is_file()}
    assert before == after
    assert all(sha256(REPO / name) == expected for name, expected in hashes.items())
    sources = [Path(__file__), REPO / "multimodal/geo_helpfulness_adapted_embedding_figures.py",
               REPO / "multimodal/geo_helpfulness_geo_embedding_visualization.py",
               REPO / "multimodal/geo_helpfulness_image_geo_figures.py",
               REPO / "tools/plot_multimodal_geo_helpfulness_adapted_embeddings.py"]
    write_json(output / "provenance.json", {
        "study_id": STUDY, "source_sha256": hashes,
        "implementation_sha256": {p.relative_to(REPO).as_posix(): sha256(p) for p in sources},
        "environment": environment(), "encoder_training_repeated": False,
        "projection_labels_used": False, "tsne_uses_test_embeddings": True,
        "pca_uses_test_embeddings_for_fitting": False,
        "geo_projection_unit": "one point per plot", "geo_coordinate_standardization": False,
        "geo_shared_across_encoder_seeds": True,
        "registry_files": len(before), "registry_unchanged": True, "sources_unchanged": True,
        "figure_files": figure_paths,
        "output_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*")
                          if p.is_file() and p.name != "provenance.json"}})
    print(f"Report: {output / 'report.md'}", flush=True)
    return 0

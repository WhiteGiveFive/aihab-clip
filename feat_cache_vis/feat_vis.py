"""
This program visualises image embeddings. 
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px

# Confirm cache inputs

# Define visualisation methods.
    # PCA pre-redcution
    # UMP
    # t-SNE
    # save the 2D embeddings. 

# Plotting
    # Static PNG
    # Color by class
    # Save figures

    # PLotly interactive webpage. 

import argparse


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize cached image embeddings in 2D.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Path to cache folder containing embeddings.pt and metadata.csv.",
    )

    reducer = parser.add_mutually_exclusive_group(required=True)
    reducer.add_argument("--umap", action="store_true", help="Use UMAP for 2D reduction.")
    reducer.add_argument("--tsne", action="store_true", help="Use t-SNE for 2D reduction.")

    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Optional PCA pre-reduction dimension.",
    )

    parser.add_argument(
        "--out_prefix",
        type=str,
        default="vis",
        help="Output prefix for coords and plot files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )

    parser.add_argument(
        "--color_by",
        type=str,
        default="ground_truth_word_label",
        help="Column in metadata.csv to color points.",
    )
    parser.add_argument(
        "--hover",
        type=str,
        default="file_name,ground_truth_word_label,ground_truth_num_label",
        help="Comma-separated metadata columns to show on hover.",
    )

    # UMAP hyperparameters
    parser.add_argument("--umap_neighbors", type=int, default=30, help="UMAP n_neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--umap_metric", type=str, default="euclidean", help="UMAP metric.")

    # t-SNE hyperparameters
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--tsne_metric", type=str, default="cosine", help="t-SNE metric.")
    parser.add_argument("--tsne_init", type=str, default="pca", help="t-SNE init method.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")

    return parser


def load_cache(cache_dir: str):
    cache_root = Path(cache_dir)
    emb_path = cache_root / "embeddings.pt"
    meta_path = cache_root / "metadata.csv"
    lab_path = cache_root / "labels.pt"

    if not emb_path.is_file():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    embeddings = torch.load(emb_path, map_location="cpu")
    metadata = pd.read_csv(meta_path)
    labels = torch.load(lab_path, map_location="cpu") if lab_path.is_file() else None

    return embeddings, metadata, labels


def main():
    args = build_argparser().parse_args()
    embeddings, metadata, labels = load_cache(args.cache_dir)

    print({
        "embeddings.shape": tuple(embeddings.shape),
        "metadata.rows": int(len(metadata)),
        "labels.shape": tuple(labels.shape) if labels is not None else None,
    })

    features = embeddings.detach().cpu().numpy()

    # Optional PCA dimension pre-reduction
    if args.pca_dim is not None:
        pca_dim = int(args.pca_dim)
        if pca_dim <= 0:
            raise ValueError("pca_dim must be a positive integer.")
        if pca_dim >= features.shape[1]:
            print(f"[warn] pca_dim={pca_dim} >= embedding_dim={features.shape[1]}; skipping PCA.")
        else:
            pca = PCA(n_components=pca_dim, random_state=args.seed)
            features = pca.fit_transform(features)
            print({
                "pca_dim": pca_dim,
                "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                "features.shape": tuple(features.shape),
            })
    # UMAP or t-SNE dimension reduction
    if args.umap:
        reducer = umap.UMAP(
            n_components=2, 
            n_neighbors=int(args.umap_neighbors),
            min_dist=float(args.umap_min_dist),
            metric=str(args.umap_metric),
            random_state=int(args.seed),
        )
        coords_2d = reducer.fit_transform(features)
        reducer_name = "umap"
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=float(args.tsne_perplexity),
            metric=str(args.tsne_metric),
            init=str(args.tsne_init),
            random_state=int(args.seed),
        )
        coords_2d = reducer.fit_transform(features)
        reducer_name = "tsne"

    print({
        "reducer": reducer_name,
        "coords_2d.shape": tuple(coords_2d.shape),
    })

    # Save the 2D coordiantes generated by UMAP or t-SNE
    coords_path = Path(args.cache_dir) / f"{args.out_prefix}_{reducer_name}_coords.npy"
    if coords_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {coords_path} (use --overwrite to replace)")
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(coords_path, coords_2d)
    print(f"[save] coords -> {coords_path}")

    if coords_2d.shape[0] != len(metadata):
        raise ValueError(
            f"Row mismatch: coords_2d has {coords_2d.shape[0]} rows, "
            f"metadata has {len(metadata)} rows."
        )

    df_plot = metadata.copy()
    df_plot["x"] = coords_2d[:, 0]
    df_plot["y"] = coords_2d[:, 1]

    hover_cols = [c.strip() for c in str(args.hover).split(",") if c.strip()]
    hover_cols = [c for c in hover_cols if c in df_plot.columns]
    if not hover_cols:
        hover_cols = None

    color_col = args.color_by if args.color_by in df_plot.columns else None
    if color_col is None and args.color_by:
        print(f"[warn] color_by column '{args.color_by}' not found; plotting without color.")

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color=color_col,
        hover_data=hover_cols,
        title=f"{reducer_name.upper()} Embeddings",
    )

    html_path = Path(args.cache_dir) / f"{args.out_prefix}_{reducer_name}.html"
    if html_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {html_path} (use --overwrite to replace)")
    fig.write_html(str(html_path))
    print(f"[save] plot -> {html_path}")


if __name__ == "__main__":
    main()

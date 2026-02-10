"""
Skeleton utilities for CS test-set outlier detection and cleaning.

Goal: provide a small, swappable API for embedding-geometry methods
(centroid first, multi-prototype later) without wiring full logic yet.

Intended usage (to be filled in later):
- load_cache(...) to read embeddings/labels/metadata from a cached split
- compute_centroids(...) to build per-class prototypes
- score_centroid_distance(...) to compute similarity/outlier scores
- select_outliers(...) to apply a fixed, reproducible rule
- materialize_clean_split(...) to build a cleaned folder of kept images

All functions below are placeholders; concrete implementations will be
added once thresholds and rules are finalized.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F


@dataclass
class CachePaths:
    """Container for cache paths derived from a cache_dir root."""

    cache_dir: Path
    embeddings: Path
    labels: Path
    metadata: Path
    meta_json: Optional[Path] = None


@dataclass
class CentroidResult:
    """Holds per-class centroids and basic stats."""

    centroids: Dict[int, torch.Tensor]
    class_counts: Dict[int, int]
    dim: int


def resolve_cache_paths(cache_dir: Path) -> CachePaths:
    """
    Build canonical file paths for an embedding cache folder.

    The layout mirrors cache_openclip_embeddings in aihab_utils/feature_cache.py.
    """

    cache_dir = Path(cache_dir)
    return CachePaths(
        cache_dir=cache_dir,
        embeddings=cache_dir / "embeddings.pt",
        labels=cache_dir / "labels.pt",
        metadata=cache_dir / "metadata.csv",
        meta_json=cache_dir / "meta.json",
    )


def _load_tensor_cpu(path: Path) -> torch.Tensor:
    """
    Load a tensor from disk onto CPU.

    Uses `weights_only=True` when supported by the local torch version.
    """

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"Expected tensor at '{path}', got {type(loaded).__name__}.")
    return loaded


def _validate_embeddings_labels(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    allow_empty: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Validate and normalize embedding/label tensors used by cache/scoring helpers.

    Returns:
        embeddings: unchanged tensor (must be [N, D], floating)
        labels: coerced to shape [N] and dtype torch.long
        num_samples: N
        dim: D
    """

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {tuple(embeddings.shape)}.")

    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape [N], got {tuple(labels.shape)}.")

    if not torch.is_floating_point(embeddings):
        raise TypeError(f"Expected floating embeddings, got dtype={embeddings.dtype}.")

    labels = labels.to(torch.long)

    num_samples, dim = int(embeddings.shape[0]), int(embeddings.shape[1])
    if int(labels.shape[0]) != num_samples:
        raise ValueError(
            "Row mismatch between embeddings and labels: "
            f"{num_samples} vs {int(labels.shape[0])}."
        )
    if not allow_empty and num_samples == 0:
        raise ValueError("Empty inputs: no samples available.")

    return embeddings, labels, num_samples, dim


def load_cache(paths: CachePaths) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """
    Load embeddings, labels, and metadata with strict row-alignment checks.
    """
    # Output warning if required files are missing.
    missing_files = [
        p for p in (paths.embeddings, paths.labels, paths.metadata) if not Path(p).is_file()
    ]
    if missing_files:
        missing_str = ", ".join(str(p) for p in missing_files)
        raise FileNotFoundError(f"Missing cache file(s): {missing_str}")

    embeddings = _load_tensor_cpu(paths.embeddings)
    labels = _load_tensor_cpu(paths.labels)
    metadata = pd.read_csv(paths.metadata)

    # Ensure tensor shape/dtype and row numbers match.
    embeddings, labels, num_samples, _ = _validate_embeddings_labels(
        embeddings, labels, allow_empty=False
    )
    if int(len(metadata)) != num_samples:
        raise ValueError(
            "Row mismatch between embeddings and metadata: "
            f"{num_samples} vs {int(len(metadata))}."
        )
    
    # Ensure metadata has the required columns
    required_columns = {"file_name", "ground_truth_num_label"}
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(
            "metadata.csv is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    # Output warning if the labels in labels.pt and metadata.csv do not match
    meta_labels = torch.tensor(
        metadata["ground_truth_num_label"].astype(int).to_numpy(),
        dtype=torch.long,
    )
    mismatch_idx = torch.nonzero(meta_labels != labels, as_tuple=False).flatten()
    if int(mismatch_idx.numel()) > 0:
        first_bad = int(mismatch_idx[0].item())
        raise ValueError(
            "Label mismatch between labels.pt and metadata.csv at row "
            f"{first_bad}: labels.pt={int(labels[first_bad])}, "
            f"metadata.csv={int(meta_labels[first_bad])}."
        )

    return embeddings, labels, metadata


def compute_centroids(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    trim_frac: Optional[float] = None,
    normalize_tol: float = 1e-3,
    eps: float = 1e-12,
) -> CentroidResult:
    """
    Compute per-class centroids for the labels present in `labels`.

    - If embeddings are not already L2-normalized (within `normalize_tol`), they are
      normalized defensively before centroiding.
    - Centroids are always L2-normalized before returning.

    Note: a trimmed-centroid mode (drop worst `trim_frac` samples per class and
    recompute the mean) is intentionally left as a placeholder to keep the first
    end-to-end workflow simple. If you pass trim_frac for now, this function will
    raise NotImplementedError.
    """

    if trim_frac is not None:
        raise NotImplementedError("trim_frac is not implemented yet (planned follow-up).")

    embeddings, labels, _, dim = _validate_embeddings_labels(
        embeddings, labels, allow_empty=False
    )
    emb = embeddings.detach()
    lab = labels.detach()

    # Defensive L2 normalization (cache is typically normalized, but not guaranteed).
    norms = emb.norm(dim=-1)
    if not torch.isfinite(norms).all():
        raise ValueError("Non-finite embedding norms found (NaN/Inf).")

    max_dev = float((norms - 1.0).abs().max().item())
    if max_dev > float(normalize_tol):
        print(f"[warn] Unnormalized embeddings detected (max |norm-1|={max_dev:.3e}); normalizing.")
        emb = emb / norms.clamp_min(eps).unsqueeze(1)

    uniq, inv = torch.unique(lab, sorted=True, return_inverse=True)
    num_classes_present = int(uniq.numel())
    if num_classes_present == 0:
        raise ValueError("No labels present to compute centroids.")

    # Sum embeddings per class using inverse indices, then divide by counts.
    sums = torch.zeros((num_classes_present, dim), dtype=emb.dtype, device=emb.device)
    sums.index_add_(0, inv, emb) # read index_add_ at https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html. 

    ones = torch.ones_like(inv, dtype=torch.long)
    counts = torch.zeros(num_classes_present, dtype=torch.long, device=inv.device)
    counts.index_add_(0, inv, ones)
    if int(counts.min().item()) <= 0:
        raise RuntimeError("Internal error: empty class bin when computing centroids.")

    means = sums / counts.to(dtype=emb.dtype).unsqueeze(1)
    means = F.normalize(means, p=2, dim=-1, eps=eps)

    centroids_dict: Dict[int, torch.Tensor] = {}
    class_counts: Dict[int, int] = {}
    for i in range(num_classes_present):
        label_id = int(uniq[i].item())
        centroids_dict[label_id] = means[i]
        class_counts[label_id] = int(counts[i].item())

    return CentroidResult(centroids=centroids_dict, class_counts=class_counts, dim=dim)


def score_centroid_distance(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    centroids: CentroidResult,
    metadata: pd.DataFrame,
    *,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute cosine-to-centroid outlier scores and merge with per-sample metadata.

    Scoring:
      - sim_to_centroid = cosine(embedding_i, centroid_of_true_label)
      - outlier_score = 1 - sim_to_centroid

    Returns a DataFrame sorted by descending outlier_score with class-wise rank fields.
    """
    # Data shape/consistency check
    embeddings, labels, num_samples, dim = _validate_embeddings_labels(
        embeddings, labels, allow_empty=False
    )
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(f"Expected metadata to be pandas DataFrame, got {type(metadata).__name__}.")
    if int(len(metadata)) != num_samples:
        raise ValueError(
            "Row mismatch between embeddings and metadata: "
            f"{num_samples} vs {int(len(metadata))}."
        )
    if int(centroids.dim) != dim:
        raise ValueError(f"Centroid dim mismatch: expected {dim}, got {int(centroids.dim)}.")
    # Can be removed as this has been checked in `load_cache` helper. 
    if "ground_truth_num_label" in metadata.columns:
        meta_labels = metadata["ground_truth_num_label"].astype(int).to_numpy()
        label_np = labels.detach().cpu().numpy()
        mismatch = (meta_labels != label_np).nonzero()[0]
        if mismatch.size > 0:
            first_bad = int(mismatch[0])
            raise ValueError(
                "Label mismatch between labels tensor and metadata at row "
                f"{first_bad}: labels={int(label_np[first_bad])}, "
                f"metadata={int(meta_labels[first_bad])}."
            )

    emb = embeddings.detach()
    if not torch.isfinite(emb).all():
        raise ValueError("Non-finite embeddings found (NaN/Inf).")
    # emb = F.normalize(emb, p=2, dim=-1, eps=eps)    # normalised again?

    lab = labels.detach()
    uniq, inv = torch.unique(lab, sorted=True, return_inverse=True)
    missing_centroids = [int(label.item()) for label in uniq if int(label.item()) not in centroids.centroids]
    if missing_centroids:
        raise ValueError(
            "Missing centroid(s) for label(s): "
            + ", ".join(str(label) for label in sorted(missing_centroids))
        )

    centroid_rows = torch.stack(
        [centroids.centroids[int(label.item())] for label in uniq], dim=0
    ).to(device=emb.device, dtype=emb.dtype)
    # centroid_rows = F.normalize(centroid_rows, p=2, dim=-1, eps=eps)    # Normalised again? 
    sample_centroids = centroid_rows[inv]

    # sim_to_centroid = (emb * sample_centroids).sum(dim=-1).clamp(min=-1.0, max=1.0)   # can be replaced by F.cosine_similarity
    sim_to_centroid = F.cosine_similarity(emb, sample_centroids, dim=-1, eps=eps)
    outlier_score = 1.0 - sim_to_centroid

    scores = metadata.copy().reset_index(drop=True)
    label_series = labels.detach().cpu().numpy().astype(int)
    sim_series = sim_to_centroid.detach().cpu().numpy()
    outlier_series = outlier_score.detach().cpu().numpy()

    # Trust labels tensor as canonical label source for downstream ranking and joins.
    scores["ground_truth_num_label"] = label_series
    if "ground_truth_word_label" not in scores.columns:
        scores["ground_truth_word_label"] = ""
    if "ground_truth_L2_num_label" not in scores.columns:
        scores["ground_truth_L2_num_label"] = -1
    if "file_name" not in scores.columns:
        scores["file_name"] = ""

    scores["sim_to_centroid"] = sim_series
    scores["outlier_score"] = outlier_series
    scores["class_size"] = scores["ground_truth_num_label"].map(centroids.class_counts).astype(int)
    scores["rank_in_class"] = (
        scores.groupby("ground_truth_num_label")["outlier_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    scores["pct_rank_in_class"] = scores["rank_in_class"] / scores["class_size"]

    sim_p05 = scores.groupby("ground_truth_num_label")["sim_to_centroid"].transform(
        lambda col: col.quantile(0.05)
    )
    scores["is_bottom_5pct"] = scores["sim_to_centroid"] <= sim_p05

    out_columns = [
        "file_name",
        "ground_truth_num_label",
        "ground_truth_word_label",
        "ground_truth_L2_num_label",
        "sim_to_centroid",
        "outlier_score",
        "class_size",
        "rank_in_class",
        "pct_rank_in_class",
        "is_bottom_5pct",
    ]
    return scores[out_columns].sort_values(
        by=["outlier_score", "ground_truth_num_label", "file_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def select_outliers(
    scores: pd.DataFrame,
    rule: Dict,
) -> pd.DataFrame:
    """
    Placeholder: apply a fixed selection rule (e.g., per-class quantile or MAD).

    `rule` should be serializable (to capture in rule.json) and include parameters
    like method, per-class q or k, seed, and any trimming used upstream.
    """

    # TODO: implement per-class filtering logic returning a DataFrame subset.
    raise NotImplementedError


def materialize_clean_split(
    source_image_root: Path,
    index_csv: Path,
    keep_list: List[str],
    output_root: Path,
    *,
    symlink: bool = True,
) -> Path:
    """
    Placeholder: create a cleaned test folder with kept images and an index CSV.

    - keep_list: filenames to retain (relative to source_image_root)
    - when symlink=True, create symlinks to avoid duplication.
    """

    # TODO: implement directory creation, linking/copying, and CSV copy.
    raise NotImplementedError

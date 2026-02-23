"""
Utilities for CS outlier detection using embedding-geometry methods.

Current workflow supports:
- loading cached embeddings/labels/metadata
- computing single-centroid class prototypes via `SingleCentroidScorer`
- computing multi-prototype class modes via `MultiPrototypeScorer`
- scoring samples by cosine distance to true-class centroid/prototype

Selection/materialization helpers remain placeholders.
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


@dataclass
class MultiPrototypeResult:
    """Holds per-class multi-prototype results and class statistics."""

    prototypes: Dict[int, torch.Tensor]
    class_counts: Dict[int, int]
    prototype_counts: Dict[int, List[int]]
    k_per_class: Dict[int, int]
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
    missing_files = [
        p for p in (paths.embeddings, paths.labels, paths.metadata) if not Path(p).is_file()
    ]
    if missing_files:
        missing_str = ", ".join(str(p) for p in missing_files)
        raise FileNotFoundError(f"Missing cache file(s): {missing_str}")

    embeddings = _load_tensor_cpu(paths.embeddings)
    labels = _load_tensor_cpu(paths.labels)
    metadata = pd.read_csv(paths.metadata)

    embeddings, labels, num_samples, _ = _validate_embeddings_labels(
        embeddings, labels, allow_empty=False
    )
    if int(len(metadata)) != num_samples:
        raise ValueError(
            "Row mismatch between embeddings and metadata: "
            f"{num_samples} vs {int(len(metadata))}."
        )

    required_columns = {"file_name", "ground_truth_num_label"}
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(
            "metadata.csv is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

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


class SingleCentroidScorer:
    """
    Single-centroid scorer that owns validated tensors/metadata and cached centroids.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        metadata: pd.DataFrame,
        *,
        normalize_tol: float = 1e-3,
        eps: float = 1e-12,
    ):
        embeddings, labels, num_samples, dim = _validate_embeddings_labels(
            embeddings, labels, allow_empty=False
        )
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError(
                f"Expected metadata to be pandas DataFrame, got {type(metadata).__name__}."
            )
        if int(len(metadata)) != num_samples:
            raise ValueError(
                "Row mismatch between embeddings and metadata: "
                f"{num_samples} vs {int(len(metadata))}."
            )

        self.embeddings = embeddings.detach()
        self.labels = labels.detach()
        self.metadata = metadata.copy().reset_index(drop=True)
        self.num_samples = num_samples
        self.dim = dim
        self.normalize_tol = float(normalize_tol)
        self.eps = float(eps)
        self._centroids: Optional[CentroidResult] = None
        self._normalized_embeddings: Optional[torch.Tensor] = None

        if "ground_truth_num_label" in self.metadata.columns:
            meta_labels = torch.tensor(
                self.metadata["ground_truth_num_label"].astype(int).to_numpy(),
                dtype=torch.long,
            )
            label_cpu = self.labels.detach().cpu()
            mismatch_idx = torch.nonzero(meta_labels != label_cpu, as_tuple=False).flatten()
            if int(mismatch_idx.numel()) > 0:
                first_bad = int(mismatch_idx[0].item())
                raise ValueError(
                    "Label mismatch between labels tensor and metadata at row "
                    f"{first_bad}: labels={int(label_cpu[first_bad])}, "
                    f"metadata={int(meta_labels[first_bad])}."
                )

    def _get_normalized_embeddings(self) -> torch.Tensor:
        """
        Return L2-normalized embeddings, normalizing on-demand when needed.
        """
        if self._normalized_embeddings is not None:
            return self._normalized_embeddings

        emb = self.embeddings
        norms = emb.norm(dim=-1)
        if not torch.isfinite(norms).all():
            raise ValueError("Non-finite embedding norms found (NaN/Inf).")

        max_dev = float((norms - 1.0).abs().max().item())
        if max_dev > self.normalize_tol:
            print(
                f"[warn] Unnormalized embeddings detected (max |norm-1|={max_dev:.3e}); normalizing."
            )
            emb = emb / norms.clamp_min(self.eps).unsqueeze(1)

        self._normalized_embeddings = emb
        return self._normalized_embeddings

    def compute_centroids(self, *, trim_frac: Optional[float] = None) -> CentroidResult:
        """
        Compute and cache per-class centroids for labels present in this scorer.

        `trim_frac` is reserved for future trimmed-centroid support.
        """
        if trim_frac is not None:
            raise NotImplementedError("trim_frac is not implemented yet (planned follow-up).")
        if self._centroids is not None:
            return self._centroids

        emb = self._get_normalized_embeddings()
        lab = self.labels

        uniq, inv = torch.unique(lab, sorted=True, return_inverse=True)
        num_classes_present = int(uniq.numel())
        if num_classes_present == 0:
            raise ValueError("No labels present to compute centroids.")

        sums = torch.zeros((num_classes_present, self.dim), dtype=emb.dtype, device=emb.device)
        sums.index_add_(0, inv, emb)

        ones = torch.ones_like(inv, dtype=torch.long)
        counts = torch.zeros(num_classes_present, dtype=torch.long, device=inv.device)
        counts.index_add_(0, inv, ones)
        if int(counts.min().item()) <= 0:
            raise RuntimeError("Internal error: empty class bin when computing centroids.")

        means = sums / counts.to(dtype=emb.dtype).unsqueeze(1)
        means = F.normalize(means, p=2, dim=-1, eps=self.eps)

        centroids_dict: Dict[int, torch.Tensor] = {}
        class_counts: Dict[int, int] = {}
        for i in range(num_classes_present):
            label_id = int(uniq[i].item())
            centroids_dict[label_id] = means[i]
            class_counts[label_id] = int(counts[i].item())

        self._centroids = CentroidResult(
            centroids=centroids_dict,
            class_counts=class_counts,
            dim=self.dim,
        )
        return self._centroids

    def score_centroid_distance(
        self, *, centroids: Optional[CentroidResult] = None
    ) -> pd.DataFrame:
        """
        Score points by cosine distance to true-class centroid.
        """
        centroid_result = centroids
        if centroid_result is None:
            centroid_result = self._centroids if self._centroids is not None else self.compute_centroids()

        if int(centroid_result.dim) != self.dim:
            raise ValueError(
                f"Centroid dim mismatch: expected {self.dim}, got {int(centroid_result.dim)}."
            )

        emb = self.embeddings
        if not torch.isfinite(emb).all():
            raise ValueError("Non-finite embeddings found (NaN/Inf).")

        lab = self.labels
        uniq, inv = torch.unique(lab, sorted=True, return_inverse=True)
        missing_centroids = [
            int(label.item())
            for label in uniq
            if int(label.item()) not in centroid_result.centroids
        ]
        if missing_centroids:
            raise ValueError(
                "Missing centroid(s) for label(s): "
                + ", ".join(str(label) for label in sorted(missing_centroids))
            )

        centroid_rows = torch.stack(
            [centroid_result.centroids[int(label.item())] for label in uniq], dim=0
        ).to(device=emb.device, dtype=emb.dtype)
        if not torch.isfinite(centroid_rows).all():
            raise ValueError("Non-finite centroid values found (NaN/Inf).")
        sample_centroids = centroid_rows[inv]   # tensor are copied and extended from centroid_rows to sample_centroids. 

        sim_to_centroid = F.cosine_similarity(emb, sample_centroids, dim=-1, eps=self.eps)
        outlier_score = 1.0 - sim_to_centroid

        scores = self.metadata.copy().reset_index(drop=True)
        label_series = lab.detach().cpu().numpy().astype(int)
        sim_series = sim_to_centroid.detach().cpu().numpy()
        outlier_series = outlier_score.detach().cpu().numpy()

        scores["ground_truth_num_label"] = label_series
        if "ground_truth_word_label" not in scores.columns:
            scores["ground_truth_word_label"] = ""
        if "ground_truth_L2_num_label" not in scores.columns:
            scores["ground_truth_L2_num_label"] = -1
        if "file_name" not in scores.columns:
            scores["file_name"] = ""

        scores["sim_to_centroid"] = sim_series
        scores["outlier_score"] = outlier_series
        scores["class_size"] = (
            scores["ground_truth_num_label"].map(centroid_result.class_counts).astype(int)
        )
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


class MultiPrototypeScorer(SingleCentroidScorer):
    """
    Multi-prototype scorer that models each class with K modes.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        metadata: pd.DataFrame,
        *,
        normalize_tol: float = 1e-3,
        eps: float = 1e-12,
    ):
        super().__init__(
            embeddings=embeddings,
            labels=labels,
            metadata=metadata,
            normalize_tol=normalize_tol,
            eps=eps,
        )
        self._prototypes: Optional[MultiPrototypeResult] = None
        self._prototype_config: Optional[Tuple[str, int, int, int, int, int, int]] = None

    def compute_prototypes(
        self,
        *,
        k_mode: str = "heuristic",
        k_fixed: int = 2,
        k_max: int = 4,
        min_samples_per_proto: int = 15,
        random_state: int = 0,
        n_init: int = 10,
        max_iter: int = 100,
    ) -> MultiPrototypeResult:
        """
        Compute and cache per-class K prototypes using spherical k-means approximation.
        """
        if k_mode not in {"heuristic", "fixed"}:
            raise ValueError(f"Unsupported k_mode '{k_mode}'. Expected one of: heuristic, fixed.")
        if int(k_fixed) < 1:
            raise ValueError(f"k_fixed must be >= 1, got {k_fixed}.")
        if int(k_max) < 1:
            raise ValueError(f"k_max must be >= 1, got {k_max}.")
        if int(min_samples_per_proto) < 1:
            raise ValueError(
                f"min_samples_per_proto must be >= 1, got {min_samples_per_proto}."
            )
        if int(n_init) < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}.")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}.")

        config = (
            str(k_mode),
            int(k_fixed),
            int(k_max),
            int(min_samples_per_proto),
            int(random_state),
            int(n_init),
            int(max_iter),
        )
        # Reuse cached prototypes only when full clustering config is identical.
        if self._prototypes is not None and self._prototype_config == config:
            return self._prototypes

        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for multi-prototype scoring. "
                "Install it to use MultiPrototypeScorer."
            ) from exc

        emb = self._get_normalized_embeddings()
        lab = self.labels
        uniq = torch.unique(lab, sorted=True)
        if int(uniq.numel()) == 0:
            raise ValueError("No labels present to compute prototypes.")

        prototypes: Dict[int, torch.Tensor] = {}
        class_counts: Dict[int, int] = {}
        prototype_counts: Dict[int, List[int]] = {} # for each class label, the number of samples assigned to each prototype.
        k_per_class: Dict[int, int] = {}

        for label in uniq:
            label_id = int(label.item())
            class_mask = lab == label
            x_c = emb[class_mask]
            n_c = int(x_c.shape[0])
            class_counts[label_id] = n_c

            if k_mode == "heuristic":
                # Conservative class-size heuristic for number of modes.
                if n_c < 20:
                    base_k = 1
                elif n_c < 100:
                    base_k = 3
                elif n_c < 200:
                    base_k = 4
                elif n_c < 300:
                    base_k = 5
                else:
                    base_k = 6
            else:
                base_k = int(k_fixed)

            # Safety cap avoids unstable tiny prototypes for small classes.
            base_k = min(base_k, int(k_max))
            safety_cap = max(1, n_c // int(min_samples_per_proto))
            k_c = min(base_k, n_c, safety_cap)
            k_c = max(1, int(k_c))

            if k_c == 1:
                # Fast fallback: for single mode, use normalized class mean directly.
                class_center = x_c.mean(dim=0, keepdim=True)
                class_center = F.normalize(class_center, p=2, dim=-1, eps=self.eps)
                if not torch.isfinite(class_center).all():
                    raise ValueError(
                        f"Non-finite prototype found for class {label_id} (k=1)."
                    )
                prototypes[label_id] = class_center
                prototype_counts[label_id] = [n_c]
                k_per_class[label_id] = 1
                continue

            # Spherical k-means approximation: run k-means on unit-normalized embeddings.
            x_np = x_c.detach().cpu().to(torch.float32).numpy()
            kmeans = KMeans(
                n_clusters=k_c,
                random_state=int(random_state),
                n_init=int(n_init),
                max_iter=int(max_iter),
            )
            kmeans.fit(x_np)

            centers = torch.from_numpy(kmeans.cluster_centers_).to(
                device=x_c.device, dtype=emb.dtype
            )
            centers = F.normalize(centers, p=2, dim=-1, eps=self.eps)
            if not torch.isfinite(centers).all():
                raise ValueError(
                    f"Non-finite prototype centers found for class {label_id}."
                )

            # Re-assign by cosine to derive stable prototype counts after center normalization.
            sim_c = x_c @ centers.t()
            if not torch.isfinite(sim_c).all():
                raise ValueError(
                    f"Non-finite prototype similarity matrix found for class {label_id}."
                )
            assign_c = sim_c.argmax(dim=1)
            counts_c = torch.bincount(assign_c, minlength=k_c)

            prototypes[label_id] = centers
            prototype_counts[label_id] = [int(v.item()) for v in counts_c]
            k_per_class[label_id] = k_c

        self._prototypes = MultiPrototypeResult(
            prototypes=prototypes,
            class_counts=class_counts,
            prototype_counts=prototype_counts,
            k_per_class=k_per_class,
            dim=self.dim,
        )
        self._prototype_config = config
        return self._prototypes

    def score_prototype_distance(
        self,
        *,
        prototypes: Optional[MultiPrototypeResult] = None,
    ) -> pd.DataFrame:
        """
        Score points by cosine distance to nearest true-class prototype.
        """
        prototype_result = prototypes
        if prototype_result is None:
            prototype_result = (
                self._prototypes if self._prototypes is not None else self.compute_prototypes()
            )
        # This check could be useful if user provides the prototyp_result.
        if int(prototype_result.dim) != self.dim:
            raise ValueError(
                f"Prototype dim mismatch: expected {self.dim}, got {int(prototype_result.dim)}."
            )

        emb = self._get_normalized_embeddings()
        lab = self.labels

        uniq = torch.unique(lab, sorted=True)
        # Another safty check
        missing_labels = [
            int(label.item())
            for label in uniq
            if int(label.item()) not in prototype_result.prototypes
        ]
        if missing_labels:
            raise ValueError(
                "Missing prototype(s) for label(s): "
                + ", ".join(str(label) for label in sorted(missing_labels))
            )

        num_samples = self.num_samples
        sim_to_prototype = torch.empty(num_samples, dtype=emb.dtype, device=emb.device)
        prototype_id = torch.empty(num_samples, dtype=torch.long, device=emb.device)
        num_prototypes_in_class = torch.empty(num_samples, dtype=torch.long, device=emb.device)
        prototype_size = torch.empty(num_samples, dtype=torch.long, device=emb.device)

        for label in uniq:
            label_id = int(label.item())
            class_idx = torch.nonzero(lab == label, as_tuple=False).flatten()
            x_c = emb[class_idx]

            class_prototypes = prototype_result.prototypes[label_id]
            if class_prototypes.ndim == 1:
                class_prototypes = class_prototypes.unsqueeze(0)
            class_prototypes = class_prototypes.to(device=emb.device, dtype=emb.dtype)

            if int(class_prototypes.shape[1]) != self.dim:
                raise ValueError(
                    f"Prototype dim mismatch in class {label_id}: "
                    f"expected {self.dim}, got {int(class_prototypes.shape[1])}."
                )
            if not torch.isfinite(class_prototypes).all():
                raise ValueError(
                    f"Non-finite prototype values found for class {label_id}."
                )

            sim_c = x_c @ class_prototypes.t()
            if not torch.isfinite(sim_c).all():
                raise ValueError(
                    f"Non-finite similarity values found for class {label_id}."
                )
            # Per-sample nearest prototype inside its true class.
            best_sim_c, best_idx_c = sim_c.max(dim=1)

            sim_to_prototype[class_idx] = best_sim_c
            prototype_id[class_idx] = best_idx_c
            num_prototypes_in_class[class_idx] = int(class_prototypes.shape[0])

            counts_c = prototype_result.prototype_counts[label_id]
            if len(counts_c) != int(class_prototypes.shape[0]):
                raise ValueError(
                    f"prototype_counts length mismatch for class {label_id}: "
                    f"{len(counts_c)} vs {int(class_prototypes.shape[0])}."
                )
            counts_c_tensor = torch.tensor(counts_c, dtype=torch.long, device=emb.device)
            prototype_size[class_idx] = counts_c_tensor[best_idx_c]

        if int(uniq.numel()) <= 1:
            # No other class exists; cross-class margin is undefined.
            sim_to_other_class_best = torch.full_like(sim_to_prototype, float("nan"))
            margin_to_other_class = torch.full_like(sim_to_prototype, float("nan"))
        else:
            proto_blocks: List[torch.Tensor] = []
            proto_owner_labels: List[int] = []
            for label_id in sorted(prototype_result.prototypes.keys()):
                block = prototype_result.prototypes[label_id]
                if block.ndim == 1:
                    block = block.unsqueeze(0)
                block = block.to(device=emb.device, dtype=emb.dtype)
                proto_blocks.append(block)
                proto_owner_labels.extend([int(label_id)] * int(block.shape[0]))

            all_prototypes = torch.cat(proto_blocks, dim=0)
            if not torch.isfinite(all_prototypes).all():
                raise ValueError("Non-finite values found in all-prototype matrix.")

            owner_tensor = torch.tensor(
                proto_owner_labels, dtype=torch.long, device=emb.device
            )
            sim_all = emb @ all_prototypes.t()
            if not torch.isfinite(sim_all).all():
                raise ValueError("Non-finite values found in global similarity matrix.")

            # Mask true-class prototypes, then keep best similarity from other classes.
            same_class_mask = owner_tensor.unsqueeze(0) == lab.unsqueeze(1)
            sim_all_other = sim_all.masked_fill(same_class_mask, float("-inf"))
            sim_to_other_class_best = sim_all_other.max(dim=1).values
            sim_to_other_class_best = sim_to_other_class_best.masked_fill(
                torch.isinf(sim_to_other_class_best), float("nan")
            )
            margin_to_other_class = sim_to_prototype - sim_to_other_class_best

        outlier_score = 1.0 - sim_to_prototype

        scores = self.metadata.copy().reset_index(drop=True)
        scores["ground_truth_num_label"] = lab.detach().cpu().numpy().astype(int)
        if "ground_truth_word_label" not in scores.columns:
            scores["ground_truth_word_label"] = ""
        if "ground_truth_L2_num_label" not in scores.columns:
            scores["ground_truth_L2_num_label"] = -1
        if "file_name" not in scores.columns:
            scores["file_name"] = ""

        scores["method"] = "multi_prototype"
        # Keep `sim_to_centroid` for backward-compatible downstream consumers.
        scores["sim_to_prototype"] = sim_to_prototype.detach().cpu().numpy()
        scores["sim_to_centroid"] = scores["sim_to_prototype"]
        scores["outlier_score"] = outlier_score.detach().cpu().numpy()
        scores["class_size"] = (
            scores["ground_truth_num_label"].map(prototype_result.class_counts).astype(int)
        )
        scores["prototype_id"] = prototype_id.detach().cpu().numpy().astype(int)
        scores["num_prototypes_in_class"] = (
            num_prototypes_in_class.detach().cpu().numpy().astype(int)
        )
        scores["prototype_size"] = prototype_size.detach().cpu().numpy().astype(int)

        scores["rank_in_class"] = (
            scores.groupby("ground_truth_num_label")["outlier_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        scores["pct_rank_in_class"] = scores["rank_in_class"] / scores["class_size"]

        sim_p05 = scores.groupby("ground_truth_num_label")["sim_to_prototype"].transform(
            lambda col: col.quantile(0.05)
        )
        # Same class-level outlier flag semantics as single-centroid scorer.
        scores["is_bottom_5pct"] = scores["sim_to_prototype"] <= sim_p05

        scores["rank_in_prototype"] = (
            scores.groupby(["ground_truth_num_label", "prototype_id"])["outlier_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        scores["pct_rank_in_prototype"] = (
            scores["rank_in_prototype"] / scores["prototype_size"]
        )

        scores["sim_to_other_class_best"] = (
            sim_to_other_class_best.detach().cpu().numpy()
        )
        scores["margin_to_other_class"] = margin_to_other_class.detach().cpu().numpy()

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
            "method",
            "sim_to_prototype",
            "prototype_id",
            "num_prototypes_in_class",
            "prototype_size",
            "rank_in_prototype",
            "pct_rank_in_prototype",
            "sim_to_other_class_best",
            "margin_to_other_class",
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

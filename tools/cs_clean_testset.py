"""
CLI skeleton for CS test-set cleaning using embedding-geometry methods.

Intended subcommands (to be filled in with real logic):
  - score: load cached embeddings and compute centroid-based scores
  - select: apply a fixed selection rule to scores
  - materialize: build a cleaned test folder with symlinks/copies

The real work will live in tools/outlier_cleaning.py; this file wires
argparse and dispatch only, leaving implementation TODOs.
"""

import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path when launched as `python tools/cs_clean_testset.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS test-set cleaning (skeleton)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # score
    p_score = sub.add_parser("score", help="Compute geometry-based scores for a cache dir")
    p_score.add_argument(
        "cache_dir",
        type=Path,
        help="Path to cached embeddings (feat_cache_vis/.../seedX)",
    )
    p_score.add_argument("output", type=Path, help="Output CSV path for scores")
    p_score.add_argument(
        "--method",
        choices=["single", "multi"],
        default="single",
        help="Scoring method: single-centroid or multi-prototype.",
    )
    p_score.add_argument(
        "--k-mode",
        choices=["heuristic", "fixed"],
        default="heuristic",
        help="Prototype count strategy for multi-prototype mode.",
    )
    p_score.add_argument(
        "--k-fixed",
        type=int,
        default=2,
        help="Fixed K per class when --k-mode fixed.",
    )
    p_score.add_argument(
        "--k-max",
        type=int,
        default=4,
        help="Upper bound K used in heuristic/fixed mode.",
    )
    p_score.add_argument(
        "--min-samples-per-proto",
        type=int,
        default=15,
        help="Safety cap to avoid too many prototypes for small classes.",
    )
    p_score.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for k-means in multi-prototype mode.",
    )
    p_score.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of k-means initializations in multi-prototype mode.",
    )
    p_score.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum k-means iterations in multi-prototype mode.",
    )

    # select
    p_sel = sub.add_parser("select", help="Select outliers given scores.csv and a rule")
    p_sel.add_argument("scores_csv", type=Path, help="CSV from the score step")
    p_sel.add_argument("rule_json", type=Path, help="JSON describing selection rule parameters")
    p_sel.add_argument("output", type=Path, help="Output CSV for removed candidates")

    # materialize
    p_mat = sub.add_parser("materialize", help="Create cleaned test folder via symlinks/copies")
    p_mat.add_argument("source_images", type=Path, help="Folder with original test images")
    p_mat.add_argument("index_csv", type=Path, help="Index CSV to copy alongside images")
    p_mat.add_argument("keep_list", type=Path, help="CSV of files to keep (or removed list to invert)")
    p_mat.add_argument("output_dir", type=Path, help="Destination folder for cleaned test set")
    p_mat.add_argument("--invert", action="store_true", help="Treat keep_list as removed list instead")
    p_mat.add_argument("--no-symlink", action="store_true", help="Copy files instead of symlinking")

    return parser.parse_args()


def _run_score(
    cache_dir: Path,
    output: Path,
    *,
    method: str,
    k_mode: str,
    k_fixed: int,
    k_max: int,
    min_samples_per_proto: int,
    random_state: int,
    n_init: int,
    max_iter: int,
):
    from tools.outlier_cleaning import (
        load_cache,
        MultiPrototypeScorer,
        resolve_cache_paths,
        SingleCentroidScorer,
    )

    cache_paths = resolve_cache_paths(cache_dir)
    embeddings, labels, metadata = load_cache(cache_paths)
    if method == "single":
        scorer = SingleCentroidScorer(embeddings, labels, metadata)
        centroids = scorer.compute_centroids()
        scores = scorer.score_centroid_distance()
        summary = {
            "method": "single",
            "num_classes_present": int(len(centroids.centroids)),
            "total_prototypes": int(len(centroids.centroids)),
        }
    elif method == "multi":
        scorer = MultiPrototypeScorer(embeddings, labels, metadata)
        prototypes = scorer.compute_prototypes(
            k_mode=k_mode,
            k_fixed=k_fixed,
            k_max=k_max,
            min_samples_per_proto=min_samples_per_proto,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        scores = scorer.score_prototype_distance(prototypes=prototypes)
        total_prototypes = int(sum(prototypes.k_per_class.values()))
        num_classes_present = int(len(prototypes.prototypes))
        summary = {
            "method": "multi",
            "num_classes_present": num_classes_present,
            "total_prototypes": total_prototypes,
            "avg_k_per_class": float(total_prototypes / max(1, num_classes_present)),
        }
    else:
        raise ValueError(f"Unknown method '{method}'.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output, index=False)

    print("\n==== Score Completed ====")
    print(
        {
            "cache_dir": str(cache_dir),
            "output_csv": str(output),
            "num_samples": int(scores.shape[0]),
            **summary,
        }
    )


def main(args: argparse.Namespace):
    if args.cmd == "score":
        _run_score(
            cache_dir=args.cache_dir,
            output=args.output,
            method=args.method,
            k_mode=args.k_mode,
            k_fixed=args.k_fixed,
            k_max=args.k_max,
            min_samples_per_proto=args.min_samples_per_proto,
            random_state=args.random_state,
            n_init=args.n_init,
            max_iter=args.max_iter,
        )
    elif args.cmd == "select":
        raise NotImplementedError("`select` is not implemented yet. Use `score` for now.")
    elif args.cmd == "materialize":
        raise NotImplementedError("`materialize` is not implemented yet. Use `score` for now.")
    else:
        raise ValueError(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main(parse_args())

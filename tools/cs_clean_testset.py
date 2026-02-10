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
    p_score = sub.add_parser("score", help="Compute centroid scores for a cache dir")
    p_score.add_argument("cache_dir", type=Path, help="Path to cached embeddings (feat_cache_vis/.../seedX)")
    p_score.add_argument("output", type=Path, help="Output CSV path for scores")

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


def _run_score(cache_dir: Path, output: Path):
    from tools.outlier_cleaning import (
        compute_centroids,
        load_cache,
        resolve_cache_paths,
        score_centroid_distance,
    )

    cache_paths = resolve_cache_paths(cache_dir)
    embeddings, labels, metadata = load_cache(cache_paths)
    centroids = compute_centroids(embeddings, labels)
    scores = score_centroid_distance(
        embeddings=embeddings,
        labels=labels,
        centroids=centroids,
        metadata=metadata,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output, index=False)

    print("\n==== Score Completed ====")
    print(
        {
            "cache_dir": str(cache_dir),
            "output_csv": str(output),
            "num_samples": int(scores.shape[0]),
            "num_classes_present": int(len(centroids.centroids)),
        }
    )


def main(args: argparse.Namespace):
    if args.cmd == "score":
        _run_score(cache_dir=args.cache_dir, output=args.output)
    elif args.cmd == "select":
        raise NotImplementedError("`select` is not implemented yet. Use `score` for now.")
    elif args.cmd == "materialize":
        raise NotImplementedError("`materialize` is not implemented yet. Use `score` for now.")
    else:
        raise ValueError(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main(parse_args())

#!/usr/bin/env python3
"""Execute the approved OOF-supervised ConfidNet and numeric25/28 RF study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multimodal import geo_helpfulness_confidnet_oof as study  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "extract-seed", "fit-seed", "fit-routers", "score", "validate", "build"))
    parser.add_argument("--output-dir", type=Path, default=study.OUTPUT)
    parser.add_argument("--seed", type=int, choices=study.SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        if args.command == "build":
            phases = [["declare"]]
            for seed in study.SEEDS:
                phases += [["extract-seed", "--seed", str(seed), "--device", args.device],
                           ["fit-seed", "--seed", str(seed)]]
            phases += [["fit-routers"], ["score"], ["validate"]]
            for phase in phases:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                            *phase, "--output-dir", str(args.output_dir)], cwd=ROOT)
                if completed.returncode:
                    return completed.returncode
            result = {"status": "complete", "path": str(args.output_dir)}
        elif args.command in ("extract-seed", "fit-seed"):
            if args.seed is None:
                parser.error(f"{args.command} requires --seed")
            kwargs = {"device": args.device} if args.command == "extract-seed" else {}
            result = getattr(study, args.command.replace("-", "_"))(args.seed, args.output_dir, **kwargs)
        else:
            result = getattr(study, args.command.replace("-", "_"))(args.output_dir)
    except study.StudyPausedError as error:
        print(json.dumps({"status": "paused_review_required", "reason": str(error)}), flush=True)
        return 3
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

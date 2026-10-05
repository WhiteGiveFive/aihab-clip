#!/usr/bin/env python3
"""Run the approved joint TCP router study in separate fitting/scoring stages."""

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

from multimodal import geo_helpfulness_joint_tcp as study  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "prepare", "prepare-seed", "fit", "fit-seed", "score", "validate", "build"))
    parser.add_argument("--output-dir", type=Path, default=study.OUTPUT)
    parser.add_argument("--seed", type=int, choices=study.SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command in ("build", "prepare", "fit"):
        phases = []
        if args.command in ("build", "prepare"):
            phases = [["declare"]] + [["prepare-seed", "--seed", str(seed), "--device", args.device] for seed in study.SEEDS]
        if args.command in ("build", "fit"):
            phases += [["fit-seed", "--seed", str(seed)] for seed in study.SEEDS]
        if args.command == "build":
            phases += [["score"], ["validate"]]
        for phase in phases:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *phase,
                                     "--output-dir", str(args.output_dir)], cwd=ROOT)
            if result.returncode:
                return result.returncode
        print(json.dumps({"status": "complete", "stage": args.command, "path": str(args.output_dir)}), flush=True)
        return 0
    try:
        if args.command == "declare":
            result = study.declare(args.output_dir)
            study.prepare_inputs(args.output_dir)
        elif args.command in ("prepare-seed", "fit-seed"):
            if args.seed is None:
                parser.error(f"{args.command} requires --seed")
            if args.command == "prepare-seed":
                result = study.prepare_seed(args.seed, args.output_dir, device=args.device)
            else:
                result = study.fit_seed(args.seed, args.output_dir)
        else:
            result = getattr(study, args.command)(args.output_dir)
    except study.StudyPausedError as error:
        print(json.dumps({"status": "paused", "reason": str(error)}), flush=True)
        return 3
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

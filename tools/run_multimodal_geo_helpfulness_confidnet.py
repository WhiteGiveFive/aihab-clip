#!/usr/bin/env python3
"""Run the predeclared same-training-set, frozen-expert ConfidNet study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multimodal import geo_helpfulness_confidnet as experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run-seed", "score", "validate", "build"))
    parser.add_argument("--output-dir", type=Path, default=experiment.OUTPUT)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command == "build":
        commands = [["declare"]]
        commands.extend([["run-seed", "--seed", str(seed), "--device", args.device] for seed in experiment.SEEDS])
        commands.extend([["score"], ["validate"]])
        for command in commands:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *command,
                            "--output-dir", str(args.output_dir)], cwd=PROJECT_ROOT, check=True)
        result = {"status": "complete", "path": str(args.output_dir)}
    elif args.command == "run-seed":
        if args.seed is None:
            parser.error("run-seed requires --seed")
        result = experiment.run_seed(args.seed, args.output_dir, device=args.device)
    else:
        result = getattr(experiment, args.command)(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

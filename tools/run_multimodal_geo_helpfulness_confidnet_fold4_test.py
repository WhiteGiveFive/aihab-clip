#!/usr/bin/env python3
"""Apply saved fold-four experts, confidence heads and routers to cleaned test."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multimodal import geo_helpfulness_confidnet_fold4_test as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "extract-seed", "predict-seed", "score", "validate", "validate-scores", "build"))
    parser.add_argument("--seed", type=int, choices=study.SEEDS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.command == "build":
        phases = [["declare"]]
        for seed in study.SEEDS:
            phases += [["extract-seed", "--seed", str(seed), "--device", args.device],
                       ["predict-seed", "--seed", str(seed)]]
        phases += [["score"], ["validate"]]
        for phase in phases:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *phase], cwd=ROOT)
            if result.returncode:
                return result.returncode
        result = {"status": "complete"}
    elif args.command in ("extract-seed", "predict-seed"):
        if args.seed is None:
            parser.error("A seed is required")
        kwargs = {"device": args.device} if args.command == "extract-seed" else {}
        result = getattr(study, args.command.replace("-", "_"))(args.seed, **kwargs)
    else:
        result = getattr(study, args.command.replace("-", "_"))()
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the approved fixed-expert fold-four ConfidNet experiment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multimodal import geo_helpfulness_confidnet_fold4 as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "extract-seed", "fit-seed", "fit-routers", "score", "validate", "build"))
    parser.add_argument("--seed", type=int, choices=study.SEEDS)
    parser.add_argument("--output-dir", type=Path, default=study.OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        if args.command == "build":
            phases = [["declare"]]
            for seed in study.SEEDS:
                phases += [["extract-seed", "--seed", str(seed), "--device", args.device], ["fit-seed", "--seed", str(seed)]]
            phases += [["fit-routers"], ["score"], ["validate"]]
            for phase in phases:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *phase,
                                         "--output-dir", str(args.output_dir)], cwd=ROOT)
                if result.returncode:
                    return result.returncode
            result = {"status": "complete"}
        elif args.command in ("extract-seed", "fit-seed"):
            if args.seed is None:
                parser.error("A seed is required")
            kwargs = {"device": args.device} if args.command == "extract-seed" else {}
            result = getattr(study, args.command.replace("-", "_"))(args.seed, args.output_dir, **kwargs)
        else:
            result = getattr(study, args.command)(args.output_dir) if args.command != "fit-routers" else study.fit_routers(args.output_dir)
    except study.StudyPausedError as error:
        print(json.dumps({"status": "paused", "reason": str(error)}), flush=True)
        return 3
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

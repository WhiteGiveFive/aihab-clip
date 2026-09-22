#!/usr/bin/env python3
"""OOF-select, seal, and evaluate the one relaxed margin-difference policy."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
BASE = PROJECT_ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
PATHS = {
    "output_dir": BASE / "relaxed_guardrails_margin_v1",
    "source_root": BASE / "relaxed_guardrails_rf_v1",
    "protocol_root": BASE / "protocol_v1",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "build",
            "validate",
            "select",
            "predict",
            "score",
            "validate-selection",
            "validate-predictions",
            "validate-scores",
        ),
    )
    for name, default in PATHS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=default)
    args = parser.parse_args(argv)
    for name in PATHS:
        setattr(args, name, getattr(args, name).resolve())
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.update(PYTHONMALLOC="malloc", CUDA_VISIBLE_DEVICES="")
    started = time.monotonic()
    try:
        if args.command in ("build", "validate"):
            phases = (
                ("select", "predict", "score")
                if args.command == "build"
                else ("validate-selection", "validate-predictions", "validate-scores")
            )
            results = []
            for phase in phases:
                command = [sys.executable, str(Path(__file__).resolve()), phase]
                for name in PATHS:
                    command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
                print(
                    json.dumps({"event": "phase_started", "phase": phase}),
                    file=sys.stderr,
                    flush=True,
                )
                process = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    check=False,
                )
                if process.returncode:
                    raise RuntimeError(
                        f"{phase} failed with exit code {process.returncode}"
                    )
                value = json.loads(process.stdout)
                expected = (
                    {"validated"}
                    if args.command == "validate"
                    else {"built", "reused_valid"}
                )
                if not isinstance(value, dict) or value.get("status") not in expected:
                    raise ValueError("unexpected child completion summary")
                # Reject non-standard JSON numbers before starting another phase.
                json.dumps(value, allow_nan=False)
                results.append(value)
                print(
                    json.dumps({"event": "phase_completed", "phase": phase}),
                    file=sys.stderr,
                    flush=True,
                )
            result = {
                "status": (
                    "validated"
                    if args.command == "validate"
                    else (
                        "reused_valid"
                        if all(v["status"] == "reused_valid" for v in results)
                        else "built"
                    )
                ),
                "phases": results,
            }
        else:
            from multimodal import geo_helpfulness_relaxed_margin as workflow

            if args.command in ("select", "validate-selection"):
                function = (
                    workflow.build_selection
                    if args.command == "select"
                    else workflow.validate_selection
                )
                result = function(args.output_dir, args.source_root, args.protocol_root)
            else:
                function = {
                    "predict": workflow.build_predictions,
                    "score": workflow.build_scores,
                    "validate-predictions": workflow.validate_predictions,
                    "validate-scores": workflow.validate_scores,
                }[args.command]
                result = function(args.output_dir)
        result.update(
            command=args.command,
            output_dir=str(args.output_dir),
            elapsed_seconds=time.monotonic() - started,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "command": args.command,
                    "error": str(error),
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

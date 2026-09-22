#!/usr/bin/env python3
"""Select relaxed RF thresholds on development, then apply and score on legacy test."""

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
    "legacy_root": BASE / "exploratory_legacy_capacity_v1",
    "capacity_root": BASE / "router_capacity_v1",
    "lr_history_root": BASE / "relaxed_guardrails_lr_v1",
    "output_dir": BASE / "relaxed_guardrails_rf_v1",
    "config": PROJECT_ROOT / "configs/multimodal_geo_relaxed_guardrails_rf_v1.json",
}


def _environment():
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = "1"
    env.update(PYTHONMALLOC="malloc", CUDA_VISIBLE_DEVICES="")
    return env


def _summary(value, phase):
    allowed = (
        {"validated"} if phase.startswith("validate-") else {"built", "reused_valid"}
    )
    if not isinstance(value, dict) or value.get("status") not in allowed:
        raise ValueError(f"{phase} returned an invalid completion summary")
    json.dumps(value, allow_nan=False)
    return value


def _phases(args):
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
        started = time.monotonic()
        print(
            json.dumps({"event": "phase_started", "phase": phase}),
            file=sys.stderr,
            flush=True,
        )
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(
                f"{phase} failed with exit code {process.returncode}; see child stderr"
            )
        result = _summary(json.loads(process.stdout), phase)
        result["phase_elapsed_seconds"] = time.monotonic() - started
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "phase_completed",
                    "phase": phase,
                    "elapsed_seconds": result["phase_elapsed_seconds"],
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    return {
        "status": (
            "validated"
            if args.command == "validate"
            else (
                "reused_valid"
                if all(r["status"] == "reused_valid" for r in results)
                else "built"
            )
        ),
        "phases": results,
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
    os.environ.update(_environment())
    started = time.monotonic()
    try:
        if args.command in ("build", "validate"):
            result = _phases(args)
        else:
            from multimodal import geo_helpfulness_relaxed_rf as workflow

            if args.command == "select":
                result = workflow.build_selection(
                    args.output_dir,
                    args.legacy_root,
                    args.capacity_root,
                    args.lr_history_root,
                    args.config,
                )
            else:
                function = {
                    "predict": workflow.build_predictions,
                    "score": workflow.build_scores,
                    "validate-selection": workflow.validate_selection,
                    "validate-predictions": workflow.validate_predictions,
                    "validate-scores": workflow.validate_scores,
                }[args.command]
                result = function(args.output_dir)
            _summary(result, args.command)
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
                    "output_dir": str(args.output_dir),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

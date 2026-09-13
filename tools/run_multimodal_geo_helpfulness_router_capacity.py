#!/usr/bin/env python3
"""Build or validate the development-only, nine-group router capacity round."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACT_PARENT = (
    PROJECT_ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
)
PATH_ARGUMENTS = ("protocol_dir", "output_dir", "config")


def _limited_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[name] = "1"
    environment["PYTHONMALLOC"] = "malloc"
    return environment


def _summary(value, phase: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{phase} must return a JSON object summary")
    if value.get("status") == "failed":
        raise RuntimeError(f"{phase} failed: {value.get('error', 'no error detail')}")
    json.dumps(value, allow_nan=False)
    return dict(value)


def _isolated_phases(args) -> dict:
    phases = (
        ("fit", "score")
        if args.command == "build"
        else ("validate-predictions", "validate-scores")
    )
    results = []
    environment = _limited_environment()
    for phase in phases:
        command = [sys.executable, str(Path(__file__).resolve()), phase]
        for name in PATH_ARGUMENTS:
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        started = time.monotonic()
        print(
            json.dumps({"event": "phase_started", "phase": phase}),
            file=sys.stderr,
            flush=True,
        )
        # Preserve child progress immediately; stdout remains one JSON result.
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            check=False,
        )
        if process.returncode:
            detail = process.stdout.strip()
            suffix = (
                f": {detail[:2000]}" if detail else "; see child stderr for details"
            )
            raise RuntimeError(
                f"{phase} failed with exit code {process.returncode}{suffix}"
            )
        try:
            result = _summary(json.loads(process.stdout), phase)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"{phase} returned an invalid JSON summary: {error}"
            ) from error
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
                if all(value.get("status") == "reused_valid" for value in results)
                else "built"
            )
        ),
        "phases": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "build",
            "validate",
            "fit",
            "score",
            "validate-predictions",
            "validate-scores",
        ),
    )
    parser.add_argument(
        "--protocol-dir", type=Path, default=ARTIFACT_PARENT / "protocol_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ARTIFACT_PARENT / "router_capacity_v1"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/multimodal_geo_router_capacity_v1.json",
    )
    args = parser.parse_args(argv)
    for name in PATH_ARGUMENTS:
        setattr(args, name, getattr(args, name).absolute())
    started = time.monotonic()
    try:
        if args.command in ("build", "validate"):
            result = _isolated_phases(args)
        elif args.command == "fit":
            from multimodal.geo_helpfulness_router_capacity import build_predictions

            result = build_predictions(args.protocol_dir, args.output_dir, args.config)
        elif args.command == "score":
            from multimodal.geo_helpfulness_router_capacity import build_scores

            result = build_scores(args.output_dir)
        elif args.command == "validate-predictions":
            from multimodal.geo_helpfulness_router_capacity import validate_predictions

            result = validate_predictions(args.output_dir)
        else:
            from multimodal.geo_helpfulness_router_capacity import validate_scores

            result = validate_scores(args.output_dir)
        result = _summary(result, args.command)
        result.update(
            {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "command": args.command,
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

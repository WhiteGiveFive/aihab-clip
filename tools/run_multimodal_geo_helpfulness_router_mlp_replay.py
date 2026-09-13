#!/usr/bin/env python3
"""Run the frozen MLP study with a versioned, apply-only curve-order repair."""

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

from tools.run_multimodal_geo_helpfulness_router_mlp import (
    ARTIFACT_PARENT,
    _limited_environment,
    _summary,
)

PATH_ARGUMENTS = ("protocol_dir", "capacity_root", "source_dir", "output_dir", "config")


def _phases(args):
    phases = (
        ("fit", "score")
        if args.command == "build"
        else ("validate-predictions", "validate-scores")
    )
    results = []
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
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_limited_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(
                f"{phase} failed with exit code {process.returncode}; "
                "see child stderr for details"
            )
        result = _summary(json.loads(process.stdout), phase)
        allowed = (
            {"built", "reused_valid"} if phase in ("fit", "score") else {"validated"}
        )
        if result.get("status") not in allowed:
            raise ValueError(f"{phase} returned an invalid completion status")
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


def _fit(args):
    from multimodal import geo_helpfulness_router_mlp as original

    if (args.source_dir / "predictions/manifest.json").exists():
        from multimodal.geo_helpfulness_router_mlp_replay import validate_predictions

        declaration = original.load_declaration(args.source_dir)
        if (
            declaration["protocol_root"] != str(args.protocol_dir)
            or declaration["capacity_root"] != str(args.capacity_root)
            or declaration["configuration"] != original.read_json(args.config)
        ):
            raise ValueError("MLP repeat build paths/configuration differ")
        result = validate_predictions(args.source_dir)
        result.update(status="reused_valid", new_fits=0)
        return result
    return original.build_predictions(
        args.protocol_dir, args.source_dir, args.config, args.capacity_root
    )


def main(argv=None):
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
        "--capacity-root", type=Path, default=ARTIFACT_PARENT / "router_capacity_v1"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=ARTIFACT_PARENT / "router_mlp_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ARTIFACT_PARENT / "router_mlp_v1_replay_v2"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/multimodal_geo_router_mlp_v1.json",
    )
    args = parser.parse_args(argv)
    for name in PATH_ARGUMENTS:
        setattr(args, name, getattr(args, name).resolve())
    os.environ.update(_limited_environment())
    started = time.monotonic()
    try:
        if args.command in ("build", "validate"):
            result = _phases(args)
        elif args.command == "fit":
            result = _fit(args)
        else:
            from multimodal import geo_helpfulness_router_mlp_replay as replay

            if args.command == "validate-predictions":
                result = replay.validate_predictions(args.source_dir)
            elif args.command == "score":
                result = replay.build_scores(args.source_dir, args.output_dir)
            else:
                result = replay.validate_scores(args.source_dir, args.output_dir)
        result = _summary(result, args.command)
        result.update(
            command=args.command,
            source_dir=str(args.source_dir),
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
                    "error": str(error),
                    "command": args.command,
                    "source_dir": str(args.source_dir),
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

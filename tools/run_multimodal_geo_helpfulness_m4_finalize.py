#!/usr/bin/env python3
"""Reproduce full M4 with the additive receipt-schema lookup correction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "score", "validate"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    root = args.artifact_root.resolve()
    try:
        if args.command == "build":
            commands = (
                (
                    PROJECT_ROOT / "tools/run_multimodal_geo_helpfulness_m4.py",
                    "prepare",
                ),
                (Path(__file__).resolve(), "score"),
            )
            for script, command in commands:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        command,
                        "--artifact-root",
                        str(root),
                    ],
                    check=False,
                )
                if result.returncode:
                    return result.returncode
            return 0
        from multimodal.geo_helpfulness_router_m4_finalize import (
            build_m4_result,
            validate_m4_result,
        )

        operation = build_m4_result if args.command == "score" else validate_m4_result
        print(json.dumps(operation(root), sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "command": args.command, "error": str(error)},
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

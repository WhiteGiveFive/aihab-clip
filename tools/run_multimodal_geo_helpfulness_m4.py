#!/usr/bin/env python3
"""Complete M4 with isolated preparation/scoring processes and immutable results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "prepare", "score", "validate"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    root = args.artifact_root.resolve()
    try:
        if args.command == "build":
            # A fit-capable input adapter and a label reader never share a process.
            for command in ("prepare", "score"):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        command,
                        "--artifact-root",
                        str(root),
                    ],
                    check=False,
                )
                if completed.returncode:
                    return completed.returncode
            return 0
        if args.command == "prepare":
            from multimodal.geo_helpfulness_router_m4_inputs import prepare_comparators

            result = prepare_comparators(root)
        else:
            from multimodal.geo_helpfulness_router_m4_analysis import (
                build_m4_result,
                validate_m4_result,
            )

            operation = (
                build_m4_result if args.command == "score" else validate_m4_result
            )
            result = operation(root)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
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

#!/usr/bin/env python3
"""Prepare sealed label-blind predictions or score the single fixed preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "validate-predictions", "preview")
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    root = args.artifact_root.resolve()
    try:
        if args.command == "preview":
            # Do not import the fit-capable prediction-preparation dependency graph.
            from multimodal.geo_helpfulness_router_preview_score import (
                score_fixed_preview,
            )

            result = score_fixed_preview(
                root / "router/router_validation_v1",
                root / "protocol/development_assignments.parquet",
                output_path=root / "router/router_validation_preview_v1.json",
            )
        else:
            from multimodal.geo_helpfulness_router_validation import (
                prepare_validation_bundle,
            )

            result = prepare_validation_bundle(
                repo_root=PROJECT_ROOT,
                protocol_root=root,
                validate_only=args.command == "validate-predictions",
            )
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

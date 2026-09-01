#!/usr/bin/env python3
"""Run and aggregate the frozen M2 geo-helpfulness expert workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


# This must be configured before importing the M2 implementation (and Torch).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal.geo_helpfulness_oof import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    aggregate,
    run_seed,
)


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Frozen M1 protocol configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=None,
        help="Sealed M1 protocol directory (defaults to the configured directory)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="M2 artifact root (defaults to the configured artifact root)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate immutable, honest out-of-fold expert outputs under the "
            "frozen geo-helpfulness protocol."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run-seed",
        help="Run all four OOF producers and the validation producer for one seed",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        choices=(1, 2, 3, 4),
        required=True,
        help="Frozen training seed",
    )
    _add_paths(run_parser)

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Validate and concatenate complete outputs from all four seeds",
    )
    _add_paths(aggregate_parser)
    return parser


def _print_result(result: Any) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "config_path": args.config,
        "protocol_dir": args.protocol_dir,
        "output_root": args.output_root,
    }
    try:
        if args.command == "run-seed":
            result = run_seed(args.seed, **common)
        else:
            result = aggregate(**common)
    except Exception as exc:
        print(f"M2 {args.command} failed: {exc}", file=sys.stderr)
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

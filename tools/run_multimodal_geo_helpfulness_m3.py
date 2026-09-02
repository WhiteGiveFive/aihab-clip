#!/usr/bin/env python3
"""Build or validate M3 geo-helpfulness targets and feature contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal.geo_helpfulness_targets_features import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    build_m3_bundle,
    validate_m3_bundle,
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
        "--artifact-root",
        type=Path,
        default=None,
        help="M2/M3 artifact root (defaults to the protocol artifact root)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive immutable seed-specific M3 targets and freeze the "
            "deployment-safe router feature contract."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser(
        "build", help="Build or strictly reuse the immutable M3 bundle"
    )
    _add_paths(build_parser)
    validate_parser = subparsers.add_parser(
        "validate", help="Revalidate M1/M2 lineage and the committed M3 bundle"
    )
    _add_paths(validate_parser)
    return parser


def _print_result(result: Any) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "config_path": args.config,
        "protocol_dir": args.protocol_dir,
        "artifact_root": args.artifact_root,
    }
    try:
        result = (
            build_m3_bundle(**common)
            if args.command == "build"
            else validate_m3_bundle(**common)
        )
    except Exception as exc:
        print(f"M3 {args.command} failed: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

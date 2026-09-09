#!/usr/bin/env python3
"""Build, validate, or explicitly reproduce the M4 router candidates on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal.geo_helpfulness_router_candidates import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    build_router_candidate_bundle,
    reproduce_router_candidate_bundle,
    validate_router_candidate_bundle,
)


DEFAULT_STUDY_CONFIG_PATH = (
    PROJECT_ROOT / "configs/multimodal_geo_router_candidates_v2.json"
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
        help="M2/M3/M4 artifact root (defaults to the protocol artifact root)",
    )
    parser.add_argument(
        "--study-config",
        type=Path,
        default=DEFAULT_STUDY_CONFIG_PATH,
        help="Predeclared primary/ablation study configuration (default: %(default)s)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and seal 48 CPU router candidates from the immutable OOF dataset; "
            "validation scoring and policy selection are separate later stages."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        (
            "build",
            "Build or resume candidate fitting; reuse valid committed candidates",
        ),
        ("validate", "Check sealed candidates and their parents without fitting"),
        (
            "reproduce",
            "Explicitly refit all 48 candidates and compare without overwriting",
        ),
    ):
        child = subparsers.add_parser(command, help=description)
        _add_paths(child)
    return parser


def _print_result(result: Any, *, stream: Any = None) -> None:
    print(
        json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False),
        file=stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    operations = {
        "build": build_router_candidate_bundle,
        "validate": validate_router_candidate_bundle,
        "reproduce": reproduce_router_candidate_bundle,
    }
    try:
        result = operations[args.command](
            config_path=args.config,
            protocol_dir=args.protocol_dir,
            artifact_root=args.artifact_root,
            study_config_path=args.study_config,
        )
        _print_result(result)
    except Exception as exc:
        _print_result(
            {"status": "failed", "command": args.command, "error": str(exc)},
            stream=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

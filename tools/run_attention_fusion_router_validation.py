#!/usr/bin/env python3
"""Apply and verify the declared R3 validation routing matrix on saved R2 scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal.attention_fusion_router_validation import (  # noqa: E402
    DEFAULT_CONFIG, run, verify,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = {"run": run, "verify": verify}[args.command](args.config)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

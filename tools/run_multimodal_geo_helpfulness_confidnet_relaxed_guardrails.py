#!/usr/bin/env python3
"""Canonical entrypoint for the frozen ConfidNet guardrail study."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal.geo_helpfulness_compat import confidnet_relaxed_main


if __name__ == "__main__":
    raise SystemExit(confidnet_relaxed_main())

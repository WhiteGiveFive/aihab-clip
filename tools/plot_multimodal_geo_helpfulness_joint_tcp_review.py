#!/usr/bin/env python3
"""Canonical entrypoint for joint-TCP report figures."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal.geo_helpfulness_compat import joint_tcp_review_main


if __name__ == "__main__":
    raise SystemExit(joint_tcp_review_main())

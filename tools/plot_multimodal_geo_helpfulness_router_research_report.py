#!/usr/bin/env python3
"""Build the concise router comparison and oracle-gap research report."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal.geo_helpfulness_router_research_report import main


if __name__ == "__main__":
    raise SystemExit(main())

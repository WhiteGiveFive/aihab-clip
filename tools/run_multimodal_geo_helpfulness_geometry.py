#!/usr/bin/env python3
"""Run the geo-helpfulness geometry studies through their canonical interface."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal.geo_helpfulness_geometry import main


if __name__ == "__main__":
    raise SystemExit(main())

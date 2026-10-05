#!/usr/bin/env python3
"""Check geo-helpfulness study naming and coverage without running experiments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal.geo_helpfulness_study_catalog import main


if __name__ == "__main__":
    raise SystemExit(main())

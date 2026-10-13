#!/usr/bin/env python3
"""Compare saved fusion, RF, and adapted-geometry confusion matrices."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal.geo_helpfulness_confusion_review import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Visualize cached adapted image embeddings across development and test."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal.geo_helpfulness_adapted_embeddings import main


if __name__ == "__main__":
    raise SystemExit(main())

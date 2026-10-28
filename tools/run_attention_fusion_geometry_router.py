#!/usr/bin/env python3
"""Prepare and verify R1/R2; routing application belongs to a later stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal.attention_fusion_geometry_router import (  # noqa: E402
    DEFAULT_CONFIG, load_config, prepare, score, verify,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "extract", "score", "verify"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--encoder", action="append", choices=("pretrained", "joint_final", "joint_best"))
    args = parser.parse_args(argv)
    if args.encoder and args.command != "extract":
        parser.error("--encoder is only applicable to extract")
    if args.command == "extract":
        from multimodal.attention_fusion_geometry_embeddings import extract_embeddings
        _, output = load_config(args.config)
        extracted = extract_embeddings(output, encoder_ids=args.encoder, device="cuda")
        result = {"status": "embeddings_complete", "output_root": extracted["output_root"],
                  "encoders": {name: {"roles": manifest["roles"],
                                       "elapsed_seconds": manifest["elapsed_seconds"]}
                               for name, manifest in extracted["encoders"].items()}}
    else:
        result = {"prepare": prepare, "score": score, "verify": verify}[args.command](args.config)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

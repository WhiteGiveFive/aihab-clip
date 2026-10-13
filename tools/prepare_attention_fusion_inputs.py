#!/usr/bin/env python3
"""Extract frozen development-train embeddings and verify validation replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multimodal.attention_fusion_inputs import prepare_inputs, prepare_verification_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multimodal_attention_fusion.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-rows", type=int, default=16)
    parser.add_argument("--verification-only", action="store_true")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    if args.verification_only:
        if args.seeds is not None and len(args.seeds) != 1:
            parser.error("Verification preparation accepts exactly one seed")
        result = prepare_verification_inputs(
            args.config, (args.seeds or [1])[0], limit=args.limit, device=args.device,
            batch_size=args.batch_size, replay_rows=args.replay_rows, num_threads=args.num_threads)
        print(json.dumps({"verification_only": True, "seed": result["seed"],
                          "training_rows": result["training_rows"],
                          "timing_seconds": result["timing_seconds"],
                          "validation_replay": result["validation_replay"]}, indent=2))
        return
    result = prepare_inputs(args.config, args.seeds, device=args.device,
                            batch_size=args.batch_size, replay_rows=args.replay_rows,
                            num_threads=args.num_threads)
    print(json.dumps({"study_id": result["study_id"],
                      "completed_seeds": [item["seed"] for item in result["seeds"]],
                      "raw_images_decoded_and_hashed": result["raw_images_decoded_and_hashed"],
                      "test_features_read": False, "test_labels_read": False}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate both legacy checkpoint families with immutable M4 router policies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
BASE = "hf-hub_timm_ViT-SO400M-16-SigLIP2-384_5_20260203_17"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "build",
            "predict",
            "score",
            "validate",
            "validate-predictions",
            "validate-scores",
        ),
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=PROJECT_ROOT
        / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1",
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=PROJECT_ROOT / "multimodal_artifacts/runs/cs" / BASE,
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT
        / "multimodal_artifacts/joined_tables/cs"
        / BASE
        / "gse_100m_cleaned_test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/exploratory_legacy_router_v1",
    )
    args = parser.parse_args(argv)
    for name in ("protocol_dir", "legacy_root", "cache_root", "output_dir"):
        setattr(args, name, getattr(args, name).absolute())
    started = time.monotonic()
    try:
        if args.command in ("build", "validate"):
            phases = (
                ("predict", "score")
                if args.command == "build"
                else ("validate-predictions", "validate-scores")
            )
            results = []
            env = os.environ.copy()
            env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONMALLOC": "malloc",
                }
            )
            for phase in phases:
                command = [sys.executable, str(Path(__file__).resolve()), phase]
                for name in ("protocol_dir", "legacy_root", "cache_root", "output_dir"):
                    command.extend(
                        ["--" + name.replace("_", "-"), str(getattr(args, name))]
                    )
                process = subprocess.run(
                    command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True
                )
                if process.returncode:
                    raise RuntimeError(
                        f"{phase} failed: {process.stdout}\n{process.stderr}"
                    )
                results.append(json.loads(process.stdout))
            result = {
                "status": (
                    "validated"
                    if args.command == "validate"
                    else (
                        "reused_valid"
                        if all(r.get("status") == "reused_valid" for r in results)
                        else "built"
                    )
                ),
                "phases": results,
            }
        elif args.command in ("predict", "validate-predictions"):
            from multimodal.geo_helpfulness_legacy_router_common import (
                load_declaration,
                publish_predictions,
                reconstruct_predictions,
                validate_predictions,
                compare_tables,
            )
            from multimodal.geo_helpfulness_legacy_router_inference import (
                create_declaration,
                infer_expert_logits,
            )

            if args.command == "predict":
                declaration = create_declaration(
                    args.protocol_dir,
                    args.legacy_root,
                    args.cache_root,
                    args.output_dir,
                )
            else:
                declaration = load_declaration(args.output_dir)
            logits = infer_expert_logits(declaration)
            tables = reconstruct_predictions(logits, declaration)
            if args.command == "predict":
                result = publish_predictions(args.output_dir, declaration, tables)
            else:
                import pyarrow.parquet as pq

                _, manifest, _ = validate_predictions(args.output_dir)
                for name, table in tables.items():
                    compare_tables(
                        pq.read_table(
                            args.output_dir / "predictions" / f"{name}.parquet"
                        ),
                        table,
                    )
                result = {
                    "status": "validated",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "original_head_inference_reproduced": True,
                }
        else:
            from multimodal.geo_helpfulness_legacy_router_score import (
                build_score,
                validate_score,
            )

            result = (
                build_score(args.output_dir)
                if args.command == "score"
                else validate_score(args.output_dir)
            )
        result.update(
            {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error": str(error), "command": args.command},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

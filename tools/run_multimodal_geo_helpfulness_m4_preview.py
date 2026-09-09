#!/usr/bin/env python3
"""Declare or run the dtype-compatible, fixed-candidate M4 preview scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1"
)
CODE_FILES = (
    "multimodal/geo_helpfulness_router_preview_score_v2.py",
    "tools/run_multimodal_geo_helpfulness_m4_preview.py",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "score"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    root = args.artifact_root.resolve()
    bundle = root / "router/router_validation_v1"
    declaration_path = root / "router/router_validation_preview_v2_declaration.json"
    try:
        if args.command == "declare":
            from multimodal import geo_helpfulness_router_preview_score as original

            summary = original.validate_sealed_predictions(bundle)
            declaration = {
                "schema_version": "geo_helpfulness.router_preview_correction_declaration.v2",
                "protocol_id": "protocol_v1",
                "prediction_manifest_sha256": summary["manifest_sha256"],
                "candidate_variant": "primary_full727",
                "training_seed": 1,
                "C_hex": float(0.01).hex(),
                "metric": "four_state_accuracy_only",
                "selection_performed": False,
                "correction_reason": "pandas_string_storage_dtype_identity_equality",
                "code_file_sha256": {
                    name: original._file_hash(PROJECT_ROOT / name)
                    for name in CODE_FILES
                },
                "previous_attempt": {
                    "computed_accuracy": False,
                    "error": "sealed predictions and validation assignments do not align",
                    "exact_identity_values_verified": True,
                    "assignment_string_storage": "pandas_string_extension",
                    "prediction_string_storage": "pandas_object",
                },
            }
            declaration["declaration_sha256"] = original._hash(declaration)
            original._save_result(declaration_path, declaration)
            result = {
                "status": "declared_or_reused_valid",
                "declaration_path": str(declaration_path),
                "declaration_sha256": declaration["declaration_sha256"],
                "assignment_content_opened": False,
            }
        else:
            from multimodal.geo_helpfulness_router_preview_score_v2 import (
                score_fixed_preview_v2,
            )

            result = score_fixed_preview_v2(
                bundle,
                root / "protocol/development_assignments.parquet",
                correction_declaration_path=declaration_path,
                output_path=root / "router/router_validation_preview_v2.json",
            )
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "command": args.command, "error": str(error)},
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

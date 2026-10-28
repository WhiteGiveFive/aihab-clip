#!/usr/bin/env python3
"""Declare, analyze and render the separate two-primary CLS-bypass extension."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multimodal.attention_fusion_bypass_protocol import (  # noqa: E402
    DEFAULT_CONFIG, build_matrix, load_config, read_frozen_plan,
)
from tools.analyze_attention_fusion_study import authenticated_analysis  # noqa: E402
from tools.prepare_attention_fusion_protocol import content_hash, file_hash  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("command", choices=("declare", "analyze", "report"))
    parser.add_argument("--analysis-dir", type=Path,
                        help="New analysis destination, or authenticated existing analysis for reporting")
    args = parser.parse_args(argv)
    if args.command == "declare":
        config, _, _, _ = load_config(args.config, REPO_ROOT)
        matrix = build_matrix(config)
    else:
        config, matrix, _ = read_frozen_plan(args.config, REPO_ROOT)
    root = (REPO_ROOT / config["paths"]["artifact_root"]).resolve()
    declaration = root / "protocol/contrasts.json"
    output = (args.analysis_dir or root / "reports/m3_bypass_comparison_v1").resolve()
    if not output.is_relative_to(root):
        raise ValueError("Bypass analysis must remain inside its separate study directory")
    if args.command == "declare":
        from multimodal.attention_fusion_bypass_comparison import freeze_contrasts
        saved = freeze_contrasts(config, matrix, declaration, repo_root=REPO_ROOT)
        result = {"declaration": str(declaration), "contrasts": len(saved["declaration"]["contrasts"]),
                  "primary_contrasts": saved["declaration"]["primary_contrast_ids"], "sha256": file_hash(declaration)}
    elif args.command == "analyze":
        from multimodal.attention_fusion_bypass_comparison import analyze_study
        analysis = analyze_study(config, matrix, declaration, output, repo_root=REPO_ROOT)
        result = {"analysis": str(output), "counts": analysis["counts"],
                  "primary_contrasts": analysis["primary_contrast_ids"],
                  "manifest_sha256": file_hash(output / "manifest.json")}
    else:
        from multimodal.attention_fusion_bypass_reporting import generate_report
        analysis = authenticated_analysis(output)
        if (analysis["study_id"] != config["study_id"] or
                analysis["provenance"]["config_sha256"] != content_hash(config) or
                analysis["provenance"]["matrix_sha256"] != content_hash(matrix)):
            raise ValueError("Analysis belongs to a different frozen bypass configuration")
        report = generate_report(analysis, output / "report")
        result = {"report": str(output / "report/report.md"),
                  "manifest_sha256": file_hash(output / "report/report_manifest.json"),
                  "render_status": report["render_status"]}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

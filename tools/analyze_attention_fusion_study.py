#!/usr/bin/env python3
"""Declare, analyze and render the frozen M3 development comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.run_multimodal_attention_fusion import read_frozen_plan  # noqa: E402
from tools.prepare_attention_fusion_protocol import DEFAULT_CONFIG, content_hash, file_hash  # noqa: E402


def authenticated_analysis(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    payload = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    if manifest.get("status") != "complete" or content_hash(payload) != manifest.get("manifest_payload_sha256"):
        raise ValueError("Analysis manifest is incomplete or has changed")
    if "analysis.json" not in manifest["files"]:
        raise ValueError("Analysis manifest lacks its analysis payload")
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Analysis payload is missing or has changed: {name}")
    return json.loads((directory / "analysis.json").read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("command", choices=("declare", "analyze", "report"))
    parser.add_argument("--analysis-dir", type=Path,
                        help="New analysis destination, or existing authenticated input for report")
    args = parser.parse_args(argv)
    config, matrix, _ = read_frozen_plan(args.config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    declaration = root / "reports/m3_contrast_declaration.json"
    output = (args.analysis_dir or root / "reports/m3_comparison_v1").resolve()
    if not output.is_relative_to(root.resolve()):
        raise ValueError("M3 analysis must remain inside the study artifact directory")
    if args.command == "declare":
        from multimodal.attention_fusion_comparison import freeze_contrasts
        saved = freeze_contrasts(config, matrix, declaration)
        result = {"declaration": str(declaration),
                  "contrasts": len(saved["declaration"]["contrasts"]),
                  "sha256": file_hash(declaration)}
    elif args.command == "analyze":
        from multimodal.attention_fusion_comparison import analyze_study
        analysis = analyze_study(config, matrix, declaration, output)
        result = {"analysis": str(output), "counts": analysis["counts"],
                  "primary_contrast": analysis["primary_contrast_id"]}
    else:
        from multimodal.attention_fusion_reporting import generate_report
        analysis = authenticated_analysis(output)
        if analysis["provenance"]["config_sha256"] != content_hash(config):
            raise ValueError("Analysis belongs to a different frozen configuration")
        report = generate_report(analysis, output / "report")
        result = {"report": str(output / "report/report.md"),
                  "manifest_sha256": file_hash(output / "report/report_manifest.json"),
                  "render_status": report["render_status"]}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

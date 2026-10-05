"""Consistent public command names for fingerprinted historical helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from multimodal.geo_helpfulness_geometry import Once, artifact_destination, repo_path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"


def launch(script: str, arguments: list[str], dry_run: bool) -> int:
    command = [sys.executable, str(ROOT / "tools" / script), *arguments]
    if dry_run:
        print(json.dumps({"cwd": str(ROOT), "argv": command, "executed": False}, indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


def confidnet_relaxed_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate the frozen ConfidNet relaxed-guardrail study.", allow_abbrev=False)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output-dir", action=Once)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = artifact_destination(args.output_dir or f"{ARTIFACTS}/confidnet_oof_relaxed_guardrails_v1",
                                      {"artifact_root": ARTIFACTS}, ROOT)
    except ValueError as error:
        parser.error(str(error))
    return launch("analyze_confidnet_relaxed_guardrails.py", [args.command, "--output-dir", str(output)], args.dry_run)


def joint_tcp_review_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the saved joint-TCP report into a separate review directory.", allow_abbrev=False)
    parser.add_argument("--report", required=True, action=Once)
    parser.add_argument("--output-dir", required=True, action=Once)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    # Historical review directories end in _review; preserve that established convention.
    output = repo_path(args.output_dir, ROOT)
    base = ROOT / ARTIFACTS
    if not output.is_relative_to(base) or output == base:
        parser.error(f"Review outputs must be below {base}")
    return launch("plot_joint_tcp_router_review.py", ["--report", str(repo_path(args.report, ROOT)),
                                                     "--output", str(output)], args.dry_run)


def confidnet_parity_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Historical ConfidNet native-runtime diagnostic. The run command may fit a recovery head; it is not routine study validation.",
        allow_abbrev=False,
    )
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    # The historical script has no parser: never forward --help or unknown flags.
    return launch("diagnose_multimodal_geo_confidnet_oof_parity.py", [], args.dry_run)

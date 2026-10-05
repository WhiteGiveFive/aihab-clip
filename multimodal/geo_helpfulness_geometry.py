"""Canonical commands for the frozen geo-helpfulness geometry studies.

The scientific backends retain their recorded paths and bytes. This module
only supplies consistent command names, artifact destinations and dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path("configs/multimodal_geo_geometry_layout_v1.json")
COMMANDS = {
    "overview": "embedding_geometry_v1",
    "disagreements": "geometry_disagreement_v1",
    "validation": "joint_geometry_validation_v1",
    "cleaned-test": "joint_geometry_cleaned_test_v1",
    "full-reference": "joint_geometry_full_reference_cleaned_test_v1",
    "k5": "joint_geometry_k5_correction_v1",
    "k-sweep": "joint_geometry_shared_k_sweep_v1",
    "thresholds": "joint_geometry_validation_thresholds_v1",
    "adapted": "joint_geometry_adapted_encoder_v1",
}
DESCRIPTIONS = {
    "overview": "Plot development embeddings (completed outputs cannot be overwritten).",
    "disagreements": "Analyze archived disagreement geometry (requires a new output directory).",
    "validation": "Evaluate the fixed K1 validation rule (requires a new output directory).",
    "cleaned-test": "Evaluate or verify the completed K1 cleaned-test study.",
    "full-reference": "Evaluate or verify the completed full-reference K1 study.",
    "k5": "Evaluate or verify the completed strict K5 study.",
    "k-sweep": "Evaluate or verify the completed shared-neighborhood study.",
    "thresholds": "Evaluate or verify validation-selected geometry thresholds.",
    "adapted": "Evaluate or verify the adapted-encoder comparison.",
}


class Once(argparse.Action):
    """Reject ambiguous repeated path flags instead of silently taking the last."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, values)


def repo_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def artifact_destination(value: str | Path, catalog: dict, root: Path) -> Path:
    path = repo_path(value, root)
    base = repo_path(catalog["artifact_root"], root)
    if not path.is_relative_to(base):
        raise ValueError(f"Outputs must be under {base}; received {path}")
    parts = path.relative_to(base).parts
    if not parts or re.search(r"_v[0-9]+$", parts[0]) is None:
        raise ValueError("Use a versioned study directory, for example geometry_example_v2")
    return path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subcommands = result.add_subparsers(dest="command", required=True)
    subcommands.add_parser("list", help="List canonical study commands and destinations.", allow_abbrev=False)
    for command, description in DESCRIPTIONS.items():
        sub = subcommands.add_parser(command, help=description, description=description, allow_abbrev=False)
        sub.add_argument("--output-dir", action=Once, help="Versioned artifact directory; defaults to the recorded study.")
        sub.add_argument("--dry-run", action="store_true", help="Show the invocation without launching the backend.")
        if command == "overview":
            sub.add_argument("--development-split", choices=("train", "all"), default="all")
        elif command == "adapted":
            sub.add_argument("--embedding-dir", action=Once)
    sub = subcommands.add_parser("extract-adapted", help="Extract or verify adapted reference/query embeddings.", allow_abbrev=False)
    sub.add_argument("--output-dir", action=Once, help="Embedding destination under a versioned study directory.")
    sub.add_argument("--device", default="cuda")
    sub.add_argument("--preflight-only", action="store_true")
    sub.add_argument("--dry-run", action="store_true")
    sub = subcommands.add_parser("validate", help="Read-only verification of the geometry layout and saved provenance.", allow_abbrev=False)
    sub.add_argument("--strict-registry-snapshot", action="store_true")
    sub.add_argument("--dry-run", action="store_true")
    return result


def invocation(args: argparse.Namespace, *, root: Path = ROOT) -> list[str]:
    """Build an argument list without importing any model or scoring backend."""
    root = root.resolve()
    catalog = json.loads((root / CATALOG).read_text())
    if args.command == "validate":
        command = [sys.executable, str(root / "tools/check_multimodal_geo_helpfulness_geometry_layout.py")]
        if args.strict_registry_snapshot:
            command.append("--strict-registry-snapshot")
        return command
    if args.command == "extract-adapted":
        output = artifact_destination(args.output_dir or catalog["embedding_root"], catalog, root)
        command = [sys.executable, str(root / "tools/extract_adapted_geometry_embeddings.py"),
                   "--output", str(output), "--device", args.device]
        if args.preflight_only:
            command.append("--preflight-only")
        return command
    study_id = COMMANDS[args.command]
    study = next(entry for entry in catalog["studies"] if entry["study_id"] == study_id)
    output = artifact_destination(args.output_dir or study["artifact_root"], catalog, root)
    command = [sys.executable, str(root / study["entrypoint"]), "--output-dir", str(output)]
    if args.command == "overview":
        command += ["--development-split", args.development_split]
    elif args.command == "adapted":
        command += ["--embedding-dir", str(repo_path(args.embedding_dir or catalog["embedding_root"], root))]
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "list":
        catalog = json.loads((ROOT / CATALOG).read_text())
        reverse = {value: key for key, value in COMMANDS.items()}
        studies = [{"command": reverse[s["study_id"]], "study_id": s["study_id"],
                    "output_dir": s["artifact_root"], "recorded_backend": s["entrypoint"]}
                   for s in catalog["studies"]]
        print(json.dumps({"studies": studies, "embedding_command": "extract-adapted",
                          "embedding_root": catalog["embedding_root"], "verification_command": "validate"}, indent=2))
        return 0
    try:
        command = invocation(args)
    except (ValueError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({"cwd": str(ROOT), "argv": command, "executed": False}, indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT).returncode

"""Verify geometry artifact relocation without changing files or running models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = "configs/multimodal_geo_geometry_layout_v1.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
INLINE_LINK = re.compile(r"\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+[^)\n]*)?\)")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", re.MULTILINE)


class VerificationError(ValueError):
    """A saved relocation invariant no longer holds."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def repo_path(value: str, root: Path, *, allow_external: bool = False) -> Path:
    require(isinstance(value, str) and bool(value), "Paths must be nonempty strings")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    # Preserve the literal path for later filesystem resolution through aliases.
    if not allow_external:
        lexical = Path(os.path.abspath(path))
        require(lexical.is_relative_to(root), f"Path escapes repository: {value}")
    return path


class FileDigests:
    """Hash each unchanged physical file once, including aliases to that file."""

    def __init__(self):
        self.cache = {}

    def digest(self, path: Path) -> str:
        require(not path.is_symlink(), f"Expected a regular file, not a file symlink: {path}")
        physical = path.resolve(strict=True)
        before = physical.stat()
        require(stat.S_ISREG(before.st_mode), f"Expected a regular file: {path}")
        key = (str(physical), before.st_dev, before.st_ino, before.st_size,
               before.st_mtime_ns, before.st_ctime_ns)
        if key not in self.cache:
            digest = hashlib.sha256()
            with physical.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            after = physical.stat()
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    f"File changed during verification: {path}")
            self.cache[key] = digest.hexdigest()
        return self.cache[key]

    def verify(self, path: Path, expected: str) -> None:
        require(isinstance(expected, str) and SHA256.fullmatch(expected) is not None,
                f"Invalid SHA256 declaration for {path}")
        require(self.digest(path) == expected, f"SHA256 mismatch: {path}")


def verify_directory_alias(alias: Path, target: Path) -> None:
    require(alias.is_symlink(), f"Expected directory symlink: {alias}")
    require(target.is_dir() and not target.is_symlink(), f"Expected real target directory: {target}")
    require(alias.is_dir(), f"Directory alias is missing or dangling: {alias}")
    require(alias.resolve(strict=True) == target.resolve(strict=True),
            f"Directory alias resolves to the wrong target: {alias} -> {target}")


def local_markdown_links(report: Path) -> list[tuple[str, Path]]:
    """Read inline and reference links, excluding fenced code and remote URLs."""
    lines = []
    fence = None
    for line in report.read_text(encoding="utf-8").splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    text = "\n".join(lines)
    destinations = [match.group(1) for pattern in (INLINE_LINK, REFERENCE_LINK)
                    for match in pattern.finditer(text)]
    links = []
    for destination in destinations:
        destination = destination.removeprefix("<").removesuffix(">")
        parts = urlsplit(destination)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        decoded = unquote(parts.path)
        decoded = re.sub(r"\\([\\() ])", r"\1", decoded)
        target = Path(decoded)
        if not target.is_absolute():
            target = report.parent / target
        # Do not normalize '..' before checking: a parent component can be a symlink.
        links.append((destination, target))
    return links


def verify_layout(catalog_path: Path, *, repo_root: Path = REPO_ROOT,
                  strict_registry_snapshot: bool = False) -> dict:
    root = repo_root.resolve(strict=True)
    catalog_path = repo_path(str(catalog_path), root)
    catalog = read_json(catalog_path)
    artifact_base = repo_path(catalog["artifact_root"], root)
    registry_root = repo_path(catalog["registry_root"], root)
    manifest_path = repo_path(catalog["migration_manifest"], root)
    manifest = read_json(manifest_path)
    digests = FileDigests()
    digests.verify(catalog_path, manifest["catalog_sha256"])
    require(artifact_base.is_dir() and not artifact_base.is_symlink(),
            f"Artifact root must be a real directory: {artifact_base}")
    require(registry_root.is_dir() and not registry_root.is_symlink(),
            f"Registry root must be a real directory: {registry_root}")
    studies = catalog["studies"]
    require(isinstance(studies, list) and bool(studies), "Catalog has no studies")
    study_ids, canonical_roots, legacy_roots, review_reports = set(), set(), set(), set()
    for study in studies:
        require(isinstance(study, dict), "Study records must be objects")
        for field in ("study_id", "legacy_root", "artifact_root", "review_report", "entrypoint", "extra_args"):
            require(field in study, f"Study is missing {field}")
        identifier = study["study_id"]
        require(isinstance(identifier, str) and identifier and identifier not in study_ids,
                f"Missing or duplicate study ID: {identifier}")
        study_ids.add(identifier)
        legacy = repo_path(study["legacy_root"], root)
        canonical = repo_path(study["artifact_root"], root)
        require(canonical.is_relative_to(artifact_base), f"Study artifact directory escapes artifact root: {canonical}")
        require(legacy not in legacy_roots and canonical not in canonical_roots,
                f"Duplicate study directory: {identifier}")
        verify_directory_alias(legacy, canonical)
        legacy_roots.add(legacy)
        canonical_roots.add(canonical)
        report = repo_path(study["review_report"], root)
        require(report.is_file(), f"Missing study review report: {report}")
        require(repo_path(study["entrypoint"], root).is_file(), f"Missing study entrypoint: {identifier}")
        require(isinstance(study["extra_args"], list)
                and all(isinstance(value, str) for value in study["extra_args"]),
                f"extra_args must contain strings: {identifier}")
        review_reports.add(report)
    aliases = manifest.get("directory_aliases", [])
    require(isinstance(aliases, list), "directory_aliases must be a list")
    seen_aliases = set()
    for record in aliases:
        alias = repo_path(record["alias_path"], root)
        target = repo_path(record["target_path"], root)
        require(alias not in seen_aliases, f"Duplicate directory alias: {alias}")
        seen_aliases.add(alias)
        verify_directory_alias(alias, target)
    files = manifest["files"]
    require(isinstance(files, list) and bool(files), "Migration manifest records no files")
    seen_legacy, seen_canonical, represented_roots = set(), set(), set()
    for record in files:
        legacy = repo_path(record["legacy_path"], root)
        canonical = repo_path(record["canonical_path"], root)
        require(legacy not in seen_legacy and canonical not in seen_canonical,
                f"Duplicate relocated file record: {legacy}")
        seen_legacy.add(legacy)
        seen_canonical.add(canonical)
        require(not legacy.is_symlink() and not canonical.is_symlink(),
                f"Relocated payload must be a regular file, not a file symlink: {legacy}")
        require(legacy.resolve(strict=True) == canonical.resolve(strict=True),
                f"Legacy and canonical file identities differ: {legacy}")
        require(canonical.is_file(), f"Missing relocated regular file: {canonical}")
        owners = [directory for directory in canonical_roots if canonical.is_relative_to(directory)]
        require(len(owners) == 1, f"Relocated file has no unique catalog study: {canonical}")
        represented_roots.update(owners)
        information = canonical.stat()
        for field, actual in (("size", information.st_size), ("mtime_ns", information.st_mtime_ns)):
            require(type(record[field]) is int and record[field] >= 0 and record[field] == actual,
                    f"Relocated file {field} changed: {canonical}")
        digests.verify(canonical, record["sha256"])
    require(represented_roots == canonical_roots, "A catalog study has no recorded relocated files")
    actual_payload = {path for directory in canonical_roots
                      for path in directory.rglob("*") if path.is_file() or path.is_symlink()}
    require(actual_payload == seen_canonical,
            "A sealed study has missing or additional payload files")
    registry_hashes = manifest["registry_sha256"]
    require(isinstance(registry_hashes, dict) and bool(registry_hashes), "Registry fingerprints are empty")
    expected_registry = {repo_path(path, root): digest for path, digest in registry_hashes.items()}
    require(all(path.is_relative_to(registry_root) for path in expected_registry),
            "Registry fingerprint points outside registry root")
    actual_registry = set()
    for parent, directories, names in os.walk(registry_root, followlinks=False):
        for directory in directories:
            require(not (Path(parent) / directory).is_symlink(),
                    f"Registry contains a directory symlink: {Path(parent) / directory}")
        for name in names:
            path = Path(parent) / name
            require(not path.is_symlink(), f"Registry contains a file symlink: {path}")
            actual_registry.add(path)
    # Later experiments may append events; historical records must stay unchanged.
    require(set(expected_registry).issubset(actual_registry)
            and (not strict_registry_snapshot or actual_registry == set(expected_registry)),
            f"Registry file set differs: missing={sorted(str(p) for p in set(expected_registry)-actual_registry)}, "
            f"extra={sorted(str(p) for p in actual_registry-set(expected_registry))}")
    for path, expected in expected_registry.items():
        digests.verify(path, expected)
    preserved = manifest["preserved_source_sha256"]
    require(isinstance(preserved, dict), "Preserved source fingerprints must be an object")
    for path, expected in preserved.items():
        digests.verify(repo_path(path, root, allow_external=True), expected)
    reviews = manifest["review_sha256"]
    require(isinstance(reviews, dict), "Review fingerprints must be an object")
    recorded_reviews = {repo_path(path, root) for path in reviews}
    require(review_reports.issubset(recorded_reviews), "A catalog review report lacks a fingerprint")
    link_count = 0
    for path, expected in reviews.items():
        report = repo_path(path, root)
        digests.verify(report, expected)
        require(report.suffix.lower() == ".md", f"Review report is not Markdown: {report}")
        for destination, target in local_markdown_links(report):
            require(target.exists(), f"Broken local Markdown link in {report}: {destination}")
            link_count += 1
    return {
        "valid": True, "catalog": str(catalog_path), "migration_manifest": str(manifest_path),
        "studies": len(studies), "directory_aliases": len(aliases), "relocated_files": len(files),
        "registry_files": len(expected_registry), "preserved_source_paths": len(preserved),
        "additional_registry_files": len(actual_registry - set(expected_registry)),
        "strict_registry_snapshot": strict_registry_snapshot,
        "review_reports": len(reviews), "local_review_links": link_count,
        "unique_physical_files_hashed": len(digests.cache),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG,
                        help="Catalog path, relative to the repository unless absolute")
    parser.add_argument("--strict-registry-snapshot", action="store_true",
                        help="Also require no test events added since this relocation")
    args = parser.parse_args()
    try:
        result = verify_layout(Path(args.catalog), strict_registry_snapshot=args.strict_registry_snapshot)
    except Exception as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

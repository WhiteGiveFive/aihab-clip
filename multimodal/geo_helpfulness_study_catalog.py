"""Read-only naming, path, and coverage checks for the geo-helpfulness studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = "configs/multimodal_geo_helpfulness_studies_v1.json"
ENTRYPOINT = re.compile(r"(?:run|plot|diagnose|check)_multimodal_geo_helpfulness(?:_[a-z0-9_]+)?\.py$")
MODULE = re.compile(r"geo_helpfulness_[a-z0-9_]+\.py$")
CONFIG = re.compile(r"multimodal_geo_[a-z0-9_]+(?:\.proposed)?\.(?:json|ya?ml)$")
NUMBERED_RECORD = re.compile(r"\d{2}_[^/]+\.md$")
INLINE_LINK = re.compile(r"\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+[^)\n]*)?\)")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", re.MULTILINE)


def markdown_links(record: Path):
    """Yield literal local targets, retaining filesystem symlink/.. semantics."""
    lines, fence = [], None
    for line in record.read_text(encoding="utf-8").splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            # Inline code can mention proposed paths or demonstrate Markdown.
            lines.append(re.sub(r"(`+).*?\1", "", line))
    text = "\n".join(lines)
    for pattern in (INLINE_LINK, REFERENCE_LINK):
        for match in pattern.finditer(text):
            destination = match.group(1).removeprefix("<").removesuffix(">")
            parts = urlsplit(destination)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target = Path(re.sub(r"\\([\\() ])", r"\1", unquote(parts.path)))
            yield destination, target if target.is_absolute() else record.parent / target


def audit_catalog(catalog_path: str | Path = DEFAULT_CATALOG, *, repo_root: Path = REPO_ROOT) -> dict:
    root = repo_root.resolve()
    location = Path(catalog_path)
    if not location.is_absolute():
        location = root / location
    catalog = json.loads(location.read_text(encoding="utf-8"))
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)
        return bool(condition)

    def path(value, *, directory=False):
        if not check(isinstance(value, str) and bool(value), f"Invalid path value: {value!r}"):
            return None
        relative = Path(value)
        if not check(not relative.is_absolute() and ".." not in relative.parts,
                     f"Catalog paths must stay relative to the repository: {value}"):
            return None
        result = root / relative
        check(result.is_dir() if directory else result.is_file(), f"Missing {'directory' if directory else 'file'}: {value}")
        return result

    def values(record, field, label):
        items = record.get(field)
        if not check(isinstance(items, list), f"{label}: {field} must be a list"):
            return []
        return items

    if not isinstance(catalog, dict):
        return {"valid": False, "errors": ["Catalog must be a JSON object"]}
    check(type(catalog.get("schema_version")) is int and catalog["schema_version"] == 1,
          "Unsupported catalog schema_version")
    base = path(catalog.get("artifact_root"), directory=True)
    studies = values(catalog, "studies", "catalog")
    check(bool(studies), "Catalog has no studies")
    covered = {name: set() for name in ("records", "entrypoints", "modules", "configs", "artifact_roots")}
    ids = set()
    naming = {"entrypoints": ("tools", ENTRYPOINT), "modules": ("multimodal", MODULE), "configs": ("configs", CONFIG)}
    for study in studies:
        if not check(isinstance(study, dict), "Study entries must be objects"):
            continue
        identifier = study.get("id")
        if check(isinstance(identifier, str) and bool(identifier), "Study is missing a string id"):
            check(identifier not in ids, f"Duplicate study id: {identifier}")
            ids.add(identifier)
        for field in ("title", "status"):
            check(isinstance(study.get(field), str) and bool(study[field]), f"{identifier}: missing {field}")
        for field in covered:
            for value in values(study, field, str(identifier)):
                target = path(value, directory=field == "artifact_roots")
                if target is None:
                    continue
                covered[field].add(target)
                if field in naming:
                    folder, pattern = naming[field]
                    check(target.parent == root / folder and pattern.fullmatch(target.name) is not None,
                          f"Noncanonical {field} name: {value}")
                    if field == "configs" and ".proposed." in target.name:
                        check(study.get("status") == "proposal",
                              f"Proposed configuration is not marked as a proposal: {value}")
                elif field == "records":
                    check(target.suffix == ".md", f"Development record must be Markdown: {value}")
                elif base is not None:
                    check(target != base and target.is_relative_to(base), f"Artifact root is outside the study base: {value}")
        supporting = study.get("supporting_files", [])
        if check(isinstance(supporting, list), f"{identifier}: supporting_files must be a list"):
            for value in supporting:
                path(value)
    shared = catalog.get("shared_modules", [])
    if check(isinstance(shared, list), "shared_modules must be a list"):
        for value in shared:
            target = path(value)
            if target is not None:
                covered["modules"].add(target)
                check(target.parent == root / "multimodal" and MODULE.fullmatch(target.name) is not None,
                      f"Noncanonical shared module name: {value}")
    legacy = values(catalog, "legacy_tools", "catalog")
    legacy_paths = set()
    for record in legacy:
        if not check(isinstance(record, dict), "Legacy tool entries must be objects"):
            continue
        target = path(record.get("path"))
        if target is not None:
            check(target not in legacy_paths, f"Duplicate legacy tool: {record['path']}")
            legacy_paths.add(target)
        canonical = path(record.get("canonical_entrypoint"))
        if canonical is not None:
            check(canonical.parent == root / "tools" and ENTRYPOINT.fullmatch(canonical.name) is not None,
                  f"Legacy mapping has a noncanonical entrypoint: {record.get('canonical_entrypoint')}")
            check(canonical in covered["entrypoints"], f"Legacy entrypoint is not assigned to a study: {canonical.relative_to(root)}")
        check("canonical_subcommand" in record and (record["canonical_subcommand"] is None
              or isinstance(record["canonical_subcommand"], str)),
              "Legacy canonical_subcommand must be a string or null")
        check(isinstance(record.get("role"), str) and bool(record["role"]), "Legacy mapping is missing its role")
    real_artifacts = set()
    compatibility_aliases = 0
    if base is not None and base.is_dir():
        for child in base.iterdir():
            if child.is_symlink():
                compatibility_aliases += 1
            elif child.is_dir():
                real_artifacts.add(child)
    owned_artifacts = {base / item.relative_to(base).parts[0] for item in covered["artifact_roots"]
                       if base is not None and item != base and item.is_relative_to(base)}
    for child in sorted(real_artifacts - owned_artifacts):
        errors.append(f"Uncatalogued artifact directory: {child.relative_to(root)}")
    tools = set((root / "tools").glob("*geo_helpfulness*.py"))
    tools.update((root / "tools").glob("*geometry*.py"))
    tools.discard(root / "tools/check_multimodal_geo_helpfulness_studies.py")
    for tool in sorted(tools - covered["entrypoints"] - legacy_paths):
        errors.append(f"Uncatalogued geo-helpfulness tool: {tool.relative_to(root)}")
    modules = set((root / "multimodal").glob("geo_helpfulness*.py"))
    for module in sorted(modules - covered["modules"]):
        errors.append(f"Uncatalogued geo-helpfulness module: {module.relative_to(root)}")
    configs = {item for item in (root / "configs").glob("multimodal_geo_*.*")
               if item.suffix in (".json", ".yaml", ".yml")}
    for config in sorted(configs - covered["configs"]):
        errors.append(f"Uncatalogued geo-helpfulness configuration: {config.relative_to(root)}")
    record_root = root / "dev_plans/reliability_aware_geo_fusion"
    numbered_records = {item for item in record_root.glob("*.md") if NUMBERED_RECORD.fullmatch(item.name)}
    for record in sorted(numbered_records - covered["records"]):
        errors.append(f"Uncatalogued numbered development record: {record.relative_to(root)}")
    links = 0
    for record in sorted(numbered_records | {item for item in covered["records"] if NUMBERED_RECORD.fullmatch(item.name)}):
        if not record.is_file():
            continue
        for destination, target in markdown_links(record):
            check(target.exists(), f"Broken local Markdown link in {record.relative_to(root)}: {destination}")
            links += 1
    return {"valid": not errors, "catalog": str(location), "studies": len(studies),
            "canonical_entrypoints": len(covered["entrypoints"]), "legacy_tools": len(legacy_paths),
            "modules": len(covered["modules"]), "configs": len(covered["configs"]),
            "development_records": len(covered["records"]), "numbered_records_checked": len(numbered_records),
            "local_markdown_links_checked": links, "artifact_directories": len(real_artifacts),
            "compatibility_aliases_ignored": compatibility_aliases, "errors": errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    args = parser.parse_args(argv)
    try:
        result = audit_catalog(args.catalog)
    except Exception as error:
        result = {"valid": False, "catalog": str(args.catalog), "errors": [str(error)]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1

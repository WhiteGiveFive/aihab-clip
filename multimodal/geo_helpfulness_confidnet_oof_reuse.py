"""Copy authenticated seed-one feature producers into an amended study.

The original study remains immutable, including its declaration and pause. Only
feature bytes and extraction metadata are reused; new manifests bind the copied
producers to the new declaration. The caller owns the extraction-stage lock and
validates the amended declaration's schema and experimental settings.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from multimodal import geo_helpfulness_confidnet_oof as original


PRODUCER_FILES = frozenset({"features.npz", "extraction.json"})


def _bound_files(old_root: Path, new_declaration: dict[str, Any], old_declaration: dict[str, Any]) -> None:
    """Require both prior source provenance and every old artifact to be bound."""

    new_hashes = new_declaration.get("file_sha256")
    if not isinstance(new_hashes, dict):
        raise ValueError("Amended declaration has no file fingerprint mapping")
    for relative, digest in old_declaration["file_sha256"].items():
        if new_hashes.get(relative) != digest:
            raise ValueError(f"Amended declaration changes or omits an old source: {relative}")
    for path in sorted(old_root.rglob("*")):
        if not path.is_file():
            continue
        relative = original._relative(path)
        if new_hashes.get(relative) != original._hash(path):
            raise ValueError(f"Old artifact is not authenticated by amended declaration: {relative}")


def _producer_manifest(directory: Path, root: Path, fold: int | None) -> dict[str, Any]:
    manifest = original._validate_stage(directory, root, role="frozen_oof_producer")
    if (manifest.get("metadata") != {"seed": 1, "fold": fold}
            or set(manifest["file_sha256"]) != PRODUCER_FILES):
        raise ValueError("Reusable producer has wrong file schema, seed or fold")
    extraction = original._json(directory / "extraction.json")
    if extraction.get("training_seed") != 1 or extraction.get("producer_fold") != fold:
        raise ValueError("Reusable extraction metadata has wrong seed or fold")
    return manifest


def import_seed_one(original_root: str | Path, new_root: str | Path) -> dict[str, Any]:
    """Import the five verified seed-one producers without fitting or overwrite.

    The original pooled bundle is validated, but never copied: the amended
    workflow will reconstruct and seal its own pool from these producers.
    Existing destinations must have the same source bytes and valid new-study
    lineage. All provenance and existing-copy checks precede any writes.
    """

    old_root = Path(original_root).resolve()
    target_root = Path(new_root).resolve()
    if old_root == target_root or old_root in target_root.parents or target_root in old_root.parents:
        raise ValueError("Original and amended study directories must be disjoint")
    old_declaration = original.validate_declaration(old_root, check_parents=True)
    new_declaration = original._json(target_root / "declaration.json")
    _bound_files(old_root, new_declaration, old_declaration)
    original._validate_features(old_root, 1)
    sources: list[tuple[int | None, Path, Path, dict[str, Any]]] = []
    for fold in (*original.FOLDS, None):
        key = "validation" if fold is None else f"fold_{fold}"
        relative = Path("producers/seed_1") / key
        source = old_root / relative
        target = target_root / relative
        manifest = _producer_manifest(source, old_root, fold)
        if target.exists():
            target_manifest = _producer_manifest(target, target_root, fold)
            if target_manifest["file_sha256"] != manifest["file_sha256"]:
                raise ValueError("Existing amended producer differs from the original feature bytes")
        sources.append((fold, source, target, manifest))

    copied = 0
    for fold, source, target, source_manifest in sources:
        if target.exists():
            continue
        key = "validation" if fold is None else f"fold_{fold}"
        staged = original._staging(target_root, f"import_seed_1_{key}")
        for filename in sorted(PRODUCER_FILES):
            shutil.copyfile(source / filename, staged / filename)
            if original._hash(staged / filename) != source_manifest["file_sha256"][filename]:
                raise ValueError("Imported producer bytes do not match the authenticated original")
        original._seal_stage(
            staged, target, target_root, role="frozen_oof_producer",
            metadata={"seed": 1, "fold": fold},
        )
        target_manifest = _producer_manifest(target, target_root, fold)
        if target_manifest["file_sha256"] != source_manifest["file_sha256"]:
            raise ValueError("Imported producer fingerprint differs after sealing")
        copied += 1
    return {
        "seed": 1,
        "copied_producers": copied,
        "reused_producers": len(sources) - copied,
        "copied_files": copied * len(PRODUCER_FILES),
        "original_declaration_sha256": original._hash(old_root / "declaration.json"),
    }

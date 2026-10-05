from __future__ import annotations

import json
from pathlib import Path

import pytest

from multimodal import geo_helpfulness_confidnet_oof as original
from multimodal import geo_helpfulness_confidnet_oof_reuse as reuse


@pytest.fixture
def old_and_new(tmp_path, monkeypatch):
    project = tmp_path / "project"
    old_root, new_root = project / "original", project / "amended"
    old_root.mkdir(parents=True)
    new_root.mkdir()
    monkeypatch.setattr(original, "PROJECT_ROOT", project)
    source = project / "old_source.py"
    source.write_text("immutable original implementation")
    old_declaration = {"file_sha256": {"old_source.py": original._hash(source)}}
    original._write_json(old_root / "declaration.json", old_declaration)
    manifests = {}
    for fold in (*original.FOLDS, None):
        key = "validation" if fold is None else f"fold_{fold}"
        staged = original._staging(old_root, key)
        (staged / "features.npz").write_bytes(f"authenticated synthetic arrays for {key}".encode())
        original._write_json(staged / "extraction.json", {"training_seed": 1, "producer_fold": fold})
        destination = old_root / "producers/seed_1" / key
        original._seal_stage(staged, destination, old_root, role="frozen_oof_producer",
                             metadata={"seed": 1, "fold": fold})
        manifests[key] = original._hash(destination / "manifest.json")
    staged = original._staging(old_root, "pooled")
    (staged / "train.npz").write_bytes(b"synthetic combined training array")
    (staged / "validation.npz").write_bytes(b"synthetic combined validation array")
    original._write_json(staged / "extraction.json", {"training_seed": 1})
    original._seal_stage(staged, old_root / "features/seed_1", old_root, role="pooled_oof_features",
                         metadata={"seed": 1, "producer_manifests": manifests})
    pause = old_root / "pauses/seed_1/image"
    pause.mkdir(parents=True)
    (pause / "reason.json").write_text('{"reason":"verification failure"}')
    hashes = {**old_declaration["file_sha256"], **{
        original._relative(path): original._hash(path)
        for path in old_root.rglob("*") if path.is_file()
    }}
    original._write_json(new_root / "declaration.json", {"file_sha256": hashes, "new_study": True})
    calls = []

    def validate_declaration(root, *, check_parents):
        assert root == old_root and check_parents is True
        calls.append("declaration")
        if original._hash(source) != old_declaration["file_sha256"]["old_source.py"]:
            raise ValueError("original source changed")
        return old_declaration

    def validate_features(root, seed):
        assert root == old_root and seed == 1
        calls.append("features")
        manifest = original._validate_stage(root / "features/seed_1", root, role="pooled_oof_features")
        for key, digest in manifest["metadata"]["producer_manifests"].items():
            if original._hash(root / "producers/seed_1" / key / "manifest.json") != digest:
                raise ValueError("pooled producer lineage changed")

    monkeypatch.setattr(original, "validate_declaration", validate_declaration)
    monkeypatch.setattr(original, "_validate_features", validate_features)
    return old_root, new_root, calls


def _snapshot(root):
    return {str(path.relative_to(root)): (original._hash(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_import_copies_only_five_producer_payloads_and_preserves_original(old_and_new):
    old_root, new_root, calls = old_and_new
    before = _snapshot(old_root)
    result = reuse.import_seed_one(old_root, new_root)
    assert calls == ["declaration", "features"]
    assert result["copied_producers"] == 5
    assert result["reused_producers"] == 0
    assert result["copied_files"] == 10
    assert _snapshot(old_root) == before
    assert not (new_root / "features").exists()
    assert not (new_root / "pauses").exists()
    assert not (new_root / "confidence").exists()
    for fold in (*original.FOLDS, None):
        key = "validation" if fold is None else f"fold_{fold}"
        relative = Path("producers/seed_1") / key
        assert {p.name for p in (new_root / relative).iterdir()} == {*reuse.PRODUCER_FILES, "manifest.json"}
        old_manifest = original._json(old_root / relative / "manifest.json")
        new_manifest = original._validate_stage(new_root / relative, new_root, role="frozen_oof_producer")
        assert old_manifest["file_sha256"] == new_manifest["file_sha256"]
        assert old_manifest["declaration_sha256"] != new_manifest["declaration_sha256"]


def test_resume_is_byte_and_mtime_preserving(old_and_new):
    old_root, new_root, _ = old_and_new
    reuse.import_seed_one(old_root, new_root)
    before = _snapshot(new_root)
    result = reuse.import_seed_one(old_root, new_root)
    assert result["copied_files"] == 0
    assert result["reused_producers"] == 5
    assert _snapshot(new_root) == before


@pytest.mark.parametrize("change", ["missing_source", "wrong_source", "unbound_artifact", "old_source_mutated", "old_artifact_mutated"])
def test_import_rejects_unbound_or_changed_original_before_writes(old_and_new, change):
    old_root, new_root, _ = old_and_new
    declaration = original._json(new_root / "declaration.json")
    if change == "missing_source":
        del declaration["file_sha256"]["old_source.py"]
    elif change == "wrong_source":
        declaration["file_sha256"]["old_source.py"] = "bad"
    elif change == "unbound_artifact":
        del declaration["file_sha256"][original._relative(old_root / "pauses/seed_1/image/reason.json")]
    elif change == "old_source_mutated":
        (old_root.parent / "old_source.py").write_text("changed")
    else:
        (old_root / "producers/seed_1/fold_0/features.npz").write_bytes(b"changed")
    (new_root / "declaration.json").write_text(json.dumps(declaration))
    with pytest.raises(ValueError):
        reuse.import_seed_one(old_root, new_root)
    assert not (new_root / "producers").exists()


def test_resume_rejects_resealed_but_different_feature_bytes(old_and_new):
    old_root, new_root, _ = old_and_new
    reuse.import_seed_one(old_root, new_root)
    directory = new_root / "producers/seed_1/fold_3"
    (directory / "features.npz").write_bytes(b"different authenticated bytes")
    manifest = original._json(directory / "manifest.json")
    manifest["file_sha256"]["features.npz"] = original._hash(directory / "features.npz")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from the original"):
        reuse.import_seed_one(old_root, new_root)


def test_resume_rejects_wrong_producer_metadata(old_and_new):
    old_root, new_root, _ = old_and_new
    reuse.import_seed_one(old_root, new_root)
    path = new_root / "producers/seed_1/fold_0/manifest.json"
    manifest = original._json(path)
    manifest["metadata"]["seed"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="wrong file schema, seed or fold"):
        reuse.import_seed_one(old_root, new_root)


@pytest.mark.parametrize("relation", ["same", "inside", "outside"])
def test_source_and_destination_must_be_disjoint(tmp_path, relation):
    old = tmp_path / "old"
    new = old if relation == "same" else (old / "new" if relation == "inside" else tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        reuse.import_seed_one(old, new)

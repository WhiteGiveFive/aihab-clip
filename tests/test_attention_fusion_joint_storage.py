"""Check shared weights, atomic references, export and retention safeguards."""

import json
from pathlib import Path
import shutil

import pytest
import torch

from multimodal.attention_fusion_joint_storage import (
    JointCheckpointIntegrityError,
    JointCheckpointStore,
    JointStorageError,
)


def test_identical_weights_are_stored_once_and_final_resume_share_them(tmp_path):
    store = JointCheckpointStore(tmp_path)
    weights = {"bias": torch.arange(3, dtype=torch.float32), "weight": torch.ones(3, 4)}
    ref = store.put_weights(weights)
    assert store.put_weights(dict(reversed(list(weights.items())))) == ref
    assert len(list(store.objects.glob("*.safetensors"))) == 1
    state = store.put_state({"optimizer": {"step": torch.tensor(2)}, "epoch": 2})
    manifest_path = store.write_manifest(tmp_path / "run/manifest.json", {
        "last": {"visual": ref, "state": state, "epoch": 2},
        "final": {"visual": ref, "epoch": 2},
    })
    manifest = store.read_manifest(manifest_path)
    assert manifest["last"]["visual"] == manifest["final"]["visual"]
    restored = store.load_weights(ref)
    assert all(torch.equal(weights[name], restored[name]) for name in weights)
    assert store.load_state(state)["epoch"] == 2
    assert not list(store.objects.glob(".pending-*"))


def test_weight_snapshot_supports_shared_storage_and_noncontiguous_inputs(tmp_path):
    store = JointCheckpointStore(tmp_path)
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    ref = store.put_weights({"a": base, "b": base, "c": base.T})
    base.add_(10)
    values = store.load_weights(ref)
    assert torch.equal(values["a"], values["b"])
    assert torch.equal(values["c"], values["a"].T)
    assert values["a"][0, 0] == 0


@pytest.mark.parametrize("mutation", ["size", "hash"])
def test_object_corruption_is_detected_before_loading(tmp_path, mutation):
    store = JointCheckpointStore(tmp_path)
    ref = store.put_weights({"x": torch.ones(5)})
    path = store.resolve(ref)
    path.chmod(0o644)
    data = path.read_bytes()
    path.write_bytes(data + b"corruption" if mutation == "size" else data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(JointStorageError, match=f"{mutation} changed"):
        store.load_weights(ref)


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", "objects/../outside", "objects\\outside"])
def test_reference_path_escapes_are_rejected(tmp_path, path):
    store = JointCheckpointStore(tmp_path)
    ref = store.put_weights({"x": torch.ones(1)})
    with pytest.raises(JointStorageError, match="outside"):
        store.load_weights({**ref, "path": path})


def test_reference_symlinks_are_rejected(tmp_path):
    store = JointCheckpointStore(tmp_path / "store")
    ref = store.put_weights({"x": torch.ones(1)})
    path = store.resolve(ref)
    outside = tmp_path / "outside"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(JointStorageError, match="symbolic"):
        store.load_weights(ref)


def test_manifest_corruption_and_missing_dependency_rejected(tmp_path):
    store = JointCheckpointStore(tmp_path)
    ref = store.put_weights({"x": torch.ones(1)})
    path = store.write_manifest(tmp_path / "manifest.json", {"final": {"weights": ref}, "epoch": 50})
    payload = json.loads(path.read_text())
    payload["epoch"] = 49
    path.write_text(json.dumps(payload))
    with pytest.raises(JointStorageError, match="self-hash"):
        store.read_manifest(path)
    store.resolve(ref).unlink()
    with pytest.raises(JointStorageError, match="missing"):
        store.write_manifest(tmp_path / "bad.json", {"final": ref})
    assert not (tmp_path / "bad.json").exists()


def test_failed_manifest_publication_preserves_previous_transaction(tmp_path, monkeypatch):
    store = JointCheckpointStore(tmp_path)
    ref = store.put_weights({"x": torch.ones(1)})
    path = store.write_manifest(tmp_path / "manifest.json", {"last": {"weights": ref, "epoch": 1}})
    original = path.read_bytes()
    import multimodal.attention_fusion_joint_storage as storage
    replace = storage.os.replace

    def interrupted(source, destination):
        if Path(destination) == path:
            raise OSError("simulated interruption")
        return replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        store.write_manifest(path, {"last": {"weights": ref, "epoch": 2}})
    assert path.read_bytes() == original
    assert store.read_manifest(path)["last"]["epoch"] == 1
    assert not list(tmp_path.glob(".manifest.json-*"))


def test_export_loads_without_original_store_including_nested_dependencies(tmp_path):
    store = JointCheckpointStore(tmp_path / "source")
    visual = store.put_weights({"visual": torch.arange(4)})
    fusion = store.put_weights({"fusion": torch.ones(2)})
    state = store.put_state({"weights": visual, "rng": torch.get_rng_state()})
    manifest = store.write_manifest(store.root / "runs/run1/manifest.json", {
        "last": {"state": state, "fusion": fusion}, "best": {"visual": visual, "fusion": fusion},
        "config": {"classes": ["a", "b"]},
    })
    exported_manifest = store.export_bundle(manifest, tmp_path / "exported")
    assert len(list((tmp_path / "exported/objects").iterdir())) == 3
    shutil.rmtree(store.root)
    exported = JointCheckpointStore(tmp_path / "exported")
    payload = exported.read_manifest(exported_manifest)
    restored_state = exported.load_state(payload["last"]["state"])
    assert torch.equal(exported.load_weights(restored_state["weights"])["visual"], torch.arange(4))
    assert torch.equal(exported.load_weights(payload["best"]["fusion"])["fusion"], torch.ones(2))
    with pytest.raises(JointStorageError, match="already exists"):
        exported.export_bundle(exported_manifest, tmp_path / "exported")


def test_cleanup_preserves_shared_encoder_and_rejects_retained_object_deletion(tmp_path):
    store = JointCheckpointStore(tmp_path)
    encoder = store.put_weights({"encoder": torch.ones(3)})
    discarded_head = store.put_weights({"head": torch.zeros(1)})
    retained_head = store.put_weights({"head": torch.ones(1)})
    discarded = store.write_manifest(tmp_path / "runs/discarded.json", {
        "final": {"visual": encoder, "fusion": discarded_head},
    })
    retained = store.write_manifest(tmp_path / "runs/retained.json", {
        "final": {"visual": encoder, "fusion": retained_head},
    })
    plan = store.plan_cleanup([retained])
    assert {ref["path"] for ref in plan["remove"]} == {discarded_head["path"]}
    assert plan["reclaim_bytes"] == discarded_head["size"]
    assert store.resolve(discarded_head).exists()  # The plan makes no changes.
    with pytest.raises(JointStorageError, match="retained"):
        store.discard_unreferenced([discarded_head, encoder], [retained])
    assert store.resolve(discarded_head).exists()  # All-or-nothing validation.
    removed = store.discard_unreferenced([discarded_head], [retained])
    assert removed == [discarded_head["path"]]
    assert store.read_manifest(retained)["final"]["visual"] == encoder
    with pytest.raises(JointStorageError, match="missing"):
        store.read_manifest(discarded)


def test_state_embedded_references_protect_shared_weights_during_cleanup(tmp_path):
    store = JointCheckpointStore(tmp_path)
    encoder = store.put_weights({"encoder": torch.ones(3)})
    state = store.put_state({"visual": encoder, "optimizer": {"x": torch.ones(2)}})
    manifest = store.write_manifest(tmp_path / "manifest.json", {"last": {"state": state}})
    assert store.plan_cleanup([manifest])["remove"] == []
    with pytest.raises(JointStorageError, match="retained"):
        store.discard_unreferenced([encoder], [manifest])

    # Pruning a superseded resume state must not implicitly prune its weights.
    store.write_manifest(manifest, {"final": {"visual": encoder}})
    assert store.discard_unreferenced([state], [manifest]) == [state["path"]]
    assert store.load_weights(encoder)["encoder"].tolist() == [1, 1, 1]


def test_cpu_contiguous_tensors_are_not_cloned_unnecessarily(tmp_path, monkeypatch):
    store = JointCheckpointStore(tmp_path)
    value = torch.ones(10)
    import multimodal.attention_fusion_joint_storage as storage
    real_save = storage.save_file
    seen = []

    def capture(tensors, path):
        seen.append(tensors["x"].data_ptr())
        return real_save(tensors, path)

    monkeypatch.setattr(storage, "save_file", capture)
    store.put_weights({"x": value})
    assert seen == [value.data_ptr()]


def test_matching_hash_has_no_cache_advice_or_diagnostics(tmp_path, monkeypatch):
    import multimodal.attention_fusion_joint_storage as storage
    store = JointCheckpointStore(tmp_path)
    ref = store.put_weights({"x": torch.ones(1)})
    calls = []
    monkeypatch.setattr(storage, "_sha256", lambda path: calls.append(path) or ref["sha256"])
    monkeypatch.setattr(storage, "_refresh_read_cache",
                        lambda path: pytest.fail("a matching read must not refresh the cache"))
    assert store.resolve(ref) == store.root / ref["path"]
    assert len(calls) == 1
    assert not (store.root / "diagnostics").exists()


@pytest.mark.parametrize("bad_reads", [1, 2])
def test_hash_retry_recovers_only_when_original_digest_matches(tmp_path, monkeypatch, bad_reads):
    import multimodal.attention_fusion_joint_storage as storage
    store = JointCheckpointStore(tmp_path)
    ref = store.put_weights({"x": torch.ones(1)})
    path = store.resolve(ref)
    original = path.read_bytes()
    hashes = iter(["0" * 64] * bad_reads + [ref["sha256"]])
    cache_calls = []
    monkeypatch.setattr(storage, "_sha256", lambda path: next(hashes))
    monkeypatch.setattr(storage, "_refresh_read_cache", lambda path: cache_calls.append(path) or {
        "status": "requested", "advice": "POSIX_FADV_DONTNEED",
    })

    assert store.resolve(ref) == path
    assert path.read_bytes() == original
    assert len(cache_calls) == bad_reads
    diagnostic = json.loads((store.root / "diagnostics/checkpoint_integrity.jsonl").read_text())
    assert diagnostic["status"] == "recovered"
    assert diagnostic["expected"] == diagnostic["observed"] == ref["sha256"]
    assert diagnostic["attempts"] == bad_reads + 1
    assert diagnostic["checks"][0]["cache_advice"] is None
    assert diagnostic["checks"][-1]["cache_advice"]["status"] == "requested"


def test_hash_retry_stops_after_three_mismatches_with_structured_error(tmp_path, monkeypatch):
    import multimodal.attention_fusion_joint_storage as storage
    store = JointCheckpointStore(tmp_path)
    ref = store.put_weights({"x": torch.ones(1)})
    calls, advice = [], []
    monkeypatch.setattr(storage, "_sha256", lambda path: calls.append(path) or "0" * 64)
    monkeypatch.setattr(storage, "_refresh_read_cache",
                        lambda path: advice.append(path) or {"status": "unavailable"})
    monkeypatch.setattr(storage, "load_file",
                        lambda *args, **kwargs: pytest.fail("unverified weights must not load"))
    with pytest.raises(JointCheckpointIntegrityError, match="hash changed") as caught:
        store.load_weights(ref)
    assert isinstance(caught.value, JointStorageError)
    assert caught.value.to_dict() == {
        "path": ref["path"], "expected": ref["sha256"], "observed": "0" * 64, "attempts": 3,
    }
    assert len(calls) == 3
    assert len(advice) == 2
    diagnostic = json.loads((store.root / "diagnostics/checkpoint_integrity.jsonl").read_text())
    assert diagnostic["status"] == "unresolved"
    assert diagnostic["attempts"] == 3
    assert diagnostic["checks"][-1]["cache_advice"]["status"] == "unavailable"


def test_retry_checks_size_again_before_hashing(tmp_path, monkeypatch):
    import multimodal.attention_fusion_joint_storage as storage
    store = JointCheckpointStore(tmp_path)
    ref = store.put_weights({"x": torch.ones(1)})
    path = store.resolve(ref)
    calls = []
    monkeypatch.setattr(storage, "_sha256", lambda path: calls.append(path) or "0" * 64)

    def refresh(path):
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"changed")
        return {"status": "requested"}

    monkeypatch.setattr(storage, "_refresh_read_cache", refresh)
    with pytest.raises(JointStorageError, match="size changed"):
        store.resolve(ref)
    assert len(calls) == 1


def test_read_cache_advice_uses_read_only_file_descriptor(tmp_path, monkeypatch):
    import fcntl
    import os
    import multimodal.attention_fusion_joint_storage as storage
    path = tmp_path / "checkpoint"
    path.write_bytes(b"unchanged")
    calls = []

    def advise(fd, offset, length, advice):
        calls.append((offset, length, advice, fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE))

    monkeypatch.setattr(storage.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(storage.os, "posix_fadvise", advise, raising=False)
    assert storage._refresh_read_cache(path)["status"] == "requested"
    assert calls == [(0, 0, 4, os.O_RDONLY)]
    assert path.read_bytes() == b"unchanged"


def test_read_cache_advice_unavailable_and_failure_are_not_bypasses(tmp_path, monkeypatch):
    import multimodal.attention_fusion_joint_storage as storage
    path = tmp_path / "checkpoint"
    path.write_bytes(b"unchanged")
    monkeypatch.delattr(storage.os, "posix_fadvise", raising=False)
    assert storage._refresh_read_cache(path) == {"status": "unavailable"}

    def unavailable(*args):
        raise OSError("simulated unsupported advice")

    monkeypatch.setattr(storage.os, "posix_fadvise", unavailable, raising=False)
    monkeypatch.setattr(storage.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    assert storage._refresh_read_cache(path) == {
        "status": "failed", "error": "simulated unsupported advice",
    }


def test_existing_object_retry_records_without_reentering_storage_lock(tmp_path, monkeypatch):
    import multimodal.attention_fusion_joint_storage as storage
    store = JointCheckpointStore(tmp_path)
    weights = {"x": torch.ones(1)}
    ref = store.put_weights(weights)
    hashes = iter([ref["sha256"], "0" * 64, ref["sha256"]])
    monkeypatch.setattr(storage, "_sha256", lambda path: next(hashes))
    monkeypatch.setattr(storage, "_refresh_read_cache", lambda path: {"status": "unavailable"})
    assert store.put_weights(weights) == ref
    diagnostic = json.loads((store.root / "diagnostics/checkpoint_integrity.jsonl").read_text())
    assert diagnostic["status"] == "recovered"

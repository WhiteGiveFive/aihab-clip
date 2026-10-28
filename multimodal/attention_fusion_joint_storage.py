"""Immutable shared weights and atomic checkpoint manifests for M3J.

Model weights live once in a content-addressed store. A run manifest atomically
binds weight references to its matching optimizer, scheduler and RNG state;
final inference and resume therefore do not need separate model-weight copies.
Resume states are trusted local PyTorch files, not an interchange format for
untrusted downloads. Safetensors are used for model weights and bundle export.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import torch
from safetensors.torch import load_file, save_file


class JointStorageError(ValueError):
    """A checkpoint reference, manifest or storage operation is invalid."""


class JointCheckpointIntegrityError(JointStorageError):
    """A saved object still fails its declared hash after bounded read retries."""

    def __init__(self, path: str, expected: str, observed: str, attempts: int):
        self.path = path
        self.expected = expected
        self.observed = observed
        self.attempts = attempts
        super().__init__(
            f"checkpoint object hash changed: {path} "
            f"(expected {expected}, observed {observed}, {attempts} read attempts)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "expected": self.expected,
                "observed": self.observed, "attempts": self.attempts}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_read_cache(path: Path) -> dict[str, Any]:
    """Advise eviction for this file only; never rewrite it or flush system caches."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return {"status": "unavailable"}
    try:
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        # The OS may ignore advice, so the subsequent digest remains authoritative.
        return {"status": "requested", "advice": "POSIX_FADV_DONTNEED"}
    except OSError as exc:
        return {"status": "failed", "error": str(exc)}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _references(value: Any) -> list[dict[str, Any]]:
    """Find explicit references, including dependencies embedded in a state ref."""
    found: dict[str, dict[str, Any]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if item.get("kind") in {"weights", "state"} and "path" in item:
                ref = dict(item)
                path = ref["path"]
                if not isinstance(path, str):
                    raise JointStorageError("checkpoint reference path must be a string")
                previous = found.get(path)
                if previous is not None:
                    for name in ("kind", "sha256", "size"):
                        if previous.get(name) != ref.get(name):
                            raise JointStorageError(f"conflicting references for {path}")
                found[path] = ref
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return [found[path] for path in sorted(found)]


class JointCheckpointStore:
    """Store referenced weights, resumable state and portable run manifests.

    Per-operation file locks prevent simultaneous object/manifest publication.
    The training orchestrator must additionally hold its study-wide writer lock
    across an epoch's put/publish/prune sequence. This protects references that
    have been written but not yet published when rolling resume objects are
    pruned. No automatic global garbage collection occurs.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "objects"
        if self.objects.is_symlink():
            raise JointStorageError("the object directory cannot be a symbolic link")
        self.objects.mkdir(exist_ok=True)

    @contextmanager
    def _locked(self):
        with (self.root / ".storage.lock").open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _path(self, ref: Mapping[str, Any]) -> Path:
        if not isinstance(ref, Mapping):
            raise JointStorageError("checkpoint reference must be a mapping")
        kind, digest, name, size = (ref.get(key) for key in ("kind", "sha256", "path", "size"))
        if kind not in {"weights", "state"}:
            raise JointStorageError("unknown checkpoint object kind")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise JointStorageError("checkpoint reference has an invalid SHA-256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise JointStorageError("checkpoint reference has an invalid size")
        suffix = "safetensors" if kind == "weights" else "pt"
        expected = f"objects/{digest}.{suffix}"
        if not isinstance(name, str) or PurePosixPath(name).is_absolute() or name != expected:
            raise JointStorageError("checkpoint reference path is outside the content-addressed layout")
        path = self.root / name
        if self.objects.is_symlink() or path.is_symlink() or path.resolve().parent != self.objects.resolve():
            raise JointStorageError("checkpoint object paths cannot escape the store or use symbolic links")
        return path

    def resolve(self, ref: Mapping[str, Any], *, verify: bool = True) -> Path:
        path = self._path(ref)
        if not path.is_file():
            raise JointStorageError(f"missing checkpoint object: {ref['path']}")
        if path.stat().st_size != ref["size"]:
            raise JointStorageError(f"checkpoint object size changed: {ref['path']}")
        if verify:
            self._verify_object_hash(path, ref)
        return path

    def _record_hash_diagnostic(self, record: Mapping[str, Any]) -> None:
        diagnostics = self.root / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        # This separate lock also works when resolve is called under _put's lock.
        with (diagnostics / "checkpoint_integrity.jsonl").open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _verify_object_hash(self, path: Path, ref: Mapping[str, Any]) -> None:
        observed = _sha256(path)
        if observed == ref["sha256"]:
            return
        checks = [{"attempt": 1, "observed": observed, "cache_advice": None}]
        for attempt in (2, 3):
            advice = _refresh_read_cache(path)
            # Recheck the existing path/size safeguards before every retry.
            path = self._path(ref)
            if not path.is_file():
                raise JointStorageError(f"missing checkpoint object: {ref['path']}")
            if path.stat().st_size != ref["size"]:
                raise JointStorageError(f"checkpoint object size changed: {ref['path']}")
            observed = _sha256(path)
            checks.append({"attempt": attempt, "observed": observed, "cache_advice": advice})
            if observed == ref["sha256"]:
                break
        recovered = observed == ref["sha256"]
        self._record_hash_diagnostic({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "path": ref["path"], "expected": ref["sha256"],
            "observed": observed, "attempts": len(checks), "checks": checks,
            "status": "recovered" if recovered else "unresolved",
            "policy": "file_cache_refresh_two_retries_v1",
        })
        if not recovered:
            raise JointCheckpointIntegrityError(
                ref["path"], ref["sha256"], observed, len(checks),
            )

    def _put(self, kind: str, writer) -> dict[str, Any]:
        descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=self.objects)
        os.close(descriptor)
        temporary = Path(name)
        try:
            writer(temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            digest = _sha256(temporary)
            suffix = "safetensors" if kind == "weights" else "pt"
            ref = {"path": f"objects/{digest}.{suffix}", "sha256": digest,
                   "size": temporary.stat().st_size, "kind": kind}
            with self._locked():
                target = self._path(ref)
                if target.exists():
                    self.resolve(ref)
                else:
                    temporary.chmod(0o444)
                    os.replace(temporary, target)
                    _sync_directory(self.objects)
            return ref
        finally:
            temporary.unlink(missing_ok=True)

    def put_weights(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Save deterministic safetensors; existing identical objects are reused.

        CPU contiguous tensors are not copied unless they share storage with
        another entry, which safetensors cannot represent independently.
        """
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise JointStorageError("weights must be a non-empty tensor mapping")
        tensors, seen_storage = {}, set()
        if any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
               for name, value in state_dict.items()):
            raise JointStorageError("weights must map string names to tensors")
        for name in sorted(state_dict):
            value = state_dict[name].detach().cpu().contiguous()
            storage = value.untyped_storage().data_ptr()
            if value.numel() and storage in seen_storage:
                value = value.clone()
            if value.numel():
                seen_storage.add(storage)
            tensors[name] = value
        return self._put("weights", lambda path: save_file(tensors, str(path)))

    def put_state(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Save trusted local optimizer/scheduler/RNG state, excluding model weights.

        The caller supplies model weights separately. If it includes weight
        references in this payload, their closure is also exposed on the state
        reference so export and retention never need to unpickle the payload.
        """
        if not isinstance(payload, Mapping):
            raise JointStorageError("training state must be a mapping")
        dependencies = _references(payload)
        for dependency in dependencies:
            self.resolve(dependency, verify=False)

        def writer(path: Path) -> None:
            # A file object gives torch.save a stable internal archive name.
            with path.open("wb") as handle:
                torch.save(dict(payload), handle)

        ref = self._put("state", writer)
        if dependencies:
            ref["dependencies"] = dependencies
        return ref

    def load_weights(self, ref: Mapping[str, Any], map_location: str | torch.device = "cpu",
                     *, verify: bool = True) -> dict[str, torch.Tensor]:
        if ref.get("kind") != "weights":
            raise JointStorageError("load_weights requires a weights reference")
        return load_file(str(self.resolve(ref, verify=verify)), device=str(map_location))

    def load_state(self, ref: Mapping[str, Any], map_location: str | torch.device = "cpu",
                   *, verify: bool = True) -> dict[str, Any]:
        if ref.get("kind") != "state":
            raise JointStorageError("load_state requires a training-state reference")
        value = torch.load(self.resolve(ref, verify=verify), map_location=map_location, weights_only=False)
        if not isinstance(value, dict):
            raise JointStorageError("saved training state is not a mapping")
        return value

    def write_manifest(self, path: str | Path, payload: Mapping[str, Any]) -> Path:
        """Atomically publish one consistent checkpoint transaction.

        Immutable objects must already exist. This operation does not rehash
        every multi-gigabyte object; reads and export verify their hashes.
        """
        path = Path(path)
        value = dict(payload)
        value.pop("_manifest_sha256", None)
        value["_storage_schema"] = 1
        for ref in _references(value):
            self.resolve(ref, verify=False)
        value["_manifest_sha256"] = hashlib.sha256(_canonical_json(value)).hexdigest()
        encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            with self._locked():
                os.replace(temporary, path)
                _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def read_manifest(self, path: str | Path, *, verify: bool = True) -> dict[str, Any]:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise JointStorageError(f"cannot read checkpoint manifest: {path}") from exc
        if not isinstance(value, dict) or value.get("_storage_schema") != 1:
            raise JointStorageError("unsupported checkpoint manifest schema")
        supplied = value.pop("_manifest_sha256", None)
        expected = hashlib.sha256(_canonical_json(value)).hexdigest()
        if supplied != expected:
            raise JointStorageError("checkpoint manifest self-hash changed")
        value["_manifest_sha256"] = supplied
        for ref in _references(value):
            self.resolve(ref, verify=verify)
        return value

    def export_bundle(self, manifest_path: str | Path, destination: str | Path) -> Path:
        """Copy a manifest and all its objects into a standalone store directory."""
        manifest = self.read_manifest(manifest_path)
        destination = Path(destination).resolve()
        if destination.exists():
            raise JointStorageError("export destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            exported = JointCheckpointStore(staging)
            for ref in _references(manifest):
                source = self.resolve(ref)
                target = exported._path(ref)
                shutil.copyfile(source, target)
                target.chmod(0o444)
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
                exported.resolve(ref)
            exported.write_manifest(staging / "manifest.json", manifest)
            _sync_directory(exported.objects)
            os.rename(staging, destination)
            _sync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination / "manifest.json"

    def _retained(self, manifests: Iterable[str | Path | Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        keep = {}
        for item in manifests:
            value = item if isinstance(item, Mapping) else self.read_manifest(item, verify=False)
            for ref in _references(value):
                self.resolve(ref, verify=False)
                keep[ref["path"]] = ref
        return keep

    def plan_cleanup(self, retained_manifests: Iterable[str | Path | Mapping[str, Any]]) -> dict[str, Any]:
        """Preview objects not needed by the supplied complete retention set.

        This never removes files. Callers must include every model to retain,
        including shared initialization manifests and desired resume states.
        """
        with self._locked():
            keep = self._retained(retained_manifests)
            remove = []
            for path in sorted(self.objects.iterdir()):
                match = re.fullmatch(r"([0-9a-f]{64})\.(safetensors|pt)", path.name)
                if match and path.is_file():
                    ref = {"path": f"objects/{path.name}", "sha256": match.group(1),
                           "size": path.stat().st_size,
                           "kind": "weights" if match.group(2) == "safetensors" else "state"}
                    self._path(ref)
                    if ref["path"] not in keep:
                        remove.append(ref)
            return {"keep": list(keep.values()), "remove": remove,
                    "reclaim_bytes": sum(ref["size"] for ref in remove)}

    def discard_unreferenced(self, refs: Iterable[Mapping[str, Any]],
                             retained_manifests: Iterable[str | Path | Mapping[str, Any]]) -> list[str]:
        """Remove only explicitly nominated, superseded rolling checkpoint objects.

        The caller must hold the study writer lock and supply all live manifests.
        Any retained reference rejects the entire request before deletion. This
        is for an epoch's obsolete resume objects, not automatic model cleanup.
        """
        # Removing a state does not nominate the weights in its dependency list
        # for removal. Those may still belong to a retained inference model.
        nominated = {}
        for ref in refs:
            self._path(ref)
            nominated[ref["path"]] = dict(ref)
        candidates = list(nominated.values())
        with self._locked():
            keep = self._retained(retained_manifests)
            for ref in candidates:
                self._path(ref)
                if ref["path"] in keep:
                    raise JointStorageError(f"cannot discard a retained checkpoint object: {ref['path']}")
                path = self._path(ref)
                if path.exists():
                    self.resolve(ref, verify=False)
            removed = []
            for ref in candidates:
                path = self._path(ref)
                if path.exists():
                    path.unlink()
                    removed.append(ref["path"])
            if removed:
                _sync_directory(self.objects)
            return removed

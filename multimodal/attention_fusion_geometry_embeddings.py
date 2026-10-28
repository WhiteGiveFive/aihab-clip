"""Read-only visual extraction for the declared M3J geometry-routing analysis.

Every encoder supplies both its training reference and validation query space.
Only completed, authenticated pairs are published; source runs and checkpoint
stores are never modified. No fusion head, optimizer or training path is used.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .attention_fusion_joint_inputs import _construct_visual_and_transforms
from .attention_fusion_joint_storage import (
    JointCheckpointIntegrityError, _refresh_read_cache,
)
from .attention_fusion_training import _environment
from .geo_helpfulness_oof import _LegacyImageDataset
from .geo_helpfulness_protocol import canonical_sha256, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLES = ("train", "validation")
SCHEMA = "attention_fusion_geometry.embeddings.v1"


class GeometryEmbeddingError(ValueError):
    """A declared input or completed embedding artifact is invalid."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise GeometryEmbeddingError(message)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_source(path: Path, expected: str, verification: Path,
                   encoder_id: str) -> dict[str, Any]:
    """Bound disk verification to three reads, with file-local cache advice."""
    _require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
             "Declared source SHA-256 is invalid")
    _require(path.is_file(), f"Missing encoder source: {path}")
    checks = []
    for attempt in range(1, 4):
        advice = None if attempt == 1 else _refresh_read_cache(path)
        observed = sha256_file(path)
        checks.append({"attempt": attempt, "observed": observed, "cache_advice": advice})
        if observed == expected:
            break
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "path": str(path),
        "expected": expected, "observed": observed, "attempts": len(checks),
        "checks": checks, "status": "verified" if observed == expected else "unresolved",
        "policy": "file_cache_refresh_two_retries_v1",
    }
    _atomic_json(verification / f"{encoder_id}_source_integrity.json", record)
    if observed != expected:
        raise JointCheckpointIntegrityError(str(path), expected, observed, len(checks))
    return record


def _source_path(value: str) -> Path:
    path = Path(value)
    _require(path.is_absolute() or ".." not in path.parts,
             "Relative encoder paths must not contain traversal")
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _settings(declaration: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    settings = dict(declaration["extraction"])
    _require(settings.get("batch_size") == 16, "Geometry extraction requires batch size 16")
    _require(settings.get("num_workers") == 0, "Geometry extraction requires zero workers")
    _require(settings.get("cpu_threads") == 1, "Geometry extraction requires one CPU thread")
    _require(settings.get("autocast") is False, "Geometry extraction requires FP32 without autocast")
    _require(settings.get("cudnn_allow_tf32") is True
             and settings.get("matmul_allow_tf32") is False,
             "Geometry extraction arithmetic differs from the declared M3J prediction recipe")
    _require(str(torch.device(settings["device"])) == str(device),
             "Extraction device differs from the frozen declaration")
    return {
        **settings, "device": str(device), "parameter_dtype": "float32",
        "image_dtype": "float32", "output_dtype": "float32",
        "float32_matmul_precision": "highest", "cudnn_benchmark": False,
        "cudnn_deterministic": True, "deterministic_algorithms": True,
        "cublas_workspace_config": settings.get("cublas_workspace_config", ":4096:8"),
        "normalization": "torch.nn.functional.normalize(encoded, dim=-1)",
        "model_mode": "eval", "training": False, "optimizer_loaded": False,
        "row_order": "ascending row_uid, separate train and validation loaders",
    }


@contextmanager
def _runtime(settings: Mapping[str, Any], device: torch.device):
    previous = {
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "precision": torch.get_float32_matmul_precision(),
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    workspace = settings["cublas_workspace_config"]
    if device.type == "cuda" and torch.cuda.is_initialized():
        _require(previous["workspace"] == workspace,
                 "CUBLAS workspace must be declared before CUDA initialization")
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        torch.set_num_threads(settings["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = settings["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = settings["matmul_allow_tf32"]
        if device.type == "cuda":
            _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        yield
    finally:
        torch.set_num_threads(previous["threads"])
        torch.use_deterministic_algorithms(previous["deterministic"], warn_only=previous["warn_only"])
        torch.backends.cudnn.benchmark = previous["benchmark"]
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.set_float32_matmul_precision(previous["precision"])
        torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
        # A created CUDA context retains its cuBLAS workspace requirements.
        # Preserve that declaration for the next encoder in the same process.
        if device.type == "cuda" and torch.cuda.is_initialized():
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        elif previous["workspace"] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous["workspace"]


def _read_rows(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    roles, identities = {}, {}
    for role in ROLES:
        path = root / "inputs" / f"{role}_rows.parquet"
        rows = pd.read_parquet(path)
        required = {"row_uid", "plot_idx", "file", "image_source"}
        _require(required.issubset(rows.columns), f"Missing required {role} row columns")
        _require(len(rows) > 0 and not rows[list(required)].isna().any().any(),
                 f"Empty or missing {role} identities")
        rows = rows.copy()
        rows["row_uid"] = rows.row_uid.astype(str)
        rows["plot_idx"] = rows.plot_idx.astype(str)
        _require(rows.row_uid.is_unique, f"Duplicate {role} row identities")
        rows = rows.sort_values("row_uid", kind="stable").reset_index(drop=True)
        roles[role] = rows
        identities[role] = {
            "rows_file_sha256": sha256_file(path), "rows": len(rows),
            "row_uid_sha256": canonical_sha256(rows.row_uid.tolist()),
            "plot_idx_sha256": canonical_sha256(rows.plot_idx.tolist()),
        }
    _require(not (set(roles["train"].row_uid) & set(roles["validation"].row_uid)),
             "Training and validation row identities overlap")
    _require(not (set(roles["train"].plot_idx) & set(roles["validation"].plot_idx)),
             "Training and validation plot identities overlap")
    return roles, identities


def _array_summary(embedding: np.ndarray, rows: pd.DataFrame, expected_dim: int) -> dict[str, Any]:
    _require(embedding.dtype == np.float32 and embedding.shape == (len(rows), expected_dim),
             "Embedding dtype or shape differs from the declared visual architecture")
    _require(np.isfinite(embedding).all(), "Embedding contains nonfinite values")
    norms = np.linalg.norm(embedding, axis=1)
    _require(np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6),
             "Embedding vectors are not L2-normalized")
    return {"shape": list(embedding.shape), "dtype": str(embedding.dtype),
            "norm_min": float(norms.min()), "norm_max": float(norms.max())}


def _existing(destination: Path, binding: Mapping[str, Any],
              roles: Mapping[str, pd.DataFrame], expected_dim: int) -> dict[str, Any] | None:
    if not destination.exists():
        return None
    _require(destination.is_dir() and not destination.is_symlink(), "Invalid embedding destination")
    manifest_path = destination / "manifest.json"
    _require(manifest_path.is_file(), "Incomplete existing embedding directory")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("schema") == SCHEMA, "Unknown embedding cache schema")
    _require(manifest.get("binding") == binding, "Embedding cache declaration or identity changed")
    _require(manifest.get("manifest_sha256") == canonical_sha256({
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }), "Embedding manifest self-hash mismatch")
    _require(set(manifest["files"]) == {f"{role}.npz" for role in ROLES},
             "Embedding manifest has an unexpected file inventory")
    for role, rows in roles.items():
        filename = f"{role}.npz"
        path = destination / filename
        _require(path.is_file() and sha256_file(path) == manifest["files"][filename],
                 f"Embedding cache file hash changed: {filename}")
        with np.load(path, allow_pickle=False) as arrays:
            _require(set(arrays.files) == {"row_uid", "plot_idx", "image_embedding"},
                     "Embedding cache array inventory changed")
            _require(np.array_equal(arrays["row_uid"], rows.row_uid.to_numpy(dtype=str))
                     and np.array_equal(arrays["plot_idx"], rows.plot_idx.to_numpy(dtype=str)),
                     "Embedding cache row alignment changed")
            summary = _array_summary(arrays["image_embedding"], rows, expected_dim)
            _require(summary == manifest["roles"][role], "Embedding cache statistics changed")
    return manifest


def _load_visual(declaration: Mapping[str, Any], source: Mapping[str, Any], path: Path):
    from safetensors import safe_open

    visual, _, transform = _construct_visual_and_transforms(declaration["model_provenance"])
    prefix = source["prefix"]
    _require(prefix in {"", "visual."}, "Unknown visual checkpoint key prefix")
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        names = [name for name in checkpoint.keys() if name.startswith(prefix)]
        _require({name.removeprefix(prefix) for name in names} == set(visual.state_dict()),
                 "Visual source keys differ from the declared architecture")
        state = {name.removeprefix(prefix): checkpoint.get_tensor(name) for name in names}
        visual.load_state_dict(state, strict=True)
        del state
    visual.requires_grad_(False)
    return visual.float().eval(), transform


def _extract_role(visual, transform, rows: pd.DataFrame, source_config: Mapping[str, Any],
                  device: torch.device, settings: Mapping[str, Any], encoder_id: str,
                  role: str, expected_dim: int) -> np.ndarray:
    dataset = _LegacyImageDataset(rows, config=source_config, transform=transform,
                                  include_labels=False)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False,
                        num_workers=settings["num_workers"], drop_last=False,
                        pin_memory=device.type == "cuda")
    output = np.empty((len(rows), expected_dim), dtype=np.float32)
    offset, started, last_report = 0, time.monotonic(), time.monotonic()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for batch_number, (images, indices) in enumerate(loader, start=1):
            count = len(images)
            _require(np.array_equal(np.asarray(indices), np.arange(offset, offset + count)),
                     "Image loader changed declared row order")
            encoded = visual(images.to(device=device, dtype=torch.float32, non_blocking=True))
            _require(encoded.dtype == torch.float32 and tuple(encoded.shape) == (count, expected_dim),
                     "Visual encoder did not return the declared FP32 global vector")
            _require(torch.isfinite(encoded).all().item()
                     and (encoded.norm(dim=-1) > 0).all().item(),
                     "Visual encoder returned zero or nonfinite features")
            output[offset:offset + count] = F.normalize(encoded, dim=-1).cpu().numpy()
            offset += count
            now = time.monotonic()
            if batch_number % 16 == 0 or now - last_report >= 30 or offset == len(rows):
                print(f"geometry embeddings {encoder_id}/{role}: {offset}/{len(rows)} rows "
                      f"({now - started:.1f}s)", flush=True)
                last_report = now
    _require(offset == len(rows), "Image loader omitted rows")
    return output


def extract_embeddings(output_root: Path, encoder_ids: list[str] | None = None,
                       device: str = "cuda") -> dict[str, Any]:
    """Extract declared encoders, retaining valid completed encoder caches."""
    root, selected_device = Path(output_root).resolve(), torch.device(device)
    declaration_path = root / "protocol" / "declaration.json"
    declaration = json.loads(declaration_path.read_text())
    declaration_digest = sha256_file(declaration_path)
    settings = _settings(declaration, selected_device)
    encoders = declaration["encoders"]
    selected = list(encoders) if encoder_ids is None else list(encoder_ids)
    _require(selected and len(selected) == len(set(selected)), "Select unique declared encoder IDs")
    _require(all(name in encoders and re.fullmatch(r"[a-z][a-z0-9_]*", name)
                 for name in selected), "Unknown or invalid encoder ID")
    roles, identities = _read_rows(root)
    for role in ROLES:
        relative = f"inputs/{role}_rows.parquet"
        _require(declaration["input_sha256"].get(relative)
                 == identities[role]["rows_file_sha256"],
                 f"Prepared {role} rows differ from the frozen declaration")
    implementation = {
        f"multimodal/{name}": sha256_file(Path(__file__).with_name(name))
        for name in ("attention_fusion_geometry_embeddings.py", "attention_fusion_joint_inputs.py",
                     "attention_fusion_joint_storage.py", "attention_fusion_training.py",
                     "geo_helpfulness_oof.py", "geo_helpfulness_protocol.py")
    }
    expected_dim = int(declaration["model_provenance"]["encoder_architecture"]["embed_dim"])
    embeddings_root, verification = root / "embeddings", root / "verification"
    _require(not embeddings_root.is_symlink() and not verification.is_symlink(),
             "Extraction output directories cannot be symbolic links")
    embeddings_root.mkdir(exist_ok=True)
    result = {"output_root": str(root), "declaration_sha256": declaration_digest, "encoders": {}}
    for encoder_id in selected:
        source = encoders[encoder_id]
        path = _source_path(source["path"])
        binding = {
            "declaration_sha256": declaration_digest, "encoder_id": encoder_id,
            "source": dict(source), "inputs": identities, "arithmetic": settings,
            "transform": declaration["model_provenance"]["historical_preprocessing_recipe"],
            "transform_for_both_roles": "prediction_transform",
            "decoder": "_LegacyImageDataset: OpenCV BGR, resize 439x439 INTER_LINEAR, PIL",
            "model_provenance_sha256": canonical_sha256(declaration["model_provenance"]),
            "implementation_sha256": implementation,
        }
        destination = embeddings_root / encoder_id
        cached = _existing(destination, binding, roles, expected_dim)
        if cached is not None:
            print(f"geometry embeddings {encoder_id}: authenticated completed cache retained", flush=True)
            result["encoders"][encoder_id] = cached
            continue
        started = time.monotonic()
        print(f"geometry embeddings {encoder_id}: verifying declared checkpoint", flush=True)
        source_check = _verify_source(path, source["sha256"], verification, encoder_id)
        source_stat = path.stat()
        staging = Path(tempfile.mkdtemp(prefix=f".{encoder_id}-", dir=embeddings_root))
        visual = None
        try:
            with _runtime(settings, selected_device):
                visual, transform = _load_visual(declaration, source, path)
                visual = visual.to(selected_device)
                summaries = {}
                for role, rows in roles.items():
                    embedding = _extract_role(visual, transform, rows, declaration["source_config"],
                                              selected_device, settings, encoder_id, role, expected_dim)
                    summaries[role] = _array_summary(embedding, rows, expected_dim)
                    np.savez_compressed(staging / f"{role}.npz", image_embedding=embedding,
                                        row_uid=rows.row_uid.to_numpy(dtype=str),
                                        plot_idx=rows.plot_idx.to_numpy(dtype=str))
                    del embedding
                environment = _environment(selected_device)
            current = path.stat()
            _require((current.st_size, current.st_mtime_ns, current.st_ino)
                     == (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ino),
                     "Encoder source changed during extraction")
            _require(sha256_file(declaration_path) == declaration_digest,
                     "Frozen declaration changed during extraction")
            _require(all(sha256_file(root / "inputs" / f"{role}_rows.parquet")
                         == identities[role]["rows_file_sha256"] for role in ROLES),
                     "Input rows changed during extraction")
            manifest = {
                "schema": SCHEMA, "binding": binding, "roles": summaries,
                "declaration_sha256": declaration_digest,
                "source_verification": source_check, "environment": environment,
                "elapsed_seconds": time.monotonic() - started,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "files": {f"{role}.npz": sha256_file(staging / f"{role}.npz") for role in ROLES},
                "test_features_or_labels_loaded": False,
            }
            manifest["file_sha256"] = dict(manifest["files"])
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            _atomic_json(staging / "manifest.json", manifest)
            _require(not destination.exists(), "Another extraction published this encoder concurrently")
            os.replace(staging, destination)
            result["encoders"][encoder_id] = manifest
            print(f"geometry embeddings {encoder_id}: complete "
                  f"({manifest['elapsed_seconds']:.1f}s)", flush=True)
        finally:
            del visual
            gc.collect()
            if selected_device.type == "cuda" and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            if staging.exists():
                shutil.rmtree(staging)
    return result

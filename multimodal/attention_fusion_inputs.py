"""Audited, development-only image/geo inputs for attention fusion.

No entry point accepts a test role or a mixed image-embedding cache.  Native
geo is joined from the two authenticated development tables; historical
standardized geo and historical expert logits are never materialized.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from multimodal.geo_helpfulness_protocol import canonical_sha256, canonicalize_file, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLES = ("train", "validation")
ARRAY_KEYS = {"image_embedding", "geo_embedding", "label", "row_uid", "plot_idx"}
SCHEMA = "attention_fusion.prepared_inputs.v1"
INPUTS_IMPLEMENTATION_SHA256 = sha256_file(Path(__file__))


class InputError(ValueError):
    """Input identities, provenance or numeric values violate the study."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise InputError(message)


def _config(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = yaml.safe_load(Path(value).read_text())
    _require(isinstance(result, dict), "Configuration must be a mapping")
    _require(result["data"]["allow_test_features_or_labels"] is False,
             "Prepared inputs cannot authorize test access")
    _require(result["image"]["encoder_frozen"] is True, "This input path requires frozen encoders")
    return result


def _path(repo: Path, value: str | Path) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             "Input paths must be repository-relative without traversal")
    result = (repo / relative).resolve()
    _require(result.is_relative_to(repo.resolve()), "Input path escapes repository")
    return result


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    _require(isinstance(result, dict), f"Expected a JSON object: {path}")
    return result


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def _validate_saved_extraction(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    runtime, image = manifest["extraction"], config["image"]
    _require(str(runtime["device"]).split(":")[0] == image["extraction_device"]
             and runtime["batch_size"] == image["extraction_batch_size"]
             and runtime["cudnn_allow_tf32"] is image["extraction_cudnn_allow_tf32"]
             and runtime["matmul_allow_tf32"] is image["extraction_matmul_allow_tf32"],
             "Saved image extraction arithmetic differs from the approved recipe")
    fingerprint = manifest["historical_development_image_fingerprint"]
    _require(fingerprint["matches_historical_protocol"] is True
             and fingerprint["development_image_file_count"]
             == config["data"]["train_rows"] + config["data"]["validation_rows"],
             "Saved inputs lack the historical raw-image fingerprint verification")


def _cache_directory(config: Mapping[str, Any], seed: int, repo: Path) -> Path:
    _require(seed in config["data"]["training_seeds"], "Undeclared training seed")
    template = config["paths"]["prepared_cache_template"]
    paths = [_path(repo, template.format(seed=seed, role=role)) for role in ROLES]
    expected = _path(repo, config["paths"]["artifact_root"]) / "inputs" / f"seed{seed}"
    _require(all(p.parent == expected and p.name == f"{role}.npz"
                 for p, role in zip(paths, ROLES)), "Unexpected prepared-cache destination")
    return expected


def _bind_m1(config: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Authenticate the frozen M1 specification without opening data payloads."""
    directory = _path(repo, config["paths"]["artifact_root"]) / "protocol"
    path = directory / "protocol_manifest.json"
    manifest = _json(path)
    _require(canonical_sha256({k: v for k, v in manifest.items()
                             if k != "manifest_payload_sha256"})
             == manifest["manifest_payload_sha256"], "M1 manifest self-hash mismatch")
    _require(manifest["study_id"] == config["study_id"], "M1 study identity mismatch")
    _require(manifest["effective_config_sha256"] == canonical_sha256(config),
             "Configuration differs from the frozen M1 specification")
    audit_path = directory / "input_audit.json"
    _require(sha256_file(audit_path) == manifest["input_audit"]["sha256"],
             "M1 input-audit hash mismatch")
    audit = _json(audit_path)
    _require(audit["status"] == "passed" and audit["checks"]
             and all(v is True for v in audit["checks"].values()), "M1 audit did not pass")
    return {"manifest": manifest, "audit": audit, "manifest_file_sha256": sha256_file(path)}


def audit_source_inputs(
    config: Mapping[str, Any] | str | Path,
    seeds: Sequence[int] | None = None,
    *,
    repo_root: str | Path = REPO_ROOT,
    authenticate_image_producers: bool = True,
) -> dict[str, Any]:
    """Reauthenticate development producers and return ordered native geo.

    M1's proven audit methods are reused individually.  Its mixed-cache
    inventory method is deliberately not called here.
    """
    from tools.audit_attention_fusion_inputs import InputAudit, DEVELOPMENT_SOURCES, IMAGE_ROOT

    config, repo = _config(config), Path(repo_root).resolve()
    seeds = list(config["data"]["training_seeds"] if seeds is None else seeds)
    _require(seeds and len(set(seeds)) == len(seeds), "Seeds must be nonempty and unique")
    _require(set(seeds) <= set(config["data"]["training_seeds"]), "Undeclared training seed")
    binding = _bind_m1(config, repo)
    source_root = _path(repo, config["paths"]["source_protocol_root"])
    _require(_path(repo, config["paths"]["assignments"])
             == source_root / "protocol/development_assignments.parquet", "Assignment path mismatch")
    _require([Path(p) for p in config["paths"]["development_geo_tables"]]
             == list(DEVELOPMENT_SOURCES), "Only the audited development geo tables are allowed")
    _require(Path(config["paths"]["image_root"]) == IMAGE_ROOT, "Image-source allowlist mismatch")
    audit = InputAudit(repo, source_root)
    audit.load_protocol()
    audit.audit_assignments()
    audit.audit_geo_and_images()
    audit.report["encoders"], audit.report["validation_image_caches"] = [], []
    for seed in seeds if authenticate_image_producers else []:
        producer = audit.audit_encoder(seed)
        audit.audit_validation_cache(seed, producer)
        _require(_path(repo, config["paths"]["encoder_template"].format(seed=seed))
                 == Path(audit.report["encoders"][-1]["checkpoint"]["path"]),
                 "Encoder differs from the audited training-partition producer")
        _require(_path(repo, config["paths"]["validation_cache_template"].format(seed=seed))
                 == Path(audit.report["validation_image_caches"][-1]["features"]["path"]),
                 "Validation cache differs from the authenticated producer")
    # Bind the newly checked bytes to the originally approved M1 inventory.
    original = binding["audit"]
    for key, identity in (("encoders", "checkpoint"), ("validation_image_caches", "features")):
        approved = {item["seed"]: item[identity]["sha256"] for item in original[key]}
        _require(all(item[identity]["sha256"] == approved[item["seed"]]
                     for item in audit.report[key]), f"M1 {key} lineage changed")
    _require(audit.report["protocol_manifest"]["sha256"]
             == original["protocol_manifest"]["sha256"], "Historical protocol changed since M1")
    columns = list(config["data"]["geo_columns"])
    source = pd.concat([
        pd.read_parquet(_path(repo, p), columns=["file", "plot_idx", *columns])
        for p in config["paths"]["development_geo_tables"]
    ], ignore_index=True)
    source["file_lower"] = source.file.map(lambda value: canonicalize_file(value).casefold())
    indexed = source.set_index("file_lower", verify_integrity=True)
    roles = {}
    for role, rows in (("train", audit.train), ("validation", audit.validation)):
        selected = indexed.loc[rows.file_lower]
        _require(np.array_equal(selected.plot_idx.astype(str), rows.plot_idx.astype(str)),
                 "Geo plot join changed order")
        # Preserve source double precision until the selected transformation
        # has fitted its statistics and produced float32 model inputs.
        geo = selected[columns].to_numpy(dtype=np.float64, copy=True)
        _require(np.isfinite(geo).all(), "Native geo must remain finite in float64")
        roles[role] = {"rows": rows, "geo": geo}
    return {"config": config, "repo": repo, "roles": roles,
            "source_config": audit.config, "source_protocol_manifest": audit.manifest,
            "audit": audit.report, "m1_binding": binding}


def load_geo_only_inputs(
    config: Mapping[str, Any] | str | Path, seed: int = 1, *, repo_root: str | Path = REPO_ROOT,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Load native geo without image feature caches or image producer weights.

    The generic fitting API carries a zero image placeholder; geo-only models
    never use it. This is explicitly identified in provenance rather than
    claiming those zeros are extracted image vectors.
    """
    context = audit_source_inputs(config, [seed], repo_root=repo_root, authenticate_image_producers=False)
    config = context["config"]
    roles = []
    for role in ROLES:
        rows = context["roles"][role]["rows"]
        roles.append(_pack(np.zeros((len(rows), config["data"]["image_dim"]), dtype=np.float32),
                           context["roles"][role]))
    provenance = {"protocol_config_sha256": canonical_sha256(config), "train_role": "train",
                  "validation_role": "validation", "source_role": "train",
                  "image_embeddings_loaded": False, "image_placeholder": "zeros_unused_by_geo_only",
                  "assignment_sha256": context["audit"]["protocol_files"]["development_assignments.parquet"]["sha256"],
                  "native_geo_source_files": context["audit"]["development_geo_source_files"],
                  "m1_manifest_file_sha256": context["m1_binding"]["manifest_file_sha256"]}
    return roles[0], roles[1], provenance


def validate_arrays(arrays: Mapping[str, np.ndarray], config: Mapping[str, Any],
                    role: str, rows: pd.DataFrame | None = None) -> None:
    _require(role in ROLES, "Only train and validation roles are accepted")
    _require(set(arrays) == ARRAY_KEYS, "Unexpected prepared-input array keys")
    count = int(config["data"][f"{role}_rows"])
    image, geo, label = (arrays[k] for k in ("image_embedding", "geo_embedding", "label"))
    _require(image.shape == (count, config["data"]["image_dim"]) and image.dtype == np.float32,
             "Image shape/dtype mismatch")
    _require(geo.shape == (count, config["data"]["geo_dim"]) and geo.dtype == np.float64,
             "Geo shape/dtype mismatch")
    _require(np.isfinite(image).all() and np.isfinite(geo).all(), "Nonfinite prepared embeddings")
    norms = np.linalg.norm(image.astype(np.float64), axis=1)
    _require(np.allclose(norms, 1.0, rtol=0, atol=1e-5), "Image vectors must be nonzero and L2-normalized")
    _require(label.shape == (count,) and label.dtype == np.int64, "Dense label shape/dtype mismatch")
    _require(((label >= 0) & (label < config["data"]["num_classes"])).all(), "Dense label outside ontology")
    for key in ("row_uid", "plot_idx"):
        _require(arrays[key].shape == (count,) and arrays[key].dtype.kind in "USiu",
                 f"Invalid {key} identifiers")
    ids = arrays["row_uid"].astype(str)
    _require(len(set(ids)) == count and ids.tolist() == sorted(ids.tolist()),
             "Row identities must be unique and sorted")
    if rows is not None:
        for key, column in (("row_uid", "row_uid"), ("plot_idx", "plot_idx"), ("label", "label_id_dense")):
            _require(np.array_equal(arrays[key].astype(str), rows[column].to_numpy().astype(str)),
                     f"Prepared {key} does not match canonical assignments")


def _image_inventory(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    import cv2

    root = _path(context["repo"], context["config"]["paths"]["image_root"])
    records = []
    for role in ROLES:
        for row in context["roles"][role]["rows"].itertuples():
            path = (root / canonicalize_file(row.file)).resolve()
            _require(path.is_relative_to(root), "Raw image escapes allowlisted source")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            _require(image is not None and image.ndim == 3 and image.shape[2] == 3,
                     f"Cannot decode development image: {path}")
            records.append({"row_uid": str(row.row_uid), "role": role,
                            "file": canonicalize_file(row.file), "bytes": path.stat().st_size,
                            "sha256": sha256_file(path), "decoded_shape": list(image.shape)})
    return records


def _verify_raw_inventory(context: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]) -> None:
    root = _path(context["repo"], context["config"]["paths"]["image_root"])
    for record in inventory:
        path = (root / record["file"]).resolve()
        _require(path.is_relative_to(root), "Raw image inventory escapes allowed source")
        _require(path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"],
                 "Raw development image changed during extraction")


def _validate_historical_image_fingerprint(
    context: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay the original aggregate over sorted row IDs and image byte hashes."""
    available = {str(item["row_uid"]): item["sha256"] for item in inventory}
    root = _path(context["repo"], context["config"]["paths"]["image_root"])
    all_rows = pd.concat([context["roles"][role]["rows"] for role in ROLES]).sort_values("row_uid")
    records = []
    for row in all_rows.itertuples():
        digest = available.get(str(row.row_uid))
        if digest is None:
            path = (root / canonicalize_file(row.file)).resolve()
            _require(path.is_relative_to(root), "Historical fingerprint image escapes source")
            digest = sha256_file(path)
        records.append({"row_uid": str(row.row_uid), "image_file_sha256": digest})
    source = context["source_protocol_manifest"]
    actual = canonical_sha256(records)
    _require(len(records) == source["development_image_file_count"], "Historical raw image count mismatch")
    _require(actual == source["development_image_content_sha256"],
             "Development raw image bytes differ from the historical protocol fingerprint")
    return {"development_image_file_count": len(records), "development_image_content_sha256": actual,
            "matches_historical_protocol": True,
            "algorithm": "sha256_canonical_json_sorted_row_uid_and_image_file_sha256"}


def _configure_extraction(
    config: Mapping[str, Any], device: str, batch_size: int, num_threads: int,
) -> dict[str, Any]:
    import torch

    image = config["image"]
    _require(torch.device(device).type == image["extraction_device"] == "cuda",
             "Historical-compatible image extraction requires CUDA; CPU is not an interchangeable cache producer")
    _require(batch_size == image["extraction_batch_size"] == 16,
             "Historical-compatible image extraction requires batch size 16")
    _require(image["extraction_cudnn_allow_tf32"] is True
             and image["extraction_matmul_allow_tf32"] is False,
             "Historical extraction requires cuDNN TF32 enabled and matrix TF32 disabled")
    _require(num_threads > 0, "Extraction thread count must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _require(os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "CUBLAS workspace configuration differs")
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    _require(torch.cuda.is_available(), "CUDA is unavailable; restore the driver before compatible image extraction")
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                               text=True, capture_output=True, check=False, timeout=10)
        driver = query.stdout.strip() if query.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        driver = None
    return {"device": str(device), "device_name": torch.cuda.get_device_name(device),
            "batch_size": batch_size, "num_threads": num_threads, "precision": "float32",
            "mode": "eval", "encoder_frozen": True, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "driver_version": driver, "deterministic_algorithms": True,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "cudnn_allow_tf32": True, "matmul_allow_tf32": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "process_environment": {name: os.environ.get(name)
                                    for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")}}


def _load_encoder(context: Mapping[str, Any], seed: int, device: str):
    import torch
    from safetensors.torch import load_file
    from multimodal.geo_helpfulness_oof import _load_fresh_encoder

    # _load_fresh_encoder resolves only the pinned, local HF snapshot and checks
    # its file hashes before constructing OpenCLIP. No hub download is enabled.
    model, _, transform, _, initialization = _load_fresh_encoder(context["source_config"], torch.device(device))
    checkpoint = _path(context["repo"], context["config"]["paths"]["encoder_template"].format(seed=seed))
    model.visual.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    model.float().eval().requires_grad_(False)
    _require(all(not p.requires_grad for p in model.parameters()), "Image encoder must remain frozen")
    return model, transform, initialization


def _extract(context: Mapping[str, Any], model: Any, transform: Any, rows: pd.DataFrame,
             *, seed: int, device: str, batch_size: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    from multimodal.geo_helpfulness_oof import _LegacyImageDataset

    _require(batch_size > 0, "Extraction batch size must be positive")
    dataset = _LegacyImageDataset(rows, config=context["source_config"],
                                  transform=transform, include_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    blocks = []
    offset = 0
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, indices in loader:
            _require(torch.equal(indices, torch.arange(offset, offset + len(images))),
                     "Extraction changed row ordering")
            encoded = model.encode_image(images.to(device=device, dtype=torch.float32)).float()
            norms = torch.linalg.vector_norm(encoded, dim=1, keepdim=True)
            _require(torch.isfinite(encoded).all().item() and torch.isfinite(norms).all().item()
                     and (norms > 0).all().item(), "Encoder emitted nonfinite or zero image vectors")
            blocks.append((encoded / norms).cpu().numpy())
            offset += len(images)
            if offset % 256 == 0 or offset == len(rows):
                print(f"Image extraction seed {seed}: {offset}/{len(rows)} rows "
                      f"in {time.perf_counter() - started:.1f}s", flush=True)
    _require(offset == len(rows), "Extraction omitted development rows")
    return np.concatenate(blocks).astype(np.float32, copy=False)


def _pack(image: np.ndarray, role_data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rows = role_data["rows"]
    return {"image_embedding": image, "geo_embedding": role_data["geo"],
            "label": rows.label_id_dense.to_numpy(dtype=np.int64),
            "row_uid": rows.row_uid.to_numpy(dtype=str),
            "plot_idx": rows.plot_idx.to_numpy(dtype=str)}


def _source_seed(context: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {key: next(item for item in context["audit"][key] if item["seed"] == seed)
            for key in ("encoders", "validation_image_caches")}


def _read_validation_image(context: Mapping[str, Any], seed: int) -> np.ndarray:
    path = _path(context["repo"], context["config"]["paths"]["validation_cache_template"].format(seed=seed))
    rows = context["roles"]["validation"]["rows"]
    # The source audit has already authenticated archive bytes and producer
    # lineage. Read identities first and exclusively materialize image values.
    with np.load(path, allow_pickle=False) as saved:
        _require(np.array_equal(saved["row_uid"].astype(str), rows.row_uid.to_numpy().astype(str)),
                 "Validation producer identities changed after audit")
        return saved["image_embedding"].copy()


def _check_replay(context: Mapping[str, Any], seed: int, rows: pd.DataFrame,
                  replay: np.ndarray, reference: np.ndarray,
                  *, device: str, batch_size: int) -> dict[str, Any]:
    left, right = replay.astype(np.float64), reference.astype(np.float64)
    error = np.abs(left - right)
    cosines = np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
    passed = bool(np.allclose(replay, reference, atol=1e-5, rtol=1e-4))
    result = {"row_uids": rows.row_uid.tolist(), "rows": len(rows),
              "atol": 1e-5, "rtol": 1e-4, "passed": passed,
              "max_absolute_error": float(error.max()), "mean_absolute_error": float(error.mean()),
              "minimum_cosine": float(cosines.min()),
              "per_row_max_absolute_error": error.max(axis=1).tolist(),
              "per_row_cosine": cosines.tolist(), "device": device, "batch_size": batch_size}
    if not passed:
        directory = _path(context["repo"], context["config"]["paths"]["artifact_root"]) / "verification/input_replay_failures"
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(result, seed=seed, encoder=_source_seed(context, seed)["encoders"]["checkpoint"],
                      implementation_sha256=INPUTS_IMPLEMENTATION_SHA256,
                      note="Failed tolerance is unchanged; no training inputs were published")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        (directory / f"seed{seed}_{stamp}.json").write_text(json.dumps(record, indent=2) + "\n")
    _require(passed, "Raw-pixel replay disagrees with authenticated validation cache: "
             f"max_abs={result['max_absolute_error']:.9g}, min_cosine={result['minimum_cosine']:.12g}")
    return result


def prepare_inputs(
    config: Mapping[str, Any] | str | Path,
    seeds: Sequence[int] | None = None,
    *,
    device: str = "cuda",
    batch_size: int = 16,
    replay_rows: int = 16,
    num_threads: int = 4,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Prepare immutable per-seed caches; commit only after replay succeeds.

    Interrupted seeds leave no completed destination and can be retried. A
    completed seed is authenticated and reused, never overwritten.
    """
    import torch

    config = _config(config)
    _require(replay_rows == 16, "Historical validation replay must use the first 16 contiguous rows")
    runtime = _configure_extraction(config, device, batch_size, num_threads)
    context = audit_source_inputs(config, seeds, repo_root=repo_root)
    config, repo = context["config"], context["repo"]
    seeds = [item["seed"] for item in context["audit"]["encoders"]]
    inventory = _image_inventory(context)
    historical_images = _validate_historical_image_fingerprint(context, inventory)
    print(f"Authenticated and decoded {len(inventory)} development images; historical byte fingerprint matches", flush=True)
    results = []
    for seed in seeds:
        print(f"Preparing attention-fusion image caches for seed {seed}", flush=True)
        destination = _cache_directory(config, seed, repo)
        if destination.exists():
            for role in ROLES:
                load_prepared_inputs(config, seed, role, repo_root=repo)
            saved = _json(destination / "manifest.json")
            _require(saved["raw_image_inventory_sha256"] == canonical_sha256(inventory),
                     "Raw development image bytes changed after extraction")
            results.append(saved)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".seed{seed}-", dir=destination.parent))
        model = None
        try:
            model, transform, initialization = _load_encoder(context, seed, device)
            validation = _read_validation_image(context, seed)
            rows = context["roles"]["validation"]["rows"]
            selected = np.arange(min(replay_rows, len(rows)))
            replay = _extract(context, model, transform, rows.iloc[selected].reset_index(drop=True),
                              seed=seed, device=device, batch_size=batch_size)
            reference = validation[selected]
            # Preserve the complete historical batch and cuDNN arithmetic.
            replay_report = _check_replay(context, seed, rows.iloc[selected], replay, reference,
                                          device=device, batch_size=batch_size)
            train = _extract(context, model, transform, context["roles"]["train"]["rows"],
                             seed=seed, device=device, batch_size=batch_size)
            _verify_raw_inventory(context, inventory)
            file_hashes = {}
            for role, image in (("train", train), ("validation", validation)):
                arrays = _pack(image, context["roles"][role])
                validate_arrays(arrays, config, role, context["roles"][role]["rows"])
                path = temporary / f"{role}.npz"
                np.savez_compressed(path, **arrays)
                file_hashes[path.name] = sha256_file(path)
            (temporary / "raw_images.json").write_text(json.dumps(inventory, indent=2) + "\n")
            file_hashes["raw_images.json"] = sha256_file(temporary / "raw_images.json")
            manifest = {
                "schema_version": SCHEMA, "study_id": config["study_id"], "seed": seed,
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
                "effective_config_sha256": canonical_sha256(config),
                "m1_manifest_file_sha256": context["m1_binding"]["manifest_file_sha256"],
                "source": _source_seed(context, seed),
                "assignment_sha256": context["audit"]["protocol_files"]["development_assignments.parquet"]["sha256"],
                "file_sha256": file_hashes, "raw_image_inventory_sha256": canonical_sha256(inventory),
                "raw_image_hash_scope": "All development bytes authenticated against the original protocol aggregate",
                "historical_development_image_fingerprint": historical_images,
                "native_geo_source_files": context["audit"]["development_geo_source_files"],
                "image_recipe": config["image"], "class_map": config["data"]["dense_to_canonical_l3"],
                "encoder_initialization": initialization,
                "extraction": runtime,
                "validation_replay": replay_report,
                "boundary": {"test_features_read": False, "test_labels_read": False,
                             "historical_geo_or_logits_materialized": False},
                "code_sha256": {"attention_fusion_inputs.py": INPUTS_IMPLEMENTATION_SHA256,
                                "geo_helpfulness_oof.py": sha256_file(REPO_ROOT / "multimodal/geo_helpfulness_oof.py")},
            }
            manifest["manifest_sha256"] = _manifest_hash(manifest)
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            _require(not destination.exists(), "Refusing to overwrite a completed cache")
            temporary.rename(destination)
            results.append(manifest)
            print(f"Completed seed {seed}: {destination}", flush=True)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"study_id": config["study_id"], "seeds": results,
            "raw_images_decoded_and_hashed": len(inventory),
            "test_features_read": False, "test_labels_read": False}


def load_prepared_inputs(
    config: Mapping[str, Any] | str | Path,
    seed: int,
    role: str,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, np.ndarray]:
    """Authenticate a completed cache and its canonical identities before use."""
    _require(role in ROLES, "Only train and validation roles are accepted; test access is prohibited")
    config, repo = _config(config), Path(repo_root).resolve()
    binding = _bind_m1(config, repo)
    directory = _cache_directory(config, seed, repo)
    manifest = _json(directory / "manifest.json")
    _require(manifest.get("schema_version") == SCHEMA and manifest.get("status") == "complete",
             "Incomplete or unsupported prepared cache")
    _require(_manifest_hash(manifest) == manifest.get("manifest_sha256"), "Prepared manifest hash mismatch")
    _require(manifest["seed"] == seed and manifest["study_id"] == config["study_id"], "Prepared cache identity mismatch")
    _require(manifest["effective_config_sha256"] == canonical_sha256(config), "Prepared cache configuration mismatch")
    _require(manifest["m1_manifest_file_sha256"] == binding["manifest_file_sha256"], "Prepared cache M1 binding mismatch")
    _require(manifest["image_recipe"] == config["image"]
             and manifest["class_map"] == config["data"]["dense_to_canonical_l3"],
             "Prepared image recipe/class map mismatch")
    _validate_saved_extraction(manifest, config)
    _require(manifest["validation_replay"]["passed"] is True
             and manifest["boundary"]["test_features_read"] is False
             and manifest["boundary"]["test_labels_read"] is False,
             "Prepared cache lacks clean-boundary replay evidence")
    _require(set(manifest["file_sha256"]) == {"train.npz", "validation.npz", "raw_images.json"},
             "Prepared file allowlist mismatch")
    expected_encoder = next(item for item in binding["audit"]["encoders"] if item["seed"] == seed)
    _require(manifest["source"]["encoders"]["checkpoint"]["sha256"]
             == expected_encoder["checkpoint"]["sha256"], "Prepared cache encoder mismatch")
    expected_validation = next(item for item in binding["audit"]["validation_image_caches"] if item["seed"] == seed)
    _require(manifest["source"]["validation_image_caches"]["features"]["sha256"]
             == expected_validation["features"]["sha256"], "Prepared validation producer mismatch")
    _require(manifest["native_geo_source_files"] == binding["audit"]["development_geo_source_files"],
             "Prepared native geo sources differ from approved development tables")
    inventory_path = directory / "raw_images.json"
    _require(sha256_file(inventory_path) == manifest["file_sha256"]["raw_images.json"],
             "Prepared raw image inventory hash mismatch")
    _require(canonical_sha256(json.loads(inventory_path.read_text())) == manifest["raw_image_inventory_sha256"],
             "Prepared raw image inventory content mismatch")
    assignment_path = _path(repo, config["paths"]["assignments"])
    approved_assignment = binding["audit"]["protocol_files"]["development_assignments.parquet"]["sha256"]
    _require(manifest["assignment_sha256"] == approved_assignment == sha256_file(assignment_path),
             "Prepared assignment file hash mismatch")
    rows = pd.read_parquet(assignment_path)
    rows = rows.loc[rows.development_role.eq(role)].sort_values("row_uid").reset_index(drop=True)
    path = directory / f"{role}.npz"
    _require(sha256_file(path) == manifest["file_sha256"][path.name], "Prepared cache file hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == ARRAY_KEYS, "Prepared cache has unexpected array keys")
        arrays = {key: saved[key].copy() for key in ARRAY_KEYS}
    validate_arrays(arrays, config, role, rows)
    return arrays


def load_prepared_input_provenance(
    config: Mapping[str, Any] | str | Path, seed: int, *, verification: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return the authenticated producer/cache references used in checkpoints."""
    config, repo = _config(config), Path(repo_root).resolve()
    if verification:
        load_verification_inputs(config, seed, repo_root=repo)
        directory = _verification_directory(config, seed, repo)
    else:
        for role in ROLES:
            load_prepared_inputs(config, seed, role, repo_root=repo)
        directory = _cache_directory(config, seed, repo)
    manifest = _json(directory / "manifest.json")
    encoder = manifest["source"]["encoders"]["checkpoint"]
    initialization = manifest["encoder_initialization"]
    construction_assets = []
    if "snapshot_path" in initialization:
        snapshot = Path(initialization["snapshot_path"])
        construction_assets = [
            {"path": str(snapshot / name), "name": name, "sha256": digest}
            for name, digest in initialization["snapshot_file_sha256"].items()
            if name.endswith(".json")
        ]
    return {"protocol_config_sha256": canonical_sha256(config),
            "source_role": "train", "train_role": "train",
            "validation_role": None if verification else "validation",
            "verification_only": verification,
            "encoder": {"path": encoder["path"], "sha256": encoder["sha256"]},
            "encoder_initialization": initialization,
            "encoder_construction_assets": construction_assets,
            "manifest_path": str(directory / "manifest.json"),
            "manifest_sha256": sha256_file(directory / "manifest.json"),
            "cache_file_sha256": manifest["file_sha256"],
            "assignment_sha256": manifest["assignment_sha256"],
            "raw_image_inventory_sha256": manifest["raw_image_inventory_sha256"],
            "historical_development_image_fingerprint": manifest["historical_development_image_fingerprint"],
            "image_extraction": manifest["extraction"]}


load_input_provenance = load_prepared_input_provenance


def _verification_directory(config: Mapping[str, Any], seed: int, repo: Path) -> Path:
    _require(seed in config["data"]["training_seeds"], "Undeclared training seed")
    return _path(repo, config["paths"]["artifact_root"]) / "verification/input_smoke" / f"seed{seed}"


def prepare_verification_inputs(
    config: Mapping[str, Any] | str | Path, seed: int = 1, *, limit: int = 32,
    device: str = "cuda", batch_size: int = 16, replay_rows: int = 16,
    num_threads: int = 4, repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Bounded raw-pixel verification; never creates a full development cache.

    Only the first sorted training rows are exposed to smoke optimization.
    Validation images are used solely to check the extraction implementation.
    """
    import copy
    import torch

    config = _config(config)
    _require(limit > 0 and replay_rows == 16, "Verification requires positive train rows and first-16 validation replay")
    runtime = _configure_extraction(config, device, batch_size, num_threads)
    started = time.perf_counter()
    context = audit_source_inputs(config, [seed], repo_root=repo_root)
    config, repo = context["config"], context["repo"]
    _require(limit <= len(context["roles"]["train"]["rows"]), "Verification limit exceeds training rows")
    destination = _verification_directory(config, seed, repo)
    if destination.exists():
        arrays = load_verification_inputs(config, seed, repo_root=repo)
        _require(len(arrays["label"]) == limit, "Existing verification cache has another limit")
        return _json(destination / "manifest.json")
    subset = dict(context)
    subset["roles"] = {}
    for role, count in (("train", limit), ("validation", replay_rows)):
        item = context["roles"][role]
        subset["roles"][role] = {"rows": item["rows"].iloc[:count].reset_index(drop=True),
                                 "geo": item["geo"][:count]}
    inventory = _image_inventory(subset)
    historical_images = _validate_historical_image_fingerprint(context, inventory)
    model = None
    temporary = None
    try:
        loading = time.perf_counter()
        model, transform, initialization = _load_encoder(context, seed, device)
        loaded = time.perf_counter()
        validation_rows = subset["roles"]["validation"]["rows"]
        replay = _extract(context, model, transform, validation_rows, seed=seed,
                          device=device, batch_size=batch_size)
        reference = _read_validation_image(context, seed)[:len(validation_rows)]
        replay_report = _check_replay(context, seed, validation_rows, replay, reference,
                                      device=device, batch_size=batch_size)
        replay_done = time.perf_counter()
        train_rows = subset["roles"]["train"]["rows"]
        train_image = _extract(context, model, transform, train_rows,
                               seed=seed, device=device, batch_size=batch_size)
        train_done = time.perf_counter()
        _verify_raw_inventory(subset, inventory)
        arrays = _pack(train_image, subset["roles"]["train"])
        small_config = copy.deepcopy(config)
        small_config["data"]["train_rows"] = limit
        validate_arrays(arrays, small_config, "train", train_rows)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".seed{seed}-", dir=destination.parent))
        np.savez_compressed(temporary / "train.npz", **arrays)
        (temporary / "raw_images.json").write_text(json.dumps(inventory, indent=2) + "\n")
        manifest = {
            "schema_version": "attention_fusion.verification_inputs.v1",
            "study_id": config["study_id"], "seed": seed, "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "verification_only": True, "source_role": "train", "training_rows": limit,
            "effective_config_sha256": canonical_sha256(config),
            "m1_manifest_file_sha256": context["m1_binding"]["manifest_file_sha256"],
            "source": _source_seed(context, seed),
            "assignment_sha256": context["audit"]["protocol_files"]["development_assignments.parquet"]["sha256"],
            "file_sha256": {name: sha256_file(temporary / name)
                            for name in ("train.npz", "raw_images.json")},
            "raw_image_inventory_sha256": canonical_sha256(inventory),
            "raw_image_hash_scope": "Subset byte records plus successful replay of the full historical development aggregate",
            "historical_development_image_fingerprint": historical_images,
            "native_geo_source_files": context["audit"]["development_geo_source_files"],
            "image_recipe": config["image"], "class_map": config["data"]["dense_to_canonical_l3"],
            "encoder_initialization": initialization,
            "validation_replay": replay_report,
            "extraction": runtime,
            "timing_seconds": {"model_load": loaded - loading, "validation_replay": replay_done - loaded,
                               "training_images": train_done - replay_done, "total": time.perf_counter() - started},
            "boundary": {"test_features_read": False, "test_labels_read": False,
                         "validation_used_for_optimization": False},
            "code_sha256": {"attention_fusion_inputs.py": INPUTS_IMPLEMENTATION_SHA256,
                            "geo_helpfulness_oof.py": sha256_file(REPO_ROOT / "multimodal/geo_helpfulness_oof.py")},
        }
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _require(not destination.exists(), "Refusing to overwrite verification inputs")
        temporary.rename(destination)
        return manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        del model
        gc.collect()


def load_verification_inputs(
    config: Mapping[str, Any] | str | Path, seed: int = 1, *, repo_root: str | Path = REPO_ROOT,
) -> dict[str, np.ndarray]:
    import copy

    config, repo = _config(config), Path(repo_root).resolve()
    binding = _bind_m1(config, repo)
    directory = _verification_directory(config, seed, repo)
    manifest = _json(directory / "manifest.json")
    _require(manifest.get("schema_version") == "attention_fusion.verification_inputs.v1"
             and manifest.get("verification_only") is True and manifest.get("source_role") == "train"
             and manifest.get("status") == "complete", "Not a completed training-only verification cache")
    _require(_manifest_hash(manifest) == manifest["manifest_sha256"], "Verification manifest hash mismatch")
    _require(manifest["effective_config_sha256"] == canonical_sha256(config)
             and manifest["m1_manifest_file_sha256"] == binding["manifest_file_sha256"],
             "Verification protocol mismatch")
    _require(manifest["seed"] == seed and manifest["study_id"] == config["study_id"],
             "Verification cache identity mismatch")
    _require(manifest["image_recipe"] == config["image"]
             and manifest["class_map"] == config["data"]["dense_to_canonical_l3"],
             "Verification image recipe/class map mismatch")
    _validate_saved_extraction(manifest, config)
    expected_encoder = next(item for item in binding["audit"]["encoders"] if item["seed"] == seed)
    _require(manifest["source"]["encoders"]["checkpoint"]["sha256"]
             == expected_encoder["checkpoint"]["sha256"], "Verification encoder mismatch")
    _require(manifest["validation_replay"]["passed"] is True
             and manifest["boundary"]["test_features_read"] is False
             and manifest["boundary"]["test_labels_read"] is False
             and manifest["boundary"]["validation_used_for_optimization"] is False,
             "Verification cache lacks clean-boundary replay evidence")
    _require(set(manifest["file_sha256"]) == {"train.npz", "raw_images.json"},
             "Verification file allowlist mismatch")
    inventory_path = directory / "raw_images.json"
    _require(sha256_file(inventory_path) == manifest["file_sha256"]["raw_images.json"]
             and canonical_sha256(json.loads(inventory_path.read_text())) == manifest["raw_image_inventory_sha256"],
             "Verification raw-image inventory mismatch")
    assignment_path = _path(repo, config["paths"]["assignments"])
    _require(sha256_file(assignment_path) == manifest["assignment_sha256"]
             == binding["audit"]["protocol_files"]["development_assignments.parquet"]["sha256"],
             "Verification assignments changed")
    rows = pd.read_parquet(assignment_path)
    rows = rows.loc[rows.development_role.eq("train")].sort_values("row_uid").reset_index(drop=True)
    count = int(manifest["training_rows"])
    _require(0 < count <= len(rows), "Invalid verification row count")
    path = directory / "train.npz"
    _require(sha256_file(path) == manifest["file_sha256"]["train.npz"], "Verification cache hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == ARRAY_KEYS, "Unexpected verification arrays")
        arrays = {key: saved[key].copy() for key in ARRAY_KEYS}
    small_config = copy.deepcopy(config)
    small_config["data"]["train_rows"] = count
    validate_arrays(arrays, small_config, "train", rows.iloc[:count])
    return arrays

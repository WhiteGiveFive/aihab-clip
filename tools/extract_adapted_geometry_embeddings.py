"""Extract fixed, development-trained visual geometry without reading test labels."""

from __future__ import annotations

import argparse
import copy
import gc
import io
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal import geo_helpfulness_oof as m2
from multimodal.geo_helpfulness_protocol import validate_locked_test_identity_manifest
from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash, check_self_hash, check_snapshot, exclusive_lock, file_hash,
    read_json, write_bytes, write_json,
)


ROOT = m2.PROJECT_ROOT
DEFAULT_OUTPUT = ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_geometry_v1/embeddings"
PREVIOUS_TEST = ROOT / "dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_joint_geometry_thresholds/test_4200_predictions.parquet"
CACHE_ROOT = ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/joint_tcp_router_v1/features"
IDENTITY_COLUMNS = ["row_uid", "file", "plot_idx"]


def bind(path: Path, hashes: dict) -> None:
    hashes[str(path.resolve())] = file_hash(path)


def npz_bytes(arrays: dict) -> bytes:
    """Preserve the identity key 'file', which conflicts with savez's argument."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, values in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(values), allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload.getvalue())
    return buffer.getvalue()


def load_rows(context):
    """Read development assignments and an explicit, label-free test projection."""
    sources = {}
    for path in (context.assignments_path, context.resolved_path, context.manifest_path, PREVIOUS_TEST):
        bind(path, sources)
    reference_path = context.protocol_dir / "locked_test_snapshot_ref.json"
    reference = read_json(reference_path)
    registry = m2._resolve_project_path(context.config["paths"]["locked_test_registry_root"])
    identity_path = (registry / reference["identity_manifest_relative_path"]).resolve()
    if not identity_path.is_relative_to(registry):
        raise ValueError("Locked identity path escaped registry")
    identity = read_json(identity_path)
    validate_locked_test_identity_manifest(identity)
    if (file_hash(identity_path) != reference["identity_manifest_file_sha256"]
            or identity["identity_projection_sha256"] != context.protocol_manifest["locked_test_identity_projection_sha256"]):
        raise ValueError("Locked test identity binding changed")
    bind(reference_path, sources)
    bind(identity_path, sources)
    test = pd.DataFrame(identity["rows"])[IDENTITY_COLUMNS].astype(str)
    previous = pd.read_parquet(PREVIOUS_TEST, columns=IDENTITY_COLUMNS).drop_duplicates().astype(str)
    pd.testing.assert_frame_equal(previous.sort_values("row_uid").reset_index(drop=True),
                                  test.sort_values("row_uid").reset_index(drop=True))
    test["image_source"] = str(ROOT / "data/CS_Xplots_2019_2023_test")
    spec = next(item for item in m2._producer_specs(1) if item.fold is None)
    fit, validation = m2._partitions_for_spec(context.assignments, spec)
    groups = [frame[IDENTITY_COLUMNS + ["image_source"]].copy().reset_index(drop=True)
              for frame in (fit, validation, test)]
    if [len(frame) for frame in groups] != [3378, 822, 1347]:
        raise ValueError("Unexpected extraction memberships")
    if [frame.plot_idx.nunique() for frame in groups] != [1300, 325, 531]:
        raise ValueError("Unexpected extraction plot memberships")
    for i in range(3):
        for j in range(i):
            for key in ("row_uid", "plot_idx"):
                if set(groups[i][key]) & set(groups[j][key]):
                    raise ValueError(f"Extraction cohorts overlap on {key}")
    rows = pd.concat(groups, ignore_index=True)
    rows["cohort"] = ["development"] * 4200 + ["test"] * 1347
    config = copy.deepcopy(context.config)
    config["development_universe"]["allowed_image_sources"] = sorted(set(rows.image_source.astype(str)))
    raw_hashes = {}
    for row in rows.to_dict("records"):
        path = m2._resolved_image_path(row, config)
        raw_hashes[str(path)] = file_hash(path)
    sources.update(raw_hashes)
    return rows, groups, config, sources, identity["identity_projection_sha256"]


def cache_reference(seed, producer, sources):
    directory = CACHE_ROOT / f"seed_{seed}"
    manifest = read_json(directory / "manifest.json")
    producers = read_json(directory / "producers.json")
    parent = CACHE_ROOT.parent / "producers" / f"seed_{seed}" / "validation"
    parent_manifest = read_json(parent / "manifest.json")
    extraction = read_json(parent / "extraction.json")
    if producers["producer_manifests"]["validation"] != file_hash(parent / "manifest.json"):
        raise ValueError("Validation extraction manifest changed")
    if (file_hash(parent / "extraction.json") != parent_manifest["file_sha256"]["extraction.json"]
            or manifest["file_sha256"]["validation.npz"] != parent_manifest["file_sha256"]["features.npz"]):
        raise ValueError("Validation extraction content binding differs")
    if (extraction["producer"]["manifest_sha256"] != producer["manifest_sha256"]
            or extraction["checkpoint_file_sha256"]["adapted_visual_tower.safetensors"]
            != producer["checkpoint_file_sha256"]["adapted_visual_tower.safetensors"]):
        raise ValueError("Validation cache belongs to another producer")
    for name in ("producers.json", "validation.npz"):
        if file_hash(directory / name) != manifest["file_sha256"][name]:
            raise ValueError("Validation cache content changed")
    for name in ("manifest.json", "producers.json", "validation.npz"):
        bind(directory / name, sources)
    for name in ("manifest.json", "extraction.json"):
        bind(parent / name, sources)
    with np.load(directory / "validation.npz", allow_pickle=False) as saved:
        return saved["row_uid"].copy(), saved["image_embedding"].copy()


def extract(output: Path, device: str, preflight_only: bool = False):
    started = time.monotonic()
    context = m2.load_frozen_context()
    rows, groups, config, sources, identity_sha = load_rows(context)
    implementation = {}
    for path in (Path(__file__), ROOT / "multimodal/geo_helpfulness_oof.py",
                 ROOT / "multimodal/geo_helpfulness_protocol.py",
                 ROOT / "multimodal/geo_helpfulness_router_m4_common.py"):
        bind(path, implementation)
    print(f"Validated 4200 development and 1347 label-free test image identities ({time.monotonic()-started:.1f}s)", flush=True)
    if preflight_only:
        return
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use an approved GPU-visible execution environment")
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        check_self_hash(manifest, "manifest_sha256")
        check_snapshot(manifest["source_sha256"])
        check_snapshot(manifest["implementation_sha256"])
        check_snapshot({str(output / key): value for key, value in manifest["output_sha256"].items()})
        if manifest["implementation_sha256"] != implementation:
            raise ValueError("Completed extraction uses another implementation")
        print("Validated completed immutable extraction", flush=True)
        return
    from safetensors.torch import load_file

    summaries = {}
    output_hashes = {}
    recipe = config["experts"]["image_encoder"]["fold_contained_adaptation"]["adaptation_recipe"]
    for seed in (1, 2, 3, 4):
        seed_started = time.monotonic()
        spec = next(item for item in m2._producer_specs(seed) if item.fold is None)
        m2.validate_producer(context, spec, seed=seed)
        directory = context.output_root / spec.relative_directory
        producer = read_json(directory / "manifest.json")
        checkpoint = directory / "adapted_visual_tower.safetensors"
        for name in ("manifest.json", "adapted_visual_tower.safetensors"):
            bind(directory / name, sources)
        saved_uids, saved_embeddings = cache_reference(seed, producer, sources)
        seed_output = output / f"seed_{seed}"
        if (seed_output / "manifest.json").exists():
            summary = read_json(seed_output / "manifest.json")
            check_self_hash(summary, "manifest_sha256")
            if (summary["implementation_sha256"] != implementation
                    or summary["checkpoint_sha256"] != file_hash(checkpoint)):
                raise ValueError("Existing seed cache has different provenance")
            check_snapshot(summary["source_sha256"])
            if file_hash(seed_output / "embeddings.npz") != summary["embedding_file_sha256"]:
                raise ValueError("Existing seed embeddings changed")
            print(f"Seed {seed}: validated existing immutable embeddings", flush=True)
        else:
            m2.reset_reproducibility(seed)
            print(f"Seed {seed}: loading frozen adapted visual tower on {target}", flush=True)
            model, _, transform, _, initialization = m2._load_fresh_encoder(config, target)
            model.visual.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
            model.eval()
            model.requires_grad_(False)
            matrices = []
            completed = 0
            for group in groups:
                for start in range(0, len(group), 256):
                    batch_rows = group.iloc[start:start + 256]
                    matrices.append(m2._extract_embeddings(model, batch_rows, transform=transform,
                                    config=config, seed=seed, device=target))
                    completed += len(batch_rows)
                    print(f"Seed {seed}: {completed}/5547 images, {time.monotonic()-seed_started:.1f}s", flush=True)
            embeddings = np.concatenate(matrices)
            if embeddings.shape != (5547, 1152) or embeddings.dtype != np.float32:
                raise ValueError("Invalid extracted embedding shape or dtype")
            validation_rows = rows.iloc[3378:4200]
            index = pd.Index(saved_uids).get_indexer(validation_rows.row_uid)
            if (index < 0).any() or len(set(index)) != 822:
                raise ValueError("Validation embedding replay identities differ")
            expected = saved_embeddings[index]
            replay = embeddings[3378:4200]
            if not np.allclose(replay, expected, atol=1e-6, rtol=1e-6):
                raise ValueError(f"Validation image embeddings do not reproduce: max error {np.max(np.abs(replay-expected))}")
            arrays = {column: rows[column].to_numpy(dtype=str) for column in IDENTITY_COLUMNS + ["cohort"]}
            arrays["image_embedding"] = embeddings
            write_bytes(seed_output / "embeddings.npz", npz_bytes(arrays))
            summary = {
                "schema_version": "adapted_geometry_embeddings.seed.v1", "training_seed": seed,
                "rows": 5547, "development_rows": 4200, "test_rows": 1347, "embedding_dim": 1152,
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
                "producer_manifest_sha256": producer["manifest_sha256"],
                "encoder_fitting_rows": 3378, "encoder_fitting_plots": 1300,
                "encoder_initialization": initialization, "preprocessing": recipe["preprocessing"],
                "input_preprocessing_hash": producer["input_preprocessing_hash"],
                "checkpoint_validation_replay": {"valid": True, "row_count": 822,
                    "comparison": "authenticated same-producer validation image embeddings",
                    "maximum_absolute_embedding_error": float(np.max(np.abs(replay-expected))),
                    "atol": 1e-6, "rtol": 1e-6,
                    "original_producer_checkpoint_reproduction": producer["checkpoint_reproduction"]},
                "source_sha256": dict(sources), "implementation_sha256": implementation,
                "embedding_file_sha256": file_hash(seed_output / "embeddings.npz"),
                "device": str(target), "seconds": time.monotonic() - seed_started,
                "test_labels_read": False, "test_outcomes_read": False, "oof_diagnostics": False,
            }
            summary["manifest_sha256"] = canonical_hash(summary)
            write_json(seed_output / "manifest.json", summary)
            del model, matrices, embeddings, arrays
            gc.collect()
            if target.type == "cuda":
                torch.cuda.empty_cache()
        summaries[str(seed)] = summary
        for name in ("embeddings.npz", "manifest.json"):
            output_hashes[f"seed_{seed}/{name}"] = file_hash(seed_output / name)
    check_snapshot(sources)
    check_snapshot(implementation)
    manifest = {
        "schema_version": "adapted_geometry_embeddings.v1", "training_seeds": [1, 2, 3, 4],
        "rows_per_seed": 5547, "development_rows": 4200, "test_rows": 1347,
        "embedding_dim": 1152, "embedding_dtype": "float32", "l2_normalized": True,
        "identity_projection_sha256": identity_sha,
        "encoder_fit": {"images": 3378, "plots": 1300, "stage": "development_train_to_validation"},
        "preprocessing_match": "Unchanged M2 evaluation: OpenCV BGR, forced 439 resize, OpenCLIP 384 transform, float32 and L2 normalization; same legacy decoding as frozen geometry.",
        "same_encoder_for_reference_and_query": True,
        "test_labels_read": False, "test_outcomes_read": False, "oof_diagnostics": False,
        "source_sha256": sources, "implementation_sha256": implementation,
        "output_sha256": output_hashes, "seeds": summaries,
        "seconds": time.monotonic() - started,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(output / "manifest.json", manifest)
    print(f"Completed and sealed all four seeds in {manifest['seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    with exclusive_lock(args.output):
        extract(args.output.resolve(), args.device, args.preflight_only)


if __name__ == "__main__":
    main()

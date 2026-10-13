#!/usr/bin/env python3
"""Audit attention-fusion inputs without opening test feature payloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from multimodal.geo_helpfulness_protocol import (  # noqa: E402
    FIXED_CLASS_ONTOLOGY,
    IDENTITY_PROJECTION_COLUMNS,
    assignment_fingerprint,
    canonical_sha256,
    canonicalize_file,
    make_row_uid,
    plot_set_fingerprint,
    sha256_file,
    validate_locked_test_identity_manifest,
)

ANALYSIS_ROOT = Path(
    "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
)
DEFAULT_PROTOCOL_ROOT = ANALYSIS_ROOT / "protocol_v1"
SOURCE_ROOT = Path(
    "multimodal_artifacts/joined_tables/cs/"
    "hf-hub_timm_ViT-SO400M-16-SigLIP2-384_pretrained/gse_100m/seed1"
)
DEVELOPMENT_SOURCES = (SOURCE_ROOT / "train.parquet", SOURCE_ROOT / "val.parquet")
IDENTITY_ROOT = Path("multimodal_artifacts/locked_test_registry/cs/gse_100m_cleaned_test")
IMAGE_ROOT = Path("data/CS_Xplots_2019_2023_train")
RAW_GEO = Path(
    "data/cs_geo_gse_100m_20192023/"
    "CS_Xplots_embeddings_per_file_100m_public.parquet"
)
GEO_COLUMNS = [f"A{i:02d}" for i in range(64)]


class AuditFailure(ValueError):
    """A required input contract failed."""


def reject_failed_checks(checks: Mapping[str, bool]) -> None:
    """Reject missing checks and every result other than an explicit True."""
    failed = [name for name, passed in checks.items() if passed is not True]
    if not checks or failed:
        raise AuditFailure(f"Required audit checks failed: {failed or ['no checks']}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AuditFailure(f"Expected JSON object: {path}")
    return value


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def version_record(repo: Path) -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy", "pandas", "pyarrow", "PyYAML", "scikit-learn", "torch",
        "torchvision", "open_clip_torch", "timm", "safetensors",
        "opencv-python", "Pillow", "huggingface_hub",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
        capture_output=True, check=False,
    )
    return {
        "script": file_record(Path(__file__).resolve()),
        "canonical_helpers": file_record(
            SCRIPT_ROOT / "multimodal/geo_helpfulness_protocol.py"
        ),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "python": platform.python_version(),
        "packages": packages,
    }


class InputAudit:
    def __init__(self, repo: Path, protocol: Path) -> None:
        self.repo = repo.resolve()
        self.protocol = protocol.resolve()
        self.report: dict[str, Any] = {
            "schema_version": "attention_fusion.input_audit.v1",
            "study_id": "attention_fusion_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(self.repo),
            "source_protocol_root": str(self.protocol),
            "runtime": version_record(self.repo),
            "checks": {},
            "boundary": {
                "test_labels_read": False,
                "test_features_read": False,
                "mixed_embedding_npz_opened": False,
                "test_access": "sealed_label_blind_identity_and_cache_manifests_only",
            },
            "limitations": [
                "Recorded fitting provenance and artifact bytes are verified; historical "
                "training and neural forward passes are not rerun.",
                "Image existence and nonempty size are checked; decoding and image byte "
                "fingerprint replay are not performed.",
                "The mixed raw geo payload is not opened. Native values are authenticated "
                "against the protocol-bound development source tables.",
                "All-development fusion heads are not trained in M1; existing fit-only "
                "encoders are intentionally reused for both head fitting routes.",
            ],
        }

    def require(self, name: str, passed: Any) -> None:
        self.report["checks"][name] = bool(passed)
        reject_failed_checks({name: self.report["checks"][name]})

    def authenticate(self, path: Path, expected: str, name: str) -> dict[str, Any]:
        record = file_record(path)
        self.require(name, record["sha256"] == expected)
        record["matches_recorded_sha256"] = True
        return record

    def within_repo(self, path: Path) -> Path:
        resolved = path.resolve()
        self.require(f"path_within_repo:{path}", resolved.is_relative_to(self.repo))
        return resolved

    def load_protocol(self) -> None:
        self.within_repo(self.protocol)
        directory = self.protocol / "protocol"
        self.manifest = read_json(directory / "protocol_manifest.json")
        manifest = self.manifest
        self.require(
            "protocol_manifest_self_hash",
            canonical_sha256({
                k: v for k, v in manifest.items() if k != "manifest_payload_sha256"
            }) == manifest["manifest_payload_sha256"],
        )
        self.report["protocol_manifest"] = file_record(directory / "protocol_manifest.json")
        expected_names = {
            "development_assignments.parquet", "locked_test_snapshot_ref.json",
            "resolved_protocol.yaml", "split_balance.csv",
        }
        self.require(
            "protocol_artifact_allowlist",
            set(manifest["artifact_file_sha256"]) == expected_names,
        )
        self.report["protocol_files"] = {
            name: self.authenticate(directory / name, digest, f"protocol_file:{name}")
            for name, digest in manifest["artifact_file_sha256"].items()
        }
        self.config = yaml.safe_load((directory / "resolved_protocol.yaml").read_text())
        self.require(
            "development_source_allowlist",
            [Path(p) for p in self.config["paths"]["development_source_tables"]]
            == list(DEVELOPMENT_SOURCES),
        )
        self.require(
            "development_source_hash_allowlist",
            set(manifest["development_source_file_sha256"])
            == {str(p) for p in DEVELOPMENT_SOURCES},
        )
        self.require(
            "identity_registry_allowlist",
            Path(self.config["paths"]["locked_test_registry_root"]) == IDENTITY_ROOT,
        )
        self.require(
            "class_ontology_hash",
            canonical_sha256(self.config["class_ontology"])
            == manifest["class_map_sha256"],
        )
        reference = read_json(directory / "locked_test_snapshot_ref.json")
        expected_relative = Path(reference["snapshot_id"]) / "locked_test_identity_manifest.json"
        self.require(
            "identity_manifest_filename",
            Path(reference["identity_manifest_relative_path"]) == expected_relative,
        )
        identity_path = self.within_repo(self.repo / IDENTITY_ROOT / expected_relative)
        self.report["label_blind_test_identity"] = self.authenticate(
            identity_path, reference["identity_manifest_file_sha256"], "test_identity_file_hash"
        )
        identity = read_json(identity_path)
        validate_locked_test_identity_manifest(identity)
        self.require(
            "test_identity_projection",
            identity["identity_projection_sha256"]
            == reference["identity_projection_sha256"],
        )
        self.require(
            "test_identity_row_allowlist",
            all(set(row) == set(IDENTITY_PROJECTION_COLUMNS) for row in identity["rows"]),
        )
        self.test_identity = pd.DataFrame(identity["rows"])
        self.assignments = pd.read_parquet(directory / "development_assignments.parquet")

    def audit_assignments(self) -> None:
        frame = self.assignments
        self.train = (
            frame[frame.development_role.eq("train")]
            .sort_values("row_uid").reset_index(drop=True)
        )
        self.validation = (
            frame[frame.development_role.eq("validation")]
            .sort_values("row_uid").reset_index(drop=True)
        )
        self.require(
            "assignment_content_hash",
            assignment_fingerprint(frame) == self.manifest["assignment_content_sha256"],
        )
        self.require("development_shape", (len(frame), frame.plot_idx.nunique()) == (4200, 1625))
        self.require(
            "development_train_shape",
            (len(self.train), self.train.plot_idx.nunique()) == (3378, 1300),
        )
        self.require(
            "development_validation_shape",
            (len(self.validation), self.validation.plot_idx.nunique()) == (822, 325),
        )
        self.require("development_roles", set(frame.development_role) == {"train", "validation"})
        self.require("unique_row_uids", frame.row_uid.is_unique)
        self.require("unique_filenames", frame.file_lower.is_unique)
        self.require(
            "one_label_per_plot",
            frame.groupby("plot_idx").label_id_dense.nunique().eq(1).all(),
        )
        self.require(
            "one_role_per_plot",
            frame.groupby("plot_idx").development_role.nunique().eq(1).all(),
        )
        self.require(
            "canonical_filenames",
            all(canonicalize_file(r.file).casefold() == r.file_lower
                for r in frame.itertuples()),
        )
        self.require(
            "row_uid_regeneration",
            all(make_row_uid(self.manifest["dataset_id"], r.file, r.plot_idx) == r.row_uid
                for r in frame.itertuples()),
        )
        for column in ("row_uid", "file_lower", "plot_idx"):
            self.require(
                f"no_test_overlap:{column}",
                not set(frame[column]) & set(self.test_identity[column]),
            )
        self.require(
            "no_train_validation_plot_overlap",
            not set(self.train.plot_idx) & set(self.validation.plot_idx),
        )
        labels = (
            frame[["label_id_dense", "canonical_l3_id", "label_name"]]
            .drop_duplicates().sort_values("label_id_dense")
        )
        self.require(
            "fixed_class_ontology",
            list(labels.itertuples(index=False, name=None)) == list(FIXED_CLASS_ONTOLOGY),
        )
        self.report["assignments"] = {
            "rows": len(frame), "plots": int(frame.plot_idx.nunique()),
            "train_rows": len(self.train), "train_plots": 1300,
            "validation_rows": len(self.validation), "validation_plots": 325,
            "content_sha256": assignment_fingerprint(frame),
            "class_map": labels.to_dict("records"),
            "support": frame.groupby(["development_role", "label_id_dense"]).agg(
                images=("row_uid", "size"), plots=("plot_idx", "nunique")
            ).reset_index().to_dict("records"),
        }

    def audit_geo_and_images(self) -> None:
        frames, records = [], []
        for relative in DEVELOPMENT_SOURCES:
            path = self.within_repo(self.repo / relative)
            records.append(self.authenticate(
                path, self.manifest["development_source_file_sha256"][str(relative)],
                f"development_source_hash:{relative.name}",
            ))
            frames.append(pd.read_parquet(path, columns=["file", "plot_idx", *GEO_COLUMNS]))
        self.report["development_geo_source_files"] = records
        source = pd.concat(frames, ignore_index=True)
        source["file_lower"] = source.file.map(lambda x: canonicalize_file(x).casefold())
        self.require("geo_unique_filenames", source.file_lower.is_unique)
        self.require(
            "geo_exact_development_universe",
            set(source.file_lower) == set(self.assignments.file_lower),
        )
        ordered = source.set_index("file_lower").loc[self.assignments.file_lower].reset_index()
        self.require(
            "geo_plot_identity_join",
            np.array_equal(ordered.plot_idx.astype(str), self.assignments.plot_idx.astype(str)),
        )
        values = ordered[GEO_COLUMNS].to_numpy(dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        train_values = values[self.assignments.development_role.eq("train").to_numpy()]
        self.require("geo_shape", values.shape == (4200, 64))
        self.require("geo_finite", np.isfinite(values).all())
        self.require("geo_nonzero_norms", (norms > 0).all())
        self.require(
            "geo_one_vector_per_plot",
            ordered.groupby("plot_idx")[GEO_COLUMNS].nunique().le(1).all().all(),
        )
        self.require("geo_nonconstant_train_coordinates", (train_values.std(axis=0) > 0).all())
        self.report["native_geo"] = {
            "feature_columns": GEO_COLUMNS, "shape": list(values.shape),
            "norm_quantiles_0_25_50_75_100": np.quantile(norms, [0, .25, .5, .75, 1]).tolist(),
            "coordinate_min": float(values.min()), "coordinate_max": float(values.max()),
            "minimum_train_coordinate_std": float(train_values.std(axis=0).min()),
            "std_ddof": 0, "train_rows_for_zscore": len(train_values),
            "matrix_sha256_ordered_float64": hashlib.sha256(values.tobytes()).hexdigest(),
            "source": "native A00..A63 in authenticated development joined tables",
            "standardized_producer_geo_reused": False,
        }
        raw_schema = pq.read_schema(self.repo / RAW_GEO).names
        spatial_names = {
            "latitude", "longitude", "lat", "lon", "easting", "northing",
            "cell_id", "pixel_id", "tile_id", "geometry",
        }
        available = [c for c in raw_schema if c.casefold() in spatial_names]
        self.report["spatial_inventory"] = {
            "raw_geo_path": str(self.repo / RAW_GEO), "schema_only_read": True,
            "raw_schema_columns": raw_schema, "explicit_spatial_columns": available,
            "spatial_overlap_audit_possible_from_current_inputs": bool(available),
            "limitation": (
                "Plot disjointness is verified. No explicit coordinate or cell identifiers "
                "are present in the audited inputs; shared pixels or nearby plots cannot "
                "be excluded. Opaque ID/REP_ID fields are not assumed to identify raster cells."
            ),
        }
        self.require(
            "image_source_allowlist",
            {Path(source) for source in self.assignments.image_source} == {IMAGE_ROOT},
        )
        paths = []
        for row in self.assignments.itertuples():
            paths.append((self.repo / IMAGE_ROOT / canonicalize_file(row.file)).resolve())
        self.require(
            "image_path_allowlist",
            all(path.is_relative_to((self.repo / IMAGE_ROOT).resolve()) for path in paths),
        )
        self.require("all_development_images_exist", all(path.is_file() for path in paths))
        sizes = [path.stat().st_size for path in paths]
        self.require("all_development_images_nonempty", all(size > 0 for size in sizes))
        self.report["development_images"] = {
            "existing_nonempty_files": len(sizes), "bytes_total": sum(sizes),
            "image_source": str(IMAGE_ROOT), "decoded_or_rehashed": False,
        }

    def audit_encoder(self, seed: int) -> dict[str, Any]:
        directory = self.protocol / f"development_validation/seed_{seed}"
        manifest = read_json(directory / "manifest.json")
        prefix = f"encoder_seed_{seed}:"
        self.require(
            prefix + "manifest_self_hash",
            canonical_sha256({
                k: v for k, v in manifest.items() if k != "manifest_sha256"
            }) == manifest["manifest_sha256"],
        )
        stage_file = self.authenticate(
            directory / "resolved_stage_config.yaml",
            manifest["component_file_sha256"]["resolved_stage_config.yaml"],
            prefix + "resolved_stage_hash",
        )
        stage = yaml.safe_load((directory / "resolved_stage_config.yaml").read_text())
        parents = manifest["parent_artifact_hashes"]
        protocol_files = self.report["protocol_files"]
        fit_support = {
            str(k): int(v)
            for k, v in self.train.groupby("label_id_dense").plot_idx.nunique().items()
        }
        adaptation = stage["adaptation_recipe"]
        declared_adaptation = self.config["experts"]["image_encoder"][
            "fold_contained_adaptation"
        ]["adaptation_recipe"]
        checks = {
            "assignment_file": (
                parents["development_assignments"]["file_sha256"]
                == protocol_files["development_assignments.parquet"]["sha256"]
            ),
            "assignment_content": (
                parents["development_assignments"]["content_sha256"]
                == assignment_fingerprint(self.assignments)
            ),
            "protocol_file": (
                parents["protocol_manifest"]["file_sha256"]
                == self.report["protocol_manifest"]["sha256"]
            ),
            "resolved_protocol_file": (
                parents["resolved_protocol"]["file_sha256"]
                == protocol_files["resolved_protocol.yaml"]["sha256"]
            ),
            "fitting_rows": manifest["fitting_row_count"] == len(self.train),
            "fitting_plot_hash": (
                manifest["fitting_plot_hash"] == plot_set_fingerprint(self.train.plot_idx)
            ),
            "prediction_plot_hash": (
                manifest["prediction_plot_hash"]
                == plot_set_fingerprint(self.validation.plot_idx)
            ),
            "fitting_row_order": (
                stage["fitting_row_uids_sha256"]
                == canonical_sha256(self.train.row_uid.tolist())
            ),
            "prediction_row_order": (
                stage["prediction_row_uids_sha256"]
                == canonical_sha256(self.validation.row_uid.tolist())
            ),
            "class_map": manifest["class_map_hash"] == self.manifest["class_map_sha256"],
            "fitting_class_support": (
                manifest["fitting_class_plot_support_by_dense_id"] == fit_support
            ),
            "recipe": (
                manifest["encoder_adaptation_recipe_hash"] == canonical_sha256(adaptation)
            ),
            "recipe_matches_protocol": adaptation == declared_adaptation,
            "preprocessing": (
                manifest["input_preprocessing_hash"]
                == canonical_sha256(adaptation["preprocessing"])
            ),
            "initialization": (
                manifest["encoder_initialization_hash"]
                == canonical_sha256(stage["encoder_initialization"])
            ),
            "seed": manifest["training_seed"] == stage["training_seed"] == seed,
            "stage": (
                manifest["stage_id"] == stage["stage_id"]
                == "development_train_to_validation"
            ),
            "zero_overlap_record": manifest["zero_plot_overlap"] is True,
        }
        for name, passed in checks.items():
            self.require(prefix + name, passed)
        checkpoint = self.authenticate(
            directory / "adapted_visual_tower.safetensors",
            manifest["checkpoint_file_sha256"]["adapted_visual_tower.safetensors"],
            prefix + "checkpoint_hash",
        )
        record = {
            "seed": seed, "manifest": file_record(directory / "manifest.json"),
            "resolved_stage": stage_file, "checkpoint": checkpoint,
            "fitting_rows": len(self.train), "fitting_plots": 1300,
            "fitting_plot_hash": manifest["fitting_plot_hash"],
            "fitting_row_uids_sha256": stage["fitting_row_uids_sha256"],
            "encoder_initialization": manifest["encoder_initialization"],
            "adaptation_recipe": stage["adaptation_recipe"],
            "historical_checkpoint_reproduction": manifest["checkpoint_reproduction"],
            "forward_replay_rerun": False,
        }
        self.report["encoders"].append(record)
        return manifest

    def audit_validation_cache(self, seed: int, producer: dict[str, Any]) -> None:
        directory = (
            self.repo / ANALYSIS_ROOT
            / f"joint_tcp_router_v1/producers/seed_{seed}/validation"
        )
        prefix = f"validation_cache_seed_{seed}:"
        manifest = read_json(directory / "manifest.json")
        self.require(prefix + "manifest_role", manifest["role"] == "joint_tcp_producer")
        self.require(
            prefix + "manifest_metadata",
            manifest["metadata"] == {"fold": None, "seed": seed},
        )
        self.require(
            prefix + "file_allowlist",
            set(manifest["file_sha256"]) == {"extraction.json", "features.npz"},
        )
        extraction_file = self.authenticate(
            directory / "extraction.json", manifest["file_sha256"]["extraction.json"],
            prefix + "extraction_hash",
        )
        extraction = read_json(directory / "extraction.json")
        expected_suffix = (
            self.protocol.relative_to(self.repo) / f"development_validation/seed_{seed}"
        )
        producer_path = Path(extraction["producer"]["directory"])
        checkpoint_hash = self.report["encoders"][-1]["checkpoint"]["sha256"]
        fit, prediction = extraction["expert_fit"], extraction["prediction"]
        checks = {
            "producer_path": (
                producer_path.parts[-len(expected_suffix.parts):] == expected_suffix.parts
            ),
            "producer_manifest": (
                extraction["producer"]["manifest_sha256"] == producer["manifest_sha256"]
            ),
            "checkpoint_lineage": (
                extraction["checkpoint_file_sha256"]["adapted_visual_tower.safetensors"]
                == checkpoint_hash
            ),
            "parent_assignments": (
                extraction["parent_artifact_hashes"] == producer["parent_artifact_hashes"]
            ),
            "seed": extraction["training_seed"] == extraction["producer"]["training_seed"] == seed,
            "fit_shape": (fit["row_count"], fit["plot_count"]) == (3378, 1300),
            "prediction_shape": (prediction["row_count"], prediction["plot_count"]) == (822, 325),
            "validation_producer": extraction["producer_fold"] is None,
            "image_feature": (
                extraction["image_feature"] == "l2_normalized_adapted_image_embedding"
                and extraction["image_dimension"] == 1152
            ),
            "no_label_payload": extraction["validation_labels_in_payload"] is False,
            "no_test_access_record": extraction["locked_test_accessed"] is False,
        }
        for name, passed in checks.items():
            self.require(prefix + name, passed)
        feature_file = self.authenticate(
            directory / "features.npz", manifest["file_sha256"]["features.npz"],
            prefix + "features_hash",
        )
        # Authenticate lineage first, then identities; image values are loaded last.
        with np.load(directory / "features.npz", allow_pickle=False) as saved:
            expected_keys = {
                "row_uid", "plot_idx", "producer_fold", "image_embedding",
                "geo_embedding", "image_logits", "geo_logits", "raw_logits",
            }
            self.require(prefix + "array_allowlist", set(saved.files) == expected_keys)
            self.require(
                prefix + "row_identity",
                np.array_equal(saved["row_uid"], self.validation.row_uid.to_numpy()),
            )
            self.require(
                prefix + "plot_identity",
                np.array_equal(saved["plot_idx"], self.validation.plot_idx.to_numpy()),
            )
            self.require(
                prefix + "fold_identity",
                np.array_equal(
                    saved["producer_fold"], np.full(len(self.validation), -1, dtype=np.int64)
                ),
            )
            image = saved["image_embedding"]
            self.require(
                prefix + "shape_dtype",
                image.shape == (822, 1152) and image.dtype == np.float32,
            )
            self.require(prefix + "finite", np.isfinite(image).all())
            norms = np.linalg.norm(image.astype(np.float64), axis=1)
            self.require(prefix + "l2_normalized", np.allclose(norms, 1, atol=1e-6, rtol=0))
        self.report["validation_image_caches"].append({
            "seed": seed, "manifest": file_record(directory / "manifest.json"),
            "extraction": extraction_file, "features": feature_file,
            "image_shape": list(image.shape), "dtype": str(image.dtype),
            "image_norm_minmax": [float(norms.min()), float(norms.max())],
            "historical_reproduction": extraction["reproduction"],
            "geo_array_not_for_native_reuse": extraction["geo_feature"],
            "forward_replay_rerun": False,
        })

    def inventory_excluded_caches(self) -> None:
        excluded = []
        for seed in range(1, 5):
            directory = self.repo / ANALYSIS_ROOT / f"adapted_geometry_v1/embeddings/seed_{seed}"
            manifest = read_json(directory / "manifest.json")
            excluded.append({
                "path": str(directory / "embeddings.npz"), "seed": seed,
                "declared_rows": manifest["rows"],
                "declared_development_rows": manifest["development_rows"],
                "declared_test_rows": manifest["test_rows"],
                "payload_opened": False,
                "reason": (
                    "Combined development/test feature array is outside the development boundary."
                ),
            })
        self.report["mixed_caches_excluded"] = excluded
        self.report["training_image_cache"] = {
            "clean_single_producer_cache_verified": False,
            "action": (
                "Extract 3378 development-train vectors from each verified encoder. "
                "Optionally re-extract validation and compare with the authenticated cache."
            ),
            "known_alternative": "joint_tcp_router_v1/features/seed_<seed>/train.npz",
            "rejection_reason": (
                "The OOF training cache combines four different fitting-partition encoders."
            ),
            "search_scope": (
                "Known protocol, joint-TCP, ConfidNet and adapted-geometry artifact families; "
                "not an exhaustive filesystem search."
            ),
        }
        candidates = []
        for family in ("confidnet_oof_v1", "confidnet_oof_v1_replay_v2", "confidnet_initial_v1"):
            for seed in range(1, 5):
                directory = self.repo / ANALYSIS_ROOT / family / f"features/seed_{seed}"
                if not (directory / "manifest.json").is_file():
                    continue
                manifest = read_json(directory / "manifest.json")
                candidates.append({
                    "directory": str(directory), "seed": seed,
                    "recorded_role": manifest["role"],
                    "producer_names": list(
                        manifest.get("metadata", {}).get("producer_manifests", {})
                    ),
                    "payload_opened": False,
                    "reason": (
                        "256-dimensional expert-head hidden activations, not 1152-dimensional "
                        "image embeddings. OOF variants also mix fold encoders for training."
                    ),
                })
        self.report["confidnet_candidates_excluded"] = candidates

    def run(self) -> dict[str, Any]:
        self.load_protocol()
        self.audit_assignments()
        self.audit_geo_and_images()
        self.report["encoders"] = []
        self.report["validation_image_caches"] = []
        for seed in range(1, 5):
            producer = self.audit_encoder(seed)
            self.audit_validation_cache(seed, producer)
        self.inventory_excluded_caches()
        reject_failed_checks(self.report["checks"])
        self.report["status"] = "passed"
        self.report["summary"] = {
            "verified_encoder_seeds": [1, 2, 3, 4],
            "verified_validation_cache_seeds": [1, 2, 3, 4],
            "development_rows": 4200, "development_plots": 1625,
            "training_cache_regeneration_required": True,
            "test_features_or_labels_opened": False,
        }
        return self.report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_ROOT)
    parser.add_argument("--source-protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    protocol = args.source_protocol_root
    if not protocol.is_absolute():
        protocol = repo / protocol
    audit = InputAudit(repo, protocol)
    try:
        audit.run()
        exit_code = 0
    except Exception as exc:
        audit.report["status"] = "failed"
        audit.report["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit.report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": audit.report["status"], "output": str(output),
        "checks": len(audit.report["checks"]),
        "summary": audit.report.get("summary"), "error": audit.report.get("error"),
    }))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

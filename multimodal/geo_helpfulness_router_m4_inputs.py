"""Apply-only, label-blind expert/comparator inputs for the final M4 scorer."""

from __future__ import annotations

from pathlib import Path

# Keep the established native initialization order; this phase never fits.
from multimodal.geo_helpfulness_router_validation_inputs import (
    load_training_inputs,
    load_validation_inputs,
)
from multimodal.geo_helpfulness_router_numeric import apply_expert_temperature
from multimodal.geo_helpfulness_targets_features import build_router_feature_frame
from multimodal.geo_helpfulness_router_dataset import audit_schema
from multimodal.geo_helpfulness_router_preview_score import validate_sealed_predictions
from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    COMPARATOR_DIR,
    CONVENTIONS_PATH,
    canonical_hash,
    check_snapshot,
    code_snapshot,
    compare_tables,
    exclusive_lock,
    file_hash,
    publish_manifest,
    read_json,
    table_bytes,
    validate_manifest,
    write_bytes,
    write_json,
    load_conventions,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "geo_helpfulness.m4_comparator_manifest.v1"
TABLE_FILENAME = "expert_probabilities_and_features.parquet"
FILES = {"declaration.json", TABLE_FILENAME, "manifest.json"}


def comparator_schema() -> pa.Schema:
    semantic_names = list(
        build_router_feature_frame(
            *[np.full((1, 18), 1 / 18, dtype=np.float64)] * 3,
            probability_basis="scalar_temperature_calibrated",
        ).columns
    )
    parent = audit_schema()
    fields = [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("protocol_id", pa.string(), nullable=False),
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("plot_idx", pa.string(), nullable=False),
        pa.field("training_seed", pa.int8(), nullable=False),
    ]
    fields += [
        pa.field(f"{prefix}_prob_{basis}", pa.list_(pa.float64(), 18), nullable=False)
        for basis in ("calibrated", "native_t1")
        for prefix in ("image", "geo", "raw")
    ]
    return pa.schema(fields + [parent.field(name) for name in semantic_names])


def derive_comparator_table(inputs: dict, outputs: pd.DataFrame) -> pa.Table:
    frames = []
    for seed in (1, 2, 3, 4):
        rows = (
            outputs.loc[outputs.training_seed.eq(seed)]
            .sort_values("row_uid")
            .reset_index(drop=True)
        )
        probabilities = {}
        for mode, prefix in (
            ("image_only", "image"),
            ("geo_only", "geo"),
            ("raw_concat", "raw"),
        ):
            logits = np.stack(rows[f"{prefix}_logits"])
            probabilities[prefix] = apply_expert_temperature(
                logits, inputs["temperatures"][(seed, mode)]
            )
            if not np.array_equal(
                probabilities[prefix].argmax(axis=1), rows[f"{prefix}_pred"].to_numpy()
            ):
                raise ValueError("calibrated expert prediction differs from sealed M2")
        semantic = build_router_feature_frame(
            probabilities["image"],
            probabilities["geo"],
            probabilities["raw"],
            probability_basis="scalar_temperature_calibrated",
        )
        identity = rows[["protocol_id", "row_uid", "plot_idx", "training_seed"]].copy()
        identity.insert(0, "schema_version", "geo_helpfulness.m4_comparator_inputs.v1")
        for prefix in ("image", "geo", "raw"):
            identity[f"{prefix}_prob_calibrated"] = list(probabilities[prefix])
        for prefix in ("image", "geo", "raw"):
            identity[f"{prefix}_prob_native_t1"] = list(
                np.stack(rows[f"{prefix}_prob_native_t1"])
            )
        frames.append(pd.concat([identity, semantic], axis=1))
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["row_uid", "training_seed"])
        .reset_index(drop=True)
    )
    if len(frame) != 3288 or frame.isna().any().any():
        raise ValueError("comparator table count/null mismatch")
    return pa.Table.from_pandas(
        frame, schema=comparator_schema(), preserve_index=False
    ).replace_schema_metadata(None)


def prepare_comparators(protocol_root: Path, *, repo_root: Path = PROJECT_ROOT) -> dict:
    protocol_root, repo_root = Path(protocol_root).resolve(), Path(repo_root).resolve()
    root = protocol_root / COMPARATOR_DIR
    with exclusive_lock(root):
        if root.is_symlink():
            raise ValueError("comparator root cannot be a symlink")
        root.mkdir(exist_ok=True)
        if not {path.name for path in root.iterdir()}.issubset(FILES):
            raise ValueError("unknown comparator files; no automatic removal")
        committed = (root / "manifest.json").exists()
        if committed:
            validate_manifest(root, schema_version=SCHEMA_VERSION)
        elif (root / TABLE_FILENAME).exists() and not (
            root / "declaration.json"
        ).is_file():
            raise ValueError("partial comparator table lacks its prior declaration")
        old_root = protocol_root / "router/router_validation_v1"
        old = validate_sealed_predictions(old_root)
        inputs = load_training_inputs(repo_root, protocol_root)
        snapshot = inputs["parent_file_hashes"]
        feature_schema = (
            protocol_root
            / "router/targets_and_feature_contract/router_feature_schema.json"
        )
        expected_schema_hash = inputs["dataset_manifest"]["parent_lineage"][
            "m3_bundle_file_sha256"
        ]["router_feature_schema.json"]
        if file_hash(feature_schema) != expected_schema_hash:
            raise ValueError("frozen M3 feature schema changed")
        snapshot[str(feature_schema)] = expected_schema_hash
        snapshot.update(code_snapshot(repo_root))
        for path in old_root.iterdir():
            snapshot[str(path)] = file_hash(path)
        for name in (
            "router_validation_preview_v2.json",
            "router_validation_preview_v2_declaration.json",
        ):
            path = protocol_root / "router" / name
            snapshot[str(path)] = file_hash(path)
        conventions = load_conventions(repo_root)
        declaration = {
            "schema_version": "geo_helpfulness.m4_scoring_declaration.v1",
            "protocol_id": "protocol_v1",
            "prediction_manifest_sha256": old["manifest_sha256"],
            "dataset_manifest_sha256": inputs["dataset_manifest"]["manifest_sha256"],
            "candidate_manifest_sha256": inputs["candidate_manifest"][
                "manifest_sha256"
            ],
            "conventions": conventions,
            "conventions_sha256": canonical_hash(conventions),
            "code_file_sha256": {
                str(Path(name).relative_to(repo_root)): digest
                for name, digest in code_snapshot(repo_root).items()
            },
            "prior_preview": {
                "candidate": "primary_full727__seed_1__C_0p01",
                "accuracy": 585 / 822,
                "metric_was_predeclared": True,
                "training_feature_or_policy_settings_changed_after_preview": False,
            },
            "access_boundary": "comparator_export_reads_no_assignment_columns_and_calls_no_fitting_APIs",
        }
        write_json(root / "declaration.json", declaration)
        outputs = load_validation_inputs(repo_root, protocol_root, inputs)
        table = derive_comparator_table(inputs, outputs)
        check_snapshot(snapshot)
        if (root / TABLE_FILENAME).exists():
            compare_tables(pq.read_table(root / TABLE_FILENAME), table)
        elif committed:
            raise ValueError("committed comparator table is missing")
        else:
            write_bytes(root / TABLE_FILENAME, table_bytes(table))
        manifest = publish_manifest(
            root,
            schema_version=SCHEMA_VERSION,
            snapshot=snapshot,
            extra={
                "prediction_manifest_sha256": old["manifest_sha256"],
                "assignment_file_sha256": inputs["assignment_file_sha256"],
                "row_count": 3288,
                "rows_per_seed": 822,
                "plots_per_seed": 325,
                "declaration_sha256": canonical_hash(declaration),
                "schema": str(table.schema),
                "validation_labels_read": False,
                "temperature_transform_router_fits": 0,
            },
        )
        return {
            "status": "reused_valid" if committed else "built",
            "manifest_sha256": manifest["manifest_sha256"],
            "row_count": 3288,
            "validation_labels_read": False,
        }

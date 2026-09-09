"""Authenticated inputs for the pre-label M4 stages, without full preflight.

Training access is restricted to sealed OOF audit/target views. Validation access
is a separate call and consumes only sealed M2 prediction tables and manifests.
No assignment columns, expert metrics, or checkpoints are parsed here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Preserve the established native-library initialization order; no validators
# that discover datasets or read assignments are invoked.
from multimodal import geo_helpfulness_oof as m2
from multimodal import geo_helpfulness_targets_features as m3
from multimodal.geo_helpfulness_protocol import canonical_sha256, plot_set_fingerprint
from multimodal.geo_helpfulness_router_candidate_state import (
    full_feature_columns,
    load_candidate_set_for_inference,
)
from multimodal.geo_helpfulness_router_numeric import (
    validate_feature_transform_state,
    validate_temperature_state,
)


TRAIN_ROWS = 3378
TRAIN_PLOTS = 1300
VALIDATION_ROWS = 822
VALIDATION_PLOTS = 325
SEEDS = (1, 2, 3, 4)
MODES = ("image_only", "geo_only", "raw_concat")
TARGET_COLUMNS = list(m3.TARGET_COLUMNS)
IDENTITY_COLUMNS = TARGET_COLUMNS[:-1]
DATASET_CHILDREN = (
    "router_dataset.parquet",
    "router_dataset_audit.parquet",
    "expert_temperatures.json",
    "router_feature_transform.json",
)
M2_CODE = {
    "multimodal/geo_helpfulness_oof.py",
    "multimodal/geo_helpfulness_oof_report.py",
    "multimodal/geo_helpfulness_protocol.py",
    "multimodal/models.py",
    "tools/run_multimodal_geo_helpfulness_m2.py",
}
M3_CODE = {
    "multimodal/geo_helpfulness_targets_features.py",
    "tools/run_multimodal_geo_helpfulness_m3.py",
}
M4_CODE = {
    "multimodal/geo_helpfulness_router_dataset.py",
    "multimodal/geo_helpfulness_router_numeric.py",
    "tools/run_multimodal_geo_helpfulness_m4_dataset.py",
}
CANDIDATE_CODE = {
    "multimodal/geo_helpfulness_router_candidates.py",
    "multimodal/geo_helpfulness_router_candidate_fit.py",
    "multimodal/geo_helpfulness_router_candidate_state.py",
    "tools/run_multimodal_geo_helpfulness_m4_candidates.py",
}


class ValidationInputError(ValueError):
    """A sealed prediction input or its lineage is invalid."""


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValidationInputError(f"expected regular nonsymlink file: {path}")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValidationInputError(f"symlink directory in input path: {path}")


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValidationInputError("expected lowercase SHA256")
    return value


def _file_hash(path: Path) -> str:
    _regular_file(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _track(path: Path, expected: str, snapshot: dict[str, str]) -> None:
    if _file_hash(path) != _digest(expected):
        raise ValidationInputError(f"file hash mismatch: {path}")
    previous = snapshot.setdefault(str(path.absolute()), expected)
    if previous != expected:
        raise ValidationInputError(f"mixed parent identities: {path}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationInputError(f"nonfinite JSON constant: {value}")


def _json(path: Path) -> dict[str, Any]:
    _regular_file(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationInputError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationInputError(f"expected JSON object: {path}")
    return value


def _manifest(
    path: Path, snapshot: dict[str, str], key: str = "manifest_sha256"
) -> dict[str, Any]:
    # Snapshot before parsing so a changed file cannot become a new parent merely
    # because it changed between interpretation and subsequent hash capture.
    _track(path, _file_hash(path), snapshot)
    value = _json(path)
    _self_hash(value, key)
    return value


def _self_hash(value: dict[str, Any], key: str = "manifest_sha256") -> None:
    if _digest(value.get(key)) != canonical_sha256(
        {name: item for name, item in value.items() if name != key}
    ):
        raise ValidationInputError(f"invalid {key}")


def _check_snapshot(snapshot: dict[str, str]) -> None:
    for filename, expected in snapshot.items():
        if _file_hash(Path(filename)) != expected:
            raise ValidationInputError(f"parent changed while reading: {filename}")


def _code_files(
    repo_root: Path,
    hashes: dict[str, str],
    expected_names: set[str],
    snapshot: dict[str, str],
    aggregate: str | None = None,
) -> None:
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        raise ValidationInputError("frozen implementation file allowlist mismatch")
    if aggregate is not None and canonical_sha256(hashes) != _digest(aggregate):
        raise ValidationInputError("implementation aggregate hash mismatch")
    for name, expected in hashes.items():
        _track(repo_root / name, expected, snapshot)


def _schema_record(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": f.name, "dtype": str(f.type), "nullable": f.nullable} for f in schema
    ]


def _validate_training_frames(audit: pd.DataFrame, targets: pd.DataFrame) -> None:
    expected_audit = (
        IDENTITY_COLUMNS
        + ["train_oof_fold"]
        + [f"{prefix}_prob_calibrated" for prefix in ("image", "geo", "raw")]
        + list(m3.FEATURE_COLUMNS)
    )
    if list(audit) != expected_audit or list(targets) != TARGET_COLUMNS:
        raise ValidationInputError("OOF audit/target column order mismatch")
    if len(audit) != TRAIN_ROWS * 4 or len(targets) != len(audit):
        raise ValidationInputError("OOF audit/target row counts mismatch")
    for frame in (audit, targets):
        if frame[IDENTITY_COLUMNS].isna().any().any():
            raise ValidationInputError("missing OOF identity")
        if set(frame.protocol_id) != {"protocol_v1"}:
            raise ValidationInputError("mixed OOF protocol identity")
        keys = list(zip(frame.row_uid, frame.training_seed))
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValidationInputError("OOF keys are not unique and canonical")
        if set(frame.training_seed) != set(SEEDS):
            raise ValidationInputError("OOF seeds mismatch")
        for _, rows in frame.groupby("training_seed"):
            if len(rows) != TRAIN_ROWS or rows.plot_idx.nunique() != TRAIN_PLOTS:
                raise ValidationInputError("OOF seed row/plot counts mismatch")
    if not audit[IDENTITY_COLUMNS].equals(targets[IDENTITY_COLUMNS]):
        raise ValidationInputError("OOF audit/target identities differ")
    if set(targets.schema_version) != {m3.TARGET_SCHEMA_VERSION}:
        raise ValidationInputError("OOF target schema version mismatch")
    if targets.target_state.isna().any() or not set(targets.target_state).issubset(
        m3.TARGET_ORDER
    ):
        raise ValidationInputError("invalid four-state training target")
    if not set(audit.train_oof_fold).issubset({0, 1, 2, 3}):
        raise ValidationInputError("invalid OOF fold")
    for seed in SEEDS:
        rows = audit.loc[audit.training_seed.eq(seed)]
        probabilities = [
            np.stack(rows[f"{prefix}_prob_calibrated"])
            for prefix in ("image", "geo", "raw")
        ]
        semantic = m3.build_router_feature_frame(
            *probabilities, probability_basis=m3.CALIBRATED_PROBABILITY_BASIS
        )
        observed = rows.loc[:, list(m3.FEATURE_COLUMNS)].reset_index(drop=True)
        for name in m3.FEATURE_COLUMNS:
            if name in m3.NUMERIC_FEATURES:
                if not np.allclose(
                    observed[name], semantic[name], atol=1e-12, rtol=1e-12
                ):
                    raise ValidationInputError(
                        f"OOF semantic reconstruction mismatch: {name}"
                    )
            elif not np.array_equal(observed[name], semantic[name]):
                raise ValidationInputError(
                    f"OOF semantic reconstruction mismatch: {name}"
                )


def load_training_inputs(repo_root: Path, protocol_root: Path) -> dict[str, Any]:
    """Authenticate OOF-only prerequisites; never read validation outputs."""
    repo_root, protocol_root = (
        Path(repo_root).absolute(),
        Path(protocol_root).absolute(),
    )
    router = protocol_root / "router"
    snapshot: dict[str, str] = {}
    manifest_path = router / "router_dataset_manifest.json"
    manifest = _manifest(manifest_path, snapshot)
    if (
        manifest.get("schema_version") != "geo_helpfulness.router_dataset_manifest.v1"
        or manifest.get("artifact_role") != "router_training_dataset"
        or manifest.get("protocol_id") != "protocol_v1"
        or manifest.get("feature_columns") != full_feature_columns()
        or set(manifest.get("artifacts", {})) != set(DATASET_CHILDREN)
    ):
        raise ValidationInputError("router dataset manifest contract mismatch")
    for name in DATASET_CHILDREN:
        _track(router / name, manifest["artifacts"][name]["file_sha256"], snapshot)
    lineage = manifest["parent_lineage"]
    _code_files(
        repo_root,
        lineage["m4_code_file_sha256"],
        M4_CODE,
        snapshot,
        lineage["m4_code_sha256"],
    )
    _track(
        repo_root / "configs/multimodal_geo_helpfulness.yaml",
        lineage["config_file_sha256"],
        snapshot,
    )
    parents = lineage["m1_m2_parent_artifact_hashes"]
    assignment_hash = parents["development_assignments"]["file_sha256"]
    _track(
        protocol_root / "protocol/development_assignments.parquet",
        assignment_hash,
        snapshot,
    )
    for name, parent_name in (
        ("protocol_manifest.json", "protocol_manifest"),
        ("resolved_protocol.yaml", "resolved_protocol"),
    ):
        _track(
            protocol_root / "protocol" / name,
            parents[parent_name]["file_sha256"],
            snapshot,
        )
    m3_path = router / "targets_and_feature_contract/manifest.json"
    _track(m3_path, lineage["m3_bundle_file_sha256"]["manifest.json"], snapshot)
    m3_manifest = _json(m3_path)
    _self_hash(m3_manifest)
    if m3_manifest["manifest_sha256"] != lineage["m3_manifest_sha256"]:
        raise ValidationInputError("M3 manifest lineage mismatch")
    for family, names in (("m2", M2_CODE), ("m3", M3_CODE)):
        if m3_manifest[f"{family}_code_sha256"] != lineage[f"{family}_code_sha256"]:
            raise ValidationInputError(f"{family} implementation lineage mismatch")
        _code_files(
            repo_root,
            m3_manifest[f"{family}_code_file_sha256"],
            names,
            snapshot,
            lineage[f"{family}_code_sha256"],
        )
    containers = {}
    for name, schema in (
        ("expert_temperatures.json", "geo_helpfulness.expert_temperatures.v1"),
        (
            "router_feature_transform.json",
            "geo_helpfulness.router_feature_transform.v1",
        ),
    ):
        value = _json(router / name)
        if canonical_sha256(value) != manifest["artifacts"][name]["content_sha256"]:
            raise ValidationInputError(f"JSON content hash mismatch: {name}")
        if (
            set(value) != {"schema_version", "protocol_id", "fit_role", "states"}
            or value["schema_version"] != schema
            or value["protocol_id"] != "protocol_v1"
            or value["fit_role"] != "development_train_oof"
            or not isinstance(value["states"], list)
        ):
            raise ValidationInputError(f"state container mismatch: {name}")
        containers[name] = value["states"]
    temperatures = {}
    for state in containers["expert_temperatures.json"]:
        validate_temperature_state(state)
        key = (state["seed"], state["mode"])
        if key in temperatures:
            raise ValidationInputError("duplicate temperature state")
        temperatures[key] = state
    transforms = {}
    for state in containers["router_feature_transform.json"]:
        validate_feature_transform_state(state)
        if state["seed"] in transforms:
            raise ValidationInputError("duplicate feature transform")
        transforms[state["seed"]] = state
    if set(temperatures) != {(s, m) for s in SEEDS for m in MODES} or set(
        transforms
    ) != set(SEEDS):
        raise ValidationInputError("temperature/transform grid incomplete")

    candidates_root = router / "router_candidates_v2"
    candidate_manifest_path = candidates_root / "manifest.json"
    _track(candidate_manifest_path, _file_hash(candidate_manifest_path), snapshot)
    study = _manifest(
        candidates_root / "study_spec.json", snapshot, "study_spec_sha256"
    )
    candidate_manifest, states = load_candidate_set_for_inference(
        candidates_root,
        expected_dataset_manifest_sha256=manifest["manifest_sha256"],
        expected_study_spec_sha256=study["study_spec_sha256"],
    )
    if study.get("schema_version") != "geo_helpfulness.router_candidate_study.v2":
        raise ValidationInputError("expected approved candidate study v2")
    _code_files(
        repo_root, study["implementation_file_sha256"], CANDIDATE_CODE, snapshot
    )
    config_path = repo_root / "configs/multimodal_geo_router_candidates_v2.json"
    _track(config_path, study["source_study_config_sha256"], snapshot)
    if _json(config_path) != study["configuration"]:
        raise ValidationInputError("candidate study configuration mismatch")
    if study["parent_config_sha256"] != lineage["config_file_sha256"]:
        raise ValidationInputError("candidate configuration parent mismatch")
    expected_dataset_files = {
        **{n: manifest["artifacts"][n]["file_sha256"] for n in DATASET_CHILDREN},
        "router_dataset_manifest.json": snapshot[str(manifest_path)],
    }
    if study["dataset_artifact_file_sha256"] != expected_dataset_files:
        raise ValidationInputError("candidate/M4 physical parent hashes differ")
    for name in ("study_spec.json", "training_report.json"):
        _track(
            candidates_root / name,
            candidate_manifest["artifacts"][name]["file_sha256"],
            snapshot,
        )
    for name in states:
        _track(
            candidates_root / name,
            candidate_manifest["candidates"][name]["file_sha256"],
            snapshot,
        )

    for name in ("router_dataset.parquet", "router_dataset_audit.parquet"):
        schema = pq.read_schema(router / name)
        if _schema_record(schema) != manifest["schemas"][name] or any(
            f.nullable for f in schema
        ):
            raise ValidationInputError(f"M4 Arrow schema mismatch: {name}")
    targets = pq.read_table(
        router / "router_dataset.parquet", columns=TARGET_COLUMNS
    ).to_pandas()
    audit = pq.read_table(router / "router_dataset_audit.parquet").to_pandas()
    _validate_training_frames(audit, targets)
    for seed in SEEDS:
        rows = targets.loc[targets.training_seed.eq(seed), ["row_uid", "training_seed"]]
        identity_hash = canonical_sha256(rows.to_dict(orient="records"))
        if study["seed_inputs"][str(seed)] != {
            "row_count": TRAIN_ROWS,
            "plot_count": TRAIN_PLOTS,
            "row_identity_sha256": identity_hash,
        }:
            raise ValidationInputError("candidate study OOF row identity mismatch")
        for state in [
            transforms[seed],
            *(temperatures[(seed, mode)] for mode in MODES),
        ]:
            if (
                state["fit_row_count"] != TRAIN_ROWS
                or state["fit_row_identity_sha256"] != identity_hash
            ):
                raise ValidationInputError(
                    "preprocessing/candidate fit identity mismatch"
                )
    _check_snapshot(snapshot)
    return {
        "dataset_manifest": manifest,
        "candidate_manifest": candidate_manifest,
        "candidate_states": states,
        "candidate_study": study,
        "temperatures": temperatures,
        "transforms": transforms,
        "audit": audit,
        "targets": targets,
        "parent_file_hashes": snapshot,
        "assignment_file_sha256": assignment_hash,
    }


def _m2_parent_binding(value: dict[str, Any], inputs: dict[str, Any]) -> None:
    lineage = inputs["dataset_manifest"]["parent_lineage"]
    for parent, keys in {
        "development_assignments": ("file_sha256", "content_sha256"),
        "protocol_manifest": ("file_sha256", "payload_sha256"),
        "resolved_protocol": ("file_sha256",),
    }.items():
        for key in keys:
            if (
                value["parent_artifact_hashes"][parent][key]
                != lineage["m1_m2_parent_artifact_hashes"][parent][key]
            ):
                raise ValidationInputError("M2/M4 parent identity mismatch")
    if value["m2_code_sha256"] != lineage["m2_code_sha256"]:
        raise ValidationInputError("M2 implementation lineage mismatch")


def load_validation_inputs(
    repo_root: Path, protocol_root: Path, training_inputs: dict[str, Any]
) -> pd.DataFrame:
    """Authenticate only M2 prediction inputs; never parse assignment labels."""
    repo_root, protocol_root = (
        Path(repo_root).absolute(),
        Path(protocol_root).absolute(),
    )
    snapshot = dict(training_inputs["parent_file_hashes"])
    _check_snapshot(snapshot)
    directory = protocol_root / "development_validation"
    aggregate_path = directory / "aggregate_manifest.json"
    aggregate = _manifest(aggregate_path, snapshot)
    _m2_parent_binding(aggregate, training_inputs)
    _code_files(
        repo_root,
        aggregate["m2_code_file_sha256"],
        M2_CODE,
        snapshot,
        aggregate["m2_code_sha256"],
    )
    declarations = {
        "schema_version": "geo_helpfulness.artifact_manifest.v1",
        "protocol_id": "protocol_v1",
        "artifact_role": "development_validation_outputs",
        "parent_roles": ["development_assignments", "development_train_expert_fit"],
        "training_seeds": list(SEEDS),
        "row_count": VALIDATION_ROWS * 4,
        "unique_key": ["row_uid", "training_seed"],
        "canonical_sort": ["row_uid", "training_seed"],
        "output_columns": list(m2.VALIDATION_OUTPUT_COLUMNS),
        "model_output_filename": "development_validation_model_outputs.parquet",
        "aggregation": "concatenation_without_averaging_or_pooling",
        "probability_basis": "native_t1_uncalibrated",
        "logits_authoritative": True,
    }
    if any(aggregate.get(key) != value for key, value in declarations.items()):
        raise ValidationInputError("M2 validation aggregate declarations mismatch")
    expected_refs = {f"development_validation/seed_{seed}" for seed in SEEDS}
    if set(aggregate.get("producer_manifest_sha256", {})) != expected_refs:
        raise ValidationInputError("M2 validation producer set mismatch")
    tables, common_identity = [], None
    training_plots = set(training_inputs["audit"].plot_idx)
    for seed in SEEDS:
        producer_dir = directory / f"seed_{seed}"
        manifest_path = producer_dir / "manifest.json"
        producer = _manifest(manifest_path, snapshot)
        if (
            producer["manifest_sha256"]
            != aggregate["producer_manifest_sha256"][
                f"development_validation/seed_{seed}"
            ]
        ):
            raise ValidationInputError("producer manifest reference mismatch")
        _m2_parent_binding(producer, training_inputs)
        if producer["m2_code_file_sha256"] != aggregate["m2_code_file_sha256"]:
            raise ValidationInputError("mixed M2 producer implementations")
        expected = {
            key: declarations[key]
            for key in (
                "schema_version",
                "protocol_id",
                "artifact_role",
                "parent_roles",
                "output_columns",
                "model_output_filename",
                "probability_basis",
                "logits_authoritative",
            )
        }
        expected.update(
            stage_id="development_train_to_validation",
            train_oof_fold=None,
            training_seed=seed,
            row_count=VALIDATION_ROWS,
            plot_count=VALIDATION_PLOTS,
            fitting_row_count=TRAIN_ROWS,
            zero_plot_overlap=True,
            temperature=1.0,
            calibrated=False,
            probability_integrity_atol=1e-8,
        )
        if any(producer.get(key) != value for key, value in expected.items()):
            raise ValidationInputError("M2 validation producer declarations mismatch")
        if (
            producer["class_map_hash"]
            != training_inputs["dataset_manifest"]["parent_lineage"][
                "m1_m2_parent_artifact_hashes"
            ]["class_map"]["content_sha256"]
        ):
            raise ValidationInputError("M2 validation class map mismatch")
        output_path = producer_dir / declarations["model_output_filename"]
        _track(output_path, producer["model_output_file_sha256"], snapshot)
        if (
            producer["component_file_sha256"][output_path.name]
            != producer["model_output_file_sha256"]
        ):
            raise ValidationInputError("M2 producer component/output hash mismatch")
        table = m2.read_output_parquet(output_path, include_fold=False)
        checked = m2.validate_output_table(
            table, include_fold=False, expected_rows=VALIDATION_ROWS
        )
        # Parquet may rename the list child from ``item`` to ``element``.
        # M2's frozen manifest hashes its pre-write canonical Arrow schema.
        canonical_schema = pa.schema(
            [
                pa.field(
                    field.name,
                    (
                        pa.list_(pa.float64(), 18)
                        if pa.types.is_fixed_size_list(field.type)
                        else field.type
                    ),
                    nullable=False,
                )
                for field in table.schema
            ]
        )
        expected_schema_hash = canonical_sha256(
            {
                "geo_columns": list(m2.GEO_COLUMNS),
                "image_embedding_dim": 1152,
                "raw_concat_order": ["image_embedding", "standardized_geo"],
                "class_count": 18,
                "output_columns": list(m2.VALIDATION_OUTPUT_COLUMNS),
                "arrow_schema": str(canonical_schema),
            }
        )
        if (
            checked["logical_table_sha256"] != producer["content_sha256"]
            or checked["plot_count"] != VALIDATION_PLOTS
            or str(canonical_schema) != producer["output_arrow_schema"]
            or producer["feature_schema_hash"] != expected_schema_hash
        ):
            raise ValidationInputError("M2 producer table fingerprint/schema mismatch")
        if set(table["training_seed"].to_pylist()) != {seed}:
            raise ValidationInputError("M2 producer contains another seed")
        if set(table["schema_version"].to_pylist()) != {
            "geo_helpfulness_protocol_config_v1"
        } or set(table["protocol_id"].to_pylist()) != {"protocol_v1"}:
            raise ValidationInputError("M2 validation row identity scope mismatch")
        identity = table.select(
            ["row_uid", "file", "file_lower", "plot_idx"]
        ).to_pandas()
        if common_identity is not None and not identity.equals(common_identity):
            raise ValidationInputError("validation identities differ across seeds")
        common_identity = identity
        validation_plots = set(identity.plot_idx)
        if (
            training_plots & validation_plots
            or plot_set_fingerprint(validation_plots)
            != producer["prediction_plot_hash"]
            or plot_set_fingerprint(training_plots) != producer["fitting_plot_hash"]
        ):
            raise ValidationInputError("M2 fitting/prediction plot provenance mismatch")
        if producer["feature_schema_hash"] != aggregate["feature_schema_hash"]:
            raise ValidationInputError("mixed M2 output schema fingerprints")
        tables.append(table)
    aggregate_output = directory / declarations["model_output_filename"]
    _track(aggregate_output, aggregate["model_output_file_sha256"], snapshot)
    table = m2.read_output_parquet(aggregate_output, include_fold=False)
    checked = m2.validate_output_table(
        table, include_fold=False, expected_rows=VALIDATION_ROWS * 4
    )
    concatenated = pa.concat_tables(tables).sort_by(
        [("row_uid", "ascending"), ("training_seed", "ascending")]
    )
    if checked["logical_table_sha256"] != aggregate[
        "content_sha256"
    ] or not table.equals(concatenated):
        raise ValidationInputError(
            "validation aggregate does not match all producer outputs"
        )
    _check_snapshot(snapshot)
    training_inputs["parent_file_hashes"].update(snapshot)
    return table.to_pandas()

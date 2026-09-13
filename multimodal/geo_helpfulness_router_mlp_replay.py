"""Additive, apply-only recovery of the sealed MLP history column ordering.

The original declaration, code, states, predictions, and curve bytes remain
unchanged. Only reconstruction column order is corrected, in a scoped override.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from multimodal import geo_helpfulness_router_mlp as source
from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    canonical_hash,
    check_self_hash,
    check_snapshot,
    compare_tables,
    exclusive_lock,
    file_hash,
    read_json,
    table_bytes,
    write_bytes,
    write_json,
)

STUDY_ID = source.STUDY_ID
RECOVERY_ID = "router_mlp_v1_replay_v2"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/router_mlp_v1"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent / RECOVERY_ID
CURVE_COLUMNS = (
    "config_id",
    "variant",
    "training_seed",
    "epoch",
    "optimization_cross_entropy",
    "train_nll",
    "train_accuracy",
    "maximum_gradient_l2_norm",
    "gradients_finite",
)
JSON_CURVE_COLUMNS = (
    "config_id",
    "variant",
    "training_seed",
    "epoch",
    "gradients_finite",
    "maximum_gradient_l2_norm",
    "optimization_cross_entropy",
    "train_accuracy",
    "train_nll",
)
KNOWN_CURVE_ORDERS = (CURVE_COLUMNS, JSON_CURVE_COLUMNS)
CODE_FILES = (
    "multimodal/geo_helpfulness_router_mlp_replay.py",
    "tools/run_multimodal_geo_helpfulness_router_mlp_replay.py",
)
SCORE_FILES = source.SCORE_FILES
_ORIGINAL_CURVES = source._curves


def _seal(value, key):
    return {**value, key: canonical_hash(value)}


def _fixed_curves(records):
    table = _ORIGINAL_CURVES(records)
    if len(table.column_names) != len(CURVE_COLUMNS) or set(table.column_names) != set(
        CURVE_COLUMNS
    ):
        raise ValueError("recovery only permits the original nine curve columns")
    return table.select(CURVE_COLUMNS)


def _source_curve_order(source_root):
    columns = tuple(
        pq.ParquetFile(
            source_root / "predictions/training_curves.parquet"
        ).schema_arrow.names
    )
    if columns not in KNOWN_CURVE_ORDERS:
        raise ValueError("source curve column order is not a known original layout")
    return columns


@contextmanager
def _curve_order_override(columns=CURVE_COLUMNS):
    columns = tuple(columns)
    if columns not in KNOWN_CURVE_ORDERS:
        raise ValueError("recovery only accepts the two known original curve orders")
    if source._curves is not _ORIGINAL_CURVES:
        raise ValueError("MLP curve reconstruction was already overridden")
    source._curves = (
        _fixed_curves
        if columns == CURVE_COLUMNS
        else lambda records: _fixed_curves(records).select(columns)
    )
    try:
        yield
    finally:
        source._curves = _ORIGINAL_CURVES


def _source_predictions(source_root):
    source._fit_incapable()
    declaration = source.load_declaration(source_root)
    expected = {
        "declaration.json",
        "predictions/manifest.json",
        *[
            "predictions/" + name
            for name in source._prediction_files(declaration["configuration"])
        ],
    }
    source.capacity._allowlist(source_root, expected, complete=True)
    if len(expected) != 148 or len(declaration["parent_file_sha256"]) != 660:
        raise ValueError(
            "recovery requires the sealed 148-file, 660-parent MLP v1 bundle"
        )
    with _curve_order_override(_source_curve_order(source_root)):
        checked_declaration, manifest = source._validate_predictions(source_root)
    if checked_declaration != declaration:
        raise ValueError("source declaration changed during replay")
    source._fit_incapable()
    return declaration, manifest


def validate_predictions(source_root):
    source_root = Path(source_root).resolve()
    source._fit_incapable()
    with exclusive_lock(source_root):
        declaration, manifest = _source_predictions(source_root)
    return {
        "status": "validated",
        "candidate_seed_count": 48,
        "source_declaration_sha256": declaration["declaration_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "recovery_id": RECOVERY_ID,
        "fitting_performed": False,
        "source_files_changed": False,
    }


def _output_allowlist(root):
    source.capacity._allowlist(
        root,
        {
            "declaration.json",
            "scores/manifest.json",
            *["scores/" + name for name in SCORE_FILES],
        },
    )


def _declare(source_root, output_root, source_declaration, prediction):
    if source.capacity._files(output_root):
        raise ValueError("unowned recovery directory is not empty")
    snapshot = dict(source_declaration["parent_file_sha256"])
    for path in [
        *[source_root / name for name in source.capacity._files(source_root)],
        *[PROJECT_ROOT / name for name in CODE_FILES],
    ]:
        digest = file_hash(path)
        if str(path) in snapshot and snapshot[str(path)] != digest:
            raise ValueError("recovery parent snapshots disagree")
        snapshot[str(path)] = digest
    declaration = _seal(
        {
            "schema_version": "geo_helpfulness.router_mlp_replay_declaration.v2",
            "study_id": STUDY_ID,
            "recovery_id": RECOVERY_ID,
            "declared_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "source_declaration_sha256": source_declaration["declaration_sha256"],
            "source_prediction_manifest_sha256": prediction["manifest_sha256"],
            "source_file_count": 148,
            "original_parent_file_count": 660,
            "parent_file_sha256": snapshot,
            "correction": "explicit_known_original_nine_column_training_curve_order_only",
            "training_curve_column_order": list(_source_curve_order(source_root)),
            "accepted_original_training_curve_orders": [
                list(order) for order in KNOWN_CURVE_ORDERS
            ],
            "source_mutations_permitted": False,
            "fitting_permitted": False,
            "new_validation_targets_decoded_before_declaration": False,
            "scientific_selection_rules_changed": False,
        },
        "declaration_sha256",
    )
    check_snapshot(snapshot)
    write_json(output_root / "declaration.json", declaration)
    return declaration


def load_declaration(source_root, output_root):
    declaration = read_json(output_root / "declaration.json")
    check_self_hash(declaration, "declaration_sha256")
    if (
        declaration.get("schema_version")
        != "geo_helpfulness.router_mlp_replay_declaration.v2"
        or declaration.get("study_id") != STUDY_ID
        or declaration.get("recovery_id") != RECOVERY_ID
        or declaration.get("source_root") != str(source_root)
        or declaration.get("training_curve_column_order")
        != list(_source_curve_order(source_root))
        or declaration.get("accepted_original_training_curve_orders")
        != [list(order) for order in KNOWN_CURVE_ORDERS]
    ):
        raise ValueError("recovery declaration identity mismatch")
    check_snapshot(declaration["parent_file_sha256"])
    _output_allowlist(output_root)
    return declaration


def _payloads(source_root, recovery, source_declaration, prediction):
    check_snapshot(recovery["parent_file_sha256"])
    source._fit_incapable()
    payloads = source._score_payloads(source_root, source_declaration, prediction)
    lineage = {
        "recovery_id": RECOVERY_ID,
        "recovery_declaration_sha256": recovery["declaration_sha256"],
        "source_declaration_sha256": source_declaration["declaration_sha256"],
        "source_prediction_manifest_sha256": prediction["manifest_sha256"],
        "correction": recovery["correction"],
        "source_files_changed": False,
    }
    frozen = payloads["frozen_policies.json"]
    for group in frozen["groups"].values():
        for seed, model in group["models"].items():
            path = (
                source_root
                / "predictions/models"
                / f"{group['config_id']}__seed_{seed}.json"
            )
            if file_hash(path) != model["file_sha256"]:
                raise ValueError("frozen recovery model hash differs")
            model["path"] = str(path)
    frozen["recovery_lineage"] = dict(lineage)
    frozen["model_path_basis"] = "absolute_unchanged_source"
    payloads["access_audit.json"]["recovery_lineage"] = dict(lineage)
    receipt = payloads["selection_receipt.json"]
    receipt.pop("receipt_sha256")
    receipt["recovery_lineage"] = dict(lineage)
    receipt["frozen_policies_sha256"] = canonical_hash(frozen)
    payloads["selection_receipt.json"] = _seal(receipt, "receipt_sha256")
    if set(payloads) != SCORE_FILES:
        raise ValueError("recovery score output allowlist differs")
    check_snapshot(recovery["parent_file_sha256"])
    source._fit_incapable()
    return payloads


def _manifest(root, declaration, prediction):
    manifest = read_json(root / "manifest.json")
    check_self_hash(manifest)
    if (
        manifest.get("schema_version")
        != "geo_helpfulness.router_mlp_replay_score_manifest.v2"
        or manifest.get("declaration_sha256") != declaration["declaration_sha256"]
        or manifest.get("prediction_manifest_sha256") != prediction["manifest_sha256"]
        or set(manifest["artifacts"]) != SCORE_FILES
    ):
        raise ValueError("recovery score manifest identity mismatch")
    source.capacity._allowlist(root, {*SCORE_FILES, "manifest.json"}, complete=True)
    for name, record in manifest["artifacts"].items():
        path = root / name
        if (
            file_hash(path) != record["file_sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"recovery score artifact changed: {name}")
    return manifest


def _publish(root, declaration, prediction, payloads):
    check_snapshot(declaration["parent_file_sha256"])
    source.capacity._allowlist(root, SCORE_FILES, complete=True)
    artifacts = {}
    for name in sorted(SCORE_FILES):
        path = root / name
        record = {"file_sha256": file_hash(path), "bytes": path.stat().st_size}
        if name.endswith(".parquet"):
            parquet = pq.ParquetFile(path)
            record.update(
                rows=parquet.metadata.num_rows,
                columns=len(parquet.schema_arrow),
                arrow_schema=str(parquet.schema_arrow),
            )
        artifacts[name] = record
    manifest = _seal(
        {
            "schema_version": "geo_helpfulness.router_mlp_replay_score_manifest.v2",
            "study_id": STUDY_ID,
            "recovery_id": RECOVERY_ID,
            "declaration_sha256": declaration["declaration_sha256"],
            "source_declaration_sha256": declaration["source_declaration_sha256"],
            "prediction_manifest_sha256": prediction["manifest_sha256"],
            "selection_receipt_sha256": payloads["selection_receipt.json"][
                "receipt_sha256"
            ],
            "artifacts": artifacts,
        },
        "manifest_sha256",
    )
    write_json(root / "manifest.json", manifest)
    return manifest


def _scores(source_root, output_root, *, build):
    source._fit_incapable()
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    initial = source.load_declaration(source_root)
    protected = [
        source_root,
        Path(initial["protocol_root"]).resolve(),
        Path(initial["capacity_root"]).resolve(),
    ]
    if any(output_root == root or root in output_root.parents for root in protected):
        raise ValueError("recovery output must be outside all frozen source roots")
    with exclusive_lock(source_root), exclusive_lock(output_root):
        source_declaration, prediction = _source_predictions(source_root)
        if (output_root / "declaration.json").exists():
            declaration = load_declaration(source_root, output_root)
        elif build:
            declaration = _declare(
                source_root, output_root, source_declaration, prediction
            )
        else:
            raise ValueError("recovery declaration has not been published")
        if (
            declaration["source_declaration_sha256"]
            != source_declaration["declaration_sha256"]
            or declaration["source_prediction_manifest_sha256"]
            != prediction["manifest_sha256"]
        ):
            raise ValueError("recovery/source lineage mismatch")
        root = output_root / "scores"
        committed = (root / "manifest.json").exists()
        if committed:
            manifest = _manifest(root, declaration, prediction)
        elif not build:
            raise ValueError("recovery scores have not been committed")
        else:
            source.capacity._allowlist(root, SCORE_FILES)
        payloads = _payloads(source_root, declaration, source_declaration, prediction)
        if (
            committed
            and manifest["selection_receipt_sha256"]
            != payloads["selection_receipt.json"]["receipt_sha256"]
        ):
            raise ValueError("recovery receipt does not reproduce")
        for name, value in payloads.items():
            path = root / name
            if path.exists():
                if isinstance(value, pa.Table):
                    compare_tables(pq.read_table(path), value)
                elif read_json(path) != value:
                    raise ValueError(f"recovery score does not reproduce: {name}")
            elif build:
                (
                    write_bytes(path, table_bytes(value))
                    if isinstance(value, pa.Table)
                    else write_json(path, value)
                )
        if not committed:
            manifest = _publish(root, declaration, prediction, payloads)
        check_snapshot(declaration["parent_file_sha256"])
        source._fit_incapable()
    return {
        "status": ("reused_valid" if committed else "built") if build else "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "recovery_declaration_sha256": declaration["declaration_sha256"],
        "candidate_seed_count": 48,
        "group_count": 3,
        "fitting_performed": False,
        "source_files_changed": False,
        "cleaned_test_accessed": False,
    }


def build_scores(source_root, output_root):
    return _scores(source_root, output_root, build=True)


def validate_scores(source_root, output_root):
    return _scores(source_root, output_root, build=False)

"""Additive v2 receipt-path correction; all sealed v1 science stays unchanged.

The original scorer looked for the receipt schema under artifact_contract,
whereas the frozen configuration places it under schemas. This module preserves
every v1 result and source byte and corrects only that contract lookup.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from multimodal import geo_helpfulness_router_m4_analysis as v1
from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    RESULT_DIR,
    canonical_hash,
    check_self_hash,
    check_snapshot,
    compare_tables,
    exclusive_lock,
    file_hash,
    read_json,
    table_bytes,
    validate_manifest,
    write_bytes,
    write_json,
)

CODE_FILES = (
    "multimodal/geo_helpfulness_router_m4_finalize.py",
    "tools/run_multimodal_geo_helpfulness_m4_finalize.py",
)
DECLARATION = "router/m4_receipt_correction_v2.json"
REASON = "receipt_required_fields_live_under_schemas_not_artifact_contract"


def _bind_correction(protocol_root: Path, inputs: dict, declaration: dict) -> None:
    snapshot = dict(inputs["snapshot"])
    snapshot.update(
        {
            str(PROJECT_ROOT / name): digest
            for name, digest in declaration["code_file_sha256"].items()
        }
    )
    path = protocol_root / DECLARATION
    snapshot[str(path)] = file_hash(path)
    snapshot = dict(sorted(snapshot.items()))
    if "correction_snapshot" in inputs and inputs["correction_snapshot"] != snapshot:
        raise ValueError("receipt correction snapshot changed during operation")
    inputs["correction_snapshot"] = snapshot
    check_snapshot(snapshot)


def _correction(protocol_root: Path, inputs: dict, *, create: bool) -> dict:
    path = protocol_root / DECLARATION
    expected = {
        "schema_version": "geo_helpfulness.m4_receipt_correction.v2",
        "protocol_id": "protocol_v1",
        "reason": REASON,
        "correct_config_path": "schemas.development_selection_receipt.required_fields",
        "comparator_manifest_sha256": inputs["comparator_manifest"]["manifest_sha256"],
        "code_file_sha256": {
            name: file_hash(PROJECT_ROOT / name) for name in CODE_FILES
        },
        "original_parent_file_sha256": dict(sorted(inputs["snapshot"].items())),
        "scientific_settings_changed": False,
        "existing_selection_repeated": False,
        "existing_result_bytes_overwritten": False,
    }
    if path.exists():
        saved = read_json(path)
        check_self_hash(saved, "declaration_sha256")
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("receipt correction declaration/code/parents changed")
        root = protocol_root / RESULT_DIR
        for name, digest in saved["preexisting_result_file_sha256"].items():
            if file_hash(root / name) != digest:
                raise ValueError(f"pre-correction result changed: {name}")
        _bind_correction(protocol_root, inputs, saved)
        return saved
    if not create:
        raise ValueError("missing receipt correction declaration")
    root = protocol_root / RESULT_DIR
    v1._check_file_allowlist(root)
    if (root / "development_selection_receipt.json").exists():
        raise ValueError("cannot introduce correction after a selection receipt")
    expected["preexisting_result_file_sha256"] = {
        str(item.relative_to(root)): file_hash(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }
    expected["declaration_sha256"] = canonical_hash(expected)
    write_json(path, expected)
    _bind_correction(protocol_root, inputs, expected)
    return expected


def _snapshot(inputs: dict, protocol_root: Path) -> dict:
    del protocol_root
    return dict(inputs["correction_snapshot"])


def _receipt(inputs, targets, selection, evaluated, root, frozen_hashes):
    """Only adapt the schema lookup; the original receipt values are unchanged."""
    corrected = dict(inputs)
    config = deepcopy(inputs["config"])
    config["artifact_contract"]["development_selection_receipt"] = config["schemas"][
        "development_selection_receipt"
    ]
    corrected["config"] = config
    receipt = v1._receipt(corrected, targets, selection, evaluated, root, frozen_hashes)
    receipt.pop("receipt_sha256")
    receipt["parent_file_sha256"] = _snapshot(inputs, inputs["protocol_root"])
    receipt["receipt_implementation_correction"] = {
        "reason": REASON,
        "declaration_file_sha256": inputs["correction_snapshot"][
            str(inputs["protocol_root"] / DECLARATION)
        ],
        "scientific_settings_changed": False,
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    return receipt


def _reproduce(inputs: dict, root: Path):
    table = pq.read_table(root / "validation_targets.parquet")
    if table.schema != v1._target_schema():
        raise ValueError("sealed scoring target physical schema mismatch")
    targets = table.to_pandas()
    v1._validate_targets(targets, inputs)
    if read_json(root / "build_declaration.json") != v1._build_declaration(
        inputs, targets
    ):
        raise ValueError("build declaration differs from inputs/targets")
    selection = read_json(root / "selection.json")
    evaluated = v1._evaluate(inputs, targets, selection)
    v1._validate_saved_selection(
        selection, inputs, targets, evaluated["candidate_metrics.json"]
    )
    for name, expected in evaluated.items():
        if isinstance(expected, pa.Table):
            compare_tables(pq.read_table(root / name), expected)
        elif read_json(root / name) != expected:
            raise ValueError(f"scoring report does not reproduce: {name}")
    if (root / "report.md").read_text() != v1._report(inputs, selection, evaluated):
        raise ValueError("human-readable report does not reproduce")
    frozen, hashes = v1._frozen_payloads(inputs, selection)
    for name, expected in frozen.items():
        if (root / name).read_bytes() != expected:
            raise ValueError(f"frozen inference component differs: {name}")
    check_snapshot(_snapshot(inputs, inputs["protocol_root"]))
    return targets, selection, evaluated, hashes


def _validate(protocol_root: Path, root: Path, inputs: dict) -> dict:
    _correction(protocol_root, inputs, create=False)
    manifest = validate_manifest(root, schema_version=v1.MANIFEST_SCHEMA)
    targets, selection, evaluated, hashes = _reproduce(inputs, root)
    v1._check_file_allowlist(root, selection, complete=True)
    receipt = read_json(root / "development_selection_receipt.json")
    expected = _receipt(inputs, targets, selection, evaluated, root, hashes)
    if (
        receipt != expected
        or manifest["selection_receipt_sha256"] != receipt["receipt_sha256"]
    ):
        raise ValueError("corrected selection receipt does not reproduce")
    if (
        manifest["parent_file_sha256"] != receipt["parent_file_sha256"]
        or manifest["primary_decision"] != receipt["go_no_go"]["primary_decision"]
        or manifest["frozen_state_file_sha256"]
        != receipt["result_artifact_file_sha256"]["router_frozen_state.json"]
        or manifest["post_receipt_reselection_permitted"] is not False
    ):
        raise ValueError("manifest receipt metadata does not reproduce")
    v1._fit_incapable()
    return {
        "status": "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "selection_receipt_sha256": receipt["receipt_sha256"],
        "primary_decision": receipt["go_no_go"]["primary_decision"],
        "selected_C_hex": receipt["selected_C_hex"],
        "threshold_specification": receipt["threshold_specification"],
        "validation_metrics": receipt["validation_metrics"],
        "assignment_label_columns_opened": False,
        "reselection_calls": 0,
        "receipt_correction": REASON,
        "bundle_root": str(root),
    }


def validate_m4_result(protocol_root: Path) -> dict:
    protocol_root = Path(protocol_root).resolve()
    root = protocol_root / RESULT_DIR
    with exclusive_lock(root):
        return _validate(protocol_root, root, v1._load_inputs(protocol_root))


def build_m4_result(protocol_root: Path) -> dict:
    protocol_root = Path(protocol_root).resolve()
    root = protocol_root / RESULT_DIR
    with exclusive_lock(root):
        root.mkdir(exist_ok=True)
        v1._check_file_allowlist(root)
        inputs = v1._load_inputs(protocol_root)
        _correction(protocol_root, inputs, create=True)
        if (root / "development_selection_receipt.json").exists():
            receipt = read_json(root / "development_selection_receipt.json")
            v1._finish_manifest(root, receipt, read_json(root / "selection.json"))
            return {**_validate(protocol_root, root, inputs), "status": "reused_valid"}
        if (root / "manifest.json").exists():
            raise ValueError("committed result lacks selection receipt")
        if any(root.iterdir()) and not (root / "build_declaration.json").exists():
            raise ValueError("unowned partial result lacks build declaration")
        # Reauthenticate before the first receipt, but never reselect a saved policy.
        targets = v1._create_targets(inputs)
        write_json(
            root / "build_declaration.json", v1._build_declaration(inputs, targets)
        )
        write_bytes(
            root / "validation_targets.parquet",
            table_bytes(v1._frame_table(targets, v1._target_schema())),
        )
        if (root / "selection.json").exists():
            selection = read_json(root / "selection.json")
        else:
            selection = v1._make_selection(inputs, targets)
            write_json(root / "selection.json", selection)
        evaluated = v1._evaluate(inputs, targets, selection)
        v1._validate_saved_selection(
            selection, inputs, targets, evaluated["candidate_metrics.json"]
        )
        for name, value in evaluated.items():
            if isinstance(value, pa.Table):
                write_bytes(root / name, table_bytes(value))
            else:
                write_json(root / name, value)
        write_bytes(
            root / "report.md", v1._report(inputs, selection, evaluated).encode()
        )
        frozen, hashes = v1._frozen_payloads(inputs, selection)
        for name, payload in frozen.items():
            write_bytes(root / name, payload)
        v1._check_file_allowlist(root, selection)
        _correction(protocol_root, inputs, create=False)
        check_snapshot(_snapshot(inputs, protocol_root))
        receipt = _receipt(inputs, targets, selection, evaluated, root, hashes)
        write_json(root / "development_selection_receipt.json", receipt)
        v1._finish_manifest(root, receipt, selection)
        return {
            **_validate(protocol_root, root, inputs),
            "status": "built",
            "assignment_label_columns_opened": True,
        }

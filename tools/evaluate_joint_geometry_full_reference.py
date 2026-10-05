"""Evaluate the authorized 4,200-image reference bank beside the frozen 3,378 bank.

Previous scripts, scores and registry events remain unchanged. Experts,
representations, distances and the strict-positive rule are reused unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_joint_geometry_cleaned_test as prior
from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash, check_self_hash, check_snapshot, exclusive_lock, file_hash,
    read_json, table_bytes, write_bytes, write_json,
)
from plot_multimodal_embedding_geometry import load_development


STUDY_ID = "joint_geometry_full_reference_cleaned_test_v1"
PREVIOUS = Path("dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_joint_geometry_policy_cleaned_test")
REGISTRY = prior.REGISTRY


def validate_previous() -> dict:
    receipt = read_json(PREVIOUS / "receipt.json")
    check_self_hash(receipt, "receipt_sha256")
    check_snapshot(receipt["source_sha256"])
    check_snapshot(receipt["implementation_sha256"])
    for name, expected in receipt["output_sha256"].items():
        if file_hash(PREVIOUS / name) != expected:
            raise ValueError(f"Previous result changed: {name}")
    return receipt


def load_full_reference() -> tuple:
    previous_receipt = validate_previous()
    declaration, train_refs, identities, experts, vectors = prior.load_sources()
    refs, image, geo, source_hashes = load_development("all")
    if len(refs) != 4200 or refs.plot_idx.nunique() != 1625:
        raise ValueError("Expected all 4,200 development images from 1,625 plots")
    train = refs.loc[refs.development_role.eq("train")]
    validation = refs.loc[refs.development_role.eq("validation")]
    if set(train.row_uid) != set(train_refs.row_uid) or len(validation) != 822:
        raise ValueError("Expanded bank must be the exact train/validation union")
    if set(train.plot_idx) & set(validation.plot_idx):
        raise ValueError("Development train/validation plots overlap")
    if set(refs.plot_idx) & set(identities.plot_idx) or set(refs.file) & set(identities.file):
        raise ValueError("Expanded references overlap cleaned test")
    if refs.row_uid.duplicated().any() or refs.file.duplicated().any():
        raise ValueError("Duplicate expanded reference identity")
    subset = pd.Series(np.arange(len(refs)), index=refs.row_uid).loc[train_refs.row_uid].to_numpy()
    for modality, reference in (("image", image), ("geo", geo)):
        normalized = reference / np.linalg.norm(reference, axis=1, keepdims=True)
        if not np.array_equal(normalized[subset], vectors[modality][0]):
            raise ValueError("Previously used reference embeddings changed")
        vectors[modality] = (normalized, vectors[modality][1])

    hashes = declaration["source_sha256"]
    for name, expected in source_hashes.items():
        prior.bind(Path(name), hashes, expected)
    prior.bind(PREVIOUS / "receipt.json", hashes)
    prediction_path = PREVIOUS / "predictions.parquet"
    prior.bind(prediction_path, hashes, previous_receipt["output_sha256"][prediction_path.name])
    previous_predictions = pd.read_parquet(prediction_path).sort_values(
        ["training_seed", "row_uid"]
    ).reset_index(drop=True)
    for column in experts:
        if not np.array_equal(experts[column], previous_predictions[column]):
            raise ValueError(f"Expert/comparator changed from prior evaluation: {column}")
    declaration.update({
        "study_id": STUDY_ID,
        "user_authorization": "Evaluate all 4,200 development references and report alongside existing result",
        "reference_images": 4200, "reference_plots": 1625,
        "reference_bank": "all protocol development_train and development_validation images; no test references",
        "reference_row_uids_sha256": canonical_hash(sorted(refs.row_uid.tolist())),
        "added_reference_images": 822, "added_reference_plots": 325,
        "previous_reference_images": 3378, "previous_reference_plots": 1300,
        "previous_evaluation": str(PREVIOUS.resolve()),
        "previous_receipt_sha256": previous_receipt["receipt_sha256"],
        "bank_expansion_requested_after_previous_test_result": True,
        "reference_bank_selection_on_test": False,
        "bootstrap": {"replicates": 2000, "random_seed": 20260930,
                      "unit": "habitat-stratified whole plots retaining all images and four seeds",
                      "comparisons": ["geometry4200 minus fusion", "geometry4200 minus RF",
                                      "geometry4200 minus geometry3378"],
                      "metrics": ["accuracy", "weighted_f1"]},
    })
    return declaration, refs, identities, experts, vectors, previous_predictions


def reserve_event(output: Path, declaration: dict, prediction_manifest: dict) -> None:
    snapshot = REGISTRY / "snapshots" / declaration["identity_projection_sha256"]
    bundle = prediction_manifest["manifest_sha256"]
    owner = {"study_id": STUDY_ID, "status": "reserved", "owner_output_root": str(output.resolve()),
             "prediction_manifest_sha256": bundle, "composite_bundle_file_sha256": bundle,
             "identity_projection_sha256": declaration["identity_projection_sha256"],
             "declaration_file_sha256": file_hash(output / "declaration.json"),
             "historical_test_exposure": True, "adaptive_reuse": True}
    with exclusive_lock(REGISTRY):
        first_path = snapshot / "first_score_event.json"
        first = read_json(first_path)
        if first["identity_projection_sha256"] != owner["identity_projection_sha256"]:
            raise ValueError("Registered snapshot differs")
        reservation = {**owner, "prior_registered_score_event": True,
                       "first_score_event_file_sha256": file_hash(first_path)}
        write_json(REGISTRY / "bundles" / f"{bundle}.json", reservation)
        write_json(output / "reservation.json", reservation)


def complete_event(output: Path, receipt: dict) -> None:
    reservation = read_json(output / "reservation.json")
    bundle = reservation["composite_bundle_file_sha256"]
    snapshot = REGISTRY / "snapshots" / reservation["identity_projection_sha256"]
    event = {**reservation, "status": "completed", "artifact_role": "locked_test_score_event",
             "receipt_file_sha256": file_hash(output / "receipt.json"),
             "receipt_sha256": receipt["receipt_sha256"],
             "evaluation_event_id": canonical_hash({"study_id": STUDY_ID, "receipt": receipt["receipt_sha256"]})}
    with exclusive_lock(REGISTRY):
        if read_json(REGISTRY / "bundles" / f"{bundle}.json") != reservation:
            raise ValueError("Registry reservation changed")
        write_json(snapshot / f"{bundle}.json", event)
        write_json(REGISTRY / "bundles" / f"{bundle}.completed.json", event)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "receipt.json").exists():
        receipt = read_json(output / "receipt.json")
        check_self_hash(receipt, "receipt_sha256")
        for key in ("source_sha256", "implementation_sha256", "preserved_registry_sha256"):
            check_snapshot(receipt[key])
        for name, expected in receipt["output_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError(f"Completed result changed: {name}")
        validate_previous()
        complete_event(output, receipt)
        print("Verified completed full-reference evaluation and preserved previous result; no rescoring.")
        return
    output.mkdir(parents=True, exist_ok=True)
    declaration, refs, identities, experts, vectors, previous_predictions = load_full_reference()
    registry_before = {str(p.resolve()): file_hash(p) for p in sorted(REGISTRY.rglob("*")) if p.is_file()}
    previous_receipt = read_json(PREVIOUS / "receipt.json")
    code_hashes = dict(previous_receipt["implementation_sha256"])
    for path in (Path(__file__).resolve(), Path("tools/joint_geometry_full_reference_report.py").resolve(),
                 Path("multimodal/geo_helpfulness_protocol.py").resolve()):
        code_hashes[str(path)] = file_hash(path)
    declaration["implementation_sha256"] = code_hashes
    declaration["declaration_sha256"] = canonical_hash(declaration)
    write_json(output / "declaration.json", declaration)
    with threadpool_limits(limits=1):
        predictions, support = prior.geometry_predictions(refs, identities, experts, vectors)
    for column in ("geometry_pred", "geometry_acted", "A_image", "A_geo"):
        renamed = column.replace("geometry_", "geometry3378_") if column.startswith("geometry_") else column + "3378"
        predictions[renamed] = previous_predictions[column]
    for column in ("d_image_fusion", "d_image_geo", "d_geo_fusion", "d_geo_geo"):
        if np.any(predictions[column].to_numpy() > previous_predictions[column].to_numpy() + 1e-12):
            raise ValueError("Adding reference examples increased a nearest-class distance")
        predictions[column + "3378"] = previous_predictions[column]
    reference_columns = ["row_uid", "file", "plot_idx", "label_id_dense", "label_name", "development_role"]
    write_bytes(output / "reference_bank.parquet", table_bytes(pa.Table.from_pandas(refs[reference_columns], preserve_index=False)))
    write_bytes(output / "predictions.parquet", table_bytes(pa.Table.from_pandas(predictions, preserve_index=False)))
    prediction_manifest = {
        "study_id": STUDY_ID, "declaration_sha256": declaration["declaration_sha256"],
        "predictions_sha256": file_hash(output / "predictions.parquet"), "rows": len(predictions),
        "test_label_columns_present": False, "reference_class_plot_support": support,
        "reference_bank_sha256": file_hash(output / "reference_bank.parquet"),
        "previous_predictions_sha256": file_hash(PREVIOUS / "predictions.parquet"),
    }
    prediction_manifest["manifest_sha256"] = canonical_hash(prediction_manifest)
    write_json(output / "prediction_manifest.json", prediction_manifest)
    reserve_event(output, declaration, prediction_manifest)
    print("Full-reference decisions sealed; acquiring canonical labels for the matched comparison.", flush=True)
    scored, label_hashes = prior.score_labels(output, predictions, declaration)
    write_bytes(output / "scored_predictions.parquet", table_bytes(pa.Table.from_pandas(scored, preserve_index=False)))
    from joint_geometry_full_reference_report import write_results
    result = write_results(output, scored, declaration)
    hashes = {**declaration["source_sha256"], **label_hashes}
    check_snapshot(hashes)
    check_snapshot(code_hashes)
    check_snapshot(registry_before)
    validate_previous()
    receipt = {
        "study_id": STUDY_ID, "source_sha256": hashes, "implementation_sha256": code_hashes,
        "prediction_manifest_sha256": prediction_manifest["manifest_sha256"],
        "declaration_sha256": declaration["declaration_sha256"], "result": result,
        "historical_test_exposure": True, "independent_confirmation": False,
        "preserved_registry_sha256": registry_before,
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    write_json(output / "receipt.json", receipt)
    complete_event(output, receipt)
    print("Completed full-reference evaluation: " + str(output / "report.md"), flush=True)


if __name__ == "__main__":
    main()

"""Evaluate fixed K=5 geometry correction on validation and both test banks.

Previous scripts and evaluations remain immutable. No threshold selection,
expert fitting, or reference selection occurs in this descriptive follow-up.
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
import evaluate_joint_geometry_full_reference as full
import evaluate_joint_geometry_policy as validation_study
from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash, check_self_hash, check_snapshot, exclusive_lock, file_hash,
    read_json, table_bytes, write_bytes, write_json,
)


STUDY_ID = "joint_geometry_k5_correction_v1"
REGISTRY = prior.REGISTRY
FULL_REFERENCE = Path("dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_joint_geometry_policy_full_reference")
K = 5


def sealed_previous(path: Path) -> dict:
    receipt = read_json(path / "receipt.json")
    check_self_hash(receipt, "receipt_sha256")
    check_snapshot(receipt["source_sha256"])
    check_snapshot(receipt["implementation_sha256"])
    for name, expected in receipt["output_sha256"].items():
        if file_hash(path / name) != expected:
            raise ValueError(f"Previous evaluation changed: {path / name}")
    return receipt


def validation_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, assignments, hashes = validation_study.load_inputs()
    path = validation_study.STUDY / "geometry_scores.parquet"
    k5 = pd.read_parquet(path, filters=[("cohort", "=", "validation"), ("distance_k", "=", 5)])
    k5 = k5.sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    for name in ("row_uid", "file", "plot_idx", "training_seed", "raw_pred", "geo_pred", "image_pred"):
        if not np.array_equal(rows[name], k5[name]):
            raise ValueError("Cached K=1 and K=5 validation inputs differ")
    keep = ["row_uid", "file", "plot_idx", "training_seed", "image_pred", "geo_pred", "raw_pred",
            "reference_plots_fusion_class", "reference_plots_geo_class"]
    result = rows[keep].copy()
    result["A_image"] = k5.A_image
    result["A_geo"] = k5.A_geo
    result["valid_k5"] = np.isfinite(k5[["A_image", "A_geo"]]).all(axis=1)
    expected_support = (rows.reference_plots_fusion_class >= K) & (rows.reference_plots_geo_class >= K)
    if not np.array_equal(result.valid_k5, expected_support):
        raise ValueError("Validation K=5 availability disagrees with class plot counts")
    result["k1_acted"] = rows.raw_pred.ne(rows.geo_pred) & rows.A_image.gt(0) & rows.A_geo.gt(0)
    result["k5_acted"] = (rows.raw_pred.ne(rows.geo_pred) & result.valid_k5
                           & result.A_image.gt(0) & result.A_geo.gt(0))
    result["k1_supported_acted"] = result.k1_acted & result.valid_k5
    for method in ("k1", "k5", "k1_supported"):
        result[f"{method}_pred"] = np.where(result[f"{method}_acted"], rows.geo_pred, rows.raw_pred).astype(np.int64)
    for column in ("d_image_fusion", "d_image_geo", "d_geo_fusion", "d_geo_geo"):
        if np.any(k5[column].to_numpy() < rows[column].to_numpy() - 1e-12):
            raise ValueError("Validation mean-five distance is smaller than nearest distance")
        result[column] = k5[column]
        result[column + "_k1"] = rows[column]
    labels = assignments.loc[assignments.development_role.eq("validation"),
                             ["row_uid", "label_id_dense", "canonical_l3_id", "label_name"]]
    return result, labels, {str(Path(p).resolve()): v for p, v in hashes.items()}


def k5_scores(refs: pd.DataFrame, vectors: dict, identities: pd.DataFrame,
              experts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Take one minimum photo distance per plot, then average five class plots."""
    order = np.lexsort((refs.file.to_numpy(), refs.plot_idx.to_numpy()))
    ordered = refs.iloc[order]
    starts = np.flatnonzero(np.r_[True, ordered.plot_idx.to_numpy()[1:] != ordered.plot_idx.to_numpy()[:-1]])
    plots = ordered.iloc[starts].plot_idx.to_numpy()
    classes = ordered.iloc[starts].label_id_dense.to_numpy(dtype=int)
    if ordered.groupby("plot_idx").label_id_dense.nunique().max() != 1:
        raise ValueError("Reference plot has multiple labels")
    support = np.bincount(classes, minlength=18)
    positions = pd.Series(np.arange(len(identities)), index=identities.row_uid).loc[experts.row_uid].to_numpy()
    raw, geo = experts.raw_pred.to_numpy(dtype=int), experts.geo_pred.to_numpy(dtype=int)
    output = experts.copy()
    output["reference_plots_fusion_class"] = support[raw]
    output["reference_plots_geo_class"] = support[geo]
    for modality, (reference, query) in vectors.items():
        distances = np.clip(1.0 - query @ reference[order].T, 0.0, 2.0)
        plot_distances = np.minimum.reduceat(distances, starts, axis=1)
        class_distances = np.full((len(query), 18), np.nan)
        nearest_plots = np.full((len(query), 18, K), -1, dtype=int)
        for c in range(18):
            candidates = np.flatnonzero(classes == c)
            if len(candidates) < K:
                continue
            chosen = candidates[np.argsort(plot_distances[:, candidates], axis=1, kind="stable")[:, :K]]
            nearest_plots[:, c] = chosen
            class_distances[:, c] = np.take_along_axis(plot_distances, chosen, axis=1).mean(axis=1)
        dr, dg = class_distances[positions, raw], class_distances[positions, geo]
        output[f"d_{modality}_fusion"] = dr
        output[f"d_{modality}_geo"] = dg
        output[f"A_{modality}"] = np.log((dr + prior.EPSILON) / (dg + prior.EPSILON))
        for candidate, prediction in (("fusion", raw), ("geo", geo)):
            chosen = nearest_plots[positions, prediction]
            output[f"nearest_{modality}_{candidate}_plots"] = [
                plots[p].tolist() if np.all(p >= 0) else [] for p in chosen]
    output["valid_k5"] = np.isfinite(output[["A_image", "A_geo"]]).all(axis=1)
    if not np.array_equal(output.valid_k5, (support[raw] >= K) & (support[geo] >= K)):
        raise ValueError("K=5 availability disagrees with plot counts")
    output["acted"] = (output.raw_pred.ne(output.geo_pred) & output.valid_k5
                        & output.A_image.gt(0) & output.A_geo.gt(0))
    output["pred"] = np.where(output.acted, output.geo_pred, output.raw_pred).astype(np.int64)
    return output, {str(c): int(n) for c, n in enumerate(support)}


def load_inputs() -> tuple:
    full_receipt = sealed_previous(FULL_REFERENCE)
    declaration, refs, identities, experts, vectors, old3378 = full.load_full_reference()
    hashes = declaration["source_sha256"]
    prior.bind(FULL_REFERENCE / "receipt.json", hashes)
    prediction_path = FULL_REFERENCE / "predictions.parquet"
    prior.bind(prediction_path, hashes, full_receipt["output_sha256"][prediction_path.name])
    old4200 = pd.read_parquet(prediction_path).sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    for column in experts:
        if not np.array_equal(experts[column], old4200[column]):
            raise ValueError("Full-bank historical experts differ")
    val_predictions, val_labels, val_hashes = validation_inputs()
    hashes.update(val_hashes)
    keep = ["expert_family", "expert_fit_images", "expert_fit_plots", "queries", "query_plots",
            "training_seeds", "reference_encoder", "distance", "epsilon", "score", "action",
            "ties_or_missing", "rf_policy_id", "rf_frozen_policy", "ontology", "label_source",
            "identity_projection_sha256", "limitations"]
    declaration = {key: declaration[key] for key in keep}
    declaration.update({
        "study_id": STUDY_ID, "user_authorization": "Try K=5 on the correction performance",
        "historical_test_exposure": True, "adaptive_reuse": True, "independent_confirmation": False,
        "distance_k": K, "class_distance": "mean cosine distance to five nearest distinct reference plots",
        "plot_distance": "minimum query-to-photo distance within the reference plot",
        "insufficient_reference_support": "retain fusion when either candidate has fewer than five plots",
        "reference_banks": {"3378": {"images": 3378, "plots": 1300, "roles": ["train"]},
                            "4200": {"images": 4200, "plots": 1625, "roles": ["train", "validation"]}},
        "validation": {"images": 822, "plots": 325, "reference_bank": "3378", "expert_fit_images": 3378},
        "cleaned_test": {"images": 1347, "plots": 531, "reference_banks": ["3378", "4200"], "expert_fit_images": 4200},
        "k1_supported_control": "K1 action AND K5 availability; otherwise retain fusion; full cohort denominator",
        "reference_row_uids_sha256": {str(n): canonical_hash(sorted(refs.loc[mask].row_uid.tolist()))
                                     for n, mask in ((3378, refs.development_role.eq("train")), (4200, np.ones(len(refs), dtype=bool)))},
        "bootstrap": {"replicates": 2000, "random_seed": 20260930,
                      "unit": "habitat-stratified whole plots retaining all photos and four expert seeds",
                      "metrics": ["accuracy", "weighted_f1"],
                      "validation_contrasts": ["K5 minus fusion", "K5 minus K1", "K5 minus K1-supported"],
                      "test_contrasts_per_bank": ["K5 minus fusion", "K5 minus RF", "K5 minus K1", "K5 minus K1-supported"]},
        "model_fitting": False, "threshold_selection": False, "winner_selection": False,
        "source_sha256": hashes,
    })
    return declaration, refs, identities, experts, vectors, {"3378": old3378, "4200": old4200}, val_predictions, val_labels


def registry_event(output: Path, declaration: dict, prediction_manifest: dict,
                   receipt: dict | None = None) -> None:
    snapshot = REGISTRY / "snapshots" / declaration["identity_projection_sha256"]
    bundle = prediction_manifest["manifest_sha256"]
    with exclusive_lock(REGISTRY):
        if receipt is None:
            first_path = snapshot / "first_score_event.json"
            if read_json(first_path)["identity_projection_sha256"] != declaration["identity_projection_sha256"]:
                raise ValueError("Registered identity differs")
            reservation = {
                "study_id": STUDY_ID, "status": "reserved", "owner_output_root": str(output.resolve()),
                "prediction_manifest_sha256": bundle, "composite_bundle_file_sha256": bundle,
                "identity_projection_sha256": declaration["identity_projection_sha256"],
                "declaration_file_sha256": file_hash(output / "declaration.json"),
                "historical_test_exposure": True, "adaptive_reuse": True,
                "prior_registered_score_event": True,
                "first_score_event_file_sha256": file_hash(first_path),
            }
            write_json(REGISTRY / "bundles" / f"{bundle}.json", reservation)
            write_json(output / "reservation.json", reservation)
            return
        reservation = read_json(output / "reservation.json")
        if read_json(REGISTRY / "bundles" / f"{bundle}.json") != reservation:
            raise ValueError("Registry reservation changed")
        event = {**reservation, "status": "completed", "artifact_role": "locked_test_score_event",
                 "receipt_file_sha256": file_hash(output / "receipt.json"), "receipt_sha256": receipt["receipt_sha256"],
                 "evaluation_event_id": canonical_hash({"study_id": STUDY_ID, "receipt": receipt["receipt_sha256"]})}
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
        sealed_previous(FULL_REFERENCE)
        full.validate_previous()
        registry_event(output, read_json(output / "declaration.json"),
                       read_json(output / "prediction_manifest.json"), receipt)
        print("Verified completed K=5 evaluation and preserved prior results; no rescoring.")
        return
    output.mkdir(parents=True, exist_ok=True)
    declaration, refs, identities, experts, vectors, old, val_predictions, val_labels = load_inputs()
    registry_before = {str(p.resolve()): file_hash(p) for p in sorted(REGISTRY.rglob("*")) if p.is_file()}
    code_hashes = dict(read_json(FULL_REFERENCE / "receipt.json")["implementation_sha256"])
    for path in (Path(__file__).resolve(), Path("tools/joint_geometry_k5_report.py").resolve(),
                 Path("tools/evaluate_joint_geometry_policy.py").resolve(),
                 Path("multimodal/geo_helpfulness_oof.py").resolve()):
        code_hashes[str(path)] = file_hash(path)
    declaration["implementation_sha256"] = code_hashes
    declaration["declaration_sha256"] = canonical_hash(declaration)
    write_json(output / "declaration.json", declaration)
    predictions = experts.copy()
    supports = {}
    details = {}
    with threadpool_limits(limits=1):
        for bank in ("3378", "4200"):
            mask = refs.development_role.eq("train").to_numpy() if bank == "3378" else np.ones(len(refs), dtype=bool)
            bank_vectors = {m: (r[mask], q) for m, (r, q) in vectors.items()}
            current, supports[bank] = k5_scores(refs.loc[mask], bank_vectors, identities, experts)
            details[bank] = current
            prefix = f"g{bank}"
            predictions[f"valid_k5_{bank}"] = current.valid_k5
            predictions[f"{prefix}_k1_acted"] = old[bank].geometry_acted
            predictions[f"{prefix}_k1_pred"] = old[bank].geometry_pred
            predictions[f"{prefix}_k5_acted"] = current.acted
            predictions[f"{prefix}_k5_pred"] = current.pred
            predictions[f"{prefix}_k1_supported_acted"] = old[bank].geometry_acted & current.valid_k5
            predictions[f"{prefix}_k1_supported_pred"] = np.where(
                predictions[f"{prefix}_k1_supported_acted"], experts.geo_pred, experts.raw_pred).astype(np.int64)
            for column in current.columns.difference(experts.columns):
                if column not in ("acted", "pred", "valid_k5"):
                    predictions[f"{prefix}_{column}"] = current[column]
            for column in ("d_image_fusion", "d_image_geo", "d_geo_fusion", "d_geo_geo"):
                if np.any(current[column].to_numpy() < old[bank][column].to_numpy() - 1e-12):
                    raise ValueError("Mean-five distance is less than nearest distance")
                if bank == "4200" and np.any(current[column].to_numpy() > details["3378"][column].to_numpy() + 1e-12):
                    raise ValueError("Adding reference plots increased mean-five distance")
    if not np.array_equal(predictions.valid_k5_3378, predictions.valid_k5_4200):
        raise ValueError("Unexpected K=5 availability change between these banks")
    # Canonical scorer expects current geometry columns; bind these to the full bank.
    predictions["geometry_pred"] = predictions.g4200_k5_pred
    predictions["geometry_acted"] = predictions.g4200_k5_acted
    for name, frame in (("predictions.parquet", predictions), ("validation_predictions.parquet", val_predictions)):
        write_bytes(output / name, table_bytes(pa.Table.from_pandas(frame, preserve_index=False)))
    prediction_manifest = {
        "study_id": STUDY_ID, "declaration_sha256": declaration["declaration_sha256"],
        "predictions_sha256": file_hash(output / "predictions.parquet"), "rows": len(predictions),
        "validation_predictions_sha256": file_hash(output / "validation_predictions.parquet"),
        "query_label_columns_present": False, "reference_class_plot_support": supports,
    }
    prediction_manifest["manifest_sha256"] = canonical_hash(prediction_manifest)
    write_json(output / "prediction_manifest.json", prediction_manifest)
    registry_event(output, declaration, prediction_manifest)
    print("K=5 predictions for validation and both test banks sealed; acquiring canonical test labels.", flush=True)
    scored, label_hashes = prior.score_labels(output, predictions, declaration)
    validation = val_predictions.merge(val_labels, on="row_uid", validate="many_to_one")
    for name, frame in (("scored_predictions.parquet", scored), ("validation_scored.parquet", validation)):
        write_bytes(output / name, table_bytes(pa.Table.from_pandas(frame, preserve_index=False)))
    from joint_geometry_k5_report import write_results
    result = write_results(output, validation, scored, declaration)
    hashes = {**declaration["source_sha256"], **label_hashes}
    for snapshot in (hashes, code_hashes, registry_before):
        check_snapshot(snapshot)
    sealed_previous(FULL_REFERENCE)
    full.validate_previous()
    receipt = {
        "study_id": STUDY_ID, "source_sha256": hashes, "implementation_sha256": code_hashes,
        "declaration_sha256": declaration["declaration_sha256"],
        "prediction_manifest_sha256": prediction_manifest["manifest_sha256"], "result": result,
        "preserved_registry_sha256": registry_before,
        "historical_test_exposure": True, "independent_confirmation": False,
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    write_json(output / "receipt.json", receipt)
    registry_event(output, declaration, prediction_manifest, receipt)
    print("Completed K=5 correction evaluation: " + str(output / "report.md"), flush=True)


if __name__ == "__main__":
    main()

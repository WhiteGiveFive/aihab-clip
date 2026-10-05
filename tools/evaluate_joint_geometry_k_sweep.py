"""Evaluate the declared shared-neighborhood geometry sweep without fitting.

Validation uses development-training references. Cleaned test uses all
development references and the unchanged September 15 expert/RF predictions.
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

import evaluate_joint_geometry_k5 as previous
from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash, check_self_hash, check_snapshot, exclusive_lock, file_hash,
    read_json, table_bytes, write_bytes, write_json,
)


STUDY_ID = "joint_geometry_shared_k_sweep_v1"
KS = (1, 5, 10, 20)
PREVIOUS = Path("dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_joint_geometry_policy_k5")
REGISTRY = previous.REGISTRY
EXPERT_COLUMNS = ["row_uid", "file", "plot_idx", "training_seed", "image_pred", "geo_pred", "raw_pred"]


def shared_k_predictions(refs: pd.DataFrame, identities: pd.DataFrame,
                         experts: pd.DataFrame, vectors: dict) -> tuple[pd.DataFrame, dict]:
    """Average nearest plot distances with equal candidate neighborhood sizes."""
    if set(refs.plot_idx) & set(identities.plot_idx) or set(refs.file) & set(identities.file):
        raise ValueError("Reference/query identities overlap")
    order = np.lexsort((refs.file.to_numpy(), refs.plot_idx.to_numpy()))
    ordered = refs.iloc[order]
    if ordered.groupby("plot_idx").label_id_dense.nunique().max() != 1:
        raise ValueError("Reference plot has multiple habitat labels")
    starts = np.flatnonzero(np.r_[True, ordered.plot_idx.to_numpy()[1:] != ordered.plot_idx.to_numpy()[:-1]])
    plots = ordered.iloc[starts].plot_idx.to_numpy()
    classes = ordered.iloc[starts].label_id_dense.to_numpy(dtype=int)
    support = np.bincount(classes, minlength=18)
    positions = pd.Series(np.arange(len(identities)), index=identities.row_uid).loc[experts.row_uid].to_numpy()
    raw, geo = experts.raw_pred.to_numpy(dtype=int), experts.geo_pred.to_numpy(dtype=int)
    output = experts.copy()
    extra = {"reference_plots_fusion_class": support[raw], "reference_plots_geo_class": support[geo]}
    effective = {k: np.minimum(k, np.minimum(support[raw], support[geo])) for k in KS}
    for k in KS:
        extra[f"k{k}_effective"] = effective[k]
    for modality, (reference, query) in vectors.items():
        if len(reference) != len(refs) or len(query) != len(identities):
            raise ValueError("Embedding arrays differ from declared identities")
        if not np.isfinite(reference).all() or not np.isfinite(query).all():
            raise ValueError("Nonfinite embeddings")
        if not all(np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-12, rtol=0) for x in (reference, query)):
            raise ValueError("Expected independently unit-normalized embeddings")
        distances = np.clip(1.0 - query @ reference[order].T, 0.0, 2.0)
        plot_distances = np.minimum.reduceat(distances, starts, axis=1)
        means = np.full((len(query), 18, max(KS)), np.nan)
        nearest = np.full((len(query), 18, max(KS)), -1, dtype=int)
        for c in range(18):
            candidates = np.flatnonzero(classes == c)
            size = min(max(KS), len(candidates))
            if size == 0:
                continue
            chosen = candidates[np.argsort(plot_distances[:, candidates], axis=1, kind="stable")[:, :size]]
            nearest[:, c, :size] = chosen
            ranked_distances = np.take_along_axis(plot_distances, chosen, axis=1)
            means[:, c, :size] = ranked_distances.cumsum(axis=1) / np.arange(1, size + 1)
        for k in KS:
            ke = effective[k]
            valid = ke > 0
            candidate_distances = {}
            for candidate, prediction in (("fusion", raw), ("geo", geo)):
                d = np.full(len(experts), np.nan)
                d[valid] = means[positions[valid], prediction[valid], ke[valid] - 1]
                candidate_distances[candidate] = d
                extra[f"k{k}_d_{modality}_{candidate}"] = d
                selected = nearest[positions, prediction]
                extra[f"k{k}_nearest_{modality}_{candidate}_plots"] = [
                    plots[row[:n]].tolist() if n > 0 else [] for row, n in zip(selected, ke)]
            dr, dg = candidate_distances["fusion"], candidate_distances["geo"]
            extra[f"k{k}_A_{modality}"] = np.log((dr + previous.prior.EPSILON) / (dg + previous.prior.EPSILON))
    for k in KS:
        ai, ag = extra[f"k{k}_A_image"], extra[f"k{k}_A_geo"]
        finite = np.isfinite(ai) & np.isfinite(ag)
        if not np.array_equal(finite, effective[k] > 0):
            raise ValueError("Reference support and score availability differ")
        acted = (raw != geo) & finite & (ai > 0) & (ag > 0)
        extra[f"k{k}_acted"] = acted
        extra[f"k{k}_pred"] = np.where(acted, geo, raw).astype(np.int64)
    for lower, upper in zip(KS[:-1], KS[1:]):
        for modality in vectors:
            for candidate in ("fusion", "geo"):
                dl = extra[f"k{lower}_d_{modality}_{candidate}"]
                du = extra[f"k{upper}_d_{modality}_{candidate}"]
                if np.any(du < dl - 1e-12):
                    raise ValueError("Larger neighborhood reduced mean distance")
    output = pd.concat([output, pd.DataFrame(extra, index=output.index)], axis=1)
    return output, {str(c): int(n) for c, n in enumerate(support)}


def load_inputs() -> tuple:
    receipt = previous.sealed_previous(PREVIOUS)
    declaration, refs, identities, experts, vectors, _, val_prior, val_labels = previous.load_inputs()
    hashes = declaration["source_sha256"]
    previous.prior.bind(PREVIOUS / "receipt.json", hashes)
    saved = {}
    for cohort, name in (("test", "predictions.parquet"), ("validation", "validation_predictions.parquet")):
        path = PREVIOUS / name
        previous.prior.bind(path, hashes, receipt["output_sha256"][name])
        saved[cohort] = pd.read_parquet(path).sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    for cohort, current in (("test", experts), ("validation", val_prior)):
        for column in current:
            comparison_options = {"equal_nan": True} if current[column].dtype.kind in "fc" else {}
            if not np.array_equal(current[column], saved[cohort][column], **comparison_options):
                raise ValueError(f"Previously sealed predictions differ: {cohort}/{column}")
    val_experts = val_prior[EXPERT_COLUMNS].copy()
    val_ids = val_experts.loc[val_experts.training_seed.eq(1), ["row_uid", "file", "plot_idx"]].reset_index(drop=True)
    val_positions = pd.Series(np.arange(len(refs)), index=refs.row_uid).loc[val_ids.row_uid].to_numpy()
    train_mask = refs.development_role.eq("train").to_numpy()
    val_vectors = {modality: (reference[train_mask], reference[val_positions])
                   for modality, (reference, _) in vectors.items()}
    if len(val_ids) != 822 or len(refs.loc[train_mask]) != 3378:
        raise ValueError("Unexpected development validation split")
    keys = ["expert_family", "expert_fit_images", "expert_fit_plots", "queries", "query_plots",
            "training_seeds", "reference_encoder", "distance", "epsilon", "score", "action",
            "ties_or_missing", "rf_policy_id", "rf_frozen_policy", "ontology", "label_source",
            "identity_projection_sha256", "limitations"]
    declaration = {key: declaration[key] for key in keys}
    contrasts = [(f"k{k}", "fusion") for k in KS] + [
        ("k5", "strict_k5"), ("k5", "k1"), ("k10", "k5"), ("k20", "k10"), ("k20", "k5")]
    declaration.update({
        "study_id": STUDY_ID,
        "user_authorization": "Approved K=1,5,10,20 with shared fallback on validation, plus cleaned test using the entire development reference bank",
        "historical_test_exposure": True, "adaptive_reuse": True, "independent_confirmation": False,
        "nominal_k": list(KS), "effective_k": "min(nominal_k, reference_plots_fusion_class, reference_plots_geo_class)",
        "same_effective_k_for": "both candidate classes and both modalities for each query and expert seed",
        "class_distance": "mean cosine distance to effective_k nearest distinct reference plots of candidate class",
        "plot_distance": "minimum query-to-photo distance within reference plot",
        "insufficient_reference_support": "reduce shared effective_k; retain fusion if effective_k=0",
        "strict_k5_comparator": "previous saved K5; retain fusion if either candidate class has fewer than five plots",
        "cohorts": {
            "validation": {"images": 822, "plots": 325, "reference_images": 3378, "reference_plots": 1300, "expert_fit_images": 3378},
            "test": {"images": 1347, "plots": 531, "reference_images": 4200, "reference_plots": 1625, "expert_fit_images": 4200}},
        "reference_row_uids_sha256": {
            "validation": canonical_hash(sorted(refs.loc[train_mask].row_uid.tolist())),
            "test": canonical_hash(sorted(refs.row_uid.tolist()))},
        "bootstrap": {"replicates": 2000, "random_seed": 20260930,
                      "unit": "habitat-stratified whole plots retaining all photos and four expert seeds",
                      "metrics": ["accuracy", "weighted_f1"],
                      "validation_contrasts": contrasts,
                      "test_contrasts": contrasts + [(f"k{k}", "rf") for k in KS]},
        "model_fitting": False, "threshold_selection": False, "winner_selection": False,
        "source_sha256": hashes,
    })
    cohorts = {
        "validation": (refs.loc[train_mask], val_ids, val_experts, val_vectors),
        "test": (refs, identities, experts, vectors),
    }
    return declaration, cohorts, saved, val_labels, receipt


def check_previous_predictions(current: pd.DataFrame, saved: pd.DataFrame, cohort: str) -> None:
    prefix = "g4200_" if cohort == "test" else ""
    for column in ("pred", "acted"):
        if not np.array_equal(current[f"k1_{column}"], saved[f"{prefix}k1_{column}"]):
            raise ValueError("Shared K1 does not reproduce previous K1")
        current[f"strict_k5_{column}"] = saved[f"{prefix}k5_{column}"]
    supported = current.k5_effective.eq(5).to_numpy()
    if not np.array_equal(current.k5_acted[supported], current.strict_k5_acted[supported]):
        raise ValueError("Shared/strict K5 disagree despite complete support")
    if np.any(current.strict_k5_acted.to_numpy() & ~supported):
        raise ValueError("Previous strict K5 acted without enough references")
    for modality in ("image", "geo"):
        for candidate in ("fusion", "geo"):
            new = current[f"k5_d_{modality}_{candidate}"].to_numpy()[supported]
            old = saved[f"{prefix}d_{modality}_{candidate}"].to_numpy()[supported]
            if not np.allclose(new, old, atol=1e-12, rtol=0):
                raise ValueError("Shared/strict K5 supported distances differ")


def registry_event(output: Path, declaration: dict, manifest: dict, receipt: dict | None = None) -> None:
    bundle = manifest["manifest_sha256"]
    snapshot = REGISTRY / "snapshots" / declaration["identity_projection_sha256"]
    with exclusive_lock(REGISTRY):
        if receipt is None:
            first = snapshot / "first_score_event.json"
            if read_json(first)["identity_projection_sha256"] != declaration["identity_projection_sha256"]:
                raise ValueError("Registered identity differs")
            reservation = {
                "study_id": STUDY_ID, "status": "reserved", "owner_output_root": str(output.resolve()),
                "prediction_manifest_sha256": bundle, "composite_bundle_file_sha256": bundle,
                "identity_projection_sha256": declaration["identity_projection_sha256"],
                "declaration_file_sha256": file_hash(output / "declaration.json"),
                "historical_test_exposure": True, "adaptive_reuse": True, "prior_registered_score_event": True,
                "first_score_event_file_sha256": file_hash(first),
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
        previous.sealed_previous(PREVIOUS)
        registry_event(output, read_json(output / "declaration.json"), read_json(output / "prediction_manifest.json"), receipt)
        print("Verified completed shared-K evaluation; no rescoring.")
        return
    declaration, inputs, old, val_labels, parent_receipt = load_inputs()
    output.mkdir(parents=True, exist_ok=True)
    registry_before = {str(p.resolve()): file_hash(p) for p in sorted(REGISTRY.rglob("*")) if p.is_file()}
    code_hashes = dict(parent_receipt["implementation_sha256"])
    for path in (Path(__file__).resolve(), Path("tools/joint_geometry_k_sweep_report.py").resolve()):
        code_hashes[str(path)] = file_hash(path)
    declaration["implementation_sha256"] = code_hashes
    declaration["declaration_sha256"] = canonical_hash(declaration)
    write_json(output / "declaration.json", declaration)
    predictions, supports = {}, {}
    with threadpool_limits(limits=1):
        for cohort, args in inputs.items():
            predictions[cohort], supports[cohort] = shared_k_predictions(*args)
            check_previous_predictions(predictions[cohort], old[cohort], cohort)
    # The historical label scorer uses only experts/RF; aliases retain its schema.
    predictions["test"]["geometry_pred"] = predictions["test"].strict_k5_pred
    predictions["test"]["geometry_acted"] = predictions["test"].strict_k5_acted
    for cohort, rows in predictions.items():
        write_bytes(output / f"{cohort}_predictions.parquet", table_bytes(pa.Table.from_pandas(rows, preserve_index=False)))
    manifest = {
        "study_id": STUDY_ID, "declaration_sha256": declaration["declaration_sha256"],
        "predictions_sha256": {c: file_hash(output / f"{c}_predictions.parquet") for c in predictions},
        "query_label_columns_present": False, "reference_class_plot_support": supports,
        "rows": {c: len(rows) for c, rows in predictions.items()},
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(output / "prediction_manifest.json", manifest)
    registry_event(output, declaration, manifest)
    print("All K predictions sealed; acquiring canonical cleaned-test labels.", flush=True)
    test, label_hashes = previous.prior.score_labels(output, predictions["test"], declaration)
    validation = predictions["validation"].merge(val_labels, on="row_uid", validate="many_to_one")
    for cohort, rows in (("validation", validation), ("test", test)):
        write_bytes(output / f"{cohort}_scored.parquet", table_bytes(pa.Table.from_pandas(rows, preserve_index=False)))
    from joint_geometry_k_sweep_report import write_results
    with threadpool_limits(limits=1):
        result = write_results(output, validation, test, declaration)
    hashes = {**declaration["source_sha256"], **label_hashes}
    for snapshot in (hashes, code_hashes, registry_before):
        check_snapshot(snapshot)
    previous.sealed_previous(PREVIOUS)
    receipt = {
        "study_id": STUDY_ID, "source_sha256": hashes, "implementation_sha256": code_hashes,
        "declaration_sha256": declaration["declaration_sha256"], "prediction_manifest_sha256": manifest["manifest_sha256"],
        "result": result, "preserved_registry_sha256": registry_before,
        "historical_test_exposure": True, "independent_confirmation": False,
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    write_json(output / "receipt.json", receipt)
    registry_event(output, declaration, manifest, receipt)
    print("Completed shared-K comparison: " + str(output / "report.md"), flush=True)


if __name__ == "__main__":
    main()

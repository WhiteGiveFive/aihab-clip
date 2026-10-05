"""Select strict-support geometry thresholds on validation, then transfer them.

All selected rules are frozen before test geometry is applied. Both reference
banks use the same validation-selected threshold for each K and guardrail.
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

import evaluate_joint_geometry_k_sweep as geometry
from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash, check_self_hash, check_snapshot, exclusive_lock, file_hash,
    read_json, table_bytes, write_bytes, write_json,
)


STUDY_ID = "joint_geometry_validation_thresholds_v1"
PREVIOUS = Path("dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_joint_geometry_policy_k_sweep")
REGISTRY = geometry.REGISTRY
KS = (5, 10)
LABEL_COLUMNS = ("label_id_dense", "canonical_l3_id", "label_name")
ROUTER_CONFIG = Path("configs/multimodal_geo_relaxed_guardrails_rf_v1.json")


def load_declaration() -> tuple[dict, dict]:
    receipt = geometry.previous.sealed_previous(PREVIOUS)
    old = read_json(PREVIOUS / "declaration.json")
    check_self_hash(old, "declaration_sha256")
    keys = ["expert_family", "expert_fit_images", "expert_fit_plots", "queries", "query_plots",
            "training_seeds", "reference_encoder", "distance", "epsilon", "ontology",
            "rf_policy_id", "rf_frozen_policy", "label_source", "identity_projection_sha256", "limitations"]
    declaration = {k: old[k] for k in keys}
    sources = dict(receipt["source_sha256"])
    for name in ("receipt.json", "declaration.json", "validation_scored.parquet", "test_predictions.parquet"):
        geometry.previous.prior.bind(PREVIOUS / name, sources)
    for path in (ROUTER_CONFIG, Path("dev_plans/reliability_aware_geo_fusion/10_relaxed_guardrails_rf.md")):
        geometry.previous.prior.bind(path, sources)
    config = read_json(ROUTER_CONFIG)
    if (config["minimum_plots_grid"] != [20, 10, 0] or config["harm_numerator_grid"] != [5, 7, 9]
            or config["harm_denominator"] != 10 or config["minimum_coverage_fraction"] != 0.01):
        raise ValueError("Historical relaxed-router grid differs")
    declaration.update({
        "study_id": STUDY_ID,
        "user_authorization": "Fit validation geometry thresholds for K5/K10 with relaxed harm rules; apply both 3378/4200 test reference banks",
        "user_support_choice": "Strict support: retain fusion when either class has fewer than K plots",
        "nominal_k": list(KS), "reference_support": "both candidate classes must have at least nominal_k distinct reference plots",
        "class_distance": "mean cosine distance to nominal_k nearest distinct reference plots of candidate class",
        "plot_distance": "minimum query-to-photo distance within reference plot",
        "score": "min(A_image, A_geo); A_m=log((d_m(fusion)+epsilon)/(d_m(geo)+epsilon))",
        "action": "expert disagreement AND strict reference support AND finite score AND score > frozen threshold",
        "ties_or_missing": "retain fusion", "model_fitting": False,
        "threshold_selection": "development-validation only; one common threshold across four seeds per K/guardrail",
        "guardrail_grid": {"minimum_plots": [20, 10, 0], "harm_numerator": [5, 7, 9], "harm_denominator": 10,
                           "minimum_coverage_fraction": 0.01, "positive_net_utility_required": True},
        "selection_objective": "maximize pooled rescues minus harms; equivalent to mean four-seed net corrections",
        "selection_ties": ["fewer harms", "fewer actions", "higher threshold"],
        "threshold_candidates": "typed never; zero; unique float64 midpoints between adjacent distinct positive eligible disagreement scores",
        "no_feasible_threshold": "typed never_intervene",
        "number_of_selected_policies": 18,
        "validation": {"images": 822, "plots": 325, "reference_images": 3378, "reference_plots": 1300, "expert_fit_images": 3378},
        "test": {"images": 1347, "plots": 531, "expert_fit_images": 4200,
                 "reference_banks": {"3378": {"images": 3378, "plots": 1300}, "4200": {"images": 4200, "plots": 1625}}},
        "test_threshold_adjustment": False, "test_policy_selection": False,
        "bootstrap": {"replicates": 2000, "random_seed": 20260930,
                      "unit": "habitat-stratified whole plots retaining all photos and four seeds",
                      "metrics": ["accuracy", "weighted_f1"],
                      "validation_contrasts": "each of 18 policies minus fusion and matching strict zero-threshold K",
                      "test_contrasts_per_bank": "each of 18 policies minus fusion, matching strict zero-threshold K, and RF",
                      "number_of_metric_intervals": 288},
        "historical_test_exposure": True, "adaptive_reuse": True, "independent_confirmation": False,
        "source_sha256": sources,
    })
    return declaration, receipt


def strict_scores(source: pd.DataFrame) -> pd.DataFrame:
    common = geometry.EXPERT_COLUMNS + [c for c in ("rf_pred", "rf_acted", *LABEL_COLUMNS) if c in source]
    common += ["reference_plots_fusion_class", "reference_plots_geo_class"]
    rows = source[common].copy()
    for k in KS:
        supported = (rows.reference_plots_fusion_class.ge(k) & rows.reference_plots_geo_class.ge(k)).to_numpy()
        scores = source[[f"k{k}_A_image", f"k{k}_A_geo"]].to_numpy(dtype=np.float64)
        if not np.isfinite(scores[supported]).all():
            raise ValueError("Strictly supported candidates lack finite geometry scores")
        if not np.array_equal(source[f"k{k}_effective"].to_numpy()[supported], np.full(supported.sum(), k)):
            raise ValueError("Supported geometry was not computed with exactly K plots")
        rows[f"k{k}_eligible"] = supported
        rows[f"k{k}_score"] = np.where(supported, scores.min(axis=1), np.nan)
        for modality in ("image", "geo"):
            rows[f"k{k}_A_{modality}"] = np.where(supported, source[f"k{k}_A_{modality}"], np.nan)
            for candidate in ("fusion", "geo"):
                column = f"k{k}_d_{modality}_{candidate}"
                rows[column] = np.where(supported, source[column], np.nan)
                column = f"k{k}_nearest_{modality}_{candidate}_plots"
                rows[column] = [list(v) if ok else [] for v, ok in zip(source[column], supported)]
        acted = rows.raw_pred.ne(rows.geo_pred) & supported & rows[f"k{k}_score"].gt(0)
        rows[f"zero_k{k}_acted"] = acted
        rows[f"zero_k{k}_pred"] = np.where(acted, rows.geo_pred, rows.raw_pred).astype(np.int64)
    return rows


def apply_frozen(rows: pd.DataFrame, policies: dict) -> pd.DataFrame:
    from joint_geometry_threshold_selection import apply_policy
    extra = {}
    for name, policy in policies.items():
        acted = apply_policy(rows, policy)
        extra[f"{name}_acted"] = acted
        extra[f"{name}_pred"] = np.where(acted, rows.geo_pred, rows.raw_pred).astype(np.int64)
        if np.any(acted & ~rows[f"zero_k{policy['k']}_acted"].to_numpy()):
            raise ValueError("Nonnegative threshold introduced new actions relative to zero")
    return pd.concat([rows, pd.DataFrame(extra, index=rows.index)], axis=1)


def load_test_scores(declaration: dict) -> dict[str, pd.DataFrame]:
    # This function is called only after the validation thresholds are sealed.
    loaded, inputs, previous_k5, _, _ = geometry.load_inputs()
    for path, expected in loaded["source_sha256"].items():
        if declaration["source_sha256"].get(path) != expected:
            raise ValueError(f"Undeclared or changed source during test application: {path}")
    refs, identities, experts, vectors = inputs["test"]
    mask = refs.development_role.eq("train").to_numpy()
    small_vectors = {m: (reference[mask], query) for m, (reference, query) in vectors.items()}
    with threadpool_limits(limits=1):
        small, _ = geometry.shared_k_predictions(refs.loc[mask], identities, experts, small_vectors)
    full = pd.read_parquet(PREVIOUS / "test_predictions.parquet").sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    old_small = previous_k5["test"]
    output = {"3378": strict_scores(small), "4200": strict_scores(full)}
    for bank, rows in output.items():
        for column in experts:
            if not np.array_equal(experts[column], rows[column]):
                raise ValueError("Test experts/RF changed between reference banks")
        for suffix in ("pred", "acted"):
            if not np.array_equal(rows[f"zero_k5_{suffix}"], old_small[f"g{bank}_k5_{suffix}"]):
                raise ValueError("Strict zero-threshold K5 does not reproduce the previous result")
        if any(column in rows for column in LABEL_COLUMNS):
            raise ValueError("Test labels present before canonical scoring")
    return output


def registry_event(output: Path, declaration: dict, manifest: dict, receipt: dict | None = None) -> None:
    bundle = manifest["manifest_sha256"]
    snapshot = REGISTRY / "snapshots" / declaration["identity_projection_sha256"]
    with exclusive_lock(REGISTRY):
        if receipt is None:
            first = snapshot / "first_score_event.json"
            if read_json(first)["identity_projection_sha256"] != declaration["identity_projection_sha256"]:
                raise ValueError("Registered test identity differs")
            reservation = {
                "study_id": STUDY_ID, "status": "reserved", "owner_output_root": str(output.resolve()),
                "prediction_manifest_sha256": bundle, "composite_bundle_file_sha256": bundle,
                "identity_projection_sha256": declaration["identity_projection_sha256"],
                "declaration_file_sha256": file_hash(output / "declaration.json"),
                "selection_manifest_file_sha256": file_hash(output / "selection_manifest.json"),
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
    output = parser.parse_args().output_dir.resolve()
    if (output / "receipt.json").exists():
        receipt = geometry.previous.sealed_previous(output)
        check_snapshot(receipt["preserved_registry_sha256"])
        geometry.previous.sealed_previous(PREVIOUS)
        registry_event(output, read_json(output / "declaration.json"), read_json(output / "prediction_manifest.json"), receipt)
        print("Verified completed validation-threshold study; no selection or rescoring.")
        return
    declaration, parent_receipt = load_declaration()
    implementation = dict(parent_receipt["implementation_sha256"])
    for path in (Path(__file__).resolve(), Path("tools/joint_geometry_threshold_selection.py").resolve(),
                 Path("tools/joint_geometry_threshold_report.py").resolve(),
                 Path("multimodal/geo_helpfulness_router_heuristics.py").resolve()):
        implementation[str(path)] = file_hash(path)
    registry_before = {str(p.resolve()): file_hash(p) for p in sorted(REGISTRY.rglob("*")) if p.is_file()}
    declaration["implementation_sha256"] = implementation
    declaration["declaration_sha256"] = canonical_hash(declaration)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "declaration.json", declaration)
    validation_source = pd.read_parquet(PREVIOUS / "validation_scored.parquet").sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    validation = strict_scores(validation_source)
    if not np.array_equal(validation.zero_k5_pred, validation_source.strict_k5_pred):
        raise ValueError("Validation strict K5 comparator changed")
    write_bytes(output / "validation_scores.parquet", table_bytes(pa.Table.from_pandas(validation, preserve_index=False)))
    from joint_geometry_threshold_selection import select_policies
    selection = select_policies(validation)
    if len(selection["policies"]) != 18:
        raise ValueError("Expected all 18 declared policies")
    write_json(output / "selection.json", selection)
    keys = ("k", "minimum_plots", "harm_numerator", "harm_denominator", "threshold_specification", "threshold_display", "selected_feasible")
    frozen = {name: {key: policy[key] for key in keys} for name, policy in selection["policies"].items()}
    write_json(output / "frozen_policies.json", frozen)
    selection_manifest = {
        "study_id": STUDY_ID, "declaration_sha256": declaration["declaration_sha256"],
        "validation_scores_sha256": file_hash(output / "validation_scores.parquet"),
        "selection_sha256": file_hash(output / "selection.json"), "frozen_policies_sha256": file_hash(output / "frozen_policies.json"),
        "selection_cohort": "development-validation", "test_outcomes_used": False,
    }
    selection_manifest["manifest_sha256"] = canonical_hash(selection_manifest)
    write_json(output / "selection_manifest.json", selection_manifest)
    print("All 18 validation thresholds frozen; applying both cleaned-test reference banks.", flush=True)
    # Reload the sealed deployment records; no selection is called below here.
    frozen = read_json(output / "frozen_policies.json")
    validation = apply_frozen(validation, frozen)
    tests = {bank: apply_frozen(rows, frozen) for bank, rows in load_test_scores(declaration).items()}
    write_bytes(output / "validation_scored.parquet", table_bytes(pa.Table.from_pandas(validation, preserve_index=False)))
    for bank, rows in tests.items():
        write_bytes(output / f"test_{bank}_predictions.parquet", table_bytes(pa.Table.from_pandas(rows, preserve_index=False)))
    manifest = {
        "study_id": STUDY_ID, "declaration_sha256": declaration["declaration_sha256"],
        "selection_manifest_file_sha256": file_hash(output / "selection_manifest.json"),
        "predictions_sha256": {bank: file_hash(output / f"test_{bank}_predictions.parquet") for bank in tests},
        "query_label_columns_present": False, "rows_per_bank": 5388, "policies_per_bank": 18,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(output / "prediction_manifest.json", manifest)
    registry_event(output, declaration, manifest)
    print("Test predictions sealed; acquiring canonical cleaned-test labels.", flush=True)
    tests["3378"], label_hashes = geometry.previous.prior.score_labels(output, tests["3378"], declaration)
    labels = tests["3378"].loc[tests["3378"].training_seed.eq(1), ["row_uid", *LABEL_COLUMNS]]
    tests["4200"] = tests["4200"].merge(labels, on="row_uid", how="left", validate="many_to_one")
    for bank, rows in tests.items():
        if rows.label_id_dense.isna().any():
            raise ValueError("Incomplete canonical label join")
        write_bytes(output / f"test_{bank}_scored.parquet", table_bytes(pa.Table.from_pandas(rows, preserve_index=False)))
    from joint_geometry_threshold_report import write_results
    with threadpool_limits(limits=1):
        result = write_results(output, validation, tests, selection, declaration)
    hashes = {**declaration["source_sha256"], **label_hashes}
    for snapshot in (hashes, implementation, registry_before):
        check_snapshot(snapshot)
    geometry.previous.sealed_previous(PREVIOUS)
    receipt = {
        "study_id": STUDY_ID, "source_sha256": hashes, "implementation_sha256": implementation,
        "declaration_sha256": declaration["declaration_sha256"], "selection_manifest_sha256": selection_manifest["manifest_sha256"],
        "prediction_manifest_sha256": manifest["manifest_sha256"], "result": result,
        "preserved_registry_sha256": registry_before, "historical_test_exposure": True, "independent_confirmation": False,
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    write_json(output / "receipt.json", receipt)
    registry_event(output, declaration, manifest, receipt)
    print("Completed geometry-threshold study: " + str(output / "report.md"), flush=True)


if __name__ == "__main__":
    main()

"""Apply the frozen joint-geometry rule to the September 15 expert family.

Predictions are sealed and an append-only event reserved before labels are read.
No expert/router fitting, reference-bank expansion, or threshold search occurs.
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

from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash, check_self_hash, check_snapshot, exclusive_lock, file_hash,
    read_json, table_bytes, write_bytes, write_json,
)
from multimodal.geo_helpfulness_router_metrics import classification_metrics
from plot_multimodal_embedding_geometry import CACHE, PROTOCOL, load_development


STUDY_ID = "joint_geometry_cleaned_test_v1"
BASE = Path("multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness")
RF = BASE / "relaxed_guardrails_rf_v1"
REGISTRY = Path("multimodal_artifacts/locked_test_registry/cs/gse_100m_cleaned_test/events")
VALIDATION = Path("dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_joint_geometry_policy_validation")
RF_POLICY = "relaxed__rf__primary_full727__plots_20__harm_0p7"
IMAGE_COLUMNS = [f"I{i:04d}" for i in range(1152)]
GEO_COLUMNS = [f"A{i:02d}" for i in range(64)]
EPSILON = 1e-8


def bind(path: Path, hashes: dict, expected: str | None = None) -> None:
    value = file_hash(path)
    if expected is not None and value != expected:
        raise ValueError(f"Source fingerprint mismatch: {path}")
    hashes[str(path.resolve())] = value


def read_stage(folder: Path, hashes: dict) -> dict:
    manifest = read_json(folder / "manifest.json")
    check_self_hash(manifest)
    bind(folder / "manifest.json", hashes)
    return manifest


def load_sources() -> tuple:
    hashes = {}
    parent = read_json(RF / "declaration.json")
    check_self_hash(parent, "declaration_sha256")
    bind(RF / "declaration.json", hashes)
    identity = parent["identity_manifest"]
    check_self_hash(identity)
    identities = pd.DataFrame(identity["rows"]).sort_values("row_uid").reset_index(drop=True)
    if len(identities) != 1347 or identities.plot_idx.nunique() != 531:
        raise ValueError("Unexpected cleaned-test universe")
    prediction_manifest = read_stage(RF / "predictions", hashes)
    selection_manifest = read_stage(RF / "selection", hashes)
    score_manifest = read_stage(RF / "scores", hashes)
    label_source = {
        "path": str((RF / "scores/labels.parquet").resolve()),
        "file_sha256": score_manifest["artifacts"]["labels.parquet"]["file_sha256"],
    }
    if prediction_manifest["declaration_sha256"] != parent["declaration_sha256"]:
        raise ValueError("Historical prediction/declaration linkage mismatch")
    if prediction_manifest["selection_manifest_sha256"] != selection_manifest["manifest_sha256"]:
        raise ValueError("Historical prediction/selection linkage mismatch")
    if (score_manifest["declaration_sha256"] != parent["declaration_sha256"]
            or score_manifest["prediction_manifest_sha256"] != prediction_manifest["manifest_sha256"]):
        raise ValueError("Historical score/prediction/declaration linkage mismatch")
    for name in ("experts.parquet", "actions.parquet"):
        bind(RF / "predictions" / name, hashes, prediction_manifest["artifacts"][name]["file_sha256"])
    frozen = RF / "selection/frozen_policies.json"
    bind(frozen, hashes, selection_manifest["artifacts"][frozen.name]["file_sha256"])
    policy = read_json(frozen)[RF_POLICY]
    expert_columns = ["row_uid", "file", "plot_idx", "training_seed",
                      "image_pred", "geo_pred", "raw_pred", "image_logits", "geo_logits", "raw_logits"]
    experts = pd.read_parquet(RF / "predictions/experts.parquet", columns=expert_columns,
                             filters=[("checkpoint_family", "=", "best_model")])
    experts = experts.sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    actions = pd.read_parquet(RF / "predictions/actions.parquet", filters=[
        ("checkpoint_family", "=", "best_model"), ("policy_id", "=", RF_POLICY)])
    actions = actions.sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    if len(experts) != 5388 or set(experts.training_seed) != {1, 2, 3, 4}:
        raise ValueError("Expected all 1,347 images under all four historical seeds")
    if experts.duplicated(["training_seed", "row_uid"]).any():
        raise ValueError("Duplicate expert query")
    for seed, group in experts.groupby("training_seed"):
        for column in ("row_uid", "file", "plot_idx"):
            if not np.array_equal(group[column], identities[column]):
                raise ValueError(f"Expert identities mismatch: seed {seed}, {column}")
    for modality in ("image", "geo", "raw"):
        logits = np.stack(experts.pop(f"{modality}_logits")).astype(np.float64)
        if logits.shape != (5388, 18) or not np.isfinite(logits).all():
            raise ValueError("Malformed native logits")
        if not np.array_equal(logits.argmax(axis=1), experts[f"{modality}_pred"]):
            raise ValueError("Native prediction/argmax mismatch")
    for column in ("training_seed", "row_uid", "plot_idx", "raw_pred", "geo_pred"):
        if not np.array_equal(experts[column], actions[column]):
            raise ValueError("RF actions are not aligned with the historical experts")
    threshold = float.fromhex(policy["threshold_specification"]["value_hex"])
    expected_action = experts.raw_pred.ne(experts.geo_pred) & actions.score.gt(threshold)
    if not np.array_equal(expected_action, actions.acted):
        raise ValueError("Historical RF strict threshold reproduction failed")
    if not np.array_equal(np.where(actions.acted, experts.geo_pred, experts.raw_pred), actions.final_pred):
        raise ValueError("Historical RF corrected predictions mismatch")
    experts["rf_pred"] = actions.final_pred
    experts["rf_acted"] = actions.acted

    refs, image, geo, reference_hashes = load_development("train")
    for name, value in reference_hashes.items():
        bind(Path(name), hashes, value)
    if len(refs) != 3378 or refs.plot_idx.nunique() != 1300:
        raise ValueError("The frozen development-training reference bank changed")
    if set(refs.plot_idx) & set(identities.plot_idx):
        raise ValueError("Reference/query plots overlap")
    validation_manifest = read_json(VALIDATION / "manifest.json")
    bind(VALIDATION / "manifest.json", hashes)
    bind(VALIDATION / "declaration.json", hashes,
         validation_manifest["output_sha256"]["declaration.json"])
    previous = read_json(VALIDATION / "declaration.json")
    if previous["distance_k"] != 1 or previous["ties_or_missing"] != "retain fusion":
        raise ValueError("Previous validation policy differs")

    query_path = CACHE / "test.parquet"
    bind(query_path, hashes)
    # Deliberately project only identities and embeddings, ignoring old test labels.
    query = pd.read_parquet(query_path, columns=["file", "plot_idx", *IMAGE_COLUMNS, *GEO_COLUMNS])
    if query.file.duplicated().any():
        raise ValueError("Duplicate source embedding identity")
    source_count = len(query)
    query = query.set_index("file").loc[identities.file].reset_index()
    if not np.array_equal(query.plot_idx, identities.plot_idx):
        raise ValueError("Frozen embedding/cleaned identity mismatch")
    vectors = {}
    for modality, bank, columns in (("image", image, IMAGE_COLUMNS), ("geo", geo, GEO_COLUMNS)):
        q = query[columns].to_numpy(dtype=np.float64)
        for matrix in (bank, q):
            if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
                raise ValueError("Invalid embedding values or norms")
        vectors[modality] = (bank / np.linalg.norm(bank, axis=1, keepdims=True),
                             q / np.linalg.norm(q, axis=1, keepdims=True))
    declaration = {
        "study_id": STUDY_ID, "user_authorized_cleaned_test_evaluation": True,
        "historical_test_exposure": True, "adaptive_reuse": True, "independent_confirmation": False,
        "expert_family": "September 15 legacy best_model; fixed epoch 50; full 4200 development images",
        "expert_fit_images": 4200, "expert_fit_plots": 1625,
        "reference_images": 3378, "reference_plots": 1300,
        "reference_bank": "unchanged protocol development_train only; validation not added",
        "reference_row_uids_sha256": canonical_hash(sorted(refs.row_uid.tolist())),
        "reference_encoder": "historical frozen pretrained SigLIP2 cache",
        "queries": 1347, "query_plots": 531, "training_seeds": [1, 2, 3, 4],
        "query_cache_rows": source_count, "query_cache_rows_excluded_by_locked_identity": source_count - 1347,
        "distance_k": 1, "distance": "cosine after independent L2 normalization",
        "class_distance": "nearest reference plot; minimum over that plot's photos",
        "epsilon": EPSILON, "score": "log((d_fusion + epsilon)/(d_geo + epsilon))",
        "action": "raw_pred != geo_pred AND finite(A_image, A_geo) AND A_image > 0 AND A_geo > 0",
        "ties_or_missing": "retain fusion", "model_fitting": False, "threshold_selection": False,
        "rf_policy_id": RF_POLICY, "rf_frozen_policy": policy, "ontology": parent["ontology"],
        "identity_projection_sha256": identity["identity_projection_sha256"],
        "label_source": label_source,
        "bootstrap": {"replicates": 2000, "random_seed": 20260930,
                      "unit": "habitat-stratified whole plots retaining all images and four seeds",
                      "comparisons": ["geometry minus fusion", "geometry minus RF"],
                      "metrics": ["accuracy", "weighted_f1"]},
        "source_sha256": hashes,
        "limitations": [
            "frozen image cache retains historical preprocessing; no encoder replay performed",
            "legacy expert cache-to-encoder lineage lacks an end-to-end cryptographic binding",
            "distinct test and reference plots may share satellite vectors",
            "test experts fit all 4200 development images; validation experts fit 3378",
        ],
    }
    return declaration, refs, identities, experts, vectors


def geometry_predictions(refs: pd.DataFrame, identities: pd.DataFrame,
                         experts: pd.DataFrame, vectors: dict) -> tuple[pd.DataFrame, dict]:
    class_ids = refs.label_id_dense.to_numpy(dtype=int)
    reference_support = refs.groupby("label_id_dense").plot_idx.nunique().reindex(range(18), fill_value=0)
    indices = pd.Series(np.arange(len(identities)), index=identities.row_uid).loc[experts.row_uid].to_numpy()
    raw = experts.raw_pred.to_numpy(dtype=int)
    geo = experts.geo_pred.to_numpy(dtype=int)
    rows = experts.copy()
    for modality, (reference, query) in vectors.items():
        distances = np.clip(1.0 - query @ reference.T, 0.0, 2.0)
        class_distance = np.full((len(query), 18), np.nan)
        nearest_index = np.full((len(query), 18), -1, dtype=int)
        for c in range(18):
            candidates = np.flatnonzero(class_ids == c)
            if len(candidates):
                best = candidates[np.argmin(distances[:, candidates], axis=1)]
                nearest_index[:, c] = best
                class_distance[:, c] = distances[np.arange(len(query)), best]
        dr, dg = class_distance[indices, raw], class_distance[indices, geo]
        rows[f"d_{modality}_fusion"] = dr
        rows[f"d_{modality}_geo"] = dg
        rows[f"A_{modality}"] = np.log((dr + EPSILON) / (dg + EPSILON))
        for candidate, pred in (("fusion", raw), ("geo", geo)):
            nearest = nearest_index[indices, pred]
            rows[f"nearest_{modality}_{candidate}_reference_uid"] = [
                refs.iloc[i].row_uid if i >= 0 else None for i in nearest]
    rows["reference_plots_fusion_class"] = reference_support.to_numpy()[raw]
    rows["reference_plots_geo_class"] = reference_support.to_numpy()[geo]
    rows["valid_geometry"] = np.isfinite(rows[["A_image", "A_geo"]]).all(axis=1)
    rows["geometry_acted"] = (rows.raw_pred.ne(rows.geo_pred) & rows.valid_geometry
                              & rows.A_image.gt(0) & rows.A_geo.gt(0))
    rows["geometry_pred"] = np.where(rows.geometry_acted, rows.geo_pred, rows.raw_pred).astype(np.int64)
    if not rows.loc[rows.raw_pred.eq(rows.geo_pred) & rows.valid_geometry,
                    ["A_image", "A_geo"]].eq(0).all().all():
        raise ValueError("Agreed candidates must have zero scores")
    return rows, {str(c): int(n) for c, n in reference_support.items()}


def reserve(output: Path, declaration: dict, prediction_manifest: dict) -> tuple[dict, dict]:
    snapshot = REGISTRY / "snapshots" / declaration["identity_projection_sha256"]
    bundle = prediction_manifest["manifest_sha256"]
    paths = {"first": snapshot / "first_score_event.json",
             "reservation": REGISTRY / "bundles" / f"{bundle}.json",
             "event": snapshot / f"{bundle}.json",
             "completion": REGISTRY / "bundles" / f"{bundle}.completed.json"}
    owner = {"study_id": STUDY_ID, "status": "reserved", "owner_output_root": str(output.resolve()),
             "prediction_manifest_sha256": bundle, "composite_bundle_file_sha256": bundle,
             "identity_projection_sha256": declaration["identity_projection_sha256"],
             "declaration_file_sha256": file_hash(output / "declaration.json"),
             "historical_test_exposure": True, "adaptive_reuse": True}
    with exclusive_lock(REGISTRY):
        first = read_json(paths["first"])
        if first["identity_projection_sha256"] != owner["identity_projection_sha256"]:
            raise ValueError("Registered snapshot differs")
        reservation = {**owner, "prior_registered_score_event": True,
                       "first_score_event_file_sha256": file_hash(paths["first"])}
        write_json(paths["reservation"], reservation)
        write_json(output / "reservation.json", reservation)
    return reservation, paths


def score_labels(output: Path, rows: pd.DataFrame, declaration: dict) -> tuple[pd.DataFrame, dict]:
    hashes = {}
    check_snapshot(declaration["source_sha256"])
    manifest = read_stage(RF / "scores", hashes)
    label_path = Path(declaration["label_source"]["path"])
    if manifest["artifacts"]["labels.parquet"]["file_sha256"] != declaration["label_source"]["file_sha256"]:
        raise ValueError("Canonical label source differs from declaration")
    bind(label_path, hashes, declaration["label_source"]["file_sha256"])
    labels = pd.read_parquet(label_path).sort_values("row_uid").reset_index(drop=True)
    identity = rows.loc[rows.training_seed.eq(1), ["row_uid", "file", "plot_idx"]]
    if len(labels) != 1347 or labels.row_uid.duplicated().any():
        raise ValueError("Cleaned label universe differs")
    for column in identity:
        if not np.array_equal(labels[column], identity[column]):
            raise ValueError("Cleaned labels/predictions are not aligned")
    ontology = pd.DataFrame(declaration["ontology"]).set_index("dense_id")
    for column in ("canonical_l3_id", "label_name"):
        if not np.array_equal(labels[column], labels.label_id_dense.map(ontology[column])):
            raise ValueError("Cleaned ontology mismatch")
    write_bytes(output / "labels.parquet", label_path.read_bytes())
    scored = rows.merge(labels.drop(columns=["file", "plot_idx"]), on="row_uid", how="left", validate="many_to_one")
    classification_path = RF / "scores/best_model/classification.json"
    bind(classification_path, hashes, manifest["artifacts"]["best_model/classification.json"]["file_sha256"])
    historical = read_json(classification_path)
    comparisons = {}
    for name, column, target in (("image", "image_pred", historical["experts"]["image"]),
                                ("geo", "geo_pred", historical["experts"]["geo"]),
                                ("fusion", "raw_pred", historical["experts"]["raw"]),
                                ("rf", "rf_pred", historical["policies"][RF_POLICY])):
        comparisons[name] = {}
        for seed, data in scored.groupby("training_seed"):
            observed = classification_metrics(data[column].to_numpy(), data.label_id_dense.to_numpy())
            expected = target["per_seed"][str(seed)]
            if observed["confusion_matrix"] != expected["confusion_matrix"]:
                raise ValueError("Historical confusion counts do not reproduce")
            for metric in ("accuracy", "weighted_f1", "macro_f1", "mcc"):
                if not np.isclose(observed[metric], expected[metric], atol=1e-14, rtol=0):
                    raise ValueError(f"Historical metric differs: {name}/{seed}/{metric}")
            comparisons[name][str(seed)] = {k: observed[k] for k in ("accuracy", "weighted_f1", "macro_f1", "mcc")}
    write_json(output / "historical_comparator_reproduction.json", comparisons)
    write_json(output / "label_source_receipt.json", {
        "source_sha256": hashes, "label_copy_sha256": file_hash(output / "labels.parquet"),
        "reservation_sha256": file_hash(output / "reservation.json"),
        "query_cache_labels_decoded": False,
    })
    return scored, hashes


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
        check_snapshot(receipt["source_sha256"])
        check_snapshot(receipt["implementation_sha256"])
        for name, expected in receipt["output_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError(f"Completed result changed: {name}")
        check_snapshot(receipt["preserved_registry_sha256"])
        complete_event(output, receipt)
        print("Verified existing evaluation and registry completion; no rescoring.")
        return
    output.mkdir(parents=True, exist_ok=True)
    declaration, refs, identities, experts, vectors = load_sources()
    registry_before = {str(p.resolve()): file_hash(p) for p in sorted(REGISTRY.rglob("*")) if p.is_file()}
    code_paths = [Path(__file__).resolve(), Path("tools/joint_geometry_test_report.py").resolve(),
                  Path("tools/plot_multimodal_embedding_geometry.py").resolve(),
                  Path("multimodal/geo_helpfulness_router_metrics.py").resolve(),
                  Path("multimodal/geo_helpfulness_router_m4_common.py").resolve()]
    code_hashes = {str(p): file_hash(p) for p in code_paths}
    declaration["implementation_sha256"] = code_hashes
    declaration["declaration_sha256"] = canonical_hash(declaration)
    write_json(output / "declaration.json", declaration)
    with threadpool_limits(limits=1):
        predictions, support = geometry_predictions(refs, identities, experts, vectors)
    prediction_path = output / "predictions.parquet"
    write_bytes(prediction_path, table_bytes(pa.Table.from_pandas(predictions, preserve_index=False)))
    prediction_manifest = {"study_id": STUDY_ID, "declaration_sha256": declaration["declaration_sha256"],
                           "predictions_sha256": file_hash(prediction_path), "rows": len(predictions),
                           "label_columns_present": False, "reference_class_plot_support": support}
    prediction_manifest["manifest_sha256"] = canonical_hash(prediction_manifest)
    write_json(output / "prediction_manifest.json", prediction_manifest)
    reserve(output, declaration, prediction_manifest)
    print("Geometry predictions sealed and score event reserved; acquiring canonical cleaned labels.", flush=True)
    scored, label_hashes = score_labels(output, predictions, declaration)
    write_bytes(output / "scored_predictions.parquet", table_bytes(pa.Table.from_pandas(scored, preserve_index=False)))
    from joint_geometry_test_report import write_results
    result = write_results(output, scored, declaration)
    hashes = {**declaration["source_sha256"], **label_hashes}
    check_snapshot(hashes)
    check_snapshot(code_hashes)
    receipt = {"study_id": STUDY_ID, "source_sha256": hashes, "implementation_sha256": code_hashes,
               "prediction_manifest_sha256": prediction_manifest["manifest_sha256"],
               "declaration_sha256": declaration["declaration_sha256"],
               "result": result, "historical_test_exposure": True, "independent_confirmation": False,
               "preserved_registry_sha256": registry_before,
               "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    check_snapshot(registry_before)
    receipt["receipt_sha256"] = canonical_hash(receipt)
    write_json(output / "receipt.json", receipt)
    complete_event(output, receipt)
    print("Completed matched cleaned-test evaluation; results: " + str(output / "report.md"), flush=True)


if __name__ == "__main__":
    main()

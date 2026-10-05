"""Compare dev-train-adapted image geometry with the frozen pretrained study.

Only validation and cleaned test are evaluated. Expert predictions, geo geometry,
reference identities, strict support and threshold selection are held fixed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_joint_geometry_k_sweep as geometry
import evaluate_joint_geometry_thresholds as thresholds
from multimodal.geo_helpfulness_router_m4_common import (
    canonical_hash, check_self_hash, check_snapshot, exclusive_lock, file_hash,
    read_json, table_bytes, write_bytes, write_json,
)
from plot_multimodal_embedding_geometry import CACHE, load_development


STUDY_ID = "joint_geometry_adapted_encoder_v1"
PREVIOUS = Path("dev_plans/reliability_aware_geo_fusion/reports/2026-09-30_joint_geometry_thresholds")
EMBEDDINGS = Path("multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/adapted_geometry_v1/embeddings")
REGISTRY = geometry.REGISTRY
SEEDS = (1, 2, 3, 4)
KS = (5, 10)
LABEL_COLUMNS = thresholds.LABEL_COLUMNS


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms == 0):
        raise ValueError("Invalid embedding values or norms")
    return values / norms


def bind(path: Path, sources: dict, expected: str | None = None) -> None:
    geometry.previous.prior.bind(path, sources, expected)


def load_embedding_arrays(root: Path, sources: dict) -> tuple[dict, dict]:
    """Read authenticated per-seed arrays; never combine encoder spaces."""
    manifest = read_json(root / "manifest.json")
    check_self_hash(manifest, "manifest_sha256")
    expected_fit = {"images": 3378, "plots": 1300, "stage": "development_train_to_validation"}
    if (manifest["training_seeds"] != list(SEEDS) or manifest["encoder_fit"] != expected_fit
            or not manifest["same_encoder_for_reference_and_query"]
            or manifest["test_labels_read"] or manifest["test_outcomes_read"]
            or manifest["oof_diagnostics"]):
        raise ValueError("Adapted extraction differs from the approved experiment")
    bind(root / "manifest.json", sources)
    for key in ("source_sha256", "implementation_sha256"):
        check_snapshot(manifest[key])
        sources.update(manifest[key])
    for name, expected in manifest["output_sha256"].items():
        bind(root / name, sources, expected)
    arrays = {}
    for seed in SEEDS:
        summary = manifest["seeds"][str(seed)]
        check_self_hash(summary, "manifest_sha256")
        if (summary["training_seed"] != seed or summary["encoder_fitting_rows"] != 3378
                or summary["encoder_fitting_plots"] != 1300
                or not summary["checkpoint_validation_replay"]["valid"]):
            raise ValueError("Adapted seed provenance or validation replay differs")
        checkpoint = str(Path(summary["checkpoint_path"]).resolve())
        if manifest["source_sha256"].get(checkpoint) != summary["checkpoint_sha256"]:
            raise ValueError("Seed-to-encoder checkpoint binding is missing")
        path = root / f"seed_{seed}" / "embeddings.npz"
        if path.relative_to(root).as_posix() not in manifest["output_sha256"]:
            raise ValueError("Undeclared embedding array")
        with np.load(path, allow_pickle=False) as saved:
            ids = pd.DataFrame({name: saved[name].astype(str) for name in ("row_uid", "file", "plot_idx", "cohort")})
            features = saved["image_embedding"]
        if ids.row_uid.duplicated().any() or features.shape != (5547, 1152):
            raise ValueError("Unexpected embedding identities or dimensions")
        if ids.cohort.value_counts().to_dict() != {"development": 4200, "test": 1347}:
            raise ValueError("Embedding cohort counts differ")
        if not np.allclose(np.linalg.norm(features.astype(np.float64), axis=1), 1, atol=2e-6, rtol=0):
            raise ValueError("Expected L2-normalized adapted embeddings")
        arrays[seed] = (ids, normalized(features))
    return arrays, manifest


def align_embeddings(array: tuple, identities: pd.DataFrame, cohort: str) -> np.ndarray:
    metadata, vectors = array
    positions = pd.Series(np.arange(len(metadata)), index=metadata.row_uid).loc[identities.row_uid].to_numpy()
    aligned = metadata.iloc[positions].reset_index(drop=True)
    for column in ("row_uid", "file", "plot_idx"):
        if not np.array_equal(aligned[column], identities[column].to_numpy()):
            raise ValueError(f"Adapted embedding identity mismatch: {column}")
    if not aligned.cohort.eq(cohort).all():
        raise ValueError("Adapted embedding cohort mismatch")
    return vectors[positions]


def score_cohort(refs: pd.DataFrame, identities: pd.DataFrame, previous: pd.DataFrame,
                 reference_geo: np.ndarray, query_geo: np.ndarray,
                 arrays: dict, cohort: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        old = previous.loc[previous.training_seed.eq(seed)].reset_index(drop=True)
        if not np.array_equal(old.row_uid, identities.row_uid):
            raise ValueError("Previous predictions and queries are not aligned")
        columns = geometry.EXPERT_COLUMNS + [c for c in ("rf_pred", "rf_acted", *LABEL_COLUMNS) if c in old]
        experts = old[columns].copy()
        vectors = {
            "image": (align_embeddings(arrays[seed], refs, "development"),
                      align_embeddings(arrays[seed], identities, cohort)),
            "geo": (reference_geo, query_geo),
        }
        raw, _ = geometry.shared_k_predictions(refs, identities, experts, vectors)
        current = thresholds.strict_scores(raw)
        for column in columns + ["reference_plots_fusion_class", "reference_plots_geo_class"]:
            if not np.array_equal(current[column], old[column]):
                raise ValueError(f"Matched experiment changed {column}")
        for k in KS:
            if not np.array_equal(current[f"k{k}_eligible"], old[f"k{k}_eligible"]):
                raise ValueError("Strict support changed between image representations")
            for suffix in ("A_geo", "d_geo_fusion", "d_geo_geo"):
                column = f"k{k}_{suffix}"
                if not np.allclose(current[column], old[column], atol=1e-12, rtol=0, equal_nan=True):
                    raise ValueError(f"Geo geometry changed: {column}")
            for candidate in ("fusion", "geo"):
                column = f"k{k}_nearest_geo_{candidate}_plots"
                if [list(v) for v in current[column]] != [list(v) for v in old[column]]:
                    raise ValueError("Nearest geo reference plots changed")
        frames.append(current)
        print(f"Computed {cohort}, {len(refs)} references, adapted seed {seed}.", flush=True)
    return pd.concat(frames, ignore_index=True)


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


def load_declaration(embedding_root: Path) -> tuple:
    previous = geometry.previous.sealed_previous(PREVIOUS)
    declaration = deepcopy(read_json(PREVIOUS / "declaration.json"))
    check_self_hash(declaration, "declaration_sha256")
    declaration.pop("declaration_sha256")
    sources = dict(previous["source_sha256"])
    for name in ("receipt.json", "declaration.json", "frozen_policies.json", "validation_scored.parquet",
                 "test_3378_predictions.parquet", "test_4200_predictions.parquet",
                 "test_3378_scored.parquet", "test_4200_scored.parquet"):
        bind(PREVIOUS / name, sources)
    arrays, embedding_manifest = load_embedding_arrays(embedding_root, sources)
    refs, _, geo, development_sources = load_development("all")
    for name, expected in development_sources.items():
        bind(Path(name), sources, expected)
    if (len(refs), refs.plot_idx.nunique()) != (4200, 1625):
        raise ValueError("Unexpected development universe")
    geo = normalized(geo)
    declaration.update({
        "study_id": STUDY_ID,
        "user_authorization": "Approved adapted-encoder comparison with validation and both cleaned-test banks; explicitly exclude OOF diagnostics",
        "reference_encoder": "habitat-adapted SigLIP2; per-seed development_train_to_validation tower fitted on 3378 images",
        "query_encoder": "same per-seed development-train-fitted tower as its reference bank",
        "encoder_fit_images": 3378, "encoder_fit_plots": 1300,
        "encoder_fit_includes_validation": False, "encoder_refit_for_full_bank": False,
        "out_of_fold_diagnostics": False, "adapted_encoder_retraining": False,
        "embedding_manifest_sha256": embedding_manifest["manifest_sha256"],
        "embedding_root": str(embedding_root.resolve()),
        "paired_pretrained_study": str(PREVIOUS.resolve()),
        "paired_pretrained_receipt_sha256": previous["receipt_sha256"],
        "reference_row_uids_sha256": {
            "3378": canonical_hash(sorted(refs.loc[refs.development_role.eq("train"), "row_uid"].tolist())),
            "4200": canonical_hash(sorted(refs.row_uid.tolist())),
        },
        "matched_factors": ["expert predictions", "geo vectors", "reference and query identities", "plot distance", "K", "strict support", "threshold fitting grid"],
        "encoder_seed_handling": "adapted encoder seed matched to expert seed; distances never mix encoders",
        "preprocessing": "legacy OpenCV BGR, force square 439, native OpenCLIP evaluation transform; preserve historical extraction convention",
        "threshold_selection": "refit on development-validation for adapted geometry; same pooled four-seed objective and guards as pretrained",
        "source_sha256": sources,
        "limitations": [
            "Pretrained comparator reuses authenticated historical caches; their original extraction checkpoint lacks complete cryptographic lineage",
            "Adapted inference preserves legacy BGR/439 preprocessing rather than introducing a simultaneous RGB change",
            "Test experts are historical full-development-fitted models; geometry encoders are fitted only on development-train",
            "Four-seed results share query plots; seeds are not independent samples",
            "Previously examined cleaned test is an exploratory transfer comparison, not independent confirmation",
        ],
    })
    declaration["bootstrap"].update({
        "encoder_contrasts": "adapted minus pretrained for two zero-threshold rules and all 18 separately validation-fitted policies, in all three cohorts",
        "encoder_metric_intervals": 120, "number_of_metric_intervals": 408,
        "threshold_selection_refitted_in_bootstrap": False,
    })
    implementation = dict(previous["implementation_sha256"])
    for path in (Path(__file__).resolve(), Path("tools/joint_geometry_adapted_report.py").resolve(),
                 Path("tools/extract_adapted_geometry_embeddings.py").resolve()):
        implementation[str(path)] = file_hash(path)
    declaration["implementation_sha256"] = implementation
    declaration["declaration_sha256"] = canonical_hash(declaration)
    return declaration, arrays, refs, geo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, default=EMBEDDINGS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "receipt.json").exists():
        receipt = geometry.previous.sealed_previous(output)
        check_snapshot(receipt["preserved_registry_sha256"])
        registry_event(output, read_json(output / "declaration.json"), read_json(output / "prediction_manifest.json"), receipt)
        print("Verified completed adapted-encoder study; no selection or rescoring.")
        return
    declaration, arrays, refs, geo = load_declaration(args.embedding_dir)
    registry_before = {str(p.resolve()): file_hash(p) for p in sorted(REGISTRY.rglob("*")) if p.is_file()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "declaration.json", declaration)
    train = refs.development_role.eq("train").to_numpy()
    validation_refs = refs.loc[train].reset_index(drop=True)
    validation_ids = refs.loc[~train, ["row_uid", "file", "plot_idx"]].reset_index(drop=True)
    if (len(validation_refs), len(validation_ids)) != (3378, 822):
        raise ValueError("Development split changed")
    old_validation = pd.read_parquet(PREVIOUS / "validation_scored.parquet").sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
    with threadpool_limits(limits=1):
        validation = score_cohort(validation_refs, validation_ids, old_validation, geo[train], geo[~train], arrays, "development")
    write_bytes(output / "validation_scores.parquet", table_bytes(pa.Table.from_pandas(validation, preserve_index=False)))
    from joint_geometry_threshold_selection import select_policies
    selection = select_policies(validation)
    if len(selection["policies"]) != 18:
        raise ValueError("Expected all 18 declared threshold policies")
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
    print("All adapted-geometry thresholds frozen on validation; applying cleaned-test banks.", flush=True)
    frozen = read_json(output / "frozen_policies.json")
    validation = thresholds.apply_frozen(validation, frozen)
    old_test = {bank: pd.read_parquet(PREVIOUS / f"test_{bank}_predictions.parquet").sort_values(["training_seed", "row_uid"]).reset_index(drop=True)
                for bank in ("3378", "4200")}
    test_ids = old_test["3378"].loc[old_test["3378"].training_seed.eq(1), ["row_uid", "file", "plot_idx"]].reset_index(drop=True)
    if (len(test_ids), test_ids.plot_idx.nunique()) != (1347, 531):
        raise ValueError("Cleaned test identity changed")
    geo_columns = [f"A{i:02d}" for i in range(64)]
    query = pd.read_parquet(CACHE / "test.parquet", columns=["file", "plot_idx", *geo_columns])
    if query.file.duplicated().any():
        raise ValueError("Duplicate test source file")
    query = query.set_index("file").loc[test_ids.file].reset_index()
    if not np.array_equal(query.plot_idx, test_ids.plot_idx):
        raise ValueError("Test geo identities differ")
    query_geo = normalized(query[geo_columns].to_numpy())
    tests = {}
    with threadpool_limits(limits=1):
        for bank, mask in (("3378", train), ("4200", np.ones(len(refs), dtype=bool))):
            if any(c in old_test[bank] for c in LABEL_COLUMNS):
                raise ValueError("Test source unexpectedly contains labels")
            current = score_cohort(refs.loc[mask].reset_index(drop=True), test_ids, old_test[bank], geo[mask], query_geo, arrays, "test")
            tests[bank] = thresholds.apply_frozen(current, frozen)
    for k in KS:
        for modality in ("image", "geo"):
            for candidate in ("fusion", "geo"):
                column = f"k{k}_d_{modality}_{candidate}"
                if np.any(tests["4200"][column].to_numpy() > tests["3378"][column].to_numpy() + 1e-12):
                    raise ValueError("Expanded bank increased a mean-K distance")
    write_bytes(output / "validation_scored.parquet", table_bytes(pa.Table.from_pandas(validation, preserve_index=False)))
    for bank, rows in tests.items():
        write_bytes(output / f"test_{bank}_predictions.parquet", table_bytes(pa.Table.from_pandas(rows, preserve_index=False)))
    manifest = {
        "study_id": STUDY_ID, "declaration_sha256": declaration["declaration_sha256"],
        "selection_manifest_file_sha256": file_hash(output / "selection_manifest.json"),
        "predictions_sha256": {bank: file_hash(output / f"test_{bank}_predictions.parquet") for bank in tests},
        "query_label_columns_present": False, "rows_per_bank": 5388, "policies_per_bank": 18,
        "embedding_manifest_sha256": declaration["embedding_manifest_sha256"],
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(output / "prediction_manifest.json", manifest)
    registry_event(output, declaration, manifest)
    print("Test predictions sealed; acquiring canonical labels for scoring.", flush=True)
    tests["3378"], label_hashes = geometry.previous.prior.score_labels(output, tests["3378"], declaration)
    labels = tests["3378"].loc[tests["3378"].training_seed.eq(1), ["row_uid", *LABEL_COLUMNS]]
    tests["4200"] = tests["4200"].merge(labels, on="row_uid", how="left", validate="many_to_one")
    for bank, rows in tests.items():
        if rows.label_id_dense.isna().any():
            raise ValueError("Incomplete canonical label join")
        write_bytes(output / f"test_{bank}_scored.parquet", table_bytes(pa.Table.from_pandas(rows, preserve_index=False)))
    from joint_geometry_adapted_report import write_results
    with threadpool_limits(limits=1):
        result = write_results(output, validation, tests, selection, declaration)
    sources = {**declaration["source_sha256"], **label_hashes}
    implementation = declaration["implementation_sha256"]
    for snapshot in (sources, implementation, registry_before):
        check_snapshot(snapshot)
    geometry.previous.sealed_previous(PREVIOUS)
    receipt = {
        "study_id": STUDY_ID, "source_sha256": sources, "implementation_sha256": implementation,
        "declaration_sha256": declaration["declaration_sha256"], "selection_manifest_sha256": selection_manifest["manifest_sha256"],
        "prediction_manifest_sha256": manifest["manifest_sha256"], "result": result,
        "preserved_registry_sha256": registry_before, "historical_test_exposure": True, "independent_confirmation": False,
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    write_json(output / "receipt.json", receipt)
    registry_event(output, declaration, manifest, receipt)
    print("Completed adapted-encoder comparison: " + str(output / "report.md"), flush=True)


if __name__ == "__main__":
    main()

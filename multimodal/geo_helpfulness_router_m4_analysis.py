"""Scoring-only M4 selection, evaluation, and immutable frozen-state delivery.

No fitting libraries or parent training pipelines may be imported here. Original
assignment labels are read only before the exclusive selection receipt exists.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from multimodal.geo_helpfulness_router_candidate_state import (
    candidate_filename,
    full_feature_columns,
    load_candidate_set_for_inference,
    predict_router_probabilities,
)
from multimodal.geo_helpfulness_router_heuristics import (
    apply_heuristic_threshold,
    enumerate_threshold_candidates,
    validate_threshold_specification,
)
from multimodal.geo_helpfulness_router_metrics import (
    binary_average_precision,
    calibration_metrics,
    classification_metrics,
    multiclass_metrics,
    policy_metrics,
    seed_summary,
)
from multimodal.geo_helpfulness_router_selection import (
    evaluate_gate,
    select_common_c,
    select_common_threshold,
)
from multimodal.geo_helpfulness_router_uncertainty import (
    bootstrap_policy_comparison,
    subgroup_policy_report,
)
from multimodal.geo_helpfulness_router_preview_score import validate_sealed_predictions
from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    COMPARATOR_DIR,
    RESULT_DIR,
    CONVENTIONS_PATH,
    VARIANTS,
    SEEDS,
    POLICIES,
    canonical_bytes,
    canonical_hash,
    check_self_hash,
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


MANIFEST_SCHEMA = "geo_helpfulness.m4_feasibility_manifest.v1"
COMPARATOR_SCHEMA = "geo_helpfulness.m4_comparator_manifest.v1"
Q_COLUMNS = ("q_rescue", "q_harm", "q_both_correct", "q_both_wrong")
TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
ASSIGNMENT_COLUMNS = (
    "protocol_id",
    "row_uid",
    "plot_idx",
    "development_role",
    "label_id_dense",
    "canonical_l3_id",
    "label_name",
)
TARGET_COLUMNS = (
    "schema_version",
    "protocol_id",
    "row_uid",
    "plot_idx",
    "training_seed",
    "label_id_dense",
    "canonical_l3_id",
    "label_name",
    "image_pred",
    "geo_pred",
    "raw_pred",
    "target_state",
    "image_geo_pred_pair",
    "geo_raw_pred_pair",
)
VALIDATION_ROWS = 822
VALIDATION_PLOTS = 325
BASE_FILES = {
    "build_declaration.json",
    "validation_targets.parquet",
    "selection.json",
    "candidate_metrics.json",
    "expert_metrics.json",
    "policy_metrics.json",
    "subgroup_metrics.parquet",
    "bootstrap_summary.json",
    "bootstrap_replicates.parquet",
    "validation_actions.parquet",
    "habitat_classification.json",
    "access_audit.json",
    "go_no_go.json",
    "report.md",
    "router_frozen_state.json",
    "development_selection_receipt.json",
    "manifest.json",
    "expert_temperatures.json",
    "router_feature_transform.json",
    "router_feature_schema.json",
    "oof_heuristics.json",
}


def _fit_incapable() -> dict:
    forbidden = sorted(
        name
        for name in sys.modules
        if name.split(".")[0] in {"torch", "scipy", "sklearn"}
    )
    if forbidden:
        raise ValueError(
            f"scoring process has forbidden fitting/native imports: {forbidden[:5]}"
        )
    return {"forbidden_imports_loaded": [], "scoring_only_process": True}


def _identities(frame: pd.DataFrame) -> list[tuple]:
    return list(
        frame[["row_uid", "plot_idx", "training_seed"]].itertuples(
            index=False, name=None
        )
    )


def _target_schema() -> pa.Schema:
    integer8 = {
        "training_seed",
        "label_id_dense",
        "canonical_l3_id",
        "image_pred",
        "geo_pred",
        "raw_pred",
    }
    return pa.schema(
        [
            pa.field(
                name,
                (
                    pa.int8()
                    if name in integer8
                    else (pa.int16() if name.endswith("_pair") else pa.string())
                ),
                nullable=False,
            )
            for name in TARGET_COLUMNS
        ]
    )


def _frame_table(frame: pd.DataFrame, schema: pa.Schema | None = None) -> pa.Table:
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    if schema is None:
        # Undefined rate/CI cells are explicit Arrow nulls; all other fields are non-null.
        schema = pa.schema(
            [
                pa.field(
                    field.name, field.type, nullable=table[field.name].null_count > 0
                )
                for field in table.schema
            ]
        )
        table = table.cast(schema)
    return table.replace_schema_metadata(None)


def _load_inputs(protocol_root: Path) -> dict:
    _fit_incapable()
    comparator_root = protocol_root / COMPARATOR_DIR
    manifest = validate_manifest(comparator_root, schema_version=COMPARATOR_SCHEMA)
    declaration = read_json(comparator_root / "declaration.json")
    if canonical_hash(declaration) != manifest["declaration_sha256"]:
        raise ValueError("scoring declaration hash mismatch")
    conventions = load_conventions()
    if declaration["conventions"] != conventions or declaration[
        "conventions_sha256"
    ] != canonical_hash(conventions):
        raise ValueError("scoring conventions changed")
    current_code = {
        str(Path(path).relative_to(PROJECT_ROOT)): digest
        for path, digest in code_snapshot().items()
    }
    if declaration["code_file_sha256"] != current_code:
        raise ValueError("predeclared scoring implementation changed")
    prediction_root = protocol_root / "router/router_validation_v1"
    predicted = validate_sealed_predictions(prediction_root)
    if predicted["manifest_sha256"] != manifest["prediction_manifest_sha256"]:
        raise ValueError("candidate prediction lineage mismatch")
    snapshot = dict(manifest["parent_file_sha256"])
    snapshot.update({str(path): file_hash(path) for path in comparator_root.iterdir()})
    dataset = read_json(protocol_root / "router/router_dataset_manifest.json")
    check_self_hash(dataset)
    if dataset["manifest_sha256"] != declaration["dataset_manifest_sha256"]:
        raise ValueError("dataset lineage mismatch")
    candidate_root = protocol_root / "router/router_candidates_v2"
    candidate_manifest, states = load_candidate_set_for_inference(
        candidate_root, expected_dataset_manifest_sha256=dataset["manifest_sha256"]
    )
    if (
        candidate_manifest["manifest_sha256"]
        != declaration["candidate_manifest_sha256"]
    ):
        raise ValueError("candidate model lineage mismatch")
    candidates = {
        variant: pq.read_table(prediction_root / f"{variant}.parquet").to_pandas()
        for variant in VARIANTS
    }
    comparator = pq.read_table(
        comparator_root / "expert_probabilities_and_features.parquet"
    ).to_pandas()
    if (
        len(comparator) != VALIDATION_ROWS * 4
        or comparator.duplicated(["row_uid", "training_seed"]).any()
    ):
        raise ValueError("comparator identity coverage mismatch")
    if _identities(comparator) != _identities(
        comparator.sort_values(["row_uid", "training_seed"])
    ):
        raise ValueError("comparator rows are not canonical")
    expected = pq.read_table(prediction_root / "expert_predictions.parquet").to_pandas()
    if _identities(comparator) != _identities(expected):
        raise ValueError("comparator/expert identities differ")
    for prefix in ("image", "geo", "raw"):
        if not np.array_equal(comparator[f"{prefix}_pred"], expected[f"{prefix}_pred"]):
            raise ValueError("comparator expert class predictions differ")
        for basis in ("calibrated", "native_t1"):
            p = np.stack(comparator[f"{prefix}_prob_{basis}"])
            if (
                p.shape != (VALIDATION_ROWS * 4, 18)
                or p.dtype != np.float64
                or not np.isfinite(p).all()
                or np.any(p < 0)
                or np.any(p > 1)
                or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)
            ):
                raise ValueError("invalid expert probability matrix")
            if not np.array_equal(p.argmax(axis=1), expected[f"{prefix}_pred"]):
                raise ValueError("expert probabilities change class predictions")
    config_path = PROJECT_ROOT / "configs/multimodal_geo_helpfulness.yaml"
    config = yaml.safe_load(config_path.read_text())
    heuristics = read_json(prediction_root / "oof_heuristics.json")
    transforms = read_json(protocol_root / "router/router_feature_transform.json")
    temperatures = read_json(protocol_root / "router/expert_temperatures.json")
    original_logits = pq.read_table(
        protocol_root
        / "development_validation/development_validation_model_outputs.parquet"
    ).to_pandas()
    if _identities(original_logits) != _identities(comparator):
        raise ValueError("expert logit/comparator identities differ")
    temperature_states = {
        (state["seed"], state["mode"]): state for state in temperatures["states"]
    }
    for seed in SEEDS:
        mask = comparator.training_seed.eq(seed).to_numpy()
        for mode, prefix in (
            ("image_only", "image"),
            ("geo_only", "geo"),
            ("raw_concat", "raw"),
        ):
            logits = np.stack(original_logits.loc[mask, f"{prefix}_logits"])
            centered = logits - logits.max(axis=1, keepdims=True)
            for basis, temperature in (
                ("native_t1", 1.0),
                ("calibrated", temperature_states[(seed, mode)]["temperature"]),
            ):
                if not np.isfinite(temperature) or temperature <= 0:
                    raise ValueError("invalid frozen temperature")
                weights = np.exp(centered / temperature)
                expected_probability = weights / weights.sum(axis=1, keepdims=True)
                observed_probability = np.stack(
                    comparator.loc[mask, f"{prefix}_prob_{basis}"]
                )
                if not np.allclose(
                    expected_probability, observed_probability, atol=1e-12, rtol=1e-12
                ):
                    raise ValueError(
                        "expert calibrated/native probabilities do not reproduce from authoritative logits"
                    )
    # Independent apply-only reconstruction through the portable selected-model API.
    transformed = {}
    for state in transforms["states"]:
        seed = state["seed"]
        rows = comparator.loc[comparator.training_seed.eq(seed)].sort_values("row_uid")
        matrix = np.zeros((len(rows), 727), dtype=np.float64)
        matrix[:, :25] = (
            rows[state["numeric_input_columns"]].to_numpy(dtype=np.float64)
            - np.asarray(state["mean"])
        ) / np.asarray(state["scale"])
        offset = 25
        for name in (
            "image_pred",
            "geo_pred",
            "raw_pred",
            "image_geo_pred_pair",
            "geo_raw_pred_pair",
        ):
            categories = rows[name].to_numpy(dtype=np.int64)
            vocabulary = state["categorical_vocabularies"][name]
            if np.any(categories < 0) or np.any(categories >= len(vocabulary)):
                raise ValueError("out-of-vocabulary comparator category")
            matrix[np.arange(len(rows)), offset + categories] = 1.0
            offset += len(vocabulary)
        if (
            offset != 727
            or state["output_columns"] != full_feature_columns()
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("portable frozen feature reconstruction failed")
        transformed[seed] = matrix
        for variant in VARIANTS:
            for c_value in (0.01, 0.1, 1.0, 10.0):
                saved = (
                    candidates[variant]
                    .loc[lambda x: x.training_seed.eq(seed) & x.C_hex.eq(c_value.hex())]
                    .sort_values("row_uid")
                )
                q = predict_router_probabilities(
                    matrix, states[candidate_filename(variant, seed, c_value)]
                )
                if not np.allclose(
                    q, saved[list(Q_COLUMNS)].to_numpy(), atol=1e-12, rtol=1e-12
                ):
                    raise ValueError("frozen candidate probabilities do not reproduce")
    check_snapshot(snapshot)
    return {
        "comparator_manifest": manifest,
        "declaration": declaration,
        "snapshot": snapshot,
        "candidates": candidates,
        "comparator": comparator,
        "states": states,
        "candidate_manifest": candidate_manifest,
        "dataset_manifest": dataset,
        "config": config,
        "heuristics": heuristics,
        "transforms": transforms,
        "temperatures": temperatures,
        "matrices": transformed,
        "protocol_root": protocol_root,
    }


def _validate_targets(rows: pd.DataFrame, inputs: dict) -> None:
    if (
        list(rows) != list(TARGET_COLUMNS)
        or len(rows) != VALIDATION_ROWS * 4
        or rows.isna().any().any()
    ):
        raise ValueError("scoring target schema/count/null mismatch")
    if set(rows.schema_version) != {"geo_helpfulness.m4_validation_targets.v1"} or set(
        rows.protocol_id
    ) != {"protocol_v1"}:
        raise ValueError("scoring target schema/protocol identity mismatch")
    if any(
        rows[name].dtype.kind not in "iu"
        for name in (
            "training_seed",
            "label_id_dense",
            "canonical_l3_id",
            "image_pred",
            "geo_pred",
            "raw_pred",
            "image_geo_pred_pair",
            "geo_raw_pred_pair",
        )
    ):
        raise ValueError("scoring target integer fields have invalid dtypes")
    if _identities(rows) != _identities(inputs["comparator"]):
        raise ValueError("scoring targets do not align with label-blind inputs")
    for name in (
        "image_pred",
        "geo_pred",
        "raw_pred",
        "image_geo_pred_pair",
        "geo_raw_pred_pair",
    ):
        if not np.array_equal(
            rows[name].to_numpy(), inputs["comparator"][name].to_numpy()
        ):
            raise ValueError(
                "scoring target expert predictions/pairs differ from label-blind comparator"
            )
    ontology = {
        item["dense_id"]: item for item in inputs["config"]["class_ontology"]["classes"]
    }
    for row in rows.itertuples(index=False):
        label = ontology.get(row.label_id_dense)
        if (
            label is None
            or row.canonical_l3_id != label["canonical_l3_id"]
            or row.label_name != label["label_name"]
        ):
            raise ValueError("scoring target ontology mismatch")
    if (
        rows.groupby("row_uid").label_id_dense.nunique().max() != 1
        or rows.groupby("plot_idx").label_id_dense.nunique().max() != 1
    ):
        raise ValueError("scoring labels differ across seeds or within a plot")
    raw_correct = rows.raw_pred.to_numpy() == rows.label_id_dense.to_numpy()
    geo_correct = rows.geo_pred.to_numpy() == rows.label_id_dense.to_numpy()
    expected = np.select(
        [
            ~raw_correct & geo_correct,
            raw_correct & ~geo_correct,
            raw_correct & geo_correct,
        ],
        TARGET_ORDER[:3],
        default="both_wrong",
    )
    if not np.array_equal(expected, rows.target_state.to_numpy()):
        raise ValueError("scoring four-state truth table mismatch")


def _create_targets(inputs: dict) -> pd.DataFrame:
    path = inputs["protocol_root"] / "protocol/development_assignments.parquet"
    if file_hash(path) != inputs["comparator_manifest"]["assignment_file_sha256"]:
        raise ValueError("assignment parent changed before scoring")
    assignment = pq.read_table(path, columns=list(ASSIGNMENT_COLUMNS)).to_pandas()
    if (
        assignment.isna().any().any()
        or set(assignment.protocol_id) != {"protocol_v1"}
        or not set(assignment.development_role).issubset({"train", "validation"})
    ):
        raise ValueError("assignment protocol/role/null contract mismatch")
    validation = (
        assignment.loc[assignment.development_role.eq("validation")]
        .sort_values("row_uid")
        .reset_index(drop=True)
    )
    if (
        len(validation) != VALIDATION_ROWS
        or validation.plot_idx.nunique() != VALIDATION_PLOTS
        or validation.row_uid.duplicated().any()
    ):
        raise ValueError("validation assignment coverage mismatch")
    base = inputs["comparator"]
    rows = base[
        [
            "protocol_id",
            "row_uid",
            "plot_idx",
            "training_seed",
            "image_pred",
            "geo_pred",
            "raw_pred",
            "image_geo_pred_pair",
            "geo_raw_pred_pair",
        ]
    ].copy()
    labels = validation.set_index("row_uid")
    if set(rows.row_uid) != set(labels.index) or not np.array_equal(
        rows.plot_idx.to_numpy(dtype=object),
        rows.row_uid.map(labels.plot_idx).to_numpy(dtype=object),
    ):
        raise ValueError("exact validation identity values differ")
    for name in ("label_id_dense", "canonical_l3_id", "label_name"):
        rows[name] = rows.row_uid.map(labels[name])
    rows.insert(0, "schema_version", "geo_helpfulness.m4_validation_targets.v1")
    raw_correct = rows.raw_pred.to_numpy() == rows.label_id_dense.to_numpy()
    geo_correct = rows.geo_pred.to_numpy() == rows.label_id_dense.to_numpy()
    rows["target_state"] = np.select(
        [
            ~raw_correct & geo_correct,
            raw_correct & ~geo_correct,
            raw_correct & geo_correct,
        ],
        TARGET_ORDER[:3],
        default="both_wrong",
    )
    rows = rows.loc[:, list(TARGET_COLUMNS)].reset_index(drop=True)
    _validate_targets(rows, inputs)
    return _frame_table(rows, _target_schema()).to_pandas()


def _candidate_metrics(inputs: dict, targets: pd.DataFrame) -> dict:
    records = {}
    states = targets.set_index(["row_uid", "training_seed"]).target_state
    for variant in VARIANTS:
        records[variant] = {}
        for c_value in (0.01, 0.1, 1.0, 10.0):
            per_seed = {}
            for seed in SEEDS:
                rows = (
                    inputs["candidates"][variant]
                    .loc[lambda x: x.training_seed.eq(seed) & x.C_hex.eq(c_value.hex())]
                    .sort_values("row_uid")
                )
                y = np.array(
                    [
                        TARGET_ORDER.index(states.loc[(uid, seed)])
                        for uid in rows.row_uid
                    ],
                    dtype=np.int64,
                )
                q = rows[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
                per_seed[str(seed)] = {
                    **multiclass_metrics(q, y),
                    "rescue_pr_auc": binary_average_precision(y == 0, q[:, 0]),
                    "harm_pr_auc": binary_average_precision(y == 1, q[:, 1]),
                    "confusion_matrix": classification_metrics(q.argmax(axis=1), y, 4)[
                        "confusion_matrix"
                    ],
                }
            records[variant][c_value.hex()] = {
                "per_seed": per_seed,
                "seed_summary": {
                    name: seed_summary([per_seed[str(seed)][name] for seed in SEEDS])
                    for name in (
                        "log_loss",
                        "brier_score",
                        "top1_accuracy",
                        "rescue_pr_auc",
                        "harm_pr_auc",
                    )
                },
            }
    return records


def _selected_rows(
    inputs: dict, targets: pd.DataFrame, variant: str, c_hex: str
) -> pd.DataFrame:
    q = (
        inputs["candidates"][variant]
        .loc[lambda x: x.C_hex.eq(c_hex)]
        .sort_values(["row_uid", "training_seed"])
        .reset_index(drop=True)
    )
    if _identities(q) != _identities(targets):
        raise ValueError("selected candidate and targets do not align")
    rows = targets.copy()
    rows["score"] = q.score.to_numpy(dtype=np.float64)
    return rows


def _policies(
    inputs: dict, targets: pd.DataFrame, selection: dict
) -> tuple[dict, dict, dict]:
    specifications, scores, actions = {}, {}, {}
    zero = {"kind": "finite", "value_hex": float(0).hex(), "comparison": "strict_gt"}
    never = {"kind": "never_intervene", "value_hex": None, "comparison": "strict_gt"}
    for variant in VARIANTS:
        selected = selection["variants"][variant]
        scores[variant] = _selected_rows(
            inputs, targets, variant, selected["regularization"]["selected_C_hex"]
        ).score.to_numpy(dtype=np.float64)
        specifications[variant] = selected["threshold"]["threshold_specification"]
    specs = {
        "never_intervene": never,
        "always_geo_on_geo_raw_disagreement": zero,
        "higher_calibrated_geo_confidence_on_geo_raw_disagreement": zero,
        **{
            name: inputs["heuristics"]["heuristics"][name]["threshold_specification"]
            for name in ("image_geo_jsd", "geo_minus_raw_margin")
        },
    }
    specifications.update(specs)
    scores.update(
        {
            "never_intervene": np.zeros(len(targets), dtype=np.float64),
            "always_geo_on_geo_raw_disagreement": np.ones(
                len(targets), dtype=np.float64
            ),
            "higher_calibrated_geo_confidence_on_geo_raw_disagreement": inputs[
                "comparator"
            ].geo_minus_raw_confidence.to_numpy(dtype=np.float64),
            "image_geo_jsd": inputs["comparator"].image_geo_jsd.to_numpy(
                dtype=np.float64
            ),
            "geo_minus_raw_margin": inputs["comparator"].geo_minus_raw_margin.to_numpy(
                dtype=np.float64
            ),
        }
    )
    for name in POLICIES:
        actions[name] = apply_heuristic_threshold(
            scores[name],
            targets.raw_pred.to_numpy(),
            targets.geo_pred.to_numpy(),
            specifications[name],
        )
    return specifications, scores, actions


def _classification_record(
    targets: pd.DataFrame, probabilities: np.ndarray, prediction: np.ndarray
) -> dict:
    def one(mask):
        rows = targets.loc[mask]
        y = rows.label_id_dense.to_numpy(dtype=np.int64)
        pred = prediction[mask]
        raw = rows.raw_pred.to_numpy(dtype=np.int64)
        result = classification_metrics(pred, y, 18)
        result["top3_accuracy"] = multiclass_metrics(probabilities[mask], y)[
            "top3_accuracy"
        ]
        flow = []
        for c in range(18):
            truth = y == c
            raw_c, final_c = raw == c, pred == c
            support_rows = rows.loc[truth]
            item = {
                "class_id": c,
                "unique_images": int(support_rows.row_uid.nunique()),
                "unique_plots": int(support_rows.plot_idx.nunique()),
                "tp_rescued": int(np.sum(truth & ~raw_c & final_c)),
                "tp_lost": int(np.sum(truth & raw_c & ~final_c)),
                "fp_introduced": int(np.sum(~truth & ~raw_c & final_c)),
                "fp_removed": int(np.sum(~truth & raw_c & ~final_c)),
                "top1_accuracy": (
                    float(np.mean(pred[truth] == y[truth])) if np.any(truth) else None
                ),
                "top3_accuracy": (
                    float(
                        np.mean(
                            np.any(
                                np.argsort(
                                    -probabilities[mask][truth], axis=1, kind="stable"
                                )[:, :3]
                                == y[truth, None],
                                axis=1,
                            )
                        )
                    )
                    if np.any(truth)
                    else None
                ),
            }
            item["low_support"] = (
                item["unique_images"] < 20 or item["unique_plots"] < 10
            )
            flow.append(item)
        result["habitat_support_and_tp_fp_flow"] = flow
        return result

    per_seed = {
        str(seed): one(targets.training_seed.eq(seed).to_numpy()) for seed in SEEDS
    }
    return {
        "per_seed": per_seed,
        "pooled_seed_realizations": one(np.ones(len(targets), dtype=bool)),
        "seed_summary": {
            name: seed_summary([per_seed[str(seed)][name] for seed in SEEDS])
            for name in ("accuracy", "top3_accuracy", "macro_f1", "weighted_f1", "mcc")
        },
    }


def _access_audit(inputs: dict) -> dict:
    boundary = _fit_incapable()
    fits = inputs["temperatures"]["states"] + inputs["transforms"]["states"]
    train_only = all(
        state["fit_role"] == "development_train_oof" and state["fit_row_count"] == 3378
        for state in fits
    )
    candidate_train_only = all(
        state.get("fit_role") == "development_train_oof"
        and state.get("fit_row_count") == 3378
        for state in inputs["states"].values()
    )
    checks = {
        "all_bound_parent_and_code_hashes_match": True,
        "preprocessing_fit_roles_training_oof_only": train_only,
        "candidate_training_only_contract_validated": candidate_train_only,
        "all_48_predictions_sealed_before_full_scoring": True,
        "oof_heuristic_specs_frozen_before_preview": inputs["heuristics"].get(
            "validation_labels_scored"
        )
        is False,
        "comparator_export_label_blind": inputs["comparator_manifest"].get(
            "validation_labels_read"
        )
        is False,
        "fitting_incapable_scoring_process": boundary["scoring_only_process"],
        "no_cleaned_test_source_resolved_or_opened": True,
        "fixed_primary_and_predeclared_variants_retained": tuple(inputs["candidates"])
        == VARIANTS,
    }
    check_snapshot(inputs["snapshot"])
    return {
        "schema_version": "geo_helpfulness.m4_access_audit.v1",
        "checks": checks,
        "clean": all(checks.values()),
        **boundary,
        "evidence_scope": "validated_provenance_hashes_explicit_reader_allowlists_import_guards_and_boundary_tests_not_OS_wide_trace",
        "disclosures": [
            "The fixed primary seed-1/C=0.01 accuracy preview preceded full scoring; no training/feature/policy setting changed from it.",
            "Legacy integrity/regression checks parsed full development assignments for split counts but did not use validation labels for fitting or model selection.",
            "The first preview scorer read labels but rejected a pandas storage-dtype equality check before computing metrics; the additive correction kept exact identity values and all scientific settings unchanged.",
            "Historical cleaned-test findings motivated the method; no cleaned-test source is opened in this M4 workflow and no independent-confirmation claim is made.",
        ],
    }


def _evaluate(inputs: dict, targets: pd.DataFrame, selection: dict) -> dict:
    _validate_targets(targets, inputs)
    specifications, scores, actions = _policies(inputs, targets, selection)
    comparator = inputs["comparator"]
    probability = {
        prefix: np.stack(comparator[f"{prefix}_prob_calibrated"])
        for prefix in ("image", "geo", "raw")
    }
    expert_metrics, classification = {}, {"experts": {}, "policies": {}}
    for prefix in ("image", "geo", "raw"):
        expert_metrics[prefix] = {}
        for basis in ("native_t1", "calibrated"):
            p = np.stack(comparator[f"{prefix}_prob_{basis}"])
            per_seed = {}
            for seed in SEEDS:
                mask = targets.training_seed.eq(seed).to_numpy()
                per_seed[str(seed)] = calibration_metrics(
                    p[mask],
                    targets.loc[mask, "label_id_dense"].to_numpy(dtype=np.int64),
                )
            expert_metrics[prefix][basis] = {
                "per_seed": per_seed,
                "seed_summary": {
                    name: seed_summary([per_seed[str(seed)][name] for seed in SEEDS])
                    for name in ("log_loss", "brier_score", "top1_accuracy", "ece")
                },
            }
        classification["experts"][prefix] = _classification_record(
            targets,
            probability[prefix],
            targets[f"{prefix}_pred"].to_numpy(dtype=np.int64),
        )
    policies, action_frames = {}, []
    raw = targets.raw_pred.to_numpy(dtype=np.int64)
    geo = targets.geo_pred.to_numpy(dtype=np.int64)
    y = targets.label_id_dense.to_numpy(dtype=np.int64)
    for name in POLICIES:
        acted = actions[name]
        final = np.where(acted, geo, raw)
        policies[name] = {
            **policy_metrics(targets, acted),
            "threshold_specification": specifications[name],
        }
        p = np.where(acted[:, None], probability["geo"], probability["raw"])
        classification["policies"][name] = _classification_record(targets, p, final)
        frame = targets[
            [
                "protocol_id",
                "row_uid",
                "plot_idx",
                "training_seed",
                "target_state",
                "raw_pred",
                "geo_pred",
            ]
        ].copy()
        frame.insert(0, "schema_version", "geo_helpfulness.m4_policy_actions.v1")
        frame["policy_id"] = name
        frame["score"] = scores[name]
        frame["threshold_specification"] = canonical_bytes(specifications[name]).decode(
            "utf-8"
        )
        frame["acted"] = acted
        frame["final_pred"] = final.astype(np.int8)
        frame["raw_correct"], frame["geo_correct"], frame["final_correct"] = (
            raw == y,
            geo == y,
            final == y,
        )
        frame["rescued"] = acted & (targets.target_state.to_numpy() == "rescue")
        frame["harmed"] = acted & (targets.target_state.to_numpy() == "harm")
        frame["net_correct"] = frame.rescued.to_numpy(
            dtype=np.int8
        ) - frame.harmed.to_numpy(dtype=np.int8)
        action_frames.append(frame)
    audit = _access_audit(inputs)
    gates = {}
    for variant in VARIANTS:
        net = actions[variant].astype(np.int64) * (
            (targets.target_state.to_numpy() == "rescue").astype(np.int64)
            - (targets.target_state.to_numpy() == "harm").astype(np.int64)
        )
        per_plot = (
            pd.DataFrame({"plot_idx": targets.plot_idx, "rescued_minus_harmed": net})
            .groupby("plot_idx")
            .rescued_minus_harmed.sum()
            .to_dict()
        )
        gates[variant] = evaluate_gate(
            policies[variant],
            {name: policies[name] for name in POLICIES[3:]},
            per_plot,
            audit["clean"],
        )
    bootstrap, replicates = bootstrap_policy_comparison(targets, actions)
    subgroups = subgroup_policy_report(targets, actions)
    combined_actions = (
        pd.concat(action_frames, ignore_index=True)
        .sort_values(["policy_id", "row_uid", "training_seed"])
        .reset_index(drop=True)
    )
    return {
        "candidate_metrics.json": _candidate_metrics(inputs, targets),
        "expert_metrics.json": expert_metrics,
        "policy_metrics.json": policies,
        "habitat_classification.json": classification,
        "access_audit.json": audit,
        "go_no_go.json": {
            "primary_variant": VARIANTS[0],
            "primary_decision": gates[VARIANTS[0]]["decision"],
            "variants": gates,
            "ablation_promotion_allowed": False,
        },
        "bootstrap_summary.json": bootstrap,
        "bootstrap_replicates.parquet": _frame_table(replicates),
        "subgroup_metrics.parquet": _frame_table(subgroups),
        "validation_actions.parquet": _frame_table(combined_actions),
    }


def _report(inputs: dict, selection: dict, evaluated: dict) -> str:
    gate = evaluated["go_no_go.json"]
    lines = [
        "# M4 — Frozen router feasibility results",
        "",
        f"Primary decision: **{gate['primary_decision']}**.",
        "",
        "Internal development feasibility on one fixed plot-grouped holdout; model and threshold selection use these same rows. No independent generalization claim is made.",
        "",
        "## Selected router policies",
        "",
        "| Variant | Common C | Threshold | Mean seed net | Mean Δaccuracy | Coverage | Pooled rescues | Pooled harms | Gate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        chosen = selection["variants"][variant]
        metric = evaluated["policy_metrics.json"][variant]
        threshold = chosen["threshold"]["threshold_specification"]
        display = (
            threshold["value_hex"]
            if threshold["kind"] == "finite"
            else "never_intervene"
        )
        lines.append(
            f"| {variant} | {chosen['regularization']['selected_C']} | {display} | {metric['mean_seed_net_utility']:.4f} | {metric['mean_seed_delta_accuracy']:.6f} | {metric['pooled']['coverage_fraction']:.6f} | {metric['pooled']['rescued_count']} | {metric['pooled']['harmed_count']} | {gate['variants'][variant]['decision']} |"
        )
    lines += [
        "",
        "## All fixed and OOF-selected comparators",
        "",
        "| Policy | Mean seed net | Mean Δaccuracy | Coverage | Pooled rescues | Pooled harms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in POLICIES[3:]:
        metric = evaluated["policy_metrics.json"][name]
        lines.append(
            f"| {name} | {metric['mean_seed_net_utility']:.4f} | {metric['mean_seed_delta_accuracy']:.6f} | {metric['pooled']['coverage_fraction']:.6f} | {metric['pooled']['rescued_count']} | {metric['pooled']['harmed_count']} |"
        )
    lines += ["", "## Primary gate", ""]
    for name, passed in gate["variants"][VARIANTS[0]]["criteria"].items():
        lines.append(f"- {name}: {passed}")
    lines += [
        "",
        "## Audit and interpretation",
        "",
        "The primary method is unchanged. Ablations are secondary and cannot replace it on the basis of validation results. All four selected seed-specific routers are retained; no best seed is chosen.",
        "",
        "Paired habitat-stratified complete-plot bootstrap intervals use 2,000 resamples and fixed selected policies. These intervals are descriptive and selection-conditional, and do not measure training-set, fixed-split, or dataset-shift uncertainty.",
        "",
        "Detailed candidate log losses/AP/Brier scores, expert native/calibrated diagnostics, per-policy rescue/harm metrics, habitat confusion matrices/F1/MCC/TP-FP flows, subgroup results, bootstrap replicates, and every row-level action are saved as sibling artifacts.",
        "",
    ]
    lines += [f"- {item}" for item in evaluated["access_audit.json"]["disclosures"]]
    return "\n".join(lines) + "\n"


def _selected_model_names(selection: dict) -> list[str]:
    return [
        "models/"
        + candidate_filename(
            variant,
            seed,
            selection["variants"][variant]["regularization"]["selected_C"],
        )
        for variant in VARIANTS
        for seed in SEEDS
    ]


def _check_file_allowlist(
    root: Path, selection: dict | None = None, *, complete: bool = False
) -> None:
    if root.is_symlink():
        raise ValueError("M4 result root cannot be a symlink")
    potential_models = {
        "models/" + candidate_filename(variant, seed, c)
        for variant in VARIANTS
        for seed in SEEDS
        for c in (0.01, 0.1, 1.0, 10.0)
    }
    expected = BASE_FILES | (
        potential_models if selection is None else set(_selected_model_names(selection))
    )
    files = set()
    for path in root.rglob("*"):
        name = str(path.relative_to(root))
        if path.is_symlink() or (path.is_dir() and name != "models"):
            raise ValueError(f"unknown or symlink M4 artifact: {name}")
        if path.is_file():
            files.add(name)
    if not files.issubset(expected) or (complete and files != expected):
        raise ValueError(
            "M4 artifact allowlist mismatch; unknown files are never adopted or removed"
        )


def _frozen_payloads(inputs: dict, selection: dict) -> tuple[dict[str, bytes], dict]:
    protocol_root = inputs["protocol_root"]
    sources = {
        "expert_temperatures.json": protocol_root / "router/expert_temperatures.json",
        "router_feature_transform.json": protocol_root
        / "router/router_feature_transform.json",
        "router_feature_schema.json": protocol_root
        / "router/targets_and_feature_contract/router_feature_schema.json",
        "oof_heuristics.json": protocol_root
        / "router/router_validation_v1/oof_heuristics.json",
    }
    for name in _selected_model_names(selection):
        sources[name] = protocol_root / "router/router_candidates_v2" / Path(name).name
    payloads = {name: path.read_bytes() for name, path in sources.items()}
    components = {
        name: {"file_sha256": file_hash(path), "source": str(path)}
        for name, path in sources.items()
    }
    primary_hashes, variants = {}, {}
    temperature_by_seed = {
        seed: [
            state for state in inputs["temperatures"]["states"] if state["seed"] == seed
        ]
        for seed in SEEDS
    }
    transform_by_seed = {
        state["seed"]: state for state in inputs["transforms"]["states"]
    }
    for variant in VARIANTS:
        chosen = selection["variants"][variant]
        seed_states = {}
        for seed in SEEDS:
            name = "models/" + candidate_filename(
                variant, seed, chosen["regularization"]["selected_C"]
            )
            state = inputs["states"][Path(name).name]
            seed_state = {
                "training_seed": seed,
                "model_file": name,
                "model_state_sha256": state["state_sha256"],
                "model_file_sha256": components[name]["file_sha256"],
                "expert_temperatures_sha256": canonical_hash(temperature_by_seed[seed]),
                "feature_transform_sha256": canonical_hash(transform_by_seed[seed]),
                "feature_schema_file_sha256": components["router_feature_schema.json"][
                    "file_sha256"
                ],
                "target_order": list(TARGET_ORDER),
                "router_output_calibrator": "none",
                "C_hex": chosen["regularization"]["selected_C_hex"],
                "threshold_specification": chosen["threshold"][
                    "threshold_specification"
                ],
                "score": "q_rescue_minus_q_harm",
                "comparison": "strict_gt",
                "require_geo_raw_disagreement": True,
                "otherwise": "retain_raw_concat_prediction",
            }
            seed_state["frozen_state_sha256"] = canonical_hash(seed_state)
            seed_states[str(seed)] = seed_state
        variants[variant] = {
            "role": "primary" if variant == VARIANTS[0] else "secondary_ablation",
            "selected_C_hex": chosen["regularization"]["selected_C_hex"],
            "threshold_specification": chosen["threshold"]["threshold_specification"],
            "seeds": seed_states,
        }
        if variant == VARIANTS[0]:
            primary_hashes = {
                seed: record["frozen_state_sha256"]
                for seed, record in seed_states.items()
            }
    frozen = {
        "schema_version": "geo_helpfulness.router_frozen_state.v1",
        "protocol_id": "protocol_v1",
        "primary_variant": VARIANTS[0],
        "variants": variants,
        "components": components,
        "numeric_feature_transform": "four_frozen_seed_local_population_standardizers",
        "categorical_vocabularies": "fixed_18_class_and_324_pair_vocabularies_in_router_feature_transform.json",
        "imputer": "none_missing_values_rejected",
        "density_estimator": "none",
        "router_output_calibrator": "none",
        "expert_refit_state_included": False,
        "freeze_rule": "byte_for_byte_unchanged_after_selection_receipt_no_refit_rebuild_or_expansion",
    }
    frozen["frozen_bundle_sha256"] = canonical_hash(frozen)
    payloads["router_frozen_state.json"] = canonical_bytes(frozen) + b"\n"
    return payloads, primary_hashes


def _make_selection(inputs: dict, targets: pd.DataFrame) -> dict:
    regularization = select_common_c(inputs["candidates"], targets)
    result = {
        "schema_version": "geo_helpfulness.m4_selection.v1",
        "protocol_id": "protocol_v1",
        "variants": {},
    }
    for variant in VARIANTS:
        selected = _selected_rows(
            inputs, targets, variant, regularization[variant]["selected_C_hex"]
        )
        result["variants"][variant] = {
            "regularization": regularization[variant],
            "threshold": select_common_threshold(selected),
        }
    result["selection_sha256"] = canonical_hash(result)
    return result


def _validate_saved_selection(
    selection: dict, inputs: dict, targets: pd.DataFrame, candidate_metrics: dict
) -> None:
    """Reproduce saved diagnostics and check optimality, never generate a new selection."""
    check_self_hash(selection, "selection_sha256")
    if set(selection["variants"]) != set(VARIANTS):
        raise ValueError("saved selection has a different variant set")
    for variant in VARIANTS:
        chosen = selection["variants"][variant]
        reg = chosen["regularization"]
        if len(reg["candidate_losses"]) != 4:
            raise ValueError("saved C diagnostics incomplete")
        if [(record["C"], record["C_hex"]) for record in reg["candidate_losses"]] != [
            (c, c.hex()) for c in (0.01, 0.1, 1.0, 10.0)
        ]:
            raise ValueError("saved C diagnostics do not cover the exact ordered grid")
        chosen_loss = reg["selected_mean_log_loss"]
        for record in reg["candidate_losses"]:
            metrics = candidate_metrics[variant][record["C_hex"]]
            per_seed = {
                str(seed): metrics["per_seed"][str(seed)]["log_loss"] for seed in SEEDS
            }
            mean = float(np.mean(np.asarray(list(per_seed.values()), dtype=np.float64)))
            if (
                per_seed != record["per_seed_log_loss"]
                or mean != record["mean_log_loss"]
            ):
                raise ValueError("saved C diagnostics do not reproduce")
            if mean < chosen_loss or (
                mean == chosen_loss and record["C"] < reg["selected_C"]
            ):
                raise ValueError("saved C violates frozen optimality/tie rule")
        selected_record = next(
            record
            for record in reg["candidate_losses"]
            if record["C_hex"] == reg["selected_C_hex"]
        )
        if (
            selected_record["C"] != reg["selected_C"]
            or selected_record["mean_log_loss"] != chosen_loss
        ):
            raise ValueError("saved selected-C identity is inconsistent")
        rows = _selected_rows(inputs, targets, variant, reg["selected_C_hex"])
        threshold = chosen["threshold"]
        curve = threshold["candidate_curve"]
        if [
            record["threshold_specification"] for record in curve
        ] != enumerate_threshold_candidates(rows.score.to_numpy(dtype=np.float64)):
            raise ValueError(
                "saved threshold curve omits or changes frozen candidate boundaries"
            )
        if (
            len(curve) != threshold["candidate_count"]
            or canonical_hash([record["threshold_specification"] for record in curve])
            != threshold["candidate_specifications_sha256"]
        ):
            raise ValueError("saved threshold curve identity mismatch")
        feasible_count = 0
        winner_spec = threshold["threshold_specification"]
        winner = None
        for record in curve:
            spec = record["threshold_specification"]
            if record["threshold_display"] != validate_threshold_specification(spec):
                raise ValueError(
                    "saved threshold display differs from its exact specification"
                )
            mask = apply_heuristic_threshold(
                rows.score.to_numpy(dtype=np.float64),
                rows.raw_pred.to_numpy(),
                rows.geo_pred.to_numpy(),
                spec,
            )
            acted = rows.loc[mask]
            rescue, harm = int(acted.target_state.eq("rescue").sum()), int(
                acted.target_state.eq("harm").sum()
            )
            counts = {
                "row_count": len(rows),
                "acted_count": int(mask.sum()),
                "rescued_count": rescue,
                "harmed_count": harm,
                "rescued_minus_harmed": rescue - harm,
                "mean_seed_net_utility": (rescue - harm) / 4,
                "coverage_fraction": float(mask.mean()),
                "unique_acted_plots": int(acted.plot_idx.nunique()),
            }
            feasible = (
                counts["acted_count"] * 100 >= len(rows)
                and counts["unique_acted_plots"] >= 20
                and 2 * harm <= rescue
                and rescue - harm > 0
            )
            if (
                any(record[key] != value for key, value in counts.items())
                or record["feasible"] != feasible
            ):
                raise ValueError("saved threshold curve does not reproduce")
            feasible_count += int(feasible)
            if spec == winner_spec:
                winner = record
        if winner is None or feasible_count != threshold["feasible_candidate_count"]:
            raise ValueError("selected threshold is absent or feasibility differs")
        if feasible_count == 0:
            if winner_spec["kind"] != "never_intervene":
                raise ValueError("infeasible selection did not retain raw fallback")
        else:

            def rank(record):
                return (
                    record["rescued_minus_harmed"],
                    -record["harmed_count"],
                    -record["acted_count"],
                    record["threshold_display"],
                )

            if not winner["feasible"] or any(
                rank(record) > rank(winner) for record in curve if record["feasible"]
            ):
                raise ValueError("saved threshold violates frozen utility/tie rule")


def _build_declaration(inputs: dict, targets: pd.DataFrame) -> dict:
    declaration = {
        "schema_version": "geo_helpfulness.m4_build_declaration.v1",
        "protocol_id": "protocol_v1",
        "comparator_manifest_sha256": inputs["comparator_manifest"]["manifest_sha256"],
        "assignment_file_sha256": inputs["comparator_manifest"][
            "assignment_file_sha256"
        ],
        "parent_file_sha256": dict(sorted(inputs["snapshot"].items())),
        "ordered_validation_target_hash": canonical_hash(
            targets.to_dict(orient="records")
        ),
        "selection_receipt_is_irreversible": True,
    }
    declaration["declaration_sha256"] = canonical_hash(declaration)
    return declaration


def _receipt(
    inputs: dict,
    targets: pd.DataFrame,
    selection: dict,
    evaluated: dict,
    root: Path,
    frozen_hashes: dict,
) -> dict:
    primary = selection["variants"][VARIANTS[0]]
    config = inputs["config"]
    validation_parent = read_json(
        inputs["protocol_root"] / "development_validation/aggregate_manifest.json"
    )
    result_hashes = {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.json", "development_selection_receipt.json"}
    }
    receipt = {
        "schema_version": "geo_helpfulness.development_selection_receipt.v1",
        "protocol_id": "protocol_v1",
        "candidate_set_hash": inputs["candidate_manifest"]["manifest_sha256"],
        "candidate_prediction_hash": inputs["comparator_manifest"][
            "prediction_manifest_sha256"
        ],
        "train_oof_parent_hash": inputs["dataset_manifest"]["manifest_sha256"],
        "validation_output_parent_hash": validation_parent["manifest_sha256"],
        "ordered_validation_target_hash": canonical_hash(
            targets.to_dict(orient="records")
        ),
        "ontology_hash": canonical_hash(config["class_ontology"]["classes"]),
        "selected_C_hex": primary["regularization"]["selected_C_hex"],
        "threshold_specification": primary["threshold"]["threshold_specification"],
        "router_frozen_state_hashes": frozen_hashes,
        "validation_metrics": evaluated["policy_metrics.json"][VARIANTS[0]],
        "go_no_go": evaluated["go_no_go.json"],
        "result_artifact_file_sha256": result_hashes,
        "parent_file_sha256": dict(sorted(inputs["snapshot"].items())),
        "secondary_variant_policy_hashes": {
            variant: canonical_hash(selection["variants"][variant])
            for variant in VARIANTS[1:]
        },
        "interpretation": "selection_conditional_internal_feasibility_not_independent_confirmation",
    }
    required = set(
        config["artifact_contract"]["development_selection_receipt"]["required_fields"]
    )
    if not required.issubset(receipt) or any(
        receipt[name] is None for name in required
    ):
        raise ValueError("official development selection receipt contract incomplete")
    receipt["receipt_sha256"] = canonical_hash(receipt)
    return receipt


def _finish_manifest(root: Path, receipt: dict, selection: dict) -> dict:
    check_self_hash(receipt, "receipt_sha256")
    _check_file_allowlist(root, selection)
    expected_pre_receipt = (BASE_FILES | set(_selected_model_names(selection))) - {
        "manifest.json",
        "development_selection_receipt.json",
    }
    if set(receipt["result_artifact_file_sha256"]) != expected_pre_receipt:
        raise ValueError(
            "selection receipt does not bind every required result artifact"
        )
    for name, expected in receipt["result_artifact_file_sha256"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"receipt-bound result artifact changed: {name}")
    manifest = publish_manifest(
        root,
        schema_version=MANIFEST_SCHEMA,
        snapshot=receipt["parent_file_sha256"],
        extra={
            "selection_receipt_sha256": receipt["receipt_sha256"],
            "primary_decision": receipt["go_no_go"]["primary_decision"],
            "frozen_state_file_sha256": receipt["result_artifact_file_sha256"][
                "router_frozen_state.json"
            ],
            "environment": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "pyarrow", "PyYAML")
                },
            },
            "post_receipt_reselection_permitted": False,
        },
    )
    _check_file_allowlist(root, selection, complete=True)
    return manifest


def _validate_result(
    protocol_root: Path, root: Path, inputs: dict | None = None
) -> dict:
    manifest = validate_manifest(root, schema_version=MANIFEST_SCHEMA)
    selection = read_json(root / "selection.json")
    _check_file_allowlist(root, selection, complete=True)
    receipt = read_json(root / "development_selection_receipt.json")
    check_self_hash(receipt, "receipt_sha256")
    if receipt["receipt_sha256"] != manifest["selection_receipt_sha256"]:
        raise ValueError("selection receipt identity changed")
    # This loader only reads label-blind inputs and hashes original assignment bytes.
    inputs = _load_inputs(protocol_root) if inputs is None else inputs
    if receipt["parent_file_sha256"] != dict(sorted(inputs["snapshot"].items())):
        raise ValueError("result parent lineage does not match current inputs")
    target_table = pq.read_table(root / "validation_targets.parquet")
    if target_table.schema != _target_schema():
        raise ValueError("sealed scoring target physical schema mismatch")
    targets = target_table.to_pandas()
    _validate_targets(targets, inputs)
    if read_json(root / "build_declaration.json") != _build_declaration(
        inputs, targets
    ):
        raise ValueError(
            "result build declaration does not match frozen inputs/targets"
        )
    if (
        canonical_hash(targets.to_dict(orient="records"))
        != receipt["ordered_validation_target_hash"]
    ):
        raise ValueError("sealed scoring target fingerprint mismatch")
    evaluated = _evaluate(inputs, targets, selection)
    _validate_saved_selection(
        selection, inputs, targets, evaluated["candidate_metrics.json"]
    )
    for name, expected in evaluated.items():
        if isinstance(expected, pa.Table):
            compare_tables(pq.read_table(root / name), expected)
        elif read_json(root / name) != expected:
            raise ValueError(f"scoring report does not reproduce: {name}")
    if (root / "report.md").read_text(encoding="utf-8") != _report(
        inputs, selection, evaluated
    ):
        raise ValueError("human-readable report does not reproduce")
    frozen, frozen_hashes = _frozen_payloads(inputs, selection)
    for name, expected in frozen.items():
        if (root / name).read_bytes() != expected:
            raise ValueError(f"frozen inference component differs: {name}")
    expected_receipt = _receipt(
        inputs, targets, selection, evaluated, root, frozen_hashes
    )
    if receipt != expected_receipt:
        raise ValueError("selection receipt contents do not reproduce")
    check_snapshot(inputs["snapshot"])
    _fit_incapable()
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
        "bundle_root": str(root),
    }


def validate_m4_result(protocol_root: Path) -> dict:
    protocol_root = Path(protocol_root).resolve()
    root = protocol_root / RESULT_DIR
    with exclusive_lock(root):
        return _validate_result(protocol_root, root)


def build_m4_result(protocol_root: Path) -> dict:
    protocol_root = Path(protocol_root).resolve()
    root = protocol_root / RESULT_DIR
    with exclusive_lock(root):
        root.mkdir(exist_ok=True)
        _check_file_allowlist(root)
        if (root / "development_selection_receipt.json").exists():
            # The receipt is an irreversible selection boundary, even after a crash.
            receipt = read_json(root / "development_selection_receipt.json")
            selection = read_json(root / "selection.json")
            _finish_manifest(root, receipt, selection)
            result = _validate_result(protocol_root, root)
            return {**result, "status": "reused_valid"}
        if (root / "manifest.json").exists():
            raise ValueError("committed result lacks its selection receipt")
        if any(root.iterdir()) and not (root / "build_declaration.json").exists():
            raise ValueError("unowned partial result lacks a build declaration")
        inputs = _load_inputs(protocol_root)
        # Before the receipt, always authenticate partial targets against the source.
        # After the receipt, the branch above validates only the sealed target copy.
        targets = _create_targets(inputs)
        write_json(root / "build_declaration.json", _build_declaration(inputs, targets))
        if (root / "validation_targets.parquet").exists():
            compare_tables(
                pq.read_table(root / "validation_targets.parquet"),
                _frame_table(targets, _target_schema()),
            )
        else:
            write_bytes(
                root / "validation_targets.parquet",
                table_bytes(_frame_table(targets, _target_schema())),
            )
        if (root / "selection.json").exists():
            selection = read_json(root / "selection.json")
            check_self_hash(selection, "selection_sha256")
        else:
            selection = _make_selection(inputs, targets)
            write_json(root / "selection.json", selection)
        _check_file_allowlist(root, selection)
        evaluated = _evaluate(inputs, targets, selection)
        _validate_saved_selection(
            selection, inputs, targets, evaluated["candidate_metrics.json"]
        )
        for name, value in evaluated.items():
            if isinstance(value, pa.Table):
                write_bytes(root / name, table_bytes(value))
            else:
                write_json(root / name, value)
        write_bytes(
            root / "report.md", _report(inputs, selection, evaluated).encode("utf-8")
        )
        frozen, hashes = _frozen_payloads(inputs, selection)
        for name, payload in frozen.items():
            write_bytes(root / name, payload)
        _check_file_allowlist(root, selection)
        check_snapshot(inputs["snapshot"])
        receipt = _receipt(inputs, targets, selection, evaluated, root, hashes)
        write_json(root / "development_selection_receipt.json", receipt)
        _finish_manifest(root, receipt, selection)
        result = _validate_result(protocol_root, root, inputs)
        return {**result, "status": "built", "assignment_label_columns_opened": True}

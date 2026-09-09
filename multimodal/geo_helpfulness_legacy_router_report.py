"""Pure reports for one legacy expert family under the frozen M4 policies.

The caller supplies sealed predictions/actions and a separate label table. This
module does no file access, fitting, selection, or M4 gate evaluation. The same
function must be called independently for each checkpoint family, because its
expert correctness and four-state targets can differ for the same test image.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from multimodal.geo_helpfulness_router_candidate_state import (
    CATEGORICAL_FEATURE_SIZES,
    NUMERIC_INPUT_COLUMNS,
)
from multimodal.geo_helpfulness_router_heuristics import (
    apply_heuristic_threshold,
    validate_threshold_specification,
)
from multimodal.geo_helpfulness_router_m4_common import POLICIES, SEEDS, VARIANTS
from multimodal.geo_helpfulness_router_metrics import (
    binary_average_precision,
    calibration_metrics,
    classification_metrics,
    multiclass_metrics,
    policy_metrics,
    seed_summary,
)
from multimodal.geo_helpfulness_router_uncertainty import (
    bootstrap_policy_comparison,
    subgroup_policy_report,
)


# Importing the protocol module would load sklearn. Keep its fixed ontology
# literal here; a synthetic parity test guards against accidental divergence.
ONTOLOGY = (
    (0, 0, "Urban"),
    (1, 1, "Broadleaved Mixed and Yew Woodland"),
    (2, 2, "Coniferous Woodland"),
    (3, 4, "Arable and Horticulture"),
    (4, 5, "Improved Grassland"),
    (5, 6, "Neutral Grassland"),
    (6, 7, "Calcareous Grassland"),
    (7, 8, "Acid Grassland"),
    (8, 9, "Bracken"),
    (9, 10, "Dwarf Shrub Heath"),
    (10, 11, "Fen, Marsh, Swamp"),
    (11, 12, "Bog"),
    (12, 13, "Littoral Rock"),
    (13, 14, "Littoral Sediment"),
    (14, 15, "Montane"),
    (15, 17, "Inland Rock"),
    (16, 18, "Supra-littoral Rock"),
    (17, 19, "Supra-littoral Sediment"),
)
TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
Q_COLUMNS = tuple(f"q_{name}" for name in TARGET_ORDER)
KEY = ("row_uid", "training_seed")
IDENTITY = ("row_uid", "plot_idx", "training_seed")
FEATURE_COLUMNS = tuple(name for name, _ in CATEGORICAL_FEATURE_SIZES) + (
    *NUMERIC_INPUT_COLUMNS,
)
LABEL_COLUMNS = (
    "row_uid",
    "file",
    "plot_idx",
    "label_id_dense",
    "canonical_l3_id",
    "label_name",
)
PROBABILITY_COLUMNS = tuple(
    f"{expert}_prob_{basis}"
    for expert in ("image", "geo", "raw")
    for basis in ("native_t1", "calibrated")
)
CALIBRATION_DISCLOSURE = (
    "M4 OOF temperatures are transferred unchanged to legacy checkpoints; "
    "calibration on these checkpoint families is not guaranteed."
)
CHECKPOINT_SELECTION_DISCLOSURE = (
    "The oracle_test_best_model family was selected using test performance, so "
    "its results are test-selection-biased and are not independent confirmation. "
    "The bootstrap conditions on the supplied checkpoints and frozen M4 policies; "
    "it does not include checkpoint-selection or policy-selection uncertainty."
)
INTERPRETATION = "descriptive_legacy_transfer_with_frozen_M4_policies"
CHECKPOINT_FAMILIES = ("best_model", "oracle_test_best_model")


class LegacyRouterReportError(ValueError):
    """Sealed predictions and supplied labels violate the report contract."""


def _require_frame(frame: Any, columns: tuple[str, ...], name: str) -> None:
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or frame.columns.duplicated().any()
        or not set(columns).issubset(frame.columns)
    ):
        raise LegacyRouterReportError(
            f"{name}: nonempty table with required columns expected"
        )
    if frame.loc[:, list(columns)].isna().any().any():
        raise LegacyRouterReportError(f"{name}: required columns contain null values")


def _label_guard(frame: pd.DataFrame, name: str) -> None:
    forbidden = {
        "label",
        "label_id",
        "label_id_dense",
        "label_id_original",
        "label_name",
        "canonical_l3_id",
        "target",
        "target_state",
        "raw_correct",
        "geo_correct",
        "image_correct",
        "final_correct",
        "rescued",
        "harmed",
        "net_correct",
        "ground_truth",
        "true_label",
        "is_correct",
    }
    if forbidden.intersection(frame.columns):
        raise LegacyRouterReportError(
            f"{name}: labels/targets/outcomes must remain separate"
        )


def _integers(frame: pd.DataFrame, name: str, allowed: Any) -> None:
    if frame[name].dtype.kind not in "iu" or not frame[name].isin(allowed).all():
        raise LegacyRouterReportError(
            f"{name}: integer values outside the fixed contract"
        )


def _identity(frame: pd.DataFrame, name: str, *, with_file: bool = False) -> None:
    for column in ("row_uid", "plot_idx", *(("file",) if with_file else ())):
        if (
            not frame[column]
            .map(lambda v: isinstance(v, str) and bool(v.strip()))
            .all()
        ):
            raise LegacyRouterReportError(f"{name}: {column} must be nonempty strings")
    _integers(frame, "training_seed", SEEDS)
    if frame.duplicated(list(KEY)).any():
        raise LegacyRouterReportError(f"{name}: duplicate image/seed identities")
    keys = list(frame.loc[:, list(KEY)].itertuples(index=False, name=None))
    if keys != sorted(keys):
        raise LegacyRouterReportError(
            f"{name}: rows must use canonical image/seed order"
        )


def _aligned(frame: pd.DataFrame, experts: pd.DataFrame, name: str) -> None:
    _identity(frame, name)
    if len(frame) != len(experts) or not np.array_equal(
        frame.loc[:, list(IDENTITY)].to_numpy(dtype=object),
        experts.loc[:, list(IDENTITY)].to_numpy(dtype=object),
    ):
        raise LegacyRouterReportError(
            f"{name}: exact image/plot/seed alignment mismatch"
        )
    for column in ("family", "checkpoint_family"):
        if (
            column in frame
            and column in experts
            and not np.array_equal(
                frame[column].to_numpy(dtype=object),
                experts[column].to_numpy(dtype=object),
            )
        ):
            raise LegacyRouterReportError(f"{name}: checkpoint family mismatch")


def _probability(frame: pd.DataFrame, column: str, classes: int = 18) -> np.ndarray:
    try:
        value = np.stack(frame[column].to_numpy())
    except (TypeError, ValueError) as exc:
        raise LegacyRouterReportError(
            f"{column}: rectangular probability vectors required"
        ) from exc
    if value.dtype != np.dtype("float64") or value.shape != (len(frame), classes):
        raise LegacyRouterReportError(
            f"{column}: float64 ({len(frame)}, {classes}) required"
        )
    if (
        not np.isfinite(value).all()
        or np.any(value < 0)
        or np.any(value > 1)
        or not np.allclose(value.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    ):
        raise LegacyRouterReportError(
            f"{column}: invalid probability range/row normalization"
        )
    return value


def _semantic_values(probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reconstruct M3 features using only transferred probability vectors."""
    values = {}
    orders = {}
    n = len(probabilities["raw"])
    row = np.arange(n)
    for name, probability in probabilities.items():
        order = np.argsort(-probability, axis=1, kind="stable")
        orders[name] = order
        values[f"{name}_pred"] = order[:, 0].astype(np.int64)
        values[f"{name}_confidence"] = probability[row, order[:, 0]]
        terms = np.zeros_like(probability)
        positive = probability > 0
        terms[positive] = probability[positive] * np.log(probability[positive])
        values[f"{name}_entropy"] = -terms.sum(axis=1, dtype=np.float64)
        values[f"{name}_top2_margin"] = (
            probability[row, order[:, 0]] - probability[row, order[:, 1]]
        )
    for left, right in (("image", "geo"), ("image", "raw"), ("geo", "raw")):
        values[f"{left}_{right}_agree"] = (
            values[f"{left}_pred"] == values[f"{right}_pred"]
        )
        if (left, right) != ("image", "raw"):
            values[f"{left}_{right}_pred_pair"] = (
                values[f"{left}_pred"] * 18 + values[f"{right}_pred"]
            )
    for right in ("image", "raw"):
        for suffix, feature in (
            ("confidence", "confidence"),
            ("entropy", "entropy"),
            ("margin", "top2_margin"),
        ):
            values[f"geo_minus_{right}_{suffix}"] = (
                values[f"geo_{feature}"] - values[f"{right}_{feature}"]
            )
    image, geo, raw = (probabilities[name] for name in ("image", "geo", "raw"))
    values["image_geo_top3_overlap"] = (
        orders["image"][:, :3, None] == orders["geo"][:, None, :3]
    ).sum(axis=(1, 2), dtype=np.int64)
    values["raw_rank_of_geo_pred"] = (
        np.argmax(orders["raw"] == values["geo_pred"][:, None], axis=1) + 1
    )
    midpoint = 0.5 * (image + geo)
    divergences = []
    for probability in (image, geo):
        terms = np.zeros_like(probability)
        positive = probability > 0
        terms[positive] = probability[positive] * np.log(
            probability[positive] / midpoint[positive]
        )
        divergences.append(terms.sum(axis=1, dtype=np.float64))
    values["image_geo_jsd"] = 0.5 * (divergences[0] + divergences[1])
    values["image_geo_total_variation"] = 0.5 * np.abs(image - geo).sum(
        axis=1, dtype=np.float64
    )
    values["image_probability_at_geo_pred"] = image[row, values["geo_pred"]]
    values["geo_probability_at_image_pred"] = geo[row, values["image_pred"]]
    values["raw_probability_at_geo_pred"] = raw[row, values["geo_pred"]]
    return values


def build_semantic_values(
    probabilities: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reconstruct the frozen M3 semantic fields without importing fit libraries.

    Inputs must be the three already-transferred float64 probability matrices,
    keyed by ``image``, ``geo``, and ``raw``. This function fits no calibration
    state and reads no labels, files, or model artifacts.
    """
    if not isinstance(probabilities, dict) or set(probabilities) != {
        "image",
        "geo",
        "raw",
    }:
        raise LegacyRouterReportError("semantic inputs require image, geo, and raw")
    matrices = {}
    for name, value in probabilities.items():
        matrix = np.asarray(value)
        if (
            matrix.dtype != np.dtype("float64")
            or matrix.ndim != 2
            or matrix.shape[1] != 18
            or len(matrix) == 0
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0)
            or np.any(matrix > 1)
            or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8, rtol=0)
        ):
            raise LegacyRouterReportError(
                f"{name}: finite normalized float64 (n,18) probabilities required"
            )
        matrices[name] = matrix
    if len({matrix.shape for matrix in matrices.values()}) != 1:
        raise LegacyRouterReportError("semantic probability matrices must align")
    return _semantic_values(matrices)


def _validated_experts(experts: pd.DataFrame, labels: pd.DataFrame):
    _require_frame(
        experts, (*IDENTITY, "file", *FEATURE_COLUMNS, *PROBABILITY_COLUMNS), "experts"
    )
    _label_guard(experts, "experts")
    _identity(experts, "experts", with_file=True)
    if (
        set(experts.training_seed) != set(SEEDS)
        or not experts.groupby("row_uid").training_seed.nunique().eq(4).all()
    ):
        raise LegacyRouterReportError("experts: every image must retain all four seeds")
    for column in ("file", "plot_idx", "family", "checkpoint_family"):
        if column in experts:
            if (
                experts[column].isna().any()
                or not experts.groupby("row_uid")[column].nunique().eq(1).all()
            ):
                raise LegacyRouterReportError(
                    f"experts: {column} differs between seed realizations"
                )
            if (
                column in ("family", "checkpoint_family")
                and experts[column].nunique() != 1
            ):
                raise LegacyRouterReportError(
                    "evaluate_family requires exactly one checkpoint family"
                )
            if (
                column in ("family", "checkpoint_family")
                and not experts[column].isin(CHECKPOINT_FAMILIES).all()
            ):
                raise LegacyRouterReportError("unknown legacy checkpoint family")
    if (
        "family" in experts
        and "checkpoint_family" in experts
        and not np.array_equal(experts.family, experts.checkpoint_family)
    ):
        raise LegacyRouterReportError("conflicting checkpoint family metadata")
    probability = {
        column: _probability(experts, column) for column in PROBABILITY_COLUMNS
    }
    calibrated = {
        name: probability[f"{name}_prob_calibrated"] for name in ("image", "geo", "raw")
    }
    for column, expected in build_semantic_values(calibrated).items():
        observed = experts[column].to_numpy()
        if expected.dtype.kind == "b":
            valid = observed.dtype.kind == "b" and np.array_equal(observed, expected)
        elif expected.dtype.kind in "iu":
            valid = observed.dtype.kind in "iu" and np.array_equal(observed, expected)
        else:
            valid = (
                observed.dtype == np.dtype("float64")
                and np.isfinite(observed).all()
                and np.allclose(observed, expected, rtol=1e-12, atol=1e-12)
            )
        if not valid:
            raise LegacyRouterReportError(
                f"experts: M3 semantic reconstruction mismatch: {column}"
            )
    for name in ("image", "geo", "raw"):
        if not np.array_equal(
            probability[f"{name}_prob_native_t1"].argmax(axis=1),
            experts[f"{name}_pred"],
        ):
            raise LegacyRouterReportError(
                f"experts: {name} native/transferred prediction mismatch"
            )
    _require_frame(labels, LABEL_COLUMNS, "labels")
    if labels.row_uid.duplicated().any() or labels.file.duplicated().any():
        raise LegacyRouterReportError("labels: duplicate image or file identity")
    for column in ("row_uid", "file", "plot_idx", "label_name"):
        if (
            not labels[column]
            .map(lambda v: isinstance(v, str) and bool(v.strip()))
            .all()
        ):
            raise LegacyRouterReportError(f"labels: {column} must be nonempty strings")
    _integers(labels, "label_id_dense", range(18))
    _integers(labels, "canonical_l3_id", [item[1] for item in ONTOLOGY])
    dense = labels.label_id_dense.to_numpy()
    if not np.array_equal(
        labels.canonical_l3_id, [ONTOLOGY[int(c)][1] for c in dense]
    ) or not np.array_equal(labels.label_name, [ONTOLOGY[int(c)][2] for c in dense]):
        raise LegacyRouterReportError("labels: fixed 18-class ontology mismatch")
    if not labels.groupby("plot_idx").label_id_dense.nunique().eq(1).all():
        raise LegacyRouterReportError(
            "labels: a complete plot has multiple habitat strata"
        )
    if set(labels.row_uid) != set(experts.row_uid):
        raise LegacyRouterReportError("labels: exact image coverage mismatch")
    indexed = labels.set_index("row_uid")
    for column in ("file", "plot_idx"):
        if not np.array_equal(experts[column], experts.row_uid.map(indexed[column])):
            raise LegacyRouterReportError(f"labels: exact {column} identity mismatch")
    targets = experts.loc[
        :,
        [
            "row_uid",
            "file",
            "plot_idx",
            "training_seed",
            "image_pred",
            "geo_pred",
            "raw_pred",
            "image_geo_pred_pair",
            "geo_raw_pred_pair",
        ],
    ].copy()
    if "checkpoint_family" in experts:
        targets["checkpoint_family"] = experts.checkpoint_family.to_numpy()
    elif "family" in experts:
        targets["checkpoint_family"] = experts.family.to_numpy()
    for column in ("label_id_dense", "canonical_l3_id", "label_name"):
        targets[column] = targets.row_uid.map(indexed[column])
    y = targets.label_id_dense.to_numpy(dtype=np.int64)
    raw_correct, geo_correct = (
        targets.raw_pred.to_numpy() == y,
        targets.geo_pred.to_numpy() == y,
    )
    targets["target_state"] = np.select(
        [
            ~raw_correct & geo_correct,
            raw_correct & ~geo_correct,
            raw_correct & geo_correct,
        ],
        TARGET_ORDER[:3],
        default="both_wrong",
    )
    targets.insert(
        0, "schema_version", "geo_helpfulness.legacy_router_scoring_targets.v1"
    )
    return targets.reset_index(drop=True), probability


def _validated_routers(routers: pd.DataFrame, experts: pd.DataFrame):
    _require_frame(
        routers, (*IDENTITY, "variant_id", "C_hex", *Q_COLUMNS, "score"), "routers"
    )
    _label_guard(routers, "routers")
    if set(routers.variant_id) != set(VARIANTS):
        raise LegacyRouterReportError(
            "routers: exactly the three frozen variants required"
        )
    results = {}
    for name in VARIANTS:
        frame = routers.loc[routers.variant_id.eq(name)].reset_index(drop=True)
        _aligned(frame, experts, name)
        if frame.C_hex.nunique() != 1 or frame.C_hex.iloc[0] not in {
            float(c).hex() for c in (0.01, 0.1, 1.0, 10.0)
        }:
            raise LegacyRouterReportError(f"{name}: one frozen common C required")
        if any(
            frame[column].dtype != np.dtype("float64")
            for column in (*Q_COLUMNS, "score")
        ):
            raise LegacyRouterReportError(
                f"{name}: router probabilities/scores must be float64"
            )
        q = frame.loc[:, list(Q_COLUMNS)].to_numpy()
        if (
            not np.isfinite(q).all()
            or np.any(q < 0)
            or np.any(q > 1)
            or not np.allclose(q.sum(axis=1), 1.0, atol=1e-8, rtol=0)
        ):
            raise LegacyRouterReportError(f"{name}: invalid four-state probabilities")
        if not np.array_equal(frame.score.to_numpy(), q[:, 0] - q[:, 1]):
            raise LegacyRouterReportError(
                f"{name}: score must equal q_rescue minus q_harm"
            )
        results[name] = frame
    return results


def _threshold(text: Any) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise LegacyRouterReportError(
                    "threshold specification has duplicate JSON keys"
                )
            result[key] = value
        return result

    if not isinstance(text, str):
        raise LegacyRouterReportError("threshold specification must be a JSON string")
    try:
        specification = json.loads(text, object_pairs_hook=pairs)
        validate_threshold_specification(specification)
    except (ValueError, TypeError) as exc:
        raise LegacyRouterReportError("invalid frozen threshold specification") from exc
    return specification


def _validated_actions(actions: pd.DataFrame, experts: pd.DataFrame, routers: dict):
    _require_frame(
        actions,
        (
            *IDENTITY,
            "policy_id",
            "score",
            "threshold_specification",
            "acted",
            "raw_pred",
            "geo_pred",
            "final_pred",
        ),
        "actions",
    )
    _label_guard(actions, "actions")
    if set(actions.policy_id) != set(POLICIES):
        raise LegacyRouterReportError(
            "actions: exactly the eight frozen policies required"
        )
    raw, geo = experts.raw_pred.to_numpy(), experts.geo_pred.to_numpy()
    zero_spec = {
        "kind": "finite",
        "value_hex": float(0).hex(),
        "comparison": "strict_gt",
    }
    never_spec = {
        "kind": "never_intervene",
        "value_hex": None,
        "comparison": "strict_gt",
    }
    expected_scores = {
        **{name: frame.score.to_numpy() for name, frame in routers.items()},
        "never_intervene": np.zeros(len(experts), dtype=np.float64),
        "always_geo_on_geo_raw_disagreement": np.ones(len(experts), dtype=np.float64),
        "higher_calibrated_geo_confidence_on_geo_raw_disagreement": experts.geo_minus_raw_confidence.to_numpy(),
        "image_geo_jsd": experts.image_geo_jsd.to_numpy(),
        "geo_minus_raw_margin": experts.geo_minus_raw_margin.to_numpy(),
    }
    frames, masks, specifications = {}, {}, {}
    for name in POLICIES:
        frame = actions.loc[actions.policy_id.eq(name)].reset_index(drop=True)
        _aligned(frame, experts, name)
        for column in ("raw_pred", "geo_pred", "final_pred"):
            _integers(frame, column, range(18))
        if not np.array_equal(frame.raw_pred, raw) or not np.array_equal(
            frame.geo_pred, geo
        ):
            raise LegacyRouterReportError(f"{name}: expert prediction mismatch")
        if frame.score.dtype != np.dtype("float64") or not np.array_equal(
            frame.score.to_numpy(), expected_scores[name]
        ):
            raise LegacyRouterReportError(f"{name}: frozen policy score mismatch")
        if frame.acted.dtype != np.dtype("bool"):
            raise LegacyRouterReportError(f"{name}: effective actions must be Boolean")
        if frame.threshold_specification.nunique() != 1:
            raise LegacyRouterReportError(
                f"{name}: one frozen common threshold required"
            )
        specification = _threshold(frame.threshold_specification.iloc[0])
        fixed = never_spec if name == "never_intervene" else zero_spec
        if name in POLICIES[3:6] and specification != fixed:
            raise LegacyRouterReportError(
                f"{name}: fixed comparator threshold mismatch"
            )
        mask = frame.acted.to_numpy()
        expected = apply_heuristic_threshold(
            expected_scores[name], raw, geo, specification
        )
        if not np.array_equal(mask, expected) or not np.array_equal(
            frame.final_pred, np.where(mask, geo, raw)
        ):
            raise LegacyRouterReportError(
                f"{name}: frozen threshold/fallback action mismatch"
            )
        frames[name], masks[name], specifications[name] = frame, mask, specification
    return frames, masks, specifications


def _seed_records(
    targets: pd.DataFrame, score, summary_fields: tuple[str, ...]
) -> dict:
    per_seed = {
        str(seed): score(targets.training_seed.eq(seed).to_numpy()) for seed in SEEDS
    }
    return {
        "per_seed": per_seed,
        "pooled_seed_realizations": score(np.ones(len(targets), dtype=bool)),
        "seed_summary": {
            name: seed_summary([record[name] for record in per_seed.values()])
            for name in summary_fields
        },
    }


def _classification_record(
    targets: pd.DataFrame, probability: np.ndarray, prediction: np.ndarray
) -> dict:
    def one(mask):
        rows = targets.loc[mask]
        y, raw = rows.label_id_dense.to_numpy(dtype=np.int64), rows.raw_pred.to_numpy(
            dtype=np.int64
        )
        pred, p = prediction[mask], probability[mask]
        result = classification_metrics(pred, y, 18)
        result["top1_accuracy"] = result["accuracy"]
        top3 = np.any(
            np.argsort(-p, axis=1, kind="stable")[:, :3] == y[:, None], axis=1
        )
        result["top3_accuracy"] = float(top3.mean())
        for item, (c, canonical, name) in zip(result["per_class"], ONTOLOGY):
            truth, raw_c, final_c = y == c, raw == c, pred == c
            support = rows.loc[truth]
            item.update(
                {
                    "canonical_l3_id": canonical,
                    "label_name": name,
                    "unique_images": int(support.row_uid.nunique()),
                    "unique_plots": int(support.plot_idx.nunique()),
                    "tp_rescued": int(np.sum(truth & ~raw_c & final_c)),
                    "tp_lost": int(np.sum(truth & raw_c & ~final_c)),
                    "fp_introduced": int(np.sum(~truth & ~raw_c & final_c)),
                    "fp_removed": int(np.sum(~truth & raw_c & ~final_c)),
                    "top1_accuracy": (
                        float(np.mean(pred[truth] == y[truth])) if truth.any() else None
                    ),
                    "top3_accuracy": float(top3[truth].mean()) if truth.any() else None,
                }
            )
            item["low_support"] = (
                item["unique_images"] < 20 or item["unique_plots"] < 10
            )
        return result

    return _seed_records(
        targets,
        one,
        (
            "accuracy",
            "top1_accuracy",
            "top3_accuracy",
            "macro_f1",
            "weighted_f1",
            "mcc",
        ),
    )


def _router_record(targets: pd.DataFrame, frame: pd.DataFrame) -> dict:
    q = frame.loc[:, list(Q_COLUMNS)].to_numpy()
    y = np.asarray(
        [TARGET_ORDER.index(value) for value in targets.target_state], dtype=np.int64
    )

    def one(mask):
        probability, label = q[mask], y[mask]
        result = {
            **classification_metrics(probability.argmax(axis=1), label, 4),
            **multiclass_metrics(probability, label),
            "rescue_pr_auc": binary_average_precision(label == 0, probability[:, 0]),
            "harm_pr_auc": binary_average_precision(label == 1, probability[:, 1]),
        }
        for item, state in zip(result["per_class"], TARGET_ORDER):
            item["target_state"] = state
        return result

    return {
        "C_hex": frame.C_hex.iloc[0],
        "target_order": list(TARGET_ORDER),
        "average_precision_convention": "noninterpolated_with_exact_score_ties_grouped",
        **_seed_records(
            targets,
            one,
            (
                "log_loss",
                "brier_score",
                "top1_accuracy",
                "macro_f1",
                "weighted_f1",
                "mcc",
                "rescue_pr_auc",
                "harm_pr_auc",
            ),
        ),
    }


def _table(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(
        frame.reset_index(drop=True), preserve_index=False
    ).replace_schema_metadata(None)


def evaluate_family(
    experts: pd.DataFrame,
    routers: pd.DataFrame,
    actions: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    """Score one sealed family against caller-supplied cleaned-test labels.

    All four training seeds and exactly 3 router variants / 8 frozen policies are
    required. Counts are pooled seed realizations where marked; image and plot
    support are deduplicated. No test metric can change a supplied policy.
    """
    targets, probability = _validated_experts(experts, labels)
    router_frames = _validated_routers(routers, experts)
    action_frames, masks, specifications = _validated_actions(
        actions, experts, router_frames
    )
    context = {
        "checkpoint_family": (
            targets.checkpoint_family.iloc[0]
            if "checkpoint_family" in targets
            else None
        ),
        "training_seeds": list(SEEDS),
        "checkpoint_selection_disclosure": CHECKPOINT_SELECTION_DISCLOSURE,
    }
    expert_results = {}
    classification = {
        **context,
        "interpretation": INTERPRETATION,
        "class_count": 18,
        "macro_f1_includes_absent_classes": True,
        "top3_probability_basis": "transferred_M4_temperature_selected_expert_probability",
        "calibration_disclosure": CALIBRATION_DISCLOSURE,
        "experts": {},
        "policies": {},
    }
    y = targets.label_id_dense.to_numpy(dtype=np.int64)
    for name in ("image", "geo", "raw"):
        expert_results[name] = {}
        for basis in ("native_t1", "calibrated"):
            p = probability[f"{name}_prob_{basis}"]
            expert_results[name][basis] = {
                **context,
                "probability_basis": (
                    "native_temperature_one"
                    if basis == "native_t1"
                    else "transferred_M4_OOF_temperature"
                ),
                "calibration_disclosure": CALIBRATION_DISCLOSURE,
                **_seed_records(
                    targets,
                    lambda mask: calibration_metrics(p[mask], y[mask]),
                    (
                        "log_loss",
                        "brier_score",
                        "top1_accuracy",
                        "top3_accuracy",
                        "ece",
                    ),
                ),
            }
        classification["experts"][name] = _classification_record(
            targets,
            probability[f"{name}_prob_calibrated"],
            targets[f"{name}_pred"].to_numpy(dtype=np.int64),
        )
    policies, outcomes = {}, []
    for name in POLICIES:
        frame, mask = action_frames[name], masks[name]
        policies[name] = {
            **context,
            **policy_metrics(targets, mask),
            "threshold_specification": specifications[name],
            "interpretation": INTERPRETATION,
            "selection_or_fitting_performed": False,
            "M4_gate_rerun": False,
            "raw_fallback_selected": specifications[name]["kind"] == "never_intervene",
            "all_predictions_equal_raw": bool(
                np.array_equal(frame.final_pred, targets.raw_pred)
            ),
        }
        selected = np.where(
            mask[:, None],
            probability["geo_prob_calibrated"],
            probability["raw_prob_calibrated"],
        )
        classification["policies"][name] = _classification_record(
            targets, selected, frame.final_pred.to_numpy(dtype=np.int64)
        )
        outcome = frame.copy()
        for column in (
            "file",
            "label_id_dense",
            "canonical_l3_id",
            "label_name",
            "target_state",
        ):
            outcome[column] = targets[column].to_numpy()
        for expert in ("raw", "geo", "final"):
            outcome[f"{expert}_correct"] = outcome[f"{expert}_pred"].to_numpy() == y
        outcome["rescued"] = mask & targets.target_state.eq("rescue").to_numpy()
        outcome["harmed"] = mask & targets.target_state.eq("harm").to_numpy()
        outcome["net_correct"] = outcome.rescued.to_numpy(
            dtype=np.int8
        ) - outcome.harmed.to_numpy(dtype=np.int8)
        outcomes.append(outcome)
    bootstrap, replicates = bootstrap_policy_comparison(
        targets, masks, replicates=2000, random_seed=20260826
    )
    bootstrap.update(
        {
            **context,
            "interpretation": INTERPRETATION,
            "policy_selection_refit_in_resamples": False,
            "calibration_disclosure": CALIBRATION_DISCLOSURE,
            "M4_gate_rerun": False,
        }
    )
    return {
        "expert_metrics.json": expert_results,
        "classification.json": classification,
        "router_metrics.json": {
            name: {**context, **_router_record(targets, frame)}
            for name, frame in router_frames.items()
        },
        "policy_metrics.json": policies,
        "bootstrap_summary.json": bootstrap,
        "bootstrap_replicates.parquet": _table(replicates),
        "subgroup_metrics.parquet": _table(subgroup_policy_report(targets, masks)),
        "actions_with_outcomes.parquet": _table(
            pd.concat(outcomes, ignore_index=True).sort_values(
                ["policy_id", *KEY], kind="stable"
            )
        ),
        "scoring_targets.parquet": _table(targets),
    }


__all__ = ["LegacyRouterReportError", "build_semantic_values", "evaluate_family"]

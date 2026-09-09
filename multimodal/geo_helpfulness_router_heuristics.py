"""Pure OOF-only threshold selection for the two predeclared M4 heuristics.

The caller supplies validated training audit rows and four-state targets. This
module discovers no artifacts and reads no validation labels or expert outputs.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_router_candidate_state import (
    CATEGORICAL_FEATURE_SIZES,
    NUMERIC_INPUT_COLUMNS,
    canonical_sha256,
)


SEEDS = (1, 2, 3, 4)
SCORES = ("image_geo_jsd", "geo_minus_raw_margin")
TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
TARGET_COLUMNS = (
    "schema_version",
    "protocol_id",
    "row_uid",
    "plot_idx",
    "training_seed",
    "target_state",
)
KEY = ("row_uid", "training_seed")
AUDIT_REQUIRED = (
    "schema_version",
    "protocol_id",
    "row_uid",
    "plot_idx",
    "training_seed",
    "train_oof_fold",
    "raw_pred",
    "geo_pred",
    *SCORES,
)
AUDIT_ALLOWED = frozenset(
    (
        *TARGET_COLUMNS[:-1],
        "train_oof_fold",
        "image_prob_calibrated",
        "geo_prob_calibrated",
        "raw_prob_calibrated",
        *NUMERIC_INPUT_COLUMNS,
        *[name for name, _ in CATEGORICAL_FEATURE_SIZES],
    )
)
GUARDRAILS = {
    "min_pooled_coverage_fraction": 0.01,
    "min_unique_acted_plots": 20,
    "max_pooled_harmed_to_rescued_ratio": 0.5,
    "require_positive_net_utility": True,
}
NEVER_THRESHOLD = {
    "kind": "never_intervene",
    "value_hex": None,
    "comparison": "strict_gt",
}
FIXED_COMPARATORS = (
    {"id": "never_intervene", "role": "fixed", "effective_action": "never"},
    {
        "id": "always_geo_on_geo_raw_disagreement",
        "role": "fixed",
        "effective_action": "geo_pred != raw_pred",
    },
    {
        "id": "higher_calibrated_geo_confidence_on_geo_raw_disagreement",
        "role": "principal",
        "effective_action": "(geo_pred != raw_pred) and (geo_confidence > raw_confidence)",
        "confidence_basis": "scalar_temperature_calibrated",
        "tie_action": "keep_raw",
    },
)


class RouterHeuristicError(ValueError):
    """The supplied training inputs or policy violate the frozen contract."""


def _finite_scores(values: Any) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype != np.dtype("float64"):
        raise RouterHeuristicError("scores must be a one-dimensional float64 vector")
    if not np.isfinite(result).all():
        raise RouterHeuristicError("scores must be finite")
    return result


def _predictions(values: Any, row_count: int) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (row_count,) or result.dtype.kind not in "iu":
        raise RouterHeuristicError("predictions must be matching integer vectors")
    if np.any(result < 0) or np.any(result >= 18):
        raise RouterHeuristicError("predictions must be dense class IDs in 0..17")
    return result


def validate_threshold_specification(specification: Mapping[str, Any]) -> float | None:
    """Return the finite threshold, or None for the typed never-action policy."""
    if not isinstance(specification, Mapping) or set(specification) != set(
        NEVER_THRESHOLD
    ):
        raise RouterHeuristicError(
            "threshold specification has missing or unknown fields"
        )
    if specification["comparison"] != "strict_gt":
        raise RouterHeuristicError("threshold comparison must be strict_gt")
    if specification["kind"] == "never_intervene":
        if specification["value_hex"] is not None:
            raise RouterHeuristicError(
                "never-intervene threshold must have a null value"
            )
        return None
    if specification["kind"] != "finite" or not isinstance(
        specification["value_hex"], str
    ):
        raise RouterHeuristicError("invalid finite threshold specification")
    try:
        value = float.fromhex(specification["value_hex"])
    except ValueError as exc:
        raise RouterHeuristicError("threshold is not a hexadecimal float") from exc
    if (
        not math.isfinite(value)
        or value < 0.0
        or specification["value_hex"] != value.hex()
        or (value == 0.0 and specification["value_hex"] != (0.0).hex())
    ):
        raise RouterHeuristicError(
            "threshold must be canonical finite nonnegative float64"
        )
    return value


def apply_heuristic_threshold(
    scores: Any,
    raw_pred: Any,
    geo_pred: Any,
    threshold_specification: Mapping[str, Any],
) -> np.ndarray:
    """Apply a frozen threshold; effective actions always require disagreement."""
    values = _finite_scores(scores)
    raw = _predictions(raw_pred, len(values))
    geo = _predictions(geo_pred, len(values))
    threshold = validate_threshold_specification(threshold_specification)
    if threshold is None:
        return np.zeros(len(values), dtype=bool)
    return (values > threshold) & (raw != geo)


def enumerate_threshold_candidates(scores: Any) -> list[dict[str, Any]]:
    """Never, zero, and deduplicated float64 midpoints of adjacent positive scores."""
    values = _finite_scores(scores)
    positive = np.unique(values[values > 0.0])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        midpoints = positive[:-1] + (positive[1:] - positive[:-1]) / np.float64(2.0)
    finite = np.unique(np.concatenate((np.array([0.0], dtype=np.float64), midpoints)))
    if not np.isfinite(finite).all():
        raise RouterHeuristicError("threshold midpoint computation became nonfinite")
    return [deepcopy(NEVER_THRESHOLD)] + [
        {"kind": "finite", "value_hex": float(value).hex(), "comparison": "strict_gt"}
        for value in finite
    ]


def _join_training_inputs(
    audit_rows: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    protocol_id: str,
    expected_rows_per_seed: int,
    expected_plots_per_seed: int,
) -> pd.DataFrame:
    for name, value in (
        ("expected_rows_per_seed", expected_rows_per_seed),
        ("expected_plots_per_seed", expected_plots_per_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RouterHeuristicError(f"{name} must be a positive integer")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise RouterHeuristicError("protocol_id must be a nonempty string")
    if not isinstance(audit_rows, pd.DataFrame) or not isinstance(
        targets, pd.DataFrame
    ):
        raise RouterHeuristicError("audit rows and targets must be DataFrames")
    if (
        audit_rows.columns.duplicated().any()
        or not set(AUDIT_REQUIRED).issubset(audit_rows.columns)
        or not set(audit_rows.columns).issubset(AUDIT_ALLOWED)
    ):
        raise RouterHeuristicError(
            "audit schema is missing required or includes forbidden columns"
        )
    if list(targets.columns) != list(TARGET_COLUMNS):
        raise RouterHeuristicError("targets must use the exact six-column M3 schema")
    audit = audit_rows.loc[:, list(AUDIT_REQUIRED)].copy()
    target = targets.copy()
    for frame in (audit, target):
        if frame.isna().any().any() or frame.loc[:, list(KEY)].duplicated().any():
            raise RouterHeuristicError(
                "missing values or duplicate row_uid/training_seed keys"
            )
        for name in ("row_uid", "plot_idx"):
            if (
                not frame[name]
                .map(lambda value: isinstance(value, str) and bool(value))
                .all()
            ):
                raise RouterHeuristicError(f"{name} must contain nonempty strings")
        if not frame.protocol_id.eq(protocol_id).all():
            raise RouterHeuristicError("protocol identity mismatch")
        if not frame.schema_version.eq("geo_helpfulness.router_targets.v1").all():
            raise RouterHeuristicError("identity schema version mismatch")
        if frame.training_seed.dtype.kind not in "iu" or set(
            frame.training_seed
        ) != set(SEEDS):
            raise RouterHeuristicError("training seeds must be exactly 1,2,3,4")
        if len(frame) != len(SEEDS) * expected_rows_per_seed:
            raise RouterHeuristicError("pooled training row count mismatch")
        reference_rows = set(frame.loc[frame.training_seed.eq(1), "row_uid"])
        for seed in SEEDS:
            rows = frame.loc[frame.training_seed.eq(seed)]
            if (
                len(rows) != expected_rows_per_seed
                or rows.plot_idx.nunique() != expected_plots_per_seed
                or set(rows.row_uid) != reference_rows
            ):
                raise RouterHeuristicError("seed-specific image/plot coverage mismatch")
        if frame.groupby("row_uid", sort=False).plot_idx.nunique().max() != 1:
            raise RouterHeuristicError("plot identity differs across seed realizations")
    if not set(target.target_state).issubset(TARGET_ORDER):
        raise RouterHeuristicError("unknown four-state target")
    if (
        audit.train_oof_fold.dtype.kind not in "iu"
        or not audit.train_oof_fold.isin(range(4)).all()
    ):
        raise RouterHeuristicError("OOF folds must be integer IDs 0..3")
    if audit.groupby("row_uid", sort=False).train_oof_fold.nunique().max() != 1:
        raise RouterHeuristicError("OOF fold identity differs across seed realizations")
    _predictions(audit.raw_pred.to_numpy(), len(audit))
    _predictions(audit.geo_pred.to_numpy(), len(audit))
    for name in SCORES:
        values = _finite_scores(audit[name].to_numpy())
        lower, upper = (0.0, math.log(2.0)) if name == "image_geo_jsd" else (-1.0, 1.0)
        if np.any(values < lower) or np.any(values > upper):
            raise RouterHeuristicError(f"{name} outside its semantic range")
    joined = audit.merge(
        target,
        on=list(KEY),
        how="outer",
        validate="one_to_one",
        suffixes=("", "_target"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RouterHeuristicError("audit and target keys do not match")
    for name in ("schema_version", "protocol_id", "plot_idx"):
        if not joined[name].eq(joined[f"{name}_target"]).all():
            raise RouterHeuristicError("audit/target identity disagreement")
    disagreement = joined.raw_pred.ne(joined.geo_pred)
    if (joined.target_state.isin(("rescue", "harm")) & ~disagreement).any() or (
        joined.target_state.eq("both_correct") & disagreement
    ).any():
        raise RouterHeuristicError("four-state targets contradict expert agreement")
    return (
        joined.loc[:, [*AUDIT_REQUIRED, "target_state"]]
        .sort_values(list(KEY), kind="stable")
        .reset_index(drop=True)
    )


def _count_summary(rows: pd.DataFrame, actions: np.ndarray) -> dict[str, Any]:
    acted = rows.loc[actions]
    rescues = int(acted.target_state.eq("rescue").sum())
    harms = int(acted.target_state.eq("harm").sum())
    return {
        "row_count": len(rows),
        "acted_count": int(actions.sum()),
        "rescued_count": rescues,
        "harmed_count": harms,
        "rescued_minus_harmed": rescues - harms,
        "coverage_fraction": float(actions.mean()),
        "unique_acted_plots": int(acted.plot_idx.nunique()),
        "harmed_to_rescued_ratio": float(harms / rescues) if rescues else None,
    }


def _selection_counts(rows: pd.DataFrame, actions: np.ndarray) -> dict[str, Any]:
    per_seed = {}
    for seed in SEEDS:
        mask = rows.training_seed.eq(seed).to_numpy()
        per_seed[str(seed)] = _count_summary(rows.loc[mask], actions[mask])
    pooled = _count_summary(rows, actions)
    return {
        "per_seed": per_seed,
        "pooled": pooled,
        "mean_rescued_minus_harmed": float(
            sum(value["rescued_minus_harmed"] for value in per_seed.values())
            / len(SEEDS)
        ),
    }


def _select_score(rows: pd.DataFrame, score_name: str) -> dict[str, Any]:
    scores = rows[score_name].to_numpy(dtype=np.float64)
    specifications = enumerate_threshold_candidates(scores)
    eligible = np.flatnonzero((scores > 0) & rows.raw_pred.ne(rows.geo_pred).to_numpy())
    order = eligible[np.argsort(scores[eligible], kind="stable")]
    ordered_scores = scores[order]
    state = rows.target_state.to_numpy()
    contributions = np.column_stack(
        (
            np.ones(len(order), dtype=np.int64),
            (state[order] == "rescue").astype(np.int64),
            (state[order] == "harm").astype(np.int64),
        )
    )
    suffix = np.zeros((len(order) + 1, 3), dtype=np.int64)
    suffix[:-1] = np.cumsum(contributions[::-1], axis=0)[::-1]
    unique_plots = np.zeros(len(order) + 1, dtype=np.int64)
    seen_plots = set()
    plots = rows.plot_idx.to_numpy()
    for offset in range(len(order) - 1, -1, -1):
        seen_plots.add(plots[order[offset]])
        unique_plots[offset] = len(seen_plots)
    winner = deepcopy(NEVER_THRESHOLD)
    best_key = None
    feasible_count = 0
    for spec in specifications[1:]:
        threshold = float.fromhex(spec["value_hex"])
        position = int(np.searchsorted(ordered_scores, threshold, side="right"))
        actions, rescues, harms = map(int, suffix[position])
        net = rescues - harms
        if (
            actions * 100 < len(rows)
            or unique_plots[position] < 20
            or 2 * harms > rescues
            or net <= 0
        ):
            continue
        feasible_count += 1
        rank = (net, -harms, -actions, threshold)
        if best_key is None or rank > best_key:
            best_key, winner = rank, deepcopy(spec)
    action_mask = apply_heuristic_threshold(
        scores, rows.raw_pred.to_numpy(), rows.geo_pred.to_numpy(), winner
    )
    return {
        "score_name": score_name,
        "threshold_specification": winner,
        "threshold_specification_sha256": canonical_sha256(winner),
        "threshold_display": validate_threshold_specification(winner),
        "candidate_count": len(specifications),
        "candidate_specifications_sha256": canonical_sha256(specifications),
        "feasible_candidate_count": feasible_count,
        "selected_feasible": best_key is not None,
        "selection_reason": (
            "maximum_feasible_utility"
            if best_key is not None
            else "no_feasible_candidate_never_intervene"
        ),
        **_selection_counts(rows, action_mask),
    }


def _declaration(rows: pd.DataFrame, protocol_id: str) -> dict[str, Any]:
    return {
        "schema_version": "geo_helpfulness.oof_heuristic_thresholds.v1",
        "protocol_id": protocol_id,
        "fit_role": "development_train_oof",
        "training_seeds": list(SEEDS),
        "row_count": len(rows),
        "unique_images": int(rows.row_uid.nunique()),
        "unique_plots": int(rows.plot_idx.nunique()),
        "seed_aggregation": "retain_all_realizations_no_averaging",
        "boundary_pooling": "all_development_train_oof_seed_realizations",
        "selection_projection_sha256": canonical_sha256(rows.to_dict(orient="records")),
        "guardrails": deepcopy(GUARDRAILS),
        "objective": "maximize_mean_rescued_minus_harmed_across_training_seeds",
        "tie_break_order": ["fewer_harms", "fewer_actions", "higher_threshold"],
        "effective_action": {
            "score_comparison": "strict_gt",
            "require_geo_raw_disagreement": True,
        },
        "threshold_candidate_rule": {
            "include_never_intervene": True,
            "include_zero": True,
            "include_midpoints_between_adjacent_unique_positive_scores": True,
            "allow_negative": False,
            "unique_score_equality": "exact_float64_value",
            "midpoint_formula": "lower_plus_half_of_upper_minus_lower_in_float64",
            "candidate_deduplication": "exact_float64_value",
        },
        "fixed_comparators": deepcopy(list(FIXED_COMPARATORS)),
        "validation_outputs_opened": False,
        "validation_labels_scored": False,
        "test_sources_opened": False,
    }


def select_oof_heuristic_thresholds(
    audit_rows: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    protocol_id: str,
    expected_rows_per_seed: int = 3378,
    expected_plots_per_seed: int = 1300,
) -> dict[str, Any]:
    """Select each secondary heuristic once using only four-seed training OOF rows.

    Expected count arguments exist for synthetic tests; they do not change policy
    guardrails. Cross-seed pooling defines boundaries and pooled support, without
    treating the seed realizations as independent biological samples.
    """
    rows = _join_training_inputs(
        audit_rows,
        targets,
        protocol_id=protocol_id,
        expected_rows_per_seed=expected_rows_per_seed,
        expected_plots_per_seed=expected_plots_per_seed,
    )
    return {
        **_declaration(rows, protocol_id),
        "heuristics": {name: _select_score(rows, name) for name in SCORES},
    }


def validate_frozen_oof_heuristics(
    state: Mapping[str, Any],
    audit_rows: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    protocol_id: str,
    expected_rows_per_seed: int = 3378,
    expected_plots_per_seed: int = 1300,
) -> None:
    """Check a saved state by application only; never enumerate or select thresholds.

    The caller must authenticate the saved artifact against its immutable bundle
    manifest (or matching prelabel declaration for an interrupted build). This
    check reproduces selected-policy counts and input identity, not optimality.
    """
    rows = _join_training_inputs(
        audit_rows,
        targets,
        protocol_id=protocol_id,
        expected_rows_per_seed=expected_rows_per_seed,
        expected_plots_per_seed=expected_plots_per_seed,
    )
    expected = _declaration(rows, protocol_id)
    if not isinstance(state, Mapping) or set(state) != set(expected) | {"heuristics"}:
        raise RouterHeuristicError(
            "frozen heuristic state has missing or unknown fields"
        )
    if canonical_sha256({key: state[key] for key in expected}) != canonical_sha256(
        expected
    ):
        raise RouterHeuristicError(
            "frozen heuristic input identity or policy declaration differs"
        )
    if not isinstance(state["heuristics"], Mapping) or set(state["heuristics"]) != set(
        SCORES
    ):
        raise RouterHeuristicError("frozen heuristic score set differs")
    keys = {
        "score_name",
        "threshold_specification",
        "threshold_specification_sha256",
        "threshold_display",
        "candidate_count",
        "candidate_specifications_sha256",
        "feasible_candidate_count",
        "selected_feasible",
        "selection_reason",
        "per_seed",
        "pooled",
        "mean_rescued_minus_harmed",
    }
    for name in SCORES:
        selected = state["heuristics"][name]
        if (
            not isinstance(selected, Mapping)
            or set(selected) != keys
            or selected["score_name"] != name
        ):
            raise RouterHeuristicError("invalid frozen score record")
        threshold = validate_threshold_specification(
            selected["threshold_specification"]
        )
        if selected["threshold_specification_sha256"] != canonical_sha256(
            selected["threshold_specification"]
        ):
            raise RouterHeuristicError("frozen threshold hash differs")
        if canonical_sha256(selected["threshold_display"]) != canonical_sha256(
            threshold
        ):
            raise RouterHeuristicError("frozen threshold display differs")
        for field in ("candidate_count", "feasible_candidate_count"):
            if type(selected[field]) is not int:
                raise RouterHeuristicError("frozen candidate counts must be integers")
        if not (
            selected["candidate_count"] >= 2
            and 0 <= selected["feasible_candidate_count"] < selected["candidate_count"]
        ):
            raise RouterHeuristicError("frozen candidate counts are inconsistent")
        digest = selected["candidate_specifications_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RouterHeuristicError(
                "candidate specification hash must be a lowercase SHA256"
            )
        actions = apply_heuristic_threshold(
            rows[name].to_numpy(),
            rows.raw_pred.to_numpy(),
            rows.geo_pred.to_numpy(),
            selected["threshold_specification"],
        )
        counts = _selection_counts(rows, actions)
        if canonical_sha256({key: selected[key] for key in counts}) != canonical_sha256(
            counts
        ):
            raise RouterHeuristicError("frozen selected-policy counts do not reproduce")
        pooled = counts["pooled"]
        feasible = (
            pooled["acted_count"] * 100 >= len(rows)
            and pooled["unique_acted_plots"] >= 20
            and 2 * pooled["harmed_count"] <= pooled["rescued_count"]
            and pooled["rescued_minus_harmed"] > 0
        )
        if threshold is None:
            valid_selection = (
                selected["selected_feasible"] is False
                and selected["feasible_candidate_count"] == 0
                and selected["selection_reason"]
                == "no_feasible_candidate_never_intervene"
            )
        else:
            valid_selection = (
                feasible
                and selected["selected_feasible"] is True
                and selected["feasible_candidate_count"] > 0
                and selected["selection_reason"] == "maximum_feasible_utility"
            )
        if not valid_selection:
            raise RouterHeuristicError(
                "frozen selection violates the guardrails or fallback contract"
            )


__all__ = [
    "RouterHeuristicError",
    "GUARDRAILS",
    "FIXED_COMPARATORS",
    "NEVER_THRESHOLD",
    "enumerate_threshold_candidates",
    "validate_threshold_specification",
    "apply_heuristic_threshold",
    "select_oof_heuristic_thresholds",
    "validate_frozen_oof_heuristics",
]

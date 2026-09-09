"""Pure scoring-only C/threshold selection and the frozen M4 feasibility gate."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256
from multimodal.geo_helpfulness_router_heuristics import (
    NEVER_THRESHOLD,
    apply_heuristic_threshold,
    enumerate_threshold_candidates,
    validate_threshold_specification,
)
from multimodal.geo_helpfulness_router_metrics import multiclass_metrics


SEEDS = (1, 2, 3, 4)
C_VALUES = (0.01, 0.1, 1.0, 10.0)
VARIANTS = ("primary_full727", "ablation_no_pairs79", "ablation_numeric25")
TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
Q_COLUMNS = tuple(f"q_{name}" for name in TARGET_ORDER)
PRINCIPAL_HEURISTIC = "higher_calibrated_geo_confidence_on_geo_raw_disagreement"
HEURISTICS = (
    "never_intervene",
    "always_geo_on_geo_raw_disagreement",
    PRINCIPAL_HEURISTIC,
    "image_geo_jsd",
    "geo_minus_raw_margin",
)
GATE_CRITERIA = (
    "positive_mean_validation_delta_accuracy",
    "positive_net_utility_in_at_least_3_of_4_seeds",
    "pooled_harmed_no_more_than_half_pooled_rescued",
    "pooled_effective_coverage_at_least_1_percent",
    "at_least_20_unique_acted_plots",
    "mean_gain_above_every_predeclared_heuristic",
    "positive_pooled_net_after_removing_best_contributing_plot",
    "development_access_audit_clean",
)


class RouterSelectionError(ValueError):
    """Scoring inputs or selection evidence violate the frozen M4 contract."""


def _keys(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return list(frame[["row_uid", "training_seed"]].itertuples(index=False, name=None))


def _validate_seed_rows(frame: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.columns.duplicated().any()
        or not set(required).issubset(frame)
    ):
        raise RouterSelectionError("missing or duplicate selection columns")
    rows = frame.loc[:, list(required)].copy()
    if rows.empty or rows.isna().any().any():
        raise RouterSelectionError("empty or missing selection values")
    if rows.training_seed.dtype.kind not in "iu" or set(rows.training_seed) != set(
        SEEDS
    ):
        raise RouterSelectionError("selection requires exactly seeds 1,2,3,4")
    for name in ("row_uid", "plot_idx"):
        if (
            name in rows
            and not rows[name]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise RouterSelectionError(f"invalid selection identity: {name}")
    if rows.duplicated(["row_uid", "training_seed"]).any():
        raise RouterSelectionError("duplicate seed/image selection identity")
    reference = set(rows.loc[rows.training_seed.eq(1), "row_uid"])
    if any(
        set(rows.loc[rows.training_seed.eq(seed), "row_uid"]) != reference
        for seed in SEEDS
    ):
        raise RouterSelectionError("selection images differ across seeds")
    if "plot_idx" in rows and rows.groupby("row_uid").plot_idx.nunique().max() != 1:
        raise RouterSelectionError("plot identities differ across seed realizations")
    if "target_state" in rows and not set(rows.target_state).issubset(TARGET_ORDER):
        raise RouterSelectionError("unknown four-state target")
    return rows.sort_values(["row_uid", "training_seed"], kind="stable").reset_index(
        drop=True
    )


def select_common_c(
    candidate_frames_by_variant: Mapping[str, pd.DataFrame],
    targets: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Choose one C per fixed variant by exact minimum mean four-seed NLL."""
    if not isinstance(candidate_frames_by_variant, Mapping) or set(
        candidate_frames_by_variant
    ) != set(VARIANTS):
        raise RouterSelectionError(
            "C selection requires all three predeclared variants"
        )
    target_rows = _validate_seed_rows(
        targets, ("row_uid", "training_seed", "target_state")
    )
    target_keys = set(_keys(target_rows))
    target_map = {
        (row.row_uid, row.training_seed): TARGET_ORDER.index(row.target_state)
        for row in target_rows.itertuples(index=False)
    }
    required = {"row_uid", "training_seed", "C_hex", *Q_COLUMNS}
    result = {}
    for variant in VARIANTS:
        frame = candidate_frames_by_variant[variant]
        if (
            not isinstance(frame, pd.DataFrame)
            or frame.columns.duplicated().any()
            or not required.issubset(frame)
        ):
            raise RouterSelectionError("candidate probability columns are incomplete")
        rows = frame.loc[:, sorted(required)].copy()
        if rows.isna().any().any() or rows.training_seed.dtype.kind not in "iu":
            raise RouterSelectionError("invalid candidate identity/probability values")
        if set(rows.C_hex) != {float(value).hex() for value in C_VALUES}:
            raise RouterSelectionError("candidate C grid differs from the frozen grid")
        if rows.duplicated(["row_uid", "training_seed", "C_hex"]).any():
            raise RouterSelectionError("duplicate candidate prediction identity")
        if any(rows[name].dtype != np.dtype("float64") for name in Q_COLUMNS):
            raise RouterSelectionError("router probabilities must retain float64 dtype")
        q_all = rows[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(q_all).all()
            or not ((q_all >= 0) & (q_all <= 1)).all()
            or not np.allclose(q_all.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise RouterSelectionError("invalid four-state candidate probabilities")
        candidate_losses = []
        for C in C_VALUES:
            selected = rows.loc[rows.C_hex.eq(float(C).hex())]
            if set(_keys(selected)) != target_keys or len(selected) != len(target_rows):
                raise RouterSelectionError("candidate/target keys do not match")
            per_seed = {}
            for seed in SEEDS:
                seed_rows = selected.loc[selected.training_seed.eq(seed)].sort_values(
                    "row_uid", kind="stable"
                )
                labels = np.array(
                    [target_map[key] for key in _keys(seed_rows)], dtype=np.int64
                )
                loss = float(
                    multiclass_metrics(
                        seed_rows[list(Q_COLUMNS)].to_numpy(dtype=np.float64), labels
                    )["log_loss"]
                )
                if not math.isfinite(loss):
                    raise RouterSelectionError(
                        "candidate four-state log loss is nonfinite"
                    )
                per_seed[str(seed)] = loss
            mean = float(np.mean(np.array(list(per_seed.values()), dtype=np.float64)))
            candidate_losses.append(
                {
                    "C": float(C),
                    "C_hex": float(C).hex(),
                    "per_seed_log_loss": per_seed,
                    "mean_log_loss": mean,
                }
            )
        minimum = min(record["mean_log_loss"] for record in candidate_losses)
        tied = [
            record for record in candidate_losses if record["mean_log_loss"] == minimum
        ]
        winner = min(tied, key=lambda record: record["C"])
        result[variant] = {
            "selected_C": winner["C"],
            "selected_C_hex": winner["C_hex"],
            "selected_mean_log_loss": winner["mean_log_loss"],
            "candidate_losses": candidate_losses,
            "exact_tied_C_hex": [record["C_hex"] for record in tied],
            "tie_break": "smallest_C_exact_float64_mean_equality",
            "seed_aggregation": "arithmetic_mean_across_four_training_seeds",
            "best_seed_selection": False,
        }
    return result


def _threshold_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _validate_seed_rows(
        frame,
        (
            "row_uid",
            "plot_idx",
            "training_seed",
            "target_state",
            "raw_pred",
            "geo_pred",
            "score",
        ),
    )
    for name in ("raw_pred", "geo_pred"):
        if rows[name].dtype.kind not in "iu" or not rows[name].between(0, 17).all():
            raise RouterSelectionError(
                "expert predictions must be dense integer classes"
            )
    scores = rows.score.to_numpy()
    if (
        scores.dtype != np.dtype("float64")
        or not np.isfinite(scores).all()
        or not ((scores >= -1.0) & (scores <= 1.0)).all()
    ):
        raise RouterSelectionError(
            "router scores must be finite float64 values in [-1,1]"
        )
    disagreement = rows.raw_pred.ne(rows.geo_pred)
    if (rows.target_state.isin(("rescue", "harm")) & ~disagreement).any() or (
        rows.target_state.eq("both_correct") & disagreement
    ).any():
        raise RouterSelectionError("targets contradict expert agreement")
    return rows


def _action_counts(rows: pd.DataFrame, actions: np.ndarray) -> dict[str, Any]:
    acted = rows.loc[actions]
    rescue = int(acted.target_state.eq("rescue").sum())
    harm = int(acted.target_state.eq("harm").sum())
    return {
        "row_count": len(rows),
        "acted_count": int(actions.sum()),
        "rescued_count": rescue,
        "harmed_count": harm,
        "rescued_minus_harmed": rescue - harm,
        "delta_accuracy": (rescue - harm) / len(rows),
        "coverage_fraction": float(actions.mean()),
        "unique_acted_plots": int(acted.plot_idx.nunique()),
    }


def select_common_threshold(rows: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the full frozen threshold curve with O(n log n) suffix counts."""
    rows = _threshold_rows(rows)
    scores = rows.score.to_numpy(dtype=np.float64)
    specifications = enumerate_threshold_candidates(scores)
    eligible = np.flatnonzero(
        (scores > 0.0) & rows.raw_pred.ne(rows.geo_pred).to_numpy()
    )
    order = eligible[np.argsort(scores[eligible], kind="stable")]
    ordered_scores = scores[order]
    states = rows.target_state.to_numpy()
    contributions = np.column_stack(
        (
            np.ones(len(order), dtype=np.int64),
            (states[order] == "rescue").astype(np.int64),
            (states[order] == "harm").astype(np.int64),
        )
    )
    suffix = np.zeros((len(order) + 1, 3), dtype=np.int64)
    suffix[:-1] = np.cumsum(contributions[::-1], axis=0)[::-1]
    unique_plots = np.zeros(len(order) + 1, dtype=np.int64)
    seen = set()
    plots = rows.plot_idx.to_numpy()
    for position in range(len(order) - 1, -1, -1):
        seen.add(plots[order[position]])
        unique_plots[position] = len(seen)
    curve, best_rank = [], None
    winner = deepcopy(NEVER_THRESHOLD)
    for index, specification in enumerate(specifications):
        threshold = validate_threshold_specification(specification)
        position = (
            len(order)
            if threshold is None
            else int(np.searchsorted(ordered_scores, threshold, side="right"))
        )
        actions, rescues, harms = map(int, suffix[position])
        net = rescues - harms
        feasible = (
            actions * 100 >= len(rows)
            and int(unique_plots[position]) >= 20
            and 2 * harms <= rescues
            and net > 0
        )
        record = {
            "candidate_index": index,
            "threshold_specification": deepcopy(specification),
            "threshold_display": threshold,
            "row_count": len(rows),
            "acted_count": actions,
            "rescued_count": rescues,
            "harmed_count": harms,
            "rescued_minus_harmed": net,
            "mean_seed_net_utility": net / len(SEEDS),
            "coverage_fraction": actions / len(rows),
            "unique_acted_plots": int(unique_plots[position]),
            "feasible": bool(feasible),
        }
        curve.append(record)
        if feasible:
            rank = (net, -harms, -actions, threshold)
            if best_rank is None or rank > best_rank:
                best_rank, winner = rank, deepcopy(specification)
    action_mask = apply_heuristic_threshold(
        scores, rows.raw_pred.to_numpy(), rows.geo_pred.to_numpy(), winner
    )
    per_seed = {}
    for seed in SEEDS:
        mask = rows.training_seed.eq(seed).to_numpy()
        per_seed[str(seed)] = _action_counts(rows.loc[mask], action_mask[mask])
    pooled = _action_counts(rows, action_mask)
    return {
        "threshold_specification": winner,
        "threshold_specification_sha256": canonical_sha256(winner),
        "threshold_display": validate_threshold_specification(winner),
        "selected_feasible": best_rank is not None,
        "selection_reason": (
            "maximum_feasible_utility"
            if best_rank is not None
            else "no_feasible_candidate_never_intervene"
        ),
        "candidate_count": len(specifications),
        "candidate_specifications_sha256": canonical_sha256(specifications),
        "feasible_candidate_count": sum(record["feasible"] for record in curve),
        "candidate_curve": curve,
        "per_seed": per_seed,
        "pooled": pooled,
        "mean_seed_net_utility": sum(
            record["rescued_minus_harmed"] for record in per_seed.values()
        )
        / len(SEEDS),
        "mean_seed_delta_accuracy": sum(
            record["delta_accuracy"] for record in per_seed.values()
        )
        / len(SEEDS),
    }


def _finite(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise RouterSelectionError(f"{name} must be a finite number")
    return float(value)


def _policy_evidence(
    metrics: Mapping[str, Any]
) -> tuple[float, float, list[int], dict[str, Any]]:
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(metrics.get("per_seed"), Mapping)
        or set(metrics["per_seed"]) != {str(seed) for seed in SEEDS}
        or not isinstance(metrics.get("pooled"), Mapping)
    ):
        raise RouterSelectionError("gate requires complete four-seed policy metrics")
    seed_net, seed_delta = [], []
    counts = (
        "row_count",
        "acted_count",
        "rescued_count",
        "harmed_count",
        "rescued_minus_harmed",
        "unique_acted_plots",
    )
    all_records = [metrics["per_seed"][str(seed)] for seed in SEEDS] + [
        metrics["pooled"]
    ]
    for record in all_records:
        if not isinstance(record, Mapping):
            raise RouterSelectionError("invalid gate count record")
        for name in counts:
            value = record.get(name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise RouterSelectionError(f"gate count {name} must be an integer")
        if (
            record["row_count"] < 1
            or not 0 <= record["acted_count"] <= record["row_count"]
            or min(
                record["rescued_count"],
                record["harmed_count"],
                record["unique_acted_plots"],
            )
            < 0
            or record["rescued_count"] + record["harmed_count"] > record["acted_count"]
            or record["unique_acted_plots"] > record["acted_count"]
            or record["rescued_minus_harmed"]
            != record["rescued_count"] - record["harmed_count"]
        ):
            raise RouterSelectionError("inconsistent gate counts")
        for name, expected in (
            ("delta_accuracy", record["rescued_minus_harmed"] / record["row_count"]),
            ("coverage_fraction", record["acted_count"] / record["row_count"]),
        ):
            if not math.isclose(
                _finite(record.get(name), name), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RouterSelectionError("gate rates do not match counts")
    for name in counts[:-1]:
        if metrics["pooled"][name] != sum(record[name] for record in all_records[:-1]):
            raise RouterSelectionError("pooled gate counts differ from seed totals")
    for record in all_records[:-1]:
        seed_net.append(int(record["rescued_minus_harmed"]))
        seed_delta.append(float(record["delta_accuracy"]))
    mean_net = _finite(metrics.get("mean_seed_net_utility"), "mean_seed_net_utility")
    mean_delta = _finite(
        metrics.get("mean_seed_delta_accuracy"), "mean_seed_delta_accuracy"
    )
    if not math.isclose(
        mean_net, sum(seed_net) / 4, rel_tol=0.0, abs_tol=1e-12
    ) or not math.isclose(mean_delta, sum(seed_delta) / 4, rel_tol=0.0, abs_tol=1e-12):
        raise RouterSelectionError("gate means differ from seed metrics")
    return mean_net, mean_delta, seed_net, dict(metrics["pooled"])


def evaluate_gate(
    policy_metrics: Mapping[str, Any],
    heuristic_metrics: Mapping[str, Mapping[str, Any]],
    per_plot_net: Mapping[str, int] | pd.DataFrame,
    access_audit_clean: bool,
) -> dict[str, Any]:
    """Apply all eight frozen pass conditions and exact no-go precedence."""
    if type(access_audit_clean) is not bool:
        raise RouterSelectionError("access_audit_clean must be a Boolean")
    if not isinstance(heuristic_metrics, Mapping) or set(heuristic_metrics) != set(
        HEURISTICS
    ):
        raise RouterSelectionError(
            "gate requires exactly the five predeclared heuristics"
        )
    mean_net, mean_delta, seed_net, pooled = _policy_evidence(policy_metrics)
    baseline_nets = {
        name: _policy_evidence(heuristic_metrics[name])[0] for name in HEURISTICS
    }
    if isinstance(per_plot_net, pd.DataFrame):
        if (
            not {"plot_idx", "rescued_minus_harmed"}.issubset(per_plot_net)
            or per_plot_net.plot_idx.duplicated().any()
        ):
            raise RouterSelectionError(
                "per-plot net table requires unique plot identities"
            )
        plot_nets = dict(
            per_plot_net[["plot_idx", "rescued_minus_harmed"]].itertuples(
                index=False, name=None
            )
        )
    elif isinstance(per_plot_net, Mapping):
        plot_nets = dict(per_plot_net)
    else:
        raise RouterSelectionError("invalid per-plot net evidence")
    if not plot_nets or any(
        not isinstance(plot, str) or not plot for plot in plot_nets
    ):
        raise RouterSelectionError("per-plot net requires nonempty plot identities")
    for value in plot_nets.values():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise RouterSelectionError(
                "per-plot utility must be an integer pooled count"
            )
    if sum(plot_nets.values()) != pooled["rescued_minus_harmed"]:
        raise RouterSelectionError("per-plot utility does not sum to pooled utility")
    positive = [(plot, int(net)) for plot, net in plot_nets.items() if net > 0]
    best_plot, best_net = (
        min(positive, key=lambda item: (-item[1], item[0])) if positive else (None, 0)
    )
    remaining = int(pooled["rescued_minus_harmed"] - best_net)
    criteria = dict(
        zip(
            GATE_CRITERIA,
            (
                mean_delta > 0.0,
                sum(net > 0 for net in seed_net) >= 3,
                2 * pooled["harmed_count"] <= pooled["rescued_count"],
                pooled["acted_count"] * 100 >= pooled["row_count"],
                pooled["unique_acted_plots"] >= 20,
                all(mean_net > baseline for baseline in baseline_nets.values()),
                remaining > 0,
                access_audit_clean,
            ),
        )
    )
    no_go_reasons = []
    if mean_net <= 0.0:
        no_go_reasons.append("mean_net_utility_nonpositive")
    if mean_net <= baseline_nets[PRINCIPAL_HEURISTIC]:
        no_go_reasons.append("does_not_exceed_principal_heuristic")
    decision = (
        "go"
        if all(criteria.values())
        else ("no_go" if no_go_reasons else "inconclusive")
    )
    rationales = {
        GATE_CRITERIA[0]: f"mean delta accuracy = {mean_delta:.17g}",
        GATE_CRITERIA[
            1
        ]: f"positive-utility seeds = {sum(net > 0 for net in seed_net)}/4",
        GATE_CRITERIA[
            2
        ]: f"pooled harmed = {pooled['harmed_count']}; rescued = {pooled['rescued_count']}",
        GATE_CRITERIA[
            3
        ]: f"pooled actions = {pooled['acted_count']}/{pooled['row_count']}",
        GATE_CRITERIA[4]: f"unique acted plots = {pooled['unique_acted_plots']}",
        GATE_CRITERIA[
            5
        ]: f"primary mean net = {mean_net:.17g}; heuristic means = {baseline_nets}",
        GATE_CRITERIA[6]: f"pooled net after removing best positive plot = {remaining}",
        GATE_CRITERIA[7]: (
            "development access audit clean"
            if access_audit_clean
            else "development access audit not clean"
        ),
    }
    return {
        "decision": decision,
        "criteria": criteria,
        "failed_criteria": [name for name, passed in criteria.items() if not passed],
        "no_go_reasons": no_go_reasons,
        "rationales": rationales,
        "primary_mean_seed_net_utility": mean_net,
        "heuristic_mean_seed_net_utility": baseline_nets,
        "principal_heuristic": PRINCIPAL_HEURISTIC,
        "best_contributing_plot": best_plot,
        "best_plot_pooled_net": best_net,
        "pooled_net_after_removing_best_plot": remaining,
    }


__all__ = [
    "RouterSelectionError",
    "select_common_c",
    "select_common_threshold",
    "evaluate_gate",
]

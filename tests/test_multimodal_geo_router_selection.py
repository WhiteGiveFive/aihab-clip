"""Synthetic tests for exact selection and all frozen M4 gate conditions."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_router_selection as selection
from multimodal.geo_helpfulness_router_metrics import policy_metrics


def c_inputs():
    targets = pd.DataFrame([
        {"row_uid": f"row-{row:03d}", "training_seed": seed, "target_state": selection.TARGET_ORDER[row % 4]}
        for seed in selection.SEEDS for row in range(8)
    ])
    records = []
    for C, confidence in zip(selection.C_VALUES, (.4, .6, .85, .8)):
        for target in targets.itertuples(index=False):
            q = np.full(4, (1 - confidence) / 3, dtype=np.float64)
            q[selection.TARGET_ORDER.index(target.target_state)] = confidence
            records.append({"row_uid": target.row_uid, "training_seed": target.training_seed,
                            "C_hex": float(C).hex(), **dict(zip(selection.Q_COLUMNS, q))})
    return {variant: pd.DataFrame(records) for variant in selection.VARIANTS}, targets


def test_select_c_uses_four_seed_log_losses_and_one_common_c_per_variant():
    frames, targets = c_inputs()
    result = selection.select_common_c(frames, targets)
    assert set(result) == set(selection.VARIANTS)
    for variant in selection.VARIANTS:
        value = result[variant]
        assert value["selected_C"] == 1.0
        assert value["selected_C_hex"] == 1.0.hex()
        assert len(value["candidate_losses"]) == 4
        for candidate, confidence in zip(value["candidate_losses"], (.4, .6, .85, .8)):
            assert set(candidate["per_seed_log_loss"]) == {"1", "2", "3", "4"}
            assert candidate["mean_log_loss"] == pytest.approx(-np.log(confidence))
        assert value["best_seed_selection"] is False
    shuffled = {name: frame.sample(frac=1, random_state=87).reset_index(drop=True) for name, frame in frames.items()}
    assert selection.select_common_c(shuffled, targets.sample(frac=1, random_state=89)) == result


def test_exact_c_ties_choose_smallest_but_near_ties_do_not(monkeypatch):
    frames, targets = c_inputs()
    monkeypatch.setattr(selection, "multiclass_metrics", lambda q, y: {"log_loss": 1.0})
    result = selection.select_common_c(frames, targets)
    assert all(value["selected_C"] == .01 for value in result.values())
    assert all(len(value["exact_tied_C_hex"]) == 4 for value in result.values())
    losses = iter(([np.nextafter(1.0, np.inf)] * 4 + [1.0] * 4 + [2.0] * 4 + [3.0] * 4) * 3)
    monkeypatch.setattr(selection, "multiclass_metrics", lambda q, y: {"log_loss": next(losses)})
    result = selection.select_common_c(frames, targets)
    assert all(value["selected_C"] == .1 for value in result.values())


@pytest.mark.parametrize("damage", ["variant", "missing_c", "duplicate", "wrong_key", "q_float32", "q_invalid", "target_seed", "target_duplicate"])
def test_bad_c_selection_inputs_rejected(damage):
    frames, targets = c_inputs()
    name = selection.VARIANTS[0]
    if damage == "variant":
        del frames[name]
    elif damage == "missing_c":
        frames[name] = frames[name].loc[frames[name].C_hex.ne(10.0.hex())]
    elif damage == "duplicate":
        frames[name] = pd.concat([frames[name], frames[name].iloc[[0]]])
    elif damage == "wrong_key":
        frames[name].loc[0, "row_uid"] = "wrong"
    elif damage == "q_float32":
        frames[name]["q_rescue"] = frames[name].q_rescue.astype(np.float32)
    elif damage == "q_invalid":
        frames[name].loc[0, "q_rescue"] = np.nan
    elif damage == "target_seed":
        targets = targets.loc[targets.training_seed.ne(4)]
    else:
        targets = pd.concat([targets, targets.iloc[[0]]])
    with pytest.raises(selection.RouterSelectionError):
        selection.select_common_c(frames, targets)


def threshold_rows(n=80):
    rng = np.random.default_rng(537)
    rows = []
    for seed in selection.SEEDS:
        for index in range(n):
            target = rng.choice(selection.TARGET_ORDER, p=[.5, .1, .1, .3])
            rows.append({
                "row_uid": f"image-{index:05d}", "plot_idx": f"plot-{index:05d}",
                "training_seed": seed, "target_state": target,
                "raw_pred": 0, "geo_pred": 0 if target == "both_correct" else 1,
                "score": float(rng.uniform(-1, 1)),
            })
    return pd.DataFrame(rows)


def brute_threshold(rows):
    scores = rows.score.to_numpy(dtype=np.float64)
    specifications = selection.enumerate_threshold_candidates(scores)
    winner, best, records = deepcopy(selection.NEVER_THRESHOLD), None, []
    for specification in specifications:
        actions = selection.apply_heuristic_threshold(scores, rows.raw_pred.to_numpy(), rows.geo_pred.to_numpy(), specification)
        acted = rows.loc[actions]
        r, h = int(acted.target_state.eq("rescue").sum()), int(acted.target_state.eq("harm").sum())
        count, plots = int(actions.sum()), int(acted.plot_idx.nunique())
        feasible = count * 100 >= len(rows) and plots >= 20 and 2 * h <= r and r - h > 0
        records.append((count, r, h, plots, feasible))
        if feasible:
            threshold = selection.validate_threshold_specification(specification)
            rank = (r - h, -h, -count, threshold)
            if best is None or rank > best:
                best, winner = rank, specification
    return winner, records


def test_suffix_threshold_curve_matches_independent_bruteforce_and_is_deterministic():
    rows = threshold_rows()
    expected, counts = brute_threshold(rows)
    result = selection.select_common_threshold(rows)
    assert result["threshold_specification"] == expected
    assert result["candidate_count"] == len(counts)
    for record, reference in zip(result["candidate_curve"], counts):
        assert (record["acted_count"], record["rescued_count"], record["harmed_count"], record["unique_acted_plots"], record["feasible"]) == reference
    assert result["mean_seed_net_utility"] == result["pooled"]["rescued_minus_harmed"] / 4
    assert selection.select_common_threshold(rows.sample(frac=1, random_state=58)) == result
    json.dumps(result, allow_nan=False)


def fixed_threshold_rows(n=40):
    rows = threshold_rows(n)
    rows["target_state"] = "both_wrong"
    rows["raw_pred"], rows["geo_pred"], rows["score"] = 0, 1, 0.0
    return rows


def test_coverage_and_plot_support_use_pooled_rows_and_unique_plot_union():
    rows = fixed_threshold_rows(2000)
    selected = rows.row_uid.isin([f"image-{index:05d}" for index in range(20)])
    rows.loc[selected, ["target_state", "score"]] = ["rescue", .5]
    value = selection.select_common_threshold(rows)
    assert value["selected_feasible"]
    assert value["pooled"]["acted_count"] == 80
    assert value["pooled"]["coverage_fraction"] == .01
    assert value["pooled"]["unique_acted_plots"] == 20
    lower = rows.copy()
    lower.loc[0, "score"] = 0.0
    assert not selection.select_common_threshold(lower)["selected_feasible"]
    lower = rows.copy()
    lower.loc[lower.plot_idx.eq("plot-00019"), "plot_idx"] = "plot-00018"
    assert not selection.select_common_threshold(lower)["selected_feasible"]


def test_harm_ratio_boundary_and_infeasible_never_fallback():
    rows = fixed_threshold_rows()
    for index in range(30):
        mask = rows.row_uid.eq(f"image-{index:05d}")
        rows.loc[mask, ["target_state", "score"]] = ["rescue" if index < 20 else "harm", .5]
    result = selection.select_common_threshold(rows)
    assert result["selected_feasible"]
    assert result["pooled"]["harmed_count"] * 2 == result["pooled"]["rescued_count"]
    rows.loc[rows.row_uid.eq("image-00030"), ["target_state", "score"]] = ["harm", .5]
    result = selection.select_common_threshold(rows)
    assert not result["selected_feasible"]
    assert result["threshold_specification"] == selection.NEVER_THRESHOLD
    assert result["threshold_display"] is None
    assert result["pooled"]["acted_count"] == 0


def test_threshold_ties_favor_fewer_actions_then_higher_threshold_and_strict_gt():
    rows = fixed_threshold_rows()
    first = rows.row_uid.isin([f"image-{index:05d}" for index in range(20)])
    rows.loc[first, ["target_state", "score"]] = ["rescue", .9]
    middle = rows.row_uid.isin([f"image-{index:05d}" for index in range(20, 30)])
    rows.loc[middle, "score"] = .3
    high_agreement = rows.row_uid.isin([f"image-{index:05d}" for index in range(30, 40)])
    rows.loc[high_agreement, ["target_state", "geo_pred", "score"]] = ["both_correct", 0, .6]
    result = selection.select_common_threshold(rows)
    assert result["threshold_display"] == .75
    assert result["pooled"]["acted_count"] == 80
    spec = {"kind": "finite", "value_hex": .9.hex(), "comparison": "strict_gt"}
    assert not selection.apply_heuristic_threshold(rows.score.to_numpy(), rows.raw_pred.to_numpy(), rows.geo_pred.to_numpy(), spec).any()


def test_equal_utility_threshold_tie_prefers_fewer_harms():
    rows = fixed_threshold_rows()
    high = rows.row_uid.isin([f"image-{index:05d}" for index in range(20)])
    rows.loc[high, ["target_state", "score"]] = ["rescue", .9]
    rows.loc[rows.row_uid.eq("image-00020"), ["target_state", "score"]] = ["rescue", .3]
    rows.loc[rows.row_uid.eq("image-00021"), ["target_state", "score"]] = ["harm", .3]
    result = selection.select_common_threshold(rows)
    zero = result["candidate_curve"][1]
    winner = result["pooled"]
    assert zero["rescued_minus_harmed"] == winner["rescued_minus_harmed"] == 80
    assert zero["harmed_count"] == 4 and winner["harmed_count"] == 0
    assert result["threshold_display"] == pytest.approx(.6)


def test_nonpositive_score_curve_uses_typed_never_fallback():
    rows = fixed_threshold_rows()
    rows["score"] = -1.0
    result = selection.select_common_threshold(rows)
    assert result["candidate_count"] == 2
    assert result["feasible_candidate_count"] == 0
    assert result["threshold_specification"] == selection.NEVER_THRESHOLD
    assert all(record["acted_count"] == 0 for record in result["candidate_curve"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("damage", ["seed", "duplicate", "plot", "target", "prediction", "agreement", "score"])
def test_invalid_threshold_rows_rejected(damage):
    rows = threshold_rows()
    if damage == "seed":
        rows = rows.loc[rows.training_seed.ne(4)]
    elif damage == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    elif damage == "plot":
        rows.loc[0, "plot_idx"] = "changed"
    elif damage == "target":
        rows.loc[0, "target_state"] = "other"
    elif damage == "prediction":
        rows.loc[0, "geo_pred"] = 18
    elif damage == "agreement":
        rows.loc[0, ["target_state", "raw_pred", "geo_pred"]] = ["rescue", 0, 0]
    else:
        rows.loc[0, "score"] = np.inf
    with pytest.raises(selection.RouterSelectionError):
        selection.select_common_threshold(rows)


def metrics_fixture(seed_net=(10, 10, 10, 10), *, harms=0, unique_plots=25, row_count=100, extra_actions=0):
    per_seed = {}
    for seed, net in zip(selection.SEEDS, seed_net):
        harmed = harms + max(-net, 0)
        rescued = net + harmed
        acted = rescued + harmed + extra_actions
        per_seed[str(seed)] = {
            "row_count": row_count, "acted_count": acted, "rescued_count": rescued,
            "harmed_count": harmed, "rescued_minus_harmed": net,
            "delta_accuracy": net / row_count, "coverage_fraction": acted / row_count,
            "unique_acted_plots": min(acted, unique_plots),
        }
    pooled = {key: sum(record[key] for record in per_seed.values()) for key in ("row_count", "acted_count", "rescued_count", "harmed_count", "rescued_minus_harmed")}
    pooled.update(delta_accuracy=pooled["rescued_minus_harmed"] / pooled["row_count"], coverage_fraction=pooled["acted_count"] / pooled["row_count"], unique_acted_plots=min(unique_plots, pooled["acted_count"]))
    return {"per_seed": per_seed, "pooled": pooled, "mean_seed_net_utility": sum(seed_net) / 4, "mean_seed_delta_accuracy": sum(seed_net) / (4 * row_count)}


def gate_inputs():
    return metrics_fixture(), {name: metrics_fixture((0, 0, 0, 0)) for name in selection.HEURISTICS}, {f"plot-{index}": 2 for index in range(20)}


def test_gate_passes_exact_eight_criteria_and_accepts_perplot_dataframe():
    primary, baselines, plots = gate_inputs()
    result = selection.evaluate_gate(primary, baselines, plots, True)
    assert result["decision"] == "go"
    assert tuple(result["criteria"]) == selection.GATE_CRITERIA
    assert all(result["criteria"].values())
    assert result["failed_criteria"] == []
    assert result["pooled_net_after_removing_best_plot"] == 38
    table = pd.DataFrame({"plot_idx": list(plots), "rescued_minus_harmed": list(plots.values())})
    assert selection.evaluate_gate(primary, baselines, table, True) == result


@pytest.mark.parametrize("failure", ["net", "principal", "seed_instability", "harm", "coverage", "plots", "secondary", "concentration", "audit"])
def test_gate_no_go_precedence_and_other_failures_are_inconclusive(failure):
    primary, baselines, plots = gate_inputs()
    clean = True
    if failure == "net":
        primary = metrics_fixture((0, 0, 0, 0))
        plots = {"plot": 0}
    elif failure == "principal":
        baselines[selection.PRINCIPAL_HEURISTIC] = deepcopy(primary)
    elif failure == "seed_instability":
        primary = metrics_fixture((20, 20, 0, 0))
    elif failure == "harm":
        primary = metrics_fixture(harms=11)
    elif failure == "coverage":
        primary = metrics_fixture(row_count=2000)
    elif failure == "plots":
        primary = metrics_fixture(unique_plots=19)
    elif failure == "secondary":
        baselines["image_geo_jsd"] = deepcopy(primary)
    elif failure == "concentration":
        plots = {"single-positive-plot": 40}
    else:
        clean = False
    result = selection.evaluate_gate(primary, baselines, plots, clean)
    assert result["decision"] == ("no_go" if failure in {"net", "principal"} else "inconclusive")
    assert result["failed_criteria"]


def test_gate_rejects_missing_heuristic_or_inconsistent_evidence():
    primary, baselines, plots = gate_inputs()
    with pytest.raises(selection.RouterSelectionError):
        selection.evaluate_gate(primary, {}, plots, True)
    with pytest.raises(selection.RouterSelectionError):
        selection.evaluate_gate(primary, baselines, {"wrong": 999}, True)
    with pytest.raises(selection.RouterSelectionError):
        selection.evaluate_gate(primary, baselines, plots, 1)
    broken = deepcopy(primary)
    broken["mean_seed_net_utility"] += 1
    with pytest.raises(selection.RouterSelectionError):
        selection.evaluate_gate(broken, baselines, plots, True)


def test_gate_accepts_actual_policy_metrics_schema():
    rows = fixed_threshold_rows()
    rows["label_id_dense"] = 1
    rows["image_pred"] = 0
    rows["target_state"] = "rescue"
    actions = np.ones(len(rows), dtype=bool)
    primary = policy_metrics(rows, actions)
    baselines = {name: policy_metrics(rows, np.zeros(len(rows), dtype=bool)) for name in selection.HEURISTICS}
    plots = {plot: int(len(group)) for plot, group in rows.groupby("plot_idx")}
    assert selection.evaluate_gate(primary, baselines, plots, True)["decision"] == "go"


def test_selection_module_imports_no_fitting_stack():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; import multimodal.geo_helpfulness_router_selection; "
        "blocked=('scipy','sklearn','torch','multimodal.geo_helpfulness_router_candidates',"
        "'multimodal.geo_helpfulness_router_candidate_fit','multimodal.geo_helpfulness_router_dataset',"
        "'multimodal.geo_helpfulness_oof'); "
        "assert not any(name==p or name.startswith(p+'.') for p in blocked for name in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

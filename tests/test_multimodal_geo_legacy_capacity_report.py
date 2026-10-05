"""Synthetic-only frozen transfer scoring, baseline parity, and boundary tests."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_legacy_capacity_report as report
from multimodal import geo_helpfulness_legacy_router_report as legacy
from test_multimodal_geo_legacy_router_report import _inputs as legacy_inputs


def inputs(family_shift=0):
    experts, old_routers, old_actions, labels = legacy_inputs(family_shift)
    for kind, frame in (
        ("experts", experts),
        ("routers", old_routers),
        ("actions", old_actions),
    ):
        frame.insert(0, "study_id", "legacy_frozen_router_transfer_v1")
        frame.insert(0, "schema_version", f"geo_helpfulness.legacy_transfer_{kind}.v1")
    declarations, routers, actions = {}, [], [old_actions]
    for name in report.OLD_POLICIES:
        frame = old_actions.loc[old_actions.policy_id.eq(name)]
        learner, variant = (
            ("lr", name) if name in report.VARIANTS else ("heuristic", "")
        )
        declarations[name] = {
            "learner": learner,
            "variant_id": variant,
            "config_id": "legacy__" + name if learner == "lr" else name,
            "threshold_specification": json.loads(
                frame.threshold_specification.iloc[0]
            ),
        }
        if learner == "lr":
            router = old_routers.loc[old_routers.variant_id.eq(name)].copy()
            declarations[name]["C_hex"] = router.C_hex.iloc[0]
            router["policy_id"], router["learner"], router["config_id"] = (
                name,
                learner,
                declarations[name]["config_id"],
            )
            router["study_id"] = report.STUDY_ID
            router["schema_version"] = "geo_helpfulness.legacy_capacity_routers.v1"
            routers.append(router)
    codes = (
        experts.row_uid.str[-2:].astype(int).to_numpy()
        + experts.training_seed.to_numpy()
        - 1
    ) % 4
    quality = np.asarray(
        [
            [0.72, 0.1, 0.08, 0.1],
            [0.1, 0.65, 0.1, 0.15],
            [0.1, 0.1, 0.72, 0.08],
            [0.2, 0.1, 0.1, 0.6],
        ],
        dtype=np.float64,
    )
    for name in report.NEW_POLICIES:
        learner, variant = report._policy_identity(name)
        config_id = "selected__" + name
        router = experts.loc[:, [*report.IDENTITY, "checkpoint_family"]].copy()
        router["schema_version"], router["study_id"] = (
            "geo_helpfulness.legacy_capacity_routers.v1",
            report.STUDY_ID,
        )
        router["policy_id"], router["learner"], router["variant_id"] = (
            name,
            learner,
            variant,
        )
        router["config_id"], router["C_hex"] = config_id, ""
        q = quality[codes].copy()
        if learner == "xgb":
            q[(codes == 0) & experts.row_uid.ge("image-04").to_numpy()] = [
                0.35,
                0.25,
                0.2,
                0.2,
            ]
        for index, column in enumerate(report.Q_COLUMNS):
            router[column] = q[:, index]
        router["score"] = q[:, 0] - q[:, 1]
        routers.append(router)
        specification = (
            deepcopy(report.NEVER_THRESHOLD)
            if learner == "mlp" and variant != "ablation_numeric25"
            else {"kind": "finite", "comparison": "strict_gt", "value_hex": (0.2).hex()}
        )
        declarations[name] = {
            "learner": learner,
            "variant_id": variant,
            "config_id": config_id,
            "threshold_specification": specification,
            "models": {"1": {"path": "must-not-be-opened.json"}},
        }
        action = old_actions.loc[old_actions.policy_id.eq(report.BASELINE)].copy()
        action["schema_version"], action["study_id"] = (
            "geo_helpfulness.legacy_capacity_actions.v1",
            report.STUDY_ID,
        )
        action["policy_id"], action["score"] = name, router.score.to_numpy()
        action["threshold_specification"] = json.dumps(specification, sort_keys=True)
        acted = legacy.apply_heuristic_threshold(
            action.score.to_numpy(),
            experts.raw_pred.to_numpy(),
            experts.geo_pred.to_numpy(),
            specification,
        )
        action["acted"], action["final_pred"] = acted, np.where(
            acted, experts.geo_pred, experts.raw_pred
        )
        actions.append(action)
    return (
        experts,
        pd.concat(routers, ignore_index=True),
        pd.concat(actions, ignore_index=True),
        labels,
        declarations,
    ), (experts, old_routers, old_actions, labels)


@pytest.fixture(scope="module")
def scored():
    args, old = inputs()
    return args, report.evaluate_family(*args), legacy.evaluate_family(*old)


def test_contract_reports_twelve_selected_routers_seventeen_policies(scored):
    _, result, _ = scored
    assert set(result) == set(report.SCORE_FILES)
    assert len(result["router_metrics.json"]) == 12
    assert len(result["policy_metrics.json"]) == 17
    assert len(result["summary.json"]["groups"]) == 12
    summary = result["summary.json"]
    assert (summary["row_count"], summary["image_count"], summary["plot_count"]) == (
        32,
        8,
        4,
    )
    assert (
        summary["original_policy_count"],
        summary["new_policy_count"],
        summary["policy_count"],
    ) == (8, 9, 17)
    assert summary["checkpoint_family"] == "best_model"
    assert summary["M4_gate_rerun"] is False
    assert summary["selection_or_fitting_performed"] is False
    assert summary["checkpoint_family_selection_performed"] is False
    assert summary["independent_confirmation"] is False
    for filename, value in result.items():
        if filename.endswith(".json"):
            json.dumps(value, allow_nan=False)
        else:
            assert isinstance(value, pa.Table)
            assert value.schema.metadata is None


def test_original_eight_policy_payloads_actions_and_targets_are_exact(scored):
    _, result, old = scored
    assert result["expert_metrics.json"] == old["expert_metrics.json"]
    assert (
        result["classification.json"]["experts"]
        == old["classification.json"]["experts"]
    )
    for name in report.OLD_POLICIES:
        assert result["policy_metrics.json"][name] == old["policy_metrics.json"][name]
        assert (
            result["classification.json"]["policies"][name]
            == old["classification.json"]["policies"][name]
        )
    for name in report.VARIANTS:
        for key, value in old["router_metrics.json"][name].items():
            assert result["router_metrics.json"][name][key] == value
    for filename, policy_column in (
        ("actions_with_outcomes.parquet", "policy_id"),
        ("subgroup_metrics.parquet", "policy"),
        ("bootstrap_replicates.parquet", "policy"),
    ):
        frame = result[filename].to_pandas()
        original = frame.loc[
            frame[policy_column].isin(report.OLD_POLICIES)
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(original, old[filename].to_pandas())
    assert result["scoring_targets.parquet"].equals(old["scoring_targets.parquet"])
    actions = result["actions_with_outcomes.parquet"].to_pandas()
    assert (
        actions.loc[actions.policy_id.isin(report.OLD_POLICIES), "study_id"]
        .eq("legacy_frozen_router_transfer_v1")
        .all()
    )
    assert (
        actions.loc[actions.policy_id.isin(report.NEW_POLICIES), "study_id"]
        .eq(report.STUDY_ID)
        .all()
    )


def test_all_router_metrics_match_direct_math_in_both_scopes(scored):
    (_, routers, _, _, _), result, _ = scored
    targets = result["scoring_targets.parquet"].to_pandas()
    y = targets.target_state.map(
        {s: i for i, s in enumerate(legacy.TARGET_ORDER)}
    ).to_numpy()
    for name, metric in result["router_metrics.json"].items():
        q = routers.loc[routers.policy_id.eq(name), list(report.Q_COLUMNS)].to_numpy()
        for scope, mask in (
            ("all_rows", np.ones(len(y), dtype=bool)),
            ("disagreement", targets.raw_pred.ne(targets.geo_pred).to_numpy()),
        ):
            actual = metric[scope]["pooled_seed_realizations"]
            assert actual["top1_accuracy"] == (q[mask].argmax(1) == y[mask]).mean()
            assert actual["log_loss"] == pytest.approx(-np.log(q[mask, y[mask]]).mean())
            assert actual["brier_score"] == pytest.approx(
                ((q[mask] - np.eye(4)[y[mask]]) ** 2).sum(1).mean()
            )
            assert actual["rescue_pr_auc"] == legacy.binary_average_precision(
                y[mask] == 0, q[mask, 0]
            )
            assert actual["harm_pr_auc"] == legacy.binary_average_precision(
                y[mask] == 1, q[mask, 1]
            )
            assert set(metric[scope]["per_seed"]) == {"1", "2", "3", "4"}


def test_new_policy_classification_uses_selected_expert_top3_and_counts(scored):
    (experts, _, _, _, _), result, _ = scored
    targets = result["scoring_targets.parquet"].to_pandas()
    actions = result["actions_with_outcomes.parquet"].to_pandas()
    y = targets.label_id_dense.to_numpy()
    for name in report.NEW_POLICIES:
        frame = actions.loc[actions.policy_id.eq(name)]
        mask = frame.acted.to_numpy()
        p = np.where(
            mask[:, None],
            np.stack(experts.geo_prob_calibrated),
            np.stack(experts.raw_prob_calibrated),
        )
        top3 = (np.argsort(-p, axis=1, kind="stable")[:, :3] == y[:, None]).any(1)
        metric = result["classification.json"]["policies"][name][
            "pooled_seed_realizations"
        ]
        assert metric["top3_accuracy"] == top3.mean()
        assert metric["top1_accuracy"] == (frame.final_pred.to_numpy() == y).mean()
        assert len(metric["per_class"]) == 18
        assert (
            result["policy_metrics.json"][name]["pooled"]
            == legacy.policy_metrics(targets, mask)["pooled"]
        )
        np.testing.assert_array_equal(
            frame.net_correct,
            frame.final_correct.astype(int) - frame.raw_correct.astype(int),
        )
    for name in (
        report.BASELINE,
        "mlp__primary_full727",
        "mlp__ablation_no_pairs79",
        "never_intervene",
    ):
        assert (
            result["classification.json"]["policies"][name]
            == result["classification.json"]["experts"]["raw"]
        )
        assert result["policy_metrics.json"][name]["pooled"]["acted_count"] == 0


def test_all_twentyseven_contrasts_use_same_plot_draws_and_correct_sign(scored):
    _, result, old = scored
    bootstrap = result["bootstrap_summary.json"]
    assert (bootstrap["replicates"], bootstrap["random_seed"]) == (2000, 20260826)
    assert (
        bootstrap["resample_multiplicity_sha256"]
        == old["bootstrap_summary.json"]["resample_multiplicity_sha256"]
    )
    assert bootstrap["selection_repeated_in_bootstrap"] is False
    r = result["bootstrap_replicates.parquet"].to_pandas()
    assert len(r) == 34000
    wide = r.pivot(
        index="replicate", columns="policy", values="mean_seed_delta_accuracy"
    )
    paired = result["paired_comparisons.json"]
    assert (
        paired["contrast_count"] == sum(map(len, paired["comparisons"].values())) == 27
    )
    assert paired["difference_sign"] == "named_policy_minus_frozen_comparator"
    nonconstant = False
    for name, comparators in paired["comparisons"].items():
        learner, variant = report._policy_identity(name)
        expected = {variant, "geo_minus_raw_margin"}
        if learner in ("rf", "mlp"):
            expected.add("capacity__xgb__" + variant)
        if learner == "mlp":
            expected.add("capacity__rf__" + variant)
        assert set(comparators) == expected
        for comparator, interval in comparators.items():
            draws = wide[name] - wide[comparator]
            nonconstant |= draws.std() > 1e-4
            np.testing.assert_allclose(
                [interval["lower"], interval["upper"]],
                np.quantile(draws, [0.025, 0.975]),
                rtol=0,
                atol=1e-15,
            )
            assert (
                interval["point_estimate"]
                == result["policy_metrics.json"][name]["mean_seed_delta_accuracy"]
                - result["policy_metrics.json"][comparator]["mean_seed_delta_accuracy"]
            )
    assert nonconstant


def test_checkpoint_family_and_seed_targets_are_independently_reconstructed(scored):
    _, first, _ = scored
    args, _ = inputs(family_shift=1)
    second = report.evaluate_family(*args)
    left, right = (
        first["scoring_targets.parquet"].to_pandas(),
        second["scoring_targets.parquet"].to_pandas(),
    )
    assert left[list(report.KEY)].equals(right[list(report.KEY)])
    assert left.groupby("row_uid").target_state.nunique().eq(4).all()
    assert (left.target_state != right.target_state).all()
    assert second["summary.json"]["oracle_checkpoint_test_selection_biased"] is True
    assert (
        "test-selection-biased"
        in second["summary.json"]["checkpoint_selection_disclosure"]
    )
    assert "not guaranteed" in second["summary.json"]["calibration_disclosure"]
    assert (
        first["bootstrap_summary.json"]["resample_multiplicity_sha256"]
        == second["bootstrap_summary.json"]["resample_multiplicity_sha256"]
    )
    name = "capacity__rf__primary_full727"
    assert (
        first["router_metrics.json"][name]["all_rows"]["pooled_seed_realizations"][
            "top1_accuracy"
        ]
        != second["router_metrics.json"][name]["all_rows"]["pooled_seed_realizations"][
            "top1_accuracy"
        ]
    )


@pytest.mark.parametrize(
    "bad",
    [
        "missing_policy",
        "extra_policy",
        "wrong_learner",
        "wrong_variant",
        "wrong_config",
        "wrong_C",
        "new_C",
        "original_no_go",
        "nonfinite_metadata",
        "router_label",
        "expert_label",
        "action_outcome",
        "router_true_probability",
        "missing_expert_probability",
        "missing_router",
        "missing_action",
        "duplicate_router",
        "duplicate_action",
        "wrong_family",
        "family_missing",
        "plot",
        "q_float32",
        "q_sum",
        "q_nan",
        "q_score",
        "action_score",
        "threshold_declaration",
        "threshold_duplicate_json",
        "acted_float",
        "acted_mismatch",
        "expert_prediction",
        "final_prediction",
        "heuristic_score",
        "label_ontology",
        "missing_label",
        "label_plot",
        "missing_seed",
        "noncanonical_rows",
    ],
)
def test_invalid_or_leaky_inputs_fail_closed(scored, bad):
    e, r, a, labels, specs = deepcopy(scored[0])
    name = "capacity__rf__primary_full727"
    ri = r.index[r.policy_id.eq(name)][0]
    ai = a.index[a.policy_id.eq(name)][0]
    if bad == "missing_policy":
        specs.pop(name)
    elif bad == "extra_policy":
        specs["extra"] = deepcopy(specs[name])
    elif bad == "wrong_learner":
        specs[name]["learner"] = "xgb"
    elif bad == "wrong_variant":
        specs[name]["variant_id"] = "ablation_numeric25"
    elif bad == "wrong_config":
        r.loc[ri, "config_id"] = "other"
    elif bad == "wrong_C":
        r.loc[r.policy_id.eq(report.BASELINE), "C_hex"] = (1.0).hex()
    elif bad == "new_C":
        r.loc[ri, "C_hex"] = (1.0).hex()
    elif bad == "original_no_go":
        specs[report.BASELINE]["threshold_specification"] = deepcopy(
            specs[name]["threshold_specification"]
        )
    elif bad == "nonfinite_metadata":
        specs[name]["bad"] = float("nan")
    elif bad == "router_label":
        r["label_id_dense"] = 0
    elif bad == "expert_label":
        e["target_state"] = "rescue"
    elif bad == "action_outcome":
        a["rescued"] = False
    elif bad == "router_true_probability":
        r["raw_true_probability"] = 0.5
    elif bad == "missing_expert_probability":
        e = e.drop(columns="raw_prob_calibrated")
    elif bad == "missing_router":
        r = r.loc[r.policy_id.ne(name)]
    elif bad == "missing_action":
        a = a.loc[a.policy_id.ne(name)]
    elif bad == "duplicate_router":
        r = pd.concat([r, r.loc[[ri]]], ignore_index=True)
    elif bad == "duplicate_action":
        a = pd.concat([a, a.loc[[ai]]], ignore_index=True)
    elif bad == "wrong_family":
        r.loc[ri, "checkpoint_family"] = "oracle_test_best_model"
    elif bad == "family_missing":
        e = e.drop(columns="checkpoint_family")
    elif bad == "plot":
        a.loc[ai, "plot_idx"] = "other"
    elif bad == "q_float32":
        r["q_rescue"] = r.q_rescue.astype(np.float32)
    elif bad == "q_sum":
        r.loc[ri, "q_rescue"] += 0.2
    elif bad == "q_nan":
        r.loc[ri, "q_rescue"] = float("nan")
    elif bad == "q_score":
        r.loc[ri, "score"] = 0.333
    elif bad == "action_score":
        a.loc[ai, "score"] = 0.333
    elif bad == "threshold_declaration":
        specs[name]["threshold_specification"]["value_hex"] = (0.6).hex()
    elif bad == "threshold_duplicate_json":
        a.loc[a.policy_id.eq(name), "threshold_specification"] = (
            '{"kind":"finite","kind":"finite","comparison":"strict_gt","value_hex":"0x0.0p+0"}'
        )
    elif bad == "acted_float":
        a["acted"] = a.acted.astype(float)
    elif bad == "acted_mismatch":
        a.loc[ai, "acted"] = not a.loc[ai, "acted"]
    elif bad == "expert_prediction":
        a.loc[ai, "raw_pred"] = 17
    elif bad == "final_prediction":
        a.loc[ai, "final_pred"] = 17
    elif bad == "heuristic_score":
        a.loc[a.policy_id.eq("geo_minus_raw_margin"), "score"] = -1.0
    elif bad == "label_ontology":
        labels.loc[0, "canonical_l3_id"] = 19
    elif bad == "missing_label":
        labels = labels.iloc[1:]
    elif bad == "label_plot":
        labels.loc[0, "plot_idx"] = "other"
    elif bad == "missing_seed":
        r = r.loc[r.training_seed.ne(4)]
    elif bad == "noncanonical_rows":
        e = e.iloc[::-1]
    with pytest.raises(report.LegacyCapacityReportError):
        report.evaluate_family(e, r, a, labels, specs)


def test_no_file_discovery_or_input_mutation(scored, monkeypatch):
    import pyarrow.parquet as pq

    args = deepcopy(scored[0])
    before = deepcopy(args)

    def forbidden(*args, **kwargs):
        raise AssertionError("report attempted selection or file access")

    for module, attribute in (
        (pd, "read_parquet"),
        (pq, "read_table"),
        (Path, "open"),
    ):
        monkeypatch.setattr(module, attribute, forbidden)
    result = report.evaluate_family(*args)
    assert result["summary.json"] == scored[1]["summary.json"]
    for actual, expected in zip(args, before):
        if isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            assert actual == expected


def test_probability_floor_and_argmax_ties_retain_frozen_conventions(scored):
    e, r, a, labels, specs = deepcopy(scored[0])
    name = "mlp__primary_full727"
    indices = r.index[r.policy_id.eq(name)]
    r.loc[indices[0], list(report.Q_COLUMNS)] = [0.0, 0.8, 0.1, 0.1]
    r.loc[indices[1], list(report.Q_COLUMNS)] = [0.25, 0.25, 0.25, 0.25]
    selected = r.loc[r.policy_id.eq(name)]
    r.loc[r.policy_id.eq(name), "score"] = (
        selected.q_rescue.to_numpy() - selected.q_harm.to_numpy()
    )
    a.loc[a.policy_id.eq(name), "score"] = r.loc[
        r.policy_id.eq(name), "score"
    ].to_numpy()
    result = report.evaluate_family(e, r, a, labels, specs)
    targets = result["scoring_targets.parquet"].to_pandas()
    q = r.loc[r.policy_id.eq(name), list(report.Q_COLUMNS)].to_numpy()
    y = targets.target_state.map(
        {s: i for i, s in enumerate(legacy.TARGET_ORDER)}
    ).to_numpy()
    expected = legacy.multiclass_metrics(q, y)
    actual = result["router_metrics.json"][name]["all_rows"]["pooled_seed_realizations"]
    assert actual["true_probability_floor_count"] == 1
    for metric in (
        "log_loss",
        "brier_score",
        "top1_accuracy",
        "true_probability_floor_hex",
    ):
        assert actual[metric] == expected[metric]


def test_import_is_fitting_incapable():
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn', 'xgboost'} or fullname in {
            'multimodal.geo_helpfulness_router_capacity_score',
            'multimodal.geo_helpfulness_router_selection',
        }:
            raise AssertionError('forbidden report import: ' + fullname)
sys.meta_path.insert(0, Guard())
from multimodal.geo_helpfulness_legacy_capacity_report import evaluate_family
assert not {'torch', 'scipy', 'sklearn', 'xgboost'}.intersection(sys.modules)
assert 'multimodal.geo_helpfulness_router_capacity_score' not in sys.modules
assert 'multimodal.geo_helpfulness_router_selection' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

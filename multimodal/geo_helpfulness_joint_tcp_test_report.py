"""Pure cleaned-test evaluation of frozen joint TCP routers and policies.

The caller authenticates models, predictions, configuration, and validation
selections. Policy application consumes no labels. Only the scoring entry point
accepts labels; neither entry point fits or selects anything or accesses files.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet_report import _probabilities
from multimodal.geo_helpfulness_joint_tcp_report import (
    JointTCPReportError, Q_COLUMNS, SEEDS, STATES, _bootstrap, _mean_tree,
    _policy_metrics, _split_diagnostics,
)
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256
from multimodal.geo_helpfulness_router_heuristics import (
    apply_heuristic_threshold, validate_threshold_specification,
)
from multimodal.geo_helpfulness_router_metrics import classification_metrics


IDENTITIES = ("row_uid", "plot_idx", "training_seed", "producer_fold")
METADATA = ("config_id", "input_id", "variant", "with_embeddings", "auxiliary_weight")
REQUIRED = frozenset((*IDENTITIES, *METADATA, *Q_COLUMNS, "split",
                      "raw_logits", "geo_logits", "image_logits", "state_logits",
                      "tcp_logits", "fusion_tcp_hat", "geo_tcp_hat"))
GUARDRAILS = tuple((p, h) for p in (20, 10, 0) for h in (5, 7, 9))


def _declarations(config: dict) -> list[dict]:
    items = config.get("configs", [])
    combinations = {(variant, emb, weight)
                    for variant in ("ablation_numeric25", "ablation_no_pairs79")
                    for emb in (False, True) for weight in (0, .1, 1, 10)}
    if (len(items) != 16 or len({x["config_id"] for x in items}) != 16
            or {(x["variant"], x["with_embeddings"], x["auxiliary_weight"])
                for x in items} != combinations):
        raise JointTCPReportError("all sixteen declared joint configurations are required")
    if (config.get("training_seeds", list(SEEDS)) != list(SEEDS)
            or config.get("minimum_plots_grid", [20, 10, 0]) != [20, 10, 0]
            or config.get("harm_numerator_grid", [5, 7, 9]) != [5, 7, 9]):
        raise JointTCPReportError("seeds and guardrail grid must match the frozen development study")
    for item in items:
        base = "numeric25" if item["variant"] == "ablation_numeric25" else "no_pairs79"
        expected_id = base + ("_embeddings" if item["with_embeddings"] else "")
        width = (25 if base == "numeric25" else 79) + 1216 * item["with_embeddings"]
        if item["input_id"] != expected_id or item["input_dim"] != width:
            raise JointTCPReportError("declared feature identity or width is inconsistent")
    return items


def _validate_selections(config: dict, selections: dict) -> None:
    expected = {f"{item['config_id']}__plots_{p}__harm_0p{h}"
                for item in _declarations(config) for p, h in GUARDRAILS}
    if not isinstance(selections, dict) or set(selections) != expected:
        raise JointTCPReportError("all 144 frozen validation selections are required")
    for item in config["configs"]:
        for plots, harm in GUARDRAILS:
            guardrail = f"plots_{plots}__harm_0p{harm}"
            name = f"{item['config_id']}__{guardrail}"
            selected = selections[name]
            expected_guard = {
                "guardrail_id": guardrail, "minimum_plots": plots,
                "harm_numerator": harm, "harm_denominator": 10,
                "minimum_coverage_fraction": .01, "require_positive_net_utility": True,
            }
            if (selected.get("config_id") != item["config_id"]
                    or selected.get("guardrails") != expected_guard):
                raise JointTCPReportError("frozen threshold configuration or guardrail identity differs")
            specification = selected.get("threshold_specification")
            validate_threshold_specification(specification)
            if selected.get("threshold_specification_sha256") != canonical_sha256(specification):
                raise JointTCPReportError("frozen threshold specification hash differs")
            if selected.get("selected_feasible") is not (specification["kind"] == "finite"):
                raise JointTCPReportError("frozen threshold feasibility identity differs")


def _validate(rows: pd.DataFrame, config: dict, *, labels: bool) -> dict[str, pd.DataFrame]:
    required = REQUIRED | ({"label_id_dense"} if labels else set())
    allowed = required | {"input_dim"}
    if (not isinstance(rows, pd.DataFrame) or rows.empty or rows.columns.duplicated().any()
            or not required.issubset(rows) or not set(rows).issubset(allowed)):
        raise JointTCPReportError("test prediction columns are incomplete or contain unexpected/label fields")
    items = _declarations(config)
    if set(rows.config_id) != {item["config_id"] for item in items} or set(rows.split) != {"test"}:
        raise JointTCPReportError("test configuration or split identities differ")
    if (rows[list(IDENTITIES)].isna().any().any()
            or rows.training_seed.dtype.kind not in "iu"
            or rows.producer_fold.dtype.kind not in "iu"
            or not rows.producer_fold.eq(-1).all()):
        raise JointTCPReportError("test identities require integer seeds and full-development producers (-1)")
    if rows.duplicated(["config_id", "training_seed", "row_uid"]).any():
        raise JointTCPReportError("duplicate test prediction identity")
    if labels and (rows.label_id_dense.dtype.kind not in "iu"
                   or not rows.label_id_dense.between(0, 17).all()):
        raise JointTCPReportError("test labels require fixed dense integer ontology 0..17")
    prepared, reference = {}, None
    for item in items:
        frame = rows.loc[rows.config_id.eq(item["config_id"])].sort_values(
            ["row_uid", "training_seed"], kind="stable"
        ).reset_index(drop=True).copy()
        for key in METADATA:
            if not frame[key].eq(item[key]).all():
                raise JointTCPReportError(f"test metadata differs: {item['config_id']}/{key}")
        if "input_dim" in frame and not frame.input_dim.eq(item["input_dim"]).all():
            raise JointTCPReportError("test input widths differ from declaration")
        present_seeds = sorted(frame.training_seed.unique())
        if (not present_seeds or not set(present_seeds).issubset(SEEDS)
                or (labels and present_seeds != list(SEEDS))):
            raise JointTCPReportError("test predictions lack a training seed or use an unknown seed")
        identity_columns = ["row_uid", "plot_idx"] + (["label_id_dense"] if labels else [])
        first_seed = frame.loc[frame.training_seed.eq(present_seeds[0]), identity_columns].reset_index(drop=True)
        for seed in present_seeds[1:]:
            current = frame.loc[frame.training_seed.eq(seed), identity_columns].reset_index(drop=True)
            if not current.equals(first_seed):
                raise JointTCPReportError("test row/plot/label identities differ across seeds")
        for expert in ("image", "raw", "geo"):
            logits = np.stack(frame[f"{expert}_logits"])
            if logits.shape != (len(frame), 18) or not np.isfinite(logits).all():
                raise JointTCPReportError("test expert logits must be finite (n,18)")
            frame[f"{expert}_pred"] = logits.argmax(axis=1)
        q = frame[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
        estimate = frame[["fusion_tcp_hat", "geo_tcp_hat"]].to_numpy(dtype=np.float64)
        if (not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()
                or not np.allclose(q.sum(1), 1, atol=1e-6, rtol=0)):
            raise JointTCPReportError("test probabilities must form finite simplex rows")
        if not np.isfinite(estimate).all() or (estimate < 0).any() or (estimate > 1).any():
            raise JointTCPReportError("test TCP estimates must be finite within [0,1]")
        state_logits = np.stack(frame.state_logits).astype(np.float64)
        tcp_logits = np.stack(frame.tcp_logits).astype(np.float64)
        if (state_logits.shape != (len(frame), 4) or tcp_logits.shape != (len(frame), 2)
                or not np.isfinite(state_logits).all() or not np.isfinite(tcp_logits).all()):
            raise JointTCPReportError("test head logits must be finite (n,4)/(n,2)")
        exponent = np.exp(-np.abs(tcp_logits))
        tcp = np.where(tcp_logits >= 0, 1 / (1 + exponent), exponent / (1 + exponent))
        if (not np.allclose(_probabilities(state_logits), q, atol=1e-6, rtol=0)
                or not np.allclose(tcp, estimate, atol=1e-6, rtol=0)):
            raise JointTCPReportError("test probabilities disagree with saved logits")
        frame["score"] = q[:, 0] - q[:, 1]
        common = list(IDENTITIES) + (["label_id_dense"] if labels else [])
        if reference is not None:
            if not frame[common].equals(reference[common]):
                raise JointTCPReportError("test identities differ across configurations")
            for expert in ("image", "raw", "geo"):
                if not np.array_equal(np.stack(frame[f"{expert}_logits"]),
                                      np.stack(reference[f"{expert}_logits"])):
                    raise JointTCPReportError("test expert logits differ across configurations")
        else:
            reference = frame
        if labels:
            fusion_ok = frame.raw_pred.eq(frame.label_id_dense)
            geo_ok = frame.geo_pred.eq(frame.label_id_dense)
            frame["target_state"] = np.select(
                [~fusion_ok & geo_ok, fusion_ok & ~geo_ok, fusion_ok & geo_ok],
                STATES[:3], default=STATES[3],
            )
        prepared[item["config_id"]] = frame
    return prepared


def _actions(prepared: dict, config: dict, selections: dict) -> tuple[dict, dict]:
    _validate_selections(config, selections)
    first = next(iter(prepared.values()))
    actions = {"fusion": np.zeros(len(first), dtype=bool)}
    metadata = {"fusion": {"kind": "reference"}}
    for item in config["configs"]:
        name, frame = item["config_id"], prepared[item["config_id"]]
        score = frame.score.to_numpy(dtype=np.float64)
        actions[f"{name}__zero"] = frame.raw_pred.ne(frame.geo_pred).to_numpy() & (score > 0)
        metadata[f"{name}__zero"] = {"kind": "fixed_zero", "config_id": name}
        for plots, harm in GUARDRAILS:
            guardrail = f"plots_{plots}__harm_0p{harm}"
            policy = f"{name}__{guardrail}"
            selected = selections[policy]
            actions[policy] = apply_heuristic_threshold(
                score, frame.raw_pred.to_numpy(), frame.geo_pred.to_numpy(),
                selected["threshold_specification"],
            )
            metadata[policy] = {
                "kind": "frozen_validation_threshold", "config_id": name,
                "guardrail_id": guardrail,
                "threshold_specification": deepcopy(selected["threshold_specification"]),
                "threshold_specification_sha256": selected["threshold_specification_sha256"],
                "validation_selected_feasible": selected["selected_feasible"],
                "validation_guardrails": deepcopy(selected["guardrails"]),
            }
    return actions, metadata


def _action_frame(reference: pd.DataFrame, actions: dict, *, labels: bool) -> pd.DataFrame:
    columns = list(IDENTITIES) + ["raw_pred", "geo_pred", "image_pred"]
    if labels:
        columns += ["label_id_dense", "target_state"]
    return pd.concat([reference[columns].copy(), pd.DataFrame(actions)], axis=1)


def policy_actions(rows: pd.DataFrame, config: dict, selections: dict) -> pd.DataFrame:
    """Apply frozen policies to label-free predictions; never select thresholds."""
    prepared = _validate(rows, config, labels=False)
    actions, _ = _actions(prepared, config, selections)
    return _action_frame(next(iter(prepared.values())), actions, labels=False)


def _expert_metrics(frame: pd.DataFrame, expert: str) -> dict:
    def metrics(part: pd.DataFrame) -> dict:
        values = classification_metrics(part[f"{expert}_pred"].to_numpy(),
                                        part.label_id_dense.to_numpy())
        return {"accuracy": values["accuracy"], "weighted_f1": values["weighted_f1"],
                "row_count": len(part), "plot_count": int(part.plot_idx.nunique())}
    by_seed = {str(seed): metrics(frame.loc[frame.training_seed.eq(seed)]) for seed in SEEDS}
    return {"per_seed": by_seed, "mean_seed": _mean_tree(list(by_seed.values())),
            "pooled": metrics(frame)}


def make_report(rows: pd.DataFrame, config: dict, selections: dict) -> dict[str, Any]:
    """Score sealed test predictions using labels joined only by the caller."""
    prepared = _validate(rows, config, labels=True)
    actions, metadata = _actions(prepared, config, selections)
    reference = next(iter(prepared.values()))
    policies = {name: {"metadata": metadata[name], **_policy_metrics(reference, action)}
                for name, action in actions.items()}
    for entry in policies.values():
        if entry["metadata"]["kind"] != "frozen_validation_threshold":
            continue
        counts = entry["pooled"]
        guard = entry["metadata"]["validation_guardrails"]
        entry["test_guardrail_diagnostics"] = {
            "harm_cap_met": 10 * counts["harms"] <= guard["harm_numerator"] * counts["rescues"],
            "minimum_plots_met": counts["plots_affected"] >= guard["minimum_plots"],
            "minimum_coverage_met": 100 * counts["actions"] >= counts["row_count"],
            "positive_net_corrections": counts["net"] > 0,
            "policy_changed_using_test_labels": False,
        }
    diagnostics = {item["config_id"]: {
        "configuration": deepcopy(item),
        "test": _split_diagnostics(prepared[item["config_id"]], item["auxiliary_weight"]),
    } for item in config["configs"]}
    intervals, draws = _bootstrap(reference, actions, config)
    intervals["threshold_source"] = "sealed_development_validation_selection"
    intervals["test_used_for_selection"] = False
    report = {
        "schema_version": "geo_helpfulness.joint_tcp_cleaned_test_report.v1",
        "study_id": "joint_tcp_router_test_v1",
        "parent_study_id": config.get("study_id", "joint_tcp_router_v1"),
        "training_seeds": list(SEEDS), "four_state_order": list(STATES),
        "configuration_count": len(config["configs"]), "model_count": 4 * len(config["configs"]),
        "selected_policy_count": len(selections), "policy_count_including_references": len(actions),
        "configuration_order": [item["config_id"] for item in config["configs"]],
        "policy_order": list(actions), "native_predictions_dtype": "float32",
        "metric_arithmetic_dtype": "float64",
        "score_convention": "float64(q_rescue)-float64(q_harm)",
        "nll_convention": "stable_float64_logsumexp_from_saved_state_logits",
        "tcp_target_convention": "native_temperature_one_softmax_authoritative_expert_logits_float64_scoring",
        "threshold_source": "sealed_development_validation_selection",
        "lambda_winner_selected": False, "independent_confirmation": False,
        "test_used_for_selection": False, "fitting_performed_by_scorer": False,
        "test_harm_caps_enforced": False,
        "validation_selections_sha256": canonical_sha256(selections),
        "expert_references": {name: _expert_metrics(reference, source)
                              for name, source in (("fusion", "raw"), ("image", "image"), ("geo", "geo"))},
        "diagnostics": diagnostics, "test_policies": policies, "test_bootstrap": intervals,
    }
    return {"report": report, "actions": _action_frame(reference, actions, labels=True),
            "bootstrap_draws": draws, "bootstrap_policy_order": list(actions)}


def format_report(report: dict) -> str:
    """Render every setting deterministically, including after sorted JSON IO."""
    policies = report["test_policies"]
    lines = [
        "# Joint TCP routers on the cleaned test set", "",
        "All 64 previously fitted routers are evaluated. Every threshold was frozen on "
        "development-validation; no model, auxiliary weight, or threshold is selected using test labels. "
        "The cleaned test set has been examined in earlier studies, so this remains exploratory.", "",
        "Accuracy and weighted F1 average four training seeds. Routing counts pool seed-image "
        "realizations; affected plots are distinct plots across seeds. Bootstrap intervals resample "
        "whole plots with all images and seed realizations together, holding models and thresholds fixed.", "",
        "## Expert references", "",
        "| Expert | Accuracy (%) | Weighted F1 |", "| --- | ---: | ---: |",
    ]
    for name in ("fusion", "image", "geo"):
        item = report["expert_references"][name]["mean_seed"]
        lines.append(f"| {name} | {100 * item['accuracy']:.4f} | {item['weighted_f1']:.5f} |")
    lines += ["", "## All sixteen configurations", "",
              "This compact table uses the frozen minimum-20-plots selections. The complete table "
              "below also includes the minimum-10 and minimum-0 policies. Harm caps describe validation "
              "selection constraints, not constraints re-enforced on test outcomes.", "",
              "| Inputs | Lambda | Fixed zero (%) | H <= 0.5R (%) | H <= 0.7R (%) | H <= 0.9R (%) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in report["configuration_order"]:
        item = report["diagnostics"][name]["configuration"]
        suffixes = ["zero"] + [f"plots_20__harm_0p{h}" for h in (5, 7, 9)]
        values = [100 * policies[f"{name}__{suffix}"]["mean_seed"]["accuracy"] for suffix in suffixes]
        lines.append(f"| {item['input_id']} | {item['auxiliary_weight']:g} | "
                     + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "## State and TCP diagnostics", "",
              "TCP outputs at lambda zero are unsupervised. NLL uses stable float64 log-sum-exp "
              "from saved native float32 logits.", "",
              "| Configuration | NLL | Brier | State accuracy | Macro F1 | Fusion TCP MSE | Geo TCP MSE |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in report["configuration_order"]:
        item = report["diagnostics"][name]["test"]["mean_seed"]
        state = item["state"]
        lines.append(f"| {name} | {state['nll']:.5f} | {state['brier']:.5f} | "
                     f"{state['accuracy']:.5f} | {state['macro_f1']:.5f} | "
                     f"{item['tcp']['fusion']['tcp_regression']['mse']:.5f} | "
                     f"{item['tcp']['geo']['tcp_regression']['mse']:.5f} |")
    lines += ["", "## Every frozen policy", "",
              "Finite thresholds and never-intervene fallbacks are applied unchanged even when test "
              "outcomes violate the validation harm cap or yield negative net corrections.", "",
              "| Policy | Accuracy (%) | Weighted F1 | Actions | Rescues | Harms | Net | Coverage (%) | Plots |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in report["policy_order"]:
        item = policies[name]
        mean, count = item["mean_seed"], item["pooled"]
        lines.append(f"| {name} | {100 * mean['accuracy']:.4f} | {mean['weighted_f1']:.5f} | "
                     f"{count['actions']} | {count['rescues']} | {count['harms']} | {count['net']} | "
                     f"{100 * count['coverage']:.3f} | {count['plots_affected']} |")
    lines += ["", "## Matched contrasts", "",
              "Differences and 95% percentile intervals are in percentage points. Each comparison "
              "uses the same guardrail identity and the thresholds previously selected for its two "
              "configurations. All comparisons are exploratory, without multiplicity adjustment.", "",
              "| Treatment minus reference | Accuracy difference [95% CI] | Weighted F1 difference [95% CI] |",
              "| --- | ---: | ---: |"]
    contrasts = report["test_bootstrap"]["contrasts"]
    for key in sorted(contrasts):
        item = contrasts[key]
        if item["kind"] == "versus_fusion":
            continue
        values = []
        for metric in ("accuracy", "weighted_f1"):
            value = item[metric]
            values.append(f"{100 * value['difference']:+.4f} "
                          f"[{100 * value['ci95_lower']:+.4f}, {100 * value['ci95_upper']:+.4f}]")
        lines.append(f"| {item['treatment']} minus {item['reference']} | {values[0]} | {values[1]} |")
    lines += ["", "The machine-readable report includes all 360 contrasts, per-seed metrics, "
              "failure-detection AP/AUC, TCP saturation and ordering, and observed test guardrail "
              "diagnostics. No winning lambda or test-selected policy is promoted.", ""]
    return "\n".join(lines)

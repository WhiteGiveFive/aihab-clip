"""Pure scoring of shared MLP capacity and auxiliary TCP supervision.

All model inputs are numerical summaries or summaries plus predicted-class
indicators. Test application transfers frozen development thresholds to the
historical best-model expert family. This module has no file access or fitting.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet_report import _probabilities
from multimodal.geo_helpfulness_joint_tcp_report import (
    JointTCPReportError, Q_COLUMNS, SEEDS, STATES, _mean_tree,
    _policy_metrics, _split_diagnostics, _select, _validate as _validate_development,
)
from multimodal.geo_helpfulness_joint_tcp_test_report import _expert_metrics
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256
from multimodal.geo_helpfulness_router_heuristics import (
    apply_heuristic_threshold, validate_threshold_specification,
)

IDENTITIES = ("row_uid", "plot_idx", "training_seed", "producer_fold")
METADATA = ("config_id", "input_id", "variant", "with_embeddings", "auxiliary_weight")
REQUIRED = frozenset((*IDENTITIES, *METADATA, *Q_COLUMNS, "split",
                      "raw_logits", "geo_logits", "image_logits", "state_logits",
                      "tcp_logits", "fusion_tcp_hat", "geo_tcp_hat"))
GUARDRAILS = tuple((p, h) for p in (20, 10, 0) for h in (5, 7, 9))
ARCHITECTURES = {
    "h32": [32], "h64_32": [64, 32], "h128_64": [128, 64],
    "h64_32_16": [64, 32, 16], "h128_64_32": [128, 64, 32],
}



def _declarations(config: dict) -> list[dict]:
    items = config.get("configs", [])
    combinations = {(architecture, variant, weight)
                    for architecture in ARCHITECTURES
                    for variant in ("ablation_numeric25", "ablation_no_pairs79")
                    for weight in (0, .1, 1, 10)}
    if (len(items) != 40 or len({item["config_id"] for item in items}) != 40
            or {(item["architecture_id"], item["variant"], item["auxiliary_weight"])
                for item in items} != combinations):
        raise JointTCPReportError("all forty declared capacity configurations are required")
    if (config.get("training_seeds", list(SEEDS)) != list(SEEDS)
            or config.get("minimum_plots_grid", [20, 10, 0]) != [20, 10, 0]
            or config.get("harm_numerator_grid", [5, 7, 9]) != [5, 7, 9]):
        raise JointTCPReportError("seeds and guardrail grid differ from the declared study")
    for item in items:
        base = item["variant"].removeprefix("ablation_")
        if (item["hidden_widths"] != ARCHITECTURES[item["architecture_id"]]
                or item["with_embeddings"] is not False or item["input_id"] != base
                or item["input_dim"] != (25 if base == "numeric25" else 79)):
            raise JointTCPReportError("architecture or input dimensions differ from declaration")
    return items


def _validate_selections(config: dict, selections: dict) -> None:
    expected = {f"{item['config_id']}__plots_{p}__harm_0p{h}"
                for item in _declarations(config) for p, h in GUARDRAILS}
    if not isinstance(selections, dict) or set(selections) != expected:
        raise JointTCPReportError("all 360 frozen validation selections are required")
    for item in config["configs"]:
        for plots, harm in GUARDRAILS:
            guardrail = f"plots_{plots}__harm_0p{harm}"
            selected = selections[f"{item['config_id']}__{guardrail}"]
            expected_guard = {
                "guardrail_id": guardrail, "minimum_plots": plots,
                "harm_numerator": harm, "harm_denominator": 10,
                "minimum_coverage_fraction": .01, "require_positive_net_utility": True,
            }
            if (selected.get("config_id") != item["config_id"]
                    or selected.get("guardrails") != expected_guard):
                raise JointTCPReportError("frozen threshold configuration or guardrail differs")
            specification = selected.get("threshold_specification")
            validate_threshold_specification(specification)
            if selected.get("threshold_specification_sha256") != canonical_sha256(specification):
                raise JointTCPReportError("frozen threshold specification hash differs")
            if selected.get("selected_feasible") is not (specification["kind"] == "finite"):
                raise JointTCPReportError("frozen threshold feasibility differs")


def _validate(rows: pd.DataFrame, config: dict, *, labels: bool) -> dict[str, pd.DataFrame]:
    required = REQUIRED | ({"label_id_dense"} if labels else set())
    allowed = required | {"input_dim", "architecture_id", "hidden_widths", "source"}
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
        raise JointTCPReportError("test identities require integer seeds and frozen legacy producers (-1)")
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
        for key in (*METADATA, "architecture_id") if "architecture_id" in frame else METADATA:
            if not frame[key].eq(item[key]).all():
                raise JointTCPReportError(f"test metadata differs: {item['config_id']}/{key}")
        if "hidden_widths" in frame and not frame.hidden_widths.map(
                lambda value: tuple(value) == tuple(item["hidden_widths"])).all():
            raise JointTCPReportError("test hidden widths differ from declaration")
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


def _add_comparators(reference: pd.DataFrame, actions: dict, metadata: dict,
                     comparators: dict | None) -> dict:
    """Align six authenticated historical policies before observing labels."""
    if comparators is None:
        return {}
    if not isinstance(comparators, dict) or set(comparators) != {"actions", "metadata"}:
        raise JointTCPReportError("comparators require actions and metadata")
    frame, records = comparators["actions"], comparators["metadata"]
    keys = ["row_uid", "plot_idx", "training_seed"]
    if (not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any()
            or not isinstance(records, dict) or len(records) != 6
            or set(frame) != set(keys + ["raw_pred", "geo_pred", *records])
            or frame[keys].isna().any().any() or frame.duplicated(keys).any()):
        raise JointTCPReportError("six complete label-free historical action policies are required")
    expected = {(family, input_id) for family in ("rf", "lr")
                for input_id in ("numeric25", "no_pairs79", "full727")}
    if {(record.get("family"), record.get("input_id")) for record in records.values()} != expected:
        raise JointTCPReportError("historical policies require each family and feature combination")
    # A prediction worker may hold one seed while receiving the authenticated
    # four-seed reference bundle. Align the declared rows, never their labels.
    frame = frame.loc[frame.training_seed.isin(reference.training_seed.unique())].sort_values(
        ["row_uid", "training_seed"], kind="stable").reset_index(drop=True)
    if not frame[keys].equals(reference[keys]):
        raise JointTCPReportError("historical policy identities differ from current predictions")
    for expert in ("raw", "geo"):
        if not np.array_equal(frame[f"{expert}_pred"], reference[f"{expert}_pred"]):
            raise JointTCPReportError("historical policies use different expert predictions")
    added = {}
    for name in sorted(records):
        record = records[name]
        if (name in actions or record.get("guardrail_id") != "plots_20__harm_0p7"
                or frame[name].dtype != np.dtype(bool)):
            raise JointTCPReportError("historical policy metadata or action type differs")
        action = frame[name].to_numpy(dtype=bool)
        if np.any(action & reference.raw_pred.eq(reference.geo_pred).to_numpy()):
            raise JointTCPReportError("historical intervention violates disagreement gate")
        actions[name] = action
        metadata[name] = {"kind": "historical_reference", **deepcopy(record)}
        added[name] = deepcopy(record)
    return added


def policy_actions(rows: pd.DataFrame, config: dict, selections: dict,
                   comparators: dict | None = None) -> pd.DataFrame:
    """Apply all frozen policies to label-free predictions, optionally per seed."""
    prepared = _validate(rows, config, labels=False)
    actions, metadata = _actions(prepared, config, selections)
    reference = next(iter(prepared.values()))
    _add_comparators(reference, actions, metadata, comparators)
    return _action_frame(reference, actions, labels=False)


def _contrast_pairs(config: dict, historical: dict | None = None) -> list[tuple[str, str, str]]:
    items = _declarations(config)
    suffixes = ["zero"] + [f"plots_{p}__harm_0p{h}" for p, h in GUARDRAILS]
    pairs = [(f"{item['config_id']}__{suffix}", "fusion", "versus_fusion")
             for item in items for suffix in suffixes]
    for item in items:
        controls = []
        if item["auxiliary_weight"] > 0:
            baseline = next(other for other in items
                            if other["architecture_id"] == item["architecture_id"]
                            and other["input_id"] == item["input_id"]
                            and other["auxiliary_weight"] == 0)
            controls.append((baseline, "auxiliary_weight_versus_zero"))
        if item["architecture_id"] != "h32":
            baseline = next(other for other in items if other["architecture_id"] == "h32"
                            and other["input_id"] == item["input_id"]
                            and other["auxiliary_weight"] == item["auxiliary_weight"])
            controls.append((baseline, "larger_architecture_versus_h32"))
        for baseline, kind in controls:
            pairs.extend((f"{item['config_id']}__{suffix}",
                          f"{baseline['config_id']}__{suffix}", kind) for suffix in suffixes)
        for name, record in sorted((historical or {}).items()):
            if record["input_id"] == item["input_id"]:
                pairs.append((f"{item['config_id']}__plots_20__harm_0p7", name,
                              f"matched_input_versus_historical_{record['family']}"))
    return pairs


def _bootstrap(frame: pd.DataFrame, actions: dict, config: dict,
               historical: dict | None = None) -> tuple[dict, np.ndarray]:
    """Whole-plot paired bootstrap; subtract integer correct counts for accuracy.

    This follows the parent scorer's sufficient-statistic algorithm and exact
    plot draws. Integer subtraction preserves exact zero differences even when
    floating subtraction of two seed-averaged accuracies would not.
    """
    names = list(actions)
    plots, inverse = np.unique(frame.plot_idx, return_inverse=True)
    settings = config.get("bootstrap", {"resamples": 2000, "seed": 20260826})
    if type(settings["resamples"]) is not int or settings["resamples"] < 1:
        raise JointTCPReportError("bootstrap resamples must be a positive integer")
    rng = np.random.default_rng(settings["seed"])
    weights = np.empty((settings["resamples"], len(plots)), dtype=np.float64)
    for index in range(len(weights)):
        weights[index] = np.bincount(rng.integers(0, len(plots), size=len(plots)),
                                    minlength=len(plots))
    unique, unique_actions, mapping = {}, [], []
    for action in actions.values():
        key = np.packbits(action).tobytes()
        if key not in unique:
            unique[key] = len(unique_actions)
            unique_actions.append(action)
        mapping.append(unique[key])
    labels = frame.label_id_dense.to_numpy(dtype=np.int64)
    draws = np.empty((len(weights), len(unique_actions), 2), dtype=np.float64)
    observed = np.empty((len(unique_actions), 2), dtype=np.float64)
    correct = np.empty((len(weights), len(unique_actions), len(SEEDS)), dtype=np.int64)
    observed_correct = np.empty((len(unique_actions), len(SEEDS)), dtype=np.int64)
    denominator, observed_denominator = None, None

    def metrics(counts: np.ndarray) -> np.ndarray:
        truth, predicted, tp = counts[..., 2:20], counts[..., 20:38], counts[..., 38:56]
        f1 = np.divide(2 * tp, truth + predicted, out=np.zeros_like(tp), where=truth + predicted != 0)
        return np.stack((counts[..., 1] / counts[..., 0],
                         (truth * f1).sum(-1) / counts[..., 0]), axis=-1).mean(axis=-2)

    for index, action in enumerate(unique_actions):
        prediction = np.where(action, frame.geo_pred, frame.raw_pred)
        group = np.zeros((len(plots), len(SEEDS), 56), dtype=np.float64)
        for seed_index, seed in enumerate(SEEDS):
            mask = frame.training_seed.eq(seed).to_numpy()
            p, y, g = prediction[mask], labels[mask], inverse[mask]
            group[:, seed_index, 0] = np.bincount(g, minlength=len(plots))
            group[:, seed_index, 1] = np.bincount(g, weights=p == y, minlength=len(plots))
            for cls in range(18):
                for offset, indicator in ((2, y == cls), (20, p == cls), (38, (y == cls) & (p == cls))):
                    group[:, seed_index, offset + cls] = np.bincount(g, weights=indicator, minlength=len(plots))
        counts = (weights @ group.reshape(len(plots), -1)).reshape(len(weights), len(SEEDS), 56)
        totals = group.sum(axis=0)
        draws[:, index], observed[index] = metrics(counts), metrics(totals)
        correct[:, index] = counts[..., 1].astype(np.int64)
        observed_correct[index] = totals[..., 1].astype(np.int64)
        if denominator is None:
            denominator = counts[..., 0].astype(np.int64)
            observed_denominator = totals[..., 0].astype(np.int64)
        elif not np.array_equal(denominator, counts[..., 0]):
            raise JointTCPReportError("bootstrap policy row counts disagree")
    lookup = {name: mapping[index] for index, name in enumerate(names)}
    contrasts = {}
    for treatment, reference, kind in _contrast_pairs(config, historical):
        left, right = lookup[treatment], lookup[reference]
        delta = draws[:, left] - draws[:, right]
        difference = observed[left] - observed[right]
        if not np.all(denominator == denominator[:, :1]):
            raise JointTCPReportError("paired bootstrap requires identical image counts across seeds")
        delta[:, 0] = (correct[:, left] - correct[:, right]).sum(axis=1) / denominator.sum(axis=1)
        difference[0] = ((observed_correct[left] - observed_correct[right]).sum()
                         / observed_denominator.sum())
        interval = np.quantile(delta, [.025, .975], axis=0)
        contrasts[f"{treatment}_minus_{reference}"] = {
            "treatment": treatment, "reference": reference, "kind": kind,
            **{metric: {"difference": float(difference[i]),
                        "ci95_lower": float(interval[0, i]), "ci95_upper": float(interval[1, i])}
               for i, metric in enumerate(("accuracy", "weighted_f1"))},
        }
    return {**settings, "plot_count": len(plots), "policy_order": names,
            "metric_order": ["accuracy", "weighted_f1"],
            "unique_action_masks": len(unique_actions), "contrasts": contrasts,
            "contrast_count": len(contrasts),
            "accuracy_contrast_arithmetic": "pooled_integer_correct_count_difference_before_division_equal_seed_denominators",
            "unit": "whole_plot_all_images_all_seed_realizations",
            "thresholds_refitted_within_bootstrap": False,
            "independent_confirmation": False}, draws[:, mapping]


def _report_base(config: dict, actions: dict, selections: dict, split: str) -> dict:
    return {
        "schema_version": "geo_helpfulness.joint_tcp_capacity_report.v1",
        "study_id": config.get("study_id", "joint_tcp_router_capacity_v1"),
        "evaluation_split": split, "training_seeds": list(SEEDS),
        "four_state_order": list(STATES), "configuration_count": len(config["configs"]),
        "model_count": 4 * len(config["configs"]), "new_fit_count": 128,
        "reused_h32_fit_count": 32,
        "selected_policy_count": len(selections), "policy_count_including_references": len(actions),
        "configuration_order": [item["config_id"] for item in config["configs"]],
        "policy_order": list(actions), "native_predictions_dtype": "float32",
        "metric_arithmetic_dtype": "float64",
        "score_convention": "float64(q_rescue)-float64(q_harm)",
        "nll_convention": "stable_float64_logsumexp_from_saved_state_logits",
        "tcp_target_convention": "native_temperature_one_softmax_authoritative_expert_logits_float64_scoring",
        "lambda_winner_selected": False, "architecture_winner_selected": False,
        "independent_confirmation": False, "fitting_performed_by_scorer": False,
    }


def make_development_report(rows: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Score all sealed train/validation rows and fit the declared thresholds."""
    items = _declarations(config)
    for item in items:
        source = rows.loc[rows.config_id.eq(item["config_id"])]
        for key in ("architecture_id", "input_id", "input_dim"):
            if key in source and not source[key].eq(item[key]).all():
                raise JointTCPReportError("development architecture/input metadata differs")
        if "hidden_widths" in source and not source.hidden_widths.map(
                lambda value: tuple(value) == tuple(item["hidden_widths"])).all():
            raise JointTCPReportError("development hidden widths differ from declaration")
    prepared = _validate_development(rows, config)
    selections, actions, metadata = _select(prepared, config)
    reference = next(iter(prepared.values()))["validation"]
    policies = {name: {"metadata": metadata[name], **_policy_metrics(reference, action)}
                for name, action in actions.items()}
    diagnostics = {}
    for item in items:
        name = item["config_id"]
        splits = {split: _split_diagnostics(frame, item["auxiliary_weight"])
                  for split, frame in prepared[name].items()}
        train, validation = splits["train"]["mean_seed"], splits["validation"]["mean_seed"]
        diagnostics[name] = {
            "configuration": deepcopy(item), "splits": splits,
            "validation_minus_train": {
                "state_nll": validation["state"]["nll"] - train["state"]["nll"],
                "state_brier": validation["state"]["brier"] - train["state"]["brier"],
                "fixed_zero_accuracy": splits["validation"]["fixed_zero_routing"]["mean_seed"]["accuracy"]
                - splits["train"]["fixed_zero_routing"]["mean_seed"]["accuracy"],
                **{f"{expert}_tcp_mse": validation["tcp"][expert]["tcp_regression"]["mse"]
                   - train["tcp"][expert]["tcp_regression"]["mse"] for expert in ("fusion", "geo")},
            },
        }
    intervals, draws = _bootstrap(reference, actions, config)
    report = {
        **_report_base(config, actions, selections, "validation"),
        "expert_family": "full_development_train_experts",
        "selection_and_reporting_use_same_validation_rows": True,
        "cleaned_test_access": False,
        "expert_references": {name: _expert_metrics(reference, source)
                              for name, source in (("fusion", "raw"), ("image", "image"), ("geo", "geo"))},
        "diagnostics": diagnostics, "validation_policies": policies,
        "validation_bootstrap": intervals,
    }
    return {"report": report, "selections": selections,
            "actions": _action_frame(reference, actions, labels=True),
            "bootstrap_draws": draws, "bootstrap_policy_order": list(actions)}


def make_test_report(rows: pd.DataFrame, config: dict, selections: dict,
                     comparators: dict | None = None) -> dict[str, Any]:
    """Score legacy-expert test predictions with frozen validation thresholds."""
    prepared = _validate(rows, config, labels=True)
    actions, metadata = _actions(prepared, config, selections)
    reference = next(iter(prepared.values()))
    historical = _add_comparators(reference, actions, metadata, comparators)
    policies = {name: {"metadata": metadata[name], **_policy_metrics(reference, action)}
                for name, action in actions.items()}
    for entry in policies.values():
        if entry["metadata"]["kind"] != "frozen_validation_threshold":
            continue
        counts, guard = entry["pooled"], entry["metadata"]["validation_guardrails"]
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
    intervals, draws = _bootstrap(reference, actions, config, historical)
    intervals.update(threshold_source="sealed_development_validation_selection", test_used_for_selection=False)
    report = {
        **_report_base(config, actions, selections, "test"),
        "expert_family": "historical_full_development_best_model_experts_september15_reference",
        "threshold_source": "sealed_development_validation_selection",
        "test_used_for_selection": False, "test_harm_caps_enforced": False,
        "validation_selections_sha256": canonical_sha256(selections),
        "historical_reference_count": len(historical), "historical_references": historical,
        "expert_references": {name: _expert_metrics(reference, source)
                              for name, source in (("fusion", "raw"), ("image", "image"), ("geo", "geo"))},
        "diagnostics": diagnostics, "test_policies": policies, "test_bootstrap": intervals,
    }
    return {"report": report, "actions": _action_frame(reference, actions, labels=True),
            "bootstrap_draws": draws, "bootstrap_policy_order": list(actions)}


def format_report(report: dict) -> str:
    """Render every configuration and policy identically after sorted JSON IO."""
    split = report["evaluation_split"]
    policies, bootstrap = report[f"{split}_policies"], report[f"{split}_bootstrap"]
    lines = ["# Joint TCP router capacity study", "",
             f"Study `{report['study_id']}`; evaluation split: **{split}**. "
             "Forty configurations cover five architectures, two feature sets, and four auxiliary weights. "
             "The 32 original [32] models are reused; 128 larger models are newly fitted. "
             "All outcomes are reported without selecting an architecture or auxiliary-weight winner.", "",
             "Accuracy and weighted F1 average four seeds. Routing counts pool seed-image realizations. "
             "Paired percentile intervals resample whole plots, preserving all images and seeds together. "
             "Models and thresholds remain fixed during resampling; these reused data provide exploratory "
             "evidence without multiplicity adjustment.", ""]
    if split == "test":
        lines += ["These are the legacy full-development best-model experts matching the September 15 report. "
                  "Development-validation thresholds transfer unchanged to this expert family. Test labels "
                  "do not retune thresholds, enforce guardrails, or select policies. Historical RF/LR "
                  "comparators use these same expert predictions.", ""]
    else:
        lines += ["Thresholds are selected and reported on development-validation. The scorer does not "
                  "access cleaned-test labels. Expert predictions are from full-development-train producers.", ""]
    lines += ["## Expert references", "", "| Expert | Accuracy (%) | Weighted F1 |",
              "| --- | ---: | ---: |"]
    for name in ("fusion", "image", "geo"):
        item = report["expert_references"][name]["mean_seed"]
        lines.append(f"| {name} | {100 * item['accuracy']:.4f} | {item['weighted_f1']:.5f} |")
    lines += ["", "## Every configuration", "",
              "The compact table uses minimum-20-acted-plots validation selections. The full policy table "
              "also includes minimum-10 and minimum-0 selections; their outcomes need not coincide.", "",
              "| Inputs | Architecture | Lambda | Fixed zero (%) | H <= 0.5R (%) | H <= 0.7R (%) | H <= 0.9R (%) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in report["configuration_order"]:
        item = report["diagnostics"][name]["configuration"]
        suffixes = ["zero"] + [f"plots_20__harm_0p{h}" for h in (5, 7, 9)]
        values = [100 * policies[f"{name}__{suffix}"]["mean_seed"]["accuracy"] for suffix in suffixes]
        lines.append(f"| {item['input_id']} | {item['hidden_widths']} | {item['auxiliary_weight']:g} | "
                     + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "## State and TCP diagnostics", "",
              "TCP outputs at lambda zero are unsupervised. Stable log-sum-exp from saved logits defines NLL. "
              "Full per-seed diagnostics include Brier score, failure AP/AUC, saturation and TCP ordering.", "",
              "| Configuration | Train NLL | Evaluation NLL | Brier | State accuracy | Macro F1 | Fusion TCP MSE | Geo TCP MSE |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in report["configuration_order"]:
        diagnostic = report["diagnostics"][name]
        if split == "validation":
            train = f"{diagnostic['splits']['train']['mean_seed']['state']['nll']:.5f}"
            item = diagnostic["splits"]["validation"]["mean_seed"]
        else:
            train, item = "—", diagnostic["test"]["mean_seed"]
        state = item["state"]
        tcp = [f"{item['tcp'][expert]['tcp_regression']['mse']:.5f}"
               if item["tcp"][expert]["supervised"] else "unsupervised" for expert in ("fusion", "geo")]
        lines.append(f"| {name} | {train} | {state['nll']:.5f} | {state['brier']:.5f} | "
                     f"{state['accuracy']:.5f} | {state['macro_f1']:.5f} | {tcp[0]} | {tcp[1]} |")
    lines += ["", "## Every policy", "",
              "| Policy | Accuracy (%) | Weighted F1 | Actions | Rescues | Harms | Net | Coverage (%) | Plots |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in report["policy_order"]:
        mean, count = policies[name]["mean_seed"], policies[name]["pooled"]
        lines.append(f"| {name} | {100 * mean['accuracy']:.4f} | {mean['weighted_f1']:.5f} | "
                     f"{count['actions']} | {count['rescues']} | {count['harms']} | {count['net']} | "
                     f"{100 * count['coverage']:.3f} | {count['plots_affected']} |")
    lines += ["", "## Every matched contrast", "",
              "Differences and 95% percentile intervals are percentage points. Auxiliary comparisons "
              "match architecture and inputs; capacity comparisons match inputs and auxiliary weight. "
              "Historical comparisons match inputs and the minimum-20/cap-0.7 policy identity. "
              "Accuracy contrasts subtract integer correct counts before division, preserving exact zeros.", "",
              "| Treatment minus reference | Kind | Accuracy difference [95% CI] | Weighted F1 difference [95% CI] |",
              "| --- | --- | ---: | ---: |"]
    for key in sorted(bootstrap["contrasts"]):
        item = bootstrap["contrasts"][key]
        values = [f"{100 * item[metric]['difference']:+.4f} "
                  f"[{100 * item[metric]['ci95_lower']:+.4f}, {100 * item[metric]['ci95_upper']:+.4f}]"
                  for metric in ("accuracy", "weighted_f1")]
        lines.append(f"| {item['treatment']} minus {item['reference']} | {item['kind']} | "
                     f"{values[0]} | {values[1]} |")
    lines += ["", f"All {len(bootstrap['contrasts'])} contrasts are retained. "
              "Identical policy masks or repeated guardrails do not constitute independent findings.", ""]
    return "\n".join(lines)

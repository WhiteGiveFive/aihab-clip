"""Pure diagnostics and paired evaluation for the joint TCP router study.

The caller authenticates sealed predictions and joins scoring labels. This
module reads no files, fits no parameters, and never chooses a model or lambda.
Native float32 predictions are promoted to float64 before metric arithmetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet_report import (
    _failure_detection, _probabilities, score_expert,
)
from multimodal.geo_helpfulness_relaxed_rf_selection import select_common_threshold
from multimodal.geo_helpfulness_router_heuristics import apply_heuristic_threshold
from multimodal.geo_helpfulness_router_metrics import classification_metrics


STATES = ("rescue", "harm", "both_correct", "both_wrong")
SEEDS = (1, 2, 3, 4)
Q_COLUMNS = tuple(f"q_{state}" for state in STATES)
IDENTITIES = ("row_uid", "plot_idx", "training_seed", "label_id_dense")


class JointTCPReportError(ValueError):
    """Predictions or configuration violate the declared scoring contract."""


def _mean_tree(values: list[Any]) -> Any:
    if isinstance(values[0], dict):
        return {key: _mean_tree([value[key] for value in values]) for key in values[0]}
    if any(value is None for value in values):
        return None
    if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
        return float(np.mean(values))
    return values[0]


def _validate(rows: pd.DataFrame, config: dict) -> dict[str, dict[str, pd.DataFrame]]:
    required = set(IDENTITIES) | set(Q_COLUMNS) | {
        "producer_fold", "split", "config_id", "variant", "with_embeddings",
        "auxiliary_weight", "raw_logits", "geo_logits", "image_logits",
        "fusion_tcp_hat", "geo_tcp_hat", "state_logits", "tcp_logits",
    }
    if (not isinstance(rows, pd.DataFrame) or rows.empty
            or rows.columns.duplicated().any() or not required.issubset(rows)):
        raise JointTCPReportError("joint prediction columns are incomplete")
    declarations = config["configs"]
    ids = [item["config_id"] for item in declarations]
    if len(ids) != len(set(ids)) or set(rows.config_id) != set(ids):
        raise JointTCPReportError("configuration identities differ from declaration")
    if config.get("training_seeds", list(SEEDS)) != list(SEEDS):
        raise JointTCPReportError("four training seeds 1, 2, 3, 4 are required")
    if set(rows.split) != {"train", "validation"}:
        raise JointTCPReportError("both train and validation predictions are required")
    if rows.duplicated(["config_id", "split", "training_seed", "row_uid"]).any():
        raise JointTCPReportError("duplicate prediction identity")
    if rows[list(IDENTITIES) + ["producer_fold"]].isna().any().any():
        raise JointTCPReportError("missing prediction identity")
    for name in ("training_seed", "label_id_dense"):
        if rows[name].dtype.kind not in "iu":
            raise JointTCPReportError(f"{name} must be integral")
    if not rows.label_id_dense.between(0, 17).all():
        raise JointTCPReportError("labels must use the fixed 18-class ontology")
    prepared: dict[str, dict[str, pd.DataFrame]] = {}
    references: dict[str, pd.DataFrame] = {}
    for item in declarations:
        name = item["config_id"]
        prepared[name] = {}
        source = rows.loc[rows.config_id.eq(name)]
        for key in ("variant", "with_embeddings", "auxiliary_weight"):
            if not source[key].eq(item[key]).all():
                raise JointTCPReportError(f"configuration metadata differs: {name}/{key}")
        for split in ("train", "validation"):
            frame = source.loc[source.split.eq(split)].sort_values(
                ["row_uid", "training_seed"], kind="stable"
            ).reset_index(drop=True).copy()
            if set(frame.training_seed) != set(SEEDS):
                raise JointTCPReportError("a split lacks a training seed")
            seed_identity = None
            for seed in SEEDS:
                identity = frame.loc[frame.training_seed.eq(seed), [
                    "row_uid", "plot_idx", "label_id_dense"
                ]].reset_index(drop=True)
                if seed_identity is not None and not identity.equals(seed_identity):
                    raise JointTCPReportError("row/plot/label identities differ across seeds")
                seed_identity = identity
            for expert in ("image", "raw", "geo"):
                logits = np.stack(frame[f"{expert}_logits"])
                if logits.shape != (len(frame), 18) or not np.isfinite(logits).all():
                    raise JointTCPReportError("expert logits must be finite (n,18)")
                frame[f"{expert}_pred"] = logits.argmax(1)
            q = frame[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
            if (not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()
                    or not np.allclose(q.sum(1), 1, atol=1e-6, rtol=0)):
                raise JointTCPReportError("router probabilities must form finite simplex rows")
            estimate = frame[["fusion_tcp_hat", "geo_tcp_hat"]].to_numpy(dtype=np.float64)
            if not np.isfinite(estimate).all() or (estimate < 0).any() or (estimate > 1).any():
                raise JointTCPReportError("TCP estimates must be finite within [0,1]")
            state_logits = np.stack(frame.state_logits).astype(np.float64)
            tcp_logits = np.stack(frame.tcp_logits).astype(np.float64)
            if (state_logits.shape != (len(frame), 4) or tcp_logits.shape != (len(frame), 2)
                    or not np.isfinite(state_logits).all() or not np.isfinite(tcp_logits).all()):
                raise JointTCPReportError("state/TCP logits must be finite (n,4)/(n,2)")
            exponent = np.exp(-np.abs(tcp_logits))
            tcp_from_logits = np.where(tcp_logits >= 0, 1 / (1 + exponent), exponent / (1 + exponent))
            if (not np.allclose(_probabilities(state_logits), q, atol=1e-6, rtol=0)
                    or not np.allclose(tcp_from_logits, estimate, atol=1e-6, rtol=0)):
                raise JointTCPReportError("native probabilities disagree with saved logits")
            raw_ok = frame.raw_pred.eq(frame.label_id_dense)
            geo_ok = frame.geo_pred.eq(frame.label_id_dense)
            frame["target_state"] = np.select(
                [~raw_ok & geo_ok, raw_ok & ~geo_ok, raw_ok & geo_ok],
                STATES[:3], default=STATES[3],
            )
            # Preserve the native output values; score arithmetic is float64.
            frame["score"] = q[:, 0] - q[:, 1]
            same_columns = list(IDENTITIES) + ["producer_fold", "target_state"]
            if split in references:
                reference = references[split]
                if not frame[same_columns].equals(reference[same_columns]):
                    raise JointTCPReportError("targets or producers differ across configurations")
                for expert in ("image", "raw", "geo"):
                    if not np.array_equal(np.stack(frame[f"{expert}_logits"]),
                                          np.stack(reference[f"{expert}_logits"])):
                        raise JointTCPReportError("expert logits differ across configurations")
            else:
                references[split] = frame
            prepared[name][split] = frame
        train_plots = set(prepared[name]["train"].plot_idx)
        if train_plots.intersection(prepared[name]["validation"].plot_idx):
            raise JointTCPReportError("development train and validation plots overlap")
        if set(prepared[name]["train"].row_uid).intersection(prepared[name]["validation"].row_uid):
            raise JointTCPReportError("development train and validation images overlap")
    return prepared


def _ranking(target: np.ndarray, scores: np.ndarray) -> dict:
    if not len(target):
        return {"average_precision": None, "roc_auc": None,
                "undefined_reason": "no_disagreements"}
    return _failure_detection(target, np.asarray(scores, dtype=np.float64))


def _ordering(frame: pd.DataFrame) -> dict:
    labels = frame.label_id_dense.to_numpy(dtype=np.int64)
    index = np.arange(len(frame))
    true_geo = _probabilities(np.stack(frame.geo_logits))[index, labels]
    true_raw = _probabilities(np.stack(frame.raw_logits))[index, labels]
    estimated_delta = (frame.geo_tcp_hat.to_numpy(dtype=np.float64)
                       - frame.fusion_tcp_hat.to_numpy(dtype=np.float64))
    result = {}
    for state, sign in (("rescue", 1), ("harm", -1)):
        mask = frame.target_state.eq(state).to_numpy()
        count = int(mask.sum())
        true_delta = sign * (true_geo[mask] - true_raw[mask])
        delta = sign * estimated_delta[mask]
        result[state] = {
            "count": count,
            "true_tcp_correct_order_count": int(np.sum(true_delta > 0)),
            "estimated_tcp_correct_order_count": int(np.sum(delta > 0)),
            "true_tcp_tie_count": int(np.sum(true_delta == 0)),
            "estimated_tcp_tie_count": int(np.sum(delta == 0)),
            "true_tcp_correct_order_fraction": float(np.mean(true_delta > 0)) if count else None,
            "estimated_tcp_correct_order_fraction": float(np.mean(delta > 0)) if count else None,
        }
    return result


def _diagnostics(frame: pd.DataFrame, auxiliary_weight: float) -> dict:
    q = frame[list(Q_COLUMNS)].to_numpy(dtype=np.float64)
    state = pd.Categorical(frame.target_state, categories=STATES).codes.astype(np.int64)
    target_q = q[np.arange(len(q)), state]
    logits = np.stack(frame.state_logits).astype(np.float64)
    centered = logits - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(centered).sum(axis=1)) - centered[np.arange(len(q)), state]
    classification = classification_metrics(q.argmax(1), state, n_classes=4)
    disagreement = frame.raw_pred.ne(frame.geo_pred).to_numpy()
    labels = frame.label_id_dense.to_numpy(dtype=np.int64)
    experts = {}
    for name, source in (("fusion", "raw"), ("geo", "geo")):
        estimate = frame[f"{name}_tcp_hat"].to_numpy(dtype=np.float64)
        metric = score_expert(labels, np.stack(frame[f"{source}_logits"]), estimate)
        metric.update({
            "supervised": auxiliary_weight > 0,
            "predicted_tcp_fraction_le_1e-6": float(np.mean(estimate <= 1e-6)),
            "predicted_tcp_fraction_ge_1_minus_1e-6": float(np.mean(estimate >= 1 - 1e-6)),
            "predicted_tcp_fraction_le_0p01": float(np.mean(estimate <= .01)),
            "predicted_tcp_fraction_ge_0p99": float(np.mean(estimate >= .99)),
            "mean_predicted_tcp": float(estimate.mean()),
        })
        experts[name] = metric
    ranking = {}
    for name, index, sign in (("rescue", 0, 1), ("harm", 1, -1)):
        target = state[disagreement] == index
        ranking[name] = {
            "state_probability": _ranking(target, q[disagreement, index]),
            "signed_utility": _ranking(target, sign * frame.score.to_numpy()[disagreement]),
        }
    return {
        "row_count": len(frame), "plot_count": int(frame.plot_idx.nunique()),
        "four_state_counts": {name: int(np.sum(state == index)) for index, name in enumerate(STATES)},
        "state": {
            "nll": float(nll.mean()),
            "probability_nll": float(-np.log(np.maximum(target_q, np.finfo(np.float64).tiny)).mean()),
            "nll_floor_count": int(np.sum(target_q < np.finfo(np.float64).tiny)),
            "brier": float(np.sum((q - np.eye(4)[state]) ** 2, axis=1).mean()),
            "accuracy": classification["accuracy"], "macro_f1": classification["macro_f1"],
            "disagreement_count": int(disagreement.sum()),
            "ranking_on_disagreements": ranking,
        },
        "tcp": experts, "tcp_ordering": _ordering(frame),
    }


def _split_diagnostics(frame: pd.DataFrame, weight: float) -> dict:
    by_seed = {str(seed): _diagnostics(frame.loc[frame.training_seed.eq(seed)], weight)
               for seed in SEEDS}
    return {"per_seed": by_seed, "mean_seed": _mean_tree(list(by_seed.values())),
            "pooled": _diagnostics(frame, weight),
            "fixed_zero_routing": _policy_metrics(
                frame, frame.raw_pred.ne(frame.geo_pred).to_numpy() & (frame.score.to_numpy() > 0)
            ),
            "fusion_reference": _policy_metrics(frame, np.zeros(len(frame), dtype=bool))}


def _counts(frame: pd.DataFrame, action: np.ndarray) -> dict:
    labels = frame.label_id_dense.to_numpy(dtype=np.int64)
    predictions = np.where(action, frame.geo_pred, frame.raw_pred)
    metric = classification_metrics(predictions, labels)
    fusion = classification_metrics(frame.raw_pred.to_numpy(), labels)
    rescue = int(np.sum(action & frame.target_state.eq("rescue").to_numpy()))
    harm = int(np.sum(action & frame.target_state.eq("harm").to_numpy()))
    neutral = int(np.sum(action & frame.target_state.eq("both_wrong").to_numpy()))
    if rescue + harm + neutral != int(action.sum()):
        raise JointTCPReportError("intervention accounting does not reconcile")
    return {
        "row_count": len(frame), "plot_count": int(frame.plot_idx.nunique()),
        "actions": int(action.sum()), "rescues": rescue, "harms": harm,
        "neutral_wrong": neutral, "net": rescue - harm,
        "accuracy": metric["accuracy"], "weighted_f1": metric["weighted_f1"],
        "accuracy_gain_vs_fusion": (rescue - harm) / len(frame),
        "weighted_f1_gain_vs_fusion": metric["weighted_f1"] - fusion["weighted_f1"],
        "coverage": float(action.mean()), "plots_affected": int(frame.loc[action, "plot_idx"].nunique()),
    }


def _policy_metrics(frame: pd.DataFrame, action: np.ndarray) -> dict:
    by_seed = {str(seed): _counts(frame.loc[frame.training_seed.eq(seed)],
                                  action[frame.training_seed.eq(seed).to_numpy()])
               for seed in SEEDS}
    return {"per_seed": by_seed, "mean_seed": _mean_tree(list(by_seed.values())),
            "pooled": _counts(frame, action)}


def _select(prepared: dict, config: dict) -> tuple[dict, dict, dict]:
    first = next(iter(prepared.values()))["validation"]
    actions = {"fusion": np.zeros(len(first), dtype=bool)}
    metadata = {"fusion": {"kind": "reference"}}
    selections = {}
    for item in config["configs"]:
        name = item["config_id"]
        frame = prepared[name]["validation"]
        values = frame.score.to_numpy(dtype=np.float64)
        zero_name = f"{name}__zero"
        actions[zero_name] = frame.raw_pred.ne(frame.geo_pred).to_numpy() & (values > 0)
        metadata[zero_name] = {"kind": "fixed_zero", "config_id": name}
        for plots in config.get("minimum_plots_grid", [20, 10, 0]):
            for numerator in config.get("harm_numerator_grid", [5, 7, 9]):
                guardrail = f"plots_{plots}__harm_0p{numerator}"
                policy = f"{name}__{guardrail}"
                selected = select_common_threshold(frame, minimum_plots=plots, harm_numerator=numerator)
                selections[policy] = {"config_id": name, **selected}
                actions[policy] = apply_heuristic_threshold(
                    values, frame.raw_pred.to_numpy(), frame.geo_pred.to_numpy(),
                    selected["threshold_specification"],
                )
                metadata[policy] = {"kind": "relaxed", "config_id": name, "guardrail_id": guardrail}
    return selections, actions, metadata


def _contrast_pairs(config: dict, names: list[str]) -> list[tuple[str, str, str]]:
    pairs = [(name, "fusion", "versus_fusion") for name in names if name != "fusion"]
    items = config["configs"]
    suffixes = ["zero"] + [
        f"plots_{p}__harm_0p{h}" for p in config.get("minimum_plots_grid", [20, 10, 0])
        for h in config.get("harm_numerator_grid", [5, 7, 9])
    ]
    for item in items:
        comparisons = []
        if item["auxiliary_weight"] > 0:
            controls = [other for other in items if other["variant"] == item["variant"]
                        and other["with_embeddings"] == item["with_embeddings"]
                        and other["auxiliary_weight"] == 0]
            if len(controls) != 1:
                raise JointTCPReportError("each auxiliary arm requires exactly one CE-only control")
            comparisons.append((controls[0], "auxiliary_weight_versus_zero"))
        if item["with_embeddings"]:
            controls = [other for other in items if other["variant"] == item["variant"]
                        and not other["with_embeddings"]
                        and other["auxiliary_weight"] == item["auxiliary_weight"]]
            if len(controls) != 1:
                raise JointTCPReportError("each embedding arm requires exactly one matching no-embedding control")
            comparisons.append((controls[0], "embeddings_versus_no_embeddings"))
        for other, kind in comparisons:
            for suffix in suffixes:
                pairs.append((f"{item['config_id']}__{suffix}",
                              f"{other['config_id']}__{suffix}", kind))
    return pairs


def _bootstrap(frame: pd.DataFrame, actions: dict, config: dict) -> tuple[dict, np.ndarray]:
    """Share exact whole-plot resamples, collapsing identical action masks."""
    names = list(actions)
    plots, inverse = np.unique(frame.plot_idx, return_inverse=True)
    settings = config.get("bootstrap", {"resamples": 2000, "seed": 20260826})
    if type(settings["resamples"]) is not int or settings["resamples"] < 1:
        raise JointTCPReportError("bootstrap resamples must be a positive integer")
    rng = np.random.default_rng(settings["seed"])
    weights = np.empty((settings["resamples"], len(plots)), dtype=np.float64)
    for index in range(len(weights)):
        draw = rng.integers(0, len(plots), size=len(plots))
        weights[index] = np.bincount(draw, minlength=len(plots))
    # Identical policies are frequent when guardrails return never-intervene.
    unique: dict[bytes, int] = {}
    unique_actions, mapping = [], []
    for action in actions.values():
        key = np.packbits(action).tobytes()
        if key not in unique:
            unique[key] = len(unique_actions)
            unique_actions.append(action)
        mapping.append(unique[key])
    labels = frame.label_id_dense.to_numpy(dtype=np.int64)
    draws = np.empty((len(weights), len(unique_actions), 2), dtype=np.float64)
    observed = np.empty((len(unique_actions), 2), dtype=np.float64)

    def metrics(counts: np.ndarray) -> np.ndarray:
        truth, predicted, tp = counts[..., 2:20], counts[..., 20:38], counts[..., 38:56]
        f1 = np.divide(2 * tp, truth + predicted, out=np.zeros_like(tp), where=truth + predicted != 0)
        return np.stack((counts[..., 1] / counts[..., 0],
                         (truth * f1).sum(-1) / counts[..., 0]), axis=-1).mean(axis=-2)

    for policy_index, action in enumerate(unique_actions):
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
        # Keep memory bounded independently of the number of policies.
        counts = (weights @ group.reshape(len(plots), -1)).reshape(len(weights), len(SEEDS), 56)
        draws[:, policy_index] = metrics(counts)
        observed[policy_index] = metrics(group.sum(axis=0))
    draws, observed = draws[:, mapping], observed[mapping]
    contrasts = {}
    lookup = {name: index for index, name in enumerate(names)}
    for treatment, reference, kind in _contrast_pairs(config, names):
        left, right = lookup[treatment], lookup[reference]
        delta = draws[:, left] - draws[:, right]
        interval = np.quantile(delta, [.025, .975], axis=0)
        contrasts[f"{treatment}_minus_{reference}"] = {
            "treatment": treatment, "reference": reference, "kind": kind,
            **{metric: {"difference": float(observed[left, i] - observed[right, i]),
                        "ci95_lower": float(interval[0, i]), "ci95_upper": float(interval[1, i])}
               for i, metric in enumerate(("accuracy", "weighted_f1"))},
        }
    return {**settings, "plot_count": len(plots), "policy_order": names,
            "metric_order": ["accuracy", "weighted_f1"],
            "unique_action_masks": len(unique_actions), "contrasts": contrasts,
            "unit": "whole_plot_all_images_all_seed_realizations",
            "thresholds_refitted_within_bootstrap": False,
            "independent_confirmation": False}, draws


def make_report(rows: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Score sealed long-form train/validation predictions without file access.

    Returns a JSON-safe report and selections, a wide validation action table,
    and float64 bootstrap draws shaped (resamples, policies, two metrics).
    """
    prepared = _validate(rows, config)
    selections, actions, metadata = _select(prepared, config)
    reference = next(iter(prepared.values()))["validation"]
    policies = {name: {"metadata": metadata[name], **_policy_metrics(reference, action)}
                for name, action in actions.items()}
    diagnostics = {}
    for item in config["configs"]:
        name = item["config_id"]
        by_split = {split: _split_diagnostics(frame, item["auxiliary_weight"])
                    for split, frame in prepared[name].items()}
        train = by_split["train"]["mean_seed"]
        validation = by_split["validation"]["mean_seed"]
        diagnostics[name] = {
            "configuration": dict(item), "splits": by_split,
            "validation_minus_train": {
                "state_nll": validation["state"]["nll"] - train["state"]["nll"],
                "state_brier": validation["state"]["brier"] - train["state"]["brier"],
                "fixed_zero_accuracy": by_split["validation"]["fixed_zero_routing"]["mean_seed"]["accuracy"]
                - by_split["train"]["fixed_zero_routing"]["mean_seed"]["accuracy"],
                **{f"{expert}_tcp_mse": validation["tcp"][expert]["tcp_regression"]["mse"]
                   - train["tcp"][expert]["tcp_regression"]["mse"] for expert in ("fusion", "geo")},
            },
        }
    intervals, draws = _bootstrap(reference, actions, config)
    report = {
        "schema_version": "geo_helpfulness.joint_tcp_report.v1",
        "study_id": config.get("study_id", "joint_tcp_router_v1"),
        "training_seeds": list(SEEDS), "four_state_order": list(STATES),
        "configuration_count": len(config["configs"]), "model_count": len(config["configs"]) * len(SEEDS),
        "selected_policy_count": len(selections), "policy_count_including_references": len(actions),
        "native_predictions_dtype": "float32", "metric_arithmetic_dtype": "float64",
        "score_convention": "float64(q_rescue)-float64(q_harm)",
        "nll_convention": "stable_float64_logsumexp_from_saved_state_logits",
        "probability_nll_convention": "negative_log_native_float32_q_with_float64_tiny_floor",
        "tcp_target_convention": "native_temperature_one_softmax_authoritative_expert_logits_float64_scoring",
        "lambda_winner_selected": False, "independent_confirmation": False,
        "selection_and_reporting_use_same_validation_rows": True,
        "cleaned_test_access": False, "fitting_performed_by_scorer": False,
        "diagnostics": diagnostics, "validation_policies": policies,
        "validation_bootstrap": intervals,
    }
    action_frame = reference[list(IDENTITIES) + ["raw_pred", "geo_pred", "target_state"]].copy()
    action_frame = pd.concat([action_frame, pd.DataFrame(actions)], axis=1)
    return {"report": report, "selections": selections, "actions": action_frame,
            "bootstrap_draws": draws, "bootstrap_policy_order": list(actions)}


def format_report(report: dict) -> str:
    """Render every declared arm and guardrail, without promoting a winner."""
    lines = [
        "# Joint four-state router with auxiliary TCP supervision", "",
        f"Study: `{report['study_id']}`. {report['model_count']} fits across four seeds. "
        "All auxiliary weights and guardrails are reported; no winning lambda is selected.", "",
        "Development-validation selects thresholds and supplies these exploratory results. "
        "Paired confidence intervals resample whole plots with all seed realizations together, "
        "holding selected thresholds fixed; they are not independent confirmation. "
        "The cleaned test set was not accessed.", "",
        "## State and confidence diagnostics", "",
        "TCP outputs at lambda zero are unsupervised and are not trained confidence estimates.", "",
        "NLL uses stable float64 log-sum-exp of saved state logits. The report also retains "
        "probability-based NLL and its floor counts to expose native float32 probability underflow.", "",
        "| Configuration | Train NLL | Validation NLL | Validation Brier | State accuracy | Macro F1 | Fusion TCP MSE | Geo TCP MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in report["diagnostics"].items():
        train, val = item["splits"]["train"]["mean_seed"], item["splits"]["validation"]["mean_seed"]
        lines.append(f"| {name} | {train['state']['nll']:.4f} | {val['state']['nll']:.4f} | "
                     f"{val['state']['brier']:.4f} | {val['state']['accuracy']:.4f} | "
                     f"{val['state']['macro_f1']:.4f} | {val['tcp']['fusion']['tcp_regression']['mse']:.5f} | "
                     f"{val['tcp']['geo']['tcp_regression']['mse']:.5f} |")
    lines += ["", "## Routing on development-validation", "",
              "Accuracy and weighted F1 are arithmetic means over seeds. Actions, rescues, harms, "
              "and net corrections are pooled seed-image counts, not independent images. "
              "Affected plots count unique plots across seeds.", "",
              "| Policy | Accuracy (%) | Weighted F1 | Actions | Rescues | Harms | Net | Coverage (%) | Plots |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, entry in report["validation_policies"].items():
        mean, pooled = entry["mean_seed"], entry["pooled"]
        lines.append(f"| {name} | {100 * mean['accuracy']:.4f} | {mean['weighted_f1']:.5f} | "
                     f"{pooled['actions']} | {pooled['rescues']} | {pooled['harms']} | {pooled['net']} | "
                     f"{100 * pooled['coverage']:.3f} | {pooled['plots_affected']} |")
    if "rf_context" in report:
        context = report["rf_context"]
        lines += ["", "## Existing matched-expert RF context", "",
                  "These previously fitted RF policies use the same development expert predictions. "
                  "They are context, rather than the matched CE-only controls for auxiliary learning. "
                  "Counts are pooled over four seeds; accuracy gains are percentage points over fusion.", "",
                  "| Existing RF policy | Accuracy gain (pp) | Actions | Rescues | Harms | Net | Coverage (%) | Plots |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, entry in context["policies"].items():
            threshold = entry["threshold"]
            pooled = threshold["pooled"]
            lines.append(f"| {name} | {100 * threshold['mean_seed_delta_accuracy']:+.4f} | "
                         f"{pooled['acted_count']} | {pooled['rescued_count']} | {pooled['harmed_count']} | "
                         f"{pooled['rescued_minus_harmed']} | {100 * pooled['coverage_fraction']:.3f} | "
                         f"{pooled['unique_acted_plots']} |")
    lines += ["", "## Matched contrasts", "",
              "Differences and 95% percentile intervals are in percentage points. "
              "The same guardrail setting is compared between each pair, with thresholds fitted separately. "
              "No correction for the multiple exploratory comparisons is applied.", "",
              "| Treatment minus reference | Accuracy difference [95% CI] | Weighted F1 difference [95% CI] |",
              "| --- | ---: | ---: |"]
    for entry in report["validation_bootstrap"]["contrasts"].values():
        if entry["kind"] == "versus_fusion":
            continue
        rendered = []
        for metric in ("accuracy", "weighted_f1"):
            value = entry[metric]
            rendered.append(f"{100 * value['difference']:+.4f} [{100 * value['ci95_lower']:+.4f}, {100 * value['ci95_upper']:+.4f}]")
        lines.append(f"| {entry['treatment']} minus {entry['reference']} | {rendered[0]} | {rendered[1]} |")
    lines += ["", "Per-seed and pooled state metrics, failure-detection AP/AUC, TCP saturation, "
              "true/estimated ordering, full threshold curves, and all fusion-reference intervals "
              "are provided in the machine-readable report and selection artifacts.", ""]
    return "\n".join(lines)

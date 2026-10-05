#!/usr/bin/env python3
"""Apply the established relaxed RF selector to frozen ConfidNet router outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multimodal.geo_helpfulness_relaxed_rf_selection import (  # noqa: E402
    select_common_threshold,
)
from multimodal.geo_helpfulness_router_heuristics import (  # noqa: E402
    apply_heuristic_threshold,
)

BASE = ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
PARENT = BASE / "confidnet_oof_v1_replay_v2"
OUTPUT = BASE / "confidnet_oof_relaxed_guardrails_v1"
CONFIG = ROOT / "configs/multimodal_geo_confidnet_relaxed_guardrails_v1.json"
ARMS = ("baseline", "augmented")
STATES = ("rescue", "harm", "both_correct", "both_wrong")
SEEDS = (1, 2, 3, 4)
SOURCE_FILES = (
    "tools/analyze_confidnet_relaxed_guardrails.py",
    "configs/multimodal_geo_confidnet_relaxed_guardrails_v1.json",
    "multimodal/geo_helpfulness_relaxed_rf_selection.py",
    "multimodal/geo_helpfulness_relaxed_guardrails_selection.py",
    "multimodal/geo_helpfulness_router_selection.py",
    "multimodal/geo_helpfulness_router_heuristics.py",
    "multimodal/geo_helpfulness_router_candidate_state.py",
    "multimodal/geo_helpfulness_router_metrics.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def snapshot(root):
    return {
        str(path.relative_to(root)): {
            "sha256": sha256(path), "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def authenticate_parent():
    """Check every saved stage against its existing manifest before decoding rows."""
    count = 0
    for path in PARENT.rglob("manifest.json"):
        for name, expected in read_json(path)["file_sha256"].items():
            if sha256(path.parent / name) != expected:
                raise ValueError(f"Parent artifact changed: {path.parent / name}")
            count += 1
    return count


def load_rows():
    columns = ["row_uid", "plot_idx", "training_seed", "split", "development_role",
               "label_id_dense", "raw_logits", "geo_logits"]
    columns += [f"router_{arm}_q_{state}" for arm in ARMS for state in STATES]
    rows = pd.read_parquet(PARENT / "scores/scored_rows.parquet", columns=columns,
                           filters=[("split", "==", "validation")])
    if len(rows) != 3288 or set(rows.training_seed) != set(SEEDS):
        raise ValueError("Expected four complete 822-image validation realizations")
    if not rows.development_role.eq("validation").all():
        raise ValueError("Selection contains non-validation rows")
    reference = None
    for seed in SEEDS:
        seed_rows = rows.loc[rows.training_seed.eq(seed)]
        if len(seed_rows) != 822 or seed_rows.row_uid.nunique() != 822 or seed_rows.plot_idx.nunique() != 325:
            raise ValueError("Validation identities/support do not match the parent")
        keys = seed_rows[["row_uid", "plot_idx", "label_id_dense"]].sort_values("row_uid").reset_index(drop=True)
        if reference is not None and not keys.equals(reference):
            raise ValueError("Seed row, plot or label identities differ")
        reference = keys
        with np.load(PARENT / f"features/seed_{seed}/validation.npz", allow_pickle=False) as features:
            if not np.array_equal(seed_rows.row_uid.to_numpy(), features["row_uid"]):
                raise ValueError("Prediction row order changed")
            for expert in ("raw", "geo"):
                if not np.array_equal(np.stack(seed_rows[f"{expert}_logits"]), features[f"{expert}_logits"]):
                    raise ValueError("Scored logits differ from sealed expert predictions")
        for arm in ARMS:
            q = seed_rows[[f"router_{arm}_q_{state}" for state in STATES]].to_numpy()
            with np.load(PARENT / f"routers/seed_{seed}/{arm}/predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(q, saved["validation"]):
                    raise ValueError("Scored router probabilities differ from saved predictions")
            if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(1), 1, atol=1e-12, rtol=0):
                raise ValueError("Invalid router probabilities")
    rows["raw_pred"] = np.stack(rows.raw_logits).argmax(1)
    rows["geo_pred"] = np.stack(rows.geo_logits).argmax(1)
    raw_ok = rows.raw_pred.eq(rows.label_id_dense)
    geo_ok = rows.geo_pred.eq(rows.label_id_dense)
    rows["target_state"] = np.select(
        [~raw_ok & geo_ok, raw_ok & ~geo_ok, raw_ok & geo_ok],
        STATES[:3], default="both_wrong",
    )
    rows = rows.drop(columns=["raw_logits", "geo_logits", "split", "development_role"])
    return rows.sort_values(["row_uid", "training_seed"], kind="stable").reset_index(drop=True)


def select_policies(rows, config):
    actions = {"fusion": np.zeros(len(rows), dtype=bool)}
    selections = {}
    disagreement = rows.raw_pred.ne(rows.geo_pred).to_numpy()
    for arm in ARMS:
        values = rows[f"router_{arm}_q_rescue"] - rows[f"router_{arm}_q_harm"]
        actions[f"{arm}_zero"] = disagreement & values.gt(0).to_numpy()
        inputs = rows.assign(score=values)
        for plots in config["minimum_plots_grid"]:
            for numerator in config["harm_numerator_grid"]:
                name = f"{arm}__plots_{plots}__harm_0p{numerator}"
                selected = select_common_threshold(inputs, minimum_plots=plots, harm_numerator=numerator)
                selections[name] = {"arm": arm, **selected}
                actions[name] = apply_heuristic_threshold(
                    values.to_numpy(), rows.raw_pred.to_numpy(), rows.geo_pred.to_numpy(),
                    selected["threshold_specification"],
                )
    return selections, actions


def classification(labels, predictions):
    cm = np.bincount(18 * labels + predictions, minlength=18 * 18).reshape(18, 18)
    support, predicted_support = cm.sum(1), cm.sum(0)
    f1 = np.divide(2 * np.diag(cm), support + predicted_support,
                   out=np.zeros(18, dtype=float), where=(support + predicted_support) != 0)
    return {"accuracy": float(np.trace(cm) / len(labels)),
            "weighted_f1": float(np.dot(support, f1) / len(labels))}


def policy_metrics(rows, action):
    predicted = np.where(action, rows.geo_pred, rows.raw_pred)
    labels = rows.label_id_dense.to_numpy(dtype=np.int64)
    state = rows.target_state.to_numpy()

    def counts(mask):
        active = action & mask
        rescue = int(np.sum(active & (state == "rescue")))
        harm = int(np.sum(active & (state == "harm")))
        return {
            "rows": int(mask.sum()), "actions": int(active.sum()), "rescues": rescue,
            "harms": harm, "net": rescue - harm,
            "neutral_wrong": int(np.sum(active & (state == "both_wrong"))),
            "plots_affected": int(rows.loc[active, "plot_idx"].nunique()),
            "coverage": float(active.sum() / mask.sum()),
            **classification(labels[mask], predicted[mask]),
        }

    per_seed = {str(s): counts(rows.training_seed.eq(s).to_numpy()) for s in SEEDS}
    pooled = counts(np.ones(len(rows), dtype=bool))
    mean = {key: float(np.mean([item[key] for item in per_seed.values()])) for key in pooled}
    contribution = action * ((state == "rescue").astype(int) - (state == "harm").astype(int))
    plot_net = pd.Series(contribution).groupby(rows.plot_idx).sum()
    best = plot_net.sort_values(ascending=False, kind="stable")
    best_plot, best_net = (str(best.index[0]), int(best.iloc[0])) if best.iloc[0] > 0 else (None, 0)
    return {"pooled": pooled, "per_seed": per_seed, "mean_seed": mean,
            "largest_positive_plot": best_plot, "largest_plot_net": best_net,
            "net_without_largest_positive_plot": pooled["net"] - best_net}


def bootstrap(rows, actions, config):
    """Use one shared whole-plot resample for every policy and seed realization."""
    names = list(actions)
    plots, inverse = np.unique(rows.plot_idx, return_inverse=True)
    # Sufficient statistics: rows, correct, then true/predicted/TP class supports.
    groups = np.zeros((len(plots), 4, len(names), 56), dtype=np.float64)
    labels = rows.label_id_dense.to_numpy(dtype=np.int64)
    for s_index, seed in enumerate(SEEDS):
        mask = rows.training_seed.eq(seed).to_numpy()
        for p_index, action in enumerate(actions.values()):
            pred = np.where(action, rows.geo_pred, rows.raw_pred)
            columns = [np.ones(len(rows)), pred == labels]
            columns += [labels == c for c in range(18)]
            columns += [pred == c for c in range(18)]
            columns += [(pred == c) & (labels == c) for c in range(18)]
            for index, column in enumerate(columns):
                groups[:, s_index, p_index, index] = np.bincount(
                    inverse[mask], weights=column[mask], minlength=len(plots))

    def metrics(counts):
        truth, predicted, tp = counts[..., 2:20], counts[..., 20:38], counts[..., 38:56]
        f1 = np.divide(2 * tp, truth + predicted, out=np.zeros_like(tp), where=(truth + predicted) != 0)
        return np.stack([counts[..., 1] / counts[..., 0],
                         (truth * f1).sum(-1) / counts[..., 0]], axis=-1).mean(0)

    settings = config["bootstrap"]
    rng = np.random.default_rng(settings["seed"])
    draws = np.empty((settings["resamples"], len(names), 2))
    for b in range(len(draws)):
        indices = rng.integers(0, len(plots), size=len(plots))
        draws[b] = metrics(groups[indices].sum(0))
    observed = metrics(groups.sum(0))
    pairs = [(name, "fusion") for name in names if name != "fusion"]
    pairs += [("augmented_zero", "baseline_zero")]
    for name in names:
        if "__" in name:
            pairs.append((name, name.split("__")[0] + "_zero"))
        if name.startswith("augmented__"):
            pairs.append((name, name.replace("augmented__", "baseline__", 1)))
    contrasts = {}
    for treatment, reference in pairs:
        a, b = names.index(treatment), names.index(reference)
        delta = draws[:, a] - draws[:, b]
        interval = np.quantile(delta, [0.025, 0.975], axis=0)
        contrasts[f"{treatment}_minus_{reference}"] = {
            metric: {"difference": float(observed[a, k] - observed[b, k]),
                     "ci95_lower": float(interval[0, k]), "ci95_upper": float(interval[1, k])}
            for k, metric in enumerate(("accuracy", "weighted_f1"))
        }
    return {**settings, "plot_count": len(plots), "policy_order": names,
            "metric_order": ["accuracy", "weighted_f1"], "contrasts": contrasts}, draws


def make_report(rows, selections, actions, config):
    metrics = {name: policy_metrics(rows, action) for name, action in actions.items()}
    previous = read_json(PARENT / "scores/report.json")
    for name, old in (("fusion", "fusion"), ("baseline_zero", "rf_baseline"), ("augmented_zero", "rf_augmented")):
        for key in ("accuracy", "weighted_f1", "actions", "rescues", "harms", "net"):
            expected = previous["splits"]["validation"]["mean_seed"]["policies"][old][key]
            if not np.isclose(metrics[name]["mean_seed"][key], expected, atol=1e-14, rtol=0):
                raise ValueError(f"Fixed-zero parent result does not reproduce: {name}/{key}")
    intervals, draws = bootstrap(rows, actions, config)
    # The parent uses these same draws for the zero-policy paired comparison.
    new = intervals["contrasts"]["augmented_zero_minus_baseline_zero"]
    old = previous["validation_bootstrap"]["contrasts"]["rf_augmented_minus_rf_baseline"]
    for metric in new:
        for key in new[metric]:
            if not np.isclose(new[metric][key], old[metric][key], atol=1e-14, rtol=0):
                raise ValueError("Parent paired bootstrap does not reproduce")
    return {
        "study_id": config["study_id"], "split": "development_validation",
        "selection_and_reporting_use_same_rows": True, "independent_confirmation": False,
        "fitting_performed": False, "cleaned_test_access": False,
        "parent_zero_policy_and_bootstrap_parity": True,
        "policies": metrics, "bootstrap": intervals,
        "selections": {name: {key: value for key, value in selected.items() if key != "candidate_curve"}
                       for name, selected in selections.items()},
    }, draws


def format_report(report):
    lines = ["# ConfidNet router: relaxed threshold sensitivity", "",
             "Development-validation selection and reporting on the same 822 images / 325 plots, seeds 1–4.",
             "Conditional intervals hold selected thresholds fixed and do not adjust for selection or validation reuse.", "",
             "Accuracies/F1 are seed means; action counts pool 3,288 seed-image observations.", "",
             "| Policy | Threshold | Accuracy (%) | F1 (%) | Actions | Rescues | Harms | Net | Plots |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|" ]
    for name, record in report["policies"].items():
        selected = report["selections"].get(name)
        threshold = selected["threshold_display"] if selected else (None if name == "fusion" else 0.)
        shown = "never" if threshold is None else f"{threshold:.12g}"
        m, p = record["mean_seed"], record["pooled"]
        lines.append(f"| {name} | {shown} | {100*m['accuracy']:.6f} | {100*m['weighted_f1']:.6f} | "
                     f"{p['actions']} | {p['rescues']} | {p['harms']} | {p['net']} | {p['plots_affected']} |")
    lines += ["", "## Paired accuracy differences", "",
              "| Contrast | Difference (pp) | Conditional 95% interval (pp) |", "|---|---:|---:|"]
    for name, entry in report["bootstrap"]["contrasts"].items():
        m = entry["accuracy"]
        lines.append(f"| {name} | {100*m['difference']:+.6f} | [{100*m['ci95_lower']:+.6f}, {100*m['ci95_upper']:+.6f}] |")
    lines += ["", "## Per-seed net corrections", "", "| Policy | Seed 1 | Seed 2 | Seed 3 | Seed 4 |", "|---|---:|---:|---:|---:|"]
    for name, record in report["policies"].items():
        lines.append("| " + name + " | " + " | ".join(str(record["per_seed"][str(s)]["net"]) for s in SEEDS) + " |")
    return "\n".join(lines) + "\n"


def validate(output):
    declaration = read_json(output / "declaration.json")
    if declaration["parent_snapshot"] != snapshot(PARENT):
        raise ValueError("Parent artifact bytes, size, or modification times changed")
    if declaration["source_sha256"] != {p: sha256(ROOT / p) for p in SOURCE_FILES}:
        raise ValueError("Declared analysis sources changed")
    manifest = read_json(output / "manifest.json")
    if set(manifest["file_sha256"]) != {str(p.relative_to(output)) for p in output.iterdir() if p.name != "manifest.json"}:
        raise ValueError("Unexpected or missing analysis artifacts")
    for name, expected in manifest["file_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"Analysis artifact changed: {name}")
    rows = load_rows()
    pd.testing.assert_frame_equal(rows, pd.read_parquet(output / "validation_rows.parquet"))
    selections, actions = select_policies(rows, declaration["config"])
    if selections != read_json(output / "selection.json"):
        raise ValueError("Threshold selection does not reproduce")
    saved_actions = pd.read_parquet(output / "actions.parquet")
    identity_columns = ["row_uid", "plot_idx", "training_seed"]
    if list(saved_actions.columns) != identity_columns + list(actions):
        raise ValueError("Action table columns do not match the declared policies")
    pd.testing.assert_frame_equal(saved_actions[identity_columns], rows[identity_columns])
    for name, values in actions.items():
        if not np.array_equal(values, saved_actions[name].to_numpy()):
            raise ValueError(f"Actions do not reproduce: {name}")
    report, draws = make_report(rows, selections, actions, declaration["config"])
    if report != read_json(output / "report.json") or format_report(report) != (output / "report.md").read_text():
        raise ValueError("Metrics or bootstrap summary do not reproduce")
    if not np.array_equal(draws, np.load(output / "bootstrap.npy", allow_pickle=False)):
        raise ValueError("Bootstrap draws do not reproduce")
    return {"status": "validated", "selected_policies": len(selections), "parent_files_unchanged": len(declaration["parent_snapshot"])}


def build(output):
    if output.exists():
        return {**validate(output), "status": "reused_valid"}
    count = authenticate_parent()
    declaration = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authorization": "User requested applying the previous relaxed rule to the TCP-augmented router",
        "config": read_json(CONFIG), "parent_stage_members_authenticated": count,
        "parent_snapshot": snapshot(PARENT),
        "source_sha256": {p: sha256(ROOT / p) for p in SOURCE_FILES},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "declaration.json", declaration)
    rows = load_rows()
    selections, actions = select_policies(rows, declaration["config"])
    write_json(output / "selection.json", selections)
    rows.to_parquet(output / "validation_rows.parquet", index=False)
    action_rows = rows[["row_uid", "plot_idx", "training_seed"]].copy()
    for name, values in actions.items():
        action_rows[name] = values
    action_rows.to_parquet(output / "actions.parquet", index=False)
    report, draws = make_report(rows, selections, actions, declaration["config"])
    write_json(output / "report.json", report)
    (output / "report.md").write_text(format_report(report))
    np.save(output / "bootstrap.npy", draws, allow_pickle=False)
    if declaration["parent_snapshot"] != snapshot(PARENT):
        raise ValueError("Analysis modified parent artifacts")
    write_json(output / "manifest.json", {"file_sha256": {
        p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()
    }})
    return {"status": "complete", "selected_policies": len(selections), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == PARENT or PARENT in output.parents or output in PARENT.parents:
        parser.error("Output must be separate from the parent artifact tree")
    result = {"build": build, "validate": validate}[args.command](output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

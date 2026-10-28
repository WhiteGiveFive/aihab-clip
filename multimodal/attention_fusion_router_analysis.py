"""R4 interpretation of immutable R3 routing outputs; no new routing or fitting."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import yaml

from .attention_fusion_geometry_router import (
    canonical_hash, load_config as load_input_config, read_json, require, resolve,
    sha256, verify_prepared, write_json,
)
from .attention_fusion_router_geometry_analysis import analyze_geometry_pair
from .attention_fusion_router_outcomes import (
    conditional_plot_bootstrap, confusion_transitions, missed_rescue_partition, plot_outcomes,
)


DEFAULT_CONFIG = Path("configs/multimodal_attention_fusion_router_analysis.yaml")
METRICS = ("acc1", "acc3", "macro_f1", "weighted_f1", "mcc")


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict, Path]:
    config = yaml.safe_load(resolve(path).read_text())
    expected = {"schema_version": 1, "stage": "R4",
        "input_config": "configs/multimodal_attention_fusion_router_validation.yaml",
        "output_subdirectory": "r4_analysis_v1", "expected_evaluations": 8,
        "analysis_role": "validation", "bootstrap": {"metric": "paired_image_weighted_acc1_delta",
            "unit": "whole_plot", "resamples": 10000, "seed": 20261023,
            "interval": "percentile_95", "quantile_method": "linear", "refit_thresholds": False},
        "case_index": {"per_outcome": 3, "order": "ascending_row_uid"},
        "training_allowed": False, "threshold_reselection_allowed": False,
        "cleaned_test_allowed": False, "result_assurance": "provisional_unresolved_memory_integrity"}
    require(config == expected, "R4 configuration differs from the declared analysis scope")
    r3_config = yaml.safe_load(resolve(config["input_config"]).read_text())
    require(r3_config["stage"] == "R3" and r3_config["output_subdirectory"] == "r3_validation_v1"
            and not r3_config["training_allowed"] and not r3_config["cleaned_test_allowed"],
            "R3 input scope differs")
    _, root = load_input_config(r3_config["input_config"])
    return config, root


def read_inputs(root: Path) -> dict:
    """Authenticate saved stages and read only development-validation analyses."""
    declaration = verify_prepared(root)
    stage = root / "r3_validation_v1"
    manifest = read_json(stage / "manifest.json")
    require(manifest["status"] == "R3_complete" and manifest["evaluations"] == 8,
            "R3 input is incomplete")
    inventory = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
    require(inventory == set(manifest["file_sha256"]) | {"manifest.json"}, "R3 input inventory differs")
    for name, digest in manifest["file_sha256"].items():
        require(sha256(stage / name) == digest, f"R3 input artifact changed: {name}")
    for name, digest in manifest["input_bindings"].items():
        require(sha256(root / name) == digest, f"R3 upstream input changed: {name}")
    score_manifest = read_json(root / "scores/manifest.json")
    reference_fields = ("reference_row_ids", "reference_row_plot_ids", "reference_row_labels",
                        "reference_plot_ids", "reference_plot_labels", "query_row_ids", "query_plot_ids")
    reference = None
    for encoder in ("geo", "pretrained", "joint_final", "joint_best"):
        name = f"{encoder}_class_distances.npz"
        require(sha256(root / "scores" / name) == score_manifest["file_sha256"][name],
                "R2 class-distance artifact changed")
        with np.load(root / "scores" / name, allow_pickle=False) as archive:
            identities = {key: archive[key] for key in reference_fields}
        if reference is None:
            reference = identities
        else:
            require(all(np.array_equal(reference[key], identities[key]) for key in reference_fields),
                    "Geometry arms do not share reference/query identities and labels")
    audit_path = root / "verification/r3_independent_audit.json"
    audit = read_json(audit_path)
    require(audit["status"] == "passed" and audit["stage_manifest_sha256"] == sha256(stage / "manifest.json"),
            "R3 independent audit binding differs")
    summary = read_json(stage / "summary.json")
    require(summary["test_data_scored"] is False and summary["training_performed"] is False,
            "R3 boundary differs")
    expected = {(f"{policy}__{arm}", rule) for policy in ("final", "best")
                for arm in ("pretrained", "checkpoint_matched") for rule in ("fixed_zero", "selected")}
    require(len(summary["evaluations"]) == 8 and
            {(e["comparison_id"], e["rule"]) for e in summary["evaluations"]} == expected,
            "R3 evaluation matrix differs")
    rows = pd.read_parquet(root / "inputs/validation_rows.parquet")
    require(len(rows) == summary["query_images"] and rows.plot_idx.nunique() == summary["query_plots"],
            "Query denominator differs")
    require(summary["class_labels"] == declaration["class_labels"], "Class order differs")
    frames = {}
    for evaluation in summary["evaluations"]:
        frame = pd.read_parquet(stage / evaluation["path"] / "query_results.parquet")
        require(frame.row_uid.is_unique and np.array_equal(frame.row_uid, rows.row_uid)
                and np.array_equal(frame.plot_idx, rows.plot_idx)
                and np.array_equal(frame.label_id_dense, rows.label_id_dense), "R3 query identity differs")
        frames[(evaluation["comparison_id"], evaluation["rule"])] = frame
    bindings = {name: sha256(root / name) for name in (
        "protocol/declaration.json", "scores/manifest.json", "r3_validation_v1/manifest.json",
        "verification/r3_independent_audit.json")}
    return {"declaration": declaration, "summary": summary, "frames": frames,
            "bindings": bindings, "stage": stage}


def habitat_metric_changes(evaluation: dict, baseline: dict, habitat: pd.DataFrame) -> pd.DataFrame:
    """Keep class F1 and recall separate: false positives can change class F1."""
    metrics, original = evaluation["metrics"], baseline["metrics"]
    result = habitat.copy()
    require(result.class_id.tolist() == list(range(18))
            and result.class_label.tolist() == metrics["class_labels"], "Habitat class order differs")
    for name, values in (("fusion", original), ("routed", metrics)):
        cm = np.asarray(values["confusion_matrix"])
        result[f"{name}_f1"] = values["class_f1"]
        result[f"{name}_false_positives"] = cm.sum(axis=0) - np.diag(cm)
        result[f"{name}_false_negatives"] = cm.sum(axis=1) - np.diag(cm)
    result["delta_f1"] = result.routed_f1 - result.fusion_f1
    result["macro_f1_delta_contribution"] = result.delta_f1 / 18
    require(np.isclose(result.macro_f1_delta_contribution.sum(),
                       metrics["macro_f1"] - original["macro_f1"], rtol=0, atol=1e-14),
            "Macro F1 decomposition differs")
    require(result.net_rescues.sum() == evaluation["counts"]["net_rescues"], "Habitat net total differs")
    return result


def case_index(frame: pd.DataFrame, classes: list[str], limit: int = 3) -> pd.DataFrame:
    """Deterministic numerical examples; this is not an ecological image review."""
    parts = []
    for outcome in ("rescue", "harm", "both_wrong_action", "missed_rescue"):
        selected = frame.loc[frame[outcome]].sort_values("row_uid").head(limit).copy()
        selected["example_outcome"] = outcome
        for source, target in (("label_id_dense", "truth_class"),
                               ("fusion_predictions", "fusion_class"), ("geo_predictions", "geo_class")):
            selected[target] = [classes[int(i)] for i in selected[source]]
        parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def _inventory(root: Path, prefix: str) -> dict:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not p.relative_to(root).parts[0].startswith(prefix)}


def _sources(config_path: str | Path) -> list[Path]:
    return [resolve(name) for name in (
        config_path, "tools/analyze_attention_fusion_router.py",
        "multimodal/attention_fusion_router_analysis.py", "multimodal/attention_fusion_router_outcomes.py",
        "multimodal/attention_fusion_router_geometry_analysis.py", "multimodal/attention_fusion_geometry_router.py",
        "multimodal/geometry_routing_policy.py",
        "dev_plans/attention_fusion_router/01_validation_protocol.md",
        "dev_plans/attention_fusion_router/03_validation_routing.md",
        "tests/test_attention_fusion_router_analysis.py", "tests/test_attention_fusion_router_outcomes.py",
        "tests/test_attention_fusion_router_geometry_analysis.py")]


def run(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    config, root = load_config(config_path)
    inputs = read_inputs(root)
    destination = root / config["output_subdirectory"]
    if destination.exists():
        return {**verify(config_path), "reused": True}
    preserved = _inventory(root, config["output_subdirectory"])
    stage = Path(tempfile.mkdtemp(prefix=config["output_subdirectory"] + ".incomplete-", dir=root))
    source_hashes = {}
    for source in _sources(config_path):
        name = str(source.relative_to(resolve(".")))
        target = stage / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[name] = sha256(target)
    write_json(stage / "protocol.json", {"stage": "R4", "config": config,
        "config_sha256": canonical_hash(config), "input_bindings": inputs["bindings"],
        "created_at": datetime.now(timezone.utc).isoformat(), "source_snapshot_sha256": source_hashes,
        "conditional_uncertainty": "Fixed experts, saved actions and fitted thresholds; resample whole plots. No threshold reselection or validation-reuse correction.",
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__}})
    classes = inputs["summary"]["class_labels"]
    summary = {"stage": "R4", "status": "complete", "class_labels": classes,
        "query_images": inputs["summary"]["query_images"], "query_plots": inputs["summary"]["query_plots"],
        "result_assurance": config["result_assurance"], "evaluations": [], "geometry_pairs": {},
        "training_performed": False, "thresholds_reselected": False, "test_data_scored": False}
    baseline = {b["checkpoint_policy"]: b for b in inputs["summary"]["baselines"] if b["expert"] == "fusion"}
    for evaluation in inputs["summary"]["evaluations"]:
        cid, rule = evaluation["comparison_id"], evaluation["rule"]
        frame = inputs["frames"][(cid, rule)]
        directory = stage / "evaluations" / cid / rule
        directory.mkdir(parents=True)
        plots, plot_summary = plot_outcomes(frame)
        plots.to_csv(directory / "plot_outcomes.csv", index=False)
        uncertainty = conditional_plot_bootstrap(plots, n_resamples=config["bootstrap"]["resamples"],
                                               seed=config["bootstrap"]["seed"])
        missed = missed_rescue_partition(frame, evaluation["routing_policy"])
        write_json(directory / "missed_rescues.json", missed)
        write_json(directory / "plot_summary.json", plot_summary)
        write_json(directory / "conditional_uncertainty.json", uncertainty)
        transitions = confusion_transitions(frame, classes)
        transitions.to_csv(directory / "confusion_transitions.csv", index=False)
        habitat = pd.read_csv(inputs["stage"] / evaluation["path"] / "habitat_changes.csv")
        habitat = habitat_metric_changes(evaluation, baseline[evaluation["policy"]], habitat)
        habitat.to_csv(directory / "habitat_metric_changes.csv", index=False)
        case_index(frame, classes, config["case_index"]["per_outcome"]).to_csv(directory / "case_index.csv", index=False)
        summary["evaluations"].append({"comparison_id": cid, "rule": rule,
            "checkpoint_policy": evaluation["policy"], "routing_policy": evaluation["routing_policy"],
            "metrics": {name: evaluation["metrics"][name] for name in METRICS},
            "metric_delta": {name: evaluation["metrics"][name] - baseline[evaluation["policy"]]["metrics"][name]
                             for name in METRICS}, "counts": evaluation["counts"],
            "plot_summary": plot_summary, "missed_rescues": missed, "conditional_uncertainty": uncertainty,
            "path": str(directory.relative_to(stage))})
    for checkpoint_policy in ("final", "best"):
        geometry, per_query = analyze_geometry_pair(inputs["frames"][(f"{checkpoint_policy}__pretrained", "fixed_zero")],
            inputs["frames"][(f"{checkpoint_policy}__checkpoint_matched", "fixed_zero")],
            checkpoint_policy=checkpoint_policy)
        directory = stage / "geometry" / checkpoint_policy
        write_json(directory / "summary.json", geometry)
        per_query.to_parquet(directory / "paired_queries.parquet", index=False)
        summary["geometry_pairs"][checkpoint_policy] = geometry
    write_json(stage / "summary.json", summary)
    _figures(stage, summary)
    _tables(stage, summary)
    for name, digest in preserved.items():
        require(sha256(root / name) == digest, f"Earlier artifact changed: {name}")
    for source, digest in inputs["declaration"]["source_sha256"].items():
        require(sha256(Path(source)) == digest, f"Bound source changed: {source}")
    write_json(stage / "preservation.json", {"status": "passed", "file_sha256": preserved,
        "bound_source_files_checked": len(inputs["declaration"]["source_sha256"])})
    manifest = {"stage": "R4", "status": "R4_complete", "config_sha256": canonical_hash(config),
        "input_bindings": inputs["bindings"], "result_assurance": config["result_assurance"],
        "file_sha256": {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
    write_json(stage / "manifest.json", manifest)
    _verify_stage(stage, config, inputs)
    stage.rename(destination)
    return {"status": "R4_complete", "output": str(destination), "evaluations": 8, "geometry_pairs": 2,
            "reused": False, "result_assurance": config["result_assurance"]}


def _figures(stage: Path, summary: dict) -> None:
    """Static research figures are paired with the underlying numerical tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = stage / "figures"
    output.mkdir()

    def save(fig, name):
        try:
            for extension in ("png", "pdf"):
                metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
                fig.savefig(output / f"{name}.{extension}", dpi=150, bbox_inches="tight", metadata=metadata)
        finally:
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    for ax, policy in zip(axes, ("final", "best")):
        frame = pd.read_csv(stage / "evaluations" / f"{policy}__pretrained/selected/habitat_metric_changes.csv")
        y = np.arange(len(frame))
        ax.barh(y, frame.net_rescues, color=np.where(frame.net_rescues >= 0, "#24805e", "#bd4d43"))
        ax.set_title(f"{policy.title()} experts / selected rule")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Rescues minus harms (images)")
        ax.set_yticks(y, [f"{r.class_label} (n={r.support})" for r in frame.itertuples()])
        for index, value in enumerate(frame.net_rescues):
            ax.text(value + (0.15 if value >= 0 else -0.15), index, str(value),
                    ha="left" if value >= 0 else "right", va="center", fontsize=8)
        ax.set_xlim(min(-3, frame.net_rescues.min() - 1), frame.net_rescues.max() + 2)
    axes[0].invert_yaxis()
    fig.suptitle("Pretrained geometry: habitat gains and harms on reused validation", fontsize=13)
    fig.tight_layout()
    save(fig, "habitat_net_changes")

    fig, ax = plt.subplots(figsize=(11, 6))
    records = summary["evaluations"]
    for index, record in enumerate(records):
        uncertainty = record["conditional_uncertainty"]
        point, low, high = _interval(uncertainty)
        ax.plot([100*low, 100*high], [index, index], color="#4973a5", linewidth=2)
        ax.scatter([100*point], [index], color="#193a64", s=30)
    ax.set_yticks(np.arange(len(records)), [f"{r['comparison_id']} / {r['rule']}" for r in records])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Acc@1 change from matching fusion (percentage points)")
    ax.set_title("Conditional 95% plot-bootstrap intervals: fixed experts, actions and thresholds")
    fig.text(0.5, 0.01, "10,000 resamples of whole plots; reused validation; no correction for policy selection", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "conditional_acc1_intervals")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, policy in zip(axes, ("final", "best")):
        # Draw from the same R3 source pair; no new policy or fitted cutoff.
        original = pd.read_parquet(stage.parent / "r3_validation_v1/comparisons" /
                                   f"{policy}__pretrained/fixed_zero/query_results.parquet")
        matched = pd.read_parquet(stage.parent / "r3_validation_v1/comparisons" /
                                  f"{policy}__checkpoint_matched/fixed_zero/query_results.parquet")
        eligible = original.eligible.to_numpy()
        for points, name, color in ((original, "Original pretrained", "#4973a5"),
                                    (matched, "Checkpoint-matched", "#c47735")):
            values = points.loc[eligible, "A_image"].to_numpy()
            ordered = np.sort(values)
            ax.step(ordered, np.arange(1, len(ordered)+1)/len(ordered), where="post", label=name, color=color)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{policy.title()} pair: {int(eligible.sum())} eligible disagreements")
        ax.set_xlabel("Image log-distance ratio (positive favors geo class)")
        ax.set_ylabel("Empirical cumulative proportion")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Image geometry comparison with expert predictions and native geo held fixed", fontsize=12)
    fig.tight_layout()
    save(fig, "image_geometry_distributions")


def _interval(uncertainty: dict) -> tuple[float, float, float]:
    # Schema supplied by the pure plot analysis helper.
    return (uncertainty["delta_acc1"], uncertainty["lower_95"], uncertainty["upper_95"])


def _tables(stage: Path, summary: dict) -> None:
    rows = []
    lines = ["# R4 analysis tables", "", "All results use the same reused 822-image / 325-plot validation cohort.",
             "Intervals condition on saved experts, actions and thresholds; they do not account for validation reuse or policy selection.", "",
             "| Evaluation | Acc@1 change, pp | Conditional 95% interval, pp | Rescues | Harms | Net |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for record in summary["evaluations"]:
        point, low, high = _interval(record["conditional_uncertainty"])
        counts = record["counts"]
        rows.append({"comparison_id": record["comparison_id"], "rule": record["rule"],
                     **record["metrics"], **{f"delta_{k}": v for k, v in record["metric_delta"].items()},
                     "ci_lower": low, "ci_upper": high, "rescues": counts["rescues"],
                     "harms": counts["harms"], "net_rescues": counts["net_rescues"]})
        lines.append(f"| {record['comparison_id']} / {record['rule']} | {100*point:+.2f} | "
                     f"[{100*low:+.2f}, {100*high:+.2f}] | {counts['rescues']} | {counts['harms']} | {counts['net_rescues']:+d} |")
    pd.DataFrame(rows).to_csv(stage / "metrics_and_uncertainty.csv", index=False)
    (stage / "results.md").write_text("\n".join(lines) + "\n")


def _verify_stage(stage: Path, config: dict, inputs: dict) -> None:
    manifest = read_json(stage / "manifest.json")
    require(manifest["status"] == "R4_complete" and manifest["config_sha256"] == canonical_hash(config)
            and manifest["input_bindings"] == inputs["bindings"], "R4 manifest binding differs")
    require({str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
            == set(manifest["file_sha256"]) | {"manifest.json"}, "R4 artifact inventory differs")
    for name, digest in manifest["file_sha256"].items():
        require(sha256(stage / name) == digest, f"R4 artifact changed: {name}")
    summary = read_json(stage / "summary.json")
    require(len(summary["evaluations"]) == 8 and set(summary["geometry_pairs"]) == {"final", "best"},
            "R4 analysis matrix differs")
    require(not any(summary[k] for k in ("training_performed", "thresholds_reselected", "test_data_scored")),
            "R4 boundary differs")
    for record in summary["evaluations"]:
        frame = inputs["frames"][(record["comparison_id"], record["rule"])]
        plots, plot_summary = plot_outcomes(frame)
        require(plot_summary == record["plot_summary"], "Plot analysis differs")
        require(missed_rescue_partition(frame, record["routing_policy"]) == record["missed_rescues"],
                "Missed-rescue partition differs")
        require(int(plots.net_rescues.sum()) == record["counts"]["net_rescues"], "Plot net total differs")
        require(conditional_plot_bootstrap(plots, n_resamples=config["bootstrap"]["resamples"],
                                          seed=config["bootstrap"]["seed"]) == record["conditional_uncertainty"],
                "Conditional plot-bootstrap result differs")
    for checkpoint in ("final", "best"):
        expected, _ = analyze_geometry_pair(inputs["frames"][(f"{checkpoint}__pretrained", "fixed_zero")],
            inputs["frames"][(f"{checkpoint}__checkpoint_matched", "fixed_zero")], checkpoint_policy=checkpoint)
        require(expected == summary["geometry_pairs"][checkpoint], "Paired geometry summary differs")


def verify(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    config, root = load_config(config_path)
    inputs = read_inputs(root)
    stage = root / config["output_subdirectory"]
    _verify_stage(stage, config, inputs)
    return {"status": "R4_verified", "output": str(stage), "evaluations": 8,
            "result_assurance": config["result_assurance"]}

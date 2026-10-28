"""R3 validation-only application of the frozen attention-fusion router recipe.

R1/R2 remain immutable inputs. Each run publishes a separate stage directory;
no encoder loading, training, geometry extraction or cleaned-test access occurs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd
import yaml

from .attention_fusion_geometry_router import (
    canonical_hash, load_config as load_input_config, read_json, require,
    resolve, sha256, validate_prediction_arrays, verify_prepared, write_json, write_npz,
)
from .geometry_routing_policy import (
    action_mask, evaluate_actions, finite_policy, select_policy,
)
from .attention_fusion_router_reporting import (
    habitat_changes, probability_metrics, save_confusion_figures,
    select_expert_probabilities,
)


DEFAULT_CONFIG = Path("configs/multimodal_attention_fusion_router_validation.yaml")
RECIPE = {
    "fixed_threshold": 0.0, "threshold_comparison": "strict_greater_than",
    "fit_separately_per_comparison": True, "minimum_acted_plots": 20,
    "minimum_coverage_fraction": 0.01, "maximum_harms_per_rescue": 0.7,
    "positive_net_rescue_required": True,
    "selection_objective": "maximize_rescues_minus_harms",
    "tie_breaks": ["fewer_harms", "fewer_actions", "higher_threshold"],
    "threshold_candidates": ["never", "zero", "positive_score_midpoints"],
    "no_feasible_policy": "retain_fusion",
    "selected_prediction": "full_probability_vector_of_chosen_expert",
}


def load_stage_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict, Path]:
    config = yaml.safe_load(resolve(path).read_text())
    expected = {
        "schema_version": 1, "stage": "R3",
        "input_config": "configs/multimodal_attention_fusion_geometry_router.yaml",
        "output_subdirectory": "r3_validation_v1",
        "checkpoint_policies": ["final", "best"],
        "geometry_arms": ["pretrained", "checkpoint_matched"],
        "routing_rules": ["fixed_zero", "selected"], "expected_evaluations": 8,
        "selection_recipe": "prospective_r3_from_frozen_input_declaration",
        "evaluation_role": "validation", "training_allowed": False,
        "cleaned_test_allowed": False,
        "result_assurance": "provisional_unresolved_memory_integrity",
    }
    require(config == expected, "R3 configuration differs from the authorized validation matrix")
    _, root = load_input_config(config["input_config"])
    return config, root


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def validate_scores(arrays: dict, rows: pd.DataFrame, fusion: dict, geo: dict) -> None:
    """Fail on misaligned candidates or masks before labels can select a policy."""
    size = len(rows)
    for key in ("row_uid", "plot_idx", "fusion_predictions", "geo_predictions", "score",
                "A_image", "A_geo", "reference_plots_fusion_class", "reference_plots_geo_class",
                "geometry_supported", "disagreement", "eligible"):
        require(key in arrays and arrays[key].shape == (size,), f"Invalid score field: {key}")
    require(np.array_equal(arrays["row_uid"], rows.row_uid)
            and np.array_equal(arrays["plot_idx"].astype(str), rows.plot_idx.astype(str)),
            "Score query alignment differs")
    require(np.array_equal(arrays["fusion_predictions"], fusion["prediction"])
            and np.array_equal(arrays["geo_predictions"], geo["prediction"]),
            "Score expert candidates differ")
    for key in ("geometry_supported", "disagreement", "eligible"):
        require(arrays[key].dtype.kind == "b", f"Non-boolean score mask: {key}")
    supported = ((arrays["reference_plots_fusion_class"] >= 5)
                 & (arrays["reference_plots_geo_class"] >= 5))
    disagreement = fusion["prediction"] != geo["prediction"]
    require(np.array_equal(arrays["geometry_supported"], supported)
            and np.array_equal(arrays["disagreement"], disagreement)
            and np.array_equal(arrays["eligible"], supported & disagreement),
            "Score eligibility/support differs")
    require(all(np.isfinite(arrays[key][supported]).all() for key in ("score", "A_image", "A_geo")),
            "Nonfinite supported score")
    require(np.array_equal(arrays["score"][supported],
                           np.minimum(arrays["A_image"][supported], arrays["A_geo"][supported])),
            "Minimum-modality score differs")


def load_inputs(root: Path) -> dict:
    declaration = verify_prepared(root)
    config = declaration["config"]
    require(config["prospective_r3"] == RECIPE, "Frozen R3 selection recipe differs")
    require(declaration["result_assurance"] == "provisional_unresolved_memory_integrity",
            "Input assurance differs")
    completion = read_json(root / "verification/completion.json")
    require(completion["status"] == "R1_R2_complete", "R2 is not complete")
    declaration_hash = sha256(root / "protocol/declaration.json")
    score_hash = sha256(root / "scores/manifest.json")
    require(completion["declaration_sha256"] == declaration_hash
            and completion["scores_manifest_sha256"] == score_hash,
            "R2 completion binding differs")
    manifest = read_json(root / "scores/manifest.json")
    require(manifest["declaration_sha256"] == declaration_hash, "Score declaration differs")
    for name in ("query_labels_used_for_geometry", "thresholds_selected", "routing_applied", "test_data_scored"):
        require(manifest[name] is False, f"R2 boundary differs: {name}")
    expected = [f"{policy}__{arm}" for policy in ("final", "best")
                for arm in ("pretrained", "checkpoint_matched")]
    require([c["comparison_id"] for c in declaration["comparisons"]] == expected
            and [c["comparison_id"] for c in manifest["comparisons"]] == expected,
            "Incomplete or different comparison matrix")
    expected_files = {f"{name}.npz" for name in expected} | {
        f"{name}_class_distances.npz" for name in ("geo", "pretrained", "joint_final", "joint_best")}
    require(set(manifest["file_sha256"]) == expected_files, "Incomplete R2 score inventory")
    for name, digest in manifest["file_sha256"].items():
        require(sha256(root / "scores" / name) == digest, f"Saved score changed: {name}")
    for name, digest in manifest["embedding_manifest_sha256"].items():
        require(sha256(root / "embeddings" / name / "manifest.json") == digest,
                f"Embedding manifest changed: {name}")
    rows = pd.read_parquet(root / "inputs/validation_rows.parquet")
    train = pd.read_parquet(root / "inputs/train_rows.parquet")
    require(len(rows) == config["data"]["validation_images"] and rows.row_uid.is_unique
            and rows.plot_idx.nunique() == config["data"]["validation_plots"], "Query cohort differs")
    require(not set(rows.row_uid) & set(train.row_uid)
            and not set(rows.plot_idx) & set(train.plot_idx), "Reference/query overlap")
    require(rows.groupby("plot_idx").label_id_dense.nunique().max() == 1,
            "Query labels vary within a plot")
    classes = declaration["class_labels"]
    experts = {}
    for policy in ("final", "best"):
        experts[policy] = {family: validate_prediction_arrays(
            _load_npz(root / "inputs" / f"{family}_{policy}_predictions.npz"), rows, len(classes))
            for family in ("fusion", "geo")}
    scores = {}
    for comparison in declaration["comparisons"]:
        cid, policy = comparison["comparison_id"], comparison["policy"]
        scores[cid] = _load_npz(root / "scores" / f"{cid}.npz")
        validate_scores(scores[cid], rows, experts[policy]["fusion"], experts[policy]["geo"])
    return {"declaration": declaration, "rows": rows, "experts": experts, "scores": scores,
            "bindings": {name: sha256(root / name) for name in (
                "protocol/declaration.json", "protocol/resolved_config.json", "scores/manifest.json",
                "verification/completion.json")}}


def decision_reasons(scores: dict, actions: np.ndarray, policy: dict) -> np.ndarray:
    reasons = np.full(len(actions), "score_not_above_threshold", dtype="U32")
    if policy["kind"] == "never_intervene":
        reasons[:] = "never_intervene_policy"
    reasons[~scores["geometry_supported"]] = "insufficient_reference_support"
    reasons[~scores["disagreement"]] = "experts_agree"
    reasons[actions] = "switch_to_geo"
    return reasons


def _inventory(root: Path, excluded_prefix: str) -> dict:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not p.relative_to(root).parts[0].startswith(excluded_prefix)}


def _source_paths(config_path: str | Path) -> list[Path]:
    return [resolve(name) for name in (
        config_path, "tools/run_attention_fusion_router_validation.py",
        "multimodal/attention_fusion_router_validation.py", "multimodal/geometry_routing_policy.py",
        "multimodal/attention_fusion_router_reporting.py", "multimodal/attention_fusion_geometry_router.py",
        "multimodal/attention_fusion_reporting.py", "dev_plans/attention_fusion_router/01_validation_protocol.md",
        "tests/test_geometry_routing_policy.py", "tests/test_attention_fusion_router_reporting.py",
        "tests/test_attention_fusion_router_validation.py")]


def _write_evaluation(directory: Path, rows: pd.DataFrame, probabilities: np.ndarray,
                      classes: list[str], *, actions: np.ndarray | None = None,
                      reasons: np.ndarray | None = None, title: str | None = None) -> dict:
    metrics, arrays = probability_metrics(rows.label_id_dense.to_numpy(), probabilities, classes)
    arrays.update(row_uid=rows.row_uid.to_numpy(dtype=str), plot_idx=rows.plot_idx.to_numpy(dtype=str),
                  label=rows.label_id_dense.to_numpy(dtype=np.int64))
    if actions is not None:
        arrays.update(action=actions, decision_reason=reasons)
    write_npz(directory / "predictions.npz", arrays)
    write_json(directory / "metrics.json", metrics)
    save_confusion_figures(metrics, classes, directory / "figures", title=title)
    return metrics


def _summary_table(summary: dict) -> str:
    lines = ["# R3 — Geometry routing on development validation", "",
             "Seed 1, pretrained-initialized partial-joint CLS + original embeddings; 822 queries.", "",
             "Both rules use fixed expert probabilities. Selected thresholds are fitted on these same validation labels.",
             "The results retain `provisional_unresolved_memory_integrity`.", "",
             "## Expert baselines", "",
             "| Checkpoint | Expert | Acc@1 % | Acc@3 % | Macro F1 % | Weighted F1 % | MCC |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in summary["baselines"]:
        m = item["metrics"]
        lines.append(f"| {item['checkpoint_policy']} | {item['expert']} | {100*m['acc1']:.2f} | "
                     f"{100*m['acc3']:.2f} | {100*m['macro_f1']:.2f} | {100*m['weighted_f1']:.2f} | {m['mcc']:.4f} |")
    lines += ["", "## All eight routing evaluations", "",
              "| Cell | Rule | Threshold | Actions / plots | Rescues | Harms | Both wrong | Net | Acc@1 % | Δ pp | Acc@3 % | Macro F1 % | Weighted F1 % | MCC |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summary["evaluations"]:
        m, c, p = item["metrics"], item["counts"], item["routing_policy"]
        threshold = "never" if p["kind"] == "never_intervene" else f"{p['value_decimal']:.10g}"
        lines.append(f"| {item['comparison_id']} | {item['rule']} | {threshold} | "
                     f"{c['actions']} / {c['action_plots']} | {c['rescues']} | {c['harms']} | "
                     f"{c['both_wrong_actions']} | {c['net_rescues']:+d} | {100*m['acc1']:.2f} | "
                     f"{100*c['delta_acc1']:+.2f} | {100*m['acc3']:.2f} | {100*m['macro_f1']:.2f} | "
                     f"{100*m['weighted_f1']:.2f} | {m['mcc']:.4f} |")
    lines += ["", "Thresholds use strict `>` and are stored exactly as hexadecimal floats in policies.json.",
              "Each selected rule independently requires 20 acted plots, 1% all-query coverage, 10H ≤ 7R and positive net rescue.",
              "Never-intervene is the prescribed fallback when no finite candidate is feasible; fixed-zero results remain reported.",
              "", "Macro F1 averages all 18 classes with undefined F1 set to zero. An absent truth class has no estimable recall;",
              "its normalized confusion row is recorded as null and rendered as zero for the figure.",
              "Detailed candidate curves, per-query actions/outcomes, habitat changes, full routed probabilities, raw/normalized matrices",
              "and PNG/PDF confusion figures are saved alongside summary.json. R4 interpretation remains a separate milestone.", ""]
    return "\n".join(lines)


def run(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    config, root = load_stage_config(config_path)
    inputs = load_inputs(root)
    destination = root / config["output_subdirectory"]
    if destination.exists():
        verified = verify(config_path)
        return {**verified, "reused": True}
    preserved = _inventory(root, config["output_subdirectory"])
    stage = Path(tempfile.mkdtemp(prefix=config["output_subdirectory"] + ".incomplete-", dir=root))
    declaration, rows = inputs["declaration"], inputs["rows"]
    classes = declaration["class_labels"]
    source_hashes = {}
    for source in _source_paths(config_path):
        relative = str(source.relative_to(resolve(".")))
        target = stage / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[relative] = sha256(target)
    protocol = {"schema_version": 1, "stage": "R3", "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": canonical_hash(config), "input_bindings": inputs["bindings"],
        "selection_recipe": RECIPE, "class_labels": classes,
        "source_snapshot_sha256": source_hashes,
        "result_assurance": declaration["result_assurance"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "validation_labels_used_for_threshold_selection": True, "training_performed": False,
        "test_data_scored": False}
    write_json(stage / "protocol.json", protocol)
    summary = {"stage": "R3", "status": "complete", "query_images": len(rows),
        "query_plots": int(rows.plot_idx.nunique()), "class_labels": classes,
        "result_assurance": declaration["result_assurance"], "baselines": [], "evaluations": [],
        "validation_labels_used_for_threshold_selection": True, "routing_applied": True,
        "test_data_scored": False, "training_performed": False}
    for checkpoint_policy, experts in inputs["experts"].items():
        for family, predictions in experts.items():
            directory = stage / "baselines" / checkpoint_policy / family
            metrics = _write_evaluation(directory, rows, predictions["probabilities"], classes,
                title=f"Seed 1 validation: {checkpoint_policy} / {family} | Provisional memory-integrity assurance")
            summary["baselines"].append({"checkpoint_policy": checkpoint_policy, "expert": family,
                                         "metrics": metrics, "path": str(directory.relative_to(stage))})
    for comparison in declaration["comparisons"]:
        cid, checkpoint_policy = comparison["comparison_id"], comparison["policy"]
        scores = inputs["scores"][cid]
        fusion, geo = (inputs["experts"][checkpoint_policy][name] for name in ("fusion", "geo"))
        selection = select_policy(scores["score"], scores["eligible"], fusion["prediction"],
                                  geo["prediction"], rows.label_id_dense.to_numpy(), rows.plot_idx.to_numpy(dtype=str))
        directory = stage / "comparisons" / cid
        write_json(directory / "policies.json", selection)
        candidates = [{"kind": item["policy"]["kind"], "value_hex": item["policy"]["value_hex"],
                       "value_decimal": item["policy"]["value_decimal"], **item["counts"],
                       "feasible": item["feasible"], "infeasibility_reasons": ";".join(item["infeasibility_reasons"])}
                      for item in selection["candidate_records"]]
        pd.DataFrame(candidates).to_csv(directory / "threshold_candidates.csv", index=False)
        for rule, policy in (("fixed_zero", finite_policy(0.0)), ("selected", selection["selected_policy"])):
            actions = action_mask(policy, scores["score"], scores["eligible"])
            counts = evaluate_actions(actions, fusion["prediction"], geo["prediction"],
                                      rows.label_id_dense.to_numpy(), rows.plot_idx.to_numpy(dtype=str))
            probabilities = select_expert_probabilities(fusion["probabilities"], geo["probabilities"], actions)
            reasons = decision_reasons(scores, actions, policy)
            evaluation_dir = directory / rule
            metrics = _write_evaluation(evaluation_dir, rows, probabilities, classes, actions=actions, reasons=reasons,
                title=f"Seed 1 validation: {cid} / {rule} | Provisional memory-integrity assurance")
            require(int(round(metrics["acc1"] * len(rows))) == counts["routed_correct"], "Routed count identity failed")
            write_json(evaluation_dir / "counts.json", counts)
            habitats = habitat_changes(rows.label_id_dense.to_numpy(), fusion["prediction"], geo["prediction"],
                                        actions, rows.plot_idx.to_numpy(dtype=str), classes)
            pd.DataFrame(habitats).to_csv(evaluation_dir / "habitat_changes.csv", index=False)
            per_query = rows[["row_uid", "file", "plot_idx", "label_id_dense", "label_name"]].copy()
            for key in scores:
                if key not in ("row_uid", "plot_idx"):
                    per_query[key] = scores[key]
            truth = rows.label_id_dense.to_numpy()
            per_query["action"] = actions
            per_query["decision_reason"] = reasons
            per_query["routed_prediction"] = probabilities.argmax(axis=1)
            per_query["rescue"] = actions & (fusion["prediction"] != truth) & (geo["prediction"] == truth)
            per_query["harm"] = actions & (fusion["prediction"] == truth) & (geo["prediction"] != truth)
            per_query["both_wrong_action"] = actions & (fusion["prediction"] != truth) & (geo["prediction"] != truth)
            per_query["missed_rescue"] = ~actions & (fusion["prediction"] != truth) & (geo["prediction"] == truth)
            per_query.to_parquet(evaluation_dir / "query_results.parquet", index=False)
            reason_counts = {str(reason): {"images": int(mask.sum()),
                "plots": int(rows.loc[mask, "plot_idx"].nunique())}
                for reason in np.unique(reasons) for mask in [reasons == reason]}
            support = {"supported_queries": int(scores["geometry_supported"].sum()),
                       "supported_query_plots": int(rows.loc[scores["geometry_supported"], "plot_idx"].nunique()),
                       "eligible_disagreements": int(scores["eligible"].sum()),
                       "eligible_disagreement_plots": int(rows.loc[scores["eligible"], "plot_idx"].nunique()),
                       "unsupported_queries": int((~scores["geometry_supported"]).sum()),
                       "unsupported_query_plots": int(rows.loc[~scores["geometry_supported"], "plot_idx"].nunique()),
                       "decision_reasons": reason_counts}
            write_json(evaluation_dir / "support_and_reasons.json", support)
            summary["evaluations"].append({**comparison, "rule": rule, "routing_policy": policy,
                "metrics": metrics, "counts": counts, "support": support,
                "selection_reason": selection["selection_reason"] if rule == "selected" else "prespecified_zero",
                "path": str(evaluation_dir.relative_to(stage))})
    write_json(stage / "summary.json", summary)
    (stage / "results.md").write_text(_summary_table(summary))
    for relative, digest in preserved.items():
        require(sha256(root / relative) == digest, f"R1/R2 input changed during R3: {relative}")
    for source, digest in declaration["source_sha256"].items():
        require(sha256(Path(source)) == digest, f"Expert/source input changed during R3: {source}")
    write_json(stage / "preservation.json", {"status": "passed", "file_sha256": preserved,
        "bound_source_files_checked": len(declaration["source_sha256"]), "training_performed": False})
    manifest = {"schema_version": 1, "status": "R3_complete", "result_assurance": declaration["result_assurance"],
        "input_bindings": inputs["bindings"], "evaluations": len(summary["evaluations"]),
        "config_sha256": canonical_hash(config),
        "file_sha256": {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
    write_json(stage / "manifest.json", manifest)
    _verify_stage(stage, config, inputs)
    stage.rename(destination)
    return {"status": "R3_complete", "output": str(destination), "evaluations": 8,
            "reused": False, "result_assurance": declaration["result_assurance"]}


def _verify_stage(stage: Path, config: dict, inputs: dict) -> None:
    manifest = read_json(stage / "manifest.json")
    require(manifest["status"] == "R3_complete" and manifest["evaluations"] == 8
            and manifest["config_sha256"] == canonical_hash(config), "Incomplete/different R3 manifest")
    require(manifest["input_bindings"] == inputs["bindings"], "R3 input binding differs")
    actual = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
    require(actual == set(manifest["file_sha256"]) | {"manifest.json"}, "R3 file inventory differs")
    for name, digest in manifest["file_sha256"].items():
        require(sha256(stage / name) == digest, f"R3 artifact changed: {name}")
    summary = read_json(stage / "summary.json")
    expected = {(cid, rule) for cid in inputs["scores"] for rule in ("fixed_zero", "selected")}
    require(len(summary["evaluations"]) == 8
            and {(r["comparison_id"], r["rule"]) for r in summary["evaluations"]} == expected,
            "R3 evaluation matrix differs")
    require(summary["test_data_scored"] is False and summary["training_performed"] is False,
            "R3 cannot train or score test data")
    rows = inputs["rows"]
    require(len(summary["baselines"]) == 4 and {(b["checkpoint_policy"], b["expert"])
        for b in summary["baselines"]} == {(p, e) for p in ("final", "best") for e in ("fusion", "geo")},
        "R3 expert baseline matrix differs")
    for baseline in summary["baselines"]:
        expert = inputs["experts"][baseline["checkpoint_policy"]][baseline["expert"]]
        metrics, predicted = probability_metrics(rows.label_id_dense.to_numpy(), expert["probabilities"],
                                                 inputs["declaration"]["class_labels"])
        saved = _load_npz(stage / baseline["path"] / "predictions.npz")
        require(metrics == baseline["metrics"] and np.array_equal(saved["probabilities"], expert["probabilities"])
                and np.array_equal(saved["row_uid"], rows.row_uid), "Expert baseline differs")
    for item in summary["evaluations"]:
        cid, checkpoint_policy = item["comparison_id"], item["policy"]
        scores = inputs["scores"][cid]
        fusion, geo = (inputs["experts"][checkpoint_policy][name] for name in ("fusion", "geo"))
        selected = select_policy(scores["score"], scores["eligible"], fusion["prediction"], geo["prediction"],
                                 rows.label_id_dense.to_numpy(), rows.plot_idx.to_numpy(dtype=str))
        require(read_json(stage / "comparisons" / cid / "policies.json") == selected,
                "Saved policy selection differs")
        policy = finite_policy(0.0) if item["rule"] == "fixed_zero" else selected["selected_policy"]
        require(policy == item["routing_policy"], "Reported policy differs")
        actions = action_mask(policy, scores["score"], scores["eligible"])
        probability = select_expert_probabilities(fusion["probabilities"], geo["probabilities"], actions)
        metrics, predicted = probability_metrics(rows.label_id_dense.to_numpy(), probability,
                                                 inputs["declaration"]["class_labels"])
        saved = _load_npz(stage / item["path"] / "predictions.npz")
        require(np.array_equal(saved["action"], actions)
                and np.array_equal(saved["probabilities"], probability)
                and np.array_equal(saved["prediction"], predicted["prediction"])
                and np.array_equal(saved["pred_top3"], predicted["pred_top3"])
                and np.array_equal(saved["row_uid"], rows.row_uid)
                and np.array_equal(saved["plot_idx"], rows.plot_idx.astype(str))
                and np.array_equal(saved["label"], rows.label_id_dense)
                and np.array_equal(saved["decision_reason"], decision_reasons(scores, actions, policy)),
                "Routed prediction mismatch")
        require(metrics == item["metrics"], "Routed metrics differ")
        require(evaluate_actions(actions, fusion["prediction"], geo["prediction"],
                                 rows.label_id_dense.to_numpy(), rows.plot_idx.to_numpy(dtype=str)) == item["counts"],
                "Routed counts differ")


def verify(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    config, root = load_stage_config(config_path)
    inputs = load_inputs(root)
    stage = root / config["output_subdirectory"]
    _verify_stage(stage, config, inputs)
    return {"status": "R3_verified", "output": str(stage), "evaluations": 8,
            "result_assurance": inputs["declaration"]["result_assurance"]}

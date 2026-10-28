#!/usr/bin/env python3
"""Measure seed-1 validation geo-rescue headroom from saved M3J predictions.

This is a label-informed diagnostic, not a fitted or evaluated router. Completed
study packages are read only; all outputs go into a new analysis directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = Path(
    "multimodal_artifacts/attention_fusion/cs/gse_100m/"
    "attention_fusion_joint_geo_only_direct_v1/reports/seed1_geo_only/summary.json"
)
DEFAULT_OUTPUT = Path(
    "multimodal_artifacts/analysis/cs/gse_100m/attention_fusion/"
    "seed1_geo_rescue_opportunity_v1"
)
FOCUS = "attention_cls_skip_original"
KEYS = ("row_uid", "plot_idx", "label", "prediction")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def repo_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve()


def display_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path: Path, classes: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in KEYS}
    n = len(arrays["row_uid"])
    require(n > 0 and all(value.shape == (n,) for value in arrays.values()),
            f"Invalid prediction shapes: {path}")
    require(len(np.unique(arrays["row_uid"])) == n, f"Duplicate row_uid: {path}")
    for key in ("label", "prediction"):
        values = arrays[key]
        require(values.dtype.kind in "iu" and bool(((values >= 0) & (values < classes)).all()),
                f"Invalid {key}: {path}")
    order = np.argsort(arrays["row_uid"])
    return {key: value[order] for key, value in arrays.items()}


def compare(model: dict, geo: dict) -> tuple[dict, np.ndarray]:
    for key in ("row_uid", "plot_idx", "label"):
        require(np.array_equal(model[key], geo[key]), f"Model/geo {key} mismatch")
    correct = model["prediction"] == model["label"]
    geo_correct = geo["prediction"] == geo["label"]
    rescue = ~correct & geo_correct
    harm = correct & ~geo_correct
    both_wrong = ~correct & ~geo_correct
    disagreement = model["prediction"] != geo["prediction"]
    n = len(correct)
    counts = {
        "n": n, "plots": len(np.unique(model["plot_idx"])),
        "model_correct": int(correct.sum()), "geo_correct": int(geo_correct.sum()),
        "model_errors": int((~correct).sum()),
        "both_correct": int((correct & geo_correct).sum()),
        "potential_geo_rescues": int(rescue.sum()),
        "potential_geo_harms": int(harm.sum()),
        "both_wrong": int(both_wrong.sum()),
        "disagreements": int(disagreement.sum()),
        "both_wrong_disagreements": int((both_wrong & disagreement).sum()),
        "both_wrong_agreements": int((both_wrong & ~disagreement).sum()),
        "rescue_plots": len(np.unique(model["plot_idx"][rescue])),
        "harm_plots": len(np.unique(model["plot_idx"][harm])),
        "model_acc1_pct": float(100 * correct.mean()),
        "geo_acc1_pct": float(100 * geo_correct.mean()),
        "oracle_acc1_pct": float(100 * (correct | geo_correct).mean()),
        "oracle_gain_pp": float(100 * rescue.mean()),
        "rescuable_model_errors_pct": float(100 * rescue.sum() / (~correct).sum())
        if (~correct).any() else 0.0,
        "switch_all_net_correct": int(rescue.sum() - harm.sum()),
    }
    require(sum(counts[key] for key in ("both_correct", "potential_geo_rescues",
                                      "potential_geo_harms", "both_wrong")) == n,
            "Correctness states do not partition the cohort")
    require(counts["disagreements"] == counts["potential_geo_rescues"]
            + counts["potential_geo_harms"] + counts["both_wrong_disagreements"],
            "Disagreement partition failed")
    require(counts["switch_all_net_correct"] == counts["geo_correct"] - counts["model_correct"],
            "Switch-all accuracy identity failed")
    outcome = np.full(n, "both_wrong", dtype="<U24")
    outcome[correct & geo_correct] = "both_correct"
    outcome[rescue] = "potential_geo_rescue"
    outcome[harm] = "potential_geo_harm"
    return counts, outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary_path, output = repo_path(args.summary), repo_path(args.output_dir)
    summary = json.loads(summary_path.read_text())
    require(summary["completed_fits"] == 33 and summary["incomplete_fits"] == 0
            and summary["seeds"] == [1], "Expected the complete M3J seed-1 summary")
    roots = [summary_path.parents[2], *[
        repo_path(Path(root)) for root in summary["completed_reference_packages"]["roots"]
    ]]
    require(not any(output == root or output.is_relative_to(root) for root in roots),
            "Analysis must be outside completed study packages")
    require(not output.exists() or not any(output.iterdir()), "Output directory must be new or empty")
    labels = summary["class_labels"]
    sources = {}

    def record(path: Path) -> None:
        sources[display_path(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    record(summary_path)
    record(Path(__file__).resolve())
    directories = {}
    for item in summary["inventory"]:
        matches = [root / "runs" / item["run_id"] for root in roots
                   if (root / "runs" / item["run_id"] / "run.json").is_file()]
        require(len(matches) == 1, f"Expected exactly one source run: {item['run_id']}")
        directory = matches[0]
        run_path = directory / "run.json"
        spec = json.loads(run_path.read_text())
        record(run_path)
        require(spec["class_labels"] == labels and spec["status"] == "complete",
                f"Incomplete run or class-order mismatch: {run_path}")
        for key in ("run_id", "seed", "candidate_id", "regime", "initialization"):
            require(spec[key] == item[key], f"Run identity differs: {run_path}/{key}")
        directories[item["run_id"]] = directory

    results, class_rows, outcome_rows = [], [], []
    for policy in ("final", "best"):
        metrics = summary["policies"][policy]["runs"]
        geo_metrics, = [row for row in metrics if row["candidate_id"] == "geo_only"]
        geo_path = directories[geo_metrics["run_id"]] / f"{policy}_predictions.npz"
        geo = load_predictions(geo_path, len(labels))
        record(geo_path)
        require(np.isclose((geo["label"] == geo["prediction"]).mean(), geo_metrics["acc1"],
                           atol=1e-12, rtol=0), "Geo accuracy differs from source report")
        require(len(geo["label"]) == 822 and len(np.unique(geo["plot_idx"])) == 325,
                "Unexpected validation cohort")
        for row in metrics:
            if row["candidate_id"] == "geo_only":
                continue
            path = directories[row["run_id"]] / f"{policy}_predictions.npz"
            model = load_predictions(path, len(labels))
            record(path)
            counts, outcomes = compare(model, geo)
            require(np.isclose(counts["model_acc1_pct"] / 100, row["acc1"],
                               atol=1e-12, rtol=0), f"Model accuracy differs: {path}")
            identity = {"policy": policy, **{key: row[key] for key in (
                "run_id", "seed", "regime", "initialization", "candidate_id")},
                "model_epoch": row["epoch"], "geo_epoch": geo_metrics["epoch"]}
            results.append({**identity, **counts})
            if row["candidate_id"] != FOCUS:
                continue
            for label_id, name in enumerate(labels):
                select = model["label"] == label_id
                if select.any():
                    subset, _ = compare({key: value[select] for key, value in model.items()},
                                        {key: value[select] for key, value in geo.items()})
                else:
                    subset = {key: None if key.endswith(("_pct", "_pp")) else 0
                              for key in counts}
                class_rows.append({**identity, "label_id": label_id, "label_name": name, **subset})
            for index in range(len(outcomes)):
                outcome_rows.append({**identity, "row_uid": str(model["row_uid"][index]),
                    "plot_idx": str(model["plot_idx"][index]),
                    "label": int(model["label"][index]),
                    "label_name": labels[int(model["label"][index])],
                    "model_prediction": int(model["prediction"][index]),
                    "geo_prediction": int(geo["prediction"][index]),
                    "outcome": str(outcomes[index])})

    require(len(results) == 64 and len(class_rows) == 144 and len(outcome_rows) == 6576,
            "Unexpected analysis dimensions")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_models.csv", results)
    write_csv(output / "cls_original_by_class.csv", class_rows)
    write_csv(output / "cls_original_cases.csv", outcome_rows)
    payload = {
        "schema_version": 1, "analysis": "seed1_validation_geo_rescue_opportunity",
        "scope": "822 development-validation images / 325 plots; no test scoring",
        "geo_reference": "M3J seed-1 standardized geo-only; final paired with final, best with best",
        "oracle_definition": "Switch only when model is wrong and geo is correct; retain model otherwise",
        "router_evaluated": False, "class_labels": labels,
        "result_assurance": summary["result_assurance"],
        "numerical_assurance_notice": summary["numerical_assurance_notice"],
        "encoder_regime_rng_limitation": summary["encoder_regime_rng_limitation"],
        "sources_sha256": sources, "comparisons": results,
        "validation": {"paired_rows_labels_plots": True, "class_order_matches": True,
                       "source_acc1_matches": True, "state_partition_checks": True},
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "comparisons": len(results),
                      "focus_class_rows": len(class_rows), "focus_case_rows": len(outcome_rows),
                      "source_files": len(sources)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

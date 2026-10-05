#!/usr/bin/env python3
"""Render the sealed joint-TCP report into a separate review directory.

This postprocessing utility is deliberately outside the sealed study sources.
It reads aggregated report metrics only; it cannot train or select a policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


VARIANTS = ("ablation_numeric25", "ablation_no_pairs79")
WEIGHTS = (0, 0.1, 1, 10)
GUARDRAILS = tuple((plots, harm) for plots in (20, 10, 0) for harm in (5, 7, 9))
INPUT_NAMES = {
    ("ablation_numeric25", False): "25 summaries",
    ("ablation_numeric25", True): "25 summaries + embeddings",
    ("ablation_no_pairs79", False): "79 features",
    ("ablation_no_pairs79", True): "79 features + embeddings",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_tables(report: dict) -> dict[str, pd.DataFrame]:
    """Build all predeclared mean-gain and matched-contrast rows in grid order."""
    if (report.get("study_id") != "joint_tcp_router_v1"
            or report.get("model_count") != 64
            or report.get("selected_policy_count") != 144):
        raise ValueError("Expected the complete sealed 64-fit joint_tcp_router_v1 report")
    declared = [entry["configuration"] for entry in report["diagnostics"].values()]
    lookup = {(item["variant"], bool(item["with_embeddings"]), item["auxiliary_weight"]): item
              for item in declared}
    expected = {(variant, emb, weight) for variant in VARIANTS
                for emb in (False, True) for weight in WEIGHTS}
    if set(lookup) != expected or len(declared) != 16:
        raise ValueError("Report configurations do not match the approved factorial grid")
    contrasts = report["validation_bootstrap"]["contrasts"]
    tables = {name: [] for name in ("fusion_gains", "auxiliary_contrasts", "embedding_contrasts")}
    for variant in VARIANTS:
        for emb in (False, True):
            for weight in WEIGHTS:
                item = lookup[(variant, emb, weight)]
                for column, (plots, harm) in enumerate(GUARDRAILS):
                    suffix = f"plots_{plots}__harm_0p{harm}"
                    treatment = f"{item['config_id']}__{suffix}"
                    base = {
                        "variant": variant, "with_embeddings": emb,
                        "auxiliary_weight": weight, "minimum_plots": plots,
                        "harm_cap": harm / 10, "column": column,
                        "treatment": treatment,
                    }

                    def add(name: str, reference: str, label: str) -> None:
                        metric = contrasts[f"{treatment}_minus_{reference}"]["accuracy"]
                        tables[name].append({
                            **base, "reference": reference, "row_label": label,
                            "gain_pp": 100 * metric["difference"],
                            "ci95_lower_pp": 100 * metric["ci95_lower"],
                            "ci95_upper_pp": 100 * metric["ci95_upper"],
                        })

                    add("fusion_gains", "fusion", f"{INPUT_NAMES[(variant, emb)]}  |  λ={weight:g}")
                    if weight > 0:
                        control = lookup[(variant, emb, 0)]["config_id"]
                        add("auxiliary_contrasts", f"{control}__{suffix}",
                            f"{INPUT_NAMES[(variant, emb)]}  |  λ={weight:g} − 0")
                    if emb:
                        control = lookup[(variant, False, weight)]["config_id"]
                        short_name = "25 summaries" if variant == VARIANTS[0] else "79 features"
                        add("embedding_contrasts", f"{control}__{suffix}",
                            f"{short_name}  |  λ={weight:g}  |  +emb − no emb")
    expected_lengths = {"fusion_gains": 144, "auxiliary_contrasts": 108, "embedding_contrasts": 72}
    if {key: len(value) for key, value in tables.items()} != expected_lengths:
        raise ValueError("Incomplete review table")
    return {name: pd.DataFrame(records) for name, records in tables.items()}


def draw_heatmap(table: pd.DataFrame, *, title: str, subtitle: str,
                 include_intervals: bool, group_size: int, output: Path) -> list[Path]:
    """Preserve the approved order and annotate every cell; do not rank arms."""
    row_names = table.row_label.drop_duplicates().tolist()
    matrix = np.array([table.loc[table.row_label.eq(name)].sort_values("column").gain_pp
                       for name in row_names])
    limit = max(float(np.max(np.abs(matrix))), .01)
    height = .55 * len(row_names) + 3.0
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(17, height))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
    axis.set_yticks(np.arange(len(row_names)), labels=row_names)
    axis.set_xticks(np.arange(9), labels=[f"≥{plots} plots\nH/R ≤ {harm / 10:g}" for plots, harm in GUARDRAILS])
    axis.tick_params(axis="both", length=0, pad=9)
    axis.set_xlabel("Validation threshold guardrail", labelpad=14)
    for row_index, name in enumerate(row_names):
        records = table.loc[table.row_label.eq(name)].sort_values("column")
        for column, record in enumerate(records.itertuples(index=False)):
            value = f"{record.gain_pp:+.2f}"
            if include_intervals:
                value += f"\n[{record.ci95_lower_pp:+.2f}, {record.ci95_upper_pp:+.2f}]"
            text_color = "white" if abs(record.gain_pp) > .62 * limit else "#111111"
            axis.text(column, row_index, value, ha="center", va="center",
                      color=text_color, fontsize=9 if include_intervals else 10)
    for boundary in range(group_size, len(row_names), group_size):
        axis.axhline(boundary - .5, color="white", linewidth=3)
    for boundary in (2.5, 5.5):
        axis.axvline(boundary, color="white", linewidth=3)
    axis.set_xticks(np.arange(-.5, 9, 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(row_names), 1), minor=True)
    axis.grid(which="minor", color="white", alpha=.35, linewidth=.6)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=.025, pad=.025)
    colorbar.set_label("Accuracy difference (percentage points)")
    figure.suptitle(title, x=.03, y=.985, ha="left", fontsize=17, weight="bold")
    figure.text(.03, .943, subtitle, ha="left", va="top", fontsize=11)
    footnote = ("Four-seed mean on development-validation; all declared settings shown in fixed order. "
                "Positive values favor the treatment.\n"
                "Intervals use 2,000 paired whole-plot resamples with fixed validation-selected thresholds; "
                "exploratory, with no multiple-comparison correction.")
    figure.text(.03, .025, footnote, ha="left", va="bottom", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.31, right=.94, bottom=.17 if len(row_names) <= 8 else .14,
                           top=.84 if len(row_names) <= 8 else .88)
    paths = [output.with_suffix(".png"), output.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)
    return paths


def build_review(report_path: Path, output: Path) -> dict:
    report_path, output = report_path.resolve(), output.resolve()
    if output == report_path.parent or report_path.parent in output.parents:
        raise ValueError("Write review figures outside the sealed score directory")
    for parent in report_path.parents:
        if parent.name == "joint_tcp_router_v1" and (output == parent or parent in output.parents):
            raise ValueError("Write review figures in a sibling of the sealed study")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Review output directory must be empty to avoid overwriting artifacts")
    report = json.loads(report_path.read_text())
    tables = ordered_tables(report)
    output.mkdir(parents=True, exist_ok=True)
    fusion = 100 * report["validation_policies"]["fusion"]["mean_seed"]["accuracy"]
    specifications = (
        ("fusion_gains", "Validation accuracy gain over fusion",
         f"All 16 router configurations × 9 guardrails. Fusion accuracy: {fusion:.4f}%. Cell values are percentage points.", False, 4),
        ("auxiliary_contrasts", "Auxiliary TCP supervision: matched comparison against CE only",
         "Same inputs and guardrail setting; each λ > 0 minus λ = 0. Cells show percentage-point differences [95% CI].", True, 3),
        ("embedding_contrasts", "Embedding inputs: matched comparison at each auxiliary weight",
         "Same base features, λ and guardrail setting; embeddings minus no embeddings. Cells show percentage points [95% CI].", True, 4),
    )
    artifacts = []
    for name, title, subtitle, intervals, group_size in specifications:
        csv_path = output / f"{name}.csv"
        tables[name].to_csv(csv_path, index=False)
        artifacts.append(csv_path)
        artifacts += draw_heatmap(tables[name], title=title, subtitle=subtitle,
                                  include_intervals=intervals, group_size=group_size,
                                  output=output / name)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": report["study_id"], "source_report": str(report_path),
        "source_report_sha256": sha256(report_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "rows": {name: len(table) for name, table in tables.items()},
        "policy_selection_performed": False, "training_performed": False,
        "labels_read": False, "cleaned_test_access": False,
        "artifacts": {str(path.name): {"sha256": sha256(path), "bytes": path.stat().st_size}
                      for path in artifacts},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_review(args.report, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "rows": receipt["rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()

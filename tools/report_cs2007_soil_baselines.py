from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback for minimal plotting envs
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ROOT = (
    PROJECT_ROOT
    / "multimodal_artifacts"
    / "runs"
    / "cs2007_soil_aligned"
    / "hf-hub_timm_ViT-SO400M-16-SigLIP2-384_5_20260203_17"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "cs2007_soil_multimodal_baselines"
MODES = ("image_only", "soil_only", "soil_raw_concat")
MODE_LABELS = {
    "image_only": "Image only",
    "soil_only": "Soil only",
    "soil_raw_concat": "Image + soil raw concat",
}
SPLITS = ("val", "test")
METRICS = (
    ("Acc@1", "top1_acc"),
    ("Acc@3", "top3_acc"),
    ("F1", "f1"),
    ("MCC", "mcc"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Report CS2007 image + soil multimodal baseline results.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Run root containing mode/seed*/metrics.json.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    return parser.parse_args()


def _read_metrics(root: Path, modes: Iterable[str], seeds: Iterable[int]) -> list[dict]:
    records = []
    for mode in modes:
        for seed in seeds:
            metrics_path = root / mode / f"seed{seed}" / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            records.append(
                {
                    "mode": mode,
                    "seed": int(seed),
                    "path": metrics_path,
                    "metrics": metrics,
                }
            )
    return records


def _validate_class_names(records: list[dict]) -> list[str]:
    first = records[0]["metrics"]["class_names"]
    for record in records:
        names = record["metrics"]["class_names"]
        if names != first:
            raise ValueError(
                "Class-name order differs across runs; this report expects aligned confusion matrices. "
                f"First mismatch: {record['path']}"
            )
    return [str(name) for name in first]


def _metric_rows(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        metrics = record["metrics"]
        for split in SPLITS:
            split_metrics = metrics[split]
            for metric_name, metric_key in METRICS:
                rows.append(
                    {
                        "mode": record["mode"],
                        "mode_label": MODE_LABELS.get(record["mode"], record["mode"]),
                        "seed": record["seed"],
                        "split": split,
                        "metric": metric_name,
                        "value": float(split_metrics[metric_key]),
                    }
                )
    return pd.DataFrame(rows)


def _metric_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    out = (
        per_seed.groupby(["mode", "mode_label", "split", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
            count=("value", "count"),
        )
    )
    split_order = {split: idx for idx, split in enumerate(SPLITS)}
    metric_order = {metric_name: idx for idx, (metric_name, _metric_key) in enumerate(METRICS)}
    mode_order = {mode: idx for idx, mode in enumerate(MODES)}
    out["_split_order"] = out["split"].map(split_order)
    out["_metric_order"] = out["metric"].map(metric_order)
    out["_mode_order"] = out["mode"].map(mode_order)
    return (
        out.sort_values(["_split_order", "_metric_order", "_mode_order"])
        .drop(columns=["_split_order", "_metric_order", "_mode_order"])
        .reset_index(drop=True)
    )


def _aggregate_cms(records: list[dict], modes: Iterable[str]) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_mode[record["mode"]].append(record)

    for mode in modes:
        for split in SPLITS:
            cm_sum = None
            for record in by_mode[mode]:
                cm = np.asarray(record["metrics"][split]["cm"], dtype=np.int64)
                cm_sum = cm if cm_sum is None else cm_sum + cm
            out[(mode, split)] = cm_sum
    return out


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    return cm.astype(float) / row_sums


def _precision_recall_table(cms: dict[tuple[str, str], np.ndarray], class_names: list[str], split: str) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        cm = cms[(mode, split)]
        support = cm.sum(axis=1).astype(float)
        predicted = cm.sum(axis=0).astype(float)
        correct = np.diag(cm).astype(float)
        recall = np.divide(correct, support, out=np.full_like(correct, np.nan), where=support != 0)
        precision = np.divide(correct, predicted, out=np.full_like(correct, np.nan), where=predicted != 0)
        for idx, habitat in enumerate(class_names):
            rows.append(
                {
                    "split": split,
                    "mode": mode,
                    "mode_label": MODE_LABELS.get(mode, mode),
                    "habitat": habitat,
                    "support": int(support[idx]),
                    "recall": float(recall[idx]) if not math.isnan(float(recall[idx])) else np.nan,
                    "precision": float(precision[idx]) if not math.isnan(float(precision[idx])) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _comparison_recall_table(per_class: pd.DataFrame, split: str) -> pd.DataFrame:
    recall = per_class[per_class["split"] == split].pivot(index=["habitat", "support"], columns="mode", values="recall")
    recall = recall.reset_index()
    recall["soil_raw_minus_image"] = recall["soil_raw_concat"] - recall["image_only"]
    recall["soil_only_minus_image"] = recall["soil_only"] - recall["image_only"]
    return recall.sort_values("support", ascending=False)


def _top_confusions(cms: dict[tuple[str, str], np.ndarray], class_names: list[str], split: str, top_n: int = 12) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        cm = cms[(mode, split)]
        for true_idx, true_name in enumerate(class_names):
            support = int(cm[true_idx].sum())
            for pred_idx, pred_name in enumerate(class_names):
                if true_idx == pred_idx:
                    continue
                count = int(cm[true_idx, pred_idx])
                if count <= 0:
                    continue
                rows.append(
                    {
                        "split": split,
                        "mode": mode,
                        "mode_label": MODE_LABELS.get(mode, mode),
                        "true_habitat": true_name,
                        "predicted_habitat": pred_name,
                        "count": count,
                        "true_support": support,
                        "rate_within_true": count / support if support else np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    return (
        frame.sort_values(["mode", "count"], ascending=[True, False])
        .groupby("mode", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _fmt_decimal(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_mean_std(mean, std, digits: int = 3) -> str:
    return f"{_fmt_decimal(mean, digits)} +/- {_fmt_decimal(std, digits)}"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value) -> str:
        return str(value).replace("\n", "<br>")

    lines = [
        "| " + " | ".join(cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def _summary_markdown_table(summary: pd.DataFrame, split: str) -> str:
    split_df = summary[summary["split"] == split]
    rows = []
    for mode in MODES:
        mode_df = split_df[split_df["mode"] == mode]
        row = [MODE_LABELS[mode]]
        for metric_name, _metric_key in METRICS:
            metric_row = mode_df[mode_df["metric"] == metric_name].iloc[0]
            row.append(_fmt_mean_std(metric_row["mean"], metric_row["std"]))
        rows.append(row)
    return _markdown_table(["Mode", "Acc@1", "Acc@3", "F1", "MCC"], rows)


def _recall_markdown_table(recall_compare: pd.DataFrame) -> str:
    rows = []
    for row in recall_compare.sort_values("support", ascending=False).itertuples(index=False):
        rows.append(
            [
                row.habitat,
                int(row.support),
                _fmt_decimal(row.image_only),
                _fmt_decimal(row.soil_only),
                _fmt_decimal(row.soil_raw_concat),
                _fmt_decimal(row.soil_raw_minus_image),
            ]
        )
    return _markdown_table(
        ["Habitat", "Support", "Image recall", "Soil recall", "Raw-concat recall", "Raw - image"],
        rows,
    )


def _plot_metric_bars(summary: pd.DataFrame, split: str, out_path: Path) -> None:
    metric_names = [m[0] for m in METRICS]
    x = np.arange(len(metric_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for offset_idx, mode in enumerate(MODES):
        mode_df = summary[(summary["split"] == split) & (summary["mode"] == mode)].set_index("metric").loc[metric_names]
        offset = (offset_idx - 1) * width
        ax.bar(
            x + offset,
            mode_df["mean"].to_numpy(),
            width=width,
            yerr=mode_df["std"].to_numpy(),
            capsize=4,
            label=MODE_LABELS[mode],
        )
    ax.set_title(f"{split.upper()} performance across seeds")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _wrapped_labels(class_names: list[str]) -> list[str]:
    return ["\n".join(textwrap.wrap(name, width=18)) for name in class_names]


def _plot_normalized_cm(cm: np.ndarray, class_names: list[str], title: str, out_path: Path) -> None:
    cm_norm = _row_normalize(cm)
    labels = _wrapped_labels(class_names)
    annot = np.array([["0" if value == 0 else f"{value:.2f}" for value in row] for row in cm_norm])
    fig, ax = plt.subplots(figsize=(15, 12))
    if sns is not None:
        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Row-normalized rate"},
        )
    else:
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, label="Row-normalized rate")
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_recall_compare(recall_compare: pd.DataFrame, split: str, out_path: Path) -> None:
    frame = recall_compare[recall_compare["support"] > 0].copy()
    frame = frame.sort_values("support", ascending=True)
    y = np.arange(len(frame))
    height = 0.24
    fig, ax = plt.subplots(figsize=(10, max(7, 0.42 * len(frame))))
    for offset_idx, mode in enumerate(MODES):
        offset = (offset_idx - 1) * height
        ax.barh(y + offset, frame[mode].to_numpy(), height=height, label=MODE_LABELS[mode])
    ax.set_yticks(y)
    ax.set_yticklabels(frame["habitat"].tolist())
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_title(f"{split.upper()} per-habitat recall from aggregated confusion matrices")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_delta_recall(recall_compare: pd.DataFrame, split: str, out_path: Path) -> None:
    frame = recall_compare[recall_compare["support"] > 0].copy()
    frame = frame.sort_values("soil_raw_minus_image", ascending=True)
    colors = ["#b9483f" if value < 0 else "#31688e" for value in frame["soil_raw_minus_image"]]
    fig, ax = plt.subplots(figsize=(10, max(7, 0.42 * len(frame))))
    ax.barh(frame["habitat"], frame["soil_raw_minus_image"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Recall difference")
    ax.set_title(f"{split.upper()} recall delta: image + soil raw concat minus image only")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_outputs(args, records: list[dict], class_names: list[str]) -> Path:
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    per_seed = _metric_rows(records)
    summary = _metric_summary(per_seed)
    cms = _aggregate_cms(records, args.modes)
    per_class = pd.concat([_precision_recall_table(cms, class_names, split) for split in SPLITS], ignore_index=True)
    recall_val = _comparison_recall_table(per_class, "val")
    recall_test = _comparison_recall_table(per_class, "test")
    confusions_test = _top_confusions(cms, class_names, "test", top_n=12)
    confusions_val = _top_confusions(cms, class_names, "val", top_n=12)

    per_seed.to_csv(tables_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(tables_dir / "metric_summary.csv", index=False)
    per_class.to_csv(tables_dir / "per_class_precision_recall.csv", index=False)
    recall_val.to_csv(tables_dir / "val_recall_comparison.csv", index=False)
    recall_test.to_csv(tables_dir / "test_recall_comparison.csv", index=False)
    pd.concat([confusions_val, confusions_test], ignore_index=True).to_csv(tables_dir / "top_confusions.csv", index=False)

    _plot_metric_bars(summary, "val", figures_dir / "validation_metrics.png")
    _plot_metric_bars(summary, "test", figures_dir / "test_metrics.png")
    _plot_recall_compare(recall_test, "test", figures_dir / "test_per_habitat_recall.png")
    _plot_delta_recall(recall_test, "test", figures_dir / "test_raw_concat_minus_image_recall.png")
    for split in SPLITS:
        for mode in MODES:
            _plot_normalized_cm(
                cms[(mode, split)],
                class_names,
                f"{split.upper()} normalized confusion matrix: {MODE_LABELS[mode]}",
                figures_dir / f"{split}_normalized_cm_{mode}.png",
            )

    report_path = output_dir / "report.md"
    report_text = _build_report(
        args=args,
        output_dir=output_dir,
        summary=summary,
        recall_val=recall_val,
        recall_test=recall_test,
        confusions_test=confusions_test,
        class_names=class_names,
    )
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def _relative(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def _mode_metric(summary: pd.DataFrame, mode: str, split: str, metric: str, stat: str = "mean") -> float:
    row = summary[(summary["mode"] == mode) & (summary["split"] == split) & (summary["metric"] == metric)].iloc[0]
    return float(row[stat])


def _dominant_confusion_table(confusions: pd.DataFrame, mode: str, limit: int = 6) -> str:
    frame = confusions[confusions["mode"] == mode].head(limit)
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            [
                row.true_habitat,
                row.predicted_habitat,
                int(row.count),
                _fmt_decimal(row.rate_within_true),
            ]
        )
    return _markdown_table(["True habitat", "Predicted as", "Count", "Rate within true class"], rows)


def _build_report(
    args,
    output_dir: Path,
    summary: pd.DataFrame,
    recall_val: pd.DataFrame,
    recall_test: pd.DataFrame,
    confusions_test: pd.DataFrame,
    class_names: list[str],
) -> str:
    fig = lambda name: _relative(output_dir / "figures" / name, output_dir)
    table = lambda name: _relative(output_dir / "tables" / name, output_dir)

    raw_test_acc = _mode_metric(summary, "soil_raw_concat", "test", "Acc@1")
    image_test_acc = _mode_metric(summary, "image_only", "test", "Acc@1")
    soil_test_acc = _mode_metric(summary, "soil_only", "test", "Acc@1")
    raw_val_acc = _mode_metric(summary, "soil_raw_concat", "val", "Acc@1")
    image_val_acc = _mode_metric(summary, "image_only", "val", "Acc@1")

    test_supported = recall_test[recall_test["support"] > 0].copy()
    raw_gains = test_supported.sort_values("soil_raw_minus_image", ascending=False).head(5)
    raw_losses = test_supported.sort_values("soil_raw_minus_image", ascending=True).head(5)
    val_supported = recall_val[recall_val["support"] > 0].copy()
    val_gains = val_supported.sort_values("soil_raw_minus_image", ascending=False).head(5)
    val_losses = val_supported.sort_values("soil_raw_minus_image", ascending=True).head(5)
    soil_only_test = (
        test_supported.loc[:, ["habitat", "support", "soil_only"]]
        .sort_values("soil_only", ascending=False)
        .head(6)
    )

    gain_rows = [
        [
            row.habitat,
            int(row.support),
            _fmt_decimal(row.image_only),
            _fmt_decimal(row.soil_raw_concat),
            _fmt_decimal(row.soil_raw_minus_image),
        ]
        for row in raw_gains.itertuples(index=False)
    ]
    loss_rows = [
        [
            row.habitat,
            int(row.support),
            _fmt_decimal(row.image_only),
            _fmt_decimal(row.soil_raw_concat),
            _fmt_decimal(row.soil_raw_minus_image),
        ]
        for row in raw_losses.itertuples(index=False)
    ]
    val_gain_rows = [
        [
            row.habitat,
            int(row.support),
            _fmt_decimal(row.image_only),
            _fmt_decimal(row.soil_raw_concat),
            _fmt_decimal(row.soil_raw_minus_image),
        ]
        for row in val_gains.itertuples(index=False)
    ]
    val_loss_rows = [
        [
            row.habitat,
            int(row.support),
            _fmt_decimal(row.image_only),
            _fmt_decimal(row.soil_raw_concat),
            _fmt_decimal(row.soil_raw_minus_image),
        ]
        for row in val_losses.itertuples(index=False)
    ]
    soil_rows = [
        [row.habitat, int(row.support), _fmt_decimal(row.soil_only)]
        for row in soil_only_test.itertuples(index=False)
    ]

    unsupported_test = recall_test[recall_test["support"] == 0]["habitat"].tolist()
    unsupported_note = (
        "The aggregated test confusion matrices have zero support for "
        + ", ".join(f"`{name}`" for name in unsupported_test)
        + "; those classes are not interpreted in the test confusion analysis."
        if unsupported_test
        else "Every class has non-zero support in the aggregated test confusion matrices."
    )

    lines = [
        "# CS2007 Image + Soil Multimodal Baseline Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        f"- Run root: `{args.root}`",
        f"- Modes: {', '.join(f'`{mode}`' for mode in args.modes)}",
        f"- Seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- Classes: {len(class_names)} project habitat labels, with no rare-label grouping.",
        "- Metrics are reported as mean +/- sample standard deviation across seeds.",
        "- `val` and `test` metrics are from the saved best validation checkpoint in each run.",
        "",
        "## Executive Summary",
        "",
        (
            f"`soil_raw_concat` is the strongest of the three requested baselines on the aggregate metrics: "
            f"validation Acc@1 is {_fmt_decimal(raw_val_acc)} vs {_fmt_decimal(image_val_acc)} for `image_only`, "
            f"and test Acc@1 is {_fmt_decimal(raw_test_acc)} vs {_fmt_decimal(image_test_acc)}."
        ),
        (
            f"The gain from adding raw soil chemistry to image features is modest: "
            f"{100 * (raw_test_acc - image_test_acc):.2f} percentage points on test Acc@1. "
            "This suggests the image embeddings carry most of the discriminative signal, while soil chemistry adds a small amount of complementary information."
        ),
        (
            f"`soil_only` is much weaker overall, with test Acc@1 {_fmt_decimal(soil_test_acc)}. "
            "It still performs reasonably on several broad soil-linked habitats, but it lacks the visual signal needed for many habitat distinctions."
        ),
        "",
        "## Aggregate Metrics",
        "",
        "### Validation",
        "",
        _summary_markdown_table(summary, "val"),
        "",
        "![Validation metrics](figures/validation_metrics.png)",
        "",
        "### Test",
        "",
        _summary_markdown_table(summary, "test"),
        "",
        "![Test metrics](figures/test_metrics.png)",
        "",
        "Detailed CSV outputs:",
        "",
        f"- [`per_seed_metrics.csv`]({table('per_seed_metrics.csv')})",
        f"- [`metric_summary.csv`]({table('metric_summary.csv')})",
        "",
        "## Habitat-Level Confusion Analysis",
        "",
        unsupported_note,
        "",
        "The tables below use recall from confusion matrices aggregated across seeds. Recall is the fraction of true samples of a habitat predicted correctly.",
        "",
        "### Test Per-Habitat Recall",
        "",
        _recall_markdown_table(recall_test),
        "",
        "![Test per-habitat recall](figures/test_per_habitat_recall.png)",
        "",
        "![Test recall delta](figures/test_raw_concat_minus_image_recall.png)",
        "",
        "### Where Raw Soil Concatenation Helps Most on Test",
        "",
        _markdown_table(
            ["Habitat", "Support", "Image recall", "Raw-concat recall", "Delta"],
            gain_rows,
        ),
        "",
        "### Where Raw Soil Concatenation Hurts Most on Test",
        "",
        _markdown_table(
            ["Habitat", "Support", "Image recall", "Raw-concat recall", "Delta"],
            loss_rows,
        ),
        "",
        "### Validation Recall Deltas for Raw Soil Concatenation",
        "",
        "Largest validation gains over `image_only`:",
        "",
        _markdown_table(
            ["Habitat", "Support", "Image recall", "Raw-concat recall", "Delta"],
            val_gain_rows,
        ),
        "",
        "Largest validation losses relative to `image_only`:",
        "",
        _markdown_table(
            ["Habitat", "Support", "Image recall", "Raw-concat recall", "Delta"],
            val_loss_rows,
        ),
        "",
        "### Soil-Only Strengths on Test",
        "",
        _markdown_table(["Habitat", "Support", "Soil-only recall"], soil_rows),
        "",
        (
            "`soil_only` is strongest for `Arable and Horticulture`, `Improved Grassland`, and `Bog`, "
            "which is plausible because these classes have clearer soil or land-management associations. "
            "It is weak for visually and structurally distinct classes such as woodland separation, wetland classes, bracken, and rare coastal or upland classes."
        ),
        "",
        "### Dominant Test Confusions",
        "",
        "#### Image Only",
        "",
        _dominant_confusion_table(confusions_test, "image_only"),
        "",
        "#### Soil Only",
        "",
        _dominant_confusion_table(confusions_test, "soil_only"),
        "",
        "#### Image + Soil Raw Concat",
        "",
        _dominant_confusion_table(confusions_test, "soil_raw_concat"),
        "",
        "### Normalized Test Confusion Matrices",
        "",
        "#### Image Only",
        "",
        "![Image-only test normalized confusion matrix](figures/test_normalized_cm_image_only.png)",
        "",
        "#### Soil Only",
        "",
        "![Soil-only test normalized confusion matrix](figures/test_normalized_cm_soil_only.png)",
        "",
        "#### Image + Soil Raw Concat",
        "",
        "![Raw-concat test normalized confusion matrix](figures/test_normalized_cm_soil_raw_concat.png)",
        "",
        "## Interpretation",
        "",
        "- `image_only` is already strong, especially for `Arable and Horticulture`, woodland classes, `Improved Grassland`, and `Bog`.",
        "- `soil_raw_concat` improves the aggregate metrics slightly and shows clearer test-recall gains for `Acid Grassland`, `Bog`, `Coniferous Woodland`, and `Improved Grassland`.",
        "- `soil_raw_concat` hurts some classes, especially low-support classes such as `Montane` and `Bracken`, and also reduces recall for `Broadleaved Mixed and Yew Woodland` and `Dwarf Shrub Heath` on test.",
        "- `soil_only` mainly learns broad edaphic or management-associated habitats and collapses many visually distinct classes into common grassland, arable, bog, or woodland predictions.",
        "- The dominant remaining confusions are ecological/visual neighbors: `Bog` vs `Acid Grassland`, `Dwarf Shrub Heath` vs `Bog`, `Neutral Grassland` vs `Improved Grassland`, and `Improved Grassland` vs `Neutral Grassland`.",
        "",
        "## Validation Confusion Outputs",
        "",
        "Validation confusion matrices and per-class tables are saved for inspection:",
        "",
        f"- [`val_recall_comparison.csv`]({table('val_recall_comparison.csv')})",
        f"- [`test_recall_comparison.csv`]({table('test_recall_comparison.csv')})",
        f"- [`per_class_precision_recall.csv`]({table('per_class_precision_recall.csv')})",
        f"- [`top_confusions.csv`]({table('top_confusions.csv')})",
        "- [`figures/val_normalized_cm_image_only.png`](figures/val_normalized_cm_image_only.png)",
        "- [`figures/val_normalized_cm_soil_only.png`](figures/val_normalized_cm_soil_only.png)",
        "- [`figures/val_normalized_cm_soil_raw_concat.png`](figures/val_normalized_cm_soil_raw_concat.png)",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    root = args.root
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    args.root = root
    records = _read_metrics(root, args.modes, args.seeds)
    class_names = _validate_class_names(records)
    report_path = _write_outputs(args, records, class_names)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()

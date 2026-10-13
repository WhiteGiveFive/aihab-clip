"""Render a concise report from authenticated, already scored test outputs."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPORT_DIR = Path(__file__).resolve().parent
REPO = REPORT_DIR.parents[3]
BASE = REPO / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
POLICY = "relaxed__rf__primary_full727__plots_20__harm_0p7"
VARIANTS = ("primary_full727", "ablation_no_pairs79", "ablation_numeric25")


def read_source(study):
    root = BASE / study
    relative = "best_model/classification.json"
    payload = (root / "scores" / relative).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    receipt = json.loads((root / "scores/receipt.json").read_text())
    assert digest == receipt["output_file_sha256"][relative]
    return json.loads(payload), digest


def draw_cm(cm, label_list, sink=None):
    """Canonical CS plotting behavior, with a local sink and no logging."""
    if not isinstance(cm, np.ndarray):
        torch_module = sys.modules.get("torch")
        if torch_module is not None and isinstance(cm, torch_module.Tensor):
            cm = cm.detach().cpu().numpy()
        else:
            try:
                cm = np.asarray(cm)
            except Exception:
                return
    assert cm.shape == (len(label_list), len(label_list))

    def plot(values, normalized):
        fig, ax = plt.subplots(figsize=(15, 12))
        annotations = (
            np.array([["0" if v == 0 else f"{v:.2f}" for v in row] for row in values])
            if normalized
            else values.astype(int)
        )
        sns.heatmap(
            values,
            annot=annotations,
            fmt="" if normalized else "d",
            cmap="Blues",
            xticklabels=label_list,
            yticklabels=label_list,
            ax=ax,
            vmin=0,
            vmax=1 if normalized else None,
            annot_kws={"fontsize": 7},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix L3" + (" (Normalized)" if normalized else ""))
        fig.tight_layout()
        if sink is not None:
            sink(fig, values, normalized)
        plt.close(fig)

    plot(cm, False)
    row_sums = cm.sum(axis=1, keepdims=True).copy()
    row_sums[row_sums == 0] = 1
    plot(cm / row_sums, True)


def main():
    rf, rf_hash = read_source("relaxed_guardrails_rf_v1")
    lr, lr_hash = read_source("relaxed_guardrails_lr_v1")
    declaration = json.loads(
        (BASE / "relaxed_guardrails_rf_v1/declaration.json").read_text()
    )
    ontology = declaration["ontology"]
    labels = [row["label_name"] for row in ontology]
    assert [row["dense_id"] for row in ontology] == list(range(18))
    class_map = next(
        ast.literal_eval(node.value)
        for node in ast.parse((REPO / "data/__init__.py").read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REASSIGN_LABEL_NAME_L3"
            for target in node.targets
        )
    )
    assert labels == [class_map[row["canonical_l3_id"]] for row in ontology]
    records = {
        "geo": rf["experts"]["geo"],
        "image": rf["experts"]["image"],
        "fusion": rf["experts"]["raw"],
        "router": rf["policies"][POLICY],
    }
    names = {
        "geo": "Geo expert",
        "image": "Image expert",
        "fusion": "Fusion expert",
        "router": "Fusion + router correction",
    }
    matrices, summary = {}, {}
    support = None
    for key, record in records.items():
        cm = np.asarray(
            record["pooled_seed_realizations"]["confusion_matrix"], dtype=np.int64
        )
        seed_cms = [
            np.asarray(
                record["per_seed"][str(seed)]["confusion_matrix"], dtype=np.int64
            )
            for seed in range(1, 5)
        ]
        assert cm.shape == (18, 18) and cm.sum() == 5388
        assert np.array_equal(cm, sum(seed_cms))
        assert all(np.array_equal(m.sum(axis=1) * 4, cm.sum(axis=1)) for m in seed_cms)
        if support is not None:
            assert np.array_equal(support, cm.sum(axis=1))
        support = cm.sum(axis=1)
        assert [
            entry["label_name"]
            for entry in record["pooled_seed_realizations"]["per_class"]
        ] == labels
        mean = record["seed_summary"]
        accuracy = mean["accuracy"]["mean"]
        assert abs(accuracy - np.trace(cm) / cm.sum()) < 1e-12
        matrices[key] = cm
        summary[key] = {
            "acc1_percent": accuracy * 100,
            "weighted_f1": mean["weighted_f1"]["mean"],
        }

    ablations = {}
    for family, source, prefix in (
        ("LR", lr, "relaxed__"),
        ("RF", rf, "relaxed__rf__"),
    ):
        ablations[family] = {}
        for variant in VARIANTS:
            key = prefix + variant + "__plots_20__harm_0p7"
            record = source["policies"][key]
            acc = record["seed_summary"]["accuracy"]["mean"] * 100
            ablations[family][variant] = {
                "acc1_percent": acc,
                "gain_pp": acc - summary["fusion"]["acc1_percent"],
            }
    all_acc = [
        record["seed_summary"]["accuracy"]["mean"] * 100
        for source in (rf, lr)
        for record in source["policies"].values()
    ]
    assert summary["router"]["acc1_percent"] == max(all_acc)
    assert np.trace(matrices["router"] - matrices["fusion"]) == 46

    figure_dir = REPORT_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)
    ticklabels = [
        name + (" [no test examples]" if support[i] == 0 else "")
        for i, name in enumerate(labels)
    ]
    for key, cm in matrices.items():

        def sink(fig, values, normalized, key=key):
            fig.suptitle(
                names[key] + f" — Acc@1 {summary[key]['acc1_percent']:.2f}%",
                y=1.005,
                fontsize=15,
            )
            suffix = "normalized" if normalized else "counts"
            fig.savefig(
                figure_dir / f"{key}_{suffix}.png", dpi=150, bbox_inches="tight"
            )

        draw_cm(cm, ticklabels, sink)

    fig, axes = plt.subplots(2, 2, figsize=(22, 20))
    row_labels = [
        str(i)
        + ": "
        + "\n".join(textwrap.wrap(name, 22))
        + ("\n[no test examples]" if support[i] == 0 else "")
        for i, name in enumerate(labels)
    ]
    for ax, (key, cm) in zip(axes.flat, matrices.items()):
        norm = cm / np.where(support[:, None] == 0, 1, support[:, None])
        sns.heatmap(
            norm,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=list(range(18)),
            yticklabels=row_labels,
            square=True,
        )
        for i in range(18):
            for j in range(18):
                if norm[i, j] >= 0.005:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{norm[i,j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "#172033",
                    )
        ax.set_title(
            names[key] + f"\nAcc@1: {summary[key]['acc1_percent']:.2f}%", fontsize=16
        )
        ax.set_xlabel("Predicted habitat (ID matches row labels)")
        ax.set_ylabel("True habitat")
        ax.tick_params(labelsize=9)
    fig.suptitle("Habitat classification on the cleaned test set", fontsize=22, y=0.997)
    fig.text(
        0.5,
        0.975,
        "Row-normalized confusion matrices; shared 0–1 colour scale; diagonal entries show habitat recall.",
        ha="center",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_dir / "confusion_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert not plt.get_fignums()
    metrics = {
        "source_sha256": {"rf_classification": rf_hash, "lr_classification": lr_hash},
        "illustrative_policy": POLICY,
        "aggregation": "Mean per-seed metrics; confusion matrices pool four equal-size seed realizations, not an ensemble",
        "unique_images": 1347,
        "unique_plots": 531,
        "seed_image_rows": 5388,
        "summary": summary,
        "ablations": ablations,
        "class_names": labels,
        "confusion_matrices": {key: cm.tolist() for key, cm in matrices.items()},
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "ablations": ablations,
                "figure_count": 9,
                "validated": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

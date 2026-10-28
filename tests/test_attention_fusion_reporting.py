"""Verify fixed-class plotting, mean-confusion semantics and complete report coverage."""

import copy
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from multimodal.attention_fusion_reporting import draw_cm, generate_report
from multimodal.attention_fusion_training import classification_metrics


def analysis_fixture():
    labels = np.array([0, 0, 0, 1, 1])
    arms, candidates = [], []
    for index, name in enumerate(("projected_concat", "attention_two_token")):
        per_seed, histories = {}, {}
        for seed in range(1, 5):
            predictions = np.array([0, 0, index, 0, 1])
            logits = np.zeros((5, 18))
            logits[np.arange(5), predictions] = 2
            metrics, _ = classification_metrics(labels, logits)
            per_seed[str(seed)] = {"metrics": metrics, "fit_id": f"{name}_seed{seed}",
                                   "parameters": 1000 + index * 500,
                                   "runtime_seconds": 10 + seed + index,
                                   "peak_cuda_bytes": 2**20 * (seed + 1),
                                   "peak_process_rss_bytes": 2**30}
            histories[str(seed)] = [{"epoch": epoch,
                                    "train": {**metrics, "loss": 2 / epoch},
                                    "validation": {**metrics, "loss": 2.5 / epoch}}
                                   for epoch in range(1, 4)]
        arm = {"arm_id": f"{name}__geo_native", "candidate_id": name, "geo_mode": "native",
               "selected_learning_rate": 0.001, "learning_rate": 0.001, "per_seed": per_seed,
               "mean": {metric: metrics[metric] for metric in ("acc1", "acc3", "weighted_f1", "mcc")},
               "sample_std": {metric: 0.0 for metric in ("acc1", "acc3", "weighted_f1", "mcc")},
               "histories_by_seed": histories}
        arms.append(arm)
        candidates.extend([{**copy.deepcopy(arm), "learning_rate": rate} for rate in (0.0001, 0.0003, 0.001)])
    difference = {metric: arms[1]["mean"][metric] - arms[0]["mean"][metric] for metric in arms[0]["mean"]}
    return {"schema_version": "synthetic.analysis.v1", "study_id": "synthetic_reporting_only",
            "class_order": list(range(18)), "class_labels": [f"Habitat {i}" for i in range(18)],
            "training_seeds": [1, 2, 3, 4], "validation_rows": 5, "validation_plots": 3,
            "validation_class_support_images": [3, 2] + [0] * 16,
            "validation_class_support_plots": [2, 1] + [0] * 16,
            "selected_arms": arms, "lr_candidates": candidates, "primary_contrast_id": "primary",
            "attention_claims": {"native": {"supported": False, "required_contrasts": ["primary"]}},
            "contrasts": [{"contrast_id": "primary", "kind": "primary",
                           "lhs_arm_id": arms[1]["arm_id"], "rhs_arm_id": arms[0]["arm_id"],
                           "mean_difference": difference,
                           "per_seed": {str(seed): difference.copy() for seed in range(1, 5)},
                           "interval95": {metric: [value - 0.3, value + 0.3] for metric, value in difference.items()},
                           "rescues_harms_by_seed": {str(seed): {"rescued": 0, "harmed": 1, "net": -1} for seed in range(1, 5)},
                           "category": "inconclusive"}]}


@pytest.mark.parametrize("as_tensor", [False, True])
def test_canonical_confusion_sink_preserves_counts_zero_rows_and_closes_figures(as_tensor):
    counts = np.array([[2, 1], [0, 0]])
    emitted = []
    before = set(plt.get_fignums())

    def sink(fig, values, normalized):
        emitted.append((values.copy(), normalized))
        assert tuple(fig.get_size_inches()) == (15, 12)
        axis = fig.axes[0]
        assert axis.get_xlabel() == "Predicted" and axis.get_ylabel() == "True"
        annotations = [text.get_text() for text in axis.texts]
        assert annotations == (["0.67", "0.33", "0", "0"] if normalized else ["2", "1", "0", "0"])

    draw_cm(torch.tensor(counts) if as_tensor else counts, ["A", "B"], sink)
    assert len(emitted) == 2 and emitted[0][1] is False and emitted[1][1] is True
    assert np.array_equal(emitted[0][0], counts)
    assert np.array_equal(emitted[1][0][1], [0, 0])
    assert set(plt.get_fignums()) == before


def test_confusion_rejects_fractional_counts_instead_of_truncating_them():
    with pytest.raises(ValueError, match="integer counts"):
        draw_cm([[0.5, 0.5], [0, 0]], ["A", "B"])


def test_report_covers_all_selected_seeds_and_reuses_unchanged_artifacts(tmp_path, monkeypatch):
    import multimodal.attention_fusion_reporting as reporting

    analysis = analysis_fixture()
    # Exercise the complete report workflow while avoiding duplicate image
    # encoding in this test; canonical heatmap rendering is tested above.
    def save(fig, stem, dpi):
        stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            stem.with_suffix(f".{extension}").write_text(f"test-only rendered figure: {stem.name}\n")
        return []

    monkeypatch.setattr(reporting, "_save", save)
    output = tmp_path / "report"
    manifest = generate_report(analysis, output, png_dpi=72)
    assert manifest["selected_arms"] == 2 and manifest["selected_seeds"] == 8
    for arm in analysis["selected_arms"]:
        folder = output / "figures/confusion" / arm["arm_id"]
        assert len(list(folder.glob("*.png"))) == 10
        assert (folder / "mean_normalized.pdf").exists()
        assert (output / "tables/confusion" / f"{arm['arm_id']}__mean_counts.csv").exists()
    text = (output / "report.md").read_text()
    assert "conditional" in text and "not an ensemble" in text
    assert "Not established" in text and "not simultaneous intervals" in text
    assert "**24 declared development fits**" in text and "**6 arm/LR summaries**" in text
    for table in ("all_fits", "learning_rates", "selected_arms", "paired_contrasts"):
        assert f"../tables/{table}.csv" in text
    assert "../selection.json" in text and "../analysis.json" in text
    assert "[Learning curves](#learning-curves)" in text
    contrasts = (output / "per_seed_contrasts.md").read_text()
    assert all(f"| primary | {seed} |" in contrasts for seed in range(1, 5))
    import csv
    with (output / "tables/contrast_per_class.csv").open() as stream:
        classes = list(csv.DictReader(stream))
    assert len(classes) == 18 and classes[2]["mean_recall_difference"] == ""
    assert float(classes[0]["mean_correct_count_difference"]) == -1.0
    per_seed = (output / "per_seed_metrics.md").read_text()
    for arm in analysis["selected_arms"]:
        for seed in range(1, 5):
            assert arm["per_seed"][str(seed)]["fit_id"] in per_seed
    assert not plt.get_fignums()
    before = {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
              for path in output.rglob("*") if path.is_file()}
    assert generate_report(analysis, output, png_dpi=72) == manifest
    after = {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
             for path in output.rglob("*") if path.is_file()}
    assert before == after
    changed = copy.deepcopy(analysis)
    changed["study_id"] = "changed_study"
    with pytest.raises(ValueError, match="changed"):
        generate_report(changed, output, png_dpi=72)


def test_report_rejects_misaligned_support_and_missing_seeds_before_writing(tmp_path):
    analysis = analysis_fixture()
    analysis["selected_arms"][0]["per_seed"].pop("4")
    with pytest.raises(ValueError, match="missing a seed"):
        generate_report(analysis, tmp_path / "missing")
    analysis = analysis_fixture()
    analysis["selected_arms"][0]["per_seed"]["1"]["metrics"]["confusion_matrix"][0][0] += 1
    with pytest.raises(ValueError, match="shared validation cohort"):
        generate_report(analysis, tmp_path / "misaligned")
    assert not list(tmp_path.iterdir())

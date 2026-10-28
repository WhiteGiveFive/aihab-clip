"""R4 stage integration and interpretation checks without image/model access."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from multimodal import attention_fusion_router_analysis as analysis
from multimodal.attention_fusion_router_reporting import habitat_changes, probability_metrics
from multimodal.attention_fusion_geometry_router import sha256, write_json
from multimodal.geometry_routing_policy import evaluate_actions, finite_policy


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "study"
    stage = root / "r3_validation_v1"
    stage.mkdir(parents=True)
    classes = [f"class{i}" for i in range(18)]
    truth = np.zeros(10, dtype=np.int64)
    fusion = np.array([1, 0, 1, 1, 1, 1, 1, 0, 1, 0])
    geo = np.array([0, 1, 2, 0, 0, 0, 0, 0, 1, 1])
    supported = np.ones(10, dtype=bool)
    supported[6] = False
    geo_score = np.array([.4, .4, .4, .4, 0, -.2, np.nan, 0, 0, -.1])
    image_scores = {"pretrained": np.array([.5, .5, -.3, 0, .3, -.3, np.nan, 0, 0, .1]),
                    "checkpoint_matched": np.array([-.2, -.2, .2, .2, .2, -.5, np.nan, 0, 0, 0])}
    baseline = []
    evaluations = []
    frames = {}
    def metrics(pred):
        probabilities = np.full((len(pred), 18), .01)
        probabilities[np.arange(len(pred)), pred] = .83
        return probability_metrics(truth, probabilities, classes)[0]
    for checkpoint in ("final", "best"):
        for expert, pred in (("fusion", fusion), ("geo", geo)):
            baseline.append({"checkpoint_policy": checkpoint, "expert": expert, "metrics": metrics(pred)})
        for arm, image in image_scores.items():
            cid = f"{checkpoint}__{arm}"
            for rule in ("fixed_zero", "selected"):
                threshold = 0 if rule == "fixed_zero" else .3
                score = np.minimum(image, geo_score)
                actions = supported & (fusion != geo) & (score > threshold)
                frame = pd.DataFrame({"row_uid": [f"row{i}" for i in range(10)],
                    "file": [f"image{i}.jpg" for i in range(10)], "plot_idx": [f"plot{i}" for i in range(10)],
                    "label_id_dense": truth, "label_name": ["class0"] * 10,
                    "fusion_predictions": fusion, "geo_predictions": geo,
                    "reference_plots_fusion_class": np.where(supported, 10, 3),
                    "reference_plots_geo_class": np.full(10, 10),
                    "geometry_supported": supported, "disagreement": fusion != geo,
                    "eligible": supported & (fusion != geo),
                    "d_geo_fusion": np.where(supported, .5*np.exp(geo_score), np.nan),
                    "d_geo_geo": np.where(supported, .5, np.nan),
                    "d_image_fusion": np.where(supported, .5*np.exp(image), np.nan),
                    "d_image_geo": np.where(supported, .5, np.nan),
                    "A_geo": geo_score, "A_image": image, "score": score, "action": actions,
                    "rescue": actions & (fusion != truth) & (geo == truth),
                    "harm": actions & (fusion == truth) & (geo != truth),
                    "both_wrong_action": actions & (fusion != truth) & (geo != truth),
                    "missed_rescue": ~actions & (fusion != truth) & (geo == truth)})
                routed = np.where(actions, geo, fusion)
                frame["routed_prediction"] = routed
                frames[(cid, rule)] = frame
                path = f"comparisons/{cid}/{rule}"
                directory = stage / path
                directory.mkdir(parents=True)
                pd.DataFrame(habitat_changes(truth, fusion, geo, actions,
                    frame.plot_idx.to_numpy(dtype=str), classes)).to_csv(directory / "habitat_changes.csv", index=False)
                evaluations.append({"comparison_id": cid, "rule": rule, "policy": checkpoint,
                    "routing_policy": finite_policy(threshold), "metrics": metrics(routed), "path": path,
                    "counts": evaluate_actions(actions, fusion, geo, truth, frame.plot_idx.to_numpy(dtype=str))})
    inputs = {"declaration": {"source_sha256": {}}, "stage": stage, "frames": frames,
        "bindings": {"r3_fixture": "immutable"}, "summary": {"baselines": baseline,
            "evaluations": evaluations, "class_labels": classes, "query_images": 10, "query_plots": 10}}
    config = yaml.safe_load(analysis.resolve(analysis.DEFAULT_CONFIG).read_text())
    monkeypatch.setattr(analysis, "load_config", lambda path: (config, root))
    monkeypatch.setattr(analysis, "read_inputs", lambda path: deepcopy(inputs))
    monkeypatch.setattr(analysis, "_sources", lambda path: [])
    monkeypatch.setattr(analysis, "_figures", lambda path, summary: None)
    return root, inputs


def inventory(root):
    return {str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_all_eight_analyses_and_two_geometry_pairs_preserve_inputs(bundle):
    root, inputs = bundle
    before = inventory(root)
    result = analysis.run()
    stage = Path(result["output"])
    after = inventory(root)
    assert all(after[name] == value for name, value in before.items())
    summary = json.loads((stage / "summary.json").read_text())
    assert len(summary["evaluations"]) == 8
    assert len(summary["geometry_pairs"]) == 2
    for record in summary["evaluations"]:
        assert record["missed_rescues"]["missed_rescues"] == record["counts"]["missed_rescues"]
        assert record["conditional_uncertainty"]["delta_acc1"] == record["counts"]["delta_acc1"]
        assert (stage / record["path"] / "habitat_metric_changes.csv").exists()
        assert (stage / record["path"] / "case_index.csv").exists()
    assert analysis.verify()["status"] == "R4_verified"
    before_reuse = inventory(root)
    assert analysis.run()["reused"] is True
    assert inventory(root) == before_reuse


def test_corrupted_analysis_not_overwritten(bundle):
    root, inputs = bundle
    stage = Path(analysis.run()["output"])
    path = stage / "summary.json"
    path.write_text(path.read_text() + "\n")
    before = inventory(root)
    with pytest.raises(ValueError, match="R4 artifact changed"):
        analysis.run()
    assert inventory(root) == before


def test_foreign_destination_preserved(bundle):
    root, _ = bundle
    stage = root / "r4_analysis_v1"
    stage.mkdir()
    (stage / "keep.txt").write_text("existing")
    with pytest.raises(FileNotFoundError):
        analysis.run()
    assert (stage / "keep.txt").read_text() == "existing"


def test_failure_keeps_unpublished_evidence(bundle, monkeypatch):
    root, _ = bundle
    def fail(*args):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(analysis, "_figures", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        analysis.run()
    assert not (root / "r4_analysis_v1").exists()
    assert len(list(root.glob("r4_analysis_v1.incomplete-*"))) == 1


def test_case_index_is_deterministic_and_not_score_ranked(bundle):
    _, inputs = bundle
    frame = inputs["frames"][("final__pretrained", "fixed_zero")]
    names = inputs["summary"]["class_labels"]
    expected = analysis.case_index(frame, names, 2)
    actual = analysis.case_index(frame.sample(frac=1, random_state=2), names, 2)
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))
    for outcome in expected.example_outcome.unique():
        ids = expected.loc[expected.example_outcome == outcome, "row_uid"].tolist()
        assert ids == sorted(frame.loc[frame[outcome], "row_uid"])[:2]


def test_habitat_f1_decomposition_accounts_for_false_positives(bundle):
    _, inputs = bundle
    evaluation = inputs["summary"]["evaluations"][0]
    baseline = inputs["summary"]["baselines"][0]
    habitat = pd.read_csv(inputs["stage"] / evaluation["path"] / "habitat_changes.csv")
    result = analysis.habitat_metric_changes(evaluation, baseline, habitat)
    assert result.macro_f1_delta_contribution.sum() == pytest.approx(
        evaluation["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"])
    assert result.fusion_false_positives.sum() == 7
    assert result.routed_false_positives.sum() == 7  # one rescue and one harm cancel overall
    assert result.loc[0, "fusion_f1"] != result.loc[1, "fusion_f1"]


@pytest.mark.parametrize("key,value", [("cleaned_test_allowed", True),
    ("training_allowed", True), ("threshold_reselection_allowed", True), ("expected_evaluations", 4)])
def test_analysis_scope_cannot_expand(tmp_path, key, value):
    config = yaml.safe_load(analysis.resolve(analysis.DEFAULT_CONFIG).read_text())
    config[key] = value
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="R4 configuration differs"):
        analysis.load_config(path)

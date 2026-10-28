"""R3 orchestration checks with small saved expert/score fixtures, no models."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from multimodal import attention_fusion_router_validation as validation
from multimodal.attention_fusion_geometry_router import canonical_hash, sha256, write_json, write_npz


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "r2"
    root.mkdir()
    classes = [f"class{i}" for i in range(18)]
    rows = pd.DataFrame({"row_uid": [f"q{i:02d}" for i in range(30)],
        "plot_idx": [f"qp{i}" for i in range(30)], "file": [f"q{i}.jpg" for i in range(30)],
        "label_id_dense": np.array([0] * 25 + [1] * 5),
        "label_name": ["class0"] * 25 + ["class1"] * 5})
    (root / "inputs").mkdir()
    rows.to_parquet(root / "inputs/validation_rows.parquet", index=False)
    pd.DataFrame({"row_uid": ["t0"], "plot_idx": ["tp0"]}).to_parquet(
        root / "inputs/train_rows.parquet", index=False)
    truth = rows.label_id_dense.to_numpy()
    fusion_prediction = 1 - truth
    probabilities = {}
    for name, prediction in (("fusion", fusion_prediction), ("geo", truth)):
        matrix = np.full((30, 18), 0.01)
        matrix[np.arange(30), prediction] = 0.83
        probabilities[name] = matrix
        for policy in ("final", "best"):
            write_npz(root / "inputs" / f"{name}_{policy}_predictions.npz", {
                "row_uid": rows.row_uid.to_numpy(dtype=str), "plot_idx": rows.plot_idx.to_numpy(dtype=str),
                "label": truth, "prediction": prediction, "probabilities": matrix})
    comparisons = [{"comparison_id": f"{policy}__{arm}", "policy": policy,
        "geometry_encoder": "pretrained" if arm == "pretrained" else f"joint_{policy}",
        "fusion_epoch": 50 if policy == "final" else 36, "geo_epoch": 50 if policy == "final" else 11}
        for policy in ("final", "best") for arm in ("pretrained", "checkpoint_matched")]
    config = {"prospective_r3": deepcopy(validation.RECIPE), "data": {"validation_images": 30,
                                                                       "validation_plots": 30}}
    declaration = {"config": config, "config_sha256": canonical_hash(config),
        "class_labels": classes, "comparisons": comparisons,
        "result_assurance": "provisional_unresolved_memory_integrity", "source_sha256": {},
        "input_sha256": {str(p.relative_to(root)): sha256(p) for p in (root / "inputs").iterdir()}}
    write_json(root / "protocol/declaration.json", declaration)
    write_json(root / "protocol/resolved_config.json", config)
    for comparison in comparisons:
        score = np.full(30, 0.5 if comparison["geometry_encoder"] == "pretrained" else -0.5)
        write_npz(root / "scores" / f"{comparison['comparison_id']}.npz", {
            "row_uid": rows.row_uid.to_numpy(dtype=str), "plot_idx": rows.plot_idx.to_numpy(dtype=str),
            "fusion_predictions": fusion_prediction, "geo_predictions": truth,
            "reference_plots_fusion_class": np.full(30, 5), "reference_plots_geo_class": np.full(30, 5),
            "geometry_supported": np.ones(30, dtype=bool), "disagreement": np.ones(30, dtype=bool),
            "eligible": np.ones(30, dtype=bool), "score": score, "A_image": score, "A_geo": score})
    for name in ("geo", "pretrained", "joint_final", "joint_best"):
        write_npz(root / "scores" / f"{name}_class_distances.npz", {"distances": np.ones((30, 18))})
    for name in ("pretrained", "joint_final", "joint_best"):
        write_json(root / "embeddings" / name / "manifest.json", {"fixture": True})
    manifest = {"declaration_sha256": sha256(root / "protocol/declaration.json"),
        "query_labels_used_for_geometry": False, "thresholds_selected": False,
        "routing_applied": False, "test_data_scored": False, "comparisons": comparisons,
        "file_sha256": {p.name: sha256(p) for p in (root / "scores").iterdir()},
        "embedding_manifest_sha256": {name: sha256(root / "embeddings" / name / "manifest.json")
                                     for name in ("pretrained", "joint_final", "joint_best")}}
    write_json(root / "scores/manifest.json", manifest)
    write_json(root / "verification/completion.json", {"status": "R1_R2_complete",
        "declaration_sha256": sha256(root / "protocol/declaration.json"),
        "scores_manifest_sha256": sha256(root / "scores/manifest.json")})
    monkeypatch.setattr(validation, "load_input_config", lambda path: (config, root))
    monkeypatch.setattr(validation, "_source_paths", lambda path: [])

    def figures(metrics, labels, directory, **kwargs):
        directory.mkdir(parents=True)
        (directory / "fixture_plot.txt").write_text("Figure behavior tested in reporting tests")
    monkeypatch.setattr(validation, "save_confusion_figures", figures)
    return root


def tree(root):
    return {str(path.relative_to(root)): (sha256(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_complete_eight_cell_publication_reuse_and_preservation(bundle):
    before = tree(bundle)
    result = validation.run()
    assert result["status"] == "R3_complete"
    stage = Path(result["output"])
    for name, identity in before.items():
        assert tree(bundle)[name] == identity
    assert validation.verify()["evaluations"] == 8
    after = tree(bundle)
    assert validation.run()["reused"] is True
    assert tree(bundle) == after
    summary = json.loads((stage / "summary.json").read_text())
    assert len(summary["baselines"]) == 4
    assert len(summary["evaluations"]) == 8
    for item in summary["evaluations"]:
        matched = item["geometry_encoder"].startswith("joint_")
        assert item["counts"]["actions"] == (0 if matched else 30)
        assert item["counts"]["net_rescues"] == (0 if matched else 30)
        if matched and item["rule"] == "selected":
            assert item["routing_policy"]["kind"] == "never_intervene"
        assert (stage / item["path"] / "habitat_changes.csv").exists()
        assert (stage / item["path"] / "query_results.parquet").exists()
    assert summary["test_data_scored"] is False
    assert summary["training_performed"] is False


def test_changed_r2_score_fails_before_any_output(bundle):
    path = bundle / "scores/final__pretrained.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Saved score changed"):
        validation.run()
    assert not list(bundle.glob("r3*"))


def test_changed_published_output_is_not_overwritten(bundle):
    stage = Path(validation.run()["output"])
    path = stage / "summary.json"
    path.write_text(path.read_text() + "\n")
    before = tree(bundle)
    with pytest.raises(ValueError, match="R3 artifact changed"):
        validation.run()
    assert tree(bundle) == before


def test_incomplete_existing_destination_is_not_overwritten(bundle):
    stage = bundle / "r3_validation_v1"
    stage.mkdir()
    (stage / "foreign.txt").write_text("keep")
    with pytest.raises(FileNotFoundError):
        validation.run()
    assert (stage / "foreign.txt").read_text() == "keep"


def test_failure_keeps_partial_evidence_without_complete_destination(bundle, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("fixture plot failure")
    monkeypatch.setattr(validation, "save_confusion_figures", failure)
    with pytest.raises(RuntimeError, match="fixture plot failure"):
        validation.run()
    assert not (bundle / "r3_validation_v1").exists()
    partial = list(bundle.glob("r3_validation_v1.incomplete-*"))
    assert len(partial) == 1 and (partial[0] / "protocol.json").exists()


@pytest.mark.parametrize("field,value", [("test_data_scored", True), ("query_labels_used_for_geometry", True)])
def test_rejects_wrong_r2_boundary(bundle, field, value):
    path = bundle / "scores/manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write_json(path, manifest)
    receipt = bundle / "verification/completion.json"
    completion = json.loads(receipt.read_text())
    completion["scores_manifest_sha256"] = sha256(path)
    write_json(receipt, completion)
    with pytest.raises(ValueError, match="R2 boundary differs"):
        validation.run()


@pytest.mark.parametrize("mutation", ["row_order", "candidate", "support", "eligibility", "minimum", "nonfinite"])
def test_invalid_scores_are_rejected(bundle, mutation):
    inputs = validation.load_inputs(bundle)
    scores = inputs["scores"]["final__pretrained"]
    if mutation == "row_order":
        scores["row_uid"] = scores["row_uid"][::-1]
    elif mutation == "candidate":
        scores["fusion_predictions"][0] = 17
    elif mutation == "support":
        scores["reference_plots_geo_class"][0] = 4
    elif mutation == "eligibility":
        scores["eligible"][0] = False
    elif mutation == "minimum":
        scores["A_image"][0] = 0.1
    else:
        scores["score"][0] = np.nan
    with pytest.raises(ValueError):
        validation.validate_scores(scores, inputs["rows"], inputs["experts"]["final"]["fusion"],
                                   inputs["experts"]["final"]["geo"])


@pytest.mark.parametrize("key,value", [("cleaned_test_allowed", True), ("training_allowed", True),
    ("evaluation_role", "test"), ("expected_evaluations", 4), ("geometry_arms", ["pretrained"])])
def test_stage_scope_cannot_expand(tmp_path, key, value):
    original = validation.resolve(validation.DEFAULT_CONFIG)
    config = yaml.safe_load(original.read_text())
    config[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="R3 configuration differs"):
        validation.load_stage_config(path)

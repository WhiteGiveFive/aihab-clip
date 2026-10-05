"""Exercise capacity orchestration without reading labels or running production fits."""

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_joint_tcp_capacity as study
from multimodal import geo_helpfulness_joint_tcp_capacity_fit as fitting
from multimodal import geo_helpfulness_joint_tcp_capacity_report as report
from multimodal import geo_helpfulness_joint_tcp_capacity_test as test_study


def _data():
    result = {"y_train": np.arange(4, dtype=np.int64),
              "tcp_targets": np.full((4, 2), .5, dtype=np.float64)}
    for split, count in (("train", 4), ("validation", 2)):
        result[f"{split}_row_uid"] = np.asarray([f"{split}_r{i}" for i in range(count)])
        result[f"{split}_plot_idx"] = np.asarray([f"{split}_p{i}" for i in range(count)])
        result[f"{split}_producer_fold"] = np.arange(count, dtype=np.int64) if split == "train" else np.full(count, -1, np.int64)
        for expert in ("image", "raw", "geo"):
            result[f"{split}_{expert}_logits"] = np.zeros((count, 18), np.float32)
        for name, width in (("numeric25", 25), ("no_pairs79", 79)):
            result[f"{split}_{name}"] = np.zeros((count, width), np.float32)
    return result


def _prediction(count):
    return {"state_logits": np.zeros((count, 4), np.float32),
            "tcp_logits": np.zeros((count, 2), np.float32),
            "q": np.full((count, 4), .25, np.float32),
            "tcp_hat": np.full((count, 2), .5, np.float32)}


def test_configuration_declares_128_new_fits_32_controls_and_all_contrasts():
    cfg = study.configuration()
    assert len(cfg["configs"]) == 40
    sources = Counter(spec["source"] for spec in cfg["configs"])
    assert sources == {"new_fit": 32, "reuse_parent": 8}
    assert cfg["new_fit_count"] == sources["new_fit"] * 4 == 128
    assert cfg["reused_fit_count"] == sources["reuse_parent"] * 4 == 32
    assert cfg["relaxed_policy_count"] == 40 * 9 == 360
    assert cfg["fixed_zero_policy_count"] == 40
    assert cfg["reference_policy_count"] == 1
    pairs = report._contrast_pairs(cfg)
    assert len(pairs) == 1020
    assert Counter(kind for _, _, kind in pairs) == {
        "versus_fusion": 400, "auxiliary_weight_versus_zero": 300,
        "larger_architecture_versus_h32": 320,
    }
    for spec in cfg["configs"]:
        assert len(study.parent.feature_names(spec)) == spec["input_dim"]
        assert spec["input_id"] == spec["variant"].removeprefix("ablation_")
        root, folder = study._state_folder(1, spec)
        assert root == (study.PARENT if spec["source"] == "reuse_parent" else study.OUTPUT)
        assert folder == root / f"models/seed_1/{spec['config_id']}"


@pytest.mark.parametrize("mutate", [
    lambda c: c["configs"][0].update(input_id="no_pairs79"),
    lambda c: c["configs"][0].update(config_id="wrong_parent_control"),
    lambda c: c["configs"][8].update(hidden_widths=[64]),
    lambda c: c["configs"][0].update(auxiliary_weight=False),
    lambda c: c["configs"][8].update(source="reuse_parent"),
    lambda c: c["parameters"].update(epochs=299),
    lambda c: c["parameters"].update(dropout=.2),
    lambda c: c.update(minimum_plots_grid=[10, 0]),
    lambda c: c.update(harm_numerator_grid=[5, 7]),
    lambda c: c.update(minimum_coverage_fraction=.02),
    lambda c: c.update(positive_net_required=False),
    lambda c: c.update(configuration_count=39),
    lambda c: c["bootstrap"].update(resamples=20),
    lambda c: c["bootstrap"].update(seed=123),
    lambda c: c["test_application"].update(expert_family="oracle_test_best_model"),
    lambda c: c["test_application"].update(threshold_selection=True),
    lambda c: c.update(with_embeddings=True),
    lambda c: c.update(unapproved_option=True),
])
def test_configuration_rejects_scientific_or_identity_drift(monkeypatch, mutate):
    cfg = deepcopy(study.configuration())
    mutate(cfg)
    monkeypatch.setattr(study, "_json", lambda _: cfg)
    with pytest.raises(ValueError, match="approved configuration"):
        study.configuration()


def test_all_160_predictions_are_verified_before_development_label_join(monkeypatch):
    cfg, data, verified = study.configuration(), _data(), []
    monkeypatch.setattr(study, "load_inputs", lambda _: data)
    monkeypatch.setattr(study, "_hash", lambda _: "digest")
    path = study.PROTOCOL / "protocol/development_assignments.parquet"
    monkeypatch.setattr(study, "configuration", lambda: cfg)
    monkeypatch.setattr(study, "_json", lambda _: {
        "source_file_sha256": {str(path.relative_to(study.ROOT)): "digest"}})

    def model_predictions(seed, spec, inputs, *, replay=True):
        assert replay is True
        verified.append((seed, spec["config_id"]))
        return {split: _prediction(len(data[f"{split}_row_uid"])) for split in study.SPLITS}

    def read_labels(source, columns):
        assert source == path
        assert len(verified) == 160
        assert set(verified) == {(seed, s["config_id"]) for seed in study.SEEDS for s in cfg["configs"]}
        verified.append("labels")
        return pd.concat([pd.DataFrame({
            "row_uid": data[f"{split}_row_uid"], "plot_idx": data[f"{split}_plot_idx"],
            "development_role": split, "label_id_dense": 0,
        }) for split in study.SPLITS], ignore_index=True)

    monkeypatch.setattr(study, "model_predictions", model_predictions)
    monkeypatch.setattr(study.pd, "read_parquet", read_labels)
    rows, manifests, verified_at = study.scored_rows()
    assert len(rows) == 160 * 6 and len(manifests) == 160
    assert verified[-1] == "labels" and verified_at
    assert "validation_labels" not in data


def test_failed_model_prevents_development_label_access(monkeypatch):
    def failure():
        raise ValueError("unsealed prediction")

    monkeypatch.setattr(study, "all_predictions", failure)
    monkeypatch.setattr(study.pd, "read_parquet", lambda *a, **kw: pytest.fail("labels read before sealing"))
    with pytest.raises(ValueError, match="unsealed"):
        study.scored_rows()


def test_changed_assignment_hash_prevents_label_access(monkeypatch):
    path = study.PROTOCOL / "protocol/development_assignments.parquet"
    monkeypatch.setattr(study, "all_predictions", lambda: (pd.DataFrame(), {}))
    monkeypatch.setattr(study, "_json", lambda _: {
        "source_file_sha256": {str(path.relative_to(study.ROOT)): "expected"}})
    monkeypatch.setattr(study, "_hash", lambda _: "different")
    monkeypatch.setattr(study.pd, "read_parquet", lambda *a, **kw: pytest.fail("changed labels read"))
    with pytest.raises(ValueError, match="assignments changed"):
        study.scored_rows()


def test_fit_boundary_receives_only_training_targets_and_reuses_sealed_model(tmp_path, monkeypatch):
    cfg, data = study.configuration(), _data()
    cfg["configs"] = [cfg["configs"][8]]
    root = tmp_path / "capacity"
    root.mkdir()
    (root / "declaration.json").write_text("{}")
    manifest = root / "inputs/seed_1/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    monkeypatch.setattr(study, "OUTPUT", root)
    monkeypatch.setattr(study, "validate_declaration", lambda **kw: {"config": cfg})
    monkeypatch.setattr(study, "load_inputs", lambda _: data)
    calls = []

    def fit(train_x, state_targets, tcp_targets, predict_x_by_split, **kwargs):
        calls.append(kwargs["config_id"])
        assert set(predict_x_by_split) == {"validation"}
        assert predict_x_by_split["validation"].shape == (2, 25)
        np.testing.assert_array_equal(state_targets, data["y_train"])
        np.testing.assert_array_equal(tcp_targets, data["tcp_targets"])
        assert kwargs["parameters"] == {**cfg["parameters"], "hidden_widths": [64, 32]}
        return {"state": {}, "history": [], "metadata": {}, "predictions": {
            split: _prediction(len(data[f"{split}_row_uid"])) for split in study.SPLITS}}

    def check_model(seed, spec, inputs):
        _, folder = study._state_folder(seed, spec)
        study._validate_stage(folder, root, role="joint_capacity_model")

    monkeypatch.setattr(fitting, "fit_joint", fit)
    monkeypatch.setattr(study, "predict", lambda state, x, native=True: _prediction(len(x)))
    monkeypatch.setattr(study, "model_predictions", check_model)
    study.fit_seed(1)
    first = study.snapshot(root / "models")
    study.fit_seed(1)
    assert len(calls) == 1 and first == study.snapshot(root / "models")
    _, folder = study._state_folder(1, cfg["configs"][0])
    (folder / "state.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="artifact changed"):
        study.fit_seed(1)
    assert len(calls) == 1


def test_missing_control_is_never_refitted_or_written_into_parent(tmp_path, monkeypatch):
    cfg = study.configuration()
    cfg["configs"] = cfg["configs"][:1]
    monkeypatch.setattr(study, "OUTPUT", tmp_path / "new")
    monkeypatch.setattr(study, "PARENT", tmp_path / "parent")
    monkeypatch.setattr(study, "validate_declaration", lambda **kw: {"config": cfg})
    monkeypatch.setattr(study, "load_inputs", lambda _: _data())
    monkeypatch.setattr(fitting, "fit_joint", lambda *a, **kw: pytest.fail("attempt to refit missing control"))
    with pytest.raises((ValueError, FileNotFoundError), match="[Cc]ontrol|parent"):
        study.fit_seed(1)
    assert not (tmp_path / "parent").exists()


def test_test_labels_acquired_only_after_all_four_seed_prediction_seals(tmp_path, monkeypatch):
    from multimodal import geo_helpfulness_joint_tcp_capacity_test_registry as registry
    monkeypatch.setattr(test_study, "OUTPUT", tmp_path / "test")
    monkeypatch.setattr(test_study, "validate_declaration", lambda **kw: {"registry_root": "registry"})
    events = []

    def predictions(seed, *, replay=False):
        assert replay is True
        events.append(seed)

    def make_build(*args):
        assert events == [1, 2, 3, 4]
        events.append("build")
        return {"build": "identity"}

    def reserve(*args):
        assert events[-1] == "build"
        events.append("reserve")
        return {"reservation": "id"}

    class LabelsReached(Exception):
        pass

    def acquire(*args):
        assert events == [1, 2, 3, 4, "build", "reserve"]
        raise LabelsReached

    monkeypatch.setattr(test_study, "predictions", predictions)
    monkeypatch.setattr(registry, "make_build_record", make_build)
    monkeypatch.setattr(registry, "reserve_event", reserve)
    monkeypatch.setattr(registry, "acquire_labels", acquire)
    with pytest.raises(LabelsReached):
        test_study.score()


def test_any_failed_test_prediction_prevents_registry_or_label_access(tmp_path, monkeypatch):
    from multimodal import geo_helpfulness_joint_tcp_capacity_test_registry as registry
    monkeypatch.setattr(test_study, "OUTPUT", tmp_path / "test")
    monkeypatch.setattr(test_study, "validate_declaration", lambda **kw: {})

    def fail(*args, **kwargs):
        raise ValueError("unsealed test predictions")

    monkeypatch.setattr(test_study, "predictions", fail)
    monkeypatch.setattr(registry, "make_build_record", lambda *a, **kw: pytest.fail("registered incomplete predictions"))
    monkeypatch.setattr(registry, "acquire_labels", lambda *a, **kw: pytest.fail("labels accessed prematurely"))
    with pytest.raises(ValueError, match="unsealed"):
        test_study.score()


def test_build_runner_orders_development_before_test_and_stops_on_failure(monkeypatch):
    path = study.ROOT / "tools/run_multimodal_geo_helpfulness_joint_tcp_capacity.py"
    spec = importlib.util.spec_from_file_location("capacity_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        calls.append(command[2:])

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [str(path), "build"])
    runner.main()
    assert calls == ([['declare']] + [['prepare-seed', '--seed', str(s)] for s in study.SEEDS] +
                     [['fit-seed', '--seed', str(s)] for s in study.SEEDS] + [['score'], ['validate'], ['test-declare']] +
                     [['test-predict', '--seed', str(s)] for s in study.SEEDS] + [['test-score'], ['test-validate']])

    def fail_first(command, **kwargs):
        raise RuntimeError("subprocess failed")

    monkeypatch.setattr(runner.subprocess, "run", fail_first)
    with pytest.raises(RuntimeError, match="subprocess failed"):
        runner.main()

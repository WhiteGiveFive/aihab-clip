from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_confidnet_fold4 as study
from multimodal import geo_helpfulness_confidnet_oof_fit_v2 as fitting


@pytest.mark.parametrize("field,value", [
    ("train_rows", 3378), ("validation_expert", "full_development_train"),
    ("threshold_selection", False), ("locked_test_access", True),
    ("confidence_input_dim", 1154), ("target", "maximum_class_probability"),
    ("router_augmented_features", ["true_tcp_geo"]),
    ("guardrails", {"minimum_plots": [0]}),
    ("bootstrap", {"resamples": 100, "seed": 1}),
    ("verification", {"native_float32": "approximate"}),
])
def test_config_rejects_changed_approved_scientific_settings(monkeypatch, field, value):
    cfg = deepcopy(study._json(study.CONFIG))
    cfg[field] = value
    monkeypatch.setattr(study, "_json", lambda path: cfg)
    with pytest.raises(ValueError, match="setting|configuration|settings|contract"):
        study._config()


def test_config_matches_approved_fold4_recipe():
    cfg = study._config()
    assert cfg["confidence_parameters"] == fitting.CONFIDENCE_DEFAULTS
    assert cfg["router_parameters"] == fitting.RF_DEFAULTS
    assert cfg["fold_ids"] == [3]
    assert cfg["train_rows"] == 852


@pytest.fixture
def output_roots(tmp_path, monkeypatch):
    base = tmp_path / "studies"
    base.mkdir()
    old = base / "older_study"
    old.mkdir()
    (old / "immutable.txt").write_text("untouched")
    output = base / "confidnet_fold4_v1"
    monkeypatch.setattr(study, "BASE", base)
    monkeypatch.setattr(study, "OUTPUT", output)
    return base, old, output


def test_outputs_cannot_enter_or_contain_prior_studies(output_roots):
    base, old, output = output_roots
    for path in (base, old, old / "child", base.parent):
        with pytest.raises(ValueError, match="disjoint|cannot contain"):
            study._output(path)
    assert study._output(output) == output
    output.mkdir()
    assert study._output(output) == output
    assert (old / "immutable.txt").read_text() == "untouched"


@pytest.fixture
def declaration(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    base = root / "studies"
    base.mkdir()
    output = base / "confidnet_fold4_v1"
    output.mkdir()
    old = base / "older"
    old.mkdir()
    (old / "artifact.txt").write_text("immutable")
    code = root / "implementation.py"
    code.write_text("saved implementation")
    config = root / "config.json"
    config.write_text("{}")
    parent = root / "parent.bin"
    parent.write_bytes(b"frozen expert")
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "BASE", base)
    monkeypatch.setattr(study, "OUTPUT", output)
    monkeypatch.setattr(study, "CONFIG", config)
    monkeypatch.setattr(study, "CODE_PATHS", ("implementation.py",))
    monkeypatch.setattr(study, "_config", lambda: {"study_id": "confidnet_fold4_v1"})
    saved = {
        "schema_version": "geo_helpfulness.confidnet_fold4_declaration.v1",
        "config": study._config(),
        "file_sha256": {p.name: study._hash(p) for p in (code, config, parent)},
        "prior_study_snapshots": {"older": study._snapshot(old)},
        "locked_test_access": False,
        "validation_label_use": "scoring_and_declared_threshold_selection_after_prediction_verification",
    }
    (output / "declaration.json").write_text(json.dumps(saved))
    return output, root, old, saved


def test_declaration_detects_changed_source_and_parent_bytes(declaration):
    output, root, _, saved = declaration
    assert study.validate_declaration(output, check_parents=True) == saved
    (root / "implementation.py").write_text("changed implementation")
    with pytest.raises(ValueError, match="source changed"):
        study.validate_declaration(output)
    (root / "implementation.py").write_text("saved implementation")
    (root / "parent.bin").write_bytes(b"changed expert")
    with pytest.raises(ValueError, match="source changed"):
        study.validate_declaration(output, check_parents=True)


def test_declaration_requires_implementation_hashes(declaration):
    output, _, _, saved = declaration
    del saved["file_sha256"]["implementation.py"]
    (output / "declaration.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="omits implementation"):
        study.validate_declaration(output)


def test_declaration_rejects_prior_bundle_mutation(declaration):
    output, _, old, _ = declaration
    (old / "artifact.txt").write_text("changed")
    with pytest.raises(ValueError, match="Prior experiment changed"):
        study.validate_declaration(output, check_parents=True)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "other_study.v1"), ("locked_test_access", True),
    ("validation_label_use", "fitting"),
])
def test_declaration_rejects_changed_identity_and_label_boundary(declaration, field, value):
    output, _, _, saved = declaration
    saved[field] = value
    (output / "declaration.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        study.validate_declaration(output)


def _small_payloads():
    result = {}
    for split in ("train", "validation"):
        data = {
            "row_uid": np.asarray([f"{split}0", f"{split}1"]),
            "plot_idx": np.asarray([f"{split}p0", f"{split}p1"]),
            "producer_fold": np.asarray([3, 3], dtype=np.int64),
            **{f"{mode}_logits": np.zeros((2, 18)) for mode in study.MODES},
        }
        if split == "train":
            data["labels"] = np.asarray([1, 0])
        result[split] = data
    return result


def test_all_twelve_heads_and_eight_routers_verified_before_validation_label_join(monkeypatch, tmp_path):
    events = []
    payloads = _small_payloads()
    monkeypatch.setattr(study, "_check_paused", lambda root: None)
    monkeypatch.setattr(study, "_features", lambda root, seed: payloads)
    monkeypatch.setattr(study, "_inputs", lambda *args: {})
    monkeypatch.setattr(study, "_matrices", lambda *args: {})

    def head(root, seed, mode, arrays):
        events.append(("head", seed, mode))
        return {split: np.full(2, .5, dtype=np.float32) for split in payloads}

    def router(root, seed, arm, matrices):
        events.append(("router", seed, arm))
        return {split: np.full((2, 4), .25) for split in payloads}

    def read_labels(path, columns):
        assert len(events) == 20
        assert set(events) == {
            *(('head', s, m) for s in study.SEEDS for m in study.MODES),
            *(('router', s, a) for s in study.SEEDS for a in study.ARMS),
        }
        events.append(("labels",))
        assert "label_id_dense" in columns
        return pd.DataFrame({
            "row_uid": ["validation1", "validation0"],
            "plot_idx": ["validationp1", "validationp0"],
            "development_role": ["validation"] * 2,
            "label_id_dense": [7, 4],
        })

    monkeypatch.setattr(study, "_head", head)
    monkeypatch.setattr(study, "_router", router)
    monkeypatch.setattr(study.pd, "read_parquet", read_labels)
    scored, rows, verified = study._payloads(tmp_path)
    assert events[-1] == ("labels",)
    assert verified
    assert len(rows) == 16
    for seed in study.SEEDS:
        np.testing.assert_array_equal(scored["validation"][seed]["labels"], [4, 7])
        np.testing.assert_array_equal(scored["train"][seed]["labels"], [1, 0])
        assert "labels" not in payloads["validation"]


def test_prediction_failure_prevents_validation_label_join(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_check_paused", lambda root: None)
    monkeypatch.setattr(study, "_features", lambda root, seed: _small_payloads())

    def invalid(*args):
        raise ValueError("failed confidence replay")

    def forbidden(*args, **kwargs):
        pytest.fail("validation labels read after failed prediction verification")

    monkeypatch.setattr(study, "_head", invalid)
    monkeypatch.setattr(study.pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="confidence replay"):
        study._payloads(tmp_path)


def test_inputs_replay_cannot_trigger_a_fit(monkeypatch, tmp_path):
    from multimodal import geo_helpfulness_confidnet_fold4_inputs as numerical
    final = tmp_path / "router_inputs/seed_2"
    final.mkdir(parents=True)
    saved = {"X_train": np.zeros((852, 25)), "X_validation": np.zeros((822, 25))}
    states, metadata = {"saved_state": "sentinel"}, {"metadata": "sentinel"}
    events = []

    def replay(train, validation, *, seed, states=None):
        assert seed == 2
        assert states == {"saved_state": "sentinel"}
        events.append("replay")
        return {"arrays": saved, "states": states, "metadata": metadata}

    monkeypatch.setattr(numerical, "build_inputs", replay)
    monkeypatch.setattr(study, "_hash", lambda path: "features-digest")
    monkeypatch.setattr(study, "_validate_stage", lambda *args, **kwargs: {
        "metadata": {"seed": 2, "features_sha256": "features-digest"}
    })
    monkeypatch.setattr(study, "_load_arrays", lambda path: saved)
    monkeypatch.setattr(study, "_json", lambda path: states if path.name == "states.json" else metadata)
    assert study._inputs(tmp_path, 2, {"train": {}, "validation": {}}) == saved
    assert events == ["replay"]


def test_missing_inputs_do_not_fit_when_build_false(tmp_path):
    with pytest.raises(ValueError, match="not yet fitted"):
        study._inputs(tmp_path, 1, {}, build=False)


@pytest.mark.parametrize("field,value", [("training_seed", 2), ("fit_row_count", 3378)])
def test_confidence_state_rejects_wrong_training_scope(monkeypatch, tmp_path, field, value):
    state = {
        "training_seed": 1, "fit_row_count": 852, "input_dim": 256,
        "status": "complete", "completed_epochs": 500,
        "parameters": fitting.CONFIDENCE_DEFAULTS,
    }
    state[field] = value
    monkeypatch.setattr(study, "_hash", lambda path: "feature-digest")
    monkeypatch.setattr(study, "_validate_stage", lambda *args, **kwargs: {
        "metadata": {"seed": 1, "mode": "image", "features_sha256": "feature-digest"}
    })
    monkeypatch.setattr(study, "_json", lambda path: state if path.name == "state.json" else [])
    with pytest.raises(ValueError, match="approved fit"):
        study._head(tmp_path, 1, "image", {})


@pytest.mark.parametrize("seed", [0, 5, True, 1.0])
def test_seed_guard_rejects_nonapproved_seed(seed):
    with pytest.raises(ValueError, match="Expected seed"):
        study._seed(seed)


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "tools/run_multimodal_geo_helpfulness_confidnet_fold4.py"
    spec = importlib.util.spec_from_file_location("fold4_test_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_build_orders_all_fits_before_scoring(monkeypatch, tmp_path):
    cli = _load_cli()
    calls = []
    monkeypatch.setattr(cli.sys, "argv", ["study", "build", "--device", "cpu", "--output-dir", str(tmp_path)])

    def run(command, **kwargs):
        calls.append(command[2:])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.main() == 0
    assert [command[0] for command in calls] == [
        "declare", *[phase for _ in range(4) for phase in ("extract-seed", "fit-seed")],
        "fit-routers", "score", "validate",
    ]
    assert all(command[-2:] == ["--output-dir", str(tmp_path)] for command in calls)


def test_cli_build_stops_at_failed_stage(monkeypatch):
    cli = _load_cli()
    calls = []
    monkeypatch.setattr(cli.sys, "argv", ["study", "build"])

    def run(command, **kwargs):
        calls.append(command[2])
        return SimpleNamespace(returncode=3 if command[2] == "fit-seed" else 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.main() == 3
    assert calls == ["declare", "extract-seed", "fit-seed"]

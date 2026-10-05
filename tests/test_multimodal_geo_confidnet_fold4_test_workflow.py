from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_confidnet_fold4_test as study
from multimodal import geo_helpfulness_confidnet_fold4_test_registry as registry
from multimodal import geo_helpfulness_confidnet_fold4_test_report as reporting


@pytest.mark.parametrize("field,value", [
    ("fitting", True), ("recalibration", True), ("threshold_selection", True),
    ("expert_producer_fold", None), ("test_rows", 3378),
    ("tie_priority", ["image", "geo", "raw"]),
    ("direct_tcp_policy", "compare_true_tcp"),
    ("confidence_execution", "float64"),
    ("test_label_access", "before_prediction"),
])
def test_config_rejects_changes_to_frozen_application(monkeypatch, field, value):
    config = deepcopy(study._json(study.CONFIG))
    config[field] = value
    monkeypatch.setattr(study, "_json", lambda path: config)
    with pytest.raises(ValueError, match="configuration changed"):
        study._config()


@pytest.mark.parametrize("seed", [True, False, 0, 5, 1.0, "1", None])
def test_seed_validation_rejects_noncanonical_inputs(seed):
    with pytest.raises(ValueError, match="seed"):
        study._seed(seed)


@pytest.mark.parametrize("seed", study.SEEDS)
def test_frozen_apply_reproduces_parent_validation_inputs_confidence_and_routers(seed, monkeypatch):
    """A real saved-state replay; accesses development-validation inputs only."""
    from multimodal import geo_helpfulness_router_numeric as numeric

    def forbidden(*args, **kwargs):
        pytest.fail("Test inference must never refit preprocessing")

    monkeypatch.setattr(numeric, "fit_expert_temperature", forbidden)
    monkeypatch.setattr(numeric, "fit_router_feature_transform", forbidden)
    features = study._load_arrays(study.PARENT / f"features/seed_{seed}/validation.npz")
    states = study._json(study.PARENT / f"router_inputs/seed_{seed}/states.json")
    untouched = deepcopy(states)
    observed = study.apply_numeric_inputs(features, states)
    with np.load(study.PARENT / f"router_inputs/seed_{seed}/arrays.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(observed, saved["X_validation"])
    assert observed.shape == (822, 25)
    assert states == untouched
    selections = study._json(study.PARENT / "scores/report.json")["relaxed"]["selections"]
    arrays, diagnostics = study._infer(seed, features, selections)
    parent_q = {}
    for mode in study.MODES:
        with np.load(study.PARENT / f"confidence/seed_{seed}/{mode}/predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(arrays[f"{mode}_tcp_hat"], saved["validation"])
        assert diagnostics[mode]["native_json_replay_exact"]
    for arm in study.ARMS:
        with np.load(study.PARENT / f"routers/seed_{seed}/{arm}/predictions.npz", allow_pickle=False) as saved:
            parent_q[arm] = saved["validation"]
            np.testing.assert_allclose(arrays[f"router_{arm}_q"], parent_q[arm], atol=2e-16, rtol=0)
            np.testing.assert_array_equal(arrays[f"router_{arm}_q"].argmax(axis=1), parent_q[arm].argmax(axis=1))
    parent_payload = study._payload(arrays)
    for arm, values in parent_q.items():
        parent_payload[f"router_{arm}_q"] = values
    for name, expected in reporting.policy_choices(parent_payload, selections).items():
        np.testing.assert_array_equal(arrays[f"choice__{name}"], expected)
    assert not any("label" in key for key in arrays)


@pytest.fixture
def declaration(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    output = root / "output"
    output.mkdir()
    parent = root / "parent"
    (parent / "scores").mkdir(parents=True)
    (parent / "state.json").write_text('{"frozen":true}')
    (parent / "scores/report.json").write_text('{"relaxed":{"selections":{"saved":true}}}')
    events = root / "events"
    events.mkdir()
    (events / "old.json").write_text('{"old":true}')
    for name, value in (("code.py", "frozen code"), ("config.json", "{}"),
                        ("source.parquet", "frozen source"), ("image.jpg", "frozen image")):
        (root / name).write_text(value)
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "OUTPUT", output)
    monkeypatch.setattr(study, "PARENT", parent)
    monkeypatch.setattr(study, "CONFIG", root / "config.json")
    monkeypatch.setattr(study, "CODE_PATHS", ("code.py",))
    monkeypatch.setattr(study, "_config", lambda: {"study_id": "confidnet_fold4_test_v1"})
    saved = {
        "study_id": "confidnet_fold4_test_v1", "config": study._config(),
        "adaptive_reuse": True, "no_go_preserved": True,
        "source_file_sha256": {name: study._hash(root / name) for name in ("code.py", "config.json", "source.parquet")},
        "image_file_sha256": {"image.jpg": study._hash(root / "image.jpg")},
        "parent_snapshot": study._snapshot(parent),
        "registry_root": str(events), "prior_registry_snapshot": study._snapshot(events),
        "validation_selections": {"saved": True},
    }
    (output / "declaration.json").write_text(json.dumps(saved))
    return root, output, parent, events, saved


@pytest.mark.parametrize("path", ["code.py", "config.json", "source.parquet", "image.jpg", "parent/state.json", "events/old.json"])
def test_full_declaration_check_detects_parent_input_and_registry_changes(declaration, path):
    root, _, _, _, saved = declaration
    assert study.validate_declaration(check_parents=True) == saved
    (root / path).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        study.validate_declaration(check_parents=True)


def test_registry_may_append_new_events_without_mutating_prior_events(declaration):
    _, _, _, events, saved = declaration
    (events / "new.json").write_text('{"new":true}')
    assert study.validate_declaration(check_parents=True) == saved


def test_parent_bundle_rejects_untracked_additions(declaration):
    _, _, parent, _, _ = declaration
    (parent / "new-state.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed files changed"):
        study.validate_declaration()


def test_declaration_requires_all_implementation_hashes(declaration):
    _, output, _, _, saved = declaration
    del saved["source_file_sha256"]["code.py"]
    (output / "declaration.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="incomplete"):
        study.validate_declaration()


class AcquisitionReached(Exception):
    pass


@pytest.fixture
def scoring_boundary(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(study, "OUTPUT", output)
    events = []
    declaration = {"registry_root": str(tmp_path / "registry")}

    def validate(*, check_parents=False):
        events.append(("validate", check_parents))
        return declaration

    def predictions(seed, **kwargs):
        events.append(("prediction", seed))

    def build(root, declaration, manifests):
        events.append(("build", tuple(manifests)))
        return {"bundle": "synthetic"}

    def reserve(root, build):
        events.append(("reserve",))
        return {"owner": "synthetic"}

    def acquire(*args):
        events.append(("acquire",))
        raise AcquisitionReached

    monkeypatch.setattr(study, "validate_declaration", validate)
    monkeypatch.setattr(study, "_predictions", predictions)
    monkeypatch.setattr(registry, "make_build_record", build)
    monkeypatch.setattr(registry, "reserve_event", reserve)
    monkeypatch.setattr(registry, "acquire_labels", acquire)
    return events


def test_all_four_predictions_and_registry_reservation_precede_label_acquisition(scoring_boundary):
    with pytest.raises(AcquisitionReached):
        study.score()
    assert scoring_boundary == [
        ("validate", True), *(('prediction', seed) for seed in study.SEEDS),
        ("build", study.SEEDS), ("reserve",), ("acquire",),
    ]


def test_missing_last_seed_prevents_registry_reservation_and_label_access(scoring_boundary, monkeypatch):
    def predictions(seed, **kwargs):
        scoring_boundary.append(("prediction", seed))
        if seed == 4:
            raise ValueError("Missing fourth prediction bundle")

    monkeypatch.setattr(study, "_predictions", predictions)
    with pytest.raises(ValueError, match="fourth"):
        study.score()
    assert not any(event[0] in {"build", "reserve", "acquire"} for event in scoring_boundary)


def _synthetic_predictions():
    # Use the canonical threshold serialization from the completed parent.
    parent = study._json(study.PARENT / "scores/report.json")["relaxed"]["selections"]
    selections = deepcopy(parent)
    result = {
        "row_uid": np.asarray(["row_a", "row_b"]),
        "plot_idx": np.asarray(["plot_a", "plot_b"]),
        "producer_fold": np.asarray([3, 3], dtype=np.int64),
        "X_numeric25": np.zeros((2, 25)),
    }
    for i, mode in enumerate(study.MODES):
        logits = np.zeros((2, 18))
        logits[:, i] = 1.0
        result[f"{mode}_logits"] = logits
        result[f"{mode}_tcp_hat"] = np.asarray([0.2 + 0.2 * i, 0.8 - 0.2 * i], dtype=np.float32)
    for arm in study.ARMS:
        result[f"router_{arm}_q"] = np.asarray([[0.6, 0.1, 0.2, 0.1], [0.1, 0.7, 0.1, 0.1]])
    for name, choices in reporting.policy_choices(study._payload(result), selections).items():
        result[f"choice__{name}"] = choices
    return result, selections


@pytest.mark.parametrize("corruption", [None, "plot", "choice", "missing_policy", "extra_policy"])
def test_scoring_join_respects_identities_and_presealed_decisions(monkeypatch, corruption):
    arrays, selections = _synthetic_predictions()
    labels = pd.DataFrame({"row_uid": ["row_b", "row_a"], "plot_idx": ["plot_b", "plot_a"], "label_id_dense": [1, 0]})
    if corruption == "plot":
        labels.loc[0, "plot_idx"] = "other"
    elif corruption == "choice":
        arrays["choice__tcp_max_three"][0] = 1
    elif corruption == "missing_policy":
        del arrays["choice__tcp_max_three"]
    elif corruption == "extra_policy":
        arrays["choice__test_tuned"] = np.zeros(2, dtype=np.int8)
    monkeypatch.setattr(study, "_predictions", lambda seed: arrays)
    if corruption is None:
        payloads = study._scoring_inputs(labels, {"validation_selections": selections})
        for payload in payloads.values():
            np.testing.assert_array_equal(payload["labels"], [0, 1])
        assert "labels" not in arrays
    else:
        with pytest.raises(ValueError, match="identity|policy|seal"):
            study._scoring_inputs(labels, {"validation_selections": selections})


@pytest.mark.parametrize("corruption", [None, "labels", "dtype", "confidence", "choice", "array_name", "diagnostics"])
def test_saved_prediction_replay_checks_arrays_choices_and_diagnostics(tmp_path, monkeypatch, corruption):
    expected, selections = _synthetic_predictions()
    output = tmp_path / "output"
    output.mkdir()
    (output / "declaration.json").write_text(json.dumps({"validation_selections": selections}))
    feature_dir = output / "features/seed_1"
    feature_dir.mkdir(parents=True)
    (feature_dir / "manifest.json").write_text("{}")
    monkeypatch.setattr(study, "OUTPUT", output)
    monkeypatch.setattr(study, "_features", lambda seed, deep: {"synthetic": True})
    diagnostics = {"native_replay_exact": True, "float64_difference": 1e-8}
    monkeypatch.setattr(study, "_infer", lambda *args: (expected, diagnostics))
    saved = {key: value.copy() for key, value in expected.items()}
    saved_diagnostics = deepcopy(diagnostics)
    if corruption == "labels":
        saved["labels"] = np.asarray([0, 1])
    elif corruption == "dtype":
        saved["raw_tcp_hat"] = saved["raw_tcp_hat"].astype(np.float64)
    elif corruption == "confidence":
        saved["raw_tcp_hat"][0] += 0.01
    elif corruption == "choice":
        saved["choice__tcp_max_three"][0] = 1
    elif corruption == "array_name":
        saved["unapproved"] = np.zeros(2)
    elif corruption == "diagnostics":
        saved_diagnostics["float64_difference"] = 2e-8
    staged = study._staging(output, "synthetic_predictions")
    study._save_arrays(staged / "arrays.npz", saved)
    study._write_json(staged / "replay.json", saved_diagnostics)
    study._seal_stage(staged, output / "predictions/seed_1", output,
                      role="fold4_cleaned_test_predictions", metadata={
                          "seed": 1, "features_sha256": study._hash(feature_dir / "manifest.json")})
    if corruption is None:
        observed = study._predictions(1, replay=True)
        for key, value in expected.items():
            np.testing.assert_array_equal(observed[key], value)
    else:
        with pytest.raises(ValueError, match="labels|replay|names changed|diagnostics"):
            study._predictions(1, replay=True)


def _runner():
    path = Path(__file__).resolve().parents[1] / "tools/run_multimodal_geo_helpfulness_confidnet_fold4_test.py"
    spec = importlib.util.spec_from_file_location("fold4_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_build_orders_all_seed_predictions_before_scoring(monkeypatch):
    runner = _runner()
    calls = []
    monkeypatch.setattr(runner.sys, "argv", ["runner", "build", "--device", "cpu"])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command[2:]) or SimpleNamespace(returncode=0))
    assert runner.main() == 0
    expected = [["declare"]]
    for seed in study.SEEDS:
        expected += [["extract-seed", "--seed", str(seed), "--device", "cpu"], ["predict-seed", "--seed", str(seed)]]
    assert calls == [*expected, ["score"], ["validate"]]


def test_cli_stops_before_test_scoring_when_inference_fails(monkeypatch):
    runner = _runner()
    calls = []
    monkeypatch.setattr(runner.sys, "argv", ["runner", "build"])

    def run(command, **kwargs):
        calls.append(command[2:])
        return SimpleNamespace(returncode=17 if command[2:4] == ["extract-seed", "--seed"] else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main() == 17
    assert len(calls) == 2
    assert all("score" not in command for command in calls)

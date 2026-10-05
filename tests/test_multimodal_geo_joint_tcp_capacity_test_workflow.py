"""Capacity legacy-test score ownership and label-free application boundaries."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_joint_tcp_capacity_test as workflow
from multimodal import geo_helpfulness_joint_tcp_capacity_test_registry as registry


# Run the established complete registry contract against the new, separate owner.
# Loading into its own namespace keeps the old test module and registry unchanged.
_spec = importlib.util.spec_from_file_location(
    "capacity_legacy_registry_contract",
    Path(__file__).with_name("test_multimodal_geo_joint_tcp_test_registry.py"),
)
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)
_contract.api = registry
for _name, _value in vars(_contract).items():
    if _name.startswith("test_"):
        globals()["test_capacity_registry_" + _name[5:]] = _value


class StopAtLabels(Exception):
    pass


@pytest.fixture
def scoring_boundary(tmp_path, monkeypatch):
    events = []
    declaration = {"registry_root": str(tmp_path / "registry")}
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path / "output")
    workflow.OUTPUT.mkdir()
    monkeypatch.setattr(workflow, "validate_declaration", lambda **kwargs: declaration)
    monkeypatch.setattr(workflow, "_lock", lambda *args: nullcontext())
    monkeypatch.setitem(sys.modules, "multimodal.geo_helpfulness_joint_tcp_capacity_report",
                        SimpleNamespace(make_test_report=lambda *args, **kwargs: None))

    def predictions(seed, *, replay):
        assert replay is True
        events.append(("prediction", seed))

    def build(output, value, manifests):
        assert events == [("prediction", s) for s in range(1, 5)]
        assert set(manifests) == {1, 2, 3, 4}
        events.append("build")
        return {"bound": True}

    def reserve(root, value):
        assert events[-1] == "build" and value == {"bound": True}
        events.append("reserve")
        return {"reserved": True}

    def acquire(root, value, reservation, saved, folder):
        assert events[-1] == "reserve" and reservation == {"reserved": True}
        events.append("labels")
        raise StopAtLabels()

    monkeypatch.setattr(workflow, "predictions", predictions)
    monkeypatch.setattr(registry, "make_build_record", build)
    monkeypatch.setattr(registry, "reserve_event", reserve)
    monkeypatch.setattr(registry, "acquire_labels", acquire)
    return events


def test_all_four_prediction_replays_and_reservation_precede_labels(scoring_boundary):
    with pytest.raises(StopAtLabels):
        workflow.score()
    assert scoring_boundary == [("prediction", s) for s in range(1, 5)] + ["build", "reserve", "labels"]


@pytest.mark.parametrize("failed_seed", [1, 2, 3, 4])
def test_failed_prediction_replay_prevents_reservation_and_labels(scoring_boundary, monkeypatch, failed_seed):
    original = workflow.predictions

    def reject(seed, *, replay):
        if seed == failed_seed:
            raise ValueError("source replay failed")
        original(seed, replay=replay)

    monkeypatch.setattr(workflow, "predictions", reject)
    with pytest.raises(ValueError, match="source replay failed"):
        workflow.score()
    assert not any(name in scoring_boundary for name in ("build", "reserve", "labels"))


@pytest.fixture
def application(monkeypatch):
    specs = [{"config_id": "example_" + short, "input_id": short,
              "variant": variant, "with_embeddings": False, "auxiliary_weight": .1,
              "architecture_id": "h64_32"}
             for short, variant in (("numeric25", "ablation_numeric25"),
                                    ("no_pairs79", "ablation_no_pairs79"))]
    features = {"row_uid": np.array(["a", "b"]), "plot_idx": np.array(["p1", "p2"]),
                "producer_fold": np.full(2, -1, dtype=np.int64),
                "X_numeric25": np.zeros((2, 25), np.float32),
                "X_no_pairs79": np.ones((2, 79), np.float32),
                **{f"{prefix}_logits": np.zeros((2, 18), np.float64) for prefix in ("image", "geo", "raw")}}
    native = {"q": np.tile(np.asarray([.6, .1, .2, .1], np.float32), (2, 1)),
              "tcp_hat": np.tile(np.asarray([.7, .3], np.float32), (2, 1)),
              "state_logits": np.zeros((2, 4), np.float32), "tcp_logits": np.zeros((2, 2), np.float32)}
    calls = []
    monkeypatch.setattr(workflow.study, "model_state", lambda seed, spec: (seed, spec["input_id"]))

    def predict(state, matrix, *, native=True):
        assert state[0] == 1
        assert matrix is features["X_" + state[1]]
        calls.append((state, native))
        return {key: value.copy() if native else value.astype(np.float64) + 1e-9
                for key, value in output.items()}

    output = native
    monkeypatch.setattr(workflow.study, "predict", predict)
    return {"configs": specs}, features, calls


def test_native_predictions_are_authoritative_and_float64_is_diagnostic(application):
    config, features, calls = application
    rows, diagnostics = workflow._infer(1, features, config)
    assert len(rows) == 4 and rows.q_rescue.dtype == np.float32
    assert rows.q_rescue.tolist() == [float(np.float32(.6))] * 4
    assert calls == [((1, short), native) for short in ("numeric25", "no_pairs79") for native in (True, True, False)]
    assert all(value["native_float32_exact"] for value in diagnostics.values())
    assert all(value["q"]["max_abs_error"] > 0 for value in diagnostics.values())
    assert not any("label" in name or "target" in name for name in rows)


def test_nonreproducible_native_model_is_rejected(application, monkeypatch):
    config, features, calls = application
    original = workflow.study.predict

    def predict(state, matrix, **kwargs):
        value = original(state, matrix, **kwargs)
        if len(calls) == 2:
            value["q"][0, 0] += .1
        return value

    monkeypatch.setattr(workflow.study, "predict", predict)
    with pytest.raises(ValueError, match="does not reproduce exactly"):
        workflow._infer(1, features, config)


@pytest.mark.parametrize("mutation", [None, "array", "dtype", "keys", "metadata"])
def test_prediction_replay_rechecks_authoritative_legacy_sources(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    values = {"row_uid": np.array(["a", "b"]), "X_numeric25": np.zeros((2, 25), np.float32)}
    cached = {name: value.copy() for name, value in values.items()}
    if mutation == "array":
        cached["X_numeric25"][0, 0] = .01
    elif mutation == "dtype":
        cached["X_numeric25"] = cached["X_numeric25"].astype(np.float64)
    elif mutation == "keys":
        cached["extra"] = np.zeros(2)
    metadata = {"native_replay": "verified"}
    saved_metadata = metadata if mutation != "metadata" else {"native_replay": "wrong"}
    rows = pd.DataFrame({"row_uid": ["a", "b"]})
    actions = pd.DataFrame({"row_uid": ["a", "b"], "policy": [True, False]})
    calls = []
    monkeypatch.setattr(workflow, "_validate_stage", lambda *args, **kwargs: {"metadata": {"seed": 1}})
    monkeypatch.setattr(workflow, "_load_arrays", lambda path: cached)
    monkeypatch.setattr(workflow, "_json", lambda path: {
        "declaration.json": {"legacy_context": {"declared": True}, "config": {}, "validation_selections": {}},
        "source_replay.json": saved_metadata, "float64_replay.json": {},
    }[path.name])
    monkeypatch.setattr(pd, "read_parquet", lambda path: rows.copy() if path.name == "rows.parquet" else actions.copy())

    def prepare(seed, *, context):
        assert seed == 1 and context == {"declared": True}
        calls.append("source_replay")
        return {"test": values, "metadata": metadata}

    monkeypatch.setattr(workflow, "prepare_seed", prepare)
    monkeypatch.setattr(workflow, "_infer", lambda *args: (rows.copy(), {}))
    monkeypatch.setitem(sys.modules, "multimodal.geo_helpfulness_joint_tcp_capacity_report",
                        SimpleNamespace(policy_actions=lambda *args: actions.copy()))
    if mutation is not None:
        with pytest.raises(ValueError, match="(feature|metadata)"):
            workflow.predictions(1, replay=True)
    else:
        actual_rows, actual_actions = workflow.predictions(1, replay=True)
        pd.testing.assert_frame_equal(actual_rows, rows)
        pd.testing.assert_frame_equal(actual_actions, actions)
    assert calls == ["source_replay"]


def test_accuracy_and_weighted_f1_reference_are_computed_independently():
    observed = workflow._classification_reference(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 2, 1]))
    # Class F1s are 2/3, 4/6 and 0, weighted by supports 2, 3 and 0.
    assert observed["accuracy"] == .6
    assert np.isclose(observed["weighted_f1"], 2 / 3, atol=1e-15, rtol=0)


@pytest.mark.parametrize("mutation", [None, "accuracy", "f1", "rows", "seed"])
def test_fusion_reference_requires_september15_baseline_and_identity(monkeypatch, mutation):
    labels = np.zeros(1347, dtype=np.int64)
    logits = np.zeros((1347, 18), dtype=np.float64)
    logits[:, 0] = 1
    rows = pd.concat([pd.DataFrame({"row_uid": [f"u{i}" for i in range(1347)],
                                    "plot_idx": [f"p{i % 531}" for i in range(1347)],
                                    "training_seed": seed, "config_id": "example",
                                    "label_id_dense": labels, "raw_logits": list(logits)})
                      for seed in range(1, 5)], ignore_index=True)
    expected = {"accuracy": 1., "weighted_f1": 1.}
    if mutation == "accuracy":
        expected["accuracy"] = workflow.FUSION_REFERENCE["accuracy"]
    elif mutation == "f1":
        expected["weighted_f1"] = workflow.FUSION_REFERENCE["weighted_f1"]
    elif mutation == "rows":
        rows = rows.iloc[1:]
    elif mutation == "seed":
        rows = rows.loc[rows.training_seed != 4]
    monkeypatch.setattr(workflow, "FUSION_REFERENCE", expected)
    if mutation is not None:
        with pytest.raises(ValueError, match="(reference|baseline)"):
            workflow._assert_fusion_reference(rows)
    else:
        result = workflow._assert_fusion_reference(rows)
        assert result["mean"] == expected and set(result["per_seed"]) == {"1", "2", "3", "4"}


def test_completed_score_reuse_runs_full_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    (tmp_path / "scores").mkdir()
    calls = []
    monkeypatch.setattr(workflow, "validate_declaration", lambda **kwargs: {"registry_root": str(tmp_path / "registry")})
    monkeypatch.setattr(workflow, "_lock", lambda *args: nullcontext())
    monkeypatch.setattr(workflow, "_validate_stage", lambda *args, **kwargs: calls.append("seal"))
    monkeypatch.setattr(workflow, "_json", lambda path: {})
    monkeypatch.setattr(registry, "finish_event", lambda *args: calls.append("finish"))
    monkeypatch.setattr(workflow, "validate", lambda: calls.append("validate"))
    monkeypatch.setitem(sys.modules, "multimodal.geo_helpfulness_joint_tcp_capacity_report",
                        SimpleNamespace(make_test_report=lambda *args, **kwargs: None))
    assert workflow.score() == {"status": "reused_valid"}
    assert calls == ["seal", "finish", "validate"]

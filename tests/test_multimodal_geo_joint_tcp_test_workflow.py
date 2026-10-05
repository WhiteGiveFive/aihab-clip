"""Frozen cleaned-test orchestration boundaries, tested without test labels."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_joint_tcp as parent
from multimodal import geo_helpfulness_joint_tcp_fit as fitting
from multimodal import geo_helpfulness_joint_tcp_test as study
from multimodal import geo_helpfulness_joint_tcp_test_registry as registry
from multimodal import geo_helpfulness_joint_tcp_test_report as reporting
from multimodal.geo_helpfulness_router_candidate_state import variant_feature_indices


@pytest.mark.parametrize("field,value", [
    ("fitting", True), ("recalibration", True), ("threshold_selection", True),
    ("test_guardrail_enforcement", True), ("winner_selection", True),
    ("producer_fold", 3), ("test_rows", 3378), ("training_seeds", [1]),
    ("router_execution", "float64"), ("test_label_access", "before_prediction"),
])
def test_config_rejects_changes_to_frozen_application(monkeypatch, field, value):
    config = study.expected_config()
    config[field] = value
    monkeypatch.setattr(study, "_json", lambda path: config)
    with pytest.raises(ValueError, match="configuration changed"):
        study._config()


@pytest.mark.parametrize("seed", [True, False, 0, 5, 1.0, "1", None])
def test_seed_validation_rejects_noncanonical_inputs(seed):
    with pytest.raises(ValueError, match="seed"):
        study._seed(seed)


@pytest.mark.parametrize("spec", parent.expected_config()["configs"][::4])
def test_router_inputs_preserve_base_column_order_then_image_and_geo(spec):
    base = np.arange(3 * 727, dtype=np.float64).reshape(3, 727)
    features = {
        "image_embedding": np.arange(3 * 1152, dtype=np.float32).reshape(3, 1152) + 10000,
        "geo_embedding": np.arange(3 * 64, dtype=np.float32).reshape(3, 64) + 20000,
    }
    matrix = study._matrix(base, features, spec)
    indices = variant_feature_indices(spec["variant"])
    np.testing.assert_array_equal(matrix[:, :len(indices)], base[:, indices].astype(np.float32))
    assert matrix.dtype == np.float32 and matrix.flags.c_contiguous
    assert matrix.shape == (3, spec["input_dim"])
    if spec["with_embeddings"]:
        np.testing.assert_array_equal(matrix[:, len(indices):-64], features["image_embedding"])
        np.testing.assert_array_equal(matrix[:, -64:], features["geo_embedding"])


@pytest.mark.parametrize("kind", ["width", "nonfinite"])
def test_router_matrix_rejects_invalid_features(kind):
    spec = parent.expected_config()["configs"][0]
    base = np.zeros((2, 727))
    if kind == "width":
        spec["input_dim"] += 1
    else:
        base[0, 0] = np.nan
    with pytest.raises(ValueError, match="dimensions or values"):
        study._matrix(base, {}, spec)


def _minimal_native(q=(.6, .1, .2, .1), tcp=(.7, .3)):
    q = np.tile(np.asarray(q, np.float32), (2, 1))
    tcp = np.tile(np.asarray(tcp, np.float32), (2, 1))
    return {"q": q, "tcp_hat": tcp, "state_logits": np.log(q),
            "tcp_logits": np.log(tcp / (1 - tcp))}


@pytest.fixture
def infer_bundle(monkeypatch):
    config = parent.expected_config()
    config["configs"] = [config["configs"][0]]
    spec = config["configs"][0]
    features = {"row_uid": np.array(["a", "b"]), "plot_idx": np.array(["p1", "p2"]),
                "producer_fold": np.array([-1, -1])}
    for prefix in study.MODES:
        features[f"{prefix}_logits"] = np.zeros((2, 18))
    base = np.zeros((2, 727))
    state = {"config_id": spec["config_id"], "training_seed": 1,
             "input_dim": spec["input_dim"], "auxiliary_weight": spec["auxiliary_weight"],
             "parameters": config["parameters"], "fit_row_count": 3378}
    names = parent.feature_names(spec)
    monkeypatch.setattr(study, "apply_base_inputs", lambda *args: base.copy())
    monkeypatch.setattr(study, "_validate_stage", lambda *args, **kwargs: {})
    monkeypatch.setattr(study, "_json", lambda path: names if path.name == "features.json" else state)
    monkeypatch.setattr(reporting, "policy_actions", lambda rows, *args: rows[["row_uid"]].copy())

    def forbidden(*args, **kwargs):
        pytest.fail("Frozen router inference attempted fitting")

    monkeypatch.setattr(fitting, "fit_joint", forbidden)
    return config, features, state, names


def test_native_predictions_are_authoritative_and_float64_is_only_diagnostic(infer_bundle, monkeypatch):
    config, features, _, _ = infer_bundle
    native = _minimal_native()
    portable = _minimal_native((.05, .8, .1, .05), (.2, .8))
    calls = []

    def predict(state, matrix, *, native):
        calls.append(native)
        return {key: value.copy() for key, value in (native_values if native else portable).items()}

    native_values = native
    monkeypatch.setattr(fitting, "predict_joint", predict)
    rows, _, _, diagnostics = study._infer(1, features, {}, config, {})
    assert calls == [True, True, False]
    np.testing.assert_array_equal(rows[[f"q_{k}" for k in config["target_order"]]].to_numpy(), native["q"])
    np.testing.assert_array_equal(rows[["fusion_tcp_hat", "geo_tcp_hat"]].to_numpy(), native["tcp_hat"])
    spec_diagnostics = diagnostics[config["configs"][0]["config_id"]]
    assert spec_diagnostics["native_float32_exact"] is True
    assert spec_diagnostics["float64_state_argmax_changes"] == 2
    assert spec_diagnostics["float64_tcp_order_changes"] == 2
    assert not any("label" in key or "target_state" == key for key in rows)


def test_native_repeat_inconsistency_stops_inference(infer_bundle, monkeypatch):
    config, features, _, _ = infer_bundle
    calls = []

    def predict(*args, **kwargs):
        values = _minimal_native()
        calls.append(1)
        if len(calls) == 2:
            values["q"][0, 0] += .01
        return values

    monkeypatch.setattr(fitting, "predict_joint", predict)
    with pytest.raises(ValueError, match="not repeatable"):
        study._infer(1, features, {}, config, {})
    assert len(calls) == 2


@pytest.mark.parametrize("field", ["training_seed", "input_dim", "auxiliary_weight", "fit_row_count", "features"])
def test_inference_rejects_wrong_fitted_identity_and_feature_order(infer_bundle, field):
    config, features, state, names = infer_bundle
    if field == "features":
        names.reverse()
    else:
        state[field] = -999
    with pytest.raises(ValueError, match="identity or recipe|feature ordering"):
        study._infer(1, features, {}, config, {})


@pytest.fixture
def declaration(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    output = root / "output"
    output.mkdir()
    prior = root / "parent"
    (prior / "scores").mkdir(parents=True)
    (prior / "state.json").write_text('{"frozen":true}')
    (prior / "declaration.json").write_text('{"config":{"frozen":true}}')
    (prior / "scores/selections.json").write_text('{"saved":true}')
    events = root / "events"
    events.mkdir()
    (events / "old.json").write_text('{"old":true}')
    for name, value in (("code.py", "frozen code"), ("config.json", "{}"),
                        ("source.parquet", "frozen source"), ("image.jpg", "frozen image")):
        (root / name).write_text(value)
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "OUTPUT", output)
    monkeypatch.setattr(study, "PARENT", prior)
    monkeypatch.setattr(study, "CONFIG", root / "config.json")
    monkeypatch.setattr(study, "CODE_PATHS", ("code.py",))
    monkeypatch.setattr(study, "_config", lambda: {"study_id": "joint_tcp_router_test_v1"})
    saved = {
        "study_id": "joint_tcp_router_test_v1", "config": study._config(),
        "parent_config": {"frozen": True}, "adaptive_reuse": True, "no_go_preserved": True,
        "source_file_sha256": {name: study._hash(root / name) for name in ("code.py", "config.json", "source.parquet")},
        "image_file_sha256": {"image.jpg": study._hash(root / "image.jpg")},
        "parent_snapshot": study._snapshot(prior),
        "registry_root": str(events), "prior_registry_snapshot": study._snapshot(events),
        "validation_selections": {"saved": True},
    }
    (output / "declaration.json").write_text(json.dumps(saved))
    return root, output, prior, events, saved


@pytest.mark.parametrize("path", ["code.py", "config.json", "source.parquet", "image.jpg", "parent/state.json", "events/old.json"])
def test_declaration_detects_source_parent_and_registry_changes(declaration, path):
    root, _, _, _, saved = declaration
    assert study.validate_declaration(check_parents=True) == saved
    (root / path).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        study.validate_declaration(check_parents=True)


def test_declaration_binds_exact_thresholds_even_if_parent_snapshot_is_updated(declaration):
    _, output, prior, _, saved = declaration
    (prior / "scores/selections.json").write_text('{"changed":true}')
    saved["parent_snapshot"] = study._snapshot(prior)
    (output / "declaration.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="validation policies changed"):
        study.validate_declaration()


def test_registry_may_append_without_mutating_prior_events(declaration):
    _, _, _, events, saved = declaration
    (events / "new.json").write_text('{"new":true}')
    assert study.validate_declaration(check_parents=True) == saved


def test_parent_bundle_rejects_untracked_additions(declaration):
    _, _, prior, _, _ = declaration
    (prior / "new-state.json").write_text("{}")
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
        events.append(("prediction", seed, kwargs.get("replay")))

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


def test_all_four_exact_replays_and_reservation_precede_label_acquisition(scoring_boundary):
    with pytest.raises(AcquisitionReached):
        study.score()
    assert scoring_boundary == [
        ("validate", True), *(("prediction", seed, True) for seed in study.SEEDS),
        ("build", study.SEEDS), ("reserve",), ("acquire",),
    ]


def test_last_seed_failure_prevents_reservation_and_labels(scoring_boundary, monkeypatch):
    def predictions(seed, **kwargs):
        scoring_boundary.append(("prediction", seed))
        if seed == 4:
            raise ValueError("Missing fourth prediction bundle")

    monkeypatch.setattr(study, "_predictions", predictions)
    with pytest.raises(ValueError, match="fourth"):
        study.score()
    assert not any(event[0] in {"build", "reserve", "acquire"} for event in scoring_boundary)


@pytest.fixture
def prediction_stage(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    declaration = {"preprocessing": {"1": {}}, "parent_config": {}, "validation_selections": {}}
    (output / "declaration.json").write_text(json.dumps(declaration))
    feature_dir = output / "features/seed_1"
    feature_dir.mkdir(parents=True)
    (feature_dir / "manifest.json").write_text("{}")
    rows = pd.DataFrame({"row_uid": ["a", "b"], "q_rescue": np.array([.6, .1], np.float32),
                         "state_logits": [np.array([1, 2, 3, 4], np.float32)] * 2})
    actions = pd.DataFrame({"row_uid": ["a", "b"], "fusion": [False, False], "policy": [True, False]})
    base = np.zeros((2, 727))
    diagnostics = {"native_float32_exact": True, "maximum_abs_error": 1e-8}
    monkeypatch.setattr(study, "OUTPUT", output)
    monkeypatch.setattr(study, "validate_declaration", lambda **kwargs: declaration)
    monkeypatch.setattr(study, "_features", lambda *args, **kwargs: {})
    monkeypatch.setattr(study, "_infer", lambda *args: (rows, actions, base, diagnostics))
    return output, rows, actions, base, diagnostics


@pytest.mark.parametrize("corruption", [None, "label", "dtype", "probability", "one_ulp", "logit_one_ulp", "action", "base", "diagnostics", "lineage"])
def test_prediction_stage_replay_detects_tampering(prediction_stage, corruption):
    output, rows, actions, base, diagnostics = prediction_stage
    saved_rows, saved_actions = rows.copy(deep=True), actions.copy(deep=True)
    saved_base, saved_diagnostics = base.copy(), deepcopy(diagnostics)
    if corruption == "label":
        saved_rows["label_id_dense"] = [0, 1]
    elif corruption == "dtype":
        saved_rows["q_rescue"] = saved_rows.q_rescue.astype(np.float64)
    elif corruption == "probability":
        saved_rows.loc[0, "q_rescue"] = np.float32(.9)
    elif corruption == "one_ulp":
        saved_rows.loc[0, "q_rescue"] = np.nextafter(
            saved_rows.loc[0, "q_rescue"], np.float32(1), dtype=np.float32
        )
    elif corruption == "logit_one_ulp":
        changed = saved_rows.loc[0, "state_logits"].copy()
        changed[0] = np.nextafter(changed[0], np.float32(2), dtype=np.float32)
        saved_rows.at[0, "state_logits"] = changed
    elif corruption == "action":
        saved_actions.loc[0, "policy"] = False
    elif corruption == "base":
        saved_base[0, 0] = 1
    elif corruption == "diagnostics":
        saved_diagnostics["maximum_abs_error"] = 2e-8
    staged = study._staging(output, "synthetic")
    saved_rows.to_parquet(staged / "rows.parquet", index=False)
    saved_actions.to_parquet(staged / "actions.parquet", index=False)
    study._save_arrays(staged / "numeric_inputs.npz", {"X727": saved_base})
    study._write_json(staged / "replay.json", saved_diagnostics)
    study._seal_stage(staged, output / "predictions/seed_1", output,
                      role="joint_tcp_cleaned_test_predictions", metadata={
                          "seed": 1, "features_sha256": "f" * 64 if corruption == "lineage" else
                          study._hash(output / "features/seed_1/manifest.json")})
    if corruption is None:
        observed_rows, observed_actions = study._predictions(1, replay=True)
        pd.testing.assert_frame_equal(observed_rows, rows)
        pd.testing.assert_frame_equal(observed_actions, actions)
    else:
        with pytest.raises((ValueError, AssertionError)):
            study._predictions(1, replay=True)


def test_predict_seed_reuse_preserves_every_byte_and_mtime(prediction_stage):
    output, _, _, _, _ = prediction_stage
    assert study.predict_seed(1)["status"] == "complete"
    before = study._snapshot(output)
    assert study.predict_seed(1) == {"status": "reused_valid", "seed": 1}
    assert study._snapshot(output) == before


@pytest.mark.parametrize("corruption", [None, "plot", "missing_label", "duplicate_label", "action", "extra_action"])
def test_scoring_joins_only_canonical_identity_and_rechecks_sealed_actions(monkeypatch, corruption):
    frames, actions = {}, {}
    for seed in study.SEEDS:
        frames[seed] = pd.DataFrame({"row_uid": ["b", "a"], "plot_idx": ["p2", "p1"],
                                    "training_seed": [seed, seed], "q_rescue": [.1, .7]})
        actions[seed] = pd.DataFrame({"row_uid": ["a", "b"], "training_seed": [seed, seed],
                                     "fusion": [False, False], "policy": [True, False]})
    actual = pd.concat(list(actions.values()), ignore_index=True).sort_values(
        ["row_uid", "training_seed"]
    ).reset_index(drop=True)
    labels = pd.DataFrame({"row_uid": ["a", "b"], "plot_idx": ["p1", "p2"], "label_id_dense": [3, 7]})
    if corruption == "plot":
        labels.loc[0, "plot_idx"] = "wrong"
    elif corruption == "missing_label":
        labels = labels.iloc[:1]
    elif corruption == "duplicate_label":
        labels = pd.concat([labels, labels.iloc[:1]], ignore_index=True)
    elif corruption == "action":
        actions[4].loc[0, "policy"] = False
    elif corruption == "extra_action":
        for frame in actions.values():
            frame["unapproved_policy"] = False
    config, selections = {"frozen_recipe": True}, {"frozen_thresholds": True}

    def fixed_policy(rows, passed_config, passed_selections):
        assert passed_config is config and passed_selections is selections
        assert "label_id_dense" not in rows
        return actual

    monkeypatch.setattr(reporting, "policy_actions", fixed_policy)
    monkeypatch.setattr(study, "_predictions", lambda seed: (frames[seed], actions[seed]))
    declaration = {"parent_config": config, "validation_selections": selections}
    if corruption is None:
        scored, choices = study._scoring_inputs(labels, declaration)
        assert scored.label_id_dense.tolist() == [7, 3] * 4
        assert scored.label_id_dense.dtype == np.int64
        pd.testing.assert_frame_equal(choices, actual)
        assert all("label_id_dense" not in frame for frame in frames.values())
    else:
        with pytest.raises((ValueError, AssertionError)):
            study._scoring_inputs(labels, declaration)


def test_report_markdown_uses_serialized_json_order(tmp_path, monkeypatch):
    result = {"report": {"z": {"second": 1}, "a": {"first": 2}},
              "actions": pd.DataFrame({"fusion": [False]}),
              "bootstrap_draws": np.zeros((2, 1, 2)), "bootstrap_policy_order": ["fusion"]}
    monkeypatch.setattr(reporting, "format_report", lambda report: "|".join(report))
    study._save_report(tmp_path, result, pd.DataFrame({"row_uid": ["a"]}))
    assert (tmp_path / "report.md").read_text() == "a|z"
    assert list(result["report"]) == ["z", "a"]


def _runner():
    path = Path(__file__).resolve().parents[1] / "tools/run_multimodal_geo_helpfulness_joint_tcp_test.py"
    spec = importlib.util.spec_from_file_location("joint_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_build_orders_predictions_before_scoring_and_never_fits(monkeypatch):
    runner = _runner()
    calls = []
    monkeypatch.setattr(runner.sys, "argv", ["runner", "build", "--device", "cpu"])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command[2:]) or SimpleNamespace(returncode=0))
    assert runner.main() == 0
    expected = [["declare"]]
    for seed in study.SEEDS:
        expected += [["extract-seed", "--seed", str(seed), "--device", "cpu"], ["predict-seed", "--seed", str(seed)]]
    assert calls == [*expected, ["score"], ["validate"]]


def test_cli_stops_before_scoring_when_inference_fails(monkeypatch):
    runner = _runner()
    calls = []
    monkeypatch.setattr(runner.sys, "argv", ["runner", "build"])

    def run(command, **kwargs):
        calls.append(command[2:])
        return SimpleNamespace(returncode=17 if command[2:4] == ["predict-seed", "--seed"] else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main() == 17
    assert not any(command[0] in {"score", "validate"} for command in calls)

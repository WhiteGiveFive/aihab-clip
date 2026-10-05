from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_joint_tcp as study
from multimodal import geo_helpfulness_joint_tcp_fit as fitting


def _payloads():
    arrays = {}
    for split, count in (("train", 4), ("validation", 2)):
        image = np.zeros((count, 1152), dtype=np.float32)
        image[:, 0] = 1
        arrays[split] = {
            "row_uid": np.asarray([f"{split}{i}" for i in range(count)]),
            "plot_idx": np.asarray([f"{split}p{i}" for i in range(count)]),
            "producer_fold": np.arange(count, dtype=np.int64) if split == "train"
                else np.full(count, -1, dtype=np.int64),
            "image_embedding": image,
            "geo_embedding": np.full((count, 64), 2, dtype=np.float32),
            **{f"{name}_logits": np.zeros((count, 18), dtype=np.float64)
               for name in ("raw", "geo", "image")},
        }
    arrays["train"]["labels"] = np.zeros(4, dtype=np.int64)
    for name, predictions in (("raw", [1, 0, 0, 1]), ("geo", [0, 1, 0, 1])):
        arrays["train"][f"{name}_logits"][np.arange(4), predictions] = 3.0
    return arrays


def _router_data(arrays):
    data = {"y_train": np.arange(4, dtype=np.int64)}
    for split in study.SPLITS:
        count = len(arrays[split]["row_uid"])
        data[f"X_{split}"] = np.tile(np.arange(727, dtype=np.float64), (count, 1))
        for name in ("row_uid", "plot_idx"):
            data[f"{split}_{name}"] = arrays[split][name].copy()
    return data


def _predictions(count):
    return {
        "state_logits": np.zeros((count, 4), dtype=np.float32),
        "tcp_logits": np.zeros((count, 2), dtype=np.float32),
        "q": np.full((count, 4), .25, dtype=np.float32),
        "tcp_hat": np.full((count, 2), .5, dtype=np.float32),
    }


def test_full_grid_widths_orders_and_fixed_recipe():
    config = study._config()
    assert config == study.expected_config()
    assert len(config["configs"]) == 16
    assert len(config["training_seeds"]) * len(config["configs"]) == 64
    assert config["parameters"] == fitting.DEFAULT_PARAMETERS
    grouped = {}
    for spec in config["configs"]:
        grouped.setdefault(spec["input_id"], []).append(spec)
        names = study.feature_names(spec)
        base_dim = 79 if spec["variant"] == "ablation_no_pairs79" else 25
        assert len(names) == spec["input_dim"] == base_dim + 1216 * spec["with_embeddings"]
        assert names[:base_dim] == study.full_feature_columns()[:base_dim]
        if spec["with_embeddings"]:
            assert names[base_dim] == "image_embedding_0000"
            assert names[base_dim + 1151] == "image_embedding_1151"
            assert names[base_dim + 1152] == "geo_embedding_00"
            assert names[-1] == "geo_embedding_63"
        assert not any("pred_pair" in name for name in names)
    assert len(grouped) == 4
    assert all({spec["auxiliary_weight"] for spec in group} == {0, .1, 1, 10}
               for group in grouped.values())


def test_returned_configuration_cannot_mutate_fixed_training_defaults():
    config = study.expected_config()
    config["parameters"]["epochs"] = 299
    config["parameters"]["betas"][0] = .5
    assert fitting.DEFAULT_PARAMETERS["epochs"] == 300
    assert fitting.DEFAULT_PARAMETERS["betas"] == [.9, .999]


@pytest.mark.parametrize("change", [
    lambda c: c["parameters"].update(epochs=299),
    lambda c: c.update(locked_test_access=True),
    lambda c: c["configs"][0].update(auxiliary_weight=.01),
    lambda c: c.update(training_seeds=[1]),
])
def test_config_rejects_undeclared_scientific_changes(monkeypatch, change):
    config = deepcopy(study.expected_config())
    change(config)
    monkeypatch.setattr(study, "_json", lambda path: config)
    with pytest.raises(ValueError, match="approved grid or recipe"):
        study._config()


@pytest.fixture
def input_bundle(tmp_path, monkeypatch):
    arrays = _payloads()
    data = _router_data(arrays)
    monkeypatch.setattr(study, "_validate_stage", lambda *args, **kwargs: {})
    monkeypatch.setattr(study, "_load_arrays", lambda path: deepcopy(data))
    monkeypatch.setattr(study, "_json", lambda path: study.full_feature_columns())
    return tmp_path, arrays, data


def test_tcp_targets_use_native_t1_true_class_probability_in_fusion_geo_order(input_bundle):
    root, arrays, _ = input_bundle
    result = study._inputs(root, 1, arrays)
    native_correct = np.exp(3) / (np.exp(3) + 17)
    native_incorrect = 1 / (np.exp(3) + 17)
    expected = np.asarray([[native_incorrect, native_correct],
                           [native_correct, native_incorrect],
                           [native_correct, native_correct],
                           [native_incorrect, native_incorrect]])
    np.testing.assert_allclose(result["tcp_targets"], expected, rtol=1e-15)
    calibrated_correct = np.exp(3 / 2) / (np.exp(3 / 2) + 17)
    assert not np.isclose(result["tcp_targets"][1, 0], calibrated_correct)
    np.testing.assert_array_equal(result["y_train"], [0, 1, 2, 3])
    assert "labels" not in arrays["validation"]


@pytest.mark.parametrize("change,message", [
    (lambda a, d: d["validation_row_uid"].__setitem__(0, "wrong"), "identities disagree"),
    (lambda a, d: d["train_plot_idx"].__setitem__(0, "wrong"), "identities disagree"),
    (lambda a, d: d["y_train"].__setitem__(0, 1), "correctness disagrees"),
    (lambda a, d: a["train"]["labels"].__setitem__(0, 1), "correctness disagrees"),
    (lambda a, d: d.update(X_train=np.zeros((4, 726))), "dimensions changed"),
])
def test_router_inputs_reject_mismatched_rows_labels_or_width(input_bundle, change, message):
    root, arrays, data = input_bundle
    change(arrays, data)
    with pytest.raises(ValueError, match=message):
        study._inputs(root, 1, arrays)


def test_matrix_contains_declared_blocks_without_embedding_rescaling():
    arrays = _payloads()
    data = _router_data(arrays)
    for spec in study.expected_config()["configs"]:
        matrix = study._matrix(data, arrays, spec, "train")
        base_dim = 79 if spec["variant"] == "ablation_no_pairs79" else 25
        assert matrix.dtype == np.float32 and matrix.flags.c_contiguous
        np.testing.assert_array_equal(matrix[:, :base_dim], data["X_train"][:, :base_dim])
        if spec["with_embeddings"]:
            np.testing.assert_array_equal(matrix[:, base_dim:base_dim + 1152], arrays["train"]["image_embedding"])
            np.testing.assert_array_equal(matrix[:, -64:], arrays["train"]["geo_embedding"])


@pytest.fixture
def output_roots(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    base = root / "studies"
    base.mkdir(parents=True)
    old = base / "older_study"
    old.mkdir()
    (old / "immutable.txt").write_text("untouched")
    output = base / "joint_tcp_router_v1"
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "BASE", base)
    monkeypatch.setattr(study, "OUTPUT", output)
    return root, base, old, output


def test_outputs_protect_existing_studies_and_repository(output_roots):
    root, base, old, output = output_roots
    before = study._snapshot(old)
    for path in (root, root.parent, base, old, old / "child"):
        with pytest.raises(ValueError, match="disjoint|cannot contain"):
            study._output(path)
    assert study._output(output) == output
    assert before == study._snapshot(old)


@pytest.fixture
def model_bundle(tmp_path, monkeypatch):
    config = study.expected_config()
    spec = config["configs"][0]
    matrices = {"train": np.zeros((4, 25), dtype=np.float32),
                "validation": np.zeros((2, 25), dtype=np.float32)}
    manifest = {"metadata": {"seed": 1, "config_id": spec["config_id"],
                             "features_manifest": "digest", "inputs_manifest": "digest"}}
    state = {"training_seed": 1, "config_id": spec["config_id"], "input_dim": 25,
             "fit_row_count": 4, "auxiliary_weight": 0, "parameters": config["parameters"]}
    predictions = {f"{split}_{key}": value
                   for split, x in matrices.items() for key, value in _predictions(len(x)).items()}
    feature_root = tmp_path / "features/seed_1"
    feature_root.mkdir(parents=True)
    for split, payload in _payloads().items():
        np.savez(feature_root / f"{split}.npz", **payload)
        for key in ("row_uid", "plot_idx", "producer_fold"):
            predictions[f"{split}_{key}"] = payload[key].copy()
    replay = {split: {key: {"maximum_abs_error": 0, "mean_abs_error": 0}
                      for key in study.PREDICTION_KEYS} for split in study.SPLITS}
    documents = {"state.json": state, "features.json": study.feature_names(spec),
                 "replay.json": replay, "history.json": [{"epoch": i} for i in range(1, 301)]}
    monkeypatch.setattr(study, "_config", lambda: config)
    monkeypatch.setattr(study, "_validate_stage", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(study, "_hash", lambda path: "digest")
    monkeypatch.setattr(study, "_json", lambda path: documents[Path(path).name])
    monkeypatch.setattr(study, "_load_arrays", lambda path: predictions)
    monkeypatch.setattr(fitting, "predict_joint", lambda state, x, native=True: _predictions(len(x)))
    return tmp_path, spec, matrices, state, predictions, manifest


def test_model_replays_native_predictions_with_declared_identity(model_bundle):
    root, spec, matrices, _, predictions, _ = model_bundle
    result = study._model(root, 1, spec, matrices)
    for split in study.SPLITS:
        for key in study.PREDICTION_KEYS:
            np.testing.assert_array_equal(result[split][key], predictions[f"{split}_{key}"])


@pytest.mark.parametrize("field,value", [
    ("training_seed", 2), ("config_id", "another_config"), ("input_dim", 79),
    ("fit_row_count", 5), ("auxiliary_weight", 1), ("parameters", {}),
])
def test_model_rejects_state_configuration_mismatch(model_bundle, field, value):
    root, spec, matrices, state, _, _ = model_bundle
    state[field] = value
    with pytest.raises(ValueError, match="identity or training recipe"):
        study._model(root, 1, spec, matrices)


def test_model_rejects_wrong_feature_parent_or_native_prediction(model_bundle):
    root, spec, matrices, _, predictions, manifest = model_bundle
    manifest["metadata"]["features_manifest"] = "changed"
    with pytest.raises(ValueError, match="provenance changed"):
        study._model(root, 1, spec, matrices)
    manifest["metadata"]["features_manifest"] = "digest"
    predictions["validation_q"][0, 0] += np.float32(1e-7)
    with pytest.raises(ValueError, match="Native float32 replay differs"):
        study._model(root, 1, spec, matrices)


def test_model_rejects_prediction_identity_drift(model_bundle):
    root, spec, matrices, _, predictions, _ = model_bundle
    predictions["validation_producer_fold"][0] = 3
    with pytest.raises(ValueError, match="prediction identities differ"):
        study._model(root, 1, spec, matrices)


def test_all_64_models_verified_before_scoring_reads_labels(tmp_path, monkeypatch):
    config = study.expected_config()
    arrays, events = _payloads(), []
    monkeypatch.setattr(study, "validate_declaration", lambda *a, **kw: {"config": config})
    monkeypatch.setattr(study, "_features", lambda *args: arrays)
    monkeypatch.setattr(study, "_inputs", lambda *args: _router_data(arrays))
    monkeypatch.setattr(study, "_hash", lambda path: "manifest")
    assignment_path = study.PROTOCOL / "protocol/development_assignments.parquet"
    monkeypatch.setattr(study, "_json", lambda path: {
        "file_sha256": {str(assignment_path.relative_to(study.ROOT)): "manifest"}
    })

    def model(root, seed, spec, matrices):
        events.append((seed, spec["config_id"]))
        return {split: _predictions(len(x)) for split, x in matrices.items()}

    def read_labels(path, columns):
        assert len(events) == 64
        assert set(events) == {(seed, spec["config_id"]) for seed in study.SEEDS for spec in config["configs"]}
        events.append("labels")
        assert columns == ["row_uid", "plot_idx", "development_role", "label_id_dense"]
        return pd.concat([pd.DataFrame({
            "row_uid": data["row_uid"], "plot_idx": data["plot_idx"],
            "development_role": split, "label_id_dense": 0,
        }) for split, data in arrays.items()], ignore_index=True)

    monkeypatch.setattr(study, "_model", model)
    monkeypatch.setattr(study.pd, "read_parquet", read_labels)
    rows, parents, verified_at = study._scored_rows(tmp_path)
    assert events[-1] == "labels" and verified_at
    assert len(parents) == 64
    assert len(rows) == 64 * 6
    assert {"row_uid", "plot_idx", "producer_fold", "training_seed", "config_id"} <= set(rows)
    assert {"state_logits", "tcp_logits"} <= set(rows)
    assert "labels" not in arrays["validation"]


def test_any_failed_prediction_prevents_validation_label_join(tmp_path, monkeypatch):
    def failed(root):
        raise ValueError("unsealed or unreproducible prediction")

    monkeypatch.setattr(study, "_all_predictions", failed)
    monkeypatch.setattr(study.pd, "read_parquet",
                        lambda *a, **kw: pytest.fail("labels must remain unread"))
    with pytest.raises(ValueError, match="unsealed"):
        study._scored_rows(tmp_path)


def test_changed_assignment_hash_prevents_scoring_label_read(tmp_path, monkeypatch):
    assignment_path = study.PROTOCOL / "protocol/development_assignments.parquet"
    monkeypatch.setattr(study, "_all_predictions", lambda root: (pd.DataFrame(), {}))
    monkeypatch.setattr(study, "_json", lambda path: {
        "file_sha256": {str(assignment_path.relative_to(study.ROOT)): "declared"}
    })
    monkeypatch.setattr(study, "_hash", lambda path: "changed")
    monkeypatch.setattr(study.pd, "read_parquet",
                        lambda *a, **kw: pytest.fail("changed labels must remain unread"))
    with pytest.raises(ValueError, match="assignments changed"):
        study._scored_rows(tmp_path)


def test_prepare_passes_all_five_keyed_producers_and_reuses_cache(output_roots, monkeypatch):
    from multimodal import geo_helpfulness_joint_tcp_extract as extraction

    _, _, old, root = output_roots
    root.mkdir()
    (root / "declaration.json").write_text("{}")
    payloads = _payloads()
    producers, calls = {}, []
    for fold in (*study.FOLDS, None):
        payload = payloads["validation"] if fold is None else {
            key: value[fold:fold + 1] for key, value in payloads["train"].items()
        }
        producers[fold] = {"payload": payload, "metadata": {"training_seed": 1, "producer_fold": fold}}
    monkeypatch.setattr(study, "validate_declaration", lambda *a, **kw: {})
    monkeypatch.setattr(study, "_features", lambda *args: payloads)
    monkeypatch.setattr(study, "_producer", lambda root, seed, fold: producers[fold])

    def extract(seed, fold, device):
        calls.append((seed, fold, device))
        return producers[fold]

    def combine(values, split):
        assert isinstance(values, dict)
        assert set(values) == {0, 1, 2, 3, None}
        return payloads[split]

    monkeypatch.setattr(extraction, "extract_producer", extract)
    monkeypatch.setattr(extraction, "combine_producers", combine)
    previous = study._snapshot(old)
    result = study.prepare_seed(1, root, device="cpu")
    assert result["status"] == "complete"
    assert calls == [(1, fold, "cpu") for fold in (*study.FOLDS, None)]
    first = study._snapshot(root)
    result = study.prepare_seed(1, root, device="cpu")
    assert result["status"] == "reused_valid"
    assert study._snapshot(root) == first
    assert len(calls) == 5 and study._snapshot(old) == previous


def test_fit_receives_only_train_targets_and_reuses_sealed_artifacts(output_roots, monkeypatch):
    _, _, old, root = output_roots
    root.mkdir()
    (root / "declaration.json").write_text("{}")
    for relative in ("features/seed_1/manifest.json", "router_inputs/manifest.json"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    arrays, config = _payloads(), study.expected_config()
    config["configs"] = config["configs"][:1]
    data = _router_data(arrays)
    data["tcp_targets"] = np.full((4, 2), .5, dtype=np.float64)
    calls = []
    monkeypatch.setattr(study, "validate_declaration", lambda *a, **kw: {"config": config})
    monkeypatch.setattr(study, "_features", lambda *args: arrays)
    monkeypatch.setattr(study, "_inputs", lambda *args: data)

    def fit(train, labels, tcp, prediction_inputs, **kwargs):
        calls.append(kwargs["config_id"])
        np.testing.assert_array_equal(labels, data["y_train"])
        np.testing.assert_array_equal(tcp, data["tcp_targets"])
        assert labels.shape == (4,) and tcp.shape == (4, 2)
        assert set(prediction_inputs) == {"validation"}
        assert prediction_inputs["validation"].shape == (2, 25)
        assert kwargs["parameters"] == fitting.DEFAULT_PARAMETERS
        assert "labels" not in arrays["validation"]
        return {"state": {}, "history": [], "metadata": {}, "predictions": {
            "train": _predictions(4), "validation": _predictions(2),
        }}

    def validate_saved(root, seed, spec, matrices):
        folder = root / f"models/seed_{seed}/{spec['config_id']}"
        study._validate_stage(folder, root, role="joint_tcp_model")
        return {split: _predictions(len(x)) for split, x in matrices.items()}

    monkeypatch.setattr(fitting, "fit_joint", fit)
    monkeypatch.setattr(fitting, "predict_joint", lambda state, x, native=True: _predictions(len(x)))
    monkeypatch.setattr(study, "_model", validate_saved)
    previous = study._snapshot(old)
    study.fit_seed(1, root)
    first = study._snapshot(root / "models")
    study.fit_seed(1, root)
    assert len(calls) == 1
    assert study._snapshot(root / "models") == first
    assert study._snapshot(old) == previous
    folder = root / f"models/seed_1/{config['configs'][0]['config_id']}"
    (folder / "state.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="artifact changed"):
        study.fit_seed(1, root)
    assert len(calls) == 1


def test_declaration_detects_changed_upstream_and_prior_study(output_roots, monkeypatch):
    repo, _, old, root = output_roots
    root.mkdir()
    code = repo / "implementation.py"
    code.write_text("frozen code")
    parent = repo / "expert.bin"
    parent.write_bytes(b"frozen weights")
    config_path = repo / "config.json"
    config_path.write_text("{}")
    config = {"study_id": "joint_tcp_router_v1"}
    monkeypatch.setattr(study, "CONFIG", config_path)
    monkeypatch.setattr(study, "CODE_PATHS", ("implementation.py",))
    monkeypatch.setattr(study, "_config", lambda: config)
    declaration = {
        "schema_version": "geo_helpfulness.joint_tcp_router_declaration.v1",
        "config": config, "locked_test_access": False,
        "file_sha256": {p.name: study._hash(p) for p in (code, parent, config_path)},
        "prior_study_snapshots": {old.name: study._snapshot(old)},
    }
    (root / "declaration.json").write_text(json.dumps(declaration))
    study.validate_declaration(root, check_parents=True)
    parent.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Declared source changed"):
        study.validate_declaration(root, check_parents=True)
    parent.write_bytes(b"frozen weights")
    (old / "immutable.txt").write_text("changed old result")
    with pytest.raises(ValueError, match="Prior experiment changed"):
        study.validate_declaration(root, check_parents=True)

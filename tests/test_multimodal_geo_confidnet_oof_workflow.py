from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_confidnet_oof as workflow


@pytest.mark.parametrize("change", [
    "confidence_width", "confidence_epochs", "router_full_features", "rf_depth",
    "direct_embeddings", "true_tcp", "bootstrap", "policies", "schema",
])
def test_configuration_rejects_unapproved_experimental_changes(tmp_path, monkeypatch, change):
    config = workflow._json(workflow.CONFIG)
    if change == "confidence_width":
        config["confidence_parameters"]["hidden_widths"] = [128] * 4
    elif change == "confidence_epochs":
        config["confidence_parameters"]["epochs"] = 100
    elif change == "router_full_features":
        config["router_base_variant"] = "full727"
    elif change == "rf_depth":
        config["router_parameters"]["max_depth"] = 5
    elif change == "direct_embeddings":
        config["optional_direct_embeddings"] = True
    elif change == "true_tcp":
        config["optional_true_tcp_router"] = True
    elif change == "bootstrap":
        config["bootstrap"]["resamples"] = 10
    elif change == "policies":
        config["policies"].append("validation_selected_threshold")
    else:
        config["schema_version"] = "different_experiment"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(workflow, "CONFIG", path)
    with pytest.raises(ValueError):
        workflow._config()


def _payload(role, *, count=None):
    count = count if count is not None else (3378 if role == "train" else 822)
    result = {
        "row_uid": np.asarray([f"{role}_row_{i:04d}" for i in range(count)]),
        "plot_idx": np.asarray([f"{role}_plot_{i:04d}" for i in range(count)]),
        "producer_fold": (np.arange(count, dtype=np.int64) % 4 if role == "train"
                          else np.full(count, -1, dtype=np.int64)),
    }
    for mode in workflow.MODES:
        result[f"{mode}_hidden"] = np.zeros((count, 256), dtype=np.float32)
        result[f"{mode}_logits"] = np.zeros((count, 18), dtype=np.float64)
    if role == "train":
        result["labels"] = np.arange(count, dtype=np.int64) % 18
    return result


@pytest.mark.parametrize("change", [
    "label_leak", "plot_overlap", "row_overlap", "plot_crosses_fold", "missing_fold",
    "validation_fold", "hidden_dtype", "logit_dtype", "unsorted", "nonfinite",
])
def test_feature_boundary_rejects_labels_identity_or_producer_corruption(change):
    train, validation = _payload("train"), _payload("validation")
    workflow._feature_check(train, validation)
    if change == "label_leak":
        validation["labels"] = np.zeros(822, dtype=np.int64)
    elif change == "plot_overlap":
        validation["plot_idx"][0] = train["plot_idx"][0]
    elif change == "row_overlap":
        validation["row_uid"][0] = train["row_uid"][0]
    elif change == "plot_crosses_fold":
        train["plot_idx"][1] = train["plot_idx"][0]
    elif change == "missing_fold":
        train["producer_fold"][train["producer_fold"] == 3] = 2
    elif change == "validation_fold":
        validation["producer_fold"][0] = 0
    elif change == "hidden_dtype":
        train["image_hidden"] = train["image_hidden"].astype(np.float64)
    elif change == "logit_dtype":
        validation["raw_logits"] = validation["raw_logits"].astype(np.float32)
    elif change == "unsorted":
        train["row_uid"][[0, 1]] = train["row_uid"][[1, 0]]
    else:
        train["geo_hidden"][0, 0] = np.inf
    with pytest.raises(ValueError):
        workflow._feature_check(train, validation)


def _tables(arrays, seed):
    result = {}
    for role, data in arrays.items():
        table = pd.DataFrame({name: data[name] for name in ("row_uid", "plot_idx", "producer_fold")})
        table["training_seed"] = seed
        for mode in workflow.MODES:
            table[f"{mode}_tcp_hat"] = np.full(len(table), 0.5, dtype=np.float32)
            table[f"{mode}_logits"] = list(data[f"{mode}_logits"])
        result[role] = table
    return result


def _mock_valid_confidence(monkeypatch):
    arrays = {role: _payload(role, count=4) for role in ("train", "validation")}
    monkeypatch.setattr(workflow, "validate_declaration", lambda *args, **kwargs: {"config": {}})
    monkeypatch.setattr(workflow, "_validate_features", lambda root, seed: arrays)
    monkeypatch.setattr(workflow, "_hash", lambda path: str(path))

    def stage(path, root, *, role):
        seed = int(path.parent.name.split("_")[-1])
        return {"file_sha256": {"train.parquet": "x", "validation.parquet": "x"},
                "metadata": {"seed": seed, "head_manifests": {
                    mode: str(root / f"confidence/seed_{seed}/{mode}/manifest.json")
                    for mode in workflow.MODES}}}

    monkeypatch.setattr(workflow, "_validate_stage", stage)
    monkeypatch.setattr(workflow, "_prediction_parents", lambda root: {})
    base_inputs = {}
    for seed in workflow.SEEDS:
        base_inputs[seed] = {}
        for role, data in arrays.items():
            base_inputs[seed][f"X_{role}"] = np.zeros((4, 25), dtype=np.float64)
            for name in ("row_uid", "plot_idx"):
                base_inputs[seed][f"{role}_{name}"] = data[name]
    monkeypatch.setattr(workflow, "_base_inputs", lambda root: base_inputs)
    return arrays


@pytest.mark.parametrize("failure", ["last_head", "last_router", "labels_reached"])
def test_score_verifies_all_twelve_heads_and_eight_routers_before_label_read(
    tmp_path, monkeypatch, failure
):
    arrays = _mock_valid_confidence(monkeypatch)
    events = []

    def head(root, seed, mode, data):
        events.append(("head", seed, mode))
        if failure == "last_head" and (seed, mode) == (4, "raw"):
            raise FileNotFoundError("last head absent")
        return {role: np.full(4, 0.5, dtype=np.float32) for role in arrays}

    def router(root, seed, arm, matrices):
        events.append(("router", seed, arm))
        if failure == "last_router" and (seed, arm) == (4, "augmented"):
            raise FileNotFoundError("last router absent")
        return {role: np.full((4, 4), 0.25) for role in arrays}

    def read(path, **kwargs):
        if path.name == "development_assignments.parquet":
            events.append(("labels",))
            assert events[:-1] == (
                [("head", seed, mode) for seed in workflow.SEEDS for mode in workflow.MODES]
                + [("router", seed, arm) for seed in workflow.SEEDS for arm in workflow.ARMS]
            )
            raise RuntimeError("all models verified before label read")
        seed = int(path.parent.parent.name.split("_")[-1])
        return _tables(arrays, seed)[path.stem]

    monkeypatch.setattr(workflow, "_validate_head", head)
    monkeypatch.setattr(workflow, "_validate_router", router)
    monkeypatch.setattr(pd, "read_parquet", read)
    expected = RuntimeError if failure == "labels_reached" else FileNotFoundError
    with pytest.raises(expected):
        workflow.score(tmp_path / "experiment")
    assert sum(event[0] == "head" for event in events) == 12
    assert sum(event[0] == "router" for event in events) == (0 if failure == "last_head" else 8)
    assert sum(event[0] == "labels" for event in events) == (failure == "labels_reached")


def test_fit_routers_checks_last_confidence_seed_before_loading_router_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "validate_declaration", lambda *args, **kwargs: {})
    seen = []

    def features(root, seed):
        seen.append(seed)
        if seed == 4:
            raise FileNotFoundError("fourth confidence seed missing")
        return {}

    monkeypatch.setattr(workflow, "_validate_features", features)
    monkeypatch.setattr(workflow, "_validate_confidence_tables", lambda *args: {})
    monkeypatch.setattr(workflow, "_base_inputs", lambda root: pytest.fail("router inputs loaded early"))
    with pytest.raises(FileNotFoundError, match="fourth confidence seed"):
        workflow.fit_routers(tmp_path / "experiment")
    assert seen == [1, 2, 3, 4]


@pytest.mark.parametrize("stage", ["fit_routers", "score", "fit_seed", "extract_seed"])
def test_sealed_training_pause_blocks_all_following_stages(tmp_path, monkeypatch, stage):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "declaration.json").write_text("{}")
    monkeypatch.setattr(workflow, "validate_declaration", lambda *args, **kwargs: {})
    with pytest.raises(workflow.StudyPausedError):
        workflow._pause(root, 2, "image", None, "five zero-gradient epochs")
    monkeypatch.setattr(workflow, "_validate_features", lambda *args: pytest.fail("features loaded after pause"))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("labels read after pause"))
    with pytest.raises(workflow.StudyPausedError, match="requires user review"):
        function = getattr(workflow, stage)
        if stage.endswith("seed"):
            function(1, root)
        else:
            function(root)


@pytest.mark.parametrize("change", ["tamper", "extra", "missing", "declaration", "role"])
def test_sealed_artifact_hashes_and_declaration_lineage_are_checked(tmp_path, change):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "declaration.json").write_text("{}")
    staged = workflow._staging(root, "fixture")
    (staged / "predictions.json").write_text("[0.1]")
    final = root / "fixture"
    workflow._seal_stage(staged, final, root, role="fixture")
    workflow._validate_stage(final, root, role="fixture")
    role = "fixture"
    if change == "tamper":
        (final / "predictions.json").write_text("[0.9]")
    elif change == "extra":
        (final / "extra.txt").write_text("unexpected")
    elif change == "missing":
        (final / "predictions.json").unlink()
    elif change == "declaration":
        (root / "declaration.json").write_text('{"changed":true}')
    else:
        role = "other_role"
    with pytest.raises(ValueError):
        workflow._validate_stage(final, root, role=role)


@pytest.mark.parametrize("change", ["row_uid", "plot_idx", "wrong_width"])
def test_router_append_preserves_exact_confidence_row_alignment(change):
    arrays = {role: _payload(role, count=4) for role in ("train", "validation")}
    tables = _tables(arrays, 1)
    base = {}
    for role, values in arrays.items():
        base[f"X_{role}"] = np.arange(100, dtype=np.float64).reshape(4, 25)
        for name in ("row_uid", "plot_idx"):
            base[f"{role}_{name}"] = values[name]
    result = workflow._router_matrices(base, tables)
    assert result["train"]["baseline"].shape == (4, 25)
    assert result["train"]["augmented"].shape == (4, 28)
    np.testing.assert_array_equal(result["train"]["augmented"][:, :25], base["X_train"])
    np.testing.assert_array_equal(result["train"]["augmented"][:, 25:], 0.5)
    if change == "wrong_width":
        base["X_train"] = np.zeros((4, 727), dtype=np.float64)
    else:
        tables["train"].loc[0, change] = "wrong_identity"
    with pytest.raises(ValueError):
        workflow._router_matrices(base, tables)


def test_confidence_table_metadata_must_match_requested_seed_and_heads(tmp_path, monkeypatch):
    arrays = _mock_valid_confidence(monkeypatch)
    monkeypatch.setattr(workflow, "_validate_stage", lambda *args, **kwargs: {
        "file_sha256": {"train.parquet": "x", "validation.parquet": "x"},
        "metadata": {"seed": 2, "head_manifests": {mode: "wrong" for mode in workflow.MODES}},
    })
    monkeypatch.setattr(workflow, "_validate_head", lambda *args: {
        role: np.full(4, 0.5, dtype=np.float32) for role in arrays
    })
    monkeypatch.setattr(pd, "read_parquet", lambda path: _tables(arrays, 1)[path.stem])
    with pytest.raises(ValueError):
        workflow._validate_confidence_tables(tmp_path, 1, arrays)


@pytest.mark.parametrize("change", ["seed", "row_count", "arm"])
def test_router_state_must_match_requested_seed_and_feature_arm(tmp_path, monkeypatch, change):
    from multimodal import geo_helpfulness_confidnet_oof_fit as fit

    state = {"training_seed": 1, "fit_row_count": 3378,
             "parameters": deepcopy(fit.RF_DEFAULTS),
             "feature_names": list(workflow.NUMERIC_COLUMNS)}
    if change == "seed":
        state["training_seed"] = 2
    elif change == "row_count":
        state["fit_row_count"] = 3377
    else:
        state["feature_names"] += [f"{mode}_tcp_hat" for mode in workflow.MODES]
    monkeypatch.setattr(workflow, "_hash", lambda path: "parent")
    monkeypatch.setattr(workflow, "_validate_stage", lambda *args, **kwargs: {
        "file_sha256": {name: "x" for name in ("state.json", "metadata.json", "predictions.npz")},
        "metadata": {"seed": 1, "arm": "baseline", "confidence_bundle_sha256": "parent",
                     "router_inputs_sha256": "parent"},
    })
    monkeypatch.setattr(workflow, "_json", lambda path: deepcopy(state))
    predictions = {role: np.full((4, 4), 0.25) for role in ("train", "validation")}
    monkeypatch.setattr(workflow, "_load_arrays", lambda path: predictions)
    monkeypatch.setattr(fit, "predict_router", lambda *args: np.full((4, 4), 0.25))
    matrices = {role: {"baseline": np.zeros((4, 25), dtype=np.float64)} for role in predictions}
    with pytest.raises(ValueError):
        workflow._validate_router(tmp_path, 1, "baseline", matrices)

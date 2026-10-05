"""Legacy expert/input authentication without reading any real test labels."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from multimodal import geo_helpfulness_joint_tcp_capacity_test_inputs as inputs
from multimodal import geo_helpfulness_router_numeric as numeric
from multimodal import geo_helpfulness_targets_features as semantic
from multimodal.geo_helpfulness_legacy_router_inference import (
    GEO_COLUMNS, IMAGE_COLUMNS, _head_logits, _identity,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _seal_context(value):
    value = deepcopy(value)
    value.pop("context_sha256", None)
    value["context_sha256"] = inputs.canonical_hash(value)
    return value


def _state(seed, mode):
    return {
        "schema_version": numeric.TEMPERATURE_SCHEMA_VERSION, "seed": seed,
        "mode": mode, "fit_role": "development_train_oof", "fit_row_count": 6,
        "fit_row_identity_sha256": "0" * 64, "class_count": 18,
        "weighting": "equal_per_image", "calibration_spec": numeric.CALIBRATION_SPEC,
        "safeguards": numeric.SAFEGUARDS, "theta": 0., "temperature": 1.,
        "native_nll": 1., "calibrated_nll": 1.,
        "optimizer": {"success": True, "status": 0, "message": "fixture", "nfev": 1, "nit": 1},
    }


@pytest.fixture
def bundle(tmp_path):
    rng = np.random.default_rng(310)
    original = pd.DataFrame({"file": [f"{i}.jpg" for i in range(6)],
                             "plot_idx": [f"p{i // 2}" for i in range(6)],
                             "label_id": [999] * 6})
    image = rng.normal(size=(6, 1152)).astype(np.float32)
    geo = rng.normal(size=(6, 64)).astype(np.float32)
    original = pd.concat([original, pd.DataFrame(image, columns=IMAGE_COLUMNS),
                          pd.DataFrame(geo, columns=GEO_COLUMNS)], axis=1)
    observed = _identity(original)
    original["row_uid"] = observed.row_uid
    ordered = original.sort_values("row_uid").reset_index(drop=True)
    source = tmp_path / "test.parquet"
    original.to_parquet(source, index=False)
    ontology = [{"label_name": f"class{i}"} for i in range(18)]
    mean, std = np.arange(64, dtype=np.float32) / 100, np.full(64, 2., np.float32)
    snapshots = {str(source): inputs.file_hash(source)}
    runs, rows, preprocessing = {mode: {} for mode in inputs.MODES}, [], {}
    for seed in inputs.SEEDS:
        part = ordered[["row_uid", "file", "plot_idx"]].copy()
        part["training_seed"] = seed
        part["checkpoint_family"] = "best_model"
        probabilities = []
        for index, (mode, prefix) in enumerate(inputs.MODES.items()):
            width = {"image_only": 1152, "geo_only": 64, "raw_concat": 1216}[mode]
            state = {"head.net.0.weight": torch.zeros((256, width)),
                     "head.net.0.bias": torch.ones(256),
                     "head.net.3.weight": torch.zeros((18, 256)),
                     "head.net.3.bias": torch.arange(18, dtype=torch.float32) / 100}
            state["head.net.3.bias"][index] = 2 + seed / 10
            checkpoint = {"model_state": state, "mode": mode, "epoch": 50,
                          "training_regime": "fixed_epoch_train_val"}
            run = tmp_path / mode / f"seed{seed}"
            run.mkdir(parents=True)
            torch.save(checkpoint, run / "best_model.pt")
            _write(run / "geo_standardization.json", {
                "fit_split": "train_val", "fit_rows": 4200, "modality_name": "geo",
                "feature_columns": list(GEO_COLUMNS), "class_names": [o["label_name"] for o in ontology],
                "label_id_remap": {str(i): i for i in range(18)}, "mean": mean.tolist(), "std": std.tolist(),
            })
            runs[mode][str(seed)] = str(run)
            for name in ("best_model.pt", "geo_standardization.json"):
                snapshots[str(run / name)] = inputs.file_hash(run / name)
            logits = _head_logits(checkpoint, np.zeros((6, width), dtype=np.float32))
            part[f"{prefix}_logits"] = list(logits)
            part[f"{prefix}_pred"] = logits.argmax(axis=1)
            probabilities.append(numeric.apply_expert_temperature(logits, _state(seed, mode)))
        features = semantic.build_router_feature_frame(*probabilities, probability_basis=semantic.CALIBRATED_PROBABILITY_BASIS)
        transform = numeric.fit_router_feature_transform(features, seed=seed,
            frozen_schema=semantic.build_router_feature_schema(), fit_row_identity_sha256="0" * 64)
        preprocessing[str(seed)] = {"temperatures": {mode: _state(seed, mode) for mode in inputs.MODES},
                                    "transform": transform}
        rows.append(part)
    experts = pd.concat(rows, ignore_index=True)
    experts["label_id_dense"] = 999  # Must never be projected by the loader.
    expert_path = tmp_path / "experts.parquet"
    experts.to_parquet(expert_path, index=False)
    snapshots[str(expert_path)] = inputs.file_hash(expert_path)
    historical = {}
    for family, prefix in (("rf", "relaxed__rf__"), ("lr", "relaxed__")):
        actions = []
        action_path = tmp_path / family / "actions.parquet"
        for variant in ("primary_full727", "ablation_no_pairs79", "ablation_numeric25"):
            pid = prefix + variant + "__plots_20__harm_0p7"
            threshold = {"comparison": "strict_gt", "kind": "finite", "value_hex": (0.).hex()}
            frame = experts[["checkpoint_family", *inputs.IDENTITY, "raw_pred", "geo_pred"]].copy()
            frame["policy_id"] = pid
            frame["score"] = np.tile([-.1, 0., .1], 8)
            frame["threshold_specification"] = json.dumps(threshold)
            frame["acted"] = frame.score > 0
            frame["final_pred"] = np.where(frame.acted, frame.geo_pred, frame.raw_pred)
            frame["label_id_dense"] = 999
            actions.append(frame)
            historical[pid] = {"family": family, "input_id": variant.split("_", 1)[1],
                               "guardrail_id": "plots_20__harm_0p7",
                               "frozen_policy": {"threshold_specification": threshold},
                               "actions_path": str(action_path)}
        action_path.parent.mkdir()
        pd.concat(actions, ignore_index=True).to_parquet(action_path, index=False)
        snapshots[str(action_path)] = inputs.file_hash(action_path)
    context = _seal_context({
        "schema_version": "geo_helpfulness.joint_tcp_capacity_test_inputs.v1",
        "checkpoint_family": "best_model", "source_file_sha256": snapshots,
        "experts_path": str(expert_path), "preprocessing": preprocessing,
        "ontology": ontology, "encoder_lineage": {"historical_encoder_to_cache_content_fingerprint_available": False},
        "identity_manifest": {"rows": ordered[["row_uid", "file", "plot_idx"]].to_dict("records")},
        "legacy_declaration": {"caches": {str(s): {"test": str(source)} for s in inputs.SEEDS}, "legacy_runs": runs},
        "historical_policies": historical,
    })
    return context


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_native_replay_and_exact_feature_order_without_labels(bundle, monkeypatch, seed):
    original = pd.read_parquet
    projections = []

    def read(path, *, columns, **kwargs):
        assert columns is not None and not any("label" in name or "target" in name for name in columns)
        projections.append(columns)
        return original(path, columns=columns, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    result = inputs.prepare_seed(seed, context=bundle)
    arrays = result["test"]
    assert len(projections) == 2
    assert arrays["X_numeric25"].shape == (6, 25)
    assert arrays["X_no_pairs79"].shape == (6, 79)
    assert arrays["X_numeric25"].dtype == arrays["X_no_pairs79"].dtype == np.float32
    np.testing.assert_array_equal(arrays["X_no_pairs79"], arrays["X_base727"][:, :79].astype(np.float32))
    np.testing.assert_array_equal(arrays["X_numeric25"], arrays["X_no_pairs79"][:, :25])
    assert arrays["X_no_pairs79"].flags.c_contiguous
    assert not any("label" in name or "target" in name for name in arrays)
    assert result["metadata"]["feature_names"]["no_pairs79"] == inputs.full_feature_columns()[:79]
    assert all(v["native_float32_logits_exact"] and v["predicted_classes_exact"] for v in result["metadata"]["native_replay"].values())
    assert result["metadata"]["test_labels_read"] is False


@pytest.mark.parametrize("seed", [True, 0, 5, 1., "1", None])
def test_invalid_seed_rejected_before_source_reads(seed):
    with pytest.raises(ValueError, match="seed"):
        inputs.prepare_seed(seed)


def test_mutated_context_is_rejected(bundle):
    bundle["preprocessing"]["1"]["transform"]["mean"][0] += 1
    with pytest.raises(ValueError):
        inputs.prepare_seed(1, context=bundle)


def test_alternate_preprocessing_rejected(bundle):
    state = deepcopy(bundle["preprocessing"]["1"])
    state["transform"]["mean"][0] += .1
    with pytest.raises(ValueError, match="preprocessing differs"):
        inputs.prepare_seed(1, context=bundle, preprocessing=state)


@pytest.mark.parametrize("source", ["cache", "head", "scaler", "experts", "actions"])
def test_source_mutation_rejected(bundle, source):
    from pathlib import Path
    path = {
        "cache": bundle["legacy_declaration"]["caches"]["1"]["test"],
        "head": bundle["legacy_declaration"]["legacy_runs"]["raw_concat"]["1"] + "/best_model.pt",
        "scaler": bundle["legacy_declaration"]["legacy_runs"]["raw_concat"]["1"] + "/geo_standardization.json",
        "experts": bundle["experts_path"],
        "actions": next(iter(bundle["historical_policies"].values()))["actions_path"],
    }[source]
    Path(path).write_bytes(Path(path).read_bytes() + b"changed")
    with pytest.raises(ValueError):
        inputs.prepare_seed(1, context=bundle)


@pytest.mark.parametrize("mutation", ["logit", "class", "duplicate", "plot", "seed"])
def test_authenticated_but_inconsistent_experts_rejected(bundle, mutation):
    frame = pd.read_parquet(bundle["experts_path"])
    if mutation == "logit":
        values = frame.at[0, "raw_logits"].copy()
        values[0] += 1e-6
        frame.at[0, "raw_logits"] = values
    elif mutation == "class":
        frame.at[0, "raw_pred"] = 4
    elif mutation == "duplicate":
        frame.at[0, "row_uid"] = frame.at[1, "row_uid"]
    elif mutation == "plot":
        frame.at[0, "plot_idx"] = "other"
    else:
        frame["training_seed"] = 8
    frame.to_parquet(bundle["experts_path"], index=False)
    bundle["source_file_sha256"][bundle["experts_path"]] = inputs.file_hash(Path(bundle["experts_path"]))
    with pytest.raises(ValueError):
        inputs.prepare_seed(1, context=_seal_context(bundle))


def test_historical_policies_use_strict_threshold_and_same_experts(bundle, monkeypatch):
    original = pd.read_parquet

    def read(path, *, columns, **kwargs):
        assert not any("label" in name or "target" in name for name in columns)
        return original(path, columns=columns, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    result = inputs.load_historical_actions(context=bundle)
    assert result["actions"].shape == (24, 11)
    for policy in result["metadata"]:
        assert result["actions"][policy].dtype == bool
        assert result["actions"][policy].sum() == 8


@pytest.mark.parametrize("mutation", ["threshold", "acted", "final", "expert", "duplicate"])
def test_inconsistent_historical_policy_rejected(bundle, mutation):
    path = next(iter(bundle["historical_policies"].values()))["actions_path"]
    frame = pd.read_parquet(path)
    if mutation == "threshold":
        value = json.loads(frame.at[0, "threshold_specification"])
        value["value_hex"] = (.1).hex()
        frame.at[0, "threshold_specification"] = json.dumps(value)
    elif mutation == "acted":
        frame.at[1, "acted"] = True  # score equals threshold exactly.
    elif mutation == "final":
        frame.at[0, "final_pred"] = 9
    elif mutation == "expert":
        frame.at[0, "raw_pred"] = 9
    else:
        frame.at[0, "row_uid"] = frame.at[1, "row_uid"]
    frame.to_parquet(path, index=False)
    bundle["source_file_sha256"][path] = inputs.file_hash(Path(path))
    with pytest.raises(ValueError):
        inputs.load_historical_actions(context=_seal_context(bundle))


def test_no_fitting_registry_or_score_reads_in_module():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(inputs.__file__).read_text())
    forbidden = {"fit_expert_temperature", "fit_router_feature_transform", "reserve_event", "acquire_labels", "make_report"}
    names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not forbidden & names
    assert "scores/" not in Path(inputs.__file__).read_text()


@pytest.mark.parametrize("mutation", [None, "hash", "fit_role", "duplicate", "missing"])
def test_parent_preprocessing_is_authenticated_and_complete(bundle, tmp_path, mutation):
    repo = tmp_path / "repo"
    parent = repo / "parent"
    temperatures = [state for seed in bundle["preprocessing"].values() for state in seed["temperatures"].values()]
    transforms = [seed["transform"] for seed in bundle["preprocessing"].values()]
    files = {}
    for name, states, schema in (
        ("expert_temperatures.json", temperatures, "geo_helpfulness.expert_temperatures.v1"),
        ("router_feature_transform.json", transforms, "geo_helpfulness.router_feature_transform.v1"),
    ):
        if name == "expert_temperatures.json":
            if mutation == "duplicate":
                states = [*states, states[0]]
            if mutation == "missing":
                states = states[:-1]
        path = repo / "preprocessing" / name
        value = {"schema_version": schema, "protocol_id": "protocol_v1",
                 "fit_role": "development_validation" if mutation == "fit_role" else "development_train_oof", "states": states}
        _write(path, value)
        files[str(path.relative_to(repo))] = inputs.file_hash(path)
        if mutation == "hash":
            path.write_text(path.read_text() + " ")
    _write(parent / "declaration.json", {"config": {"study_id": "joint_tcp_router_v1"}, "file_sha256": files})
    if mutation is not None:
        with pytest.raises(ValueError):
            inputs._preprocessing(repo, parent, {})
    else:
        snapshot = {}
        assert inputs._preprocessing(repo, parent, snapshot) == bundle["preprocessing"]
        assert len(snapshot) == 3


@pytest.mark.parametrize("mutation", [None, "payload", "declaration", "manifest_hash"])
def test_comparator_manifest_authentication(tmp_path, mutation):
    folder = tmp_path / "predictions"
    folder.mkdir()
    payload = folder / "actions.parquet"
    payload.write_bytes(b"fixture payload")
    declaration = {"study_id": "relaxed_guardrails_rf_v1", "declaration_sha256": "a" * 64}
    value = {**declaration, "artifacts": {"actions.parquet": {"file_sha256": inputs.file_hash(payload)}}}
    if mutation == "declaration":
        value["declaration_sha256"] = "b" * 64
    value["manifest_sha256"] = inputs.canonical_hash(value)
    if mutation == "manifest_hash":
        value["manifest_sha256"] = "c" * 64
    _write(folder / "manifest.json", value)
    if mutation == "payload":
        payload.write_bytes(b"altered")
    if mutation is not None:
        with pytest.raises(ValueError):
            inputs._manifest(folder, {}, declaration)
    else:
        snapshot = {}
        assert inputs._manifest(folder, snapshot, declaration) == value
        assert len(snapshot) == 2

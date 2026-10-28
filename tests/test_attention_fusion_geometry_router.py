"""Cohort alignment, artifact ownership and label-free R2 boundaries."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from multimodal import attention_fusion_geometry_router as router


def prediction_fixture():
    rows = pd.DataFrame({"row_uid": ["q0", "q1", "q2"],
                         "plot_idx": ["v0", "v0", "v1"], "label_id_dense": [0, 1, 2]})
    arrays = {"row_uid": rows.row_uid.to_numpy(copy=True), "plot_idx": rows.plot_idx.to_numpy(copy=True),
              "label": np.array([0, 1, 2]), "prediction": np.array([0, 2, 2]),
              "probabilities": np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]])}
    return rows, arrays


def test_prediction_exports_are_key_aligned_without_changing_source_arrays():
    rows, arrays = prediction_fixture()
    permutation = [2, 0, 1]
    shuffled = {key: values[permutation].copy() for key, values in arrays.items()}
    snapshots = {key: values.copy() for key, values in shuffled.items()}
    aligned = router.validate_prediction_arrays(shuffled, rows, 3)
    for key in arrays:
        np.testing.assert_array_equal(aligned[key], arrays[key])
        np.testing.assert_array_equal(shuffled[key], snapshots[key])


@pytest.mark.parametrize("mutation", [
    "duplicate_prediction_uid", "duplicate_query_uid", "wrong_cohort", "wrong_plot",
    "wrong_label", "out_of_range_label", "out_of_range_query_label", "float_label",
    "out_of_range_prediction", "float_prediction", "wrong_prediction_shape",
    "nan_probabilities", "negative_probabilities", "unnormalized_probabilities",
    "wrong_probability_width", "wrong_argmax",
])
def test_malformed_prediction_exports_fail_closed(mutation):
    rows, arrays = prediction_fixture()
    if mutation == "duplicate_prediction_uid":
        arrays["row_uid"][1] = arrays["row_uid"][0]
    elif mutation == "duplicate_query_uid":
        rows.loc[1, "row_uid"] = rows.loc[0, "row_uid"]
    elif mutation == "wrong_cohort":
        arrays["row_uid"][1] = "foreign"
    elif mutation == "wrong_plot":
        arrays["plot_idx"][0] = "foreign"
    elif mutation == "wrong_label":
        arrays["label"][0] = 1
    elif mutation == "out_of_range_label":
        arrays["label"][0] = 3
    elif mutation == "out_of_range_query_label":
        rows.loc[0, "label_id_dense"] = 3
        arrays["label"][0] = 3
    elif mutation == "float_label":
        arrays["label"] = arrays["label"].astype(float)
    elif mutation == "out_of_range_prediction":
        arrays["prediction"][0] = -1
    elif mutation == "float_prediction":
        arrays["prediction"] = arrays["prediction"].astype(float)
    elif mutation == "wrong_prediction_shape":
        arrays["prediction"] = arrays["prediction"][:, None]
    elif mutation == "nan_probabilities":
        arrays["probabilities"][0, 0] = np.nan
    elif mutation == "negative_probabilities":
        arrays["probabilities"][0] = [1.2, -0.1, -0.1]
    elif mutation == "unnormalized_probabilities":
        arrays["probabilities"][0] *= 0.9
    elif mutation == "wrong_probability_width":
        arrays["probabilities"] = arrays["probabilities"][:, :2]
    else:
        arrays["prediction"][0] = 1
    with pytest.raises(ValueError):
        router.validate_prediction_arrays(arrays, rows, 3)


@pytest.fixture
def base_config():
    return yaml.safe_load((router.REPO_ROOT / router.DEFAULT_CONFIG).read_text())


def save_config(path, config):
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("key, value", [
    ("seed", 2), ("candidate", "raw_concat"), ("initialization", "cs_adapted"),
    ("regime", "frozen_online"), ("checkpoint_policies", ["best"]),
    ("geometry_arms", ["pretrained"]), ("training_allowed", True), ("cleaned_test_allowed", True),
])
def test_config_rejects_scope_expansion(tmp_path, base_config, key, value):
    base_config["scope"][key] = value
    with pytest.raises(ValueError):
        router.load_config(save_config(tmp_path / "config.yaml", base_config))


@pytest.mark.parametrize("section, key, value", [
    ("geometry", "distance", "euclidean"), ("geometry", "support", "shared_effective_k"),
    ("geometry", "query_role", "test"), ("data", "train_images", 4200),
])
def test_declared_recipe_cannot_silently_differ_from_computation(tmp_path, base_config, section, key, value):
    base_config[section][key] = value
    with pytest.raises(ValueError, match="Declared"):
        router.load_config(save_config(tmp_path / "config.yaml", base_config))


@pytest.mark.parametrize("destination", [
    "multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router",
    "multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router/../foreign_v1",
    "multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_direct_v1",
])
def test_config_output_stays_in_new_side_study_namespace(tmp_path, base_config, destination):
    base_config["artifact_root"] = destination
    with pytest.raises(ValueError, match="separate router-study"):
        router.load_config(save_config(tmp_path / "config.yaml", base_config))


def test_config_output_symlink_cannot_escape_namespace(tmp_path, base_config, monkeypatch):
    monkeypatch.setattr(router, "REPO_ROOT", tmp_path)
    base = tmp_path / "multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router"
    base.mkdir(parents=True)
    outside = tmp_path / "completed_study"
    outside.mkdir()
    (base / "escape_v1").symlink_to(outside, target_is_directory=True)
    base_config["artifact_root"] = str(base / "escape_v1")
    with pytest.raises(ValueError, match="separate router-study"):
        router.load_config(save_config(tmp_path / "config.yaml", base_config))


@pytest.fixture
def prepared_sources(tmp_path, base_config, monkeypatch):
    """Real small source files exercise prepare without any model or real data."""
    config = deepcopy(base_config)
    source = tmp_path / "sources"
    output = tmp_path / "new_study"
    fusion_root, geo_root = source / "fusion", source / "geo"
    fusion_dir, geo_dir = fusion_root / "runs/fit", geo_root / "runs/fit"
    images = source / "images"
    images.mkdir(parents=True)
    config.update(fusion_study=str(fusion_root), geo_study=str(geo_root),
                  fusion_run="runs/fit", geo_run="runs/fit", artifact_root=str(output))
    config["data"] = dict(train_images=2, train_plots=2, validation_images=2,
                          validation_plots=2, classes=2, image_dim=1152, geo_dim=2)
    config_path = save_config(source / "config.yaml", config)
    source_config = source / "source.yaml"
    source_config.write_text(yaml.safe_dump({"development_universe": {
        "allowed_image_sources": [str(images)]}}))
    config["source_config"] = str(source_config)
    save_config(config_path, config)
    source_inputs = {"file_sha256": {}, "provenance": {"geo_source_files": []}}
    frames = {}
    for role, prefix in (("train", "t"), ("validation", "v")):
        frame = pd.DataFrame({"row_uid": [f"{prefix}0", f"{prefix}1"],
                              "plot_idx": [f"{prefix}p0", f"{prefix}p1"],
                              "file": [f"{prefix}0.jpg", f"{prefix}1.jpg"],
                              "label_id_dense": [0, 1], "label_name": ["class0", "class1"],
                              "development_role": [role, role], "image_source": [str(images)] * 2})
        frames[role] = frame
        for filename in frame.file:
            (images / filename).write_bytes(b"fixture image identity only")
        path = fusion_root / "inputs" / f"{role}_rows.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        source_inputs["file_sha256"][path.name] = router.sha256(path)
        geo_path = source / f"{role}_native.parquet"
        frame[["file", "plot_idx"]].assign(label_id=[0, 1], A00=[1.0, 0.0], A01=[0.0, 1.0]).to_parquet(
            geo_path, index=False)
        source_inputs["provenance"]["geo_source_files"].append(
            {"path": str(geo_path), "sha256": router.sha256(geo_path)})
    router.write_json(fusion_root / "inputs/manifest.json", source_inputs)
    pretrained = source / "pretrained"
    pretrained.mkdir()
    original = pretrained / "open_clip_model.safetensors"
    original.write_bytes(b"not loaded by preparation")
    provenance = {"pretrained": {"snapshot_path": str(pretrained),
                                "snapshot_file_sha256": {original.name: router.sha256(original)}}}
    for family, directory in (("fusion", fusion_dir), ("geo", geo_dir)):
        router.write_json(directory / "run.json", {
            "status": "complete", "seed": 1,
            "candidate_id": "attention_cls_skip_original" if family == "fusion" else "geo_only",
            "regime": "joint_partial" if family == "fusion" else "geo_only", "initialization": "pretrained"})
        router.write_json(directory / "run_spec.json", {"model_provenance": provenance})
        manifest = {"class_labels": ["class0", "class1"]}
        for policy in ("final", "best"):
            manifest[policy] = {"epoch": config["checkpoints"][family][policy], "metrics": {"acc1": 1.0}}
            if family == "fusion":
                weight = fusion_root / "objects" / f"{policy}.safetensors"
                weight.parent.mkdir(parents=True, exist_ok=True)
                weight.write_bytes(f"fixture {policy} weights".encode())
                manifest[policy]["visual"] = {"path": str(weight.relative_to(fusion_root)),
                                                "size": weight.stat().st_size, "sha256": router.sha256(weight)}
            frame = frames["validation"]
            router.write_npz(directory / f"{policy}_predictions.npz", {
                "row_uid": frame.row_uid.to_numpy(dtype=str), "plot_idx": frame.plot_idx.to_numpy(dtype=str),
                "label": np.array([0, 1]), "prediction": np.array([0, 1]),
                "probabilities": np.array([[0.9, 0.1], [0.1, 0.9]])})
        router.write_json(directory / "checkpoint_manifest.json", manifest)
    monkeypatch.setattr(router, "load_config", lambda path: (config, output))
    return config_path, source, output


def snapshot_tree(root):
    return {str(path.relative_to(root)): (router.sha256(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_prepare_preserves_real_source_files_and_reuses_owned_complete_output(prepared_sources):
    config, sources, output = prepared_sources
    before = snapshot_tree(sources)
    result = router.prepare(config)
    assert result["status"] == "prepared"
    assert snapshot_tree(sources) == before
    prepared = snapshot_tree(output)
    assert router.prepare(config)["status"] == "reused"
    assert snapshot_tree(output) == prepared
    declaration = router.verify_prepared(output)
    assert declaration["test_data_scored"] is False
    assert declaration["routing_applied"] is False
    assert len(declaration["comparisons"]) == 4


def test_prepare_cannot_overwrite_foreign_or_partial_output(prepared_sources):
    config, sources, output = prepared_sources
    output.mkdir()
    sentinel = output / "inputs/train_geo.npz"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"someone else's work")
    before = snapshot_tree(output)
    with pytest.raises(ValueError, match="new/empty|ownership|owned"):
        router.prepare(config)
    assert snapshot_tree(output) == before
    assert not (output / "protocol/declaration.json").exists()


def test_prepare_rejects_changed_bound_source_and_does_not_republish(prepared_sources):
    config, source, output = prepared_sources
    router.prepare(config)
    before = snapshot_tree(output)
    path = source / "fusion/runs/fit/run.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Recorded source changed"):
        router.prepare(config)
    assert snapshot_tree(output) == before


@pytest.fixture
def scoring_bundle(tmp_path, monkeypatch):
    """Prepared numeric bundle with no validation labels in any input archive."""
    output = tmp_path / "geometry"
    config = {"geometry": {"k": 5, "epsilon": 1e-8}}
    frames = {
        "train": pd.DataFrame({"row_uid": [f"t{i}" for i in range(10)],
                               "plot_idx": [f"tp{i}" for i in range(10)],
                               "label_id_dense": [0] * 5 + [1] * 5}),
        "validation": pd.DataFrame({"row_uid": ["q0", "q1", "q2"], "plot_idx": ["v0", "v1", "v2"]}),
    }
    vectors = {"train": np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5),
               "validation": np.array([[0.0, 1.0], [1.0, 0.0], [1, 1]])}
    vectors["validation"][2] /= np.sqrt(2)
    for role, frame in frames.items():
        path = output / "inputs" / f"{role}_rows.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        router.write_npz(output / "inputs" / f"{role}_geo.npz", {"geo_native": vectors[role]})
    for family, predictions in (("fusion", [0, 1, 0]), ("geo", [1, 0, 0])):
        router.write_npz(output / "inputs" / f"{family}_final_predictions.npz", {
            "prediction": np.array(predictions, dtype=int)})
    declaration = {"config": config, "config_sha256": router.canonical_hash(config),
                   "source_sha256": {}, "input_sha256": {
                       str(path.relative_to(output)): router.sha256(path) for path in (output / "inputs").iterdir()},
                   "class_labels": ["class0", "class1"], "encoders": {"pretrained": {}},
                   "comparisons": [{"comparison_id": "final__pretrained", "policy": "final",
                                    "geometry_encoder": "pretrained"}], "result_assurance": "fixture_only"}
    router.write_json(output / "protocol/declaration.json", declaration)
    folder = output / "embeddings/pretrained"
    for role, frame in frames.items():
        router.write_npz(folder / f"{role}.npz", {
            "row_uid": frame.row_uid.to_numpy(dtype=str), "plot_idx": frame.plot_idx.to_numpy(dtype=str),
            "image_embedding": np.pad(vectors[role], ((0, 0), (0, 1150)))})
    router.write_json(folder / "manifest.json", {
        "declaration_sha256": router.sha256(output / "protocol/declaration.json"),
        "file_sha256": {f"{role}.npz": router.sha256(folder / f"{role}.npz") for role in frames}})
    monkeypatch.setattr(router, "load_config", lambda path=router.DEFAULT_CONFIG: (config, output))
    return output


def test_scoring_works_without_query_labels_and_does_not_apply_routing(scoring_bundle):
    output = scoring_bundle
    result = router.score()
    assert result["status"] == "scored_without_routing"
    with np.load(output / "scores/final__pretrained.npz", allow_pickle=False) as saved:
        assert saved["score"][0] > 0
        assert saved["score"][1] > 0
        assert saved["score"][2] == 0
        np.testing.assert_array_equal(saved["eligible"], [True, True, False])
        assert not {"label", "label_id_dense", "acted", "routed_prediction", "accuracy"} & set(saved.files)
    manifest = router.read_json(output / "scores/manifest.json")
    for key in ("query_labels_used_for_geometry", "thresholds_selected", "routing_applied", "test_data_scored"):
        assert manifest[key] is False


@pytest.mark.parametrize("mutation", ["missing_role_digest", "wrong_declaration", "changed_embedding"])
def test_scoring_requires_bound_embedding_inputs(scoring_bundle, mutation):
    folder = scoring_bundle / "embeddings/pretrained"
    manifest_path = folder / "manifest.json"
    manifest = router.read_json(manifest_path)
    if mutation == "missing_role_digest":
        manifest["file_sha256"].pop("validation.npz")
        router.write_json(manifest_path, manifest)
    elif mutation == "wrong_declaration":
        manifest["declaration_sha256"] = "0" * 64
        router.write_json(manifest_path, manifest)
    else:
        with (folder / "validation.npz").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match="Embedding"):
        router.score()
    assert not (scoring_bundle / "scores/manifest.json").exists()


def test_complete_small_bundle_verifies_actual_counts_without_labels(scoring_bundle):
    router.score()
    summary = router.verify()
    assert summary["status"] == "R1_R2_complete"
    assert (summary["reference_images"], summary["reference_plots"],
            summary["query_images"], summary["query_plots"]) == (10, 10, 3, 3)
    assert summary["image_vectors"] == 13
    assert summary["routing_applied"] is False
    assert summary["test_data_scored"] is False


@pytest.mark.parametrize("mutation", ["missing_score", "missing_encoder_binding", "missing_comparison",
                                      "used_labels", "used_test"])
def test_verify_requires_complete_scores_and_stage_boundaries(scoring_bundle, mutation):
    output = scoring_bundle
    router.score()
    path = output / "scores/manifest.json"
    manifest = router.read_json(path)
    if mutation == "missing_score":
        manifest["file_sha256"].pop("final__pretrained.npz")
    elif mutation == "missing_encoder_binding":
        manifest["embedding_manifest_sha256"].clear()
    elif mutation == "missing_comparison":
        manifest["comparisons"] = []
    elif mutation == "used_labels":
        manifest["query_labels_used_for_geometry"] = True
    else:
        manifest["test_data_scored"] = True
    router.write_json(path, manifest)
    with pytest.raises(ValueError):
        router.verify()
    assert not (output / "verification/completion.json").exists()

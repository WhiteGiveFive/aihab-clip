"""Check separately versioned protocol and resumable orchestration without GPUs."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from multimodal import attention_fusion_bypass_protocol as protocol
from tools import run_attention_fusion_bypass_study as study

REPO = Path(__file__).resolve().parents[1]


def parent_config():
    return yaml.safe_load((REPO / protocol.PARENT_CONFIG).read_text())


def test_extension_matches_declared_recipe_and_preserves_parent_inputs():
    parent = parent_config()
    before = copy.deepcopy(parent)
    config = yaml.safe_load((REPO / protocol.DEFAULT_CONFIG).read_text())
    protocol.validate_config(config, parent)
    assert parent == before
    assert config["study_id"] != parent["study_id"]
    assert config["initialization_seed_namespace"] == parent["study_id"]
    for key in ("training", "model", "data", "geo", "image", "metrics", "artifacts"):
        assert config[key] == parent[key]
    assert {k: v for k, v in config["paths"].items() if k != "artifact_root"} == {
        k: v for k, v in parent["paths"].items() if k != "artifact_root"}
    assert config["source_study"]["config_sha256"] == protocol.content_hash(parent)
    assert config["final_fit"]["routes"] == ["development_train"]
    assert config["final_fit"]["test_evaluation"] is False


@pytest.mark.parametrize("change", ["rate", "namespace", "output", "test", "refit", "head"])
def test_undeclared_scientific_changes_are_rejected(change):
    parent = parent_config()
    config = protocol.expected_config(parent)
    if change == "rate":
        config["training"]["learning_rates"].append(0.01)
    elif change == "namespace":
        config["initialization_seed_namespace"] = config["study_id"]
    elif change == "output":
        config["paths"]["artifact_root"] = parent["paths"]["artifact_root"]
    elif change == "test":
        config["data"]["allow_test_features_or_labels"] = True
    elif change == "refit":
        config["final_fit"]["routes"].append("all_development")
    else:
        config["model"]["classifier_hidden_dim"] *= 2
    with pytest.raises(protocol.BypassProtocolError, match="contract"):
        protocol.validate_config(config, parent)


def test_matrix_is_seventy_two_fits_with_twenty_four_references_and_no_refit():
    config = protocol.expected_config(parent_config())
    matrix = protocol.build_matrix(config)
    assert matrix["counts"] == {"candidate_geo_arms": 6, "development_fits": 72,
        "selected_train_checkpoint_references": 24, "all_development_refits": 0}
    fits = matrix["development_fits"]
    assert len({f["fit_id"] for f in fits}) == len({f["output_dir"] for f in fits}) == 72
    assert matrix["all_development_refits"] == []
    for arm in matrix["candidate_geo_arms"]:
        arm_fits = [f for f in fits if f["arm_id"] == arm["arm_id"]]
        assert {(f["seed"], f["learning_rate"]) for f in arm_fits} == {
            (seed, rate) for seed in range(1, 5) for rate in (0.0001, 0.0003, 0.001)}
    for fit in fits:
        assert fit["fit_role"] == "train" and fit["fit_rows"] == 3378
        assert fit["epochs"] == fit["checkpoint_epoch"] == 50
        assert fit["evaluation_role"] == "validation"
        assert Path(fit["output_dir"]).is_relative_to(config["paths"]["artifact_root"])
        assert "attention_fusion_v1/inputs/" in fit["inputs"]["image_caches"]["train"]
    for selected in matrix["selected_train_checkpoint_references"]:
        assert len(selected["candidate_fit_ids"]) == 3 and selected["retrain"] is False


@pytest.fixture
def declared_fixture(tmp_path, monkeypatch):
    parent = parent_config()
    config = protocol.expected_config(parent)
    path = tmp_path / "bypass.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(protocol, "read_parent_plan", lambda *args: (parent, {}, {"manifest_payload_sha256": "old"}))
    parent_file = tmp_path / "old-evidence.json"
    parent_file.write_text('{"existing": true}')
    evidence = {"file_sha256": {parent_file.name: protocol.file_hash(parent_file)}}
    monkeypatch.setattr(protocol, "_authenticate_parent_evidence", lambda *args: evidence)
    sources = {"new_module.py": "original source hash"}
    monkeypatch.setattr(protocol, "source_hashes", lambda *args: sources.copy())
    return path, tmp_path, config, parent_file, sources


def test_declaration_binds_contrasts_and_is_immutable(declared_fixture):
    path, repo, config, parent_file, sources = declared_fixture
    before = parent_file.read_bytes(), parent_file.stat().st_mtime_ns
    manifest = protocol.declare_protocol(path, repo)
    assert set(manifest["files"]) == {"resolved_protocol.json", "candidate_matrix.json", "parent_evidence.json", "contrasts.json"}
    directory = repo / config["paths"]["artifact_root"] / "protocol"
    declaration = protocol.read_json(directory / "contrasts.json")["declaration"]
    assert len(declaration["contrasts"]) == 37 and len(declaration["primary_contrast_ids"]) == 2
    saved = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    assert protocol.declare_protocol(path, repo) == manifest
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()} == saved
    assert (parent_file.read_bytes(), parent_file.stat().st_mtime_ns) == before
    assert protocol.read_frozen_plan(path, repo)[0] == config
    protocol.verify_parent_bindings(config, repo)
    sources["new_module.py"] = "changed"
    with pytest.raises(protocol.BypassProtocolError, match="source changed"):
        protocol.read_frozen_plan(path, repo)


@pytest.mark.parametrize("mutation", ["contrast", "parent"])
def test_declaration_detects_evidence_mutation(declared_fixture, mutation):
    path, repo, config, parent_file, _ = declared_fixture
    protocol.declare_protocol(path, repo)
    if mutation == "contrast":
        changed = repo / config["paths"]["artifact_root"] / "protocol/contrasts.json"
        changed.write_text('{}')
        with pytest.raises(protocol.BypassProtocolError, match="payload changed"):
            protocol.read_frozen_plan(path, repo)
    else:
        parent_file.write_text("changed old evidence")
        with pytest.raises(protocol.BypassProtocolError, match="Reused parent artifact changed"):
            protocol.verify_parent_bindings(config, repo)


def test_declaration_rejects_preexisting_scientific_fit(declared_fixture):
    path, repo, config, _, _ = declared_fixture
    fit = protocol.build_matrix(config)["development_fits"][0]
    directory = repo / fit["output_dir"]
    directory.mkdir(parents=True)
    (directory / "run_spec.json").write_text('{}')
    with pytest.raises(protocol.BypassProtocolError, match="before scientific fitting"):
        protocol.declare_protocol(path, repo)


def test_load_fit_inputs_authenticates_old_config_but_binds_new_recipe(monkeypatch, tmp_path):
    from multimodal import attention_fusion_inputs as inputs

    parent = parent_config()
    config = protocol.expected_config(parent)
    calls = []
    shared_provenance = {"producer": "old", "encoder": {"sha256": "encoder"}}
    monkeypatch.setattr(protocol, "read_parent_plan", lambda *args: (parent, {}, {}))

    def load(configuration, seed, role, *, repo_root):
        assert configuration == parent and configuration != config
        assert seed == 3 and repo_root == tmp_path
        calls.append(role)
        return {"role": role}

    monkeypatch.setattr(inputs, "load_prepared_inputs", load)
    monkeypatch.setattr(inputs, "load_input_provenance", lambda configuration, seed, **kwargs: shared_provenance.copy())
    train, validation, provenance = protocol.load_fit_inputs(config, 3, tmp_path)
    assert calls == ["train", "validation"]
    assert train == {"role": "train"} and validation == {"role": "validation"}
    assert provenance["protocol_config_sha256"] == protocol.content_hash(config)
    assert provenance["source_protocol_config_sha256"] == protocol.content_hash(parent)
    assert provenance["train_role"] == "train" and provenance["validation_role"] == "validation"
    assert "protocol_config_sha256" not in shared_provenance


@pytest.fixture
def workload(tmp_path, monkeypatch):
    config = protocol.expected_config(parent_config())
    config["paths"]["artifact_root"] = "new_study"
    fits = protocol.build_matrix(config)["development_fits"][:3]
    matrix = {"development_fits": fits}
    sources = {name: f"source_{index}" for index, name in enumerate(sorted(study.MODEL_SOURCE_FILES))}
    sources["tools/run_attention_fusion_bypass_study.py"] = "orchestrator"
    input_path = tmp_path / "frozen-input"
    input_path.write_text("unchanged")
    monkeypatch.setattr(study, "read_frozen_plan", lambda *a: (config, matrix, {"manifest_payload_sha256": "p"}))
    monkeypatch.setattr(study, "source_hashes", lambda *a: sources.copy())
    monkeypatch.setattr(study, "preflight", lambda *a: {
        "input_file_sha256": {str(input_path): study.file_hash(input_path)},
        "execution": {"device": "cuda", "cpu_threads": 1}, "protocol_payload_sha256": "p"})
    calls = []

    def complete(config, fit, **kwargs):
        directory = tmp_path / fit["output_dir"]
        if (directory / "manifest.json").exists():
            assert kwargs["resume"] is True
            return {"status": "reused_complete"}
        calls.append((fit["fit_id"], kwargs["resume"]))
        directory.mkdir(parents=True, exist_ok=True)
        spec = {"study_id": config["study_id"], "candidate_id": fit["candidate_id"], "geo_mode": fit["geo_preprocessing"],
                "learning_rate": fit["learning_rate"], "seed": fit["seed"], "verification": False,
                "fit_role": "train", "epochs": 50, "config_sha256": study.content_hash(config),
                "source_hashes": {Path(name).name: value for name, value in sources.items() if name in study.MODEL_SOURCE_FILES},
                "input_provenance": {"protocol_config_sha256": study.content_hash(config),
                    "source_protocol_config_sha256": config["source_study"]["config_sha256"],
                    "train_role": "train", "validation_role": "validation"}}
        for name in study.REQUIRED_BUNDLE_FILES:
            (directory / name).write_text("{}")
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "run_spec.json").write_text(json.dumps(spec))
        (directory / "run.json").write_text(json.dumps({"status": "complete", "completed_epoch": 50}))
        manifest = {"status": "complete", "completed_epoch": 50, "verification": False,
                    "class_map": config["data"]["dense_to_canonical_l3"],
                    "boundary": {"test_data_loaded": False, "encoder_executed": False},
                    "trainable_parameters": config["candidates"][0]["expected_trainable_parameters"],
                    "run_identity_sha256": study.content_hash(spec),
                    "files": {name: study.file_hash(directory / name) for name in study.REQUIRED_BUNDLE_FILES}}
        manifest["manifest_payload_sha256"] = study.content_hash(manifest)
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return {"status": "complete"}

    monkeypatch.setattr(study, "execute_fit", complete)
    return {"repo": tmp_path, "config": config, "matrix": matrix, "calls": calls,
            "sources": sources, "complete": complete, "input": input_path}


def test_orchestration_completes_and_reuses_without_changing_bundle_bytes(workload):
    data = workload
    result = study.run_study(repo=data["repo"])
    assert result["counts"] == {"complete": 3}
    assert data["calls"] == [(f["fit_id"], False) for f in data["matrix"]["development_fits"]]
    bundle = data["repo"] / data["matrix"]["development_fits"][0]["output_dir"]
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in bundle.iterdir()}
    study.run_study(repo=data["repo"])
    assert len(data["calls"]) == 3
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in bundle.iterdir()} == before


def test_orchestration_stops_at_failure_and_resumes_partial(workload, monkeypatch):
    data = workload
    failed_fit = data["matrix"]["development_fits"][1]

    def fail(config, fit, **kwargs):
        if fit == failed_fit:
            directory = data["repo"] / fit["output_dir"]
            directory.mkdir(parents=True)
            (directory / "run.json").write_text(json.dumps({"status": "interrupted_or_failed", "completed_epoch": 12}))
            raise RuntimeError("simulated interruption")
        return data["complete"](config, fit, **kwargs)

    monkeypatch.setattr(study, "execute_fit", fail)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        study.run_study(repo=data["repo"])
    ledger = study.read_json(data["repo"] / "new_study/orchestration/m3_bypass/status.json")
    assert ledger["status"] == "failed_or_interrupted" and ledger["active_fit"] == failed_fit["fit_id"]
    assert not (data["repo"] / data["matrix"]["development_fits"][2]["output_dir"]).exists()
    monkeypatch.setattr(study, "execute_fit", data["complete"])
    study.run_study(repo=data["repo"])
    assert data["calls"][1] == (failed_fit["fit_id"], True)


def test_corrupted_bundle_is_never_reused(workload):
    data = workload
    study.run_study(repo=data["repo"])
    path = data["repo"] / data["matrix"]["development_fits"][0]["output_dir"] / "final.pt"
    path.write_text("corrupted")
    with pytest.raises(study.StudyError, match="payload changed"):
        study.run_study(repo=data["repo"])
    assert len(data["calls"]) == 3
    assert study.matrix_status(data["config"], data["matrix"], data["repo"])["counts"] == {"invalid_artifacts": 1, "complete": 2}


@pytest.mark.parametrize("change", ["source", "input"])
def test_midrun_source_or_input_change_stops_before_next_fit(workload, monkeypatch, change):
    data = workload

    def mutate(config, fit, **kwargs):
        result = data["complete"](config, fit, **kwargs)
        if change == "source":
            data["sources"]["multimodal/attention_fusion_bypass.py"] = "changed"
        else:
            data["input"].write_text("changed during fitting")
        return result

    monkeypatch.setattr(study, "execute_fit", mutate)
    with pytest.raises(study.StudyError, match="changed while"):
        study.run_study(repo=data["repo"])
    assert len(data["calls"]) == 1


def test_frozen_run_cannot_resume_with_changed_source(workload):
    data = workload
    study.run_study(repo=data["repo"])
    data["sources"]["multimodal/attention_fusion_bypass.py"] = "changed"
    with pytest.raises(study.StudyError, match="Orchestration identity changed"):
        study.run_study(repo=data["repo"])


def test_runner_rejects_fit_not_in_full_protocol_before_loading_inputs(monkeypatch):
    config = protocol.expected_config(parent_config())
    fit = copy.deepcopy(protocol.build_matrix(config)["development_fits"][0])
    fit["epochs"] = 51
    monkeypatch.setattr(study, "load_fit_inputs", lambda *args: pytest.fail("Inputs loaded before rejecting undeclared fit"))
    with pytest.raises(study.StudyError, match="72-fit bypass matrix"):
        study.execute_fit(config, fit)


def test_new_bundle_cannot_claim_original_three_file_source(workload):
    data = workload
    study.run_study(repo=data["repo"])
    fit = data["matrix"]["development_fits"][0]
    path = data["repo"] / fit["output_dir"]
    spec = study.read_json(path / "run_spec.json")
    spec["source_hashes"].pop("attention_fusion_bypass_training.py")
    (path / "run_spec.json").write_text(json.dumps(spec))
    manifest = study.read_json(path / "manifest.json")
    manifest["run_identity_sha256"] = study.content_hash(spec)
    manifest["files"]["run_spec.json"] = study.file_hash(path / "run_spec.json")
    manifest["manifest_payload_sha256"] = study.content_hash({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"})
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(study.StudyError, match="model/training source differs"):
        study.authenticate_complete(path, fit, data["config"], data["sources"])

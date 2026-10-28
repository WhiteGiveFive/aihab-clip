"""Exercise matrix orchestration without allocating a GPU or fitting models."""

import json
from pathlib import Path

import pytest

from tools import run_attention_fusion_study as study


@pytest.fixture
def workload(tmp_path, monkeypatch):
    config = {"study_id": "fixture", "paths": {"artifact_root": "study"},
              "training": {"epochs": 50, "checkpoint_epoch": 50}}
    fits = [{"fit_id": f"fit{index}", "candidate_id": "image_only", "geo_preprocessing": "not_applicable",
             "learning_rate": 0.001, "seed": index + 1,
             "output_dir": f"study/development/frozen_cached/fit{index}"} for index in range(3)]
    matrix = {"development_fits": fits}
    sources = {"multimodal/attention_fusion_training.py": "a", "multimodal/attention_fusion.py": "b",
               "multimodal/models.py": "c", "tools/run_attention_fusion_study.py": "d"}
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
        spec = {"candidate_id": fit["candidate_id"], "geo_mode": fit["geo_preprocessing"],
                "learning_rate": fit["learning_rate"], "seed": fit["seed"], "verification": False,
                "fit_role": "train", "epochs": 50, "config_sha256": study.content_hash(config),
                "source_hashes": {Path(name).name: value for name, value in sources.items()
                                  if name.startswith("multimodal/")}}
        for name in study.REQUIRED_BUNDLE_FILES:
            (directory / name).write_text("{}")
        (directory / "run_spec.json").write_text(json.dumps(spec))
        (directory / "run.json").write_text(json.dumps({"status": "complete", "completed_epoch": 50}))
        manifest = {"status": "complete", "completed_epoch": 50,
                    "run_identity_sha256": study.content_hash(spec),
                    "files": {name: study.file_hash(directory / name) for name in study.REQUIRED_BUNDLE_FILES}}
        manifest["manifest_payload_sha256"] = study.content_hash(manifest)
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return {"status": "complete"}

    monkeypatch.setattr(study, "execute_fit", complete)
    return {"repo": tmp_path, "config": config, "matrix": matrix, "calls": calls,
            "sources": sources, "complete": complete, "input": input_path}


def test_all_declared_fits_run_once_and_completed_bundles_are_immutable(workload):
    data = workload
    result = study.run_study(repo=data["repo"])
    assert result["counts"] == {"complete": 3}
    assert data["calls"] == [("fit0", False), ("fit1", False), ("fit2", False)]
    bundle = data["repo"] / data["matrix"]["development_fits"][0]["output_dir"]
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in bundle.iterdir()}
    study.run_study(repo=data["repo"])
    assert len(data["calls"]) == 3
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in bundle.iterdir()} == before


def test_failure_stops_matrix_and_next_invocation_resumes_partial(workload, monkeypatch):
    data = workload

    def fail(config, fit, **kwargs):
        if fit["fit_id"] == "fit1":
            directory = data["repo"] / fit["output_dir"]
            directory.mkdir(parents=True)
            (directory / "run.json").write_text(json.dumps({"status": "interrupted_or_failed", "completed_epoch": 12}))
            raise RuntimeError("simulated interruption")
        return data["complete"](config, fit, **kwargs)

    monkeypatch.setattr(study, "execute_fit", fail)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        study.run_study(repo=data["repo"])
    ledger = study.read_json(data["repo"] / "study/orchestration/m3/status.json")
    assert ledger["status"] == "failed_or_interrupted"
    assert ledger["active_fit"] == "fit1"
    assert not (data["repo"] / data["matrix"]["development_fits"][2]["output_dir"]).exists()
    monkeypatch.setattr(study, "execute_fit", data["complete"])
    study.run_study(repo=data["repo"])
    assert data["calls"] == [("fit0", False), ("fit1", True), ("fit2", False)]


def test_corrupted_completed_payload_is_not_reused(workload):
    data = workload
    study.run_study(repo=data["repo"])
    path = data["repo"] / data["matrix"]["development_fits"][0]["output_dir"] / "final.pt"
    path.write_text("corrupted")
    with pytest.raises(study.StudyError, match="payload changed"):
        study.run_study(repo=data["repo"])
    assert len(data["calls"]) == 3
    status = study.matrix_status(data["config"], data["matrix"], data["repo"])
    assert status["counts"] == {"invalid_artifacts": 1, "complete": 2}


@pytest.mark.parametrize("change", ["source", "input"])
def test_midrun_mutation_stops_before_next_fit(workload, monkeypatch, change):
    data = workload

    def mutate(config, fit, **kwargs):
        result = data["complete"](config, fit, **kwargs)
        if change == "source":
            data["sources"]["multimodal/attention_fusion.py"] = "changed"
        else:
            data["input"].write_text("changed during fitting")
        return result

    monkeypatch.setattr(study, "execute_fit", mutate)
    with pytest.raises(study.StudyError, match="changed while"):
        study.run_study(repo=data["repo"])
    assert data["calls"] == [("fit0", False)]


def test_changed_source_cannot_resume_old_orchestration(workload):
    data = workload
    study.run_study(repo=data["repo"])
    data["sources"]["multimodal/models.py"] = "new source"
    with pytest.raises(study.StudyError, match="Orchestration identity changed"):
        study.run_study(repo=data["repo"])


def test_writer_lock_excludes_concurrent_orchestrators(tmp_path):
    with study.writer_lock(tmp_path):
        with pytest.raises(study.StudyError, match="Another study orchestrator"):
            with study.writer_lock(tmp_path):
                pytest.fail("Second writer entered")


def test_status_reads_actual_epoch_of_an_active_fit(workload):
    data = workload
    directory = data["repo"] / data["matrix"]["development_fits"][0]["output_dir"]
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({"status": "running", "completed_epoch": 27}))
    result = study.matrix_status(data["config"], data["matrix"], data["repo"])
    assert result["fits"][0]["completed_epoch"] == 27
    assert result["counts"] == {"running": 1, "planned": 2}


def test_preflight_configures_workspace_before_any_cuda_probe(tmp_path, monkeypatch):
    import os
    import torch

    config = {"training": {"cublas_workspace_config": ":4096:8"}}
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def probe():
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        raise RuntimeError("probe reached with correct workspace")

    monkeypatch.setattr(torch.cuda, "is_available", probe)
    with pytest.raises(RuntimeError, match="correct workspace"):
        study.preflight(config, {}, {}, tmp_path, "cuda", 1)


def test_preflight_rejects_incompatible_preinitialized_cuda(tmp_path, monkeypatch):
    import torch

    config = {"training": {"cublas_workspace_config": ":4096:8"}}
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA probe occurred before workspace guard"))
    with pytest.raises(study.StudyError, match="initialized before"):
        study.preflight(config, {}, {}, tmp_path, "cuda", 1)

"""Synthetic-only checks for the additive replay CLI and process boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import multimodal
from tools import run_multimodal_geo_helpfulness_router_mlp as original_cli
from tools import run_multimodal_geo_helpfulness_router_mlp_replay as runner


@pytest.fixture(autouse=True)
def restore_environment(monkeypatch):
    for name in (
        *original_cli.THREAD_ENVIRONMENT,
        "PYTHONMALLOC",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(name, os.environ.get(name, ""))


def _module(monkeypatch, name):
    module = ModuleType("multimodal." + name)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(multimodal, name, module, raising=False)
    return module


def _limited():
    assert all(os.environ[name] == "1" for name in original_cli.THREAD_ENVIRONMENT)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["PYTHONMALLOC"] == "malloc"


@pytest.mark.parametrize(
    "command,expected_phases,statuses,expected_status",
    [
        ("build", ["fit", "score"], ["built", "built"], "built"),
        ("build", ["fit", "score"], ["reused_valid", "reused_valid"], "reused_valid"),
        ("build", ["fit", "score"], ["reused_valid", "built"], "built"),
        (
            "validate",
            ["validate-predictions", "validate-scores"],
            ["validated", "validated"],
            "validated",
        ),
    ],
)
def test_each_phase_runs_in_its_own_cpu_limited_process_with_all_paths(
    monkeypatch, tmp_path, capsys, command, expected_phases, statuses, expected_status
):
    observed = []
    overrides = {name: tmp_path / name for name in runner.PATH_ARGUMENTS}
    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MLP_CLI_TEST_MARKER", "preserved")

    def forbidden(*args, **kwargs):
        raise AssertionError("parent must not run either phase in its own process")

    monkeypatch.setattr(runner, "_fit", forbidden)

    def run(arguments, **kwargs):
        phase = arguments[2]
        assert arguments[0] == sys.executable
        assert (
            Path(arguments[1]).name
            == "run_multimodal_geo_helpfulness_router_mlp_replay.py"
        )
        assert kwargs["cwd"] == runner.PROJECT_ROOT
        assert kwargs["stdout"] == runner.subprocess.PIPE
        assert kwargs["stderr"] is None
        assert kwargs["text"] is True and kwargs["check"] is False
        assert "capture_output" not in kwargs
        for name in original_cli.THREAD_ENVIRONMENT:
            assert kwargs["env"][name] == "1"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONMALLOC"] == "malloc"
        assert kwargs["env"]["MLP_CLI_TEST_MARKER"] == "preserved"
        for name, path in overrides.items():
            option = "--" + name.replace("_", "-")
            assert Path(arguments[arguments.index(option) + 1]) == path.resolve()
        observed.append(phase)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"status": statuses[len(observed) - 1], "candidate_seed_count": 48}
            ),
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    argv = [command]
    for name, path in overrides.items():
        argv.extend(["--" + name.replace("_", "-"), str(path)])
    assert runner.main(argv) == 0
    assert observed == expected_phases
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == expected_status
    assert result["source_dir"] == str(overrides["source_dir"])
    assert result["output_dir"] == str(overrides["output_dir"])
    assert all(
        item["candidate_seed_count"] == 48 and item["phase_elapsed_seconds"] >= 0
        for item in result["phases"]
    )
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert [item["event"] for item in events] == [
        "phase_started",
        "phase_completed",
        "phase_started",
        "phase_completed",
    ]
    assert [item["phase"] for item in events[::2]] == expected_phases


@pytest.mark.parametrize(
    "returncode,stdout",
    [
        (7, ""),
        (0, "not-json"),
        (0, "[]"),
        (0, '{"status":"failed","error":"sealed source changed"}'),
        (0, '{"status":"built","metric":NaN}'),
        (0, "{}"),
        (0, '{"status":"unknown"}'),
    ],
)
def test_invalid_first_phase_stops_before_scoring(
    monkeypatch, capsys, returncode, stdout
):
    observed = []

    def run(arguments, **kwargs):
        observed.append(arguments[2])
        return SimpleNamespace(returncode=returncode, stdout=stdout)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["build"]) == 1
    assert observed == ["fit"]
    captured = capsys.readouterr()
    assert not captured.out
    failure = json.loads(captured.err.splitlines()[-1])
    assert failure["status"] == "failed" and failure["command"] == "build"
    assert isinstance(failure["error"], str) and failure["error"]
    assert failure["elapsed_seconds"] >= 0


def test_validation_prediction_failure_never_decodes_score_phase(monkeypatch, capsys):
    observed = []

    def run(arguments, **kwargs):
        observed.append(arguments[2])
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["validate"]) == 1
    assert observed == ["validate-predictions"]
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["status"] == "failed"


@pytest.mark.parametrize(
    "command,function,argument_count",
    [
        ("score", "build_scores", 2),
        ("validate-predictions", "validate_predictions", 1),
        ("validate-scores", "validate_scores", 2),
    ],
)
def test_apply_only_dispatch_keeps_source_and_output_separate(
    monkeypatch, tmp_path, capsys, command, function, argument_count
):
    replay = _module(monkeypatch, "geo_helpfulness_router_mlp_replay")
    source, output = tmp_path / "source", tmp_path / "correction"
    calls = []

    def call(*args):
        _limited()
        calls.append(args)
        return {"status": "built" if command == "score" else "validated"}

    setattr(replay, function, call)
    assert (
        runner.main([command, "--source-dir", str(source), "--output-dir", str(output)])
        == 0
    )
    assert calls == [(source,) if argument_count == 1 else (source, output)]
    result = json.loads(capsys.readouterr().out)
    assert result["source_dir"] == str(source) and result["output_dir"] == str(output)
    assert result["command"] == command


def test_default_roots_target_existing_source_and_new_correction_output(
    monkeypatch, capsys
):
    replay = _module(monkeypatch, "geo_helpfulness_router_mlp_replay")
    calls = []
    replay.build_scores = lambda *args: (calls.append(args) or {"status": "built"})
    assert runner.main(["score"]) == 0
    assert calls == [
        (
            runner.ARTIFACT_PARENT / "router_mlp_v1",
            runner.ARTIFACT_PARENT / "router_mlp_v1_replay_v2",
        )
    ]
    result = json.loads(capsys.readouterr().out)
    assert result["source_dir"] != result["output_dir"]


def test_relative_overrides_resolve_before_dispatch(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    replay = _module(monkeypatch, "geo_helpfulness_router_mlp_replay")
    calls = []
    replay.validate_scores = lambda *args: (
        calls.append(args) or {"status": "validated"}
    )
    assert (
        runner.main(
            ["validate-scores", "--source-dir", "source", "--output-dir", "correction"]
        )
        == 0
    )
    assert calls == [(tmp_path / "source", tmp_path / "correction")]
    assert json.loads(capsys.readouterr().out)["status"] == "validated"


def _existing_source(monkeypatch, tmp_path):
    original = _module(monkeypatch, "geo_helpfulness_router_mlp")
    replay = _module(monkeypatch, "geo_helpfulness_router_mlp_replay")
    source = tmp_path / "source"
    (source / "predictions").mkdir(parents=True)
    (source / "predictions/manifest.json").write_text("synthetic existence marker")
    config = {"epochs": 300, "dtype": "float64", "synthetic": True}
    declaration = {
        "protocol_root": str(runner.ARTIFACT_PARENT / "protocol_v1"),
        "capacity_root": str(runner.ARTIFACT_PARENT / "router_capacity_v1"),
        "configuration": config,
    }
    calls = []

    def load(path):
        assert path == source
        calls.append("load_declaration")
        return declaration

    def read(path):
        assert path == runner.PROJECT_ROOT / "configs/multimodal_geo_router_mlp_v1.json"
        calls.append("read_config")
        return config

    def validate(path):
        _limited()
        assert path == source
        calls.append("validate_predictions")
        return {
            "status": "validated",
            "candidate_seed_count": 48,
            "manifest_sha256": "a" * 64,
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("existing sealed source must not call the original fitter")

    original.load_declaration, original.read_json = load, read
    original.build_predictions = forbidden
    replay.validate_predictions = validate
    return source, declaration, calls, original, replay


def test_existing_source_validates_without_refitting(monkeypatch, tmp_path, capsys):
    source, _, calls, _, _ = _existing_source(monkeypatch, tmp_path)
    assert runner.main(["fit", "--source-dir", str(source)]) == 0
    assert calls == ["load_declaration", "read_config", "validate_predictions"]
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "reused_valid" and result["new_fits"] == 0
    assert result["candidate_seed_count"] == 48


@pytest.mark.parametrize("field", ["protocol_root", "capacity_root", "configuration"])
def test_existing_source_path_or_configuration_mismatch_fails_before_replay(
    monkeypatch, tmp_path, capsys, field
):
    source, declaration, calls, _, _ = _existing_source(monkeypatch, tmp_path)
    declaration[field] = {"epochs": 200} if field == "configuration" else "wrong-path"
    assert runner.main(["fit", "--source-dir", str(source)]) == 1
    assert "validate_predictions" not in calls
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "failed"
    assert "paths/configuration differ" in failure["error"]


def test_existing_source_invalid_predictions_return_json_failure(
    monkeypatch, tmp_path, capsys
):
    source, _, _, _, replay = _existing_source(monkeypatch, tmp_path)

    def fail(*args):
        raise ValueError("prediction model hash changed")

    replay.validate_predictions = fail
    assert runner.main(["fit", "--source-dir", str(source)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["error"] == "prediction model hash changed"


def test_fresh_source_delegates_original_fitting_unchanged(
    monkeypatch, tmp_path, capsys
):
    original = _module(monkeypatch, "geo_helpfulness_router_mlp")
    source, output = tmp_path / "fresh-source", tmp_path / "correction"
    protocol, capacity, config = (
        tmp_path / name for name in ("protocol", "capacity", "config.json")
    )
    calls = []

    def build(*args):
        _limited()
        calls.append(args)
        return {"status": "built", "candidate_seed_count": 48, "new_fits": 48}

    original.build_predictions = build
    assert (
        runner.main(
            [
                "fit",
                "--protocol-dir",
                str(protocol),
                "--capacity-root",
                str(capacity),
                "--source-dir",
                str(source),
                "--output-dir",
                str(output),
                "--config",
                str(config),
            ]
        )
        == 0
    )
    assert calls == [(protocol, source, config, capacity)]
    assert output not in calls[0]
    assert json.loads(capsys.readouterr().out)["new_fits"] == 48


def test_apply_only_exception_produces_nonzero_json(monkeypatch, tmp_path, capsys):
    replay = _module(monkeypatch, "geo_helpfulness_router_mlp_replay")

    def fail(*args):
        raise ValueError("sealed correction declaration differs")

    replay.build_scores = fail
    assert (
        runner.main(
            [
                "score",
                "--source-dir",
                str(tmp_path / "source"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["error"] == "sealed correction declaration differs"


@pytest.mark.parametrize(
    "option",
    [
        "--force",
        "--seed",
        "--epochs",
        "--hidden-width",
        "--learning-rate",
        "--threshold",
        "--temperature",
        "--test-dir",
    ],
)
def test_replay_has_no_scientific_or_test_overrides(option):
    with pytest.raises(SystemExit) as error:
        runner.main(["build", option, "1"])
    assert error.value.code == 2

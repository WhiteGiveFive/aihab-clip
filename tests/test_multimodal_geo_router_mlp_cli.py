"""The MLP CLI isolates phases and forwards immutable input roots unchanged."""

import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools import run_multimodal_geo_helpfulness_router_mlp as runner


@pytest.fixture(autouse=True)
def restore_thread_environment(monkeypatch):
    for name in (*runner.THREAD_ENVIRONMENT, "PYTHONMALLOC", "CUDA_VISIBLE_DEVICES"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.setenv(name, "")


@pytest.mark.parametrize(
    "command,phases",
    [
        ("build", ["fit", "score"]),
        ("validate", ["validate-predictions", "validate-scores"]),
    ],
)
def test_process_order_paths_limits_and_live_stderr(
    monkeypatch, tmp_path, capsys, command, phases
):
    observed = []
    capacity = tmp_path / "capacity-parent"

    def run(arguments, **kwargs):
        observed.append(arguments[2])
        assert arguments[0] == sys.executable
        assert Path(arguments[1]).name == "run_multimodal_geo_helpfulness_router_mlp.py"
        assert kwargs["cwd"] == runner.PROJECT_ROOT
        for name in runner.THREAD_ENVIRONMENT:
            assert kwargs["env"][name] == "1"
        assert kwargs["env"]["PYTHONMALLOC"] == "malloc"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["stdout"] == runner.subprocess.PIPE
        assert kwargs["stderr"] is None
        assert "capture_output" not in kwargs
        for name in runner.PATH_ARGUMENTS:
            option = "--" + name.replace("_", "-")
            assert Path(arguments[arguments.index(option) + 1]).is_absolute()
        assert Path(arguments[arguments.index("--capacity-root") + 1]) == capacity
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "reused_valid", "candidate_seed_count": 7}),
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(
        [command, "--output-dir", str(tmp_path), "--capacity-root", str(capacity)]
    ) == 0
    assert observed == phases
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == ("reused_valid" if command == "build" else "validated")
    assert result["elapsed_seconds"] >= 0
    assert all(phase["phase_elapsed_seconds"] >= 0 for phase in result["phases"])
    assert all(phase["candidate_seed_count"] == 7 for phase in result["phases"])
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert [event["event"] for event in events] == [
        "phase_started", "phase_completed", "phase_started", "phase_completed"
    ]


@pytest.mark.parametrize(
    "returncode,stdout",
    [
        (1, ""),
        (0, "not-json"),
        (0, "[]"),
        (0, '{"status":"failed","error":"bad-state"}'),
        (0, '{"status":"built","loss":NaN}'),
    ],
)
def test_failed_fit_or_invalid_summary_never_scores(
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
    assert failure["status"] == "failed"
    assert "fit" in failure["error"]


@pytest.mark.parametrize(
    "command,function",
    [
        ("fit", "build_predictions"),
        ("score", "build_scores"),
        ("validate-predictions", "validate_predictions"),
        ("validate-scores", "validate_scores"),
    ],
)
def test_single_phase_dispatch_is_lazy_and_cpu_limited(
    monkeypatch, tmp_path, capsys, command, function
):
    observed = []
    module = ModuleType("multimodal.geo_helpfulness_router_mlp")
    for name in runner.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    def call(*arguments):
        observed.append(arguments)
        assert all(os.environ[name] == "1" for name in runner.THREAD_ENVIRONMENT)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        return {"status": "validated" if command.startswith("validate") else "built"}

    setattr(module, function, call)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert runner.main([command, "--output-dir", str(tmp_path)]) == 0
    assert len(observed) == 1
    assert len(observed[0]) == (4 if command == "fit" else 1)
    assert all(isinstance(path, Path) and path.is_absolute() for path in observed[0])
    if command == "fit":
        protocol, output, config, capacity = observed[0]
        assert protocol == runner.ARTIFACT_PARENT / "protocol_v1"
        assert output == tmp_path
        assert config == runner.PROJECT_ROOT / "configs/multimodal_geo_router_mlp_v1.json"
        assert capacity == runner.ARTIFACT_PARENT / "router_capacity_v1"
    result = json.loads(capsys.readouterr().out)
    assert result["command"] == command


def test_default_output_is_new_mlp_root(monkeypatch, capsys):
    module = ModuleType("multimodal.geo_helpfulness_router_mlp")

    def call(output):
        assert output == runner.ARTIFACT_PARENT / "router_mlp_v1"
        return {"status": "validated"}

    module.validate_predictions = call
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert runner.main(["validate-predictions"]) == 0
    assert json.loads(capsys.readouterr().out)["output_dir"].endswith("/router_mlp_v1")


def test_single_phase_failure_is_nonzero_json(monkeypatch, capsys):
    module = ModuleType("multimodal.geo_helpfulness_router_mlp")

    def fail(*arguments):
        raise ValueError("sealed capacity parent differs")

    module.build_scores = fail
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert runner.main(["score"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["error"] == "sealed capacity parent differs"


@pytest.mark.parametrize(
    "option",
    [
        "--force", "--seed", "--threshold", "--temperature", "--test-dir",
        "--family", "--variant", "--epochs", "--hidden-width", "--learning-rate",
    ],
)
def test_no_unrecorded_scientific_or_test_overrides(option):
    with pytest.raises(SystemExit) as error:
        runner.main(["build", option, "1"])
    assert error.value.code == 2

"""The capacity runner must isolate fitting, scoring, and validation phases."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools import run_multimodal_geo_helpfulness_router_capacity as runner


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

    def run(arguments, **kwargs):
        observed.append(arguments[2])
        assert kwargs["cwd"] == runner.PROJECT_ROOT
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "1"
        assert kwargs["env"]["PYTHONMALLOC"] == "malloc"
        assert kwargs["stdout"] == runner.subprocess.PIPE
        assert kwargs["stderr"] is None
        assert "capture_output" not in kwargs
        for option in ("--protocol-dir", "--output-dir", "--config"):
            assert Path(arguments[arguments.index(option) + 1]).is_absolute()
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"status": "reused_valid"})
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main([command, "--output-dir", str(tmp_path)]) == 0
    assert observed == phases
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == ("reused_valid" if command == "build" else "validated")
    assert result["elapsed_seconds"] >= 0
    assert all(phase["phase_elapsed_seconds"] >= 0 for phase in result["phases"])
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert [event["event"] for event in events] == [
        "phase_started",
        "phase_completed",
    ] * 2


@pytest.mark.parametrize(
    "returncode,stdout",
    [
        (1, ""),
        (0, "not-json"),
        (0, "[]"),
        (0, '{"status":"failed","error":"bad-state"}'),
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
def test_single_phase_dispatch_is_lazy(
    monkeypatch, tmp_path, capsys, command, function
):
    observed = []
    module = ModuleType("multimodal.geo_helpfulness_router_capacity")

    def call(*arguments):
        observed.append(arguments)
        return {"status": "validated" if command.startswith("validate") else "built"}

    setattr(module, function, call)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert runner.main([command, "--output-dir", str(tmp_path)]) == 0
    assert len(observed) == 1
    assert len(observed[0]) == (3 if command == "fit" else 1)
    assert all(isinstance(path, Path) and path.is_absolute() for path in observed[0])
    result = json.loads(capsys.readouterr().out)
    assert result["command"] == command


def test_single_phase_failure_is_nonzero_json(monkeypatch, capsys):
    module = ModuleType("multimodal.geo_helpfulness_router_capacity")

    def fail(*arguments):
        raise ValueError("sealed parent differs")

    module.build_scores = fail
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert runner.main(["score"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["error"] == "sealed parent differs"


@pytest.mark.parametrize(
    "option",
    [
        "--force",
        "--seed",
        "--threshold",
        "--temperature",
        "--test-dir",
        "--family",
        "--variant",
    ],
)
def test_no_unrecorded_scientific_or_test_overrides(option):
    with pytest.raises(SystemExit) as error:
        runner.main(["build", option, "1"])
    assert error.value.code == 2

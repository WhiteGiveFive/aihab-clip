"""Process boundaries and immutable arguments for the legacy capacity CLI."""

import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools import run_multimodal_geo_helpfulness_relaxed_rf as runner


@pytest.fixture(autouse=True)
def restore_environment(monkeypatch):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "PYTHONMALLOC",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(name, os.environ.get(name, ""))


@pytest.mark.parametrize(
    "command,phases",
    [
        ("build", ["select", "predict", "score"]),
        ("validate", ["validate-selection", "validate-predictions", "validate-scores"]),
    ],
)
def test_isolated_phases_paths_cpu_limits_and_live_stderr(
    monkeypatch, capsys, tmp_path, command, phases
):
    observed = []

    def run(arguments, **kwargs):
        phase = arguments[2]
        observed.append(phase)
        assert arguments[0] == sys.executable
        assert Path(arguments[1]).name == "run_multimodal_geo_helpfulness_relaxed_rf.py"
        assert kwargs["cwd"] == runner.PROJECT_ROOT
        assert kwargs["stderr"] is None and kwargs["stdout"] == runner.subprocess.PIPE
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONMALLOC"] == "malloc"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            assert kwargs["env"][name] == "1"
        for name in runner.PATHS:
            flag = "--" + name.replace("_", "-")
            assert Path(arguments[arguments.index(flag) + 1]).is_absolute()
        assert arguments[arguments.index("--output-dir") + 1] == str(tmp_path)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": (
                        "validated" if phase.startswith("validate") else "reused_valid"
                    )
                }
            ),
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main([command, "--output-dir", str(tmp_path)]) == 0
    assert observed == phases
    capture = capsys.readouterr()
    value = json.loads(capture.out)
    assert value["status"] == ("validated" if command == "validate" else "reused_valid")
    assert value["elapsed_seconds"] >= 0
    events = [json.loads(line) for line in capture.err.splitlines()]
    assert [e["event"] for e in events] == ["phase_started", "phase_completed"] * 3


@pytest.mark.parametrize(
    "returncode,stdout",
    [
        (1, ""),
        (0, "not-json"),
        (0, "[]"),
        (0, "{}"),
        (0, '{"status":"failed"}'),
        (0, '{"status":"validated"}'),
        (0, '{"status":"built","value":NaN}'),
    ],
)
def test_selection_failure_never_starts_prediction_or_scoring(
    monkeypatch, capsys, returncode, stdout
):
    observed = []

    def run(arguments, **kwargs):
        observed.append(arguments[2])
        return SimpleNamespace(returncode=returncode, stdout=stdout)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["build"]) == 1
    assert observed == ["select"]
    capture = capsys.readouterr()
    assert not capture.out
    assert json.loads(capture.err.splitlines()[-1])["status"] == "failed"


def test_prediction_failure_after_selection_never_starts_scoring(monkeypatch, capsys):
    observed = []

    def run(arguments, **kwargs):
        phase = arguments[2]
        observed.append(phase)
        return SimpleNamespace(
            returncode=1 if phase == "predict" else 0,
            stdout=json.dumps({"status": "built"}),
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["build"]) == 1
    assert observed == ["select", "predict"]
    capture = capsys.readouterr()
    assert not capture.out
    assert "predict failed" in json.loads(capture.err.splitlines()[-1])["error"]


@pytest.mark.parametrize(
    "command,function",
    [
        ("select", "build_selection"),
        ("validate-selection", "validate_selection"),
        ("predict", "build_predictions"),
        ("score", "build_scores"),
        ("validate-predictions", "validate_predictions"),
        ("validate-scores", "validate_scores"),
    ],
)
def test_single_phase_dispatch_and_defaults(monkeypatch, capsys, command, function):
    import multimodal

    module = ModuleType("multimodal.geo_helpfulness_relaxed_rf")
    observed = []

    def call(*args):
        observed.append(args)
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        return {"status": "validated" if command.startswith("validate") else "built"}

    def forbidden(*args):
        raise AssertionError("another workflow phase was called")

    for name in (
        "build_selection",
        "validate_selection",
        "build_predictions",
        "build_scores",
        "validate_predictions",
        "validate_scores",
    ):
        setattr(module, name, call if name == function else forbidden)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(multimodal, "geo_helpfulness_relaxed_rf", module, raising=False)
    assert runner.main([command]) == 0
    expected = [runner.PATHS["output_dir"]]
    if command == "select":
        expected += [
            runner.PATHS[k]
            for k in (
                "legacy_root",
                "capacity_root",
                "lr_history_root",
                "config",
            )
        ]
    assert observed == [tuple(expected)]
    assert all(path.is_absolute() for path in observed[0])
    assert json.loads(capsys.readouterr().out)["command"] == command


def test_single_phase_exception_produces_only_stderr_json(monkeypatch, capsys):
    import multimodal

    module = ModuleType("multimodal.geo_helpfulness_relaxed_rf")

    def fail(*args):
        raise ValueError("source hash differs")

    for name in (
        "build_selection",
        "validate_selection",
        "build_predictions",
        "build_scores",
        "validate_predictions",
        "validate_scores",
    ):
        setattr(module, name, fail)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(multimodal, "geo_helpfulness_relaxed_rf", module, raising=False)
    assert runner.main(["score"]) == 1
    capture = capsys.readouterr()
    assert not capture.out
    assert json.loads(capture.err)["error"] == "source hash differs"


@pytest.mark.parametrize(
    "flag",
    [
        "--force",
        "--seed",
        "--family",
        "--learner",
        "--variant",
        "--threshold",
        "--temperature",
        "--test-dir",
        "--epochs",
        "--learning-rate",
    ],
)
def test_no_scientific_or_test_access_overrides(flag):
    with pytest.raises(SystemExit) as error:
        runner.main(["build", flag, "1"])
    assert error.value.code == 2

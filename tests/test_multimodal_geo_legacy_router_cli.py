"""The orchestration command must preserve inference/scoring process isolation."""

import json
from types import SimpleNamespace

import pytest

from tools import run_multimodal_geo_helpfulness_legacy_router as runner


@pytest.mark.parametrize(
    "command,phases",
    [
        ("build", ["predict", "score"]),
        ("validate", ["validate-predictions", "validate-scores"]),
    ],
)
def test_process_order_and_fixed_scope(monkeypatch, tmp_path, capsys, command, phases):
    observed = []

    def run(arguments, **kwargs):
        observed.append(arguments[2])
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert "--output-dir" in arguments
        assert "--checkpoint-family" not in arguments
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"status": "reused_valid"}), stderr=""
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main([command, "--output-dir", str(tmp_path)]) == 0
    assert observed == phases
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == ("reused_valid" if command == "build" else "validated")


def test_failed_inference_never_scores(monkeypatch, capsys):
    observed = []

    def run(arguments, **kwargs):
        observed.append(arguments[2])
        return SimpleNamespace(returncode=1, stdout="", stderr="invalid parent")

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["build"]) == 1
    assert observed == ["predict"]
    assert "invalid parent" in json.loads(capsys.readouterr().err)["error"]


@pytest.mark.parametrize(
    "option",
    ["--force", "--C", "--threshold", "--temperature", "--seed", "--checkpoint-family"],
)
def test_no_scientific_or_overwrite_overrides(option):
    with pytest.raises(SystemExit) as error:
        runner.main(["build", option, "1"])
    assert error.value.code == 2

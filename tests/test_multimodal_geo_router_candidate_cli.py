"""CLI contract for reproducible M4 primary and ablation router fitting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_multimodal_geo_helpfulness_m4_candidates.py"
)
COMMANDS = ("build", "validate", "reproduce")


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("m4_candidate_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", COMMANDS)
def test_cli_exposes_only_path_overrides(runner, command):
    args = runner._parser().parse_args([command])
    assert runner.DEFAULT_STUDY_CONFIG_PATH == (
        RUNNER_PATH.parents[1] / "configs/multimodal_geo_router_candidates_v2.json"
    )
    assert vars(args) == {
        "command": command,
        "config": runner.DEFAULT_CONFIG_PATH,
        "protocol_dir": None,
        "artifact_root": None,
        "study_config": runner.DEFAULT_STUDY_CONFIG_PATH,
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["build", "--force-overwrite"],
        ["build", "--C", "0.1"],
        ["build", "--features", "image_confidence"],
        ["build", "--max-iter", "10000"],
        ["build", "--validation-labels", "validation.parquet"],
        ["build", "--test-data", "test.parquet"],
        ["train"],
        [],
    ],
)
def test_cli_rejects_scientific_or_expansive_overrides(runner, argv):
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("command", COMMANDS)
def test_cli_forwards_paths_and_reports_json(runner, monkeypatch, capsys, command):
    captured = []

    def operation(**kwargs):
        captured.append(kwargs)
        return {"status": "reused_valid", "artifact_root": Path("artifacts")}

    def unexpected(**kwargs):
        pytest.fail("Unexpected operation invoked")

    for candidate_command in COMMANDS:
        monkeypatch.setattr(
            runner,
            f"{candidate_command}_router_candidate_bundle",
            operation if candidate_command == command else unexpected,
        )
    assert runner.main(
        [
            command,
            "--config", "config.yaml",
            "--protocol-dir", "protocol",
            "--artifact-root", "artifacts",
            "--study-config", "study.json",
        ]
    ) == 0
    assert captured == [{
        "config_path": Path("config.yaml"),
        "protocol_dir": Path("protocol"),
        "artifact_root": Path("artifacts"),
        "study_config_path": Path("study.json"),
    }]
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "status": "reused_valid", "artifact_root": "artifacts"
    }
    assert output.err == ""


@pytest.mark.parametrize("command", COMMANDS)
def test_cli_reports_failure_as_json_and_exit_one(runner, monkeypatch, capsys, command):
    def operation(**kwargs):
        raise ValueError("parent hash mismatch")

    monkeypatch.setattr(runner, f"{command}_router_candidate_bundle", operation)
    assert runner.main([command]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "status": "failed", "command": command, "error": "parent hash mismatch"
    }

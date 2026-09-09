"""CLI contract for reproducible M4 router-dataset preparation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_multimodal_geo_helpfulness_m4_dataset.py"
)


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("m4_dataset_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", ["build", "validate"])
def test_cli_exposes_only_path_overrides(runner, command):
    args = runner._parser().parse_args([command])
    assert vars(args) == {
        "command": command,
        "config": runner.DEFAULT_CONFIG_PATH,
        "protocol_dir": None,
        "artifact_root": None,
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["build", "--force-overwrite"],
        ["build", "--temperature", "2"],
        ["build", "--test-data", "test.parquet"],
        ["train"],
    ],
)
def test_cli_rejects_scientific_or_expansive_overrides(runner, argv):
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["build", "validate"])
def test_cli_forwards_paths_and_reports_json(runner, monkeypatch, capsys, command):
    captured = []

    def operation(**kwargs):
        captured.append(kwargs)
        return {"status": "reused_valid", "artifact_root": Path("artifacts")}

    monkeypatch.setattr(runner, f"{command}_router_dataset_bundle", operation)
    other = "validate" if command == "build" else "build"

    def unexpected(**kwargs):
        pytest.fail(f"Unexpected {other} operation")

    monkeypatch.setattr(runner, f"{other}_router_dataset_bundle", unexpected)
    assert (
        runner.main(
            [
                command,
                "--config",
                "config.yaml",
                "--protocol-dir",
                "protocol",
                "--artifact-root",
                "artifacts",
            ]
        )
        == 0
    )
    assert captured == [
        {
            "config_path": Path("config.yaml"),
            "protocol_dir": Path("protocol"),
            "artifact_root": Path("artifacts"),
        }
    ]
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "status": "reused_valid",
        "artifact_root": "artifacts",
    }
    assert output.err == ""


@pytest.mark.parametrize("command", ["build", "validate"])
def test_cli_reports_failure_as_json_and_exit_one(runner, monkeypatch, capsys, command):
    def operation(**kwargs):
        raise ValueError("parent hash mismatch")

    monkeypatch.setattr(runner, f"{command}_router_dataset_bundle", operation)
    assert runner.main([command]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "status": "failed",
        "command": command,
        "error": "parent hash mismatch",
    }

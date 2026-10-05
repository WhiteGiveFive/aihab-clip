"""Check canonical dispatch without running any scientific backend."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from multimodal import geo_helpfulness_compat as compat
from multimodal import geo_helpfulness_geometry as geometry


@pytest.mark.parametrize("command,study_id", geometry.COMMANDS.items())
def test_default_geometry_destination(command, study_id):
    args = geometry.parser().parse_args([command])
    argv = geometry.invocation(args)
    output = Path(argv[argv.index("--output-dir") + 1])
    assert output.name == study_id
    assert output.parent == geometry.ROOT / compat.ARTIFACTS
    assert Path(argv[1]).is_file()
    if command == "overview":
        assert argv[-2:] == ["--development-split", "all"]


def test_extract_translates_output_option():
    args = geometry.parser().parse_args(["extract-adapted", "--device", "cpu", "--preflight-only"])
    argv = geometry.invocation(args)
    assert "--output-dir" not in argv
    assert Path(argv[argv.index("--output") + 1]).parts[-2:] == ("adapted_geometry_v1", "embeddings")
    assert argv[-3:] == ["--device", "cpu", "--preflight-only"]


def test_geometry_help_and_dry_run_never_launch(capsys):
    with patch.object(geometry.subprocess, "run") as child:
        with pytest.raises(SystemExit) as error:
            geometry.main(["adapted", "--help"])
        assert error.value.code == 0
        capsys.readouterr()
        assert geometry.main(["adapted", "--dry-run"]) == 0
        assert json.loads(capsys.readouterr().out)["executed"] is False
        child.assert_not_called()


@pytest.mark.parametrize("arguments", [
    ["k5", "--out", "somewhere"],
    ["k5", "--output-dir=x", "--output-dir", "y"],
    ["k5", "--unknown"],
])
def test_ambiguous_or_unknown_options_fail_before_dispatch(arguments):
    with patch.object(geometry.subprocess, "run") as child:
        with pytest.raises(SystemExit) as error:
            geometry.main(arguments)
        assert error.value.code == 2
        child.assert_not_called()


@pytest.mark.parametrize("destination", ["/tmp/outside_geometry_v2", "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/unversioned"])
def test_invalid_output_location_never_launches(destination):
    with patch.object(geometry.subprocess, "run") as child:
        assert geometry.main(["k5", f"--output-dir={destination}"]) == 2
        child.assert_not_called()


def test_output_symlink_cannot_escape_artifact_root(tmp_path):
    base = tmp_path / "artifacts"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "escape_v2").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Outputs must be under"):
        geometry.artifact_destination("artifacts/escape_v2", {"artifact_root": "artifacts"}, tmp_path)


def test_dispatch_preserves_arguments_and_return_code_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = f"{compat.ARTIFACTS}/geometry_space_v2/with spaces"
    with patch.object(geometry.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as child:
        assert geometry.main(["k5", f"--output-dir={relative}"]) == 7
    argv = child.call_args.args[0]
    assert argv[argv.index("--output-dir") + 1] == str(geometry.ROOT / relative)
    assert child.call_args.kwargs["cwd"] == geometry.ROOT


@pytest.mark.parametrize("entrypoint", [compat.confidnet_relaxed_main, compat.joint_tcp_review_main, compat.confidnet_parity_main])
def test_compatibility_help_is_handled_without_entering_legacy_code(entrypoint):
    with patch.object(compat.subprocess, "run") as child:
        with pytest.raises(SystemExit) as error:
            entrypoint(["--help"])
        assert error.value.code == 0
        child.assert_not_called()


def test_parity_diagnostic_requires_explicit_run(capsys):
    with patch.object(compat.subprocess, "run") as child:
        for arguments in ([], ["--unknown"], ["run", "--unknown"]):
            with pytest.raises(SystemExit):
                compat.confidnet_parity_main(arguments)
        assert compat.confidnet_parity_main(["run", "--dry-run"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["executed"] is False
        assert data["argv"][-1].endswith("diagnose_multimodal_geo_confidnet_oof_parity.py")
        child.assert_not_called()


def test_confidnet_alias_preserves_validation_command():
    with patch.object(compat.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
        assert compat.confidnet_relaxed_main(["validate"]) == 0
    argv = child.call_args.args[0]
    assert argv[2:4] == ["validate", "--output-dir"]
    assert argv[4].endswith("/confidnet_oof_relaxed_guardrails_v1")


def test_review_alias_translates_output_option(capsys):
    with patch.object(compat.subprocess, "run") as child:
        assert compat.joint_tcp_review_main([
            "--report", f"{compat.ARTIFACTS}/joint_tcp_router_v1/scores/report.json",
            "--output-dir", f"{compat.ARTIFACTS}/joint_tcp_router_v1_review/figures",
            "--dry-run",
        ]) == 0
        argv = json.loads(capsys.readouterr().out)["argv"]
        assert "--output" in argv and "--output-dir" not in argv
        child.assert_not_called()

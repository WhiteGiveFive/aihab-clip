"""Practical M3J receipts and fresh-start orchestration preserve prior attempts."""

import json

import pytest

from multimodal import attention_fusion_joint_protocol as protocol
from tools import run_attention_fusion_joint_study as cli


class GatePassed(Exception):
    """Stop before declaration or training after exercising the real CLI gate."""


@pytest.fixture
def isolated_gate(tmp_path, monkeypatch):
    root = tmp_path / "study"
    sources = {"fixture.py": "a" * 64}
    provenance = {"roles": "training-and-validation-only", "test_data_loaded": False}
    runtime = {"fixture": "mocked-runtime-with-no-gpu-use"}
    monkeypatch.setattr(cli, "study_root", lambda config: root)
    monkeypatch.setattr(cli, "source_hashes", lambda: sources)
    monkeypatch.setattr(cli, "prepare_inputs", lambda config: {
        "provenance": dict(provenance, role_rows={"train": 3378, "validation": 822})})
    monkeypatch.setattr(cli, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(cli, "_environment", lambda device: runtime)
    monkeypatch.setattr(cli, "read_recovery_amendment", lambda *args: None)

    def reached_declaration(*args):
        raise GatePassed()

    monkeypatch.setattr(cli, "declare", reached_declaration)
    monkeypatch.setattr(cli, "train_fit", lambda *a, **k: pytest.fail("Training must not start"))

    def receipt(config, status="passed"):
        value = dict(status=status, sources=sources, verification_policy=config["verification_policy"],
                     config_sha256=cli.configuration_hash(config), runtime=runtime,
                     input_provenance=cli.prepare_inputs(config)["provenance"])
        latest = root / "verification/latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps(value))
        return latest, value

    return receipt


@pytest.mark.parametrize("status", ["passed_exploratory", "failed", "running"])
def test_run_rejects_unaccepted_receipt(isolated_gate, status):
    config = protocol.load_config()
    isolated_gate(config, status)
    with pytest.raises(ValueError, match="Training-only verification"):
        cli.run(config, "cpu")


def test_matching_receipt_reaches_declaration_without_training(isolated_gate):
    config = protocol.load_config()
    isolated_gate(config)
    with pytest.raises(GatePassed):
        cli.run(config, "cpu")


@pytest.mark.parametrize("field", ["sources", "config_sha256", "runtime", "input_provenance", "verification_policy"])
def test_gate_binds_code_configuration_runtime_inputs_and_check_policy(isolated_gate, field):
    config = protocol.load_config()
    latest, value = isolated_gate(config)
    value[field] = "different" if isinstance(value[field], str) else {"different": True}
    latest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Training-only verification"):
        cli.run(config, "cpu")


def test_old_pass_receipt_without_policy_is_not_reused(isolated_gate):
    config = protocol.load_config()
    latest, value = isolated_gate(config)
    del value["verification_policy"]
    latest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Training-only verification"):
        cli.run(config, "cpu")


def test_failed_verification_invalidates_previous_pass(tmp_path, monkeypatch):
    root = tmp_path / "study"
    attempt = root / "verification/attempt"
    attempt.mkdir(parents=True)
    latest = root / "verification/latest.json"
    previous = dict(status="passed", directory="study/verification/attempt", sources={"fixture": "sha"})
    latest.write_text(json.dumps(previous))
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "study_root", lambda config: root)

    def fail_verification(*args):
        raise ValueError("injected training failure")

    monkeypatch.setattr(cli, "verify", fail_verification)
    with pytest.raises(ValueError, match="injected training failure"):
        cli.main(["--config", str(protocol.DEFAULT_CONFIG), "--device", "cpu", "verify"])
    failed = json.loads(latest.read_text())
    assert failed["status"] == "failed"
    assert failed["error"] == "ValueError: injected training failure"
    assert failed["result_assurance"] == protocol.PROVISIONAL_ASSURANCE
    assert failed["sources"] == previous["sources"]
    assert json.loads((attempt / "failure.json").read_text()) == failed


def test_status_discloses_policy_and_assurance_before_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "study_root", lambda config: tmp_path)
    status = cli.status(protocol.load_config())
    assert status["authorized_fits"] == 16 and status["complete"] == 0
    assert all(fit["status"] == "planned" for fit in status["fits"])
    assert status["verification_policy"] == protocol.VERIFICATION_POLICY
    assert status["result_assurance"] == protocol.PROVISIONAL_ASSURANCE
    assert "numerical_policy" not in status


def test_incomplete_attempt_is_preserved_and_canonical_directory_freed(tmp_path):
    run_id = "seed1/joint_partial/image_only/pretrained"
    current = tmp_path / "runs" / run_id
    current.mkdir(parents=True)
    (current / "history.json").write_text('[{"epoch": 7}]')
    (current / "checkpoint_manifest.json").write_text('{"checkpoint": "untouched"}')
    archived = cli.archive_incomplete_attempt(tmp_path, current, run_id)
    assert archived.is_relative_to(tmp_path / "interrupted_attempts" / run_id)
    assert json.loads((archived / "history.json").read_text()) == [{"epoch": 7}]
    assert json.loads((archived / "checkpoint_manifest.json").read_text()) == {"checkpoint": "untouched"}
    assert not current.exists()
    assert cli.archive_incomplete_attempt(tmp_path, current, run_id) is None


@pytest.fixture(params=[1, 2])
def queue_fixture(tmp_path, monkeypatch, request):
    from multimodal import attention_fusion_joint_reporting as reporting
    config = protocol.load_config(protocol.package_config_path(request.param, "joint_partial"))
    root = tmp_path / "queue"
    sources = {"fixture.py": "a" * 64}
    inputs = {"provenance": {"role_rows": {"train": 3378, "validation": 822}}}
    runtime = {"device": "test"}
    fits = protocol.authorized_fits(config)[:4]
    monkeypatch.setattr(cli, "study_root", lambda config: root)
    monkeypatch.setattr(cli, "source_hashes", lambda: sources)
    monkeypatch.setattr(cli, "prepare_inputs", lambda config: inputs)
    monkeypatch.setattr(cli, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(cli, "_environment", lambda device: runtime)
    monkeypatch.setattr(cli, "declare", lambda *args: None)
    monkeypatch.setattr(cli, "authorized_fits", lambda config: fits)
    monkeypatch.setattr(reporting, "build_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.JointCheckpointStore, "read_manifest", lambda *args: {})
    monkeypatch.setattr(cli, "archive_incomplete_attempt",
                        lambda *args: pytest.fail("Continuation must never archive/retrain"))
    receipt = dict(status="passed", sources=sources, verification_policy=config["verification_policy"],
                   config_sha256=cli.configuration_hash(config), runtime=runtime,
                   input_provenance=inputs["provenance"])
    (root / "verification").mkdir(parents=True)
    cli._write_json(root / "verification/latest.json", receipt)

    def record(fit, status, epoch):
        directory = root / "runs" / fit["run_id"]
        directory.mkdir(parents=True, exist_ok=True)
        cli._write_json(directory / "run.json", dict(
            **fit, status=status, completed_epoch=epoch, expected_epochs=50))
        return directory

    return config, root, fits, record


@pytest.mark.parametrize("seed", [2, 3, 4])
@pytest.mark.parametrize("regime,fit_count", [("joint_partial", 16), ("frozen_online", 16), ("geo_only", 1)])
def test_prepare_records_seed_package_without_training_or_gpu(tmp_path, monkeypatch, seed, regime, fit_count):
    config = protocol.load_config(protocol.package_config_path(seed, regime))
    root = tmp_path / config["paths"]["artifact_root"]
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "study_root", lambda config: root)
    monkeypatch.setattr(cli, "source_hashes", lambda: {"fixture.py": "a" * 64})
    monkeypatch.setattr(cli, "prepare_inputs", lambda config: dict(
        provenance={"role_rows": {"train": 3378, "validation": 822}},
        manifest_path=str(root / "inputs/manifest.json"), encoder_producers={str(seed): "fixture"}))

    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not invoke models, GPU verification, declaration or fitting")

    for name in ("make_model", "verify", "train_fit", "finalize_fit", "declare", "configure_runtime"):
        monkeypatch.setattr(cli, name, forbidden)
    result = cli.prepare(config)
    assert result["status"] == "prepared"
    assert result["seed"] == seed and result["package"] == f"seed{seed}_{regime}"
    assert len(result["authorized_fits"]) == fit_count
    assert {fit["seed"] for fit in result["authorized_fits"]} == {seed}
    assert not result["comparative_training_started"] and not result["validation_data_scored"]
    assert result["gpu_verification_required_before_fitting"]
    assert all(item["status"] == "not_started" for item in result["preceding_packages"])
    assert len(result["preceding_packages"]) == {"joint_partial": 0, "frozen_online": 1, "geo_only": 1}[regime]
    assert all(item["required"] for item in result["preceding_packages"])
    if regime == "geo_only":
        assert result["deferred_preceding_packages"] == [{
            "artifact_root": f"multimodal_artifacts/attention_fusion/cs/gse_100m/{protocol.package_identity(seed, 'frozen_online')}",
            "regime": "frozen_online", "seed": seed, "required": False,
            "status": "deferred", "planned_fits": 16}]
    else:
        assert result["deferred_preceding_packages"] == []
    assert json.loads((root / "preparation/manifest.json").read_text()) == result
    assert not (root / "runs").exists()
    assert not (root / "verification/latest.json").exists()


def test_prepare_preserves_superseded_manifest_and_does_not_read_deferred_package(tmp_path, monkeypatch):
    config = protocol.load_config(protocol.package_config_path(2, "geo_only"))
    root = tmp_path / config["paths"]["artifact_root"]
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "study_root", lambda config: root)
    monkeypatch.setattr(cli, "source_hashes", lambda: {"fixture.py": "a" * 64})
    monkeypatch.setattr(cli, "prepare_inputs", lambda config: dict(
        provenance={"role_rows": {"train": 3378, "validation": 822}},
        manifest_path=str(root / "inputs/manifest.json"), encoder_producers={}))
    old_manifest = root / "preparation/manifest.json"
    old_manifest.parent.mkdir(parents=True)
    old = b'{"status": "prepared", "historical": "both predecessors required"}\n'
    old_manifest.write_bytes(old)
    # A malformed file in the deferred package would fail parsing if opened.
    deferred_root = tmp_path / "multimodal_artifacts/attention_fusion/cs/gse_100m" / protocol.package_identity(2, "frozen_online")
    (deferred_root / "protocol").mkdir(parents=True)
    (deferred_root / "protocol/protocol.json").write_text("must not be read")
    reference_config = protocol.load_config(protocol.package_config_path(2, "joint_partial"))
    reference_root = tmp_path / reference_config["paths"]["artifact_root"]
    (reference_root / "protocol").mkdir(parents=True)
    (reference_root / "orchestration/seed2_joint_partial").mkdir(parents=True)
    cli._write_json(reference_root / "protocol/protocol.json", {"config": reference_config})
    cli._write_json(reference_root / "orchestration/seed2_joint_partial/status.json",
                    {"queue_status": "complete", "complete": 16})
    result = cli.prepare(config)
    assert len(result["preceding_packages"]) == 1
    assert result["preceding_packages"][0]["status"] == "complete"
    assert result["preceding_packages"][0]["completed_fits"] == 16
    archive = root / result["superseded_preparation_manifest"]
    assert archive.read_bytes() == old
    assert json.loads(old_manifest.read_text()) == result
    assert (deferred_root / "protocol/protocol.json").read_text() == "must not be read"


def test_continuation_finalizes_completed_training_and_trains_only_unstarted(queue_fixture, monkeypatch):
    config, root, fits, record = queue_fixture
    complete = record(fits[0], "complete", 50)
    complete_before = (complete / "run.json").read_bytes()
    record(fits[1], "failed_or_interrupted", 50)
    partial = record(fits[2], "failed_or_interrupted", 10)
    partial_before = (partial / "run.json").read_bytes()
    calls = []

    def finalize(config, fit, *args, **kwargs):
        calls.append(("finalize", fit["run_id"]))
        record(fit, "complete", 50)

    def train(config, fit, *args, **kwargs):
        calls.append(("train", fit["run_id"]))
        record(fit, "complete", 50)

    monkeypatch.setattr(cli, "finalize_fit", finalize)
    monkeypatch.setattr(cli, "train_fit", train)
    result = cli.run(config, "cpu", continuation=True)
    assert calls == [("finalize", fits[1]["run_id"]), ("train", fits[3]["run_id"])]
    assert (complete / "run.json").read_bytes() == complete_before
    assert (partial / "run.json").read_bytes() == partial_before
    assert result["unresolved"][0]["kind"] == "incomplete_training_preserved"
    assert result["active_fit"] is None


def test_persistent_checkpoint_failure_does_not_stop_independent_fits(queue_fixture, monkeypatch):
    config, root, fits, record = queue_fixture
    record(fits[0], "failed_or_interrupted", 50)
    trained = []

    def finalize(*args, **kwargs):
        raise cli.JointCheckpointIntegrityError("objects/bad.safetensors", "a" * 64, "b" * 64, 3)

    def train(config, fit, *args, **kwargs):
        trained.append(fit["run_id"])
        record(fit, "complete", 50)

    monkeypatch.setattr(cli, "finalize_fit", finalize)
    monkeypatch.setattr(cli, "train_fit", train)
    result = cli.run(config, "cpu", continuation=True)
    assert trained == [fit["run_id"] for fit in fits[1:]]
    failed_dir = root / "runs" / fits[0]["run_id"]
    assert json.loads((failed_dir / "run.json").read_text())["status"] == "checkpoint_unresolved"
    assert len(list((failed_dir / "checkpoint_issues").glob("*.json"))) == 1
    assert result["unresolved"][0]["error"]["attempts"] == 3
    assert result["queue_status"] == "finished_with_unresolved_fits"


def test_non_checkpoint_training_failure_still_stops_queue(queue_fixture, monkeypatch):
    config, root, fits, record = queue_fixture
    calls = []

    def train(config, fit, *args, **kwargs):
        calls.append(fit["run_id"])
        raise ValueError("Nonfinite gradient")

    monkeypatch.setattr(cli, "train_fit", train)
    with pytest.raises(ValueError, match="Nonfinite gradient"):
        cli.run(config, "cpu", continuation=True)
    assert calls == [fits[0]["run_id"]]


def test_frozen_probe_requires_downstream_update_and_visual_eval():
    import torch
    model = torch.nn.Module()
    model.visual = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.fusion = torch.nn.Linear(2, 2)
    model.visual.requires_grad_(False).eval()
    probes = cli._update_probes(model, "frozen_online")
    assert set(probes) == {"fusion"}
    with pytest.raises(ValueError, match="fusion parameter did not update"):
        cli._check_updates(model, "frozen_online", probes)
    with torch.no_grad():
        model.fusion.weight.add_(0.1)
    assert cli._check_updates(model, "frozen_online", probes) == {"fusion": "weight"}
    model.visual.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        cli._check_updates(model, "frozen_online", probes)
    model.visual.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable visual parameters"):
        cli._update_probes(model, "frozen_online")


def test_partial_probe_checks_both_visual_and_downstream_updates():
    import torch
    model = torch.nn.Module()
    model.visual = torch.nn.Linear(2, 2)
    model.fusion = torch.nn.Linear(2, 2)
    probes = cli._update_probes(model, "joint_partial")
    assert set(probes) == {"visual", "fusion"}
    with torch.no_grad():
        model.fusion.weight.add_(0.1)
    with pytest.raises(ValueError, match="visual parameter did not update"):
        cli._check_updates(model, "joint_partial", probes)


def test_geo_probe_checks_only_classifier_and_rejects_visual_encoder():
    import torch
    model = torch.nn.Module()
    model.visual = None
    model.fusion = torch.nn.Linear(2, 2)
    probes = cli._update_probes(model, "geo_only")
    assert set(probes) == {"fusion"}
    with pytest.raises(ValueError, match="fusion parameter did not update"):
        cli._check_updates(model, "geo_only", probes)
    with torch.no_grad():
        model.fusion.weight.add_(0.1)
    assert cli._check_updates(model, "geo_only", probes) == {"fusion": "weight"}
    model.visual = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="requires no visual encoder"):
        cli._update_probes(model, "geo_only")


def test_frozen_package_report_routes_regime_and_reference(tmp_path, monkeypatch):
    from multimodal import attention_fusion_joint_reporting as reporting
    config = protocol.load_config(protocol.FROZEN_ONLINE_CONFIG)
    monkeypatch.setattr(cli, "study_root", lambda config: tmp_path)
    calls = []
    monkeypatch.setattr(reporting, "build_report", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.package_report(config, render_confusions=False)
    args, kwargs = calls[0]
    assert args == (tmp_path, tmp_path / "reports/seed1_frozen_online")
    assert kwargs["seed"] == 1 and kwargs["regime"] == "frozen_online"
    assert kwargs["paired_reference_root"] == cli.REPO_ROOT / config["reporting"]["paired_reference_root"]
    assert kwargs["render_confusions"] is False


def test_frozen_package_status_contains_only_authorized_frozen_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "study_root", lambda config: tmp_path)
    config = protocol.load_config(protocol.FROZEN_ONLINE_CONFIG)
    result = cli.status(config)
    assert result["authorized_fits"] == 16 and result["complete"] == 0
    assert {fit["regime"] for fit in result["fits"]} == {"frozen_online"}
    assert {fit["seed"] for fit in result["fits"]} == {1}


def test_report_routes_completed_packages_without_mutating_them(tmp_path, monkeypatch):
    from multimodal import attention_fusion_joint_reporting as reporting
    config = protocol.load_config()
    config["reporting"]["completed_reference_roots"] = ["previous_partial", "previous_frozen"]
    monkeypatch.setattr(cli, "study_root", lambda config: tmp_path)
    calls = []
    monkeypatch.setattr(reporting, "build_report", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.package_report(config, render_confusions=False)
    assert calls[0][1]["completed_reference_roots"] == [
        cli.REPO_ROOT / "previous_partial", cli.REPO_ROOT / "previous_frozen"]


def test_report_routes_deferred_regime_metadata(tmp_path, monkeypatch):
    from multimodal import attention_fusion_joint_reporting as reporting
    config = protocol.load_config(protocol.package_config_path(2, "geo_only"))
    monkeypatch.setattr(cli, "study_root", lambda config: tmp_path)
    calls = []
    monkeypatch.setattr(reporting, "build_report", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.package_report(config, render_confusions=False)
    kwargs = calls[0][1]
    assert kwargs["seed"] == 2 and kwargs["regime"] == "geo_only"
    assert kwargs["deferred_reference_regimes"] == ["frozen_online"]
    assert len(kwargs["completed_reference_roots"]) == 1

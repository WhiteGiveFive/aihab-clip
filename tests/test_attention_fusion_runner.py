"""The execution interface must use the frozen matrix without loading test data."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from tools.prepare_attention_fusion_protocol import (
    DEFAULT_CONFIG, REPO_ROOT, ProtocolError, build_matrix, content_hash, file_hash,
)
from tools.run_multimodal_attention_fusion import (
    plan_status, read_frozen_plan, select_fit,
)


@pytest.fixture
def frozen(tmp_path):
    config = yaml.safe_load((REPO_ROOT / DEFAULT_CONFIG).read_text())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    directory = tmp_path / config["paths"]["artifact_root"] / "protocol"
    directory.mkdir(parents=True)
    matrix = build_matrix(config)
    (directory / "resolved_protocol.yaml").write_text(yaml.safe_dump(config))
    (directory / "candidate_matrix.json").write_text(json.dumps(matrix))
    (directory / "input_audit.json").write_text('{"fixture": true}\n')
    manifest = {
        "study_id": config["study_id"],
        "effective_config_sha256": content_hash(config),
        "outputs": {key: {"sha256": file_hash(directory / filename)}
                    for key, filename in (("resolved_protocol", "resolved_protocol.yaml"),
                                          ("candidate_matrix", "candidate_matrix.json"))},
        "input_audit": {"sha256": file_hash(directory / "input_audit.json")},
    }
    manifest["manifest_payload_sha256"] = content_hash(manifest)
    (directory / "protocol_manifest.json").write_text(json.dumps(manifest))
    return config, matrix, config_path, directory


def test_plan_and_status_need_no_input_payloads(frozen, tmp_path):
    config, matrix, path, _ = frozen
    loaded, actual, _ = read_frozen_plan(path, tmp_path)
    assert loaded == config and actual == matrix
    status = plan_status(config, matrix, tmp_path)
    assert status["development_status_counts"] == {"planned": 228}
    assert not status["verification_runs_count_as_development"]
    run = tmp_path / matrix["development_fits"][0]["output_dir"]
    run.mkdir(parents=True)
    (run / "run.json").write_text('{"status":"interrupted","completed_epoch":7}')
    status = plan_status(config, matrix, tmp_path)
    assert status["development_status_counts"] == {"interrupted": 1, "planned": 227}
    assert status["fits"][0]["completed_epoch"] == 7


@pytest.mark.parametrize("target", ["config", "manifest", "matrix", "audit", "resolved"])
def test_frozen_bindings_reject_drift_before_loading_data(frozen, tmp_path, target):
    config, _, path, directory = frozen
    if target == "config":
        config["training"]["batch_size"] = 64
        path.write_text(yaml.safe_dump(config))
    else:
        file = {"manifest": "protocol_manifest.json", "matrix": "candidate_matrix.json",
                "audit": "input_audit.json", "resolved": "resolved_protocol.yaml"}[target]
        with (directory / file).open("a") as stream:
            stream.write(" ")
        if target == "manifest":
            value = json.loads((directory / file).read_text())
            value["study_id"] = "altered"
            (directory / file).write_text(json.dumps(value))
    with pytest.raises(ProtocolError):
        read_frozen_plan(path, tmp_path)


def test_fit_lookup_rejects_unregistered_and_ambiguous_trials(frozen):
    _, matrix, _, _ = frozen
    fit_id = matrix["development_fits"][0]["fit_id"]
    assert select_fit(matrix, fit_id) == matrix["development_fits"][0]
    with pytest.raises(ProtocolError, match="Unknown"):
        select_fit(matrix, "undeclared_lr_and_seed")
    duplicate = copy.deepcopy(matrix)
    duplicate["development_fits"].append(duplicate["development_fits"][0])
    with pytest.raises(ProtocolError, match="ambiguous"):
        select_fit(duplicate, fit_id)


def test_status_distinguishes_uncommitted_and_corrupt_runs(frozen, tmp_path):
    config, matrix, _, _ = frozen
    run = tmp_path / matrix["development_fits"][0]["output_dir"]
    run.mkdir(parents=True)
    (run / "run.json").write_text('{"status":"complete","completed_epoch":50}')
    status = plan_status(config, matrix, tmp_path)
    assert status["development_status_counts"] == {"incomplete_commit": 1, "planned": 227}
    fit = matrix["development_fits"][0]
    spec = {"config_sha256": content_hash(config), "verification": False,
            "candidate_id": fit["candidate_id"], "geo_mode": fit["geo_preprocessing"],
            "learning_rate": fit["learning_rate"], "seed": fit["seed"]}
    manifest = {"run_identity_sha256": content_hash(spec)}
    manifest["manifest_payload_sha256"] = content_hash(manifest)
    (run / "run_spec.json").write_text(json.dumps(spec))
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert plan_status(config, matrix, tmp_path)["fits"][0]["status"] == "complete"
    (run / "manifest.json").write_text('{"broken":true}')
    assert plan_status(config, matrix, tmp_path)["fits"][0]["status"] == "invalid_run_metadata"
    (run / "run.json").write_text("interrupted while writing invalid JSON")
    assert plan_status(config, matrix, tmp_path)["fits"][0]["status"] == "invalid_run_metadata"


def test_verification_reads_train_prefix_and_cannot_fallback_from_corrupt_full_cache(frozen, tmp_path, monkeypatch):
    import numpy as np
    import multimodal.attention_fusion_inputs as inputs
    import tools.run_multimodal_attention_fusion as runner

    config, _, _, _ = frozen
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    prepared = tmp_path / config["paths"]["prepared_cache_template"].format(seed=1, role="train")
    prepared.parent.mkdir(parents=True)
    (prepared.parent / "manifest.json").write_text("{}")
    seen = []
    monkeypatch.setattr(inputs, "load_prepared_inputs", lambda c, s, role: (
        seen.append(role) or {"label": np.arange(4), "row_uid": np.array(list("abcd"))}))
    monkeypatch.setattr(inputs, "load_input_provenance", lambda *a, **kw: {})
    monkeypatch.setattr(inputs, "load_verification_inputs", lambda *a, **kw: pytest.fail("unexpected fallback"))
    arrays, provenance = runner.training_verification_data(config, 1, 2)
    assert seen == ["train"] and arrays["row_uid"].tolist() == ["a", "b"]
    assert provenance["source_role"] == "train" and provenance["verification_training_prefix_rows"] == 2

    def reject(*args):
        raise inputs.InputError("Corrupt full cache")
    monkeypatch.setattr(inputs, "load_prepared_inputs", reject)
    with pytest.raises(inputs.InputError, match="Corrupt"):
        runner.training_verification_data(config, 1, 2)

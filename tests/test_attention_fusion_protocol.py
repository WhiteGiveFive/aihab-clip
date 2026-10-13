"""Check workload identity, development isolation and post-fit protocol immutability."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tools.prepare_attention_fusion_protocol import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    REPORT,
    ProtocolError,
    build_matrix,
    prepare_protocol,
    validate_audit,
    validate_config,
)


@pytest.fixture
def config():
    return yaml.safe_load((REPO_ROOT / DEFAULT_CONFIG).read_text())


def synthetic_audit(config, repo):
    """Provide authenticated-record shapes without creating any input payloads."""
    paths = config["paths"]

    def record(path):
        return {"path": str(repo / path), "sha256": "a" * 64}

    runtime = {}
    for key, relative in (("script", "tools/audit_attention_fusion_inputs.py"),
                          ("canonical_helpers", "multimodal/geo_helpfulness_protocol.py")):
        code = repo / relative
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text("# synthetic audit source\n")
        runtime[key] = {"path": str(code), "sha256": hashlib.sha256(code.read_bytes()).hexdigest()}

    return {
        "schema_version": "attention_fusion.input_audit.v1",
        "study_id": config["study_id"], "status": "passed",
        "repo_root": str(repo), "source_protocol_root": str(repo / paths["source_protocol_root"]),
        "checks": {"checkpoint_lineage": True, "no_test_overlap": True},
        "summary": {"test_features_or_labels_opened": False},
        "runtime": runtime,
        "boundary": {"test_labels_read": False, "test_features_read": False,
                     "mixed_embedding_npz_opened": False},
        "protocol_manifest": record(Path(paths["source_protocol_root"]) / "protocol/protocol_manifest.json"),
        "protocol_files": {"development_assignments.parquet": record(paths["assignments"])},
        "label_blind_test_identity": record("identity_only.json"),
        "development_geo_source_files": [record(path) for path in paths["development_geo_tables"]],
        "development_images": {"image_source": paths["image_root"]},
        "native_geo": {"feature_columns": config["data"]["geo_columns"]},
        "encoders": [{"seed": seed, "checkpoint": record(paths["encoder_template"].format(seed=seed))}
                     for seed in [1, 2, 3, 4]],
        "validation_image_caches": [
            {"seed": seed, "features": record(paths["validation_cache_template"].format(seed=seed))}
            for seed in [1, 2, 3, 4]
        ],
    }


def test_complete_workload_and_shared_image_only(config):
    matrix = build_matrix(config)
    assert matrix["counts"] == {
        "candidate_geo_arms": 19, "development_fits": 228,
        "selected_train_checkpoint_references": 76, "all_development_refits": 76,
    }
    fits = matrix["development_fits"]
    assert len({fit["fit_id"] for fit in fits}) == 228
    assert len({fit["output_dir"] for fit in fits}) == 228
    image = [fit for fit in fits if fit["candidate_id"] == "image_only"]
    assert len(image) == 12
    assert {fit["geo_preprocessing"] for fit in image} == {"not_applicable"}
    for candidate in config["candidates"][1:]:
        assert len([fit for fit in fits if fit["candidate_id"] == candidate["id"]]) == 36
    assert sum(fit["epochs"] for fit in fits) == 11400
    assert all(fit["fit_role"] == "train" and fit["evaluation_role"] == "validation" for fit in fits)
    assert all(fit["regime"] == "frozen_cached"
               and "/development/frozen_cached/" in fit["output_dir"] for fit in fits)
    for selected in matrix["selected_train_checkpoint_references"]:
        assert selected["learning_rate"] is None and selected["selection_pending"]
        assert not selected["retrain"]
        assert len(selected["candidate_fit_ids"]) == 3
    for refit in matrix["all_development_refits"]:
        assert refit["fit_rows"] == 4200 and refit["evaluation_role"] is None
        assert refit["learning_rate"] is None and refit["selection_pending"]
        assert refit["refit_geo_statistics"] == (refit["geo_preprocessing"] == "standardize")


def test_matrix_order_and_geo_identity_are_deterministic(config):
    first = build_matrix(config)
    second = build_matrix(copy.deepcopy(config))
    assert first == second
    assert [arm["arm_id"] for arm in first["candidate_geo_arms"][:4]] == [
        "image_only__geo_not_applicable", "geo_only__geo_native",
        "geo_only__geo_l2", "geo_only__geo_standardize",
    ]
    assert [(fit["seed"], fit["learning_rate"]) for fit in first["development_fits"][:4]] == [
        (1, 0.0001), (1, 0.0003), (1, 0.001), (2, 0.0001),
    ]
    assert all(fit["inputs"]["encoder_checkpoint"] is None
               and fit["inputs"]["image_caches"] == {}
               for fit in first["development_fits"] if fit["candidate_id"] == "geo_only")


@pytest.mark.parametrize("section,key,value", [
    ("budget", "development_fits", 227),
    ("training", "epochs", 51),
    ("training", "learning_rates", [0.0001, float("nan"), 0.001]),
    ("data", "allow_test_features_or_labels", True),
    ("final_fit", "keep_same_train_partition_encoder", False),
    ("final_fit", "initialization", "selected_head_warm_start"),
    ("selection", "select_best_seed", True),
    ("geo", "comparison", ["native", "l2"]),
    ("model", "classifier", "linear_gelu_dropout_linear"),
    ("image", "extraction_batch_size", 1),
    ("image", "extraction_cudnn_allow_tf32", False),
])
def test_reject_changed_budget_or_scientific_boundary(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ProtocolError):
        validate_config(config)


@pytest.mark.parametrize("mutation", ["failed_status", "failed_check", "no_checks", "test_opened", "wrong_source", "stale_audit_code"])
def test_reject_failed_or_unbound_input_audit(config, tmp_path, mutation):
    audit = synthetic_audit(config, tmp_path)
    if mutation == "failed_status":
        audit["status"] = "failed"
    elif mutation == "failed_check":
        audit["checks"]["checkpoint_lineage"] = False
    elif mutation == "no_checks":
        audit["checks"] = {}
    elif mutation == "test_opened":
        audit["boundary"]["test_features_read"] = True
    elif mutation == "wrong_source":
        audit["development_geo_source_files"][0]["path"] = str(tmp_path / "test.parquet")
    else:
        (tmp_path / "tools/audit_attention_fusion_inputs.py").write_text("# source changed\n")
    with pytest.raises(ProtocolError):
        validate_audit(audit, config, tmp_path)


def test_prepare_uses_audit_records_without_opening_inputs_and_guards_started_study(config, tmp_path):
    root = tmp_path / config["paths"]["artifact_root"]
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    audit = synthetic_audit(config, tmp_path)
    audit_path = protocol / "input_audit.json"
    audit_path.write_text(json.dumps(audit))
    config_path = tmp_path / DEFAULT_CONFIG
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    report = tmp_path / REPORT
    report.parent.mkdir(parents=True)
    report.write_text("Synthetic protocol report\n")
    # None of the recorded model, image, parquet or cache paths exist.
    manifest = prepare_protocol(config_path, tmp_path)
    assert manifest["counts"]["development_fits"] == 228
    assert manifest["runtime"]["git_dirty"] is None
    assert not manifest["boundary"]["data_payload_loaded"]
    assert yaml.safe_load((protocol / "resolved_protocol.yaml").read_text()) == config
    matrix_before = (protocol / "candidate_matrix.json").read_bytes()
    run = root / "development/example/run_status.json"
    run.parent.mkdir(parents=True)
    run.write_text('{"status": "running"}\n')
    audit["created_at"] = "new timestamp does not change input lineage"
    audit_path.write_text(json.dumps(audit))
    repeated = prepare_protocol(config_path, tmp_path)
    assert repeated["existing_run_files"]
    assert (protocol / "candidate_matrix.json").read_bytes() == matrix_before
    assert repeated["effective_config_sha256"] == manifest["effective_config_sha256"]
    manifest_before = (protocol / "protocol_manifest.json").read_bytes()
    config["training"]["weight_decay"] = 0.0002
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ProtocolError, match="changed configuration"):
        prepare_protocol(config_path, tmp_path)
    assert (protocol / "protocol_manifest.json").read_bytes() == manifest_before


def test_reject_changed_input_lineage_after_selection(config, tmp_path):
    root = tmp_path / config["paths"]["artifact_root"]
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    audit = synthetic_audit(config, tmp_path)
    audit_path = protocol / "input_audit.json"
    audit_path.write_text(json.dumps(audit))
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config))
    report = tmp_path / REPORT
    report.parent.mkdir(parents=True)
    report.write_text("Synthetic protocol report\n")
    prepare_protocol(config_path, tmp_path)
    selection = root / "selection/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}\n")
    audit["encoders"][0]["checkpoint"]["sha256"] = "b" * 64
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ProtocolError, match="changed input lineage"):
        prepare_protocol(config_path, tmp_path)

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_confidnet as workflow
from multimodal import geo_helpfulness_oof as m2


@pytest.fixture
def declared(tmp_path, monkeypatch):
    config = workflow._json(workflow.CONFIG)
    project = tmp_path / "project"
    project.mkdir()
    config_path = project / "config.json"
    config_path.write_text(json.dumps(config))
    protocol = project / "protocol_v1"
    parent = protocol / "protocol"
    parent.mkdir(parents=True)
    for name in (
        "development_assignments.parquet",
        "resolved_protocol.yaml",
        "protocol_manifest.json",
    ):
        (parent / name).write_text("synthetic parent; never deserialize")
    for relative in workflow.CODE_PATHS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic implementation identity")
    monkeypatch.setattr(workflow, "PROJECT_ROOT", project)
    monkeypatch.setattr(workflow, "CONFIG", config_path)
    monkeypatch.setattr(workflow, "PROTOCOL", protocol)
    producers = []
    monkeypatch.setattr(m2, "load_frozen_context", lambda: object())

    def validate_producer(context, spec, *, seed):
        producers.append(seed)

    monkeypatch.setattr(m2, "validate_producer", validate_producer)
    output = project / "experiment"
    result = workflow.declare(output)
    assert result["status"] == "declared"
    assert producers == [1, 2, 3, 4]
    return output, config_path, parent


def test_declaration_reuse_checks_same_config_and_all_parents(declared):
    output, _, _ = declared
    assert workflow.declare(output)["status"] == "reused_valid"
    value = workflow.validate_declaration(output, check_parents=True)
    assert value["locked_test_access"] is False
    assert value["validation_selection"] == "none"
    assert set(workflow.CODE_PATHS) <= set(value["file_sha256"])


def test_declaration_rejects_changed_config(declared):
    output, config_path, _ = declared
    config = json.loads(config_path.read_text())
    config["parameters"]["epochs"] = 101
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        workflow.validate_declaration(output)


@pytest.mark.parametrize("changed", ["implementation", "parent"])
def test_declaration_rejects_changed_source_hash(declared, changed):
    output, _, parent = declared
    path = (
        workflow.PROJECT_ROOT / workflow.CODE_PATHS[0]
        if changed == "implementation"
        else parent / "protocol_manifest.json"
    )
    path.write_text("changed source")
    with pytest.raises(ValueError, match="changed"):
        workflow.validate_declaration(output, check_parents=True)


@pytest.mark.parametrize("change", ["tamper", "extra", "missing", "declaration", "role"])
def test_sealed_stage_detects_tampering_and_lineage_changes(tmp_path, change):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "declaration.json").write_text("{}")
    staging = workflow._staging(root, "fixture")
    (staging / "artifact.json").write_text('{"value":1}')
    destination = root / "fixture"
    workflow._seal_stage(staging, destination, root, role="fixture")
    assert workflow._validate_stage(destination, root, role="fixture")["role"] == "fixture"
    role = "fixture"
    if change == "tamper":
        (destination / "artifact.json").write_text('{"value":2}')
    elif change == "extra":
        (destination / "unexpected.txt").write_text("extra")
    elif change == "missing":
        (destination / "artifact.json").unlink()
    elif change == "declaration":
        (root / "declaration.json").write_text('{"changed":true}')
    else:
        role = "different_role"
    with pytest.raises(ValueError):
        workflow._validate_stage(destination, root, role=role)


def _payload(role):
    rows = 3378 if role == "train" else 822
    hidden = np.zeros((rows, 256), dtype=np.float32)
    logits = np.zeros((rows, 18), dtype=np.float64)
    result = {
        "row_uid": np.asarray([f"{role}_row_{i}" for i in range(rows)]),
        "plot_idx": np.asarray([f"{role}_plot_{i}" for i in range(rows)]),
    }
    for mode in workflow.MODES:
        result[f"{mode}_hidden"] = hidden
        result[f"{mode}_logits"] = logits
    if role == "train":
        result["labels"] = np.arange(rows, dtype=np.int64) % 18
    return result


def test_npz_never_writes_or_loads_pickled_feature_objects(tmp_path):
    path = tmp_path / "features.npz"
    with pytest.raises(ValueError, match="object"):
        workflow._save_arrays(path, {"hidden": np.asarray([object()], dtype=object)})
    assert not path.exists()
    np.savez_compressed(path, hidden=np.asarray([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        workflow._load_arrays(path)


@pytest.mark.parametrize("forbidden", ["labels", "label_id_dense", "tcp_target"])
def test_validation_feature_cache_rejects_supervision(forbidden):
    train, validation = _payload("train"), _payload("validation")
    workflow._check_arrays(train, validation)
    validation[forbidden] = np.zeros(822, dtype=np.int64)
    with pytest.raises(ValueError):
        workflow._check_arrays(train, validation)


def test_score_requires_every_seed_before_any_validation_label_read(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(workflow, "validate_declaration", lambda *args, **kwargs: {})

    def validate_seed(root, seed):
        seen.append(seed)
        if seed == 4:
            raise FileNotFoundError("last seed has not been sealed")

    def forbidden(*args, **kwargs):
        raise AssertionError("labels were read before all seed predictions were verified")

    monkeypatch.setattr(workflow, "_validate_seed", validate_seed)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(FileNotFoundError, match="last seed"):
        workflow.score(tmp_path / "experiment")
    assert seen == [1, 2, 3, 4]


def _reseal_file(stage, filename):
    """Keep hashes valid to test semantic replay checks rather than hash checks."""
    manifest_path = stage / "manifest.json"
    manifest = workflow._json(manifest_path)
    manifest["file_sha256"][filename] = workflow._hash(stage / filename)
    manifest_path.write_text(json.dumps(manifest))


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    from multimodal import geo_helpfulness_confidnet_extract as extraction

    root = tmp_path / "experiment"
    root.mkdir()
    config = workflow._json(workflow.CONFIG)
    config["parameters"].update({"hidden_widths": [4, 4, 4, 4], "epochs": 1})
    declaration = {"config": config}
    (root / "declaration.json").write_text(json.dumps(declaration))
    monkeypatch.setattr(workflow, "SEEDS", (1,))
    monkeypatch.setattr(workflow, "_config", lambda: deepcopy(config))
    monkeypatch.setattr(
        workflow, "validate_declaration", lambda *args, **kwargs: deepcopy(declaration)
    )
    monkeypatch.setattr(
        extraction,
        "extract_seed",
        lambda *args, **kwargs: {
            "train": _payload("train"),
            "validation": _payload("validation"),
            "metadata": {"fixture": "synthetic representations", "training_seed": 1},
        },
    )
    result = workflow.run_seed(1, root, device="cpu")
    assert result == {"seed": 1, "status": "complete", "models": 3}
    return root


def test_synthetic_extraction_fit_json_reload_and_validation_complete(fitted):
    result = workflow.validate(fitted)
    assert result == {"status": "valid", "models": 3, "scored": False}
    table = pd.read_parquet(fitted / "fits/seed_1/validation_predictions.parquet")
    assert not set(table) & {"labels", "label_id_dense", "tcp_target"}
    assert workflow.run_seed(1, fitted, device="cpu")["status"] == "reused_valid"


@pytest.mark.parametrize("change", ["row_uid", "plot_idx", "training_seed", "prediction", "logits"])
def test_replay_detects_semantic_identity_or_output_change_even_with_valid_hashes(
    fitted, change
):
    stage = fitted / "fits/seed_1"
    filename = "validation_predictions.parquet"
    table = pd.read_parquet(stage / filename)
    if change in {"row_uid", "plot_idx"}:
        table.loc[0, change] = "incorrect_identity"
    elif change == "training_seed":
        table.loc[0, "training_seed"] = 2
    elif change == "prediction":
        table.loc[0, "geo_tcp_hat"] += 0.01
    else:
        value = table.loc[0, "geo_logits"].copy()
        value[0] += 0.01
        table.at[0, "geo_logits"] = value
    table.to_parquet(stage / filename, index=False)
    _reseal_file(stage, filename)
    with pytest.raises(ValueError):
        workflow.validate(fitted)

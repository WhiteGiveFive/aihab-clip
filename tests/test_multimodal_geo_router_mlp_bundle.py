"""Synthetic MLP publication, recovery, and label-blind replay checks."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_router_mlp as bundle
from multimodal import geo_helpfulness_router_mlp_state as portable
from multimodal.geo_helpfulness_router_candidate_state import variant_feature_indices


def _configuration():
    records = []
    for variant in bundle.VARIANTS:
        for widths in bundle.ARCHITECTURES:
            records.append({
                "config_id": f"mlp__{variant}__h" + "_".join(map(str, widths)),
                "family": "mlp",
                "variant": variant,
                "parameters": {**portable.MLP_SHARED, "hidden_widths": list(widths)},
            })
    return {
        "schema_version": "geo_helpfulness.router_mlp_config.v1",
        "study_id": bundle.STUDY_ID,
        "training_seeds": [1, 2, 3, 4],
        "target_order": list(bundle.TARGET_ORDER),
        "configuration_count": 12,
        "candidate_seed_count": 48,
        "configurations": records,
    }


def test_declared_configuration_has_exact_order_twelve_specs_and_48_states():
    configuration = _configuration()
    assert bundle._config(configuration) == configuration["configurations"]
    files = bundle._prediction_files(configuration)
    assert len(files) == 146
    assert sum(name.startswith("models/") for name in files) == 48
    assert sum(name.startswith("train/") for name in files) == 48
    assert sum(name.startswith("validation/") for name in files) == 48
    assert {"training_curves.parquet", "validation_features.parquet"} <= files


def test_repository_predeclaration_is_accepted_without_coercion():
    configuration = bundle.read_json(
        bundle.PROJECT_ROOT / "configs/multimodal_geo_router_mlp_v1.json"
    )
    assert len(bundle._config(configuration)) == 12


@pytest.mark.parametrize("problem", [
    "missing_candidate", "candidate_order", "duplicate_id", "unsafe_id",
    "wrong_family", "missing_default", "changed_dropout", "extra_parameter",
    "seed_order", "target_order", "configuration_count", "candidate_seed_count",
])
def test_configuration_rejects_undeclared_or_ambiguous_settings(problem):
    config = _configuration()
    records = config["configurations"]
    if problem == "missing_candidate":
        records.pop()
    elif problem == "candidate_order":
        records[0], records[1] = records[1], records[0]
    elif problem == "duplicate_id":
        records[1]["config_id"] = records[0]["config_id"]
    elif problem == "unsafe_id":
        records[0]["config_id"] = "../outside"
    elif problem == "wrong_family":
        records[0]["family"] = "rf"
    elif problem == "missing_default":
        records[0]["parameters"].pop("dropout")
    elif problem == "changed_dropout":
        records[0]["parameters"]["dropout"] = 0.2
    elif problem == "extra_parameter":
        records[0]["parameters"]["class_weight"] = "balanced"
    elif problem == "seed_order":
        config["training_seeds"] = [4, 3, 2, 1]
    elif problem == "target_order":
        config["target_order"] = list(reversed(bundle.TARGET_ORDER))
    elif problem == "configuration_count":
        config["configuration_count"] = 13
    else:
        config["candidate_seed_count"] = 49
    with pytest.raises(ValueError):
        bundle._config(config)


def _state(spec, seed):
    indices = variant_feature_indices(spec["variant"])
    dimensions = [len(indices), *spec["parameters"]["hidden_widths"], 4]
    return portable.seal_state({
        "schema_version": portable.SCHEMA_VERSION,
        **deepcopy(spec),
        "training_seed": seed,
        "feature_input_names": bundle.FEATURES,
        "feature_indices": indices,
        "classes": [0, 1, 2, 3],
        "target_order": list(bundle.TARGET_ORDER),
        "fit_role": "development_train_oof",
        "fit_row_count": 3378,
        "initialization_seed": seed,
        "shuffle_seed": seed,
        "dropout_seed": seed,
        "parameter_count": portable.parameter_count(
            spec["variant"], spec["parameters"]["hidden_widths"]
        ),
        "implementation": deepcopy(portable.IMPLEMENTATION),
        "layers": [
            {"weight": np.zeros((right, left)).tolist(), "bias": [0.0] * right}
            for left, right in zip(dimensions, dimensions[1:])
        ],
    })


def _diagnostics(state):
    return {
        "config_id": state["config_id"], "family": "mlp",
        "variant": state["variant"], "training_seed": state["training_seed"],
        "fit_role": "development_train_oof", "fit_row_count": 3378,
        "feature_count": len(state["feature_indices"]),
        "hidden_widths": state["parameters"]["hidden_widths"],
        "parameter_count": state["parameter_count"],
        "fit_performed": True, "final_epoch": 300, "checkpoint": "final_epoch",
        "history": [
            {
                "epoch": epoch, "optimization_cross_entropy": float(np.log(4)),
                "train_nll": float(np.log(4)), "train_accuracy": 0.25,
                "maximum_gradient_l2_norm": 0.0, "gradients_finite": True,
            }
            for epoch in range(1, 301)
        ],
        "validation_labels_accessed": False,
        "validation_features_used_for_fitting": False,
        "early_stopping_used": False, "sample_or_class_weighting_used": False,
        "optimizer": "AdamW", "learning_rate": 0.001,
        "weight_decay": 0.001, "bias_weight_decay": 0.0,
        "threadpool_limit": 1, "deterministic_algorithms": True,
        "warnings": [], "state_sha256": state["state_sha256"],
        "fit_and_export_seconds": 0.0,
    }


@pytest.fixture
def partial_bundle(tmp_path, monkeypatch):
    output, protocol, capacity_root = (
        tmp_path / "output", tmp_path / "protocol", tmp_path / "capacity"
    )
    protocol.mkdir()
    capacity_root.mkdir()
    parent = protocol / "frozen.txt"
    parent.write_text("frozen synthetic parent\n")
    spec = _configuration()["configurations"][8]
    # Keep publication tests small; the complete 12/48 contract is tested above.
    config = {"study_id": bundle.STUDY_ID, "configurations": [spec]}
    monkeypatch.setattr(bundle, "_config", lambda value: value["configurations"])
    config_path = tmp_path / "configuration.json"
    bundle.write_json(config_path, config)
    declaration = bundle.capacity._seal({
        "schema_version": "geo_helpfulness.router_mlp_declaration.v1",
        "study_id": bundle.STUDY_ID, "configuration": config,
        "protocol_root": str(protocol), "capacity_root": str(capacity_root),
        "parent_file_sha256": {
            str(parent): bundle.file_hash(parent),
            str(config_path): bundle.file_hash(config_path),
        },
    }, "declaration_sha256")
    bundle.write_json(output / "declaration.json", declaration)
    inputs, validation_frames, target_frames = {}, [], []
    for seed in range(1, 5):
        matrix = np.zeros((4, 727), dtype=np.float64)
        identity = pd.DataFrame({
            "row_uid": [f"image-{i}" for i in range(4)],
            "plot_idx": [f"plot-{i}" for i in range(4)],
            "training_seed": np.full(4, seed, dtype=np.int8),
        })
        validation_frames.append(pd.concat([
            identity, pd.DataFrame(matrix, columns=bundle.FEATURES)
        ], axis=1))
        target_frames.append(identity.assign(target_state=list(bundle.TARGET_ORDER)))
        inputs[seed] = {
            "X_train": matrix, "X_validation": matrix.copy(),
            "train_identity": identity, "validation_identity": identity.copy(),
            "y_train": np.arange(4),
        }
        state = _state(spec, seed)
        record = bundle.capacity._seal({
            "declaration_sha256": declaration["declaration_sha256"],
            "state": state, "diagnostics": _diagnostics(state),
        }, "record_sha256")
        bundle.write_json(
            output / "predictions/models" / f"{bundle.capacity._stem(spec, seed)}.json",
            record,
        )
    complete_inputs = {
        "seed_inputs": inputs,
        "validation": pd.concat(validation_frames, ignore_index=True),
    }
    monkeypatch.setitem(
        sys.modules, "multimodal.geo_helpfulness_router_capacity_inputs",
        SimpleNamespace(load_capacity_inputs=lambda *args: complete_inputs),
    )

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("owned partial recovery cannot refit existing models")

    monkeypatch.setitem(
        sys.modules, "multimodal.geo_helpfulness_router_mlp_fit",
        SimpleNamespace(fit_candidate=forbidden_fit),
    )
    monkeypatch.setattr(bundle.capacity, "_replay_inputs", lambda *args: inputs)
    original_read = pq.read_table
    targets = pa.Table.from_pandas(pd.concat(target_frames, ignore_index=True))

    def label_blind_read(path, *args, **kwargs):
        if Path(path) == protocol / "router/router_dataset.parquet":
            assert kwargs.get("columns") == [*bundle.IDENTITY, "target_state"]
            return targets
        if "validation_targets" in str(path) or Path(path).name == "test.parquet":
            raise AssertionError("prediction replay decoded forbidden targets")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(bundle.pq, "read_table", label_blind_read)
    return SimpleNamespace(
        output=output, protocol=protocol, capacity=capacity_root,
        config_path=config_path, declaration=declaration, spec=spec, inputs=inputs,
    )


def _build(fixture):
    return bundle.build_predictions(
        fixture.protocol, fixture.output, fixture.config_path, fixture.capacity
    )


def _snapshot(root):
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


def _replace_json(path, value, hash_key=None):
    if hash_key:
        value.pop(hash_key, None)
        value = bundle.capacity._seal(value, hash_key)
    path.chmod(0o644)
    path.unlink()
    bundle.write_json(path, value)


def _reseal_prediction_manifest(fixture):
    root = fixture.output / "predictions"
    manifest = bundle.read_json(root / "manifest.json")
    for name, record in manifest["artifacts"].items():
        record["file_sha256"] = bundle.file_hash(root / name)
        record["bytes"] = (root / name).stat().st_size
    _replace_json(root / "manifest.json", manifest, "manifest_sha256")


def test_owned_partial_resume_and_committed_reuse_preserve_existing_bytes(partial_bundle):
    f = partial_bundle
    before = _snapshot(f.output / "predictions/models")
    result = _build(f)
    assert result["status"] == "built" and result["new_fits"] == 0
    assert _snapshot(f.output / "predictions/models") == before
    sealed = _snapshot(f.output)
    assert bundle.validate_predictions(f.output)["status"] == "validated"
    assert _build(f)["status"] == "reused_valid"
    assert _snapshot(f.output) == sealed
    curves = pq.read_table(f.output / "predictions/training_curves.parquet").to_pandas()
    assert len(curves) == 1200
    assert curves.groupby("training_seed").epoch.apply(list).tolist() == [
        list(range(1, 301))
    ] * 4


@pytest.mark.parametrize("problem", [
    "record_hash", "state_hash", "wrong_seed", "wrong_parameters", "wrong_parent",
    "missing_epoch", "epoch_order", "nonfinite_gradient_flag", "negative_nll",
    "accuracy_above_one",
])
def test_invalid_partial_model_records_fail_closed(partial_bundle, problem):
    f = partial_bundle
    path = f.output / "predictions/models" / f"{bundle.capacity._stem(f.spec, 1)}.json"
    record = bundle.read_json(path)
    if problem == "record_hash":
        record["record_sha256"] = "0" * 64
    elif problem == "state_hash":
        record["state"]["state_sha256"] = "0" * 64
    elif problem == "wrong_seed":
        record["state"]["training_seed"] = 2
    elif problem == "wrong_parameters":
        record["state"]["parameters"]["dropout"] = 0.2
    elif problem == "wrong_parent":
        record["declaration_sha256"] = "0" * 64
    elif problem == "missing_epoch":
        record["diagnostics"]["history"].pop()
    elif problem == "epoch_order":
        record["diagnostics"]["history"][0]["epoch"] = 2
    elif problem == "nonfinite_gradient_flag":
        record["diagnostics"]["history"][0]["gradients_finite"] = False
    elif problem == "negative_nll":
        record["diagnostics"]["history"][0]["train_nll"] = -0.1
    else:
        record["diagnostics"]["history"][0]["train_accuracy"] = 1.1
    _replace_json(path, record, None if problem == "record_hash" else "record_sha256")
    before = _snapshot(f.output / "predictions/models")
    with pytest.raises(ValueError):
        _build(f)
    assert _snapshot(f.output / "predictions/models") == before
    assert not (f.output / "predictions/manifest.json").exists()


@pytest.mark.parametrize("target", [
    "declaration.json", "predictions/manifest.json",
    "predictions/validation_features.parquet",
])
def test_sealed_file_tampering_is_rejected(partial_bundle, target):
    f = partial_bundle
    _build(f)
    path = f.output / target
    if target.endswith(".json"):
        record = bundle.read_json(path)
        key = "declaration_sha256" if target == "declaration.json" else "manifest_sha256"
        record[key] = "0" * 64
        _replace_json(path, record)
    else:
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        bundle.validate_predictions(f.output)


@pytest.mark.parametrize("problem", ["nll", "accuracy", "training_curve"])
def test_replay_rejects_resealed_final_epoch_or_curve_mismatch(partial_bundle, problem):
    f = partial_bundle
    _build(f)
    if problem == "training_curve":
        path = f.output / "predictions/training_curves.parquet"
        frame = pq.read_table(path).to_pandas()
        frame.loc[0, "train_nll"] += 0.1
        path.chmod(0o644)
        path.unlink()
        bundle.write_bytes(path, bundle.table_bytes(bundle.capacity._table(frame)))
    else:
        path = f.output / "predictions/models" / f"{bundle.capacity._stem(f.spec, 1)}.json"
        record = bundle.read_json(path)
        key = "train_nll" if problem == "nll" else "train_accuracy"
        record["diagnostics"]["history"][-1][key] += 0.1
        _replace_json(path, record, "record_sha256")
    _reseal_prediction_manifest(f)
    with pytest.raises(ValueError, match="diagnostics do not reproduce|table values mismatch"):
        bundle.validate_predictions(f.output)


def test_parent_hash_change_blocks_publication(partial_bundle):
    f = partial_bundle
    (f.protocol / "frozen.txt").write_text("changed parent")
    with pytest.raises(ValueError, match="parent or code changed"):
        _build(f)
    assert not (f.output / "predictions/manifest.json").exists()


@pytest.mark.parametrize("kind", ["unknown", "symlink"])
def test_unknown_or_symlink_partial_artifact_is_preserved_and_rejected(partial_bundle, kind):
    f = partial_bundle
    path = f.output / "predictions/user-owned.json"
    if kind == "unknown":
        path.write_text("user-owned")
    else:
        path.symlink_to(f.protocol / "frozen.txt")
    with pytest.raises(ValueError, match="allowlist|symlink"):
        _build(f)
    assert path.exists()
    assert path.is_symlink() if kind == "symlink" else path.read_text() == "user-owned"
    assert not (f.output / "predictions/manifest.json").exists()


@pytest.mark.parametrize("which", ["protocol", "capacity"])
@pytest.mark.parametrize("alias", [False, True])
def test_frozen_root_containment_rejected_before_lock_or_import(tmp_path, which, alias):
    protocol, capacity = tmp_path / "protocol", tmp_path / "capacity"
    protocol.mkdir()
    capacity.mkdir()
    frozen = protocol if which == "protocol" else capacity
    target = frozen
    if alias:
        target = tmp_path / "alias"
        target.symlink_to(frozen, target_is_directory=True)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="outside frozen protocol/capacity"):
        bundle.build_predictions(protocol, target / "new-output", tmp_path / "missing.json", capacity)
    assert _snapshot(tmp_path) == before
    assert not (frozen / "new-output").exists()
    assert not (frozen / ".new-output.lock").exists()


def test_unowned_output_is_not_adopted(tmp_path):
    output = tmp_path / "unowned"
    output.mkdir()
    path = output / "user.txt"
    path.write_text("keep me")
    before = _snapshot(output)
    with pytest.raises(ValueError, match="unowned MLP"):
        bundle.build_predictions(tmp_path / "protocol", output, tmp_path / "missing.json", tmp_path / "capacity")
    assert _snapshot(output) == before


def test_fresh_import_and_validation_boundary_are_fitting_free():
    code = """
import sys
from multimodal import geo_helpfulness_router_mlp as bundle
bundle._fit_incapable()
assert not any(name.split('.')[0] in {'sklearn','scipy','torch','xgboost'} for name in sys.modules)
sys.modules['multimodal.geo_helpfulness_router_mlp_fit'] = object()
try:
    bundle._fit_incapable()
except ValueError:
    pass
else:
    raise AssertionError('MLP fitting-capable process accepted')
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

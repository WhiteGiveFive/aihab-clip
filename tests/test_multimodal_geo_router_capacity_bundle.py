"""Synthetic publication/recovery checks for capacity orchestration."""

from __future__ import annotations

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

from multimodal import geo_helpfulness_router_capacity as bundle
from multimodal import geo_helpfulness_router_capacity_state as portable
from multimodal import geo_helpfulness_router_candidate_state as legacy


def _model(spec, seed):
    indices = legacy.variant_feature_indices(spec["variant"])
    old = legacy.seal_candidate_state(
        {
            "schema_version": legacy.SCHEMA_VERSION,
            "protocol_id": "protocol_v1",
            "training_seed": seed,
            "variant_id": spec["variant"],
            "C": 1.0,
            "C_hex": (1.0).hex(),
            "target_order": list(legacy.TARGET_ORDER),
            "classes": [0, 1, 2, 3],
            "fit_role": "development_train_oof",
            "fit_row_count": 3378,
            "fit_row_identity_sha256": "a" * 64,
            "study_spec_sha256": "b" * 64,
            "dataset_manifest_sha256": "c" * 64,
            "feature_input_names": bundle.FEATURES,
            "feature_indices": indices,
            "feature_names": [bundle.FEATURES[i] for i in indices],
            "fit_parameters": dict(legacy.FROZEN_FIT_PARAMETERS, C=1.0),
            "implementation": dict(legacy.IMPLEMENTATION),
            "coef": np.zeros((4, len(indices)), dtype=np.float64).tolist(),
            "intercept": [0.0] * 4,
            "n_iter": [1],
        }
    )
    return portable.seal_state(
        {
            "schema_version": portable.SCHEMA_VERSION,
            **spec,
            "training_seed": seed,
            "feature_input_names": bundle.FEATURES,
            "feature_indices": indices,
            "classes": [0, 1, 2, 3],
            "target_order": list(legacy.TARGET_ORDER),
            "fit_role": "development_train_oof",
            "fit_row_count": 3378,
            "implementation": {
                "library": "sklearn",
                "version": "1.5.2",
                "split_input_dtype": "float64",
                "threadpool_limit": 1,
            },
            "model": old,
        }
    )


@pytest.fixture
def partial_bundle(tmp_path, monkeypatch):
    output, protocol = tmp_path / "output", tmp_path / "protocol"
    protocol.mkdir()
    parent = protocol / "frozen.txt"
    parent.write_text("synthetic frozen parent\n")
    spec = {
        "config_id": "synthetic_lr",
        "family": "lr",
        "variant": "ablation_numeric25",
        "parameters": {"C": 1.0},
    }
    config = {"study_id": bundle.STUDY_ID, "configurations": [spec]}
    config_path = tmp_path / "configuration.json"
    bundle.write_json(config_path, config)
    declaration = bundle._seal(
        {
            "schema_version": "geo_helpfulness.router_capacity_declaration.v1",
            "study_id": bundle.STUDY_ID,
            "configuration": config,
            "protocol_root": str(protocol),
            "parent_file_sha256": {
                str(parent): bundle.file_hash(parent),
                str(config_path): bundle.file_hash(config_path),
            },
            "score_files": sorted(bundle.SCORE_FILES),
        },
        "declaration_sha256",
    )
    bundle.write_json(output / "declaration.json", declaration)
    inputs, validation_frames = {}, []
    for seed in range(1, 5):
        x = np.zeros((4, 727), dtype=np.float64)
        identity = pd.DataFrame(
            {
                "row_uid": [f"image-{i}" for i in range(4)],
                "plot_idx": [f"plot-{i}" for i in range(4)],
                "training_seed": np.full(4, seed, dtype=np.int8),
            }
        )
        validation_frames.append(
            pd.concat([identity, pd.DataFrame(x, columns=bundle.FEATURES)], axis=1)
        )
        inputs[seed] = {
            "X_train": x,
            "X_validation": x.copy(),
            "train_identity": identity,
            "validation_identity": identity.copy(),
            "y_train": np.arange(4),
        }
        record = bundle._seal(
            {
                "declaration_sha256": declaration["declaration_sha256"],
                "state": _model(spec, seed),
                "diagnostics": {"synthetic": True},
            },
            "record_sha256",
        )
        bundle.write_json(
            output / "predictions/models" / f"{bundle._stem(spec, seed)}.json", record
        )
    complete_inputs = {
        "seed_inputs": inputs,
        "validation": pd.concat(validation_frames, ignore_index=True),
    }
    monkeypatch.setitem(
        sys.modules,
        "multimodal.geo_helpfulness_router_capacity_inputs",
        SimpleNamespace(load_capacity_inputs=lambda *args: complete_inputs),
    )

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("owned partial recovery must not refit any existing model")

    monkeypatch.setitem(
        sys.modules,
        "multimodal.geo_helpfulness_router_capacity_fit",
        SimpleNamespace(fit_candidate=forbidden_fit),
    )
    monkeypatch.setattr(bundle, "_replay_inputs", lambda *args: inputs)
    return output, protocol, config_path, declaration, spec, inputs


def _snapshot(root):
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_owned_partial_models_publish_then_reuse_without_refitting(partial_bundle):
    output, protocol, config_path, _, _, _ = partial_bundle
    models_before = _snapshot(output / "predictions/models")
    result = bundle.build_predictions(protocol, output, config_path)
    assert result["status"] == "built" and result["new_fits"] == 0
    assert _snapshot(output / "predictions/models") == models_before
    before = _snapshot(output)
    assert bundle.validate_predictions(output)["status"] == "validated"
    assert (
        bundle.build_predictions(protocol, output, config_path)["status"]
        == "reused_valid"
    )
    assert _snapshot(output) == before


def test_prediction_table_keeps_parent_identity_dtype_and_exact_policy_score():
    identity = pd.DataFrame(
        {
            "row_uid": ["a", "b"],
            "plot_idx": ["p", "q"],
            "training_seed": np.asarray([1, 1], dtype=np.int8),
        }
    )
    probabilities = np.asarray(
        [[0.4, 0.1, 0.2, 0.3], [0.1, 0.6, 0.2, 0.1]], dtype=np.float64
    )
    spec = {"config_id": "fixture", "family": "lr", "variant": "ablation_numeric25"}
    table = bundle._prediction_table(identity, probabilities, spec)
    assert table.schema == bundle.PREDICTION_SCHEMA
    assert all(not field.nullable for field in table.schema)
    restored = table.to_pandas()
    assert identity.equals(restored[bundle.IDENTITY])
    assert np.array_equal(
        restored.score.to_numpy(), probabilities[:, 0] - probabilities[:, 1]
    )
    for bad in (probabilities.astype(np.float32), probabilities * 2):
        with pytest.raises(ValueError, match="probabilities"):
            bundle._prediction_table(identity, bad, spec)


def _transform_fixture():
    numeric = list(legacy.NUMERIC_INPUT_COLUMNS)
    frame = pd.DataFrame(np.zeros((3, 25), dtype=np.float64), columns=numeric)
    vocabularies = {}
    for name, count in legacy.CATEGORICAL_FEATURE_SIZES:
        vocabularies[name] = list(range(count))
        frame[name] = np.asarray([0, 1, 2], dtype=np.int64)
    state = {
        "output_columns": bundle.FEATURES,
        "numeric_input_columns": numeric,
        "mean": [0.0] * 25,
        "scale": [1.0] * 25,
        "categorical_vocabularies": vocabularies,
    }
    return frame, state


def test_pure_transform_preserves_full727_column_order():
    frame, state = _transform_fixture()
    result = bundle._pure_transform(frame, state)
    assert result.shape == (3, 727) and result.dtype == np.float64
    assert np.array_equal(result[:, :25], np.zeros((3, 25)))
    offset = 25
    for _, count in legacy.CATEGORICAL_FEATURE_SIZES:
        assert np.array_equal(result[np.arange(3), offset + np.arange(3)], np.ones(3))
        assert np.array_equal(
            result[:, offset : offset + count].sum(axis=1), np.ones(3)
        )
        offset += count


@pytest.mark.parametrize(
    "problem", ["fractional_category", "bad_scale", "numeric_order", "vocab_size"]
)
def test_pure_transform_rejects_invalid_frozen_semantics(problem):
    frame, state = _transform_fixture()
    if problem == "fractional_category":
        frame["image_pred"] = [0.0, 1.9, 2.0]
    elif problem == "bad_scale":
        state["scale"][0] = -1
    elif problem == "numeric_order":
        state["numeric_input_columns"] = list(reversed(state["numeric_input_columns"]))
    else:
        state["categorical_vocabularies"]["image_pred"] = list(range(17))
        state["categorical_vocabularies"]["geo_pred"] = list(range(19))
    with pytest.raises(ValueError):
        bundle._pure_transform(frame, state)


@pytest.mark.parametrize(
    "problem", ["state_hash", "wrong_seed", "wrong_parameters", "wrong_parent"]
)
def test_owned_partial_model_corruption_is_rejected(partial_bundle, problem):
    output, protocol, config_path, declaration, spec, _ = partial_bundle
    path = output / "predictions/models" / f"{bundle._stem(spec, 1)}.json"
    record = bundle.read_json(path)
    if problem == "state_hash":
        record["state"]["state_sha256"] = "0" * 64
    elif problem == "wrong_seed":
        record["state"]["training_seed"] = 2
    elif problem == "wrong_parameters":
        record["state"]["parameters"]["C"] = 10.0
    else:
        record["declaration_sha256"] = "0" * 64
    record.pop("record_sha256")
    record = bundle._seal(record, "record_sha256")
    path.chmod(0o644)
    path.unlink()
    bundle.write_json(path, record)
    with pytest.raises(ValueError):
        bundle.build_predictions(protocol, output, config_path)
    assert not (output / "predictions/manifest.json").exists()


@pytest.mark.parametrize(
    "target",
    [
        "declaration.json",
        "predictions/manifest.json",
        "predictions/validation_features.parquet",
    ],
)
def test_sealed_bundle_tampering_fails_closed(partial_bundle, target):
    output, protocol, config_path, _, _, _ = partial_bundle
    bundle.build_predictions(protocol, output, config_path)
    path = output / target
    path.chmod(0o644)
    if target.endswith(".json"):
        record = bundle.read_json(path)
        key = (
            "declaration_sha256" if target == "declaration.json" else "manifest_sha256"
        )
        record[key] = "0" * 64
        path.unlink()
        bundle.write_json(path, record)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        bundle.validate_predictions(output)


def test_parent_hash_change_prevents_publication(partial_bundle):
    output, protocol, config_path, _, _, _ = partial_bundle
    (protocol / "frozen.txt").write_text("changed parent")
    with pytest.raises(ValueError, match="parent or code changed"):
        bundle.build_predictions(protocol, output, config_path)
    assert not (output / "predictions/manifest.json").exists()


def test_unknown_file_and_symlink_are_rejected_before_publication(partial_bundle):
    output, protocol, config_path, _, _, _ = partial_bundle
    unknown = output / "predictions/unowned.json"
    unknown.write_text("user-owned")
    with pytest.raises(ValueError, match="allowlist"):
        bundle.build_predictions(protocol, output, config_path)
    assert unknown.read_text() == "user-owned"


def test_output_ancestor_symlink_cannot_bypass_frozen_protocol_containment(tmp_path):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(protocol, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the frozen protocol"):
        bundle.build_predictions(
            protocol, alias / "unwanted_new_output", tmp_path / "missing_config.json"
        )
    assert not (protocol / "unwanted_new_output").exists()


def test_orchestrator_import_and_validation_boundary_are_fitting_free():
    code = """
import sys
from multimodal import geo_helpfulness_router_capacity as bundle
bundle._fit_incapable()
assert not any(name.split('.')[0] in {'sklearn','scipy','torch','xgboost'} for name in sys.modules)
sys.modules['multimodal.geo_helpfulness_router_capacity_fit'] = object()
try:
    bundle._fit_incapable()
except ValueError:
    pass
else:
    raise AssertionError('fit-capable process accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr

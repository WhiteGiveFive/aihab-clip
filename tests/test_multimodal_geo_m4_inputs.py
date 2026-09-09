"""Synthetic-only checks for final M4 label-blind comparator preparation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_router_m4_common as common
from multimodal import geo_helpfulness_router_m4_inputs as inputs_module
from multimodal import geo_helpfulness_router_numeric as numeric
from multimodal.geo_helpfulness_targets_features import build_router_feature_frame


MODES = (("image_only", "image"), ("geo_only", "geo"), ("raw_concat", "raw"))


def _softmax(logits, temperature=1.0):
    weights = np.exp((logits - logits.max(axis=1, keepdims=True)) / temperature)
    return weights / weights.sum(axis=1, keepdims=True)


def _synthetic_temperature(seed, mode, mode_index):
    theta = 0.1 * seed + 0.2 * mode_index
    return {
        "schema_version": numeric.TEMPERATURE_SCHEMA_VERSION,
        "seed": seed,
        "mode": mode,
        "fit_role": "development_train_oof",
        "fit_row_count": 3378,
        "fit_row_identity_sha256": "a" * 64,
        "class_count": 18,
        "weighting": "equal_per_image",
        "calibration_spec": copy.deepcopy(numeric.CALIBRATION_SPEC),
        "safeguards": copy.deepcopy(numeric.SAFEGUARDS),
        "theta": theta,
        "temperature": math.exp(theta),
        "native_nll": 2.0,
        "calibrated_nll": 1.5,
        "optimizer": {
            "success": True,
            "status": 0,
            "message": "synthetic saved state; never fitted in this test",
            "nfev": 8,
            "nit": None,
        },
    }


@pytest.fixture
def derivation(monkeypatch):
    rng = np.random.default_rng(312)
    frames, temperatures, expected = [], {}, {}
    for seed in range(1, 5):
        frame = pd.DataFrame(
            {
                "protocol_id": ["protocol_v1"] * 822,
                "row_uid": [f"{index:064x}" for index in range(822)],
                "plot_idx": [f"plot_{index % 325}" for index in range(822)],
                "training_seed": np.full(822, seed, dtype=np.int8),
                # Poison columns prove derivation explicitly projects its inputs.
                "label_id_dense": [-999] * 822,
                "target_state": ["DO_NOT_READ"] * 822,
            }
        )
        expected[seed] = {}
        for mode_index, (mode, prefix) in enumerate(MODES):
            state = _synthetic_temperature(seed, mode, mode_index)
            temperatures[seed, mode] = state
            logits = rng.normal(size=(822, 18)).astype(np.float64)
            logits[0] = 0.0
            logits[1, :2], logits[1, 2:] = 10.0, -10.0
            native = _softmax(logits)
            calibrated = _softmax(logits, state["temperature"])
            frame[f"{prefix}_logits"] = list(logits)
            frame[f"{prefix}_pred"] = logits.argmax(axis=1)
            frame[f"{prefix}_prob_native_t1"] = list(native)
            expected[seed][prefix] = (logits, native, calibrated)
        frames.append(frame)
    outputs = pd.concat(frames).sample(frac=1, random_state=87).reset_index(drop=True)
    calls = []
    apply = numeric.apply_expert_temperature

    def record_apply(logits, state):
        calls.append((state["seed"], state["mode"]))
        expected_logits = expected[state["seed"]][dict(MODES)[state["mode"]]][0]
        np.testing.assert_array_equal(logits, expected_logits)
        return apply(logits, state)

    def no_fit(*args, **kwargs):
        raise AssertionError("comparator preparation must never fit")

    monkeypatch.setattr(inputs_module, "apply_expert_temperature", record_apply)
    monkeypatch.setattr(numeric, "fit_expert_temperature", no_fit)
    monkeypatch.setattr(numeric, "fit_router_feature_transform", no_fit)
    monkeypatch.setattr(numeric, "minimize_scalar", no_fit)
    return SimpleNamespace(
        inputs={"temperatures": temperatures},
        outputs=outputs,
        expected=expected,
        calls=calls,
    )


def test_comparator_derivation_is_seedwise_sorted_apply_only_and_label_blind(
    derivation,
):
    table = inputs_module.derive_comparator_table(derivation.inputs, derivation.outputs)
    assert table.num_rows == 3288 and table.num_columns == 41
    assert table.schema.equals(inputs_module.comparator_schema(), check_metadata=True)
    assert table.schema.metadata is None
    assert all(not field.nullable for field in table.schema)
    assert derivation.calls == [
        (seed, mode) for seed in range(1, 5) for mode, _ in MODES
    ]
    frame = table.to_pandas()
    assert frame.equals(
        frame.sort_values(["row_uid", "training_seed"]).reset_index(drop=True)
    )
    assert frame.row_uid.nunique() == 822 and frame.plot_idx.nunique() == 325
    assert frame.groupby("training_seed").size().to_dict() == {
        1: 822,
        2: 822,
        3: 822,
        4: 822,
    }
    assert not {
        "target_state",
        "label_id_dense",
        "correctness",
        "oof_fold",
        "loss",
        "image_logits",
        "geo_logits",
        "raw_logits",
    }.intersection(frame.columns)
    semantic_names = list(
        build_router_feature_frame(
            *[np.full((1, 18), 1 / 18, dtype=np.float64)] * 3,
            probability_basis="scalar_temperature_calibrated",
        ).columns
    )
    assert list(frame.columns)[11:] == semantic_names
    for seed in range(1, 5):
        observed = frame.loc[frame.training_seed.eq(seed)].reset_index(drop=True)
        calibrated = []
        for _, prefix in MODES:
            _, native, cal = derivation.expected[seed][prefix]
            for basis, expected in (("calibrated", cal), ("native_t1", native)):
                values = np.stack(observed[f"{prefix}_prob_{basis}"])
                assert values.dtype == np.float64 and values.shape == (822, 18)
                np.testing.assert_array_equal(values, expected)
            calibrated.append(cal)
        semantic = build_router_feature_frame(
            *calibrated, probability_basis="scalar_temperature_calibrated"
        )
        pd.testing.assert_frame_equal(
            observed[semantic_names], semantic, check_dtype=False
        )


def test_comparator_derivation_rejects_argmax_mismatch(derivation):
    derivation.outputs.loc[0, "geo_pred"] = (
        int(derivation.outputs.loc[0, "geo_pred"]) + 1
    ) % 18
    with pytest.raises(ValueError, match="prediction differs"):
        inputs_module.derive_comparator_table(derivation.inputs, derivation.outputs)


def test_comparator_derivation_rejects_wrong_total_count(derivation, monkeypatch):
    monkeypatch.setattr(
        inputs_module, "apply_expert_temperature", numeric.apply_expert_temperature
    )
    with pytest.raises(ValueError, match="count/null mismatch"):
        inputs_module.derive_comparator_table(
            derivation.inputs, derivation.outputs.iloc[1:]
        )


def test_comparator_derivation_rejects_null_identity(derivation):
    derivation.outputs.loc[0, "plot_idx"] = None
    with pytest.raises(ValueError, match="count/null mismatch"):
        inputs_module.derive_comparator_table(derivation.inputs, derivation.outputs)


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    repo, protocol = tmp_path / "repo", tmp_path / "protocol_v1"
    for name in common.CODE_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic frozen source: {name}\n")
    conventions = common.load_conventions(common.PROJECT_ROOT)
    conventions_path = repo / common.CONVENTIONS_PATH
    conventions_path.parent.mkdir(parents=True, exist_ok=True)
    conventions_path.write_text(json.dumps(conventions))
    parent = tmp_path / "training_parent.json"
    parent.write_text("synthetic immutable training parent\n")
    feature_schema = (
        protocol / "router/targets_and_feature_contract/router_feature_schema.json"
    )
    feature_schema.parent.mkdir(parents=True)
    feature_schema.write_text("{}\n")
    feature_schema_hash = common.file_hash(feature_schema)
    old_root = protocol / "router/router_validation_v1"
    old_root.mkdir(parents=True)
    for index in range(7):
        (old_root / f"old_artifact_{index}.json").write_text(
            f"synthetic old artifact {index}\n"
        )
    for name in (
        "router_validation_preview_v2.json",
        "router_validation_preview_v2_declaration.json",
    ):
        (protocol / "router" / name).write_text(f"synthetic prior preview: {name}\n")
    root = protocol / common.COMPARATOR_DIR
    schema = pa.schema(
        [
            pa.field("row_uid", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("probability", pa.list_(pa.float64(), 2), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {"row_uid": "a", "value": 0.5, "probability": [0.25, 0.75]},
            {"row_uid": "b", "value": 0.25, "probability": [0.75, 0.25]},
        ],
        schema=schema,
    )
    calls = []
    original_read_table = pq.read_table

    def sealed(path):
        assert path == old_root
        calls.append("sealed")
        return {"manifest_sha256": "b" * 64}

    def training(repo_arg, protocol_arg):
        assert repo_arg == repo and protocol_arg == protocol
        calls.append("training")
        return {
            "parent_file_hashes": {str(parent): common.file_hash(parent)},
            "dataset_manifest": {
                "manifest_sha256": "c" * 64,
                "parent_lineage": {
                    "m3_bundle_file_sha256": {
                        "router_feature_schema.json": feature_schema_hash
                    }
                },
            },
            "candidate_manifest": {"manifest_sha256": "d" * 64},
            "assignment_file_sha256": "e" * 64,
        }

    def validation(repo_arg, protocol_arg, inputs):
        assert repo_arg == repo and protocol_arg == protocol
        assert inputs["assignment_file_sha256"] == "e" * 64
        assert (root / "declaration.json").is_file()
        declaration = common.read_json(root / "declaration.json")
        assert declaration["conventions_sha256"] == common.canonical_hash(conventions)
        calls.append("validation_label_blind")
        return "synthetic expert outputs without labels"

    def derive(inputs, outputs):
        assert outputs == "synthetic expert outputs without labels"
        calls.append("derive_apply_only")
        return table

    def guarded_read_table(path, *args, **kwargs):
        assert Path(path) == root / inputs_module.TABLE_FILENAME
        calls.append("read_own_comparator")
        return original_read_table(path, *args, **kwargs)

    def no_assignment_parse(*args, **kwargs):
        raise AssertionError(
            "assignment columns must not be parsed by comparator orchestration"
        )

    monkeypatch.setattr(inputs_module, "validate_sealed_predictions", sealed)
    monkeypatch.setattr(inputs_module, "load_training_inputs", training)
    monkeypatch.setattr(inputs_module, "load_validation_inputs", validation)
    monkeypatch.setattr(inputs_module, "derive_comparator_table", derive)
    monkeypatch.setattr(pq, "read_table", guarded_read_table)
    monkeypatch.setattr(pd, "read_parquet", no_assignment_parse)
    return SimpleNamespace(
        repo=repo,
        protocol=protocol,
        root=root,
        parent=parent,
        feature_schema=feature_schema,
        old_root=old_root,
        table=table,
        calls=calls,
        conventions=conventions,
    )


def _build(workflow):
    return inputs_module.prepare_comparators(workflow.protocol, repo_root=workflow.repo)


def test_comparator_publication_pins_lineage_code_conventions_and_reuses_bytes(
    workflow,
):
    result = _build(workflow)
    assert result["status"] == "built" and result["validation_labels_read"] is False
    assert workflow.calls == [
        "sealed",
        "training",
        "validation_label_blind",
        "derive_apply_only",
    ]
    assert {path.name for path in workflow.root.iterdir()} == inputs_module.FILES
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in workflow.root.iterdir()
    }
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in workflow.root.iterdir())
    manifest = common.validate_manifest(
        workflow.root, schema_version=inputs_module.SCHEMA_VERSION
    )
    assert manifest["row_count"] == 3288 and manifest["rows_per_seed"] == 822
    assert manifest["plots_per_seed"] == 325
    assert manifest["temperature_transform_router_fits"] == 0
    assert manifest["assignment_file_sha256"] == "e" * 64
    assert manifest["validation_labels_read"] is False
    assert manifest["prediction_manifest_sha256"] == "b" * 64
    assert manifest["parent_file_sha256"][str(workflow.parent)] == common.file_hash(
        workflow.parent
    )
    assert manifest["parent_file_sha256"][
        str(workflow.feature_schema)
    ] == common.file_hash(workflow.feature_schema)
    for path, digest in common.code_snapshot(workflow.repo).items():
        assert manifest["parent_file_sha256"][path] == digest
    declaration = common.read_json(workflow.root / "declaration.json")
    assert declaration["conventions"] == workflow.conventions
    assert declaration["conventions_sha256"] == common.canonical_hash(
        workflow.conventions
    )
    assert declaration["candidate_manifest_sha256"] == "d" * 64
    assert declaration["dataset_manifest_sha256"] == "c" * 64
    assert set(declaration["code_file_sha256"]) == set(common.CODE_FILES) | {
        common.CONVENTIONS_PATH
    }
    assert manifest["declaration_sha256"] == common.canonical_hash(declaration)
    workflow.calls.clear()
    repeated = _build(workflow)
    assert repeated == {**result, "status": "reused_valid"}
    assert workflow.calls == [
        "sealed",
        "training",
        "validation_label_blind",
        "derive_apply_only",
        "read_own_comparator",
    ]
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in workflow.root.iterdir()
    }


@pytest.mark.parametrize(
    "damage", ["table", "declaration", "parent", "code", "conventions", "manifest"]
)
def test_committed_tampering_fails_before_any_loader(workflow, damage):
    _build(workflow)
    paths = {
        "table": workflow.root / inputs_module.TABLE_FILENAME,
        "declaration": workflow.root / "declaration.json",
        "parent": workflow.parent,
        "code": workflow.repo / common.CODE_FILES[0],
        "conventions": workflow.repo / common.CONVENTIONS_PATH,
        "manifest": workflow.root / "manifest.json",
    }
    path = paths[damage]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"tampered")
    workflow.calls.clear()
    with pytest.raises(ValueError):
        _build(workflow)
    assert workflow.calls == []


def test_parent_mutation_during_derivation_prevents_table_publication(
    workflow, monkeypatch
):
    def mutated(inputs, outputs):
        workflow.parent.write_text("changed while deriving")
        return workflow.table

    monkeypatch.setattr(inputs_module, "derive_comparator_table", mutated)
    with pytest.raises(ValueError, match="frozen parent or code changed"):
        _build(workflow)
    assert (workflow.root / "declaration.json").is_file()
    assert not (workflow.root / inputs_module.TABLE_FILENAME).exists()
    assert not (workflow.root / "manifest.json").exists()


@pytest.mark.parametrize(
    "entry",
    [
        "unknown_file",
        "unknown_directory",
        "orphan_table",
        "root_symlink",
        "declaration_symlink",
    ],
)
def test_unknown_or_unowned_partial_paths_are_never_removed(workflow, entry):
    if entry == "root_symlink":
        target = workflow.protocol / "elsewhere"
        target.mkdir()
        workflow.root.symlink_to(target, target_is_directory=True)
        preserved = workflow.root
    else:
        workflow.root.mkdir()
        if entry == "unknown_directory":
            preserved = workflow.root / "unknown"
            preserved.mkdir()
        elif entry == "declaration_symlink":
            preserved = workflow.root / "declaration.json"
            preserved.symlink_to(workflow.parent)
        else:
            name = (
                inputs_module.TABLE_FILENAME
                if entry == "orphan_table"
                else "unknown.txt"
            )
            preserved = workflow.root / name
            preserved.write_text("unowned bytes")
    with pytest.raises(ValueError):
        _build(workflow)
    assert preserved.exists() or preserved.is_symlink()
    assert "validation_label_blind" not in workflow.calls


def test_lock_contention_hard_fails_before_loaders(workflow):
    with common.exclusive_lock(workflow.root):
        with pytest.raises(ValueError, match="another M4 process"):
            _build(workflow)
    assert workflow.calls == []


def test_lock_symlink_is_rejected_without_changing_target(workflow):
    lock = workflow.root.parent / f".{workflow.root.name}.lock"
    lock.symlink_to(workflow.parent)
    before = workflow.parent.read_bytes()
    with pytest.raises(OSError):
        _build(workflow)
    assert workflow.parent.read_bytes() == before and lock.is_symlink()
    assert workflow.calls == []


@pytest.mark.parametrize("stage", ["table", "manifest"])
def test_interrupted_publication_recovers_owned_outputs_without_changing_bytes(
    workflow, monkeypatch, stage
):
    original = inputs_module.write_bytes if stage == "table" else common.write_json

    def interrupted(path, content):
        if path.name == (
            inputs_module.TABLE_FILENAME if stage == "table" else "manifest.json"
        ):
            raise RuntimeError("synthetic interruption")
        return original(path, content)

    if stage == "table":
        monkeypatch.setattr(inputs_module, "write_bytes", interrupted)
    else:
        monkeypatch.setattr(common, "write_json", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _build(workflow)
    assert not (workflow.root / "manifest.json").exists()
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in workflow.root.iterdir()
    }
    assert "declaration.json" in before
    if stage == "table":
        monkeypatch.setattr(inputs_module, "write_bytes", original)
    else:
        monkeypatch.setattr(common, "write_json", original)
    assert _build(workflow)["status"] == "built"
    for name, saved in before.items():
        path = workflow.root / name
        assert (path.read_bytes(), path.stat().st_mtime_ns) == saved


def test_partial_declaration_code_change_is_not_overwritten(workflow, monkeypatch):
    original = inputs_module.load_validation_inputs

    def interrupted(*args):
        raise RuntimeError("stop")

    monkeypatch.setattr(inputs_module, "load_validation_inputs", interrupted)
    with pytest.raises(RuntimeError, match="stop"):
        _build(workflow)
    declaration = workflow.root / "declaration.json"
    before = declaration.read_bytes()
    (workflow.repo / common.CODE_FILES[0]).write_text("changed implementation")
    monkeypatch.setattr(inputs_module, "load_validation_inputs", original)
    workflow.calls.clear()
    with pytest.raises(ValueError, match="immutable artifact differs"):
        _build(workflow)
    assert declaration.read_bytes() == before
    assert "validation_label_blind" not in workflow.calls


def test_reuse_reconstructs_values_and_rejects_changed_output(workflow, monkeypatch):
    _build(workflow)
    before = (workflow.root / inputs_module.TABLE_FILENAME).read_bytes()
    changed = workflow.table.set_column(
        1, workflow.table.schema.field("value"), pa.array([0.9, 0.25])
    )
    monkeypatch.setattr(inputs_module, "derive_comparator_table", lambda *args: changed)
    with pytest.raises(ValueError, match="reconstructed table values mismatch"):
        _build(workflow)
    assert (workflow.root / inputs_module.TABLE_FILENAME).read_bytes() == before


def test_common_atomic_writer_reuses_exact_bytes_and_never_overwrites(tmp_path):
    path = tmp_path / "artifact.json"
    common.write_bytes(path, b"first\n")
    before = path.stat().st_mtime_ns
    common.write_bytes(path, b"first\n")
    assert path.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="immutable artifact differs"):
        common.write_bytes(path, b"other\n")
    assert path.read_bytes() == b"first\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["artifact.json"]


def test_common_atomic_writer_cleans_owned_temporary_on_link_failure(
    tmp_path, monkeypatch
):
    def interrupted(*args, **kwargs):
        raise OSError("synthetic link failure")

    monkeypatch.setattr(common.os, "link", interrupted)
    with pytest.raises(OSError, match="synthetic link failure"):
        common.write_bytes(tmp_path / "artifact", b"owned content")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', "[1,2]"]
)
def test_common_json_reader_rejects_ambiguous_or_nonstandard_values(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        common.read_json(path)


def test_common_manifest_detects_extra_files_and_bad_self_hash(workflow):
    _build(workflow)
    extra = workflow.root / "extra"
    extra.write_text("unknown")
    with pytest.raises(ValueError, match="allowlist"):
        common.validate_manifest(
            workflow.root, schema_version=inputs_module.SCHEMA_VERSION
        )
    manifest = common.read_json(workflow.root / "manifest.json")
    manifest["row_count"] += 1
    with pytest.raises(ValueError, match="invalid manifest_sha256"):
        common.check_self_hash(manifest)


def test_common_tables_enforce_float_tolerance_schema_and_vector_values(workflow):
    table = workflow.table
    near = table.set_column(
        1, table.schema.field("value"), pa.array([0.5 + 1e-13, 0.25])
    )
    common.compare_tables(near, table)
    vector_change = table.set_column(
        2,
        table.schema.field("probability"),
        pa.array([[0.5, 0.5], [0.75, 0.25]], type=pa.list_(pa.float64(), 2)),
    )
    with pytest.raises(ValueError, match="values mismatch: probability"):
        common.compare_tables(vector_change, table)
    with pytest.raises(ValueError, match="schema/count"):
        common.compare_tables(table.slice(1), table)
    with pytest.raises(ValueError, match="schema/count"):
        common.compare_tables(
            table.rename_columns(["other", "value", "probability"]), table
        )
    assert (
        common.canonical_hash({"b": 2, "a": 1})
        == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    )


def test_production_conventions_hash_matches_frozen_config():
    conventions = common.read_json(common.PROJECT_ROOT / common.CONVENTIONS_PATH)
    assert common.canonical_hash(conventions) == common.CONVENTIONS_SHA256
    assert common.load_conventions(common.PROJECT_ROOT) == conventions


def test_altered_conventions_are_rejected_before_validation_access(workflow):
    path = workflow.repo / common.CONVENTIONS_PATH
    changed = dict(workflow.conventions, primary_variant="ablation_numeric25")
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="predeclared v1 contract"):
        _build(workflow)
    assert "validation_label_blind" not in workflow.calls
    assert not (workflow.root / "declaration.json").exists()


def test_m3_schema_mismatch_is_rejected_before_validation_access(workflow):
    workflow.feature_schema.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="frozen M3 feature schema changed"):
        _build(workflow)
    assert "validation_label_blind" not in workflow.calls
    assert not (workflow.root / "declaration.json").exists()

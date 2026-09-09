"""OOF-only candidate adapter and inherited snapshot tests with synthetic files."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from multimodal import geo_helpfulness_router_candidates as candidates


def test_approved_amendment_changes_only_iteration_budget():
    original = json.loads(
        (
            candidates.PROJECT_ROOT / "configs/multimodal_geo_router_candidates_v1.json"
        ).read_text()
    )
    amended = json.loads(candidates.DEFAULT_STUDY_CONFIG_PATH.read_text())
    assert amended == candidates.EXPECTED_STUDY_CONFIG
    amendment = amended.pop("optimizer_budget_amendment")
    assert amendment["previous_value"] == 1000
    assert amendment["approved_value"] == 5000
    assert amendment["scope"] == "all_48_candidates"
    for key in ("schema_version", "study_id"):
        original.pop(key)
        amended.pop(key)
    assert original == amended
    expected = dict(candidates.FROZEN_PARENT_PARAMETERS, max_iter=5000)
    assert candidates.MODEL_PARAMETERS == expected
    assert candidates.FROZEN_PARENT_PARAMETERS["max_iter"] == 1000


class FrameTable:
    def __init__(self, frame):
        self.frame = frame

    @property
    def column_names(self):
        return list(self.frame.columns)

    def to_pandas(self):
        return self.frame


@pytest.fixture(scope="module")
def training_frame():
    row_ids = np.tile(np.arange(3378), 4)
    seeds = np.repeat(np.arange(1, 5), 3378)
    values = np.zeros((13512, 727), dtype=np.float64)
    values[:, 0], values[:, 1] = seeds, row_ids
    values[:, 2] = (row_ids + seeds) % 4
    columns = candidates.full_feature_columns()
    frame = pd.DataFrame(values, columns=columns)
    metadata = {
        "schema_version": "geo_helpfulness.router_targets.v1",
        "protocol_id": "protocol_v1",
        "row_uid": [f"{row:064x}" for row in row_ids],
        "plot_idx": [f"plot_{row % 1300}" for row in row_ids],
        "training_seed": seeds.astype(np.int8),
        "target_state": [
            candidates.TARGET_ORDER[value] for value in (row_ids + seeds) % 4
        ],
    }
    for name in reversed(candidates.dataset_api.m3.TARGET_COLUMNS):
        frame.insert(0, name, metadata[name])
    # Reverse order to prove the adapter canonicalizes independently per seed.
    return frame.iloc[::-1].reset_index(drop=True)


@pytest.fixture
def inputs(tmp_path, monkeypatch, training_frame):
    monkeypatch.setattr(candidates, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(candidates, "CODE_PATHS", ("training_impl.py",))
    (tmp_path / "training_impl.py").write_text("# synthetic implementation\n")
    root = tmp_path / "artifacts" / "router"
    root.mkdir(parents=True)
    protocol = tmp_path / "protocol"
    implementation = dict(candidates.FROZEN_PARENT_PARAMETERS)
    penalty = implementation.pop("penalty")
    implementation["library"] = "sklearn"
    implementation["tolerance"] = implementation.pop("tol")
    config = {
        "paths": {"protocol_root": str(protocol)},
        "router": {
            "implementation": implementation,
            "penalty": penalty,
            "regularization_grid_C": list(candidates.C_GRID),
            "target_order": list(candidates.TARGET_ORDER),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    study_path = tmp_path / "study.json"
    study_path.write_bytes(candidates._bytes(candidates.EXPECTED_STUDY_CONFIG))
    artifact_hashes = {}
    for name in candidates.dataset_api.CHILD_FILENAMES:
        path = root / name
        path.write_bytes(f"synthetic validated {name}\n".encode())
        artifact_hashes[name] = {"file_sha256": candidates.sha256_file(path)}
    manifest = {
        "protocol_id": "protocol_v1",
        "manifest_sha256": "d" * 64,
        "feature_columns": list(candidates.full_feature_columns()),
        "artifacts": artifact_hashes,
        "parent_lineage": {"config_file_sha256": candidates.sha256_file(config_path)},
    }
    (root / candidates.dataset_api.MANIFEST_FILENAME).write_bytes(
        candidates._bytes(manifest)
    )
    inherited_file = tmp_path / "inherited_parent.json"
    inherited_file.write_text("immutable parent\n")
    inherited_calls = []

    def inherited(bundle, sealed_dir):
        inherited_calls.append((bundle, sealed_dir))
        return {str(inherited_file): candidates.sha256_file(inherited_file)}

    monkeypatch.setattr(candidates, "_inherited_parent_snapshot", inherited)
    bundle = SimpleNamespace(
        root=root,
        manifest=manifest,
        dataset=FrameTable(training_frame.copy(deep=True)),
    )
    loader_calls = []

    def loader(**kwargs):
        loader_calls.append(kwargs)
        return bundle

    monkeypatch.setattr(
        candidates.dataset_api, "load_validated_router_dataset_bundle", loader
    )
    monkeypatch.setattr(
        candidates,
        "fit_router_candidate",
        lambda *args, **kwargs: pytest.fail("adapter fitted a router"),
    )
    return SimpleNamespace(
        bundle=bundle,
        config=config,
        config_path=config_path,
        study_path=study_path,
        protocol=protocol,
        loader_calls=loader_calls,
        inherited_calls=inherited_calls,
        inherited_file=inherited_file,
        kwargs={
            "config_path": config_path,
            "protocol_dir": protocol,
            "artifact_root": root.parent,
            "study_config_path": study_path,
        },
    )


def test_prepare_is_canonical_seed_specific_and_uses_public_loader(inputs):
    prepared = candidates._prepare_candidates(**inputs.kwargs)
    assert inputs.loader_calls == [
        {
            key: value
            for key, value in inputs.kwargs.items()
            if key != "study_config_path"
        }
    ]
    assert inputs.inherited_calls == [(inputs.bundle, inputs.protocol)]
    assert prepared.root == inputs.bundle.root / "router_candidates_v2"
    assert set(prepared.seeds) == {1, 2, 3, 4}
    for seed, values in prepared.seeds.items():
        assert values.matrix.shape == (3378, 727)
        assert values.matrix.dtype == np.dtype("float64")
        assert values.matrix.flags.c_contiguous
        assert values.targets.dtype == np.dtype("int64")
        assert values.row_count == 3378
        assert values.plot_count == 1300
        np.testing.assert_array_equal(values.matrix[:, 0], np.full(3378, seed))
        np.testing.assert_array_equal(values.matrix[:, 1], np.arange(3378))
        np.testing.assert_array_equal(values.targets, (np.arange(3378) + seed) % 4)
        assert values.row_identity_sha256 == candidates.canonical_sha256(
            [{"row_uid": f"{row:064x}", "training_seed": seed} for row in range(3378)]
        )
    assert len({value.row_identity_sha256 for value in prepared.seeds.values()}) == 4
    assert prepared.study["configuration"]["fit_role"] == "development_train_oof"
    assert prepared.study["parent_model_parameters"]["max_iter"] == 1000
    assert prepared.study["model_parameters"]["max_iter"] == 5000
    assert prepared.study["candidate_state_schema"].endswith(".v2")
    assert prepared.study["feature_input_names"] == candidates.full_feature_columns()
    assert prepared.study["source_access"] == {
        "fit_inputs": "development_train_oof_only",
        "validation_expert_outputs_opened": False,
        "validation_scoring_performed": False,
        "test_sources_opened": False,
        "inherited_parent_preflight_reads_full_development_assignments": True,
    }
    assert str(inputs.inherited_file) in prepared.parent_files
    candidates._check_snapshot(prepared)
    inputs.inherited_file.write_text("changed after preparation\n")
    with pytest.raises(
        candidates.RouterCandidateError, match="parent/config/code changed"
    ):
        candidates._check_snapshot(prepared)


def test_default_protocol_path_resolves_from_config(inputs):
    kwargs = dict(inputs.kwargs, protocol_dir=None)
    candidates._prepare_candidates(**kwargs)
    assert inputs.loader_calls[0]["protocol_dir"] is None
    assert inputs.inherited_calls[0][1] == inputs.protocol


def test_amendment_does_not_rewrite_or_relax_original_parent_settings(inputs):
    config = yaml.safe_load(inputs.config_path.read_text())
    config["router"]["implementation"]["max_iter"] = 5000
    inputs.config_path.write_text(yaml.safe_dump(config))
    inputs.bundle.manifest["parent_lineage"]["config_file_sha256"] = (
        candidates.sha256_file(inputs.config_path)
    )
    with pytest.raises(
        candidates.RouterCandidateError, match="frozen primary router settings"
    ):
        candidates._prepare_candidates(**inputs.kwargs)


@pytest.mark.parametrize(
    "damage",
    ["extra_label_column", "reordered_feature_columns", "missing_feature_column"],
)
def test_prepare_rejects_changed_feature_schema(inputs, damage):
    frame = inputs.bundle.dataset.frame
    if damage == "extra_label_column":
        frame["label_id_dense"] = 0
    elif damage == "missing_feature_column":
        inputs.bundle.dataset.frame = frame.iloc[:, :-1]
    else:
        columns = list(frame.columns)
        columns[-1], columns[-2] = columns[-2], columns[-1]
        inputs.bundle.dataset.frame = frame.loc[:, columns]
    with pytest.raises(candidates.RouterCandidateError, match="column order changed"):
        candidates._prepare_candidates(**inputs.kwargs)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_seed",
        "missing_row",
        "duplicate_row",
        "wrong_target",
        "missing_target_state",
        "plot_coverage",
        "nonfinite",
    ],
)
def test_prepare_rejects_invalid_seed_inputs(inputs, damage):
    frame = inputs.bundle.dataset.frame
    if damage == "missing_seed":
        frame.loc[frame.training_seed.eq(4), "training_seed"] = 3
    elif damage == "missing_row":
        inputs.bundle.dataset.frame = frame.iloc[:-1]
    elif damage == "duplicate_row":
        frame.loc[0, "row_uid"] = frame.loc[1, "row_uid"]
    elif damage == "wrong_target":
        frame.loc[0, "target_state"] = "validation_label"
    elif damage == "missing_target_state":
        frame.loc[frame.target_state.eq("harm"), "target_state"] = "rescue"
    elif damage == "plot_coverage":
        frame["plot_idx"] = "single_plot"
    else:
        frame.loc[0, candidates.full_feature_columns()[0]] = np.nan
    with pytest.raises(candidates.RouterCandidateError):
        candidates._prepare_candidates(**inputs.kwargs)


@pytest.mark.parametrize("artifact", list(candidates.dataset_api.BUNDLE_FILENAMES))
def test_post_validation_dataset_mutation_is_not_adopted(inputs, artifact):
    path = inputs.bundle.root / artifact
    if artifact == candidates.dataset_api.MANIFEST_FILENAME:
        changed = deepcopy(inputs.bundle.manifest)
        changed["manifest_sha256"] = "e" * 64
        path.write_bytes(candidates._bytes(changed))
    else:
        path.write_bytes(b"changed after validated read\n")
    with pytest.raises(
        candidates.RouterCandidateError, match="changed after validation"
    ):
        candidates._prepare_candidates(**inputs.kwargs)


def test_post_validation_config_mutation_is_not_adopted(inputs):
    # Even an inert textual edit must preserve the validated physical identity.
    with inputs.config_path.open("a") as handle:
        handle.write("\n# changed after validation\n")
    with pytest.raises(candidates.RouterCandidateError, match="config changed"):
        candidates._prepare_candidates(**inputs.kwargs)


def test_rejects_unpredeclared_study_before_loading_dataset(inputs):
    changed = deepcopy(candidates.EXPECTED_STUDY_CONFIG)
    changed["regularization_grid_C"].append(100.0)
    inputs.study_path.write_bytes(candidates._bytes(changed))
    with pytest.raises(candidates.RouterCandidateError, match="predeclared"):
        candidates._prepare_candidates(**inputs.kwargs)
    assert inputs.loader_calls == []


def _inherited_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(candidates, "PROJECT_ROOT", tmp_path)
    base = tmp_path / "artifacts"
    root = base / "router"
    sealed = tmp_path / "protocol"
    m3_root = root / "targets_and_feature_contract"
    m3_root.mkdir(parents=True)
    m3_manifest = {
        "m2_code_file_sha256": {"m2.py": "2" * 64},
        "m3_code_file_sha256": {"m3.py": "3" * 64},
    }
    m3_path = m3_root / "manifest.json"
    m3_path.write_bytes(candidates._bytes(m3_manifest))
    producers = {
        f"development_train_oof/seed_{seed}/fold_{fold}": f"{seed * 4 + fold:064x}"
        for seed in range(1, 5)
        for fold in range(4)
    }
    lineage = {
        "m1_m2_parent_artifact_hashes": {
            "development_assignments": {"file_sha256": "a" * 64},
            "resolved_protocol": {"file_sha256": "b" * 64},
            "protocol_manifest": {"file_sha256": "c" * 64},
            "development_train_oof_aggregate": {
                "file_sha256": "d" * 64,
                "manifest_file_sha256": "e" * 64,
            },
            "train_oof_producer_manifests": {"file_sha256": producers},
        },
        "m3_bundle_file_sha256": {
            "manifest.json": candidates.sha256_file(m3_path),
            "router_targets.parquet": "f" * 64,
        },
        "m4_code_file_sha256": {"m4_dataset.py": "4" * 64},
    }
    return SimpleNamespace(root=root, manifest={"parent_lineage": lineage}), sealed


def test_inherited_snapshot_covers_protocol_oof_m3_and_code(tmp_path, monkeypatch):
    dataset, sealed = _inherited_fixture(tmp_path, monkeypatch)
    snapshot = candidates._inherited_parent_snapshot(dataset, sealed)
    expected_paths = {
        sealed / "development_assignments.parquet",
        sealed / "resolved_protocol.yaml",
        sealed / "protocol_manifest.json",
        dataset.root.parent
        / "development_train_oof/development_train_oof_model_outputs.parquet",
        dataset.root.parent / "development_train_oof/aggregate_manifest.json",
        dataset.root / "targets_and_feature_contract/manifest.json",
        dataset.root / "targets_and_feature_contract/router_targets.parquet",
        tmp_path / "m2.py",
        tmp_path / "m3.py",
        tmp_path / "m4_dataset.py",
    } | {
        dataset.root.parent / relative / "manifest.json"
        for relative in dataset.manifest["parent_lineage"][
            "m1_m2_parent_artifact_hashes"
        ]["train_oof_producer_manifests"]["file_sha256"]
    }
    assert set(snapshot) == {str(path) for path in expected_paths}
    assert len(snapshot) == 26
    assert snapshot[str(tmp_path / "m2.py")] == "2" * 64
    assert snapshot[str(tmp_path / "m3.py")] == "3" * 64
    assert snapshot[str(tmp_path / "m4_dataset.py")] == "4" * 64
    assert not any("validation" in path or "locked_test" in path for path in snapshot)


def test_inherited_snapshot_rejects_m3_manifest_mutation(tmp_path, monkeypatch):
    dataset, sealed = _inherited_fixture(tmp_path, monkeypatch)
    (dataset.root / "targets_and_feature_contract/manifest.json").write_text("{}\n")
    with pytest.raises(candidates.RouterCandidateError, match="M3 manifest changed"):
        candidates._inherited_parent_snapshot(dataset, sealed)

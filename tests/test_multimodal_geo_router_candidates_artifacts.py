"""Synthetic publication/reproduction checks; never opens real OOF artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from multimodal import geo_helpfulness_router_candidates as candidates
from multimodal import geo_helpfulness_router_candidate_state as state_api


DATASET_DIGEST = "a" * 64


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    parent = tmp_path / "parent.json"
    parent.write_text('{"immutable": true}\n', encoding="utf-8")
    seeds = {}
    for seed in candidates.SEEDS:
        rng = np.random.default_rng(100 + seed)
        matrix = np.zeros((16, 727), dtype=np.float64)
        matrix[:, :25] = rng.normal(scale=0.2, size=(16, 25))
        targets = np.tile(np.arange(4, dtype=np.int64), 4)
        for offset, size in ((25, 18), (43, 18), (61, 18), (79, 324), (403, 324)):
            matrix[np.arange(16), offset + (np.arange(16) + seed) % size] = 1.0
        row_hash = candidates.canonical_sha256(
            [{"row_uid": f"{row:064x}", "training_seed": seed} for row in range(16)]
        )
        seeds[seed] = candidates.SeedInputs(matrix, targets, row_hash, 16, 4)
    study = {
        "schema_version": "geo_helpfulness.router_candidate_study.v2",
        "study_id": "router_candidates_v2",
        "protocol_id": "protocol_v1",
        "configuration": deepcopy(candidates.EXPECTED_STUDY_CONFIG),
        "model_parameters": dict(candidates.MODEL_PARAMETERS),
        "dataset_manifest_sha256": DATASET_DIGEST,
        "predeclared_before_candidate_fitting": True,
        "feature_input_names": state_api.full_feature_columns(),
        "seed_inputs": {
            str(seed): {
                "row_count": value.row_count,
                "plot_count": value.plot_count,
                "row_identity_sha256": value.row_identity_sha256,
            }
            for seed, value in seeds.items()
        },
    }
    study["study_spec_sha256"] = candidates._hash(study, "study_spec_sha256")
    value = candidates.PreparedCandidates(
        tmp_path / "router_candidates_v2",
        "protocol_v1",
        seeds,
        study,
        {str(parent): candidates.sha256_file(parent)},
    )
    monkeypatch.setattr(candidates, "_prepare_candidates", lambda **kwargs: value)
    return value


def _forbid_fit(*args, **kwargs):
    raise AssertionError("fitting must not occur in this operation")


def _hashes(root: Path):
    return {path.name: candidates.sha256_file(path) for path in root.iterdir()}


def _overwrite_json(path, payload):
    path.chmod(0o644)
    path.write_bytes(candidates._bytes(payload))


def _partial_state(prepared, spec=None):
    prepared.root.mkdir()
    candidates._publish_json(prepared.root / candidates.STUDY_FILENAME, prepared.study)
    spec = spec or candidates.candidate_specs()[0]
    state, _ = candidates._fit(prepared, *spec)
    path = prepared.root / candidates.candidate_filename(*spec)
    candidates._publish_json(path, state)
    return path, state


def test_build_seals_predeclaration_first_and_manifest_last(prepared, monkeypatch):
    publications, fits = [], []
    original_publish, original_fit = candidates._publish_json, candidates._fit

    def publish(path, payload):
        publications.append(path.name)
        if path.name == candidates.MANIFEST_FILENAME:
            assert set(
                p.name for p in prepared.root.iterdir()
            ) == candidates._allowed_files() - {path.name}
        return original_publish(path, payload)

    def fit(inputs, variant, seed, C):
        study_path = inputs.root / candidates.STUDY_FILENAME
        assert candidates._read_json(study_path) == inputs.study
        assert study_path.stat().st_mode & 0o222 == 0
        assert not (inputs.root / candidates.MANIFEST_FILENAME).exists()
        fits.append((variant, seed, C))
        return original_fit(inputs, variant, seed, C)

    monkeypatch.setattr(candidates, "_publish_json", publish)
    monkeypatch.setattr(candidates, "_fit", fit)
    result = candidates.build_router_candidate_bundle()
    assert result["status"] == "created"
    assert result["fits_this_call"] == result["candidate_count"] == 48
    assert result["resumed_candidate_count"] == 0
    assert fits == candidates.candidate_specs()
    assert publications[0] == candidates.STUDY_FILENAME
    assert publications[-2:] == [
        candidates.REPORT_FILENAME,
        candidates.MANIFEST_FILENAME,
    ]
    assert set(publications) == candidates._allowed_files()
    assert all(path.stat().st_mode & 0o222 == 0 for path in prepared.root.iterdir())
    assert not list(prepared.root.parent.glob(".router_candidates.staging-*"))
    manifest, states = state_api.load_candidate_set_for_inference(
        prepared.root,
        expected_dataset_manifest_sha256=DATASET_DIGEST,
        expected_study_spec_sha256=prepared.study["study_spec_sha256"],
    )
    assert len(states) == len(manifest["candidates"]) == 48
    assert manifest["selection_performed"] is False
    assert manifest["validation_predictions_generated"] is False
    for state in states.values():
        assert (
            state["fit_row_identity_sha256"]
            == prepared.seeds[state["training_seed"]].row_identity_sha256
        )
        assert len(state["feature_indices"]) == state_api.VARIANTS[state["variant_id"]]


def test_validate_and_build_reuse_without_fit_or_byte_changes(prepared, monkeypatch):
    candidates.build_router_candidate_bundle()
    before = _hashes(prepared.root)
    monkeypatch.setattr(candidates, "_fit", _forbid_fit)
    monkeypatch.setattr(candidates, "fit_router_candidate", _forbid_fit)
    monkeypatch.setattr(candidates, "_publish_json", _forbid_fit)
    for function in (
        candidates.validate_router_candidate_bundle,
        candidates.build_router_candidate_bundle,
    ):
        result = function()
        assert result["valid"] is True
        assert result["status"] == "reused_valid"
        assert result["fits_this_call"] == 0
        assert _hashes(prepared.root) == before


def test_explicit_reproduction_refits_all_and_preserves_bytes(prepared, monkeypatch):
    candidates.build_router_candidate_bundle()
    before = _hashes(prepared.root)
    fits = []
    original_fit = candidates._fit

    def fit(inputs, *spec):
        fits.append(spec)
        return original_fit(inputs, *spec)

    monkeypatch.setattr(candidates, "_fit", fit)
    monkeypatch.setattr(candidates, "_publish_json", _forbid_fit)
    result = candidates.reproduce_router_candidate_bundle()
    assert result["status"] == "reproduced_valid"
    assert result["fits_this_call"] == 48
    assert result["artifacts_unchanged"] is True
    assert result["max_coefficient_absolute_error"] == 0.0
    assert result["max_probability_absolute_error"] == 0.0
    assert fits == candidates.candidate_specs()
    assert _hashes(prepared.root) == before


def test_training_report_is_independent_of_ambient_native_thread_count(
    prepared, monkeypatch
):
    candidates.build_router_candidate_bundle()
    original = candidates.predict_router_probabilities
    observed_thread_counts = []

    def predict(matrix, state):
        active = [
            pool["num_threads"]
            for pool in threadpool_info()
            if pool["user_api"] == "blas"
        ]
        observed_thread_counts.extend(active)
        assert all(count == 1 for count in active)
        return original(matrix, state)

    monkeypatch.setattr(candidates, "predict_router_probabilities", predict)
    with threadpool_limits(limits=1):
        one_thread_report = candidates._report(prepared)
    with threadpool_limits(limits=4):
        four_thread_report = candidates._report(prepared)
        validation = candidates.validate_router_candidate_bundle()
    assert observed_thread_counts
    assert one_thread_report == four_thread_report
    assert validation["status"] == "reused_valid"


@pytest.mark.parametrize("coefficient_delta", [1.0, 5e-11])
def test_reproduction_detects_coefficient_or_probability_drift(
    prepared, monkeypatch, coefficient_delta
):
    candidates.build_router_candidate_bundle()
    before = _hashes(prepared.root)
    original = candidates._fit

    def changed(inputs, *spec):
        state, diagnostics = original(inputs, *spec)
        state["intercept"][0] += coefficient_delta
        return state_api.seal_candidate_state(state), diagnostics

    monkeypatch.setattr(candidates, "_fit", changed)
    with pytest.raises(candidates.RouterCandidateError, match="do not reproduce"):
        candidates.reproduce_router_candidate_bundle()
    assert _hashes(prepared.root) == before


def test_interrupted_candidate_publication_resumes_only_missing(prepared, monkeypatch):
    publish = candidates._publish_json
    candidate_publications = []

    def interrupted(path, payload):
        if "__seed_" in path.name:
            candidate_publications.append(path.name)
            if len(candidate_publications) == 3:
                raise RuntimeError("simulated interruption before third commit")
        return publish(path, payload)

    monkeypatch.setattr(candidates, "_publish_json", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        candidates.build_router_candidate_bundle()
    assert not (prepared.root / candidates.MANIFEST_FILENAME).exists()
    before = _hashes(prepared.root)
    assert len(before) == 3  # study and two committed candidates
    monkeypatch.setattr(candidates, "_publish_json", publish)
    fits = []
    original_fit = candidates._fit

    def fit(inputs, *spec):
        fits.append(spec)
        return original_fit(inputs, *spec)

    monkeypatch.setattr(candidates, "_fit", fit)
    result = candidates.build_router_candidate_bundle()
    assert result["fits_this_call"] == 46
    assert result["resumed_candidate_count"] == 2
    assert fits == candidates.candidate_specs()[2:]
    assert all(
        candidates.sha256_file(prepared.root / name) == digest
        for name, digest in before.items()
    )


@pytest.mark.parametrize(
    "artifact", [candidates.REPORT_FILENAME, candidates.MANIFEST_FILENAME]
)
def test_resume_after_all_candidates_are_committed_requires_no_fitting(
    prepared, monkeypatch, artifact
):
    publish = candidates._publish_json

    def interrupted(path, payload):
        if path.name == artifact:
            raise RuntimeError("publication interrupted")
        return publish(path, payload)

    monkeypatch.setattr(candidates, "_publish_json", interrupted)
    with pytest.raises(RuntimeError):
        candidates.build_router_candidate_bundle()
    before = _hashes(prepared.root)
    monkeypatch.setattr(candidates, "_publish_json", publish)
    monkeypatch.setattr(candidates, "_fit", _forbid_fit)
    result = candidates.build_router_candidate_bundle()
    assert result["fits_this_call"] == 0
    assert result["resumed_candidate_count"] == 48
    assert all(
        candidates.sha256_file(prepared.root / name) == digest
        for name, digest in before.items()
    )


@pytest.mark.parametrize(
    "damage",
    [
        "corrupt_state",
        "stale_state",
        "stale_study",
        "missing_study",
        "unknown",
        "child_symlink",
        "incomplete_report",
    ],
)
def test_invalid_partial_inputs_fail_before_any_new_fit(prepared, monkeypatch, damage):
    path, state = _partial_state(prepared, candidates.candidate_specs()[-1])
    if damage == "corrupt_state":
        state["intercept"][0] += 0.1
        _overwrite_json(path, state)
    elif damage == "stale_state":
        state["dataset_manifest_sha256"] = "b" * 64
        _overwrite_json(path, state_api.seal_candidate_state(state))
    elif damage == "stale_study":
        study = deepcopy(prepared.study)
        study["environment"] = "changed"
        study["study_spec_sha256"] = candidates._hash(study, "study_spec_sha256")
        _overwrite_json(prepared.root / candidates.STUDY_FILENAME, study)
    elif damage == "missing_study":
        (prepared.root / candidates.STUDY_FILENAME).unlink()
    elif damage == "unknown":
        (prepared.root / "keep_user_data.txt").write_text("owned by user")
    elif damage == "child_symlink":
        path.unlink()
        path.symlink_to(next(iter(prepared.parent_files)))
    elif damage == "incomplete_report":
        candidates._publish_json(
            prepared.root / candidates.REPORT_FILENAME, {"premature": True}
        )
    before = {p.name: p.read_bytes() for p in prepared.root.iterdir()}
    monkeypatch.setattr(candidates, "_fit", _forbid_fit)
    with pytest.raises(ValueError):
        candidates.build_router_candidate_bundle()
    assert {p.name: p.read_bytes() for p in prepared.root.iterdir()} == before


@pytest.mark.parametrize("damage", ["state", "report", "manifest", "parent"])
def test_committed_tampering_is_hard_failure_without_refitting(
    prepared, monkeypatch, damage
):
    candidates.build_router_candidate_bundle()
    if damage == "state":
        path = prepared.root / candidates.candidate_filename(
            *candidates.candidate_specs()[0]
        )
        value = candidates._read_json(path)
        value["intercept"][0] += 0.1
        _overwrite_json(path, state_api.seal_candidate_state(value))
    elif damage == "report":
        path = prepared.root / candidates.REPORT_FILENAME
        value = candidates._read_json(path)
        value["candidates"][0]["train_log_loss"] += 0.01
        _overwrite_json(path, value)
    elif damage == "manifest":
        path = prepared.root / candidates.MANIFEST_FILENAME
        value = candidates._read_json(path)
        value["selection_performed"] = True
        value["manifest_sha256"] = candidates._hash(value, "manifest_sha256")
        _overwrite_json(path, value)
    elif damage == "parent":
        Path(next(iter(prepared.parent_files))).write_text("changed")
    before = _hashes(prepared.root)
    monkeypatch.setattr(candidates, "_fit", _forbid_fit)
    for function in (
        candidates.validate_router_candidate_bundle,
        candidates.build_router_candidate_bundle,
        candidates.reproduce_router_candidate_bundle,
    ):
        with pytest.raises(ValueError):
            function()
        assert _hashes(prepared.root) == before


def test_parent_change_during_fit_prevents_candidate_publication(prepared, monkeypatch):
    original_fit = candidates._fit

    def mutate(inputs, *spec):
        result = original_fit(inputs, *spec)
        Path(next(iter(inputs.parent_files))).write_text("changed during fitting")
        return result

    monkeypatch.setattr(candidates, "_fit", mutate)
    with pytest.raises(candidates.RouterCandidateError, match="changed"):
        candidates.build_router_candidate_bundle()
    assert {p.name for p in prepared.root.iterdir()} == {candidates.STUDY_FILENAME}


def test_changed_prepared_study_rejects_existing_bundle(prepared, monkeypatch):
    _partial_state(prepared)
    study = deepcopy(prepared.study)
    study["implementation_file_sha256"] = {"code.py": "b" * 64}
    study["study_spec_sha256"] = candidates._hash(study, "study_spec_sha256")
    changed = replace(prepared, study=study)
    monkeypatch.setattr(candidates, "_prepare_candidates", lambda **kwargs: changed)
    monkeypatch.setattr(candidates, "_fit", _forbid_fit)
    with pytest.raises(candidates.RouterCandidateError, match="stale"):
        candidates.build_router_candidate_bundle()


def test_lock_contention_fails_without_fit_or_creating_bundle(prepared, monkeypatch):
    monkeypatch.setattr(candidates, "_fit", _forbid_fit)
    with candidates._workflow_lock(prepared.root):
        with pytest.raises(candidates.RouterCandidateError, match="workflow lock"):
            candidates.build_router_candidate_bundle()
    assert not prepared.root.exists()


def test_root_symlink_and_unknown_directory_rejected(prepared, monkeypatch):
    target = prepared.root.parent / "user_directory"
    target.mkdir()
    prepared.root.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(candidates, "_fit", _forbid_fit)
    with pytest.raises(candidates.RouterCandidateError, match="non-symlink"):
        candidates.build_router_candidate_bundle()
    assert list(target.iterdir()) == []


def test_publication_never_overwrites_existing_json(prepared):
    prepared.root.mkdir()
    path = prepared.root / candidates.STUDY_FILENAME
    candidates._publish_json(path, prepared.study)
    original = path.read_bytes()
    with pytest.raises(candidates.RouterCandidateError, match="already exists"):
        candidates._publish_json(path, {"overwrite": True})
    assert path.read_bytes() == original
    assert not list(prepared.root.parent.glob(".router_candidates.staging-*"))

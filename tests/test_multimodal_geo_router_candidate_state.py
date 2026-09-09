from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from multimodal import geo_helpfulness_router_candidate_state as candidate


DIGEST = "a" * 64


def state_fixture(variant="ablation_numeric25", seed=1, C=1.0):
    columns = candidate.full_feature_columns()
    indices = candidate.variant_feature_indices(variant)
    return candidate.seal_candidate_state({
        "schema_version": candidate.SCHEMA_VERSION,
        "protocol_id": "protocol_v1",
        "training_seed": seed,
        "variant_id": variant,
        "C": C,
        "C_hex": C.hex(),
        "target_order": list(candidate.TARGET_ORDER),
        "classes": [0, 1, 2, 3],
        "fit_role": "development_train_oof",
        "fit_row_count": 40,
        "fit_row_identity_sha256": DIGEST,
        "study_spec_sha256": DIGEST,
        "dataset_manifest_sha256": DIGEST,
        "feature_input_names": columns,
        "feature_indices": indices,
        "feature_names": [columns[index] for index in indices],
        "fit_parameters": dict(candidate.FROZEN_FIT_PARAMETERS, C=C),
        "implementation": dict(candidate.IMPLEMENTATION),
        "coef": np.zeros((4, len(indices)), dtype=np.float64).tolist(),
        "intercept": [0.0, 0.0, 0.0, 0.0],
        "n_iter": [1],
    })


def test_full_schema_matches_sealed_m4_schema_without_importing_parent():
    # The literal expected names make this test independent of heavy M1-M3 imports.
    columns = candidate.full_feature_columns()
    assert len(columns) == len(set(columns)) == 727
    assert columns[0] == "scaled__image_geo_agree"
    assert columns[24] == "scaled__raw_probability_at_geo_pred"
    assert columns[25] == "onehot__image_pred__0"
    assert columns[78] == "onehot__raw_pred__17"
    assert columns[79] == "onehot__image_geo_pred_pair__0"
    assert columns[402] == "onehot__image_geo_pred_pair__323"
    assert columns[403] == "onehot__geo_raw_pred_pair__0"
    assert columns[-1] == "onehot__geo_raw_pred_pair__323"
    for variant, count in candidate.VARIANTS.items():
        assert candidate.variant_feature_indices(variant) == list(range(count))
    with pytest.raises(candidate.CandidateStateError):
        candidate.variant_feature_indices("adaptive_selection")


def test_state_reseal_copy_json_roundtrip_and_hash():
    state = state_fixture()
    restored = json.loads(json.dumps(state, allow_nan=False))
    candidate.validate_candidate_state(restored)
    assert candidate.seal_candidate_state(state) == state
    assert candidate.seal_candidate_state(state) is not state
    assert candidate.candidate_state_sha256(restored) == state["state_sha256"]
    reversed_state = dict(reversed(list(state.items())))
    assert candidate.candidate_state_sha256(reversed_state) == state["state_sha256"]
    x = np.zeros((2, 727), dtype=np.float64)
    np.testing.assert_array_equal(candidate.predict_router_probabilities(x, restored), np.full((2, 4), .25))


@pytest.mark.parametrize("variant", tuple(candidate.VARIANTS))
def test_affine_probability_output_order_and_selected_columns(variant):
    state = state_fixture(variant)
    state["coef"][0][0] = 1.0
    state["coef"][1][1] = 2.0
    state["intercept"] = [0.0, 1.0, 2.0, 3.0]
    state = candidate.seal_candidate_state(state)
    x = np.zeros((3, 727), dtype=np.float64)
    x[:, :2] = [[1.0, 2.0], [-1.0, -2.0], [0.0, 0.0]]
    score = np.array([[1.0, 5.0, 2.0, 3.0], [-1.0, -3.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(candidate.decision_function(x, state), score)
    expected = np.exp(score - score.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(candidate.predict_router_probabilities(x, state), expected)
    x[:, len(state["feature_indices"]):] = 9876.0
    np.testing.assert_array_equal(candidate.decision_function(x, state), score)


def test_extreme_scores_stable_underflow_and_ties():
    state = state_fixture()
    state["intercept"] = [10000.0, 10000.0, -10000.0, -10000.0]
    state = candidate.seal_candidate_state(state)
    probability = candidate.predict_router_probabilities(np.zeros((1, 727)), state)
    np.testing.assert_array_equal(probability, [[.5, .5, 0.0, 0.0]])
    assert np.argmax(probability) == 0


@pytest.mark.parametrize("updates", [
    {"schema_version": "wrong"}, {"protocol_id": ""}, {"training_seed": True},
    {"training_seed": 5}, {"variant_id": "other"}, {"C": 1}, {"C": .2},
    {"C_hex": "wrong"}, {"target_order": list(reversed(candidate.TARGET_ORDER))},
    {"classes": [0, True, 2, 3]}, {"fit_role": "development_validation"},
    {"fit_row_count": True}, {"fit_row_count": 3}, {"fit_row_identity_sha256": "X" * 64},
    {"study_spec_sha256": "invalid"}, {"dataset_manifest_sha256": "A" * 64},
    {"feature_input_names": []}, {"feature_indices": [True] * 25},
    {"feature_names": []}, {"fit_parameters": {}}, {"implementation": {}},
    {"coef": [[0.0] * 24] * 4}, {"coef": [[False] * 25] * 4},
    {"intercept": [0.0] * 3}, {"intercept": [float("inf")] * 4},
    {"n_iter": [True]}, {"n_iter": [-1]}, {"n_iter": [5000]}, {"n_iter": []},
    {"extra_key": 0},
])
def test_invalid_state_rejected_even_when_resealed(updates):
    state = state_fixture()
    state.update(updates)
    with pytest.raises(candidate.CandidateStateError):
        candidate.seal_candidate_state(state)


def test_state_hash_tamper_and_missing_key_rejected():
    state = state_fixture()
    state["intercept"][0] = .5
    with pytest.raises(candidate.CandidateStateError, match="self-hash"):
        candidate.validate_candidate_state(state)
    del state["intercept"]
    with pytest.raises(candidate.CandidateStateError, match="keys"):
        candidate.validate_candidate_state(state)


def test_zero_iteration_stationary_optimum_allowed():
    state = state_fixture()
    state["n_iter"] = [0]
    candidate.seal_candidate_state(state)


@pytest.mark.parametrize("iterations", [1000, 1500, 4999])
def test_amended_v2_budget_accepts_iterations_above_legacy_limit(iterations):
    state = state_fixture()
    assert state["schema_version"] == "geo_helpfulness.router_candidate.v2"
    assert state["fit_parameters"]["max_iter"] == 5000
    state["n_iter"] = [iterations]
    candidate.seal_candidate_state(state)


def test_legacy_v1_state_remains_readable_and_applies_identically(tmp_path):
    state = state_fixture()
    legacy = dict(state, schema_version=candidate.LEGACY_SCHEMA_VERSION)
    legacy["fit_parameters"] = dict(state["fit_parameters"], max_iter=1000)
    legacy["n_iter"] = [999]
    legacy = candidate.seal_candidate_state(legacy)
    path = tmp_path / "original_v1_state.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    restored = candidate.read_candidate_state(path)
    assert restored == legacy
    matrix = np.zeros((2, 727), dtype=np.float64)
    np.testing.assert_array_equal(
        candidate.predict_router_probabilities(matrix, restored),
        candidate.predict_router_probabilities(matrix, state),
    )
    legacy["n_iter"] = [1000]
    with pytest.raises(candidate.CandidateStateError, match="iteration cap"):
        candidate.seal_candidate_state(legacy)


@pytest.mark.parametrize("schema,max_iter", [(candidate.SCHEMA_VERSION, 1000), (candidate.LEGACY_SCHEMA_VERSION, 5000)])
def test_schema_version_and_exact_budget_cannot_be_mixed(schema, max_iter):
    state = state_fixture()
    state["schema_version"] = schema
    state["fit_parameters"]["max_iter"] = max_iter
    with pytest.raises(candidate.CandidateStateError, match="optimizer parameters"):
        candidate.seal_candidate_state(state)


@pytest.mark.parametrize("matrix", [
    np.zeros((1, 727), dtype=np.float32), np.zeros((1, 25)), np.zeros((0, 727)),
    np.zeros(727), np.full((1, 727), np.nan), np.full((1, 727), np.inf),
    [[0.0] * 727],
])
def test_invalid_inference_input(matrix):
    with pytest.raises(candidate.CandidateStateError):
        candidate.predict_router_probabilities(matrix, state_fixture())


def test_nonfinite_affine_computation_rejected():
    state = state_fixture()
    state["coef"][0][0] = 1e308
    state = candidate.seal_candidate_state(state)
    with pytest.raises(candidate.CandidateStateError, match="nonfinite scores"):
        candidate.predict_router_probabilities(np.full((1, 727), 1e308), state)


def test_single_state_reader_parent_binding_and_strict_json(tmp_path):
    path = tmp_path / "state.json"
    state = state_fixture()
    path.write_text(json.dumps(state), encoding="utf-8")
    assert candidate.read_candidate_state(path, expected_parents={"dataset_manifest_sha256": DIGEST}) == state
    with pytest.raises(candidate.CandidateStateError, match="parent identity"):
        candidate.read_candidate_state(path, expected_parents={"study_spec_sha256": "b" * 64})
    with pytest.raises(candidate.CandidateStateError, match="unknown expected"):
        candidate.read_candidate_state(path, expected_parents={"coef": "bad"})
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(candidate.CandidateStateError, match="nonsymlink"):
        candidate.read_candidate_state(link)
    path.write_text('{"duplicate": 1, "duplicate": 2}', encoding="utf-8")
    with pytest.raises(candidate.CandidateStateError, match="duplicate"):
        candidate.read_candidate_state(path)
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(candidate.CandidateStateError, match="nonfinite"):
        candidate.read_candidate_state(path)


def write_json(path, value):
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def bundle_fixture(root, *, legacy=False):
    root.mkdir()
    study = {
        "study_id": "three_predeclared_variants",
        "schema_version": f"geo_helpfulness.router_candidate_study.v{1 if legacy else 2}",
        "model_parameters": dict(candidate.FROZEN_FIT_PARAMETERS, max_iter=1000 if legacy else 5000),
        "candidate_state_schema": candidate.LEGACY_SCHEMA_VERSION if legacy else candidate.SCHEMA_VERSION,
    }
    study["study_spec_sha256"] = candidate.canonical_sha256(study)
    manifest = {
        "schema_version": "geo_helpfulness.router_candidate_manifest.v1",
        "study_spec_sha256": study["study_spec_sha256"],
        "dataset_manifest_sha256": DIGEST,
        "candidates": {},
        "artifacts": {
            "study_spec.json": {"file_sha256": write_json(root / "study_spec.json", study)},
            "training_report.json": {"file_sha256": write_json(root / "training_report.json", {"fit_count": 48})},
        },
    }
    for variant in candidate.VARIANTS:
        for seed in candidate.TRAINING_SEEDS:
            for C in candidate.C_GRID:
                state = state_fixture(variant, seed, C)
                if legacy:
                    state["schema_version"] = candidate.LEGACY_SCHEMA_VERSION
                    state["fit_parameters"]["max_iter"] = 1000
                state["study_spec_sha256"] = manifest["study_spec_sha256"]
                state = candidate.seal_candidate_state(state)
                filename = candidate.candidate_filename(variant, seed, C)
                manifest["candidates"][filename] = {
                    "file_sha256": write_json(root / filename, state),
                    "state_sha256": state["state_sha256"],
                    "variant_id": variant, "training_seed": seed, "C_hex": C.hex(),
                }
    manifest["manifest_sha256"] = candidate.canonical_sha256(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def reseal_manifest(root, manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = candidate.canonical_sha256(payload)
    write_json(root / "manifest.json", manifest)


def test_complete_inference_bundle_loads_without_touching_parent(tmp_path):
    root = tmp_path / "candidates"
    manifest = bundle_fixture(root)
    loaded, states = candidate.load_candidate_set_for_inference(
        root, expected_dataset_manifest_sha256=DIGEST,
        expected_study_spec_sha256=manifest["study_spec_sha256"],
    )
    assert loaded == manifest
    assert len(states) == 48
    for state in states.values():
        np.testing.assert_array_equal(candidate.predict_router_probabilities(np.zeros((1, 727)), state), [[.25] * 4])


def test_complete_legacy_bundle_remains_inference_compatible(tmp_path):
    root = tmp_path / "legacy_candidates"
    bundle_fixture(root, legacy=True)
    _, states = candidate.load_candidate_set_for_inference(
        root, expected_dataset_manifest_sha256=DIGEST,
    )
    assert {state["schema_version"] for state in states.values()} == {candidate.LEGACY_SCHEMA_VERSION}
    assert {state["fit_parameters"]["max_iter"] for state in states.values()} == {1000}


def test_complete_v2_bundle_cannot_mix_in_legacy_state_even_with_refreshed_hashes(tmp_path):
    root = tmp_path / "mixed_candidates"
    manifest = bundle_fixture(root)
    filename = next(iter(manifest["candidates"]))
    state = json.loads((root / filename).read_text())
    state["schema_version"] = candidate.LEGACY_SCHEMA_VERSION
    state["fit_parameters"]["max_iter"] = 1000
    state = candidate.seal_candidate_state(state)
    record = manifest["candidates"][filename]
    record["file_sha256"] = write_json(root / filename, state)
    record["state_sha256"] = state["state_sha256"]
    reseal_manifest(root, manifest)
    with pytest.raises(candidate.CandidateStateError, match="study optimizer budget"):
        candidate.load_candidate_set_for_inference(root, expected_dataset_manifest_sha256=DIGEST)


@pytest.mark.parametrize("changed_field", ["model_parameters", "candidate_state_schema", "parent_model_parameters"])
def test_complete_bundle_requires_versioned_study_budget_consistency(tmp_path, changed_field):
    root = tmp_path / "inconsistent_study"
    manifest = bundle_fixture(root)
    study = json.loads((root / "study_spec.json").read_text())
    if changed_field == "model_parameters":
        study[changed_field]["max_iter"] = 1000
    elif changed_field == "candidate_state_schema":
        study[changed_field] = candidate.LEGACY_SCHEMA_VERSION
    else:
        study[changed_field] = dict(candidate.FROZEN_FIT_PARAMETERS)
    study["study_spec_sha256"] = candidate.canonical_sha256({
        key: value for key, value in study.items() if key != "study_spec_sha256"
    })
    manifest["study_spec_sha256"] = study["study_spec_sha256"]
    manifest["artifacts"]["study_spec.json"]["file_sha256"] = write_json(root / "study_spec.json", study)
    reseal_manifest(root, manifest)
    with pytest.raises(candidate.CandidateStateError, match="study"):
        candidate.load_candidate_set_for_inference(root, expected_dataset_manifest_sha256=DIGEST)


@pytest.mark.parametrize("tamper", ["extra", "missing", "file", "manifest", "parent", "study", "traversal", "identity", "state_parent"])
def test_bundle_tampering_rejected(tmp_path, tamper):
    root = tmp_path / "candidates"
    manifest = bundle_fixture(root)
    filename = next(iter(manifest["candidates"]))
    if tamper == "extra":
        (root / "unrecognized.json").write_text("{}")
    elif tamper == "missing":
        (root / filename).unlink()
    elif tamper == "file":
        (root / filename).write_text("{}")
    elif tamper == "manifest":
        manifest["dataset_manifest_sha256"] = "b" * 64
        write_json(root / "manifest.json", manifest)
    elif tamper == "parent":
        manifest["dataset_manifest_sha256"] = "b" * 64
        reseal_manifest(root, manifest)
    elif tamper == "study":
        manifest["artifacts"]["study_spec.json"]["file_sha256"] = write_json(root / "study_spec.json", {"changed": True})
        reseal_manifest(root, manifest)
    elif tamper == "traversal":
        manifest["candidates"]["../outside.json"] = manifest["candidates"].pop(filename)
        reseal_manifest(root, manifest)
    elif tamper == "identity":
        manifest["candidates"][filename]["training_seed"] = 2
        reseal_manifest(root, manifest)
    elif tamper == "state_parent":
        state = json.loads((root / filename).read_text())
        state["study_spec_sha256"] = "b" * 64
        state = candidate.seal_candidate_state(state)
        manifest["candidates"][filename]["state_sha256"] = state["state_sha256"]
        manifest["candidates"][filename]["file_sha256"] = write_json(root / filename, state)
        reseal_manifest(root, manifest)
    with pytest.raises(candidate.CandidateStateError):
        candidate.load_candidate_set_for_inference(root, expected_dataset_manifest_sha256=DIGEST)


def test_module_import_is_fitting_free():
    repository = Path(__file__).resolve().parents[1]
    script = (
        "import sys; import multimodal.geo_helpfulness_router_candidate_state; "
        "blocked=('sklearn','scipy','torch','pandas','pyarrow',"
        "'multimodal.geo_helpfulness_router_dataset','multimodal.geo_helpfulness_oof'); "
        "assert not any(name==p or name.startswith(p+'.') for p in blocked for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=repository, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

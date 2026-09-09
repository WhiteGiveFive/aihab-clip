"""Synthetic apply-only replay tests for the two frozen legacy families."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_legacy_router_common as common
from multimodal import geo_helpfulness_router_candidate_state as candidate


DIGEST = "a" * 64


def _state(variant, seed):
    names = candidate.full_feature_columns()
    indices = candidate.variant_feature_indices(variant)
    coef = np.zeros((4, len(indices)), dtype=np.float64)
    coef[0, 0], coef[1, 1], coef[2, 2] = .15, -.2, .05
    return candidate.seal_candidate_state({
        "schema_version": candidate.SCHEMA_VERSION,
        "protocol_id": "protocol_v1", "training_seed": seed,
        "variant_id": variant, "C": .01, "C_hex": .01.hex(),
        "target_order": list(candidate.TARGET_ORDER), "classes": [0, 1, 2, 3],
        "fit_role": "development_train_oof", "fit_row_count": 40,
        "fit_row_identity_sha256": DIGEST, "study_spec_sha256": DIGEST,
        "dataset_manifest_sha256": DIGEST, "feature_input_names": names,
        "feature_indices": indices,
        "feature_names": [names[i] for i in indices],
        "fit_parameters": dict(candidate.FROZEN_FIT_PARAMETERS, C=.01),
        "implementation": dict(candidate.IMPLEMENTATION),
        "coef": coef.tolist(), "intercept": [.6, -.2, .1, seed * .02],
        "n_iter": [1],
    })


def _transform(seed=1):
    from multimodal import geo_helpfulness_router_numeric as numeric
    from multimodal import geo_helpfulness_targets_features as m3

    variance = np.linspace(0, 1, 25, dtype=np.float64)
    return {
        "schema_version": numeric.TRANSFORM_SCHEMA_VERSION,
        "seed": seed, "fit_role": "development_train_oof", "fit_row_count": 40,
        "fit_row_identity_sha256": DIGEST,
        "feature_schema_sha256": common.canonical_hash(m3.build_router_feature_schema()),
        "numeric_input_columns": list(numeric.NUMERIC_INPUT_COLUMNS),
        "output_columns": list(numeric.output_feature_columns()),
        "categorical_vocabularies": deepcopy(numeric.CATEGORICAL_VOCABULARIES),
        "numeric_contract": deepcopy(numeric.NUMERIC_CONTRACT),
        "mean": np.linspace(-.5, .5, 25).tolist(),
        "variance": variance.tolist(),
        "scale": np.where(variance == 0, 1.0, np.sqrt(variance)).tolist(),
    }


def _features():
    from multimodal import geo_helpfulness_targets_features as m3

    rng = np.random.default_rng(71)
    probabilities = []
    for _ in range(3):
        weights = rng.uniform(.01, 1, (12, 18))
        probabilities.append(weights / weights.sum(axis=1, keepdims=True))
    return m3.build_router_feature_frame(
        *probabilities, probability_basis=m3.CALIBRATED_PROBABILITY_BASIS
    )


@pytest.fixture
def frozen(tmp_path):
    from multimodal import geo_helpfulness_targets_features as m3

    root, output = tmp_path / "frozen", tmp_path / "output"
    root.mkdir()
    output.mkdir()
    identities = [
        {"row_uid": f"image-{i:02d}", "file": f"photos/{i}.jpg", "plot_idx": f"P{i // 2}"}
        for i in range(4)
    ]
    transforms = [_transform(seed) for seed in common.SEEDS]
    temperatures = [
        {"seed": seed, "mode": mode, "temperature": 1.0 + seed * .1 + offset * .03}
        for seed in common.SEEDS
        for offset, mode in enumerate(common.MODES.values())
    ]
    documents = {
        "expert_temperatures.json": {"states": temperatures},
        "router_feature_transform.json": {"states": transforms},
        "router_feature_schema.json": m3.build_router_feature_schema(),
        "oof_heuristics.json": {"heuristics": {
            name: {"threshold_specification": common.ZERO}
            for name in ("image_geo_jsd", "geo_minus_raw_margin")
        }},
    }
    variants, states = {}, {}
    for variant in common.VARIANTS:
        variants[variant] = {"seeds": {}}
        for seed in common.SEEDS:
            state = _state(variant, seed)
            filename = f"{variant}_seed{seed}.json"
            states[variant, seed] = state
            documents[filename] = state
            variants[variant]["seeds"][str(seed)] = {
                "model_file": filename, "model_state_sha256": state["state_sha256"],
                "C_hex": state["C_hex"],
                "threshold_specification": common.NEVER if variant == "primary_full727" else common.ZERO,
            }
    for filename, document in documents.items():
        common.write_json(root / filename, document)
    frozen_state = {
        "components": {filename: {"file_sha256": common.file_hash(root / filename)} for filename in documents},
        "variants": variants,
    }
    common.write_json(root / "router_frozen_state.json", frozen_state)
    declaration = {
        "study_id": common.STUDY_ID, "no_go_preserved": True,
        "caches": {str(seed): {} for seed in common.SEEDS},
        "frozen_root": str(root), "identity_manifest": {"rows": identities},
        "source_snapshot": {str(path): common.file_hash(path) for path in root.iterdir()},
    }
    declaration["declaration_sha256"] = common.canonical_hash(declaration)
    common.write_json(output / "declaration.json", declaration)
    records = []
    for family_index, family in enumerate(common.FAMILIES):
        for index, identity in enumerate(identities):
            for seed in common.SEEDS:
                record = {"checkpoint_family": family, **identity, "training_seed": seed}
                for offset, expert in enumerate(common.MODES):
                    logits = np.linspace(-1, 1, 18, dtype=np.float64)
                    logits[(index + seed + offset + family_index) % 18] = 3.5 + offset
                    record[f"{expert}_logits"] = logits
                records.append(record)
    frame = pd.DataFrame.from_records(records, columns=common.LOGIT_INPUT_COLUMNS)
    return {
        "root": root, "output": output, "declaration": declaration,
        "logits": frame, "states": states, "transforms": transforms,
        "temperatures": temperatures, "feature_schema": documents["router_feature_schema.json"],
    }


def test_transform_exact_parity_with_frozen_m4_numeric():
    from multimodal import geo_helpfulness_router_numeric as numeric

    features, state = _features(), _transform()
    expected = numeric.transform_router_features(features, state)
    observed = common.apply_transform(features, state)
    np.testing.assert_array_equal(observed, expected)
    assert observed.shape == (12, 727)
    assert observed.dtype == np.float64
    np.testing.assert_array_equal(observed[:, 25:].sum(axis=1), np.full(12, 5))


@pytest.mark.parametrize("bad", ["order", "negative_variance", "bad_scale", "category", "nonfinite"])
def test_transform_rejects_corrupt_state_or_features(bad):
    features, state = _features(), _transform()
    if bad == "order":
        state["numeric_input_columns"] = state["numeric_input_columns"][::-1]
    elif bad == "negative_variance":
        state["variance"][0] = -1.0
    elif bad == "bad_scale":
        state["scale"][1] += .1
    elif bad == "category":
        features.loc[0, "image_pred"] = 18
    else:
        features.loc[0, "raw_confidence"] = np.nan
    with pytest.raises(ValueError):
        common.apply_transform(features, state)


def test_normalize_identity_preserves_opaque_plot_and_matches_protocol():
    from multimodal.geo_helpfulness_protocol import canonicalize_file, canonicalize_plot_idx

    for file, plot in (("./ABC\\photo.JPG", "001"), ("images//./Cafe\u0301.JPG", "Cafe\u0301"), ("Straße/a.jpg", "AbC")):
        assert common.normalize_identity(file, plot) == (
            canonicalize_file(file).casefold(), canonicalize_plot_idx(plot)
        )
    assert common.normalize_identity("a.jpg", "001")[1] == "001"


@pytest.mark.parametrize("file,plot", [
    ("/absolute.jpg", "P"), ("C:/absolute.jpg", "P"), ("./C:/absolute.jpg", "P"),
    ("../escape.jpg", "P"), ("a/../escape.jpg", "P"), ("./", "P"),
    ("a.jpg", " P"), ("a.jpg", "P "), ("a.jpg", ""),
    ("a.jpg", "bad\u0085plot"), ("a.jpg", "bad\ud800plot"),
    ("a\ud800.jpg", "P"), ("a\x00.jpg", "P"), (None, "P"), ("a.jpg", 1),
])
def test_normalize_identity_rejects_unsafe_paths_and_plot_text(file, plot):
    with pytest.raises(ValueError):
        common.normalize_identity(file, plot)


def test_probability_stability_zero_terms_and_dense_id_ties():
    logits = np.full((3, 18), -10000.0, dtype=np.float64)
    logits[0, :2] = 10000
    logits[1, 4] = 10000
    logits[2, :] = 10000
    probability = common.transferred_probability(logits, 2.0)
    np.testing.assert_array_equal(probability[0, :2], [.5, .5])
    assert np.count_nonzero(probability[0, 2:]) == 0
    np.testing.assert_array_equal(probability.argmax(axis=1), [0, 4, 0])
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(probability, common.transferred_probability(logits + 1234, 2.0))


@pytest.mark.parametrize("temperature", [True, False, 0.0, -1.0, np.inf, np.nan])
def test_probability_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        common.transferred_probability(np.zeros((2, 18), dtype=np.float64), temperature)


@pytest.mark.parametrize("value", [
    np.zeros((2, 18), dtype=np.float32), np.zeros((2, 17)), np.zeros((0, 18)),
    np.full((2, 18), np.nan), np.full((2, 18), np.inf), np.zeros(18),
])
def test_probability_rejects_invalid_logits(value):
    with pytest.raises(ValueError):
        common.transferred_probability(value, 1.0)


def test_reconstruction_replays_all_families_seeds_and_schema(frozen):
    tables = common.reconstruct_predictions(frozen["logits"], frozen["declaration"])
    assert {name: table.num_rows for name, table in tables.items()} == {
        "experts": 32, "routers": 96, "actions": 256,
    }
    for kind, table in tables.items():
        assert table.schema.equals(common._schema(kind, frozen["feature_schema"]))
        assert all(not field.nullable for field in table.schema)
        assert table.schema.metadata is None
        assert all(table[name].null_count == 0 for name in table.column_names)
    experts, routers, actions = (tables[name].to_pandas() for name in ("experts", "routers", "actions"))
    assert set(experts.checkpoint_family) == set(common.FAMILIES)
    assert set(experts.training_seed) == set(common.SEEDS)
    assert not set(experts.columns).intersection({"label_id_dense", "label_name", "target_state"})
    for family in common.FAMILIES:
        for seed in common.SEEDS:
            expert = experts.loc[experts.checkpoint_family.eq(family) & experts.training_seed.eq(seed)]
            matrix = common.apply_transform(expert, frozen["transforms"][seed - 1])
            for variant in common.VARIANTS:
                rows = routers.loc[routers.checkpoint_family.eq(family) & routers.training_seed.eq(seed) & routers.variant_id.eq(variant)]
                expected = candidate.predict_router_probabilities(matrix, frozen["states"][variant, seed])
                np.testing.assert_array_equal(rows[[f"q_{name}" for name in common.TARGET_ORDER]], expected)
                np.testing.assert_array_equal(rows.score, expected[:, 0] - expected[:, 1])
    primary = actions.loc[actions.policy_id.eq("primary_full727")]
    assert not primary.acted.any()
    np.testing.assert_array_equal(primary.final_pred, primary.raw_pred)
    replay = common.reconstruct_predictions(tables["experts"].select(common.LOGIT_INPUT_COLUMNS).to_pandas(), frozen["declaration"])
    for kind in tables:
        common.compare_tables(tables[kind], replay[kind])


@pytest.mark.parametrize("bad", ["label_column", "row_order", "missing_family", "missing_seed", "identity"])
def test_reconstruction_rejects_alignment_or_label_boundary_violations(frozen, bad):
    frame = frozen["logits"].copy(deep=True)
    if bad == "label_column":
        frame["label_id_dense"] = 0
    elif bad == "row_order":
        frame = frame.iloc[::-1]
    elif bad == "missing_family":
        frame = frame.loc[frame.checkpoint_family.eq(common.FAMILIES[0])]
    elif bad == "missing_seed":
        frame = frame.loc[frame.training_seed.ne(4)]
    else:
        frame.loc[0, "file"] = "other-file.jpg"
    with pytest.raises(ValueError):
        common.reconstruct_predictions(frame, frozen["declaration"])


def test_publication_reload_and_reuse_reproduce_the_same_tables(frozen):
    tables = common.reconstruct_predictions(frozen["logits"], frozen["declaration"])
    first = common.publish_predictions(frozen["output"], frozen["declaration"], tables)
    assert first["status"] == "built"
    _, manifest, frames = common.validate_predictions(frozen["output"])
    assert manifest["manifest_sha256"] == first["manifest_sha256"]
    assert len(frames["experts"]) == 32
    second = common.publish_predictions(frozen["output"], frozen["declaration"], tables)
    assert second == {"status": "reused_valid", "manifest_sha256": first["manifest_sha256"]}


def test_publication_rejects_symlink_before_writing(frozen, tmp_path):
    destination = tmp_path / "untouched"
    destination.mkdir()
    (frozen["output"] / "predictions").symlink_to(destination, target_is_directory=True)
    tables = common.reconstruct_predictions(frozen["logits"], frozen["declaration"])
    with pytest.raises(ValueError, match="symlink"):
        common.publish_predictions(frozen["output"], frozen["declaration"], tables)
    assert list(destination.iterdir()) == []


def test_common_import_and_replay_do_not_load_fitting_libraries(frozen, tmp_path):
    path = tmp_path / "synthetic_logits.parquet"
    pq.write_table(pa.Table.from_pandas(frozen["logits"], preserve_index=False), path)
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn'}:
            raise AssertionError('forbidden apply-only import: ' + fullname)
sys.meta_path.insert(0, Guard())
import pyarrow.parquet as pq
from multimodal import geo_helpfulness_legacy_router_common as common
declaration = common.read_json(__import__('pathlib').Path(sys.argv[1]))
tables = common.reconstruct_predictions(pq.read_table(sys.argv[2]).to_pandas(), declaration)
assert tables['actions'].num_rows == 256
assert not {'torch', 'scipy', 'sklearn'}.intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(frozen["output"] / "declaration.json"), str(path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr

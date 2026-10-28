"""Independent numerical and identity checks for strict geometry scoring."""

from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pytest

from multimodal.geometry_routing import class_plot_distances, score_candidates


def fixture_inputs():
    rng = np.random.default_rng(20261022)
    plots = np.array([f"p{p:02d}" for p in range(14) for _ in range(1 + p % 3)])
    labels = np.array([p // 6 for p in range(14) for _ in range(1 + p % 3)])
    size = len(plots)
    metadata = dict(
        reference_row_ids=np.array([f"r{i:03d}" for i in range(size)]),
        reference_plot_ids=plots, reference_labels=labels,
        query_row_ids=np.array([f"q{i}" for i in range(5)]),
        query_plot_ids=np.array(["v0", "v0", "v1", "v2", "v3"]),
    )
    return rng.normal(size=(size, 7)), rng.normal(size=(5, 7)), metadata


def brute_force(reference, queries, metadata, k, classes):
    """Scalar distance/photo/plot loops independent of the vectorized kernel."""
    expected = np.full((len(queries), classes), np.nan)
    expected_plots = np.full((len(queries), classes, k), "", dtype="<U10")
    plots = sorted(set(metadata["reference_plot_ids"]))
    counts = np.zeros(classes, dtype=int)
    grouped = {}
    for plot in plots:
        indices = [i for i, value in enumerate(metadata["reference_plot_ids"]) if value == plot]
        label = int(metadata["reference_labels"][indices[0]])
        grouped[plot] = (label, indices)
        counts[label] += 1
    for q, query in enumerate(queries):
        query_norm = sum(float(x) ** 2 for x in query) ** 0.5
        ranked = {c: [] for c in range(classes)}
        for plot, (label, indices) in grouped.items():
            photo_distances = []
            for index in indices:
                row = reference[index]
                norm = sum(float(x) ** 2 for x in row) ** 0.5
                cosine = sum(float(a) * float(b) for a, b in zip(query, row)) / query_norm / norm
                photo_distances.append(min(2.0, max(0.0, 1.0 - cosine)))
            ranked[label].append((min(photo_distances), plot))
        for c in range(classes):
            if counts[c] >= k:
                selected = sorted(ranked[c])[:k]
                expected[q, c] = sum(distance for distance, _ in selected) / k
                expected_plots[q, c] = [plot for _, plot in selected]
    return expected, expected_plots, counts


@pytest.mark.parametrize("k", [1, 2, 5, 8])
def test_distances_match_independent_brute_force(k):
    reference, queries, metadata = fixture_inputs()
    result = class_plot_distances(reference, queries, **metadata, k=k, num_classes=4)
    expected, nearest, counts = brute_force(reference, queries, metadata, k, 4)
    np.testing.assert_allclose(result.distances, expected, atol=5e-16, rtol=0, equal_nan=True)
    np.testing.assert_array_equal(result.nearest_plot_ids, nearest)
    np.testing.assert_array_equal(result.reference_plot_counts, counts)
    assert result.nearest_plot_indices.shape == (5, 4, k)


def test_multiple_photos_contribute_only_the_closest_photo_per_plot():
    metadata = dict(reference_row_ids=["r0", "r1", "r2"],
                    reference_plot_ids=["p0", "p0", "p1"], reference_labels=[0, 0, 0],
                    query_row_ids=["q"], query_plot_ids=["v"])
    result = class_plot_distances([[1, 0], [-1, 0], [0, 1]], [[1, 0]],
                                 **metadata, k=2, num_classes=2)
    assert result.distances[0, 0] == 0.5
    np.testing.assert_array_equal(result.nearest_plot_ids[0, 0], ["p0", "p1"])
    np.testing.assert_array_equal(result.reference_plot_counts, [2, 0])


def test_tied_neighbors_and_canonical_metadata_do_not_depend_on_reference_order():
    metadata = dict(reference_row_ids=["r3", "r1", "r2", "r0"],
                    reference_plot_ids=["p2", "p0", "p1", "p0"], reference_labels=[0] * 4,
                    query_row_ids=["q"], query_plot_ids=["v"])
    reference = np.array([[1, 0]] * 4)
    original = class_plot_distances(reference, [[0, 1]], **metadata, k=2, num_classes=2)
    permutation = [3, 2, 0, 1]
    reordered = {key: np.asarray(values)[permutation] if key.startswith("reference_") else values
                 for key, values in metadata.items()}
    other = class_plot_distances(reference[permutation], [[0, 1]], **reordered, k=2, num_classes=2)
    np.testing.assert_array_equal(original.nearest_plot_ids[0, 0], ["p0", "p1"])
    np.testing.assert_array_equal(original.reference_row_ids, ["r0", "r1", "r2", "r3"])
    np.testing.assert_array_equal(original.nearest_plot_indices, other.nearest_plot_indices)
    np.testing.assert_array_equal(original.reference_row_ids, other.reference_row_ids)


def test_independent_positive_scaling_preserves_cosine_geometry_and_inputs():
    reference, queries, metadata = fixture_inputs()
    original_reference, original_queries = reference.copy(), queries.copy()
    original = class_plot_distances(reference, queries, **metadata)
    reference_scales = np.linspace(0.01, 100, len(reference))[:, None]
    query_scales = np.linspace(0.1, 10, len(queries))[:, None]
    scaled = class_plot_distances(reference * reference_scales, queries * query_scales, **metadata)
    normalized = class_plot_distances(reference / np.linalg.norm(reference, axis=1, keepdims=True),
                                     queries / np.linalg.norm(queries, axis=1, keepdims=True), **metadata)
    for result in (scaled, normalized):
        np.testing.assert_allclose(original.distances, result.distances, atol=4e-16, rtol=0)
        np.testing.assert_array_equal(original.nearest_plot_indices, result.nearest_plot_indices)
    np.testing.assert_array_equal(reference, original_reference)
    np.testing.assert_array_equal(queries, original_queries)


@pytest.mark.parametrize("mutation, match", [
    ("conflicting_plot_labels", "multiple class labels"),
    ("plot_overlap", "plot identities overlap"),
    ("row_overlap", "row identities overlap"),
    ("duplicate_reference_row", "duplicate row identities"),
    ("duplicate_query_row", "duplicate row identities"),
    ("empty_identity", "nonempty string"),
    ("bad_class", "integer class IDs"),
    ("float_labels", "integer class IDs"),
])
def test_identity_and_label_errors_fail_closed(mutation, match):
    reference, queries, metadata = fixture_inputs()
    if mutation == "conflicting_plot_labels":
        metadata["reference_labels"][2] = 1
    elif mutation == "plot_overlap":
        metadata["query_plot_ids"] = metadata["query_plot_ids"].astype("<U10")
        metadata["query_plot_ids"][0] = metadata["reference_plot_ids"][0]
    elif mutation == "row_overlap":
        metadata["query_row_ids"] = metadata["query_row_ids"].astype("<U10")
        metadata["query_row_ids"][0] = metadata["reference_row_ids"][0]
    elif mutation == "duplicate_reference_row":
        metadata["reference_row_ids"][1] = metadata["reference_row_ids"][0]
    elif mutation == "duplicate_query_row":
        metadata["query_row_ids"][1] = metadata["query_row_ids"][0]
    elif mutation == "empty_identity":
        metadata["query_row_ids"][0] = " "
    elif mutation == "bad_class":
        metadata["reference_labels"][0] = 18
    else:
        metadata["reference_labels"] = metadata["reference_labels"].astype(float)
    with pytest.raises(ValueError, match=match):
        class_plot_distances(reference, queries, **metadata)


@pytest.mark.parametrize("value, match", [(0.0, "nonzero"), (np.nan, "nonfinite"), (np.inf, "nonfinite")])
@pytest.mark.parametrize("side", ["reference", "query"])
def test_invalid_vectors_are_rejected(value, match, side):
    reference, queries, metadata = fixture_inputs()
    (reference if side == "reference" else queries)[0] = value
    with pytest.raises(ValueError, match=match):
        class_plot_distances(reference, queries, **metadata)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_k_must_be_positive_integer(k):
    reference, queries, metadata = fixture_inputs()
    with pytest.raises(ValueError, match="k must be a positive integer"):
        class_plot_distances(reference, queries, **metadata, k=k)


def candidate_geometries():
    metadata = dict(reference_row_ids=["r0", "r1", "r2", "r3"],
                    reference_plot_ids=["p0", "p1", "p2", "p3"], reference_labels=[0, 0, 1, 1],
                    query_row_ids=["q0", "q1", "q2", "q3", "q4"],
                    query_plot_ids=["v0", "v1", "v2", "v3", "v4"])
    reference = [[1, 0], [1, 0], [0, 1], [0, 1]]
    image = class_plot_distances(reference, [[0, 1], [1, 0], [1, 1], [1, 0], [1, 0]],
                                 **metadata, k=2, num_classes=3)
    geo = class_plot_distances(reference, [[0, 1], [0, 1], [1, 1], [0, 1], [1, 0]],
                               **metadata, k=2, num_classes=3)
    return image, geo


def test_candidate_direction_joint_minimum_agreements_ties_and_unsupported_class():
    image, geo = candidate_geometries()
    scores = score_candidates(image, geo, [0, 0, 0, 0, 0], [1, 1, 1, 0, 2])
    assert scores.A_image[0] > 0 and scores.A_geo[0] > 0 and scores.score[0] > 0
    assert scores.A_image[1] < 0 < scores.A_geo[1]
    assert scores.score[1] == scores.A_image[1]
    assert scores.score[2] == 0  # Equal distances are an eligible disagreement, not an action.
    assert scores.score[3] == 0  # A supported agreement has finite, identical distances.
    np.testing.assert_array_equal(scores.geometry_supported, [True, True, True, True, False])
    np.testing.assert_array_equal(scores.eligible, [True, True, True, False, False])
    np.testing.assert_array_equal(scores.disagreement, [True, True, True, False, True])
    for name in ("score", "A_image", "A_geo", "d_image_fusion", "d_image_geo", "d_geo_fusion", "d_geo_geo"):
        assert np.isnan(getattr(scores, name)[4])
    np.testing.assert_array_equal(scores.reference_plots_geo_class, [2, 2, 2, 2, 0])


def test_candidate_scoring_requires_no_query_labels_and_allows_distinct_modality_widths():
    reference, queries, metadata = fixture_inputs()
    image = class_plot_distances(reference, queries, **metadata)
    geo = class_plot_distances(reference[:, :3], queries[:, :3], **metadata)
    scored = score_candidates(image, geo, [0] * 5, [1] * 5)
    assert scored.eligible.all()
    assert np.isfinite(scored.score).all()


def test_modalities_must_have_same_ordered_queries_and_reference_bank():
    image, geo = candidate_geometries()
    with pytest.raises(ValueError, match="query_row_ids"):
        score_candidates(image, replace(geo, query_row_ids=geo.query_row_ids[::-1]), [0] * 5, [1] * 5)
    with pytest.raises(ValueError, match="reference_row_ids"):
        score_candidates(image, replace(geo, reference_row_ids=np.array(["changed"] * 4)), [0] * 5, [1] * 5)
    with pytest.raises(ValueError, match="k or class count"):
        score_candidates(image, replace(geo, k=1), [0] * 5, [1] * 5)


@pytest.mark.parametrize("predictions", [[-1] * 5, [3] * 5, [0.0] * 5, [0] * 4])
def test_invalid_candidate_classes_rejected(predictions):
    image, geo = candidate_geometries()
    with pytest.raises(ValueError, match="integer class IDs"):
        score_candidates(image, geo, predictions, [1] * 5)


@pytest.mark.parametrize("epsilon", [0, -1, np.inf, np.nan, True, "small", 1j])
def test_epsilon_must_be_finite_positive_real(epsilon):
    image, geo = candidate_geometries()
    with pytest.raises(ValueError, match="epsilon must be finite and positive"):
        score_candidates(image, geo, [0] * 5, [1] * 5, epsilon=epsilon)


def test_synthetic_supported_candidates_match_unchanged_legacy_k5(monkeypatch):
    # Import only the historical numerical function; never invoke a loader/main.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    legacy = importlib.import_module("evaluate_joint_geometry_k5")
    import pandas as pd

    reference, queries, metadata = fixture_inputs()
    vectors = {
        "image": (reference / np.linalg.norm(reference, axis=1, keepdims=True),
                  queries / np.linalg.norm(queries, axis=1, keepdims=True)),
        "geo": (reference[:, :3] / np.linalg.norm(reference[:, :3], axis=1, keepdims=True),
                queries[:, :3] / np.linalg.norm(queries[:, :3], axis=1, keepdims=True)),
    }
    refs = pd.DataFrame({"file": metadata["reference_row_ids"],
                         "row_uid": metadata["reference_row_ids"],
                         "plot_idx": metadata["reference_plot_ids"],
                         "label_id_dense": metadata["reference_labels"]})
    identities = pd.DataFrame({"row_uid": metadata["query_row_ids"], "plot_idx": metadata["query_plot_ids"]})
    experts = identities.assign(raw_pred=[0, 1, 0, 1, 0], geo_pred=[1, 0, 0, 1, 1])
    previous, support = legacy.k5_scores(refs, vectors, identities, experts)
    geometries = {name: class_plot_distances(ref, query, **metadata)
                  for name, (ref, query) in vectors.items()}
    scores = score_candidates(geometries["image"], geometries["geo"], experts.raw_pred, experts.geo_pred)
    for column in ("d_image_fusion", "d_image_geo", "d_geo_fusion", "d_geo_geo", "A_image", "A_geo"):
        np.testing.assert_allclose(getattr(scores, column), previous[column], atol=1e-14, rtol=0)
    for modality, geometry in geometries.items():
        for candidate, predictions in (("fusion", experts.raw_pred), ("geo", experts.geo_pred)):
            actual = geometry.nearest_plot_ids[np.arange(len(queries)), predictions]
            np.testing.assert_array_equal(actual, np.stack(previous[f"nearest_{modality}_{candidate}_plots"]))
    assert support == {str(c): int(n) for c, n in enumerate(geometries["image"].reference_plot_counts)}
    np.testing.assert_array_equal(scores.geometry_supported, previous.valid_k5)

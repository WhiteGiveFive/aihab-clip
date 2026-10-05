"""Ordering, append-only scoring, and replay boundaries of the new workflow."""

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

from multimodal import geo_helpfulness_legacy_capacity as bundle
from multimodal import geo_helpfulness_legacy_capacity_registry as registry
from multimodal.geo_helpfulness_legacy_router_report import ONTOLOGY


def _snapshot(root):
    return {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Scientific model/report logic has its own tests; retain real publication,
    # hash, label-copy, receipt, and registry machinery in this small fixture.
    monkeypatch.setattr(bundle, "_fit_incapable", lambda: None)
    root = tmp_path / "new-output"
    paths = {
        name: tmp_path / name
        for name in (
            "legacy_root",
            "capacity_root",
            "mlp_source_root",
            "mlp_replay_root",
        )
    }
    for path in paths.values():
        path.mkdir()
        (path / "parent.txt").write_text("frozen source\n")
    protocol, event_root = tmp_path / "protocol", tmp_path / "registry"
    protocol.mkdir()
    (protocol / "parent.txt").write_text("frozen protocol\n")
    identity = [
        {"row_uid": f"row-{i}", "file": f"image-{i}.jpg", "plot_idx": f"plot-{i}"}
        for i in range(2)
    ]
    ontology = [
        {"dense_id": d, "canonical_l3_id": c, "label_name": n} for d, c, n in ONTOLOGY
    ]
    label_frame = pd.DataFrame(
        [
            {
                **row,
                "label_id_dense": i,
                "canonical_l3_id": i,
                "label_name": ontology[i]["label_name"],
            }
            for i, row in enumerate(identity)
        ]
    )
    label_table = pa.Table.from_pandas(
        label_frame, schema=bundle.old_score.LABEL_SCHEMA, preserve_index=False
    ).replace_schema_metadata(None)
    bundle.write_bytes(
        paths["legacy_root"] / "scores/labels.parquet", bundle.table_bytes(label_table)
    )
    old_build = {
        "composite_bundle_file_sha256": "d" * 64,
        "identity_projection_sha256": "b" * 64,
    }
    old = {
        "declaration_sha256": "c" * 64,
        "protocol_root": str(protocol),
        "registry_root": str(event_root),
        "source_snapshot": {
            str(protocol / "parent.txt"): bundle.file_hash(protocol / "parent.txt")
        },
        "identity_manifest": {"identity_projection_sha256": "b" * 64, "rows": identity},
        "ontology": ontology,
        "family_descriptions": {
            name: "synthetic historically exposed family"
            for name in bundle.legacy.FAMILIES
        },
    }
    old_prediction = {"manifest_sha256": "d" * 64}
    old_manifest = {"manifest_sha256": "e" * 64, "build_declaration": old_build}
    old_registry = bundle.old_score._registry_paths(old, old_build)
    old_owner = {
        "status": "reserved",
        "study_id": bundle.legacy.STUDY_ID,
        "owner_output_root": str(paths["legacy_root"]),
        **old_build,
    }
    for name, path in old_registry.items():
        if name != "root":
            bundle.write_json(path, old_owner)
    monkeypatch.setattr(
        bundle,
        "_legacy_metadata",
        lambda path: (deepcopy(old), deepcopy(old_prediction), deepcopy(old_manifest)),
    )
    monkeypatch.setattr(bundle.old_score, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(bundle.old_score, "EXPECTED_PLOTS", 2)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "original labels, historical metrics, or model fitting were accessed"
        )

    monkeypatch.setattr(bundle.old_score, "_read_original_labels", forbidden)
    monkeypatch.setattr(bundle.old_score, "_read_historical_replay", forbidden)
    selection = {
        "parent_file_sha256": {},
        "lineage": {"fixture": "selected"},
        "policies": {"frozen": True},
    }
    frame = pd.DataFrame(
        [
            {
                "checkpoint_family": family,
                "row_uid": row["row_uid"],
                "plot_idx": row["plot_idx"],
                "training_seed": seed,
                "value": 0.25,
            }
            for family in bundle.legacy.FAMILIES
            for seed in range(1, 5)
            for row in identity
        ]
    )
    tables = {
        name: pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            None
        )
        for name in ("experts", "routers", "actions")
    }
    events = []

    def feature_decode(path):
        assert path == paths["legacy_root"]
        assert (
            root / "declaration.json"
        ).is_file(), "test feature decode before declaration"
        declaration = bundle.load_declaration(root)
        assert declaration["frozen_policies"] == selection["policies"]
        events.append("feature_decode_after_declaration")
        return old, old_prediction, {name: frame.copy() for name in tables}

    monkeypatch.setattr(bundle.legacy, "validate_predictions", feature_decode)
    monkeypatch.setitem(
        sys.modules,
        "multimodal.geo_helpfulness_legacy_capacity_predictions",
        SimpleNamespace(
            load_selected_policies=lambda *args: deepcopy(selection),
            reconstruct_predictions=lambda *args: tables,
        ),
    )
    monkeypatch.setattr(
        bundle,
        "_metadata",
        lambda values: {
            "table_rows": {name: table.num_rows for name, table in values.items()},
            "test_labels_decoded": False,
        },
    )
    monkeypatch.setattr(bundle, "CODE_FILES", ())
    config_path = tmp_path / "configuration.json"
    bundle.write_json(config_path, bundle.CONFIGURATION)
    original_read = pq.read_table

    def guarded_read(path, *args, **kwargs):
        path = Path(path)
        if path == paths["legacy_root"] / "scores/labels.parquet":
            raise AssertionError(
                "original sealed label source should be byte-copied, not decoded"
            )
        if path == root / "scores/labels.parquet":
            assert (
                root / "predictions/manifest.json"
            ).is_file(), "labels before combined prediction seal"
            declaration = bundle.load_declaration(root)
            build = bundle.read_json(root / "scores/build_declaration.json")
            reservation = bundle.read_json(root / "scores/reservation.json")
            registry.validate_reservation(
                Path(declaration["registry_root"]), build, reservation
            )
            assert reservation["prior_registered_score_event"] is True
            assert (root / "scores/source_receipt.json").exists()
            events.append("label_decode_after_seal_reservation_receipt")
        if path.name == "test.parquet":
            raise AssertionError("original test cache opened")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(bundle.pq, "read_table", guarded_read)

    def reports(frames, labels, declaration):
        assert len(labels) == 2 and list(labels) == bundle.old_score.LABEL_SCHEMA.names
        results = {}
        for family in bundle.legacy.FAMILIES:
            values = {
                name: (
                    pa.table({"value": [1]})
                    if name.endswith(".parquet")
                    else {"fixture": True}
                )
                for name in bundle.FAMILY_FILES
            }
            values["classification.json"] = {"experts": {}, "policies": {}}
            values["policy_metrics.json"] = {}
            results[family] = values
        return results

    monkeypatch.setattr(bundle, "_family_reports", reports)
    return SimpleNamespace(
        root=root,
        paths=paths,
        protocol=protocol,
        registry=event_root,
        config=config_path,
        events=events,
    )


def _predict(f):
    return bundle.build_predictions(
        f.root,
        *[
            f.paths[name]
            for name in (
                "legacy_root",
                "capacity_root",
                "mlp_source_root",
                "mlp_replay_root",
            )
        ],
        f.config,
    )


def _replace_json(path, value, self_hash=None):
    if self_hash:
        value.pop(self_hash, None)
        value = bundle._seal(value, self_hash)
    path.chmod(0o644)
    path.unlink()
    bundle.write_json(path, value)


def test_full_ordering_apply_only_replay_and_reuse_preserve_bytes(fixture):
    f = fixture
    parent_before = {name: _snapshot(path) for name, path in f.paths.items()}
    registry_before = _snapshot(f.registry)
    assert _predict(f)["status"] == "built"
    assert f.events and all(e.startswith("feature") for e in f.events)
    assert not (f.root / "scores").exists()
    assert bundle.build_scores(f.root)["status"] == "built"
    assert any(e.startswith("label") for e in f.events)
    before = _snapshot(f.root)
    registry_sealed = _snapshot(f.registry)
    assert len(registry_sealed) == len(registry_before) + 3
    assert all(
        registry_sealed[name] == value for name, value in registry_before.items()
    )
    assert bundle.validate_predictions(f.root)["status"] == "validated"
    assert bundle.validate_scores(f.root)["status"] == "validated"
    assert _predict(f)["status"] == "reused_valid"
    assert bundle.build_scores(f.root)["status"] == "reused_valid"
    assert _snapshot(f.root) == before and _snapshot(f.registry) == registry_sealed
    assert {name: _snapshot(path) for name, path in f.paths.items()} == parent_before


@pytest.mark.parametrize("stage", ["predictions", "scores"])
def test_manifest_last_interruption_is_resumable_without_rewriting(
    fixture, monkeypatch, stage
):
    f = fixture
    if stage == "scores":
        _predict(f)
    original = bundle._publish

    def interrupted(root, declaration, kind, expected, extra):
        if kind == stage:
            assert set(bundle._files(root)) == expected
            assert not (root / "manifest.json").exists()
            raise RuntimeError("interruption before manifest")
        return original(root, declaration, kind, expected, extra)

    monkeypatch.setattr(bundle, "_publish", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        _predict(f) if stage == "predictions" else bundle.build_scores(f.root)
    before = _snapshot(f.root)
    registry_before = _snapshot(f.registry)
    monkeypatch.setattr(bundle, "_publish", original)
    result = _predict(f) if stage == "predictions" else bundle.build_scores(f.root)
    assert result["status"] == "built"
    after = _snapshot(f.root)
    assert all(after[name] == value for name, value in before.items())
    assert all(
        _snapshot(f.registry)[name] == value for name, value in registry_before.items()
    )


@pytest.mark.parametrize("target", ["source", "declaration", "prediction_manifest"])
def test_source_and_prediction_tampering_prevents_label_decode(fixture, target):
    f = fixture
    _predict(f)
    if target == "source":
        (f.paths["capacity_root"] / "parent.txt").write_text("changed")
    else:
        path = f.root / (
            "declaration.json"
            if target == "declaration"
            else "predictions/manifest.json"
        )
        value = bundle.read_json(path)
        value["study_id"] = "changed"
        _replace_json(path, value)
    f.events.clear()
    with pytest.raises(ValueError):
        bundle.build_scores(f.root)
    assert not any(e.startswith("label") for e in f.events)


@pytest.mark.parametrize(
    "target", ["labels.parquet", "source_receipt.json", "receipt.json"]
)
def test_sealed_label_and_receipt_tampering_is_rejected(fixture, target):
    f = fixture
    _predict(f)
    bundle.build_scores(f.root)
    path = f.root / "scores" / target
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    f.events.clear()
    with pytest.raises(ValueError):
        bundle.validate_scores(f.root)
    assert not any(e.startswith("label") for e in f.events)


def test_resealed_label_acquisition_receipt_is_still_rejected(fixture):
    f = fixture
    _predict(f)
    bundle.build_scores(f.root)
    path = f.root / "scores/source_receipt.json"
    value = bundle.read_json(path)
    value["parent_label_file_sha256"] = "0" * 64
    _replace_json(path, value)
    manifest_path = f.root / "scores/manifest.json"
    manifest = bundle.read_json(manifest_path)
    manifest["artifacts"]["source_receipt.json"].update(
        file_sha256=bundle.file_hash(path), bytes=path.stat().st_size
    )
    _replace_json(manifest_path, manifest, "manifest_sha256")
    f.events.clear()
    with pytest.raises(ValueError, match="does not reproduce"):
        bundle.validate_scores(f.root)
    assert not any(e.startswith("label") for e in f.events)


@pytest.mark.parametrize("stage", ["unowned", "partial_prediction", "partial_score"])
def test_unknown_partial_outputs_are_preserved(fixture, stage):
    f = fixture
    if stage == "unowned":
        f.root.mkdir()
        unknown = f.root / "user.txt"
    elif stage == "partial_prediction":
        _predict(f)
        unknown = f.root / "predictions/user.txt"
    else:
        _predict(f)
        (f.root / "scores").mkdir()
        unknown = f.root / "scores/user.txt"
    unknown.write_text("keep user content")
    before = _snapshot(f.root)
    with pytest.raises(ValueError, match="unowned|allowlist"):
        bundle.build_scores(f.root) if stage == "partial_score" else _predict(f)
    assert _snapshot(f.root) == before


def test_unowned_allowed_score_file_is_not_adopted(fixture):
    f = fixture
    _predict(f)
    bundle.write_json(f.root / "scores/summary.json", {"user": True})
    before = _snapshot(f.root)
    with pytest.raises(ValueError, match="unowned partial score"):
        bundle.build_scores(f.root)
    assert _snapshot(f.root) == before


def test_registry_reservation_tamper_precedes_new_label_decode(fixture):
    f = fixture
    _predict(f)
    bundle.build_scores(f.root)
    build = bundle.read_json(f.root / "scores/build_declaration.json")
    path = registry._paths(f.registry, build)["bundle"]
    value = bundle.read_json(path)
    value["owner_output_root"] = str(f.root.parent / "another-owner")
    _replace_json(path, value)
    f.events.clear()
    with pytest.raises(ValueError, match="reservation ownership"):
        bundle.validate_scores(f.root)
    assert not any(e.startswith("label") for e in f.events)


def test_output_lock_prevents_mutation_or_feature_decode(fixture):
    f = fixture
    with bundle.exclusive_lock(f.root):
        with pytest.raises(ValueError, match="owns this workflow lock"):
            _predict(f)
    assert not f.root.exists() and not f.events


@pytest.mark.parametrize("location", ["descendant", "ancestor"])
def test_source_containment_rejected_before_creation(fixture, location):
    f = fixture
    output = (
        f.paths["capacity_root"] / "new-output"
        if location == "descendant"
        else f.root.parent
    )
    before = _snapshot(f.root.parent)
    with pytest.raises(ValueError, match="disjoint"):
        bundle.build_predictions(
            output,
            *[
                f.paths[name]
                for name in (
                    "legacy_root",
                    "capacity_root",
                    "mlp_source_root",
                    "mlp_replay_root",
                )
            ],
            f.config,
        )
    assert _snapshot(f.root.parent) == before


def test_fresh_workflow_import_has_no_fitting_or_selection_capability():
    code = """
import sys
from multimodal import geo_helpfulness_legacy_capacity as workflow
from multimodal import geo_helpfulness_legacy_capacity_predictions
from multimodal import geo_helpfulness_legacy_capacity_report
from multimodal import geo_helpfulness_legacy_capacity_registry
workflow._fit_incapable()
assert not any(name.split('.')[0] in {'torch','sklearn','scipy','xgboost'} for name in sys.modules)
assert 'multimodal.geo_helpfulness_router_selection' not in sys.modules
sys.modules['multimodal.geo_helpfulness_router_capacity_fit'] = object()
try:
    workflow._fit_incapable()
except ValueError:
    pass
else:
    raise AssertionError('fitting-capable workflow accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kind", ["absent", "empty", "nonempty", "symlink", "directory", "unknown_lock"]
)
def test_legacy_metadata_allows_only_known_empty_regular_prediction_lock(
    tmp_path, monkeypatch, kind
):
    root = tmp_path / "legacy"
    declaration = {"declaration_sha256": "c" * 64, "source_snapshot": {}}
    bundle.write_json(root / "declaration.json", declaration)
    monkeypatch.setattr(bundle.legacy, "load_declaration", lambda path: declaration)
    for name in bundle.legacy.PREDICTION_FILES:
        bundle.write_bytes(root / "predictions" / name, b"authenticated fixture bytes")
    prediction = bundle._seal(
        {
            "study_id": bundle.legacy.STUDY_ID,
            "declaration_file_sha256": bundle.file_hash(root / "declaration.json"),
            "source_snapshot": {},
            "artifacts": {
                name: {"file_sha256": bundle.file_hash(root / "predictions" / name)}
                for name in bundle.legacy.PREDICTION_FILES
            },
        },
        "manifest_sha256",
    )
    bundle.write_json(root / "predictions/manifest.json", prediction)
    build = {
        "prediction_manifest_sha256": prediction["manifest_sha256"],
        "declaration_sha256": declaration["declaration_sha256"],
    }
    for name in bundle.old_score.SCORE_FILES - {"receipt.json", "source_receipt.json"}:
        bundle.write_json(root / "scores" / name, {"synthetic": True})
    source_receipt = bundle.old_score._source_receipt(root / "scores", build)
    bundle.write_json(root / "scores/source_receipt.json", source_receipt)
    receipt = bundle._seal({"synthetic": True}, "receipt_sha256")
    bundle.write_json(root / "scores/receipt.json", receipt)
    manifest = bundle._seal(
        {
            "study_id": bundle.legacy.STUDY_ID,
            "schema_version": bundle.old_score.SCHEMA,
            "build_declaration": build,
            "receipt_sha256": receipt["receipt_sha256"],
            "artifact_file_sha256": {
                name: bundle.file_hash(root / "scores" / name)
                for name in bundle.old_score.SCORE_FILES
            },
        },
        "manifest_sha256",
    )
    bundle.write_json(root / "scores/manifest.json", manifest)
    monkeypatch.setattr(bundle.old_score, "_event", lambda *args: {})
    monkeypatch.setattr(bundle.old_score, "_validate_registry", lambda *args: None)
    lock = root / ".predictions.lock"
    if kind in ("empty", "nonempty"):
        lock.write_bytes(b"" if kind == "empty" else b"not a lock")
    elif kind == "symlink":
        target = tmp_path / "empty-target"
        target.write_bytes(b"")
        lock.symlink_to(target)
    elif kind == "directory":
        lock.mkdir()
    elif kind == "unknown_lock":
        (root / ".unregistered.lock").write_bytes(b"")
    before = _snapshot(root)
    if kind in ("absent", "empty"):
        assert bundle._legacy_metadata(root) == (declaration, prediction, manifest)
    else:
        with pytest.raises(ValueError):
            bundle._legacy_metadata(root)
    assert _snapshot(root) == before

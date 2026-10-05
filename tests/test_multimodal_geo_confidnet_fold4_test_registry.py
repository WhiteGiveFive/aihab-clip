"""Fold-four test labels require one reservation bound to all four predictions."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_confidnet_fold4_test_registry as api
from multimodal.geo_helpfulness_router_m4_common import table_bytes


def _snapshot(root):
    return {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def _fixture(tmp_path):
    root = tmp_path / "output"
    source = tmp_path / "prior/labels.parquet"
    labels = pd.DataFrame([
        {"row_uid": "a", "file": "a.jpg", "plot_idx": "p1", "label_id_dense": 0,
         "canonical_l3_id": 0, "label_name": "class0"},
        {"row_uid": "b", "file": "b.jpg", "plot_idx": "p1", "label_id_dense": 0,
         "canonical_l3_id": 0, "label_name": "class0"},
        {"row_uid": "c", "file": "c.jpg", "plot_idx": "p2", "label_id_dense": 1,
         "canonical_l3_id": 1, "label_name": "class1"},
    ], columns=api.LABEL_SCHEMA.names)
    table = pa.Table.from_pandas(labels, schema=api.LABEL_SCHEMA, preserve_index=False)
    api.write_bytes(source, table_bytes(table))
    declaration = {
        "study_id": api.STUDY_ID,
        "identity_manifest": {
            "identity_projection_sha256": "b" * 64, "row_count": 3, "plot_count": 2,
            "rows": labels[["row_uid", "file", "plot_idx"]].to_dict(orient="records"),
        },
        "ontology": [{"dense_id": i, "canonical_l3_id": i, "label_name": f"class{i}"}
                     for i in range(18)],
        "label_source": {"path": str(source), "file_sha256": api.file_hash(source)},
    }
    api.write_json(root / "declaration.json", declaration)
    paths = {}
    for seed in range(1, 5):
        folder = root / f"predictions/seed_{seed}"
        api.write_bytes(folder / "predictions.npz", f"fake label-blind seed {seed}".encode())
        value = {
            "schema_version": "geo_helpfulness.confidnet_stage.v1",
            "role": "fold4_cleaned_test_predictions",
            "declaration_sha256": api.file_hash(root / "declaration.json"),
            "file_sha256": {"predictions.npz": api.file_hash(folder / "predictions.npz")},
            "metadata": {"seed": seed},
        }
        paths[seed] = folder / "manifest.json"
        api.write_json(paths[seed], value)
    return root, source, declaration, paths, api.make_build_record(root, declaration, paths)


def _prior_events(registry, build):
    first = None
    for number, study in enumerate(("legacy_frozen_router_transfer_v1",
                                    "legacy_capacity_router_transfer_v1",
                                    "relaxed_guardrails_lr_v1", "relaxed_guardrails_rf_v1",
                                    "relaxed_guardrails_margin_v1"), 1):
        digest = str(number) * 64
        snapshot = registry / "snapshots" / build["identity_projection_sha256"]
        owner = {"status": "reserved", "study_id": study,
                 "composite_bundle_file_sha256": digest,
                 "identity_projection_sha256": build["identity_projection_sha256"]}
        if first is None:
            first = snapshot / "first_score_event.json"
            api.write_json(first, owner)
        api.write_json(registry / f"bundles/{digest}.json", owner)
        event = {**owner, "artifact_role": "locked_test_score_event"}
        api.write_json(registry / f"bundles/{digest}.completed.json", event)
        api.write_json(snapshot / f"{digest}.json", event)
    assert len(_snapshot(registry)) == 16


def _receipt(root, build):
    receipt = {**build, "evaluation_event_id": "e" * 64}
    receipt["receipt_sha256"] = api.canonical_hash(receipt)
    path = root / "scores/receipt.json"
    api.write_json(path, receipt)
    return receipt, path


def test_build_binds_saved_declaration_and_all_four_manifest_files(tmp_path):
    root, _, declaration, paths, build = _fixture(tmp_path)
    assert build["declaration_sha256"] == api.file_hash(root / "declaration.json")
    assert build["prediction_manifest_sha256"] == api._composite(build)
    for seed, path in paths.items():
        assert build["seed_prediction_manifests"][str(seed)]["file_sha256"] == api.file_hash(path)
    assert api.make_build_record(root, declaration, paths) == build


@pytest.mark.parametrize("role,seed", [("other_predictions", 2),
                                      ("fold4_cleaned_test_predictions", 3),
                                      ("fold4_cleaned_test_predictions", "2"),
                                      ("fold4_cleaned_test_predictions", True),
                                      ("fold4_cleaned_test_predictions", 2.0)])
def test_prediction_manifest_role_and_exact_integer_seed_enforced(tmp_path, role, seed):
    root, _, declaration, paths, _ = _fixture(tmp_path)
    manifest = api.read_json(paths[2])
    manifest["role"] = role
    manifest["metadata"]["seed"] = seed
    paths[2].chmod(0o644)
    paths[2].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest (role|seed) differs"):
        api.make_build_record(root, declaration, paths)


@pytest.mark.parametrize("folder", ["scores", "unowned", "../outside"])
def test_labels_cannot_be_acquired_outside_the_designated_owner_directory(tmp_path, folder):
    root, _, declaration, _, build = _fixture(tmp_path)
    registry = tmp_path / "registry"
    reservation = api.reserve_event(registry, build)
    with pytest.raises(ValueError, match="(label_acquisition directory|canonical nonsymlink)"):
        api.acquire_labels(registry, build, reservation, declaration, root / folder)


def test_new_event_preserves_five_predecessors_and_completed_reuse(tmp_path, monkeypatch):
    root, source, declaration, _, build = _fixture(tmp_path)
    registry = tmp_path / "registry"
    _prior_events(registry, build)
    before = _snapshot(registry)
    reservation = api.reserve_event(registry, build)
    assert reservation["prior_registered_score_event"] is True
    labels, source_receipt = api.acquire_labels(registry, build, reservation, declaration, root / "label_acquisition")
    assert len(labels) == 3
    assert source_receipt["original_test_sources_opened"] is False
    receipt, receipt_path = _receipt(root, build)
    event = api.finish_event(registry, build, reservation, receipt, receipt_path)
    after = _snapshot(registry)
    assert len(after) == 19
    assert all(after[name] == value for name, value in before.items())
    local_before = _snapshot(root)
    source.unlink()
    original_read = Path.read_bytes
    def forbid_parent(path):
        if path == source:
            raise AssertionError("reuse tried to reopen parent labels")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", forbid_parent)
    repeated, repeated_receipt = api.acquire_labels(registry, build, reservation, declaration, root / "label_acquisition")
    assert repeated.equals(labels) and repeated_receipt == source_receipt
    assert api.reserve_event(registry, build) == reservation
    assert api.finish_event(registry, build, reservation, receipt, receipt_path) == event
    api.validate_event(registry, build, reservation, receipt, receipt_path)
    assert _snapshot(registry) == after
    assert _snapshot(root) == local_before


def test_label_access_without_reservation_never_opens_source(tmp_path, monkeypatch):
    root, source, declaration, _, build = _fixture(tmp_path)
    original = Path.read_bytes
    def forbid(path):
        if path == source:
            raise AssertionError("source opened before reservation")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", forbid)
    with pytest.raises((ValueError, FileNotFoundError)):
        api.acquire_labels(tmp_path / "registry", build, {}, declaration, root / "label_acquisition")
    assert not (root / "label_acquisition/labels.parquet").exists()


@pytest.mark.parametrize("seed", range(1, 5))
def test_missing_seed_manifest_rejected_before_reservation(tmp_path, seed):
    root, _, declaration, paths, _ = _fixture(tmp_path)
    paths.pop(seed)
    with pytest.raises(ValueError, match="four"):
        api.make_build_record(root, declaration, paths)


@pytest.mark.parametrize("target", ["declaration", "manifest", "payload", "extra_payload"])
def test_changed_predictions_or_declaration_block_labels(tmp_path, monkeypatch, target):
    root, source, declaration, paths, build = _fixture(tmp_path)
    registry = tmp_path / "registry"
    reservation = api.reserve_event(registry, build)
    path = {"declaration": root / "declaration.json", "manifest": paths[2],
            "payload": paths[2].parent / "predictions.npz",
            "extra_payload": paths[2].parent / "extra.json"}[target]
    if path.exists():
        path.chmod(0o644)
        if target in {"declaration", "manifest"}:
            value = api.read_json(path)
            value["changed"] = True
            path.write_text(json.dumps(value))
        else:
            path.write_bytes(b"changed")
    else:
        path.write_bytes(b"unexpected")
    original = Path.read_bytes
    def forbid(path):
        if path == source:
            raise AssertionError("source opened despite changed prediction binding")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", forbid)
    with pytest.raises(ValueError):
        api.acquire_labels(registry, build, reservation, declaration, root / "label_acquisition")
    assert not (root / "label_acquisition/labels.parquet").exists()


@pytest.mark.parametrize("target", ["first", "bundle", "event", "completed"])
def test_tampered_registry_records_rejected_without_overwrite(tmp_path, target):
    root, _, _, _, build = _fixture(tmp_path)
    registry = tmp_path / "registry"
    reservation = api.reserve_event(registry, build)
    receipt, path = _receipt(root, build)
    api.finish_event(registry, build, reservation, receipt, path)
    record = api._paths(registry, build)[target]
    record.chmod(0o644)
    value = api.read_json(record)
    value["changed"] = True
    record.write_text(json.dumps(value))
    before = _snapshot(registry)
    with pytest.raises(ValueError):
        api.validate_event(registry, build, reservation, receipt, path)
    assert _snapshot(registry) == before


@pytest.mark.parametrize("field", ["declaration_sha256", "seed_prediction_manifests", "owner_output_root"])
def test_receipt_must_bind_complete_score_owner(tmp_path, field):
    root, _, _, _, build = _fixture(tmp_path)
    registry = tmp_path / "registry"
    reservation = api.reserve_event(registry, build)
    receipt = {**build, "evaluation_event_id": "e" * 64}
    receipt.pop(field)
    receipt["receipt_sha256"] = api.canonical_hash(receipt)
    path = root / "scores/receipt.json"
    api.write_json(path, receipt)
    with pytest.raises(ValueError, match="another score owner"):
        api.finish_event(registry, build, reservation, receipt, path)


@pytest.mark.parametrize("field", ["row_uid", "plot_idx", "file", "label_name", "label_id_dense"])
def test_invalid_canonical_labels_rejected(tmp_path, field):
    _, source, declaration, _, _ = _fixture(tmp_path)
    labels = api.pq.read_table(source).to_pandas()
    labels.loc[0, field] = 2 if field == "label_id_dense" else "wrong"
    with pytest.raises(ValueError):
        api.validate_labels(labels, declaration)


def test_source_hash_mismatch_cannot_create_local_copy(tmp_path):
    root, source, declaration, _, build = _fixture(tmp_path)
    registry = tmp_path / "registry"
    reservation = api.reserve_event(registry, build)
    source.chmod(0o644)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source hash differs"):
        api.acquire_labels(registry, build, reservation, declaration, root / "label_acquisition")
    assert not (root / "label_acquisition/labels.parquet").exists()


@pytest.mark.parametrize("which", ["bundle", "event", "completed"])
def test_interrupted_registry_append_resumes_without_rewriting(tmp_path, monkeypatch, which):
    root, _, _, _, build = _fixture(tmp_path)
    registry = tmp_path / "registry"
    _prior_events(registry, build)
    before = _snapshot(registry)
    paths = api._paths(registry, build)
    real_write = api.write_json
    def interrupt(path, value):
        if path == paths[which]:
            raise RuntimeError("simulated interruption")
        return real_write(path, value)
    monkeypatch.setattr(api, "write_json", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        reservation = api.reserve_event(registry, build)
        receipt, path = _receipt(root, build)
        api.finish_event(registry, build, reservation, receipt, path)
    partial = _snapshot(registry)
    assert all(partial[key] == value for key, value in before.items())
    monkeypatch.setattr(api, "write_json", real_write)
    reservation = api.reserve_event(registry, build)
    receipt, path = _receipt(root, build)
    api.finish_event(registry, build, reservation, receipt, path)
    api.validate_event(registry, build, reservation, receipt, path)
    after = _snapshot(registry)
    assert len(after) == 19 and all(after[key] == value for key, value in partial.items())


def test_wrong_composite_cannot_be_reserved(tmp_path):
    _, _, _, _, build = _fixture(tmp_path)
    broken = deepcopy(build)
    broken["seed_prediction_manifests"]["2"]["file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="composite does not bind"):
        api.reserve_event(tmp_path / "registry", broken)
    assert not (tmp_path / "registry").exists()


def test_exclusive_lock_rejects_concurrent_reservation(tmp_path):
    _, _, _, _, build = _fixture(tmp_path)
    registry = tmp_path / "registry"
    with api.exclusive_lock(registry):
        with pytest.raises(ValueError, match="owns this workflow lock"):
            api.reserve_event(registry, build)
    assert not registry.exists()

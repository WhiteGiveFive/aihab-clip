"""Three-stage ordering, immutable replay, and append-only scoring boundaries."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_relaxed_rf as bundle
from multimodal import geo_helpfulness_relaxed_rf_registry as registry
from multimodal import geo_helpfulness_legacy_capacity_report as parent_report
from multimodal import geo_helpfulness_legacy_capacity_registry as parent_registry
from multimodal import geo_helpfulness_legacy_capacity_predictions as parent_predictions
from multimodal.geo_helpfulness_legacy_router_report import ONTOLOGY

REAL_TEST_TABLES = bundle._test_tables


def _snapshot(root):
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _replace_json(path, value, self_hash=None):
    if self_hash:
        value.pop(self_hash, None)
        value = bundle._seal(value, self_hash)
    path.chmod(0o644)
    path.unlink()
    bundle.write_json(path, value)


def _old_policies():
    policies = {}
    for name in parent_report.POLICIES:
        learner, variant = parent_report._policy_identity(name)
        policies[name] = {
            "learner": learner,
            "variant_id": variant,
            "config_id": name if learner == "heuristic" else "selected__" + name,
            "threshold_specification": deepcopy(parent_report.NEVER_THRESHOLD),
        }
        if learner == "lr":
            policies[name]["C_hex"] = float(0.1).hex()
    return policies


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Mock only scientific calculations; keep publication, hashes, labels, and
    # registry real. The separate subprocess test retains the production guard.
    monkeypatch.setattr(bundle, "_fit_incapable", lambda **kwargs: None)
    monkeypatch.setattr(bundle, "CODE_FILES", ())
    root = tmp_path / "new-output"
    paths = {
        name: tmp_path / name
        for name in (
            "legacy_root",
            "capacity_root",
            "lr_history_root",
            "original_legacy_root",
            "mlp_source_root",
            "mlp_replay_root",
        )
    }
    protocol, event_root = tmp_path / "protocol", tmp_path / "registry"
    for path in (*paths.values(), protocol):
        path.mkdir()
        (path / "parent.txt").write_text("frozen parent\n")
    identity = [
        {"row_uid": f"row-{i}", "file": f"image-{i}.jpg", "plot_idx": f"plot-{i}"}
        for i in range(2)
    ]
    ontology = [
        {"dense_id": d, "canonical_l3_id": c, "label_name": n} for d, c, n in ONTOLOGY
    ]
    labels = pd.DataFrame(
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
        labels, schema=bundle.old_score.LABEL_SCHEMA, preserve_index=False
    ).replace_schema_metadata(None)
    bundle.write_bytes(
        paths["legacy_root"] / "scores/labels.parquet", bundle.table_bytes(label_table)
    )
    old = {
        "declaration_sha256": "c" * 64,
        "protocol_root": str(protocol),
        "frozen_root": str(paths["capacity_root"]),
        "registry_root": str(event_root),
        "paths": {
            "legacy_root": str(paths["original_legacy_root"]),
            "capacity_root": str(paths["capacity_root"]),
            "mlp_source_root": str(paths["mlp_source_root"]),
            "mlp_replay_root": str(paths["mlp_replay_root"]),
        },
        "frozen_policies": _old_policies(),
        "parent_file_sha256": {
            str(protocol / "parent.txt"): bundle.file_hash(protocol / "parent.txt")
        },
        "identity_manifest": {"identity_projection_sha256": "b" * 64, "rows": identity},
        "ontology": ontology,
        "family_descriptions": {
            family: "synthetic historically exposed family"
            for family in bundle.legacy.FAMILIES
        },
    }
    parents = (
        old,
        {"manifest_sha256": "d" * 64},
        {"manifest_sha256": "e" * 64},
        {
            "declaration_sha256": "f" * 64,
            "parent_file_sha256": {},
        },
    )
    monkeypatch.setattr(bundle, "_parents", lambda *args: deepcopy(parents))
    monkeypatch.setattr(
        bundle.legacy, "load_declaration", lambda path: {"protocol_root": str(protocol)}
    )
    # Three completed predecessor studies: ten records, including one first claim.
    for study_id, digest in (
        ("legacy_frozen_router_transfer_v1", "d" * 64),
        ("legacy_capacity_router_transfer_v1", "e" * 64),
        ("relaxed_guardrails_lr_v1", "7" * 64),
    ):
        build = {
            "owner_output_root": str(tmp_path / study_id),
            "prediction_manifest_sha256": digest,
            "composite_bundle_file_sha256": digest,
            "identity_projection_sha256": "b" * 64,
        }
        registered = registry._paths(event_root, build)
        owner = {"status": "reserved", "study_id": study_id, **build}
        for name in ("bundle", "event", "completed"):
            bundle.write_json(registered[name], owner)
        if not registered["first"].exists():
            bundle.write_json(registered["first"], owner)
    monkeypatch.setattr(bundle.old_score, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(bundle.old_score, "EXPECTED_PLOTS", 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("original labels, historical metrics, or fitting accessed")

    monkeypatch.setattr(bundle.old_score, "_read_original_labels", forbidden)
    monkeypatch.setattr(bundle.old_score, "_read_historical_replay", forbidden)
    events = []
    policies = {}
    for variant in bundle.VARIANTS:
        for minimum in (20, 10, 0):
            for numerator in (5, 7, 9):
                guardrail = f"plots_{minimum}__harm_0p{numerator}"
                name = f"relaxed__rf__{variant}__{guardrail}"
                policies[name] = {
                    "policy_id": name,
                    "variant_id": variant,
                    "guardrail_id": guardrail,
                    "minimum_plots": minimum,
                    "harm_numerator": numerator,
                    "harm_denominator": 10,
                    "minimum_coverage_fraction": 0.01,
                    "learner": "rf",
                    "original_policy_id": "capacity__rf__" + variant,
                    "config_id": "selected__capacity__rf__" + variant,
                    "threshold_specification": {
                        "kind": "never_intervene",
                        "comparison": "strict_gt",
                        "value_hex": None,
                    },
                }

    def selection_values(declaration):
        assert (
            root / "declaration.json"
        ).is_file(), "development read before declaration"
        assert bundle.load_declaration(root) == declaration
        events.append("development_decode_after_declaration")
        return {
            "selection.json": {"fixture": True},
            "frozen_policies.json": deepcopy(policies),
            "report.md": "Synthetic development-only selection\n",
        }

    monkeypatch.setattr(bundle, "_selection_values", selection_values)
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
            for seed in bundle.SEEDS
            for row in identity
        ]
    )
    tables = {
        name: pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            None
        )
        for name in ("experts", "routers", "actions")
    }
    for name in ("experts", "routers"):
        bundle.write_bytes(
            paths["legacy_root"] / "predictions" / (name + ".parquet"),
            bundle.table_bytes(tables[name]),
        )

    def test_tables(output, declaration):
        selection, saved = bundle._load_policies(output, declaration)
        assert saved == policies
        assert (
            output / "selection/manifest.json"
        ).is_file(), "test features before selection seal"
        events.append("feature_decode_after_selection_seal")
        return selection, saved, tables

    monkeypatch.setattr(bundle, "_test_tables", test_tables)
    monkeypatch.setattr(
        bundle,
        "_prediction_metadata",
        lambda selection: {
            "selection_manifest_sha256": selection["manifest_sha256"],
            "table_rows": {name: value.num_rows for name, value in tables.items()},
            "test_labels_decoded": False,
        },
    )
    config = tmp_path / "configuration.json"
    bundle.write_json(config, bundle.CONFIGURATION)
    original_read = pq.read_table

    def guarded_read(path, *args, **kwargs):
        path = Path(path)
        if path == paths["legacy_root"] / "scores/labels.parquet":
            raise AssertionError("sealed source labels must be copied, not decoded")
        if path.name == "test.parquet":
            raise AssertionError("original test cache accessed")
        if path == root / "scores/labels.parquet":
            assert (
                root / "predictions/manifest.json"
            ).is_file(), "labels before prediction seal"
            declaration = bundle.load_declaration(root)
            build = bundle.read_json(root / "scores/build_declaration.json")
            reservation = bundle.read_json(root / "scores/reservation.json")
            registry.validate_reservation(
                Path(declaration["registry_root"]), build, reservation
            )
            assert reservation["prior_registered_score_event"] is True
            receipt = bundle.read_json(root / "scores/source_receipt.json")
            assert receipt["label_copy_file_sha256"] == bundle.file_hash(path)
            events.append("label_decode_after_seal_reservation_source_receipt")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(bundle.pq, "read_table", guarded_read)

    def reports(frames, labels, saved, old_saved):
        assert len(labels) == 2 and list(labels) == bundle.old_score.LABEL_SCHEMA.names
        assert saved == policies and old_saved == old["frozen_policies"]
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
        config=config,
        events=events,
    )


def _select(f):
    return bundle.build_selection(
        f.root,
        f.paths["legacy_root"],
        f.paths["capacity_root"],
        f.paths["lr_history_root"],
        f.config,
    )


def _predict(f):
    _select(f)
    return bundle.build_predictions(f.root)


def test_full_stage_order_replay_and_reuse_preserve_all_bytes(fixture):
    f = fixture
    parent_before = {name: _snapshot(path) for name, path in f.paths.items()}
    registry_before = _snapshot(f.registry)
    assert _select(f)["status"] == "built"
    assert f.events == ["development_decode_after_declaration"]
    assert not (f.root / "predictions").exists() and not (f.root / "scores").exists()
    assert bundle.build_predictions(f.root)["status"] == "built"
    assert not any(e.startswith("label") for e in f.events)
    assert bundle.build_scores(f.root)["status"] == "built"
    assert any(e.startswith("label") for e in f.events)
    before, registered = _snapshot(f.root), _snapshot(f.registry)
    assert len(registered) == len(registry_before) + 3 == 13
    assert all(registered[name] == value for name, value in registry_before.items())
    f.events.clear()
    assert bundle.validate_predictions(f.root)["status"] == "validated"
    assert bundle.validate_scores(f.root)["status"] == "validated"
    assert not any(e.startswith("development") for e in f.events)
    assert bundle.validate_selection(f.root)["status"] == "validated"
    assert _select(f)["status"] == "reused_valid"
    assert bundle.build_predictions(f.root)["status"] == "reused_valid"
    assert bundle.build_scores(f.root)["status"] == "reused_valid"
    assert _snapshot(f.root) == before and _snapshot(f.registry) == registered
    assert {name: _snapshot(path) for name, path in f.paths.items()} == parent_before


@pytest.mark.parametrize("stage", ["selection", "predictions", "scores"])
def test_manifest_last_interruption_resumes_without_rewriting(
    fixture, monkeypatch, stage
):
    f = fixture
    if stage in ("predictions", "scores"):
        _select(f)
    if stage == "scores":
        bundle.build_predictions(f.root)
    original = bundle._publish

    def interrupted(root, declaration, kind, expected, extra=None):
        if kind == stage:
            assert set(bundle._files(root)) == expected
            assert not (root / "manifest.json").exists()
            raise RuntimeError("interruption before manifest")
        return original(root, declaration, kind, expected, extra)

    action = {
        "selection": lambda: _select(f),
        "predictions": lambda: bundle.build_predictions(f.root),
        "scores": lambda: bundle.build_scores(f.root),
    }[stage]
    monkeypatch.setattr(bundle, "_publish", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        action()
    before, registered = _snapshot(f.root), _snapshot(f.registry)
    monkeypatch.setattr(bundle, "_publish", original)
    assert action()["status"] == "built"
    assert all(_snapshot(f.root)[name] == value for name, value in before.items())
    assert all(
        _snapshot(f.registry)[name] == value for name, value in registered.items()
    )


@pytest.mark.parametrize(
    "target", ["source", "config", "declaration", "selection", "predictions"]
)
def test_parent_and_manifest_tamper_precedes_labels(fixture, target):
    f = fixture
    _predict(f)
    if target == "source":
        (f.paths["capacity_root"] / "parent.txt").write_text("changed")
    else:
        path = (
            f.config
            if target == "config"
            else (
                f.root / "declaration.json"
                if target == "declaration"
                else f.root / target / "manifest.json"
            )
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
def test_label_and_receipt_tamper_is_rejected_before_decode(fixture, target):
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


def test_resealed_source_receipt_still_fails_before_labels(fixture):
    f = fixture
    _predict(f)
    bundle.build_scores(f.root)
    path = f.root / "scores/source_receipt.json"
    receipt = bundle.read_json(path)
    receipt["parent_label_file_sha256"] = "0" * 64
    _replace_json(path, receipt)
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


@pytest.mark.parametrize("stage", ["unowned", "selection", "predictions", "scores"])
def test_unknown_partial_outputs_preserved(fixture, stage):
    f = fixture
    if stage == "unowned":
        f.root.mkdir()
        unknown = f.root / "user.txt"
    else:
        _predict(f)
        (f.root / stage).mkdir(exist_ok=True)
        unknown = f.root / stage / "user.txt"
    unknown.write_text("keep user content")
    before = _snapshot(f.root)
    with pytest.raises(ValueError, match="unowned|allowlist"):
        bundle.build_scores(f.root) if stage == "scores" else _select(f)
    assert _snapshot(f.root) == before


def test_unowned_allowed_score_file_not_adopted(fixture):
    f = fixture
    _predict(f)
    bundle.write_json(f.root / "scores/summary.json", {"user": True})
    before = _snapshot(f.root)
    with pytest.raises(ValueError, match="unowned partial score"):
        bundle.build_scores(f.root)
    assert _snapshot(f.root) == before


def test_registry_owner_tamper_precedes_label_decode(fixture):
    f = fixture
    _predict(f)
    bundle.build_scores(f.root)
    build = bundle.read_json(f.root / "scores/build_declaration.json")
    path = registry._paths(f.registry, build)["bundle"]
    reservation = bundle.read_json(path)
    reservation["owner_output_root"] = str(f.root.parent / "other-owner")
    _replace_json(path, reservation)
    f.events.clear()
    with pytest.raises(ValueError, match="reservation ownership"):
        bundle.validate_scores(f.root)
    assert not any(e.startswith("label") for e in f.events)


@pytest.mark.parametrize("stage", ["selection", "predictions", "scores"])
def test_output_lock_precedes_mutation_and_decoding(fixture, stage):
    f = fixture
    if stage != "selection":
        _predict(f)
    f.events.clear()
    before = _snapshot(f.root)
    with bundle.exclusive_lock(f.root):
        with pytest.raises(ValueError, match="owns this workflow lock"):
            {
                "selection": lambda: _select(f),
                "predictions": lambda: bundle.build_predictions(f.root),
                "scores": lambda: bundle.build_scores(f.root),
            }[stage]()
    assert _snapshot(f.root) == before and not f.events


@pytest.mark.parametrize(
    "source",
    [
        "legacy_root",
        "capacity_root",
        "lr_history_root",
        "original_legacy_root",
        "mlp_source_root",
        "mlp_replay_root",
        "protocol",
        "registry",
    ],
)
@pytest.mark.parametrize("location", ["equal", "descendant", "ancestor"])
def test_source_containment_rejected_before_creation(fixture, source, location):
    f = fixture
    parent = f.paths[source] if source in f.paths else getattr(f, source)
    root = {
        "equal": parent,
        "descendant": parent / "new-output",
        "ancestor": parent.parent,
    }[location]
    before = _snapshot(f.root.parent)
    with pytest.raises(ValueError, match="disjoint"):
        bundle.build_selection(
            root,
            f.paths["legacy_root"],
            f.paths["capacity_root"],
            f.paths["lr_history_root"],
            f.config,
        )
    assert _snapshot(f.root.parent) == before


def test_prediction_before_selection_seal_cannot_open_features(fixture):
    f = fixture
    with pytest.raises((ValueError, FileNotFoundError)):
        bundle.build_predictions(f.root)
    assert not f.events
    _select(f)
    (f.root / "selection/manifest.json").unlink()
    f.events.clear()
    with pytest.raises((ValueError, FileNotFoundError)):
        bundle.build_predictions(f.root)
    assert not f.events


@pytest.mark.parametrize(
    "mutation",
    ["policy_identity", "guardrail_metadata", "config_id", "threshold"],
)
def test_malformed_sealed_policy_fails_before_test_source_open(
    fixture, monkeypatch, mutation
):
    f = fixture
    _select(f)
    path = f.root / "selection/frozen_policies.json"
    policies = bundle.read_json(path)
    name = next(iter(policies))
    if mutation == "policy_identity":
        policies["undeclared-policy"] = policies.pop(name)
    elif mutation == "guardrail_metadata":
        policies[name]["minimum_plots"] = 999
    elif mutation == "config_id":
        policies[name]["config_id"] = "different-selected-model"
    else:
        policies[name]["threshold_specification"]["comparison"] = "greater_equal"
    _replace_json(path, policies)
    manifest_path = f.root / "selection/manifest.json"
    manifest = bundle.read_json(manifest_path)
    manifest["artifacts"]["frozen_policies.json"].update(
        file_sha256=bundle.file_hash(path), bytes=path.stat().st_size
    )
    _replace_json(manifest_path, manifest, "manifest_sha256")

    def forbidden(*args, **kwargs):
        raise AssertionError("test source opened before policy contract validation")

    monkeypatch.setattr(bundle, "_test_tables", REAL_TEST_TABLES)
    monkeypatch.setattr(bundle.predecessor, "load_declaration", forbidden)
    monkeypatch.setattr(bundle.predecessor, "_validate_predictions", forbidden)
    before, registered = _snapshot(f.root), _snapshot(f.registry)
    f.events.clear()
    with pytest.raises(ValueError):
        bundle.build_predictions(f.root)
    assert not f.events
    assert not (f.root / "predictions").exists()
    assert _snapshot(f.root) == before and _snapshot(f.registry) == registered


def test_declaration_binds_config_new_code_and_all_existing_registry_records(
    fixture, monkeypatch
):
    f = fixture
    code = f.root.parent / "new-study-code.py"
    code.write_text("# declared implementation\n")
    monkeypatch.setattr(bundle, "PROJECT_ROOT", f.root.parent)
    monkeypatch.setattr(bundle, "CODE_FILES", (code.name,))
    registry_before = _snapshot(f.registry)
    _select(f)
    declaration = bundle.load_declaration(f.root)
    hashes = declaration["parent_file_sha256"]
    assert hashes[str(code)] == bundle.file_hash(code)
    assert hashes[str(f.config)] == bundle.file_hash(f.config)
    assert all(hashes[name] == bundle.file_hash(Path(name)) for name in registry_before)
    assert declaration["m4_no_go_preserved"] is True
    assert declaration["model_reselection_permitted"] is False
    code.write_text("# changed implementation\n")
    f.events.clear()
    with pytest.raises(ValueError):
        bundle.build_predictions(f.root)
    assert not f.events


@pytest.mark.parametrize("mutation", ["bool_for_integer", "new_grid", "extra_key"])
def test_scientific_configuration_is_exact_and_checked_before_declaration(
    fixture, mutation
):
    f = fixture
    configuration = bundle.read_json(f.config)
    if mutation == "bool_for_integer":
        configuration["training_seeds"][0] = True
    elif mutation == "new_grid":
        configuration["harm_numerator_grid"] = [5, 7, 10]
    else:
        configuration["post_hoc_override"] = True
    _replace_json(f.config, configuration)
    with pytest.raises(ValueError, match="configuration differs"):
        _select(f)
    assert not f.root.exists() and not f.events


def _metadata_manifest(root, declaration, schema, study, names, **extra):
    value = bundle._seal(
        {
            "schema_version": schema,
            "study_id": study,
            "declaration_sha256": declaration["declaration_sha256"],
            "artifacts": {
                name: {
                    "file_sha256": bundle.file_hash(root / name),
                    "bytes": (root / name).stat().st_size,
                }
                for name in names
            },
            **extra,
        },
        "manifest_sha256",
    )
    bundle.write_json(root / "manifest.json", value)
    return value


@pytest.fixture
def authenticated_parents(tmp_path, monkeypatch):
    """Real parent metadata/registry authentication; no valid data payloads exist."""
    predecessor = bundle.predecessor
    history = bundle.lr_history
    root, capacity_root, history_root = (
        tmp_path / name for name in ("legacy", "capacity", "history")
    )
    original_root, protocol, event_root = (
        tmp_path / name for name in ("original", "protocol", "registry")
    )
    marker = tmp_path / "frozen-parent.txt"
    marker.write_text("unchanged parent")
    identity = {"identity_projection_sha256": "b" * 64, "rows": []}
    old = bundle._seal(
        {
            "schema_version": predecessor.SCHEMA + ".declaration",
            "study_id": predecessor.STUDY_ID,
            "configuration": predecessor.CONFIGURATION,
            "paths": {
                "legacy_root": str(original_root),
                "capacity_root": str(capacity_root),
                "mlp_source_root": str(tmp_path / "mlp_source"),
                "mlp_replay_root": str(tmp_path / "mlp_replay"),
            },
            "parent_file_sha256": {str(marker): bundle.file_hash(marker)},
            "registry_root": str(event_root),
            "identity_manifest": identity,
            "ontology": [],
            "m4_no_go_preserved": True,
            "frozen_policies": _old_policies(),
            "router_parent_lineage": {"frozen_model_parent": "fixture"},
            "legacy_declaration_sha256": "1" * 64,
            "legacy_prediction_manifest_sha256": "2" * 64,
            "legacy_score_manifest_sha256": "3" * 64,
        },
        "declaration_sha256",
    )
    bundle.write_json(root / "declaration.json", old)
    for name in predecessor.PREDICTION_FILES:
        bundle.write_bytes(root / "predictions" / name, b"opaque prediction payload")
    prediction = _metadata_manifest(
        root / "predictions",
        old,
        predecessor.SCHEMA + ".predictions",
        predecessor.STUDY_ID,
        predecessor.PREDICTION_FILES,
    )
    build = predecessor._build_record(root, old, prediction)
    bundle.write_json(root / "scores/build_declaration.json", build)
    reservation = parent_registry.reserve_event(event_root, build)
    bundle.write_json(root / "scores/reservation.json", reservation)
    for target in (
        original_root / "scores/labels.parquet",
        root / "scores/labels.parquet",
    ):
        bundle.write_bytes(target, b"opaque authenticated labels; must not decode")
    bundle.write_json(
        root / "scores/source_receipt.json",
        predecessor._source_receipt(root, old, build),
    )
    for name in predecessor.SCORE_FILES:
        path = root / "scores" / name
        if not path.exists() and name != "receipt.json":
            bundle.write_bytes(path, b"opaque metric payload; must not decode")
    receipt = bundle._seal(
        {
            **build,
            "reservation": reservation,
            "evaluation_event_id": "4" * 64,
            "output_file_sha256": {
                name: bundle.file_hash(root / "scores" / name)
                for name in predecessor.SCORE_FILES - {"receipt.json"}
            },
        },
        "receipt_sha256",
    )
    bundle.write_json(root / "scores/receipt.json", receipt)
    parent_registry.finish_event(
        event_root, build, reservation, receipt, root / "scores/receipt.json"
    )
    score = _metadata_manifest(
        root / "scores",
        old,
        predecessor.SCHEMA + ".scores",
        predecessor.STUDY_ID,
        predecessor.SCORE_FILES,
        prediction_manifest_sha256=prediction["manifest_sha256"],
        receipt_sha256=receipt["receipt_sha256"],
    )
    assert len(bundle._files(root)) == 36
    history_config = tmp_path / "history_config.json"
    bundle.write_json(history_config, history.CONFIGURATION)
    history_declaration = bundle._seal(
        {
            "schema_version": history.SCHEMA + ".declaration",
            "study_id": history.STUDY_ID,
            "configuration": history.CONFIGURATION,
            "output_root": str(history_root),
            "config_path": str(history_config),
            "parent_file_sha256": {str(marker): bundle.file_hash(marker)},
            "m4_no_go_preserved": True,
            "registry_root": str(event_root),
            "identity_manifest": identity,
        },
        "declaration_sha256",
    )
    bundle.write_json(history_root / "declaration.json", history_declaration)
    for folder, names in (
        ("selection", history.SELECTION_FILES),
        ("predictions", history.PREDICTION_FILES),
        ("scores", history.SCORE_FILES),
    ):
        for name in names:
            bundle.write_bytes(
                history_root / folder / name, b"opaque preceding LR payload"
            )
        _metadata_manifest(
            history_root / folder,
            history_declaration,
            history.SCHEMA + "." + folder,
            history.STUDY_ID,
            names,
        )
    assert len(bundle._files(history_root)) == 40
    original = (
        {"declaration_sha256": "1" * 64, "protocol_root": str(protocol)},
        {"manifest_sha256": "2" * 64},
        {"manifest_sha256": "3" * 64},
    )
    selected = {
        "policies": deepcopy(old["frozen_policies"]),
        "lineage": deepcopy(old["router_parent_lineage"]),
    }
    monkeypatch.setattr(
        predecessor, "_legacy_metadata", lambda path: deepcopy(original)
    )
    monkeypatch.setattr(
        parent_predictions, "load_selected_policies", lambda *args: deepcopy(selected)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "test labels, test features, metrics, or selection decoded"
        )

    monkeypatch.setattr(bundle.pq, "read_table", forbidden)
    monkeypatch.setattr(predecessor, "_validate_predictions", forbidden)
    monkeypatch.setattr(predecessor, "validate_scores", forbidden)
    monkeypatch.setattr(history, "validate_scores", forbidden)
    return SimpleNamespace(
        root=root,
        capacity=capacity_root,
        history=history_root,
        registry=event_root,
        declaration=old,
        prediction=prediction,
        score=score,
        history_declaration=history_declaration,
        marker=marker,
        build=build,
        reservation=reservation,
        receipt=receipt,
        selected=selected,
    )


def test_parents_authenticate_completed_bundle_without_any_data_decode(
    authenticated_parents,
):
    f = authenticated_parents
    before = _snapshot(f.root.parent)
    assert bundle._parents(f.root, f.capacity, f.history) == (
        f.declaration,
        f.prediction,
        f.score,
        f.history_declaration,
    )
    assert _snapshot(f.root.parent) == before


@pytest.mark.parametrize(
    "damage",
    [
        "parent_hash",
        "prediction_study",
        "score_study",
        "score_prediction_link",
        "build_owner",
        "receipt_owner",
        "receipt_self_hash",
        "receipt_outputs",
        "receipt_reservation",
        "source_receipt",
        "label_copy",
        "registry_owner",
        "selected_model",
        "selected_lineage",
        "capacity_path",
        "no_go",
        "history_identity",
        "history_artifact",
    ],
)
def test_parent_metadata_tamper_fails_closed_without_data_decode(
    authenticated_parents, damage
):
    f = authenticated_parents
    root = f.root
    if damage == "parent_hash":
        f.marker.write_text("changed")
    elif damage in ("prediction_study", "score_study", "score_prediction_link"):
        path = (
            root
            / ("predictions" if damage == "prediction_study" else "scores")
            / "manifest.json"
        )
        value = bundle.read_json(path)
        value[
            (
                "prediction_manifest_sha256"
                if damage == "score_prediction_link"
                else "study_id"
            )
        ] = "changed"
        _replace_json(path, value, "manifest_sha256")
    elif damage.startswith("receipt_") or damage in (
        "build_owner",
        "source_receipt",
        "label_copy",
    ):
        name = (
            "build_declaration.json"
            if damage == "build_owner"
            else (
                "source_receipt.json"
                if damage == "source_receipt"
                else "labels.parquet" if damage == "label_copy" else "receipt.json"
            )
        )
        path = root / "scores" / name
        if damage == "label_copy":
            path.chmod(0o644)
            path.write_bytes(b"changed labels without decoding")
        else:
            value = bundle.read_json(path)
            if damage in ("receipt_owner", "build_owner"):
                value["owner_output_root"] = str(root.parent / "different-owner")
            elif damage == "receipt_self_hash":
                value["receipt_sha256"] = "0" * 64
            elif damage == "receipt_outputs":
                value["output_file_sha256"].pop("summary.json")
            elif damage == "receipt_reservation":
                value["reservation"]["study_id"] = "wrong"
            else:
                value["parent_label_file_sha256"] = "0" * 64
            _replace_json(
                path,
                value,
                (
                    "receipt_sha256"
                    if damage.startswith("receipt_") and damage != "receipt_self_hash"
                    else None
                ),
            )
        manifest_path = root / "scores/manifest.json"
        manifest = bundle.read_json(manifest_path)
        manifest["artifacts"][name].update(
            file_sha256=bundle.file_hash(path), bytes=path.stat().st_size
        )
        if damage.startswith("receipt_") and damage != "receipt_self_hash":
            manifest["receipt_sha256"] = bundle.read_json(path)["receipt_sha256"]
        _replace_json(manifest_path, manifest, "manifest_sha256")
    elif damage == "registry_owner":
        path = parent_registry._paths(f.registry, f.build)["completed"]
        value = bundle.read_json(path)
        value["owner_output_root"] = str(root.parent / "different-owner")
        _replace_json(path, value)
    elif damage == "selected_model":
        name = "capacity__rf__primary_full727"
        f.selected["policies"][name]["config_id"] = "different-selected-model"
    elif damage == "selected_lineage":
        f.selected["lineage"]["frozen_model_parent"] = "different"
    elif damage in ("capacity_path", "no_go"):
        path = root / "declaration.json"
        value = bundle.read_json(path)
        if damage == "capacity_path":
            value["paths"]["capacity_root"] = str(root.parent / "different-capacity")
        else:
            value["m4_no_go_preserved"] = False
        _replace_json(path, value, "declaration_sha256")
        for folder in ("predictions", "scores"):
            manifest_path = root / folder / "manifest.json"
            manifest = bundle.read_json(manifest_path)
            manifest["declaration_sha256"] = bundle.read_json(path)[
                "declaration_sha256"
            ]
            _replace_json(manifest_path, manifest, "manifest_sha256")
    elif damage == "history_identity":
        path = f.history / "declaration.json"
        value = bundle.read_json(path)
        value["identity_manifest"]["identity_projection_sha256"] = "0" * 64
        _replace_json(path, value, "declaration_sha256")
        for folder in ("selection", "predictions", "scores"):
            manifest_path = f.history / folder / "manifest.json"
            manifest = bundle.read_json(manifest_path)
            manifest["declaration_sha256"] = bundle.read_json(path)[
                "declaration_sha256"
            ]
            _replace_json(manifest_path, manifest, "manifest_sha256")
    else:
        path = f.history / "scores/summary.json"
        path.chmod(0o644)
        path.write_bytes(b"changed opaque preceding LR history")
    before = _snapshot(f.root.parent)
    with pytest.raises(ValueError):
        bundle._parents(f.root, f.capacity, f.history)
    assert _snapshot(f.root.parent) == before


@pytest.mark.parametrize(
    "function",
    ["build_predictions", "validate_predictions", "build_scores", "validate_scores"],
)
def test_public_test_stages_reject_selector_import_before_access(function):
    code = f"""
import sys
from pathlib import Path
from multimodal import geo_helpfulness_relaxed_rf as workflow
sys.modules['multimodal.geo_helpfulness_relaxed_rf_selection'] = object()
try:
    workflow.{function}(Path('/nonexistent-relaxed-study-fixture'))
except ValueError as error:
    assert 'prohibited' in str(error) or 'prohibits' in str(error), str(error)
else:
    raise AssertionError('test stage accepted development selector')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "forbidden",
    [
        "torch",
        "sklearn",
        "scipy",
        "xgboost",
        "multimodal.geo_helpfulness_relaxed_rf_selection",
        "multimodal.geo_helpfulness_relaxed_guardrails_selection",
        "multimodal.geo_helpfulness_router_selection",
        "multimodal.geo_helpfulness_router_capacity_fit",
    ],
)
def test_fresh_test_stage_has_no_fitting_or_selection_capability(forbidden):
    code = f"""
import sys
from multimodal import geo_helpfulness_relaxed_rf as workflow
workflow._fit_incapable(test_stage=True)
assert not any(name.split('.')[0] in {{'torch', 'sklearn', 'scipy', 'xgboost'}} for name in sys.modules)
assert 'multimodal.geo_helpfulness_relaxed_rf_selection' not in sys.modules
sys.modules[{forbidden!r}] = object()
try:
    workflow._fit_incapable(test_stage=True)
except ValueError:
    pass
else:
    raise AssertionError('prohibited capability accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr

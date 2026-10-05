"""Synthetic margin workflow sealing, label boundaries, and isolated CLI phases."""

import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pyarrow as pa

from multimodal import geo_helpfulness_relaxed_margin as workflow

import pytest

from tools import run_multimodal_geo_helpfulness_relaxed_margin as runner


@pytest.fixture(autouse=True)
def restore_environment(monkeypatch):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "PYTHONMALLOC",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(name, os.environ.get(name, ""))


@pytest.mark.parametrize(
    "command,phases",
    [
        ("build", ["select", "predict", "score"]),
        ("validate", ["validate-selection", "validate-predictions", "validate-scores"]),
    ],
)
def test_isolated_phases_paths_cpu_limits_and_live_stderr(
    monkeypatch, capsys, tmp_path, command, phases
):
    observed = []

    def run(arguments, **kwargs):
        phase = arguments[2]
        observed.append(phase)
        assert arguments[0] == sys.executable
        assert (
            Path(arguments[1]).name
            == "run_multimodal_geo_helpfulness_relaxed_margin.py"
        )
        assert kwargs["cwd"] == runner.PROJECT_ROOT
        assert (
            kwargs.get("stderr") is None and kwargs["stdout"] == runner.subprocess.PIPE
        )
        assert kwargs.get("env", os.environ)["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs.get("env", os.environ)["PYTHONMALLOC"] == "malloc"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            assert kwargs.get("env", os.environ)[name] == "1"
        for name in runner.PATHS:
            flag = "--" + name.replace("_", "-")
            assert Path(arguments[arguments.index(flag) + 1]).is_absolute()
        assert arguments[arguments.index("--output-dir") + 1] == str(tmp_path)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": (
                        "validated" if phase.startswith("validate") else "reused_valid"
                    )
                }
            ),
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main([command, "--output-dir", str(tmp_path)]) == 0
    assert observed == phases
    capture = capsys.readouterr()
    value = json.loads(capture.out)
    assert value["status"] == ("validated" if command == "validate" else "reused_valid")
    assert value["elapsed_seconds"] >= 0
    events = [json.loads(line) for line in capture.err.splitlines()]
    assert [e["event"] for e in events] == ["phase_started", "phase_completed"] * 3


@pytest.mark.parametrize(
    "returncode,stdout",
    [
        (1, ""),
        (0, "not-json"),
        (0, "[]"),
        (0, "{}"),
        (0, '{"status":"failed"}'),
        (0, '{"status":"validated"}'),
        (0, '{"status":"built","value":NaN}'),
    ],
)
def test_selection_failure_never_starts_prediction_or_scoring(
    monkeypatch, capsys, returncode, stdout
):
    observed = []

    def run(arguments, **kwargs):
        observed.append(arguments[2])
        return SimpleNamespace(returncode=returncode, stdout=stdout)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["build"]) == 1
    assert observed == ["select"]
    capture = capsys.readouterr()
    assert not capture.out
    assert json.loads(capture.err.splitlines()[-1])["status"] == "failed"


def test_prediction_failure_after_selection_never_starts_scoring(monkeypatch, capsys):
    observed = []

    def run(arguments, **kwargs):
        phase = arguments[2]
        observed.append(phase)
        return SimpleNamespace(
            returncode=1 if phase == "predict" else 0,
            stdout=json.dumps({"status": "built"}),
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["build"]) == 1
    assert observed == ["select", "predict"]
    capture = capsys.readouterr()
    assert not capture.out
    assert "predict failed" in json.loads(capture.err.splitlines()[-1])["error"]


@pytest.mark.parametrize(
    "command,function",
    [
        ("select", "build_selection"),
        ("validate-selection", "validate_selection"),
        ("predict", "build_predictions"),
        ("score", "build_scores"),
        ("validate-predictions", "validate_predictions"),
        ("validate-scores", "validate_scores"),
    ],
)
def test_single_phase_dispatch_and_defaults(monkeypatch, capsys, command, function):
    import multimodal

    module = ModuleType("multimodal.geo_helpfulness_relaxed_margin")
    observed = []

    def call(*args):
        observed.append(args)
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        return {"status": "validated" if command.startswith("validate") else "built"}

    def forbidden(*args):
        raise AssertionError("another workflow phase was called")

    for name in (
        "build_selection",
        "validate_selection",
        "build_predictions",
        "build_scores",
        "validate_predictions",
        "validate_scores",
    ):
        setattr(module, name, call if name == function else forbidden)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        multimodal, "geo_helpfulness_relaxed_margin", module, raising=False
    )
    assert runner.main([command]) == 0
    expected = [runner.PATHS["output_dir"]]
    if command in ("select", "validate-selection"):
        expected += [
            runner.PATHS[k]
            for k in (
                "source_root",
                "protocol_root",
            )
        ]
    assert observed == [tuple(expected)]
    assert all(path.is_absolute() for path in observed[0])
    assert json.loads(capsys.readouterr().out)["command"] == command


def test_single_phase_exception_produces_only_stderr_json(monkeypatch, capsys):
    import multimodal

    module = ModuleType("multimodal.geo_helpfulness_relaxed_margin")

    def fail(*args):
        raise ValueError("source hash differs")

    for name in (
        "build_selection",
        "validate_selection",
        "build_predictions",
        "build_scores",
        "validate_predictions",
        "validate_scores",
    ):
        setattr(module, name, fail)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        multimodal, "geo_helpfulness_relaxed_margin", module, raising=False
    )
    assert runner.main(["score"]) == 1
    capture = capsys.readouterr()
    assert not capture.out
    assert json.loads(capture.err)["error"] == "source hash differs"


@pytest.mark.parametrize(
    "flag",
    [
        "--force",
        "--seed",
        "--family",
        "--learner",
        "--variant",
        "--threshold",
        "--temperature",
        "--test-dir",
        "--epochs",
        "--learning-rate",
    ],
)
def test_no_scientific_or_test_access_overrides(flag):
    with pytest.raises(SystemExit) as error:
        runner.main(["build", flag, "1"])
    assert error.value.code == 2


def _snapshot(root):
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _rewrite_json(path, value):
    path.chmod(0o644)
    path.write_text(json.dumps(value))


def _synthetic_bundle(tmp_path, monkeypatch, *, seal_selection=True):
    # These tests exercise orchestration, not real cohort dimensions or selection.
    # A separate fresh-process test below enforces the real capability guard.
    monkeypatch.setattr(workflow, "_fit_incapable", lambda **kwargs: None)
    root, source, protocol = (tmp_path / n for n in ("output", "source", "protocol"))
    original = {"kind": "finite", "comparison": "strict_gt", "value_hex": (0.45).hex()}
    relaxed = {"kind": "finite", "comparison": "strict_gt", "value_hex": (0.3).hex()}
    original_path = workflow._training_paths(protocol)["original"]
    workflow.write_json(
        original_path,
        {
            "heuristics": {
                workflow.ORIGINAL_POLICY: {"threshold_specification": original}
            }
        },
    )
    declaration = workflow._seal(
        {
            "schema_version": workflow.SCHEMA + ".declaration",
            "study_id": workflow.STUDY_ID,
            "output_root": str(root),
            "configuration": workflow.CONFIGURATION,
            "paths": {"source_root": str(source), "protocol_root": str(protocol)},
            "parent_file_sha256": {
                str(original_path): workflow.file_hash(original_path)
            },
            "identity_manifest": {"identity_projection_sha256": "b" * 64, "rows": []},
            "registry_root": str(tmp_path / "registry"),
            "ontology": [],
        },
        "declaration_sha256",
    )
    workflow.write_json(root / "declaration.json", declaration)
    policies = {workflow.ORIGINAL_POLICY: original, workflow.NEW_POLICY: relaxed}
    selection_values = {
        "selection.json": {
            "policies": {
                "original": {"threshold_specification": original},
                "relaxed": {"threshold_specification": relaxed},
            }
        },
        "frozen_policies.json": policies,
        "report.md": "Synthetic OOF selection only.\n",
    }
    selection = None
    if seal_selection:
        for name, value in selection_values.items():
            workflow._compare_or_write(root / "selection" / name, value, True)
        selection = workflow._publish(
            root / "selection", declaration, "selection", workflow.SELECTION_FILES
        )
    return SimpleNamespace(
        root=root,
        source=source,
        protocol=protocol,
        declaration=declaration,
        policies=policies,
        selection_values=selection_values,
        selection=selection,
    )


def _prediction_stub(bundle):
    # Deliberately tiny label-blind tables: full scientific checks belong to the
    # report and prediction tests, while this fixture isolates phase publication.
    return (
        bundle.selection,
        bundle.policies,
        {
            "experts.parquet": pa.table({"row_uid": ["synthetic-image"]}),
            "actions.parquet": pa.table({"policy_id": [workflow.NEW_POLICY]}),
            "metadata.json": {
                "study_id": workflow.STUDY_ID,
                "test_labels_decoded": False,
                "threshold_selection_performed": False,
            },
        },
    )


def test_study_configuration_and_artifact_contract_are_fixed():
    from multimodal import geo_helpfulness_relaxed_margin_report as report

    config = workflow.CONFIGURATION
    assert config["new_policy_count"] == 1
    assert config["harm_numerator"] == 7 and config["harm_denominator"] == 10
    assert config["minimum_plots"] == 20
    assert config["minimum_coverage_fraction"] == 0.01
    assert config["checkpoint_family"] == "best_model"
    assert config["threshold_selection_data"] == "development_training_oof_only"
    assert config["training_seeds"] == [1, 2, 3, 4]
    assert config["bootstrap_replicates"] == 2000
    assert config["bootstrap_seed"] == 20260826
    assert set(report.SCORE_FILES) == workflow.REPORT_FILES
    assert workflow.ORIGINAL_POLICY == report.ORIGINAL_POLICY
    assert workflow.NEW_POLICY == report.RELAXED_POLICY
    assert workflow.STUDY_ID == report.STUDY_ID
    assert "labels.parquet" not in workflow.PREDICTION_FILES
    assert "labels.parquet" not in workflow.SELECTION_FILES


@pytest.mark.parametrize("problem", ["unexpected", "symlink"])
def test_output_allowlist_rejects_unowned_files_and_symlinks(tmp_path, problem):
    root = tmp_path / "output"
    root.mkdir()
    if problem == "unexpected":
        (root / "unowned.json").write_text("{}")
    else:
        target = tmp_path / "other.json"
        target.write_text("{}")
        (root / "declaration.json").symlink_to(target)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError):
        workflow._allow_output(root)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "target,relation",
    [
        ("source", "same"),
        ("source", "child"),
        ("source", "parent"),
        ("protocol", "same"),
        ("protocol", "child"),
        ("protocol", "parent"),
    ],
)
def test_output_overlap_rejected_before_declaration_or_writes(
    tmp_path, monkeypatch, target, relation
):
    monkeypatch.setattr(workflow, "_fit_incapable", lambda **kwargs: None)
    source, protocol = tmp_path / "source", tmp_path / "protocol"
    protected = source if target == "source" else protocol
    root = {"same": protected, "child": protected / "nested", "parent": tmp_path}[
        relation
    ]

    def forbidden(*args):
        raise AssertionError("source metadata should not be opened")

    monkeypatch.setattr(workflow, "_declare", forbidden)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="overlaps"):
        workflow.build_selection(root, source, protocol)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("study_id", "other-study"),
        ("schema_version", "wrong-schema"),
        ("output_root", "/some-other-output"),
        ("configuration", {"harm_numerator": 9}),
    ],
)
def test_rehashed_declaration_cannot_change_study_contract(
    tmp_path, monkeypatch, field, value
):
    bundle = _synthetic_bundle(tmp_path, monkeypatch)
    changed = dict(bundle.declaration)
    changed.pop("declaration_sha256")
    changed[field] = value
    _rewrite_json(
        bundle.root / "declaration.json", workflow._seal(changed, "declaration_sha256")
    )
    before = _snapshot(bundle.root)
    with pytest.raises(ValueError, match="declaration differs"):
        workflow.load_declaration(bundle.root)
    assert _snapshot(bundle.root) == before


def test_parent_snapshot_tampering_rejected(tmp_path, monkeypatch):
    bundle = _synthetic_bundle(tmp_path, monkeypatch)
    original = workflow._training_paths(bundle.protocol)["original"]
    _rewrite_json(original, {"mutated": True})
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError):
        workflow.load_declaration(bundle.root)
    assert _snapshot(tmp_path) == before


def test_manifest_records_parquet_metadata_and_detects_content_tampering(tmp_path):
    folder = tmp_path / "phase"
    declaration = {"declaration_sha256": "c" * 64, "parent_file_sha256": {}}
    workflow._compare_or_write(folder / "table.parquet", pa.table({"x": [1, 2]}), True)
    workflow.write_json(folder / "metadata.json", {"label_blind": True})
    names = {"table.parquet", "metadata.json"}
    manifest = workflow._publish(folder, declaration, "predictions", names)
    assert manifest["artifacts"]["table.parquet"]["rows"] == 2
    assert "x: int64" in manifest["artifacts"]["table.parquet"]["arrow_schema"]
    assert workflow._manifest(folder, declaration, "predictions", names) == manifest
    _rewrite_json(folder / "metadata.json", {"label_blind": False})
    before = _snapshot(folder)
    with pytest.raises(ValueError, match="artifact changed"):
        workflow._manifest(folder, declaration, "predictions", names)
    assert _snapshot(folder) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("study_id", "other-study"),
        ("schema_version", "other-schema"),
        ("declaration_sha256", "f" * 64),
        ("artifacts", {}),
    ],
)
def test_rehashed_phase_manifest_identity_is_not_accepted(
    tmp_path, monkeypatch, field, value
):
    bundle = _synthetic_bundle(tmp_path, monkeypatch)
    changed = dict(bundle.selection)
    changed.pop("manifest_sha256")
    changed[field] = value
    _rewrite_json(bundle.root / "selection/manifest.json", workflow._seal(changed))
    with pytest.raises(ValueError, match="manifest identity"):
        workflow._manifest(
            bundle.root / "selection",
            bundle.declaration,
            "selection",
            workflow.SELECTION_FILES,
        )


def test_selection_publication_and_replay_are_immutable_and_label_blind(
    tmp_path, monkeypatch
):
    bundle = _synthetic_bundle(tmp_path, monkeypatch, seal_selection=False)
    monkeypatch.setattr(
        workflow, "_selection_values", lambda declaration: bundle.selection_values
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no Parquet reads are expected in mocked selection")

    monkeypatch.setattr(workflow.pq, "read_table", forbidden)
    args = (bundle.root, bundle.source, bundle.protocol)
    assert workflow.build_selection(*args)["status"] == "built"
    before = _snapshot(bundle.root)
    replay = workflow.build_selection(*args)
    assert replay["status"] == "reused_valid" and replay["test_data_decoded"] is False
    assert workflow.validate_selection(*args)["status"] == "validated"
    assert _snapshot(bundle.root) == before
    bundle.selection_values["report.md"] = "different reconstruction"
    with pytest.raises(ValueError, match="does not reproduce"):
        workflow.build_selection(*args)
    assert _snapshot(bundle.root) == before


def test_prediction_publication_replays_without_labels_or_reselection(
    tmp_path, monkeypatch
):
    bundle = _synthetic_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        workflow, "_prediction_values", lambda *args: _prediction_stub(bundle)
    )
    assert workflow.build_predictions(bundle.root)["status"] == "built"
    before = _snapshot(bundle.root)
    replay = workflow.build_predictions(bundle.root)
    assert replay["status"] == "reused_valid" and replay["test_labels_decoded"] is False
    assert workflow.validate_predictions(bundle.root)["status"] == "validated"
    assert _snapshot(bundle.root) == before
    assert not (bundle.root / "scores").exists()
    assert not (bundle.root / "predictions/labels.parquet").exists()


@pytest.mark.parametrize("phase", ["selection", "predictions"])
def test_missing_seal_prevents_feature_or_label_decoding(tmp_path, monkeypatch, phase):
    bundle = _synthetic_bundle(tmp_path, monkeypatch, seal_selection=False)
    reads = []

    def forbidden(*args, **kwargs):
        reads.append(str(args[0]))
        raise AssertionError("unsealed inputs must not be decoded")

    monkeypatch.setattr(workflow.pq, "read_table", forbidden)
    with pytest.raises(ValueError):
        if phase == "selection":
            workflow._prediction_values(bundle.root, bundle.declaration)
        else:
            workflow.build_scores(bundle.root)
    assert reads == []
    assert not (bundle.root / "scores").exists()


def test_saved_policy_mismatch_fails_before_expert_decoding(tmp_path, monkeypatch):
    bundle = _synthetic_bundle(tmp_path, monkeypatch, seal_selection=False)
    # Publish mutually inconsistent artifacts, so hashes alone are not sufficient.
    bundle.selection_values["frozen_policies.json"][workflow.NEW_POLICY] = {
        "kind": "finite",
        "comparison": "strict_gt",
        "value_hex": (0.2).hex(),
    }
    for name, value in bundle.selection_values.items():
        workflow._compare_or_write(bundle.root / "selection" / name, value, True)
    workflow._publish(
        bundle.root / "selection",
        bundle.declaration,
        "selection",
        workflow.SELECTION_FILES,
    )
    monkeypatch.setattr(
        workflow.pq,
        "read_table",
        lambda *a, **k: pytest.fail("expert decoding occurred"),
    )
    with pytest.raises(ValueError, match="policy differs from sealed selection"):
        workflow._prediction_values(bundle.root, bundle.declaration)


def test_score_reservation_is_checked_before_any_label_acquisition(
    tmp_path, monkeypatch
):
    from multimodal import geo_helpfulness_relaxed_margin_registry as registry

    bundle = _synthetic_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        workflow, "_prediction_values", lambda *args: _prediction_stub(bundle)
    )
    workflow.build_predictions(bundle.root)
    prediction_bytes = _snapshot(bundle.root / "predictions")
    events = []

    def reject(*args):
        events.append("reservation-validated")
        raise ValueError("synthetic reservation rejected")

    original_read = Path.read_bytes

    def guarded_read(path):
        if path.name == "labels.parquet":
            pytest.fail("labels acquired before registry validation")
        return original_read(path)

    monkeypatch.setattr(registry, "validate_reservation", reject)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(ValueError, match="synthetic reservation rejected"):
        workflow.build_scores(bundle.root)
    assert events == ["reservation-validated"]
    assert (bundle.root / "scores/build_declaration.json").exists()
    assert (bundle.root / "scores/reservation.json").exists()
    assert not (bundle.root / "scores/labels.parquet").exists()
    assert _snapshot(bundle.root / "predictions") == prediction_bytes


def test_score_label_decode_occurs_only_after_both_seals_and_reservation(
    tmp_path, monkeypatch
):
    from multimodal import geo_helpfulness_relaxed_margin_registry as registry

    bundle = _synthetic_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        workflow, "_prediction_values", lambda *args: _prediction_stub(bundle)
    )
    workflow.build_predictions(bundle.root)
    source_labels = bundle.source / "scores/labels.parquet"
    workflow._compare_or_write(source_labels, pa.table({"synthetic_label": [1]}), True)
    # Authenticate this synthetic parent label file before exercising the ordering.
    bundle.declaration.pop("declaration_sha256")
    bundle.declaration["parent_file_sha256"][str(source_labels)] = workflow.file_hash(
        source_labels
    )
    # Keep the original declaration hash/phase linkage by mocking only its loader;
    # the hash and all files above are separately exercised by the manifest tests.
    monkeypatch.setattr(
        workflow,
        "load_declaration",
        lambda root: {
            **bundle.declaration,
            "declaration_sha256": bundle.selection["declaration_sha256"],
        },
    )
    checked = []
    original_validate = registry.validate_reservation

    def validate(*args):
        original_validate(*args)
        checked.append(True)

    original_read = workflow.pq.read_table

    def stop_at_labels(path, *args, **kwargs):
        if Path(path).name == "labels.parquet":
            assert checked == [True]
            assert (bundle.root / "selection/manifest.json").exists()
            assert (bundle.root / "predictions/manifest.json").exists()
            assert (bundle.root / "scores/build_declaration.json").exists()
            assert (bundle.root / "scores/reservation.json").exists()
            assert (bundle.root / "scores/source_receipt.json").exists()
            assert Path(path).read_bytes() == source_labels.read_bytes()
            raise RuntimeError("synthetic stop at authorized label decode")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(registry, "validate_reservation", validate)
    monkeypatch.setattr(workflow.pq, "read_table", stop_at_labels)
    with pytest.raises(RuntimeError, match="authorized label decode"):
        workflow.build_scores(bundle.root)


def test_fresh_test_process_excludes_fitting_and_rejects_selector_imports():
    code = """
import sys
from types import ModuleType
from multimodal import geo_helpfulness_relaxed_margin as workflow
from multimodal import geo_helpfulness_relaxed_margin_report
workflow._fit_incapable(test_stage=True)
for name in ("torch", "sklearn", "scipy", "xgboost",
             "multimodal.geo_helpfulness_router_selection",
             "multimodal.geo_helpfulness_relaxed_margin_selection"):
    assert name not in sys.modules, name
    sys.modules[name] = ModuleType(name)
    try:
        workflow._fit_incapable(test_stage=True)
    except ValueError:
        pass
    else:
        raise AssertionError("test stage accepted " + name)
    finally:
        del sys.modules[name]
workflow._fit_incapable(test_stage=True)
print("fresh-process-boundaries-pass")
"""
    env = {
        **os.environ,
        "PYTHONMALLOC": "malloc",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = runner.subprocess.run(
        [sys.executable, "-c", code],
        cwd=runner.PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fresh-process-boundaries-pass"

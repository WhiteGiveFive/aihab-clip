"""Synthetic integrity/state-transition checks; no real study inputs are used."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal import geo_helpfulness_legacy_router_score as score
from multimodal import geo_helpfulness_legacy_router_report as report
from multimodal.geo_helpfulness_legacy_router_report import ONTOLOGY


def _fake_report(experts, routers, actions, labels):
    del routers, actions
    family = experts.checkpoint_family.iloc[0]
    summary = {
        name: {"mean": 0.5, "std": 0.0}
        for name in ("top1_accuracy", "top3_accuracy")
    }
    metric = {
        "mean_seed_net_utility": 0.0, "mean_seed_delta_accuracy": 0.0,
        "pooled": {"rescued_count": 0, "harmed_count": 0, "coverage_fraction": 0.0},
        "per_seed": {}, "seed_summary": {},
    }
    basic = {"checkpoint_family": family, "label_hash": score._label_hash(labels)}
    result = {
        name: pa.Table.from_pydict({"row_uid": labels.row_uid.to_list()})
        if name.endswith(".parquet") else dict(basic)
        for name in score.FAMILY_REPORTS
    }
    result["classification.json"] = {
        **basic,
        "experts": {name: {"seed_summary": summary} for name in score.EXPERTS.values()},
        "policies": {name: {"seed_summary": summary} for name in score.POLICIES},
    }
    result["policy_metrics.json"] = {name: dict(metric) for name in score.POLICIES}
    return result


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(score, "EXPECTED_PLOTS", 3)
    monkeypatch.setattr(score, "evaluate_family", _fake_report)
    # Other test modules may intentionally import fit-capable libraries. The
    # scorer's fresh-process boundary is checked independently below.
    monkeypatch.setattr(score, "_fit_incapable", lambda: None)
    output = tmp_path / "study"
    sources = tmp_path / "sources"
    sources.mkdir()
    frozen = sources / "frozen.txt"
    frozen.write_text("frozen development source\n")
    ontology = [
        {"dense_id": dense, "canonical_l3_id": canonical, "label_name": name}
        for dense, canonical, name in ONTOLOGY
    ]
    identity = [
        {
            "row_uid": f"{i:064x}", "file": f"photo_{i}.jpg", "file_lower": f"photo_{i}.jpg",
            "plot_idx": ("plot_a", "plot_a", "plot_b", "plot_c")[i],
            "normalized_plot_idx": ("plot_a", "plot_a", "plot_b", "plot_c")[i],
        }
        for i in range(4)
    ]
    y = np.asarray([0, 0, 1, 2], dtype=np.int8)
    label_frame = pd.DataFrame([
        {
            "row_uid": item["row_uid"], "file": item["file"], "plot_idx": item["plot_idx"],
            "label_id_dense": int(label), "canonical_l3_id": ontology[label]["canonical_l3_id"],
            "label_name": ontology[label]["label_name"],
        }
        for item, label in zip(identity, y)
    ], columns=score.LABEL_SCHEMA.names)
    caches = {}
    for seed in score.SEEDS:
        path = sources / f"cache{seed}.parquet"
        frame = label_frame[["file", "plot_idx", "label_name"]].copy()
        frame["forbidden_feature"] = "must never be projected by scorer"
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
        caches[str(seed)] = {"test": str(path)}
    records = []
    for family in score.FAMILIES:
        for item, label in zip(identity, y):
            for seed in score.SEEDS:
                record = {
                    **{key: item[key] for key in ("row_uid", "file", "plot_idx")},
                    "checkpoint_family": family, "training_seed": seed,
                }
                for expert in score.EXPERTS.values():
                    probability = np.full(18, 0.001, dtype=np.float64)
                    predicted = int(label) if family == "oracle_test_best_model" else (int(label) + seed) % 18
                    probability[predicted] = 0.983
                    record[f"{expert}_prob_native_t1"] = probability
                records.append(record)
    experts = pd.DataFrame(records).sort_values(["checkpoint_family", "row_uid", "training_seed"]).reset_index(drop=True)
    probabilities = {}
    for expert in score.EXPERTS.values():
        experts[f"{expert}_prob_calibrated"] = experts[f"{expert}_prob_native_t1"]
        probabilities[expert] = np.stack(experts[f"{expert}_prob_calibrated"].to_numpy())
    for name, values in report.build_semantic_values(probabilities).items():
        experts[name] = values
    routers, actions = [], []
    for variant in score.POLICIES[:3]:
        frame = experts[["checkpoint_family", "row_uid", "plot_idx", "training_seed"]].copy()
        frame["variant_id"], frame["C_hex"] = variant, (0.01).hex()
        q = np.tile([0.45, 0.15, 0.25, 0.15], (len(frame), 1))
        for index, name in enumerate(report.Q_COLUMNS):
            frame[name] = q[:, index]
        frame["score"] = q[:, 0] - q[:, 1]
        routers.append(frame)
    routers = pd.concat(routers, ignore_index=True)
    for policy in score.POLICIES:
        frame = experts[["checkpoint_family", "row_uid", "plot_idx", "training_seed"]].copy()
        if policy in score.POLICIES[:3]:
            values = routers.loc[routers.variant_id.eq(policy), "score"].to_numpy()
        elif policy == "never_intervene":
            values = np.zeros(len(frame), dtype=np.float64)
        elif policy == "always_geo_on_geo_raw_disagreement":
            values = np.ones(len(frame), dtype=np.float64)
        elif policy == "higher_calibrated_geo_confidence_on_geo_raw_disagreement":
            values = experts.geo_minus_raw_confidence.to_numpy()
        else:
            values = experts[policy].to_numpy()
        never = policy in (score.POLICIES[0], "never_intervene")
        specification = {
            "kind": "never_intervene" if never else "finite",
            "value_hex": None if never else (0.0).hex(), "comparison": "strict_gt",
        }
        frame["policy_id"], frame["score"] = policy, values
        frame["threshold_specification"] = json.dumps(specification, sort_keys=True)
        frame["raw_pred"], frame["geo_pred"] = experts.raw_pred, experts.geo_pred
        frame["acted"] = np.zeros(len(frame), dtype=bool) if never else (
            (values > 0) & (experts.raw_pred.to_numpy() != experts.geo_pred.to_numpy())
        )
        frame["final_pred"] = np.where(frame.acted, frame.geo_pred, frame.raw_pred)
        actions.append(frame)
    frames = {"experts": experts, "routers": routers, "actions": pd.concat(actions, ignore_index=True)}
    runs = {}
    for mode, expert in score.EXPERTS.items():
        runs[mode] = {}
        for seed in score.SEEDS:
            folder = sources / mode / f"seed{seed}"
            runs[mode][str(seed)] = str(folder)
            metrics = {}
            for family in score.FAMILIES:
                counts = score._expert_counts(experts, label_frame, family, seed, expert)
                metrics["test" if family == "best_model" else "oracle_test_best"] = {
                    "top1_acc": counts["top1_correct"] / 4,
                    "top3_acc": counts["top3_correct"] / 4,
                }
            score.write_json(folder / "metrics.json", metrics)
    declaration = {
        "study_id": score.STUDY_ID, "ontology": ontology,
        "identity_manifest": {
            "identity_projection_sha256": score.canonical_hash(identity), "rows": identity,
        },
        "source_snapshot": {str(frozen): score.file_hash(frozen)},
        "registry_root": str(tmp_path / "events"), "caches": caches, "legacy_runs": runs,
        "historical_metrics_sha256": {
            str(Path(folder) / "metrics.json"): score.file_hash(Path(folder) / "metrics.json")
            for seeds in runs.values() for folder in seeds.values()
        },
        "family_descriptions": {
            "best_model": "legacy final saved checkpoint",
            "oracle_test_best_model": "legacy test-selected biased diagnostic",
        },
        "user_authorization": "evaluate both saved families without changing frozen policies",
        "no_go_preserved": True,
    }
    declaration["declaration_sha256"] = score.canonical_hash(declaration)
    prediction = {"manifest_sha256": score.canonical_hash({"families": list(score.FAMILIES)})}
    score.write_json(output / "declaration.json", declaration)
    score.write_json(output / "predictions/manifest.json", prediction)
    state = {"sealed": True}

    def validate_predictions(root):
        assert root == output.resolve()
        if not state["sealed"]:
            raise ValueError("prediction manifest is not sealed")
        score.check_snapshot(declaration["source_snapshot"])
        return declaration, prediction, frames

    monkeypatch.setattr(score, "validate_predictions", validate_predictions)
    return output, declaration, prediction, frames, state


def _bytes_and_mtimes(root):
    return {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_build_seals_both_families_and_reuses_without_original_readers(study, monkeypatch):
    output, declaration, _, _, _ = study
    source_paths = {Path(spec["test"]) for spec in declaration["caches"].values()}
    observed = []
    original_read = score.pq.read_table

    def projected(path, *args, **kwargs):
        if Path(path) in source_paths:
            observed.append(Path(path))
            assert kwargs["columns"] == ["file", "plot_idx", "label_name"]
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(score.pq, "read_table", projected)
    result = score.build_score(output)
    assert result["status"] == "built"
    assert len(observed) == 4
    assert set(result["summary"]["checkpoint_families"]) == set(score.FAMILIES)
    receipt = score.read_json(output / "scores" / "receipt.json")
    assert receipt["adaptive_reuse"] is True
    assert receipt["reservation"]["prior_registered_score_event"] is False
    assert receipt["m4_no_go_preserved"] is True
    assert len(score.read_json(output / "scores/historical_replay.json")["records"]) == 24
    before = _bytes_and_mtimes(output)
    registry_before = _bytes_and_mtimes(Path(declaration["registry_root"]))

    def forbidden(*args, **kwargs):
        raise AssertionError("original source reader invoked on completed reuse")

    monkeypatch.setattr(score, "_read_original_labels", forbidden)
    monkeypatch.setattr(score, "_read_historical_replay", forbidden)
    assert score.validate_score(output)["status"] == "validated"
    assert score.build_score(output)["status"] == "reused_valid"
    assert _bytes_and_mtimes(output) == before
    assert _bytes_and_mtimes(Path(declaration["registry_root"])) == registry_before
    assert len(observed) == 4


def test_prediction_seal_is_required_before_any_label_or_metric_access(study, monkeypatch):
    output, _, _, _, state = study
    state["sealed"] = False

    def forbidden(*args, **kwargs):
        raise AssertionError("source accessed before predictions were sealed")

    monkeypatch.setattr(score, "_read_original_labels", forbidden)
    monkeypatch.setattr(score, "_read_historical_replay", forbidden)
    with pytest.raises(ValueError, match="not sealed"):
        score.build_score(output)
    assert not (output / "scores").exists()


def test_global_reservation_crash_resumes_as_the_same_owned_event(study, monkeypatch):
    output, declaration, prediction, _, _ = study
    original_write = score.write_json
    failures = []

    def fail_after_first(path, value):
        if path.parent.name == "bundles" and not failures:
            failures.append(path)
            raise RuntimeError("crash between global first-event and bundle reservation")
        return original_write(path, value)

    monkeypatch.setattr(score, "write_json", fail_after_first)
    with pytest.raises(RuntimeError, match="crash"):
        score.build_score(output)
    build = score._build_declaration(output, declaration, prediction)
    paths = score._registry_paths(declaration, build)
    assert paths["first"].exists()
    assert not paths["bundle"].exists()
    assert not (output / "scores/labels.parquet").exists()
    first_before = (paths["first"].read_bytes(), paths["first"].stat().st_mtime_ns)
    monkeypatch.setattr(score, "write_json", original_write)
    assert score.build_score(output)["status"] == "built"
    assert first_before == (paths["first"].read_bytes(), paths["first"].stat().st_mtime_ns)
    assert score.read_json(paths["bundle"])["prior_registered_score_event"] is False


def test_prior_registered_event_is_separate_from_historical_adaptive_reuse(study):
    output, declaration, prediction, _, _ = study
    build = score._build_declaration(output, declaration, prediction)
    paths = score._registry_paths(declaration, build)
    score.write_json(paths["first"], {"composite_bundle_file_sha256": "a" * 64})
    score.build_score(output)
    reservation = score.read_json(paths["bundle"])
    assert reservation["adaptive_reuse"] is True
    assert reservation["prior_registered_score_event"] is True


@pytest.mark.parametrize("target", ["summary.json", "labels.parquet", "historical_replay.json"])
def test_sealed_score_tampering_fails_closed(study, target):
    output, _, _, _, _ = study
    score.build_score(output)
    path = output / "scores" / target
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hashes changed"):
        score.validate_score(output)


def test_unknown_partial_files_are_never_overwritten(study):
    output, _, _, _, _ = study
    root = output / "scores"
    root.mkdir(parents=True)
    unknown = root / "unknown.json"
    unknown.write_text("user data")
    with pytest.raises(ValueError, match="allowlist"):
        score.build_score(output)
    assert unknown.read_text() == "user data"


def test_historical_accuracy_mismatch_stops_before_score_manifest(study):
    output, declaration, _, _, _ = study
    path = Path(declaration["legacy_runs"]["image_only"]["1"]) / "metrics.json"
    value = score.read_json(path)
    value["test"]["top1_acc"] = 1.0
    path.chmod(0o644)
    path.unlink()
    score.write_json(path, value)
    declaration["historical_metrics_sha256"][str(path)] = score.file_hash(path)
    with pytest.raises(ValueError, match="historical expert reproduction failed"):
        score.build_score(output)
    assert (output / "scores/labels.parquet").exists()
    assert not (output / "scores/historical_replay.json").exists()
    assert not (output / "scores/manifest.json").exists()


def test_four_test_label_projections_must_agree(study):
    output, declaration, _, _, _ = study
    path = Path(declaration["caches"]["4"]["test"])
    table = pq.read_table(path).to_pandas()
    table.loc[2, "label_name"] = ONTOLOGY[2][2]
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), path)
    with pytest.raises(ValueError, match="differ between the four"):
        score.build_score(output)
    assert not (output / "scores/labels.parquet").exists()
    assert not (output / "scores/manifest.json").exists()


def test_real_family_reports_round_trip_through_score_bundle(study, monkeypatch):
    output, _, _, _, _ = study
    monkeypatch.setattr(score, "evaluate_family", report.evaluate_family)
    result = score.build_score(output)
    assert result["status"] == "built"
    summary = result["summary"]
    assert summary["checkpoint_families"]["best_model"]["classification_seed_summary"]["raw"]["top1_accuracy"]["mean"] == 0
    assert summary["checkpoint_families"]["oracle_test_best_model"]["classification_seed_summary"]["raw"]["top1_accuracy"]["mean"] == 1
    for family in score.FAMILIES:
        for artifact in score.FAMILY_REPORTS:
            assert (output / "scores" / family / artifact).is_file()
        primary = summary["checkpoint_families"][family]["policies"][score.POLICIES[0]]
        assert primary["pooled"]["coverage_fraction"] == 0
    assert score.validate_score(output)["status"] == "validated"


def test_fresh_scorer_process_has_no_fitting_imports_and_rejects_them():
    code = """
import sys
from multimodal.geo_helpfulness_legacy_router_score import _fit_incapable
_fit_incapable()
sys.modules['sklearn'] = object()
try:
    _fit_incapable()
except ValueError as error:
    assert 'fresh process' in str(error)
else:
    raise AssertionError('fit-capable scorer process was accepted')
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_owned_partial_source_seal_resumes_without_original_readers(study, monkeypatch):
    output, _, _, _, _ = study

    def crash(*args, **kwargs):
        raise RuntimeError("crash after the source acquisition seal")

    monkeypatch.setattr(score, "evaluate_family", crash)
    with pytest.raises(RuntimeError, match="source acquisition"):
        score.build_score(output)
    assert (output / "scores/source_receipt.json").exists()
    assert not (output / "scores/manifest.json").exists()

    def forbidden(*args, **kwargs):
        raise AssertionError("sealed source copy should be reused")

    monkeypatch.setattr(score, "_read_original_labels", forbidden)
    monkeypatch.setattr(score, "_read_historical_replay", forbidden)
    monkeypatch.setattr(score, "evaluate_family", _fake_report)
    result = score.build_score(output)
    assert result["status"] == "built"
    assert result["original_label_columns_opened"] is False
    assert result["original_historical_metrics_opened"] is False


def test_owned_partial_source_copy_tampering_is_rejected(study, monkeypatch):
    output, _, _, _, _ = study

    def crash(*args, **kwargs):
        raise RuntimeError("crash after source acquisition")

    monkeypatch.setattr(score, "evaluate_family", crash)
    with pytest.raises(RuntimeError, match="source acquisition"):
        score.build_score(output)
    path = output / "scores/labels.parquet"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="acquisition receipt"):
        score.build_score(output)

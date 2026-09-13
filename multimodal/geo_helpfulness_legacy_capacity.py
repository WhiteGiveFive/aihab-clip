"""Immutable, apply-only transfer of selected RF/XGB/MLP legacy-test policies."""

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from multimodal import geo_helpfulness_legacy_router_common as legacy
from multimodal import geo_helpfulness_legacy_router_score as old_score
from multimodal.geo_helpfulness_router_capacity import _allowlist, _files
from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    canonical_hash,
    check_self_hash,
    check_snapshot,
    compare_tables,
    exclusive_lock,
    file_hash,
    read_json,
    table_bytes,
    write_bytes,
    write_json,
)

STUDY_ID = "legacy_capacity_router_transfer_v1"
SCHEMA = "geo_helpfulness.legacy_capacity.v1"
BASE = PROJECT_ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
DEFAULT_OUTPUT = BASE / "exploratory_legacy_capacity_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/multimodal_geo_legacy_capacity_v1.json"
PREDICTION_FILES = {
    "experts.parquet",
    "routers.parquet",
    "actions.parquet",
    "metadata.json",
}
FAMILY_FILES = {*old_score.FAMILY_REPORTS, "summary.json", "paired_comparisons.json"}
SCORE_FILES = {
    "build_declaration.json",
    "reservation.json",
    "labels.parquet",
    "source_receipt.json",
    "summary.json",
    "report.md",
    "receipt.json",
    *[f"{family}/{name}" for family in legacy.FAMILIES for name in FAMILY_FILES],
}
CODE_FILES = (
    "multimodal/geo_helpfulness_legacy_capacity.py",
    "multimodal/geo_helpfulness_legacy_capacity_predictions.py",
    "multimodal/geo_helpfulness_legacy_capacity_report.py",
    "multimodal/geo_helpfulness_legacy_capacity_registry.py",
    "tools/run_multimodal_geo_helpfulness_legacy_capacity.py",
)
CONFIGURATION = {
    "schema_version": "geo_helpfulness.legacy_capacity_config.v1",
    "study_id": STUDY_ID,
    "checkpoint_families": list(legacy.FAMILIES),
    "training_seeds": [1, 2, 3, 4],
    "new_learners": ["rf", "xgb", "mlp"],
    "feature_variants": list(legacy.VARIANTS),
    "new_selected_model_count": 36,
    "router_policy_count": 12,
    "total_policy_count": 17,
    "unique_images": 1347,
    "unique_plots": 531,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20260826,
    "paired_contrasts_per_checkpoint_family": 27,
    "fitting_permitted": False,
    "selection_permitted": False,
    "recalibration_permitted": False,
    "threshold_search_permitted": False,
    "m4_no_go_preserved": True,
    "adaptive_test_reuse": True,
    "replay_atol": 1e-12,
    "replay_rtol": 1e-12,
    "probability_normalization_atol": 1e-8,
}


def _seal(value, key):
    return {**value, key: canonical_hash(value)}


def _fit_incapable():
    old_score._fit_incapable()
    forbidden = (
        "xgboost",
        "multimodal.geo_helpfulness_router_capacity_fit",
        "multimodal.geo_helpfulness_router_capacity_inputs",
        "multimodal.geo_helpfulness_router_capacity_score",
        "multimodal.geo_helpfulness_router_mlp_fit",
        "multimodal.geo_helpfulness_router_mlp_score",
    )
    if any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in forbidden
    ):
        raise ValueError(
            "legacy capacity transfer prohibits fitting/selection capabilities"
        )


def _legacy_metadata(root):
    """Authenticate old labels/results as bytes, without decoding their contents."""
    declaration = legacy.load_declaration(root)
    prediction = read_json(root / "predictions/manifest.json")
    check_self_hash(prediction)
    if (
        prediction.get("study_id") != legacy.STUDY_ID
        or prediction.get("declaration_file_sha256")
        != file_hash(root / "declaration.json")
        or prediction.get("source_snapshot") != declaration["source_snapshot"]
        or set(prediction["artifacts"]) != legacy.PREDICTION_FILES
    ):
        raise ValueError("legacy prediction parent lineage mismatch")
    for name, record in prediction["artifacts"].items():
        if file_hash(root / "predictions" / name) != record["file_sha256"]:
            raise ValueError("legacy prediction parent changed")
    manifest = read_json(root / "scores/manifest.json")
    check_self_hash(manifest)
    old_score._allowlist(root / "scores", complete=True)
    build = manifest["build_declaration"]
    if (
        manifest.get("study_id") != legacy.STUDY_ID
        or manifest.get("schema_version") != old_score.SCHEMA
        or build["prediction_manifest_sha256"] != prediction["manifest_sha256"]
        or build["declaration_sha256"] != declaration["declaration_sha256"]
        or set(manifest["artifact_file_sha256"]) != old_score.SCORE_FILES
    ):
        raise ValueError("legacy scored parent lineage mismatch")
    for name, digest in manifest["artifact_file_sha256"].items():
        if file_hash(root / "scores" / name) != digest:
            raise ValueError("legacy scored parent changed")
    if read_json(root / "scores/source_receipt.json") != old_score._source_receipt(
        root / "scores", build
    ):
        raise ValueError("legacy label acquisition receipt differs")
    receipt = read_json(root / "scores/receipt.json")
    check_self_hash(receipt, "receipt_sha256")
    if receipt["receipt_sha256"] != manifest["receipt_sha256"]:
        raise ValueError("legacy score receipt mismatch")
    old_score._validate_registry(
        declaration,
        build,
        read_json(root / "scores/reservation.json"),
        old_score._event(receipt, root / "scores/receipt.json"),
    )
    allowed = {
        "declaration.json",
        "predictions/manifest.json",
        "scores/manifest.json",
        *["predictions/" + n for n in legacy.PREDICTION_FILES],
        *["scores/" + n for n in old_score.SCORE_FILES],
    }
    # The predecessor's prediction-stage lock lives beside its two bundles.
    lock_path = root / ".predictions.lock"
    if lock_path.exists():
        if (
            lock_path.is_symlink()
            or not lock_path.is_file()
            or lock_path.stat().st_size
        ):
            raise ValueError("legacy prediction lock must be an empty regular file")
        allowed.add(".predictions.lock")
    _allowlist(root, allowed, complete=True)
    return declaration, prediction, manifest


def _allow_output(root):
    _allowlist(
        root,
        {
            "declaration.json",
            "predictions/manifest.json",
            "scores/manifest.json",
            *["predictions/" + n for n in PREDICTION_FILES],
            *["scores/" + n for n in SCORE_FILES],
        },
    )


def _configuration(path):
    value = read_json(path)
    if value != CONFIGURATION or canonical_hash(value) != canonical_hash(CONFIGURATION):
        raise ValueError(
            "legacy capacity configuration differs from fixed application contract"
        )
    return value


def _declare(
    root, paths, config_path, selection, old_declaration, old_prediction, old_manifest
):
    if _files(root):
        raise ValueError("unowned legacy capacity output is not empty")
    snapshot = dict(selection["parent_file_sha256"])
    snapshot.update(old_declaration["source_snapshot"])
    for parent in paths.values():
        snapshot.update(
            {str(parent / name): file_hash(parent / name) for name in _files(parent)}
        )
    snapshot.update(
        {
            str(PROJECT_ROOT / name): file_hash(PROJECT_ROOT / name)
            for name in CODE_FILES
        }
    )
    snapshot[str(config_path)] = file_hash(config_path)
    old_build = old_manifest["build_declaration"]
    for name, path in old_score._registry_paths(old_declaration, old_build).items():
        if name != "root":
            snapshot[str(path)] = file_hash(path)
    declaration = _seal(
        {
            "schema_version": SCHEMA + ".declaration",
            "study_id": STUDY_ID,
            "declared_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": _configuration(config_path),
            "paths": {name: str(path) for name, path in paths.items()},
            "config_path": str(config_path),
            "parent_file_sha256": snapshot,
            "legacy_declaration_sha256": old_declaration["declaration_sha256"],
            "legacy_prediction_manifest_sha256": old_prediction["manifest_sha256"],
            "legacy_score_manifest_sha256": old_manifest["manifest_sha256"],
            "router_parent_lineage": selection["lineage"],
            "frozen_policies": selection["policies"],
            "identity_manifest": old_declaration["identity_manifest"],
            "ontology": old_declaration["ontology"],
            "registry_root": old_declaration["registry_root"],
            "family_descriptions": old_declaration["family_descriptions"],
            "preprocessing": "unchanged_sealed_legacy_logits_M4_temperatures_and_transforms",
            "user_authorization": "Evaluate sealed RF/XGB/MLP on cleaned-test legacy experts as for logistic regression, including all existing feature ablations and both prior checkpoint families.",
            "fitting_permitted": False,
            "selection_permitted": False,
            "historical_test_exposure": True,
            "adaptive_reuse": True,
            "m4_no_go_preserved": True,
            "independent_confirmation": False,
            "environment": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "pyarrow")
            },
            "python": sys.version,
        },
        "declaration_sha256",
    )
    check_snapshot(snapshot)
    write_json(root / "declaration.json", declaration)
    return declaration


def load_declaration(root):
    root = Path(root).resolve()
    value = read_json(root / "declaration.json")
    check_self_hash(value, "declaration_sha256")
    if (
        value.get("study_id") != STUDY_ID
        or value.get("schema_version") != SCHEMA + ".declaration"
        or value.get("configuration") != CONFIGURATION
    ):
        raise ValueError("legacy capacity declaration mismatch")
    check_snapshot(value["parent_file_sha256"])
    _allow_output(root)
    return value


def _manifest(root, declaration, kind, expected):
    value = read_json(root / "manifest.json")
    check_self_hash(value)
    if (
        value.get("schema_version") != SCHEMA + "." + kind
        or value.get("declaration_sha256") != declaration["declaration_sha256"]
        or set(value["artifacts"]) != expected
    ):
        raise ValueError("legacy capacity manifest mismatch")
    _allowlist(root, expected | {"manifest.json"}, complete=True)
    for name, record in value["artifacts"].items():
        path = root / name
        if (
            file_hash(path) != record["file_sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"legacy capacity artifact changed: {name}")
    return value


def _publish(root, declaration, kind, expected, extra):
    check_snapshot(declaration["parent_file_sha256"])
    _allowlist(root, expected, complete=True)
    artifacts = {}
    for name in sorted(expected):
        path = root / name
        record = {"file_sha256": file_hash(path), "bytes": path.stat().st_size}
        if name.endswith(".parquet"):
            info = pq.ParquetFile(path)
            record.update(
                rows=info.metadata.num_rows,
                columns=len(info.schema_arrow),
                arrow_schema=str(info.schema_arrow),
            )
        artifacts[name] = record
    value = _seal(
        {
            "schema_version": SCHEMA + "." + kind,
            "study_id": STUDY_ID,
            "declaration_sha256": declaration["declaration_sha256"],
            "artifacts": artifacts,
            **extra,
        },
        "manifest_sha256",
    )
    write_json(root / "manifest.json", value)
    return value


def _inputs(declaration):
    from multimodal.geo_helpfulness_legacy_capacity_predictions import (
        load_selected_policies,
    )

    paths = {k: Path(v) for k, v in declaration["paths"].items()}
    old, old_prediction, old_manifest = _legacy_metadata(paths["legacy_root"])
    if (
        old["declaration_sha256"] != declaration["legacy_declaration_sha256"]
        or old_prediction["manifest_sha256"]
        != declaration["legacy_prediction_manifest_sha256"]
        or old_manifest["manifest_sha256"]
        != declaration["legacy_score_manifest_sha256"]
    ):
        raise ValueError("legacy parent differs from new declaration")
    selected = load_selected_policies(
        paths["capacity_root"], paths["mlp_source_root"], paths["mlp_replay_root"], old
    )
    if (
        selected["policies"] != declaration["frozen_policies"]
        or selected["lineage"] != declaration["router_parent_lineage"]
    ):
        raise ValueError("selected router parent differs from new declaration")
    # The caller must have committed the new declaration before this decode.
    _, _, frames = legacy.validate_predictions(paths["legacy_root"])
    return old, selected, frames


def _metadata(tables):
    expected = {"experts": 10776, "routers": 129312, "actions": 183192}
    if set(tables) != set(expected) or any(
        tables[k].num_rows != n for k, n in expected.items()
    ):
        raise ValueError("legacy capacity prediction cross-product mismatch")
    return {
        "study_id": STUDY_ID,
        "table_rows": expected,
        "checkpoint_families": list(legacy.FAMILIES),
        "new_model_count": 36,
        "router_policy_count": 12,
        "policy_count": 17,
        "test_labels_decoded": False,
        "fitting_performed": False,
        "selection_performed": False,
    }


def _reconstruct(declaration):
    from multimodal.geo_helpfulness_legacy_capacity_predictions import (
        reconstruct_predictions,
    )

    old, selection, frames = _inputs(declaration)
    return reconstruct_predictions(frames, selection, old)


def _validate_predictions(root):
    declaration = load_declaration(root)
    manifest = _manifest(
        root / "predictions", declaration, "predictions", PREDICTION_FILES
    )
    tables = _reconstruct(declaration)
    for name, table in tables.items():
        compare_tables(pq.read_table(root / "predictions" / (name + ".parquet")), table)
    if read_json(root / "predictions/metadata.json") != _metadata(tables):
        raise ValueError("legacy capacity prediction metadata differs")
    check_snapshot(declaration["parent_file_sha256"])
    return declaration, manifest, {k: v.to_pandas() for k, v in tables.items()}


def validate_predictions(output_root):
    _fit_incapable()
    root = Path(output_root).resolve()
    with exclusive_lock(root):
        _, manifest, _ = _validate_predictions(root)
    return {
        "status": "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "fitting_performed": False,
    }


def build_predictions(
    output_root,
    legacy_root,
    capacity_root,
    mlp_source_root,
    mlp_replay_root,
    config_path=DEFAULT_CONFIG,
):
    _fit_incapable()
    from multimodal.geo_helpfulness_legacy_capacity_predictions import (
        load_selected_policies,
    )

    root, config_path = Path(output_root).resolve(), Path(config_path).resolve()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "legacy_root": legacy_root,
            "capacity_root": capacity_root,
            "mlp_source_root": mlp_source_root,
            "mlp_replay_root": mlp_replay_root,
        }.items()
    }
    old, old_prediction, old_manifest = _legacy_metadata(paths["legacy_root"])
    protected = [
        *paths.values(),
        Path(old["protocol_root"]).resolve(),
        Path(old["registry_root"]).resolve(),
    ]
    if any(
        root == parent or parent in root.parents or root in parent.parents
        for parent in protected
    ):
        raise ValueError(
            "new transfer root must be disjoint from all frozen source roots"
        )
    _configuration(config_path)
    with exclusive_lock(root):
        if (root / "declaration.json").exists():
            declaration = load_declaration(root)
            if declaration["paths"] != {
                k: str(v) for k, v in paths.items()
            } or declaration["config_path"] != str(config_path):
                raise ValueError("repeat build source paths differ")
            if (root / "predictions/manifest.json").exists():
                _, manifest, _ = _validate_predictions(root)
                return {
                    "status": "reused_valid",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "fitting_performed": False,
                }
        else:
            selection = load_selected_policies(
                paths["capacity_root"],
                paths["mlp_source_root"],
                paths["mlp_replay_root"],
                old,
            )
            declaration = _declare(
                root, paths, config_path, selection, old, old_prediction, old_manifest
            )
        _allowlist(root / "predictions", PREDICTION_FILES)
        tables = _reconstruct(declaration)
        for name, table in tables.items():
            write_bytes(root / "predictions" / (name + ".parquet"), table_bytes(table))
        write_json(root / "predictions/metadata.json", _metadata(tables))
        manifest = _publish(
            root / "predictions",
            declaration,
            "predictions",
            PREDICTION_FILES,
            {"labels_decoded": False},
        )
        _validate_predictions(root)
    return {
        "status": "built",
        "manifest_sha256": manifest["manifest_sha256"],
        "fitting_performed": False,
    }


def _build_record(root, declaration, prediction):
    return {
        "schema_version": SCHEMA + ".score_build",
        "study_id": STUDY_ID,
        "owner_output_root": str(root),
        "declaration_sha256": declaration["declaration_sha256"],
        "prediction_manifest_sha256": prediction["manifest_sha256"],
        "composite_bundle_file_sha256": prediction["manifest_sha256"],
        "identity_projection_sha256": declaration["identity_manifest"][
            "identity_projection_sha256"
        ],
        "ontology_sha256": canonical_hash(declaration["ontology"]),
        "historical_test_exposure": True,
        "adaptive_reuse": True,
        "m4_no_go_preserved": True,
    }


def _source_receipt(root, declaration, build):
    source = Path(declaration["paths"]["legacy_root"]) / "scores/labels.parquet"
    own = root / "scores/labels.parquet"
    if file_hash(source) != file_hash(own):
        raise ValueError("copied legacy labels are not byte-identical")
    return {
        "study_id": STUDY_ID,
        "build_sha256": canonical_hash(build),
        "reservation_file_sha256": file_hash(root / "scores/reservation.json"),
        "parent_label_file": str(source),
        "parent_label_file_sha256": file_hash(source),
        "label_copy_file_sha256": file_hash(own),
        "original_label_cache_opened": False,
        "original_historical_metrics_opened": False,
        "parent_legacy_score_manifest_sha256": declaration[
            "legacy_score_manifest_sha256"
        ],
    }


def _family_reports(frames, labels, declaration):
    from multimodal.geo_helpfulness_legacy_capacity_report import evaluate_family

    results = {}
    for family in legacy.FAMILIES:
        inputs = [
            frames[k].loc[frames[k].checkpoint_family.eq(family)].reset_index(drop=True)
            for k in ("experts", "routers", "actions")
        ]
        reports = evaluate_family(*inputs, labels, declaration["frozen_policies"])
        if set(reports) != FAMILY_FILES:
            raise ValueError("legacy capacity family report allowlist mismatch")
        results[family] = reports
    return results


def _summary(results):
    return {
        "study_id": STUDY_ID,
        "unique_images": 1347,
        "unique_plots": 531,
        "training_seeds": [1, 2, 3, 4],
        "checkpoint_families": {f: r["summary.json"] for f, r in results.items()},
        "adaptive_reuse": True,
        "independent_confirmation": False,
        "m4_no_go_preserved": True,
        "selection_or_fitting_performed": False,
        "oracle_family_interpretation": "test_selected_biased_diagnostic_not_oracle_ceiling",
    }


def _report(results):
    lines = [
        "# Frozen RF/XGB/MLP transfer to legacy experts",
        "",
        "Descriptive cleaned-test reuse; oracle checkpoints are test-selected biased diagnostics.",
        "No fitting, recalibration, threshold search, model selection, or M4 gate rerun.",
        "",
        "Top-1 and top-3 are mean percentages over four seeds; counts pool seed-image realizations.",
        "",
        "| Checkpoint family | Classifier/policy | Top-1 (%) | Top-3 (%) | Gain (pp) | Actions | Rescues | Harms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, reports in results.items():
        classification = reports["classification.json"]
        for name, metrics in classification["experts"].items():
            s = metrics["seed_summary"]
            lines.append(
                f"| {family} | {name} | {100*s['top1_accuracy']['mean']:.6f} | {100*s['top3_accuracy']['mean']:.6f} | — | — | — | — |"
            )
        for name, metrics in classification["policies"].items():
            s = metrics["seed_summary"]
            p = reports["policy_metrics.json"][name]
            q = p["pooled"]
            lines.append(
                f"| {family} | {name} | {100*s['top1_accuracy']['mean']:.6f} | {100*s['top3_accuracy']['mean']:.6f} | {100*p['mean_seed_delta_accuracy']:.6f} | {q['acted_count']} | {q['rescued_count']} | {q['harmed_count']} |"
            )
    return "\n".join(lines) + "\n"


def _score_values(root, declaration, prediction, build, reservation, labels, results):
    values = {
        f"{family}/{name}": v
        for family, reports in results.items()
        for name, v in reports.items()
    }
    values["summary.json"] = _summary(results)
    values["report.md"] = _report(results)
    return values


def _compare_or_write(path, value, build):
    if path.exists():
        if isinstance(value, pa.Table):
            compare_tables(pq.read_table(path), value)
        elif isinstance(value, str):
            if path.read_text() != value:
                raise ValueError(f"report does not reproduce: {path.name}")
        elif read_json(path) != value:
            raise ValueError(f"JSON artifact does not reproduce: {path.name}")
    elif not build:
        raise ValueError(f"missing committed artifact: {path.name}")
    elif isinstance(value, pa.Table):
        write_bytes(path, table_bytes(value))
    elif isinstance(value, str):
        write_bytes(path, value.encode())
    else:
        write_json(path, value)


def _scores(output_root, *, build):
    _fit_incapable()
    from multimodal.geo_helpfulness_legacy_capacity_registry import (
        reserve_event,
        validate_reservation,
        finish_event,
        validate_event,
    )

    root = Path(output_root).resolve()
    with exclusive_lock(root):
        declaration, prediction, frames = _validate_predictions(root)
        score = root / "scores"
        committed = (score / "manifest.json").exists()
        manifest = (
            _manifest(score, declaration, "scores", SCORE_FILES) if committed else None
        )
        if not committed and not build:
            raise ValueError("legacy capacity score bundle is not committed")
        _allowlist(score, SCORE_FILES | ({"manifest.json"} if committed else set()))
        build_record = _build_record(root, declaration, prediction)
        registry = Path(declaration["registry_root"])
        if committed:
            if read_json(score / "build_declaration.json") != build_record:
                raise ValueError("score build declaration differs")
            reservation = read_json(score / "reservation.json")
        else:
            if _files(score) and not (score / "build_declaration.json").exists():
                raise ValueError("unowned partial score outputs")
            write_json(score / "build_declaration.json", build_record)
            reservation = reserve_event(registry, build_record)
            write_json(score / "reservation.json", reservation)
        validate_reservation(registry, build_record, reservation)
        # Decode labels only after both the full prediction seal and reservation.
        source = Path(declaration["paths"]["legacy_root"]) / "scores/labels.parquet"
        if not (score / "labels.parquet").exists():
            if committed:
                raise ValueError("committed label copy missing")
            write_bytes(score / "labels.parquet", source.read_bytes())
        expected_source = _source_receipt(root, declaration, build_record)
        _compare_or_write(score / "source_receipt.json", expected_source, build)
        table = pq.read_table(score / "labels.parquet")
        if table.schema != old_score.LABEL_SCHEMA:
            raise ValueError("label physical schema differs")
        labels = old_score._validate_labels(table.to_pandas(), declaration)
        results = _family_reports(frames, labels, declaration)
        values = _score_values(
            root, declaration, prediction, build_record, reservation, labels, results
        )
        for name, value in values.items():
            _compare_or_write(score / name, value, build)
        hashes = {
            name: file_hash(score / name)
            for name in sorted(SCORE_FILES - {"receipt.json"})
        }
        receipt = _seal(
            {
                **build_record,
                "artifact_role": "legacy_capacity_transfer_score_receipt",
                "reservation": reservation,
                "ordered_scoring_label_sha256": old_score._label_hash(labels),
                "output_file_sha256": hashes,
                "evaluation_event_id": canonical_hash(
                    {
                        "study_id": STUDY_ID,
                        "prediction": prediction["manifest_sha256"],
                        "identity": build_record["identity_projection_sha256"],
                        "labels": old_score._label_hash(labels),
                    }
                ),
                "fitting": False,
                "selection": False,
                "threshold_search": False,
                "recalibration": False,
                "independent_confirmation": False,
            },
            "receipt_sha256",
        )
        _compare_or_write(score / "receipt.json", receipt, build)
        check_snapshot(declaration["parent_file_sha256"])
        if not committed:
            finish_event(
                registry, build_record, reservation, receipt, score / "receipt.json"
            )
            manifest = _publish(
                score,
                declaration,
                "scores",
                SCORE_FILES,
                {
                    "prediction_manifest_sha256": prediction["manifest_sha256"],
                    "receipt_sha256": receipt["receipt_sha256"],
                },
            )
        if (
            manifest["prediction_manifest_sha256"] != prediction["manifest_sha256"]
            or manifest["receipt_sha256"] != receipt["receipt_sha256"]
        ):
            raise ValueError("score receipt/manifest linkage differs")
        validate_event(
            registry, build_record, reservation, receipt, score / "receipt.json"
        )
        check_snapshot(declaration["parent_file_sha256"])
        _fit_incapable()
    return {
        "status": (
            ("reused_valid" if committed else "built") if build else "validated"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "evaluation_event_id": receipt["evaluation_event_id"],
        "fitting_performed": False,
        "original_label_cache_opened": False,
        "m4_no_go_preserved": True,
    }


def build_scores(output_root):
    return _scores(output_root, build=True)


def validate_scores(output_root):
    return _scores(output_root, build=False)

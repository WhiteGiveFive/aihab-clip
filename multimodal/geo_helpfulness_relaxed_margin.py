"""One OOF-selected margin policy, sealed before reused cleaned-test scoring."""

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal import geo_helpfulness_relaxed_rf as parent
from multimodal import geo_helpfulness_legacy_router_report as legacy_report
from multimodal import geo_helpfulness_legacy_router_score as legacy_score
from multimodal.geo_helpfulness_legacy_capacity import _compare_or_write
from multimodal.geo_helpfulness_router_capacity import _allowlist, _files
from multimodal.geo_helpfulness_router_heuristics import (
    TARGET_COLUMNS,
    apply_heuristic_threshold,
    validate_threshold_specification,
)
from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    DEFAULT_ROOT,
    SEEDS,
    canonical_bytes,
    canonical_hash,
    check_self_hash,
    check_snapshot,
    compare_tables,
    exclusive_lock,
    file_hash,
    read_json,
    write_bytes,
    write_json,
)

STUDY_ID = "relaxed_guardrails_margin_v1"
SCHEMA = "geo_helpfulness.relaxed_margin.v1"
BASE = PROJECT_ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
DEFAULT_OUTPUT = BASE / STUDY_ID
DEFAULT_SOURCE = BASE / "relaxed_guardrails_rf_v1"
ORIGINAL_POLICY = "geo_minus_raw_margin"
NEW_POLICY = "relaxed__geo_minus_raw_margin__plots_20__harm_0p7"
CONFIGURATION = {
    "study_id": STUDY_ID,
    "score": "calibrated_geo_top2_margin_minus_calibrated_raw_top2_margin",
    "threshold_selection_data": "development_training_oof_only",
    "new_policy_count": 1,
    "harm_numerator": 7,
    "harm_denominator": 10,
    "minimum_plots": 20,
    "minimum_coverage_fraction": 0.01,
    "positive_net_required": True,
    "comparison": "strict_gt",
    "require_geo_raw_disagreement": True,
    "training_seeds": list(SEEDS),
    "checkpoint_family": "best_model",
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20260826,
    "independent_confirmation": False,
    "m4_no_go_preserved": True,
}
SELECTION_FILES = {"selection.json", "frozen_policies.json", "report.md"}
PREDICTION_FILES = {"experts.parquet", "actions.parquet", "metadata.json"}
REPORT_FILES = {
    "classification.json",
    "policy_metrics.json",
    "bootstrap_summary.json",
    "paired_comparisons.json",
    "summary.json",
    "scoring_targets.parquet",
    "actions_with_outcomes.parquet",
    "bootstrap_replicates.parquet",
    "subgroup_metrics.parquet",
}
SCORE_FILES = REPORT_FILES | {
    "build_declaration.json",
    "reservation.json",
    "labels.parquet",
    "source_receipt.json",
    "receipt.json",
    "report.md",
}
CODE_FILES = (
    "multimodal/geo_helpfulness_relaxed_margin.py",
    "multimodal/geo_helpfulness_relaxed_margin_selection.py",
    "multimodal/geo_helpfulness_relaxed_margin_report.py",
    "multimodal/geo_helpfulness_relaxed_margin_registry.py",
    "tools/run_multimodal_geo_helpfulness_relaxed_margin.py",
    "multimodal/geo_helpfulness_router_heuristics.py",
    "multimodal/geo_helpfulness_legacy_router_report.py",
    "multimodal/geo_helpfulness_legacy_router_score.py",
    "multimodal/geo_helpfulness_router_metrics.py",
    "multimodal/geo_helpfulness_router_uncertainty.py",
    "multimodal/geo_helpfulness_router_m4_common.py",
)


def _seal(value, key="manifest_sha256"):
    return {**value, key: canonical_hash(value)}


def _fit_incapable(*, test_stage=False):
    forbidden = {"torch", "sklearn", "scipy", "xgboost"}
    if any(name.split(".")[0] in forbidden for name in sys.modules):
        raise ValueError("this workflow must not import model-fitting libraries")
    if test_stage:
        legacy_score._fit_incapable()
        if "multimodal.geo_helpfulness_relaxed_margin_selection" in sys.modules:
            raise ValueError("test stages must not import threshold selection")


def _allow_output(root):
    allowed = {"declaration.json"}
    for folder, names in (
        ("selection", SELECTION_FILES),
        ("predictions", PREDICTION_FILES),
        ("scores", SCORE_FILES),
    ):
        allowed |= {f"{folder}/{n}" for n in names | {"manifest.json"}}
    _allowlist(root, allowed)


def _parent_metadata(source):
    value = parent.load_declaration(source)
    for folder, names in (
        ("selection", parent.SELECTION_FILES),
        ("predictions", parent.PREDICTION_FILES),
        ("scores", parent.SCORE_FILES),
    ):
        parent._manifest(source / folder, value, folder, names)
    # Validate score ownership without decoding any labels or predictions.
    from multimodal.geo_helpfulness_relaxed_rf_registry import validate_event

    prediction = read_json(source / "predictions/manifest.json")
    build = parent._build_record(source, value, prediction)
    if read_json(source / "scores/build_declaration.json") != build:
        raise ValueError("parent score build identity differs")
    receipt = read_json(source / "scores/receipt.json")
    check_self_hash(receipt, "receipt_sha256")
    if receipt["output_file_sha256"] != {
        n: file_hash(source / "scores" / n)
        for n in parent.SCORE_FILES - {"receipt.json"}
    }:
        raise ValueError("parent scoring receipt differs")
    validate_event(
        Path(value["registry_root"]),
        build,
        read_json(source / "scores/reservation.json"),
        receipt,
        source / "scores/receipt.json",
    )
    return value


def _training_paths(protocol):
    router = protocol / "router"
    return {
        "audit": router / "router_dataset_audit.parquet",
        "targets": router / "router_dataset.parquet",
        "dataset_manifest": router / "router_dataset_manifest.json",
        "original": router / "m4_feasibility_v1/oof_heuristics.json",
    }


def _declare(root, source, protocol):
    if _files(root):
        raise ValueError("unowned output directory is not empty")
    old = _parent_metadata(source)
    snapshot = dict(old["parent_file_sha256"])
    inputs = _training_paths(protocol)
    if any(str(path) not in snapshot for path in inputs.values()):
        raise ValueError("OOF inputs must belong to the sealed parent lineage")
    dataset = read_json(inputs["dataset_manifest"])
    check_self_hash(dataset)
    for name, record in dataset["artifacts"].items():
        if file_hash(protocol / "router" / name) != record["file_sha256"]:
            raise ValueError("OOF dataset artifact differs")
    snapshot.update({str(source / n): file_hash(source / n) for n in _files(source)})
    registry = Path(old["registry_root"])
    snapshot.update({str(p): file_hash(p) for p in sorted(registry.rglob("*.json"))})
    snapshot.update(
        {str(PROJECT_ROOT / n): file_hash(PROJECT_ROOT / n) for n in CODE_FILES}
    )
    value = _seal(
        {
            "schema_version": SCHEMA + ".declaration",
            "study_id": STUDY_ID,
            "declared_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(root),
            "configuration": CONFIGURATION,
            "paths": {"source_root": str(source), "protocol_root": str(protocol)},
            "parent_file_sha256": snapshot,
            "source_declaration_sha256": old["declaration_sha256"],
            "identity_manifest": old["identity_manifest"],
            "ontology": old["ontology"],
            "registry_root": str(registry),
            "user_authorization": "Complete the OOF margin-difference heuristic under H<=0.7R on the cleaned test set; retain 1% coverage and 20 plots; one previously reported expert family only.",
            "historical_test_exposure": True,
            "adaptive_reuse": True,
            "independent_confirmation": False,
            "m4_no_go_preserved": True,
            "model_fitting_permitted": False,
            "recalibration_permitted": False,
            "test_threshold_search_permitted": False,
            "environment": {
                n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")
            },
            "python": sys.version,
        },
        "declaration_sha256",
    )
    check_snapshot(snapshot)
    write_json(root / "declaration.json", value)
    return value


def load_declaration(root):
    root = Path(root).resolve()
    _allow_output(root)
    value = read_json(root / "declaration.json")
    check_self_hash(value, "declaration_sha256")
    if (
        value.get("schema_version") != SCHEMA + ".declaration"
        or value.get("study_id") != STUDY_ID
        or value.get("output_root") != str(root)
        or canonical_hash(value.get("configuration")) != canonical_hash(CONFIGURATION)
    ):
        raise ValueError("margin study declaration differs")
    check_snapshot(value["parent_file_sha256"])
    return value


def _manifest(folder, declaration, phase, names):
    value = read_json(folder / "manifest.json")
    check_self_hash(value)
    if (
        value.get("study_id") != STUDY_ID
        or value.get("schema_version") != SCHEMA + "." + phase
        or value.get("declaration_sha256") != declaration["declaration_sha256"]
        or set(value.get("artifacts", {})) != names
    ):
        raise ValueError("phase manifest identity differs")
    _allowlist(folder, names | {"manifest.json"}, complete=True)
    for name, record in value["artifacts"].items():
        path = folder / name
        if (
            file_hash(path) != record["file_sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"artifact changed: {path}")
    return value


def _publish(folder, declaration, phase, names, **extra):
    check_snapshot(declaration["parent_file_sha256"])
    _allowlist(folder, names, complete=True)
    artifacts = {}
    for name in sorted(names):
        path = folder / name
        record = {"file_sha256": file_hash(path), "bytes": path.stat().st_size}
        if name.endswith(".parquet"):
            info = pq.ParquetFile(path)
            record.update(
                rows=info.metadata.num_rows, arrow_schema=str(info.schema_arrow)
            )
        artifacts[name] = record
    value = _seal(
        {
            "schema_version": SCHEMA + "." + phase,
            "study_id": STUDY_ID,
            "declaration_sha256": declaration["declaration_sha256"],
            "artifacts": artifacts,
            **extra,
        }
    )
    write_json(folder / "manifest.json", value)
    return value


def _selection_values(declaration):
    from multimodal.geo_helpfulness_relaxed_margin_selection import build_selection

    paths = _training_paths(Path(declaration["paths"]["protocol_root"]))
    audit = pq.read_table(paths["audit"]).to_pandas()
    targets = pq.read_table(paths["targets"], columns=list(TARGET_COLUMNS)).to_pandas()
    selection = build_selection(audit, targets, read_json(paths["original"]))
    policies = {
        ORIGINAL_POLICY: selection["policies"]["original"]["threshold_specification"],
        NEW_POLICY: selection["policies"]["relaxed"]["threshold_specification"],
    }
    lines = [
        "# OOF-only margin threshold selection",
        "",
        "One relaxed rule: H≤0.7R, coverage ≥1%, ≥20 plots; original rule reproduced unchanged.",
        "No validation/test rows decoded. No model fitting or recalibration.",
        "",
        "| Rule | Threshold | Actions | Rescues | Harms | Net | Plots |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, record in selection["policies"].items():
        p = record["pooled"]
        lines.append(
            f"| {key} | {record['threshold_display']} | {p['acted_count']} | {p['rescued_count']} | {p['harmed_count']} | {p['rescued_minus_harmed']} | {p['unique_acted_plots']} |"
        )
    return {
        "selection.json": selection,
        "frozen_policies.json": policies,
        "report.md": "\n".join(lines) + "\n",
    }


def _select(root, source, protocol, *, build):
    _fit_incapable()
    root, source, protocol = [Path(p).resolve() for p in (root, source, protocol)]
    # Avoid writes into the protocol, source studies, or their ancestor directories.
    for protected in (
        source,
        protocol,
        BASE / "exploratory_legacy_capacity_v1",
        BASE / "relaxed_guardrails_lr_v1",
        PROJECT_ROOT / "multimodal_artifacts/locked_test_registry",
    ):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError("output overlaps frozen source or registry")
    with exclusive_lock(root):
        _allow_output(root)
        if (root / "declaration.json").exists():
            declaration = load_declaration(root)
        elif build:
            declaration = _declare(root, source, protocol)
        else:
            raise ValueError("study declaration is missing")
        if declaration["paths"] != {
            "source_root": str(source),
            "protocol_root": str(protocol),
        }:
            raise ValueError("requested paths differ from declaration")
        folder = root / "selection"
        committed = (folder / "manifest.json").exists()
        if committed:
            manifest = _manifest(folder, declaration, "selection", SELECTION_FILES)
        elif not build:
            raise ValueError("selection is not committed")
        for name, value in _selection_values(declaration).items():
            _compare_or_write(folder / name, value, build and not committed)
        if not committed:
            manifest = _publish(folder, declaration, "selection", SELECTION_FILES)
        check_snapshot(declaration["parent_file_sha256"])
    return {
        "status": ("reused_valid" if committed else "built") if build else "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "test_data_decoded": False,
    }


def build_selection(
    output_root=DEFAULT_OUTPUT, source_root=DEFAULT_SOURCE, protocol_root=DEFAULT_ROOT
):
    return _select(output_root, source_root, protocol_root, build=True)


def validate_selection(
    output_root=DEFAULT_OUTPUT, source_root=DEFAULT_SOURCE, protocol_root=DEFAULT_ROOT
):
    return _select(output_root, source_root, protocol_root, build=False)


def _policies(root, declaration):
    manifest = _manifest(root / "selection", declaration, "selection", SELECTION_FILES)
    policies = read_json(root / "selection/frozen_policies.json")
    if set(policies) != {ORIGINAL_POLICY, NEW_POLICY}:
        raise ValueError("expected exactly original and one relaxed policy")
    for value in policies.values():
        validate_threshold_specification(value)
    original = read_json(
        _training_paths(Path(declaration["paths"]["protocol_root"]))["original"]
    )
    if (
        policies[ORIGINAL_POLICY]
        != original["heuristics"][ORIGINAL_POLICY]["threshold_specification"]
    ):
        raise ValueError("original comparator changed")
    saved = read_json(root / "selection/selection.json")["policies"]
    for name, key in ((ORIGINAL_POLICY, "original"), (NEW_POLICY, "relaxed")):
        if policies[name] != saved[key]["threshold_specification"]:
            raise ValueError("policy differs from sealed selection")
    return manifest, policies


def _prediction_values(root, declaration):
    selection, policies = _policies(root, declaration)
    source = Path(declaration["paths"]["source_root"])
    # Filter at the Arrow reader; the other expert family is not evaluated.
    table = pq.read_table(
        source / "predictions/experts.parquet",
        filters=[("checkpoint_family", "=", "best_model")],
    )
    e = (
        table.to_pandas()
        .sort_values(["row_uid", "training_seed"], kind="stable")
        .reset_index(drop=True)
    )
    legacy_report._label_guard(e, "experts")
    legacy_report._identity(e, "experts", with_file=True)
    reference = (
        pd.DataFrame(declaration["identity_manifest"]["rows"])[
            ["row_uid", "file", "plot_idx"]
        ]
        .sort_values("row_uid")
        .reset_index(drop=True)
    )
    for seed in SEEDS:
        observed = e.loc[e.training_seed.eq(seed), reference.columns].reset_index(
            drop=True
        )
        if not observed.equals(reference):
            raise ValueError("expert test identity differs")
    if len(e) != 5388 or e.row_uid.nunique() != 1347 or e.plot_idx.nunique() != 531:
        raise ValueError("cleaned-test dimensions differ")
    values = legacy_report.build_semantic_values(
        {
            name: np.stack(e[f"{name}_prob_calibrated"].to_numpy())
            for name in ("image", "geo", "raw")
        }
    )
    for name, value in values.items():
        if not np.allclose(e[name].to_numpy(), value, atol=1e-12, rtol=1e-12):
            raise ValueError(f"expert semantic feature differs: {name}")
    outputs = []
    for name, spec in policies.items():
        frame = e[
            [
                "schema_version",
                "study_id",
                "checkpoint_family",
                "row_uid",
                "plot_idx",
                "training_seed",
            ]
        ].copy()
        frame["schema_version"], frame["study_id"] = SCHEMA + ".actions", STUDY_ID
        frame["policy_id"], frame["score"] = name, e[ORIGINAL_POLICY].to_numpy()
        frame["threshold_specification"] = canonical_bytes(spec).decode()
        acted = apply_heuristic_threshold(
            frame.score.to_numpy(), e.raw_pred.to_numpy(), e.geo_pred.to_numpy(), spec
        )
        frame["acted"] = acted
        frame["raw_pred"], frame["geo_pred"] = (
            e.raw_pred.to_numpy(),
            e.geo_pred.to_numpy(),
        )
        frame["final_pred"] = np.where(acted, frame.geo_pred, frame.raw_pred)
        outputs.append(frame)
    actions = (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["policy_id", "row_uid", "training_seed"], kind="stable")
        .reset_index(drop=True)
    )
    prior = (
        pq.read_table(
            source / "predictions/actions.parquet",
            filters=[
                ("checkpoint_family", "=", "best_model"),
                ("policy_id", "=", ORIGINAL_POLICY),
            ],
        )
        .to_pandas()
        .sort_values(["row_uid", "training_seed"], kind="stable")
        .reset_index(drop=True)
    )
    compared = [
        "row_uid",
        "plot_idx",
        "training_seed",
        "policy_id",
        "score",
        "threshold_specification",
        "acted",
        "raw_pred",
        "geo_pred",
        "final_pred",
    ]
    current = actions.loc[actions.policy_id.eq(ORIGINAL_POLICY), compared].reset_index(
        drop=True
    )
    # Small integer Arrow storage widths are normalized for semantic equality.
    if not np.array_equal(prior[compared].to_numpy(), current.to_numpy()):
        raise ValueError("original margin test actions do not reproduce")
    from multimodal.geo_helpfulness_relaxed_margin_report import validate_actions

    validate_actions(e, actions, policies[ORIGINAL_POLICY], policies[NEW_POLICY])
    action_schema = pq.read_schema(source / "predictions/actions.parquet")
    tables = {
        "experts.parquet": pa.Table.from_pandas(
            e, schema=table.schema, preserve_index=False
        ),
        "actions.parquet": pa.Table.from_pandas(
            actions, schema=action_schema, preserve_index=False
        ),
    }
    metadata = {
        "study_id": STUDY_ID,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "checkpoint_family": "best_model",
        "unique_images": 1347,
        "unique_plots": 531,
        "expert_rows": 5388,
        "action_rows": 10776,
        "test_labels_decoded": False,
        "threshold_selection_performed": False,
        "model_fitting_performed": False,
    }
    return selection, policies, {**tables, "metadata.json": metadata}


def _predictions(root, *, build):
    _fit_incapable(test_stage=True)
    root = Path(root).resolve()
    with exclusive_lock(root):
        declaration = load_declaration(root)
        folder = root / "predictions"
        committed = (folder / "manifest.json").exists()
        manifest = (
            _manifest(folder, declaration, "predictions", PREDICTION_FILES)
            if committed
            else None
        )
        if not committed and not build:
            raise ValueError("predictions are not committed")
        selection, _, values = _prediction_values(root, declaration)
        for name, value in values.items():
            _compare_or_write(folder / name, value, build and not committed)
        if not committed:
            manifest = _publish(
                folder,
                declaration,
                "predictions",
                PREDICTION_FILES,
                selection_manifest_sha256=selection["manifest_sha256"],
            )
        if manifest["selection_manifest_sha256"] != selection["manifest_sha256"]:
            raise ValueError("prediction-selection lineage differs")
        check_snapshot(declaration["parent_file_sha256"])
    _fit_incapable(test_stage=True)
    return {
        "status": ("reused_valid" if committed else "built") if build else "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "test_labels_decoded": False,
    }


def build_predictions(output_root=DEFAULT_OUTPUT):
    return _predictions(output_root, build=True)


def validate_predictions(output_root=DEFAULT_OUTPUT):
    return _predictions(output_root, build=False)


def _report_text(reports, policies):
    classification = reports["classification.json"]
    lines = [
        "# Relaxed margin-difference policy: cleaned-test results",
        "",
        "Exploratory evaluation on a previously inspected test set; not independent confirmation. Original M4 no-go remains unchanged.",
        "Threshold selected using training OOF rows only, with H≤0.7R, coverage ≥1%, and ≥20 distinct plots. No model fitting or recalibration.",
        f"Original threshold: {validate_threshold_specification(policies[ORIGINAL_POLICY])}; relaxed threshold: {validate_threshold_specification(policies[NEW_POLICY])}.",
        "",
        "Metrics average four seeds on the same 1,347 images; counts pool 5,388 seed-image evaluations, not independent images.",
        "",
        "| Method | Acc@1 (%) | Weighted F1 |",
        "|---|---:|---:|",
    ]
    for name, record in [
        ("Fusion", classification["experts"]["raw"]),
        ("Original margin rule", classification["policies"][ORIGINAL_POLICY]),
        ("Relaxed margin rule", classification["policies"][NEW_POLICY]),
    ]:
        summary = record["seed_summary"]
        lines.append(
            f"| {name} | {100*summary['top1_accuracy']['mean']:.6f} | {summary['weighted_f1']['mean']:.6f} |"
        )
    lines += [
        "",
        "| Rule | Actions | Rescues | Harms | Net | Coverage (%) | Plots |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (ORIGINAL_POLICY, NEW_POLICY):
        p = reports["policy_metrics.json"][name]["pooled"]
        lines.append(
            f"| {name} | {p['acted_count']} | {p['rescued_count']} | {p['harmed_count']} | {p['rescued_minus_harmed']} | {100*p['coverage_fraction']:.6f} | {p['unique_acted_plots']} |"
        )
    lines += [
        "",
        "Conditional whole-plot-bootstrap intervals and the paired contrast against the original rule are saved in bootstrap_summary.json and paired_comparisons.json. They do not account for adaptive test reuse or threshold-selection uncertainty. Development guardrails are not re-enforced on test outcomes.",
        "",
    ]
    return "\n".join(lines)


def _scores(root, *, build):
    _fit_incapable(test_stage=True)
    from multimodal import geo_helpfulness_relaxed_margin_registry as registry_api
    from multimodal.geo_helpfulness_relaxed_margin_report import build_report

    root = Path(root).resolve()
    with exclusive_lock(root):
        declaration = load_declaration(root)
        prediction = _manifest(
            root / "predictions", declaration, "predictions", PREDICTION_FILES
        )
        selection, policies, reconstructed = _prediction_values(root, declaration)
        if prediction["selection_manifest_sha256"] != selection["manifest_sha256"]:
            raise ValueError("prediction selection identity differs")
        for name, value in reconstructed.items():
            _compare_or_write(root / "predictions" / name, value, False)
        folder = root / "scores"
        committed = (folder / "manifest.json").exists()
        manifest = (
            _manifest(folder, declaration, "scores", SCORE_FILES) if committed else None
        )
        if not committed and not build:
            raise ValueError("scores are not committed")
        record = {
            "schema_version": SCHEMA + ".score_build",
            "study_id": STUDY_ID,
            "owner_output_root": str(root),
            "declaration_sha256": declaration["declaration_sha256"],
            "prediction_manifest_sha256": prediction["manifest_sha256"],
            "composite_bundle_file_sha256": prediction["manifest_sha256"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "identity_projection_sha256": declaration["identity_manifest"][
                "identity_projection_sha256"
            ],
            "historical_test_exposure": True,
            "adaptive_reuse": True,
            "m4_no_go_preserved": True,
        }
        registry_root = Path(declaration["registry_root"])
        if committed:
            if read_json(folder / "build_declaration.json") != record:
                raise ValueError("score build differs")
            reservation = read_json(folder / "reservation.json")
        else:
            if _files(folder) and not (folder / "build_declaration.json").exists():
                raise ValueError("unowned partial score outputs")
            write_json(folder / "build_declaration.json", record)
            reservation = registry_api.reserve_event(registry_root, record)
            write_json(folder / "reservation.json", reservation)
        registry_api.validate_reservation(registry_root, record, reservation)
        source_labels = (
            Path(declaration["paths"]["source_root"]) / "scores/labels.parquet"
        )
        if not (folder / "labels.parquet").exists():
            write_bytes(folder / "labels.parquet", source_labels.read_bytes())
        if file_hash(folder / "labels.parquet") != file_hash(source_labels):
            raise ValueError("label copy differs from authenticated parent")
        source_receipt = {
            "study_id": STUDY_ID,
            "build_sha256": canonical_hash(record),
            "reservation_file_sha256": file_hash(folder / "reservation.json"),
            "parent_label_file": str(source_labels),
            "label_file_sha256": file_hash(source_labels),
            "original_test_sources_opened": False,
        }
        _compare_or_write(
            folder / "source_receipt.json", source_receipt, build and not committed
        )
        labels_table = pq.read_table(folder / "labels.parquet")
        if labels_table.schema != legacy_score.LABEL_SCHEMA:
            raise ValueError("label schema differs")
        labels = legacy_score._validate_labels(labels_table.to_pandas(), declaration)
        reports = build_report(
            reconstructed["experts.parquet"].to_pandas(),
            reconstructed["actions.parquet"].to_pandas(),
            labels,
            declaration["ontology"],
            policies[ORIGINAL_POLICY],
            policies[NEW_POLICY],
        )
        if set(reports) != REPORT_FILES:
            raise ValueError("scoring report file contract differs")
        reports["report.md"] = _report_text(reports, policies)
        for name, value in reports.items():
            _compare_or_write(folder / name, value, build and not committed)
        receipt = _seal(
            {
                **record,
                "reservation": reservation,
                "ordered_scoring_label_sha256": legacy_score._label_hash(labels),
                "output_file_sha256": {
                    n: file_hash(folder / n)
                    for n in sorted(SCORE_FILES - {"receipt.json"})
                },
                "evaluation_event_id": canonical_hash(
                    {
                        "study_id": STUDY_ID,
                        "prediction": prediction["manifest_sha256"],
                        "labels": legacy_score._label_hash(labels),
                    }
                ),
                "independent_confirmation": False,
                "model_fitting": False,
                "recalibration": False,
                "test_threshold_search": False,
            },
            "receipt_sha256",
        )
        _compare_or_write(folder / "receipt.json", receipt, build and not committed)
        check_snapshot(declaration["parent_file_sha256"])
        if not committed:
            registry_api.finish_event(
                registry_root, record, reservation, receipt, folder / "receipt.json"
            )
            manifest = _publish(
                folder,
                declaration,
                "scores",
                SCORE_FILES,
                prediction_manifest_sha256=prediction["manifest_sha256"],
                receipt_sha256=receipt["receipt_sha256"],
            )
        if (
            manifest["prediction_manifest_sha256"] != prediction["manifest_sha256"]
            or manifest["receipt_sha256"] != receipt["receipt_sha256"]
        ):
            raise ValueError("score manifest linkage differs")
        registry_api.validate_event(
            registry_root, record, reservation, receipt, folder / "receipt.json"
        )
    _fit_incapable(test_stage=True)
    return {
        "status": ("reused_valid" if committed else "built") if build else "validated",
        "manifest_sha256": manifest["manifest_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "test_threshold_search_performed": False,
        "independent_confirmation": False,
    }


def build_scores(output_root=DEFAULT_OUTPUT):
    return _scores(output_root, build=True)


def validate_scores(output_root=DEFAULT_OUTPUT):
    return _scores(output_root, build=False)

"""Append-only scoring of both sealed legacy-checkpoint prediction families.

Original test-label columns and historical metrics are opened only by the first
score build, after one combined prediction bundle and a global event reservation
exist. Validation and completed-build reuse use the sealed scoring copies.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal.geo_helpfulness_legacy_router_common import (
    FAMILIES,
    POLICIES,
    SEEDS,
    STUDY_ID,
    canonical_hash,
    check_self_hash,
    check_snapshot,
    compare_tables,
    exclusive_lock,
    file_hash,
    normalize_identity,
    read_json,
    table_bytes,
    validate_predictions,
    write_bytes,
    write_json,
)
from multimodal.geo_helpfulness_legacy_router_report import evaluate_family


EXPECTED_ROWS = 1347
EXPECTED_PLOTS = 531
SCHEMA = "geo_helpfulness.legacy_router_score.v1"
EXPERTS = {"image_only": "image", "geo_only": "geo", "raw_concat": "raw"}
FAMILY_REPORTS = (
    "expert_metrics.json",
    "classification.json",
    "router_metrics.json",
    "policy_metrics.json",
    "bootstrap_summary.json",
    "bootstrap_replicates.parquet",
    "subgroup_metrics.parquet",
    "actions_with_outcomes.parquet",
    "scoring_targets.parquet",
)
LABEL_SCHEMA = pa.schema(
    [
        pa.field(name, dtype, nullable=False)
        for name, dtype in [
            ("row_uid", pa.string()),
            ("file", pa.string()),
            ("plot_idx", pa.string()),
            ("label_id_dense", pa.int8()),
            ("canonical_l3_id", pa.int8()),
            ("label_name", pa.string()),
        ]
    ]
)
FIXED_FILES = {
    "build_declaration.json",
    "reservation.json",
    "labels.parquet",
    "historical_replay.json",
    "source_receipt.json",
    "summary.json",
    "report.md",
    "receipt.json",
}
SCORE_FILES = FIXED_FILES | {
    f"{family}/{name}" for family in FAMILIES for name in FAMILY_REPORTS
}
DISCLOSURES = (
    "This is descriptive transfer of frozen M4 routers to historical experts. "
    "Both checkpoint families have historical test exposure; adaptive_reuse is true.",
    "oracle_test_best_model checkpoints were selected using test accuracy. Their "
    "results are a biased diagnostic, not an oracle ceiling or independent evidence.",
    "The M4 development no-go decision remains unchanged. Router models, feature "
    "transforms, temperatures, thresholds, and policy identities remain frozen.",
    "Reported seed summaries are descriptive. Plots, rather than seed realizations, "
    "are the resampling units for paired bootstrap uncertainty.",
)


def _allowlist(root: Path, *, complete: bool = False) -> None:
    if root.is_symlink():
        raise ValueError("score root must not be a symlink")
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"score artifact must not be a symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
        elif path.is_dir() and path.relative_to(root).as_posix() not in FAMILIES:
            raise ValueError(f"unexpected score directory: {path}")
    allowed = SCORE_FILES | {"manifest.json"}
    if files - allowed or (complete and files != allowed):
        raise ValueError("score artifact file allowlist mismatch")


def _fit_incapable() -> None:
    forbidden = {"torch", "scipy", "sklearn"}
    forbidden_modules = {
        "multimodal.geo_helpfulness_router_selection",
        "multimodal.geo_helpfulness_router_candidate_fit",
        "multimodal.geo_helpfulness_oof",
        "multimodal.trainer",
    }
    if any(
        name.split(".")[0] in forbidden or name in forbidden_modules
        for name in sys.modules
    ):
        raise ValueError(
            "legacy scoring requires a fresh process without fitting/selection imports"
        )


def _identity(declaration: dict) -> pd.DataFrame:
    frame = pd.DataFrame(declaration["identity_manifest"]["rows"])
    required = ["row_uid", "file", "plot_idx"]
    if (
        len(frame) != EXPECTED_ROWS
        or not set(required).issubset(frame)
        or frame[required].isna().any().any()
        or frame.row_uid.duplicated().any()
        or frame.file.str.casefold().duplicated().any()
        or frame.plot_idx.nunique() != EXPECTED_PLOTS
    ):
        raise ValueError("frozen scoring identity has the wrong rows, plots, or keys")
    return frame.sort_values("row_uid", kind="stable").reset_index(drop=True)


def _ontology(declaration: dict) -> dict:
    records = declaration["ontology"]
    if (
        len(records) != 18
        or {record["dense_id"] for record in records} != set(range(18))
        or len({record["canonical_l3_id"] for record in records}) != 18
        or len({record["label_name"] for record in records}) != 18
    ):
        raise ValueError("invalid frozen 18-class ontology")
    return {record["label_name"]: record for record in records}


def _text(value, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = unicodedata.normalize("NFC", value)
    if (
        not value
        or value != value.strip()
        or any(unicodedata.category(character) in {"Cc", "Cs"} for character in value)
    ):
        raise ValueError(f"invalid {name}")
    return value


def _validate_labels(labels: pd.DataFrame, declaration: dict) -> pd.DataFrame:
    if list(labels.columns) != LABEL_SCHEMA.names or labels.isna().any().any():
        raise ValueError("scoring label schema or null values are invalid")
    identity = _identity(declaration)
    labels = labels.sort_values("row_uid", kind="stable").reset_index(drop=True)
    if not np.array_equal(
        labels[["row_uid", "file", "plot_idx"]].to_numpy(),
        identity[["row_uid", "file", "plot_idx"]].to_numpy(),
    ):
        raise ValueError("scoring labels differ from the exact frozen identity")
    ontology = _ontology(declaration)
    for row in labels.itertuples(index=False):
        spec = ontology.get(row.label_name)
        if spec is None or (
            row.label_id_dense != spec["dense_id"]
            or row.canonical_l3_id != spec["canonical_l3_id"]
        ):
            raise ValueError("scoring labels differ from the frozen ontology")
    if not labels.groupby("plot_idx").label_id_dense.nunique().eq(1).all():
        raise ValueError("scoring labels disagree within a plot")
    for name in ("label_id_dense", "canonical_l3_id"):
        labels[name] = labels[name].astype(np.int8)
    return labels


def _read_original_labels(declaration: dict) -> pd.DataFrame:
    """The sole source-label reader; use an explicit storage-level projection."""
    identity = _identity(declaration)
    by_file = {
        normalize_identity(row.file, row.plot_idx)[0]: row
        for row in identity.itertuples(index=False)
    }
    ontology = _ontology(declaration)
    reference = None
    for seed in SEEDS:
        path = Path(declaration["caches"][str(seed)]["test"])
        frame = pq.read_table(
            path, columns=["file", "plot_idx", "label_name"]
        ).to_pandas()
        if len(frame) != EXPECTED_ROWS or frame.isna().any().any():
            raise ValueError(
                f"seed {seed} scoring label rows differ from the frozen cohort"
            )
        records = []
        for raw in frame.itertuples(index=False):
            filename, plot = normalize_identity(raw.file, raw.plot_idx)
            label = _text(raw.label_name, name="label_name")
            frozen = by_file.get(filename)
            spec = ontology.get(label)
            if frozen is None or frozen.plot_idx != plot or spec is None:
                raise ValueError(
                    "source test identity or label differs from frozen contract"
                )
            records.append(
                {
                    "row_uid": frozen.row_uid,
                    "file": frozen.file,
                    "plot_idx": plot,
                    "label_id_dense": spec["dense_id"],
                    "canonical_l3_id": spec["canonical_l3_id"],
                    "label_name": label,
                }
            )
        current = _validate_labels(
            pd.DataFrame(records, columns=LABEL_SCHEMA.names), declaration
        )
        if reference is None:
            reference = current
        elif not reference.equals(current):
            raise ValueError("test labels differ between the four source caches")
    return reference


def _label_table(labels: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(
        labels, schema=LABEL_SCHEMA, preserve_index=False
    ).replace_schema_metadata(None)


def _label_hash(labels: pd.DataFrame) -> str:
    return canonical_hash(labels.to_dict(orient="records"))


def _build_declaration(
    output_root: Path, declaration: dict, prediction_manifest: dict
) -> dict:
    return {
        "schema_version": SCHEMA,
        "study_id": STUDY_ID,
        "owner_output_root": str(output_root.resolve()),
        "declaration_sha256": declaration["declaration_sha256"],
        "declaration_file_sha256": file_hash(output_root / "declaration.json"),
        "prediction_manifest_sha256": prediction_manifest["manifest_sha256"],
        "prediction_manifest_file_sha256": file_hash(
            output_root / "predictions/manifest.json"
        ),
        "composite_bundle_file_sha256": prediction_manifest["manifest_sha256"],
        "composite_bundle_hash_basis": "prediction_manifest_payload_sha256",
        "identity_projection_sha256": declaration["identity_manifest"][
            "identity_projection_sha256"
        ],
        "ontology_sha256": canonical_hash(declaration["ontology"]),
        "source_snapshot": declaration["source_snapshot"],
        "checkpoint_families": list(FAMILIES),
        "user_authorization": declaration.get(
            "user_authorization", declaration.get("authorization")
        ),
        "historical_test_exposure": True,
        "adaptive_reuse": True,
        "m4_no_go_preserved": True,
    }


def _registry_paths(declaration: dict, build: dict) -> dict[str, Path]:
    root = Path(declaration["registry_root"])
    bundle = build["composite_bundle_file_sha256"]
    snapshot = root / "snapshots" / build["identity_projection_sha256"]
    return {
        "root": root,
        "bundle": root / "bundles" / f"{bundle}.json",
        "first": snapshot / "first_score_event.json",
        "event": snapshot / f"{bundle}.json",
        "completed": root / "bundles" / f"{bundle}.completed.json",
    }


def _reserve_score_event(declaration: dict, build: dict) -> dict:
    paths = _registry_paths(declaration, build)
    owner = {
        "status": "reserved",
        "study_id": STUDY_ID,
        "owner_output_root": build["owner_output_root"],
        "composite_bundle_file_sha256": build["composite_bundle_file_sha256"],
        "identity_projection_sha256": build["identity_projection_sha256"],
        "prediction_manifest_sha256": build["prediction_manifest_sha256"],
        "historical_test_exposure": True,
        "adaptive_reuse": True,
    }
    # Creating the first-event record before the bundle claim makes a crash
    # between those two writes resumable without losing the original ordering.
    with exclusive_lock(paths["root"]):
        if paths["first"].exists():
            prior = read_json(paths["first"]) != owner
        else:
            write_json(paths["first"], owner)
            prior = False
        reservation = {
            **owner,
            "prior_registered_score_event": prior,
            "first_score_event_file_sha256": file_hash(paths["first"]),
        }
        if paths["bundle"].exists() and read_json(paths["bundle"]) != reservation:
            raise ValueError(
                "combined prediction bundle already reserved by another score owner"
            )
        write_json(paths["bundle"], reservation)
    return reservation


def _expert_counts(
    experts: pd.DataFrame, labels: pd.DataFrame, family: str, seed: int, expert: str
) -> dict:
    rows = experts.loc[
        experts.checkpoint_family.eq(family) & experts.training_seed.eq(seed)
    ].sort_values("row_uid", kind="stable")
    if len(rows) != len(labels) or not np.array_equal(rows.row_uid, labels.row_uid):
        raise ValueError(
            "historical replay expert identities differ from sealed labels"
        )
    values = np.stack(rows[f"{expert}_prob_native_t1"].to_numpy())
    if values.shape != (len(labels), 18) or not np.isfinite(values).all():
        raise ValueError(
            "historical replay requires finite 18-class native probabilities"
        )
    ranking = np.argsort(-values, axis=1, kind="stable")[:, :3]
    y = labels.label_id_dense.to_numpy(dtype=np.int64)
    return {
        "row_count": len(y),
        "top1_correct": int(np.sum(ranking[:, 0] == y)),
        "top3_correct": int(np.sum(np.any(ranking == y[:, None], axis=1))),
    }


def _check_historical_replay(
    replay: dict, declaration: dict, experts: pd.DataFrame, labels: pd.DataFrame
) -> None:
    if replay.get("schema_version") != SCHEMA or replay.get("study_id") != STUDY_ID:
        raise ValueError("historical replay schema/study mismatch")
    expected = {
        (mode, str(seed), family)
        for mode in EXPERTS
        for seed in SEEDS
        for family in FAMILIES
    }
    observed = set()
    for record in replay["records"]:
        mode, seed, family = (
            record["expert_mode"],
            record["training_seed"],
            record["checkpoint_family"],
        )
        key = (mode, str(seed), family)
        if key not in expected or key in observed:
            raise ValueError(
                "historical replay has unexpected or repeated expert/seed/family"
            )
        observed.add(key)
        counts = _expert_counts(experts, labels, family, int(seed), EXPERTS[mode])
        if record["reproduced_counts"] != counts:
            raise ValueError("historical replay sealed prediction counts changed")
        for metric, count in (
            ("top1_acc", "top1_correct"),
            ("top3_acc", "top3_correct"),
        ):
            old = record["historical_metrics"][metric]
            if (
                isinstance(old, bool)
                or not isinstance(old, (int, float))
                or not math.isfinite(old)
                or not 0 <= old <= 1
                or round(old * counts["row_count"]) != counts[count]
                or abs(old - counts[count] / counts["row_count"]) > 1e-12
            ):
                raise ValueError(
                    f"historical expert reproduction failed: {mode}/{seed}/{family}/{metric}"
                )
    if observed != expected:
        raise ValueError(
            "historical replay does not cover all 24 expert/seed/family results"
        )
    sources = {
        str(Path(declaration["legacy_runs"][mode][str(seed)]) / "metrics.json")
        for mode in EXPERTS
        for seed in SEEDS
    }
    if (
        set(replay["source_metrics_file_sha256"]) != sources
        or replay["source_metrics_file_sha256"]
        != declaration["historical_metrics_sha256"]
    ):
        raise ValueError(
            "historical replay source list differs from declared legacy runs"
        )


def _read_historical_replay(
    declaration: dict, experts: pd.DataFrame, labels: pd.DataFrame
) -> dict:
    records, hashes = [], {}
    for mode, expert in EXPERTS.items():
        for seed in SEEDS:
            path = Path(declaration["legacy_runs"][mode][str(seed)]) / "metrics.json"
            before = file_hash(path)
            if before != declaration["historical_metrics_sha256"][str(path)]:
                raise ValueError(
                    "historical metric source differs from the frozen declaration"
                )
            source = read_json(path)
            if file_hash(path) != before:
                raise ValueError("historical metric source changed while being read")
            hashes[str(path)] = before
            for family in FAMILIES:
                metric_key = "test" if family == "best_model" else "oracle_test_best"
                records.append(
                    {
                        "expert_mode": mode,
                        "training_seed": seed,
                        "checkpoint_family": family,
                        "historical_metrics": {
                            name: source[metric_key][name]
                            for name in ("top1_acc", "top3_acc")
                        },
                        "reproduced_counts": _expert_counts(
                            experts, labels, family, seed, expert
                        ),
                    }
                )
    replay = {
        "schema_version": SCHEMA,
        "study_id": STUDY_ID,
        "records": records,
        "source_metrics_file_sha256": hashes,
        "selection_performed": False,
        "historical_metrics_used_only_for_reproduction": True,
        "topk_rule": "stable_descending_probability_dense_id_ties",
    }
    _check_historical_replay(replay, declaration, experts, labels)
    return replay


def _family_reports(frames: dict, labels: pd.DataFrame) -> dict:
    results = {}
    for family in FAMILIES:
        inputs = [
            frames[name]
            .loc[frames[name].checkpoint_family.eq(family)]
            .reset_index(drop=True)
            for name in ("experts", "routers", "actions")
        ]
        reports = evaluate_family(*inputs, labels)
        if set(reports) != set(FAMILY_REPORTS):
            raise ValueError("family report artifact contract mismatch")
        results[family] = reports
    return results


def _summary(declaration: dict, results: dict, labels: pd.DataFrame) -> dict:
    families = {}
    for family, reports in results.items():
        classification = reports["classification.json"]
        families[family] = {
            "description": declaration["family_descriptions"][family],
            "test_selected_biased_diagnostic": family == "oracle_test_best_model",
            "classification_seed_summary": {
                **{
                    name: classification["experts"][name]["seed_summary"]
                    for name in EXPERTS.values()
                },
                **{
                    name: classification["policies"][name]["seed_summary"]
                    for name in POLICIES[:3]
                },
            },
            "policies": {
                name: {
                    key: reports["policy_metrics.json"][name][key]
                    for key in (
                        "mean_seed_net_utility",
                        "mean_seed_delta_accuracy",
                        "pooled",
                        "per_seed",
                        "seed_summary",
                    )
                }
                for name in POLICIES
            },
        }
    return {
        "schema_version": SCHEMA,
        "study_id": STUDY_ID,
        "unique_images": len(labels),
        "unique_plots": int(labels.plot_idx.nunique()),
        "seed_count": len(SEEDS),
        "checkpoint_families": families,
        "adaptive_reuse": True,
        "m4_no_go_preserved": True,
        "disclosures": list(DISCLOSURES),
    }


def _report(summary: dict) -> str:
    lines = [
        "# Frozen M4 router transfer to both legacy expert families",
        "",
        *DISCLOSURES,
        "",
    ]
    lines.extend(
        [
            f"Cohort: {summary['unique_images']} images, {summary['unique_plots']} plots, {summary['seed_count']} seeds.",
            "",
            "Top-1 and top-3 are descriptive mean ± population SD over seeds.",
            "",
            "| Checkpoint family | Classifier | Top-1 (%) | Top-3 (%) |",
            "|---|---|---:|---:|",
        ]
    )
    for family, record in summary["checkpoint_families"].items():
        for name, metrics in record["classification_seed_summary"].items():
            top1, top3 = metrics["top1_accuracy"], metrics["top3_accuracy"]
            lines.append(
                f"| {family} | {name} | {100 * top1['mean']:.3f} ± {100 * top1['std']:.3f} "
                f"| {100 * top3['mean']:.3f} ± {100 * top3['std']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Rescue/harm counts below pool seed realizations; Δaccuracy is the mean seed gain in percentage points.",
            "",
            "| Checkpoint family | Policy | Rescues | Harms | Net | Δaccuracy (pp) | Coverage (%) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family, record in summary["checkpoint_families"].items():
        for name, metric in record["policies"].items():
            pooled = metric["pooled"]
            rescued, harmed = pooled["rescued_count"], pooled["harmed_count"]
            lines.append(
                f"| {family} | {name} | {rescued} | {harmed} | {rescued - harmed} "
                f"| {100 * metric['mean_seed_delta_accuracy']:.3f} | {100 * pooled['coverage_fraction']:.3f} |"
            )
    lines += [
        "",
        "Historical top-1/top-3 counts for all 24 expert/seed/family results reproduce the 12 legacy metric files.",
        "",
    ]
    return "\n".join(lines)


def _hashes(root: Path, names: set[str]) -> dict:
    return {name: file_hash(root / name) for name in sorted(names)}


def _source_receipt(root: Path, build: dict) -> dict:
    return {
        "schema_version": SCHEMA,
        "study_id": STUDY_ID,
        "build_declaration_sha256": canonical_hash(build),
        "reservation_file_sha256": file_hash(root / "reservation.json"),
        "source_copy_file_sha256": _hashes(
            root, {"labels.parquet", "historical_replay.json"}
        ),
    }


def _receipt(
    build: dict, reservation: dict, labels: pd.DataFrame, hashes: dict
) -> dict:
    receipt = {
        **build,
        "artifact_role": "legacy_frozen_router_transfer_score_receipt",
        "ordered_scoring_label_sha256": _label_hash(labels),
        "unique_images": len(labels),
        "unique_plots": int(labels.plot_idx.nunique()),
        "output_file_sha256": hashes,
        "reservation": reservation,
        "oracle_family_interpretation": "test_selected_biased_diagnostic_not_oracle_ceiling",
        "capabilities": {
            "fitting": False,
            "selection": False,
            "threshold_search": False,
            "recalibration": False,
        },
        "disclosures": list(DISCLOSURES),
    }
    receipt["evaluation_event_id"] = canonical_hash(
        {
            "study_id": STUDY_ID,
            "prediction_hash": build["prediction_manifest_sha256"],
            "identity_hash": build["identity_projection_sha256"],
            "ordered_scoring_label_hash": receipt["ordered_scoring_label_sha256"],
            "ontology_hash": build["ontology_sha256"],
        }
    )
    receipt["receipt_sha256"] = canonical_hash(receipt)
    return receipt


def _event(receipt: dict, receipt_path: Path) -> dict:
    return {
        "artifact_role": "locked_test_score_event",
        "study_id": STUDY_ID,
        "adaptive_reuse": True,
        "prior_registered_score_event": receipt["reservation"][
            "prior_registered_score_event"
        ],
        "composite_bundle_file_sha256": receipt["composite_bundle_file_sha256"],
        "identity_projection_sha256": receipt["identity_projection_sha256"],
        "receipt_file_sha256": file_hash(receipt_path),
        "evaluation_event_id": receipt["evaluation_event_id"],
        "owner_output_root": receipt["owner_output_root"],
    }


def _validate_registry(
    declaration: dict, build: dict, reservation: dict, event: dict
) -> None:
    paths = _registry_paths(declaration, build)
    if (
        read_json(paths["bundle"]) != reservation
        or file_hash(paths["first"]) != reservation["first_score_event_file_sha256"]
        or read_json(paths["event"]) != event
        or read_json(paths["completed"]) != event
    ):
        raise ValueError("global score registry differs from the sealed score event")


def _validate(
    output_root: Path, declaration: dict, prediction_manifest: dict, frames: dict
) -> dict:
    root = output_root / "scores"
    _allowlist(root, complete=True)
    manifest = read_json(root / "manifest.json")
    check_self_hash(manifest)
    build = _build_declaration(output_root, declaration, prediction_manifest)
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("build_declaration") != build
        or set(manifest["artifact_file_sha256"]) != SCORE_FILES
        or manifest["artifact_file_sha256"] != _hashes(root, SCORE_FILES)
        or read_json(root / "build_declaration.json") != build
    ):
        raise ValueError("score manifest, sources, or artifact hashes changed")
    check_snapshot(declaration["source_snapshot"])
    if read_json(root / "source_receipt.json") != _source_receipt(root, build):
        raise ValueError(
            "sealed scoring source copies differ from their acquisition receipt"
        )
    table = pq.read_table(root / "labels.parquet")
    if table.schema != LABEL_SCHEMA:
        raise ValueError("sealed scoring label physical schema changed")
    labels = _validate_labels(table.to_pandas(), declaration)
    _check_historical_replay(
        read_json(root / "historical_replay.json"),
        declaration,
        frames["experts"],
        labels,
    )
    results = _family_reports(frames, labels)
    for family, reports in results.items():
        for name, value in reports.items():
            path = root / family / name
            if isinstance(value, pa.Table):
                compare_tables(pq.read_table(path), value)
            elif read_json(path) != value:
                raise ValueError(
                    f"sealed family report does not reproduce: {family}/{name}"
                )
    summary = _summary(declaration, results, labels)
    if read_json(root / "summary.json") != summary or (
        root / "report.md"
    ).read_text() != _report(summary):
        raise ValueError("sealed summary or human-readable report does not reproduce")
    reservation = read_json(root / "reservation.json")
    receipt = _receipt(
        build, reservation, labels, _hashes(root, SCORE_FILES - {"receipt.json"})
    )
    if (
        read_json(root / "receipt.json") != receipt
        or manifest["receipt_sha256"] != receipt["receipt_sha256"]
    ):
        raise ValueError("scoring receipt does not reproduce")
    _validate_registry(
        declaration, build, reservation, _event(receipt, root / "receipt.json")
    )
    check_snapshot(declaration["source_snapshot"])
    return {
        "status": "validated",
        "study_id": STUDY_ID,
        "score_root": str(root),
        "manifest_sha256": manifest["manifest_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "adaptive_reuse": True,
        "m4_no_go_preserved": True,
        "original_label_columns_opened": False,
        "original_historical_metrics_opened": False,
        "summary": summary,
    }


def validate_score(output_root: Path) -> dict:
    _fit_incapable()
    output_root = Path(output_root).resolve()
    with exclusive_lock(output_root):
        declaration, prediction_manifest, frames = validate_predictions(output_root)
        return _validate(output_root, declaration, prediction_manifest, frames)


def build_score(output_root: Path) -> dict:
    _fit_incapable()
    output_root = Path(output_root).resolve()
    with exclusive_lock(output_root):
        # This must finish before any source-label or historical-metric reader.
        declaration, prediction_manifest, frames = validate_predictions(output_root)
        root = output_root / "scores"
        _allowlist(root)
        if (root / "manifest.json").exists():
            return {
                **_validate(output_root, declaration, prediction_manifest, frames),
                "status": "reused_valid",
            }
        build = _build_declaration(output_root, declaration, prediction_manifest)
        if (
            root.exists()
            and any(root.iterdir())
            and not (root / "build_declaration.json").exists()
        ):
            raise ValueError(
                "unowned partial scoring artifacts lack a build declaration"
            )
        check_snapshot(declaration["source_snapshot"])
        write_json(root / "build_declaration.json", build)
        reservation = _reserve_score_event(declaration, build)
        write_json(root / "reservation.json", reservation)
        sources_sealed = (root / "source_receipt.json").exists()
        if sources_sealed:
            if read_json(root / "source_receipt.json") != _source_receipt(root, build):
                raise ValueError(
                    "partial scoring source copies differ from their acquisition receipt"
                )
            table = pq.read_table(root / "labels.parquet")
            if table.schema != LABEL_SCHEMA:
                raise ValueError(
                    "partial scoring label schema differs from frozen contract"
                )
            labels = _validate_labels(table.to_pandas(), declaration)
            _check_historical_replay(
                read_json(root / "historical_replay.json"),
                declaration,
                frames["experts"],
                labels,
            )
        else:
            # A crash before the acquisition receipt left unsealed copies.
            # Re-read source projections and compare any existing immutable
            # bytes instead of accepting an unauthenticated partial label file.
            labels = _read_original_labels(declaration)
            write_bytes(root / "labels.parquet", table_bytes(_label_table(labels)))
            write_json(
                root / "historical_replay.json",
                _read_historical_replay(declaration, frames["experts"], labels),
            )
            check_snapshot(declaration["source_snapshot"])
            write_json(root / "source_receipt.json", _source_receipt(root, build))
        results = _family_reports(frames, labels)
        for family, reports in results.items():
            for name, value in reports.items():
                path = root / family / name
                if isinstance(value, pa.Table):
                    write_bytes(path, table_bytes(value))
                else:
                    write_json(path, value)
        summary = _summary(declaration, results, labels)
        write_json(root / "summary.json", summary)
        write_bytes(root / "report.md", _report(summary).encode())
        check_snapshot(declaration["source_snapshot"])
        receipt = _receipt(
            build, reservation, labels, _hashes(root, SCORE_FILES - {"receipt.json"})
        )
        write_json(root / "receipt.json", receipt)
        event = _event(receipt, root / "receipt.json")
        paths = _registry_paths(declaration, build)
        with exclusive_lock(paths["root"]):
            write_json(paths["event"], event)
            write_json(paths["completed"], event)
        _validate_registry(declaration, build, reservation, event)
        check_snapshot(declaration["source_snapshot"])
        _allowlist(root)
        manifest = {
            "schema_version": SCHEMA,
            "study_id": STUDY_ID,
            "build_declaration": build,
            "receipt_sha256": receipt["receipt_sha256"],
            "artifact_file_sha256": _hashes(root, SCORE_FILES),
        }
        manifest["manifest_sha256"] = canonical_hash(manifest)
        write_json(root / "manifest.json", manifest)
        return {
            **_validate(output_root, declaration, prediction_manifest, frames),
            "status": "built",
            "original_label_columns_opened": not sources_sealed,
            "original_historical_metrics_opened": not sources_sealed,
        }

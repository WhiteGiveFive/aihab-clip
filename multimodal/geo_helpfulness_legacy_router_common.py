"""Immutable, apply-only legacy transfer artifacts; no fitting dependencies."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal.geo_helpfulness_router_m4_common import (
    PROJECT_ROOT,
    POLICIES,
    SEEDS,
    VARIANTS,
    canonical_bytes,
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
from multimodal.geo_helpfulness_router_candidate_state import (
    CATEGORICAL_FEATURE_SIZES,
    NUMERIC_INPUT_COLUMNS,
    TARGET_ORDER,
    full_feature_columns,
    predict_router_probabilities,
    read_candidate_state,
)
from multimodal.geo_helpfulness_router_heuristics import apply_heuristic_threshold
from multimodal.geo_helpfulness_legacy_router_report import build_semantic_values

STUDY_ID = "legacy_frozen_router_transfer_v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/exploratory_legacy_router_v1"
)
FAMILIES = ("best_model", "oracle_test_best_model")
MODES = {"image": "image_only", "geo": "geo_only", "raw": "raw_concat"}
LOGIT_COLUMNS = [f"{name}_logits" for name in MODES]
LOGIT_INPUT_COLUMNS = [
    "checkpoint_family",
    "row_uid",
    "file",
    "plot_idx",
    "training_seed",
    *LOGIT_COLUMNS,
]
NEVER = {"comparison": "strict_gt", "kind": "never_intervene", "value_hex": None}
ZERO = {"comparison": "strict_gt", "kind": "finite", "value_hex": (0.0).hex()}
PREDICTION_FILES = {
    "experts.parquet",
    "routers.parquet",
    "actions.parquet",
    "metadata.json",
}


def normalize_identity(file, plot) -> tuple[str, str]:
    """The M1 relative-path/opaque-plot normalization without protocol imports."""
    if not isinstance(file, str) or not isinstance(plot, str):
        raise ValueError("identity fields must be strings")
    file = unicodedata.normalize("NFC", file).replace("\\", "/")
    plot = unicodedata.normalize("NFC", plot)
    if (
        not file
        or file != file.strip()
        or file.startswith("/")
        or re.match(r"^[A-Za-z]:", file)
        or not plot
        or plot != plot.strip()
        or any(unicodedata.category(c) in {"Cc", "Cs"} for c in file + plot)
    ):
        raise ValueError("invalid file/plot identity")
    parts = [part for part in file.split("/") if part not in ("", ".")]
    if not parts or ".." in parts or re.match(r"^[A-Za-z]:", parts[0]):
        raise ValueError("invalid relative file identity")
    return "/".join(parts).casefold(), plot


def load_declaration(output_root: Path) -> dict:
    declaration = read_json(Path(output_root) / "declaration.json")
    check_self_hash(declaration, "declaration_sha256")
    if (
        declaration.get("study_id") != STUDY_ID
        or declaration.get("no_go_preserved") is not True
    ):
        raise ValueError("not the fixed exploratory transfer declaration")
    if set(declaration["caches"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("declaration must retain all four seeds")
    check_snapshot(declaration["source_snapshot"])
    return declaration


def transferred_probability(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits)
    if (
        logits.dtype != np.float64
        or logits.ndim != 2
        or logits.shape[1] != 18
        or not len(logits)
        or not np.isfinite(logits).all()
        or isinstance(temperature, (bool, np.bool_))
        or not isinstance(temperature, (float, int, np.floating, np.integer))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("finite float64 Nx18 logits and positive temperature required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        centered = (logits - logits.max(axis=1, keepdims=True)) / temperature
        weights = np.exp(centered)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)
        or not np.array_equal(probabilities.argmax(axis=1), logits.argmax(axis=1))
    ):
        raise ValueError("probability normalization or prediction parity failed")
    return probabilities


def apply_transform(features: pd.DataFrame, state: dict) -> np.ndarray:
    if (
        state["numeric_input_columns"] != list(NUMERIC_INPUT_COLUMNS)
        or state["output_columns"] != full_feature_columns()
    ):
        raise ValueError("frozen transform feature order mismatch")
    mean, variance, scale = (
        np.asarray(state[key], dtype=np.float64)
        for key in ("mean", "variance", "scale")
    )
    if (
        any(
            value.shape != (25,) or not np.isfinite(value).all()
            for value in (mean, variance, scale)
        )
        or np.any(variance < 0)
        or np.any(scale <= 0)
        or not np.array_equal(scale, np.where(variance == 0, 1.0, np.sqrt(variance)))
    ):
        raise ValueError("invalid frozen numeric transform")
    values = features.loc[:, list(NUMERIC_INPUT_COLUMNS)].to_numpy(dtype=np.float64)
    matrix = np.zeros((len(features), 727), dtype=np.float64)
    matrix[:, :25] = (values - mean) / scale
    offset = 25
    for name, size in CATEGORICAL_FEATURE_SIZES:
        if state["categorical_vocabularies"][name] != list(range(size)):
            raise ValueError("frozen categorical vocabulary mismatch")
        codes = features[name].to_numpy()
        if codes.dtype.kind not in "iu" or np.any(codes < 0) or np.any(codes >= size):
            raise ValueError(f"invalid category: {name}")
        matrix[np.arange(len(features)), offset + codes] = 1.0
        offset += size
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite transformed features")
    return matrix


def _schema(kind: str, feature_schema: dict) -> pa.Schema:
    fields = [
        ("schema_version", pa.string()),
        ("study_id", pa.string()),
        ("checkpoint_family", pa.string()),
        ("row_uid", pa.string()),
    ]
    if kind == "experts":
        fields.append(("file", pa.string()))
    fields.extend([("plot_idx", pa.string()), ("training_seed", pa.int8())])
    if kind == "experts":
        fields.extend((name, pa.list_(pa.float64(), 18)) for name in LOGIT_COLUMNS)
        fields.extend(
            (f"{name}_prob_{basis}", pa.list_(pa.float64(), 18))
            for name in MODES
            for basis in ("native_t1", "calibrated")
        )
        types = {
            "int8": pa.int8(),
            "int16": pa.int16(),
            "bool": pa.bool_(),
            "float64": pa.float64(),
        }
        fields.extend(
            (name, types[feature_schema["dtypes"][name]])
            for name in feature_schema["ordered_semantic_features"]
        )
    elif kind == "routers":
        fields.extend([("variant_id", pa.string()), ("C_hex", pa.string())])
        fields.extend((f"q_{name}", pa.float64()) for name in TARGET_ORDER)
        fields.append(("score", pa.float64()))
    elif kind == "actions":
        fields.extend(
            [
                ("policy_id", pa.string()),
                ("score", pa.float64()),
                ("threshold_specification", pa.string()),
                ("acted", pa.bool_()),
                ("raw_pred", pa.int8()),
                ("geo_pred", pa.int8()),
                ("final_pred", pa.int8()),
            ]
        )
    else:
        raise ValueError("unknown prediction table")
    return pa.schema([pa.field(name, dtype, nullable=False) for name, dtype in fields])


def _table(frame: pd.DataFrame, kind: str, feature_schema: dict) -> pa.Table:
    frame = frame.copy()
    frame["schema_version"] = f"geo_helpfulness.legacy_transfer_{kind}.v1"
    frame["study_id"] = STUDY_ID
    schema = _schema(kind, feature_schema)
    table = pa.Table.from_pandas(
        frame.loc[:, schema.names], schema=schema, preserve_index=False, safe=True
    ).replace_schema_metadata(None)
    if any(table[name].null_count for name in table.column_names):
        raise ValueError("null inference table values")
    return table


def reconstruct_predictions(
    logits_frame: pd.DataFrame, declaration: dict
) -> dict[str, pa.Table]:
    """Reapply immutable M4 components to authoritative saved expert logits."""
    if list(logits_frame.columns) != LOGIT_INPUT_COLUMNS:
        raise ValueError("logit input is a strict label-blind projection")
    root = Path(declaration["frozen_root"])
    frozen = read_json(root / "router_frozen_state.json")
    for name, record in frozen["components"].items():
        if file_hash(root / name) != record["file_sha256"]:
            raise ValueError(f"frozen component mismatch: {name}")
    temperatures = {
        (s["seed"], s["mode"]): s["temperature"]
        for s in read_json(root / "expert_temperatures.json")["states"]
    }
    transforms = {
        s["seed"]: s
        for s in read_json(root / "router_feature_transform.json")["states"]
    }
    if len(temperatures) != 12 or set(transforms) != set(SEEDS):
        raise ValueError("expected twelve temperatures and four transforms")
    feature_schema = read_json(root / "router_feature_schema.json")
    heuristics = read_json(root / "oof_heuristics.json")["heuristics"]
    identity = (
        pd.DataFrame(declaration["identity_manifest"]["rows"])[
            ["row_uid", "file", "plot_idx"]
        ]
        .sort_values("row_uid")
        .reset_index(drop=True)
    )
    canonical = logits_frame.sort_values(
        ["checkpoint_family", "row_uid", "training_seed"]
    ).reset_index(drop=True)
    if not logits_frame.reset_index(drop=True).equals(canonical) or set(
        logits_frame.checkpoint_family
    ) != set(FAMILIES):
        raise ValueError("canonical ordering and both checkpoint families required")
    experts, routers, actions = [], [], []
    for family in FAMILIES:
        family_frame = logits_frame[logits_frame.checkpoint_family == family]
        if set(family_frame.training_seed) != set(SEEDS):
            raise ValueError("missing training seed")
        for seed in SEEDS:
            frame = (
                family_frame[family_frame.training_seed == seed]
                .reset_index(drop=True)
                .copy()
            )
            if not frame[["row_uid", "file", "plot_idx"]].equals(identity):
                raise ValueError("expert identities differ from frozen test universe")
            probability = {}
            for name, mode in MODES.items():
                logits = np.stack(frame[f"{name}_logits"].to_numpy())
                native = transferred_probability(logits, 1.0)
                probability[name] = transferred_probability(
                    logits, temperatures[(seed, mode)]
                )
                frame[f"{name}_prob_native_t1"] = list(native)
                frame[f"{name}_prob_calibrated"] = list(probability[name])
            values = build_semantic_values(probability)
            for name in feature_schema["ordered_semantic_features"]:
                frame[name] = values[name].astype(feature_schema["dtypes"][name])
            matrix = apply_transform(frame, transforms[seed])
            base = frame[["checkpoint_family", "row_uid", "plot_idx", "training_seed"]]
            scores, specifications = {}, {}
            for variant in VARIANTS:
                selected = frozen["variants"][variant]["seeds"][str(seed)]
                state = read_candidate_state(root / selected["model_file"])
                if (
                    state["state_sha256"] != selected["model_state_sha256"]
                    or state["training_seed"] != seed
                    or state["variant_id"] != variant
                    or state["C_hex"] != selected["C_hex"]
                ):
                    raise ValueError("selected router identity mismatch")
                q = predict_router_probabilities(matrix, state)
                scores[variant] = q[:, 0] - q[:, 1]
                specifications[variant] = selected["threshold_specification"]
                record = base.copy()
                record["variant_id"], record["C_hex"] = variant, selected["C_hex"]
                for index, name in enumerate(TARGET_ORDER):
                    record[f"q_{name}"] = q[:, index]
                record["score"] = scores[variant]
                routers.append(record)
            scores.update(
                {
                    "never_intervene": np.zeros(len(frame)),
                    "always_geo_on_geo_raw_disagreement": np.ones(len(frame)),
                    "higher_calibrated_geo_confidence_on_geo_raw_disagreement": values[
                        "geo_minus_raw_confidence"
                    ],
                    "image_geo_jsd": values["image_geo_jsd"],
                    "geo_minus_raw_margin": values["geo_minus_raw_margin"],
                }
            )
            specifications.update(
                {
                    "never_intervene": NEVER,
                    "always_geo_on_geo_raw_disagreement": ZERO,
                    "higher_calibrated_geo_confidence_on_geo_raw_disagreement": ZERO,
                    **{
                        name: item["threshold_specification"]
                        for name, item in heuristics.items()
                    },
                }
            )
            for policy in POLICIES:
                acted = apply_heuristic_threshold(
                    scores[policy],
                    values["raw_pred"],
                    values["geo_pred"],
                    specifications[policy],
                )
                if policy == "primary_full727" and (
                    specifications[policy] != NEVER or acted.any()
                ):
                    raise ValueError("M4 no-go primary fallback must be preserved")
                record = base.copy()
                record["policy_id"], record["score"] = policy, scores[policy]
                record["threshold_specification"] = canonical_bytes(
                    specifications[policy]
                ).decode()
                record["acted"] = acted
                record["raw_pred"], record["geo_pred"] = (
                    values["raw_pred"],
                    values["geo_pred"],
                )
                record["final_pred"] = np.where(
                    acted, values["geo_pred"], values["raw_pred"]
                )
                actions.append(record)
            experts.append(frame)
    result = {}
    for kind, items, middle in (
        ("experts", experts, []),
        ("routers", routers, ["variant_id"]),
        ("actions", actions, ["policy_id"]),
    ):
        frame = (
            pd.concat(items, ignore_index=True)
            .sort_values(["checkpoint_family", *middle, "row_uid", "training_seed"])
            .reset_index(drop=True)
        )
        result[kind] = _table(frame, kind, feature_schema)
    return result


def prediction_metadata(declaration: dict, tables: dict) -> dict:
    return {
        "schema_version": "geo_helpfulness.legacy_transfer_prediction_metadata.v1",
        "study_id": STUDY_ID,
        "declaration_sha256": declaration["declaration_sha256"],
        "checkpoint_families": list(FAMILIES),
        "training_seeds": list(SEEDS),
        "policies": list(POLICIES),
        "router_refitting": False,
        "test_labels_opened": False,
        "calibration": "M4 temperatures transferred unchanged; legacy calibration is not guaranteed",
        "oracle_interpretation": "test-selected biased diagnostic, not independent evidence or an oracle ceiling",
        "dimensions": {
            name: [table.num_rows, table.num_columns] for name, table in tables.items()
        },
        "comparison_atol": 1e-12,
        "comparison_rtol": 1e-12,
        "owned_filenames": sorted(PREDICTION_FILES | {"manifest.json"}),
    }


def publish_predictions(output_root: Path, declaration: dict, tables: dict) -> dict:
    output_root = Path(output_root)
    root = output_root / "predictions"
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("prediction publication must not follow symlinks")
    with exclusive_lock(root):
        if (root / "manifest.json").exists():
            saved_declaration, manifest, frames = validate_predictions(output_root)
            if saved_declaration != declaration:
                raise ValueError("prediction declaration changed")
            for name, table in tables.items():
                compare_tables(pq.read_table(root / f"{name}.parquet"), table)
            return {
                "status": "reused_valid",
                "manifest_sha256": manifest["manifest_sha256"],
            }
        existing = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if existing - PREDICTION_FILES or (
            existing and "metadata.json" not in existing
        ):
            raise ValueError("unknown or unowned partial prediction outputs")
        metadata = prediction_metadata(declaration, tables)
        write_json(root / "metadata.json", metadata)
        for name, table in tables.items():
            write_bytes(root / f"{name}.parquet", table_bytes(table))
        check_snapshot(declaration["source_snapshot"])
        manifest = {
            "schema_version": "geo_helpfulness.legacy_transfer_predictions.v1",
            "study_id": STUDY_ID,
            "declaration_file_sha256": file_hash(output_root / "declaration.json"),
            "source_snapshot": declaration["source_snapshot"],
            "artifacts": {
                name: {"file_sha256": file_hash(root / name)}
                for name in sorted(PREDICTION_FILES)
            },
        }
        manifest["manifest_sha256"] = canonical_hash(manifest)
        write_json(root / "manifest.json", manifest)
    validate_predictions(output_root)
    return {"status": "built", "manifest_sha256": manifest["manifest_sha256"]}


def validate_predictions(output_root: Path):
    """Reload/reconstruct sealed predictions with no original-label or fit capability."""
    output_root = Path(output_root)
    declaration = load_declaration(output_root)
    root = output_root / "predictions"
    manifest = read_json(root / "manifest.json")
    check_self_hash(manifest)
    expected_files = PREDICTION_FILES | {"manifest.json"}
    observed_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if (
        root.is_symlink()
        or observed_files != expected_files
        or set(manifest["artifacts"]) != PREDICTION_FILES
    ):
        raise ValueError("prediction bundle file allowlist mismatch")
    if (
        manifest.get("study_id") != STUDY_ID
        or manifest.get("schema_version")
        != "geo_helpfulness.legacy_transfer_predictions.v1"
        or manifest["declaration_file_sha256"]
        != file_hash(output_root / "declaration.json")
        or manifest["source_snapshot"] != declaration["source_snapshot"]
    ):
        raise ValueError("prediction manifest lineage mismatch")
    for name, record in manifest["artifacts"].items():
        if file_hash(root / name) != record["file_sha256"]:
            raise ValueError(f"prediction artifact changed: {name}")
    tables = {
        name: pq.read_table(root / f"{name}.parquet")
        for name in ("experts", "routers", "actions")
    }
    expected = reconstruct_predictions(
        tables["experts"].select(LOGIT_INPUT_COLUMNS).to_pandas(), declaration
    )
    for name, table in tables.items():
        compare_tables(table, expected[name])
    if read_json(root / "metadata.json") != prediction_metadata(declaration, expected):
        raise ValueError("prediction metadata does not reproduce")
    check_snapshot(declaration["source_snapshot"])
    return (
        declaration,
        manifest,
        {name: table.to_pandas() for name, table in tables.items()},
    )

#!/usr/bin/env python3
"""Freeze and validate the reliability-aware geo-helpfulness protocol.

M1 intentionally implements only metadata/identity freezing and validation.
Later training entry points exist but fail closed.  Locked prediction and
scoring are executable only for synthetic fixtures so their unequal
capabilities can be tested without opening the real cleaned-test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal.geo_helpfulness_locked_eval import (
    canonical_json_bytes,
    exclusive_write_bytes,
    exclusive_write_json,
    locked_predict_synthetic,
    locked_score_synthetic,
    sha256_file,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "multimodal_geo_helpfulness.yaml"
M1_NOT_IMPLEMENTED = (
    "is not implemented in M1; complete and validate protocol_v1 "
    "before running any real expert, router, or final-fit work"
)
_PROTOCOL_FILES = (
    "development_assignments.parquet",
    "split_balance.csv",
    "resolved_protocol.yaml",
    "locked_test_snapshot_ref.json",
    "protocol_manifest.json",
)
_FORBIDDEN_CONFIG_KEYS = {
    "locked_test_identity_source",
    "locked_test_source",
    "cleaned_test_source",
    "cleaned_test_table",
    "test_source_table",
    "test_table",
    "test_labels",
    "locked_test_labels",
}


class ProtocolCommandError(ValueError):
    """Raised for a fail-closed protocol command violation."""


def _core():
    # Kept lazy so --help and fail-closed future commands do not import protocol
    # implementation code or touch any data path.
    from multimodal import geo_helpfulness_protocol

    return geo_helpfulness_protocol


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ProtocolCommandError(f"Cannot read protocol config {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolCommandError(f"Protocol config must be a YAML mapping: {source}")
    _assert_no_test_source_config(value)
    return value


def _assert_no_test_source_config(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject shared configuration that can resolve the underlying test data."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            here = (*path, key)
            if key in _FORBIDDEN_CONFIG_KEYS:
                raise ProtocolCommandError(
                    "Shared protocol configuration must not contain an underlying "
                    f"locked-test path ({'.'.join(here)}); pass it only as "
                    "freeze-test-identity --source"
                )
            _assert_no_test_source_config(child, here)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_test_source_config(child, (*path, str(index)))


def _path(value: str | Path) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _required(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise ProtocolCommandError(f"Missing {context}.{key}")
    return mapping[key]


def _read_columns(path: Path, columns: Sequence[str], *, name: str) -> pd.DataFrame:
    """Read only an explicit projection from a source table."""

    if not path.is_file():
        raise ProtocolCommandError(f"{name} does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path, columns=list(columns))
        if suffix == ".csv":
            return pd.read_csv(path, usecols=list(columns))
    except Exception as exc:
        raise ProtocolCommandError(
            f"Cannot read the declared columns from {name} {path}: {exc}"
        ) from exc
    raise ProtocolCommandError(f"{name} must be parquet or CSV: {path}")


def _json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolCommandError(f"Cannot read {name} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolCommandError(f"{name} must be a JSON object: {source}")
    return value


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exclusive_write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    """Reserve a parquet artifact using O_EXCL before serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    os.close(descriptor)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def _identity_hash(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("identity_projection_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolCommandError(
            "Locked-test manifest has no valid identity_projection_sha256"
        )
    return value.lower()


def _manifest_relative_path(active: Mapping[str, Any], registry_root: Path) -> Path:
    raw = active.get("manifest_relative_path", active.get("manifest_path"))
    if not isinstance(raw, str) or not raw:
        snapshot_id = active.get("snapshot_id", active.get("identity_projection_sha256"))
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ProtocolCommandError("Active snapshot has no manifest path or snapshot ID")
        raw = f"{snapshot_id}/locked_test_identity_manifest.json"
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolCommandError("Active snapshot manifest path must remain inside registry")
    resolved = (registry_root / relative).resolve()
    try:
        resolved.relative_to(registry_root.resolve())
    except ValueError as exc:
        raise ProtocolCommandError("Active snapshot manifest escapes its registry") from exc
    return resolved


def _load_active_snapshot(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    paths = _required(config, "paths", context="config")
    if not isinstance(paths, Mapping):
        raise ProtocolCommandError("config.paths must be a mapping")
    registry_root = _path(
        _required(paths, "locked_test_registry_root", context="config.paths")
    )
    active_path = registry_root / "active_snapshot.json"
    active = _json(active_path, name="active locked-test snapshot")
    manifest_path = _manifest_relative_path(active, registry_root)
    manifest = _json(manifest_path, name="locked-test identity manifest")
    if active.get("identity_projection_sha256") not in {None, _identity_hash(manifest)}:
        raise ProtocolCommandError("Active snapshot identity hash does not match its manifest")
    expected_file_hash = active.get("manifest_file_sha256")
    if expected_file_hash is not None and expected_file_hash != sha256_file(manifest_path):
        raise ProtocolCommandError("Active snapshot manifest file hash mismatch")
    _core().validate_locked_test_identity_manifest(manifest)
    return registry_root, active, manifest_path, manifest


def freeze_test_identity(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_yaml(args.config)
    source = _path(args.source)
    paths = _required(config, "paths", context="config")
    registry_root = _path(
        args.registry_root
        if args.registry_root is not None
        else _required(paths, "locked_test_registry_root", context="config.paths")
    )
    configured_dataset_id = str(_required(config, "dataset_id", context="config"))
    if args.dataset_id is not None and str(args.dataset_id) != configured_dataset_id:
        raise ProtocolCommandError(
            "--dataset-id must match the frozen config dataset_id: "
            f"{args.dataset_id!r} != {configured_dataset_id!r}"
        )
    dataset_id = configured_dataset_id
    active_path = registry_root / "active_snapshot.json"
    if active_path.exists():
        raise FileExistsError(
            f"Global locked-test snapshot is already sealed; refusing replacement: {active_path}"
        )

    identity_cfg = config.get("locked_test_identity", {})
    source_projection = identity_cfg.get("source_projection_allowlist")
    if source_projection != ["file", "plot_idx"]:
        raise ProtocolCommandError(
            "locked_test_identity.source_projection_allowlist must be exactly "
            "['file', 'plot_idx']"
        )

    # This is the only command allowed to open the underlying test source, and
    # it asks the storage engine for exactly the two frozen identity columns.
    # Column-name overrides are deliberately not exposed: otherwise a caller
    # could project labels or features while invoking an identity-only command.
    source_frame = _read_columns(
        source,
        source_projection,
        name="locked-test identity source",
    )
    projection = _core().build_identity_projection(
        source_frame,
        dataset_id,
        file_column="file",
        plot_column="plot_idx",
    )
    expected_rows = identity_cfg.get("expected_rows")
    expected_plots = identity_cfg.get("expected_plots")
    if expected_rows is not None and len(projection) != int(expected_rows):
        raise ProtocolCommandError(
            f"Locked-test row count mismatch: {len(projection)} != {expected_rows}"
        )
    if expected_plots is not None and projection["normalized_plot_idx"].nunique() != int(
        expected_plots
    ):
        raise ProtocolCommandError(
            "Locked-test plot count mismatch: "
            f"{projection['normalized_plot_idx'].nunique()} != {expected_plots}"
        )

    manifest = _core().build_locked_test_identity_manifest(projection, dataset_id)
    validation = _core().validate_locked_test_identity_manifest(manifest)
    identity_sha256 = _identity_hash(manifest)
    snapshot_id = str(manifest.get("snapshot_id", identity_sha256))
    manifest_path = (
        registry_root / identity_sha256 / "locked_test_identity_manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite sealed identity manifest: {manifest_path}")
    exclusive_write_json(manifest_path, manifest)
    active = {
        "schema_version": "geo_helpfulness.locked_test_active_snapshot.v1",
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "identity_projection_sha256": identity_sha256,
        "manifest_relative_path": manifest_path.relative_to(registry_root).as_posix(),
        "manifest_file_sha256": sha256_file(manifest_path),
    }
    exclusive_write_json(active_path, active)
    return {
        "status": "sealed",
        "active_snapshot": str(active_path),
        "identity_manifest": str(manifest_path),
        "identity_projection_sha256": identity_sha256,
        "validation": validation,
    }


def _validate_freezable_config(config: Mapping[str, Any]) -> None:
    universe = config.get("development_universe", {})
    assignment = config.get("assignment", {})
    role = assignment.get("role", {}) if isinstance(assignment, Mapping) else {}
    expected_plots = int(_required(universe, "expected_plots", context="development_universe"))
    exact_train = int(_required(role, "exact_train_plots", context="assignment.role"))
    exact_validation = int(
        _required(role, "exact_validation_plots", context="assignment.role")
    )
    if exact_train + exact_validation != expected_plots:
        raise ProtocolCommandError(
            "Frozen train and validation plot counts must sum to expected_plots"
        )

    ontology_cfg = config.get("class_ontology", {})
    classes = ontology_cfg.get("classes") if isinstance(ontology_cfg, Mapping) else None
    output_size = int(_required(ontology_cfg, "output_size", context="class_ontology"))
    if not isinstance(classes, list) or len(classes) != output_size:
        raise ProtocolCommandError("Frozen ontology length must equal output_size")
    dense_ids = [int(row["dense_id"]) for row in classes if isinstance(row, Mapping)]
    canonical_ids = [
        int(row["canonical_l3_id"]) for row in classes if isinstance(row, Mapping)
    ]
    label_names = [str(row["label_name"]) for row in classes if isinstance(row, Mapping)]
    if dense_ids != list(range(output_size)):
        raise ProtocolCommandError("Frozen ontology dense IDs must be ordered 0..K-1")
    if len(set(canonical_ids)) != output_size or len(set(label_names)) != output_size:
        raise ProtocolCommandError("Frozen ontology IDs and names must be unique")

    experts = config.get("experts")
    if not isinstance(experts, Mapping):
        raise ProtocolCommandError("config.experts must be a mapping")
    encoder = experts.get("image_encoder")
    if not isinstance(encoder, Mapping):
        raise ProtocolCommandError("config.experts.image_encoder must be a mapping")
    strategy = encoder.get("strategy")
    allowed = encoder.get("allowed_strategies", [])
    if strategy not in allowed:
        raise ProtocolCommandError(
            "Encoder strategy is unresolved. Select exactly one of "
            f"{list(allowed)} before freezing protocol_v1."
        )
    if strategy == "fold_contained_adaptation":
        recipe = encoder.get("fold_contained_adaptation", {}).get("adaptation_recipe")
        if not isinstance(recipe, Mapping):
            raise ProtocolCommandError(
                "fold_contained_adaptation requires a fully resolved adaptation_recipe mapping"
            )
        unresolved = _find_unresolved_marker(recipe)
        if unresolved is not None:
            raise ProtocolCommandError(
                "fold_contained_adaptation recipe is unresolved at "
                f"{unresolved}"
            )
        text_tower = recipe.get("text_tower")
        if not isinstance(text_tower, Mapping) or text_tower.get("trainable") is not False:
            raise ProtocolCommandError(
                "protocol-v1 fold-contained adaptation must keep the text tower frozen"
            )
        prompt_values = recipe.get("prompts", {}).get("values")
        if not isinstance(prompt_values, list) or len(prompt_values) != output_size:
            raise ProtocolCommandError(
                "fold-contained adaptation requires exactly one frozen prompt per class"
            )
    if int(experts.get("output_classes", -1)) != output_size:
        raise ProtocolCommandError("Expert output_classes must equal ontology output_size")
    graph = experts.get("execution_graph")
    if not isinstance(graph, Mapping):
        raise ProtocolCommandError("experts.execution_graph must be a mapping")
    stages = graph.get("stages")
    encoder_fits = int(graph.get("encoder_fits_per_training_seed", -1))
    heads_per_stage = int(graph.get("expert_head_fits_per_encoder_stage", -1))
    if not isinstance(stages, list) or len(stages) != encoder_fits:
        raise ProtocolCommandError("Execution stage count must equal encoder fits per seed")
    if int(graph.get("expert_head_fits_per_training_seed", -1)) != (
        encoder_fits * heads_per_stage
    ):
        raise ProtocolCommandError("Expert head-fit totals are internally inconsistent")

    router = config.get("router", {})
    allowlist = router.get("feature_allowlist", {}) if isinstance(router, Mapping) else {}
    matrix = router.get("feature_matrix", {}) if isinstance(router, Mapping) else {}
    try:
        scaled_count = sum(
            len(allowlist[family]) for family in ("boolean", "integer", "numeric")
        )
        categorical = allowlist["categorical"]
    except (KeyError, TypeError) as exc:
        raise ProtocolCommandError("Router feature allow-list is incomplete") from exc
    if scaled_count != int(matrix.get("scaled_numeric_column_count", -1)):
        raise ProtocolCommandError("Router scaled numeric column count is inconsistent")
    categorical_widths = {
        "image_pred": output_size,
        "geo_pred": output_size,
        "raw_pred": output_size,
        "image_geo_pred_pair": output_size * output_size,
        "geo_raw_pred_pair": output_size * output_size,
    }
    try:
        one_hot_count = sum(categorical_widths[name] for name in categorical)
    except KeyError as exc:
        raise ProtocolCommandError(f"Unknown router categorical feature: {exc}") from exc
    if one_hot_count != int(matrix.get("one_hot_column_count", -1)):
        raise ProtocolCommandError("Router one-hot column count is inconsistent")
    if scaled_count + one_hot_count != int(matrix.get("total_column_count", -1)):
        raise ProtocolCommandError("Router total feature-matrix width is inconsistent")
    status = str(config.get("protocol_status", "")).strip().lower()
    if status != "frozen":
        raise ProtocolCommandError(
            "protocol_status must be frozen before artifacts are sealed"
        )


def _find_unresolved_marker(value: Any, path: tuple[str, ...] = ()) -> str | None:
    """Return the first active recipe field that still contains a freeze marker."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            unresolved = _find_unresolved_marker(child, (*path, str(key)))
            if unresolved is not None:
                return unresolved
    elif isinstance(value, list):
        for index, child in enumerate(value):
            unresolved = _find_unresolved_marker(child, (*path, str(index)))
            if unresolved is not None:
                return unresolved
    elif isinstance(value, str) and (
        value.startswith("REQUIRES_USER_SELECTION")
        or value.startswith("MUST_BE_RESOLVED")
    ):
        return ".".join(path) or "<root>"
    return None


def _development_frame(config: Mapping[str, Any]) -> tuple[pd.DataFrame, list[Path]]:
    paths = _required(config, "paths", context="config")
    source_values = _required(
        paths, "development_source_tables", context="config.paths"
    )
    if not isinstance(source_values, list) or not source_values:
        raise ProtocolCommandError("development_source_tables must be a nonempty list")
    columns = config.get("development_universe", {}).get("source_columns", {})
    if not isinstance(columns, Mapping):
        raise ProtocolCommandError("development_universe.source_columns must be a mapping")
    required_keys = (
        "file",
        "plot_idx",
        "source_split",
        "image_source",
        "label_name",
    )
    missing = [key for key in required_keys if key not in columns]
    if missing:
        raise ProtocolCommandError(f"Missing development source column mappings: {missing}")
    source_columns = [str(columns[key]) for key in required_keys]
    source_paths = [_path(value) for value in source_values]
    frames = [
        _read_columns(path, source_columns, name="development source table")
        for path in source_paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    rename = {
        str(columns["file"]): "file",
        str(columns["plot_idx"]): "plot_idx",
        str(columns["source_split"]): "source_split",
        str(columns["image_source"]): "image_source",
        str(columns["label_name"]): "label_name",
    }
    frame = frame.rename(columns=rename)
    source_roles = config.get("development_universe", {}).get("source_roles")
    if not isinstance(source_roles, list) or not source_roles:
        raise ProtocolCommandError("development_universe.source_roles must be nonempty")
    observed_roles = set(frame["source_split"].astype(str))
    if observed_roles != set(map(str, source_roles)):
        raise ProtocolCommandError(
            "Development source roles do not match the frozen union: "
            f"observed={sorted(observed_roles)}, expected={sorted(map(str, source_roles))}"
        )

    ontology = config.get("class_ontology", {}).get("classes")
    if not isinstance(ontology, list) or not ontology:
        raise ProtocolCommandError("class_ontology.classes must be a nonempty list")
    dense_to_canonical: dict[int, int] = {}
    dense_to_name: dict[int, str] = {}
    for row in ontology:
        if not isinstance(row, Mapping):
            raise ProtocolCommandError("Every ontology entry must be a mapping")
        dense = int(row["dense_id"])
        dense_to_canonical[dense] = int(row["canonical_l3_id"])
        dense_to_name[dense] = str(row["label_name"])
    name_to_dense = {name: dense for dense, name in dense_to_name.items()}
    name_to_canonical = {
        dense_to_name[dense]: canonical
        for dense, canonical in dense_to_canonical.items()
    }
    if bool(frame["label_name"].isna().any()):
        raise ProtocolCommandError("Development label names must be non-null")
    actual_names = frame["label_name"].astype("string")
    unknown_names = sorted(set(actual_names).difference(name_to_dense))
    if unknown_names:
        raise ProtocolCommandError(
            f"Development rows use names outside the frozen ontology: {unknown_names}"
        )
    frame["label_name"] = actual_names
    frame["label_id_dense"] = actual_names.map(name_to_dense).astype("int8")
    frame["canonical_l3_id"] = actual_names.map(name_to_canonical).astype("int8")
    return frame, source_paths


def _assignment_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    assignment = config.get("assignment", {})
    role = assignment.get("role", {})
    oof = assignment.get("train_oof", {})
    universe = config.get("development_universe", {})
    return {
        "dataset_id": str(config["dataset_id"]),
        "protocol_id": str(config["protocol_id"]),
        "role_seed": int(role["seed"]),
        "oof_seed": int(oof["random_state"]),
        "validation_plot_count": int(role["exact_validation_plots"]),
        "n_oof_folds": int(oof["n_splits"]),
        "expected_rows": int(universe["expected_rows"]),
        "expected_plots": int(universe["expected_plots"]),
    }


def _split_balance(assignments: pd.DataFrame) -> pd.DataFrame:
    balance = (
        assignments.groupby(
            ["canonical_l3_id", "label_name", "development_role", "train_oof_fold"],
            dropna=False,
            observed=True,
        )
        .agg(image_count=("row_uid", "size"), plot_count=("plot_idx", "nunique"))
        .reset_index()
    )
    balance["train_oof_fold"] = balance["train_oof_fold"].astype("Int8")
    return balance.sort_values(
        ["canonical_l3_id", "development_role", "train_oof_fold"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _development_image_fingerprint(
    config: Mapping[str, Any],
    assignments: pd.DataFrame,
) -> dict[str, Any]:
    """Validate and hash every raw development image used by encoder fits."""

    universe = config.get("development_universe", {})
    allowed_values = universe.get("allowed_image_sources")
    if not isinstance(allowed_values, list) or not allowed_values:
        raise ProtocolCommandError(
            "development_universe.allowed_image_sources must be a nonempty list"
        )
    allowed_text = [str(value) for value in allowed_values]
    if len(set(allowed_text)) != len(allowed_text):
        raise ProtocolCommandError("allowed development image sources must be unique")
    source_roots = {value: _path(value) for value in allowed_text}
    observed_sources = set(assignments["image_source"].astype(str))
    if observed_sources != set(source_roots):
        raise ProtocolCommandError(
            "Development image sources do not match the frozen allow-list: "
            f"observed={sorted(observed_sources)}, allowed={sorted(source_roots)}"
        )

    records: list[dict[str, str]] = []
    resolved_paths: set[Path] = set()
    core = _core()
    for row in assignments.sort_values("row_uid", kind="mergesort").itertuples(
        index=False
    ):
        source = str(row.image_source)
        root = source_roots[source].resolve()
        canonical_file = core.canonicalize_file(row.file)
        candidate = (root / canonical_file).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ProtocolCommandError(
                f"Development image escapes its declared source root: {candidate}"
            ) from exc
        if not candidate.is_file():
            raise ProtocolCommandError(f"Development image does not exist: {candidate}")
        if candidate in resolved_paths:
            raise ProtocolCommandError(
                f"Multiple development rows resolve to the same image: {candidate}"
            )
        resolved_paths.add(candidate)
        records.append(
            {
                "row_uid": str(row.row_uid),
                "image_file_sha256": sha256_file(candidate),
            }
        )
    return {
        "development_image_file_count": len(records),
        "development_image_content_sha256": _hash_json(records),
    }


def _resolved_yaml_bytes(config: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(config),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).encode("utf-8")


def _git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.decode("utf-8", errors="replace").strip()


def _environment() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyyaml": str(yaml.__version__),
    }
    for module_name in ("numpy", "sklearn", "pyarrow"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(module.__version__)
        except (ImportError, AttributeError):
            versions[module_name] = "unavailable"
    for distribution, key in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("scipy", "scipy"),
        ("open_clip_torch", "open_clip_torch"),
        ("timm", "timm"),
        ("opencv-python", "opencv_python"),
        ("Pillow", "pillow"),
        ("huggingface-hub", "huggingface_hub"),
        ("tokenizers", "tokenizers"),
        ("safetensors", "safetensors"),
    ):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def _protocol_manifest(
    *,
    config: Mapping[str, Any],
    source_paths: Sequence[Path],
    assignments: pd.DataFrame,
    validation: Mapping[str, Any],
    output_dir: Path,
    identity_manifest_path: Path,
    identity_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    core = _core()
    artifact_hashes = {
        filename: sha256_file(output_dir / filename)
        for filename in _PROTOCOL_FILES
        if filename != "protocol_manifest.json"
    }
    ontology = config["class_ontology"]
    code_files = [
        PROJECT_ROOT / "multimodal" / "geo_helpfulness_protocol.py",
        PROJECT_ROOT / "multimodal" / "geo_helpfulness_locked_eval.py",
        Path(__file__).resolve(),
    ]
    code_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in code_files
    }
    environment = _environment()
    identity_columns = ["row_uid", "file_lower", "plot_idx"]
    identity_records = (
        assignments.loc[:, identity_columns]
        .sort_values(identity_columns, kind="mergesort")
        .to_dict(orient="records")
    )
    feature_allowlist = config.get("router", {}).get("feature_allowlist", {})
    image_fingerprint = _development_image_fingerprint(config, assignments)
    manifest: dict[str, Any] = {
        "schema_version": "geo_helpfulness.protocol_manifest.v1",
        "artifact_role": "frozen_experimental_protocol",
        "protocol_id": str(config["protocol_id"]),
        "dataset_id": str(config["dataset_id"]),
        "protocol_status": "frozen",
        "development_source_file_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix()
            if path.is_relative_to(PROJECT_ROOT)
            else str(path): sha256_file(path)
            for path in source_paths
        },
        "locked_test_identity_manifest_file_sha256": sha256_file(identity_manifest_path),
        "locked_test_identity_projection_sha256": _identity_hash(identity_manifest),
        "development_identity_projection_sha256": _hash_json(identity_records),
        **image_fingerprint,
        "class_map_sha256": _hash_json(ontology),
        "resolved_protocol_sha256": artifact_hashes["resolved_protocol.yaml"],
        "effective_config_sha256": _hash_json(config),
        "feature_allowlist_sha256": _hash_json(feature_allowlist),
        "assignment_content_sha256": core.assignment_fingerprint(assignments),
        "artifact_file_sha256": artifact_hashes,
        "row_count": int(len(assignments)),
        "plot_count": int(assignments["plot_idx"].nunique()),
        "development_train_plot_count": int(
            assignments.loc[assignments["development_role"] == "train", "plot_idx"].nunique()
        ),
        "development_validation_plot_count": int(
            assignments.loc[
                assignments["development_role"] == "validation", "plot_idx"
            ].nunique()
        ),
        "assignment_validation": dict(validation),
        "code_file_sha256": code_hashes,
        "code_sha256": _hash_json(code_hashes),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty_diff_sha256": hashlib.sha256(
            (_git_value("diff", "--binary", "HEAD") or "").encode("utf-8")
        ).hexdigest(),
        "environment": environment,
        "environment_sha256": _hash_json(environment),
        "allowed_entry_points": list(config.get("commands", {}).get("m1_implemented", [])),
        "fail_closed_entry_points": list(config.get("commands", {}).get("m1_fail_closed", [])),
    }
    manifest["manifest_payload_sha256"] = _hash_json(manifest)
    return manifest


def freeze_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_yaml(args.config)
    _validate_freezable_config(config)
    _, active, identity_manifest_path, identity_manifest = _load_active_snapshot(config)
    frame, source_paths = _development_frame(config)
    assignments = _core().build_development_assignments(
        frame,
        test_identity_manifest=identity_manifest,
        **_assignment_kwargs(config),
    )
    validation = _core().validate_development_assignments(
        assignments,
        test_identity_manifest=identity_manifest,
        **_assignment_kwargs(config),
    )
    balance = _split_balance(assignments)

    paths = config["paths"]
    output_dir = _path(args.output_dir or paths["protocol_root"])
    targets = [output_dir / filename for filename in _PROTOCOL_FILES]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite immutable protocol artifacts: " + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = _exclusive_write_parquet(
        output_dir / "development_assignments.parquet", assignments
    )
    balance_path = output_dir / "split_balance.csv"
    exclusive_write_bytes(
        balance_path,
        balance.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    resolved_path = output_dir / "resolved_protocol.yaml"
    exclusive_write_bytes(resolved_path, _resolved_yaml_bytes(config))
    snapshot_ref = {
        "schema_version": "geo_helpfulness.locked_test_snapshot_ref.v1",
        "protocol_id": str(config["protocol_id"]),
        "snapshot_id": active.get("snapshot_id", _identity_hash(identity_manifest)),
        "identity_projection_sha256": _identity_hash(identity_manifest),
        "identity_manifest_file_sha256": sha256_file(identity_manifest_path),
        "active_snapshot_file_sha256": sha256_file(
            _path(config["paths"]["locked_test_registry_root"]) / "active_snapshot.json"
        ),
        "identity_manifest_relative_path": identity_manifest_path.relative_to(
            _path(config["paths"]["locked_test_registry_root"])
        ).as_posix(),
    }
    snapshot_ref_path = exclusive_write_json(
        output_dir / "locked_test_snapshot_ref.json", snapshot_ref
    )
    manifest = _protocol_manifest(
        config=config,
        source_paths=source_paths,
        assignments=assignments,
        validation=validation,
        output_dir=output_dir,
        identity_manifest_path=identity_manifest_path,
        identity_manifest=identity_manifest,
    )
    manifest_path = exclusive_write_json(output_dir / "protocol_manifest.json", manifest)
    return {
        "status": "frozen",
        "protocol_dir": str(output_dir),
        "assignments": str(assignments_path),
        "split_balance": str(balance_path),
        "resolved_protocol": str(resolved_path),
        "locked_test_snapshot_ref": str(snapshot_ref_path),
        "protocol_manifest": str(manifest_path),
        "assignment_content_sha256": manifest["assignment_content_sha256"],
        "validation": validation,
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.protocol_dir is not None:
        protocol_dir = _path(args.protocol_dir)
    else:
        config_hint = _load_yaml(args.config)
        protocol_dir = _path(config_hint["paths"]["protocol_root"])
    missing = [name for name in _PROTOCOL_FILES if not (protocol_dir / name).is_file()]
    if missing:
        raise ProtocolCommandError(f"Protocol directory is missing artifacts: {missing}")

    resolved_path = protocol_dir / "resolved_protocol.yaml"
    config = _load_yaml(resolved_path)
    _validate_freezable_config(config)
    manifest = _json(protocol_dir / "protocol_manifest.json", name="protocol manifest")
    expected_header = {
        "schema_version": "geo_helpfulness.protocol_manifest.v1",
        "artifact_role": "frozen_experimental_protocol",
        "protocol_id": str(config["protocol_id"]),
        "dataset_id": str(config["dataset_id"]),
        "protocol_status": "frozen",
    }
    observed_header = {key: manifest.get(key) for key in expected_header}
    if observed_header != expected_header:
        raise ProtocolCommandError(
            f"Protocol manifest header mismatch: {observed_header} != {expected_header}"
        )
    payload_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if payload_hash != _hash_json(payload):
        raise ProtocolCommandError("Protocol manifest payload hash mismatch")

    expected_artifacts = manifest.get("artifact_file_sha256")
    if not isinstance(expected_artifacts, Mapping):
        raise ProtocolCommandError("Protocol manifest has no artifact_file_sha256 mapping")
    expected_artifact_names = set(_PROTOCOL_FILES).difference({"protocol_manifest.json"})
    if set(expected_artifacts) != expected_artifact_names:
        raise ProtocolCommandError(
            "Protocol manifest artifact set does not match the frozen contract"
        )
    for filename in _PROTOCOL_FILES:
        if filename == "protocol_manifest.json":
            continue
        if expected_artifacts.get(filename) != sha256_file(protocol_dir / filename):
            raise ProtocolCommandError(f"Protocol artifact content hash mismatch: {filename}")

    registry_root, _, identity_manifest_path, identity_manifest = _load_active_snapshot(config)
    snapshot_ref = _json(
        protocol_dir / "locked_test_snapshot_ref.json", name="locked-test snapshot ref"
    )
    if snapshot_ref.get("identity_projection_sha256") != _identity_hash(identity_manifest):
        raise ProtocolCommandError("Protocol references a non-active test identity snapshot")
    active_snapshot_path = registry_root / "active_snapshot.json"
    if snapshot_ref.get("active_snapshot_file_sha256") != sha256_file(
        active_snapshot_path
    ):
        raise ProtocolCommandError("Active locked-test snapshot changed after protocol freeze")
    if snapshot_ref.get("identity_manifest_file_sha256") != sha256_file(
        identity_manifest_path
    ):
        raise ProtocolCommandError("Locked-test identity manifest changed after protocol freeze")
    if manifest.get("locked_test_identity_manifest_file_sha256") != sha256_file(
        identity_manifest_path
    ):
        raise ProtocolCommandError("Protocol manifest's test identity hash is stale")

    assignments = pd.read_parquet(protocol_dir / "development_assignments.parquet")
    validation = _core().validate_development_assignments(
        assignments,
        test_identity_manifest=identity_manifest,
        **_assignment_kwargs(config),
    )
    assignment_hash = _core().assignment_fingerprint(assignments)
    if manifest.get("assignment_content_sha256") != assignment_hash:
        raise ProtocolCommandError("Development assignment content hash mismatch")
    observed_counts = {
        "row_count": int(len(assignments)),
        "plot_count": int(assignments["plot_idx"].nunique()),
        "development_train_plot_count": int(
            assignments.loc[
                assignments["development_role"] == "train", "plot_idx"
            ].nunique()
        ),
        "development_validation_plot_count": int(
            assignments.loc[
                assignments["development_role"] == "validation", "plot_idx"
            ].nunique()
        ),
    }
    for key, value in observed_counts.items():
        if manifest.get(key) != value:
            raise ProtocolCommandError(
                f"Protocol manifest {key} mismatch: {manifest.get(key)} != {value}"
            )
    if manifest.get("assignment_validation") != validation:
        raise ProtocolCommandError("Protocol manifest assignment validation summary is stale")
    expected_balance = _split_balance(assignments).to_csv(index=False, lineterminator="\n")
    actual_balance = (protocol_dir / "split_balance.csv").read_text(encoding="utf-8")
    if actual_balance != expected_balance:
        raise ProtocolCommandError("split_balance.csv does not match assignments")

    source_hashes = manifest.get("development_source_file_sha256", {})
    if not isinstance(source_hashes, Mapping):
        raise ProtocolCommandError("Protocol manifest has no development source hashes")
    for raw_path in config["paths"]["development_source_tables"]:
        source = _path(raw_path)
        key = (
            source.relative_to(PROJECT_ROOT).as_posix()
            if source.is_relative_to(PROJECT_ROOT)
            else str(source)
        )
        if source_hashes.get(key) != sha256_file(source):
            raise ProtocolCommandError(f"Development source content changed: {source}")
    if manifest.get("class_map_sha256") != _hash_json(config["class_ontology"]):
        raise ProtocolCommandError("Frozen class ontology hash mismatch")
    if manifest.get("resolved_protocol_sha256") != sha256_file(resolved_path):
        raise ProtocolCommandError("Resolved protocol fingerprint mismatch")
    if manifest.get("effective_config_sha256") != _hash_json(config):
        raise ProtocolCommandError("Effective protocol configuration fingerprint mismatch")
    if manifest.get("feature_allowlist_sha256") != _hash_json(
        config.get("router", {}).get("feature_allowlist", {})
    ):
        raise ProtocolCommandError("Router feature allow-list fingerprint mismatch")
    identity_columns = ["row_uid", "file_lower", "plot_idx"]
    identity_records = (
        assignments.loc[:, identity_columns]
        .sort_values(identity_columns, kind="mergesort")
        .to_dict(orient="records")
    )
    if manifest.get("development_identity_projection_sha256") != _hash_json(
        identity_records
    ):
        raise ProtocolCommandError("Development identity projection fingerprint mismatch")
    image_fingerprint = _development_image_fingerprint(config, assignments)
    for key, value in image_fingerprint.items():
        if manifest.get(key) != value:
            raise ProtocolCommandError(
                f"Development image fingerprint mismatch for {key}"
            )
    environment = _environment()
    if manifest.get("environment_sha256") != _hash_json(manifest.get("environment")):
        raise ProtocolCommandError("Recorded environment fingerprint is internally inconsistent")
    if manifest.get("environment") != environment:
        raise ProtocolCommandError("Runtime environment changed after protocol freeze")

    regenerated_source, _ = _development_frame(config)
    regenerated = _core().build_development_assignments(
        regenerated_source,
        test_identity_manifest=identity_manifest,
        **_assignment_kwargs(config),
    )
    regenerated_hash = _core().assignment_fingerprint(regenerated)
    if regenerated_hash != assignment_hash:
        raise ProtocolCommandError(
            "Regenerated development assignments do not match the sealed artifact"
        )

    current_code_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in (
            PROJECT_ROOT / "multimodal" / "geo_helpfulness_protocol.py",
            PROJECT_ROOT / "multimodal" / "geo_helpfulness_locked_eval.py",
            Path(__file__).resolve(),
        )
    }
    if manifest.get("code_file_sha256") != current_code_hashes:
        raise ProtocolCommandError("Protocol implementation changed after freeze")
    if manifest.get("code_sha256") != _hash_json(current_code_hashes):
        raise ProtocolCommandError("Protocol implementation aggregate hash mismatch")
    return {
        "status": "valid",
        "protocol_dir": str(protocol_dir),
        "protocol_id": str(config["protocol_id"]),
        "assignment_content_sha256": assignment_hash,
        "artifact_file_sha256": dict(expected_artifacts),
        "validation": validation,
    }


def _future_command(args: argparse.Namespace) -> dict[str, Any]:
    raise ProtocolCommandError(f"{args.command} {M1_NOT_IMPLEMENTED}")


def locked_predict(args: argparse.Namespace) -> dict[str, Any]:
    if not args.synthetic_fixture:
        raise ProtocolCommandError(
            f"locked-predict {M1_NOT_IMPLEMENTED}; M1 permits only --synthetic-fixture"
        )
    missing = [name for name in ("bundle", "identity_manifest", "features", "output") if getattr(args, name) is None]
    if missing:
        raise ProtocolCommandError(f"locked-predict is missing required fixture arguments: {missing}")
    return locked_predict_synthetic(
        bundle_path=args.bundle,
        identity_manifest_path=args.identity_manifest,
        features_path=args.features,
        output_path=args.output,
    )


def locked_score(args: argparse.Namespace) -> dict[str, Any]:
    if not args.synthetic_fixture:
        raise ProtocolCommandError(
            f"locked-score {M1_NOT_IMPLEMENTED}; M1 permits only --synthetic-fixture"
        )
    missing = [name for name in ("predictions", "labels", "output") if getattr(args, name) is None]
    if missing:
        raise ProtocolCommandError(f"locked-score is missing required fixture arguments: {missing}")
    return locked_score_synthetic(
        predictions_path=args.predictions,
        labels_path=args.labels,
        output_path=args.output,
        event_registry=args.event_registry,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and validate the reliability-aware geo-helpfulness protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_identity = subparsers.add_parser(
        "freeze-test-identity",
        help="Seal the one global label-blind cleaned-test identity snapshot.",
    )
    freeze_identity.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze_identity.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Command-local test table; never stored in shared protocol config.",
    )
    freeze_identity.add_argument("--registry-root", type=Path)
    freeze_identity.add_argument("--dataset-id")
    freeze_identity.set_defaults(handler=freeze_test_identity)

    freeze = subparsers.add_parser(
        "freeze-protocol", help="Materialize immutable M1 protocol artifacts."
    )
    freeze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze.add_argument("--output-dir", type=Path)
    freeze.set_defaults(handler=freeze_protocol)

    validate = subparsers.add_parser(
        "validate-protocol", help="Validate all frozen M1 protocol fingerprints."
    )
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.add_argument("--protocol-dir", type=Path)
    validate.set_defaults(handler=validate_protocol)

    for name in (
        "build-train-oof",
        "fit-router-candidates",
        "score-router-candidates",
        "fit-final-experts",
    ):
        future = subparsers.add_parser(name, help=f"Fail-closed M1 shell for {name}.")
        future.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        future.set_defaults(handler=_future_command)

    predict = subparsers.add_parser(
        "locked-predict",
        help="Exercise label-blind locked prediction on an M1 synthetic fixture.",
    )
    predict.add_argument("--synthetic-fixture", action="store_true")
    predict.add_argument("--bundle", type=Path)
    predict.add_argument("--identity-manifest", "--identity", dest="identity_manifest", type=Path)
    predict.add_argument("--features", type=Path)
    predict.add_argument("--output", type=Path)
    predict.set_defaults(handler=locked_predict)

    score = subparsers.add_parser(
        "locked-score",
        help="Exercise fit-incapable locked scoring on an M1 synthetic fixture.",
    )
    score.add_argument("--synthetic-fixture", action="store_true")
    score.add_argument("--predictions", type=Path)
    score.add_argument("--labels", type=Path)
    score.add_argument("--output", type=Path)
    score.add_argument("--event-registry", type=Path)
    score.set_defaults(handler=locked_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (FileExistsError, KeyError, OSError, ProtocolCommandError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

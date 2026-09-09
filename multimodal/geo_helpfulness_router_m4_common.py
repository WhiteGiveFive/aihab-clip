"""Fit-incapable artifact utilities and fixed paths for final M4 delivery."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1"
)
COMPARATOR_DIR = Path("router/m4_comparator_inputs_v1")
RESULT_DIR = Path("router/m4_feasibility_v1")
CONVENTIONS_PATH = "configs/multimodal_geo_router_m4_evaluation_v1.json"
CONVENTIONS_SHA256 = "d3af535356a2cab571535d7ad4e59917d7df3026a1e4492059bf0d0e7d82c469"
CODE_FILES = (
    "multimodal/geo_helpfulness_router_m4_common.py",
    "multimodal/geo_helpfulness_router_m4_inputs.py",
    "multimodal/geo_helpfulness_router_m4_analysis.py",
    "multimodal/geo_helpfulness_router_metrics.py",
    "multimodal/geo_helpfulness_router_selection.py",
    "multimodal/geo_helpfulness_router_uncertainty.py",
    "tools/run_multimodal_geo_helpfulness_m4.py",
)
VARIANTS = ("primary_full727", "ablation_no_pairs79", "ablation_numeric25")
SEEDS = (1, 2, 3, 4)
POLICIES = (
    *VARIANTS,
    "never_intervene",
    "always_geo_on_geo_raw_disagreement",
    "higher_calibrated_geo_confidence_on_geo_raw_disagreement",
    "image_geo_jsd",
    "geo_minus_raw_margin",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected nonsymlink regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_json(path: Path) -> dict:
    file_hash(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object,
        parse_constant=_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def check_self_hash(value: dict, key="manifest_sha256") -> None:
    if value.get(key) != canonical_hash({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"invalid {key}")


def check_snapshot(snapshot: dict[str, str]) -> None:
    for filename, expected in snapshot.items():
        if file_hash(Path(filename)) != expected:
            raise ValueError(f"frozen parent or code changed: {filename}")


def code_snapshot(repo_root: Path = PROJECT_ROOT) -> dict[str, str]:
    return {
        str(repo_root / name): file_hash(repo_root / name)
        for name in (*CODE_FILES, CONVENTIONS_PATH)
    }


def load_conventions(repo_root: Path = PROJECT_ROOT) -> dict:
    value = read_json(repo_root / CONVENTIONS_PATH)
    if canonical_hash(value) != CONVENTIONS_SHA256:
        raise ValueError(
            "M4 reporting conventions differ from the predeclared v1 contract"
        )
    return value


def write_bytes(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        if file_hash(path) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    write_bytes(path, canonical_bytes(value) + b"\n")


def table_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table.replace_schema_metadata(None), sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def compare_tables(observed: pa.Table, expected: pa.Table) -> None:
    if observed.schema != expected.schema or observed.num_rows != expected.num_rows:
        raise ValueError("reconstructed table schema/count mismatch")
    for name in observed.column_names:
        field = observed.schema.field(name)
        if not observed[name].is_null().equals(expected[name].is_null()):
            raise ValueError(f"reconstructed table null mask mismatch: {name}")
        if pa.types.is_floating(field.type):
            for column in (observed[name], expected[name]):
                valid = np.asarray(column.drop_null().to_numpy())
                if not np.isfinite(valid).all():
                    raise ValueError(
                        f"nonnull numerical table value is nonfinite: {name}"
                    )
            same = np.allclose(
                observed[name].to_numpy(),
                expected[name].to_numpy(),
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            )
        elif pa.types.is_fixed_size_list(field.type) and pa.types.is_floating(
            field.type.value_type
        ):
            same = np.allclose(
                np.asarray(observed[name].to_pylist()),
                np.asarray(expected[name].to_pylist()),
                atol=1e-12,
                rtol=1e-12,
            )
        else:
            same = observed[name].equals(expected[name])
        if not same:
            raise ValueError(f"reconstructed table values mismatch: {name}")


@contextmanager
def exclusive_lock(root: Path):
    root.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        root.parent / f".{root.name}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another M4 process owns this workflow lock") from error
        yield
    finally:
        os.close(descriptor)


def validate_manifest(root: Path, *, schema_version: str) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("bundle must be a nonsymlink directory")
    manifest = read_json(root / "manifest.json")
    check_self_hash(manifest)
    if (
        manifest.get("schema_version") != schema_version
        or manifest.get("protocol_id") != "protocol_v1"
    ):
        raise ValueError("bundle manifest schema/protocol mismatch")
    files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if files != set(manifest["artifacts"]) | {"manifest.json"}:
        raise ValueError("bundle file allowlist differs from manifest")
    for name, record in manifest["artifacts"].items():
        if file_hash(root / name) != record["file_sha256"]:
            raise ValueError(f"bundle artifact changed: {name}")
    check_snapshot(manifest["parent_file_sha256"])
    return manifest


def publish_manifest(
    root: Path, *, schema_version: str, snapshot: dict[str, str], extra: dict
) -> dict:
    check_snapshot(snapshot)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": schema_version,
        "protocol_id": "protocol_v1",
        "artifacts": {
            str(path.relative_to(root)): {"file_sha256": file_hash(path)}
            for path in files
        },
        "parent_file_sha256": dict(sorted(snapshot.items())),
        **extra,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(root / "manifest.json", manifest)
    return validate_manifest(root, schema_version=schema_version)

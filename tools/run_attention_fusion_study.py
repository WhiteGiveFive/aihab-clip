#!/usr/bin/env python3
"""Run or inspect the entire frozen M3 matrix, stopping on the first failure."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_attention_fusion_protocol import DEFAULT_CONFIG, content_hash, file_hash  # noqa: E402
from tools.run_multimodal_attention_fusion import execute_fit, read_frozen_plan  # noqa: E402

SOURCE_FILES = (
    "tools/run_attention_fusion_study.py",
    "tools/run_multimodal_attention_fusion.py",
    "tools/prepare_attention_fusion_protocol.py",
    "tools/audit_attention_fusion_inputs.py",
    "multimodal/attention_fusion_training.py",
    "multimodal/attention_fusion.py",
    "multimodal/attention_fusion_inputs.py",
    "multimodal/geo_helpfulness_protocol.py",
    "multimodal/models.py",
)
REQUIRED_BUNDLE_FILES = {
    "final.pt", "resume.pt", "run_spec.json", "config.json", "run.json",
    "metrics.json", "history.json", "geo_preprocessing.json",
    "train_predictions.npz", "validation_predictions.npz",
}


class StudyError(ValueError):
    """The scientific workload or its immutable evidence changed."""


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise StudyError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def event(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps({"time": now(), **record}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _refresh_counts(state: dict) -> None:
    state["counts"] = {}
    for row in state["fits"]:
        state["counts"][row["status"]] = state["counts"].get(row["status"], 0) + 1


@contextmanager
def writer_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "writer.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StudyError("Another study orchestrator holds the writer lock") from exc
        stream.seek(0)
        stream.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started_at": now()}, stream)
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _study_root(config: dict, repo: Path) -> Path:
    root = (repo / config["paths"]["artifact_root"]).resolve()
    require(root.is_relative_to(repo.resolve()), "Study root escapes repository")
    return root


def _fit_directory(fit: dict, config: dict, repo: Path) -> Path:
    directory = (repo / fit["output_dir"]).resolve()
    require(directory.is_relative_to(_study_root(config, repo) / "development"),
            "Matrix fit must stay inside the scientific development namespace")
    return directory


def source_hashes(repo: Path) -> dict[str, str]:
    return {relative: file_hash(repo / relative) for relative in SOURCE_FILES}


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def preflight(config: dict, matrix: dict, protocol: dict, repo: Path,
              device: str, cpu_threads: int) -> dict:
    """Verify every input once, including large encoder references, before fits."""
    import torch
    from multimodal.attention_fusion_inputs import load_input_provenance, load_prepared_inputs

    require(cpu_threads > 0, "CPU thread count must be positive")
    workspace = config["training"]["cublas_workspace_config"]
    if torch.cuda.is_initialized():
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == workspace,
                "CUDA was initialized before the frozen CUBLAS workspace configuration")
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    selected = torch.device(device)
    require(selected.type == "cuda", "This M3 orchestrator requires the declared CUDA execution regime")
    require(torch.cuda.is_available(), "CUDA is unavailable; no scientific fit was started")
    torch.set_num_threads(cpu_threads)
    probe = torch.ones(1, device=selected)
    require(float(probe.cpu().item()) == 1.0, "CUDA allocation/retrieval probe failed")
    require(len(matrix["development_fits"]) == config["budget"]["development_fits"],
            "Expanded workload size differs from the protocol")
    files: dict[str, str] = {}
    seeds = sorted({fit["seed"] for fit in matrix["development_fits"]})
    for seed in seeds:
        for role in ("train", "validation"):
            load_prepared_inputs(config, seed, role, repo_root=repo)
        provenance = load_input_provenance(config, seed, repo_root=repo)
        encoder = provenance["encoder"]
        encoder_path = Path(encoder["path"])
        require(file_hash(encoder_path) == encoder["sha256"], f"Seed {seed} encoder bytes changed")
        files[str(encoder_path.resolve())] = encoder["sha256"]
        manifest_path = Path(provenance["manifest_path"])
        files[str(manifest_path.resolve())] = file_hash(manifest_path)
        manifest = read_json(manifest_path)
        for name, digest in manifest["file_sha256"].items():
            path = (manifest_path.parent / name).resolve()
            require(path.is_relative_to(manifest_path.parent.resolve()), "Input manifest path traversal")
            require(file_hash(path) == digest, f"Seed {seed} input payload changed: {name}")
            files[str(path)] = digest
        print(json.dumps({"preflight_seed": seed, "status": "authenticated"}), flush=True)
    # The native geo/assignment files also feed the geo-only path directly.
    for name in [config["paths"]["assignments"], *config["paths"]["development_geo_tables"]]:
        path = (repo / name).resolve()
        require(path.is_relative_to(repo.resolve()), "Development input escapes repository")
        files[str(path)] = file_hash(path)
    protocol_root = _study_root(config, repo) / "protocol"
    for name in ("protocol_manifest.json", "resolved_protocol.yaml", "candidate_matrix.json", "input_audit.json"):
        path = protocol_root / name
        files[str(path)] = file_hash(path)
    return {"input_file_sha256": files,
            "execution": {"device": str(selected), "device_name": torch.cuda.get_device_name(selected),
                          "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                          "cpu_threads": cpu_threads},
            "protocol_payload_sha256": protocol["manifest_payload_sha256"]}


def authenticate_complete(directory: Path, fit: dict, config: dict, sources: dict) -> dict:
    """Hash every bundle member before an existing result can count as complete."""
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("status") == "complete", "Bundle is not complete")
    require(content_hash({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"})
            == manifest.get("manifest_payload_sha256"), "Completed bundle manifest checksum failed")
    require(REQUIRED_BUNDLE_FILES <= set(manifest["files"]), "Completed bundle lacks required artifacts")
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()) and file_hash(path) == digest,
                f"Completed bundle payload changed: {name}")
    spec = read_json(directory / "run_spec.json")
    require(content_hash(spec) == manifest["run_identity_sha256"], "Completed run specification checksum failed")
    expected = {"candidate_id": fit["candidate_id"], "geo_mode": fit["geo_preprocessing"],
                "learning_rate": fit["learning_rate"], "seed": fit["seed"],
                "verification": False, "fit_role": "train", "epochs": config["training"]["epochs"],
                "config_sha256": content_hash(config)}
    require(all(spec.get(key) == value for key, value in expected.items()),
            "Completed bundle belongs to another scientific fit")
    expected_source = {Path(name).name: digest for name, digest in sources.items()
                       if name in {"multimodal/attention_fusion_training.py", "multimodal/attention_fusion.py", "multimodal/models.py"}}
    require(spec["source_hashes"] == expected_source, "Completed model/training source differs from this execution")
    require(manifest["completed_epoch"] == config["training"]["checkpoint_epoch"], "Completed bundle has wrong epoch")
    return {"manifest_sha256": file_hash(directory / "manifest.json"),
            "run_identity_sha256": manifest["run_identity_sha256"],
            "completed_epoch": manifest["completed_epoch"]}


def matrix_status(config: dict, matrix: dict, repo: Path = REPO_ROOT) -> dict:
    """Read live progress; authenticate completed bundles without evaluating data."""
    sources = source_hashes(repo)
    fits, counts = [], {}
    for fit in matrix["development_fits"]:
        directory = _fit_directory(fit, config, repo)
        row = {"fit_id": fit["fit_id"], "output_dir": fit["output_dir"],
               "status": "planned", "completed_epoch": 0}
        try:
            if (directory / "manifest.json").is_file():
                row.update(authenticate_complete(directory, fit, config, sources), status="complete")
            elif (directory / "run.json").is_file():
                saved = read_json(directory / "run.json")
                row.update(status=saved.get("status", "unknown"), completed_epoch=saved.get("completed_epoch", 0))
            elif directory.exists() and any(directory.iterdir()):
                row["status"] = "partial_without_run_metadata"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status="invalid_artifacts", error=f"{type(exc).__name__}: {exc}")
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        fits.append(row)
    return {"study_id": config["study_id"], "updated_at": now(),
            "planned_fits": len(fits), "counts": counts, "fits": fits,
            "complete_means": "all bundle payload hashes and declared scientific identity authenticated"}


def run_study(config_path: Path = DEFAULT_CONFIG, *, device: str = "cuda",
              cpu_threads: int = 1, repo: Path = REPO_ROOT) -> dict:
    config, matrix, protocol = read_frozen_plan(Path(config_path), repo)
    root = _study_root(config, repo) / "orchestration/m3"
    with writer_lock(root):
        sources = source_hashes(repo)
        checked = preflight(config, matrix, protocol, repo, device, cpu_threads)
        identity = {"study_id": config["study_id"], "config_sha256": content_hash(config),
                    "matrix_sha256": content_hash(matrix), "source_sha256": sources,
                    "fit_ids_in_order": [fit["fit_id"] for fit in matrix["development_fits"]], **checked}
        identity_path = root / "identity.json"
        if identity_path.exists():
            require(read_json(identity_path) == identity, "Orchestration identity changed; restore the declared source/input/runtime before resuming")
        else:
            write_json(identity_path, identity)
        stamps = {path: _stamp(Path(path)) for path in checked["input_file_sha256"]}

        def guard() -> None:
            require(source_hashes(repo) == sources, "Scientific execution source changed while the matrix was running")
            for path, stamp in stamps.items():
                require(_stamp(Path(path)) == stamp, f"A frozen input/protocol artifact changed while running: {path}")

        state = matrix_status(config, matrix, repo)
        state.update(status="running", active_fit=None, execution=checked["execution"])
        write_json(root / "status.json", state)
        event(root / "events.jsonl", {"event": "orchestration_started", "pid": os.getpid(),
                                      "identity_sha256": content_hash(identity)})
        try:
            for index, fit in enumerate(matrix["development_fits"]):
                guard()
                directory = _fit_directory(fit, config, repo)
                if (directory / "manifest.json").exists():
                    authenticated = authenticate_complete(directory, fit, config, sources)
                    # The existing trainer recomputes the full run identity,
                    # including arrays, provenance and runtime, before its
                    # immutable completed-bundle reuse branch returns.
                    reused = execute_fit(config, fit, device=device, cpu_threads=cpu_threads, resume=True)
                    require(reused.get("status") == "reused_complete", "Completed fit was not reused immutably")
                    guard()
                    state["fits"][index].update(authenticated, status="complete")
                    continue
                resume = directory.exists() and any(directory.iterdir())
                state.update(active_fit=fit["fit_id"], updated_at=now())
                state["fits"][index]["status"] = "running"
                _refresh_counts(state)
                write_json(root / "status.json", state)
                event(root / "events.jsonl", {"event": "fit_started", "fit_id": fit["fit_id"], "resume": resume})
                print(json.dumps({"fit": index + 1, "of": len(matrix["development_fits"]),
                                  "fit_id": fit["fit_id"], "resume": resume}), flush=True)
                result = execute_fit(config, fit, device=device, cpu_threads=cpu_threads, resume=resume)
                require(result.get("status") in {"complete", "reused_complete"}, "Fit did not complete its frozen epoch budget")
                guard()
                authenticated = authenticate_complete(directory, fit, config, sources)
                state["fits"][index].update(authenticated, status="complete")
                _refresh_counts(state)
                state.update(active_fit=None, updated_at=now())
                write_json(root / "status.json", state)
                event(root / "events.jsonl", {"event": "fit_completed", "fit_id": fit["fit_id"], **authenticated})
                print(json.dumps({"completed": fit["fit_id"], "complete_fits": state["counts"].get("complete", 0)}), flush=True)
            guard()
            state = matrix_status(config, matrix, repo)
            require(state["counts"] == {"complete": len(matrix["development_fits"])}, "Matrix still contains incomplete or invalid fits")
            state.update(status="complete", active_fit=None, execution=checked["execution"])
            write_json(root / "status.json", state)
            event(root / "events.jsonl", {"event": "orchestration_completed", "fits": len(matrix["development_fits"])})
            return state
        except BaseException as exc:
            state.update(status="failed_or_interrupted", updated_at=now(), error=f"{type(exc).__name__}: {exc}")
            for row in state["fits"]:
                if row["fit_id"] == state.get("active_fit"):
                    row.update(status="failed_or_interrupted", error=state["error"])
            _refresh_counts(state)
            write_json(root / "status.json", state)
            event(root / "events.jsonl", {"event": "orchestration_failed_or_interrupted", "active_fit": state.get("active_fit"),
                                          "error": state["error"]})
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Execute or resume all declared M3 fits; no scientific subset selection")
    run.add_argument("--device", default="cuda")
    run.add_argument("--cpu-threads", type=int, default=1)
    sub.add_parser("status", help="Read live run progress and authenticate completed bundles")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_study(args.config, device=args.device, cpu_threads=args.cpu_threads)
    else:
        config, matrix, _ = read_frozen_plan(args.config)
        result = matrix_status(config, matrix)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

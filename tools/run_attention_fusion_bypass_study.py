#!/usr/bin/env python3
"""Declare, run or inspect the separately versioned 72-fit CLS-bypass study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multimodal.attention_fusion_bypass_protocol import (  # noqa: E402
    DEFAULT_CONFIG, PARENT_CONFIG, build_matrix, content_hash, declare_protocol,
    file_hash, load_fit_inputs, path_in_repo, read_frozen_plan, read_parent_plan,
    source_hashes, verify_parent_bindings,
)
from tools.run_attention_fusion_study import (  # noqa: E402
    REQUIRED_BUNDLE_FILES, StudyError, _fit_directory, _refresh_counts, _stamp,
    _study_root, event, now, preflight as parent_preflight, read_json, require,
    write_json, writer_lock,
)

MODEL_SOURCE_FILES = {
    "multimodal/attention_fusion_training.py", "multimodal/attention_fusion.py",
    "multimodal/models.py", "multimodal/attention_fusion_bypass.py",
    "multimodal/attention_fusion_bypass_training.py",
}


def execute_fit(config: dict, fit: dict, *, device: str = "cuda", cpu_threads: int = 1,
                resume: bool = False, repo: Path = REPO_ROOT) -> dict:
    from multimodal.attention_fusion_bypass_training import train_run

    require(fit in build_matrix(config)["development_fits"], "Fit differs from the frozen 72-fit bypass matrix")
    train, validation, provenance = load_fit_inputs(config, fit["seed"], repo)
    return train_run(config, fit["candidate_id"], fit["geo_preprocessing"],
                     fit["learning_rate"], fit["seed"], train, validation,
                     _fit_directory(fit, config, repo), device=device,
                     input_provenance=provenance, cpu_num_threads=cpu_threads,
                     fit_role="train", verification=False, resume=resume)


def preflight(config: dict, matrix: dict, protocol: dict, repo: Path,
              device: str, cpu_threads: int) -> dict:
    """Reuse original cache verification and bind all extension evidence."""
    parent, parent_matrix, parent_protocol = read_parent_plan(PARENT_CONFIG, repo)
    require(matrix == build_matrix(config), "Bypass workload changed")
    evidence = verify_parent_bindings(config, repo)
    checked = parent_preflight(parent, parent_matrix, parent_protocol, repo, device, cpu_threads)
    for relative, digest in evidence["file_sha256"].items():
        checked["input_file_sha256"][str(path_in_repo(repo, relative))] = digest
    directory = _study_root(config, repo) / "protocol"
    for name in ["protocol_manifest.json", *protocol["files"]]:
        path = directory / name
        checked["input_file_sha256"][str(path)] = file_hash(path)
    checked.update(protocol_payload_sha256=protocol["manifest_payload_sha256"],
                   original_protocol_payload_sha256=parent_protocol["manifest_payload_sha256"],
                   contrast_declaration_sha256=file_hash(directory / "contrasts.json"))
    return checked


def authenticate_complete(directory: Path, fit: dict, config: dict, sources: dict) -> dict:
    """Authenticate a full epoch-50 bundle from the isolated bypass engine."""
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
    expected = {"study_id": config["study_id"], "candidate_id": fit["candidate_id"],
                "geo_mode": fit["geo_preprocessing"], "learning_rate": fit["learning_rate"],
                "seed": fit["seed"], "verification": False, "fit_role": "train",
                "epochs": config["training"]["epochs"], "config_sha256": content_hash(config)}
    require(all(spec.get(key) == value for key, value in expected.items()),
            "Completed bundle belongs to another scientific fit")
    expected_source = {Path(name).name: digest for name, digest in sources.items() if name in MODEL_SOURCE_FILES}
    require(spec["source_hashes"] == expected_source, "Completed model/training source differs from this execution")
    require(read_json(directory / "config.json") == config, "Completed bundle resolved configuration differs")
    require(manifest["completed_epoch"] == config["training"]["checkpoint_epoch"]
            and manifest["verification"] is False, "Completed bundle has wrong epoch or verification role")
    require(manifest["class_map"] == config["data"]["dense_to_canonical_l3"]
            and manifest["boundary"] == {"test_data_loaded": False, "encoder_executed": False},
            "Completed bundle class map or data boundary differs")
    candidate = next(c for c in config["candidates"] if c["id"] == fit["candidate_id"])
    require(manifest["trainable_parameters"] == candidate["expected_trainable_parameters"],
            "Completed bundle parameter count differs")
    provenance = spec["input_provenance"]
    require(provenance.get("protocol_config_sha256") == content_hash(config)
            and provenance.get("source_protocol_config_sha256") == config["source_study"]["config_sha256"]
            and provenance.get("train_role") == "train" and provenance.get("validation_role") == "validation",
            "Completed bundle input configuration or fitting roles differ")
    return {"manifest_sha256": file_hash(directory / "manifest.json"),
            "run_identity_sha256": manifest["run_identity_sha256"], "completed_epoch": manifest["completed_epoch"]}


def matrix_status(config: dict, matrix: dict, repo: Path = REPO_ROOT) -> dict:
    """Authenticate saved bundles without loading feature or prediction arrays."""
    sources = source_hashes(repo)
    rows, counts = [], {}
    for fit in matrix["development_fits"]:
        directory = _fit_directory(fit, config, repo)
        row = {"fit_id": fit["fit_id"], "output_dir": fit["output_dir"], "status": "planned", "completed_epoch": 0}
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
        rows.append(row)
    return {"study_id": config["study_id"], "updated_at": now(), "planned_fits": len(rows),
            "counts": counts, "fits": rows, "original_comparator_fits": 228,
            "complete_means": "all bundle payload hashes and declared scientific identity authenticated"}


def run_study(config_path: Path = DEFAULT_CONFIG, *, device: str = "cuda",
              cpu_threads: int = 1, repo: Path = REPO_ROOT) -> dict:
    repo = Path(repo).resolve()
    config, matrix, protocol = read_frozen_plan(config_path, repo)
    root = _study_root(config, repo) / "orchestration/m3_bypass"
    with writer_lock(root):
        sources = source_hashes(repo)
        checked = preflight(config, matrix, protocol, repo, device, cpu_threads)
        identity = {"study_id": config["study_id"], "config_sha256": content_hash(config),
                    "matrix_sha256": content_hash(matrix), "source_sha256": sources,
                    "fit_ids_in_order": [fit["fit_id"] for fit in matrix["development_fits"]], **checked}
        identity_path = root / "identity.json"
        if identity_path.exists():
            require(read_json(identity_path) == identity,
                    "Orchestration identity changed; restore declared source/input/runtime before resuming")
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
                state.update(active_fit=fit["fit_id"], updated_at=now())
                if (directory / "manifest.json").exists():
                    authenticated = authenticate_complete(directory, fit, config, sources)
                    # Trainer rechecks exact array, runtime and provenance identity.
                    reused = execute_fit(config, fit, device=device, cpu_threads=cpu_threads, resume=True, repo=repo)
                    require(reused.get("status") == "reused_complete", "Completed fit was not reused immutably")
                    guard()
                    state["fits"][index].update(authenticated, status="complete")
                    continue
                resume = directory.exists() and any(directory.iterdir())
                state["fits"][index]["status"] = "running"
                _refresh_counts(state)
                write_json(root / "status.json", state)
                event(root / "events.jsonl", {"event": "fit_started", "fit_id": fit["fit_id"], "resume": resume})
                print(json.dumps({"fit": index + 1, "of": len(matrix["development_fits"]),
                                  "fit_id": fit["fit_id"], "resume": resume}), flush=True)
                result = execute_fit(config, fit, device=device, cpu_threads=cpu_threads, resume=resume, repo=repo)
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
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("declare", help="Authenticate reused M3 evidence and freeze the extension before fitting")
    run = commands.add_parser("run", help="Execute or resume all 72 declared fits")
    run.add_argument("--device", default="cuda")
    run.add_argument("--cpu-threads", type=int, default=1)
    commands.add_parser("status", help="Read progress and authenticate completed bundles")
    args = parser.parse_args(argv)
    if args.command == "declare":
        result = declare_protocol(args.config)
    elif args.command == "run":
        result = run_study(args.config, device=args.device, cpu_threads=args.cpu_threads)
    else:
        config, matrix, _ = read_frozen_plan(args.config)
        result = matrix_status(config, matrix)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

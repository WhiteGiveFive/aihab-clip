#!/usr/bin/env python3
"""Run a frozen attention-fusion fit or isolated training-only verification."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_attention_fusion_protocol import (  # noqa: E402
    DEFAULT_CONFIG, ProtocolError, build_matrix, content_hash, file_hash,
    validate_config,
)


def read_frozen_plan(config_path: Path, repo: Path = REPO_ROOT) -> tuple[dict, dict, dict]:
    """Authenticate the saved recipe/matrix without loading any data payload."""
    path = config_path if config_path.is_absolute() else repo / config_path
    config = yaml.safe_load(path.read_text())
    validate_config(config)
    root = (repo / config["paths"]["artifact_root"]).resolve()
    if not root.is_relative_to(repo.resolve()):
        raise ProtocolError("Study output must remain within the repository")
    protocol = root / "protocol"
    manifest = json.loads((protocol / "protocol_manifest.json").read_text())
    payload = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    if content_hash(payload) != manifest.get("manifest_payload_sha256"):
        raise ProtocolError("Frozen protocol manifest checksum failed")
    if manifest.get("effective_config_sha256") != content_hash(config):
        raise ProtocolError("Configuration differs from its frozen protocol; record an amendment or new study")
    if manifest.get("study_id") != config["study_id"]:
        raise ProtocolError("Study identity differs from its frozen protocol")
    for key, name in (("resolved_protocol", "resolved_protocol.yaml"),
                      ("candidate_matrix", "candidate_matrix.json")):
        if file_hash(protocol / name) != manifest["outputs"][key]["sha256"]:
            raise ProtocolError(f"Frozen {name} checksum failed")
    if file_hash(protocol / "input_audit.json") != manifest["input_audit"]["sha256"]:
        raise ProtocolError("Input audit differs from the frozen protocol")
    if yaml.safe_load((protocol / "resolved_protocol.yaml").read_text()) != config:
        raise ProtocolError("Resolved and requested configurations differ")
    matrix = json.loads((protocol / "candidate_matrix.json").read_text())
    if matrix != build_matrix(config):
        raise ProtocolError("Saved matrix does not describe the requested recipe")
    return config, matrix, manifest


def select_fit(matrix: dict, fit_id: str) -> dict:
    fits = [item for item in matrix["development_fits"] if item["fit_id"] == fit_id]
    if len(fits) != 1:
        raise ProtocolError(f"Unknown or ambiguous declared fit: {fit_id}")
    return fits[0]


def plan_status(config: dict, matrix: dict, repo: Path = REPO_ROOT) -> dict:
    """Report all declared fits, including interrupted/failed/missing ones."""
    rows = []
    for fit in matrix["development_fits"]:
        directory = repo / fit["output_dir"]
        state_path = directory / "run.json"
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get("status") == "complete":
                if not (directory / "manifest.json").exists():
                    state["status"] = "incomplete_commit"
                else:
                    manifest = json.loads((directory / "manifest.json").read_text())
                    spec = json.loads((directory / "run_spec.json").read_text())
                    payload = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
                    if (content_hash(payload) != manifest.get("manifest_payload_sha256")
                            or content_hash(spec) != manifest.get("run_identity_sha256")
                            or spec["config_sha256"] != content_hash(config)
                            or spec["verification"] is not False
                            or spec["candidate_id"] != fit["candidate_id"]
                            or spec["geo_mode"] != fit["geo_preprocessing"]
                            or spec["learning_rate"] != fit["learning_rate"]
                            or spec["seed"] != fit["seed"]):
                        raise ProtocolError("Completed run metadata is inconsistent")
        except (ValueError, OSError, KeyError, TypeError):
            state = {"status": "invalid_run_metadata"}
        rows.append({"fit_id": fit["fit_id"], "output_dir": fit["output_dir"],
                     "status": state.get("status", "planned"),
                     "completed_epoch": state.get("completed_epoch", 0)})
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"study_id": config["study_id"], "counts": matrix["counts"],
            "development_status_counts": counts, "fits": rows,
            "status_verification": "metadata_only; bundle payload hashes must be checked before selection",
            "verification_runs_count_as_development": False}


def execute_fit(config: dict, fit: dict, *, device: str = "cpu",
                cpu_threads: int = 1, resume: bool = False,
                stop_after_epoch: int | None = None) -> dict:
    from multimodal.attention_fusion_inputs import (
        load_prepared_inputs, load_input_provenance, load_geo_only_inputs,
    )
    from multimodal.attention_fusion_training import train_run

    seed = fit["seed"]
    if fit["candidate_id"] == "geo_only":
        train, validation, provenance = load_geo_only_inputs(config, seed)
    else:
        train = load_prepared_inputs(config, seed, "train")
        validation = load_prepared_inputs(config, seed, "validation")
        provenance = load_input_provenance(config, seed)
    provenance.update({"protocol_config_sha256": content_hash(config),
                       "train_role": "train", "validation_role": "validation"})
    return train_run(config, fit["candidate_id"], fit["geo_preprocessing"],
                     fit["learning_rate"], seed, train, validation,
                     REPO_ROOT / fit["output_dir"], device=device,
                     input_provenance=provenance, cpu_num_threads=cpu_threads,
                     resume=resume, stop_after_epoch=stop_after_epoch)


def training_verification_data(config: dict, seed: int, samples: int) -> tuple[dict, dict]:
    """Use a verified train-cache prefix; a bounded extraction is also supported."""
    from multimodal.attention_fusion_inputs import (
        load_prepared_inputs, load_verification_inputs, load_input_provenance,
    )

    prepared = REPO_ROOT / config["paths"]["prepared_cache_template"].format(seed=seed, role="train")
    bounded = not (prepared.parent / "manifest.json").is_file()
    arrays = (load_verification_inputs(config, seed) if bounded
              else load_prepared_inputs(config, seed, "train"))
    if samples < 2 or samples > len(arrays["label"]):
        raise ProtocolError("Requested more rows than the verified training cache contains")
    provenance = load_input_provenance(config, seed, verification=bounded)
    provenance.update(source_role="train", verification_training_prefix_rows=samples)
    return {key: value[:samples].copy() for key, value in arrays.items()}, provenance


def run_smoke(config: dict, matrix: dict, *, name: str, seed: int = 1,
              samples: int = 32, epochs: int = 3, learning_rate: float = 0.001,
              device: str = "cpu", cpu_threads: int = 1) -> dict:
    """Exercise every declared arm using only a saved training-row subset."""
    import numpy as np
    from multimodal.attention_fusion_training import load_inference_bundle, train_run

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ProtocolError("Verification name must be a simple directory name")
    if samples < 2 or epochs < 1 or seed not in config["data"]["training_seeds"]:
        raise ProtocolError("Invalid verification seed, row count or epochs")
    if epochs > config["training"]["epochs"] or learning_rate not in config["training"]["learning_rates"]:
        raise ProtocolError("Verification epochs/rate exceed the declared recipe")
    root = REPO_ROOT / config["paths"]["artifact_root"] / "verification" / name
    if root.exists():
        raise ProtocolError("Verification output already exists; use a new name to retain evidence")
    train, provenance = training_verification_data(config, seed, samples)
    root.mkdir(parents=True)
    results = []
    for arm in matrix["candidate_geo_arms"]:
        directory = root / arm["candidate_id"] / arm["geo_preprocessing"] / f"seed{seed}"
        result = train_run(
            config, arm["candidate_id"], arm["geo_preprocessing"], learning_rate,
            seed, train, None, directory, device=device,
            input_provenance=provenance, verification=True,
            verification_epochs=epochs, cpu_num_threads=cpu_threads,
        )
        bundle = load_inference_bundle(directory, device=device)
        predicted = bundle.predict(train["image_embedding"], train["geo_embedding"])
        with np.load(directory / "train_predictions.npz", allow_pickle=False) as saved:
            expected = saved["logits"]
        logits = predicted["logits"] if isinstance(predicted, dict) else predicted
        error = float(np.max(np.abs(np.asarray(logits) - expected)))
        if not np.allclose(logits, expected, atol=1e-6, rtol=1e-5):
            raise RuntimeError(f"Bundle prediction replay failed for {arm['arm_id']}: {error}")
        results.append({"arm_id": arm["arm_id"], "directory": str(directory),
                        "bundle_replay_max_abs_error": error, "result": result})
        print(json.dumps({"verified": arm["arm_id"], "replay_max_abs_error": error}), flush=True)
    report = {"study_id": config["study_id"], "purpose": "M2 training-only optimization smoke",
              "seed": seed, "rows": samples, "epochs": epochs, "learning_rate": learning_rate,
              "bundle_replay_tolerance": {"atol": 1e-6, "rtol": 1e-5},
              "validation_or_test_metrics_used": False,
              "counts_towards_development_comparison": False, "arms": results}
    (root / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def verify_resume(config: dict, *, name: str, seed: int = 1, device: str = "cpu",
                  cpu_threads: int = 1) -> dict:
    """Replay actual training-only inputs through interruption and portable export."""
    import random
    import numpy as np
    import torch
    from multimodal.attention_fusion_training import (
        export_inference_bundle, load_inference_bundle, train_run,
    )

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ProtocolError("Verification name must be a simple directory name")
    root = REPO_ROOT / config["paths"]["artifact_root"] / "verification" / name
    if root.exists():
        raise ProtocolError("Verification output already exists; retain evidence under a new name")
    train, provenance = training_verification_data(config, seed, 32)
    options = dict(device=device, cpu_num_threads=cpu_threads, input_provenance=provenance,
                   verification=True, verification_epochs=3)
    root.mkdir(parents=True)

    def fit(path, **kwargs):
        return train_run(config, "attention_two_token", "standardize", 0.001, seed,
                         train, None, path, **options, **kwargs)

    def equal(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and np.array_equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(
                equal(value, right[key]) for key, value in left.items())
        if isinstance(left, (tuple, list)):
            return isinstance(right, type(left)) and len(left) == len(right) and all(
                equal(a, b) for a, b in zip(left, right))
        return left == right

    full, resumed = root / "uninterrupted", root / "resumed"
    fit(full)
    fit(resumed, stop_after_epoch=1)
    random.seed(98451)
    np.random.seed(1654)
    torch.manual_seed(87544)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(16525)
    fit(resumed, resume=True)
    # These are trusted local checkpoints created immediately above.
    first = torch.load(full / "resume.pt", map_location="cpu", weights_only=False)
    second = torch.load(resumed / "resume.pt", map_location="cpu", weights_only=False)
    checks = {key: bool(equal(first[key], second[key])) for key in
              ("model_state", "optimizer_state", "rng", "history", "geo_preprocessing")}
    with np.load(full / "train_predictions.npz", allow_pickle=False) as a, \
            np.load(resumed / "train_predictions.npz", allow_pickle=False) as b:
        checks["saved_predictions"] = a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
        expected_logits = a["logits"].copy()
        expected_top3 = a["pred_top3"].copy()
    replay_errors = {}
    for label, directory in (("uninterrupted", full), ("resumed", resumed)):
        output = load_inference_bundle(directory, device=device).predict(
            train["image_embedding"], train["geo_embedding"], batch_size=config["training"]["batch_size"])
        replay_errors[label] = float(np.max(np.abs(output["logits"] - expected_logits)))
        checks[f"{label}_bundle_logits_exact"] = bool(np.array_equal(output["logits"], expected_logits))
        checks[f"{label}_bundle_top3_exact"] = bool(np.array_equal(output["pred_top3"], expected_top3))
    exported = root / "portable_bundle"
    export_inference_bundle(full, exported)
    portable = load_inference_bundle(exported, device=device).predict(
        train["image_embedding"], train["geo_embedding"], batch_size=config["training"]["batch_size"])
    checks["portable_bundle_logits_exact"] = bool(np.array_equal(portable["logits"], expected_logits))
    report = {"purpose": "M2 actual training-only resume and portable bundle verification",
              "seed": seed, "rows": len(train["label"]), "epochs": 3, "device": device,
              "checks": checks, "passed": all(checks.values()),
              "bundle_replay_max_abs_errors": replay_errors,
              "portable_bundle": str(exported),
              "validation_or_test_data_used": False,
              "counts_towards_development_comparison": False}
    (root / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"Resume/replay verification failed; see {root / 'verification.json'}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="List the complete declared workload and statuses; loads no data")
    train = sub.add_parser("train", help="Execute exactly one declared development fit")
    train.add_argument("--fit-id", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after-epoch", type=int)
    smoke = sub.add_parser("smoke", help="Check all 19 arms on a saved training-only image subset")
    smoke.add_argument("--name", required=True)
    smoke.add_argument("--seed", type=int, default=1)
    smoke.add_argument("--samples", type=int, default=32)
    smoke.add_argument("--epochs", type=int, default=3)
    smoke.add_argument("--learning-rate", type=float, default=0.001)
    resume_check = sub.add_parser("resume-check", help="Verify actual train-only resume and portable export")
    resume_check.add_argument("--name", required=True)
    resume_check.add_argument("--seed", type=int, default=1)
    for command in (train, smoke, resume_check):
        command.add_argument("--device", default="cpu")
        command.add_argument("--cpu-threads", type=int, default=1)
    export = sub.add_parser("export", help="Export an existing bundle including encoder bytes")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            from multimodal.attention_fusion_training import export_inference_bundle
            result = export_inference_bundle(args.source, args.destination)
        else:
            config, matrix, _ = read_frozen_plan(args.config)
            if args.command == "plan":
                result = plan_status(config, matrix)
            elif args.command == "train":
                result = execute_fit(config, select_fit(matrix, args.fit_id), device=args.device,
                                     cpu_threads=args.cpu_threads, resume=args.resume,
                                     stop_after_epoch=args.stop_after_epoch)
            elif args.command == "smoke":
                result = run_smoke(config, matrix, name=args.name, seed=args.seed,
                                   samples=args.samples, epochs=args.epochs,
                                   learning_rate=args.learning_rate, device=args.device,
                                   cpu_threads=args.cpu_threads)
            else:
                result = verify_resume(config, name=args.name, seed=args.seed,
                                       device=args.device, cpu_threads=args.cpu_threads)
    except (ValueError, RuntimeError, OSError, KeyError, yaml.YAMLError) as exc:
        parser.exit(1, f"Attention-fusion execution failed: {exc}\n")
    if args.command == "smoke":
        result = {"status": "complete", "verified_arms": len(result["arms"]),
                  "rows": result["rows"], "epochs": result["epochs"],
                  "report": str(REPO_ROOT / config["paths"]["artifact_root"]
                                / "verification" / args.name / "verification.json")}
    elif args.command == "train":
        result = {key: result[key] for key in ("status", "output_dir")}
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

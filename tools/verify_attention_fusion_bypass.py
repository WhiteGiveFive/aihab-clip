#!/usr/bin/env python3
"""Verify training-only bypass resume or replay all selected development heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multimodal.attention_fusion_bypass_protocol import load_config, read_frozen_plan  # noqa: E402
from multimodal.attention_fusion_bypass_training import load_inference_bundle, train_run  # noqa: E402
from multimodal.attention_fusion_inputs import load_input_provenance, load_prepared_inputs  # noqa: E402
from multimodal.attention_fusion_training import (  # noqa: E402
    _configure_runtime, configuration_hash, sha256_file,
)

DEFAULT_CONFIG = Path("configs/multimodal_attention_fusion_cls_bypass.yaml")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact(left, right) -> bool:
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            exact(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            exact(a, b) for a, b in zip(left, right))
    return left == right


def check_predictions(directory: Path, arrays: dict, config: dict, device: str,
                      role: str) -> dict:
    bundle = load_inference_bundle(directory, device=device)
    replay = bundle.predict(arrays["image_embedding"], arrays["geo_embedding"],
                            batch_size=config["training"]["batch_size"])
    with np.load(directory / f"{role}_predictions.npz", allow_pickle=False) as saved:
        for key in ("row_uid", "plot_idx", "label"):
            require(np.array_equal(saved[key], arrays[key]), f"Replayed {key} differs")
        logits_equal = bool(np.array_equal(replay["logits"], saved["logits"]))
        top3_equal = bool(np.array_equal(replay["pred_top3"], saved["pred_top3"]))
        error = float(np.max(np.abs(replay["logits"] - saved["logits"])))
    require(logits_equal and top3_equal, f"Prediction replay is not exact: {directory} ({error})")
    return {"rows": len(arrays["label"]), "logits_exact": logits_equal,
            "top3_exact": top3_equal, "max_abs_logit_error": error}


def smoke(config: dict, matrix: dict, parent: dict, output: Path, device: str) -> dict:
    require(not output.exists(), "Keep previous verification evidence; use a new output directory")
    arrays = load_prepared_inputs(parent, 1, "train")
    train = {key: value[:64].copy() for key, value in arrays.items()}
    provenance = load_input_provenance(parent, 1)
    provenance.update(source_role="train", verification_training_prefix_rows=64)
    output.mkdir(parents=True)
    options = dict(device=device, cpu_num_threads=1, input_provenance=provenance,
                   verification=True, verification_epochs=3)
    checks = []
    for arm in matrix["candidate_geo_arms"]:
        directory = output / arm["arm_id"]
        train_run(config, arm["candidate_id"], arm["geo_preprocessing"], 0.001,
                  1, train, None, directory, **options)
        result = {"arm_id": arm["arm_id"], **check_predictions(directory, train, config, device, "train")}
        history = json.loads((directory / "history.json").read_text())
        result["initial_optimization_loss"] = history[0]["optimization_loss"]
        result["final_optimization_loss"] = history[-1]["optimization_loss"]
        if arm["geo_preprocessing"] == "standardize":
            resumed = output / f"{arm['arm_id']}__resume"
            train_run(config, arm["candidate_id"], "standardize", 0.001, 1,
                      train, None, resumed, stop_after_epoch=1, **options)
            random.seed(9381)
            np.random.seed(8274)
            torch.manual_seed(7193)
            if torch.device(device).type == "cuda":
                torch.cuda.manual_seed_all(6187)
            train_run(config, arm["candidate_id"], "standardize", 0.001, 1,
                      train, None, resumed, resume=True, **options)
            first = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
            second = torch.load(resumed / "resume.pt", map_location="cpu", weights_only=False)
            resume_checks = {key: exact(first[key], second[key]) for key in
                             ("model_state", "optimizer_state", "rng", "history", "geo_preprocessing")}
            require(all(resume_checks.values()), f"Resume differs: {resume_checks}")
            with np.load(directory / "train_predictions.npz", allow_pickle=False) as a, \
                    np.load(resumed / "train_predictions.npz", allow_pickle=False) as b:
                require(a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files),
                        "Resumed predictions differ from uninterrupted training")
            result["resume_checks"] = resume_checks
            result["resumed_replay"] = check_predictions(resumed, train, config, device, "train")
        checks.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    return {"status": "passed", "stage": "training_only_smoke_and_resume",
            "rows": 64, "epochs": 3, "seed": 1, "checks": checks,
            "validation_cache_opened_for_provenance_authentication": True,
            "validation_used_for_training_or_scoring": False, "test_data_opened": False,
            "counts_towards_scientific_budget": False}


def replay(config: dict, matrix: dict, parent: dict, analysis_dir: Path,
           device: str) -> dict:
    manifest = json.loads((analysis_dir / "manifest.json").read_text())
    require(manifest.get("status") == "complete", "Analysis is incomplete")
    require(configuration_hash({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"})
            == manifest["manifest_payload_sha256"], "Analysis manifest checksum differs")
    selection_path = analysis_dir / "selection.json"
    require(sha256_file(selection_path) == manifest["files"]["selection.json"], "Selection changed")
    selection = json.loads(selection_path.read_text())
    require(configuration_hash({k: v for k, v in selection.items() if k != "selection_payload_sha256"})
            == selection["selection_payload_sha256"], "Selection payload checksum differs")
    require(selection["config_sha256"] == configuration_hash(config), "Selection belongs to another config")
    refs = selection["selected_references"]
    require(len(refs) == 24, "Expected all 24 selected extension checkpoints")
    require(len({ref["fit_id"] for ref in refs}) == 24, "Duplicate selected checkpoint")
    allowed = {fit["output_dir"]: fit for fit in matrix["development_fits"]}
    expected_pairs = {(fit["arm_id"], fit["seed"]) for fit in allowed.values()}
    actual_pairs = {(ref["arm_id"], ref["seed"]) for ref in refs}
    require(actual_pairs == expected_pairs, "Selected checkpoints do not cover every arm and seed")
    validation = {seed: load_prepared_inputs(parent, seed, "validation")
                  for seed in config["data"]["training_seeds"]}
    checks = []
    for ref in refs:
        require(ref["bundle_path"] in allowed, "Selected bundle is outside the declared fits")
        declared = allowed[ref["bundle_path"]]
        require(all(ref[key] == declared[key] for key in ("fit_id", "arm_id", "seed")),
                "Selected checkpoint identity differs from declared fit")
        directory = REPO_ROOT / ref["bundle_path"]
        require(sha256_file(directory / "manifest.json") == ref["bundle_manifest_sha256"],
                "Bundle manifest changed")
        require(sha256_file(directory / "final.pt") == ref["checkpoint_sha256"], "Checkpoint changed")
        require(allowed[ref["bundle_path"]]["seed"] == ref["seed"], "Selected seed differs")
        checks.append({"fit_id": ref["fit_id"], "checkpoint_sha256": ref["checkpoint_sha256"],
                       **check_predictions(directory, validation[ref["seed"]], config, device, "validation")})
    return {"status": "passed", "stage": "selected_checkpoint_replay",
            "selected_checkpoints": len(checks), "checks": checks,
            "selection_sha256": sha256_file(selection_path), "all_logits_exact": True,
            "all_top3_exact": True, "max_abs_logit_error": 0.0,
            "scope": "fresh-process cached-vector inference; no training, pixel inference or test data"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "replay"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True,
                        help="New verification directory (smoke) or receipt file (replay)")
    parser.add_argument("--analysis-dir", type=Path)
    args = parser.parse_args(argv)
    config, matrix, _ = read_frozen_plan(args.config)
    _, parent, _, _ = load_config(args.config)
    output = args.output.resolve()
    root = (REPO_ROOT / config["paths"]["artifact_root"]).resolve()
    require(output.is_relative_to(root), "Verification output must remain in the extension study")
    require(not output.exists(), "Keep earlier verification evidence; choose a new output")
    _configure_runtime(config, 1, torch.device(args.device), 1)
    if args.stage == "smoke":
        require(output.is_relative_to(root / "verification"), "Smoke fits require the verification namespace")
        result = smoke(config, matrix, parent, output, args.device)
        destination = output / "verification.json"
    else:
        require(args.analysis_dir is not None, "Replay requires --analysis-dir")
        result = replay(config, matrix, parent, args.analysis_dir.resolve(), args.device)
        destination = output
    result.update(study_id=config["study_id"], config_sha256=configuration_hash(config),
                  script_sha256=sha256_file(Path(__file__)), device=args.device,
                  torch=str(torch.__version__))
    if torch.device(args.device).type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

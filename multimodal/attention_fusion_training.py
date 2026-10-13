"""Development-only frozen-feature training and replay for attention fusion.

Inputs are explicit in-memory development arrays. This module has no test-data
loader and never reconstructs or executes an image encoder during head fitting.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .attention_fusion import GeoPreprocessor, build_model, count_parameters


ARRAY_KEYS = {"image_embedding", "geo_embedding", "label", "row_uid", "plot_idx"}


class TrainingError(ValueError):
    """A fitting, provenance or checkpoint boundary was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def configuration_hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _atomic(path: Path, writer: Any) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _atomic(path, lambda temporary: temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def _write_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
    _atomic(path, write)


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {name: sha256_file(directory / name) for name in (
        "attention_fusion_training.py", "attention_fusion.py", "models.py",
    )}


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256(_json({"shape": array.shape, "dtype": array.dtype.str}).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validated_data(data: dict[str, Any], config: dict[str, Any], role: str) -> dict[str, np.ndarray]:
    _require(isinstance(data, dict) and set(data) == ARRAY_KEYS,
             f"{role} must contain exactly {sorted(ARRAY_KEYS)}")
    result = {"image_embedding": np.asarray(data["image_embedding"], dtype=np.float32),
              "geo_embedding": np.asarray(data["geo_embedding"], dtype=np.float64),
              "row_uid": np.asarray(data["row_uid"], dtype=str),
              "plot_idx": np.asarray(data["plot_idx"], dtype=str)}
    labels = np.asarray(data["label"])
    _require(labels.dtype.kind in "iu", f"{role} labels must be integer dense class IDs")
    result["label"] = labels.astype(np.int64, copy=False)
    n = len(result["label"])
    _require(n > 0, f"{role} cannot be empty")
    for name in ("label", "row_uid", "plot_idx"):
        _require(result[name].shape == (n,), f"{role}/{name} must have one item per row")
    for name, width in (("image_embedding", config["data"]["image_dim"]),
                        ("geo_embedding", config["data"]["geo_dim"])):
        _require(result[name].shape == (n, width), f"Wrong {role}/{name} dimensions")
        _require(bool(np.isfinite(result[name]).all()), f"Nonfinite {role}/{name}")
    _require(len(set(result["row_uid"].tolist())) == n, f"Duplicate {role} row IDs")
    _require(bool((result["row_uid"] != "").all() and (result["plot_idx"] != "").all()),
             f"Empty {role} identity")
    _require(result["row_uid"].tolist() == sorted(result["row_uid"].tolist()),
             f"{role} must be ordered by row_uid")
    _require(bool(((labels >= 0) & (labels < config["data"]["num_classes"])).all()),
             f"{role} class ID outside the declared class mapping")
    return result


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Stable top-k, fixed class axes and zero-support-aware metrics (fractions)."""
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits)
    _require(logits.ndim == 2 and len(logits) == len(labels) and len(labels) > 0
             and bool(np.isfinite(logits).all()), "Invalid metric inputs")
    n, classes = logits.shape
    top3 = np.argsort(-logits, axis=1, kind="stable")[:, :min(3, classes)]
    predictions = top3[:, 0]
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros(classes, dtype=np.float64), where=denominator > 0)
    row_normalized = np.divide(cm, support[:, None], out=np.zeros_like(cm, dtype=np.float64), where=support[:, None] > 0)
    mcc_denominator = float((n * n - np.square(predicted.astype(float)).sum())
                            * (n * n - np.square(support.astype(float)).sum()))
    mcc = (float(tp.sum() * n - np.dot(support, predicted)) / np.sqrt(mcc_denominator)
           if mcc_denominator > 0 else 0.0)
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    log_partition = np.log(exponentials.sum(axis=1))
    loss = float(np.mean(log_partition - shifted[np.arange(n), labels]))
    metrics = {"rows": n, "acc1": float(np.mean(predictions == labels)),
               "acc3": float(np.mean((top3 == labels[:, None]).any(axis=1))),
               "weighted_f1": float(np.dot(f1, support) / n), "mcc": float(mcc),
               "loss": loss, "confusion_matrix": cm.tolist(),
               "confusion_row_normalized": [row.tolist() if count else [None] * classes
                                            for row, count in zip(row_normalized, support)],
               "class_support": support.tolist(),
               "class_recall": [float(tp[i] / count) if count else None for i, count in enumerate(support)]}
    return metrics, {"logits": logits.astype(np.float32), "probabilities": probabilities,
                     "pred_top3": top3.astype(np.int64), "prediction": predictions.astype(np.int64)}


def _environment(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("open_clip_torch", "torchvision", "safetensors", "Pillow", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver_version = None
    if device.type == "cuda" and platform.system() == "Linux":
        driver_path = Path("/proc/driver/nvidia/version")
        if driver_path.is_file():
            driver_version = driver_path.read_text().strip()
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if device.type == "cuda" else None,
            "nvidia_kernel_driver_version": driver_version,
            "construction_packages": packages,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _configure_runtime(config: dict[str, Any], seed: int, device: torch.device,
                       cpu_num_threads: int | None) -> None:
    if cpu_num_threads is not None:
        _require(cpu_num_threads > 0, "cpu_num_threads must be positive")
        torch.set_num_threads(cpu_num_threads)
    required_workspace = config["training"]["cublas_workspace_config"]
    if device.type == "cuda" and torch.cuda.is_initialized():
        _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == required_workspace,
                 "Set deterministic CUBLAS workspace before initializing CUDA")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required_workspace
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        # Allocator-stat resets do not initialize CUDA in PyTorch 2.5.1.
        torch.cuda.init()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)


def _rng_state(generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "shuffle": generator.get_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    generator.set_state(state["shuffle"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _predict_logits(model: nn.Module, image: np.ndarray, geo: np.ndarray,
                    device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(image), batch_size):
            x = torch.from_numpy(image[start:start + batch_size]).to(device)
            g = torch.from_numpy(geo[start:start + batch_size]).to(device)
            pieces.append(model(x, g).cpu().numpy())
    result = np.concatenate(pieces)
    _require(bool(np.isfinite(result).all()), "Nonfinite evaluation logits")
    return result


def _identity(config: dict[str, Any], candidate_id: str, geo_mode: str, rate: float,
              seed: int, data: dict[str, dict[str, np.ndarray]], provenance: dict[str, Any],
              fit_role: str, verification: bool, epochs: int, environment: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "attention_fusion.run.v1", "config_sha256": configuration_hash(config),
            "study_id": config["study_id"], "regime": config["regime"],
            "candidate_id": candidate_id, "geo_mode": geo_mode, "learning_rate": rate,
            "seed": seed, "fit_role": fit_role, "verification": verification, "epochs": epochs,
            "input_hashes": {role: {name: _array_hash(value) for name, value in sorted(values.items())}
                             for role, values in data.items()},
            "input_provenance": provenance, "source_hashes": _source_hashes(), "environment": environment}


def _check_recipe(config: dict[str, Any], candidate_id: str, geo_mode: str, rate: float,
                  seed: int, train: dict[str, np.ndarray], validation: dict[str, np.ndarray] | None,
                  output_dir: Path, provenance: dict[str, Any], fit_role: str,
                  verification: bool, verification_epochs: int | None) -> int:
    _require(fit_role in {"train", "all_development"}, "Only development fitting roles are permitted")
    _require(config["data"]["allow_test_features_or_labels"] is False, "Test-data access is prohibited")
    _require(config["regime"] == "frozen_cached" and config["image"]["encoder_frozen"] is True,
             "This runner only trains heads on frozen cached representations")
    _require(candidate_id in {candidate["id"] for candidate in config["candidates"]}, "Unknown candidate")
    _require(geo_mode in ({"not_applicable"} if candidate_id == "image_only" else set(config["geo"]["comparison"])),
             "Invalid candidate/preprocessing combination")
    _require(rate in config["training"]["learning_rates"] and seed in config["data"]["training_seeds"],
             "Learning rate or seed outside the declared study")
    _require(config["training"]["optimizer"] == "adamw"
             and config["training"]["scheduler"] == "none"
             and config["training"]["gradient_clipping"] == "none"
             and config["training"]["precision"] == "float32"
             and config["training"]["early_stopping"] is False
             and config["training"]["validation_selects_epoch"] is False,
             "Unsupported optimizer/selection recipe")
    supported_training = {
        "loss": "unweighted_cross_entropy", "label_smoothing": 0.0,
        "shuffle": True, "drop_last": False, "num_workers": 0,
        "decay_parameter_scope": "all_trainable", "validation_frequency_epochs": 1,
        "seed_python_numpy_torch": True, "independent_shuffle_generator": True,
        "deterministic_algorithms": True, "cudnn_benchmark": False, "allow_tf32": False,
        "save_every_candidate_final_checkpoint": True,
        "batch_schedule": "seed_matched_across_candidates_and_learning_rates",
    }
    for key, expected in supported_training.items():
        _require(config["training"].get(key) == expected, f"Unsupported training setting: {key}")
    _require(config["artifacts"].get("resume_boundary") == "completed_epoch"
             and config["artifacts"].get("save_latest_resume_each_epoch") is True,
             "Only completed-epoch resume is implemented")
    _require(isinstance(config["training"]["batch_size"], int)
             and config["training"]["batch_size"] > 0, "Batch size must be positive")
    epochs = config["training"]["epochs"]
    _require(isinstance(epochs, int) and epochs > 0
             and config["training"]["checkpoint_epoch"] == epochs, "Invalid final epoch contract")
    if verification:
        _require("verification" in output_dir.parts, "Verification outputs need a verification namespace")
        _require(provenance.get("source_role") == "train", "Verification data must originate only from development train")
        if verification_epochs is not None:
            _require(isinstance(verification_epochs, int) and 0 < verification_epochs <= epochs,
                     "Invalid verification epoch budget")
            epochs = verification_epochs
    else:
        _require(verification_epochs is None and "verification" not in output_dir.parts,
                 "Scientific fits cannot override their epoch budget")
        _require(provenance.get("protocol_config_sha256") == configuration_hash(config),
                 "Scientific fit must bind the frozen protocol configuration")
        _require(provenance.get("train_role") == fit_role, "Missing audited fitting-role binding")
        expected_rows = config["data"]["train_rows" if fit_role == "train" else "development_rows"]
        expected_plots = config["data"]["train_plots" if fit_role == "train" else "development_plots"]
        _require(len(train["label"]) == expected_rows and len(set(train["plot_idx"])) == expected_plots,
                 "Scientific fitting rows/plots differ from protocol")
        if candidate_id != "geo_only":
            encoder = provenance.get("encoder", {})
            _require(isinstance(encoder.get("path"), str) and isinstance(encoder.get("sha256"), str)
                     and len(encoder["sha256"]) == 64, "Missing authenticated encoder reference")
    if fit_role == "all_development":
        _require(validation is None, "All-development refit has no independent validation partition")
        _require(config["final_fit"]["keep_same_train_partition_encoder"] is True
                 and config["final_fit"]["initialization"] == "fresh_same_seed_and_selected_recipe"
                 and config["final_fit"]["epochs"] == config["training"]["epochs"], "Unsupported final-refit recipe")
    elif not verification:
        _require(validation is not None and provenance.get("validation_role") == "validation",
                 "Development selection fits require bound validation data")
        _require(len(validation["label"]) == config["data"]["validation_rows"]
                 and len(set(validation["plot_idx"])) == config["data"]["validation_plots"],
                 "Validation rows/plots differ from protocol")
    if validation is not None:
        for key in ("row_uid", "plot_idx"):
            _require(not set(train[key]) & set(validation[key]), f"Train/validation {key} overlap")
    if candidate_id != "geo_only":
        for values in (train, validation):
            if values is not None:
                norms = np.linalg.norm(values["image_embedding"].astype(np.float64), axis=1)
                _require(bool(np.allclose(norms, 1.0, atol=1e-6, rtol=0)), "Image embeddings must be L2-normalized")
    return epochs


def _verify_bundle(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    _require(manifest.get("status") == "complete", "Bundle is not complete")
    _require(manifest.get("manifest_payload_sha256") == configuration_hash({
        key: value for key, value in manifest.items() if key != "manifest_payload_sha256"
    }), "Bundle manifest self-hash differs")
    for relative, digest in manifest["files"].items():
        file = (path / relative).resolve()
        _require(file.is_relative_to(path.resolve()) and sha256_file(file) == digest,
                 f"Bundle file is missing or changed: {relative}")
    return manifest


def train_run(config: dict[str, Any], candidate_id: str, geo_mode: str,
              learning_rate: float, seed: int, train: dict[str, Any],
              validation: dict[str, Any] | None, output_dir: str | Path,
              device: str = "cpu", *, input_provenance: dict[str, Any] | None = None,
              fit_role: str = "train", verification: bool = False,
              verification_epochs: int | None = None, resume: bool = False,
              stop_after_epoch: int | None = None, cpu_num_threads: int | None = None) -> dict[str, Any]:
    """Fit one declared candidate; checkpoints resume only at completed epochs.

    Verification explicitly allows train-derived small subsets/shorter budgets.
    Resume files are trusted local artifacts generated by this function.
    """
    config = copy.deepcopy(config)
    provenance = copy.deepcopy(input_provenance or {})
    _json(provenance)
    output = Path(output_dir).resolve()
    train_data = _validated_data(train, config, "train")
    validation_data = _validated_data(validation, config, "validation") if validation is not None else None
    epochs = _check_recipe(config, candidate_id, geo_mode, learning_rate, seed, train_data,
                           validation_data, output, provenance, fit_role, verification, verification_epochs)
    _require(stop_after_epoch is None or isinstance(stop_after_epoch, int) and 0 < stop_after_epoch <= epochs,
             "stop_after_epoch must be a completed epoch within this run")
    selected_device = torch.device(device)
    _configure_runtime(config, seed, selected_device, cpu_num_threads)
    data = {"train": train_data}
    if validation_data is not None:
        data["validation"] = validation_data
    spec = _identity(config, candidate_id, geo_mode, learning_rate, seed, data, provenance,
                     fit_role, verification, epochs, _environment(selected_device))
    identity = configuration_hash(spec)
    if (output / "manifest.json").exists():
        completed = _verify_bundle(output)
        _require(resume and completed["run_identity_sha256"] == identity,
                 "Completed bundle is immutable; use resume only for identical verified reuse")
        return {"status": "reused_complete", "output_dir": str(output), "manifest": completed}
    if output.exists() and any(output.iterdir()):
        _require(resume and (output / "run_spec.json").is_file(), "Existing partial run requires explicit resume")
        previous = json.loads((output / "run_spec.json").read_text())
        _require(configuration_hash(previous) == identity,
                 "Resume identity differs in source, input, config, fitting recipe or environment")
    else:
        _require(not resume, "There is no partial run to resume")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise TrainingError("Run already has a writer lock; inspect the prior process before recovery") from exc
    os.close(descriptor)
    start = time.monotonic()
    try:
        model = build_model(config, candidate_id, seed).to(selected_device)
        processor = GeoPreprocessor("native" if geo_mode == "not_applicable" else geo_mode,
                                    config["data"]["geo_columns"],
                                    l2_epsilon=config["geo"]["l2_epsilon"],
                                    constant_epsilon=config["geo"]["standardize_constant_epsilon"])
        processor.fit(train_data["geo_embedding"], train_data["row_uid"])
        geo, zero_counts = {}, {}
        for role, values in data.items():
            geo[role] = processor.transform(values["geo_embedding"])
            zero_counts[role] = processor.last_transform_zero_vector_count
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                     betas=tuple(config["training"]["betas"]),
                                     eps=config["training"]["epsilon"],
                                     weight_decay=config["training"]["weight_decay"])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        history, last_epoch, previous_seconds = [], 0, 0.0
        if resume:
            _require((output / "resume.pt").is_file(), "No completed-epoch checkpoint exists")
            saved = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
            _require(saved["run_identity_sha256"] == identity, "Resume checkpoint identity differs")
            _require(saved["geo_preprocessing"] == processor.to_dict(), "Resume geo fitting state differs")
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and key != "step":
                        state[key] = value.to(selected_device)
            history, last_epoch = saved["history"], saved["completed_epoch"]
            previous_seconds = saved["elapsed_seconds"]
            _restore_rng(saved["rng"], generator)
        else:
            _write_json(output / "run_spec.json", spec)
            _write_json(output / "config.json", config)
            _write_json(output / "geo_preprocessing.json", processor.to_dict())
        _write_json(output / "run.json", {"status": "running", "run_identity_sha256": identity,
                                         "completed_epoch": last_epoch, "verification": verification})
        batch_size = config["training"]["batch_size"]
        cutoff = stop_after_epoch if stop_after_epoch is not None else epochs
        _require(cutoff >= last_epoch, "Cannot resume to an earlier epoch")
        for epoch in range(last_epoch + 1, cutoff + 1):
            model.train()
            order = torch.randperm(len(train_data["label"]), generator=generator).numpy()
            loss_total = 0.0
            for offset in range(0, len(order), batch_size):
                indices = order[offset:offset + batch_size]
                x = torch.from_numpy(train_data["image_embedding"][indices]).to(selected_device)
                g = torch.from_numpy(geo["train"][indices]).to(selected_device)
                y = torch.from_numpy(train_data["label"][indices]).to(selected_device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x, g), y)
                _require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                loss.backward()
                _require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                             for parameter in model.parameters()), "Nonfinite gradient")
                optimizer.step()
                loss_total += float(loss.item()) * len(indices)
            entry: dict[str, Any] = {"epoch": epoch, "optimization_loss": loss_total / len(order)}
            for role, values in data.items():
                logits = _predict_logits(model, values["image_embedding"], geo[role], selected_device, batch_size)
                entry[role] = classification_metrics(values["label"], logits)[0]
            history.append(entry)
            saved = {"run_identity_sha256": identity, "completed_epoch": epoch,
                     "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "rng": _rng_state(generator, selected_device), "history": history,
                     "geo_preprocessing": processor.to_dict(),
                     "elapsed_seconds": previous_seconds + time.monotonic() - start}
            _atomic(output / "resume.pt", lambda path: torch.save(saved, path))
            last_epoch = epoch
            _write_json(output / "history.json", history)
            _write_json(output / "run.json", {"status": "running", "run_identity_sha256": identity,
                                              "completed_epoch": epoch, "verification": verification})
        if last_epoch < epochs:
            _write_json(output / "run.json", {"status": "interrupted_at_epoch_boundary",
                                              "run_identity_sha256": identity,
                                              "completed_epoch": last_epoch, "verification": verification})
            return {"status": "interrupted_at_epoch_boundary", "completed_epoch": last_epoch,
                    "output_dir": str(output), "run_identity_sha256": identity}
        final_metrics = {}
        for role, values in data.items():
            logits = _predict_logits(model, values["image_embedding"], geo[role], selected_device, batch_size)
            metrics, predictions = classification_metrics(values["label"], logits)
            final_metrics[role] = metrics
            _write_npz(output / f"{role}_predictions.npz", {
                "row_uid": values["row_uid"], "plot_idx": values["plot_idx"],
                "label": values["label"], **predictions,
            })
        _atomic(output / "final.pt", lambda path: torch.save(
            {"epoch": epochs, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, path))
        _write_json(output / "metrics.json", final_metrics)
        completed_at = datetime.now(timezone.utc).isoformat()
        result = {"status": "complete", "run_identity_sha256": identity, "completed_epoch": epochs,
                  "verification": verification, "trainable_parameters": count_parameters(model),
                  "elapsed_seconds": previous_seconds + time.monotonic() - start,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(selected_device) if selected_device.type == "cuda" else None,
                  "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                                            * (1 if platform.system() == "Darwin" else 1024),
                  "peak_process_rss_scope": "process_lifetime_high_water_not_isolated_run_peak",
                  "geo_zero_vector_counts": zero_counts,
                  "completed_at": completed_at}
        _write_json(output / "run.json", result)
        root = Path(__file__).resolve().parents[1]
        git = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
        files = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name not in {".writer.lock", "manifest.json"}}
        manifest = {**result, "schema_version": "attention_fusion.bundle.v1", "files": files,
                    "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
                    "git_dirty": bool(git.stdout.strip()) if git.returncode == 0 else None,
                    "boundary": {"test_data_loaded": False, "encoder_executed": False},
                    "class_map": config["data"]["dense_to_canonical_l3"]}
        manifest["manifest_payload_sha256"] = configuration_hash(manifest)
        _write_json(output / "manifest.json", manifest)
        return {"status": "complete", "output_dir": str(output), "manifest": manifest,
                "metrics": final_metrics}
    except BaseException as exc:
        if not (output / "manifest.json").exists():
            _write_json(output / "run.json", {"status": "interrupted_or_failed", "run_identity_sha256": identity,
                                              "completed_epoch": locals().get("last_epoch", 0),
                                              "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        lock.unlink(missing_ok=True)


class InferenceBundle:
    """A restored head and its fitted preprocessing; accepts cached global vectors."""

    def __init__(self, directory: Path, device: str = "cpu") -> None:
        self.directory = directory.resolve()
        self.manifest = _verify_bundle(self.directory)
        self.config = json.loads((directory / "config.json").read_text())
        self.spec = json.loads((directory / "run_spec.json").read_text())
        _require(configuration_hash(self.spec) == self.manifest["run_identity_sha256"],
                 "Bundle run identity differs from specification")
        _require(self.spec["source_hashes"] == _source_hashes(), "Bundle model/trainer source differs from installed implementation")
        _require(configuration_hash(self.config) == self.spec["config_sha256"], "Bundle config identity mismatch")
        self.device = torch.device(device)
        self.model = build_model(self.config, self.spec["candidate_id"], self.spec["seed"]).to(self.device)
        state = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)
        _require(state["epoch"] == self.spec["epochs"], "Bundle checkpoint is not the declared final epoch")
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        self.preprocessor = GeoPreprocessor.from_dict(json.loads((directory / "geo_preprocessing.json").read_text()))
        self.encoder_reference = self.spec["input_provenance"].get("encoder")
        if (directory / "construction.json").exists():
            self.encoder_reference = json.loads((directory / "construction.json").read_text())["encoder"]

    def predict(self, image_embedding: Any, geo_embedding: Any,
                batch_size: int = 128) -> dict[str, np.ndarray]:
        image = np.asarray(image_embedding, dtype=np.float32)
        geo = self.preprocessor.transform(geo_embedding)
        _require(image.shape == (len(geo), self.config["data"]["image_dim"])
                 and bool(np.isfinite(image).all()) and len(image) > 0 and batch_size > 0,
                 "Invalid inference feature shapes/values")
        if self.spec["candidate_id"] != "geo_only":
            _require(bool(np.allclose(np.linalg.norm(image.astype(np.float64), axis=1),
                                      1.0, atol=1e-6, rtol=0)), "Inference image embeddings must be L2-normalized")
        logits = _predict_logits(self.model, image, geo, self.device, batch_size)
        # Labels here only reuse the stable probability/ranking implementation;
        # no ground-truth metrics are exposed for these artificial zeros.
        return classification_metrics(np.zeros(len(logits), dtype=np.int64), logits)[1]


def load_inference_bundle(directory: str | Path, device: str = "cpu") -> InferenceBundle:
    return InferenceBundle(Path(directory), device)


def export_inference_bundle(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Export head, encoder bytes, construction metadata and matching model code.

    Reconstruction still requires the recorded Python/PyTorch/OpenCLIP runtime;
    this function never loads the heavyweight vision model.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    bundle = load_inference_bundle(source)
    _require(not destination.exists(), "Export destination must not already exist")
    encoder = bundle.encoder_reference
    construction_assets = bundle.spec["input_provenance"].get("encoder_construction_assets", [])
    if bundle.spec["candidate_id"] != "geo_only":
        _require(isinstance(encoder, dict) and "path" in encoder and "sha256" in encoder,
                 "Portable image inference requires an authenticated encoder reference")
        encoder_path = Path(encoder["path"])
        if not encoder_path.is_absolute():
            encoder_path = source / encoder_path
        _require(sha256_file(encoder_path) == encoder["sha256"], "Referenced encoder bytes changed")
        _require(bool(construction_assets), "Portable image inference requires pinned architecture JSON assets")
        _require(any(asset.get("name") == "open_clip_config.json" for asset in construction_assets),
                 "Pinned OpenCLIP architecture config is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name in ("config.json", "run_spec.json", "geo_preprocessing.json", "final.pt"):
            shutil.copyfile(source / name, staging / name)
        exported_encoder = None
        exported_assets = []
        if bundle.spec["candidate_id"] != "geo_only":
            filename = "encoder" + encoder_path.suffix
            shutil.copyfile(encoder_path, staging / filename)
            _require(sha256_file(staging / filename) == encoder["sha256"], "Exported encoder copy failed verification")
            exported_encoder = {"path": filename, "sha256": encoder["sha256"]}
            asset_directory = staging / "encoder_assets"
            asset_directory.mkdir()
            seen = set()
            for asset in construction_assets:
                name = asset["name"]
                _require(name == Path(name).name and name.endswith(".json") and name not in seen,
                         "Construction asset names must be unique JSON basenames")
                seen.add(name)
                _require(sha256_file(Path(asset["path"])) == asset["sha256"], "Pinned construction asset changed")
                shutil.copyfile(asset["path"], asset_directory / name)
                _require(sha256_file(asset_directory / name) == asset["sha256"], "Construction asset copy changed")
                exported_assets.append({"path": f"encoder_assets/{name}", "sha256": asset["sha256"]})
        code_directory = staging / "implementation"
        code_directory.mkdir()
        for name in bundle.spec["source_hashes"]:
            shutil.copyfile(Path(__file__).parent / name, code_directory / name)
        _write_json(staging / "construction.json", {
            "encoder": exported_encoder, "image_construction_and_preprocessing": bundle.config["image"],
            "encoder_construction_assets": exported_assets,
            "architecture_json": "encoder_assets/open_clip_config.json" if exported_encoder else None,
            "source_encoder_initialization": bundle.spec["input_provenance"].get("encoder_initialization"),
            "class_map": bundle.config["data"]["dense_to_canonical_l3"],
            "head_config": "config.json", "head_weights": "final.pt",
            "geo_preprocessing": "geo_preprocessing.json", "implementation": "implementation/",
            "source_environment": bundle.spec["environment"],
            "limitation": "Pinned architecture JSON and encoder bytes are included; a compatible OpenCLIP runtime is still required. Pixel-level execution is separate from cached-vector replay.",
        })
        files = {str(path.relative_to(staging)): sha256_file(path)
                 for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = {"schema_version": "attention_fusion.bundle.v1", "status": "complete",
                    "run_identity_sha256": bundle.manifest["run_identity_sha256"],
                    "files": files, "source_bundle_manifest_sha256": sha256_file(source / "manifest.json"),
                    "portable_export": True}
        manifest["manifest_payload_sha256"] = configuration_hash(manifest)
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)

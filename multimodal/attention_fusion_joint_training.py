"""Fresh-run M3J fitting with direct freezing checks and shared checkpoints."""

from __future__ import annotations

import gc
import json
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from .attention_fusion import component_seed
from .attention_fusion_training import (
    _environment, _write_json, _write_npz, classification_metrics, configuration_hash,
)
from .attention_fusion_joint_protocol import require, source_hashes
from .attention_fusion_joint_storage import JointCheckpointIntegrityError, JointCheckpointStore


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_runtime(config: dict, seed: int, device: torch.device) -> None:
    cfg = config["training"]
    torch.set_num_threads(int(cfg["cpu_threads"]))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "The declared M3J recipe requires CUDA")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)


def rng_state() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def is_better(metrics: dict, best: dict | None) -> bool:
    if best is None:
        return True
    # Strict tuple ordering preserves the earliest epoch in exact ties.
    return tuple(metrics[k] for k in ("acc1", "weighted_f1", "mcc")) > tuple(
        best[k] for k in ("acc1", "weighted_f1", "mcc"))


def _batch(batch: dict, device: torch.device) -> tuple:
    image = batch.get("image")
    image = image.to(device, non_blocking=True) if image is not None else None
    return image, batch["geo"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)


def _epoch(model: nn.Module, loader: Any, device: torch.device,
           optimizer: torch.optim.Optimizer | None = None) -> tuple[dict, dict, float]:
    training = optimizer is not None
    model.train(training)
    synchronize(device)
    started = time.perf_counter()
    logits_all, labels_all, rows_all, plots_all = [], [], [], []
    loss_sum = 0.0
    seen = 0
    for batch in loader:
        image, geo, label = _batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = nullcontext() if training else torch.no_grad()
        with context:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=training and device.type == "cuda"):
                logits = model(image, geo)
                loss = nn.functional.cross_entropy(logits, label)
            require(bool(torch.isfinite(loss).item()), "Nonfinite loss; numerical policy cannot change silently")
            if training:
                loss.backward()
                # One scalar transfer checks every gradient without a sync per tensor.
                finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                require(bool(torch.stack(finite).all().item()), "Nonfinite gradient; fit stopped at last complete epoch")
                optimizer.step()
        loss_sum += float(loss.detach()) * len(label)
        seen += len(label)
        logits_all.append(logits.detach().float().cpu().numpy())
        labels_all.append(label.detach().cpu().numpy())
        rows_all.extend(list(batch["row_uid"]))
        plots_all.extend(list(batch.get("plot_idx", batch["row_uid"])))
    require(seen > 0, "Empty epoch")
    synchronize(device)
    elapsed = time.perf_counter() - started
    labels, logits = np.concatenate(labels_all), np.concatenate(logits_all)
    metrics, predictions = classification_metrics(labels, logits)
    metrics["optimization_loss" if training else "computed_loss"] = loss_sum / seen
    predictions.update(label=labels, row_uid=np.asarray(rows_all, dtype=str),
                       plot_idx=np.asarray(plots_all, dtype=str))
    return metrics, predictions, elapsed


def predict(model: nn.Module, loader: Any, device: torch.device) -> tuple[dict, dict, float]:
    return _epoch(model, loader, device)


def _state_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _state_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_state_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_state_cpu(v) for v in value)
    return value


def _finite_state(state: dict, context: str) -> None:
    for name, value in state.items():
        require(isinstance(value, torch.Tensor), f"Non-tensor state entry {context}: {name}")
        require(bool(torch.isfinite(value).all().item()), f"Nonfinite state {context}: {name}")


def _visual_parameters(model: nn.Module) -> dict:
    visual = getattr(model, "visual", None)
    return dict(visual.named_parameters()) if visual is not None else {}


def _model_loaders(loader_factory: Callable, inputs: dict, config: dict,
                   seed: int, epoch: int, model: nn.Module) -> tuple:
    return loader_factory(inputs, config, seed, epoch,
        train_transform=model.train_transform, prediction_transform=model.prediction_transform,
        include_images=getattr(model, "visual", None) is not None)


def frozen_visual_integrity(model: nn.Module, *, store: JointCheckpointStore | None = None) -> dict:
    """Anchor frozen values to the authenticated source, independent of GPU copies."""
    visual = getattr(model, "visual", None)
    values = _visual_parameters(model)
    parameters = [dict(name=name, shape=list(value.shape), dtype=str(value.dtype), elements=value.numel())
                  for name, value in sorted(values.items()) if not value.requires_grad]
    if getattr(model, "_initial_visual_source", None) is not None:
        from .attention_fusion_joint_inputs import canonical_frozen_values

        # The original encoder is the invariant for frozen parameters. Capturing
        # another device-to-host copy can itself introduce a bad reference.
        reference = canonical_frozen_values(model, [p["name"] for p in parameters])
        require(reference is not None, "Missing authenticated frozen source reference")
        reference_origin = "authenticated_original_encoder_checkpoint"
    else:
        reference = {p["name"]: values[p["name"]].detach().cpu().clone() for p in parameters}
        reference_origin = ("independent_initial_model_copy" if visual is not None
                            else "not_applicable_no_visual_encoder")
    _finite_state(reference, "in initial frozen snapshot")
    result = dict(algorithm="direct_frozen_values_v1", parameters=parameters,
                  reference_origin=reference_origin)
    if store is not None:
        result["reference"] = store.put_weights(reference) if reference else None
    else:
        # Short in-memory checks may avoid storage; production manifests always
        # use the serializable shared-store reference above.
        result["_snapshot"] = reference
    return result


def _verify_frozen_visual(snapshot: dict, expected: dict, context: str, *,
                          store: JointCheckpointStore | None = None) -> dict:
    require(expected["algorithm"] == "direct_frozen_values_v1", "Unknown frozen comparison schema")
    parameters = expected["parameters"]
    if "_snapshot" in expected:
        reference = expected["_snapshot"]
    elif parameters:
        require(store is not None and expected.get("reference") is not None,
                "Frozen comparison requires its independent initial reference")
        reference = store.load_weights(expected["reference"])
    else:
        reference = {}
    require(set(reference) == {p["name"] for p in parameters}, "Initial frozen snapshot keys changed")
    _finite_state(reference, "in initial frozen reference")
    for parameter in parameters:
        name = parameter["name"]
        require(name in snapshot, f"Frozen parameter missing {context}: {name}")
        actual = snapshot[name].detach().cpu()
        original = reference[name]
        require(list(actual.shape) == parameter["shape"] and str(actual.dtype) == parameter["dtype"]
                and original.shape == actual.shape and original.dtype == actual.dtype,
                f"Frozen parameter shape or dtype changed {context}: {name}")
        require(bool(torch.isfinite(actual).all().item()), f"Nonfinite frozen parameter {context}: {name}")
        if not torch.equal(actual, original):
            left, right = actual.reshape(-1), original.reshape(-1)
            for start in range(0, left.numel(), 65536):
                changed = torch.nonzero(left[start:start + 65536] != right[start:start + 65536]).reshape(-1)
                if changed.numel():
                    flat_index = start + int(changed[0])
                    index = tuple(int(i) for i in np.unravel_index(flat_index, tuple(actual.shape)))
                    initial, current = right[flat_index].item(), left[flat_index].item()
                    raise ValueError(f"Frozen parameter values changed {context}: {name}; first_index={index}; "
                                     f"initial={initial!r}; current={current!r}; "
                                     f"absolute_difference={abs(current - initial)!r}")
    applicable = expected.get("reference_origin") != "not_applicable_no_visual_encoder"
    return dict(status="exact" if applicable else "not_applicable", context=context,
                frozen_parameters_unchanged=True, visual_encoder_present=applicable,
                checked_tensors=len(parameters), checked_elements=sum(p["elements"] for p in parameters),
                comparison="direct_tensor_values_no_tolerance")


def verify_freezing(model: nn.Module, optimizer: torch.optim.Optimizer, *,
                    require_gradients: bool = False) -> dict:
    """Check trainability, optimizer membership and absence of frozen gradients."""
    named = list(model.named_parameters())
    trainable = {id(value): name for name, value in named if value.requires_grad}
    frozen = [(name, value) for name, value in named if not value.requires_grad]
    optimized = [value for group in optimizer.param_groups for value in group["params"]]
    require(len({id(value) for value in optimized}) == len(optimized), "Optimizer contains duplicate parameters")
    require({id(value) for value in optimized} == set(trainable),
            "Optimizer parameters differ from the declared trainable parameters")
    require(all(value.grad is None for _, value in frozen), "Frozen parameter has a gradient")
    if require_gradients:
        require(all(value.grad is not None for _, value in named if value.requires_grad),
                "Trainable parameter is missing its gradient")
    return dict(frozen_requires_grad_false=True, frozen_excluded_from_optimizer=True,
                frozen_gradients_absent=True, trainable_tensors=len(trainable), frozen_tensors=len(frozen),
                trainable_gradients_checked=require_gradients)


def _verify_loaded_state(module: nn.Module, saved: dict, component: str) -> None:
    """Validate saved/loaded structure and finiteness without a replay identity gate."""
    actual = module.state_dict()
    require(actual.keys() == saved.keys(), f"Loaded {component} state keys changed during checkpoint copy")
    _finite_state(saved, f"in saved {component}")
    _finite_state(actual, f"in loaded {component}")
    for name, expected in saved.items():
        value = actual[name]
        require(value.dtype == expected.dtype and value.shape == expected.shape,
                f"Loaded {component} tensor shape or dtype differs: {name}")


def _plot(history: list, directory: Path, title: str, final: bool = False,
          metadata: dict | None = None) -> None:
    from .attention_fusion_joint_reporting import plot_history
    state = rng_state()
    try:
        plot_history(history, directory / "curves", title=title, save_pdf=final, metadata=metadata)
    finally:
        restore_rng(state)


def _weight_refs(store: JointCheckpointStore, model: nn.Module,
                 cached_visual: dict | None = None) -> dict:
    weights = {}
    if getattr(model, "visual", None) is not None and cached_visual is None:
        snapshot = _state_cpu(model.visual.state_dict())
        _finite_state(snapshot, "before visual checkpoint publication")
        weights["visual"] = store.put_weights(snapshot)
    elif getattr(model, "visual", None) is not None:
        weights["visual"] = cached_visual
    fusion = _state_cpu(model.fusion.state_dict())
    _finite_state(fusion, "before fusion checkpoint publication")
    weights["fusion"] = store.put_weights(fusion)
    return weights


def load_weight_refs(store: JointCheckpointStore, model: nn.Module, refs: dict,
                     *, expected_frozen: dict | None = None) -> None:
    """Load authenticated objects; frozen-value checks are explicit start/end checks.

    ``expected_frozen`` is accepted for callers loading older inference bundles;
    it does not introduce an additional per-load frozen-value comparison.
    """
    has_visual = getattr(model, "visual", None) is not None
    require(("visual" in refs) == has_visual, "Checkpoint visual component differs from model")
    for component in (("visual", "fusion") if has_visual else ("fusion",)):
        saved = store.load_weights(refs[component])
        module = getattr(model, component)
        _verify_loaded_state(module, saved, component)
        module.load_state_dict(saved, strict=True)
        _verify_loaded_state(module, saved, component)


def _evaluate_checkpoints(config: dict, fit: dict, inputs: dict, directory: Path,
                          store: JointCheckpointStore, model: nn.Module, manifest: dict,
                          spec: dict, history: list, run: dict, *, selected: torch.device,
                          loader_factory: Callable, make_plots: bool,
                          guard: Callable | None = None) -> dict:
    """Publish each authenticated checkpoint evaluation without changing training."""
    attempt_path = directory / "finalization_attempts" / f"{time.time_ns()}.json"
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = dict(started_at=now(), status="running", run_identity=run["run_identity"],
                   training_epochs_performed=0, sources=source_hashes(),
                   previous_run=dict(run), completed_policies=[])
    _write_json(attempt_path, attempt)
    started = time.perf_counter()
    evaluations = dict(manifest.get("checkpoint_evaluation", {}))
    policy = None
    try:
        if guard is not None:
            guard()
        _, validation_loader = _model_loaders(loader_factory, inputs, config,
                                             int(fit["seed"]), len(history), model)
        for policy in ("final", "best"):
            if guard is not None:
                guard()
            attempt.update(current_policy=policy, updated_at=now())
            _write_json(attempt_path, attempt)
            load_weight_refs(store, model, manifest[policy])
            metrics, predictions, duration = predict(model, validation_loader, selected)
            manifest[policy]["metrics"] = metrics
            manifest[policy]["metrics_source"] = "reloaded_checkpoint_evaluation"
            _write_npz(directory / f"{policy}_predictions.npz", predictions)
            evaluations[policy] = dict(epoch=manifest[policy]["epoch"], seconds=duration,
                metrics_source="reloaded_checkpoint_evaluation", compared_for_exact_identity=False,
                result_assurance=spec["result_assurance"])
            complete = all(name in evaluations for name in ("final", "best"))
            metrics_source = ("reloaded_checkpoint_evaluation" if complete
                              else "partial_reloaded_checkpoint_evaluation")
            manifest.update(checkpoint_evaluation=evaluations, metrics_source=metrics_source)
            # Commit successful final inference before attempting best inference:
            # one unreadable object must not discard the other policy's results.
            store.write_manifest(directory / "checkpoint_manifest.json", manifest)
            _write_json(directory / "metrics.json", dict(
                result_assurance=spec["result_assurance"],
                verification_policy=spec["verification_policy"], metrics_source=metrics_source,
                **{name: dict(epoch=manifest[name]["epoch"], metrics=manifest[name]["metrics"],
                              metrics_source=manifest[name].get("metrics_source", "epoch_validation"))
                   for name in ("final", "best")}))
            _write_json(directory / "checkpoint_evaluation.json", evaluations)
            attempt["completed_policies"].append(policy)
            _write_json(attempt_path, attempt)
        if make_plots:
            _plot(history, directory, fit["run_id"], final=True, metadata=spec)
        run.pop("error", None)
        run.pop("checkpoint_unresolved", None)
        run.pop("checkpoint_issue", None)
        run.update(status="complete", updated_at=now(), completed_epoch=len(history),
                   best_epoch=manifest["best"]["epoch"], checkpoint_evaluation=evaluations,
                   metrics_source="reloaded_checkpoint_evaluation",
                   finalization_seconds=time.perf_counter() - started)
        _write_json(directory / "run.json", run)
        attempt.update(status="complete", finished_at=now(), seconds=time.perf_counter() - started)
        _write_json(attempt_path, attempt)
        return run
    except BaseException as error:
        status = ("checkpoint_unresolved" if isinstance(error, JointCheckpointIntegrityError)
                  else "failed_or_interrupted")
        detail = dict(policy=policy, error=f"{type(error).__name__}: {error}")
        if isinstance(error, JointCheckpointIntegrityError):
            detail["integrity"] = error.to_dict()
        run.update(status=status, updated_at=now(), error=detail["error"],
                   checkpoint_evaluation=evaluations,
                   finalization_seconds=time.perf_counter() - started)
        if status == "checkpoint_unresolved":
            run["checkpoint_unresolved"] = detail
        _write_json(directory / "run.json", run)
        attempt.update(status=status, finished_at=now(), seconds=time.perf_counter() - started,
                       failure=detail)
        _write_json(attempt_path, attempt)
        raise


def _completed_fit_for_finalization(config: dict, fit: dict, inputs: dict, directory: Path,
                                     store: JointCheckpointStore) -> tuple[dict, dict, list, dict]:
    """Validate original identity and completion without loading training state."""
    spec = json.loads((directory / "run_spec.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    # Authenticate the manifest itself now; authenticate model files separately
    # for each policy so a bad best file cannot suppress final evaluation.
    manifest = store.read_manifest(directory / "checkpoint_manifest.json", verify=False)
    require(configuration_hash(config) == spec["config_sha256"]
            and config == json.loads((directory / "config.json").read_text())
            and config == manifest["config"], "Finalization configuration differs from original fit")
    require(manifest["fit"] == fit and all(spec.get(key) == value for key, value in fit.items())
            and all(run.get(key) == value for key, value in fit.items()),
            "Finalization fit identity differs from original fit")
    require(manifest["provenance"] == spec
            and configuration_hash(spec) == manifest["run_identity"] == run["run_identity"],
            "Finalization run identity or original provenance differs")
    require(inputs["provenance"] == spec["input_provenance"],
            "Finalization input provenance differs from original fit")
    require(inputs["scaler"] == manifest["preprocessing"]
            == json.loads((directory / "geo_preprocessing.json").read_text()),
            "Finalization preprocessing differs from original fit")
    epochs = int(spec["expected_epochs"])
    require(epochs == int(config["training"]["epochs"]) or spec.get("verification") is True,
            "Finalization epoch budget differs from configured training")
    require(epochs > 0 and epochs <= int(config["training"]["epochs"])
            and run["expected_epochs"] == epochs and run["completed_epoch"] == epochs
            and len(history) == epochs
            and [entry["epoch"] for entry in history] == list(range(1, epochs + 1)),
            "Finalization requires the complete declared training history")
    require(manifest.get("final", {}).get("epoch") == epochs
            and manifest["last"]["epoch"] == epochs
            and all(manifest["final"][name] == manifest["last"][name]
                    for name in (("fusion",) if fit["candidate_id"] == "geo_only"
                                 else ("visual", "fusion"))),
            "Finalization requires the saved final-epoch model")
    best = history[0]
    for entry in history[1:]:
        if is_better(entry["validation"], best["validation"]):
            best = entry
    require(manifest["best"]["epoch"] == best["epoch"]
            and manifest["best"]["selection_metrics"] == best["validation"]
            and manifest["final"]["training_metrics"] == history[-1]["validation"],
            "Finalization checkpoint selection differs from original epoch history")
    freeze_checks = json.loads((directory / "freeze_checks.json").read_text())
    end = freeze_checks.get("end", {})
    require(freeze_checks == manifest["freeze_checks"] and end.get("epoch") == epochs
            and end.get("values", {}).get("frozen_parameters_unchanged") is True
            and all(end.get("structure", {}).get(key) is True for key in (
                "frozen_requires_grad_false", "frozen_excluded_from_optimizer", "frozen_gradients_absent")),
            "Finalization requires the original successful end-of-fit freezing checks")
    labels = [item.get("label_name", item.get("name", str(item["dense_id"])))
              for item in inputs["source_config"]["class_ontology"]["classes"]]
    require(labels == manifest["class_labels"] == run["class_labels"],
            "Finalization class ontology differs from original fit")
    return spec, run, history, manifest


def finalize_fit(config: dict, fit: dict, inputs: dict, directory: Path, store: JointCheckpointStore,
                 *, device: str = "cuda", guard: Callable | None = None,
                 make_plots: bool = True) -> dict:
    """Evaluate saved final/best weights after all epochs, never resume training.

    Original run provenance, history, selection and weights remain authoritative.
    The caller validates any authorized execution-source migration via ``guard``.
    """
    from .attention_fusion_joint_inputs import make_model_from_manifest, build_loaders
    directory = Path(directory)
    if guard is not None:
        guard()
    spec, run, history, manifest = _completed_fit_for_finalization(config, fit, inputs, directory, store)
    selected = torch.device(device)
    require(spec.get("verification") or selected.type == "cuda",
            "Comparative M3J finalization must use the declared CUDA arithmetic")
    configure_runtime(config, int(fit["seed"]), selected)
    model = make_model_from_manifest(spec["model_provenance"], selected)
    try:
        return _evaluate_checkpoints(config, fit, inputs, directory, store, model, manifest,
            spec, history, run, selected=selected, loader_factory=build_loaders,
            make_plots=make_plots, guard=guard)
    finally:
        del model
        gc.collect()
        if selected.type == "cuda":
            torch.cuda.empty_cache()


def train_fit(config: dict, fit: dict, inputs: dict, directory: Path, store: JointCheckpointStore,
              *, device: str = "cuda", resume: bool = False, verification: bool = False,
              verification_epochs: int | None = None, stop_after_epoch: int | None = None,
              model: nn.Module | None = None, loader_factory: Callable | None = None,
              make_plots: bool = True, guard: Callable | None = None) -> dict:
    from .attention_fusion_joint_inputs import make_model, build_loaders
    require(not resume, "Training resume is disabled; restart the fit in a fresh directory")
    directory = Path(directory)
    require(not directory.exists() or not any(directory.iterdir()),
            "Existing fit artifacts cannot be overwritten; use a fresh directory")
    selected = torch.device(device)
    require(verification or selected.type == "cuda", "Comparative M3J runs must use the declared CUDA arithmetic")
    require(verification or verification_epochs is None, "Verification budget cannot alter comparative fits")
    epochs = int(verification_epochs if verification_epochs is not None else config["training"]["epochs"])
    require(epochs > 0 and epochs <= config["training"]["epochs"], "Invalid epoch budget")
    cutoff = min(epochs, stop_after_epoch if stop_after_epoch is not None else epochs)
    require(cutoff > 0, "Invalid stopping epoch")
    seed = int(fit["seed"])
    configure_runtime(config, seed, selected)
    model = model if model is not None else make_model(
        config, fit["candidate_id"], fit["initialization"], seed, fit["regime"], selected)
    model.to(selected)
    frozen_integrity = frozen_visual_integrity(model, store=store)
    start_values = _verify_frozen_visual(_visual_parameters(model), frozen_integrity,
                                         "at fit start", store=store)
    dropout_seed = component_seed(config["training"].get("dropout_seed_namespace", config["study_id"]),
                                  seed, "training_dropout")
    torch.manual_seed(dropout_seed)
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(dropout_seed)
    loader_factory = loader_factory or build_loaders
    cfg = config["training"]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
                                 eps=cfg["epsilon"], weight_decay=0.0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["scheduler_t_max"], eta_min=0.0)
    start_structure = verify_freezing(model, optimizer)
    verification_policy = config.get("verification_policy", "direct_freeze_checks_v1")
    require(verification_policy == "direct_freeze_checks_v1", "Unknown direct verification policy")
    spec = dict(**fit, config_sha256=configuration_hash(config), sources=source_hashes(),
                input_provenance=inputs["provenance"], expected_epochs=epochs,
                verification=verification, verification_policy=verification_policy,
                training_resume_supported=False, runtime=_environment(selected),
                model_provenance=getattr(model, "provenance", {}),
                trainable_manifest=getattr(model, "trainable_manifest", {}),
                frozen_visual_integrity=frozen_integrity,
                result_assurance=config.get("result_assurance", "provisional_unresolved_memory_integrity"),
                arithmetic={k: cfg[k] for k in ("autocast_dtype", "parameter_dtype", "grad_scaler",
                    "validation_autocast", "cudnn_allow_tf32", "matmul_allow_tf32", "float32_matmul_precision")})
    identity = configuration_hash(spec)
    manifest_path = directory / "checkpoint_manifest.json"
    history, manifest, best_metrics = [], {}, None
    last_epoch = 0
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "run_spec.json", spec)
    _write_json(directory / "config.json", config)
    _write_json(directory / "geo_preprocessing.json", inputs["scaler"])
    freeze_checks = dict(verification_policy=verification_policy,
                         start=dict(values=start_values, structure=start_structure))
    _write_json(directory / "freeze_checks.json", freeze_checks)
    class_labels = [c["label_name"] if "label_name" in c else c.get("name", str(c["dense_id"]))
                    for c in inputs["source_config"]["class_ontology"]["classes"]]
    run = dict(**fit, run_identity=identity, status="running", completed_epoch=0,
               expected_epochs=epochs, verification=verification, class_labels=class_labels,
               verification_policy=verification_policy, result_assurance=spec["result_assurance"],
               training_resume_supported=False,
               epoch_seconds_scope="training_validation_and_metric_computation; full wall time in timing.jsonl",
               updated_at=now(), elapsed_seconds=0.0)
    _write_json(directory / "run.json", run)
    started = time.perf_counter()
    cached_visual = None
    if (fit["regime"] == "frozen_online"
            and frozen_integrity.get("reference_origin") == "authenticated_original_encoder_checkpoint"
            and not any(parameter.requires_grad for parameter in model.visual.parameters())
            and set(model.visual.state_dict()) == {p["name"] for p in frozen_integrity["parameters"]}):
        # This already saved canonical object contains the entire frozen tower.
        # Reusing it avoids another device-to-host copy during checkpointing.
        cached_visual = frozen_integrity["reference"]
    try:
        for epoch in range(1, cutoff + 1):
            if guard is not None:
                guard()
            epoch_start = time.perf_counter()
            train_loader, validation_loader = _model_loaders(loader_factory, inputs, config,
                                                           seed, epoch, model)
            used_lr = float(optimizer.param_groups[0]["lr"])
            train_metrics, _, train_seconds = _epoch(model, train_loader, selected, optimizer)
            validation_metrics, validation_predictions, validation_seconds = predict(model, validation_loader, selected)
            scheduler.step()
            entry = dict(epoch=epoch, optimizer_steps=epoch * len(train_loader), learning_rate=used_lr,
                         next_learning_rate=float(optimizer.param_groups[0]["lr"]),
                         train=train_metrics, validation=validation_metrics,
                         train_seconds=train_seconds, validation_seconds=validation_seconds,
                         peak_cuda_bytes=torch.cuda.max_memory_allocated(selected) if selected.type == "cuda" else 0,
                         epoch_seconds=time.perf_counter() - epoch_start)
            history.append(entry)
            # Write curves' underlying history before any terminal check; a
            # failed fit retains its observed trajectory and previous checkpoint.
            _write_json(directory / "history.json", history)
            if epoch == cutoff:
                freeze_checks["end"] = dict(
                    structure=verify_freezing(model, optimizer, require_gradients=True),
                    values=_verify_frozen_visual(_visual_parameters(model), frozen_integrity,
                                                  "at fit end before checkpoint publication", store=store),
                    epoch=epoch)
                _write_json(directory / "freeze_checks.json", freeze_checks)
            better = is_better(validation_metrics, best_metrics)
            old_manifest = manifest
            checkpoint_start = time.perf_counter()
            weights = _weight_refs(store, model, cached_visual)
            if fit["regime"] == "frozen_online":
                cached_visual = weights["visual"]
            elapsed = time.perf_counter() - started
            state_ref = store.put_state(_state_cpu(dict(
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                completed_epoch=epoch, history=history, elapsed_seconds=elapsed,
                next_epoch=epoch + 1, shuffle_policy="seed_epoch_stateless", training_resume_supported=False)))
            manifest = dict(schema_version="attention_fusion_joint.checkpoint.v1", run_identity=identity,
                config=config, fit=fit, preprocessing=inputs["scaler"], class_labels=class_labels,
                provenance=spec, last=dict(**weights, state=state_ref, epoch=epoch),
                best=old_manifest.get("best"), freeze_checks=freeze_checks,
                training_resume_supported=False)
            if better:
                best_metrics = validation_metrics
                manifest["best"] = dict(**weights, epoch=epoch, metrics=validation_metrics,
                                        selection_metrics=validation_metrics)
                _write_npz(directory / "best_training_predictions.npz", validation_predictions)
            if epoch == epochs:
                manifest["final"] = dict(**weights, epoch=epoch, metrics=validation_metrics,
                                         training_metrics=validation_metrics)
                _write_npz(directory / "final_training_predictions.npz", validation_predictions)
            store.write_manifest(manifest_path, manifest)
            last_epoch = epoch
            _write_json(directory / "metrics.json", dict(
                result_assurance=spec["result_assurance"], verification_policy=verification_policy,
                metrics_source="epoch_validation",
                best=dict(epoch=manifest["best"]["epoch"], metrics=manifest["best"]["metrics"]),
                **({"final": dict(epoch=epoch, metrics=validation_metrics)} if epoch == epochs else {})))
            # Retain all active and archived fit references while pruning only
            # this trainer's superseded rolling objects.
            if old_manifest:
                candidates = list(old_manifest.get("last", {}).values())
                candidates.extend((old_manifest.get("best") or {}).values())
                refs = [x for x in candidates if isinstance(x, dict) and "sha256" in x and "path" in x]
                runs_root = store.root / "runs"
                retained = (list(runs_root.rglob("checkpoint_manifest.json")) if runs_root.exists()
                            else list(store.root.glob("*/checkpoint_manifest.json")))
                retained.extend((store.root / "initializations").rglob("checkpoint_manifest.json"))
                retained.extend((store.root / "interrupted_attempts").rglob("checkpoint_manifest.json"))
                removable = {r["path"] for r in store.plan_cleanup(retained)["remove"]}
                store.discard_unreferenced([r for r in refs if r["path"] in removable], retained)
            run.update(completed_epoch=epoch, updated_at=now(), elapsed_seconds=elapsed,
                       best_epoch=manifest["best"]["epoch"], validation=validation_metrics,
                       peak_cuda_bytes=entry["peak_cuda_bytes"])
            _write_json(directory / "run.json", run)
            checkpoint_seconds = time.perf_counter() - checkpoint_start
            plot_start = time.perf_counter()
            if make_plots:
                _plot(history, directory, fit["run_id"], final=epoch == epochs, metadata=spec)
            with (directory / "timing.jsonl").open("a") as timing:
                timing.write(json.dumps(dict(epoch=epoch, checkpoint_seconds=checkpoint_seconds,
                    plot_seconds=time.perf_counter() - plot_start,
                    full_epoch_seconds=time.perf_counter() - epoch_start)) + "\n")
            print(json.dumps(dict(event="epoch", run_id=fit["run_id"], epoch=epoch,
                 acc1=validation_metrics["acc1"], best_epoch=manifest["best"]["epoch"],
                 train_seconds=train_seconds, validation_seconds=validation_seconds,
                 elapsed_seconds=elapsed)), flush=True)
        if last_epoch < epochs:
            run.update(status="interrupted_at_epoch_boundary", updated_at=now())
            _write_json(directory / "run.json", run)
            return run
        # Saved checkpoints are the authority for published inference metrics.
        # This same inference-only path can recover completed training later.
        result = _evaluate_checkpoints(config, fit, inputs, directory, store, model, manifest,
            spec, history, run, selected=selected, loader_factory=loader_factory,
            make_plots=make_plots, guard=guard)
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_json(directory / "run.json", result)
        return result
    except BaseException as error:
        status = ("checkpoint_unresolved" if isinstance(error, JointCheckpointIntegrityError)
                  else "failed_or_interrupted")
        run.update(status=status, updated_at=now(),
                   error=f"{type(error).__name__}: {error}", completed_epoch=last_epoch)
        _write_json(directory / "run.json", run)
        raise
    finally:
        del optimizer, scheduler, model
        gc.collect()
        if selected.type == "cuda":
            torch.cuda.empty_cache()

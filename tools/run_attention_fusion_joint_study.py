#!/usr/bin/env python3
"""Prepare, check and run one explicitly selected seed/regime M3J package."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from multimodal.attention_fusion_joint_protocol import (
    DEFAULT_CONFIG, authorized_fits, declare, declare_recovery_amendment, geo_reference_regimes, load_config,
    package_identity, package_name, read_recovery_amendment, realized_loader_namespaces, require,
    source_hashes, study_root,
)
from multimodal.attention_fusion_joint_inputs import prepare_inputs, make_model, build_loaders
from multimodal.attention_fusion_joint_storage import JointCheckpointIntegrityError, JointCheckpointStore
from multimodal.attention_fusion_joint_training import (
    _epoch, _model_loaders, _verify_frozen_visual, _visual_parameters,
    configure_runtime, frozen_visual_integrity,
    finalize_fit, load_weight_refs, now, predict, train_fit, verify_freezing,
)
from multimodal.attention_fusion_training import _environment, _write_json, configuration_hash


@contextmanager
def writer_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".study-writer.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _subset(inputs: dict, train_count: int = 32, prediction_count: int = 16) -> dict:
    """Verification holds out training rows only; actual validation is not scored."""
    result = dict(inputs)
    result["roles"] = {}
    source = inputs["roles"]["train"]
    for role, selection in (("train", slice(0, train_count)),
                            ("validation", slice(train_count, train_count + prediction_count))):
        item = {}
        for key, value in source.items():
            item[key] = value.iloc[selection].reset_index(drop=True) if key == "rows" else value[selection].copy()
        result["roles"][role] = item
    result["provenance"] = dict(inputs["provenance"], verification=True,
        prediction_source_role="train", train_rows=train_count, prediction_rows=prediction_count)
    return result


def _update_probes(model, regime: str) -> dict:
    """Check downstream optimization and visual updates only where applicable."""
    components = {"fusion": model.fusion}
    if regime == "joint_partial":
        components["visual"] = model.visual
    elif regime == "frozen_online":
        require(not any(p.requires_grad for p in model.visual.parameters()),
                "Frozen-online verification found trainable visual parameters")
    else:
        require(regime == "geo_only" and getattr(model, "visual", None) is None,
                "Geo-only verification requires no visual encoder")
    probes = {}
    for component, module in components.items():
        name, parameter = next((n, p) for n, p in module.named_parameters() if p.requires_grad)
        probes[component] = (name, parameter, parameter.detach().cpu().clone())
    return probes


def _check_updates(model, regime: str, probes: dict) -> dict:
    checked = {}
    for component, (name, parameter, before) in probes.items():
        require(not torch.equal(parameter.detach().cpu(), before),
                f"Representative trainable {component} parameter did not update: {name}")
        checked[component] = name
    if regime == "frozen_online":
        require(not any(module.training for module in model.visual.modules()),
                "Frozen-online visual modules must remain in evaluation mode")
    return checked


def package_report(config: dict, *, render_confusions: bool = True) -> dict:
    from multimodal.attention_fusion_joint_reporting import build_report
    root = study_root(config)
    reference = config["reporting"].get("paired_reference_root")
    completed_references = config["reporting"].get("completed_reference_roots")
    return build_report(root, root / "reports" / package_name(config),
                        seed=config["authorized_scope"]["seeds"][0],
                        regime=config["authorized_scope"]["regime"],
                        paired_reference_root=REPO_ROOT / reference if reference else None,
                        completed_reference_roots=([REPO_ROOT / path for path in completed_references]
                                                   if completed_references else None),
                        deferred_reference_regimes=config["reporting"].get("deferred_reference_regimes"),
                        render_confusions=render_confusions)


def prepare(config: dict) -> dict:
    """Record CPU input readiness without fitting or requiring future results."""
    root = study_root(config)
    with writer_lock(root):
        inputs = prepare_inputs(config)
        seed = config["authorized_scope"]["seeds"][0]
        regime = config["authorized_scope"]["regime"]
        references = config["reporting"].get("completed_reference_roots", [])
        reference_regimes = geo_reference_regimes(config) if regime == "geo_only" else ()
        if config["reporting"].get("paired_reference_root"):
            references = [config["reporting"]["paired_reference_root"]]
            reference_regimes = ("joint_partial",)
        dependencies = []
        for relative, reference_regime in zip(references, reference_regimes):
            reference_root = REPO_ROOT / relative
            protocol_path = reference_root / "protocol/protocol.json"
            record = dict(artifact_root=relative, regime=reference_regime,
                          seed=seed, required=True, status="not_started")
            if protocol_path.is_file():
                reference_config = json.loads(protocol_path.read_text())["config"]
                status_path = reference_root / "orchestration" / package_name(reference_config) / "status.json"
                previous = json.loads(status_path.read_text()) if status_path.is_file() else {}
                record.update(status=previous.get("queue_status", "incomplete"),
                              completed_fits=previous.get("complete", 0))
            dependencies.append(record)
        deferred = [dict(
            artifact_root=f"multimodal_artifacts/attention_fusion/cs/gse_100m/{package_identity(seed, item)}",
            regime=item, seed=seed, required=False, status="deferred", planned_fits=16,
        ) for item in config["reporting"].get("deferred_reference_regimes", [])]
        result = dict(
            status="prepared", scope="cpu_inputs_and_configuration_only",
            prepared_at=now(), study_id=config["study_id"], package=package_name(config),
            seed=seed, config=config, config_sha256=configuration_hash(config),
            sources=source_hashes(), input_provenance=inputs["provenance"],
            input_manifest=str(Path(inputs["manifest_path"]).relative_to(REPO_ROOT)),
            encoder_producers=inputs["encoder_producers"],
            authorized_fits=authorized_fits(config),
            realized_loader_rng=realized_loader_namespaces(config),
            preceding_packages=dependencies,
            deferred_preceding_packages=deferred,
            gpu_verification_required_before_fitting=True,
            comparative_training_started=False, validation_data_scored=False,
            test_data_scored=False, verification_policy=config["verification_policy"],
            result_assurance=config["result_assurance"],
        )
        (root / "preparation").mkdir(exist_ok=True)
        manifest = root / "preparation/manifest.json"
        if manifest.is_file():
            archive = root / "preparation/history" / f"manifest_{time.time_ns()}.json"
            archive.parent.mkdir(exist_ok=True)
            shutil.copy2(manifest, archive)
            result["superseded_preparation_manifest"] = str(archive.relative_to(root))
        _write_json(manifest, result)
        return result


def verify(config: dict, device: str) -> dict:
    """Check ordinary training and saved-model use without trajectory replay."""
    require(device.startswith("cuda"), "The production training/resource check requires CUDA")
    root = study_root(config)
    with writer_lock(root):
        print(json.dumps(dict(event="input_audit_start", stage="verification")), flush=True)
        inputs = prepare_inputs(config)
        print(json.dumps(dict(event="input_audit_passed", role_rows=inputs["provenance"]["role_rows"])), flush=True)
        sources = source_hashes()
        identity = configuration_hash(dict(config=config, sources=sources))
        # Each check is a fresh attempt; previous diagnostics remain available.
        directory = root / "verification" / f"{identity[:12]}_{time.time_ns()}"
        directory.mkdir(parents=True)
        assurance = config["result_assurance"]
        _write_json(root / "verification/latest.json", dict(
            status="running", config_sha256=configuration_hash(config),
            verification_policy=config["verification_policy"],
            directory=str(directory.relative_to(REPO_ROOT)), sources=sources,
            result_assurance=assurance))
        store = JointCheckpointStore(directory)
        selected = torch.device(device)
        probe_inputs = _subset(inputs, 16, 16)
        evidence = []
        for fit in authorized_fits(config):
            print(json.dumps(dict(event="gradient_check_start", run_id=fit["run_id"])), flush=True)
            configure_runtime(config, fit["seed"], selected)
            model = make_model(config, fit["candidate_id"], fit["initialization"],
                               fit["seed"], fit["regime"], selected)
            frozen_before = frozen_visual_integrity(model, store=store)
            probes = _update_probes(model, fit["regime"])
            torch.cuda.reset_peak_memory_stats(selected)
            cfg = config["training"]
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), eps=cfg["epsilon"],
                weight_decay=0.0, amsgrad=False)
            verify_freezing(model, optimizer)
            train_loader, val_loader = _model_loaders(build_loaders, probe_inputs, config,
                                                    fit["seed"], 1, model)
            train_metrics, _, train_seconds = _epoch(model, train_loader, selected, optimizer)
            verify_freezing(model, optimizer, require_gradients=True)
            frozen_check = _verify_frozen_visual(_visual_parameters(model), frozen_before,
                                                 "after a training step", store=store)
            checked_parameters = _check_updates(model, fit["regime"], probes)
            prediction_metrics, _, prediction_seconds = predict(model, val_loader, selected)
            record = dict(run_id=fit["run_id"], status="passed",
                          frozen_parameters_unchanged=True, frozen_check=frozen_check,
                          verification_policy=config["verification_policy"], result_assurance=assurance,
                          selected_visual_updated="visual" in checked_parameters,
                          selected_fusion_updated=True,
                          checked_trainable_parameters=checked_parameters,
                          all_trainable_parameters_have_gradients=True,
                          trainable_manifest=model.trainable_manifest, train_seconds=train_seconds,
                          prediction_seconds=prediction_seconds, training_rows=16, prediction_rows=16,
                          loss=train_metrics["optimization_loss"], prediction_loss=prediction_metrics["loss"],
                          peak_cuda_bytes=torch.cuda.max_memory_allocated(selected))
            evidence.append(record)
            _write_json(directory / "gradient_checks.json", evidence)
            print(json.dumps(dict(event="gradient_check", **{k: record[k] for k in (
                "run_id", "status", "train_seconds", "prediction_seconds", "peak_cuda_bytes")})), flush=True)
            del model, optimizer, train_loader, val_loader, probes, frozen_before
            gc.collect()
            torch.cuda.empty_cache()
        # One small fresh fit exercises epoch logging, best/final checkpoints,
        # plots and portable export. It never uses the development validation rows.
        fits = authorized_fits(config)
        fit = next((f for f in fits if f["candidate_id"] == "attention_two_token"
                    and f["initialization"] == "cs_adapted"), fits[0])
        small = _subset(inputs, 16, 16)
        train_fit(config, fit, small, directory / "checkpoint_check", store, device=device,
                  verification=True, verification_epochs=1, make_plots=True)
        saved = store.read_manifest(directory / "checkpoint_check/checkpoint_manifest.json")
        export_payload = {k: v for k, v in saved.items() if k not in ("last", "best", "_manifest_sha256")}
        store.write_manifest(directory / "export_source.json", export_payload)
        destination = directory / "standalone_export"
        export_manifest = store.export_bundle(directory / "export_source.json", destination)
        export_store = JointCheckpointStore(destination)
        exported = export_store.read_manifest(export_manifest)
        from multimodal.attention_fusion_joint_inputs import make_model_from_manifest
        model = make_model_from_manifest(exported["provenance"]["model_provenance"], selected)
        load_weight_refs(export_store, model, exported["final"])
        _, val_loader = _model_loaders(build_loaders, small, config, fit["seed"], 1, model)
        export_metrics, _, _ = predict(model, val_loader, selected)
        del model, val_loader
        gc.collect()
        torch.cuda.empty_cache()
        require(source_hashes() == sources, "Scientific source changed during GPU verification")
        result = dict(status="passed", verification_policy=config["verification_policy"],
                      config_sha256=configuration_hash(config), sources=sources,
                      input_provenance=inputs["provenance"], runtime=_environment(selected),
                      gradient_checks=evidence, fresh_checkpoint_check=True,
                      standalone_load_and_evaluate=True, standalone_metrics=export_metrics,
                      interrupted_training_comparison=False, exact_prediction_replay_required=False,
                      result_assurance=assurance, validation_data_scored=False,
                      directory=str(directory.relative_to(REPO_ROOT)), completed_at=now())
        _write_json(directory / "verification.json", result)
        _write_json(root / "verification/latest.json", result)
        return result


def archive_incomplete_attempt(root: Path, directory: Path, run_id: str) -> Path | None:
    """Preserve interrupted output and free the canonical path for epoch one."""
    if not directory.exists() or not any(directory.iterdir()):
        return None
    archived = root / "interrupted_attempts" / run_id / str(time.time_ns())
    archived.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(archived)
    print(json.dumps(dict(event="incomplete_attempt_archived", run_id=run_id,
                          archived=str(archived), restart_epoch=1)), flush=True)
    return archived


def status(config: dict) -> dict:
    root = study_root(config)
    rows = []
    for fit in authorized_fits(config):
        path = root / "runs" / fit["run_id"] / "run.json"
        item = json.loads(path.read_text()) if path.exists() else dict(**fit, status="planned", completed_epoch=0)
        rows.append(item)
    return dict(study_id=config["study_id"], authorized_fits=len(rows),
                verification_policy=config["verification_policy"],
                result_assurance=config["result_assurance"],
                complete=sum(r["status"] == "complete" for r in rows), fits=rows)


def _checkpoint_problem(directory: Path, error: JointCheckpointIntegrityError) -> None:
    """Retain the prior run record and exclude unresolved checkpoints from reports."""
    path = directory / "run.json"
    if not path.exists():
        return
    run_record = json.loads(path.read_text())
    (directory / "checkpoint_issues").mkdir(exist_ok=True)
    _write_json(directory / "checkpoint_issues" / f"{time.time_ns()}.json",
                dict(recorded_at=now(), previous_run=run_record, error=error.to_dict()))
    run_record.update(status="checkpoint_unresolved", updated_at=now(),
                      error=f"{type(error).__name__}: {error}", checkpoint_issue=error.to_dict())
    _write_json(path, run_record)


def run(config: dict, device: str, *, continuation: bool = False) -> dict:
    root = study_root(config)
    with writer_lock(root):
        print(json.dumps(dict(event="input_audit_start", stage="comparative_fitting")), flush=True)
        inputs = prepare_inputs(config)
        print(json.dumps(dict(event="input_audit_passed", role_rows=inputs["provenance"]["role_rows"])), flush=True)
        checked = json.loads((root / "verification/latest.json").read_text())
        sources = source_hashes()
        configure_runtime(config, config["authorized_scope"]["seeds"][0], torch.device(device))
        require(checked["status"] == "passed"
                and checked.get("verification_policy") == config["verification_policy"]
                and checked["config_sha256"] == configuration_hash(config)
                and checked["runtime"] == _environment(torch.device(device))
                and checked["input_provenance"] == inputs["provenance"],
                "Training-only verification is missing or belongs to different code/config/inputs")
        if checked["sources"] != sources:
            amendment = read_recovery_amendment(config, inputs["provenance"])
            require(amendment is not None and amendment["previous_sources"] == checked["sources"]
                    and amendment["prior_verification_sha256"] == configuration_hash(checked),
                    "Training-only verification is missing or belongs to different code/config/inputs")
        declare(config, inputs["provenance"])
        store = JointCheckpointStore(root)
        orchestration = root / "orchestration" / package_name(config)
        orchestration.mkdir(parents=True, exist_ok=True)

        def guard():
            require(source_hashes() == sources, "Scientific execution code/config changed during the study")

        issues = []

        def refresh(active=None, error=None):
            value = status(config)
            value.update(updated_at=now(), active_fit=active, error=error,
                         continuation_without_retraining=continuation, unresolved=issues)
            _write_json(orchestration / "status.json", value)
            return value

        for fit in authorized_fits(config):
            guard()
            directory = root / "runs" / fit["run_id"]
            refresh(fit["run_id"])
            try:
                existing = (json.loads((directory / "run.json").read_text())
                            if (directory / "run.json").exists() else None)
                if existing is not None and existing["status"] == "complete":
                    store.read_manifest(directory / "checkpoint_manifest.json")
                    print(json.dumps(dict(event="completed_fit_preserved", run_id=fit["run_id"])), flush=True)
                    continue
                if existing is not None and existing.get("completed_epoch") == config["training"]["epochs"]:
                    print(json.dumps(dict(event="checkpoint_evaluation_recovery_start", **fit)), flush=True)
                    finalize_fit(config, fit, inputs, directory, store, device=device, guard=guard)
                elif continuation and directory.exists() and any(directory.iterdir()):
                    issues.append(dict(run_id=fit["run_id"], kind="incomplete_training_preserved",
                                       completed_epoch=(existing or {}).get("completed_epoch", 0),
                                       message="No retraining was authorized by this continuation."))
                    print(json.dumps(dict(event="incomplete_training_preserved", **issues[-1])), flush=True)
                    refresh()
                    continue
                else:
                    if not continuation:
                        archive_incomplete_attempt(root, directory, fit["run_id"])
                    print(json.dumps(dict(event="fit_start", **fit)), flush=True)
                    train_fit(config, fit, inputs, directory, store, device=device, guard=guard)
                package_report(config, render_confusions=False)
                refresh()
            except JointCheckpointIntegrityError as error:
                _checkpoint_problem(directory, error)
                issues.append(dict(run_id=fit["run_id"], kind="checkpoint_unresolved",
                                   error=error.to_dict()))
                print(json.dumps(dict(event="checkpoint_unresolved_continuing_queue", **issues[-1])), flush=True)
                refresh()
            except BaseException as error:
                refresh(fit["run_id"], f"{type(error).__name__}: {error}")
                raise
        package_report(config, render_confusions=True)
        result = refresh()
        result["queue_status"] = ("complete" if result["complete"] == result["authorized_fits"]
                                  else "finished_with_unresolved_fits")
        _write_json(orchestration / "status.json", result)
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--background", action="store_true",
                        help="Launch the authorized run in a separate session with a persistent log")
    parser.add_argument("command", choices=("prepare", "verify", "amend-recovery", "run", "continue", "status", "report"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.background:
        require(args.command in ("run", "continue"), "Background execution is supported for the authorized run only")
        directory = study_root(config) / "orchestration" / package_name(config)
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / f"run_{time.time_ns()}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--config", str(args.config.resolve()), "--device", args.device, args.command]
        with log.open("wb") as output:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        launch = dict(pid=process.pid, log=str(log.relative_to(REPO_ROOT)),
                      command=command, launched_at=now(),
                      scope=f"{len(authorized_fits(config))} fits: {package_name(config)}",
                      continuation_without_retraining=args.command == "continue")
        (directory / "launches").mkdir(exist_ok=True)
        _write_json(directory / "launches" / f"{process.pid}_{time.time_ns()}.json", launch)
        if (directory / "launch.json").exists():
            previous = json.loads((directory / "launch.json").read_text())
            _write_json(directory / "launches" / f"previous_{previous['pid']}.json", previous)
        _write_json(directory / "launch.json", launch)
        print(json.dumps(launch, indent=2))
        return 0
    if args.command == "prepare":
        result = prepare(config)
    elif args.command == "verify":
        try:
            result = verify(config, args.device)
        except Exception as error:
            root = study_root(config)
            latest = root / "verification/latest.json"
            previous = json.loads(latest.read_text()) if latest.exists() else {}
            failure = dict(previous, status="failed", failed_at=now(),
                           error=f"{type(error).__name__}: {error}",
                           result_assurance=config["result_assurance"])
            latest.parent.mkdir(parents=True, exist_ok=True)
            _write_json(latest, failure)
            if "directory" in previous:
                directory = (REPO_ROOT / previous["directory"]).resolve()
                if directory.is_relative_to((root / "verification").resolve()):
                    _write_json(directory / "failure.json", failure)
            raise
    elif args.command == "amend-recovery":
        root = study_root(config)
        with writer_lock(root):
            inputs = prepare_inputs(config)
            checked = json.loads((root / "verification/latest.json").read_text())
            result = declare_recovery_amendment(config, inputs["provenance"], checked)
    elif args.command in ("run", "continue"):
        result = run(config, args.device, continuation=args.command == "continue")
    elif args.command == "status":
        result = status(config)
    else:
        result = package_report(config)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    return 2 if result.get("queue_status") == "finished_with_unresolved_fits" else 0


if __name__ == "__main__":
    raise SystemExit(main())

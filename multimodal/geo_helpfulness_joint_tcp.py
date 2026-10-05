"""Versioned orchestration for the approved joint four-state/TCP router study."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet import (
    _hash, _json, _load_arrays, _lock, _now, _save_arrays, _seal_stage,
    _staging, _validate_stage, _write_json,
)
from multimodal.geo_helpfulness_router_candidate_state import (
    TARGET_ORDER, full_feature_columns, variant_feature_indices,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
PROTOCOL = BASE / "protocol_v1"
OUTPUT = BASE / "joint_tcp_router_v1"
CONFIG = ROOT / "configs/multimodal_geo_joint_tcp_v1.json"
PLAN = ROOT / "dev_plans/reliability_aware_geo_fusion/18_joint_tcp_router_exploration.md"
SEEDS = (1, 2, 3, 4)
FOLDS = (0, 1, 2, 3)
SPLITS = ("train", "validation")
PREDICTION_KEYS = ("state_logits", "tcp_logits", "q", "tcp_hat")
CODE_PATHS = (
    "multimodal/geo_helpfulness_joint_tcp.py",
    "multimodal/geo_helpfulness_joint_tcp_extract.py",
    "multimodal/geo_helpfulness_joint_tcp_fit.py",
    "multimodal/geo_helpfulness_joint_tcp_report.py",
    "tools/run_multimodal_geo_helpfulness_joint_tcp.py",
)
PARENT_STUDIES = ("router_capacity_v1", "confidnet_oof_v1_replay_v2",
                  "confidnet_oof_relaxed_guardrails_v1")
RF_SELECTION = BASE / "relaxed_guardrails_rf_v1/selection"


class StudyPausedError(RuntimeError):
    """A failed scientific invariant stops further fitting or scoring."""


def expected_config():
    from multimodal.geo_helpfulness_joint_tcp_fit import DEFAULT_PARAMETERS
    configurations = []
    for short, variant in (("numeric25", "ablation_numeric25"),
                           ("no_pairs79", "ablation_no_pairs79")):
        for embeddings in (False, True):
            input_id = short + ("_embeddings" if embeddings else "")
            for weight, token in ((0.0, "0"), (0.1, "0p1"), (1.0, "1"), (10.0, "10")):
                configurations.append({
                    "config_id": f"joint__{input_id}__lambda_{token}",
                    "input_id": input_id, "variant": variant,
                    "with_embeddings": embeddings, "auxiliary_weight": weight,
                    "input_dim": len(variant_feature_indices(variant)) + (1216 if embeddings else 0),
                })
    return {
        "schema_version": "geo_helpfulness.joint_tcp_router_config.v1",
        "study_id": "joint_tcp_router_v1", "training_seeds": list(SEEDS),
        "configs": configurations, "configuration_count": 16, "fit_count": 64,
        "parameters": deepcopy(DEFAULT_PARAMETERS), "target_order": list(TARGET_ORDER),
        "tcp_order": ["fusion", "geo"], "tcp_target": "authoritative_native_t1_true_class_probability",
        "embedding_source": "paired_frozen_expert_image_l2_1152_and_producer_standardized_geo64",
        "additional_embedding_scaling": False, "score_dtype": "float64_after_native_float32_inference",
        "minimum_plots_grid": [20, 10, 0], "harm_numerator_grid": [5, 7, 9],
        "harm_denominator": 10, "minimum_coverage_fraction": 0.01,
        "positive_net_required": True, "weight_selection": "none_report_all",
        "bootstrap": {"resamples": 2000, "seed": 20260826,
                      "unit": "whole_plot_all_images_all_seed_realizations",
                      "selection_adjusted": False},
        "expert_updates": False, "locked_test_access": False,
        "verification": {"native_float32": "exact_serialized_state_replay",
                         "portable_float64": "diagnostic_only"},
        "evidence_status": "exploratory_reused_development_validation",
    }


def _config():
    config = _json(CONFIG)
    if config != expected_config():
        raise ValueError("Joint TCP configuration differs from the approved grid or recipe")
    return config


def _output(value):
    root = Path(value).resolve()
    if root == ROOT or root in ROOT.parents:
        raise ValueError("Study output cannot contain the repository")
    for protected in BASE.iterdir():
        if not protected.is_dir() or protected == OUTPUT:
            continue
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError("Study output must be disjoint from existing experiments")
    return root


def _snapshot(directory):
    return {str(p.relative_to(directory)): {
        "sha256": _hash(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
    } for p in sorted(directory.rglob("*")) if p.is_file()}


def declare(output_dir=OUTPUT):
    root = _output(output_dir)
    with _lock(root, "declare"):
        if (root / "declaration.json").exists():
            validate_declaration(root, check_parents=True)
            return {"status": "reused_valid"}
        config = _config()
        from multimodal import geo_helpfulness_oof as m2
        from multimodal.geo_helpfulness_router_capacity_inputs import load_capacity_inputs
        inputs = load_capacity_inputs(ROOT, PROTOCOL)
        context = m2.load_frozen_context()
        files = {Path(p) for p in inputs["parent_file_hashes"]}
        files.update(ROOT / p for p in CODE_PATHS)
        files.update((ROOT / "multimodal").glob("*.py"))
        files.update([CONFIG, ROOT / "configs/multimodal_geo_helpfulness.yaml"])
        files.update((PROTOCOL / "protocol").glob("*"))
        files.update(RF_SELECTION.glob("*"))
        for seed in SEEDS:
            for spec in m2._producer_specs(seed):
                m2.validate_producer(context, spec, seed=seed)
                files.update((PROTOCOL / spec.relative_directory).glob("*"))
        files = sorted(p.resolve() for p in files if p.is_file() and p.suffix != ".pyc")
        if any(not (ROOT / p).is_file() for p in CODE_PATHS):
            raise ValueError("Joint study implementation is incomplete")
        declaration = {
            "schema_version": "geo_helpfulness.joint_tcp_router_declaration.v1",
            "created_at": _now(), "config": config,
            "approved_plan_text": PLAN.read_text(),
            "file_sha256": {str(p.relative_to(ROOT)): _hash(p) for p in files},
            "prior_study_snapshots": {name: _snapshot(BASE / name) for name in PARENT_STUDIES},
            "feature_order": {spec["input_id"]: feature_names(spec) for spec in config["configs"]},
            "environment": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "torch", "scikit-learn")},
            "validation_label_use": "scoring_after_all_64_models_and_predictions_verified",
            "locked_test_access": False,
        }
        _write_json(root / "declaration.json", declaration)
        return {"status": "declared", "source_files": len(files), "path": str(root)}


def validate_declaration(output_dir=OUTPUT, *, check_parents=False):
    root = _output(output_dir)
    declaration = _json(root / "declaration.json")
    if declaration.get("schema_version") != "geo_helpfulness.joint_tcp_router_declaration.v1":
        raise ValueError("Unexpected study declaration")
    if declaration.get("config") != _config() or declaration.get("locked_test_access") is not False:
        raise ValueError("Study scope or configuration changed")
    required = set(CODE_PATHS) | {str(CONFIG.relative_to(ROOT))}
    if not required <= set(declaration["file_sha256"]):
        raise ValueError("Declaration omits implementation files")
    for path, digest in declaration["file_sha256"].items():
        if check_parents or path.endswith(".py") or ROOT / path == CONFIG:
            if _hash(ROOT / path) != digest:
                raise ValueError(f"Declared source changed: {path}")
    if check_parents:
        for name, expected in declaration["prior_study_snapshots"].items():
            if _snapshot(BASE / name) != expected:
                raise ValueError(f"Prior experiment changed: {name}")
    return declaration


def feature_names(spec):
    names = [full_feature_columns()[i] for i in variant_feature_indices(spec["variant"])]
    if spec["with_embeddings"]:
        names += [f"image_embedding_{i:04d}" for i in range(1152)]
        names += [f"geo_embedding_{i:02d}" for i in range(64)]
    if len(names) != spec["input_dim"]:
        raise ValueError("Declared input width disagrees with feature ordering")
    return names


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Seed must be 1, 2, 3, or 4")
    return seed


def _check_paused(root):
    if (root / "pauses").exists() and any((root / "pauses").iterdir()):
        raise StudyPausedError("A preserved failure requires inspection before continuation")


def _producer(root, seed, fold):
    from multimodal.geo_helpfulness_joint_tcp_extract import validate_split_payload
    name = "validation" if fold is None else f"fold_{fold}"
    folder = root / f"producers/seed_{seed}/{name}"
    manifest = _validate_stage(folder, root, role="joint_tcp_producer")
    if manifest["metadata"] != {"seed": seed, "fold": fold}:
        raise ValueError("Producer seed or fold lineage mismatch")
    payload = _load_arrays(folder / "features.npz")
    validate_split_payload(payload, "validation" if fold is None else "train")
    metadata = _json(folder / "extraction.json")
    from multimodal.geo_helpfulness_joint_tcp_extract import _fold_spec
    spec = _fold_spec(seed, fold)
    directory = PROTOCOL / spec.relative_directory
    declaration = _json(root / "declaration.json")
    expected_sources = {**metadata["checkpoint_file_sha256"],
                        "geo_standardization.json": metadata["scaler_file_sha256"],
                        spec.output_filename: metadata["reference_output_file_sha256"]}
    for name, digest in expected_sources.items():
        relative = str((directory / name).relative_to(ROOT))
        if declaration["file_sha256"].get(relative) != digest:
            raise ValueError("Producer used sources different from the declaration")
    if metadata.get("training_seed") != seed or metadata.get("producer_fold") != fold:
        raise ValueError("Extraction metadata identity mismatch")
    return {"payload": payload, "metadata": metadata}


def prepare_seed(seed, output_dir=OUTPUT, *, device="cuda"):
    root, seed = _output(output_dir), _seed(seed)
    validate_declaration(root)
    _check_paused(root)
    from multimodal.geo_helpfulness_joint_tcp_extract import extract_producer, combine_producers
    with _lock(root, f"prepare_{seed}"):
        combined = root / f"features/seed_{seed}"
        if combined.exists():
            _features(root, seed)
            return {"status": "reused_valid", "seed": seed}
        producers = {}
        for fold in (*FOLDS, None):
            name = "validation" if fold is None else f"fold_{fold}"
            folder = root / f"producers/seed_{seed}/{name}"
            if not folder.exists():
                print(f"seed {seed}: replaying frozen producer {name}", flush=True)
                result = extract_producer(seed, fold, device=device)
                staged = _staging(root, f"producer_{seed}_{name}")
                _save_arrays(staged / "features.npz", result["payload"])
                _write_json(staged / "extraction.json", result["metadata"])
                _seal_stage(staged, folder, root, role="joint_tcp_producer", metadata={"seed": seed, "fold": fold})
            producers[fold] = _producer(root, seed, fold)
        staged = _staging(root, f"features_{seed}")
        for split in SPLITS:
            _save_arrays(staged / f"{split}.npz", combine_producers(producers, split))
        _write_json(staged / "producers.json", {
            "producer_manifests": {("validation" if fold is None else f"fold_{fold}"):
                _hash(root / f"producers/seed_{seed}" / ("validation" if fold is None else f"fold_{fold}") / "manifest.json")
                for fold in (*FOLDS, None)},
        })
        _seal_stage(staged, combined, root, role="joint_tcp_features", metadata={"seed": seed})
        _features(root, seed)
        return {"status": "complete", "seed": seed}


def _features(root, seed):
    from multimodal.geo_helpfulness_joint_tcp_extract import validate_split_payload
    folder = root / f"features/seed_{seed}"
    manifest = _validate_stage(folder, root, role="joint_tcp_features")
    if manifest["metadata"] != {"seed": seed}:
        raise ValueError("Feature seed mismatch")
    sources = _json(folder / "producers.json")["producer_manifests"]
    if set(sources) != {"validation", *(f"fold_{fold}" for fold in FOLDS)}:
        raise ValueError("Combined features omit a producer")
    producers = {}
    for name, digest in sources.items():
        if _hash(root / f"producers/seed_{seed}/{name}/manifest.json") != digest:
            raise ValueError("Combined feature producer changed")
        fold = None if name == "validation" else int(name.split("_")[1])
        producers[fold] = _producer(root, seed, fold)["payload"]
    arrays = {split: _load_arrays(folder / f"{split}.npz") for split in SPLITS}
    for split, n, plots in (("train", 3378, 1300), ("validation", 822, 325)):
        data = arrays[split]
        validate_split_payload(data, split)
        if len(data["row_uid"]) != n or len(np.unique(data["plot_idx"])) != plots:
            raise ValueError("Development image/plot counts changed")
        if data["row_uid"].tolist() != sorted(data["row_uid"].tolist()):
            raise ValueError("Features must be canonically ordered")
        parts = [producers[fold] for fold in FOLDS] if split == "train" else [producers[None]]
        order = np.argsort(np.concatenate([part["row_uid"] for part in parts]), kind="stable")
        for key, values in data.items():
            rebuilt = np.concatenate([part[key] for part in parts])[order]
            if not np.array_equal(values, rebuilt):
                raise ValueError(f"Combined {split}/{key} differs from producer cache")
    for name in ("row_uid", "plot_idx"):
        if set(arrays["train"][name]) & set(arrays["validation"][name]):
            raise ValueError("Training and validation identities overlap")
    return arrays


def prepare_inputs(output_dir=OUTPUT):
    root = _output(output_dir)
    validate_declaration(root)
    from multimodal.geo_helpfulness_router_capacity_inputs import load_capacity_inputs
    with _lock(root, "router_inputs"):
        folder = root / "router_inputs"
        if folder.exists():
            _validate_stage(folder, root, role="joint_tcp_router_inputs")
            return {"status": "reused_valid"}
        inputs = load_capacity_inputs(ROOT, PROTOCOL)
        declared = _json(root / "declaration.json")["file_sha256"]
        for path, digest in inputs["parent_file_hashes"].items():
            if declared.get(str(Path(path).relative_to(ROOT))) != digest:
                raise ValueError("Router input parent changed after declaration")
        staged = _staging(root, "router_inputs")
        for seed, data in inputs["seed_inputs"].items():
            saved = {key: data[key] for key in ("X_train", "X_validation", "y_train")}
            for split in SPLITS:
                saved[f"{split}_row_uid"] = data[f"{split}_identity"].row_uid.to_numpy(dtype=str)
                saved[f"{split}_plot_idx"] = data[f"{split}_identity"].plot_idx.to_numpy(dtype=str)
            _save_arrays(staged / f"seed_{seed}.npz", saved)
        _write_json(staged / "columns.json", inputs["feature_columns"])
        _write_json(staged / "parents.json", inputs["parent_file_hashes"])
        _seal_stage(staged, folder, root, role="joint_tcp_router_inputs")
        return {"status": "complete"}


def _inputs(root, seed, arrays):
    from multimodal.geo_helpfulness_confidnet_report import _probabilities
    folder = root / "router_inputs"
    _validate_stage(folder, root, role="joint_tcp_router_inputs")
    data = _load_arrays(folder / f"seed_{seed}.npz")
    if _json(folder / "columns.json") != full_feature_columns():
        raise ValueError("Original router feature order changed")
    for split in SPLITS:
        for name in ("row_uid", "plot_idx"):
            if not np.array_equal(data[f"{split}_{name}"], arrays[split][name]):
                raise ValueError("Router and embedding row identities disagree")
        if data[f"X_{split}"].shape != (len(arrays[split]["row_uid"]), 727):
            raise ValueError("Base feature dimensions changed")
    train = arrays["train"]
    labels = train["labels"]
    correct = {name: train[f"{name}_logits"].argmax(1) == labels for name in ("raw", "geo")}
    states = np.full(len(labels), 3, dtype=np.int64)
    states[~correct["raw"] & correct["geo"]] = 0
    states[correct["raw"] & ~correct["geo"]] = 1
    states[correct["raw"] & correct["geo"]] = 2
    if not np.array_equal(states, data["y_train"]):
        raise ValueError("OOF expert correctness disagrees with saved router supervision")
    data["tcp_targets"] = np.stack([
        _probabilities(train[f"{name}_logits"])[np.arange(len(labels)), labels]
        for name in ("raw", "geo")], axis=1)
    return data


def _matrix(data, arrays, spec, split):
    base = data[f"X_{split}"][:, variant_feature_indices(spec["variant"])]
    blocks = [base]
    if spec["with_embeddings"]:
        blocks += [arrays[split]["image_embedding"], arrays[split]["geo_embedding"]]
    matrix = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)
    if matrix.shape[1] != spec["input_dim"] or not np.isfinite(matrix).all():
        raise ValueError("Invalid joint router feature matrix")
    return matrix


def _model(root, seed, spec, matrices):
    from multimodal.geo_helpfulness_joint_tcp_fit import predict_joint
    folder = root / f"models/seed_{seed}/{spec['config_id']}"
    manifest = _validate_stage(folder, root, role="joint_tcp_model")
    expected = {"seed": seed, "config_id": spec["config_id"],
                "features_manifest": _hash(root / f"features/seed_{seed}/manifest.json"),
                "inputs_manifest": _hash(root / "router_inputs/manifest.json")}
    if manifest["metadata"] != expected:
        raise ValueError("Model feature provenance changed")
    state = _json(folder / "state.json")
    expected_state = {"training_seed": seed, "config_id": spec["config_id"],
                      "input_dim": spec["input_dim"], "fit_row_count": len(matrices["train"]),
                      "auxiliary_weight": spec["auxiliary_weight"], "parameters": _config()["parameters"]}
    if any(state.get(key) != value for key, value in expected_state.items()):
        raise ValueError("Model identity or training recipe differs from declaration")
    if _json(folder / "features.json") != feature_names(spec):
        raise ValueError("Model feature order differs from declaration")
    saved = _load_arrays(folder / "predictions.npz")
    diagnostics = {}
    for split in SPLITS:
        with np.load(root / f"features/seed_{seed}/{split}.npz", allow_pickle=False) as features:
            for key in ("row_uid", "plot_idx", "producer_fold"):
                if not np.array_equal(saved.get(f"{split}_{key}"), features[key]):
                    raise ValueError("Saved prediction identities differ from feature rows")
        native = predict_joint(state, matrices[split], native=True)
        portable = predict_joint(state, matrices[split], native=False)
        diagnostics[split] = {}
        for key in PREDICTION_KEYS:
            values = saved[f"{split}_{key}"]
            if values.dtype != np.float32 or not np.array_equal(values, native[key]):
                raise ValueError(f"Native float32 replay differs for {seed}/{spec['config_id']}/{split}/{key}")
            delta = np.abs(native[key].astype(np.float64) - portable[key])
            diagnostics[split][key] = {"maximum_abs_error": float(delta.max()),
                                       "mean_abs_error": float(delta.mean())}
    if diagnostics != _json(folder / "replay.json"):
        raise ValueError("Float64 replay diagnostics changed")
    history = _json(folder / "history.json")
    if len(history) != 300 or [x["epoch"] for x in history] != list(range(1, 301)):
        raise ValueError("Training budget changed")
    return {split: {key: saved[f"{split}_{key}"] for key in PREDICTION_KEYS} for split in SPLITS}


def fit_seed(seed, output_dir=OUTPUT):
    from multimodal.geo_helpfulness_joint_tcp_fit import fit_joint, predict_joint
    root, seed = _output(output_dir), _seed(seed)
    config = validate_declaration(root)["config"]
    _check_paused(root)
    with _lock(root, f"fit_{seed}"):
        arrays = _features(root, seed)
        data = _inputs(root, seed, arrays)
        for spec in config["configs"]:
            matrices = {split: _matrix(data, arrays, spec, split) for split in SPLITS}
            folder = root / f"models/seed_{seed}/{spec['config_id']}"
            if folder.exists():
                _model(root, seed, spec, matrices)
                continue
            print(f"seed {seed}: fitting {spec['config_id']} (300 epochs)", flush=True)
            try:
                result = fit_joint(matrices["train"], data["y_train"], data["tcp_targets"],
                                   {"validation": matrices["validation"]}, seed=seed,
                                   config_id=spec["config_id"], auxiliary_weight=spec["auxiliary_weight"],
                                   parameters=config["parameters"])
            except Exception as error:
                paused = root / f"pauses/seed_{seed}/{spec['config_id']}"
                paused.mkdir(parents=True, exist_ok=False)
                _write_json(paused / "reason.json", {"reason": str(error), "at": _now()})
                partial = getattr(error, "partial_result", None)
                if partial:
                    for key in ("state", "history", "metadata"):
                        if key in partial:
                            _write_json(paused / f"{key}.json", partial[key])
                raise StudyPausedError(f"Fit failed; saved details at {paused}: {error}") from error
            staged = _staging(root, f"model_{seed}_{spec['config_id']}")
            for key in ("state", "history", "metadata"):
                _write_json(staged / f"{key}.json", result[key])
            _write_json(staged / "features.json", feature_names(spec))
            predictions = {f"{split}_{key}": result["predictions"][split][key]
                           for split in SPLITS for key in PREDICTION_KEYS}
            for split in SPLITS:
                for key in ("row_uid", "plot_idx", "producer_fold"):
                    predictions[f"{split}_{key}"] = arrays[split][key]
            _save_arrays(staged / "predictions.npz", predictions)
            diagnostics = {}
            for split in SPLITS:
                portable = predict_joint(result["state"], matrices[split], native=False)
                diagnostics[split] = {}
                for key in PREDICTION_KEYS:
                    delta = np.abs(result["predictions"][split][key].astype(np.float64) - portable[key])
                    diagnostics[split][key] = {"maximum_abs_error": float(delta.max()),
                                               "mean_abs_error": float(delta.mean())}
            _write_json(staged / "replay.json", diagnostics)
            _seal_stage(staged, folder, root, role="joint_tcp_model", metadata={
                "seed": seed, "config_id": spec["config_id"],
                "features_manifest": _hash(root / f"features/seed_{seed}/manifest.json"),
                "inputs_manifest": _hash(root / "router_inputs/manifest.json"),
            })
            _model(root, seed, spec, matrices)
        return {"status": "complete", "seed": seed, "fits": 16}


def _all_predictions(root):
    config = validate_declaration(root)["config"]
    _check_paused(root)
    frames = []
    parents = {}
    for seed in SEEDS:
        arrays = _features(root, seed)
        data = _inputs(root, seed, arrays)
        for spec in config["configs"]:
            matrices = {split: _matrix(data, arrays, spec, split) for split in SPLITS}
            predictions = _model(root, seed, spec, matrices)
            path = root / f"models/seed_{seed}/{spec['config_id']}/manifest.json"
            parents[str(path.relative_to(root))] = _hash(path)
            for split in SPLITS:
                features, pred = arrays[split], predictions[split]
                frame = pd.DataFrame({name: features[name] for name in ("row_uid", "plot_idx", "producer_fold")})
                frame["training_seed"] = seed
                frame["split"] = split
                for name in ("config_id", "input_id", "variant", "with_embeddings", "auxiliary_weight"):
                    frame[name] = spec[name]
                for name in ("image", "geo", "raw"):
                    frame[f"{name}_logits"] = list(features[f"{name}_logits"])
                for index, state in enumerate(TARGET_ORDER):
                    frame[f"q_{state}"] = pred["q"][:, index]
                frame["fusion_tcp_hat"] = pred["tcp_hat"][:, 0]
                frame["geo_tcp_hat"] = pred["tcp_hat"][:, 1]
                frame["state_logits"] = list(pred["state_logits"])
                frame["tcp_logits"] = list(pred["tcp_logits"])
                frames.append(frame)
    return pd.concat(frames, ignore_index=True), parents


def _scored_rows(root):
    rows, parents = _all_predictions(root)
    verified_at = _now()
    assignment_path = PROTOCOL / "protocol/development_assignments.parquet"
    declared = _json(root / "declaration.json")["file_sha256"]
    if _hash(assignment_path) != declared[str(assignment_path.relative_to(ROOT))]:
        raise ValueError("Development assignments changed before scoring")
    assignments = pd.read_parquet(assignment_path,
                                 columns=["row_uid", "plot_idx", "development_role", "label_id_dense"])
    assignments = assignments.rename(columns={"development_role": "split"})
    result = rows.merge(assignments, on=["row_uid", "plot_idx", "split"], how="left", validate="many_to_one", sort=False)
    if len(result) != len(rows) or result.label_id_dense.isna().any():
        raise ValueError("Scoring label join failed")
    return result, parents, verified_at


def _rf_context():
    """Read only the development selection, never the historical test scores."""
    selection = _json(RF_SELECTION / "selection.json")
    policies = {}
    for key, value in selection["policies"].items():
        if value.get("variant") not in {"ablation_numeric25", "ablation_no_pairs79"}:
            continue
        policies[key] = {name: value[name] for name in ("config_id", "variant", "guardrail_id")}
        policies[key]["threshold"] = {name: value["threshold"][name] for name in (
            "pooled", "per_seed", "mean_seed_delta_accuracy", "threshold_specification", "selected_feasible")}
    return {"source": str((RF_SELECTION / "selection.json").relative_to(ROOT)),
            "source_sha256": _hash(RF_SELECTION / "selection.json"),
            "scope": "frozen_matched_expert_development_RF_context_only",
            "policies": policies}


def score(output_dir=OUTPUT):
    from multimodal.geo_helpfulness_joint_tcp_report import make_report, format_report
    root = _output(output_dir)
    config = validate_declaration(root)["config"]
    with _lock(root, "score"):
        folder = root / "scores"
        if folder.exists():
            _validate_stage(folder, root, role="joint_tcp_scores")
            return {"status": "reused_valid"}
        rows, parents, verified_at = _scored_rows(root)
        result = make_report(rows, config)
        report = result["report"]
        report["all_predictions_verified_before_label_join_at"] = verified_at
        report["rf_context"] = _rf_context()
        staged = _staging(root, "scores")
        _write_json(staged / "report.json", report)
        (staged / "report.md").write_text(format_report(report))
        _write_json(staged / "selections.json", result["selections"])
        _write_json(staged / "bootstrap_policy_order.json", result["bootstrap_policy_order"])
        _save_arrays(staged / "bootstrap_draws.npz", {"draws": result["bootstrap_draws"]})
        result["actions"].to_parquet(staged / "actions.parquet", index=False)
        rows.to_parquet(staged / "scored_rows.parquet", index=False)
        _seal_stage(staged, folder, root, role="joint_tcp_scores", metadata={
            "prediction_manifests": parents, "all_predictions_verified_before_label_join_at": verified_at,
        })
        return {"status": "complete", "path": str(folder)}


def validate(output_dir=OUTPUT):
    from multimodal.geo_helpfulness_joint_tcp_report import make_report, format_report
    root = _output(output_dir)
    config = validate_declaration(root, check_parents=True)["config"]
    rows, parents, _ = _scored_rows(root)
    folder = root / "scores"
    manifest = _validate_stage(folder, root, role="joint_tcp_scores")
    if manifest["metadata"]["prediction_manifests"] != parents:
        raise ValueError("Scoring parents changed")
    pd.testing.assert_frame_equal(pd.read_parquet(folder / "scored_rows.parquet"), rows)
    result = make_report(rows, config)
    saved_report = _json(folder / "report.json")
    result["report"]["all_predictions_verified_before_label_join_at"] = saved_report["all_predictions_verified_before_label_join_at"]
    result["report"]["rf_context"] = _rf_context()
    if result["report"] != saved_report or result["selections"] != _json(folder / "selections.json"):
        raise ValueError("Scored metrics or threshold selections do not reproduce")
    if format_report(saved_report) != (folder / "report.md").read_text():
        raise ValueError("Markdown report differs")
    if result["bootstrap_policy_order"] != _json(folder / "bootstrap_policy_order.json"):
        raise ValueError("Bootstrap policy order differs")
    if not np.array_equal(result["bootstrap_draws"], _load_arrays(folder / "bootstrap_draws.npz")["draws"]):
        raise ValueError("Whole-plot bootstrap does not reproduce")
    pd.testing.assert_frame_equal(result["actions"], pd.read_parquet(folder / "actions.parquet"))
    return {"status": "valid", "fits": 64, "threshold_policies": 144, "locked_test_access": False}

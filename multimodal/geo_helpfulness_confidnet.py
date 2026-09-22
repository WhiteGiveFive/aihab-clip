"""Additive, fixed-recipe ConfidNet experiment on existing development experts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/multimodal_geo_confidnet_initial_v1.json"
PROTOCOL = PROJECT_ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/protocol_v1"
OUTPUT = PROTOCOL.parent / "confidnet_initial_v1"
MODES = ("image", "geo", "raw")
SEEDS = (1, 2, 3, 4)
CODE_PATHS = (
    "multimodal/geo_helpfulness_confidnet.py",
    "multimodal/geo_helpfulness_confidnet_extract.py",
    "multimodal/geo_helpfulness_confidnet_fit.py",
    "multimodal/geo_helpfulness_confidnet_report.py",
    "tools/run_multimodal_geo_helpfulness_confidnet.py",
)


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _relative(path):
    return str(Path(path).relative_to(PROJECT_ROOT))


def _config():
    value = _json(CONFIG)
    if value.get("study_id") != "confidnet_initial_v1":
        raise ValueError("Unexpected ConfidNet study identity")
    if value.get("training_seeds") != list(SEEDS) or value.get("expert_modes") != list(MODES):
        raise ValueError("ConfidNet seeds or expert modes differ from the declared study")
    if value.get("locked_test_access") is not False or value.get("augmented_router_training") is not False:
        raise ValueError("Initial ConfidNet study is development-only without router fitting")
    return value


def _output(path):
    result = Path(path).resolve()
    # A separate sibling prevents experimental writes entering immutable M1–M4.
    if result == PROTOCOL or PROTOCOL in result.parents or result in PROTOCOL.parents:
        raise ValueError("ConfidNet output must be disjoint from the frozen protocol")
    return result


@contextmanager
def _lock(root, name):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f".{name}.lock"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def _source_files():
    files = [CONFIG, *(PROJECT_ROOT / p for p in CODE_PATHS)]
    files.extend(PROJECT_ROOT / p for p in (
        "multimodal/geo_helpfulness_oof.py", "multimodal/models.py",
        "multimodal/geo_helpfulness_protocol.py", "configs/multimodal_geo_helpfulness.yaml"
    ))
    files.extend((PROTOCOL / "protocol" / name) for name in (
        "development_assignments.parquet", "resolved_protocol.yaml", "protocol_manifest.json"
    ))
    for seed in SEEDS:
        files.extend(sorted((PROTOCOL / f"development_validation/seed_{seed}").glob("*")))
    return [path for path in files if path.is_file()]


def declare(output_dir=OUTPUT):
    """Seal recipe, sources and implementation before any confidence fitting."""
    root = _output(output_dir)
    with _lock(root, "declaration"):
        target = root / "declaration.json"
        if target.exists():
            validate_declaration(root, check_parents=True)
            return {"status": "reused_valid", "path": str(target)}
        config = _config()
        for relative in CODE_PATHS:
            if not (PROJECT_ROOT / relative).is_file():
                raise ValueError(f"Implementation is incomplete: {relative}")
        from multimodal.geo_helpfulness_oof import load_frozen_context, _producer_specs, validate_producer
        context = load_frozen_context()
        for seed in SEEDS:
            validate_producer(context, _producer_specs(seed)[-1], seed=seed)
        declaration = {
            "schema_version": "geo_helpfulness.confidnet_declaration.v1",
            "created_at": _now(),
            "config": config,
            "file_sha256": {_relative(p): _hash(p) for p in _source_files()},
            "python_version": platform.python_version(),
            "approval_scope": "Frozen development experts, same-training-set ConfidNet, validation scoring",
            "evidence_status": config["evidence_status"],
            "validation_selection": "none",
            "locked_test_access": False,
        }
        _write_json(target, declaration)
        return {"status": "declared", "path": str(target), "sha256": _hash(target)}


def validate_declaration(output_dir=OUTPUT, *, check_parents=False):
    root = _output(output_dir)
    value = _json(root / "declaration.json")
    if value["config"] != _config():
        raise ValueError("Predeclared ConfidNet configuration changed")
    for relative, expected in value["file_sha256"].items():
        if check_parents or relative in CODE_PATHS or relative == _relative(CONFIG):
            if _hash(PROJECT_ROOT / relative) != expected:
                raise ValueError(f"ConfidNet source changed after declaration: {relative}")
    return value


def _seal_stage(staging, destination, root, *, role, metadata=None):
    files = {str(p.relative_to(staging)): _hash(p) for p in sorted(staging.rglob("*")) if p.is_file()}
    _write_json(staging / "manifest.json", {
        "schema_version": "geo_helpfulness.confidnet_stage.v1",
        "role": role,
        "created_at": _now(),
        "declaration_sha256": _hash(root / "declaration.json"),
        "file_sha256": files,
        "metadata": metadata or {},
    })
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)


def _validate_stage(path, root, *, role):
    manifest = _json(path / "manifest.json")
    if manifest["role"] != role or manifest["declaration_sha256"] != _hash(root / "declaration.json"):
        raise ValueError(f"ConfidNet stage lineage mismatch: {path}")
    observed = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if observed != set(manifest["file_sha256"]) | {"manifest.json"}:
        raise ValueError(f"ConfidNet stage file set changed: {path}")
    for relative, expected in manifest["file_sha256"].items():
        if _hash(path / relative) != expected:
            raise ValueError(f"ConfidNet artifact changed: {path / relative}")
    return manifest


def _staging(root, name):
    directory = root / "staging"
    directory.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{name}_", dir=directory))


def _save_arrays(path, arrays):
    values = {key: np.asarray(value, dtype=str) if key in {"row_uid", "plot_idx"}
              else np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError("Feature caches cannot contain pickled object arrays")
    np.savez_compressed(path, **values)


def _load_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


def _check_arrays(train, validation):
    for role, data, rows in (("train", train, 3378), ("validation", validation, 822)):
        required = {"row_uid", "plot_idx"} | {f"{m}_{kind}" for m in MODES for kind in ("hidden", "logits")}
        if role == "train":
            required.add("labels")
        if set(data) != required:
            raise ValueError(f"Unexpected {role} cache keys: {set(data) ^ required}")
        if data["row_uid"].shape != (rows,) or len(set(data["row_uid"])) != rows:
            raise ValueError(f"Invalid {role} feature identities")
        if data["plot_idx"].shape != (rows,):
            raise ValueError(f"Invalid {role} plot identities")
        for mode in MODES:
            for kind, width in (("hidden", 256), ("logits", 18)):
                matrix = data[f"{mode}_{kind}"]
                if matrix.shape != (rows, width) or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid {role}/{mode}/{kind} features")
    if set(train["plot_idx"]) & set(validation["plot_idx"]):
        raise ValueError("Training and validation plots overlap")
    if train["labels"].shape != (3378,) or train["labels"].dtype.kind not in "iu" or np.any((train["labels"] < 0) | (train["labels"] >= 18)):
        raise ValueError("Invalid training labels")


def _probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = np.exp(values - values.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def run_seed(seed, output_dir=OUTPUT, *, device="cuda"):
    """Extract frozen features, fit three heads, then seal label-blind predictions."""
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError("ConfidNet seed must be 1, 2, 3 or 4")
    root = _output(output_dir)
    declaration = validate_declaration(root)
    with _lock(root, f"seed_{seed}"):
        final = root / f"fits/seed_{seed}"
        if final.exists():
            _validate_stage(final, root, role="confidence_fit_and_predictions")
            return {"seed": seed, "status": "reused_valid"}
        features = root / f"features/seed_{seed}"
        if features.exists():
            _validate_stage(features, root, role="frozen_expert_features")
        else:
            from multimodal.geo_helpfulness_confidnet_extract import extract_seed
            print(f"seed {seed}: extracting frozen training and validation representations", flush=True)
            result = extract_seed(seed, device=device)
            _check_arrays(result["train"], result["validation"])
            staged = _staging(root, f"features_seed_{seed}")
            _save_arrays(staged / "train.npz", result["train"])
            _save_arrays(staged / "validation.npz", result["validation"])
            _write_json(staged / "extraction.json", result["metadata"])
            _seal_stage(staged, features, root, role="frozen_expert_features", metadata={"seed": seed})
        train = _load_arrays(features / "train.npz")
        validation = _load_arrays(features / "validation.npz")
        _check_arrays(train, validation)
        from multimodal.geo_helpfulness_confidnet_fit import fit_confidnet, predict_confidnet
        tables = {role: pd.DataFrame({"row_uid": data["row_uid"], "plot_idx": data["plot_idx"], "training_seed": seed})
                  for role, data in (("train", train), ("validation", validation))}
        staged = _staging(root, f"fits_seed_{seed}")
        fit_metadata = {}
        for mode in MODES:
            print(f"seed {seed}: fitting {mode} confidence head for 100 fixed epochs", flush=True)
            probabilities = _probabilities(train[f"{mode}_logits"])
            target = probabilities[np.arange(len(probabilities)), train["labels"]]
            fit = fit_confidnet(train[f"{mode}_hidden"], target, validation[f"{mode}_hidden"],
                                seed=seed, config=declaration["config"]["parameters"])
            _write_json(staged / f"{mode}_state.json", fit["state"])
            _write_json(staged / f"{mode}_history.json", fit["history"])
            fit_metadata[mode] = fit["metadata"]
            for role, data in (("train", train), ("validation", validation)):
                prediction = np.asarray(fit[f"{role}_prediction"], dtype=np.float64)
                replay = predict_confidnet(_json(staged / f"{mode}_state.json"), data[f"{mode}_hidden"])
                if not np.allclose(prediction, replay, atol=1e-6, rtol=1e-6):
                    raise ValueError("Reloaded ConfidNet predictions do not reproduce fitting output")
                tables[role][f"{mode}_tcp_hat"] = prediction
                tables[role][f"{mode}_logits"] = list(data[f"{mode}_logits"])
        for role, table in tables.items():
            table.to_parquet(staged / f"{role}_predictions.parquet", index=False)
        _write_json(staged / "fit_metadata.json", fit_metadata)
        _seal_stage(staged, final, root, role="confidence_fit_and_predictions", metadata={
            "seed": seed, "features_manifest_sha256": _hash(features / "manifest.json"),
            "validation_labels_used_for_fitting": False,
        })
        return {"seed": seed, "status": "complete", "models": 3}


def _report_markdown(report):
    lines = ["# Initial ConfidNet development experiment", "", report["evidence_status"], "",
             "Experts and confidence heads were fitted on development-train; validation predictions were sealed before scoring.", "",
             "## Exact frozen expert and confidence metrics", "",
             "| Split | Seed | Expert | Errors | Accuracy (%) | TCP MSE | Failure AP: TCP estimate | Failure AP: MCP |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    def number(value, digits=6):
        return "undefined" if value is None else f"{value:.{digits}f}"

    for role, seeds in report["experts"].items():
        for seed, modes in seeds.items():
            for mode, metrics in modes.items():
                lines.append(
                    f"| {role} | {seed} | {mode} | {metrics['top1_error_count']} | "
                    f"{100 * metrics['top1_accuracy']:.4f} | {metrics['tcp_regression']['mse']:.6f} | "
                    f"{number(metrics['failure_detection']['predicted_tcp']['average_precision'])} | "
                    f"{number(metrics['failure_detection']['mcp']['average_precision'])} |"
                )
    lines += ["", "## Native training TCP target distribution", "",
              "| Seed | Expert | Minimum | 1st percentile | Median | TCP >= 0.99 (%) | TCP >= 0.999 (%) |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for seed, modes in report["experts"]["train"].items():
        for mode, metrics in modes.items():
            q = metrics["tcp_quantiles"]
            lines.append(f"| {seed} | {mode} | {q['0.0']:.6f} | {q['0.01']:.6f} | {q['0.5']:.6f} | "
                         f"{100 * metrics['tcp_fraction_ge_0.99']:.4f} | {100 * metrics['tcp_fraction_ge_0.999']:.4f} |")
    lines += ["", "## Validation direct policies (four-seed pooled counts)", "",
              "The same 822 images supply 3,288 seed-image rows; seeds are not independent new images.", "",
              "| Policy | Accuracy (%) | Gain (pp) | Actions | Rescues | Harms | Net | Coverage (%) | Paired 95% gain interval (pp) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, policy in report["pooled_policies"].items():
        interval = report["bootstrap"][name]
        lines.append(f"| {name} | {100 * policy['accuracy']:.4f} | {100 * policy['accuracy_gain_vs_fusion']:+.4f} | "
                     f"{policy['actions']} | {policy['rescues']} | {policy['harms']} | {policy['net']:+d} | "
                     f"{100 * policy['coverage']:.4f} | [{100 * interval['ci95_lower']:+.4f}, {100 * interval['ci95_upper']:+.4f}] |")
    lines += ["", report["bootstrap_scope"], "",
              "Full per-seed metrics, quantiles, failure AP/AUC and TCP regression errors are in `report.json`.", "",
              "No model or threshold was selected on validation. The TCP rule uses strict positive geo-minus-fusion estimated TCP on disagreements.", "",
              "These results do not change the original M4 decision. No cleaned-test examples were accessed.", ""]
    return "\n".join(lines)


def _validate_seed(root, seed):
    """Check paired artifacts and replay all three states without validation labels."""
    from multimodal.geo_helpfulness_confidnet_fit import predict_confidnet, resolve_config
    features, fit = root / f"features/seed_{seed}", root / f"fits/seed_{seed}"
    extraction = _validate_stage(features, root, role="frozen_expert_features")
    manifest = _validate_stage(fit, root, role="confidence_fit_and_predictions")
    if extraction["metadata"].get("seed") != seed or manifest["metadata"].get("seed") != seed:
        raise ValueError("ConfidNet seed identity differs from its artifact directory")
    expected = {"train_predictions.parquet", "validation_predictions.parquet", "fit_metadata.json"}
    expected.update(f"{mode}_{suffix}.json" for mode in MODES for suffix in ("state", "history"))
    if set(manifest["file_sha256"]) != expected:
        raise ValueError("Incomplete confidence states, histories, or predictions")
    if set(extraction["file_sha256"]) != {"train.npz", "validation.npz", "extraction.json"}:
        raise ValueError("Unexpected frozen feature artifact set")
    if manifest["metadata"]["features_manifest_sha256"] != _hash(features / "manifest.json"):
        raise ValueError("Confidence feature lineage changed")
    if _json(features / "extraction.json")["training_seed"] != seed:
        raise ValueError("Frozen expert seed does not match its confidence heads")
    arrays = {role: _load_arrays(features / f"{role}.npz") for role in ("train", "validation")}
    _check_arrays(arrays["train"], arrays["validation"])
    parameters = resolve_config(_config()["parameters"])
    states = {}
    for mode in MODES:
        state = _json(fit / f"{mode}_state.json")
        if (state.get("training_seed") != seed or state.get("fit_row_count") != 3378
                or state.get("input_dim") != 256 or state.get("parameters") != parameters):
            raise ValueError("Confidence state does not match the declared seed/recipe")
        history = _json(fit / f"{mode}_history.json")
        if [item["epoch"] for item in history] != list(range(1, parameters["epochs"] + 1)):
            raise ValueError("Confidence epoch history differs from fixed budget")
        states[mode] = state
    for role, data in arrays.items():
        table = pd.read_parquet(fit / f"{role}_predictions.parquet")
        expected_columns = {"row_uid", "plot_idx", "training_seed"} | {
            f"{mode}_{suffix}" for mode in MODES for suffix in ("tcp_hat", "logits")}
        if set(table) != expected_columns:
            raise ValueError("Confidence prediction columns differ from label-blind schema")
        if not np.array_equal(table.row_uid.to_numpy(), data["row_uid"]):
            raise ValueError("Saved prediction row identities changed")
        if not np.array_equal(table.plot_idx.to_numpy(), data["plot_idx"]):
            raise ValueError("Saved prediction plot identities changed")
        if not (table.training_seed == seed).all():
            raise ValueError("Saved prediction training_seed changed")
        for mode in MODES:
            prediction = predict_confidnet(states[mode], data[f"{mode}_hidden"])
            if not np.allclose(prediction, table[f"{mode}_tcp_hat"], atol=1e-6, rtol=1e-6):
                raise ValueError("Confidence checkpoint replay failed")
            if not np.array_equal(np.stack(table[f"{mode}_logits"]), data[f"{mode}_logits"]):
                raise ValueError("Expert logits changed in confidence predictions")
    return {"seed": seed, "models": 3}


def score(output_dir=OUTPUT):
    """Score only after all twelve confidence states and predictions are sealed."""
    root = _output(output_dir)
    declaration = validate_declaration(root, check_parents=True)
    with _lock(root, "score"):
        destination = root / "scores"
        feature_paths, fit_paths = {}, {}
        for seed in SEEDS:
            feature_paths[seed] = root / f"features/seed_{seed}"
            fit_paths[seed] = root / f"fits/seed_{seed}"
            _validate_seed(root, seed)
        if destination.exists():
            manifest = _validate_stage(destination, root, role="development_scoring")
            for seed in SEEDS:
                if manifest["metadata"]["fit_manifest_sha256"][str(seed)] != _hash(fit_paths[seed] / "manifest.json"):
                    raise ValueError("Saved score parents differ from current confidence states")
            return {"status": "reused_valid", "path": str(destination)}
        sealed_at = _now()
        assignments = pd.read_parquet(PROTOCOL / "protocol/development_assignments.parquet",
                                     columns=["row_uid", "plot_idx", "development_role", "label_id_dense"])
        from multimodal.geo_helpfulness_confidnet_report import score_expert, score_policies, paired_plot_bootstrap_gain
        report = {"schema_version": "geo_helpfulness.confidnet_report.v1",
                  "evidence_status": declaration["config"]["evidence_status"],
                  "all_predictions_verified_before_label_join_at": sealed_at,
                  "experts": {"train": {}, "validation": {}}, "policies": {},
                  "training_rows_per_seed": 3378, "validation_rows_per_seed": 822,
                  "independent_training_plots": 1300, "independent_validation_plots": 325}
        validation_tables = []
        staged = _staging(root, "scores")
        for seed in SEEDS:
            for role in ("train", "validation"):
                table = pd.read_parquet(fit_paths[seed] / f"{role}_predictions.parquet")
                labels = assignments[assignments.development_role == role]
                table = table.merge(labels, on=["row_uid", "plot_idx"], how="left", validate="one_to_one")
                if table.label_id_dense.isna().any() or len(table) != (3378 if role == "train" else 822):
                    raise ValueError("Prediction-to-label identity join failed")
                y = table.label_id_dense.to_numpy(dtype=np.int64)
                report["experts"][role][str(seed)] = {
                    mode: score_expert(y, np.stack(table[f"{mode}_logits"]), table[f"{mode}_tcp_hat"].to_numpy())
                    for mode in MODES}
                if role == "validation":
                    report["policies"][str(seed)] = score_policies(
                        y, np.stack(table.geo_logits), np.stack(table.raw_logits),
                        table.geo_tcp_hat.to_numpy(), table.raw_tcp_hat.to_numpy())
                    validation_tables.append(table)
        joined = pd.concat(validation_tables, ignore_index=True)
        y = joined.label_id_dense.to_numpy(dtype=np.int64)
        args = (y, np.stack(joined.geo_logits), np.stack(joined.raw_logits),
                joined.geo_tcp_hat.to_numpy(), joined.raw_tcp_hat.to_numpy())
        report["pooled_policies"] = score_policies(*args)
        settings = declaration["config"]["bootstrap"]
        report["bootstrap"] = paired_plot_bootstrap_gain(
            *args, joined.plot_idx.to_numpy(), joined.training_seed.to_numpy(),
            n_resamples=settings["resamples"], random_seed=settings["seed"])
        report["bootstrap_scope"] = "Unstratified paired whole-plot percentile intervals; fixed models; repeated seeds retained together; reused validation remains exploratory."
        joined.to_parquet(staged / "validation_scored_rows.parquet", index=False)
        _write_json(staged / "report.json", report)
        (staged / "report.md").write_text(_report_markdown(report))
        _seal_stage(staged, destination, root, role="development_scoring", metadata={
            "fit_manifest_sha256": {str(s): _hash(fit_paths[s] / "manifest.json") for s in SEEDS}})
        return {"status": "complete", "path": str(destination), "policies": report["pooled_policies"]}


def validate(output_dir=OUTPUT):
    """Verify sources, artifact hashes and state-to-prediction replay without refitting."""
    root = _output(output_dir)
    validate_declaration(root, check_parents=True)
    models = 0
    for seed in SEEDS:
        models += _validate_seed(root, seed)["models"]
    scored = root / "scores"
    if scored.exists():
        record = _validate_stage(scored, root, role="development_scoring")
        for seed in SEEDS:
            if record["metadata"]["fit_manifest_sha256"][str(seed)] != _hash(root / f"fits/seed_{seed}/manifest.json"):
                raise ValueError("Score lineage differs from frozen confidence predictions")
    return {"status": "valid", "models": models, "scored": scored.exists()}

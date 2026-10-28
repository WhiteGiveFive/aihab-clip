"""Authenticated, development-only selection and paired M3 comparisons.

The contrast declaration is written before fitting. Analysis accepts only the
complete declared matrix, authenticates saved bundles against the development
inputs, and never selects a seed or an intermediate epoch.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .attention_fusion import GeoPreprocessor
from .attention_fusion_training import (
    _array_hash, _source_hashes, _validated_data, _verify_bundle, classification_metrics,
    configuration_hash, sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("acc1", "acc3", "weighted_f1", "mcc")
IDENTITIES = ("row_uid", "plot_idx", "label")


class ComparisonError(ValueError):
    """The declared comparison or authenticated evidence is incomplete."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ComparisonError(message)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_new_json(path: Path, value: Any) -> None:
    """Atomically publish complete bytes without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link provides an atomic, no-overwrite publish.
        # A crash before publication cannot leave truncated selected.json.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inside(repo: Path, relative: str) -> Path:
    _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
             "Artifact paths must be repository-relative without traversal")
    result = (repo / relative).resolve()
    _require(result.is_relative_to(repo.resolve()), "Artifact path escapes repository")
    return result


def declare_contrasts(config: dict, matrix: dict) -> dict:
    """Enumerate primary and supplementary contrasts without inspecting scores."""
    arms = {(a["candidate_id"], a["geo_preprocessing"]): a["arm_id"]
            for a in matrix["candidate_geo_arms"]}
    contrasts = []

    def add(lhs: tuple, rhs: tuple, kind: str) -> None:
        if lhs not in arms or rhs not in arms:
            return
        left, right = arms[lhs], arms[rhs]
        contrasts.append({"contrast_id": f"{left}__minus__{right}",
                          "lhs_arm_id": left, "rhs_arm_id": right, "kind": kind})

    geo_candidates = [c["id"] for c in config["candidates"] if c["geo"]]
    for geo in config["geo"]["comparison"]:
        for candidate in geo_candidates:
            add((candidate, geo), ("image_only", "not_applicable"), "baseline_image")
            if candidate != "raw_concat":
                add((candidate, geo), ("raw_concat", geo), "baseline_raw")
        for comparator in ("projected_concat", "tokenwise_capacity"):
            kind = ("primary" if geo == config["selection"]["primary_geo_arm"]
                    and comparator == "projected_concat" else "attention_control")
            add(("attention_two_token", geo), (comparator, geo), kind)
        for comparator in ("projected_concat", "tokenwise_capacity", "attention_two_token"):
            add(("attention_cls", geo), (comparator, geo), "readout")
        add(("tokenwise_capacity", geo), ("projected_concat", geo), "capacity")
    for candidate in geo_candidates:
        for geo in config["geo"]["comparison"]:
            if geo != "native":
                add((candidate, geo), (candidate, "native"), "preprocessing")
    _require(len({c["contrast_id"] for c in contrasts}) == len(contrasts), "Duplicate contrast")
    primary = [c["contrast_id"] for c in contrasts if c["kind"] == "primary"]
    _require(len(primary) == 1, "Exactly one primary contrast is required")
    return {"schema_version": "attention_fusion.contrasts.v1", "study_id": config["study_id"],
            "config_sha256": configuration_hash(config), "matrix_sha256": configuration_hash(matrix),
            "primary_contrast_id": primary[0], "contrasts": contrasts,
            "bootstrap": config["metrics"]["bootstrap"],
            "interpretation": config["interpretation"],
            "selection": config["selection"],
            "claim_scope": "descriptive development evidence conditional on selected recipes and reused split",
            "multiple_comparison_adjustment": "none; one predeclared primary contrast, supplementary results descriptive"}


def freeze_contrasts(config: dict, matrix: dict, path: str | Path, *,
                     repo_root: str | Path = REPO_ROOT) -> dict:
    """Create an immutable declaration before any declared development fit."""
    path, repo = Path(path), Path(repo_root).resolve()
    _validate_matrix(config, matrix)
    declaration = declare_contrasts(config, matrix)
    if path.exists():
        saved = _read(path)
        _require(saved.get("declaration") == declaration, "Existing contrast declaration differs")
        _verify_declaration(saved)
        return saved
    _require(not any((_inside(repo, f["output_dir"]) / "run_spec.json").exists()
                     for f in matrix["development_fits"]),
             "Contrasts must be frozen before development fits start")
    value = {"declaration": declaration, "frozen_at": datetime.now(timezone.utc).isoformat(),
             "development_runs_present_at_declaration": False,
             "training_source_sha256": _source_hashes(),
             "input_loader_sha256": sha256_file(Path(__file__).with_name("attention_fusion_inputs.py"))}
    value["manifest_payload_sha256"] = configuration_hash(value)
    _write_new_json(path, value)
    return value


def _verify_declaration(value: dict) -> None:
    _require(value.get("manifest_payload_sha256") == configuration_hash({
        k: v for k, v in value.items() if k != "manifest_payload_sha256"}),
        "Contrast declaration checksum differs")
    _require(value.get("development_runs_present_at_declaration") is False,
             "Contrast declaration was not frozen before fitting")
    _require(value["training_source_sha256"] == _source_hashes(), "Training sources changed after declaration")
    _require(value["input_loader_sha256"] == sha256_file(Path(__file__).with_name("attention_fusion_inputs.py")),
             "Input loader changed after declaration")


def _validate_matrix(config: dict, matrix: dict) -> None:
    arms = matrix["candidate_geo_arms"]
    _require(len({a["arm_id"] for a in arms}) == len(arms), "Duplicate candidate arm")
    expected = {(a["arm_id"], seed, rate) for a in arms
                for seed in config["data"]["training_seeds"]
                for rate in config["training"]["learning_rates"]}
    fits = matrix["development_fits"]
    keys = [(f["arm_id"], f["seed"], f["learning_rate"]) for f in fits]
    _require(len(keys) == len(set(keys)) and set(keys) == expected,
             "Incomplete or duplicate arm/seed/learning-rate accounting")
    _require(len({f["fit_id"] for f in fits}) == len(fits)
             and len({f["output_dir"] for f in fits}) == len(fits), "Duplicate fit ID or destination")
    _require(len(fits) == config["budget"]["development_fits"], "Fit count differs from budget")
    by_arm = {a["arm_id"]: a for a in arms}
    for fit in fits:
        arm = by_arm[fit["arm_id"]]
        _require(fit["candidate_id"] == arm["candidate_id"]
                 and fit["geo_preprocessing"] == arm["geo_preprocessing"]
                 and fit["epochs"] == config["training"]["epochs"]
                 and fit["checkpoint_epoch"] == config["training"]["checkpoint_epoch"]
                 and fit["fit_role"] == "train" and fit["evaluation_role"] == "validation",
                 f"Fit differs from declared arm: {fit['fit_id']}")


def load_reference_inputs(config: dict, *, repo_root: str | Path = REPO_ROOT) -> dict:
    """Load only authenticated, separate development caches and native geo."""
    from .attention_fusion_inputs import (
        load_geo_only_inputs, load_input_provenance, load_prepared_inputs,
    )
    references = {}
    for seed in config["data"]["training_seeds"]:
        regular = {role: load_prepared_inputs(config, seed, role, repo_root=repo_root)
                   for role in ("train", "validation")}
        regular_provenance = load_input_provenance(config, seed, repo_root=repo_root)
        geo_train, geo_validation, geo_provenance = load_geo_only_inputs(config, seed, repo_root=repo_root)
        for name, arrays, provenance in (("image", regular, regular_provenance),
                ("geo_only", {"train": geo_train, "validation": geo_validation}, geo_provenance)):
            # Match execute_fit -> train_run exactly, including canonical dtype
            # conversion before array hashing and its explicit role bindings.
            arrays = {role: _validated_data(values, config, role) for role, values in arrays.items()}
            provenance = dict(provenance)
            provenance.update(protocol_config_sha256=configuration_hash(config),
                              train_role="train", validation_role="validation")
            references[(seed, name)] = {"arrays": arrays, "provenance": provenance,
                "input_hashes": {role: {key: _array_hash(value) for key, value in sorted(data.items())}
                                 for role, data in arrays.items()}}
    return references


def _equal_metrics(actual: dict, expected: dict, context: str) -> None:
    _require(set(actual) == set(expected), f"Metric fields differ: {context}")
    for key, value in expected.items():
        if isinstance(value, float):
            _require(np.isclose(actual[key], value, rtol=1e-12, atol=1e-12), f"Metric {key} differs: {context}")
        else:
            _require(actual[key] == value, f"Metric {key} differs: {context}")


def authenticate_runs(config: dict, matrix: dict, references: dict, *,
                      repo_root: str | Path = REPO_ROOT) -> list[dict]:
    """Authenticate complete runs and recompute final validation metrics.

``references`` comes from ``load_reference_inputs`` in the public study API;
explicit references also allow isolated synthetic contract tests.
    """
    _validate_matrix(config, matrix)
    repo, records, environment = Path(repo_root).resolve(), [], None
    expected_sources = _source_hashes()
    canonical_validation = None
    geo_states = {}
    required_files = {"run_spec.json", "config.json", "run.json", "metrics.json", "history.json",
                      "final.pt", "resume.pt", "geo_preprocessing.json", "train_predictions.npz",
                      "validation_predictions.npz"}
    for fit in matrix["development_fits"]:
        directory = _inside(repo, fit["output_dir"])
        _require("verification" not in directory.parts, "Verification fits cannot enter selection")
        _require((directory / "manifest.json").is_file(), f"Incomplete fit: {fit['fit_id']}")
        manifest = _verify_bundle(directory)
        _require(set(manifest["files"]) == required_files, "Unexpected or missing bundle payload files")
        spec, run = _read(directory / "run_spec.json"), _read(directory / "run.json")
        _require(_read(directory / "config.json") == config, "Saved configuration differs")
        identity = configuration_hash(spec)
        _require(manifest["run_identity_sha256"] == run["run_identity_sha256"] == identity,
                 "Run identity checksum differs")
        expected = {"config_sha256": configuration_hash(config), "study_id": config["study_id"],
                    "regime": config["regime"], "candidate_id": fit["candidate_id"],
                    "geo_mode": fit["geo_preprocessing"], "learning_rate": fit["learning_rate"],
                    "seed": fit["seed"], "fit_role": "train", "verification": False,
                    "epochs": config["training"]["epochs"], "source_hashes": expected_sources}
        _require(all(spec.get(k) == v for k, v in expected.items()), f"Run recipe differs: {fit['fit_id']}")
        _require(manifest["completed_epoch"] == run["completed_epoch"] == spec["epochs"]
                 and run["status"] == "complete" and run["verification"] is False
                 and manifest["verification"] is False, "Incomplete or verification checkpoint")
        _require(manifest["class_map"] == config["data"]["dense_to_canonical_l3"]
                 and manifest["boundary"] == {"test_data_loaded": False, "encoder_executed": False},
                 "Class order or development boundary differs")
        input_kind = "geo_only" if fit["candidate_id"] == "geo_only" else "image"
        ref = references[(fit["seed"], input_kind)]
        _require(spec["input_hashes"] == ref["input_hashes"]
                 and spec["input_provenance"] == ref["provenance"], "Run inputs or encoder provenance differ")
        current_environment = spec["environment"]
        _require(current_environment.get("deterministic_algorithms") is True
                 and current_environment.get("cublas_workspace_config") == config["training"]["cublas_workspace_config"],
                 "Run lacks deterministic runtime evidence")
        if environment is None:
            environment = current_environment
        _require(current_environment == environment, "Candidate environments differ; record and resolve before selection")
        validation = ref["arrays"]["validation"]
        if canonical_validation is None:
            canonical_validation = {key: validation[key] for key in IDENTITIES}
        _require(all(np.array_equal(validation[k], canonical_validation[k]) for k in IDENTITIES),
                 "Validation identities, labels or plot assignments differ across seeds")
        with np.load(directory / "validation_predictions.npz", allow_pickle=False) as saved:
            _require(set(saved.files) == {*IDENTITIES, "logits", "probabilities", "pred_top3", "prediction"},
                     "Unexpected prediction payload")
            predictions = {key: saved[key].copy() for key in saved.files}
        _require(all(np.array_equal(predictions[k], validation[k]) for k in IDENTITIES),
                 "Prediction row/plot/label alignment differs")
        _require(predictions["logits"].shape == (config["data"]["validation_rows"], config["data"]["num_classes"]),
                 "Validation prediction dimensions differ")
        metrics, recomputed = classification_metrics(validation["label"], predictions["logits"])
        _require(all(np.array_equal(predictions[k], value) for k, value in recomputed.items()),
                 "Saved probabilities or predictions do not match logits")
        saved_metrics, history = _read(directory / "metrics.json"), _read(directory / "history.json")
        _require(set(saved_metrics) == {"train", "validation"}, "Unexpected metric roles")
        _equal_metrics(saved_metrics["validation"], metrics, fit["fit_id"])
        _require([h["epoch"] for h in history] == list(range(1, spec["epochs"] + 1)),
                 "Missing or reordered epoch history")
        _require(all(np.isfinite(h["optimization_loss"]) and all(
            np.isfinite(h[role][key]) for role in ("train", "validation") for key in (*METRICS, "loss"))
            for h in history), "Nonfinite training history")
        for role in ("train", "validation"):
            _equal_metrics(history[-1][role], saved_metrics[role], f"final history/{role}")
        # Direct dataframe geo and cached geo have identical values but can
        # retain different array layouts. NumPy's float64 reduction order can
        # then differ by a few ulps: recompute against each actual producer,
        # never reuse another producer's exact fitted statistics.
        state_key = (fit["seed"], input_kind, fit["geo_preprocessing"])
        if state_key not in geo_states:
            mode = "native" if fit["geo_preprocessing"] == "not_applicable" else fit["geo_preprocessing"]
            geo_states[state_key] = GeoPreprocessor(mode, config["data"]["geo_columns"],
                l2_epsilon=config["geo"]["l2_epsilon"], constant_epsilon=config["geo"]["standardize_constant_epsilon"]
                ).fit(ref["arrays"]["train"]["geo_embedding"], ref["arrays"]["train"]["row_uid"]).to_dict()
        _require(_read(directory / "geo_preprocessing.json") == geo_states[state_key],
                 "Geo preprocessing differs from the allowed fitting rows")
        checkpoint = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)
        _require(checkpoint["epoch"] == spec["epochs"] and checkpoint["model_state"], "Wrong final checkpoint epoch")
        weights = checkpoint["model_state"]
        _require(all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in weights.values()),
                 "Nonfinite checkpoint weights")
        parameters = next(c["expected_trainable_parameters"] for c in config["candidates"]
                          if c["id"] == fit["candidate_id"])
        _require(sum(v.numel() for v in weights.values()) == manifest["trainable_parameters"]
                 == run["trainable_parameters"] == parameters, "Checkpoint parameter count differs")
        records.append({**fit, "metrics": metrics, "history": history, "predictions": predictions,
            "manifest_sha256": sha256_file(directory / "manifest.json"), "run_identity_sha256": identity,
            "checkpoint_sha256": manifest["files"]["final.pt"], "parameters": parameters,
            "runtime_seconds": run["elapsed_seconds"], "peak_cuda_bytes": run["peak_cuda_bytes"],
            "peak_process_rss_bytes": run.get("peak_process_rss_bytes"),
            "peak_process_rss_scope": run.get("peak_process_rss_scope"), "environment": environment})
    return records


def select_learning_rates(config: dict, matrix: dict, records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return every LR summary and one winner per arm; all seeds are retained."""
    _validate_matrix(config, matrix)
    expected = {f["fit_id"] for f in matrix["development_fits"]}
    _require(len(records) == len(expected) and {r["fit_id"] for r in records} == expected,
             "Selection requires every declared completed fit")
    summaries, selected = [], []
    for arm in matrix["candidate_geo_arms"]:
        candidates = []
        for rate in config["training"]["learning_rates"]:
            matching = [r for r in records if r["arm_id"] == arm["arm_id"] and r["learning_rate"] == rate]
            _require(sorted(r["seed"] for r in matching) == sorted(config["data"]["training_seeds"]),
                     "Selection requires the complete declared seed set")
            per_seed = {str(r["seed"]): {key: r[key] for key in (
                "fit_id", "metrics", "parameters", "runtime_seconds", "peak_cuda_bytes",
                "peak_process_rss_bytes", "peak_process_rss_scope")} for r in matching}
            summary = {"arm_id": arm["arm_id"], "candidate_id": arm["candidate_id"],
                "geo_mode": arm["geo_preprocessing"], "learning_rate": rate, "per_seed": per_seed,
                "mean": {k: math.fsum(r["metrics"][k] for r in matching) / len(matching) for k in METRICS},
                "sample_std": {k: float(np.std([r["metrics"][k] for r in matching], ddof=1)) for k in METRICS},
                "histories_by_seed": {str(r["seed"]): r["history"] for r in matching}}
            # All seeds score the same row universe. Sum integer correct counts
            # before division so exactly tied accuracies reach the F1/MCC tie
            # breakers regardless of floating-point summation order.
            rows = [r["metrics"]["rows"] for r in matching]
            _require(len(set(rows)) == 1, "Selection requires matched validation row counts")
            for key in ("acc1", "acc3"):
                summary["mean"][key] = sum(round(r["metrics"][key] * rows[0]) for r in matching) / (rows[0] * len(matching))
            candidates.append(summary)
        winner = min(candidates, key=lambda r: (-r["mean"]["acc1"], -r["mean"]["weighted_f1"],
                     -r["mean"]["mcc"], r["learning_rate"], r["candidate_id"]))
        selected.append({**winner, "selected_learning_rate": winner["learning_rate"]})
        summaries.extend(candidates)
    return summaries, selected


def _metrics_from_counts(tp: np.ndarray, predicted: np.ndarray, support: np.ndarray,
                         top3_correct: np.ndarray) -> np.ndarray:
    total = support.sum(axis=-1)
    correct = tp.sum(axis=-1)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom > 0)
    mcc_den = np.maximum((total ** 2 - np.square(predicted).sum(axis=-1))
                         * (total ** 2 - np.square(support).sum(axis=-1)), 0)
    numerator = correct * total - (support * predicted).sum(axis=-1)
    mcc = np.divide(numerator, np.sqrt(mcc_den), out=np.zeros_like(numerator), where=mcc_den > 0)
    return np.stack((correct / total, top3_correct / total, (f1 * support).sum(axis=-1) / total, mcc), axis=-1)


def paired_plot_bootstrap(labels: np.ndarray, plot_ids: np.ndarray, predictions: dict,
                          contrasts: list[dict], *, num_classes: int, replicates: int,
                          seed: int, chunk_size: int = 256) -> tuple[dict, np.ndarray, np.ndarray]:
    """Bootstrap sufficient class counts, sharing each plot draw across methods/seeds.

``predictions[arm_id]`` has shape (seeds, images, top_k). Plot multiplicity
weights every image in a sampled plot. No seed realizations are resampled.
    """
    labels, plots = np.asarray(labels), np.asarray(plot_ids)
    _require(labels.ndim == 1 and labels.shape == plots.shape and len(labels) > 0, "Invalid bootstrap identities")
    _require(replicates > 0 and chunk_size > 0, "Invalid bootstrap budget")
    arm_ids = list(predictions)
    _require(bool(arm_ids), "No bootstrap methods")
    nseeds = np.asarray(predictions[arm_ids[0]]).shape[0]
    unique_plots, inverse = np.unique(plots, return_inverse=True)
    p, c = len(unique_plots), num_classes
    supports = np.zeros((p, c), dtype=np.float64)
    np.add.at(supports, (inverse, labels), 1)
    sufficient = np.zeros((p, len(arm_ids), nseeds, 2 * c + 1), dtype=np.float64)
    for a, name in enumerate(arm_ids):
        values = np.asarray(predictions[name])
        _require(values.ndim == 3 and values.shape[:2] == (nseeds, len(labels))
                 and bool(((values >= 0) & (values < c)).all()), "Bootstrap predictions are not paired")
        for s in range(nseeds):
            hard = values[s, :, 0]
            correct = hard == labels
            np.add.at(sufficient[:, a, s, :c], (inverse[correct], labels[correct]), 1)
            np.add.at(sufficient[:, a, s, c:2*c], (inverse, hard), 1)
            np.add.at(sufficient[:, a, s, -1], inverse, (values[s] == labels[:, None]).any(axis=1))
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, p), dtype=np.int32)
    samples = rng.integers(0, p, size=(replicates, p))
    np.add.at(weights, (np.arange(replicates)[:, None], samples), 1)
    draws = np.empty((replicates, len(contrasts), len(METRICS)), dtype=np.float64)
    lookup = {name: i for i, name in enumerate(arm_ids)}
    flat = sufficient.reshape(p, -1)
    for start in range(0, replicates, chunk_size):
        batch_weights = weights[start:start + chunk_size].astype(np.float64)
        counts = (batch_weights @ flat).reshape(-1, len(arm_ids), nseeds, 2*c + 1)
        support = (batch_weights @ supports)[:, None, None, :]
        metrics = _metrics_from_counts(counts[..., :c], counts[..., c:2*c], support, counts[..., -1])
        total = support.sum(axis=-1)[:, 0, 0]
        for j, contrast in enumerate(contrasts):
            left, right = lookup[contrast["lhs_arm_id"]], lookup[contrast["rhs_arm_id"]]
            differences = (metrics[:, left] - metrics[:, right]).mean(axis=1)
            # Subtract integer counts before division: exactly tied accuracy
            # differences must remain exactly zero for the declared sign rules.
            differences[:, 0] = (counts[:, left, :, :c].sum(axis=-1)
                                  - counts[:, right, :, :c].sum(axis=-1)).mean(axis=1) / total
            differences[:, 1] = (counts[:, left, :, -1] - counts[:, right, :, -1]).mean(axis=1) / total
            draws[start:start + len(batch_weights), j] = differences
    intervals = {item["contrast_id"]: {metric: np.percentile(draws[:, j, m], [2.5, 97.5]).tolist()
                 for m, metric in enumerate(METRICS)} for j, item in enumerate(contrasts)}
    return intervals, draws, weights


def interpretation_category(per_seed: dict, mean: dict, interval95: dict) -> str:
    differences = [values["acc1"] for values in per_seed.values()]
    _require(len(differences) == 4, "The declared interpretation requires four training seeds")
    if (mean["acc1"] > 0 and sum(v > 0 for v in differences) >= 3
            and interval95["acc1"][0] > 0
            and all(mean[k] >= 0 for k in ("acc3", "weighted_f1", "mcc"))):
        return "supported_improvement"
    if mean["acc1"] <= 0 and sum(v <= 0 for v in differences) >= 3:
        return "no_improvement"
    return "inconclusive"


def compare_selected(config: dict, selected: list[dict], records: list[dict], declaration: dict
                     ) -> tuple[list[dict], np.ndarray, np.ndarray]:
    by_fit = {r["fit_id"]: r for r in records}
    arms = {a["arm_id"]: a for a in selected}
    seeds = config["data"]["training_seeds"]
    predictions = {a["arm_id"]: np.stack([by_fit[a["per_seed"][str(s)]["fit_id"]]["predictions"]["pred_top3"]
                                         for s in seeds]) for a in selected}
    reference = records[0]["predictions"]
    settings = config["metrics"]["bootstrap"]
    intervals, draws, weights = paired_plot_bootstrap(reference["label"], reference["plot_idx"], predictions,
        declaration["contrasts"], num_classes=config["data"]["num_classes"],
        replicates=settings["replicates"], seed=settings["seed"])
    results = []
    for item in declaration["contrasts"]:
        left, right = arms[item["lhs_arm_id"]], arms[item["rhs_arm_id"]]
        per_seed, changes = {}, {}
        for i, seed in enumerate(seeds):
            a, b = left["per_seed"][str(seed)]["metrics"], right["per_seed"][str(seed)]["metrics"]
            delta = {key: a[key] - b[key] for key in METRICS}
            lp, rp = predictions[left["arm_id"]][i], predictions[right["arm_id"]][i]
            lc, rc = lp[:, 0] == reference["label"], rp[:, 0] == reference["label"]
            delta["acc1"] = float((int(lc.sum()) - int(rc.sum())) / len(lc))
            delta["acc3"] = float((int((lp == reference["label"][:, None]).any(axis=1).sum())
                                   - int((rp == reference["label"][:, None]).any(axis=1).sum())) / len(lc))
            per_seed[str(seed)] = delta
            rescued, harmed = int((lc & ~rc).sum()), int((~lc & rc).sum())
            changes[str(seed)] = {"rescued": rescued, "harmed": harmed, "net": rescued - harmed}
        mean = {key: math.fsum(d[key] for d in per_seed.values()) / len(per_seed) for key in METRICS}
        mean["acc1"] = sum(d["net"] for d in changes.values()) / (len(reference["label"]) * len(seeds))
        mean["acc3"] = sum(round(d["acc3"] * len(reference["label"])) for d in per_seed.values()) / (len(reference["label"]) * len(seeds))
        result = {**item, "per_seed": per_seed, "mean_difference": mean,
            "sample_std_difference": {k: float(np.std([d[k] for d in per_seed.values()], ddof=1)) for k in METRICS},
            "interval95": intervals[item["contrast_id"]], "rescues_harms_by_seed": changes,
            "mean_rescues_harms": {k: float(np.mean([d[k] for d in changes.values()])) for k in ("rescued", "harmed", "net")}}
        result["category"] = interpretation_category(per_seed, mean, result["interval95"])
        results.append(result)
    return results, draws, weights


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selected_references(config: dict, matrix: dict, selected: list[dict], records: list[dict],
                         declaration_hash: str) -> list[dict]:
    by_fit = {r["fit_id"]: r for r in records}
    planned = {(r["arm_id"], r["seed"]): r for r in matrix["selected_train_checkpoint_references"]}
    references = []
    for arm in selected:
        arm["checkpoint_refs"] = []
        for seed in config["data"]["training_seeds"]:
            fit = by_fit[arm["per_seed"][str(seed)]["fit_id"]]
            reference = {"study_id": config["study_id"], "arm_id": arm["arm_id"], "seed": seed,
                "selected_learning_rate": arm["selected_learning_rate"], "fit_id": fit["fit_id"],
                "checkpoint_epoch": config["training"]["checkpoint_epoch"], "retrain": False,
                "bundle_path": fit["output_dir"], "bundle_manifest_sha256": fit["manifest_sha256"],
                "checkpoint_sha256": fit["checkpoint_sha256"], "run_identity_sha256": fit["run_identity_sha256"],
                "config_sha256": configuration_hash(config), "contrast_manifest_sha256": declaration_hash,
                "selection_path": str(Path(planned[(arm["arm_id"], seed)]["output_dir"]) / "selected.json")}
            references.append(reference)
            arm["checkpoint_refs"].append(reference)
    _require(len(references) == config["budget"]["selected_train_checkpoints"], "Selected-reference count differs")
    return references


def analyze_study(config: dict, matrix: dict, contrast_manifest_path: str | Path,
                  output_dir: str | Path, *, repo_root: str | Path = REPO_ROOT) -> dict:
    """Authenticate, select and write immutable M3 analysis; no fitting occurs."""
    from tools.run_multimodal_attention_fusion import read_frozen_plan

    repo, output = Path(repo_root).resolve(), Path(output_dir).resolve()
    _require(not output.exists(), "Completed/partial analysis exists; retain it and use a new destination")
    _require(output.is_relative_to(repo / config["paths"]["artifact_root"]), "Analysis must stay in the study directory")
    frozen, frozen_matrix, protocol = read_frozen_plan(
        Path(config["paths"]["artifact_root"]) / "protocol/resolved_protocol.yaml", repo)
    _require(config == frozen and matrix == frozen_matrix, "Analysis differs from frozen protocol")
    contrast_path = Path(contrast_manifest_path)
    saved = _read(contrast_path)
    _verify_declaration(saved)
    declaration = declare_contrasts(config, matrix)
    _require(saved["declaration"] == declaration, "Contrast declaration differs from the current study")
    references = load_reference_inputs(config, repo_root=repo)
    records = authenticate_runs(config, matrix, references, repo_root=repo)
    summaries, selected = select_learning_rates(config, matrix, records)
    contrasts, draws, weights = compare_selected(config, selected, records, declaration)
    selected_refs = _selected_references(config, matrix, selected, records, sha256_file(contrast_path))
    reference = records[0]["predictions"]
    labels, plots = reference["label"], reference["plot_idx"]
    audit = _read(repo / config["paths"]["artifact_root"] / "protocol/input_audit.json")
    ontology = sorted(audit["assignments"]["class_map"], key=lambda row: row["label_id_dense"])
    _require([r["canonical_l3_id"] for r in ontology] == config["data"]["dense_to_canonical_l3"], "Class names misaligned")
    claims = {}
    for geo in config["geo"]["comparison"]:
        requirements = [c for c in contrasts if c["lhs_arm_id"] == f"attention_two_token__geo_{geo}"
                        and c["rhs_arm_id"] in {f"{name}__geo_{geo}" for name in (
                            "projected_concat", "raw_concat", "tokenwise_capacity")}]
        claims[geo] = {"supported": len(requirements) == 3 and all(c["category"] == "supported_improvement" for c in requirements),
                       "required_contrasts": [c["contrast_id"] for c in requirements]}
    analysis = {"schema_version": "attention_fusion.comparison.v1", "study_id": config["study_id"],
        "class_order": config["data"]["dense_to_canonical_l3"], "class_labels": [r["label_name"] for r in ontology],
        "training_seeds": config["data"]["training_seeds"], "metrics_order": list(METRICS),
        "metric_units": {"acc1": "fraction", "acc3": "fraction", "weighted_f1": "unit_interval", "mcc": "coefficient"},
        "validation_rows": len(labels), "validation_plots": len(set(plots)),
        "validation_class_support_images": np.bincount(labels, minlength=config["data"]["num_classes"]).tolist(),
        "validation_class_support_plots": [len(set(plots[labels == c])) for c in range(config["data"]["num_classes"])],
        "counts": {"development_fits": len(records), "selected_arms": len(selected),
                   "selected_checkpoints": len(selected_refs), "contrasts": len(contrasts)},
        "primary_contrast_id": declaration["primary_contrast_id"], "lr_candidates": summaries,
        "selected_arms": selected, "contrasts": contrasts, "attention_claims": claims,
        "bootstrap": {**config["metrics"]["bootstrap"], "plot_order": sorted(set(plots.tolist())),
                      "draw_axis_order": ["replicate", "contrast", "metric"],
                      "contrast_order": [c["contrast_id"] for c in contrasts], "metric_order": list(METRICS)},
        "provenance": {"config_sha256": configuration_hash(config), "matrix_sha256": configuration_hash(matrix),
            "protocol_manifest_payload_sha256": protocol["manifest_payload_sha256"],
            "contrast_manifest_sha256": sha256_file(contrast_path),
            "analysis_source_sha256": sha256_file(Path(__file__)), "training_source_sha256": _source_hashes(),
            "numpy": np.__version__, "environment": records[0]["environment"],
            "fit_manifests": {r["fit_id"]: r["manifest_sha256"] for r in records},
            "test_features_or_labels_read": False, "all_candidate_bundles_authenticated": True},
        "limitations": [declaration["claim_scope"], "No practical-effect threshold was declared.",
                        "Seed realizations share a fixed validation split; they are not independent datasets.",
                        "Bootstrap intervals do not include recipe-selection or new-split uncertainty."]}
    selection = {"study_id": config["study_id"], "selection_rule": config["selection"],
                 "all_declared_fits_complete": True, "selected_references": selected_refs,
                 "config_sha256": configuration_hash(config), "contrast_manifest_sha256": sha256_file(contrast_path)}
    selection["selection_payload_sha256"] = configuration_hash(selection)
    # Validate every destination before publishing any immutable reference.
    for ref in selected_refs:
        path = _inside(repo, ref["selection_path"])
        _require(not path.exists() or _read(path) == ref, "Existing selected-checkpoint reference differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write(staging / "analysis.json", analysis)
        _write(staging / "selection.json", selection)
        _write(staging / "contrasts.json", saved)
        np.savez_compressed(staging / "bootstrap_replicates.npz", differences=draws, plot_multiplicities=weights)
        tables = staging / "tables"
        tables.mkdir()
        _csv(tables / "all_fits.csv", [{"fit_id": r["fit_id"], "arm_id": r["arm_id"],
             "learning_rate": r["learning_rate"], "seed": r["seed"], **{k: r["metrics"][k] for k in METRICS},
             "parameters": r["parameters"], "runtime_seconds": r["runtime_seconds"],
             "peak_cuda_bytes": r["peak_cuda_bytes"], "bundle_manifest_sha256": r["manifest_sha256"]} for r in records])
        _csv(tables / "learning_rates.csv", [{"arm_id": a["arm_id"], "learning_rate": a["learning_rate"],
             **{f"mean_{k}": a["mean"][k] for k in METRICS}, **{f"sample_std_{k}": a["sample_std"][k] for k in METRICS}}
             for a in summaries])
        _csv(tables / "selected_arms.csv", [{"arm_id": a["arm_id"], "learning_rate": a["learning_rate"],
             **{f"mean_{k}": a["mean"][k] for k in METRICS}, **{f"sample_std_{k}": a["sample_std"][k] for k in METRICS}}
             for a in selected])
        _csv(tables / "paired_contrasts.csv", [{"contrast_id": c["contrast_id"], "kind": c["kind"], "metric": k,
             "mean_difference": c["mean_difference"][k], "lower_95": c["interval95"][k][0],
             "upper_95": c["interval95"][k][1], "category": c["category"]} for c in contrasts for k in METRICS])
        manifest = {"status": "complete", "study_id": config["study_id"], "files": {
            str(p.relative_to(staging)): sha256_file(p) for p in sorted(staging.rglob("*")) if p.is_file()},
            "counts": analysis["counts"], "provenance": analysis["provenance"]}
        manifest["manifest_payload_sha256"] = configuration_hash(manifest)
        _write(staging / "manifest.json", manifest)
        # Publish the final analysis directory last. If reference publication
        # is interrupted, an identical retry can reuse the immutable subset.
        for ref in selected_refs:
            path = _inside(repo, ref["selection_path"])
            if not path.exists():
                _write_new_json(path, ref)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis

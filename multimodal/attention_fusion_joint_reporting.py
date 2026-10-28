"""Live M3J curves and development-only, checkpoint-policy-separated reports.

Reporting reads saved metrics and predictions; it never opens image datasets or
model checkpoints. Single-seed comparisons are descriptive and provisional.
"""

from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .attention_fusion_reporting import METRICS, METRIC_LABELS, STYLE, draw_cm
from .attention_fusion_training import classification_metrics


MODEL_LABELS = {
    "image_only": "Image only", "geo_only": "Geo only",
    "raw_concat": "Raw concatenation", "projected_concat": "Projected concatenation",
    "tokenwise_capacity": "Token-wise capacity", "attention_two_token": "Two-token attention",
    "attention_cls": "CLS attention", "attention_cls_skip_adapted": "CLS + adapted bypass",
    "attention_cls_skip_original": "CLS + original bypass",
}
POLICIES = ("final", "best")
PROVISIONAL_ASSURANCE = "provisional_unresolved_memory_integrity"
PROVISIONAL_TITLE = "PROVISIONAL — unresolved memory integrity"
PROVISIONAL_NOTICE = (
    "Earlier memory/read inconsistencies remain unresolved. These results are provisional; "
    "the training checks do not establish machine health or bit-for-bit reproducibility."
)
DIRECT_VERIFICATION_POLICY = "direct_freeze_checks_v1"


def _is_exploratory(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    if (metadata.get("result_assurance") == PROVISIONAL_ASSURANCE
            or metadata.get("numerical_policy", {}).get("mode") == "exploratory"):
        return True
    return any(_is_exploratory(metadata.get(key)) for key in ("config", "spec", "run"))


def _directory_metadata(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for name in ("run.json", "run_spec.json", "config.json")
            if (path := directory / name).is_file()]


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    if not isinstance(metadata, dict) or not metadata:
        return None
    if key in metadata:
        return metadata[key]
    for nested in ("config", "spec", "run"):
        value = _metadata_value(metadata.get(nested, {}), key)
        if value is not None:
            return value
    return None


def _assurance(metadata: list[dict[str, Any]]) -> str:
    if any(_is_exploratory(item) for item in metadata):
        return PROVISIONAL_ASSURANCE
    return next((item["result_assurance"] for item in metadata if item.get("result_assurance")),
                "strict_checks_required")


def _loader_rng_metadata(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Interpret recorded loader configuration without inferring it from seed alone."""
    def value(key: str) -> Any:
        return next((found for item in metadata
                     if (found := _metadata_value(item, key)) is not None), None)

    training = value("training")
    study_id = value("study_id")
    geo_only = value("candidate_id") == "geo_only" or value("regime") == "geo_only"
    if not isinstance(training, dict):
        return {"policy": "unknown", "shuffle_namespace": None,
                "augmentation_namespace": None}
    policy = training.get("loader_rng_policy", "legacy_study_id_v1")
    if policy == "declared_namespaces_v1":
        namespaces = {name: training.get(f"{name}_seed_namespace")
                      for name in ("shuffle", "augmentation")}
    elif policy == "legacy_study_id_v1":
        # Historical image loaders ignored the shared namespace declarations.
        namespaces = {name: study_id for name in ("shuffle", "augmentation")}
        if geo_only:
            namespaces["shuffle"] = training.get("shuffle_seed_namespace", study_id)
    else:
        namespaces = {}
    if geo_only:
        namespaces["augmentation"] = None
    return {"policy": policy,
            **{f"{name}_namespace": (namespaces.get(name)
                                    if isinstance(namespaces.get(name), str)
                                    and namespaces[name] else None)
               for name in ("shuffle", "augmentation")}}


def _loader_rng_pair(left: dict[str, Any] | None,
                     right: dict[str, Any] | None) -> dict[str, Any]:
    values = {side: (run or {}).get("loader_rng", {})
              for side, run in (("lhs", left), ("rhs", right))}
    fields = ("shuffle_namespace", "augmentation_namespace")
    known = all(values[side].get(field) for side in values for field in fields)
    status = ("matched_namespaces" if all(values["lhs"][field] == values["rhs"][field]
                                         for field in fields) else "different_namespaces") if known else "unknown"
    return {"loader_rng_status": status,
            **{f"{side}_loader_rng_{field}": values[side].get(field)
               for side in values for field in ("policy", *fields)}}


def _loader_rng_interpretation(pairing: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {status: sum(item["loader_rng_status"] == status for item in pairing)
              for status in ("matched_namespaces", "different_namespaces", "unknown")}
    present = [status for status, count in counts.items() if count]
    status = present[0] if len(present) == 1 else "mixed" if present else "unknown"
    messages = []
    if counts["matched_namespaces"]:
        messages.append(f"{counts['matched_namespaces']} encoder-regime pairs have matching shuffle and "
                        "augmentation namespaces in their saved loader configurations. This records matched "
                        "RNG configuration, not bit-for-bit execution equivalence.")
    if counts["different_namespaces"]:
        messages.append(f"{counts['different_namespaces']} encoder-regime pairs have different shuffle or "
                        "augmentation namespaces in their saved loader configurations. These comparisons are "
                        "not a strict trainability-only contrast: training-order or image-crop randomness also differs.")
    if counts["unknown"] or not pairing:
        messages.append("Loader RNG provenance is missing or unsupported for some encoder-regime pairs; "
                        "matched training order and image crops cannot be established for those pairs.")
    return {"status": status, "pair_counts": counts, "interpretation": " ".join(messages)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _atomic(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _text(path: Path, value: str) -> None:
    _atomic(path, lambda temporary: temporary.write_text(value))


def _json(path: Path, value: Any) -> None:
    _text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if not rows and columns is None:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _text(path, stream.getvalue())


def _save(fig: Any, stem: Path, pdf: bool = True) -> list[str]:
    paths = []
    for extension in (("png", "pdf") if pdf else ("png",)):
        path = stem.with_suffix(f".{extension}")
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        _atomic(path, lambda temporary: fig.savefig(
            temporary, dpi=130, bbox_inches="tight", facecolor="white", metadata=metadata))
        paths.append(str(path))
    return paths


def _validate_history(history: list[dict[str, Any]]) -> None:
    _require(bool(history), "Cannot plot or select from an empty history")
    _require([row["epoch"] for row in history] == list(range(1, len(history) + 1)),
             "History must contain contiguous completed epochs starting at 1")
    for row in history:
        for role in ("train", "validation"):
            for metric in ("loss", *METRICS):
                _require(np.isfinite(row[role][metric]), f"Nonfinite {role}/{metric} history")
        _require(np.isfinite(row["train"].get("optimization_loss", row["train"]["loss"])),
                 "Nonfinite training optimization loss")
        for key in ("learning_rate", "next_learning_rate", "train_seconds",
                    "validation_seconds", "epoch_seconds", "peak_cuda_bytes", "optimizer_steps"):
            _require(np.isfinite(row[key]) and row[key] >= 0, f"Invalid history field {key}")
    steps = [row["optimizer_steps"] for row in history]
    _require(all(right > left for left, right in zip([0, *steps[:-1]], steps)),
             "Optimizer-step history must increase after every epoch")


def best_epoch(history: list[dict[str, Any]]) -> int:
    """Return one eligible checkpoint using the declared unrounded tie-breaks."""
    _validate_history(history)
    row = max(history, key=lambda item: (
        item["validation"]["acc1"], item["validation"]["weighted_f1"],
        item["validation"]["mcc"], -item["epoch"]))
    return int(row["epoch"])


def history_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten plotted quantities for an auditable, lossless numeric readout."""
    _validate_history(history)
    rows = []
    for item in history:
        row = {key: item[key] for key in (
            "epoch", "optimizer_steps", "learning_rate", "next_learning_rate",
            "train_seconds", "validation_seconds", "epoch_seconds", "peak_cuda_bytes")}
        for role in ("train", "validation"):
            row.update({f"{role}_{metric}": item[role][metric] for metric in ("loss", *METRICS)})
        row["train_optimization_loss"] = item["train"].get("optimization_loss", item["train"]["loss"])
        rows.append(row)
    return rows


def plot_history(history: list[dict[str, Any]], output_dir: str | Path,
                 title: str = "M3J training", *, save_pdf: bool = True,
                 metadata: dict[str, Any] | None = None) -> list[str]:
    """Refresh metrics, resources and a local auto-refreshing monitoring page.

    ``save_pdf=False`` avoids re-encoding a PDF on every epoch. Call with the
    default once at completion to keep a publication/export format as well.
    """
    rows = history_rows(history)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_sources = [metadata or {}, *_directory_metadata(output), *_directory_metadata(output.parent)]
    assurance = _assurance(metadata_sources)
    geo_only = any(_metadata_value(item, "candidate_id") == "geo_only" for item in metadata_sources)
    training_label = "Training pass (dropout)" if geo_only else "Training pass (augmented)"
    training_conditions = "dropout" if geo_only else "augmentation/dropout"
    exploratory = assurance == PROVISIONAL_ASSURANCE
    display_title = f"{title}\n{PROVISIONAL_TITLE}" if exploratory else title
    if exploratory:
        for row in rows:
            row["result_assurance"] = assurance
    epochs = [row["epoch"] for row in rows]
    selected = best_epoch(history)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        axes = axes.ravel()
        try:
            for axis, metric in zip(axes[:5], ("loss", *METRICS)):
                scale = 100 if metric in {"acc1", "acc3", "weighted_f1"} else 1
                train_key = "train_optimization_loss" if metric == "loss" else f"train_{metric}"
                axis.plot(epochs, [row[train_key] * scale for row in rows],
                          color="#0072b2", label=training_label)
                axis.plot(epochs, [row[f"validation_{metric}"] * scale for row in rows],
                          color="#d55e00", label="Validation (fixed checkpoint)")
                axis.set_title("Cross-entropy (training optimization loss)" if metric == "loss" else METRIC_LABELS[metric])
                axis.axvline(selected, color="0.5", ls=":", lw=1)
                lower, upper = axis.get_ylim()
                if metric in {"acc1", "acc3", "weighted_f1"}:
                    axis.set_ylim(max(0, lower), min(100, upper))
                elif metric == "mcc":
                    axis.set_ylim(max(-1, lower), min(1, upper))
            axes[0].legend(fontsize=8)
            axes[5].plot(epochs, [row["learning_rate"] for row in rows], label="Used this epoch")
            axes[5].plot(epochs, [row["next_learning_rate"] for row in rows], ls="--", label="After scheduler step")
            axes[5].set_title("Shared trainable-module learning rate")
            axes[5].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axes[5].legend(fontsize=8)
            for key, label in (("train_seconds", "Training"), ("validation_seconds", "Validation"),
                               ("epoch_seconds", "Compute + validation")):
                axes[6].plot(epochs, [row[key] for row in rows], label=label)
            axes[6].set_title("Compute duration (seconds; excludes checkpoint I/O)")
            axes[6].legend(fontsize=8)
            axes[7].plot(epochs, [row["peak_cuda_bytes"] / 2**30 for row in rows])
            axes[7].set_title("Peak allocated CUDA memory (GiB)")
            axes[8].plot(epochs, [row["optimizer_steps"] for row in rows])
            axes[8].set_title("Cumulative optimizer updates")
            for axis in axes:
                axis.set_xlabel("Completed epoch")
                axis.set_xlim(0.5, max(1.5, epochs[-1] + 0.5))
                axis.grid(alpha=0.15)
            fig.suptitle(f"{display_title}\nCompleted epoch {epochs[-1]}; best validation checkpoint: epoch {selected}")
            fig.text(0.5, 0.005,
                     f"Training uses {training_conditions} and changing weights. Validation uses evaluation mode and fixed weights.\n"
                     "Dotted markers show the single best-validation epoch for all metrics; they do not trigger early stopping.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.045, 1, 0.92 if exploratory else 0.94))
            paths = _save(fig, output / "learning_curves", pdf=save_pdf)
        finally:
            plt.close(fig)
    _csv(output / "history.csv", rows)
    warning = f"<p><strong>{html.escape(PROVISIONAL_NOTICE)}</strong></p>" if exploratory else ""
    _text(output / "monitoring.html", "<!doctype html><html><head><meta charset='utf-8'>"
          "<meta http-equiv='refresh' content='30'><title>" + html.escape(display_title) +
          "</title></head><body><h1>" + html.escape(display_title) + "</h1>" + warning +
          f"<p>Completed epoch {epochs[-1]}; best-validation epoch {selected}. Refreshes every 30 seconds.</p>"
          "<p>Training-pass and fixed-checkpoint validation metrics have different evaluation conditions.</p>"
          "<p><a href='history.csv'>Numeric history</a></p>"
          "<img src='learning_curves.png' alt='Training and validation curves, learning rate and resource use' "
          "style='max-width:100%;height:auto'></body></html>\n")
    return paths + [str(output / "history.csv"), str(output / "monitoring.html")]


def _predictions(path: Path, metrics: dict[str, Any], *, legacy_metric_check: bool = True) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("row_uid", "label", "plot_idx", "prediction", "logits")}
    count = len(arrays["label"])
    _require(all(arrays[key].shape == (count,) for key in ("row_uid", "label", "plot_idx", "prediction")),
             f"Prediction identity shape differs in {path}")
    _require(len(np.unique(arrays["row_uid"])) == count, f"Duplicate prediction identities in {path}")
    _require(arrays["logits"].shape == (count, 18), f"Expected the fixed 18 class axes in {path}")
    _require(np.isfinite(arrays["logits"]).all(), f"Nonfinite prediction logits in {path}")
    rebuilt, values = classification_metrics(arrays["label"], arrays["logits"])
    _require(np.array_equal(values["prediction"], arrays["prediction"]), f"Prediction/logit disagreement in {path}")
    _require(np.array_equal(rebuilt["confusion_matrix"], metrics["confusion_matrix"]),
             f"Confusion/prediction disagreement in {path}")
    for metric in ("loss", *METRICS):
        _require(np.isfinite(metrics[metric]), f"Nonfinite reported {metric} in {path}")
        if legacy_metric_check:
            _require(np.isclose(rebuilt[metric], metrics[metric], rtol=0, atol=1e-7),
                     f"Reported {metric} disagrees with predictions in {path}")
    return arrays


def _cohort_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    for key in ("row_uid", "label", "plot_idx"):
        _require(np.array_equal(left[key], right[key]), f"Paired validation {key} are not aligned")


def _paired_difference(left: dict[str, Any], right: dict[str, Any], policy: str,
                       kind: str) -> dict[str, Any]:
    lhs, rhs = left["spec"], right["spec"]
    _require(lhs["class_labels"] == rhs["class_labels"], "Paired class-label order differs")
    lp, rp = left["policies"][policy], right["policies"][policy]
    _cohort_equal(lp["predictions"], rp["predictions"])
    lc = lp["predictions"]["prediction"] == lp["predictions"]["label"]
    rc = rp["predictions"]["prediction"] == rp["predictions"]["label"]
    rescued, harmed = int((lc & ~rc).sum()), int((~lc & rc).sum())
    return {"policy": policy, "kind": kind, "seed": lhs["seed"], "regime": lhs["regime"],
            "lhs_run_id": lhs["run_id"], "rhs_run_id": rhs["run_id"],
            "lhs_epoch": lp["epoch"], "rhs_epoch": rp["epoch"],
            **{f"difference_{metric}": lp["metrics"][metric] - rp["metrics"][metric] for metric in METRICS},
            "rescued": rescued, "harmed": harmed, "net_correct": rescued - harmed,
            "interpretation": "single_seed_descriptive",
            "result_assurance": (PROVISIONAL_ASSURANCE if PROVISIONAL_ASSURANCE in
                                 (left["result_assurance"], right["result_assurance"])
                                 else left["result_assurance"])}


def _read_run(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    _require(all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) and part not in {".", ".."}
                 for part in spec["run_id"].split("/")), f"Unsafe run identifier in {path}")
    _require(spec["candidate_id"] in MODEL_LABELS, f"Unknown candidate in {path}")
    _require(len(spec["class_labels"]) == 18, f"Expected 18 ordered class labels in {path}")
    directory = path.parent
    history_path = directory / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if history:
        _validate_history(history)
    result = {"spec": spec, "directory": directory, "history": history, "policies": {}}
    metadata = _directory_metadata(directory)
    result["loader_rng"] = _loader_rng_metadata(metadata)
    result["result_assurance"] = _assurance(metadata)
    result["verification_policy"] = next(
        (value for item in metadata if (value := _metadata_value(item, "verification_policy"))), None)
    result["legacy_numerical_policy"] = any(
        (_metadata_value(item, "numerical_policy") or {}).get("mode") == "exploratory"
        for item in metadata)
    audit_path = directory / "numerical_audit.json"
    result["numerical_audit"] = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    metrics_path = directory / "metrics.json"
    if (spec.get("status") != "complete" or len(history) != spec["expected_epochs"]
            or not metrics_path.is_file()):
        return result
    metrics = json.loads(metrics_path.read_text())
    exploratory = result["result_assurance"] == PROVISIONAL_ASSURANCE
    direct_checks = result["verification_policy"] == DIRECT_VERIFICATION_POLICY
    expected = {"final": len(history), "best": best_epoch(history)}
    for policy in POLICIES:
        item = metrics[policy]
        _require(item["epoch"] == expected[policy], f"Incorrect {policy} checkpoint epoch in {metrics_path}")
        epoch_metrics = history[item["epoch"] - 1]["validation"]
        # The active policy evaluates the reloaded checkpoint for reported
        # scores. Original epoch scores select its identity, not replay equality.
        for metric in (() if direct_checks else ("loss", *METRICS)):
            if exploratory:
                # Replayed predictions can have audited loss drift. Class-derived
                # metrics still identify the same selected epoch exactly.
                agrees = (np.isclose(item["metrics"][metric], epoch_metrics[metric],
                                     rtol=1e-5, atol=1e-6)
                          if metric == "loss" else item["metrics"][metric] == epoch_metrics[metric])
            else:
                agrees = np.isclose(item["metrics"][metric], epoch_metrics[metric], rtol=0, atol=1e-7)
            _require(agrees,
                     f"{policy} metrics do not match selected epoch in {metrics_path}")
        arrays = _predictions(directory / f"{policy}_predictions.npz", item["metrics"],
                              legacy_metric_check=not direct_checks)
        result["policies"][policy] = {
            **item, "predictions": arrays,
            "metrics_source": ("reloaded_checkpoint_evaluation" if direct_checks
                               else "legacy_checkpoint_replay"),
            "selection_metrics": epoch_metrics,
        }
    _cohort_equal(result["policies"]["final"]["predictions"], result["policies"]["best"]["predictions"])
    return result


def _contrasts(runs: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    lookup = {(run["spec"]["seed"], run["spec"]["regime"], run["spec"]["initialization"],
               run["spec"]["candidate_id"]): run for run in runs}
    result, seen = [], set()
    geo_by_seed = {run["spec"]["seed"]: run for run in runs
                   if run["spec"]["candidate_id"] == "geo_only"}
    _require(len(geo_by_seed) == sum(run["spec"]["candidate_id"] == "geo_only" for run in runs),
             "Expected one shared geo-only fit per seed")

    def add(kind: str, left: dict[str, Any], right: dict[str, Any]) -> None:
        lhs, rhs = left["spec"], right["spec"]
        identity = (kind, lhs["run_id"], rhs["run_id"])
        if identity in seen:
            return
        seen.add(identity)
        result.append(_paired_difference(left, right, policy, kind))

    for key, left in lookup.items():
        seed, regime, initialization, candidate = key
        if candidate != "geo_only" and seed in geo_by_seed:
            add("versus_shared_geo_only", left, geo_by_seed[seed])
        pairs = []
        if candidate == "attention_two_token":
            pairs.extend((("primary_attention_vs_projected", "projected_concat"),
                          ("attention_vs_capacity", "tokenwise_capacity")))
        if candidate != "image_only":
            pairs.append(("versus_image_only", "image_only"))
        if candidate not in {"raw_concat", "geo_only"}:
            pairs.append(("versus_raw_concat", "raw_concat"))
        if candidate.startswith("attention_cls_skip_"):
            pairs.append(("bypass_vs_cls", "attention_cls"))
        for kind, comparator in pairs:
            right = lookup.get((seed, regime, initialization, comparator))
            if right is not None:
                add(kind, left, right)
        if initialization == "cs_adapted":
            right = lookup.get((seed, regime, "pretrained", candidate))
            if right is not None:
                add("cs_adapted_minus_pretrained", left, right)
    return result


def _encoder_regime_comparison(runs: list[dict[str, Any]], reference_root: Path) -> dict[str, Any]:
    """Pair independent packages without changing the completed reference artifacts."""
    _require(all(run["spec"]["regime"] == "frozen_online" for run in runs),
             "A paired reference requires a frozen_online report")
    _require((reference_root / "runs").is_dir(), f"Paired reference run directory is missing: {reference_root}")
    seeds = {run["spec"]["seed"] for run in runs}
    reference = []
    for path in sorted((reference_root / "runs").rglob("run.json")):
        if any(part.startswith(("verification", "smoke", ".")) for part in path.relative_to(reference_root / "runs").parts):
            continue
        spec = json.loads(path.read_text())
        if (not spec.get("verification", False) and spec["seed"] in seeds
                and spec["regime"] == "joint_partial"):
            reference.append(_read_run(path))
    return _regime_comparison_from_runs(runs, reference, reference_root=str(reference_root))


def _regime_comparison_from_runs(frozen_runs: list[dict[str, Any]],
                                 joint_runs: list[dict[str, Any]], **metadata: Any) -> dict[str, Any]:
    """Compare already-read packages while keeping checkpoint policies separate."""
    key = lambda run: (run["spec"]["seed"], run["spec"]["candidate_id"], run["spec"]["initialization"])
    frozen = {key(run): run for run in frozen_runs}
    joint = {key(run): run for run in joint_runs}
    _require(len(frozen) == len(frozen_runs) and len(joint) == len(joint_runs),
             "Duplicate encoder-regime pairing identities")
    result = {"direction": "joint_partial_minus_frozen_online", **metadata,
              "reference_access": "read_only", "policies": {}}
    for policy in POLICIES:
        contrasts, pairing = [], []
        for identity in sorted(set(frozen) | set(joint)):
            seed, candidate, initialization = identity
            left, right = joint.get(identity), frozen.get(identity)
            lhs_status = "missing" if left is None else "complete" if policy in left["policies"] else "incomplete"
            rhs_status = "missing" if right is None else "complete" if policy in right["policies"] else "incomplete"
            paired = lhs_status == rhs_status == "complete"
            pairing.append({"policy": policy, "seed": seed, "candidate_id": candidate,
                            "initialization": initialization, "joint_partial_status": lhs_status,
                            "frozen_online_status": rhs_status, "paired_complete": paired,
                            "lhs_run_id": left["spec"]["run_id"] if left else None,
                            "rhs_run_id": right["spec"]["run_id"] if right else None,
                            **_loader_rng_pair(left, right)})
            if paired:
                contrasts.append({**_paired_difference(left, right, policy, result["direction"]),
                                  "candidate_id": candidate, "initialization": initialization,
                                  "lhs_regime": "joint_partial", "rhs_regime": "frozen_online"})
        result["policies"][policy] = {"paired_complete": len(contrasts), "pairing": pairing,
                                      "unavailable_pairs": [item for item in pairing if not item["paired_complete"]],
                                      "contrasts": contrasts}
    result["loader_rng_comparison"] = _loader_rng_interpretation(result["policies"][POLICIES[0]]["pairing"])
    return result


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(value) for value in row) + " |" for row in rows]])


def build_report(study_root: str | Path, output_dir: str | Path | None = None, *,
                 render_confusions: bool = True, seed: int | None = 1,
                 regime: str | None = "joint_partial",
                 paired_reference_root: str | Path | None = None,
                 completed_reference_roots: list[str | Path] | None = None,
                 deferred_reference_regimes: list[str] | None = None) -> dict[str, Any]:
    """Build a report from complete runs, explicitly listing unfinished fits.

    A report can refresh between fits. It neither labels intermediate epochs as
    final results nor combines final and best-validation checkpoint policies.
    Completed reference packages contribute saved results to a combined report
    without rechecking source-code identity or touching their existing reports.
    """
    root = Path(study_root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "report"
    reference_root = Path(paired_reference_root).resolve() if paired_reference_root is not None else None
    completed_roots = [Path(path).resolve() for path in (completed_reference_roots or [])]
    deferred = deferred_reference_regimes or []
    if deferred:
        _require(deferred == ["frozen_online"] and seed in (2, 3, 4) and regime == "geo_only",
                 "Only seed 2–4 geo-only reports may defer frozen-online")
        _require(len(completed_roots) == 1 and reference_root is None,
                 "Deferred frozen-online requires one completed partial-joint reference")
    _require(not (reference_root is not None and completed_roots),
             "Use either a paired reference or completed reference packages")
    _require(len(set(completed_roots)) == len(completed_roots), "Duplicate completed reference roots")
    for completed_root in completed_roots:
        _require(completed_root != root, "Completed reference must be a separate study root")
        _require(output != completed_root and completed_root not in output.parents,
                 "Report output cannot modify a completed reference root")
    if reference_root is not None:
        _require(reference_root != root, "Paired reference must be a separate study root")
        _require(output != reference_root and reference_root not in output.parents,
                 "Report output cannot modify the paired reference root")
    search = root / "runs" if (root / "runs").is_dir() else root
    paths = [path for path in sorted(search.rglob("run.json"))
             if not any(part.startswith(("verification", "smoke", ".")) for part in path.relative_to(search).parts)
             and output not in path.parents]
    runs = []
    for path in paths:
        spec = json.loads(path.read_text())
        if (not spec.get("verification", False) and (seed is None or spec["seed"] == seed)
                and (regime is None or spec["regime"] == regime)):
            runs.append(_read_run(path))
    _require(bool(runs), "No M3J run manifests found")
    for completed_root in completed_roots:
        reference_runs = []
        search_reference = completed_root / "runs"
        _require(search_reference.is_dir(), f"Completed reference run directory is missing: {completed_root}")
        for path in sorted(search_reference.rglob("run.json")):
            if any(part.startswith(("verification", "smoke", "."))
                   for part in path.relative_to(search_reference).parts):
                continue
            spec = json.loads(path.read_text())
            if (not spec.get("verification", False) and (seed is None or spec["seed"] == seed)
                    and spec["regime"] in {"joint_partial", "frozen_online"}):
                _require(spec["regime"] not in deferred,
                         "A deferred regime cannot also supply completed reference runs")
                reference_runs.append(_read_run(path))
        _require(bool(reference_runs), f"No matching runs in completed reference: {completed_root}")
        _require(all(run["policies"] for run in reference_runs),
                 f"Completed reference contains an unfinished fit: {completed_root}")
        runs.extend(reference_runs)
    _require(len({run["spec"]["run_id"] for run in runs}) == len(runs), "Duplicate run IDs")
    identities = [(run["spec"]["seed"], run["spec"]["regime"], run["spec"]["initialization"],
                   run["spec"]["candidate_id"]) for run in runs]
    _require(len(set(identities)) == len(runs), "Duplicate fit identities")
    completed = [run for run in runs if run["policies"]]
    incomplete = [run for run in runs if not run["policies"]]
    regime_comparison = _encoder_regime_comparison(runs, reference_root) if reference_root is not None else None
    if completed_roots and not deferred:
        regime_comparison = _regime_comparison_from_runs(
            [run for run in runs if run["spec"]["regime"] == "frozen_online"],
            [run for run in runs if run["spec"]["regime"] == "joint_partial"],
            reference_roots=[str(path) for path in completed_roots])
    exploratory = (any(run["result_assurance"] == PROVISIONAL_ASSURANCE for run in runs)
                   or bool(regime_comparison and any(
                       row["result_assurance"] == PROVISIONAL_ASSURANCE
                       for policy in regime_comparison["policies"].values() for row in policy["contrasts"])))
    labels = runs[0]["spec"]["class_labels"]
    _require(all(run["spec"]["class_labels"] == labels for run in runs), "Class-label order differs between runs")
    output.mkdir(parents=True, exist_ok=True)
    if completed:
        first = completed[0]["policies"]["final"]["predictions"]
        for run in completed:
            _cohort_equal(first, run["policies"]["final"]["predictions"])
    inventory = [{"run_id": run["spec"]["run_id"], "candidate_id": run["spec"]["candidate_id"],
                  "initialization": run["spec"]["initialization"], "regime": run["spec"]["regime"],
                  "seed": run["spec"]["seed"], "completed_epochs": len(run["history"]),
                  "expected_epochs": run["spec"]["expected_epochs"],
                  "complete": bool(run["policies"]), "status": run["spec"].get("status", "unknown"),
                  "result_assurance": run["result_assurance"],
                  "verification_policy": run["verification_policy"]}
                 for run in runs]
    _csv(output / "tables/inventory.csv", inventory)
    summary = {"schema_version": "attention_fusion_m3j_report_v1", "completed_fits": len(completed),
               "incomplete_fits": len(incomplete), "seeds": sorted({run["spec"]["seed"] for run in runs}),
               "class_labels": labels, "inventory": inventory, "policies": {}}
    if regime_comparison is not None:
        summary["encoder_regime_comparison"] = regime_comparison
        rng_comparison = regime_comparison["loader_rng_comparison"]
        summary["encoder_regime_rng_interpretation"] = rng_comparison["interpretation"]
        if rng_comparison["status"] != "matched_namespaces":
            summary["encoder_regime_rng_limitation"] = rng_comparison["interpretation"]
    if completed_roots:
        summary["completed_reference_packages"] = {
            "roots": [str(path) for path in completed_roots], "access": "read_only"}
    if deferred:
        summary.update(
            active_planned_fits=17, deferred_planned_fits=16, full_planned_fits=33,
            active_scope_complete=len(completed) == 17 and not incomplete,
            deferred_reference_packages=[{
                "seed": seed, "regime": "frozen_online", "status": "deferred", "planned_fits": 16}],
            encoder_regime_comparison_status="deferred",
        )
    if exploratory:
        summary["result_assurance"] = PROVISIONAL_ASSURANCE
        summary["numerical_assurance_notice"] = PROVISIONAL_NOTICE
    text = ["# M3J development results" + (f" — {PROVISIONAL_TITLE}" if exploratory else ""), "",
            f"**{len(completed)} complete fits; {len(incomplete)} incomplete fits in the current run inventory.**", "",
            "All scores use the fixed development validation cohort. No cleaned-test results are used. "
            "Comparisons are provisional, descriptive seed-level observations; they do not establish supported improvement across seeds. "
            "The original pretrained and CS-adapted initializations receive equal additional M3J training, "
            "but the CS-adapted encoder already received five CS adaptation epochs.", "",
            "Final and best-validation results remain separate. Best-validation scores are conditional on checkpoint "
            "selection using this same validation cohort. A single chosen checkpoint supplies all metrics; "
            "there is no per-metric epoch selection or early stopping.", "",
            "Under direct_freeze_checks_v1, the original epoch validation history selects the checkpoint "
            "by Acc@1, weighted F1, MCC and earliest epoch. Reported final/best scores and confusion matrices "
            "come from evaluating those saved checkpoints after loading. The original curves remain unchanged; "
            "identical repeated predictions are not a completion requirement.", "",
            "[Run inventory](tables/inventory.csv)", ""]
    if completed_roots:
        reference_label = "partial-joint package" if deferred else "partial-joint and frozen-online packages"
        text.extend([f"This combined report reads the completed {reference_label} "
                     "without modifying them. Each seed has one shared geo-only fit, counted once in the "
                     "inventory and reused as the same comparator across both image initializations and active regimes. "
                     "It does not have an image encoder or a pretrained/CS-adapted initialization.", ""])
    if deferred:
        text.extend(["**Current scope: 17 planned active fits (16 partial-joint + one geo-only); "
                     "16 frozen-online fits deferred for later; 33 fits remain in the full seed plan.** "
                     "Deferred fits are not failed or incomplete runs. This report includes no frozen-online "
                     "scores or partial-joint versus frozen-online comparison. That comparison will be revisited "
                     "when the deferred package is run.", ""])
    if regime_comparison is not None:
        text.extend([f"**Cross-regime interpretation:** {summary['encoder_regime_rng_interpretation']}", ""])
    if exploratory:
        text[2:2] = [f"**{PROVISIONAL_NOTICE}**", ""]
    audits = []
    for run in runs:
        audit = run["numerical_audit"]
        if audit is not None or run["legacy_numerical_policy"]:
            audits.append({"run_id": run["spec"]["run_id"], "result_assurance": run["result_assurance"],
                           "audit_present": audit is not None,
                           "audit_path": os.path.relpath(run["directory"] / "numerical_audit.json", output)
                                         if audit is not None else None,
                           "summary": audit.get("summary", {}) if audit is not None else None})
    if audits:
        summary["numerical_audits"] = audits
        text.extend(["## Historical numerical audit", "",
                     "Legacy results retain their original tolerance audit. The active direct-check policy has no tolerance audit. "
                     "Passing historical limits does not "
                     "resolve the machine's memory-integrity problem or establish that undetected errors were absent.", "",
                     _table(["Run", "Accepted numerical comparisons", "Rejected comparisons", "Frozen changed coordinates",
                             "Maximum absolute difference", "Audit"],
                            [[item["run_id"],
                              *[(item["summary"] or {}).get(key, "unavailable") for key in (
                                  "accepted_numerical_comparisons", "rejected_comparisons", "frozen_changed_coordinates",
                                  "max_absolute_error")],
                              f"[Recorded audit]({item['audit_path']})" if item["audit_present"] else "**Missing audit**"]
                             for item in audits]), ""])
    for policy in POLICIES:
        rows, classes = [], []
        for run in completed:
            spec, selected = run["spec"], run["policies"][policy]
            metrics, arrays = selected["metrics"], selected["predictions"]
            rows.append({"run_id": spec["run_id"], "candidate_id": spec["candidate_id"],
                         "initialization": spec["initialization"], "regime": spec["regime"], "seed": spec["seed"],
                         "epoch": selected["epoch"], **{metric: metrics[metric] for metric in ("loss", *METRICS)},
                         "validation_rows": len(arrays["label"]), "elapsed_seconds": spec.get("elapsed_seconds"),
                         "result_assurance": run["result_assurance"],
                         "verification_policy": run["verification_policy"],
                         "metrics_source": selected["metrics_source"],
                         **{f"original_epoch_{metric}": selected["selection_metrics"][metric]
                            for metric in ("loss", *METRICS)}})
            cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            for index, label in enumerate(labels):
                support = int(cm[index].sum())
                plot_count = len(np.unique(arrays["plot_idx"][arrays["label"] == index]))
                classes.append({"run_id": spec["run_id"], "policy": policy, "dense_id": index,
                                "class_name": label, "support_images": support, "support_plots": plot_count,
                                "correct": int(cm[index, index]),
                                "recall": int(cm[index, index]) / support if support else None,
                                "low_support": support < 20 or plot_count < 10,
                                "result_assurance": run["result_assurance"],
                                **{f"pred_{j:02d}": int(value) for j, value in enumerate(cm[index])}})
            if render_confusions:
                def sink(fig: Any, values: np.ndarray, normalized: bool,
                         spec: dict[str, Any] = spec, epoch: int = selected["epoch"],
                         assurance: str = run["result_assurance"]) -> None:
                    suffix = "normalized" if normalized else "counts"
                    warning = f"\n{PROVISIONAL_TITLE}" if assurance == PROVISIONAL_ASSURANCE else ""
                    fig.suptitle(f"{spec['run_id']} · {policy} epoch {epoch}{warning}", y=1.005, fontsize=12)
                    _save(fig, output / "figures/confusion" / policy / f"{spec['run_id']}_{suffix}")
                draw_cm(cm, labels, sink=sink)
        contrasts = _contrasts(completed, policy)
        summary["policies"][policy] = {"runs": rows, "contrasts": contrasts}
        _csv(output / "tables" / f"{policy}_metrics.csv", rows)
        _csv(output / "tables" / f"{policy}_contrasts.csv", contrasts, columns=[
            "policy", "kind", "seed", "regime", "lhs_run_id", "rhs_run_id", "lhs_epoch", "rhs_epoch",
            *[f"difference_{metric}" for metric in METRICS], "rescued", "harmed", "net_correct", "interpretation",
            "result_assurance"])
        _csv(output / "tables" / f"{policy}_confusion_by_class.csv", classes)
        title = "Final epoch" if policy == "final" else "Best validation"
        text.extend([f"## {title}", ""])
        if rows:
            text.extend([_table(["Regime", "Initialization", "Model", "Seed", "Epoch", "Acc@1 (%)", "Acc@3 (%)", "Weighted F1 (%)", "MCC"],
                                [[row["regime"], row["initialization"], MODEL_LABELS[row["candidate_id"]], row["seed"], row["epoch"],
                                  f"{100 * row['acc1']:.2f}", f"{100 * row['acc3']:.2f}",
                                  f"{100 * row['weighted_f1']:.2f}", f"{row['mcc']:.4f}"] for row in rows]), "",
                         f"[Metrics](tables/{policy}_metrics.csv) · [All paired contrasts and rescue/harm counts](tables/{policy}_contrasts.csv) · "
                         f"[Per-class counts, recall and support](tables/{policy}_confusion_by_class.csv)", "",
                         "Differences are left minus right. Rescue means a left model corrects a right-model error; "
                         "harm means it breaks a correct right-model prediction. These are paired image counts, "
                         "not independent repeated validation samples. Fractional metric differences in CSV become "
                         "percentage-point differences after multiplication by 100 (except MCC).", ""])
            geo_contrasts = [item for item in contrasts if item["kind"] == "versus_shared_geo_only"]
            if geo_contrasts:
                text.extend(["### Shared geo-only reference", "",
                             "Each row is an image-based model minus the same seed's geo-only checkpoint "
                             "under the current checkpoint policy. Best-validation comparisons use each fit's "
                             "own selected epoch. These shared comparisons do not constitute additional geo-only fits.", "",
                             _table(["Image-model run", "Geo-only run", "ΔAcc@1 (pp)", "Rescued", "Harmed", "Net"],
                                    [[item["lhs_run_id"], item["rhs_run_id"],
                                      f"{100 * item['difference_acc1']:+.2f}", item["rescued"], item["harmed"],
                                      item["net_correct"]] for item in geo_contrasts]), ""])
            highlighted = [item for item in contrasts if item["kind"] in {
                "primary_attention_vs_projected", "cs_adapted_minus_pretrained"}]
            if highlighted:
                text.extend([_table(["Comparison", "Left run", "Right run", "ΔAcc@1 (pp)", "Rescued", "Harmed", "Net"],
                                    [[item["kind"], item["lhs_run_id"], item["rhs_run_id"],
                                      f"{100 * item['difference_acc1']:+.2f}", item["rescued"], item["harmed"],
                                      item["net_correct"]] for item in highlighted]), ""])
            if render_confusions:
                text.extend([_table(["Run", "Confusion counts", "Row-normalized confusion"],
                                    [[row["run_id"],
                                      f"[PNG](figures/confusion/{policy}/{row['run_id']}_counts.png) · "
                                      f"[PDF](figures/confusion/{policy}/{row['run_id']}_counts.pdf)",
                                      f"[PNG](figures/confusion/{policy}/{row['run_id']}_normalized.png) · "
                                      f"[PDF](figures/confusion/{policy}/{row['run_id']}_normalized.pdf)"] for row in rows]), ""])
        else:
            text.extend(["No complete checkpoint-policy comparisons are available yet.", ""])
        if regime_comparison is not None:
            comparison = regime_comparison["policies"][policy]
            regime_contrasts = comparison["contrasts"]
            _csv(output / "tables" / f"{policy}_encoder_regime_contrasts.csv", regime_contrasts, columns=[
                "policy", "kind", "seed", "regime", "candidate_id", "initialization", "lhs_regime", "rhs_regime",
                "lhs_run_id", "rhs_run_id", "lhs_epoch", "rhs_epoch",
                *[f"difference_{metric}" for metric in METRICS], "rescued", "harmed", "net_correct",
                "interpretation", "result_assurance"])
            _csv(output / "tables" / f"{policy}_encoder_regime_pairing.csv", comparison["pairing"])
            text.extend([f"### {title}: matched encoder regimes", "",
                         "Differences below are **partial-joint minus frozen-online**, matched by seed, architecture, "
                         "initialization and checkpoint policy. Validation image identities, labels, plots and class axes "
                         "must align. Each regime selects its own best-validation epoch using the same rule. "
                         "The reference partial-joint package is read without modification.", "",
                         f"{comparison['paired_complete']} complete pairs; "
                         f"{len(comparison['unavailable_pairs'])} unavailable pairs. "
                         "Missing or unfinished fits do not supply a score or a zero difference.", "",
                         f"[Regime differences](tables/{policy}_encoder_regime_contrasts.csv) · "
                         f"[Pair completion inventory](tables/{policy}_encoder_regime_pairing.csv)", ""])
            if regime_contrasts:
                text.extend([_table(
                    ["Initialization", "Model", "Seed", "Joint epoch", "Frozen epoch", "ΔAcc@1 (pp)",
                     "ΔAcc@3 (pp)", "Δweighted F1 (pp)", "ΔMCC", "Rescued", "Harmed"],
                    [[item["initialization"], MODEL_LABELS[item["candidate_id"]], item["seed"],
                      item["lhs_epoch"], item["rhs_epoch"],
                      *[f"{100 * item[f'difference_{metric}']:+.2f}" for metric in ("acc1", "acc3", "weighted_f1")],
                      f"{item['difference_mcc']:+.4f}", item["rescued"], item["harmed"]]
                     for item in regime_contrasts]), ""])
            if comparison["unavailable_pairs"]:
                text.extend([_table(
                    ["Initialization", "Model", "Seed", "Partial-joint", "Frozen-online"],
                    [[item["initialization"], MODEL_LABELS[item["candidate_id"]], item["seed"],
                      item["joint_partial_status"], item["frozen_online_status"]]
                     for item in comparison["unavailable_pairs"]]), ""])
    text.extend(["## Learning curves and checkpoint reuse", "",
                 "Each run's monitoring page refreshes from saved epoch history. Training-pass metrics use dropout "
                 "(and image augmentation for image-based fits) "
                 "and weights that change within the epoch, so they are not directly comparable to fixed-checkpoint validation. "
                 "The shared learning rate and resource traces are recorded alongside the metrics.", ""])
    for run in runs:
        directory = run["directory"]
        curve = directory / "monitoring.html"
        if not curve.exists():
            curve = directory / "curves" / "monitoring.html"
        if curve.exists():
            link = os.path.relpath(curve, output)
            text.append(f"- [{run['spec']['run_id']}]({link})")
    text.extend(["", "Class recall is unavailable when validation support is zero. Classes with fewer than 20 images "
                 "or 10 plots are flagged as low support. Confusion plots show zero for unsupported rows; "
                 "the numeric tables preserve unavailable recall.", ""])
    _text(output / "report.md", "\n".join(text))
    _json(output / "summary.json", summary)
    return summary

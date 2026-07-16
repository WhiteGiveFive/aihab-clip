"""Single-seed agreement analysis for the multimodal CS baselines.

The module deliberately separates checkpoint inference from analysis.  The
canonical ``model_outputs.parquet`` contains ordered logits and native-T=1
probabilities; all agreement metrics, bootstraps, plots, and reports consume
that cache and therefore do not need to load a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import uuid
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef
from torch.utils.data import DataLoader

from multimodal.data import (
    FeatureTableDataset,
    image_feature_columns,
    joined_table_dir,
    load_joined_splits,
    run_dir,
    tabular_feature_columns,
    tabular_modality_name,
)
from multimodal.trainer import _apply_tabular_standardization, _build_model, _reindex_labels


MODEL_PREFIXES: Dict[str, str] = {
    "image_only": "image",
    "geo_only": "geo",
    "raw_concat": "fusion",
}
PREFIXES = ("image", "geo", "fusion")
PROBABILITY_BASIS = "native_t1_uncalibrated"


@dataclass(frozen=True)
class AgreementConfig:
    """Configuration for one cached, single-seed agreement analysis."""

    seed: int = 1
    joined_table_tag: str = "gse_100m_cleaned_test"
    run_tag: str = "gse_100m_train_cleaned_test_epoch50"
    checkpoint_name: str = "best_model.pt"
    device: str = "auto"
    batch_size: int = 256
    num_workers: int = 0
    output_root: str | Path | None = None
    report_root: str | Path | None = None
    cache_policy: str = "reuse_or_build"
    modes: tuple[str, ...] = ("image_only", "geo_only", "raw_concat")
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260714
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        allowed_policies = {"reuse_or_build", "cache_only", "rebuild"}
        if self.cache_policy not in allowed_policies:
            raise ValueError(
                f"cache_policy must be one of {sorted(allowed_policies)}; "
                f"got {self.cache_policy!r}"
            )
        missing = set(MODEL_PREFIXES).difference(self.modes)
        extra = set(self.modes).difference(MODEL_PREFIXES)
        if missing or extra or len(self.modes) != 3:
            raise ValueError(
                "Agreement analysis requires exactly image_only, geo_only, and "
                f"raw_concat; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")


def _safe_tag(value: object) -> str:
    return (
        str(value)
        .replace("hf-hub:", "hf-hub_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _resolved_output_root(cfg: Mapping, configured: str | Path | None) -> Path:
    root = Path(str(cfg.get("root_path", "./"))).resolve()
    value = configured
    if value is None:
        value = cfg.get("multimodal", {}).get("output_dir", "./multimodal_artifacts")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def analysis_paths(cfg: Mapping, spec: AgreementConfig) -> Dict[str, Path]:
    """Return canonical cache and report paths for ``spec``."""

    dataset = _safe_tag(cfg.get("dataset", "cs"))
    output_root = _resolved_output_root(cfg, spec.output_root)
    analysis_dir = (
        output_root
        / "analysis"
        / dataset
        / _safe_tag(spec.joined_table_tag)
        / "baseline_agreement"
        / _safe_tag(spec.run_tag)
        / f"seed{int(spec.seed)}"
    )
    if spec.report_root is None:
        report_base = output_root / "reports"
    else:
        report_base = _resolved_output_root(cfg, spec.report_root)
    report_dir = (
        report_base
        / dataset
        / _safe_tag(spec.joined_table_tag)
        / "baseline_agreement"
        / _safe_tag(spec.run_tag)
        / f"seed{int(spec.seed)}"
    )
    return {
        "output_root": output_root,
        "analysis_dir": analysis_dir,
        "report_dir": report_dir,
        "model_outputs": analysis_dir / "model_outputs.parquet",
        "model_outputs_manifest": analysis_dir / "model_outputs_manifest.json",
        "per_instance_metrics": analysis_dir / "per_instance_metrics_native_t1.parquet",
        "summary_md": report_dir / "summary.md",
        "summary_json": report_dir / "summary.json",
        "figures_dir": report_dir / "figures",
    }


def _analysis_cfg(cfg: Mapping, spec: AgreementConfig, mode: str) -> dict:
    out = copy.deepcopy(dict(cfg))
    out["seed"] = int(spec.seed)
    mm = copy.deepcopy(dict(out.get("multimodal", {})))
    mm["joined_table_tag"] = spec.joined_table_tag
    mm["run_tag"] = spec.run_tag
    mm["fusion_mode"] = mode
    out["multimodal"] = mm
    return out


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(stat.st_size),
    }


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
    }
    try:
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import pyarrow

        versions["pyarrow"] = pyarrow.__version__
    except ImportError:
        pass
    return versions


def _canonical_label_remap(value: Mapping) -> dict[str, int]:
    return {str(int(key)): int(mapped) for key, mapped in value.items()}


def validate_artifacts(cfg: Mapping, spec: AgreementConfig) -> dict[str, object]:
    """Validate joined tables and all three final-checkpoint artifacts.

    This function intentionally touches checkpoint files.  ``cache_only`` uses
    :func:`validate_model_outputs` instead and never calls this function.
    """

    joined_cfg = _analysis_cfg(cfg, spec, "raw_concat")
    joined_dir = joined_table_dir(joined_cfg)
    joined_paths = {split: joined_dir / f"{split}.parquet" for split in ("train", "val", "test")}
    missing_joined = [str(path) for path in joined_paths.values() if not path.exists()]
    if missing_joined:
        raise FileNotFoundError(f"Missing joined tables: {missing_joined}")

    model_records: dict[str, dict[str, object]] = {}
    expected_names: list[str] | None = None
    expected_remap: dict[str, int] | None = None
    expected_features: list[str] | None = None
    for mode in spec.modes:
        mode_cfg = _analysis_cfg(cfg, spec, mode)
        directory = run_dir(mode_cfg)
        scaler_name = "geo_standardization.json" if tabular_modality_name(mode_cfg) == "geo" else "tabular_standardization.json"
        paths = {
            "checkpoint": directory / spec.checkpoint_name,
            "scaler": directory / scaler_name,
            "metrics": directory / "metrics.json",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {mode} artifacts: {missing}")
        scaler = _read_json(paths["scaler"])
        metrics = _read_json(paths["metrics"])
        checkpoint_header = _load_checkpoint(paths["checkpoint"], torch.device("cpu"))
        checkpoint_epoch = checkpoint_header.get("epoch")
        checkpoint_regime = checkpoint_header.get("training_regime")
        expected_epoch = (
            int(metrics["history"][-1]["epoch"])
            if metrics.get("history")
            else None
        )
        if checkpoint_regime != "fixed_epoch_train_val" or metrics.get("training_regime") != "fixed_epoch_train_val":
            raise ValueError(
                f"{mode} is not a fixed-epoch final-fit checkpoint: "
                f"checkpoint={checkpoint_regime!r}, metrics={metrics.get('training_regime')!r}"
            )
        if checkpoint_epoch is None or expected_epoch is None or int(checkpoint_epoch) != expected_epoch:
            raise ValueError(
                f"{mode} checkpoint is not the saved final epoch: "
                f"checkpoint epoch={checkpoint_epoch!r}, final metrics epoch={expected_epoch!r}"
            )
        if not bool(metrics.get("train_on_train_val", False)) or bool(metrics.get("early_stopping", True)):
            raise ValueError(f"{mode} metrics do not describe fixed-epoch train+val fitting")
        class_names = [str(value) for value in scaler.get("class_names", [])]
        remap = _canonical_label_remap(scaler.get("label_id_remap", {}))
        feature_columns = [str(value) for value in scaler.get("feature_columns", [])]
        if not class_names or not remap:
            raise ValueError(f"{mode} scaler lacks class_names or label_id_remap")
        if metrics.get("mode") != mode:
            raise ValueError(
                f"{mode} metrics mode mismatch: found {metrics.get('mode')!r}"
            )
        if [str(v) for v in metrics.get("class_names", [])] != class_names:
            raise ValueError(f"{mode} class order differs between scaler and metrics")
        if _canonical_label_remap(metrics.get("label_id_remap", {})) != remap:
            raise ValueError(f"{mode} label remap differs between scaler and metrics")
        if expected_names is None:
            expected_names = class_names
            expected_remap = remap
            expected_features = feature_columns
        elif class_names != expected_names or remap != expected_remap:
            raise ValueError(f"{mode} class mapping differs from the other baselines")
        elif feature_columns != expected_features:
            raise ValueError(f"{mode} tabular feature columns differ from the other baselines")
        model_records[mode] = {
            "run_dir": str(directory.resolve()),
            "paths": {key: str(path.resolve()) for key, path in paths.items()},
            "scaler": scaler,
            "metrics": metrics,
            "checkpoint_metadata": {
                "epoch": int(checkpoint_epoch),
                "training_regime": str(checkpoint_regime),
                "mode": str(checkpoint_header.get("mode", mode)),
            },
            "fingerprints": {key: _fingerprint(path) for key, path in paths.items()},
        }
        del checkpoint_header

    assert expected_names is not None and expected_remap is not None
    fingerprints = {
        "joined_tables": {split: _fingerprint(path) for split, path in joined_paths.items()},
        "models": {mode: record["fingerprints"] for mode, record in model_records.items()},
    }
    return {
        "joined_table_dir": str(joined_dir.resolve()),
        "joined_paths": {key: str(path.resolve()) for key, path in joined_paths.items()},
        "models": model_records,
        "class_names": expected_names,
        "label_id_remap": expected_remap,
        "tabular_feature_columns": expected_features or [],
        "fingerprints": fingerprints,
    }


def _resolve_device(value: str) -> torch.device:
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {value!r}, but CUDA is not available")
    return device


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _metric_from_predictions(
    labels: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    pred = logits.argmax(axis=1)
    top3 = np.argpartition(-logits, kth=min(2, num_classes - 1), axis=1)[:, : min(3, num_classes)]
    cm = confusion_matrix(labels, pred, labels=np.arange(num_classes))
    support = cm.sum(axis=1).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support != 0)
    denom = precision + recall
    per_class_f1 = np.divide(2 * precision * recall, denom, out=np.zeros(num_classes), where=denom != 0)
    total = float(support.sum())
    return {
        "top1_acc": float((pred == labels).mean()),
        "top3_acc": float(np.any(top3 == labels[:, None], axis=1).mean()),
        "f1": float(np.dot(per_class_f1, support) / total) if total else math.nan,
        "mcc": float(matthews_corrcoef(labels, pred)),
        "cm": cm,
    }


def _reproduction_check(saved: Mapping, recomputed: Mapping) -> dict[str, object]:
    names = ("top1_acc", "top3_acc", "f1", "mcc")
    saved_values = {name: float(saved[name]) for name in names}
    recomputed_values = {name: float(recomputed[name]) for name in names}
    delta = {name: abs(saved_values[name] - recomputed_values[name]) for name in names}
    saved_cm = np.asarray(saved["cm"], dtype=np.int64)
    observed_cm = np.asarray(recomputed["cm"], dtype=np.int64)
    return {
        "saved": saved_values,
        "recomputed": recomputed_values,
        "absolute_delta": delta,
        "confusion_matrix_exact": bool(np.array_equal(saved_cm, observed_cm)),
    }


def _identity_columns(test: pd.DataFrame, remap: Mapping[str, int]) -> pd.DataFrame:
    joined_label = test["label_id"].astype(int)
    missing = sorted(set(joined_label.astype(str)).difference(remap))
    if missing:
        raise ValueError(f"Test labels are absent from checkpoint remap: {missing}")
    source_label = (
        test["label_id_original"].astype(int)
        if "label_id_original" in test.columns
        else joined_label
    )
    data: dict[str, object] = {
        "row_index": np.arange(len(test), dtype=np.int64),
        "file": test["file"].astype(str).to_numpy(),
        "file_normalized": test["file"].astype(str).str.strip().str.lower().to_numpy(),
        "source_label_id": source_label.to_numpy(dtype=np.int64),
        "class_index": joined_label.astype(str).map(remap).to_numpy(dtype=np.int64),
        "label_name": test["label_name"].astype(str).to_numpy(),
    }
    for column in ("plot_idx", "image_source", "l2_label", "split", "label_id_original"):
        if column in test.columns:
            data[column] = test[column].to_numpy()
    return pd.DataFrame(data)


def _model_output_columns(
    prefix: str,
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    order = np.argsort(-probabilities, axis=1, kind="stable")
    pred = order[:, 0]
    top3 = order[:, :3]
    rows = np.arange(len(labels))
    true_prob = probabilities[rows, labels]
    entropy = normalized_entropy(probabilities)
    margin = probabilities[rows, order[:, 0]] - probabilities[rows, order[:, 1]]
    values: dict[str, object] = {}
    for class_index in range(logits.shape[1]):
        values[f"{prefix}_logit_c{class_index:02d}"] = logits[:, class_index].astype(np.float32)
        values[f"{prefix}_prob_t1_c{class_index:02d}"] = probabilities[:, class_index].astype(np.float32)
    values.update(
        {
            f"{prefix}_pred": pred.astype(np.int16),
            f"{prefix}_top3_1": top3[:, 0].astype(np.int16),
            f"{prefix}_top3_2": top3[:, 1].astype(np.int16),
            f"{prefix}_top3_3": top3[:, 2].astype(np.int16),
            f"{prefix}_correct": (pred == labels),
            f"{prefix}_confidence_t1": probabilities[rows, pred].astype(np.float32),
            f"{prefix}_entropy_normalized_t1": entropy.astype(np.float32),
            f"{prefix}_top2_margin_t1": margin.astype(np.float32),
            f"{prefix}_true_class_probability_t1": true_prob.astype(np.float32),
            f"{prefix}_nll_t1": (-np.log(np.clip(true_prob, 1e-12, 1.0))).astype(np.float32),
        }
    )
    return values


def run_ordered_test_inference(
    cfg: Mapping,
    spec: AgreementConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reconstruct all classifiers and run ordered inference on the joined test table."""

    artifacts = validate_artifacts(cfg, spec)
    joined_cfg = _analysis_cfg(cfg, spec, "raw_concat")
    raw_tables = load_joined_splits(joined_cfg)
    source_train_rows = len(raw_tables["train"])
    source_val_rows = len(raw_tables["val"])
    combined_train = pd.concat([raw_tables["train"], raw_tables["val"]], ignore_index=True)
    tables, remap_int, class_names = _reindex_labels(
        {"train": combined_train, "test": raw_tables["test"]},
        training_split_name="combined train+validation split",
    )
    canonical_remap = _canonical_label_remap(remap_int)
    if class_names != artifacts["class_names"] or canonical_remap != artifacts["label_id_remap"]:
        raise ValueError("Joined table class mapping does not match saved classifier artifacts")

    output = _identity_columns(raw_tables["test"], artifacts["label_id_remap"])
    labels = output["class_index"].to_numpy(dtype=np.int64)
    device = _resolve_device(spec.device)
    reproduction: dict[str, object] = {}
    feature_dims: dict[str, int] = {
        "image": len(image_feature_columns(tables["train"])),
        "tabular": len(tabular_feature_columns(joined_cfg)),
        "classes": len(class_names),
    }
    checkpoint_metadata: dict[str, object] = {}

    for mode in spec.modes:
        prefix = MODEL_PREFIXES[mode]
        mode_cfg = _analysis_cfg(cfg, spec, mode)
        record = artifacts["models"][mode]
        scaler = record["scaler"]
        feature_cols = [str(v) for v in scaler["feature_columns"]]
        standardized_train = _apply_tabular_standardization(tables["train"], scaler, feature_cols)
        standardized_test = _apply_tabular_standardization(tables["test"], scaler, feature_cols)
        dataset = FeatureTableDataset(standardized_test, mode=mode, tabular_cols=feature_cols)
        loader = DataLoader(
            dataset,
            batch_size=int(spec.batch_size),
            shuffle=False,
            num_workers=int(spec.num_workers),
            pin_memory=device.type == "cuda",
        )
        model = _build_model(mode_cfg, standardized_train, feature_cols).to(device)
        checkpoint_path = Path(record["paths"]["checkpoint"])
        checkpoint = _load_checkpoint(checkpoint_path, device)
        if str(checkpoint.get("mode", mode)) != mode:
            raise ValueError(f"Checkpoint mode mismatch for {mode}")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for image_values, tabular_values, _targets in loader:
                logits = model(image_values.to(device), tabular_values.to(device))
                batches.append(logits.detach().cpu().numpy().astype(np.float32))
        logits_np = np.concatenate(batches, axis=0)
        if logits_np.shape != (len(output), len(class_names)):
            raise ValueError(
                f"{mode} output shape mismatch: expected {(len(output), len(class_names))}, "
                f"got {logits_np.shape}"
            )
        output = pd.concat(
            [output, pd.DataFrame(_model_output_columns(prefix, logits_np, labels))],
            axis=1,
        )
        recomputed = _metric_from_predictions(labels, logits_np, len(class_names))
        reproduction[mode] = _reproduction_check(record["metrics"]["test"], recomputed)
        if not reproduction[mode]["confusion_matrix_exact"]:
            raise ValueError(f"{mode} recomputed confusion matrix differs from metrics.json")
        if max(reproduction[mode]["absolute_delta"].values()) > 1e-6:
            raise ValueError(f"{mode} recomputed final metrics differ from metrics.json")
        checkpoint_metadata[mode] = {
            "epoch": checkpoint.get("epoch"),
            "training_regime": checkpoint.get("training_regime", record["metrics"].get("training_regime")),
            "mode": checkpoint.get("mode", mode),
            "checkpoint_name": spec.checkpoint_name,
        }
        del model, checkpoint

    manifest: dict[str, object] = {
        "schema_version": spec.schema_version,
        "artifact_kind": "multimodal_baseline_model_outputs",
        "immutable": True,
        "temperature": 1.0,
        "calibrated": False,
        "probability_basis": PROBABILITY_BASIS,
        "dataset": str(cfg.get("dataset", "cs")),
        "split": "test",
        "seed": int(spec.seed),
        "joined_table_tag": spec.joined_table_tag,
        "run_tag": spec.run_tag,
        "checkpoint_name": spec.checkpoint_name,
        "modes": list(spec.modes),
        "model_prefixes": MODEL_PREFIXES,
        "rows": int(len(output)),
        "unique_files": int(output["file_normalized"].nunique()),
        "plots": int(output["plot_idx"].astype(str).nunique()) if "plot_idx" in output else None,
        "class_count": int(len(class_names)),
        "class_names": class_names,
        "label_id_remap": canonical_remap,
        "feature_dimensions": feature_dims,
        "source_train_rows": int(source_train_rows),
        "source_val_rows": int(source_val_rows),
        "optimization_rows": int(len(tables["train"])),
        "checkpoint_metadata": checkpoint_metadata,
        "reproduction_checks": reproduction,
        "package_versions": _package_versions(),
        "git_revision": _git_revision(Path(str(cfg.get("root_path", "./"))).resolve()),
        "source_fingerprints": artifacts["fingerprints"],
        "source_artifact_paths": {
            "joined_tables": artifacts["joined_paths"],
            "models": {
                mode: artifacts["models"][mode]["paths"] for mode in spec.modes
            },
        },
        "column_count": int(len(output.columns)),
    }
    validate_model_outputs(output, manifest)
    return output, manifest


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_model_outputs(
    frame: pd.DataFrame,
    manifest: Mapping[str, object],
    *,
    probability_tolerance: float = 1e-6,
) -> None:
    """Validate a loaded cache without touching any checkpoint artifact."""

    required_identity = {
        "row_index",
        "file",
        "file_normalized",
        "plot_idx",
        "source_label_id",
        "class_index",
        "label_name",
    }
    missing = required_identity.difference(frame.columns)
    if missing:
        raise ValueError(f"model_outputs is missing identity columns: {sorted(missing)}")
    if int(manifest.get("rows", -1)) != len(frame):
        raise ValueError("model_outputs row count differs from manifest")
    if int(manifest.get("class_count", -1)) <= 1:
        raise ValueError("manifest has an invalid class_count")
    class_count = int(manifest["class_count"])
    if len(manifest.get("class_names", [])) != class_count:
        raise ValueError("manifest class_names length differs from class_count")
    if frame["file_normalized"].duplicated().any():
        raise ValueError("model_outputs must contain one unique row per normalized filename")
    expected_index = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), expected_index):
        raise ValueError("model_outputs row_index is not ordered from zero")
    labels = frame["class_index"].to_numpy(dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("model_outputs contains out-of-range class indices")

    for prefix in PREFIXES:
        logit_cols = [f"{prefix}_logit_c{i:02d}" for i in range(class_count)]
        prob_cols = [f"{prefix}_prob_t1_c{i:02d}" for i in range(class_count)]
        scalar_cols = {
            f"{prefix}_pred",
            f"{prefix}_correct",
            f"{prefix}_confidence_t1",
            f"{prefix}_entropy_normalized_t1",
            f"{prefix}_top2_margin_t1",
            f"{prefix}_true_class_probability_t1",
            f"{prefix}_nll_t1",
            f"{prefix}_top3_1",
            f"{prefix}_top3_2",
            f"{prefix}_top3_3",
        }
        missing_model = set(logit_cols + prob_cols).union(scalar_cols).difference(frame.columns)
        if missing_model:
            raise ValueError(f"model_outputs is missing {prefix} columns: {sorted(missing_model)}")
        logits = frame[logit_cols].to_numpy(dtype=np.float64)
        probabilities = frame[prob_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise ValueError(f"{prefix} outputs contain non-finite values")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=probability_tolerance, rtol=0):
            raise ValueError(f"{prefix} native-T=1 probabilities do not sum to one")
        if np.any(probabilities < -probability_tolerance) or np.any(probabilities > 1 + probability_tolerance):
            raise ValueError(f"{prefix} probabilities fall outside [0, 1]")
        predicted = probabilities.argmax(axis=1)
        if not np.array_equal(predicted, frame[f"{prefix}_pred"].to_numpy(dtype=np.int64)):
            raise ValueError(f"{prefix} cached predictions do not match probabilities")


def _manifest_matches_spec(
    cached: Mapping[str, object],
    spec: AgreementConfig,
) -> tuple[bool, str]:
    expected = {
        "schema_version": spec.schema_version,
        "seed": int(spec.seed),
        "joined_table_tag": spec.joined_table_tag,
        "run_tag": spec.run_tag,
        "checkpoint_name": spec.checkpoint_name,
        "modes": list(spec.modes),
        "temperature": 1.0,
        "calibrated": False,
    }
    for key, value in expected.items():
        if cached.get(key) != value:
            return False, f"manifest field {key!r} differs: {cached.get(key)!r} != {value!r}"
    return True, "matching requested specification"


def _cache_manifest_matches(
    cached: Mapping[str, object],
    live_artifacts: Mapping[str, object],
    spec: AgreementConfig,
) -> tuple[bool, str]:
    matches, reason = _manifest_matches_spec(cached, spec)
    if not matches:
        return matches, reason
    if cached.get("source_fingerprints") != live_artifacts.get("fingerprints"):
        return False, "source artifact fingerprints differ"
    return True, "matching provenance"


def load_or_build_model_outputs(
    cfg: Mapping,
    spec: AgreementConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load or create the canonical ordered model-output cache.

    ``cache_only`` validates only the cache and never resolves, hashes, or opens
    checkpoint/scaler/metrics files.
    """

    paths = analysis_paths(cfg, spec)
    output_path = paths["model_outputs"]
    manifest_path = paths["model_outputs_manifest"]
    cache_exists = output_path.exists() and manifest_path.exists()

    if spec.cache_policy == "cache_only":
        if not cache_exists:
            raise FileNotFoundError(
                f"cache_only requires {output_path} and {manifest_path}"
            )
        manifest = _read_json(manifest_path)
        matches, reason = _manifest_matches_spec(manifest, spec)
        if not matches:
            raise ValueError(f"cache_only cache does not match the requested analysis: {reason}")
        cached_fingerprint = manifest.get("model_outputs_fingerprint")
        if cached_fingerprint is not None and cached_fingerprint != _fingerprint(output_path):
            raise ValueError("model_outputs.parquet fingerprint differs from its manifest")
        frame = pd.read_parquet(output_path)
        validate_model_outputs(frame, manifest)
        return frame, manifest

    live_artifacts = validate_artifacts(cfg, spec)
    if cache_exists and spec.cache_policy == "reuse_or_build":
        manifest = _read_json(manifest_path)
        matches, reason = _cache_manifest_matches(manifest, live_artifacts, spec)
        if not matches:
            raise ValueError(
                "Existing model-output cache has mismatched provenance; use "
                f"cache_policy='rebuild' to replace it ({reason})."
            )
        frame = pd.read_parquet(output_path)
        validate_model_outputs(frame, manifest)
        return frame, manifest

    frame, manifest = run_ordered_test_inference(cfg, spec)
    _atomic_parquet(frame, output_path)
    manifest = dict(manifest)
    manifest["model_outputs_fingerprint"] = _fingerprint(output_path)
    _atomic_json(manifest, manifest_path)
    return frame, manifest


# ---------------------------------------------------------------------------
# Pure native-T=1 diagnostics and instance-level outcomes
# ---------------------------------------------------------------------------


def _as_probability_array(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("probabilities must have at least one dimension")
    if not np.isfinite(array).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(array < 0):
        raise ValueError("probabilities must be non-negative")
    return array


def normalized_entropy(
    probabilities: np.ndarray | Sequence[float],
    axis: int = -1,
) -> np.ndarray | float:
    """Shannon entropy divided by ``log(K)`` (native T=1 when used here)."""

    values = _as_probability_array(probabilities)
    classes = values.shape[axis]
    if classes <= 1:
        result = np.zeros(np.delete(values.shape, axis), dtype=np.float64)
    else:
        terms = np.where(values > 0, values * np.log(np.clip(values, 1e-300, None)), 0.0)
        result = -np.sum(terms, axis=axis) / math.log(classes)
    return float(result) if np.ndim(result) == 0 else result


def jensen_shannon_divergence(
    p: np.ndarray | Sequence[float],
    q: np.ndarray | Sequence[float],
    axis: int = -1,
    normalized: bool = True,
) -> np.ndarray | float:
    """Jensen-Shannon divergence, optionally normalized to the range [0, 1]."""

    left = _as_probability_array(p)
    right = _as_probability_array(q)
    if left.shape != right.shape:
        raise ValueError(f"JSD inputs must have equal shapes: {left.shape} != {right.shape}")
    middle = 0.5 * (left + right)

    def kl_divergence(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        terms = np.where(
            source > 0,
            source * (np.log(np.clip(source, 1e-300, None)) - np.log(np.clip(target, 1e-300, None))),
            0.0,
        )
        return np.sum(terms, axis=axis)

    result = 0.5 * (kl_divergence(left, middle) + kl_divergence(right, middle))
    if normalized:
        result = result / math.log(2.0)
    return float(result) if np.ndim(result) == 0 else result


def total_variation_distance(
    p: np.ndarray | Sequence[float],
    q: np.ndarray | Sequence[float],
    axis: int = -1,
) -> np.ndarray | float:
    """Half the L1 distance between two categorical distributions."""

    left = _as_probability_array(p)
    right = _as_probability_array(q)
    if left.shape != right.shape:
        raise ValueError(f"TV inputs must have equal shapes: {left.shape} != {right.shape}")
    result = 0.5 * np.sum(np.abs(left - right), axis=axis)
    return float(result) if np.ndim(result) == 0 else result


def top3_overlap(
    image_top3: np.ndarray | Sequence[int],
    geo_top3: np.ndarray | Sequence[int],
) -> np.ndarray | float:
    """Set overlap of top-3 predictions, divided by three."""

    left = np.asarray(image_top3)
    right = np.asarray(geo_top3)
    if left.shape != right.shape or left.shape[-1] != 3:
        raise ValueError("top3 arrays must have matching shape (..., 3)")
    one_row = left.ndim == 1
    left_2d = left.reshape(-1, 3)
    right_2d = right.reshape(-1, 3)
    overlap = np.asarray(
        [len(set(a.tolist()).intersection(b.tolist())) / 3.0 for a, b in zip(left_2d, right_2d)],
        dtype=np.float64,
    ).reshape(left.shape[:-1])
    return float(overlap) if one_row else overlap


def correctness_states(
    image_correct: np.ndarray | Sequence[bool],
    geo_correct: np.ndarray | Sequence[bool],
) -> np.ndarray | str:
    """Assign the four image/geo correctness states."""

    image = np.asarray(image_correct, dtype=bool)
    geo = np.asarray(geo_correct, dtype=bool)
    if image.shape != geo.shape:
        raise ValueError("correctness arrays must have matching shapes")
    result = np.full(image.shape, "neither_correct", dtype=object)
    result[image & geo] = "both_correct"
    result[image & ~geo] = "image_only_correct"
    result[~image & geo] = "geo_only_correct"
    return str(result.item()) if result.ndim == 0 else result.astype(str)


def f1_flow_counts(
    y_true: np.ndarray | Sequence[int],
    baseline_pred: np.ndarray | Sequence[int],
    fusion_pred: np.ndarray | Sequence[int],
    class_index: int,
) -> dict[str, int]:
    """Count the TP/FP changes caused by replacing a baseline with fusion."""

    truth = np.asarray(y_true, dtype=np.int64)
    baseline = np.asarray(baseline_pred, dtype=np.int64)
    fusion = np.asarray(fusion_pred, dtype=np.int64)
    if truth.shape != baseline.shape or truth.shape != fusion.shape:
        raise ValueError("truth and prediction arrays must have matching shapes")
    cls = int(class_index)
    positive = truth == cls
    negative = ~positive
    baseline_positive = baseline == cls
    fusion_positive = fusion == cls
    baseline_tp = int(np.sum(positive & baseline_positive))
    fusion_tp = int(np.sum(positive & fusion_positive))
    baseline_fp = int(np.sum(negative & baseline_positive))
    fusion_fp = int(np.sum(negative & fusion_positive))
    return {
        "baseline_tp": baseline_tp,
        "fusion_tp": fusion_tp,
        "baseline_fp": baseline_fp,
        "fusion_fp": fusion_fp,
        "tp_rescued": int(np.sum(positive & ~baseline_positive & fusion_positive)),
        "tp_lost": int(np.sum(positive & baseline_positive & ~fusion_positive)),
        "fp_introduced": int(np.sum(negative & ~baseline_positive & fusion_positive)),
        "fp_removed": int(np.sum(negative & baseline_positive & ~fusion_positive)),
    }


def _probability_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if str(column).startswith(f"{prefix}_prob_t1_c")]
    return sorted(columns)


def derive_per_instance_metrics(
    outputs: pd.DataFrame,
    class_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Derive hard and soft agreement fields without accessing a checkpoint."""

    frame = outputs.copy()
    if class_names is None:
        class_count = len(_probability_columns(frame, "image"))
        class_names = [str(index) for index in range(class_count)]
    names = [str(value) for value in class_names]
    if not names:
        raise ValueError("class_names must not be empty")
    labels = frame["class_index"].to_numpy(dtype=np.int64)

    for prefix in PREFIXES:
        prediction = frame[f"{prefix}_pred"].to_numpy(dtype=np.int64)
        frame[f"{prefix}_correct"] = prediction == labels
        frame[f"{prefix}_pred_name"] = [names[index] for index in prediction]

    image_pred = frame["image_pred"].to_numpy(dtype=np.int64)
    geo_pred = frame["geo_pred"].to_numpy(dtype=np.int64)
    fusion_pred = frame["fusion_pred"].to_numpy(dtype=np.int64)
    image_correct = frame["image_correct"].to_numpy(dtype=bool)
    geo_correct = frame["geo_correct"].to_numpy(dtype=bool)
    fusion_correct = frame["fusion_correct"].to_numpy(dtype=bool)

    agreement = image_pred == geo_pred
    disagreement = ~agreement
    state = correctness_states(image_correct, geo_correct)
    frame["image_geo_top1_agree"] = agreement
    frame["image_geo_correctness_state"] = state
    frame["both_correct"] = image_correct & geo_correct
    frame["image_only_correct"] = image_correct & ~geo_correct
    frame["geo_only_correct"] = ~image_correct & geo_correct
    frame["neither_correct"] = ~image_correct & ~geo_correct
    frame["both_wrong_same_prediction"] = (~image_correct & ~geo_correct & agreement)
    frame["both_wrong_different_prediction"] = (~image_correct & ~geo_correct & disagreement)
    frame["routing_oracle_correct"] = image_correct | geo_correct
    frame["exclusive_correctness"] = image_correct ^ geo_correct

    image_top3 = frame[["image_top3_1", "image_top3_2", "image_top3_3"]].to_numpy(dtype=np.int64)
    geo_top3 = frame[["geo_top3_1", "geo_top3_2", "geo_top3_3"]].to_numpy(dtype=np.int64)
    frame["image_geo_top3_overlap"] = top3_overlap(image_top3, geo_top3)
    frame["image_geo_top3_intersection_size"] = np.rint(
        frame["image_geo_top3_overlap"].to_numpy(dtype=float) * 3
    ).astype(np.int8)

    frame["fusion_consensus_preserved"] = agreement & (fusion_pred == image_pred)
    frame["fusion_consensus_changed"] = agreement & (fusion_pred != image_pred)
    frame["fusion_captures_image_exclusive"] = (image_correct & ~geo_correct & fusion_correct)
    frame["fusion_captures_geo_exclusive"] = (~image_correct & geo_correct & fusion_correct)
    frame["fusion_synergy"] = (~image_correct & ~geo_correct & fusion_correct)
    frame["fusion_missed_oracle"] = ((image_correct | geo_correct) & ~fusion_correct)
    frame["fusion_negative_transfer"] = frame["fusion_missed_oracle"]
    frame["fusion_selects_image_on_disagreement"] = disagreement & (fusion_pred == image_pred)
    frame["fusion_selects_geo_on_disagreement"] = disagreement & (fusion_pred == geo_pred)
    frame["fusion_selects_third_class_on_disagreement"] = disagreement & (
        (fusion_pred != image_pred) & (fusion_pred != geo_pred)
    )

    image_probability = frame[_probability_columns(frame, "image")].to_numpy(dtype=np.float64)
    geo_probability = frame[_probability_columns(frame, "geo")].to_numpy(dtype=np.float64)
    frame["image_geo_jsd_native_t1"] = jensen_shannon_divergence(
        image_probability, geo_probability, normalized=True
    )
    frame["image_geo_tv_native_t1"] = total_variation_distance(image_probability, geo_probability)
    frame["geo_minus_image_true_class_nll_native_t1"] = (
        frame["geo_nll_t1"].to_numpy(dtype=float) - frame["image_nll_t1"].to_numpy(dtype=float)
    )
    return frame


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _cohen_kappa_or_na(left: Sequence[int], right: Sequence[int]) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.size == 0 or np.unique(np.concatenate([left_array, right_array])).size < 2:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(cohen_kappa_score(left_array, right_array))
    return value if np.isfinite(value) else math.nan


def _rate_row(
    metric: str,
    numerator: int | float,
    denominator: int | float,
    *,
    unit: str = "rate",
    probability_basis: str | None = None,
) -> dict[str, object]:
    return {
        "metric": metric,
        "value": _safe_rate(numerator, denominator),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "unit": unit,
        "probability_basis": probability_basis,
    }


def _value_row(
    metric: str,
    value: float,
    *,
    denominator: int | None = None,
    unit: str = "value",
    probability_basis: str | None = None,
) -> dict[str, object]:
    return {
        "metric": metric,
        "value": float(value),
        "numerator": math.nan,
        "denominator": float(denominator) if denominator is not None else math.nan,
        "unit": unit,
        "probability_basis": probability_basis,
    }


def _binary_class_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_index: int,
) -> dict[str, float | int]:
    positive = truth == class_index
    predicted = prediction == class_index
    support = int(positive.sum())
    tp = int(np.sum(positive & predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    if support == 0:
        return {
            "support": 0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
        }
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else 0.0
    )
    return {
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _overall_table(frame: pd.DataFrame) -> pd.DataFrame:
    count = len(frame)
    image_correct = frame["image_correct"].to_numpy(dtype=bool)
    geo_correct = frame["geo_correct"].to_numpy(dtype=bool)
    fusion_correct = frame["fusion_correct"].to_numpy(dtype=bool)
    state = frame["image_geo_correctness_state"].astype(str)
    rows: list[dict[str, object]] = [
        _rate_row("image_accuracy", image_correct.sum(), count),
        _rate_row("geo_accuracy", geo_correct.sum(), count),
        _rate_row("fusion_accuracy", fusion_correct.sum(), count),
        _rate_row("image_geo_top1_agreement", frame["image_geo_top1_agree"].sum(), count),
        _value_row(
            "image_geo_cohen_kappa",
            _cohen_kappa_or_na(frame["image_pred"], frame["geo_pred"]),
            denominator=count,
        ),
        _value_row(
            "mean_top3_overlap",
            frame["image_geo_top3_overlap"].mean(),
            denominator=count,
        ),
        _rate_row("both_correct_rate", (state == "both_correct").sum(), count),
        _rate_row("image_exclusive_rate", (state == "image_only_correct").sum(), count),
        _rate_row("geo_exclusive_rate", (state == "geo_only_correct").sum(), count),
        _rate_row("double_fault_rate", (state == "neither_correct").sum(), count),
        _rate_row("routing_oracle_accuracy", frame["routing_oracle_correct"].sum(), count),
        _rate_row(
            "exploitable_complementarity",
            int(frame["routing_oracle_correct"].sum())
            - max(int(image_correct.sum()), int(geo_correct.sum())),
            count,
        ),
        _rate_row("exclusive_correctness_rate", frame["exclusive_correctness"].sum(), count),
        _rate_row("fusion_synergy_rate", frame["fusion_synergy"].sum(), count),
        _rate_row("fusion_missed_oracle_rate", frame["fusion_missed_oracle"].sum(), count),
        _rate_row(
            "both_wrong_same_prediction_rate",
            frame["both_wrong_same_prediction"].sum(),
            int(frame["neither_correct"].sum()),
        ),
        _value_row(
            "mean_image_geo_jsd_native_t1",
            frame["image_geo_jsd_native_t1"].mean(),
            denominator=count,
            probability_basis=PROBABILITY_BASIS,
        ),
        _value_row(
            "mean_image_geo_tv_native_t1",
            frame["image_geo_tv_native_t1"].mean(),
            denominator=count,
            probability_basis=PROBABILITY_BASIS,
        ),
    ]
    return pd.DataFrame(rows)


def _correctness_state_table(frame: pd.DataFrame) -> pd.DataFrame:
    order = ["both_correct", "image_only_correct", "geo_only_correct", "neither_correct"]
    counts = frame["image_geo_correctness_state"].value_counts()
    return pd.DataFrame(
        {
            "state": order,
            "count": [int(counts.get(value, 0)) for value in order],
            "rate": [_safe_rate(int(counts.get(value, 0)), len(frame)) for value in order],
        }
    )


def _fusion_capture_table(frame: pd.DataFrame) -> pd.DataFrame:
    oracle_count = int(frame["routing_oracle_correct"].sum())
    missed_count = int(frame["fusion_missed_oracle"].sum())
    synergy_count = int(frame["fusion_synergy"].sum())
    if int(frame["fusion_correct"].sum()) != oracle_count - missed_count + synergy_count:
        raise AssertionError("Fusion accuracy identity U - M + S is violated")
    definitions = [
        ("consensus_preservation", "fusion_consensus_preserved", "image_geo_top1_agree"),
        ("image_exclusive_capture", "fusion_captures_image_exclusive", "image_only_correct"),
        ("geo_exclusive_capture", "fusion_captures_geo_exclusive", "geo_only_correct"),
        ("synergy_rate", "fusion_synergy", "__all__"),
        ("conditional_synergy", "fusion_synergy", "neither_correct"),
        ("missed_oracle_rate", "fusion_missed_oracle", "__all__"),
        ("conditional_negative_transfer", "fusion_negative_transfer", "routing_oracle_correct"),
        (
            "selects_image_when_baselines_disagree",
            "fusion_selects_image_on_disagreement",
            None,
        ),
        (
            "selects_geo_when_baselines_disagree",
            "fusion_selects_geo_on_disagreement",
            None,
        ),
        (
            "selects_third_class_when_baselines_disagree",
            "fusion_selects_third_class_on_disagreement",
            None,
        ),
    ]
    disagreement = int((~frame["image_geo_top1_agree"]).sum())
    rows = []
    for metric, numerator_col, denominator_col in definitions:
        numerator = int(frame[numerator_col].sum())
        if denominator_col is None:
            denominator = disagreement
        elif denominator_col == "__all__":
            denominator = len(frame)
        else:
            denominator = int(frame[denominator_col].sum())
        rows.append(
            {
                "metric": metric,
                "value": _safe_rate(numerator, denominator),
                "numerator": numerator,
                "denominator": denominator,
            }
        )
    return pd.DataFrame(rows)


def _per_habitat_table(
    frame: pd.DataFrame,
    class_names: Sequence[str],
) -> pd.DataFrame:
    truth = frame["class_index"].to_numpy(dtype=np.int64)
    plot_values = frame["plot_idx"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for class_index, label_name in enumerate(class_names):
        mask = truth == class_index
        support = int(mask.sum())
        plots = int(np.unique(plot_values[mask]).size) if support else 0
        row: dict[str, object] = {
            "class_index": class_index,
            "label_name": str(label_name),
            "support": support,
            "plots": plots,
            "low_support": bool(support < 20 or plots < 10),
        }
        for prefix in PREFIXES:
            metrics = _binary_class_metrics(
                truth,
                frame[f"{prefix}_pred"].to_numpy(dtype=np.int64),
                class_index,
            )
            for metric in ("precision", "recall", "f1", "tp", "fp", "fn"):
                row[f"{prefix}_{metric}"] = metrics[metric]
        if support:
            subset = frame.loc[mask]
            image_exclusive = int(subset["image_only_correct"].sum())
            geo_exclusive = int(subset["geo_only_correct"].sum())
            neither = int(subset["neither_correct"].sum())
            oracle = int(subset["routing_oracle_correct"].sum())
            consensus = int(subset["image_geo_top1_agree"].sum())
            disagreement = support - consensus
            both_wrong_same = int(subset["both_wrong_same_prediction"].sum())
            both_wrong_different = int(subset["both_wrong_different_prediction"].sum())
            row.update(
                {
                    "image_geo_top1_agreement": float(subset["image_geo_top1_agree"].mean()),
                    "image_geo_cohen_kappa": float(
                        _cohen_kappa_or_na(subset["image_pred"], subset["geo_pred"])
                    ),
                    "mean_top3_overlap": float(subset["image_geo_top3_overlap"].mean()),
                    "double_fault_rate": float(subset["neither_correct"].mean()),
                    "routing_oracle_accuracy": float(subset["routing_oracle_correct"].mean()),
                    "fusion_accuracy": float(subset["fusion_correct"].mean()),
                    "mean_jsd_native_t1": float(subset["image_geo_jsd_native_t1"].mean()),
                    "mean_tv_native_t1": float(subset["image_geo_tv_native_t1"].mean()),
                    "mean_geo_minus_image_true_class_nll_native_t1": float(
                        subset["geo_minus_image_true_class_nll_native_t1"].mean()
                    ),
                    "both_correct_rate": float(subset["both_correct"].mean()),
                    "image_only_correct_rate": _safe_rate(image_exclusive, support),
                    "geo_only_correct_rate": _safe_rate(geo_exclusive, support),
                    "neither_correct_rate": _safe_rate(neither, support),
                    "both_wrong_same_prediction_rate": _safe_rate(both_wrong_same, support),
                    "both_wrong_different_prediction_rate": _safe_rate(
                        both_wrong_different, support
                    ),
                    "same_prediction_given_both_wrong": _safe_rate(
                        both_wrong_same, neither
                    ),
                    "different_prediction_given_both_wrong": _safe_rate(
                        both_wrong_different, neither
                    ),
                    "fusion_consensus_preservation": _safe_rate(
                        int(subset["fusion_consensus_preserved"].sum()), consensus
                    ),
                    "fusion_image_exclusive_capture": _safe_rate(
                        int(subset["fusion_captures_image_exclusive"].sum()), image_exclusive
                    ),
                    "fusion_geo_exclusive_capture": _safe_rate(
                        int(subset["fusion_captures_geo_exclusive"].sum()), geo_exclusive
                    ),
                    "fusion_synergy_rate": float(subset["fusion_synergy"].mean()),
                    "fusion_conditional_synergy": _safe_rate(
                        int(subset["fusion_synergy"].sum()), neither
                    ),
                    "fusion_missed_oracle_rate": float(subset["fusion_missed_oracle"].mean()),
                    "fusion_conditional_negative_transfer": _safe_rate(
                        int(subset["fusion_missed_oracle"].sum()), oracle
                    ),
                    "fusion_selects_image_on_disagreement": _safe_rate(
                        int(subset["fusion_selects_image_on_disagreement"].sum()), disagreement
                    ),
                    "fusion_selects_geo_on_disagreement": _safe_rate(
                        int(subset["fusion_selects_geo_on_disagreement"].sum()), disagreement
                    ),
                    "fusion_selects_third_class_on_disagreement": _safe_rate(
                        int(subset["fusion_selects_third_class_on_disagreement"].sum()), disagreement
                    ),
                }
            )
            for prefix in PREFIXES:
                row[f"{prefix}_mean_confidence_native_t1"] = float(
                    subset[f"{prefix}_confidence_t1"].mean()
                )
                row[f"{prefix}_mean_entropy_normalized_native_t1"] = float(
                    subset[f"{prefix}_entropy_normalized_t1"].mean()
                )
                row[f"{prefix}_mean_top2_margin_native_t1"] = float(
                    subset[f"{prefix}_top2_margin_t1"].mean()
                )
                row[f"{prefix}_mean_true_class_nll_native_t1"] = float(
                    subset[f"{prefix}_nll_t1"].mean()
                )
        else:
            row.update(
                {
                    "image_geo_top1_agreement": math.nan,
                    "image_geo_cohen_kappa": math.nan,
                    "mean_top3_overlap": math.nan,
                    "double_fault_rate": math.nan,
                    "routing_oracle_accuracy": math.nan,
                    "fusion_accuracy": math.nan,
                    "mean_jsd_native_t1": math.nan,
                    "mean_tv_native_t1": math.nan,
                    "mean_geo_minus_image_true_class_nll_native_t1": math.nan,
                    "both_correct_rate": math.nan,
                    "image_only_correct_rate": math.nan,
                    "geo_only_correct_rate": math.nan,
                    "neither_correct_rate": math.nan,
                    "both_wrong_same_prediction_rate": math.nan,
                    "both_wrong_different_prediction_rate": math.nan,
                    "same_prediction_given_both_wrong": math.nan,
                    "different_prediction_given_both_wrong": math.nan,
                    "fusion_consensus_preservation": math.nan,
                    "fusion_image_exclusive_capture": math.nan,
                    "fusion_geo_exclusive_capture": math.nan,
                    "fusion_synergy_rate": math.nan,
                    "fusion_conditional_synergy": math.nan,
                    "fusion_missed_oracle_rate": math.nan,
                    "fusion_conditional_negative_transfer": math.nan,
                    "fusion_selects_image_on_disagreement": math.nan,
                    "fusion_selects_geo_on_disagreement": math.nan,
                    "fusion_selects_third_class_on_disagreement": math.nan,
                }
            )
            for prefix in PREFIXES:
                row[f"{prefix}_mean_confidence_native_t1"] = math.nan
                row[f"{prefix}_mean_entropy_normalized_native_t1"] = math.nan
                row[f"{prefix}_mean_top2_margin_native_t1"] = math.nan
                row[f"{prefix}_mean_true_class_nll_native_t1"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _f1_flow_table(
    frame: pd.DataFrame,
    class_names: Sequence[str],
    baseline_prefix: str,
) -> pd.DataFrame:
    truth = frame["class_index"].to_numpy(dtype=np.int64)
    baseline = frame[f"{baseline_prefix}_pred"].to_numpy(dtype=np.int64)
    fusion = frame["fusion_pred"].to_numpy(dtype=np.int64)
    rows = []
    for class_index, label_name in enumerate(class_names):
        baseline_metrics = _binary_class_metrics(truth, baseline, class_index)
        fusion_metrics = _binary_class_metrics(truth, fusion, class_index)
        flow = f1_flow_counts(truth, baseline, fusion, class_index)
        if flow["baseline_tp"] + flow["tp_rescued"] - flow["tp_lost"] != flow["fusion_tp"]:
            raise AssertionError("TP flow conservation identity is violated")
        if flow["baseline_fp"] + flow["fp_introduced"] - flow["fp_removed"] != flow["fusion_fp"]:
            raise AssertionError("FP flow conservation identity is violated")
        support = int(baseline_metrics["support"])
        rows.append(
            {
                "class_index": class_index,
                "label_name": str(label_name),
                "support": support,
                "baseline": baseline_prefix,
                "baseline_precision": baseline_metrics["precision"],
                "baseline_recall": baseline_metrics["recall"],
                "baseline_f1": baseline_metrics["f1"],
                "fusion_precision": fusion_metrics["precision"],
                "fusion_recall": fusion_metrics["recall"],
                "fusion_f1": fusion_metrics["f1"],
                "f1_delta": (
                    float(fusion_metrics["f1"] - baseline_metrics["f1"])
                    if support
                    else math.nan
                ),
                **flow,
                "tp_delta": flow["tp_rescued"] - flow["tp_lost"],
                "fp_delta": flow["fp_introduced"] - flow["fp_removed"],
            }
        )
    return pd.DataFrame(rows)


def _prediction_pair_table(frame: pd.DataFrame, class_names: Sequence[str]) -> pd.DataFrame:
    pair = (
        frame.groupby(["image_pred", "geo_pred"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    full = pd.MultiIndex.from_product(
        [range(len(class_names)), range(len(class_names))],
        names=["image_pred", "geo_pred"],
    ).to_frame(index=False)
    table = full.merge(pair, on=["image_pred", "geo_pred"], how="left")
    table["count"] = table["count"].fillna(0).astype(int)
    table["rate"] = table["count"] / len(frame)
    table["image_pred_name"] = table["image_pred"].map(dict(enumerate(class_names)))
    table["geo_pred_name"] = table["geo_pred"].map(dict(enumerate(class_names)))
    return table


def _soft_diagnostic_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "image_geo_jsd_native_t1",
        "image_geo_tv_native_t1",
        "geo_minus_image_true_class_nll_native_t1",
        "image_confidence_t1",
        "geo_confidence_t1",
        "fusion_confidence_t1",
        "image_entropy_normalized_t1",
        "geo_entropy_normalized_t1",
        "fusion_entropy_normalized_t1",
        "image_top2_margin_t1",
        "geo_top2_margin_t1",
        "fusion_top2_margin_t1",
    ]
    groups: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    groups.extend(
        (state, subset)
        for state, subset in frame.groupby("image_geo_correctness_state", sort=False)
    )
    groups.extend(
        [("fusion_correct", frame[frame["fusion_correct"]]), ("fusion_wrong", frame[~frame["fusion_correct"]])]
    )
    rows = []
    for group_name, subset in groups:
        for metric in metrics:
            values = subset[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "probability_basis": PROBABILITY_BASIS,
                    "count": int(len(finite)),
                    "mean": float(np.mean(finite)) if len(finite) else math.nan,
                    "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan,
                    "q05": float(np.quantile(finite, 0.05)) if len(finite) else math.nan,
                    "median": float(np.median(finite)) if len(finite) else math.nan,
                    "q95": float(np.quantile(finite, 0.95)) if len(finite) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _priority_sample_table(frame: pd.DataFrame, limit_per_category: int = 25) -> pd.DataFrame:
    categories = {
        "synergy_rescue": frame["fusion_synergy"],
        "image_exclusive_capture": frame["fusion_captures_image_exclusive"],
        "geo_exclusive_capture": frame["fusion_captures_geo_exclusive"],
        "missed_oracle": frame["fusion_missed_oracle"],
        "consensus_changed": frame["fusion_consensus_changed"],
        "third_class_choice": frame["fusion_selects_third_class_on_disagreement"],
    }
    identity = [
        "row_index",
        "file",
        "file_normalized",
        "plot_idx",
        "label_name",
        "class_index",
        "image_pred",
        "image_pred_name",
        "geo_pred",
        "geo_pred_name",
        "fusion_pred",
        "fusion_pred_name",
        "image_geo_correctness_state",
        "image_geo_jsd_native_t1",
        "image_geo_tv_native_t1",
        "image_nll_t1",
        "geo_nll_t1",
        "fusion_nll_t1",
    ]
    if "image_source" in frame.columns:
        identity.insert(5, "image_source")
    pieces = []
    for category, mask in categories.items():
        subset = frame.loc[mask, identity].copy()
        subset.insert(0, "category", category)
        subset = subset.sort_values(
            ["image_geo_jsd_native_t1", "fusion_nll_t1"],
            ascending=[False, False],
        ).head(limit_per_category)
        subset.insert(1, "rank_within_category", np.arange(1, len(subset) + 1))
        pieces.append(subset)
    if not pieces:
        return pd.DataFrame(columns=["category", "rank_within_category", *identity])
    return pd.concat(pieces, ignore_index=True)


def build_analysis_tables(
    per_instance: pd.DataFrame,
    manifest: Mapping[str, object],
) -> dict[str, pd.DataFrame]:
    """Build all reproducible tabular summaries from the instance cache."""

    class_names = [str(value) for value in manifest.get("class_names", [])]
    if not class_names:
        raise ValueError("manifest must contain class_names")
    return {
        "overall": _overall_table(per_instance),
        "prediction_pair": _prediction_pair_table(per_instance, class_names),
        "correctness_state": _correctness_state_table(per_instance),
        "fusion_capture": _fusion_capture_table(per_instance),
        "soft_diagnostic": _soft_diagnostic_table(per_instance),
        "per_habitat": _per_habitat_table(per_instance, class_names),
        "f1_flow_vs_image": _f1_flow_table(per_instance, class_names, "image"),
        "f1_flow_vs_geo": _f1_flow_table(per_instance, class_names, "geo"),
        "priority_sample": _priority_sample_table(per_instance),
    }


# ---------------------------------------------------------------------------
# Paired habitat-stratified plot-cluster bootstrap
# ---------------------------------------------------------------------------


def _weighted_f1_from_predictions(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_count: int,
) -> float:
    cm = confusion_matrix(truth, prediction, labels=np.arange(class_count))
    support = cm.sum(axis=1).astype(float)
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(class_count), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(class_count), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count),
        where=(precision + recall) != 0,
    )
    return float(np.dot(f1, support) / support.sum()) if support.sum() else math.nan


def _bootstrap_headline_values(
    frame: pd.DataFrame,
    class_count: int,
) -> dict[str, float]:
    n = len(frame)
    image_correct = frame["image_correct"].to_numpy(dtype=bool)
    geo_correct = frame["geo_correct"].to_numpy(dtype=bool)
    fusion_correct = frame["fusion_correct"].to_numpy(dtype=bool)
    agree = frame["image_geo_top1_agree"].to_numpy(dtype=bool)
    neither = ~image_correct & ~geo_correct
    oracle = image_correct | geo_correct
    image_exclusive = image_correct & ~geo_correct
    geo_exclusive = ~image_correct & geo_correct
    disagreement = ~agree
    synergy = neither & fusion_correct
    missed = oracle & ~fusion_correct
    image_pred = frame["image_pred"].to_numpy(dtype=np.int64)
    geo_pred = frame["geo_pred"].to_numpy(dtype=np.int64)
    fusion_pred = frame["fusion_pred"].to_numpy(dtype=np.int64)
    truth = frame["class_index"].to_numpy(dtype=np.int64)
    oracle_count = int(oracle.sum())
    missed_count = int(missed.sum())
    synergy_count = int(synergy.sum())
    if int(fusion_correct.sum()) != oracle_count - missed_count + synergy_count:
        raise AssertionError("Bootstrap fusion identity U - M + S is violated")

    values = {
        "image_accuracy": float(image_correct.mean()),
        "geo_accuracy": float(geo_correct.mean()),
        "fusion_accuracy": float(fusion_correct.mean()),
        "image_geo_top1_agreement": float(agree.mean()),
        "image_geo_cohen_kappa": _cohen_kappa_or_na(image_pred, geo_pred),
        "mean_top3_overlap": float(frame["image_geo_top3_overlap"].mean()),
        "double_fault_rate": float(neither.mean()),
        "routing_oracle_accuracy": float(oracle.mean()),
        "exploitable_complementarity": float(
            oracle.mean() - max(image_correct.mean(), geo_correct.mean())
        ),
        "fusion_synergy_rate": float(synergy.mean()),
        "fusion_missed_oracle_rate": float(missed.mean()),
        "fusion_consensus_preservation": _safe_rate(
            int((agree & (fusion_pred == image_pred)).sum()), int(agree.sum())
        ),
        "fusion_image_exclusive_capture": _safe_rate(
            int((image_exclusive & fusion_correct).sum()), int(image_exclusive.sum())
        ),
        "fusion_geo_exclusive_capture": _safe_rate(
            int((geo_exclusive & fusion_correct).sum()), int(geo_exclusive.sum())
        ),
        "fusion_conditional_synergy": _safe_rate(int(synergy.sum()), int(neither.sum())),
        "fusion_conditional_negative_transfer": _safe_rate(int(missed.sum()), int(oracle.sum())),
        "fusion_selects_image_on_disagreement": _safe_rate(
            int((disagreement & (fusion_pred == image_pred)).sum()), int(disagreement.sum())
        ),
        "fusion_selects_geo_on_disagreement": _safe_rate(
            int((disagreement & (fusion_pred == geo_pred)).sum()), int(disagreement.sum())
        ),
        "fusion_selects_third_class_on_disagreement": _safe_rate(
            int((disagreement & (fusion_pred != image_pred) & (fusion_pred != geo_pred)).sum()),
            int(disagreement.sum()),
        ),
        "weighted_f1_delta_fusion_vs_image": _weighted_f1_from_predictions(
            truth, fusion_pred, class_count
        )
        - _weighted_f1_from_predictions(truth, image_pred, class_count),
        "weighted_f1_delta_fusion_vs_geo": _weighted_f1_from_predictions(
            truth, fusion_pred, class_count
        )
        - _weighted_f1_from_predictions(truth, geo_pred, class_count),
    }
    return values


def _bootstrap_class_f1_values(
    frame: pd.DataFrame,
    class_count: int,
) -> dict[tuple[int, str], float]:
    truth = frame["class_index"].to_numpy(dtype=np.int64)
    fusion = frame["fusion_pred"].to_numpy(dtype=np.int64)
    result: dict[tuple[int, str], float] = {}
    for class_index in range(class_count):
        support = int((truth == class_index).sum())
        for prefix in ("image", "geo"):
            if support == 0:
                result[(class_index, f"f1_delta_fusion_vs_{prefix}")] = math.nan
                continue
            baseline = frame[f"{prefix}_pred"].to_numpy(dtype=np.int64)
            baseline_f1 = _binary_class_metrics(truth, baseline, class_index)["f1"]
            fusion_f1 = _binary_class_metrics(truth, fusion, class_index)["f1"]
            result[(class_index, f"f1_delta_fusion_vs_{prefix}")] = float(
                fusion_f1 - baseline_f1
            )
    return result


def bootstrap_uncertainty(
    per_instance: pd.DataFrame,
    class_names: Sequence[str] | None = None,
    n_replicates: int | None = None,
    seed: int | None = None,
    min_valid_fraction: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap complete plots within each habitat using paired replicates.

    Intervals describe sampling uncertainty for the current test plots.  They do
    not describe training-seed variability.
    """

    replicates = int(2000 if n_replicates is None else n_replicates)
    random_seed = int(20260714 if seed is None else seed)
    if replicates <= 0:
        raise ValueError("n_replicates must be positive")
    if not (0 < min_valid_fraction <= 1):
        raise ValueError("min_valid_fraction must be in (0, 1]")
    if "plot_idx" not in per_instance:
        raise ValueError("plot_idx is required for cluster bootstrap")

    truth = per_instance["class_index"].to_numpy(dtype=np.int64)
    class_count = int(truth.max() + 1) if class_names is None else len(class_names)
    names = (
        [str(index) for index in range(class_count)]
        if class_names is None
        else [str(value) for value in class_names]
    )
    plot_frame = per_instance[["plot_idx", "class_index"]].copy()
    plot_frame["plot_idx"] = plot_frame["plot_idx"].astype(str)
    label_counts = plot_frame.groupby("plot_idx")["class_index"].nunique()
    if bool((label_counts > 1).any()):
        examples = label_counts[label_counts > 1].index.tolist()[:10]
        raise ValueError(f"plot_idx spans multiple habitats and cannot be stratified: {examples}")

    indices_by_plot = {
        str(plot): np.asarray(indices, dtype=np.int64)
        for plot, indices in plot_frame.groupby("plot_idx", sort=False).indices.items()
    }
    plots_by_class: dict[int, list[str]] = {}
    for plot, group in plot_frame.groupby("plot_idx", sort=False):
        cls = int(group["class_index"].iloc[0])
        plots_by_class.setdefault(cls, []).append(str(plot))
    class_plot_lists = [plots_by_class[key] for key in sorted(plots_by_class)]

    conditional_metrics = {
        "fusion_consensus_preservation",
        "fusion_image_exclusive_capture",
        "fusion_geo_exclusive_capture",
        "fusion_conditional_synergy",
        "fusion_conditional_negative_transfer",
        "fusion_selects_image_on_disagreement",
        "fusion_selects_geo_on_disagreement",
        "fusion_selects_third_class_on_disagreement",
    }
    headline_estimate = _bootstrap_headline_values(per_instance, class_count)
    class_estimate = _bootstrap_class_f1_values(per_instance, class_count)
    headline_samples = {key: np.full(replicates, np.nan) for key in headline_estimate}
    class_samples = {
        key: np.full(replicates, np.nan) for key in class_estimate
    }
    rng = np.random.default_rng(random_seed)
    for replicate in range(replicates):
        sampled_indices: list[np.ndarray] = []
        for plots in class_plot_lists:
            sampled_plots = rng.choice(plots, size=len(plots), replace=True)
            sampled_indices.extend(indices_by_plot[str(plot)] for plot in sampled_plots)
        indices = np.concatenate(sampled_indices)
        sample = per_instance.iloc[indices]
        for metric, value in _bootstrap_headline_values(sample, class_count).items():
            headline_samples[metric][replicate] = value
        for key, value in _bootstrap_class_f1_values(sample, class_count).items():
            class_samples[key][replicate] = value

    rows: list[dict[str, object]] = []

    def append_interval(
        scope: str,
        metric: str,
        estimate: float,
        values: np.ndarray,
        *,
        class_index: int | None = None,
        label_name: str | None = None,
        conditional: bool = False,
    ) -> None:
        finite = values[np.isfinite(values)]
        valid_fraction = len(finite) / replicates
        interval_available = bool(len(finite) and (not conditional or valid_fraction >= min_valid_fraction))
        rows.append(
            {
                "scope": scope,
                "class_index": class_index,
                "label_name": label_name,
                "metric": metric,
                "estimate": estimate,
                "ci_lower": float(np.quantile(finite, 0.025)) if interval_available else math.nan,
                "ci_upper": float(np.quantile(finite, 0.975)) if interval_available else math.nan,
                "confidence_level": 0.95,
                "n_replicates": replicates,
                "n_valid_replicates": int(len(finite)),
                "valid_fraction": valid_fraction,
                "conditional": conditional,
                "interval_available": interval_available,
                "resampling_unit": "plot_idx",
                "stratified_by": "class_index",
                "bootstrap_seed": random_seed,
            }
        )

    for metric, estimate in headline_estimate.items():
        append_interval(
            "overall",
            metric,
            estimate,
            headline_samples[metric],
            conditional=metric in conditional_metrics,
        )
    for (class_index, metric), estimate in class_estimate.items():
        append_interval(
            "per_habitat",
            metric,
            estimate,
            class_samples[(class_index, metric)],
            class_index=class_index,
            label_name=names[class_index],
        )

    result = pd.DataFrame(rows)
    result.attrs.update(
        {
            "n_replicates": replicates,
            "seed": random_seed,
            "paired": True,
            "habitat_stratified": True,
            "cluster": "plot_idx",
            "interpretation": "current-test plot uncertainty, not training-seed variability",
        }
    )
    return result


# ---------------------------------------------------------------------------
# Reproducible CSV/figure/Markdown report export
# ---------------------------------------------------------------------------


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _load_plotting() -> Any:
    import matplotlib

    # Safe in scripts and no-op when a notebook backend was selected already.
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt = _load_plotting()
    plt.close(fig)


def _plot_prediction_contingency(
    table: pd.DataFrame,
    class_names: Sequence[str],
    path: Path,
) -> None:
    plt = _load_plotting()
    matrix = table.pivot(index="image_pred", columns="geo_pred", values="count").to_numpy()
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="magma")
    ax.set_title("Image-only vs geo-only top-1 predictions")
    ax.set_xlabel("Geo-only prediction")
    ax.set_ylabel("Image-only prediction")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
    fig.colorbar(image, ax=ax, label="Images")
    _save_figure(fig, path)


def _plot_correctness_states(table: pd.DataFrame, path: Path) -> None:
    plt = _load_plotting()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(table["state"], table["rate"] * 100, color=["#2e7d32", "#1565c0", "#ef6c00", "#c62828"])
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel("Test images (%)")
    ax.set_title("Image/geo correctness complementarity")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(5, float(table["rate"].max() * 115)))
    _save_figure(fig, path)


def _plot_fusion_capture_heatmap(table: pd.DataFrame, path: Path) -> None:
    plt = _load_plotting()
    metrics = [
        "fusion_consensus_preservation",
        "fusion_image_exclusive_capture",
        "fusion_geo_exclusive_capture",
        "fusion_conditional_synergy",
        "fusion_conditional_negative_transfer",
    ]
    matrix = table[metrics].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, max(6, len(table) * 0.35)))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), [value.replace("fusion_", "").replace("_", " ") for value in metrics], rotation=35, ha="right")
    ax.set_yticks(range(len(table)), table["label_name"], fontsize=8)
    ax.set_title("Fusion outcome rates by ground-truth habitat")
    fig.colorbar(image, ax=ax, label="Conditional rate")
    _save_figure(fig, path)


def _plot_f1_delta(
    image_flow: pd.DataFrame,
    geo_flow: pd.DataFrame,
    path: Path,
) -> None:
    plt = _load_plotting()
    positions = np.arange(len(image_flow))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(positions - width / 2, image_flow["f1_delta"] * 100, height=width, label="Fusion - image")
    ax.barh(positions + width / 2, geo_flow["f1_delta"] * 100, height=width, label="Fusion - geo")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(positions, image_flow["label_name"], fontsize=8)
    ax.set_xlabel("Per-habitat F1 difference (percentage points)")
    ax.set_title("Fusion F1 change relative to each single modality")
    ax.legend()
    _save_figure(fig, path)


def _plot_tp_fp_flow(image_flow: pd.DataFrame, geo_flow: pd.DataFrame, path: Path) -> None:
    plt = _load_plotting()
    positions = np.arange(len(image_flow))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, table, title in zip(axes, (image_flow, geo_flow), ("Versus image only", "Versus geo only")):
        ax.barh(positions, table["tp_rescued"], label="TP rescued", color="#2e7d32")
        ax.barh(positions, -table["tp_lost"], label="TP lost", color="#c62828")
        ax.scatter(table["fp_removed"], positions, marker="o", label="FP removed", color="#1565c0")
        ax.scatter(-table["fp_introduced"], positions, marker="x", label="FP introduced", color="#ef6c00")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Count (loss/introduction shown negative)")
    axes[0].set_yticks(positions, image_flow["label_name"], fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Fusion true-positive and false-positive flow")
    fig.subplots_adjust(bottom=0.12)
    _save_figure(fig, path)


def _plot_jsd_distributions(frame: pd.DataFrame, path: Path) -> None:
    plt = _load_plotting()
    states = ["both_correct", "image_only_correct", "geo_only_correct", "neither_correct"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for state in states:
        values = frame.loc[
            frame["image_geo_correctness_state"] == state,
            "image_geo_jsd_native_t1",
        ].to_numpy(dtype=float)
        if len(values):
            ax.hist(values, bins=np.linspace(0, 1, 31), histtype="step", density=True, linewidth=1.5, label=f"{state} (n={len(values)})")
    ax.set_xlabel("Normalized image-geo JSD (native T=1, uncalibrated)")
    ax.set_ylabel("Density")
    ax.set_title("Soft disagreement by correctness state")
    ax.legend(fontsize=8)
    _save_figure(fig, path)


def _plot_nll_comparison(frame: pd.DataFrame, path: Path) -> None:
    plt = _load_plotting()
    values = [frame[f"{prefix}_nll_t1"].to_numpy(dtype=float) for prefix in PREFIXES]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(values, labels=["Image", "Geo", "Fusion"], showfliers=False)
    ax.set_ylabel("True-class NLL (native T=1, uncalibrated)")
    ax.set_title("Per-instance true-class negative log likelihood")
    _save_figure(fig, path)


def _plot_bootstrap_intervals(table: pd.DataFrame, path: Path) -> None:
    plt = _load_plotting()
    desired = [
        "image_accuracy",
        "geo_accuracy",
        "fusion_accuracy",
        "routing_oracle_accuracy",
        "fusion_synergy_rate",
        "fusion_missed_oracle_rate",
        "weighted_f1_delta_fusion_vs_image",
        "weighted_f1_delta_fusion_vs_geo",
    ]
    subset = table[(table["scope"] == "overall") & table["metric"].isin(desired)].copy()
    subset["order"] = subset["metric"].map({metric: index for index, metric in enumerate(desired)})
    subset = subset.sort_values("order")
    y = np.arange(len(subset))
    estimate = subset["estimate"].to_numpy(dtype=float) * 100
    lower = subset["ci_lower"].to_numpy(dtype=float) * 100
    upper = subset["ci_upper"].to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(estimate, y, xerr=np.vstack([estimate - lower, upper - estimate]), fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, subset["metric"].str.replace("_", " "))
    ax.set_xlabel("Estimate and 95% plot-cluster bootstrap interval (percentage points)")
    ax.set_title("Single-seed current-test uncertainty")
    _save_figure(fig, path)


def create_priority_image_montage(
    priority_samples: pd.DataFrame,
    output_path: str | Path,
    *,
    root_path: str | Path = ".",
    category: str | None = None,
    max_images: int = 12,
    columns: int = 4,
) -> Path | None:
    """Optionally render ranked rescue/failure images from the priority table.

    The function is deliberately best-effort: unreadable images are labelled in
    place, while an empty selection returns ``None``.  It never participates in
    metric calculation.
    """

    if max_images <= 0 or columns <= 0:
        raise ValueError("max_images and columns must be positive")
    selected = priority_samples
    if category is not None:
        selected = selected[selected["category"].astype(str) == str(category)]
    selected = selected.head(max_images)
    if selected.empty:
        return None
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError("Pillow is required for the optional image montage") from exc
    plt = _load_plotting()
    rows = int(math.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows))
    axes_array = np.atleast_1d(axes).reshape(-1)
    root = Path(root_path)
    for axis, (_, record) in zip(axes_array, selected.iterrows()):
        source = Path(str(record.get("image_source", "")))
        if not source.is_absolute():
            source = root / source
        image_path = source / str(record["file"])
        try:
            with Image.open(image_path) as image:
                axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        except (OSError, ValueError):
            axis.text(0.5, 0.5, f"Unreadable\n{image_path.name}", ha="center", va="center")
        axis.set_title(
            f"{record['category']} #{record['rank_within_category']}\n"
            f"true={record['label_name']}\n"
            f"I={record['image_pred_name']} | G={record['geo_pred_name']} | F={record['fusion_pred_name']}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes_array[len(selected) :]:
        axis.axis("off")
    destination = Path(output_path)
    _save_figure(fig, destination)
    return destination


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    work = frame[list(columns)].copy() if columns is not None else frame.copy()
    try:
        return work.to_markdown(index=False)
    except ImportError:
        return "```text\n" + work.to_string(index=False) + "\n```"


def _summary_markdown(
    tables: Mapping[str, pd.DataFrame],
    bootstrap: pd.DataFrame,
    manifest: Mapping[str, object],
) -> str:
    overall = tables["overall"].set_index("metric")
    correctness = tables["correctness_state"].copy()
    fusion_capture = tables["fusion_capture"].copy()
    soft_overall = tables["soft_diagnostic"].loc[
        tables["soft_diagnostic"]["group"] == "overall",
        ["metric", "probability_basis", "count", "mean", "std", "median", "q05", "q95"],
    ].copy()
    habitat = tables["per_habitat"].copy()
    flow_image = tables["f1_flow_vs_image"].copy()
    flow_geo = tables["f1_flow_vs_geo"].copy()
    bootstrap_overall = bootstrap[
        (bootstrap["scope"] == "overall")
        & bootstrap["metric"].isin(
            [
                "image_accuracy",
                "geo_accuracy",
                "fusion_accuracy",
                "routing_oracle_accuracy",
                "weighted_f1_delta_fusion_vs_image",
                "weighted_f1_delta_fusion_vs_geo",
            ]
        )
    ].copy()
    bootstrap_replicates = (
        int(bootstrap["n_replicates"].iloc[0]) if not bootstrap.empty else 0
    )
    for column in ("estimate", "ci_lower", "ci_upper"):
        bootstrap_overall[column] = bootstrap_overall[column] * 100
    supported = habitat[habitat["support"] > 0].copy()
    supported["fusion_minus_image_f1_pp"] = flow_image.loc[supported.index, "f1_delta"].to_numpy() * 100
    supported["fusion_minus_geo_f1_pp"] = flow_geo.loc[supported.index, "f1_delta"].to_numpy() * 100
    class_summary = supported[
        [
            "label_name",
            "support",
            "plots",
            "low_support",
            "image_f1",
            "geo_f1",
            "fusion_f1",
            "fusion_minus_image_f1_pp",
            "fusion_minus_geo_f1_pp",
        ]
    ].copy()
    for column in ("image_f1", "geo_f1", "fusion_f1"):
        class_summary[column] = class_summary[column] * 100

    lines = [
        "# Single-seed multimodal baseline agreement",
        "",
        "## Scope",
        "",
        (
            f"This report analyses {int(manifest['rows']):,} cleaned test images from "
            f"{int(manifest.get('plots') or 0):,} plots for seed {manifest['seed']}. "
            "It compares final-checkpoint image-only, geo-only, and raw-concatenation classifiers."
        ),
        "",
        "> All probability-derived diagnostics use the native T=1 softmax and are uncalibrated. They are descriptive, not calibrated-confidence claims.",
        "",
        "## Overall hard agreement and complementarity",
        "",
        _markdown_table(
            overall.reset_index(),
            ["metric", "value", "numerator", "denominator"],
        ),
        "",
        "## Correctness states",
        "",
        _markdown_table(correctness),
        "",
        "## Fusion preservation, capture, synergy, and negative transfer",
        "",
        _markdown_table(fusion_capture),
        "",
        "The unconditional identity `fusion correct = routing oracle correct - missed oracle + synergy` is checked exactly before reporting. Conditional rates are `NA` when their conditioning set is empty.",
        "",
        "## Native-T=1 soft diagnostics",
        "",
        _markdown_table(soft_overall),
        "",
        "These softmax-derived quantities are explicitly uncalibrated. Normalized JSD and total variation compare image-only with geo-only distributions; signed geo-minus-image NLL is positive when geo assigns less probability to the true class.",
        "",
        "## Paired plot-cluster bootstrap",
        "",
        (
            f"Intervals use {bootstrap_replicates:,} habitat-stratified resamples of complete plot groups. "
            "They quantify "
            "uncertainty over the current test plots, not training-seed variability."
        ),
        "",
        _markdown_table(
            bootstrap_overall,
            ["metric", "estimate", "ci_lower", "ci_upper", "n_valid_replicates"],
        ),
        "",
        "## Per-habitat F1 and support",
        "",
        _markdown_table(class_summary),
        "",
        "Habitats with fewer than 20 images or 10 plots are marked as low support. Zero-support classes retain `NA` metrics rather than being treated as zero performance.",
        "",
        "## Figures",
        "",
        "- [Prediction contingency](figures/prediction_contingency.png)",
        "- [Correctness-state bars](figures/correctness_states.png)",
        "- [Fusion-capture heatmap](figures/fusion_capture_by_habitat.png)",
        "- [Per-habitat F1 differences](figures/f1_delta_by_habitat.png)",
        "- [TP/FP flow](figures/tp_fp_flow.png)",
        "- [Native-T=1 JSD distributions](figures/jsd_distributions_native_t1.png)",
        "- [Native-T=1 true-class NLL](figures/true_class_nll_native_t1.png)",
        "- [Bootstrap intervals](figures/bootstrap_intervals.png)",
        "",
        "## Reproduction checks",
        "",
        "The ordered inference cache was required to reproduce every saved final confusion matrix exactly and saved top-1, top-3, weighted F1, and MCC within 1e-6 before this report was generated. Saved evaluator loss is intentionally not compared because it averages batch means, whereas cached per-instance NLL is sample weighted.",
        "",
    ]
    return "\n".join(lines)


def export_analysis_report(
    per_instance: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    bootstrap: pd.DataFrame,
    manifest: Mapping[str, object],
    spec: AgreementConfig,
    cfg: Mapping | None = None,
) -> dict[str, Path]:
    """Write derived cache, CSV tables, figures, and Markdown/JSON summaries."""

    if cfg is None:
        cfg = {
            "root_path": "./",
            "dataset": manifest.get("dataset", "cs"),
            "multimodal": {"output_dir": spec.output_root or "./multimodal_artifacts"},
        }
    paths = analysis_paths(cfg, spec)
    paths["analysis_dir"].mkdir(parents=True, exist_ok=True)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    _atomic_parquet(per_instance, paths["per_instance_metrics"])

    exports: dict[str, Path] = {
        "per_instance_metrics": paths["per_instance_metrics"],
        "summary_md": paths["summary_md"],
        "summary_markdown": paths["summary_md"],
        "summary_json": paths["summary_json"],
    }
    for name, table in tables.items():
        destination = paths["report_dir"] / f"{name}.csv"
        _atomic_csv(table, destination)
        exports[name] = destination
    combined_flow = pd.concat(
        [tables["f1_flow_vs_image"], tables["f1_flow_vs_geo"]],
        ignore_index=True,
    )
    combined_flow_path = paths["report_dir"] / "f1_flow.csv"
    _atomic_csv(combined_flow, combined_flow_path)
    exports["f1_flow"] = combined_flow_path
    bootstrap_path = paths["report_dir"] / "bootstrap.csv"
    _atomic_csv(bootstrap, bootstrap_path)
    exports["bootstrap"] = bootstrap_path

    class_names = [str(value) for value in manifest["class_names"]]
    figures = {
        "prediction_contingency": paths["figures_dir"] / "prediction_contingency.png",
        "correctness_states": paths["figures_dir"] / "correctness_states.png",
        "fusion_capture_by_habitat": paths["figures_dir"] / "fusion_capture_by_habitat.png",
        "f1_delta_by_habitat": paths["figures_dir"] / "f1_delta_by_habitat.png",
        "tp_fp_flow": paths["figures_dir"] / "tp_fp_flow.png",
        "jsd_distributions_native_t1": paths["figures_dir"] / "jsd_distributions_native_t1.png",
        "true_class_nll_native_t1": paths["figures_dir"] / "true_class_nll_native_t1.png",
        "bootstrap_intervals": paths["figures_dir"] / "bootstrap_intervals.png",
    }
    _plot_prediction_contingency(tables["prediction_pair"], class_names, figures["prediction_contingency"])
    _plot_correctness_states(tables["correctness_state"], figures["correctness_states"])
    _plot_fusion_capture_heatmap(tables["per_habitat"], figures["fusion_capture_by_habitat"])
    _plot_f1_delta(tables["f1_flow_vs_image"], tables["f1_flow_vs_geo"], figures["f1_delta_by_habitat"])
    _plot_tp_fp_flow(tables["f1_flow_vs_image"], tables["f1_flow_vs_geo"], figures["tp_fp_flow"])
    _plot_jsd_distributions(per_instance, figures["jsd_distributions_native_t1"])
    _plot_nll_comparison(per_instance, figures["true_class_nll_native_t1"])
    _plot_bootstrap_intervals(bootstrap, figures["bootstrap_intervals"])
    exports.update({f"figure_{name}": path for name, path in figures.items()})

    markdown = _summary_markdown(tables, bootstrap, manifest)
    _atomic_text(markdown, paths["summary_md"])
    summary_payload = {
        "schema_version": spec.schema_version,
        "analysis": {
            "dataset": manifest.get("dataset"),
            "seed": manifest.get("seed"),
            "joined_table_tag": manifest.get("joined_table_tag"),
            "run_tag": manifest.get("run_tag"),
            "rows": manifest.get("rows"),
            "plots": manifest.get("plots"),
            "temperature": 1.0,
            "calibrated": False,
            "probability_basis": PROBABILITY_BASIS,
        },
        "overall": tables["overall"].to_dict(orient="records"),
        "correctness_state": tables["correctness_state"].to_dict(orient="records"),
        "fusion_capture": tables["fusion_capture"].to_dict(orient="records"),
        "per_habitat": tables["per_habitat"].to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "reproduction_checks": manifest.get("reproduction_checks", {}),
        "uncertainty_statement": (
            "Paired habitat-stratified plot-cluster percentile intervals quantify "
            "current-test plot uncertainty, not training-seed variability."
        ),
        "files": {key: str(value) for key, value in exports.items()},
    }
    _atomic_json(_json_safe(summary_payload), paths["summary_json"])
    return exports

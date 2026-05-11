from __future__ import annotations

import argparse
import json
import sys
from ast import literal_eval
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import load_configs, set_seed
from aihab_utils.model_init import init_clip_and_text_head
from data.cs2007_soil_aligned import (
    CS2007SoilAlignedDataset,
    build_soil_aligned_splits,
    map_to_soil_model_label,
    output_dir_from_cfg,
    save_split_artifacts,
)
from data.templates import CS_CLASSNAMES
from methods.PEFT_openclip import FTOpenCLIP, _compute_text_weights_from_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune OpenCLIP on the CS2007 soil-aligned image subset.")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/cs2007_soil_aligned.yaml")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--inspect_only", action="store_true", help="Build aligned splits/loaders but skip training.")
    return parser.parse_args()


def _coerce_scalar(value: str):
    try:
        return literal_eval(value)
    except Exception:
        return value


def _set_nested(cfg: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _apply_opts(cfg: dict, opts: Sequence[str] | None) -> dict:
    if not opts:
        return cfg
    if len(opts) % 2 != 0:
        raise ValueError(f"--opts must contain KEY VALUE pairs, got: {opts}")
    for key, value in zip(opts[0::2], opts[1::2]):
        _set_nested(cfg, key, _coerce_scalar(value))
    return cfg


def _load_cfg(args) -> dict:
    base_args = SimpleNamespace(
        base_config=args.base_config,
        dataset_config=args.dataset_config,
        opts=None,
        inspect_only=args.inspect_only,
    )
    cfg = load_configs(base_args)
    return _apply_opts(cfg, args.opts)


def _build_loaders(cfg: Mapping, train_transform, test_transform, splits):
    soil_cfg = cfg["soil_aligned"]
    image_root = Path(splits.manifest["image_root"])
    batch_size = int(cfg.get("data", {}).get("batch_size", 16))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))
    shuffle = bool(cfg.get("data", {}).get("shuffle", True))
    pin_memory = torch.cuda.is_available()

    datasets = {
        "train": CS2007SoilAlignedDataset(
            splits.frames["train"],
            image_root=image_root,
            transform=train_transform,
            return_metadata=False,
        ),
        "valid": CS2007SoilAlignedDataset(
            splits.frames["valid"],
            image_root=image_root,
            transform=test_transform,
            return_metadata=True,
        ),
        "test": CS2007SoilAlignedDataset(
            splits.frames["test"],
            image_root=image_root,
            transform=test_transform,
            return_metadata=True,
        ),
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    print(
        "[aligned loaders] "
        f"train={len(datasets['train'])}, valid={len(datasets['valid'])}, test={len(datasets['test'])}, "
        f"batch_size={batch_size}, output_dir={soil_cfg.get('output_dir')}"
    )
    return loaders


def _metadata_item(metadata: Mapping, key: str, idx: int):
    value = metadata.get(key, "")
    if torch.is_tensor(value):
        item = value[idx]
        return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def _metric_dict(y_true, y_pred, labels: Sequence, top1: float | None = None, top3: float | None = None) -> dict:
    metrics = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if top1 is not None:
        metrics["top1_acc"] = float(top1)
    if top3 is not None:
        metrics["top3_acc"] = float(top3)
    return metrics


def _normalize_cm(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def _save_cm_plots(cm: np.ndarray, label_list: Sequence[str], output_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not isinstance(cm, np.ndarray):
        if torch.is_tensor(cm):
            cm = cm.detach().cpu().numpy()
        else:
            cm = np.asarray(cm)

    def _fmt(value):
        return "0" if value == 0 else f"{value:.2f}"

    def _plot(cm_plot: np.ndarray, normalized: bool) -> None:
        plt.figure(figsize=(15, 12))
        if normalized:
            annot = np.array([[_fmt(value) for value in row] for row in cm_plot])
            fmt = ""
            suffix = "_normalized"
            title = "Confusion Matrix L3 (Normalized)"
        else:
            annot = cm_plot.astype(int)
            fmt = "d"
            suffix = ""
            title = "Confusion Matrix L3"
        sns.heatmap(
            cm_plot,
            annot=annot,
            fmt=fmt,
            cmap="Blues",
            xticklabels=label_list,
            yticklabels=label_list,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_confusion_matrix{suffix}.png", dpi=200)
        plt.close()

    _plot(cm, normalized=False)
    _plot(_normalize_cm(cm), normalized=True)


def _write_cm_artifacts(cm: np.ndarray, labels: Sequence[str], output_dir: Path, prefix: str) -> None:
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
    pd.DataFrame(_normalize_cm(cm), index=labels, columns=labels).to_csv(
        output_dir / f"{prefix}_confusion_matrix_normalized.csv"
    )
    _save_cm_plots(cm, labels, output_dir, prefix)


def _compute_final_text_weights(cfg: Mapping, model, clip_bundle, device: str):
    if bool(cfg.get("finetune", {}).get("tune_text", False)):
        model.eval()
        with torch.no_grad():
            return _compute_text_weights_from_tokens(
                model=model,
                prompt_tokens=clip_bundle["prompt_tokens"].to(device),
                num_classes=len(CS_CLASSNAMES),
                num_templates=int(clip_bundle["num_templates"]),
            )
    return clip_bundle["text_weights"].to(device)


def _evaluate_and_save(
    cfg: Mapping,
    model,
    loader,
    text_weights: torch.Tensor,
    splits,
    output_dir: Path,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    text_weights = text_weights.to(device)

    rows = []
    true_ids = []
    pred_ids = []
    top1_correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets, metadata in loader:
            images = images.to(device)
            targets = targets.to(device)
            feats = model.encode_image(images)
            feats = F.normalize(feats, dim=-1)
            logits = 100.0 * feats @ text_weights
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)

            batch_size = int(targets.shape[0])
            total += batch_size
            top1_correct += int((preds == targets).sum().item())
            top3_correct += int(top3_indices.eq(targets.view(-1, 1)).any(dim=1).sum().item())
            true_ids.extend(targets.detach().cpu().tolist())
            pred_ids.extend(preds.detach().cpu().tolist())

            for idx in range(batch_size):
                true_id = int(targets[idx].item())
                pred_id = int(preds[idx].item())
                true_name = CS_CLASSNAMES[true_id]
                pred_name = CS_CLASSNAMES[pred_id]
                true_soil = map_to_soil_model_label(true_name, splits.rare_labels, splits.common_labels)
                pred_soil = map_to_soil_model_label(pred_name, splits.rare_labels, splits.common_labels)
                top3_label_ids = [int(v) for v in top3_indices[idx].detach().cpu().tolist()]
                top3_label_names = [CS_CLASSNAMES[v] for v in top3_label_ids]
                top3_label_probs = [float(v) for v in top3_probs[idx].detach().cpu().tolist()]
                rows.append(
                    {
                        "file_name": _metadata_item(metadata, "file_name", idx),
                        "plot_idx": _metadata_item(metadata, "plot_idx", idx),
                        "true_label_id": true_id,
                        "true_label_name": true_name,
                        "pred_label_id": pred_id,
                        "pred_label_name": pred_name,
                        "true_soil_label": true_soil,
                        "pred_soil_label": pred_soil,
                        "project_correct": bool(true_id == pred_id),
                        "soil_aggregated_correct": bool(true_soil == pred_soil),
                        "top3_label_ids": "|".join(str(v) for v in top3_label_ids),
                        "top3_label_names": "|".join(top3_label_names),
                        "top3_probs": "|".join(f"{v:.8f}" for v in top3_label_probs),
                    }
                )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    project_labels = list(range(len(CS_CLASSNAMES)))
    project_cm = confusion_matrix(true_ids, pred_ids, labels=project_labels)
    project_metrics = _metric_dict(
        true_ids,
        pred_ids,
        labels=project_labels,
        top1=top1_correct / max(total, 1),
        top3=top3_correct / max(total, 1),
    )
    project_metrics["labels"] = CS_CLASSNAMES
    with (output_dir / "metrics_project_20class.json").open("w", encoding="utf-8") as handle:
        json.dump(project_metrics, handle, indent=2)
    _write_cm_artifacts(project_cm, CS_CLASSNAMES, output_dir, "project_20class")

    soil_true = predictions["true_soil_label"].tolist()
    soil_pred = predictions["pred_soil_label"].tolist()
    soil_labels = list(splits.soil_label_order)
    soil_cm = confusion_matrix(soil_true, soil_pred, labels=soil_labels)
    soil_metrics = _metric_dict(soil_true, soil_pred, labels=soil_labels)
    soil_metrics["labels"] = soil_labels
    soil_metrics["other_rare_test_support_after_boundary_drop"] = int((predictions["true_soil_label"] == "Other_rare").sum())
    with (output_dir / "metrics_soil_13class_aggregated.json").open("w", encoding="utf-8") as handle:
        json.dump(soil_metrics, handle, indent=2)
    _write_cm_artifacts(soil_cm, soil_labels, output_dir, "soil_13class_aggregated")

    return {"project": project_metrics, "soil_aggregated": soil_metrics}


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args)
    set_seed(int(cfg.get("seed", 1)))

    if str(cfg.get("clip_backend", "openclip")).lower() != "openclip":
        raise ValueError("The CS2007 soil-aligned runner supports only clip_backend=openclip.")

    output_dir = output_dir_from_cfg(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_soil_aligned_splits(cfg)
    save_split_artifacts(splits, output_dir)

    clip_bundle = init_clip_and_text_head(cfg)
    backend = str(cfg.get("clip_backend", "openclip")).lower()
    use_model_preprocess = bool(cfg.get("use_model_preprocess", backend == "openclip"))
    if use_model_preprocess:
        train_transform = clip_bundle["preprocess_train"]
        test_transform = clip_bundle["preprocess_val"]
    else:
        raise ValueError("This isolated runner expects use_model_preprocess=True for OpenCLIP.")

    loaders = _build_loaders(cfg, train_transform, test_transform, splits)
    print(
        "[aligned text head] "
        f"classes={len(CS_CLASSNAMES)}, text_weights_shape={tuple(clip_bundle['text_weights'].shape)}"
    )

    if args.inspect_only:
        print("[aligned] Inspection-only run; split artifacts written and training skipped.")
        return

    finetuner = FTOpenCLIP(cfg)
    model = clip_bundle["clip_model"]
    finetuner(
        train_loader=loaders["train"],
        val_loader=loaders["valid"],
        test_loader=loaders["test"],
        text_weights=clip_bundle["text_weights"],
        model=model,
        classnames=CS_CLASSNAMES,
        shots=int(cfg.get("shots", 0) or 0),
        config_file=Path(args.dataset_config).stem,
        return_valid=False,
        prompt_tokens=clip_bundle["prompt_tokens"],
        num_templates=clip_bundle["num_templates"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_text_weights = _compute_final_text_weights(cfg, model, clip_bundle, device)
    metrics = _evaluate_and_save(cfg, model, loaders["test"], final_text_weights, splits, output_dir)
    print("\n==== CS2007 Soil-Aligned OpenCLIP artifacts ====")
    print(f"output_dir: {output_dir}")
    print(f"project top1: {metrics['project']['top1_acc']:.4f}")
    print(f"soil aggregated balanced_accuracy: {metrics['soil_aggregated']['balanced_accuracy']:.4f}")
    print(f"soil aggregated macro_f1: {metrics['soil_aggregated']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

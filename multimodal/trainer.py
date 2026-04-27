from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader

try:
    from torcheval.metrics import MulticlassConfusionMatrix, MulticlassF1Score
except ImportError:
    MulticlassConfusionMatrix = None
    MulticlassF1Score = None

from multimodal.data import FeatureTableDataset, GEO_FEATURE_COLUMNS, load_joined_splits, run_dir
from multimodal.models import ConcatFusion, IdentityEncoder, LateFusionClassifier, MLPHead


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cls_acc(output, target, topk=1):
    k = min(topk, int(output.shape[1]))
    pred = output.topk(k, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:k].reshape(-1).float().sum().item())
    return 100.0 * acc / target.shape[0]


class _FallbackWeightedF1:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.y_true = []
        self.y_pred = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.y_true.append(targets.detach().cpu())
        self.y_pred.append(preds.detach().cpu())

    def compute(self) -> torch.Tensor:
        y_true = torch.cat(self.y_true).numpy()
        y_pred = torch.cat(self.y_pred).numpy()
        labels = np.arange(self.num_classes)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        support = cm.sum(axis=1)
        tp = np.diag(cm)
        predicted = cm.sum(axis=0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=np.float64), where=predicted != 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=np.float64), where=support != 0)
        denom = precision + recall
        f1_per_class = np.divide(
            2.0 * precision * recall,
            denom,
            out=np.zeros_like(precision, dtype=np.float64),
            where=denom != 0,
        )
        total = support.sum()
        weighted = float(np.dot(f1_per_class, support) / total) if total > 0 else 0.0
        return torch.tensor(weighted, dtype=torch.float32)


class _FallbackConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.y_true = []
        self.y_pred = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.y_true.append(targets.detach().cpu())
        self.y_pred.append(preds.detach().cpu())

    def compute(self) -> torch.Tensor:
        y_true = torch.cat(self.y_true).numpy()
        y_pred = torch.cat(self.y_pred).numpy()
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))
        return torch.tensor(cm, dtype=torch.int64)


def _build_f1_metric(num_classes: int):
    if MulticlassF1Score is not None:
        return MulticlassF1Score(num_classes=num_classes, average="weighted")
    return _FallbackWeightedF1(num_classes=num_classes)


def _build_cm_metric(num_classes: int):
    if MulticlassConfusionMatrix is not None:
        return MulticlassConfusionMatrix(num_classes=num_classes)
    return _FallbackConfusionMatrix(num_classes=num_classes)


def _image_feature_columns(frame: pd.DataFrame):
    return sorted([c for c in frame.columns if c.startswith("I")])


def _compute_geo_stats(train_df: pd.DataFrame) -> Dict[str, list]:
    if train_df.empty:
        raise ValueError("Joined train split is empty after geo matching; cannot fit multimodal classifier.")
    geo = train_df[GEO_FEATURE_COLUMNS].astype(np.float32)
    mean = geo.mean(axis=0).to_numpy(dtype=np.float32)
    std = geo.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std[std == 0] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def _apply_geo_standardization(frame: pd.DataFrame, stats: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    out = frame.copy()
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    out.loc[:, GEO_FEATURE_COLUMNS] = (out[GEO_FEATURE_COLUMNS].astype(np.float32) - mean) / std
    return out


def _build_dataloader(frame: pd.DataFrame, mode: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    ds = FeatureTableDataset(frame, mode=mode)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def _build_model(cfg: Mapping, train_df: pd.DataFrame) -> LateFusionClassifier:
    mm_cfg = cfg.get("multimodal", {})
    mode = str(mm_cfg.get("fusion_mode", "raw_concat")).lower()
    hidden_dim = int(mm_cfg.get("hidden_dim", 256))
    dropout = float(mm_cfg.get("dropout", 0.1))
    image_dim = len(_image_feature_columns(train_df))
    geo_dim = len(GEO_FEATURE_COLUMNS)
    num_classes = int(train_df["label_id"].nunique())

    if mode == "image_only":
        head_in = image_dim
        fusion = None
    elif mode == "geo_only":
        head_in = geo_dim
        fusion = None
    else:
        head_in = image_dim + geo_dim
        fusion = ConcatFusion()

    head = MLPHead(head_in, hidden_dim=hidden_dim, output_dim=num_classes, dropout=dropout)
    return LateFusionClassifier(
        mode=mode,
        image_encoder=IdentityEncoder(),
        geo_encoder=IdentityEncoder(),
        head=head,
        fusion=fusion,
    )


def _reindex_labels(tables: Mapping[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], Dict[int, int], list[str]]:
    train_df = tables["train"]
    if train_df.empty:
        raise ValueError("Joined train split is empty after geo matching.")

    kept_labels = sorted(int(v) for v in train_df["label_id"].astype(int).unique().tolist())
    label_map = {old: idx for idx, old in enumerate(kept_labels)}
    class_names = []
    for old in kept_labels:
        rows = train_df[train_df["label_id"].astype(int) == old]
        class_names.append(str(rows.iloc[0]["label_name"]))

    remapped = {}
    for split, frame in tables.items():
        if frame.empty:
            raise ValueError(f"Joined {split} split is empty after geo matching.")
        unseen = sorted(set(frame["label_id"].astype(int).unique().tolist()).difference(label_map))
        if unseen:
            raise ValueError(
                f"Joined {split} split contains labels absent from the geo-matched train split: {unseen}"
            )
        out = frame.copy()
        out["label_id"] = out["label_id"].astype(int).map(label_map).astype(int)
        remapped[split] = out
    return remapped, label_map, class_names


def _evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_seen = 0
    total_batches = 0
    y_true, y_pred = [], []
    f1_metric = _build_f1_metric(num_classes=num_classes)
    cm_metric = _build_cm_metric(num_classes=num_classes)

    with torch.no_grad():
        for image_features, geo_features, targets in loader:
            image_features = image_features.to(device)
            geo_features = geo_features.to(device)
            targets = targets.to(device)
            logits = model(image_features, geo_features)
            loss = ce(logits, targets)
            total_loss += loss.item()
            total_top1 += cls_acc(logits, targets, topk=1) / 100.0 * len(targets)
            total_top3 += cls_acc(logits, targets, topk=3) / 100.0 * len(targets)
            total_seen += len(targets)
            total_batches += 1
            preds = logits.argmax(dim=1)
            y_true.append(targets.detach().cpu())
            y_pred.append(preds.detach().cpu())
            f1_metric.update(preds, targets)
            cm_metric.update(preds, targets)

    if total_seen == 0:
        raise ValueError("Attempted evaluation on an empty split.")

    y_true_np = torch.cat(y_true).numpy()
    y_pred_np = torch.cat(y_pred).numpy()
    return {
        "loss": total_loss / max(total_batches, 1),
        "top1_acc": total_top1 / max(total_seen, 1),
        "top3_acc": total_top3 / max(total_seen, 1),
        "f1": float(f1_metric.compute().item()),
        "mcc": float(matthews_corrcoef(y_true_np, y_pred_np)),
        "cm": cm_metric.compute().cpu().numpy(),
    }


def _history_payload(train_loss: float,
                     epoch: int,
                     val_metrics: Mapping[str, object],
                     test_metrics: Mapping[str, object] | None) -> Dict[str, float]:
    entry: Dict[str, float] = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        # Legacy validation keys kept for backward compatibility.
        "loss": float(val_metrics["loss"]),
        "top1_acc": float(val_metrics["top1_acc"]),
        "top3_acc": float(val_metrics["top3_acc"]),
        "f1": float(val_metrics["f1"]),
        "mcc": float(val_metrics["mcc"]),
        # Explicit validation keys for clarity.
        "val_loss": float(val_metrics["loss"]),
        "val_top1_acc": float(val_metrics["top1_acc"]),
        "val_top3_acc": float(val_metrics["top3_acc"]),
        "val_f1": float(val_metrics["f1"]),
        "val_mcc": float(val_metrics["mcc"]),
    }
    if test_metrics is not None:
        entry.update(
            {
                "test_loss": float(test_metrics["loss"]),
                "test_top1_acc": float(test_metrics["top1_acc"]),
                "test_top3_acc": float(test_metrics["top3_acc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_mcc": float(test_metrics["mcc"]),
            }
        )
    return entry


def _print_epoch_metrics(epoch: int,
                         epochs: int,
                         train_loss: float,
                         val_metrics: Mapping[str, object],
                         test_metrics: Mapping[str, object] | None) -> None:
    message = (
        f"[epoch {epoch}/{epochs}] "
        f"train_loss={train_loss:.4f} "
        f"val_loss={float(val_metrics['loss']):.4f} "
        f"val_top1={float(val_metrics['top1_acc']):.4f} "
        f"val_top3={float(val_metrics['top3_acc']):.4f} "
        f"val_f1={float(val_metrics['f1']):.4f} "
        f"val_mcc={float(val_metrics['mcc']):.4f}"
    )
    if test_metrics is not None:
        message += (
            f" test_loss={float(test_metrics['loss']):.4f}"
            f" test_top1={float(test_metrics['top1_acc']):.4f}"
            f" test_top3={float(test_metrics['top3_acc']):.4f}"
            f" test_f1={float(test_metrics['f1']):.4f}"
            f" test_mcc={float(test_metrics['mcc']):.4f}"
        )
    print(message)


def train_and_evaluate(cfg: Mapping) -> Dict[str, Path]:
    tables = load_joined_splits(cfg)
    mm_cfg = cfg.get("multimodal", {})
    mode = str(mm_cfg.get("fusion_mode", "raw_concat")).lower()

    tables, label_map, class_names = _reindex_labels(tables)
    geo_stats = _compute_geo_stats(tables["train"])
    tables = {split: _apply_geo_standardization(frame, geo_stats) for split, frame in tables.items()}

    batch_size = int(mm_cfg.get("train_batch_size", 128))
    num_workers = int(mm_cfg.get("train_num_workers", 0))
    epochs = int(mm_cfg.get("train_epoch", 50))
    patience_limit = int(mm_cfg.get("patience", 10))
    lr = float(mm_cfg.get("lr", 1e-3))
    weight_decay = float(mm_cfg.get("weight_decay", 1e-4))
    report_test_each_epoch = bool(mm_cfg.get("report_test_each_epoch", True))
    print_epoch_metrics = bool(mm_cfg.get("print_epoch_metrics", True))

    train_loader = _build_dataloader(tables["train"], mode=mode, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = _build_dataloader(tables["val"], mode=mode, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = _build_dataloader(tables["test"], mode=mode, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = _device()
    model = _build_model(cfg, tables["train"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    output_dir = run_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = output_dir / "geo_standardization.json"
    with scaler_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": geo_stats["mean"],
                "std": geo_stats["std"],
                "label_id_remap": {str(k): int(v) for k, v in label_map.items()},
                "class_names": class_names,
            },
            f,
            indent=2,
        )

    best_acc = -1.0
    patience = 0
    best_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for image_features, geo_features, targets in train_loader:
            image_features = image_features.to(device)
            geo_features = geo_features.to(device)
            targets = targets.to(device)
            logits = model(image_features, geo_features)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            total_seen += len(targets)

        val_metrics = _evaluate(model, val_loader, int(tables["train"]["label_id"].nunique()), device)
        test_metrics_epoch = None
        if report_test_each_epoch:
            test_metrics_epoch = _evaluate(model, test_loader, int(tables["train"]["label_id"].nunique()), device)

        avg_train_loss = total_loss / max(total_seen, 1)
        history.append(_history_payload(avg_train_loss, epoch + 1, val_metrics, test_metrics_epoch))
        if print_epoch_metrics:
            _print_epoch_metrics(epoch + 1, epochs, avg_train_loss, val_metrics, test_metrics_epoch)
        if val_metrics["top1_acc"] > best_acc:
            best_acc = float(val_metrics["top1_acc"])
            patience = 0
            torch.save({"model_state": model.state_dict(), "mode": mode}, best_path)
        else:
            patience += 1
            if patience > patience_limit:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    val_metrics = _evaluate(model, val_loader, int(tables["train"]["label_id"].nunique()), device)
    test_metrics = _evaluate(model, test_loader, int(tables["train"]["label_id"].nunique()), device)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "train_rows": int(len(tables["train"])),
                "val_rows": int(len(tables["val"])),
                "test_rows": int(len(tables["test"])),
                "report_test_each_epoch": report_test_each_epoch,
                "class_names": class_names,
                "label_id_remap": {str(k): int(v) for k, v in label_map.items()},
                "history": history,
                "val": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val_metrics.items()},
                "test": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in test_metrics.items()},
            },
            f,
            indent=2,
        )
    np.save(output_dir / "test_confusion_matrix.npy", test_metrics["cm"])
    return {
        "run_dir": output_dir,
        "checkpoint": best_path,
        "metrics": metrics_path,
        "scaler": scaler_path,
    }

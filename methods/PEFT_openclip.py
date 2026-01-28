import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Optional, Dict
from .method import FSCLIPmethod
from .utils import cls_acc
from aihab_utils.evaluation import L2MetricsAccumulator, ClassificationTracker
from tqdm import tqdm
from collections import defaultdict
from torcheval.metrics import MulticlassF1Score, MulticlassConfusionMatrix
from sklearn.metrics import matthews_corrcoef


def _compute_text_weights_from_tokens(model, prompt_tokens, num_classes: int, num_templates: int):
    """
    Compute per-class text weights from a flattened prompt token tensor.

    Args:
        model: OpenCLIP model with `encode_text`.
        prompt_tokens: Tensor[int] shaped [num_classes * num_templates, context_length]
        num_classes: Number of classes.
        num_templates: Number of templates per class.

    Returns:
        text_weights: Tensor[float] shaped [dim, num_classes]
    """

    expected = int(num_classes) * int(num_templates)
    if int(prompt_tokens.shape[0]) != expected:
        raise ValueError(
            f"Prompt token count mismatch: got {int(prompt_tokens.shape[0])}, "
            f"expected {expected} (= num_classes {num_classes} * num_templates {num_templates})."
        )

    text_feats = model.encode_text(prompt_tokens)
    text_feats = F.normalize(text_feats, dim=-1)
    # text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    dim = int(text_feats.shape[-1])
    text_feats = text_feats.view(num_classes, num_templates, dim)
    text_feats = text_feats.mean(dim=1)
    text_feats = F.normalize(text_feats, dim=-1)
    # text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats.t().contiguous()


def _run_validation(
    model,
    loader,
    text_weights,
    device,
    return_confusion_matrix: bool = False,
    cls_track: bool = False, 
    l2_eval_ctx: Optional[Dict] = None,
):
    model.eval()
    total_loss, total_top1, total_top3, total_seen, batches = 0.0, 0, 0, 0, 0
    y_true_all, y_pred_all = [], []
    ce = torch.nn.CrossEntropyLoss()

    num_classes = int(text_weights.shape[1])
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="weighted")
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes) if return_confusion_matrix else None
    cls_tracker = ClassificationTracker() if cls_track else None

    l2_acc = None
    if l2_eval_ctx is not None:
        l2_acc = L2MetricsAccumulator(
            l3_to_l2=l2_eval_ctx["l3_to_l2"],
            num_l2=l2_eval_ctx["num_l2"],
            reduce=l2_eval_ctx.get("reduce", "mean"),
            topk=l2_eval_ctx.get("topk", (1, 3)),
            mode=l2_eval_ctx.get("mode", "argmax"),
            return_confusion_matrix=l2_eval_ctx.get("return_confusion_matrix", False),
        )

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, targets, metadata = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
                metadata = None
            else:
                raise ValueError("Expected batch to be (images, targets) or (images, targets, metadata).")
            images, targets = images.to(device), targets.to(device)
            feats = model.encode_image(images)
            feats = F.normalize(feats, dim=-1)
            logits = 100.0 * feats @ text_weights
            loss = ce(logits, targets)

            top1 = cls_acc(logits, targets)
            top3 = cls_acc(logits, targets, topk=3)
            preds = logits.argmax(dim=1)

            total_loss += loss.item()

            total_top1 += top1 / 100 * len(targets)
            total_top3 += top3 / 100 * len(targets)

            total_seen += len(targets)
            batches += 1

            y_true_all.append(targets.detach().cpu())
            y_pred_all.append(preds.detach().cpu())

            f1_metric.update(preds, targets)
            if cm_metric is not None:
                cm_metric.update(preds, targets)
            if cls_tracker is not None:
                _, top3_pred_indices, top3_probs = cls_tracker.top3_metrics(logits, targets)
                cls_tracker.track_classification(preds, targets, top3_pred_indices, top3_probs, metadata)
            if l2_acc is not None:
                l2_acc.update(logits, targets)

    avg_loss = total_loss / max(batches, 1)
    avg_top1 = total_top1 / max(total_seen, 1)
    avg_top3 = total_top3 / max(total_seen, 1)
    f1_weighted = float(f1_metric.compute().item())
    y_true_all = torch.cat(y_true_all).numpy()
    y_pred_all = torch.cat(y_pred_all).numpy()
    mcc = float(matthews_corrcoef(y_true_all, y_pred_all))

    l2_metrics = l2_acc.compute() if l2_acc is not None else None

    if cls_tracker is not None:
        cls_tracker.save_classification()

    if cm_metric is None:
        confusion_matrix = None
        return avg_loss, avg_top1, avg_top3, f1_weighted, mcc, confusion_matrix, l2_metrics
    confusion_matrix = cm_metric.compute().cpu().numpy()
    return avg_loss, avg_top1, avg_top3, f1_weighted, mcc, confusion_matrix, l2_metrics


class FTOpenCLIP(FSCLIPmethod):
    """
    The loss calculation can refer to https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py. 
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.cfg = args

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                text_weights: torch.tensor,
                model: nn.Module,
                classnames,
                shots: int, 
                config_file: str,
                return_valid: bool, 
                prompt_tokens: torch.tensor = None,
                num_templates: int = None,
                ):

        cfg = self.cfg
        ft_cfg = cfg['finetune']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)    # perhaps we can skip this, as we have loaded the model on device in model init

        # Optional L2 evaluation context (only for full label set).
        l2_eval_ctx = None
        eval_l2 = bool(cfg.get('finetune', {}).get('eval_l2', False))
        if eval_l2:
            subset_l2_names = cfg.get('subset_l2_names', []) or []
            if isinstance(subset_l2_names, str):
                subset_l2_names = [subset_l2_names]
            if len(subset_l2_names) > 0:
                print("[warn] L2 eval disabled because subset_l2_names is set.")
                eval_l2 = False

        if eval_l2:
            from data import build_l3_to_l2_map
            l3_to_l2, l2_names = build_l3_to_l2_map()
            if len(l3_to_l2) != int(text_weights.shape[1]):
                print("[warn] L2 eval disabled due to L3 mapping size mismatch.")
            else:
                l2_mode = str(ft_cfg.get("l2_eval_mode", "argmax")).lower()
                l2_topk = (1,) if l2_mode == "argmax" else (1, 3)
                l2_eval_ctx = {
                    "l3_to_l2": torch.tensor(l3_to_l2, device=device, dtype=torch.long),
                    "num_l2": len(l2_names),
                    "l2_names": l2_names,
                    "reduce": "mean",
                    "topk": l2_topk,
                    "mode": l2_mode,
                    "return_confusion_matrix": False,
                }

        # Set up the trainable layers
        unlocked_groups = int(ft_cfg.get('unlocked_groups', 1))
        model.lock_image_tower(unlocked_groups=unlocked_groups)

        tune_text = bool(ft_cfg.get('tune_text', False))
        if not tune_text:
            model.lock_text_tower()
        else:
            # Leave text tower trainable; compute text weights from prompt tokens each step.
            # prompt_tokens = prompt_tokens.to(device)
            text_unlocked_layers = int(ft_cfg.get('unlocked_layers', 1))
            model.lock_text_tower(unlocked_layers=text_unlocked_layers)

        # Print out the information about the trainable layers
        frozen, trainable = [], []
        for name, p in model.named_parameters():
            (trainable if p.requires_grad else frozen).append(name)

        print(f"Trainable params: {len(trainable)} ({len(trainable)/(len(trainable)+len(frozen)):.1%})")
        print(f"Frozen params   : {len(frozen)}")
        
        groups = defaultdict(list)
        for n in trainable:
            top = n.split('.')[1] if '.' in n else n  # e.g., visual.trunk.attn_pool.*
            groups[top].append(n)
        for g, names in groups.items():
            print(f"  {g}: {len(names)} params")

        print("Trainable (sample):", trainable[:10])
        trainable_visual = [n for n in trainable if n.startswith("visual")]
        trainable_text = [n for n in trainable if n.startswith("text")]
        print(f"Trainable vision params : {len(trainable_visual)}")
        print("Trainable vision (sample):", trainable_visual[:10])
        print(f"Trainable text params : {len(trainable_text)}")
        print("Trainable text (sample)  :", trainable_text[:10])

        # Initialize the optimizer and scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=cfg['lr_v'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'])

        # Training loop
        val_interval = int(ft_cfg.get('val_interval', 0))  # 0 -> only final

        print('\nStart Training procedure')
        for train_idx in range(cfg['train_epoch']):
            correct_samples, all_samples = 0, 0
            running_loss, running_batches = 0.0, 0
            print('Train Epoch: {:} / {:}'.format(train_idx + 1, cfg['train_epoch']))

            model.train()

            for i, (images, targets) in enumerate(tqdm(train_loader)):
                images, targets = images.to(device), targets.to(device)
                with torch.autocast(device_type=device):
                    image_features = model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
                    if tune_text:
                        text_weights_step = _compute_text_weights_from_tokens(
                            model=model,
                            prompt_tokens=prompt_tokens,
                            num_classes=len(classnames),
                            num_templates=num_templates,
                        )
                    else:
                        text_weights_step = text_weights

                    logits = 100.0 * image_features @ text_weights_step  # logit_scale is ignored

                    loss = F.cross_entropy(logits, targets)

                acc = cls_acc(logits, targets)
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                running_loss += loss.item()
                running_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = running_loss / max(running_batches, 1)
            lr_curr = optimizer.param_groups[0]['lr']
            print('Acc: {:.4f} ({:}/{:}), Avg Loss: {:.4f}, LR: {:.2e}'.format(correct_samples / all_samples, correct_samples, all_samples, avg_loss, lr_curr))
            scheduler.step()

            # Validation
            val_loss, val_top1_acc, val_top3_acc, val_f1, val_mcc, val_cm = None, None, None, None, None, None
            do_val = (val_interval and ((train_idx + 1) % val_interval == 0)) or ((train_idx + 1) == cfg['train_epoch'])
            if do_val:
                if val_loader is not None:
                    if tune_text:
                        model.eval()
                        with torch.no_grad():
                            text_weights_val = _compute_text_weights_from_tokens(
                                model=model,
                                prompt_tokens=prompt_tokens,
                                num_classes=len(classnames),
                                num_templates=num_templates,
                            )
                    else:
                        text_weights_val = text_weights
                    val_loss, val_top1_acc, val_top3_acc, val_f1, val_mcc, val_cm, val_l2 = _run_validation(
                        model,
                        val_loader,
                        text_weights_val,
                        device,
                        return_confusion_matrix=False,
                        l2_eval_ctx=l2_eval_ctx,
                    )
                    print(f"[val epoch {train_idx+1}] loss={val_loss:.4f}, top1_acc={val_top1_acc:.4f}, top3_acc={val_top3_acc:.4f}, f1={val_f1:.4f}, mcc={val_mcc:.4f}")
                    if val_l2 is not None:
                        msg = f"[val epoch {train_idx+1} L2] top1={val_l2['top1']:.4f}, "
                        if "top3" in val_l2:
                            msg += f"top3={val_l2['top3']:.4f}, "
                        msg += f"f1={val_l2['f1']:.4f}, mcc={val_l2['mcc']:.4f}"
                        print(msg)
                else:
                    print(f"[val epoch {train_idx+1}] skipped (val_loader=None)")

        # Evaluation
        test_loss, test_top1_acc, test_top3_acc, test_f1, test_mcc, test_cm = None, None, None, None, None, None
        if test_loader is not None:
            if tune_text:
                model.eval()
                with torch.no_grad():
                    text_weights_test = _compute_text_weights_from_tokens(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        num_classes=len(classnames),
                        num_templates=num_templates,
                    )
            else:
                text_weights_test = text_weights
            test_loss, test_top1_acc, test_top3_acc, test_f1, test_mcc, test_cm, test_l2 = _run_validation(
                model,
                test_loader,
                text_weights_test,
                device,
                return_confusion_matrix=True,
                cls_track=True,
                l2_eval_ctx=l2_eval_ctx,
            )
            print(f"[test] loss={test_loss:.4f}, top1_acc={test_top1_acc:.4f}, top3_acc={test_top3_acc:.4f}, f1={test_f1:.4f}, mcc={test_mcc:.4f}")
            if test_l2 is not None:
                msg = f"[test L2] top1={test_l2['top1']:.4f}, "
                if "top3" in test_l2:
                    msg += f"top3={test_l2['top3']:.4f}, "
                msg += f"f1={test_l2['f1']:.4f}, mcc={test_l2['mcc']:.4f}"
                print(msg)
        else:
            print("[test] skipped (test_loader=None)")
            
        torch.cuda.empty_cache()
        
        if return_valid:
            return val_loss, val_top1_acc, val_top3_acc, val_f1, val_mcc, val_cm
        else:
            return test_loss, test_top1_acc, test_top3_acc, test_f1, test_mcc, test_cm

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Sequence, Union
from torcheval.metrics import MulticlassF1Score, MulticlassConfusionMatrix
from sklearn.metrics import matthews_corrcoef
from data import REASSIGN_LABEL_NAME_L3
import pandas as pd
import torch.nn.functional as F


def draw_cm(cm, label_list) -> None:
    """
    Plot a confusion matrix using Seaborn/Matplotlib and log it to W&B.

    :param cm: Confusion matrix (expected: `numpy.ndarray` of shape `[num_classes, num_classes]`).
        If a `torch.Tensor` is provided, it will be converted to a NumPy array.
    :param label_list: List of class names used for x/y tick labels. Length should match `cm.shape[0]`.
    :return: None
    """
    if not isinstance(cm, np.ndarray):
        print(f"[warn] draw_cm expected `cm` as numpy.ndarray, got {type(cm)}; attempting conversion.")

        if isinstance(cm, torch.Tensor):
            cm = cm.detach().cpu().numpy()
        else:
            try:
                cm = np.asarray(cm)
            except Exception as e:
                print(f"[warn] draw_cm could not convert `cm` to numpy.ndarray ({e}); skipping plot.")
                return

    def _custom_format(x):
        """
        Custom formatting for the confusion matrix, if an entry in the confusion matrix is a float number,
        it is shown with .2f precision.
        :param x:
        :return:
        """
        if x == 0:
            return '0'
        else:
            return f'{x:.2f}'

    def _plot_cm(cm, class_names: list, level: str, normalized: bool = False) -> None:
        title_suffix = ' (Normalized)' if normalized else ''

        # Set up annotations based on whether it's normalized or not
        if normalized:
            annot_data = np.array([[_custom_format(val) for val in row] for row in cm])
            fmt = ''
        else:
            annot_data = cm.astype(int)  # Convert to integer format for non-normalized matrix
            fmt = 'd'

        # Create plot
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=annot_data, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix {level} {title_suffix}')
        plt.tight_layout()

        # Log CM to W&B
        wandb.log({"Confusion Matrix": wandb.Image(plt)})
        plt.close()  # Close figure to free memory
        # logging.info(f'Confusion matrix {level} plot saved.')

    _plot_cm(cm, label_list, 'L3', normalized=False)

    # Calculate and save ground-truth-normalized confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Deal with the case where a row has no items.
    cm_norm = cm / row_sums
    _plot_cm(cm_norm, label_list, 'L3', normalized=True)


def map_l3_targets_to_l2(targets_l3: torch.Tensor, l3_to_l2: Union[Sequence[int], torch.Tensor]) -> torch.Tensor:
    """
    Map L3 target indices to L2 indices using a lookup vector. Ensures l3_to_l2 is a torch.Tensor on the same device as targets_l3.
    """
    if not torch.is_tensor(l3_to_l2):
        l3_to_l2 = torch.tensor(list(l3_to_l2), device=targets_l3.device, dtype=torch.long)
    else:
        l3_to_l2 = l3_to_l2.to(device=targets_l3.device, dtype=torch.long)

    return l3_to_l2[targets_l3.long()]


def aggregate_logits_to_l2(
    logits_l3: torch.Tensor,
    l3_to_l2: Union[Sequence[int], torch.Tensor],
    num_l2: int,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Aggregate L3 logits into L2 logits by summing/averaging (or log-sum-exp) per L2 group.
    Interpretation: "sum" totals subclass evidence, "mean" averages to reduce bias from
    different L3 counts per L2, and "logsumexp" approximates log of summed probabilities.
    """
    if torch.is_tensor(l3_to_l2):
        l3_to_l2_list = l3_to_l2.detach().cpu().tolist()
        l3_count = int(l3_to_l2.numel())
    else:
        l3_to_l2_list = list(l3_to_l2)
        l3_count = len(l3_to_l2_list)

    if int(logits_l3.shape[1]) != l3_count:
        raise ValueError(
            f"logits_l3 has {int(logits_l3.shape[1])} classes, but l3_to_l2 has {l3_count} entries."
        )

    if reduce not in {"sum", "mean", "logsumexp"}:
        raise ValueError(f"Unsupported reduce='{reduce}'. Expected one of: sum, mean, logsumexp.")

    if reduce == "logsumexp":
        l2_logits = torch.full(
            (logits_l3.shape[0], num_l2),
            float("-inf"),
            device=logits_l3.device,
            dtype=logits_l3.dtype,
        )
        for l3_id, l2_id in enumerate(l3_to_l2_list):
            l2_logits[:, l2_id] = torch.logaddexp(l2_logits[:, l2_id], logits_l3[:, l3_id])
        return l2_logits

    l2_logits = torch.zeros(
        (logits_l3.shape[0], num_l2),
        device=logits_l3.device,
        dtype=logits_l3.dtype,
    )
    counts = torch.zeros(num_l2, device=logits_l3.device, dtype=logits_l3.dtype)
    for l3_id, l2_id in enumerate(l3_to_l2_list):
        l2_logits[:, l2_id] += logits_l3[:, l3_id]
        counts[l2_id] += 1

    if reduce == "mean":
        l2_logits = l2_logits / counts.clamp_min(1)

    return l2_logits


class L2MetricsAccumulator:
    """
    Accumulate L2 metrics from L3 logits/targets using a fixed L3->L2 mapping.

    Tracks top-k accuracy, weighted F1, MCC, and optional confusion matrix on L2 labels.
    Mode "argmax" maps the L3 argmax prediction to L2 and reports top-1 only.
    Mode "logits" aggregates L3 logits into L2 logits and supports top-k.
    """
    def __init__(
        self,
        l3_to_l2: Union[Sequence[int], torch.Tensor],
        num_l2: int,
        reduce: str = "mean",
        topk: Sequence[int] = (1, 3),
        return_confusion_matrix: bool = False,
        mode: str = "argmax",
    ) -> None:
        self.l3_to_l2 = l3_to_l2
        self.num_l2 = int(num_l2)
        self.reduce = reduce
        self.mode = mode
        self.topk = tuple(int(k) for k in topk)
        self.return_confusion_matrix = return_confusion_matrix
        if self.mode not in {"argmax", "logits"}:
            raise ValueError(f"Unsupported mode='{self.mode}'. Expected 'argmax' or 'logits'.")
        if self.mode == "argmax":
            # Only report top-1 in argmax mode.
            self.topk = (1,)

        self.total_seen = 0
        self.correct_at_k = {k: 0 for k in self.topk}
        self.y_true_all = []
        self.y_pred_all = []

        self.f1_metric = MulticlassF1Score(num_classes=self.num_l2, average="weighted")
        self.cm_metric = (
            MulticlassConfusionMatrix(num_classes=self.num_l2)
            if return_confusion_matrix
            else None
        )

    def update(self, logits_l3: torch.Tensor, targets_l3: torch.Tensor) -> None:
        """
        Update metrics with a batch of L3 logits and L3 targets.
        """
        targets_l2 = map_l3_targets_to_l2(targets_l3, self.l3_to_l2)

        batch_size = int(targets_l2.shape[0])
        self.total_seen += batch_size

        if batch_size == 0:
            return

        if self.mode == "argmax":
            preds_l3 = logits_l3.argmax(dim=1)
            preds = map_l3_targets_to_l2(preds_l3, self.l3_to_l2)
            self.correct_at_k[1] += int((preds == targets_l2).sum().item())
        else:
            logits_l2 = aggregate_logits_to_l2(
                logits_l3, self.l3_to_l2, self.num_l2, reduce=self.reduce
            )
            max_k = min(max(self.topk), self.num_l2)
            topk_preds = logits_l2.topk(max_k, dim=1).indices  # [B, max_k]
            correct = topk_preds.eq(targets_l2.view(-1, 1))
            for k in self.topk:
                k_eff = min(k, max_k)
                if k_eff < 1:
                    continue
                correct_k = correct[:, :k_eff].any(dim=1).sum().item()
                self.correct_at_k[k] += int(correct_k)
            preds = logits_l2.argmax(dim=1)
        self.y_true_all.append(targets_l2.detach().cpu())
        self.y_pred_all.append(preds.detach().cpu())

        self.f1_metric.update(preds, targets_l2)
        if self.cm_metric is not None:
            self.cm_metric.update(preds, targets_l2)

    def compute(self) -> dict:
        """
        Return a dict of L2 metrics accumulated so far.
        """
        metrics = {}
        denom = max(self.total_seen, 1)

        for k in self.topk:
            metrics[f"top{k}"] = self.correct_at_k.get(k, 0) / denom

        if self.total_seen == 0:
            metrics["f1"] = 0.0
            metrics["mcc"] = 0.0
            metrics["cm"] = None if self.cm_metric is None else np.zeros((self.num_l2, self.num_l2))
            return metrics

        metrics["f1"] = float(self.f1_metric.compute().item())

        y_true_all = torch.cat(self.y_true_all).numpy()
        y_pred_all = torch.cat(self.y_pred_all).numpy()
        metrics["mcc"] = float(matthews_corrcoef(y_true_all, y_pred_all))

        if self.cm_metric is None:
            metrics["cm"] = None
        else:
            metrics["cm"] = self.cm_metric.compute().cpu().numpy()

        return metrics


class ClassificationTracker:
    """
    Track the classcification.
    """
    def __init__(self) -> None:
        self.misclassified = []
        self.accurate_classified = []
    
    def top3_metrics(self, outputs, labels):
        top3_pred_indices = torch.topk(outputs, 3, dim=1).indices

        # Convert logits to probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Get the probabilities corresponding to the top-3 indices
        top3_probs = torch.gather(probabilities, 1, top3_pred_indices)

        # top3_pred_indices = top3_pred_indices.cpu().numpy()
        # top3_probs = top3_probs.cpu().numpy()
        top3_correct = torch.sum(torch.any(top3_pred_indices == labels.unsqueeze(1), dim=1))
        return top3_correct, top3_pred_indices, top3_probs
    
    def track_classification(self, predictions, labels, top3_labels, top3_probs, metadata):
        """
        :param predictions: predicted labels in a batch data
        :param labels: the ground truth labels in a batch data
        :param top3_labels: top 3 labels in a batch data
        :param top3_probs: top 3 probabilities in a batch data
        :param metadata: the metadata associated to the batch data
        :return: a list of misclassified sample details
        """
        for i in range(len(labels)):
            instance_result = {
                'file_name': metadata['file_name'][i],
                'ground_truth_num_label': labels[i].item(),
                'ground_truth_word_label': metadata['plot_word_label'][i],
                'predicted_label': predictions[i].item(),
                'predicted_word_label': REASSIGN_LABEL_NAME_L3[predictions[i].item()],
                'top3_predictions': [
                    {'label': int(top3_labels[i][j]), 'probability': float(top3_probs[i][j])}
                    for j in range(3)
                ],
                'dataset': metadata['image_source'][i]
            }

            if predictions[i] != labels[i]:
                self.misclassified.append(instance_result)
            else:
                self.accurate_classified.append(instance_result)
    
    def save_classification(self):
        """
        Save classification results to CSVs.
        :return:
        """

        def flatten_instance_result(instance_results):
            """
            Flatten the nested top-3 predictions into a format suitable for CSV.
            :param instance_results: List of instance results (either misclassified or accurate).
            """
            flat_data = []
            for instance in instance_results:
                # Base information
                base_info = {
                    'file_name': instance['file_name'],
                    'ground_truth_num_label': instance['ground_truth_num_label'],
                    'ground_truth_word_label': instance['ground_truth_word_label'],
                    'predicted_label': instance['predicted_label'],
                    'predicted_word_label': instance['predicted_word_label'],
                    'dataset': instance['dataset'],
                }

                # Flatten top-3 predictions
                for i, top3_entry in enumerate(instance['top3_predictions']):
                    base_info[f'top3_label_{i + 1}'] = top3_entry['label']
                    base_info[f'top3_prob_{i + 1}'] = top3_entry['probability']

                flat_data.append(base_info)

            # Convert to DataFrame and save
            df = pd.DataFrame(flat_data)
            return df

        if self.misclassified:
            mis_df = flatten_instance_result(self.misclassified)
            wandb.log({"Misclassifications": wandb.Table(dataframe=mis_df)})
        else:
            print('No misclassified samples')

        if self.accurate_classified:
            cor_df = flatten_instance_result(self.accurate_classified)
            wandb.log({"Corclassifications": wandb.Table(dataframe=cor_df)})
        else:
            print('No correctly classified samples')
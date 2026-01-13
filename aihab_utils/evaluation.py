import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


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

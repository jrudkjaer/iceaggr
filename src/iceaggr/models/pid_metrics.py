"""Classification metrics for PID: confusion matrix, accuracy, per-class accuracy."""

import numpy as np
import matplotlib.pyplot as plt
import torch

CLASS_NAMES = ("noise", "muon", "neutrino")   # class 0, 1, 2


def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Count (true class, predicted class) pairs.

    logits: (batch, num_classes), targets: (batch,) integers.
    Returns a (num_classes, num_classes) tensor: rows = true class, columns = predicted class.
    """
    preds = logits.argmax(dim=1)
    index = targets * num_classes + preds
    counts = torch.bincount(index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def metrics_from_confusion(cm: torch.Tensor) -> dict:
    """Overall accuracy and per-class accuracy from a confusion matrix."""
    cm = cm.double()
    accuracy = (cm.diag().sum() / cm.sum().clamp(min=1)).item()
    per_class = (cm.diag() / cm.sum(dim=1).clamp(min=1)).tolist()
    return {"accuracy": accuracy, "per_class_accuracy": per_class}

def plot_confusion_matrix(cm, class_names, normalize=True, path=None):
    """Heatmap of a confusion matrix (rows = true class, columns = predicted class).

    normalize=True: each row is divided by its total, so a cell shows the fraction
    of that true class that was predicted as the column's class.
    """
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) if normalize else cm

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    fig.colorbar(im, ax=ax)

    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if shown[i, j] > 0.5 else "black")

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig, ax
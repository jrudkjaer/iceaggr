"""Classification metrics for PID: confusion matrix, accuracy, per-class accuracy."""

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
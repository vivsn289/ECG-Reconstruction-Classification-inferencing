# src_ecg_1d/training/metrics.py
#
# Evaluation metrics for multi-class ECG classification.
#
# All metrics operate on integer class predictions and targets (not logits).
# They are designed to work with torch.Tensor inputs.
#
# Why Macro-F1 is the primary metric:
#   PTB-XL is class-imbalanced (NORM is the largest class at ~28%).
#   Raw accuracy can be misleading — a model predicting NORM for everything
#   would get ~28% accuracy but be clinically useless.
#   Macro-F1 averages F1 equally across classes, so rare classes matter equally.

import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Overall classification accuracy.

    Args:
        preds:   integer class predictions, shape (N,)
        targets: integer ground truth labels, shape (N,)

    Returns:
        fraction of correctly classified samples in [0, 1]
    """
    return (preds == targets).float().mean().item()


def per_class_recall(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> list:
    """Recall (sensitivity) per class.

    Recall for class c = TP_c / (TP_c + FN_c)

    Args:
        preds:       integer class predictions, shape (N,)
        targets:     integer ground truth labels, shape (N,)
        num_classes: total number of classes

    Returns:
        list of floats, one recall value per class
    """
    recalls = []

    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        recall = tp / (tp + fn + 1e-8)
        recalls.append(recall)

    return recalls


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 score.

    Computes F1 for each class independently, then averages uniformly.
    This treats all classes equally regardless of frequency.

    F1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
    Macro-F1 = mean(F1_c for c in classes)

    Args:
        preds:       integer class predictions, shape (N,)
        targets:     integer ground truth labels, shape (N,)
        num_classes: total number of classes

    Returns:
        scalar float in [0, 1]
    """
    f1s = []

    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)

    return sum(f1s) / num_classes

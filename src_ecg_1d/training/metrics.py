# src_ecg_1d/training/metrics.py
#
# Evaluation metrics for multi-label ECG classification.
#
# PTB-XL is multi-label: ~20% of records carry more than one superclass
# diagnosis simultaneously. All metrics here operate on binary prediction
# and target matrices of shape (N, num_classes), where each column is an
# independent binary classification problem for one superclass.
#
# Why Macro-F1 is the primary metric:
#   PTB-XL is class-imbalanced (NORM is the largest class at ~28%).
#   Macro-F1 averages F1 equally across classes, so rare classes matter
#   as much as common ones. Subset accuracy (all classes correct for a
#   sample) is reported as a secondary, stricter metric.

import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Subset accuracy: fraction of samples where every class matches exactly.

    This is a strict metric for multi-label predictions — a single wrong
    class anywhere in the (num_classes,) row counts as a miss. See
    macro_f1 for the primary metric used in this pipeline.

    Args:
        preds:   binary predictions, shape (N, num_classes)
        targets: binary ground truth, shape (N, num_classes)

    Returns:
        fraction of exactly-matching samples in [0, 1]
    """
    return (preds == targets).all(dim=1).float().mean().item()


def per_class_recall(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> list:
    """Recall (sensitivity) per class.

    Recall for class c = TP_c / (TP_c + FN_c)

    Args:
        preds:       binary predictions, shape (N, num_classes)
        targets:     binary ground truth, shape (N, num_classes)
        num_classes: total number of classes

    Returns:
        list of floats, one recall value per class
    """
    recalls = []

    for cls in range(num_classes):
        p, t = preds[:, cls], targets[:, cls]
        tp = ((p == 1) & (t == 1)).sum().item()
        fn = ((p == 0) & (t == 1)).sum().item()
        recall = tp / (tp + fn + 1e-8)
        recalls.append(recall)

    return recalls


def per_class_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> list:
    """F1 score per class.

    Args:
        preds:       binary predictions, shape (N, num_classes)
        targets:     binary ground truth, shape (N, num_classes)
        num_classes: total number of classes

    Returns:
        list of floats, one F1 value per class
    """
    f1s = []

    for cls in range(num_classes):
        p, t = preds[:, cls], targets[:, cls]
        tp = ((p == 1) & (t == 1)).sum().item()
        fp = ((p == 1) & (t == 0)).sum().item()
        fn = ((p == 0) & (t == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)

    return f1s


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 score.

    Computes F1 for each class independently, then averages uniformly.
    This treats all classes equally regardless of frequency.

    Args:
        preds:       binary predictions, shape (N, num_classes)
        targets:     binary ground truth, shape (N, num_classes)
        num_classes: total number of classes

    Returns:
        scalar float in [0, 1]
    """
    f1s = per_class_f1(preds, targets, num_classes)
    return sum(f1s) / num_classes

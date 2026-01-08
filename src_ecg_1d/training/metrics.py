# src_ecg_1d/training/metrics.py

import torch


def accuracy(preds, targets):
    return (preds == targets).float().mean().item()


def per_class_recall(preds, targets, num_classes):
    recalls = []

    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()

        recall = tp / (tp + fn + 1e-8)
        recalls.append(recall)

    return recalls


def macro_f1(preds, targets, num_classes):
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

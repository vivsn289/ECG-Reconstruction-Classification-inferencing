# src_ecg_1d/training/losses.py
#
# Loss functions for ECG classification.
#
# Note: the primary training loop (train_1d.py) uses BCEWithLogitsLoss with
# pos_weight (multi-label), which is effective for the moderate imbalance in
# PTB-XL. FocalLoss is included here as a single-label alternative for more
# severe imbalance scenarios or when per-class weighting is insufficient.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced multi-class classification.

    Focal Loss down-weights easy (well-classified) examples and focuses
    training on hard (misclassified) examples. This is useful when a
    majority class is well-learned early in training but rare classes
    are still being misclassified.

    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
               https://arxiv.org/abs/1708.02002

    Args:
        alpha: optional per-class weight tensor of shape (num_classes,)
            applied before the focal modulation. None = uniform.
        gamma: focusing parameter. gamma=0 → standard cross-entropy.
            gamma=2 is a common default.
        reduction: "mean", "sum", or "none"

    Usage:
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Standard cross-entropy per sample (shape: N)
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)

        # p_t = exp(-CE) = probability assigned to the correct class
        pt = torch.exp(-ce_loss)

        # Focal modulation: down-weight easy examples (high pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

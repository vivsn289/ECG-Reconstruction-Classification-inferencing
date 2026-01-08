# src_ecg_1d/models/ecg_model.py

import torch
import torch.nn as nn

from src_ecg_1d.models.cnn_backbone import ECGCNNBackbone
from src_ecg_1d.models.transformer import ECGTransformer


class ECGClassifier1D(nn.Module):
    """
    Full ECG classification model:
    CNN + ECA backbone → Transformer → Global pooling → Classifier
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 12,
        embed_dim: int = 256,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # CNN feature extractor
        self.backbone = ECGCNNBackbone(in_channels=in_channels)

        # Transformer for temporal reasoning
        self.transformer = ECGTransformer(
            embed_dim=embed_dim,
            num_heads=transformer_heads,
            depth=transformer_depth,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x, return_attention=False):
        """
        Args:
            x: ECG tensor (N, 12, L)
            return_attention: whether to return attention maps

        Returns:
            logits: (N, num_classes)
            attentions (optional): list of attention tensors
        """
        # CNN backbone
        x = self.backbone(x)          # (N, C, T)

        # Prepare for transformer
        x = x.permute(0, 2, 1)        # (N, T, C)

        if return_attention:
            x, attentions = self.transformer(x, return_attention=True)
        else:
            x = self.transformer(x)

        # Global average pooling over time
        x = x.mean(dim=1)             # (N, C)

        # Classification
        logits = self.classifier(x)

        if return_attention:
            return logits, attentions

        return logits

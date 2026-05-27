# src_ecg_1d/models/ecg_model.py
#
# Top-level ECG classifier: CNN backbone → Transformer encoder → classifier head.
#
# Architecture overview:
#   Input:      (N, 12, 1000)         — batch of 12-lead ECG windows
#   CNN:        (N, 256, T_reduced)   — local feature extraction with downsampling
#   Permute:    (N, T_reduced, 256)   — reshape for Transformer
#   Transformer:(N, T_reduced, 256)   — temporal reasoning with self-attention
#   Mean pool:  (N, 256)              — aggregate sequence into fixed-size vector
#   Classifier: (N, num_classes)      — final logits
#
# The CNN backbone uses residual blocks with ECA (Efficient Channel Attention)
# to learn lead-aware local features. The Transformer then reasons about
# long-range temporal dependencies across the compressed sequence.

import torch.nn as nn

from src_ecg_1d.models.cnn_backbone import ECGCNNBackbone
from src_ecg_1d.models.transformer import ECGTransformer


class ECGClassifier1D(nn.Module):
    """End-to-end 1D ECG classifier (CNN + Transformer).

    Explainability compatibility:
        - Integrated Gradients: supported (fully differentiable forward pass)
        - Attention maps: supported via return_attention=True
        - Gradient-based saliency: supported

    Args:
        num_classes: number of output classes (5 for PTB-XL superclasses)
        in_channels: number of ECG leads (12 for standard 12-lead)
        embed_dim: channel dimension output by CNN / input to Transformer
        transformer_depth: number of Transformer encoder layers
        transformer_heads: number of self-attention heads per layer
        dropout: dropout rate applied inside Transformer blocks
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

        self.backbone = ECGCNNBackbone(in_channels=in_channels)

        self.transformer = ECGTransformer(
            embed_dim=embed_dim,
            num_heads=transformer_heads,
            depth=transformer_depth,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x, return_attention=False):
        """Forward pass.

        Args:
            x: input tensor, shape (N, in_channels, time)
               e.g. (N, 12, 1000) for 12-lead ECG at 100 Hz
            return_attention: if True, also return the list of attention
               tensors from each Transformer layer (for visualization)

        Returns:
            logits: (N, num_classes)
            attentions (optional): list of (N, num_heads, T, T) tensors,
               one per Transformer layer
        """
        # CNN backbone: (N, in_channels, 1000) → (N, embed_dim, T_reduced)
        x = self.backbone(x)

        # Transformer expects sequence-first: (N, T_reduced, embed_dim)
        x = x.permute(0, 2, 1)

        if return_attention:
            x, attentions = self.transformer(x, return_attention=True)
        else:
            x = self.transformer(x)

        # Global mean pooling over time: (N, T_reduced, embed_dim) → (N, embed_dim)
        x = x.mean(dim=1)

        logits = self.classifier(x)

        if return_attention:
            return logits, attentions

        return logits

# src_ecg_1d/models/ecg_model.py

import torch
import torch.nn as nn

from src_ecg_1d.models.cnn_backbone import ECGCNNBackbone
from src_ecg_1d.models.transformer import ECGTransformer


class ECGClassifier1D(nn.Module):


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

        # CNN backbone
        x = self.backbone(x)       

        # Prepare for transformer
        x = x.permute(0, 2, 1)        

        if return_attention:
            x, attentions = self.transformer(x, return_attention=True)
        else:
            x = self.transformer(x)


        x = x.mean(dim=1)     

        logits = self.classifier(x)

        if return_attention:
            return logits, attentions

        return logits

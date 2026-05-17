# src_ecg_1d/models/eca.py

import torch
import torch.nn as nn


class ECA1D(nn.Module):


    def __init__(self, channels, k_size=3):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Global average pooling: (N, C, 1)
        y = self.avg_pool(x)

        # Reshape for channel-wise convolution
        # (N, C, 1) → (N, 1, C)
        y = y.transpose(1, 2)

        # Channel interaction
        y = self.conv(y)

        # Attention weights
        y = self.sigmoid(y)

        # Reshape back: (N, 1, C) → (N, C, 1)
        y = y.transpose(1, 2)

        # Scale input
        return x * y

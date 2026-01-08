# src_ecg_1d/models/eca.py

import torch
import torch.nn as nn


class ECA1D(nn.Module):
    """
    Efficient Channel Attention for 1D signals (ECG).

    Reference:
    Wang et al., "ECA-Net: Efficient Channel Attention for Deep CNNs"
    Adapted for 1D temporal feature maps.
    """

    def __init__(self, channels, k_size=3):
        """
        Args:
            channels (int): Number of input channels
            k_size (int): Kernel size for 1D convolution (odd number)
        """
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
        """
        Args:
            x: Tensor of shape (N, C, L)

        Returns:
            Tensor of same shape (N, C, L)
        """
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

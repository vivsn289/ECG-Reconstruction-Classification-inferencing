# src_ecg_1d/models/eca.py
#
# Efficient Channel Attention (ECA) for 1D signals.
#
# Reference: Wang et al. (2020) "ECA-Net: Efficient Channel Attention for
#            Deep Convolutional Neural Networks"
#            https://arxiv.org/abs/1910.03151
#
# ECA learns to re-weight channels (ECG leads, in our case) using a
# lightweight 1D convolution over the globally-pooled channel vector.
# This is more efficient than SE (Squeeze-and-Excitation) because it
# avoids the two-fully-connected-layer bottleneck.
#
# In the ECG context, ECA allows each layer to selectively emphasize
# leads that are most informative for the current representation.

import torch.nn as nn


class ECA1D(nn.Module):
    """Efficient Channel Attention for 1D feature maps.

    Computes a per-channel attention weight vector, then scales the
    input features channel-wise.

    Args:
        channels: number of input channels (C)
        k_size: kernel size for the 1D cross-channel convolution.
            Small k (e.g. 3) captures local channel interactions.

    Shape semantics:
        Input:  (N, C, T)
        Output: (N, C, T)   — same shape, channels re-weighted
    """

    def __init__(self, channels, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Cross-channel interaction with a small 1D conv
        # Operates on (N, 1, C) to allow interaction between adjacent channels
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling across time: (N, C, T) → (N, C, 1)
        y = self.avg_pool(x)

        # Reshape for channel-wise 1D conv: (N, C, 1) → (N, 1, C)
        y = y.transpose(1, 2)

        # Cross-channel interaction: (N, 1, C) → (N, 1, C)
        y = self.conv(y)

        # Attention weights in [0, 1]
        y = self.sigmoid(y)

        # Reshape back to channel dimension: (N, 1, C) → (N, C, 1)
        y = y.transpose(1, 2)

        # Scale input features by attention weights (broadcast over time)
        return x * y

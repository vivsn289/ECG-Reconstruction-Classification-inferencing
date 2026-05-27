# src_ecg_1d/models/cnn_backbone.py
#
# Residual CNN backbone for 1D ECG feature extraction.
#
# Design rationale:
# - 4 stages with progressively larger channel counts and stride-2 downsampling
# - Residual (skip) connections to ease gradient flow in deep networks
# - ECA attention after each conv block to let the model weight leads
#   differently at each representation level
# - Kernel sizes decrease as the network deepens, matching the reduced
#   temporal resolution at later stages

import torch.nn as nn

from src_ecg_1d.models.eca import ECA1D


class ConvBlock1D(nn.Module):
    """Single residual CNN block with ECA channel attention.

    Architecture per block:
        Conv1D → BN → ReLU → ECA → (+skip) → ReLU

    The residual connection uses a 1×1 projection conv when the
    channel count or stride changes (standard ResNet convention).

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: temporal kernel size (larger = more context)
        stride: temporal downsampling factor

    Shape semantics:
        Input:  (N, in_channels,  T_in)
        Output: (N, out_channels, T_in // stride)
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.eca = ECA1D(out_channels)

        # Projection shortcut: needed when spatial or channel dims change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ECGCNNBackbone(nn.Module):
    """4-stage residual CNN for ECG temporal feature extraction.

    Each stage downsamples the temporal dimension by 2, so a 1000-sample
    input becomes ~63 patches at the output (1000 / 2^4 ≈ 63).

    Channel progression: in_channels → 32 → 64 → 128 → 256

    The output can be used as a sequence of 256-d patch embeddings,
    suitable for input to a Transformer encoder.

    Args:
        in_channels: number of ECG leads (12 for standard 12-lead)

    Shape semantics:
        Input:  (N, 12,  1000)
        Output: (N, 256, ~63)
    """

    def __init__(self, in_channels=12):
        super().__init__()

        self.stage1 = ConvBlock1D(in_channels=in_channels, out_channels=32,  kernel_size=7, stride=2)
        self.stage2 = ConvBlock1D(in_channels=32,          out_channels=64,  kernel_size=5, stride=2)
        self.stage3 = ConvBlock1D(in_channels=64,          out_channels=128, kernel_size=3, stride=2)
        self.stage4 = ConvBlock1D(in_channels=128,         out_channels=256, kernel_size=3, stride=2)

    def forward(self, x):
        # x: (N, 12, 1000)
        x = self.stage1(x)   # (N, 32,  500)
        x = self.stage2(x)   # (N, 64,  250)
        x = self.stage3(x)   # (N, 128, 125)
        x = self.stage4(x)   # (N, 256, ~63)
        return x

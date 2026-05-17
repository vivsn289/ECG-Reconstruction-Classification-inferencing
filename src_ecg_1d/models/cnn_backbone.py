# src_ecg_1d/models/cnn_backbone.py

import torch
import torch.nn as nn

from src_ecg_1d.models.eca import ECA1D


class ConvBlock1D(nn.Module):


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

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ECGCNNBackbone(nn.Module):


    def __init__(self, in_channels=12):
        super().__init__()

        self.stage1 = ConvBlock1D(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=7,
            stride=2,
        )

        self.stage2 = ConvBlock1D(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=2,
        )

        self.stage3 = ConvBlock1D(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
        )

        self.stage4 = ConvBlock1D(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

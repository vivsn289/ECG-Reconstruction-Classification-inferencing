# src/model_architecture_pytorch.py
"""
MobileLite-style model with a correct ECA implementation (registered weights).
Kept the same overall stage sizes so the base architecture is unchanged.
"""
import math
import torch
import torch.nn as nn
from src import config_pytorch as cfg

class ECALayer2D(nn.Module):
    """
    Efficient Channel Attention for 2D feature maps.
    Input: x (B, C, H, W) -> global pool (H,W) -> Conv1d across channels -> sigmoid -> scale
    Kernel-size adapts to channel count using the paper's formula.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b
        # adaptive kernel size
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if (t % 2 == 1) else (t + 1)
        if k < 1:
            k = 1
        self.k = k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # conv across channels implemented as Conv1d on (B,1,C)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k, padding=(self.k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)   # (B, C)
        y = y.unsqueeze(-1).transpose(1, 2)            # (B, 1, C)
        y = self.conv(y)                               # (B, 1, C)
        # after sigmoid: (B, 1, C) -> transpose -> (B, C, 1) -> unsqueeze -> (B, C, 1, 1)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)



class InvertedResidualBlock(nn.Module):
    """
    MobileNetV2 inverted residual block + optional ECA.
    Keep block topology identical to your previous code but use registered ECA.
    """
    def __init__(self, in_channels, out_channels, stride, expansion, use_eca=True):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        if expansion != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ]

        self.block = nn.Sequential(*layers)

        self.eca = ECALayer2D(hidden_dim) if use_eca else nn.Identity()
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = self.eca(out)
        out = self.project(out)
        if self.use_res_connect:
            return identity + out
        return out


class MobileLiteECA(nn.Module):
    def __init__(self, num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            InvertedResidualBlock(32, 16, 1, 1, use_eca=True),
            InvertedResidualBlock(16, 24, 2, 6, use_eca=True),
            InvertedResidualBlock(24, 32, 2, 6, use_eca=True),
            InvertedResidualBlock(32, 64, 2, 6, use_eca=True),
            InvertedResidualBlock(64, 96, 1, 6, use_eca=True),
            InvertedResidualBlock(96, 160, 2, 6, use_eca=True),

            nn.Conv2d(160, 640, 1, bias=False),
            nn.BatchNorm2d(640),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(cfg.SPATIAL_DROPOUT),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(640, num_classes),
        )

        # initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# model/adaptive_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCNN(nn.Module):
    """
    输入：原始轴承信号 (N, 1, L)
    使用自适应池化，适应不同长度
    """
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1)  # -> (N, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),           # -> (N, 256)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

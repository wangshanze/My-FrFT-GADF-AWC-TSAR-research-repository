import torch
import torch.nn as nn
import sys
sys.path.append("..")

from model.TSAR import TSAR
from model.attention_modules import (
    SEWeighting,
    ECAWeighting,
    CAWeighting,
    CBAMWeighting,
)


class _BaseTSARsNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(_BaseTSARsNet, self).__init__()

        self.attn_module = None

        self.initial_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block1   = TSAR(32,  64)
        self.pool1    = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.1)

        self.block2   = TSAR(64,  96)
        self.pool2    = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.1)

        self.block3   = TSAR(96,  128)
        self.pool3    = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.2)

        self.block4   = TSAR(128, 192)
        self.pool4    = nn.MaxPool2d(kernel_size=2)
        self.dropout4 = nn.Dropout2d(0.2)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        self.order_weights = None

    def forward(self, x: torch.Tensor):
        x, weights = self.attn_module(x)
        self.order_weights = weights

        x = self.initial_conv(x)

        x = self.dropout1(self.pool1(self.block1(x)))
        x = self.dropout2(self.pool2(self.block2(x)))
        x = self.dropout3(self.pool3(self.block3(x)))
        x = self.dropout4(self.pool4(self.block4(x)))

        x = self.global_pool(x)
        return self.classifier(x)


# ===========================================================================
# 1. SE-TSARs-Net
# ===========================================================================
class SEMultiChannelNet(_BaseTSARsNet):
    def __init__(self, num_classes: int = 10, input_channels: int = 22):
        super(SEMultiChannelNet, self).__init__(num_classes=num_classes)
        self.attn_module = SEWeighting(
            input_channels=input_channels,
            output_channels=32
        )


# ===========================================================================
# 2. ECA-TSARs-Net
# ===========================================================================
class ECAMultiChannelNet(_BaseTSARsNet):
    def __init__(self, num_classes: int = 10, input_channels: int = 22):
        super(ECAMultiChannelNet, self).__init__(num_classes=num_classes)
        self.attn_module = ECAWeighting(
            input_channels=input_channels,
            output_channels=32
        )


# ===========================================================================
# 3. CA-TSARs-Net
# ===========================================================================
class CAMultiChannelNet(_BaseTSARsNet):
    def __init__(self, num_classes: int = 10, input_channels: int = 22):
        super(CAMultiChannelNet, self).__init__(num_classes=num_classes)
        self.attn_module = CAWeighting(
            input_channels=input_channels,
            output_channels=32
        )


# ===========================================================================
# 4. CBAM-TSARs-Net
# ===========================================================================
class CBAMMultiChannelNet(_BaseTSARsNet):
    def __init__(self, num_classes: int = 10, input_channels: int = 22):
        super(CBAMMultiChannelNet, self).__init__(num_classes=num_classes)
        self.attn_module = CBAMWeighting(
            input_channels=input_channels,
            output_channels=32
        )

import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2  # 保持空间尺寸

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)


class FrFTGADFVGG16(nn.Module):
    def __init__(self, num_classes: int, input_channels: int):
        super().__init__()

        self.features = nn.Sequential(
            VGG(input_channels, 64, kernel_size=7, stride=2),

            # WDCNN layer2: Conv(kernel=3)
            VGG(64, 128, kernel_size=3),

            # WDCNN layer3
            VGG(128, 256, kernel_size=3),

            # WDCNN layer4
            VGG(256, 384, kernel_size=3),

            # WDCNN layer5（最终通道设为 512，适配 FC）
            VGG(384, 512, kernel_size=3),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # nn.Dropout(),# HIT

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(), # HIT,HUST

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)   
        x = torch.flatten(x, 1)  # 变成 (N, 25088)
        return self.classifier(x)


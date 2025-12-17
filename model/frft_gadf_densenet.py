# model/frft_gadf_densenet.py
import torch.nn as nn
from torchvision.models import densenet121


class FrFTGADFDenseNet(nn.Module):
    """
    输入：GADF 图像 (N, C, H, W)
    """
    def __init__(self, num_classes: int, input_channels: int):
        super().__init__()
        backbone = densenet121(weights=None)

        # 修改第一层卷积
        backbone.features.conv0 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(num_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

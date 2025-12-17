import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

"""
输入 (x)
  |
  ├─────────────────┬───────────────────┐
  |                 |                   |
主路径            辅助路径            残差连接
(标准卷积)     (深度可分离卷积)      (shortcut)
  |                 |                   |
Conv2d(1,5)    DepthwiseConv2d(1,5)     |
  |                 |                   |
BatchNorm2d     PointwiseConv1d         |
  |                 |                   |
  ReLU          BatchNorm2d             |
  |                 |                   |
  |                ReLU                 |
  |                 |                   |
  └─────────────────┘                   |
          |                             |
     特征拼接(cat)                       |
          |                             |
     融合卷积(1x1)                       |
          |                             |
      BatchNorm2d                       |
          |                             |
         ReLU                           |
          |                             |
     通道注意力机制                      |
   (Channel Attention)                  |
          |                             |
     特征加权(乘法)                      |
          |                             |
     空间注意力机制                      |
   (Spatial Attention)                  |
          |                             |
     特征加权(乘法)                      |
          |                             |
          └─────────────────────────────┘
                      |
                  特征相加(+)
                      |
                    输出
    通道注意力+空间注意力 = CBAM卷积注意力
"""

class TSAR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TSAR, self).__init__()

        # 主路径 - 标准卷积
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 辅助路径 - 深度可分离卷积
        self.aux_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # 深度卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 逐点卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 特征互补融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 残差连接的1x1卷积，用于维度匹配
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        # 双路径处理
        main_features = self.main_path(x)
        aux_features = self.aux_path(x)

        # 特征互补融合
        combined = torch.cat([main_features, aux_features], dim=1)
        fused = self.fusion(combined)

        # 应用通道注意力
        channel_att = self.channel_attention(fused)
        refined = fused * channel_att

        # 应用空间注意力
        avg_out = torch.mean(refined, dim=1, keepdim=True)
        max_out, _ = torch.max(refined, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        refined = refined * spatial_att

        # 残差连接
        output = refined + identity

        return output
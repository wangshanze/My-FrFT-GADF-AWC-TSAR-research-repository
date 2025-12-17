import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """SENet"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),# HIT
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid = out_channels // 3 

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),# HIT,HUST
        )

        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),# HIT,HUST
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * mid, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),# HIT,HUST
        )

        self.se = SEBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        a = self.branch3(x)
        b = self.branch5(x)
        c = self.branch7(x)

        x_cat = torch.cat([a, b, c], dim=1)
        x_fused = self.fuse(x_cat)
        x_se = self.se(x_fused)
        x_out = self.pool(x_se)
        return x_out


class SE_MSCNN_Backbone(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # 第一层卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),# HIT
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 三个多尺度块
        self.ms1 = MultiScaleBlock(32, 64)
        self.ms2 = MultiScaleBlock(64, 128)
        self.ms3 = MultiScaleBlock(128, 256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (N, 1, H, W)
        x = self.stem(x)
        x = self.ms1(x)
        x = self.ms2(x)
        x = self.ms3(x)
        x = self.gap(x).view(x.size(0), -1)
        out = self.fc(x)
        return out


class LMSWT_SE_MSCNN(nn.Module):
    """
    整体模型：
    输入：原始信号 (N, 1, L)
    内部：STFT -> 频率方向局部最大池化 (LMSWT-like) -> 归一化 -> resize 到 32x32
         -> SE-MSCNN 2D 主干
    输出：分类 logits
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        signal_length: int = 2048,
        tf_size: int = 32,
        n_fft: int = 256,
        hop_length: int = 64,
    ):
        super().__init__()
        assert in_channels == 1, 

        self.signal_length = signal_length
        self.tf_size = tf_size
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.backbone = SE_MSCNN_Backbone(in_channels=1, num_classes=num_classes)

    def _lmswt_like_tf(self, x: torch.Tensor) -> torch.Tensor:

        # x -> (N, L)
        x = x.squeeze(1)

        # Hann 窗
        window = torch.hann_window(self.n_fft, device=x.device)

        # STFT: (N, freq, time, 2) 复数
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )  # (N, F, T)

        spec = spec.abs()  # 幅度谱 (N, F, T)

        # 频率方向做一个 3 点局部最大池化，LMSWT 的局部最大重分配
        spec = spec.unsqueeze(1)  # (N, 1, F, T)
        pad_f = 1
        spec_padded = F.pad(spec, (0, 0, pad_f, pad_f), mode="replicate")
        spec_lms = F.max_pool2d(spec_padded, kernel_size=(3, 1), stride=1)
        # (N, 1, F, T)

        # 每个样本独立归一化到 [0,1]
        N, C, Freq, Time = spec_lms.shape
        spec_reshaped = spec_lms.view(N, -1)
        min_vals = spec_reshaped.min(dim=1, keepdim=True).values
        max_vals = spec_reshaped.max(dim=1, keepdim=True).values
        denom = (max_vals - min_vals + 1e-8)
        spec_norm = (spec_reshaped - min_vals) / denom
        spec_norm = spec_norm.view(N, C, Freq, Time)

        # resize 到 (tf_size, tf_size)
        tf_img = F.interpolate(
            spec_norm,
            size=(self.tf_size, self.tf_size),
            mode="bilinear",
            align_corners=False,
        )  # (N,1,H,W)

        return tf_img

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1, L)
        """
        tf_img = self._lmswt_like_tf(x)  # (N,1,H,W)
        out = self.backbone(tf_img)
        return out

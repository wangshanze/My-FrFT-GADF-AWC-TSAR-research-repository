import torch
import torch.nn as nn
import torch.nn.functional as F


class SEWeighting(nn.Module):

    def __init__(self, input_channels: int = 22,
                 output_channels: int = 32,
                 reduction: int = 4):
        super(SEWeighting, self).__init__()
        mid = max(8, input_channels // reduction)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                                  # [B, C, 1, 1]
            nn.Conv2d(input_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, input_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        weights = self.se(x)          # [B, C, 1, 1]
        out = self.proj(x * weights)  # [B, C_out, H, W]
        return out, weights


class ECAWeighting(nn.Module):
    def __init__(self, input_channels: int = 22,
                 output_channels: int = 32,
                 k_size: int = 3):
        super(ECAWeighting, self).__init__()
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.conv1d  = nn.Conv1d(1, 1, kernel_size=k_size,
                                 padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.proj = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        y = self.gap(x)                              # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)          # [B, 1, C]
        y = self.conv1d(y)                           # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)        # [B, C, 1, 1]
        weights = self.sigmoid(y)
        out = self.proj(x * weights)
        return out, weights


class CAWeighting(nn.Module):

    def __init__(self, input_channels: int = 22,
                 output_channels: int = 32,
                 reduction: int = 4):
        super(CAWeighting, self).__init__()
        mip = max(8, input_channels // reduction)

        self.conv1  = nn.Conv2d(input_channels, mip, kernel_size=1, bias=False)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, input_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, input_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.proj = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Coordinate pooling
        x_h = F.adaptive_avg_pool2d(x, (H, 1))                      # [B, C, H, 1]
        x_w = F.adaptive_avg_pool2d(x, (1, W)).permute(0, 1, 3, 2)  # [B, C, W, 1]
        y   = torch.cat([x_h, x_w], dim=2)                          # [B, C, H+W, 1]

        y = self.act(self.bn1(self.conv1(y)))                        # [B, mip, H+W, 1]

        x_h_out, x_w_out = torch.split(y, [H, W], dim=2)
        x_w_out = x_w_out.permute(0, 1, 3, 2)                       # [B, mip, 1, W]

        a_h = self.sigmoid(self.conv_h(x_h_out))                    # [B, C, H, 1]
        a_w = self.sigmoid(self.conv_w(x_w_out))                    # [B, C, 1, W]
        attn = a_h * a_w                                             # [B, C, H, W]

        out = self.proj(x * attn)

        weights = attn.mean(dim=[2, 3], keepdim=True)
        return out, weights


class CBAMWeighting(nn.Module):
    def __init__(self, input_channels: int = 22,
                 output_channels: int = 32,
                 reduction: int = 4):
        super(CBAMWeighting, self).__init__()
        mid = max(8, input_channels // reduction)

        # 通道注意力（共享 MLP）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(input_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, input_channels, kernel_size=1, bias=False)
        )

        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.proj = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        # ---------- 通道注意力 ----------
        ca = self.sigmoid(self.mlp(self.avg_pool(x)) +
                          self.mlp(self.max_pool(x)))   # [B, C, 1, 1]
        x  = x * ca

        # ---------- 空间注意力 ----------
        avg_s = torch.mean(x, dim=1, keepdim=True)      # [B, 1, H, W]
        max_s, _ = torch.max(x, dim=1, keepdim=True)    # [B, 1, H, W]
        sa = self.sigmoid(
            self.spatial_conv(torch.cat([avg_s, max_s], dim=1))
        )                                                # [B, 1, H, W]
        x  = x * sa

        out = self.proj(x)
        return out, ca

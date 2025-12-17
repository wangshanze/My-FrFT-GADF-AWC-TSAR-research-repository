import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pywt
except ImportError:
    pywt = None



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class MiniResNet4(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.inplanes = 32

        self.conv1 = nn.Conv2d(
            in_channels, 32,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, stride=1)   # 56×56
        self.layer2 = self._make_layer(64, stride=2)   # 28×28
        self.layer3 = self._make_layer(128, stride=2)  # 14×14
        self.layer4 = self._make_layer(256, stride=2)  # 7×7

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes, stride):
        layer = BasicBlock(self.inplanes, planes, stride=stride)
        self.inplanes = planes
        return layer

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def build_marp_from_signal(
    signal_1d: np.ndarray,
    wavelet: str = "sym8",
    level: int = 3,
    rp_size: int = 64,
) -> np.ndarray:

    wp = pywt.WaveletPacket(
        data=signal_1d,
        wavelet=wavelet,
        mode="symmetric",
        maxlevel=level,
    )

    node_paths = ["aaa", "aad", "add", "ddd"]
    coeffs = []
    for p in node_paths:
        c = np.asarray(wp[p].data, dtype=np.float32)
        coeffs.append(c)

    rps = []
    for c in coeffs:
        L = len(c)
        if L < rp_size:
            pad = rp_size - L
            c_pad = np.pad(c, (0, pad), mode="edge")
        else:
            idx = np.linspace(0, L - 1, rp_size).astype(int)
            c_pad = c[idx]

        diff = c_pad[:, None] - c_pad[None, :]
        rp = np.abs(diff).astype(np.float32)

        iu = np.triu_indices(rp_size, k=0)
        mask = np.zeros_like(rp)
        mask[iu] = rp[iu]
        rp = mask

        vmin, vmax = rp.min(), rp.max()
        if vmax > vmin:
            rp = (rp - vmin) / (vmax - vmin + 1e-8)
        else:
            rp = np.zeros_like(rp)

        rps.append(rp)

    top = np.concatenate([rps[0], rps[1]], axis=1)
    bottom = np.concatenate([rps[2], rps[3]], axis=1)
    marp = np.concatenate([top, bottom], axis=0)

    return marp.astype(np.float32)


class MARPResNet34(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        signal_length: int = 2048,
        rp_size: int = 64,
    ):
        super().__init__()
        self.signal_length = signal_length
        
        self.square_size = int(np.sqrt(signal_length))
        if self.square_size * self.square_size != signal_length:
            self.square_size = int(np.sqrt(signal_length)) + 1
        
        self.backbone = MiniResNet4(num_classes=num_classes, in_channels=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, L = x.shape
        assert C == 1, "输入应为 (N, 1, L)"
        
        target_length = self.square_size ** 2
        if L < target_length:
            # 填充
            pad_size = target_length - L
            x = F.pad(x, (0, pad_size), mode='constant', value=0)
        else:
            # 截断
            x = x[:, :, :target_length]
        
        x_2d = x.view(N, 1, self.square_size, self.square_size)
        
        x_2d = self.adaptive_pool(x_2d)
        
        out = self.backbone(x_2d)
        return out

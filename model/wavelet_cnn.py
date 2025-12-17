# model/wavelet_cnn.py
import torch
import torch.nn as nn
import pywt
import numpy as np


def dwt_multiscale(x, wavelet='db4', level=3):
    """
    x: Tensor, shape (N, 1, L)
    return: Tensor, shape (N, C, L')   例如 level=3 → C = 4 (A3, D3, D2, D1)
    """
    x_np = x.squeeze(1).cpu().numpy()  # (N, L)
    coeff_list = []

    for sig in x_np:
        # 多尺度小波分解
        coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level)
        # coeffs = [A3, D3, D2, D1]
        coeff_list.append(coeffs)

    # 找最长的系数长度 pad
    max_len = max(len(c) for coeffs in coeff_list for c in coeffs)
    
    all_coeffs = []
    for coeffs in coeff_list:
        padded = []
        for c in coeffs:
            if len(c) < max_len:
                pad = np.pad(c, (0, max_len - len(c)))
            else:
                pad = c
            padded.append(pad)
        all_coeffs.append(np.stack(padded, axis=0))  # (C, L')

    all_coeffs = np.stack(all_coeffs, axis=0)  # (N, C, L')
    return torch.tensor(all_coeffs, dtype=torch.float32)


class WaveletCNN(nn.Module):
    """
    使用真实小波变换的小波 CNN
    输入：原始信号 (N, 1, L)
    小波分解：db4，多尺度
    CNN 输入：(N, C, L')
    """
    def __init__(self, num_classes: int, wavelet='db4', dwt_level=3):
        super().__init__()
        self.wavelet = wavelet
        self.dwt_level = dwt_level
        self.c_in = dwt_level + 1  # A + D1...Dn

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(self.c_in, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.5),# HIT
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.5),# HIT
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),# HIT,HUST
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),# HIT,HUST
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x: 原始信号 (N, 1, L)
        """
        with torch.no_grad():
            x_wt = dwt_multiscale(x, self.wavelet, self.dwt_level).to(x.device)

        x_feat = self.feature_extractor(x_wt)
        out = self.classifier(x_feat)
        return out

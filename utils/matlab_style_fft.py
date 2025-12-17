import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 假设你有以下变量（请根据实际情况修改）
# normal_raw, ir1_raw, ir2_raw, orf_raw 是你的原始信号
# fs = 25000  # 采样频率

def matlab_style_fft(signal, fs):
    """
    按照MATLAB代码的风格处理FFT
    """
    L = len(signal)
    # 计算最接近的2的幂次方作为FFT点数
    NFFT = 2 ** int(np.ceil(np.log2(L)))
    
    # 去除直流分量
    yh = signal - np.mean(signal)
    
    # 傅里叶变换并归一化（与MATLAB一致）
    Y = fft(yh, NFFT) / L
    
    # 计算单边幅度谱（乘以2补偿负频率）
    half_length = NFFT // 2
    X_A = 2 * np.abs(Y[:half_length])
    
    # 频率轴（与MATLAB一致）
    f = fs / 2 * np.linspace(0, 1, half_length + 1)
    f = f[:-1]  # 去掉最后一个点以匹配X_A的长度
    
    return f, X_A
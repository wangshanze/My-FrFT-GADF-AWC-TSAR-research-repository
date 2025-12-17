import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import hilbert

def matlab_style_envelope_spectrum(signal, fs):
    """
    按照MATLAB代码的风格计算包络谱
    """
    # Hilbert变换求包络
    yh1 = np.abs(hilbert(signal))
    
    L1 = len(yh1)
    # 计算最接近的2的幂次方作为FFT点数
    NFFT1 = 2 ** int(np.ceil(np.log2(L1)))
    
    # 去除直流分量
    yh1 = yh1 - np.mean(yh1)
    
    # 傅里叶变换并归一化
    Y1 = fft(yh1, NFFT1) / L1
    
    # 计算单边幅度谱（乘以2补偿负频率）
    X_A1 = 2 * np.abs(Y1[:NFFT1//2])
    
    # 频率轴（与MATLAB一致）
    freqs = (np.arange(NFFT1//2) / NFFT1) * fs
    
    return freqs, X_A1
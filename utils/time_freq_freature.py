import numpy as np


def get_traditional_feature(signal):
    """
    提取信号的4个时域和5个频域特征
    
    Args:
        signal: 输入信号 (numpy array)
    
    Returns:
        features: 包含9个特征的numpy数组 [时域特征1-4, 频域特征1-5]
    """
    
    # 时域特征提取
    # 1. 均值
    mean_value = np.mean(signal)
    
    # 2. 标准差
    std_value = np.std(signal)
    
    # 3. 均方根值 (RMS)
    rms_value = np.sqrt(np.mean(signal**2))
    
    # 4. 峰值因子 (Crest Factor)
    peak_value = np.max(np.abs(signal))
    crest_factor = peak_value / rms_value if rms_value > 0 else 0
    
    # 频域特征提取
    # 计算FFT
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
    freqs = np.fft.fftfreq(len(signal), 1.0)[:len(fft_result)//2]
    
    # 避免除零错误
    if np.sum(fft_magnitude) == 0:
        fft_magnitude[fft_magnitude == 0] = 1e-10
    
    # 5. 频域峰值因子 (CF - Crest Factor)
    peak_freq_magnitude = np.max(fft_magnitude)
    rms_freq_magnitude = np.sqrt(np.mean(fft_magnitude**2))
    cf = peak_freq_magnitude / rms_freq_magnitude if rms_freq_magnitude > 0 else 0
    
    # 6. 均方频率 (MSF - Mean Square Frequency)
    msf = np.sum((freqs**2) * fft_magnitude) / np.sum(fft_magnitude)
    
    # 7. 均方根频率 (RMSF - Root Mean Square Frequency)
    rmsf = np.sqrt(msf) if msf > 0 else 0
    
    # 8. 频率方差 (VF - Variance Frequency)
    mean_freq = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
    vf = np.sum(((freqs - mean_freq)**2) * fft_magnitude) / np.sum(fft_magnitude)
    
    # 9. 根方差频率 (RVF - Root Variance Frequency)
    rvf = np.sqrt(vf) if vf > 0 else 0
    
    # 组合所有特征
    features = np.array([
        mean_value,      # 时域特征1: 均值
        std_value,       # 时域特征2: 标准差
        rms_value,       # 时域特征3: 均方根值
        crest_factor,    # 时域特征4: 峰值因子
        cf,              # 频域特征1: 频域峰值因子
        msf,             # 频域特征2: 均方频率
        rmsf,            # 频域特征3: 均方根频率
        vf,              # 频域特征4: 频率方差
        rvf              # 频域特征5: 根方差频率
    ])
    
    return features

# import numpy as np
# from FrFT import frft
# from pyts.image import GramianAngularField


# def extract_all(dataset='HIT', samples_num=1000):
#     """
#     一体化脚本：
#     1) 抽取原始信号（每类 samples_num 样本，固定 seed=42）
#     2) 保存抽样后的原始信号与标签
#     3) 基于抽样信号生成 FrFT-GADF 并保存
#     """

#     # =============================
#     # 1. 数据路径设置
#     # =============================
#     if dataset.upper() == 'HIT':
#         print("载入 HIT 原始数据...")

#         src_data_path = 'data/hit/hit_bearing_data.npy'
#         src_label_path = 'data/hit/hit_bearing_label.npy'

#         bearing = np.load(src_data_path)            # (N,5,2048)
#         labels = np.load(src_label_path)
#         signal = bearing[:, 4, :]                   # sensor #5

#         # 保存抽样后的信号（最终 main.py 使用的版本）
#         save_signal_path = 'data/hit/hit_bearing_data1q.npy'
#         save_label_path  = 'data/hit/hit_bearing_label1q.npy'

#         # 保存 FrFT-GADF 结果
#         save_gadf_path   = 'data/hit/HIT_gadf_images.npy'
#         save_gadf_label  = 'data/hit/HIT_gadf_labels.npy'

#     elif dataset.upper() == 'HUST':
#         print("载入 HUST 原始数据...")

#         src_data_path = 'data/hust/HUST_bearing_data_prossessed.npy'
#         src_label_path = 'data/hust/HUST_bearing_label.npy'

#         bearing = np.load(src_data_path)            # (N,?,2048)
#         labels  = np.load(src_label_path)
#         signal = bearing[:, 1, :]                   # sensor #2

#         save_signal_path = 'data/hust/HUST_bearing_data_prossessed1q.npy'
#         save_label_path  = 'data/hust/HUST_bearing_label1q.npy'

#         save_gadf_path   = 'data/hust/HUST_gadf_images.npy'
#         save_gadf_label  = 'data/hust/HUST_gadf_labels.npy'

#     else:
#         raise ValueError("dataset 必须为 HIT 或 HUST")

#     print(f"{dataset} 原始样本数: {len(labels)}")

#     # =============================
#     # 2. 抽样（固定种子）
#     # =============================
#     np.random.seed(42)
#     unique_labels = np.unique(labels)
#     print(f"共有 {len(unique_labels)} 类别，每类抽 {samples_num} 个...")

#     selected_samples = []
#     selected_labels = []

#     for lab in unique_labels:
#         idx = np.where(labels == lab)[0]

#         if len(idx) <= samples_num:
#             chosen = idx
#         else:
#             chosen = np.random.choice(idx, samples_num, replace=False)

#         selected_samples.extend(signal[chosen])
#         selected_labels.extend(labels[chosen])

#     selected_samples = np.array(selected_samples)   # (C*1000, 2048)
#     selected_labels  = np.array(selected_labels)

#     print(f"抽样后总样本 {len(selected_labels)}")

#     # reshape 为 main.py 格式 (N,1,2048)
#     selected_samples_reshaped = selected_samples.reshape(
#         -1, 1, selected_samples.shape[-1]
#     )

#     # =============================
#     # 3. 保存抽样后的信号
#     # =============================
#     print("保存抽样后的信号与标签...")
#     np.save(save_signal_path, selected_samples_reshaped)
#     np.save(save_label_path,  selected_labels)

#     print(f"信号保存: {save_signal_path}")
#     print(f"标签保存: {save_label_path}")

#     # =============================
#     # 4. FrFT + GADF
#     # =============================
#     print("开始生成 FrFT-GADF 图像...")

#     a_values = np.arange(0, 1.1, 0.1)   # 11 阶
#     gadf = GramianAngularField(image_size=128, method='difference')

#     gadf_images = np.zeros(
#         (len(selected_samples), len(a_values), 128, 128)
#     )

#     for i, sig in enumerate(selected_samples):
#         if i % 20 == 0:
#             print(f"GADF 处理进度：{i}/{len(selected_samples)}")

#         for a_idx, a in enumerate(a_values):
#             # FRFT
#             frft_result = frft(sig, a)
#             frft_signal = np.abs(frft_result)

#             # GADF 要求归一化 [-1,1]
#             min_v, max_v = frft_signal.min(), frft_signal.max()
#             if max_v > min_v:
#                 frft_signal = 2 * (frft_signal - min_v) / (max_v - min_v) - 1

#             gadf_image = gadf.fit_transform(frft_signal.reshape(1, -1))

#             gadf_images[i, a_idx] = gadf_image[0]

#     # =============================
#     # 5. 保存 GADF 结果
#     # =============================
#     print("保存 GADF 图像与标签...")
#     np.save(save_gadf_path, gadf_images)
#     np.save(save_gadf_label, selected_labels)

#     print(f"GADF 图像保存到: {save_gadf_path}")
#     print(f"GADF 标签保存到: {save_gadf_label}")
#     print("==== 全部处理完成！====\n")


# if __name__ == '__main__':
#     extract_all('HIT', samples_num=300)
#     extract_all('HUST', samples_num=300)
import numpy as np
from FrFT import frft
from pyts.image import GramianAngularField


def add_noise_by_snr(signal, snr_db):
    """
    根据 SNR(dB) 给信号加高斯白噪声
    SNR = 10 * log10(P_signal / P_noise)
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def extract_all(dataset='HIT', samples_num=1000, noise_snr_db=None):
    """
    noise_snr_db: float or None
        为 None 时不加噪声
        设置为 30~40dB 时可使模型轻微下降 1%
    """

    # =============================
    # 1. 数据路径设置
    # =============================
    if dataset.upper() == 'HIT':
        print("载入 HIT 原始数据...")

        src_data_path = 'data/hit/hit_bearing_data.npy'
        src_label_path = 'data/hit/hit_bearing_label.npy'

        bearing = np.load(src_data_path)            # (N,5,2048)
        labels = np.load(src_label_path)
        signal = bearing[:, 4, :]                   # sensor #5

        save_signal_path = 'data/hit/hit_bearing_data1q.npy'
        save_label_path  = 'data/hit/hit_bearing_label1q.npy'

        save_gadf_path   = 'data/hit/HIT_gadf_images.npy'
        save_gadf_label  = 'data/hit/HIT_gadf_labels.npy'

    elif dataset.upper() == 'HUST':
        print("载入 HUST 原始数据...")

        src_data_path = 'data/hust/HUST_bearing_data_prossessed.npy'
        src_label_path = 'data/hust/HUST_bearing_label.npy'

        bearing = np.load(src_data_path)            # (N,?,2048)
        labels  = np.load(src_label_path)
        signal = bearing[:, 1, :]                   # sensor #2

        save_signal_path = 'data/hust/HUST_bearing_data_prossessed1q.npy'
        save_label_path  = 'data/hust/HUST_bearing_label1q.npy'

        save_gadf_path   = 'data/hust/HUST_gadf_images.npy'
        save_gadf_label  = 'data/hust/HUST_gadf_labels.npy'

    else:
        raise ValueError("dataset 必须为 HIT 或 HUST")

    print(f"{dataset} 原始样本数: {len(labels)}")

    # =============================
    # 2. 抽样（固定种子）
    # =============================
    np.random.seed(42)
    unique_labels = np.unique(labels)
    print(f"共有 {len(unique_labels)} 类别，每类抽 {samples_num} 个...")

    selected_samples = []
    selected_labels = []

    for lab in unique_labels:
        idx = np.where(labels == lab)[0]

        if len(idx) <= samples_num:
            chosen = idx
        else:
            chosen = np.random.choice(idx, samples_num, replace=False)

        selected_samples.extend(signal[chosen])
        selected_labels.extend(labels[chosen])

    selected_samples = np.array(selected_samples)
    selected_labels  = np.array(selected_labels)

    print(f"抽样后总样本 {len(selected_labels)}")

    # reshape 为 main.py 格式 (N,1,2048)
    selected_samples_reshaped = selected_samples.reshape(
        -1, 1, selected_samples.shape[-1]
    )

    # =============================
    # 3. 保存抽样后的信号
    # =============================
    print("保存抽样后的信号与标签...")
    np.save(save_signal_path, selected_samples_reshaped)
    np.save(save_label_path,  selected_labels)

    print(f"信号保存: {save_signal_path}")
    print(f"标签保存: {save_label_path}")

    # =============================
    # 4. FrFT + GADF
    # =============================
    print("开始生成 FrFT-GADF 图像...")

    a_values = np.arange(0, 1.1, 0.1)
    gadf = GramianAngularField(image_size=128, method='difference')

    gadf_images = np.zeros(
        (len(selected_samples), len(a_values), 128, 128)
    )

    for i, sig in enumerate(selected_samples):
        if i % 20 == 0:
            print(f"GADF 处理进度：{i}/{len(selected_samples)}")

        # -----------★ 这里加入噪声（仅此行）★-----------
        if noise_snr_db is not None:
            sig = add_noise_by_snr(sig, noise_snr_db)
        # ----------------------------------------------

        for a_idx, a in enumerate(a_values):
            frft_result = frft(sig, a)
            frft_signal = np.abs(frft_result)

            min_v, max_v = frft_signal.min(), frft_signal.max()
            if max_v > min_v:
                frft_signal = 2 * (frft_signal - min_v) / (max_v - min_v) - 1

            gadf_image = gadf.fit_transform(frft_signal.reshape(1, -1))
            gadf_images[i, a_idx] = gadf_image[0]

    # =============================
    # 5. 保存 GADF 结果
    # =============================
    print("保存 GADF 图像与标签...")
    np.save(save_gadf_path, gadf_images)
    np.save(save_gadf_label, selected_labels)

    print(f"GADF 图像保存到: {save_gadf_path}")
    print(f"GADF 标签保存到: {save_gadf_label}")
    print("==== 全部处理完成！====\n")


if __name__ == '__main__':
    # extract_all('HIT', samples_num=300, noise_snr_db=10)
    extract_all('HUST', samples_num=300, noise_snr_db=None)

import numpy as np
import sys

from FrFT import frft
from pyts.image import GramianAngularField

def data_posseed(name='HIT',samples_num=1000):
    if name == 'HIT':
        HIT_bearing = np.load('data/hit/hit_bearing_data.npy')
        HIT_label = np.load('data/hit/hit_bearing_label.npy')
        sensor5_signal = HIT_bearing[:, 4, :]
        np.random.seed(42)
        a_values = np.arange(0, 1.1, 0.1)
        unique_labels = np.unique(HIT_label)
        print(f"共有 {len(unique_labels)} 个类别")
        samples_per_class = samples_num
        # 创建保存抽样后样本的列表
        selected_samples = []
        selected_labels = []
        # 从每个类别中随机抽取样本
        for label in unique_labels:
            # 找出属于当前类别的样本索引
            indices = np.where(HIT_label == label)[0]

            # 如果该类别样本数量小于1000，则全部选取，否则随机抽取1000个
            if len(indices) <= samples_per_class:
                selected_indices = indices
            else:
                selected_indices = np.random.choice(indices, samples_per_class, replace=False)

            # 将选中的样本和标签添加到列表中
            for idx in selected_indices:
                selected_samples.append(sensor5_signal[idx])
                selected_labels.append(HIT_label[idx])

        # 转换为numpy数组
        selected_samples = np.array(selected_samples)
        selected_labels = np.array(selected_labels)

        print(f"抽样后共有 {len(selected_samples)} 个样本")

        # 初始化GramianAngularField对象，设置图像大小为128x128
        gadf = GramianAngularField(image_size=128, method='difference')

        # 创建保存GADF图像的数组
        # 每个样本有len(a_values)个不同阶次的GADF图像
        # 形状为 [样本数, 阶次数, 图像高度, 图像宽度]
        gadf_images = np.zeros((len(selected_samples), len(a_values), 128, 128))

        # 对每个样本进行处理
        for sample_idx in range(len(selected_samples)):
            if sample_idx % 10 == 0:
                print(f"正在处理第 {sample_idx}/{len(selected_samples)} 个样本...")

            # 获取当前样本
            current_sample = selected_samples[sample_idx]

            # 进行不同阶次的分数阶傅里叶变换并转换为GADF
            for a_idx, a in enumerate(a_values):
                # 计算分数阶傅里叶变换
                frft_result = frft(current_sample, a)
                frft_signal = np.abs(frft_result)

                frft_signal = frft_signal[:]

                # 归一化到[-1,1]之间，这是GADF的要求
                min_val = np.min(frft_signal)
                max_val = np.max(frft_signal)
                if max_val > min_val:  # 避免除以零
                    frft_signal = 2 * (frft_signal - min_val) / (max_val - min_val) - 1

                # 转换为所需的形状并计算GADF
                frft_signal = frft_signal.reshape(1, -1)
                gadf_image = gadf.fit_transform(frft_signal)

                # 保存GADF图像
                gadf_images[sample_idx, a_idx] = gadf_image[0]

        print("所有样本处理完成！")

        # 保存GADF图像数据和标签
        np.save('data/hit/HIT_gadf_images.npy', gadf_images)
        np.save('data/hit/HIT_gadf_labels.npy', selected_labels)
        print("GADF图像数据已保存为 HIT_gadf_images.npy")
        print("对应标签已保存为 HIT_gadf_labels.npy")

        # 输出GADF图像的形状
        print(f"gadf_images 形状: {gadf_images.shape}")
        print(f"selected_labels 形状: {selected_labels.shape}")
    else:
        HUST_bearing = np.load('data/hust/HUST_bearing_data_prossessed.npy')
        HUST_label = np.load('data/hust/HUST_bearing_label.npy')
        sensor5_signal = HUST_bearing[:, 1, :]
        print("开始处理样本...")

        # 固定随机种子
        np.random.seed(42)

        # 生成从0到1的阶次，步长为0.1
        a_values = np.arange(0, 1.1, 0.1)  # 11个阶次

        # 获取所有类别
        unique_labels = np.unique(HUST_label)
        print(f"共有 {len(unique_labels)} 个类别")

        # 每个类别抽取的样本数
        samples_per_class = samples_num

        # 创建保存抽样后样本的列表
        selected_samples = []
        selected_labels = []

        # 从每个类别中随机抽取样本
        for label in unique_labels:
            # 找出属于当前类别的样本索引
            indices = np.where(HUST_label == label)[0]

            # 如果该类别样本数量小于1000，则全部选取，否则随机抽取1000个
            if len(indices) <= samples_per_class:
                selected_indices = indices
            else:
                selected_indices = np.random.choice(indices, samples_per_class, replace=False)

            # 将选中的样本和标签添加到列表中
            for idx in selected_indices:
                selected_samples.append(sensor5_signal[idx])
                selected_labels.append(HUST_label[idx])

        # 转换为numpy数组
        selected_samples = np.array(selected_samples)
        selected_labels = np.array(selected_labels)

        print(f"抽样后共有 {len(selected_samples)} 个样本")

        # 初始化GramianAngularField对象，设置图像大小为128x128
        gadf = GramianAngularField(image_size=128, method='difference')

        # 创建保存GADF图像的数组
        # 每个样本有len(a_values)个不同阶次的GADF图像
        # 形状为 [样本数, 阶次数, 图像高度, 图像宽度]
        gadf_images = np.zeros((len(selected_samples), len(a_values), 128, 128))

        # 对每个样本进行处理
        for sample_idx in range(len(selected_samples)):
            if sample_idx % 10 == 0:
                print(f"正在处理第 {sample_idx}/{len(selected_samples)} 个样本...")

            # 获取当前样本
            current_sample = selected_samples[sample_idx]

            # 进行不同阶次的分数阶傅里叶变换并转换为GADF
            for a_idx, a in enumerate(a_values):
                # 计算分数阶傅里叶变换
                frft_result = frft(current_sample, a)
                frft_signal = np.abs(frft_result)

                frft_signal = frft_signal[:]

                # 归一化到[-1,1]之间，这是GADF的要求
                min_val = np.min(frft_signal)
                max_val = np.max(frft_signal)
                if max_val > min_val:  # 避免除以零
                    frft_signal = 2 * (frft_signal - min_val) / (max_val - min_val) - 1

                # 转换为所需的形状并计算GADF
                frft_signal = frft_signal.reshape(1, -1)
                gadf_image = gadf.fit_transform(frft_signal)

                # 保存GADF图像
                gadf_images[sample_idx, a_idx] = gadf_image[0]

        print("所有样本处理完成！")

        # 保存GADF图像数据和标签
        np.save('data/hust/HUST_gadf_images.npy', gadf_images)
        np.save('data/hust/HUST_gadf_labels.npy', selected_labels)
        print("GADF图像数据已保存为 HUST_gadf_images.npy")
        print("对应标签已保存为 HUST_gadf_labels.npy")

        # 输出GADF图像的形状
        print(f"gadf_images 形状: {gadf_images.shape}")
        print(f"selected_labels 形状: {selected_labels.shape}")
if __name__ == '__main__':
    data_posseed('HUST',samples_num=200)
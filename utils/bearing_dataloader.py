import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BearingDataset(Dataset):
    """
    轴承故障数据集类
    """
    def __init__(self, data, labels):
        """
        初始化数据集
        
        参数:
        data: 特征数据 (n_samples, sequence_length)
        labels: 标签 (n_samples,)
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
        print(f"数据集初始化完成:")
        print(f"数据形状: {self.data.shape}")
        print(f"标签形状: {self.labels.shape}")
        print(f"标签范围: {self.labels.min()} - {self.labels.max()}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
        idx: 样本索引
        
        返回:
        sample: 特征数据，形状为 (sequence_length,)
        label: 对应标签
        """
        sample = self.data[idx]
        label = self.labels[idx]
        
        # 为一维卷积添加通道维度 (sequence_length,) -> (1, sequence_length)
        # sample = sample.unsqueeze(0)
        
        return sample, label

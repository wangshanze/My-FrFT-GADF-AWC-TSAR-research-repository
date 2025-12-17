import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class AdaptiveWeighting(nn.Module):
    def __init__(self, input_channels=22, output_channels=32):
        super(AdaptiveWeighting, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        

        self.channel_statistics = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(input_channels, input_channels*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(input_channels*2, input_channels, kernel_size=1)
        )
        

        self.correlation_net = nn.Sequential(
            nn.Conv2d(input_channels, input_channels*2, kernel_size=3, padding=1, groups=input_channels),
            nn.BatchNorm2d(input_channels*2),
            nn.ReLU(),
            nn.Conv2d(input_channels*2, input_channels*input_channels, kernel_size=1)
        )
        

        self.weight_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, input_channels, kernel_size=1),
            nn.Softmax(dim=1)
        )
        

        self.channels_per_order = max(1, output_channels // input_channels)
        self.extra_channels = output_channels - (self.channels_per_order * input_channels)
        

        self.transform = nn.ModuleList([
            nn.Conv2d(1, self.channels_per_order + (1 if i < self.extra_channels else 0), kernel_size=3, padding=1) 
            for i in range(input_channels)
        ])
        

        self.pre_proj = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        

        self.output_proj = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size, channels, h, w = x.shape
        
        
        # 1. 计算通道统计信息
        channel_stats = self.channel_statistics(x)  # [B, C, 1, 1]
        
        # 2. 计算通道相关性矩阵
        corr = self.correlation_net(x)  # [B, C*C, H, W]
        corr = corr.view(batch_size, self.input_channels, self.input_channels, h, w)
        corr = corr.mean(dim=[-1, -2])  # [B, C, C]
        
        # 3. 计算自适应权重
        weights = self.weight_net(x)  # [B, C, 1, 1]
        
        # 4. 调整权重
        mutual_info = torch.bmm(corr, channel_stats.squeeze(-1).squeeze(-1).unsqueeze(-1))  # [B, C, 1]
        mutual_info = F.softmax(mutual_info.squeeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 融合两种权重
        final_weights = 0.7 * weights + 0.3 * mutual_info
        final_weights = F.softmax(final_weights, dim=1)
        
        # 5. 使用1x1卷积直接调整通道数
        output = self.pre_proj(x)  # [B, output_channels, H, W]
        
        # 6. 应用归一化和激活
        output = self.output_proj(output)
        
        return output, final_weights
    
def extract_foaw_features(model, data_loader, num_samples=20):
    """提取分数阶自适应加权层的权重"""
    model.eval()
    all_weights = []
    
    # 原始前向传播
    original_forward = model.foaw.forward
    

    def modified_forward(self, x):
        batch_size, channels, h, w = x.shape
        
        # 计算通道统计信息
        channel_stats = self.channel_statistics(x)  # [B, C, 1, 1]
        
        # 计算通道相关性矩阵
        corr = self.correlation_net(x)  # [B, C*C, H, W]
        corr = corr.view(batch_size, self.input_channels, self.input_channels, h, w)
        corr = corr.mean(dim=[-1, -2])  # [B, C, C]
        
        # 计算自适应权重
        weights = self.weight_net(x)  # [B, C, 1, 1]
        
        # 计算通道间的互信息得分
        mutual_info = torch.bmm(corr, channel_stats.squeeze(-1).squeeze(-1).unsqueeze(-1))  # [B, C, 1]
        mutual_info = F.softmax(mutual_info.squeeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 融合两种权重
        final_weights = 0.7 * weights + 0.3 * mutual_info
        final_weights = F.softmax(final_weights, dim=1)
        
        # 使用1x1卷积直接调整通道数，同时保持输入的空间维度
        output = self.pre_proj(x)  # [B, output_channels, H, W]
        
        # 应用归一化和激活
        output = self.output_proj(output)
        
        return output, final_weights
    
    # 替换前向传播
    import types
    model.foaw.forward = types.MethodType(modified_forward, model.foaw)
    
    # 收集样本数据
    samples_collected = 0
    with torch.no_grad():
        for inputs, _ in data_loader:
            if samples_collected >= num_samples:
                break
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # 前向传播，获取权重
            _, weights = model.foaw(inputs)
            
            # 只收集我们需要的样本数量
            actual_samples = min(batch_size, num_samples - samples_collected)
            
            # 收集权重
            all_weights.append(weights[:actual_samples].cpu())
            
            samples_collected += actual_samples
    
    # 恢复原始前向传播函数
    model.foaw.forward = original_forward
    
    # 合并收集的数据
    weights = torch.cat(all_weights, dim=0)
    
    return weights

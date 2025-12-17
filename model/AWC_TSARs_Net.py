import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
sys.path.append("..")
from model.AWC import AdaptiveWeighting
from model.TSAR import TSAR

class GADFMultiChannelNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=22): 
        super(GADFMultiChannelNet, self).__init__()


        self.foaw = AdaptiveWeighting(input_channels=input_channels, output_channels=32)

        # 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 堆叠双路径块
        self.block1 = TSAR(32, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.1)

        self.block2 = TSAR(64, 96)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.1)

        self.block3 = TSAR(96, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.2)

        self.block4 = TSAR(128, 192)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.dropout4 = nn.Dropout2d(0.2)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # 存储阶次权重，用于可视化
        self.order_weights = None

    def forward(self, x):

        x, weights = self.foaw(x)
        self.order_weights = weights  # 保存用于可视化

        # 初始特征提取
        x = self.initial_conv(x)

        # 第一个双路径块
        x = self.block1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # 第二个双路径块
        x = self.block2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # 第三个双路径块
        x = self.block3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # 第四个双路径块
        x = self.block4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        # 全局池化
        x = self.global_pool(x)

        # 分类
        output = self.classifier(x)

        return output

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    固定所有随机种子，确保实验可重现
    
    参数:
    seed: 随机种子数值
    """
    # 1. 设置Python随机种子
    random.seed(seed)
    
    # 2. 设置NumPy随机种子
    np.random.seed(seed)
    
    # 3. 设置PyTorch随机种子
    torch.manual_seed(seed)
    
    # 4. 如果使用GPU，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU环境
    
    # 5. 设置cuDNN（影响卷积操作性能）
    cudnn.benchmark = False  # 关闭自动优化，保证结果可重现
    cudnn.deterministic = True  # 使用确定性算法
    
    # 6. 设置环境变量（影响一些底层操作）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ 随机种子已固定为: {seed}")
    print(f"🔧 CUDA可用: {torch.cuda.is_available()}")


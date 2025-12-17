import numpy as np

def get_samples_by_class(data, labels, samples_per_class=300, random_seed=42):
    """
    简单版本：从每个类别取出指定数量的样本
    
    参数:
    data: 原始数据 (n_samples, n_features)
    labels: 标签 (n_samples,)
    samples_per_class: 每个类别要取的样本数
    random_seed: 随机种子，默认42
    
    返回:
    selected_data: 选择的数据
    selected_labels: 对应的标签
    """
    
    # 设置随机种子，确保每次运行结果相同
    np.random.seed(random_seed)
    
    print("=== 开始按类别选择样本 ===")
    print(f"使用随机种子: {random_seed}")
    
    # 1. 统计每个类别有多少样本
    unique_classes = np.unique(labels)
    print(f"发现的类别: {unique_classes}")
    
    # 2. 为每个类别收集索引
    all_selected_indices = []
    
    for class_num in unique_classes:
        # 找到当前类别的所有索引
        class_indices = np.where(labels == class_num)[0]
        print(f"类别 {class_num}: 找到 {len(class_indices)} 个样本")
        
        # 随机选择指定数量的索引
        if len(class_indices) >= samples_per_class:
            # 如果样本足够，随机选择300个
            selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            # 如果样本不够，就全选
            selected_indices = class_indices
            print(f"  警告：类别 {class_num} 样本不够，只选择了 {len(class_indices)} 个")
        
        print(f"  类别 {class_num}: 选择了 {len(selected_indices)} 个样本")
        all_selected_indices.extend(selected_indices)
    
    # 3. 根据选中的索引提取数据
    all_selected_indices = np.array(all_selected_indices)
    selected_data = data[all_selected_indices]
    selected_labels = labels[all_selected_indices]
    
    print(f"\n=== 选择完成 ===")
    print(f"总共选择了 {len(all_selected_indices)} 个样本")
    print(f"数据形状: {selected_data.shape}")
    print(f"标签形状: {selected_labels.shape}")
    
    # 4. 验证每个类别的数量
    print("\n=== 验证结果 ===")
    for class_num in unique_classes:
        count = np.sum(selected_labels == class_num)
        print(f"类别 {class_num}: {count} 个样本")
    
    return selected_data, selected_labels

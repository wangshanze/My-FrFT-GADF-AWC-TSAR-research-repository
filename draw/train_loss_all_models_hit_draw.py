import pandas as pd

# 读取 CSV 文件
csv_path = 'draw/train_loss_all_models_hust.csv'
df = pd.read_csv(csv_path)

# 查看前几行
print(df.head())

# 获取 epoch 列
epochs = df.iloc[:, 0]

# 获取模型名称（除第一列）
model_names = df.columns[1:]

print("Models:", model_names.tolist())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

for model in model_names:
    plt.plot(epochs, df[model], label=model,linewidth=3)

plt.tick_params(axis='both', labelsize=18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Training Loss', fontsize=18)
plt.ylim(0, 2)
plt.legend(loc='upper right', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('draw/train_loss_curve_hust.pdf', dpi=300, bbox_inches='tight')
plt.show()

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from utils.bearing_dataloader import BearingDataset
from utils.set_seed import set_seed
from model.AWC_TSARs_Net import GADFMultiChannelNet
from model.frft_gadf_resnet import FrFTGADFResNet
from model.frft_gadf_vgg16 import FrFTGADFVGG16
from model.frft_gadf_densenet import FrFTGADFDenseNet
from model.wavelet_cnn import WaveletCNN 
from model.adaptive_cnn import AdaptiveCNN
from model.lmswt_se_mscnn import LMSWT_SE_MSCNN
from model.marp_resnet import MARPResNet34


set_seed(42)
dataset_choosen = "hit" 

EPOCHS = 100
BATCH_SIZE = 64
N_RUNS = 5
BASE_SEED = 42
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

RESULT_ROOT = "Result"
os.makedirs(RESULT_ROOT, exist_ok=True)


if dataset_choosen == "hit":
    bearing_signal = np.load("data/hit/hit_bearing_data1q.npy")
    bearing_label = np.load("data/hit/hit_bearing_label1q.npy")
    X = np.load("data/hit/HIT_gadf_images.npy")
    y = np.load("data/hit/HIT_gadf_labels.npy")

    X = X.transpose(0, 1, 2, 3) 
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    print(f"GADF 数据形状: {X.shape}")
    print(f"GADF 标签形状: {y.shape}")
    num_cata = 4

elif dataset_choosen == "hust":
    bearing_signal = np.load("data/hust/HUST_bearing_data_prossessed1q.npy")
    bearing_label = np.load("data/hust/HUST_bearing_label1q.npy")

    X = np.load("data/hust/HUST_gadf_images.npy")
    y = np.load("data/hust/HUST_gadf_labels.npy")

    X = X.transpose(0, 1, 2, 3)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    print(f"GADF 数据形状: {X.shape}")
    print(f"GADF 标签形状: {y.shape}")
    num_cata = 9

else:
    raise ValueError(f"Unknown dataset_choosen: {dataset_choosen}")

# 确保 GADF 与 原始信号/标签 对应同一顺序
assert len(X) == len(bearing_signal) == len(y) == len(bearing_label)

# 结果目录
DATASET_RESULT_DIR = os.path.join(RESULT_ROOT, dataset_choosen)
os.makedirs(DATASET_RESULT_DIR, exist_ok=True)

indices = np.arange(len(y))
idx_train, idx_test, _, _ = train_test_split(
    indices,
    y,
    test_size=0.3,
    random_state=BASE_SEED,
    stratify=y,
)

# GADF 图像划分
X_train_gadf = X[idx_train]
X_test_gadf = X[idx_test]
y_train_gadf = y[idx_train]
y_test_gadf = y[idx_test]

N, C, L = bearing_signal.shape
signal_2d = bearing_signal.reshape(N, L)   # (N, 2048)

scaler = StandardScaler()
signal_scaled = scaler.fit_transform(signal_2d)
signal_scaled = signal_scaled.reshape(N, 1, L)

X_train_sig = signal_scaled[idx_train]
X_test_sig = signal_scaled[idx_test]
y_train_sig = bearing_label[idx_train]
y_test_sig = bearing_label[idx_test]

print(f"训练集 GADF: {X_train_gadf.shape}, 测试集 GADF: {X_test_gadf.shape}")
print(f"训练集 信号: {X_train_sig.shape}, 测试集 信号: {X_test_sig.shape}")

# ==========================
# 构造 DataLoader
# ==========================
train_dataset_gadf = BearingDataset(X_train_gadf, y_train_gadf)
test_dataset_gadf = BearingDataset(X_test_gadf, y_test_gadf)

train_loader_gadf = DataLoader(
    train_dataset_gadf, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader_gadf = DataLoader(
    test_dataset_gadf, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

train_dataset_sig = BearingDataset(X_train_sig, y_train_sig)
test_dataset_sig = BearingDataset(X_test_sig, y_test_sig)

train_loader_sig = DataLoader(
    train_dataset_sig, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader_sig = DataLoader(
    test_dataset_sig, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

input_channels_gadf = X.shape[1]
signal_length = signal_scaled.shape[-1]
print(f"GADF 输入通道数: {input_channels_gadf}")
print(f"原始信号长度: {signal_length}")


# ==========================
# 训练 & 评估函数
# ==========================
def train_one_model(create_model_fn, train_loader, test_loader, num_epochs, lr, seed):
    if seed is not None:
        set_seed(seed)

    model = create_model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=3
    )

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    best_test_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # ---------- Train ----------
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        # ---------- Eval ----------
        model.eval()
        running_test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)

                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_accuracy = 100.0 * test_correct / test_total

        scheduler.step(avg_test_loss)

        train_loss_history.append(avg_train_loss)
        test_loss_history.append(avg_test_loss)
        train_acc_history.append(train_accuracy)
        test_acc_history.append(test_accuracy)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_state = model.state_dict()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
            f"| Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
        )

    history = {
        "train_loss": train_loss_history,
        "test_loss": test_loss_history,
        "train_acc": train_acc_history,
        "test_acc": test_acc_history,
    }
    return best_state, history


def evaluate_with_metrics(model, test_loader):
    """
    使用当前模型在 test_loader 上计算：
    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - F1 (macro)
    以及混淆矩阵 & 预测标签
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            running_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

    avg_test_loss = running_test_loss / len(test_loader)
    acc = 100.0 * test_correct / test_total

    all_targets_arr = np.array(all_targets)
    all_preds_arr = np.array(all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets_arr, all_preds_arr, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_targets_arr, all_preds_arr)

    # 转换为百分比
    return (
        avg_test_loss,
        acc,
        precision * 100.0,
        recall * 100.0,
        f1 * 100.0,
        cm,
        all_targets_arr,
        all_preds_arr,
    )


# ==========================
# 定义 6 个实验配置
# ==========================
experiments = [
    {
        "name": "FrFT-GADF-AWC-TSAR",
        "input_type": "gadf",
        "lr": 0.0005,
        "create_model": lambda: GADFMultiChannelNet(
            num_classes=num_cata, input_channels=input_channels_gadf
        ),
    },
    
    {
        "name": "FrFT-GADF-ResNet",
        "input_type": "gadf",
        "lr": 0.0005,
        "create_model": lambda: FrFTGADFResNet(
            num_classes=num_cata, input_channels=input_channels_gadf
        ),
    },
    
    {
        "name": "FrFT-GADF-VGG16",
        "input_type": "gadf",
        "lr": 1e-3,
        "create_model": lambda: FrFTGADFVGG16(
            num_classes=num_cata, input_channels=input_channels_gadf
        ),
    },
    
    {
        "name": "FrFT-GADF-DenseNet",
        "input_type": "gadf",
        "lr": 0.0005,
        "create_model": lambda: FrFTGADFDenseNet(
            num_classes=num_cata, input_channels=input_channels_gadf
        ),
    },
    
    {
        "name": "Wavelet-CNN",
        "input_type": "signal",
        "lr": 0.0005,
        "create_model": lambda: WaveletCNN(
            num_classes=num_cata, wavelet="db4", dwt_level=3
        ),
    },
    
    {
        "name": "Adaptive-CNN",
        "input_type": "signal",
        "lr": 0.0005,
        "create_model": lambda: AdaptiveCNN(num_classes=num_cata, in_channels=1),
    },
    
 {
        "name": "LMSWT-SE-MSCNN",
        "input_type": "signal",
        "lr": 0.0005,
        "create_model": lambda: LMSWT_SE_MSCNN(
            num_classes=num_cata,
            in_channels=1,
            signal_length=signal_length,
            tf_size=32,
        ),
    },
    
    {
        "name": "MARP-ResNet34",
        "input_type": "signal",
        "lr": 1e-4,
        "create_model": lambda: MARPResNet34(
            num_classes=num_cata,
            in_channels=1,
            signal_length=signal_length,
            rp_size=64,
        ),
    },
]


# ==========================
# 跑所有实验 & 统计指标（5 次 mean ± std）
# ==========================
summary_records = []
best_run_histories = {}  
best_run_predictions = {}  

for exp in experiments:
    name = exp["name"]
    input_type = exp["input_type"]
    lr = exp["lr"]
    create_model = exp["create_model"]

    print("=" * 100)
    print(f"开始实验: {name}  (输入类型: {input_type}, lr={lr})")
    print("=" * 100)

    if input_type == "gadf":
        train_loader = train_loader_gadf
        test_loader = test_loader_gadf
    else:
        train_loader = train_loader_sig
        test_loader = test_loader_sig

    run_accs = []
    run_precs = []
    run_recalls = []
    run_f1s = []
    run_histories = []
    run_states = []
    run_cm = []
    run_targets_preds = []

    for run_idx in range(N_RUNS):
        print(f"\n---- {name} | Run {run_idx + 1}/{N_RUNS} ----")
        seed = BASE_SEED + run_idx

        best_state, history = train_one_model(
            create_model_fn=create_model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=EPOCHS,
            lr=lr,
            seed=seed,
        )

        # 使用最佳权重重新构建模型并评估指标
        model_best = create_model().to(device)
        model_best.load_state_dict(best_state)

        (
            avg_test_loss,
            acc,
            precision,
            recall,
            f1,
            cm,
            all_targets_arr,
            all_preds_arr,
        ) = evaluate_with_metrics(model_best, test_loader)

        print(
            f"[{name} | Run {run_idx + 1}] "
            f"Test Acc: {acc:.2f}%, "
            f"Precision(macro): {precision:.2f}%, "
            f"Recall(macro): {recall:.2f}%, "
            f"F1(macro): {f1:.2f}%"
        )

        run_accs.append(acc)
        run_precs.append(precision)
        run_recalls.append(recall)
        run_f1s.append(f1)
        run_histories.append(history)
        run_states.append(best_state)
        run_cm.append(cm)
        run_targets_preds.append((all_targets_arr, all_preds_arr))

    # 计算均值 ± 标准差
    run_accs = np.array(run_accs)
    run_precs = np.array(run_precs)
    run_recalls = np.array(run_recalls)
    run_f1s = np.array(run_f1s)

    acc_mean, acc_std = run_accs.mean(), run_accs.std()
    prec_mean, prec_std = run_precs.mean(), run_precs.std()
    recall_mean, recall_std = run_recalls.mean(), run_recalls.std()
    f1_mean, f1_std = run_f1s.mean(), run_f1s.std()

    print(
        f"\n##### 实验 {name} 完成: "
        f"Accuracy = {acc_mean:.2f} ± {acc_std:.2f} %, "
        f"Precision = {prec_mean:.2f} ± {prec_std:.2f} %, "
        f"Recall = {recall_mean:.2f} ± {recall_std:.2f} %, "
        f"F1 = {f1_mean:.2f} ± {f1_std:.2f} % #####\n"
    )

    summary_records.append(
        {
            "Dataset": dataset_choosen,
            "Model": name,
            "Accuracy": f"{acc_mean:.2f} ± {acc_std:.2f}",
            "Precision": f"{prec_mean:.2f} ± {prec_std:.2f}",
            "Recall": f"{recall_mean:.2f} ± {recall_std:.2f}",
            "F1-Score": f"{f1_mean:.2f} ± {f1_std:.2f}",
        }
    )

    # 选择最佳一次 run
    best_run_idx = int(run_accs.argmax())
    best_run_histories[name] = run_histories[best_run_idx]
    best_run_predictions[name] = run_targets_preds[best_run_idx]

print("\n所有实验完成！")


# ==========================
# 保存大结果表
# ==========================
summary_df = pd.DataFrame(summary_records)
summary_csv_path = os.path.join(
    DATASET_RESULT_DIR, f"summary_{dataset_choosen}.csv"
)
summary_df.to_csv(summary_csv_path, index=False)
print(f"结果表格已保存到: {summary_csv_path}")


# ==========================
# 绘图设置（全局）——Times New Roman 等
# ==========================
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.unicode_minus"] = False


def plot_multi_model_curve(
    histories,
    key,
    ylabel,
    save_name,
):
    """
    histories: {model_name: history_dict}
    key: 'train_loss' / 'test_loss' / 'train_acc' / 'test_acc'
    """

    # ==== 生成 CSV 文件 ====
    csv_path = os.path.join(DATASET_RESULT_DIR, save_name.replace(".png", ".csv"))
    epochs = np.arange(1, len(next(iter(histories.values()))[key]) + 1)

    df = pd.DataFrame({"Epoch": epochs})
    for model_name, history in histories.items():
        df[model_name] = history[key]
    df.to_csv(csv_path, index=False)
    print(f"曲线对应 CSV 已保存到: {csv_path}")

    # ==== 绘图 ====
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, history in histories.items():
        values = history[key]
        epochs_range = np.arange(1, len(values) + 1)
        ax.plot(epochs_range, values, label=model_name)

    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)

    # ----- 图例位置：Loss → 右上角；Accuracy → 右下角 -----
    if "loss" in key.lower():
        legend_loc = "upper right"
    else:
        legend_loc = "lower right"

    ax.legend(loc=legend_loc, fontsize=14)

    fig.tight_layout()
    save_path = os.path.join(DATASET_RESULT_DIR, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"曲线图已保存到: {save_path}")



# 4 张图：Train Loss / Test Loss / Train Acc / Test Acc
plot_multi_model_curve(
    best_run_histories,
    key="train_loss",
    ylabel="Training Loss",
    save_name=f"train_loss_all_models_{dataset_choosen}.png",
)

plot_multi_model_curve(
    best_run_histories,
    key="test_loss",
    ylabel="Test Loss",
    save_name=f"test_loss_all_models_{dataset_choosen}.png",
)

plot_multi_model_curve(
    best_run_histories,
    key="train_acc",
    ylabel="Training Accuracy (%)",
    save_name=f"train_acc_all_models_{dataset_choosen}.png",
)

plot_multi_model_curve(
    best_run_histories,
    key="test_acc",
    ylabel="Test Accuracy (%)",
    save_name=f"test_acc_all_models_{dataset_choosen}.png",
)


# ==========================
# 每个模型的混淆矩阵
# ==========================
for model_name, (all_targets_arr, all_preds_arr) in best_run_predictions.items():
    cm = confusion_matrix(all_targets_arr, all_preds_arr)

    fig, ax = plt.subplots(figsize=(6, 6))

    # ---- 使用黑白灰阶色图 ----
    im = ax.imshow(cm, cmap="Greys", interpolation="nearest")


    # ---- 坐标轴标签 ----
    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)

    ax.set_xticks(np.arange(num_cata))
    ax.set_yticks(np.arange(num_cata))
    ax.set_xticklabels(np.arange(num_cata))
    ax.set_yticklabels(np.arange(num_cata))

    # ---- Cell 中写红色数字 ----
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                color="red",
                fontweight="bold",
            )

    fig.tight_layout()
    cm_path = os.path.join(
        DATASET_RESULT_DIR,
        f"confusion_matrix_{model_name.replace(' ', '_')}_{dataset_choosen}.png",
    )
    fig.savefig(cm_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"{model_name} 混淆矩阵已保存到: {cm_path}")


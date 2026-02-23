import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from torch.utils.data import DataLoader



from utils.bearing_dataloader import BearingDataset
from utils.set_seed import set_seed

# 四种注意力变体模型
from model.Attn_TSARs_Net import (
    SEMultiChannelNet,
    ECAMultiChannelNet,
    CAMultiChannelNet,
    CBAMMultiChannelNet,
)

# ===========================================================================
# 全局配置
# ===========================================================================
set_seed(42)
dataset_choosen = "hust"

EPOCHS      = 100
BATCH_SIZE  = 64
N_RUNS      = 5
BASE_SEED   = 42
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

RESULT_ROOT = "Result_Supp"
os.makedirs(RESULT_ROOT, exist_ok=True)

# ===========================================================================
# 1. 数据加载
# ===========================================================================
if dataset_choosen == "hit":
    bearing_signal = np.load("data/hit/hit_bearing_data1q.npy")
    bearing_label  = np.load("data/hit/hit_bearing_label1q.npy")
    X = np.load("data/hit/HIT_gadf_images.npy")
    y = np.load("data/hit/HIT_gadf_labels.npy")
    num_cata = 4

elif dataset_choosen == "hust":
    bearing_signal = np.load("data/hust/HUST_bearing_data_prossessed1q.npy")
    bearing_label  = np.load("data/hust/HUST_bearing_label1q.npy")
    X = np.load("data/hust/HUST_gadf_images.npy")
    y = np.load("data/hust/HUST_gadf_labels.npy")
    num_cata = 9

else:
    raise ValueError(f"未知数据集: {dataset_choosen}")

X = X.transpose(0, 1, 2, 3)   # 确保 (N, C, H, W)
print(f"GADF 数据形状: {X.shape}，标签形状: {y.shape}")
assert len(X) == len(bearing_signal) == len(y) == len(bearing_label)

DATASET_RESULT_DIR = os.path.join(RESULT_ROOT, dataset_choosen)
os.makedirs(DATASET_RESULT_DIR, exist_ok=True)

# ===========================================================================
# 2. 划分训练 / 测试索引
# ===========================================================================
indices = np.arange(len(y))
idx_train, idx_test, _, _ = train_test_split(
    indices, y,
    test_size=0.3,
    random_state=BASE_SEED,
    stratify=y,
)

X_train_gadf = X[idx_train]
X_test_gadf  = X[idx_test]
y_train_gadf = y[idx_train]
y_test_gadf  = y[idx_test]

N, C, L = bearing_signal.shape
signal_2d    = bearing_signal.reshape(N, L)
scaler       = StandardScaler()
signal_scaled = scaler.fit_transform(signal_2d).reshape(N, 1, L)

print(f"训练集 GADF: {X_train_gadf.shape}  测试集 GADF: {X_test_gadf.shape}")

# ===========================================================================
# 3. DataLoader
# ===========================================================================
train_loader = DataLoader(
    BearingDataset(X_train_gadf, y_train_gadf),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    BearingDataset(X_test_gadf, y_test_gadf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

input_channels_gadf = X.shape[1]
print(f"GADF 输入通道数: {input_channels_gadf}")


# ===========================================================================
# 4. 训练 & 评估函数
# ===========================================================================
def train_one_model(create_model_fn, train_loader, test_loader,
                    num_epochs, lr, seed):
    if seed is not None:
        set_seed(seed)

    model     = create_model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=3
    )

    train_loss_hist, test_loss_hist = [], []
    train_acc_hist,  test_acc_hist  = [], []
    best_test_acc = 0.0
    best_state    = None

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total   += targets.size(0)
            correct += (pred == targets).sum().item()

        avg_train_loss = run_loss / len(train_loader)
        train_acc      = 100.0 * correct / total

        # ---- Eval ----
        model.eval()
        run_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                run_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total   += targets.size(0)
                correct += (pred == targets).sum().item()

        avg_test_loss = run_loss / len(test_loader)
        test_acc      = 100.0 * correct / total

        scheduler.step(avg_test_loss)
        train_loss_hist.append(avg_train_loss)
        test_loss_hist.append(avg_test_loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state    = model.state_dict()

        print(f"  Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}  Train Acc: {train_acc:.2f}%"
              f" | Test Loss: {avg_test_loss:.4f}  Test Acc: {test_acc:.2f}%")

    history = dict(
        train_loss=train_loss_hist, test_loss=test_loss_hist,
        train_acc=train_acc_hist,   test_acc=test_acc_hist,
    )
    return best_state, history


def evaluate_with_metrics(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    run_loss  = 0.0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            run_loss += criterion(outputs, targets).item()
            _, pred = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(pred.cpu().numpy().tolist())

    avg_loss  = run_loss / len(test_loader)
    t_arr, p_arr = np.array(all_targets), np.array(all_preds)
    acc = 100.0 * (t_arr == p_arr).sum() / len(t_arr)
    prec, rec, f1, _ = precision_recall_fscore_support(
        t_arr, p_arr, average="macro", zero_division=0
    )
    cm = confusion_matrix(t_arr, p_arr)
    return avg_loss, acc, prec * 100, rec * 100, f1 * 100, cm, t_arr, p_arr


# ===========================================================================
# 5. 实验配置：4 种注意力变体
# ===========================================================================
experiments = [
    {
        "name": "SE-TSARs-Net",
        "lr": 0.0003,
        "create_model": lambda: SEMultiChannelNet(
            num_classes=num_cata, input_channels=input_channels_gadf
        ),
    },
    {
        "name": "ECA-TSARs-Net",
        "lr": 0.0001,
        "create_model": lambda: ECAMultiChannelNet(
            num_classes=num_cata, input_channels=input_channels_gadf
        ),
    },
    {
        "name": "CA-TSARs-Net",
        "lr": 0.0005,
        "create_model": lambda: CAMultiChannelNet(
            num_classes=num_cata, input_channels=input_channels_gadf
        ),
    },
    {
        "name": "CBAM-TSARs-Net",
        "lr": 0.0003,
        "create_model": lambda: CBAMMultiChannelNet(
            num_classes=num_cata, input_channels=input_channels_gadf
        ),
    },
]


# ===========================================================================
# 6. 主循环：每个模型跑 N_RUNS 次，统计均值 ± 标准差
# ===========================================================================
summary_records    = []
best_run_histories = {}
best_run_preds     = {}

for exp in experiments:
    name         = exp["name"]
    lr           = exp["lr"]
    create_model = exp["create_model"]

    print("\n" + "=" * 80)
    print(f"实验: {name}  (lr={lr})")
    print("=" * 80)

    run_accs, run_precs, run_recs, run_f1s = [], [], [], []
    run_histories, run_targets_preds       = [], []

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

        model_best = create_model().to(device)
        model_best.load_state_dict(best_state)

        _, acc, prec, rec, f1, cm, t_arr, p_arr = evaluate_with_metrics(
            model_best, test_loader
        )
        print(f"  => Acc: {acc:.2f}%  Prec: {prec:.2f}%  "
              f"Recall: {rec:.2f}%  F1: {f1:.2f}%")

        run_accs.append(acc);  run_precs.append(prec)
        run_recs.append(rec);  run_f1s.append(f1)
        run_histories.append(history)
        run_targets_preds.append((t_arr, p_arr))

    run_accs  = np.array(run_accs)
    run_precs = np.array(run_precs)
    run_recs  = np.array(run_recs)
    run_f1s   = np.array(run_f1s)

    best_run_idx = int(run_accs.argmax())
    best_run_histories[name] = run_histories[best_run_idx]
    best_run_preds[name]     = run_targets_preds[best_run_idx]

    print(f"\n### {name} 汇总: "
          f"Acc={run_accs.mean():.2f}±{run_accs.std():.2f}%  "
          f"F1={run_f1s.mean():.2f}±{run_f1s.std():.2f}% ###")

    summary_records.append({
        "Dataset":   dataset_choosen,
        "Model":     name,
        "Accuracy":  f"{run_accs.mean():.2f} ± {run_accs.std():.2f}",
        "Precision": f"{run_precs.mean():.2f} ± {run_precs.std():.2f}",
        "Recall":    f"{run_recs.mean():.2f} ± {run_recs.std():.2f}",
        "F1-Score":  f"{run_f1s.mean():.2f} ± {run_f1s.std():.2f}",
    })

print("\n所有实验完成！")


# ===========================================================================
# 7. 保存汇总 CSV
# ===========================================================================
summary_df = pd.DataFrame(summary_records)
csv_path   = os.path.join(DATASET_RESULT_DIR,
                           f"summary_supp_{dataset_choosen}.csv")
summary_df.to_csv(csv_path, index=False)
print(f"汇总表格已保存: {csv_path}")
print(summary_df.to_string(index=False))


# ===========================================================================
# 8. 绘制训练曲线并保存 CSV
# ===========================================================================
plt.rcParams["font.family"]      = "DejaVu Serif"
plt.rcParams["axes.unicode_minus"] = False

def save_curve(histories, key, ylabel, fname_stem):
    epochs = np.arange(1, len(next(iter(histories.values()))[key]) + 1)
    df     = pd.DataFrame({"Epoch": epochs})
    for mname, h in histories.items():
        df[mname] = h[key]
    csv_out = os.path.join(DATASET_RESULT_DIR, fname_stem + ".csv")
    df.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for mname, h in histories.items():
        ax.plot(epochs, h[key], label=mname)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    loc = "upper right" if "loss" in key else "lower right"
    ax.legend(loc=loc, fontsize=13)
    fig.tight_layout()
    png_out = os.path.join(DATASET_RESULT_DIR, fname_stem + ".png")
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"曲线图已保存: {png_out}")

save_curve(best_run_histories, "train_loss", "Training Loss",
           f"train_loss_supp_{dataset_choosen}")
save_curve(best_run_histories, "test_loss",  "Test Loss",
           f"test_loss_supp_{dataset_choosen}")
save_curve(best_run_histories, "train_acc",  "Training Accuracy (%)",
           f"train_acc_supp_{dataset_choosen}")
save_curve(best_run_histories, "test_acc",   "Test Accuracy (%)",
           f"test_acc_supp_{dataset_choosen}")


# ===========================================================================
# 9. 混淆矩阵 —— 保存为 PDF
# ===========================================================================
def save_confusion_matrix_pdf(cm, model_name, num_classes, save_path):
    """绘制混淆矩阵并保存为 PDF 文件"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # 灰阶色图
    ax.imshow(cm, cmap="Greys", interpolation="nearest")

    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label",      fontsize=18)
    ax.tick_params(axis="both", labelsize=16)

    ticks = np.arange(num_classes)
    ax.set_xticks(ticks);  ax.set_xticklabels(ticks)
    ax.set_yticks(ticks);  ax.set_yticklabels(ticks)

    # 单元格数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=16, color="red", fontweight="bold")

    ax.set_title(model_name, fontsize=16, pad=10)
    fig.tight_layout()

    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"混淆矩阵 PDF 已保存: {save_path}")


for model_name, (t_arr, p_arr) in best_run_preds.items():
    cm = confusion_matrix(t_arr, p_arr)
    safe_name = model_name.replace(" ", "_").replace("-", "_")
    pdf_path  = os.path.join(
        DATASET_RESULT_DIR,
        f"confusion_matrix_{safe_name}_{dataset_choosen}.pdf"
    )
    save_confusion_matrix_pdf(cm, model_name, num_cata, pdf_path)

print("\n全部完成，结果已保存到:", DATASET_RESULT_DIR)

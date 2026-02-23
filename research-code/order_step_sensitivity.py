import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from utils.bearing_dataloader import BearingDataset
from utils.set_seed import set_seed
from model.AWC_TSARs_Net import GADFMultiChannelNet

# ==============================================================
# 全局配置
# ==============================================================
DATASET_CHOSEN = "hit" 
EPOCHS = 100
BATCH_SIZE = 64
N_RUNS = 5         
BASE_SEED = 42
WEIGHT_DECAY = 1e-4
LR = 0.0005
NOISE_DB = 10            # 测试时注入的高斯白噪声

RESULT_ROOT = "Result_OrderStep"
os.makedirs(RESULT_ROOT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


ORDER_CONFIGS = {
    "step=1.0\n(N=2)":  {"step": 1.0,  "indices": [0, 10]},
    "step=0.5\n(N=3)":  {"step": 0.5,  "indices": [0, 5, 10]},
    "step=0.2\n(N=6)":  {"step": 0.2,  "indices": [0, 2, 4, 6, 8, 10]},
    "step=0.1\n(N=11)": {"step": 0.1,  "indices": list(range(11))},
    "step=0.05\n(N=21)":{"step": 0.05, "indices": None}, 
}

# ==============================================================
# 1. 数据加载
# ==============================================================
if DATASET_CHOSEN == "hit":
    X_full = np.load("data/hit/HIT_gadf_images.npy")   # (N, 11, 128, 128)
    y_full = np.load("data/hit/HIT_gadf_labels.npy")
    num_classes = len(np.unique(y_full))
elif DATASET_CHOSEN == "hust":
    X_full = np.load("data/hust/HUST_gadf_images.npy")
    y_full = np.load("data/hust/HUST_gadf_labels.npy")
    num_classes = len(np.unique(y_full))
else:
    raise ValueError(f"Unknown dataset: {DATASET_CHOSEN}")

print(f"完整GADF数据形状 (step=0.1): {X_full.shape}, 类别数: {num_classes}")

# -----------------------------------------------------------------------
# 加载 / 自动生成 step=0.05 所需的 21 通道 GADF 数据
# -----------------------------------------------------------------------
def _get_step005_paths(dataset: str):
    if dataset == "hit":
        return ("data/hit/HIT_gadf_images_step05.npy",
                "data/hit/HIT_gadf_labels_step05.npy")
    return ("data/hust/HUST_gadf_images_step05.npy",
            "data/hust/HUST_gadf_labels_step05.npy")

_img05_path, _lbl05_path = _get_step005_paths(DATASET_CHOSEN)

if not (os.path.exists(_img05_path) and os.path.exists(_lbl05_path)):

    import importlib.util as _ilu
    _utils_dir = os.path.join(os.getcwd(), "utils")

    if _utils_dir not in sys.path:
        sys.path.insert(0, _utils_dir)
    _maker_path = os.path.join(_utils_dir, "data_maker_for_frft_gadf.py")
    _spec = _ilu.spec_from_file_location("data_maker_for_frft_gadf", _maker_path)
    _maker_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_maker_mod)
    _ds_name = "HIT" if DATASET_CHOSEN == "hit" else "HUST"

    _maker_mod.data_posseed(name=_ds_name, samples_num=300, a_step=0.05)
    print("[INFO] step=0.05 数据生成完成。")

X_full_05 = np.load(_img05_path)   # (N, 21, 128, 128)
y_full_05 = np.load(_lbl05_path)
print(f"完整GADF数据形状 (step=0.05): {X_full_05.shape}")



# 统一划分训练/测试索引
indices_all = np.arange(len(y_full))
idx_train, idx_test, _, _ = train_test_split(
    indices_all, y_full,
    test_size=0.3,
    random_state=BASE_SEED,
    stratify=y_full,
)

# ==============================================================
# 噪声注入工具函数（仅对测试集注入噪声）
# ==============================================================
def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(*x.shape) * np.sqrt(noise_power)
    return x + noise


# ==============================================================
# 训练 / 测试工具函数
# ==============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return correct / total


@torch.no_grad()
def measure_inference_time(model, loader) -> float:
    model.eval()
    for xb, _ in loader:
        _ = model(xb.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = 0
    for xb, _ in loader:
        _ = model(xb.to(device))
        total += xb.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / total * 1000   # ms / sample


# ==============================================================
# 主实验循环
# ==============================================================
results = {}  
params_dict = {}    
train_time_dict = {}  
infer_time_dict = {} 

for config_name, cfg in ORDER_CONFIGS.items():
    channel_indices = cfg["indices"]

    n_orders = X_full_05.shape[1] if channel_indices is None else len(channel_indices)
    print(f"\n{'='*60}")
    print(f"配置: {config_name.replace(chr(10), ' ')}  "
          f"(N={n_orders}, 通道: {'全部21阶' if channel_indices is None else channel_indices})")
    print(f"{'='*60}")

    if channel_indices is None:
        X_sub = X_full_05
        y_src = y_full_05
    else:
        X_sub = X_full[:, channel_indices, :, :]
        y_src = y_full

    X_train_raw = X_sub[idx_train].astype(np.float32)
    X_test_raw  = X_sub[idx_test].astype(np.float32)
    y_train = y_src[idx_train]
    y_test  = y_src[idx_test]

    _tmp = GADFMultiChannelNet(num_classes=num_classes, input_channels=n_orders).to(device)
    params_dict[config_name] = sum(p.numel() for p in _tmp.parameters() if p.requires_grad)
    del _tmp
    print(f"  参数量: {params_dict[config_name]/1e6:.3f} M")

    run_accs        = []
    run_train_times = []   # 每次运行的 avg epoch time (s)
    run_infer_times = []   # 每次运行的 avg inference time (ms/sample)

    for run in range(N_RUNS):
        seed = BASE_SEED + run
        set_seed(seed)
        np.random.seed(seed)

        X_test_noisy = add_awgn(X_test_raw.copy(), NOISE_DB)

        train_ds = BearingDataset(X_train_raw, y_train)
        test_ds  = BearingDataset(X_test_noisy.astype(np.float32), y_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        model = GADFMultiChannelNet(
            num_classes=num_classes,
            input_channels=n_orders
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        best_acc    = 0.0
        epoch_times = []

        for epoch in range(1, EPOCHS + 1):
            t_ep = time.perf_counter()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion
            )
            epoch_times.append(time.perf_counter() - t_ep)
            scheduler.step()
            if epoch % 20 == 0 or epoch == EPOCHS:
                test_acc = evaluate(model, test_loader)
                if test_acc > best_acc:
                    best_acc = test_acc
                print(f"  Run {run+1}/{N_RUNS} | Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc*100:.2f}% | "
                      f"Test Acc: {test_acc*100:.2f}%")

        infer_ms = measure_inference_time(model, test_loader)

        run_accs.append(best_acc * 100)
        run_train_times.append(float(np.mean(epoch_times)))
        run_infer_times.append(infer_ms)
        print(f"  ✓ Run {run+1} | Best Acc: {best_acc*100:.2f}% | "
              f"Train Time/Epoch: {np.mean(epoch_times):.2f}s | "
              f"Infer: {infer_ms:.3f} ms/sample")

    results[config_name]         = run_accs
    train_time_dict[config_name] = float(np.mean(run_train_times))
    infer_time_dict[config_name] = float(np.mean(run_infer_times))

    mean_acc = np.mean(run_accs)
    std_acc  = np.std(run_accs)
    print(f"\n  → {config_name.replace(chr(10), ' ')} "
          f"Acc: {mean_acc:.2f}% ± {std_acc:.2f}% | "
          f"Params: {params_dict[config_name]/1e6:.3f}M | "
          f"Train/Epoch: {train_time_dict[config_name]:.2f}s | "
          f"Infer: {infer_time_dict[config_name]:.3f}ms/sample")

def _n_orders(cfg):
    return X_full_05.shape[1] if cfg["indices"] is None else len(cfg["indices"])

summary_rows = []
for config_name, accs in results.items():
    label = config_name.replace("\n", " ")
    summary_rows.append({
        "Configuration":           label,
        "Step":                    ORDER_CONFIGS[config_name]["step"],
        "N_Orders":                _n_orders(ORDER_CONFIGS[config_name]),
        "Params (M)":              round(params_dict[config_name] / 1e6, 3),
        "Train_Time/Epoch (s)":    round(train_time_dict[config_name], 3),
        "Infer_Time/Sample (ms)":  round(infer_time_dict[config_name], 4),
        "Mean_Acc (%)":            round(np.mean(accs), 2),
        "Std_Acc (%)":             round(np.std(accs),  2),
    })

df_summary = pd.DataFrame(summary_rows)
csv_path = os.path.join(RESULT_ROOT, f"order_step_sensitivity_{DATASET_CHOSEN}.csv")
df_summary.to_csv(csv_path, index=False)
print(df_summary.to_string(index=False))

config_names = list(ORDER_CONFIGS.keys())
x_labels = [f"N={_n_orders(ORDER_CONFIGS[c])}\n(step={ORDER_CONFIGS[c]['step']})"
            for c in config_names]
x_pos   = np.arange(len(x_labels))
means   = [np.mean(results[c])     for c in config_names]
stds    = [np.std(results[c])      for c in config_names]
infers  = [infer_time_dict[c]      for c in config_names]
params  = [params_dict[c] / 1e6   for c in config_names]   # 单位 M

fig, ax1 = plt.subplots(figsize=(10, 5.5))


colors = ["#4C96D7", "#4C96D7", "#4C96D7", "#E05A5A", "#4C96D7"]
bars = ax1.bar(x_pos, means, yerr=stds,
               capsize=5, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.8,
               error_kw=dict(elinewidth=1.2, capthick=1.2),
               label="Test Accuracy (%, left axis)")

y_lo = max(0,   min(means) - max(stds) - 6)
y_hi = min(100, max(means) + max(stds) + 9)
ax1.set_ylim([y_lo, y_hi])

for bar, mean, std, p in zip(bars, means, stds, params):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + std + 0.3,
             f"{mean:.2f}%±{std:.2f}\n({p:.2f}M params)",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax1.axvline(x=3, color="#E05A5A", linestyle="--", linewidth=1.5, alpha=0.7)
ax1.text(3.07, y_lo + (y_hi - y_lo) * 0.04,
         "Proposed\n(step=0.1)", color="#E05A5A", fontsize=9)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels, fontsize=10)
ax1.set_xlabel("Number of FrFT Orders  (Step Size)", fontsize=11)
ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax1.grid(axis="y", linestyle="--", alpha=0.35)

ax2 = ax1.twinx()
ax2.plot(x_pos, infers, color="#2CA02C", marker="o",
         linewidth=2, markersize=8, linestyle="-",
         label="Infer Time (ms/sample, right axis)")
for xi, yi in zip(x_pos, infers):
    ax2.annotate(f"{yi:.3f}ms",
                 xy=(xi, yi), xytext=(0, 10),
                 textcoords="offset points",
                 ha="center", color="#2CA02C", fontsize=8.5)
ax2.set_ylabel("Inference Time  (ms / sample)", fontsize=11, color="#2CA02C")
ax2.tick_params(axis="y", labelcolor="#2CA02C")
ax2.set_ylim([0, max(infers) * 1.6])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="lower right", fontsize=9, framealpha=0.85)

ax1.set_title(
    f"FrFT Order Step Sensitivity: Accuracy vs. Computational Cost\n"
    f"(Dataset: {DATASET_CHOSEN.upper()},  SNR: {NOISE_DB} dB,  "
    f"Runs: {N_RUNS};  bar top: acc ± std  /  #params)",
    fontsize=10
)

plt.tight_layout()
fig_path = os.path.join(RESULT_ROOT, f"order_step_sensitivity_{DATASET_CHOSEN}.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"图像已保存至: {fig_path}")
plt.show()

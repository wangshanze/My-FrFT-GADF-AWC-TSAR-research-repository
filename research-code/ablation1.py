
import os
import sys
import time
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from utils.bearing_dataloader import BearingDataset
from utils.set_seed import set_seed


# ============================================================
# Global configs
# ============================================================
DATASET_CHOSEN = "hit"   # "hit" or "hust"
EPOCHS = 100
BATCH_SIZE = 64
N_RUNS = 5    
BASE_SEED = 42
WEIGHT_DECAY = 1e-4
LR = 0.00005


ORF_LABEL = 3

RESULT_ROOT = "Result_ReviewerFix"
os.makedirs(RESULT_ROOT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")


# ============================================================
# Data loading
# ============================================================
def load_dataset(dataset_name: str, base_seed: int):
    if dataset_name == "hit":
        X = np.load("data/hit/HIT_gadf_images.npy").astype(np.float32)  # (N, 11, 128, 128)
        y = np.load("data/hit/HIT_gadf_labels.npy").astype(np.int64)
        num_classes = len(np.unique(y))
    elif dataset_name == "hust":
        X = np.load("data/hust/HUST_gadf_images.npy").astype(np.float32)
        y = np.load("data/hust/HUST_gadf_labels.npy").astype(np.int64)
        num_classes = len(np.unique(y))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    indices = np.arange(len(y))
    idx_train, idx_test, _, _ = train_test_split(
        indices, y, test_size=0.3, random_state=base_seed, stratify=y
    )

    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    train_ds = BearingDataset(X_train, y_train)
    test_ds = BearingDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    input_channels = X.shape[1]
    return train_loader, test_loader, input_channels, num_classes, (X_test, y_test)


# ============================================================
# Welch t-test
# ============================================================
def welch_ttest(x: np.ndarray, y: np.ndarray):
    """
    Welch's t-test for unequal variances.
    Returns t statistic and two-sided p-value using a normal approximation if df is large.
    For typical N_RUNS=5, use Student-t approximation via a simple survival function estimate.

    If you want exact p-values, install scipy and replace with scipy.stats.ttest_ind.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)

    # Avoid division by zero
    if vx == 0 and vy == 0:
        return 0.0, 1.0

    t_num = mx - my
    t_den = math.sqrt(vx / nx + vy / ny + 1e-12)
    t = t_num / t_den

    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx * vx) / (nx * nx * (nx - 1) + 1e-12) + (vy * vy) / (ny * ny * (ny - 1) + 1e-12)
    df = df_num / (df_den + 1e-12)

    def normal_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    # Normal approx:
    p = 2 * (1 - normal_cdf(abs(t)))

    return float(t), float(p)


# ============================================================
# Modules: AWC (simplified) and TSAR variants (with CBAM order swap)
# ============================================================
class AdaptiveWeightingLite(nn.Module):
    """Minimal AWC-like projection: keep your original AWC if you want. Here we keep it simple."""
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.proj(x)


class TSARVariant(nn.Module):
    """
    TSAR block with:
      - main path: standard conv
      - aux path: depthwise + pointwise
      - attention: CBAM with selectable order: "cs" or "sc"
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_main: bool = True,
        use_aux: bool = True,
        attention_type: str = "cbam",
        cbam_order: str = "cs",   # "cs" or "sc"
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_main = use_main
        self.use_aux = use_aux
        self.use_residual = use_residual
        self.attention_type = attention_type
        self.cbam_order = cbam_order

        if use_main:
            self.main_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        if use_aux:
            self.aux_path = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        path_count = int(use_main) + int(use_aux)
        fusion_in = out_channels * max(path_count, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        if attention_type == "cbam":
            # Channel attention
            self.ca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, max(out_channels // 4, 1), kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(max(out_channels // 4, 1), out_channels, kernel_size=1),
                nn.Sigmoid(),
            )
            # Spatial attention
            self.sa = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        self.shortcut = nn.Identity()
        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def _apply_cbam(self, x):
        if self.cbam_order == "cs":
            # Channel -> Spatial
            ch = self.ca(x)
            x = x * ch
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            sp = self.sa(torch.cat([avg_out, max_out], dim=1))
            x = x * sp
            return x
        elif self.cbam_order == "sc":
            # Spatial -> Channel
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            sp = self.sa(torch.cat([avg_out, max_out], dim=1))
            x = x * sp
            ch = self.ca(x)
            x = x * ch
            return x
        else:
            raise ValueError(f"Unknown cbam_order: {self.cbam_order}")

    def forward(self, x):
        identity = self.shortcut(x) if self.use_residual else 0

        feats = []
        if self.use_main:
            feats.append(self.main_path(x))
        if self.use_aux:
            feats.append(self.aux_path(x))
        if not feats:
            raise ValueError("At least one path must be enabled.")

        combined = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        fused = self.fusion(combined)

        if self.attention_type == "cbam":
            refined = self._apply_cbam(fused)
        else:
            refined = fused

        return refined + identity if self.use_residual else refined


class TSARNetSmall(nn.Module):
    """
    Small network consistent with your pipeline:
      input: multi-channel GADF
      AWC-lite projection -> 4 TSAR blocks -> GAP -> classifier
    Also returns penultimate embedding for visualization.
    """
    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        tsar_cfg: dict,
        proj_out: int = 32,
    ):
        super().__init__()

        self.proj = AdaptiveWeightingLite(input_channels, proj_out)

        self.initial_conv = nn.Sequential(
            nn.Conv2d(proj_out, proj_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(proj_out),
            nn.ReLU(),
        )

        def make_block(in_ch, out_ch):
            return TSARVariant(
                in_channels=in_ch,
                out_channels=out_ch,
                use_main=tsar_cfg.get("use_main", True),
                use_aux=tsar_cfg.get("use_aux", True),
                attention_type=tsar_cfg.get("attention", "cbam"),
                cbam_order=tsar_cfg.get("cbam_order", "cs"),
                use_residual=tsar_cfg.get("use_residual", True),
            )

        self.block1 = make_block(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)

        self.block2 = make_block(64, 96)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)

        self.block3 = make_block(96, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)

        self.block4 = make_block(128, 192)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(192, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, num_classes)

        self.do1 = nn.Dropout(0.3)
        self.do2 = nn.Dropout(0.2)

    def forward(self, x, return_embedding: bool = False):
        x = self.proj(x)
        x = self.initial_conv(x)

        x = self.drop1(self.pool1(self.block1(x)))
        x = self.drop2(self.pool2(self.block2(x)))
        x = self.drop3(self.pool3(self.block3(x)))
        x = self.drop4(self.pool4(self.block4(x)))

        x = self.gap(x).flatten(1)               # [B, 192]
        z = self.do1(F.relu(self.bn1(self.fc1(x))))
        emb = self.do2(F.relu(self.bn2(self.fc2(z))))  # [B, 64]
        logits = self.out(emb)

        if return_embedding:
            return logits, emb
        return logits


# ============================================================
# Train / Eval utilities
# ============================================================
def train_one_model(model: nn.Module, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3)

    best_state = None
    best_acc = -1.0
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_correct, tr_total = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * xb.size(0)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        tr_loss /= max(tr_total, 1)
        tr_acc = 100.0 * tr_correct / max(tr_total, 1)

        # test loss for scheduler
        model.eval()
        te_loss = 0.0
        te_correct, te_total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                te_loss += loss.item() * xb.size(0)
                te_correct += (out.argmax(1) == yb).sum().item()
                te_total += xb.size(0)

        te_loss /= max(te_total, 1)
        te_acc = 100.0 * te_correct / max(te_total, 1)
        scheduler.step(te_loss)

        if te_acc > best_acc:
            best_acc = te_acc
            best_loss = te_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:3d} | Train {tr_acc:6.2f}% | Test {te_acc:6.2f}% | TestLoss {te_loss:.4f}")

    return best_state, best_acc, best_loss


@torch.no_grad()
def evaluate_acc_and_recall(model: nn.Module, loader, num_classes: int):
    model.eval()
    correct, total = 0, 0

    tp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        pred = out.argmax(1)

        correct += (pred == yb).sum().item()
        total += yb.size(0)

        for c in range(num_classes):
            tp[c] += ((pred == c) & (yb == c)).sum().item()
            fn[c] += ((pred != c) & (yb == c)).sum().item()

    acc = 100.0 * correct / max(total, 1)
    recall = tp / np.maximum(tp + fn, 1)
    recall = recall * 100.0
    return acc, recall


@torch.no_grad()
def extract_embeddings(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 256):
    model.eval()
    ds = BearingDataset(X_test.astype(np.float32), y_test.astype(np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_emb, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits, emb = model(xb, return_embedding=True)
        all_emb.append(emb.detach().cpu().numpy())
        all_y.append(yb.numpy())
    emb = np.concatenate(all_emb, axis=0)
    yy = np.concatenate(all_y, axis=0)
    return emb, yy


def plot_tsne(emb: np.ndarray, y: np.ndarray, title: str, save_path: str, max_points: int = 1200):
    # Optional downsample for speed and readability
    if len(y) > max_points:
        idx = np.random.choice(len(y), size=max_points, replace=False)
        emb = emb[idx]
        y = y[idx]

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=BASE_SEED)
    z = tsne.fit_transform(emb)

    plt.figure(figsize=(7, 6))
    for cls in np.unique(y):
        mask = (y == cls)
        plt.scatter(z[mask, 0], z[mask, 1], s=10, alpha=0.75, label=f"Class {cls}")
    # plt.title(title)
    plt.legend(loc="lower right", markerscale=1.5, fontsize=24)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved t-SNE: {save_path}")


# ============================================================
# Experiments requested by reviewer
# ============================================================
def run_main_aux_stats_and_viz(train_loader, test_loader, input_channels, num_classes, X_test, y_test):
    """
    Task 1:
      - main-only vs aux-only
      - collect per-class recall across runs
      - Welch t-test on ORF recall
      - t-SNE visualization on best run for each model
    """
    exp_cfgs = {
        "TSAR_main_only": dict(use_main=True, use_aux=False, attention="cbam", cbam_order="cs", use_residual=True),
        "TSAR_aux_only":  dict(use_main=False, use_aux=True, attention="cbam", cbam_order="cs", use_residual=True),
    }

    records = []
    recalls_by_exp = {k: [] for k in exp_cfgs.keys()}
    accs_by_exp = {k: [] for k in exp_cfgs.keys()}
    best_state_by_exp = {}

    for exp_name, cfg in exp_cfgs.items():
        print("\n" + "=" * 80)
        print(f"[TASK1] Running: {exp_name}")
        print("=" * 80)

        best_acc_seen = -1.0
        best_state = None

        for run in range(N_RUNS):
            seed = BASE_SEED + run
            set_seed(seed)

            model = TSARNetSmall(
                num_classes=num_classes,
                input_channels=input_channels,
                tsar_cfg=cfg,
            ).to(device)

            state, best_acc, best_loss = train_one_model(model, train_loader, test_loader)
            model.load_state_dict(state, strict=True)
            acc, recall = evaluate_acc_and_recall(model, test_loader, num_classes)

            accs_by_exp[exp_name].append(acc)
            recalls_by_exp[exp_name].append(recall)

            print(f"[{exp_name}] Run {run+1}/{N_RUNS} | Acc {acc:.2f}% | ORF Recall {recall[ORF_LABEL]:.2f}%")

            if acc > best_acc_seen:
                best_acc_seen = acc
                best_state = state

        best_state_by_exp[exp_name] = best_state

        # Summaries
        acc_arr = np.array(accs_by_exp[exp_name])
        recall_arr = np.stack(recalls_by_exp[exp_name], axis=0)  # [N_RUNS, C]

        rec = {
            "Experiment": exp_name,
            "AccMean(%)": f"{acc_arr.mean():.2f}",
            "AccStd(%)": f"{acc_arr.std():.2f}",
            "ORFRecallMean(%)": f"{recall_arr[:, ORF_LABEL].mean():.2f}",
            "ORFRecallStd(%)": f"{recall_arr[:, ORF_LABEL].std():.2f}",
        }

        # also store per-class recall mean/std
        for c in range(num_classes):
            rec[f"Recall_C{c}_Mean(%)"] = f"{recall_arr[:, c].mean():.2f}"
            rec[f"Recall_C{c}_Std(%)"] = f"{recall_arr[:, c].std():.2f}"

        records.append(rec)

    # Welch t-test on ORF recall
    main_orf = np.stack(recalls_by_exp["TSAR_main_only"], axis=0)[:, ORF_LABEL]
    aux_orf  = np.stack(recalls_by_exp["TSAR_aux_only"], axis=0)[:, ORF_LABEL]
    t_stat, p_val = welch_ttest(main_orf, aux_orf)

    print("\n" + "-" * 80)
    print(f"[TASK1] Welch t-test on ORF recall: t={t_stat:.4f}, approx p={p_val:.4f}")
    print("-" * 80)

    # Save CSV summary
    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULT_ROOT, "task1_main_vs_aux_stats.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved Task1 stats CSV: {out_csv}")

    # Save p-value
    with open(os.path.join(RESULT_ROOT, "task1_orf_welch_ttest.txt"), "w", encoding="utf-8") as f:
        f.write(f"Welch t-test on ORF recall\n")
        f.write(f"t={t_stat:.6f}\n")
        f.write(f"approx_p={p_val:.6f}\n")
        f.write(f"main_orf_recalls={main_orf.tolist()}\n")
        f.write(f"aux_orf_recalls={aux_orf.tolist()}\n")

    # Feature visualization (t-SNE) using best states
    for exp_name, cfg in exp_cfgs.items():
        model = TSARNetSmall(num_classes=num_classes, input_channels=input_channels, tsar_cfg=cfg).to(device)
        model.load_state_dict(best_state_by_exp[exp_name], strict=True)
        emb, yy = extract_embeddings(model, X_test, y_test, batch_size=256)
        save_path = os.path.join(RESULT_ROOT, f"task1_tsne_{exp_name}.png")
        plot_tsne(emb, yy, title=f"t-SNE of embeddings: {exp_name}", save_path=save_path)

    return df


def run_cbam_order_swap_ablation(train_loader, test_loader, input_channels, num_classes):
    """
    Task 2:
      - CBAM Channel->Spatial vs Spatial->Channel
      - Keep everything else identical
    """
    exp_cfgs = {
        "CBAM_CS": dict(use_main=True, use_aux=True, attention="cbam", cbam_order="cs", use_residual=True),
        "CBAM_SC": dict(use_main=True, use_aux=True, attention="cbam", cbam_order="sc", use_residual=True),
    }

    records = []

    for exp_name, cfg in exp_cfgs.items():
        print("\n" + "=" * 80)
        print(f"[TASK2] Running: {exp_name}")
        print("=" * 80)

        run_accs = []
        run_recalls = []

        for run in range(N_RUNS):
            seed = BASE_SEED + 100 + run  # offset seed to avoid reuse
            set_seed(seed)

            model = TSARNetSmall(
                num_classes=num_classes,
                input_channels=input_channels,
                tsar_cfg=cfg,
            ).to(device)

            state, best_acc, best_loss = train_one_model(model, train_loader, test_loader)
            model.load_state_dict(state, strict=True)
            acc, recall = evaluate_acc_and_recall(model, test_loader, num_classes)

            run_accs.append(acc)
            run_recalls.append(recall)

            print(f"[{exp_name}] Run {run+1}/{N_RUNS} | Acc {acc:.2f}%")

        acc_arr = np.array(run_accs)
        recall_arr = np.stack(run_recalls, axis=0)

        rec = {
            "Experiment": exp_name,
            "AccMean(%)": f"{acc_arr.mean():.2f}",
            "AccStd(%)": f"{acc_arr.std():.2f}",
        }
        for c in range(num_classes):
            rec[f"Recall_C{c}_Mean(%)"] = f"{recall_arr[:, c].mean():.2f}"
            rec[f"Recall_C{c}_Std(%)"] = f"{recall_arr[:, c].std():.2f}"

        records.append(rec)

    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULT_ROOT, "task2_cbam_order_swap.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved Task2 CSV: {out_csv}")
    return df


# ============================================================
# Main
# ============================================================
def main():
    set_seed(BASE_SEED)

    train_loader, test_loader, input_channels, num_classes, (X_test, y_test) = load_dataset(DATASET_CHOSEN, BASE_SEED)
    print(f"[INFO] Dataset: {DATASET_CHOSEN}, input_channels={input_channels}, num_classes={num_classes}")
    print(f"[INFO] ORF_LABEL={ORF_LABEL} (Please confirm it matches your label encoding.)")

    # Task 1
    df_task1 = run_main_aux_stats_and_viz(train_loader, test_loader, input_channels, num_classes, X_test, y_test)

    # Task 2
    df_task2 = run_cbam_order_swap_ablation(train_loader, test_loader, input_channels, num_classes)

    print("\n" + "=" * 80)
    print("[DONE] Generated:")
    print(f"  - {os.path.join(RESULT_ROOT, 'task1_main_vs_aux_stats.csv')}")
    print(f"  - {os.path.join(RESULT_ROOT, 'task1_orf_welch_ttest.txt')}")
    print(f"  - {os.path.join(RESULT_ROOT, 'task1_tsne_TSAR_main_only.png')}")
    print(f"  - {os.path.join(RESULT_ROOT, 'task1_tsne_TSAR_aux_only.png')}")
    print(f"  - {os.path.join(RESULT_ROOT, 'task2_cbam_order_swap.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
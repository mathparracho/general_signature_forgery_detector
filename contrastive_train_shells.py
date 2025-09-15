import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from architectures.custom_siamese_shells import ContrastiveNetworkCascas

# =====
SPLIT_FILES_DIR   = os.getenv("SPLIT_FILES_DIR", "./data/split_files")
ICDAR_SPLIT_FILE  = os.getenv("ICDAR_SPLIT_FILE", "icdar_writerdisjoint_split_balanced_label1balanced.csv")
CEDAR_SPLIT_FILE  = os.getenv("CEDAR_SPLIT_FILE", "best_style_writerdisjoint_split_balanced.csv")
GPDS_SPLIT_FILE   = os.getenv("GPDS_SPLIT_FILE",  "gpds_writerdisjoint_split_balanced_clipped_fixed_2.csv")

ROOT_FOLDER_ICDAR = os.getenv("ROOT_FOLDER_ICDAR", "./data/cascas_icdar_signatures/")
ROOT_FOLDER_CEDAR = os.getenv("ROOT_FOLDER_CEDAR", "./data/cascas_13_cedar_signatures/")
ROOT_FOLDER_GPDS  = os.getenv("ROOT_FOLDER_GPDS",  "./data/cascas_gpds_signatures/")

USE_DATASET   = os.getenv("USE_DATASET", "CEDAR")  # CEDAR | ICDAR | GPDS | MERGED
MODEL_DIR     = os.getenv("MODEL_DIR", "./models")
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "64"))
NUM_WORKERS   = int(os.getenv("NUM_WORKERS", "4"))
EPOCHS        = int(os.getenv("EPOCHS", "100"))
LR            = float(os.getenv("LR", "1e-4"))
WEIGHT_DECAY  = float(os.getenv("WEIGHT_DECAY", "1e-2"))
MARGIN        = float(os.getenv("MARGIN", "1.0"))
SEED          = int(os.getenv("SEED", "0"))

# ====
writer = SummaryWriter()
timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(MODEL_DIR, f"model_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ====
def _load_df(path, tag):
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        df.insert(0, "dataset", tag)
    return df

cedar_path = os.path.join(SPLIT_FILES_DIR, CEDAR_SPLIT_FILE)
icdar_path = os.path.join(SPLIT_FILES_DIR, ICDAR_SPLIT_FILE)
gpds_path  = os.path.join(SPLIT_FILES_DIR, GPDS_SPLIT_FILE)

dfs = []
if USE_DATASET in ("CEDAR", "MERGED"):
    df_cedar = _load_df(cedar_path, "CEDAR")
    df_cedar["split"] = df_cedar["split"].replace("valid", "val")
    dfs.append(df_cedar)

if USE_DATASET in ("ICDAR", "MERGED"):
    df_icdar = _load_df(icdar_path, "ICDAR")
    dfs.append(df_icdar)

if USE_DATASET in ("GPDS", "MERGED"):
    df_gpds = _load_df(gpds_path, "GPDS")
    dfs.append(df_gpds)

df_all = pd.concat(dfs, ignore_index=True) if USE_DATASET == "MERGED" else dfs[0]

df_train = df_all[df_all["split"] == "train"].sample(frac=1, random_state=SEED).reset_index(drop=True)
df_val   = df_all[df_all["split"] == "val"].reset_index(drop=True)
df_test  = df_all[df_all["split"] == "test"].reset_index(drop=True)

# ====
def jitter(x, sigma=0.03):
    noise = sigma * torch.randn_like(x)
    return torch.clamp(x + noise, 0.0, 1.0)

def cutout(x, width=20, num_cutouts=3):
    for _ in range(num_cutouts):
        start = torch.randint(0, x.size(1) - width, (1,))
        x[:, start:start + width] = 0
    return x

def add_gaussian_noise(x, std=0.02):
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)

# ======
class CascasPairsDataset(Dataset):
    def __init__(self, df, augment=False):
        df = df.copy()
        df["forged"] = df["forged"].astype(np.float32)
        self.s1 = df["image1"].tolist()
        self.s2 = df["image2"].tolist()
        self.labels = df["forged"].tolist()
        self.datasets = df["dataset"].tolist()
        self.augment = augment

    def _root(self, ds):
        if ds == "ICDAR":
            return ROOT_FOLDER_ICDAR
        if ds == "CEDAR":
            return ROOT_FOLDER_CEDAR
        return ROOT_FOLDER_GPDS

    def _load_pair(self, root, rel_path, index):
        base = os.path.join(root, rel_path.split("/")[0] + "_cascas", rel_path.split("/")[1].split(".")[0])
        casca_csv = os.path.join(base, f"casca{index}.csv")
        press_csv = os.path.join(base, f"matrix_pressure{index}.csv")

        m = pd.read_csv(casca_csv, header=None, dtype=np.float32).to_numpy(dtype="float32")
        p = pd.read_csv(press_csv, header=None, dtype=np.float32).to_numpy(dtype="float32")

        m = (m - m.mean()) / (m.std() + 1e-8)
        p = (p - p.mean()) / (p.std() + 1e-8)

        return torch.tensor(m), torch.tensor(p)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ds = self.datasets[idx]
        root = self._root(ds)

        i1 = random.randint(0, 9)
        i2 = random.randint(0, 9)

        m1, p1 = self._load_pair(root, self.s1[idx], i1)
        m2, p2 = self._load_pair(root, self.s2[idx], i2)

        if self.augment:
            if random.random() < 0.5: m1 = jitter(m1)
            if random.random() < 0.4: m1 = cutout(m1, num_cutouts=2)
            if random.random() < 0.5: m1 = add_gaussian_noise(m1)
            if random.random() < 0.5: m2 = jitter(m2)
            if random.random() < 0.4: m2 = cutout(m2, num_cutouts=2)
            if random.random() < 0.3: m2 = add_gaussian_noise(m2)

        m1 = torch.clamp(m1, 0.0, 1.0)
        m2 = torch.clamp(m2, 0.0, 1.0)

        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return m1, p1, m2, p2, y

# ======
train_ds = CascasPairsDataset(df_train, augment=False)
val_ds   = CascasPairsDataset(df_val,   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# =====
model = ContrastiveNetworkCascas()
for p in model.parameters():
    p.requires_grad = True
model.to(device)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, out1, out2, label):
        d = F.pairwise_distance(out1, out2)
        return torch.mean((1 - label) * d.pow(2) + label * torch.clamp(self.margin - d, min=0.0).pow(2))

criterion = ContrastiveLoss(margin=MARGIN)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# =====
best_val = float("inf")
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    train_sum = 0.0
    dists_same, dists_diff = [], []

    for m1, p1, m2, p2, y in tqdm(train_loader):
        m1, p1 = m1.to(device), p1.to(device)
        m2, p2 = m2.to(device), p2.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        z1, z2 = model(m1, p1, m2, p2)
        loss = criterion(z1, z2, y)
        loss.backward()
        optimizer.step()
        train_sum += loss.item()

        d_batch = F.pairwise_distance(z1, z2).detach().cpu().numpy()
        y_np = y.cpu().numpy()
        dists_same += d_batch[y_np == 0].tolist()
        dists_diff += d_batch[y_np == 1].tolist()

    plt.figure()
    sns.histplot(dists_same, color="blue", label="Genuine", kde=True, stat="density", bins=30)
    sns.histplot(dists_diff, color="red",  label="Forged",  kde=True, stat="density", bins=30)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("Training Distance Distribution")
    plt.legend()
    png_train = os.path.join(run_dir, f"distance_distribution_TRAIN_epoch{epoch}.png")
    plt.savefig(png_train, bbox_inches="tight", dpi=150)
    plt.close()

    train_loss = train_sum / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_sum = 0.0
    dists_same, dists_diff = [], []

    with torch.no_grad():
        for m1, p1, m2, p2, y in tqdm(val_loader):
            m1, p1 = m1.to(device), p1.to(device)
            m2, p2 = m2.to(device), p2.to(device)
            y = y.to(device)

            z1, z2 = model(m1, p1, m2, p2)
            loss = criterion(z1, z2, y)
            val_sum += loss.item()

            d_batch = F.pairwise_distance(z1, z2).cpu().numpy()
            y_np = y.cpu().numpy()
            dists_same += d_batch[y_np == 0].tolist()
            dists_diff += d_batch[y_np == 1].tolist()

    val_loss = val_sum / len(val_loader)
    val_losses.append(val_loss)

    plt.figure()
    sns.histplot(dists_same, color="blue", label="Genuine", kde=True, stat="density", bins=30)
    sns.histplot(dists_diff, color="red",  label="Forged",  kde=True, stat="density", bins=30)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("Validation Distance Distribution")
    plt.legend()
    png_val = os.path.join(run_dir, f"distance_distribution_VALID_epoch{epoch}.png")
    plt.savefig(png_val, bbox_inches="tight", dpi=150)
    plt.close()

    writer.add_scalars("Losses", {"Train": train_loss, "Validation": val_loss}, epoch)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {train_loss:.4f}")
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Val   Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        print("Saving best model...")

torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))
print("Finished Training")

plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolution of Training Loss")
plt.legend()
plt.savefig(os.path.join(run_dir, "loss.png"), bbox_inches="tight", dpi=150)
plt.close()

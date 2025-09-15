import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

from architectures.siamese_contrastive_resnet34 import ContrastiveNetwork

# =====
SPLIT_FILES_DIR = os.getenv("SPLIT_FILES_DIR", "./data/split_files")
ICDAR_SPLIT_FILE = os.getenv("ICDAR_SPLIT_FILE", "train_data_2ndAdjs.csv")
CEDAR_SPLIT_FILE = os.getenv("CEDAR_SPLIT_FILE", "best_style_writerdisjoint_split_balanced.csv")
GPDS_SPLIT_FILE = os.getenv("GPDS_SPLIT_FILE", "GPDS_full_split.csv")

ICDAR_ROOT = os.getenv("ICDAR_ROOT", "./data/icdar_signatures/train/")
CEDAR_ROOT = os.getenv("CEDAR_ROOT", "./data/cedar_signatures/")
GPDS_ROOT = os.getenv("GPDS_ROOT", "./data/firmasSINTESISmanuscritas/")

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LR = float(os.getenv("LR", "1e-4"))
MARGIN = float(os.getenv("MARGIN", "1.0"))

USE_DATASET = os.getenv("USE_DATASET", "CEDAR")  # CEDAR | ICDAR | GPDS | MERGED
SEED = int(os.getenv("SEED", "0"))

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
def _read_csv(path):
    return pd.read_csv(path)

cedar_path = os.path.join(SPLIT_FILES_DIR, CEDAR_SPLIT_FILE)
icdar_path = os.path.join(SPLIT_FILES_DIR, ICDAR_SPLIT_FILE)
gpds_path  = os.path.join(SPLIT_FILES_DIR, GPDS_SPLIT_FILE)

dfs = []
if USE_DATASET in ("CEDAR", "MERGED"):
    df_cedar = _read_csv(cedar_path)
    if "dataset" not in df_cedar.columns:
        df_cedar.insert(0, "dataset", "CEDAR")
    dfs.append(df_cedar)

if USE_DATASET in ("ICDAR", "MERGED"):
    df_icdar = _read_csv(icdar_path)
    if "dataset" not in df_icdar.columns:
        df_icdar.insert(0, "dataset", "ICDAR")
    dfs.append(df_icdar)

if USE_DATASET in ("GPDS", "MERGED"):
    df_gpds = _read_csv(gpds_path)
    if "dataset" not in df_gpds.columns:
        df_gpds.insert(0, "dataset", "GPDS")
    dfs.append(df_gpds)

if USE_DATASET == "MERGED":
    df = pd.concat(dfs, ignore_index=True)
else:
    df = dfs[0]

df_train = df[df["split"] == "train"].sample(frac=1, random_state=SEED).reset_index(drop=True)
df_valid = df[df["split"] == "valid"].sample(frac=1, random_state=SEED).reset_index(drop=True)
df_test  = df[df["split"] == "test"].sample(frac=1, random_state=SEED).reset_index(drop=True)

# =====
train_tf = A.Compose(
    [
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.05,
            shear=(-2, 2),
            scale=(0.7, 1.3),
            p=0.5,
            rotate_method="ellipse",
            keep_ratio=True,
            mode=1,
        ),
        A.Sharpen(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.01, contrast_limit=0.01),
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_tf = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ]
)

# =====
class SiamesePairsDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.signature1_paths = df["image1"].tolist()
        self.signature2_paths = df["image2"].tolist()
        self.labels = df["forged"].tolist()
        self.datasets = df["dataset"].tolist()
        self.transforms = transforms

    def _root(self, ds):
        if ds == "ICDAR":
            return ICDAR_ROOT
        if ds == "CEDAR":
            return CEDAR_ROOT
        return GPDS_ROOT

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ds = self.datasets[idx]
        root = self._root(ds)

        s1 = np.asarray(Image.open(os.path.join(root, self.signature1_paths[idx])).convert("L"))
        s2 = np.asarray(Image.open(os.path.join(root, self.signature2_paths[idx])).convert("L"))
        y = float(self.labels[idx])

        if self.transforms:
            s1 = self.transforms(image=s1)["image"]
            s2 = self.transforms(image=s2)["image"]

        return s1, s2, torch.tensor(y, dtype=torch.float32)

# ====
train_set = SiamesePairsDataset(df_train, transforms=train_tf)
val_set   = SiamesePairsDataset(df_valid, transforms=val_tf)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# =====
model = ContrastiveNetwork()
for p in model.parameters():
    p.requires_grad = True
model.to(device)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        d = torch.nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(d, 2) + label * torch.pow(torch.clamp(self.margin - d, min=0.0), 2))
        return loss

criterion = ContrastiveLoss(margin=MARGIN)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ========
best_val = float("inf")
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    train_sum = 0.0

    for s1, s2, y in tqdm(train_loader):
        s1 = s1.to(device)
        s2 = s2.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        z1, z2 = model(s1, s2)
        loss = criterion(z1, z2, y)
        loss.backward()
        optimizer.step()
        train_sum += loss.item()

    train_loss = train_sum / len(train_loader)
    train_losses.append(train_loss)
    writer.add_scalar("Loss/train", train_loss, epoch)

    model.eval()
    val_sum = 0.0
    dists_same, dists_diff = [], []

    with torch.no_grad():
        for s1, s2, y in tqdm(val_loader):
            s1 = s1.to(device)
            s2 = s2.to(device)
            y = y.to(device)

            z1, z2 = model(s1, s2)
            loss = criterion(z1, z2, y)
            val_sum += loss.item()

            d_batch = F.pairwise_distance(z1, z2).cpu().numpy()
            y_np = y.cpu().numpy()
            dists_same += d_batch[y_np == 0].tolist()
            dists_diff += d_batch[y_np == 1].tolist()

    val_loss = val_sum / len(val_loader)
    val_losses.append(val_loss)
    writer.add_scalar("Loss/val", val_loss, epoch)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}")
    print(f"Epoch [{epoch+1}/{EPOCHS}] Val   Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        print("Saving best model...")

    plt.figure()
    sns.histplot(dists_same, label="Genuine", kde=True, stat="density", bins=30)
    sns.histplot(dists_diff, label="Forged", kde=True, stat="density", bins=30)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("Validation Distance Distribution")
    plt.legend()
    out_png = os.path.join(run_dir, f"distance_distribution_VALID_epoch{epoch}.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()

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

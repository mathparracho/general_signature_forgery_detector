import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from architectures.resnet34_1D_shells import ContrastiveNetworkCascas

# =====
DATASET = os.getenv("DATASET", "GPDS")  # CEDAR | ICDAR | GPDS | MERGED

SPLIT_FILES_DIR = os.getenv("SPLIT_FILES_DIR", "./data/split_files")
CEDAR_SPLIT_FILE = os.getenv("CEDAR_SPLIT_FILE", "best_style_writerdisjoint_split_balanced.csv")
ICDAR_SPLIT_FILE = os.getenv("ICDAR_SPLIT_FILE", "icdar_writerdisjoint_split_balanced_label1balanced.csv")
GPDS_SPLIT_FILE  = os.getenv("GPDS_SPLIT_FILE",  "GPDS_full_test.csv")
MERGED_GPDS_SPLIT_FILE = os.getenv("MERGED_GPDS_SPLIT_FILE", "gpds_writerdisjoint_split_balanced_clipped_fixed_2.csv")
MERGED_CEDAR_SPLIT_FILE = os.getenv("MERGED_CEDAR_SPLIT_FILE", "best_style_writerdisjoint_split_balanced.csv")
MERGED_ICDAR_SPLIT_FILE = os.getenv("MERGED_ICDAR_SPLIT_FILE", "icdar_writerdisjoint_split_balanced_label1balanced.csv")

ROOT_FOLDER_CEDAR = os.getenv("ROOT_FOLDER_CEDAR", "./data/cascas_13_cedar_signatures/")
ROOT_FOLDER_ICDAR = os.getenv("ROOT_FOLDER_ICDAR", "./data/cascas_icdar_signatures/")
ROOT_FOLDER_GPDS  = os.getenv("ROOT_FOLDER_GPDS",  "./data/cascas_gpds_signatures/")

MODEL_DIR = os.getenv("MODEL_DIR", "./models/model_2025-08-04_16:12:16")
BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME", "best_model.pth")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", MODEL_DIR)

# Inference params
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
OPTIMAL_THRESHOLD = float(os.getenv("OPTIMAL_THRESHOLD", "0.5"))
IN_CHANNELS = int(os.getenv("IN_CHANNELS", "72"))
SEED = int(os.getenv("SEED", "0"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======
def _load_csv(path, tag=None, fix_valid_to_val=False):
    df = pd.read_csv(path)
    if tag is not None and "dataset" not in df.columns:
        df.insert(0, "dataset", tag)
    if fix_valid_to_val and "split" in df.columns:
        df["split"] = df["split"].replace("valid", "val")
    return df

if DATASET == "CEDAR":
    df = _load_csv(os.path.join(SPLIT_FILES_DIR, CEDAR_SPLIT_FILE), "CEDAR", fix_valid_to_val=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

elif DATASET == "ICDAR":
    df = _load_csv(os.path.join(SPLIT_FILES_DIR, ICDAR_SPLIT_FILE), "ICDAR")
    df_test = df[df["split"] == "test"].reset_index(drop=True)

elif DATASET == "GPDS":
    df = _load_csv(os.path.join(SPLIT_FILES_DIR, GPDS_SPLIT_FILE), "GPDS")
    df_test = df[df["split"] == "test"].reset_index(drop=True)

else:  # MERGED
    df_icdar = _load_csv(os.path.join(SPLIT_FILES_DIR, MERGED_ICDAR_SPLIT_FILE), "ICDAR")
    df_cedar = _load_csv(os.path.join(SPLIT_FILES_DIR, MERGED_CEDAR_SPLIT_FILE), "CEDAR", fix_valid_to_val=True)
    df_gpds  = _load_csv(os.path.join(SPLIT_FILES_DIR, MERGED_GPDS_SPLIT_FILE), "GPDS")
    df = pd.concat([df_icdar, df_cedar, df_gpds], ignore_index=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

# ======
class CascasTestDataset(Dataset):
    def __init__(self, df, dataset_name):
        self.s1 = df["image1"].tolist()
        self.s2 = df["image2"].tolist()
        self.labels = df["forged"].tolist()
        if "dataset" in df.columns:
            self.datasets = df["dataset"].tolist()
        else:
            self.datasets = [dataset_name] * len(self.labels)

    def _root(self, tag):
        if tag == "CEDAR":
            return ROOT_FOLDER_CEDAR
        if tag == "ICDAR":
            return ROOT_FOLDER_ICDAR
        return ROOT_FOLDER_GPDS

    def _load_pair(self, root, rel_path, idx):
        base = os.path.join(
            root,
            rel_path.split("/")[0] + "_cascas",
            rel_path.split("/")[1].split(".")[0],
        )
        casca_csv = os.path.join(base, f"casca{idx}.csv")
        press_csv = os.path.join(base, f"matrix_pressure{idx}.csv")

        m = pd.read_csv(casca_csv, header=None, dtype=np.float32).to_numpy(dtype="float32")
        p = pd.read_csv(press_csv, header=None, dtype=np.float32).to_numpy(dtype="float32")

        # z-score per matrix
        m = (m - m.mean()) / (m.std() + 1e-8)
        p = (p - p.mean()) / (p.std() + 1e-8)

        return torch.tensor(m), torch.tensor(p)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tag = self.datasets[idx]
        root = self._root(tag)

        r1 = random.randint(0, 9)
        r2 = random.randint(0, 9)

        m1, pr1 = self._load_pair(root, self.s1[idx], r1)
        m2, pr2 = self._load_pair(root, self.s2[idx], r2)

        return m1, pr1, m2, pr2, int(self.labels[idx])

# =======
test_ds = CascasTestDataset(df_test, DATASET)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# ======
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ContrastiveNetworkCascas(in_channels=IN_CHANNELS)
model_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======
def compute_distances(dataloader):
    distances, labels, d_same, d_diff = [], [], [], []
    with torch.no_grad():
        for m1, pr1, m2, pr2, y in tqdm(dataloader):
            m1, pr1 = m1.to(device), pr1.to(device)
            m2, pr2 = m2.to(device), pr2.to(device)

            z1, z2 = model(m1, pr1, m2, pr2)
            batch_dist = F.pairwise_distance(z1, z2).cpu().numpy()

            y_np = np.array(y)
            d_same += batch_dist[y_np == 0].tolist()
            d_diff += batch_dist[y_np == 1].tolist()

            distances.extend(batch_dist)
            labels.extend(y_np.tolist())
    return np.array(distances), np.array(labels).astype(int), d_same, d_diff

# ======
test_distances, test_labels, d_same, d_diff = compute_distances(test_loader)
y_pred_test = (test_distances >= OPTIMAL_THRESHOLD).astype(int)

acc = accuracy_score(test_labels, y_pred_test)
auc = roc_auc_score(test_labels, test_distances)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test ROC AUC: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, y_pred_test))

# Confusion Matrix
conf = confusion_matrix(test_labels, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
suffix = DATASET.lower()
plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{suffix}.png"), bbox_inches="tight", dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(test_labels, test_distances)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Test ROC Curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{suffix}.png"), bbox_inches="tight", dpi=150)
plt.close()

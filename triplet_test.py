import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from architectures.siamese_triplet import TripletNetwork
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATASET = os.getenv("DATASET", "MERGED")  # CEDAR | ICDAR | GPDS | MERGED
SPLIT_FILES_DIR = os.getenv("SPLIT_FILES_DIR", "./data/split_files")
ICDAR_ROOT = os.getenv("ICDAR_ROOT", "./data/icdar_signatures/train/")
CEDAR_ROOT = os.getenv("CEDAR_ROOT", "./data/cedar_signatures/")
GPDS_ROOT = os.getenv("GPDS_ROOT", "./data/firmasSINTESISmanuscritas/")
MODEL_DIR = os.getenv("MODEL_DIR", "./models/latest")
BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME", "best_model.pth")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", MODEL_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if DATASET == "CEDAR":
    df = pd.read_csv(f"{SPLIT_FILES_DIR}/full_data_split_CEDAR.csv")
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_valid = df[df["split"] == "valid"].reset_index(drop=True)

elif DATASET == "ICDAR":
    df = pd.read_csv(f"{SPLIT_FILES_DIR}/train_data_2ndAdjs.csv")
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_valid = df[df["split"] == "valid"].reset_index(drop=True)

elif DATASET == "GPDS":
    df = pd.read_csv(f"{SPLIT_FILES_DIR}/GPDS_full_split.csv")
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_valid = df[df["split"] == "valid"].reset_index(drop=True)

else:  # MERGED
    df_icdar = pd.read_csv(f"{SPLIT_FILES_DIR}/train_data_2ndAdjs.csv")
    df_icdar.insert(0, "dataset", "ICDAR")

    df_cedar = pd.read_csv(f"{SPLIT_FILES_DIR}/full_data_split_CEDAR.csv")
    df_cedar.insert(0, "dataset", "CEDAR")

    df_gpds = pd.read_csv(f"{SPLIT_FILES_DIR}/GPDS_full_split.csv")
    df_gpds.insert(0, "dataset", "GPDS")

    df = pd.concat([df_icdar, df_cedar, df_gpds])
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_valid = df[df["split"] == "valid"].reset_index(drop=True)


transforms_test_albm = A.Compose(
    [
        A.CLAHE(),
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ]
)

class Siamese_Test_Dataset(Dataset):
    def __init__(self, df, transforms=None, dataset_name="MERGED"):
        self.signature1_paths = df["image1"].tolist()
        self.signature2_paths = df["image2"].tolist()
        self.labels = df["forged"].tolist()
        self.dataset_name = dataset_name
        self.transforms = transforms

        if self.dataset_name == "MERGED":
            self.dataset_col = df["dataset"].tolist()
        else:
            self.dataset_col = [self.dataset_name] * len(self.labels)

    def _root_for(self, ds):
        if ds == "ICDAR":
            return ICDAR_ROOT
        if ds == "CEDAR":
            return CEDAR_ROOT
        return GPDS_ROOT  # GPDS

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ds = self.dataset_col[idx]
        root_folder = self._root_for(ds)

        signature1 = np.asarray(Image.open(os.path.join(root_folder, self.signature1_paths[idx])).convert("L"))
        signature2 = np.asarray(Image.open(os.path.join(root_folder, self.signature2_paths[idx])).convert("L"))
        label = int(self.labels[idx])

        if self.transforms:
            signature1 = self.transforms(image=signature1)["image"]
            signature2 = self.transforms(image=signature2)["image"]

        return signature1, signature2, label

# ===== 
dataset_valid = Siamese_Test_Dataset(df_valid, transforms=transforms_test_albm, dataset_name=DATASET)
dataloader_valid = DataLoader(dataset_valid, batch_size=32, shuffle=False, num_workers=4)

dataset_test = Siamese_Test_Dataset(df_test, transforms=transforms_test_albm, dataset_name=DATASET)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4)

# ===== 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TripletNetwork()
path_best_model = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
model.load_state_dict(torch.load(path_best_model, map_location=device))
model.to(device)
model.eval()

# ===== 
def compute_distances(dataloader):
    distances, labels = [], []
    with torch.no_grad():
        for signature1, signature2, label in tqdm(dataloader):
            signature1 = signature1.to(device)
            signature2 = signature2.to(device)

            emb1 = model.embedding_network(signature1)
            emb2 = model.embedding_network(signature2)

            emb1 = emb1.view(emb1.size(0), -1)
            emb2 = emb2.view(emb2.size(0), -1)

            batch_dist = torch.norm(emb1 - emb2, p=2, dim=1).cpu().numpy()
            distances.extend(batch_dist)
            labels.extend(label.numpy())
    return np.array(distances), np.array(labels).astype(int)

# =====
val_distances, val_labels = compute_distances(dataloader_valid)
fpr, tpr, thresholds = roc_curve(val_labels, val_distances)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.6f}")

# =====
test_distances, test_labels = compute_distances(dataloader_test)
y_pred_test = (test_distances >= optimal_threshold).astype(int)

accuracy_test = accuracy_score(test_labels, y_pred_test)
roc_auc_test = roc_auc_score(test_labels, test_distances)
print(f"Test Accuracy: {accuracy_test:.4f}")
print(f"Test ROC AUC: {roc_auc_test:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, y_pred_test))

# =====
conf_matrix = confusion_matrix(test_labels, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
suffix = DATASET.lower()
plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{suffix}.png"), bbox_inches="tight", dpi=150)

# =====
fpr_test, tpr_test, _ = roc_curve(test_labels, test_distances)
plt.figure()
plt.plot(fpr_test, tpr_test, label=f"ROC (AUC = {roc_auc_test:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Test ROC Curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{suffix}.png"), bbox_inches="tight", dpi=150)
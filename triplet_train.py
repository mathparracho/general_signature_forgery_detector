import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.nn import TripletMarginLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt

from architectures.resnet34_siamese_triplet import TripletNetwork

writer = SummaryWriter()

c = datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S')
newpath = f"./models/model_{c}/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

SPLIT_FILES_DIR = "./data/split_files"
ICDAR_ROOT = "./data/icdar_signatures/train/"
CEDAR_ROOT = "./data/cedar_signatures/"
GPDS_ROOT = "./data/firmasSINTESISmanuscritas/"

df_icdar = pd.read_csv(f"{SPLIT_FILES_DIR}/train_data_2ndAdjs.csv")
df_icdar.insert(0, "dataset", "ICDAR")

df_cedar = pd.read_csv(f"{SPLIT_FILES_DIR}/full_data_split_CEDAR.csv")
df_cedar.insert(0, "dataset", "CEDAR")

df = pd.concat([df_icdar, df_cedar])
df = df[df["forged"] == 1]

df_train = df[df["split"] == "train"].sample(frac=1)
df_valid = df[df["split"] == "valid"].sample(frac=1)
df_test = df[df["split"] == "test"].sample(frac=1)

transforms_training_albm = A.Compose(
    [
        A.CLAHE(),
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

transforms_val_albm = A.Compose(
    [
        A.CLAHE(),
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ]
)


class Siamese_Pairs_Dataset(Dataset):
    def __init__(self, df, transforms=None):
        self.signature1_paths = df["image1"].tolist()
        self.signature2_paths = df["image2"].tolist()
        self.forged_paths = df["forged"].tolist()
        self.dataset = df["dataset"].tolist()
        self.transform = transforms

    def __len__(self):
        return len(self.forged_paths)

    def __getitem__(self, idx):
        if self.dataset[idx] == "ICDAR":
            root_folder = ICDAR_ROOT
        elif self.dataset[idx] == "CEDAR":
            root_folder = CEDAR_ROOT
        else:
            root_folder = GPDS_ROOT

        signature1_path = self.signature1_paths[idx]
        signature1 = np.asarray(Image.open(root_folder + signature1_path).convert("L"))

        if self.dataset[idx] == "ICDAR":
            anchor_path = root_folder + signature1_path.split("/")[0]
            anchor_path = anchor_path + "/" + os.listdir(anchor_path)[
                random.randint(0, len(os.listdir(anchor_path)) - 1)
            ]
        elif self.dataset[idx] == "CEDAR":
            id_person = signature1_path.split("/")[1].split("_")[1]
            anchor_path = root_folder + f"full_org/original_{id_person}_{random.randint(1,9)}.png"
        else:
            id_person = signature1_path.split("/")[0]
            anchor_path = root_folder + id_person + f"/c-{id_person}-{random.randint(1,24):02d}.jpg"

        anchor = np.asarray(Image.open(anchor_path).convert("L"))

        signature2_path = self.signature2_paths[idx]
        signature2 = np.asarray(Image.open(root_folder + signature2_path).convert("L"))

        label = self.forged_paths[idx]

        if self.transform:
            signature1 = self.transform(image=signature1)
            anchor = self.transform(image=anchor)
            signature2 = self.transform(image=signature2)

        return anchor["image"], signature1["image"], signature2["image"], label


dataset_train = Siamese_Pairs_Dataset(df_train.reset_index(drop=True), transforms=transforms_training_albm)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=4)

dataset_val = Siamese_Pairs_Dataset(df_valid.reset_index(drop=True), transforms=transforms_val_albm)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)

model = TripletNetwork()
for param in model.parameters():
    param.requires_grad = True

model.to(device)

criterion = TripletMarginLoss(margin=1)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 20
train_losses = []
val_losses = []
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    for i, (anchor, signature1, signature2, _) in enumerate(tqdm(dataloader_train)):
        anchor = anchor.to(device)
        signature1 = signature1.to(device)
        signature2 = signature2.to(device)
        optimizer.zero_grad()
        anchor, signature1, signature2 = model(anchor, signature1, signature2)
        anchor, signature1, signature2 = anchor.squeeze(dim=1), signature1.squeeze(dim=1), signature2.squeeze(dim=1)
        loss = criterion(anchor, signature1, signature2)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    train_loss = training_loss / len(dataloader_train)
    train_losses.append(train_loss)
    writer.add_scalar("Loss/train", train_loss, epoch)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (anchor, signature1, signature2, _) in enumerate(tqdm(dataloader_val)):
            anchor = anchor.to(device)
            signature1 = signature1.to(device)
            signature2 = signature2.to(device)
            anchor, signature1, signature2 = model(anchor, signature1, signature2)
            anchor, signature1, signature2 = anchor.squeeze(dim=1), signature1.squeeze(dim=1), signature2.squeeze(dim=1)
            loss = criterion(anchor, signature1, signature2)
            val_loss += loss.item()

    val_loss /= len(dataloader_val)
    val_losses.append(val_loss)
    writer.add_scalar("Loss/val", val_loss, epoch)

    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        print("Saving best model...")
        best_val_loss = val_loss
        path_best_model = newpath + "/best_model.pth"
        torch.save(model.state_dict(), path_best_model)

print("Finished Training")

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolution of Training Loss")
plt.legend()
plt.savefig(newpath + "loss.png")

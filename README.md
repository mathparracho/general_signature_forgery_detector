# Signature Verification Project

This repository contains code and experiments for **offline handwritten signature verification** using **Siamese Networks**, **ResNet-based embeddings**, and **contrastive/triplet loss**.  
The models are designed to generalize across multiple datasets such as **CEDAR**, **ICDAR**, and **GPDS**, as well as merged splits.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/signature-project.git
cd signature-project
```

### 2. Create and configure your environment
We use environment variables to manage dataset paths, model directories, and experiment settings.

1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` with your preferred settings:
```env
DATASET=GPDS
BATCH_SIZE=32
MODEL_DIR=./models/model_2025-08-04_16:12:16
BEST_MODEL=best_model.pth
ROOT_FOLDER_GPDS=./data/cascas_gpds_signatures/
```

### 3. Install dependencies
It is recommended to use **Python 3.10+** and a virtual environment.
```bash
pip install -r requirements.txt
```

### 4. Train a model
```bash
python train_triplet.py
```

### 5. Evaluate a model
```bash
python test_contrastive.py
```

---

## 📂 Project Structure

```
.
├── architectures/            # Model architectures (Siamese, ResNet, etc.)
├── data/                     # Datasets and processed cascas
│   ├── split_files/          # Predefined train/val/test splits
├── models/                   # Saved checkpoints
├── helpers/                  # Utility functions (losses, augmentations, etc.)
├── train_triplet.py          # Training script (triplet loss)
├── train_contrastive.py      # Training script (contrastive loss)
├── test_contrastive.py       # Testing/evaluation script
├── .env.example              # Example environment configuration
└── README.md
```

---

## ⚙️ Environment Variables

| Variable                  | Description                                      | Example |
|----------------------------|--------------------------------------------------|---------|
| `DATASET`                 | Dataset to use (`CEDAR`, `ICDAR`, `GPDS`, `MERGED`) | GPDS |
| `SPLIT_FILES_DIR`         | Path to CSV split files                          | `./data/split_files` |
| `MODEL_DIR`               | Directory to save/load models                    | `./models/model_2025-08-04_16:12:16` |
| `BEST_MODEL`              | Filename of the best checkpoint                  | `best_model.pth` |
| `ROOT_FOLDER_CEDAR`       | Path to CEDAR cascas                             | `./data/cascas_13_cedar_signatures/` |
| `ROOT_FOLDER_ICDAR`       | Path to ICDAR cascas                             | `./data/cascas_icdar_signatures/` |
| `ROOT_FOLDER_GPDS`        | Path to GPDS cascas                              | `./data/cascas_gpds_signatures/` |
| `SEED`                    | Random seed for reproducibility                  | 0 |
| `BATCH_SIZE`              | Dataloader batch size                            | 32 |
| `NUM_WORKERS`             | Number of dataloader workers                     | 4 |

---

## 📊 Outputs
- **Training Loss curves** (`loss.png`)
- **Confusion Matrices** (`confusion_matrix_*.png`)
- **ROC Curves** (`roc_curve_*.png`)
- **Best model checkpoint** (`best_model.pth`)

---

## 🤝 Contributing
Feel free to open issues or pull requests.  
All code is cleaned for open-source release (no private tokens or credentials).

---

## 📜 License
This project is released under the MIT License.

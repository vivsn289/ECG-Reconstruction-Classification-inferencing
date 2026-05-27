# scripts/generate_confusion_matrix.py
#
# Generate a normalized confusion matrix for the validation set.
#
# Reconstructs the same train/val record split used during training
# (same RANDOM_SEED and VAL_RATIO from configs/config_1d.py) so that
# the confusion matrix reflects the model's true validation performance.
#
# Usage (from repository root):
#   python -m scripts.generate_confusion_matrix

import os
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from configs.config_1d import (
    DATA_ROOT,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LABEL_ENCODER,
    CLASS_NAMES,
    BATCH_SIZE,
    VAL_RATIO,
    RANDOM_SEED,
    SAMPLING_RATE,
)
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.dataset_1d import ECGWindowDataset

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_records(records, val_ratio, seed):
    """Same split logic as train_1d.py — must be kept in sync."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]  # train, val


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
all_records = loader.get_records()

_, val_records = split_records(all_records, VAL_RATIO, RANDOM_SEED)

val_dataset = ECGWindowDataset(
    records=val_records,
    loader=loader,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    transform=None,
    label_encoder=LABEL_ENCODER,
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(y)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------
os.makedirs("visuals", exist_ok=True)

disp = ConfusionMatrixDisplay.from_predictions(
    all_targets,
    all_preds,
    display_labels=CLASS_NAMES,
    cmap="Blues",
    normalize="true",
)

plt.title("Normalized Confusion Matrix (Validation Set)")
plt.tight_layout()
plt.savefig("visuals/confusion_matrix.png", dpi=300)
plt.close()

print("[INFO] Saved visuals/confusion_matrix.png")

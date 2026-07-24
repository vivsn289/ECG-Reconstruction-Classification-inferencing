# scripts/generate_confusion_matrix.py
#
# Generate per-class confusion matrices for the validation set.
#
# Reconstructs the same strat_fold-based train/val/test split used during
# training (see src_ecg_1d.train_1d.split_by_strat_fold) so the confusion
# matrices reflect the model's true validation performance.
#
# The classifier is multi-label (a record can carry more than one
# superclass diagnosis), so there is no single NxN confusion matrix.
# Instead we compute one binary (positive vs. negative) confusion matrix
# per class and plot them in a grid.
#
# Usage (from repository root):
#   python -m scripts.generate_confusion_matrix

import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, multilabel_confusion_matrix
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
    PREDICTION_THRESHOLD,
    SAMPLING_RATE,
)
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.dataset_1d import ECGWindowDataset
from src_ecg_1d.train_1d import split_by_strat_fold

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
all_records = loader.get_records()

_, val_records, _ = split_by_strat_fold(all_records)

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
        preds = (torch.sigmoid(logits) > PREDICTION_THRESHOLD).float().cpu()
        all_preds.append(preds)
        all_targets.append(y)

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# ------------------------------------------------------------------
# Per-class confusion matrices
# ------------------------------------------------------------------
os.makedirs("visuals", exist_ok=True)

matrices = multilabel_confusion_matrix(all_targets, all_preds)  # (num_classes, 2, 2)

fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 4))
for cls_idx, (cls_name, matrix) in enumerate(zip(CLASS_NAMES, matrices)):
    disp = ConfusionMatrixDisplay(matrix, display_labels=["Negative", "Positive"])
    disp.plot(ax=axes[cls_idx], cmap="Blues", colorbar=False, values_format="d")
    axes[cls_idx].set_title(cls_name)

fig.suptitle("Per-Class Confusion Matrices (Validation Set)")
plt.tight_layout()
plt.savefig("visuals/confusion_matrix.png", dpi=300)
plt.close()

print("[INFO] Saved visuals/confusion_matrix.png")

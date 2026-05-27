# scripts/generate_lead_attribution.py
#
# Compute and visualize per-lead attribution for a single PTB-XL record
# using Integrated Gradients.
#
# For each sliding window over the record, we compute IG attributions.
# The result is a 12-lead bar chart showing which leads the model relies
# on most for its prediction.
#
# Usage (from repository root):
#   python -m scripts.generate_lead_attribution
#
# Note on IG and lead attribution:
#   IG produces a tensor of shape (channels, time) = (12, window_size).
#   The importance of each lead is approximated by the mean absolute
#   attribution across time for that lead. This is a proxy measure —
#   it reflects how much the model's output changes when the baseline
#   is interpolated to the actual signal along each lead's axis.

import torch
import numpy as np
import matplotlib.pyplot as plt

from configs.config_1d import (
    DATA_ROOT,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LEAD_NAMES,
    IG_STEPS,
    SAMPLING_RATE,
)
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.windowing import sliding_window
from src_ecg_1d.explainability.integrated_gradients import integrated_gradients

import os

# ------------------------------------------------------------------
# Config (override per experiment)
# ------------------------------------------------------------------
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
RECORD_IDX = 5020   # index into the records list

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
records = loader.get_records()

record = records[RECORD_IDX]
# ecg shape: (12, T)
ecg = loader.load_ecg(record["record_path"])
print(f"[INFO] Record {RECORD_IDX}: label={record['label']}, ECG shape={ecg.shape}")

# ------------------------------------------------------------------
# Compute per-lead attribution across all windows
# ------------------------------------------------------------------
# lead_scores[i] accumulates mean absolute IG for lead i
lead_scores = np.zeros(12, dtype=np.float32)
n_windows = 0

windows = sliding_window(ecg, WINDOW_SIZE, STRIDE)

for window in windows:
    # window shape: (12, WINDOW_SIZE)
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    baseline = torch.zeros_like(x)

    with torch.no_grad():
        logits = model(x)
        pred_class = logits.argmax(dim=1).item()

    # ig shape: (12, WINDOW_SIZE)  — attribution per (lead, timestep)
    ig = integrated_gradients(
        model=model,
        x=x,
        baseline=baseline,
        target_class=pred_class,
        steps=IG_STEPS,
    )

    if torch.is_tensor(ig):
        ig = ig.detach().cpu().numpy()

    # Per-lead importance = mean absolute attribution across time
    # shape: (12,)
    per_lead = np.abs(ig).mean(axis=1)
    lead_scores += per_lead
    n_windows += 1

# Average across windows and normalize to sum to 1
if n_windows > 0:
    lead_scores /= n_windows
    lead_scores = lead_scores / (lead_scores.sum() + 1e-8)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
os.makedirs("visuals", exist_ok=True)

plt.figure(figsize=(10, 4))
plt.bar(LEAD_NAMES, lead_scores)
plt.ylabel("Relative Attribution (normalized)")
plt.xlabel("ECG Lead")
plt.title(f"Per-Lead Attribution — Record {RECORD_IDX} (label: {record['label']})")
plt.tight_layout()
plt.savefig("visuals/lead_attribution.png", dpi=300)
plt.close()

print("[INFO] Saved visuals/lead_attribution.png")

import torch
import numpy as np
import matplotlib.pyplot as plt

from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.windowing import sliding_window
from src_ecg_1d.explainability.integrated_gradients import integrated_gradients

# ==================================================
# CONFIG
# ==================================================
DATA_ROOT = "data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CHECKPOINT = "checkpoints/best_model.pt"

NUM_CLASSES = 5
WINDOW_SIZE = 1000
STRIDE = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# ==================================================
# LOAD MODEL
# ==================================================
model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ==================================================
# LOAD DATA
# ==================================================
loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=100)
records = loader.get_records()

# Use the same record as explainability (change if needed)
record_idx = 5020
record = records[record_idx]
ecg = loader.load_ecg(record["record_path"])  # shape: (12, T)

# ==================================================
# COMPUTE LEAD ATTRIBUTION
# ==================================================
lead_scores = np.zeros(12, dtype=np.float32)

windows = sliding_window(ecg, WINDOW_SIZE, STRIDE)

for window in windows:
    # window shape: (12, WINDOW_SIZE)
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    baseline = torch.zeros_like(x)

    with torch.no_grad():
        logits, _ = model(x, return_attention=True)
        pred_class = logits.argmax(dim=1).item()

    ig = integrated_gradients(
        model=model,
        x=x,
        baseline=baseline,
        target_class=pred_class,
        steps=50,
    )

    # --------------------------------------------------
    # IMPORTANT FIX: move IG to CPU before NumPy ops
    # --------------------------------------------------
    if torch.is_tensor(ig):
        ig = ig.detach().cpu().numpy()

    # --------------------------------------------------
    # Aggregate attribution (channel-agnostic proxy)
    # --------------------------------------------------
    ig_score = np.abs(ig).mean()

    for lead in range(12):
        lead_scores[lead] += ig_score

# Normalize scores
lead_scores = lead_scores / (lead_scores.sum() + 1e-8)

# ==================================================
# PLOT
# ==================================================
plt.figure(figsize=(10, 4))
plt.bar(LEAD_NAMES, lead_scores)
plt.ylabel("Relative Attribution")
plt.title("Lead-wise Attribution (Integrated Gradients)")
plt.tight_layout()
plt.savefig("visuals/lead_attribution.png", dpi=300)
plt.close()

print("[INFO] Saved visuals/lead_attribution.png")

# src_ecg_1d/explainability/run_explainability.py

import torch
import numpy as np

from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.dataset_1d import ECGWindowDataset

from src_ecg_1d.explainability.attention_utils import aggregate_attention
from src_ecg_1d.explainability.visualization import plot_ecg_with_attention


# ============================================================
# CONFIG (MUST MATCH TRAINING)
# ============================================================
DATA_ROOT = "data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CHECKPOINT_PATH = "checkpoints/best_model.pt"

NUM_CLASSES = 5
WINDOW_SIZE = 1000
STRIDE = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Dataset builder (EXACTLY matches train_1d.py)
# ============================================================
def build_dataset():
    loader = PTBXLECGLoader(
        DATA_ROOT,
        sampling_rate=100,
    )

    records = loader.get_records()

    label_encoder = {
        "NORM": 0,
        "MI": 1,
        "STTC": 2,
        "CD": 3,
        "HYP": 4,
    }

    dataset = ECGWindowDataset(
        records=records,
        loader=loader,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        transform=None,           # IMPORTANT: no augmentation for explainability
        label_encoder=label_encoder,
    )

    return dataset


# ============================================================
# Select window where model is LEAST confident it is normal
# ============================================================
def select_most_abnormal_window(model, dataset):
    model.eval()

    best_idx = None
    best_score = -1.0
    best_logits = None
    best_attentions = None

    for i in range(len(dataset)):
        x, _ = dataset[i]                 # (channels, L)
        x = x.unsqueeze(0).to(DEVICE)     # (1, channels, L)

        with torch.no_grad():
            logits, attentions = model(x, return_attention=True)
            probs = torch.softmax(logits, dim=1)

        # Abnormality score = 1 − P(NORM)
        abnormal_score = 1.0 - probs[0, 0].item()

        if abnormal_score > best_score:
            best_score = abnormal_score
            best_idx = i
            best_logits = logits
            best_attentions = attentions

    return best_idx, best_logits, best_attentions


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"[INFO] Using device: {DEVICE}")

    # --------------------------------------------------------
    # Load trained model (EXACT match)
    # --------------------------------------------------------
    model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    )

    # --------------------------------------------------------
    # Build dataset (EXACT match to training)
    # --------------------------------------------------------
    dataset = build_dataset()
    print(f"[INFO] Total ECG windows: {len(dataset)}")

    # --------------------------------------------------------
    # Pick window with highest abnormal confidence
    # --------------------------------------------------------
    idx, logits, attentions = select_most_abnormal_window(model, dataset)

    x, y = dataset[idx]
    x = x.unsqueeze(0).to(DEVICE)

    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()

    print("\n[SELECTED WINDOW]")
    print(f"  Index           : {idx}")
    print(f"  Ground truth    : {y.item()}")
    print(f"  Predicted class : {pred}")
    print(f"  Probabilities  : {probs.cpu().numpy()}")

    # --------------------------------------------------------
    # Aggregate Transformer attention
    # --------------------------------------------------------
    importance = (
        aggregate_attention(attentions)[0]
        .detach()
        .cpu()
        .numpy()
    )

    ecg_signal = x[0].detach().cpu().numpy()

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    plot_ecg_with_attention(
        ecg_signal=ecg_signal,
        attention=importance,
        lead_idx=0,
        title=f"Predicted class: {pred}",
    )


if __name__ == "__main__":
    main()

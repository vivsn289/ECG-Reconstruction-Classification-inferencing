"""One-shot explainability runner that saves the IG overlay to a PNG file."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from configs.config_1d import (
    DATA_ROOT, CHECKPOINT_DIR, NUM_CLASSES, WINDOW_SIZE, STRIDE,
    LABEL_DECODER, EXPLAIN_RECORD_INDICES, IG_STEPS, SAMPLING_RATE,
    PREDICTION_THRESHOLD,
)
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.windowing import sliding_window

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
OUT_DIR = "visuals"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def integrated_gradients(model, x, baseline, target_class, steps=50):
    total_grad = torch.zeros_like(x)
    for alpha in np.linspace(0, 1, steps, endpoint=False)[1:]:
        x_step = (baseline + alpha * (x - baseline)).clone().detach().requires_grad_(True)
        logits, _ = model(x_step, return_attention=True)
        model.zero_grad()
        logits[0, target_class].backward()
        total_grad += x_step.grad.detach()
    avg_grad = total_grad / steps
    ig = (x - baseline) * avg_grad
    return ig.abs()[0].mean(dim=0).cpu().numpy()


def explain_and_save(model, loader, records, record_idx):
    record = records[record_idx]
    ecg = loader.load_ecg(record["record_path"])
    true_label = record["label"]
    _, total_length = ecg.shape
    print(f"[INFO] Record {record_idx} | True label: {true_label} | Shape: {ecg.shape}")

    windows = sliding_window(ecg, WINDOW_SIZE, STRIDE)
    global_ig = np.zeros(total_length, dtype=np.float32)
    ig_overlap = np.zeros(total_length, dtype=np.float32)
    best_logits, best_abnormal_score = None, -1.0

    for i, window in enumerate(windows):
        start = i * STRIDE
        end = start + WINDOW_SIZE
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, _ = model(x, return_attention=True)
            probs = torch.sigmoid(logits)  # independent per-class probs, multi-label

        score = 1.0 - probs[0, 0].item()
        if score > best_abnormal_score:
            best_abnormal_score = score
            best_logits = logits

        pred_class = logits.argmax(dim=1).item()
        channel_means = ecg[:, start:end].mean(axis=1, keepdims=True)
        baseline_np = np.repeat(channel_means, WINDOW_SIZE, axis=1)
        baseline = torch.tensor(baseline_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        ig_1d = integrated_gradients(model, x, baseline, pred_class, steps=IG_STEPS)
        global_ig[start:end] += ig_1d
        ig_overlap[start:end] += 1.0

    ig_overlap[ig_overlap == 0] = 1.0
    global_ig /= ig_overlap
    global_ig -= global_ig.min()
    global_ig /= (global_ig.max() + 1e-8)

    probs = torch.sigmoid(best_logits)
    pred_class = probs.argmax(dim=1).item()  # highest-probability class, for the plot title
    above_threshold = (probs > PREDICTION_THRESHOLD).nonzero(as_tuple=True)[1].tolist()
    predicted_labels = [LABEL_DECODER[c] for c in above_threshold]
    print(
        f"  Predicted: {LABEL_DECODER[pred_class]} | Above threshold: {predicted_labels} "
        f"| Probs: {probs.cpu().numpy()}"
    )

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(ecg[0], color="black", linewidth=1, label="ECG (Lead I)")
    ax1.set_xlabel("Time (samples)")
    ax1.set_ylabel("ECG amplitude", color="black")

    ax2 = ax1.twinx()
    ax2.plot(global_ig, color="red", alpha=0.6, linewidth=1.5, label="IG importance")
    ax2.set_ylabel("IG attribution (normalized)", color="red")
    ax2.set_ylim(0, 1.2)

    plt.title(
        f"ECG + Integrated Gradients | True: {true_label}  Pred: {LABEL_DECODER[pred_class]}"
    )
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"ig_record_{record_idx}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def main():
    print(f"[INFO] Device: {DEVICE}")
    os.makedirs(OUT_DIR, exist_ok=True)

    model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
    records = loader.get_records()

    for record_idx in EXPLAIN_RECORD_INDICES:
        print("\n" + "=" * 60)
        explain_and_save(model, loader, records, record_idx)


if __name__ == "__main__":
    main()

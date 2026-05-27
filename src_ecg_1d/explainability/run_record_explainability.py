# src_ecg_1d/explainability/run_record_explainability.py
#
# Explainability analysis for individual PTB-XL records.
#
# For each configured record, this script:
#   1. Loads the ECG signal
#   2. Runs the model over all sliding windows
#   3. Computes Integrated Gradients (IG) per window
#   4. Stitches IG attributions back to the full signal length
#   5. Displays/saves ECG + IG overlay plots
#
# Usage (from repository root):
#   python -m src_ecg_1d.explainability.run_record_explainability
#
# IG baseline: per-window channel mean repeated across time.
# This reflects the model's response relative to a "flat" signal
# at each lead's local average — a more informative baseline than
# an all-zeros baseline when signals have non-zero DC offsets.

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from configs.config_1d import (
    DATA_ROOT,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LABEL_DECODER,
    EXPLAIN_RECORD_INDICES,
    IG_STEPS,
    SAMPLING_RATE,
)
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.windowing import sliding_window

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def integrated_gradients(model, x, baseline, target_class, steps=50):
    """Compute Integrated Gradients attributions.

    Approximates the path integral from baseline to input by interpolating
    between them in `steps` steps, accumulating gradients at each step.

    Args:
        model: the ECGClassifier1D model in eval mode
        x: input tensor, shape (1, channels, time)
        baseline: baseline tensor, same shape as x.
            Conventionally: zeros or per-channel mean.
        target_class: integer class index to explain
        steps: number of interpolation steps (more → smoother, slower)

    Returns:
        ig_1d: 1D numpy array of shape (time,) — mean absolute attribution
               across channels, used for full-signal overlay visualization.

    Notes:
        - We skip α=0 (baseline) since its gradient contributes trivially.
        - We average across all steps (Riemann approximation of the integral).
        - The raw IG tensor has shape (1, channels, time); we take the
          absolute value and average across channels to produce ig_1d.
    """
    assert x.shape == baseline.shape

    total_grad = torch.zeros_like(x)

    # Interpolation steps from just above baseline (α=1/steps) to input (α=1)
    for alpha in np.linspace(0, 1, steps, endpoint=False)[1:]:
        x_step = baseline + alpha * (x - baseline)
        x_step = x_step.clone().detach().requires_grad_(True)

        logits, _ = model(x_step, return_attention=True)
        target_logit = logits[0, target_class]

        model.zero_grad()
        target_logit.backward()
        total_grad += x_step.grad.detach()

    avg_grad = total_grad / steps

    # IG formula: (input - baseline) * average_gradient
    # Shape: (1, channels, time)
    ig = (x - baseline) * avg_grad

    # Collapse to 1D by taking mean absolute value across channels
    # Shape: (time,)
    ig_1d = ig.abs()[0].mean(dim=0)

    return ig_1d.cpu().numpy()


def explain_record(model, loader, records, record_idx):
    """Run IG-based explanation for one record and display the result."""
    record = records[record_idx]
    # ecg shape: (12, T)
    ecg = loader.load_ecg(record["record_path"])
    true_label = record["label"]

    _, total_length = ecg.shape
    print(f"[INFO] True label: {true_label} | ECG shape: {ecg.shape}")

    windows = sliding_window(ecg, WINDOW_SIZE, STRIDE)

    # Accumulate IG over all windows, averaging overlapping regions
    global_ig = np.zeros(total_length, dtype=np.float32)
    ig_overlap = np.zeros(total_length, dtype=np.float32)  # count of windows covering each position

    best_logits = None
    best_abnormal_score = -1.0

    for i, window in enumerate(windows):
        start = i * STRIDE
        end = start + WINDOW_SIZE

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, _ = model(x, return_attention=True)
            probs = torch.softmax(logits, dim=1)

        # Track the window with the highest abnormality signal (least NORM prob)
        abnormal_score = 1.0 - probs[0, 0].item()
        if abnormal_score > best_abnormal_score:
            best_abnormal_score = abnormal_score
            best_logits = logits

        pred_class = logits.argmax(dim=1).item()

        # Baseline: per-channel mean of this window, broadcast across time
        # This gives a more meaningful baseline than zeros when signals are offset
        channel_means = ecg[:, start:end].mean(axis=1, keepdims=True)  # (12, 1)
        baseline_np = np.repeat(channel_means, WINDOW_SIZE, axis=1)    # (12, WINDOW_SIZE)
        baseline = torch.tensor(baseline_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        ig_1d = integrated_gradients(
            model=model,
            x=x,
            baseline=baseline,
            target_class=pred_class,
            steps=IG_STEPS,
        )

        global_ig[start:end] += ig_1d
        ig_overlap[start:end] += 1.0

    # Avoid division by zero in non-covered regions
    ig_overlap[ig_overlap == 0] = 1.0
    global_ig /= ig_overlap

    # Normalize to [0, 1] for visualization
    global_ig -= global_ig.min()
    global_ig /= (global_ig.max() + 1e-8)

    # Final prediction: from the most abnormal window
    probs = torch.softmax(best_logits, dim=1)
    pred_class = probs.argmax(dim=1).item()

    print("[RESULT]")
    print(f"  Predicted label : {LABEL_DECODER[pred_class]}")
    print(f"  Probabilities   : {probs.cpu().numpy()}")

    # Plot ECG signal (Lead I) with IG overlay on secondary axis
    fig, ax1 = plt.subplots(figsize=(14, 4))

    ax1.plot(ecg[0], color="black", linewidth=1, label="ECG (Lead I)")
    ax1.set_xlabel("Time (samples)")
    ax1.set_ylabel("ECG amplitude", color="black")

    ax2 = ax1.twinx()
    ax2.plot(global_ig, color="red", alpha=0.6, linewidth=1.5, label="IG importance")
    ax2.set_ylabel("IG attribution (normalized)", color="red")
    ax2.set_ylim(0, 1.2)

    plt.title(
        f"ECG + Integrated Gradients | "
        f"True: {true_label}  Pred: {LABEL_DECODER[pred_class]}"
    )
    plt.tight_layout()
    plt.show()


def main():
    print(f"[INFO] Device: {DEVICE}")

    model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
    records = loader.get_records()

    for record_idx in EXPLAIN_RECORD_INDICES:
        print("\n" + "=" * 70)
        print(f"[INFO] Explaining record index: {record_idx}")
        explain_record(model, loader, records, record_idx)


if __name__ == "__main__":
    main()

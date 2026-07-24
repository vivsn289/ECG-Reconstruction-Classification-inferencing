# src_ecg_1d/contrastive/run_contrastive.py
#
# Layer 1 end-to-end test: Contrastive IG + Evidence Extraction.
#
# For each record in EXPLAIN_RECORD_INDICES:
#   1. Computes per-lead ΔIG over all sliding windows
#   2. Extracts structured evidence items
#   3. Prints evidence table and natural-language query string
#   4. Saves a 12-panel visualization to visuals/
#
# Usage (from repository root):
#   python -m src_ecg_1d.contrastive.run_contrastive

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.config_1d import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    EXPLAIN_RECORD_INDICES,
    LABEL_DECODER,
    LEAD_NAMES,
    NUM_CLASSES,
    SAMPLING_RATE,
)
from src_ecg_1d.contrastive.delta_ig import compute_record_contrastive_ig
from src_ecg_1d.contrastive.evidence import evidence_to_query, extract_evidence
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.models.ecg_model import ECGClassifier1D

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISUALS_DIR = "visuals"


def visualize_contrastive_ig(
    ecg: np.ndarray,
    delta_ig: np.ndarray,
    record_idx: int,
    pred_class_name: str,
    runner_up_class_name: str,
    save_dir: str = VISUALS_DIR,
) -> None:
    """Save a 12-panel ECG plot with ΔIG polarity overlay.

    Each panel shows one ECG lead (black trace). Timestep regions where ΔIG
    is positive (favoring the predicted class) are highlighted in green;
    regions where ΔIG is negative (favoring the runner-up) are highlighted
    in red.

    Args:
        ecg: numpy array (12, T) — raw ECG signal.
        delta_ig: numpy array (12, T) — contrastive IG attribution map.
        record_idx: record index used in the output filename.
        pred_class_name: predicted class label (e.g. "MI").
        runner_up_class_name: runner-up class label (e.g. "STTC").
        save_dir: directory to write the PNG file into.
    """
    os.makedirs(save_dir, exist_ok=True)
    _, T = ecg.shape
    time = np.arange(T)

    fig, axes = plt.subplots(12, 1, figsize=(18, 24), sharex=True)
    fig.suptitle(
        f"Contrastive IG — Record {record_idx} | "
        f"Predicted: {pred_class_name}   Runner-up: {runner_up_class_name}",
        fontsize=13,
        y=0.998,
    )

    for lead_idx, ax in enumerate(axes):
        signal = ecg[lead_idx]
        dig = delta_ig[lead_idx]

        sig_min = float(signal.min())
        sig_max = float(signal.max())
        pad = 0.05 * (sig_max - sig_min or 1.0)
        y_lo, y_hi = sig_min - pad, sig_max + pad

        ax.plot(time, signal, color="black", linewidth=0.7)

        ax.fill_between(
            time, y_lo, y_hi,
            where=(dig > 0),
            color="green", alpha=0.35, linewidth=0,
        )
        ax.fill_between(
            time, y_lo, y_hi,
            where=(dig < 0),
            color="red", alpha=0.35, linewidth=0,
        )

        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel(LEAD_NAMES[lead_idx], rotation=0, labelpad=30, fontsize=9)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (samples)", fontsize=10)

    legend_handles = [
        mpatches.Patch(color="green", alpha=0.5, label=f"Favors {pred_class_name}"),
        mpatches.Patch(color="red",   alpha=0.5, label=f"Favors {runner_up_class_name}"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    out_path = os.path.join(save_dir, f"contrastive_ig_record_{record_idx}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def explain_record(model, loader, records, record_idx: int) -> None:
    """Run the full Layer 1 pipeline for one record."""
    record = records[record_idx]
    ecg = loader.load_ecg(record["record_path"])  # (12, T)
    true_labels = sorted(record["labels"])  # multi-label ground truth, e.g. ["MI", "STTC"]

    print(f"\n{'=' * 70}")
    print(f"[INFO] Record {record_idx} | True labels: {true_labels} | Shape: {ecg.shape}")

    # ── Step 1: Contrastive IG ──────────────────────────────────────────────
    print("[STEP 1] Computing contrastive IG over all windows ...")
    delta_ig_full, attention_full, window_metadata = compute_record_contrastive_ig(
        model, ecg, DEVICE, include_attention=True
    )

    # Aggregate class probabilities across windows to pick representative labels
    mean_probs = np.mean([m["probs"] for m in window_metadata], axis=0)
    sorted_idx = np.argsort(mean_probs)[::-1]
    pred_class = int(sorted_idx[0])
    runner_up_class = int(sorted_idx[1])
    pred_name = LABEL_DECODER[pred_class]
    runner_up_name = LABEL_DECODER[runner_up_class]

    print(f"  Predicted : {pred_name}  |  Runner-up : {runner_up_name}")
    print(f"  Mean probs: { {LABEL_DECODER[i]: round(float(p), 3) for i, p in enumerate(mean_probs)} }")

    # ── Step 2: Evidence extraction ─────────────────────────────────────────
    print("[STEP 2] Extracting structured evidence ...")
    evidence = extract_evidence(
        delta_ig_full,
        lead_names=LEAD_NAMES,
        attention_upsampled=attention_full,
    )
    print(f"  Found {len(evidence)} evidence items  (showing top 10):")
    print(
        f"  {'Lead':>5}  {'Region':<16}  {'Polarity':<9}  "
        f"{'Magnitude':>10}  {'Window':^16}  Attn"
    )
    print("  " + "-" * 68)
    for item in evidence[:10]:
        attn_flag = "yes" if item.attention_agreement else " - "
        print(
            f"  {item.lead_name:>5}  {item.morphology_region:<16}  "
            f"{item.polarity:<9}  {item.magnitude:>10.5f}  "
            f"[{item.start_timestep:>4}, {item.end_timestep:>4})  {attn_flag}"
        )

    # ── Step 3: Natural-language query ──────────────────────────────────────
    query = evidence_to_query(evidence, pred_name, runner_up_name)
    print(f"\n[QUERY]\n  {query}")

    # ── Step 4: Visualization ───────────────────────────────────────────────
    print("\n[STEP 4] Generating visualization ...")
    visualize_contrastive_ig(
        ecg=ecg,
        delta_ig=delta_ig_full,
        record_idx=record_idx,
        pred_class_name=pred_name,
        runner_up_class_name=runner_up_name,
    )


def main() -> None:
    print(f"[INFO] Device: {DEVICE}")

    model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
    records = loader.get_records()

    for record_idx in EXPLAIN_RECORD_INDICES:
        explain_record(model, loader, records, record_idx)


if __name__ == "__main__":
    main()

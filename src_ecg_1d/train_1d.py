# src_ecg_1d/train_1d.py
#
# Main training entry point for the 1D ECG signal classification pipeline.
#
# Architecture: 12-lead ECG → sliding windows → CNN backbone (residual + ECA)
#               → Transformer encoder → mean pool → linear classifier
#
# Run from the repository root:
#   python -m src_ecg_1d.train_1d

import os
import random

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from configs.config_1d import (
    DATA_ROOT,
    CHECKPOINT_DIR,
    SAMPLING_RATE,
    WINDOW_SIZE,
    STRIDE,
    LABEL_ENCODER,
    NUM_CLASSES,
    IN_CHANNELS,
    EMBED_DIM,
    TRANSFORMER_DEPTH,
    TRANSFORMER_HEADS,
    DROPOUT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    RANDOM_SEED,
    EARLY_STOPPING_PATIENCE,
    AUG_AMPLITUDE_SCALE_RANGE,
    AUG_AMPLITUDE_SCALE_P,
    AUG_GAUSSIAN_NOISE_STD,
    AUG_GAUSSIAN_NOISE_P,
    AUG_BASELINE_WANDER_MAX_AMP,
    AUG_BASELINE_WANDER_P,
    AUG_TIME_SHIFT_MAX,
    AUG_TIME_SHIFT_P,
)
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.training.trainer_1d import Trainer1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.dataset_1d import ECGWindowDataset
from src_ecg_1d.data.transforms_1d import (
    Compose,
    AmplitudeScaling,
    GaussianNoise,
    BaselineWander,
    TimeShift,
)


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_strat_fold(records):
    """Split records using PTB-XL's patient-level strat_fold column.

    strat_fold (1-10) is a stratified, patient-level split provided by the
    PTB-XL authors — using it (instead of a random per-record shuffle)
    guarantees no patient's records leak across train/val/test.

    Standard PTB-XL protocol: folds 1-8 train, fold 9 val, fold 10 test.

    Args:
        records: list of record dicts from PTBXLECGLoader.get_records()
            (must contain a "strat_fold" key)

    Returns:
        train_records, val_records, test_records
    """
    train_records = [r for r in records if r["strat_fold"] <= 8]
    val_records = [r for r in records if r["strat_fold"] == 9]
    test_records = [r for r in records if r["strat_fold"] == 10]

    return train_records, val_records, test_records


def compute_pos_weights(dataset, num_classes: int, device):
    """Compute positive class weights for BCEWithLogitsLoss.

    pos_weight[c] = num_negative_samples[c] / num_positive_samples[c]

    This upweights the loss on positive examples for rare classes (e.g.
    HYP), counteracting PTB-XL's class imbalance.

    Args:
        dataset: ECGWindowDataset (labels are multi-hot float tensors)
        num_classes: total number of classes
        device: torch device to place the weight tensor on

    Returns:
        pos_weights: FloatTensor of shape (num_classes,)
    """
    all_labels = torch.stack([label for _, label in dataset.samples])  # (N, num_classes)

    pos_counts = all_labels.sum(dim=0)
    neg_counts = len(dataset) - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-8)

    return pos_weights.to(device)


def main():
    set_seed(RANDOM_SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading and record-level train/val split
    # ------------------------------------------------------------------
    loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
    all_records = loader.get_records()

    train_records, val_records, test_records = split_by_strat_fold(all_records)
    print(
        f"[DATA] Train records: {len(train_records)} | "
        f"Val records: {len(val_records)} | "
        f"Test records: {len(test_records)} (held out, folds 1-8/9/10)"
    )

    # Training augmentation pipeline
    # Applied to train windows only — val receives no augmentation.
    train_transform = Compose([
        AmplitudeScaling(scale_range=AUG_AMPLITUDE_SCALE_RANGE, p=AUG_AMPLITUDE_SCALE_P),
        GaussianNoise(std=AUG_GAUSSIAN_NOISE_STD, p=AUG_GAUSSIAN_NOISE_P),
        BaselineWander(max_amplitude=AUG_BASELINE_WANDER_MAX_AMP, p=AUG_BASELINE_WANDER_P),
        TimeShift(max_shift=AUG_TIME_SHIFT_MAX, p=AUG_TIME_SHIFT_P),
    ])

    # Two separate datasets — different transforms, no shared state
    train_dataset = ECGWindowDataset(
        records=train_records,
        loader=loader,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        transform=train_transform,
        label_encoder=LABEL_ENCODER,
    )

    val_dataset = ECGWindowDataset(
        records=val_records,
        loader=loader,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        transform=None,   # no augmentation on val
        label_encoder=LABEL_ENCODER,
    )

    print(f"[DATA] Train windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Pos weights for BCEWithLogitsLoss (computed from training set only)
    # ------------------------------------------------------------------
    print("[INFO] Computing pos_weights...")
    pos_weights = compute_pos_weights(train_dataset, NUM_CLASSES, DEVICE)
    print(f"[INFO] Pos weights: {[f'{w:.3f}' for w in pos_weights.tolist()]}")

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler
    # ------------------------------------------------------------------
    model = ECGClassifier1D(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        transformer_depth=TRANSFORMER_DEPTH,
        transformer_heads=TRANSFORMER_HEADS,
        dropout=DROPOUT,
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    trainer = Trainer1D(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        pos_weights=pos_weights,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
    epochs_without_improvement = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch + 1}/{EPOCHS}]")

        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        print(
            f"  Train  | Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val    | Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}"
            f"  Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [Saved] New best model — Macro-F1 = {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\n[EARLY STOPPING] Val Macro-F1 has not improved for "
                    f"{EARLY_STOPPING_PATIENCE} epochs. Stopping at epoch {epoch + 1}."
                )
                break

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------
    os.makedirs("results_1d", exist_ok=True)
    csv_path = os.path.join("results_1d", "training_metrics.csv")
    pd.DataFrame(history).to_csv(csv_path, index=False)

    print(f"\n[Done] Best validation Macro-F1: {best_f1:.4f}")
    print(f"[Done] Training history saved to {csv_path}")


if __name__ == "__main__":
    main()

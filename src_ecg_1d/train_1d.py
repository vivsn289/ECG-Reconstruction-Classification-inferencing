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
    VAL_RATIO,
    NUM_WORKERS,
    RANDOM_SEED,
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


def split_records(records, val_ratio: float, seed: int):
    """Split a list of record dicts into train and val at the record level.

    Splitting at the record level (not the window level) ensures:
    - No temporal leakage between train and val windows
    - Val windows are never augmented
    - The split is deterministic given the same seed

    Args:
        records: list of record dicts from PTBXLECGLoader.get_records()
        val_ratio: fraction of records to use for validation
        seed: random seed for the shuffle

    Returns:
        train_records, val_records
    """
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n_val = int(len(shuffled) * val_ratio)
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]

    return train_records, val_records


def compute_class_weights(dataset, num_classes: int, device):
    """Compute inverse-frequency class weights from a dataset's labels.

    Returns a tensor of shape (num_classes,) normalized so the mean weight
    is 1.0. Used to down-weight majority classes in CrossEntropyLoss.

    Args:
        dataset: ECGWindowDataset (labels are pre-encoded integers)
        num_classes: total number of classes
        device: torch device to place the weight tensor on

    Returns:
        class_weights: FloatTensor of shape (num_classes,)
    """
    # Collect all labels without applying transforms
    labels = torch.tensor([label for _, label in dataset.samples], dtype=torch.long)

    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()

    return weights.to(device)


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

    train_records, val_records = split_records(all_records, VAL_RATIO, RANDOM_SEED)
    print(f"[DATA] Train records: {len(train_records)} | Val records: {len(val_records)}")

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
    # Class weights (computed from training set only)
    # ------------------------------------------------------------------
    print("[INFO] Computing class weights...")
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES, DEVICE)
    print(f"[INFO] Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

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
        class_weights=class_weights,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
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
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [Saved] New best model — Macro-F1 = {best_f1:.4f}")

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

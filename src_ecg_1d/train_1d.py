# src_ecg_1d/train_1d.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

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


def main():
    # ==================================================
    # CONFIG
    # ==================================================
    DATA_ROOT = "data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    NUM_CLASSES = 5
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    WINDOW_SIZE = 1000
    STRIDE = 500
    VAL_RATIO = 0.2
    NUM_WORKERS = 4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {DEVICE}")

    os.makedirs("checkpoints", exist_ok=True)

    # ==================================================
    # DATA LOADING
    # ==================================================
    loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=100)
    records = loader.get_records()

    label_encoder = {
        "NORM": 0,
        "MI": 1,
        "STTC": 2,
        "CD": 3,
        "HYP": 4,
    }

    train_transform = Compose([
        AmplitudeScaling(),
        GaussianNoise(),
        BaselineWander(),
        TimeShift(),
    ])

    dataset = ECGWindowDataset(
        records=records,
        loader=loader,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        transform=train_transform,
        label_encoder=label_encoder,
    )

    # ==================================================
    # TRAIN / VAL SPLIT
    # ==================================================
    total_size = len(dataset)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

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

    print(f"[DATA] Train samples: {len(train_dataset)}")
    print(f"[DATA] Val samples:   {len(val_dataset)}")

    # ==================================================
    # CLASS WEIGHTS (IMPORTANT FOR PTB-XL)
    # ==================================================
    print("[INFO] Computing class weights...")
    labels = []

    for _, y in train_dataset:
        labels.append(y)

    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    class_weights = class_counts.sum() / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.mean()

    print("[INFO] Class weights:", class_weights.tolist())

    # ==================================================
    # MODEL
    # ==================================================
    model = ECGClassifier1D(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    trainer = Trainer1D(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        class_weights=class_weights,
    )

    # ==================================================
    # TRAINING LOOP
    # ==================================================
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        print(f"\n🟢 Epoch {epoch+1}/{EPOCHS}")

        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

        # ==================================================
        # SAVE BEST MODEL (BY MACRO F1)
        # ==================================================
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pt",
            )
            print(f"✅ Saved new best model (Macro-F1 = {best_f1:.4f})")

    print("\n🎉 Training complete.")
    print(f"🏆 Best validation Macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

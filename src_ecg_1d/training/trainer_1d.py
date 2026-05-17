# src_ecg_1d/training/trainer_1d.py

import torch
from tqdm import tqdm

from src_ecg_1d.training.metrics import (
    accuracy,
    macro_f1,
    per_class_recall,
)


class Trainer1D:


    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        num_classes,
        class_weights=None,
        use_amp=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes

        self.use_amp = use_amp and device.type == "cuda"

        self.criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, loader):
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        correct = 0

        for x, y in tqdm(loader, desc="Train", leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

        self.scheduler.step()

        return {
            "loss": total_loss / total_samples,
            "accuracy": correct / total_samples,
        }


    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_targets = []

        for x, y in tqdm(loader, desc="Val", leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return {
            "loss": total_loss / total_samples,
            "accuracy": accuracy(all_preds, all_targets),
            "macro_f1": macro_f1(all_preds, all_targets, self.num_classes),
            "recall_per_class": per_class_recall(
                all_preds, all_targets, self.num_classes
            ),
        }

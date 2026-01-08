# src/train_pytorch.py
"""
Robust PyTorch training loop for MobileLiteECA (patched)
- Uses data/processed/images/{train,val}
- AMP (torch.amp.autocast) safe usage
- SWA BN update via update_bn_cuda to avoid CPU/CUDA dtype mismatch
- ECG-safe augmentations
- Minimal changes to your original training flow
"""
import os
import math
import time
import gc
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.swa_utils import AveragedModel, SWALR

from src import config_pytorch as cfg
from src.model_architecture_pytorch import MobileLiteECA

# reproducibility
torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n[🚀] CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"[🟩] Using GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
else:
    print("[🟥] WARNING: Using CPU")

# ------- Dataset
class ECGDataset(Dataset):
    def __init__(self, root, class_names, train=True):
        self.samples = []
        for fname in sorted(os.listdir(root)):
            if fname.endswith(".png"):
                # label is trailing token in filename before extension
                lbl = fname.split("_")[-1].replace(".png", "")
                if lbl in class_names:
                    self.samples.append((os.path.join(root, fname), class_names.index(lbl)))
        if train:
            # ECG-safe augmentations (avoid horizontal flip or color jitter)
            self.transform = transforms.Compose([
                transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                transforms.RandomApply([transforms.RandomAffine(degrees=2, translate=(0.02, 0.02),
                                                               scale=(0.98, 1.02))], p=0.5),
                transforms.ToTensor(),
                # small amplitude scaling (implemented as a lambda on tensor)
                transforms.RandomApply([transforms.Lambda(lambda x: x * (0.9 + 0.2 * torch.rand(1)))], p=0.5),
                transforms.RandomApply([transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))], p=0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                transforms.ToTensor()
            ])

        print(f"[📦] {len(self.samples)} samples loaded from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, lbl

    def get_all_labels(self):
        return [lbl for _, lbl in self.samples]


# ------- EMA (unchanged)
class EMA:
    def __init__(self, model, decay=cfg.EMA_DECAY, device=None):
        self.decay = decay
        self.device = device
        self.shadow = {}
        for n, p in model.named_parameters():
            self.shadow[n] = p.detach().clone().to(device if device is not None else p.device)
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.detach().to(self.shadow[n].device), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            self.backup[n] = p.detach().clone()
            p.copy_(self.shadow[n].to(p.device))

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            p.copy_(self.backup[n].to(p.device))
        self.backup = None


# ------- Mixup helpers (kept)
def mixup_data(x, y, alpha=cfg.MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ------- Safe BN update for SWA (move inputs to GPU to avoid dtype mismatch)
def update_bn_cuda(loader, model, device):
    model.train()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            model(imgs)


# ------- Train
def train():
    print(f"\nConfig loaded: IMG_SIZE={cfg.IMG_SIZE}, BATCH_SIZE={cfg.BATCH_SIZE}, BASE_LR={cfg.BASE_LR}")

    # derive dataset locations (explicit)
    train_dir = os.path.join(cfg.DATA_DIR, "processed", "images", "train")
    val_dir = os.path.join(cfg.DATA_DIR, "processed", "images", "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError("Train/Val image directories not found. Expected:\n"
                                f"  {train_dir}\n  {val_dir}")

    train_ds = ECGDataset(train_dir, cfg.CLASS_NAMES, train=True)
    val_ds = ECGDataset(val_dir, cfg.CLASS_NAMES, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

    # compute class weights robustly (map to 0..C-1)
    all_lbls = train_ds.get_all_labels()
    if len(all_lbls) == 0:
        raise ValueError("No training samples found in train directory.")
    # compute_class_weight returns weight per class label; ensure classes argument includes all labels
    classes = np.arange(len(cfg.CLASS_NAMES))
    class_weights_np = compute_class_weight('balanced', classes=classes, y=np.array(all_lbls))
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    print(f"[⚖️] Class Weights: {class_weights.tolist()}")

    model = MobileLiteECA().to(device)
    print(f"[⚙️] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=getattr(cfg, "LABEL_SMOOTHING", 0.0))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)

    # scheduler
    if cfg.USE_ONE_CYCLE:
        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.ONE_CYCLE_MAX_LR,
                                                        steps_per_epoch=steps_per_epoch, epochs=cfg.EPOCHS)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.EPOCHS - cfg.WARMUP_EPOCHS), eta_min=cfg.MIN_LR)

    ema = EMA(model, decay=cfg.EMA_DECAY, device=device)
    swa_model = AveragedModel(model).to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=max(cfg.BASE_LR * 0.5, 1e-6))

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.USE_AMP and torch.cuda.is_available()))

    best_f1 = 0.0
    for epoch in range(cfg.EPOCHS):
        print(f"\n🟢 Epoch {epoch + 1}/{cfg.EPOCHS}")
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, leave=False)
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            imgs, y_a, y_b, lam = mixup_data(imgs, lbls, alpha=cfg.MIXUP_ALPHA)

            optimizer.zero_grad()
            # new amp usage
            with torch.amp.autocast(device_type="cuda", enabled=(scaler.is_enabled() if hasattr(scaler, "is_enabled") else torch.cuda.is_available())):
                outputs = model(imgs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if cfg.USE_ONE_CYCLE:
                scheduler.step()

            ema.update(model)
            train_loss += loss.item() * imgs.size(0)

        if not cfg.USE_ONE_CYCLE:
            scheduler.step()

        # Validation with EMA applied
        model.eval()
        ema.apply_to(model)
        val_loss = 0.0
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=(scaler.is_enabled() if hasattr(scaler, "is_enabled") else torch.cuda.is_available())):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds_all.extend(outputs.argmax(1).cpu().numpy())
                labels_all.extend(lbls.cpu().numpy())

        ema.restore(model)

        # metrics
        val_f1 = f1_score(labels_all, preds_all, average='macro') if len(labels_all) > 0 else 0.0
        val_acc = float(np.mean(np.array(labels_all) == np.array(preds_all))) if len(labels_all) > 0 else 0.0
        avg_val_loss = val_loss / max(1, len(val_loader.dataset))
        print(f"Epoch {epoch+1:03d} | Val Acc={val_acc:.4f} | F1={val_f1:.4f} | Val Loss={avg_val_loss:.4f}")

        # SWA: start collecting after configured epoch
        if epoch >= cfg.SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Checkpoint best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(cfg.MODELS_DIR, "best_model.pt"))
            print(f"[💾] Saved new best model (F1={best_f1:.4f})")

    # finalize SWA: update BN statistics with safe cuda loader and save SWA model
    print("[SWA] Updating BN for final SWA model...")
    update_bn_cuda(train_loader, swa_model, device)
    swa_state = swa_model.module.state_dict() if hasattr(swa_model, "module") else swa_model.state_dict()
    torch.save(swa_state, os.path.join(cfg.MODELS_DIR, "swa_final.pt"))
    print(f"[✅] Training complete — best F1={best_f1:.4f}")


if __name__ == "__main__":
    train()

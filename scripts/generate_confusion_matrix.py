import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.dataset_1d import ECGWindowDataset
from torch.utils.data import DataLoader, random_split

# ---------------- CONFIG ----------------
DATA_ROOT = "data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CHECKPOINT = "checkpoints/best_model.pt"

NUM_CLASSES = 5
WINDOW_SIZE = 1000
STRIDE = 500
BATCH_SIZE = 32
VAL_RATIO = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

# ---------------- DATA ----------------
loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=100)
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
    transform=None,
    label_encoder=label_encoder,
)

val_size = int(len(dataset) * VAL_RATIO)
_, val_dataset = random_split(
    dataset,
    [len(dataset) - val_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ---------------- MODEL ----------------
model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ---------------- PREDICTIONS ----------------
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_targets.append(y)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

# ---------------- CONFUSION MATRIX ----------------
disp = ConfusionMatrixDisplay.from_predictions(
    all_targets,
    all_preds,
    display_labels=LABELS,
    cmap="Blues",
    normalize="true",
)

plt.title("Normalized Confusion Matrix (Validation Set)")
plt.tight_layout()
plt.savefig("visuals/confusion_matrix.png", dpi=300)
plt.close()



import numpy as np
import torch
import matplotlib.pyplot as plt

from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.data.windowing import sliding_window

DATA_ROOT = "data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CHECKPOINT_PATH = "checkpoints/best_model.pt"

NUM_CLASSES = 5
WINDOW_SIZE = 1000
STRIDE = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_DECODER = {
    0: "NORM",
    1: "MI",
    2: "STTC",
    3: "CD",
    4: "HYP",
}


TEST_RECORD_INDICES = [1000, 5020, 8918]


IG_STEPS = 50

#gradietn step added for weighing channel importance in explainiability

def integrated_gradients(model, x, baseline, target_class, steps=50):

    assert x.shape == baseline.shape

    total_grad = torch.zeros_like(x)

    for alpha in np.linspace(0, 1, steps, endpoint=False)[1:]:
        x_step = baseline + alpha * (x - baseline)
        x_step = x_step.clone().detach().requires_grad_(True)

        logits, _ = model(x_step, return_attention=True)
        target_logit = logits[0, target_class]

        model.zero_grad()
        target_logit.backward()
        total_grad += x_step.grad.detach()

    avg_grad = total_grad / steps
    ig = (x - baseline) * avg_grad        
    ig_1d = ig.abs()[0].mean(dim=0)        

    return ig_1d.cpu().numpy()


def main():
    print(f"[INFO] Using device: {DEVICE}")


    model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()


    loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=100)
    records = loader.get_records()

    for record_idx in TEST_RECORD_INDICES:
        print("\n" + "=" * 70)
        print(f"[INFO] Explaining record index: {record_idx}")

        record = records[record_idx]
        ecg = loader.load_ecg(record["record_path"])
        true_label = record["label"]

        channels, total_length = ecg.shape

        print(f"[INFO] True label: {true_label}")
        print(f"[INFO] ECG shape: {ecg.shape}")

        windows = sliding_window(ecg, WINDOW_SIZE, STRIDE)

        global_ig = np.zeros(total_length, dtype=np.float32)
        ig_overlap = np.zeros(total_length, dtype=np.float32)

        best_logits = None
        best_abnormal_score = -1.0


        for i, window in enumerate(windows):
            start = i * STRIDE
            end = start + WINDOW_SIZE

            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, _ = model(x, return_attention=True)
                probs = torch.softmax(logits, dim=1)

            abnormal_score = 1.0 - probs[0, 0].item()
            if abnormal_score > best_abnormal_score:
                best_abnormal_score = abnormal_score
                best_logits = logits

            pred_class = logits.argmax(dim=1).item()


            baseline = torch.tensor(
                ecg[:, start:end].mean(axis=1, keepdims=True),
                dtype=torch.float32,
            ).repeat(1, WINDOW_SIZE).unsqueeze(0).to(DEVICE)

            ig_1d = integrated_gradients(
                model=model,
                x=x,
                baseline=baseline,
                target_class=pred_class,
                steps=IG_STEPS,
            )

            global_ig[start:end] += ig_1d
            ig_overlap[start:end] += 1.0

        ig_overlap[ig_overlap == 0] = 1.0
        global_ig /= ig_overlap

        global_ig -= global_ig.min()
        global_ig /= (global_ig.max() + 1e-8)

    
        probs = torch.softmax(best_logits, dim=1)
        pred_class = probs.argmax(dim=1).item()

        print("[RESULT]")
        print(f"  Predicted label : {LABEL_DECODER[pred_class]}")
        print(f"  Probabilities  : {probs.cpu().numpy()}")

        fig, ax1 = plt.subplots(figsize=(12, 4))

        ax1.plot(ecg[0], color="black", linewidth=1)
        ax1.set_xlabel("Time (samples)")
        ax1.set_ylabel("ECG amplitude", color="black")

        ax2 = ax1.twinx()
        ax2.plot(global_ig, color="red", alpha=0.6, linewidth=2)
        ax2.set_ylabel("IG importance", color="red")

        plt.title(
            f"ECG + Integrated Gradients "
            f"(True: {true_label}, Pred: {LABEL_DECODER[pred_class]})"
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

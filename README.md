# ECG Classification Research Framework

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Dataset](https://img.shields.io/badge/dataset-PTB--XL-green)
![License](https://img.shields.io/badge/license-research-lightgrey)

A research framework for **interpretable multi-lead ECG classification** using deep learning.
Built around a CNN + Transformer architecture trained on the PTB-XL dataset, with first-class
support for Integrated Gradients explainability and attention visualization.

---

## Quick Start

```bash
# 1. Install PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Download PTB-XL (see Dataset section below)

# 4. Train
python -m src_ecg_1d.train_1d

# 5. Visualize results
python -m scripts.generate_confusion_matrix
python -m scripts.generate_lead_attribution
python -m src_ecg_1d.explainability.run_record_explainability
```

---

## Project Overview

5-class diagnostic classification on the **PTB-XL** ECG dataset using a hybrid CNN +
Transformer architecture. Primary metric is **Macro-F1** because PTB-XL is class-imbalanced
and raw accuracy is misleading.

| Label | Condition                  |
|-------|----------------------------|
| NORM  | Normal ECG                 |
| MI    | Myocardial Infarction      |
| STTC  | ST/T-wave Change           |
| CD    | Conduction Disturbance     |
| HYP   | Hypertrophy                |

**Best validation Macro-F1 (1D pipeline): ~0.69**

---

## Repository Structure

```
nag_proj/
├── configs/
│   └── config_1d.py         # All hyperparameters for the 1D pipeline
│
├── src_ecg_1d/              # PRIMARY — 1D signal pipeline (PyTorch)
│   ├── train_1d.py          # Training entry point
│   ├── data/
│   │   ├── loaders.py       # PTBXLECGLoader — reads .dat files via wfdb
│   │   ├── dataset_1d.py    # ECGWindowDataset — sliding windows
│   │   ├── transforms_1d.py # Augmentations (amplitude scaling, noise, etc.)
│   │   └── windowing.py     # sliding_window() utility
│   ├── models/
│   │   ├── ecg_model.py     # ECGClassifier1D (top-level model)
│   │   ├── cnn_backbone.py  # Residual CNN with ECA attention
│   │   ├── eca.py           # Efficient Channel Attention (1D)
│   │   └── transformer.py   # Multi-head self-attention encoder
│   ├── training/
│   │   ├── trainer_1d.py    # Training and validation loop
│   │   ├── losses.py        # FocalLoss (alternative to CrossEntropy)
│   │   └── metrics.py       # accuracy, macro_f1, per_class_recall
│   └── explainability/
│       ├── integrated_gradients.py       # IG attribution computation
│       ├── attention_utils.py            # Attention aggregation
│       ├── visualization.py              # ECG + attribution overlay plots
│       └── run_record_explainability.py  # Per-record IG script
│
├── scripts/                 # Analysis and visualization scripts
│   ├── generate_confusion_matrix.py
│   ├── generate_lead_attribution.py
│   └── draw_architecture.py
│
├── src/                     # LEGACY — image-based pipeline (TF + PyTorch)
│
├── data/
│   └── raw/ptb-xl/          # Raw PTB-XL dataset (not tracked in git)
│
├── checkpoints/
│   └── best_model.pt        # Best checkpoint by Macro-F1 (1D pipeline)
│
├── visuals/                 # Generated plots and visualizations
└── requirements.txt
```

---

## Setup

### 1. Install PyTorch

Install first, matching your CUDA version:

```bash
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only:
pip install torch torchvision
```

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 3. Download PTB-XL

```bash
# Option A: wget
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/raw/ptb-xl/

# Option B: manual download from https://physionet.org/content/ptb-xl/
```

The dataset must be at:
```
data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
```

---

## Training

```bash
python -m src_ecg_1d.train_1d
```

All hyperparameters live in `configs/config_1d.py`. Key settings:

| Parameter         | Default | Description                            |
|-------------------|---------|----------------------------------------|
| `BATCH_SIZE`      | 32      | Training batch size                    |
| `EPOCHS`          | 50      | Total training epochs                  |
| `LEARNING_RATE`   | 1e-3    | AdamW learning rate                    |
| `WINDOW_SIZE`     | 1000    | Timesteps per sliding window           |
| `STRIDE`          | 500     | Stride between windows (50% overlap)   |
| `EMBED_DIM`       | 256     | CNN output channels / Transformer dim  |
| `TRANSFORMER_DEPTH` | 2     | Number of Transformer encoder layers   |
| `TRANSFORMER_HEADS` | 4     | Self-attention heads per layer         |
| `RANDOM_SEED`     | 42      | Seed for reproducibility               |

**Outputs:**
- `checkpoints/best_model.pt` — best checkpoint by val Macro-F1
- `training_metrics.csv` — per-epoch train/val loss, accuracy, and Macro-F1

---

## Evaluation

**Confusion matrix:**
```bash
python -m scripts.generate_confusion_matrix
# → visuals/confusion_matrix.png
```

**Lead attribution (IG-based per-lead importance):**
```bash
python -m scripts.generate_lead_attribution
# → visuals/lead_attribution.png
```

**Per-record explainability:**
```bash
python -m src_ecg_1d.explainability.run_record_explainability
# → visuals/ig_record_<index>.png for each configured record
```

Sample outputs from `visuals/`:

| Visualization | Description |
|---|---|
| `confusion_matrix.png` | Per-class prediction accuracy across all 5 diagnostic categories |
| `lead_attribution.png` | Which of the 12 leads carry the most predictive signal |
| `ig_record_*.png` | IG attribution overlaid on raw ECG for individual records |
| `attention_viz_mi_case.png` | Transformer attention rollout for an MI case |
| `training_curves.png` | Loss and Macro-F1 vs. epoch |
| `macro_f1_progress.png` | Macro-F1 convergence over training |

---

## Model Architecture

```
Input: (N, 12, 1000)           12 leads × 1000 timesteps (10 s at 100 Hz)
  ↓
CNN Backbone (4 residual stages, stride=2 each):
  Stage 1: Conv1D(12→32,  k=7, stride=2) + BN + ReLU + ECA + residual
  Stage 2: Conv1D(32→64,  k=5, stride=2) + BN + ReLU + ECA + residual
  Stage 3: Conv1D(64→128, k=3, stride=2) + BN + ReLU + ECA + residual
  Stage 4: Conv1D(128→256,k=3, stride=2) + BN + ReLU + ECA + residual
  Output:  (N, 256, 63)
  ↓
Permute → (N, 63, 256)
  ↓
Transformer Encoder (2 layers, 4 heads, pre-norm):
  Multi-head Self-Attention + MLP (expansion=4)
  Output: (N, 63, 256)
  ↓
Global Mean Pool → (N, 256)
  ↓
LayerNorm + Linear(256→5) → Logits: (N, 5)
```

**ECA (Efficient Channel Attention):** After each CNN stage, global average pooling +
lightweight 1D convolution generates per-channel weights. This allows the model to
selectively emphasize informative ECG leads at each resolution level.

**Explainability compatibility:**
- Integrated Gradients — fully differentiable forward pass
- Attention maps — accessible via `model(x, return_attention=True)`
- Gradient-based saliency — supported

---

## Explainability

**Integrated Gradients (IG)** attributes each (lead, timestep) pair with a scalar
importance score relative to a per-channel-mean baseline.

For full-signal visualization over a 10-second recording:

1. The recording is split into overlapping 1000-timestep windows
2. IG is computed per window with a zero/channel-mean baseline
3. Attribution values are averaged where windows overlap
4. The result is normalized to [0, 1] and overlaid on the ECG waveform

**Important:** IG attributions reflect what the model attends to, not clinically confirmed
pathology locations. They are useful for model auditing and behavioral analysis —
not clinical diagnosis.

---

## Data Augmentation

Applied to training windows only (`configs/config_1d.py`):

| Transform              | Default probability | Description                          |
|------------------------|---------------------|--------------------------------------|
| `AmplitudeScaling`     | 50%                 | Random amplitude scale ×[0.9, 1.1]   |
| `GaussianNoise`        | 50%                 | Additive noise (σ = 0.01)            |
| `BaselineWander`       | 30%                 | Low-frequency drift (amp ≤ 0.05)     |
| `TimeShift`            | 30%                 | Random temporal shift (≤ 50 steps)   |

---

## Legacy Pipeline (`src/`)

An earlier image-based approach is preserved for reference:

1. ECG signals are rendered as PNG images (multi-lead grid layout)
2. A MobileNetV2-inspired CNN classifies the images
3. Implemented in both TensorFlow/Keras and PyTorch

This pipeline is superseded by the 1D signal approach, which:
- Avoids information loss from rendering signals as images
- Supports direct signal-level explainability (IG on the raw waveform)
- Has faster data loading (no image I/O)
- Achieves higher Macro-F1

---

## Future Directions

- Self-supervised pre-training (contrastive or masked ECG modeling)
- Transformer-only architecture
- Beat-level segmentation and analysis
- Multi-dataset evaluation (MIT-BIH, Chapman, PhysioNet)
- Uncertainty estimation
- Experiment tracking integration (W&B / TensorBoard)

---

## Citation

PTB-XL dataset:
> Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2020).
> PTB-XL, a large publicly available electrocardiography dataset.
> *Scientific Data*, 7, 154. https://doi.org/10.1038/s41597-020-0495-6

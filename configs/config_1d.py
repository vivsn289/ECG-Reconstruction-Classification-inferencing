# configs/config_1d.py
#
# Central configuration for the 1D ECG signal classification pipeline.
#
# Philosophy: all tunable hyperparameters live here.
# Training scripts and evaluation scripts import from this file
# instead of hardcoding constants inline.
#
# To run a new experiment with different settings, change values here
# or override them in a separate experiment config that imports this
# module and patches specific values.

import os

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# All paths are relative to the repository root.
# If you run training from a different directory, set DATA_ROOT to
# an absolute path or adjust accordingly.

DATA_ROOT = (
    "data/raw/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results_1d"

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

# PTB-XL sampling rate: 100 Hz (lr) or 500 Hz (hr)
SAMPLING_RATE = 100

# Sliding window parameters
# ECG shape: (12 leads, SAMPLING_RATE * 10 seconds) = (12, 1000) at 100 Hz
WINDOW_SIZE = 1000   # timesteps per window
STRIDE = 500         # stride between windows (50% overlap)

# ------------------------------------------------------------------
# Label Mapping
# ------------------------------------------------------------------
# PTB-XL diagnostic superclasses.
# This is the canonical label encoding for the 1D pipeline.
# All scripts should import LABEL_ENCODER from here rather than
# defining their own to avoid ordering inconsistencies.

LABEL_ENCODER = {
    "NORM": 0,
    "MI":   1,
    "STTC": 2,
    "CD":   3,
    "HYP":  4,
}

LABEL_DECODER = {v: k for k, v in LABEL_ENCODER.items()}

CLASS_NAMES = list(LABEL_ENCODER.keys())   # ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(CLASS_NAMES)             # 5

# Standard ECG lead names (12-lead)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------

IN_CHANNELS = 12      # number of ECG leads
EMBED_DIM = 256       # CNN output channels / Transformer embedding dimension
TRANSFORMER_DEPTH = 2
TRANSFORMER_HEADS = 4
DROPOUT = 0.1

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2    # AdamW default weight decay
VAL_RATIO = 0.2        # fraction of records held out for validation
NUM_WORKERS = 4

# Random seed for reproducibility (dataset split, weight init, augmentation)
RANDOM_SEED = 42

# ------------------------------------------------------------------
# Augmentation (applied to training windows only)
# ------------------------------------------------------------------

AUG_AMPLITUDE_SCALE_RANGE = (0.9, 1.1)
AUG_AMPLITUDE_SCALE_P = 0.5

AUG_GAUSSIAN_NOISE_STD = 0.01
AUG_GAUSSIAN_NOISE_P = 0.5

AUG_BASELINE_WANDER_MAX_AMP = 0.05
AUG_BASELINE_WANDER_P = 0.3

AUG_TIME_SHIFT_MAX = 50
AUG_TIME_SHIFT_P = 0.3

# ------------------------------------------------------------------
# Explainability
# ------------------------------------------------------------------

# Number of interpolation steps for Integrated Gradients.
# More steps → smoother attributions, but slower.
IG_STEPS = 50

# Default record indices for explainability scripts
# (indices into the records list returned by PTBXLECGLoader.get_records())
EXPLAIN_RECORD_INDICES = [1000, 5020, 8918]

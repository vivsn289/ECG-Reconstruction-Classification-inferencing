# -----------------------------------------
#  Clean, Corrected Configuration for
#  PTB-XL Classification (PNG or Signal)
# -----------------------------------------

import os

# -----------------------------------------
# Project Paths
# -----------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# -----------------------------------------
# PTB-XL Metadata Location
# -----------------------------------------
# This MUST point to the actual ptbxl_database.csv in your downloaded dataset.
POSSIBLE_METADATA_PATHS = [
    "data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
]

# -----------------------------------------
# Label Definitions (PTB-XL Diagnostic Superclass)
# -----------------------------------------
CLASS_NAMES = ["NORM", "STTC", "MI", "CD", "HYP"]
NUM_CLASSES = len(CLASS_NAMES)

# -----------------------------------------
# Dataset Creation Settings
# -----------------------------------------

# 20% validation (if strat_fold not used)
TEST_SIZE = 0.2

# Optional: limit class size (for balancing train set only)
# None = disable balancing.
MAX_SAMPLES_PER_CLASS = None

# Image classification settings
IMG_SIZE = 224
IN_CHANNELS = 3  # PNG Images = 3 channels

# Random seed for reproducibility
RANDOM_SEED = 42

# -----------------------------------------
# Training Hyperparameters
# -----------------------------------------
BATCH_SIZE = 32
EPOCHS = 100

BASE_LR = 2e-4
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.3

EMA_DECAY = 0.999
SWA_START = 60

# Learning rate schedule
WARMUP_EPOCHS = 5
MIN_LR = 1e-6

USE_ONE_CYCLE = False
ONE_CYCLE_MAX_LR = 1e-3

# -----------------------------------------
# Runtime / Performance Settings
# -----------------------------------------
USE_AMP = True              # automatic mixed precision
NUM_WORKERS = 4
PIN_MEMORY = True

# -----------------------------------------
# Augmentation Settings (Image-based ECG Only)
# NOTE: These are conservative to avoid distorting ECG morphology
# -----------------------------------------
AUGMENT_BRIGHTNESS = 0.05
AUGMENT_CONTRAST = 0.05
AUGMENT_SATURATION = 0.05
AUGMENT_ROTATION = 2.0
AUGMENT_TRANSLATE = 0.02
AUGMENT_SCALE = (0.98, 1.02)

SPATIAL_DROPOUT = 0.1
DROPOUT_RATE = 0.2

# -----------------------------------------
# Print confirmation
# -----------------------------------------
# Note: avoid printing here to prevent noise on every import of this module.

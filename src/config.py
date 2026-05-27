# src/config.py — tuned for smoother validation & generalization
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ptb-xl')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
IMAGES_CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'images')

for directory in [RESULTS_DIR, MODELS_DIR, LOGS_DIR, IMAGES_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

CLASS_NAMES = ['NORM', 'STTC', 'MI', 'CD', 'HYP']
NUM_CLASSES = len(CLASS_NAMES)

# Data
MAX_SAMPLES_PER_CLASS = 4000
TRAIN_MAX_SAMPLES = 12000
VAL_MAX_SAMPLES = 3000
TEST_SIZE = 0.2

# Images
IMG_SIZE = 224            # reduced for faster, less-noisy gradients
IMG_DPI = 150
IMG_FORMAT = 'png'
NUM_LEADS = 12

# Training
BATCH_SIZE = 32           # typically 32 on RTX4050 (adjust if memory allows)
EPOCHS = 100
RANDOM_SEED = 1337

# Optimizer / LR
BASE_LR = 2e-4
MIN_LR = 1e-7
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-3      # slightly stronger reg
LABEL_SMOOTHING = 0.05

# Mixup / augmentation
MIXUP_ALPHA = 0.2
AUGMENTATION_INTENSITY = 'low'  # 'none'|'low'|'medium'
AUGMENT_HORIZONTAL_FLIP = True
AUGMENT_ROTATION_RANGE = 0.01    # radians ~0.6 deg
AUGMENT_ZOOM_RANGE = 0.03
AUGMENT_CONTRAST_RANGE = 0.08
AUGMENT_BRIGHTNESS_DELTA = 0.05
AUGMENT_NOISE_STDDEV = 0.02

# Model parameters
MOBILENET_ALPHA = 0.9    # width multiplier to reduce params slightly
DROPOUT_RATE = 0.30
SPATIAL_DROPOUT = 0.15   # spatial dropout inside model for conv features

# EMA / SWA
EMA_DECAY = 0.999
SWA_START_EPOCH = 30
SWA_FREQ = 1

# Monitoring / checkpointing
CHECKPOINT_MONITOR = 'val_macro_f1'   # prefer macro f1 for class-imbalanced problems
CHECKPOINT_MODE = 'max'
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 5e-4

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(LOGS_DIR, 'training.log')

# Note: avoid printing here to prevent noise on every import of this module.

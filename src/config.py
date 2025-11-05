"""
Configuration file for ECG Classification Project
Centralizes all hyperparameters, paths, and constants
"""

import os

# ============================================================================
# PROJECT PATHS (Cross-platform compatible)
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ptb-xl')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
IMAGES_CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'images')

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, LOGS_DIR, IMAGES_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
CLASS_NAMES = ['NORM', 'STTC', 'MI', 'CD', 'HYP']
NUM_CLASSES = len(CLASS_NAMES)

# Dataset sampling limits (memory-safe)
MAX_SAMPLES_PER_CLASS = 3000
TRAIN_MAX_SAMPLES = 10000
VAL_MAX_SAMPLES = 2500
TEST_SIZE = 0.2  # Train/validation split ratio

# ============================================================================
# IMAGE GENERATION PARAMETERS (Enhanced)
# ============================================================================
IMG_SIZE = 512  # High-resolution output (512x512)
IMG_DPI = 150   # High DPI for quality
IMG_FORMAT = 'png'  # Output format

# ECG plotting parameters
ECG_DURATION = 10.0  # seconds
ECG_SAMPLE_RATE = 100  # Hz (for PTB-XL low-res)
ECG_TARGET_LENGTH = 5000  # samples (10s * 500Hz or padded)
NUM_LEADS = 12

# Grid specifications (standard ECG paper)
GRID_SMALL_SQUARE_MM = 1  # 1mm squares
GRID_LARGE_SQUARE_MM = 5  # 5mm squares (bold)
GRID_COLOR_SMALL = '#ffcccc'  # Light red/pink
GRID_COLOR_LARGE = '#ff9999'  # Darker red/pink
GRID_LINEWIDTH_SMALL = 0.3
GRID_LINEWIDTH_LARGE = 0.8

# Signal appearance
SIGNAL_COLOR = 'black'
SIGNAL_LINEWIDTH = 1.5
BACKGROUND_COLOR = 'white'

# Fixed scales (clinical standard)
VOLTAGE_SCALE_MIN = -2.0  # mV
VOLTAGE_SCALE_MAX = 2.0   # mV
TIME_SCALE_MAX = ECG_DURATION  # seconds

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-3

# MobileNet configuration
MOBILENET_ALPHA = 1.0  # Width multiplier
DROPOUT_RATE = 0.2

# Regularization
L1_LAMBDA = 1e-5

# Knowledge distillation (future use)
TEACHER_TEMP = 4.0
ALPHA_DISTILL = 0.7

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
AUGMENT_HORIZONTAL_FLIP = True
AUGMENT_ROTATION_RANGE = 0.01  # radians (~0.6 degrees)
AUGMENT_ZOOM_RANGE = 0.05
AUGMENT_CONTRAST_RANGE = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# Model checkpointing
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'
CHECKPOINT_SAVE_BEST_ONLY = True

# ============================================================================
# SCP CODE MAPPINGS (PTB-XL diagnostic codes to MOODY classes)
# ============================================================================
SCP_TO_MOODY_MAPPING = {
    'NORM': ['NORM'],
    'STTC': ['STTC', 'NST_', 'DIG', 'LNGQT', 'ABNORMAL_T', 'STD_', 'STE_', 'ABQRS'],
    'MI': ['MI', 'IMI', 'AMI', 'LMI', 'PMI', 'SCAR', 'QWAVE'],
    'CD': ['CD', '1AVB', '2AVB', '3AVB', 'IVCB', 'LBBB', 'RBBB', 'LAFB', 'LPFB', 'WPW'],
    'HYP': ['HYP', 'LVH', 'RVH', 'LAO/LAE', 'RAO/RAE', 'LVOLT', 'HVOLT']
}

# ============================================================================
# METADATA FILE PATHS (PTB-XL specific)
# ============================================================================
METADATA_FILENAME = 'ptbxl_database.csv'
POSSIBLE_METADATA_PATHS = [
    os.path.join(DATA_DIR, METADATA_FILENAME),
    os.path.join(DATA_DIR, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3', METADATA_FILENAME),
    os.path.join(DATA_DIR, 'records500', METADATA_FILENAME),
    os.path.join(DATA_DIR, 'database.csv')
]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOGS_DIR, 'training.log')

print(f"✓ Configuration loaded")
print(f"  - Data directory: {DATA_DIR}")
print(f"  - Models directory: {MODELS_DIR}")
print(f"  - Image size: {IMG_SIZE}x{IMG_SIZE} @ {IMG_DPI} DPI")
print(f"  - Classes: {CLASS_NAMES}")

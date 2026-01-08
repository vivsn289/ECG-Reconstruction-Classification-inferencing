"""
Fix train/val split for existing images
Moves 20% of training images to validation folder
"""

import os
import random
import shutil
from collections import defaultdict

# Paths
TRAIN_DIR = 'data/processed/images/train'
VAL_DIR = 'data/processed/images/val'

# Ensure val directory exists
os.makedirs(VAL_DIR, exist_ok=True)

# Get all training images
train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.png')]
print(f"Found {len(train_files)} images in train folder")

# Group by class to maintain class distribution
class_files = defaultdict(list)
for fname in train_files:
    # Extract class from filename (format: ID_w0_CLASS.png)
    class_name = fname.split('_')[-1].replace('.png', '')
    class_files[class_name].append(fname)

print("\nClass distribution:")
for class_name, files in class_files.items():
    print(f"  {class_name}: {len(files)} images")

# Move 20% from each class to val
total_moved = 0
for class_name, files in class_files.items():
    # Calculate 20% for this class
    val_count = int(len(files) * 0.2)
    
    # Randomly select files to move
    random.seed(42)  # For reproducibility
    files_to_move = random.sample(files, val_count)
    
    # Move files
    for fname in files_to_move:
        src = os.path.join(TRAIN_DIR, fname)
        dst = os.path.join(VAL_DIR, fname)
        shutil.move(src, dst)
        total_moved += 1
    
    print(f"  {class_name}: Moved {val_count} images to val")

print(f"\n✓ Moved {total_moved} images to validation folder")

# Verify new counts
train_count = len([f for f in os.listdir(TRAIN_DIR) if f.endswith('.png')])
val_count = len([f for f in os.listdir(VAL_DIR) if f.endswith('.png')])
total = train_count + val_count

print(f"\n📊 New distribution:")
print(f"   Train: {train_count} ({100*train_count/total:.1f}%)")
print(f"   Val:   {val_count} ({100*val_count/total:.1f}%)")
print(f"   Total: {total}")

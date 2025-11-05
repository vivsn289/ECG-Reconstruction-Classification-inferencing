"""
Generate ECG images with PARALLEL PROCESSING
Combines Kaggle's signal processing + multi-window extraction
3-4x faster than sequential processing
"""

import os
import numpy as np
import time
from src import config
from src.dataset_maker import prepare_dataset
from src.ecg_to_image import batch_convert_ecgs_parallel
import gc


def generate_and_save_images():
    """Generate all ECG images using parallel processing"""
    
    print("="*80)
    print("ECG IMAGE GENERATION (PARALLEL + KAGGLE LOGIC + MULTI-WINDOW)")
    print("="*80)
    
    start_time = time.time()
    
    # Create output directories
    train_img_dir = os.path.join(config.IMAGES_CACHE_DIR, 'train')
    val_img_dir = os.path.join(config.IMAGES_CACHE_DIR, 'val')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    
    # Prepare dataset
    print("\nLoading dataset metadata...")
    train_df, val_df, label_encoder = prepare_dataset()
    
    # Save label encoder
    np.save(os.path.join(config.IMAGES_CACHE_DIR, 'label_encoder.npy'), label_encoder)
    print(f"✓ Label encoder saved")
    
    # Generate training images (PARALLEL + MULTI-WINDOW)
    print("\n" + "="*80)
    print(f"GENERATING TRAINING IMAGES")
    print(f"Target: {config.TRAIN_MAX_SAMPLES} records")
    print(f"Windows per record: 3")
    print(f"Expected images: ~{config.TRAIN_MAX_SAMPLES * 3}")
    print("="*80)
    
    train_start = time.time()
    X_train, y_train, train_success, train_fail = batch_convert_ecgs_parallel(
        train_df,
        config.DATA_DIR,
        output_dir=train_img_dir,
        max_samples=config.TRAIN_MAX_SAMPLES,
        windows_per_record=3,
        n_workers=6  # Use 6 CPU cores
    )
    train_time = time.time() - train_start
    
    # Save training arrays
    if len(X_train) > 0:
        print("\nSaving training data arrays...")
        np.save(os.path.join(config.IMAGES_CACHE_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(config.IMAGES_CACHE_DIR, 'y_train.npy'), y_train)
        print(f"✓ Saved X_train: {X_train.shape}")
        print(f"✓ Saved y_train: {y_train.shape}")
        print(f"⏱  Training images generated in: {train_time/60:.2f} minutes")
    else:
        print("❌ No training images generated!")
    
    # Clear memory
    del X_train, y_train
    gc.collect()
    
    # Generate validation images (PARALLEL + MULTI-WINDOW)
    print("\n" + "="*80)
    print(f"GENERATING VALIDATION IMAGES")
    print(f"Target: {config.VAL_MAX_SAMPLES} records")
    print(f"Windows per record: 3")
    print(f"Expected images: ~{config.VAL_MAX_SAMPLES * 3}")
    print("="*80)
    
    val_start = time.time()
    X_val, y_val, val_success, val_fail = batch_convert_ecgs_parallel(
        val_df,
        config.DATA_DIR,
        output_dir=val_img_dir,
        max_samples=config.VAL_MAX_SAMPLES,
        windows_per_record=3,
        n_workers=6  # Use 6 CPU cores
    )
    val_time = time.time() - val_start
    
    # Save validation arrays
    if len(X_val) > 0:
        print("\nSaving validation data arrays...")
        np.save(os.path.join(config.IMAGES_CACHE_DIR, 'X_val.npy'), X_val)
        np.save(os.path.join(config.IMAGES_CACHE_DIR, 'y_val.npy'), y_val)
        print(f"✓ Saved X_val: {X_val.shape}")
        print(f"✓ Saved y_val: {y_val.shape}")
        print(f"⏱  Validation images generated in: {val_time/60:.2f} minutes")
    else:
        print("❌ No validation images generated!")
    
    # Clear memory
    del X_val, y_val
    gc.collect()
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("IMAGE GENERATION COMPLETE")
    print("="*80)
    print(f"📊 Training:   {train_success} records processed, {train_fail} failed")
    print(f"📊 Validation: {val_success} records processed, {val_fail} failed")
    print(f"⏱  Total time: {total_time/60:.2f} minutes")
    print(f"📁 Images saved to: {config.IMAGES_CACHE_DIR}")
    print(f"   - Training:   {train_img_dir}/")
    print(f"   - Validation: {val_img_dir}/")
    print(f"\n✅ Ready for training!")


if __name__ == "__main__":
    generate_and_save_images()

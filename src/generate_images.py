"""
Generate ECG images with PROPER TRAIN/VAL SPLIT
Combines Kaggle's signal processing + multi-window extraction
3-4x faster than sequential processing
"""

import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from src import config
from src.dataset_maker import process_and_split

from src.ecg_to_image import batch_convert_ecgs_parallel
import gc


def generate_and_save_images():
    """Generate all ECG images with proper train/val split"""
    
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
    train_df, val_df, label_encoder = process_and_split()
    
    # Combine all records
    all_records = np.concatenate([train_df.index.values, val_df.index.values])
    
    # Create proper 80/20 split
    """print("Creating train/val split (80/20)...")
    train_indices, val_indices = train_test_split(
        range(len(train_df) + len(val_df)),
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=None  # No stratification for resume compatibility
    )
    
    # Split the dataframes
    all_df = np.concatenate([train_df, val_df])
    train_df_split = train_df.iloc[train_indices[:len(train_df)]] if len(train_indices) <= len(train_df) else train_df
    
    # Actually, simpler approach: just use a split ratio
    total_records = len(train_df) + len(val_df)
    train_split = int(total_records * 0.8)
    
    # Recombine and re-split
    import pandas as pd
    all_records_df = pd.concat([train_df, val_df], ignore_index=False)
    all_records_df = all_records_df.reset_index(drop=True)
    
    train_split_idx = int(len(all_records_df) * 0.8)
    train_records = all_records_df.iloc[:train_split_idx]
    val_records = all_records_df.iloc[train_split_idx:]
    
    print(f"✓ Train records: {len(train_records)}")
    print(f"✓ Val records: {len(val_records)}")"""

    print("Using dataset_maker train/val split...")
    train_records = train_df
    val_records = val_df

    print(f"✓ Train records: {len(train_records)}")
    print(f"✓ Val records: {len(val_records)}")

    
    # Save label encoder
    np.save(os.path.join(config.IMAGES_CACHE_DIR, 'label_encoder.npy'), label_encoder)
    print(f"✓ Label encoder saved")
    
    # Generate training images (PARALLEL + MULTI-WINDOW)
    print("\n" + "="*80)
    print(f"GENERATING TRAINING IMAGES")
    print(f"Target: {len(train_records)} records")
    print(f"Windows per record: 3")
    print(f"Expected images: ~{len(train_records) * 3}")
    print("="*80)
    
    train_start = time.time()
    X_train, y_train, train_success, train_fail = batch_convert_ecgs_parallel(
        train_records,
        config.DATA_DIR,
        output_dir=train_img_dir,
        max_samples=len(train_records),
        windows_per_record=3,
        n_workers=6
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
    print(f"Target: {len(val_records)} records")
    print(f"Windows per record: 3")
    print(f"Expected images: ~{len(val_records) * 3}")
    print("="*80)
    
    val_start = time.time()
    X_val, y_val, val_success, val_fail = batch_convert_ecgs_parallel(
        val_records,
        config.DATA_DIR,
        output_dir=val_img_dir,
        max_samples=len(val_records),
        windows_per_record=3,
        n_workers=6
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
    print(f"📊 Training:   {train_success} records processed → {len(X_train) if 'X_train' in locals() else 0} images")
    print(f"📊 Validation: {val_success} records processed → {len(X_val) if 'X_val' in locals() else 0} images")
    print(f"⏱  Total time: {total_time/60:.2f} minutes")
    print(f"📁 Images saved to: {config.IMAGES_CACHE_DIR}")
    print(f"   - Training:   {train_img_dir}/ (80%)")
    print(f"   - Validation: {val_img_dir}/ (20%)")
    print(f"\n✅ Ready for training!")


if __name__ == "__main__":
    generate_and_save_images()

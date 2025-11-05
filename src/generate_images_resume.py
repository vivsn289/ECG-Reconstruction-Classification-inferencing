"""
Resume image generation from checkpoint
Adds more images to existing dataset WITHOUT repetition
Uses PARALLEL PROCESSING with Kaggle's signal processing
3-4x faster than sequential
"""

import os
import numpy as np
import pandas as pd
import time
from src import config
from src.dataset_maker import prepare_dataset
from src.ecg_to_image import batch_convert_ecgs_parallel
import gc


def get_processed_records():
    """
    Get list of records already processed
    by checking filenames in train directory
    """
    processed = set()
    train_dir = os.path.join(config.IMAGES_CACHE_DIR, 'train')
    
    if os.path.exists(train_dir):
        for fname in os.listdir(train_dir):
            if fname.endswith('.png'):
                # Extract record ID from filename (format: ID_w0_CLASS.png)
                try:
                    record_id = int(fname.split('_')[0])
                    processed.add(record_id)
                except ValueError:
                    pass
    
    return processed


def get_current_image_count():
    """Count existing images in train and val directories"""
    train_dir = os.path.join(config.IMAGES_CACHE_DIR, 'train')
    val_dir = os.path.join(config.IMAGES_CACHE_DIR, 'val')
    
    train_count = len([f for f in os.listdir(train_dir) if f.endswith('.png')]) if os.path.exists(train_dir) else 0
    val_count = len([f for f in os.listdir(val_dir) if f.endswith('.png')]) if os.path.exists(val_dir) else 0
    
    return train_count, val_count


def resume_generation(target_total=10000, windows_per_record=3, n_workers=6):
    """
    Resume image generation from checkpoint
    Adds new images without reprocessing old ones
    
    Uses PARALLEL PROCESSING with Kaggle's signal processing
    
    Args:
        target_total: Target number of total images
        windows_per_record: Number of windows per ECG record (3 = ~3x data)
        n_workers: Number of CPU cores to use for parallel processing
    """
    print("="*80)
    print("RESUME ECG IMAGE GENERATION (PARALLEL + KAGGLE LOGIC + MULTI-WINDOW)")
    print("="*80)
    
    # Get current status
    train_count, val_count = get_current_image_count()
    total_current = train_count + val_count
    
    print(f"\n📊 Current Status:")
    print(f"   Training images: {train_count}")
    print(f"   Validation images: {val_count}")
    print(f"   Total: {total_current}")
    
    # Calculate target
    images_needed = target_total - total_current
    
    print(f"\n🎯 Target: {target_total} total images")
    print(f"   Need to add: {images_needed} more images")
    print(f"   Progress: {total_current}/{target_total} ({100*total_current/target_total:.1f}%)\n")
    
    if images_needed <= 0:
        print("✅ Target already reached!")
        return
    
    # Load dataset
    print("Loading dataset metadata...")
    train_df, val_df, label_encoder = prepare_dataset()
    
    # Get records already processed
    processed_records = get_processed_records()
    print(f"Found {len(processed_records)} records already processed\n")
    
    # Filter to unprocessed records only
    all_records = pd.concat([train_df, val_df], ignore_index=True)
    all_records = all_records.reset_index(drop=True)
    
    unprocessed = all_records[~all_records.index.isin(processed_records)]
    
    print(f"Unprocessed records available: {len(unprocessed)}")
    
    if len(unprocessed) == 0:
        print("⚠️  No more unprocessed records available!")
        print("All PTB-XL records have been converted.")
        return
    
    # Calculate how many records to process
    # Each record generates ~windows_per_record images
    records_to_process = min(len(unprocessed), (images_needed // windows_per_record) + 1)
    
    print(f"\n🔄 Processing {records_to_process} new records...")
    print(f"   Windows per record: {windows_per_record}")
    print(f"   Expected new images: ~{records_to_process * windows_per_record}")
    print(f"   Parallel workers: {n_workers}\n")
    
    # Create output directories
    train_img_dir = os.path.join(config.IMAGES_CACHE_DIR, 'train')
    val_img_dir = os.path.join(config.IMAGES_CACHE_DIR, 'val')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    
    # Split unprocessed into train/val (80/20)
    split_point = int(len(unprocessed) * 0.8)
    unprocessed_train = unprocessed.iloc[:split_point]
    unprocessed_val = unprocessed.iloc[split_point:]
    
    start_time = time.time()
    
    # Process training records (PARALLEL + MULTI-WINDOW)
    if len(unprocessed_train) > 0:
        print("="*80)
        print(f"PROCESSING {len(unprocessed_train)} NEW TRAINING RECORDS (Parallel + Multi-window)")
        print("="*80)
        
        X_train_new, y_train_new, train_success, _ = batch_convert_ecgs_parallel(
            unprocessed_train,
            config.DATA_DIR,
            output_dir=train_img_dir,
            max_samples=len(unprocessed_train),
            windows_per_record=windows_per_record,
            n_workers=n_workers
        )
        
        gc.collect()
    else:
        print("⚠️  No new training records to process")
    
    # Process validation records (PARALLEL + MULTI-WINDOW)
    if len(unprocessed_val) > 0:
        print("\n" + "="*80)
        print(f"PROCESSING {len(unprocessed_val)} NEW VALIDATION RECORDS (Parallel + Multi-window)")
        print("="*80)
        
        X_val_new, y_val_new, val_success, _ = batch_convert_ecgs_parallel(
            unprocessed_val,
            config.DATA_DIR,
            output_dir=val_img_dir,
            max_samples=len(unprocessed_val),
            windows_per_record=windows_per_record,
            n_workers=n_workers
        )
        
        gc.collect()
    else:
        print("\n⚠️  No new validation records to process")
    
    # Get new counts
    train_count_new, val_count_new = get_current_image_count()
    total_new = train_count_new + val_count_new
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ RESUME COMPLETE")
    print("="*80)
    print(f"\n📊 Updated Status:")
    print(f"   Training images: {train_count_new} (added {train_count_new - train_count})")
    print(f"   Validation images: {val_count_new} (added {val_count_new - val_count})")
    print(f"   Total: {total_new} (added {total_new - total_current})")
    print(f"   Progress: {total_new}/{target_total} ({100*total_new/target_total:.1f}%)")
    
    print(f"\n⏱  Processing time: {total_elapsed/60:.2f} minutes")
    print(f"   Rate: {(total_new - total_current)/total_elapsed:.1f} images/second")
    
    if total_new >= target_total:
        print(f"\n🎉 TARGET REACHED! Ready for training!")
        print(f"   Total images: {total_new}")
    else:
        images_still_needed = target_total - total_new
        print(f"\n   Still need: {images_still_needed} more images")
        print(f"   Run this script again to continue!")
        print(f"\n   Command: python -m src.generate_images_resume")


if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - Adjust these values as needed
    # ========================================================================
    
    TARGET_IMAGES = 10000              # Target total number of images
    WINDOWS_PER_RECORD = 3              # Number of windows per ECG (3 = ~3x data)
    N_WORKERS = 6                       # Number of CPU cores to use (adjust based on your system)
    
    # ========================================================================
    
    resume_generation(
        target_total=TARGET_IMAGES,
        windows_per_record=WINDOWS_PER_RECORD,
        n_workers=N_WORKERS
    )

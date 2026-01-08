# src/generate_images_resume.py
"""
Resume image generation from checkpoint (patched).
- Uses dataset_maker.process_and_split() for correct PTB-XL labels (diagnostic_class).
- Resumable: detects already converted ecg_id from filenames.
- Chunked processing to avoid OOM / kill (configurable CHUNK_SIZE).
- DOES NOT change any signal-conversion functions; uses your existing batch_convert_ecgs_parallel.
"""

import os
import math
import numpy as np
import pandas as pd
import time
import gc
from src import config
# Use the corrected dataset_maker API
from src.dataset_maker import process_and_split
from src.ecg_to_image import batch_convert_ecgs_parallel

# ---------- small runtime constants you can tweak ----------
CHUNK_SIZE = 300   # number of records to convert per batch (adjust to fit RAM)
# -----------------------------------------------------------


def get_processed_records():
    """
    Return a set of ecg_id integers that have already been processed
    by scanning train and val image directories and extracting the
    leading integer (ecg_id) from filenames.
    """
    processed = set()
    train_dir = os.path.join(config.IMAGES_CACHE_DIR, "train")
    val_dir = os.path.join(config.IMAGES_CACHE_DIR, "val")

    for d in (train_dir, val_dir):
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if not fname.lower().endswith(".png"):
                continue
            # Expect filenames like: <ecg_id>_w0_<CLASS>.png or <ecg_id>_<CLASS>.png
            parts = fname.split("_")
            if len(parts) == 0:
                continue
            # sometimes filenames may not be purely numeric; guard
            try:
                rec_id = int(parts[0])
                processed.add(rec_id)
            except ValueError:
                # if the first segment is not integer, skip
                continue

    return processed


def get_current_image_count():
    """Count existing images in train and val directories"""
    train_dir = os.path.join(config.IMAGES_CACHE_DIR, "train")
    val_dir = os.path.join(config.IMAGES_CACHE_DIR, "val")
    train_count = (
        len([f for f in os.listdir(train_dir) if f.lower().endswith(".png")])
        if os.path.exists(train_dir)
        else 0
    )
    val_count = (
        len([f for f in os.listdir(val_dir) if f.lower().endswith(".png")])
        if os.path.exists(val_dir)
        else 0
    )
    return train_count, val_count


def resume_generation(target_total=25000, windows_per_record=3, n_workers=6, chunk_size=CHUNK_SIZE):
    print("=" * 80)
    print("RESUME ECG IMAGE GENERATION (PARALLEL + KAGGLE LOGIC + RESUMABLE CHUNKS)")
    print("=" * 80)

    # status
    train_count, val_count = get_current_image_count()
    total_current = train_count + val_count
    train_frac = 100 * train_count / total_current if total_current > 0 else 0.0

    print(f"\n📊 Current Status:")
    print(f"   Training images: {train_count}")
    print(f"   Validation images: {val_count}")
    print(f"   Total: {total_current}")
    print(f"   Train/Val ratio: {train_count}/{val_count} ({train_frac:.1f}% train)")

    # compute how many images we need
    images_needed = target_total - total_current
    print(f"\n🎯 Target: {target_total} total images")
    print(f"   Need to add: {images_needed} more images")
    print(f"   Progress: {total_current}/{target_total} ({0 if target_total==0 else 100*total_current/target_total:.1f}%)\n")

    if images_needed <= 0:
        print("✅ Target already reached!")
        return

    # load the corrected dataset (patient-level split, correct labels)
    print("Loading dataset metadata...")
    train_df, val_df, label_encoder = process_and_split()
    # Ensure 'ecg_id' column exists
    if "ecg_id" not in train_df.columns:
        raise KeyError("train_df must contain 'ecg_id' column from PTB-XL metadata.")

    # find already processed ecg_ids from disk
    processed_records = get_processed_records()
    print(f"Found {len(processed_records)} records already processed (scanned train+val folders)\n")

    # concat train+val to obtain the full pool of records (preserve ecg_id)
    all_records = pd.concat([train_df, val_df], ignore_index=False)
    all_records = all_records.reset_index(drop=True)

    # Select records that are not yet processed based on ecg_id column
    unprocessed = all_records[~all_records["ecg_id"].isin(processed_records)].copy().reset_index(drop=True)
    print(f"Unprocessed records available: {len(unprocessed)}")

    if len(unprocessed) == 0:
        print("⚠️  No more unprocessed records available! All PTB-XL records may have been converted.")
        return

    # estimate how many records we need (each record -> windows_per_record images)
    records_to_process = min(len(unprocessed), (images_needed // windows_per_record) + 1)
    print(f"\n🔄 Planning to process up to {records_to_process} records (each -> ~{windows_per_record} images)")

    # Create output directories if missing
    train_img_dir = os.path.join(config.IMAGES_CACHE_DIR, "train")
    val_img_dir = os.path.join(config.IMAGES_CACHE_DIR, "val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # Determine targets for train/val folders (proportional to total target)
    target_train = int(target_total * 0.8)
    target_val = int(target_total * 0.2)
    train_needed = max(0, target_train - train_count)
    val_needed = max(0, target_val - val_count)

    print(f"\n📊 Split targets (global): Train target {target_train}, Val target {target_val}")
    print(f"   Need: Train {train_needed}, Val {val_needed}")

    # allocate how many records should go to train vs val from the to-process set
    if (train_needed + val_needed) > 0:
        train_ratio = train_needed / (train_needed + val_needed)
    else:
        train_ratio = 0.8  # default

    train_records_count = int(records_to_process * train_ratio)
    val_records_count = records_to_process - train_records_count
    print(f"   Allocating {train_records_count} records to train, {val_records_count} to val\n")

    unprocessed_train = unprocessed.iloc[:train_records_count].copy().reset_index(drop=True)
    unprocessed_val = unprocessed.iloc[train_records_count: train_records_count + val_records_count].copy().reset_index(drop=True)

    start_time = time.time()

    # helper to process a DataFrame of records in chunks
    def process_in_chunks(df_records, out_dir, description):
        total = len(df_records)
        added_images = 0
        if total == 0:
            print(f"⚠️  No new {description} records to process")
            return 0

        print("=" * 80)
        print(f"PROCESSING {total} NEW {description.upper()} RECORDS (chunk_size={chunk_size})")
        print("=" * 80)

        for i in range(0, total, chunk_size):
            batch = df_records.iloc[i: i + chunk_size].copy().reset_index(drop=True)
            print(f"\n→ Converting batch {i // chunk_size + 1} / {math.ceil(total / chunk_size)} "
                  f"(records {i} .. {i + len(batch) - 1}) ...")

            # call your conversion routine without changing its behaviour
            X_new, y_new, success_count, info = batch_convert_ecgs_parallel(
                batch,
                config.DATA_DIR,
                output_dir=out_dir,
                max_samples=len(batch),
                windows_per_record=windows_per_record,
                n_workers=n_workers
            )

            added_images += success_count * windows_per_record if success_count is not None else 0

            # best-effort cleanup
            gc.collect()
            # small pause to reduce disk pressure
            time.sleep(0.5)

            # update live counts
            cur_train, cur_val = get_current_image_count()
            print(f"   Current counts -> Train: {cur_train}, Val: {cur_val}, Total: {cur_train + cur_val}")

        return added_images

    # Process train and val sets with chunking
    train_added = process_in_chunks(unprocessed_train, train_img_dir, "train")
    val_added = process_in_chunks(unprocessed_val, val_img_dir, "val")

    # Final stats
    train_count_new, val_count_new = get_current_image_count()
    total_new = train_count_new + val_count_new
    total_elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("✅ RESUME COMPLETE")
    print("=" * 80)
    print(f"\n📊 Updated Status:")
    print(f"   Training images: {train_count_new} (added {train_count_new - train_count})")
    print(f"   Validation images: {val_count_new} (added {val_count_new - val_count})")
    print(f"   Total: {total_new} (added {total_new - total_current})")
    print(f"   Progress: {total_new}/{target_total} ({100 * total_new / target_total:.1f}%)")
    if total_elapsed > 0:
        print(f"   Rate: {(total_new - total_current) / total_elapsed:.2f} images/sec")
    print(f"\n⏱  Processing time: {total_elapsed / 60:.2f} minutes")

    if total_new >= target_total:
        print("\n🎉 TARGET REACHED! Ready for training!")
    else:
        print(f"\nStill need: {max(0, target_total - total_new)} more images.")
        print("Run this script again to continue.")
        print("Command: python -m src.generate_images_resume")


if __name__ == "__main__":
    # --------------------------------------------------------------------
    # configuration
    # --------------------------------------------------------------------
    TARGET_IMAGES = 25000
    WINDOWS_PER_RECORD = 3
    N_WORKERS = 6
    CHUNK = CHUNK_SIZE

    resume_generation(
        target_total=TARGET_IMAGES,
        windows_per_record=WINDOWS_PER_RECORD,
        n_workers=N_WORKERS,
        chunk_size=CHUNK,
    )

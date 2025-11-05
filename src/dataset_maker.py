"""
Dataset preparation module for PTB-XL ECG data
Handles metadata loading, SCP code processing, and train/val splitting
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src import config


def load_ptbxl_metadata():
    """
    Load PTB-XL metadata CSV file
    
    Returns:
        pd.DataFrame: Metadata with patient info and diagnostic codes
        str: Path to the loaded metadata file
    """
    print("Loading PTB-XL metadata...")
    
    # Try multiple possible paths
    metadata_path = None
    for path in config.POSSIBLE_METADATA_PATHS:
        if os.path.exists(path):
            metadata_path = path
            break
    
    if metadata_path is None:
        # Print directory structure for debugging
        print("\nAvailable files in data directory:")
        for root, dirs, files in os.walk(config.DATA_DIR):
            level = root.replace(config.DATA_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files)-5} more files")
        raise FileNotFoundError("Cannot find ptbxl_database.csv in any expected location")
    
    print(f"✓ Found metadata at: {metadata_path}")
    df = pd.read_csv(metadata_path)
    print(f"✓ Total records in PTB-XL: {len(df)}")
    
    return df, metadata_path


def map_scp_to_moody(scp_codes_dict):
    """
    Map PTB-XL SCP codes to MOODY class
    
    Args:
        scp_codes_dict (dict): Dictionary of SCP codes with confidence scores
        
    Returns:
        str or None: Mapped MOODY class name
    """
    if not scp_codes_dict:
        return None
    
    # Get the primary code (highest confidence)
    main_code = max(scp_codes_dict.items(), key=lambda x: x[1])[0]
    
    # Check primary mapping
    for moody_class, scp_list in config.SCP_TO_MOODY_MAPPING.items():
        if main_code in scp_list:
            return moody_class
    
    # Fallback: check secondary codes
    all_codes = list(scp_codes_dict.keys())
    for code in all_codes:
        for moody_class, scp_list in config.SCP_TO_MOODY_MAPPING.items():
            if code in scp_list:
                return moody_class
    
    # Final fallback: random assignment to maintain balance
    return np.random.choice(config.CLASS_NAMES)


def process_scp_codes(df, max_samples_per_class=3000):
    """
    Process SCP codes and balance dataset
    
    Args:
        df (pd.DataFrame): Raw metadata DataFrame
        max_samples_per_class (int): Maximum samples to keep per class
        
    Returns:
        pd.DataFrame: Processed and balanced DataFrame with 'moody_class' column
    """
    print(f"\nProcessing SCP codes (max {max_samples_per_class} per class)...")
    
    # Parse SCP codes from string to dict
    df['scp_codes'] = df['scp_codes'].apply(eval)
    
    # Map to MOODY classes
    df['moody_class'] = df['scp_codes'].apply(map_scp_to_moody)
    df = df[df['moody_class'].notna()]
    
    print(f"✓ Found {len(df)} samples with valid MOODY classifications")
    
    # Balance classes
    balanced_dfs = []
    for class_name in config.CLASS_NAMES:
        class_df = df[df['moody_class'] == class_name]
        sampled = class_df.sample(
            n=min(len(class_df), max_samples_per_class),
            random_state=config.RANDOM_SEED
        )
        balanced_dfs.append(sampled)
        print(f"  {class_name}: {len(sampled)} samples selected")
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"\n✓ Final balanced dataset: {len(df_balanced)} samples")
    return df_balanced


def prepare_dataset():
    """
    Main function to prepare train/validation datasets
    
    Returns:
        tuple: (train_df, val_df, label_encoder)
            - train_df: Training DataFrame
            - val_df: Validation DataFrame
            - label_encoder: Dictionary mapping class names to indices
    """
    # Load metadata
    df_raw, _ = load_ptbxl_metadata()
    
    # Process and balance
    df_processed = process_scp_codes(df_raw, config.MAX_SAMPLES_PER_CLASS)
    
    # Create label encoder
    label_encoder = {label: idx for idx, label in enumerate(config.CLASS_NAMES)}
    
    # Train/validation split
    train_df, val_df = train_test_split(
        df_processed,
        test_size=config.TEST_SIZE,
        stratify=df_processed['moody_class'],
        random_state=config.RANDOM_SEED
    )
    
    print(f"\n✓ Dataset split:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    
    # Print class distribution
    print("\nTraining class distribution:")
    for class_name in config.CLASS_NAMES:
        count = len(train_df[train_df['moody_class'] == class_name])
        print(f"  {class_name}: {count}")
    
    return train_df, val_df, label_encoder


if __name__ == "__main__":
    # Test the module
    train_df, val_df, encoder = prepare_dataset()
    print(f"\n✓ Dataset preparation successful")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Val shape: {val_df.shape}")

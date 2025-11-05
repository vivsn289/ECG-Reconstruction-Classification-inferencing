"""
Enhanced ECG to Image Conversion Module
Creates realistic ECG report-style images with proper waveform expansion
Optimized for PTB-XL dataset
With MULTI-WINDOW capability for data augmentation
WITH PARALLEL PROCESSING for 3-4x speed improvement
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import wfdb
import gc
from scipy import signal as scipy_signal
from scipy.ndimage import median_filter
from multiprocessing import Pool, cpu_count
from src import config
import time


def load_ecg_signal(record_path, target_length=5000):
    """
    Load ECG signal from WFDB format with proper handling
    
    Args:
        record_path (str): Path to WFDB record (without extension)
        target_length (int): Target number of samples per lead
        
    Returns:
        tuple: (signal array, sampling_frequency) or (None, None)
    """
    try:
        if record_path.endswith('.dat'):
            record_path = record_path[:-4]
        
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal
        fs = record.fs  # Sampling frequency
        
        # Standardize to 12 leads
        if signal.shape[1] >= 12:
            signal = signal[:, :12]
        elif signal.shape[1] < 12:
            leads_needed = 12 - signal.shape[1]
            padding = np.tile(signal[:, -1:], (1, leads_needed))
            signal = np.concatenate([signal, padding], axis=1)
        
        # Don't resize yet - keep original for better processing
        return signal, fs
        
    except Exception as e:
        return None, None


def extract_multiple_windows(ecg_signal, fs=100, num_windows=3):
    """
    Extract multiple non-overlapping windows from a single ECG recording
    
    Args:
        ecg_signal: (N, 12) signal array
        fs: Sampling frequency
        num_windows: Number of windows to extract (default 3 per record)
        
    Returns:
        List of signal windows
    """
    windows = []
    display_duration = 2.5  # seconds per window
    segment_samples = int(display_duration * fs)
    
    # Calculate how many non-overlapping windows we can extract
    max_possible_windows = len(ecg_signal) // segment_samples
    actual_num_windows = min(num_windows, max_possible_windows)
    
    if actual_num_windows == 0:
        return [ecg_signal]
    
    # Extract equally-spaced non-overlapping windows
    for i in range(actual_num_windows):
        start_idx = i * segment_samples
        end_idx = start_idx + segment_samples
        
        if end_idx <= len(ecg_signal):
            window = ecg_signal[start_idx:end_idx]
            
            # Check if window has meaningful data
            if np.std(window) > 1e-6:
                windows.append(window)
    
    # If we couldn't get enough windows, use sliding window approach
    if len(windows) < num_windows and len(ecg_signal) > segment_samples:
        stride = max(1, (len(ecg_signal) - segment_samples) // (num_windows - len(windows)))
        for i in range(num_windows - len(windows)):
            start_idx = i * stride
            end_idx = start_idx + segment_samples
            if end_idx <= len(ecg_signal):
                window = ecg_signal[start_idx:end_idx]
                if np.std(window) > 1e-6:
                    windows.append(window)
    
    return windows if windows else [ecg_signal]


def ecg_to_image_converter(ecg_signal, fs=100, output_path=None, return_array=True):
    """
    Convert 12-lead ECG to realistic report-style image
    
    Uses KAGGLE'S SIGNAL PROCESSING:
    - Variance-based segment selection
    - Median filter baseline removal
    - Percentile-based robust normalization
    
    Args:
        ecg_signal (np.ndarray): ECG signal array of shape (N, 12)
        fs (int): Sampling frequency in Hz
        output_path (str, optional): Path to save image
        return_array (bool): Whether to return array
        
    Returns:
        np.ndarray or None: Image array (IMG_SIZE, IMG_SIZE, 3)
    """
    if ecg_signal is None or len(ecg_signal) < 100:
        return None
    
    try:
        # Create high-quality figure
        fig = plt.figure(figsize=(24, 16), dpi=100)
        fig.patch.set_facecolor('white')
        
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Grid layout with padding
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35,
                             left=0.04, right=0.96, top=0.96, bottom=0.04)
        
        # Duration to display per lead (in seconds)
        display_duration = 2.5
        
        for lead_idx in range(12):
            ax = fig.add_subplot(gs[lead_idx // 4, lead_idx % 4])
            ax.set_facecolor('#FFF9F0')
            
            # Extract and clean signal for this lead
            signal = ecg_signal[:, lead_idx].copy()
            
            # Remove NaN and inf values
            signal = signal[np.isfinite(signal)]
            
            if len(signal) < 50:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=14, color='red')
                continue
            
            # =================================================================
            # KAGGLE'S SIGNAL PROCESSING LOGIC
            # =================================================================
            
            # 1. Select a good segment (middle portion, avoiding edges)
            segment_samples = int(display_duration * fs)
            
            # Find segment with highest variance (most interesting cardiac activity)
            if len(signal) > segment_samples * 3:
                # Split into chunks and find most active
                n_chunks = len(signal) // segment_samples
                max_var = -1
                best_start = len(signal) // 3
                
                for i in range(1, n_chunks - 1):
                    start = i * segment_samples
                    end = start + segment_samples
                    chunk_var = np.var(signal[start:end])
                    if chunk_var > max_var:
                        max_var = chunk_var
                        best_start = start
                
                signal_segment = signal[best_start:best_start + segment_samples]
            else:
                # Use middle portion
                start_idx = max(0, (len(signal) - segment_samples) // 2)
                signal_segment = signal[start_idx:start_idx + segment_samples]
            
            # 2. Apply high-pass filter to remove baseline wander
            if len(signal_segment) > 20:
                # Simple baseline removal using median filter
                baseline = median_filter(signal_segment, size=int(fs * 0.2))
                signal_segment = signal_segment - baseline
            
            # 3. Resample to target number of points for clean display
            target_points = 200  # Sweet spot for ECG visualization
            
            if len(signal_segment) != target_points:
                # Use scipy resample for high-quality resampling
                signal_display = scipy_signal.resample(signal_segment, target_points)
            else:
                signal_display = signal_segment
            
            # 4. Robust normalization
            # Use percentile-based normalization to handle outliers
            p1, p99 = np.percentile(signal_display, [1, 99])
            signal_range = p99 - p1
            
            if signal_range > 1e-6:
                # Normalize to [-1.5, 1.5] range
                signal_norm = 3.0 * (signal_display - p1) / signal_range - 1.5
                # Clip extreme outliers
                signal_norm = np.clip(signal_norm, -2.5, 2.5)
            else:
                # Flat signal - center it
                signal_norm = signal_display - np.median(signal_display)
            
            # 5. Create time axis
            time_axis = np.linspace(0, display_duration, len(signal_norm))
            
            # =================================================================
            # PLOTTING WITH ECG PAPER STYLE
            # =================================================================
            
            # ECG paper grid (standard red/pink color)
            ax.set_axisbelow(True)
            
            # Major grid lines (0.2 second intervals, 0.5 mV)
            ax.grid(True, which='major', color='#FF6B9D', alpha=0.5, 
                   linewidth=1.2, linestyle='-')
            
            # Minor grid lines (0.04 second intervals, 0.1 mV)  
            ax.grid(True, which='minor', color='#FFA5C5', alpha=0.3,
                   linewidth=0.6, linestyle='-')
            
            # Set grid locators
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            
            # Plot ECG trace - thick black line
            ax.plot(time_axis, signal_norm,
                   color='#000000',
                   linewidth=2.5,
                   antialiased=True,
                   solid_capstyle='round',
                   solid_joinstyle='round',
                   zorder=10)
            
            # Axis limits
            ax.set_xlim(0, display_duration)
            ax.set_ylim(-2.8, 2.8)
            
            # Remove tick labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
            
            # Border styling
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('#999999')
                spine.set_alpha(0.7)
            
            # Lead label
            ax.text(0.02, 0.98, lead_names[lead_idx],
                   transform=ax.transAxes,
                   fontsize=20,
                   color='#000000',
                   verticalalignment='top',
                   horizontalalignment='left',
                   fontweight='bold',
                   family='monospace',
                   bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='white',
                           edgecolor='#666666',
                           linewidth=1.5,
                           alpha=0.95))
        
        # Save image if path provided
        if output_path:
            plt.savefig(output_path,
                       dpi=100,
                       bbox_inches='tight',
                       pad_inches=0.1,
                       facecolor='white',
                       edgecolor='none')
            print(f"Saved ECG image to: {output_path}")
        
        if return_array:
            # Convert figure to array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = fig.canvas.get_width_height()
            img_array = img_array.reshape((height, width, 3))
            
            # Resize to target size
            img_resized = cv2.resize(img_array,
                                    (config.IMG_SIZE, config.IMG_SIZE),
                                    interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            plt.close(fig)
            gc.collect()
            
            return img_normalized
        
        plt.close(fig)
        gc.collect()
        return None
        
    except Exception as e:
        print(f"Error in ecg_to_image_converter: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        gc.collect()
        return None


def process_single_record(args):
    """
    Worker function for parallel processing
    Processes a single record and extracts multiple windows
    """
    row, data_dir, output_dir, windows_per_record, label_encoder = args
    
    filename_lr = row.get('filename_lr', '')
    possible_paths = [
        os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3', filename_lr),
        os.path.join(data_dir, filename_lr),
        os.path.join(data_dir, 'records100', filename_lr.replace('records100/', '')),
    ]
    
    results = []
    for record_path in possible_paths:
        if os.path.exists(record_path + '.dat'):
            signal, fs = load_ecg_signal(record_path)
            
            if signal is not None:
                windows = extract_multiple_windows(signal, fs, num_windows=windows_per_record)
                
                for window_idx, window_signal in enumerate(windows):
                    img_output_path = None
                    if output_dir:
                        img_filename = f"{row.name}_w{window_idx}_{row['moody_class']}.{config.IMG_FORMAT}"
                        img_output_path = os.path.join(output_dir, img_filename)
                    
                    img_array = ecg_to_image_converter(window_signal, fs, img_output_path, return_array=True)
                    
                    if img_array is not None:
                        label_idx = label_encoder[row['moody_class']]
                        label_onehot = np.zeros(config.NUM_CLASSES)
                        label_onehot[label_idx] = 1
                        results.append((img_array, label_onehot))
                
                return results
    
    return []


def batch_convert_ecgs(df, data_dir, output_dir=None, max_samples=None):
    """Batch convert ECG records to images"""
    images = []
    labels = []
    success_count = 0
    fail_count = 0
    
    label_encoder = {label: idx for idx, label in enumerate(config.CLASS_NAMES)}
    total = len(df) if max_samples is None else min(max_samples, len(df))
    
    print(f"Converting {total} ECG records to images...")
    
    for idx, row in df.iterrows():
        if max_samples and success_count >= max_samples:
            break
        
        filename_lr = row.get('filename_lr', '')
        filename_hr = row.get('filename_hr', '')
        
        possible_paths = [
            os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3', filename_lr),
            os.path.join(data_dir, filename_lr),
            os.path.join(data_dir, 'records100', filename_lr.replace('records100/', '')),
            os.path.join(data_dir, filename_hr),
        ]
        
        ecg_loaded = False
        for record_path in possible_paths:
            if os.path.exists(record_path + '.dat') and os.path.exists(record_path + '.hea'):
                signal, fs = load_ecg_signal(record_path)
                
                if signal is not None:
                    img_output_path = None
                    if output_dir:
                        img_filename = f"{row.name}_{row['moody_class']}.{config.IMG_FORMAT}"
                        img_output_path = os.path.join(output_dir, img_filename)
                    
                    img_array = ecg_to_image_converter(signal, fs, img_output_path, return_array=True)
                    
                    if img_array is not None:
                        images.append(img_array)
                        
                        label_idx = label_encoder[row['moody_class']]
                        label_onehot = np.zeros(config.NUM_CLASSES)
                        label_onehot[label_idx] = 1
                        labels.append(label_onehot)
                        
                        success_count += 1
                        ecg_loaded = True
                        
                        if success_count % 100 == 0:
                            print(f"  ✓ Converted {success_count}/{total} images...")
                        
                        break
        
        if not ecg_loaded:
            fail_count += 1
            if fail_count % 100 == 0:
                print(f"  ⚠ Failed: {fail_count} records")
    
    print(f"\n✓ Conversion complete: {success_count} success, {fail_count} failed")
    
    if success_count > 0:
        return np.array(images), np.array(labels), success_count, fail_count
    else:
        return np.array([]), np.array([]), success_count, fail_count


def batch_convert_ecgs_with_progress(df, data_dir, output_dir=None, max_samples=None):
    """Batch convert with detailed progress tracking"""
    images = []
    labels = []
    success_count = 0
    fail_count = 0
    
    label_encoder = {label: idx for idx, label in enumerate(config.CLASS_NAMES)}
    total = len(df) if max_samples is None else min(max_samples, len(df))
    
    print(f"\n{'='*80}")
    print(f"Converting {total} ECG records...")
    print(f"{'='*80}")
    
    start_time = time.time()
    last_print = start_time
    
    for idx, row in df.iterrows():
        if max_samples and success_count >= max_samples:
            break
        
        filename_lr = row.get('filename_lr', '')
        filename_hr = row.get('filename_hr', '')
        
        possible_paths = [
            os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3', filename_lr),
            os.path.join(data_dir, filename_lr),
            os.path.join(data_dir, 'records100', filename_lr.replace('records100/', '')),
            os.path.join(data_dir, filename_hr),
        ]
        
        ecg_loaded = False
        for record_path in possible_paths:
            if os.path.exists(record_path + '.dat'):
                signal, fs = load_ecg_signal(record_path)
                
                if signal is not None:
                    img_output_path = None
                    if output_dir:
                        img_filename = f"{row.name}_{row['moody_class']}.{config.IMG_FORMAT}"
                        img_output_path = os.path.join(output_dir, img_filename)
                    
                    img_array = ecg_to_image_converter(signal, fs, img_output_path, return_array=True)
                    
                    if img_array is not None:
                        images.append(img_array)
                        
                        label_idx = label_encoder[row['moody_class']]
                        label_onehot = np.zeros(config.NUM_CLASSES)
                        label_onehot[label_idx] = 1
                        labels.append(label_onehot)
                        
                        success_count += 1
                        ecg_loaded = True
                        
                        current = time.time()
                        if success_count % 50 == 0 or (current - last_print) > 5:
                            elapsed = current - start_time
                            rate = success_count / elapsed if elapsed > 0 else 0
                            eta = (total - success_count) / rate if rate > 0 else 0
                            pct = (success_count / total) * 100
                            
                            print(f"  [{success_count:5d}/{total}] ({pct:5.1f}%) "
                                  f"| {rate:4.1f} img/s | ETA: {eta/60:4.1f} min | Failed: {fail_count}")
                            last_print = current
                        
                        break
        
        if not ecg_loaded:
            fail_count += 1
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✓ Complete: {success_count} success, {fail_count} failed")
    print(f"  Time: {total_time/60:.2f} min | Rate: {success_count/total_time:.2f} img/s")
    print(f"{'='*80}\n")
    
    if success_count > 0:
        return np.array(images), np.array(labels), success_count, fail_count
    else:
        return np.array([]), np.array([]), success_count, fail_count


def batch_convert_ecgs_multi_window(df, data_dir, output_dir=None, max_samples=None, windows_per_record=3):
    """
    Batch convert ECG records to MULTIPLE images per record
    
    Args:
        df: DataFrame with ECG metadata
        data_dir: Base data directory
        output_dir: Where to save images
        max_samples: Max records to process (will create max_samples * windows_per_record images)
        windows_per_record: How many windows to extract per ECG (default 3)
    """
    images = []
    labels = []
    success_count = 0
    fail_count = 0
    total_windows = 0
    
    label_encoder = {label: idx for idx, label in enumerate(config.CLASS_NAMES)}
    total = len(df) if max_samples is None else min(max_samples, len(df))
    
    print(f"\n{'='*80}")
    print(f"Converting {total} ECG records ({windows_per_record} windows each)...")
    print(f"Expected total images: ~{total * windows_per_record}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    last_print = start_time
    
    for idx, row in df.iterrows():
        if max_samples and success_count >= max_samples:
            break
        
        filename_lr = row.get('filename_lr', '')
        filename_hr = row.get('filename_hr', '')
        
        possible_paths = [
            os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3', filename_lr),
            os.path.join(data_dir, filename_lr),
            os.path.join(data_dir, 'records100', filename_lr.replace('records100/', '')),
            os.path.join(data_dir, filename_hr),
        ]
        
        ecg_loaded = False
        for record_path in possible_paths:
            if os.path.exists(record_path + '.dat'):
                signal, fs = load_ecg_signal(record_path)
                
                if signal is not None:
                    # Extract multiple windows from this single record
                    windows = extract_multiple_windows(signal, fs, num_windows=windows_per_record)
                    
                    for window_idx, window_signal in enumerate(windows):
                        img_output_path = None
                        if output_dir:
                            img_filename = f"{row.name}_w{window_idx}_{row['moody_class']}.{config.IMG_FORMAT}"
                            img_output_path = os.path.join(output_dir, img_filename)
                        
                        img_array = ecg_to_image_converter(window_signal, fs, img_output_path, return_array=True)
                        
                        if img_array is not None:
                            images.append(img_array)
                            
                            label_idx = label_encoder[row['moody_class']]
                            label_onehot = np.zeros(config.NUM_CLASSES)
                            label_onehot[label_idx] = 1
                            labels.append(label_onehot)
                            
                            total_windows += 1
                    
                    success_count += 1
                    ecg_loaded = True
                    
                    current = time.time()
                    if success_count % 50 == 0 or (current - last_print) > 5:
                        elapsed = current - start_time
                        rate = success_count / elapsed if elapsed > 0 else 0
                        eta = (total - success_count) / rate if rate > 0 else 0
                        pct = (success_count / total) * 100
                        
                        print(f"  [{success_count:5d}/{total}] ({pct:5.1f}%) "
                              f"| {total_windows} images | ETA: {eta/60:4.1f} min")
                        last_print = current
                    
                    break
        
        if not ecg_loaded:
            fail_count += 1
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✓ Conversion complete:")
    print(f"  Records processed: {success_count}")
    print(f"  Total images created: {total_windows} ({total_windows/success_count:.1f} per record)")
    print(f"  Failed: {fail_count}")
    print(f"  Time: {total_time/60:.2f} min")
    print(f"{'='*80}\n")
    
    if len(images) > 0:
        return np.array(images), np.array(labels), success_count, fail_count
    else:
        return np.array([]), np.array([]), success_count, fail_count


def batch_convert_ecgs_parallel(df, data_dir, output_dir=None, max_samples=None, windows_per_record=3, n_workers=None):
    """
    PARALLEL batch conversion using multiple CPU cores
    Combines Kaggle's signal processing with multi-window extraction
    3-4x faster than sequential processing
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)
    
    label_encoder = {label: idx for idx, label in enumerate(config.CLASS_NAMES)}
    
    total = len(df) if max_samples is None else min(max_samples, len(df))
    
    print(f"\n{'='*80}")
    print(f"PARALLEL Processing ({n_workers} workers)")
    print(f"Converting {total} records → ~{total * windows_per_record} images")
    print(f"{'='*80}\n")
    
    args_list = [(row, data_dir, output_dir, windows_per_record, label_encoder)
                 for idx, row in df.iterrows()]
    
    if max_samples:
        args_list = args_list[:max_samples]
    
    images = []
    labels = []
    success_count = 0
    start_time = time.time()
    
    with Pool(processes=n_workers) as pool:
        for i, results in enumerate(pool.imap_unordered(process_single_record, args_list)):
            if results:
                for img_array, label_onehot in results:
                    images.append(img_array)
                    labels.append(label_onehot)
                success_count += 1
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(args_list) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:5d}/{len(args_list)}] → {len(images)} images | "
                      f"Rate: {rate:.1f} rec/s | ETA: {eta/60:.1f} min")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✓ Complete: {len(images)} images in {total_time/60:.1f} minutes")
    print(f"  Rate: {len(images)/total_time:.1f} images/second")
    print(f"{'='*80}\n")
    
    if len(images) > 0:
        return np.array(images), np.array(labels), success_count, 0
    else:
        return np.array([]), np.array([]), 0, 0


if __name__ == "__main__":
    print("="*80)
    print("ECG TO IMAGE CONVERTER - TEST")
    print("="*80)
    
    test_path = os.path.join(config.DATA_DIR, 
                            'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3',
                            'records100/00000/00001_lr')
    
    print(f"\nTesting with: {test_path}")
    
    signal, fs = load_ecg_signal(test_path)
    
    if signal is not None:
        print(f"Signal loaded: shape={signal.shape}, fs={fs} Hz")
        print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
        
        img = ecg_to_image_converter(signal, fs, 'test_ecg_ultimate.png', return_array=True)
        
        if img is not None:
            print(f"\n✅ SUCCESS! Image generated")
            print(f"   Shape: {img.shape}")
            print(f"   Range: [{img.min():.3f}, {img.max():.3f}]")
            print(f"   Saved to: test_ecg_ultimate.png")
            print(f"\n   Check the image - should show clear ECG waveforms with proper expansion!")
        else:
            print("\n❌ Failed to generate image")
    else:
        print("\n❌ Failed to load signal")
    
    print("\n" + "="*80)

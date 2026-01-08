# train.py — stabilized + mixup + reduce-on-plateau + val_macro_f1 monitoring
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import pandas as pd

from src import config
from src.model_architecture import create_model

# --- Mixup helper (batch-wise) ---
def sample_beta(alpha):
    # sample Beta(alpha, alpha) via two Gammas
    a = tf.random.gamma(shape=[], alpha=alpha)
    b = tf.random.gamma(shape=[], alpha=alpha)
    return a / (a + b)

def mixup_batch(images, labels, alpha):
    """images: [B,H,W,C], labels: [B, num_classes]"""
    batch_size = tf.shape(images)[0]
    # shuffle indices
    idx = tf.random.shuffle(tf.range(batch_size))
    x2 = tf.gather(images, idx)
    y2 = tf.gather(labels, idx)
    lam = sample_beta(alpha)
    lam = tf.cast(lam, tf.float32)
    lam_x = tf.reshape(lam, [1, 1, 1, 1])
    lam_y = tf.reshape(lam, [1, 1])
    mixed_x = lam_x * images + (1.0 - lam_x) * x2
    mixed_y = lam_y * labels + (1.0 - lam_y) * y2
    return mixed_x, mixed_y

# --- EMA & SWA callbacks (same as previously robust) ---
class EMACallback(callbacks.Callback):
    def __init__(self, decay=0.999, verbose=1):
        super().__init__()
        self.decay = float(decay)
        self.verbose = verbose
        self.shadow_weights = None
        self._backup_weights = None

    def on_train_begin(self, logs=None):
        self.shadow_weights = [w.copy() for w in self.model.get_weights()]
        setattr(self.model, "_ema_weights", self.shadow_weights)
        if self.verbose:
            print(f"\n✓ EMA initialized (decay={self.decay})")

    def on_train_batch_end(self, batch, logs=None):
        current = self.model.get_weights()
        for i in range(len(self.shadow_weights)):
            self.shadow_weights[i] = self.decay * self.shadow_weights[i] + (1.0 - self.decay) * current[i]
        setattr(self.model, "_ema_weights", self.shadow_weights)

    def on_test_begin(self, logs=None):
        if self.shadow_weights is not None:
            self._backup_weights = [w.copy() for w in self.model.get_weights()]
            try:
                self.model.set_weights(self.shadow_weights)
            except Exception:
                self.model.set_weights([np.array(w) for w in self.shadow_weights])
            if self.verbose:
                print("\n[EMA] Swapped in EMA weights for validation")

    def on_test_end(self, logs=None):
        if self._backup_weights is not None:
            try:
                self.model.set_weights(self._backup_weights)
            except Exception:
                self.model.set_weights([np.array(w) for w in self._backup_weights])
            if self.verbose:
                print("[EMA] Restored weights after validation")
            self._backup_weights = None

    def on_train_end(self, logs=None):
        if self.shadow_weights is not None:
            try:
                self.model.set_weights(self.shadow_weights)
            except Exception:
                self.model.set_weights([np.array(w) for w in self.shadow_weights])
            if self.verbose:
                print("\n✓ Applied EMA weights for final model")
            setattr(self.model, "_ema_weights", self.shadow_weights)

class SWACallback(callbacks.Callback):
    def __init__(self, swa_start_epoch=30, swa_freq=1, verbose=1):
        super().__init__()
        self.swa_start = int(swa_start_epoch)
        self.swa_freq = int(swa_freq)
        self.verbose = verbose
        self.swa_weights = None
        self.swa_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            curr = [w.copy() for w in self.model.get_weights()]
            if self.swa_weights is None:
                self.swa_weights = curr
                self.swa_count = 1
            else:
                for i in range(len(self.swa_weights)):
                    self.swa_weights[i] = (self.swa_weights[i] * self.swa_count + curr[i]) / (self.swa_count + 1)
                self.swa_count += 1
            if self.verbose and (self.swa_count % 5 == 0 or self.swa_count == 1):
                print(f"  [SWA] Averaged {self.swa_count} checkpoints (epoch {epoch+1})")

    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            try:
                self.model.set_weights(self.swa_weights)
            except Exception:
                self.model.set_weights([np.array(w) for w in self.swa_weights])
            if self.verbose:
                print(f"\n✓ Applied SWA averaged weights (count={self.swa_count})")

class PerClassMetrics(callbacks.Callback):
    def __init__(self, validation_data, class_names, log_path, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.log_path = log_path
        self.verbose = verbose
        self.history = {'epoch': [], 'macro_f1': []}
        for cn in class_names:
            self.history[f'{cn}_acc'] = []

    def on_epoch_end(self, epoch, logs=None):
        use_ema = hasattr(self.model, "_ema_weights") and self.model._ema_weights is not None
        backup = None
        if use_ema:
            backup = [w.copy() for w in self.model.get_weights()]
            try:
                self.model.set_weights(self.model._ema_weights)
            except Exception:
                self.model.set_weights([np.array(w) for w in self.model._ema_weights])

        y_pred = []
        y_true = []
        for imgs, labels in self.validation_data:
            preds = self.model.predict(imgs, verbose=0)
            y_pred.append(preds)
            y_true.append(labels.numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)

        per_class_acc = []
        for i, cn in enumerate(self.class_names):
            mask = (true_classes == i)
            acc = float((pred_classes[mask] == i).mean()) if mask.sum() > 0 else 0.0
            per_class_acc.append(acc)
            self.history[f'{cn}_acc'].append(acc)

        macro_f1 = float(f1_score(true_classes, pred_classes, average='macro'))
        self.history['epoch'].append(epoch + 1)
        self.history['macro_f1'].append(macro_f1)
        if self.verbose:
            print("\n  Per-class acc: " + " ".join([f"{cn}={a:.3f}" for cn, a in zip(self.class_names, per_class_acc)]) + f" | Macro F1={macro_f1:.4f}")
        if logs is not None:
            logs['val_macro_f1'] = macro_f1

        if use_ema and backup is not None:
            try:
                self.model.set_weights(backup)
            except Exception:
                self.model.set_weights([np.array(w) for w in backup])

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_path, index=False)
        if self.verbose:
            print(f"\n✓ Per-class metrics saved to {self.log_path}")

# --- Data loading (same PNG loader) ---
def create_dataset_from_png(image_dir, batch_size, is_training=True, mixup_alpha=None):
    image_files = []
    labels = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith('.png'):
            class_name = fname.split('_')[-1].replace('.png', '')
            if class_name in config.CLASS_NAMES:
                image_files.append(os.path.join(image_dir, fname))
                labels.append(config.CLASS_NAMES.index(class_name))

    image_files = np.array(image_files)
    labels = np.array(labels, dtype=np.int32)

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE], method='lanczos3')
        img = tf.ensure_shape(img, [config.IMG_SIZE, config.IMG_SIZE, 3])
        label_oh = tf.one_hot(label, config.NUM_CLASSES)
        return img, tf.cast(label_oh, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(2000, len(image_files)), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(lambda p, l: tf.py_function(lambda pp, ll: load_image(pp, ll), [p, l], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Apply lightweight augmentations for ECG
    if is_training:
        def augment(img, lab):
            if config.AUGMENTATION_INTENSITY != 'none':
                if config.AUGMENT_HORIZONTAL_FLIP and tf.random.uniform([]) < 0.5:
                    img = tf.image.flip_left_right(img)
                img = tf.image.random_contrast(img, 1.0 - config.AUGMENT_CONTRAST_RANGE, 1.0 + config.AUGMENT_CONTRAST_RANGE)
                img = tf.image.random_brightness(img, config.AUGMENT_BRIGHTNESS_DELTA)
                noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=config.AUGMENT_NOISE_STDDEV)
                img = tf.clip_by_value(img + noise, 0.0, 1.0)
            return img, lab
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)

    # Mixup applied on batches (if requested)
    if is_training and mixup_alpha is not None and mixup_alpha > 0.0:
        def mixup_map(images, labels):
            mixed_x, mixed_y = mixup_batch(images, labels, mixup_alpha)
            return mixed_x, mixed_y
        dataset = dataset.map(lambda x, y: tf.py_function(lambda a, b: mixup_map(a, b),
                                                          inp=[x, y],
                                                          Tout=(tf.float32, tf.float32)),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- Utility: class weights ---
def calculate_class_weights(image_dir):
    labels = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith('.png'):
            class_name = fname.split('_')[-1].replace('.png', '')
            if class_name in config.CLASS_NAMES:
                labels.append(config.CLASS_NAMES.index(class_name))
    labels = np.array(labels)
    cls_w = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = {i: float(cls_w[i]) for i in range(len(cls_w))}
    return class_weight_dict

# --- Training function ---
def train_model():
    print("Starting training — Mixup + EMA + SWA + ReduceLROnPlateau")
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    train_dir = os.path.join(config.IMAGES_CACHE_DIR, 'train')
    val_dir = os.path.join(config.IMAGES_CACHE_DIR, 'val')
    batch_size = config.BATCH_SIZE

    train_ds = create_dataset_from_png(train_dir, batch_size, is_training=True, mixup_alpha=config.MIXUP_ALPHA)
    val_ds = create_dataset_from_png(val_dir, batch_size, is_training=False, mixup_alpha=None)

    class_weights = calculate_class_weights(train_dir)
    model = create_model(input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3), num_classes=config.NUM_CLASSES)
    print("Model params:", model.count_params())

    # optimizer and compile
    try:
        opt = keras.optimizers.AdamW(learning_rate=config.BASE_LR, weight_decay=config.WEIGHT_DECAY, clipnorm=1.0)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=config.BASE_LR, clipnorm=1.0)
        print("Using Adam fallback (no AdamW)")

    loss_fn = lambda y_true, y_pred: keras.losses.categorical_crossentropy(
        y_true * (1.0 - config.LABEL_SMOOTHING) + config.LABEL_SMOOTHING / tf.cast(config.NUM_CLASSES, tf.float32),
        y_pred)

    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    # Callbacks
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.MODELS_DIR, 'best_model.h5')

    ema_cb = EMACallback(decay=config.EMA_DECAY, verbose=1)
    swa_cb = SWACallback(swa_start_epoch=config.SWA_START_EPOCH, swa_freq=config.SWA_FREQ, verbose=1)
    perclass_cb = PerClassMetrics(validation_data=val_ds, class_names=config.CLASS_NAMES,
                                  log_path=os.path.join(config.LOGS_DIR, 'per_class_metrics.csv'), verbose=1)

    reduce_on_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=config.MIN_LR, verbose=1)

    model_ckpt = callbacks.ModelCheckpoint(checkpoint_path, monitor=config.CHECKPOINT_MONITOR,
                                           mode=config.CHECKPOINT_MODE, save_best_only=True, verbose=1)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=config.EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True, verbose=1)

    lr_schedule = callbacks.LearningRateScheduler(
        lambda epoch, lr: (config.BASE_LR * (epoch + 1) / config.WARMUP_EPOCHS) if epoch < config.WARMUP_EPOCHS
        else float(config.MIN_LR + (config.BASE_LR - config.MIN_LR) * 0.5 * (1 + math.cos(math.pi * (epoch - config.WARMUP_EPOCHS) / max(1, (config.EPOCHS - config.WARMUP_EPOCHS))))),
        verbose=0)

    csvlogger = callbacks.CSVLogger(os.path.join(config.LOGS_DIR, 'training_log.csv'))

    callbacks_list = [model_ckpt, early_stop, lr_schedule, reduce_on_plateau, ema_cb, swa_cb, perclass_cb, csvlogger]

    history = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS, callbacks=callbacks_list,
                        class_weight=class_weights, verbose=1)

    # Evaluate + save
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    # predictions
    y_pred = []
    y_true = []
    for img, lab in val_ds:
        preds = model.predict(img, verbose=0)
        y_pred.append(preds)
        y_true.append(lab.numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    val_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    print("Final validation: loss", val_loss, "acc", val_acc, "macro_f1", val_f1)

    # Save model & plots (omitted for brevity; similar to your existing code)
    model.save(os.path.join(config.MODELS_DIR, 'final_model.h5'))
    print("Saved final model.")

if __name__ == "__main__":
    train_model()

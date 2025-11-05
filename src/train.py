"""
Main training script for ECG classification
With GPU optimization and proper error handling
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import gc

from src import config
from src.model_architecture import create_model


def setup_gpu():
    """Configure GPU with proper memory management"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0].name}")
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set visible GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Print GPU info
            print(f"   Device name: {gpus[0].name}")
            print(f"   Memory growth enabled")
            
            return True
        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
            return False
    else:
        print("⚠️  No GPU detected, using CPU")
        print("   Training will be slower")
        return False


def load_preprocessed_data():
    """Load pre-generated images from disk"""
    print("Loading pre-generated images from disk...")
    
    train_path = os.path.join(config.IMAGES_CACHE_DIR, 'X_train.npy')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"\n❌ Pre-generated images not found!\n"
            f"Please run: python -m src.generate_images\n"
            f"Expected location: {config.IMAGES_CACHE_DIR}"
        )
    
    X_train = np.load(os.path.join(config.IMAGES_CACHE_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(config.IMAGES_CACHE_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(config.IMAGES_CACHE_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(config.IMAGES_CACHE_DIR, 'y_val.npy'))
    label_encoder = np.load(os.path.join(config.IMAGES_CACHE_DIR, 'label_encoder.npy'),
                           allow_pickle=True).item()
    
    print(f"✓ Loaded training data: {X_train.shape}")
    print(f"✓ Loaded validation data: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val, label_encoder


def create_data_pipeline(X, y, is_training=True):
    """Create optimized TensorFlow dataset pipeline"""
    if is_training:
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(config.AUGMENT_ROTATION_RANGE),
            keras.layers.RandomZoom(config.AUGMENT_ZOOM_RANGE),
            keras.layers.RandomContrast(config.AUGMENT_CONTRAST_RANGE),
        ])
        
        def preprocess(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.clip_by_value(image, 0.0, 1.0)
            image = data_augmentation(image)
            return image, label
    else:
        def preprocess(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(1000, len(X)), seed=config.RANDOM_SEED)
    
    dataset = (dataset
              .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(config.BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))
    
    return dataset


def train_model():
    """Main training function"""
    print("="*80)
    print("ECG CLASSIFICATION TRAINING")
    print("="*80)
    
    # Setup
    has_gpu = setup_gpu()
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    
    # Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING PRE-GENERATED IMAGES")
    print("="*80)
    
    X_train, y_train, X_val, y_val, label_encoder = load_preprocessed_data()
    
    # Preprocess
    print("\n" + "="*80)
    print("STEP 2: CREATING DATA PIPELINES")
    print("="*80)
    
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    
    print("\nCreating TensorFlow datasets...")
    train_dataset = create_data_pipeline(X_train, y_train, is_training=True)
    val_dataset = create_data_pipeline(X_val, y_val, is_training=False)
    
    print(f"✓ Pipelines ready")
    print(f"  Training batches: {len(train_dataset)}")
    print(f"  Validation batches: {len(val_dataset)}")
    
    # Build model
    print("\n" + "="*80)
    print("STEP 3: BUILDING MODEL")
    print("="*80)
    
    with tf.device('/GPU:0' if has_gpu else '/CPU:0'):
        model = create_model(
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
            num_classes=config.NUM_CLASSES
        )
    
    print(f"✓ Model architecture:")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Device: {'GPU' if has_gpu else 'CPU'}")
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model compiled")
    
    # Callbacks
    checkpoint_path = os.path.join(config.MODELS_DIR, 'best_model.h5')
    
    callbacks_list = [
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.CSVLogger(
            os.path.join(config.LOGS_DIR, 'training_log.csv')
        )
    ]
    
    # Train
    print("\n" + "="*80)
    print("STEP 4: TRAINING MODEL")
    print("="*80)
    
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("STEP 5: EVALUATING MODEL")
    print("="*80)
    
    val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    
    print("\nGenerating predictions...")
    y_pred = model.predict(val_dataset, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    val_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    
    print(f"\n✓ Final Results:")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation F1 Score: {val_f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=config.CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    print(f"✓ Confusion matrix saved")
    
    # Training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'training_history.png'), dpi=150)
    print(f"✓ Training history saved")
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    train_model()

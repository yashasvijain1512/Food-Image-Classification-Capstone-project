"""
Training script for Food-101 classification with baseline CNN and EfficientNetB0.

Workflow:
1. Train both models on 20 selected food classes
2. Evaluate and compare performance
3. Fine-tune both models on all 101 classes
4. Save best performing models

Usage:
    python train_models.py --mode baseline --num-classes 20 --epochs 10
    python train_models.py --mode transfer --num-classes 20 --epochs 15
    python train_models.py --mode all --num-classes 101 --epochs 20
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessDataset import create_data_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 20 selected food classes for initial training
SELECTED_20_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beignets',
    'bread_pudding', 'caesar_salad', 'carrot_cake', 'cheesecake', 'chocolate_cake',
    'chicken_curry', 'donuts', 'french_fries', 'grilled_cheese_sandwich',
    'hamburger', 'hot_dog', 'ice_cream', 'pizza', 'sushi', 'tacos',
]


def get_class_names() -> List[str]:
    """Get all 101 Food-101 class names."""
    builder = tfds.builder('food101')
    builder.download_and_prepare()
    return builder.info.features['label'].names


def filter_classes(ds: tf.data.Dataset,
                   class_names: List[str],
                   selected_classes: Optional[List[str]] = None) -> tf.data.Dataset:
    """
    Filter dataset to only include selected classes.
    
    Args:
        ds: Input dataset
        class_names: All class names
        selected_classes: List of class names to include. If None, uses all.
    
    Returns:
        Filtered dataset with remapped labels
    """
    if selected_classes is None:
        return ds
    
    # Create mapping from old labels to new labels
    old_to_new = {}
    new_label = 0
    for old_label, class_name in enumerate(class_names):
        if class_name in selected_classes:
            old_to_new[old_label] = new_label
            new_label += 1
    
    # Filter and remap
    def filter_and_remap(example):
        old_label = int(example['label'])
        if old_label in old_to_new:
            example['label'] = old_to_new[old_label]
            return True
        return False
    
    def remap_label(image, label):
        return image, tf.cast(old_to_new[int(label)], tf.int32)
    
    ds = ds.filter(lambda x: tf.py_function(
        lambda ex: old_to_new.get(int(ex['label'].numpy()), -1) != -1,
        [x], tf.bool
    ))
    
    # Simpler approach: filter in Python
    filtered_examples = []
    
    def filter_fn(example):
        label = int(example['label'])
        return label in old_to_new
    
    # Use dataset operations for filtering
    ds_filtered = ds.filter(lambda ex: tf.reduce_any(
        tf.equal(ex['label'], tf.constant([list(old_to_new.keys())], dtype=tf.int64))
    ))
    
    # Actually, let's do this more efficiently
    class_indices = [class_names.index(c) for c in selected_classes]
    
    def keep_class(example):
        label = example['label']
        return tf.reduce_any(tf.equal(label, tf.constant(class_indices, dtype=label.dtype)))
    
    def remap_label(image, label):
        new_idx = tf.where(tf.equal(label, tf.constant(class_indices, dtype=label.dtype)))[0][0]
        return image, new_idx
    
    ds = ds.filter(keep_class)
    ds = ds.map(lambda ex: remap_label(
        tf.cast(ex['image'], tf.float32),
        ex['label']
    ), num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds


def build_baseline_cnn(num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """
    Build a baseline CNN model.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 4
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Global Average Pooling
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return model


def build_efficientnet_transfer(num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """
    Build EfficientNetB0 transfer learning model.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Preprocessing layer
        tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),
        
        # Base model
        base_model,
        
        # Custom head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return model


def get_filtered_pipeline(split: str,
                          batch_size: int = 32,
                          img_size: int = 224,
                          selected_classes: Optional[List[str]] = None) -> Tuple[tf.data.Dataset, Dict]:
    """
    Get a filtered data pipeline for specific classes.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        batch_size: Batch size
        img_size: Image size
        selected_classes: List of class names to include
    
    Returns:
        Tuple of (dataset, info)
    """
    ds, info = tfds.load('food101', split=split, as_supervised=False, with_info=True)
    class_names = info.features['label'].names
    
    if selected_classes is not None:
        num_classes = len(selected_classes)
        # Create mapping from old class indices to new class indices
        class_indices = [class_names.index(c) for c in selected_classes]
        old_to_new_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
        
        def keep_class(example):
            label = example['label']
            # Check if label is in our selected indices
            return tf.reduce_any(tf.equal(label, tf.constant(class_indices, dtype=label.dtype)))
        
        def remap_and_preprocess(image, label):
            # Convert to numpy for remapping
            label_np = label.numpy()
            new_label = old_to_new_dict[int(label_np)]
            
            # Preprocess image: just normalize, resize will be done in TF
            image_np = image.numpy().astype(np.float32)
            image_norm = image_np / 127.5 - 1.0
            
            return image_norm, np.int64(new_label)
        
        def add_resize(image, label):
            # Resize image after py_function
            image = tf.image.resize(image, (img_size, img_size))
            return image, label
        
        def py_remap_and_preprocess(image, label):
            outputs = tf.py_function(
                remap_and_preprocess,
                [image, label],
                [tf.float32, tf.int64]
            )
            # Set shape for the output tensors
            outputs[0].set_shape((None, None, 3))  # image shape
            outputs[1].set_shape(())  # label shape (scalar)
            return outputs[0], outputs[1]
        
        # Extract image and label, then apply preprocessing
        ds = ds.map(lambda ex: (ex['image'], ex['label']), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.filter(lambda img, lbl: tf.reduce_any(tf.equal(lbl, tf.constant(class_indices, dtype=lbl.dtype))))
        ds = ds.map(py_remap_and_preprocess, num_parallel_calls=1)
        ds = ds.map(add_resize, num_parallel_calls=tf.data.AUTOTUNE)
        
        info_dict = {
            'num_examples': len(selected_classes) * 750,  # Approximate: ~750 imgs per class
            'num_classes': num_classes,
            'class_names': selected_classes,
        }
    else:
        def preprocess(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (img_size, img_size))
            image = image / 127.5 - 1.0
            return image, label
        
        ds = ds.map(lambda ex: (ex['image'], ex['label']), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        info_dict = {
            'num_examples': info.splits[split].num_examples,
            'num_classes': info.features['label'].num_classes,
            'class_names': list(class_names),
        }
    
    # Shuffle for training
    if split == 'train':
        shuffle_buffer = min(info_dict['num_examples'], 5000)
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds, info_dict


def save_class_names(class_names: List[str], path: str) -> None:
    """Save one class name per line to a text file."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f'{name}\n')


def train_model(model: tf.keras.Model,
                train_ds: tf.data.Dataset,
                val_ds: tf.data.Dataset,
                model_name: str,
                num_classes: int,
                class_names: List[str],
                epochs: int = 10,
                output_dir: str = 'models') -> Dict:
    """
    Train a model on the given dataset.
    
    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        model_name: Name for saving (e.g., 'baseline_20' or 'efficientnet_101')
        num_classes: Number of classes
        epochs: Number of epochs
        output_dir: Directory to save models
    
    Returns:
        Training history and metrics
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
        ),  
    ]
    
    logger.info(f'Training {model_name} for {num_classes} classes, {epochs} epochs')
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save class names for the model artifacts
    save_class_names(class_names, os.path.join(output_dir, f'{model_name}_best_class_names.txt'))
    model_path = os.path.join(output_dir, f'{model_name}_final.h5')
    model.save(model_path)
    save_class_names(class_names, os.path.join(output_dir, f'{model_name}_final_class_names.txt'))
    logger.info(f'Saved model to {model_path}')
    
    return {
        'model_name': model_name,
        'num_classes': num_classes,
        'epochs': epochs,
        'history': {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy'],
        },
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
        'best_val_acc': float(max(history.history['val_accuracy'])),
    }


def finetune_transfer_model(model: tf.keras.Model,
                            train_ds: tf.data.Dataset,
                            val_ds: tf.data.Dataset,
                            model_name: str,
                            num_classes: int,
                            class_names: List[str],
                            epochs: int = 10,
                            output_dir: str = 'models') -> Dict:
    """
    Fine-tune a transfer learning model (unfreeze base layers).
    
    Args:
        model: Pre-trained Keras model
        train_ds: Training dataset
        val_ds: Validation dataset
        model_name: Name for saving
        num_classes: Number of classes
        class_names: List of class names
        epochs: Number of epochs for fine-tuning
        output_dir: Directory to save models
    
    Returns:
        Training history and metrics
    """
    # Unfreeze base model for fine-tuning
    for layer in model.layers:
        if hasattr(layer, 'trainable'):
            layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, f'{model_name}_finetuned_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
        ),
    ]
    
    logger.info(f'Fine-tuning {model_name} for {num_classes} classes, {epochs} epochs')
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    save_class_names(class_names, os.path.join(output_dir, f'{model_name}_finetuned_best_class_names.txt'))
    model_path = os.path.join(output_dir, f'{model_name}_finetuned_final.h5')
    model.save(model_path)
    save_class_names(class_names, os.path.join(output_dir, f'{model_name}_finetuned_final_class_names.txt'))
    logger.info(f'Saved fine-tuned model to {model_path}')
    
    return {
        'model_name': f'{model_name}_finetuned',
        'num_classes': num_classes,
        'epochs': epochs,
        'fine_tuned': True,
        'history': {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy'],
        },
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
        'best_val_acc': float(max(history.history['val_accuracy'])),
    }


def main():
    parser = argparse.ArgumentParser(description='Train Food-101 classification models')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['baseline', 'transfer', 'all'],
                        help='Training mode')
    parser.add_argument('--num-classes', type=int, default=20,
                        choices=[20, 101],
                        help='Number of classes (20 selected or 101 all)')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='Epochs for fine-tuning on all classes')
    
    args = parser.parse_args()
    
    logger.info(f'Starting Food-101 training: mode={args.mode}, num_classes={args.num_classes}, epochs={args.epochs}')
    
    # Select classes
    selected_classes = SELECTED_20_CLASSES if args.num_classes == 20 else None
    class_label = '20' if args.num_classes == 20 else '101'
    
    # Load data
    logger.info(f'Loading Food-101 dataset ({class_label} classes)...')
    train_ds, train_info = get_filtered_pipeline(
        'train', args.batch_size, args.img_size, selected_classes
    )
    val_ds, _ = get_filtered_pipeline(
        'validation', args.batch_size, args.img_size, selected_classes
    )
    
    logger.info(f'  Train: {train_info["num_examples"]} examples')
    logger.info(f'  Classes: {train_info["num_classes"]}')
    
    results = []
    
    # Train baseline CNN
    if args.mode in ['baseline', 'all']:
        logger.info('Building baseline CNN...')
        baseline = build_baseline_cnn(train_info['num_classes'])
        logger.info(baseline.summary())
        
        baseline_results = train_model(
            baseline, train_ds, val_ds,
            f'baseline_{class_label}',
            train_info['num_classes'],
            train_info['class_names'],
            args.epochs,
            args.output_dir,
        )
        results.append(baseline_results)
    
    # Train transfer learning model
    if args.mode in ['transfer', 'all']:
        logger.info('Building EfficientNetB0 transfer learning model...')
        transfer = build_efficientnet_transfer(train_info['num_classes'])
        logger.info(transfer.summary())
        
        transfer_results = train_model(
            transfer, train_ds, val_ds,
            f'efficientnet_{class_label}',
            train_info['num_classes'],
            train_info['class_names'],
            args.epochs,
            args.output_dir,
        )
        results.append(transfer_results)
    
    # Scale to all 101 classes if currently on 20
    if args.num_classes == 20 and args.mode == 'all':
        logger.info('\n' + '='*60)
        logger.info('Scaling to all 101 classes...')
        logger.info('='*60 + '\n')
        
        # Load all classes data
        train_ds_all, train_info_all = get_filtered_pipeline(
            'train', args.batch_size, args.img_size, selected_classes=None
        )
        val_ds_all, _ = get_filtered_pipeline(
            'validation', args.batch_size, args.img_size, selected_classes=None
        )
        
        logger.info(f'  Train (all): {train_info_all["num_examples"]} examples')
        logger.info(f'  Classes (all): {train_info_all["num_classes"]}')
        
        # Rebuild models for 101 classes
        logger.info('Rebuilding baseline CNN for 101 classes...')
        baseline_101 = build_baseline_cnn(train_info_all['num_classes'])
        baseline_101_results = train_model(
            baseline_101, train_ds_all, val_ds_all,
            'baseline_101',
            train_info_all['num_classes'],
            train_info_all['class_names'],
            args.epochs,
            args.output_dir,
        )
        results.append(baseline_101_results)
        
        logger.info('Rebuilding EfficientNetB0 for 101 classes...')
        transfer_101 = build_efficientnet_transfer(train_info_all['num_classes'])
        transfer_101_results = train_model(
            transfer_101, train_ds_all, val_ds_all,
            'efficientnet_101',
            train_info_all['num_classes'],
            train_info_all['class_names'],
            args.epochs,
            args.output_dir,
        )
        results.append(transfer_101_results)
        
        # Fine-tune transfer model on all classes
        logger.info('\nFine-tuning EfficientNetB0 on all 101 classes...')
        finetune_results = finetune_transfer_model(
            transfer_101, train_ds_all, val_ds_all,
            'efficientnet_101',
            train_info_all['num_classes'],
            train_info_all['class_names'],
            args.finetune_epochs,
            args.output_dir,
        )
        results.append(finetune_results)
    
    # Save results summary
    results_file = os.path.join(args.output_dir, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nSaved results to {results_file}')
    
    # Print summary
    logger.info('\n' + '='*60)
    logger.info('TRAINING SUMMARY')
    logger.info('='*60)
    for r in results:
        logger.info(f"\n{r['model_name']} ({r['num_classes']} classes):")
        logger.info(f"  Train Accuracy: {r['final_train_acc']:.4f}")
        logger.info(f"  Val Accuracy:   {r['final_val_acc']:.4f}")
        logger.info(f"  Best Val Acc:   {r['best_val_acc']:.4f}")


if __name__ == '__main__':
    main()

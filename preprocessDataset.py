"""
Food-101 Dataset Preprocessing with TFDS Streaming

This module provides utilities to preprocess the Food-101 dataset directly from
TensorFlow Datasets (TFDS) using streaming. Preprocessing includes:
- Image resizing and normalization
- Data augmentation for training
- Batching and prefetching for performance
- Caching (optional)

Usage:
    python preprocessDataset.py --split train --batch-size 32 --img-size 224
"""

import argparse
import logging
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_tfds_food101(split: str = 'train') -> Tuple[tf.data.Dataset, dict]:
    """
    Load Food-101 dataset from TFDS with metadata.
    
    Args:
        split: Dataset split ('train', 'validation', or 'test')
    
    Returns:
        Tuple of (dataset, info_dict)
    """
    builder = tfds.builder('food101')
    builder.download_and_prepare()
    ds, info = tfds.load(
        'food101',
        split=split,
        as_supervised=False,
        with_info=True,
    )
    
    num_examples = info.splits[split].num_examples
    num_classes = info.features['label'].num_classes
    class_names = info.features['label'].names
    
    info_dict = {
        'num_examples': num_examples,
        'num_classes': num_classes,
        'class_names': class_names,
    }
    
    logger.info(f'Loaded {split} split: {num_examples} examples, {num_classes} classes')
    return ds, info_dict


def preprocess_image(example: dict,
                     img_size: int = 224,
                     normalize: bool = True,
                     augment: bool = False) -> Tuple[tf.Tensor, int]:
    """
    Preprocess a single example from Food-101.
    
    Args:
        example: Dictionary with 'image' and 'label' keys
        img_size: Target image size (square)
        normalize: Whether to normalize to [-1, 1] range
        augment: Whether to apply data augmentation (for training)
    
    Returns:
        Tuple of (preprocessed_image, label)
    """
    image = example['image']
    label = example['label']
    
    # Convert to float32
    image = tf.cast(image, tf.float32)
    
    # Resize image
    image = tf.image.resize(image, (img_size, img_size))
    
    # Data augmentation (only for training)
    if augment:
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random vertical flip
        image = tf.image.random_flip_up_down(image)
        
        # Random brightness adjustment
        image = tf.image.adjust_brightness(image, delta=0.1)
        
        # Random contrast adjustment
        image = tf.image.adjust_contrast(image, contrast_factor=0.9)
    
    # Normalize
    if normalize:
        # Normalize to [-1, 1] using standard normalization
        image = image / 127.5 - 1.0
    else:
        # Normalize to [0, 1]
        image = image / 255.0
    
    # Ensure image is in valid range
    image = tf.clip_by_value(image, -1.0 if normalize else 0.0, 1.0)
    
    return image, label


def create_data_pipeline(split: str = 'train',
                        batch_size: int = 32,
                        img_size: int = 224,
                        augment: bool = True,
                        normalize: bool = True,
                        cache: bool = False,
                        prefetch_buffer: int = tf.data.AUTOTUNE) -> Tuple[tf.data.Dataset, dict]:
    """
    Create a complete data pipeline for Food-101.
    
    Args:
        split: Dataset split ('train', 'validation', or 'test')
        batch_size: Batch size for training
        img_size: Target image size
        augment: Whether to apply augmentation
        normalize: Whether to normalize images
        cache: Whether to cache the dataset (memory-intensive)
        prefetch_buffer: Prefetch buffer size for performance
    
    Returns:
        Tuple of (pipeline_dataset, info_dict)
    """
    # Load dataset
    ds, info_dict = get_tfds_food101(split=split)
    
    # Apply preprocessing
    def preprocess_fn(example):
        return preprocess_image(
            example,
            img_size=img_size,
            normalize=normalize,
            augment=augment and split == 'train',
        )
    
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache (optional)
    if cache:
        logger.info('Caching dataset...')
        ds = ds.cache()
    
    # Shuffle (only for training)
    if split == 'train':
        shuffle_buffer = min(info_dict['num_examples'], 10000)
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    
    # Batch
    ds = ds.batch(batch_size)
    
    # Prefetch for performance
    ds = ds.prefetch(buffer_size=prefetch_buffer)
    
    logger.info(f'Created pipeline for {split}: batch_size={batch_size}, img_size={img_size}')
    return ds, info_dict


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Food-101 dataset from TFDS with streaming.'
    )
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to load.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for the data pipeline.')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Target image size (square).')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Apply data augmentation.')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='Disable data augmentation.')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize to [-1, 1].')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Disable normalization.')
    parser.add_argument('--cache', action='store_true', default=False,
                        help='Cache dataset in memory.')
    parser.add_argument('--num-batches', type=int, default=10,
                        help='Number of batches to display for verification.')
    
    args = parser.parse_args()
    
    # Create pipeline
    ds, info = create_data_pipeline(
        split=args.split,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=args.augment,
        normalize=args.normalize,
        cache=args.cache,
    )
    
    logger.info(f'Dataset info: {info}')
    logger.info(f'Verifying {args.num_batches} batches...')
    
    # Iterate and verify
    for batch_idx, (images, labels) in enumerate(ds.take(args.num_batches)):
        logger.info(f'Batch {batch_idx + 1}: images shape={images.shape}, labels shape={labels.shape}')
        logger.info(f'  Image range: [{tf.reduce_min(images).numpy():.3f}, {tf.reduce_max(images).numpy():.3f}]')
        logger.info(f'  Label range: [{tf.reduce_min(labels).numpy()}, {tf.reduce_max(labels).numpy()}]')
    
    logger.info('Preprocessing verification complete!')


if __name__ == '__main__':
    main()

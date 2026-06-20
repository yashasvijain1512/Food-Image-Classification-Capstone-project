"""
Evaluate a saved Keras model on the Food-101 TFDS validation split.

Usage:
    python evaluate_model.py --model models/efficientnet_101_best.h5 --num-batches 50

This script streams the validation split from TFDS, preprocesses images to the
requested size, and computes accuracy and a simple classification report.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a saved Keras model on Food-101 validation data.')
    p.add_argument('--model', required=True, help='Path to saved Keras model (.h5 or SavedModel dir)')
    p.add_argument('--dataset', default='food101', help='TFDS dataset name')
    p.add_argument('--split', default='validation', help='TFDS split to evaluate')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-batches', type=int, default=20, help='Number of validation batches to evaluate')
    p.add_argument('--top-k', type=int, default=1, help='Compute top-k accuracy')
    p.add_argument('--selected-classes-file', type=str, default=None,
                   help='Optional path to a text file listing selected class names to evaluate')
    p.add_argument('--out', type=str, default='evaluation_report.json')
    p.add_argument('--plot', action='store_true', help='Save confusion matrix image next to JSON report')
    return p.parse_args()


def load_selected_class_names(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def make_val_pipeline(dataset_name: str,
                      split: str,
                      img_size: int,
                      batch_size: int,
                      selected_class_names=None):
    ds, info = tfds.load(dataset_name, split=split, as_supervised=False, with_info=True)
    class_names = info.features['label'].names

    table = None
    if selected_class_names is not None:
        selected_indices = [class_names.index(name) for name in selected_class_names]
        keys = tf.constant(selected_indices, dtype=tf.int64)
        values = tf.range(len(selected_indices), dtype=tf.int64)
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1,
        )

    def preprocess(example):
        image = tf.cast(example['image'], tf.float32)
        image = tf.image.resize(image, (img_size, img_size))
        image = image / 127.5 - 1.0
        label = example['label']

        if table is not None:
            label = table.lookup(label)

        return image, label

    if table is not None:
        def filter_selected(example):
            label = example['label']
            return tf.not_equal(table.lookup(label), -1)

        ds = ds.filter(filter_selected)

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, info, class_names


def compute_metrics(model: tf.keras.Model,
                    ds: tf.data.Dataset,
                    top_k: int = 1) -> Tuple[list, list]:
    y_true = []
    y_pred = []

    for images, labels in ds:
        preds = model.predict(images)
        if top_k == 1:
            preds_cls = np.argmax(preds, axis=-1)
            y_pred.extend(preds_cls.tolist())
        else:
            topk_preds = np.argsort(preds, axis=-1)[:, -top_k:][:, ::-1]
            y_pred.extend(topk_preds.tolist())
        y_true.extend(labels.numpy().tolist())

    return y_true, y_pred


def compute_top_k_accuracy(y_true, y_pred, top_k):
    if top_k == 1:
        return np.mean(np.array(y_true) == np.array(y_pred)).item()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    hits = [y_true[i] in y_pred[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def save_confusion_matrix(cm, class_names, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)

    if len(class_names) <= 30:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(model_path: str,
             dataset_name: str,
             split: str,
             img_size: int,
             batch_size: int,
             num_batches: int,
             top_k: int = 1,
             selected_class_names=None) -> dict:
    model = tf.keras.models.load_model(model_path)
    ds, info, full_class_names = make_val_pipeline(
        dataset_name,
        split,
        img_size,
        batch_size,
        selected_class_names,
    )

    y_true, y_pred = compute_metrics(model, ds.take(num_batches), top_k=top_k)

    if top_k == 1:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
    else:
        report = {'top_{}_accuracy'.format(top_k): compute_top_k_accuracy(y_true, y_pred, top_k)}
        cm = []

    class_names = selected_class_names if selected_class_names is not None else full_class_names

    return {
        'model': str(model_path),
        'dataset': dataset_name,
        'split': split,
        'num_samples': len(y_true),
        'top_k': top_k,
        'report': report,
        'confusion_matrix': cm,
        'class_names': class_names,
    }


def main():
    args = parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(f'Model path not found: {args.model}')

    selected_class_names = None
    if args.selected_classes_file:
        selected_class_names = load_selected_class_names(args.selected_classes_file)

    out = evaluate(
        args.model,
        args.dataset,
        args.split,
        args.img_size,
        args.batch_size,
        args.num_batches,
        args.top_k,
        selected_class_names,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Wrote evaluation report to {output_path}')

    if args.plot and out['confusion_matrix']:
        png_path = output_path.with_suffix('.png')
        save_confusion_matrix(np.array(out['confusion_matrix']), out['class_names'], png_path)
        print(f'Wrote confusion matrix to {png_path}')


if __name__ == '__main__':
    main()

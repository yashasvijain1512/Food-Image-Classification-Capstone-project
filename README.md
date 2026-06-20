# Food Image Classification — Capstone Project

This repository contains code to train and serve food image classifiers using the Food-101 dataset streamed directly from TensorFlow Datasets (TFDS). It includes preprocessing utilities, a baseline CNN implementation, an EfficientNetB0 transfer-learning pipeline, and a minimal FastAPI inference service.

**Project Goals**
- **Experiment** with a baseline CNN and an EfficientNetB0 transfer-learning model.
- **Train** first on 20 selected classes, then scale to all 101 Food-101 classes.
- **Stream** the Food-101 dataset from TFDS to avoid local dataset export and disk pressure.
- **Provide** preprocessing for both classification and object-detection (full-image boxes / COCO-style annotations).

**Included Files**
- **`preprocessDataset.py`**: Streaming TFDS preprocessing pipeline for training (resizing, normalization, augmentation, batching).
- **`preprocess_food101_od.py`**: Creates COCO-style annotations and saves full-image bounding boxes for OD workflows.
- **`train_models.py`**: Training entrypoint with both baseline CNN and EfficientNetB0 transfer-learning flows. Supports initial 20-class training and scaling to 101 classes.
- **`app.py`**: Minimal FastAPI app for image upload and prediction.
- **`requirements.txt`**: Project dependencies.

**Models**
- **Baseline CNN**: Small convolutional model (Conv blocks, BatchNorm, GlobalAveragePooling) trained from scratch. Fast to iterate; useful as a baseline.
- **EfficientNetB0 (Transfer Learning)**: Pretrained ImageNet backbone with a custom classification head. Base layers frozen initially, then optionally unfrozen for fine-tuning.

**Workflow & Commands**

1) Install dependencies (use your virtualenv):

```bash
python -m pip install -r requirements.txt
```

2) Preprocess / verify TFDS streaming pipeline (quick smoke test):

```bash
# verify a few batches
python preprocessDataset.py --split train --batch-size 32 --img-size 224 --num-batches 3
```

3) Generate object-detection-style dataset (COCO annotations with full-image boxes):

```bash
python preprocess_food101_od.py --output-dir dataset/od --img-size 512 --max-examples 200
```

4) Train models on 20 selected classes (fast experiment):

```bash
# Baseline CNN on 20 classes
python train_models.py --mode baseline --num-classes 20 --epochs 10 --batch-size 32

# EfficientNetB0 transfer on 20 classes
python train_models.py --mode transfer --num-classes 20 --epochs 10 --batch-size 32
```

5) Scale to all 101 classes (rebuild/retrain and optionally fine-tune):

```bash
# Train both models on all classes
python train_models.py --mode all --num-classes 101 --epochs 20 --batch-size 32

# Fine-tune EfficientNetB0 (lower LR) after initial training
python train_models.py --mode transfer --num-classes 101 --epochs 10 --finetune-epochs 5
```

6) Run the FastAPI inference server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# then open http://localhost:8000/ to use the upload UI
```

**Notes & Best Practices**
- The project uses TFDS streaming to avoid exporting the full Food-101 archive to disk; this conserves space and works well in constrained environments.
- If disk space becomes tight, remove the TFDS cache at `~/tensorflow_datasets/` to free several GBs.
- Start with the 20-class experiments to iterate quickly. When results are satisfactory, scale up to all 101 classes and fine-tune the transfer model.
- Model artifacts are saved under the `models/` folder by default.

**Useful Links**
- [preprocessDataset.py](preprocessDataset.py)
- [preprocess_food101_od.py](preprocess_food101_od.py)
- [train_models.py](train_models.py)
- [app.py](app.py)
- [requirements.txt](requirements.txt)

If you want, I can also add example notebook snippets, a short evaluation script, or CI steps for automated experiments. Which would you like next?

Added artifacts
- `evaluate_model.py`: Stream validation split and compute classification report (JSON output).
- `notebooks/example_training.ipynb`: Minimal notebook showing quick training and saving a model.
- `.github/workflows/ci.yml`: Minimal CI that installs requirements and verifies script imports.


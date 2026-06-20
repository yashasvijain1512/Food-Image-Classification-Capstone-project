
# Food Image Classification — Capstone Project

This repository contains an image classification pipeline for identifying food items from photos. The project covers dataset preprocessing, model training, evaluation, and a planned UI for inference and dataset browsing.

## Table of Contents
- **Project**: Brief overview
- **Dataset**: source and layout
- **Preprocessing Pipeline**: steps and scripts
- **Modeling**: architecture and training
- **Evaluation**: metrics and validation
- **Deployment & UI Plan**: architecture and UI planning
- **Tools & Dependencies**: primary packages and CLI
- **How to run**: setup and example commands
- **Project Structure**: repository layout

## Project

Goal: Build a robust classifier that recognizes food items from images using modern deep learning practices and a repeatable pipeline.

Use cases:
- Mobile or web app for meal logging
- Dietary tracking and calorie estimation pipeline (future work)

## Dataset

- Source: collect images from public datasets (e.g., Food-101), web scraping, or user-provided photos.
- Expected layout (recommended):

	dataset/
	- train/
		- class_1/
		- class_2/
		- ...
	- val/
	- test/

- The included script `preprocessDataset.py` prepares raw images into the train/val/test splits and applies basic cleaning and resizing. See [preprocessDataset.py](preprocessDataset.py) for details.

## Preprocessing Pipeline

The preprocessing stage should be deterministic and reproducible. Typical steps performed by `preprocessDataset.py`:

- **Ingest**: read raw images and labels from source folders or CSV metadata.
- **Clean**: remove corrupted or unreadable images, unify file formats.
- **Resize / Re-encode**: scale to a consistent resolution (e.g., 224x224) and convert color spaces consistently.
- **Augmentation** (training only): random flips, rotations, color jitter, random crops — apply with a library like `albumentations`.
- **Normalization**: scale pixel values and (optionally) apply dataset-specific mean/std normalization.
- **Split**: create reproducible train / val / test splits (by seed) and save manifests (CSV or JSON) listing image paths and labels.
- **Serialization** (optional): save preprocessed data as TFRecord, LMDB, or a flat file manifest to speed training.

Example usage:

```bash
# Create virtual env and activate
python -m venv .venv
source .venv/bin/activate

# Run preprocessing (adjust args as needed)
python preprocessDataset.py --source raw_images/ --output dataset/ --img-size 224 --val-split 0.15 --test-split 0.10
```

## Modeling & Architecture

This project is framework-agnostic; either TensorFlow/Keras or PyTorch can be used. Recommended architecture choices:

- Baseline: transfer learning with a pre-trained CNN backbone (e.g., EfficientNet-B0, ResNet50, MobileNetV2).
- Head: global average pooling → dropout → dense softmax output for classification.
- Training recipe:
	- Use an ImageNet-pretrained backbone.
	- Fine-tune head first, then unfreeze some backbone layers for gradual fine-tuning.
	- Optimizers: AdamW or SGD with momentum.
	- Learning rate scheduling: cosine annealing or step decay; consider a warmup period.
	- Batch size & augmentation tuned to hardware constraints.

Example training command (PyTorch pseudocode):

```bash
python train.py --data dataset/ --arch efficientnet_b0 --epochs 30 --batch-size 32 --lr 1e-3
```

## Evaluation

Track these metrics on validation and test sets:

- **Accuracy** (top-1)
- **Top-k accuracy** (top-3, top-5 as appropriate)
- **Precision / Recall / F1** per class and macro-averaged
- **Confusion matrix** to find confusing class pairs
- **Calibration**: reliability diagrams if using probabilities for downstream decisions

Use cross-validation or stratified splits for robust estimates if data is limited.

## Deployment & UI Plan

Architecture overview:

- Inference service: lightweight REST API (Flask/FastAPI) exposing a `/predict` endpoint that accepts image uploads and returns class + confidence.
- Model storage: serialized model artifacts (PyTorch `.pt` or TensorFlow SavedModel) in `models/` with versioned filenames.
- Frontend: single-page web UI (React or plain HTML/JS) to upload images, display predictions, and browse dataset examples.
- Optional mobile: an exported TensorFlow Lite or ONNX model for on-device inference.

UI Planning — minimum viable product (MVP):

- **Home / Upload**: dropzone to upload an image, preview, and `Classify` button.
- **Results**: show predicted label(s) with confidence bars and top-3 suggestions.
- **Gallery / Dataset Browser**: paginated view of dataset images by class, with filters and sample counts.
- **Admin**: allow users to flag bad images or relabel; enable incremental dataset updates.

UX considerations:

- Keep inference latency under 500ms on the server for good experience.
- Show helpful failure states (e.g., "Image too small" or "Uncertain prediction").
- Add an attribution or source note for model versions used.

Tech stack suggestions:

- Backend: `FastAPI` for REST endpoints + `uvicorn` for ASGI server
- Frontend: React + Material UI (or simple Bootstrap-based UI for MVP)
- Model serving: containerize model server with Docker; use GPU instance when needed

## Tools & Dependencies

- Languages: Python 3.8+
- ML frameworks: TensorFlow 2.x or PyTorch
- Data and image: `numpy`, `pandas`, `opencv-python`, `Pillow`, `albumentations`
- Utilities: `scikit-learn` (metrics & splits), `matplotlib`/`seaborn` (plots)
- Devops: `docker`, `uvicorn`, `gunicorn` (optional), `fastapi` (or `flask`)
- Install all dependencies from `requirements.txt`

Recommended `requirements.txt` snippet:

```
tensorflow>=2.14.0
tensorflow-datasets>=4.9.2
keras>=2.14.0
numpy>=1.26.0
pandas>=2.2.0
opencv-python>=4.9.0.74
Pillow>=10.0.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.2
albumentations>=1.4.0
fastapi>=0.118.0
uvicorn>=0.26.0
```

## How to run (local)

1. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Preprocess dataset

```bash
python preprocessDataset.py --source raw_images/ --output dataset/ --img-size 224
```

3. Train model

```bash
python train.py --data dataset/ --arch resnet50 --epochs 20
```

4. Run inference server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

5. Open the UI

Point the browser to `http://localhost:8000` (or the frontend host).

## Project Structure

- `preprocessDataset.py` — dataset preprocessing and splitting script. See [preprocessDataset.py](preprocessDataset.py).
- `train.py` — training entrypoint (not included yet; add per framework).
- `app.py` or `server/` — inference API (FastAPI/Flask) for serving model predictions.
- `models/` — store model artifacts and versioned checkpoints.
- `dataset/` — produced train/val/test dataset by `preprocessDataset.py`.
- `notebooks/` — optional exploratory analysis and visualization notebooks.

## Next steps (suggested)

1. Add `train.py` with configurable backbone, optimizer, and logging (WandB or TensorBoard).
2. Create `requirements.txt` and CI checks for linting and unit tests.
3. Implement a simple FastAPI inference endpoint and a minimal React UI.
4. Add dataset labeling and active learning loop for continuous improvement.

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests and submit pull requests for fixes and enhancements.

## License

Choose a license appropriate for your project (MIT/Apache-2.0/etc.).

---

If you'd like, I can:
- create a `requirements.txt` from the tools listed,
- scaffold `train.py` for PyTorch or TensorFlow,
- or scaffold a minimal FastAPI + React UI for the inference flow.
Tell me which next item you'd like me to implement.




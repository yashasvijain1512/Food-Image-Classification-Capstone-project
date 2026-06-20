from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Tuple
import uvicorn
import io
from PIL import Image
import numpy as np
import os
import tensorflow as tf

APP_MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
IMG_SIZE = int(os.environ.get('IMG_SIZE', 224))
TOP_K = int(os.environ.get('TOP_K', 3))

app = FastAPI(title='Food Image Classification - Inference')


def load_model_and_labels(model_dir: str):
    """Load a TensorFlow SavedModel or .h5 and the class names file.
    Returns (model, class_names)
    """
    model = None
    # Try SavedModel folder first
    saved_model_path = os.path.join(model_dir, 'saved_model')
    h5_path = os.path.join(model_dir, 'best_model.h5')
    if os.path.isdir(saved_model_path):
        model = tf.keras.models.load_model(saved_model_path)
    elif os.path.exists(h5_path):
        model = tf.keras.models.load_model(h5_path)
    else:
        raise FileNotFoundError(f'No model found in {model_dir} (tried saved_model/ and best_model.h5)')

    class_file = os.path.join(model_dir, 'class_names.txt')
    class_names = None
    if os.path.exists(class_file):
        with open(class_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # If not present, try to infer number of outputs
        out_dim = model.output_shape[-1]
        class_names = [f'class_{i}' for i in range(out_dim)]

    return model, class_names


@app.on_event('startup')
def startup():
    global MODEL, CLASS_NAMES
    try:
        MODEL, CLASS_NAMES = load_model_and_labels(APP_MODEL_DIR)
        print(f'Loaded model from {APP_MODEL_DIR} with {len(CLASS_NAMES)} classes')
    except Exception as e:
        MODEL = None
        CLASS_NAMES = []
        print('Model not loaded:', e)


def preprocess_imagefile(file_bytes: bytes, img_size: int) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    image = image.resize((img_size, img_size))
    arr = np.array(image).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def top_k_predictions(preds: np.ndarray, class_names: List[str], k: int) -> List[dict]:
    probs = preds[0]
    k = min(k, probs.shape[-1])
    top_idx = probs.argsort()[-k:][::-1]
    result = []
    for i in top_idx:
        result.append({
            'label': class_names[i] if i < len(class_names) else str(i),
            'index': int(i),
            'score': float(probs[i])
        })
    return result


@app.get('/', response_class=HTMLResponse)
def home():
    html = f"""
    <html>
      <head>
        <title>Food Classifier</title>
      </head>
      <body>
        <h1>Food Image Classification — Inference</h1>
        <form action="/predict" enctype="multipart/form-data" method="post">
          <input name="file" type="file" accept="image/*">
          <input type="submit" value="Classify">
        </form>
        <p>Model directory: {APP_MODEL_DIR}</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded on server')
    data = await file.read()
    try:
        x = preprocess_imagefile(data, IMG_SIZE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image file: {e}')

    preds = MODEL.predict(x)
    results = top_k_predictions(preds, CLASS_NAMES, TOP_K)
    return JSONResponse({'predictions': results})


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': MODEL is not None}


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=False)

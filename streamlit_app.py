import io
import os
from typing import List

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
IMG_SIZE = int(os.environ.get('IMG_SIZE', 224))
TOP_K_DEFAULT = int(os.environ.get('TOP_K', 3))


@st.cache_resource(show_spinner=False)
def load_model_and_labels(model_dir: str):
    """Load a TensorFlow SavedModel or .h5 and class names from disk."""
    model = None
    saved_model_path = os.path.join(model_dir, 'saved_model')
    h5_path = os.path.join(model_dir, 'best_model.h5')

    if os.path.isdir(saved_model_path):
        model = tf.keras.models.load_model(saved_model_path)
    elif os.path.exists(h5_path):
        model = tf.keras.models.load_model(h5_path)
    else:
        raise FileNotFoundError(
            f'No model found in {model_dir}. Put a SavedModel directory at "{saved_model_path}" or a file at "{h5_path}".'
        )

    class_file = os.path.join(model_dir, 'class_names.txt')
    if os.path.exists(class_file):
        with open(class_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        output_shape = model.output_shape
        class_count = int(output_shape[-1])
        class_names = [f'class_{i}' for i in range(class_count)]

    return model, class_names


def preprocess_imagefile(file_bytes: bytes, img_size: int) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    image = image.resize((img_size, img_size))
    arr = np.array(image).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def top_k_predictions(preds: np.ndarray, class_names: List[str], k: int) -> List[dict]:
    probs = preds[0]
    k = min(k, probs.shape[-1])
    top_idx = probs.argsort()[-k:][::-1]
    return [
        {
            'label': class_names[i] if i < len(class_names) else f'class_{i}',
            'index': int(i),
            'score': float(probs[i]),
        }
        for i in top_idx
    ]


def main():
    st.set_page_config(page_title='Food Image Classification', layout='centered')
    st.title('Food Image Classification')
    st.write('Upload a food image and get top model predictions.')

    st.sidebar.header('Settings')
    st.sidebar.write(f'Model directory: `{APP_MODEL_DIR}`')
    top_k = st.sidebar.slider('Top K predictions', 1, 5, TOP_K_DEFAULT)
    img_size = st.sidebar.select_slider('Image size', options=[128, 160, 224, 256], value=IMG_SIZE)

    try:
        model, class_names = load_model_and_labels(APP_MODEL_DIR)
    except Exception as exc:
        st.error(f'Unable to load model: {exc}')
        return

    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is None:
        st.info('Upload an image to classify it.')
        return

    image_bytes = uploaded_file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as exc:
        st.error(f'Invalid image: {exc}')
        return

    st.image(image, caption='Uploaded image', use_column_width=True)

    if st.button('Classify image'):
        with st.spinner('Running prediction...'):
            x = preprocess_imagefile(image_bytes, img_size)
            preds = model.predict(x)
            results = top_k_predictions(preds, class_names, top_k)

        st.success('Prediction complete')
        for item in results:
            st.write(f"**{item['label']}** — {item['score']:.4f}")

        if len(results) == 0:
            st.warning('Model produced no predictions.')


if __name__ == '__main__':
    main()

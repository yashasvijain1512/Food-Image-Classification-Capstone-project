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


def list_model_options(model_dir: str):
    options = []
    if os.path.isdir(os.path.join(model_dir, 'saved_model')):
        options.append(('SavedModel directory', model_dir))

    for entry in sorted(os.listdir(model_dir)):
        entry_path = os.path.join(model_dir, entry)
        if os.path.isfile(entry_path) and entry_path.endswith('.h5'):
            options.append((entry, entry_path))
        elif os.path.isdir(entry_path):
            nested = os.path.join(entry_path, 'saved_model')
            if os.path.isdir(nested):
                options.append((f'{entry}/saved_model', entry_path))
    return options


def _find_class_names(model_path: str):
    model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    candidates = []

    if os.path.isfile(model_path):
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        candidates.extend([
            os.path.join(model_dir, f'{base_name}_class_names.txt'),
            os.path.join(model_dir, f'{base_name}_best_class_names.txt'),
            os.path.join(model_dir, f'{base_name}_final_class_names.txt'),
        ])

    candidates.append(os.path.join(model_dir, 'class_names.txt'))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_model_and_labels(model_path: str):
    """Load a TensorFlow SavedModel or .h5 and class names from disk."""
    if os.path.isdir(model_path):
        saved_model_path = os.path.join(model_path, 'saved_model')
        if not os.path.isdir(saved_model_path):
            raise FileNotFoundError(f'No SavedModel directory found in {model_path}')
        model = tf.keras.models.load_model(saved_model_path)
    elif os.path.isfile(model_path) and model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f'No loadable model found at {model_path}')

    class_file = _find_class_names(model_path)
    if class_file is not None:
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
    top_k = st.sidebar.slider('Top K predictions', 1, 5, TOP_K_DEFAULT)
    img_size = st.sidebar.select_slider('Image size', options=[128, 160, 224, 256], value=IMG_SIZE)

    model_options = list_model_options(APP_MODEL_DIR)
    if not model_options:
        st.error(f'No models found in `{APP_MODEL_DIR}`. Place a SavedModel or .h5 file there.')
        return

    model_choice = st.sidebar.selectbox(
        'Select model',
        [label for label, _ in model_options],
        index=0,
    )
    selected_path = dict(model_options)[model_choice]

    st.sidebar.write(f'Model path: `{selected_path}`')

    try:
        model, class_names = load_model_and_labels(selected_path)
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

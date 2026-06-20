import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models, optimizers, losses, metrics


print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Datasets version: {tfds.__version__}")

#loading of dataset
ds, info = tfds.load('food101', as_supervised=True, with_info=True)

train_ds, test_ds = ds['train'], ds['test']
validation_ds = ds['validation']
BATCH_SIZE = 32
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

IMG_SIZE = 224
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), y))

def preprocess_image(image, label):
  """
  Preprocesses images by resizing and normalizing them to a [0, 1] range.
  """
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = tf.cast(image, tf.float32) / 255.0 # Scale pixel values to [0, 1]
  return image, label

train_ds = train_ds.map(preprocess_image)
validation_ds = validation_ds.map(preprocess_image)
test_ds = test_ds.map(preprocess_image)     

#DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.2),
  layers.RandomBrightness(0.2)], name="data_augmentation")

def augment_data(image, label):
    """
    Applies data augmentation to the image.
    """
    # Only augment images during training
    image = data_augmentation(image, training=True)
    return image, label

# Get all class names
class_names = info.features['label'].names

# Select the first 20 classes for experimentation
SELECTED_CLASSES = class_names[:20]
NUM_SELECTED_CLASSES = len(SELECTED_CLASSES)

# Get the integer indices of the selected classes
selected_indices = [class_names.index(c) for c in SELECTED_CLASSES]
selected_class_indices_tensor = tf.constant(selected_indices, dtype=tf.int64)

print(f"Selected {NUM_SELECTED_CLASSES} classes: {SELECTED_CLASSES}")

from keras import Sequential

def build_baseline_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(info.features['label'].names)):
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Build the baseline CNN model with 20 classes
baseline_model_20_classes = build_baseline_cnn(num_classes=NUM_SELECTED_CLASSES)

# Compile the baseline model
baseline_model_20_classes.compile(optimizer='adam',
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'])

print("\nBaseline CNN Model (20 classes) Summary:")
baseline_model_20_classes.summary()

# Build the baseline CNN model_with the original number of classes (101)
baseline_model = build_baseline_cnn()

# Compile the baseline model
baseline_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# Display the model summary
baseline_model.summary()

#EfficientNetB0
from keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

def build_transfer_learning_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(info.features['label'].names)):
    # Load the pre-trained EfficientNetB0 model, excluding the top (classification) layer
    base_model = tf.keras.applications.EfficientNetB0(include_top=False,
                                                      input_shape=input_shape,
                                                      weights='imagenet')

    # Freeze the base model to prevent its weights from being updated during training
    base_model.trainable = False

    # Create a new model on top
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False) # Ensure the base model is in inference mode
    x = layers.GlobalAveragePooling2D()(x) # Reduce spatial dimensions to a single vector
    x = layers.Dense(128, activation='relu')(x) # Add a dense layer for more feature extraction
    x = layers.Dropout(0.3)(x) # Add dropout for regularization
    outputs = layers.Dense(num_classes, activation='softmax')(x) # Output layer with softmax for classification

    model = models.Model(inputs, outputs)

    return model

# Build the model
model = build_transfer_learning_model()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Build the transfer learning model with 20 classes
model_20_classes = build_transfer_learning_model(num_classes=NUM_SELECTED_CLASSES)

# Compile the model
model_20_classes.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

print("Transfer Learning Model (20 classes) Summary:")
model_20_classes.summary()



import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.tensorflow_model.model import build_model

# Dataset Paths
TRAIN_DIR = "data/Training"
TEST_DIR = "data/Testing"

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Training Dataset
train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Validation Dataset
validation_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Test Dataset
test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Build Model
model = build_model()

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display Model Summary
model.summary()
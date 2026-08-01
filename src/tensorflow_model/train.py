import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import TensorBoard
import mlflow
import mlflow.tensorflow




from src.tensorflow_model.model import build_model

# Dataset Paths
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

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

# Optimize dataset performance

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Build Model
model = build_model()

mlflow.set_experiment("Brain Tumor Diagnosis - TensorFlow CNN")

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

EPOCHS = 10

tensorboard_callback = TensorBoard(
    log_dir="logs",
    histogram_freq=0
)

with mlflow.start_run():

    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("image_size", IMAGE_SIZE)
    mlflow.log_param("optimizer", "adam")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback]
    )

    mlflow.log_metric(
        "train_accuracy",
        history.history["accuracy"][-1]
    )

    mlflow.log_metric(
        "validation_accuracy",
        history.history["val_accuracy"][-1]
    )

    mlflow.log_metric(
        "train_loss",
        history.history["loss"][-1]
    )

    mlflow.log_metric(
        "validation_loss",
        history.history["val_loss"][-1]
    )

    mlflow.tensorflow.log_model(
        model,
        name="brain_tumor_tensorflow_model"
    )







history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[tensorboard_callback]
)





model.save("models/tensorflow_brain_tumor_model.keras")

print("Model saved successfully!")


from src.tensorflow_model.plot_history import plot_history

plot_history(history)
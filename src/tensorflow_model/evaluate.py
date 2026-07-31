import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load model
model = tf.keras.models.load_model(
    "models/tensorflow_brain_tumor_model.keras"
)

# Dataset
test_dataset = image_dataset_from_directory(
    "data/Testing",
    image_size=(224,224),
    batch_size=32,
    shuffle=False
)

# Predictions
predictions = model.predict(test_dataset)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.concatenate(
    [labels.numpy() for images, labels in test_dataset]
)

class_names = test_dataset.class_names

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        true_labels,
        predicted_labels
    )
)
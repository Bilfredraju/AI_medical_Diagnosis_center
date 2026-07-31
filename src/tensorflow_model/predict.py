import numpy as np
import tensorflow as tf
from PIL import Image

from src.tensorflow_model.class_names import CLASS_NAMES

MODEL_PATH = "models/tensorflow_brain_tumor_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    print("\nPrediction")
    print("---------------------")
    print(f"Class      : {CLASS_NAMES[predicted_index]}")
    print(f"Confidence : {confidence:.2%}")


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    predict_image(image_path)
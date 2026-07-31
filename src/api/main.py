from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

from src.api.schemas import PredictionResponse
from src.tensorflow_model.class_names import CLASS_NAMES

app = FastAPI(
    title="AI Medical Diagnosis API",
    version="1.0.0"
)

# Load the trained TensorFlow model
model = tf.keras.models.load_model(
    "models/tensorflow_brain_tumor_model.keras"
)


@app.get("/")
def home():
    return {
        "message": "AI Medical Diagnosis API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Normalize
    image = np.array(image) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    # Probability of each class
    probabilities = {
        CLASS_NAMES[i]: float(prediction[0][i])
        for i in range(len(CLASS_NAMES))
    }

    # Return response
    return PredictionResponse(
        prediction=CLASS_NAMES[predicted_index],
        confidence=confidence,
        probabilities=probabilities
    )
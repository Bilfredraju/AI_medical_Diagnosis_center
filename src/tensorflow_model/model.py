import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4


def build_model():
    model = Sequential([

        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(224, 224, 3)
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(256, activation="relu"),

        Dropout(0.5),

        Dense(NUM_CLASSES, activation="softmax")
    ])

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
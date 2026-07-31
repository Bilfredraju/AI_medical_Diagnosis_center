import tensorflow as tf

from src.transfer_learning.model import build_model

# Dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/Training",
    image_size=(224, 224),
    batch_size=32
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32
)

# Build model
model = build_model()

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)

# Save
model.save(
    "models/efficientnet_brain_tumor.keras"
)

print("\nEfficientNet model saved successfully!")
import os
import shutil
import random

random.seed(42)

train_dir = "data/train"
validation_dir = "data/validation"

os.makedirs(validation_dir, exist_ok=True)

classes = os.listdir(train_dir)

for class_name in classes:

    source = os.path.join(train_dir, class_name)
    destination = os.path.join(validation_dir, class_name)

    os.makedirs(destination, exist_ok=True)

    images = os.listdir(source)

    random.shuffle(images)

    split = int(len(images) * 0.2)

    validation_images = images[:split]

    for image in validation_images:

        shutil.move(
            os.path.join(source, image),
            os.path.join(destination, image)
        )

print("Validation dataset created successfully!")
import matplotlib.pyplot as plt

from src.preprocessing.data_loader import train_loader


# Get one batch of images
images, labels = next(iter(train_loader))

# Display first 9 images
plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)

    # Convert tensor to image
    image = images[i].permute(1, 2, 0).numpy()

    # Undo normalization for display
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)

    plt.imshow(image)
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.tight_layout()
plt.show()
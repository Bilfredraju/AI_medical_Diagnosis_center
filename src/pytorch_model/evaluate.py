import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.pytorch_model.model import BrainTumorCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Test Dataset
test_dataset = datasets.ImageFolder(
    "data/Testing",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Load Model
model = BrainTumorCNN().to(device)

model.load_state_dict(
    torch.load(
        "models/pytorch_brain_tumor_model.pth",
        map_location=device
    )
)

model.eval()

predictions = []
labels_list = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.cpu().numpy())

        labels_list.extend(labels.numpy())

print("\nClassification Report\n")

print(
    classification_report(
        labels_list,
        predictions,
        target_names=test_dataset.classes
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        labels_list,
        predictions
    )
)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.preprocessing.transforms import train_transform, test_transform


# Dataset paths
TRAIN_DIR = "data/Training"
VALID_DIR = "data/Testing"
TEST_DIR = "data/Testing"

# Create datasets
train_dataset = ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

valid_dataset = ImageFolder(
    root=VALID_DIR,
    transform=test_transform
)

test_dataset = ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Device agnostic code
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Setup dir paths
DATA_PATH = Path("data/")
TRAIN_PATH = DATA_PATH / "train"
TEST_PATH = DATA_PATH / "test"

classes = ("cat", "dog", "snake")

def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(DATA_PATH)

# Manual seed
torch.manual_seed(27)
random.seed(27)

# 1. Get all image paths
data_path_list = list(DATA_PATH.glob("*/*/*.jpg"))

# 2. Choose random image
random_image_path = random.choice(data_path_list)

# 3. Image class
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")

# Image as array
img_as_array = np.asarray(img)

# Plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, colour_channels]")
plt.axis(False)
plt.show()

# Transforming data
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

print(data_transform(img).shape)
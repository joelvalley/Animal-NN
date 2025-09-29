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

train_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)), 
    transforms.ToTensor()])

train_data = datasets.ImageFolder(root=TRAIN_PATH,
                         transform=train_transform,
                         target_transform=None)

test_data = datasets.ImageFolder(root=TEST_PATH,
                         transform=test_transform,
                         target_transform=None)

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data, 
                              batch_size=BATCH_SIZE,
                              shuffle=True)

class AnimalNN(nn.Module):
    def __init__(self, 
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )

    def forward(self, x: int):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

image_batch, label_batch = next(iter(train_dataloader))

model = AnimalNN(input_shape=3,
                 hidden_units=10,
                 output_shape=3).to(device)

model(image_batch.to(device))
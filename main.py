import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from timeit import default_timer as timer

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
    transforms.TrivialAugmentWide(num_magnitude_bins=31), 
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

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        #1. Forward pass
        y_logits = model(x)
        y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

        # 2. Calculate loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        train_acc += (y_preds==y).sum().item()/len(y_preds)

    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module):
    # Put model in evaluation mode
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # 1. Forward pass
            y_logits = model(x)
            y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

            # 2. Calculate loss
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            test_acc += (y_preds==y).sum().item()/len(y_preds)

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    results = {"train_loss" : [],
                   "train_acc" : [],
                   "test_loss" : [],
                   "test_acc" : []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

def plot_loss_curves(results: dict[str : list[float]]):
    epochs = list(range(len(results["train_loss"])))
    
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="train loss")
    plt.plot(epochs, results["test_loss"], label="test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="train acc")
    plt.plot(epochs, results["test_acc"], label="test acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

model = AnimalNN(input_shape=3,
                 hidden_units=10,
                 output_shape=3).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

summary(model=model, input_size=(BATCH_SIZE, 3, 64, 64), device=device)

NUM_EPOCHS = 10

start_time = timer()

results = train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=NUM_EPOCHS)

end_time = timer()
print(f"Model train time: {(end_time-start_time):.4f} seconds")

plot_loss_curves(results=results)
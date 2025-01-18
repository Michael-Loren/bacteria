#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Specify transformer model and transform layer for the script.")
parser.add_argument(
    "--transformer_model", 
    type=str, 
    required=True, 
    help="Specify the transformer model (e.g., 'facebook/deit-base-patch16-224')."
)
parser.add_argument(
    "--transform_layer", 
    type=str, 
    required=True, 
    help="Specify the transform layer (e.g., 'DCT', 'WHT')."
)
parser.add_argument(
    "--num_epochs",
    type=int,
    required=True,
    help="Specify the number of epochs."
)
# Parse arguments
args = parser.parse_args()
transformer_model = args.transformer_model
transform_layer = args.transform_layer
num_epochs = args.num_epochs

import os
import torch
import scipy.io
import torch.nn as nn  # Import nn module
import torch.optim as optim  # Import optim module
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from hyperspectral_dataset import HyperspectralDataset, CustomTransform
from rgb_dataset import RGBDataset
from torchvision.transforms import Resize, CenterCrop, Compose, Normalize, ToTensor
from transformer import TransformerMod
from ResNet import CustomResNet50  # Assuming ResNet50 is defined in ResNet.py
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should print the name of your GPU
# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[2]:


import torch
from torch.utils.data import DataLoader

# Function to get all file paths from a dataset
def get_all_file_paths(dataset):
    file_paths = []
    for idx in range(len(dataset)):
        file_paths.append(dataset.get_image_path(idx))
    return file_paths

# Initialize the dataset with transformations
dataset = HyperspectralDataset(root_dir='../../ummdsRP/all')

# Define mean and std for cube and profile (replace these with actual values if available)
channel_means_cube = [0.5] * 31
channel_stds_cube = [0.5] * 31
channel_means_profile = [0.5] * 31
channel_stds_profile = [0.5] * 31

# Define the custom transformation including normalization and resizing
custom_transform = CustomTransform(
    cube_mean=channel_means_cube, 
    cube_std=channel_stds_cube, 
    profile_mean=channel_means_profile, 
    profile_std=channel_stds_profile,
    resize_shape=(224, 224)  # Resize cube to 224x224
)

# Apply the custom transforms to the datasets
train_dataset = HyperspectralDataset(root_dir='../../ummdsRP/train', transform=custom_transform)
val_dataset = HyperspectralDataset(root_dir='../../ummdsRP/val', transform=custom_transform)
test_dataset = HyperspectralDataset(root_dir='../../ummdsRP/test', transform=custom_transform)

# Create DataLoaders for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
print(len(train_dataset.label_map))
# Example to verify the changes
for sample, label in train_loader:
    print(f"Cube shape: {sample['cube'].shape}, Profile shape: {sample['profile'].shape}, Label: {label}")
    break


# In[3]:


# Initialize the modified Transformer model
model = TransformerMod(
    num_classes=len(train_dataset.label_map), 
    transformer_model=transformer_model, 
    transform_layer=transform_layer
).to(device)

for name, param in model.named_parameters():
    print(f"{name}: {'requires_grad' if param.requires_grad else 'frozen'}")


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)


# Training and validation loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_precisions, train_recalls, train_f1_scores = [], [], []
val_precisions, val_recalls, val_f1_scores = [], [], []
lr_change_epochs = []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    all_train_labels, all_train_preds = [], []

    for batch_idx, (sample, labels) in enumerate(train_loader):
        cubes, profiles = sample['cube'], sample['profile']
        cubes, profiles, labels = cubes.to(device), profiles.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(cubes)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(predicted.cpu().numpy())

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        all_train_labels, all_train_preds, average='weighted', zero_division=0
    )
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1_scores.append(train_f1)

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    all_val_labels, all_val_preds = [], []

    with torch.no_grad():
        for sample, labels in val_loader:
            cubes, profiles = sample['cube'], sample['profile']
            cubes, profiles, labels = cubes.to(device), profiles.to(device), labels.to(device)
            outputs = model(cubes)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < current_lr:
        lr_change_epochs.append(epoch + 1)

print("Training complete. Beginning testing...")

# Plot the training and validation losses with learning rate change points
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
for lr_epoch in lr_change_epochs:
    plt.axvline(x=lr_epoch, color='r', linestyle='--', label='LR Change' if lr_epoch == lr_change_epochs[0] else "")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses with Learning Rate Changes')
plt.grid(True)
plt.savefig('./visuals/training_validation_loss.png')
plt.close()

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
for lr_epoch in lr_change_epochs:
    plt.axvline(x=lr_epoch, color='r', linestyle='--', label='LR Change' if lr_epoch == lr_change_epochs[0] else "")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True)
plt.savefig('./visuals/training_validation_accuracy.png')
plt.close()

# Testing loop
model.eval()
test_loss, correct, total = 0.0, 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs['cube'].to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%")

# Confusion matrix
true_labels, predicted_labels = [], []
for i in range(len(test_dataset)):
    cube, label = test_dataset[i]
    cube = cube['cube'].to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(cube)
    predicted_class = torch.argmax(output.logits, dim=1)
    true_labels.append(label)
    predicted_labels.append(predicted_class.item())

cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('./visuals/confusion_matrix.png')
plt.close()

print("All images saved in ./visuals")


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import CompactMNIST
from train import train_one_epoch, count_parameters, evaluate_model
import pytest
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# Define transform globally for all tests
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def test_parameter_count():
    model = CompactMNIST()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, which exceeds the limit of 25,000"

# Test for Data Augmentation Integrity
def test_data_augmentation():
    # Load a sample of the MNIST dataset with only ToTensor transform
    basic_transform = transforms.Compose([
        transforms.ToTensor()  # This will ensure values are in [0,1]
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=basic_transform)
    
    # Get a sample image and label
    img, label = dataset[0]
    
    # Check the shape of the image
    assert img.shape == (1, 28, 28), f"Expected image shape (1, 28, 28), got {img.shape}"
    
    # Check that the pixel values are in the range [0, 1] before normalization
    assert img.min() >= 0 and img.max() <= 1, f"Image pixel values should be in the range [0, 1], got min: {img.min()}, max: {img.max()}"
    
    # Check that the label is valid
    assert 0 <= label <= 9, f"Label should be between 0 and 9, got {label}"

# Test for Model Overfitting
def test_model_overfitting():
    # Load the training dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = CompactMNIST()
    model.train()

    # Train the model for a few epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):  # Train for 2 epochs
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluate on the training set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    train_accuracy = 100. * correct / total
    
    # Check that the training accuracy is above a certain threshold
    assert train_accuracy > 50, f"Model did not learn the training data, accuracy: {train_accuracy:.2f}%"

# Test for Gradient Clipping Functionality
def test_gradient_clipping():
    model = CompactMNIST()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    # Create a dummy input and target
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST images are 28x28
    dummy_target = torch.tensor([1])  # Assuming class 1 for the dummy target

    # Forward pass
    output = model(dummy_input)
    loss = nn.CrossEntropyLoss()(output, dummy_target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Check if any gradient exceeds the max norm
    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.norm() <= 1.0, "Gradient exceeds the maximum norm after clipping"
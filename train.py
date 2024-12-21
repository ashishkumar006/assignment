import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CompactMNIST
import numpy as np
import random
import os

# Define transform with image augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),  # Randomly rotate images by ±10 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch():
    os.environ['MKL_NUM_THREADS'] = '4'
    
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cpu")
    torch.set_num_threads(4)
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    model = CompactMNIST().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.005,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=len(train_loader) // 3,
        gamma=0.5
    )

    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        running_loss += loss.item()
        
        # Print loss and accuracy every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            avg_accuracy = 100. * correct / total
            print(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.3f}, Accuracy: {avg_accuracy:.2f}%')

    accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {accuracy:.2f}%')
    
    return model, accuracy

def evaluate_model(model):
    device = torch.device("cpu")
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return accuracy

if __name__ == "__main__":
    num_epochs = 1  # Set to 1 for a single epoch
    for epoch in range(num_epochs):
        train_one_epoch()
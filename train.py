import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CompactMNIST
import math
from tqdm import tqdm
import numpy as np
import random
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch():
    # Set environment variable for Intel MKL optimization
    os.environ['MKL_NUM_THREADS'] = '4'  # For Intel CPU optimization
    
    # Set all random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)

    # Optimized transforms for Intel GPU
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0
            )
        ], p=0.5)
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=96,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # Simplified device selection
    device = torch.device("cpu")
    torch.set_num_threads(4)  # Use all threads on i3
    
    print(f"Using device: {device}")
    
    model = CompactMNIST().to(device)
    model.train()
    
    param_count = count_parameters(model)
    print(f"Model has {param_count} parameters")
    if param_count >= 25000:
        raise ValueError("Model has too many parameters!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.002,  # Slightly lower learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Adjusted scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=len(train_loader) // 4,  # More frequent steps
        gamma=0.2  # Less aggressive decay
    )

    correct = 0
    total = 0
    running_loss = 0.0

    # Add progress bar
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {accuracy:.2f}%')
    
    if accuracy < 95:
        raise ValueError("Model accuracy is below 95%!")
    
    return accuracy

if __name__ == "__main__":
    train_one_epoch() 
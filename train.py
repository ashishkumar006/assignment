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
    # Check if running in GitHub Actions
    is_github = os.getenv('GITHUB_ACTIONS') == 'true'
    
    # Set environment variable for Intel MKL optimization
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # Set all random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)

    # Enhanced transforms for GitHub environment
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=6,
                translate=(0.06, 0.06),
                scale=(0.94, 1.06),
                fill=0
            )
        ], p=0.6)
    ])

    # Adjusted batch size for GitHub
    batch_size = 96 if is_github else 128

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cpu")
    torch.set_num_threads(4)
    
    model = CompactMNIST().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    
    # Optimized parameters for GitHub
    if is_github:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.004,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=0.008
        )
        
        # Custom learning rate schedule for GitHub
        def lr_schedule(step, total_steps):
            warmup_steps = total_steps // 6
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        total_steps = len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_schedule(step, total_steps)
        )
    else:
        # Original settings for local training
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.003,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=len(train_loader) // 3,
            gamma=0.3
        )

    correct = 0
    total = 0
    running_loss = 0.0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        running_loss += loss.item()
        
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
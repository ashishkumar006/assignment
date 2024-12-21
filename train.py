import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CompactMNIST
from tqdm import tqdm
import numpy as np
import random
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch():
    os.environ['MKL_NUM_THREADS'] = '4'
    
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)

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
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cpu")
    torch.set_num_threads(4)
    
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

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
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
    num_epochs = 1  # Set to 1 for a single epoch
    for epoch in range(num_epochs):
        train_one_epoch() 
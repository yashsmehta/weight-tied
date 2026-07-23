"""
Training script for ECTiedNet on CIFAR-10.

Usage:
    python train.py                          # Train with defaults
    python train.py --channels 96 --lr 0.05  # Custom config
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ECTiedNet, count_parameters


# ============================================================================
# Data Loading
# ============================================================================

def get_dataloaders(batch_size: int = 128, num_workers: int = 4):
    """
    Create CIFAR-10 train/test dataloaders with standard augmentation.

    Train augmentation: RandomCrop + HorizontalFlip
    Test: Just normalize (no augmentation)
    """
    # CIFAR-10 channel-wise mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),      # 50% chance flip
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download CIFAR-10 if not present
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set. Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100. * correct / total


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ECTiedNet on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--channels', type=int, default=64, help='Base channel width')
    parser.add_argument('--iterations', type=int, default=6, help='Block reuse count')
    parser.add_argument('--expansion', type=int, default=4, help='Expansion ratio')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Model
    model = ECTiedNet(
        num_classes=10,
        channels=args.channels,
        expansion=args.expansion,
        num_iterations=args.iterations,
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Test: {test_loss:.4f} / {test_acc:.2f}% | "
              f"Best: {best_acc:.2f}%")

    print(f"\nDone. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

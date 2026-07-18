"""
Training script for ECTiedNet on CIFAR-10.

Usage
-----
    # Baseline run
    python train.py

    # Custom width and longer schedule
    python train.py --channels 128 --epochs 300

    # Resume from checkpoint
    python train.py --resume checkpoints/best.pt

Training additions over a plain SGD loop
-----------------------------------------
AMP (mixed precision):
    Forward and loss computed in float16; weights updated in float32.
    Negligible accuracy cost; significant speedup on modern GPUs.

Gradient clipping (max_norm=1.0):
    The same weights accumulate gradients from all N iterations of the shared
    block — the gradient w.r.t. W is the sum of N per-iteration terms, which
    can be substantially larger than a single-layer gradient. Clipping prevents
    any one destabilising update. LayerScale in ECBlock provides complementary
    protection by keeping early residual contributions near zero.

LR warmup (5 epochs, linear):
    Ramps LR from 10% of its target to full value before cosine decay begins.
    At random initialisation, recursive block application compounds noise across
    iterations; small early updates reduce the chance of an unlucky large step
    setting the shared weights onto a poor trajectory.

Label smoothing:
    Off by default (0.0) for CIFAR-10 development; set to 0.1 for ImageNet.
    Prevents overconfident predictions and keeps gradients informative late
    in training when the model would otherwise coast on saturated softmax outputs.

Checkpointing:
    Saves full training state (model, optimiser, scheduler, scaler) every
    --save-every epochs and whenever a new best accuracy is reached. Pass
    --resume <path> to continue an interrupted run without losing progress.
"""
import argparse
import os

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
    CIFAR-10 train/test loaders with standard augmentation.

    Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
    Test:  Normalize only (no augmentation)
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10('./data', train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, scaler, criterion, device):
    """Train for one epoch. Returns (avg_loss, accuracy %)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

        scaler.scale(loss).backward()

        # Unscale before clipping so clip threshold is in the original gradient scale
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        correct    += outputs.argmax(1).eq(targets).sum().item()
        total      += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set. Returns (avg_loss, accuracy %)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct    += outputs.argmax(1).eq(targets).sum().item()
        total      += targets.size(0)

    return total_loss / total, 100.0 * correct / total


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ECTiedNet on CIFAR-10')

    # Model
    parser.add_argument('--channels',    type=int,   default=64,   help='Base channel width')
    parser.add_argument('--expansion',   type=int,   default=4,    help='Block expansion ratio')
    parser.add_argument('--iterations',  type=int,   default=6,    help='Block reuse count')

    # Training
    parser.add_argument('--epochs',          type=int,   default=200)
    parser.add_argument('--batch-size',      type=int,   default=128)
    parser.add_argument('--lr',              type=float, default=0.1)
    parser.add_argument('--weight-decay',    type=float, default=5e-4)
    parser.add_argument('--warmup-epochs',   type=int,   default=5)
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='0.0 for CIFAR-10; use 0.1 for ImageNet')

    # Checkpointing
    parser.add_argument('--save-dir',   type=str, default='checkpoints')
    parser.add_argument('--save-every', type=int, default=50,  help='Save checkpoint every N epochs')
    parser.add_argument('--resume',     type=str, default=None, help='Path to checkpoint to resume from')

    # Misc
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed',        type=int, default=42)

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Device: {device}\n")

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    # Model
    model = ECTiedNet(
        num_classes=10,
        channels=args.channels,
        expansion=args.expansion,
        num_iterations=args.iterations,
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss, optimiser, LR schedule, AMP scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)
    scaler    = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    # Linear warmup → cosine decay
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_epochs
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs],
    )

    # Resume from checkpoint
    start_epoch, best_acc = 0, 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt['best_acc']
        print(f"Resumed from epoch {ckpt['epoch']}  (best acc so far: {best_acc:.2f}%)\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, criterion, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        is_best  = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train {train_loss:.4f} / {train_acc:.2f}% | "
              f"test {test_loss:.4f} / {test_acc:.2f}% | "
              f"best {best_acc:.2f}%  "
              f"{'*' if is_best else ''}")

        # Save checkpoint
        if is_best or (epoch + 1) % args.save_every == 0:
            ckpt = {
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler':    scaler.state_dict(),
                'best_acc':  best_acc,
                'args':      vars(args),
            }
            if is_best:
                torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
            if (epoch + 1) % args.save_every == 0:
                torch.save(ckpt, os.path.join(args.save_dir, f'epoch_{epoch+1:03d}.pt'))

    print(f"\nDone. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

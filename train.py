"""
Training script for ECTiedNet on CIFAR-10.

Usage
-----
    python train.py                           # defaults
    python train.py --channels 128            # wider model
    python train.py --resume checkpoints/best.pt

Key training decisions
----------------------
Separate LR for DivisiveNorm sigma:
    log_sigma accumulates gradients from all N block iterations (one per
    call), making its effective gradient N times larger than other parameters.
    A separate, lower LR (--sigma-lr-scale, default 0.1x) prevents sigma
    from oscillating and dragging the loss into a spike-then-NaN trajectory.

Gradient clipping applied to main params only:
    log_sigma is excluded from the clip budget. When included, its large
    gradient norm dominates the global norm and forces the actual conv weights
    to be under-updated (most of the clip scaling is spent on sigma).

LR warmup (LambdaLR):
    Ramps LR from 0 to full value over --warmup-epochs. Implemented with
    LambdaLR rather than SequentialLR to avoid PyTorch's off-by-one
    scheduler step warning and the associated unpredictable LR at epoch 0.

NaN detection:
    Training halts immediately if loss becomes NaN or Inf, saving the last
    valid checkpoint rather than running indefinitely on a broken model.
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ECTiedNet, count_parameters, divisive_norm_params, gamma_params, main_params


# ============================================================================
# Data Loading
# ============================================================================

def get_dataloaders(batch_size: int = 128, num_workers: int = 4):
    """
    CIFAR-10 train/test loaders with standard augmentation.

    Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
    Test:  Normalize only
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

def train_epoch(model, loader, optimizer, scaler, criterion, device, main_param_list):
    """
    Train for one epoch. Returns (avg_loss, accuracy %).
    Clips gradients on main parameters only — log_sigma excluded so its
    large per-iteration gradient accumulation does not consume the clip budget.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Clip main params only — log_sigma has its own reduced LR instead
        torch.nn.utils.clip_grad_norm_(main_param_list, max_norm=1.0)

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
    parser.add_argument('--channels',      type=int,   default=64)
    parser.add_argument('--expansion',     type=int,   default=4)
    parser.add_argument('--iterations',    type=int,   default=6)

    # Training
    parser.add_argument('--epochs',          type=int,   default=200)
    parser.add_argument('--batch-size',      type=int,   default=128)
    parser.add_argument('--lr',              type=float, default=0.01)
    parser.add_argument('--weight-decay',    type=float, default=5e-4)
    parser.add_argument('--warmup-epochs',   type=int,   default=5)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--sigma-lr-scale',  type=float, default=0.1,
                        help='LR multiplier for DivisiveNorm log_sigma.')
    parser.add_argument('--gamma-lr-scale',  type=float, default=0.01,
                        help='LR multiplier for LayerScale gamma. '
                             'Gamma accumulates gradients from all N block '
                             'iterations so without a reduced LR it grows '
                             '~20x in the first epoch and destabilises training.')

    # Checkpointing
    parser.add_argument('--save-dir',   type=str, default='checkpoints')
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--resume',     type=str, default=None)

    # Misc
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed',        type=int, default=42)

    args = parser.parse_args()

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
    print(f"Parameters: {count_parameters(model):,}\n")

    # Three parameter groups:
    #   main   — conv weights, norms  → full lr
    #   sigma  — DivisiveNorm sigma   → 0.1x lr (large per-iteration grad)
    #   gamma  — LayerScale gamma     → 0.01x lr (accumulates N×grad, grows 20x/epoch)
    sigma_params = divisive_norm_params(model)
    g_params     = gamma_params(model)
    other_params = main_params(model)
    optimizer = optim.SGD([
        {'params': other_params, 'lr': args.lr},
        {'params': sigma_params, 'lr': args.lr * args.sigma_lr_scale},
        {'params': g_params,     'lr': args.lr * args.gamma_lr_scale},
    ], momentum=0.9, weight_decay=args.weight_decay)

    # LambdaLR: linear warmup then cosine decay
    # LambdaLR is used instead of SequentialLR to avoid PyTorch's
    # off-by-one step warning and unpredictable epoch-0 LR behaviour.
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Clip only main params — sigma/gamma have their own reduced LRs
    main_param_list = other_params

    # Resume
    start_epoch, best_acc = 0, 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt['best_acc']
        print(f"Resumed from epoch {ckpt['epoch']}  (best: {best_acc:.2f}%)\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, criterion, device, main_param_list
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # NaN detection — halt rather than running on a broken model
        if not math.isfinite(train_loss):
            print(f"Epoch {epoch+1:3d}: NaN/Inf loss detected — halting. "
                  f"Last valid checkpoint saved.")
            break

        is_best  = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"lr={lr:.5f} | "
              f"train {train_loss:.4f} / {train_acc:.2f}% | "
              f"test {test_loss:.4f} / {test_acc:.2f}% | "
              f"best {best_acc:.2f}%  {'*' if is_best else ''}")

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

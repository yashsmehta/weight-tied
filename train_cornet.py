"""
Training script for CORnet-Z / CORnet-S / CORnet-RT on ImageNet-1k.

Uses the OFFICIAL architecture code from github.com/dicarlolab/CORnet
(installed as the `cornet` package — see Installation below), but trains
from scratch with THIS project's own SGD + LambdaLR setup rather than the
original CORnet training recipe or CORnet's public pretrained checkpoints.
This is intentional (PROGRESS.md): using a different training setup per
model would confound the comparison table. Every model in this project
(ECTiedNet variants + CORnet family) is trained the same way; only the
architecture differs.

Installation
------------
    pip install git+https://github.com/dicarlolab/CORnet

Usage
-----
    python train_cornet.py --model rt --data-path /path/to/imagenet
    python train_cornet.py --model s  --data-path /path/to/imagenet --epochs 100
    python train_cornet.py --resume checkpoints/cornet_rt/best.pt --model rt --data-path ...

Data layout
-----------
Expects the standard torchvision ImageFolder layout:
    <data-path>/train/<class>/*.JPEG
    <data-path>/val/<class>/*.JPEG
(matches the original CORnet repo's own data-prep instructions)

Model selection (--model): z | s | rt | r
    z, s, rt map directly to CORnet_Z / CORnet_S / CORnet_RT.

    IMPORTANT — CORnet-R vs CORnet-RT: the upstream repo deprecated the
    original CORnet-R. Its published forward() has a bug where, at each
    recurrent timestep, a later area's input is drawn from the SAME
    timestep's already-updated earlier-area output rather than the
    PREVIOUS timestep's output — i.e. it is not doing the biological
    temporal unrolling described in the CORnet paper (confirmed by
    inspecting cornet_r.py: it mutates `outputs` in place across areas
    within a timestep, whereas cornet_rt.py builds a fresh `new_outputs`
    dict each timestep so every area reads the previous timestep's
    values). The repo's own README says the same thing and recommends
    RT as the corrected replacement. `--model r` is provided ONLY for
    completeness / matching older comparisons; default to `rt` unless you
    have a specific reason to reproduce the original (buggy) CORnet-R.
    CONTEXT.md/project_plan.md currently say "CORnet-Z/R/S" as the
    baseline family — worth flagging to your supervisor whether that
    should read Z/RT/S instead.

Reused training decisions from train.py (see that file for rationale)
-----------------------------------------------------------------------
Same SGD (momentum=0.9) + LambdaLR (linear warmup -> cosine decay),
same NaN-detection halt, same checkpoint dict format, same per-model
checkpoint directory convention (checkpoints/cornet_<model>/).

Differences from train.py, and why
-----------------------------------------------------------------------
No sigma/gamma param groups:
    CORnet has no DivisiveNorm or LayerScale — every model in this family
    uses plain (Group/Batch)Norm and no learned residual scale, so a
    single param group with one LR is sufficient. Gradient clipping is
    applied to all parameters (there's no over-large per-iteration
    gradient accumulation to isolate, unlike ECTiedNet's log_sigma/gamma).

AMP enabled by default:
    train.py disables AMP because ECTiedNet's custom DivisiveNorm
    overflows float16. CORnet uses only standard Conv2d/BatchNorm2d/
    GroupNorm2d, which don't have that failure mode, so AMP defaults on
    here. Pass --no-amp if you hit instability and want to rule it out.

Label smoothing defaults to 0.1:
    Matches PROGRESS.md's locked decision ("Label smoothing: ... 0.1 for
    ImageNet").

ImageNet epoch count is UNRESOLVED (PROGRESS.md open question: CORnet
used 70; this project's smaller models may want >=100). --epochs has no
principled default baked in here — set it explicitly per your decision
with your supervisor, and use the SAME value for every model in the
comparison table.
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from cornet.cornet_z import CORnet_Z
    from cornet.cornet_s import CORnet_S
    from cornet.cornet_rt import CORnet_RT
    from cornet.cornet_r import CORnet_R  # deprecated upstream; kept for completeness
except ImportError as e:
    raise ImportError(
        "The official CORnet package is required and was not found. Install it with:\n"
        "    pip install git+https://github.com/dicarlolab/CORnet\n"
        "This project intentionally uses CORnet's own architecture code (not a "
        "reimplementation) so the baseline comparison is exactly the published "
        "architecture. See PROGRESS.md: do NOT use CORnet's public *pretrained* "
        "checkpoints, but the architecture code itself is the right dependency."
    ) from e


# ============================================================================
# Model registry
# ============================================================================
# name -> (constructor, kwargs). Z/S take no args; R/RT take `times` (number
# of recurrent passes, default 5 upstream).
MODEL_REGISTRY = {
    'z':  (CORnet_Z,  {}),
    's':  (CORnet_S,  {}),
    'rt': (CORnet_RT, {'times': 5}),
    'r':  (CORnet_R,  {'times': 5}),  # deprecated upstream — see module docstring
}


def build_model(name: str, times: int = 5) -> nn.Module:
    ctor, kwargs = MODEL_REGISTRY[name]
    kwargs = dict(kwargs)
    if 'times' in kwargs:
        kwargs['times'] = times
    model = ctor(**kwargs)
    if name == 'r':
        print("WARNING: --model r uses the upstream-deprecated CORnet-R "
              "(non-biological unrolling bug). Prefer --model rt unless you "
              "specifically need to reproduce the original CORnet-R.")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Data Loading
# ============================================================================

def get_dataloaders(data_path: str, batch_size: int = 256, num_workers: int = 8):
    """
    ImageNet-1k train/val loaders, standard ImageFolder layout.
    Normalization matches the original CORnet repo's own preprocessing
    (torchvision ImageNet mean/std), for consistency with how CORnet was
    originally evaluated even though we're not using their training recipe.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(data_path, 'val'),   val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, scaler, criterion, device, use_amp=True):
    """Train for one epoch. Returns (avg_loss, top1_acc %, top5_acc %)."""
    model.train()
    total_loss, correct1, correct5, total = 0.0, 0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        top5 = outputs.topk(5, dim=1).indices
        correct1 += top5[:, :1].eq(targets.unsqueeze(1)).sum().item()
        correct5 += top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        total    += targets.size(0)

    return total_loss / total, 100.0 * correct1 / total, 100.0 * correct5 / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    """Evaluate on val set. Returns (avg_loss, top1_acc %, top5_acc %)."""
    model.eval()
    total_loss, correct1, correct5, total = 0.0, 0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        top5 = outputs.topk(5, dim=1).indices
        correct1 += top5[:, :1].eq(targets.unsqueeze(1)).sum().item()
        correct5 += top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        total    += targets.size(0)

    return total_loss / total, 100.0 * correct1 / total, 100.0 * correct5 / total


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CORnet-Z/S/RT on ImageNet-1k')

    # Model
    parser.add_argument('--model', type=str, default='rt',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='z/s/rt = official CORnet architectures. r = deprecated '
                             'upstream CORnet-R, kept only for completeness (see '
                             'module docstring). Defaults to rt, not r.')
    parser.add_argument('--times', type=int, default=5,
                        help='Number of recurrent timesteps for rt/r (ignored for z/s).')

    # Data
    parser.add_argument('--data-path', type=str, required=True,
                        help='ImageNet root containing train/ and val/ subfolders.')

    # Training — same SGD + LambdaLR conventions as train.py
    parser.add_argument('--epochs',          type=int,   default=100,
                        help='NOT settled — PROGRESS.md open question. CORnet '
                             'originally used 70; use the same value for every '
                             'model in the comparison table.')
    parser.add_argument('--batch-size',      type=int,   default=256)
    parser.add_argument('--lr',              type=float, default=0.1)
    parser.add_argument('--weight-decay',    type=float, default=5e-4)
    parser.add_argument('--warmup-epochs',   type=int,   default=5)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # Checkpointing
    parser.add_argument('--save-dir',   type=str, default=None,
                        help='Defaults to checkpoints/cornet_<model>/.')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume',     type=str, default=None)

    # Misc
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--no-amp',      action='store_true',
                        help='Disable mixed precision. CORnet uses standard Conv2d/'
                             'BatchNorm/GroupNorm so AMP is expected to be safe here '
                             '(unlike ECTiedNet\'s custom DivisiveNorm) — pass this '
                             'only if you observe instability.')

    args = parser.parse_args()

    save_dir = args.save_dir if args.save_dir is not None else os.path.join('checkpoints', f'cornet_{args.model}')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Model: cornet_{args.model}" + (f" (times={args.times})" if args.model in ('r', 'rt') else "") + "\n")

    # Data
    train_loader, val_loader = get_dataloaders(args.data_path, args.batch_size, args.num_workers)

    # Model
    model = build_model(args.model, times=args.times).to(device)
    print(f"Parameters: {count_parameters(model):,}\n")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                           weight_decay=args.weight_decay)

    # LambdaLR: linear warmup then cosine decay — identical schedule shape to train.py
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp   = (device.type == 'cuda') and not args.no_amp
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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
        print(f"Resumed from epoch {ckpt['epoch']}  (best top-1: {best_acc:.2f}%)\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_top1, train_top5 = train_epoch(
            model, train_loader, optimizer, scaler, criterion, device, use_amp=use_amp,
        )
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        # NaN detection — halt rather than running on a broken model
        if not math.isfinite(train_loss):
            print(f"Epoch {epoch+1:3d}: NaN/Inf loss detected — halting. "
                  f"Last valid checkpoint saved.")
            break

        is_best  = val_top1 > best_acc
        best_acc = max(val_top1, best_acc)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"lr={lr:.5f} | "
              f"train {train_loss:.4f} / top1={train_top1:.2f}% top5={train_top5:.2f}% | "
              f"val {val_loss:.4f} / top1={val_top1:.2f}% top5={val_top5:.2f}% | "
              f"best top1={best_acc:.2f}%  {'*' if is_best else ''}")

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
                torch.save(ckpt, os.path.join(save_dir, 'best.pt'))
            if (epoch + 1) % args.save_every == 0:
                torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch+1:03d}.pt'))

    print(f"\nDone. Best val top-1 accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

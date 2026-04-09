"""
Standalone dataloader supporting CIFAR-10 and ImageNet-Mini-50 (1000 classes, 50 images each).

Usage:
    from imagenet_mini_dataloader import get_dataloaders

    # ImageNet-Mini-50 (default)
    train_loader, test_loader = get_dataloaders(dataset="imagenet-mini-50", batch_size=64)

    # CIFAR-10
    train_loader, test_loader = get_dataloaders(dataset="cifar10", batch_size=128)

    for images, labels in train_loader:
        ...

Requires: torch, torchvision, Pillow.
"""

import json
import os
import warnings
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ---------- paths (edit these for your setup) ----------
IMAGENET_DATA_DIR = "/data/shared/datasets/imagenet-mini-50"
FOLDER_LABELS = "/home/rsingh55/folder_labels.json"
CIFAR10_DATA_DIR = "./data"

# ---------- normalization constants ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]


# ---------- ImageNet transforms ----------
def _imagenet_train_transform(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _imagenet_val_transform(size=224, resize=256):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------- CIFAR-10 transforms ----------
def _cifar10_train_transform(size=32):
    return transforms.Compose([
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _cifar10_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class ImageNetMini50(Dataset):
    """ImageNet-Mini-50: 1000 classes × 50 images, with deterministic 80/20 split."""

    def __init__(self, split="train", transform=None, train_ratio=0.8):
        assert split in ("train", "test", "all")
        self.transform = transform

        with open(FOLDER_LABELS) as f:
            folder_labels = json.load(f)

        # Collect all (path, label) tuples
        samples = []
        for folder in sorted(os.listdir(IMAGENET_DATA_DIR)):
            if folder not in folder_labels:
                continue
            label = int(folder_labels[folder])
            folder_path = os.path.join(IMAGENET_DATA_DIR, folder)
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".jpeg", ".jpg")):
                    samples.append((os.path.join(folder_path, fname), label))

        # Deterministic split (seed=42) — matches the visreps training pipeline
        if split in ("train", "test"):
            g = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(samples), generator=g).tolist()
            cut = int(len(samples) * train_ratio)
            indices = indices[:cut] if split == "train" else indices[cut:]
            samples = [samples[i] for i in indices]

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(dataset="imagenet-mini-50", batch_size=64, num_workers=4, image_size=None):
    """Return (train_loader, test_loader) for the selected dataset.

    Args:
        dataset:     "imagenet-mini-50" or "cifar10"
        batch_size:  samples per batch
        num_workers: DataLoader worker processes
        image_size:  override default image size (224 for ImageNet, 32 for CIFAR-10)
    """
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    if dataset == "imagenet-mini-50":
        size = image_size or 224
        train_ds = ImageNetMini50("train", _imagenet_train_transform(size))
        test_ds  = ImageNetMini50("test",  _imagenet_val_transform(size))

    elif dataset == "cifar10":
        size = image_size or 32
        train_ds = datasets.CIFAR10(CIFAR10_DATA_DIR, train=True,  download=True, transform=_cifar10_train_transform(size))
        test_ds  = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=True, transform=_cifar10_val_transform())

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'imagenet-mini-50' or 'cifar10'.")

    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imagenet-mini-50", choices=["imagenet-mini-50", "cifar10"])
    args = parser.parse_args()

    train_loader, test_loader = get_dataloaders(dataset=args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Train: {len(train_loader.dataset)} images, Test: {len(test_loader.dataset)} images")
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels: {labels[:8]}")

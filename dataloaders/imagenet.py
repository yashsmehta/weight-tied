"""ImageNet / imagenet-mini-N training dataloader.

Ported from visreps/dataloaders/obj_cls.py, trimmed to imagenet-only (drops
tiny-imagenet and PCA-label support, neither used by this project).
"""
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import utils

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

DS_MEAN = {"imgnet": [0.485, 0.456, 0.406]}
DS_STD = {"imgnet": [0.229, 0.224, 0.225]}


def get_transform(ds_stats="imgnet", data_augment=False, image_size=224, preprocess=True):
    """Composed transform for the given dataset stats and augmentation flag."""
    if not preprocess:
        return transforms.Compose([transforms.ToTensor()])

    resize_size, crop_size = 256, image_size
    tfms = [
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(crop_size),
    ]
    if data_augment:
        tfms += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    tfms += [transforms.ToTensor(), transforms.Normalize(DS_MEAN[ds_stats], DS_STD[ds_stats])]
    return transforms.Compose(tfms)


class ImageNetDataset(Dataset):
    """Loader for ImageNet with a flat WordNet-synset folder structure.

    Folder -> label mapping is read from IMAGENET_LOCAL_DIR/folder_labels.json.
    Loads 'train', 'test', or 'all' splits (80/20 train/test split, seed=42).
    """

    def __init__(self, base_path, split="train", transform=None, train_ratio=0.8, train_fraction=1.0):
        assert split in ["train", "test", "all"], f"Invalid split: {split}"
        self.transform = transform
        label_file = os.path.join(utils.get_env_var("IMAGENET_LOCAL_DIR"), "folder_labels.json")
        self.num_classes = 1000

        try:
            with open(label_file, "r") as f:
                self.folder_labels = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Label file not found: {label_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {label_file}")

        self.samples = []
        valid_folders = set(self.folder_labels.keys())
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"ImageNet base path not found or not a directory: {base_path}")

        for folder in os.listdir(base_path):
            if not folder.startswith("n"):
                continue
            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path) or folder not in valid_folders:
                continue
            label = int(self.folder_labels[folder])
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpeg", ".jpg")):
                    img_path = os.path.join(folder_path, fname)
                    self.samples.append((img_path, label, fname))

        self.samples.sort(key=lambda s: s[2])
        total_found = len(self.samples)

        if split in ["train", "test"]:
            if total_found == 0:
                self.samples = []
            else:
                g = torch.Generator().manual_seed(42)
                indices = torch.randperm(total_found, generator=g).tolist()
                split_idx = int(total_found * train_ratio)
                if split == "train":
                    self.samples = [self.samples[i] for i in indices[:split_idx]]
                else:
                    self.samples = [self.samples[i] for i in indices[split_idx:]]

        if split == "train" and train_fraction < 1.0 and len(self.samples) > 0:
            g = torch.Generator().manual_seed(42)
            n_total = len(self.samples)
            n_keep = max(1, int(n_total * train_fraction))
            indices = torch.randperm(n_total, generator=g).tolist()[:n_keep]
            self.samples = [self.samples[i] for i in sorted(indices)]
            print(f"train_fraction={train_fraction}: kept {n_keep} of {n_total} train samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, _ = self.samples[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_wnid_from_label(self, label_idx: int) -> str:
        """Convert a class index (0-999) to its WordNet ID."""
        for wnid, idx in self.folder_labels.items():
            if int(idx) == label_idx:
                return wnid
        raise ValueError(f"Label index {label_idx} not found.")


def create_collate_fn():
    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)
    return collate_fn


def create_dataloader(dataset: Dataset, batch_size: int = 32, num_workers: int = 4,
                       shuffle: bool = True, collate_fn=None) -> DataLoader:
    prefetch_factor = 8 if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        collate_fn=collate_fn or create_collate_fn(),
    )


def prepare_imgnet_data(cfg, shuffle, preprocess, train_test_split, base_path=None):
    """Prepare ImageNet or imagenet-mini-N datasets + dataloaders."""
    if base_path is None:
        base_path = cfg.get("dataset_path", utils.get_env_var("IMAGENET_DATA_DIR"))
    datasets, loaders = {}, {}

    splits_to_load = ["train", "test"] if train_test_split else ["all"]
    split_info = []

    for split in splits_to_load:
        augment = cfg.get("data_augment", False) and split == "train" and shuffle and preprocess
        tfms = get_transform(ds_stats="imgnet", data_augment=augment, image_size=224, preprocess=preprocess)

        train_fraction = cfg.get("train_fraction", 1.0)
        dataset = ImageNetDataset(base_path, split=split, transform=tfms, train_fraction=train_fraction)

        datasets[split] = dataset
        loaders[split] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 4),
            shuffle=shuffle,
        )
        split_info.append(f"{split}={len(dataset)}")

    print(f"ImageNet: {', '.join(split_info)}")
    return datasets, loaders


def sample_manifold_panel(cfg, n_categories=50, images_per_category=50, seed=0):
    """Reproducible, balanced ImageNet panel for manifold-geometry analysis.

    Draws n_categories classes (each with >= images_per_category images) from
    the full label pool (no train/test split -- manifold geometry doesn't
    involve training) and images_per_category images per class, matching the
    sampling convention in visreps/experiments/manifold_analysis's reference
    scripts, adapted to this project's folder_labels.json-based dataset
    instead of a plain torchvision ImageFolder.

    Returns (labels, image_paths): labels is the sorted list of selected class
    indices, image_paths[i] is the sorted list of file paths for labels[i].
    """
    base_path = cfg.get("dataset_path", utils.get_env_var("IMAGENET_DATA_DIR"))
    dataset = ImageNetDataset(base_path, split="all")

    by_label = defaultdict(list)
    for path, label, _ in dataset.samples:
        by_label[label].append(path)

    eligible = sorted(label for label, paths in by_label.items() if len(paths) >= images_per_category)
    if len(eligible) < n_categories:
        raise ValueError(
            f"only {len(eligible)} ImageNet classes have >= {images_per_category} images "
            f"(need {n_categories})"
        )

    rng = np.random.default_rng(seed)
    labels = sorted(rng.choice(eligible, size=n_categories, replace=False).tolist())
    image_paths = [
        sorted(rng.choice(by_label[label], size=images_per_category, replace=False).tolist())
        for label in labels
    ]
    return labels, image_paths


def get_obj_cls_loader(cfg, shuffle=True, preprocess=True, train_test_split=True):
    """Return (datasets, loaders) dicts for object classification."""
    dataset_name = cfg.get("dataset", "imagenet")

    if dataset_name == "imagenet":
        return prepare_imgnet_data(cfg, shuffle, preprocess, train_test_split)

    if dataset_name.startswith("imagenet-mini-"):
        try:
            num_images = int(dataset_name.split("-")[-1])
        except ValueError:
            raise ValueError(f"Invalid imagenet-mini format: {dataset_name}. Expected imagenet-mini-<number>")

        imagenet_base = Path(utils.get_env_var("IMAGENET_DATA_DIR"))
        mini_path = imagenet_base.parent / f"imagenet-mini-{num_images}"
        if not mini_path.exists():
            raise ValueError(f"ImageNet mini dataset not found at {mini_path}")

        return prepare_imgnet_data(cfg, shuffle, preprocess, train_test_split, base_path=str(mini_path))

    raise ValueError(f"Unsupported dataset: {dataset_name}")

"""
Extract final-layer activations from trained ECTiedNet models and plot 2D PCA.

Usage:
    python pca_plot.py  # Plots PCA for depths 2, 4, 6 (expects trained models)
"""
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import ECTiedNet

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_test_loader(batch_size=256):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def extract_activations(model, loader, device):
    """Extract activations after GAP (input to final linear layer)."""
    model.eval()
    all_activations = []
    all_labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        # Forward through stem + blocks + blur (everything before head)
        x = model.stem(inputs)
        for t in range(model.num_iterations):
            x = model.block(x, dilation=model.dilations[t])
            if t == (model.num_iterations // 2) - 1:
                x = model.blur(x)
        x = x.mean(dim=(2, 3))  # GAP -> these are the final-layer activations

        all_activations.append(x.cpu().numpy())
        all_labels.append(targets.numpy())

    return np.concatenate(all_activations), np.concatenate(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_loader()
    depths = [2, 4, 6]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, depth in zip(axes, depths):
        model_path = f"best_model_depth{depth}.pth"
        print(f"Loading {model_path}...")

        model = ECTiedNet(num_classes=10, channels=64, expansion=4, num_iterations=depth).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        activations, labels = extract_activations(model, test_loader, device)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(activations)

        sil = silhouette_score(activations, labels)
        ch = calinski_harabasz_score(activations, labels)
        print(f"  Depth {depth} — Silhouette: {sil:.3f}, Calinski-Harabasz: {ch:.1f}")

        for class_idx in range(10):
            mask = labels == class_idx
            ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.5, label=CIFAR10_CLASSES[class_idx])

        ax.set_title(f"Depth {depth}\nSilhouette: {sil:.3f} | CH: {ch:.1f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=3, fontsize=7, loc="best")

    fig.suptitle("2D PCA of Final-Layer Activations by Depth", fontsize=14)
    fig.tight_layout()
    plt.savefig("pca_activations.png", dpi=150, bbox_inches="tight")
    print("Saved pca_activations.png")


if __name__ == "__main__":
    main()

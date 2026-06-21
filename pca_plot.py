"""
PCA of per-stage activations from a trained ECTiedNet.

Plots three panels — Stage1 (V1), Stage2 (V4), Stage3 (IT) — to visualize
how class separation improves across the ventral stream hierarchy.

For ImageNet (1000 classes), pass --n-classes to show a readable subset.

Usage:
    python pca_plot.py --checkpoint best_model_imagenet_iter4-4-4_....pth
    python pca_plot.py --checkpoint ... --dataset imagenet-mini-50 --n-classes 20
    python pca_plot.py --checkpoint ... --dataset cifar10 --n-classes 10
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from model import ECTiedNet
from imagenet_mini_dataloader import get_dataloaders

STAGE_LABELS = ["Stage 1 (V1)", "Stage 2 (V4)", "Stage 3 (IT)"]


@torch.no_grad()
def extract_all_stage_features(model, loader, device):
    """Extract per-stage post-GAP features for the full dataset.

    Returns:
        stage_feats: list of 3 numpy arrays, each (N, C_stage)
        labels:      (N,) numpy array
    """
    model.eval()
    stage_bufs = [[] for _ in range(3)]
    label_buf  = []
    for imgs, lbls in loader:
        per_stage = model.extract_stage_features(imgs.to(device))
        for i, feat in enumerate(per_stage):
            stage_bufs[i].append(feat.cpu().numpy())
        label_buf.append(lbls.numpy())
    labels = np.concatenate(label_buf)
    stage_feats = [np.concatenate(buf) for buf in stage_bufs]
    return stage_feats, labels


def main():
    parser = argparse.ArgumentParser(description="PCA of ECTiedNet stage activations")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model_*.pth or checkpoint_*.pth")
    parser.add_argument("--dataset", default="imagenet",
                        choices=["imagenet", "imagenet-mini-50", "cifar10"])
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--stage-iterations", type=int, nargs=3, default=[4, 4, 4],
                        metavar=("N1", "N2", "N3"))
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dilations", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-classes", type=int, default=10,
                        help="Number of classes to plot (randomly sampled; default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = {"cifar10": 10, "imagenet-mini-50": 1000, "imagenet": 1000}[args.dataset]
    model = ECTiedNet(
        num_classes=num_classes,
        channels=args.channels,
        stage_iterations=tuple(args.stage_iterations),
        expansion=args.expansion,
        dilations=args.dilations,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    print(f"Loaded: {args.checkpoint}")

    _, test_loader = get_dataloaders(dataset=args.dataset, batch_size=args.batch_size)
    print(f"Extracting per-stage features from {len(test_loader.dataset)} images...")
    stage_feats, labels = extract_all_stage_features(model, test_loader, device)
    for i, (feats, label) in enumerate(zip(stage_feats, STAGE_LABELS)):
        print(f"  {label}: {feats.shape}")

    # Select which classes to plot
    all_classes = np.unique(labels)
    n_plot = min(args.n_classes, len(all_classes))
    chosen = rng.choice(all_classes, size=n_plot, replace=False)
    chosen.sort()
    mask = np.isin(labels, chosen)
    print(f"Plotting {n_plot} classes out of {len(all_classes)}: {chosen.tolist()}")

    cmap = plt.cm.get_cmap("tab20", n_plot)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, feats, stage_label in zip(axes, stage_feats, STAGE_LABELS):
        feats_sub  = feats[mask]
        labels_sub = labels[mask]

        # Metrics on all samples (not just the plotted subset)
        sil = silhouette_score(feats, labels, sample_size=min(5000, len(labels)),
                               random_state=args.seed)
        ch  = calinski_harabasz_score(feats, labels)
        print(f"  {stage_label} — Silhouette: {sil:.3f}  Calinski-Harabasz: {ch:.1f}")

        coords = PCA(n_components=2, random_state=args.seed).fit_transform(feats_sub)

        for i, cls in enumerate(chosen):
            m = labels_sub == cls
            ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6,
                       color=cmap(i), label=str(cls))

        ax.set_title(f"{stage_label}\nSilhouette: {sil:.3f}  CH: {ch:.1f}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(title="Class", markerscale=3, fontsize=6, loc="best",
                  ncol=2 if n_plot > 10 else 1)

    fig.suptitle(
        "PCA of stage activations — class separation across ventral stream hierarchy",
        fontsize=12,
    )
    fig.tight_layout()
    out = "pca_stages.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

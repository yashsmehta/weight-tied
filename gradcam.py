"""
Grad-CAM visualization for ECTiedNet on CIFAR-10.

Shows which image regions each ECBlock iteration attends to.
Since the same weights are reused 6 times with different dilation rates,
Grad-CAM reveals how dilation changes what the network "looks at" spatially.

Usage:
    python gradcam.py
    python gradcam.py --model best_model_depth6_dil1-1-2-1-2-3.pth
    python gradcam.py --n-images 6 --class-filter 5   # only dogs
    python gradcam.py --only-incorrect                 # failure analysis
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import ECTiedNet

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ============================================================================
# Data
# ============================================================================

def get_test_loader(batch_size=1):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def unnormalize(tensor):
    """Convert a normalized CHW tensor back to a displayable HWC numpy array."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


# ============================================================================
# Grad-CAM
# ============================================================================

def compute_gradcam(model, x, target_class=None):
    """
    Run Grad-CAM across all ECBlock iterations simultaneously.

    The tricky part of this architecture: ECBlock is the SAME module called
    N times. We cannot hook the module — that would mix gradients from all
    iterations. Instead we replicate the forward pass manually, call
    retain_grad() on each intermediate tensor, then backprop once.

    Args:
        model:        trained ECTiedNet (eval mode)
        x:            [1, 3, 32, 32] normalized input image on correct device
        target_class: class index to explain; defaults to argmax (predicted class)

    Returns:
        cams:       list of num_iterations heatmaps, each [32, 32] in [0, 1]
        pred_class: predicted class index
    """
    model.eval()
    model.zero_grad()

    # Dict to hold each iteration's output tensor (different object each time)
    saved = {}

    # --- Replicate ECTiedNet.forward() so we can intercept intermediate tensors ---
    feat = model.stem(x)

    for t in range(model.num_iterations):
        feat = model.block(feat, dilation=model.dilations[t])

        # Downsample halfway through (matches model.forward exactly)
        if t == (model.num_iterations // 2) - 1:
            feat = model.blur(feat)

        # retain_grad() keeps the gradient on this non-leaf tensor after backward()
        feat.retain_grad()
        saved[t] = feat   # store the tensor object *before* feat is rebound next iter

    logits = model.head(feat.mean(dim=(2, 3)))

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # Single backward pass — gradients flow to all saved tensors
    logits[0, target_class].backward()

    # Build one heatmap per iteration
    cams = []
    for t in range(model.num_iterations):
        acts  = saved[t]        # [1, C, H, W]  (H=32 for t<3, H=16 for t>=3)
        grads = acts.grad       # [1, C, H, W]

        # Importance weight per channel = spatial mean of its gradient
        weights = grads.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Upsample to input resolution so all heatmaps are comparable
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()        # [32, 32]

        # Normalize to [0, 1]
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)
        cams.append(cam)

    return cams, target_class


# ============================================================================
# Plotting
# ============================================================================

def plot_gradcam(image_tensor, cams, true_label, pred_label, dilations, save_path):
    """
    One figure per image: original + one heatmap per ECBlock iteration.

    Layout:  [Original | Iter1 dil=1 | Iter2 dil=1 | ... | Iter6 dil=3]
    """
    n_iters = len(cams)
    fig, axes = plt.subplots(1, n_iters + 1, figsize=(3.0 * (n_iters + 1), 3.8))

    img_rgb = unnormalize(image_tensor.squeeze(0))
    correct  = true_label == pred_label
    color    = "green" if correct else "red"
    status   = "correct" if correct else "wrong"

    # Original image
    axes[0].imshow(img_rgb, interpolation="nearest")
    axes[0].set_title(
        f"True:  {CIFAR10_CLASSES[true_label]}\n"
        f"Pred:  {CIFAR10_CLASSES[pred_label]}  ({status})",
        fontsize=9, color=color,
    )
    axes[0].axis("off")

    # Grad-CAM overlay for each iteration
    for t, cam in enumerate(cams):
        ax = axes[t + 1]
        ax.imshow(img_rgb, interpolation="nearest")
        im = ax.imshow(cam, cmap="jet", alpha=0.50, vmin=0, vmax=1)
        ax.set_title(f"Iter {t + 1}  (dil={dilations[t]})", fontsize=9)
        ax.axis("off")

    # Shared colorbar
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Importance")

    fig.suptitle(
        "Grad-CAM — same ECBlock weights, different dilation → different spatial attention",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_summary_grid(records, save_path="gradcam_summary.png"):
    """
    Summary figure: rows = images, columns = iterations.
    Useful for comparing attention patterns across many examples at a glance.

    records: list of dicts with keys 'img', 'cams', 'true', 'pred', 'dilations'
    """
    n_images = len(records)
    n_iters  = len(records[0]["cams"])
    n_cols   = n_iters + 1  # original + one per iteration

    fig, axes = plt.subplots(n_images, n_cols, figsize=(2.5 * n_cols, 2.8 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    col_labels = ["Original"] + [f"Iter {t+1}\ndil={records[0]['dilations'][t]}"
                                  for t in range(n_iters)]

    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=8)

    for row, rec in enumerate(records):
        img_rgb = unnormalize(rec["img"].squeeze(0))
        correct = rec["true"] == rec["pred"]

        # Original
        axes[row, 0].imshow(img_rgb, interpolation="nearest")
        axes[row, 0].set_ylabel(
            f"{CIFAR10_CLASSES[rec['true']]}\n→ {CIFAR10_CLASSES[rec['pred']]}",
            fontsize=7, color="green" if correct else "red",
            rotation=0, labelpad=60, va="center",
        )
        axes[row, 0].axis("off")

        # Heatmaps
        for t, cam in enumerate(rec["cams"]):
            axes[row, t + 1].imshow(img_rgb, interpolation="nearest")
            axes[row, t + 1].imshow(cam, cmap="jet", alpha=0.50, vmin=0, vmax=1)
            axes[row, t + 1].axis("off")

    fig.suptitle(
        "Grad-CAM Summary — ECTiedNet (weight-tied, varying dilation)",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary grid → {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for ECTiedNet")
    parser.add_argument(
        "--model", type=str, default="best_model_depth6_dil1-1-2-1-2-3.pth",
        help="Path to a trained model checkpoint (default: best_model_depth6_dil1-1-2-1-2-3.pth)",
    )
    parser.add_argument(
        "--iterations", type=int, default=6,
        help="Number of ECBlock iterations — must match the checkpoint (default: 6)",
    )
    parser.add_argument(
        "--channels", type=int, default=64,
        help="Base channel width — must match the checkpoint (default: 64)",
    )
    parser.add_argument(
        "--n-images", type=int, default=4,
        help="How many test images to visualize (default: 4)",
    )
    parser.add_argument(
        "--class-filter", type=int, default=None, metavar="[0-9]",
        help="Only visualize images of this CIFAR-10 class index",
    )
    parser.add_argument(
        "--only-incorrect", action="store_true",
        help="Only visualize misclassified images (useful for failure analysis)",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip the combined summary grid (gradcam_summary.png)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    try:
        model = ECTiedNet(
            num_classes=10,
            channels=args.channels,
            num_iterations=args.iterations,
        ).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        raise SystemExit(
            f"\nCheckpoint not found: {args.model}\n"
            "Train a model first with:\n"
            "    python train.py\n"
            "Then run gradcam.py again."
        )

    print(f"Loaded:    {args.model}")
    print(f"Dilations: {model.dilations}")
    if args.class_filter is not None:
        print(f"Filtering: class {args.class_filter} ({CIFAR10_CLASSES[args.class_filter]})")
    if args.only_incorrect:
        print("Mode:      misclassified images only")

    loader = get_test_loader(batch_size=1)
    records = []

    for idx, (image, label) in enumerate(loader):
        label_item = label.item()

        if args.class_filter is not None and label_item != args.class_filter:
            continue

        image = image.to(device)
        cams, pred_class = compute_gradcam(model, image, target_class=None)

        if args.only_incorrect and pred_class == label_item:
            continue

        # Per-image plot
        fname = f"gradcam_{len(records):02d}_{CIFAR10_CLASSES[label_item]}.png"
        plot_gradcam(image, cams, label_item, pred_class, model.dilations, fname)

        records.append({
            "img":      image.cpu(),
            "cams":     cams,
            "true":     label_item,
            "pred":     pred_class,
            "dilations": model.dilations,
        })

        if len(records) >= args.n_images:
            break

    if not records:
        print("No images matched the filters. Try relaxing --class-filter or --only-incorrect.")
        return

    # Summary grid
    if not args.no_summary:
        plot_summary_grid(records, save_path="gradcam_summary.png")

    print(f"\nDone. Visualized {len(records)} image(s).")


if __name__ == "__main__":
    main()

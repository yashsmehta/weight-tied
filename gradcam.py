"""
Grad-CAM visualization for ECTiedNet — ventral stream hierarchy.

Default: one heatmap per stage (Stage1/V1, Stage2/V4, Stage3/IT).
Shows how the same weight-tied circuit, applied at different resolutions
and dilation schedules, produces hierarchically different spatial attention.

With --all-iterations: one heatmap per ECBlock call (N1+N2+N3 total),
grouped by stage — useful for inspecting within-stage dilation effects.

Usage:
    python gradcam.py --checkpoint best_model_imagenet_iter4-4-4_....pth
    python gradcam.py --checkpoint ... --dataset imagenet-mini-50 --n-images 8
    python gradcam.py --checkpoint ... --all-iterations
    python gradcam.py --checkpoint ... --only-incorrect
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import ECTiedNet
from imagenet_mini_dataloader import get_dataloaders

STAGE_LABELS = ["Stage 1 (V1)", "Stage 2 (V4)", "Stage 3 (IT)"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CIFAR_MEAN    = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_STD     = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def unnormalize(tensor, dataset):
    mean, std = (CIFAR_MEAN, CIFAR_STD) if dataset == "cifar10" else (IMAGENET_MEAN, IMAGENET_STD)
    return (tensor.cpu() * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()


# ============================================================================
# Grad-CAM
# ============================================================================

def _cam_from_feat(feat, input_size):
    """Grad-CAM heatmap from a feature tensor with a retained gradient."""
    if feat.grad is None:
        return np.zeros(input_size)
    weights = feat.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    lo, hi = cam.min(), cam.max()
    return (cam - lo) / (hi - lo + 1e-8)


def compute_stage_gradcam(model, x, target_class=None):
    """One Grad-CAM heatmap per stage (V1 / V4 / IT).

    Calls model._forward_body, retains gradients on the three stage tensors,
    then does one backward pass.

    Returns:
        cams:       list of 3 heatmaps, each (H, W) in [0,1]
        pred_class: predicted class index
    """
    model.eval()
    model.zero_grad()
    input_size = x.shape[-2:]

    s1, s2, s3, feat = model._forward_body(x)
    for s in [s1, s2, s3]:
        s.retain_grad()

    logits = model.head(feat.mean(dim=(2, 3)))
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    logits[0, target_class].backward()

    return [_cam_from_feat(s, input_size) for s in [s1, s2, s3]], target_class


def compute_iteration_gradcam(model, x, target_class=None):
    """One Grad-CAM heatmap per ECBlock call (N1+N2+N3 total).

    Replays the forward pass explicitly to retain gradients at each iteration
    before the single backward call.

    Returns:
        cams:       list of N heatmaps, each (H, W) in [0,1]
        meta:       list of dicts with 'stage', 'iter', 'dilation' per heatmap
        pred_class: predicted class index
    """
    model.eval()
    model.zero_grad()
    input_size = x.shape[-2:]
    N1, N2, N3 = model.stage_iterations
    saved = []  # (tensor, stage_idx, iter_within_stage, dilation)
    t = 0

    feat = model.stem(x)

    for n_stage, n_iters in enumerate([N1, N2, N3]):
        for local_t in range(n_iters):
            feat = model.block(feat, dilation=model.dilations[t])
            feat.retain_grad()
            saved.append((feat, n_stage, local_t, model.dilations[t]))
            t += 1
        feat = model.blur(feat)

    logits = model.head(feat.mean(dim=(2, 3)))
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    logits[0, target_class].backward()

    cams = [_cam_from_feat(f, input_size) for f, *_ in saved]
    meta = [{"stage": s + 1, "iter": li + 1, "dilation": d} for _, s, li, d in saved]
    return cams, meta, target_class


# ============================================================================
# Plotting
# ============================================================================

def plot_stage_gradcam(img_tensor, cams, pred_class, dataset, save_path):
    """Original image + one heatmap per stage (3 columns)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    img = unnormalize(img_tensor.squeeze(0), dataset)

    axes[0].imshow(img)
    axes[0].set_title(f"Original\nPred class: {pred_class}", fontsize=10)
    axes[0].axis("off")

    for ax, cam, label in zip(axes[1:], cams, STAGE_LABELS):
        ax.imshow(img)
        im = ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Importance")
    fig.suptitle(
        "Grad-CAM — same weight-tied ECBlock at different stages → different spatial attention",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_iteration_gradcam(img_tensor, cams, meta, pred_class, dataset, save_path):
    """Original image + one heatmap per ECBlock iteration, grouped by stage."""
    n_cols = len(cams) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 4.5))
    img = unnormalize(img_tensor.squeeze(0), dataset)

    axes[0].imshow(img)
    axes[0].set_title(f"Original\nPred: {pred_class}", fontsize=9)
    axes[0].axis("off")

    stage_colors = ["#1f77b4", "#2ca02c", "#d62728"]  # blue/green/red per stage
    for ax, cam, m in zip(axes[1:], cams, meta):
        ax.imshow(img)
        im = ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        color = stage_colors[m["stage"] - 1]
        ax.set_title(f"S{m['stage']} iter{m['iter']}\ndil={m['dilation']}", fontsize=8, color=color)
        ax.axis("off")

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_summary_grid(records, dataset, save_path="gradcam_summary.png", all_iterations=False):
    """Rows = images, columns = stages (or iterations if all_iterations)."""
    n_images = len(records)
    n_cams   = len(records[0]["cams"])
    n_cols   = n_cams + 1

    fig, axes = plt.subplots(n_images, n_cols, figsize=(2.8 * n_cols, 3.2 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    if all_iterations:
        col_labels = ["Original"] + [
            f"S{m['stage']} it{m['iter']}\ndil={m['dilation']}"
            for m in records[0]["meta"]
        ]
    else:
        col_labels = ["Original"] + STAGE_LABELS

    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=8)

    for row, rec in enumerate(records):
        img = unnormalize(rec["img"].squeeze(0), dataset)
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"Pred: {rec['pred']}", fontsize=7, rotation=0,
                                labelpad=55, va="center")
        axes[row, 0].axis("off")
        for col, cam in enumerate(rec["cams"]):
            axes[row, col + 1].imshow(img)
            axes[row, col + 1].imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[row, col + 1].axis("off")

    fig.suptitle("Grad-CAM Summary — ECTiedNet ventral stream stages", fontsize=11, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for ECTiedNet")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model_*.pth or checkpoint_*.pth")
    parser.add_argument("--dataset", default="imagenet",
                        choices=["imagenet", "imagenet-mini-50", "cifar10"])
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--stage-iterations", type=int, nargs=3, default=[4, 4, 4],
                        metavar=("N1", "N2", "N3"))
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dilations", type=int, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=4,
                        help="Number of images to visualize (default: 4)")
    parser.add_argument("--all-iterations", action="store_true",
                        help="Show one heatmap per ECBlock iteration instead of per stage")
    parser.add_argument("--only-incorrect", action="store_true",
                        help="Only visualize misclassified images")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip the combined summary grid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
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
    model.eval()
    print(f"Loaded:           {args.checkpoint}")
    print(f"Dilations:        {model.dilations}")
    print(f"Mode:             {'all iterations' if args.all_iterations else 'per stage (V1/V4/IT)'}")

    _, test_loader = get_dataloaders(dataset=args.dataset, batch_size=1)
    records = []

    for img, label in test_loader:
        img   = img.to(device)
        label = label.item()

        if args.all_iterations:
            cams, meta, pred = compute_iteration_gradcam(model, img)
            if args.only_incorrect and pred == label:
                continue
            fname = f"gradcam_{len(records):02d}_class{label}.png"
            plot_iteration_gradcam(img, cams, meta, pred, args.dataset, fname)
            records.append({"img": img.cpu(), "cams": cams, "meta": meta,
                            "true": label, "pred": pred})
        else:
            cams, pred = compute_stage_gradcam(model, img)
            if args.only_incorrect and pred == label:
                continue
            fname = f"gradcam_{len(records):02d}_class{label}.png"
            plot_stage_gradcam(img, cams, pred, args.dataset, fname)
            records.append({"img": img.cpu(), "cams": cams, "meta": None,
                            "true": label, "pred": pred})

        if len(records) >= args.n_images:
            break

    if not records:
        print("No images matched the filters.")
        return

    if not args.no_summary:
        plot_summary_grid(records, args.dataset, all_iterations=args.all_iterations)

    print(f"\nDone. Visualized {len(records)} image(s).")


if __name__ == "__main__":
    main()

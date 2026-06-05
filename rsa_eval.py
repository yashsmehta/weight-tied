"""
RSA evaluation: brain-model alignment for ECTiedNet.

Extracts pre-classifier (post-GAP) activations, builds a model RDM using
1 - Pearson correlation (same as visreps), then scores against a brain RDM
using Spearman correlation.

Usage:
    # Save model RDM now; score later when brain data arrives
    python rsa_eval.py --checkpoint best_model_imagenet-mini-50_....pth

    # Full RSA with brain data
    python rsa_eval.py --checkpoint best_model_....pth --brain-data neural.npz

Brain data format (.npz):
    responses : float32 array (N, V)  — N stimuli × V voxels
                stimuli must be in the same order as the dataset split used here
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats

from model import ECTiedNet
from imagenet_mini_dataloader import get_dataloaders

DATASET_NUM_CLASSES = {"cifar10": 10, "imagenet-mini-50": 1000}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device):
    """Returns (N, C) numpy array of post-GAP activations and (N,) labels."""
    model.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        feats.append(model.extract_features(imgs.to(device)).cpu())
        labels.append(lbls)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


# ---------------------------------------------------------------------------
# RSA (mirrors visreps/analysis/rsa.py)
# ---------------------------------------------------------------------------

def compute_rdm(X):
    """RDM as 1 - Pearson correlation. X: (N, D) float64."""
    rdm = 1.0 - np.corrcoef(X.astype(np.float64))
    np.fill_diagonal(rdm, 0.0)
    return rdm


def compute_rsa(rdm_model, rdm_brain):
    """Spearman correlation of upper-triangular entries."""
    idx = np.triu_indices(rdm_model.shape[0], k=1)
    return stats.spearmanr(rdm_model[idx], rdm_brain[idx])


def bootstrap_rsa(rdm_model, rdm_brain, n_bootstrap=1000, seed=42):
    """90% bootstrap CI over stimuli (rows/cols resampled together)."""
    rng = np.random.default_rng(seed)
    n = rdm_model.shape[0]
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = compute_rsa(rdm_model[np.ix_(idx, idx)], rdm_brain[np.ix_(idx, idx)])
        scores.append(r)
    scores = np.array(scores)
    return float(np.percentile(scores, 5)), float(np.percentile(scores, 95))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args, device):
    model = ECTiedNet(
        num_classes=DATASET_NUM_CLASSES[args.dataset],
        channels=args.channels,
        expansion=args.expansion,
        num_iterations=args.iterations,
        dilations=args.dilations,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model_*.pth or checkpoint_*.pth")
    parser.add_argument("--brain-data", default=None,
                        help=".npz with 'responses' (N, V), stimuli in dataset order")
    parser.add_argument("--dataset", default="imagenet-mini-50",
                        choices=list(DATASET_NUM_CLASSES))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dilations", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", action="store_true",
                        help="Compute 90%% bootstrap CI (n=1000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    print(f"Loaded: {args.checkpoint}")

    train_loader, test_loader = get_dataloaders(dataset=args.dataset, batch_size=args.batch_size)
    loader = test_loader if args.split == "test" else train_loader
    n_imgs = len(loader.dataset)

    print(f"Extracting features ({n_imgs} images, {args.split} split)...")
    feats, labels = extract_features(model, loader, device)
    print(f"Features: {feats.shape}")

    print("Computing model RDM (1 - Pearson)...")
    model_rdm = compute_rdm(feats)

    stem = Path(args.checkpoint).stem
    np.save(f"{stem}_rdm.npy", model_rdm)
    np.save(f"{stem}_labels.npy", labels)
    print(f"Saved {stem}_rdm.npy  ({model_rdm.shape[0]}×{model_rdm.shape[0]})")

    if args.brain_data is None:
        print("\nNo brain data provided — re-run with --brain-data <path> when available.")
        return

    print(f"\nLoading brain data: {args.brain_data}")
    brain = np.load(args.brain_data)
    responses = brain["responses"].astype(np.float64)  # (N, V)
    print(f"Brain responses: {responses.shape}")

    n = min(len(feats), len(responses))
    if len(feats) != len(responses):
        print(f"Warning: stimulus count mismatch ({len(feats)} vs {len(responses)}), using {n}")

    brain_rdm = compute_rdm(responses[:n])
    r, p = compute_rsa(model_rdm[:n, :n], brain_rdm)
    print(f"RSA (Spearman r): {r:.4f}  p={p:.3e}")

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "split": args.split,
        "n_stimuli": n,
        "rsa_spearman_r": float(r),
        "p_value": float(p),
    }

    if args.bootstrap:
        ci_lo, ci_hi = bootstrap_rsa(model_rdm[:n, :n], brain_rdm)
        results["ci_90"] = [ci_lo, ci_hi]
        print(f"90% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    out = f"{stem}_rsa.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

"""
RSA evaluation: ventral stream alignment for ECTiedNet.

Extracts per-stage post-GAP representations (Stage1, Stage2, Stage3),
builds one RDM per stage, then scores each against brain ROI responses
(V1, V4, IT) via Spearman correlation.

The core result is a 3×3 RSA matrix (model stages × brain ROIs). If the
weight-tied hierarchy tracks the ventral stream, the diagonal should dominate:
  Stage1 ↔ V1 > Stage1 ↔ V4, IT
  Stage2 ↔ V4 > Stage2 ↔ V1, IT
  Stage3 ↔ IT > Stage3 ↔ V1, V4

Brain data .npz format:
    v1 : float32 (N, V1_voxels)  — V1 responses, N stimuli in dataset order
    v4 : float32 (N, V4_voxels)  — V4 responses
    it : float32 (N, IT_voxels)  — IT/HVC responses
    (include any subset; missing ROIs are skipped)

Usage:
    # Save per-stage RDMs now; score later when brain data arrives
    python rsa_eval.py --checkpoint best_model_imagenet_iter4-4-4_....pth

    # Full 3×3 RSA matrix against V1/V4/IT
    python rsa_eval.py --checkpoint best_model_....pth --brain-data nsd_responses.npz

    # With 90% bootstrap CIs
    python rsa_eval.py --checkpoint ... --brain-data ... --bootstrap
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats

from model import ECTiedNet
from imagenet_mini_dataloader import get_dataloaders

DATASET_NUM_CLASSES = {"cifar10": 10, "imagenet-mini-50": 1000, "imagenet": 1000}

# Stage index → model stage name → expected brain area
STAGE_NAMES = ["stage1", "stage2", "stage3"]
STAGE_AREAS = ["V1",     "V4",     "IT"]
ROI_KEYS    = ["v1",     "v4",     "it"]   # expected keys in brain .npz


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_stage_features(model, loader, device):
    """Extract per-stage post-GAP features for all images.

    Returns:
        stage_dict: {"stage1": (N,C1), "stage2": (N,C2), "stage3": (N,C3)} numpy arrays
        labels:     (N,) numpy array
    """
    model.eval()
    stage_bufs = [[] for _ in range(3)]
    label_buf  = []
    for imgs, lbls in loader:
        per_stage = model.extract_stage_features(imgs.to(device))  # list of 3 tensors
        for i, feat in enumerate(per_stage):
            stage_bufs[i].append(feat.cpu())
        label_buf.append(lbls)
    labels = torch.cat(label_buf).numpy()
    stage_dict = {
        name: torch.cat(buf).numpy()
        for name, buf in zip(STAGE_NAMES, stage_bufs)
    }
    return stage_dict, labels


# ---------------------------------------------------------------------------
# RSA
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
        stage_iterations=tuple(args.stage_iterations),
        expansion=args.expansion,
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
                        help=".npz with keys 'v1', 'v4', 'it' — each (N, voxels)")
    parser.add_argument("--dataset", default="imagenet",
                        choices=list(DATASET_NUM_CLASSES))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--stage-iterations", type=int, nargs=3, default=[4, 4, 4],
                        metavar=("N1", "N2", "N3"))
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dilations", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", action="store_true",
                        help="Compute 90%% bootstrap CI per RSA score (n=1000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    print(f"Loaded: {args.checkpoint}")

    train_loader, test_loader = get_dataloaders(dataset=args.dataset, batch_size=args.batch_size)
    loader = test_loader if args.split == "test" else train_loader
    n_imgs = len(loader.dataset)

    print(f"Extracting per-stage features ({n_imgs} images, {args.split} split)...")
    stage_dict, labels = extract_stage_features(model, loader, device)
    for name, area, feats in zip(STAGE_NAMES, STAGE_AREAS, stage_dict.values()):
        print(f"  {name} ({area}): {feats.shape}")

    # Compute and save one RDM per stage
    stem = Path(args.checkpoint).stem
    stage_rdms = {}
    for name, feats in stage_dict.items():
        print(f"Computing RDM for {name}...")
        rdm = compute_rdm(feats)
        stage_rdms[name] = rdm
        rdm_path = f"{stem}_{name}_rdm.npy"
        np.save(rdm_path, rdm)
        print(f"  Saved {rdm_path}  ({rdm.shape[0]}×{rdm.shape[0]})")
    np.save(f"{stem}_labels.npy", labels)

    if args.brain_data is None:
        print("\nNo brain data provided — re-run with --brain-data <path> when available.")
        print(f"Expected .npz keys: {ROI_KEYS}  (one array per brain ROI, shape: N × voxels)")
        return

    # Load brain data
    print(f"\nLoading brain data: {args.brain_data}")
    brain = np.load(args.brain_data)
    available_rois = [k for k in ROI_KEYS if k in brain]
    if not available_rois:
        raise ValueError(
            f"No expected ROI keys found. Got: {list(brain.keys())}. "
            f"Expected one or more of: {ROI_KEYS}"
        )
    print(f"Found ROIs: {[r.upper() for r in available_rois]}")

    # Compute brain RDMs, align stimulus count
    brain_rdms = {}
    n_stimuli = len(labels)
    for roi_key in available_rois:
        responses = brain[roi_key].astype(np.float64)  # (N, voxels)
        n = min(n_stimuli, len(responses))
        if len(responses) != n_stimuli:
            print(f"  Warning: {roi_key.upper()} stimulus count mismatch "
                  f"({len(responses)} vs {n_stimuli}), truncating to {n}")
        brain_rdms[roi_key] = compute_rdm(responses[:n])
        n_stimuli = n  # align all ROIs to the same N
        print(f"  {roi_key.upper()}: {responses.shape} → RDM {n}×{n}")

    # Print and compute full RSA matrix: model stages × brain ROIs
    print(f"\nRSA matrix (Spearman r), n={n_stimuli} stimuli:")
    roi_labels = [r.upper() for r in available_rois]
    col_w = 22 if args.bootstrap else 10
    print(f"{'':16s}" + "".join(f"{r:>{col_w}s}" for r in roi_labels))

    rsa_matrix = {}
    for stage_name, area_name in zip(STAGE_NAMES, STAGE_AREAS):
        rdm_model = stage_rdms[stage_name][:n_stimuli, :n_stimuli]
        row = {}
        row_str = f"{stage_name} ({area_name}):  "
        for roi_key in available_rois:
            rdm_brain = brain_rdms[roi_key][:n_stimuli, :n_stimuli]
            r, p = compute_rsa(rdm_model, rdm_brain)
            entry = {"r": float(r), "p": float(p)}
            if args.bootstrap:
                ci_lo, ci_hi = bootstrap_rsa(rdm_model, rdm_brain)
                entry["ci_90"] = [ci_lo, ci_hi]
                row_str += f"  {r:.3f}[{ci_lo:.3f},{ci_hi:.3f}]"
            else:
                row_str += f"  {r:>8.4f}  "
            row[roi_key] = entry
        print(row_str)
        rsa_matrix[stage_name] = row

    # Save results
    results = {
        "checkpoint":      args.checkpoint,
        "dataset":         args.dataset,
        "split":           args.split,
        "n_stimuli":       n_stimuli,
        "stage_to_area":   dict(zip(STAGE_NAMES, STAGE_AREAS)),
        "rsa_matrix":      rsa_matrix,
    }
    out = f"{stem}_rsa.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

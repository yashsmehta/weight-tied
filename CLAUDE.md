# Weight-Tied ECNet

## Project overview
CNN for ImageNet with a single weight-tied circuit modeling the ventral stream (V1 → V4 → IT).
Core claim: one shared ECBlock applied iteratively at three spatial scales produces representations
more aligned with the ventral stream than untied equivalents, with 4.6× fewer parameters.
~420K params (default). Untied equivalent: ~1.95M.

## Architecture
ONE shared ECBlock, reused across three resolution levels:
```
Stem (stride-4) → [N1 iters at 56×56] → BlurPool → [N2 iters at 28×28] → BlurPool
                → [N3 iters at 14×14] → BlurPool → 7×7 → GAP → head
```
Total downsampling: 4 (stem) × 2 × 2 × 2 = **32×** → 7×7 before GAP (same as ResNet/EfficientNet).
Theoretical receptive field: **511px** >> 224px input (full image coverage at any spatial position).
Default: channels=128, stage_iterations=(4,4,4), 12 total ECBlock calls.

Key components:
- **ONE ECBlock** (`model.block`): inverted bottleneck — expand (1×1) → dw 3×3 at runtime dilation → DivisiveNorm → GELU → contract (1×1), + per-channel gamma residual
- **DivisiveNorm**: biological gain control, y = x / (eps + avg(|x|)) — Carandini & Heeger 2012
- **ONE BlurPool** (`model.blur`): stateless, reused 3× (56→28, 28→14, 14→7) — Zhang 2019
- **Dilation schedule**: [1,2,3,2] cycling continuously across all 12 iterations
- **Per-channel gamma**: learnable residual scale (ConvNeXt-style), init=1e-6

Stage-to-brain-area mapping (for RSA):
- Stage 1 = end of N1 iters at 56×56 → **V1**
- Stage 2 = end of N2 iters at 28×28 → **V4**
- Stage 3 = end of N3 iters at 14×14 → **IT**

## Training defaults
- AdamW, weight decay 0.01, cosine annealing LR
- Batch 32, 200 epochs. ImageNet LR: tune via sweep (1e-3 is a good starting point).
- Batch size is constrained by activation memory, not parameter count. At stage 1 (56×56,
  512 hidden channels), backprop must retain ~7 tensors × 822 MB × 4 iterations ≈ 23 GB
  for batch=128 — exceeding the 4090's 24 GB. Batch=32 uses ~7 GB. Batch=64 (~14 GB) also
  fits with headroom.
- `num_workers=4`, `pin_memory=True` in DataLoader (default from `get_dataloaders` in `imagenet_mini_dataloader.py`).

## CLI args (train.py)
- `--dataset imagenet|imagenet-mini-50|cifar10` (default: cifar10)
- `--channels C` — single channel width shared across all iterations (default: 128)
- `--stage-iterations N1 N2 N3` — ECBlock iterations per stage (default: 4 4 4)
- `--dilations D D D D` — dilation schedule, cycles continuously (default: 1 2 3 2)
- `--expansion N` — expansion ratio in ECBlock (default: 4)
- `--lr F`, `--epochs N`, `--batch-size N`
- `--optimizer adamw|sgd` (default: adamw)
- `--weight-decay F` (default: 0.01)
- `--resume checkpoint_....pth`
- `--checkpoint-every N` (default: 10)

## Run name format
`{dataset}_ch{C}_iter{N1}-{N2}-{N3}_dil{schedule}_{optimizer}_lr{lr}`
e.g. `imagenet_ch128_iter4-4-4_dil1-2-3-2_adamw_lr1e-3`

## Output files
- `best_model_{run_name}.pth` — best model weights
- `checkpoint_{run_name}.pth` — full resumable state
- `history_{run_name}.json` — per-epoch train/test loss and accuracy
- `training_curves_{run_name}.png` — accuracy and loss plots

## Experiment sweeps (sweep.py)
```bash
python sweep.py                              # full sweep on imagenet (both phases)
python sweep.py --phase 1                    # LR search only (iters=4-4-4)
python sweep.py --phase 2 --lr 1e-3         # iteration-depth sweep with fixed LR
python sweep.py --dataset imagenet-mini-50 --epochs 50  # quick smoke-test
python sweep.py --dry-run                    # print commands without executing
```
Phase 1: LR search at fixed iters=(4,4,4). Tests: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2.
Phase 2: Iteration sweep at best LR. Tests (N,N,N) for N in [2,3,4,6] = 6/9/12/18 total.
Both phases skip runs whose history_{run_name}.json already exists (safe to re-run).

## Key ablation commands
```bash
# Dilation ablations
python train.py --dataset imagenet --dilations 1 1 1 1    # no dilation baseline
python train.py --dataset imagenet --dilations 1 2 3 2    # default
python train.py --dataset imagenet --dilations 3 2 1 2    # reversed schedule

# Stage-skewed configs (for ventral stream depth analysis)
python train.py --dataset imagenet --stage-iterations 6 4 2  # V1-heavy
python train.py --dataset imagenet --stage-iterations 2 4 6  # IT-heavy

# Narrower model (ablation: param count effects)
python train.py --dataset imagenet --channels 64
```

## Grad-CAM visualization (gradcam.py)
```bash
# Per-stage heatmaps (default): Original + Stage1/V1 + Stage2/V4 + Stage3/IT
python gradcam.py --checkpoint best_model_imagenet_ch128_iter4-4-4_....pth

# Per-iteration heatmaps (N1+N2+N3 total, grouped by stage)
python gradcam.py --checkpoint ... --all-iterations

# Failure analysis
python gradcam.py --checkpoint ... --dataset imagenet-mini-50 --only-incorrect --n-images 8
```
Stage-level CAM: calls `model._forward_body()`, retains grad on s1/s2/s3, one backward pass.
Iteration-level CAM: replays `model.block` calls with `retain_grad()` per iteration, one backward pass.
Outputs: `gradcam_NN_classK.png` per image + `gradcam_summary.png` grid.
Stage colors in `--all-iterations` mode: blue=S1, green=S2, red=S3.
All CLI flags (`--channels`, `--stage-iterations`, `--dataset`, etc.) mirror train.py.

## PCA visualization (pca_plot.py)
```bash
# 3-panel PCA: Stage1 vs Stage2 vs Stage3 class separation
python pca_plot.py --checkpoint best_model_imagenet_ch128_iter4-4-4_....pth

# Show 20 randomly sampled classes (useful for ImageNet's 1000 classes)
python pca_plot.py --checkpoint ... --n-classes 20

# Quick run on smaller dataset
python pca_plot.py --checkpoint ... --dataset imagenet-mini-50 --n-classes 15
```
Output: `pca_stages.png` — three panels with 2D PCA scatter + silhouette/CH scores.
Metrics computed on all samples; only `--n-classes` classes plotted for readability.
If the ventral stream hierarchy holds, silhouette and CH should increase Stage1 → Stage3.

## RSA evaluation (rsa_eval.py)
```bash
# Save per-stage RDMs now; score against brain data when available
python rsa_eval.py --checkpoint best_model_imagenet_ch128_iter4-4-4_....pth

# Full 3×3 RSA matrix against V1/V4/IT
python rsa_eval.py --checkpoint best_model_....pth --brain-data nsd_responses.npz

# With 90% bootstrap CIs
python rsa_eval.py --checkpoint ... --brain-data nsd_responses.npz --bootstrap
```
Brain data `.npz` format — one array per ROI, each `(N_stimuli, N_voxels)`:
```
v1 : float32 (N, V1_voxels)
v4 : float32 (N, V4_voxels)
it : float32 (N, IT_voxels)
```
Any subset of ROI keys works; missing ones are skipped. Stimuli must be in the same
order as the dataset split used (test split, seed=42).

Outputs per run:
- `{stem}_stage1_rdm.npy`, `_stage2_rdm.npy`, `_stage3_rdm.npy` — one RDM per stage (reusable)
- `{stem}_labels.npy` — class labels
- `{stem}_rsa.json` — full 3×3 RSA matrix, e.g.:
```json
{
  "stage1": {"v1": {"r": 0.51, "p": 1e-8}, "v4": {"r": 0.33}, "it": {"r": 0.21}},
  "stage2": {"v1": {"r": 0.38}, "v4": {"r": 0.55}, "it": {"r": 0.40}},
  "stage3": {"v1": {"r": 0.24}, "v4": {"r": 0.41}, "it": {"r": 0.63}}
}
```
The diagonal dominating (Stage1↔V1 > Stage1↔V4/IT, etc.) is the ventral stream hierarchy claim.
All CLI flags (`--channels`, `--stage-iterations`, `--dataset`, `--split`) mirror train.py.

## Resuming interrupted training
```bash
python train.py --resume checkpoint_imagenet_ch128_iter4-4-4_dil1-2-3-2_adamw_lr1e-3.pth
```

## Datasets
| Dataset | Classes | Images/class | Path (server) |
|---|---|---|---|
| `cifar10` | 10 | 5000 train / 1000 test | auto-downloaded |
| `imagenet-mini-50` | 1000 | 50 | `/data/shared/datasets/imagenet-mini-50` |
| `imagenet` | 1000 | ~1200 | `/data/shared/datasets/imagenet` |

ImageNet datasets use `/home/rsingh55/folder_labels.json` for WordNet ID → label mapping.
80/20 train/test split (seed=42).

## File layout
- `model.py` — ECTiedNet, ECBlock, DivisiveNorm, BlurPool2d
- `train.py` — CLI training script
- `sweep.py` — two-phase LR + iteration-depth sweep
- `imagenet_mini_dataloader.py` — dataloader for all three datasets
- `rsa_eval.py` — RSA brain-model alignment evaluation
- `gradcam.py` — Grad-CAM per stage (V1/V4/IT) or per iteration (`--all-iterations`)
- `pca_plot.py` — 3-panel PCA of stage activations with silhouette/CH scores
- `data/` — CIFAR-10 (auto-downloaded)

## Environment
- macOS CPU locally for testing; Colab/server GPU for full training
- Dependencies: torch, torchvision, matplotlib, scipy, scikit-learn

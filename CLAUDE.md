# Weight-Tied ECNet

## Project overview
Parameter-efficient CNN that reuses one convolutional block (ECBlock) multiple times with different dilation rates, trained on CIFAR-10. ~115K parameters.

## Architecture
- Stem (3x3 conv) -> ECBlock x6 (weight-tied, varying dilation) -> BlurPool -> GAP -> Linear
- Dilation schedule: [1, 1, 2, 1, 2, 3]
- Uses DivisiveNorm (bio-inspired) and BlurPool (anti-aliased downsampling)

## Training defaults
- AdamW, weight decay 0.01, cosine annealing LR
- Batch size 256, standard CIFAR-10 augmentation (RandomCrop + HFlip)
- (Legacy SGD: momentum 0.9, weight decay 5e-4; use --optimizer sgd to restore)

## CLI args (train.py)
- `--optimizer adamw|sgd` — optimizer choice (default: adamw)
- `--weight-decay F` — weight decay (default: 0.01 for AdamW, 5e-4 for SGD)
- `--dilations 1 1 2 1 2 3` — dilation schedule (space-separated, cycles if depth > len)
- `--resume checkpoint_....pth` — resume from checkpoint
- `--checkpoint-every 10` — save resumable checkpoint every N epochs (default: 10)
- `--iterations N` — number of ECBlock applications (depth)
- `--epochs N`, `--lr F`, `--channels N`, `--expansion N`, `--batch-size N`

## Run name format
`{dataset}_depth{N}_dil{schedule}_{optimizer}_lr{lr}` — e.g. `cifar10_depth4_dil1-1-2-1-2-3_adamw_lr1e-3`

## Output files
All outputs are named by run: `depth{N}_dil{schedule}` e.g. `depth6_dil1-1-2-1-2-3`
- `best_model_{run_name}.pth` — best model weights only
- `checkpoint_{run_name}.pth` — full resumable state (model + optimizer + scheduler + epoch + history)
- `history_{run_name}.json` — per-epoch train/test loss and accuracy
- `training_curves_{run_name}.png` — accuracy and loss plots

## Experiment sweeps (sweep.py)
```bash
python sweep.py                        # full sweep: phase 1 (LR search) then phase 2 (depth sweep)
python sweep.py --phase 1              # LR search at depth=4 only
python sweep.py --phase 2 --lr 3e-3   # depth sweep with a known best LR
python sweep.py --epochs 50            # quick smoke-test (50 epochs per run)
python sweep.py --dry-run              # print commands without executing
```
Phase 1 searches LRs: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 at depth=4.
Phase 2 sweeps depths: 1, 2, 4, 8, 12 at the best LR from phase 1.
Both phases skip runs whose history_{run_name}.json already exists (safe to re-run).

## Dilation ablation commands (50 epochs for quick comparison)
```bash
python train.py --epochs 50 --dilations 1 1 1 1 1 1   # no dilation baseline
python train.py --epochs 50 --dilations 1 2 3 1 2 3   # cycling, no gap-filling
python train.py --epochs 50                            # original [1,1,2,1,2,3]
python train.py --epochs 50 --dilations 3 2 1 2 1 1   # reversed
```

## Resuming interrupted training
```bash
python train.py --resume checkpoint_depth6_dil1-1-2-1-2-3.pth
```
Checkpoint saves model, optimizer, scheduler, epoch, best_acc, and full history.
Resumes from the next epoch with LR schedule intact.

## Datasets
| Dataset | Classes | Images/class | Path (server) |
|---|---|---|---|
| `cifar10` | 10 | 5000 train / 1000 test | auto-downloaded |
| `imagenet-mini-50` | 1000 | 50 | `/data/shared/datasets/imagenet-mini-50` |
| `imagenet` | 1000 | ~1200 | `/data/shared/datasets/imagenet` |

All ImageNet datasets use `/home/rsingh55/folder_labels.json` to map folder names (WordNet IDs) to integer labels. ImageNet-mini-50 and ImageNet use a deterministic 80/20 train/test split (seed=42).

## File layout
- `model.py` — ECTiedNet, ECBlock, DivisiveNorm, BlurPool
- `train.py` — CLI training script
- `imagenet_mini_dataloader.py` — dataloader for all three datasets
- `rsa_eval.py` — RSA brain-model alignment evaluation
- `weight_tied_ecnet.ipynb` — interactive notebook (mirrors train.py)
- `data/` — CIFAR-10 (auto-downloaded)

## Environment
- macOS, typically runs on CPU locally (use Colab for GPU)
- Dependencies: torch, torchvision, matplotlib, scipy

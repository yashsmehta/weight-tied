# Weight-Tied ECNet

## Project overview
Parameter-efficient CNN that reuses one convolutional block (ECBlock) multiple times with different dilation rates, trained on CIFAR-10. ~115K parameters.

## Architecture
- Stem (3x3 conv) -> ECBlock x6 (weight-tied, varying dilation) -> BlurPool -> GAP -> Linear
- Dilation schedule: [1, 1, 2, 1, 2, 3]
- Uses DivisiveNorm (bio-inspired) and BlurPool (anti-aliased downsampling)

## Training defaults
- SGD with momentum 0.9, weight decay 5e-4
- Cosine annealing LR from 0.1 over 200 epochs
- Batch size 128, standard CIFAR-10 augmentation (RandomCrop + HFlip)

## File layout
- `model.py` — ECTiedNet, ECBlock, DivisiveNorm, BlurPool
- `train.py` — CLI training script
- `weight_tied_ecnet.ipynb` — interactive notebook (mirrors train.py)
- `data/` — CIFAR-10 (auto-downloaded)
- `best_model.pth` — saved checkpoint

## Environment
- macOS, typically runs on CPU locally (use Colab for GPU)
- Dependencies: torch, torchvision, matplotlib

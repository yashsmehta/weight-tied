# Weight-Tied ECNet

A parameter-efficient CNN that reuses the same convolutional block multiple times, trained on CIFAR-10.

## What is this?

Most CNNs stack many different layers, each with their own weights. This network takes a different approach: **one block, used repeatedly**.

The trick: each time the block is reused, it processes the image at a different scale (via dilation). This captures both fine details and broad context without adding parameters.

```
Image → Stem → Block → Block → Block → ... → Classifier
                 ↑________↑________↑
                   Same weights!
                   Different scales
```

**Result**: ~115K parameters (vs millions in typical CNNs) while still achieving competitive accuracy.

## Quick Start

```bash
pip install torch torchvision
python train.py
```

CIFAR-10 downloads automatically. Training logs accuracy each epoch and saves the best model.

## Configuration

```bash
python train.py --channels 96 --iterations 8 --epochs 300
```

| Option | Default | What it does |
|--------|---------|--------------|
| `--channels` | 64 | Width of the network (more = bigger model) |
| `--iterations` | 6 | How many times to reuse the block |
| `--expansion` | 4 | Internal expansion factor in each block |
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 0.1 | Initial learning rate |

## How it works

**The Block (ECBlock)**: Expand → Process → Contract
1. Expand channels with 1x1 conv
2. Apply 3x3 depthwise conv with configurable dilation
3. Contract back to original channels
4. Add residual connection

**Dilation Schedule**: `[1, 1, 2, 1, 2, 3]`
- Early iterations use dilation=1 (local features)
- Later iterations use dilation=2,3 (larger receptive field)

**Other components**:
- *DivisiveNorm*: Normalizes by local neighborhood magnitude (inspired by visual cortex)
- *BlurPool*: Anti-aliased downsampling for better shift invariance

## Files

| File | Description |
|------|-------------|
| `model.py` | Network architecture (ECTiedNet, ECBlock, etc.) |
| `train.py` | Training loop with CIFAR-10 |
| `requirements.txt` | Dependencies |

## References

- [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486) (BlurPool)
- Divisive normalization is inspired by gain control mechanisms in biological vision

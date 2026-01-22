# Weight-Tied ECNet for CIFAR-10

A minimal implementation of a weight-tied Expansion-Contraction CNN trained on CIFAR-10.

## Key Idea

Traditional CNNs have separate weights for each layer. This network **reuses the same block** multiple times with different dilation rates, dramatically reducing parameters while maintaining expressiveness.

```
Input -> Stem -> [ECBlock x N] -> Global Pool -> Classifier
                    ^
                    |
            Same weights reused N times
            with different dilations
```

## Architecture Components

| Component | Purpose |
|-----------|---------|
| **ECBlock** | Expand (1x1) -> Depthwise 3x3 -> Contract (1x1) with residual |
| **DivisiveNorm** | Biologically-inspired local gain control |
| **BlurPool** | Anti-aliased downsampling to improve shift invariance |
| **Weight Tying** | Same block applied N times with dilation schedule [1,1,2,1,2,3] |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train with defaults
python train.py

# Custom configuration
python train.py --channels 96 --iterations 8 --epochs 300
```

## Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 0.1 | Learning rate |
| `--channels` | 64 | Base channel width |
| `--iterations` | 6 | Number of block reuses |
| `--expansion` | 4 | Expansion ratio in ECBlock |

## Files

- `model.py` - ECTiedNet architecture
- `train.py` - Training script with CIFAR-10
- `requirements.txt` - Dependencies

## References

- BlurPool: [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)
- Divisive Normalization: Inspired by visual cortex gain control mechanisms

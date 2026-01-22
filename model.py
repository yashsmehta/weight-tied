"""
Weight-Tied ECNet for CIFAR-10

Key idea: Instead of having separate weights per layer, we reuse ONE block
multiple times with different dilation rates. This reduces parameters while
capturing multi-scale features through varying receptive fields.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Utility
# ============================================================================

def gn_groups(channels: int, max_groups: int = 16) -> int:
    """Find largest divisor of channels <= max_groups for GroupNorm."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ============================================================================
# Building Blocks
# ============================================================================

class DivisiveNorm(nn.Module):
    """
    Biologically-inspired normalization mimicking visual cortex gain control.

    Formula: y = x / (eps + local_avg(|x|))

    This normalizes each activation by the average magnitude of its neighbors,
    making the network more robust to contrast variations.
    """
    def __init__(self, kernel_size: int = 3, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (self.pool(x.abs()) + self.eps)


class BlurPool2d(nn.Module):
    """
    Anti-aliased downsampling using a fixed low-pass filter.

    Standard strided convolutions can cause aliasing artifacts. BlurPool
    applies a blur before downsampling to preserve shift-invariance.

    Uses binomial kernel [1,2,1] x [1,2,1] (normalized).
    Reference: "Making Convolutional Networks Shift-Invariant Again"
    """
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.groups = channels

        # Create 2D binomial kernel via outer product
        k1d = torch.tensor([1., 2., 1.])
        k2d = torch.einsum('i,j->ij', k1d, k1d)  # [1,2,1] x [1,2,1]
        k2d = k2d / k2d.sum()  # Normalize to sum=1

        # Shape: (channels, 1, 3, 3) for depthwise conv
        self.register_buffer('kernel', k2d[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=self.groups)


class ECBlock(nn.Module):
    """
    Expansion-Contraction block (inverted bottleneck with weight reuse).

    Structure:
        1. Expand:   1x1 conv (C -> C*expansion)
        2. Process:  Depthwise 3x3 with configurable dilation
        3. Contract: 1x1 conv (C*expansion -> C)
        4. Residual: output = input + gamma * processed

    The dilation parameter allows the SAME weights to capture different
    spatial scales when the block is reused multiple times.
    """
    def __init__(self, channels: int, expansion: int = 4, layer_scale: float = 1e-3):
        super().__init__()
        hidden = channels * expansion

        # 1x1 pointwise expansion
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups(hidden), hidden)
        self.act = nn.SiLU(inplace=True)

        # 3x3 depthwise conv (dilation set at runtime)
        self.dw_weight = nn.Parameter(torch.empty(hidden, 1, 3, 3))
        self.dw_bias = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_normal_(self.dw_weight, mode='fan_out', nonlinearity='relu')

        self.divnorm = DivisiveNorm()

        # 1x1 pointwise contraction
        self.contract = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups(channels), channels)

        # Learnable residual scaling (starts small for stable training)
        self.gamma = nn.Parameter(torch.ones(1) * layer_scale)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        identity = x

        # Expand channels
        out = self.act(self.gn1(self.expand(x)))

        # Depthwise conv with runtime dilation (key for weight reuse!)
        out = F.conv2d(out, self.dw_weight, self.dw_bias,
                       padding=dilation, dilation=dilation, groups=out.shape[1])
        out = self.divnorm(out)

        # Contract back to original channels
        out = self.gn2(self.contract(out))

        # Residual connection with learned scaling
        return identity + self.gamma * out


# ============================================================================
# Main Network
# ============================================================================

class ECTiedNet(nn.Module):
    """
    Weight-Tied Expansion-Contraction Network for CIFAR-10.

    Architecture:
        Stem (3x3) -> [ECBlock x N with dilations] -> BlurPool -> GAP -> Linear

    The core idea is weight tying: ONE ECBlock is instantiated and reused
    N times. Different dilation rates at each iteration give varying
    receptive fields without adding parameters.

    Default dilation schedule: [1, 1, 2, 1, 2, 3]
    - Early iterations: local features (dilation=1)
    - Later iterations: broader context (dilation=2,3)
    """
    def __init__(
        self,
        num_classes: int = 10,
        channels: int = 64,
        expansion: int = 4,
        num_iterations: int = 6,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        self.num_iterations = num_iterations

        # Stem: simple 3x3 for 32x32 images (no downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

        # THE weight-tied block (reused num_iterations times)
        self.block = ECBlock(channels, expansion=expansion)

        # Downsample midway through iterations
        self.blur = BlurPool2d(channels, stride=2)

        # Classification head
        self.head = nn.Linear(channels, num_classes)

        # Dilation schedule for multi-scale processing
        self.dilations = (dilations or [1, 1, 2, 1, 2, 3])[:num_iterations]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # [B, C, 32, 32]

        for t in range(self.num_iterations):
            # Reuse same block with different dilation
            x = self.block(x, dilation=self.dilations[t])

            # Downsample halfway through
            if t == (self.num_iterations // 2) - 1:
                x = self.blur(x)  # [B, C, 16, 16]

        x = x.mean(dim=(2, 3))  # Global average pooling -> [B, C]
        return self.head(x)     # [B, num_classes]


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = ECTiedNet(num_classes=10, channels=64, num_iterations=6)
    print(f"ECTiedNet for CIFAR-10")
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

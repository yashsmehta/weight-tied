"""
ECTiedNet — Weight-Tied Expansion-Contraction Network

One ECBlock, reused N times with a growing dilation schedule.
The same weights process the image at multiple spatial scales without
adding parameters — each reuse changes only the dilation rate.

Architectural sources
---------------------
Macro-pattern (weight sharing + iterated dilation):
    Strubell et al. 2017 (ID-CNN); Yu & Koltun 2016
Micro-architecture (inverted residual / MBConv block):
    Sandler et al. 2018 (MobileNetV2)
DivisiveNorm (canonical cortical computation):
    Carandini & Heeger 2012, Nature Reviews Neuroscience
BlurPool (shift-invariant downsampling):
    Zhang 2019, ICML
LayerScale (stable training of recursive blocks):
    Touvron et al. 2021
Biological framing (dilation ≈ eCRF temporal expansion):
    Angelucci & Bressloff; Lamme & Roelfsema
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Utilities
# ============================================================================

def gn_groups(channels: int, max_groups: int = 16) -> int:
    """Largest divisor of channels that is <= max_groups, for GroupNorm."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ============================================================================
# Building Blocks
# ============================================================================

class DivisiveNorm(nn.Module):
    """
    Divisive normalisation (Carandini & Heeger 2012).

    Formula: y = x / (sigma + sqrt(pool(x²)) + eps)

    Each activation is divided by the RMS energy of its local spatial
    neighbourhood, implementing the surround suppression observed in V1.
    sigma is a learnable per-channel semi-saturation constant.

    eps is set to 1e-3 (not 1e-6) to prevent the denominator collapsing
    to near-zero when sigma is small early in training — which would amplify
    activations and cause loss spikes in a recursively-applied block.

    Applied only after the depthwise (spatial) convolution step — not after
    1x1 pointwise steps, which have no spatial footprint.
    """
    def __init__(self, num_channels: int, kernel_size: int = 3, eps: float = 1e-3):
        super().__init__()
        self.eps         = eps
        self.kernel_size = kernel_size
        self.padding     = kernel_size // 2
        # Initialised to 1.0 so sigma = softplus(1) ≈ 1.31 at step 0,
        # providing meaningful suppression before sigma is learned.
        self.log_sigma   = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.log_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = F.avg_pool2d(x.pow(2), self.kernel_size, stride=1, padding=self.padding)
        return x / (self.sigma + energy.sqrt() + self.eps)


class BlurPool2d(nn.Module):
    """
    Anti-aliased downsampling (Zhang 2019).

    Applies a fixed binomial blur kernel before strided subsampling,
    preserving shift-equivariance that standard strided convolutions destroy.
    Kernel [1,2,1]x[1,2,1] (normalised) registered as a non-learned buffer.
    """
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        k1d = torch.tensor([1., 2., 1.])
        k2d = torch.einsum('i,j->ij', k1d, k1d)
        k2d = k2d / k2d.sum()
        self.register_buffer('kernel', k2d[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=x.shape[1])


class ECBlock(nn.Module):
    """
    Expand-Contract block — the weight-shared building block.

    Structure per forward call:
        Expand:   1x1 conv -> GroupNorm -> SiLU       (channel mixing)
        Process:  3x3 dw conv (dil=d) -> DivisiveNorm -> SiLU  (spatial)
        Contract: 1x1 conv -> GroupNorm               (channel mixing)
        Residual: x + gamma * out                     (LayerScale)

    Normalisation rationale:
        GroupNorm on expand/contract — 1x1 convolutions have no spatial
        footprint; spatial suppression (DivisiveNorm) is not meaningful here.
        DivisiveNorm only after the spatial depthwise step, mapping onto
        cortical surround inhibition via horizontal connections.

    LayerScale (gamma):
        Initialises the residual branch near zero so recursive application
        of the block is stable at the start of training. gamma grows during
        training as the block learns useful representations.

    SiLU via F.silu():
        Applied with F.silu() rather than a shared nn.SiLU(inplace=True)
        module to avoid autograd version-counter issues when the same
        activation is called at two graph nodes.
    """
    def __init__(self, channels: int, expansion: int = 4, layer_scale: float = 1e-3):
        super().__init__()
        hidden = channels * expansion

        # Expand: channel mixing
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm1  = nn.GroupNorm(gn_groups(hidden), hidden)

        # Process: spatial filtering, dilation set at forward time
        self.dw_weight = nn.Parameter(torch.empty(hidden, 1, 3, 3))
        self.dw_bias   = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_normal_(self.dw_weight, mode='fan_out', nonlinearity='relu')
        self.divnorm   = DivisiveNorm(hidden)

        # Contract: channel mixing
        self.contract = nn.Conv2d(hidden, channels, 1, bias=False)
        self.norm2    = nn.GroupNorm(gn_groups(channels), channels)

        # LayerScale: learned residual scaling, initialised near zero
        self.gamma = nn.Parameter(torch.ones(1) * layer_scale)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        identity = x

        out = F.silu(self.norm1(self.expand(x)))

        out = F.conv2d(
            out, self.dw_weight, self.dw_bias,
            padding=dilation, dilation=dilation, groups=out.shape[1],
        )
        out = F.silu(self.divnorm(out))

        out = self.norm2(self.contract(out))

        return identity + self.gamma * out


# ============================================================================
# Network
# ============================================================================

class ECTiedNet(nn.Module):
    """
    Weight-Tied ECNet.

    One ECBlock is instantiated and reused N times. The dilation schedule
    [1, 1, 2, 1, 2, 3] varies the spatial scale processed at each iteration
    while keeping weights identical — a computational model of the temporally
    expanding extra-classical receptive field (eCRF) in visual cortex.

    BlurPool is inserted at the iteration midpoint so the block explicitly
    processes two spatial scales: full resolution for the first half,
    half resolution for the second half.

    Architecture:
        Stem -> [ECBlock x 3] -> BlurPool -> [ECBlock x 3] -> GAP -> head
    """
    def __init__(
        self,
        num_classes:    int            = 10,
        channels:       int            = 64,
        expansion:      int            = 4,
        num_iterations: int            = 6,
        dilations:      list[int] | None = None,
    ):
        super().__init__()
        self.dilations      = (dilations or [1, 1, 2, 1, 2, 3])[:num_iterations]
        self.num_iterations = len(self.dilations)
        self.downsample_at  = self.num_iterations // 2 - 1

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

        self.block = ECBlock(channels, expansion=expansion)
        self.blur  = BlurPool2d(channels, stride=2)
        self.head  = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for t, dilation in enumerate(self.dilations):
            x = self.block(x, dilation=dilation)
            if t == self.downsample_at:
                x = self.blur(x)

        x = x.mean(dim=(2, 3))
        return self.head(x)


# ============================================================================
# Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def divisive_norm_params(model: nn.Module):
    """Return log_sigma parameters separately for reduced-LR optimiser group."""
    return [p for n, p in model.named_parameters() if 'log_sigma' in n]

def main_params(model: nn.Module):
    """Return all parameters except log_sigma."""
    return [p for n, p in model.named_parameters() if 'log_sigma' not in n]


# Capacity sweep
def ectiednet_tiny(num_classes: int = 10, **kwargs) -> ECTiedNet:
    """~38K parameters. Baseline spec."""
    return ECTiedNet(num_classes=num_classes, channels=64,  expansion=4, **kwargs)

def ectiednet_small(num_classes: int = 10, **kwargs) -> ECTiedNet:
    """~143K parameters."""
    return ECTiedNet(num_classes=num_classes, channels=128, expansion=4, **kwargs)

def ectiednet_medium(num_classes: int = 10, **kwargs) -> ECTiedNet:
    """~548K parameters."""
    return ECTiedNet(num_classes=num_classes, channels=256, expansion=4, **kwargs)


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    print("ECTiedNet — capacity sweep\n")
    x = torch.randn(2, 3, 32, 32)
    for name, fn in [("tiny", ectiednet_tiny), ("small", ectiednet_small), ("medium", ectiednet_medium)]:
        model = fn(num_classes=10)
        y = model(x)
        print(f"ectiednet_{name:6s}  params: {count_parameters(model):>9,}   {list(x.shape)} -> {list(y.shape)}")

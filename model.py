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
    sigma is a learnable per-channel semi-saturation constant, corresponding
    directly to the sigma parameter in the Carandini & Heeger model.

    Applied only after the depthwise (spatial) convolution step — not after
    1×1 pointwise steps, which have no spatial footprint and do not benefit
    from spatial suppression.
    """
    def __init__(self, num_channels: int, kernel_size: int = 3, eps: float = 1e-6):
        super().__init__()
        self.eps         = eps
        self.kernel_size = kernel_size
        self.padding     = kernel_size // 2
        # Learnable semi-saturation constant, one per channel, always positive
        self.log_sigma   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

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
    Kernel [1,2,1]×[1,2,1] (normalised) registered as a non-learned buffer.
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
        Expand:   1×1 conv → GroupNorm → SiLU       (channel mixing)
        Process:  3×3 dw conv (dil=d) → DivisiveNorm → SiLU  (spatial)
        Contract: 1×1 conv → GroupNorm               (channel mixing)
        Residual: x + gamma × out                    (LayerScale)

    Normalisation choice rationale:
        GroupNorm on expand/contract — these are 1×1 pointwise convolutions
        with no spatial footprint; spatial suppression (DivisiveNorm) is not
        meaningful here and GroupNorm is more stable for channel mixing.
        DivisiveNorm only after the spatial depthwise step — this is the only
        operation with a spatial neighbourhood, mapping onto cortical surround
        inhibition mediated by local horizontal connections.

    LayerScale (gamma):
        Initialises the residual branch contribution near zero (gamma ≈ 1e-3).
        Prevents large perturbations from randomly-initialised weights when the
        block is applied 6 times in sequence. gamma is learned and grows during
        training, providing an implicit warm-start without requiring a long LR
        warmup to compensate for recursive noise accumulation.

    SiLU via F.silu():
        Activations are applied with F.silu() rather than a shared nn.SiLU
        module to avoid autograd version-counter issues that arise when the
        same inplace module is called at two different points in the graph.
    """
    def __init__(self, channels: int, expansion: int = 4, layer_scale: float = 1e-3):
        super().__init__()
        hidden = channels * expansion

        # Expand: channel mixing, GroupNorm appropriate (no spatial footprint)
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm1  = nn.GroupNorm(gn_groups(hidden), hidden)

        # Process: spatial filtering, dilation set at forward time
        self.dw_weight = nn.Parameter(torch.empty(hidden, 1, 3, 3))
        self.dw_bias   = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_normal_(self.dw_weight, mode='fan_out', nonlinearity='relu')
        self.divnorm   = DivisiveNorm(hidden)

        # Contract: channel mixing, GroupNorm appropriate (no spatial footprint)
        self.contract = nn.Conv2d(hidden, channels, 1, bias=False)
        self.norm2    = nn.GroupNorm(gn_groups(channels), channels)

        # LayerScale: learned residual scaling, initialised near zero
        self.gamma = nn.Parameter(torch.ones(1) * layer_scale)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        identity = x

        # Expand (channel mixing)
        out = F.silu(self.norm1(self.expand(x)))

        # Process (spatial, dilation varies per iteration)
        out = F.conv2d(
            out, self.dw_weight, self.dw_bias,
            padding=dilation, dilation=dilation, groups=out.shape[1],
        )
        out = F.silu(self.divnorm(out))

        # Contract (channel mixing, no activation — follows MobileNetV2)
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
    expanding extra-classical receptive field (eCRF) in visual cortex:

        Iterations 1–2  (dilation 1): near-surround integration
                                       → horizontal connections within V1
        Iterations 3, 5 (dilation 2): intermediate surround
                                       → onset of feedback from V2/V4
        Iteration  4    (dilation 1): local consolidation between feedback waves
        Iteration  6    (dilation 3): far-surround integration
                                       → long-range feedback from higher areas

    BlurPool is inserted at the iteration midpoint so the block explicitly
    processes two spatial scales: full resolution for the first half of
    iterations, half resolution for the second half.

    Architecture:
        Stem  →  [ECBlock × 3]  →  BlurPool  →  [ECBlock × 3]  →  GAP  →  head
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

        # Stem: project RGB to feature channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

        # The one shared block — called num_iterations times in forward()
        self.block = ECBlock(channels, expansion=expansion)

        # BlurPool inserted at the midpoint of iterations
        self.blur = BlurPool2d(channels, stride=2)

        # Classification head
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for t, dilation in enumerate(self.dilations):
            x = self.block(x, dilation=dilation)
            if t == self.downsample_at:
                x = self.blur(x)

        x = x.mean(dim=(2, 3))   # global average pool
        return self.head(x)


# ============================================================================
# Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Capacity sweep — channel width is the single knob
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

    for name, fn in [
        ("tiny",   ectiednet_tiny),
        ("small",  ectiednet_small),
        ("medium", ectiednet_medium),
    ]:
        model = fn(num_classes=10)
        y = model(x)
        print(f"ectiednet_{name:6s}  params: {count_parameters(model):>9,}   "
              f"{list(x.shape)} -> {list(y.shape)}")

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
        # Run entirely in float32 to prevent AMP float16 overflow in both
        # the forward pass (x.pow(2) overflow at |x|>256) and the backward
        # pass (reciprocal of small denominator can overflow in float16).
        # Only the final output is cast back to the original dtype.
        x_f32 = x.float()
        energy = F.avg_pool2d(x_f32.pow(2), self.kernel_size, stride=1, padding=self.padding)
        out = x_f32 / (self.sigma.float() + energy.sqrt() + self.eps)
        return out.to(x.dtype)


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

    Depthwise conv forced to float32 (AMP safety):
        Under AMP, casting the depthwise conv's weights to float16 can
        overflow once |value| exceeds ~65504 after enough training,
        producing inf *before* DivisiveNorm's own float32 upcast ever gets
        a chance to help — by the time DivisiveNorm sees the tensor, the
        damage is already done. Verified empirically: routing this conv
        through a plain nn.Conv2d module makes no numerical difference
        under autocast versus the previous raw-Parameter + F.conv2d call —
        both hit the exact same overflow at the same weight magnitude,
        since both dispatch to the same underlying op. The actual fix is
        to run this specific conv outside autocast, always in float32,
        exactly the same treatment DivisiveNorm already gets. It's kept as
        a proper nn.Conv2d module (rather than a raw Parameter pair) purely
        for readability — see forward().
    """
    def __init__(self, channels: int, expansion: int = 4, layer_scale: float = 1e-3):
        super().__init__()
        hidden = channels * expansion

        # Expand: channel mixing
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm1  = nn.GroupNorm(gn_groups(hidden), hidden)

        # Process: spatial filtering. dilation/padding are mutated per
        # forward call (see forward()) since the schedule changes them
        # every iteration — the same pattern CORnet-S uses to mutate a
        # conv's stride per timestep.
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, groups=hidden)
        nn.init.kaiming_normal_(self.dw.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.dw.bias)
        self.divnorm = DivisiveNorm(hidden)

        # Contract: channel mixing
        self.contract = nn.Conv2d(hidden, channels, 1, bias=False)
        self.norm2    = nn.GroupNorm(gn_groups(channels), channels)

        # LayerScale: learned residual scaling, initialised near zero
        self.gamma = nn.Parameter(torch.ones(1) * layer_scale)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        identity = x

        out = F.silu(self.norm1(self.expand(x)))

        self.dw.dilation = (dilation, dilation)
        self.dw.padding  = (dilation, dilation)
        with torch.autocast(device_type=out.device.type, enabled=False):
            out = self.dw(out.float()).to(out.dtype)
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


class UntiedECTiedNet(nn.Module):
    """
    Untied control — the H4 ablation.

    Structurally identical to ECTiedNet (same stem, same dilation schedule,
    same BlurPool midpoint placement, same GAP + head) but instantiates a
    SEPARATE ECBlock for every iteration instead of calling one shared block
    repeatedly. This isolates weight *tying* from iteration count / effective
    depth: if accuracy or brain alignment differs from ECTiedNet at matched
    width, the difference is attributable to tying itself, not to depth.

    Parameter count is intentionally NOT matched to ectiednet_tiny — at the
    same channel width, N independent blocks have ~N times the parameters of
    one shared block (project_plan.md 3.4). The width/depth-matched capacity
    sweep (ectiednet_small/medium) already covers the "same params, different
    architecture" comparison; this ablation's job is to hold width and depth
    fixed and let tying be the only variable. Its resulting parameter count
    (~6x tiny) happens to land near the small/medium capacity-sweep points,
    which is useful for cross-checking H3 against H4.
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

        # One independent ECBlock per iteration — the only structural
        # difference from ECTiedNet, which calls a single shared block
        # self.num_iterations times. BlurPool stays a single shared instance
        # since it carries no learned weights, so tying is not at stake there.
        self.blocks = nn.ModuleList([
            ECBlock(channels, expansion=expansion) for _ in range(self.num_iterations)
        ])
        self.blur = BlurPool2d(channels, stride=2)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for t, dilation in enumerate(self.dilations):
            x = self.blocks[t](x, dilation=dilation)
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

def gamma_params(model: nn.Module):
    """Return LayerScale gamma parameters for very-low-LR group.
    Gamma accumulates gradients from all N block iterations so its
    effective gradient is N times larger than other parameters.
    Without a separate low LR, gamma grows 20x in the first epoch
    and causes the block contribution to explode recursively."""
    return [p for n, p in model.named_parameters() if 'gamma' in n]

def main_params(model: nn.Module):
    """Return all parameters except log_sigma and gamma."""
    return [p for n, p in model.named_parameters()
            if 'log_sigma' not in n and 'gamma' not in n]


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


# H4 ablation
def untied_control(num_classes: int = 10, **kwargs) -> UntiedECTiedNet:
    """
    Untied H4 ablation. Channel width matched to ectiednet_tiny (64) so the
    tying comparison is width/depth-matched, not capacity-matched — see
    UntiedECTiedNet docstring. Expect ~6x ectiednet_tiny's parameter count.
    """
    return UntiedECTiedNet(num_classes=num_classes, channels=64, expansion=4, **kwargs)


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

    print("\nH4 ablation — untied control\n")
    tiny_params = count_parameters(ectiednet_tiny(num_classes=10))
    model = untied_control(num_classes=10)
    y = model(x)
    ratio = count_parameters(model) / tiny_params
    print(f"untied_control  params: {count_parameters(model):>9,}   "
          f"{list(x.shape)} -> {list(y.shape)}   ({ratio:.2f}x tiny)")

    # Sanity: param-group helpers should partition ALL params with no overlap,
    # and should pick up gamma/log_sigma correctly across the ModuleList too.
    sigma_p = divisive_norm_params(model)
    gamma_p = gamma_params(model)
    main_p  = main_params(model)
    total_via_groups = sum(p.numel() for p in sigma_p + gamma_p + main_p)
    assert total_via_groups == count_parameters(model), "param-group helpers dropped or double-counted params"
    assert len(gamma_p) == model.num_iterations, "expected one gamma per (independent) block"
    print(f"param groups OK: main={sum(p.numel() for p in main_p):,}  "
          f"sigma={sum(p.numel() for p in sigma_p):,}  gamma={len(gamma_p)} scalars")

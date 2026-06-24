"""
Weight-Tied ECNet

ONE shared ECBlock applied across three resolution levels.

Spatial flow:
    224 → stem → 56 → [N1 iters] → BlurPool → 28 → [N2 iters] → BlurPool
        → 14 → [N3 iters] → BlurPool → 7 → GAP → head

Total downsampling: 4 (stem) × 2 × 2 × 2 = 32×  →  7×7 before GAP.
Theoretical receptive field: ~511px  >>  224px input (full image coverage).

Default (~420K params): channels=128, stage_iterations=(4,4,4)
Untied equivalent (unique weights per iteration): ~1.95M params  →  4.6× reduction.

The single shared block is the core claim: one canonical circuit, applied
iteratively at three spatial scales, recapitulates the ventral stream hierarchy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Utility
# ============================================================================

def gn_groups(channels: int, max_groups: int = 32) -> int:
    """Largest divisor of channels <= max_groups for GroupNorm."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ============================================================================
# Building Blocks
# ============================================================================

class DivisiveNorm(nn.Module):
    """
    Biologically-inspired normalization (visual cortex gain control).
    y = x / (eps + local_avg(|x|))   —   Carandini & Heeger 2012.
    """
    def __init__(self, kernel_size: int = 3, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (self.pool(x.abs()) + self.eps)


class BlurPool2d(nn.Module):
    """
    Anti-aliased stride-2 downsampling (Zhang 2019).
    Fixed binomial [1,2,1]×[1,2,1] kernel — no learned parameters.
    Called three times in the network body (same instance, stateless).
    """
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        k1d = torch.tensor([1., 2., 1.])
        k2d = torch.einsum('i,j->ij', k1d, k1d)
        k2d /= k2d.sum()
        self.register_buffer('kernel', k2d[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=x.shape[1])


class ECBlock(nn.Module):
    """
    Expansion-Contraction block — instantiated once, shared across all iterations.

    Forward per call:
        expand:   1×1 conv (C → C×expansion) + GN + GELU
        process:  3×3 depthwise conv at runtime dilation → DivisiveNorm → GELU
        contract: 1×1 conv (C×expansion → C) + GN
        residual: output = input + γ_per_channel * processed

    The dilation argument lets the same weights capture different spatial scales
    at each iteration — the mechanism that makes weight reuse effective.
    """
    def __init__(self, channels: int, expansion: int = 4, layer_scale: float = 1e-6):
        super().__init__()
        hidden = channels * expansion

        self.expand   = nn.Conv2d(channels, hidden, 1, bias=False)
        self.gn1      = nn.GroupNorm(gn_groups(hidden), hidden)
        self.act      = nn.GELU()

        self.dw_weight = nn.Parameter(torch.empty(hidden, 1, 3, 3))
        self.dw_bias   = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_normal_(self.dw_weight, mode='fan_out', nonlinearity='relu')

        self.divnorm  = DivisiveNorm()
        self.act2     = nn.GELU()

        self.contract = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gn2      = nn.GroupNorm(gn_groups(channels), channels)

        # Per-channel residual scale — init near zero for stable early training
        self.gamma    = nn.Parameter(torch.ones(channels) * layer_scale)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        identity = x
        out = self.act(self.gn1(self.expand(x)))
        out = F.conv2d(out, self.dw_weight, self.dw_bias,
                       padding=dilation, dilation=dilation, groups=out.shape[1])
        out = self.act2(self.divnorm(out))
        out = self.gn2(self.contract(out))
        return identity + self.gamma.view(1, -1, 1, 1) * out


# ============================================================================
# Main Network
# ============================================================================

class ECTiedNet(nn.Module):
    """
    Weight-Tied ECNet for ImageNet.

    Architecture:
        Stem (stride-4) → [N1 iters] → BlurPool → [N2 iters] → BlurPool
                        → [N3 iters] → BlurPool → GAP → head

    ONE ECBlock instance is shared across all N1+N2+N3 iterations and all
    three resolution levels. ONE BlurPool instance is reused three times
    (it is stateless — no learned parameters).

    Between resolution levels the dilation schedule cycles continuously, so
    each level sees the same mix of local and broad-context processing.

    Fixes vs. original design:
        - 32× downsampling  (was 8×)  →  7×7 before GAP
        - RF ≈ 511px  (was 143px)    →  full image coverage
        - channels=128  (was 64)     →  richer 128-dim representation
        - GELU after DivisiveNorm    →  nonlinearity after gain control
        - per-channel gamma          →  ConvNeXt-style residual scaling
    """
    def __init__(
        self,
        num_classes: int = 1000,
        channels: int = 128,
        stage_iterations: tuple[int, int, int] = (4, 4, 4),
        expansion: int = 4,
        dilations: list[int] | None = None,
        layer_scale: float = 1e-6,
    ):
        super().__init__()
        assert len(stage_iterations) == 3
        self.stage_iterations = tuple(stage_iterations)

        # Stem: two 3×3 stride-2 convs  →  224×224 to 56×56
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_groups(channels), channels),
        )

        # THE shared block — one instance reused across all iterations
        self.block = ECBlock(channels, expansion=expansion, layer_scale=layer_scale)

        # THE shared BlurPool — stateless, reused three times (56→28, 28→14, 14→7)
        self.blur = BlurPool2d(channels, stride=2)

        self.head = nn.Linear(channels, num_classes)

        # Flat dilation schedule — cycles continuously across all iterations
        total = sum(stage_iterations)
        base = dilations or [1, 2, 3, 2]
        self.dilations = [base[t % len(base)] for t in range(total)]

    def _forward_body(self, x: torch.Tensor):
        """Shared forward through stem + all iterations + BlurPools.

        Returns (s1, s2, s3, out) where:
            s1  — spatial map after N1 iters at 56×56  (before first BlurPool)
            s2  — spatial map after N2 iters at 28×28  (before second BlurPool)
            s3  — spatial map after N3 iters at 14×14  (before third BlurPool)
            out — spatial map after third BlurPool at 7×7  (used for classification)
        """
        N1, N2, N3 = self.stage_iterations
        t = 0
        x = self.stem(x)

        for _ in range(N1):
            x = self.block(x, dilation=self.dilations[t]); t += 1
        s1 = x
        x = self.blur(x)

        for _ in range(N2):
            x = self.block(x, dilation=self.dilations[t]); t += 1
        s2 = x
        x = self.blur(x)

        for _ in range(N3):
            x = self.block(x, dilation=self.dilations[t]); t += 1
        s3 = x
        x = self.blur(x)

        return s1, s2, s3, x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Final post-GAP representation (7×7 → channels-dim). Used for classification."""
        *_, out = self._forward_body(x)
        return out.mean(dim=(2, 3))

    def extract_stage_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Post-GAP features at each resolution level. For RSA: s1→V1, s2→V4, s3→IT.

        Returns [(B,C), (B,C), (B,C)] — same channel width, different spatial context.
        """
        s1, s2, s3, _ = self._forward_body(x)
        return [s.mean(dim=(2, 3)) for s in [s1, s2, s3]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))


# ============================================================================
# Utility
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ECTiedNet(num_classes=1000)
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input:  {x.shape}  →  Output: {y.shape}")

    stages = model.extract_stage_features(x)
    print(f"Stage features: {[tuple(s.shape) for s in stages]}")

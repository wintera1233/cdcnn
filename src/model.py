"""The paper's ResNet backbone and classifier head.

Section 5.2: "we use a neural network with five ResNet blocks. The kernel's shape
of each ResNet is 3 x 1, and there are two convolution layers and a shortcut
function with 1x1 kernel in each block, and the channels of convolution increase
from 1 to 128 step by step. ... this network doesn't apply the pooling layer."

Fig. 2's notation, from Section 2.2: "3 Conv 128 means the kernel size is 3, and
the number of filters is 128. The activation function chooses RULE."

Channel widths follow the prose capped at 128 rather than Fig. 2's 256 and 512;
see `proposal.md` section 2.2. Each variant's channel tuple is a frozen literal,
so adding a variant can never change what an existing one builds.
"""

from __future__ import annotations

import torch
from torch import nn

from src.protocol import ProtocolError

N_FEATURES = 128
N_CLASSES = 6

# (in, mid, out) per block: `mid` is the width of the main path's first
# convolution, `out` the width of its second and of the 1x1 shortcut.
TEXT_CAPPED_128 = ((1, 32, 32), (32, 64, 64), (64, 128, 128),
                   (128, 128, 128), (128, 128, 128))

# Fig. 2 as printed, with `3 Conv 2` in Resnet1 read as 32. The figure reaches
# 256 in Resnet4 and 512 inside Resnet5, contradicting Section 5.2's "from 1 to
# 128 step by step". Restored 2026-09-23: both flatten cells of the v7.0 ladder
# scored 0.032 and 0.049 below their counterparts on the previous project's
# wider backbone, which is evidence for the figure; see baseline.md section 4.2.
FIGURE_WIDTHS = ((1, 32, 32), (32, 64, 64), (64, 128, 128),
                 (128, 256, 256), (256, 512, 128))

HEADS = ("flatten", "gap")

# The head's normalisation layer, the "Batch Normal" box of Fig. 2, sitting
# between FC128 and FC6. BatchNorm1d stores Batch 1's running mean and variance
# and applies them to drifted target activations at inference, which is the same
# failure mode as a fitted StandardScaler on the inputs. LayerNorm normalises
# each sample against itself and stores nothing. Both cost 256 parameters, so
# the comparison changes nothing else.
HEAD_NORMS = ("batchnorm", "layernorm")


def _head_norm(kind: str) -> nn.Module:
    if kind == "batchnorm":
        return nn.BatchNorm1d(N_FEATURES)
    if kind == "layernorm":
        return nn.LayerNorm(N_FEATURES)
    raise ProtocolError(f"unknown head normalisation {kind!r}")

# The 2x2 factorial of the v7.0 ladder: {flatten, GAP} head x {StandardScaler,
# per-sample} Normal block, on one backbone.
VARIANTS: dict[str, dict] = {
    "R-txt": {"channels": TEXT_CAPPED_128, "head": "flatten",
              "normalizer": "standard_scaler"},
    "R-txt-ps": {"channels": TEXT_CAPPED_128, "head": "flatten",
                 "normalizer": "per_sample"},
    "R-lite": {"channels": TEXT_CAPPED_128, "head": "gap",
               "normalizer": "standard_scaler"},
    "R-lite-ps": {"channels": TEXT_CAPPED_128, "head": "gap",
                  "normalizer": "per_sample"},
    # v7.1: the same two normalizers on Fig. 2's channel widths.
    "R-fig": {"channels": FIGURE_WIDTHS, "head": "flatten",
              "normalizer": "standard_scaler"},
    "R-fig-ps": {"channels": FIGURE_WIDTHS, "head": "flatten",
                 "normalizer": "per_sample"},
    # v7.2: the head's normalisation crossed with the input normalisation, on
    # Fig. 2's widths at lr 0.0003. R-fig and R-fig-ps supply the batchnorm row.
    "R-fig-ln": {"channels": FIGURE_WIDTHS, "head": "flatten",
                 "normalizer": "standard_scaler", "head_norm": "layernorm"},
    "R-fig-ps-ln": {"channels": FIGURE_WIDTHS, "head": "flatten",
                    "normalizer": "per_sample", "head_norm": "layernorm"},
    # v7.4: three input normalisations that the centroid-misplacement diagnostic
    # in docs/why-acetaldehyde.md ranks above plain per-sample. Everything else
    # matches R-fig-ps@lr0.0003.
    "R-fig-ssps": {"channels": FIGURE_WIDTHS, "head": "flatten",
                   "normalizer": "standard_then_per_sample"},
    "R-fig-logps": {"channels": FIGURE_WIDTHS, "head": "flatten",
                    "normalizer": "signed_log_then_per_sample"},
    "R-fig-grp": {"channels": FIGURE_WIDTHS, "head": "flatten",
                  "normalizer": "per_statistic_group"},
}


# Retired 2026-09-23. Global average pooling cost -0.078 target mean, separable
# on both levels of the other factor; see baseline.md section 4.1. The paper's
# "this network doesn't apply the pooling layer" is load-bearing. These two stay
# defined so the 400 frozen epoch checkpoints of the v7.0 ladder remain loadable
# and that result stays reproducible; `src.config` refuses them in a new
# configuration.
RETIRED_VARIANTS = ("R-lite", "R-lite-ps")


class ResidualBlock1D(nn.Module):
    """`Conv1d(3) -> ReLU -> Conv1d(3)` summed with `Conv1d(1)`, then ReLU.

    `padding=1` on the 3-kernels keeps the length at 128, which is what "doesn't
    apply the pooling layer" requires: nothing in the backbone changes length.
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1))
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class FlattenHead(nn.Module):
    """`FC 128 -> Batch Normal -> FC 6` over the flattened backbone output.

    Fig. 2 as printed. Costs 2,098,310 parameters, of which `Linear(16384, 128)`
    is 2,097,280.
    """

    def __init__(self, channels: int, length: int = N_FEATURES,
                 norm: str = "batchnorm"):
        super().__init__()
        self.reduce = nn.Flatten()
        self.fc128 = nn.Linear(channels * length, N_FEATURES)
        self.norm = _head_norm(norm)
        self.fc6 = nn.Linear(N_FEATURES, N_CLASSES)

    def forward(self, x):
        return self.fc6(self.norm(self.fc128(self.reduce(x))))


class GapHead(nn.Module):
    """The same head fed by a global average over the length axis.

    Contradicts "this network doesn't apply the pooling layer" and is run as a
    declared variant. Costs 17,542 parameters: `Linear(128, 128)` is 16,512.
    """

    def __init__(self, channels: int, length: int = N_FEATURES,
                 norm: str = "batchnorm"):
        super().__init__()
        self.reduce = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fc128 = nn.Linear(channels, N_FEATURES)
        self.norm = _head_norm(norm)
        self.fc6 = nn.Linear(N_FEATURES, N_CLASSES)

    def forward(self, x):
        return self.fc6(self.norm(self.fc128(self.reduce(x))))


class BaselineResNet(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ProtocolError(f"unknown variant {variant!r}")
        spec = VARIANTS[variant]
        self.variant = variant
        self.blocks = nn.Sequential(
            *[ResidualBlock1D(*widths) for widths in spec["channels"]])
        out_channels = spec["channels"][-1][2]
        head = spec["head"]
        # Fig. 2's "Batch Normal" unless the variant declares otherwise.
        norm = spec.get("head_norm", "batchnorm")
        if head == "flatten":
            self.head = FlattenHead(out_channels, norm=norm)
        elif head == "gap":
            self.head = GapHead(out_channels, norm=norm)
        else:
            raise ProtocolError(f"unknown head {head!r}")

    def features(self, x):
        """`Phi(X)` of Eq. (S2): the five blocks, before the classifier."""
        return self.blocks(x)

    def forward(self, x):
        return self.head(self.features(x))


def build(variant: str) -> BaselineResNet:
    return BaselineResNet(variant)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def parameter_breakdown(model: BaselineResNet) -> dict[str, int]:
    return {"backbone": parameter_count(model.blocks),
            "head": parameter_count(model.head),
            "total": parameter_count(model)}


def to_input(x) -> torch.Tensor:
    """`[N, 128]` -> `[N, 1, 128]`: one channel, length 128.

    Never a 16x8 or 8x16 image; see `CLAUDE.md` section 5.
    """
    tensor = torch.as_tensor(x, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != N_FEATURES:
        raise ProtocolError(f"expected [N, {N_FEATURES}], got {tuple(tensor.shape)}")
    return tensor.unsqueeze(1)

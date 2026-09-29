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

from typing import NamedTuple

import torch
from torch import nn

from src.generate import FeatureGeneration
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
# The settled baseline of `baseline.md`, which every v8 cell builds on.
_LOGPS = {"channels": FIGURE_WIDTHS, "head": "flatten",
          "normalizer": "signed_log_then_per_sample"}

# The displacement multiplier, in units of Batch 1's own two-month block offset.
#
# v8.0 used 18, from "36 months of collection divided by Batch 1's two". That
# reasoning assumed drift accumulates linearly in time, and it does not: measured
# against the block offset, every real per-class drift in every target batch lies
# between 0.83 and 3.74, and Batch 10 at 36 months drifts *less* than Batch 8 at
# 22. Coverage peaks at 3. Variants carry their own value; this is the default.
AUGMENT_DISPLACEMENT = 3.0
AUGMENT_LAMBDA = 0.5

# Section 2.4: "Rs(.) means the first three convolutional layers; the remaining
# part is recorded as Rf(.)" and "The feature generates a block in the middle of
# the third and the fourth ResNet block." The block carries no parameters, so a
# variant that declares it has the same state_dict as one that does not.
GENERATION_SPLIT = 3

# Eq. (4) weights L_MSE by lambda_MSE, "a weighting factor that belongs to
# [0,1]". Neither the paper nor the supplement gives its value. 0.5 is the
# declared default; `R-gen-m10` measures the sensitivity to it.
LAMBDA_MSE = 0.5

# Eq. (4)'s third weight. The supplement says only that it "is a weight used to
# adjust the value of" L_con; 0.5 mirrors lambda_MSE and the previous branch.
# tau is likewise unstated; 0.07 is SupCon's published default and the previous
# branch's choice, and `R-con-t5` measures the sensitivity to it.
LAMBDA_CON = 0.5
CONTRASTIVE_TEMPERATURE = 0.07

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
    "R-fig-logps": dict(_LOGPS),
    "R-fig-grp": {"channels": FIGURE_WIDTHS, "head": "flatten",
                  "normalizer": "per_statistic_group"},
    # v8.0: the data-augmentation factorial. Everything matches R-fig-logps
    # except the `augment` field, which train_one reads. isotropic is the
    # paper's Eq. (7) noise; direction adds a displacement along the drift
    # estimated from Batch 1's own two acquisition sessions.
    "R-aug-paper": {**_LOGPS, "augment": {"isotropic": True, "direction": None,
                                          "displacement": 18.0}},
    "R-aug-eth": {**_LOGPS, "augment": {"isotropic": True, "direction": "ethanol",
                                        "displacement": 18.0}},
    "R-aug-ethd": {**_LOGPS, "augment": {"isotropic": False, "direction": "ethanol",
                                         "displacement": 18.0}},
    # v8.1: the displacement multiplier, swept around the coverage peak at 3.
    "R-aug-t2": {**_LOGPS, "augment": {"isotropic": False, "direction": "ethanol",
                                       "displacement": 2.0}},
    "R-aug-t3": {**_LOGPS, "augment": {"isotropic": False, "direction": "ethanol",
                                       "displacement": 3.0}},
    "R-aug-t4": {**_LOGPS, "augment": {"isotropic": False, "direction": "ethanol",
                                       "displacement": 4.0}},
    # v8.2: displace inside the span of all three block offsets rather than
    # along one of them. The basis needs no choosing, so unlike the Ethanol
    # direction it is a source-only estimator.
    "R-aug-sub2": {**_LOGPS, "augment": {"isotropic": False, "direction": "subspace",
                                         "displacement": 2.0}},
    "R-aug-sub4": {**_LOGPS, "augment": {"isotropic": False, "direction": "subspace",
                                         "displacement": 4.0}},
    "R-aug-sph2": {**_LOGPS, "augment": {"isotropic": False, "direction": "sphere",
                                         "displacement": 2.0}},
}

# v9.0: the feature generation block, on top of v8.1's best augmentation
# (`R-aug-t2`, target mean 0.5770, the only separable augmentation gain). Every
# v9 cell carries that same `augment` field, so the factor under test is the
# generation block alone.
#
# `style` is the axis Eqs. (10)-(11) reduce over. Per-position is settled: the
# three readings explain 1.2 %, 15.3 % and 30.1 % of the real drift at this
# depth; see `src/generate.py` and `docs/drift-geometry.md`.
#
# `ce_on_generated` is the one textual ambiguity left. Eq. (S2) writes L_ce over
# Phi alone, so the generated branch reaches the loss only through L_MSE; the
# algorithm box's line 6, "out = FC(Zs), out = FC(Zs) and calculate Ll2 and Lce",
# reads as both. The grid runs both.
_GEN = {**VARIANTS["R-aug-t2"],
        "generate": {"style": "position", "pool": 2}}

VARIANTS.update({
    "R-gen": {**_GEN, "lambda_mse": LAMBDA_MSE, "ce_on_generated": False},
    "R-gen-ce2": {**_GEN, "lambda_mse": LAMBDA_MSE, "ce_on_generated": True},
    "R-gen-m10": {**_GEN, "lambda_mse": 1.0, "ce_on_generated": False},
})

# v10.0: a signed Eq. (14). v9.0 measured the block's perturbation as well aimed
# (7.6 % of its energy in the drift span, 25x the null) but symmetric along the
# drift axis (signed component -7.5 % of its magnitude), and its paired
# displacement along any single drift axis at 3 % of the drift. Two repairs, both
# directed by Batch 1's own acquisition-block offset measured in style space:
# "fold" keeps the paper's magnitude and fixes only the sign; "shift" adds two
# block offsets along the direction, the v8.1 construction moved to block 3.
# The Ethanol offset is used, as R-aug-t2 does, so the direction choice is
# target-informed in the same sense; the sign itself is computed from Batch 1.
VARIANTS.update({
    "R-gen-sign": {**_GEN, "lambda_mse": LAMBDA_MSE, "ce_on_generated": False,
                   "generate": {"style": "position", "pool": 2, "sign": "fold",
                                "direction": "ethanol"}},
    "R-gen-shift": {**_GEN, "lambda_mse": LAMBDA_MSE, "ce_on_generated": False,
                    "generate": {"style": "position", "pool": 2, "sign": "shift",
                                 "direction": "ethanol", "displacement": 2.0}},
})

# v11.0: the contrastive loss L_con (Eq. S5) over the (z_f, z_bar_f) pair, the
# last of the paper's three components and the only one whose loss asks for
# invariance. `R-con` is the paper's CDCNN as written on top of R-gen; `R-con-
# shift` pairs z_f with v10.0's shifted z_bar_f, the one generated feature that
# has a target-facing displacement; `R-con-t5` varies the one constant the paper
# never gives. f is an L2 normalisation of the flattened z_f, no learned head.
_CON = {**VARIANTS["R-gen"], "lambda_con": LAMBDA_CON,
        "temperature": CONTRASTIVE_TEMPERATURE}

VARIANTS.update({
    "R-con": {**_CON},
    "R-con-shift": {**_CON, "generate": VARIANTS["R-gen-shift"]["generate"]},
    "R-con-t5": {**_CON, "temperature": 0.5},
})


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


class Branches(NamedTuple):
    """One forward pass of a model that carries a feature generation block.

    `generated` fields are `None` when the variant declares no block, which is
    every v7 and v8 cell.
    """

    logits: torch.Tensor
    features: torch.Tensor
    logits_generated: torch.Tensor | None
    features_generated: torch.Tensor | None


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
        generate = spec.get("generate")
        self.generation = (None if generate is None
                           else FeatureGeneration(**generate))

    def stem(self, x):
        """`Rs(X)` of Section 2.4: the first three blocks."""
        return self.blocks[:GENERATION_SPLIT](x)

    def trunk(self, z):
        """`Rf(.)`: the remaining two blocks."""
        return self.blocks[GENERATION_SPLIT:](z)

    def features(self, x):
        """`Phi(X)` of Eq. (S2): the five blocks, before the classifier.

        The generation block is never on this path. Fig. 2 requires it: "during
        forecasting, the program separates the data manipulation block from the
        prediction process", so `forward`, and with it every evaluation, runs
        the plain backbone whether or not the variant declares a block.
        """
        return self.blocks(x)

    def forward(self, x):
        return self.head(self.features(x))

    def forward_pair(self, x, generator: torch.Generator | None = None
                     ) -> Branches:
        """Both branches of Eq. (S4), sharing `Rs` and the classifier head.

        `Phi_bar` of S4 is `Rf . generation . Rs`; `Phi` is `Rf . Rs`. Each is
        run once, so the block sees `Rs(X)` exactly as Section 2.4 specifies.
        """
        mid = self.stem(x)
        z_f = self.trunk(mid)
        if self.generation is None:
            return Branches(self.head(z_f), z_f, None, None)
        z_bar_f = self.trunk(self.generation(mid, generator=generator))
        return Branches(self.head(z_f), z_f, self.head(z_bar_f), z_bar_f)


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

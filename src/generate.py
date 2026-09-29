"""Feature generation: the paper's block between Resnet3 and Resnet4.

Section 4, Eqs. (8)-(16). A sample's features are split into a pooled part and a
residual; the residual's first and second moments are treated as the sample's
"style", carrying the domain rather than the label; those moments are replaced
with values drawn from a distribution fitted across the batch; the two parts are
recombined.

    z^H = sample(MaxPool(z))                                  (8)   pooled, kept
    z^L = z - z^H                                             (9)   residual
    mu, delta          = moments of z^L                       (10)(11)
    mu_hat, delta_hat  = batch moments of mu                  (12)
    mu_bar, delta_bar  = batch moments of delta               (13)
    mu' ~ N(mu_hat, delta_hat),  delta' ~ N(mu_bar, delta_bar)      (14)
    z~^L = delta' (z^L - mu) / delta + mu'                     (16)
    z~   = z^H + z~^L                                          (15)

The paper's naming inverts the usual one, calling the pooled part
high-frequency and the residual low-frequency. That reads correctly if the
frequency meant is temporal: the residual is what carries the environment and
the slow sensor drift.

**Which axis the moments reduce over** is the one real ambiguity. Eq. (10)-(11)
sum over both the channel and the length axis, giving one scalar per sample; the
sentence after says the result lives in R^{B x CH}, which is the shape of z^L
itself and would make the restyle an identity. Neither is AdaIN's convention,
which is per-channel. Measured against the real drift at this depth, the
readings explain 1.2 %, 15.3 % and 30.1 % of it: scalar, per-channel, and
per-position, the last being the one the paper never considers. Per-position
corresponds approximately to per-sensor, because the length axis still carries
the input layout - position 8(s-1)+k came from sensor s's statistic k, smeared
by a receptive field of plus or minus six. See `docs/drift-geometry.md`.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.protocol import ProtocolError

STYLE_AXES = ("scalar", "channel", "position")
SIGN_MODES = ("fold", "shift")
EPSILON = 1e-5


def reduce_dims(style: str) -> tuple[int, ...]:
    """The axes of `[B, C, H]` the style moments are taken over."""
    if style == "scalar":
        return (1, 2)
    if style == "channel":
        return (2,)
    if style == "position":
        return (1,)
    raise ProtocolError(f"unknown style axis {style!r}")


class FeatureGeneration(nn.Module):
    """Eqs. (8)-(16), returning a restyled copy of the features.

    Training only: `src.evaluate` never calls it, as Fig. 2 requires - "during
    forecasting, the program separates the data manipulation block from the
    prediction process".
    """

    def __init__(self, style: str = "position", pool: int = 2,
                 sign: str | None = None, direction: str = "ethanol",
                 displacement: float = 2.0):
        super().__init__()
        if style not in STYLE_AXES:
            raise ProtocolError(f"unknown style axis {style!r}")
        if sign is not None and sign not in SIGN_MODES:
            raise ProtocolError(f"unknown sign mode {sign!r}")
        self.style = style
        self.pool = pool
        # v10.0: a signed Eq. (14). `sign` is None for the paper's symmetric
        # draw. "fold" keeps the paper's magnitude and fixes only the sign;
        # "shift" adds `displacement` block offsets along the direction on top
        # of the paper's noise. The direction itself is set by the training loop
        # through `set_direction`, from Batch 1's two acquisition blocks pushed
        # through the current stem; see `docs/v10-signed-generation.md`.
        self.sign = sign
        self.direction = direction
        self.displacement = float(displacement)
        self._direction_mean: torch.Tensor | None = None
        self._direction_std: torch.Tensor | None = None
        self._direction_report: dict[str, float] | None = None
        self._negative_scale = 0.0

    def style_moments(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. (10)-(11): each sample's style mean and std of the residual."""
        _, residual = self.decompose(z)
        dims = reduce_dims(self.style)
        mean = residual.mean(dim=dims, keepdim=True)
        std = residual.std(dim=dims, keepdim=True, unbiased=False) + EPSILON
        return mean, std

    @torch.no_grad()
    def set_direction(self, z_first: torch.Tensor, z_second: torch.Tensor) -> None:
        """The first-to-second acquisition block offset, in style space.

        `z_first` and `z_second` are the block-3 features of the two acquisition
        sessions of one Batch 1 class (`src.augment.SOURCE_BLOCKS`). The offset
        of their style moments is the direction Eq. (14) is signed by. Computed
        from Batch 1 alone; the training loop refreshes it every epoch because
        the stem it is measured through is itself training.
        """
        if self.sign is None:
            return
        mean_a, std_a = self.style_moments(z_first)
        mean_b, std_b = self.style_moments(z_second)
        self._direction_mean = (mean_b.mean(dim=0, keepdim=True)
                                - mean_a.mean(dim=0, keepdim=True))
        self._direction_std = (std_b.mean(dim=0, keepdim=True)
                               - std_a.mean(dim=0, keepdim=True))

    @property
    def direction_report(self) -> dict[str, float] | None:
        """The last minibatch's direction scale against the paper's noise scale."""
        return self._direction_report

    def decompose(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. (8)-(9): MaxPool then nearest-neighbour upsample, and the rest."""
        pooled = F.max_pool1d(z, kernel_size=self.pool, stride=self.pool)
        pooled = F.interpolate(pooled, size=z.shape[-1], mode="nearest")
        return pooled, z - pooled

    def forward(self, z: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        if z.dim() != 3:
            raise ProtocolError(f"expected [B, C, H], got {tuple(z.shape)}")
        if z.shape[0] < 2:
            # Eqs. (12)-(13) need a batch to estimate the style distribution.
            return z
        pooled, residual = self.decompose(z)
        mean, std = self.style_moments(z)

        # Eqs. (12)-(13). Under the source-only protocol a "batch" is a minibatch
        # of Batch 1, so this is the minibatch-to-minibatch spread of the style.
        mean_centre = mean.mean(dim=0, keepdim=True)
        mean_spread = mean.std(dim=0, keepdim=True, unbiased=False)
        std_centre = std.mean(dim=0, keepdim=True)
        std_spread = std.std(dim=0, keepdim=True, unbiased=False)

        if self.sign is not None and self._direction_mean is None:
            raise ProtocolError(
                f"sign mode {self.sign!r} needs a direction; call set_direction first")

        def sample(centre: torch.Tensor, spread: torch.Tensor,
                   direction: torch.Tensor | None) -> torch.Tensor:
            noise = torch.empty(mean.shape, device=z.device, dtype=z.dtype)
            noise.normal_(generator=generator)
            if self.sign is None:
                return centre + spread * noise
            direction = direction.to(device=z.device, dtype=z.dtype)
            if self.sign == "fold":
                # The paper's magnitude, the block offset's sign: a half-normal
                # draw per style coordinate, pointed along the offset.
                return centre + spread * noise.abs() * torch.sign(direction)
            # "shift": the paper's symmetric noise plus `displacement` block
            # offsets along the direction, the v8.1 construction at block 3.
            return centre + self.displacement * direction + spread * noise

        # Eq. (14) is an ordinary Gaussian and can return a negative scale. The
        # paper says so; this follows it and records how often it happens.
        new_mean = sample(mean_centre, mean_spread, self._direction_mean)
        new_std = sample(std_centre, std_spread, self._direction_std)
        if self.sign is not None:
            self._direction_report = {
                "mean_offset_over_noise": float(
                    self._direction_mean.norm()
                    / (mean_spread.detach().norm() + EPSILON)),
                "std_offset_over_noise": float(
                    self._direction_std.norm()
                    / (std_spread.detach().norm() + EPSILON))}
        self._negative_scale = float((new_std < 0).float().mean())

        restyled = new_std * (residual - mean) / std + new_mean     # Eq. (16)
        return pooled + restyled                                     # Eq. (15)

    @property
    def negative_scale_fraction(self) -> float:
        return self._negative_scale

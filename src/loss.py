"""`L_ce` exactly as the supplement defines it.

Eq. (S2), read from the rendered supplement:

    L_ce = -(1/B) sum_{i=1..B} (1/n) sum_{j=1..n} Y_j^i log( S( FC( Phi(X_j^i) ) ) )

with `S` the softmax, `B` the number of batches, `n` the size of each batch, and
`FC(Phi(.))` the five blocks followed by `FC128 -> BatchNorm -> FC6`.

The inner term is the mean over one minibatch, which is what
`nn.CrossEntropyLoss(reduction="mean")` computes on logits. The outer term is an
**unweighted** mean over minibatches, which differs from a global mean over the
epoch whenever the batch size does not divide the row count: with 445 source rows
at batch size 64 the final batch holds 61 samples, and S2 weights each of those
more than a sample in a full batch. `EpochLoss` reproduces S2 and records the
sample-weighted figure alongside it so the difference stays visible.

The baseline trains on this term alone. `L_MSE` (S4) needs the feature generation
block and `L_con` (S5) needs the `(z_f, z_bar_f)` pair, neither of which a plain
ResNet has; see `proposal.md` section 2.4.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.protocol import ProtocolError


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The inner mean of Eq. (S2) for one minibatch.

    `targets` are zero-based class indices, not the 1..6 labels on disk.
    """
    if logits.ndim != 2:
        raise ProtocolError(f"expected logits [N, C], got {tuple(logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ProtocolError("logits and targets disagree on the batch size")
    return F.cross_entropy(logits, targets, reduction="mean")


class EpochLoss:
    """Accumulates one epoch of minibatch losses under both conventions."""

    def __init__(self) -> None:
        self._batch_means: list[float] = []
        self._weighted_total = 0.0
        self._samples = 0
        self._correct = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor,
               targets: torch.Tensor) -> None:
        value = float(loss.detach())
        count = int(targets.shape[0])
        self._batch_means.append(value)
        self._weighted_total += value * count
        self._samples += count
        self._correct += int((logits.detach().argmax(dim=1) == targets).sum())

    @property
    def s2(self) -> float:
        """Eq. (S2): the unweighted mean of the per-batch means."""
        if not self._batch_means:
            raise ProtocolError("no batches were accumulated")
        return sum(self._batch_means) / len(self._batch_means)

    @property
    def sample_weighted(self) -> float:
        """The global mean over the epoch, for comparison with S2."""
        if not self._samples:
            raise ProtocolError("no samples were accumulated")
        return self._weighted_total / self._samples

    @property
    def accuracy(self) -> float:
        return self._correct / self._samples

    def summary(self) -> dict[str, float]:
        return {"loss_s2": self.s2,
                "loss_sample_weighted": self.sample_weighted,
                "s2_minus_sample_weighted": self.s2 - self.sample_weighted,
                "accuracy": self.accuracy,
                "batches": float(len(self._batch_means)),
                "samples": float(self._samples)}


def loss_module() -> nn.Module:
    """The equivalent `nn.Module`, used by tests to check `cross_entropy`."""
    return nn.CrossEntropyLoss(reduction="mean")

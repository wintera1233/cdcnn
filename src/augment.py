"""Data augmentation: the paper's block, and a directed variant.

The paper's Eq. (5)-(7) displaces a sample by noise drawn from a Gaussian whose
mean and variance are a mix of two samples' own first and second moments:

    mu_j, delta_j = mean and variance of sample j's own 128 values
    k             = another sample of a *different* gas
    G_j = f( N(lam*mu_j + (1-lam)*mu_k, lam*delta_j + (1-lam)*delta_k) ) + X_j

Fig. 2 places this after the `Normal` block. Under any per-sample normalisation
every sample has mean 0 and variance 1, so mu_j = mu_k = 0 and delta_j =
delta_k = 1: the mixing weight is inert and the block degenerates to adding
N(0, 1). Measured, the three values of `lam` give identical results. The
displacement is isotropic over 128 dimensions, so only about 1/sqrt(128) of it
lands along any one direction.

`docs/batch1-internal-drift.md` shows that the drift this project needs to cover
runs along a single direction which Batch 1 can estimate on its own, from the
offset between its two acquisition sessions. The directed variant adds a
displacement along that direction to the paper's isotropic noise:

    x~ = x + s*u + eps,   s ~ U(0, ||offset||),   eps ~ Eq. (7)

Nothing here reads a target batch.
"""

from __future__ import annotations

import numpy as np

from src.protocol import ProtocolError

# The two acquisition blocks of the three classes that have them, as row slices
# of batch1.dat. Verified against the label run-lengths; see
# docs/batch1-internal-drift.md.
SOURCE_BLOCKS = {
    1: ((0, 84), (248, 254)),      # Ethanol
    2: ((84, 172), (254, 264)),    # Ammonia
    3: ((172, 248), (264, 271)),   # Ethylene
}
DIRECTION_SOURCES = ("average", "ethanol", "subspace", "sphere")
AUGMENTATIONS = ("paper", "directed")


def block_offsets(z: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    """The first-block to second-block centroid offset of each class that has two.

    `z` is already through the `Normal` block, which is where Fig. 2 puts the
    augmentation.
    """
    offsets = {}
    for label, ((a, b), (c, d)) in SOURCE_BLOCKS.items():
        if not (y[a:b] == label).all() or not (y[c:d] == label).all():
            raise ProtocolError(
                f"block layout does not match the data for label {label}")
        offsets[label] = z[c:d].mean(axis=0) - z[a:b].mean(axis=0)
    return offsets


def drift_direction(z: np.ndarray, y: np.ndarray,
                    source: str = "average") -> tuple[np.ndarray, float]:
    """A unit drift direction and a displacement scale, from Batch 1 alone.

    `average` takes the mean of all three block offsets. It is the choice a
    practitioner with only Batch 1 would make, having no reason to prefer one
    class, and is the one this project reports as a source-only result.

    `ethanol` takes Ethanol's offset alone. It aligns better with the real drift
    (0.717 against 0.470 mean absolute cosine) but that was established by
    measuring against target data, so a run using it is target-informed and must
    say so.
    """
    if source not in DIRECTION_SOURCES:
        raise ProtocolError(f"unknown direction source {source!r}")
    offsets = block_offsets(z, y)
    if source == "average":
        # Normalise each offset before averaging, so the three classes weigh
        # equally. Averaging the raw offsets would let the largest dominate.
        unit = np.mean([v / np.linalg.norm(v) for v in offsets.values()], axis=0)
        unit = unit / np.linalg.norm(unit)
        scale = float(np.mean([np.linalg.norm(v) for v in offsets.values()]))
    else:
        chosen = offsets[1]
        scale = float(np.linalg.norm(chosen))
        unit = chosen / scale
    if scale <= 0:
        raise ProtocolError("the block offset is degenerate")
    return unit, scale


def paper_noise(z: np.ndarray, y: np.ndarray, lam: float,
                rng: np.random.Generator) -> np.ndarray:
    """Eq. (5)-(7): noise from a Gaussian mixing two samples' own moments."""
    if not 0.0 <= lam <= 1.0:
        raise ProtocolError(f"lambda must lie in [0, 1], got {lam}")
    mean = z.mean(axis=1)
    variance = z.var(axis=1)
    # "k means another gas sensor data different from j in this dataset."
    partner = np.empty(len(z), dtype=np.int64)
    for position, label in enumerate(y):
        other = np.flatnonzero(y != label)
        partner[position] = other[rng.integers(len(other))]
    mixed_mean = lam * mean + (1.0 - lam) * mean[partner]
    mixed_variance = lam * variance + (1.0 - lam) * variance[partner]
    scale = np.sqrt(np.maximum(mixed_variance, 0.0))
    return rng.normal(mixed_mean[:, None], scale[:, None], size=z.shape)


def subspace_displacement(z: np.ndarray, y: np.ndarray,
                          rng: np.random.Generator, displacement: float,
                          rows: int, mode: str = "subspace") -> np.ndarray:
    """A displacement inside the span of all three block offsets, per sample.

    A single direction captures only the first of the drift's several
    components: `reports/drift_dimension.json` puts 70.6 % of the drift energy in
    one component and needs four for 90 %. The three offsets are three observed
    two-month domain displacements, so the domain plausibly moves like some
    combination of them, and their span reaches a good deal more of the drift
    than any one of them does.

    Each offset keeps its own magnitude - the relative sizes are data, not an
    artefact - and each weight is drawn on [0, 1], so the displacement stays
    forward in time, the direction each offset points.
    """
    raw = list(block_offsets(z, y).values())
    scale = float(np.mean([np.linalg.norm(v) for v in raw]))
    unit = np.stack([v / np.linalg.norm(v) for v in raw], axis=0)
    if mode == "subspace":
        # A non-negative combination of the three observed directions, each
        # weighted equally: their raw lengths differ by more than twice and the
        # longest points furthest from the drift, so raw weighting is dominated
        # by the least useful offset.
        weights = rng.uniform(0.0, 1.0, size=(rows, len(unit)))
        return displacement * scale * (weights @ unit)
    if mode == "sphere":
        # Uniform direction within the span, magnitude uniform in [0, T*scale].
        basis, _ = np.linalg.qr(unit.T)
        coefficients = rng.normal(size=(rows, basis.shape[1]))
        coefficients /= np.linalg.norm(coefficients, axis=1, keepdims=True)
        magnitude = rng.uniform(0.0, displacement * scale, size=(rows, 1))
        return magnitude * (coefficients @ basis.T)
    raise ProtocolError(f"unknown subspace mode {mode!r}")


def augment(z: np.ndarray, y: np.ndarray, rng: np.random.Generator, *,
            isotropic: bool = True, direction: str | None = None,
            lam: float = 0.5, displacement: float = 18.0) -> np.ndarray:
    """One augmented copy of every row of `z`, with the same labels.

    `displacement` multiplies the block-offset scale; the drawn magnitude is
    uniform on [0, displacement * ||offset||]. v8.0 used 18, from the
    collection's 36 months over Batch 1's two, which assumed drift accumulates
    linearly. It does not: every real per-class drift is 0.83 to 3.74 block
    offsets. Each variant carries its own value.
    """
    if not isotropic and direction is None:
        raise ProtocolError("an augmentation must be isotropic, directed, or both")
    out = z.copy()
    if isotropic:
        out = out + paper_noise(z, y, lam, rng)
    if direction in ("subspace", "sphere"):
        out = out + subspace_displacement(z, y, rng, displacement, len(z),
                                          mode=direction)
    elif direction is not None:
        unit, scale = drift_direction(z, y, direction)
        magnitude = rng.uniform(0.0, displacement * scale, size=(len(z), 1))
        out = out + magnitude * unit[None, :]
    return out

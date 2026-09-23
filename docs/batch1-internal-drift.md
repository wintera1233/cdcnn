# Batch 1 contains its own drift, and it points the right way

Batch 1 covers months 1 and 2 of a 36-month collection. Two months is a small
fraction of three years but it is not zero, so the source batch should carry a
little of the same drift the target batches carry a lot of. It does, and the part
of it that is measurable is a usable estimate of the drift direction - computed
from Batch 1 alone.

Post-hoc analysis. Everything that compares against the real drift is
target-informed and labelled as such; the estimator itself reads only Batch 1.

## The row order is not a clock

`batch1.dat` is sorted by label, not by acquisition time:

| rows | n | class |
|---|---:|---|
| 0-83 | 84 | Ethanol |
| 84-171 | 88 | Ammonia |
| 172-247 | 76 | Ethylene |
| **248-253** | **6** | **Ethanol** |
| **254-263** | **10** | **Ammonia** |
| **264-270** | **7** | **Ethylene** |
| 271-300 | 30 | Acetaldehyde |
| 301-370 | 70 | Acetone |
| 371-444 | 74 | Toluene |

So a first-half/second-half split of the file is a class split, not a time split.
Within a class the row order is no better: the displacement from a class's first
third to its last third is only 0.58 to 1.41 class radii, and its direction
matches the real drift for three classes, opposes it for two and is null for one.
Whatever the within-class order carries - concentration, most likely - it is not
usable as a drift proxy.

## The block structure is

The first three classes appear **twice**, in a large block and then a small one.
That looks like two acquisition sessions. The offset between them:

| Class | first block | second block | displacement / radius | cosine with real drift |
|---|---:|---:|---:|---:|
| Ethanol | 84 | 6 | 3.46 | **0.882** |
| Ammonia | 88 | 10 | 1.13 | **0.850** |
| Ethylene | 76 | 7 | 3.10 | 0.163 |

Two of the three point along the real three-year drift at cosine 0.85 or better,
at displacements of the same order as the real drift of the gentler classes
(2.3 to 3.1 radii).

## Drift is a sensor phenomenon, not a gas phenomenon

Cosines between the real per-class drift directions:

| | Ethanol | Acetaldehyde | Acetone | Toluene | Ammonia | Ethylene |
|---|---:|---:|---:|---:|---:|---:|
| Ethanol | 1.000 | 0.973 | 0.869 | 0.927 | 0.423 | -0.033 |
| Acetaldehyde | 0.973 | 1.000 | 0.916 | 0.950 | 0.357 | 0.017 |
| Acetone | 0.869 | 0.916 | 1.000 | 0.985 | 0.488 | 0.188 |
| Toluene | 0.927 | 0.950 | 0.985 | 1.000 | 0.521 | 0.119 |
| Ammonia | 0.423 | 0.357 | 0.488 | 0.521 | 1.000 | 0.333 |
| Ethylene | -0.033 | 0.017 | 0.188 | 0.119 | 0.333 | 1.000 |

**Four of the six gases drift along essentially one shared direction**, pairwise
0.87 to 0.985. The two that do not - Ammonia and Ethylene - are also the two that
drift least (3.12 and 2.30 radii) and the two both this project and the paper
score highest on. The dominant drift is a property of the sensor array over time,
not of the analyte.

## The estimator

Because the direction is shared, a block offset measured on one class estimates
the drift of the others. Cosines between each Batch 1 block offset and every
class's real drift:

| Offset measured on | Ethanol | Ammonia | Ethylene | **Acetaldehyde** | Acetone | Toluene |
|---|---:|---:|---:|---:|---:|---:|
| **Ethanol** | 0.882 | 0.398 | -0.067 | **0.850** | 0.671 | 0.750 |
| Ammonia | 0.484 | 0.850 | 0.238 | 0.463 | 0.462 | 0.525 |
| Ethylene | -0.832 | -0.402 | 0.163 | -0.772 | -0.662 | -0.728 |
| mean of the three | 0.452 | 0.717 | 0.283 | 0.458 | 0.399 | 0.464 |

**Batch 1's Ethanol block offset is 0.850 aligned with the real drift of
Acetaldehyde**, the class no configuration in either project can transfer. The
offset is computed entirely from Batch 1.

## What this is for

`docs/why-acetaldehyde.md` established that Acetaldehyde's cloud travels about
eighteen class radii along one consistent direction, that the direction lies
almost entirely inside Batch 1's own variation subspace, and that no augmentation
scale the previous project tested could produce a displacement of that size. The
missing pieces were the direction and the magnitude. The magnitude is eighteen
radii. This gives the direction, from the source batch.

That turns the paper's augmentation block from an undirected noise process into a
directed one, and it is the first thing in this project that could plausibly move
the dead class.

## Three limits, stated plainly

1. **Ethylene's block offset points the wrong way**, cosine -0.83 against
   Ethanol's drift, and averaging the three offsets dilutes the estimate from
   0.850 to 0.458 on Acetaldehyde. Preferring Ethanol's offset is a choice made
   after seeing which one aligns, so it is target-informed and must be declared
   as such if it is used.
2. **The Ethanol second block holds six rows.** Any estimate from it is fragile.
3. **The estimator is source-only; knowing that it works is not.** Every cosine
   in this document was computed against target data. An experiment may use the
   estimator legitimately, but may not claim it was selected without target
   knowledge.

# The dead class, and why it is not the one the paper loses

Computed from the frozen final checkpoints of
`runs/20260922T175939406519Z_baseline_smoke`'s successors - the v7.0 ladder and
the v7.1 grid - with no retraining. Target figures are pooled over Batches 2-10
and averaged over three seeds.

## Ethylene is learned and then lost

| Cell | `Normal` | Batch 1 Ethylene | Target Ethylene |
|---|---|---:|---:|
| `R-txt` | StandardScaler | **1.000** | **0.001** |
| `R-txt-ps@lr0.001` | per-sample | 0.900 | 0.001 |
| `R-txt-ps@lr0.0003` | per-sample | 0.633 | 0.000 |
| `R-txt-ps@lr0.0001` | per-sample | 0.078 | 0.000 |
| `R-fig-ps@lr0.001` | per-sample | 0.900 | 0.027 |
| `R-fig-ps@lr0.0003` | per-sample | 0.567 | 0.000 |
| `R-lite-ps` | per-sample | 0.000 | 0.000 |

Two separate things are happening.

**On the source domain, per-sample normalisation damages Ethylene and nothing
else.** With a Batch-1-fitted `StandardScaler`, `R-txt` learns Ethylene
perfectly, 1.000 on all three seeds. Under per-sample inputs the same class falls
to 0.90, then to 0.63 and 0.08 as the learning rate drops, while every other
class stays above 0.94 in every configuration. Fig. 4 of the paper shows why:
Ethylene occupies a long arm extending away from the main cluster in raw
principal-component space, so what identifies it is largely overall magnitude -
exactly what dividing each row by its own standard deviation removes.

**On the target domain it is lost regardless.** `R-txt` carries Ethylene at
1.000 on Batch 1 and delivers 0.001 on Batches 2-10. No configuration this
project has ever trained exceeds 0.027. Class weighting cannot address this: the
class is already learned perfectly where it can be learned. The identifying
feature is magnitude, and sensor magnitude is precisely what three years of drift
destroys.

## The paper loses a different class

Fig. S3, the paper's own confusion matrices, against the best cell here:

| Class | paper CDCNN | paper CDWC | `R-fig-ps@lr0.0003` |
|---|---:|---:|---:|
| Ethanol | 0.97 | 0.88 | 0.62 |
| Ethylene | **0.95** | 0.87 | **0.00** |
| Ammonia | 0.91 | 0.87 | 0.51 |
| Acetaldehyde | **0.00** | **0.00** | **0.86** |
| Acetone | 0.94 | 0.85 | 0.67 |
| Toluene | 0.92 | 0.27 | 0.39 |

The paper's CDCNN sends **100 % of Acetaldehyde to Ethanol**, and its CDWC does
the same at 0.92. Both are C2 oxygenates, so that confusion is chemically
unsurprising. Ethylene, which this project cannot transfer at all, it recovers at
0.95.

The arithmetic is consistent: Acetaldehyde is 21 % of the target rows, so five
classes near 0.94 with one at zero gives roughly 0.74 against the reported
0.7230.

## What this corrects

An earlier reading in the previous project matched Fig. S3's zero-diagonal class
to Ethylene. That was wrong. The zero diagonal is Acetaldehyde, and the two
projects fail on opposite classes.

It also corrects the conclusion drawn from `docs/channel-and-lr.md`. Recovering
Ethylene is still worth +0.146 by arithmetic, but the gap to the paper is not one
class:

| Class | gap to paper CDCNN |
|---|---:|
| Ethylene | -0.95 |
| Toluene | -0.53 |
| Ammonia | -0.40 |
| Ethanol | -0.35 |
| Acetone | -0.27 |
| Acetaldehyde | **+0.86** |

The paper is ahead on five of six classes, not merely on the one this project
zeroes. Any account of the remaining deficit has to explain all five.

## What follows

- Class weighting is ruled out for Ethylene: source recall is already 1.000 under
  `StandardScaler`.
- Per-sample normalisation is a genuine trade: it is worth +0.093 overall while
  costing Ethylene on the source domain. Those two effects have never been
  separated per class.
- The Ammonia-into-Toluene collapse after Batch 6 remains unexplained and is
  worth 0.117 of target mean on its own.

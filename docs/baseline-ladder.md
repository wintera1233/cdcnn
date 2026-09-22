# Baseline ladder results (v7.0)

Run: `runs/20260922T180104804597Z_baseline_ladder_full`. Twelve checkpoints (four variants x three seeds), all frozen and
hashed before any target file was opened; 412 checkpoints frozen in total
(12 finals plus 400 per-epoch checkpoints of seed 1042). The run's
`leakage_target_access_audit.json` status is `passed`.

Environment: `cdcnn:cu121`, torch 2.5.1+cu121, driver 535.309.01,
RTX 4060 Laptop. `L_ce` only, 100 epochs, SGD(0.001, 0.9, 1e-4),
StepLR(25, 0.5), batch size 64, seeds 1042 / 2024 / 3407.

## The grid

| Variant | Head | `Normal` | Params | Batch 1 | Target mean | SD | Per seed |
|---|---|---|---:|---:|---:|---:|---|
| `R-txt` | flatten | StandardScaler | 2,434,726 | 1.0000 | 0.3777 | 0.0365 | 0.349 / 0.365 / 0.419 |
| `R-txt-ps` | flatten | per-sample | 2,434,726 | 0.9813 | **0.4673** | 0.0443 | 0.506 / 0.477 / 0.419 |
| `R-lite` | GAP | StandardScaler | 353,958 | 0.9551 | 0.2958 | 0.0524 | 0.348 / 0.297 / 0.243 |
| `R-lite-ps` | GAP | per-sample | 353,958 | 0.8787 | 0.3931 | 0.0412 | 0.421 / 0.412 / 0.346 |

Paper's ResNet: **0.6344**. Previous project's best of any kind: 0.5613.

## Factorial effects

Pooled within-cell SD 0.0440; the standard error of a three-against-three
difference is 0.0359, so a difference is called separable above 0.0719.

| Effect | Value | Separable |
|---|---:|---|
| head: flatten -> GAP, under StandardScaler | **-0.0818** | yes |
| head: flatten -> GAP, under per-sample | **-0.0742** | yes |
| `Normal`: StandardScaler -> per-sample, under flatten | **+0.0896** | yes |
| `Normal`: StandardScaler -> per-sample, under GAP | **+0.0973** | yes |
| main effect, head | -0.0780 | |
| main effect, `Normal` | +0.0934 | |
| interaction | +0.0077 | no |

## Findings

1. **Global average pooling hurts, by about -0.078.** The head hypothesis,
   which was the reason for this ladder, is falsified in the opposite direction
   to the one predicted, on both levels of the other factor. The paper's "this
   network doesn't apply the pooling layer" (Section 5.2) is load-bearing. The
   risk recorded in `proposal.md` section 6 - that averaging a 1e4 steady-state
   feature together with six order-1 transients destroys the cross-sensor ratios
   gas identity lives in - is the reading the data supports.

2. **Per-sample normalisation is worth about +0.093**, replicating the previous
   project's +0.109 on a different backbone, and it is the only lever in this
   grid that works.

3. **The two factors are additive.** The interaction is +0.008 against a 0.072
   separability threshold. They do not overlap, contrary to the expectation in
   `proposal.md` section 6.

4. **The flatten head memorises Batch 1 completely**: training accuracy 1.0000 on
   all three seeds of `R-txt`, final loss 0.015-0.018. GAP reaches 0.955.
   Memorisation does not explain the target gap, because the cell that memorises
   most scores higher than the cell that memorises least.

## Two things that did not go to plan

**The correctness check missed its band.** `proposal.md` section 6 predicted
`R-txt` would land near 0.41, the previous project's `B0`, and called a result
outside roughly 0.40-0.42 a sign of a bug. It landed at 0.3777, 1.5 standard
errors below `B0`'s 0.4097.

The likeliest cause is not a bug but the channel decision. `R-txt` caps the
backbone at 128 where `B0` used the figure's 256 in block 4. Two independent
comparisons point the same way and by a similar amount:

| Pair | Previous project | Here | Change |
|---|---:|---:|---:|
| flatten + StandardScaler | `B0` 0.4097 | `R-txt` 0.3777 | -0.032 |
| flatten + per-sample | `B0-PS` 0.5164 | `R-txt-ps` 0.4673 | -0.049 |

Both are above the 0.004 threshold below which cross-run differences should not
be read as effects, and both say the narrower backbone is worse. That is
evidence against the reading of Section 5.2 this ladder adopted and for Fig. 2's
wider channels. It is not conclusive: the comparison crosses two codebases.

**The escalation rule fired.** Every cell's standard deviation exceeds the 0.02
threshold declared in `proposal.md`: 0.0365, 0.0443, 0.0524, 0.0412. Seeds 42
and 123 are held in reserve for this case. Note that the four factor
comparisons are separable anyway, so the escalation affects confidence in the
*levels*, not in the direction of either factor.

## Per-batch accuracy

| Batch | `R-txt` | `R-txt-ps` | `R-lite` | `R-lite-ps` | paper ResNet |
|---|---:|---:|---:|---:|---:|
| 2 | 0.708 | 0.786 | 0.631 | 0.434 | 0.769 |
| 3 | 0.500 | 0.732 | 0.412 | 0.618 | 0.662 |
| 4 | 0.476 | 0.586 | 0.240 | 0.460 | 0.642 |
| 5 | 0.401 | 0.433 | 0.315 | 0.384 | 0.716 |
| 6 | 0.457 | 0.573 | 0.193 | 0.360 | 0.725 |
| 7 | 0.276 | 0.325 | 0.168 | 0.329 | 0.502 |
| 8 | 0.132 | 0.218 | 0.197 | 0.168 | 0.660 |
| 9 | 0.168 | 0.242 | 0.230 | 0.354 | 0.604 |
| 10 | 0.281 | 0.310 | 0.276 | 0.432 | 0.430 |
| **mean** | **0.378** | **0.467** | **0.296** | **0.393** | **0.635** |

The best cell reaches the paper on B10 (0.310 against 0.430 - no) and comes
closest on B2 (0.786 against 0.769, the one batch it beats). The deficit is
concentrated where it was before: B8 0.218 against 0.660, B5 0.433 against
0.716, B9 0.242 against 0.604.

## Status against the success criteria

- **Reproduced**: no. The best cell is 0.4673 against 0.6344, and is also below
  the previous project's best of 0.5613.
- **Explained**: partially. The head is ruled out as the missing factor, and
  ruled out decisively. The Fig. S1 overlay (work plan item 26) has not been
  computed yet.
- **Failed**: on the declared reading, yes. `proposal.md` section 6 names the
  optimizer constants as the next suspect; the channel evidence above suggests
  Fig. 2's widths should be reinstated first.

# The head's normalisation layer (v7.2)

Run: `runs/20260922T184129389577Z_head_norm_full`. Twelve checkpoints (four cells x three seeds), frozen before any
target file was opened; leakage audit `passed`. Fig. 2's channel widths, flatten
head, lr 0.0003, `L_ce` only. All four cells cost 3,156,390 parameters, so
nothing but the two normalisations differs.

## The question

Fig. 2's "Batch Normal" between FC128 and FC6 is `BatchNorm1d(128)`. In eval mode
it applies Batch 1's stored running mean and variance to target activations
recorded up to three years later - the same failure mode as a fitted
`StandardScaler` on the inputs, which v7.0 measured at -0.093. `LayerNorm(128)`
normalises each sample against itself and stores nothing.

The hypothesis: per-sample input normalisation costs Ethylene because it divides
out the magnitude that identifies the class, while a LayerNorm head might supply
the same drift robustness without touching the input, so `StandardScaler` inputs
with a LayerNorm head would keep both.

## Results

| Input  | Head norm | Batch 1 | Target mean | SD |
|---|---|---:|---:|---:|
| StandardScaler | BatchNorm | 0.9955 | 0.4105 | 0.0173 |
| StandardScaler | LayerNorm | 1.0000 | 0.4760 | 0.0040 |
| **per-sample** | **BatchNorm** | 0.9633 | **0.5222** | 0.0277 |
| per-sample | LayerNorm | 0.9685 | 0.4468 | 0.0022 |

`R-fig-ps@lr0.0003` returns 0.5222 for the third time, matching v7.1 exactly.

Pooled within-cell SD 0.0165, so a difference is separable above 0.0269. Unlike
v7.1, **every comparison clears it**:

| Effect | Value | Separable |
|---|---:|---|
| head BatchNorm -> LayerNorm, StandardScaler inputs | **+0.0654** | yes |
| head BatchNorm -> LayerNorm, per-sample inputs | **-0.0754** | yes |
| input StandardScaler -> per-sample, BatchNorm head | **+0.1116** | yes |
| input StandardScaler -> per-sample, LayerNorm head | -0.0292 | yes |
| **interaction** | **-0.1408** | |

## Finding

**The two normalisations are substitutes, and applying both is worse than
applying either.** Each helps substantially alone - per-sample inputs are worth
+0.112 under a BatchNorm head, a LayerNorm head is worth +0.065 under
StandardScaler inputs - and together they land at 0.4468, below both.

They do the same thing in two places: replace stored Batch 1 statistics with the
sample's own. Doing it twice removes the same information twice.

This also replicates the previous project's +0.085 for swapping BatchNorm for
LayerNorm at this layer. Measured under the same condition it was measured in
then, StandardScaler inputs, the value here is +0.0654.

**The hypothesis is refuted.** A LayerNorm head does supply drift robustness
without touching the input, but it only reaches 0.4760, well short of per-sample
inputs with a BatchNorm head at 0.5222. Per-sample at the input is simply the
better of the two, and the two cannot be stacked.

## Prohibited

`BatchNorm1d(track_running_stats=False)` would normalise a target batch by its
own statistics at inference. That is test-time adaptation on target data, and it
makes a prediction depend on which other samples share its batch. It is
prohibited, and `tests/test_baseline.py` asserts that every `_BatchNorm` in every
variant keeps `track_running_stats` true and a live `running_mean`.

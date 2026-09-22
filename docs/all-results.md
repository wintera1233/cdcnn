# All results: target mean on Batches 2-10 (v7)

Every cell this branch has trained, sorted by target mean. All are the plain
ResNet baseline with `L_ce` only - no augmentation, no feature generation, no
contrastive loss. Three seeds each (1042, 2024, 3407); SD is across seeds.

| Cell | run | channels | head | input `Normal` | head norm | lr | Batch 1 | Target mean | SD |
|---|---|---|---|---|---|---:|---:|---:|---:|
| `R-fig-ps@lr0.0003` | v7.1 | Fig. 2 | flatten | per-sample | BatchNorm | 0.0003 | 0.9633 | **0.5222** | 0.0277 |
| `R-fig-ps@lr0.0003` | v7.2 | Fig. 2 | flatten | per-sample | BatchNorm | 0.0003 | 0.9633 | **0.5222** | 0.0277 |
| `R-txt-ps@lr0.0003` | v7.1 | capped 128 | flatten | per-sample | BatchNorm | 0.0003 | 0.9700 | **0.4928** | 0.0062 |
| `R-fig-ps@lr0.001` | v7.1 | Fig. 2 | flatten | per-sample | BatchNorm | 0.001 | 0.9813 | **0.4770** | 0.0362 |
| `R-fig-ln@lr0.0003` | v7.2 | Fig. 2 | flatten | scaler | LayerNorm | 0.0003 | 1.0000 | **0.4760** | 0.0040 |
| `R-fig-ps@lr0.0001` | v7.1 | Fig. 2 | flatten | per-sample | BatchNorm | 0.0001 | 0.9318 | **0.4724** | 0.0183 |
| `R-txt-ps` | v7.0 | capped 128 | flatten | per-sample | BatchNorm | 0.001 | 0.9813 | **0.4673** | 0.0443 |
| `R-txt-ps@lr0.001` | v7.1 | capped 128 | flatten | per-sample | BatchNorm | 0.001 | 0.9813 | **0.4673** | 0.0443 |
| `R-txt-ps@lr0.0001` | v7.1 | capped 128 | flatten | per-sample | BatchNorm | 0.0001 | 0.9303 | **0.4621** | 0.0233 |
| `R-fig-ps-ln@lr0.0003` | v7.2 | Fig. 2 | flatten | per-sample | LayerNorm | 0.0003 | 0.9685 | **0.4468** | 0.0022 |
| `R-fig@lr0.0003` | v7.2 | Fig. 2 | flatten | scaler | BatchNorm | 0.0003 | 0.9955 | **0.4105** | 0.0173 |
| `R-lite-ps` | v7.0 | capped 128 | GAP | per-sample | BatchNorm | 0.001 | 0.8787 | **0.3931** | 0.0412 |
| `R-txt` | v7.0 | capped 128 | flatten | scaler | BatchNorm | 0.001 | 1.0000 | **0.3777** | 0.0365 |
| `R-lite` | v7.0 | capped 128 | GAP | scaler | BatchNorm | 0.001 | 0.9551 | **0.2958** | 0.0524 |

## References

| Model | Target mean | Source |
|---|---:|---|
| paper CDCNN | 0.7230 | Table 3 |
| paper CDWC | 0.6705 | Table 3 |
| **paper ResNet** | **0.6344** | Table 3, the target of this branch |
| previous project best (`B0-stab-PS`) | 0.5613 | `exp/a3-confound-ablation` |
| previous project `B0` | 0.4097 | `exp/a3-confound-ablation` |

## The best cell, per batch

`R-fig-ps@lr0.0003`, target mean 0.5222.

| Batch | this branch | paper ResNet | gap |
|---|---:|---:|---:|
| 2 | 0.827 | 0.769 | +0.058 |
| 3 | 0.729 | 0.662 | +0.067 |
| 4 | 0.673 | 0.642 | +0.030 |
| 5 | 0.501 | 0.716 | **-0.215** |
| 6 | 0.688 | 0.725 | -0.038 |
| 7 | 0.466 | 0.502 | -0.036 |
| 8 | 0.171 | 0.660 | **-0.489** |
| 9 | 0.255 | 0.604 | **-0.349** |
| 10 | 0.390 | 0.430 | -0.040 |

Six of the nine batches match or beat the paper's ResNet. The whole deficit is
B5, B8 and B9, and its cause is documented in `docs/dead-class.md`.

## What moved the number, and what did not

| Change | Effect | Separable | Evidence |
|---|---:|---|---|
| per-sample inputs instead of `StandardScaler`, BatchNorm head | **+0.112** | yes | `docs/head-normalisation.md` |
| LayerNorm head instead of BatchNorm, `StandardScaler` inputs | **+0.065** | yes | `docs/head-normalisation.md` |
| lr 0.001 -> 0.0003, Fig. 2 widths | +0.045 | no | `docs/channel-and-lr.md` |
| Fig. 2 channel widths instead of the prose's cap at 128 | +0.029 | no | `docs/channel-and-lr.md` |
| lr 0.001 -> 0.0001 | -0.005 | no | `docs/channel-and-lr.md` |
| LayerNorm head **on top of** per-sample inputs | **-0.075** | yes | `docs/head-normalisation.md` |
| global average pooling instead of flatten | **-0.078** | yes | `docs/baseline-ladder.md` |

The two normalisations are substitutes with an interaction of -0.141: each is
worth a lot alone and using both is worse than using either.


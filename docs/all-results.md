# All results: target mean on Batches 2-10 (v7)

The narrative and the evidence live in `baseline.md`; this file is the raw table.

Every cell this branch has trained, sorted by target mean. All are the plain
ResNet baseline with `L_ce` only - no augmentation, no feature generation, no
contrastive loss. Seeds 1042, 2024, 3407, and for the v7.5 grid also 42 and 123.

| Cell | run | channels | head | input `Normal` | head norm | lr | seeds | Batch 1 | Target mean | SD |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| `R-fig-logps@lr0.0003` | v7.5 | Fig. 2 | flatten | signed-log→per-sample | BatchNorm | 0.0003 | 5 | 0.9906 | **0.5556** | 0.0137 |
| `R-fig-ps@lr0.0003` | v7.5 | Fig. 2 | flatten | per-sample | BatchNorm | 0.0003 | 5 | 0.9636 | **0.5232** | 0.0203 |
| `R-fig-ssps@lr0.0003` | v7.5 | Fig. 2 | flatten | SS→per-sample | BatchNorm | 0.0003 | 5 | 0.9982 | **0.5034** | 0.0138 |
| `R-txt-ps@lr0.0003` | v7.1 | capped 128 | flatten | per-sample | BatchNorm | 0.0003 | 3 | 0.9700 | **0.4928** | 0.0062 |
| `R-fig-ps@lr0.001` | v7.1 | Fig. 2 | flatten | per-sample | BatchNorm | 0.001 | 3 | 0.9813 | **0.4770** | 0.0362 |
| `R-fig-ln@lr0.0003` | v7.2 | Fig. 2 | flatten | StandardScaler | LayerNorm | 0.0003 | 3 | 1.0000 | **0.4760** | 0.0040 |
| `R-fig-ps@lr0.0001` | v7.1 | Fig. 2 | flatten | per-sample | BatchNorm | 0.0001 | 3 | 0.9318 | **0.4724** | 0.0183 |
| `R-txt-ps` | v7.0 | capped 128 | flatten | per-sample | BatchNorm | 0.001 | 3 | 0.9813 | **0.4673** | 0.0443 |
| `R-txt-ps@lr0.0001` | v7.1 | capped 128 | flatten | per-sample | BatchNorm | 0.0001 | 3 | 0.9303 | **0.4621** | 0.0233 |
| `R-fig-ps-ln@lr0.0003` | v7.2 | Fig. 2 | flatten | per-sample | LayerNorm | 0.0003 | 3 | 0.9685 | **0.4468** | 0.0022 |
| `R-fig-grp@lr0.0003` | v7.5 | Fig. 2 | flatten | per-statistic-group | BatchNorm | 0.0003 | 5 | 0.9978 | **0.4263** | 0.0183 |
| `R-fig@lr0.0003` | v7.2 | Fig. 2 | flatten | StandardScaler | BatchNorm | 0.0003 | 3 | 0.9955 | **0.4105** | 0.0173 |
| `R-lite-ps` | v7.0 | capped 128 | GAP | per-sample | BatchNorm | 0.001 | 3 | 0.8787 | **0.3931** | 0.0412 |
| `R-txt` | v7.0 | capped 128 | flatten | StandardScaler | BatchNorm | 0.001 | 3 | 1.0000 | **0.3777** | 0.0365 |
| `R-lite` | v7.0 | capped 128 | GAP | StandardScaler | BatchNorm | 0.001 | 3 | 0.9551 | **0.2958** | 0.0524 |

## References

| Model | Target mean | Source |
|---|---:|---|
| paper CDCNN | 0.7230 | Table 3 |
| paper CDWC | 0.6705 | Table 3 |
| **paper ResNet** | **0.6344** | Table 3, the target of this branch |
| previous project best (`B0-stab-PS`) | 0.5613 | `exp/a3-confound-ablation` |
| previous project `B0` | 0.4097 | `exp/a3-confound-ablation` |

## The best cell, per batch

`R-fig-logps@lr0.0003`, target mean 0.5556 over 5 seeds.

| Batch | this branch | paper ResNet | gap |
|---|---:|---:|---:|
| 2 | 0.888 | 0.769 | **+0.119** |
| 3 | 0.723 | 0.662 | +0.062 |
| 4 | 0.591 | 0.642 | -0.051 |
| 5 | 0.566 | 0.716 | **-0.150** |
| 6 | 0.560 | 0.725 | **-0.166** |
| 7 | 0.499 | 0.502 | -0.003 |
| 8 | 0.231 | 0.660 | **-0.429** |
| 9 | 0.397 | 0.604 | **-0.206** |
| 10 | 0.544 | 0.430 | **+0.113** |

Four of the nine batches match or beat the paper's ResNet. The deficit is
concentrated in B8, B9, B6 and B5; `baseline.md` section 5 gives the cause.

## What moved the number, and what did not

| Change | Effect | Separable | Evidence |
|---|---:|---|---|
| per-sample inputs instead of `StandardScaler`, BatchNorm head | **+0.112** | yes | `baseline.md` §4.3 |
| LayerNorm head instead of BatchNorm, `StandardScaler` inputs | **+0.065** | yes | `baseline.md` §4.3 |
| lr 0.001 -> 0.0003, Fig. 2 widths | +0.045 | no | `baseline.md` §4.2 |
| **signed-log before per-sample** | **+0.032** | **yes (5 seeds)** | `baseline.md` §4.4 |
| Fig. 2 channel widths instead of the prose's cap at 128 | +0.029 | no | `baseline.md` §4.2 |
| averaging three seeds' softmax | +0.004 | no | `baseline.md` §4.4 |
| lr 0.001 -> 0.0001 | -0.005 | no | `baseline.md` §4.2 |
| `StandardScaler` before per-sample | -0.020 | no | `baseline.md` §4.4 |
| LayerNorm head **on top of** per-sample inputs | **-0.075** | yes | `baseline.md` §4.3 |
| global average pooling instead of flatten | **-0.078** | yes | `baseline.md` §4.1 |
| per-statistic-group normalisation | **-0.097** | yes | `baseline.md` §4.4 |


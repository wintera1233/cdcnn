# All results: target mean on Batches 2-10 (v7)

The narrative and the evidence live in `baseline.md`; this file is the raw table.

The plain-baseline cells, sorted by target mean. All are the bare
ResNet with `L_ce` only - no augmentation, no feature generation, no
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

## v12.0: the drift axis projected out of the input

`R-fig-logps` plus a fixed projection `x - U U^T x` after the Normal block, `U`
from Batch 1's two acquisition sessions. No augmentation, no block, five seeds;
see `docs/v12-drift-projection.md`. Run `20261005T145146298993Z_baseline_ladder_full`.

| Cell | removed | k | Batch 1 | Target mean | SD | vs `R-fig-logps` | separable |
|---|---|---:|---:|---:|---:|---:|---|
| `R-proj-eth` | Ethanol's session offset (target-informed choice) | 1 | 0.9910 | **0.5747** | 0.0119 | +0.0191 | yes |
| `R-proj-axis` | the common axis of the three offsets | 1 | 0.9897 | **0.5684** | 0.0088 | +0.0128 | no |
| `R-proj-sub3` | the span of the three offsets | 3 | 0.9910 | **0.5610** | 0.0164 | +0.0055 | no |
| `R-fig-logps` (re-run) | — | 0 | 0.9906 | **0.5556** | 0.0137 | — | — |

## Cells with CDCNN components

Every cell below is `R-fig-logps` plus the named component, at lr 0.0003 on five
seeds. The v8.1 and v9.0 cells all carry the directed augmentation, whose
direction and displacement were both chosen against target data, so they are
target-informed and report an upper bound. **The source-only settled number
remains `R-fig-logps` at 0.5556.**

| Cell | run | augmentation | feature generation | `L_MSE` | Batch 1 | Target mean | SD | vs its own reference |
|---|---|---|---|---:|---:|---:|---:|---:|
| `R-con-t5` | v11.0 | directed, T=2 | per-position + L_con tau 0.5, lambda 0.5 | λ=0.5 | 0.9942 | **0.5760** | 0.0096 | −0.0014 |
| `R-con-shift` | v11.0 | directed, T=2 | shifted block + L_con tau 0.07 | λ=0.5 | 0.9933 | **0.5666** | 0.0078 | −0.0107 |
| `R-con` | v11.0 | directed, T=2 | per-position + L_con tau 0.07, lambda 0.5 (the paper's CDCNN) | λ=0.5 | 0.9933 | **0.5635** | 0.0102 | −0.0138 |
| `R-gen-shift` | v10.0 | directed, T=2 | per-position, Eq. (14) + 2 block offsets along Batch 1's style-space offset | λ=0.5 | 0.9937 | **0.5796** | 0.0077 | +0.0023 |
| `R-gen-m10` | v9.0 | directed, T=2 | per-position | λ=1.0 | 0.9933 | **0.5776** | 0.0062 | +0.0006 |
| `R-gen` | v9.0 | directed, T=2 | per-position | λ=0.5 | 0.9942 | **0.5774** | 0.0065 | +0.0004 |
| `R-gen` (v10.0 re-run) | v10.0 | directed, T=2 | per-position | λ=0.5 | 0.9942 | **0.5773** | 0.0069 | +0.0003 |
| `R-aug-t2` (v10.0 re-run) | v10.0 | directed, T=2 | — | — | 0.9951 | **0.5770** | 0.0078 | +0.0000 |
| `R-gen-sign` | v10.0 | directed, T=2 | per-position, Eq. (14) folded along the offset's sign | λ=0.5 | 0.9942 | **0.5767** | 0.0115 | −0.0006 |
| `R-aug-t2` | v8.1 | directed, T=2 | — | — | 0.9951 | **0.5770** | 0.0078 | +0.0214 |
| `R-gen-ce2` | v9.0 | directed, T=2 | per-position, `L_ce` on both | λ=0.5 | 0.9937 | **0.5749** | 0.0062 | -0.0022 |
| `R-aug-t3` | v8.1 | directed, T=3 | — | — | — | **0.5731** | 0.0104 | +0.0175 |
| `R-aug-t4` | v8.1 | directed, T=4 | — | — | — | **0.5675** | 0.0108 | +0.0119 |
| `R-aug-ethd` | v8.0 | directed, T=18 | — | — | 0.9924 | **0.5670** | 0.0096 | +0.0115 |
| `R-fig-logps` | v7.5 | — | — | — | 0.9906 | **0.5556** | 0.0137 | — |
| `R-aug-eth` | v8.0 | directed + isotropic, T=18 | — | — | 0.9825 | **0.5462** | 0.0190 | -0.0093 |
| `R-aug-paper` | v8.0 | isotropic (Eq. 7) | — | — | 0.9811 | **0.5340** | 0.0249 | -0.0216 |

The augmentation reference is `R-fig-logps`; the feature generation reference is
`R-aug-t2`, because every v9 cell carries that augmentation. The v10.0 and v11.0 cells are
compared against `R-gen` of their own runs; see `docs/v10-signed-generation.md`
and `docs/v11-contrastive.md`.

**Feature generation is worth nothing under this protocol.** The three v9 cells
span 0.5749 to 0.5776 against a reference of 0.5770, and the separability
threshold at five seeds is 0.0089 to 0.0091. `exp/a3-confound-ablation` measured
the same component at +0.001 to +0.014 on a different backbone; v9.0 narrows
that to +0.0004. See `docs/v9-feature-generation.md`.

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


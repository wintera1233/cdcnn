# There is no source-only stopping rule here

Runs: `runs/20260922T190507701285Z_cv_early_stop` (accuracy only) and
`runs/20260922T191507710557Z_cv_early_stop_loss` (accuracy and cross-entropy).
5-fold stratified cross-validation on Batch 1, folds fixed by the constant
20260923 so only the initialisation varies with seed, the `Normal` block fitted
on each fold's training part. Four cells, three seeds, sixty folds per run.
No target file was opened; both leakage audits pass on that ground.

## The question

Target accuracy peaks long before epoch 100 for some configurations and not at
all for others, and the peak sits at Batch 1 training accuracies anywhere from
0.58 to 0.975, so no threshold on training accuracy can find it. Held-out
accuracy and held-out loss are different quantities. Can either locate the peak?

## Answer: no, and not because they are noisy

| Cell | target peaks at | held-out accuracy peaks at | held-out loss bottoms at |
|---|---:|---:|---:|
| `R-fig@lr0.0003` | **7** | 60 | **100** |
| `R-fig-ps@lr0.001` | **40** | 93 | 87 |
| `R-fig-ps@lr0.0003` | **70** | 99 | 98 |
| `R-fig@lr0.001` | - | 66 | 89 |

Both signals answer "late" for every cell regardless of when the target actually
peaked, spanning epochs 60-100 against target peaks at 7, 40 and 70.

## The decisive cell

`R-fig@lr0.0003` loses 0.164 of target mean between epoch 7 and epoch 100. Over
exactly that window:

| Epoch | held-out accuracy | held-out loss | target mean |
|---:|---:|---:|---:|
| 5 | 0.8449 | 1.5934 | 0.5296 |
| **7** | 0.9250 | 1.4284 | **0.5605** |
| 10 | 0.9513 | 1.0126 | 0.4811 |
| 15 | 0.9612 | 0.3047 | 0.4495 |
| 20 | 0.9695 | 0.1871 | 0.4212 |
| 40 | 0.9739 | 0.1481 | 0.4207 |
| 100 | 0.9754 | 0.1309 | **0.3968** |

Across the collapse, held-out accuracy **rises** by 0.050 and held-out loss
**falls** by 1.298. Held-out loss never turns upward at all: its minimum is
epoch 100.

The two source-only signals are not merely uninformative about the target over
this window. They are anti-correlated with it. A rule that stopped when held-out
loss stopped improving would run to epoch 100 and take the worst available model.

## Why

Held-out cross-validation measures generalisation to **the same domain**:
different samples, same sensors, same two months. The failure this project needs
to detect is generalisation to a **shifted** domain, three years and sixteen
ageing sensors away. Fitting Batch 1 harder improves the first and destroys the
second. Nothing computed from Batch 1 alone can see that trade.

## Consequence

Under a strict source-only protocol on this dataset, the number of epochs is not
selectable. It must be fixed in advance by fiat, which is what the paper does
with its 100 epochs and what this project therefore also does.

The cost is measurable. Stopping each cell at its own target peak - which no
legitimate rule can find - would be worth up to +0.057 on the best cell and
+0.164 on `R-fig`. That is an upper bound this protocol leaves on the table, and
it is smaller than the 0.146 the dead Ethylene class costs.

## What would be legitimate, and is untried

Validating on Batch 1 samples perturbed by a **known synthetic drift** rather than
on clean held-out samples. That stays inside Batch 1, so it is source-only, and it
would measure the right kind of generalisation. It is also, in effect, what the
paper's data augmentation block builds - which would make it the first component
of CDCNN with an independent reason to exist rather than an assertion to test.

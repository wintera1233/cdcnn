# Why Ethylene, and why both implementations lose the same class

Post-hoc diagnosis, run after every checkpoint was frozen and evaluated. It
selects nothing; it only measures the dataset's geometry.

## The observation that started it

Two independent implementations - v6.3 on `exp/a3-confound-ablation` and v7 here -
share no architecture, no normalisation, no head and no optimizer settings, and
both put Ethylene's target recall at zero in all nine batches. The paper loses
Acetaldehyde instead. When two unrelated models fail identically, the cause is
more likely in the data than in either model.

## Every target batch's Ethylene lands on Batch 1's Acetone

Class centroids in the per-sample normalised space the model sees. For each
target batch and class, the nearest Batch 1 class centroid:

| Batch | Acetone | Acetaldehyde | Ethanol | Ethylene | Ammonia | Toluene |
|---|---|---|---|---|---|---|
| 2 | Acetone | Acetaldehyde | Ethanol | **-> Acetone** | Ammonia | Toluene |
| 3 | Acetone | Acetaldehyde | Ethanol | **-> Acetone** | -> Acetone | - |
| 4 | -> Ethanol | Acetaldehyde | -> Acetaldehyde | **-> Acetone** | -> Acetone | - |
| 5 | Acetone | Acetaldehyde | Ethanol | **-> Acetone** | -> Acetone | - |
| 6 | Acetone | Acetaldehyde | Ethanol | **-> Acetone** | -> Acetone | -> Acetone |
| 7 | Acetone | Acetaldehyde | -> Acetaldehyde | **-> Acetone** | -> Acetone | -> Acetone |
| 8 | -> Ethanol | Acetaldehyde | -> Acetaldehyde | **-> Ethanol** | -> Acetone | -> Acetone |
| 9 | Acetone | Acetaldehyde | -> Acetaldehyde | **-> Acetone** | -> Acetone | -> Acetone |
| 10 | Acetone | -> Ethanol | -> Acetaldehyde | **-> Acetone** | -> Acetone | -> Acetone |
| **misplaced** | 2/9 | 1/9 | 5/9 | **9/9** | 8/9 | 5/6 |

**Ethylene never once lands nearest its own Batch 1 centroid.** A decision
boundary fitted on Batch 1 assigns that region to Acetone, so a Batch-1-trained
classifier is not merely failing to recognise Ethylene - it is correctly applying
a rule that the target data has invalidated.

The three classes that collapse in both implementations - Ethylene, Ammonia,
Toluene - are exactly the three that are misplaced most often.

## It is not the input representation

Misplaced batches out of nine, per class, under five input spaces:

| Space | Acetone | Acetaldehyde | Ethanol | **Ethylene** | Ammonia | Toluene |
|---|---:|---:|---:|---:|---:|---:|
| raw features | 5/9 | 3/9 | 0/9 | **9/9** | 9/9 | 5/6 |
| `StandardScaler` | 7/9 | 2/9 | 6/9 | **9/9** | 7/9 | 3/6 |
| per-sample (in use) | 2/9 | 1/9 | 5/9 | **9/9** | 8/9 | 5/6 |
| `StandardScaler` then per-sample | 0/9 | 1/9 | 0/9 | **9/9** | 7/9 | 4/6 |
| signed-log then per-sample | 1/9 | 1/9 | 1/9 | **9/9** | 8/9 | 5/6 |

Ethylene is 9/9 in every one, including the raw features. No normalisation this
project can write moves it back.

Worth noting separately: **`StandardScaler` followed by per-sample** puts both
Acetone and Ethanol at 0/9, better than the per-sample normalisation currently in
use. That is an untrained configuration and a concrete lead.

## The drift is large, and it is the most directionally consistent of the six

Centroid displacement divided by the class's own Batch 1 radius, averaged over
the nine batches, and the mean cosine between drift vectors across batch pairs:

| Gas | displacement / radius | mean cosine | min cosine |
|---|---:|---:|---:|
| Ethanol | 2.30 | 0.759 | 0.326 |
| Acetaldehyde | 3.12 | 0.340 | -0.740 |
| Acetone | 7.89 | 0.652 | -0.110 |
| Ammonia | 12.08 | 0.804 | 0.377 |
| **Ethylene** | **18.32** | **0.904** | **0.761** |
| Toluene | 30.53 | 0.639 | 0.129 |

Ethylene moves about eighteen times its own cloud radius, and it moves in very
nearly one fixed direction: every one of the thirty-six batch pairs agrees to
cosine 0.761 or better, the tightest of any class. This is not noise. It is a
single systematic displacement that grows with time.

## The direction is reachable from Batch 1

Fraction of each class's mean drift direction that lies in the span of Batch 1's
top-k within-class variation axes:

| Gas | k=1 | k=2 | k=5 | k=10 |
|---|---:|---:|---:|---:|
| Acetone | 0.591 | 0.711 | 0.932 | 0.991 |
| Acetaldehyde | 0.239 | 0.343 | 0.799 | 0.976 |
| Ethanol | 0.000 | 0.036 | 0.592 | 0.969 |
| **Ethylene** | **0.483** | **0.687** | **0.969** | **0.998** |
| Ammonia | 0.383 | 0.444 | 0.798 | 0.999 |
| Toluene | 0.452 | 0.533 | 0.864 | 0.999 |

**96.9 % of Ethylene's drift direction lies in the top five axes of Batch 1's own
within-class variation**, and 48.3 % in the first axis alone. The direction is not
orthogonal to anything Batch 1 contains; it is one of Batch 1's own principal
directions, travelled far.

## What this implies

1. **The failure is a property of the dataset under this protocol**, which is why
   two unrelated implementations reproduce it exactly. No architecture,
   normalisation, class weight or stopping rule addresses it.
2. **A domain-generalisation method has a real target here.** The displacement is
   along a direction Batch 1 already contains, so augmenting Batch 1 along its own
   principal directions can in principle reach where target Ethylene sits. This is
   the first concrete job for the paper's augmentation block that does not rest on
   the paper's say-so.
3. **The magnitude is the parameter that matters, and it was never swept far
   enough.** Reaching the target region needs a displacement of order eighteen
   class radii. The previous project tested augmentation noise at 0, 0.05, 0.2 and
   0.5 in the augmentation's own units and measured -0.013 to -0.009 at every
   scale. Those scales cannot produce a displacement of that size, so the
   experiment answered a different question than the one that matters.
4. **It reopens how the paper reaches 0.95 on Ethylene.** Either its augmentation
   does travel that far, or its numbers are not strictly source-only. The first is
   testable here.

## The next experiment this suggests

Sweep augmentation displacement by orders of magnitude rather than by small
factors - measured in units of the class radius, not in the noise parameter - and
check at each scale whether Batch 1's augmented Ethylene cloud covers the region
where target Ethylene actually sits. The coverage check is computable before any
training, from Batch 1 plus the geometry above.

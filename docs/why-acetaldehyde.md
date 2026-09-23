# Why Acetaldehyde, and why both implementations lose the same class

Post-hoc diagnosis, run after every checkpoint was frozen and evaluated. It
selects nothing; it only measures the dataset's geometry. Gas names follow
`docs/label-mapping.md`.

## The observation that started it

Two independent implementations - v6.3 on `exp/a3-confound-ablation` and v7 here -
share no architecture, no normalisation, no head and no optimizer settings, and
both put label 4's target recall at zero in all nine batches. The paper's Fig. S3
also has exactly one dead class. When two unrelated models fail identically, the
cause is more likely in the data than in either model.

Under the adopted mapping all three lose the **same** class, Acetaldehyde, and
all three send it substantially to Ethanol:

| | recall | where it goes |
|---|---:|---|
| this project | 0.000 | 37.7 % Ethanol, 59.3 % Toluene |
| paper CDCNN | 0.00 | 100 % Ethanol |
| paper CDWC | 0.00 | 92 % Ethanol |

Acetaldehyde (CH3CHO) and Ethanol (C2H5OH) are both C2 oxygenates.

## Every target batch's Acetaldehyde lands on Batch 1's Ethanol

Class centroids in the per-sample normalised space the model sees. For each
target batch and class, the nearest Batch 1 class centroid:

| Batch | Ethanol | Ammonia | Ethylene | Acetaldehyde | Acetone | Toluene |
|---|---|---|---|---|---|---|
| 2 | Ethanol | Ammonia | Ethylene | **-> Ethanol** | Acetone | Toluene |
| 3 | Ethanol | Ammonia | Ethylene | **-> Ethanol** | -> Ethanol | - |
| 4 | -> Ethylene | Ammonia | -> Ammonia | **-> Ethanol** | -> Ethanol | - |
| 5 | Ethanol | Ammonia | Ethylene | **-> Ethanol** | -> Ethanol | - |
| 6 | Ethanol | Ammonia | Ethylene | **-> Ethanol** | -> Ethanol | -> Ethanol |
| 7 | Ethanol | Ammonia | -> Ammonia | **-> Ethanol** | -> Ethanol | -> Ethanol |
| 8 | -> Ethylene | Ammonia | -> Ammonia | **-> Ethylene** | -> Ethanol | -> Ethanol |
| 9 | Ethanol | Ammonia | -> Ammonia | **-> Ethanol** | -> Ethanol | -> Ethanol |
| 10 | Ethanol | -> Ethylene | -> Ammonia | **-> Ethanol** | -> Ethanol | -> Ethanol |
| **misplaced** | 2/9 | 1/9 | 5/9 | **9/9** | 8/9 | 5/6 |

**Acetaldehyde never once lands nearest its own Batch 1 centroid**, and eight
times of nine it lands on Ethanol's. A decision boundary fitted on Batch 1
assigns that region to Ethanol, so a Batch-1-trained classifier is not merely
failing to recognise Acetaldehyde - it is correctly applying a rule that the
target data has invalidated. That is also precisely the error the paper makes.

## It is not the input representation

Misplaced batches out of nine, per class, under five input spaces:

| Space | Ethanol | Ammonia | Ethylene | **Acetaldehyde** | Acetone | Toluene |
|---|---:|---:|---:|---:|---:|---:|
| raw features | 5/9 | 3/9 | 0/9 | **9/9** | 9/9 | 5/6 |
| `StandardScaler` | 7/9 | 2/9 | 6/9 | **9/9** | 7/9 | 3/6 |
| per-sample (in use) | 2/9 | 1/9 | 5/9 | **9/9** | 8/9 | 5/6 |
| `StandardScaler` then per-sample | 0/9 | 1/9 | 0/9 | **9/9** | 7/9 | 4/6 |
| signed-log then per-sample | 1/9 | 1/9 | 1/9 | **9/9** | 8/9 | 5/6 |

Acetaldehyde is 9/9 in every one, including the raw features. No normalisation
this project can write moves it back.

Worth noting separately: **`StandardScaler` followed by per-sample** puts both
Ethanol and Ethylene at 0/9, better than the per-sample normalisation currently
in use. That is an untrained configuration and a concrete lead.

## The drift is large, and it is the most directionally consistent of the six

Centroid displacement divided by the class's own Batch 1 radius, averaged over
the nine batches, and the mean cosine between drift vectors across batch pairs:

| Gas | displacement / radius | mean cosine | min cosine |
|---|---:|---:|---:|
| Ethylene | 2.30 | 0.759 | 0.326 |
| Ammonia | 3.12 | 0.340 | -0.740 |
| Ethanol | 7.89 | 0.652 | -0.110 |
| Acetone | 12.08 | 0.804 | 0.377 |
| **Acetaldehyde** | **18.32** | **0.904** | **0.761** |
| Toluene | 30.53 | 0.639 | 0.129 |

Acetaldehyde moves about eighteen times its own cloud radius, and it moves in
very nearly one fixed direction: every one of the thirty-six batch pairs agrees
to cosine 0.761 or better, the tightest of any class. This is not noise. It is a
single systematic displacement that grows with time.

The two classes that drift least, Ethylene and Ammonia, are the two the paper
scores highest on after Ethanol (0.95 and 0.91), and are this project's second
and first best as well (0.616 and 0.858). The ordering is consistent.

## The direction is reachable from Batch 1

Fraction of each class's mean drift direction that lies in the span of Batch 1's
top-k within-class variation axes:

| Gas | k=1 | k=2 | k=5 | k=10 |
|---|---:|---:|---:|---:|
| Ethanol | 0.591 | 0.711 | 0.932 | 0.991 |
| Ammonia | 0.239 | 0.343 | 0.799 | 0.976 |
| Ethylene | 0.000 | 0.036 | 0.592 | 0.969 |
| **Acetaldehyde** | **0.483** | **0.687** | **0.969** | **0.998** |
| Acetone | 0.383 | 0.444 | 0.798 | 0.999 |
| Toluene | 0.452 | 0.533 | 0.864 | 0.999 |

**96.9 % of Acetaldehyde's drift direction lies in the top five axes of Batch 1's
own within-class variation**, and 48.3 % in the first axis alone. The direction is
not orthogonal to anything Batch 1 contains; it is one of Batch 1's own principal
directions, travelled far.

## What this implies

1. **The failure is a property of the dataset under this protocol**, which is why
   two unrelated implementations reproduce it exactly and why the paper's own
   model makes the same error. No architecture, normalisation, class weight or
   stopping rule addresses it.
2. **A domain-generalisation method has a real target here.** The displacement is
   along a direction Batch 1 already contains, so augmenting Batch 1 along its own
   principal directions can in principle reach where target Acetaldehyde sits.
   This is the first concrete job for the paper's augmentation block that does not
   rest on the paper's say-so.
3. **The magnitude is the parameter that matters, and it was never swept far
   enough.** Reaching the target region needs a displacement of order eighteen
   class radii. The previous project tested augmentation noise at 0, 0.05, 0.2 and
   0.5 in the augmentation's own units and measured -0.013 to -0.009 at every
   scale. Those scales cannot produce a displacement of that size, so the
   experiment answered a different question than the one that matters.
4. **The paper does not escape it either.** Its CDCNN reaches 0.7230 with
   Acetaldehyde at zero, so whatever its augmentation achieves, it does not
   recover this class. The gap to the paper is therefore in the other five
   classes, not in the dead one.

## The next experiment this suggests

Sweep augmentation displacement by orders of magnitude rather than by small
factors - measured in units of the class radius, not in the noise parameter - and
check at each scale whether Batch 1's augmented Acetaldehyde cloud covers the
region where target Acetaldehyde actually sits. The coverage check is computable
before any training, from Batch 1 plus the geometry above.

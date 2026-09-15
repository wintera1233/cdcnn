# Review of New CDCNN Experiment Settings

Review date: 2026-09-08  
Reviewed specification: `CDCNN_four_experiment_spec.md`  
Review scope: requirements and implementation-readiness review only; no source code or configuration was changed.

## Executive summary

The updated specification clearly defines a four-stage, source-only ablation: B0, A1, A2, and A3. It also fixes the backbone layout, the location and output shape of the feature-generation operation, the common training parameters for A1-A3, the evaluation protocol, and the required artifacts.

The proposed A1 and A2 methods are materially different from the currently recorded implementation settings:

- A1 is now statistical mixing followed by a Gaussian perturbation added to each source input. The current configuration and implementation instead add independent zero-mean Gaussian noise with standard deviation 0.05.
- A2 is now high/low-frequency decomposition and AdaIN-like low-frequency restyling after ResNet block 3. The current configuration and implementation instead use same-class convex interpolation of two latent maps.

Accordingly, existing A1-A3 results produced by the current implementation would not validate the newly specified experiments. No experiment should be launched against the new specification until the unresolved items in this report are approved and the implementation is later brought into agreement.

## Resolved experiment design

### Ablation sequence

| ID | Required setting | Increment over previous stage |
|---|---|---|
| B0 | Existing ResNet baseline | None |
| A1 | ResNet plus input statistical augmentation | Input augmentation before the backbone |
| A2 | A1 plus latent feature generation | Feature-generation operation between residual blocks 3 and 4 |
| A3 / CDCNN | A2 plus auxiliary objectives | MSE/L2 and contrastive objectives added to classification loss |

B0 should remain the validated reference and should not be retrained unless an exact controlled comparison is required.

### Data and evaluation protocol

- Expected dataset structure must be validated rather than forced: 13,910 measurements, 10 batches, 6 gas labels, and 128 finite extracted features.
- Batch 1 is the only training and model-selection source. Batches 2-10 are final target evaluation data only.
- Existing Batch 1 five-fold assignments must be reused.
- In cross-validation, the scaler is fit only on the training portion of each Batch 1 fold.
- After source-only selection, the scaler and model are fit on all of Batch 1 and frozen before evaluating Batches 2-10.
- Target results must not affect hyperparameter selection, early stopping, or model choice.
- The primary target metric is the unweighted mean of the nine per-batch accuracies. Pooled target accuracy must be reported separately.
- No sample removal, deduplication, clipping, outlier removal, or imputation is authorized. Every exclusion, if any validation issue later requires one, must be explicit and traceable.

### Backbone and common fixed parameters

| Item | Required setting |
|---|---|
| Input layout | `N x 1 x 16 x 8` |
| Residual blocks | 5 |
| Main convolutions | Kernel 3, stride 1, padding 1; two per block |
| Shortcut | Kernel 1 |
| Block channel interpretation | Current assumption-based layout, beginning with 32 channels and producing 128 channels after block 3 |
| Pooling in backbone | None |
| Tensor before classifier | `128 x 16 x 8`, flattened to 16,384 |
| Classifier | FC128, BatchNorm1d(128), then FC6 |
| Activation | ReLU after the first main-path convolution and after each residual addition |
| Residual-block normalization | No unshown BatchNorm layers |
| Input normalization | StandardScaler |
| Optimizer for A1-A3 | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Epochs | 100 |
| Reproducibility | Record all seeds and deterministic settings |

The optimizer choice, StandardScaler, channel interpretation, and flattening geometry are project assumptions rather than exact paper reproductions and must be labeled that way in results.

## New augmentation and latent-feature requirements

### A1: input statistical augmentation

For each Batch 1 training sample `X_j`, the proposed procedure is:

1. Compute the sample's mean and variance.
2. Select another source-training sample `X_k`.
3. Draw `lambda` in `[0, 1]` from a documented distribution.
4. Form mixed mean and variance statistics:
   - `mu_mix = lambda * mu_j + (1 - lambda) * mu_k`
   - `delta_mix = lambda * delta_j + (1 - lambda) * delta_k`
5. Draw a Gaussian perturbation using the mixed statistics, apply the paper's unspecified function `f`, and add the result to `X_j`.
6. Retain the label of `X_j` and record the generated sample's provenance.

Generation is restricted to the active Batch 1 training split. Target samples and target statistics are prohibited. The method must not be described as a VAE.

### A2: latent feature generation

The added feature-generation operation is placed after residual block 3 and before residual block 4. Its input and output must both have shape `[batch, 128, 16, 8]`.

The proposed method is:

1. Approximate a high-frequency component with max pooling followed by nearest-neighbor upsampling.
2. Define the residual low-frequency component as the original latent tensor minus that approximation.
3. Compute channel-wise mean and standard deviation for the low-frequency component.
4. Sample replacement low-frequency mean and standard-deviation statistics from configured source-domain Gaussian statistics.
5. Apply AdaIN-like restyling to the low-frequency component.
6. Recombine the preserved high-frequency component with the restyled low-frequency component.
7. Pass both required branches through the shared residual blocks 4 and 5, as dictated by the experiment's training objective.

Only Batch 1 training features and statistics may be used. The claim that low-frequency statistics encode domain/drift style while high-frequency structure preserves gas identity must be reported as a modeling assumption, not as an established conclusion.

The specification describes this as a feature-generation **block/operation**, but it does not define any new trainable parameters. If “added layer” is intended to mean a learned neural-network layer rather than the stated pooling, upsampling, normalization, and recombination operations, that architecture is not yet specified.

### A3: losses and latent representations

The required total objective is:

`L = L_ce + lambda_MSE * L_MSE + lambda_con * L_con`

Classification loss remains explicit. The exact inputs and reductions for MSE/L2 and contrastive loss, as well as their weights and any temperature or margin, remain unresolved. The specification also does not yet identify whether “latent feature” in the contrastive term means the output after block 3, the generated tensor, the output after block 5, the 16,384-value flattened tensor, or the 128-value FC representation.

## Items requiring approval before implementation

These are not minor documentation details; different choices produce different experiments.

| Area | Missing decision | Why it matters |
|---|---|---|
| A1 amount | Augmentation ratio or exact generated count | Determines training-set size and weighting of generated data |
| A1 statistics | Axes over which each sample's mean and variance are computed | Could yield one scalar, per-sensor values, or per-feature-group values |
| A1 dispersion | Whether `delta` is variance or standard deviation | A Gaussian parameterization needs an unambiguous scale convention |
| A1 sampling | Distribution and parameters for `lambda` | Changes the generated perturbation distribution |
| A1 pairing | Uniform/random, same-class or any-class, replacement policy, and whether self-pairs are allowed | Controls whether class-dependent statistics are mixed |
| A1 transform | Exact definition of `f` in `f(N(mu_mix, delta_mix))` | The generated input cannot be reproduced without it |
| A1 Gaussian shape | Scalar, 128-dimensional independent, or covariance-aware draw | Determines how perturbations vary across features |
| A1 schedule | Fixed generated copies versus regeneration per epoch/mini-batch | Changes sample exposure and reproducibility |
| A2 amount | Latent-generation ratio or exact count | Determines loss weighting and effective sample count |
| A2 pooling | Kernel, stride, padding, and handling of odd dimensions | Defines the decomposition while preserving `16 x 8` after upsampling |
| A2 statistics | Exact reduction axes for channel-wise mean/std | Must state whether statistics are per sample and per channel over both spatial axes |
| A2 stability | Epsilon and standard-deviation convention | Prevents division by zero and affects numerical results |
| A2 source distribution | How Gaussian means/stds are estimated and sampled, including covariance/independence and positivity handling for sampled std | Defines the core restyling distribution and guards invalid negative scales |
| A2 gradient behavior | Whether sampled statistics/high-frequency paths are detached | Changes optimization dynamics |
| A2 branches | Whether original and generated features are both classified in A2, and their CE weighting | “Preserve the original latent feature” does not fully define the training loss |
| A3 MSE/L2 | Compared tensors, distance, reduction, and branch direction | Logits, probabilities, and embeddings yield different objectives |
| A3 contrastive | Positive/negative construction, representation, similarity/distance, reduction, temperature or margin | Required for a complete contrastive objective |
| A3 weights | Numeric `lambda_MSE` and `lambda_con` | Currently unspecified in the updated document |
| Model selection | What settings are candidates and the source-CV selection/tie rule | Prevents post hoc selection and target leakage |
| Training details | Adam variant/defaults, LR damping decision, initialization, shuffle/drop-last, deterministic mode, and checkpoint selection | Needed for reproducibility; “100 epochs” alone does not define which checkpoint is evaluated |

The phrase “specified numbers” is only partly satisfied. The document specifies 5 folds, 5 residual blocks, 128 input features, a 128-unit FC layer, 6 output classes, learning rate 0.001, weight decay 0.0001, batch size 64, and 100 epochs. It does **not** specify numerical augmentation counts/ratios, pooling geometry, epsilon, sampling-distribution parameters, or A3 loss weights.

## Differences from the current project settings

The following comparison is included to prevent accidental execution of the old algorithm under the new experiment names.

| Component | Updated specification | Current recorded setting | Status |
|---|---|---|---|
| A1 augmentation | Mix statistics from `X_j` and `X_k`, sample a Gaussian perturbation, apply `f`, add to `X_j` | One noise copy per sample; independent `N(0, 0.05)` noise in standardized feature space | Incompatible |
| A1 ratio | Not fixed | 1.0 | Current number is not authorized by the new spec unless approved |
| A2 generation | High/low-frequency decomposition plus AdaIN-like restyling | Same-class convex interpolation of post-block-3 maps | Incompatible |
| A2 ratio | Not fixed | 1.0 | Current number is not authorized by the new spec unless approved |
| A2 sampling | Source-domain Gaussian statistics, details open | Partner selected from same class; `alpha ~ Beta(2,2)` | Incompatible |
| A3 MSE | Exact definition open | MSE between original/generated class-probability vectors | Proposed implementation assumption only |
| A3 contrastive | Exact definition and representation open | Paired cosine distance after block-5 flattening | Proposed implementation assumption only; also lacks negatives |
| A3 weights | Not fixed | 0.1 for MSE and 0.1 for contrastive | Current numbers are not authorized by the new spec unless approved |

## Required outputs and audit record

Each approved experiment must write to a new unique `runs/<run_id>/` directory and never overwrite a prior run. At minimum it must contain:

- Resolved configuration and all implementation assumptions.
- Input identities/hashes, software versions, random seeds, deterministic settings, and preprocessing fit scope.
- Model checkpoint, parameter counts, training/validation curves, and Batch 1 fold metrics.
- Per-batch target metrics, unweighted target mean, pooled target accuracy, sample-level predictions, and confusion matrices.
- Generated-sample counts and provenance for A1-A3.
- Explicit exclusions (including an empty list when none occur), a run report, and a complete output inventory.

The final comparison must include B0, A1, A2, and A3; Batch 1 CV; accuracy for every target batch; the unweighted target mean; and pooled target accuracy. Interpret only the successive ablation differences. Measured cross-batch distribution shifts must not automatically be presented as proof that sensor drift is their unique cause.

## Approval recommendation

Status: **not implementation-ready**.

Approve the resolved protocol and backbone as written, but first add or approve a single explicit parameter table covering every unresolved A1, A2, and A3 item above. After that approval, the configuration and implementation can be revised in a separate step. Until then, no A1-A3 code should be generated, rewritten, or executed as representing this updated specification.

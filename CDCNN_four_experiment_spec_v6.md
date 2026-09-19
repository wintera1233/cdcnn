# Four Stage CDCNN Experiment Specification

## Goal

Measure the contribution of CDCNN-inspired components under a strict source-only
sensor-drift protocol. The canonical project model is `A2-semantic`, which uses
physical-semantic frequency names. A separate `A2-paper-literal` diagnostic is
defined only to expose the paper's H/L notation and is not silently mixed with
the canonical model.

All experiments use Batch 1 for training, tuning, and source-only cross-validation.
Batches 2–10 are loaded only after the selected checkpoint is frozen for final
evaluation. No target-batch data or metrics may influence augmentation design,
hyperparameter selection, checkpoint selection, or early stopping.

## Paper and Supporting Information Traceability

Page numbers refer to `Sensors and Actuators: A. Physical 372 (2024) 115314`.
`S1–S5` refer to `1-s2.0-S0924424724003078-mmc1.docx`.

| Configuration item | Paper or appendix evidence | Status in this project |
|---|---|---|
| Five ResNet blocks; 1D convolution; 3 and 1x1 kernels | Main text Fig. 2, pp. 3–5; Appendix Fig. S4(a) | Paper-supported at architecture level |
| Feature generation between ResNet 3 and 4 | Main text Section 2.4, p. 4; Fig. 2; Appendix Fig. S4(b) | Paper-supported at location and operation level |
| H/L branch labels | Main text Eqs. (8)–(9); Appendix Fig. S4(b) | Paper uses nonstandard labels: pooled/upsampled branch is called H and residual is called L |
| Physical-semantic branch names | Signal-processing interpretation of the two branches | Project-controlled adaptation: pooled/upsampled = low-frequency-like; residual = high-frequency-like |
| 100 training epochs | Appendix S2, Fig. S1 | Paper-supported for the shown networks; applying it uniformly to B0–A3 is project protocol |
| SGD and learning-rate damping | Main text Section 5.2, p. 5 | Paper-supported at concept level |
| Momentum 0.9, lr 0.001, StepLR 0.5 every 25 epochs | No exact value shown in the provided paper/appendix | Project-controlled settings |
| Input augmentation and variance mixing | Main text Section 3, Eqs. (5)–(7); Appendix Fig. S5 | Paper-supported at equation level; same-class pairing is a project assumption |
| A2 MSE consistency | Appendix S1 Eq. (S4); S2 | Paper-supported; computed on softmax probabilities |
| Contrastive projection and unit-sphere normalization | Appendix S1 Eq. (S5) | Paper-supported at objective level; head architecture and temperature are project choices |
| Batch 1 source and Batches 2–10 targets | Main text Section 5.2, p. 5; Fig. S7; Appendix S4 | Paper-supported; the strict load-order audit is project protocol |

## A2 Dual-Implementation Definition

Define the pooled-and-upsamped operator:

$$P(z_s)=\operatorname{upsample}(\operatorname{MaxPool1d}(z_s)).$$

The paper-literal diagnostic keeps the printed notation:

$$z_{s,\mathrm{paper}}^H=P(z_s),\qquad
z_{s,\mathrm{paper}}^L=z_s-P(z_s).$$

It applies the paper's restyling operation to the component denoted
$z_{s,\mathrm{paper}}^L$.

The canonical physical-semantic implementation uses:

$$z_{\mathrm{low}}=P(z_s),\qquad
z_{\mathrm{high}}=z_s-P(z_s),$$

and restyles $z_{\mathrm{low}}$. These are different generated-feature
procedures because they restyle different branches; they are not merely different
variable names.

## Experiment Sequence

| ID | Model | Added component | Objective |
|---|---|---|---|
| **B0** | ResNet baseline | Base 1D ResNet | $L_{ce}$ |
| **A1** | ResNet + input augmentation | VAE-inspired statistical input augmentation | $L_{ce}$ |
| **A2-semantic** | A1 + feature generation | Physical-semantic feature generation + MSE consistency | $L_{ce}+\lambda_{MSE}L_{MSE}$ |
| **A3** | A2-semantic + supervised contrastive learning | Shared projection head + supervised contrastive loss | $L_{ce}+\lambda_{MSE}L_{MSE}+\lambda_{con}L_{con}$ |

The main project sequence is B0 → A1 → A2-semantic → A3. The paper-literal
diagnostic is an optional predeclared comparison, not an unreported replacement
of A2-semantic.

## Strict Source-Only Protocol

1. Load and validate `Dataset/batch1.dat` only: six labels, 128 features, finite values, and expected input hash.
2. Execute the existing five-fold assignments on Batch 1.
3. Fit each `StandardScaler` only on the corresponding Batch 1 training fold.
4. Select hyperparameters and the final training checkpoint using Batch 1 CV only.
5. Refit the selected configuration on all Batch 1 data.
6. Freeze the model and scaler.
7. Only then load and evaluate `Dataset/batch2.dat` through `Dataset/batch10.dat`.
8. Report per-batch accuracy, unweighted mean of nine batch accuracies, and pooled target accuracy separately.
9. Save sample-level predictions and confusion matrices.
10. Never remove a target batch because of PCA appearance or target accuracy.

## Unified ResNet and Training Protocol

All four stages share the same backbone and training protocol. The addition of a
loss or augmentation is the only intended change between stages.

```text
N x 1 x 128 -> five residual blocks -> flatten
             -> FC128 -> BatchNorm1d(128) -> FC6
```

- Block output channels: 32, 64, 128, 256, 128.
- Main-path convolution: `Conv1d(kernel_size=3, stride=1, padding=1)`.
- Shortcut convolution: `Conv1d(kernel_size=1)`.
- No pooling inside the ResNet blocks; pooling is used only inside A2 feature generation.
- Feature generation is inserted between Blocks 3 and 4.
- Input normalization uses a source-fold-fitted `StandardScaler`.
- Optimizer: SGD, momentum `0.9`, learning rate `0.001`, weight decay `1e-4`.
- Scheduler: `StepLR(gamma=0.5, step_size=25)`. This schedule is project-controlled and fixed before target evaluation.
- Batch size: `64`.
- Total training: exactly `100` epochs.
- Early stopping: disabled. The selected model for the fixed schedule is the epoch-100 checkpoint unless a source-only checkpoint rule is explicitly recorded before training.

## A1 Input Data Augmentation

This section follows the paper's Section 3 and Eqs. (5)–(7). For each source
training sample $X_j\in\mathbb{R}^{128}$:

$$\mu_j=\operatorname{mean}(X_j),\qquad
v_j=\operatorname{var}(X_j).$$

1. Select a non-self sample $X_k$ from the same gas class. Same-class pairing is a project-controlled assumption; the paper only explicitly requires a different sample.
2. Draw $\lambda\sim\operatorname{Uniform}(0,1)$.
3. Mix the statistics:

$$\mu_{mix}=\lambda\mu_j+(1-\lambda)\mu_k,$$
$$v_{mix}=\lambda v_j+(1-\lambda)v_k.$$

4. Draw one noise vector from a Gaussian whose mathematical second parameter is variance:

$$N\sim\mathcal{N}(\mu_{mix},v_{mix}).$$

In NumPy, the API scale is standard deviation:

```python
noise = rng.normal(loc=mu_mix, scale=np.sqrt(v_mix), size=128)
augmented_sample = X_j + noise
```

The paper does not introduce an additional perturbation multiplier, so the
canonical setting is `perturbation_scale = 1.0`. Any other value is a separate
sensitivity experiment. Generate one fixed view per source training sample per
fold using the recorded seed and preserve the anchor label $Y_j$.

## A2-semantic Physical-Semantic Feature Generation

The intermediate feature after ResNet Block 3 is

$$z_s=R_s(X),\qquad z_s\in\mathbb{R}^{B\times128\times128}.$$

### Frequency-like decomposition

The canonical project names are:

$$z_{\mathrm{low}}=\operatorname{upsample}(\operatorname{MaxPool1d}(z_s)),$$
$$z_{\mathrm{high}}=z_s-z_{\mathrm{low}}.$$

Use `MaxPool1d(kernel_size=2, stride=2)` and nearest-neighbor interpolation back
to length 128. The pooled branch is called low-frequency-like because it is the
coarse/style component; the residual is called high-frequency-like because it
contains local structural detail. These are physical-semantic names, not the
paper's literal H/L labels.

### Channel-wise AdaIN statistics

For each active source-training mini-batch, compute channel-wise statistics across
the length dimension:

$$\mu_{low}=\operatorname{mean}(z_{low},\operatorname{axis}=\mathrm{length}),$$
$$\sigma_{low}=\sqrt{\operatorname{var}(z_{low},\operatorname{axis}=\mathrm{length})+\epsilon}.$$

The resulting shapes are `[B, 128, 1]`, using population variance (`ddof=0`).
This is the A2-semantic AdaIN adaptation; it is not a literal implementation of
the paper's printed Eqs. (10)–(11), which reduce over both channel and length.

### Positivity-guaranteed style sampling

Compute the following statistics over the active source-training mini-batch only:

$$\hat\mu=\operatorname{mean}_B(\mu_{low}),\qquad
\tilde\mu=\operatorname{std}_B(\mu_{low}),$$
$$\hat\sigma=\operatorname{mean}_B(\sigma_{low}),\qquad
\tilde\sigma=\operatorname{std}_B(\sigma_{low}).$$

The target mean is sampled as:

$$\mu'\sim\mathcal{N}(\hat\mu,\tilde\mu^2).$$

To guarantee a positive target standard deviation, use the single canonical
log-normal rule:

$$\log\sigma'\sim\mathcal{N}\left(
\log\max(\hat\sigma,\epsilon),
\left(\frac{\tilde\sigma}{\max(\hat\sigma,\epsilon)}\right)^2\right),$$
$$\sigma'=\exp(\log\sigma').$$

Softplus and clipping are not alternative canonical implementations. All sampled
statistics must be generated from source-training data only.

### Restyling and recombination

$$\tilde z_{low}=\sigma'\left(\frac{z_{low}-\mu_{low}}{\sigma_{low}}\right)+\mu',$$
$$\tilde z_s=z_{high}+\tilde z_{low}.$$

### Dual branch and A2 loss

Pass $z_s$ and $\tilde z_s$ through the shared ResNet Blocks 4–5 and classifier:

$$\hat Y=FC(R_f(z_s)),\qquad
\tilde Y=FC(R_f(\tilde z_s)).$$

Use raw logits for cross-entropy:

$$L_{ce}=\frac12\left[CE(\hat Y,Y)+CE(\tilde Y,Y)\right].$$

Use softmax probabilities for the consistency loss:

$$p=\operatorname{softmax}(\hat Y),\qquad
\tilde p=\operatorname{softmax}(\tilde Y),$$
$$L_{MSE}=\frac1B\sum_{i=1}^{B}\|p_i-\tilde p_i\|_2^2.$$

The canonical project default is $\lambda_{MSE}=0.5$:

$$L_{A2}=L_{ce}+\lambda_{MSE}L_{MSE}.$$

For `A2-paper-literal`, preserve the paper's H/L symbols and restyle the paper's
$z_{s,\mathrm{paper}}^L$ branch instead. Do not combine that branch assignment
with the semantic assignment above.

## A3 CDCNN Supervised Contrastive Learning

A3 extends A2-semantic while keeping the same generated features, CE, and MSE.

Use one shared project-controlled linear projection head with output dimension
128 for both branches:

$$u_i=P(Z_{f,i}),\qquad z_i=\frac{u_i}{\|u_i\|_2},$$

where $Z_f=R_f(z_s)$ and $\tilde Z_f=R_f(\tilde z_s)$. Combine the normalized
original and generated projections into a set of $2B$ samples. The projection
head weights are shared between the two branches.

For anchor $i$, define

$$P(i)=\{p\neq i:Y_p=Y_i\},$$

and let $I$ contain all $2B$ projected samples. Use:

$$L_{con}=\sum_{i\in I}\frac{-1}{|P(i)|}\sum_{p\in P(i)}
\log\frac{\exp(z_i\cdot z_p/\tau)}
{\sum_{a\in I\setminus\{i\}}\exp(z_i\cdot z_a/\tau)}.$$

Use $\tau=0.07$ as a project-controlled default. The projection head dimension,
temperature, and loss weights are not fully specified in the provided appendix
and must not be described as author-provided numerical settings.

The canonical A3 objective is:

$$L_{A3}=L_{ce}+\lambda_{MSE}L_{MSE}+\lambda_{con}L_{con},$$

with project-controlled defaults $\lambda_{MSE}=0.5$ and
$\lambda_{con}=0.5$.

## v6.3 Implementation Status and Deviations

The executable implementation is `src/cdcnn_ablation.py` with
`configs/cdcnn_v6.json`, launched through `scripts/run_cdcnn_v6_full.py`. It
reports implementation version `CDCNN_v6.3_A3_numerical_stabilization` and
deviates from the A3 definition printed above in four recorded ways. Full detail
is in `docs/cdcnn-v6.3-numerical-stabilization.md`.

| This specification | v6.3 implementation | Scope |
|---|---|---|
| `FC128 -> BatchNorm1d(128) -> FC6` for every stage | A3 uses `LayerNorm(128)`, so generated samples cannot enter inference-time normalization state | A3 only |
| "Softplus and clipping are not alternative canonical implementations" | A3 clamps sampled sigma to `[0.001, 10]` and log-sigma dispersion to `[0, 2]` | A3 only |
| No bound on intermediate values | A3 clamps residual, generated, and contrastive vectors to `[-20, 20]` | A3 only |
| $L_{con}$ sums over all $2B$ anchors | the implementation averages over anchors, so the term does not scale with batch size | A3 only |

A3 additionally applies `clip_grad_norm_(max_norm=1.0, error_if_nonfinite=True)`
before every optimizer step. B0, A1, A2-semantic, and A2-paper-literal follow
this specification exactly.

These are project-controlled numerical choices, never tuned on Batches 2-10.
They do mean that the sentence "the addition of a loss or augmentation is the
only intended change between stages" no longer holds between A2-semantic and
A3: A3 differs by the contrastive loss *and* by normalization, bounds, and
clipping. The A3 confound ablation
(`configs/cdcnn_v6_3_a3_confound.json`, `docs/a3-confound-ablation.md`) adds the
diagnostic stages `B0-LN`, `B0-stab`, and `A2-stab` to separate those effects.
`A2-stab` is A3 without the contrastive loss.

### Configuration keys that document rather than control

`validate_config` pins every protocol value, but some keys are descriptive: the
behavior they name is fixed in code. These include `pool_kernel_size`,
`pool_stride`, `upsample_mode`, `statistics_axis`, `population_variance`,
`batch_spread_unbiased`, `style_sampling`, `detach_sampled_style_statistics`,
`partner`, `lambda_distribution`, `views_per_source_sample`, `include_originals`,
`perturbation_scale`, `mse_on`, and `contrastive_reduction`. Editing one of them
is rejected by the validator rather than silently applied, so a sensitivity
experiment such as `perturbation_scale = 0.1` requires a code change and a new
implementation version, not only a configuration edit.

### Environment sensitivity

Runs are deterministic within one software environment, not across environments.
The 2026-09-11 full run used torch 2.5.1+cu121; a 2026-09-15 rerun of the same
seeds under torch 2.8.0+cu126 on the same GPU reproduced four of five B0 seeds
bit-for-bit and differed by one validation sample in the fifth. Stage
comparisons must therefore come from a single environment.

## Final Comparison Table

The paper values are reference benchmarks only. They are not expected to match
the semantic adaptation exactly.

| Model stage | Batch 1 CV Acc | Target mean Acc | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | Paper reference only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0 ResNet** | 0.9779 | 0.4097 | 0.7413 | 0.5090 | 0.4969 | 0.4508 | 0.4763 | 0.3088 | 0.2007 | 0.2119 | 0.2914 | 0.6344 |
| **A1 Input Aug** | 0.9797 | 0.3721 | 0.7315 | 0.4981 | 0.4870 | 0.3299 | 0.4177 | 0.2733 | 0.1259 | 0.1817 | 0.3036 | - |
| **A2-semantic** | 0.9824 | 0.3968 | 0.7666 | 0.5090 | 0.4944 | 0.3594 | 0.4154 | 0.2855 | 0.1714 | 0.2609 | 0.3089 | 0.6705 |
| **A3 CDCNN** | 0.9676 | 0.4600 | 0.8934 | 0.6835 | 0.5764 | 0.3492 | 0.4145 | 0.3299 | 0.1721 | 0.3026 | 0.4188 | **0.7230** |

The 0.6705 and 0.7230 values are the paper's CDWC and CDCNN results,
respectively; they are reference points, not guaranteed outcomes for the
physical-semantic implementation.

The measured columns are the five-seed means of the completed run
`runs/20260911T092326995583Z_cdcnn_v6_3_full`, taken from its `stage_summary.csv`
and `per_batch_summary.csv`. They were produced by the v6.3 implementation
described in the next section, not by the literal A3 definition printed above.

## Required Outputs and Reproducibility

Each run must save:

- Resolved `config.json` and input hashes.
- Model checkpoint and exact code/software versions.
- Five-fold Batch 1 metrics and epoch-level training history.
- Per-target-batch metrics for Batches 2–10, unweighted target mean, and pooled accuracy.
- Per-sample predictions and confusion matrices.
- Random seeds, augmentation provenance, feature-generation geometry, epsilon, and style-sampling statistics.
- A report that identifies paper-supported facts, project-controlled assumptions, and any unresolved paper ambiguities.

Run five fixed seeds: `1042`, `2024`, `3407`, `42`, and `123`. All four stages
must use the same seeds, data split, backbone, optimizer, scheduler, batch size,
and 100-epoch schedule. Target evaluation occurs once after the corresponding
checkpoint is frozen.

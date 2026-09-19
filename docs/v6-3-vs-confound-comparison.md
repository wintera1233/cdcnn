# Updated code versus original v6.3

Comparison of the A3 confound-ablation code (branch `exp/a3-confound-ablation`)
against the original v6.3 implementation (commit `43a4852`), and of the
measured results of the two experiments.

- Original v6.3 run: `runs/20260911T092326995583Z_cdcnn_v6_3_full` (four stages, five seeds, torch 2.5.1+cu121).
- Updated run: `runs/20260915T025701880293Z_cdcnn_v6_3_a3_confound_full` (six stages, five seeds, torch 2.8.0+cu126).

## 1. Why the code changed

v6.3 gave A3 four safeguards that no other stage received: `LayerNorm(128)`
instead of `BatchNorm1d(128)`, hard bounds on sigma and intermediate values,
gradient clipping at max-norm 1.0, and an anchor-mean contrastive reduction.
Those arrived together with the supervised contrastive loss, so the v6.3
A3-minus-A2 difference measured all of them at once. The update adds stages that
apply the safeguards one at a time.

## 2. Code comparison

| Area | Original v6.3 | Updated | Effect on canonical stages |
|---|---|---|---|
| Stage constants | `CANONICAL_STAGES` + `DIAGNOSTIC_STAGES`, five stages total | adds `CONFOUND_STAGES` (`B0-LN`, `B0-stab`, `A2-stab`), `LAYERNORM_STAGES`, `STABILIZED_STAGES` | none; canonical tuples unchanged |
| `validate_config` | one hardcoded implementation version and stage list | `STAGE_SETS` maps each implementation version to its own stage, augmentation and feature lists | none; canonical config validates identically |
| Normalization choice | `LayerNorm` if `stage == "A3"` | `LayerNorm` if `stage in LAYERNORM_STAGES` | none; A3 still the only canonical LayerNorm stage |
| Stabilizer and clipping | `stage == "A3"` | `stage in STABILIZED_STAGES` | none |
| Augmentation gating | `if stage == "B0": no views` | `if stage not in AUGMENTED_STAGES` | none; B0 still the only unaugmented canonical stage |
| Loss selection | `if model.stage in ("B0", "A1")` | `if model.stage not in FEATURE_STAGES` | none |
| Feature branch | `if self.stage in ("A2-semantic", "A3")` | `if self.stage != "A2-paper-literal"` | none |
| Launcher stage list | module-level `STAGES` tuple | `config_stages(cfg)` reads the validated config | none for the canonical config |
| Source-audit keys | `a3_has_no_batchnorm_running_state`, `a3_sigma_within_hard_bounds`, `a3_gradients_clipped_to_max_norm` | `layernorm_stage_has_no_batchnorm_running_state`, `stabilized_sigma_within_hard_bounds`, `stabilized_gradients_clipped_to_max_norm` | renamed keys only; artifacts written before 2026-09-15 keep the old names |
| Configuration | `configs/cdcnn_v6.json` | adds `configs/cdcnn_v6_3_a3_confound.json`, generated from it; only stage lists and a `confound_ablation` block differ | canonical config byte-identical |
| Tests | 23 | 29 (new `tests/test_a3_confound.py`) | existing tests unchanged |
| Smoke suite | five modes | eight modes | same per-mode procedure |

Diff against `43a4852`: 393 insertions, 58 deletions across four files
(`src/cdcnn_ablation.py` +58/-26, `scripts/run_cdcnn_v6_full.py` +56/-32, plus
the new config and test file).

### Stage definitions added

| Stage | Normalization | Bounds + clipping | Feature generation + MSE | Contrastive |
|---|---|---|---|---|
| `B0-LN` | LayerNorm | no | no | no |
| `B0-stab` | LayerNorm | yes | no | no |
| `A2-stab` | LayerNorm | yes | yes | no |

`A2-stab` is exactly A3 minus the contrastive loss.

## 3. Evidence that canonical behavior is unchanged

- **Bit-identical training.** The original and updated modules were trained side
  by side on the same seed for B0, A1, A2-semantic, A2-paper-literal and A3. All
  five produced identical per-epoch losses and identical predictions.
- **Unit tests.** 29/29 pass, including the 23 pre-existing ones.
- **Smoke suite.** All eight modes pass Batch-1-only.
- **Re-run agreement.** B0, A2-semantic and A3 were re-run from scratch in the
  updated code; the differences below are attributable to the torch build, not
  the code change.

| Stage | CV (v6.3) | CV (updated) | Target mean (v6.3) | Target mean (updated) | Difference |
|---|---:|---:|---:|---:|---:|
| B0 | 0.9779 | 0.9784 | 0.4097 | 0.4077 | -0.0020 |
| A2-semantic | 0.9824 | 0.9806 | 0.3968 | 0.3931 | -0.0037 |
| A3 | 0.9676 | 0.9676 | 0.4600 | 0.4602 | +0.0001 |

The largest shift is 0.0037 target mean, smaller than the seed-to-seed standard
deviation of every stage.

## 4. Measured results

All six stages, five seeds, unweighted mean over Batches 2-10:

| Stage | Batch 1 CV | Target mean | SD | Pooled |
|---|---:|---:|---:|---:|
| B0 | 0.9784 | 0.4077 | 0.0186 | 0.3943 |
| B0-LN | 0.9910 | 0.4922 | 0.0415 | 0.5077 |
| B0-stab | 0.9654 | 0.4901 | 0.0304 | 0.4993 |
| A2-semantic | 0.9806 | 0.3931 | 0.0084 | 0.3831 |
| A2-stab | 0.9672 | 0.4573 | 0.0233 | 0.4604 |
| A3 | 0.9676 | 0.4602 | 0.0251 | 0.4607 |

Attribution, paired per seed:

| Step | Isolates | Mean change | SD | Seeds improved |
|---|---|---:|---:|---:|
| B0 -> B0-LN | LayerNorm | +0.0845 | 0.0339 | 5/5 |
| B0-LN -> B0-stab | bounds + clipping | -0.0022 | 0.0377 | 1/5 |
| B0-stab -> A2-stab | augmentation + feature generation + MSE | -0.0328 | 0.0169 | 0/5 |
| A2-stab -> A3 | contrastive loss | +0.0029 | 0.0107 | 4/5 |

## 5. Performance improvement

| Comparison | Target mean | Pooled |
|---|---:|---:|
| Best v6.3 stage (A3) | 0.4600 | 0.4607 |
| Best updated stage (B0-LN) | 0.4922 | 0.5077 |
| **Improvement** | **+0.0322** | **+0.0470** |

Within the updated run, `B0-LN` beats `B0` by +0.0845 target mean on 5/5 seeds.

Two points matter for how this is reported:

1. **The improvement is a normalization effect, not a CDCNN effect.** The
   supervised contrastive loss contributes +0.0029; bounds and clipping are
   accuracy-neutral; augmentation plus feature generation costs
   -0.0328. The v6.3 A3 advantage over A2-semantic was the
   LayerNorm swap bundled with the loss.
2. **The winning stage is selectable without target data.** `B0-LN` also has the
   best Batch 1 CV of any stage (0.9910 against 0.9784 for B0), so the
   source-only protocol would choose it on CV alone. No target metric was used
   for any selection.

The stage ranking is B0-LN ~ B0-stab > A3 ~ A2-stab > B0 > A2-semantic, which
does not reproduce the paper's reported ordering of B0 < CDWC < CDCNN.

## 6. Caveats

- The two runs use different torch builds (2.5.1+cu121 and 2.8.0+cu126). Runs are
  deterministic within one environment only; compare stages within a single run.
- The updated run's controller was killed at 23/30 checkpoints when its session
  ended. The remaining tasks were retrained from code verified unchanged by
  hash, and the interrupted attempt is preserved and listed in
  `source_failure_records.json`.
- `CDCNN_four_experiment_spec_v6.md` was edited between task 28 and the
  evaluation phase, so the run records two specification hashes. The
  specification is documentation and is not executed.
- Neither run tunes anything. Every setting was fixed before training and no
  target metric influenced augmentation, hyperparameters, or checkpoints.

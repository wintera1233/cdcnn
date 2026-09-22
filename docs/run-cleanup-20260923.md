# Run cleanup 2026-09-23

## Rationale

Branch `exp/v7-redesign` restarts the study from the baseline. The paper's plain
ResNet baseline reaches target mean 0.6346 while this project's best stage of any
kind reached 0.5613, so the deficit sits in the baseline rather than in any CDCNN
component. Every run under `runs/` was produced by the v6 stage pipeline
(`src/cdcnn_ablation.py`), which is removed on this branch; none of those runs is
a valid reference for the redesigned protocol.

Per the quarantine protocol, no run directory was deleted. All were moved to
`.trash/run-cleanup-20260923/`.

## Quarantined

40 of the 51 run directories were moved on 2026-09-23. They cover:

- the v6.3 canonical and A3 confound ablation full runs,
- the v6.4 to v6.11 experiment runs (input normalization, normalized ladder,
  paper-literal ladder, augmentation scale, duplication control, epoch
  alignment, contrastive placement, class balance pilot),
- all associated Batch 1 smokes, GPU smokes, plot runs, and the GAS4 prediction
  run.

## Not yet moved

Eleven directories written inside the pinned Docker container are owned by
`root` and could not be moved without elevated privileges:

```
20260921T145711584311Z_v6_3_b0_batch1_gpu_smoke
20260921T145836801855Z_v6_3_b0_batch1_gpu_smoke
20260921T145852844886Z_cdcnn_v6_9_epoch_aligned_full
20260921T155116278470Z_cdcnn_v6_3_batch1_smoke
20260921T155146208838Z_v6_3_b0_batch1_gpu_smoke
20260921T155203687340Z_cdcnn_v6_10_contrastive_placement_full
20260921T164137322383Z_cdcnn_v6_3_batch1_smoke
20260921T164207489435Z_v6_3_b0_batch1_gpu_smoke
20260921T164306477921Z_v6_3_b0_batch1_gpu_smoke
20260921T164419128475Z_v6_3_b0_batch1_gpu_smoke
20260921T164436164787Z_cdcnn_v6_11_class_balance_one_seed_pilot
```

They are moved with:

```bash
sudo mv runs/* .trash/run-cleanup-20260923/
```

## Recovery

`runs/` is gitignored, so the quarantine directory is the only copy of these
results. `.trash/` is also gitignored and is never committed. The code, docs, and
reports that produced and described them remain on branch
`exp/a3-confound-ablation` at commit `95115a2` and are restored with:

```bash
git checkout exp/a3-confound-ablation -- src docs reports
```

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


## Recovery

`runs/` is gitignored, so the quarantine directory is the only copy of these
results. `.trash/` is also gitignored and is never committed. The code, docs, and
reports that produced and described them remain on branch
`exp/a3-confound-ablation` at commit `95115a2` and are restored with:

```bash
git checkout exp/a3-confound-ablation -- src docs reports
```

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


## The quarantine is now empty

Checked 2026-09-23 03:25: `.trash/` contains nothing. The forty directories moved
there, and the failed baseline smoke moved there later, are gone. This session
did not remove them; it only moved directories in. Whatever removed them, the
consequence stands:

**The v6 run outputs no longer exist anywhere.** `runs/` was gitignored, so the
quarantine was their only copy. What survives of that work is the code, the
documents and the review deck on `exp/a3-confound-ablation` at `95115a2`, and the
numbers quoted in this branch's documents. The runs themselves cannot be
re-inspected, only re-trained.

The eleven root-owned directories listed above were among them and are also gone,
so the `sudo mv` recorded there is no longer needed.

## Recovery

`runs/` is gitignored, so the quarantine directory is the only copy of these
results. `.trash/` is also gitignored and is never committed. The code, docs, and
reports that produced and described them remain on branch
`exp/a3-confound-ablation` at commit `95115a2` and are restored with:

```bash
git checkout exp/a3-confound-ablation -- src docs reports
```
